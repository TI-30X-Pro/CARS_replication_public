"""Compute the adversary's PPO reward after each five-step block (paper Eq. 8).

From the last 6 available history points of the target (as the adversary's neighbour), take
the first (t0) and the last (t1); both agents' world poses at those two steps give the
target-frame offsets lon (along the target's heading) and lat (across it):
  progress      clip((|lon0| - |lon1|) / |lon0|, -0.5, 1)
  lane          exp(-|lat1| / 1.5) if |lat| shrank, else -0.5 exp(-|lat1| / 1.5)
  lane closing  clip((|lat0| - |lat1|) / 0.5 s / |lat0|, -0.5, 1)
  hit           5 if the boxes overlap at t1 (rollout box test)
  reward        0.4 progress + 0.3 lane + 0.3 lane closing + hit - 0.1
No reward (0) if the target is not among the adversary's neighbours or has fewer than 6
history points.

`adversary_view` cuts the adversary's neighbours to a window of 20 around the target, as the
original's pad_trunc_neighbors does; it returns None when the target is not a neighbour any
more (the episode then ends).
"""
import numpy as np
import torch

from cars.selection.libm import atan2f
from cars.sim.geometry import affine_f32, veh_veh_collision

NEIGH_KEYS = ("all_other_agents_history_positions", "all_other_agents_history_speeds",
              "all_other_agents_history_yaws", "all_other_agents_history_extents",
              "all_other_agents_history_availability", "neigh_indices")


def _np(v):
    return v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v)


def adversary_view(row, target, max_n=20):
    """row: one agent's observation (numpy, no batch dim). Window of max_n neighbours."""
    neigh = _np(row["neigh_indices"])
    hits = np.where(neigh == target)[0]
    if len(hits) == 0:
        return None
    n = len(neigh)
    start = min(max(int(hits[0]) - max_n // 2, 0), max(n - max_n, 0))
    sel = np.arange(start, min(start + max_n, n))
    out = dict(row)
    for k in NEIGH_KEYS:
        x = _np(row[k])[sel]
        if x.shape[0] < max_n:
            pad = np.full((max_n - x.shape[0], *x.shape[1:]), -1 if k == "neigh_indices" else 0, dtype=x.dtype)
            x = np.concatenate([x, pad], axis=0)
        out[k] = x
    out["target_idx"] = target
    return out


def world_poses(view, j, t):
    """World x, y, yaw (float32) of the adversary and of its neighbour j at history step t.
    The yaw sums are float64 before rounding (NumPy 1.21: float32 scalar + float)."""
    w = _np(view["world_from_agent"]).astype(np.float32)
    heading = float(atan2f(w[1, 0], w[0, 0]))
    adv_xy = affine_f32(_np(view["history_positions"])[t], w)
    tgt_xy = affine_f32(_np(view["all_other_agents_history_positions"])[j, t], w)
    yaw_a = heading + float(np.squeeze(_np(view["history_yaws"])[t]))
    yaw_t = heading + float(np.squeeze(_np(view["all_other_agents_history_yaws"])[j, t]))
    return (np.array([adv_xy[0], adv_xy[1], yaw_a], dtype=np.float32),
            np.array([tgt_xy[0], tgt_xy[1], yaw_t], dtype=np.float32))


def _target_frame(p_adv, p_tgt):
    v = p_adv[:2].astype(np.float32) - p_tgt[:2].astype(np.float32)
    yaw = float(p_tgt[2])
    lon = float(np.abs(np.dot(v, np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float32))))
    lat = float(np.abs(np.dot(v, np.array([-np.sin(yaw), np.cos(yaw)], dtype=np.float32))))
    return lon, lat


class ApproachReward:
    def __init__(self, dt=0.1, hit_bonus=5.0, time_penalty=0.1, l0=1.5, w_progress=0.4, w_lane=0.3,
                 w_lane_closing=0.3):
        self.dt, self.hit_bonus, self.time_penalty, self.l0 = dt, hit_bonus, time_penalty, l0
        self.w_progress, self.w_lane, self.w_lane_closing = w_progress, w_lane, w_lane_closing

    def __call__(self, view, target, target_extent, adv_extent, n_steps=5):
        """(reward, metrics); metrics has collision (0/1) and the final lon / lat offsets."""
        hits = np.where(_np(view["neigh_indices"]) == target)[0]
        if len(hits) == 0:
            return 0.0, {}
        j = int(hits[0])
        k = int(max(n_steps, 1)) + 1
        valid = np.where(_np(view["all_other_agents_history_availability"])[j].astype(bool))[0]
        if len(valid) < k:
            return 0.0, {}
        t0, t1 = int(valid[-k]), int(valid[-1])
        dt_eff = self.dt * float(k - 1)

        p_a0, p_t0 = world_poses(view, j, t0)
        p_a1, p_t1 = world_poses(view, j, t1)
        lon0, lat0 = _target_frame(p_a0, p_t0)
        lon1, lat1 = _target_frame(p_a1, p_t1)

        r_progress = float(np.clip((lon0 - lon1) / (lon0 + 1e-6), -0.5, 1.0))
        d_lat = lat0 - lat1
        lat_exp = float(np.exp(-lat1 / (self.l0 + 1e-6)))
        r_lane = lat_exp if d_lat > 0 else -lat_exp * 0.5
        r_closing = float(np.clip(d_lat / (dt_eff + 1e-6) / (lat0 + 1e-6), -0.5, 1.0))

        s_a = np.array([[float(adv_extent[0]), float(adv_extent[1])]], dtype=np.float32)
        s_t = np.array([[float(target_extent[0]), float(target_extent[1])]], dtype=np.float32)
        hit = float(veh_veh_collision(p_a1[None], p_t1[None], s_a, s_t)[0]) < 0
        total = (self.w_progress * r_progress + self.w_lane * r_lane + self.w_lane_closing * r_closing
                 + (self.hit_bonus if hit else 0.0) - self.time_penalty)
        return float(total), {"d_end": lon1, "abs_l": lat1, "collision": float(hit), "lateral_dist_world": lat1}
