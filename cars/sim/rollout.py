"""Run closed-loop episodes and write their HDF5 logs.

Loop (tbsim's safety_critical_guided_rollout_with_custom_target):
  every `n_step` steps, the policy sees the observation of all agents and returns a plan
  per agent; the next `n_step` steps follow it. After each re-plan block, the target and
  the adversary used for that block are checked for a collision (rollout box test, the
  adversary's box enlarged). On a collision the current frame is logged once more and the
  episode stops. Collisions inside a block are only seen if they last to its end.

A policy is any callable `policy(obs, step) -> dict` with
  xy:  (N, T, 2) float32 planned positions in each agent's frame
  yaw: (N, T, 1) float32 planned yaw changes
  adv_idx: index of the adversary for this block, or None
  info: optional dict of per-agent arrays to log (e.g. action_samples, adv_idx)

Episodes go to HDF5 as `/<scene>_<start:04d>_t_<target index>/<field>`, with arrays shaped
`(agents, steps, ...)`. `cars.run.evaluate` reads this layout directly.
"""
from collections import defaultdict

import h5py
import numpy as np
import torch

from cars.sim.geometry import affine_f32, veh_veh_collision

LOG_OBS = ("centroid", "yaw", "extent", "world_from_agent", "raster_from_world")


def _np(v):
    return v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v)


def world_pose(obs, i):
    """Return agent `i` world x, y, and yaw in float32."""
    wfa = _np(obs["world_from_agent"]).astype(np.float32)
    xy = affine_f32(_np(obs["history_positions"])[i, -1], wfa[i])
    yaw = np.arctan2(wfa[i, 1, 0], wfa[i, 0, 0])
    return np.array([xy[0], xy[1], yaw], dtype=np.float32)


def rollout_collision(obs, target_idx, adv_idx):
    if adv_idx is None or target_idx is None:
        return False
    ext = _np(obs["extent"])
    p_t, p_a = world_pose(obs, target_idx), world_pose(obs, adv_idx)
    s_t = np.array([[float(ext[target_idx, 0]), float(ext[target_idx, 1])]], dtype=np.float32)
    s_a = np.array([[float(ext[adv_idx, 0]), float(ext[adv_idx, 1])]], dtype=np.float32)
    return float(veh_veh_collision(p_a[None], p_t[None], s_a, s_t)[0]) < 0


class EpisodeLog:
    """Per-step records of one scene, serialised like tbsim's RolloutLogger."""

    def __init__(self, scene_index=-1):
        self.scene_index = scene_index
        self.steps = []

    def log(self, obs, plan_xy, plan_yaw, info=None):
        n = len(obs["agent_name"])
        rec = {k: _np(obs[k]) for k in LOG_OBS}
        rec["agent_name"] = np.array(obs["agent_name"], dtype=object)
        rec["scene_ids"] = np.array(obs["scene_ids"], dtype=object)
        rec["scene_index"] = np.full(n, self.scene_index, dtype=np.int64)
        rec["track_id"] = np.arange(n, dtype=np.int64)
        rec["action_positions"] = plan_xy[:, [0]]
        rec["action_yaws"] = plan_yaw[:, [0]]
        rec["action_traj_positions"] = plan_xy
        rec["action_traj_yaws"] = plan_yaw
        for k, v in (info or {}).items():
            if k == "action_samples":
                rec["action_sample_positions"] = _np(v["positions"])
                rec["action_sample_yaws"] = _np(v["yaws"])
            else:
                rec[k] = _np(v)
        self.steps.append(rec)

    def serialize(self):
        """{field: (agents, steps, ...)}; shorter per-step arrays are padded with NaN (or 0)."""
        out = {}
        keys = []
        for rec in self.steps:
            keys += [k for k in rec if k not in keys]
        for k in keys:
            per_step = [rec[k] if k in rec else None for rec in self.steps]
            ref = next(v for v in per_step if v is not None)
            per_step = [v if v is not None else np.full_like(ref, np.nan) for v in per_step]
            shapes = {v.shape for v in per_step}
            if len(shapes) > 1:
                pad = np.nan if np.issubdtype(ref.dtype, np.floating) else 0
                width = max(v.shape[1] for v in per_step)
                per_step = [np.concatenate([v, np.full((v.shape[0], width - v.shape[1], *v.shape[2:]), pad,
                                                        dtype=v.dtype)], axis=1) if v.shape[1] < width else v
                            for v in per_step]
            out[k] = np.stack(per_step, axis=1)
        return out


def run_episode(scene, policy, target_name=None, n_step=5, horizon=120, log=True, scene_index=-1):
    """Simulate one scene. Returns dict(buffer, had_collision, steps, target_idx)."""
    target_idx = scene.agents.index(target_name) if target_name in scene.agents else None
    episode = EpisodeLog(scene_index)
    obs = scene.observe(get_map=True)
    counter, frame, had_collision = 0, 0, False
    while True:
        out = policy(obs, counter)
        plan_xy = np.asarray(out["xy"], dtype=np.float32)
        plan_yaw = np.asarray(out["yaw"], dtype=np.float32)
        adv_idx, info = out.get("adv_idx"), out.get("info", {})

        done = False
        for k in range(n_step):
            if k >= plan_xy.shape[1]:
                done = True
                break
            if log:
                episode.log(scene.observe(get_map=False), plan_xy[:, k:], plan_yaw[:, k:], info)
            xy, yaw = scene.plan_to_world(obs, plan_xy, plan_yaw, k)
            scene.place(xy, yaw)
        if frame + n_step >= horizon:
            done = True
        else:
            frame += n_step
        counter += n_step

        obs = scene.observe(get_map=True)
        if rollout_collision(obs, target_idx, adv_idx):
            had_collision = True
            if log:
                n = len(scene.agents)
                extra = {"adv_idx": np.full(n, adv_idx, dtype=np.int32)} if adv_idx is not None else {}
                episode.log(scene.observe(get_map=False), np.zeros((n, 1, 2), np.float32),
                            np.zeros((n, 1, 1), np.float32), extra)
            break
        if done or counter >= horizon:
            break

    buffer = episode.serialize() if log else None
    if buffer is not None and target_name is not None:
        buffer["target_name"] = np.array([target_name], dtype=object)
    return dict(buffer=buffer, had_collision=had_collision, steps=counter, target_idx=target_idx)


def write_episode(h5_path, scene_name, start, target_idx, buffer, pair_idx=0):
    """Append one episode to an HDF5 file and return its group key."""
    suffix = f"_t_{target_idx}" if target_idx is not None else f"_{pair_idx}"
    key = f"{scene_name}_{start:04d}{suffix}"
    utf8 = h5py.string_dtype(encoding="utf-8")
    with h5py.File(h5_path, "a") as f:
        if key in f:
            del f[key]
        for k, v in buffer.items():
            if v.dtype == object or v.dtype.kind in "US":
                f.create_dataset(f"{key}/{k}", data=np.asarray(v.astype(str), dtype=object), dtype=utf8)
            else:
                f.create_dataset(f"{key}/{k}", data=v)
    return key


def trajdata_scene_index(scene_names, locations, splits, wanted=("train", "train_val")):
    """Scene positions in trajdata's index for nusc_trainval-train + -train_val.

    scene_names / locations: scenes in scene.json order; splits: name -> split.
    trajdata enumerates the splits, and within each split boston then singapore, each in
    scene.json order.
    """
    out, k = {}, 0
    for split in wanted:
        for city in ("boston", "singapore"):
            for name, loc in zip(scene_names, locations):
                if loc.startswith(city) and splits.get(name) == split:
                    out[name] = k
                    k += 1
    return out
