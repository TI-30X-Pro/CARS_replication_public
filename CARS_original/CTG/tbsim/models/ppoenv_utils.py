"""PPO environment utilities: helper classes, constants, and reward shaping."""

import numpy as np
import torch
from typing import Any, NamedTuple, Optional, Tuple
import tbsim.utils.tensor_utils as TensorUtils
from tbsim.utils.geometry_utils import transform_points_tensor, transform_points_np, VEH_VEH_collision
from tianshou.data import to_numpy
# from tbsim.utils.safety_critical_fig_vis import plot_test

# ============================================================================
# Constants
# ============================================================================

FILTERED_KEYS = [
    "maps",
    "history_positions", "history_speeds", "history_yaws", "extent", "history_availabilities",
    "ego_fut_positions", "ego_fut_speeds", "ego_fut_yaws", "ego_fut_availabilities",
    "all_other_agents_history_positions", "all_other_agents_history_speeds",
    "all_other_agents_history_yaws", "all_other_agents_history_extents",
    "all_other_agents_history_availability",
    "extras", "curr_speed", "curr_speed_signed", "extent",
    "raster_from_agent", "world_from_agent",
    "target_positions", "target_yaws", "target_availabilities",
    "neigh_indices", "map_is_singapore"
]

NEIGH_KEYS = [
    "all_other_agents_history_positions", "all_other_agents_history_speeds",
    "all_other_agents_history_yaws", "all_other_agents_history_extents",
    "all_other_agents_history_availability",
    "neigh_indices"
]

FILTERED_KEYS_SET = frozenset(FILTERED_KEYS)


# ============================================================================
# Data Structures
# ============================================================================

class BatchPair(NamedTuple):

    adv: dict
    
    @classmethod
    def from_obs(cls, obs: dict,  adv_idx: int,
                 keys: frozenset = FILTERED_KEYS_SET) -> "BatchPair":  # `keys` is the whitelist; everything else is dropped.

        agents = obs['agents']
        return cls(adv={k: take_by_idx(agents[k], adv_idx) for k in keys & agents.keys()})


# ============================================================================
# Utility Functions
# ============================================================================

def take_by_idx(x: Any, idx: int) -> Any:
    """Extract a single sample from batch data by index."""
    if isinstance(x, dict):
        return {k: take_by_idx(v, idx) for k, v in x.items()}
    if isinstance(x, (np.ndarray, list)):
        result = x[idx]
        return np.asarray(result) if isinstance(result, np.generic) else result
    return x


def pad_trunc_neighbors(batch: dict, target_idx: int, max_n: int = 20) -> dict:
    neigh_idx = batch["neigh_indices"]
    N = neigh_idx.shape[0]
    matches = np.where(neigh_idx == target_idx)[0]
    if len(matches) == 0:
        return None
    pos = int(matches[0])

    half = max_n // 2
    start = pos - half
    start = max(start, 0)
    start = min(start, max(N - max_n, 0))
    end = min(start + max_n, N)

    sel = np.arange(start, end)

    for k in NEIGH_KEYS:
        x = batch[k][sel]
        if x.shape[0] < max_n:
            pad_val = -1 if k == "neigh_indices" else 0
            pad = np.full((max_n - x.shape[0], *x.shape[1:]), pad_val, dtype=x.dtype)
            x = np.concatenate([x, pad], axis=0)
        batch[k] = x

    return batch




# ============================================================================
# Reward Functions
# ============================================================================
import numpy as np





# ============================================================================
# PPO Ego/Others Policy Wrapper
# ============================================================================



class PPOTargetOthersPolicy:
    """Produce target + others trajectories during PPO training (adv is sampled
    from the PPO policy elsewhere; this class never sees the adversary).
    """

    def __init__(self, target_policy, others_policy, n_steps_per_call: int = 5):
        self.target_policy = target_policy
        self.others_policy = others_policy
        self.n_steps_per_call = n_steps_per_call
        self.device = target_policy.device

    def get_actions(
        self,
        all_agents_obs: dict,
        target_idx: int,
        adv_idx: int,
        step_index: int = 0,
    ):
        with torch.no_grad():
            # target: [1, ...]
            target_obs = self._take_idx(all_agents_obs, [target_idx])
            target_action, _ = self.target_policy.get_action(target_obs, step_index=step_index)
            target_pos = to_numpy(target_action.positions)  # (1, T, 2)
            target_yaw = to_numpy(target_action.yaws)       # (1, T, 1)

            # others: exclude adv + target
            others_obs, others_indices = self._take_others(all_agents_obs, adv_idx, target_idx)
            if others_obs is None:
                return target_pos, target_yaw, None, None, None

            others_action, _ = self.others_policy.get_action(others_obs, step_index=step_index)
            others_pos = to_numpy(others_action.positions)  # (K, T, 2)
            others_yaw = to_numpy(others_action.yaws)       # (K, T, 1)
            return target_pos, target_yaw, others_pos, others_yaw, others_indices

    def _take_idx(self, obs: dict, idx_list):
        # Pull device from any tensor in obs.
        sample = obs.get("history_positions", None)
        device = sample.device if isinstance(sample, torch.Tensor) else self.device
        idx_t = torch.as_tensor(idx_list, device=device, dtype=torch.long)

        return TensorUtils.recursive_dict_list_tuple_apply(
            obs,
            {torch.Tensor: lambda x: x.index_select(0, idx_t)},
            ignore_if_unspecified=True,
        )

    def _take_others(self, obs: dict, adv_idx: int, target_idx: int):
        sample = obs["history_positions"]
        M = sample.size(0)
        device = sample.device

        all_idx = torch.arange(M, device=device)
        others_idx = all_idx[(all_idx != adv_idx) & (all_idx != target_idx)]
        if others_idx.numel() == 0:
            return None, None

        others_obs = TensorUtils.recursive_dict_list_tuple_apply(
            obs,
            {torch.Tensor: lambda x: x.index_select(0, others_idx)},
            ignore_if_unspecified=True,
        )
        return others_obs, others_idx.tolist()




def get_adv_target_world_pose(adv_batch, j, t_idx):
    """adv_batch: agent-centric (ego=adv); j: target neighbor index; t_idx: timestep index.
    Returns (p_adv, p_target) as np arrays, each (3,) for (x, y, yaw) in world frame.

    Coordinate and yaw conventions:
      - world_from_agent (w2a): adv-current-frame -> world frame.
        R[:, 0] = (cos(yaw), sin(yaw)) gives the world-frame heading vector.
      - adv_yaw_world = atan2(R[1, 0], R[0, 0]) is adv's current heading.
      - history_positions and all_other_agents_history_positions are expressed
        in the adv-current frame, so we transform them via w2a.
      - Final yaw = adv_yaw_world + yaw_rel (rel = stored relative heading).
    """
    w2a = np.asarray(adv_batch["world_from_agent"], dtype=np.float32)
    adv_yaw_world = np.arctan2(w2a[1, 0], w2a[0, 0])

    adv_pos_agent = np.asarray(adv_batch["history_positions"][t_idx], dtype=np.float32).reshape(1, 2)
    adv_pos_world = transform_points_np(adv_pos_agent, w2a)[0]
    adv_yaw_rel = float(np.squeeze(adv_batch["history_yaws"][t_idx]))
    yaw_adv = adv_yaw_world + adv_yaw_rel

    target_pos_agent = np.asarray(adv_batch["all_other_agents_history_positions"][j, t_idx], dtype=np.float32).reshape(1, 2)
    target_pos_world = transform_points_np(target_pos_agent, w2a)[0]
    target_yaw_rel = float(np.squeeze(adv_batch["all_other_agents_history_yaws"][j, t_idx]))
    yaw_target = adv_yaw_world + target_yaw_rel

    p_adv = np.array([adv_pos_world[0], adv_pos_world[1], yaw_adv], dtype=np.float32)
    p_target = np.array([target_pos_world[0], target_pos_world[1], yaw_target], dtype=np.float32)
    return p_adv, p_target


class FastApproachRewardV2:
    """Simplified variant of FastApproachReward.

    Same structure, but the lateral distance is computed in world coordinates
    so adv yaw changes don't distort the agent-centric p[1]. Keeps the
    `VEH_VEH_collision`-based hit reward and the time-penalty term.
    """
    def __init__(
        self,
        dt=0.1,
        hit_bonus=5.0,
        time_penalty=0.1,
        l0=1.5,
        w_progress=0.4,
        w_lane=0.3,
        w_lane_closing=0.3,
    ):
        self.dt = float(dt)
        self.hit_bonus = float(hit_bonus)
        self.time_penalty = float(time_penalty)
        self.l0 = float(l0)
        self.w_progress = float(w_progress)
        self.w_lane = float(w_lane)
        self.w_lane_closing = float(w_lane_closing)

    def _get_segment(self, adv_batch, target_idx, n_steps):
        """Returns (j, seg, d0, d1, dt_eff, t_idx_0, t_idx_1) or None.

        t_idx_0, t_idx_1 are the timestep indices fed to
        get_adv_target_world_pose so the world-frame lateral distance can be
        evaluated at both endpoints.
        """
        j_all = np.where(adv_batch["neigh_indices"] == target_idx)[0]
        if len(j_all) == 0:
            return None
        j = int(j_all[0])
        rel_pos_raw = adv_batch["all_other_agents_history_positions"][j]
        mask = adv_batch.get("all_other_agents_history_availability", None)
        if mask is not None:
            m = mask[j].astype(bool)
            valid_idx = np.where(m)[0]
            k = int(max(n_steps, 1)) + 1
            if len(valid_idx) < k:
                return None
            t_idx_0 = int(valid_idx[-k])
            t_idx_1 = int(valid_idx[-1])
            rel_pos = rel_pos_raw[m]
        else:
            k = int(max(n_steps, 1)) + 1
            if rel_pos_raw.shape[0] < k:
                return None
            t_idx_0, t_idx_1 = -k, -1
            rel_pos = rel_pos_raw
        seg = rel_pos[-k:]
        d0 = float(np.linalg.norm(seg[0]))
        d1 = float(np.linalg.norm(seg[-1]))
        dt_eff = self.dt * float(k - 1)
        return (j, seg, d0, d1, dt_eff, t_idx_0, t_idx_1)

    def _lateral_dist_world(self, p_adv, p_target):
        """World-frame perpendicular distance from adv to target's centerline
        (independent of adv yaw).
        """
        v = np.asarray(p_adv[:2], dtype=np.float32) - np.asarray(p_target[:2], dtype=np.float32)
        yaw_t = float(p_target[2])
        lat_dir = np.array([-np.sin(yaw_t), np.cos(yaw_t)], dtype=np.float32)
        return float(np.abs(np.dot(v, lat_dir)))

    def _longitudinal_dist_world(self, p_adv, p_target):
        """World-frame longitudinal distance from adv to target along the
        target's heading direction (absolute value).
        """
        v = np.asarray(p_adv[:2], dtype=np.float32) - np.asarray(p_target[:2], dtype=np.float32)
        yaw_t = float(p_target[2])
        lon_dir = np.array([np.cos(yaw_t), np.sin(yaw_t)], dtype=np.float32)
        return float(np.abs(np.dot(v, lon_dir)))

    def __call__(self, adv_batch, target_idx, target_extent, adv_extent, n_steps=1):
        seg_data = self._get_segment(adv_batch, target_idx, n_steps)
        if seg_data is None:
            return 0.0, {}

        j, seg, _d0, _d1, dt_eff, t_idx_0, t_idx_1 = seg_data

        # World-frame lateral/longitudinal distances avoid the agent-centric
        # distortion that would otherwise appear when adv yaws.
        p_adv_0, p_t_0 = get_adv_target_world_pose(adv_batch, j, t_idx_0)
        p_adv_1, p_t_1 = get_adv_target_world_pose(adv_batch, j, t_idx_1)
        lat_0 = self._lateral_dist_world(p_adv_0, p_t_0)
        lat_1 = self._lateral_dist_world(p_adv_1, p_t_1)
        lon_0 = self._longitudinal_dist_world(p_adv_0, p_t_0)
        lon_1 = self._longitudinal_dist_world(p_adv_1, p_t_1)

        # Longitudinal progress along target heading; negative values penalise
        # moving away.
        prog = (lon_0 - lon_1) / (lon_0 + 1e-6)
        r_progress = float(np.clip(prog, -0.5, 1.0))

        # Lateral closing in world frame; negative values penalise moving
        # sideways out (evaluated at the current position lat_1).
        delta_lat = lat_0 - lat_1
        lat_exp = float(np.exp(-lat_1 / (self.l0 + 1e-6)))
        r_lane = lat_exp if delta_lat > 0 else -lat_exp * 0.5
        l_closing = delta_lat / (dt_eff + 1e-6)
        r_lane_closing = float(np.clip(l_closing / (lat_0 + 1e-6), -0.5, 1.0))

        # Hit reward uses VEH_VEH_collision (more accurate than half-diagonal).
        S1 = np.array([[float(adv_extent[0]), float(adv_extent[1])]], dtype=np.float32)
        S2 = np.array([[float(target_extent[0]), float(target_extent[1])]], dtype=np.float32)
        min_dis = VEH_VEH_collision(p_adv_1[np.newaxis], p_t_1[np.newaxis], S1, S2)
        hit = 1.0 if float(min_dis[0]) < 0 else 0.0
        r_hit = self.hit_bonus if hit > 0.5 else 0.0

        total = (
            self.w_progress * r_progress
            + self.w_lane * r_lane
            + self.w_lane_closing * r_lane_closing
            + r_hit
            - self.time_penalty
        )

        metrics = {
            "d_end": lon_1,
            "abs_l": lat_1,
            "collision": float(hit),
            "lateral_dist_world": lat_1,
        }
        return float(total), metrics