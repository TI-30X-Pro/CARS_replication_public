"""Reactive (closed-loop) Bezier adversary rollout wrapper.

Every call to `get_action` (= every `n_step_action` simulation steps) re-fits
a fresh cubic-Bezier adversary trajectory based on the current observation:

  - p0 = adv's *current* world position (read from obs)
  - p1 = velocity-matched takeoff tangent (preserves C^1 continuity across refits)
  - p2 = target's predicted world position at step 2T/3
  - p3 = target's predicted world position at step T

The target-future predictions come from the same `target_policy.get_action(...)`
call that drives the target itself, so no extra forward pass is needed.
"""
from __future__ import annotations

import numpy as np
import torch

from tbsim.policies.common import Action, RolloutAction
import tbsim.utils.tensor_utils as TensorUtils


# ---------------------------------------------------------------------------
# Bezier interception trajectory (CAT-style rule, Zhang et al. CoRL 2023)
# ---------------------------------------------------------------------------

def compute_bezier_trajectory(adv_p0, adv_yaw0, target_fut_world, T,
                              adv_speed=None, total_time=None):
    """Cubic Bezier interception curve.

    Control points:
        p0 = adv start (current world position of adv)
        p1 = velocity-matched takeoff if (adv_speed, total_time) given, else
             p0 + (L/3) * forward(yaw0) with L = ||p3 - p0||
        p2 = target_fut_world[2T/3]                    # intercept at 2/3 of target horizon
        p3 = target_fut_world[-1]                      # intercept at end of target horizon

    Velocity-matched p1 (preferred when the curve is re-fit in a closed-loop
    refit loop, since it enforces C^1 continuity at refit boundaries):
        For a cubic Bezier with parameter u in [0,1] covering total_time seconds,
        the time-derivative at u=0 is 3(p1-p0)/total_time. Setting this equal to
        adv_speed * forward(yaw0) yields
            p1 = p0 + (adv_speed * total_time / 3) * forward(yaw0).

    Args:
        adv_p0: (2,) np.array, adv world-frame position at t=0.
        adv_yaw0: scalar, adv world-frame yaw at t=0.
        target_fut_world: (T_tgt, 2) np.array, target future positions in world frame.
        T: int, number of curve samples to return.
        adv_speed: optional scalar, adv longitudinal speed (m/s) at t=0.
        total_time: optional scalar, total seconds spanned by the returned curve.

    Returns:
        positions: (T, 2) float32 world positions.
        yaws:      (T,)   float32 world yaws (finite-difference).
    """
    adv_p0 = np.asarray(adv_p0, dtype=np.float64).reshape(2)
    target_fut = np.asarray(target_fut_world, dtype=np.float64).reshape(-1, 2)
    assert target_fut.shape[0] >= 1, "target_fut_world must have at least 1 point"

    p0 = adv_p0
    p3 = target_fut[-1]

    forward = np.array([np.cos(adv_yaw0), np.sin(adv_yaw0)], dtype=np.float64)
    if adv_speed is not None and total_time is not None:
        speed_eff = max(float(adv_speed), 0.5)
        p1 = p0 + (speed_eff * float(total_time) / 3.0) * forward
    else:
        L = float(np.linalg.norm(p3 - p0))
        L = max(L, 1.0)
        p1 = p0 + (L / 3.0) * forward
    if target_fut.shape[0] >= 2:
        i2 = max(0, int(round(2.0 * target_fut.shape[0] / 3.0)) - 1)
        p2 = target_fut[i2]
    else:
        p2 = p3

    u = np.linspace(0.0, 1.0, T, dtype=np.float64)
    one_minus_u = 1.0 - u
    B = (
        (one_minus_u ** 3)[:, None] * p0 +
        3.0 * ((one_minus_u ** 2) * u)[:, None] * p1 +
        3.0 * (one_minus_u * (u ** 2))[:, None] * p2 +
        (u ** 3)[:, None] * p3
    )

    yaws = np.zeros(T, dtype=np.float64)
    if T >= 2:
        dB = np.diff(B, axis=0)
        yaws[:-1] = np.arctan2(dB[:, 1], dB[:, 0])
        yaws[-1] = yaws[-2]
    else:
        yaws[0] = adv_yaw0

    return B.astype(np.float32), yaws.astype(np.float32)


# ---------------------------------------------------------------------------
# Closed-loop Bezier wrapper
# ---------------------------------------------------------------------------

def _target_centric_to_world(positions_tc: np.ndarray,
                              curr_pos: np.ndarray,
                              curr_yaw: float) -> np.ndarray:
    """Convert target-centric positions to world frame using the same
    formula `EnvSplitUnifiedSimulation._step` applies when executing actions.
    """
    cos_y = np.cos(curr_yaw)
    sin_y = np.sin(curr_yaw)
    M_env = np.array([[cos_y,  sin_y],
                      [-sin_y, cos_y]], dtype=np.float64)
    return (positions_tc.astype(np.float64) @ M_env) + curr_pos[None, :]


class BezierReactiveAdvWrapper:
    """Closed-loop Bezier adversary: re-fit curve on every outer get_action call.

    The Bezier curve is re-fit each call using the target's freshly-predicted
    plan as interception target. Drop-in compatible with
    `safety_critical_guided_rollout_with_custom_target`.
    """

    def __init__(self, target_policy, others_predictor,
                 target_name, adv_name, debug=False):
        self.target_policy   = target_policy
        self.others_predictor = others_predictor
        self.target_name  = target_name
        self.adv_name     = adv_name
        self.device       = target_policy.device
        self.current_adv_idx = None
        self._debug = debug

    def eval(self):
        if self.target_policy is not None:
            self.target_policy.eval()
        if self.others_predictor is not None:
            self.others_predictor.eval()

    def _split_agents(self, obs_agents, target_idx, adv_idx):
        obs_agents = TensorUtils.to_torch(obs_agents, device=self.device,
                                           ignore_if_unspecified=True)
        M = len(obs_agents["agent_name"])

        idx_target = torch.tensor([target_idx], device=self.device, dtype=torch.long)
        idx_adv    = torch.tensor([adv_idx],    device=self.device, dtype=torch.long)
        mask = torch.ones(M, device=self.device, dtype=torch.bool)
        mask[idx_target] = False
        mask[idx_adv]    = False
        idx_others = torch.arange(M, device=self.device, dtype=torch.long)[mask]

        def take(x, idx):
            if isinstance(x, torch.Tensor):
                return x.index_select(0, idx)
            if isinstance(x, dict):
                return {k: take(v, idx) for k, v in x.items()}
            if isinstance(x, list):
                il = idx.tolist()
                return [x[i] for i in il]
            return x

        target_batch = {k: take(v, idx_target) for k, v in obs_agents.items()}
        adv_batch    = {k: take(v, idx_adv)    for k, v in obs_agents.items()}
        others_batch = {k: take(v, idx_others) for k, v in obs_agents.items()}
        adv_batch["target_idx"] = idx_target
        return target_batch, adv_batch, others_batch, idx_target, idx_adv, idx_others, M

    def _adv_action_from_bezier(self, adv_batch, target_batch,
                                 target_positions_tc, horizon, dt=0.1):
        adv_cas = adv_batch["curr_agent_state"][0].detach().cpu().numpy().astype(np.float64)
        adv_p0       = adv_cas[:2].reshape(2)
        adv_curr_yaw = float(adv_cas[-1])

        if "curr_speed_signed" in adv_batch:
            adv_speed = float(adv_batch["curr_speed_signed"][0].detach().cpu().numpy())
        elif "curr_speed" in adv_batch:
            adv_speed = float(adv_batch["curr_speed"][0].detach().cpu().numpy())
        else:
            adv_speed = 0.0

        tgt_cas = target_batch["curr_agent_state"][0].detach().cpu().numpy().astype(np.float64)
        tgt_curr_pos = tgt_cas[:2].reshape(2)
        tgt_curr_yaw = float(tgt_cas[-1])

        target_world = _target_centric_to_world(
            target_positions_tc, tgt_curr_pos, tgt_curr_yaw,
        )

        total_time = float(horizon) * float(dt)
        bezier_world_pos, bezier_world_yaw = compute_bezier_trajectory(
            adv_p0, adv_curr_yaw, target_world, T=horizon + 1,
            adv_speed=adv_speed, total_time=total_time,
        )

        # drop index 0 (adv's current position itself);
        # action describes positions at t+1 ... t+horizon
        fut_pos_world = bezier_world_pos[1:horizon + 1]
        fut_yaw_world = bezier_world_yaw[1:horizon + 1]

        cos_y = np.cos(adv_curr_yaw)
        sin_y = np.sin(adv_curr_yaw)
        R_plus = np.array([[cos_y, -sin_y], [sin_y, cos_y]], dtype=np.float64)

        world_delta = fut_pos_world - adv_p0[None, :]
        ac_pos = world_delta @ R_plus

        dy = fut_yaw_world - adv_curr_yaw
        dy = (dy + np.pi) % (2.0 * np.pi) - np.pi

        ac_pos_t = torch.tensor(ac_pos, device=self.device, dtype=torch.float32).unsqueeze(0)
        ac_yaw_t = torch.tensor(dy,     device=self.device, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

        if self._debug:
            d_end = float(np.linalg.norm(fut_pos_world[-1] - target_world[-1]))
            print(f"    [Bezier-reactive] horizon={horizon}  v0={adv_speed:.2f}m/s  "
                  f"|p3-target_end|={d_end:.3f}m")
        return Action(positions=ac_pos_t, yaws=ac_yaw_t)

    def _merge_actions(self, target_pos, target_yaw, adv_pos, adv_yaw,
                       others_pos, others_yaw,
                       idx_target, idx_adv, idx_others, M):
        has_others = others_pos.shape[0] > 0
        T_pos_candidates = [target_pos.shape[1], adv_pos.shape[1]]
        T_yaw_candidates = [target_yaw.shape[1], adv_yaw.shape[1]]
        if has_others:
            T_pos_candidates.append(others_pos.shape[1])
            T_yaw_candidates.append(others_yaw.shape[1])
        T_pos = min(T_pos_candidates)
        T_yaw = min(T_yaw_candidates)

        target_pos = target_pos[:, :T_pos, ...]
        target_yaw = target_yaw[:, :T_yaw, ...]
        adv_pos    = adv_pos   [:, :T_pos, ...]
        adv_yaw    = adv_yaw   [:, :T_yaw, ...]
        if has_others:
            others_pos = others_pos[:, :T_pos, ...]
            others_yaw = others_yaw[:, :T_yaw, ...]

        pos = torch.empty((M,) + target_pos.shape[1:], device=target_pos.device, dtype=target_pos.dtype)
        yaw = torch.empty((M,) + target_yaw.shape[1:], device=target_yaw.device, dtype=target_yaw.dtype)

        pos[idx_target] = target_pos
        yaw[idx_target] = target_yaw
        pos[idx_adv]    = adv_pos
        yaw[idx_adv]    = adv_yaw
        if has_others:
            pos[idx_others] = others_pos
            yaw[idx_others] = others_yaw
        return pos, yaw

    def get_action(self, obs, step_index):
        agent_names = list(obs["agents"]["agent_name"])
        if self.target_name not in agent_names:
            raise RuntimeError(f"target_name {self.target_name!r} not in current obs agents")
        if self.adv_name not in agent_names:
            raise RuntimeError(f"adv_name {self.adv_name!r} not in current obs agents")
        target_idx = agent_names.index(self.target_name)
        adv_idx    = agent_names.index(self.adv_name)
        self.current_adv_idx = adv_idx

        target_batch, adv_batch, others_batch, idx_target, idx_adv, idx_others, M = \
            self._split_agents(obs["agents"], target_idx, adv_idx)

        target_action, _ = self.target_policy.get_action(target_batch,
                                                          step_index=step_index)
        horizon = int(target_action.positions.shape[1])

        if idx_others.numel() > 0:
            others_action, _ = self.others_predictor.get_action(others_batch,
                                                                  step_index=step_index)
        else:
            _dev = target_action.positions.device
            _dtp = target_action.positions.dtype
            others_action = Action(
                positions=torch.zeros((0,) + target_action.positions.shape[1:],
                                       device=_dev, dtype=_dtp),
                yaws=torch.zeros((0,) + target_action.yaws.shape[1:],
                                  device=_dev, dtype=_dtp),
            )

        target_positions_tc = target_action.positions[0].detach().cpu().numpy().astype(np.float64)
        adv_action = self._adv_action_from_bezier(
            adv_batch, target_batch, target_positions_tc, horizon=horizon,
        )

        pos, yaw = self._merge_actions(
            target_action.positions, target_action.yaws,
            adv_action.positions,    adv_action.yaws,
            others_action.positions, others_action.yaws,
            idx_target, idx_adv, idx_others, M,
        )
        agents_action = Action(positions=pos, yaws=yaw)
        info = {"adv_idx": np.full(M, adv_idx, dtype=np.int32)}
        return RolloutAction(agents=agents_action, agents_info=info)
