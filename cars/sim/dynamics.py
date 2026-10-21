"""Integrate and invert the bounded unicycle dynamics used by CARS.

State: x, y, speed, yaw. Action: acceleration and yaw rate. At dt = 0.1 s, the bounds are:
  a in [-3, 3], v in [0, 8], |yaw rate| <= max(0.1, min(0.25 |v|, 2 / max(|v|, 0.1))).

`unicycle_forward` integrates a whole action sequence at once (tbsim's
`unicyle_forward_dynamics`, mode "parallel"): acceleration is clipped, speed is the clipped
running sum, the yaw-rate bound uses the speed at the start of each step, and position
uses the average of the speeds at the start and end of each step.
`state_to_action` is the inverse used for training targets
(tbsim's `convert_state_to_state_and_action`).
"""
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class UnicycleBounds:
    max_steer: float = 0.25
    max_yawvel: float = 2.0
    acce_bound: tuple = (-3.0, 3.0)
    vbound: tuple = (0.0, 8.0)


def _bmm(a, b):
    if a.ndim == 3:
        return torch.bmm(a, b)
    return torch.einsum("bijk,bikl->bijl", a, b)


def unicycle_forward(initial_states, actions, dt=0.1, bounds=UnicycleBounds()):
    """initial_states (B, [A], 4): x, y, v, yaw; actions (B, [A], T, 2) -> states (B, [A], T, 4)."""
    n = actions.shape[-2]
    bm = actions.shape[:-2]
    device = initial_states.device
    with torch.no_grad():
        cum = torch.tril(torch.ones(n + 1, n + 1, device=device)).repeat(*bm, 1, 1)
        m2 = torch.ones(n, n + 1, device=device)
        avg = (torch.logical_xor(torch.tril(m2, diagonal=1), torch.tril(m2, diagonal=-1)).float() * 0.5)
        avg = avg.repeat(*bm, 1, 1)

    acc = torch.clip(actions[..., :1], bounds.acce_bound[0], bounds.acce_bound[1])
    yawvel = actions[..., 1:]

    v_full = torch.clip(_bmm(cum, torch.cat((initial_states[..., -2:-1].unsqueeze(-2), acc * dt), dim=-2)),
                        bounds.vbound[0], bounds.vbound[1])
    v_avg = _bmm(avg, v_full)
    v = v_full[..., 1:, :]

    with torch.no_grad():
        v_prev = v_full[..., :-1, :]
        yawbound = torch.minimum(bounds.max_steer * torch.abs(v_prev),
                                 bounds.max_yawvel / torch.clip(torch.abs(v_prev), min=0.1))
        yawbound = torch.clip(yawbound, min=0.1)
    yawvel = torch.clip(yawvel, -yawbound, yawbound)

    yaw_full = _bmm(cum, torch.cat((initial_states[..., -1:].unsqueeze(-2), yawvel * dt), dim=-2))
    yaw = yaw_full[..., 1:, :]
    yaw_prev = yaw_full[..., :-1, :]
    vxy = torch.cat((v_avg * torch.cos(yaw_prev), v_avg * torch.sin(yaw_prev)), dim=-1)
    xy = _bmm(cum, torch.cat((initial_states[..., :2].unsqueeze(-2), vxy * dt), dim=-2))[..., 1:, :]
    return torch.cat((xy, v, yaw), dim=-1)


def angle_diff(a, b):
    """Smallest signed difference a - b."""
    period = 2 * np.pi
    d = (a - b + period / 2) % period - period / 2
    d[d > np.pi] = d[d > np.pi] - 2 * np.pi
    return d


def state_to_action(traj, v0, dt=0.1):
    """traj (B, [A], T, 3): x, y, yaw in the agent frame (start at the origin, yaw 0);
    v0 (B, [A]) current speed -> (B, [A], T, 6): x, y, v, yaw, a, yaw rate.
    Speed is the displacement projected on the new heading; works on torch and numpy."""
    is_torch = isinstance(traj, torch.Tensor)
    sin, cos = (torch.sin, torch.cos) if is_torch else (np.sin, np.cos)
    cat = (lambda a, d: torch.cat(a, dim=d)) if is_torch else (lambda a, d: np.concatenate(a, axis=d))
    bm = traj.shape[:-2]
    if is_torch:
        pos0 = torch.zeros(*bm, 1, 2, device=traj.device)
        yaw0 = torch.zeros(*bm, 1, 1, device=traj.device)
    else:
        pos0, yaw0 = np.zeros((*bm, 1, 2)), np.zeros((*bm, 1, 1))
    pos = cat((pos0, traj[..., :2]), -2)
    yaw = cat((yaw0, traj[..., 2:]), -2)
    vel = ((pos[..., 1:, 0:1] - pos[..., :-1, 0:1]) / dt * cos(yaw[..., 1:, :])
           + (pos[..., 1:, 1:2] - pos[..., :-1, 1:2]) / dt * sin(yaw[..., 1:, :]))
    vel = cat((v0[..., None, None], vel), -2)
    acc = (vel[..., 1:, :] - vel[..., :-1, :]) / dt
    yawvel = angle_diff(yaw[..., 1:, :], yaw[..., :-1, :]) / dt
    return cat((pos[..., 1:, :], vel[..., 1:, :], yaw[..., 1:, :], acc, yawvel), -1)
