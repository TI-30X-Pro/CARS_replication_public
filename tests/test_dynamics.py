"""Tests for bounded unicycle integration and inversion."""

import numpy as np
import torch

from cars.sim.dynamics import state_to_action, unicycle_forward


def test_forward_respects_bounds():
    x0 = torch.tensor([[0.0, 0.0, 7.5, 0.0]])
    u = torch.tensor([[[3.0, 5.0]] * 20])
    x = unicycle_forward(x0, u)
    assert x[..., 2].max() <= 8.0 + 1e-6                     # speed cap
    yawrate = torch.diff(torch.cat([x0[:, None, 3:], x[..., 3:]], 1), dim=1) / 0.1
    assert yawrate.abs().max() <= 2.0 / 7.5 + 1e-5           # max_yawvel / v


def test_inverse_of_straight_line():
    t = np.arange(1, 21) * 0.1
    traj = np.stack([5.0 * t, np.zeros_like(t), np.zeros_like(t)], -1)[None]
    out = state_to_action(traj, np.array([5.0]))
    assert np.allclose(out[..., 2], 5.0) and np.allclose(out[..., 4:], 0.0)
