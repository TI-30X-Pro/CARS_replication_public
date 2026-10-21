"""Tests for diffusion sampling, checkpoint loading, plan selection, and PPO utilities.

Checkpoint-loading tests use CARS_CKPT, which defaults to work/checkpoints.
"""
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from cars.generation.gmdm import GMDM, ddim_timesteps
from cars.generation.policy import closest_to_log

CKPT = Path(os.environ.get("CARS_CKPT", Path(__file__).resolve().parents[1] / "work" / "checkpoints"))


def ckpt(name):
    p = CKPT / f"{name}.ckpt"
    if not p.exists():
        pytest.skip(f"{p} not found (set CARS_CKPT)")
    return p


def test_ddim_timesteps():
    t, t_next = ddim_timesteps(20, 100)
    assert len(t) == 20 and t[0] == 99 and t[-1] == 0 and t_next[-1] == 0 and t_next[:-1] == t[1:]


@pytest.mark.parametrize("name", ["g8", "cars_adversary"])
def test_checkpoint_loads_and_schedule_matches(name):
    sd = torch.load(ckpt(name), map_location="cpu", weights_only=False)["state_dict"]
    fresh = GMDM()
    for k, v in fresh.named_buffers(recurse=False):
        np.testing.assert_allclose(v.numpy(), sd["nets.policy." + k].numpy(), rtol=1e-6, err_msg=k)
    net = GMDM.from_checkpoint(ckpt(name))
    assert 18.4e6 < sum(p.numel() for p in net.parameters()) < 18.6e6


def test_closest_to_log_ignores_missing_steps():
    pos = torch.zeros(1, 2, 3, 2)
    pos[0, 1, :, 0] = torch.tensor([0.0, 0.0, 9.0])
    pos[0, 0, :, 0] = 1.0
    batch = {"target_positions": torch.zeros(1, 3, 2),
             "target_availabilities": torch.tensor([[True, True, False]])}
    assert closest_to_log(pos, batch).tolist() == [1]


def test_gae_cuts_bootstrap_only_on_termination():
    from cars.generation.ppo import gae
    rew, v, v_next = [1.0, 1.0, 1.0], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    ret, adv = gae(rew, v, v_next, terminated=[False, False, True], truncated=[False, False, False])
    assert ret[-1] == pytest.approx(1.0)
    ret, adv = gae(rew, v, v_next, terminated=[False, False, False], truncated=[False, False, True])
    assert ret[-1] == pytest.approx(1.0 + 0.99 * 0.5)


def test_minibatches_merge_a_short_last_one():
    from cars.generation.ppo import splits
    rng = np.random.RandomState(0)
    sizes = [len(b) for b in splits(30, 16, rng)]
    assert sizes == [30]
    assert [len(b) for b in splits(40, 16, rng)] == [16, 24]
    assert [len(b) for b in splits(32, 16, rng)] == [16, 16]
