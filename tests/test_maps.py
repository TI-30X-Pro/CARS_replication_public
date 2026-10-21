"""Tests for agent-centric map-image rotation."""

import torch

from cars.data.batch import rotate


def test_rotate_zero_is_identity():
    img = torch.rand(2, 3, 11, 11)
    assert torch.allclose(rotate(img, torch.zeros(2)), img, atol=1e-6)


def test_rotate_quarter_turn_matches_rot90():
    img = torch.rand(1, 1, 9, 9)
    out = rotate(img, torch.tensor([90.0]))
    assert torch.allclose(out, torch.rot90(img, k=1, dims=(-2, -1)), atol=1e-5)
