"""Tests for track interpolation, heading wrapping, and finite differences."""

import numpy as np

from cars.data.tracks import _differences, angle_wrap, upsample


def test_angle_wrap_range():
    a = np.linspace(-10, 10, 1001)
    w = angle_wrap(a)
    assert np.all(w >= -np.pi) and np.all(w < np.pi)
    assert np.allclose(np.sin(w), np.sin(a)) and np.allclose(np.cos(w), np.cos(a))


def test_differences_constant_velocity():
    t = np.arange(6) * 0.5
    pos = np.stack([2.0 * t, -1.0 * t, np.zeros_like(t)], axis=1)
    vel, acc = _differences(pos)
    assert np.allclose(vel, [[2.0, -1.0]] * 6)
    assert np.allclose(acc, 0.0)


def test_upsample_keeps_keyframes_and_is_linear():
    ego = np.zeros((3, 8))
    ego[:, 0] = [0.0, 5.0, 10.0]
    other = np.zeros((2, 8))
    other[:, 7] = [3.1, -3.1]          # heading crosses +-pi
    out = upsample(np.array([0, 1]), [ego, other], 5)
    assert out[0].shape == (11, 8) and out[1].shape == (6, 8)
    assert np.allclose(out[0][:, 0], np.arange(11))
    # interpolated the short way round, not through 0
    assert np.all(np.abs(out[1][:, 7]) > 3.0)
