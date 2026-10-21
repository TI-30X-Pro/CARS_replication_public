"""Compute the 13 adversary-selection features from paper S2.

Where the code and the paper differ, we follow the code:
- The target's direction u is its last displacement (world frame, 0.1 s), not its heading.
- Longitudinal / lateral offsets use the agents' last history points, not their centroids.
- gap = s - (L_tgt + L_i) / 2 - 1 m, TTC = gap / closing if closing > 0.1 m/s, else 999.
- cos_bearing is the cosine of the angle between agent i's heading and the line from i to the
  target (centroids).

Everything is float32, and three operations are pinned down for stable results:
- |v| + 1e-6 was a float64 scalar under NumPy 1.21 (float32 under NumPy 2), so u is divided by
  float32(float64 sum);
- history points go to the world frame product by product (no fused multiply-add);
- arctan2 is glibc's atan2f (libm.py), which the original got from NumPy on Linux.
These choices preserve the feature values expected by the released 800-tree classifier.
"""
import numpy as np
import torch

from cars.selection.libm import atan2f

NAMES = ("rel_dist", "s", "l", "gap", "closing", "lateral_closing", "rel_speed", "curr_speed",
         "cos_yaw_diff", "ttc", "cos_bearing", "dist_rate_5", "abs_l_delta_5")

DT = 0.1
BUMPER_MARGIN = 1.0


def _np(v):
    return v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v)


def to_world(points, m):
    """Transform `(N, T, 2)` points with per-agent matrices using stable float32 arithmetic."""
    p = points.astype(np.float32)
    m = m.astype(np.float32)
    x, y = p[..., 0], p[..., 1]
    wx = (x * m[:, None, 0, 0] + y * m[:, None, 0, 1]) + m[:, None, 0, 2]
    wy = (x * m[:, None, 1, 0] + y * m[:, None, 1, 1]) + m[:, None, 1, 2]
    return np.stack([wx, wy], axis=-1)


def features(obs, target):
    """(N, 13) float32 features of all N agents in `obs` relative to agent `target`.

    obs needs centroid, curr_speed, yaw, history_positions, world_from_agent, extent. The
    target's own row is computed too; callers drop it."""
    centroid = _np(obs["centroid"]).astype(np.float32)
    speed = _np(obs["curr_speed"]).astype(np.float32)
    yaw = _np(obs["yaw"]).astype(np.float32)
    extent = _np(obs["extent"]).astype(np.float32)
    hist = to_world(_np(obs["history_positions"]), _np(obs["world_from_agent"]))   # (N, T, 2)
    hist_t = hist[target]

    rel_dist = np.linalg.norm(centroid - centroid[target], axis=-1)
    rel_speed = speed - speed[target]
    raw = yaw - yaw[target]
    cos_yaw_diff = np.cos(atan2f(np.sin(raw), np.cos(raw)))

    v_t = (hist_t[-1] - hist_t[-2]) / np.float32(DT)
    v = (hist[:, -1] - hist[:, -2]) / np.float32(DT)
    u = v_t / np.float32(np.float64(np.linalg.norm(v_t)) + 1e-6)
    u_perp = np.array([-u[1], u[0]], dtype=np.float32)

    dp = hist[:, -1] - hist_t[-1]
    s = np.sum(dp * u, axis=1)
    l = np.sum(dp * u_perp, axis=1)
    closing = np.sum((v_t - v) * u, axis=1)
    lateral_closing = np.sum((v_t - v) * u_perp, axis=1)

    gap = s - (np.float32(0.5) * (extent[target, 0] + extent[:, 0]) + np.float32(BUMPER_MARGIN))
    ttc = np.where(closing > np.float32(0.1), gap / (closing + np.float32(1e-6)), np.float32(999.0))
    ttc = np.clip(ttc, np.float32(0), np.float32(999.0))

    rel = hist - hist_t                                    # (N, T, 2)
    dist = np.linalg.norm(rel, axis=-1)
    dist_rate_5 = (dist[:, -1] - dist[:, -6]) / np.float32(5 * DT + 1e-6)
    hist_l = np.sum(rel * u_perp, axis=-1)
    abs_l_delta_5 = np.abs(hist_l[:, -1]) - np.abs(hist_l[:, -6])

    bearing = atan2f(centroid[target, 1] - centroid[:, 1], centroid[target, 0] - centroid[:, 0])
    cos_bearing = np.cos(atan2f(np.sin(bearing - yaw), np.cos(bearing - yaw)))

    return np.stack([rel_dist, s, l, gap, closing, lateral_closing, rel_speed, speed, cos_yaw_diff,
                     ttc, cos_bearing, dist_rate_5, abs_l_delta_5], axis=-1).astype(np.float32)
