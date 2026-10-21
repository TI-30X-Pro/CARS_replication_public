"""Box transforms and the rollout and replay collision tests.

Both tests take box 1, enlarge it by 1.0 m in length and 0.3 m in width, and report a
collision when one of its four corners lies inside box 2:
  - rollout (`veh_veh_collision`): only box 1 is enlarged, box 2 keeps its size. The
    caller supplies the adversary as box 1 and the target as box 2.
  - replay (`obb_collision`): both boxes are enlarged.
Neither test is symmetric: a crossing where no corner of box 1 is inside box 2 is missed.
"""
import numpy as np
import torch

OFFSET_X, OFFSET_Y = 1.0, 0.3


def rotate_2d(xy, theta):
    """Rotate points (..., 2) by angles theta (...)."""
    if isinstance(xy, torch.Tensor):
        x1 = xy[..., 0] * torch.cos(theta) - xy[..., 1] * torch.sin(theta)
        y1 = xy[..., 1] * torch.cos(theta) + xy[..., 0] * torch.sin(theta)
        return torch.stack([x1, y1], dim=-1)
    x1 = xy[..., 0] * np.cos(theta) - xy[..., 1] * np.sin(theta)
    y1 = xy[..., 1] * np.cos(theta) + xy[..., 0] * np.sin(theta)
    return np.concatenate((x1.reshape(-1, 1), y1.reshape(-1, 1)), axis=-1)


def veh_veh_collision(p1, p2, s1, s2, offset_x=OFFSET_X, offset_y=OFFSET_Y):
    """Signed corner distance of the enlarged box 1 to box 2; < 0 means collision.

    p: (..., 3) x, y, yaw; s: (..., 2) length, width. Same arithmetic as tbsim's
    VEH_VEH_collision (numpy or torch).
    """
    if isinstance(p1, torch.Tensor):
        cx = torch.kron(s1[..., 0] + offset_x, torch.tensor([0.5, 0.5, -0.5, -0.5]).to(p1.device))
        cy = torch.kron(s1[..., 1] + offset_y, torch.tensor([0.5, -0.5, 0.5, -0.5]).to(p1.device))
        corners = torch.stack([cx, cy], dim=-1)
        dx = (p1[..., 0:2] - p2[..., 0:2]).repeat_interleave(4, dim=-2)
        d1 = rotate_2d(corners, p1[..., 2].repeat_interleave(4, dim=-1)) + dx
        d2 = rotate_2d(d1, -p2[..., 2].repeat_interleave(4, dim=-1))
        dis = torch.maximum(torch.abs(d2[..., 0]) - 0.5 * s2[..., 0].repeat_interleave(4, dim=-1),
                            torch.abs(d2[..., 1]) - 0.5 * s2[..., 1].repeat_interleave(4, dim=-1))
        return torch.min(dis.view(*s1.shape[:-1], 4), dim=-1)[0]
    cx = np.kron(s1[..., 0] + offset_x, np.array([0.5, 0.5, -0.5, -0.5]))
    cy = np.kron(s1[..., 1] + offset_y, np.array([0.5, -0.5, 0.5, -0.5]))
    corners = np.stack([cx, cy], axis=-1)
    dx = (p1[..., 0:2] - p2[..., 0:2]).repeat(4, axis=-2)
    d1 = rotate_2d(corners, p1[..., 2].repeat(4, axis=-1)) + dx
    d2 = rotate_2d(d1, -p2[..., 2].repeat(4, axis=-1))
    dis = np.maximum(np.abs(d2[..., 0]) - 0.5 * s2[..., 0].repeat(4, axis=-1),
                     np.abs(d2[..., 1]) - 0.5 * s2[..., 1].repeat(4, axis=-1))
    return np.min(dis.reshape(*s1.shape[:-1], 4), axis=-1)


def obb_collision(pos1, yaw1, ext1, pos2, yaw2, ext2, offset_x=OFFSET_X, offset_y=OFFSET_Y):
    """Replay version: both boxes enlarged. Returns (collision, min corner distance)."""
    p1 = np.array([pos1[0], pos1[1], yaw1]).reshape(1, 3)
    p2 = np.array([pos2[0], pos2[1], yaw2]).reshape(1, 3)
    s1 = np.array([[ext1[0] + offset_x, ext1[1] + offset_y]], dtype=np.float64)
    s2 = np.array([[ext2[0] + offset_x, ext2[1] + offset_y]], dtype=np.float64)
    d = veh_veh_collision(p1, p2, s1, s2, offset_x=0.0, offset_y=0.0)
    min_dis = float(d[0])
    return min_dis < 0, min_dis


def transform_points(points, matrix):
    """Points (..., 2) through 3x3 transforms (..., 3, 3), row-vector convention."""
    if isinstance(points, torch.Tensor):
        return points @ matrix[..., :2, :2].transpose(-1, -2) + matrix[..., None, :2, 2]
    return points @ np.swapaxes(matrix[..., :2, :2], -1, -2) + matrix[..., None, :2, 2]


def affine_f32(xy, m):
    """x, y (2,) through a 3x3 transform (row i: m[i,0] x + m[i,1] y + m[i,2]), in float32,
    each product rounded on its own (no fused multiply-add).

    Avoiding fused multiply-add keeps boundary decisions stable across supported platforms.
    """
    x, y = np.float32(xy[0]), np.float32(xy[1])
    m = np.asarray(m, dtype=np.float32)
    return np.array([np.float32(np.float32(x * m[0, 0]) + np.float32(y * m[0, 1])) + m[0, 2],
                     np.float32(np.float32(x * m[1, 0]) + np.float32(y * m[1, 1])) + m[1, 2]],
                    dtype=np.float32)
