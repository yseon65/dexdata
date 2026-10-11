"""Quaternion / frame helpers (wxyz). Work with torch tensors and numpy arrays alike."""

import numpy as np

try:
    import torch
except ImportError:  # dataset reading does not require torch
    torch = None


def _lib(x):
    return torch if (torch is not None and isinstance(x, torch.Tensor)) else np


def _stack(xs, axis=-1):
    return torch.stack(xs, dim=axis) if _lib(xs[0]) is torch else np.stack(xs, axis=axis)


def quat_conj(q):
    return _stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]])


def quat_mul(a, b):
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return _stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_apply(q, v):
    """Rotate vectors v (...,3) by unit quaternions q (...,4)."""
    w = q[..., 0:1]
    xyz = q[..., 1:4]
    lib = _lib(q)
    cross = (lambda a, b: torch.cross(a, b, dim=-1)) if lib is torch else (lambda a, b: np.cross(a, b))
    t = 2.0 * cross(xyz, v)
    return v + w * t + cross(xyz, t)


def to_frame(pos, quat, frame_pos, frame_quat):
    """Express poses (pos (...,3), quat (...,4)) in the frame (frame_pos, frame_quat), all world-frame inputs."""
    inv = quat_conj(frame_quat)
    return quat_apply(inv, pos - frame_pos), quat_mul(inv, quat)


def rotate_into(vec, frame_quat):
    """Express a world-frame vector in a frame with orientation frame_quat."""
    return quat_apply(quat_conj(frame_quat), vec)


def xyzw_to_wxyz(q):
    return _stack([q[..., 3], q[..., 0], q[..., 1], q[..., 2]])


def wxyz_to_xyzw(q):
    return _stack([q[..., 1], q[..., 2], q[..., 3], q[..., 0]])


def quat_angle(a, b):
    """Rotation angle between two unit quaternions (numpy)."""
    d = np.abs(np.sum(a * b, axis=-1)).clip(0.0, 1.0)
    return 2.0 * np.arccos(d)
