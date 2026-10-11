"""Unit tests for dexdata.frames (wxyz). Run: ~/env_isaaclab6/bin/python tests/test_frames.py"""
import os
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from dexdata import frames  # noqa: E402


def rand_quat(n, rng):
    xyzw = Rotation.random(n, random_state=rng).as_quat()
    return xyzw[:, [3, 0, 1, 2]]


def test_against_scipy():
    rng = np.random.default_rng(0)
    a, b = rand_quat(100, rng), rand_quat(100, rng)
    v = rng.normal(size=(100, 3))
    ra, rb = Rotation.from_quat(a[:, [1, 2, 3, 0]]), Rotation.from_quat(b[:, [1, 2, 3, 0]])
    assert np.allclose(frames.quat_apply(a, v), ra.apply(v), atol=1e-9)
    ab = frames.quat_mul(a, b)
    ref = (ra * rb).as_quat()[:, [3, 0, 1, 2]]
    assert np.allclose(np.abs((ab * ref).sum(-1)), 1.0, atol=1e-9)


def test_to_frame_roundtrip_torch_numpy():
    rng = np.random.default_rng(1)
    fq, q = rand_quat(50, rng), rand_quat(50, rng)
    fp, p = rng.normal(size=(50, 3)), rng.normal(size=(50, 3))
    lp, lq = frames.to_frame(p, q, fp, fq)
    # back to world
    assert np.allclose(frames.quat_apply(fq, lp) + fp, p, atol=1e-9)
    assert np.allclose(np.abs((frames.quat_mul(fq, lq) * q).sum(-1)), 1.0, atol=1e-9)
    tp, tq = frames.to_frame(*(torch.as_tensor(x) for x in (p, q, fp, fq)))
    assert np.allclose(tp.numpy(), lp) and np.allclose(tq.numpy(), lq)


def test_convention_conversion():
    q = np.array([[0.1, 0.2, 0.3, 0.9]])
    assert np.allclose(frames.wxyz_to_xyzw(frames.xyzw_to_wxyz(q)), q)


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('PASS', name)
