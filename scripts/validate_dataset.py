"""Consistency checks for a dexdata dataset (no simulator needed).

  ~/env_isaaclab6/bin/python scripts/validate_dataset.py data/v1 [--urdf .../xhand_left.urdf] [--samples 5000]

Checks
  1. shapes / dtypes match the schema, no NaN in fields that must be finite
  2. quaternions are unit norm
  3. control_error == joint_target - joint_pos
  4. within an episode: joint_target[t+1] == applied_target[t]   (row t+1 is the next state of row t)
  5. forward kinematics: keypoint poses in the wrist frame recomputed from joint_pos with the URDF
     match the stored keypoints (position < 1 mm, orientation < 0.5 deg)
"""
import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from dexdata import schema  # noqa: E402
from dexdata.dataset import TrajectoryDataset  # noqa: E402
from dexdata.frames import quat_angle  # noqa: E402

ALLOW_NAN = {"object_dof_pos", "object_dof_vel", "finger_contact_force", "goal_pos_wrist", "goal_quat_wrist"}


class URDFKinematics:
    def __init__(self, path):
        root = ET.parse(path).getroot()
        self.joints = {j.find('child').get('link'): j for j in root.findall('joint')}

    def _local(self, joint, q):
        o = joint.find('origin')
        xyz = np.array([float(v) for v in (o.get('xyz') or '0 0 0').split()]) if o is not None else np.zeros(3)
        rpy = [float(v) for v in (o.get('rpy') or '0 0 0').split()] if o is not None else [0, 0, 0]
        T = np.repeat(np.eye(4)[None], len(q), axis=0)
        T[:, :3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
        T[:, :3, 3] = xyz
        if joint.get('type') in ('revolute', 'continuous'):
            axis = np.array([float(v) for v in joint.find('axis').get('xyz').split()])
            J = np.repeat(np.eye(4)[None], len(q), axis=0)
            J[:, :3, :3] = Rotation.from_rotvec(axis[None] * q[:, None]).as_matrix()
            T = T @ J
        return T

    def link_pose(self, link, qdict, n):
        """Pose of `link` in the root (wrist) frame for n samples; qdict: joint name -> (n,) angles."""
        if link not in self.joints:
            return np.repeat(np.eye(4)[None], n, axis=0)
        j = self.joints[link]
        q = qdict.get(j.get('name'), np.zeros(n))
        return self.link_pose(j.find('parent').get('link'), qdict, n) @ self._local(j, q)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('root')
    p.add_argument('--urdf', default=os.path.expanduser('~/IsaacLab/projects/dexscrew/assets/xhand_left/urdf/xhand_left.urdf'))
    p.add_argument('--samples', type=int, default=5000)
    p.add_argument('--fk_pos_tol', type=float, default=1e-3)
    p.add_argument('--fk_rot_tol_deg', type=float, default=0.5)
    a = p.parse_args()
    ds = TrajectoryDataset(a.root)
    ok = True

    def report(name, passed, detail):
        nonlocal ok
        ok &= passed
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")

    print('summary:', ds.summary())
    rng = np.random.default_rng(0)
    fk = URDFKinematics(a.urdf)
    for si, sh in enumerate(ds.shards):
        name = os.path.basename(sh['path'])
        T = len(ds.field(si, 'joint_pos'))
        # 1. shapes and finiteness
        bad = []
        for f, (shape, dtype, _) in schema.STEP_FIELDS.items():
            v = ds.field(si, f)
            if v.shape != (T, *shape) or v.dtype != np.dtype(dtype):
                bad.append(f'{f} {v.shape} {v.dtype}')
            elif f not in ALLOW_NAN and dtype != 'bool' and not np.isfinite(v).all():
                bad.append(f'{f} non-finite')
        report(f'{name} schema', not bad, bad or f'{T} rows, {len(sh["episodes"]["length"])} episodes')
        lengths = sh['episodes']['length']
        report(f'{name} episode table', int(lengths.sum()) == T and (np.diff(sh['episodes']['start']) == lengths[:-1]).all(),
               f'sum(length)={int(lengths.sum())}')
        # 2. quaternion norms
        qn = [np.abs(np.linalg.norm(ds.field(si, f), axis=-1) - 1).max()
              for f in ('keypoint_quat_wrist', 'wrist_quat_world', 'object_quat_wrist')]
        report(f'{name} unit quaternions', max(qn) < 1e-4, f'max |norm-1| = {max(qn):.2e}')
        # 3. control error
        ce = np.abs(ds.field(si, 'control_error') - (ds.field(si, 'joint_target') - ds.field(si, 'joint_pos'))).max()
        report(f'{name} control_error', ce < 1e-5, f'max diff {ce:.2e}')
        # 4. temporal consistency inside episodes
        nxt = np.ones(T, dtype=bool)
        nxt[(sh['episodes']['start'] + lengths - 1)] = False  # last row of each episode has no next row
        idx = np.nonzero(nxt)[0]
        jt = ds.field(si, 'joint_target')
        at = ds.field(si, 'applied_target')
        d = np.abs(jt[idx + 1] - at[idx]).max() if len(idx) else 0.0
        report(f'{name} next-state alignment', d < 1e-5, f'max |joint_target[t+1]-applied_target[t]| = {d:.2e}')
        # 5. forward kinematics
        rows = np.sort(rng.choice(T, size=min(a.samples, T), replace=False))
        q = np.asarray(ds.field(si, 'joint_pos')[rows], dtype=np.float64)
        qdict = {j: q[:, i] for i, j in enumerate(schema.JOINT_NAMES)}
        kp_p = np.asarray(ds.field(si, 'keypoint_pos_wrist')[rows])
        kp_q = np.asarray(ds.field(si, 'keypoint_quat_wrist')[rows])
        perr, rerr = [], []
        for k, body in enumerate(schema.KEYPOINT_BODIES):
            M = fk.link_pose(body, qdict, len(rows))
            perr.append(np.linalg.norm(M[:, :3, 3] - kp_p[:, k], axis=-1))
            xyzw = Rotation.from_matrix(M[:, :3, :3]).as_quat()
            rerr.append(np.degrees(quat_angle(xyzw[:, [3, 0, 1, 2]], kp_q[:, k])))
        perr, rerr = np.stack(perr, -1), np.stack(rerr, -1)
        report(f'{name} FK keypoint position', np.percentile(perr, 99.9) < a.fk_pos_tol,
               f'mean {perr.mean() * 1e3:.3f} mm, p99.9 {np.percentile(perr, 99.9) * 1e3:.3f} mm, max {perr.max() * 1e3:.3f} mm')
        report(f'{name} FK keypoint orientation', np.percentile(rerr, 99.9) < a.fk_rot_tol_deg,
               f'mean {rerr.mean():.3f} deg, p99.9 {np.percentile(rerr, 99.9):.3f} deg, max {rerr.max():.3f} deg')
    print('ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
