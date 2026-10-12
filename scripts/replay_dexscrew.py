"""Replay a stored dexscrew (nut-bolt) episode kinematically in the Isaac Sim GUI.

Every control step (20 Hz) the hand joints and the nut joint are set to the stored joint_pos / object_dof_pos,
so what you see is exactly the recorded trajectory (no policy, no physics drift). The episode loops.

  ~/env_isaaclab6/bin/python scripts/replay_dexscrew.py --shard nutbolt_student_s0_00003 --index 311
  ~/env_isaaclab6/bin/python scripts/replay_dexscrew.py --episode_id 22358 --run nutbolt_student_s0
"""
import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

HERE = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('--data', default=os.path.join(HERE, '..', 'data', 'v1'))
parser.add_argument('--shard', default=None, help='shard name, e.g. nutbolt_student_s0_00003 (with --index)')
parser.add_argument('--index', type=int, default=None, help='episode index inside the shard')
parser.add_argument('--episode_id', type=int, default=None, help='alternatively: episode_id (with --run)')
parser.add_argument('--run', default='nutbolt_student_s0')
parser.add_argument('--speed', type=float, default=1.0, help='playback speed (1 = real time)')
parser.add_argument('--loops', type=int, default=0, help='0 = loop until the window is closed')
parser.add_argument('--dexscrew_root', default=os.path.expanduser('~/IsaacLab/projects/dexscrew'))
args = parser.parse_args()
app = AppLauncher(headless=False).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, args.dexscrew_root)
from dexdata import schema  # noqa: E402
from dexdata.dataset import TrajectoryDataset  # noqa: E402
from dexscrew.tasks import isaaclab_task_map  # noqa: E402
from dexscrew.utils.reformat import omegaconf_to_dict  # noqa: E402

# ---- episode
ds = TrajectoryDataset(args.data, runs=[args.run] if args.shard is None else None)
si = i = None
for k, sh in enumerate(ds.shards):
    name = os.path.basename(sh['path'])
    if args.shard is not None and name == args.shard:
        si, i = k, args.index
        break
    if args.episode_id is not None:
        hit = np.nonzero(sh['episodes']['episode_id'] == args.episode_id)[0]
        if len(hit):
            si, i = k, int(hit[0])
            break
if si is None or i is None:
    sys.exit('episode not found')
ep = {k: v[i] for k, v in ds.shards[si]['episodes'].items()}
s, n = int(ep['start']), int(ep['length'])
q = np.asarray(ds.field(si, 'joint_pos')[s:s + n])
nut = np.array(ds.field(si, 'object_dof_pos')[s:s + n, 0])
nut[-1] = nut[-2]  # the last stored row holds the post-reset nut angle (known collector issue)
scale = float(ep['object_scale'])
reasons = {v: k for k, v in schema.END_REASONS.items()}
print(f"[replay] {os.path.basename(ds.shards[si]['path'])} #{i} episode_id {ep['episode_id']} | {n} steps "
      f"({n * 0.05:.1f} s) | scale {scale} | mass {ep['object_mass'] * 1e3:.0f} g | friction {ep['friction']:.2f} "
      f"| end {reasons[int(ep['end_reason'])]} | nut {nut[-1]:.1f} rad", flush=True)

# ---- env with one nut of the episode's size (student collection preset)
for name, fn in (('eq', lambda x, y: x.lower() == y.lower()), ('contains', lambda x, y: x.lower() in y.lower()),
                 ('if', lambda pred, a, b: a if pred else b),
                 ('resolve_default', lambda default, arg: default if arg == '' else arg)):
    OmegaConf.register_new_resolver(name, fn)
with initialize_config_dir(config_dir=os.path.join(args.dexscrew_root, 'configs'), version_base=None):
    config = compose('config', overrides=['task=XHandHoraNutBolt', 'task.env.numEnvs=1', 'test=True',
                                          'task.env.object.type=screw_trinut',
                                          f'task.env.randomization.randomizeScaleList=[{scale}]',
                                          '+task.env.keepNutHeight=True'])
env = isaaclab_task_map['XHandHoraNutBolt'](omegaconf_to_dict(config.task), config.sim_device, 0, False)
env.reset()
hand, obj, dev = env.hand, env.object, env.device
jids = hand.find_joints(list(schema.JOINT_NAMES), preserve_order=True)[0]
q_t = torch.tensor(q, device=dev)
nut_t = torch.tensor(nut, device=dev)[:, None]
zq, zn = torch.zeros((1, schema.N_J), device=dev), torch.zeros((1, 1), device=dev)
hand.set_joint_effort_target(zq, joint_ids=jids)

loop = 0
while app.is_running() and (args.loops == 0 or loop < args.loops):
    for t in range(n):
        t0 = time.time()
        hand.write_joint_state_to_sim(q_t[t:t + 1], zq, joint_ids=jids)
        hand.set_joint_position_target(q_t[t:t + 1], joint_ids=jids)
        obj.write_joint_state_to_sim(nut_t[t:t + 1], zn)
        env.scene.write_data_to_sim()
        env.sim.step(render=True)
        env.scene.update(env.sim.get_physics_dt())
        if t % 100 == 0:
            print(f'[replay] t {t * 0.05:5.1f} s | nut {nut[t]:6.2f} rad', flush=True)
        if not app.is_running():
            break
        time.sleep(max(0.0, 0.05 / args.speed - (time.time() - t0)))
    loop += 1
os._exit(0)
