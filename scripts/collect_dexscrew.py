"""Collect XHand trajectories from a dexscrew (Isaac Lab port) policy into a dexdata dataset.

Default = decision D6: nut-bolt (trinut) ProprioAdapt student, student training conditions.

  ~/env_isaaclab6/bin/python scripts/collect_dexscrew.py --out data/v1 --run_name nutbolt_student_s0 \
      --num_envs 1024 --target_transitions 30000000 \
      --ckpt ~/IsaacLab/projects/dexscrew/outputs/XHandHoraNutBolt_student_padapt/lab_run1/stage2_nn/model_last.ckpt

Extra hydra overrides for the dexscrew task config can be appended (e.g. task.env.object.type=screw_boxnut).
"""
import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

HERE = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('--dexscrew_root', default=os.path.expanduser('~/IsaacLab/projects/dexscrew'))
parser.add_argument('--task', default='XHandHoraNutBolt', choices=['XHandHoraNutBolt', 'XHandHoraScrewDriver'])
parser.add_argument('--policy', default='student', choices=['student', 'teacher'])
parser.add_argument('--ckpt', required=True)
parser.add_argument('--stochastic', action='store_true', help='teacher only: sample from the PPO Gaussian')
parser.add_argument('--preset', default='auto', choices=['auto', 'student', 'teacher', 'none'],
                    help='env overrides matching the policy training conditions (auto: follows --policy)')
parser.add_argument('--out', required=True, help='dataset root (created if missing)')
parser.add_argument('--run_name', required=True)
parser.add_argument('--task_name', default=None, help='dataset task label (default derived from env)')
parser.add_argument('--policy_name', default=None, help='dataset policy label (default derived)')
parser.add_argument('--num_envs', type=int, default=1024)
parser.add_argument('--target_transitions', type=float, default=1e6)
parser.add_argument('--shard_transitions', type=float, default=2e6)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--headless', action='store_true', default=True)
args, overrides = parser.parse_known_args()
app = AppLauncher(headless=args.headless).app

import torch  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, args.dexscrew_root)
from dexdata.adapters import DexscrewAdapter  # noqa: E402
from dexdata.recorder import EpisodeRecorder  # noqa: E402
from dexscrew.algo.ppo.padapt import ProprioAdapt  # noqa: E402
from dexscrew.algo.ppo.ppo import PPO  # noqa: E402
from dexscrew.tasks import isaaclab_task_map  # noqa: E402
from dexscrew.utils.reformat import omegaconf_to_dict  # noqa: E402

for name, fn in (('eq', lambda x, y: x.lower() == y.lower()), ('contains', lambda x, y: x.lower() in y.lower()),
                 ('if', lambda pred, a, b: a if pred else b),
                 ('resolve_default', lambda default, arg: default if arg == '' else arg)):
    OmegaConf.register_new_resolver(name, fn)

# Same conditions as scripts/lab_*_student_padapt.sh / lab_*_teacher.sh in dexscrew.
PRESETS = {
    ('XHandHoraNutBolt', 'student'): ['task.env.object.type=screw_trinut', 'task.env.reset_dist_threshold=0.12',
                                      'task.env.randomization.obs_noise_t_scale=0.01',
                                      'task.env.randomization.obs_noise_e_scale=0.02'],
    ('XHandHoraNutBolt', 'teacher'): ['task.env.object.type=screw_trinut', 'task.env.reset_dist_threshold=0.07'],
    ('XHandHoraScrewDriver', 'student'): ['task.env.reset_dist_threshold=0.15',
                                          'task.env.randomization.obs_noise_t_scale=0.01',
                                          'task.env.randomization.obs_noise_e_scale=0.02'],
    ('XHandHoraScrewDriver', 'teacher'): ['task.env.reset_dist_threshold=0.1'],
}
preset = args.policy if args.preset == 'auto' else args.preset
env_overrides = (PRESETS.get((args.task, preset), []) if preset != 'none' else []) + overrides

with initialize_config_dir(config_dir=os.path.join(args.dexscrew_root, 'configs'), version_base=None):
    config = compose('config', overrides=[f'task={args.task}', f'task.env.numEnvs={args.num_envs}', 'test=True',
                                          f'seed={args.seed}'] + env_overrides
                     + (['train.algo=ProprioAdapt', 'train.ppo.proprio_adapt=True'] if args.policy == 'student'
                        else ['train.algo=PPO']))
task_cfg = omegaconf_to_dict(config.task)
task_cfg['seed'] = args.seed
env = isaaclab_task_map[args.task](task_cfg, config.sim_device, 0, True)
scratch = os.path.join(os.path.abspath(args.out), '.agent_scratch')
agent = (ProprioAdapt if args.policy == 'student' else PPO)(env, scratch, full_config=config)
agent.restore_test(os.path.expanduser(args.ckpt))
agent.set_eval()


@torch.no_grad()
def policy(obs):
    pc = obs['point_cloud_info']
    if agent.normalize_point_cloud:
        pc = agent.point_cloud_mean_std(pc.reshape(-1, 3)).reshape((pc.shape[0], -1, 3))
    if args.policy == 'student':  # ProprioAdapt.test(): proprio history only
        inp = {'obs': agent.running_mean_std(obs['obs']), 'proprio_hist': agent.sa_mean_std(obs['proprio_hist']),
               'point_cloud_info': pc}
        mu, _, _ = agent.model.act_inference(inp)
        return mu.clamp(-1.0, 1.0)
    inp = {'obs': agent.running_mean_std(obs['obs']),
           'priv_info': agent.priv_mean_std(obs['priv_info']) if agent.normalize_priv else obs['priv_info'],
           'proprio_hist': obs['proprio_hist'], 'point_cloud_info': pc}
    if args.stochastic:
        return agent.model.act(inp)['actions'].clamp(-1.0, 1.0)
    mu, _, _ = agent.model.act_inference(inp)
    return mu.clamp(-1.0, 1.0)


object_type = task_cfg['env']['object']['type'].replace('screw_', '')
task_name = args.task_name or ('nutbolt_' + object_type if args.task == 'XHandHoraNutBolt' else 'screwdriver')
policy_name = args.policy_name or f'dexscrew_{args.policy}' + ('_stochastic' if args.stochastic else '')
adapter = DexscrewAdapter(env)
step_dt = env.control_freq_inv * env.dt
rec = EpisodeRecorder(
    args.out, args.run_name, task=task_name, policy=policy_name, num_envs=args.num_envs,
    max_episode_steps=env.max_episode_length, step_dt=step_dt, shard_transitions=args.shard_transitions,
    run_info={'collector': 'collect_dexscrew', 'env_task': args.task, 'checkpoint': os.path.expanduser(args.ckpt),
              'policy': args.policy, 'stochastic': args.stochastic, 'env_overrides': env_overrides,
              'seed': args.seed, 'dexscrew_root': args.dexscrew_root, 'object_type': object_type})

obs = env.reset()
t0, last = time.time(), 0
status = 'complete'
try:
    while rec.kept_transitions < args.target_transitions:
        state = adapter.read_state()
        action = policy(obs)
        obs, post, done, reason = adapter.step(action)
        rec.add({**state, **post}, done, reason, adapter.episode_params)
        if rec.raw_transitions - last >= 50 * args.num_envs:
            last = rec.raw_transitions
            s = rec.stats()
            print(f"[collect] kept {s['transitions']:,} / {int(args.target_transitions):,} | episodes {s['episodes']} "
                  f"| raw {s['raw_transitions']:,} | {s['raw_transitions'] / (time.time() - t0):,.0f} steps/s "
                  f"| end reasons {s['end_reasons']}", flush=True)
except KeyboardInterrupt:
    status = 'interrupted'
finally:
    rec.close(status)
    print(f'[collect] {status}: {rec.stats()}', flush=True)
    os._exit(0)
