"""Collect XHand trajectories from a dexgen env + RSL-RL policy into a dexdata dataset.

  ~/env_isaaclab6/bin/python scripts/collect_dexgen.py --out data/v1 --run_name nut_goal_style0_s0 \
      --checkpoint ~/IsaacLab/projects/dexgen/outputs/xhand_nut_goal/<run>/model_XXXX.pt \
      --num_envs 1024 --target_transitions 30000000 --stochastic

The env class / cfg / PPO cfg are given as "module:Name" so that modified or new dexgen envs can be used
without touching this script. Cfg attributes can be overridden with --cfg key=value (python literals), e.g.
--cfg dr_enabled=True wrist_randomization_deg=30.0. Without --checkpoint, --random_policy rolls out uniform
random actions (for testing the pipeline only).
"""
import argparse
import ast
import importlib
import importlib.metadata
import os
import sys
import time

from isaaclab.app import AppLauncher

HERE = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('--dexgen_root', default=os.path.expanduser('~/IsaacLab/projects/dexgen'))
parser.add_argument('--env_class', default='dexgen.nut_goal_env:NutGoalEnv')
parser.add_argument('--env_cfg', default='dexgen.nut_goal_env:NutGoalEnvCfg')
parser.add_argument('--agent_cfg', default='dexgen.ppo_cfg:NutSpinPpoCfg')
parser.add_argument('--object_name', default='nut', help='scene key of the manipulated rigid object')
parser.add_argument('--cfg', nargs='*', default=[], help='env cfg overrides key=value')
parser.add_argument('--checkpoint', default=None)
parser.add_argument('--random_policy', action='store_true')
parser.add_argument('--stochastic', action='store_true', help='sample actions from the PPO policy distribution')
parser.add_argument('--out', required=True)
parser.add_argument('--run_name', required=True)
parser.add_argument('--task_name', default=None, help='dataset task label (default: env class name)')
parser.add_argument('--policy_name', default=None, help='dataset policy label (default: derived from checkpoint)')
parser.add_argument('--num_envs', type=int, default=1024)
parser.add_argument('--target_transitions', type=float, default=1e6)
parser.add_argument('--shard_transitions', type=float, default=2e6)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--device', default='cuda:0')
args = parser.parse_args()
if args.checkpoint is None and not args.random_policy:
    parser.error('--checkpoint is required (or --random_policy for pipeline tests)')
app = AppLauncher(headless=True, device=args.device).app

import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, args.dexgen_root)
from dexdata.adapters import DexgenAdapter  # noqa: E402
from dexdata.recorder import EpisodeRecorder  # noqa: E402


def load(spec):
    module, name = spec.split(':')
    return getattr(importlib.import_module(module), name)


cfg = load(args.env_cfg)()
cfg.seed = args.seed
cfg.scene.num_envs = args.num_envs
cfg.sim.device = args.device
cfg_overrides = {}
for kv in args.cfg:
    k, v = kv.split('=', 1)
    try:
        v = ast.literal_eval(v)
    except (ValueError, SyntaxError):
        pass
    obj = cfg
    *path, leaf = k.split('.')
    for p in path:
        obj = getattr(obj, p)
    if not hasattr(obj, leaf):
        raise AttributeError(f'env cfg has no attribute {k}')
    setattr(obj, leaf, v)
    cfg_overrides[k] = v

agent_cfg = load(args.agent_cfg)()
agent_cfg.seed = args.seed
agent_cfg.device = args.device
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version('rsl-rl-lib'))
raw_env = load(args.env_class)(cfg)
env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)

if args.random_policy:
    def policy(obs):
        return 2.0 * torch.rand((args.num_envs, 12), device=args.device) - 1.0
    policy_name = args.policy_name or 'random'
else:
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
    runner.load(os.path.abspath(os.path.expanduser(args.checkpoint)), map_location=args.device)
    _policy = runner.get_inference_policy(device=args.device)

    def policy(obs):
        return _policy(obs, stochastic_output=args.stochastic)
    ckpt = os.path.abspath(os.path.expanduser(args.checkpoint))
    policy_name = args.policy_name or (os.path.basename(os.path.dirname(ckpt)) + '/' +
                                       os.path.splitext(os.path.basename(ckpt))[0] +
                                       ('_stochastic' if args.stochastic else ''))

task_name = args.task_name or args.env_class.split(':')[1]
adapter = DexgenAdapter(raw_env, env, object_name=args.object_name)
rec = EpisodeRecorder(
    args.out, args.run_name, task=task_name, policy=policy_name, num_envs=args.num_envs,
    max_episode_steps=raw_env.max_episode_length, step_dt=raw_env.step_dt, shard_transitions=args.shard_transitions,
    run_info={'collector': 'collect_dexgen', 'env_class': args.env_class, 'env_cfg': args.env_cfg,
              'cfg_overrides': {k: repr(v) for k, v in cfg_overrides.items()}, 'checkpoint': args.checkpoint,
              'stochastic': args.stochastic, 'random_policy': args.random_policy, 'seed': args.seed,
              'dexgen_root': args.dexgen_root})

obs = env.get_observations()
t0, last = time.time(), 0
status = 'complete'
try:
    with torch.inference_mode():
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
