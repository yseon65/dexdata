"""Measure the reachable in-hand twist range of a dexgen policy (for choosing goal bounds, theta_max).

Per episode, the nut twist about its own body z axis is accumulated from the episode start:
    dq_t = conj(q_{t-1}) * q_t (body frame),  twist_t += 2 * atan2(dq_t.z, dq_t.w)
Reported per episode:
    max_twist    largest forward excursion   (max over t of the cumulative twist)
    min_twist    largest reverse excursion   (min over t, <= 0)
    first_peak   cumulative twist at the first reversal, i.e. when the twist drops more than
                 --reversal_drop below its running maximum (NaN if it never reverses)

  ~/env_isaaclab6/bin/python scripts/measure_twist_range.py --checkpoint <model_X.pt> --num_envs 64 --steps 1600 \
      [--cfg dr_enabled=False]
"""
import argparse
import ast
import importlib
import importlib.metadata
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--dexgen_root', default=os.path.expanduser('~/IsaacLab/projects/dexgen'))
parser.add_argument('--env_class', default='dexgen.nut_goal_env:NutGoalEnv')
parser.add_argument('--env_cfg', default='dexgen.nut_goal_env:NutGoalEnvCfg')
parser.add_argument('--agent_cfg', default='dexgen.ppo_cfg:NutSpinPpoCfg')
parser.add_argument('--checkpoint', nargs='+', required=True)
parser.add_argument('--cfg', nargs='*', default=[])
parser.add_argument('--num_envs', type=int, default=64)
parser.add_argument('--steps', type=int, default=1600)
parser.add_argument('--stochastic', action='store_true')
parser.add_argument('--reversal_drop', type=float, default=0.3, help='rad below running max that counts as reversal')
parser.add_argument('--min_length', type=int, default=100, help='ignore episodes shorter than this (policy steps)')
parser.add_argument('--device', default='cuda:0')
args = parser.parse_args()
app = AppLauncher(headless=True, device=args.device).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

sys.path.insert(0, args.dexgen_root)


def load(spec):
    module, name = spec.split(':')
    return getattr(importlib.import_module(module), name)


cfg = load(args.env_cfg)()
cfg.scene.num_envs = args.num_envs
cfg.sim.device = args.device
for kv in args.cfg:
    k, v = kv.split('=', 1)
    try:
        v = ast.literal_eval(v)
    except (ValueError, SyntaxError):
        pass
    assert hasattr(cfg, k), k
    setattr(cfg, k, v)
agent_cfg = handle_deprecated_rsl_rl_cfg(load(args.agent_cfg)(), importlib.metadata.version('rsl-rl-lib'))
agent_cfg.device = args.device
raw = load(args.env_class)(cfg)
env = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
nut = raw.scene['nut']
N = args.num_envs


def twist_delta(q_prev, q):  # wxyz, body-frame twist about z
    w1, x1, y1, z1 = q_prev.unbind(-1)
    w2, x2, y2, z2 = q.unbind(-1)
    # conj(q_prev) * q
    w = w1 * w2 + x1 * x2 + y1 * y2 + z1 * z2
    z = w1 * z2 - x1 * y2 + y1 * x2 - z1 * w2
    s = torch.where(w < 0, -1.0, 1.0)
    return 2.0 * torch.atan2(z * s, w * s)


def pct(v, ps=(50, 80, 95)):
    v = np.asarray([x for x in v if np.isfinite(x)])
    return 'n/a' if len(v) == 0 else ' '.join(f'p{p}={np.percentile(v, p):.2f}' for p in ps) + f' max={v.max():.2f}'


def measure(ckpt):
        runner.load(os.path.abspath(os.path.expanduser(ckpt)), map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)
        obs, _ = env.reset()
        q_prev = nut.data.root_link_quat_w.clone()
        twist = torch.zeros(N, device=args.device)
        tmax = torch.zeros(N, device=args.device)
        tmin = torch.zeros(N, device=args.device)
        first_peak = torch.full((N,), float('nan'), device=args.device)
        first_trough = torch.full((N,), float('nan'), device=args.device)
        tilt_prev = torch.zeros(N, device=args.device)
        dir_prev = torch.ones(N, device=args.device)
        tilt_sum = {1: 0.0, -1: 0.0}
        tilt_cnt = {1: 0, -1: 0}
        tilt_hi = {1: 0, -1: 0}
        ends = {1: {}, -1: {}}
        length = torch.zeros(N, device=args.device)
        rec = {'max': [], 'min': [], 'first_peak': [], 'first_trough': [], 'length': [], 'reversed': [], 'end_tilt': []}
        with torch.inference_mode():
            for _ in range(args.steps):
                obs, _, done, info = env.step(policy(obs, stochastic_output=args.stochastic))
                tilt = getattr(raw, 'last_tilt', torch.zeros(N, device=args.device))
                gdir = info.get('task/goal_dir', torch.ones(N, device=args.device))
                treason = info.get('task/terminal_reason', torch.zeros(N, dtype=torch.long, device=args.device))
                q = nut.data.root_link_quat_w
                d = done.bool()
                live = ~d
                twist = torch.where(live, twist + twist_delta(q_prev, q), twist)
                tmax = torch.maximum(tmax, twist)
                tmin = torch.minimum(tmin, twist)
                rev = live & torch.isnan(first_peak) & (twist < tmax - args.reversal_drop)
                first_peak = torch.where(rev, tmax, first_peak)
                rrev = live & torch.isnan(first_trough) & (twist > tmin + args.reversal_drop)
                first_trough = torch.where(rrev, tmin, first_trough)
                for sgn in (1, -1):
                    m = live & (gdir * sgn > 0)
                    tilt_sum[sgn] += tilt[m].sum().item(); tilt_cnt[sgn] += int(m.sum()); tilt_hi[sgn] += int((tilt[m] > 0.785).sum())
                length += 1
                for e in d.nonzero().squeeze(-1).tolist():
                    sgn = 1 if dir_prev[e] > 0 else -1
                    r = int(treason[e])
                    ends[sgn][r] = ends[sgn].get(r, 0) + 1
                    rec['end_tilt'].append(tilt_prev[e].item())
                    if length[e] >= args.min_length:
                        rec['first_trough'].append(first_trough[e].item())
                        rec['max'].append(tmax[e].item()); rec['min'].append(tmin[e].item())
                        rec['first_peak'].append(first_peak[e].item()); rec['length'].append(length[e].item())
                        rec['reversed'].append(not np.isnan(first_peak[e].item()))
                twist[d] = 0.0; tmax[d] = 0.0; tmin[d] = 0.0; first_peak[d] = float('nan'); length[d] = 0
                first_trough[d] = float('nan')
                tilt_prev = tilt.clone(); dir_prev = gdir.float().clone()
                q_prev = q.clone()
        n = len(rec['max'])
        print(f"[twist] {os.path.basename(os.path.dirname(ckpt))}/{os.path.basename(ckpt)} cfg={args.cfg} "
              f"stochastic={args.stochastic}: episodes {n} (len >= {args.min_length}), "
              f"mean length {np.mean(rec['length']) if n else float('nan'):.0f}, reversed {np.mean(rec['reversed']) if n else 0:.0%}", flush=True)
        print(f"[twist]   max_twist  (rad) {pct(rec['max'])}  | deg p80={np.degrees(np.percentile(rec['max'], 80)) if n else float('nan'):.0f}", flush=True)
        print(f"[twist]   first_peak (rad) {pct(rec['first_peak'])}", flush=True)
        print(f"[twist]   min_twist  (rad) {pct([-x for x in rec['min']])} (magnitude of reverse excursion)", flush=True)
        print(f"[twist]   first_trough (rad, magnitude) {pct([-x for x in rec['first_trough']])}", flush=True)
        for sgn, name in ((1, 'forward'), (-1, 'reverse')):
            c = max(tilt_cnt[sgn], 1)
            print(f"[twist]   {name:7s} goals: mean tilt {np.degrees(tilt_sum[sgn] / c):.1f} deg, tilt>45deg {tilt_hi[sgn] / c:.2%} of steps, "
                  f"episode ends while this goal active (terminal_reason: count) {ends[sgn]}", flush=True)


exit_code = 0
try:
    for ckpt in args.checkpoint:
        measure(ckpt)
except BaseException:
    import traceback
    traceback.print_exc()
    exit_code = 1
finally:
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)

