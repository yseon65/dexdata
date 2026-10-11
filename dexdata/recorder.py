"""Episode-wise recorder that turns vectorized rollouts into filtered, contiguous episode shards.

Usage (inside a collector):

    rec = EpisodeRecorder(root, run_name, task="nutbolt_trinut", policy="student_padapt",
                          num_envs=N, max_episode_steps=800, step_dt=0.05, run_info={...})
    while rec.kept_transitions < target:
        state = adapter.read_state()          # dict of (N, ...) tensors observed before the action
        ...step the env...
        rec.add({**state, **post}, done, reason, adapter.episode_params)
    rec.close()

Layout on disk:

    root/manifest.json
    root/shards/<run_name>_<k>/<step_field>.npy      (T, *shape)   one row per policy step
    root/shards/<run_name>_<k>/ep_<episode_field>.npy (E, *shape)  one entry per stored episode
"""

import datetime
import json
import os

import numpy as np

from . import schema


class EpisodeRecorder:
    def __init__(self, root, run_name, task, policy, num_envs, max_episode_steps, step_dt, run_info=None,
                 shard_transitions=2_000_000, drop_tail_reasons=schema.DEFAULT_DROP_TAIL_REASONS,
                 drop_tail_seconds=schema.DEFAULT_DROP_TAIL_SECONDS, skip_first_steps=1, min_length=8):
        self.root = os.path.abspath(root)
        self.run_name = run_name
        self.num_envs = num_envs
        self.max_len = max_episode_steps + 2
        self.step_dt = float(step_dt)
        self.shard_transitions = int(shard_transitions)
        self.drop_tail_codes = {schema.END_REASONS[r] for r in drop_tail_reasons}
        self.drop_tail_steps = int(round(drop_tail_seconds / self.step_dt))
        self.skip_first = int(skip_first_steps)
        self.min_length = int(min_length)

        self.names = list(schema.STEP_FIELDS)
        self.slices, off = {}, 0
        for name in self.names:
            n = schema.field_numel(schema.STEP_FIELDS[name][0])
            self.slices[name] = slice(off, off + n)
            off += n
        self.width = off
        self.buf = np.zeros((num_envs, self.max_len, self.width), dtype=np.float32)
        self.ptr = np.zeros(num_envs, dtype=np.int64)
        self.raw_steps = np.zeros(num_envs, dtype=np.int64)
        self.ep_params = {k: np.zeros((num_envs, *schema.EPISODE_FIELDS[k][0]), dtype=schema.EPISODE_FIELDS[k][1])
                          for k in self._param_names()}

        self._staged_rows, self._staged_eps = [], []
        self._staged_count = 0
        self.shard_index = 0
        self.kept_transitions = 0
        self.kept_episodes = 0
        self.discarded_episodes = 0
        self.dropped_tail_transitions = 0
        self.raw_transitions = 0
        self.reason_counts = {k: 0 for k in schema.END_REASONS}
        self._next_episode_id = 0

        os.makedirs(os.path.join(self.root, "shards"), exist_ok=True)
        manifest = self._load_manifest()
        if any(r["run_name"] == run_name for r in manifest["runs"]):
            raise ValueError(f"run '{run_name}' already exists in {self.root}/manifest.json")
        self.task_id = self._register(manifest["tasks"], task)
        self.policy_id = self._register(manifest["policies"], policy)
        self.run = {
            "run_name": run_name, "task": task, "policy": policy, "created": datetime.datetime.now().isoformat(),
            "step_dt": self.step_dt, "num_envs": num_envs,
            "filter": {"skip_first_steps": self.skip_first, "drop_tail_reasons": list(drop_tail_reasons),
                       "drop_tail_seconds": drop_tail_seconds, "min_length": self.min_length},
            "info": run_info or {}, "shards": [], "status": "running",
        }
        manifest["runs"].append(self.run)
        self._save_manifest(manifest)

    # ------------------------------------------------------------------ manifest
    @staticmethod
    def _param_names():
        return [k for k in schema.EPISODE_FIELDS if k not in
                ("episode_id", "env_id", "task_id", "policy_id", "start", "length", "raw_length", "end_reason", "success")]

    @staticmethod
    def _register(names, name):
        if name not in names:
            names.append(name)
        return names.index(name)

    def _manifest_path(self):
        return os.path.join(self.root, "manifest.json")

    def _load_manifest(self):
        path = self._manifest_path()
        if os.path.exists(path):
            with open(path) as f:
                m = json.load(f)
            if m["schema_version"] != schema.SCHEMA_VERSION:
                raise ValueError(f"schema mismatch: dataset {m['schema_version']} vs code {schema.SCHEMA_VERSION}")
            return m
        return {
            "schema_version": schema.SCHEMA_VERSION,
            "conventions": schema.__doc__,
            "joint_names": list(schema.JOINT_NAMES),
            "wrist_body": schema.WRIST_BODY,
            "keypoint_bodies": list(schema.KEYPOINT_BODIES),
            "keypoint_sets": schema.KEYPOINT_SETS,
            "finger_order": list(schema.FINGER_ORDER),
            "end_reasons": schema.END_REASONS,
            "step_fields": {k: {"shape": list(v[0]), "dtype": v[1], "desc": v[2]} for k, v in schema.STEP_FIELDS.items()},
            "episode_fields": {k: {"shape": list(v[0]), "dtype": v[1], "desc": v[2]} for k, v in schema.EPISODE_FIELDS.items()},
            "tasks": [], "policies": [], "runs": [],
        }

    def _save_manifest(self, manifest):
        tmp = self._manifest_path() + ".tmp"
        with open(tmp, "w") as f:
            json.dump(manifest, f, indent=1)
        os.replace(tmp, self._manifest_path())

    def _update_run(self, **kw):
        manifest = self._load_manifest()
        for r in manifest["runs"]:
            if r["run_name"] == self.run_name:
                r.update(self.run)
                r.update(kw)
        self._save_manifest(manifest)

    # ------------------------------------------------------------------ recording
    def add(self, step, done, reason, params_fn):
        """step: dict name -> (N, ...) tensor/array; done: (N,) bool; reason: (N,) int END_REASONS code."""
        flat = np.empty((self.num_envs, self.width), dtype=np.float32)
        for name in self.names:
            v = step[name]
            v = v.detach().float().cpu().numpy() if hasattr(v, "detach") else np.asarray(v, dtype=np.float32)
            flat[:, self.slices[name]] = v.reshape(self.num_envs, -1)
        done = _np(done).astype(bool).reshape(-1)
        reason = _np(reason).astype(np.int64).reshape(-1)

        self.raw_steps += 1
        self.raw_transitions += self.num_envs
        rec = np.nonzero(self.raw_steps > self.skip_first)[0]
        if len(rec):
            new = rec[self.ptr[rec] == 0]
            if len(new):
                params = params_fn(new)
                for k in self.ep_params:
                    self.ep_params[k][new] = params[k]
            full = rec[self.ptr[rec] >= self.max_len]
            for e in full:  # safety net; should not trigger with a correct max_episode_steps
                self._finalize(e, schema.END_REASONS["truncated"])
            rec = rec[self.ptr[rec] < self.max_len]
            self.buf[rec, self.ptr[rec]] = flat[rec]
            self.ptr[rec] += 1
        for e in np.nonzero(done)[0]:
            self._finalize(e, int(reason[e]))
        if self._staged_count >= self.shard_transitions:
            self._flush()

    def _finalize(self, e, reason_code):
        n = int(self.ptr[e])
        raw = int(self.raw_steps[e])
        self.ptr[e] = 0
        self.raw_steps[e] = 0
        name = next(k for k, v in schema.END_REASONS.items() if v == reason_code)
        self.reason_counts[name] += 1
        if reason_code in self.drop_tail_codes:
            cut = min(n, self.drop_tail_steps)
            self.dropped_tail_transitions += cut
            n -= cut
        if n < self.min_length:
            self.discarded_episodes += 1
            return
        rows = self.buf[e, :n].copy()
        succ = rows[:, self.slices["success_event"]].max() > 0.5
        ep = {"episode_id": self._next_episode_id, "env_id": e, "task_id": self.task_id, "policy_id": self.policy_id,
              "length": n, "raw_length": raw, "end_reason": reason_code, "success": succ}
        ep.update({k: self.ep_params[k][e].copy() for k in self.ep_params})
        self._next_episode_id += 1
        self._staged_rows.append(rows)
        self._staged_eps.append(ep)
        self._staged_count += n
        self.kept_transitions += n
        self.kept_episodes += 1

    def _flush(self):
        if not self._staged_rows:
            return
        shard = f"{self.run_name}_{self.shard_index:05d}"
        path = os.path.join(self.root, "shards", shard)
        os.makedirs(path, exist_ok=True)
        rows = np.concatenate(self._staged_rows, axis=0)
        for name in self.names:
            shape, dtype, _ = schema.STEP_FIELDS[name]
            v = rows[:, self.slices[name]].reshape(len(rows), *shape)
            v = (v > 0.5) if dtype == "bool" else v.astype(dtype)
            np.save(os.path.join(path, f"{name}.npy"), v)
        starts = np.cumsum([0] + [ep["length"] for ep in self._staged_eps[:-1]])
        for i, ep in enumerate(self._staged_eps):
            ep["start"] = int(starts[i])
        for k, (shape, dtype, _) in schema.EPISODE_FIELDS.items():
            np.save(os.path.join(path, f"ep_{k}.npy"), np.asarray([ep[k] for ep in self._staged_eps], dtype=dtype))
        self.run["shards"].append({"name": shard, "transitions": int(len(rows)), "episodes": len(self._staged_eps)})
        self.shard_index += 1
        self._staged_rows, self._staged_eps, self._staged_count = [], [], 0
        self._update_run(**self.stats())

    def stats(self):
        return {"transitions": int(self.kept_transitions), "episodes": int(self.kept_episodes),
                "discarded_episodes": int(self.discarded_episodes),
                "dropped_tail_transitions": int(self.dropped_tail_transitions),
                "raw_transitions": int(self.raw_transitions), "end_reasons": dict(self.reason_counts)}

    def close(self, status="complete"):
        self._flush()
        self.run["status"] = status
        self._update_run(**self.stats())


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
