"""Read-only access to a collected dataset (numpy only, memory-mapped).

    ds = TrajectoryDataset("~/IsaacLab/projects/dexdata/data/v1", tasks=["nutbolt_trinut"])
    print(ds.summary())
    for ep in ds.episodes():                       # dict of per-step arrays + episode metadata
        ...
    idx = ds.window_index(history=4, horizon=4)    # (M, 2) [shard, row] where a window fits in one episode
    w = ds.get_window(*idx[0], history=4, horizon=4, fields=["joint_pos", "keypoint_pos_wrist"])
"""

import json
import os

import numpy as np

from . import schema


class TrajectoryDataset:
    def __init__(self, root, tasks=None, policies=None, runs=None):
        self.root = os.path.abspath(os.path.expanduser(root))
        with open(os.path.join(self.root, "manifest.json")) as f:
            self.manifest = json.load(f)
        if self.manifest["schema_version"] != schema.SCHEMA_VERSION:
            raise ValueError(f"schema {self.manifest['schema_version']} != {schema.SCHEMA_VERSION}")
        task_ids = None if tasks is None else {self.manifest["tasks"].index(t) for t in tasks}
        policy_ids = None if policies is None else {self.manifest["policies"].index(p) for p in policies}
        self.shards = []
        for run in self.manifest["runs"]:
            if runs is not None and run["run_name"] not in runs:
                continue
            for sh in run["shards"]:
                path = os.path.join(self.root, "shards", sh["name"])
                eps = {k: np.load(os.path.join(path, f"ep_{k}.npy")) for k in schema.EPISODE_FIELDS}
                keep = np.ones(len(eps["length"]), dtype=bool)
                if task_ids is not None:
                    keep &= np.isin(eps["task_id"], list(task_ids))
                if policy_ids is not None:
                    keep &= np.isin(eps["policy_id"], list(policy_ids))
                if keep.any():
                    self.shards.append({"path": path, "run": run["run_name"],
                                        "episodes": {k: v[keep] for k, v in eps.items()}, "cache": {}})

    def field(self, shard, name):
        sh = self.shards[shard]
        if name not in sh["cache"]:
            sh["cache"][name] = np.load(os.path.join(sh["path"], f"{name}.npy"), mmap_mode="r")
        return sh["cache"][name]

    def episodes(self, fields=None):
        fields = fields or list(schema.STEP_FIELDS)
        for si, sh in enumerate(self.shards):
            eps = sh["episodes"]
            for i in range(len(eps["length"])):
                s, n = int(eps["start"][i]), int(eps["length"][i])
                ep = {k: v[i] for k, v in eps.items()}
                ep.update({f: np.asarray(self.field(si, f)[s:s + n]) for f in fields})
                yield ep

    def window_index(self, history, horizon, stride=1):
        """Rows t with t-history+1 >= episode start and t+horizon <= episode end (exclusive)."""
        out = []
        for si, sh in enumerate(self.shards):
            eps = sh["episodes"]
            for s, n in zip(eps["start"], eps["length"]):
                t = np.arange(int(s) + history - 1, int(s) + int(n) - horizon, stride)
                out.append(np.stack([np.full_like(t, si), t], axis=-1))
        return np.concatenate(out) if out else np.zeros((0, 2), dtype=np.int64)

    def get_window(self, shard, t, history, horizon, fields):
        """Returns {field: (history + horizon, ...)} covering rows t-history+1 ... t+horizon."""
        return {f: np.asarray(self.field(shard, f)[t - history + 1:t + horizon + 1]) for f in fields}

    def summary(self):
        tasks, policies = self.manifest["tasks"], self.manifest["policies"]
        reasons = {v: k for k, v in schema.END_REASONS.items()}
        rows = {}
        for sh in self.shards:
            eps = sh["episodes"]
            for tid, pid, n, r, succ in zip(eps["task_id"], eps["policy_id"], eps["length"], eps["end_reason"], eps["success"]):
                key = (tasks[tid], policies[pid])
                d = rows.setdefault(key, {"episodes": 0, "transitions": 0, "success": 0, "end_reasons": {}})
                d["episodes"] += 1
                d["transitions"] += int(n)
                d["success"] += int(succ)
                d["end_reasons"][reasons[int(r)]] = d["end_reasons"].get(reasons[int(r)], 0) + 1
        return {f"{t} / {p}": v for (t, p), v in rows.items()}
