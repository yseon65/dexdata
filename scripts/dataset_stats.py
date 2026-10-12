"""Dataset statistics and figures (CPU only, numpy + matplotlib).

  ~/env_isaaclab6/bin/python scripts/dataset_stats.py data/v1 [--out docs/figures/dataset_stats] [--rows 1500000]

Episode-level statistics use every episode. Step-level statistics use whole episodes sampled at random
(about ``--rows`` rows per policy), so within-episode differences (speed, 4-step displacement) are exact.
Writes PNG figures and ``stats.json`` into ``--out``.
"""

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dexdata import schema  # noqa: E402
from dexdata.dataset import TrajectoryDataset  # noqa: E402

COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3, fixed per policy order in the manifest
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DPI = 110
HORIZON = 4  # 0.2 s at 20 Hz (DexGen motion target)
TIPS = schema.KEYPOINT_SETS["tips"]
SHORT_J = [n.replace("left_hand_", "").replace("_joint", "") for n in schema.JOINT_NAMES]
ROW_FIELDS = ["joint_pos", "joint_vel", "control_error", "joint_target", "applied_target", "object_angvel_wrist",
              "object_dof_pos", "object_dof_vel", "object_contact_force", "finger_contact_force", "goal_reached_event"]

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "legend.frameon": False, "lines.linewidth": 1.6,
})


def joint_limits(urdf):
    if not urdf or not os.path.exists(urdf):
        return None
    lim = {}
    for j in ET.parse(urdf).getroot().iter("joint"):
        el = j.find("limit")
        if el is not None and el.get("lower") is not None:
            lim[j.get("name")] = (float(el.get("lower")), float(el.get("upper")))
    return [lim.get(n, (np.nan, np.nan)) for n in schema.JOINT_NAMES]


def episode_table(ds, pid):
    out = {}
    for si, sh in enumerate(ds.shards):
        e = sh["episodes"]
        m = e["policy_id"] == pid
        if not m.any():
            continue
        for k, v in e.items():
            out.setdefault(k, []).append(v[m])
        out.setdefault("shard", []).append(np.full(int(m.sum()), si))
    return {k: np.concatenate(v) for k, v in out.items()}


def sample_rows(ds, eps, target_rows, rng):
    """Random whole episodes until ~target_rows rows; returns dict of per-row arrays + episode index per row."""
    order = rng.permutation(len(eps["length"]))
    csum = np.cumsum(eps["length"][order])
    sel = np.sort(order[:int(np.searchsorted(csum, target_rows)) + 1])
    data = {f: [] for f in ROW_FIELDS + ["tips", "ep"]}
    for si in np.unique(eps["shard"][sel]):
        es = sel[eps["shard"][sel] == si]
        idx = np.concatenate([np.arange(eps["start"][i], eps["start"][i] + eps["length"][i]) for i in es])
        data["ep"].append(np.repeat(es, eps["length"][es]))
        for f in ROW_FIELDS:
            data[f].append(np.asarray(ds.field(si, f)[idx]))
        data["tips"].append(np.asarray(ds.field(si, "keypoint_pos_wrist")[idx])[:, TIPS])
    return {k: np.concatenate(v) for k, v in data.items()}, sel


def offset_mask(ep, k):
    """True for rows t whose row t+k is in the same episode."""
    m = np.zeros(len(ep), dtype=bool)
    m[:-k] = ep[k:] == ep[:-k]
    return m


def pct_stats(x, label=""):
    x = x[np.isfinite(x)]
    p = np.percentile(x, [1, 25, 50, 75, 99])
    return {"label": label, "whislo": p[0], "q1": p[1], "med": p[2], "q3": p[3], "whishi": p[4], "fliers": []}


def grouped_box(ax, per_policy, names, ylabel, limits=None):
    """per_policy: list of (N, J) arrays. Boxes = 25-75 %, whiskers = 1-99 %."""
    nj, npol = per_policy[0].shape[1], len(per_policy)
    w = 0.8 / npol
    if limits is not None:
        for j, (lo, hi) in enumerate(limits):
            ax.plot([j - 0.45, j + 0.45], [lo, lo], color=INK2, lw=1.0, ls="--")
            ax.plot([j - 0.45, j + 0.45], [hi, hi], color=INK2, lw=1.0, ls="--")
    for p, arr in enumerate(per_policy):
        stats = [pct_stats(arr[:, j]) for j in range(nj)]
        pos = np.arange(nj) - 0.4 + w * (p + 0.5)
        ax.bxp(stats, positions=pos, widths=w * 0.85, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor=COLORS[p], edgecolor=COLORS[p], alpha=0.55),
               medianprops=dict(color=INK, lw=1.2), whiskerprops=dict(color=COLORS[p]),
               capprops=dict(color=COLORS[p]))
    ax.set_xticks(np.arange(nj))
    ax.set_xticklabels(SHORT_J, rotation=35, ha="right")
    ax.set_xlim(-0.6, nj - 0.4)
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)


def legend_handles(names):
    return [plt.Line2D([], [], color=c, lw=6, alpha=0.7) for c in COLORS[:len(names)]]


def hist_overlay(ax, vals, names, bins, xlabel, density=True, logy=False, weights=None):
    for p, v in enumerate(vals):
        if v is None:
            continue
        v = np.asarray(v, dtype=np.float64)
        w = None if weights is None else weights[p]
        fin = np.isfinite(v)
        if not fin.any():
            continue
        ax.hist(v[fin], bins=bins, density=density, histtype="step", color=COLORS[p], lw=1.6, label=names[p],
                weights=None if w is None else w[fin])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density" if density else "count")
    if logy:
        ax.set_yscale("log")


def save(fig, out, name, figs):
    path = os.path.join(out, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    figs.append(name)
    print("wrote", path)


def q(x, ps=(5, 50, 95)):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return {f"p{p}": float(np.percentile(x, p)) for p in ps} | {"mean": float(x.mean())} if len(x) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "figures",
                                                  "dataset_stats"))
    ap.add_argument("--rows", type=int, default=1_500_000, help="sampled rows per policy for step-level stats")
    ap.add_argument("--urdf", default=os.path.expanduser(
        "~/IsaacLab/projects/dexscrew/assets/xhand_left/urdf/xhand_left.urdf"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    ds = TrajectoryDataset(a.root)
    names = ds.manifest["policies"]
    dt = {r["policy"]: r["step_dt"] for r in ds.manifest["runs"]}
    tasks = {r["policy"]: r["task"] for r in ds.manifest["runs"]}
    reasons = sorted(schema.END_REASONS, key=schema.END_REASONS.get)
    limits = joint_limits(a.urdf)

    E, S = [], []
    stats = {"root": os.path.abspath(a.root), "sampled_rows_target": a.rows, "seed": a.seed, "policies": {}}
    for pid, name in enumerate(names):
        eps = episode_table(ds, pid)
        d, sel = sample_rows(ds, eps, a.rows, rng)
        E.append(eps)
        S.append(d)
        print(f"{name}: {len(eps['length'])} episodes, sampled {len(sel)} episodes / {len(d['ep'])} rows")

    # ---------------- derived step-level quantities
    for p, d in enumerate(S):
        h = dt[names[p]]
        m1 = offset_mask(d["ep"], 1)
        mH = offset_mask(d["ep"], HORIZON)
        tips = d["tips"]
        d["tip_speed"] = (np.linalg.norm(tips[1:] - tips[:-1], axis=-1) / h)[m1[:-1]]           # (M, 5)
        d["tip_disp"] = np.linalg.norm(tips[HORIZON:] - tips[:-HORIZON], axis=-1)[mH[:-HORIZON]]  # (M, 5) [m]
        d["dtarget"] = d["applied_target"] - d["joint_target"]

    # ---------------- summary + stats.json
    for p, name in enumerate(names):
        eps, d = E[p], S[p]
        n_ep, rows = len(eps["length"]), int(eps["length"].sum())
        er = np.bincount(eps["end_reason"], minlength=len(reasons))
        ep_rows = np.bincount(np.searchsorted(np.unique(d["ep"]), d["ep"]))
        gr = np.bincount(np.searchsorted(np.unique(d["ep"]), d["ep"]), weights=d["goal_reached_event"][:, 0])
        s = {
            "task": tasks[name], "step_dt": dt[name], "rows": rows, "episodes": n_ep,
            "raw_rows": int(eps["raw_length"].sum()), "kept_fraction": rows / int(eps["raw_length"].sum()),
            "length_steps": q(eps["length"]), "length_seconds_mean": float(eps["length"].mean() * dt[name]),
            "end_reason_episode_pct": {r: 100 * float(er[i]) / n_ep for i, r in enumerate(reasons) if er[i]},
            "end_reason_row_pct": {r: 100 * float(eps["length"][eps["end_reason"] == i].sum()) / rows
                                   for i, r in enumerate(reasons) if er[i]},
            "success_rate": float(eps["success"].mean()),
            "object_scale": q(eps["object_scale"]), "object_mass": q(eps["object_mass"]),
            "friction": q(eps["friction"]), "restitution": q(eps["restitution"]),
            "p_gain": q(eps["p_gain"]), "d_gain": q(eps["d_gain"]),
            "sampled_rows": int(len(d["ep"])), "sampled_episodes": int(len(ep_rows)),
            "fingertip_speed_mps": q(d["tip_speed"]),
            "fingertip_disp_0p2s_mm": q(1000 * d["tip_disp"]),
            "joint_vel_abs": q(np.abs(d["joint_vel"])),
            "control_error_abs": q(np.abs(d["control_error"])),
            "delta_target_abs": q(np.abs(d["dtarget"])),
            "delta_target_max_abs": float(np.abs(d["dtarget"]).max()),
            "delta_target_saturated_pct": 100 * float((np.abs(d["dtarget"]) > 0.99 * np.abs(d["dtarget"]).max()).mean()),
            "fingertip_still_pct": {f: 100 * float((d["tip_speed"][:, i] < 0.005).mean())
                                    for i, f in enumerate(schema.FINGER_ORDER)},
            "object_angvel_norm": q(np.linalg.norm(d["object_angvel_wrist"], axis=-1)),
            "object_angvel_z": q(d["object_angvel_wrist"][:, 2]),
            "object_contact_force": q(d["object_contact_force"][:, 0]),
        }
        if np.isfinite(d["finger_contact_force"]).any():
            fcf = d["finger_contact_force"]
            s["finger_contact_force"] = {f: q(fcf[:, i]) for i, f in enumerate(schema.FINGER_ORDER)}
            s["finger_in_contact_pct"] = {f: 100 * float((fcf[:, i] > 0.05).mean()) for i, f in
                                          enumerate(schema.FINGER_ORDER)}
        if d["goal_reached_event"].any():
            s["goals_per_episode"] = q(gr)
            s["goals_per_minute"] = float(gr.sum() / (ep_rows.sum() * dt[name] / 60))
        if np.isfinite(d["object_dof_pos"]).any():
            # Note: the last stored row of every episode has object_dof_pos == 0 (reset), so use row n-2.
            ep_u, first = np.unique(d["ep"], return_index=True)
            last2 = np.append(first[1:], len(d["ep"])) - 2
            rot = d["object_dof_pos"][last2, 0] - d["object_dof_pos"][first, 0]
            d["nut_rot"] = rot
            d["nut_rate"] = rot / np.maximum(ep_rows - 2, 1) / dt[name]
            s["nut_rotation_per_episode_rad"] = q(rot)
            s["nut_rotation_rate_radps"] = q(d["nut_rate"])
            s["object_dof_vel"] = q(d["object_dof_vel"][:, 0])
            lastrow = np.append(first[1:], len(d["ep"])) - 1
            s["last_row_dof_pos_zero_pct"] = 100 * float((d["object_dof_pos"][lastrow, 0] == 0).mean())
        stats["policies"][name] = s

    # ---------------- figures
    figs = []
    lab = names

    # 01 episode length
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    mx = max(int(e["length"].max()) for e in E)
    bins = np.arange(0, mx + 20, 20)
    hist_overlay(ax[0], [e["length"] for e in E], lab, bins, "episode length [steps] (20 Hz)")
    ax[0].set_title("Episode length (per episode)")
    ax[0].legend()
    hist_overlay(ax[1], [e["length"] for e in E], lab, bins, "episode length [steps]", density=True,
                 weights=[e["length"].astype(float) for e in E])
    ax[1].set_title("Episode length (row-weighted)")
    ax[1].set_ylabel("row density")
    save(fig, out, "fig01_episode_length.png", figs)

    # 02 end reasons (episode % and row %)
    used = [r for r in reasons if any(r in stats["policies"][n]["end_reason_episode_pct"] for n in names)]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    for k, key in enumerate(["end_reason_episode_pct", "end_reason_row_pct"]):
        w = 0.8 / len(names)
        for p, n in enumerate(names):
            v = [stats["policies"][n][key].get(r, 0.0) for r in used]
            x = np.arange(len(used)) - 0.4 + w * (p + 0.5)
            bars = ax[k].bar(x, v, width=w * 0.92, color=COLORS[p], label=n)
            for b, val in zip(bars, v):
                if val > 0:
                    ax[k].text(b.get_x() + b.get_width() / 2, val + 1, f"{val:.0f}", ha="center", fontsize=7,
                               color=INK2)
        ax[k].set_xticks(np.arange(len(used)))
        ax[k].set_xticklabels(used)
        ax[k].set_ylim(0, 105)
        ax[k].grid(axis="x", visible=False)
        ax[k].set_ylabel("%")
    ax[0].set_title("End reason (% of episodes)")
    ax[1].set_title("End reason (% of rows)")
    ax[0].legend()
    save(fig, out, "fig02_end_reason.png", figs)

    # 03 domain randomization
    fig, axs = plt.subplots(2, 3, figsize=(11, 5.6))
    dr = [("object_scale", "object scale", np.linspace(0.65, 1.35, 57)),
          ("object_mass", "object mass [kg]", 50),
          ("friction", "friction", 50),
          ("restitution", "restitution", 40),
          ("p_gain", "P gain (all joints)", 50),
          ("d_gain", "D gain (all joints)", 50)]
    for axx, (k, xl, b) in zip(axs.ravel(), dr):
        allv = np.concatenate([e[k].ravel() for e in E])
        allv = allv[np.isfinite(allv)]
        lo_, hi_ = allv.min(), allv.max()
        if hi_ - lo_ < 1e-6:
            lo_, hi_ = lo_ - 0.05, hi_ + 0.05
        bb = b if not isinstance(b, int) else np.linspace(lo_, hi_, b)
        hist_overlay(axx, [e[k].ravel() for e in E], lab, bb, xl)
    axs[0, 0].legend(fontsize=7)
    fig.suptitle("Domain randomization coverage (per episode)", y=1.0)
    fig.tight_layout()
    save(fig, out, "fig03_domain_randomization.png", figs)

    # 04 outcome vs scale
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    sb = np.linspace(0.7, 1.3, 13)
    for p, e in enumerate(E):
        uq = np.unique(e["object_scale"])
        if len(uq) <= 12:  # discrete scale set (dexscrew): one point per scale
            c, ctr = np.searchsorted(uq, e["object_scale"]), uq
        else:
            c = np.clip(np.digitize(e["object_scale"], sb) - 1, 0, len(sb) - 2)
            ctr = 0.5 * (sb[1:] + sb[:-1])
        cnt = np.bincount(c, minlength=len(ctr))
        ok = cnt > 50
        timeout = np.bincount(c, weights=(e["end_reason"] == 0), minlength=len(ctr))
        ln = np.bincount(c, weights=e["length"], minlength=len(ctr))
        stats["policies"][names[p]]["by_scale"] = [
            {"scale": float(ctr[i]), "episodes": int(cnt[i]), "timeout_pct": float(100 * timeout[i] / cnt[i]),
             "mean_length_s": float(ln[i] / cnt[i] * dt[names[p]]), "row_pct": float(100 * ln[i] / e["length"].sum())}
            for i in range(len(ctr)) if ok[i]]
        ax[0].plot(ctr[ok], 100 * timeout[ok] / cnt[ok], "o-", color=COLORS[p], ms=4, label=names[p])
        ax[1].plot(ctr[ok], ln[ok] / cnt[ok] * dt[names[p]], "o-", color=COLORS[p], ms=4)
    ax[0].set_xlabel("object scale")
    ax[0].set_ylabel("% episodes ending in timeout")
    ax[0].set_title("Timeout (survival) rate vs scale")
    ax[0].legend(fontsize=7)
    ax[1].set_xlabel("object scale")
    ax[1].set_ylabel("mean stored length [s]")
    ax[1].set_title("Mean episode length vs scale")
    save(fig, out, "fig04_outcome_vs_scale.png", figs)

    # 05 joint positions vs limits
    fig, ax = plt.subplots(figsize=(11, 3.8))
    grouped_box(ax, [d["joint_pos"] for d in S], lab, "joint_pos [rad]", limits)
    ax.legend(legend_handles(lab) + [plt.Line2D([], [], color=INK2, ls="--")], lab + ["URDF limit"], ncol=4,
              loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=8)
    ax.set_title("Joint position distribution (box 25-75 %, whisker 1-99 %)")
    save(fig, out, "fig05_joint_pos.png", figs)
    if limits is not None:
        lo, hi = np.array(limits).T
        for p, n in enumerate(names):
            jp = S[p]["joint_pos"]
            rngj = np.percentile(jp, 99, axis=0) - np.percentile(jp, 1, axis=0)
            stats["policies"][n]["joint_range_1_99_frac_of_limit"] = dict(zip(SHORT_J, (rngj / (hi - lo)).round(3).tolist()))

    # 06 joint vel / control error / delta target
    fig, axs = plt.subplots(3, 1, figsize=(11, 10))
    grouped_box(axs[0], [d["joint_vel"] for d in S], lab, "joint_vel [rad/s]")
    axs[0].set_title("Joint velocity")
    grouped_box(axs[1], [d["control_error"] for d in S], lab, "joint_target - joint_pos [rad]")
    axs[1].set_title("Control error")
    grouped_box(axs[2], [d["dtarget"] for d in S], lab, "applied_target - joint_target [rad]")
    axs[2].set_title("Per-step target change (DexGen inverse-dynamics target)")
    axs[0].legend(legend_handles(lab), lab, ncol=3, loc="upper right", fontsize=8)
    fig.tight_layout()
    save(fig, out, "fig06_joint_dynamics.png", figs)

    # 07 fingertip workspace (wrist frame), 5 tips pooled
    allt = np.concatenate([d["tips"].reshape(-1, 3) for d in S])
    lo3, hi3 = np.percentile(allt, 0.05, axis=0), np.percentile(allt, 99.95, axis=0)
    fig, axs = plt.subplots(len(S), 2, figsize=(9, 3.4 * len(S)))
    for p, d in enumerate(S):
        t = d["tips"].reshape(-1, 3) * 1000
        for c, (i, j) in enumerate([(0, 1), (0, 2)]):
            axx = axs[p, c]
            axx.hist2d(t[:, i], t[:, j], bins=160, range=[[lo3[i] * 1000, hi3[i] * 1000], [lo3[j] * 1000, hi3[j] * 1000]],
                       cmap="Blues", norm=LogNorm(), density=True)
            med = np.median(d["tips"], axis=0) * 1000
            for f, fn in enumerate(schema.FINGER_ORDER):
                axx.annotate(fn, (med[f, i], med[f, j]), fontsize=7, color=INK, ha="left", va="bottom",
                             xytext=(3, 3), textcoords="offset points")
                axx.plot(med[f, i], med[f, j], "o", ms=3, color=INK)
            axx.set_xlabel("xyz"[i] + " [mm]")
            axx.set_ylabel("xyz"[j] + " [mm]")
            axx.set_aspect("equal")
            axx.grid(False)
            axx.set_title(f"{names[p]}: fingertips {'xyz'[i]}{'xyz'[j]}", color=INK)
            for sp in axx.spines.values():
                sp.set_edgecolor(COLORS[p])
    fig.tight_layout()
    save(fig, out, "fig07_fingertip_workspace.png", figs)

    # 08 fingertip motion
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    hist_overlay(ax[0], [d["tip_speed"].ravel() for d in S], lab, np.linspace(0, 0.6, 121),
                 "fingertip speed [m/s] (per tip, 1-step finite diff)", logy=True)
    ax[0].set_title("Fingertip speed")
    ax[0].legend()
    hist_overlay(ax[1], [1000 * d["tip_disp"].ravel() for d in S], lab, np.linspace(0, 40, 121),
                 "|p(t+4) - p(t)| [mm] (per tip)", logy=True)
    ax[1].set_title("0.2 s future fingertip displacement (DexGen motion target)")
    save(fig, out, "fig08_fingertip_motion.png", figs)

    # 09 object
    fig, axs = plt.subplots(2, 3, figsize=(14, 6.6))
    hist_overlay(axs[0, 0], [d["object_angvel_wrist"][:, 2] for d in S], lab, np.linspace(-12, 12, 121),
                 "object angular velocity about wrist z [rad/s]", logy=True)
    axs[0, 0].set_title("Object twist about wrist z")
    axs[0, 0].legend(fontsize=7)
    hist_overlay(axs[0, 1], [np.linalg.norm(d["object_angvel_wrist"], axis=-1) for d in S], lab,
                 np.linspace(0, 15, 121), "|object angular velocity| [rad/s]", logy=True)
    axs[0, 1].set_title("Object angular speed")
    dv = [d["object_dof_vel"][:, 0] if np.isfinite(d["object_dof_vel"]).any() else None for d in S]
    hist_overlay(axs[0, 2], dv, lab, np.linspace(-8, 8, 161), "object_dof_vel [rad/s]", logy=True)
    axs[0, 2].axvline(2 * np.pi, color=INK2, ls="--", lw=1.0)
    axs[0, 2].text(2 * np.pi, axs[0, 2].get_ylim()[1], " URDF\n velocity limit 2π", fontsize=7, color=INK2, va="top")
    axs[0, 2].set_title("Nut spin rate (dexscrew, nut_joint)")
    axs[0, 2].legend(fontsize=7, loc="upper left")
    hist_overlay(axs[1, 0], [d["object_contact_force"][:, 0] for d in S], lab, np.linspace(0, 30, 121),
                 "net object contact force [N]", logy=True)
    axs[1, 0].set_title("Object contact force (norm)")
    oc = [S[p]["object_contact_force"][:, 0] for p in range(len(S))]
    hist_overlay(axs[1, 2], [np.clip(x, 0, 30) for x in oc], lab, np.linspace(0, 30.0001, 121),
                 "net object contact force [N] (clipped at 30)", density=False, logy=False)
    axs[1, 2].set_ylabel("rows (sampled)")
    axs[1, 2].set_title("Object contact force, linear scale")
    fcf = [d["finger_contact_force"] if np.isfinite(d["finger_contact_force"]).any() else None for d in S]
    have = [p for p in range(len(S)) if fcf[p] is not None]
    if have:
        w = 0.8 / len(have)
        for k, p in enumerate(have):
            st = [pct_stats(fcf[p][:, i]) for i in range(5)]
            axs[1, 1].bxp(st, positions=np.arange(5) - 0.4 + w * (k + 0.5), widths=w * 0.85, showfliers=False,
                          patch_artist=True, boxprops=dict(facecolor=COLORS[p], edgecolor=COLORS[p], alpha=0.55),
                          medianprops=dict(color=INK), whiskerprops=dict(color=COLORS[p]),
                          capprops=dict(color=COLORS[p]))
            pct = [100 * (fcf[p][:, i] > 0.05).mean() for i in range(5)]
            for i in range(5):
                axs[1, 1].text(i - 0.4 + w * (k + 0.5), -0.6, f"{pct[i]:.0f}%", ha="center", fontsize=6, color=INK2)
        axs[1, 1].set_xticks(range(5))
        axs[1, 1].set_xticklabels(schema.FINGER_ORDER)
        axs[1, 1].set_ylabel("finger contact force [N]")
        axs[1, 1].set_title("Per-finger contact force (dexgen; % = rows in contact)")
        axs[1, 1].grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, out, "fig09_object.png", figs)

    # 10 coverage overlap: PCA of joint_pos
    nfit = min(min(len(d["joint_pos"]) for d in S), 300_000)
    fit = np.concatenate([d["joint_pos"][rng.choice(len(d["joint_pos"]), nfit, replace=False)] for d in S]).astype(np.float64)
    mu = fit.mean(0)
    _, sv, vt = np.linalg.svd(fit - mu, full_matrices=False)
    evr = sv ** 2 / (sv ** 2).sum()
    proj = [(d["joint_pos"] - mu) @ vt[:2].T for d in S]
    allp = np.concatenate(proj)
    r = [np.percentile(allp[:, i], [0.2, 99.8]) for i in range(2)]
    fig, axs = plt.subplots(1, len(S) + 1, figsize=(4 * (len(S) + 1), 3.8))
    for p, pr in enumerate(proj):
        axs[p].hist2d(pr[:, 0], pr[:, 1], bins=120, range=r, cmap="Blues", norm=LogNorm(), density=True)
        axs[p].set_title(names[p])
        axs[p].grid(False)
        for sp in axs[p].spines.values():
            sp.set_edgecolor(COLORS[p])
    # overlap panel: occupancy contours
    H = []
    for p, pr in enumerate(proj):
        h, xe, ye = np.histogram2d(pr[:, 0], pr[:, 1], bins=80, range=r, density=True)
        H.append(h)
        thr = np.sort(h.ravel())[::-1]
        thr = thr[np.searchsorted(np.cumsum(thr) / thr.sum(), 0.9)]  # contour enclosing 90 % mass
        axs[-1].contour(0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]), h.T, levels=[thr], colors=[COLORS[p]],
                        linewidths=1.6)
    axs[-1].set_title("90 % mass contours (overlap)")
    axs[-1].legend(legend_handles(lab), lab, fontsize=7, loc="upper right")
    for axx in axs:
        axx.set_xlabel(f"PC1 ({100 * evr[0]:.0f} %)")
        axx.set_ylabel(f"PC2 ({100 * evr[1]:.0f} %)")
    fig.tight_layout()
    save(fig, out, "fig10_joint_pca.png", figs)
    # overlap coefficient (histogram intersection) between policies in the 2D PCA plane
    ov = {}
    for i in range(len(H)):
        for j in range(i + 1, len(H)):
            a_, b_ = H[i] / H[i].sum(), H[j] / H[j].sum()
            ov[f"{names[i]} vs {names[j]}"] = float(np.minimum(a_, b_).sum())
    stats["joint_pca"] = {"explained_variance_ratio": evr[:6].round(4).tolist(), "histogram_intersection_pc12": ov}

    # 11 task progress
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    for p, d in enumerate(S):
        n = names[p]
        if "goals_per_episode" in stats["policies"][n]:
            u = np.unique(d["ep"])
            gr = np.bincount(np.searchsorted(u, d["ep"]), weights=d["goal_reached_event"][:, 0])
            ax[0].hist(gr, bins=np.arange(-0.5, gr.max() + 1.5), density=True, histtype="step", color=COLORS[p],
                       lw=1.6, label=n)
            ln = np.bincount(np.searchsorted(u, d["ep"])) * dt[n]
            ax[1].scatter(ln, gr + rng.uniform(-0.25, 0.25, len(gr)), s=2, color=COLORS[p], alpha=0.15, rasterized=True)
        if "nut_rot" in d:
            ax[2].hist(d["nut_rot"], bins=np.linspace(-5, 60, 131), density=True, histtype="step", color=COLORS[p],
                       lw=1.6, label=n)
    ax[0].set_xlabel("goal_reached events per episode")
    ax[0].set_ylabel("density")
    ax[0].set_title("Goals reached per episode (dexgen)")
    ax[0].legend(fontsize=7)
    ax[1].set_xlabel("stored episode length [s]")
    ax[1].set_ylabel("goals reached (jittered)")
    ax[1].set_title("Goals vs episode length (dexgen)")
    ax[1].legend(legend_handles(lab[:2]), lab[:2], fontsize=7)
    ax[2].set_xlabel("cumulative nut rotation per episode [rad]")
    ax[2].set_ylabel("density")
    ax[2].set_title("Nut rotation per episode (dexscrew)")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    save(fig, out, "fig11_task_progress.png", figs)

    stats["figures"] = figs
    with open(os.path.join(out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print("wrote", os.path.join(out, "stats.json"))


if __name__ == "__main__":
    main()
