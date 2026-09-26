#!/usr/bin/env python3
"""Regenerate every figure in the root README from the run records.

    ops/login_capped.sh 4G 1 env JAX_PLATFORMS=cpu .venv-jax/bin/python ops/figures/readme_figures.py

Reads lanerl_jax/runs/*/metrics.jsonl and lanerl_jax/runs/EVAL/summary.jsonl
(read-only) and writes docs/figures/*.png. Colours are fixed per run (the
dataviz reference palette, validated light mode), never by rank.
"""
import glob, json, os, shutil, statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "lanerl_jax/runs"
OUT = REPO / "docs/figures"

# Fixed categorical slots (dataviz reference palette, light): identity follows the run.
RUN_COLOR = {"E01": "#2a78d6", "E04": "#eb6834", "E06": "#1baf7a", "E07": "#eda100", "E05": "#e87ba4"}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#ebeae6", "#fcfcfb"
BUTTONS = ("noop", "move", "attack_move", "q", "w", "e", "r", "recall")
BUTTON_COLOR = dict(zip(BUTTONS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]))

GATE, UNTRAINED = 30, 8   # gate: AGENTS.md; untrained median: ledger, dead-controls/all-five (CS 14/4/6/9/8)

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def load(root, exclude=()):
    """Same supersession rule as ops/plot_train_curve.py: segments are run dirs in
    time order; a later segment that resumed from update R supersedes the
    earlier one past R. Returns (episodes, updates) with a `decisions` field
    taken from the update rows' cumulative `steps`."""
    segs = []
    for d in sorted(glob.glob(str(root) + "/*-s[0-9]*/")):
        if any(x in d for x in exclude):
            continue
        eps, ups = [], []
        for line in open(d + "metrics.jsonl"):
            r = json.loads(line)
            if "entropy" in r:
                ups.append(r)
            elif "episode" in r and "cs" in r:
                eps.append(r)
        if ups:
            segs.append((eps, ups))
    eps, ups = [], []
    for i, (e, u) in enumerate(segs):
        cut = segs[i + 1][1][0]["update"] if i + 1 < len(segs) else float("inf")
        eps += [r for r in e if r["update"] < cut]
        ups += [r for r in u if r["update"] < cut]
    steps = {u["update"]: u["steps"] for u in ups}
    for e in eps:
        e["decisions"] = steps.get(e["update"], steps[max(k for k in steps if k <= e["update"])])
    return eps, ups, steps


# E01: the last dir (232101) is a slurm requeue from a stale u461 checkpoint
# after the 23:20 reboot (docs/EXPERIMENTS.md). Its frozen evaluations and E04's
# branch point both come from the 195150 segment, so the requeue is excluded.
E01 = dict(root=RUNS / "server_train/mirror-wave-s0", exclude=("232101",))
E04 = dict(root=RUNS / "E04_mirror_wave_snap_noxp/seed0", exclude=())
E05 = dict(root=RUNS / "E05_mirror_wave_gru_standard/seed0", exclude=())
# E06: the 093005 segment resumed from u620, so supersession already drops the
# OPS-004 collapse (updates 621-971 of the 064502 segment).
E06 = dict(root=RUNS / "E06_mirror_wave_gru_ent3/seed0", exclude=())
# E07: only the relaunch; seed0_v1_inherited_schedule is INVALID (not globbed:
# it is a sibling of seed0, not inside it).
E07 = dict(root=RUNS / "E07_frozen_opponent/seed0", exclude=())
E07_INIT_UPDATE = 3140   # --init-from E06 eval_u3140 (experiments/E07_frozen_opponent.json)

INVALID_EVALS = {("E06", 920), ("E06", 960)}   # OPS-004


def run_id(path):
    for k, key in [("E04", "E04_"), ("E05", "E05_"), ("E06", "E06_"), ("E07", "E07_"), ("E01", "mirror-wave-s0")]:
        if key in (path or ""):
            return k
    return None


def frozen_evals(step_maps):
    """[(run, update, decisions, blue_mean, red_mean)] from summary.jsonl, plus E01
    u760 which predates summary.jsonl (docs/EXPERIMENTS.md frozen table)."""
    out = [("E01", 760, 760 * 2560, 19.4, 16.0)]
    for line in open(RUNS / "EVAL/summary.jsonl"):
        r = json.loads(line)
        s = r.get("summary")
        rid = run_id(r.get("run") or r.get("checkpoint"))
        if not s or rid is None or (rid, r["update"]) in INVALID_EVALS:
            continue
        steps = step_maps[rid]
        out.append((rid, r["update"], steps[r["update"]], s["0"]["mean_cs"], s["1"]["mean_cs"]))
    return sorted(out, key=lambda t: (t[0], t[1]))


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def refs(ax, x0):
    ax.axhline(GATE, color=MUTED, lw=1, ls="--")
    ax.text(0.995, GATE + 0.6, "30-CS gate", color=INK2, fontsize=9, ha="right",
            transform=ax.get_yaxis_transform())
    ax.axhline(UNTRAINED, color=MUTED, lw=1, ls=":")
    ax.text(0.995, UNTRAINED - 0.6, "untrained policy, median 8", color=INK2, fontsize=9, ha="right", va="top",
            transform=ax.get_yaxis_transform())


def rolling(y, k=24):
    return np.convolve(y, np.ones(k) / k, mode="valid")


def training_curves(data, evals):
    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=150)
    style(ax)
    for rid in ("E01", "E04", "E06", "E07"):
        eps, ups, steps, offset = data[rid]
        eps = sorted(eps, key=lambda e: e["update"])
        x = (np.array([e["decisions"] for e in eps]) + offset) / 1e6
        y = np.array([e["cs"] for e in eps])
        if len(y) < 24:
            continue
        yr = rolling(y)
        ax.plot(x[23:], yr, color=RUN_COLOR[rid], lw=2, label=f"{rid} train (24-episode mean)")
        ax.annotate(rid, (x[-1], yr[-1]), xytext=(4, 0), textcoords="offset points",
                    color=INK, fontsize=9, va="center", fontweight="bold")
    for rid, u, dec, b, r in evals:
        if rid not in ("E01", "E04", "E06", "E07"):
            continue
        off = data[rid][3]
        ax.scatter((dec + off) / 1e6, b, marker="o", s=70, color=RUN_COLOR[rid], edgecolors=SURFACE, linewidths=1.5, zorder=5)
        ax.scatter((dec + off) / 1e6, r, marker="o", s=46, facecolors=SURFACE, edgecolors=RUN_COLOR[rid], linewidths=1.8, zorder=5)
    ax.scatter([], [], marker="o", s=46, color=INK2, label="frozen eval, blue-side mean")
    ax.scatter([], [], marker="o", s=46, facecolors=SURFACE, edgecolors=INK2, linewidths=1.8, label="frozen eval, red-side mean")
    refs(ax, 0.05)
    ax.set_xlabel("agent decisions (millions); E04 continues E01 at u1520, E07 continues E06 at u3140")
    ax.set_ylabel("CS per 600 s episode")
    ax.set_ylim(0, 36)
    ax.set_xlim(0, None)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=8, frameon=False, ncol=3)
    ax.set_title("Training CS (sampled, changing weights) and frozen evaluations, seed 0", color=INK, loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "training_curves.png")
    plt.close(fig)


def frozen_eval_plot(data, evals):
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=150)
    style(ax)
    for rid in ("E01", "E04", "E05", "E06"):
        pts = [(dec + data.get(rid, (0, 0, 0, 0))[3], b, r) for q, u, dec, b, r in evals if q == rid]
        if not pts:
            continue
        x = np.array([p[0] for p in pts]) / 1e6
        b = np.array([p[1] for p in pts]); r = np.array([p[2] for p in pts])
        ax.vlines(x, np.minimum(b, r), np.maximum(b, r), color=RUN_COLOR[rid], lw=1.2, alpha=0.6)
        ax.plot(x, (b + r) / 2, color=RUN_COLOR[rid], lw=2, label=f"{rid} (line: mean of both sides)")
        ax.scatter(x, b, marker="o", s=70, color=RUN_COLOR[rid], edgecolors=SURFACE, linewidths=1.5, zorder=5)
        ax.scatter(x, r, marker="o", s=46, facecolors=SURFACE, edgecolors=RUN_COLOR[rid], linewidths=1.8, zorder=5)
        ax.annotate(rid, (x[-1], max(b[-1], r[-1])), xytext=(0, 7), textcoords="offset points",
                    ha="center", color=INK, fontsize=9, fontweight="bold")
    ax.scatter([], [], marker="o", s=46, color=INK2, label="blue side (4-5 episodes)")
    ax.scatter([], [], marker="o", s=46, facecolors=SURFACE, edgecolors=INK2, linewidths=1.8, label="red side (4-5 episodes)")
    refs(ax, 0.05)
    ax.set_xlabel("agent decisions at the evaluated checkpoint (millions)")
    ax.set_ylabel("mean CS per frozen 600 s episode")
    ax.set_ylim(0, 36); ax.set_xlim(0, None)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=8, frameon=False, ncol=3)
    ax.set_title("Frozen-policy evaluations (sampled actions, mirror, near-wave start); E06 u920/u960 excluded (OPS-004)",
                 color=INK, loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "frozen_evals.png")
    plt.close(fig)


def action_mix(ups):
    ups = sorted(ups, key=lambda u: u["update"])
    x = np.array([u["update"] for u in ups])
    counts = np.array([[u["sampled_buttons"].get(b, 0) for b in BUTTONS] for u in ups], float)
    k = 40
    sm = np.stack([np.convolve(counts[:, j], np.ones(k) / k, mode="valid") for j in range(len(BUTTONS))], 1)
    frac = sm / sm.sum(1, keepdims=True) * 100
    xs = x[k - 1:]
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.spines[["top", "right"]].set_visible(False)
    ax.stackplot(xs, frac.T, colors=[BUTTON_COLOR[b] for b in BUTTONS], labels=BUTTONS,
                 edgecolor=SURFACE, linewidth=0.6)
    cum = np.cumsum(frac[-1]) - frac[-1] / 2
    for j, b in enumerate(BUTTONS):
        if frac[-1, j] > 3:
            ax.annotate(b, (xs[-1], cum[j]), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=INK)
    ax.set_xlim(xs[0], xs[-1]); ax.set_ylim(0, 100)
    ax.set_xlabel("E06 update (2,560 decisions per update)")
    ax.set_ylabel("share of sampled buttons (%, 40-update mean)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=8, fontsize=8, frameon=False)
    ax.set_title("E06 sampled-button mix during training (both sides, sampled actions)", color=INK, loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "action_mix.png")
    plt.close(fig)


def rate(root, exclude=()):
    """Aggregate decisions/s of a run: sum of steps over sum of wall time, per
    segment from its second update on (the first carries set-up and compile).
    Episode resets (fresh server processes) are included, so this is what a run
    actually delivers. Returns (aggregate, slowest segment, fastest segment)."""
    tot_s = tot_w = 0.0
    seg = []
    for d in sorted(glob.glob(str(root) + "/*-s[0-9]*/")):
        if any(x in d for x in exclude):
            continue
        u = [json.loads(l) for l in open(d + "metrics.jsonl")]
        u = [r for r in u if "entropy" in r and "wall_s" in r]
        if len(u) < 20:
            continue
        ds, dw = u[-1]["steps"] - u[1]["steps"], u[-1]["wall_s"] - u[1]["wall_s"]
        tot_s += ds; tot_w += dw; seg.append(ds / dw)
    return tot_s / tot_w, min(seg), max(seg)


def throughput():
    rows = [
        ("Codex collector, 2 servers, 30 Hz\n(dead-train, before batched observe)",
         rate(RUNS / "server_first_20260925/dead-train/seed0")),
        ("batched observe (8adaba4), 4 servers, 10 Hz\n(idle-wave2-s0 = E02)",
         rate(RUNS / "server_train/idle-wave2-s0")),
        ("10 servers mirror, MLP\n(E01, all segments)",
         rate(E01["root"])),
        ("12 servers mirror, MLP, node probe\n(EXPERIMENTS.md throughput row: 768 dec / 1.0 s)",
         (768.0, None, None)),
        ("10 servers mirror, GRU + BPTT update\n(E06, all segments)",
         rate(E06["root"])),
    ]
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=150)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    y = np.arange(len(rows))[::-1]
    for yi, (label, (med, lo, hi)) in zip(y, rows):
        ax.barh(yi, med, height=0.55, color=RUN_COLOR["E01"])
        txt = f"{med:.0f}" + ((f"  (segments {lo:.0f}-{hi:.0f})" if hi - lo >= 1 else "  (one segment)") if lo is not None else "  (single probe)")
        ax.text(med + 8, yi, txt, va="center", fontsize=9, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("agent decisions per second (total steps / total wall time, resets included)")
    ax.set_xlim(0, 950)
    ax.set_title("Collector + learner throughput on the desktop (16 cores, RTX 5080), C# server", color=INK, loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "throughput.png")
    plt.close(fig)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data, step_maps = {}, {}
    for rid, spec in [("E01", E01), ("E04", E04), ("E05", E05), ("E06", E06), ("E07", E07)]:
        eps, ups, steps = load(spec["root"], spec["exclude"])
        data[rid] = [eps, ups, steps, 0]
        step_maps[rid] = steps
    data["E07"][3] = step_maps["E06"][E07_INIT_UPDATE]
    evals = frozen_evals(step_maps)
    training_curves(data, evals)
    frozen_eval_plot(data, evals)
    action_mix(data["E06"][1])
    rows = throughput()
    for src, dst in [("plots/lane_potential_map.png", "lane_potential_map.png"),
                     ("EVAL/replay_E06_u3460/combat/preview.png", "replay_E06_u3460_combat.png"),
                     ("EVAL/replay_E06_u3460/map/preview.png", "replay_E06_u3460_map.png")]:
        shutil.copyfile(RUNS / src, OUT / dst)
    for rid, u, dec, b, r in evals:
        print(f"eval {rid} u{u} {dec/1e6:.2f}M blue {b} red {r}")
    for label, (med, lo, hi) in rows:
        print("throughput", label.replace("\n", " "), round(med), lo and round(lo), hi and round(hi))
    print("E07 updates so far:", max(data["E07"][2]), "episodes:", len(data["E07"][0]))
    print("wrote", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
