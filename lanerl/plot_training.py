#!/usr/bin/env python
"""Learning curves for a lane-rl run.

Deliberately plots the diagnostics that can FALSIFY progress, not just the
ones that look like activity. In symmetric self-play `score` is 0.5 by
construction, so it is excluded: a flat line there means nothing and plotting
it invites reading it as "stable".
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

run = Path(sys.argv[1])
ups, eps = [], []
for line in (run / "metrics.jsonl").read_text().splitlines():
    try: m = json.loads(line)
    except Exception: continue
    if m.get("kind") == "update": ups.append(m)
    elif m.get("kind") == "episode": eps.append(m)
print(f"{len(ups)} updates, {len(eps)} episodes")

def smooth(v, k=101):
    v = np.asarray(v, dtype=float)
    if len(v) < k: return v
    return np.convolve(v, np.ones(k) / k, mode="valid")

x = [u["update"] for u in ups]
panels = [
    ("loss/entropy",    "policy entropy  (must FALL if it is learning)"),
    ("loss/value_loss", "value loss  (critic fit)"),
    ("loss/approx_kl",  "approx KL per update"),
    ("loss/clip_frac",  "clip fraction"),
    ("loss/grad_norm",  "grad norm"),
]
fig, axes = plt.subplots(3, 2, figsize=(13, 10))
for ax, (k, title) in zip(axes.flat, panels):
    v = [u.get(k, np.nan) for u in ups]
    ax.plot(x, v, lw=0.4, alpha=0.25, color="tab:blue")
    s = smooth(v)
    ax.plot(x[len(x) - len(s):], s, lw=1.8, color="tab:blue")
    ax.set_title(title, fontsize=10); ax.set_xlabel("update"); ax.grid(alpha=0.3)

ax = axes.flat[5]
if eps:
    cs = [(i, e["cs_at_10"]) for i, e in enumerate(eps) if isinstance(e.get("cs_at_10"), (int, float))]
    if cs:
        xi, yv = zip(*cs)
        ax.plot(xi, yv, ".", ms=3, alpha=0.4)
        s = smooth(list(yv), 51)
        if len(s): ax.plot(range(len(yv) - len(s), len(yv)), s, lw=2, color="tab:red")
        ax.set_title(f"CS@10  (n={len(cs)})", fontsize=10)
    else:
        ax.text(.5, .5, "cs_at_10 never recorded\n(fixed 2026-09-11; needs a fresh run)",
                ha="center", va="center", fontsize=11, color="crimson")
        ax.set_title("CS@10 -- THE skill metric", fontsize=10)
    ax.set_xlabel("episode"); ax.grid(alpha=0.3)
fig.suptitle(f"{run.name}: {len(ups)} updates, {len(eps)} episodes", fontsize=12)
fig.tight_layout()
out = Path("lanerl/figs") / f"curves_{run.name}.png"
fig.savefig(out, dpi=120)
print("wrote", out)
