#!/usr/bin/env python
"""Where the agent dies, over the lane geometry, straight from metrics.jsonl.

This is the cheap replacement for re-running games under a position tracer.
``death_positions`` rides on every episode record of every training run, so
this works on the games that actually happened rather than on games somebody
thought to reproduce.

Panels:
  1. death locations over the lane_approach field, with the lane_presence
     corridor and both turrets drawn, split early/late in the run so a drift
     is visible rather than averaged away;
  2. distance from each death to the ENEMY turret -- the turret-dive test;
  3. deaths per episode over training.

    python lanerl/plot_deaths.py <run-dir-or-metrics.jsonl> [...] -o out.png
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from lanerl_rl import constants as C

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_lane_reward import (  # noqa: E402
    approach_field, corridor_polygon, in_corridor, lane_frame,
)

TEAM, ENEMY = C.TEAM_BLUE, C.TEAM_RED


def load(path: str):
    if os.path.isdir(path):
        path = os.path.join(path, "metrics.jsonl")
    eps = []
    with open(path) as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("kind") == "episode":
                eps.append(d)
    return eps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("-o", "--out", default="deaths.png")
    ap.add_argument("--corridor", type=float, default=C.LANE_HALF_WIDTH)
    args = ap.parse_args()

    from lanerl_rl.reward import RewardWeights
    per_1000 = RewardWeights().lane_approach

    lane = lane_frame()
    gx = np.linspace(-500, 15000, 300)
    gy = np.linspace(-500, 15500, 300)
    field = approach_field(lane, args.corridor, per_1000, gx, gy)
    et = C.TOP_OUTER_TURRET[ENEMY]

    n = len(args.runs)
    fig, axes = plt.subplots(n, 3, figsize=(19, 6.2 * n), squeeze=False)

    for row, run in enumerate(args.runs):
        eps = load(run)
        pts, order, per_ep = [], [], []
        for i, e in enumerate(eps):
            dp = e.get("death_positions") or []
            per_ep.append(len(dp))
            for (_t, x, y) in dp:
                pts.append((x, y))
                order.append(i / max(1, len(eps) - 1))
        name = os.path.basename(run.rstrip("/")) or run

        ax = axes[row][0]
        im = ax.contourf(gx, gy, field, levels=24, cmap="RdYlGn", alpha=0.5)
        ax.contour(gx, gy, field, levels=12, colors="k", linewidths=0.3, alpha=0.3)
        ax.add_patch(Polygon(corridor_polygon(lane, args.corridor), closed=True,
                             fill=False, edgecolor="black", linewidth=2.4, zorder=5))
        if pts:
            P = np.array(pts)
            sc = ax.scatter(P[:, 0], P[:, 1], c=order, cmap="cool", s=34,
                            edgecolors="k", linewidths=0.4, zorder=7)
            plt.colorbar(sc, ax=ax, shrink=0.75, label="early -> late in run")
        ax.plot(*C.TOP_OUTER_TURRET[TEAM], "s", color="royalblue", ms=14, zorder=8)
        ax.plot(*et, "s", color="crimson", ms=14, zorder=8)
        ax.plot(*C.NEXUS_POSITION[TEAM], "o", color="lime", ms=13,
                markeredgecolor="k", zorder=8)
        inside = in_corridor(lane, args.corridor,
                             np.array([p[0] for p in pts]) if pts else np.array([]),
                             np.array([p[1] for p in pts]) if pts else np.array([]))
        frac = float(np.mean(inside)) if pts else float("nan")
        ax.set_title(f"{name}: {len(pts)} deaths over {len(eps)} episodes\n"
                     f"{100*frac:.0f}% of deaths inside the lane corridor")
        ax.set_aspect("equal"); ax.set_xlim(-500, 15000); ax.set_ylim(-500, 15500)
        ax.grid(alpha=0.2)

        ax = axes[row][1]
        if pts:
            d = [math.hypot(x - et[0], y - et[1]) for x, y in pts]
            ax.hist(d, bins=34, color="crimson", alpha=0.8)
            ax.axvline(775, color="k", ls="--",
                       label="turret range (775)")
            ax.axvline(float(np.median(d)), color="navy", ls=":",
                       label=f"median {np.median(d):.0f}")
            under = 100.0 * float(np.mean(np.array(d) <= 775))
            ax.set_title(f"distance from death to ENEMY turret\n"
                         f"{under:.0f}% died inside its range")
            ax.legend(fontsize=9)
        ax.set_xlabel("units"); ax.grid(alpha=0.2)

        ax = axes[row][2]
        if per_ep:
            k = max(1, len(per_ep) // 40)
            sm = [np.mean(per_ep[i:i + k]) for i in range(0, len(per_ep), k)]
            ax.plot(np.arange(len(sm)) * k, sm, lw=2, color="darkred")
            ax.set_title(f"deaths per episode (mean {np.mean(per_ep):.2f})")
        ax.set_xlabel("episode"); ax.grid(alpha=0.2)

    fig.suptitle("Where the agent dies", fontsize=14)
    fig.savefig(args.out, dpi=105, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
