#!/usr/bin/env python
"""Draw a trajectory ON TOP OF the lane reward field, to scale.

"CS = 0" and "it wandered" are two different claims, and only the second one
can be checked against what the reward actually pays. This overlays:

  * the ``lane_presence`` corridor -- the RECTANGLE where the indicator pays,
    |n| <= lane_corridor and -corridor <= s <= length + corridor;
  * the ``lane_approach`` potential as a filled contour, so the direction the
    shaping gradient points is visible rather than inferred;
  * the path, coloured by game time, with an explicit marker wherever the
    champion was being paid lane_presence.

The number printed under each panel is the one that settles the argument:
the fraction of decisions spent inside the corridor, which is exactly what
the episode's ``lane_presence`` reward term divides out to.

    python lanerl/plot_lane_reward.py <trace.json> [...] -o out.png
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
from lanerl_rl.frame import LaneFrame

TEAM = C.TEAM_BLUE
ENEMY = C.TEAM_RED


def lane_frame() -> LaneFrame:
    return LaneFrame(C.TOP_OUTER_TURRET[TEAM], C.TOP_OUTER_TURRET[ENEMY],
                     C.NEXUS_POSITION[TEAM])


def corridor_polygon(lane: LaneFrame, corridor: float):
    """The lane_presence rectangle, back in world coordinates."""
    ox, oy = lane.origin
    ax, ay = lane.axis
    nx, ny = lane.normal
    s0, s1 = -corridor, lane.length + corridor
    pts = []
    for s, n in ((s0, -corridor), (s1, -corridor), (s1, corridor), (s0, corridor)):
        pts.append((ox + ax * s + nx * n, oy + ay * s + ny * n))
    return pts


def approach_field(lane: LaneFrame, corridor: float, per_1000: float,
                   xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """-(per_1000/1000) * distance to the corridor rectangle, on a grid."""
    X, Y = np.meshgrid(xs, ys)
    ox, oy = lane.origin
    ax, ay = lane.axis
    nx, ny = lane.normal
    dx, dy = X - ox, Y - oy
    s = dx * ax + dy * ay
    n = dx * nx + dy * ny
    off_n = np.maximum(0.0, np.abs(n) - corridor)
    off_s = np.maximum(0.0, np.maximum(-corridor - s, s - (lane.length + corridor)))
    return -(per_1000 / 1000.0) * np.hypot(off_n, off_s)


def in_corridor(lane: LaneFrame, corridor: float, x, y):
    ox, oy = lane.origin
    ax, ay = lane.axis
    nx, ny = lane.normal
    dx, dy = np.asarray(x) - ox, np.asarray(y) - oy
    s = dx * ax + dy * ay
    n = dx * nx + dy * ny
    return (np.abs(n) <= corridor) & (s >= -corridor) & (s <= lane.length + corridor)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("-o", "--out", default="lane_reward.png")
    ap.add_argument("--corridor", type=float, default=C.LANE_HALF_WIDTH)
    ap.add_argument("--per-1000", type=float, default=None,
                    help="lane_approach weight (default: RewardWeights)")
    args = ap.parse_args()

    if args.per_1000 is None:
        from lanerl_rl.reward import RewardWeights
        args.per_1000 = RewardWeights().lane_approach

    lane = lane_frame()
    n = len(args.traces)
    fig, axes = plt.subplots(1, n, figsize=(7.0 * n, 7.4), squeeze=False)

    gx = np.linspace(-500, 15000, 320)
    gy = np.linspace(-500, 15500, 320)
    field = approach_field(lane, args.corridor, args.per_1000, gx, gy)

    for ax_, path in zip(axes[0], args.traces):
        rows = json.load(open(path))
        xs = np.array([r["x"] for r in rows])
        ys = np.array([r["y"] for r in rows])
        ts = np.array([r["t"] for r in rows]) / 1000.0
        cs = [r["cs"] for r in rows if r["cs"] is not None]

        im = ax_.contourf(gx, gy, field, levels=24, cmap="RdYlGn", alpha=0.55)
        ax_.contour(gx, gy, field, levels=12, colors="k", linewidths=0.3, alpha=0.35)

        ax_.add_patch(Polygon(corridor_polygon(lane, args.corridor),
                              closed=True, fill=False, edgecolor="black",
                              linewidth=2.4, zorder=5,
                              label=f"lane_presence corridor (|n|<={args.corridor:.0f})"))

        inside = in_corridor(lane, args.corridor, xs, ys)
        ax_.scatter(xs[~inside], ys[~inside], s=1.5, c="0.25", alpha=0.35,
                    zorder=6, label="paid nothing")
        ax_.scatter(xs[inside], ys[inside], s=3.0, c="blue", alpha=0.8,
                    zorder=7, label="paid lane_presence")

        ax_.plot(*C.TOP_OUTER_TURRET[TEAM], "s", color="royalblue", ms=13,
                 zorder=8, label="own turret")
        ax_.plot(*C.TOP_OUTER_TURRET[ENEMY], "s", color="crimson", ms=13,
                 zorder=8, label="enemy turret")
        ax_.plot(*C.NEXUS_POSITION[TEAM], "o", color="lime", ms=13,
                 markeredgecolor="k", zorder=8, label="spawn")

        frac = float(inside.mean())
        ep_pay = frac * len(rows) * 1e-4
        title = os.path.basename(path)
        ax_.set_title(
            f"{title}   final CS={cs[-1] if cs else 0}\n"
            f"{100*frac:.1f}% of decisions in corridor "
            f"(lane_presence pays {ep_pay:.2f} this episode)",
            fontsize=11)
        ax_.set_aspect("equal")
        ax_.set_xlim(-500, 15000)
        ax_.set_ylim(-500, 15500)
        ax_.grid(alpha=0.2)

    axes[0][0].legend(loc="lower right", fontsize=8, framealpha=0.9)
    cb = fig.colorbar(im, ax=axes[0], shrink=0.8, pad=0.01)
    cb.set_label(f"lane_approach potential  (w={args.per_1000}/1000)")
    fig.suptitle("Where the lane reward pays, and where the policy actually went",
                 fontsize=13)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
