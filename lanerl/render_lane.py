#!/usr/bin/env python3
"""Render the recording so movement is actually VISIBLE.

The first attempt plotted the whole 14k-unit map at 1x-ish speed: only 0.5% of
pixels changed between frames, so it read as a static scatter even though every
one of 456 minions travelled thousands of units. Fixes here:
  - subsample ticks (~10x time compression) so per-frame motion is large
  - split view: whole map for context + TOP LANE zoom (this project's lane)
  - motion trails, so each unit's recent path is drawn behind it
  - filled markers, sized by unit class
Usage: render_lane.py <state.jsonl> <outdir> [stride]
"""
import json, sys, os
from collections import defaultdict, deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

BLUE, RED, NEUT = "#4aa3ff", "#ff5555", "#9a9a9a"
TRAIL = 22  # ticks of history drawn behind each unit


def team_color(t):
    return {100: BLUE, 200: RED}.get(t, NEUT)


def cls(u):
    k = u.get("k", "")
    if "Champion" in k:
        return "champ"
    if "LaneMinion" in k:
        return "minion"
    if "Turret" in k:
        return "turret"
    if "Monster" in k:
        return "monster"
    return "struct"


SIZE = {"champ": 220, "minion": 60, "turret": 150, "monster": 40, "struct": 230}


def main(src, outdir, stride):
    os.makedirs(outdir, exist_ok=True)
    frames = []
    with open(src) as fh:
        for i, line in enumerate(fh):
            if i % stride:
                continue
            line = line.strip()
            if line:
                try:
                    frames.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    print(f"{len(frames)} frames after stride={stride}")

    fig, (axm, axl) = plt.subplots(1, 2, figsize=(19, 9.5))
    for ax in (axm, axl):
        ax.set_facecolor("#0d1108")
        ax.set_aspect("equal")
        ax.tick_params(colors="#666", labelsize=8)
    fig.patch.set_facecolor("#0d1108")

    axm.set_xlim(-400, 15000); axm.set_ylim(-400, 15000)
    axm.set_title("Summoner's Rift — full map", color="#ddd")
    # top lane corridor: minions run along high-y then down the left side
    # mid-lane corridor: this is where waves visibly march in and collide
    axl.set_xlim(4200, 10200); axl.set_ylim(4200, 10200)
    axl.set_title("MID LANE (zoom) — waves marching in and fighting", color="#ddd")

    # facecolors (not edgecolors) so the dots are solid, not hollow rings
    scat_m = axm.scatter([], [], s=[], facecolors=[], edgecolors="none")
    scat_l = axl.scatter([], [], s=[], facecolors=[], edgecolors="none")
    trail_m, = axm.plot([], [], ".", ms=2.2, color="#4d7a4d", alpha=.7)
    trail_l, = axl.plot([], [], ".", ms=5.0, color="#4d7a4d", alpha=.7)
    sup = fig.suptitle("", color="#eee", fontsize=15)

    hist = defaultdict(lambda: deque(maxlen=TRAIL))
    path = os.path.join(outdir, "lane_replay.mp4")
    writer = FFMpegWriter(fps=30, bitrate=3600)
    with writer.saving(fig, path, dpi=100):
        for f in frames:
            X = Y = None
            xs, ys, cs, ss = [], [], [], []
            tx, ty = [], []
            live = set()
            for u in f["u"]:
                c = cls(u)
                live.add(u["id"])
                hist[u["id"]].append((u["x"], u["y"]))
                xs.append(u["x"]); ys.append(u["y"])
                cs.append(team_color(u.get("tm"))); ss.append(SIZE[c])
                if c in ("minion", "champ"):
                    for hx, hy in hist[u["id"]]:
                        tx.append(hx); ty.append(hy)
            for dead in [k for k in hist if k not in live]:
                del hist[dead]

            off = list(zip(xs, ys))
            scat_m.set_offsets(off); scat_m.set_facecolor(cs); scat_m.set_sizes(ss)
            scat_l.set_offsets(off); scat_l.set_facecolor(cs)
            scat_l.set_sizes([s * 5.0 for s in ss])
            trail_m.set_data(tx, ty); trail_l.set_data(tx, ty)
            mins = sum(1 for u in f["u"] if cls(u) == "minion")
            sup.set_text(f"game time {f['t']/1000:6.1f}s     units {len(f['u'])}     lane minions {mins}")
            writer.grab_frame()
    print(f"wrote {path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 3)
