#!/usr/bin/env python3
"""Render the headless server recording as a top-down replay + verification plots.

This is the evidence the client never gave us: whether the sim actually behaves
like League. Checks that matter for a laning agent, against real 4.20 values:
  - minion waves spawn every 30s (first wave at 65s)
  - minions travel down lanes rather than milling about (known LeagueSandbox bug)
  - turrets exist, hold position, and damage things
  - champion gold accrues (~passive 'gold per tick' plus last-hit income)
Usage: render_replay.py <state.jsonl> <outdir>
"""
import json, sys, os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

TEAM_COLOR = {100: "#4488ff", 200: "#ff4444", 300: "#888888"}


def load(path):
    frames = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                frames.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # tolerate a truncated final line
    return frames


def kind_of(u):
    k = u.get("k", "")
    if "Champion" in k:
        return "champion"
    if "Minion" in k:
        return "minion"
    if "Turret" in k:
        return "turret"
    if "Inhibitor" in k or "Nexus" in k:
        return "structure"
    return "other"


def summarize(frames):
    """Quantitative checks, printed and returned for the plots."""
    counts = defaultdict(list)     # kind -> per-frame count
    times = []
    gold = defaultdict(list)       # champion id -> gold
    minion_ids_seen = set()
    spawn_events = []              # (game_time_s, n_new_minions)

    prev_minions = set()
    for f in frames:
        t = f["t"] / 1000.0
        times.append(t)
        per = defaultdict(int)
        cur_minions = set()
        for u in f["u"]:
            k = kind_of(u)
            per[k] += 1
            if k == "minion":
                cur_minions.add(u["id"])
            if k == "champion" and "gold" in u:
                gold[u["id"]].append((t, u["gold"]))
        for k in ("champion", "minion", "turret", "structure"):
            counts[k].append(per[k])
        new = cur_minions - prev_minions
        if new and len(new) >= 3:          # a wave, not a single respawn
            spawn_events.append((t, len(new)))
        minion_ids_seen |= cur_minions
        prev_minions = cur_minions

    print(f"frames                : {len(frames)}")
    if times:
        print(f"game time covered     : {times[0]:.0f}s -> {times[-1]:.0f}s")
    if counts["champion"]:
        print(f"champions (max)       : {max(counts['champion'])}")
        print(f"turrets (max)         : {max(counts['turret'])}")
        print(f"minions (max on map)  : {max(counts['minion'])}")
        print(f"distinct minions ever : {len(minion_ids_seen)}")
    print(f"wave spawn events     : {len(spawn_events)}")
    for t, n in spawn_events[:10]:
        print(f"    t={t:6.1f}s  +{n} minions")
    if len(spawn_events) >= 2:
        gaps = [b[0] - a[0] for a, b in zip(spawn_events, spawn_events[1:])]
        print(f"    inter-wave gaps     : {[f'{g:.0f}s' for g in gaps[:8]]}  (real League: 30s)")
    for cid, series in gold.items():
        if series:
            print(f"champion {cid} gold    : {series[0][1]} -> {series[-1][1]}")
    return times, counts, gold, spawn_events


def plot_checks(times, counts, gold, spawns, outdir):
    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    ax[0].plot(times, counts["minion"], color="#cc7722")
    for t, _ in spawns:
        ax[0].axvline(t, color="k", alpha=.25, lw=.8)
    ax[0].set_ylabel("minions on map")
    ax[0].set_title("Headless LoLServer simulation — verification")
    ax[1].plot(times, counts["turret"], label="turrets", color="#666")
    ax[1].plot(times, counts["champion"], label="champions", color="#4488ff")
    ax[1].legend(); ax[1].set_ylabel("count")
    for cid, series in gold.items():
        ax[2].plot([t for t, _ in series], [g for _, g in series], label=f"champ {cid}")
    ax[2].set_ylabel("gold"); ax[2].set_xlabel("game time (s)")
    if gold:
        ax[2].legend()
    fig.tight_layout()
    p = os.path.join(outdir, "verification.png")
    fig.savefig(p, dpi=110)
    print(f"wrote {p}")


def render_video(frames, outdir, fps=20):
    """Top-down replay: this is the 'watch it play' artifact, from server truth."""
    xs = [u["x"] for f in frames for u in f["u"]]
    ys = [u["y"] for f in frames for u in f["u"]]
    if not xs:
        print("no units to render")
        return
    pad = 500
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal"); ax.set_facecolor("#12160f")
    title = ax.set_title("")
    scat = ax.scatter([], [], s=[], c=[])

    path = os.path.join(outdir, "replay.mp4")
    writer = FFMpegWriter(fps=fps, bitrate=2400)
    with writer.saving(fig, path, dpi=100):
        for f in frames:
            X, Y, C, S = [], [], [], []
            for u in f["u"]:
                k = kind_of(u)
                X.append(u["x"]); Y.append(u["y"])
                C.append(TEAM_COLOR.get(u.get("tm"), "#aaaaaa"))
                S.append({"champion": 120, "turret": 90, "structure": 140}.get(k, 18))
            scat.set_offsets(list(zip(X, Y)))
            scat.set_color(C); scat.set_sizes(S)
            title.set_text(f"t = {f['t']/1000:6.1f}s    units = {len(f['u'])}")
            writer.grab_frame()
    print(f"wrote {path}")


if __name__ == "__main__":
    src, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    frames = load(src)
    if not frames:
        print("no frames recorded"); sys.exit(1)
    times, counts, gold, spawns = summarize(frames)
    plot_checks(times, counts, gold, spawns, outdir)
    render_video(frames, outdir)
