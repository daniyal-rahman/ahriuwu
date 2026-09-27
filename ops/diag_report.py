#!/usr/bin/env python3
"""Summarise a run's per-step diagnostic record (<run>/diag/steps_*.npz).

    ops/diag_report.py <run dir or eval dir> [--by wall|update] [--updates a:b]

By wall distance (default): time share, value estimate, reward terms per
minute, click mass on unwalkable cells, sampled-click unwalkable rate, CS per
minute -- what the policy computed and earned where it stood."""
import argparse, glob, numpy as np
from pathlib import Path

def load(root):
    files = sorted(glob.glob(str(Path(root) / "**" / "diag" / "steps_*.npz"), recursive=True))
    if not files: raise SystemExit(f"no diag files under {root}")
    cols = None; parts = []
    for f in files:
        z = np.load(f); cols = list(z["cols"]); parts.append(z["data"])
    d = np.concatenate(parts); return {c: d[:, i] for i, c in enumerate(cols)}, len(files)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--by", default="wall", choices=("wall", "update"))
    ap.add_argument("--updates", default=None, help="a:b update range"); ap.add_argument("--hz", type=float, default=10.)
    a = ap.parse_args(); d, nf = load(a.root)
    if a.updates:
        lo, hi = (int(x) for x in a.updates.split(":")); m = (d["update"] >= lo) & (d["update"] < hi); d = {k: v[m] for k, v in d.items()}
    alive = d["alive"] > 0
    print(f"{a.root}: {len(d['t'])} agent-steps from {nf} files; updates {int(d['update'].min())}-{int(d['update'].max())}; "
          f"mean value {np.nanmean(d['value']):.3f}; p_unwalkable mean {np.nanmean(d['p_unwalkable']):.2f}; act_unwalkable {np.nanmean(d['act_unwalkable']):.2f}")
    if a.by == "wall":
        buckets = (("at wall <150", d["wall_dist"] < 150), ("near 150-400", (d["wall_dist"] >= 150) & (d["wall_dist"] < 400)), ("open >=400", d["wall_dist"] >= 400))
    else:
        edges = np.quantile(d["update"], [0, .25, .5, .75, 1.0]); buckets = [(f"u{int(edges[i])}-{int(edges[i+1])}", (d["update"] >= edges[i]) & (d["update"] < edges[i+1] + (1 if i == 3 else 0))) for i in range(4)]
    print(f"{'bucket':16s} {'share':>6s} {'value':>7s} {'CS/min':>7s} {'r_death/min':>11s} {'r_appr/min':>10s} {'r_xp/min':>8s} {'p_unwalk':>8s} {'act_unw':>7s} {'ent_x':>6s}")
    for name, sel in buckets:
        sel = sel & alive; n = sel.sum()
        if n == 0: print(f"{name:16s} {'0':>6s}"); continue
        mins = n / a.hz / 60
        print(f"{name:16s} {n/alive.sum():6.0%} {np.nanmean(d['value'][sel]):7.3f} {d['r_cs'][sel].sum()/mins:7.2f} {d['r_death'][sel].sum()/mins:11.3f} "
              f"{d['r_approach'][sel].sum()/mins:10.3f} {d['r_xp'][sel].sum()/mins:8.3f} {np.nanmean(d['p_unwalkable'][sel]):8.2f} {np.nanmean(d['act_unwalkable'][sel]):7.2f} {np.nanmean(d['ent_x'][sel]):6.2f}")

if __name__ == "__main__":
    main()
