#!/usr/bin/env python
"""Draw what a policy actually DOES on the map.

Summary statistics hide spatial bugs. "CS = 5" does not tell you whether the
champion never left the fountain, walked into the enemy tower, or farmed fine
and died to the wave. A trajectory does.

Panels, left to right:
  1. where it walked (path, coloured by game time)
  2. where it pressed things (one marker per action type)
  3. how far from spawn it got, over time -- the reach-the-lane test
  4. HP and CS over time

  python lanerl/plot_behaviour.py <trace.json|record.jsonl> [...] -o out.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lanerl_rl import constants as C  # noqa: E402

BLUE_T = C.TOP_OUTER_TURRET[C.TEAM_BLUE]
RED_T = C.TOP_OUTER_TURRET[C.TEAM_RED]
CONTACT = ((BLUE_T[0] + RED_T[0]) / 2, (BLUE_T[1] + RED_T[1]) / 2)


def load(path: Path):
    """Either a policy trace (list of dicts) or a LANERL_RECORD jsonl."""
    if path.suffix == ".json":
        rows = json.loads(path.read_text())
        return [(r["t"], r["x"], r["y"], r.get("hp"), r.get("cs"), r.get("a")) for r in rows]
    out = []
    for line in path.read_text().splitlines():
        try:
            f = json.loads(line)
        except Exception:
            continue
        ch = next((u for u in f.get("u", ())
                   if u.get("k") == "Champion" and u.get("tm") == 100), None)
        if ch:
            out.append((f["t"], ch["x"], ch["y"], ch.get("hp"), ch.get("cs"), None))
    return out


def draw_map(ax):
    ax.plot([BLUE_T[0], RED_T[0]], [BLUE_T[1], RED_T[1]], "k--", lw=1, alpha=0.4,
            label="lane axis")
    ax.plot(*BLUE_T, "s", ms=11, color="tab:blue", label="own turret")
    ax.plot(*RED_T, "s", ms=11, color="tab:red", label="enemy turret")
    ax.plot(*CONTACT, "*", ms=16, color="gold", markeredgecolor="k", label="wave contact")
    ax.set_xlim(-500, 14500); ax.set_ylim(-500, 15000)
    ax.set_aspect("equal"); ax.grid(alpha=0.2)


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "-o"]
    out = Path(args[-1]) if sys.argv[-2] == "-o" else Path("lanerl/figs/behaviour.png")
    paths = [Path(a) for a in args if a.endswith((".json", ".jsonl"))]
    paths = [p for p in paths if p.exists()]
    if not paths:
        print("no traces found"); return 1

    fig, axes = plt.subplots(len(paths), 4, figsize=(21, 4.6 * len(paths)), squeeze=False)
    for row, path in enumerate(paths):
        d = load(path)
        if not d:
            continue
        t = np.array([r[0] for r in d]) / 1000.0
        x = np.array([r[1] for r in d], dtype=float)
        y = np.array([r[2] for r in d], dtype=float)
        hp = np.array([r[3] if r[3] is not None else np.nan for r in d], dtype=float)
        cs = np.array([r[4] if r[4] is not None else np.nan for r in d], dtype=float)
        acts = [r[5] for r in d]
        spawn = (x[0], y[0])
        dist = np.hypot(x - spawn[0], y - spawn[1])

        ax = axes[row][0]; draw_map(ax)
        ax.scatter(x, y, c=t, s=3, cmap="viridis")
        ax.plot(*spawn, "o", ms=9, color="lime", markeredgecolor="k", label="spawn")
        ax.set_title(f"{path.name}: path (colour = time)", fontsize=10)
        if row == 0:
            ax.legend(fontsize=7, loc="upper left")

        ax = axes[row][1]; draw_map(ax)
        if any(a for a in acts):
            styles = {"move": ("tab:green", "."), "attack": ("tab:red", "x"),
                      "cast": ("magenta", "*"), "noop": ("grey", ",")}
            for name, (col, mk) in styles.items():
                sel = [i for i, a in enumerate(acts) if a == name]
                if sel:
                    ax.plot(x[sel], y[sel], mk, color=col, ms=4, alpha=0.5,
                            label=f"{name} ({len(sel)})")
            ax.legend(fontsize=7, loc="upper left")
            ax.set_title("where each action was pressed", fontsize=10)
        else:
            ax.set_title("(no action labels in this record)", fontsize=10)

        ax = axes[row][2]
        ax.plot(t, dist, lw=1.2)
        ax.axhline(np.hypot(CONTACT[0] - spawn[0], CONTACT[1] - spawn[1]),
                   color="gold", ls="--", label="distance to wave contact")
        ax.axhline(np.hypot(BLUE_T[0] - spawn[0], BLUE_T[1] - spawn[1]),
                   color="tab:blue", ls=":", label="distance to own turret")
        ax.set_title(f"distance from spawn (max {dist.max():,.0f})", fontsize=10)
        ax.set_xlabel("game time (s)"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

        ax = axes[row][3]
        ax.plot(t, hp, color="tab:red", lw=1, label="HP")
        ax.set_xlabel("game time (s)"); ax.set_ylabel("HP", color="tab:red")
        ax2 = ax.twinx()
        if not np.all(np.isnan(cs)):
            ax2.plot(t, cs, color="tab:green", lw=1.6, label="CS")
            ax2.set_ylabel("CS", color="tab:green")
        ax.set_title("HP and CS", fontsize=10); ax.grid(alpha=0.3)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
