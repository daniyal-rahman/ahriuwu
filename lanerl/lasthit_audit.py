#!/usr/bin/env python
"""Where do the missed minions go? Killable, in range, and actually attacked.

CS alone cannot say why an agent farms badly, and the obvious ceiling
experiment does not work here: removing the opponent made CS go DOWN (37.5
uncontested against 45.4 versus the bot), because with nobody killing your
minions your wave overruns theirs and your own minions take the kills. So
"uncontested CS" is not an upper bound.

What the agent was DOING is measurable instead. For every decision this
classifies the nearest killable enemy minion -- killable meaning current hp is
at or below one auto-attack -- into:

    in_range_attacked   the agent issued an attack on it. Working as intended.
    in_range_missed     it was killable, it was inside attack range, and the
                        agent did something else. A TARGETING or timing
                        failure.
    out_of_range        it was killable and the agent was too far away. A
                        POSITIONING failure, and a different fix entirely.
    none_killable       nothing was in a killable state this decision.

The split is the whole point: "in range but not attacked" and "killable but
out of position" have nothing in common as problems, and CS cannot tell them
apart.

Also reported, because the uncontested runs suggested it: how much of the game
the agent spends inside attack range at all, and its mean speed. A Garen
sustaining ~290 units/s of a 345 maximum for ten minutes is not standing still
to last-hit anything, whatever its target head is doing.

    python lanerl/lasthit_audit.py --capture <viz_capture.jsonl>
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", required=True)
    ap.add_argument("--side", default="blue")
    ap.add_argument("--aa-range", type=float, default=190.0,
                    help="Garen's attack range plus the minion's selectable radius")
    ap.add_argument("--aa-damage", type=float, default=78.14,
                    help="level-1 AD with the rune page; the kill window widens later")
    args = ap.parse_args()

    recs = []
    for line in Path(args.capture).read_text(errors="ignore").splitlines():
        if line.strip():
            try:
                recs.append(json.loads(line))
            except Exception:
                pass
    if not recs:
        print("no records")
        return 2

    verdict = Counter()
    in_range_any = 0
    speeds = []
    prev = None
    nearest_hist = Counter()

    for r in recs:
        s = (r.get("sides") or {}).get(args.side)
        if not s:
            continue
        cx, cy = s["champ_world"]
        if prev is not None:
            dt = max(1e-6, (r["t"] - prev[0]) / 1000.0)
            speeds.append(math.dist((cx, cy), prev[1]) / dt)
        prev = (r["t"], (cx, cy))

        order = s.get("order") or {}
        attacking = order.get("t") == "attack"
        attacked_id = order.get("id")

        enemies = [e for e in (s.get("slots") or [])
                   if e.get("etype") == "minion" and e.get("team") != 100]
        if any(math.dist((cx, cy), e["world"]) <= args.aa_range for e in enemies):
            in_range_any += 1

        killable = [e for e in enemies
                    if (e.get("hp_frac") or 0) * 500.0 <= args.aa_damage]
        if not killable:
            verdict["none_killable"] += 1
            continue
        nearest = min(killable, key=lambda e: math.dist((cx, cy), e["world"]))
        d = math.dist((cx, cy), nearest["world"])
        nearest_hist[int(d // 100) * 100] += 1
        if d > args.aa_range:
            verdict["out_of_range"] += 1
        elif attacking and attacked_id == nearest["netid"]:
            verdict["in_range_attacked"] += 1
        else:
            verdict["in_range_missed"] += 1

    n = sum(verdict.values()) or 1
    print(f"\n=== {len(recs)} decisions, side {args.side} ===")
    for k in ("in_range_attacked", "in_range_missed", "out_of_range", "none_killable"):
        print(f"  {k:20s} {verdict[k]:7d}  {100*verdict[k]/n:5.1f}%")
    print(f"\n  any enemy minion within {args.aa_range:.0f}u: "
          f"{100*in_range_any/len(recs):.1f}% of decisions")
    if speeds:
        speeds.sort()
        print(f"  speed: mean {sum(speeds)/len(speeds):7.1f} u/s   "
              f"median {speeds[len(speeds)//2]:7.1f}   (Garen base move speed 345)")
    print("\n  distance to the nearest KILLABLE minion:")
    for band in sorted(nearest_hist)[:8]:
        print(f"    {band:5d}-{band+99:<5d} {nearest_hist[band]:7d}")
    print("\nin_range_missed is a targeting/timing problem; out_of_range is a "
          "positioning problem. They do not share a fix.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
