"""
Break gate 3's outcome gap into factors that can be COUNTED on both sides.

WHY
---
Gate 3 reports an outcome difference -- most recently sim `cs=7 deaths=2`
against server `cs=4 deaths=1`, and a level gap. An outcome gap is not a
mechanism, and the standing temptation is to pick the most interesting-
sounding cause and build it. That is how this project nearly spent a block
on `SmoothPath`: gate 3's own flip test then showed that substituting the
server's champion POSITION into the oracle's inputs does not change the
decision, while substituting the server's MINION LIST does. Position was off
by 70 units and was not the mechanism.

So decompose first, into factors each of which is a number in both streams:

    level over time          -- the XP gap, which is the systematic bias
    visible minion count     -- the input the flip test actually implicated
    champion alive/dead      -- time spent dead cannot earn XP
    distance to the wave     -- XP requires being in range when a minion dies
    decision mix             -- attack/move/hold, to see WHERE the two differ

Whichever factor separates first is the mechanism. Nothing is built until one
does.

This reads the streams `gate3_first_divergence --save` already writes, so it
costs no server time.

USAGE
    python -m lanerl_jax.parity.gate3_decompose --streams <g3_streams.json>
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path


def _series(side: dict, key: str):
    """decision index -> value, as a sorted list of (idx, value)."""
    out = []
    for k, v in side["inputs"].items():
        out.append((int(k), v))
    out.sort()
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--streams", type=Path, required=True)
    ap.add_argument("--bucket", type=int, default=1500,
                    help="decisions per reporting bucket (30 Hz, so 1500 = 50 s)")
    a = ap.parse_args(argv)

    d = json.loads(a.streams.read_text())
    sim, srv = d["sim"], d["server"]

    S = {"sim": _series(sim, "inputs"), "server": _series(srv, "inputs")}
    dec = {"sim": sim["decisions"], "server": srv["decisions"]}

    print("-- factor decomposition, by decision bucket --")
    print("   Each row is a bucket of decisions. `lvl` is the champion's level\n"
          "   at the END of the bucket (the XP proxy), `mins` the mean number of\n"
          "   minions the oracle could see, `d_wave` the mean distance to the\n"
          "   nearest visible minion, and the last three the decision mix.\n")
    hdr = (f"   {'bucket':>8} {'side':>7} {'n':>6} {'lvl':>4} {'mins':>6} "
           f"{'d_wave':>8} {'atk':>6} {'mv':>5} {'hold':>6}")
    print(hdr)

    buckets = collections.defaultdict(lambda: collections.defaultdict(list))
    for side in ("sim", "server"):
        for idx, v in S[side]:
            b = idx // a.bucket
            buckets[b][side].append((idx, v))

    first_level_gap = None
    for b in sorted(buckets):
        for side in ("sim", "server"):
            rows = buckets[b][side]
            if not rows:
                continue
            n = len(rows)
            lvl = rows[-1][1]["level"]
            mins = sum(len(v["minions"]) for _i, v in rows) / n
            ds = []
            for _i, v in rows:
                if not v["minions"]:
                    continue
                ds.append(min(math.hypot(m[0] - v["cx"], m[1] - v["cy"])
                              for m in v["minions"]))
            dw = sum(ds) / len(ds) if ds else float("nan")
            mix = collections.Counter(
                dec[side].get(str(i), ("hold", None))[0] for i, _v in rows)
            print(f"   {b * a.bucket:>8} {side:>7} {n:>6} {lvl:>4} "
                  f"{mins:>6.2f} {dw:>8.1f} {mix['attack']:>6} "
                  f"{mix['move']:>5} {mix['hold']:>6}")
        # first bucket where the levels differ at all
        if first_level_gap is None:
            ls = {s: (buckets[b][s][-1][1]["level"] if buckets[b][s] else None)
                  for s in ("sim", "server")}
            if ls["sim"] is not None and ls["server"] is not None \
                    and ls["sim"] != ls["server"]:
                first_level_gap = (b * a.bucket, ls["sim"], ls["server"])
        print()

    print("-- where the factors separate --")
    if first_level_gap:
        i, a_, b_ = first_level_gap
        print(f"   LEVEL first differs at decision {i}: sim {a_} vs server {b_}")
    else:
        print("   LEVEL never differs in this run")

    # the oracle's own inputs: how often does each side even SEE a wave?
    for side in ("sim", "server"):
        rows = S[side]
        empty = sum(1 for _i, v in rows if not v["minions"])
        print(f"   {side}: {empty}/{len(rows)} decisions with NO visible minion "
              f"({100 * empty / len(rows):.1f}%)")

    print("\n   A factor that separates EARLY and stays separated is the\n"
          "   mechanism. One that only separates after the outcome gap opens\n"
          "   is downstream of it. Build nothing until one factor does the\n"
          "   former -- gate 3's flip test already ruled out champion position\n"
          "   and AD, and implicated the minion list.")


if __name__ == "__main__":
    main()
