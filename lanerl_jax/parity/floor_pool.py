"""
Pool one-step order-floor samples across deferred-shuffle runs.

WHY
---
`FLOOR-001` measured the one-step order floor from three runs -- 60 minion
unit-ticks, 6 target events. That was enough to show the simulator sits
50-100x below the floor, and NOT enough to pin the floor: 6 events puts
`target` at roughly 4-20% with 95% confidence, and fields with zero events
are bounded only by the rule of three (<=5%).

A floor quoted to classify residuals has to be firmer than that, because
every FLOOR classification leans on it. This pools every deferred-shuffle
run into one estimate with an interval.

METHOD
------
Each run sets `LANERL_SHUFFLE_FROM=<tick>`, so it is byte-identical to the
reference until the first permuted tick. That tick -- and only that tick --
is a true one-step sample: the same input state, one tick, two update
orders. Later ticks are contaminated by the divergence the first one caused.

The first divergent tick is found EMPIRICALLY per run, because the shuffle
counter starts at process start and does not map to game time by any fixed
offset. Assuming it did once produced an empty curve.

Seeds are varied across runs so the result is not one permutation's quirk.

USAGE
    python -m lanerl_jax.parity.floor_pool --ref <reference.log> \
        --runs <shuffled1.log> <shuffled2.log> ...
"""

from __future__ import annotations

import argparse
import collections
import math
import re
from pathlib import Path

_LINE = re.compile(r"LANERL_INTERNAL t=(-?\d+) ai id=(\d+) (.*)")

#: scored field -> dump key. `wps` is compared by COUNT, as the parity report
#: does; `pos` by exact bits, which is stricter than the report's 1/16 L-inf
#: and therefore a conservative (larger) floor.
_FIELDS = {
    "target": "target", "aa_cooldown": "aacd", "is_attacking": "attacking",
    "has_auto_attacked": "hasaa", "waypoints": "wps", "position": "xbits",
    "hp": "hp", "move_order": "mo",
}


def load(path: Path, kinds=("LaneMinion",)) -> dict:
    out = {}
    with path.open(errors="replace") as fh:
        for line in fh:
            if "LANERL_INTERNAL" not in line:
                continue
            m = _LINE.search(line)
            if m is None:
                continue
            rest = m.group(3)
            f = {}
            for tok in rest.split():
                k, _, v = tok.partition("=")
                if v:
                    f[k] = v
            if f.get("kind") not in kinds:
                continue
            out[(int(m.group(1)), int(m.group(2)))] = f
    return out


def _wilson(k: int, n: int, z: float = 1.96):
    """Wilson score interval -- correct at k=0, unlike normal approximation."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def first_divergent_tick(A: dict, B: dict):
    best = None
    for k in A.keys() & B.keys():
        if best is not None and k[0] >= best:
            continue
        fa, fb = A[k], B[k]
        for key in set(_FIELDS.values()) | {"ybits"}:
            va, vb = fa.get(key), fb.get(key)
            if va is None or vb is None:
                continue
            if key == "wps":
                if va.count(";") != vb.count(";"):
                    best = k[0]
                    break
            elif va != vb:
                best = k[0]
                break
    return best


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    a = ap.parse_args(argv)

    A = load(a.ref)
    tot = collections.Counter()
    n_units = 0
    per_run = []

    for r in a.runs:
        B = load(r)
        t0 = first_divergent_tick(A, B)
        if t0 is None:
            per_run.append((r.parent.parent.name, None, 0, {}))
            continue
        keys = [k for k in A if k[0] == t0 and k in B]
        d = collections.Counter()
        for k in keys:
            fa, fb = A[k], B[k]
            for name, key in _FIELDS.items():
                va, vb = fa.get(key), fb.get(key)
                if va is None or vb is None:
                    continue
                if name == "waypoints":
                    bad = va.count(";") != vb.count(";")
                elif name == "position":
                    bad = (fa.get("xbits") != fb.get("xbits")
                           or fa.get("ybits") != fb.get("ybits"))
                else:
                    bad = va != vb
                if bad:
                    d[name] += 1
        for name in _FIELDS:
            # only count a field's denominator where BOTH sides publish it
            if any(k in B and _FIELDS[name] in A[k] and _FIELDS[name] in B[k]
                   for k in keys):
                tot[name + "_n"] += sum(
                    1 for k in keys
                    if _FIELDS[name] in A[k] and _FIELDS[name] in B[k])
            tot[name] += d[name]
        n_units += len(keys)
        per_run.append((r.parent.parent.name, t0, len(keys), dict(d)))

    print(f"pooled over {len(a.runs)} deferred-shuffle runs\n")
    print(f"   {'run':<14} {'first div t_ms':>14} {'minions':>8}  events")
    for name, t0, n, d in per_run:
        ev = ", ".join(f"{k}={v}" for k, v in sorted(d.items())) or "-"
        print(f"   {name:<14} {str(t0):>14} {n:>8}  {ev}")

    print(f"\n-- ONE-STEP ORDER FLOOR, pooled ({n_units} minion unit-ticks) --")
    print(f"   {'field':<20} {'events':>7} {'n':>7} {'rate':>8} "
          f"{'95% CI':>16}")
    for name in _FIELDS:
        n = tot.get(name + "_n", 0)
        if not n:
            print(f"   {name:<20} {'-':>7} {'0':>7} "
                  f"{'not published':>8}")
            continue
        k = tot[name]
        lo, hi = _wilson(k, n)
        print(f"   {name:<20} {k:>7} {n:>7} {100*k/n:>7.2f}% "
              f"{100*lo:>7.2f}-{100*hi:<7.2f}%")

    print("\n   Compare each rate against the simulator's one-step residual for\n"
          "   the same field. Below the CI's LOWER bound is unambiguously FLOOR.\n"
          "   Wilson interval, which is correct at zero events where the normal\n"
          "   approximation is not.")


if __name__ == "__main__":
    main()
