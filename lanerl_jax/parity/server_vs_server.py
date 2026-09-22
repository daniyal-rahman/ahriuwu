"""
Score two server recordings against each other, with the parity metrics.

WHY
---
`ORDER-003` says some of gate 1's residual can never reach zero: the server
updates units serially (`ObjectManager.Update`), so each unit's view depends
on where it falls in an arbitrary iteration order, and the vectorised port
updates everything from one snapshot. That is a design trade and it puts a
floor under every order-sensitive field.

The size of that floor has only ever been asserted. Five asserted mechanisms
on this project have been refuted by their first direct measurement, so an
unmeasured floor is not evidence -- it is an argument that happens to be
convenient, because it excuses any residual it is applied to.

`lanerl/patch_shuffle.py` permutes the server's per-tick update order behind
`LANERL_SHUFFLE_ORDER`. Running the server against ITSELF that way puts the
floor in the same units as the parity report:

    sim residual <= shuffled residual  ->  FLOOR. Stop investigating.
    sim residual >  shuffled residual  ->  a real defect. Worth the forensics.

And it answers the larger question first: if shuffling barely moves anything,
update order does not matter, the vectorised tick never cost fidelity, and
most of gate 1 can close on the spot.

WHAT IS COMPARED
----------------
The per-unit `LANERL_INTERNAL` stream, joined on `(t_ms, net_id)`, which is
stable across a shuffle because neither the clock nor NetId depends on update
order. Only ticks and units present in BOTH recordings are scored, and the
coverage is reported -- a floor measured over a shrinking intersection would
flatter itself.

`hp` and `move_order` are NOT in this stream and so are not scored here.
Six of the eight scored families are, including `target`, which is the field
`ORDER-003` is actually about.

USAGE
    python -m lanerl_jax.parity.server_vs_server --a <normal.log> --b <shuffled.log>
"""

from __future__ import annotations

import argparse
import collections
import re
import struct
from pathlib import Path

PosQ = 16
StatQ = 1024

_LINE = re.compile(r"LANERL_INTERNAL t=(-?\d+) ai id=(\d+) (.*)")

#: field -> (dump key, how to compare)
_FIELDS = {
    "target": ("target", "exact"),
    "waypoints": ("wps", "wpcount"),
    "aa_cooldown": ("aacd", "exact"),
    "is_attacking": ("attacking", "exact"),
    "has_auto_attacked": ("hasaa", "exact"),
    "aa_state": ("aastate", "exact"),
}


def _f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def load(path: Path) -> dict:
    out = {}
    with path.open(errors="replace") as fh:
        for line in fh:
            if "LANERL_INTERNAL" not in line:
                continue
            m = _LINE.search(line)
            if m is None:
                continue
            t, nid, rest = int(m.group(1)), int(m.group(2)), m.group(3)
            f = {}
            for tok in rest.split():
                k, _, v = tok.partition("=")
                if v:
                    f[k] = v
            out[(t, nid)] = f
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--a", type=Path, required=True, help="reference recording")
    ap.add_argument("--b", type=Path, required=True, help="the other one")
    ap.add_argument("--from-ms", type=int, default=90_000)
    a = ap.parse_args(argv)

    A, B = load(a.a), load(a.b)
    keys = [k for k in (A.keys() & B.keys()) if k[0] >= a.from_ms]
    only_a = len([k for k in A if k[0] >= a.from_ms]) - len(keys)
    only_b = len([k for k in B if k[0] >= a.from_ms]) - len(keys)
    print(f"A: {len(A)} unit-ticks   B: {len(B)} unit-ticks")
    print(f"shared at t>={a.from_ms}: {len(keys)}   "
          f"A-only {only_a}   B-only {only_b}")
    if not keys:
        raise SystemExit("no shared unit-ticks -- are these the same scenario?")

    # A unit that exists in one run and not the other at the same tick is
    # itself an order effect (spawn/death landing on a different tick), so it
    # is reported rather than silently dropped.
    per_kind = collections.defaultdict(lambda: collections.Counter())
    totals = collections.defaultdict(lambda: collections.Counter())

    for k in keys:
        fa, fb = A[k], B[k]
        kind = fa.get("kind", "?")
        totals[kind]["n"] += 1
        for name, (key, how) in _FIELDS.items():
            va, vb = fa.get(key), fb.get(key)
            if va is None or vb is None:
                continue
            totals[kind][name + "_n"] += 1
            if how == "wpcount":
                bad = va.count(";") != vb.count(";")
            else:
                bad = va != vb
            if bad:
                per_kind[kind][name] += 1
        # position, at the parity report's own L-inf tolerance
        try:
            ax, ay = _f32(int(fa["xbits"])), _f32(int(fa["ybits"]))
            bx, by = _f32(int(fb["xbits"])), _f32(int(fb["ybits"]))
        except (KeyError, ValueError):
            continue
        totals[kind]["position_n"] += 1
        if max(abs(ax - bx), abs(ay - by)) > 1.0 / PosQ:
            per_kind[kind]["position_linf"] += 1

    print()
    for kind in sorted(totals):
        n = totals[kind]["n"]
        print(f"== {kind}: {n} shared unit-ticks ==")
        rows = sorted(set(list(_FIELDS) + ["position_linf"]))
        for name in rows:
            denom = totals[kind].get(
                (name if name == "position_linf" else name) + "_n", 0)
            if not denom:
                continue
            miss = per_kind[kind].get(name, 0)
            print(f"   {name:<20} {miss:>7} / {denom:<8} "
                  f"{100 * (denom - miss) / denom:6.2f}% agree")
        print()

    print("This is the FLOOR, in the parity report's own units. A simulator\n"
          "residual at or below the number for its field is order-dependence,\n"
          "not a defect, and `GATE1-004` says to mark it FLOOR and stop.")


if __name__ == "__main__":
    main()
