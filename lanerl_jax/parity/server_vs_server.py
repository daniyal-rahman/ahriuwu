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


_CURVE_FIELDS = ("target", "aacd", "attacking", "hasaa", "wps", "xbits", "ybits")


def _first_divergence(A: dict, B: dict):
    """The earliest tick on which any shared unit differs on any scored field.

    Measured, not derived. `LanerlShuffle`'s tick counter starts at process
    start and includes pre-game ticks, so `LANERL_SHUFFLE_FROM` does not map
    to game time by any fixed offset -- deriving it cost one run and an empty
    curve.
    """
    best = None
    for k in A.keys() & B.keys():
        if best is not None and k[0] >= best:
            continue
        fa, fb = A[k], B[k]
        if any(fa.get(f) != fb.get(f) for f in _CURVE_FIELDS):
            best = k[0]
    return best


def _curve_at(A: dict, B: dict, t0: int, n_ticks: int = 25) -> None:
    """Divergence per tick from the first divergent tick.

    Offset 0 is the ONE-STEP order floor: both runs are byte-identical before
    it, so the two states entering that tick are the same state. That is the
    only floor commensurable with a tier-1 residual, which also re-injects
    state every tick. Later offsets are contaminated by the divergence offset
    0 created and show the amplification rate, which is what decides how many
    seeds an outcome comparison needs.
    """
    ts = sorted({k[0] for k in A if k[0] >= t0})[:n_ticks]
    pop = collections.Counter()
    for (t, _n), f in A.items():
        if f.get("kind") == "LaneMinion":
            pop[t] += 1
    print(f"\n-- divergence from the first divergent tick, t={t0} ms --")
    print(f"   {'off':>4} {'t_ms':>8} {'minions':>8} {'target':>7} "
          f"{'aa_cd':>7} {'attack':>7} {'wps':>6} {'pos':>6}")
    for i, t in enumerate(ts):
        d = collections.Counter()
        n = 0
        for k in ((t, nid) for nid in ()):
            pass
        for k, fa in A.items():
            if k[0] != t or fa.get("kind") != "LaneMinion" or k not in B:
                continue
            fb = B[k]
            n += 1
            if fa.get("target") != fb.get("target"):
                d["target"] += 1
            if fa.get("aacd") != fb.get("aacd"):
                d["aacd"] += 1
            if fa.get("attacking") != fb.get("attacking"):
                d["attacking"] += 1
            if fa.get("wps", "").count(";") != fb.get("wps", "").count(";"):
                d["wps"] += 1
            if (fa.get("xbits") != fb.get("xbits")
                    or fa.get("ybits") != fb.get("ybits")):
                d["pos"] += 1
        print(f"   {i:>4} {t:>8} {n:>8} {d['target']:>7} {d['aacd']:>7} "
              f"{d['attacking']:>7} {d['wps']:>6} {d['pos']:>6}")


def _curve(A: dict, B: dict, t0: int, n_offsets: int = 24) -> None:
    """Disagreement by tick offset from the first shuffled tick.

    Offset 0 is the number that matters. Both runs are byte-identical before
    it, so the two states going into that tick are the SAME state -- which
    makes it the one-step order-dependence floor, directly comparable to a
    tier-1 residual. Everything after offset 0 is contaminated by the
    divergence offset 0 created, and is reported only to show the growth rate.
    """
    step = 1000.0 / 60.0
    print(f"\n-- divergence by tick offset from t={t0} ms "
          f"(offset 0 = the one-step floor) --")
    print(f"   {'off':>4} {'t_ms':>8} {'minions':>8} {'target':>8} "
          f"{'aa_cd':>7} {'attacking':>10} {'wps':>6}")
    for off in range(n_offsets):
        t = int(round(t0 + off * step))
        # the dump's clock is integer ms, so accept the nearest tick
        keys = [k for k in A if abs(k[0] - t) <= 1 and k in B]
        mins = [k for k in keys if A[k].get("kind") == "LaneMinion"]
        if not mins:
            continue
        d = collections.Counter()
        for k in mins:
            fa, fb = A[k], B[k]
            if fa.get("target") != fb.get("target"):
                d["target"] += 1
            if fa.get("aacd") != fb.get("aacd"):
                d["aacd"] += 1
            if fa.get("attacking") != fb.get("attacking"):
                d["attacking"] += 1
            if fa.get("wps", "").count(";") != fb.get("wps", "").count(";"):
                d["wps"] += 1
        print(f"   {off:>4} {t:>8} {len(mins):>8} {d['target']:>8} "
              f"{d['aacd']:>7} {d['attacking']:>10} {d['wps']:>6}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--a", type=Path, required=True, help="reference recording")
    ap.add_argument("--b", type=Path, required=True, help="the other one")
    ap.add_argument("--from-ms", type=int, default=90_000)
    ap.add_argument("--first-divergence", action="store_true",
                    help="find the first tick where the two runs differ, "
                        "EMPIRICALLY, and report the divergence curve from "
                        "there. Use this rather than computing the tick from "
                        "LANERL_SHUFFLE_FROM: that counter starts at process "
                        "start and includes pre-game ticks, so it does not map "
                        "to game time by any fixed offset. Assuming it did "
                        "produced an empty curve and cost a run.")
    ap.add_argument("--curve-from-ms", type=int, default=None,
                    help="the game time at which B's shuffle first fires "
                        "(LANERL_SHUFFLE_FROM x 1000/60). Reports disagreement "
                        "per tick offset from it. Offset 0 is a TRUE ONE-STEP "
                        "sample -- identical input state, one tick, two update "
                        "orders -- which is the only floor commensurable with a "
                        "tier-1 residual. Later offsets show how fast a single "
                        "tick of order difference amplifies.")
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

    if a.first_divergence:
        t0 = _first_divergence(A, B)
        if t0 is None:
            print("\nthe two runs never diverge on any scored field")
        else:
            _curve_at(A, B, t0)
    elif a.curve_from_ms is not None:
        _curve(A, B, a.curve_from_ms)

    print()
    for kind in sorted(totals):
        n = totals[kind]["n"]
        print(f"== {kind}: {n} shared unit-ticks ==")
        rows = sorted(set(list(_FIELDS) + ["position_linf"]))
        for name in rows:
            # The denominator for `position_linf` is written as `position_n`
            # (see the increment above), so it needs the mapping. The previous
            # expression was `(name if name == "position_linf" else name)`,
            # which is `name` in BOTH branches -- it looked up `position_linf_n`,
            # got 0, and `continue`d. The position row was therefore never
            # printed at all, whatever the L-inf residual was, and `FLOOR-001`
            # quoted a position floor this tool could not have produced.
            denom_key = ("position" if name == "position_linf" else name) + "_n"
            denom = totals[kind].get(denom_key, 0)
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
