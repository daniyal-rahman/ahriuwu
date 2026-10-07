"""Differential test of the native lane-slice tick against the JAX world (top lane, champions idle).

Every tick: import the JAX state into a native env, step it natively, and compare with the JAX state one tick later
(per-tick error, no drift). ``--free`` also runs the native env on its own from 0:00 and reports when it parts from
JAX. Run on CPU (both sides float32, JAX_PLATFORMS=cpu).

    python -m ops.native.diff_tick --ticks 2400 [--free] [--every 300]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "native" / "python"))

SKIP = {"memo_route", "memo_anchor", "ev_n", "ev_src", "ev_dst", "rec_n", "rec"}
# Champion rows the slice does not simulate (idle champions: their route anchors steer by the move goal).
CHAMPION_ROWS = {"route_anchor", "ad", "aspd", "ms", "armor", "mr", "range", "windup", "hp", "max_hp"}
ULP_REL = 1e-6                    # float differences at or below this relative size are rounding (reported apart)


def compare(world, got: dict, want: dict) -> dict:
    out = {}
    n = world.cfg.n_units
    for name, _, size in world.fields:
        if name in SKIP:
            continue
        a, b = got[name], want[name]
        if name in CHAMPION_ROWS:
            a, b = a[2:], b[2:]
        if a.dtype.kind == "f":
            bad = ~((a == b) | (np.isnan(a) & np.isnan(b)))
            if bad.any():
                fin = np.isfinite(a) & np.isfinite(b)
                rel = np.where(fin, np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), 1.0), np.inf)
                big = bad & ~(rel <= ULP_REL)
                gap = float(np.max(np.abs(a[bad & fin] - b[bad & fin]))) if (bad & fin).any() else float("inf")
                idx = np.flatnonzero(big if big.any() else bad)[:6]
                out[name] = {"n": int(bad.sum()), "rounding": not big.any(), "max_abs": gap, "at": idx.tolist(),
                             "got": a[idx].tolist(), "want": b[idx].tolist()}
        else:
            bad = a != b
            if bad.any():
                idx = np.flatnonzero(bad)[:6]
                out[name] = {"n": int(bad.sum()), "at": idx.tolist(), "got": a[idx].tolist(), "want": b[idx].tolist()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ticks", type=int, default=2400)
    ap.add_argument("--free", action="store_true")
    ap.add_argument("--every", type=int, default=300)
    ap.add_argument("--max-report", type=int, default=40)
    args = ap.parse_args()
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax

    from lanerl_jax.modern import world as MS
    from lanerl_jax.modern.jax_cache import enable_compile_cache
    from ops.modern.golden import build
    import lanesim as LS
    enable_compile_cache()
    cfg = build("top")
    world = LS.NativeWorld(cfg)
    step = jax.jit(lambda s: MS.step(s, MS.no_orders(), cfg)[0])
    s = MS.init_state(cfg)
    free = LS.env_from_state(world, s) if args.free else None
    totals, reported, first_free = {}, 0, None
    for t in range(args.ticks):
        s1 = step(s)
        env = LS.env_from_state(world, s)
        stats = world.step(env)
        want = LS.env_from_state(world, s1)
        diff = compare(world, env, want)
        for k, v in diff.items():
            key = k + (" (rounding)" if v.get("rounding") else "")
            totals.setdefault(key, [0, t])[0] += 1
        serious = {k: v for k, v in diff.items() if not v.get("rounding")}
        if serious and reported < args.max_report:
            print(json.dumps({"tick": t, "game_s": float(s1.t), "diff": serious}), flush=True)
            reported += 1
        if free is not None:
            world.step(free)
            fd = compare(world, free, want)
            if fd and first_free is None:
                first_free = t
                print(json.dumps({"free_run_parts_at": t, "fields": sorted(fd)}), flush=True)
        if (t + 1) % args.every == 0:
            alive = int(np.sum(np.asarray(s1.alive) & (np.asarray(s1.kind) == 2)))
            print(json.dumps({"tick": t + 1, "minions_alive": alive, "stats": stats.tolist(),
                              "ticks_with_diff": {k: v[0] for k, v in totals.items()}}), flush=True)
        s = s1
    print(json.dumps({"done": args.ticks, "fields_ever_differing": {k: {"ticks": v[0], "first": v[1]}
                                                                     for k, v in totals.items()},
                      "free_run_parts_at": first_free}))


if __name__ == "__main__":
    main()
