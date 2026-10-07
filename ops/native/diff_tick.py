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
# The lane slice's fields (``--fields slice``), and the champion rows of them it does not simulate.
SLICE = set(['alive', 'armor', 'att_cooldown_left', 'att_target', 'att_target_seq', 'att_windup_left', 'attack_damage', 'attack_range', 'attack_speed', 'bounty_gold', 'bounty_level', 'bounty_xp', 'cc_champion_cc_until', 'cc_knockup_until', 'cc_root_until', 'cc_silence_until', 'cc_slow', 'cc_slow_until', 'cc_stun_until', 'econ_level', 'game_over', 'hp', 'kind', 'lane_ai_champion_aggro', 'lane_ai_engaged', 'lane_ai_first_wave', 'lane_ai_ignore_until', 'lane_ai_lane', 'lane_ai_last_attack', 'lane_ai_seq', 'lane_ai_since_attack', 'lane_ai_sweep_timer', 'lane_ai_target', 'lane_ai_target_priority', 'lane_ai_target_seq', 'lane_ai_warm_stacks', 'lane_ai_warm_until', 'lane_ai_waypoint', 'magic_resist', 'max_hp', 'missile_speed', 'missiles_alive', 'missiles_cast_id', 'missiles_crit', 'missiles_dst', 'missiles_dst_seq', 'missiles_dtype', 'missiles_flags', 'missiles_raw', 'missiles_speed', 'missiles_src', 'missiles_x', 'missiles_y', 'move_speed', 'next_seq', 'prev_damage_matrix', 'radius', 'reveal_until', 'reveal_x', 'reveal_y', 'route_anchor', 'spawn_seq', 'spawn_supers', 'spawn_time', 'spawn_unit', 'spawn_wave', 'sub', 't', 'targetable', 'team', 'tick', 'towers_first_turret_taken', 'towers_is_structure', 'towers_lane', 'towers_prereq', 'towers_targetable', 'towers_team', 'towers_turret_backdoor_until', 'towers_turret_bulwark_until', 'towers_turret_growth_active', 'towers_turret_growth_since', 'towers_turret_hp', 'towers_turret_max_hp', 'towers_turret_plates', 'towers_turret_respawn_at', 'towers_turret_tier', 'towers_turret_warm_stacks', 'towers_turret_warm_until', 'visible', 'windup', 'winner', 'x', 'y'])
CHAMPION_ROWS = {"route_anchor", "attack_damage", "attack_speed", "move_speed", "armor", "magic_resist",
                 "attack_range", "windup", "hp", "max_hp"}
ULP_REL = 1e-6                    # float differences at or below this relative size are rounding (reported apart)


def compare(world, got: dict, want: dict, only=None) -> dict:
    out = {}
    for name, _ in world.fields:
        if name in SKIP or (only is not None and name not in only):
            continue
        a, b = got[name], want[name]
        if only is not None and name in CHAMPION_ROWS:
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
    ap.add_argument("--fields", choices=("slice", "all"), default="all")
    ap.add_argument("--orders", choices=("none", "scripted", "chaos"), default="none",
                    help="champion orders: none, the JAX bench's scripted ones, or golden's chaos orders")
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
    from ops.modern.bench import scripted_orders
    from ops.modern.golden import chaos_orders
    lane_mid = cfg.lane_path[cfg.lane_path.shape[0] // 2]
    key = jax.random.PRNGKey(1234)
    make_orders = {"none": lambda s, t: MS.no_orders(), "scripted": lambda s, t: scripted_orders(s, lane_mid),
                   "chaos": lambda s, t: chaos_orders(s, jax.random.fold_in(key, t), lane_mid)}[args.orders]
    orders_at = jax.jit(make_orders)
    step = jax.jit(lambda s, o: MS.step(s, o, cfg)[0])
    only = SLICE if args.fields == "slice" else None
    s = MS.init_state(cfg)
    free = LS.env_from_state(world, s) if args.free else None
    totals, reported, first_free = {}, 0, None
    for t in range(args.ticks):
        o = orders_at(s, t)
        s1 = step(s, o)
        no = LS.orders_from(o)
        env = LS.env_from_state(world, s)
        stats = world.step(env, no)
        want = LS.env_from_state(world, s1)
        diff = compare(world, env, want, only)
        for k, v in diff.items():
            key = k + (" (rounding)" if v.get("rounding") else "")
            totals.setdefault(key, [0, t])[0] += 1
        serious = {k: v for k, v in diff.items() if not v.get("rounding")}
        if serious and reported < args.max_report:
            print(json.dumps({"tick": t, "game_s": float(s1.t), "diff": serious}), flush=True)
            reported += 1
        if free is not None:
            world.step(free, LS.orders_from(make_orders(LS.state_from_env(world, free, s), t)))
            fd = compare(world, free, want, only)
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
