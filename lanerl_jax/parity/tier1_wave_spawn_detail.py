"""Task 2 follow-up: characterise EVERY wave-spawn mismatch on a fresh trace,
not just the first few -- after the confirmed timing fix
(`replay_wave_states`'s pairing + `sim/step.py`'s `t_now`) only moved
1551/1551 -> 1447/1551 mismatched on the full 420s run, so most of the
mismatch is NOT (or not only) the timing bug. For every mismatching tick,
records the nearest pre-tick same-group candidate distance for each
unmatched real ("new") entity -- if that distance clusters far above the
8-unit match radius, the harness's population-count number is being
corrupted by something other than a raw wave-spawn timing bug (e.g. a
same-group re-identification failure for tightly-spaced marching minions).

    sbatch slurm/parity_g1.sbatch python -m lanerl_jax.parity.tier1_wave_spawn_detail
"""
from __future__ import annotations

import collections
import math
from pathlib import Path

import numpy as np


def main() -> None:
    import jax.numpy as jnp
    from lanerl_jax.data.patch import load_patch
    from lanerl_jax.parity.diff import LANE_KINDS, _match_group
    from lanerl_jax.parity.inject import inject_snapshot, replay_wave_states
    from lanerl_jax.parity.one_step import compare_one_tick, record_idle_trace
    from lanerl_jax.parity.trace import load_trace
    from lanerl_jax.sim.init import TOP_LANE_PATH, lane_params
    from lanerl_jax.sim.profiles import PROFILES
    from lanerl_jax.parity.tier1_full import FIRST_WAVE_MS

    out = Path("lanerl_jax/runs/tier1_wave_spawn_detail")
    log = record_idle_trace(out, game_seconds=420.0, port_base=52300)
    print(f"recorded: {log}", flush=True)
    trace = load_trace(log)
    snaps = [s for s in trace.snapshots if s.t_ms >= FIRST_WAVE_MS]
    print(f"{len(trace)} total, {len(snaps)} after first wave", flush=True)

    patch = load_patch()
    params = lane_params(patch)
    wave_states = replay_wave_states(snaps)
    lane_path = jnp.asarray(np.asarray(TOP_LANE_PATH, np.float32))
    MATCH_RADIUS_Q = 16 * 8

    def by_group(entities, key_fn):
        out = {}
        for e in entities:
            k = key_fn(e)
            if k is None:
                continue
            out.setdefault(k, []).append(e)
        return out

    n_spawn_ticks = 0
    n_mismatch = 0
    nearest_dists = []          # for every unmatched "new" real entity
    undercounts = 0             # n_real_new > n_sim_new
    overcounts = 0              # n_real_new < n_sim_new
    exact_double_mismatch_examples = []

    n = len(snaps) - 1
    for i in range(n):
        if i % 2000 == 0:
            print(f"  ... {i}/{n}", flush=True)
        sn, sn1 = snaps[i], snaps[i + 1]
        dt = sn1.t_ms - sn.t_ms
        if dt <= 0 or dt > 34:
            continue
        state_n, report = inject_snapshot(sn, wave_states[i], params, PROFILES)
        tr = compare_one_tick(state_n, report.notes, sn1, params, lane_path)

        real_all = by_group(sn1.entities, lambda e: (e.kind, e.team)
                            if e.kind in LANE_KINDS and e.team is not None else None)
        pre_by_group = by_group(report.notes, lambda nt: (nt.kind, nt.team))

        for sp in tr.spawns:
            if sp.kind != "LaneMinion":
                continue
            if sp.n_real_new == 0 and sp.n_sim_new == 0:
                continue
            n_spawn_ticks += 1
            if sp.n_real_new == sp.n_sim_new:
                continue
            n_mismatch += 1
            if sp.n_real_new > sp.n_sim_new:
                undercounts += 1
            else:
                overcounts += 1

            key = ("LaneMinion", sp.team)
            pre_list = pre_by_group.get(key, [])
            real_list = real_all.get(key, [])
            left = [nt.entity for nt in pre_list]
            _matched, _only_l, only_r = _match_group(left, real_list, MATCH_RADIUS_Q)
            for e in only_r:
                if left:
                    d = min(math.hypot(e.x - a.x, e.y - a.y) for a in left)
                else:
                    d = float("inf")
                nearest_dists.append(d)
                if len(exact_double_mismatch_examples) < 30 and d > 8.0:
                    exact_double_mismatch_examples.append(
                        (sn.t_ms, sp.team, sp.n_real_new, sp.n_sim_new, round(d, 2)))

    print(f"\nscanned {n} tick-pairs")
    print(f"LaneMinion spawn-ticks: {n_spawn_ticks}  mismatched: {n_mismatch} "
          f"(undercount={undercounts} overcount={overcounts})")
    if nearest_dists:
        d = np.asarray(nearest_dists)
        print(f"\nnearest-pre-tick-candidate distance for each unmatched 'new' "
              f"entity (n={len(d)}):")
        print(f"  <= 8 units (i.e. should have matched -- a REAL count "
              f"disagreement): {np.sum(d <= 8.0)} ({100*np.mean(d<=8.0):.1f}%)")
        print(f"  >  8 units (matching failure, not a count disagreement): "
              f"{np.sum(d > 8.0)} ({100*np.mean(d>8.0):.1f}%)")
        print(f"  median={np.median(d):.2f} p10={np.percentile(d,10):.2f} "
              f"p50={np.percentile(d,50):.2f} p90={np.percentile(d,90):.2f} "
              f"max={d[np.isfinite(d)].max() if np.isfinite(d).any() else float('nan'):.2f}")
    print("\nexamples with nearest-candidate > 8 units "
          "(t_ms, team, n_real_new, n_sim_new, nearest_dist):")
    for row in exact_double_mismatch_examples:
        print(f"  {row}")


if __name__ == "__main__":
    main()
