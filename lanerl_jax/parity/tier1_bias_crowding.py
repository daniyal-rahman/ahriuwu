"""Task 3 support: does the residual position_along_heading bias concentrate
where a unit has >=2 collision-overlapping neighbours at the pre-tick
position -- i.e. is it `sim/collision.py`'s already-booked "one push per unit
per tick" approximation (the server resolves collisions pair-by-pair,
sequentially, so one unit can be pushed several times in one update; this
sim applies one push, from the lowest-index neighbour, simultaneously for
all units) rather than an undiscovered tick-ordering bug?

Restricted to missile-free, trustworthy-movement ticks, per
docs/TIER1_POST_REORDER.md task 3. Records its own short trace (does not
depend on any other job's output) -- run via slurm, not the login node:

    sbatch slurm/parity_g1.sbatch python -m lanerl_jax.parity.tier1_bias_crowding
"""
from __future__ import annotations

import collections
import math
from pathlib import Path

import numpy as np

from .inject import inject_snapshot, replay_wave_states
from .one_step import POS_Q_UNIT, record_idle_trace
from .trace import load_trace
from ..sim.init import TOP_LANE_PATH, lane_params
from ..sim.profiles import PROFILES
from .tier1_full import FIRST_WAVE_MS


def main() -> None:
    import jax.numpy as jnp
    from ..data.patch import load_patch
    from .one_step import compare_one_tick

    out = Path("lanerl_jax/runs/tier1_bias_crowding")
    log = record_idle_trace(out, game_seconds=200.0, port_base=51700)
    print(f"recorded: {log}", flush=True)
    trace = load_trace(log)
    snaps = [s for s in trace.snapshots if s.t_ms >= FIRST_WAVE_MS]
    print(f"{len(trace)} total, {len(snaps)} after first wave", flush=True)

    patch = load_patch()
    params = lane_params(patch)
    wave_states = replay_wave_states(snaps)
    lane_path = jnp.asarray(np.asarray(TOP_LANE_PATH, np.float32))
    pf_radius = np.asarray(params["pathfinding_radius"])

    by_crowd = collections.defaultdict(list)

    n = len(snaps) - 1
    for i in range(n):
        if i % 1000 == 0:
            print(f"  ... {i}/{n}", flush=True)
        sn, sn1 = snaps[i], snaps[i + 1]
        dt = sn1.t_ms - sn.t_ms
        if dt <= 0 or dt > 34:
            continue
        has_missile = any(e.kind == "SpellMissile" for e in sn.entities)
        if has_missile:
            continue

        state_n, report = inject_snapshot(sn, wave_states[i], params, PROFILES)
        tr = compare_one_tick(state_n, report.notes, sn1, params, lane_path)

        notes = [nt for nt in report.notes if nt.kind in ("LaneMinion", "Champion")]
        xs = np.array([nt.x for nt in notes])
        ys = np.array([nt.y for nt in notes])
        rows = np.array([nt.model_row if nt.model_row is not None else 0
                         for nt in notes])
        slot_to_idx = {nt.slot: k for k, nt in enumerate(notes)}
        note_by_slot = {nt.slot: nt for nt in report.notes}

        for m in tr.matched:
            if not m.movement_trustworthy or m.kind != "LaneMinion":
                continue
            note = note_by_slot.get(m.slot)
            if note is None or note.slot not in slot_to_idx:
                continue
            k0 = slot_to_idx[note.slot]
            r0 = pf_radius[rows[k0]]
            d = np.hypot(xs - xs[k0], ys - ys[k0])
            touching = (r0 + 1.0) + pf_radius[rows]
            overlap = (d > 0) & (d < touching)
            n_overlap = int(overlap.sum())

            dx = m.pred.x - m.real.x
            dy = m.pred.y - m.real.y
            mvx, mvy = m.pred.x - m.pre_x, m.pred.y - m.pre_y
            mag = math.hypot(mvx, mvy)
            if mag <= 1e-6 or math.isnan(m.pre_x):
                continue
            hx, hy = mvx / mag, mvy / mag
            signed_along = dx * hx + dy * hy
            by_crowd[min(n_overlap, 3)].append(signed_along)

    print(f"\nsigned position-along-heading error, missile-free trustworthy "
          f"ticks, bucketed by pre-tick collision-neighbour count:")
    for k in sorted(by_crowd):
        v = np.asarray(by_crowd[k])
        label = f"{k}" if k < 3 else "3+"
        frac_at_target = np.mean(np.abs(v) < POS_Q_UNIT)
        print(f"  neighbours={label}: n={len(v)} mean={v.mean():+.4f} "
              f"median={np.median(v):+.4f} frac|err|<1/16={100*frac_at_target:.1f}%")


if __name__ == "__main__":
    main()
