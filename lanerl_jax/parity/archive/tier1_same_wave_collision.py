"""RESET-004's ten same-wave collision residuals: are they ordering, hidden
state, or float tie-breaking?

The ledger records a 300-pair raw-float pre-clash corpus at 2,160/2,170
componentwise minion positions, with ten residuals "roughly 0.4 world units
perpendicular to travel" that diagnostic replay had not explained, and calls
them same-wave collision.  This module reproduces that exact measurement and
then asks the one question a percentage cannot answer: *which* ten samples,
and what is different about them.

Why the measurement is rebuilt here rather than read off `tier1_full`
---------------------------------------------------------------------
The aggregate report compares the dump's 1/16-quantised integers.  That test
cannot see a 0.4-unit residual as anything but "6 or 7 quanta", and it cannot
distinguish a sim error from the wire's own rounding.  The diagnostic stream
carries the server's raw float32 position BITS (`xbits`/`ybits`), so this
module compares float against float -- the only version of the comparison in
which "exact" means exact.

The counterfactual, which is the actual experiment
--------------------------------------------------
``--rounded-spawn`` restores `sim.init.MINION_SPAWN` to the rounded
(918, 1720)/(12451, 13218) barracks coordinates that were in the tree when
the ten residuals were recorded, before commit 12f2a25 replaced them with the
map's own fractional `CentralPoint` values.  Everything else is held fixed --
same recorded server log, same injector, same collision pass.  If the ten
residuals come back under the rounded constant and vanish under the
fractional one, the cause is the sim's spawn coordinate, and no amount of
collision-ordering work would have removed them.  Run both arms as separate
processes (the constant is read at `tick` trace time and baked into the
compiled program):

    python -m lanerl_jax.parity.tier1_same_wave_collision --existing-log LOG
    python -m lanerl_jax.parity.tier1_same_wave_collision --existing-log LOG \\
        --rounded-spawn
"""
from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

#: The sample the ledger's "2,170" comes from: the first 300 tick-pairs at or
#: after the first wave, i.e. the window in which both waves are marching and
#: nothing has met anything yet.  Collision here is same-wave only, which is
#: exactly what makes the residual attributable.
PRE_CLASH_PAIRS = 300

#: Rounded barracks coordinates as they stood before the fractional
#: `CentralPoint` fix.  Kept here, not imported, precisely because the point
#: is to run the sim with a constant it no longer has.
ROUNDED_MINION_SPAWN = {"blue": (918.0, 1720.0), "red": (12451.0, 13218.0)}


def raw_xy(internal) -> Optional[tuple]:
    """The server's own float32 position, reassembled from its dumped bits."""
    if internal is None or internal.x_bits is None or internal.y_bits is None:
        return None
    return (struct.unpack("<f", struct.pack("<i", internal.x_bits))[0],
            struct.unpack("<f", struct.pack("<i", internal.y_bits))[0])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--existing-log", default=None,
                    help="server log with LANERL_STATE_DUMP_INTERNALS=1")
    ap.add_argument("--out", default="lanerl_jax/runs/tier1_same_wave_collision")
    ap.add_argument("--game-seconds", type=float, default=200.0)
    ap.add_argument("--port-base", type=int, default=52700)
    ap.add_argument("--pairs", type=int, default=PRE_CLASH_PAIRS)
    ap.add_argument("--rounded-spawn", action="store_true",
                    help="run with the PRE-FIX rounded barracks coordinates")
    a = ap.parse_args(argv)

    from ..sim import step as step_module
    from ..sim.state import Team

    if a.rounded_spawn:
        # Patched before anything imports `tick` into a jit cache; `tick`
        # reads this dict at trace time, so the constant is baked into the
        # compiled program and cannot be changed afterwards in-process.
        step_module.MINION_SPAWN = {
            Team.BLUE: ROUNDED_MINION_SPAWN["blue"],
            Team.RED: ROUNDED_MINION_SPAWN["red"],
        }
        print("MINION_SPAWN patched to the PRE-FIX rounded coordinates: "
              f"{step_module.MINION_SPAWN}", flush=True)
    else:
        print(f"MINION_SPAWN as committed: {step_module.MINION_SPAWN}",
              flush=True)

    import jax.numpy as jnp

    from ..data.patch import load_patch
    from ..sim.init import TOP_LANE_PATH, lane_params
    from ..sim.profiles import PROFILES
    from ..sim.state import Kind
    from .diagnostic_identity import net_id_to_entity
    from .inject import inject_snapshot, replay_wave_states
    from .one_step import POS_GATE_SLACK, POS_GATE_TOL, compare_one_tick, \
        record_idle_trace
    from .tier1_full import FIRST_WAVE_MS
    from .trace import load_trace

    if a.existing_log:
        log = Path(a.existing_log)
        print(f"reusing existing log: {log}", flush=True)
    else:
        log = record_idle_trace(Path(a.out), game_seconds=a.game_seconds,
                                port_base=a.port_base)
        print(f"recorded: {log}", flush=True)

    trace = load_trace(log)
    snaps = [s for s in trace.snapshots if s.t_ms >= FIRST_WAVE_MS]
    print(f"{len(trace)} snapshots, {len(snaps)} at/after the first wave",
          flush=True)

    patch = load_patch()
    params = lane_params(patch)
    wave_states = replay_wave_states(snaps)
    lane_path = jnp.asarray(np.asarray(TOP_LANE_PATH, np.float32))
    cr = np.asarray(params["collision_radius"])

    n_samples = 0
    residuals: List[Dict] = []
    n_pairs_used = 0
    for i in range(min(a.pairs, len(snaps) - 1)):
        sn, sn1 = snaps[i], snaps[i + 1]
        dt = sn1.t_ms - sn.t_ms
        if dt <= 0 or dt > 34:
            continue
        previous = snaps[i - 1] if i else None
        if previous is not None and not (0 < sn.t_ms - previous.t_ms <= 34):
            previous = None
        state_n, report = inject_snapshot(
            sn, wave_states[i], params, PROFILES, previous_snapshot=previous)
        from .diagnostic_identity import net_id_to_injected_slot
        net_id_slots = net_id_to_injected_slot(sn, report.notes)
        tr = compare_one_tick(state_n, report.notes, sn1, params, lane_path,
                              pre_net_id_to_slot=net_id_slots)
        n_pairs_used += 1

        # The server's raw float32 position at N+1, by entity identity.
        internal_by_entity = {}
        net_by_entity = {}
        real_internals = {iv.net_id: iv for iv in sn1.ai_internals}
        for net_id, entity in net_id_to_entity(sn1).items():
            internal_by_entity[id(entity)] = real_internals.get(net_id)
            net_by_entity[id(entity)] = net_id

        x0 = np.asarray(state_n.x)
        y0 = np.asarray(state_n.y)
        kind0 = np.asarray(state_n.kind)
        alive0 = np.asarray(state_n.alive)
        model0 = np.asarray(state_n.model)
        seq0 = np.asarray(state_n.spawn_seq)
        note_by_slot = {n.slot: n for n in report.notes}

        for m in tr.matched:
            if m.kind != "LaneMinion":
                continue
            server_xy = raw_xy(internal_by_entity.get(id(m.real)))
            if server_xy is None:
                continue
            n_samples += 1
            dx = m.pred.x - server_xy[0]
            dy = m.pred.y - server_xy[1]
            linf = max(abs(dx), abs(dy))
            if linf <= POS_GATE_TOL + POS_GATE_SLACK:
                continue

            note = note_by_slot.get(m.slot)
            fresh = note is None
            # Pre-tick crowding, using the server's own trigger radii.
            if not fresh:
                d = np.hypot(x0 - x0[m.slot], y0 - y0[m.slot])
                touch = cr[model0[m.slot]] + cr[model0]
                over = alive0 & (kind0 != Kind.NONE) & (d > 0) & (d < touch)
                neighbours = [int(j) for j in np.flatnonzero(over)]
            else:
                neighbours = []
            heading = math.hypot(m.pred.x - m.pre_x, m.pred.y - m.pre_y) \
                if not fresh else 0.0
            if heading > 1e-9:
                hx = (m.pred.x - m.pre_x) / heading
                hy = (m.pred.y - m.pre_y) / heading
                along = dx * hx + dy * hy
                perp = -dx * hy + dy * hx
            else:
                along = perp = float("nan")
            residuals.append(dict(
                t_ms=sn.t_ms, slot=m.slot, team=m.team,
                net_id=net_by_entity.get(id(m.real)),
                fresh=fresh, dx=dx, dy=dy, linf=linf, along=along, perp=perp,
                sim=(m.pred.x, m.pred.y), server=server_xy,
                pre=(m.pre_x, m.pre_y),
                spawn_seq=int(seq0[m.slot]),
                n_neighbours=len(neighbours),
                neighbour_slots=neighbours,
                collision_cache=("n/a (spawned this tick)" if fresh
                                 else note.collision_cache_recovery),
                movement_reason=m.movement_reason,
            ))

    print(f"\npairs used: {n_pairs_used}")
    print(f"LaneMinion raw-float samples: {n_samples}")
    print(f"componentwise (L-inf) within 1/16: "
          f"{n_samples - len(residuals)}/{n_samples} "
          f"({100 * (n_samples - len(residuals)) / max(1, n_samples):.2f}%)")
    print(f"residuals: {len(residuals)}")
    fresh_n = sum(1 for r in residuals if r["fresh"])
    print(f"  of which freshly spawned this tick: {fresh_n}")
    for r in residuals:
        print(f"  t={r['t_ms']} slot={r['slot']} team={r['team']} "
              f"net={r['net_id']} fresh={r['fresh']} "
              f"spawn_seq={r['spawn_seq']} neighbours={r['n_neighbours']}\n"
              f"      sim=({r['sim'][0]:.4f},{r['sim'][1]:.4f}) "
              f"server=({r['server'][0]:.4f},{r['server'][1]:.4f}) "
              f"d=({r['dx']:+.4f},{r['dy']:+.4f}) Linf={r['linf']:.4f}\n"
              f"      along={r['along']:+.4f} perp={r['perp']:+.4f} "
              f"cache={r['collision_cache']!r} "
              f"reason={r['movement_reason']!r}")


if __name__ == "__main__":
    main()
