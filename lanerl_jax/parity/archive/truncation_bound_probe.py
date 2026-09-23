"""Measure the true per-tick maxima behind ``MAX_STEPS_PER_TICK`` (movement)
and ``MAX_ESCAPES_PER_UNIT`` (collision), over a real, combat-heavy corpus
window, instead of quoting the "chosen from precedent" 8 either module's
docstring admits to.

Both bounds are advertised as CORRECTNESS bounds, not performance knobs:
too small silently truncates a unit's movement/escapes for that tick, and
that reads as a pathing or position bug rather than a measurement gap
(``lanerl_jax/sim/movement_jax.py``, ``lanerl_jax/sim/collision.py``). Both
modules already expose the measurement hook -- ``max_steps_used`` and
``max_escapes_used`` -- this script just drives them over injected real
server state instead of a synthetic fixture.

Per-tick state comes from replaying a WINDOW of a recorded server trace
through the injector (``lanerl_jax.parity.inject.inject_snapshot``), using
``lanerl_jax.parity.trace.load_trace_window`` so a 575 MB log does not need
its full ~26 GB parse. This does NOT run the JAX sim forward -- every tick's
(waypoints, key, n_waypoints, move_speed) / (x, y, kind, alive, spawn_seq,
radii, ghosted, frozen candidate positions) is the server's OWN recorded
state for that tick, injected independently, exactly as ``inject_snapshot``
is used everywhere else in ``parity/``.

Movement is measured for every alive unit every tick (matching
``max_steps_used``'s own contract: it forces ``can_move=True`` internally,
so it measures "how many waypoints WOULD this data consume", not "did this
particular tick's order happen to allow movement" -- see its docstring).
Collision is measured for every "affected" unit every tick (alive, not a
turret, not ghosted -- the same ``who can be pushed`` mask
``sim.collision`` itself uses), since turrets/ghosted/dead units are
guaranteed zero by construction and would only dilute the histogram.

Usage::

    ops/login_capped.sh 4G 1 .venv-jax/bin/python -m \\
        lanerl_jax.parity.archive.truncation_bound_probe \\
        --log lanerl_jax/runs/tier1_full/server/instance000.log \\
        --from-ms 200000 --to-ms 250000 --probe-steps 32 --probe-escapes 32
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

import numpy as np


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True)
    ap.add_argument("--from-ms", type=int, required=True)
    ap.add_argument("--to-ms", type=int, required=True)
    ap.add_argument("--max-snapshots", type=int, default=4000,
                    help="cap on snapshots materialised in full (memory guard)")
    ap.add_argument("--probe-steps", type=int, default=32,
                    help="probe passed to movement_jax.max_steps_used")
    ap.add_argument("--probe-escapes", type=int, default=32,
                    help="probe passed to collision.max_escapes_used")
    a = ap.parse_args(argv)

    import jax.numpy as jnp

    from ...data.patch import load_patch
    from ...sim.collision import max_escapes_used
    from ...sim.init import lane_params
    from ...sim.movement_jax import TICK_MS, max_steps_used
    from ...sim.profiles import PROFILES
    from ...sim.spells import E_BUFF_SLOT, BuffId, Q_HASTE_BUFF_SLOT, Q_HASTE_MULTIPLIER, Slot
    from ...sim.state import Kind
    from ..inject import inject_snapshot, replay_wave_states
    from ..tier1_full import FIRST_WAVE_MS
    from ..trace import load_trace_window

    print(f"loading window [{a.from_ms}, {a.to_ms}] ms from {a.log} "
          f"(max_snapshots={a.max_snapshots})", flush=True)
    trace = load_trace_window(Path(a.log), from_ms=a.from_ms, to_ms=a.to_ms,
                              max_snapshots=a.max_snapshots)
    snaps = [s for s in trace.snapshots if s.t_ms >= FIRST_WAVE_MS]
    print(f"{len(trace)} total snapshots, {len(snaps)} after first wave "
          f"(t>={FIRST_WAVE_MS})", flush=True)
    wave_states = replay_wave_states(snaps)

    patch = load_patch()
    params = lane_params(patch)
    dtype = jnp.float32

    step_hist: collections.Counter = collections.Counter()
    escape_hist: collections.Counter = collections.Counter()
    n_step_saturated = 0
    n_escape_saturated = 0
    n_ticks = 0
    n_step_unit_ticks = 0
    n_escape_unit_ticks = 0
    max_steps_seen = 0
    max_escapes_seen = 0
    step_examples = []
    escape_examples = []

    for i, sn in enumerate(snaps):
        if sn.t_ms < a.from_ms:
            continue
        if sn.t_ms > a.to_ms:
            break
        if sn.placeholder:
            continue
        previous = snaps[i - 1] if i and not snaps[i - 1].placeholder else None
        state_n, report = inject_snapshot(
            sn, wave_states[i], params, PROFILES, previous_snapshot=previous)
        n_ticks += 1

        P = lambda k: params[k][state_n.model]  # noqa: E731

        # ---- movement: mirror sim/step.py's move_speed exactly ----------
        q_hasted = (state_n.buff_id[:, Q_HASTE_BUFF_SLOT] == BuffId.GAREN_Q_HASTE)
        move_speed = P("move_speed") * jnp.where(
            q_hasted, jnp.asarray(Q_HASTE_MULTIPLIER, dtype), 1.0)
        steps = max_steps_used(state_n.waypoints, state_n.waypoint_key,
                               state_n.n_waypoints, move_speed,
                               delta_ms=TICK_MS, probe=a.probe_steps)
        steps_np = np.asarray(steps)
        alive_np = np.asarray(state_n.alive)
        for cnt in steps_np[alive_np]:
            cnt = int(cnt)
            step_hist[cnt] += 1
            n_step_unit_ticks += 1
            if cnt == a.probe_steps:
                n_step_saturated += 1
            if cnt > max_steps_seen:
                max_steps_seen = cnt
            if cnt >= 8 and len(step_examples) < 10:
                step_examples.append((sn.t_ms, cnt))

        # ---- collision: mirror sim/step.py's resolve_collisions call ----
        pre_ghosted = ((state_n.buff_id[:, E_BUFF_SLOT] == BuffId.GAREN_E)
                      & state_n.alive)
        collision_radius = P("collision_radius")
        pathfinding_radius = P("pathfinding_radius")
        escapes = max_escapes_used(
            state_n.x, state_n.y, state_n.kind, state_n.alive,
            state_n.spawn_seq, collision_radius, pathfinding_radius,
            ghosted=pre_ghosted, probe=a.probe_escapes,
            candidate_x=state_n.collision_x, candidate_y=state_n.collision_y,
            candidate_present=state_n.collision_present)
        escapes_np = np.asarray(escapes)
        kind_np = np.asarray(state_n.kind)
        ghosted_np = np.asarray(pre_ghosted)
        affected_np = (alive_np & (kind_np != int(Kind.NONE))
                      & (kind_np != int(Kind.TURRET)) & ~ghosted_np)
        for cnt in escapes_np[affected_np]:
            cnt = int(cnt)
            escape_hist[cnt] += 1
            n_escape_unit_ticks += 1
            if cnt == a.probe_escapes:
                n_escape_saturated += 1
            if cnt > max_escapes_seen:
                max_escapes_seen = cnt
            if cnt >= 8 and len(escape_examples) < 10:
                escape_examples.append((sn.t_ms, cnt))

        if n_ticks % 500 == 0:
            print(f"  ...{n_ticks} ticks processed (t={sn.t_ms})", flush=True)

    print(f"\nprocessed {n_ticks} ticks, {n_step_unit_ticks} alive unit-ticks "
          f"(movement), {n_escape_unit_ticks} affected unit-ticks (collision)",
          flush=True)

    def report(name, hist, n_unit_ticks, n_saturated, probe, max_seen,
              current_bound, examples):
        print(f"\n=== {name} ===")
        print(f"  probe = {probe}")
        if not hist:
            print("  NO unit-ticks recorded -- window produced no eligible units.")
            return
        total = sum(hist.values())
        print(f"  histogram (count -> unit-ticks, {total} total):")
        for k in sorted(hist):
            frac = 100.0 * hist[k] / total
            bar = "#" * max(1, int(frac))
            print(f"    {k:3d}: {hist[k]:8d} ({frac:6.3f}%)  {bar}")
        n_at_or_above = sum(v for k, v in hist.items() if k >= current_bound)
        n_at_exactly = hist.get(current_bound, 0)
        print(f"  observed max: {max_seen}")
        print(f"  unit-ticks AT current bound ({current_bound}): {n_at_exactly}")
        print(f"  unit-ticks AT-OR-ABOVE current bound ({current_bound}): "
              f"{n_at_or_above}")
        print(f"  unit-ticks where the PROBE itself saturated (count==probe="
              f"{probe}): {n_saturated}")
        if n_saturated:
            print("  *** PROBE SATURATED -- observed max is a LOWER BOUND, "
                 "not the true maximum. Re-run with a larger --probe-* value "
                 "to settle it. ***")
        if examples:
            print(f"  example (t_ms, count) unit-ticks at/above 8: {examples}")

    report("movement_jax.max_steps_used", step_hist, n_step_unit_ticks,
          n_step_saturated, a.probe_steps, max_steps_seen, 8, step_examples)
    report("collision.max_escapes_used", escape_hist, n_escape_unit_ticks,
          n_escape_saturated, a.probe_escapes, max_escapes_seen, 8,
          escape_examples)


if __name__ == "__main__":
    main()
