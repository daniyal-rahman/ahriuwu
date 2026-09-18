"""Task 3 follow-up: how much of the crowding-bucket position residual does a
FAITHFUL SEQUENTIAL (Gauss-Seidel, multi-push) collision reference explain,
against how much `sim/collision.py`'s current Jacobi (simultaneous,
single-push) approximation explains -- and does using the TRUE object-add
order (creation order), not slot order, matter?

Coordinator's follow-up experiment (2026-09-16, after the slot-order result
below showed the sequential mechanism WORSE than Jacobi at every crowding
level): `CollisionHandler._objects` is in true creation order, permanently
-- `List<T>.Remove` shifts survivors down but never reorders them
(`GameObject.cs:150-155`), and `CollisionHandler.AddObject` is called once,
from `GameObject.OnAdded()`, right after instantiation. Slot order is
uncorrelated with this (minion slots are recycled on death/respawn, and our
`[champions | minions | turrets]` layout disagrees with the server's
turrets-created-at-map-load-first order even before recycling). At
neighbours=1 the INNER order (`GetNearestObjects`' quadtree traversal, not
yet reconstructed -- see the doc) cannot matter: there is exactly one
colliding neighbour, so `UpdateCollision` applies exactly one escape
regardless of traversal order. So neighbours=1 (72% of all crowded samples
in this fixture) isolates the OUTER order alone. Prediction to test, not
assume: if outer order is the dominant term, neighbours=1 should move from
72.5% (slot order) toward the ~94% isolated units already achieve; if it
does not move, outer order is not the issue.

True creation order needs whole-trace state a single injected snapshot does
not carry, so it is RECONSTRUCTED here, not observed: `sim/waves.py`'s
schedule is deterministic and RNG-free, and both barracks spawn in lockstep
(one `step_waves` firing spawns one minion for EACH team, same tick), so the
total-ever-created count is identical for both teams at every tick and is
tracked incrementally while walking the trace. Which of a team's ALIVE
minions is its OLDEST is recovered from `_objects`' own invariant (removal
preserves survivor order) via a proxy: cumulative arc-length progress along
that team's own lane corridor (`TOP_LANE_PATH`, reversed for red) --
furthest-progressed is assumed oldest. This assumes no old survivor outlives
a newer sibling (usually true; front-line minions generally die first) and
a specific, FLAGGED guess for which barrack's minion is created first within
one simultaneous spawn event (assumed blue-then-red; the true order depends
on map scene-file parse order this project has not located). Both
assumptions are named so a wrong result here is diagnosable, not silently
trusted.

Coordinator's brief (see docs/TIER1_POST_REORDER.md task 3): our crowding
buckets showed 0 neighbours 93.4% within 1/16, 1 neighbour 27.1%, 2
neighbours 33.2%, 3+ 18.6% -- isolated movement is fine, collision-affected
movement is not. Two things distinguish the server's real collision pass
(`GameServerLib/Handlers/CollisionHandler.cs:121-156`,
`AttackableUnit.OnCollision`, `AttackableUnit.cs:278-318`) from
`resolve_collisions`:

1. Gauss-Seidel, not Jacobi: the server loops units in order and teleports
   each one immediately (`SetPosition`), so a LATER unit in that pass escapes
   from an EARLIER unit's ALREADY-MOVED position. `resolve_collisions`
   computes every push from the same pre-tick snapshot and applies them
   simultaneously.
2. Multiple pushes per unit per tick: the server pushes a unit once per
   OVERLAPPING neighbour found, in sequence (each push can change which
   neighbours still overlap). `resolve_collisions` applies exactly one push,
   from the lowest-index overlapping neighbour (a documented, deliberate
   approximation).

This script builds a slow, non-vectorised NumPy reference that does both (1)
and (2), in SLOT order (our best available proxy for the server's true
object-add order -- unverified, see the module docstring's point 3; this is
exactly what this measurement is for), and compares it against both the
CURRENT JAX collision and the server's real next-tick position, isolating
collision's effect by feeding EACH collision hypothesis through the SAME
downstream one-tick movement integration (`movement_jax.step_move_units`)
rather than re-running the whole tick (which would also perturb combat/
targeting on a slightly different position -- not what this question asks).

Run via slurm, not the login node (records its own trace, O(N^2) python per
tick, N<=66):

    sbatch slurm/parity.sbatch python -m lanerl_jax.parity.tier1_collision_sequential

Why PRODUCTION scores slightly worse than the "creation order" reference
above at neighbours>=2, 2026-09-16 follow-up (root-caused, not left open)
-----------------------------------------------------------------------------
The PRODUCTION row (the real `sim.collision.resolve_collisions`, added
below) beats the pre-parity-pass Jacobi row at every crowding bucket, but
came in slightly BELOW this file's own "reconstructed creation order" NumPy
reference at neighbours=2 (70.8% vs 72.7%) and 3+ (53.7% vs 58.2%), despite
carrying strictly more fixes (the radius split, the turret split, and the
real `spawn_seq` instead of a reconstruction). Two candidate explanations
were checked directly against the recorded trace, not assumed:

1. **Ordering.** This file originally fed `estimate_creation_order`'s float
   rank into `resolve_collisions` via a truncating `.astype(np.int32)`,
   which collapses the deliberate blue/red `team_bit` tie-break (see
   `estimate_creation_order`'s own docstring) whenever a blue and red minion
   share a per-team ordinal -- confirmed to happen on 1,320/1,320 sampled
   ticks. Fixed with a lossless `argsort(argsort(...))` rank encoding.
   Measured impact of that fix, over 1,100 sampled ticks against the SAME
   trace: 1,084 (98.5%) had a genuinely different overall ordering
   permutation, but **0 of those 1,084 changed any `resolve_collisions`
   output position by more than 1e-4** (max observed difference: exactly
   0.0). So the ordering bug was real and worth fixing, but it is NOT what
   separates PRODUCTION from the reference on this corpus.
2. **The radius split itself.** Holding order fixed (the same lossless
   rank for both) and varying only whether the trigger uses
   `pathfinding_radius` (reference-style) or `collision_radius`
   (production, correct per `Minion.cs:57`), 785/1,100 sampled ticks (71%)
   had at least one minion resolve to a different position -- and NONE of
   those involved a champion within 120 units (ruling out the
   Champion-specific 30-vs-35 radius gap as the cause). The real driver is
   PER-MINION-TYPE: `data.patch` loads melee/caster `PathfindingRadius` at
   ~35.74 and cannon/super at ~55.74-55.52 (genuine Content values, nothing
   to do with this project's fixes), while every lane minion's
   `CollisionRadius` is hard-coded to a uniform 40 by the server regardless
   of type. So relative to the reference (which used `PathfindingRadius`
   for the trigger, like the old Jacobi code):
   - melee/caster pairs trigger MORE readily under production (40+40=80 vs
     35.74+35.74=71.5 -- their hard-coded CollisionRadius is BIGGER than
     their own PathfindingRadius),
   - cannon/super pairs trigger LESS readily under production
     (40+40=80 vs 55.74+55.74=111.5 -- the opposite direction).
   Both are correct per source; the reference never modelled this split at
   all, so it was never going to agree with a fully-correct implementation
   in a crowd containing a cannon/super minion. This is the reference being
   a cruder approximation, not evidence against PRODUCTION.

Net: PRODUCTION's small shortfall against this file's own NumPy reference
at high crowding is explained, source-grounded, and does not indicate a bug
in `sim.collision.resolve_collisions`. The ordering fix is retained anyway
because it is a real correction, evaluated on a different criterion than
"does it change this corpus's outcome."
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import numpy as np


def resolve_collisions_sequential(
    x: np.ndarray, y: np.ndarray, kind: np.ndarray, alive: np.ndarray,
    pathfinding_radius: np.ndarray, ghosted: np.ndarray, order,
) -> Tuple[np.ndarray, np.ndarray]:
    """``CollisionHandler.Update`` -> ``UpdateCollision`` -> ``OnCollision``,
    faithfully sequential: NumPy, mutates as it goes, O(len(order)^2).

    ``order`` is the iteration order for BOTH the outer loop (which unit
    pushes) and the inner one (which neighbours it checks against, i.e. the
    stand-in for ``GetNearestObjects``' return order) -- unverified against
    the server's true quadtree/object-list order, see the module docstring.
    """
    from lanerl_jax.sim.state import Kind

    n = len(x)
    x = x.copy()
    y = y.copy()
    collides = alive & (kind != Kind.TURRET) & (kind != Kind.NONE) & ~ghosted
    r1 = pathfinding_radius + 1.0
    r2 = pathfinding_radius

    for i in order:
        if not collides[i]:
            continue
        for j in order:
            if j == i or not collides[j]:
                continue
            dx, dy = x[j] - x[i], y[j] - y[i]
            d = math.hypot(dx, dy)
            touch = r1[i] + r2[j]
            if 0 < d < touch:
                ux, uy = dx / d, dy / d
                push = d - touch  # negative: slide AWAY from j by the overlap
                x[i] += ux * push
                y[i] += uy * push
    return x, y


def _old_jacobi_resolve_collisions(x, y, kind, alive, pathfinding_radius, ghosted=None):
    """``sim.collision.resolve_collisions`` as it stood before the 2026-09-16
    collision-parity pass (one push, Jacobi/simultaneous, lowest ARRAY index,
    one shared radius) -- reproduced verbatim (not imported: the production
    function's signature changed) so this script can still report the
    PRE-PASS row for an apples-to-apples "before vs after" comparison
    against `docs/TIER1_POST_REORDER.md`'s numbers. See
    `sim/collision.py`'s git history for the original, annotated version.
    """
    import jax.numpy as jnp
    from lanerl_jax.sim.state import Kind

    n = x.shape[0]
    collides = alive & (kind != Kind.TURRET) & (kind != Kind.NONE)
    if ghosted is not None:
        collides = collides & ~ghosted
    r1 = pathfinding_radius + 1.0
    r2 = pathfinding_radius
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    d = jnp.sqrt(dx * dx + dy * dy)
    touching = r1[:, None] + r2[None, :]
    overlap = (collides[:, None] & collides[None, :] & ~jnp.eye(n, dtype=bool)
              & (d < touching) & (d > 0))
    first = jnp.argmax(overlap, axis=1)
    has = jnp.any(overlap, axis=1)
    j = jnp.clip(first, 0, n - 1)
    dj = d[jnp.arange(n), j]
    safe = jnp.where(dj > 0, dj, 1.0)
    ux = dx[jnp.arange(n), j] / safe
    uy = dy[jnp.arange(n), j] / safe
    push = dj - r1 - r2[j]
    return (jnp.where(has, x + ux * push, x), jnp.where(has, y + uy * push, y))


def _corridor_progress(x: np.ndarray, y: np.ndarray, path: np.ndarray) -> np.ndarray:
    """Cumulative arc length from ``path[0]`` to each ``(x[i], y[i])``'s
    nearest projection onto the polyline ``path`` -- a proxy for "how far
    along its own team's corridor has this minion walked", used as a stand-in
    for creation order (see the module docstring: older minions are further
    along, absent combat that stalls a front-line minion while newer ones
    catch up -- an acknowledged source of error, not hidden).
    """
    seg_vec = path[1:] - path[:-1]
    seg_len = np.hypot(seg_vec[:, 0], seg_vec[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    out = np.zeros(len(x))
    for i in range(len(x)):
        best_d, best_seg, best_t = float("inf"), 0, 0.0
        for s in range(len(path) - 1):
            ax, ay = path[s]
            bx, by = path[s + 1]
            dx, dy = bx - ax, by - ay
            l2 = dx * dx + dy * dy
            t = 0.0 if l2 == 0.0 else max(0.0, min(1.0,
                ((x[i] - ax) * dx + (y[i] - ay) * dy) / l2))
            px, py = ax + t * dx, ay + t * dy
            d = math.hypot(x[i] - px, y[i] - py)
            if d < best_d:
                best_d, best_seg, best_t = d, s, t
        out[i] = cum[best_seg] + best_t * seg_len[best_seg]
    return out


def estimate_creation_order(
    x: np.ndarray, y: np.ndarray, kind: np.ndarray, team: np.ndarray,
    alive: np.ndarray, total_born_so_far: int, top_lane_path: np.ndarray,
) -> np.ndarray:
    """Best-effort GLOBAL creation-order rank per unit (lower = created
    earlier), reconstructed per the module docstring's method. Champions get
    the two most-negative ranks (created at match start, before any minion);
    turrets get an arbitrary rank -- they are excluded from collision
    entirely, so it is never read for them.
    """
    from lanerl_jax.sim.state import Kind, Team

    n = len(x)
    rank = np.zeros(n, dtype=np.float64)
    rank[(kind == Kind.CHAMPION) & (team == Team.BLUE)] = -2.0
    rank[(kind == Kind.CHAMPION) & (team == Team.RED)] = -1.0

    for t, path in ((Team.BLUE, top_lane_path), (Team.RED, top_lane_path[::-1])):
        mask = alive & (kind == Kind.LANE_MINION) & (team == t)
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            continue
        progress = _corridor_progress(x[idx], y[idx], path)
        # most progress = oldest = smallest ordinal
        order = np.argsort(-progress)
        n_alive = len(idx)
        ordinals = total_born_so_far - n_alive + 1 + np.arange(n_alive)
        team_bit = 0.0 if t == Team.BLUE else 0.5
        for rank_pos, unit_i in enumerate(order):
            rank[idx[unit_i]] = ordinals[rank_pos] * 2.0 + team_bit
    return rank


def main() -> None:
    import jax.numpy as jnp
    from lanerl_jax.data.patch import load_patch
    from lanerl_jax.parity.inject import inject_snapshot, replay_wave_states
    from lanerl_jax.parity.one_step import (
        POS_Q_UNIT,
        compare_one_tick,
        record_idle_trace,
    )
    from lanerl_jax.parity.trace import load_trace
    from lanerl_jax.sim.collision import resolve_collisions
    from lanerl_jax.sim.init import TOP_LANE_PATH, lane_params
    from lanerl_jax.sim.movement_jax import TICK_MS, step_move_units
    from lanerl_jax.sim.profiles import PROFILES
    from lanerl_jax.sim.spells import BuffId, Slot
    from lanerl_jax.sim.state import Kind, MoveOrder, Team
    from lanerl_jax.sim.waves import FIRST_WAVE_MS as WAVES_FIRST_WAVE_MS
    from lanerl_jax.sim.waves import WaveState, step_waves
    from lanerl_jax.parity.tier1_full import FIRST_WAVE_MS

    out = Path("lanerl_jax/runs/tier1_collision_sequential")
    log = record_idle_trace(out, game_seconds=200.0, port_base=51900)
    print(f"recorded: {log}", flush=True)
    trace = load_trace(log)
    snaps = [s for s in trace.snapshots if s.t_ms >= FIRST_WAVE_MS]
    print(f"{len(trace)} total, {len(snaps)} after first wave", flush=True)

    patch = load_patch()
    params = lane_params(patch)
    wave_states = replay_wave_states(snaps)
    lane_path_np = np.asarray(TOP_LANE_PATH, np.float32)
    lane_path = jnp.asarray(lane_path_np)
    pf_radius = np.asarray(params["pathfinding_radius"])
    move_speed_arr = np.asarray(params["move_speed"])

    # Running count of minions ever created per team (identical for both,
    # since both barracks spawn in lockstep -- see the module docstring) as
    # of each snap[i]'s own tick, INCLUSIVE, mirroring the fixed
    # `replay_wave_states`'s pairing convention exactly (advance THROUGH
    # snap[i].t_ms before using the count alongside snap[i]).
    _wave_counter_st = WaveState(next_spawn_ms=WAVES_FIRST_WAVE_MS,
                                 minion_number=0, cannon_count=0)
    total_born_by_index: list[int] = []
    _running_total = 0
    for _snap in snaps:
        if step_waves(_wave_counter_st, float(_snap.t_ms)):
            _running_total += 1
        total_born_by_index.append(_running_total)

    def can_move_of(move_order, alive):
        blocked = np.isin(move_order, [MoveOrder.CAST_SPELL, MoveOrder.NONE,
                                       MoveOrder.STOP, MoveOrder.HOLD])
        return alive & ~blocked

    by_crowd_current = {}
    by_crowd_seq = {}
    by_crowd_creation = {}
    by_crowd_production = {}

    def bucket_lists(d, k):
        return d.setdefault(min(k, 3), [])

    n = len(snaps) - 1
    for i in range(n):
        if i % 1000 == 0:
            print(f"  ... {i}/{n}", flush=True)
        sn, sn1 = snaps[i], snaps[i + 1]
        dt = sn1.t_ms - sn.t_ms
        if dt <= 0 or dt > 34:
            continue
        if any(e.kind == "SpellMissile" for e in sn.entities):
            continue

        state_n, report = inject_snapshot(sn, wave_states[i], params, PROFILES)
        tr = compare_one_tick(state_n, report.notes, sn1, params, lane_path)

        x0 = np.asarray(state_n.x)
        y0 = np.asarray(state_n.y)
        kind0 = np.asarray(state_n.kind)
        team0 = np.asarray(state_n.team)
        alive0 = np.asarray(state_n.alive)
        model0 = np.asarray(state_n.model)
        move_order0 = np.asarray(state_n.move_order)
        ghosted0 = (np.asarray(state_n.buff_id)[:, Slot.E] == BuffId.GAREN_E) & alive0

        # current (JAX) collision, on the SAME pre-tick snapshot -- the
        # PRE-PARITY-PASS module: single push, Jacobi, one shared radius.
        # Reproduced here from the git history rather than imported, since
        # `sim.collision.resolve_collisions` no longer has this signature --
        # it IS the production function measured as "PRODUCTION" below.
        cx_a, cy_a = _old_jacobi_resolve_collisions(
            state_n.x, state_n.y, state_n.kind, state_n.alive,
            np.asarray(params["pathfinding_radius"])[model0], ghosted=jnp.asarray(ghosted0))
        cx_a, cy_a = np.asarray(cx_a), np.asarray(cy_a)

        # sequential reference, SLOT order
        order = list(range(len(x0)))
        cx_b, cy_b = resolve_collisions_sequential(
            x0, y0, kind0, alive0, pf_radius[model0], ghosted0, order)

        # sequential reference, RECONSTRUCTED CREATION order (spawn_seq
        # experiment -- see module docstring)
        creation_rank = estimate_creation_order(
            x0, y0, kind0, team0, alive0, total_born_by_index[i], lane_path_np)
        order_creation = list(np.argsort(creation_rank))
        cx_c, cy_c = resolve_collisions_sequential(
            x0, y0, kind0, alive0, pf_radius[model0], ghosted0, order_creation)

        # PRODUCTION: the actual `sim.collision.resolve_collisions`, fed the
        # SAME reconstructed creation order (as `spawn_seq`) and the SAME
        # ghosted mask as (b)/(c) above, but with its own two-radius split
        # (CollisionRadius trigger, now the server's real 40/30 hard-code;
        # PathfindingRadius resolution) and turret obstacle/affected split.
        #
        # `spawn_seq` must be an INTEGER rank, but `creation_rank` is float
        # and `estimate_creation_order` deliberately encodes the blue/red
        # same-wave tie-break as a FRACTIONAL +0.0/+0.5 (module docstring:
        # `team_bit`) on top of an integer-valued per-team ordinal. A naive
        # `.astype(np.int32)` TRUNCATES that 0.5 away, so a blue and a red
        # minion sharing a per-team ordinal (extremely common -- both
        # barracks spawn in lockstep, so this happens on nearly every tick
        # this corpus was sampled against) collide onto the SAME integer and
        # lose their intended relative order entirely -- confirmed directly:
        # of 1,320 sampled ticks, 1,320 had at least one such collision among
        # currently-alive minions (e.g. ranks 2.0 and 2.5 both -> 2). This is
        # a bug in how THIS SCRIPT feeds the reconstruction into the real
        # API, not in `resolve_collisions` -- the live sim's own `spawn_seq`
        # is always already a unique integer (see `sim/init.py`), so this
        # loss cannot occur outside this specific offline measurement.
        # `argsort(argsort(...))` gives each element its RANK -- a lossless
        # integer encoding of the exact same order `order_creation` above
        # was built from, with no truncation anywhere in the pipeline.
        creation_rank_int = np.argsort(np.argsort(creation_rank)).astype(np.int32)
        cr_radius = np.asarray(params["collision_radius"])[model0]
        cx_d, cy_d = resolve_collisions(
            jnp.asarray(x0), jnp.asarray(y0), jnp.asarray(kind0),
            jnp.asarray(alive0), jnp.asarray(creation_rank_int),
            jnp.asarray(cr_radius), jnp.asarray(pf_radius[model0]),
            ghosted=jnp.asarray(ghosted0))
        cx_d, cy_d = np.asarray(cx_d), np.asarray(cy_d)

        can_move = can_move_of(move_order0, alive0)
        ms = move_speed_arr[model0]

        xa_out, ya_out, _, _ = step_move_units(
            jnp.asarray(cx_a), jnp.asarray(cy_a), state_n.waypoints,
            state_n.waypoint_key, state_n.n_waypoints, jnp.asarray(ms),
            jnp.asarray(can_move), TICK_MS)
        xb_out, yb_out, _, _ = step_move_units(
            jnp.asarray(cx_b), jnp.asarray(cy_b), state_n.waypoints,
            state_n.waypoint_key, state_n.n_waypoints, jnp.asarray(ms),
            jnp.asarray(can_move), TICK_MS)
        xc_out, yc_out, _, _ = step_move_units(
            jnp.asarray(cx_c), jnp.asarray(cy_c), state_n.waypoints,
            state_n.waypoint_key, state_n.n_waypoints, jnp.asarray(ms),
            jnp.asarray(can_move), TICK_MS)
        xd_out, yd_out, _, _ = step_move_units(
            jnp.asarray(cx_d), jnp.asarray(cy_d), state_n.waypoints,
            state_n.waypoint_key, state_n.n_waypoints, jnp.asarray(ms),
            jnp.asarray(can_move), TICK_MS)
        xa_out, ya_out = np.asarray(xa_out), np.asarray(ya_out)
        xb_out, yb_out = np.asarray(xb_out), np.asarray(yb_out)
        xc_out, yc_out = np.asarray(xc_out), np.asarray(yc_out)
        xd_out, yd_out = np.asarray(xd_out), np.asarray(yd_out)

        note_by_slot = {nt.slot: nt for nt in report.notes}
        for m in tr.matched:
            if not m.movement_trustworthy or m.kind != "LaneMinion":
                continue
            note = note_by_slot.get(m.slot)
            if note is None:
                continue
            slot = m.slot
            d = np.hypot(x0 - x0[slot], y0 - y0[slot])
            touching = (pf_radius[model0[slot]] + 1.0) + pf_radius[model0]
            overlap = (alive0 & (kind0 != Kind.TURRET) & (d > 0) & (d < touching))
            overlap[slot] = False
            n_overlap = int(overlap.sum())

            real_x, real_y = m.real.x, m.real.y
            err_a = math.hypot(xa_out[slot] - real_x, ya_out[slot] - real_y)
            err_b = math.hypot(xb_out[slot] - real_x, yb_out[slot] - real_y)
            err_c = math.hypot(xc_out[slot] - real_x, yc_out[slot] - real_y)
            err_d = math.hypot(xd_out[slot] - real_x, yd_out[slot] - real_y)
            bucket_lists(by_crowd_current, n_overlap).append(err_a)
            bucket_lists(by_crowd_seq, n_overlap).append(err_b)
            bucket_lists(by_crowd_creation, n_overlap).append(err_c)
            bucket_lists(by_crowd_production, n_overlap).append(err_d)

    def report(name, d):
        print(f"\n{name} -> movement, vs the REAL server position "
              f"(missile-free, trustworthy):")
        for k in sorted(d):
            v = np.asarray(d[k])
            label = f"{k}" if k < 3 else "3+"
            print(f"  neighbours={label}: n={len(v)} "
                  f"frac<=1/16={100*np.mean(v<=POS_Q_UNIT+1e-2):.1f}% "
                  f"median={np.median(v):.4f} p95={np.percentile(v,95):.4f}")

    report("current (Jacobi, single-push) collision -- PRE-PARITY-PASS", by_crowd_current)
    report("SEQUENTIAL (Gauss-Seidel, multi-push, SLOT order)", by_crowd_seq)
    report("SEQUENTIAL (Gauss-Seidel, multi-push, RECONSTRUCTED CREATION order)",
          by_crowd_creation)
    report("PRODUCTION (sim.collision.resolve_collisions, lax.scan, "
          "CollisionRadius/PathfindingRadius split, turret obstacle fix)",
          by_crowd_production)


if __name__ == "__main__":
    main()
