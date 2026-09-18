"""Fixed-shape local static routing for bounded follow-camera clicks.

The policy cannot issue an arbitrary map-to-map route: its destination is a
bounded offset from the champion.  A production route table therefore stores
one row per covered *source cell* and one column per relative goal-cell offset,
not every ordered pair of Map1 cells.  Repeated gathers reconstruct the route
without an A* in the training step.

This module owns the device-side contract.  It deliberately keeps route status
observable: an uncovered cell, a goal outside the baked local window, a
genuine no-route entry, or a fixed-shape overflow must never be mislabeled as
successful pathing.  Callers may use the server's two-point fallback for a
non-ready status, but that trajectory is an approximation and is recorded in
``docs/JAX_FIDELITY_LEDGER.md``.

The table stores adjacent raw grid hops.  Runtime removes collinear interior
cells, which preserves the raw polyline while fitting ordinary routes into the
state's waypoint slots.  This is *not* the server's more aggressive
``SmoothPath`` line-of-sight pass; PATH-003/PATH-005 track that remaining
fidelity gap.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ..data.route_artifact import DIRECTION_OFFSETS, NO_ROUTE, STAY
from .state import MAX_WAYPOINTS
from .terrain_jax import TerrainGrid, closest_terrain_exit

__all__ = [
    "MAX_RAW_ROUTE_HOPS", "LocalRouteStatus", "LocalRouteTable",
    "LocalRouteResult", "lookup_local_hop", "build_local_waypoints",
]


# The calibrated perspective projection and local detours can span many raw
# 50-unit cell edges even though the action is viewport-bounded. This is a
# correctness bound and overflow is returned as a status rather than silently
# truncating the route.
MAX_RAW_ROUTE_HOPS = 128
# Keep several route hops in each XLA while-body invocation.  At the production
# batch size the vectorised loop otherwise launches a separate GPU loop body
# for every raw grid edge.  This is only a scheduling/layout change: every hop
# still executes in order and the inactive masks retain the original result.
ROUTE_LOOP_UNROLL = 8


class LocalRouteStatus:
    READY = 0
    SOURCE_UNCOVERED = 1
    GOAL_OUTSIDE_WINDOW = 2
    NO_ROUTE = 3
    RAW_HOP_OVERFLOW = 4
    WAYPOINT_OVERFLOW = 5
    ENDPOINT_UNANCHORED = 6
    TABLE_DISABLED = 7


class LocalRouteTable(NamedTuple):
    """Arrays for a source-row × relative-goal-offset route artifact.

    ``cell_to_row`` maps a global flattened navgrid cell to a table row, or
    ``-1`` when that source is not covered. Production ``next_hop`` packs two
    4-bit values per byte in row-major local-offset order (15 is NO_ROUTE).
    A legacy/unpacked ``(rows, side, side)`` array remains accepted for tiny
    unit fixtures. Direction values use the stable ABI from
    :mod:`lanerl_jax.data.route_artifact`.
    """

    cell_to_row: jax.Array
    next_hop: jax.Array
    offset_radius: int
    # Version-3 production artifacts carry one uint8 per logical entry.  It
    # counts consecutive raw edges that share this entry's direction, allowing
    # reconstruction to jump between exactly the same turn cells.  ``None``
    # retains the defensive one-hop path used by compact fixtures/v2 callers.
    run_length: jax.Array | None = None


class LocalRouteResult(NamedTuple):
    waypoints: jax.Array       # (MAX_WAYPOINTS, 2)
    n_waypoints: jax.Array     # int8
    status: jax.Array          # int8 LocalRouteStatus
    projected_goal_x: jax.Array
    projected_goal_y: jax.Array


def lookup_local_hop(source_cell, goal_cell, table: LocalRouteTable,
                     grid_width: int):
    """Return ``(direction_code, status)`` for one route-table state."""
    source_cell = jnp.asarray(source_cell, jnp.int32)
    goal_cell = jnp.asarray(goal_cell, jnp.int32)
    n_grid = table.cell_to_row.shape[0]
    source_valid = (source_cell >= 0) & (source_cell < n_grid)
    goal_valid = (goal_cell >= 0) & (goal_cell < n_grid)
    safe_source = jnp.clip(source_cell, 0, n_grid - 1)
    row = table.cell_to_row[safe_source]

    sx = source_cell % jnp.int32(grid_width)
    sy = source_cell // jnp.int32(grid_width)
    gx = goal_cell % jnp.int32(grid_width)
    gy = goal_cell // jnp.int32(grid_width)
    dx, dy = gx - sx, gy - sy
    r = jnp.int32(table.offset_radius)
    in_window = goal_valid & (jnp.abs(dx) <= r) & (jnp.abs(dy) <= r)
    side = 2 * table.offset_radius + 1
    ix = jnp.clip(dx + r, 0, side - 1)
    iy = jnp.clip(dy + r, 0, side - 1)
    safe_row = jnp.clip(row, 0, table.next_hop.shape[0] - 1)
    if table.next_hop.ndim == 3:  # compact test fixtures / v1 compatibility
        code = table.next_hop[safe_row, iy, ix]
    else:
        logical = iy * jnp.int32(side) + ix
        packed = table.next_hop[safe_row, logical // jnp.int32(2)]
        shift = (logical & jnp.int32(1)) * jnp.int32(4)
        nibble = (packed.astype(jnp.int32) >> shift) & jnp.int32(0xF)
        code = jnp.where(nibble == 0xF, jnp.int32(NO_ROUTE), nibble).astype(jnp.uint8)
    status = jnp.where(
        ~source_valid | (row < 0), LocalRouteStatus.SOURCE_UNCOVERED,
        jnp.where(~in_window, LocalRouteStatus.GOAL_OUTSIDE_WINDOW,
                  jnp.where(code == jnp.asarray(NO_ROUTE, code.dtype),
                            LocalRouteStatus.NO_ROUTE,
                            LocalRouteStatus.READY))).astype(jnp.int8)
    return code, status


def lookup_local_run_length(source_cell, goal_cell, table: LocalRouteTable,
                            grid_width: int):
    """Return a safe same-direction run length for a v3 table entry.

    Bounds mirror :func:`lookup_local_hop` because this helper is also invoked
    on lanes which have just failed or completed under a batched loop.  A zero
    sidecar value degrades to one ordinary hop; it can never make a route skip
    a turning point.
    """
    if table.run_length is None:
        return jnp.asarray(1, jnp.int32)
    source_cell = jnp.asarray(source_cell, jnp.int32)
    goal_cell = jnp.asarray(goal_cell, jnp.int32)
    n_grid = table.cell_to_row.shape[0]
    safe_source = jnp.clip(source_cell, 0, n_grid - 1)
    row = table.cell_to_row[safe_source]
    sx, sy = source_cell % jnp.int32(grid_width), source_cell // jnp.int32(grid_width)
    gx, gy = goal_cell % jnp.int32(grid_width), goal_cell // jnp.int32(grid_width)
    r = jnp.int32(table.offset_radius)
    side = 2 * table.offset_radius + 1
    ix = jnp.clip(gx - sx + r, 0, side - 1)
    iy = jnp.clip(gy - sy + r, 0, side - 1)
    safe_row = jnp.clip(row, 0, table.run_length.shape[0] - 1)
    run = table.run_length[safe_row, iy, ix].astype(jnp.int32)
    return jnp.maximum(run, jnp.int32(1))


def _world_cell(x, y, terrain: TerrainGrid):
    """World coordinate to C#-style truncated grid coordinate."""
    nx = (x - jnp.asarray(terrain.min_x, x.dtype)) / terrain.cell_size
    ny = (y - jnp.asarray(terrain.min_y, y.dtype)) / terrain.cell_size
    return jnp.trunc(nx).astype(jnp.int32), jnp.trunc(ny).astype(jnp.int32)


def _cell_world(ix, iy, dtype, terrain: TerrainGrid):
    cs = jnp.asarray(terrain.cell_size, dtype)
    return (jnp.asarray(terrain.min_x, dtype) + (ix.astype(dtype) + 0.5) * cs,
            jnp.asarray(terrain.min_y, dtype) + (iy.astype(dtype) + 0.5) * cs)


# Exact fractional endpoints can be radius-walkable even when their containing
# cell centre is not. The server handles that with a special first/final
# CastCircle edge. The cell table needs a covered anchor for the interior
# route; search a small deterministic neighbourhood and retain the exact float
# endpoint as waypoint 0/final. PATH-003 records that nearest-centre selection
# is not yet the server's exact CastCircle endpoint choice.
ENDPOINT_ANCHOR_RADIUS_CELLS = 3


def _nearest_covered_cell(x, y, ix, iy, table: LocalRouteTable,
                          terrain: TerrainGrid):
    offsets = jnp.arange(-ENDPOINT_ANCHOR_RADIUS_CELLS,
                         ENDPOINT_ANCHOR_RADIUS_CELLS + 1, dtype=jnp.int32)
    cx = ix + jnp.broadcast_to(offsets[None, :], (offsets.size, offsets.size))
    cy = iy + jnp.broadcast_to(offsets[:, None], (offsets.size, offsets.size))
    height, width = terrain.walkable.shape
    valid_cell = (cx >= 0) & (cx < width) & (cy >= 0) & (cy < height)
    flat = cy * jnp.int32(width) + cx
    safe = jnp.clip(flat, 0, table.cell_to_row.shape[0] - 1)
    covered = valid_cell & (table.cell_to_row[safe] >= 0)
    wx, wy = _cell_world(cx, cy, x.dtype, terrain)
    distance2 = (wx - x) ** 2 + (wy - y) ** 2
    score = jnp.where(covered, distance2, jnp.inf)
    best = jnp.argmin(score.reshape(-1))
    anchored = jnp.any(covered)
    chosen = flat.reshape(-1)[best]
    return chosen, anchored


def build_local_waypoints(source_x, source_y, goal_x, goal_y,
                          pathfinding_radius, table: LocalRouteTable,
                          terrain: TerrainGrid,
                          max_raw_hops: int = MAX_RAW_ROUTE_HOPS
                          ) -> LocalRouteResult:
    """Reconstruct one bounded route as fixed-shape JAX waypoints.

    The destination first passes through the server-equivalent
    ``GetClosestTerrainExit`` port. On any non-ready status the returned
    waypoint array is the explicit raw two-point fallback and ``status`` tells
    the caller why it is approximate.
    """
    dtype = jnp.result_type(source_x, source_y, goal_x, goal_y, jnp.float32)
    source_x, source_y = jnp.asarray(source_x, dtype), jnp.asarray(source_y, dtype)
    goal_x, goal_y = jnp.asarray(goal_x, dtype), jnp.asarray(goal_y, dtype)
    pgx, pgy, terrain_exhausted = closest_terrain_exit(
        goal_x, goal_y, jnp.asarray(pathfinding_radius, dtype), terrain)

    sx, sy = _world_cell(source_x, source_y, terrain)
    gx, gy = _world_cell(pgx, pgy, terrain)
    width = terrain.walkable.shape[1]
    height = terrain.walkable.shape[0]
    source_in_bounds = (sx >= 0) & (sx < width) & (sy >= 0) & (sy < height)
    goal_in_bounds = (gx >= 0) & (gx < width) & (gy >= 0) & (gy < height)
    raw_source_cell = sy * jnp.int32(width) + sx
    raw_goal_cell = gy * jnp.int32(width) + gx
    source_cell, source_anchored = _nearest_covered_cell(
        source_x, source_y, sx, sy, table, terrain)
    goal_cell, goal_anchored = _nearest_covered_cell(
        pgx, pgy, gx, gy, table, terrain)

    source_was_reanchored = source_cell != raw_source_cell
    anchor_sx, anchor_sy = source_cell % width, source_cell // width
    anchor_swx, anchor_swy = _cell_world(anchor_sx, anchor_sy, dtype, terrain)
    initial_count = jnp.where(source_was_reanchored, jnp.int32(2), jnp.int32(1))

    # The final result needs float waypoint pairs, but the reconstruction loop
    # only needs to remember *which grid cell* each turn occurred in.  Keeping
    # the full 64x2 float output in the loop carry made every raw hop carry and
    # dynamically scatter a much larger array.  Store the compact integer turn
    # log, then materialise exactly the same points after the route is known.
    # There can be at most 62 turn cells: source and exact final goal always
    # occupy the remaining two waypoint slots.
    turn_cells = jnp.zeros((MAX_WAYPOINTS - 2,), jnp.int32)

    # carry: current cell, previous direction, turn cells, count, status, done,
    # raw hops consumed.  The latter is per-route once a run sidecar is used:
    # vectorised lanes can jump different distances but share one outer loop.
    def body(carry, _, within_budget=True):
        current, previous, turns, count, status, done, raw_hops = carry
        code, hop_status = lookup_local_hop(current, goal_cell, table, width)
        active = within_budget & ~done & (status == LocalRouteStatus.READY)
        failed = active & (hop_status != LocalRouteStatus.READY)
        arrived = active & (code == jnp.asarray(STAY, code.dtype))
        moving = active & ~failed & ~arrived

        # A direction change occurs at the current cell, before taking the new
        # edge. The first direction does not add the source cell twice.
        turning = moving & (previous != STAY) & (code != previous)
        room = count < (MAX_WAYPOINTS - 1)  # reserve one slot for exact goal
        overflow = turning & ~room
        write_index = jnp.clip(count - initial_count, 0, MAX_WAYPOINTS - 3)
        # A turn is rare on the local grid.  Keep the dynamic update inside a
        # scalar branch so an ordinary straight hop leaves the compact turn
        # log untouched rather than materialising a gather/scatter pair.
        turns = jax.lax.cond(
            turning & room,
            lambda p: p.at[write_index].set(current),
            lambda p: p,
            turns)
        count = count + (turning & room).astype(count.dtype)

        safe_code = jnp.clip(code.astype(jnp.int32), 0,
                             DIRECTION_OFFSETS.shape[0] - 1)
        offsets = jnp.asarray(DIRECTION_OFFSETS, jnp.int32)[safe_code]
        run = lookup_local_run_length(current, goal_cell, table, width)
        remaining = jnp.asarray(max_raw_hops, jnp.int32) - raw_hops
        jumped_hops = jnp.minimum(run, remaining)
        jumped_hops = jnp.where(moving, jumped_hops, jnp.int32(0))
        next_cell = current + jumped_hops * (
            offsets[1] * jnp.int32(width) + offsets[0])
        current = jnp.where(moving & ~overflow, next_cell, current)
        previous = jnp.where(moving & ~overflow, code, previous)
        status = jnp.where(
            failed, hop_status,
            jnp.where(overflow, LocalRouteStatus.WAYPOINT_OVERFLOW, status))
        done = done | arrived | failed | overflow
        raw_hops = raw_hops + jumped_hops
        return (current, previous, turns, count, status, done, raw_hops), None

    initially_ready = (source_in_bounds & goal_in_bounds & ~terrain_exhausted
                       & source_anchored & goal_anchored)
    initial_status = jnp.where(initially_ready, LocalRouteStatus.READY,
                               LocalRouteStatus.ENDPOINT_UNANCHORED).astype(jnp.int8)
    initial_done = source_cell == goal_cell
    # A fixed ``scan`` made every routed decision pay all 128 table gathers,
    # including routes which had reached their goal after a handful of hops.
    # Keep the body and its inactive-lane masking identical, but stop the XLA
    # loop once no route in the vectorised program remains live.  The explicit
    # counter preserves the RAW_HOP_OVERFLOW boundary exactly.  ``jnp.any`` is
    # intentional: under the caller's outer ``vmap`` it reduces the mapped
    # routes to one scalar while predicate, while the body continues to mask
    # already-finished lanes exactly as the former scan did.
    #
    # Version-3 artifacts additionally carry a same-direction run length, so
    # a body advances to the next turn while counting every skipped raw edge.
    # Thus the fixed 128-hop overflow contract is unchanged even though a
    # straight segment needs one table-body iteration rather than one per cell.
    # The scalar while is deliberately unrolled by a small constant for legacy
    # one-hop tables. A v3 jump already lands at each turn, so unrolling it
    # would only execute extra masked lookup bodies after the short route ends.
    # A routed
    # production batch has thousands of lanes, so its termination is set by
    # the longest active path.  Leaving one raw hop per while-body then pays
    # GPU launch/control overhead for each edge.  Chaining eight *identical*
    # masked bodies leaves the observed path, status, and 128-hop boundary
    # untouched; it just amortises that control overhead.  The budget mask is
    # needed for callers/tests which pass a non-multiple of eight bound.
    carry = (source_cell, jnp.asarray(STAY, jnp.uint8), turn_cells,
             initial_count, initial_status, initial_done,
             jnp.asarray(0, jnp.int32))

    def cond(carry):
        _current, _previous, _turns, _count, status, done, raw_hops = carry
        active = ((~done) & (status == LocalRouteStatus.READY)
                  & (raw_hops < max_raw_hops))
        return jnp.any(active)

    loop_unroll = 1 if table.run_length is not None else ROUTE_LOOP_UNROLL

    def while_body(carry):
        for _ in range(loop_unroll):
            current, previous, turns, count, status, done, raw_hops = carry
            (current, previous, turns, count, status, done, raw_hops), _ = body(
                (current, previous, turns, count, status, done, raw_hops), None,
                raw_hops < max_raw_hops)
            carry = (current, previous, turns, count, status, done, raw_hops)
        return carry

    current, _previous, turns, count, status, done, _raw_hops = jax.lax.while_loop(
        cond, while_body, carry)

    reached = done & (status == LocalRouteStatus.READY) & (
        (current == goal_cell) | initial_done)
    status = jnp.where(
        (status == LocalRouteStatus.READY) & ~reached,
        LocalRouteStatus.RAW_HOP_OVERFLOW, status).astype(jnp.int8)

    # Materialise the exactly ordered source/turn waypoints after the compact
    # reconstruction loop.  Slot one intentionally receives the source anchor
    # even when it is unused, matching the former fixed-shape output byte for
    # byte outside the reported ``n_waypoints`` too.
    points = jnp.zeros((MAX_WAYPOINTS, 2), dtype)
    points = points.at[0].set(jnp.stack([source_x, source_y]))
    points = points.at[1].set(jnp.stack([anchor_swx, anchor_swy]))
    turn_x, turn_y = turns % width, turns // width
    turn_wx, turn_wy = _cell_world(turn_x, turn_y, dtype, terrain)
    turn_points = jnp.stack([turn_wx, turn_wy], axis=-1)
    turn_slots = jnp.clip(initial_count + jnp.arange(MAX_WAYPOINTS - 2),
                          0, MAX_WAYPOINTS - 1)
    old_turn_slots = points[turn_slots]
    valid_turn = jnp.arange(MAX_WAYPOINTS - 2) < (count - initial_count)
    points = points.at[turn_slots].set(jnp.where(
        valid_turn[:, None], turn_points, old_turn_slots))

    # If the fractional goal needed a neighbouring covered anchor, explicitly
    # visit that centre before the exact terrain-projected destination. This
    # avoids silently drawing the final segment from the previous turn through
    # the blocked centre that caused re-anchoring in the first place.
    goal_was_reanchored = goal_cell != raw_goal_cell
    anchor_gx, anchor_gy = goal_cell % width, goal_cell // width
    anchor_gwx, anchor_gwy = _cell_world(anchor_gx, anchor_gy, dtype, terrain)
    anchor_room = count < (MAX_WAYPOINTS - 1)
    anchor_index = jnp.clip(count, 0, MAX_WAYPOINTS - 1)
    old_anchor_slot = points[anchor_index]
    points = points.at[anchor_index].set(jnp.where(
        goal_was_reanchored & anchor_room,
        jnp.stack([anchor_gwx, anchor_gwy]), old_anchor_slot))
    status = jnp.where(
        (status == LocalRouteStatus.READY) & goal_was_reanchored & ~anchor_room,
        LocalRouteStatus.WAYPOINT_OVERFLOW, status).astype(jnp.int8)
    count = count + (goal_was_reanchored & anchor_room).astype(count.dtype)

    # Successful routes end at the exact terrain-projected float destination,
    # not at the goal anchor's cell centre. Failed routes visibly use
    # [source, goal].
    room_for_goal = count < MAX_WAYPOINTS
    status = jnp.where(
        (status == LocalRouteStatus.READY) & ~room_for_goal,
        LocalRouteStatus.WAYPOINT_OVERFLOW, status).astype(jnp.int8)
    success = status == LocalRouteStatus.READY
    goal_index = jnp.clip(count, 1, MAX_WAYPOINTS - 1)
    points = points.at[goal_index].set(jnp.stack([pgx, pgy]))
    success_count = jnp.minimum(count + 1, MAX_WAYPOINTS)

    fallback = jnp.zeros_like(points)
    fallback = fallback.at[0].set(jnp.stack([source_x, source_y]))
    # LanerlControl's fallback is constructed by the caller from the ORIGINAL
    # order point when GetPath returns null; the internally projected terrain
    # exit is not returned from GetPath. Preserve that distinction here.
    fallback = fallback.at[1].set(jnp.stack([goal_x, goal_y]))
    points = jnp.where(success, points, fallback)
    n_points = jnp.where(success, success_count, 2).astype(jnp.int8)
    return LocalRouteResult(points, n_points, status, pgx, pgy)
