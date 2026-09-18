"""JIT-compatible access to the server's Map1 terrain grid.

``NavigationGrid.GetClosestTerrainExit`` is used in two places in the server:
when ``CollisionHandler`` finds an object centre inside terrain, and as the
fallback after a unit-to-unit escape point is not walkable.  The host-side
reference in :mod:`lanerl_jax.data.navgrid` is intentionally a literal Python
port.  This module is its small device-side counterpart: grid reads are gathers
from one shared 86k-cell boolean constant and the source's data-dependent
spiral is a ``lax.while_loop``.

The radius stencil is deliberately fixed at two cells.  Map1's largest loaded
``PathfindingCollisionRadius + 1`` is 89.4 world units and cells are 50 units,
so every server ``GetAllCellsInRange`` candidate is within two cells of the
truncated centre.  It is a map/content contract, not an approximation.  A
roster with a larger pathfinding radius must raise
``MAX_TERRAIN_RADIUS_CELLS`` before using this module.
"""
from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple

import jax
import jax.numpy as jnp

__all__ = [
    "TerrainGrid",
    "MAX_TERRAIN_RADIUS_CELLS",
    "MAX_TERRAIN_EXIT_STEPS",
    "MAX_CAST_CIRCLE_LINE_STEPS",
    "MAX_CAST_CIRCLE_SPAN_CELLS",
    "map1_terrain",
    "row_prefix",
    "is_walkable",
    "cast_circle_blocked",
    "closest_terrain_exit",
    "exit_terrain_collision",
    "exit_blocked_escape",
    "repair_collision_terrain_batch",
]


class TerrainGrid(NamedTuple):
    """Static map data needed by the device-side terrain queries.

    ``walkable`` is ``(height, width)`` and follows the server's definition:
    both ``NOT_PASSABLE`` and ``SEE_THROUGH`` cells are blocked.  The scalar
    fields are Python floats so map geometry is compile-time metadata, while
    the mask remains a normal JAX array and can be shared across environments.
    """

    walkable: jax.Array
    cell_size: float
    min_x: float
    min_y: float
    #: ``(height, width + 1)`` exclusive prefix sum of ``walkable`` along x.
    #: Purely derived -- see :func:`row_prefix` -- and carried on the grid so
    #: it is built once instead of per call.  ``None`` is valid; small
    #: fixtures let it be recomputed.
    walkable_prefix: jax.Array | None = None


# Map1 has 50-unit cells and the largest currently loaded pathfinding radius
# is 88.4.  +1 is the largest threshold passed by OnCollision.
MAX_TERRAIN_RADIUS_CELLS = 2

# The server loop is unbounded.  JAX needs a safety cap; this is intentionally
# enormous relative to Map1's normal wall exits, and callers can use the
# returned `exhausted` diagnostic to detect a violated assumption instead of
# mistaking a non-exit for a valid result.
MAX_TERRAIN_EXIT_STEPS = 4096

# # ``LocalRouteTable`` bounds a goal to +/-50 cells and endpoint anchoring adds
# at most three.  A source ``CastCircle`` line therefore has <256 iterator
# turns and spans <128 cells on the y axis.  These are correctness bounds:
# callers receive ``exhausted`` and must fail closed rather than declaring an
# uninspected segment visible.  There is deliberately no x bound: the interior
# test reads whole rows out of the prefix table, so a wide row costs nothing
# and needs no window.
MAX_CAST_CIRCLE_LINE_STEPS = 256
MAX_CAST_CIRCLE_SPAN_CELLS = 128


@lru_cache(maxsize=1)
def map1_terrain() -> TerrainGrid:
    """Load Map1's static walkability mask once, ready to close over in JIT.

    This is deliberately lazy: data-only users of the simulation must not need
    the vendored server content merely by importing a mechanics module.
    """
    from lanerl_jax.data.navgrid import NavGrid

    grid = NavGrid.load()
    walkable = jnp.asarray(grid.walkable_mask())
    return TerrainGrid(
        walkable=walkable,
        cell_size=float(grid.cell_size),
        min_x=float(grid.min_grid[0]),
        min_y=float(grid.min_grid[2]),
        walkable_prefix=row_prefix(walkable),
    )


def row_prefix(walkable: jax.Array) -> jax.Array:
    """Exclusive prefix sum of ``walkable`` along x, one extra column.

    ``CastCircle``'s last step asks, for each row of the swept band, whether
    every cell strictly between that row's leftmost and rightmost enumerated
    cell is walkable.  Done directly that is one grid read per candidate cell,
    and the fixed-shape device form has to size the window for the worst case
    -- a 128x128 square of gathers per call, nearly all of it masked off.

    With a prefix sum the same question is two reads: the run [a, b) is fully
    walkable exactly when ``P[y, b] - P[y, a] == b - a``.  Same answer, and it
    removes the square.  Map1's table is 294 x 294 int32, about 345 KiB.
    """
    height, width = walkable.shape
    out = jnp.zeros((height, width + 1), jnp.int32)
    return out.at[:, 1:].set(jnp.cumsum(walkable.astype(jnp.int32), axis=1))


def _trunc_to_i32(v):
    """C# ``(short)`` truncation toward zero for ordinary Map1 coordinates."""
    return jnp.trunc(v).astype(jnp.int32)


def is_walkable(x: jax.Array, y: jax.Array, radius: jax.Array,
                terrain: TerrainGrid) -> jax.Array:
    """Device port of ``NavigationGrid.IsWalkable`` for scalar positions.

    The radius-zero branch uses ``GetCell`` directly.  Positive radii mirror
    ``GetAllCellsInRange``: candidates outside the grid are omitted (rather
    than treated as blocked), and the distance is to each cell *square*, not
    its centre.  This function is scalar by design; use ``jax.vmap`` for a
    batch, which avoids forcing every unit to wait for another's spiral.
    """
    walk = terrain.walkable.reshape(-1)
    height, width = terrain.walkable.shape
    dtype = jnp.result_type(x, y, radius, jnp.float32)
    x = jnp.asarray(x, dtype)
    y = jnp.asarray(y, dtype)
    radius = jnp.asarray(radius, dtype)
    nx = (x - jnp.asarray(terrain.min_x, dtype)) / jnp.asarray(terrain.cell_size, dtype)
    ny = (y - jnp.asarray(terrain.min_y, dtype)) / jnp.asarray(terrain.cell_size, dtype)

    def cell_value(ix, iy):
        # NavigationGrid.GetCell's odd x == CellCountX behaviour is retained:
        # it aliases the first cell in the following row when that row exists.
        linear = iy * jnp.int32(width) + ix
        valid = ((ix >= 0) & (ix <= width) & (iy >= 0) & (iy <= height)
                 & (linear >= 0) & (linear < walk.shape[0]))
        got = walk[jnp.clip(linear, 0, walk.shape[0] - 1)]
        return got, valid

    def point_query(_):
        got, valid = cell_value(_trunc_to_i32(nx), _trunc_to_i32(ny))
        return valid & got

    def radius_query(_):
        r = radius / jnp.asarray(terrain.cell_size, dtype)
        base_x, base_y = _trunc_to_i32(nx), _trunc_to_i32(ny)
        offsets = jnp.arange(-MAX_TERRAIN_RADIUS_CELLS,
                             MAX_TERRAIN_RADIUS_CELLS + 1, dtype=jnp.int32)
        ix = base_x + offsets[:, None]
        iy = base_y + offsets[None, :]
        linear = iy * jnp.int32(width) + ix
        valid = ((ix >= 0) & (ix <= width) & (iy >= 0) & (iy <= height)
                 & (linear >= 0) & (linear < walk.shape[0]))
        got = walk[jnp.clip(linear, 0, walk.shape[0] - 1)]

        # The C# loops are inclusive between short(origin +/- r), then retain
        # cells whose unit square is within r.  The fixed stencil contains all
        # such cells for the Map1 radius contract above.
        ix_f, iy_f = ix.astype(dtype), iy.astype(dtype)
        dx = jnp.maximum(jnp.abs(nx - (ix_f + 0.5)) - 0.5, 0.0)
        dy = jnp.maximum(jnp.abs(ny - (iy_f + 0.5)) - 0.5, 0.0)
        in_range = ((ix >= _trunc_to_i32(nx - r))
                    & (ix <= _trunc_to_i32(nx + r))
                    & (iy >= _trunc_to_i32(ny - r))
                    & (iy <= _trunc_to_i32(ny + r))
                    & (dx * dx + dy * dy <= r * r))
        # GetAllCellsInRange silently omits null GetCell results.
        return jnp.all(jnp.where(in_range & valid, got, True))

    return jax.lax.cond(radius == 0, point_query, radius_query, operand=None)


def cast_circle_blocked(x0: jax.Array, y0: jax.Array,
                        x1: jax.Array, y1: jax.Array,
                        radius: jax.Array, terrain: TerrainGrid,
                        max_line_steps: int = MAX_CAST_CIRCLE_LINE_STEPS,
                        span_cells: int = MAX_CAST_CIRCLE_SPAN_CELLS):
    """``NavigationGrid.CastCircle`` in cell coordinates.

    Returns ``(blocked, exhausted)`` with the server's polarity (``blocked``
    means that the swept segment is unusable).  This is the exact source
    construction: endpoint ``GetAllCellsInRange`` sets, two perpendicular
    offset ``GetAllCellsInLine`` walks including the error==0 double-cell
    branch, then each row's strict interior.  The fixed buffers are valid for
    the local-route window; exhaustion is deliberately blocked as well.
    """
    dtype = jnp.result_type(x0, y0, x1, y1, radius, jnp.float32)
    x0, y0 = jnp.asarray(x0, dtype), jnp.asarray(y0, dtype)
    x1, y1 = jnp.asarray(x1, dtype), jnp.asarray(y1, dtype)
    radius = jnp.asarray(radius, dtype)
    height, width = terrain.walkable.shape
    walk = terrain.walkable.reshape(-1)
    rad = radius / jnp.asarray(terrain.cell_size, dtype)
    span = int(span_cells)
    big = jnp.int32(1 << 29)
    ybase = jnp.floor(jnp.minimum(y0, y1) - rad).astype(jnp.int32) - 1

    def walkable_cell(ix, iy):
        linear = iy * jnp.int32(width) + ix
        valid = ((ix >= 0) & (ix <= width) & (iy >= 0) & (iy <= height)
                 & (linear >= 0) & (linear < walk.shape[0]))
        return valid & walk[jnp.clip(linear, 0, walk.shape[0] - 1)]

    def add(lo, hi, bad, ix, iy, active):
        """Add one or many source-enumerated cells to row extrema."""
        local_y = iy - ybase
        inside = active & (local_y >= 0) & (local_y < span)
        safe_y = jnp.clip(local_y, 0, span - 1).reshape(-1)
        flat_ix = ix.reshape(-1)
        flat_iy = iy.reshape(-1)
        flat_inside = inside.reshape(-1)
        lo = lo.at[safe_y].min(jnp.where(flat_inside, flat_ix, big))
        hi = hi.at[safe_y].max(jnp.where(flat_inside, flat_ix, -big))
        bad = bad | jnp.any(flat_inside & ~jax.vmap(walkable_cell)(flat_ix, flat_iy))
        return lo, hi, bad

    lo0 = jnp.full((span,), big, jnp.int32)
    hi0 = jnp.full((span,), -big, jnp.int32)
    bad0 = jnp.asarray(False)
    off = jnp.arange(-MAX_TERRAIN_RADIUS_CELLS,
                     MAX_TERRAIN_RADIUS_CELLS + 1, dtype=jnp.int32)

    def add_endpoint(carry, px, py):
        lo, hi, bad = carry
        bx, by = _trunc_to_i32(px), _trunc_to_i32(py)
        ix, iy = jnp.broadcast_arrays(bx + off[:, None], by + off[None, :])
        dx = jnp.maximum(jnp.abs(px - (ix.astype(dtype) + .5)) - .5, 0.)
        dy = jnp.maximum(jnp.abs(py - (iy.astype(dtype) + .5)) - .5, 0.)
        active = ((ix >= _trunc_to_i32(px-rad)) & (ix <= _trunc_to_i32(px+rad))
                  & (iy >= _trunc_to_i32(py-rad)) & (iy <= _trunc_to_i32(py+rad))
                  & (dx*dx + dy*dy <= rad*rad))
        return add(lo, hi, bad, ix, iy, active)

    lo0, hi0, bad0 = add_endpoint((lo0, hi0, bad0), x0, y0)
    lo0, hi0, bad0 = add_endpoint((lo0, hi0, bad0), x1, y1)
    length = jnp.sqrt((x1-x0)**2 + (y1-y0)**2)
    safe = jnp.where(length == 0, jnp.ones_like(length), length)
    px = jnp.where(length == 0, 0., -(y1-y0)/safe*rad)
    py = jnp.where(length == 0, 0., (x1-x0)/safe*rad)

    def consume_line(carry, ax, ay, bx, by):
        dx, dy = jnp.abs(bx-ax), jnp.abs(by-ay)
        ix0, iy0 = jnp.floor(ax).astype(jnp.int32), jnp.floor(ay).astype(jnp.int32)
        fx, fy = jnp.floor(bx).astype(jnp.int32), jnp.floor(by).astype(jnp.int32)
        xp, yp = bx > ax, by > ay
        xi = jnp.where(dx == 0, 0, jnp.where(xp, 1, -1)).astype(jnp.int32)
        yi = jnp.where(dy == 0, 0, jnp.where(yp, 1, -1)).astype(jnp.int32)
        ex = jnp.where(dx == 0, jnp.inf, jnp.where(xp, (ix0.astype(dtype)+1-ax)*dy, (ax-ix0.astype(dtype))*dy))
        ey = jnp.where(dy == 0, -jnp.inf, jnp.where(yp, -(iy0.astype(dtype)+1-ay)*dx, -(ay-iy0.astype(dtype))*dx))
        n = jnp.int32(1) + jnp.where(dx == 0, 0, jnp.abs(fx-ix0)) + jnp.where(dy == 0, 0, jnp.abs(fy-iy0))
        def body(c, _):
            ix, iy, err, left, lo, hi, bad = c
            active = left > 0
            lo, hi, bad = add(lo, hi, bad, ix, iy, active)
            tie = err == 0
            lo, hi, bad = add(lo, hi, bad, ix+xi, iy, active & tie)
            lo, hi, bad = add(lo, hi, bad, ix, iy+yi, active & tie)
            gt, lt = err > 0, err < 0
            # NavigationGrid.GetAllCellsInLine: positive error advances Y,
            # negative advances X; a zero tie visits both then advances both.
            return (jnp.where(lt|tie, ix+xi, ix), jnp.where(gt|tie, iy+yi, iy),
                    jnp.where(gt, err-dx, jnp.where(lt, err+dy, err+dy-dx)),
                    left-jnp.where(active, 1+tie.astype(jnp.int32), 0), lo, hi, bad), None
        ix, iy, err, left, lo, hi, bad = carry
        (ix, iy, err, left, lo, hi, bad), _ = jax.lax.scan(body, (ix0, iy0, ex+ey, n, lo, hi, bad), None, length=max_line_steps)
        return (ix, iy, err, left, lo, hi, bad), left > 0

    initial = (jnp.int32(0), jnp.int32(0), jnp.asarray(0., dtype), jnp.int32(0), lo0, hi0, bad0)
    a, ex_a = consume_line(initial, x0+px, y0+py, x1+px, y1+py)
    b, ex_b = consume_line(a, x0-px, y0-py, x1-px, y1-py)
    _ix, _iy, _err, _left, lo, hi, bad = b
    # ``for (int x = xRanges[y,0] + 1; x < xRanges[y,1]; x++)``: the strict
    # interior of each row's enumerated span.  Two prefix reads per row answer
    # it exactly; see :func:`row_prefix` for why that matters here.
    ys = ybase + jnp.arange(span, dtype=jnp.int32)
    prefix = terrain.walkable_prefix
    if prefix is None:
        prefix = row_prefix(terrain.walkable)
    first = lo + 1
    run = jnp.maximum(hi - first, 0)          # untouched rows: lo=big, hi=-big
    live = (run > 0) & (ys >= 0) & (ys < height)
    # Off-grid cells are NOT walkable on the server (``GetCell`` returns null
    # and ``IsWalkable(null)`` is false), so an interior run that leaves the
    # grid blocks rather than being skipped.
    off_grid = live & ((first < 0) | (hi > width))
    safe_y = jnp.clip(ys, 0, height - 1)
    walkable_run = (prefix[safe_y, jnp.clip(hi, 0, width)]
                    - prefix[safe_y, jnp.clip(first, 0, width)])
    row_blocked = live & (off_grid | (walkable_run != run))
    # The x extent used to need its own bound, because the interior scan was a
    # fixed span x span window and a wider row would have gone unexamined.  The
    # prefix form reads the whole row, so that bound is gone -- not relaxed,
    # unnecessary.  The y span still bounds how many rows can be recorded.
    exhausted = ex_a | ex_b
    blocked = bad | jnp.any(row_blocked)
    return blocked | exhausted, exhausted


def closest_terrain_exit(x: jax.Array, y: jax.Array, radius: jax.Array,
                         terrain: TerrainGrid,
                         max_steps: int = MAX_TERRAIN_EXIT_STEPS):
    """``GetClosestTerrainExit``'s cumulative, drifting spiral.

    Returns ``(x, y, exhausted)``.  ``exhausted`` is false for a true server
    exit and makes the JAX-only safety cap observable to corpus diagnostics.
    The caller must treat a true value as a bound failure, not a walkable
    result.  ``lax.while_loop`` executes zero iterations for a walkable seed,
    unlike an unrolled masked scan that would pay the full cap every tick.
    """
    dtype = jnp.result_type(x, y, radius, jnp.float32)
    radius = jnp.asarray(radius, dtype)
    max_steps_i = jnp.asarray(max_steps, jnp.int32)

    def cond(carry):
        px, py, r = carry
        return (~is_walkable(px, py, radius, terrain)) & (r <= max_steps_i)

    def body(carry):
        px, py, r = carry
        # C# starts with angle=pi/4 and increments it after each cumulative
        # displacement.  r*pi/4 gives the same sequence in float32 geometry.
        angle = r.astype(dtype) * jnp.asarray(jnp.pi / 4.0, dtype)
        px = px + r.astype(dtype) * jnp.cos(angle)
        py = py + r.astype(dtype) * jnp.sin(angle)
        return px, py, r + jnp.int32(1)

    px, py, next_r = jax.lax.while_loop(
        cond, body, (jnp.asarray(x, dtype), jnp.asarray(y, dtype), jnp.int32(1)))
    return px, py, ~is_walkable(px, py, radius, terrain)


def exit_terrain_collision(x: jax.Array, y: jax.Array,
                           pathfinding_radius: jax.Array,
                           terrain: TerrainGrid):
    """Terrain branch of ``AttackableUnit.OnCollision``.

    ``CollisionHandler`` detects terrain with a point query, then
    ``OnCollision`` calls ``GetClosestTerrainExit(..., PathfindingRadius+1)``.
    Those are intentionally different radii.
    """
    inside = ~is_walkable(x, y, jnp.asarray(0.0, jnp.asarray(x).dtype), terrain)

    def exit_branch(_):
        return closest_terrain_exit(x, y, pathfinding_radius + 1.0, terrain)

    def stay_branch(_):
        return x, y, jnp.asarray(False)

    return jax.lax.cond(inside, exit_branch, stay_branch, operand=None)


def exit_blocked_escape(x: jax.Array, y: jax.Array,
                        pathfinding_radius: jax.Array,
                        terrain: TerrainGrid):
    """Terrain fallback after a unit-to-unit circle escape point.

    This mirrors ``if (!IsWalkable(exit, PathfindingRadius))
GetClosestTerrainExit(exit, PathfindingRadius + 1)`` exactly, including the
one-unit difference between the guard and the eventual exit threshold.
    """
    blocked = ~is_walkable(x, y, pathfinding_radius, terrain)

    def exit_branch(_):
        return closest_terrain_exit(x, y, pathfinding_radius + 1.0, terrain)

    def stay_branch(_):
        return x, y, jnp.asarray(False)

    return jax.lax.cond(blocked, exit_branch, stay_branch, operand=None)


def repair_collision_terrain_batch(x: jax.Array, y: jax.Array,
                                   pathfinding_radius: jax.Array,
                                   moved_by_unit: jax.Array,
                                   terrain: TerrainGrid,
                                   max_steps: int = MAX_TERRAIN_EXIT_STEPS):
    """Vectorized deferred collision-terrain repair for one environment.

    Unmoved units use CollisionHandler's point-query guard; units moved by a
    circle escape use OnCollision's radius-aware guard.  The spiral itself is
    independent per unit, but one masked while-loop lets the common case (no
    blocked result anywhere in the lane) execute zero iterations instead of
    lowering one conditional/while pair per unit.

    Deferring repair until the dynamic sweep completes is not source-order
    exact when an escape into terrain would have changed a later neighbour
    test.  Callers must expose that approximation; this helper only makes the
    deferred mode efficient and preserves each individual terrain query.
    """
    x, y = jnp.asarray(x), jnp.asarray(y)
    radius = jnp.asarray(pathfinding_radius, x.dtype)
    moved = jnp.asarray(moved_by_unit, bool)
    guard_radius = jnp.where(moved, radius, jnp.zeros_like(radius))
    walkable = jax.vmap(lambda px, py, r: is_walkable(px, py, r, terrain))(
        x, y, guard_radius)
    active0 = ~walkable
    r0 = jnp.ones_like(radius, dtype=jnp.int32)
    cap = jnp.asarray(max_steps, jnp.int32)
    exit_radius = radius + jnp.asarray(1.0, radius.dtype)

    def cond(carry):
        _px, _py, step, active = carry
        return jnp.any(active & (step <= cap))

    def body(carry):
        px, py, step, active = carry
        angle = step.astype(px.dtype) * jnp.asarray(jnp.pi / 4.0, px.dtype)
        nx = px + step.astype(px.dtype) * jnp.cos(angle)
        ny = py + step.astype(py.dtype) * jnp.sin(angle)
        px = jnp.where(active, nx, px)
        py = jnp.where(active, ny, py)
        now_walkable = jax.vmap(
            lambda qx, qy, rr: is_walkable(qx, qy, rr, terrain))(
                px, py, exit_radius)
        active = active & ~now_walkable
        step = jnp.where(active, step + jnp.int32(1), step)
        return px, py, step, active

    px, py, _step, exhausted = jax.lax.while_loop(
        cond, body, (x, y, r0, active0))
    return px, py, exhausted
