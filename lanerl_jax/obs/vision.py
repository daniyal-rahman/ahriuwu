"""Map-grid vision rays for the source server's brush and wall rules.

Port of NavigationGrid.CastRay(checkVisible=True). A bounded supercover walk
checks every crossed cell, including both side cells at a corner. Near-corner
float32 ties are treated conservatively as crossings of both side cells;
the server accumulates its error in double precision. This can hide an extra
sliver at a grid corner and is recorded as an approximation, not exact parity.
"""
from functools import lru_cache
from typing import NamedTuple

import jax
import jax.numpy as jnp


class VisionGrid(NamedTuple):
    flags: jax.Array
    cell_size: float
    min_x: float
    min_y: float


@lru_cache(maxsize=1)
def map1_vision():
    from ..data.navgrid import NavGrid
    grid = NavGrid.load()
    return VisionGrid(jnp.asarray(grid.flags), float(grid.cell_size),
                      float(grid.min_grid[0]), float(grid.min_grid[2]))


def clear_ray(grid, x0, y0, x1, y1, *, enabled=True):
    """Broadcastable rays. Fail closed outside the grid or the 64-cell bound.

    Disabled pairs return false and do not keep the bounded loop active.
    Map1 sight radius <=1200 and cell size 50 imply at most 36 crossings.
    64 is a bound for this visibility task, not a general-purpose ray caster.
    """
    x0, y0, x1, y1 = jnp.broadcast_arrays(
        (x0-grid.min_x)/grid.cell_size, (y0-grid.min_y)/grid.cell_size,
        (x1-grid.min_x)/grid.cell_size, (y1-grid.min_y)/grid.cell_size)
    ix, iy = jnp.floor(x0).astype(jnp.int32), jnp.floor(y0).astype(jnp.int32)
    ex, ey = jnp.floor(x1).astype(jnp.int32), jnp.floor(y1).astype(jnp.int32)
    dx, dy = jnp.abs(x1-x0), jnp.abs(y1-y0)
    sx, sy = jnp.sign(x1-x0).astype(jnp.int32), jnp.sign(y1-y0).astype(jnp.int32)
    error = jnp.where(sx > 0, ix+1-x0, x0-ix)*dy - jnp.where(sy > 0, iy+1-y0, y0-iy)*dx
    error = jnp.where(dx == 0, jnp.inf, error)
    error = jnp.where(dy == 0, -jnp.inf, error)
    remaining = 1+jnp.abs(ex-ix)+jnp.abs(ey-iy)
    height, width = grid.flags.shape

    def flags_at(x, y):
        valid = (x >= 0) & (y >= 0) & (x < width) & (y < height)
        return grid.flags[jnp.clip(y, 0, height-1), jnp.clip(x, 0, width-1)], valid

    start, valid0 = flags_at(ix, iy)
    end, valid1 = flags_at(ex, ey)
    start_grass, end_grass = (start & 1) != 0, (end & 1) != 0

    def cell_clear(x, y):
        flags, valid = flags_at(x, y)
        transparent = ((flags & 2) == 0) | ((flags & (0x40 | 0x100)) != 0)
        grass = (flags & 1) != 0
        brush_ok = jnp.where(start_grass, ~end_grass | grass, ~grass)
        return valid & transparent & brush_ok

    def step(_, carry):
        x, y, err, left, clear = carry
        active = (left > 0) & clear
        # Conservative around double-vs-float32 corner ambiguity. Crossing
        # both side cells avoids leaking vision through a blocked corner.
        corner = jnp.abs(err) <= 1e-3
        ok = cell_clear(x, y) & (~corner | (cell_clear(x+sx, y) & cell_clear(x, y+sy)))
        clear &= ~active | ok
        move_x = (err < 0) | corner
        move_y = (err > 0) | corner
        x += jnp.where(active & move_x, sx, 0)
        y += jnp.where(active & move_y, sy, 0)
        err += jnp.where(move_x & move_y, dy-dx, jnp.where(move_x, dy, -dx))
        left -= jnp.where(active, 1+corner.astype(jnp.int32), 0)
        return x, y, err, left, clear

    # Dead/out-of-range pairs and rays already stopped by brush/walls cannot
    # extend the loop. The scalar bound still fails closed for long rays.
    initial = (ix, iy, error, remaining, valid0 & valid1 & enabled)
    def pending(carry):
        iteration, (_, _, _, left, clear) = carry
        return (iteration < 64) & jnp.any((left > 0) & clear)
    def advance(carry):
        iteration, state = carry
        return iteration + 1, step(iteration, state)
    _, (_, _, _, left, clear) = jax.lax.while_loop(pending, advance, (0, initial))
    return clear & (left <= 0)
