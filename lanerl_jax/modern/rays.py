"""Navgrid sight rays: a bounded supercover walk that checks every crossed cell (both side cells at a
corner, conservative for float32 ties). Walls block unless transparent (0x40) or always-visible (0x100).
From outside brush every brush cell blocks; from inside brush, a target in brush must be reached through brush."""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .core.arrays import first_true


class VisionGrid(NamedTuple):
    flags: jax.Array
    cell_size: float
    min_x: float
    min_y: float
    bush_ids: jax.Array | None = None


def with_bush_ids(grid):
    """Label brush patches once on the host (edge-connected cells; 0 = no brush)."""
    import numpy as np
    from scipy.ndimage import label
    ids, _ = label((np.asarray(grid.flags) & 1) != 0)
    return grid._replace(bush_ids=jnp.asarray(ids, jnp.int32))


def bush_visible(grid, x0, y0, x1, y1, *, enabled=True):
    """``fog="fast"``: visible when the target is outside brush or in the viewer's brush patch (no walls)."""
    height, width = grid.bush_ids.shape

    def lookup(x, y):
        x = jnp.floor((x-grid.min_x)/grid.cell_size).astype(jnp.int32)
        y = jnp.floor((y-grid.min_y)/grid.cell_size).astype(jnp.int32)
        valid = (x >= 0) & (y >= 0) & (x < width) & (y < height)
        return grid.bush_ids[jnp.clip(y, 0, height-1), jnp.clip(x, 0, width-1)], valid

    start, valid0 = lookup(x0, y0)
    end, valid1 = lookup(x1, y1)
    return enabled & valid0 & valid1 & ((end == 0) | (start == end))


def clear_ray(grid, x0, y0, x1, y1, *, enabled=True):
    """Wall/brush ray test (fused kernel on CUDA, ``clear_ray_reference`` elsewhere); brush lookup only
    when the grid has ``bush_ids``."""
    if grid.bush_ids is not None:
        return bush_visible(grid, x0, y0, x1, y1, enabled=enabled)
    from .ray_kernel import clear_ray_fused
    def reference(*args):
        return clear_ray_reference(grid, *args[:4], enabled=args[4])
    def fused(*args):
        return clear_ray_fused(grid, *args[:4], enabled=args[4])
    return jax.lax.platform_dependent(x0, y0, x1, y1, enabled,
                                      cuda=fused, default=reference)


def clear_pairs(grid, x0, y0, x1, y1, enabled, capacity=None):
    """``clear_ray`` from (V,) viewers to (T,) targets as (V, T), cast only for the enabled pairs, compacted to
    ``capacity`` rays (None = all). ``(clear, dropped)``: ``dropped`` enabled pairs past capacity read False."""
    if capacity is None or capacity >= enabled.size or grid.bush_ids is not None:
        return clear_ray(grid, x0[:, None], y0[:, None], x1[None, :], y1[None, :], enabled=enabled), jnp.int32(0)
    flat = enabled.reshape(-1)
    pair, count = first_true(flat, capacity)
    v, t = pair // x1.shape[0], pair % x1.shape[0]                  # fill slots clamp on gather, drop on scatter
    clear = clear_ray(grid, x0[v], y0[v], x1[t], y1[t], enabled=pair < flat.size)
    clear = jnp.zeros_like(flat).at[pair].set(clear, mode="drop").reshape(enabled.shape)
    return clear, jnp.maximum(count - capacity, 0)


def clear_ray_reference(grid, x0, y0, x1, y1, *, enabled=True):
    """Broadcastable supercover rays; false outside the grid, past 64 cells or where disabled."""
    if grid.bush_ids is not None:
        return bush_visible(grid, x0, y0, x1, y1, enabled=enabled)
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

    initial = (ix, iy, error, remaining, valid0 & valid1 & enabled)
    def pending(carry):
        iteration, (_, _, _, left, clear) = carry
        return (iteration < 64) & jnp.any((left > 0) & clear)
    def advance(carry):
        iteration, state = carry
        return iteration + 1, step(iteration, state)
    _, (_, _, _, left, clear) = jax.lax.while_loop(pending, advance, (0, initial))
    return clear & (left <= 0)
