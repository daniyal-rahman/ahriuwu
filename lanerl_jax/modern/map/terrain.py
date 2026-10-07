"""Static navgrid walkability queries; unsupported radii fail closed (blocked)."""
from typing import NamedTuple

import jax
import jax.numpy as jnp

GAP_PAD = 8                   # blocked rows around ``row_gaps`` tables; bounds ``max_radius_cells`` below it
_GAP_CAP = 127                # int8; only gaps up to max_radius_cells + 1 matter


class StaticTerrain(NamedTuple):
    walkable: object          # (H, W) bool, or (L, H, W) stacked masks read through ``layer``
    cell_size: float
    min_x: float
    min_z: float
    max_x: float
    max_z: float
    layer: object = None      # index into stacked ``walkable`` (e.g. per team); None = 2-D grid
    gaps: object = None       # ``row_gaps(walkable)``; None = derived per query


def row_gaps(walkable):
    """(..., W, H + 2*GAP_PAD, 2) int8: per column x and row z (GAP_PAD blocked rows each side), the distance in
    cells to the nearest blocked cell at or left / right of x in row z, off-grid cells blocked, capped at 127."""
    blocked = ~jnp.asarray(walkable, bool)
    w = blocked.shape[-1]
    i = jnp.arange(w, dtype=jnp.int32)
    left = i - jax.lax.cummax(jnp.where(blocked, i, -1), axis=blocked.ndim - 1)
    right = jax.lax.cummin(jnp.where(blocked, i, w), axis=blocked.ndim - 1, reverse=True) - i
    g = jnp.minimum(jnp.stack([left, right], -1), _GAP_CAP).astype(jnp.int8)
    g = jnp.swapaxes(g, -3, -2)
    return jnp.pad(g, ((0, 0),) * (g.ndim - 2) + ((GAP_PAD, GAP_PAD), (0, 0)))


def team_view(terrain: tuple, layer) -> StaticTerrain:
    """One team's terrain from the per-team tuple as a ``layer`` of the stacked masks (no grid copy under vmap)."""
    return terrain[0]._replace(walkable=jnp.stack([jnp.asarray(t.walkable, bool) for t in terrain]),
                               gaps=jnp.stack([row_gaps(t.walkable) if t.gaps is None else t.gaps for t in terrain]),
                               layer=jnp.asarray(layer, jnp.int32))


def is_walkable(x, z, radius, terrain: StaticTerrain, *, max_radius_cells: int = 3):
    """Conservative disk/cell intersection. A zero radius selects its half-open cell; a disk touching the border is
    blocked. ``max_radius_cells`` is a static bound on the radius in cells.

    Same as testing every cell of the (k, k) window around the base cell of the grid padded with blocked cells: per
    window row only the nearest blocked cell on each side of the base column can be the closest, so ``row_gaps``
    leaves two candidates per row."""
    if not 1 <= max_radius_cells < GAP_PAD:
        raise ValueError(f"max_radius_cells must be in [1, {GAP_PAD})")
    x, z, radius = (jnp.asarray(v, jnp.float32) for v in (x, z, radius))
    nx, nz = (x-terrain.min_x)/terrain.cell_size, (z-terrain.min_z)/terrain.cell_size
    r = radius/terrain.cell_size
    base_x, base_z = jnp.floor(nx).astype(jnp.int32), jnp.floor(nz).astype(jnp.int32)
    m, k = max_radius_cells + 1, 2 * max_radius_cells + 3
    gaps = row_gaps(terrain.walkable) if terrain.gaps is None else terrain.gaps
    # The window of an off-grid base is clamped onto the grid, while distances stay measured from the base.
    sx = jnp.clip(base_x, 0, gaps.shape[-3] - 1)
    sz = jnp.clip(base_z, 0, gaps.shape[-2] - 2 * GAP_PAD - 1) + GAP_PAD - m
    if terrain.layer is None:
        g = jax.lax.dynamic_slice(gaps, (sx, sz, jnp.int32(0)), (1, k, 2))[0]
    else:
        g = jax.lax.dynamic_slice(gaps, (terrain.layer, sx, sz, jnp.int32(0)), (1, 1, k, 2))[0, 0]
    ix = base_x + g * jnp.array([-1, 1])
    iz = base_z + jnp.arange(-m, m+1)[:, None]
    dx = jnp.maximum(jnp.abs(nx-(ix+.5))-.5, 0)
    dz = jnp.maximum(jnp.abs(nz-(iz+.5))-.5, 0)
    disk = ~jnp.any(dx*dx + dz*dz <= r*r)
    point = g[m, 0] > 0
    point_bounds = ((x >= terrain.min_x) & (x < terrain.max_x)
                    & (z >= terrain.min_z) & (z < terrain.max_z))
    disk_bounds = ((x-radius > terrain.min_x) & (x+radius < terrain.max_x)
                   & (z-radius > terrain.min_z) & (z+radius < terrain.max_z))
    supported = jnp.isfinite(x) & jnp.isfinite(z) & jnp.isfinite(r) & (r >= 0) & (r <= max_radius_cells)
    return supported & jnp.where(radius == 0, point_bounds & point, disk_bounds & disk)
