"""Static navgrid walkability queries; unsupported radii fail closed (blocked)."""
from typing import NamedTuple

import jax
import jax.numpy as jnp


class StaticTerrain(NamedTuple):
    walkable: object          # (H, W) bool, or (L, H, W) stacked masks read through ``layer``
    cell_size: float
    min_x: float
    min_z: float
    max_x: float
    max_z: float
    layer: object = None      # index into stacked ``walkable`` (e.g. per team); None = 2-D grid


def team_view(terrain: tuple, layer) -> StaticTerrain:
    """One team's terrain from the per-team tuple as a ``layer`` of the stacked masks (no grid copy under vmap)."""
    return terrain[0]._replace(walkable=jnp.stack([jnp.asarray(t.walkable, bool) for t in terrain]),
                               layer=jnp.asarray(layer, jnp.int32))


def is_walkable(x, z, radius, terrain: StaticTerrain, *, max_radius_cells: int = 3):
    """Conservative disk/cell intersection. A zero radius selects its half-open cell; a disk touching the border is
    blocked. ``max_radius_cells`` is a static bound on the radius in cells."""
    if max_radius_cells < 1:
        raise ValueError("max_radius_cells must be positive")
    x, z, radius = (jnp.asarray(v, jnp.float32) for v in (x, z, radius))
    nx, nz = (x-terrain.min_x)/terrain.cell_size, (z-terrain.min_z)/terrain.cell_size
    r = radius/terrain.cell_size
    base_x, base_z = jnp.floor(nx).astype(jnp.int32), jnp.floor(nz).astype(jnp.int32)
    offsets = jnp.arange(-max_radius_cells-1, max_radius_cells+2)
    ix, iz = base_x+offsets[None, :], base_z+offsets[:, None]
    dx = jnp.maximum(jnp.abs(nx-(ix+.5))-.5, 0)
    dz = jnp.maximum(jnp.abs(nz-(iz+.5))-.5, 0)
    touched = dx*dx + dz*dz <= r*r
    # (k, k) window around the base cell of the grid padded with blocked cells; off-grid bases fail the bounds tests.
    m, k = max_radius_cells + 1, 2 * max_radius_cells + 3
    pad = ((0, 0),) * (terrain.walkable.ndim - 2) + ((m, m), (m, m))
    padded = jnp.pad(jnp.asarray(terrain.walkable, bool), pad)
    if terrain.layer is None:
        values = jax.lax.dynamic_slice(padded, (base_z, base_x), (k, k))
    else:
        values = jax.lax.dynamic_slice(padded, (terrain.layer, base_z, base_x), (1, k, k))[0]
    disk = jnp.all(~touched | values)
    point = values[m, m]
    point_bounds = ((x >= terrain.min_x) & (x < terrain.max_x)
                    & (z >= terrain.min_z) & (z < terrain.max_z))
    disk_bounds = ((x-radius > terrain.min_x) & (x+radius < terrain.max_x)
                   & (z-radius > terrain.min_z) & (z+radius < terrain.max_z))
    supported = jnp.isfinite(x) & jnp.isfinite(z) & jnp.isfinite(r) & (r >= 0) & (r <= max_radius_cells)
    return supported & jnp.where(radius == 0, point_bounds & point, disk_bounds & disk)
