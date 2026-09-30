"""Strict static-map queries for the modern port (not legacy NavGrid parity).

The caller selects a team's gate mask on map load. Dynamic terrain and modern
vision are deliberately outside this first map slice. Queries use fixed-shape
arrays under jit/vmap; unsupported radii return blocked, never a partial check.
"""
from typing import NamedTuple

import jax.numpy as jnp


class StaticTerrain(NamedTuple):
    walkable: object
    cell_size: float
    min_x: float
    min_z: float
    max_x: float
    max_z: float


def is_walkable(x, z, radius, terrain: StaticTerrain, *, max_radius_cells: int = 3):
    """Conservative disk/cell intersection, independently checked by host oracle.

    Float32 query coordinates. A zero-radius point selects its half-open cell,
    including at internal edges/corners. World maximum is exclusive.
    Positive-radius contact with the outer border
    is blocked. ``max_radius_cells`` is a static compilation bound (150 units
    at the common 50-unit cell size), not a maximum champion gameplay radius.
    """
    if max_radius_cells < 1:
        raise ValueError("max_radius_cells must be positive")
    x, z, radius = (jnp.asarray(v, jnp.float32) for v in (x, z, radius))
    nx, nz = (x-terrain.min_x)/terrain.cell_size, (z-terrain.min_z)/terrain.cell_size
    r = radius/terrain.cell_size
    width, height = terrain.walkable.shape[1], terrain.walkable.shape[0]
    base_x, base_z = jnp.floor(nx).astype(jnp.int32), jnp.floor(nz).astype(jnp.int32)
    offsets = jnp.arange(-max_radius_cells-1, max_radius_cells+2)
    ix, iz = base_x+offsets[None, :], base_z+offsets[:, None]
    dx = jnp.maximum(jnp.abs(nx-(ix+.5))-.5, 0)
    dz = jnp.maximum(jnp.abs(nz-(iz+.5))-.5, 0)
    touched = dx*dx + dz*dz <= r*r
    valid = (ix >= 0) & (ix < width) & (iz >= 0) & (iz < height)
    values = terrain.walkable[jnp.clip(iz, 0, height-1), jnp.clip(ix, 0, width-1)]
    disk = jnp.all(~touched | (valid & values))
    point = terrain.walkable[jnp.clip(base_z, 0, height-1), jnp.clip(base_x, 0, width-1)]
    point_bounds = ((x >= terrain.min_x) & (x < terrain.max_x)
                    & (z >= terrain.min_z) & (z < terrain.max_z))
    disk_bounds = ((x-radius > terrain.min_x) & (x+radius < terrain.max_x)
                   & (z-radius > terrain.min_z) & (z+radius < terrain.max_z))
    supported = jnp.isfinite(x) & jnp.isfinite(z) & jnp.isfinite(r) & (r >= 0) & (r <= max_radius_cells)
    return supported & jnp.where(radius == 0, point_bounds & point, disk_bounds & disk)
