"""Structure footprints on the 26.19 navgrid and per-team dynamic walkable masks (LANES_TERRAIN §2).

Every structure pad carries the ``StructureWall`` navgrid flag. Turret pads stay blocked after death (wiki Turret);
for inhibitors no source says, and the pads are baked into the static grid, so by default every footprint stays
blocked (INFERRED M) and ``walkable_masks`` is the identity; ``RELEASE_ON_DEATH`` lets a ruleset open them.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..core import types as W
from ..core.arrays import first_true
from .terrain import is_walkable, row_gaps, team_view

STRUCTURE_FLAG = 4
FOOTPRINT_MAX_DISTANCE = 450.0      # component centroid to structure position (largest pad radius ~400)
RELEASE_ON_DEATH = {W.KIND_TURRET: False, W.KIND_INHIBITOR: False, W.KIND_NEXUS: False}


class Footprints(NamedTuple):
    """``owner[z, x]``: world unit index of the structure whose pad covers the cell, -1 none."""
    owner: Any              # (Z, X) int32
    cells: Any              # (N,) int32 footprint cell count per unit (0 for non-structures)


def build_footprints(grid, unit_kind, unit_x, unit_y, *, max_distance: float = FOOTPRINT_MAX_DISTANCE) -> Footprints:
    """Host-side: give each 8-connected StructureWall component to the nearest structure within ``max_distance``
    (farther pieces stay blocked); raises if a structure gets no footprint."""
    from scipy import ndimage
    kind = np.asarray(unit_kind)
    xs, ys = np.asarray(unit_x, np.float64), np.asarray(unit_y, np.float64)
    struct = np.isin(kind, W.STRUCTURE_KINDS)
    mask = (np.asarray(grid.flags) & STRUCTURE_FLAG) != 0
    lab, count = ndimage.label(mask, structure=np.ones((3, 3), bool))
    owner = np.full(mask.shape, -1, np.int32)
    cs, x0, z0 = grid.cell_size, grid.min_bounds[0], grid.min_bounds[2]
    sidx = np.nonzero(struct)[0]
    centers = ndimage.center_of_mass(mask, lab, range(1, count + 1))
    for comp, (cz, cx) in enumerate(centers, start=1):
        wx, wz = x0 + (cx + 0.5) * cs, z0 + (cz + 0.5) * cs
        d = np.hypot(xs[sidx] - wx, ys[sidx] - wz)
        k = int(np.argmin(d))
        if d[k] <= max_distance:
            owner[lab == comp] = sidx[k]
    cells = np.bincount(owner[owner >= 0].ravel(), minlength=len(kind)).astype(np.int32)
    missing = sidx[cells[sidx] == 0]
    if missing.size:
        raise ValueError(f"structures without a navgrid footprint: {missing.tolist()}")
    return Footprints(jnp.asarray(owner), jnp.asarray(cells))


def release_mask(unit_kind, policy: dict | None = None) -> Any:
    """(N,) bool: the unit's footprint opens while it is dead (``RELEASE_ON_DEATH`` by default)."""
    policy = RELEASE_ON_DEATH if policy is None else policy
    kind = jnp.asarray(unit_kind, jnp.int32)
    out = jnp.zeros(kind.shape, bool)
    for k, rel in policy.items():
        out = out | ((kind == k) & bool(rel))
    return out


def walkable_masks(terrain: tuple, footprints: Footprints, alive, release) -> tuple:
    """Per-team terrain with the pads of dead, released structures opened for both teams; a respawn closes them
    again (``eject`` units standing there)."""
    owner = footprints.owner
    n = jnp.asarray(alive).shape[0]
    o = jnp.clip(owner, 0, n - 1)
    opened = (owner >= 0) & ~jnp.asarray(alive, bool)[o] & jnp.asarray(release, bool)[o]
    walk = tuple(jnp.asarray(t.walkable, bool) | opened for t in terrain)
    return tuple(t._replace(walkable=w, gaps=row_gaps(w)) for t, w in zip(terrain, walk))


_EJECT_RINGS = (50.0, 100.0, 150.0, 200.0, 275.0, 350.0, 450.0)
_EJECT_DIRS = 16
EJECT_MAX_UNITS = 16                     # ring searches per call; further stuck units move on later calls


def eject(x, y, team, radius, terrain: tuple, active=None, max_units: int = EJECT_MAX_UNITS):
    """Move active units whose disk is not walkable to the nearest walkable point on rings of 50-450 (16
    directions). Only the first ``max_units`` stuck units are searched per call; the rest move on later ticks."""
    x, y, r = (jnp.asarray(v, jnp.float32) for v in (x, y, radius))
    team = jnp.asarray(team, jnp.int32)
    act = jnp.ones(x.shape, bool) if active is None else jnp.asarray(active, bool)
    ang = jnp.arange(_EJECT_DIRS) * (2 * jnp.pi / _EJECT_DIRS)
    rings = jnp.asarray(_EJECT_RINGS, jnp.float32)
    cx = (rings[:, None] * jnp.cos(ang)[None, :]).reshape(-1)
    cy = (rings[:, None] * jnp.sin(ang)[None, :]).reshape(-1)
    cr = jnp.repeat(rings, _EJECT_DIRS)
    t0, t1 = terrain[0], terrain[-1]

    def walkable(px, py, tm, rr):
        return is_walkable(px, py, jnp.minimum(rr, 150.0), team_view((t0, t1), jnp.where(tm == 1, 1, 0)))

    def search(px, py, tm, rr):
        ok = jax.vmap(lambda u, v: walkable(px + u, py + v, tm, rr))(cx, cy)
        k = jnp.argmin(jnp.where(ok, cr, jnp.inf))
        return jnp.any(ok), px + cx[k], py + cy[k]

    n = x.shape[0]
    stuck = act & ~jax.vmap(walkable)(x, y, team, r)
    sel, _ = first_true(stuck, min(max_units, n))
    i = jnp.clip(sel, 0, n - 1)
    found, nx, ny = jax.vmap(search)(x[i], y[i], team[i], r[i])
    return (x.at[sel].set(jnp.where(found, nx, x[i]), mode="drop"),
            y.at[sel].set(jnp.where(found, ny, y[i]), mode="drop"))
