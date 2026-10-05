"""Structure footprints on the 26.19 Map11 navgrid and per-team dynamic walkable masks.

Evidence (docs/modern/LANES_TERRAIN.md §2):

* The navgrid marks every structure pedestal with the ``StructureWall`` flag
  (0x4; always together with WALL|TRANSPARENT, flag value 70). Connected
  components: 22 turret pads of 18-21 cells (5x5 cells = 250 units, the
  turret pathfinding radius 125, CLIENT H), 6 inhibitor pads of 44-61 cells
  (radius ~213.75, CLIENT H), 2 Nexus pads of 157/158 cells (radius ~304,
  CLIENT H) and two small fountain-platform pieces (not structures).
* Whether a footprint opens when its structure dies: turrets **do not**
  ("Turrets are units located on top of impassable terrain. This terrain
  remains even after the turret is destroyed", wiki Turret, WIKI H; the
  client also has a ``TurretRubble`` character with pathfinding radius 100).
  For inhibitors no source says either way; the client ships navgrid
  overlays only for the Baron pit and the dragon-soul terrain, none for
  structures, and the StructureWall cells are baked into the static base
  grid, so the default is that inhibitor pads stay blocked too (INFERRED M).
  A Nexus dying ends the game.

So the default ``release`` policy keeps every footprint blocked, which makes
``walkable_masks`` an identity on the base masks; the function exists so a
ruleset (or a measurement proving otherwise) can release footprints per
structure kind without touching the tick. Destroyed structures already stop
blocking *units* (dead units are skipped by unit collision).

Everything here is host-side numpy (``build_footprints``) or fixed-shape JAX
(``walkable_masks``, ``eject``) over the ``StaticTerrain`` contract.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..core import types as W
from .terrain import is_walkable, team_view

STRUCTURE_FLAG = 4
WALL_FLAG = 2
TRANSPARENT_FLAG = 64
FOOTPRINT_MAX_DISTANCE = 450.0      # component centroid to structure position (largest pad radius ~400)
# Release-on-death policy per structure kind (see module docstring).
RELEASE_ON_DEATH = {W.KIND_TURRET: False, W.KIND_INHIBITOR: False, W.KIND_NEXUS: False}


class Footprints(NamedTuple):
    """``owner[z, x]``: world unit index of the structure whose pad covers the cell, -1 none."""
    owner: Any              # (Z, X) int32
    cells: Any              # (N,) int32 footprint cell count per unit (0 for non-structures)


def build_footprints(grid, unit_kind, unit_x, unit_y, *, max_distance: float = FOOTPRINT_MAX_DISTANCE) -> Footprints:
    """Assign each 8-connected StructureWall component of ``grid`` to the nearest structure unit.

    ``grid`` is a ``data.modern_map.ModernMapGrid``; the unit arrays are the
    static world layout (``WorldConfig.unit_kind/unit_x/unit_y``). Components
    farther than ``max_distance`` from every structure (fountain platform
    pieces) stay unowned, i.e. permanently blocked. Raises if a structure
    gets no footprint, so a layout/navgrid mismatch cannot pass silently.
    """
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
    """Per-team ``StaticTerrain`` with the pads of dead, released structures opened.

    ``terrain`` is ``WorldConfig.terrain`` (one StaticTerrain per team, gates
    resolved); ``alive``/``release`` are (N,) bool. A released cell becomes
    walkable for both teams (its WALL|TRANSPARENT bits belong to the pad);
    a respawned structure (inhibitor) blocks its pad again, so call
    ``eject`` for units standing on it. Pure and jit-safe; O(Z*X).
    """
    owner = footprints.owner
    n = jnp.asarray(alive).shape[0]
    o = jnp.clip(owner, 0, n - 1)
    opened = (owner >= 0) & ~jnp.asarray(alive, bool)[o] & jnp.asarray(release, bool)[o]
    return tuple(t._replace(walkable=jnp.asarray(t.walkable, bool) | opened) for t in terrain)


_EJECT_RINGS = (50.0, 100.0, 150.0, 200.0, 275.0, 350.0, 450.0)
_EJECT_DIRS = 16
EJECT_MAX_UNITS = 16                     # ring searches per call; further stuck units move on later calls


def eject(x, y, team, radius, terrain: tuple, active=None, max_units: int = EJECT_MAX_UNITS):
    """Move each unit (N,) whose disk is not walkable to the nearest walkable point on
    rings of 50-450 units (16 directions); units already clear, or inactive, stay put.
    Use after a footprint closes (inhibitor respawn) or after a blink/dash ends.

    Every unit is tested, but only the first ``max_units`` stuck units (slot order) are
    searched per call: closures are rare and local, and the world calls this every tick,
    so any remaining stuck units are moved on the following ticks.
    """
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
    (sel,) = jnp.nonzero(stuck, size=min(max_units, n), fill_value=n)    # n = no unit (dropped below)
    i = jnp.clip(sel, 0, n - 1)
    found, nx, ny = jax.vmap(search)(x[i], y[i], team[i], r[i])
    return (x.at[sel].set(jnp.where(found, nx, x[i]), mode="drop"),
            y.at[sel].set(jnp.where(found, ny, y[i]), mode="drop"))


__all__ = ["STRUCTURE_FLAG", "RELEASE_ON_DEATH", "Footprints", "build_footprints", "release_mask",
           "walkable_masks", "eject"]
