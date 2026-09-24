"""Policy-action decoding shared by rollout and benchmark paths.

The target head names an *observation slot*, not a simulator unit index.  That
distinction is invisible while the enemy champion happens to be in slot 0, but
it is decisive for minions and turrets: slot 13 is an enemy-minion slot, not
unit 13.  Keep the conversion here, at the observation/action boundary, so the
simulator continues to receive only semantic unit-index orders.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from lanerl_rl.constants import BUTTON_INDEX, N_SCREEN_X, N_SCREEN_Y
from lanerl_rl.projection import (
    DEFAULT_CAM_Y,
    DEFAULT_FOV_V_DEG,
    DEFAULT_RESOLUTION,
    DEFAULT_TILT_DEG,
    FLOOR_Y,
)

from ..sim.config import ROUTE_PATHFINDING_RADIUS
from ..sim.orders import OrderKind, Orders
from ..sim.state import Team

__all__ = ["orders_from", "MoveSnapTable", "move_snap_table", "snap_move_point",
           "MOVE_SNAP_RADIUS"]


# ---------------------------------------------------------------------------
# MOVE-point snap (`PATH-010`, `docs/NONFARMING_FAILURES.md` R2).
#
# Real League paths a click on unwalkable or off-map ground to the nearest
# reachable point, walks there and stops.  The vendored server does not: a
# Move whose `GetPath` is null gets the two-point waypoint list
# `[pos, raw click]` (`LanerlControl.cs:368`), walks the raw line into the
# wall and stays pinned there, jittering, with the order never completing.
# The sim reproduces that faithfully (`SERVER_NULL`, deferred terrain repair).
# From either spawn ~50% of the 96x54 screen bins decode off-grid, ~58%
# decode to a null path and 71.8% land on a cell a 35-u champion cannot stand
# on (36-48% from mid-lane positions; `tests/test_move_snap.py`); the pin
# then makes ~99% of the next clicks null as well.  That is how c13s1's blue sat at its fountain for 62% of a game.
#
# This is fixed in the ACTION SPACE, not the engine: the decoder never emits
# a Move goal a 35-u champion cannot stand on.  Every MOVE point (plain move
# and the targetless attack-move fallback) is clamped to the navgrid and, if
# its cell is not standable, replaced by the centre of the nearest standable
# cell.  A click whose cell is already standable is passed through
# bit-identically.  `parity/policy_driver.py` decodes through `orders_from`
# too, so the server receives the same snapped Move and sim/server parity is
# untouched.  What this does NOT cure: a source that already overlaps terrain
# (first-expansion null, `PATH-008`) still gets a null path; the snap only
# removes the goal-side trap.
#
# "Standable" is the sim's own query: `terrain_jax.is_walkable(centre, 35)`,
# the radius the route artifact is baked for (`ROUTE_PATHFINDING_RADIUS`;
# Garen's `pathfinding_radius` in `profiles.py`).  On Map1 that is 47,477 of
# 86,142 cells, one 8-connected component, and exactly the route artifact's
# `source_cells`.  The nearest-cell map is an exact Euclidean distance
# transform over cell indices (`scipy.ndimage.distance_transform_edt`).
# Build cost: ~0.4 s on a CPU (first use, cached per process); the table is
# 294x293 int32 (344,568 bytes) plus a same-shape bool mask, closed over as
# JIT constants.
MOVE_SNAP_RADIUS = ROUTE_PATHFINDING_RADIUS


class MoveSnapTable(NamedTuple):
    """Static nearest-standable-cell map. Scalars are compile-time floats."""

    #: ``(H*W,)`` bool: the cell's centre is walkable at ``MOVE_SNAP_RADIUS``.
    standable: jax.Array
    #: ``(H*W,)`` int32: flat index of the nearest standable cell (itself if
    #: standable).
    nearest: jax.Array
    height: int
    width: int
    cell_size: float
    min_x: float
    min_y: float


def build_move_snap_table(terrain, radius: float = MOVE_SNAP_RADIUS) -> MoveSnapTable:
    """Precompute the snap table from a ``TerrainGrid``."""
    from scipy.ndimage import distance_transform_edt

    from ..sim.terrain_jax import is_walkable

    height, width = terrain.walkable.shape
    cx = (terrain.min_x + (np.arange(width) + 0.5) * terrain.cell_size).astype(np.float32)
    cy = (terrain.min_y + (np.arange(height) + 0.5) * terrain.cell_size).astype(np.float32)
    gx, gy = np.meshgrid(cx, cy)
    query = jax.jit(jax.vmap(
        lambda x, y: is_walkable(x, y, jnp.float32(radius), terrain)))
    standable = np.asarray(query(jnp.asarray(gx.ravel()),
                                 jnp.asarray(gy.ravel()))).reshape(height, width)
    if not standable.any():
        raise ValueError("no standable cell at this radius")
    _, (ny, nx) = distance_transform_edt(~standable, return_indices=True)
    nearest = (ny.astype(np.int64) * width + nx).astype(np.int32).reshape(-1)
    return MoveSnapTable(standable=jnp.asarray(standable.reshape(-1)),
                         nearest=jnp.asarray(nearest), height=int(height),
                         width=int(width), cell_size=float(terrain.cell_size),
                         min_x=float(terrain.min_x), min_y=float(terrain.min_y))


@lru_cache(maxsize=1)
def move_snap_table() -> MoveSnapTable:
    """Map1's table, built on first use (not at import: data-only importers
    must not need the vendored navgrid).

    The first call may happen while ``orders_from`` is being traced under
    ``jit``/``vmap``; ``ensure_compile_time_eval`` makes the build run eagerly
    there, so the cached arrays are concrete constants and never tracers.
    """
    from ..sim.terrain_jax import map1_terrain

    with jax.ensure_compile_time_eval():
        return build_move_snap_table(map1_terrain())


def snap_move_point(x, y, table: MoveSnapTable | None = None):
    """Clamp a world point to the grid and snap it to a standable cell.

    Points whose (in-grid) cell is standable are returned unchanged; all
    others go to the centre of the nearest standable cell of the clamped
    cell.  Elementwise, jit/vmap-safe.
    """
    t = move_snap_table() if table is None else table
    dtype = jnp.result_type(x, y, jnp.float32)
    cs = jnp.asarray(t.cell_size, dtype)
    nx = (x - jnp.asarray(t.min_x, dtype)) / cs
    ny = (y - jnp.asarray(t.min_y, dtype)) / cs
    in_grid = (nx >= 0) & (nx < t.width) & (ny >= 0) & (ny < t.height)
    ix = jnp.clip(jnp.floor(nx), 0, t.width - 1).astype(jnp.int32)
    iy = jnp.clip(jnp.floor(ny), 0, t.height - 1).astype(jnp.int32)
    cell = iy * jnp.int32(t.width) + ix
    keep = in_grid & t.standable[cell]
    tgt = t.nearest[cell]
    tx = jnp.asarray(t.min_x, dtype) + ((tgt % t.width).astype(dtype) + 0.5) * cs
    ty = jnp.asarray(t.min_y, dtype) + ((tgt // t.width).astype(dtype) + 0.5) * cs
    return jnp.where(keep, x, tx), jnp.where(keep, y, ty)


# The deployed 352x352 vision frame's measured minimap rectangle. A factored
# x/y head cannot represent a joint rectangular action mask without becoming
# autoregressive, so decode those sampled pairs as NOOP. This prevents the
# live client from reinterpreting a supposed local ground click as a global
# minimap order. See docs/archive/INFERENCE_FAILURE_ANALYSIS.md M11/H100.
MINIMAP_X_MIN = 275.0 / 352.0
MINIMAP_Y_MIN = 240.0 / 352.0


def _screen_to_centred_lane(sx, sy):
    """JAX form of ``projection.screen_to_world_centred(0, 0, ...)``."""
    dtype = jnp.result_type(sx, sy, jnp.float32)
    tilt = math.radians(DEFAULT_TILT_DEG)
    cos_t = jnp.asarray(math.cos(tilt), dtype)
    sin_t = jnp.asarray(math.sin(tilt), dtype)
    dy = jnp.asarray(FLOOR_Y - DEFAULT_CAM_Y, dtype)
    tan_v = jnp.asarray(math.tan(math.radians(DEFAULT_FOV_V_DEG) / 2.0), dtype)
    tan_h = tan_v * (DEFAULT_RESOLUTION[0] / DEFAULT_RESOLUTION[1])

    # centred_on(0, 0): choose cz so screen centre intersects world z=0.
    dz_centre = -dy * cos_t / sin_t
    camera_z = -dz_centre
    kv = (jnp.asarray(0.5, dtype) - sy) * 2.0 * tan_v
    den = kv * cos_t - sin_t
    dz = dy * (cos_t + kv * sin_t) / den
    vz = -dy * sin_t + dz * cos_t
    ds = (sx - jnp.asarray(0.5, dtype)) * 2.0 * tan_h * vz
    dn = camera_z + dz
    return ds, dn


def orders_from(action, state, slot_unit, frame=None,
                cfg_x=N_SCREEN_X, cfg_y=N_SCREEN_Y, snap_moves=True):
    """Decode factored policy actions into semantic champion orders.

    ``slot_unit`` has shape ``(n_champions, n_slots)`` and comes from the same
    observation that produced ``action``.  An invalid slot is represented by
    ``-1``.  Attack-move follows the wire's behaviour: attack a selected
    visible unit, otherwise move to the selected screen point.

    Every MOVE point is snapped to a cell a 35-u champion can stand on
    (:func:`snap_move_point`, `PATH-010`). ``snap_moves=False`` is the raw
    pre-snap decode, kept only for tests and before/after measurements.
    """
    button, sx, sy, target_slot = action
    n_slots = slot_unit.shape[-1]
    target_slot = jnp.clip(target_slot.astype(jnp.int32), 0, n_slots - 1)
    target = jnp.take_along_axis(slot_unit, target_slot[:, None], axis=1)[:, 0]
    has_target = target >= 0

    screen_x = (sx + 0.5) / cfg_x
    screen_y = (sy + 0.5) / cfg_y
    ds, dn = _screen_to_centred_lane(screen_x, screen_y)
    if frame is None:
        # A world-aligned frame is retained only for narrow decoder tests.
        # Production trainer/benchmark call sites always pass the top-lane
        # frame, matching lanerl_rl.env.decode_action.
        axis = jnp.asarray([1.0, 0.0], state.x.dtype)
        normal = jnp.asarray([0.0, 1.0], state.x.dtype)
    else:
        axis, normal = frame.axis, frame.normal
    # The shared policy acts in its own-side canonical lane frame. Blue's
    # down-lane axis is `frame.axis`; red's is its negative. Both sides retain
    # the same handedness-corrected normal, exactly `(s,n)->(L-s,n)`.
    side = jnp.where(state.team[:2] == Team.BLUE, 1.0, -1.0)
    world_dx = side * ds * axis[0] + dn * normal[0]
    world_dy = side * ds * axis[1] + dn * normal[1]
    is_attack_move = button == BUTTON_INDEX["attack_move"]
    in_minimap = (screen_x >= MINIMAP_X_MIN) & (screen_y >= MINIMAP_Y_MIN)

    # BUTTONS is the canonical tuple in lanerl_rl.constants.  Q/W/E/R and the
    # blue pill all have concrete semantic order handlers.
    kind = jnp.where(
        button == BUTTON_INDEX["move"], OrderKind.MOVE,
        jnp.where(
            is_attack_move & has_target, OrderKind.ATTACK,
            jnp.where(
                is_attack_move, OrderKind.MOVE,
                jnp.where(
                    button == BUTTON_INDEX["q"], OrderKind.CAST_Q,
                    jnp.where(
                        button == BUTTON_INDEX["w"], OrderKind.CAST_W,
                        jnp.where(
                            button == BUTTON_INDEX["e"], OrderKind.CAST_E,
                            jnp.where(
                                button == BUTTON_INDEX["r"], OrderKind.CAST_R,
                                jnp.where(button == BUTTON_INDEX["recall"],
                                          OrderKind.RECALL,
                                          OrderKind.NOOP))))))))
    # Attack-move with a visible target is a semantic unit order and does not
    # consume the sampled ground point. Plain Move (including targetless
    # attack-move) would click the minimap and is suppressed.
    kind = jnp.where((kind == OrderKind.MOVE) & in_minimap,
                     OrderKind.NOOP, kind)
    x = state.x[:2] + world_dx
    y = state.y[:2] + world_dy
    if snap_moves:
        # PATH-010: never emit a Move goal the champion cannot stand on (see
        # the block comment above `MoveSnapTable`). Casts keep the raw point.
        sx_w, sy_w = snap_move_point(x, y)
        is_move = kind == OrderKind.MOVE
        x = jnp.where(is_move, sx_w, x)
        y = jnp.where(is_move, sy_w, y)
    return Orders(
        kind=kind.astype(jnp.int8),
        x=x,
        y=y,
        target=target.astype(jnp.int8),
    )
