"""Policy-action decoding shared by rollout and benchmark paths.

The target head names an *observation slot*, not a simulator unit index.  That
distinction is invisible while the enemy champion happens to be in slot 0, but
it is decisive for minions and turrets: slot 13 is an enemy-minion slot, not
unit 13.  Keep the conversion here, at the observation/action boundary, so the
simulator continues to receive only semantic unit-index orders.
"""
from __future__ import annotations

import math

import jax.numpy as jnp

from lanerl_rl.constants import BUTTON_INDEX, N_SCREEN_X, N_SCREEN_Y
from lanerl_rl.projection import (
    DEFAULT_CAM_Y,
    DEFAULT_FOV_V_DEG,
    DEFAULT_RESOLUTION,
    DEFAULT_TILT_DEG,
    FLOOR_Y,
)

from ..sim.orders import OrderKind, Orders
from ..sim.state import Team

__all__ = ["orders_from"]


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
                cfg_x=N_SCREEN_X, cfg_y=N_SCREEN_Y):
    """Decode factored policy actions into semantic champion orders.

    ``slot_unit`` has shape ``(n_champions, n_slots)`` and comes from the same
    observation that produced ``action``.  An invalid slot is represented by
    ``-1``.  Attack-move follows the wire's behaviour: attack a selected
    visible unit, otherwise move to the selected screen point.
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
    return Orders(
        kind=kind.astype(jnp.int8),
        x=state.x[:2] + world_dx,
        y=state.y[:2] + world_dy,
        target=target.astype(jnp.int8),
    )
