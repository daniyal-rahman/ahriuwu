"""Screen-click geometry of the locked League camera (1280x720, 40 deg vertical FOV, 56 deg tilt, centred on
the champion): the 96x54 click grid, screen -> lane-frame ground offset, and which offsets are clickable."""
from __future__ import annotations

import math

import jax.numpy as jnp

N_SCREEN_X, N_SCREEN_Y = 96, 54
FLOOR_Y = 52.0                        # ground height of the recorded camera model
FOV_V_DEG, TILT_DEG, CAM_Y = 40.0, 56.0, 1912.0
RESOLUTION = (1280, 720)
MINIMAP_X_MIN, MINIMAP_Y_MIN = 275.0 / 352.0, 240.0 / 352.0     # minimap corner, normalised screen units


def screen_to_lane(sx, sy):
    """Normalised screen point -> ground offset ``(ds, dn)`` from the champion (camera centred on it)."""
    dtype = jnp.result_type(sx, sy, jnp.float32)
    tilt = math.radians(TILT_DEG)
    cos_t, sin_t = jnp.asarray(math.cos(tilt), dtype), jnp.asarray(math.sin(tilt), dtype)
    dy = jnp.asarray(FLOOR_Y - CAM_Y, dtype)
    tan_v = jnp.asarray(math.tan(math.radians(FOV_V_DEG) / 2.0), dtype)
    tan_h = tan_v * (RESOLUTION[0] / RESOLUTION[1])
    camera_z = dy * cos_t / sin_t                  # the screen centre looks at ground offset 0
    kv = (jnp.asarray(0.5, dtype) - sy) * 2.0 * tan_v
    dz = dy * (cos_t + kv * sin_t) / (kv * cos_t - sin_t)
    vz = -dy * sin_t + dz * cos_t
    return (sx - jnp.asarray(0.5, dtype)) * 2.0 * tan_h * vz, camera_z + dz


def target_on_screen(ds, dn):
    """Whether a ground offset is inside the camera view and not under the minimap."""
    tilt = math.radians(TILT_DEG)
    ct, st = math.cos(tilt), math.sin(tilt)
    dy = FLOOR_Y - CAM_Y
    dz = dn - dy * ct / st
    vy, vz = dy * ct + dz * st, -dy * st + dz * ct
    tv = math.tan(math.radians(FOV_V_DEG) / 2)
    th = tv * RESOLUTION[0] / RESOLUTION[1]
    inside = (vz > 0) & (abs(ds) <= vz * th) & (abs(vy) <= vz * tv)
    minimap = (ds >= (2 * MINIMAP_X_MIN - 1) * vz * th) & (-vy >= (2 * MINIMAP_Y_MIN - 1) * vz * tv)
    return inside & ~minimap
