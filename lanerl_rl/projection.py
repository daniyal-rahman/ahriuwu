"""World <-> screen projection for the locked, champion-centred camera.

This is the same tilted-perspective model the replay pipeline records in
``labels.json``, reproduced here so the RL action space can be expressed in
SCREEN coordinates rather than in world offsets.

Why that matters: a screen position is what a human actually commands.  One
2-D click covers move, attack-move, every ground-targeted cast and every
skillshot, for every champion, with no per-ability special-casing -- and it is
the literal thing a deployed vision+mouse shim would emit.  The current action
space instead carries a direction (9x9 lane-local bins) with the distance
HARDCODED at ``move_distance = 500``: the agent can pick a heading but never
how far, so "step 150 units back" is not expressible at all.

The model
---------
A camera at ``(cx, cam_y, cz)`` looking down at ``tilt_deg`` from horizontal.
Only the ground plane matters, so every world point is taken at ``FLOOR_Y``::

    dy = FLOOR_Y - cam_y                      (constant, negative)
    dx = wx - cx
    dz = wz - cz
    vy =  dy*cos_t + dz*sin_t                 (camera-space up)
    vz = -dy*sin_t + dz*cos_t                 (camera-space depth)
    sx = 0.5 + (dx/vz)/tan_h * 0.5
    sy = 0.5 - (vy/vz)/tan_v * 0.5            (screen y grows DOWNWARD)

``fov_h`` is derived from ``fov_v`` and the aspect ratio, not stored:
``2*atan(tan(fov_v/2) * w/h)``.  For 40 deg at 1280x720 that is 65.8 deg,
which matches the value ``labels.json`` records but never reads.

Output is deliberately NOT clamped to [0, 1].  A command aimed off the
viewport -- an edge walk, a minimap click -- keeps a real signed coordinate
instead of being silently discarded, which in the recorded data is about 24%
of frames.

Why the inverse is exact even though ``cam_y`` is unobservable
--------------------------------------------------------------
Both screen coordinates depend only on the ratios ``dx/vz`` and ``vy/vz``, and
``vy``/``vz`` are linear in ``(dy, dz)``.  Scaling ``(dy, dx, dz)`` by any
common factor leaves both ratios unchanged, so ANY assumed ``cam_y`` yields a
camera that reproduces the projection exactly for every point on the floor.
The recovered camera is one of a one-parameter family, and they are all
equivalent on the ground plane.

The one number that is not recoverable
--------------------------------------
``FLOOR_Y = 52.0`` is not written to ``labels.json`` and cannot be derived
from it.  If the recorder ever used a different ground height, nothing in the
data would reveal the mismatch -- it would show up only as a systematic
world-space error that looks like bad aim.  It is asserted in the tests so a
change has to be deliberate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

#: Ground-plane height in world units.  NOT present in labels.json.
FLOOR_Y = 52.0

#: Depth floor.  A point at or behind the camera plane has no sane projection;
#: clamping ``vz`` keeps the DIRECTION rather than flipping the sign, which is
#: what makes an off-screen command still point the right way.
VZ_MIN = 10.0

DEFAULT_FOV_V_DEG = 40.0
DEFAULT_TILT_DEG = 56.0
DEFAULT_CAM_Y = 1912.0
DEFAULT_RESOLUTION = (1280, 720)


@dataclass(frozen=True)
class Camera:
    """A locked camera over the ground plane."""

    cx: float
    cz: float
    fov_v_deg: float = DEFAULT_FOV_V_DEG
    tilt_deg: float = DEFAULT_TILT_DEG
    cam_y: float = DEFAULT_CAM_Y
    width: int = DEFAULT_RESOLUTION[0]
    height: int = DEFAULT_RESOLUTION[1]

    @property
    def tan_v(self) -> float:
        return math.tan(math.radians(self.fov_v_deg) / 2.0)

    @property
    def fov_h_deg(self) -> float:
        return math.degrees(2.0 * math.atan(self.tan_v * self.width / self.height))

    @property
    def tan_h(self) -> float:
        return math.tan(math.radians(self.fov_h_deg) / 2.0)

    @property
    def _trig(self) -> Tuple[float, float, float]:
        t = math.radians(self.tilt_deg)
        return math.cos(t), math.sin(t), FLOOR_Y - self.cam_y


def world_to_screen(cam: Camera, wx: float, wz: float) -> Tuple[float, float]:
    """World ground point -> NORMALISED screen (sx, sy), unclamped."""
    cos_t, sin_t, dy = cam._trig
    dx = wx - cam.cx
    dz = wz - cam.cz
    vy = dy * cos_t + dz * sin_t
    vz = -dy * sin_t + dz * cos_t
    if vz < VZ_MIN:
        vz = VZ_MIN
    sx = 0.5 + (dx / vz) / cam.tan_h * 0.5
    sy = 0.5 - (vy / vz) / cam.tan_v * 0.5
    return sx, sy


def screen_to_world(cam: Camera, sx: float, sy: float) -> Tuple[float, float]:
    """NORMALISED screen -> the world ground point under it.

    This is the one the action space needs: the policy names a point on
    screen, and the server is told the world position it corresponds to.
    """
    cos_t, sin_t, dy = cam._trig
    kv = (0.5 - sy) * 2.0 * cam.tan_v
    den = kv * cos_t - sin_t
    if abs(den) < 1e-12:
        # The ray is parallel to the ground plane: the horizon. No finite
        # intersection exists, so refuse rather than return a huge number.
        raise ValueError(
            f"screen y={sy} is on the horizon for tilt={cam.tilt_deg} deg; "
            f"no ground point projects there"
        )
    dz = dy * (cos_t + kv * sin_t) / den
    vz = -dy * sin_t + dz * cos_t
    dx = (sx - 0.5) * 2.0 * cam.tan_h * vz
    return cam.cx + dx, cam.cz + dz


def centred_on(px: float, pz: float, **kw) -> Camera:
    """The camera that puts the champion at ``(px, pz)`` dead centre.

    Solved rather than taken from ``labels.json``'s ``cam_z_offset``: at screen
    centre ``sx=0.5`` gives ``dx=0`` and ``sy=0.5`` gives ``kv=0``, so
    ``dz = -dy/tan(tilt)``.  With the defaults that is 1254.6 units, against
    the 1292.0 the recorder stores -- close, but the stored value is never read
    by the projection either, so deriving it keeps the camera exactly
    consistent with this model instead of nearly consistent with another one.
    """
    probe = Camera(cx=0.0, cz=0.0, **kw)
    cos_t, sin_t, dy = probe._trig
    dz_centre = -dy * cos_t / sin_t
    return Camera(cx=px, cz=pz - dz_centre, **kw)


def screen_to_world_centred(px: float, pz: float, sx: float, sy: float,
                            **kw) -> Tuple[float, float]:
    """Champion-centred convenience: where does this click land in the world?"""
    return screen_to_world(centred_on(px, pz, **kw), sx, sy)


def units_per_pixel_at_centre(cam: Camera) -> Tuple[float, float]:
    """World units per pixel at screen centre, (horizontal, vertical).

    Used to choose an action-grid resolution: a bin must be smaller than the
    thing it needs to click.  A minion's selectable radius is 40 units, so it
    is ~80 units across.
    """
    w, h = cam.width, cam.height
    x0, z0 = screen_to_world(cam, 0.5, 0.5)
    x1, _ = screen_to_world(cam, 0.5 + 1.0 / w, 0.5)
    _, z1 = screen_to_world(cam, 0.5, 0.5 + 1.0 / h)
    return abs(x1 - x0), abs(z1 - z0)


# Same HUD exclusion as the movement decoder. Kept here so entity clicks and
# ground clicks share the same definition of the playable viewport.
MINIMAP_X_MIN = 275.0 / 352.0
MINIMAP_Y_MIN = 240.0 / 352.0


def target_on_screen(ds, dn):
    """Whether a canonical ground offset is clickable in the centred camera.

    Works with scalars, NumPy or JAX arrays. Cross-multiplied projection avoids
    divisions at/behind the camera plane. This is a screen bound, not a radius
    or a team-vision check; callers must additionally require a visible unit.
    """
    tilt = math.radians(DEFAULT_TILT_DEG)
    ct, st = math.cos(tilt), math.sin(tilt)
    dy = FLOOR_Y - DEFAULT_CAM_Y
    dz = dn - dy * ct / st
    vy, vz = dy * ct + dz * st, -dy * st + dz * ct
    tv = math.tan(math.radians(DEFAULT_FOV_V_DEG) / 2)
    th = tv * DEFAULT_RESOLUTION[0] / DEFAULT_RESOLUTION[1]
    inside = (vz > 0) & (abs(ds) <= vz * th) & (abs(vy) <= vz * tv)
    minimap = ((ds >= (2 * MINIMAP_X_MIN - 1) * vz * th)
               & (-vy >= (2 * MINIMAP_Y_MIN - 1) * vz * tv))
    return inside & ~minimap
