"""The lane frame: ``s`` progress from my outer turret towards the enemy's, ``n`` the perpendicular offset
with my Nexus at ``n < 0``. Both teams' top-lane frames share the world normal, so the map between them
is the reflection ``(s, n) -> (L - s, n)`` (valid for chirally symmetric kits such as Garen's)."""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class LaneFrame(NamedTuple):
    origin: jax.Array   # (2,) own outer turret
    axis: jax.Array     # (2,) unit vector towards the enemy turret
    normal: jax.Array   # (2,) unit, own Nexus on the negative side
    length: jax.Array   # turret-to-turret distance


def make_lane_frame(own_turret, enemy_turret, own_nexus) -> LaneFrame:
    o = jnp.asarray(own_turret, jnp.float32)
    d = jnp.asarray(enemy_turret, jnp.float32) - o
    length = jnp.linalg.norm(d)
    axis = d / jnp.where(length > 0, length, 1.0)
    normal = jnp.stack([-axis[1], axis[0]])
    normal = jnp.where(jnp.dot(jnp.asarray(own_nexus, jnp.float32) - o, normal) > 0, -normal, normal)
    return LaneFrame(origin=o, axis=axis, normal=normal, length=length)


def to_lane(frame: LaneFrame, x, y):
    """World position -> ``(s, n)``."""
    return delta_to_lane(frame, x - frame.origin[0], y - frame.origin[1])


def delta_to_lane(frame: LaneFrame, dx, dy):
    """World offset -> ``(ds, dn)`` (no origin shift)."""
    return dx * frame.axis[0] + dy * frame.axis[1], dx * frame.normal[0] + dy * frame.normal[1]
