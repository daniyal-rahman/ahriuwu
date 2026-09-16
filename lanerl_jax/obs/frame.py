"""The lane frame: signed progress along the lane, perpendicular offset.

Why a reflection and not a rotation
-----------------------------------
Both champions in a same-lane 1v1 occupy the **same corridor**; they only enter
it from opposite ends.  ``lanerl_rl/frame.py`` records what happens if you
canonicalise with a 180-degree world rotation instead: ``rot180(574, 10220)``
lands on ``(13409, 4227)``, red's **bot** outer turret -- a different corridor
with an oppositely signed normal, so the two agents were not sampling one
distribution at all.

The frame here is built from two anchors, the agent's own outer turret and the
enemy's.  ``s`` is progress from mine towards theirs, ``n`` is the perpendicular
offset, and the normal's sign is pinned by a handedness rule -- the agent's own
nexus must lie at ``n < 0``.  Because the top lane hugs the map's top-left edge,
*both* nexuses sit on the same side of it, so that rule makes both agents adopt
the **same world normal** and ``n > 0`` means "towards the outer wall" for each.

The map between the two frames is then ``(s, n) -> (L - s, n)``: a reflection,
determinant -1.

**Chirality assumption.** A reflection swaps left and right, so this is a valid
canonicalisation only for a chirally symmetric kit. Garen qualifies exactly: Q
is a self-buff, W a self-shield, E a symmetric spin, R point-target; no
skillshots, dashes, cones or walls. For any other champion a reflected policy
would aim into the mirror image of where it means to. Restated here because the
assumption is invisible at the call site and expensive to rediscover.
"""
from __future__ import annotations

from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp

__all__ = ["LaneFrame", "make_lane_frame", "to_lane", "delta_to_lane"]


class LaneFrame(NamedTuple):
    origin: jax.Array   # (2,) own outer turret
    axis: jax.Array     # (2,) unit vector towards the enemy turret
    normal: jax.Array   # (2,) unit, handedness-corrected
    length: jax.Array   # scalar, turret-to-turret distance


def make_lane_frame(own_turret, enemy_turret, own_nexus) -> LaneFrame:
    """Build the frame, with the normal flipped so ``own_nexus`` is at ``n < 0``."""
    o = jnp.asarray(own_turret, jnp.float32)
    e = jnp.asarray(enemy_turret, jnp.float32)
    d = e - o
    length = jnp.linalg.norm(d)
    axis = d / jnp.where(length > 0, length, 1.0)
    # plain left-hand normal, then the handedness correction
    normal = jnp.stack([-axis[1], axis[0]])
    nx = jnp.asarray(own_nexus, jnp.float32) - o
    normal = jnp.where(jnp.dot(nx, normal) > 0, -normal, normal)
    return LaneFrame(origin=o, axis=axis, normal=normal, length=length)


def to_lane(frame: LaneFrame, x: jax.Array, y: jax.Array
            ) -> Tuple[jax.Array, jax.Array]:
    """World -> ``(s, n)``. Absolute, i.e. relative to the frame's origin."""
    dx = x - frame.origin[0]
    dy = y - frame.origin[1]
    return (dx * frame.axis[0] + dy * frame.axis[1],
            dx * frame.normal[0] + dy * frame.normal[1])


def delta_to_lane(frame: LaneFrame, dx: jax.Array, dy: jax.Array
                  ) -> Tuple[jax.Array, jax.Array]:
    """A world **offset** -> ``(ds, dn)``. No origin shift, so this is what
    entity features use: ``E_DS``/``E_DN`` are *(target - me)* in lane axes."""
    return (dx * frame.axis[0] + dy * frame.axis[1],
            dx * frame.normal[0] + dy * frame.normal[1])
