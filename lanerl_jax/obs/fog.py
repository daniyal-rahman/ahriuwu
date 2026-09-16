"""Fog of war: which units a team can see.

The rule the server applies is ``GameObject.IsVisibleByTeam``, exposed on the
control wire as ``vb``/``vr``.  On device it is reconstructed from vision radii:
a unit is visible to a team when **any living unit of that team** is within that
*viewer's* radius of it.  The radius belongs to the viewer, not the target
(``lanerl_rl/constants.py``, each cited to its server file)::

    champion 1200 (Champion.cs:52)   minion 1100 (Minion.cs:57)
    turret    800                    building 1350

Why this matters more than it looks
-----------------------------------
``obs.py``'s third design rule: *a remembered-but-currently-fogged entity keeps
its slot with ``valid = 0`` ... it is never written into a slot with
``ds = dn = 0, valid = 1``, which would tell the policy "the enemy is standing
on top of me" -- the single worst hallucination available.*  So fog is not a
nicety; getting it wrong puts a phantom enemy in the agent's lap.

**Booked approximation:** this is a radius model, not the server's own
brush/terrain-aware vision. `lanerl_rl` carries the same approximation as
`ApproxFogModel` and gates production on the server's `vb`/`vr` instead, with a
differential test (`test_fog_comes_from_the_server`) proving which one ran.
The JAX sim has no server to ask, so the approximation is the only option here
and its cost is unmeasured -- brush in particular is invisible to it.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from ..sim.state import Kind

__all__ = ["VISION_RADIUS", "vision_radius_of", "visible_to"]

#: viewer kind -> sight radius, from ``lanerl_rl.constants``.
VISION_RADIUS = {
    Kind.CHAMPION: 1200.0,
    Kind.LANE_MINION: 1100.0,
    Kind.TURRET: 800.0,
    Kind.NONE: 0.0,
}


def vision_radius_of(kind: jax.Array) -> jax.Array:
    table = jnp.asarray(
        [VISION_RADIUS[Kind.NONE], VISION_RADIUS[Kind.CHAMPION],
         VISION_RADIUS[Kind.LANE_MINION], VISION_RADIUS[Kind.TURRET]],
        jnp.float32)
    return table[jnp.clip(kind, 0, 3)]


def visible_to(team: int, x: jax.Array, y: jax.Array, kind: jax.Array,
               unit_team: jax.Array, alive: jax.Array) -> jax.Array:
    """``(N,)`` bool: which units the given team can currently see.

    A unit always sees itself and its allies -- the server's own units are
    unconditionally visible to their team -- so the ally mask is ORed in rather
    than relying on a radius test that would blink a distant ally out.
    """
    r = vision_radius_of(kind)
    viewer = alive & (unit_team == team)
    d2 = (x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2
    seen = jnp.any(viewer[:, None] & (d2 <= (r[:, None] ** 2)), axis=0)
    return alive & (seen | (unit_team == team))
