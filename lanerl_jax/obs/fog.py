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

Production configs supply the map brush/wall grid and require an unobstructed
source-style visibility ray as well as viewer range. ``vision=None`` retains
the radius-only model for synthetic fixtures. The bounded float32 ray caster
conservatively handles near-corner ties; it is not bit-exact server parity.
Server collectors use authoritative ``vb``/``vr`` instead of either model.

Turrets are not fogged, and this is not the radius approximation talking
------------------------------------------------------------------------
``GameObject.IsVisibleByTeam`` is ``!IsAffectedByFoW || _visibleByTeam[team]``
(``GameServerLib/GameObjects/GameObject.cs:328-330``): an object exempt from FoW
is visible unconditionally, full stop, no radius check ever runs. ``BaseTurret``
sets exactly that exemption (``GameServerLib/GameObjects/AttackableUnits/AI/
BaseTurret.cs:36``, ``IsAffectedByFoW => false``; buildings do the same at
``AttackableUnits/Buildings/ObjBuilding.cs:9``), while ``AttackableUnit`` --
the common base of ``Champion`` and ``Minion``, neither of which overrides it
-- defaults it to ``true`` (``AttackableUnits/AttackableUnit.cs:117``). So a
turret out of every ally's sight radius is still visible on the server; the
800 above is the radius a turret *projects* to its own team as a vision
provider, a completely separate fact from whether the turret itself can be
hidden. Folding the two together would have hidden enemy turrets whenever no
ally stood within 800 units of them -- on this map's numbers that is most of
the game, since acquisition/attack ranges are all smaller than the matching
vision radius (turret attack range 750 <= 800, see
``lanerl_jax/sim/profiles.py``) but nothing keeps a champion or wave *near*
the enemy turret most of a lane. ``turret_acquire`` and every observation slot
would have gone dark on a live, fully-visible objective. Reached on this
config: ``garen1v1.json`` pins ``map: 1``, and ``init_lane`` places all 24 map-1
lane turrets (``sim/init.py``'s ``ALL_TURRETS``) as ``Kind.TURRET`` regardless
of ``LANERL_TOPONLY``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from ..sim.state import Kind, Team

__all__ = ["VISION_RADIUS", "vision_radius_of", "visible_to", "visible_to_enemy"]

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
               unit_team: jax.Array, alive: jax.Array, vision=None) -> jax.Array:
    """``(N,)`` bool: which units the given team can currently see.

    A unit always sees itself and its allies -- the server's own units are
    unconditionally visible to their team -- so the ally mask is ORed in rather
    than relying on a radius test that would blink a distant ally out.

    A turret is a THIRD unconditional case, ORed in separately from the ally
    rule above because it applies to enemy turrets too: ``BaseTurret.
    IsAffectedByFoW => false`` (see the module docstring) makes
    ``IsVisibleByTeam`` return ``true`` outright, with no radius test at all.
    Kind.NONE (an empty slot) and Kind.CHAMPION/LANE_MINION fall through to
    the radius rule, matching ``AttackableUnit.IsAffectedByFoW => true``.
    """
    r = vision_radius_of(kind)
    viewer = alive & (unit_team == team)
    d2 = (x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2
    within = d2 < (r[:, None] ** 2) if vision is not None else d2 <= (r[:, None] ** 2)
    if vision is not None:
        from .vision import clear_ray
        candidate = (within & viewer[:, None] & alive[None, :]
                     & (unit_team[None, :] != team) & (kind[None, :] != Kind.TURRET))
        within &= clear_ray(vision, x[:, None], y[:, None], x[None, :], y[None, :],
                            enabled=candidate)
    seen = jnp.any(viewer[:, None] & within, axis=0)
    never_fogged = kind == Kind.TURRET
    return alive & (seen | (unit_team == team) | never_fogged)


def visible_to_enemy(x: jax.Array, y: jax.Array, kind: jax.Array,
                     unit_team: jax.Array, alive: jax.Array, vision=None) -> jax.Array:
    """``(N,)`` bool: is each unit visible to the team that is NOT its own.

    This is the one number target acquisition actually needs. The server's
    gate is always framed from a *seeker's* point of view -- ``ObjAIBase.
    UpdateTarget`` drops ``TargetUnit`` when ``!TargetUnit.IsVisibleByTeam
    (Team)``, i.e. when the target has become invisible to the SEEKER's own
    team (``GameServerLib/GameObjects/AttackableUnits/AI/ObjAIBase.cs:1183``)
    -- and in a two-team lane a unit's only possible seekers are the one team
    it is not on. So "visible to team(u)'s opponent" is a property of the
    target alone, not of the (seeker, target) pair, and a single ``(N,)``
    array carries it for every seeker at once: a blue minion, a blue turret
    and a blue champion all read the same entry for a given red candidate.

    That is also why this is not stored as ``visible_to_blue`` AND
    ``visible_to_red`` (2*(N,)) or a full ``(N, N)`` who-sees-whom matrix
    (66x an ``(N,)`` field, and per ``LaneState``'s own docstring the state
    is a fixed-shape pytree replicated across thousands of parallel envs --
    every extra ``(N, N)`` leaf is 66x the memory of this one, for
    information a two-team lane cannot use). Only the two enemy calls to
    :func:`visible_to` are made; each unit then reads off whichever one
    belongs to its own opponent.
    """
    seen_by_blue = visible_to(Team.BLUE, x, y, kind, unit_team, alive, vision)
    seen_by_red = visible_to(Team.RED, x, y, kind, unit_team, alive, vision)
    # unit_team == BLUE -> its opponent is RED -> read seen_by_red, and vice
    # versa. A NEUTRAL-team slot (an empty/unused unit slot) takes the RED
    # branch by construction of jnp.where's default; harmless, since such a
    # slot is never `alive` and every consumer already gates on that.
    return jnp.where(unit_team == Team.BLUE, seen_by_red, seen_by_blue)
