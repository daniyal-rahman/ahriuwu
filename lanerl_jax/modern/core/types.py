"""Shared contract between the 26.19 modern world tick and its subsystems.

``world.tick`` owns the world arrays and the tick order. Subsystems (lane AI
for minions and turrets, champion kits, summoner spells) are pure functions
over the types below and return events/packets; they never write world arrays
themselves. Shapes: (N,) world units, (C,) champions (holder ``c`` is world
unit ``champion_unit[c]``, champions occupy units 0..C-1).

Damage always goes through ``core.damage`` packets, so items, runes and
champions see every hit the same way (README hook crosswalk).
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from . import damage as D

# Unit kinds of the modern world (``state.Kind`` keeps the legacy 0..3 meanings).
KIND_NONE, KIND_CHAMPION, KIND_MINION, KIND_TURRET, KIND_INHIBITOR, KIND_NEXUS, KIND_MONSTER, KIND_WARD = range(8)
STRUCTURE_KINDS = (KIND_TURRET, KIND_INHIBITOR, KIND_NEXUS)
BLUE, RED = 0, 1
NEUTRAL = 2                 # team of jungle monsters (never equal to a champion's team)

# World unit layout (slot ranges, in this order; ``world.config.unit_ranges``):
#   champions [0, C) | lane minions, all three lanes [C, C+M) | monsters [.., +MAX_MONSTERS)
#   | wards [.., +2*MAX_WARDS_PER_TEAM) | structures (22 turrets, 6 inhibitors, 2 Nexuses) last.
# Structures stay last so "fogged" units are exactly the slots before them.
MAX_MINIONS_PER_LANE = 40
MAX_MONSTERS = 48
MAX_WARDS_PER_TEAM = 8


def is_structure(kind: Any) -> Any:
    """Turret, inhibitor or Nexus."""
    kind = jnp.asarray(kind)
    return (kind == KIND_TURRET) | (kind == KIND_INHIBITOR) | (kind == KIND_NEXUS)


def damage_class(kind: Any) -> Any:
    """World kind -> ``core.damage.CLASS_*`` (DMG.45 unit-class ratios)."""
    kind = jnp.asarray(kind)
    return jnp.where(kind == KIND_CHAMPION, D.CLASS_CHAMPION,
                     jnp.where((kind == KIND_TURRET) | (kind == KIND_INHIBITOR) | (kind == KIND_NEXUS),
                               D.CLASS_STRUCTURE,
                               jnp.where(kind == KIND_MONSTER, D.CLASS_MONSTER, D.CLASS_MINION))).astype(jnp.int32)


class WorldUnits(NamedTuple):
    """Read-only view of every world unit for one tick, shape (N,).

    ``attack_speed`` is attacks per second after STAT.60 caps; ``attack_range``
    is the champion/minion/turret stat range (edge-to-edge rule: an attack
    reaches when ``dist <= attack_range + radius_attacker + radius_target``,
    DAMAGE_AND_STATS §8.3). ``sub`` is the minion type (0 melee, 1 caster,
    2 siege, 3 super; ``lane.minions.MinionType``), the turret tier
    (0 outer, 1 inner, 2 inhibitor, 3 nexus), the monster type
    (``jungle.camps.Monster``) or the ward type (``wards.WardType``);
    0 for others.
    """
    kind: Any
    sub: Any
    team: Any
    alive: Any
    targetable: Any
    x: Any
    y: Any
    radius: Any
    hp: Any
    max_hp: Any
    armor: Any
    magic_resist: Any
    attack_damage: Any
    attack_range: Any
    attack_speed: Any
    move_speed: Any
    spawn_seq: Any          # int32 identity, changes when a slot is reused
    spawn_time: Any         # seconds the unit spawned (minions: upgrade count source)


class AttackState(NamedTuple):
    """Generic basic-attack machine shared by champions, minions and turrets (N,).

    A unit with ``target >= 0`` in range and ``cooldown_left <= 0`` starts a
    windup (``windup_left = windup``); when the windup ends the attack
    *launches* (on-attack) and ``cooldown_left`` is set to the attack period
    from the windup start; melee attacks hit on launch, ranged ones spawn a
    missile that hits on arrival. Any pre-launch cancel resets the timer to 0
    (DAMAGE_AND_STATS §8.2, U-08 default).
    """
    target: Any             # int32, -1 none
    target_seq: Any         # spawn_seq of the target when it was chosen
    windup_left: Any        # seconds, > 0 while winding up
    cooldown_left: Any      # seconds until the next windup may start


def init_attack_state(n: int) -> AttackState:
    z = jnp.zeros((n,), jnp.float32)
    return AttackState(jnp.full((n,), -1, jnp.int32), jnp.zeros((n,), jnp.int32), z, z)


class AttackLaunch(NamedTuple):
    """Basic attacks launched this tick (N,), produced by ``world.tick``."""
    launched: Any           # bool
    target: Any             # int32
    ranged: Any             # bool: becomes a missile
    is_crit: Any            # bool (champions; Bernoulli roll at launch, X-8)
    cast_id: Any            # int32 attack instance id


class CastOrder(NamedTuple):
    """One ability / summoner request per champion this tick (C,)."""
    slot: Any               # int32: -1 none, 0..3 Q/W/E/R
    target: Any             # int32 unit, -1 none
    x: Any                  # target point
    y: Any


class CCOut(NamedTuple):
    """Crowd control applied this tick by champion c to unit n, (C, N).

    Durations are seconds *before* tenacity (``world.tick`` applies
    ``core.stat_pipeline.cc_duration`` with the target's tenacity, except for
    the types tenacity does not affect).
    """
    stun: Any
    root: Any
    silence: Any
    knockup: Any
    slow: Any               # strength (0..1); duration in ``slow_duration``
    slow_duration: Any
    cast_id: Any            # (C, N) int32 instance that applied it (Cheap Shot / Electrocute pairing)


def no_cc(c: int, n: int) -> CCOut:
    z = jnp.zeros((c, n), jnp.float32)
    return CCOut(z, z, z, z, z, z, jnp.zeros((c, n), jnp.int32))


def merge_cc(a: CCOut, b: CCOut) -> CCOut:
    stronger = b.slow > a.slow
    return CCOut(jnp.maximum(a.stun, b.stun), jnp.maximum(a.root, b.root), jnp.maximum(a.silence, b.silence),
                 jnp.maximum(a.knockup, b.knockup), jnp.maximum(a.slow, b.slow),
                 jnp.where(stronger, b.slow_duration, jnp.maximum(a.slow_duration, b.slow_duration)),
                 jnp.where(b.cast_id != 0, b.cast_id, a.cast_id))


class Dash(NamedTuple):
    """Movement a kit or summoner imposes on its champion (C,)."""
    active: Any             # bool: start a dash/blink this tick
    to_x: Any
    to_y: Any
    speed: Any              # units per second; inf = instant blink (Flash)
    target: Any             # int32 unit to dash to (-1 = point)
    blink: Any              # bool: counts as a blink for Sudden Impact / Flash rules


def no_dash(c: int) -> Dash:
    z = jnp.zeros((c,), jnp.float32)
    return Dash(jnp.zeros((c,), bool), z, z, z, jnp.full((c,), -1, jnp.int32), jnp.zeros((c,), bool))
