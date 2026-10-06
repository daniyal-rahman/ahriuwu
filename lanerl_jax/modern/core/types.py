"""Contract between the 26.19 world tick and its subsystems.

``world.tick`` owns the world arrays; subsystems are pure functions over these types and return packets, CC,
dashes and ``UnitWrite`` rows. Shapes: (N,) world units, (C,) champions (champion c is unit c). All damage goes
through ``core.damage`` packets.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from . import damage as D

KIND_NONE, KIND_CHAMPION, KIND_MINION, KIND_TURRET, KIND_INHIBITOR, KIND_NEXUS, KIND_MONSTER, KIND_WARD = range(8)
STRUCTURE_KINDS = (KIND_TURRET, KIND_INHIBITOR, KIND_NEXUS)
BLUE, RED = 0, 1
NEUTRAL = 2                 # jungle monsters' team

# Slot capacities of the world unit blocks (``world.config.Layout``).
MINION_SLOTS_PER_LANE = 40
JUNGLE_SLOTS = 40           # jungle.camps uses 38
EPIC_SLOTS = 8
MAX_WARDS_PER_TEAM = 8


def is_structure(kind: Any) -> Any:
    kind = jnp.asarray(kind)
    return (kind == KIND_TURRET) | (kind == KIND_INHIBITOR) | (kind == KIND_NEXUS)


def damage_class(kind: Any) -> Any:
    """World kind -> ``core.damage.CLASS_*`` (DMG.45)."""
    kind = jnp.asarray(kind)
    return jnp.where(kind == KIND_CHAMPION, D.CLASS_CHAMPION,
                     jnp.where(is_structure(kind), D.CLASS_STRUCTURE,
                               jnp.where(kind == KIND_MONSTER, D.CLASS_MONSTER, D.CLASS_MINION))).astype(jnp.int32)


class WorldUnits(NamedTuple):
    """Read-only view of every world unit for one tick, (N,).

    ``attack_speed`` is attacks/s after caps. An attack reaches when ``dist <= attack_range + r_attacker +
    r_target`` (DAMAGE_AND_STATS §8.3). ``sub``: minion type, turret tier, monster type or ward type; else 0.
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
    spawn_time: Any


class AttackState(NamedTuple):
    """Basic-attack machine shared by all units (N,); see ``mechanics.attack_step`` (DAMAGE_AND_STATS §8.2)."""
    target: Any             # int32, -1 none
    target_seq: Any         # spawn_seq of the target when chosen
    windup_left: Any        # seconds, > 0 while winding up
    cooldown_left: Any      # seconds until the next windup may start


def init_attack_state(n: int) -> AttackState:
    z = jnp.zeros((n,), jnp.float32)
    return AttackState(jnp.full((n,), -1, jnp.int32), jnp.zeros((n,), jnp.int32), z, z)


class UnitWrite(NamedTuple):
    """Rows a subsystem asks ``world.units.write_units`` to (re)write where ``mask``, (N,).

    Written rows become alive and targetable; ``new`` rows also get a fresh ``spawn_seq``/``spawn_time`` and a
    reset slot (attack state, CC). ``bounty_*``: lane-minion rewards fixed at spawn (None: unchanged)."""
    mask: Any
    new: Any
    kind: Any
    sub: Any
    team: Any
    x: Any
    y: Any
    hp: Any
    max_hp: Any
    radius: Any
    armor: Any
    magic_resist: Any
    attack_damage: Any
    attack_range: Any
    attack_speed: Any
    move_speed: Any
    windup: Any
    missile_speed: Any
    bounty_gold: Any = None
    bounty_xp: Any = None
    bounty_level: Any = None


UNIT_COLUMNS = UnitWrite._fields[2:18]          # the columns a UnitWrite writes


class AttackLaunch(NamedTuple):
    """Basic attacks launched this tick (N,)."""
    launched: Any
    target: Any
    ranged: Any             # becomes a missile
    is_crit: Any            # champions: rolled at launch
    cast_id: Any


class CastOrder(NamedTuple):
    """One ability / summoner request per champion this tick (C,)."""
    slot: Any               # -1 none, 0..3 Q/W/E/R
    target: Any             # unit, -1 none
    x: Any
    y: Any


class CCOut(NamedTuple):
    """CC applied this tick by champion c to unit n, (C, N); durations before tenacity."""
    stun: Any
    root: Any
    silence: Any
    knockup: Any
    slow: Any               # strength 0..1
    slow_duration: Any
    cast_id: Any            # int32 instance that applied it (Cheap Shot / Electrocute pairing)


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
    active: Any
    to_x: Any
    to_y: Any
    speed: Any              # units/s; inf = blink
    target: Any             # unit to dash to, -1 = point
    blink: Any


def no_dash(c: int) -> Dash:
    z = jnp.zeros((c,), jnp.float32)
    return Dash(jnp.zeros((c,), bool), z, z, z, jnp.full((c,), -1, jnp.int32), jnp.zeros((c,), bool))
