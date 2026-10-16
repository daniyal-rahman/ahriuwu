"""The 26.19 champion kits and their registry.

``KITS`` holds one module per champion. A kit defines ``NAME``, ``ID``, ``SKILL_ORDER`` (default rank per level),
``TRAITS`` (rune legality), ``UNIT_TARGET_RANGE``, ``State``/``init`` and the hooks ``cast``, ``periodic``,
``on_attack``, ``on_hit``, ``on_damage``, ``on_takedown``, ``stats``, ``defense``, ``attack_mods``, ``debuffs`` and
optionally ``ghosted``, ``dodging``, ``immobile``, ``displacement``, ``displacement_immune``, ``hp_cap``,
``mana_override``, ``size``, ``resource``, ``attack_missile_speed`` and ``unit_target_range`` (contract: ``core``).
``MAX_RANKS`` / ``INNATE_RANKS`` override the 5/5/5/3 rank-up table (Jayce).

A world compiles only its roster (``roster``; ``WorldConfig.kits``: the kits of the champions that can appear in the
match). ``ChampionState`` has one field per roster kit, so every hook runs the state's kits for every holder and each
kit gates itself on ``KitCtx.champion_id``; shapes stay fixed per roster.
"""
from __future__ import annotations

import functools
from typing import Any, NamedTuple

import jax.numpy as jnp

from ..items.catalog import combine_stats
from ..items.effects.core import Debuffs, combine_debuffs
from . import aatrox
from . import ambessa
from . import camille
from . import darius
from . import ezreal
from . import fiora
from . import garen
from . import jax as jax_kit
from . import jayce
from . import ksante
from . import malphite
from . import nasus
from . import renekton
from . import yone
from .core import (KitAttackMods, KitCtx, KitDefense, KitOut, combine_attack_mods, merge_out,
                   neutral_attack_mods, neutral_defense)

KITS = (garen, jax_kit, nasus, renekton, darius, malphite, aatrox, fiora, yone, ambessa, camille, ksante, jayce,
        ezreal)
BY_NAME = {k.NAME: k for k in KITS}
_BY_FIELD = {k.NAME.lower(): k for k in KITS}


@functools.cache
def state_type(kits: tuple) -> type:
    """The ``ChampionState`` of a roster: one field per kit (lowercase name), registry order."""
    return NamedTuple("ChampionState", [(k.NAME.lower(), Any) for k in kits])


ChampionState = state_type(KITS)        # every kit (unit tests, tools)


def kit(name: str):
    """Kit module of a champion name (host-side; unsupported champions raise)."""
    if name not in BY_NAME:
        raise ValueError(f"unsupported modern champion {name!r}")
    return BY_NAME[name]


def roster(names=None) -> tuple:
    """Kit modules of champion ``names`` (any order, repeats allowed) in registry order; None = every kit."""
    if names is None:
        return KITS
    want = {kit(nm) for nm in names}
    return tuple(k for k in KITS if k in want)


def kits_of(state) -> tuple:
    """The roster a ``ChampionState`` was built for."""
    return tuple(_BY_FIELD[f] for f in state._fields)


def _zip(state):
    return zip(kits_of(state), state)


def init(n_champions: int, n_units: int, kits: tuple = KITS) -> ChampionState:
    return state_type(kits)(*(k.init(n_champions, n_units) for k in kits))


def _all(fn_name, state, kctx, units, *args) -> tuple[ChampionState, KitOut]:
    c, n = kctx.unit.shape[0], units.x.shape[0]
    pairs = [getattr(k, fn_name)(st, kctx, units, *args) for k, st in _zip(state)]
    return type(state)(*(p[0] for p in pairs)), merge_out([p[1] for p in pairs], c, n)


def _slot_table(champion_ids, attr, kits) -> Any:
    ids = jnp.asarray(champion_ids, jnp.int32)
    out = jnp.zeros(ids.shape + (4,), jnp.float32)
    for k in kits:
        row = jnp.asarray(getattr(k, attr, (0.0, 0.0, 0.0, 0.0)), jnp.float32)
        out = jnp.where((ids == k.ID)[:, None], row[None, :], out)
    return out


def unit_target_ranges(champion_ids, state: ChampionState | None = None) -> Any:
    """(C, 4) center-to-edge cast range of each slot's unit-targeted spell (0: not unit-targeted);
    the world walks a champion into this range before casting (``UNIT_TARGET_RANGE`` per kit, or the kit's
    ``unit_target_range`` hook given ``state``: Jayce's stance). Kits: the state's roster, else every kit."""
    out = _slot_table(champion_ids, "UNIT_TARGET_RANGE", KITS if state is None else kits_of(state))
    if state is not None:
        ids = jnp.asarray(champion_ids, jnp.int32)
        for k, st in _zip(state):
            if hasattr(k, "unit_target_range"):
                out = jnp.where((ids == k.ID)[:, None], k.unit_target_range(st, ids), out)
    return out


def rank_tables(champion_ids, kits: tuple = KITS) -> tuple[Any, Any]:
    """``(max_ranks, innate)`` (C, 4) int32: ranks per slot (default 5/5/5/3) and ranks held without a skill
    point (Jayce's Transform), over the roster ``kits``."""
    ids = jnp.asarray(champion_ids, jnp.int32)
    cap = jnp.broadcast_to(jnp.asarray([5, 5, 5, 3], jnp.int32), ids.shape + (4,))
    innate = jnp.zeros(ids.shape + (4,), jnp.int32)
    for k in kits:
        mine = (ids == k.ID)[:, None]
        cap = jnp.where(mine, jnp.asarray(getattr(k, "MAX_RANKS", (5, 5, 5, 3)), jnp.int32)[None, :], cap)
        innate = jnp.where(mine, jnp.asarray(getattr(k, "INNATE_RANKS", (0, 0, 0, 0)), jnp.int32)[None, :], innate)
    return cap, innate


def attack_missile_speed(state: ChampionState, kctx: KitCtx) -> Any:
    """(C,) champion attack missile speed from kits (Jayce's Mercury Cannon), 0 = melee."""
    out = jnp.zeros(kctx.unit.shape, jnp.float32)
    for k, st in _zip(state):
        if hasattr(k, "attack_missile_speed"):
            out = jnp.maximum(out, k.attack_missile_speed(st, kctx))
    return out


def point_target_ranges(champion_ids, kits: tuple = KITS) -> Any:
    """(C, 4) center-to-point cast range of each slot's location-targeted spell that walks into range
    (0: none); ``POINT_TARGET_RANGE`` per kit (Nasus E), over the roster ``kits``."""
    return _slot_table(champion_ids, "POINT_TARGET_RANGE", kits)


def cast(state: ChampionState, kctx: KitCtx, units, order) -> tuple[ChampionState, KitOut]:
    return _all("cast", state, kctx, units, order)


def periodic(state: ChampionState, kctx: KitCtx, units) -> tuple[ChampionState, KitOut]:
    return _all("periodic", state, kctx, units)


def on_attack(state: ChampionState, kctx: KitCtx, units, launch) -> tuple[ChampionState, KitOut]:
    return _all("on_attack", state, kctx, units, launch)


def dodging_units(state: ChampionState, kctx: KitCtx, n_units: int) -> Any:
    """(N,) bool: world units currently dodging basic attacks (Jax E)."""
    dodge = jnp.zeros(kctx.unit.shape, bool)
    for k, st in _zip(state):
        if hasattr(k, "dodging"):
            dodge = dodge | k.dodging(st, kctx)
    return jnp.zeros((n_units,), bool).at[kctx.unit].max(dodge)


def on_hit(state: ChampionState, kctx: KitCtx, units, launch) -> tuple[ChampionState, KitOut]:
    return _all("on_hit", state, kctx, units, launch, dodging_units(state, kctx, units.x.shape[0]))


def on_damage(state: ChampionState, kctx: KitCtx, units, report) -> tuple[ChampionState, KitOut]:
    return _all("on_damage", state, kctx, units, report)


def on_takedown(state: ChampionState, kctx: KitCtx, units, kills) -> ChampionState:
    return type(state)(*(k.on_takedown(st, kctx, units, kills) for k, st in _zip(state)))


def stats(state: ChampionState, kctx: KitCtx):
    parts = [k.stats(st, kctx) for k, st in _zip(state)]
    out = parts[0]
    for p in parts[1:]:
        out = combine_stats(out, p)
    return out


def defense(state: ChampionState, kctx: KitCtx) -> KitDefense:
    out = neutral_defense(kctx.unit.shape[0])
    for d in (k.defense(st, kctx) for k, st in _zip(state)):
        parry = out.parry if d.parry is None else out.parry | d.parry
        out = KitDefense(out.received_mult * d.received_mult, out.dodge_basic | d.dodge_basic,
                         out.aoe_received_mult * d.aoe_received_mult,
                         1.0 - (1.0 - out.tenacity_bonus) * (1.0 - d.tenacity_bonus), parry)
    return out


def attack_mods(state: ChampionState, kctx: KitCtx) -> KitAttackMods:
    out = neutral_attack_mods(kctx.unit.shape[0])
    for m in (k.attack_mods(st, kctx) for k, st in _zip(state)):
        out = combine_attack_mods(out, m)
    return out


def ghosted(state: ChampionState, kctx: KitCtx) -> Any:
    """(C,) bool: holder ignores unit collision (Garen E Judgment)."""
    out = jnp.zeros(kctx.unit.shape, bool)
    for k, st in _zip(state):
        if hasattr(k, "ghosted"):
            out = out | k.ghosted(st, kctx)
    return out


def immobile(state: ChampionState, kctx: KitCtx) -> Any:
    """(C,) bool: the kit holds its champion in place without a cast lock, so its other spells stay castable
    (Aatrox Q and W cast times)."""
    out = jnp.zeros(kctx.unit.shape, bool)
    for k, st in _zip(state):
        if hasattr(k, "immobile"):
            out = out | k.immobile(st, kctx)
    return out


def cc_immune(state: ChampionState, kctx: KitCtx) -> Any:
    """(C,) bool: the holder ignores CC and slows applied this tick (Fiora Riposte's debuff immunity)."""
    out = jnp.zeros(kctx.unit.shape, bool)
    for k, st in _zip(state):
        if hasattr(k, "cc_immune"):
            out = out | k.cc_immune(st, kctx)
    return out


def displacement_immune(state: ChampionState, kctx: KitCtx) -> Any:
    """(C,) bool: the holder ignores displacements and knock-ups but still takes other CC (K'Sante's Path Maker
    charge and All Out cast)."""
    out = jnp.zeros(kctx.unit.shape, bool)
    for k, st in _zip(state):
        if hasattr(k, "displacement_immune"):
            out = out | k.displacement_immune(st, kctx)
    return out


def hp_cap(state: ChampionState, kctx: KitCtx) -> Any:
    """(C,) the holder's current health cannot exceed this (K'Sante's All Out threshold); +inf = none."""
    out = jnp.full(kctx.unit.shape, jnp.inf, jnp.float32)
    for k, st in _zip(state):
        if hasattr(k, "hp_cap"):
            out = jnp.minimum(out, k.hp_cap(st, kctx))
    return out


def mana_override(state: ChampionState, kctx: KitCtx) -> tuple[Any, Any]:
    """(C,) ``(active, mana)``: the holder's mana is set to ``mana`` after this tick's costs and regeneration
    (K'Sante's All Out refill, decay and restore)."""
    active = jnp.zeros(kctx.unit.shape, bool)
    value = jnp.zeros(kctx.unit.shape, jnp.float32)
    for k, st in _zip(state):
        if hasattr(k, "mana_override"):
            a, v = k.mana_override(st, kctx)
            value = jnp.where(a, v, value)
            active = active | a
    return active, value


def on_parry(state: ChampionState, kctx: KitCtx, parried) -> ChampionState:
    """``parried`` (C,) bool: a hostile immobilizing effect (stun, root, knock-up) was negated on the holder this
    tick by ``cc_immune`` (Fiora Riposte's stun condition)."""
    return type(state)(*(k.on_parry(st, kctx, parried) if hasattr(k, "on_parry") else st for k, st in _zip(state)))


def untargetable(state: ChampionState, champion_ids, now) -> Any:
    """(C,) bool: the holder cannot be targeted (Camille's leap); read before the tick's orders are filtered."""
    out = jnp.zeros(jnp.shape(champion_ids), bool)
    for k, st in _zip(state):
        if hasattr(k, "untargetable"):
            out = out | k.untargetable(st, jnp.asarray(champion_ids, jnp.int32), now)
    return out


def leash(state: ChampionState, kctx: KitCtx, n_units: int) -> tuple[Any, Any, Any, Any]:
    """(N,) ``(active, cx, cy, radius)``: units that cannot leave a circle this tick (The Hextech Ultimatum)."""
    active = jnp.zeros((n_units,), bool)
    cx = cy = rr = jnp.zeros((n_units,), jnp.float32)
    for k, st in _zip(state):
        if hasattr(k, "leash"):
            a, x, y, r = k.leash(st, kctx, n_units)
            cx, cy, rr = jnp.where(a, x, cx), jnp.where(a, y, cy), jnp.where(a, r, rr)
            active = active | a
    return active, cx, cy, rr


def resource_fraction(state: ChampionState, champion_ids, mana_fraction) -> Any:
    """(C,) resource bar: a kit's own resource where it has one (Renekton Fury), else ``mana_fraction``."""
    out = jnp.asarray(mana_fraction, jnp.float32)
    for k, st in _zip(state):
        if hasattr(k, "resource"):
            r = k.resource(st, jnp.asarray(champion_ids, jnp.int32))
            out = jnp.where(jnp.isnan(r), out, r)
    return out


def size(state: ChampionState, kctx: KitCtx) -> Any:
    """(C,) bonus size fraction from kits (Nasus R, Renekton R, Malphite passive); scales the gameplay radius
    (wiki Unit size; pathing and selection radii are unaffected)."""
    out = jnp.zeros(kctx.unit.shape, jnp.float32)
    for k, st in _zip(state):
        if hasattr(k, "size"):
            out = out + k.size(st, kctx)
    return out


def displacement(state: ChampionState, kctx: KitCtx, n_units: int) -> tuple[Any, Any, Any, Any]:
    """(N,) forced movement from kits (Darius E pull): ``(active, to_x, to_y, speed)``."""
    active = jnp.zeros((n_units,), bool)
    tx = ty = sp = jnp.zeros((n_units,), jnp.float32)
    for k, st in _zip(state):
        if hasattr(k, "displacement"):
            a, x, y, v = k.displacement(st, kctx, n_units)
            tx, ty, sp = jnp.where(a, x, tx), jnp.where(a, y, ty), jnp.where(a, v, sp)
            active = active | a
    return active, tx, ty, sp


def debuffs(state: ChampionState, kctx: KitCtx, units) -> Debuffs:
    """(N,) target-side reductions from kits (Garen E armor shred)."""
    n = units.x.shape[0]
    return combine_debuffs([k.debuffs(st, kctx, units) for k, st in _zip(state)], n)


__all__ = ["ChampionState", "KitAttackMods", "KitCtx", "KitDefense", "KitOut", "KITS", "BY_NAME", "kit", "init",
           "roster", "state_type", "kits_of",
           "cast", "periodic", "on_attack", "on_hit", "on_damage", "on_takedown", "stats", "defense",
           "attack_mods", "cc_immune", "debuffs", "displacement_immune", "hp_cap", "mana_override", "dodging_units", "ghosted", "immobile", "leash", "on_parry", "untargetable", "resource_fraction", "point_target_ranges", "displacement", "size",
           "unit_target_ranges", "rank_tables", "attack_missile_speed"]
