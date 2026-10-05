"""26.19 champion kits (Garen 86, Jax 24) for the modern world tick.

Every hook runs both kits for every holder; each kit gates itself on
``KitCtx.champion_id``, so holders are independent and shapes are fixed.
See ``core`` for the contract and the cast-id scheme.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from ..items.catalog import combine_stats
from ..items.effects.core import Debuffs, combine_debuffs
from . import garen, jax as jax_kit
from .core import (GAREN, JAX, KitAttackMods, KitCtx, KitDefense, KitOut, combine_attack_mods, merge_out,
                   neutral_attack_mods, neutral_defense)

KITS = {GAREN: garen, JAX: jax_kit}


class ChampionState(NamedTuple):
    garen: garen.State
    jax: jax_kit.State


def init(n_champions: int, n_units: int) -> ChampionState:
    return ChampionState(garen.init(n_champions, n_units), jax_kit.init(n_champions, n_units))


def _sizes(kctx, units):
    return kctx.unit.shape[0], units.x.shape[0]


def _both(fn_name, state, kctx, units, *args):
    c, n = _sizes(kctx, units)
    sg, og = getattr(garen, fn_name)(state.garen, kctx, units, *args)
    sj, oj = getattr(jax_kit, fn_name)(state.jax, kctx, units, *args)
    return ChampionState(sg, sj), merge_out([og, oj], c, n)


def unit_target_ranges(champion_ids) -> Any:
    """(C, 4) center-to-edge cast range of each slot's unit-targeted spell (0: not unit-targeted);
    the world walks a champion into this range before casting (``UNIT_TARGET_RANGE`` per kit)."""
    ids = jnp.asarray(champion_ids, jnp.int32)
    out = jnp.zeros(ids.shape + (4,), jnp.float32)
    for cid, mod in KITS.items():
        out = jnp.where((ids == cid)[:, None], jnp.asarray(mod.UNIT_TARGET_RANGE, jnp.float32)[None, :], out)
    return out


def cast(state: ChampionState, kctx: KitCtx, units, order) -> tuple[ChampionState, KitOut]:
    return _both("cast", state, kctx, units, order)


def periodic(state: ChampionState, kctx: KitCtx, units) -> tuple[ChampionState, KitOut]:
    return _both("periodic", state, kctx, units)


def on_attack(state: ChampionState, kctx: KitCtx, units, launch) -> tuple[ChampionState, KitOut]:
    return _both("on_attack", state, kctx, units, launch)


def dodging_units(state: ChampionState, kctx: KitCtx, n_units: int) -> Any:
    """(N,) bool: world units currently dodging basic attacks (Jax E)."""
    return jnp.zeros((n_units,), bool).at[kctx.unit].max(jax_kit.dodging(state.jax, kctx))


def on_hit(state: ChampionState, kctx: KitCtx, units, launch) -> tuple[ChampionState, KitOut]:
    return _both("on_hit", state, kctx, units, launch, dodging_units(state, kctx, units.x.shape[0]))


def on_damage(state: ChampionState, kctx: KitCtx, units, report) -> tuple[ChampionState, KitOut]:
    return _both("on_damage", state, kctx, units, report)


def on_takedown(state: ChampionState, kctx: KitCtx, units, kills) -> ChampionState:
    return ChampionState(garen.on_takedown(state.garen, kctx, units, kills),
                         jax_kit.on_takedown(state.jax, kctx, units, kills))


def stats(state: ChampionState, kctx: KitCtx):
    return combine_stats(garen.stats(state.garen, kctx), jax_kit.stats(state.jax, kctx))


def defense(state: ChampionState, kctx: KitCtx) -> KitDefense:
    out = neutral_defense(kctx.unit.shape[0])
    for d in (garen.defense(state.garen, kctx), jax_kit.defense(state.jax, kctx)):
        out = KitDefense(out.received_mult * d.received_mult, out.dodge_basic | d.dodge_basic,
                         out.aoe_received_mult * d.aoe_received_mult,
                         1.0 - (1.0 - out.tenacity_bonus) * (1.0 - d.tenacity_bonus))
    return out


def attack_mods(state: ChampionState, kctx: KitCtx) -> KitAttackMods:
    out = neutral_attack_mods(kctx.unit.shape[0])
    for m in (garen.attack_mods(state.garen, kctx), jax_kit.attack_mods(state.jax, kctx)):
        out = combine_attack_mods(out, m)
    return out


def ghosted(state: ChampionState, kctx: KitCtx) -> Any:
    """(C,) bool: holder ignores unit collision (Garen E Judgment)."""
    return garen.ghosted(state.garen, kctx)


def debuffs(state: ChampionState, kctx: KitCtx, units) -> Debuffs:
    """(N,) target-side reductions from kits (Garen E armor shred)."""
    n = units.x.shape[0]
    return combine_debuffs([garen.debuffs(state.garen, kctx, units), jax_kit.debuffs(state.jax, kctx, units)], n)


__all__ = ["ChampionState", "KitAttackMods", "KitCtx", "KitDefense", "KitOut", "GAREN", "JAX", "KITS", "init",
           "cast", "periodic", "on_attack", "on_hit", "on_damage", "on_takedown", "stats", "defense",
           "attack_mods", "debuffs", "dodging_units", "ghosted"]
