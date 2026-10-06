"""Potions, elixirs and rune-granted consumables (ITEMS.md §9.1, F18, U-12; RUNES.md §7.3).

``active`` starts an effect when ``request == item_id``; ``state.consume_row`` (C,) is the catalog row whose unit
the world removes this tick (``inventory.consume_one``), -1 none. Refillable Potion keeps its charges here and
refills ``on_shop``. Potion HoTs tick every 0.5 s through ``heal_plain`` (first tick 0.5 s after drinking) and
stack as independent HoTs in ``HOT_SLOTS`` (U-12); with all slots busy the one with fewest ticks left is replaced.
Elixirs last mEffectAmount minutes, can be drunk while dead and replace each other. Biscuit heal is computed when
it starts (up to 3 queue) and each eaten biscuit adds permanent ``silent_health``.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from ...core.damage import (CLASS_CHAMPION, CLASS_MINION, CLASS_STRUCTURE, ON_HIT_ITEM, PHYSICAL, TAG_AOE,
                            TAG_ITEM, TAG_PROC, TRUE, has, packets)
from ..catalog import ItemStats, catalog
from .core import BIG, ActiveOut, dealt_by_holder, dv, effects, holds, row, target_class

HEALTH_POTION, REFILLABLE, IRON, SORCERY, WRATH = 2003, 2031, 2138, 2139, 2140
BISCUIT, SKILL, AVARICE, FORCE = 2010, 2150, 2151, 2152
BISCUIT_DURATION = 5.0             # perk 8345 DurationOfEffect
BISCUIT_FLAT, BISCUIT_PCT = 20.0, 0.015
BISCUIT_MISSING_FULL = 0.70        # heal doubles at <= 30% HP (MinHPThresholdTOOLTIP 0.3)
BISCUIT_PERMANENT_HP = 30.0
BISCUIT_QUEUE = 3
ELIXIRS = (IRON, SORCERY, WRATH)
HOT_SLOTS = 4
HOT_PERIOD = 0.5           # wiki (ITEMS.md F18)
POTION_COOLDOWN = 1.0      # wiki
WRATH_AOE_MULT = 0.33      # wiki (ITEMS.md §7)


def _effect(item_id: int, index: int) -> float:
    return catalog()[item_id].effect_amount[index]


ELIXIR_DURATION = {IRON: _effect(IRON, 2) * 60.0, SORCERY: _effect(SORCERY, 3) * 60.0,
                   WRATH: _effect(WRATH, 3) * 60.0}
IRON_HP, IRON_TENACITY = _effect(IRON, 0), _effect(IRON, 1)
SORCERY_AP, SORCERY_TRUE, SORCERY_ICD = _effect(SORCERY, 1), _effect(SORCERY, 2), _effect(SORCERY, 4)
SORCERY_MANA_REGEN = _effect(SORCERY, 5)   # per second
WRATH_AD, WRATH_DRAIN = _effect(WRATH, 1), _effect(WRATH, 2)

COVERAGE = {
    HEALTH_POTION: "HoT (heal_plain), shared 1 s potion cd, consumed",
    REFILLABLE: "2-charge HoT, charges refill on_shop; never consumed",
    IRON: "HP and tenacity (size and ally MS not modelled); replaces other elixir",
    SORCERY: "AP, mana regen; true damage on champion (per-target cd) or structure damage; replaces other elixir",
    WRATH: "AD; heals a share of physical damage to champions (AoE reduced); replaces other elixir",
    BISCUIT: "missing-HP-scaled HoT, queue of 3; permanent silent max HP",
    SKILL: "+1 skill point (skill_points counter for the world)",
    AVARICE: "true on-hit vs minions; gold on expiry",
    FORCE: "adaptive force",
}


class State(NamedTuple):
    hot_per_tick: Any       # (C, K)
    hot_next: Any           # (C, K) next tick time
    hot_left: Any           # (C, K) ticks remaining
    potion_cd: Any          # (C,)
    refill_charges: Any     # (C,)
    elixir: Any             # (C,) int32 active elixir id, 0 none
    elixir_until: Any       # (C,)
    sorcery_cd: Any         # (C, N) per-target true-damage cooldown end
    consume_row: Any        # (C,) int32 row to consume this tick, -1 none
    drank: Any              # (C,) int32 potion started this tick (Time Warp Tonic), 0 none
    biscuit_queue: Any      # (C,) biscuits eaten but not started
    biscuit_next: Any       # (C,) next HoT tick of the running biscuit
    biscuit_left: Any       # (C,) ticks remaining of the running biscuit
    biscuit_per_tick: Any   # (C,)
    biscuits_eaten: Any     # (C,) total (permanent HP)
    skill_points: Any       # (C,) int32 granted so far
    avarice_until: Any      # (C,)
    force_until: Any        # (C,)


def init(n_champions: int, n_units: int) -> State:
    z = jnp.zeros((n_champions, HOT_SLOTS), jnp.float32)
    zc = jnp.zeros((n_champions,), jnp.float32)
    return State(z, z, z, zc - BIG, zc + dv(REFILLABLE, "MaxCharges"), jnp.zeros((n_champions,), jnp.int32),
                 zc - BIG, jnp.full((n_champions, n_units), -BIG, jnp.float32),
                 jnp.full((n_champions,), -1, jnp.int32), jnp.zeros((n_champions,), jnp.int32),
                 zc, zc, zc, zc, zc, jnp.zeros((n_champions,), jnp.int32), zc - BIG, zc - BIG)


def _elixir_on(state: State, ctx, item: int) -> Any:
    return (state.elixir == item) & (ctx.now < state.elixir_until)


def stats(state: State, own, ctx) -> ItemStats:
    iron, sorc, wrath = (_elixir_on(state, ctx, i) for i in ELIXIRS)
    biscuit_hp = BISCUIT_PERMANENT_HP * state.biscuits_eaten
    return ItemStats(health=jnp.where(iron, IRON_HP, 0.0) + biscuit_hp, silent_health=biscuit_hp,
                     adaptive_force=jnp.where(ctx.now < state.force_until, dv(FORCE, "AdaptiveAmount"), 0.0),
                     tenacity=jnp.where(iron, IRON_TENACITY, 0.0),
                     ability_power=jnp.where(sorc, SORCERY_AP, 0.0),
                     mana_regen=jnp.where(sorc, SORCERY_MANA_REGEN, 0.0),
                     attack_damage=jnp.where(wrath, WRATH_AD, 0.0))


def _add_hot(state: State, go, per_tick, ticks, now) -> State:
    free = state.hot_left <= 0
    slot = jnp.where(jnp.any(free, axis=1), jnp.argmax(free, axis=1), jnp.argmin(state.hot_left, axis=1))
    put = (jnp.arange(HOT_SLOTS)[None, :] == slot[:, None]) & go[:, None]
    return state._replace(hot_per_tick=jnp.where(put, per_tick[:, None], state.hot_per_tick),
                          hot_next=jnp.where(put, now + HOT_PERIOD, state.hot_next),
                          hot_left=jnp.where(put, ticks, state.hot_left))


def active(state: State, own, ctx, units, request):
    c, n = ctx.level.shape[0], units.x.shape[0]
    potion_ok = ctx.alive & (ctx.now >= state.potion_cd)
    hp_go = (request == HEALTH_POTION) & holds(own, HEALTH_POTION) & potion_ok
    rf_go = (request == REFILLABLE) & holds(own, REFILLABLE) & potion_ok & (state.refill_charges >= 1.0)
    ticks_hp = dv(HEALTH_POTION, "PotionDuration") / HOT_PERIOD
    ticks_rf = dv(REFILLABLE, "PotionDuration") / HOT_PERIOD
    per = jnp.where(hp_go, dv(HEALTH_POTION, "HealAmount") / ticks_hp, dv(REFILLABLE, "HealAmount") / ticks_rf)
    ticks = jnp.where(hp_go, ticks_hp, ticks_rf)[:, None]
    pot = hp_go | rf_go
    state = _add_hot(state, pot, per, ticks, ctx.now)
    consume = jnp.where(hp_go, row(HEALTH_POTION), -1)
    elixir, until = state.elixir, state.elixir_until
    for item in ELIXIRS:
        go = (request == item) & holds(own, item)       # usable while dead
        elixir = jnp.where(go, item, elixir)
        until = jnp.where(go, ctx.now + ELIXIR_DURATION[item], until)
        consume = jnp.where(go, row(item), consume)
    # Rune-granted consumables have no potion cooldown.
    bis = (request == BISCUIT) & holds(own, BISCUIT) & ctx.alive
    skill = (request == SKILL) & holds(own, SKILL)
    avarice = (request == AVARICE) & holds(own, AVARICE)
    force = (request == FORCE) & holds(own, FORCE)
    for go, item in ((bis, BISCUIT), (skill, SKILL), (avarice, AVARICE), (force, FORCE)):
        consume = jnp.where(go, row(item), consume)
    state = state._replace(
        biscuit_queue=jnp.where(bis, jnp.minimum(state.biscuit_queue + 1.0, BISCUIT_QUEUE), state.biscuit_queue),
        biscuits_eaten=state.biscuits_eaten + bis,
        skill_points=state.skill_points + skill.astype(jnp.int32),
        avarice_until=jnp.where(avarice, ctx.now + dv(AVARICE, "Duration"), state.avarice_until),
        force_until=jnp.where(force, ctx.now + dv(FORCE, "Duration"), state.force_until))
    used = pot | (consume >= 0)
    state = state._replace(potion_cd=jnp.where(pot, ctx.now + POTION_COOLDOWN, state.potion_cd),
                           refill_charges=jnp.where(rf_go, state.refill_charges - 1.0, state.refill_charges),
                           elixir=elixir, elixir_until=until, consume_row=consume.astype(jnp.int32),
                           drank=jnp.where(hp_go, HEALTH_POTION, jnp.where(rf_go, REFILLABLE, 0)).astype(jnp.int32))
    f = jnp.zeros((c,), bool)
    return state, effects(c, n), ActiveOut(used, jnp.zeros((c,), jnp.float32), ~f, f)


def periodic(state: State, own, ctx, units):
    c, n = ctx.level.shape[0], units.x.shape[0]
    due = (state.hot_left > 0) & (ctx.now >= state.hot_next)
    k = jnp.where(due, jnp.minimum(jnp.floor((ctx.now - state.hot_next) / HOT_PERIOD) + 1.0, state.hot_left), 0.0)
    alive = ctx.alive[:, None]
    heal = jnp.sum(jnp.where(alive, k * state.hot_per_tick, 0.0), axis=1)
    # HoTs end on death (INFERRED).
    left = jnp.where(alive, state.hot_left - k, 0.0)
    expired = state.elixir_until <= ctx.now
    # Biscuits: start the next queued one when none is running.
    b_due = (state.biscuit_left > 0) & (ctx.now >= state.biscuit_next)
    b_k = jnp.where(b_due, jnp.minimum(jnp.floor((ctx.now - state.biscuit_next) / HOT_PERIOD) + 1.0,
                                       state.biscuit_left), 0.0)
    heal = heal + jnp.where(ctx.alive, b_k * state.biscuit_per_tick, 0.0)
    b_left = jnp.where(ctx.alive, state.biscuit_left - b_k, 0.0)
    start = (b_left <= 0) & (state.biscuit_queue > 0) & ctx.alive
    missing = jnp.clip((1.0 - ctx.hp / jnp.maximum(ctx.max_hp, 1.0)) / BISCUIT_MISSING_FULL, 0.0, 1.0)
    ticks = BISCUIT_DURATION / HOT_PERIOD
    b_total = (BISCUIT_FLAT + BISCUIT_PCT * ctx.max_hp) * (1.0 + missing)
    gold = jnp.where((state.avarice_until > ctx.now - ctx.dt) & (state.avarice_until <= ctx.now),
                     dv(AVARICE, "GoldAmount"), 0.0)
    state = state._replace(
        biscuit_queue=state.biscuit_queue - start, biscuit_left=jnp.where(start, ticks, b_left),
        biscuit_next=jnp.where(start, ctx.now + HOT_PERIOD, state.biscuit_next + b_k * HOT_PERIOD),
        biscuit_per_tick=jnp.where(start, b_total / ticks, state.biscuit_per_tick))
    state = state._replace(
        hot_next=state.hot_next + k * HOT_PERIOD, hot_left=left,
        # A re-bought Refillable starts full.
        refill_charges=jnp.where(holds(own, REFILLABLE), state.refill_charges, dv(REFILLABLE, "MaxCharges")),
        elixir=jnp.where(expired, 0, state.elixir))
    return state, effects(c, n, heal_plain=heal, gold=gold)


def on_hit(state: State, own, ctx, units, attack):
    c, n = ctx.level.shape[0], units.x.shape[0]
    go = attack.hit & (ctx.now < state.avarice_until) & (target_class(units, attack.target) == CLASS_MINION)
    p = packets(go, ctx.unit, jnp.maximum(attack.target, 0), dv(AVARICE, "OnHitDamage"), TRUE, ON_HIT_ITEM,
                item=AVARICE)
    return state, effects(c, n, packets=p)


def on_shop(state: State, own, ctx) -> State:
    refill = holds(own, REFILLABLE) & ctx.in_shop
    return state._replace(refill_charges=jnp.where(refill, dv(REFILLABLE, "MaxCharges"), state.refill_charges))


def on_damage(state: State, own, ctx, units, report):
    c, n = ctx.level.shape[0], units.x.shape[0]
    p, r = report.packets, report.resolved
    dcls = units.cls[jnp.clip(p.dst, 0, n - 1)]
    enemy_n = units.team[None, :] != ctx.team[:, None]
    champ_n = (units.cls == CLASS_CHAMPION)[None, :] & enemy_n

    phys = p.valid & (p.dtype == PHYSICAL) & (dcls == CLASS_CHAMPION)
    single = dealt_by_holder(report, ctx, n, phys & ~has(p.flags, TAG_AOE))
    aoe = dealt_by_holder(report, ctx, n, phys & has(p.flags, TAG_AOE))
    drained = jnp.sum(jnp.where(champ_n, single + WRATH_AOE_MULT * aoe, 0.0), axis=1)
    heal = jnp.where(_elixir_on(state, ctx, WRATH), WRATH_DRAIN * drained, 0.0)

    dealt = dealt_by_holder(report, ctx, n, p.valid & (r.final > 0.0) & (p.item != SORCERY)) > 0.0
    is_struct = (units.cls == CLASS_STRUCTURE)[None, :] & enemy_n
    sorc = _elixir_on(state, ctx, SORCERY)[:, None] & dealt & units.alive[None, :] \
        & ((champ_n & (ctx.now >= state.sorcery_cd)) | is_struct)
    p_true = packets(sorc, ctx.unit[:, None], jnp.arange(n)[None, :], SORCERY_TRUE, TRUE, TAG_PROC | TAG_ITEM,
                     item=SORCERY)
    state = state._replace(sorcery_cd=jnp.where(sorc & champ_n, ctx.now + SORCERY_ICD, state.sorcery_cd))
    return state, effects(c, n, packets=p_true, heal=heal)
