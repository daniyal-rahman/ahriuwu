"""Tier-2/3 boots passives (ITEMS.md §9.2, §6.7, §13 D3/D5); their stat lines come from the catalog.

``Feats*`` data values are ignored: Feats of Strength was removed in 26.1.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from ...core.damage import CLASS_CHAMPION, MAGIC, PHYSICAL, SHIELD_MAGIC, SHIELD_PHYSICAL, TAG_ACTIVE_SPELL, has
from ..catalog import ItemStats, catalog, level_bp
from .core import (Effects, HolderDefense, dst_class, dv, effects, holds, holds_any, neutral_defense,
                   shield_grants, src_class)

BERSERKERS, GLUTTONOUS, SWIFTNESS, SORCERERS, STEELCAPS, MERCURYS, IONIAN = 3006, 3008, 3009, 3020, 3047, 3111, 3158
IMMORTAL_PATH, SWIFTMARCH, CRIMSON, GUNMETAL, CHAINLACED, ARMORED, SPELLSLINGER = (
    3168, 3170, 3171, 3172, 3173, 3174, 3175)

STEELCAPS_REDUCTION = catalog()[STEELCAPS].effect_amount[0]
ARMORED_REDUCTION = dv(ARMORED, "DamageReduction")

SLAY_ITEMS = (GLUTTONOUS, IMMORTAL_PATH)
SLAY_PER_STACK = dv(GLUTTONOUS, "OmnivampOnTakedown")
SLAY_MAX = dv(GLUTTONOUS, "MaxStacks")
IP_DAMAGE, IP_HEALING = dv(IMMORTAL_PATH, "DamageMod"), dv(IMMORTAL_PATH, "HealingMod")
IP_THRESHOLD = 0.5        # tooltip "half Health"

SWIFTMARCH_AF = dv(SWIFTMARCH, "MSAdaptiveRatio")
ADAPTIVE_AD_PER_AF = 0.6  # DAMAGE_AND_STATS §3.5 (1 AF = 0.6 bonus AD or 1 AP)

CRIMSON_MS, CRIMSON_RANGED = dv(CRIMSON, "MeleeMS"), dv(CRIMSON, "RangedMSMultiplier")
CRIMSON_DURATION = dv(CRIMSON, "Duration")


def _shield_calc(item_id: int):
    parts = catalog()[item_id].calculations["ShieldAmountCalc"]["mFormulaParts"]
    bp = parts[0]
    (step,) = bp["mBreakpoints"]
    return bp["mLevel1Value"], step["mBonusPerLevelAtAndAfter"], step["mLevel"], dv(item_id, parts[1]["mDataValue"])


# Noxian Endurance (physical) / Persistence (magic): level_bp parts, bonus-HP ratio, cd, duration, dtype, kind.
NOXIAN = {
    ARMORED: (*_shield_calc(ARMORED), dv(ARMORED, "Cooldown"), dv(ARMORED, "ShieldDuration"), PHYSICAL,
              SHIELD_PHYSICAL),
    CHAINLACED: (*_shield_calc(CHAINLACED), dv(CHAINLACED, "Cooldown"), dv(CHAINLACED, "ShieldDuration"), MAGIC,
                 SHIELD_MAGIC),
}

COVERAGE = {
    BERSERKERS: "stats only", SWIFTNESS: "stats only", SORCERERS: "stats only", MERCURYS: "stats only",
    SPELLSLINGER: "stats only",
    GUNMETAL: "stats only (MeleeMS/Duration data values are unreferenced)",
    GLUTTONOUS: "Slay: omnivamp per champion takedown, kept while a Slay boot is held",
    STEELCAPS: "Plating: basic attacks taken x0.9 (not turrets or on-hit parts, U-6)",
    IONIAN: "Ionian Insight: summoner haste",
    IMMORTAL_PATH: "Slay (shared stacks); Now and Forever: damage amp above 50% HP, incoming heal below",
    SWIFTMARCH: "Noxian Fervor: adaptive force from move speed (unreferenced MoveSpeedMultiplier ignored, D5)",
    CRIMSON: "Ionian Insight; Noxian Haste on ability damage to champions (ally and summoner triggers not modelled)",
    CHAINLACED: "Noxian Persistence: magic shield after enemy champion magic damage (client formula, D3)",
    ARMORED: "Plating; Noxian Endurance: physical shield after enemy champion physical damage (D3)",
}


class State(NamedTuple):
    slay_stacks: Any        # (C,) Gluttonous/Immortal Path takedown stacks
    noxian_cd_until: Any    # (C,) Armored Advance / Chainlaced shield cooldown (one boot ownable)
    crimson_until: Any      # (C,) Noxian Haste MS end


def init(n_champions: int, n_units: int) -> State:
    z = jnp.zeros((n_champions,), jnp.float32)
    return State(z, z - 1e9, z - 1e9)


def stats(state: State, own, ctx) -> ItemStats:
    slay = holds_any(own, SLAY_ITEMS)
    summ = jnp.where(holds(own, IONIAN), dv(IONIAN, "SummonerHaste"), 0.0) \
        + jnp.where(holds(own, CRIMSON), dv(CRIMSON, "SummonerHaste"), 0.0)
    low = ctx.hp < IP_THRESHOLD * ctx.max_hp
    ip_heal = jnp.where(holds(own, IMMORTAL_PATH) & low, IP_HEALING, 0.0)
    # Swiftmarch adaptive type: bonus AD vs AP, tie -> AD (the champion's adaptive type is not in Ctx).
    af = jnp.where(holds(own, SWIFTMARCH), SWIFTMARCH_AF * ctx.move_speed, 0.0)
    to_ad = ctx.bonus_ad >= ctx.ap
    crimson = holds(own, CRIMSON) & (ctx.now < state.crimson_until)
    cms = CRIMSON_MS * jnp.where(ctx.is_ranged, CRIMSON_RANGED, 1.0)
    return ItemStats(omnivamp=jnp.where(slay, SLAY_PER_STACK * state.slay_stacks, 0.0),
                     summoner_haste=summ, incoming_heal=ip_heal,
                     attack_damage=jnp.where(to_ad, ADAPTIVE_AD_PER_AF * af, 0.0),
                     ability_power=jnp.where(to_ad, 0.0, af),
                     percent_move_speed=jnp.where(crimson, cms, 0.0))


def dealt_amp(state: State, own, ctx, units) -> Any:
    n = units.x.shape[0]
    high = ctx.hp > IP_THRESHOLD * ctx.max_hp
    amp = jnp.where(holds(own, IMMORTAL_PATH) & high, IP_DAMAGE, 0.0)
    return jnp.broadcast_to(amp[:, None], (amp.shape[0], n)).astype(jnp.float32)


def defense(state: State, own, ctx) -> HolderDefense:
    c = ctx.level.shape[0]
    mult = jnp.where(holds(own, STEELCAPS), 1.0 - STEELCAPS_REDUCTION, 1.0) \
        * jnp.where(holds(own, ARMORED), 1.0 - ARMORED_REDUCTION, 1.0)
    return neutral_defense(c)._replace(basic_attack_mult=mult.astype(jnp.float32))


def on_damage(state: State, own, ctx, units, report) -> tuple[State, Effects]:
    c, n = ctx.level.shape[0], units.x.shape[0]
    p, r = report.packets, report.resolved
    nsrc = units.team.shape[0]
    src_team = units.team[jnp.clip(p.src, 0, nsrc - 1)]
    from_enemy_champ = p.valid & (r.final > 0.0) & (src_class(report, units) == CLASS_CHAMPION) \
        & (p.src != p.dst)
    to_me = p.dst[None, :] == ctx.unit[:, None]                               # (C, P)
    enemy = src_team[None, :] != ctx.team[:, None]
    ready = ctx.alive & (ctx.now >= state.noxian_cd_until)
    amount = jnp.zeros((c,), jnp.float32)
    kind = jnp.zeros((c,), jnp.int32)
    duration = jnp.zeros((c,), jnp.float32)
    cd_until = state.noxian_cd_until
    for item, (l1, per, at, bonus_ratio, cd, dur, dtype, skind) in NOXIAN.items():
        took = jnp.any(to_me & enemy & (from_enemy_champ & (p.dtype == dtype))[None, :], axis=1)
        go = ready & holds(own, item) & took
        val = level_bp(l1, per, at, ctx.level) + bonus_ratio * ctx.bonus_hp
        amount = jnp.where(go, val, amount)
        kind = jnp.where(go, skind, kind)
        duration = jnp.where(go, dur, duration)
        cd_until = jnp.where(go, ctx.now + cd, cd_until)

    # Crimson Lucidity: the holder's ability damage to an enemy champion.
    from_me = p.src[None, :] == ctx.unit[:, None]
    dst_team = units.team[jnp.clip(p.dst, 0, nsrc - 1)]
    spell_hit = p.valid & (r.final > 0.0) & has(p.flags, TAG_ACTIVE_SPELL) \
        & (dst_class(report, units) == CLASS_CHAMPION)
    hit_champ = jnp.any(from_me & (dst_team[None, :] != ctx.team[:, None]) & spell_hit[None, :], axis=1)
    crim = holds(own, CRIMSON) & ctx.alive & hit_champ
    state = state._replace(noxian_cd_until=cd_until,
                           crimson_until=jnp.where(crim, ctx.now + CRIMSON_DURATION, state.crimson_until))
    return state, effects(c, n, shields=shield_grants(amount, kind, duration))


def on_takedown(state: State, own, ctx, units, kills) -> tuple[State, Effects]:
    c, n = ctx.level.shape[0], units.x.shape[0]
    gain = jnp.where(holds_any(own, SLAY_ITEMS), kills.champion_kill + kills.champion_assist, 0.0)
    state = state._replace(slay_stacks=jnp.minimum(state.slay_stacks + gain, SLAY_MAX).astype(jnp.float32))
    return state, effects(c, n)


def periodic(state: State, own, ctx, units) -> tuple[State, Effects]:
    """Selling every Slay boot clears the stacks (INFERRED M); upgrading Gluttonous keeps them."""
    c, n = ctx.level.shape[0], units.x.shape[0]
    state = state._replace(slay_stacks=jnp.where(holds_any(own, SLAY_ITEMS), state.slay_stacks, 0.0))
    return state, effects(c, n)
