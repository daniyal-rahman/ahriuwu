"""Starter items, Glory, Cull and the Manaflow (Tear) line (ITEMS.md §6.3, §6.6, §6.9, §6.10, §9.1).

The module cannot edit inventories: the world applies ``pending_transforms`` with ``inventory.replace_item``; the
transformed items then carry their full mana statically and are not ``TearItems``, so Manaflow stacks stop counting.
Approximations: "bonus mana" = static item mana + Manaflow stacks (runes ignored); one Manaflow charge / Muramana
ability proc per target per cast start within ``InternalCDPerCastID`` (no cast ids); Diadem Consonance only heals the
holder, when it is the lowest-%HP allied champion in range.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np

from ...core.damage import (CLASS_CHAMPION, CLASS_MINION, CLASS_STRUCTURE, ON_HIT_ITEM, PHYSICAL, SHIELD_ALL,
                            TAG_ACTIVE_SPELL, TAG_AOE, TAG_BASIC_ATTACK, TAG_ITEM, TAG_PERIODIC, TAG_PROC,
                            concat_packets, has, packets)
from ..catalog import STAT_INDEX, ItemStats, catalog
from .core import (BIG, counts, dv, effects, enemy_mask, hit_by_holder, holds, holds_any, in_circle, neutral_defense,
                   onehot_units, row, shield_grants, target_class)

DORANS_SHIELD, DORANS_RING, DARK_SEAL, MEJAIS, CULL, DORANS_BOW, DORANS_HELM = \
    1054, 1056, 1082, 3041, 1083, 1086, 1120
TEAR, ARCHANGELS, SERAPHS, MANAMUNE, MURAMANA = 3070, 3003, 3040, 3004, 3042
WINTERS, FIMBULWINTER, CIRCLET, DIADEM = 3119, 3121, 2526, 2530

# Manaflow holders (client group TearItems) and their transforms.
MANAFLOW_ITEMS = (TEAR, ARCHANGELS, MANAMUNE, WINTERS, CIRCLET)
TRANSFORMS = ((ARCHANGELS, SERAPHS), (MANAMUNE, MURAMANA), (WINTERS, FIMBULWINTER), (CIRCLET, DIADEM))
# Tooltips: Manamune and Winter's charge on attacks too, the rest on abilities only (client over wiki for Circlet).
MANAFLOW_ON_HIT = (MANAMUNE, WINTERS)

# Helping Hand (§6.6): one named effect; the first held item in this order gives the value.
HELPING_HAND = ((DORANS_SHIELD, "BonusDamageToMinions"), (DORANS_RING, "BonusDamage"),
                (DORANS_HELM, "BonusDamageToMinions"), (TEAR, "BonusMinionDamage"))

DS_FULL_MISSING = 0.75           # Enduring Focus regen is full at 75% missing HP (wiki, ITEMS.md U-7)
CONSONANCE_PERIOD = 1.0          # wiki
EVERLASTING_NEARBY = 1000.0      # INFERRED: "more than one enemy nearby" radius

COVERAGE = {
    DORANS_SHIELD: "Enduring Focus missing-HP regen after champion damage (AoE/DoT reduced); Helping Hand",
    DORANS_RING: "Drain mana regen, boosted after damaging a champion (HP regen if manaless); Helping Hand",
    DARK_SEAL: "Glory stacks from takedowns, lost on death; AP per stack",
    MEJAIS: "Glory (stacks carried over from Dark Seal); AP per stack, MS at threshold",
    CULL: "on-hit heal; minion-kill gold then completion gold",
    DORANS_BOW: "stats only (dv HealthOnHit not in tooltip: not applied)",
    DORANS_HELM: "Helping Hand",
    TEAR: "Manaflow (abilities); Helping Hand",
    ARCHANGELS: "Manaflow; Awe AP from bonus mana; transforms to Seraph's (pending_transforms)",
    SERAPHS: "Awe AP; Lifeline shield from max mana",
    MANAMUNE: "Manaflow (attacks and abilities); Awe AD from max mana; transforms to Muramana",
    MURAMANA: "Awe AD; Shock on-hit and per-cast ability damage vs champions",
    WINTERS: "Manaflow (attacks and abilities); Awe HP from bonus mana; transforms to Fimbulwinter",
    FIMBULWINTER: "Awe HP; Everlasting shield on CC applied (on_cc)",
    CIRCLET: "Manaflow (abilities); Harmony HSP from bonus mana; transforms to Diadem",
    DIADEM: "Harmony HSP; Consonance heal for the holder only (ally heals deferred)",
}


def _coef(item_id: int, calc: str) -> float:
    return float(catalog()[item_id].calculations[calc]["mFormulaParts"][0]["mCoefficient"])


def _number(item_id: int, calc: str) -> float:
    return float(catalog()[item_id].calculations[calc]["mFormulaParts"][0]["mNumber"])


_ITEM_MANA = np.asarray(catalog().arrays.stats)[:, STAT_INDEX["mana"]].astype(np.float32)

SERAPH_SHIELD = _coef(SERAPHS, "ShieldValue")            # x max mana
MANAMUNE_AD = _coef(MANAMUNE, "BonusADFromMana")
MURAMANA_AD = _coef(MURAMANA, "BonusADFromMana")
MURAMANA_ONHIT = _coef(MURAMANA, "OnHitDamage")
MURAMANA_ABILITY_MELEE = _coef(MURAMANA, "MeleeItemCalcValue")
MURAMANA_ABILITY_RANGED = _coef(MURAMANA, "RangedItemCalcValue")
WINTERS_HP = _coef(WINTERS, "BonusHPFromMana")
FIMBUL_HP = _coef(FIMBULWINTER, "BonusHPFromMana")
CIRCLET_HSP = _coef(CIRCLET, "BonusHSPCalc")             # percent
DIADEM_HSP = _coef(DIADEM, "BonusHSPCalc")
DIADEM_HEAL = _coef(DIADEM, "ManaToHeal")                # dv PercentManaToHeal is unused
FIMBUL_SHIELD_BASE = _number(FIMBULWINTER, "ShieldBase")


class State(NamedTuple):
    ds_until: Any           # (C,) Enduring Focus end
    ds_eff: Any             # (C,) 1.0 or RangeRegenMult
    ring_until: Any         # (C,) Doran's Ring upgraded restore end
    glory: Any              # (C,) Glory stacks (per holder, shared Dark Seal/Mejai's)
    cull_kills: Any         # (C,) minion kills counted
    cull_done: Any          # (C,) bool, completion gold paid
    tear_mana: Any          # (C,) Manaflow bonus mana 0..360
    charges: Any            # (C,) Manaflow charges
    next_charge: Any        # (C,) time of next charge (when below max)
    cast_start: Any         # (C,) start time of the holder's latest ability cast
    flow_cast: Any          # (C,) cast_start that already consumed Manaflow
    shock_cast: Any         # (C, N) cast_start that already procced Muramana on unit n
    seraph_cd: Any          # (C,) Lifeline cooldown end
    fimbul_cd: Any          # (C,) Everlasting cooldown end
    consonance_next: Any    # (C,) next Consonance heal time


def init(n_champions: int, n_units: int) -> State:
    z = jnp.zeros((n_champions,), jnp.float32)
    return State(z - BIG, z + 1.0, z - BIG, z, z, jnp.zeros((n_champions,), bool), z, z, z + BIG,
                 z - BIG, z - BIG, jnp.full((n_champions, n_units), -BIG, jnp.float32), z - BIG, z - BIG,
                 z - BIG)


def _manaflow_value(own, name: str) -> Any:
    """(C,) data value of the held Manaflow item (0 if none)."""
    out = jnp.zeros(counts(own).shape[:1], jnp.float32)
    for iid in MANAFLOW_ITEMS:
        out = jnp.where(holds(own, iid), dv(iid, name), out)
    return out


def item_bonus_mana(state: State, own) -> Any:
    """(C,) bonus mana: static item mana + Manaflow stacks."""
    static = jnp.asarray(counts(own), jnp.float32) @ jnp.asarray(_ITEM_MANA)
    return static + jnp.where(holds_any(own, MANAFLOW_ITEMS), state.tear_mana, 0.0)


def _max_mana(state: State, own, ctx) -> Any:
    return ctx.max_mana + jnp.where(holds_any(own, MANAFLOW_ITEMS), state.tear_mana, 0.0)


def pending_transforms(state: State, own):
    """Tear-line transforms due now: ``(from_row, to_row, do)`` (C,), rows -1 = none."""
    c = counts(own).shape[0]
    frm = jnp.full((c,), -1, jnp.int32)
    to = jnp.full((c,), -1, jnp.int32)
    for a, b in TRANSFORMS:
        due = holds(own, a) & (state.tear_mana >= dv(a, "MaxMana"))
        frm = jnp.where(due, row(a), frm)
        to = jnp.where(due, row(b), to)
    return frm, to, frm >= 0


def stats(state: State, own, ctx) -> ItemStats:
    # Enduring Focus regen follows current missing HP.
    ds = holds(own, DORANS_SHIELD) & (ctx.now < state.ds_until) & ctx.alive
    max_regen = jnp.where(ctx.is_ranged, dv(DORANS_SHIELD, "MaxRangeRegenAmount"),
                          dv(DORANS_SHIELD, "MaxRegenAmount"))
    missing = jnp.clip(1.0 - ctx.hp / jnp.maximum(ctx.max_hp, 1.0), 0.0, 1.0)
    ds_regen = jnp.where(ds, max_regen / dv(DORANS_SHIELD, "RegenDuration")
                         * jnp.minimum(missing / DS_FULL_MISSING, 1.0) * state.ds_eff, 0.0)
    ring = holds(own, DORANS_RING)
    rate = jnp.where(ctx.now < state.ring_until, dv(DORANS_RING, "ManaRestorePerSecondUpgraded"),
                     dv(DORANS_RING, "ManaRestorePerSecond"))
    manaless = ctx.max_mana <= 0.0
    ring_mana = jnp.where(ring & ~manaless, rate, 0.0)
    ring_hp = jnp.where(ring & manaless, rate * dv(DORANS_RING, "ManaToHealthConversion"), 0.0)
    seal, mej = holds(own, DARK_SEAL), holds(own, MEJAIS)
    glory_ap = jnp.where(mej, dv(MEJAIS, "APPerGlory") * jnp.minimum(state.glory, dv(MEJAIS, "MaxGloryStacks")),
                         jnp.where(seal, dv(DARK_SEAL, "APPerGlory")
                                   * jnp.minimum(state.glory, dv(DARK_SEAL, "MaxGloryStacks")), 0.0))
    mej_ms = jnp.where(mej & (state.glory >= dv(MEJAIS, "GloryThreshold")), dv(MEJAIS, "MoveSpeedMod"), 0.0)
    flow = jnp.where(holds_any(own, MANAFLOW_ITEMS), state.tear_mana, 0.0)
    bonus_mana = item_bonus_mana(state, own)
    max_mana = _max_mana(state, own, ctx)
    awe_ap = jnp.where(holds(own, ARCHANGELS), dv(ARCHANGELS, "APFromMana") * bonus_mana, 0.0) \
        + jnp.where(holds(own, SERAPHS), dv(SERAPHS, "APFromMana") * bonus_mana, 0.0)
    awe_ad = jnp.where(holds(own, MANAMUNE), MANAMUNE_AD * max_mana, 0.0) \
        + jnp.where(holds(own, MURAMANA), MURAMANA_AD * max_mana, 0.0)
    awe_hp = jnp.where(holds(own, WINTERS), WINTERS_HP * bonus_mana, 0.0) \
        + jnp.where(holds(own, FIMBULWINTER), FIMBUL_HP * bonus_mana, 0.0)
    hsp = (jnp.where(holds(own, CIRCLET), CIRCLET_HSP * bonus_mana, 0.0)
           + jnp.where(holds(own, DIADEM), DIADEM_HSP * bonus_mana, 0.0)) / 100.0
    return ItemStats(health=awe_hp, attack_damage=awe_ad, ability_power=glory_ap + awe_ap,
                     percent_move_speed=mej_ms, health_regen=ds_regen + ring_hp, mana=flow,
                     mana_regen=ring_mana, heal_shield_power=hsp)


def defense(state: State, own, ctx):
    c = ctx.level.shape[0]
    ready = holds(own, SERAPHS) & (ctx.now >= state.seraph_cd) & ctx.alive
    return neutral_defense(c)._replace(
        lifeline_ready=ready, lifeline_shield=jnp.where(ready, SERAPH_SHIELD * ctx.max_mana, 0.0),
        lifeline_shield_kind=jnp.full((c,), SHIELD_ALL, jnp.int32),
        lifeline_duration=jnp.full((c,), dv(SERAPHS, "ShieldDuration"), jnp.float32),
        lifeline_decay_hold=jnp.full((c,), jnp.inf, jnp.float32))


def on_hit(state: State, own, ctx, units, attack):
    c, n = ctx.level.shape[0], units.x.shape[0]
    hit = attack.hit & ctx.alive & (attack.target >= 0)
    tcls = target_class(units, attack.target)
    tgt = jnp.maximum(attack.target, 0)
    enemy = units.team[tgt] != ctx.team

    hh_val = jnp.zeros((c,), jnp.float32)
    hh_item = jnp.zeros((c,), jnp.int32)
    for iid, name in reversed(HELPING_HAND):
        hh_val = jnp.where(holds(own, iid), dv(iid, name), hh_val)
        hh_item = jnp.where(holds(own, iid), iid, hh_item)
    hh = hit & (hh_item != 0) & (tcls == CLASS_MINION)
    p_hh = packets(hh, ctx.unit, tgt, hh_val, PHYSICAL, ON_HIT_ITEM, item=hh_item)

    mx = _max_mana(state, own, ctx)
    shock = hit & holds(own, MURAMANA) & (tcls == CLASS_CHAMPION) & enemy
    p_shock = packets(shock, ctx.unit, tgt, MURAMANA_ONHIT * mx, PHYSICAL, ON_HIT_ITEM, item=MURAMANA)

    heal = jnp.where(hit & holds(own, CULL), dv(CULL, "OnHitHeal"), 0.0)

    trig = hit & holds_any(own, MANAFLOW_ON_HIT) & enemy & (tcls != CLASS_STRUCTURE)
    state = _consume_charge(state, own, ctx, trig, tcls == CLASS_CHAMPION)
    return state, effects(c, n, packets=concat_packets(p_hh, p_shock), heal=heal)


def _consume_charge(state: State, own, ctx, trigger, champ) -> State:
    has_flow = holds_any(own, MANAFLOW_ITEMS)
    cap = _manaflow_value(own, "MaxMana")
    go = trigger & has_flow & (state.charges >= 1.0) & (state.tear_mana < cap)
    per = _manaflow_value(own, "ManaPerCharge") * jnp.where(champ, 2.0, 1.0)
    was_full = state.charges >= _manaflow_value(own, "ManaChargeMaxAmmo")
    return state._replace(
        tear_mana=jnp.where(go, jnp.minimum(state.tear_mana + per, cap), state.tear_mana),
        charges=jnp.where(go, state.charges - 1.0, state.charges),
        next_charge=jnp.where(go & was_full, ctx.now + _manaflow_value(own, "ManaChargeAmmoCD"),
                              state.next_charge))


def on_cast(state: State, own, ctx, units, cast):
    c, n = ctx.level.shape[0], units.x.shape[0]
    relevant = holds_any(own, MANAFLOW_ITEMS + (MURAMANA,))
    state = state._replace(cast_start=jnp.where(cast.started & relevant, ctx.now, state.cast_start))
    return state, effects(c, n)


def _ability_mask(report):
    f = report.packets.flags
    return has(f, TAG_ACTIVE_SPELL) & ~has(f, TAG_ITEM) & ~has(f, TAG_BASIC_ATTACK)


def on_damage(state: State, own, ctx, units, report):
    c, n = ctx.level.shape[0], units.x.shape[0]
    p, r = report.packets, report.resolved
    cls = units.cls
    src = jnp.clip(p.src, 0, n - 1)
    enemy_n = units.team[None, :] != ctx.team[:, None]                       # (C, N)
    champ_n = (cls == CLASS_CHAMPION)[None, :] & enemy_n

    from_champ = p.valid & (r.final > 0.0) & (cls[src] == CLASS_CHAMPION)
    to_me = (p.dst[None, :] == ctx.unit[:, None]) & from_champ[None, :] \
        & (units.team[src][None, :] != ctx.team[:, None])                    # (C, P)
    weak = has(p.flags, TAG_AOE) | has(p.flags, TAG_PERIODIC)
    any_hit = jnp.any(to_me, axis=1)
    strong = jnp.any(to_me & ~weak[None, :], axis=1)
    ds_go = any_hit & holds(own, DORANS_SHIELD)
    state = state._replace(
        ds_until=jnp.where(ds_go, ctx.now + dv(DORANS_SHIELD, "RegenDuration"), state.ds_until),
        ds_eff=jnp.where(ds_go, jnp.where(strong, 1.0, dv(DORANS_SHIELD, "RangeRegenMult")), state.ds_eff))

    dealt_any = hit_by_holder(report, ctx, n, p.valid & (r.final > 0.0))
    ring_go = holds(own, DORANS_RING) & jnp.any(dealt_any & champ_n, axis=1)
    state = state._replace(ring_until=jnp.where(ring_go, ctx.now + dv(DORANS_RING, "UpgradeDuration"),
                                                state.ring_until))

    # Manaflow / Muramana ability hits count once per cast instance.
    abil = hit_by_holder(report, ctx, n, _ability_mask(report)) & enemy_n \
        & units.alive[None, :] & (cls != CLASS_STRUCTURE)[None, :]
    live_cast = (state.cast_start > -BIG / 2) \
        & (ctx.now - state.cast_start <= _manaflow_value(own, "InternalCDPerCastID"))
    flow_trig = live_cast & (state.flow_cast < state.cast_start) & jnp.any(abil, axis=1)
    flow_champ = jnp.any(abil & champ_n, axis=1)
    before = state.tear_mana
    state = _consume_charge(state, own, ctx, flow_trig, flow_champ)
    used = flow_trig & holds_any(own, MANAFLOW_ITEMS) & (state.tear_mana != before)
    state = state._replace(flow_cast=jnp.where(used, state.cast_start, state.flow_cast))

    mura_cast = (state.cast_start > -BIG / 2) & (ctx.now - state.cast_start <= dv(MURAMANA, "PerCastIDLockout"))
    shock = abil & champ_n & (holds(own, MURAMANA) & ctx.alive & mura_cast)[:, None] \
        & (state.shock_cast < state.cast_start[:, None])
    ratio = jnp.where(ctx.is_ranged, MURAMANA_ABILITY_RANGED, MURAMANA_ABILITY_MELEE)
    dmg = ratio * _max_mana(state, own, ctx)
    p_shock = packets(shock, ctx.unit[:, None], jnp.arange(n)[None, :], dmg[:, None], PHYSICAL,
                      TAG_PROC | TAG_ITEM, item=MURAMANA)
    state = state._replace(shock_cast=jnp.where(shock, state.cast_start[:, None], state.shock_cast))

    fired = r.lifeline_fired[jnp.clip(ctx.unit, 0, n - 1)] & holds(own, SERAPHS) & (ctx.now >= state.seraph_cd)
    state = state._replace(seraph_cd=jnp.where(fired, ctx.now + dv(SERAPHS, "LifelineCooldown"), state.seraph_cd))
    return state, effects(c, n, packets=p_shock)


def periodic(state: State, own, ctx, units):
    c, n = ctx.level.shape[0], units.x.shape[0]
    has_flow = holds_any(own, MANAFLOW_ITEMS)
    max_ammo = _manaflow_value(own, "ManaChargeMaxAmmo")
    cd = jnp.maximum(_manaflow_value(own, "ManaChargeAmmoCD"), 1e-3)
    # Charges accrue while below max; a new Manaflow item starts at 0 (INFERRED).
    below = state.charges < max_ammo
    nxt = jnp.where(below & (state.next_charge >= BIG / 2), ctx.now + cd, state.next_charge)
    due = below & (ctx.now >= nxt)
    k = jnp.where(due, jnp.floor((ctx.now - nxt) / cd) + 1.0, 0.0)
    charges = jnp.minimum(state.charges + k, max_ammo)
    nxt = jnp.where(due, nxt + k * cd, nxt)
    nxt = jnp.where(charges >= max_ammo, BIG, nxt)
    state = state._replace(
        charges=jnp.where(has_flow, charges, 0.0), next_charge=jnp.where(has_flow, nxt, BIG),
        tear_mana=jnp.where(has_flow, state.tear_mana, 0.0),
        # Selling Cull resets its counter (INFERRED).
        cull_kills=jnp.where(holds(own, CULL), state.cull_kills, 0.0),
        cull_done=state.cull_done & holds(own, CULL))

    diadem = holds(own, DIADEM) & ctx.alive & ctx.in_combat
    me = onehot_units(ctx.unit, n)
    ally = (units.team[None, :] == ctx.team[:, None]) & (units.cls == CLASS_CHAMPION)[None, :] \
        & units.alive[None, :] & in_circle(units, ctx.x, ctx.y, jnp.full((c,), dv(DIADEM, "AllyRangeCheck")),
                                           edge=False)
    ratio = units.hp / jnp.maximum(units.max_hp, 1.0)
    my_ratio = ctx.hp / jnp.maximum(ctx.max_hp, 1.0)
    others_min = jnp.min(jnp.where(ally & ~me, ratio[None, :], jnp.inf), axis=1)
    lowest = my_ratio <= others_min
    tick = diadem & (ctx.now >= state.consonance_next)
    heal = jnp.where(tick & lowest, DIADEM_HEAL * ctx.max_mana, 0.0)
    state = state._replace(consonance_next=jnp.where(tick, ctx.now + CONSONANCE_PERIOD, state.consonance_next))
    return state, effects(c, n, heal=heal)


def on_takedown(state: State, own, ctx, units, kills):
    c, n = ctx.level.shape[0], units.x.shape[0]
    seal, mej = holds(own, DARK_SEAL), holds(own, MEJAIS)
    item = lambda name: jnp.where(mej, dv(MEJAIS, name), jnp.where(seal, dv(DARK_SEAL, name), 0.0))
    glory = state.glory + item("GloryOnKill") * kills.champion_kill + item("GloryOnAssist") * kills.champion_assist
    glory = jnp.minimum(glory, jnp.maximum(item("MaxGloryStacks"), state.glory))
    glory = jnp.where(kills.holder_died, glory - item("GloryLossOnDeath"), glory)
    glory = jnp.where(seal | mej, jnp.maximum(glory, 0.0), state.glory)

    cull = holds(own, CULL)
    limit = dv(CULL, "MinionKillThreshold")
    new_kills = jnp.minimum(state.cull_kills + kills.minion_kill, limit)
    finish = cull & (new_kills >= limit) & ~state.cull_done
    gold = jnp.where(cull, (new_kills - state.cull_kills) * dv(CULL, "MinionKillGold"), 0.0) \
        + jnp.where(finish, dv(CULL, "CompleteGold"), 0.0)
    state = state._replace(glory=glory, cull_kills=jnp.where(cull, new_kills, state.cull_kills),
                           cull_done=state.cull_done | finish)
    return state, effects(c, n, gold=gold)


def everlasting(state: State, own, ctx, units, immobilized, slowed):
    """Fimbulwinter Everlasting when the holder immobilizes/slows (C,) an enemy champion; slows count for melee."""
    c, n = ctx.level.shape[0], units.x.shape[0]
    trig = holds(own, FIMBULWINTER) & ctx.alive & (ctx.now >= state.fimbul_cd) \
        & (immobilized | (slowed & ~ctx.is_ranged))
    near = in_circle(units, ctx.x, ctx.y, jnp.full((c,), EVERLASTING_NEARBY)) & enemy_mask(ctx, units) \
        & (units.cls == CLASS_CHAMPION)[None, :]
    mult = jnp.where(jnp.sum(near, axis=1) > 1, 1.0 + dv(FIMBULWINTER, "Multiplier"), 1.0)
    amount = jnp.where(trig, (FIMBUL_SHIELD_BASE + dv(FIMBULWINTER, "CurrentManaShieldRatio") * ctx.mana) * mult, 0.0)
    state = state._replace(fimbul_cd=jnp.where(trig, ctx.now + dv(FIMBULWINTER, "Cooldown"), state.fimbul_cd))
    return state, effects(c, n, shields=shield_grants(amount, SHIELD_ALL, dv(FIMBULWINTER, "ShieldDuration")))


def on_cc(state: State, own, ctx, units, cc):
    """ITEMS H14: the holder slowed/immobilized an enemy champion this tick."""
    champs = enemy_mask(ctx, units) & (units.cls == CLASS_CHAMPION)[None, :]
    return everlasting(state, own, ctx, units, jnp.any(cc.immobilized & champs, axis=1),
                       jnp.any(cc.slowed & champs, axis=1))
