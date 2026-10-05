"""Tank, Lifeline, Annul, Thorns and Immolate items (ITEMS.md §6.3–6.7, §9.3).

Values come from the 16.19.8230722 item data: data values via ``dv`` and
client calculations through the small evaluator ``calc`` (no hand-copied
numbers). Rules the data does not encode follow ITEMS.md / ITEMS_CATALOG
defaults and are marked INFERRED below:

* Champion combat (Unending Despair, Jak'Sho, Maw omnivamp) = any packet
  between the holder and an enemy champion within the last 5 s
  (DAMAGE_AND_STATS §13 default window).
* Immolate first tick 1 s after activation, then 1 Hz while the 3 s aura is
  refreshed (U-4); its own damage does not refresh the aura.
* Unending Despair pulses as soon as champion combat starts, then at most
  every ``Cooldown`` (4) s while it lasts.
* Kaenic's shield persists until broken (very long duration); a regrant
  only tops the remaining Kaenic shield up to 15% max HP.
* Warmog's Heart heals on 0.5 s game-clock boundaries (regen convention).
* Heartsteel charge accrues continuously while the enemy champion is in
  range; leaving range for ``RangeTrackingBuffDuration`` (5 s) resets it.
* Shield amounts (Lifeline, Kaenic) are BASE values: heal/shield power and
  Spirit Visage ``incoming_heal`` (which also covers ShieldIncrease) are left
  to the integrator (SHIELD.10/20), exactly like ``Effects.heal``.
"""
from __future__ import annotations

import json
from typing import Any, NamedTuple

import jax.numpy as jnp

from ...core.damage import (CLASS_CHAMPION, CLASS_MINION, CLASS_MONSTER, CLASS_STRUCTURE, MAGIC, ON_HIT_ITEM,
                            PHYSICAL, PROP_LIFESTEAL, PROP_REACTIVE, SHIELD_ALL, SHIELD_MAGIC, TAG_AOE,
                            TAG_BASIC_ATTACK, TAG_ITEM, TAG_PERIODIC, TAG_PROC, concat_packets, has, packets,
                            shield_value)
from ..catalog import DATA_PATH, STAT_INDEX, ItemStats, catalog, lerp_level, ranged_mult
from .core import (BIG, Debuffs, HolderDefense, dv, effects, enemy_mask, holds, holds_any, in_circle,
                   neutral_defense, onehot_units, shield_grants, target_class)

UNENDING, KAENIC, PROTOPLASM, GA, STERAKS = 2502, 2504, 2525, 3026, 3053
SPIRIT, SUNFIRE, THORNMAIL, BRAMBLE, WARDENS = 3065, 3068, 3075, 3076, 3082
WARMOGS, HEARTSTEEL, BANSHEES, FROZEN_HEART, RANDUINS = 3083, 3084, 3102, 3110, 3143
HEXDRINKER, MAW, SPECTRES, FORCE_OF_NATURE, VERDANT = 3155, 3156, 3211, 4401, 4632
EDGE_OF_NIGHT, BAMIS, HOLLOW, JAKSHO, SHIELDBOW = 3814, 6660, 6664, 6665, 6673
ABYSSAL, QUICKSILVER, SEEKERS, ZHONYAS = 8020, 3140, 2420, 3157

LIFELINE = (STERAKS, HEXDRINKER, MAW, SHIELDBOW, PROTOPLASM)   # Seraph's lives in mage.py
ANNUL = (BANSHEES, VERDANT, EDGE_OF_NIGHT)
THORNS = (BRAMBLE, THORNMAIL)
IMMOLATE = (BAMIS, SUNFIRE, HOLLOW)

CHAMP_COMBAT_WINDOW = 5.0       # INFERRED (DAMAGE_AND_STATS §13 default)
KAENIC_DURATION = 1e9           # "until destroyed"
KAENIC_TAG = 1e8                # remaining-life threshold identifying the Kaenic slot
EPS = 1e-4

COVERAGE = {
    UNENDING: "Anguish: every 4 s in champion combat 3% bonus HP magic to enemy champions in 650, "
              "heal 250% of post-mitigation damage (pulse immediately on combat start, INFERRED)",
    KAENIC: "Magebane: magic shield 15% max HP after 15 s without magic damage; persists until broken",
    PROTOPLASM: "Lifeline (shared 90 s cd): +100-300 max HP 5 s via lifeline_bonus_health + stats, heal "
                "100-400 + 1.75 bonus armor/MR over 5 s, +10% MS / +25% tenacity while healing (size n/a)",
    GA: "Rebirth: lethal packet -> Effects.revive, 4 s stasis, 50% base HP, 100% max mana on completion; "
        "cd 300 s from revive end",
    STERAKS: "Claws +50% base AD (stats); Lifeline shield 60% bonus HP, 4.5 s, decay after 0.75 s hold",
    SPIRIT: "Boundless Vitality: incoming_heal +25% (stat; integrator applies to heals/regen/vamp/shields)",
    SUNFIRE: "Immolate 20 + 1.5% bonus HP per s, r325, x1.5 minions x1.8 monsters",
    THORNMAIL: "Thorns: on being struck by an enemy basic attack 20 + 10% bonus armor reactive magic; "
               "GW 3 s on champion attackers",
    BRAMBLE: "Thorns: 10 reactive magic; GW 3 s on champion attackers",
    WARDENS: "Rock Solid: -15 vs champion basic attacks (20% cap in core.damage)",
    WARMOGS: "Vitality +12% item HP (stats); Heart: >=2000 bonus HP, no champion damage 8 s / other 3 s -> "
             "1.5% max HP per 0.5 s (heal_plain)",
    HEARTSTEEL: "Colossal Consumption: 3 s near an enemy champion (700) charges it; next attack 70 + 6% max HP "
                "physical on-hit (life steal), +10% of it as permanent max HP, 30 s per target (Goliath size n/a)",
    BANSHEES: "Annul spell shield, 40 s cd restarted by champion damage while cooling",
    FROZEN_HEART: "Winter's Caress: 20% AS cripple on enemy champions within 700 (debuffs)",
    RANDUINS: "Resilience: critical basic attacks x0.7; Humility active in actives",
    HEXDRINKER: "Lifeline vs magic: magic shield 110-280 (ranged x0.75), 2.5 s, shared 90 s cd",
    MAW: "Lifeline vs magic: magic shield 200 + 1.5 bonus AD (ranged x0.75), 3 s; 10% omnivamp 5 s, "
         "extended 3 s by champion combat (INFERRED)",
    SPECTRES: "stats only: client HealthRegenPassive = 0 and no passive in the tooltip",
    FORCE_OF_NATURE: "Steadfast: stack per enemy champion magic source per 1 s, 8 max, 7 s refresh; at 8: "
                     "+70 MR, +6% MS (immobilize stacks need a CC event: not modelled)",
    VERDANT: "Annul spell shield, 60 s cd restarted by champion damage while cooling",
    EDGE_OF_NIGHT: "Annul spell shield, 40 s cd restarted by champion damage while cooling (lethality is a stat)",
    BAMIS: "Immolate 15 per s, r325, x1.5 minions x2.0 monsters",
    HOLLOW: "Immolate 15 + 1% bonus HP per s, r325, x1.25 minions/monsters; Desolate 2x tick r350 on "
            "non-champion kills, 4x r500 on champion takedowns within 3 s of damaging them",
    JAKSHO: "Voidborn Resilience: after 5 s of champion combat bonus armor/MR x1.3 until combat ends",
    SHIELDBOW: "Lifeline: shield 400 (+30/level from 9, ranged x0.8), 3 s, shared 90 s cd",
    ABYSSAL: "Unmake: enemy champions within 700 take +12% magic damage (one Unmake at a time)",
    QUICKSILVER: "no passive; Quicksilver cleanse active in actives",
    SEEKERS: "no passive; Time Stop stasis active in actives",
    ZHONYAS: "no passive; Time Stop stasis active in actives",
}


# ---- client calculation evaluator ------------------------------------------

def _effect_amount(item_id: int) -> list[float]:
    """mEffectAmount (kept in items_client.json, not exposed by the catalog)."""
    payload = json.loads(DATA_PATH.read_text())
    return list(payload["items"][str(item_id)]["effect_amount"])


def _stat(ctx, stat: int, formula: int | None, max_hp):
    mhp = ctx.max_hp if max_hp is None else max_hp
    table = {
        0: (0.0 * ctx.ap, ctx.ap),
        1: (ctx.base_armor, ctx.bonus_armor),
        2: (ctx.base_ad, ctx.bonus_ad),
        6: (ctx.base_mr, ctx.bonus_mr),
        12: (ctx.base_hp, mhp - ctx.base_hp),
    }
    if stat not in table:
        raise KeyError(f"unsupported calc stat {stat}")
    base, bonus = table[stat]
    return base if formula == 1 else bonus if formula == 2 else base + bonus


def _part(item_id: int, part: dict, ctx, max_hp):
    t = part["__type"]
    if t == "NumberCalculationPart":
        return part.get("mNumber", 0.0)
    if t == "NamedDataValueCalculationPart":
        return dv(item_id, part["mDataValue"])
    if t == "StatByCoefficientCalculationPart":
        return part.get("mCoefficient", 0.0) * _stat(ctx, part.get("mStat", 0), part.get("mStatFormula"), max_hp)
    if t == "StatByNamedDataValueCalculationPart":
        return dv(item_id, part["mDataValue"]) * _stat(ctx, part.get("mStat", 0), part.get("mStatFormula"), max_hp)
    if t == "ByCharLevelInterpolationCalculationPart":
        return lerp_level(part.get("mStartValue", 0.0), part.get("mEndValue", 0.0), ctx.level)
    if t == "ByCharLevelBreakpointsCalculationPart":
        lv = jnp.asarray(ctx.level, jnp.float32)
        v = part.get("mLevel1Value", 0.0) + 0.0 * lv
        for bp in part.get("mBreakpoints", []):
            k = bp.get("mLevel", 1)
            v = v + bp.get("mBonusPerLevelAtAndAfter", 0.0) * jnp.maximum(0.0, lv - k + 1.0)
            v = v + bp.get("mAdditionalBonusAtThisLevel", 0.0) * (lv >= k)
        return v
    if t == "{f3cbe7b2}":   # reference to another calculation of the same item
        return calc(item_id, part["mSpellCalculationKey"], ctx, max_hp=max_hp)
    raise KeyError(f"item {item_id}: unsupported calculation part {t}")


def calc(item_id: int, name: str, ctx, *, max_hp=None):
    """Evaluate a client GameCalculation for every holder (C,)."""
    c = catalog()[item_id].calculations[name]
    t = c["__type"]
    if t == "GameCalculationModified":
        v = calc(item_id, c["mModifiedGameCalculation"], ctx, max_hp=max_hp)
        v = v * _part(item_id, c["mMultiplier"], ctx, max_hp)
    else:
        v = 0.0 * ctx.level
        for p in c["mFormulaParts"]:
            v = v + _part(item_id, p, ctx, max_hp)
        if "mMultiplier" in c:
            v = v * _part(item_id, c["mMultiplier"], ctx, max_hp)
    if "mRangedMultiplier" in c:
        v = v * ranged_mult(ctx.is_ranged, _part(item_id, c["mRangedMultiplier"], ctx, max_hp))
    return jnp.asarray(v, jnp.float32)


# Sanity: the 20% Rock Solid cap is hard-coded in core.damage.
assert abs(dv(WARDENS, "WardenDamageMax") - 0.2) < 1e-6

_GA = _effect_amount(GA)        # [base-HP fraction, stasis s, cd, max-mana fraction, 0]
GA_HP, GA_DELAY, GA_MANA = _GA[0], _GA[1], _GA[3]
GA_COOLDOWN = dv(GA, "Cooldown")


class State(NamedTuple):
    lifeline_cd: Any        # (C,) shared LifeLineCooldown
    proto_start: Any        # (C,) Protoplasm trigger time
    proto_hp: Any           # (C,) granted max HP
    proto_rate: Any         # (C,) heal per second
    maw_until: Any          # (C,) Maw omnivamp expiry
    annul_ready: Any        # (C,) spell shield ready time
    ga_cd: Any              # (C,)
    ga_revive_at: Any       # (C,) pending revive completion (BIG = none)
    champ_last: Any         # (C,) last champion-combat event
    champ_start: Any        # (C,) start of current champion combat
    immo_until: Any         # (C,) aura expiry
    immo_next: Any          # (C,) next tick time
    ud_next: Any            # (C,) Unending Despair next pulse allowed
    kaenic_last_magic: Any  # (C,)
    kaenic_granted: Any     # (C,) bool
    kaenic_left: Any        # (C,) remaining Kaenic shield
    fon_stacks: Any         # (C,)
    fon_expire: Any         # (C,)
    fon_src_ready: Any      # (C, N) per-source stack ICD
    warmog_block: Any       # (C,) Warmog's Heart disabled until
    hs_charge: Any          # (C, N) seconds in range
    hs_last_in: Any         # (C, N)
    hs_cd: Any              # (C, N)
    hs_hp: Any              # (C,) permanent Heartsteel max HP
    dealt_t: Any            # (C, N) last time the holder damaged unit n


def init(n_champions: int, n_units: int) -> State:
    z = jnp.zeros((n_champions,), jnp.float32)
    zn = jnp.zeros((n_champions, n_units), jnp.float32)
    f = jnp.zeros((n_champions,), bool)
    never = z - BIG
    return State(
        lifeline_cd=never, proto_start=never, proto_hp=z, proto_rate=z, maw_until=never,
        annul_ready=never, ga_cd=never, ga_revive_at=z + BIG, champ_last=never, champ_start=never,
        immo_until=never, immo_next=z, ud_next=never, kaenic_last_magic=z, kaenic_granted=f,
        kaenic_left=z, fon_stacks=z, fon_expire=never, fon_src_ready=zn - BIG, warmog_block=never,
        hs_charge=zn, hs_last_in=zn - BIG, hs_cd=zn - BIG, hs_hp=z, dealt_t=zn - BIG)


# ---- shared helpers ----------------------------------------------------------

def _item_hp(own):
    col = jnp.asarray(catalog().arrays.stats[:, STAT_INDEX["health"]])
    return own.astype(jnp.float32) @ col


def _proto_active(state, ctx):
    return (ctx.now >= state.proto_start) & (ctx.now < state.proto_start + dv(PROTOPLASM, "Duration"))


def _vitality(own, ctx):
    return jnp.where(holds(own, WARMOGS), dv(WARMOGS, "HPAmp") * _item_hp(own), 0.0)


def _max_hp(state, own, ctx):
    """Max HP including this module's dynamic health."""
    return ctx.max_hp + _vitality(own, ctx) + state.hs_hp + jnp.where(_proto_active(state, ctx), state.proto_hp, 0.0)


def _in_champ_combat(state, ctx):
    return ctx.now - state.champ_last <= CHAMP_COMBAT_WINDOW


def _pick(own, table: dict, default=0.0):
    """Per-holder value selected by which item of ``table`` is held."""
    out = default
    for iid, v in table.items():
        out = jnp.where(holds(own, iid), v, out)
    return out


def _overlap(now, dt, start, end):
    return jnp.clip(jnp.minimum(now, end) - jnp.maximum(now - dt, start), 0.0, None)


# ---- hooks -------------------------------------------------------------------

def stats(state: State, own, ctx) -> ItemStats:
    now = ctx.now
    proto = holds(own, PROTOPLASM) & _proto_active(state, ctx)
    maw = holds(own, MAW) & (now < state.maw_until)
    fon = holds(own, FORCE_OF_NATURE) & (state.fon_stacks >= dv(FORCE_OF_NATURE, "MaxStacks")) \
        & (now < state.fon_expire)
    jak = holds(own, JAKSHO) & _in_champ_combat(state, ctx) \
        & (now - state.champ_start >= dv(JAKSHO, "MaxStacks") - EPS)
    jak_r = dv(JAKSHO, "BonusResistPercentage")
    return ItemStats(
        attack_damage=jnp.where(holds(own, STERAKS), calc(STERAKS, "BonusAD", ctx), 0.0),
        incoming_heal=jnp.where(holds(own, SPIRIT), dv(SPIRIT, "HealingIncrease"), 0.0),
        health=_vitality(own, ctx) + state.hs_hp + jnp.where(proto, state.proto_hp, 0.0),
        percent_move_speed=jnp.where(proto, dv(PROTOPLASM, "MSAmount"), 0.0)
        + jnp.where(fon, dv(FORCE_OF_NATURE, "MoveSpeed"), 0.0),
        tenacity=jnp.where(proto, dv(PROTOPLASM, "TenacityAmount"), 0.0),
        omnivamp=jnp.where(maw, dv(MAW, "BuffVamp"), 0.0),
        magic_resist=jnp.where(fon, dv(FORCE_OF_NATURE, "BonusMagicResist"), 0.0)
        + jnp.where(jak, jak_r * ctx.bonus_mr, 0.0),
        armor=jnp.where(jak, jak_r * ctx.bonus_armor, 0.0))


def _lifeline_ready(state, own, ctx):
    return holds_any(own, LIFELINE) & (ctx.now >= state.lifeline_cd) & ctx.alive


def _annul_ready(state, own, ctx):
    return holds_any(own, ANNUL) & (ctx.now >= state.annul_ready) & ctx.alive


def defense(state: State, own, ctx) -> HolderDefense:
    c = ctx.level.shape[0]
    mhp = _max_hp(state, own, ctx)
    ready = _lifeline_ready(state, own, ctx)
    hex_shield = jnp.where(ctx.is_ranged, calc(HEXDRINKER, "RangedItemCalcValue", ctx),
                           calc(HEXDRINKER, "MeleeItemCalcValue", ctx))
    maw_shield = jnp.where(ctx.is_ranged, calc(MAW, "RangedItemCalcValue", ctx), calc(MAW, "MeleeItemCalcValue", ctx))
    shield = _pick(own, {STERAKS: calc(STERAKS, "ShieldSize", ctx, max_hp=mhp), HEXDRINKER: hex_shield,
                         MAW: maw_shield, SHIELDBOW: calc(SHIELDBOW, "ShieldAmount", ctx), PROTOPLASM: 0.0})
    kind = _pick(own, {HEXDRINKER: SHIELD_MAGIC, MAW: SHIELD_MAGIC}, SHIELD_ALL).astype(jnp.int32)
    duration = _pick(own, {STERAKS: dv(STERAKS, "ShieldDuration"), HEXDRINKER: dv(HEXDRINKER, "ShieldLifetime"),
                           MAW: dv(MAW, "ShieldDuration"), SHIELDBOW: dv(SHIELDBOW, "ShieldDuration")})
    hold = _pick(own, {STERAKS: dv(STERAKS, "TimeBeforeDecay")}, jnp.inf)
    bonus_hp = jnp.where(holds(own, PROTOPLASM), calc(PROTOPLASM, "MaxHealthGain", ctx), 0.0)
    magic_only = holds(own, HEXDRINKER) | holds(own, MAW)
    base = neutral_defense(c)
    f = lambda v: jnp.broadcast_to(jnp.asarray(v, jnp.float32), (c,))
    return base._replace(
        crit_taken_mult=f(jnp.where(holds(own, RANDUINS), 1.0 - dv(RANDUINS, "PercentCritDamageReduction"), 1.0)),
        champion_attack_block=f(jnp.where(holds(own, WARDENS), dv(WARDENS, "BlockBase"), 0.0)),
        lifeline_ready=ready, lifeline_magic_only=magic_only, lifeline_shield=f(shield),
        lifeline_shield_kind=jnp.broadcast_to(kind, (c,)), lifeline_duration=f(duration),
        lifeline_decay_hold=f(hold), lifeline_bonus_health=f(bonus_hp),
        spell_shield=_annul_ready(state, own, ctx))


def debuffs(state: State, own, ctx, units) -> Debuffs:
    n = units.x.shape[0]
    champs = enemy_mask(ctx, units) & (units.cls[None, :] == CLASS_CHAMPION) & ctx.alive[:, None]
    fh = champs & holds(own, FROZEN_HEART)[:, None] \
        & in_circle(units, ctx.x, ctx.y, jnp.full(ctx.x.shape, dv(FROZEN_HEART, "AuraRadius")))
    am = champs & holds(own, ABYSSAL)[:, None] \
        & in_circle(units, ctx.x, ctx.y, jnp.full(ctx.x.shape, dv(ABYSSAL, "Radius")))
    z = jnp.zeros((n,), jnp.float32)
    cripple = jnp.max(jnp.where(fh, abs(dv(FROZEN_HEART, "ASPDSlow")), 0.0), axis=0)
    unmake = jnp.max(jnp.where(am, dv(ABYSSAL, "DamageAmp"), 0.0), axis=0)   # one Unmake at a time
    return Debuffs(z, z, z, z, z, unmake.astype(jnp.float32), cripple.astype(jnp.float32))


def on_hit(state: State, own, ctx, units, attack):
    c, n = ctx.level.shape[0], units.x.shape[0]
    tgt = onehot_units(attack.target, n)
    charged = jnp.sum(jnp.where(tgt, state.hs_charge, 0.0), axis=1)
    cd = jnp.sum(jnp.where(tgt, state.hs_cd, 0.0), axis=1)
    demolish = dv(HEARTSTEEL, "TrackerTickRate") * dv(HEARTSTEEL, "NumTicksToTrigger")
    go = attack.hit & ctx.alive & holds(own, HEARTSTEEL) & (target_class(units, attack.target) == CLASS_CHAMPION) \
        & (attack.target >= 0) & (charged >= demolish - EPS) & (ctx.now >= cd)
    mhp = _max_hp(state, own, ctx)
    dmg = calc(HEARTSTEEL, "DamageProcCalc", ctx, max_hp=mhp)
    gain = calc(HEARTSTEEL, "ProcHealthGain", ctx, max_hp=mhp)
    p = packets(go, ctx.unit, jnp.maximum(attack.target, 0), dmg, PHYSICAL, ON_HIT_ITEM | PROP_LIFESTEAL,
                item=HEARTSTEEL)
    hit = tgt & go[:, None]
    state = state._replace(
        hs_hp=state.hs_hp + jnp.where(go, gain, 0.0),
        hs_charge=jnp.where(hit, 0.0, state.hs_charge),
        hs_cd=jnp.where(hit, ctx.now + dv(HEARTSTEEL, "PerTargetCooldown"), state.hs_cd))
    return state, effects(c, n, packets=p)


def on_damage(state: State, own, ctx, units, report):
    c, n = ctx.level.shape[0], units.x.shape[0]
    now = ctx.now
    p, r = report.packets, report.resolved
    nn = units.x.shape[0]
    src_i, dst_i = jnp.clip(p.src, 0, nn - 1), jnp.clip(p.dst, 0, nn - 1)
    src_cls, dst_cls = units.cls[src_i], units.cls[dst_i]
    to_h = p.valid[None, :] & (p.dst[None, :] == ctx.unit[:, None])          # (C, P)
    from_h = p.valid[None, :] & (p.src[None, :] == ctx.unit[:, None])
    enemy_src = units.team[src_i][None, :] != ctx.team[:, None]
    enemy_dst = units.team[dst_i][None, :] != ctx.team[:, None]
    dmg = (r.final > 0.0)[None, :]
    champ_src = (src_cls == CLASS_CHAMPION)[None, :]
    champ_dst = (dst_cls == CLASS_CHAMPION)[None, :]
    magic = (p.dtype == MAGIC)[None, :]
    taken = to_h & enemy_src & dmg
    dealt = from_h & enemy_dst & dmg
    took_champ = jnp.any(taken & champ_src, axis=1)
    took_other = jnp.any(taken & ~champ_src, axis=1)

    # Champion combat bookkeeping (Jak'Sho, Unending Despair, Maw).
    ev = jnp.any((to_h & enemy_src & champ_src) | (from_h & enemy_dst & champ_dst), axis=1)
    fresh = ev & (now - state.champ_last > CHAMP_COMBAT_WINDOW)
    champ_start = jnp.where(fresh, now, state.champ_start)
    champ_last = jnp.where(ev, now, state.champ_last)
    onehot_dst = (p.dst[:, None] == jnp.arange(n)[None, :]).astype(jnp.float32)   # (P, N)
    onehot_src = (p.src[:, None] == jnp.arange(n)[None, :]).astype(jnp.float32)
    dealt_to = (dealt.astype(jnp.float32) @ onehot_dst) > 0.0
    dealt_t = jnp.where(dealt_to, now, state.dealt_t)

    # Lifeline (one cooldown per champion).
    ll = r.lifeline_fired[ctx.unit] & _lifeline_ready(state, own, ctx)
    lifeline_cd = jnp.where(ll, now + _pick(own, {i: dv(i, "Cooldown") for i in LIFELINE}), state.lifeline_cd)
    proto = ll & holds(own, PROTOPLASM)
    pdur = dv(PROTOPLASM, "Duration")
    proto_start = jnp.where(proto, now, state.proto_start)
    proto_hp = jnp.where(proto, calc(PROTOPLASM, "MaxHealthGain", ctx), state.proto_hp)
    proto_rate = jnp.where(proto, calc(PROTOPLASM, "TotalHealthRegen", ctx) / pdur, state.proto_rate)
    maw_until = jnp.where(ll & holds(own, MAW), now + dv(MAW, "BuffDuration"), state.maw_until)
    maw_until = jnp.where(ev & (now < maw_until), jnp.maximum(maw_until, now + dv(MAW, "BuffExtension")), maw_until)

    # Annul: pop starts the cooldown; champion damage restarts it while cooling.
    popped = r.spell_shield_popped[ctx.unit] & _annul_ready(state, own, ctx)
    cooling = holds_any(own, ANNUL) & (now < state.annul_ready)
    acd = _pick(own, {i: dv(i, "Cooldown") for i in ANNUL})
    annul_ready = jnp.where(popped | (cooling & took_champ), now + acd, state.annul_ready)

    # Guardian Angel.
    lethal = jnp.any(to_h & r.killed[None, :], axis=1)
    revive = lethal & holds(own, GA) & (now >= state.ga_cd) & ctx.alive
    ga_cd = jnp.where(revive, now + GA_DELAY + GA_COOLDOWN, state.ga_cd)
    ga_revive_at = jnp.where(revive, now + GA_DELAY, state.ga_revive_at)

    # Immolate activation (its own damage does not refresh it).
    own_immo = jnp.isin(p.item, jnp.asarray(IMMOLATE))[None, :] & from_h
    trig = jnp.any((taken | dealt) & ~own_immo, axis=1) & holds_any(own, IMMOLATE) & ctx.alive
    active = now <= state.immo_until
    immo_next = jnp.where(trig & ~active, now + 1.0 / dv(SUNFIRE, "TicksPerSecond"), state.immo_next)
    immo_until = jnp.where(trig, now + dv(SUNFIRE, "AuraDuration"), state.immo_until)

    # Unending Despair drain heal: 250% of post-mitigation damage of its packets.
    ud_dmg = jnp.sum(jnp.where(from_h & (p.item == UNENDING)[None, :], r.final[None, :], 0.0), axis=1)
    heal = jnp.where(holds(own, UNENDING), dv(UNENDING, "HealMultiplier") * ud_dmg, 0.0)

    # Kaenic: magic damage resets the timer; read the remaining Kaenic shield.
    magic_taken = jnp.any(to_h & magic & dmg, axis=1)
    kaenic_last = jnp.where(magic_taken, now, state.kaenic_last_magic)
    kaenic_granted = state.kaenic_granted & ~magic_taken
    sh = r.shields
    val = shield_value(sh, now)[ctx.unit]                                      # (C, K)
    kslot = (sh.kind[ctx.unit] == SHIELD_MAGIC) & (sh.expires_at[ctx.unit] - now > KAENIC_TAG)
    kaenic_left = jnp.sum(jnp.where(kslot, val, 0.0), axis=1)

    # Force of Nature stacks.
    fon_hits = ((taken & magic & champ_src).astype(jnp.float32) @ onehot_src) > 0.0   # (C, N)
    eligible = fon_hits & (now >= state.fon_src_ready) & holds(own, FORCE_OF_NATURE)[:, None]
    gained = jnp.sum(eligible, axis=1).astype(jnp.float32)
    expired = now >= state.fon_expire
    stacks = jnp.minimum(jnp.where(expired, 0.0, state.fon_stacks) + gained, dv(FORCE_OF_NATURE, "MaxStacks"))
    dealt_champ = jnp.any(dealt & champ_dst, axis=1)
    refresh = (gained > 0) | (dealt_champ & (stacks > 0))
    fon_expire = jnp.where(refresh, now + dv(FORCE_OF_NATURE, "BuffDuration"), state.fon_expire)
    fon_src_ready = jnp.where(eligible, now + dv(FORCE_OF_NATURE, "StackRefreshTimer"), state.fon_src_ready)

    # Warmog's Heart disable timers.
    warmog_block = jnp.maximum(state.warmog_block, jnp.maximum(
        jnp.where(took_champ, now + dv(WARMOGS, "OOCTimerChampion"), -BIG),
        jnp.where(took_other, now + dv(WARMOGS, "OOCTimer"), -BIG)))

    # Thorns: enemy basic attacks that struck the holder.
    struck = to_h & enemy_src & has(p.flags, TAG_BASIC_ATTACK)[None, :] & ~has(p.flags, PROP_REACTIVE)[None, :] \
        & (holds_any(own, THORNS) & ctx.alive)[:, None]
    thorn_dmg = jnp.where(holds(own, THORNMAIL), calc(THORNMAIL, "TotalDamage", ctx), calc(BRAMBLE, "TotalDamage", ctx))
    thorn_item = jnp.where(holds(own, THORNMAIL), THORNMAIL, BRAMBLE)
    p_thorns = packets(struck, ctx.unit[:, None], p.src[None, :], thorn_dmg[:, None], MAGIC,
                       PROP_REACTIVE | TAG_ITEM | TAG_PROC, item=thorn_item[:, None])
    gw_src = jnp.any(struck & champ_src, axis=0)                               # (P,)
    gw_dur = dv(THORNMAIL, "GrievousDuration")
    grievous = jnp.zeros((n,), jnp.float32).at[src_i].max(jnp.where(gw_src, gw_dur, 0.0))

    state = state._replace(
        lifeline_cd=lifeline_cd, proto_start=proto_start, proto_hp=proto_hp, proto_rate=proto_rate,
        maw_until=maw_until, annul_ready=annul_ready, ga_cd=ga_cd, ga_revive_at=ga_revive_at,
        champ_last=champ_last, champ_start=champ_start, immo_until=immo_until, immo_next=immo_next,
        kaenic_last_magic=kaenic_last, kaenic_granted=kaenic_granted, kaenic_left=kaenic_left,
        fon_stacks=stacks, fon_expire=fon_expire, fon_src_ready=fon_src_ready, warmog_block=warmog_block,
        dealt_t=dealt_t)
    eff = effects(c, n, packets=p_thorns, heal=heal, grievous=grievous, revive=revive,
                  revive_delay=jnp.where(revive, GA_DELAY, 0.0),
                  revive_hp=jnp.where(revive, GA_HP * ctx.base_hp, 0.0))
    return state, eff


_IMMO = {   # item -> (minion mult, monster mult)
    BAMIS: (1.0 + dv(BAMIS, "MinionMod"), 1.0 + dv(BAMIS, "MonsterMod")),
    SUNFIRE: (1.0 + dv(SUNFIRE, "MinionMod"), 1.0 + dv(SUNFIRE, "MonsterMod")),
    HOLLOW: (1.0 + dv(HOLLOW, "MinionMod"), 1.0 + dv(HOLLOW, "MonsterMod")),
}


def _immolate_dpt(own, ctx, mhp):
    return _pick(own, {i: calc(i, "DamagePerTick", ctx, max_hp=mhp) for i in IMMOLATE})


def periodic(state: State, own, ctx, units):
    c, n = ctx.level.shape[0], units.x.shape[0]
    now, dt = ctx.now, ctx.dt
    alive = ctx.alive
    mhp = _max_hp(state, own, ctx)
    full = lambda v: jnp.full((c,), v, jnp.float32)
    enemies = enemy_mask(ctx, units) & alive[:, None]
    cls = units.cls[None, :]

    # Protoplasm heal over time.
    pdur = dv(PROTOPLASM, "Duration")
    heal = jnp.where(holds(own, PROTOPLASM) & alive,
                     state.proto_rate * _overlap(now, dt, state.proto_start, state.proto_start + pdur), 0.0)

    # Guardian Angel revive completion: restore mana.
    done = now >= state.ga_revive_at
    mana = jnp.where(done, GA_MANA * ctx.max_mana, 0.0)
    ga_revive_at = jnp.where(done, BIG, state.ga_revive_at)

    # Immolate ticks.
    period = 1.0 / dv(SUNFIRE, "TicksPerSecond")
    last = jnp.minimum(now, state.immo_until)
    due = (state.immo_next <= last + EPS) & holds_any(own, IMMOLATE) & alive
    n_due = jnp.where(due, jnp.floor((last - state.immo_next) / period + EPS) + 1.0, 0.0)
    immo_next = state.immo_next + n_due * period
    dpt = _immolate_dpt(own, ctx, mhp)
    minion_m = _pick(own, {i: v[0] for i, v in _IMMO.items()}, 1.0)
    monster_m = _pick(own, {i: v[1] for i, v in _IMMO.items()}, 1.0)
    mult = jnp.where(cls == CLASS_MINION, minion_m[:, None], jnp.where(cls == CLASS_MONSTER, monster_m[:, None], 1.0))
    near = enemies & (cls != CLASS_STRUCTURE) & in_circle(units, ctx.x, ctx.y, full(dv(SUNFIRE, "Range"))) \
        & (n_due > 0)[:, None]
    immo_item = _pick(own, {i: i for i in IMMOLATE}, 0).astype(jnp.int32)
    p_immo = packets(near, ctx.unit[:, None], jnp.arange(n)[None, :], (dpt * n_due)[:, None] * mult, MAGIC,
                     TAG_AOE | TAG_PERIODIC | TAG_ITEM, item=immo_item[:, None])
    immo_until = jnp.where(alive, state.immo_until, -BIG)

    # Unending Despair pulse.
    pulse = holds(own, UNENDING) & alive & _in_champ_combat(state, ctx) & (now >= state.ud_next)
    ud_t = enemies & (cls == CLASS_CHAMPION) & in_circle(units, ctx.x, ctx.y, full(dv(UNENDING, "DrainRange"))) \
        & pulse[:, None]
    p_ud = packets(ud_t, ctx.unit[:, None], jnp.arange(n)[None, :],
                   calc(UNENDING, "DrainCalc", ctx, max_hp=mhp)[:, None], MAGIC, TAG_AOE | TAG_ITEM, item=UNENDING)
    ud_next = jnp.where(pulse, now + dv(UNENDING, "Cooldown"), state.ud_next)

    # Kaenic Rookern grant.
    target = calc(KAENIC, "ShieldCalc", ctx, max_hp=mhp)
    kgo = holds(own, KAENIC) & alive & ~state.kaenic_granted \
        & (now - state.kaenic_last_magic >= dv(KAENIC, "OutOfCombatDuration") - EPS)
    kamount = jnp.where(kgo, jnp.maximum(target - state.kaenic_left, 0.0), 0.0)
    kaenic_granted = (state.kaenic_granted | kgo) & alive & holds(own, KAENIC)
    kaenic_left = jnp.where(kgo, jnp.maximum(target, state.kaenic_left), jnp.where(alive, state.kaenic_left, 0.0))
    shields = shield_grants(kamount, SHIELD_MAGIC, KAENIC_DURATION)

    # Warmog's Heart (0.5 s game-clock boundaries).
    step = dv(WARMOGS, "SecondsPerHeal")
    ticks = jnp.floor(now / step + EPS) - jnp.floor((now - dt) / step + EPS)
    wgo = holds(own, WARMOGS) & alive & (mhp - ctx.base_hp >= dv(WARMOGS, "HealthThreshold")) \
        & (now >= state.warmog_block)
    heal_plain = jnp.where(wgo, ticks * calc(WARMOGS, "TotalHealing", ctx, max_hp=mhp), 0.0)

    # Heartsteel charge.
    hs_in = enemies & (cls == CLASS_CHAMPION) & holds(own, HEARTSTEEL)[:, None] \
        & in_circle(units, ctx.x, ctx.y, full(dv(HEARTSTEEL, "DistanceToChampion")))
    demolish = dv(HEARTSTEEL, "TrackerTickRate") * dv(HEARTSTEEL, "NumTicksToTrigger")
    stale = now - state.hs_last_in > dv(HEARTSTEEL, "RangeTrackingBuffDuration")
    hs_charge = jnp.where(hs_in, jnp.minimum(state.hs_charge + dt, demolish), jnp.where(stale, 0.0, state.hs_charge))
    hs_last_in = jnp.where(hs_in, now, state.hs_last_in)

    # Death clears buffs (FoN, Kaenic tracking above).
    fon_stacks = jnp.where(alive & (now < state.fon_expire), state.fon_stacks, 0.0)

    state = state._replace(ga_revive_at=ga_revive_at, immo_next=immo_next, immo_until=immo_until, ud_next=ud_next,
                           kaenic_granted=kaenic_granted, kaenic_left=kaenic_left, hs_charge=hs_charge,
                           hs_last_in=hs_last_in, fon_stacks=fon_stacks)
    return state, effects(c, n, packets=concat_packets(p_immo, p_ud), heal=heal, heal_plain=heal_plain,
                          mana=mana, shields=shields)


def on_takedown(state: State, own, ctx, units, kills):
    """Hollow Radiance Desolate."""
    c, n = ctx.level.shape[0], units.x.shape[0]
    go = holds(own, HOLLOW) & ctx.alive
    ku = kills.killed_units & go[:, None]
    vcls = units.cls[None, :]
    small = ku & (vcls != CLASS_CHAMPION) & (vcls != CLASS_STRUCTURE)
    champ = ku & (vcls == CLASS_CHAMPION) & (ctx.now - state.dealt_t <= dv(HOLLOW, "TakedownWindow") + EPS)
    d = jnp.sqrt((units.x[:, None] - units.x[None, :]) ** 2 + (units.y[:, None] - units.y[None, :]) ** 2)  # (V, N)
    in_small = (d <= dv(HOLLOW, "ProcAoE") + units.radius[None, :]).astype(jnp.float32)
    in_big = (d <= dv(HOLLOW, "ChampProcAoE") + units.radius[None, :]).astype(jnp.float32)
    weight = dv(HOLLOW, "ProcDPSMultiplier") * (small.astype(jnp.float32) @ in_small) \
        + dv(HOLLOW, "ChampProcDPSMultiplier") * (champ.astype(jnp.float32) @ in_big)   # (C, N)
    dpt = calc(HOLLOW, "DamagePerTick", ctx, max_hp=_max_hp(state, own, ctx))
    mask = enemy_mask(ctx, units) & (vcls != CLASS_STRUCTURE) & (weight > 0.0)
    p = packets(mask, ctx.unit[:, None], jnp.arange(n)[None, :], dpt[:, None] * weight, MAGIC,
                TAG_AOE | TAG_PROC | TAG_ITEM, item=HOLLOW)
    return state, effects(c, n, packets=p)
