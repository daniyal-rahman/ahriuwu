"""Fighter / AD-bruiser item passives for 26.19 (ITEMS.md §6.4, §9.3; ITEMS_CATALOG).

Values come from the 16.19.8230722 item data: ``dv`` for mDataValues and
``_part`` for numbers that only exist inside client calculations (Endless
Hunger Famine, Death's Dance split, Wit's End, Terminus, Hullbreaker,
Dead Man's Plate). Behaviour that the data does not encode follows the
ITEMS.md / wiki text and is marked with a short citation.

Conventions used throughout:
  * "ability damage" = packets tagged ``TAG_ACTIVE_SPELL`` without ``TAG_ITEM``
    (item actives carry both tags).
  * "dealt damage" triggers read the tick's ``Report`` and require
    post-mitigation ``final > 0`` (shielded damage still counts, ITEMS §6.4).
  * one damage *instance* per (holder, target) per tick stands in for cast
    instances (no cast ids in the packet stream).
  * item on-hit damage (BotRK, Wit's End, Terminus, Dead Man's Plate) does not
    apply to structures (WIKI on-hit, INFERRED M); Hullbreaker's Skipper has
    an explicit structure value and does.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from ...core.damage import (CLASS_CHAMPION, CLASS_MINION, CLASS_MONSTER, CLASS_STRUCTURE, MAGIC, ON_HIT_ITEM,
                            PHYSICAL, PROP_LIFESTEAL, PROP_NO_DAMAGE_MOD, PROP_NO_OMNIVAMP, TAG_ACTIVE_SPELL,
                            TAG_BASIC_ATTACK, TAG_ITEM, TAG_PERIODIC, TAG_PROC, TRUE, concat_packets, has,
                            packets)
from ..catalog import ItemStats, catalog
from .core import (AttackMods, dst_class, dv, effects, holds, merge_effects, neutral_debuffs, neutral_defense,
                   onehot_units, shield_grants, target_class)

OVERLORD, HUNGER, BASTION, CLEAVER, HEXPLATE = 2501, 2517, 2520, 3071, 3073
BOTRK, SHOJIN, HULL, DMP, DEATHS_DANCE = 3153, 3161, 3181, 3742, 6333
CHEMPUNK, SUNDERED, ECLIPSE, EXECUTIONER, MORTAL = 6609, 6610, 6692, 3123, 3033
SERYLDA, STEEL_SIGIL, BRUTALIZER, DIRK, WITS, TERMINUS, MERCURIAL = (
    6694, 2019, 2020, 3134, 3091, 3302, 3139)
GW_ITEMS = (EXECUTIONER, MORTAL, CHEMPUNK)

COVERAGE = {
    OVERLORD: "Tyranny 2.5% bonus HP -> AD; Retribution up to 12% of other-source AD by missing HP (full at 70% missing)",
    HUNGER: "Famine AH 5 + 13% bAD (ranged 10%); Feast +15% omnivamp 8 s on champion takedown damaged within 3 s",
    BASTION: "Shaped Charge 50+1.5 leth true (ranged x0.5) on next ability damage to a champion, cd 20; "
             "Sabotage 90 s after takedown -> next attack vs structure 300+25 leth true over 3 s (ranged x0.8). "
             "Epic-monster targets not modelled (no epic flag)",
    CLEAVER: "Carve 6%/stack armor reduction, 5 stacks, 6 s, per target (non-basic ICD 0.01 s); Fervor +20 MS (ranged 10) 2 s",
    HEXPLATE: "Hexcharged +30 ultimate haste; Overdrive +50% AS / +20% MS (ranged 35/14) 8 s on ult cast, cd 30",
    BOTRK: "Mist's Edge 9% (ranged 6%) target current HP physical on-hit, LS, cap 100 vs minions/monsters; "
           "Clawing Shadows 3 hits on a champion in 6 s -> 30% (ranged 15%) slow 1 s, cd 15",
    SHOJIN: "Dragonforce +25 basic ability haste; Focused Will stacks (4, 6 s, 1/s per cast instance). The +3%/stack "
            "(ranged 1.5%) ability-damage amp is exposed via shojin_ability_amp(); needs a framework ability-amp hook",
    HULL: "Skipper stacks (10 s, any target); 5th attack vs champion 1.2 bAD+5% maxHP / structure 3 bAD+10% maxHP "
          "(ranged x0.7), LS; Boarding Party via boarding_party_resists() helper",
    DMP: "Shipwrecker momentum 100 stacks over 4 s moving (client DurationToMaxStack), +20 flat MS at 100; "
         "next attack discharges s/100*(1.0 bAD + 40) physical, no LS",
    DEATHS_DANCE: "Ignore Pain store 30% (ranged 10%) via defense().store_fraction, bled as true damage over 3 s; "
                  "Defy: takedown within 3 s cleanses the pool and heals 0.75 bAD over 2 s",
    CHEMPUNK: "Hackshorn: physical damage to enemy champions -> Grievous Wounds 3 s",
    SUNDERED: "Lightshield Strike: first attack vs a champion per 10 s per target crits at 0.8x crit bonus, heals "
              "0.9 bAD (ranged x0.5) + 4% missing HP; overheal -> bonus health 8 s",
    ECLIPSE: "Ever Rising Moon: 2 separate damage instances on a champion within 2 s -> 8% (ranged 5%) max HP physical "
             "+ 150 + 0.4 bAD shield (ranged x0.5) 2 s, cd 6",
    EXECUTIONER: "Grievous Wounds on physical damage to enemy champions, 3 s",
    MORTAL: "Grievous Wounds on physical damage to enemy champions, 3 s (30% armor pen is static)",
    SERYLDA: "Bitter Cold: ability damage to enemies at <= 50% HP slows 30% for 1 s (35% armor pen static)",
    STEEL_SIGIL: "stats only (FlatDR=3 data value has no tooltip/passive in 26.19; not modelled)",
    BRUTALIZER: "stats only (5 lethality in static stats)",
    DIRK: "stats only (10 lethality in static stats)",
    WITS: "Fray: 45 bonus magic on-hit, LS",
    TERMINUS: "Shadow 30 + 10% bAD + 10% AP magic on-hit, LS; Juxtaposition alternating Light (+6/7/8 armor & MR) / "
              "Dark (+10% armor & magic pen) stacks vs champions, 3 each, 5 s",
    MERCURIAL: "stats only; Quicksilver active in actives",
}


# ---- client calculation readers ---------------------------------------------

def _calc(item_id: int, name: str) -> dict:
    return catalog()[item_id].calculations[name]


def _part(item_id: int, name: str, i: int, key: str = "mNumber") -> float:
    return float(_calc(item_id, name)["mFormulaParts"][i][key])


# Overlord's Bloodmail
TYRANNY = dv(OVERLORD, "HPToADPercentage")
RETRIBUTION = dv(OVERLORD, "MissingHealthAD")
RETRIBUTION_FULL = dv(OVERLORD, "MissingHealthThreshold")
# Endless Hunger (calc hashes {e4d9f16b} melee / {87892572} ranged, matched to the wiki 13%/10%)
FAMINE_BASE = _part(HUNGER, "{e4d9f16b}", 0)
FAMINE_MELEE = _part(HUNGER, "{e4d9f16b}", 1, "mCoefficient")
FAMINE_RANGED = _part(HUNGER, "{87892572}", 1, "mCoefficient")
TAKEDOWN_WINDOW = dv(HUNGER, "TakedownWindow")
# Bastionbreaker. The calcs' ranged multipliers point at unresolved hashes; the
# data values RangeModifier/AbilityDamageRangeMod match the wiki 0.8 / 0.5.
SHAPED_BASE = _part(BASTION, "AbilityDamageCalc", 0)
SHAPED_LETH = _part(BASTION, "AbilityDamageCalc", 1, "mCoefficient")
SABOTAGE_BASE = _part(BASTION, "DamageCalc", 0)
SABOTAGE_LETH = _part(BASTION, "DamageCalc", 1, "mCoefficient")
# Black Cleaver
CARVE_PER_STACK = dv(CLEAVER, "ShredPerStack")
CARVE_MAX = dv(CLEAVER, "MaxStacks")
CARVE_DURATION = dv(CLEAVER, "DebuffDuration")
CARVE_ICD = dv(CLEAVER, "InternalCD")
# Death's Dance store split (MeleeItemCalcValue / RangedItemCalcValue)
DD_MELEE = _part(DEATHS_DANCE, "MeleeItemCalcValue", 0)
DD_RANGED = _part(DEATHS_DANCE, "RangedItemCalcValue", 0)
DD_BLEED = dv(DEATHS_DANCE, "BleedDurationWorst")
DD_BUCKET = 0.25                       # merge stored chunks within 0.25 s (timing error < 0.25 s)
DD_SLOTS = int(round(DD_BLEED / DD_BUCKET)) + 1
# Hullbreaker
HULL_STACKS = 5                        # "every fifth Attack" (tooltip)
HULL_RANGED = float(_calc(HULL, "MaxStackDamage")["mRangedMultiplier"]["mNumber"])
_bp = _calc(HULL, "BonusMinionResists")
BOARDING_L1 = float(_bp["mFormulaParts"][0]["mLevel1Value"])
BOARDING_PER = float(_bp["mFormulaParts"][0]["mBreakpoints"][0]["mBonusPerLevelAtAndAfter"])
BOARDING_FROM = float(_bp["mFormulaParts"][0]["mBreakpoints"][0]["mLevel"])
BOARDING_RANGED = float(_bp["mRangedMultiplier"]["mNumber"])
BOARDING_PARTY_RANGE = 1050.0          # WIKI "within 1050 [Estimated]"
# Dead Man's Plate
DMP_MAX = dv(DMP, "MaxStacks")
DMP_RATE = DMP_MAX / dv(DMP, "DurationToMaxStack")   # stacks per second moving (client 4 s; wiki 3.75 s)
DMP_FLAT_FULL = dv(DMP, "BonusDamagePerStack") * float(
    _calc(DMP, "MaxDamageCalc")["mFormulaParts"][1]["mPart2"]["mNumber"])
# Wit's End / Terminus
WITS_DAMAGE = _part(WITS, "OnHitDamage", 0)
TERM_BASE = _part(TERMINUS, "OnHitDamage", 0)
TERM_BAD = _part(TERMINUS, "OnHitDamage", 1, "mCoefficient")
TERM_AP = _part(TERMINUS, "OnHitDamage", 2, "mCoefficient")
_tp = _calc(TERMINUS, "ARMRPerHitScaling")["mFormulaParts"][0]
TERM_RES_L1 = float(_tp["mLevel1Value"])
TERM_RES_STEPS = tuple((float(b["mLevel"]), float(b["mAdditionalBonusAtThisLevel"])) for b in _tp["mBreakpoints"])
TERM_LIGHT_MAX = float(_calc(TERMINUS, "ARMRMaxScaling")["mMultiplier"]["mNumber"])
TERM_DARK_PER = dv(TERMINUS, "PenPerHit")
TERM_DARK_MAX = round(dv(TERMINUS, "PenMax") / TERM_DARK_PER)

NEG = -1e9


class State(NamedTuple):
    last_dmg: Any        # (C, N) last time holder damaged champion n (takedown windows)
    carve: Any           # (C, N) Carve stacks
    carve_until: Any     # (C, N)
    carve_last: Any      # (C, N) last non-basic stack (ICD)
    fervor_until: Any    # (C,)
    feast_until: Any     # (C,)
    hex_until: Any       # (C,)
    hex_cd: Any          # (C,)
    botrk_count: Any     # (C, N)
    botrk_until: Any     # (C, N)
    botrk_cd: Any        # (C,)
    shojin: Any          # (C,) Focused Will stacks
    shojin_until: Any    # (C,)
    shojin_last: Any     # (C,) last stack time
    shojin_fresh: Any    # (C,) bool: a cast started since the last stack
    hull: Any            # (C,) Skipper stacks
    hull_until: Any      # (C,)
    dmp: Any             # (C,) momentum 0..100
    dd_amt: Any          # (C, K) remaining stored damage per bucket
    dd_rate: Any         # (C, K) bleed rate per bucket (per second)
    dd_bucket: Any       # (C, K) int32 bucket id owning the slot
    dd_heal_rate: Any    # (C,)
    dd_heal_until: Any   # (C,)
    sky_cd: Any          # (C, N) per-target Lightshield Strike cooldown end
    sky_bonus: Any       # (C,) temporary bonus health
    sky_bonus_until: Any  # (C,)
    ecl_first: Any       # (C, N) time of the first stack in the open window
    ecl_cd: Any          # (C,)
    bb_cd: Any           # (C,) Shaped Charge cooldown end
    sabotage_until: Any  # (C,)
    bb_dot_target: Any   # (C,) int32
    bb_dot_rate: Any     # (C,)
    bb_dot_until: Any    # (C,)
    term_light: Any      # (C,)
    term_light_until: Any
    term_dark: Any
    term_dark_until: Any
    term_next_dark: Any  # (C,) bool


def init(n_champions: int, n_units: int) -> State:
    c, n = n_champions, n_units
    z, zn = jnp.zeros((c,), jnp.float32), jnp.zeros((c, n), jnp.float32)
    zk = jnp.zeros((c, DD_SLOTS), jnp.float32)
    f = jnp.zeros((c,), bool)
    return State(zn + NEG, zn, zn + NEG, zn + NEG, z + NEG, z + NEG, z + NEG, z + NEG,
                 zn, zn + NEG, z + NEG, z, z + NEG, z + NEG, f, z, z + NEG, z,
                 zk, zk, jnp.full((c, DD_SLOTS), -1, jnp.int32), z, z + NEG,
                 zn + NEG, z, z + NEG, zn + NEG, z + NEG, z + NEG, z + NEG,
                 jnp.zeros((c,), jnp.int32), z, z + NEG, z, z + NEG, z, z + NEG, f)


def _rm(ctx, ranged_value):
    return jnp.where(ctx.is_ranged, ranged_value, 1.0)


def _enemy(ctx, units):
    return units.team[None, :] != ctx.team[:, None]


# ---- stats / defense / debuffs ----------------------------------------------

def overlord_bonus_ad(ctx) -> Any:
    """(C,) Tyranny + Retribution AD (F16; D12 default: % of AD from other sources)."""
    tyranny = TYRANNY * ctx.bonus_hp
    missing = jnp.clip(1.0 - ctx.hp / jnp.maximum(ctx.max_hp, 1.0), 0.0, 1.0)
    frac = RETRIBUTION * jnp.clip(missing / RETRIBUTION_FULL, 0.0, 1.0)
    return tyranny + frac * (ctx.total_ad + tyranny)


def terminus_resist(level) -> Any:
    lv = jnp.asarray(level, jnp.float32)
    out = TERM_RES_L1 + jnp.zeros_like(lv)
    for at, add in TERM_RES_STEPS:
        out = out + jnp.where(lv >= at, add, 0.0)
    return out


def stats(state: State, own, ctx) -> ItemStats:
    now = ctx.now
    ad = jnp.where(holds(own, OVERLORD), overlord_bonus_ad(ctx), 0.0)
    famine = FAMINE_BASE + jnp.where(ctx.is_ranged, FAMINE_RANGED, FAMINE_MELEE) * ctx.bonus_ad
    ah = jnp.where(holds(own, HUNGER), famine, 0.0)
    omni = jnp.where(holds(own, HUNGER) & (now < state.feast_until), dv(HUNGER, "OmnivampOnTakedown"), 0.0)
    fervor = jnp.where(holds(own, CLEAVER) & (now < state.fervor_until),
                       dv(CLEAVER, "MoveSpeedBonus") * _rm(ctx, dv(CLEAVER, "RangedMod")), 0.0)
    dmp_ms = jnp.where(holds(own, DMP), dv(DMP, "MaxMovementSpeed") * state.dmp / DMP_MAX, 0.0)
    hex_held = holds(own, HEXPLATE)
    hex_on = hex_held & (now < state.hex_until)
    hex_as = jnp.where(hex_on, jnp.where(ctx.is_ranged, dv(HEXPLATE, "BonusASRanged"),
                                         dv(HEXPLATE, "BonusASMelee")) / 100.0, 0.0)
    hex_ms = jnp.where(hex_on, jnp.where(ctx.is_ranged, dv(HEXPLATE, "BonusMSRanged"),
                                         dv(HEXPLATE, "BonusMSMelee")) / 100.0, 0.0)
    ult_haste = jnp.where(hex_held, dv(HEXPLATE, "UltimateHaste"), 0.0)
    basic_haste = jnp.where(holds(own, SHOJIN), dv(SHOJIN, "AHBase"), 0.0)
    sky_hp = jnp.where(holds(own, SUNDERED) & (now < state.sky_bonus_until), state.sky_bonus, 0.0)
    term = holds(own, TERMINUS)
    light = jnp.where(term & (now < state.term_light_until), state.term_light, 0.0) * terminus_resist(ctx.level)
    dark = jnp.where(term & (now < state.term_dark_until), state.term_dark, 0.0) * TERM_DARK_PER
    return ItemStats(attack_damage=ad, ability_haste=ah, omnivamp=omni, move_speed=fervor + dmp_ms,
                     attack_speed=hex_as, percent_move_speed=hex_ms, ultimate_haste=ult_haste,
                     basic_ability_haste=basic_haste, health=sky_hp, armor=light, magic_resist=light,
                     percent_armor_pen=dark, percent_magic_pen=dark)


def defense(state: State, own, ctx):
    frac = jnp.where(holds(own, DEATHS_DANCE), jnp.where(ctx.is_ranged, DD_RANGED, DD_MELEE), 0.0)
    return neutral_defense(ctx.level.shape[0])._replace(store_fraction=frac.astype(jnp.float32))


def debuffs(state: State, own, ctx, units):
    n = units.x.shape[0]
    stacks = jnp.where(holds(own, CLEAVER)[:, None] & (ctx.now < state.carve_until), state.carve, 0.0)
    keep = jnp.prod(1.0 - CARVE_PER_STACK * stacks, axis=0)
    return neutral_debuffs(n)._replace(percent_armor_reduction=1.0 - keep)


def attack_mods(state: State, own, ctx, units, target) -> AttackMods:
    n = units.x.shape[0]
    t = jnp.clip(target, 0, n - 1)
    ready = ctx.now >= jnp.take_along_axis(state.sky_cd, t[:, None], axis=1)[:, 0]
    force = holds(own, SUNDERED) & (target >= 0) & (target_class(units, target) == CLASS_CHAMPION) & ready
    return AttackMods(force, jnp.where(force, dv(SUNDERED, "CritModifier"), 1.0).astype(jnp.float32))


def shojin_ability_amp(state: State, own, ctx) -> Any:
    """(C,) additive DMG.40 amp for the holder's ability/proc damage (Focused Will).

    Applied to ability packets through ``packet_amp``.
    """
    stacks = jnp.where(holds(own, SHOJIN) & (ctx.now < state.shojin_until), state.shojin, 0.0)
    return stacks * dv(SHOJIN, "SpellDamageIncrease") * _rm(ctx, dv(SHOJIN, "RangedMod"))


def packet_amp(state: State, own, ctx, units, p) -> Any:
    """(P,) Focused Will on the holder's ability damage (ActiveSpell, not item)."""
    amp = shojin_ability_amp(state, own, ctx)
    src_is = p.src[:, None] == ctx.unit[None, :]
    ability = has(p.flags, TAG_ACTIVE_SPELL) & ~has(p.flags, TAG_ITEM)
    return jnp.where(ability, jnp.sum(jnp.where(src_is, amp[None, :], 0.0), axis=1), 0.0)


def boarding_party_resists(own, ctx, units, radius: float = BOARDING_PARTY_RANGE) -> Any:
    """(N,) bonus armor and MR (same value) for allied siege/super minions near a
    living Hullbreaker holder: level_bp(70, +6 at L>=9), ranged holder x0.5.
    Strongest holder wins (named aura). Minion stats are owned by the minion module."""
    lv = jnp.asarray(ctx.level, jnp.float32)
    val = (BOARDING_L1 + BOARDING_PER * jnp.maximum(0.0, lv - BOARDING_FROM + 1.0)) * _rm(ctx, BOARDING_RANGED)
    d = jnp.sqrt((units.x[None, :] - ctx.x[:, None]) ** 2 + (units.y[None, :] - ctx.y[:, None]) ** 2)
    ok = (holds(own, HULL) & ctx.alive)[:, None] & (units.team[None, :] == ctx.team[:, None]) \
        & units.is_siege_or_super[None, :] & units.alive[None, :] & (units.cls[None, :] == CLASS_MINION) \
        & (d <= radius)
    return jnp.max(jnp.where(ok, val[:, None], 0.0), axis=0)


# ---- event hooks ------------------------------------------------------------

def on_cast(state: State, own, ctx, units, cast):
    c, n = ctx.level.shape[0], units.x.shape[0]
    go = cast.started & ctx.alive
    ult = go & cast.is_ultimate & holds(own, HEXPLATE) & (ctx.now >= state.hex_cd)
    state = state._replace(
        hex_until=jnp.where(ult, ctx.now + dv(HEXPLATE, "HasteDuration"), state.hex_until),
        hex_cd=jnp.where(ult, ctx.now + dv(HEXPLATE, "Cooldown"), state.hex_cd),
        shojin_fresh=state.shojin_fresh | (go & holds(own, SHOJIN)))
    return state, effects(c, n)


def on_hit(state: State, own, ctx, units, attack):
    c, n = ctx.level.shape[0], units.x.shape[0]
    now = ctx.now
    hit = attack.hit & ctx.alive & (attack.target >= 0)
    t = jnp.clip(attack.target, 0, n - 1)
    tcls = target_class(units, t)
    champ, struct = tcls == CLASS_CHAMPION, tcls == CLASS_STRUCTURE
    onehot = onehot_units(jnp.where(hit, attack.target, -1), n)
    parts = []

    # BotRK Mist's Edge (target HP before this attack, ITEMS §10 INFERRED M) + Clawing Shadows.
    b = hit & holds(own, BOTRK) & ~struct
    mist = jnp.where(ctx.is_ranged, dv(BOTRK, "RangedValue"), dv(BOTRK, "MeleeValue")) * units.hp[t]
    capped = (tcls == CLASS_MINION) | (tcls == CLASS_MONSTER)
    mist = jnp.where(capped, jnp.minimum(mist, dv(BOTRK, "MonsterDamageCap")), mist)
    p_botrk = packets(b, ctx.unit, t, mist, PHYSICAL, ON_HIT_ITEM | PROP_LIFESTEAL, item=BOTRK)
    claw = b & champ & (now >= state.botrk_cd)
    claw_oh = onehot & claw[:, None]
    count = jnp.where(now < state.botrk_until, state.botrk_count, 0.0) + claw_oh
    proc = claw & (jnp.sum(jnp.where(claw_oh, count, 0.0), axis=1) >= 3)
    count = jnp.where(proc[:, None] & claw_oh, 0.0, count)
    botrk_until = jnp.where(claw_oh, now + dv(BOTRK, "AttackCounterDuration"), state.botrk_until)
    strength = -jnp.where(ctx.is_ranged, dv(BOTRK, "RangedMoveSpeedMod"), dv(BOTRK, "MoveSpeedMod"))
    slow = jnp.max(jnp.where(claw_oh & proc[:, None], strength[:, None], 0.0), axis=0)
    parts.append(effects(c, n, packets=p_botrk, slow=slow,
                         slow_duration=jnp.where(slow > 0, dv(BOTRK, "MoveSpeedDuration"), 0.0)))

    # Wit's End Fray, Terminus Shadow + Juxtaposition.
    w = hit & holds(own, WITS) & ~struct
    p_wits = packets(w, ctx.unit, t, WITS_DAMAGE, MAGIC, ON_HIT_ITEM | PROP_LIFESTEAL, item=WITS)
    tm = hit & holds(own, TERMINUS) & ~struct
    p_term = packets(tm, ctx.unit, t, TERM_BASE + TERM_BAD * ctx.bonus_ad + TERM_AP * ctx.ap, MAGIC,
                     ON_HIT_ITEM | PROP_LIFESTEAL, item=TERMINUS)
    jux = tm & champ
    dark_hit, light_hit = jux & state.term_next_dark, jux & ~state.term_next_dark
    dur = dv(TERMINUS, "BuffDuration")
    light = jnp.where(now < state.term_light_until, state.term_light, 0.0)
    dark = jnp.where(now < state.term_dark_until, state.term_dark, 0.0)
    state = state._replace(
        term_light=jnp.where(light_hit, jnp.minimum(light + 1, TERM_LIGHT_MAX), state.term_light),
        term_light_until=jnp.where(light_hit, now + dur, state.term_light_until),
        term_dark=jnp.where(dark_hit, jnp.minimum(dark + 1, TERM_DARK_MAX), state.term_dark),
        term_dark_until=jnp.where(dark_hit, now + dur, state.term_dark_until),
        term_next_dark=jnp.where(jux, ~state.term_next_dark, state.term_next_dark),
        botrk_count=count, botrk_until=botrk_until,
        botrk_cd=jnp.where(proc, now + dv(BOTRK, "Cooldown"), state.botrk_cd))

    # Hullbreaker Skipper: stacks from any attack; the 5th vs a champion/structure procs.
    hb = hit & holds(own, HULL)
    stacks = jnp.where(now < state.hull_until, state.hull, 0.0)
    hproc = hb & (champ | struct) & (stacks >= HULL_STACKS - 1)
    vs_champ = dv(HULL, "SkipperADRatio") * ctx.base_ad + dv(HULL, "MaxStackDamageHPRatio") * ctx.max_hp
    vs_struct = dv(HULL, "SkipperADRatioVSStructures") * ctx.base_ad \
        + dv(HULL, "MaxStackDamageVSStructuresHPRatio") * ctx.max_hp
    hdmg = jnp.where(struct, vs_struct, vs_champ) * _rm(ctx, HULL_RANGED)
    p_hull = packets(hproc, ctx.unit, t, hdmg, PHYSICAL, ON_HIT_ITEM | PROP_LIFESTEAL, item=HULL)
    state = state._replace(
        hull=jnp.where(hproc, 0.0, jnp.where(hb, jnp.minimum(stacks + 1, HULL_STACKS), state.hull)),
        hull_until=jnp.where(hb & ~hproc, now + dv(HULL, "SkipperStackDuration"), state.hull_until))

    # Dead Man's Plate discharge (no life steal, ITEMS §7).
    dm = hit & holds(own, DMP) & ~struct & (state.dmp > 0)
    s = state.dmp / DMP_MAX
    ddmg = s * (dv(DMP, "MaxStacksADRatio") * ctx.base_ad + DMP_FLAT_FULL)
    p_dmp = packets(dm, ctx.unit, t, ddmg, PHYSICAL, ON_HIT_ITEM, item=DMP)
    state = state._replace(dmp=jnp.where(dm, 0.0, state.dmp))

    # Sundered Sky heal (the crit itself comes from attack_mods at launch).
    sky_ready = now >= jnp.take_along_axis(state.sky_cd, t[:, None], axis=1)[:, 0]
    sk = hit & holds(own, SUNDERED) & champ & sky_ready
    missing = jnp.maximum(ctx.max_hp - ctx.hp, 0.0)
    heal = dv(SUNDERED, "HealBaseADRatio") * ctx.base_ad * _rm(ctx, dv(SUNDERED, "RangedHealMod")) \
        + dv(SUNDERED, "MissingHealthHeal") * missing
    heal = jnp.where(sk, heal, 0.0)
    # Overheal estimate uses source HSP only (GW / incoming heal are applied by the integrator).
    excess = jnp.maximum(heal * (1.0 + ctx.heal_shield_power) - missing, 0.0)
    gain = sk & (excess > 0)
    state = state._replace(
        sky_cd=jnp.where(onehot & sk[:, None], now + dv(SUNDERED, "Cooldown"), state.sky_cd),
        sky_bonus=jnp.where(gain, excess, state.sky_bonus),
        sky_bonus_until=jnp.where(gain, now + 8.0, state.sky_bonus_until))   # WIKI: bonus health 8 s

    # Bastionbreaker Sabotage: consumed by the next attack on a structure.
    sab = hit & holds(own, BASTION) & struct & (now < state.sabotage_until)
    total = (SABOTAGE_BASE + SABOTAGE_LETH * ctx.lethality) * _rm(ctx, dv(BASTION, "RangeModifier"))
    dot_dur = dv(BASTION, "DoTDuration")
    state = state._replace(
        sabotage_until=jnp.where(sab, NEG, state.sabotage_until),
        bb_dot_target=jnp.where(sab, t, state.bb_dot_target),
        bb_dot_rate=jnp.where(sab, total / dot_dur, state.bb_dot_rate),
        bb_dot_until=jnp.where(sab, now + dot_dur, state.bb_dot_until))

    parts.append(effects(c, n, packets=concat_packets(p_wits, p_term, p_hull, p_dmp), heal=heal))
    return state, merge_effects(parts, c, n)


def on_damage(state: State, own, ctx, units, report):
    c, n = ctx.level.shape[0], units.x.shape[0]
    now = ctx.now
    p, r = report.packets, report.resolved
    dcls = dst_class(report, units)
    landed = p.valid & (r.final > 0.0)
    src_is = p.src[None, :] == ctx.unit[:, None]                              # (C, P)
    dst_oh = (p.dst[:, None] == jnp.arange(n)[None, :]).astype(jnp.float32)   # (P, N)
    enemy = _enemy(ctx, units)

    def hit(mask):
        """(C, N) bool: holder's selected packets reached enemy unit n."""
        sel = (src_is & mask[None, :]).astype(jnp.float32)
        return ((sel @ dst_oh) > 0.0) & enemy

    is_phys = p.dtype == PHYSICAL
    basic = has(p.flags, TAG_BASIC_ATTACK)
    ability = has(p.flags, TAG_ACTIVE_SPELL) & ~has(p.flags, TAG_ITEM)
    champ_p = dcls == CLASS_CHAMPION
    any_champ = hit(landed & champ_p)
    state = state._replace(last_dmg=jnp.where(any_champ, now, state.last_dmg))
    parts = []

    # Black Cleaver Carve (basic attacks each; non-basic once per ICD) and Fervor.
    bc = holds(own, CLEAVER) & ctx.alive
    add_basic = hit(landed & is_phys & champ_p & basic)
    nonbasic = hit(landed & is_phys & champ_p & ~basic) & (now - state.carve_last >= CARVE_ICD - 1e-6)
    add = jnp.where(bc[:, None], add_basic.astype(jnp.float32) + nonbasic, 0.0)
    cur = jnp.where(now < state.carve_until, state.carve, 0.0)
    state = state._replace(
        carve=jnp.where(add > 0, jnp.minimum(cur + add, CARVE_MAX), cur),
        carve_until=jnp.where(add > 0, now + CARVE_DURATION, state.carve_until),
        carve_last=jnp.where(bc[:, None] & nonbasic, now, state.carve_last),
        fervor_until=jnp.where(bc & jnp.any(hit(landed & is_phys), axis=1),
                               now + dv(CLEAVER, "MoveSpeedDuration"), state.fervor_until))

    # Grievous Wounds family (Executioner's / Mortal Reminder / Chempunk).
    gw_holder = jnp.zeros((c,), bool)
    for iid in GW_ITEMS:
        gw_holder = gw_holder | holds(own, iid)
    gw_hit = hit(landed & is_phys & champ_p) & gw_holder[:, None]
    grievous = jnp.where(jnp.any(gw_hit, axis=0), dv(EXECUTIONER, "GrievousDuration"), 0.0)

    # Serylda's Bitter Cold: ability damage to enemies at or below 50% HP (after this tick).
    low = (r.hp <= dv(SERYLDA, "SlowThreshold") * r.max_hp) & (r.hp > 0.0)
    sl = hit(landed & ability) & holds(own, SERYLDA)[:, None] & low[None, :]
    sl_any = jnp.any(sl, axis=0)
    slow = jnp.where(sl_any, dv(SERYLDA, "SlowAmount"), 0.0)
    parts.append(effects(c, n, grievous=grievous, slow=slow,
                         slow_duration=jnp.where(sl_any, dv(SERYLDA, "SlowDuration"), 0.0)))

    # Spear of Shojin Focused Will stacks.
    sj = holds(own, SHOJIN) & jnp.any(hit(landed & ability), axis=1)
    grant = sj & (state.shojin_fresh | (now - state.shojin_last >= dv(SHOJIN, "CastIDLockout")))
    sj_cur = jnp.where(now < state.shojin_until, state.shojin, 0.0)
    state = state._replace(
        shojin=jnp.where(grant, jnp.minimum(sj_cur + 1, dv(SHOJIN, "StackCount")), state.shojin),
        shojin_until=jnp.where(grant, now + dv(SHOJIN, "StackDuration"), state.shojin_until),
        shojin_last=jnp.where(grant, now, state.shojin_last),
        shojin_fresh=state.shojin_fresh & ~grant)

    # Eclipse Ever Rising Moon.
    ecl = holds(own, ECLIPSE) & ctx.alive & (now >= state.ecl_cd)
    stack = hit(landed & champ_p & (p.item != ECLIPSE)) & ecl[:, None]
    window = dv(ECLIPSE, "WindowDuration")
    open_ = (now - state.ecl_first <= window) & (state.ecl_first < now)
    cand = stack & open_
    first_idx = jnp.argmax(cand, axis=1)
    fire = jnp.any(cand, axis=1)
    fire_oh = onehot_units(jnp.where(fire, first_idx, -1), n)
    ecl_first = jnp.where(fire_oh, NEG, jnp.where(stack & ~open_, now, state.ecl_first))
    pct = dv(ECLIPSE, "MeleePercMaxHP") * _rm(ctx, dv(ECLIPSE, "RangedPercMaxHPMult"))
    p_ecl = packets(fire, ctx.unit, first_idx, pct * units.max_hp[first_idx], PHYSICAL,
                    TAG_PROC | TAG_ITEM, item=ECLIPSE)
    shield = (dv(ECLIPSE, "MeleeBaseShield") + dv(ECLIPSE, "MeleeBonusADShieldRatio") * ctx.bonus_ad) \
        * _rm(ctx, dv(ECLIPSE, "RangedShieldMult"))
    state = state._replace(ecl_first=ecl_first,
                           ecl_cd=jnp.where(fire, now + dv(ECLIPSE, "Cooldown"), state.ecl_cd))

    # Bastionbreaker Shaped Charge: first ability packet on an enemy champion this tick.
    bb = holds(own, BASTION) & ctx.alive & (now >= state.bb_cd)
    team_ok = units.team[jnp.clip(p.dst, 0, n - 1)][None, :] != ctx.team[:, None]
    sel = src_is & (landed & ability & champ_p)[None, :] & team_ok & bb[:, None]
    bb_go = jnp.any(sel, axis=1)
    bb_dst = p.dst[jnp.argmax(sel, axis=1)]
    shaped = (SHAPED_BASE + SHAPED_LETH * ctx.lethality) * _rm(ctx, dv(BASTION, "AbilityDamageRangeMod"))
    p_bb = packets(bb_go, ctx.unit, bb_dst, shaped, TRUE, TAG_PROC | TAG_ITEM, item=BASTION)
    state = state._replace(bb_cd=jnp.where(bb_go, now + dv(BASTION, "Cooldown"), state.bb_cd))
    parts.append(effects(c, n, packets=concat_packets(p_ecl, p_bb),
                         shields=shield_grants(jnp.where(fire, shield, 0.0),
                                               duration=dv(ECLIPSE, "ShieldDuration"))))

    # Death's Dance: newly stored damage joins the current 0.25 s bucket.
    # NOTE: the pipeline stores every dtype (incl. this bleed's true damage); using
    # the full dd_pool_add conserves total damage (the re-stored share bleeds later).
    stored = jnp.where(holds(own, DEATHS_DANCE) | (jnp.sum(state.dd_amt, axis=1) > 0),
                       r.dd_pool_add[ctx.unit], 0.0)
    bucket = jnp.floor(now / DD_BUCKET).astype(jnp.int32)
    slot_oh = (jnp.arange(DD_SLOTS)[None, :] == (bucket % DD_SLOTS)) & (stored > 0)[:, None]
    same = state.dd_bucket == bucket
    new_amt = state.dd_amt + stored[:, None]
    dd_amt = jnp.where(slot_oh, new_amt, state.dd_amt)
    dd_rate = jnp.where(slot_oh, jnp.where(same, state.dd_rate + stored[:, None] / DD_BLEED, new_amt / DD_BLEED),
                        state.dd_rate)
    state = state._replace(dd_amt=dd_amt, dd_rate=dd_rate,
                           dd_bucket=jnp.where(slot_oh, bucket, state.dd_bucket))
    return state, merge_effects(parts, c, n)


def periodic(state: State, own, ctx, units):
    c, n = ctx.level.shape[0], units.x.shape[0]
    now, dt = ctx.now, ctx.dt
    # Dead Man's Plate momentum (time-based, dt-agnostic).
    moving = holds(own, DMP) & ctx.alive & ctx.moving
    dmp = jnp.where(moving, jnp.minimum(state.dmp + DMP_RATE * dt, DMP_MAX), state.dmp)
    dmp = jnp.where(holds(own, DMP) & ctx.alive, dmp, 0.0)
    # Death's Dance bleed (true, self-inflicted, no omnivamp / damage mods).
    bleed_k = jnp.minimum(state.dd_amt, state.dd_rate * dt)
    bleed = jnp.where(ctx.alive, jnp.sum(bleed_k, axis=1), 0.0)
    dd_amt = jnp.where(ctx.alive[:, None], state.dd_amt - bleed_k, 0.0)
    dd_rate = jnp.where(dd_amt > 1e-6, state.dd_rate, 0.0)
    dd_amt = jnp.where(dd_amt > 1e-6, dd_amt, 0.0)
    p_dd = packets(bleed > 0, ctx.unit, ctx.unit, bleed, TRUE,
                   PROP_NO_OMNIVAMP | PROP_NO_DAMAGE_MOD | TAG_PERIODIC, item=DEATHS_DANCE)
    heal = state.dd_heal_rate * jnp.clip(state.dd_heal_until - now, 0.0, dt)
    heal = jnp.where(ctx.alive, heal, 0.0)
    # Bastionbreaker Sabotage burn.
    burn = state.bb_dot_rate * jnp.clip(state.bb_dot_until - now, 0.0, dt)
    p_bb = packets(burn > 0, ctx.unit, state.bb_dot_target, burn, TRUE,
                   TAG_PERIODIC | TAG_ITEM | PROP_NO_OMNIVAMP, item=BASTION)
    state = state._replace(dmp=dmp, dd_amt=dd_amt, dd_rate=dd_rate)
    return state, effects(c, n, packets=concat_packets(p_dd, p_bb), heal=heal)


def on_takedown(state: State, own, ctx, units, kills):
    c, n = ctx.level.shape[0], units.x.shape[0]
    now = ctx.now
    recent = (now - state.last_dmg) <= TAKEDOWN_WINDOW
    td = jnp.any(kills.killed_units & (units.cls == CLASS_CHAMPION)[None, :] & recent, axis=1) & ctx.alive
    feast = td & holds(own, HUNGER)
    defy = td & holds(own, DEATHS_DANCE)
    sab = td & holds(own, BASTION)
    heal_total = dv(DEATHS_DANCE, "BonusADRatio") * ctx.bonus_ad
    hd = dv(DEATHS_DANCE, "HealDuration")
    state = state._replace(
        feast_until=jnp.where(feast, now + dv(HUNGER, "OmnivampDuration"), state.feast_until),
        dd_amt=jnp.where(defy[:, None], 0.0, state.dd_amt),
        dd_rate=jnp.where(defy[:, None], 0.0, state.dd_rate),
        dd_heal_rate=jnp.where(defy, heal_total / hd, state.dd_heal_rate),
        dd_heal_until=jnp.where(defy, now + hd, state.dd_heal_until),
        sabotage_until=jnp.where(sab, now + dv(BASTION, "BuffDuration"), state.sabotage_until))
    return state, effects(c, n)
