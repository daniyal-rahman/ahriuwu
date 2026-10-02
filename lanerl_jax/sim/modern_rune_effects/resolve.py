"""Resolve tree (8400) for patch 26.19 (RUNES.md §6, fixtures §13 F-13..F-19, F-30, F-31).

Numbers come from the 16.19.8230722 rune bin (``ea``); only trigger rules
the data does not encode are hard-coded here, each with its RUNES.md
citation:

* Grasp generation: combat event at ``t`` keeps generating until ``t + 3``,
  1 stack per 1.0 s up to 4; everything (primed or partial) drops 5 s after
  the last combat event (§6.1, U-12). ``gen_until`` is derived from the
  runtime clock ``ev.clocks.last_combat`` (``periodic`` sees the previous
  tick's clock, so the accumulator covers ``(now - dt, now]`` exactly).
* Second Wind: 0.4% of *current* missing HP per second, continuous at the
  sim tick (U-20), for 10 s after health damage > 0 from an enemy champion.
* Bone Plating: cooldown starts when the 1.5 s window ends or the 3rd block
  is used, whichever is first (U-22); at most one block per cast instance.
* Overgrowth: own line of sight via ``ev.sight`` (U-15), weight 1 per unit,
  ``{1663f8e7}``/``{ca029a2b}`` unused (U-15).
* Guardian shield 1.5 s (CLIENT, U-13); Demolish ``{8d810998}`` unused (U-14).
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from ..modern_damage import (CLASS_CHAMPION, CLASS_MINION, CLASS_MONSTER, CLASS_STRUCTURE, MAGIC, PHYSICAL,
                             TAG_AOE, TAG_BASIC_ATTACK, TAG_ON_HIT, TAG_PROC, concat_packets, packets)
from ..modern_item_data import ItemStats
from ..modern_item_effects.core import ShieldGrant
from .core import (BIG, RuneEvents, adaptive_damage_type, by_range, ea, effects, first_instance, has_rune,
                   in_circle, lin, onehot_units, packet_src_cls, rune_item, target_class)

GRASP, AFTERSHOCK, GUARDIAN = 8437, 8439, 8465
DEMOLISH, FONT_OF_LIFE, SHIELD_BASH = 8446, 8463, 8401
CONDITIONING, SECOND_WIND, BONE_PLATING = 8429, 8444, 8473
OVERGROWTH, REVITALIZE, UNFLINCHING = 8451, 8453, 8242

# ---- Grasp of the Undying (§6.1) --------------------------------------------
GRASP_PCT_DAMAGE = ea(GRASP, "PercentHealthDamage")
GRASP_PCT_HEAL = ea(GRASP, "PercentHealthHeal")
GRASP_HP_MELEE = ea(GRASP, "MaxHealthPerProc")
GRASP_HP_RANGED = ea(GRASP, "RangedHealthPerProc")
GRASP_RANGED_MOD = ea(GRASP, "RangedPenaltyMod")
GRASP_STACKS = ea(GRASP, "TriggerTime")         # 4 stacks, 1 per second ("every 4s in combat")
GRASP_WINDOW = ea(GRASP, "Window")              # primed / partial stacks lost 5 s after last combat (U-12)
GRASP_STACK_PERIOD = 1.0                        # RUNES §6.1 (wiki): 1 stack per 1.0 s
GRASP_GEN_AFTER = 3.0                           # RUNES §6.1 (wiki): generation continues 3 s after an event
_EPS = 1e-4                                     # float32 accumulator tolerance

# ---- Aftershock (§6.2) --------------------------------------------------------
AS_FLAT = ea(AFTERSHOCK, "FlatResists")
AS_PCT = ea(AFTERSHOCK, "PercentBonusResist")
AS_CAP_MIN, AS_CAP_MAX = ea(AFTERSHOCK, "BonusResistMin"), ea(AFTERSHOCK, "BonusResistMax")
AS_DELAY = ea(AFTERSHOCK, "DelayBeforeBurst")
AS_DMG_MIN, AS_DMG_MAX = ea(AFTERSHOCK, "StartingBaseDamage"), ea(AFTERSHOCK, "MaxBaseDamage")
AS_HP_RATIO = ea(AFTERSHOCK, "HealthRatio")
AS_RADIUS = ea(AFTERSHOCK, "DamageRadius")
AS_COOLDOWN = ea(AFTERSHOCK, "Cooldown")

# ---- Guardian (§6.3) ----------------------------------------------------------
GD_RANGE = ea(GUARDIAN, "SnuggleRange")
GD_GUARD = ea(GUARDIAN, "GuardDuration")
GD_SHIELD_MIN, GD_SHIELD_MAX = ea(GUARDIAN, "ShieldBase"), ea(GUARDIAN, "ShieldMax")
GD_AP, GD_HP = ea(GUARDIAN, "APRatio"), ea(GUARDIAN, "HPRatio")
GD_SHIELD_DURATION = ea(GUARDIAN, "ShieldDuration")   # CLIENT 1.5 (wiki 2), U-13
GD_CD_MIN, GD_CD_MAX = ea(GUARDIAN, "Cooldown"), ea(GUARDIAN, "CooldownMaxLevel")
GD_THR_MIN, GD_THR_MAX = ea(GUARDIAN, "ThresholdMin"), ea(GUARDIAN, "ThresholdMax")
GD_BUCKET = 0.25                                 # sliding 2.5 s damage window in 0.25 s buckets
GD_BUCKETS = int(round(GD_GUARD / GD_BUCKET))

# ---- Demolish (§6.4) ----------------------------------------------------------
DEMO_BASE_MELEE, DEMO_BASE_RANGED = ea(DEMOLISH, "BaseDamageMelee"), ea(DEMOLISH, "BaseDamageRanged")
DEMO_HP_MELEE, DEMO_HP_RANGED = ea(DEMOLISH, "HPRatioMelee"), ea(DEMOLISH, "HPRatioRanged")
DEMO_COOLDOWN = ea(DEMOLISH, "CooldownSeconds")
DEMO_LOCK = ea(DEMOLISH, "{97664ba4}")           # per-turret lock for every user (wiki: 3 s)
DEMO_STACKS = 3                                  # RUNES §6.4: 3rd stack consumes

# ---- Font of Life / Shield Bash (§6.5) ---------------------------------------
FONT_MIN, FONT_MAX = 10.0, 50.0                  # BaseHeal calc lin(10, 50)
FONT_RANGED = ea(FONT_OF_LIFE, "RangedMod")
FONT_COOLDOWN = ea(FONT_OF_LIFE, "Cooldown")
FONT_RANGE = 1000.0                              # RUNES §6.5 (wiki)
SB_MIN, SB_MAX = ea(SHIELD_BASH, "ProcBaseMin"), ea(SHIELD_BASH, "ProcBaseMax")
SB_HP = ea(SHIELD_BASH, "BonusHealthRatio") / 100.0
SB_SHIELD = ea(SHIELD_BASH, "ShieldRatio") / 100.0
SB_LINGER = ea(SHIELD_BASH, "ProcDuration")
# ``ev.shield_gained`` carries no shield duration; the empowerment window
# assumes a 2 s shield lifetime (INFERRED-L) and then lingers ProcDuration.
SB_ASSUMED_SHIELD_LIFE = 2.0

# ---- Conditioning / Second Wind / Bone Plating (§6.6) ------------------------
COND_TIME = ea(CONDITIONING, "MinutesRequired") * 60.0
COND_ARMOR, COND_MR = ea(CONDITIONING, "ArmorBase"), ea(CONDITIONING, "MRBase")
COND_PCT = ea(CONDITIONING, "ExtraResist")
SW_DURATION = ea(SECOND_WIND, "RegenSeconds")
SW_RATE = ea(SECOND_WIND, "RegenPercentMax") / SW_DURATION   # 0.4% missing HP per second (U-20)
BP_MIN, BP_MAX = ea(BONE_PLATING, "BlockBase"), ea(BONE_PLATING, "BlockMax")
BP_COUNT = int(ea(BONE_PLATING, "BlockCount"))
BP_DURATION = ea(BONE_PLATING, "BlockDuration")
BP_COOLDOWN = ea(BONE_PLATING, "Cooldown")

# ---- Overgrowth / Revitalize / Unflinching (§6.7) ----------------------------
OG_RANGE = ea(OVERGROWTH, "Range")
OG_PER_TIER = ea(OVERGROWTH, "UnitsPerTier")
OG_HP_PER_TIER = ea(OVERGROWTH, "FlatHealthPerTier")
OG_THRESHOLD = ea(OVERGROWTH, "ThresholdUnits")
OG_PCT = ea(OVERGROWTH, "ThresholdMaxHealthRatio")
REV_HSP = ea(REVITALIZE, "HealShieldPower")
REV_CUTOFF = ea(REVITALIZE, "HealthCutOff") / 100.0
REV_AMP = 1.0 + ea(REVITALIZE, "ExtraAmp") / 100.0
UNF_RESIST = ea(UNFLINCHING, "ResistMax")
UNF_LINGER = ea(UNFLINCHING, "Duration")

COVERAGE = {
    GRASP: "4 stacks at 1/s while generating (3 s after any enemy-unit damage event), lost 5 s after last "
           "combat (U-12); next on-hit vs champion: 3.5% max HP magic proc, 1.3% heal, +5 perm HP "
           "(ranged 40%/+2). Blocked attacks still proc (no block signal in on_hit)",
    AFTERSHOCK: "immobilize on enemy champion: +min(45 + 75% bonus resist, lin(80,150)) armor/MR for 2.5 s, "
                "then lin(25,120) + 8% bonus HP magic AoE to champions/monsters in 350; cd 20 s",
    GUARDIAN: "guard same-team champions within 350 (or 2.5 s after a unit-targeted cast on them); "
              ">= lin(50,165) post-mitigation from champions/monsters/turrets in a sliding 2.5 s window "
              "(0.25 s buckets) shields both lin(40,150)+20% AP+6% bonus HP for 1.5 s (U-13); cd lin(75,40). "
              "Lethal-damage trigger not modelled (shield cannot pre-empt resolved damage); allies must be holders",
    DEMOLISH: "per-(holder, turret) stacks on turret on-hit; 3rd deals 85+28% max HP (ranged 50+20%) physical "
              "proc; clears all holder stacks, 30 s cd, global 3 s per-turret lock (no stacks while locked "
              "or on cd); {8d810998} unused (U-14)",
    FONT_OF_LIFE: "slow/immobilize on enemy champion heals self and the lowest-HP%-then-nearest allied holder "
                  "champion within 1000 for lin(10,50) (ranged x0.7); cd 20 s, also at full HP",
    SHIELD_BASH: "shield gained (ev.shield_gained, post_tick) empowers the next on-hit vs a champion: lin(5,30) "
                 "+ 2.5% bonus HP + 15% shield, adaptive proc; window = shield duration (ev; 2 s if unknown) + 2 s "
                 "(no shield duration in ev); larger shields replace smaller",
    CONDITIONING: "from 12:00 game time: +8 armor/MR and total armor/MR x1.03 (percent_armor/percent_magic_resist)",
    SECOND_WIND: "after health damage > 0 from an enemy champion: heal_plain 0.4% of current missing HP per "
                 "second for 10 s (continuous, U-20); refreshes; champion-sourced packets only (pets, U-02, "
                 "need packet owner)",
    BONE_PLATING: "after health damage from an enemy champion: next 3 instances (1 per cast_id) from that "
                  "champion within 1.5 s reduced by lin(30,60) via packet_block; cd 55 s from window end or "
                  "3rd block (U-22). Later packets in the trigger pass are not reduced",
    OVERGROWTH: "enemy minion/monster deaths within 1400 with own sight (ev.sight, U-15), counted while dead: "
                "+3 HP per 8, x1.035 max HP at 120",
    REVITALIZE: "+5% heal and shield power; x1.10 on heals/shields received while below 40% HP (heal_mult). "
                "Heals/shields cast on low-HP allies are not amplified",
    UNFLINCHING: "+10 armor/MR while CC'd by an enemy champion (ev.holder_cc_from_champion) and 2 s after; "
                 "starts the tick after CC is seen (damage resolves first)",
}


class State(NamedTuple):
    grasp_acc: Any          # (C,) stack-time accumulator, stacks = floor(acc), capped at 4
    grasp_hp: Any           # (C,) permanent bonus HP from procs
    grasp_procs: Any        # (C,) int32
    as_until: Any           # (C,) resist window end = burst time
    as_pending: Any         # (C,) bool: burst still due
    as_armor: Any           # (C,) snapshotted bonus armor
    as_mr: Any
    as_cd_until: Any
    gd_cd_until: Any        # (C,)
    gd_guard_until: Any     # (C, N) guarded by unit-targeted cast
    gd_buf: Any             # (C, N, K) damage buckets
    gd_epoch: Any           # (C, K) int32 bucket epochs
    demo_stacks: Any        # (C, N) int32
    demo_cd_until: Any      # (C,)
    demo_lock_until: Any    # (N,) global per-turret lock
    font_cd_until: Any      # (C,)
    sb_amount: Any          # (C,) shield amount backing the empowerment
    sb_until: Any           # (C,)
    sw_until: Any           # (C,)
    bp_source: Any          # (C,) int32 unit
    bp_until: Any           # (C,)
    bp_left: Any            # (C,) int32 blocks left
    bp_seen: Any            # (C, BP_COUNT) int32 cast ids already blocked (0 = none)
    bp_cd_until: Any        # (C,)
    og_count: Any           # (C,) int32 counted deaths
    unf_until: Any          # (C,)


def init(n_champions: int, n_units: int) -> State:
    c, n = n_champions, n_units
    z = jnp.zeros((c,), jnp.float32)
    zi = jnp.zeros((c,), jnp.int32)
    never = z - BIG
    return State(
        grasp_acc=z, grasp_hp=z, grasp_procs=zi,
        as_until=never, as_pending=jnp.zeros((c,), bool), as_armor=z, as_mr=z, as_cd_until=never,
        gd_cd_until=never, gd_guard_until=jnp.full((c, n), -BIG, jnp.float32),
        gd_buf=jnp.zeros((c, n, GD_BUCKETS), jnp.float32), gd_epoch=jnp.full((c, GD_BUCKETS), -1, jnp.int32),
        demo_stacks=jnp.zeros((c, n), jnp.int32), demo_cd_until=never,
        demo_lock_until=jnp.full((n,), -BIG, jnp.float32),
        font_cd_until=never, sb_amount=z, sb_until=never, sw_until=never,
        bp_source=zi - 1, bp_until=never, bp_left=zi, bp_seen=jnp.zeros((c, BP_COUNT), jnp.int32),
        bp_cd_until=never, og_count=zi, unf_until=never)


def _f(x):
    return jnp.asarray(x, jnp.float32)


def _enemy_champion_target(ctx, units, idx):
    i = jnp.clip(idx, 0, units.cls.shape[0] - 1)
    return (idx >= 0) & (units.cls[i] == CLASS_CHAMPION) & (units.team[i] != ctx.team) & units.alive[i]


def _to_rows(per_unit, ctx):
    """Gather a (..., N) per-unit value at each holder row's unit -> (..., C)."""
    return per_unit[..., jnp.clip(ctx.unit, 0, per_unit.shape[-1] - 1)]


# ---- stats / heal multiplier ------------------------------------------------

def grasp_stacks(state: State) -> Any:
    return jnp.minimum(jnp.floor(state.grasp_acc + _EPS), GRASP_STACKS)


def stats(state: State, page, ctx, ev: RuneEvents) -> ItemStats:
    now = ctx.now
    cond = has_rune(page, CONDITIONING) & (ev.game_time >= COND_TIME)
    as_on = has_rune(page, AFTERSHOCK) & (now < state.as_until)
    unf = has_rune(page, UNFLINCHING) & (now < state.unf_until)
    og = has_rune(page, OVERGROWTH)
    tiers = jnp.floor(state.og_count.astype(jnp.float32) / OG_PER_TIER)
    health = jnp.where(has_rune(page, GRASP), state.grasp_hp, 0.0) + jnp.where(og, OG_HP_PER_TIER * tiers, 0.0)
    pct_hp = jnp.where(og & (state.og_count >= OG_THRESHOLD), OG_PCT, 0.0)
    armor = jnp.where(cond, COND_ARMOR, 0.0) + jnp.where(as_on, state.as_armor, 0.0) + jnp.where(unf, UNF_RESIST, 0.0)
    mr = jnp.where(cond, COND_MR, 0.0) + jnp.where(as_on, state.as_mr, 0.0) + jnp.where(unf, UNF_RESIST, 0.0)
    pct = jnp.where(cond, COND_PCT, 0.0)
    return ItemStats(health=_f(health), armor=_f(armor), magic_resist=_f(mr), percent_armor=_f(pct),
                     percent_magic_resist=_f(pct), percent_health=_f(pct_hp),
                     heal_shield_power=_f(jnp.where(has_rune(page, REVITALIZE), REV_HSP, 0.0)))


def heal_mult(state: State, page, ctx, ev) -> Any:
    low = ctx.hp < REV_CUTOFF * ctx.max_hp
    return _f(jnp.where(has_rune(page, REVITALIZE) & low, REV_AMP, 1.0))


# ---- action phase -------------------------------------------------------------

def on_cast(state: State, page, ctx, units, ev: RuneEvents):
    c, n = ctx.level.shape[0], units.x.shape[0]
    tgt = ev.cast.target
    ti = jnp.clip(tgt, 0, n - 1)
    ally = (tgt >= 0) & (units.cls[ti] == CLASS_CHAMPION) & (units.team[ti] == ctx.team) & (tgt != ctx.unit)
    go = has_rune(page, GUARDIAN) & ev.cast.started & ally
    guard = jnp.where(onehot_units(tgt, n) & go[:, None], ctx.now + GD_GUARD, state.gd_guard_until)
    return state._replace(gd_guard_until=_f(guard)), effects(c, n)


def on_hit(state: State, page, ctx, units, ev: RuneEvents):
    c, n = ctx.level.shape[0], units.x.shape[0]
    now, atk = ctx.now, ev.attack
    hit = atk.hit & ctx.alive
    tgt = jnp.maximum(atk.target, 0)
    vs_champ = hit & _enemy_champion_target(ctx, units, atk.target)

    # Grasp proc: primed (4 stacks, last combat < 5 s ago) and an on-hit vs an enemy champion.
    primed = (state.grasp_acc + _EPS >= GRASP_STACKS) & (now - ev.clocks.last_combat < GRASP_WINDOW)
    g = has_rune(page, GRASP) & vs_champ & primed
    gmod = by_range(ctx, 1.0, GRASP_RANGED_MOD)
    p_grasp = packets(g, ctx.unit, tgt, GRASP_PCT_DAMAGE * gmod * ctx.max_hp, MAGIC, TAG_PROC | TAG_ON_HIT,
                      item=rune_item(GRASP))
    heal = jnp.where(g, GRASP_PCT_HEAL * gmod * ctx.max_hp, 0.0)
    grasp_hp = state.grasp_hp + jnp.where(g, by_range(ctx, GRASP_HP_MELEE, GRASP_HP_RANGED), 0.0)

    # Shield Bash: empowered on-hit vs an enemy champion.
    sb = has_rune(page, SHIELD_BASH) & vs_champ & (now < state.sb_until) & (state.sb_amount > 0.0)
    sb_dmg = lin(SB_MIN, SB_MAX, ctx.level) + SB_HP * ctx.bonus_hp + SB_SHIELD * state.sb_amount
    p_sb = packets(sb, ctx.unit, tgt, sb_dmg, adaptive_damage_type(ev), TAG_PROC | TAG_ON_HIT,
                   item=rune_item(SHIELD_BASH))

    # Demolish: per-(holder, turret) stacks; the 3rd consumes.
    tcls = target_class(units, atk.target)
    ti = jnp.clip(atk.target, 0, n - 1)
    on_turret = has_rune(page, DEMOLISH) & hit & (atk.target >= 0) & (tcls == CLASS_STRUCTURE) \
        & ev.is_turret[ti] & (units.team[ti] != ctx.team)
    oh = onehot_units(atk.target, n) & on_turret[:, None]
    can = oh & (now >= state.demo_cd_until)[:, None] & (now >= state.demo_lock_until)[None, :]
    stacks = jnp.minimum(state.demo_stacks + can.astype(jnp.int32), DEMO_STACKS)
    full = can & (stacks >= DEMO_STACKS)
    full = full & (jnp.cumsum(full.astype(jnp.int32), axis=0) == 1)     # one consumer per turret per tick
    consumed = jnp.any(full, axis=1)
    stacks = jnp.where(consumed[:, None], 0, stacks).astype(jnp.int32)
    demo_dmg = by_range(ctx, DEMO_BASE_MELEE + DEMO_HP_MELEE * ctx.max_hp, DEMO_BASE_RANGED + DEMO_HP_RANGED * ctx.max_hp)
    p_demo = packets(consumed, ctx.unit, tgt, demo_dmg, PHYSICAL, TAG_PROC | TAG_BASIC_ATTACK,
                     item=rune_item(DEMOLISH))
    lock = jnp.where(jnp.any(full, axis=0), now + DEMO_LOCK, state.demo_lock_until)

    state = state._replace(
        grasp_acc=_f(jnp.where(g, 0.0, state.grasp_acc)), grasp_hp=_f(grasp_hp),
        grasp_procs=(state.grasp_procs + g.astype(jnp.int32)).astype(jnp.int32),
        sb_amount=_f(jnp.where(sb, 0.0, state.sb_amount)), sb_until=_f(jnp.where(sb, -BIG, state.sb_until)),
        demo_stacks=stacks, demo_cd_until=_f(jnp.where(consumed, now + DEMO_COOLDOWN, state.demo_cd_until)),
        demo_lock_until=_f(lock))
    return state, effects(c, n, packets=concat_packets(p_grasp, p_sb, p_demo), heal=_f(heal))


def periodic(state: State, page, ctx, units, ev: RuneEvents):
    c, n = ctx.level.shape[0], units.x.shape[0]
    now, dt = ctx.now, ctx.dt
    start = now - dt

    # Grasp generation over (now - dt, now] while now < last_combat + 3; decay after 5 s.
    lc = ev.clocks.last_combat
    gen = jnp.clip(lc + GRASP_GEN_AFTER - start, 0.0, dt)
    acc = jnp.minimum(state.grasp_acc + jnp.where(state.grasp_acc + _EPS < GRASP_STACKS, gen, 0.0), GRASP_STACKS)
    acc = jnp.where((now - lc >= GRASP_WINDOW) | ~has_rune(page, GRASP), 0.0, acc)

    # Second Wind regen on current missing HP over the active part of the tick.
    sw_t = jnp.clip(state.sw_until - start, 0.0, dt)
    regen = jnp.where(has_rune(page, SECOND_WIND) & ctx.alive & (ctx.hp > 0.0),
                      SW_RATE * jnp.maximum(ctx.max_hp - ctx.hp, 0.0) * sw_t, 0.0)

    # Aftershock burst.
    burst = has_rune(page, AFTERSHOCK) & state.as_pending & (now >= state.as_until)
    fire = burst & ctx.alive
    near = in_circle(units, ctx.x, ctx.y, jnp.full((c,), AS_RADIUS, jnp.float32))
    tgt = near & (units.team[None, :] != ctx.team[:, None]) & units.alive[None, :] & units.targetable[None, :] \
        & ((units.cls[None, :] == CLASS_CHAMPION) | (units.cls[None, :] == CLASS_MONSTER)) & fire[:, None]
    as_dmg = lin(AS_DMG_MIN, AS_DMG_MAX, ctx.level) + AS_HP_RATIO * ctx.bonus_hp
    p_as = packets(tgt, ctx.unit[:, None], jnp.arange(n)[None, :], as_dmg[:, None], MAGIC, TAG_PROC | TAG_AOE,
                   item=rune_item(AFTERSHOCK))

    # Overgrowth: enemy minion/monster deaths within 1400 in own sight (counted while dead).
    d2 = (units.x[None, :] - ctx.x[:, None]) ** 2 + (units.y[None, :] - ctx.y[:, None]) ** 2
    farm = (units.cls == CLASS_MINION) | (units.cls == CLASS_MONSTER)
    counted = ev.deaths[None, :] & farm[None, :] & (units.team[None, :] != ctx.team[:, None]) \
        & (d2 <= OG_RANGE ** 2) & ev.sight
    og = state.og_count + jnp.where(has_rune(page, OVERGROWTH), jnp.sum(counted, axis=1), 0)

    # Unflinching: refresh the linger while under champion CC.
    unf = jnp.where(has_rune(page, UNFLINCHING) & ev.holder_cc_from_champion, now + UNF_LINGER, state.unf_until)

    state = state._replace(grasp_acc=_f(acc), as_pending=state.as_pending & ~burst, og_count=og.astype(jnp.int32),
                           unf_until=_f(unf))
    return state, effects(c, n, packets=p_as, heal_plain=_f(regen))


def on_cc(state: State, page, ctx, units, ev: RuneEvents):
    c, n = ctx.level.shape[0], units.x.shape[0]
    now = ctx.now
    champ = (units.cls[None, :] == CLASS_CHAMPION) & (units.team[None, :] != ctx.team[:, None])
    immob = jnp.any(ev.cc.immobilized & champ, axis=1) & ctx.alive
    impair = jnp.any((ev.cc.immobilized | ev.cc.slowed) & champ, axis=1) & ctx.alive

    # Aftershock trigger: snapshot the capped resists.
    a = has_rune(page, AFTERSHOCK) & immob & (now >= state.as_cd_until)
    cap = lin(AS_CAP_MIN, AS_CAP_MAX, ctx.level)
    state = state._replace(
        as_until=_f(jnp.where(a, now + AS_DELAY, state.as_until)), as_pending=state.as_pending | a,
        as_armor=_f(jnp.where(a, jnp.minimum(AS_FLAT + AS_PCT * ctx.bonus_armor, cap), state.as_armor)),
        as_mr=_f(jnp.where(a, jnp.minimum(AS_FLAT + AS_PCT * ctx.bonus_mr, cap), state.as_mr)),
        as_cd_until=_f(jnp.where(a, now + AS_COOLDOWN, state.as_cd_until)))

    # Font of Life: self + the most-wounded (then nearest) allied holder champion within 1000.
    f = has_rune(page, FONT_OF_LIFE) & impair & (now >= state.font_cd_until)
    amount = lin(FONT_MIN, FONT_MAX, ctx.level) * by_range(ctx, 1.0, FONT_RANGED)
    d = jnp.sqrt((ctx.x[None, :] - ctx.x[:, None]) ** 2 + (ctx.y[None, :] - ctx.y[:, None]) ** 2)   # (C, C')
    ally = (ctx.team[None, :] == ctx.team[:, None]) & (ctx.unit[None, :] != ctx.unit[:, None]) \
        & ctx.alive[None, :] & (d <= FONT_RANGE)
    frac = ctx.hp / jnp.maximum(ctx.max_hp, 1.0)
    key = jnp.where(ally, frac[None, :] + d * 1e-7, jnp.inf)
    pick = ally & (jnp.argsort(jnp.argsort(key, axis=1), axis=1) == 0) & f[:, None]
    heal = jnp.where(f, amount, 0.0) + jnp.sum(jnp.where(pick, amount[:, None], 0.0), axis=0)
    state = state._replace(font_cd_until=_f(jnp.where(f, now + FONT_COOLDOWN, state.font_cd_until)))
    return state, effects(c, n, heal=_f(heal))


# ---- damage pipeline ----------------------------------------------------------

def _bp_select(state: State, page, ctx, p) -> Any:
    """(C, P) packets Bone Plating reduces for each holder."""
    active = has_rune(page, BONE_PLATING) & (ctx.now < state.bp_until) & (state.bp_left > 0)
    sel = active[:, None] & p.valid[None, :] & (p.dst[None, :] == ctx.unit[:, None]) \
        & (p.src[None, :] == state.bp_source[:, None])
    seen = jnp.any((p.cast_id[None, :, None] == state.bp_seen[:, None, :]) & (p.cast_id != 0)[None, :, None], axis=2)
    sel = first_instance(p, sel & ~seen)
    rank = jnp.cumsum(sel.astype(jnp.int32), axis=1)
    return sel & (rank <= state.bp_left[:, None])


def bone_plating_block(ctx) -> Any:
    return lin(BP_MIN, BP_MAX, ctx.level)


def packet_block(state: State, page, ctx, units, ev, p) -> Any:
    sel = _bp_select(state, page, ctx, p)
    return _f(jnp.sum(jnp.where(sel, bone_plating_block(ctx)[:, None], 0.0), axis=0))


def _champion_health_hits(report, ctx, units) -> Any:
    """(C, P) packets that dealt health damage > 0 to the holder from an enemy champion (RUNES §1.5)."""
    p, r = report.packets, report.resolved
    n = units.x.shape[0]
    steam = units.team[jnp.clip(p.src, 0, n - 1)]
    from_champ = (packet_src_cls(p, units) == CLASS_CHAMPION)
    return p.valid[None, :] & (p.dst[None, :] == ctx.unit[:, None]) & from_champ[None, :] \
        & (steam[None, :] != ctx.team[:, None]) & (r.health_loss > 0.0)[None, :]


def on_damage(state: State, page, ctx, units, ev: RuneEvents):
    c, n = ctx.level.shape[0], units.x.shape[0]
    now = ctx.now
    rep = ev.report
    p = rep.packets
    hits = _champion_health_hits(rep, ctx, units)
    hurt = jnp.any(hits, axis=1)

    # Second Wind refresh.
    sw_until = jnp.where(has_rune(page, SECOND_WIND) & hurt, now + SW_DURATION, state.sw_until)

    # Bone Plating: consume blocks used by this pass (same selection as packet_block), then activation.
    used = _bp_select(state, page, ctx, p)
    n_used = jnp.sum(used.astype(jnp.int32), axis=1)
    before = BP_COUNT - state.bp_left
    slot = before[:, None] + jnp.cumsum(used.astype(jnp.int32), axis=1) - 1            # (C, P)
    seen = state.bp_seen
    for k in range(BP_COUNT):
        put = used & (slot == k)
        val = jnp.sum(jnp.where(put, p.cast_id[None, :], 0), axis=1)
        seen = seen.at[:, k].set(jnp.where(jnp.any(put, axis=1), val, seen[:, k]).astype(jnp.int32))
    left = state.bp_left - n_used
    spent = (n_used > 0) & (left <= 0)
    bp_until = jnp.where(spent, now, state.bp_until)
    bp_cd = jnp.where(spent, now + BP_COOLDOWN, state.bp_cd_until)
    trig = has_rune(page, BONE_PLATING) & hurt & (now >= state.bp_cd_until)
    first = jnp.argmax(hits, axis=1)
    src = jnp.take_along_axis(jnp.broadcast_to(p.src[None, :], hits.shape), first[:, None], axis=1)[:, 0]
    state = state._replace(
        sw_until=_f(sw_until),
        bp_source=jnp.where(trig, src, state.bp_source).astype(jnp.int32),
        bp_until=_f(jnp.where(trig, now + BP_DURATION, bp_until)),
        bp_left=jnp.where(trig, BP_COUNT, left).astype(jnp.int32),
        bp_seen=jnp.where(trig[:, None], 0, seen).astype(jnp.int32),
        bp_cd_until=_f(jnp.where(trig, now + BP_DURATION + BP_COOLDOWN, bp_cd)))

    # Guardian: sliding 2.5 s window of damage on self / guarded same-team champions.
    state, eff = _guardian(state, page, ctx, units, ev)
    return state, eff


def _guardian(state: State, page, ctx, units, ev: RuneEvents):
    c, n = ctx.level.shape[0], units.x.shape[0]
    now = ctx.now
    p, r = ev.report.packets, ev.report.resolved
    si, di = jnp.clip(p.src, 0, n - 1), jnp.clip(p.dst, 0, n - 1)
    scls = units.cls[si]
    src_ok = (scls == CLASS_CHAMPION) | (scls == CLASS_MONSTER) | ((scls == CLASS_STRUCTURE) & ev.is_turret[si])
    ok = p.valid & src_ok & (units.team[si] != units.team[di]) & (units.cls[di] == CLASS_CHAMPION)
    per_unit = jnp.zeros((n,), jnp.float32).at[di].add(jnp.where(ok, r.final, 0.0).astype(jnp.float32))
    friendly = (units.team[None, :] == ctx.team[:, None]) & (units.cls[None, :] == CLASS_CHAMPION)   # (C, N)
    epoch = jnp.floor(now / GD_BUCKET).astype(jnp.int32)
    b = jnp.mod(epoch, GD_BUCKETS)
    hot = jnp.arange(GD_BUCKETS) == b                                                               # (K,)
    stale = hot[None, :] & (state.gd_epoch != epoch)                                                # (C, K)
    buf = jnp.where(stale[:, None, :], 0.0, state.gd_buf)
    buf = buf + jnp.where(hot[None, None, :] & friendly[:, :, None], per_unit[None, :, None], 0.0)
    ep = jnp.where(hot[None, :], epoch, state.gd_epoch)
    live = (ep > epoch - GD_BUCKETS) & (ep >= 0)
    window = jnp.sum(jnp.where(live[:, None, :], buf, 0.0), axis=2)                                # (C, N)

    self_n = onehot_units(ctx.unit, n)
    d = jnp.sqrt((units.x[None, :] - ctx.x[:, None]) ** 2 + (units.y[None, :] - ctx.y[:, None]) ** 2)
    allies = friendly & ~self_n & units.alive[None, :]
    guarded = allies & ((d <= GD_RANGE) | (now < state.gd_guard_until))
    thr = lin(GD_THR_MIN, GD_THR_MAX, ctx.level)
    over = jnp.any((self_n | guarded) & (window >= thr[:, None]), axis=1)
    trig = has_rune(page, GUARDIAN) & ctx.alive & (now >= state.gd_cd_until) & jnp.any(guarded, axis=1) & over
    amount = lin(GD_SHIELD_MIN, GD_SHIELD_MAX, ctx.level) + GD_AP * ev.ap + GD_HP * ctx.bonus_hp
    # Ally shields go to the ally's holder row (non-holder allies cannot receive Effects).
    to_ally = guarded & trig[:, None]                                                              # (C, N)
    ally_rows = _to_rows(to_ally, ctx)                                                             # (C, C')
    ally_amt = jnp.max(jnp.where(ally_rows, amount[:, None], 0.0), axis=0, initial=0.0)
    amt = jnp.stack([jnp.where(trig, amount, 0.0), ally_amt], axis=1)
    shields = ShieldGrant(_f(amt), jnp.zeros((c, 2), jnp.int32), jnp.full((c, 2), GD_SHIELD_DURATION, jnp.float32),
                          jnp.full((c, 2), jnp.inf, jnp.float32))
    cd = lin(GD_CD_MIN, GD_CD_MAX, ctx.level)
    state = state._replace(gd_buf=_f(jnp.where(trig[:, None, None], 0.0, buf)), gd_epoch=ep.astype(jnp.int32),
                           gd_cd_until=_f(jnp.where(trig, now + cd, state.gd_cd_until)))
    return state, effects(c, n, shields=shields)


# ---- end of tick ----------------------------------------------------------------

def post_tick(state: State, page, ctx, units, ev: RuneEvents) -> State:
    """Shield Bash: arm on any shield gained this tick (largest current/recent shield wins)."""
    gained = has_rune(page, SHIELD_BASH) & (ev.shield_gained > 0.0)
    active = ctx.now < state.sb_until
    current = jnp.where(active, state.sb_amount, 0.0)
    amount = jnp.where(gained, jnp.maximum(current, ev.shield_gained), state.sb_amount)
    life = jnp.where(ev.shield_gained_duration > 0.0, ev.shield_gained_duration, SB_ASSUMED_SHIELD_LIFE)
    until = jnp.where(gained, jnp.maximum(state.sb_until, ctx.now + life + SB_LINGER),
                      state.sb_until)
    return state._replace(sb_amount=_f(amount), sb_until=_f(until))
