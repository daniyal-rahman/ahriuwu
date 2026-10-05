"""Support items and the support quest line (ITEMS_CATALOG "Support quest line", ITEMS.md §10–12).

Values are read from the 16.19.8230722 item data (``dv``). Hard-coded numbers
come from wiki text where the client has no data value; each is marked.

Framework scope limits and how this module handles them
-------------------------------------------------------
* ``Effects`` heals/shields/mana target the HOLDER only, and the 1v1 lane has
  no allied champions. Ally-side parts are exposed as pure helpers
  (``on_ally_support``, ``bandlepipes_aura``, ``ally_*``) that return exact
  amounts. ``units`` is used to find allied champions, so the holder-side
  conditions ("near an ally") already work once allies exist.
* The core protocol has no "holder applied crowd control" or "holder healed or
  shielded an ally" event. This module adds two hooks with the standard
  shape; the integrator calls them directly (they are not in
  ``__init__._each``):
      on_cc(state, own, ctx, units, cc: CC) -> (state, Effects)
      on_ally_support(state, own, ctx, units, ev: AllySupport)
          -> (state, Effects, AllyBenefit)
  Slows emitted by this module (Zeke's storm, Celestial shockwave) are fed
  to ``on_cc`` internally.
* Celestial Opposition reduces damage from CHAMPIONS only. ``HolderDefense``
  has no source-class mask, so ``defense`` cannot carry it. The multiplier is
  exposed as ``celestial_champion_damage_mult``. The integrator multiplies the
  raw damage of champion-sourced packets aimed at the holder by it before
  resolution (pre-mitigation, wiki).
* Item actives (Shurelya's, Locket, Redemption, Mikael's, Knight's Vow Pledge,
  support-line Stealth Ward charges) are DEFERRED (MODERN-009). Amount
  helpers are provided so an integrator can add them later.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from ...core.damage import (MAGIC, ON_HIT_ITEM, PROP_EXECUTE, PROP_REACTIVE, TAG_AOE, TAG_BASIC_ATTACK, TAG_ITEM,
                            TAG_ON_HIT, TAG_PERIODIC, TAG_PROC, TRUE, concat_packets, has, packets)
from ..catalog import STAT_INDEX, ItemStats, catalog, level_bp
from .core import (CC, CLASS_CHAMPION, CLASS_MINION, CLASS_MONSTER, CLASS_STRUCTURE, Debuffs, Effects, dv,
                   effects, enemy_mask, holds, in_circle, merge_effects, neutral_defense, target_class, unit_pos)

SHURELYA, BANDLEPIPES, ZEKES, REDEMPTION, KNIGHTS_VOW = 2065, 2524, 3050, 3107, 3109
LOCKET, MIKAELS, CENSER, MANDATE, FLOWING = 3190, 3222, 3504, 4005, 6616
MOONSTONE, ECHOES, DAWNCORE = 6617, 6620, 6621
ATLAS, BOUNTY, CELESTIAL, DREAM, ZAZZAK, SLEIGH = 3865, 3867, 3869, 3870, 3871, 3876
BLOODSONG = 3877  # spellblade module; listed only for the gold table note below

NEVER = -1e9

# ---- client values ----------------------------------------------------------
# Support-line gold generation, gold per 10 s (GP10). Bloodsong 3877 also has
# GP10 = 9; it belongs to the spellblade module, which must emit it (see
# ``support_line_gold``).
GP10 = {ATLAS: dv(ATLAS, "GP10"), BOUNTY: dv(BOUNTY, "GP10"), CELESTIAL: dv(CELESTIAL, "GP10"),
        DREAM: dv(DREAM, "GP10"), ZAZZAK: dv(ZAZZAK, "GP10"), SLEIGH: dv(SLEIGH, "GP10")}

ATLAS_QUEST_GOLD = dv(ATLAS, "QuestGoldRequirement")          # 400
ATLAS_MAX_CHARGES = dv(ATLAS, "MaxCharges")                   # 3
ATLAS_CHARGE_CD = dv(ATLAS, "ChargeCooldown")                 # 20
ATLAS_FIRST_CHARGE = dv(ATLAS, "FirstChargeOffset")           # 20
ATLAS_ALLY_RADIUS = dv(ATLAS, "AllyChampNearbyRadius")        # 2000 (wiki: 1050)
ATLAS_ALLY_RADIUS_MINION = dv(ATLAS, "AllyChampNearbyRadiusMinion")  # 1300
ATLAS_GOLD_MELEE = dv(ATLAS, "GoldOnHitMelee")                # 18
ATLAS_GOLD_RANGED = dv(ATLAS, "GoldOnHit")                    # 18
ATLAS_MINION_GOLD = dv(ATLAS, "ExecuteMinionGold")            # 18
ATLAS_CANNON_GOLD = dv(ATLAS, "ExecuteCannonGold")            # 20
ATLAS_EXEC_MELEE = dv(ATLAS, "MeleeExecutePerc")              # 0.5
ATLAS_EXEC_RANGED = dv(ATLAS, "RangedExecutePerc")            # 0.333 (wiki: 30%)

BANDLE_DURATION = dv(BANDLEPIPES, "Duration")                 # 8, ranged x0.5 (calc BuffDuration)
BANDLE_RANGED_DURATION_MULT = 0.5                             # client calc mRangedMultiplier
BANDLE_MS = dv(BANDLEPIPES, "MoveSpeed")                      # 20 flat
BANDLE_AS = dv(BANDLEPIPES, "MeleeAuraAttackSpeed")           # 0.30
BANDLE_AS_RANGED_MULT = dv(BANDLEPIPES, "RangedAttackSpeedMultiplier")  # 0.667 -> 20%
BANDLE_AURA = dv(BANDLEPIPES, "AuraRange")                    # 900

ZEKE_ULT_HASTE = dv(ZEKES, "UltimateHaste")
ZEKE_CD = dv(ZEKES, "Cooldown")
ZEKE_READY = dv(ZEKES, "ReadyDuration")
ZEKE_DURATION = dv(ZEKES, "Duration")
ZEKE_DPS = dv(ZEKES, "DamagePerSecond")
ZEKE_RADIUS = dv(ZEKES, "StormRadius")
ZEKE_SLOW = dv(ZEKES, "SlowAmount")
ZEKE_TICK = 1.0          # INFERRED M: DamagePerSecond applied as 1 Hz ticks, first at summon
ZEKE_SLOW_REFRESH = 0.25  # INFERRED M: slow re-applied each tick while inside, lingers 0.25 s

CENSER_AS = dv(CENSER, "AttackSpeedMin")
CENSER_ONHIT = dv(CENSER, "OnHitMin")
CENSER_DURATION = dv(CENSER, "Duration")

MANDATE_AMP = dv(MANDATE, "DamageAmp")
MANDATE_DURATION = dv(MANDATE, "DamageAmpDuration")
MANDATE_IMMOBILIZE_AH = dv(MANDATE, "ImmobilizingAbilityAH")

FLOWING_AP = dv(FLOWING, "APMod")
FLOWING_AH = dv(FLOWING, "AHMod")
FLOWING_DURATION = dv(FLOWING, "BuffDuration")

MOONSTONE_RANGE = dv(MOONSTONE, "EffectRange")
MOONSTONE_HEAL = dv(MOONSTONE, "ChainHeal")
MOONSTONE_SHIELD = dv(MOONSTONE, "ChainShield")
MOONSTONE_SINGLE_HEAL = dv(MOONSTONE, "SingleHeal")
MOONSTONE_SINGLE_SHIELD = dv(MOONSTONE, "SingleShield")

ECHOES_RATE = dv(ECHOES, "DamageStorageRate")
ECHOES_CONVERSION = dv(ECHOES, "ChargeToHealConversion")
ECHOES_MIN_TRIGGER = dv(ECHOES, "MinimumHealToTrigger")
ECHOES_CAP_L1, ECHOES_CAP_PER_LEVEL = 80.0, 10.0   # calc MaxCharges: L1 80, mInitialBonusPerLevel 10

DAWN_AP = dv(DAWNCORE, "APPerManaRegen")
DAWN_HSP = dv(DAWNCORE, "HSPowerPerManaRegen")

CEL_DR_MELEE = dv(CELESTIAL, "MeleeShieldDRPercentage")
CEL_DR_RANGED = dv(CELESTIAL, "RangedShieldDRPercentage")
CEL_LINGER = dv(CELESTIAL, "ShieldLingerAfterInitiallyPopped")
CEL_CD = dv(CELESTIAL, "Cooldown")
CEL_RADIUS = dv(CELESTIAL, "Radius")
CEL_SLOW = dv(CELESTIAL, "SlowAmount")
CEL_SLOW_DURATION = dv(CELESTIAL, "SlowDuration")

DREAM_CD = dv(DREAM, "RechargeTime")
DREAM_DURATION = dv(DREAM, "BubbleDuration")
DREAM_AOE_MOD = dv(DREAM, "PurpleBubbleAoEMod")
DREAM_MIN_TRIGGER = dv(DREAM, "MinHealAmount")

ZAZ_BASE = dv(ZAZZAK, "BaseDamage")
ZAZ_AP = dv(ZAZZAK, "APRatio")
ZAZ_PCT = dv(ZAZZAK, "PercentHPDamage")
ZAZ_MONSTER_CAP = dv(ZAZZAK, "MonsterDamageCap")
ZAZ_CD = 10.0            # client calc Cooldown = NumberCalculationPart 10
ZAZ_DELAY = 0.5          # WIKI "after a 0.5-second delay"
ZAZ_RADIUS = 250.0       # INFERRED L: no client or wiki radius; edge-inclusive

SLEIGH_CD = dv(SLEIGH, "Cooldown")
SLEIGH_DURATION = dv(SLEIGH, "BuffDuration")
SLEIGH_MS = dv(SLEIGH, "MoveSpeedBuff")
SLEIGH_RANGE = dv(SLEIGH, "MoveSpeedRange")
SLEIGH_REQUIRES_ALLY = False   # tooltip "near allies"; wiki has no ally condition (default)

LOCKET_RANGE = dv(LOCKET, "ShieldRange")
REDEMPTION_DAMAGE = dv(REDEMPTION, "DamageToChampions")
KV_REDIRECT = dv(KNIGHTS_VOW, "DamageRedirection")
KV_THRESHOLD = dv(KNIGHTS_VOW, "DamageRedirectionThreshold")
KV_HEAL = dv(KNIGHTS_VOW, "AllyHealingConversion")
KV_RANGE = dv(KNIGHTS_VOW, "TetherRange")

COVERAGE = {
    SHURELYA: "no passive; Inspiring Speech active (30% MS 4 s, r1000, cd 75) in actives",
    BANDLEPIPES: "Fanfare via on_cc (slow/immobilize enemy champion): 8 s (ranged 4 s) +20 MS and 30% "
                 "(ranged x0.667) AS to self; ally AS aura r900 via bandlepipes_aura (no allies in 1v1)",
    ZEKES: "15 ultimate haste; Frostfire Tempest: ult cast (cd 45 from cast) readies 5 s, storm on enemy "
           "champion within 350 or at window end, 5 s, 30 magic/s (1 Hz) to champions+monsters, 30% slow",
    REDEMPTION: "no passive; Intervention active in actives (redemption_heal helper)",
    KNIGHTS_VOW: "Sacrifice needs a Pledged ally (Pledge is ally-only: inert, actives.INERT); knights_vow_* helpers only",
    LOCKET: "no passive; Devotion active in actives (locket_shield helper)",
    MIKAELS: "no passive; Purify is ally-only: inert (actives.INERT; mikael_heal helper)",
    CENSER: "Sanctify via on_ally_support: holder 25% AS + 20 magic on-hit for 6 s; ally buff returned "
            "in AllyBenefit (no allies in 1v1)",
    MANDATE: "Command via on_cc immobilize: target 7% vulnerable (received_amp) 4 s, refresh not stack; "
             "Control +20 AH on immobilizing abilities exposed as mandate_immobilize_haste (champion side)",
    FLOWING: "Rapids via on_ally_support: holder +40 AP +15 AH 6 s; ally buff in AllyBenefit",
    MOONSTONE: "ally-only chain heal/shield: amounts and chain target in AllyBenefit (no allies in 1v1)",
    ECHOES: "Soul Charges 30% pre-mitigation damage to champions, cap 80+10/level; consumed by "
            "on_ally_support into an ally heal in AllyBenefit",
    DAWNCORE: "First Light: +2% HSP and +10 AP per 100% base mana regen from items (prorated; "
              "Ctx lacks rune/shard base mana regen %)",
    ATLAS: "GP10 3; Shared Riches charges (first at +20 s, 1/20 s, max 3) with ally champion nearby: "
           "18 g on champion/structure damage (r2000), minion kill 18/20 g (r1300) + kill-gold redirect "
           "count, basic-attack execute below 50%/33.3% HP + AD; quest gold toward 400 -> atlas_done",
    BOUNTY: "GP10 5; Stealth Ward charges DEFERRED (vision, MODERN-009)",
    CELESTIAL: "GP10 9; Blessing of the Mountain state machine (pop on champion damage, 2 s linger, 50% "
               "slow r500 1.5 s, cd 18 restarted by champion damage); DR via "
               "celestial_champion_damage_mult (needs integrator, see module doc); wards DEFERRED",
    DREAM: "GP10 9; bubbles every 8 s granted to ally via on_ally_support (FlatDR/ProcDmg in "
           "AllyBenefit; ally-only effect); wards DEFERRED",
    ZAZZAK: "GP10 9; Void Explosion: ability damage to a champion -> 0.5 s delay, 10+15% AP+3% max HP "
            "magic in r250 (radius INFERRED), monster cap 300, cd 10; wards DEFERRED",
    SLEIGH: "GP10 9; Going Sledding via on_cc: heal 50 (+15/level from 7), 20% MS decaying 2.5 s, cd 30; "
            "most wounded ally within 1500 in state.sleigh_ally; wards DEFERRED",
}


class State(NamedTuple):
    atlas_seen: Any          # (C,) bool: held World Atlas last tick (charge timer start)
    atlas_charges: Any       # (C,) Shared Riches charges
    atlas_next_charge: Any   # (C,) seconds
    atlas_gold: Any          # (C,) gold earned from World Atlas (quest progress)
    atlas_done: Any          # (C,) bool: 400 reached; integrator transforms the item
    atlas_redirect: Any      # (C,) minion kills this tick whose kill gold goes to the nearest ally
    fanfare_until: Any       # (C,) Bandlepipes
    zeke_cd: Any             # (C,) Frostfire Tempest cooldown end
    zeke_ready_until: Any    # (C,) ready window end (NEVER = not readied)
    zeke_storm_until: Any    # (C,)
    zeke_next_tick: Any      # (C,)
    censer_until: Any
    flowing_until: Any
    echoes_charges: Any
    dream_ready_at: Any
    cel_cd_until: Any        # (C,) Blessing ready when now >= this and not popped
    cel_popped: Any          # (C,) bool
    cel_linger_until: Any
    zaz_cd: Any
    zaz_at: Any              # (C,) pending explosion time (inf = none)
    zaz_x: Any
    zaz_y: Any
    sleigh_cd: Any
    sleigh_start: Any
    sleigh_ally: Any         # (C,) int32 ally unit that received Going Sledding this tick (-1)
    mandate_until: Any       # (C, N) Command vulnerability end per target


def init(n_champions: int, n_units: int) -> State:
    z = jnp.zeros((n_champions,), jnp.float32)
    f = jnp.zeros((n_champions,), bool)
    never = z + NEVER
    return State(f, z, z, z, f, z, never, never, never, never, never, never, never, z, z, z, f, never,
                 never, z + jnp.inf, z, z, never, never, jnp.full((n_champions,), -1, jnp.int32),
                 jnp.full((n_champions, n_units), NEVER, jnp.float32))


class AllySupport(NamedTuple):
    """Holder c healed and/or shielded allied champion ``target`` this tick (excludes self)."""
    target: Any              # (C,) int32 unit index, -1 = none
    heal: Any                # (C,) amount healed (after HSP)
    shield: Any              # (C,) shield amount granted


class AllyBenefit(NamedTuple):
    """Ally-side results of ``on_ally_support`` for an integrator with allies, (C,)."""
    target: Any
    echoes_heal: Any         # heal for ``target`` (consumed Soul Charges)
    moonstone_target: Any    # int32 chain recipient (== target when no other ally in range)
    moonstone_heal: Any
    moonstone_shield: Any
    censer: Any              # bool: target gains 25% AS + 20 magic on-hit for 6 s
    flowing: Any             # bool: target gains 40 AP + 15 AH for 6 s
    dream: Any               # bool: target gains both Dream Bubbles for 3 s
    dream_flat_dr: Any       # Blue Bubble: next non-minion damage reduced by this
    dream_proc: Any          # Purple Bubble: bonus magic damage (x0.333 AoE vs non-champions)


# ---- small helpers -----------------------------------------------------------

def _allies(ctx, units):
    """(C, N) living allied champions other than the holder."""
    n = units.x.shape[0]
    return (units.team[None, :] == ctx.team[:, None]) & units.alive[None, :] \
        & (units.cls[None, :] == CLASS_CHAMPION) & (jnp.arange(n)[None, :] != ctx.unit[:, None])


def _ally_near(ctx, units, radius):
    d = jnp.sqrt((units.x[None, :] - ctx.x[:, None]) ** 2 + (units.y[None, :] - ctx.y[:, None]) ** 2)
    return jnp.any(_allies(ctx, units) & (d <= radius), axis=1)


def _most_wounded(mask, units):
    """(C,) int32 lowest-HP% unit in ``mask`` (-1 if none)."""
    frac = jnp.where(mask, units.hp[None, :] / jnp.maximum(units.max_hp[None, :], 1.0), jnp.inf)
    return jnp.where(jnp.any(mask, axis=1), jnp.argmin(frac, axis=1), -1).astype(jnp.int32)


def _enemy_champions(ctx, units):
    return enemy_mask(ctx, units) & (units.cls[None, :] == CLASS_CHAMPION)


def _sel(report, ctx, units):
    """(C, P) masks: packet from holder c that dealt damage, and dst class / team."""
    p, r = report.packets, report.resolved
    dst = jnp.clip(p.dst, 0, units.x.shape[0] - 1)
    src_is = (p.src[None, :] == ctx.unit[:, None]) & (p.valid & (r.final > 0.0))[None, :]
    enemy = units.team[dst][None, :] != ctx.team[:, None]
    return src_is & enemy, units.cls[dst][None, :]


def echoes_cap(level):
    return ECHOES_CAP_L1 + ECHOES_CAP_PER_LEVEL * (jnp.asarray(level, jnp.float32) - 1.0)


def dream_values(level):
    """(FlatDR, ProcDmg) of Dream Maker bubbles: level_bp(50, +12 at >=7), level_bp(40, +10 at >=7)."""
    return level_bp(50.0, 12.0, 7.0, level), level_bp(40.0, 10.0, 7.0, level)


def sleigh_heal(level):
    """BonusHealthBuff = level_bp(L1 50, +15/level at L>=7)."""
    return level_bp(50.0, 15.0, 7.0, level)


def locket_shield(level):
    """Devotion shield (deferred active): level_bp(290, +7/level at L>=9); 25% within 20 s."""
    return level_bp(290.0, 7.0, 9.0, level)


def redemption_heal(level):
    """Intervention heal (deferred active): lerp 150 -> 350 by the target's level; 10% max HP true."""
    lv = jnp.maximum(jnp.asarray(level, jnp.float32), 1.0)
    return dv(REDEMPTION, "HealMin") + (lv - 1.0) / 17.0 * (dv(REDEMPTION, "HealMax") - dv(REDEMPTION, "HealMin"))


def mikael_heal(level):
    """Purify heal (deferred active): lerp 100 -> 250 by level."""
    lv = jnp.maximum(jnp.asarray(level, jnp.float32), 1.0)
    return dv(MIKAELS, "HealAmountMin") + (lv - 1.0) / 17.0 * (dv(MIKAELS, "HealAmountMax") - dv(MIKAELS, "HealAmountMin"))


def knights_vow_redirect(ally_raw_damage, holder_hp, holder_max_hp, tethered):
    """Sacrifice: pre-mitigation damage moved from the Worthy ally to the holder (same type)."""
    ok = tethered & (holder_hp > KV_THRESHOLD * holder_max_hp)
    return jnp.where(ok, KV_REDIRECT * ally_raw_damage, 0.0)


def knights_vow_heal(ally_post_damage_to_champions, tethered):
    """Sacrifice: holder heals 12% of post-mitigation damage the Worthy ally deals to champions."""
    return jnp.where(tethered, KV_HEAL * ally_post_damage_to_champions, 0.0)


def mandate_immobilize_haste(own):
    """(C,) Control: extra ability haste for the holder's abilities that immobilize."""
    return jnp.where(holds(own, MANDATE), MANDATE_IMMOBILIZE_AH, 0.0)


def support_line_gold(own, dt, item_ids=tuple(GP10)):
    """(C,) GP10 gold of this tick for the support-line items held."""
    g = jnp.zeros(own.shape[:1], jnp.float32)
    for iid in item_ids:
        g = g + jnp.where(holds(own, iid), GP10[iid] / 10.0 * dt, 0.0)
    return g


def _mana_regen_pct(own):
    """(C,) bonus base-mana-regen ratio from held items (1.0 = 100%)."""
    col = jnp.asarray(catalog().arrays.stats[:, STAT_INDEX["percent_base_mana_regen"]], jnp.float32)
    return own.astype(jnp.float32) @ col


def celestial_blessed(state, own, ctx):
    ready = (ctx.now >= state.cel_cd_until) & ~state.cel_popped
    lingering = state.cel_popped & (ctx.now < state.cel_linger_until)
    return holds(own, CELESTIAL) & ctx.alive & (ready | lingering)


def celestial_champion_damage_mult(state, own, ctx):
    """(C,) multiplier for pre-mitigation damage from enemy champions to the holder."""
    dr = jnp.where(ctx.is_ranged, CEL_DR_RANGED, CEL_DR_MELEE)
    return jnp.where(celestial_blessed(state, own, ctx), 1.0 - dr, 1.0)


def defense(state, own, ctx):
    """Celestial Opposition blessing: reduces damage from enemy champions only."""
    return neutral_defense(ctx.level.shape[0])._replace(
        champion_received_mult=celestial_champion_damage_mult(state, own, ctx))


def bandlepipes_aura(state, own, ctx, units):
    """(C, N) bonus AS granted to allied champions within 900 while the holder has Fanfare."""
    active = holds(own, BANDLEPIPES) & ctx.alive & (ctx.now < state.fanfare_until)
    near = in_circle(units, ctx.x, ctx.y, jnp.full(ctx.x.shape, BANDLE_AURA), edge=False)
    aspd = BANDLE_AS * jnp.where(ctx.is_ranged, BANDLE_AS_RANGED_MULT, 1.0)
    return jnp.where(_allies(ctx, units) & near & active[:, None], aspd[:, None], 0.0)


# ---- hooks -------------------------------------------------------------------

def stats(state: State, own, ctx) -> ItemStats:
    now = ctx.now
    fan = holds(own, BANDLEPIPES) & (now < state.fanfare_until)
    censer = holds(own, CENSER) & (now < state.censer_until)
    flowing = holds(own, FLOWING) & (now < state.flowing_until)
    dawn = jnp.where(holds(own, DAWNCORE), _mana_regen_pct(own), 0.0)
    sleigh_left = jnp.clip(1.0 - (now - state.sleigh_start) / SLEIGH_DURATION, 0.0, 1.0)
    sleigh_ms = jnp.where(holds(own, SLEIGH) & (now >= state.sleigh_start), SLEIGH_MS * sleigh_left, 0.0)
    bandle_as = BANDLE_AS * jnp.where(ctx.is_ranged, BANDLE_AS_RANGED_MULT, 1.0)
    return ItemStats(
        move_speed=jnp.where(fan, BANDLE_MS, 0.0),
        attack_speed=jnp.where(fan, bandle_as, 0.0) + jnp.where(censer, CENSER_AS, 0.0),
        ability_power=jnp.where(flowing, FLOWING_AP, 0.0) + DAWN_AP * dawn,
        ability_haste=jnp.where(flowing, FLOWING_AH, 0.0),
        heal_shield_power=DAWN_HSP * dawn,
        ultimate_haste=jnp.where(holds(own, ZEKES), ZEKE_ULT_HASTE, 0.0),
        percent_move_speed=sleigh_ms)


def debuffs(state: State, own, ctx, units):
    n = units.x.shape[0]
    marked = holds(own, MANDATE)[:, None] & (ctx.now < state.mandate_until)
    amp = jnp.where(jnp.any(marked, axis=0), MANDATE_AMP, 0.0)   # refresh, never stacks
    z = jnp.zeros((n,), jnp.float32)
    return Debuffs(z, z, z, z, amp.astype(jnp.float32), z, z)


def on_cc(state: State, own, ctx, units, cc: CC) -> tuple[State, Effects]:
    """Holder applied slow/immobilize (Bandlepipes, Imperial Mandate, Solstice Sleigh)."""
    c, n = ctx.level.shape[0], units.x.shape[0]
    champs = _enemy_champions(ctx, units)
    any_cc = jnp.any((cc.slowed | cc.immobilized) & champs, axis=1) & ctx.alive
    imm = cc.immobilized & champs & holds(own, MANDATE)[:, None] & ctx.alive[:, None]
    mandate_until = jnp.where(imm, ctx.now + MANDATE_DURATION, state.mandate_until)

    fan_dur = BANDLE_DURATION * jnp.where(ctx.is_ranged, BANDLE_RANGED_DURATION_MULT, 1.0)
    fanfare = jnp.where(any_cc & holds(own, BANDLEPIPES), ctx.now + fan_dur, state.fanfare_until)

    ally_ok = _ally_near(ctx, units, SLEIGH_RANGE) | (not SLEIGH_REQUIRES_ALLY)
    sled = any_cc & holds(own, SLEIGH) & (ctx.now >= state.sleigh_cd) & ally_ok
    d = jnp.sqrt((units.x[None, :] - ctx.x[:, None]) ** 2 + (units.y[None, :] - ctx.y[:, None]) ** 2)
    ally = _most_wounded(_allies(ctx, units) & (d <= SLEIGH_RANGE), units)
    state = state._replace(
        mandate_until=mandate_until, fanfare_until=fanfare,
        sleigh_cd=jnp.where(sled, ctx.now + SLEIGH_CD, state.sleigh_cd),
        sleigh_start=jnp.where(sled, ctx.now, state.sleigh_start),
        sleigh_ally=jnp.where(sled, ally, -1).astype(jnp.int32))
    return state, effects(c, n, heal=jnp.where(sled, sleigh_heal(ctx.level), 0.0))


def on_ally_support(state: State, own, ctx, units, ev: AllySupport):
    """Holder healed/shielded another allied champion (Censer, Flowing Water, Echoes, Moonstone, Dream)."""
    c, n = ctx.level.shape[0], units.x.shape[0]
    tgt = jnp.asarray(ev.target, jnp.int32)
    ti = jnp.clip(tgt, 0, n - 1)
    valid = (tgt >= 0) & (tgt != ctx.unit) & ctx.alive & (units.team[ti] == ctx.team) \
        & (units.cls[ti] == CLASS_CHAMPION) & ((ev.heal > 0.0) | (ev.shield > 0.0))
    total = ev.heal + ev.shield

    censer = valid & holds(own, CENSER)
    flowing = valid & holds(own, FLOWING)
    echo = valid & holds(own, ECHOES) & (total >= ECHOES_MIN_TRIGGER)
    echoes_heal = jnp.where(echo, ECHOES_CONVERSION * state.echoes_charges, 0.0)
    dream = valid & holds(own, DREAM) & (total >= DREAM_MIN_TRIGGER) & (ctx.now >= state.dream_ready_at)
    flat_dr, proc = dream_values(ctx.level)

    moon = valid & holds(own, MOONSTONE)
    tx, ty = unit_pos(units, ti)
    in_range = in_circle(units, tx, ty, jnp.full((c,), MOONSTONE_RANGE), edge=False)
    others = _allies(ctx, units) & in_range & (jnp.arange(n)[None, :] != ti[:, None])
    chain = _most_wounded(others, units)
    single = chain < 0
    m_heal = jnp.where(moon, ev.heal * jnp.where(single, MOONSTONE_SINGLE_HEAL, MOONSTONE_HEAL), 0.0)
    m_shield = jnp.where(moon, ev.shield * jnp.where(single, MOONSTONE_SINGLE_SHIELD, MOONSTONE_SHIELD), 0.0)

    state = state._replace(
        censer_until=jnp.where(censer, ctx.now + CENSER_DURATION, state.censer_until),
        flowing_until=jnp.where(flowing, ctx.now + FLOWING_DURATION, state.flowing_until),
        echoes_charges=jnp.where(echo, 0.0, state.echoes_charges),
        dream_ready_at=jnp.where(dream, ctx.now + DREAM_CD, state.dream_ready_at))
    out = AllyBenefit(jnp.where(valid, tgt, -1), echoes_heal, jnp.where(moon, jnp.where(single, tgt, chain), -1),
                      m_heal, m_shield, censer, flowing, dream,
                      jnp.where(dream, flat_dr, 0.0), jnp.where(dream, proc, 0.0))
    return state, effects(c, n), out


def on_hit(state: State, own, ctx, units, attack) -> tuple[State, Effects]:
    c, n = ctx.level.shape[0], units.x.shape[0]
    hit = attack.hit & ctx.alive
    tgt = jnp.maximum(attack.target, 0)
    # Ardent Censer: 20 magic on-hit while Sanctified.
    cen = hit & holds(own, CENSER) & (ctx.now < state.censer_until)
    p_cen = packets(cen, ctx.unit, tgt, CENSER_ONHIT, MAGIC, ON_HIT_ITEM, item=CENSER)
    # World Atlas execute (client ExecuteHealthThreshold = pct x target max HP + AD; needs an
    # allied champion within 1300 and a charge). The kill then consumes the charge in on_takedown.
    ti = jnp.clip(attack.target, 0, n - 1)
    pct = jnp.where(ctx.is_ranged, ATLAS_EXEC_RANGED, ATLAS_EXEC_MELEE)
    minion = (target_class(units, attack.target) == CLASS_MINION) & (units.team[ti] != ctx.team) \
        & units.alive[ti] & (attack.target >= 0)
    execute = hit & holds(own, ATLAS) & minion & (state.atlas_charges >= 1.0) \
        & _ally_near(ctx, units, ATLAS_ALLY_RADIUS_MINION) \
        & (units.hp[ti] <= pct * units.max_hp[ti] + ctx.total_ad)
    p_exe = packets(execute, ctx.unit, tgt, units.max_hp[ti] + ctx.total_ad, TRUE,
                    TAG_PROC | TAG_ITEM | PROP_EXECUTE, item=ATLAS)
    return state, effects(c, n, packets=concat_packets(p_cen, p_exe))


def on_cast(state: State, own, ctx, units, cast) -> tuple[State, Effects]:
    c, n = ctx.level.shape[0], units.x.shape[0]
    go = cast.started & cast.is_ultimate & holds(own, ZEKES) & ctx.alive & (ctx.now >= state.zeke_cd)
    state = state._replace(zeke_cd=jnp.where(go, ctx.now + ZEKE_CD, state.zeke_cd),
                           zeke_ready_until=jnp.where(go, ctx.now + ZEKE_READY, state.zeke_ready_until))
    return state, effects(c, n)


def on_damage(state: State, own, ctx, units, report) -> tuple[State, Effects]:
    c, n = ctx.level.shape[0], units.x.shape[0]
    p = report.packets
    mine, dcls = _sel(report, ctx, units)
    flags = p.flags[None, :]
    not_item = ~has(flags, TAG_ITEM) & ~has(flags, PROP_REACTIVE)
    to_champ = mine & (dcls == CLASS_CHAMPION)

    # Echoes of Helia: 30% of pre-mitigation damage to champions.
    gain = ECHOES_RATE * jnp.sum(jnp.where(to_champ, p.raw[None, :], 0.0), axis=1)
    charges = jnp.where(holds(own, ECHOES),
                        jnp.minimum(state.echoes_charges + gain, echoes_cap(ctx.level)), state.echoes_charges)

    # Zaz'Zak's: ability (not attack/on-hit/item) damage to a champion queues an explosion.
    ability = to_champ & not_item & ~has(flags, TAG_BASIC_ATTACK) & ~has(flags, TAG_ON_HIT)
    any_ab = jnp.any(ability, axis=1)
    if p.dst.shape[0] == 0:   # static: no packets this tick
        zt = jnp.zeros((c,), jnp.int32)
    else:
        zt = p.dst[jnp.argmax(ability, axis=1)]
    zx, zy = unit_pos(units, zt)
    zgo = any_ab & holds(own, ZAZZAK) & ctx.alive & (ctx.now >= state.zaz_cd) & jnp.isinf(state.zaz_at)

    # World Atlas Shared Riches: champion/structure damage by attack or ability, ally within 2000.
    riches = holds(own, ATLAS) & (state.atlas_charges >= 1.0) & _ally_near(ctx, units, ATLAS_ALLY_RADIUS) \
        & jnp.any(mine & not_item & ((dcls == CLASS_CHAMPION) | (dcls == CLASS_STRUCTURE)), axis=1)
    rgold = jnp.where(riches, jnp.where(ctx.is_ranged, ATLAS_GOLD_RANGED, ATLAS_GOLD_MELEE), 0.0)

    # Celestial Opposition: champion damage pops the blessing / restarts the cooldown.
    src_cls = units.cls[jnp.clip(p.src, 0, n - 1)]
    src_team = units.team[jnp.clip(p.src, 0, n - 1)]
    champ_hit = jnp.any((p.dst[None, :] == ctx.unit[:, None]) & p.valid[None, :]
                                    & (src_cls == CLASS_CHAMPION)[None, :]
                                    & (report.resolved.final > 0.0)[None, :]
                                    & (src_team[None, :] != ctx.team[:, None]), axis=1)
    cel = holds(own, CELESTIAL) & champ_hit
    ready = (ctx.now >= state.cel_cd_until) & ~state.cel_popped
    pop = cel & ready & ctx.alive
    on_cd = cel & ~ready & ~state.cel_popped
    state = state._replace(
        echoes_charges=charges,
        zaz_at=jnp.where(zgo, ctx.now + ZAZ_DELAY, state.zaz_at),
        zaz_x=jnp.where(zgo, zx, state.zaz_x), zaz_y=jnp.where(zgo, zy, state.zaz_y),
        zaz_cd=jnp.where(zgo, ctx.now + ZAZ_CD, state.zaz_cd),
        atlas_charges=state.atlas_charges - riches.astype(jnp.float32),
        atlas_gold=state.atlas_gold + rgold,
        atlas_done=state.atlas_done | (holds(own, ATLAS) & (state.atlas_gold + rgold >= ATLAS_QUEST_GOLD)),
        cel_popped=state.cel_popped | pop,
        cel_linger_until=jnp.where(pop, ctx.now + CEL_LINGER, state.cel_linger_until),
        cel_cd_until=jnp.where(on_cd, ctx.now + CEL_CD, state.cel_cd_until))
    return state, effects(c, n, gold=rgold)


def on_takedown(state: State, own, ctx, units, kills) -> tuple[State, Effects]:
    c, n = ctx.level.shape[0], units.x.shape[0]
    minions = kills.killed_units & (units.cls[None, :] == CLASS_MINION)
    cannon = jnp.sum(minions & units.is_siege_or_super[None, :], axis=1).astype(jnp.float32)
    normal = jnp.sum(minions & ~units.is_siege_or_super[None, :], axis=1).astype(jnp.float32)
    ok = holds(own, ATLAS) & _ally_near(ctx, units, ATLAS_ALLY_RADIUS_MINION)
    avail = jnp.where(ok, jnp.floor(state.atlas_charges), 0.0)
    use_c = jnp.minimum(cannon, avail)
    use_n = jnp.minimum(normal, avail - use_c)
    gold = ATLAS_CANNON_GOLD * use_c + ATLAS_MINION_GOLD * use_n
    total = state.atlas_gold + gold
    state = state._replace(atlas_charges=state.atlas_charges - use_c - use_n, atlas_gold=total,
                           atlas_done=state.atlas_done | (holds(own, ATLAS) & (total >= ATLAS_QUEST_GOLD)),
                           atlas_redirect=use_c + use_n)
    return state, effects(c, n, gold=gold)


def periodic(state: State, own, ctx, units) -> tuple[State, Effects]:
    c, n = ctx.level.shape[0], units.x.shape[0]
    now = ctx.now
    parts = []

    # Gold generation (all support-line items of this module) and World Atlas charges/quest.
    gold = support_line_gold(own, ctx.dt)
    atlas = holds(own, ATLAS)
    fresh = atlas & ~state.atlas_seen
    nxt = jnp.where(fresh, now + ATLAS_FIRST_CHARGE, state.atlas_next_charge)
    charges = jnp.where(fresh, 0.0, state.atlas_charges)
    grow = atlas & (charges < ATLAS_MAX_CHARGES) & (now >= nxt)
    charges = charges + grow.astype(jnp.float32)
    nxt = jnp.where(grow, nxt + ATLAS_CHARGE_CD, nxt)
    nxt = jnp.where(atlas & (charges >= ATLAS_MAX_CHARGES), now + ATLAS_CHARGE_CD, nxt)  # paused while full
    atlas_gold = state.atlas_gold + jnp.where(atlas, GP10[ATLAS] / 10.0 * ctx.dt, 0.0)
    state = state._replace(atlas_seen=atlas, atlas_charges=charges, atlas_next_charge=nxt,
                           atlas_gold=atlas_gold,
                           atlas_done=state.atlas_done | (atlas & (atlas_gold >= ATLAS_QUEST_GOLD)))
    parts.append(effects(c, n, gold=gold))

    # Zeke's Convergence: readied storm summons on contact or at window end.
    radius = jnp.full((c,), ZEKE_RADIUS)
    champs_in = jnp.any(in_circle(units, ctx.x, ctx.y, radius) & _enemy_champions(ctx, units), axis=1)
    readied = holds(own, ZEKES) & ctx.alive & (state.zeke_ready_until > NEVER / 2)
    summon = readied & (champs_in | (now >= state.zeke_ready_until))
    storm_until = jnp.where(summon, now + ZEKE_DURATION, state.zeke_storm_until)
    next_tick = jnp.where(summon, now, state.zeke_next_tick)
    ready_until = jnp.where(summon, NEVER, state.zeke_ready_until)
    storming = holds(own, ZEKES) & ctx.alive & (now < storm_until)
    victims = in_circle(units, ctx.x, ctx.y, radius) & enemy_mask(ctx, units) \
        & ((units.cls[None, :] == CLASS_CHAMPION) | (units.cls[None, :] == CLASS_MONSTER))
    tick = storming & (now >= next_tick)
    p_zeke = packets(victims & tick[:, None], ctx.unit[:, None], jnp.arange(n)[None, :], ZEKE_DPS * ZEKE_TICK,
                     MAGIC, TAG_AOE | TAG_PERIODIC | TAG_ITEM, item=ZEKES)
    next_tick = jnp.where(tick, next_tick + ZEKE_TICK, next_tick)
    zslow = victims & storming[:, None]
    any_z = jnp.any(zslow, axis=0)
    parts.append(effects(c, n, packets=p_zeke, slow=jnp.where(any_z, ZEKE_SLOW, 0.0),
                         slow_duration=jnp.where(any_z, ZEKE_SLOW_REFRESH, 0.0)))
    state = state._replace(zeke_storm_until=storm_until, zeke_next_tick=next_tick, zeke_ready_until=ready_until)

    # Zaz'Zak's Realmspike: pending explosion.
    boom = holds(own, ZAZZAK) & (now >= state.zaz_at)
    hit = in_circle(units, state.zaz_x, state.zaz_y, jnp.full((c,), ZAZ_RADIUS)) & enemy_mask(ctx, units) \
        & (units.cls[None, :] != CLASS_STRUCTURE) & boom[:, None]
    dmg = ZAZ_BASE + ZAZ_AP * ctx.ap[:, None] + ZAZ_PCT * units.max_hp[None, :]
    dmg = jnp.where(units.cls[None, :] == CLASS_MONSTER, jnp.minimum(dmg, ZAZ_MONSTER_CAP), dmg)
    parts.append(effects(c, n, packets=packets(hit, ctx.unit[:, None], jnp.arange(n)[None, :], dmg, MAGIC,
                                               TAG_AOE | TAG_PROC | TAG_ITEM, item=ZAZZAK)))
    state = state._replace(zaz_at=jnp.where(now >= state.zaz_at, jnp.inf, state.zaz_at))

    # Celestial Opposition: linger end -> shockwave slow and cooldown.
    end = holds(own, CELESTIAL) & state.cel_popped & (now >= state.cel_linger_until)
    wave = in_circle(units, ctx.x, ctx.y, jnp.full((c,), CEL_RADIUS)) & enemy_mask(ctx, units) \
        & (units.cls[None, :] != CLASS_STRUCTURE) & (end & ctx.alive)[:, None]
    any_w = jnp.any(wave, axis=0)
    parts.append(effects(c, n, slow=jnp.where(any_w, CEL_SLOW, 0.0),
                         slow_duration=jnp.where(any_w, CEL_SLOW_DURATION, 0.0)))
    state = state._replace(cel_popped=state.cel_popped & ~end,
                           cel_cd_until=jnp.where(end, state.cel_linger_until + CEL_CD, state.cel_cd_until))

    # Slows from this module feed the CC-applied passives.
    state, cc_eff = on_cc(state, own, ctx, units,
                          CC(zslow | wave, jnp.zeros((c, n), bool)))
    parts.append(cc_eff)
    return state, merge_effects(parts, c, n)
