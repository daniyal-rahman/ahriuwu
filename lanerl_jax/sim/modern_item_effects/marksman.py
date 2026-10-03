"""Marksman / lethality item passives (26.19, client 16.19.8230722).

Values come from the client item data (``dv``); behaviour not encoded in the
data follows the wiki pass text quoted in docs/modern/ITEMS_CATALOG.md.
Every INFERRED default is tagged where it is chosen.

Shared Energized (ITEMS.md §6.8, RUNES.md §3.4): one 0..100 charge per
holder of any Energized item (RFC, Statikk, Stormrazor, Voltaic). +6 per
basic attack on-attack (+9 more with Statikk Electroshock), +1 per 24 units
moved (``ctx.moved``). An attack launched at 100 is Energized; its on-hit
consumes all 100 and fires every owned Energized effect.

Integrator obligations the hook protocol cannot express (read the helpers at
the bottom of this file):

* ``extra_on_hit_targets``: Runaan's bolts and Statikk secondary bounces
  apply on-hit effects. Run ``on_hit`` dispatch for those (holder, unit)
  pairs with ``Attack.raw = 0`` (the bolt/bounce damage packets themselves
  are emitted here).
* ``phantom_hit_due``: Guinsoo's Phantom Hit. Re-run ``on_hit`` dispatch for
  the same target (``raw = 0``, wiki delay 0.15 s may be collapsed).
* ``basic_attack_amp``: Hexoptics Magnification only amplifies basic attacks;
  add it to the ``amp`` of the holder's base-attack packet (not ``dealt_amp``).
* ``attack_range_bonus``: RFC Sharpshooter and Hexoptics Arcane Aim range.
* ``basic_cooldown_scale``: Navori multiplies remaining Q/W/E cooldowns.
* ``ult_refund_fraction``: Axiom Arc refunds that fraction of the
  ultimate's total cooldown.
* ``shield_reaver``: Serpent's Fang shield reduction on targets.
* ``attack_mods``: Fiendhunter forces a crit with ``crit_scale = 0.8``. The
  scale must be applied only when the natural roll failed; a natural crit
  stays a full crit (and gets the 15% true bonus emitted in ``on_hit``).
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from ..modern_damage import (CLASS_CHAMPION, CLASS_STRUCTURE, MAGIC, ON_HIT_ITEM, PHYSICAL,
                             PROP_CRIT, PROP_EXECUTE, PROP_LIFESTEAL, TAG_ACTIVE_SPELL,
                             TAG_AOE, TAG_ITEM, TAG_PROC, TRUE, concat_packets, has, packets,
                             shield_value)
from ..modern_item_data import ItemStats, level_bp, ranged_mult
from .core import (AttackMods, Effects, StatusFlags, dealt_by_holder, dv, effects, enemy_mask,
                   hit_by_holder, holds, holds_any, in_circle, nearest_k, onehot_units,
                   shield_grants, target_class, unit_pos)

RECURVE, FIENDHUNTER, HEXOPTICS, YUNTAL, LDR, PHANTOM, BLOODTHIRSTER = 1043, 2512, 2523, 3032, 3036, 3046, 3072
RUNAANS, STATIKK, RFC, STORMRAZOR, GUINSOO, SLINGSHOT, NOONQUIVER = 3085, 3087, 3094, 3095, 3124, 3144, 6670
KRAKEN, NAVORI, COLLECTOR, YOUMUU, HUBRIS, AXIOM, UMBRAL, SERPENT, VOLTAIC = \
    6672, 6675, 6676, 3142, 6697, 6696, 3179, 6695, 6699
ENERGIZED_ITEMS = (RFC, STATIKK, STORMRAZOR, VOLTAIC)

NEVER = -1e9

# ---- client values ----------------------------------------------------------
RECURVE_DMG = dv(RECURVE, "OnHitDamage")
GUINSOO_DMG = dv(GUINSOO, "OnHitDamage")
GUINSOO_AS = dv(GUINSOO, "AttackSpeedPerStack")
GUINSOO_MAX = dv(GUINSOO, "MaxStacks")
GUINSOO_DUR = dv(GUINSOO, "BuffDuration")
GUINSOO_PHANTOM_MAX = 2.0          # wiki: 2 Phantom stacks, the next attack fires (tooltip "every third")
KRAKEN_COUNT = dv(KRAKEN, "AttackCount")
KRAKEN_DUR = dv(KRAKEN, "BuffDuration")
KRAKEN_MAX_AMP = dv(KRAKEN, "MaxAmpNumber")
KRAKEN_RANGED = dv(KRAKEN, "RangedDamageMultiplier")
YT_DUR, YT_CD = dv(YUNTAL, "ASDuration"), dv(YUNTAL, "Cooldown")
YT_AS, YT_CRIT_MAX = dv(YUNTAL, "ASMod"), dv(YUNTAL, "CritMax") / 100.0
YT_AA_CDR, YT_CRIT_CDR = dv(YUNTAL, "AACDR"), dv(YUNTAL, "CritCDR")
# CritPerStackCalc resolves to unnamed hashes (=0); the named values 0.4 (melee, percent)
# and StackRangedMultiplier 0.5 match the wiki 0.4% / 0.2%.
YT_CRIT_PER = dv(YUNTAL, "CritPerStackMelee") / 100.0
YT_RANGED = dv(YUNTAL, "StackRangedMultiplier")
FH_HASTE, FH_CD, FH_DUR = dv(FIENDHUNTER, "UltimateHaste"), dv(FIENDHUNTER, "Cooldown"), dv(FIENDHUNTER, "Duration")
FH_AS, FH_N = dv(FIENDHUNTER, "BonusAS"), dv(FIENDHUNTER, "NumberOfAttacks")
FH_CRIT, FH_TRUE = dv(FIENDHUNTER, "CritModifier"), dv(FIENDHUNTER, "BonusTrueDamage")
LDR_MAX, LDR_HP = dv(LDR, "MaxBonusDamagePercent"), dv(LDR, "MaxBonusHealth")
HEX_AMP, HEX_RANGE = dv(HEXOPTICS, "MaxDamageAmp"), dv(HEXOPTICS, "MaxRange")
HEX_EXTRA, HEX_DUR = dv(HEXOPTICS, "ExtraRange"), dv(HEXOPTICS, "Duration")
TAKEDOWN_WINDOW = dv(HEXOPTICS, "TakedownWindow")      # 3 s, same for Hubris/Axiom
RUNAAN_RATIO = 0.65                # calc BoltDamage = 0.65 x AD (no named data value)
RUNAAN_EXTRA = dv(RUNAANS, "ExtraRangeOnBoltCheck")
RUNAAN_BOLTS_RANGED, RUNAAN_BOLTS_MELEE = 2, 1           # calc ChampRange 2 / 1
RUNAAN_ATTACK_RANGE = 550.0        # INFERRED: holder range is not in Ctx; integrator may pass its own
STATIKK_BONUS = dv(STATIKK, "BonusEnergizedStacks")
STATIKK_CHAMP, STATIKK_OTHER = dv(STATIKK, "ChainDamage"), dv(STATIKK, "NonChampChainDamage")
STATIKK_RANGE = dv(STATIKK, "BounceRange")
STATIKK_MAX_BOUNCES = 8            # BounceCount at L20
RFC_DMG, RFC_RANGE_PCT, RFC_RANGE_MAX = dv(RFC, "BonusDamage"), dv(RFC, "RangePercentIncrease"), dv(RFC, "MaxRangeIncrease")
STORM_DMG = 100.0                  # calc TotalProcDamage = 100 (no named data value)
STORM_MS, STORM_DUR = dv(STORMRAZOR, "BuffStrength"), dv(STORMRAZOR, "BuffDuration")
VOLT_PCT_M, VOLT_PCT_R = dv(VOLTAIC, "PercentCurrentHPMelee") / 100.0, dv(VOLTAIC, "PercentCurrentHPRanged") / 100.0
VOLT_LETH_M, VOLT_LETH_R = dv(VOLTAIC, "LethalityBonusModMelee"), dv(VOLTAIC, "LethalityBonusModRanged")
VOLT_DUR, VOLT_CAP = dv(VOLTAIC, "LethalityBonusDuration"), dv(VOLTAIC, "NonChampCap")
SLING_DMG = 40.0                   # calc DamageAmount = 40
SLING_CD = dv(SLINGSHOT, "Cooldown")
SLING_ATTACK_CDR = 1.0             # tooltip "Attacks reduce this cooldown by 1 second"
NAVORI_CDR = dv(NAVORI, "CDRAmount")
COLLECTOR_THRESHOLD, COLLECTOR_GOLD = dv(COLLECTOR, "ExecuteThreshold"), dv(COLLECTOR, "GoldAmount")
YOUMUU_OOC_MS, YOUMUU_TIMER = dv(YOUMUU, "BaseOOCMS"), dv(YOUMUU, "CombatTimer")
YOUMUU_RANGED = 0.5                # calc OOCMS mRangedMultiplier
HUBRIS_BASE, HUBRIS_PER, HUBRIS_DUR = dv(HUBRIS, "BaseADBonus"), dv(HUBRIS, "ADPerStatue"), dv(HUBRIS, "BuffDuration")
AXIOM_BASE = dv(AXIOM, "UltimateRefundBase") / 100.0
AXIOM_PER_LETHALITY = 0.25 / 100.0  # calc 0.25 x Lethality (percent)
SERPENT_MELEE, SERPENT_RANGED = dv(SERPENT, "ShieldShred") / 100.0, dv(SERPENT, "ShieldShredRange") / 100.0
SERPENT_DUR = dv(SERPENT, "DebuffDuration")
ENERGY_MAX, ENERGY_PER_ATTACK, ENERGY_UNITS_PER_STACK = 100.0, 6.0, 24.0   # wiki Energized info
ICHOR_DURATION = 1e6               # "lasts until destroyed"

COVERAGE = {
    RECURVE: "Sting: 15 physical on-hit (life steal)",
    FIENDHUNTER: "Night Vigil +30 ult haste; Opening Barrage after R (cd 45): 3 attacks in 8 s +50% AS, "
                 "forced crit x0.8 (attack_mods), natural crit +15% raw as true",
    HEXOPTICS: "Magnification up to +10% at 500 edge range (basic_attack_amp helper); Arcane Aim +100 range "
               "8 s after champion takedown within 3 s (attack_range_bonus helper)",
    YUNTAL: "Practice Makes Lethal +0.4%/0.2% crit per attack up to 25%; Flurry +30% AS 6 s on champion "
            "attack (cd 30, -1 s per hit, -2 s per crit)",
    LDR: "Giant Slayer dealt_amp 15% x min(bonus HP / 1500, 1) vs champions",
    PHANTOM: "Spectral Waltz: ghosted (status)",
    BLOODTHIRSTER: "Ichorshield: vamp heal beyond missing HP -> shield, total capped 165 (+15/lvl from 9)",
    RUNAANS: "Wind's Fury: on-attack bolts (0.65 AD, crits) to 2 (melee 1) nearest enemies in front; "
             "on-hit application via extra_on_hit_targets",
    STATIKK: "Electroshock +9 Energize per attack; Electrospark chain 60 (90 non-champion) magic to "
             "4/5/6/7/8 targets within 500; secondary on-hits via extra_on_hit_targets",
    RFC: "Energized Sharpshooter: 40 magic on-hit; +35% range capped 150 (attack_range_bonus helper)",
    STORMRAZOR: "Energized Bolt: 100 magic on-hit, +45% MS 1.5 s",
    GUINSOO: "Wrath 30 magic on-hit (life steal); Seething Strike +8% AS x4 for 4 s; Phantom Hit every "
             "third attack at max stacks (phantom_hit_due helper)",
    SLINGSHOT: "Bullseye: damaging a champion deals 40 magic (cd 40, -1 s per attack)",
    NOONQUIVER: "stats only (no passive data values in client 16.19; should move to STATS_ONLY)",
    KRAKEN: "Bring It Down: 3rd attack 150 (+5/lvl from 9, ranged x0.8) physical, +0-75% by target "
            "missing HP; stacks on-hit melee / on-attack ranged, 4 s",
    NAVORI: "Transcendence: on-attack basic-ability cooldowns x0.85 (basic_cooldown_scale helper)",
    COLLECTOR: "Death: execute champions left below 5% (PROP_EXECUTE, on_damage); Taxes +25 gold per kill",
    YOUMUU: "Haunt: +20 (ranged 10) flat MS out of champion combat 3 s; Wraith Step active in modern_item_actives",
    HUBRIS: "Eminence: champion takedown within 3 s of damage -> +12 +3 per stack AD for 90 s, permanent stack",
    AXIOM: "Flux: champion takedown within 3 s -> ult refund 10% + 0.25% per lethality (ult_refund_fraction helper)",
    UMBRAL: "stats only; Nightstalker and Blackout need vision -> DEFERRED (MODERN-009)",
    SERPENT: "Shield Reaver: champions damaged get 3 s venom, shields gained -50% (ranged 35%), existing "
             "non-magic shields reduced once (shield_reaver helper)",
    VOLTAIC: "Energized Firmament: 9%/7% target current HP physical (cap 200 non-champion), +15/12 lethality "
             "4 s; Galvanize: ability damage to a champion fires Energized",
}


class State(NamedTuple):
    energy: Any             # (C,) Energize charge 0..100
    en_pending: Any         # (C,) bool: the launched attack is Energized
    guinsoo_stacks: Any     # (C,)
    guinsoo_until: Any
    phantom: Any            # (C,) Phantom stacks 0..2
    phantom_pending: Any    # (C,) bool: launched attack fires a Phantom Hit
    phantom_at: Any         # (C,) time a Phantom Hit became due (helper)
    kraken_stacks: Any
    kraken_until: Any
    kraken_pending: Any     # (C,) bool, ranged: launched attack consumes the stacks
    yt_crit: Any            # (C,) permanent crit from Practice Makes Lethal
    yt_cd_until: Any
    yt_as_until: Any
    fh_charges: Any         # (C,) Opening Barrage attacks left
    fh_until: Any
    fh_cd_until: Any
    fh_attack: Any          # (C,) bool: launched attack is empowered
    sling_cd_until: Any
    storm_until: Any
    volt_until: Any
    hubris_stacks: Any
    hubris_until: Any
    hex_until: Any
    axiom_refund: Any       # (C,) fraction granted at axiom_at
    axiom_at: Any
    navori_at: Any          # (C,) time of the last Navori on-attack
    last_dmg: Any           # (C, N) time holder last damaged enemy champion n
    champ_combat: Any       # (C,) last damage dealt to / taken from a champion
    venom_until: Any        # (C, N)
    venom_fresh: Any        # (C, N) bool: newly afflicted at venom_fresh_at
    venom_fresh_at: Any     # (C,)
    ichor: Any              # (C,) tracked Ichorshield amount
    extra_hits: Any         # (C, N) bool: bolt / bounce on-hit targets at extra_at
    extra_at: Any           # (C,)


def init(n_champions: int, n_units: int) -> State:
    c, n = n_champions, n_units
    z = jnp.zeros((c,), jnp.float32)
    never = z + NEVER
    f = jnp.zeros((c,), bool)
    zn = jnp.zeros((c, n), jnp.float32)
    fn = jnp.zeros((c, n), bool)
    return State(z, f, z, never, z, f, never, z, never, f, z, never, never, z, never, never, f, never,
                 never, never, z, never, never, z, never, never, zn + NEVER, never, zn + NEVER, fn, never,
                 z, fn, never)


# ---- shared helpers ---------------------------------------------------------

def _enemy_champs(ctx, units):
    return (units.cls[None, :] == CLASS_CHAMPION) & (units.team[None, :] != ctx.team[:, None])


def _rm(ctx, value):
    return ranged_mult(ctx.is_ranged, value)


def statikk_bounces(level):
    """BounceCount: 4, +1 once at levels 6, 10, 14 and 20."""
    lv = jnp.asarray(level, jnp.float32)
    return 4.0 + sum((lv >= k).astype(jnp.float32) for k in (6, 10, 14, 20))


def _energized(state: State, own, ctx, units, fire, tgt):
    """Fire every owned Energized effect at ``tgt`` (C,) for holders ``fire``."""
    c, n = ctx.level.shape[0], units.x.shape[0]
    tgt = jnp.maximum(tgt, 0)
    tcls = target_class(units, tgt)
    champ = tcls == CLASS_CHAMPION
    out = []
    # RFC Sharpshooter / Stormrazor Bolt: flat magic on-hit.
    rfc = fire & holds(own, RFC)
    out.append(packets(rfc, ctx.unit, tgt, RFC_DMG, MAGIC, ON_HIT_ITEM, item=RFC))
    storm = fire & holds(own, STORMRAZOR)
    out.append(packets(storm, ctx.unit, tgt, STORM_DMG, MAGIC, ON_HIT_ITEM, item=STORMRAZOR))
    # Voltaic Firmament: % current HP (before this attack), capped vs non-champions.
    volt = fire & holds(own, VOLTAIC)
    vdmg = jnp.where(ctx.is_ranged, VOLT_PCT_R, VOLT_PCT_M) * units.hp[tgt]
    vdmg = jnp.where(champ, vdmg, jnp.minimum(vdmg, VOLT_CAP))
    out.append(packets(volt, ctx.unit, tgt, vdmg, PHYSICAL, ON_HIT_ITEM, item=VOLTAIC))
    # Statikk Electrospark: chain from the target to the nearest unhit enemy within 500.
    st = fire & holds(own, STATIKK) & (tcls != CLASS_STRUCTURE)
    count = statikk_bounces(ctx.level)
    enemies = enemy_mask(ctx, units) & (units.cls[None, :] != CLASS_STRUCTURE)
    hit = onehot_units(tgt, n) & st[:, None]
    cx, cy = unit_pos(units, tgt)
    for k in range(1, STATIKK_MAX_BOUNCES):
        cand = enemies & ~hit & in_circle(units, cx, cy, jnp.full((c,), STATIKK_RANGE))
        d = jnp.sqrt((units.x[None, :] - cx[:, None]) ** 2 + (units.y[None, :] - cy[:, None]) ** 2)
        pick = nearest_k(d, cand, 1) & (st & (k < count))[:, None]
        hit = hit | pick
        go = jnp.any(pick, axis=1)
        nxt = jnp.argmax(pick, axis=1)
        cx = jnp.where(go, units.x[nxt], cx)
        cy = jnp.where(go, units.y[nxt], cy)
    chain_dmg = jnp.where(units.cls == CLASS_CHAMPION, STATIKK_CHAMP, STATIKK_OTHER)[None, :]
    primary = onehot_units(tgt, n)
    flags = jnp.where(primary, ON_HIT_ITEM, TAG_PROC | TAG_ITEM | TAG_AOE)
    out.append(packets(hit, ctx.unit[:, None], jnp.arange(n)[None, :], chain_dmg, MAGIC, flags, item=STATIKK))
    secondary = hit & ~primary
    state = state._replace(
        energy=jnp.where(fire, 0.0, state.energy),
        en_pending=jnp.where(fire, False, state.en_pending),
        storm_until=jnp.where(storm, ctx.now + STORM_DUR, state.storm_until),
        volt_until=jnp.where(volt, ctx.now + VOLT_DUR, state.volt_until),
        extra_hits=jnp.where((ctx.now == state.extra_at)[:, None], state.extra_hits | secondary, secondary),
        extra_at=jnp.where(jnp.any(secondary, axis=1), ctx.now, state.extra_at))
    return state, concat_packets(*out)


# ---- hooks ------------------------------------------------------------------

def stats(state: State, own, ctx) -> ItemStats:
    now = ctx.now
    guinsoo = jnp.where(holds(own, GUINSOO) & (now < state.guinsoo_until), GUINSOO_AS * state.guinsoo_stacks, 0.0)
    flurry = jnp.where(holds(own, YUNTAL) & (now < state.yt_as_until), YT_AS, 0.0)
    barrage = jnp.where(holds(own, FIENDHUNTER) & (state.fh_charges > 0) & (now < state.fh_until), FH_AS, 0.0)
    crit = jnp.where(holds(own, YUNTAL), state.yt_crit, 0.0)
    storm_ms = jnp.where(holds(own, STORMRAZOR) & (now < state.storm_until), STORM_MS, 0.0)
    ooc = holds(own, YOUMUU) & (now - state.champ_combat >= YOUMUU_TIMER)
    youmuu = jnp.where(ooc, YOUMUU_OOC_MS * _rm(ctx, YOUMUU_RANGED), 0.0)
    volt = jnp.where(holds(own, VOLTAIC) & (now < state.volt_until),
                     jnp.where(ctx.is_ranged, VOLT_LETH_R, VOLT_LETH_M), 0.0)
    hubris = jnp.where(holds(own, HUBRIS) & (now < state.hubris_until),
                       HUBRIS_BASE + HUBRIS_PER * state.hubris_stacks, 0.0)
    ult = jnp.where(holds(own, FIENDHUNTER), FH_HASTE, 0.0)
    return ItemStats(attack_speed=guinsoo + flurry + barrage, crit_chance=crit, percent_move_speed=storm_ms,
                     move_speed=youmuu, lethality=volt, attack_damage=hubris, ultimate_haste=ult)


def status(state: State, own, ctx) -> StatusFlags:
    return StatusFlags(holds(own, PHANTOM) & ctx.alive)


def dealt_amp(state: State, own, ctx, units):
    frac = jnp.clip(units.bonus_hp / LDR_HP, 0.0, 1.0)
    return jnp.where(holds(own, LDR)[:, None] & _enemy_champs(ctx, units), LDR_MAX * frac[None, :], 0.0)


def attack_mods(state: State, own, ctx, units, target) -> AttackMods:
    ready = holds(own, FIENDHUNTER) & (state.fh_charges > 0) & (ctx.now < state.fh_until) & ctx.alive
    return AttackMods(ready, jnp.full(ready.shape, FH_CRIT, jnp.float32))


def on_cast(state: State, own, ctx, units, cast):
    c, n = ctx.level.shape[0], units.x.shape[0]
    go = cast.started & cast.is_ultimate & holds(own, FIENDHUNTER) & (ctx.now >= state.fh_cd_until) & ctx.alive
    state = state._replace(fh_charges=jnp.where(go, FH_N, state.fh_charges),
                           fh_until=jnp.where(go, ctx.now + FH_DUR, state.fh_until),
                           fh_cd_until=jnp.where(go, ctx.now + FH_CD, state.fh_cd_until))
    return state, effects(c, n)


def on_attack(state: State, own, ctx, units, attack):
    c, n = ctx.level.shape[0], units.x.shape[0]
    now = ctx.now
    la = attack.launched & ctx.alive
    tgt = jnp.maximum(attack.target, 0)
    vs_champ = (attack.target >= 0) & (target_class(units, tgt) == CLASS_CHAMPION)

    # Energized: an attack launched at 100 is Energized; otherwise it charges.
    has_en = holds_any(own, ENERGIZED_ITEMS)
    en_now = la & has_en & (state.energy >= ENERGY_MAX)
    gain = ENERGY_PER_ATTACK + jnp.where(holds(own, STATIKK), STATIKK_BONUS, 0.0)
    energy = jnp.where(la & has_en & ~en_now, jnp.minimum(ENERGY_MAX, state.energy + gain), state.energy)
    en_pending = jnp.where(la, en_now, state.en_pending)

    # Guinsoo's: refresh-all stacks; Phantom stacks once fully stacked (INFERRED L:
    # the attack reaching 4 stacks grants the first Phantom stack).
    g = la & holds(own, GUINSOO)
    alive_g = now < state.guinsoo_until
    gst = jnp.where(alive_g, state.guinsoo_stacks, 0.0)
    ph = jnp.where(alive_g, state.phantom, 0.0)
    fire_ph = g & (ph >= GUINSOO_PHANTOM_MAX)
    gst_new = jnp.minimum(GUINSOO_MAX, gst + 1.0)
    ph_new = jnp.where(fire_ph, 0.0, jnp.where(gst_new >= GUINSOO_MAX, jnp.minimum(GUINSOO_PHANTOM_MAX, ph + 1.0), ph))

    # Kraken (ranged): stacks on-attack; the 3rd attack consumes at launch, deals on-hit.
    kr = la & holds(own, KRAKEN) & ctx.is_ranged
    kst = jnp.where(now < state.kraken_until, state.kraken_stacks, 0.0)
    k_fire = kr & (kst >= KRAKEN_COUNT - 1)

    # Yun Tal: permanent crit per attack; Flurry on champion attacks.
    y = la & holds(own, YUNTAL)
    yt_crit = jnp.where(y, jnp.minimum(YT_CRIT_MAX, state.yt_crit + YT_CRIT_PER * _rm(ctx, YT_RANGED)), state.yt_crit)
    flurry = y & vs_champ & (now >= state.yt_cd_until)

    # Fiendhunter: consume one empowered attack.
    fh = la & (state.fh_charges > 0) & (now < state.fh_until) & holds(own, FIENDHUNTER)

    # Runaan's bolts (INFERRED: fired at launch, travel time ignored).
    run = la & holds(own, RUNAANS)
    enemies = enemy_mask(ctx, units) & (units.cls[None, :] != CLASS_STRUCTURE) & ~onehot_units(attack.target, n)
    hx, hy = unit_pos(units, ctx.unit)
    rx, ry = units.x[None, :] - hx[:, None], units.y[None, :] - hy[:, None]
    front = rx * ctx.facing_x[:, None] + ry * ctx.facing_y[:, None] >= 0.0
    reach = in_circle(units, hx, hy, jnp.full((c,), RUNAAN_ATTACK_RANGE + RUNAAN_EXTRA))
    dist = jnp.sqrt(rx ** 2 + ry ** 2)
    cand = enemies & front & reach
    two = nearest_k(dist, cand, RUNAAN_BOLTS_RANGED)
    one = nearest_k(dist, cand, RUNAAN_BOLTS_MELEE)
    bolts = jnp.where(ctx.is_ranged[:, None], two, one) & run[:, None]
    crit_mult = jnp.where(attack.is_crit, ctx.crit_damage, 1.0)
    bolt_flags = TAG_PROC | TAG_ITEM | PROP_LIFESTEAL | jnp.where(attack.is_crit, PROP_CRIT, 0)
    p_bolts = packets(bolts, ctx.unit[:, None], jnp.arange(n)[None, :], (RUNAAN_RATIO * ctx.total_ad * crit_mult)[:, None],
                      PHYSICAL, bolt_flags[:, None], item=RUNAANS)
    any_bolt = jnp.any(bolts, axis=1)

    navori = la & holds(own, NAVORI)
    state = state._replace(
        energy=energy, en_pending=en_pending,
        guinsoo_stacks=jnp.where(g, gst_new, state.guinsoo_stacks),
        guinsoo_until=jnp.where(g, now + GUINSOO_DUR, state.guinsoo_until),
        phantom=jnp.where(g, ph_new, state.phantom),
        phantom_pending=jnp.where(la, fire_ph, state.phantom_pending),
        kraken_stacks=jnp.where(kr, jnp.where(k_fire, 0.0, kst + 1.0), state.kraken_stacks),
        kraken_until=jnp.where(kr, now + KRAKEN_DUR, state.kraken_until),
        kraken_pending=jnp.where(la & ctx.is_ranged, k_fire, state.kraken_pending),
        yt_crit=yt_crit,
        yt_as_until=jnp.where(flurry, now + YT_DUR, state.yt_as_until),
        yt_cd_until=jnp.where(flurry, now + YT_CD, state.yt_cd_until),
        fh_attack=jnp.where(la, fh, state.fh_attack),
        fh_charges=jnp.where(fh, state.fh_charges - 1.0, state.fh_charges),
        sling_cd_until=jnp.where(la & holds(own, SLINGSHOT), state.sling_cd_until - SLING_ATTACK_CDR,
                                 state.sling_cd_until),
        navori_at=jnp.where(navori, now, state.navori_at),
        extra_hits=jnp.where(any_bolt[:, None], bolts, state.extra_hits),
        extra_at=jnp.where(any_bolt, now, state.extra_at))
    return state, effects(c, n, packets=p_bolts)


def on_hit(state: State, own, ctx, units, attack):
    c, n = ctx.level.shape[0], units.x.shape[0]
    now = ctx.now
    h = attack.hit & ctx.alive & (attack.target >= 0)
    tgt = jnp.maximum(attack.target, 0)
    out = []
    lsf = ON_HIT_ITEM | PROP_LIFESTEAL
    out.append(packets(h & holds(own, RECURVE), ctx.unit, tgt, RECURVE_DMG, PHYSICAL, lsf, item=RECURVE))
    out.append(packets(h & holds(own, GUINSOO), ctx.unit, tgt, GUINSOO_DMG, MAGIC, lsf, item=GUINSOO))

    # Kraken: melee stacks on-hit; ranged consumes the stacks reserved at launch.
    km = h & holds(own, KRAKEN) & ~ctx.is_ranged
    kst = jnp.where(now < state.kraken_until, state.kraken_stacks, 0.0)
    km_fire = km & (kst >= KRAKEN_COUNT - 1)
    kr_fire = h & holds(own, KRAKEN) & ctx.is_ranged & state.kraken_pending
    k_fire = km_fire | kr_fire
    missing = 1.0 - units.hp[tgt] / jnp.maximum(units.max_hp[tgt], 1.0)
    kdmg = level_bp(150.0, 5.0, 9.0, ctx.level) * _rm(ctx, KRAKEN_RANGED) \
        * (1.0 + (KRAKEN_MAX_AMP - 1.0) * jnp.clip(missing, 0.0, 1.0))
    out.append(packets(k_fire, ctx.unit, tgt, kdmg, PHYSICAL, lsf, item=KRAKEN))

    # Fiendhunter: a natural crit on an empowered attack adds 15% of its raw damage as true.
    # ``Attack.natural_crit`` distinguishes the crit roll from the forced crit; when the
    # integrator does not supply it, fall back to raw above the forced x0.8 value.
    forced_raw = ctx.total_ad * (1.0 + FH_CRIT * (ctx.crit_damage - 1.0))
    natural = attack.is_crit & (attack.raw > forced_raw * 1.001) if attack.natural_crit is None \
        else attack.natural_crit
    fh_true = h & state.fh_attack & natural
    out.append(packets(fh_true, ctx.unit, tgt, FH_TRUE * attack.raw, TRUE, TAG_PROC | TAG_ITEM, item=FIENDHUNTER))

    yt = h & holds(own, YUNTAL)
    yt_cd = jnp.where(yt, state.yt_cd_until - jnp.where(attack.is_crit, YT_CRIT_CDR, YT_AA_CDR), state.yt_cd_until)
    phantom_due = h & state.phantom_pending
    state = state._replace(
        kraken_stacks=jnp.where(km, jnp.where(km_fire, 0.0, kst + 1.0), state.kraken_stacks),
        kraken_until=jnp.where(km, now + KRAKEN_DUR, state.kraken_until),
        kraken_pending=jnp.where(kr_fire, False, state.kraken_pending),
        fh_attack=jnp.where(h, False, state.fh_attack),
        yt_cd_until=yt_cd,
        phantom_pending=jnp.where(h, False, state.phantom_pending),
        phantom_at=jnp.where(phantom_due, now, state.phantom_at))

    # Energized attack lands.
    en = h & state.en_pending & (state.energy >= ENERGY_MAX) & holds_any(own, ENERGIZED_ITEMS)
    state, p_en = _energized(state, own, ctx, units, en, tgt)
    return state, effects(c, n, packets=concat_packets(*out, p_en))


def on_damage(state: State, own, ctx, units, report):
    c, n = ctx.level.shape[0], units.x.shape[0]
    now = ctx.now
    p, r = report.packets, report.resolved
    champs = _enemy_champs(ctx, units)
    dealt = dealt_by_holder(report, ctx, n)
    hit_champ = (dealt > 0.0) & champs
    last_dmg = jnp.where(hit_champ, now, state.last_dmg)
    src_cls = units.cls[jnp.clip(p.src, 0, n - 1)]
    from_champ = p.valid & (r.final > 0.0) & (src_cls == CLASS_CHAMPION) & (p.src != p.dst)
    took = jnp.any((p.dst[None, :] == ctx.unit[:, None]) & from_champ[None, :], axis=1)
    combat = jnp.any(hit_champ, axis=1) | took
    champ_combat = jnp.where(combat, now, state.champ_combat)

    # Serpent's Fang venom (fresh = not already afflicted by any holder).
    sf = holds(own, SERPENT)[:, None] & hit_champ
    already = jnp.any(now < state.venom_until, axis=0)
    fresh = sf & ~already[None, :]
    venom_until = jnp.where(sf, now + SERPENT_DUR, state.venom_until)

    # Scout's Slingshot Bullseye: first enemy champion damaged by another packet.
    other = p.valid & (r.final > 0.0) & (p.item != SLINGSHOT)
    sling_mask = hit_by_holder(report, ctx, n, other) & champs
    sling = holds(own, SLINGSHOT) & ctx.alive & (now >= state.sling_cd_until) & jnp.any(sling_mask, axis=1)
    sling_tgt = jnp.argmax(sling_mask, axis=1)
    p_sling = packets(sling, ctx.unit, sling_tgt, SLING_DMG, MAGIC, TAG_PROC | TAG_ITEM, item=SLINGSHOT)

    # The Collector: execute champions this holder left below 5% max HP.
    hp_after, max_after = r.hp[None, :], r.max_hp[None, :]
    execute = holds(own, COLLECTOR)[:, None] & hit_champ & (hp_after > 0.0) & (hp_after < COLLECTOR_THRESHOLD * max_after)
    p_exec = packets(execute, ctx.unit[:, None], jnp.arange(n)[None, :], hp_after, TRUE,
                     PROP_EXECUTE | TAG_ITEM, item=COLLECTOR)

    # Bloodthirster Ichorshield: vamp heal beyond missing HP (post-damage HP). The report's
    # heal includes omnivamp and ignores heal modifiers (approximation). The tracked
    # Ichorshield absorbs last (infinite expiry), so it is min(tracked, remaining shields).
    remaining = jnp.sum(shield_value(r.shields, now)[ctx.unit], axis=1)
    ichor = jnp.minimum(state.ichor, remaining)
    heal = report.life_steal_heal[ctx.unit]
    missing = jnp.maximum(r.max_hp[ctx.unit] - r.hp[ctx.unit], 0.0)
    cap = level_bp(165.0, 15.0, 9.0, ctx.level)
    bt = holds(own, BLOODTHIRSTER) & ctx.alive & (r.hp[ctx.unit] > 0.0)
    grant = jnp.where(bt, jnp.clip(jnp.minimum(heal - missing, cap - ichor), 0.0, None), 0.0)

    state = state._replace(last_dmg=last_dmg, champ_combat=champ_combat, venom_until=venom_until,
                           venom_fresh=jnp.where(jnp.any(fresh, axis=1)[:, None], fresh, state.venom_fresh),
                           venom_fresh_at=jnp.where(jnp.any(fresh, axis=1), now, state.venom_fresh_at),
                           sling_cd_until=jnp.where(sling, now + SLING_CD, state.sling_cd_until),
                           ichor=ichor + grant)

    # Voltaic Galvanize: ability damage to an enemy champion fires Energized (packets
    # resolve in the next pass).
    abil = p.valid & (r.final > 0.0) & has(p.flags, TAG_ACTIVE_SPELL) & ~has(p.flags, TAG_ITEM)
    gal_mask = hit_by_holder(report, ctx, n, abil) & champs
    gal = holds(own, VOLTAIC) & ctx.alive & (state.energy >= ENERGY_MAX) & jnp.any(gal_mask, axis=1)
    state, p_gal = _energized(state, own, ctx, units, gal, jnp.argmax(gal_mask, axis=1))
    eff = effects(c, n, packets=concat_packets(p_sling, p_exec, p_gal),
                  shields=shield_grants(grant, duration=ICHOR_DURATION))
    return state, eff


def periodic(state: State, own, ctx, units):
    c, n = ctx.level.shape[0], units.x.shape[0]
    gain = jnp.where(holds_any(own, ENERGIZED_ITEMS) & ctx.alive, ctx.moved / ENERGY_UNITS_PER_STACK, 0.0)
    state = state._replace(energy=jnp.minimum(ENERGY_MAX, state.energy + gain))
    return state, effects(c, n)


def on_takedown(state: State, own, ctx, units, kills):
    c, n = ctx.level.shape[0], units.x.shape[0]
    now = ctx.now
    recent = kills.killed_units & _enemy_champs(ctx, units) & (now - state.last_dmg <= TAKEDOWN_WINDOW)
    count = jnp.sum(recent, axis=1).astype(jnp.float32)
    any_td = count > 0
    hub = any_td & holds(own, HUBRIS)
    ax = any_td & holds(own, AXIOM)
    refund = count * (AXIOM_BASE + AXIOM_PER_LETHALITY * ctx.lethality)
    gold = jnp.where(holds(own, COLLECTOR), COLLECTOR_GOLD * kills.champion_kill, 0.0)
    state = state._replace(
        hubris_stacks=jnp.where(hub, state.hubris_stacks + count, state.hubris_stacks),
        hubris_until=jnp.where(hub, now + HUBRIS_DUR, state.hubris_until),
        hex_until=jnp.where(any_td & holds(own, HEXOPTICS), now + HEX_DUR, state.hex_until),
        axiom_refund=jnp.where(ax, refund, state.axiom_refund),
        axiom_at=jnp.where(ax, now, state.axiom_at))
    return state, effects(c, n, gold=gold)


# ---- integrator helpers -----------------------------------------------------

def extra_on_hit_targets(state: State, ctx):
    """(C, N) Runaan's bolt and Statikk secondary-bounce targets this tick.

    Damage packets are already emitted; the integrator must apply the holder's
    on-hit effects (``on_hit`` dispatch with ``raw = 0``) to each.
    """
    return state.extra_hits & (state.extra_at == ctx.now)[:, None]


def phantom_hit_due(state: State, ctx):
    """(C,) Guinsoo's Phantom Hit: re-apply on-hit effects to the attack target."""
    return state.phantom_at == ctx.now


def basic_attack_amp(state: State, own, ctx, units):
    """(C, N) Hexoptics Magnification: additive amp for basic-attack packets only.

    Edge-to-edge distance from holder to target, linear 0 -> 10% at 500.
    """
    hx, hy = unit_pos(units, ctx.unit)
    hr = units.radius[jnp.clip(ctx.unit, 0, units.x.shape[0] - 1)]
    d = jnp.sqrt((units.x[None, :] - hx[:, None]) ** 2 + (units.y[None, :] - hy[:, None]) ** 2)
    edge = jnp.maximum(d - hr[:, None] - units.radius[None, :], 0.0)
    return jnp.where(holds(own, HEXOPTICS)[:, None], HEX_AMP * jnp.clip(edge / HEX_RANGE, 0.0, 1.0), 0.0)


def packet_amp(state: State, own, ctx, units, p):
    """(P,) Hexoptics Magnification on the holder's basic-attack packets only."""
    from ..modern_damage import TAG_BASIC_ATTACK, has
    amp = basic_attack_amp(state, own, ctx, units)                     # (C, N)
    src_is = p.src[:, None] == ctx.unit[None, :]
    per = amp[:, jnp.clip(p.dst, 0, amp.shape[1] - 1)].T
    return jnp.where(has(p.flags, TAG_BASIC_ATTACK), jnp.sum(jnp.where(src_is, per, 0.0), axis=1), 0.0)


def attack_range_bonus(state: State, own, ctx, base_range):
    """(C,) bonus attack range: RFC on the Energized attack (35%, max 150) + Arcane Aim 100."""
    rfc = holds(own, RFC) & (state.energy >= ENERGY_MAX)
    bonus = jnp.where(rfc, jnp.minimum(RFC_RANGE_PCT * base_range, RFC_RANGE_MAX), 0.0)
    return bonus + jnp.where(holds(own, HEXOPTICS) & (ctx.now < state.hex_until), HEX_EXTRA, 0.0)


def basic_cooldown_scale(state: State, ctx):
    """(C,) Navori: multiply remaining Q/W/E cooldowns by this (1.0 = no effect)."""
    return jnp.where(state.navori_at == ctx.now, 1.0 - NAVORI_CDR, 1.0)


def ult_refund_fraction(state: State, ctx):
    """(C,) Axiom Arc: fraction of the ultimate's total cooldown to refund this tick."""
    return jnp.where(state.axiom_at == ctx.now, state.axiom_refund, 0.0)


def shield_reaver(state: State, own, ctx, units):
    """Serpent's Fang, per world unit (N,): ``(strength, fresh)``.

    ``strength``: fraction by which non-magic shields *granted* to the unit
    are reduced now (strongest active venom; 0.50 melee / 0.35 ranged holder).
    ``fresh``: the unit was newly afflicted this tick, so its existing
    non-magic shields must also be reduced by ``strength`` once.
    """
    per = jnp.where(ctx.is_ranged, SERPENT_RANGED, SERPENT_MELEE)[:, None]
    active = ctx.now < state.venom_until
    strength = jnp.max(jnp.where(active, per, 0.0), axis=0)
    fresh = jnp.any(state.venom_fresh & (state.venom_fresh_at == ctx.now)[:, None], axis=0)
    return strength, fresh
