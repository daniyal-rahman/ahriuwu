"""Marksman / lethality item passives (wiki text in docs/modern/ITEMS_CATALOG.md where data is silent).

Energized (ITEMS.md §6.8): one 0..100 charge per holder of any Energized item, +6 per attack (+Statikk bonus) and
+1 per 24 units moved; an attack launched at 100 is Energized and its on-hit fires every owned Energized effect.
Integrator helpers at the bottom expose what the hooks cannot: on-hit targets of Runaan's bolts and Statikk bounces
(re-run ``on_hit`` with ``raw = 0``), Guinsoo's Phantom Hit, Hexoptics basic-attack amp and Arcane Aim / RFC range,
Navori and Axiom Arc cooldown effects and Serpent's Fang shield reaving.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from ...core.damage import (CLASS_CHAMPION, CLASS_STRUCTURE, MAGIC, ON_HIT_ITEM, PHYSICAL, PROP_CRIT,
                            PROP_EXECUTE, PROP_LIFESTEAL, TAG_ACTIVE_SPELL, TAG_AOE, TAG_BASIC_ATTACK, TAG_ITEM,
                            TAG_PROC, TRUE, concat_packets, has, packets, shield_value)
from ..catalog import ItemStats, level_bp
from .core import (AttackMods, StatusFlags, by_range, dealt_by_holder, dv, effects, enemy_mask, hit_by_holder, holds,
                   holds_any, in_circle, nearest_k, onehot_units, shield_grants, target_class, unit_pos)

RECURVE, FIENDHUNTER, HEXOPTICS, YUNTAL, LDR, PHANTOM, BLOODTHIRSTER = 1043, 2512, 2523, 3032, 3036, 3046, 3072
RUNAANS, STATIKK, RFC, STORMRAZOR, GUINSOO, SLINGSHOT, NOONQUIVER = 3085, 3087, 3094, 3095, 3124, 3144, 6670
KRAKEN, NAVORI, COLLECTOR, YOUMUU, HUBRIS, AXIOM, UMBRAL, SERPENT, VOLTAIC = \
    6672, 6675, 6676, 3142, 6697, 6696, 3179, 6695, 6699
ENERGIZED_ITEMS = (RFC, STATIKK, STORMRAZOR, VOLTAIC)

NEVER = -1e9

RECURVE_DMG = dv(RECURVE, "OnHitDamage")
GUINSOO_DMG = dv(GUINSOO, "OnHitDamage")
GUINSOO_AS = dv(GUINSOO, "AttackSpeedPerStack")
GUINSOO_MAX = dv(GUINSOO, "MaxStacks")
GUINSOO_DUR = dv(GUINSOO, "BuffDuration")
GUINSOO_PHANTOM_MAX = 2.0          # wiki: the attack after 2 Phantom stacks fires ("every third")
KRAKEN_COUNT = dv(KRAKEN, "AttackCount")
KRAKEN_DUR = dv(KRAKEN, "BuffDuration")
KRAKEN_MAX_AMP = dv(KRAKEN, "MaxAmpNumber")
KRAKEN_RANGED = dv(KRAKEN, "RangedDamageMultiplier")
YT_DUR, YT_CD = dv(YUNTAL, "ASDuration"), dv(YUNTAL, "Cooldown")
YT_AS, YT_CRIT_MAX = dv(YUNTAL, "ASMod"), dv(YUNTAL, "CritMax") / 100.0
YT_AA_CDR, YT_CRIT_CDR = dv(YUNTAL, "AACDR"), dv(YUNTAL, "CritCDR")
# CritPerStackCalc points at unnamed hashes; the named values match the wiki.
YT_CRIT_PER = dv(YUNTAL, "CritPerStackMelee") / 100.0
YT_RANGED = dv(YUNTAL, "StackRangedMultiplier")
FH_HASTE, FH_CD, FH_DUR = dv(FIENDHUNTER, "UltimateHaste"), dv(FIENDHUNTER, "Cooldown"), dv(FIENDHUNTER, "Duration")
FH_AS, FH_N = dv(FIENDHUNTER, "BonusAS"), dv(FIENDHUNTER, "NumberOfAttacks")
FH_CRIT, FH_TRUE = dv(FIENDHUNTER, "CritModifier"), dv(FIENDHUNTER, "BonusTrueDamage")
LDR_MAX, LDR_HP = dv(LDR, "MaxBonusDamagePercent"), dv(LDR, "MaxBonusHealth")
HEX_AMP, HEX_RANGE = dv(HEXOPTICS, "MaxDamageAmp"), dv(HEXOPTICS, "MaxRange")
HEX_EXTRA, HEX_DUR = dv(HEXOPTICS, "ExtraRange"), dv(HEXOPTICS, "Duration")
TAKEDOWN_WINDOW = dv(HEXOPTICS, "TakedownWindow")      # also Hubris/Axiom
RUNAAN_RATIO = 0.65                # calc BoltDamage
RUNAAN_EXTRA = dv(RUNAANS, "ExtraRangeOnBoltCheck")
RUNAAN_BOLTS_RANGED, RUNAAN_BOLTS_MELEE = 2, 1           # calc ChampRange 2 / 1
RUNAAN_ATTACK_RANGE = 550.0        # INFERRED: holder range is not read from Ctx
STATIKK_BONUS = dv(STATIKK, "BonusEnergizedStacks")
STATIKK_CHAMP, STATIKK_OTHER = dv(STATIKK, "ChainDamage"), dv(STATIKK, "NonChampChainDamage")
STATIKK_RANGE = dv(STATIKK, "BounceRange")
STATIKK_MAX_BOUNCES = 8            # BounceCount at L20
RFC_DMG = dv(RFC, "BonusDamage")
RFC_RANGE_PCT, RFC_RANGE_MAX = dv(RFC, "RangePercentIncrease"), dv(RFC, "MaxRangeIncrease")
STORM_DMG = 100.0                  # calc TotalProcDamage
STORM_MS, STORM_DUR = dv(STORMRAZOR, "BuffStrength"), dv(STORMRAZOR, "BuffDuration")
VOLT_PCT_M, VOLT_PCT_R = dv(VOLTAIC, "PercentCurrentHPMelee") / 100.0, dv(VOLTAIC, "PercentCurrentHPRanged") / 100.0
VOLT_LETH_M, VOLT_LETH_R = dv(VOLTAIC, "LethalityBonusModMelee"), dv(VOLTAIC, "LethalityBonusModRanged")
VOLT_DUR, VOLT_CAP = dv(VOLTAIC, "LethalityBonusDuration"), dv(VOLTAIC, "NonChampCap")
SLING_DMG = 40.0                   # calc DamageAmount
SLING_CD = dv(SLINGSHOT, "Cooldown")
SLING_ATTACK_CDR = 1.0             # tooltip
NAVORI_CDR = dv(NAVORI, "CDRAmount")
COLLECTOR_THRESHOLD, COLLECTOR_GOLD = dv(COLLECTOR, "ExecuteThreshold"), dv(COLLECTOR, "GoldAmount")
YOUMUU_OOC_MS, YOUMUU_TIMER = dv(YOUMUU, "BaseOOCMS"), dv(YOUMUU, "CombatTimer")
YOUMUU_RANGED = 0.5                # calc OOCMS mRangedMultiplier
HUBRIS_BASE, HUBRIS_PER, HUBRIS_DUR = dv(HUBRIS, "BaseADBonus"), dv(HUBRIS, "ADPerStatue"), dv(HUBRIS, "BuffDuration")
AXIOM_BASE = dv(AXIOM, "UltimateRefundBase") / 100.0
AXIOM_PER_LETHALITY = 0.25 / 100.0  # calc
SERPENT_MELEE, SERPENT_RANGED = dv(SERPENT, "ShieldShred") / 100.0, dv(SERPENT, "ShieldShredRange") / 100.0
SERPENT_DUR = dv(SERPENT, "DebuffDuration")
ENERGY_MAX, ENERGY_PER_ATTACK, ENERGY_UNITS_PER_STACK = 100.0, 6.0, 24.0   # wiki
ICHOR_DURATION = 1e6               # "lasts until destroyed"

COVERAGE = {
    RECURVE: "Sting physical on-hit",
    FIENDHUNTER: "Night Vigil ult haste; Opening Barrage after R: empowered attacks with AS, forced reduced crit, "
                 "true bonus on natural crits",
    HEXOPTICS: "Magnification by distance on basic attacks; Arcane Aim range after a takedown",
    YUNTAL: "Practice Makes Lethal permanent crit per attack; Flurry AS on champion attack (cd cut by hits/crits)",
    LDR: "Giant Slayer amp by target bonus HP vs champions",
    PHANTOM: "Spectral Waltz: ghosted (status)",
    BLOODTHIRSTER: "Ichorshield: overheal from vamp -> shield, level-capped",
    RUNAANS: "Wind's Fury bolts at the nearest enemies in front; on-hits via extra_on_hit_targets",
    STATIKK: "Electroshock energize; Electrospark chain by level; secondary on-hits via extra_on_hit_targets",
    RFC: "Energized Sharpshooter magic on-hit and range (attack_range_bonus)",
    STORMRAZOR: "Energized Bolt magic on-hit and MS",
    GUINSOO: "Wrath on-hit; Seething Strike AS stacks; Phantom Hit (phantom_hit_due)",
    SLINGSHOT: "Bullseye magic damage on damaging a champion (attacks cut the cd)",
    NOONQUIVER: "stats only (no passive data values in client 16.19)",
    KRAKEN: "Bring It Down every 3rd attack, missing-HP scaled",
    NAVORI: "Transcendence: basic cooldowns reduced on attack (basic_cooldown_scale)",
    COLLECTOR: "Death: execute low-HP champions; Taxes gold per kill",
    YOUMUU: "Haunt out-of-combat MS; Wraith Step active in actives",
    HUBRIS: "Eminence: AD buff and permanent stack on recent champion takedown",
    AXIOM: "Flux: ult refund on recent champion takedown (ult_refund_fraction)",
    UMBRAL: "stats only; Nightstalker and Blackout need vision (deferred)",
    SERPENT: "Shield Reaver venom on damaged champions (shield_reaver)",
    VOLTAIC: "Energized Firmament %current-HP damage and lethality; Galvanize: ability damage fires Energized",
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


def _enemy_champs(ctx, units):
    return (units.cls[None, :] == CLASS_CHAMPION) & (units.team[None, :] != ctx.team[:, None])


def statikk_bounces(level):
    """BounceCount: 4, +1 at levels 6, 10, 14 and 20."""
    lv = jnp.asarray(level, jnp.float32)
    return 4.0 + sum((lv >= k).astype(jnp.float32) for k in (6, 10, 14, 20))


def _energized(state: State, own, ctx, units, fire, tgt):
    """Fire every owned Energized effect at ``tgt`` (C,) for holders ``fire``."""
    c, n = ctx.level.shape[0], units.x.shape[0]
    tgt = jnp.maximum(tgt, 0)
    tcls = target_class(units, tgt)
    champ = tcls == CLASS_CHAMPION
    out = []
    rfc = fire & holds(own, RFC)
    out.append(packets(rfc, ctx.unit, tgt, RFC_DMG, MAGIC, ON_HIT_ITEM, item=RFC))
    storm = fire & holds(own, STORMRAZOR)
    out.append(packets(storm, ctx.unit, tgt, STORM_DMG, MAGIC, ON_HIT_ITEM, item=STORMRAZOR))
    # Voltaic: % current HP before this attack, capped vs non-champions.
    volt = fire & holds(own, VOLTAIC)
    vdmg = jnp.where(ctx.is_ranged, VOLT_PCT_R, VOLT_PCT_M) * units.hp[tgt]
    vdmg = jnp.where(champ, vdmg, jnp.minimum(vdmg, VOLT_CAP))
    out.append(packets(volt, ctx.unit, tgt, vdmg, PHYSICAL, ON_HIT_ITEM, item=VOLTAIC))
    # Statikk: chain to the nearest unhit enemy in range.
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


def stats(state: State, own, ctx) -> ItemStats:
    now = ctx.now
    guinsoo = jnp.where(holds(own, GUINSOO) & (now < state.guinsoo_until), GUINSOO_AS * state.guinsoo_stacks, 0.0)
    flurry = jnp.where(holds(own, YUNTAL) & (now < state.yt_as_until), YT_AS, 0.0)
    barrage = jnp.where(holds(own, FIENDHUNTER) & (state.fh_charges > 0) & (now < state.fh_until), FH_AS, 0.0)
    crit = jnp.where(holds(own, YUNTAL), state.yt_crit, 0.0)
    storm_ms = jnp.where(holds(own, STORMRAZOR) & (now < state.storm_until), STORM_MS, 0.0)
    ooc = holds(own, YOUMUU) & (now - state.champ_combat >= YOUMUU_TIMER)
    youmuu = jnp.where(ooc, YOUMUU_OOC_MS * by_range(ctx, YOUMUU_RANGED), 0.0)
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

    has_en = holds_any(own, ENERGIZED_ITEMS)
    en_now = la & has_en & (state.energy >= ENERGY_MAX)
    gain = ENERGY_PER_ATTACK + jnp.where(holds(own, STATIKK), STATIKK_BONUS, 0.0)
    energy = jnp.where(la & has_en & ~en_now, jnp.minimum(ENERGY_MAX, state.energy + gain), state.energy)
    en_pending = jnp.where(la, en_now, state.en_pending)

    # Guinsoo's: the attack reaching max stacks grants the first Phantom stack (INFERRED L).
    g = la & holds(own, GUINSOO)
    alive_g = now < state.guinsoo_until
    gst = jnp.where(alive_g, state.guinsoo_stacks, 0.0)
    ph = jnp.where(alive_g, state.phantom, 0.0)
    fire_ph = g & (ph >= GUINSOO_PHANTOM_MAX)
    gst_new = jnp.minimum(GUINSOO_MAX, gst + 1.0)
    ph_new = jnp.where(fire_ph, 0.0, jnp.where(gst_new >= GUINSOO_MAX, jnp.minimum(GUINSOO_PHANTOM_MAX, ph + 1.0), ph))

    # Kraken (ranged): stacks at launch; the 3rd consumes them and deals on-hit.
    kr = la & holds(own, KRAKEN) & ctx.is_ranged
    kst = jnp.where(now < state.kraken_until, state.kraken_stacks, 0.0)
    k_fire = kr & (kst >= KRAKEN_COUNT - 1)

    y = la & holds(own, YUNTAL)
    yt_crit = jnp.where(y, jnp.minimum(YT_CRIT_MAX, state.yt_crit + YT_CRIT_PER * by_range(ctx, YT_RANGED)),
                        state.yt_crit)
    flurry = y & vs_champ & (now >= state.yt_cd_until)

    fh = la & (state.fh_charges > 0) & (now < state.fh_until) & holds(own, FIENDHUNTER)

    # Runaan's bolts fire at launch, no travel time (INFERRED).
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
    p_bolts = packets(bolts, ctx.unit[:, None], jnp.arange(n)[None, :],
                      (RUNAAN_RATIO * ctx.total_ad * crit_mult)[:, None],
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

    # Kraken: melee stacks on-hit; ranged consumes what launch reserved.
    km = h & holds(own, KRAKEN) & ~ctx.is_ranged
    kst = jnp.where(now < state.kraken_until, state.kraken_stacks, 0.0)
    km_fire = km & (kst >= KRAKEN_COUNT - 1)
    kr_fire = h & holds(own, KRAKEN) & ctx.is_ranged & state.kraken_pending
    k_fire = km_fire | kr_fire
    missing = 1.0 - units.hp[tgt] / jnp.maximum(units.max_hp[tgt], 1.0)
    kdmg = level_bp(150.0, 5.0, 9.0, ctx.level) * by_range(ctx, KRAKEN_RANGED) \
        * (1.0 + (KRAKEN_MAX_AMP - 1.0) * jnp.clip(missing, 0.0, 1.0))
    out.append(packets(k_fire, ctx.unit, tgt, kdmg, PHYSICAL, lsf, item=KRAKEN))

    # Fiendhunter: a natural crit on an empowered attack adds true damage. Without ``Attack.natural_crit``, a
    # crit is natural when raw exceeds the forced-crit value.
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

    # Serpent's Fang: fresh = not already afflicted by any holder.
    sf = holds(own, SERPENT)[:, None] & hit_champ
    already = jnp.any(now < state.venom_until, axis=0)
    fresh = sf & ~already[None, :]
    venom_until = jnp.where(sf, now + SERPENT_DUR, state.venom_until)

    # Slingshot: first enemy champion damaged by another packet.
    other = p.valid & (r.final > 0.0) & (p.item != SLINGSHOT)
    sling_mask = hit_by_holder(report, ctx, n, other) & champs
    sling = holds(own, SLINGSHOT) & ctx.alive & (now >= state.sling_cd_until) & jnp.any(sling_mask, axis=1)
    sling_tgt = jnp.argmax(sling_mask, axis=1)
    p_sling = packets(sling, ctx.unit, sling_tgt, SLING_DMG, MAGIC, TAG_PROC | TAG_ITEM, item=SLINGSHOT)

    hp_after, max_after = r.hp[None, :], r.max_hp[None, :]
    execute = holds(own, COLLECTOR)[:, None] & hit_champ & (hp_after > 0.0) \
        & (hp_after < COLLECTOR_THRESHOLD * max_after)
    p_exec = packets(execute, ctx.unit[:, None], jnp.arange(n)[None, :], hp_after, TRUE,
                     PROP_EXECUTE | TAG_ITEM, item=COLLECTOR)

    # Bloodthirster: life-steal heal beyond post-damage missing HP (heal modifiers ignored). The Ichorshield never
    # expires so it absorbs last: what remains is min(tracked, remaining shields).
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

    # Voltaic Galvanize: its Energized packets resolve in the next pass.
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


def extra_on_hit_targets(state: State, ctx):
    """(C, N) Runaan's bolt and Statikk bounce targets this tick; their damage is emitted, on-hits are not."""
    return state.extra_hits & (state.extra_at == ctx.now)[:, None]


def phantom_hit_due(state: State, ctx):
    """(C,) Guinsoo's Phantom Hit: re-apply on-hit effects to the attack target."""
    return state.phantom_at == ctx.now


def basic_attack_amp(state: State, own, ctx, units):
    """(C, N) Hexoptics Magnification amp, linear in edge-to-edge distance."""
    hx, hy = unit_pos(units, ctx.unit)
    hr = units.radius[jnp.clip(ctx.unit, 0, units.x.shape[0] - 1)]
    d = jnp.sqrt((units.x[None, :] - hx[:, None]) ** 2 + (units.y[None, :] - hy[:, None]) ** 2)
    edge = jnp.maximum(d - hr[:, None] - units.radius[None, :], 0.0)
    return jnp.where(holds(own, HEXOPTICS)[:, None], HEX_AMP * jnp.clip(edge / HEX_RANGE, 0.0, 1.0), 0.0)


def packet_amp(state: State, own, ctx, units, p):
    amp = basic_attack_amp(state, own, ctx, units)                     # (C, N)
    src_is = p.src[:, None] == ctx.unit[None, :]
    per = amp[:, jnp.clip(p.dst, 0, amp.shape[1] - 1)].T
    return jnp.where(has(p.flags, TAG_BASIC_ATTACK), jnp.sum(jnp.where(src_is, per, 0.0), axis=1), 0.0)


def attack_range_bonus(state: State, own, ctx, base_range):
    """(C,) bonus attack range: RFC on the Energized attack plus Arcane Aim."""
    rfc = holds(own, RFC) & (state.energy >= ENERGY_MAX)
    bonus = jnp.where(rfc, jnp.minimum(RFC_RANGE_PCT * base_range, RFC_RANGE_MAX), 0.0)
    return bonus + jnp.where(holds(own, HEXOPTICS) & (ctx.now < state.hex_until), HEX_EXTRA, 0.0)


def basic_cooldown_scale(state: State, ctx):
    """(C,) Navori multiplier on remaining Q/W/E cooldowns."""
    return jnp.where(state.navori_at == ctx.now, 1.0 - NAVORI_CDR, 1.0)


def ult_refund_fraction(state: State, ctx):
    """(C,) Axiom Arc fraction of the ultimate's cooldown refunded this tick."""
    return jnp.where(state.axiom_at == ctx.now, state.axiom_refund, 0.0)


def shield_reaver(state: State, own, ctx, units):
    """Serpent's Fang per unit (N,): ``strength`` reduces non-magic shields granted to it (strongest venom);
    ``fresh`` units were newly afflicted, so their existing non-magic shields are reduced once too."""
    per = jnp.where(ctx.is_ranged, SERPENT_RANGED, SERPENT_MELEE)[:, None]
    active = ctx.now < state.venom_until
    strength = jnp.max(jnp.where(active, per, 0.0), axis=0)
    fresh = jnp.any(state.venom_fresh & (state.venom_fresh_at == ctx.now)[:, None], axis=0)
    return strength, fresh
