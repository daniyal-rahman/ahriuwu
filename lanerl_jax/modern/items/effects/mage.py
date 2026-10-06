"""Mage items: burns, ability-damage passives, AP scaling and on-damage procs (ITEMS_CATALOG wiki text).

"Ability damage" = holder packets tagged ActiveSpell and not Item with ``final > 0``. Burns and zones are
per-(holder, target) timers ticking every ``TickFrequency`` after application; a refresh extends the end and keeps
the phase, so an unrefreshed burn deals BurnDuration / TickFrequency ticks for any ``dt``. Champion combat
(Madness, Suffering, Void Corruption) is this module's own timer: stacks = whole seconds since it began, and it ends
after BuffCounterDuration without damage dealt to or taken from an enemy champion (INFERRED M).
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from ...core.damage import (CLASS_CHAMPION, CLASS_MONSTER, CLASS_STRUCTURE, MAGIC, ON_HIT_ITEM, PROP_LIFESTEAL,
                            TAG_ACTIVE_SPELL, TAG_AOE, TAG_ITEM, TAG_PERIODIC, TAG_PET, TAG_PROC, TRUE,
                            concat_packets, has, packets, per_unit)
from ..catalog import ItemStats
from .core import BIG, Debuffs, dv, effects, enemy_mask, holds, in_circle, nearest_k, onehot_units, unit_pos

ASHES, BLACKFIRE, ACTUALIZER, RABADON = 2508, 2503, 2522, 3089
NASHOR, RYLAI, MALIGNANCE, CRYPTBLOOM = 3115, 3116, 3118, 3137
ALTERNATOR, GUNBLADE, GUISE, ROCKETBELT = 3145, 3146, 3147, 3152
MORELLO, CHAPTER, CATALYST, ORB = 3165, 3802, 3803, 3916
HORIZON, COSMIC, RIFTMAKER, SHADOWFLAME = 4628, 4629, 4633, 4645
STORMSURGE, LIANDRY, LUDEN, ROA, BLOODLETTER = 4646, 6653, 6655, 6657, 8010

EPS = 1e-4

ULT_ATTRIBUTION_WINDOW = 1.5    # INFERRED L: packets carry no ability slot; damage this soon after R counts as R
MALIGNANCE_TICK = 0.25          # calc {8e8f7a34}
STORM_BUCKET = 0.25             # Stormsurge sliding-window ring buffer resolution
STORM_SLOTS = int(round(dv(STORMSURGE, "WindowDuration") / STORM_BUCKET))
STORM_AOE = 600.0               # wiki: target dies before the Squall
HORIZON_FOCUS_RADIUS = dv(HORIZON, "VisionRadius")

COVERAGE = {
    ASHES: "Inflame: ability damage burns (bonus vs monsters)",
    BLACKFIRE: "Baleful Blaze burn by target class; Blackfire %AP per burning champion/monster",
    ACTUALIZER: "stats only; Mana Made Real active in actives",
    RABADON: "Magical Opus: % total AP",
    NASHOR: "Icathian Bite magic on-hit",
    RYLAI: "Rimefrost: ability damage slows",
    MALIGNANCE: "Scorn ult haste; Hatefog zone on ult damage (MR shred, ticking magic damage); ult damage attributed "
                "by a window after R",
    CRYPTBLOOM: "Life From Death: heal on recent champion takedown (nova travel and ally heals not modelled)",
    ALTERNATOR: "Revved: magic damage on damaging a champion",
    GUNBLADE: "stats only; Lightning Bolt active in actives",
    GUISE: "Madness: damage amp ramping in champion combat",
    ROCKETBELT: "stats only; Supersonic active in actives",
    MORELLO: "Grievous Wounds on magic damage to champions",
    CHAPTER: "Enlighten: level up restores mana over time",
    CATALYST: "Eternity mana from champion damage taken (mana-spent heal needs ability mana costs)",
    ORB: "Grievous Wounds on magic damage to champions",
    HORIZON: "Hypershot long-range mark and Focus (vision reveal deferred; distance from position at damage time)",
    COSMIC: "Spelldance: MS after magic/true damage to champions",
    RIFTMAKER: "Void Infusion bonus HP -> AP; Void Corruption amp ramp, omnivamp at max",
    SHADOWFLAME: "Cinderbloom: follow-up packet on magic/true damage to low-HP enemies (resolves next pass)",
    STORMSURGE: "Stormraider: windowed champion damage threshold -> delayed Squall, AoE if the target dies first",
    LIANDRY: "Torment %max-HP burn from ability/pet damage (monster cap); Suffering amp ramp",
    LUDEN: "Echo: ability damage fires at the target and nearest enemies; unused echoes hit the target",
    ROA: "Timeless stacks over time held; Eternity mana (max-stack level-up and mana-spent heal not modelled)",
    BLOODLETTER: "Vile Decay: magic ability damage to champions stacks %MR shred",
}


class State(NamedTuple):
    combat_last: Any      # (C,) last damage dealt to / taken from an enemy champion
    combat_start3: Any    # (C,) combat start, 3 s linger (Madness/Suffering)
    combat_start4: Any    # (C,) combat start, 4 s linger (Void Corruption)
    ashes_until: Any      # (C, N)
    ashes_next: Any
    bf_until: Any
    bf_next: Any
    bf_qual: Any          # (C, N) bool: burning target counts for Blackfire AP
    lia_until: Any
    lia_next: Any
    luden_cd: Any         # (C,)
    alt_cd: Any
    ch_level: Any         # (C,) last seen level (-1 = unknown)
    ch_rem: Any           # (C,) Enlighten mana still to restore
    ch_until: Any
    roa_elapsed: Any      # (C,) seconds Rod of Ages has been held
    ult_until: Any        # (C,) Malignance attribution window end
    mal_x: Any            # (C, N) Hatefog zone created on champion n
    mal_y: Any
    mal_r: Any
    mal_until: Any        # also the 3 s per-target cooldown
    mal_next: Any
    crypt_last: Any       # (C, N) last time holder damaged unit n
    crypt_cd: Any         # (C,)
    storm_hist: Any       # (C, N, B) post-mitigation damage per bucket
    storm_epoch: Any      # (C, B) int32 bucket epoch
    storm_cd: Any         # (C,)
    storm_target: Any     # (C,) int32, -1 none
    storm_at: Any         # (C,)
    storm_x: Any
    storm_y: Any
    cosmic_until: Any     # (C,)
    hz_until: Any         # (C, N) Hypershot mark
    hz_cd: Any            # (C,) Focus cooldown
    bl_stacks: Any        # (C, N)
    bl_until: Any
    bl_icd: Any


def init(n_champions: int, n_units: int) -> State:
    c, n = n_champions, n_units
    neg = lambda *s: jnp.full(s, -BIG, jnp.float32)
    z = lambda *s: jnp.zeros(s, jnp.float32)
    return State(
        neg(c), neg(c), neg(c), neg(c, n), z(c, n), neg(c, n), z(c, n), jnp.zeros((c, n), bool),
        neg(c, n), z(c, n), neg(c), neg(c), jnp.full((c,), -1.0, jnp.float32), z(c), neg(c), z(c),
        neg(c), z(c, n), z(c, n), z(c, n), neg(c, n), z(c, n), neg(c, n), neg(c),
        z(c, n, STORM_SLOTS), jnp.full((c, STORM_SLOTS), -1, jnp.int32), neg(c),
        jnp.full((c,), -1, jnp.int32), neg(c), z(c), z(c), neg(c), neg(c, n), neg(c),
        z(c, n), neg(c, n), neg(c, n))


def _ticks(until, nxt, now, period):
    """(ticks due by now, next tick time) for a timer ticking at nxt, nxt + period, ... <= until."""
    end = jnp.minimum(now, until)
    k = jnp.where(nxt <= end + EPS, jnp.floor((end - nxt) / period + EPS) + 1.0, 0.0)
    return k, nxt + k * period


def _apply_timer(until, nxt, trig, now, duration, period):
    """(Re)apply a ticking timer: refresh keeps phase; a finished timer restarts."""
    active = nxt <= until + EPS
    return (jnp.where(trig, now + duration, until),
            jnp.where(trig & ~active, now + period, nxt))


def _stacks(state: State, ctx, start, linger, per_second, cap):
    in_combat = (ctx.now - state.combat_last) <= linger
    n_max = jnp.round(cap / per_second)
    return jnp.where(in_combat, jnp.clip(jnp.floor(ctx.now - start + EPS), 0.0, n_max), 0.0)


def _guise_amp(state, own, ctx):
    out = jnp.zeros(ctx.level.shape, jnp.float32)
    for item in (GUISE, LIANDRY):
        per, cap = dv(item, "DamageIncreasePerSecond"), dv(item, "DamageIncreaseMax")
        s = _stacks(state, ctx, state.combat_start3, dv(item, "BuffCounterDuration"), per, cap)
        out = out + jnp.where(holds(own, item), per * s, 0.0)
    return out


def _rift_stacks(state, ctx):
    return _stacks(state, ctx, state.combat_start4, dv(RIFTMAKER, "BuffCounterDuration"),
                   dv(RIFTMAKER, "EternityDamageIncreasePerSecond"), dv(RIFTMAKER, "EternityDamageIncreaseMax"))


def _roa_stacks(state, own):
    s = jnp.clip(jnp.floor(state.roa_elapsed / dv(ROA, "SecondsPerStack") + EPS), 0.0, dv(ROA, "MaxStacks"))
    return jnp.where(holds(own, ROA), s, 0.0)


def _first_dst(mask_cp, p):
    """(C,) destination of the first selected packet, -1 if none."""
    any_ = jnp.any(mask_cp, axis=1)
    idx = jnp.argmax(mask_cp, axis=1)
    return jnp.where(any_, p.dst[idx], -1)


def stats(state: State, own, ctx) -> ItemStats:
    now = ctx.now
    roa = _roa_stacks(state, own)
    roa_hp = dv(ROA, "HealthPerStack") * roa
    rift_ap = jnp.where(holds(own, RIFTMAKER), dv(RIFTMAKER, "HealthToAPConversionPercent")
                        * (ctx.bonus_hp + roa_hp), 0.0)
    flat_ap = rift_ap + dv(ROA, "APPerStack") * roa
    burning = jnp.sum((state.bf_until > now) & state.bf_qual, axis=1).astype(jnp.float32)
    pct = jnp.where(holds(own, RABADON), dv(RABADON, "APAmp"), 0.0) \
        + jnp.where(holds(own, BLACKFIRE), dv(BLACKFIRE, "APPerStack") * burning, 0.0)
    rift_max = _rift_stacks(state, ctx) >= jnp.round(dv(RIFTMAKER, "EternityDamageIncreaseMax")
                                                     / dv(RIFTMAKER, "EternityDamageIncreasePerSecond"))
    vamp = jnp.where(ctx.is_ranged, dv(RIFTMAKER, "VampAmountRanged"), dv(RIFTMAKER, "VampAmountMelee"))
    return ItemStats(
        health=roa_hp, mana=dv(ROA, "ManaPerStack") * roa,
        ability_power=flat_ap + pct * (ctx.ap + flat_ap),
        omnivamp=jnp.where(holds(own, RIFTMAKER) & rift_max, vamp, 0.0),
        ultimate_haste=jnp.where(holds(own, MALIGNANCE), dv(MALIGNANCE, "UltimateHaste"), 0.0),
        move_speed=jnp.where(holds(own, COSMIC) & (now < state.cosmic_until), 20.0, 0.0))  # calc MoveSpeedAmount


def dealt_amp(state: State, own, ctx, units):
    rift = jnp.where(holds(own, RIFTMAKER),
                     dv(RIFTMAKER, "EternityDamageIncreasePerSecond") * _rift_stacks(state, ctx), 0.0)
    marked = holds(own, HORIZON)[:, None] & (state.hz_until > ctx.now)
    return (_guise_amp(state, own, ctx) + rift)[:, None] + jnp.where(marked, dv(HORIZON, "DamageAmp"), 0.0)


def _in_zones(state: State, own, ctx, units, *, inclusive: bool = False):
    """(C, Z, N) enemies inside each Hatefog zone (Z = N, keyed by the zoned champion); ``inclusive`` keeps a
    zone on its final tick for damage."""
    live = (state.mal_until + EPS >= ctx.now) if inclusive else (state.mal_until > ctx.now)
    active = holds(own, MALIGNANCE)[:, None] & live                                   # (C, Z)
    d = jnp.sqrt((units.x[None, None, :] - state.mal_x[:, :, None]) ** 2
                 + (units.y[None, None, :] - state.mal_y[:, :, None]) ** 2)
    inside = d <= state.mal_r[:, :, None] + units.radius[None, None, :]
    return inside & active[:, :, None] & enemy_mask(ctx, units)[:, None, :] \
        & (units.cls != CLASS_STRUCTURE)[None, None, :]


def debuffs(state: State, own, ctx, units) -> Debuffs:
    n = units.x.shape[0]
    z = jnp.zeros((n,), jnp.float32)
    cursed = jnp.any(_in_zones(state, own, ctx, units), axis=(0, 1))
    flat_mr = jnp.where(cursed, 10.0, 0.0)          # calc MagicResistanceShred; one curse at a time
    stacks = jnp.where(holds(own, BLOODLETTER)[:, None] & (state.bl_until > ctx.now), state.bl_stacks, 0.0)
    pct_mr = jnp.max(stacks, axis=0, initial=0.0) * dv(BLOODLETTER, "ShredPerStack")
    return Debuffs(z, z, pct_mr, flat_mr, z, z, z)


def on_hit(state: State, own, ctx, units, attack):
    c, n = ctx.level.shape[0], units.x.shape[0]
    go = attack.hit & holds(own, NASHOR) & ctx.alive & (attack.target >= 0)
    dmg = dv(NASHOR, "NashorsBaseValue") + dv(NASHOR, "NashorsAPValue") * ctx.ap
    p = packets(go, ctx.unit, jnp.maximum(attack.target, 0), dmg, MAGIC, ON_HIT_ITEM | PROP_LIFESTEAL,
                item=NASHOR)
    return state, effects(c, n, packets=p)


def on_cast(state: State, own, ctx, units, cast):
    c, n = ctx.level.shape[0], units.x.shape[0]
    ult = cast.started & cast.is_ultimate
    return state._replace(ult_until=jnp.where(ult, ctx.now + ULT_ATTRIBUTION_WINDOW, state.ult_until)), \
        effects(c, n)


def on_damage(state: State, own, ctx, units, report):
    c, n = ctx.level.shape[0], units.x.shape[0]
    now = ctx.now
    p, r = report.packets, report.resolved
    nu = units.x.shape[0]
    dst = jnp.clip(p.dst, 0, nu - 1)
    srcc = jnp.clip(p.src, 0, nu - 1)
    dcls, scls = units.cls[dst], units.cls[srcc]
    src_is = p.src[None, :] == ctx.unit[:, None]                                       # (C, P)
    dst_is = p.dst[None, :] == ctx.unit[:, None]
    enemy_dst = units.team[dst][None, :] != ctx.team[:, None]
    enemy_src = units.team[srcc][None, :] != ctx.team[:, None]
    landed = p.valid & (r.final > 0.0)
    dealt = src_is & landed[None, :] & enemy_dst                                        # (C, P)
    champ, struct = dcls == CLASS_CHAMPION, dcls == CLASS_STRUCTURE
    ability = has(p.flags, TAG_ACTIVE_SPELL) & ~has(p.flags, TAG_ITEM)
    pet = has(p.flags, TAG_PET)
    magic = p.dtype == MAGIC
    magic_true = magic | (p.dtype == TRUE)
    enemies = enemy_mask(ctx, units) & (units.cls != CLASS_STRUCTURE)[None, :]
    all_packets = []

    taken_champ = dst_is & (p.valid & (p.raw > 0.0) & (scls == CLASS_CHAMPION))[None, :] & enemy_src
    event = jnp.any(dealt & champ[None, :], axis=1) | jnp.any(taken_champ, axis=1)
    gap = now - state.combat_last
    start3 = jnp.where(event & (gap > dv(GUISE, "BuffCounterDuration")), now, state.combat_start3)
    start4 = jnp.where(event & (gap > dv(RIFTMAKER, "BuffCounterDuration")), now, state.combat_start4)
    combat_last = jnp.where(event, now, state.combat_last)

    ab_hit = per_unit(dealt & (ability & ~struct)[None, :], p.dst, n)                   # (C, N)
    ab_pet_hit = per_unit(dealt & ((ability | pet) & ~struct)[None, :], p.dst, n)
    trig = lambda item, m: m & holds(own, item)[:, None]
    ashes_until, ashes_next = _apply_timer(state.ashes_until, state.ashes_next, trig(ASHES, ab_hit), now,
                                           dv(ASHES, "BurnDuration"), dv(ASHES, "TickFrequency"))
    bf_trig = trig(BLACKFIRE, ab_hit)
    bf_until, bf_next = _apply_timer(state.bf_until, state.bf_next, bf_trig, now,
                                     dv(BLACKFIRE, "BurnDuration"), dv(BLACKFIRE, "TickFrequency"))
    qual = ((units.cls == CLASS_CHAMPION) | (units.cls == CLASS_MONSTER))[None, :]
    bf_qual = jnp.where(bf_trig, qual, state.bf_qual)
    lia_until, lia_next = _apply_timer(state.lia_until, state.lia_next, trig(LIANDRY, ab_pet_hit), now,
                                       dv(LIANDRY, "BurnDuration"), dv(LIANDRY, "TickFrequency"))

    rylai = jnp.any(trig(RYLAI, ab_hit), axis=0)
    slow = jnp.where(rylai, dv(RYLAI, "SlowAmount"), 0.0)
    slow_duration = jnp.where(rylai, dv(RYLAI, "SlowDuration"), 0.0)

    gw_holder = holds(own, MORELLO) | holds(own, ORB)
    gw = jnp.any(per_unit(dealt & (magic & champ)[None, :], p.dst, n) & gw_holder[:, None], axis=0)
    grievous = jnp.where(gw, dv(MORELLO, "GrievousDuration"), 0.0)

    luden_sel = dealt & (ability & ~struct)[None, :]
    l_go = holds(own, LUDEN) & (now >= state.luden_cd) & jnp.any(luden_sel, axis=1)
    prim = jnp.where(l_go, _first_dst(luden_sel, p), -1)
    px, py = unit_pos(units, prim)
    prim_mask = onehot_units(prim, n)
    dist = jnp.sqrt((units.x[None, :] - px[:, None]) ** 2 + (units.y[None, :] - py[:, None]) ** 2)
    n_extra = int(dv(LUDEN, "MaxCharges")) - 1
    near = in_circle(units, px, py, jnp.full((c,), dv(LUDEN, "MissileRange"))) & enemies & ~prim_mask
    second = nearest_k(dist, near, n_extra) & l_go[:, None]
    l_dmg = dv(LUDEN, "BaseDamage") + dv(LUDEN, "APRatio") * ctx.ap
    left = n_extra - jnp.sum(second, axis=1).astype(jnp.float32)
    l_flags = TAG_ITEM | TAG_PROC
    all_packets += [
        packets(prim_mask & l_go[:, None], ctx.unit[:, None], jnp.arange(n)[None, :], l_dmg[:, None], MAGIC,
                l_flags, item=LUDEN),
        packets(second, ctx.unit[:, None], jnp.arange(n)[None, :], l_dmg[:, None], MAGIC, l_flags | TAG_AOE,
                item=LUDEN),
        packets(prim_mask & (l_go & (left > 0))[:, None], ctx.unit[:, None], jnp.arange(n)[None, :],
                (dv(LUDEN, "RepeatDamageReduction") * left * l_dmg)[:, None], MAGIC, l_flags, item=LUDEN)]
    luden_cd = jnp.where(l_go, now + dv(LUDEN, "Cooldown"), state.luden_cd)

    alt_sel = dealt & champ[None, :] & (p.item != ALTERNATOR)[None, :]
    a_go = holds(own, ALTERNATOR) & (now >= state.alt_cd) & jnp.any(alt_sel, axis=1) & ctx.alive
    a_tgt = _first_dst(alt_sel, p)
    all_packets.append(packets(a_go, ctx.unit, jnp.maximum(a_tgt, 0), 65.0, MAGIC,  # calc DamageAmount
                               TAG_ITEM | TAG_PROC, item=ALTERNATOR))
    alt_cd = jnp.where(a_go, now + dv(ALTERNATOR, "Cooldown"), state.alt_cd)

    cos = holds(own, COSMIC) & jnp.any(dealt & (magic_true & champ)[None, :], axis=1)
    cosmic_until = jnp.where(cos, now + dv(COSMIC, "StackDuration"), state.cosmic_until)

    hd = jnp.sqrt((units.x[None, :] - ctx.x[:, None]) ** 2 + (units.y[None, :] - ctx.y[:, None]) ** 2)
    hyper = per_unit(dealt & (ability & ~pet & champ)[None, :], p.dst, n) \
        & (hd >= dv(HORIZON, "SnipeRange")) & holds(own, HORIZON)[:, None]
    hz_until = jnp.where(hyper, jnp.maximum(state.hz_until, now + dv(HORIZON, "BuffDuration")), state.hz_until)
    focus = jnp.any(hyper, axis=1) & (now >= state.hz_cd)
    f_src = jnp.argmax(hyper, axis=1)
    fx, fy = unit_pos(units, f_src)
    f_area = in_circle(units, fx, fy, jnp.full((c,), HORIZON_FOCUS_RADIUS), edge=False) \
        & enemies & (units.cls == CLASS_CHAMPION)[None, :] & ~hyper & focus[:, None]
    hz_until = jnp.where(f_area, jnp.maximum(hz_until, now + dv(HORIZON, "SecondaryBuffDuration")), hz_until)
    hz_cd = jnp.where(focus, now + dv(HORIZON, "Cooldown"), state.hz_cd)

    ult_sel = dealt & ((ability | pet) & ~has(p.flags, TAG_PROC) & champ)[None, :] \
        & (now <= state.ult_until)[:, None] & holds(own, MALIGNANCE)[:, None]
    inst = per_unit(jnp.where(ult_sel, r.final[None, :], 0.0), p.dst, n, "max")
    zone = per_unit(ult_sel, p.dst, n) & (now >= state.mal_until)
    radius = jnp.minimum(dv(MALIGNANCE, "AOESize") + 2.0 ** (jnp.minimum(inst, 2000.0) / 100.0),
                         dv(MALIGNANCE, "MaxRadius"))
    mal_x = jnp.where(zone, units.x[None, :], state.mal_x)
    mal_y = jnp.where(zone, units.y[None, :], state.mal_y)
    mal_r = jnp.where(zone, radius, state.mal_r)
    mal_until = jnp.where(zone, now + dv(MALIGNANCE, "GroundDuration"), state.mal_until)
    mal_next = jnp.where(zone, now + MALIGNANCE_TICK, state.mal_next)

    crypt_last = jnp.where(per_unit(dealt, p.dst, n), now, state.crypt_last)

    epoch = jnp.floor(now / STORM_BUCKET + EPS).astype(jnp.int32)
    slot = epoch % STORM_SLOTS
    slot_hot = jnp.arange(STORM_SLOTS) == slot                                          # (B,)
    stale = slot_hot[None, :] & (state.storm_epoch != epoch)                            # (C, B)
    hist = jnp.where(stale[:, None, :], 0.0, state.storm_hist)
    amount = per_unit(jnp.where(dealt & champ[None, :], r.final[None, :], 0.0), p.dst, n, "add")
    hist = hist + jnp.where(slot_hot[None, None, :], amount[:, :, None], 0.0)
    storm_epoch = jnp.where(slot_hot[None, :], epoch, state.storm_epoch)
    recent = storm_epoch > epoch - STORM_SLOTS                                          # (C, B)
    window = jnp.sum(jnp.where(recent[:, None, :], hist, 0.0), axis=2)                  # (C, N)
    over = (window >= dv(STORMSURGE, "DamageThreshold") * units.max_hp[None, :]) \
        & (units.cls == CLASS_CHAMPION)[None, :] & enemies
    s_go = holds(own, STORMSURGE) & (now >= state.storm_cd) & (state.storm_target < 0) & jnp.any(over, axis=1)
    s_tgt = jnp.argmax(over, axis=1).astype(jnp.int32)
    sx, sy = unit_pos(units, s_tgt)
    storm_target = jnp.where(s_go, s_tgt, state.storm_target)
    storm_at = jnp.where(s_go, now + dv(STORMSURGE, "DelayDuration"), state.storm_at)
    storm_cd = jnp.where(s_go, now + dv(STORMSURGE, "Cooldown"), state.storm_cd)
    storm_x = jnp.where(s_go, sx, state.storm_x)
    storm_y = jnp.where(s_go, sy, state.storm_y)

    # Eternity: mana from pre-mitigation champion damage taken.
    eternity = holds(own, CATALYST) | holds(own, ROA)
    taken_raw = jnp.sum(jnp.where(taken_champ & (p.valid & (scls == CLASS_CHAMPION))[None, :],
                                  p.raw[None, :], 0.0), axis=1)
    mana = jnp.where(eternity & ctx.alive, dv(CATALYST, "EternityManaRestore") * taken_raw, 0.0)

    hp_frac = units.hp[dst] / jnp.maximum(units.max_hp[dst], 1e-6)
    sf_src = jnp.any(src_is & holds(own, SHADOWFLAME)[:, None] & enemy_dst, axis=0)       # (P,)
    sf = sf_src & landed & magic_true & ~struct & (p.item != SHADOWFLAME) \
        & (hp_frac < dv(SHADOWFLAME, "HealthThreshold"))
    sf_flags = TAG_ITEM | TAG_PROC | (p.flags & (TAG_PERIODIC | TAG_AOE))
    all_packets.append(packets(sf, p.src, p.dst, dv(SHADOWFLAME, "SpellItemDamageAmp") * p.raw, p.dtype,
                               sf_flags, amp=p.amp, item=SHADOWFLAME))

    bl_hit = per_unit(dealt & (magic & ability & champ)[None, :], p.dst, n) & holds(own, BLOODLETTER)[:, None] \
        & (now >= state.bl_icd)
    live = state.bl_until > now
    bl_stacks = jnp.where(bl_hit, jnp.minimum(jnp.where(live, state.bl_stacks, 0.0) + 1.0,
                                              dv(BLOODLETTER, "MaxStacks")), state.bl_stacks)
    bl_until = jnp.where(bl_hit, now + dv(BLOODLETTER, "DebuffDuration"), state.bl_until)
    bl_icd = jnp.where(bl_hit, now + dv(BLOODLETTER, "InternalCD"), state.bl_icd)

    state = state._replace(
        combat_last=combat_last, combat_start3=start3, combat_start4=start4,
        ashes_until=ashes_until, ashes_next=ashes_next, bf_until=bf_until, bf_next=bf_next, bf_qual=bf_qual,
        lia_until=lia_until, lia_next=lia_next, luden_cd=luden_cd, alt_cd=alt_cd, cosmic_until=cosmic_until,
        hz_until=hz_until, hz_cd=hz_cd, mal_x=mal_x, mal_y=mal_y, mal_r=mal_r, mal_until=mal_until,
        mal_next=mal_next, crypt_last=crypt_last, storm_hist=hist, storm_epoch=storm_epoch,
        storm_cd=storm_cd, storm_target=storm_target, storm_at=storm_at, storm_x=storm_x, storm_y=storm_y,
        bl_stacks=bl_stacks, bl_until=bl_until, bl_icd=bl_icd)
    return state, effects(c, n, packets=concat_packets(*all_packets), mana=mana, slow=slow,
                          slow_duration=slow_duration, grievous=grievous)


def periodic(state: State, own, ctx, units):
    c, n = ctx.level.shape[0], units.x.shape[0]
    now, dt = ctx.now, ctx.dt
    alive = units.alive[None, :]
    idx = jnp.arange(n)[None, :]
    src = ctx.unit[:, None]
    monster = (units.cls == CLASS_MONSTER)[None, :]
    minion_like = ~monster & (units.cls != CLASS_CHAMPION)[None, :]
    burn = TAG_PERIODIC | TAG_ITEM
    out = []

    # Burns end with the target.
    ashes_until = jnp.where(alive, state.ashes_until, -BIG)
    bf_until = jnp.where(alive, state.bf_until, -BIG)
    lia_until = jnp.where(alive, state.lia_until, -BIG)

    k, ashes_next = _ticks(ashes_until, state.ashes_next, now, dv(ASHES, "TickFrequency"))
    tf = dv(ASHES, "TickFrequency")
    per = dv(ASHES, "BurnFlatDamagePerSecond") * tf + jnp.where(monster, dv(ASHES, "MonsterDamageBonus") * tf, 0.0)
    out.append(packets((k > 0) & holds(own, ASHES)[:, None], src, idx, k * per, MAGIC, burn, item=ASHES))

    k, bf_next = _ticks(bf_until, state.bf_next, now, dv(BLACKFIRE, "TickFrequency"))
    tf = dv(BLACKFIRE, "TickFrequency")
    ap = ctx.ap[:, None]
    rate = jnp.where(monster, dv(BLACKFIRE, "MonsterDPS") + dv(BLACKFIRE, "MonsterAP") * ap,
                     jnp.where(minion_like, dv(BLACKFIRE, "MinionDPS") + dv(BLACKFIRE, "MinionAP") * ap,
                               dv(BLACKFIRE, "BurnFlatDamagePerSecond") + dv(BLACKFIRE, "APRatio") * ap))
    out.append(packets((k > 0) & holds(own, BLACKFIRE)[:, None], src, idx, k * rate * tf, MAGIC, burn,
                       item=BLACKFIRE))

    k, lia_next = _ticks(lia_until, state.lia_next, now, dv(LIANDRY, "TickFrequency"))
    tf = dv(LIANDRY, "TickFrequency")
    lrate = dv(LIANDRY, "BurnPercentHealthDamage") * units.max_hp[None, :]
    lrate = jnp.where(monster, jnp.minimum(lrate, dv(LIANDRY, "MonsterDamageCap")), lrate)
    out.append(packets((k > 0) & holds(own, LIANDRY)[:, None], src, idx, k * lrate * tf, MAGIC, burn,
                       item=LIANDRY))

    # Hatefog: one tick per unit per zone tick (max over overlapping zones).
    kz, mal_next = _ticks(state.mal_until, state.mal_next, now, MALIGNANCE_TICK)        # (C, Z)
    inside = _in_zones(state, own, ctx, units, inclusive=True)                                       # (C, Z, N)
    kn = jnp.max(jnp.where(inside, kz[:, :, None], 0.0), axis=1)                        # (C, N)
    zdmg = (dv(MALIGNANCE, "BaseDamage") + dv(MALIGNANCE, "APRatio") * ctx.ap) * MALIGNANCE_TICK
    out.append(packets(kn > 0, src, idx, kn * zdmg[:, None], MAGIC, burn | TAG_AOE, item=MALIGNANCE))

    pending = state.storm_target >= 0
    t = jnp.maximum(state.storm_target, 0)
    t_alive = units.alive[t]
    sx = jnp.where(pending & t_alive, units.x[t], state.storm_x)
    sy = jnp.where(pending & t_alive, units.y[t], state.storm_y)
    sq_dmg = dv(STORMSURGE, "BaseDamage") + dv(STORMSURGE, "APRatio") * ctx.ap
    strike = pending & t_alive & (now >= state.storm_at) & holds(own, STORMSURGE)
    burst = pending & ~t_alive & holds(own, STORMSURGE)
    out.append(packets(strike, ctx.unit, t, sq_dmg, MAGIC, TAG_ITEM | TAG_PROC, item=STORMSURGE))
    field = in_circle(units, sx, sy, jnp.full((c,), STORM_AOE)) & enemy_mask(ctx, units) \
        & (units.cls == CLASS_CHAMPION)[None, :] & burst[:, None]
    out.append(packets(field, src, idx, sq_dmg[:, None], MAGIC, TAG_ITEM | TAG_PROC | TAG_AOE, item=STORMSURGE))
    storm_target = jnp.where(strike | burst | (pending & ~holds(own, STORMSURGE)), -1, state.storm_target)

    chapter = holds(own, CHAPTER)
    give = state.ch_rem * jnp.clip(dt / jnp.maximum(state.ch_until - now + dt, 1e-6), 0.0, 1.0)
    give = jnp.where(chapter, give, 0.0)
    rem = jnp.where(chapter, state.ch_rem - give, 0.0)
    gained = jnp.where(state.ch_level >= 0.0, jnp.maximum(ctx.level - state.ch_level, 0.0), 0.0)
    up = chapter & (gained > 0)
    rem = rem + jnp.where(up, dv(CHAPTER, "ManaRestorePercent") * ctx.max_mana * gained, 0.0)
    ch_until = jnp.where(up, now + dv(CHAPTER, "RestorationDuration"), state.ch_until)

    # Rod of Ages clock resets when the item leaves the inventory.
    roa_elapsed = jnp.where(holds(own, ROA), state.roa_elapsed + dt, 0.0)

    state = state._replace(
        ashes_until=ashes_until, ashes_next=ashes_next, bf_until=bf_until, bf_next=bf_next,
        lia_until=lia_until, lia_next=lia_next, mal_next=mal_next, storm_target=storm_target,
        storm_x=sx, storm_y=sy, ch_rem=rem, ch_until=ch_until, ch_level=ctx.level.astype(jnp.float32),
        roa_elapsed=roa_elapsed)
    return state, effects(c, n, packets=concat_packets(*out), mana=give)


def on_takedown(state: State, own, ctx, units, kills):
    c, n = ctx.level.shape[0], units.x.shape[0]
    recent = (ctx.now - state.crypt_last) <= dv(CRYPTBLOOM, "TakedownWindow")
    champs = kills.killed_units & (units.cls == CLASS_CHAMPION)[None, :] & recent
    go = holds(own, CRYPTBLOOM) & ctx.alive & (ctx.now >= state.crypt_cd) & jnp.any(champs, axis=1)
    heal = jnp.where(go, dv(CRYPTBLOOM, "BaseHeal") + dv(CRYPTBLOOM, "HealAPRatio") * ctx.ap, 0.0)
    state = state._replace(crypt_cd=jnp.where(go, ctx.now + dv(CRYPTBLOOM, "Cooldown"), state.crypt_cd))
    return state, effects(c, n, heal=heal)
