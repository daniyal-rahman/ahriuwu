"""Sorcery tree 8200 (RUNES.md §5).

"Ability damage" (Comet, Deathfire, Scorch, Manaflow) is holder ``TAG_ACTIVE_SPELL`` damage that is not a basic
attack, item or proc, plus the holder's ``TAG_PET`` packets; rune packets never trigger Sorcery runes, and
triggers need ``final > 0`` on a living enemy champion. Delayed damage is queued by ``on_damage`` (stats
snapshotted) and emitted by ``periodic``.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from ...core.damage import (CLASS_CHAMPION, MAGIC, PROP_SUMMONER, PROP_ULTIMATE, TAG_ACTIVE_SPELL, TAG_AOE,
                            TAG_BASIC_ATTACK, TAG_INDIRECT, TAG_ITEM, TAG_PERIODIC, TAG_PET, TAG_PROC,
                            concat_packets, has, packets)
from ...items.catalog import ItemStats, catalog
from .core import (BIG, RuneOutputs, adaptive_damage_type, ea, effects, has_rune, in_circle, lin, no_outputs,
                   rune_item, unit_pos, variable_damage_type)

AERY, COMET, STORMRAIDER, DEATHFIRE = 8214, 8229, 8230, 8992
AXIOM, MANAFLOW, NIMBUS = 8224, 8226, 8275
TRANSCENDENCE, CELERITY, ABSOLUTE_FOCUS = 8210, 8234, 8233
SCORCH, WATERWALKING, GATHERING_STORM = 8237, 8232, 8236

COVERAGE = {
    STORMRAIDER: "Stormraider's Surge (not Phase Rush): post-mitigation damage >= 25% of a champion's max HP in a "
                 "sliding window (0.1 s buckets) -> MS and slow resist",
    AERY: "attack/ability/item damage on a champion (not persistent damage; Ignite's first tick via a gap) -> "
          "delayed adaptive damage; return-time model (U-09). Ally shield not implemented (no ally-buff event)",
    COMET: "ability/pet damage on a champion -> lands at the trigger position, distance-amped variable damage to "
           "champions in the radius; no projectile blocking",
    DEATHFIRE: "ability/pet damage -> snapshotted magic burn, amped after continuous burning; spell/AoE/"
               "persistent-or-pet durations with the §5.4 refresh rule",
    AXIOM: "amp on the holder's PROP_ULTIMATE packets (AoE lower); takedown refunds current R cooldown. "
           "Ultimate heal/shield amp not implemented",
    MANAFLOW: "ability damage or CC on a champion -> max mana stacks; at cap restores % missing mana periodically",
    NIMBUS: "summoner cast -> ghosting + MS by cooldown bracket (Teleport top), decaying (U-10); strongest wins",
    TRANSCENDENCE: "ability haste at two levels; later, takedowns cut current basic cooldowns",
    CELERITY: "other bonus MS amp (bonus_ms_amp) and its own MS (U-11)",
    ABSOLUTE_FOCUS: "AF while above the HP threshold",
    SCORCH: "ability damage on a champion -> delayed magic damage; cooldown",
    WATERWALKING: "in river: MS (decaying after leaving) and AF (lost immediately)",
    GATHERING_STORM: "AF 4·m(m−1), m = 1 + floor(game_time / 10 min)",
}

SR_THRESHOLD = ea(STORMRAIDER, "DamageThreshold")
SR_WINDOW = ea(STORMRAIDER, "Window")
SR_DURATION = ea(STORMRAIDER, "Duration")
SR_HASTE = ea(STORMRAIDER, "HasteMax")
SR_RANGED = ea(STORMRAIDER, "RangedEffectiveness")
SR_SLOW_RESIST = ea(STORMRAIDER, "SlowResist")
SR_CD = (ea(STORMRAIDER, "MaxCooldown"), ea(STORMRAIDER, "MinCooldown"))
SR_BUCKET = 0.1                       # RUNES §5.1
SR_WINDOW_BUCKETS = int(round(SR_WINDOW / SR_BUCKET))
SR_SLOTS = SR_WINDOW_BUCKETS + 1

# Aery travel/linger/return are not in client data (RUNES §5.2, U-09).
AERY_TRAVEL = 0.45
AERY_LINGER = 2.0
AERY_RETURN_SPEED = ((1, 200.0), (6, 300.0), (11, 600.0))
AERY_ACCEL = 10.0 / 51.0              # +10 speed per 51 units travelled
AERY_NEAR = 200.0
AERY_NEAR_ACCEL = 2000.0 / 51.0       # within 200 units: +2000 per 51 units
AERY_SUMMONER_GAP = 1.5               # summoner damage after this gap is a new Ignite (first tick)

# Comet flight and radius: wiki, RUNES §5.3.
COMET_DELAY = 0.8
COMET_RADIUS = 140.0
COMET_MAX_RANGE = ea(COMET, "MaxRange")
COMET_MAX_AMP = ea(COMET, "MaxDamageAmp")

DFT_TICK = 1.0 / ea(DEATHFIRE, "TicksPerSecond")
DFT_TIME_TO_AMP = ea(DEATHFIRE, "TimeToAmp")
DFT_AMP = 1.0 + ea(DEATHFIRE, "DamageMultiplier")
DFT_SPELL, DFT_AOE, DFT_DOT = ea(DEATHFIRE, "Duration"), ea(DEATHFIRE, "AoEDuration"), ea(DEATHFIRE, "DotDuration")

SCORCH_DELAY = ea(SCORCH, "DotDuration")
SCORCH_CD = ea(SCORCH, "BurnlockoutDuration")
MF_CAP = int(round(ea(MANAFLOW, "MaxManaIncrease") / ea(MANAFLOW, "ManaIncrease")))

EPS = 1e-3


class State(NamedTuple):
    sr_buf: Any             # (C, N, S) post-mitigation damage per 0.1 s bucket
    sr_bucket: Any          # (C, S) int32 absolute bucket of each slot (-1 empty)
    sr_until: Any           # (C,) haste end
    sr_cd_until: Any
    aery_due: Any           # (C,) pending Aery hit time (BIG = none)
    aery_dst: Any
    aery_raw: Any
    aery_dtype: Any
    aery_free_t: Any        # Aery back with the owner
    aery_summ_last: Any     # last summoner damage dealt to an enemy champion
    comet_due: Any
    comet_x: Any
    comet_y: Any
    comet_raw: Any
    comet_dtype: Any
    comet_cd_until: Any
    dft_start: Any          # (C, N) continuous burn start
    dft_end: Any
    dft_total: Any          # total duration of the current application
    dft_next: Any           # next tick time (> dft_end + EPS = no burn)
    dft_dmg: Any            # per-tick snapshot before the 3 s amp
    scorch_due: Any
    scorch_dst: Any
    scorch_raw: Any
    scorch_cd_until: Any
    mf_stacks: Any          # (C,) int32 Manaflow stacks (×25 max mana)
    mf_cd_until: Any
    mf_next_restore: Any
    nim_ms: Any             # Nimbus MS fraction at activation
    nim_start: Any
    ww_last_river: Any      # last tick spent in the river
    basic_refund: Any       # Transcendence refund fraction written this tick
    ult_refund: Any         # Axiom refund fraction written this tick


def init(n_champions: int, n_units: int) -> State:
    c, n = n_champions, n_units
    z = jnp.zeros((c,), jnp.float32)
    zi = jnp.zeros((c,), jnp.int32)
    zn = jnp.zeros((c, n), jnp.float32)
    return State(
        jnp.zeros((c, n, SR_SLOTS), jnp.float32), jnp.full((c, SR_SLOTS), -1, jnp.int32), z - BIG, z - BIG,
        z + BIG, zi, z, zi, z - BIG, z - BIG,
        z + BIG, z, z, z, zi, z - BIG,
        zn - BIG, zn - BIG, zn, zn + BIG, zn,
        z + BIG, zi, z, z - BIG,
        zi, z - BIG, z + BIG,
        z, z - BIG, z - BIG, z, z)


def _dealt(p, r, ctx, units):
    """(C, P) holder's non-rune packet landed (> 0) on an enemy champion."""
    n = units.x.shape[0]
    dst = jnp.clip(p.dst, 0, n - 1)
    champ = (units.cls[dst] == CLASS_CHAMPION) & (p.item >= 0) & p.valid & (r.final > 0.0)
    enemy = units.team[dst][None, :] != ctx.team[:, None]
    return (p.src[None, :] == ctx.unit[:, None]) & champ[None, :] & enemy & ctx.alive[:, None]


def _ability(p):
    f = p.flags
    spell = has(f, TAG_ACTIVE_SPELL) & ~has(f, TAG_BASIC_ATTACK) & ~has(f, TAG_ITEM) & ~has(f, TAG_PROC)
    return spell | has(f, TAG_PET)


def _first(sel):
    """(C,) any, (C,) index of the first selected packet."""
    return jnp.any(sel, axis=1), jnp.argmax(sel, axis=1).astype(jnp.int32)


def _per_unit(sel, p, n, value):
    """(C, N) max of ``value`` (P,) over selected packets per destination unit (0 if none)."""
    onehot = p.dst[:, None] == jnp.arange(n)[None, :]                         # (P, N)
    v = jnp.where(sel[:, :, None] & onehot[None], value[None, :, None], 0.0)
    return jnp.max(v, axis=1) if sel.shape[1] else jnp.zeros((sel.shape[0], n), jnp.float32)


def _aery_return_time(dist, level):
    v0 = jnp.full(dist.shape, AERY_RETURN_SPEED[0][1], jnp.float32)
    for lv, v in AERY_RETURN_SPEED[1:]:
        v0 = jnp.where(level >= lv, v, v0)
    far = jnp.maximum(dist - AERY_NEAR, 0.0)
    t1 = jnp.log1p(AERY_ACCEL * far / v0) / AERY_ACCEL
    v1 = v0 + AERY_ACCEL * far
    k2 = AERY_ACCEL + AERY_NEAR_ACCEL
    t2 = jnp.log1p(k2 * jnp.minimum(dist, AERY_NEAR) / v1) / k2
    return t1 + t2


def stormraider_cooldown(level):
    return lin(SR_CD[0], SR_CD[1], level)


def comet_cooldown(level):
    return jnp.clip(lin(20.0, ea(COMET, "RechargeTimeMin"), level), 0.3, 20.0)


def _nimbus_now(state, now):
    left = jnp.clip(1.0 - (now - state.nim_start) / ea(NIMBUS, "Duration"), 0.0, 1.0)
    return state.nim_ms * left, (now - state.nim_start) < ea(NIMBUS, "Duration")


def stats(state: State, page, ctx, ev) -> ItemStats:
    now, lv = ctx.now, ctx.level
    sr_on = has_rune(page, STORMRAIDER) & (now < state.sr_until)
    sr_ms = jnp.where(sr_on, SR_HASTE * jnp.where(ctx.is_ranged, SR_RANGED, 1.0), 0.0)
    sr_resist = jnp.where(sr_on, SR_SLOW_RESIST, 0.0)
    nim, _ = _nimbus_now(state, now)
    nim = jnp.where(has_rune(page, NIMBUS), nim, 0.0)
    # Celerity's own MS is pre-divided so the runtime's bonus_ms_amp leaves it exact.
    cel = has_rune(page, CELERITY)
    cel_amp = ea(CELERITY, "PercentHasteMod")
    cel_ms = jnp.where(cel, ea(CELERITY, "PercentMS") / (1.0 + cel_amp), 0.0)
    ww = has_rune(page, WATERWALKING)
    river = jnp.asarray(ev.in_river, bool)
    ww_decay = jnp.clip(1.0 - (now - state.ww_last_river) / ea(WATERWALKING, "{6f2f0d30}", 1.0), 0.0, 1.0)
    ww_ms = jnp.where(ww, ea(WATERWALKING, "MovementSpeed") * jnp.where(river, 1.0, ww_decay), 0.0)
    ww_af = jnp.where(ww & river, lin(ea(WATERWALKING, "MinAdaptive"), ea(WATERWALKING, "MaxAdaptive"), lv), 0.0)
    af_on = has_rune(page, ABSOLUTE_FOCUS) & (ctx.hp > ea(ABSOLUTE_FOCUS, "HealthPercent") * ctx.max_hp)
    abs_af = jnp.where(af_on, lin(ea(ABSOLUTE_FOCUS, "MinAdaptive"), ea(ABSOLUTE_FOCUS, "MaxAdaptive"), lv), 0.0)
    m = 1.0 + jnp.floor(jnp.asarray(ev.game_time, jnp.float32) / (60.0 * ea(GATHERING_STORM, "UpdateAfterMinutes")))
    gs_af = jnp.where(has_rune(page, GATHERING_STORM), ea(GATHERING_STORM, "AdaptiveAP") / 2.0 * m * (m - 1.0), 0.0)
    tr = has_rune(page, TRANSCENDENCE)
    tr_ah = 100.0 * (jnp.where(lv >= ea(TRANSCENDENCE, "LevelToTurnOn"), ea(TRANSCENDENCE, "HasteBonus1"), 0.0)
                     + jnp.where(lv >= ea(TRANSCENDENCE, "LevelToTurnOn2"), ea(TRANSCENDENCE, "HasteBonus2"), 0.0))
    mana = jnp.where(has_rune(page, MANAFLOW), ea(MANAFLOW, "ManaIncrease") * state.mf_stacks, 0.0)
    return ItemStats(
        percent_move_speed=sr_ms + nim + cel_ms, slow_resist=sr_resist, move_speed=ww_ms,
        bonus_ms_amp=jnp.where(cel, cel_amp, 0.0), adaptive_force=ww_af + abs_af + gs_af,
        ability_haste=jnp.where(tr, tr_ah, 0.0), mana=mana)


def packet_amp(state: State, page, ctx, units, ev, p):
    """Axiom Arcanist."""
    mine = (p.src[None, :] == ctx.unit[:, None]) & has_rune(page, AXIOM)[:, None]          # (C, P)
    ult = p.valid & has(p.flags, PROP_ULTIMATE)
    amt = jnp.where(has(p.flags, TAG_AOE), ea(AXIOM, "AOEAmp"), ea(AXIOM, "DamageAmp"))
    return jnp.where(jnp.any(mine, axis=0) & ult, amt, 0.0).astype(jnp.float32)


def on_damage(state: State, page, ctx, units, ev):
    rep = ev.report
    p, r = rep.packets, rep.resolved
    c, n = ctx.level.shape[0], units.x.shape[0]
    now, lv = ctx.now, ctx.level
    dealt = _dealt(p, r, ctx, units)
    ability = dealt & _ability(p)[None, :]

    # Stormraider's Surge: sliding window per enemy champion.
    k = jnp.floor(now / SR_BUCKET + EPS).astype(jnp.int32)
    slot = k % SR_SLOTS
    here = jnp.arange(SR_SLOTS)[None, :] == slot
    stale = here & (state.sr_bucket != k)                                                  # (C, S)
    buf = jnp.where(stale[:, None, :], 0.0, state.sr_buf)
    bucket = jnp.where(here, k, state.sr_bucket)
    add = _sum_per_unit(dealt, p, r.final, n)                                              # (C, N)
    buf = buf + jnp.where(here[:, None, :], add[:, :, None], 0.0)
    live = (bucket > k - SR_WINDOW_BUCKETS) & (bucket >= 0)
    window = jnp.sum(jnp.where(live[:, None, :], buf, 0.0), axis=2)
    champ = (units.cls[None, :] == CLASS_CHAMPION) & (units.team[None, :] != ctx.team[:, None])
    over = champ & (window >= SR_THRESHOLD * units.max_hp[None, :])
    sr_go = has_rune(page, STORMRAIDER) & ctx.alive & (now >= state.sr_cd_until) & jnp.any(over, axis=1)
    buf = jnp.where(sr_go[:, None, None], 0.0, buf)
    state = state._replace(
        sr_buf=buf.astype(jnp.float32), sr_bucket=bucket,
        sr_until=jnp.where(sr_go, now + SR_DURATION, state.sr_until),
        sr_cd_until=jnp.where(sr_go, now + stormraider_cooldown(lv), state.sr_cd_until))

    # Summon Aery.
    f = p.flags
    summ = has(f, PROP_SUMMONER)
    summ_first = (now - state.aery_summ_last > AERY_SUMMONER_GAP)[:, None] & summ[None, :]
    kind = (has(f, TAG_BASIC_ATTACK) | has(f, TAG_ACTIVE_SPELL) | has(f, TAG_ITEM)) & ~has(f, TAG_PERIODIC) \
        & ~summ
    aery_sel = dealt & (kind[None, :] | summ_first)
    any_a, ia = _first(aery_sel)
    a_go = has_rune(page, AERY) & any_a & (now >= state.aery_free_t)
    a_dst = p.dst[ia]
    hx, hy = unit_pos(units, ctx.unit)
    tx, ty = unit_pos(units, a_dst)
    dist = jnp.sqrt((tx - hx) ** 2 + (ty - hy) ** 2)
    a_raw = lin(ea(AERY, "DamageBase"), ea(AERY, "DamageMax"), lv) + ea(AERY, "DamageADRatio") * ev.bonus_ad \
        + ea(AERY, "DamageAPRatio") * ev.ap
    state = state._replace(
        aery_due=jnp.where(a_go, now + AERY_TRAVEL, state.aery_due),
        aery_dst=jnp.where(a_go, a_dst, state.aery_dst),
        aery_raw=jnp.where(a_go, a_raw, state.aery_raw).astype(jnp.float32),
        aery_dtype=jnp.where(a_go, adaptive_damage_type(ev), state.aery_dtype).astype(jnp.int32),
        aery_free_t=jnp.where(a_go, now + AERY_TRAVEL + AERY_LINGER + _aery_return_time(dist, lv),
                              state.aery_free_t),
        aery_summ_last=jnp.where(jnp.any(dealt & summ[None, :], axis=1), now, state.aery_summ_last))

    # Arcane Comet.
    any_c, ic = _first(ability)
    c_go = has_rune(page, COMET) & any_c & (now >= state.comet_cd_until)
    cx, cy = unit_pos(units, p.dst[ic])
    cdist = jnp.minimum(jnp.sqrt((cx - hx) ** 2 + (cy - hy) ** 2), COMET_MAX_RANGE)
    ad_t, ap_t = ea(COMET, "ADRatio") * ev.bonus_ad, ea(COMET, "APRatio") * ev.ap
    c_raw = (lin(ea(COMET, "DamageBase"), ea(COMET, "DamageMax"), lv) + ad_t + ap_t) \
        * (1.0 + COMET_MAX_AMP * cdist / COMET_MAX_RANGE)
    state = state._replace(
        comet_due=jnp.where(c_go, now + COMET_DELAY, state.comet_due),
        comet_x=jnp.where(c_go, cx, state.comet_x), comet_y=jnp.where(c_go, cy, state.comet_y),
        comet_raw=jnp.where(c_go, c_raw, state.comet_raw).astype(jnp.float32),
        comet_dtype=jnp.where(c_go, variable_damage_type(ad_t, ap_t), state.comet_dtype).astype(jnp.int32),
        comet_cd_until=jnp.where(c_go, now + comet_cooldown(lv), state.comet_cd_until))

    # Deathfire Touch: per-target burn, §5.4 refresh rule.
    dur = jnp.where(has(f, TAG_PET) | has(f, TAG_PERIODIC), DFT_DOT, jnp.where(has(f, TAG_AOE), DFT_AOE, DFT_SPELL))
    new_dur = _per_unit(ability, p, n, dur)                                                # (C, N)
    burning = state.dft_next <= state.dft_end + EPS
    remaining = state.dft_end - now
    apply = has_rune(page, DEATHFIRE)[:, None] & (new_dur > 0.0) \
        & (~burning | (new_dur >= state.dft_total) | (remaining < new_dur))
    tick = (lin(ea(DEATHFIRE, "Level1DamageTOOLTIP"), ea(DEATHFIRE, "{2156e250}"), lv)
            + ea(DEATHFIRE, "ADRatio") * ev.bonus_ad + ea(DEATHFIRE, "APRatio") * ev.ap) * DFT_TICK
    state = state._replace(
        dft_start=jnp.where(apply & ~burning, now, state.dft_start),
        dft_end=jnp.where(apply, now + new_dur, state.dft_end),
        dft_total=jnp.where(apply, new_dur, state.dft_total).astype(jnp.float32),
        dft_next=jnp.where(apply & ~burning, now + DFT_TICK, state.dft_next),
        dft_dmg=jnp.where(apply, tick[:, None], state.dft_dmg).astype(jnp.float32))

    s_go = has_rune(page, SCORCH) & any_c & (now >= state.scorch_cd_until)
    state = state._replace(
        scorch_due=jnp.where(s_go, now + SCORCH_DELAY, state.scorch_due),
        scorch_dst=jnp.where(s_go, p.dst[ic], state.scorch_dst),
        scorch_raw=jnp.where(s_go, lin(ea(SCORCH, "Damage"), ea(SCORCH, "DamageMax"), lv),
                             state.scorch_raw).astype(jnp.float32),
        scorch_cd_until=jnp.where(s_go, now + SCORCH_CD, state.scorch_cd_until))

    state = _manaflow_stack(state, page, ctx, any_c)
    return state, effects(c, n)


def _sum_per_unit(sel, p, amount, n):
    onehot = (p.dst[:, None] == jnp.arange(n)[None, :]).astype(jnp.float32)
    return jnp.where(sel, amount[None, :], 0.0) @ onehot


def _manaflow_stack(state, page, ctx, trigger):
    go = has_rune(page, MANAFLOW) & ctx.alive & trigger & (ctx.now >= state.mf_cd_until) & (state.mf_stacks < MF_CAP)
    stacks = state.mf_stacks + go.astype(jnp.int32)
    capped = go & (stacks >= MF_CAP)
    return state._replace(
        mf_stacks=stacks, mf_cd_until=jnp.where(go, ctx.now + ea(MANAFLOW, "Cooldown"), state.mf_cd_until),
        mf_next_restore=jnp.where(capped, ctx.now + ea(MANAFLOW, "PercentManaRestoreCooldown"),
                                  state.mf_next_restore))


def on_cc(state: State, page, ctx, units, ev):
    """Manaflow Band stacks from impairing an enemy champion."""
    c, n = ctx.level.shape[0], units.x.shape[0]
    champ = (units.cls[None, :] == CLASS_CHAMPION) & (units.team[None, :] != ctx.team[:, None]) \
        & units.alive[None, :]
    hit = jnp.any((ev.cc.slowed | ev.cc.immobilized) & champ, axis=1)
    return _manaflow_stack(state, page, ctx, hit), effects(c, n)


def on_cast(state: State, page, ctx, units, ev):
    """Nimbus Cloak."""
    c, n = ctx.level.shape[0], units.x.shape[0]
    cd = ev.summoner_cooldown
    lo, hi = ea(NIMBUS, "{2fd68801}"), ea(NIMBUS, "{b0d06764}")
    boost = jnp.where(cd < lo, ea(NIMBUS, "LowCDMSBoost"),
                      jnp.where(cd <= hi, ea(NIMBUS, "{1c32110c}"), ea(NIMBUS, "HighCDMSBoost")))
    boost = jnp.where(ev.summoner_is_teleport, ea(NIMBUS, "HighCDMSBoost"), boost)
    cur, _ = _nimbus_now(state, ctx.now)
    go = has_rune(page, NIMBUS) & ev.summoner_cast & ctx.alive & (boost >= cur)
    return state._replace(nim_ms=jnp.where(go, boost, state.nim_ms).astype(jnp.float32),
                          nim_start=jnp.where(go, ctx.now, state.nim_start)), effects(c, n)


def periodic(state: State, page, ctx, units, ev):
    c, n = ctx.level.shape[0], units.x.shape[0]
    now = ctx.now
    holder = ctx.unit
    alive_at = lambda idx: units.alive[jnp.clip(idx, 0, n - 1)]

    a_hit = (now >= state.aery_due - EPS) & alive_at(state.aery_dst)
    p_aery = packets(a_hit, holder, state.aery_dst, state.aery_raw, state.aery_dtype, TAG_PROC, item=rune_item(AERY))
    aery_due = jnp.where(now >= state.aery_due - EPS, BIG, state.aery_due)

    # Comet.
    land = now >= state.comet_due - EPS
    champ = (units.cls[None, :] == CLASS_CHAMPION) & (units.team[None, :] != ctx.team[:, None]) \
        & units.alive[None, :] & units.targetable[None, :]
    area = in_circle(units, state.comet_x, state.comet_y, jnp.full((c,), COMET_RADIUS)) & champ & land[:, None]
    p_comet = packets(area, holder[:, None], jnp.arange(n)[None, :], state.comet_raw[:, None],
                      state.comet_dtype[:, None], TAG_PROC | TAG_AOE, item=rune_item(COMET))
    comet_due = jnp.where(land, BIG, state.comet_due)

    # Deathfire.
    due = (now >= state.dft_next - EPS) & (state.dft_next <= state.dft_end + EPS) & units.alive[None, :]
    amp = jnp.where(state.dft_next - state.dft_start >= DFT_TIME_TO_AMP - EPS, DFT_AMP, 1.0)
    p_dft = packets(due, holder[:, None], jnp.arange(n)[None, :], state.dft_dmg * amp, MAGIC,
                    TAG_PROC | TAG_PERIODIC, item=rune_item(DEATHFIRE))
    dft_next = jnp.where(due, state.dft_next + DFT_TICK, state.dft_next)
    dead = ~units.alive[None, :]
    dft_next = jnp.where(dead, BIG, dft_next)

    # Scorch.
    s_hit = (now >= state.scorch_due - EPS) & alive_at(state.scorch_dst)
    p_scorch = packets(s_hit, holder, state.scorch_dst, state.scorch_raw, MAGIC,
                       TAG_PROC | TAG_PERIODIC | TAG_INDIRECT, item=rune_item(SCORCH))
    scorch_due = jnp.where(now >= state.scorch_due - EPS, BIG, state.scorch_due)

    # Manaflow restore (ctx.max_mana excludes the rune's own mana).
    restore = has_rune(page, MANAFLOW) & ctx.alive & (state.mf_stacks >= MF_CAP) & (now >= state.mf_next_restore - EPS)
    max_mana = ctx.max_mana + ea(MANAFLOW, "ManaIncrease") * state.mf_stacks
    mana = jnp.where(restore, ea(MANAFLOW, "PercentManaRestore") * jnp.maximum(max_mana - ctx.mana, 0.0), 0.0)
    mf_next = jnp.where(restore, state.mf_next_restore + ea(MANAFLOW, "PercentManaRestoreCooldown"),
                        state.mf_next_restore)

    ww_last = jnp.where(jnp.asarray(ev.in_river, bool), now, state.ww_last_river)

    state = state._replace(aery_due=aery_due, comet_due=comet_due, dft_next=dft_next.astype(jnp.float32),
                           scorch_due=scorch_due, mf_next_restore=mf_next,
                           ww_last_river=jnp.broadcast_to(ww_last, (c,)).astype(jnp.float32))
    return state, effects(c, n, packets=concat_packets(p_aery, p_comet, p_dft, p_scorch),
                          mana=mana.astype(jnp.float32))


def on_takedown(state: State, page, ctx, units, ev):
    """This tick's Axiom / Transcendence refunds (0 without a takedown)."""
    c, n = ctx.level.shape[0], units.x.shape[0]
    k = jnp.asarray(ev.kills.champion_kill, jnp.float32) + jnp.asarray(ev.kills.champion_assist, jnp.float32)
    ult = 1.0 - (1.0 - ea(AXIOM, "UltimateRefundBase") / 100.0) ** k
    basic = 1.0 - (1.0 - ea(TRANSCENDENCE, "KillCooldownRefund")) ** k
    tr_on = has_rune(page, TRANSCENDENCE) & (ctx.level >= ea(TRANSCENDENCE, "LevelToTurnOn3"))
    return state._replace(ult_refund=jnp.where(has_rune(page, AXIOM), ult, 0.0).astype(jnp.float32),
                          basic_refund=jnp.where(tr_on, basic, 0.0).astype(jnp.float32)), effects(c, n)


def outputs(state: State, page, ctx, ev) -> RuneOutputs:
    out = no_outputs(ctx.level.shape[0], len(catalog().ids))
    _, active = _nimbus_now(state, ctx.now)
    return out._replace(basic_cd_refund=state.basic_refund, ult_cd_refund=state.ult_refund,
                        ghosted=has_rune(page, NIMBUS) & active)
