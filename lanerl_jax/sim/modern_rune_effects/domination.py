"""Domination tree 8100 (RUNES.md §4; client build 16.19.8230722).

Numbers come from ``runes_client.json`` (``ea``); level scaling is
``lin`` (RUNES §1.1, extrapolating past 18 per U-01). Hard-coded values are
the ones the client data lacks, each citing RUNES.md:

* Electrocute 0.25 s damage delay (§4.1, wiki); 3 stacks (§4.1).
* Dark Harvest 1.75 s soul delay and the 2-damage floor (§4.2, §1.5).
* Bounty Hunter cap of 5 unique champions (§4.8).

Defaults for open rules:

* Electrocute stacks: 1 per cast instance per champion (``first_instance``
  within a tick plus an ``ELEC_SEEN``-slot (cast_id, dst) ring across
  ticks); ``cast_id`` 0 packets are each their own instance. Proc packets
  stack only with ``TAG_PET`` (§1.3). CC (``ev.cc``) on an enemy champion
  adds a stack in ``on_cc``; the first damage instance on that champion in
  the same tick is treated as the same instance (packets do not tie CC to a
  cast). Stacks are not gained on cooldown. Damage is snapshotted at the
  trigger and dealt after the delay even if the holder died; the target
  must still be alive.
* Dark Harvest: post-hit HP threshold (U-07) read from the resolved pass;
  post-mitigation ``final >= 2``; the target must survive the trigger hit.
  Two pending-soul slots (a takedown reset to 1 s can re-trigger inside the
  1.75 s soul delay).
* Hail of Blades: stacks are consumed by every on-attack while active (any
  target); the duration refreshes only on attacks against champions. A
  cancelled triggering windup locks the rune for ``HOB_CANCEL_LOCKOUT`` s
  (wiki "brief cooldown"; no client value). The on-hit true damage goes to
  the empowered attack's target when it lands.
* Cheap Shot: ``ev.impaired`` is tick-start state, so CC applied in the same
  tick never counts (the "CC applied on-hit by that instance" exception is
  not modelled).
* Taste of Blood: instant heal (U-08); any damage packet (incl. 0 and rune
  procs) to an enemy champion while the holder is below max HP after the pass.
* Sudden Impact: arms on ``ev.blinked``; triggers on any damage packet
  (incl. 0) to an enemy champion except its own.
* Relentless Hunter uses the generic ``last_combat`` clock (the "modern"
  combat system is not tracked separately).
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from ..modern_damage import CLASS_CHAMPION, TAG_ON_HIT, TAG_PET, TAG_PROC, TRUE, concat_packets, has, packets
from ..modern_item_data import ItemStats
from .core import (BIG, COMBAT_TIMEOUT, adaptive_damage_type, by_range, ea, effects, first_instance,
                   has_rune, lin, rune_item, variable_damage_type)

ELECTROCUTE, DARK_HARVEST, HAIL_OF_BLADES = 8112, 8128, 9923
CHEAP_SHOT, TASTE_OF_BLOOD, SUDDEN_IMPACT = 8126, 8139, 8143
GRISLY_MEMENTOS = 8140
TREASURE_HUNTER, RELENTLESS_HUNTER, ULTIMATE_HUNTER = 8135, 8105, 8106

ELEC_STACKS = 3            # RUNES §4.1 "3 stacks within 3 s"
ELEC_DELAY = 0.25          # RUNES §4.1 "after a 0.25 s delay"
ELEC_SEEN = 8              # (cast_id, dst) ring per holder
DH_SOUL_DELAY = 1.75       # RUNES §4.2
DH_MIN_DAMAGE = 2.0        # RUNES §1.5 "Dark Harvest excludes damage < 2"
DH_SOUL_SLOTS = 2
HOB_CANCEL_LOCKOUT = 1.0   # RUNES §4.3 "brief cooldown" (no client value; default)
BOUNTY_MAX = 5             # RUNES §4.8

COVERAGE = {
    ELECTROCUTE: "3 stacks (1 per cast instance per champion, incl. CC) within 3 s -> after 0.25 s "
                 "lin(70,240) + 10% bonus AD + 5% AP variable damage, proc; cd 20 s",
    DARK_HARVEST: "non-proc damage >= 2 to a champion below 50% (post-hit, U-07): 30 + 11/soul + 10% bonus AD "
                  "+ 5% AP adaptive, proc; +1 soul after 1.75 s; cd 35 s, takedown -> 1 s; "
                  "execute-credit souls while ready",
    HAIL_OF_BLADES: "windup on a champion -> 3 empowered attacks (+2 per activation from attack resets), "
                    "+90%/60% AS with cap lift, lin(2,20) + 12% bonus AD + 10% AP true on-hit proc; "
                    "3 s refresh window; cd 10 s after the end; cancelled windup -> 1 s lockout (default)",
    CHEAP_SHOT: "non-proc damage to an already-impaired champion: lin(10,45) true proc; cd 4 s "
                "(on-hit same-instance CC counts via ev.cc_on_hit/cc_cast_id)",
    TASTE_OF_BLOOD: "damage to a champion below full HP: instant heal lin(16,40) + 10% bonus AD + 5% AP "
                    "(U-08); cd 20 s",
    SUDDEN_IMPACT: "dash/blink/Flash/TP/stealth exit arms 4 s; first damage to a champion: lin(20,80) true "
                   "proc; cd 10 s after use or expiry",
    GRISLY_MEMENTOS: "+1 memento per champion takedown (max 18), +6 trinket haste each (trinkets deferred)",
    TREASURE_HUNTER: "Bounty Hunter stacks (unique champion takedowns, max 5): 50 + 20*stacks_before gold each",
    RELENTLESS_HUNTER: "+8 flat MS per Bounty Hunter stack while out of combat (5 s after the modern combat clock)",
    ULTIMATE_HUNTER: "6 + 5 per Bounty Hunter stack ultimate haste",
}


class State(NamedTuple):
    elec_first_t: Any       # (C, N) first stack of the current window
    elec_stacks: Any        # (C, N) int32
    elec_cc_t: Any          # (C, N) tick of an unpaired CC stack (pairs with that tick's first damage)
    elec_seen_id: Any       # (C, K) int32 cast ids that already stacked
    elec_seen_dst: Any      # (C, K) int32 their target
    elec_seen_ptr: Any      # (C,) int32 ring write position
    elec_cd_until: Any      # (C,)
    elec_due: Any           # (C,) delayed hit time (BIG = none)
    elec_dst: Any           # (C,) int32
    elec_raw: Any           # (C,) snapshotted damage
    elec_dtype: Any         # (C,) int32
    dh_souls: Any           # (C,)
    dh_cd_until: Any        # (C,)
    dh_soul_due: Any        # (C, DH_SOUL_SLOTS)
    hob_pending: Any        # (C,) bool triggering windup in progress
    hob_active: Any         # (C,) bool
    hob_stacks: Any         # (C,) int32 empowered attacks left
    hob_expire: Any         # (C,)
    hob_bonus_used: Any     # (C,) int32 reset stacks granted this activation
    hob_cd_until: Any       # (C,)
    hob_inflight: Any       # (C,) bool launched empowered attack awaiting its hit
    cs_cd_until: Any        # (C,)
    tob_cd_until: Any       # (C,)
    si_armed_until: Any     # (C,) BIG-negative = not armed
    si_cd_until: Any        # (C,)
    bounty: Any             # (C, N) bool unique enemy champion takedowns
    mementos: Any           # (C,)


def init(n_champions: int, n_units: int) -> State:
    c, n = n_champions, n_units
    z = jnp.zeros((c,), jnp.float32)
    zi = jnp.zeros((c,), jnp.int32)
    f = jnp.zeros((c,), bool)
    never = z - BIG
    return State(
        jnp.full((c, n), -BIG, jnp.float32), jnp.zeros((c, n), jnp.int32), jnp.full((c, n), -BIG, jnp.float32),
        jnp.zeros((c, ELEC_SEEN), jnp.int32), jnp.full((c, ELEC_SEEN), -1, jnp.int32), zi, never,
        z + BIG, zi, z, zi,
        z, never, jnp.full((c, DH_SOUL_SLOTS), BIG, jnp.float32),
        f, f, zi, never, zi, never, f,
        never, never, never, never,
        jnp.zeros((c, n), bool), z)


# ---- helpers ------------------------------------------------------------------

def _f32(x):
    return jnp.asarray(x).astype(jnp.float32)


def _enemy_champ_units(ctx, units):
    """(C, N) enemy champion units (dead ones included: kills still count)."""
    return (units.cls[None, :] == CLASS_CHAMPION) & (units.team[None, :] != ctx.team[:, None])


def _to_enemy_champ(p, ctx, units):
    """(C, P) valid packet from holder c onto an enemy champion."""
    n = units.x.shape[0]
    d = jnp.clip(p.dst, 0, n - 1)
    champ = (units.cls[d] == CLASS_CHAMPION)[None, :] & (units.team[d][None, :] != ctx.team[:, None])
    return p.valid[None, :] & (p.src[None, :] == ctx.unit[:, None]) & champ


def _non_proc(p):
    """Proc damage never triggers unless it is also pet damage (RUNES §1.3)."""
    return ~has(p.flags, TAG_PROC) | has(p.flags, TAG_PET)


def _first_dst(sel, p):
    """(C,) any, (C,) dst of the first selected packet per holder."""
    return jnp.any(sel, axis=1), p.dst[jnp.argmax(sel, axis=1)]


def _per_unit(sel, p, n):
    """(C, N) count of selected packets per destination."""
    onehot = (p.dst[:, None] == jnp.arange(n)[None, :]).astype(jnp.float32)
    return (sel.astype(jnp.float32) @ onehot).astype(jnp.int32)


def _bounty_stacks(state):
    return jnp.minimum(jnp.sum(state.bounty, axis=1), BOUNTY_MAX).astype(jnp.float32)


def _proc(valid, ctx, dst, raw, dtype, perk, flags=TAG_PROC):
    return packets(valid, ctx.unit, jnp.maximum(dst, 0), raw, dtype, flags, item=rune_item(perk))


# ---- Electrocute ----------------------------------------------------------------

def _elec_damage(ctx, ev):
    ad = ea(ELECTROCUTE, "BonusADRatio") * ev.bonus_ad
    ap = ea(ELECTROCUTE, "APRatio") * ev.ap
    raw = lin(ea(ELECTROCUTE, "DamageBase"), ea(ELECTROCUTE, "DamageMax"), ctx.level) + ad + ap
    return raw, variable_damage_type(ad, ap)


def _elec_add(state: State, page, ctx, ev, new) -> State:
    """Add ``new`` (C, N) stacks, expire 3 s windows, queue the delayed hit."""
    now = ctx.now
    ready = has_rune(page, ELECTROCUTE) & (now >= state.elec_cd_until)
    new = jnp.where(ready[:, None], new, 0)
    expired = (state.elec_stacks > 0) & (now - state.elec_first_t > ea(ELECTROCUTE, "WindowDuration"))
    stacks = jnp.where(expired, 0, state.elec_stacks)
    first = jnp.where(expired, -BIG, state.elec_first_t)
    first = jnp.where((stacks == 0) & (new > 0), now, first)
    stacks = stacks + new
    full = stacks >= ELEC_STACKS
    fire = jnp.any(full, axis=1)
    tgt = jnp.argmax(full, axis=1).astype(jnp.int32)
    raw, dtype = _elec_damage(ctx, ev)
    return state._replace(
        elec_stacks=jnp.where(fire[:, None], 0, stacks).astype(jnp.int32),
        elec_first_t=_f32(jnp.where(fire[:, None], -BIG, first)),
        elec_cd_until=_f32(jnp.where(fire, now + ea(ELECTROCUTE, "Cooldown"), state.elec_cd_until)),
        elec_due=_f32(jnp.where(fire, now + ELEC_DELAY, state.elec_due)),
        elec_dst=jnp.where(fire, tgt, state.elec_dst),
        elec_raw=_f32(jnp.where(fire, raw, state.elec_raw)),
        elec_dtype=jnp.where(fire, dtype, state.elec_dtype).astype(jnp.int32))


def _elec_from_packets(state: State, page, ctx, units, ev, p) -> State:
    n = units.x.shape[0]
    sel = _to_enemy_champ(p, ctx, units) & _non_proc(p)[None, :]
    first = first_instance(p, sel)
    # Across ticks: drop instances whose (cast_id, dst) already stacked.
    seen = jnp.any((state.elec_seen_id[:, None, :] == p.cast_id[None, :, None])
                   & (state.elec_seen_dst[:, None, :] == p.dst[None, :, None]), axis=2) & (p.cast_id != 0)[None, :]
    inst = first & ~seen
    # A CC stack this tick pairs with the first damage instance on that champion.
    onehot = p.dst[:, None] == jnp.arange(n)[None, :]           # (P, N)
    cc_now = state.elec_cc_t == ctx.now                          # (C, N)
    cc_pkt = jnp.any(cc_now[:, None, :] & onehot[None, :, :], axis=2)   # (C, P)
    earlier = jnp.arange(p.valid.shape[0])[None, :] < jnp.arange(p.valid.shape[0])[:, None]  # [p, q]
    same_dst = (p.dst[:, None] == p.dst[None, :]) & earlier
    has_earlier = jnp.einsum("pq,cq->cp", same_dst.astype(jnp.float32), inst.astype(jnp.float32)) > 0.0
    paired = inst & cc_pkt & ~has_earlier
    counted = inst & ~paired
    new = _per_unit(counted, p, n)
    used_cc = cc_now & (_per_unit(inst, p, n) > 0)
    # Record new non-zero cast ids in the ring.
    rec = inst & (p.cast_id != 0)[None, :]
    order = jnp.cumsum(rec.astype(jnp.int32), axis=1) - 1
    ids, dsts, ptr = state.elec_seen_id, state.elec_seen_dst, state.elec_seen_ptr
    slots = jnp.arange(ELEC_SEEN)[None, :]
    for j in range(ELEC_SEEN):
        pick = rec & (order == j)
        any_j = jnp.any(pick, axis=1)
        cid = jnp.sum(jnp.where(pick, p.cast_id[None, :], 0), axis=1)
        dj = jnp.sum(jnp.where(pick, p.dst[None, :], 0), axis=1)
        w = (slots == ((ptr + j) % ELEC_SEEN)[:, None]) & any_j[:, None]
        ids = jnp.where(w, cid[:, None], ids)
        dsts = jnp.where(w, dj[:, None], dsts)
    n_rec = jnp.minimum(jnp.sum(rec, axis=1), ELEC_SEEN).astype(jnp.int32)
    state = state._replace(elec_seen_id=ids.astype(jnp.int32), elec_seen_dst=dsts.astype(jnp.int32),
                           elec_seen_ptr=((ptr + n_rec) % ELEC_SEEN).astype(jnp.int32),
                           elec_cc_t=_f32(jnp.where(used_cc, -BIG, state.elec_cc_t)))
    return _elec_add(state, page, ctx, ev, new)


def on_cc(state: State, page, ctx, units, ev):
    c, n = ctx.level.shape[0], units.x.shape[0]
    hit = (ev.cc.slowed | ev.cc.immobilized) & _enemy_champ_units(ctx, units) & units.alive[None, :]
    ready = (has_rune(page, ELECTROCUTE) & (ctx.now >= state.elec_cd_until))[:, None]
    state = state._replace(elec_cc_t=_f32(jnp.where(hit & ready, ctx.now, state.elec_cc_t)))
    return _elec_add(state, page, ctx, ev, hit.astype(jnp.int32)), effects(c, n)


# ---- action phase -----------------------------------------------------------------

def on_cast(state: State, page, ctx, units, ev):
    """Sudden Impact arming (RUNES §9 step 3); expiry starts the cooldown."""
    c, n = ctx.level.shape[0], units.x.shape[0]
    now = ctx.now
    armed = state.si_armed_until > -BIG / 2
    lapsed = armed & (now >= state.si_armed_until)
    cd = jnp.where(lapsed, state.si_armed_until + ea(SUDDEN_IMPACT, "Cooldown"), state.si_cd_until)
    until = jnp.where(lapsed, -BIG, state.si_armed_until)
    arm = has_rune(page, SUDDEN_IMPACT) & ev.blinked & ~(armed & ~lapsed) & (now >= cd)
    until = jnp.where(arm, now + ea(SUDDEN_IMPACT, "ArmedDuration"), until)
    return state._replace(si_armed_until=_f32(until), si_cd_until=_f32(cd)), effects(c, n)


def _hob_end(state: State, ended, at) -> State:
    return state._replace(
        hob_active=state.hob_active & ~ended, hob_stacks=jnp.where(ended, 0, state.hob_stacks),
        hob_bonus_used=jnp.where(ended, 0, state.hob_bonus_used),
        hob_cd_until=_f32(jnp.where(ended, at + ea(HAIL_OF_BLADES, "Cooldown"), state.hob_cd_until)))


def on_attack(state: State, page, ctx, units, ev):
    """Hail of Blades: windup trigger, cancel, launch consumption, resets."""
    c, n = ctx.level.shape[0], units.x.shape[0]
    now = ctx.now
    own = has_rune(page, HAIL_OF_BLADES)
    duration = ea(HAIL_OF_BLADES, "Duration")
    # Timeout: 3 s without an attack.
    state = _hob_end(state, state.hob_active & (now >= state.hob_expire), state.hob_expire)
    # Cancelled triggering windup: no stacks, brief lockout.
    cancel = state.hob_pending & ev.attack_cancelled
    state = state._replace(hob_pending=state.hob_pending & ~cancel,
                           hob_cd_until=_f32(jnp.where(cancel, now + HOB_CANCEL_LOCKOUT, state.hob_cd_until)))
    # Windup start on an enemy champion arms the trigger.
    tgt = jnp.clip(ev.attack_start_target, 0, n - 1)
    on_champ = (units.cls[tgt] == CLASS_CHAMPION) & (units.team[tgt] != ctx.team) & (ev.attack_start_target >= 0)
    start = own & ev.attack_started & on_champ & ~state.hob_active & (now >= state.hob_cd_until)
    pending = state.hob_pending | start
    # Launch: activation (triggering attack is the first empowered one) or consumption.
    launched = ev.attack.launched
    at = jnp.clip(ev.attack.target, 0, n - 1)
    at_champ = (units.cls[at] == CLASS_CHAMPION) & (units.team[at] != ctx.team)
    activate = launched & pending
    stacks = jnp.where(activate, int(ea(HAIL_OF_BLADES, "NumHits")), state.hob_stacks)
    active = state.hob_active | activate
    empowered = launched & active & (stacks > 0)
    stacks = jnp.where(empowered, stacks - 1, stacks)
    expire = jnp.where(activate | (empowered & at_champ), now + duration, state.hob_expire)
    state = state._replace(hob_pending=pending & ~activate, hob_active=active, hob_stacks=stacks.astype(jnp.int32),
                           hob_expire=_f32(expire), hob_inflight=state.hob_inflight | empowered,
                           hob_bonus_used=jnp.where(activate, 0, state.hob_bonus_used))
    # Trait_AttackReset: +1 stack, at most MaxBonusHits per activation.
    bonus = state.hob_active & ev.attack_reset & (state.hob_stacks > 0) \
        & (state.hob_bonus_used < int(ea(HAIL_OF_BLADES, "MaxBonusHits")))
    state = state._replace(hob_stacks=jnp.where(bonus, state.hob_stacks + 1, state.hob_stacks),
                           hob_bonus_used=jnp.where(bonus, state.hob_bonus_used + 1, state.hob_bonus_used))
    # Out of stacks: the effect ends now and the cooldown starts.
    state = _hob_end(state, state.hob_active & (state.hob_stacks <= 0), now)
    return state, effects(c, n)


def on_hit(state: State, page, ctx, units, ev):
    """Hail of Blades on-hit bonus true damage of an empowered attack."""
    c, n = ctx.level.shape[0], units.x.shape[0]
    land = state.hob_inflight & ev.attack.hit & has_rune(page, HAIL_OF_BLADES)
    raw = lin(ea(HAIL_OF_BLADES, "BonusDamageMin"), ea(HAIL_OF_BLADES, "BonusDamageMax"), ctx.level) \
        + ea(HAIL_OF_BLADES, "BonusADRatio") * ev.bonus_ad + ea(HAIL_OF_BLADES, "APRatio") * ev.ap
    p = _proc(land, ctx, ev.attack.target, raw, TRUE, HAIL_OF_BLADES, TAG_PROC | TAG_ON_HIT)
    return state._replace(hob_inflight=state.hob_inflight & ~ev.attack.hit), effects(c, n, packets=p)


def periodic(state: State, page, ctx, units, ev):
    """Electrocute delayed hit; Dark Harvest soul delivery."""
    c, n = ctx.level.shape[0], units.x.shape[0]
    now = ctx.now
    due = now >= state.elec_due
    dst = jnp.clip(state.elec_dst, 0, n - 1)
    p = _proc(due & units.alive[dst], ctx, state.elec_dst, state.elec_raw, state.elec_dtype, ELECTROCUTE)
    soul = now >= state.dh_soul_due
    state = state._replace(elec_due=_f32(jnp.where(due, BIG, state.elec_due)),
                           dh_souls=_f32(state.dh_souls + jnp.sum(soul, axis=1)),
                           dh_soul_due=_f32(jnp.where(soul, BIG, state.dh_soul_due)))
    return state, effects(c, n, packets=p)


# ---- damage triggers ----------------------------------------------------------------

def on_damage(state: State, page, ctx, units, ev):
    """RUNES §9 4.6 order: Electrocute, Dark Harvest, Cheap Shot, Sudden Impact, Taste of Blood."""
    c, n = ctx.level.shape[0], units.x.shape[0]
    now = ctx.now
    rep = ev.report
    p, r = rep.packets, rep.resolved
    champ = _to_enemy_champ(p, ctx, units)
    non_proc = _non_proc(p)[None, :]
    d = jnp.clip(p.dst, 0, n - 1)

    state = _elec_from_packets(state, page, ctx, units, ev, p)

    # Dark Harvest (post-hit HP, U-07).
    below = (units.hp[d] < ea(DARK_HARVEST, "HarvestThreshold") * units.max_hp[d]) & units.alive[d]
    dh_sel = champ & non_proc & (r.final >= DH_MIN_DAMAGE)[None, :] & below[None, :]
    dh_any, dh_dst = _first_dst(dh_sel, p)
    dh_go = dh_any & has_rune(page, DARK_HARVEST) & (now >= state.dh_cd_until)
    dh_raw = ea(DARK_HARVEST, "BaseDamage") + ea(DARK_HARVEST, "DamagePerSoulEssence") * state.dh_souls \
        + ea(DARK_HARVEST, "ADRatio") * ev.bonus_ad + ea(DARK_HARVEST, "APRatio") * ev.ap
    p_dh = _proc(dh_go, ctx, dh_dst, dh_raw, adaptive_damage_type(ev), DARK_HARVEST)
    free = state.dh_soul_due >= BIG / 2
    slot = jnp.argmax(free, axis=1)
    put = (jnp.arange(DH_SOUL_SLOTS)[None, :] == slot[:, None]) & dh_go[:, None]
    state = state._replace(dh_cd_until=_f32(jnp.where(dh_go, now + ea(DARK_HARVEST, "Cooldown"), state.dh_cd_until)),
                           dh_soul_due=_f32(jnp.where(put, now + DH_SOUL_DELAY, state.dh_soul_due)))

    # Cheap Shot: target impaired at tick start, or impaired on-hit by the same
    # cast instance this tick (RUNES §4.4 exception; ``ev.cc_on_hit`` / ``cc_cast_id``).
    nn = units.x.shape[0]
    dcl = jnp.clip(d, 0, nn - 1)
    same_hit = ev.cc_on_hit[:, dcl] & (ev.cc.slowed | ev.cc.immobilized)[:, dcl] \
        & (ev.cc_cast_id[:, dcl] == p.cast_id[None, :]) & (p.cast_id[None, :] != 0)
    cs_sel = champ & non_proc & (ev.impaired[d][None, :] | same_hit)
    cs_any, cs_dst = _first_dst(cs_sel, p)
    cs_go = cs_any & has_rune(page, CHEAP_SHOT) & (now >= state.cs_cd_until)
    p_cs = _proc(cs_go, ctx, cs_dst, lin(ea(CHEAP_SHOT, "DamageIncMin"), ea(CHEAP_SHOT, "DamageIncMax"), ctx.level),
                 TRUE, CHEAP_SHOT)
    state = state._replace(cs_cd_until=_f32(jnp.where(cs_go, now + ea(CHEAP_SHOT, "Cooldown"), state.cs_cd_until)))

    # Sudden Impact: first damage while armed.
    si_sel = champ & (p.item != rune_item(SUDDEN_IMPACT))[None, :]
    si_any, si_dst = _first_dst(si_sel, p)
    armed = (state.si_armed_until > -BIG / 2) & (now < state.si_armed_until)
    si_go = si_any & armed & has_rune(page, SUDDEN_IMPACT)
    p_si = _proc(si_go, ctx, si_dst, lin(ea(SUDDEN_IMPACT, "MinDamageTooltip"), ea(SUDDEN_IMPACT, "MaxDamageTooltip"),
                                         ctx.level), TRUE, SUDDEN_IMPACT)
    state = state._replace(si_armed_until=_f32(jnp.where(si_go, -BIG, state.si_armed_until)),
                           si_cd_until=_f32(jnp.where(si_go, now + ea(SUDDEN_IMPACT, "Cooldown"), state.si_cd_until)))

    # Taste of Blood: any damage, not at full HP.
    tob_go = jnp.any(champ, axis=1) & has_rune(page, TASTE_OF_BLOOD) & (now >= state.tob_cd_until) \
        & ctx.alive & (ctx.hp < ctx.max_hp)
    heal = lin(ea(TASTE_OF_BLOOD, "HealAmount"), ea(TASTE_OF_BLOOD, "HealAmountMax"), ctx.level) \
        + ea(TASTE_OF_BLOOD, "ADRatio") * ev.bonus_ad + ea(TASTE_OF_BLOOD, "APRatio") * ev.ap
    state = state._replace(tob_cd_until=_f32(jnp.where(tob_go, now + ea(TASTE_OF_BLOOD, "Cooldown"),
                                                       state.tob_cd_until)))
    return state, effects(c, n, packets=concat_packets(p_dh, p_cs, p_si), heal=jnp.where(tob_go, heal, 0.0))


# ---- takedowns and stats ----------------------------------------------------------

def on_takedown(state: State, page, ctx, units, ev):
    c, n = ctx.level.shape[0], units.x.shape[0]
    now = ctx.now
    took = ev.kills.killed_units & _enemy_champ_units(ctx, units)
    n_took = jnp.sum(took, axis=1).astype(jnp.float32)
    # Dark Harvest: takedown resets the remaining cooldown to 1 s; execute credit souls while ready.
    dh = has_rune(page, DARK_HARVEST)
    ready = now >= state.dh_cd_until
    extra = jnp.where(dh & ready, ev.execute_credit, 0.0)
    reset = dh & (n_took > 0)
    dh_cd = jnp.where(reset, jnp.minimum(state.dh_cd_until, now + ea(DARK_HARVEST, "CooldownResetValue")),
                      state.dh_cd_until)
    # Bounty Hunter stacks (unique enemy champions) and Treasure Hunter gold.
    before = _bounty_stacks(state)
    bounty = state.bounty | took
    after = jnp.minimum(jnp.sum(bounty, axis=1), BOUNTY_MAX).astype(jnp.float32)
    k = after - before
    gold = ea(TREASURE_HUNTER, "BaseGoldAmount") * k + ea(TREASURE_HUNTER, "GoldGrowth") * (before * k + k * (k - 1) / 2)
    gold = jnp.where(has_rune(page, TREASURE_HUNTER), gold, 0.0)
    memento = jnp.minimum(state.mementos + n_took, ea(GRISLY_MEMENTOS, "MaxStacks"))
    state = state._replace(dh_souls=_f32(state.dh_souls + extra), dh_cd_until=_f32(dh_cd), bounty=bounty,
                           mementos=_f32(jnp.where(has_rune(page, GRISLY_MEMENTOS), memento, state.mementos)))
    return state, effects(c, n, gold=gold)


def stats(state: State, page, ctx, ev) -> ItemStats:
    now = ctx.now
    hob = has_rune(page, HAIL_OF_BLADES) & state.hob_active & (state.hob_stacks > 0) & (now < state.hob_expire)
    hob_as = by_range(ctx, ea(HAIL_OF_BLADES, "ASBoost"), ea(HAIL_OF_BLADES, "ASBoostRanged"))
    stacks = _bounty_stacks(state)
    last = ev.clocks.last_combat if ev.clocks.last_combat_modern is None else ev.clocks.last_combat_modern
    ooc = now - last >= COMBAT_TIMEOUT
    ms = jnp.where(has_rune(page, RELENTLESS_HUNTER) & ooc,
                   ea(RELENTLESS_HUNTER, "StartingOOCMS") + ea(RELENTLESS_HUNTER, "OOCMS") * stacks, 0.0)
    uh = jnp.where(has_rune(page, ULTIMATE_HUNTER),
                   ea(ULTIMATE_HUNTER, "StartingUltAH") + ea(ULTIMATE_HUNTER, "AdditionalUltAH") * stacks, 0.0)
    trinket = jnp.where(has_rune(page, GRISLY_MEMENTOS), ea(GRISLY_MEMENTOS, "TrinketAH") * state.mementos, 0.0)
    return ItemStats(attack_speed=_f32(jnp.where(hob, hob_as, 0.0)),
                     attack_speed_cap_lift=_f32(jnp.where(hob, 1.0, 0.0)),
                     move_speed=_f32(ms), ultimate_haste=_f32(uh), trinket_haste=_f32(trinket))
