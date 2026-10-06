"""Jax (24), 26.19. Rules JAX.* and their evidence: docs/modern/CHAMPIONS.md; numbers from the pinned client
JSON (``modern/data/26.19/champions``)."""
from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from ..core import damage as D
from ..core.types import KIND_CHAMPION, KIND_MONSTER, KIND_NONE, KIND_WARD, CCOut, Dash
from ..items.catalog import ItemStats
from ..items.effects.core import Debuffs, neutral_debuffs
from ..runes.catalog import ChampionTraits
from .core import (CODE_JAX_R_PASSIVE, NEVER, KitAttackMods, KitDefense, KitOut, cc_matrix, cooldown_row,
                   due, enemies, f32, gather, holder_rows, is_structure, later, later_after_tick, make_cast_id,
                   mana_row, out, ranked, scalar, target_dist, within_edge)

NAME = "Jax"
ID = 24
SKILL_ORDER = (2, 0, 1, 1, 1, 3, 1, 2, 1, 2, 3, 2, 2, 0, 0, 3, 0, 0)    # default ranks per level: W > E > Q
TRAITS = ChampionTraits(has_immobilize=True, resource="mana")          # rune legality: E stuns
P_DURATION = scalar(NAME, "Passive", "BuffDuration")     # 2.5
P_MAX = int(scalar(NAME, "Passive", "MaxStacks"))         # 8
P_FALLOFF = scalar(NAME, "Passive", "FallOffRate")        # 0.35
Q_RANGE = 700.0
UNIT_TARGET_RANGE = (Q_RANGE, 0.0, 0.0, 0.0)   # per slot; 0 = not unit-targeted (walk-in casting)
Q_SPEED = 1400.0
W_DURATION = 10.0
W_AP = 0.6
W_STRUCTURE = scalar(NAME, "W", "StructureMod")           # 0.5
RANGE_BONUS = 50.0
E_DURATION = scalar(NAME, "E", "DodgeDuration")           # 2
E_MIN = 1.0
E_RADIUS = 375.0
E_AP = scalar(NAME, "E", "APRatio")                       # 0.7
E_PCT_HP = scalar(NAME, "E", "PercentHealthDamage") / 100.0
E_PER_DODGE = scalar(NAME, "E", "PercentIncreasedPerDodge")
E_MAX_DODGES = scalar(NAME, "E", "MaxDodgesForDamageIncrease")
E_STUN = scalar(NAME, "E", "StunDuration")
E_AOE_MULT = 1.0 - scalar(NAME, "E", "AoEDamageReduction") / 100.0
E_MONSTER_CAP = 9000.0                                    # MonsterDamageCap on the %HP part
R_DELAY = 0.25
R_RADIUS = scalar(NAME, "R", "AoESize")                   # 375
R_AP = 1.0
R_DURATION = scalar(NAME, "R", "Duration")                # 8
R_MR_MULT = scalar(NAME, "R", "MRMult")                   # 0.6
R_PASSIVE_AP = 0.6
R_FALLOFF = scalar(NAME, "R", "PassiveFallOffTime")       # 2.5
R_STRUCTURE = scalar(NAME, "R", "StructureMod")
R_PASSIVE_STACKS = 2                                      # stacks needed (1 under the active)
P_LEVEL_CAP = 19.0                                        # last AttackSpeedPerStack breakpoint

SPELL = D.TAG_ACTIVE_SPELL
AOE_SPELL = D.TAG_AOE | D.TAG_ACTIVE_SPELL
R_FLAGS = D.TAG_AOE | D.TAG_ACTIVE_SPELL | D.PROP_ULTIMATE
R_PASSIVE_FLAGS = D.TAG_ON_HIT | D.TAG_PROC | D.PROP_ULTIMATE


class State(NamedTuple):
    q_pending: Any          # (C,) leaping
    q_land_at: Any
    q_target: Any
    q_seq: Any
    q_cast_id: Any
    w_on: Any
    w_until: Any
    w_cast_id: Any
    e_on: Any
    e_start: Any
    e_until: Any
    e_dodges: Any           # int32
    e_cast_id: Any
    r_pending: Any
    r_fire_at: Any
    r_cast_id: Any
    r_buff_on: Any
    r_buff_until: Any
    r_armor: Any
    r_mr: Any
    r_hits: Any             # int32 landed attacks toward the R passive
    r_hit_until: Any
    stacks: Any             # int32 passive stacks
    stack_until: Any
    reset_at: Any


def init(n_champions: int, n_units: int) -> State:
    c = n_champions
    z = jnp.zeros((c,), jnp.float32)
    f = jnp.zeros((c,), bool)
    i = jnp.zeros((c,), jnp.int32)
    return State(f, z + NEVER, i - 1, i, i, f, z, i, f, z, z, i, i, f, z + NEVER, i, f, z, z, z, i, z, i, z,
                 z - NEVER)


def _mine(kctx) -> Any:
    return kctx.champion_id == ID


def _base_cd(kctx) -> Any:
    return jnp.where(_mine(kctx)[:, None], cooldown_row(NAME, kctx.ranks), 0.0)


def dodging(state: State, kctx) -> Any:
    """(C,) holder dodges basic attacks (Counter Strike)."""
    return _mine(kctx) & state.e_on & kctx.alive


def _r_needed(state: State) -> Any:
    return jnp.where(state.r_buff_on, R_PASSIVE_STACKS - 1, R_PASSIVE_STACKS)


def _w_damage(kctx) -> Any:
    return ranked(NAME, "W", "Damage", kctx.ranks[:, 1]) + W_AP * kctx.ap


def _release(state: State, kctx, units, mask) -> tuple[State, D.Packets, Any, Any]:
    """Counter Strike release for holders in ``mask``: packets, stun (C, N), cast ids."""
    n = units.x.shape[0]
    hit = mask[:, None] & enemies(kctx, units) & within_edge(kctx, units, E_RADIUS)
    base = ranked(NAME, "E", "BaseDamage", kctx.ranks[:, 2]) + E_AP * kctx.ap
    pct = E_PCT_HP * units.max_hp
    pct = jnp.where(units.kind == KIND_MONSTER, jnp.minimum(pct, E_MONSTER_CAP), pct)
    raw = (base[:, None] + pct[None, :]) \
        * (1.0 + E_PER_DODGE * jnp.minimum(state.e_dodges, E_MAX_DODGES))[:, None]
    p = D.packets(hit, kctx.unit[:, None], jnp.arange(n)[None, :], raw, D.MAGIC, AOE_SPELL,
                  cast_id=state.e_cast_id[:, None])
    stun, sid = cc_matrix(hit, E_STUN, state.e_cast_id[:, None])
    return state._replace(e_on=state.e_on & ~mask), p, stun, sid


def cast(state: State, kctx, units, order) -> tuple[State, KitOut]:
    c, n = kctx.unit.shape[0], units.x.shape[0]
    j = _mine(kctx)
    r = kctx.ranks
    free = j & kctx.alive & ~kctx.stunned & ~kctx.silenced & ~state.r_pending
    cost = mana_row(NAME, r)
    ready = (r > 0) & (kctx.cooldowns <= 0) & (kctx.mana[:, None] >= cost)
    want = lambda s: (order.slot == s) & free          # noqa: E731
    t = jnp.clip(order.target, 0, n - 1)
    dist = target_dist(kctx, units, order.target)
    valid_q = (order.target >= 0) & (order.target != kctx.unit) & units.alive[t] & units.targetable[t] \
        & (units.kind[t] != KIND_NONE) & ~is_structure(units.kind[t]) & (dist <= Q_RANGE + units.radius[t])
    rooted = jnp.zeros_like(j) if kctx.rooted is None else kctx.rooted
    q = want(0) & ready[:, 0] & valid_q & ~state.q_pending & ~rooted
    w = want(1) & ready[:, 1] & ~state.w_on
    e_start = want(2) & ready[:, 2] & ~state.e_on
    e_release = want(2) & state.e_on & (kctx.now - state.e_start >= E_MIN - 1e-6)
    rr = want(3) & ready[:, 3]
    started = q | w | e_start | rr
    slot = jnp.where(q, 0, jnp.where(w, 1, jnp.where(e_start, 2, jnp.where(rr, 3, -1)))).astype(jnp.int32)
    cid = jnp.where(started, make_cast_id(kctx, jnp.maximum(slot, 0)), 0).astype(jnp.int32)
    now = f32(kctx.now)

    state, p_e, stun, sid = _release(state, kctx, units, e_release)
    state = state._replace(
        q_pending=state.q_pending | q,
        q_land_at=jnp.where(q, f32(now + jnp.maximum(dist / Q_SPEED, 1e-3)), state.q_land_at),
        q_target=jnp.where(q, order.target, state.q_target).astype(jnp.int32),
        q_seq=jnp.where(q, units.spawn_seq[t], state.q_seq).astype(jnp.int32),
        q_cast_id=jnp.where(q, cid, state.q_cast_id),
        w_on=state.w_on | w, w_until=jnp.where(w, later(kctx, W_DURATION), state.w_until),
        w_cast_id=jnp.where(w, cid, state.w_cast_id),
        e_on=state.e_on | e_start, e_start=jnp.where(e_start, now, state.e_start),
        e_until=jnp.where(e_start, later(kctx, E_DURATION), state.e_until),
        e_dodges=jnp.where(e_start, 0, state.e_dodges), e_cast_id=jnp.where(e_start, cid, state.e_cast_id),
        r_pending=state.r_pending | rr, r_fire_at=jnp.where(rr, later(kctx, R_DELAY), state.r_fire_at),
        r_cast_id=jnp.where(rr, cid, state.r_cast_id),
        reset_at=jnp.where(w, now, state.reset_at))
    mana = jnp.sum(jnp.where(jnp.stack([q, w, e_start, rr], -1), cost, 0.0), axis=1)
    z = jnp.zeros((c, n), jnp.float32)
    dash = Dash(q, f32(units.x[t]), f32(units.y[t]), jnp.full((c,), Q_SPEED, jnp.float32),
                jnp.where(q, order.target, -1).astype(jnp.int32), jnp.zeros((c,), bool))
    return state, out(c, n, packets=p_e, cc=CCOut(stun, z, z, z, z, z, sid), dash=dash,
                      mana_cost=f32(mana), cooldown_start=jnp.stack([q, jnp.zeros_like(q), e_release, rr], -1),
                      base_cooldown=_base_cd(kctx), attack_reset=w, cast_started=started, cast_slot=slot,
                      cast_id=cid)


def periodic(state: State, kctx, units) -> tuple[State, KitOut]:
    c, n = kctx.unit.shape[0], units.x.shape[0]
    j = _mine(kctx)
    alive = kctx.alive
    dead = ~alive
    r = kctx.ranks

    # Leap landing (target identity guards against a recycled slot).
    t = jnp.clip(state.q_target, 0, n - 1)
    target_ok = units.alive[t] & (units.spawn_seq[t] == state.q_seq)
    land = state.q_pending & due(kctx, state.q_land_at) & alive & target_ok
    strike = land & (units.team[t] != kctx.team) & (units.kind[t] != KIND_WARD)
    q_raw = ranked(NAME, "Q", "Damage", r[:, 0]) + kctx.bonus_ad
    p_q = D.packets(strike, kctx.unit, t, q_raw, D.PHYSICAL, SPELL, cast_id=state.q_cast_id)
    w_on_q = strike & state.w_on
    p_wq = D.packets(w_on_q, kctx.unit, t, _w_damage(kctx), D.MAGIC, SPELL, cast_id=state.w_cast_id)
    q_pending = state.q_pending & ~(land | dead | ~target_ok)

    # Empower: expiry, consumption by Q, death.
    w_end = state.w_on & (due(kctx, state.w_until) | w_on_q | dead)

    # Counter Strike: expiry releases (alive); death ends it silently.
    e_expire = state.e_on & due(kctx, state.e_until) & alive
    state, p_e, stun, sid = _release(state, kctx, units, e_expire)
    e_dead = state.e_on & dead
    e_end = e_expire | e_dead

    # Grandmaster's Might swing.
    r_due = state.r_pending & (due(kctx, state.r_fire_at) | dead)
    r_fire = r_due & alive
    r_hit = r_fire[:, None] & enemies(kctx, units) & within_edge(kctx, units, R_RADIUS)
    r_raw = ranked(NAME, "R", "SwingDamageBase", r[:, 3]) + R_AP * kctx.ap
    p_r = D.packets(r_hit, kctx.unit[:, None], jnp.arange(n)[None, :], r_raw[:, None], D.MAGIC, R_FLAGS,
                    cast_id=state.r_cast_id[:, None])
    champs = jnp.sum(r_hit & (units.kind[None, :] == KIND_CHAMPION), axis=1)
    armor = ranked(NAME, "R", "BaseResists", r[:, 3]) + 0.4 * kctx.bonus_ad \
        + jnp.maximum(champs - 1, 0) * (ranked(NAME, "R", "ResistsPerExtraTarget", r[:, 3]) + 0.1 * kctx.bonus_ad)
    gain = r_fire & (champs > 0)
    r_buff_on = (state.r_buff_on & ~(due(kctx, state.r_buff_until) | dead)) | gain

    # Passive stacks fall off one at a time once the buff lapses.
    expired = (state.stacks > 0) & due(kctx, state.stack_until)
    stacks = jnp.where(dead, 0, jnp.maximum(0, state.stacks - expired.astype(jnp.int32)))
    stack_until = jnp.where(expired, later_after_tick(kctx, P_FALLOFF), state.stack_until)
    r_hits = jnp.where(due(kctx, state.r_hit_until) | dead, 0, state.r_hits)

    state = state._replace(
        q_pending=q_pending, w_on=state.w_on & ~w_end, e_on=state.e_on & ~e_dead,
        r_pending=state.r_pending & ~r_due, r_buff_on=r_buff_on,
        r_buff_until=jnp.where(gain, later_after_tick(kctx, R_DURATION), state.r_buff_until),
        r_armor=jnp.where(gain, f32(armor), state.r_armor), r_mr=jnp.where(gain, f32(R_MR_MULT * armor), state.r_mr),
        r_hits=r_hits.astype(jnp.int32), stacks=stacks.astype(jnp.int32), stack_until=stack_until)
    z = jnp.zeros((c, n), jnp.float32)
    cd_start = jnp.zeros((c, 4), bool).at[:, 1].set(w_end & j).at[:, 2].set(e_end & j)
    follow = jnp.where(strike & (units.kind[t] == KIND_CHAMPION), state.q_target, -1).astype(jnp.int32)
    return state, out(c, n, packets=D.concat_packets(p_q, p_wq, p_e, p_r), cc=CCOut(stun, z, z, z, z, z, sid),
                      cooldown_start=cd_start, base_cooldown=_base_cd(kctx), attack_target=follow)


def on_attack(state: State, kctx, units, launch) -> tuple[State, KitOut]:
    go = holder_rows(launch.launched, kctx) & _mine(kctx) & kctx.alive
    state = state._replace(stacks=jnp.where(go, jnp.minimum(P_MAX, state.stacks + 1), state.stacks).astype(jnp.int32),
                           stack_until=jnp.where(go, later_after_tick(kctx, P_DURATION), state.stack_until))
    return state, out(kctx.unit.shape[0], units.x.shape[0], base_cooldown=_base_cd(kctx))


def on_hit(state: State, kctx, units, launch, dodging_units) -> tuple[State, KitOut]:
    """Empower and the R passive on a landed attack. ``dodging_units`` (N,) bool."""
    c, n = kctx.unit.shape[0], units.x.shape[0]
    j = _mine(kctx)
    target = holder_rows(launch.target, kctx)
    hit = holder_rows(launch.launched, kctx) & j & kctx.alive & (target >= 0)
    t = jnp.clip(target, 0, n - 1)
    landed = hit & ~gather(dodging_units, target)
    struct = jnp.where(is_structure(units.kind[t]), W_STRUCTURE, 1.0)
    w = landed & state.w_on
    p_w = D.packets(w, kctx.unit, t, _w_damage(kctx) * struct, D.MAGIC, SPELL, cast_id=state.w_cast_id)
    has_r = kctx.ranks[:, 3] > 0
    ward = units.kind[t] == KIND_WARD
    ready = has_r & (state.r_hits >= _r_needed(state))
    proc = landed & ready & ~ward          # vs wards: triggers, but neither consumed nor applied
    r_raw = (ranked(NAME, "R", "PassiveBaseDamage", kctx.ranks[:, 3]) + R_PASSIVE_AP * kctx.ap) \
        * jnp.where(is_structure(units.kind[t]), R_STRUCTURE, 1.0)
    p_r = D.packets(proc, kctx.unit, t, r_raw, D.MAGIC, R_PASSIVE_FLAGS,
                    cast_id=make_cast_id(kctx, CODE_JAX_R_PASSIVE))
    state = state._replace(
        w_on=state.w_on & ~w,
        r_hits=jnp.where(landed & has_r, jnp.where(proc, 0, jnp.minimum(state.r_hits + 1, R_PASSIVE_STACKS)),
                         state.r_hits).astype(jnp.int32),
        r_hit_until=jnp.where(landed, later_after_tick(kctx, R_FALLOFF), state.r_hit_until))
    cd_start = jnp.zeros((c, 4), bool).at[:, 1].set(w)
    return state, out(c, n, packets=D.concat_packets(p_w, p_r), cooldown_start=cd_start,
                      base_cooldown=_base_cd(kctx))


def on_damage(state: State, kctx, units, report) -> tuple[State, KitOut]:
    """Count dodged attack instances (one per basic-attack cast id) during Counter Strike."""
    p = report.packets
    src_kind = gather(units.kind, p.src)
    basic = p.valid & D.has(p.flags, D.TAG_BASIC_ATTACK) & ~D.has(p.flags, D.TAG_ON_HIT) & ~is_structure(src_kind)
    cid = p.cast_id
    P = cid.shape[0]
    earlier = jnp.arange(P)[None, :] < jnp.arange(P)[:, None]
    same = (cid[:, None] == cid[None, :]) & (cid[:, None] != 0) & (p.dst[:, None] == p.dst[None, :]) & earlier
    first = basic & ~jnp.any(same & basic[None, :], axis=1)
    on_me = (p.dst[None, :] == kctx.unit[:, None]) & first[None, :]
    count = jnp.sum(on_me, axis=1).astype(jnp.int32)
    state = state._replace(e_dodges=jnp.where(dodging(state, kctx), state.e_dodges + count, state.e_dodges))
    return state, out(kctx.unit.shape[0], units.x.shape[0])


def on_takedown(state: State, kctx, units, kills) -> State:
    return state


def stats(state: State, kctx) -> ItemStats:
    j = _mine(kctx)
    lv = jnp.minimum(jnp.asarray(kctx.level, jnp.float32), P_LEVEL_CAP)
    per_stack = 0.05 + 0.015 * jnp.floor((lv - 1.0) / 3.0)
    buff = j & state.r_buff_on
    return ItemStats(attack_speed=f32(jnp.where(j, per_stack * state.stacks, 0.0)),
                     armor=f32(jnp.where(buff, state.r_armor, 0.0)), magic_resist=f32(jnp.where(buff, state.r_mr, 0.0)))


def defense(state: State, kctx) -> KitDefense:
    c = kctx.unit.shape[0]
    on = dodging(state, kctx)
    return KitDefense(jnp.ones((c,), jnp.float32), on, f32(jnp.where(on, E_AOE_MULT, 1.0)),
                      jnp.zeros((c,), jnp.float32))


def attack_mods(state: State, kctx) -> KitAttackMods:
    j = _mine(kctx)
    c = kctx.unit.shape[0]
    z = jnp.zeros((c,), jnp.float32)
    empowered = j & (state.w_on | ((kctx.ranks[:, 3] > 0) & (state.r_hits >= _r_needed(state))))
    return KitAttackMods(f32(jnp.where(j & state.w_on, RANGE_BONUS, 0.0)),
                         j & (jnp.abs(kctx.now - state.reset_at) < 0.5 * kctx.dt),
                         j & (state.q_pending | state.r_pending), jnp.zeros((c,), bool), z, z, empowered)


def debuffs(state: State, kctx, units) -> Debuffs:
    return neutral_debuffs(units.x.shape[0])
