"""Garen (86) 26.19 kit on the modern event/packet contract.

Numbers come from the pinned 16.19 client JSON (``data/modern_26_19``); rules
from the client spell records/calculations, the wiki ability data and the
26.1-26.19 patch notes. Spec table with evidence levels: docs/modern/CHAMPIONS.md.

* Perseverance: ``RegenCalc`` (1.5% + 0.2%/lv to 6, +0.8%/lv 7-13, +0.4%/lv
  14+) of max HP per 5 s, paid as 1/10 of it on every 0.5 s grid boundary.
  Disabled for ``DamageTimer`` (8 s, not hasted) after Garen loses health to an
  enemy champion (attacks, abilities, summoner damage) or turret; minion and
  monster damage, shield-absorbed and zero damage do not count. Tracked by the
  kit from ``on_damage`` reports.
* Q: removes slows, +35% MS for ``MovementSpeedDuration``; next attack within
  4.5 s deals ``BaseDamage + 0.5 * AD`` bonus physical (``tADRatio`` 1.5 total,
  the world's basic attack supplies 1.0 AD) and silences 1.5 s. The attack
  itself rolls crit as normal (``GarenQAttack.mRollForCriticalHit``); the bonus
  never crits. Attack reset; uncancellable windup; overridden attack time
  ``T = 1.7 - 0.2 * bonus AS`` with windup ``0.2 T`` (``mOverrideAttackTime``),
  so Garen cannot attack again until ``0.8 T`` after the Q hit. Lunge: +50
  range against champions. Cooldown starts post-effect (hit, dodge, expiry or
  death), not at cast; no recast while the window is open.
* W: passive +0.2 armor/MR per stack, 150 stacks (30). One stack per enemy
  champion killing blow (no assists), minion last hit or monster kill; wards
  and structures give none; only once W is learned. Active: shield
  ``BaseShield + 0.18 * bonus HP`` and 60% tenacity for 0.75 s, ``DRPercent``
  damage reduction for 4 s. Cooldown at cast.
* E: 3 s spin, ``7 + floor(bonus AS / 0.25)`` spins fixed at cast; spin ``k``
  completes (and hits) at ``k * 3 / n`` after the cast. Per spin
  ``BaseDamagePerTick + ADRatioPerTick * AD`` (evaluated live) to enemies within
  325 (center-to-edge), +25% on the nearest; each spin rolls crit for
  ``1 + 0.3 * (crit damage - 1)`` (``CriticalDamage`` calc). Enemy champions hit
  6 times lose 25% armor for 6 s, refreshed on the 7th hit and every 6th after;
  the hit count carries across casts while the shred is active. Recast after 1 s
  or R ends it; cooldown starts at the end. No attacks, ghosted. Each spin is
  its own cast instance (Conqueror stacks per spin).
* R: enemy champion within 400 (center-to-edge), 0.435 s cast, then
  ``BaseDamage + ExecuteDamage * missing HP`` true damage if the target is
  still the same living unit (no Villain since its removal). Cooldown at cast.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from .. import modern_damage as D
from ..modern_item_data import ItemStats
from ..modern_item_effects.core import Debuffs, neutral_debuffs, shield_grants
from ..modern_world_types import KIND_CHAMPION, KIND_MONSTER, KIND_TURRET, CCOut
from .core import (CODE_GAREN_E_TICK, GAREN, NEVER, KitAttackMods, KitDefense, KitOut, cc_matrix, center_dist,
                   cooldown_row, due, enemies, f32, gather, holder_rows, later, later_after_tick, make_cast_id,
                   out, pulses, ranked, scalar, target_dist, tick_index)

NAME = "Garen"
Q_WINDOW = scalar(NAME, "Q", "AttackWindow")          # 4.5
Q_MS = scalar(NAME, "Q", "MovementSpeedAmount")       # 0.35
Q_AD_RATIO = scalar(NAME, "Q", "tADRatio")            # 1.5 total AD (1.0 from the basic attack)
Q_RANGE_BONUS = 50.0                                   # lunge: 50 units past attack range vs champions (wiki)
Q_ATTACK_TIME = 1.7                                    # GarenQAttack mOverrideAttackTime: 1.7 - 0.2 * bonus AS
Q_ATTACK_TIME_AS = 0.2
Q_WINDUP_FRACTION = 0.2                                # mCastTimePercent
W_SHIELD_RATIO = scalar(NAME, "W", "ShieldHealthRatio")
W_UPFRONT = scalar(NAME, "W", "UpfrontDuration")      # 0.75 s shield + tenacity
W_TENACITY = scalar(NAME, "W", "UpfrontTenacity")     # 0.6
W_DR_DURATION = scalar(NAME, "W", "DRDuration")       # 4
W_RESIST_PER_STACK = scalar(NAME, "W", "ResistGainOnKill")   # 0.2
W_MAX_STACKS = round(scalar(NAME, "W", "ResistMax") / W_RESIST_PER_STACK)   # 150
E_DURATION = scalar(NAME, "E", "Duration")            # 3
E_RADIUS = 325.0                                       # castRangeDisplayOverride / wiki effect radius
E_MIN_SPIN = 1.0                                       # recast allowed after 1 s
E_NEAREST = scalar(NAME, "E", "NearestEnemyBonus")    # 0.25
E_CRIT_MOD = scalar(NAME, "E", "CritMod")             # 0.3 of the bonus crit damage
E_SHRED = scalar(NAME, "E", "ShredAmount")            # 0.25
E_SHRED_DURATION = scalar(NAME, "E", "ShredDuration")  # 6
E_SHRED_HITS = int(scalar(NAME, "E", "StacksToShred"))  # 6
R_RANGE = 400.0
UNIT_TARGET_RANGE = (0.0, 0.0, 0.0, R_RANGE)   # per slot; 0 = not unit-targeted (walk-in casting)
R_CAST_TIME = 0.435                                    # spellCastTime
PASSIVE_DELAY = scalar(NAME, "Passive", "DamageTimer")   # 8 s
PASSIVE_PULSE = 0.5                                    # wiki: (RegenCalc / 10) every 0.5 s

E_FLAGS = D.TAG_AOE | D.TAG_ACTIVE_SPELL | D.TAG_PERIODIC
Q_FLAGS = D.TAG_ACTIVE_SPELL
R_FLAGS = D.TAG_ACTIVE_SPELL | D.PROP_ULTIMATE


class State(NamedTuple):
    q_on: Any               # (C,) empowered attack window open
    q_until: Any
    q_cast_id: Any
    q_haste_on: Any
    q_haste_until: Any
    w_on: Any               # damage reduction window
    w_until: Any
    w_dr: Any               # DRPercent snapshot at cast
    w_ten_on: Any           # upfront tenacity window
    w_ten_until: Any
    w_stacks: Any           # permanent resist stacks (count)
    e_on: Any
    e_start: Any
    e_ticks: Any            # int32 spins this cast
    e_done: Any             # int32 spins completed
    e_hits: Any             # (C, N) int32 hits per unit (carried while shredded)
    shred_until: Any        # (C, N) armor shred end
    r_pending: Any
    r_fire_at: Any
    r_target: Any
    r_seq: Any
    r_cast_id: Any
    reset_at: Any           # time of the last attack-reset request
    q_lock_until: Any       # no new attack before this (Q attack's overridden attack time)
    p_block_until: Any      # Perseverance disabled until
    key: Any                # E crit RNG (folded with the tick index)


def init(n_champions: int, n_units: int) -> State:
    c, n = n_champions, n_units
    z = jnp.zeros((c,), jnp.float32)
    f = jnp.zeros((c,), bool)
    i = jnp.zeros((c,), jnp.int32)
    return State(f, z, i, f, z, f, z, z, f, z, z, f, z, i, i,
                 jnp.zeros((c, n), jnp.int32), jnp.zeros((c, n), jnp.float32),
                 f, z + NEVER, i - 1, i, i, z - NEVER, z - NEVER, z - NEVER, jax.random.PRNGKey(86))


def _mine(kctx) -> Any:
    return kctx.champion_id == GAREN


def _base_cd(kctx) -> Any:
    return jnp.where(_mine(kctx)[:, None], cooldown_row(NAME, kctx.ranks), 0.0)


def regen_rate(level: Any) -> Any:
    """RegenCalc percent of max HP per 5 s (1.5 + 0.2/lv to 6, +0.8/lv 7-13, +0.4/lv 14+)."""
    lv = jnp.asarray(level, jnp.float32)
    return 1.5 + 0.2 * jnp.clip(lv - 1, 0, 5) + 0.8 * jnp.clip(lv - 6, 0, 7) + 0.4 * jnp.maximum(lv - 13, 0)


def q_attack_time(bonus_attack_speed: Any) -> Any:
    """GarenQAttack total attack time ``1.7 - 0.2 * bonus AS`` (windup is 20% of it)."""
    return Q_ATTACK_TIME - Q_ATTACK_TIME_AS * jnp.asarray(bonus_attack_speed, jnp.float32)


def e_crit_multiplier(crit_damage: Any) -> Any:
    """Judgment ``CriticalDamage``: ``1 + CritMod * (crit damage - 1)`` (26.1: 30% of bonus)."""
    return 1.0 + E_CRIT_MOD * (jnp.asarray(crit_damage, jnp.float32) - 1.0)


def cast(state: State, kctx, units, order) -> tuple[State, KitOut]:
    c, n = kctx.unit.shape[0], units.x.shape[0]
    g = _mine(kctx)
    r = kctx.ranks
    free = g & kctx.alive & ~kctx.stunned & ~kctx.silenced & ~state.r_pending
    ready = (r > 0) & (kctx.cooldowns <= 0)
    want = lambda s: (order.slot == s) & free          # noqa: E731
    t = jnp.clip(order.target, 0, n - 1)
    valid_r = (order.target >= 0) & units.alive[t] & units.targetable[t] & (units.kind[t] == KIND_CHAMPION) \
        & (units.team[t] != kctx.team) & (target_dist(kctx, units, order.target) <= R_RANGE + units.radius[t])
    q = want(0) & ready[:, 0] & ~state.q_on
    w = want(1) & ready[:, 1]
    e_start = want(2) & ready[:, 2] & ~state.e_on
    rr = want(3) & ready[:, 3] & valid_r
    # Recast after 1 s ends Judgment; Demacian Justice interrupts it.
    e_cancel = state.e_on & ((want(2) & (kctx.now - state.e_start >= E_MIN_SPIN - 1e-6)) | rr)
    started = q | w | e_start | rr
    slot = jnp.where(q, 0, jnp.where(w, 1, jnp.where(e_start, 2, jnp.where(rr, 3, -1)))).astype(jnp.int32)
    cid = jnp.where(started, make_cast_id(kctx, jnp.maximum(slot, 0)), 0).astype(jnp.int32)

    now = f32(kctx.now)
    ticks = ranked(NAME, "E", "NumTicks", r[:, 2]) \
        + jnp.floor(jnp.maximum(kctx.bonus_attack_speed, 0.0) / ranked(NAME, "E", "ASPerTick", r[:, 2]) + 1e-6)
    shield = jnp.where(w, ranked(NAME, "W", "BaseShield", r[:, 1]) + W_SHIELD_RATIO * kctx.bonus_hp, 0.0)
    # A new spin keeps the hit count on targets whose shred is still running (V25.12 fix).
    keep = kctx.now < state.shred_until
    state = state._replace(
        q_on=state.q_on | q, q_until=jnp.where(q, later(kctx, Q_WINDOW), state.q_until),
        q_cast_id=jnp.where(q, cid, state.q_cast_id),
        q_haste_on=state.q_haste_on | q,
        q_haste_until=jnp.where(q, f32(now + ranked(NAME, "Q", "MovementSpeedDuration", r[:, 0])), state.q_haste_until),
        w_on=state.w_on | w, w_until=jnp.where(w, later(kctx, W_DR_DURATION), state.w_until),
        w_dr=jnp.where(w, ranked(NAME, "W", "DRPercent", r[:, 1]), state.w_dr),
        w_ten_on=state.w_ten_on | w, w_ten_until=jnp.where(w, later(kctx, W_UPFRONT), state.w_ten_until),
        e_on=(state.e_on | e_start) & ~e_cancel, e_start=jnp.where(e_start, now, state.e_start),
        e_ticks=jnp.where(e_start, ticks.astype(jnp.int32), state.e_ticks),
        e_done=jnp.where(e_start, 0, state.e_done),
        e_hits=jnp.where(e_start[:, None] & ~keep, 0, state.e_hits),
        r_pending=state.r_pending | rr, r_fire_at=jnp.where(rr, later(kctx, R_CAST_TIME), state.r_fire_at),
        r_target=jnp.where(rr, order.target, state.r_target).astype(jnp.int32),
        r_seq=jnp.where(rr, units.spawn_seq[t], state.r_seq).astype(jnp.int32),
        r_cast_id=jnp.where(rr, cid, state.r_cast_id),
        reset_at=jnp.where(q, now, state.reset_at))
    cd_start = jnp.stack([jnp.zeros_like(q), w, e_cancel, rr], -1)
    return state, out(c, n, shield=shield_grants(shield, D.SHIELD_ALL, W_UPFRONT),
                      cooldown_start=cd_start, base_cooldown=_base_cd(kctx), attack_reset=q,
                      cast_started=started, cast_slot=slot, cast_id=cid,
                      cast_lockout=jnp.where(rr, R_CAST_TIME, 0.0).astype(jnp.float32), cleanse_slow=q)


def periodic(state: State, kctx, units) -> tuple[State, KitOut]:
    c, n = kctx.unit.shape[0], units.x.shape[0]
    g = _mine(kctx)
    alive = kctx.alive
    dead = ~alive
    r = kctx.ranks

    # Buff clocks (expire on the tick the countdown reaches 0, or on death).
    q_end = state.q_on & (due(kctx, state.q_until) | dead)
    q_haste_on = state.q_haste_on & ~(due(kctx, state.q_haste_until) | dead)
    w_on = state.w_on & ~(due(kctx, state.w_until) | dead)
    w_ten_on = state.w_ten_on & ~(due(kctx, state.w_ten_until) | dead)

    # Judgment: spin k completes at e_start + k * Duration / n (at most one per sim tick).
    count = state.e_ticks
    next_at = state.e_start + (state.e_done + 1).astype(jnp.float32) * E_DURATION / jnp.maximum(count, 1)
    fires = state.e_on & g & alive & (state.e_done < count) & due(kctx, next_at)
    d2 = center_dist(kctx, units)
    near = enemies(kctx, units) & (d2 <= E_RADIUS + units.radius[None, :])
    hit = fires[:, None] & near
    nearest = jnp.argmin(jnp.where(near, d2, jnp.inf), axis=1)
    bonus = jnp.where(jnp.arange(n)[None, :] == nearest[:, None], 1.0 + E_NEAREST, 1.0)
    roll = jax.random.uniform(jax.random.fold_in(state.key, tick_index(kctx)), (c,))
    crit = roll < kctx.crit_chance
    power = ranked(NAME, "E", "BaseDamagePerTick", r[:, 2]) + ranked(NAME, "E", "ADRatioPerTick", r[:, 2]) * kctx.total_ad
    raw = (power * jnp.where(crit, e_crit_multiplier(kctx.crit_damage), 1.0))[:, None] * bonus
    tick_id = make_cast_id(kctx, CODE_GAREN_E_TICK)
    flags = E_FLAGS | jnp.where(crit, D.PROP_CRIT, 0)
    p_e = D.packets(hit, kctx.unit[:, None], jnp.arange(n)[None, :], raw, D.PHYSICAL, flags[:, None],
                    cast_id=tick_id[:, None])
    hits = state.e_hits + hit.astype(jnp.int32)
    k = E_SHRED_HITS
    shred = hit & (units.kind[None, :] == KIND_CHAMPION) \
        & ((hits == k) | (hits == k + 1) | ((hits > k + 1) & ((hits - (k + 1)) % k == 0)))
    shred_until = jnp.where(shred, later_after_tick(kctx, E_SHRED_DURATION), state.shred_until)
    e_done = state.e_done + fires.astype(jnp.int32)
    e_end = state.e_on & ((e_done >= count) | dead)

    # Demacian Justice: true damage at the end of the cast time.
    rt = jnp.clip(state.r_target, 0, n - 1)
    r_due = state.r_pending & (due(kctx, state.r_fire_at) | dead)
    r_fire = r_due & alive & units.alive[rt] & (units.spawn_seq[rt] == state.r_seq)
    missing = jnp.maximum(units.max_hp[rt] - units.hp[rt], 0.0)
    r_raw = ranked(NAME, "R", "BaseDamage", r[:, 3]) + ranked(NAME, "R", "ExecuteDamage", r[:, 3]) * missing
    p_r = D.packets(r_fire, kctx.unit, rt, r_raw, D.TRUE, R_FLAGS, cast_id=state.r_cast_id)

    # Perseverance: RegenCalc per 5 s, paid in 0.5 s pulses while not disabled.
    on = g & alive & (kctx.now >= state.p_block_until - 1e-6)
    heal = jnp.where(on, kctx.max_hp * regen_rate(kctx.level) / 100.0 * (PASSIVE_PULSE / 5.0) * pulses(kctx, PASSIVE_PULSE),
                     0.0)

    state = state._replace(
        q_on=state.q_on & ~q_end, q_haste_on=q_haste_on, w_on=w_on, w_ten_on=w_ten_on,
        e_on=state.e_on & ~e_end, e_done=e_done, e_hits=hits,
        shred_until=shred_until, r_pending=state.r_pending & ~r_due)
    cd_start = jnp.zeros((c, 4), bool).at[:, 0].set(q_end & g).at[:, 2].set(e_end & g)
    return state, out(c, n, packets=D.concat_packets(p_e, p_r), heal=heal.astype(jnp.float32),
                      cooldown_start=cd_start, base_cooldown=_base_cd(kctx))


def on_attack(state: State, kctx, units, launch) -> tuple[State, KitOut]:
    return state, out(kctx.unit.shape[0], units.x.shape[0])


def on_hit(state: State, kctx, units, launch, dodging) -> tuple[State, KitOut]:
    """Empowered Q attack. ``dodging`` (N,) bool: target dodges basic attacks."""
    c, n = kctx.unit.shape[0], units.x.shape[0]
    g = _mine(kctx)
    target = holder_rows(launch.target, kctx)
    hit = holder_rows(launch.launched, kctx) & g & kctx.alive & state.q_on & (target >= 0)
    t = jnp.clip(target, 0, n - 1)
    land = hit & ~gather(dodging, target)
    raw = ranked(NAME, "Q", "BaseDamage", kctx.ranks[:, 0]) + (Q_AD_RATIO - 1.0) * kctx.total_ad
    p_q = D.packets(land, kctx.unit, t, raw, D.PHYSICAL, Q_FLAGS, cast_id=state.q_cast_id)
    mask = land[:, None] & (jnp.arange(n)[None, :] == t[:, None])
    silence, sid = cc_matrix(mask, ranked(NAME, "Q", "SilenceDuration", kctx.ranks[:, 0])[:, None],
                             state.q_cast_id[:, None])
    z = jnp.zeros((c, n), jnp.float32)
    cc = CCOut(z, z, silence, z, z, z, sid)
    lock = f32(kctx.now + (1.0 - Q_WINDUP_FRACTION) * q_attack_time(kctx.bonus_attack_speed))
    state = state._replace(q_on=state.q_on & ~hit, q_lock_until=jnp.where(hit, lock, state.q_lock_until))
    cd_start = jnp.zeros((c, 4), bool).at[:, 0].set(hit)
    return state, out(c, n, packets=p_q, cc=cc, cooldown_start=cd_start, base_cooldown=_base_cd(kctx))


def on_damage(state: State, kctx, units, report) -> tuple[State, KitOut]:
    """Perseverance lockout: health lost to an enemy champion or turret."""
    p = report.packets
    lost = p.raw > 0 if report.resolved is None else report.resolved.health_loss > 0
    src_kind = gather(units.kind, p.src)
    src_team = gather(units.team, p.src)
    counts = p.valid & lost & ((src_kind == KIND_CHAMPION) | (src_kind == KIND_TURRET))
    struck = jnp.any(counts[None, :] & (p.dst[None, :] == kctx.unit[:, None])
                     & (src_team[None, :] != kctx.team[:, None]), axis=1)
    block = jnp.where(_mine(kctx) & struck, f32(kctx.now + PASSIVE_DELAY), state.p_block_until)
    return state._replace(p_block_until=f32(block)), out(kctx.unit.shape[0], units.x.shape[0])


def on_takedown(state: State, kctx, units, kills) -> State:
    """Courage stacks: champion killing blows (no assists), minion last hits, monster kills."""
    monsters = jnp.sum(jnp.asarray(kills.killed_units, bool) & (units.kind[None, :] == KIND_MONSTER), axis=1)
    earned = jnp.asarray(kills.champion_kill, jnp.float32) + jnp.asarray(kills.minion_kill, jnp.float32) \
        + monsters.astype(jnp.float32)
    learned = kctx.ranks[:, 1] > 0
    stacks = jnp.where(_mine(kctx) & learned, jnp.minimum(W_MAX_STACKS, state.w_stacks + earned), state.w_stacks)
    return state._replace(w_stacks=f32(stacks))


def stats(state: State, kctx) -> ItemStats:
    g = _mine(kctx)
    resist = jnp.where(g, jnp.minimum(state.w_stacks, W_MAX_STACKS) * W_RESIST_PER_STACK, 0.0)
    return ItemStats(armor=f32(resist), magic_resist=f32(resist),
                     percent_move_speed=f32(jnp.where(g & state.q_haste_on, Q_MS, 0.0)))


def defense(state: State, kctx) -> KitDefense:
    g = _mine(kctx)
    c = kctx.unit.shape[0]
    return KitDefense(f32(jnp.where(g & state.w_on, 1.0 - state.w_dr, 1.0)), jnp.zeros((c,), bool),
                      jnp.ones((c,), jnp.float32), f32(jnp.where(g & state.w_ten_on, W_TENACITY, 0.0)))


def attack_mods(state: State, kctx) -> KitAttackMods:
    g = _mine(kctx)
    q = g & state.q_on
    # Lunge range only against champions; without the world input every target gets it.
    vs_champ = jnp.ones_like(q) if kctx.attack_target_kind is None else kctx.attack_target_kind == KIND_CHAMPION
    T = q_attack_time(kctx.bonus_attack_speed)
    return KitAttackMods(f32(jnp.where(q & vs_champ, Q_RANGE_BONUS, 0.0)),
                         g & (jnp.abs(kctx.now - state.reset_at) < 0.5 * kctx.dt),
                         g & (state.e_on | state.r_pending | (kctx.now < state.q_lock_until - 1e-6)),
                         jnp.zeros_like(q),
                         f32(jnp.where(q, Q_WINDUP_FRACTION * T, 0.0)), f32(jnp.where(q, T, 0.0)), q)


def ghosted(state: State, kctx) -> Any:
    return _mine(kctx) & state.e_on & kctx.alive


def debuffs(state: State, kctx, units) -> Debuffs:
    """Judgment armor shred on enemy champions, (N,)."""
    n = units.x.shape[0]
    on = jnp.any(_mine(kctx)[:, None] & (kctx.now < state.shred_until), axis=0)
    return neutral_debuffs(n)._replace(percent_armor_reduction=f32(jnp.where(on, E_SHRED, 0.0)))


__all__ = ["State", "init", "cast", "periodic", "on_attack", "on_hit", "on_damage", "on_takedown",
           "stats", "defense", "attack_mods", "debuffs", "ghosted", "regen_rate", "q_attack_time",
           "e_crit_multiplier"]
