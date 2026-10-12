"""Nasus (75), 26.19. Rules NAS.* and their evidence: docs/modern/CHAMPIONS.md; numbers from the pinned client
JSON (``modern/data/26.19/champions``)."""
from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from ..core import damage as D
from ..core.stat_pipeline import cc_duration
from ..core.types import KIND_CHAMPION, KIND_MINION, KIND_MONSTER, CCOut
from ..data.champions import spell
from ..items.catalog import ItemStats
from ..items.effects.core import Debuffs, neutral_debuffs
from ..jungle import camps as JC
from ..jungle import objectives as OBJ
from ..runes.catalog import ChampionTraits
from .core import (NEVER, KitAttackMods, KitDefense, KitOut, center_dist, cooldown_row, due, enemies, f32,
                   gather, holder_rows, later, make_cast_id, mana_row, out, ranked, scalar, target_dist)

NAME = "Nasus"
ID = 75
SKILL_ORDER = (0, 2, 1, 0, 0, 3, 0, 2, 0, 2, 3, 2, 2, 1, 1, 3, 1, 1)    # default ranks per level: Q > E > W
TRAITS = ChampionTraits(has_immobilize=False, resource="mana")          # rune legality: W only slows
_LS = spell(NAME, "Passive")["calculations"]["LifestealTooltip"]["mFormulaParts"][0]
P_LIFESTEAL = (_LS["mLevel1Value"], tuple((b["mLevel"], b["mAdditionalBonusAtThisLevel"]) for b in _LS["mBreakpoints"]))
Q_WINDOW = scalar(NAME, "Q", "BuffDuration")          # 10
Q_RANGE_BONUS = scalar(NAME, "Q", "BonusRange")       # 50
Q_STACKS = scalar(NAME, "Q", "BasicStacks")           # 4
Q_BIG_STACKS = scalar(NAME, "Q", "BigStacks")         # 10: champions, large minions, large monsters
W_RANGE = 700.0
W_CAST_TIME = 0.25
W_SLOW = scalar(NAME, "W", "SlowBase") / 100.0        # 0.35
W_DURATION = scalar(NAME, "W", "Duration")            # 5
W_STEPS = round(W_DURATION) - 1                       # the slow grows at 1..4 s and holds the 4 s value to the end
W_CRIPPLE = scalar(NAME, "W", "AttackSpeedSlowMult")  # 0.75 of the slow
E_RANGE = 650.0
E_CAST_TIME = 0.25
E_DELAY = 0.25                                        # wiki: the fire hits 0.25 s after the cast
E_RADIUS = 380.0                                      # client castRadius (wiki effect radius 400)
E_AP = 0.6
E_TICK_AP = 0.12
E_DURATION = scalar(NAME, "E", "Duration")            # 5: one tick per second
E_LINGER = 1.0
R_CAST_TIME = 0.2
R_DURATION = scalar(NAME, "R", "Duration")            # 15
R_RANGE_BONUS = scalar(NAME, "R", "AttackRangeIncrease")   # 50
R_TICK = scalar(NAME, "R", "TickRate")                # 0.5
R_CAP = scalar(NAME, "R", "MaxDamageCap")             # 240 per second
R_AP_PCT = 1e-4                                       # DamageCalc: +1% max HP per 100 AP per second
R_QCDR = scalar(NAME, "R", "QCDR")                    # 0.5
R_MIN_RADIUS = 400.0                                  # wiki aura radius at base size
R_RADIUS_PER_SIZE = 80.0                              # +0.8 per 1% bonus size (base gameplay radius 80)
UNIT_TARGET_RANGE = (0.0, W_RANGE, 0.0, 0.0)          # per slot; 0 = not unit-targeted (walk-in casting)
POINT_TARGET_RANGE = (0.0, 0.0, E_RANGE, 0.0)         # Spirit Fire walks to the location
LARGE_EPICS = (OBJ.T_GRUB, OBJ.T_HERALD, OBJ.T_BARON, OBJ.T_DRAKE, OBJ.T_ELDER)

Q_FLAGS = D.TAG_ACTIVE_SPELL | D.PROP_LIFESTEAL        # basic and spell damage at once (wiki)
E_FLAGS = D.TAG_AOE | D.TAG_ACTIVE_SPELL
E_TICK_FLAGS = D.TAG_AOE | D.TAG_ACTIVE_SPELL | D.TAG_PERIODIC
R_FLAGS = D.TAG_AOE | D.TAG_ACTIVE_SPELL | D.TAG_PERIODIC | D.PROP_ULTIMATE


class State(NamedTuple):
    q_on: Any               # (C,) empowered attack window open
    q_until: Any
    q_cast_id: Any
    q_stacks: Any           # permanent Siphoning Strike stacks
    q_hit: Any              # (C,) int32 unit the last empowered attack landed on (-1 none)
    q_hit_at: Any           # its time (stacks are granted if it dies to Nasus on that tick)
    w_pending: Any
    w_fire_at: Any
    w_target: Any
    w_seq: Any
    w_cast_id: Any
    w_on: Any               # Wither running on w_target
    w_start: Any
    w_scale: Any            # CC duration scale of the target's tenacity at application
    w_per: Any              # slow added per step (rank at cast)
    w_step: Any             # int32 steps applied
    e_pending: Any          # cast and delay before the first hit
    e_fire_at: Any
    e_on: Any               # fire burning
    e_x: Any
    e_y: Any
    e_until: Any
    e_ticks: Any            # int32 DoT ticks paid
    e_dot: Any              # per-tick raw (rank and AP at cast)
    e_shred: Any            # armor reduction (rank at cast)
    e_cast_id: Any
    shred_until: Any        # (C, N) armor reduction end
    r_pending: Any
    r_fire_at: Any
    r_on: Any
    r_start: Any
    r_until: Any
    r_cast_id: Any
    r_rank: Any             # int32 rank the storm was cast at
    r_ticks: Any            # int32 storm ticks paid


def init(n_champions: int, n_units: int) -> State:
    c, n = n_champions, n_units
    z = jnp.zeros((c,), jnp.float32)
    f = jnp.zeros((c,), bool)
    i = jnp.zeros((c,), jnp.int32)
    return State(f, z, i, z, i - 1, z - NEVER,
                 f, z + NEVER, i - 1, i, i, f, z, z + 1.0, z, i,
                 f, z + NEVER, f, z, z, z, i, z, z, i, jnp.zeros((c, n), jnp.float32),
                 f, z + NEVER, f, z, z, i, i, i)


def _mine(kctx) -> Any:
    return kctx.champion_id == ID


def _base_cd(state: State, kctx) -> Any:
    """Base cooldowns; Fury of the Sands halves Siphoning Strike's (NAS.R2)."""
    cd = cooldown_row(NAME, kctx.ranks)
    cd = cd.at[:, 0].multiply(jnp.where(state.r_on, 1.0 - R_QCDR, 1.0))
    return jnp.where(_mine(kctx)[:, None], cd, 0.0)


def lifesteal(level: Any) -> Any:
    """Soul Eater life steal: 10% from level 1, +5% at 7 and 13 (``LifestealTooltip`` breakpoints)."""
    lv = jnp.asarray(level, jnp.float32)
    base, bps = P_LIFESTEAL
    return (base + sum(jnp.where(lv >= l, add, 0.0) for l, add in bps)) / 100.0


def w_slow(start_slow: Any, per: Any, step: Any) -> Any:
    return start_slow + per * jnp.asarray(step, jnp.float32)


def r_radius(rank: Any) -> Any:
    """Storm radius (center to edge): 400 + 0.8 per 1% bonus size (``SizeIncreasePercent`` 30/35/40%)."""
    return R_MIN_RADIUS + R_RADIUS_PER_SIZE * ranked(NAME, "R", "SizeIncreasePercent", rank)


def _large(units, t) -> Any:
    kind, sub = units.kind[t], units.sub[t]
    size = jnp.asarray(JC.SIZE, jnp.int32)[jnp.clip(sub, 0, len(JC.SIZE) - 1)]
    epic = jnp.zeros(jnp.shape(sub), bool)
    for e in LARGE_EPICS:
        epic = epic | (sub == OBJ.SUB_BASE + e)
    camp = (sub >= 0) & (sub < len(JC.SIZE)) & (size == JC.LARGE)
    return (kind == KIND_CHAMPION) | ((kind == KIND_MINION) & (sub >= 2)) | ((kind == KIND_MONSTER) & (camp | epic))


def cast(state: State, kctx, units, order) -> tuple[State, KitOut]:
    c, n = kctx.unit.shape[0], units.x.shape[0]
    m = _mine(kctx)
    r = kctx.ranks
    casting_e = state.e_pending & (kctx.now < state.e_fire_at - E_DELAY - 1e-6)      # the 0.25 s delay is free
    busy = state.w_pending | casting_e | state.r_pending
    free = m & kctx.alive & ~kctx.stunned & ~kctx.silenced & ~busy
    cost = mana_row(NAME, r)
    ready = (r > 0) & (kctx.cooldowns <= 0) & (kctx.mana[:, None] >= cost)
    want = lambda s: (order.slot == s) & free          # noqa: E731
    t = jnp.clip(order.target, 0, n - 1)
    valid_w = (order.target >= 0) & units.alive[t] & units.targetable[t] & (units.kind[t] == KIND_CHAMPION) \
        & (units.team[t] != kctx.team) & (target_dist(kctx, units, order.target) <= W_RANGE + units.radius[t])
    to_point = jnp.sqrt((order.x - kctx.x) ** 2 + (order.y - kctx.y) ** 2)
    q = want(0) & ready[:, 0] & ~state.q_on
    w = want(1) & ready[:, 1] & valid_w
    e = want(2) & ready[:, 2] & (to_point <= E_RANGE)
    rr = want(3) & ready[:, 3]
    started = q | w | e | rr
    slot = jnp.where(q, 0, jnp.where(w, 1, jnp.where(e, 2, jnp.where(rr, 3, -1)))).astype(jnp.int32)
    cid = jnp.where(started, make_cast_id(kctx, jnp.maximum(slot, 0)), 0).astype(jnp.int32)
    state = state._replace(
        q_on=state.q_on | q, q_until=jnp.where(q, later(kctx, Q_WINDOW), state.q_until),
        q_cast_id=jnp.where(q, cid, state.q_cast_id),
        w_pending=state.w_pending | w, w_fire_at=jnp.where(w, later(kctx, W_CAST_TIME), state.w_fire_at),
        w_target=jnp.where(w, order.target, state.w_target).astype(jnp.int32),
        w_seq=jnp.where(w, units.spawn_seq[t], state.w_seq).astype(jnp.int32),
        w_cast_id=jnp.where(w, cid, state.w_cast_id),
        e_pending=state.e_pending | e, e_fire_at=jnp.where(e, later(kctx, E_CAST_TIME + E_DELAY), state.e_fire_at),
        e_x=jnp.where(e, f32(order.x), state.e_x), e_y=jnp.where(e, f32(order.y), state.e_y),
        e_cast_id=jnp.where(e, cid, state.e_cast_id),
        r_pending=state.r_pending | rr, r_fire_at=jnp.where(rr, later(kctx, R_CAST_TIME), state.r_fire_at),
        r_cast_id=jnp.where(rr, cid, state.r_cast_id), r_rank=jnp.where(rr, r[:, 3], state.r_rank))
    mana = jnp.sum(jnp.where(jnp.stack([q, w, e, rr], -1), cost, 0.0), axis=1)
    lock = jnp.where(w, W_CAST_TIME, jnp.where(e, E_CAST_TIME, jnp.where(rr, R_CAST_TIME, 0.0)))
    return state, out(c, n, mana_cost=f32(mana), cooldown_start=jnp.stack([jnp.zeros_like(q), w, e, rr], -1),
                      base_cooldown=_base_cd(state, kctx), attack_reset=q, cast_started=started, cast_slot=slot,
                      cast_id=cid, cast_lockout=f32(lock))


def periodic(state: State, kctx, units) -> tuple[State, KitOut]:
    c, n = kctx.unit.shape[0], units.x.shape[0]
    m = _mine(kctx)
    alive = kctx.alive
    dead = ~alive
    r = kctx.ranks
    now = f32(kctx.now)
    cols = jnp.arange(n)[None, :]

    # Siphoning Strike window: expiry or death starts the cooldown (post-effect).
    q_end = state.q_on & (due(kctx, state.q_until) | dead)

    # Wither: applied at the end of the cast time, then one step per second scaled by the target's tenacity.
    wt = jnp.clip(state.w_target, 0, n - 1)
    w_ok = units.alive[wt] & (units.spawn_seq[wt] == state.w_seq)
    w_due = state.w_pending & (due(kctx, state.w_fire_at) | dead)
    w_fire = w_due & alive & w_ok
    ten = jnp.zeros((c,), jnp.float32) if kctx.tenacity is None else kctx.tenacity
    tgt_ten = jnp.where(wt < c, gather(ten, wt), 0.0)
    scale = jnp.where(w_fire, cc_duration(W_DURATION, tgt_ten) / W_DURATION, state.w_scale)
    per = jnp.where(w_fire, ranked(NAME, "W", "SlowPerTick", r[:, 1]) / 100.0, state.w_per)
    w_on = (state.w_on | w_fire) & w_ok
    start = jnp.where(w_fire, now, state.w_start)
    # Steps fire on the tick at their time, so each re-applied slow ends with the first one (5 x scale).
    step_due = w_on & ~w_fire & (state.w_step < W_STEPS) \
        & (now >= start + (state.w_step + 1).astype(jnp.float32) * scale - 1e-4)
    step = jnp.where(w_fire, 0, state.w_step + step_due.astype(jnp.int32))
    emit = w_fire | step_due
    w_mask = emit[:, None] & (cols == wt[:, None])
    slow = jnp.where(w_mask, w_slow(W_SLOW, per, step)[:, None], 0.0)
    slow_dur = jnp.where(w_mask, (W_DURATION - step.astype(jnp.float32))[:, None], 0.0)
    w_sid = jnp.where(w_mask, state.w_cast_id[:, None], 0)
    w_on = w_on & (now < start + W_DURATION * scale - 1e-4)

    # Spirit Fire: the first hit, then one tick per second; enemies inside lose armor until 1 s after leaving.
    e_cancel = state.e_pending & dead & (now < state.e_fire_at - E_DELAY - 1e-6)    # died during the cast time
    e_due = state.e_pending & ~e_cancel & due(kctx, state.e_fire_at)
    e_rank = r[:, 2]
    in_fire = enemies(kctx, units) & (jnp.sqrt((units.x[None, :] - state.e_x[:, None]) ** 2
                                               + (units.y[None, :] - state.e_y[:, None]) ** 2)
                                      <= E_RADIUS + units.radius[None, :])
    e_hit = e_due[:, None] & in_fire
    e_raw = ranked(NAME, "E", "InitialHitDamage", e_rank) + E_AP * kctx.ap
    p_e = D.packets(e_hit, kctx.unit[:, None], cols, e_raw[:, None], D.MAGIC, E_FLAGS,
                    cast_id=state.e_cast_id[:, None])
    e_on = state.e_on | e_due
    e_until = jnp.where(e_due, f32(now + E_DURATION), state.e_until)
    e_dot = jnp.where(e_due, ranked(NAME, "E", "DamagePerTick", e_rank) + E_TICK_AP * kctx.ap, state.e_dot)
    e_shred = jnp.where(e_due, -ranked(NAME, "E", "ArmorShredPercent", e_rank), state.e_shred)
    e_ticks = jnp.where(e_due, 0, state.e_ticks)
    tick = e_on & ~e_due & (e_ticks < round(E_DURATION)) \
        & due(kctx, e_until - E_DURATION + (e_ticks + 1).astype(jnp.float32))
    t_hit = tick[:, None] & in_fire
    p_t = D.packets(t_hit, kctx.unit[:, None], cols, e_dot[:, None], D.MAGIC, E_TICK_FLAGS,
                    cast_id=state.e_cast_id[:, None])
    e_ticks = e_ticks + tick.astype(jnp.int32)
    burning = e_on & (now < e_until - 1e-6)
    shred_until = jnp.where(burning[:, None] & in_fire, f32(now + kctx.dt + E_LINGER), state.shred_until)
    e_on = e_on & (e_ticks < round(E_DURATION))

    # Fury of the Sands: starts after its cast time; storm ticks every 0.5 s from the start; ends on death.
    r_due = state.r_pending & (due(kctx, state.r_fire_at) | dead)
    r_fire = r_due & alive
    r_on = (state.r_on & ~(due(kctx, state.r_until) | dead)) | r_fire
    r_start = jnp.where(r_fire, now, state.r_start)
    r_until = jnp.where(r_fire, f32(now + R_DURATION), state.r_until)
    r_ticks = jnp.where(r_fire, 0, state.r_ticks)
    storm = state.r_on & alive & ~r_fire & (r_ticks < round(R_DURATION / R_TICK)) \
        & due(kctx, r_start + (r_ticks + 1).astype(jnp.float32) * R_TICK)
    r_hit = storm[:, None] & enemies(kctx, units) \
        & (center_dist(kctx, units) <= r_radius(state.r_rank)[:, None] + units.radius[None, :])
    pct = (ranked(NAME, "R", "AOEDamagePercent", state.r_rank) + R_AP_PCT * kctx.ap) * R_TICK
    r_raw = jnp.minimum(pct[:, None] * units.max_hp[None, :], R_CAP * R_TICK)
    p_r = D.packets(r_hit, kctx.unit[:, None], cols, r_raw, D.MAGIC, R_FLAGS, cast_id=state.r_cast_id[:, None])

    state = state._replace(
        q_on=state.q_on & ~q_end,
        w_pending=state.w_pending & ~w_due, w_on=w_on, w_start=start, w_scale=f32(scale), w_per=f32(per),
        w_step=step.astype(jnp.int32),
        e_pending=state.e_pending & ~(e_due | e_cancel), e_on=e_on, e_until=f32(e_until), e_ticks=e_ticks.astype(jnp.int32),
        e_dot=f32(e_dot), e_shred=f32(e_shred), shred_until=f32(shred_until),
        r_pending=state.r_pending & ~r_due, r_on=r_on, r_start=f32(r_start), r_until=f32(r_until),
        r_ticks=(r_ticks + storm.astype(jnp.int32)).astype(jnp.int32))
    z = jnp.zeros((c, n), jnp.float32)
    cc = CCOut(z, z, z, z, f32(slow), f32(slow_dur), w_sid.astype(jnp.int32))
    cd_start = jnp.zeros((c, 4), bool).at[:, 0].set(q_end & m)
    return state, out(c, n, packets=D.concat_packets(p_e, p_t, p_r), cc=cc, cooldown_start=cd_start,
                      base_cooldown=_base_cd(state, kctx))


def on_attack(state: State, kctx, units, launch) -> tuple[State, KitOut]:
    return state, out(kctx.unit.shape[0], units.x.shape[0], base_cooldown=_base_cd(state, kctx))


def on_hit(state: State, kctx, units, launch, dodging) -> tuple[State, KitOut]:
    """Siphoning Strike on the landed (or dodged) empowered attack. ``dodging`` (N,) bool."""
    c, n = kctx.unit.shape[0], units.x.shape[0]
    m = _mine(kctx)
    target = holder_rows(launch.target, kctx)
    hit = holder_rows(launch.launched, kctx) & m & kctx.alive & state.q_on & (target >= 0)
    t = jnp.clip(target, 0, n - 1)
    land = hit & ~gather(dodging, target)
    crit = jnp.zeros((c,), bool) if launch.is_crit is None else holder_rows(launch.is_crit, kctx)
    # NAS.Q2: the flat bonus crits with the attack, the stacks never do (26.18).
    raw = ranked(NAME, "Q", "BonusDamage", kctx.ranks[:, 0]) * jnp.where(crit, kctx.crit_damage, 1.0) \
        + state.q_stacks
    p_q = D.packets(land, kctx.unit, t, raw, D.PHYSICAL, Q_FLAGS, cast_id=state.q_cast_id)
    state = state._replace(q_on=state.q_on & ~hit,
                           q_hit=jnp.where(land, target, state.q_hit).astype(jnp.int32),
                           q_hit_at=jnp.where(land, f32(kctx.now), state.q_hit_at))
    cd_start = jnp.zeros((c, 4), bool).at[:, 0].set(hit)
    return state, out(c, n, packets=p_q, cooldown_start=cd_start, base_cooldown=_base_cd(state, kctx))


def on_damage(state: State, kctx, units, report) -> tuple[State, KitOut]:
    return state, out(kctx.unit.shape[0], units.x.shape[0])


def on_takedown(state: State, kctx, units, kills) -> State:
    """Siphoning Strike stacks when the empowered attack's target dies to Nasus on the tick it landed; champions
    need the killing blow."""
    t = jnp.clip(state.q_hit, 0, units.x.shape[0] - 1)
    now_hit = (state.q_hit >= 0) & (jnp.abs(state.q_hit_at - kctx.now) < 0.5 * kctx.dt)
    took = jnp.take_along_axis(jnp.asarray(kills.killed_units, bool), t[:, None], axis=1)[:, 0]
    champ = units.kind[t] == KIND_CHAMPION
    killed = now_hit & took & (~champ | (jnp.asarray(kills.champion_kill) > 0))
    gain = jnp.where(_large(units, t), Q_BIG_STACKS, Q_STACKS)
    return state._replace(q_stacks=f32(jnp.where(_mine(kctx) & killed, state.q_stacks + gain, state.q_stacks)))


def stats(state: State, kctx) -> ItemStats:
    m = _mine(kctx)
    r = m & state.r_on
    resist = jnp.where(r, ranked(NAME, "R", "InitialResistGain", state.r_rank), 0.0)
    return ItemStats(life_steal=f32(jnp.where(m, lifesteal(kctx.level), 0.0)),
                     health=f32(jnp.where(r, ranked(NAME, "R", "BonusHealth", state.r_rank), 0.0)),
                     armor=f32(resist), magic_resist=f32(resist))


def defense(state: State, kctx) -> KitDefense:
    c = kctx.unit.shape[0]
    o = jnp.ones((c,), jnp.float32)
    return KitDefense(o, jnp.zeros((c,), bool), o, jnp.zeros((c,), jnp.float32))


def attack_mods(state: State, kctx) -> KitAttackMods:
    m = _mine(kctx)
    c = kctx.unit.shape[0]
    z = jnp.zeros((c,), jnp.float32)
    q = m & state.q_on
    rng = jnp.where(q, Q_RANGE_BONUS, 0.0) + jnp.where(m & state.r_on, R_RANGE_BONUS, 0.0)
    return KitAttackMods(f32(rng), jnp.zeros((c,), bool), jnp.zeros((c,), bool), jnp.zeros((c,), bool), z, z, q)


def debuffs(state: State, kctx, units) -> Debuffs:
    """Spirit Fire armor reduction and the Wither cripple (75% of the current slow), (N,)."""
    n = units.x.shape[0]
    m = _mine(kctx)
    shred = jnp.max(jnp.where(m[:, None] & (kctx.now < state.shred_until), state.e_shred[:, None], 0.0), axis=0)
    w = m & state.w_on
    cripple = W_CRIPPLE * w_slow(W_SLOW, state.w_per, state.w_step)
    row = (jnp.arange(n)[None, :] == state.w_target[:, None]) & w[:, None]
    as_cripple = jnp.max(jnp.where(row, cripple[:, None], 0.0), axis=0)
    return neutral_debuffs(n)._replace(percent_armor_reduction=f32(shred), attack_speed_cripple=f32(as_cripple))


__all__ = ["State", "init", "cast", "periodic", "on_attack", "on_hit", "on_damage", "on_takedown", "stats",
           "defense", "attack_mods", "debuffs", "lifesteal", "w_slow", "r_radius"]
