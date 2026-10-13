"""Ambessa (799), 26.19. Rules AMB.* and their evidence: docs/modern/CHAMPIONS.md; numbers from the pinned client
JSON (``modern/data/26.19/champions``). Ambessa uses energy (the world's resource pool with ``uses_energy``).
Drakehound's Step reads the champion's move / attack commands (``KitCtx.last_order_at``, ``move_goal``,
``attack_order``) around each ability lockout; its dash stops at walls (``KitOut.dash_stops_at_walls``)."""
from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from ..core import damage as D
from ..core.types import KIND_CHAMPION, KIND_MINION, KIND_MONSTER, KIND_TURRET, CCOut, Dash
from ..data.champions import spell
from ..items.catalog import ItemStats
from ..items.effects.core import Debuffs, ShieldGrant, neutral_debuffs
from ..runes.catalog import ChampionTraits
from .core import (CODE_AMBESSA_P, CODE_AMBESSA_SPIN2, NEVER, KitAttackMods, KitDefense, KitOut, cc_matrix,
                   cooldown_row, due, enemies, f32, gather, holder_rows, later, make_cast_id, mana_row, out, ranked,
                   scalar)
from .nasus import _large

NAME = "Ambessa"
ID = 799
SKILL_ORDER = (0, 2, 1, 0, 0, 3, 0, 2, 0, 2, 3, 2, 2, 1, 1, 3, 1, 1)    # default ranks per level: Q > E > W
TRAITS = ChampionTraits(has_immobilize=True, resource="energy")        # rune legality: R suppresses and stuns
UNIT_TARGET_RANGE = (0.0, 0.0, 0.0, 0.0)

_P = spell(NAME, "Passive")["calculations"]
P_DASH_MIN = scalar(NAME, "Passive", "Buffer_Dash_Min_Distance")      # 175
P_DASH_MAX = scalar(NAME, "Passive", "Buffer_Dash_Distance")          # 350
P_WINDOW = scalar(NAME, "Passive", "Buffer_Dash_Forgiveness_Window")  # 0.275 after a lockout
P_WINDOW_W = scalar(NAME, "Passive", "Buffer_Dash_W_Forgiveness_Window")   # 0.25 after Repudiation's
_SPD = _P["{57d26a55}"]["mFormulaParts"][0]
P_SPEED = _SPD["mLevel1Value"]                         # 770, +60 at levels 6, 11 and 16, + move speed
P_SPEED_STEPS = tuple(b["mLevel"] for b in _SPD["mBreakpoints"])
P_SPEED_STEP = _SPD["mBreakpoints"][0]["mAdditionalBonusAtThisLevel"]
P_STACKS = int(scalar(NAME, "Passive", "Attack_Buff_Max_Stacks"))    # 3
P_STACK_TIME = scalar(NAME, "Passive", "Attack_Buff_Duration")       # 4
P_RANGE = scalar(NAME, "Passive", "Attack_Range_Amount")             # +75
_PD = _P["Calc_OnHit_Damage_Flat"]["mFormulaParts"]
P_DAMAGE = (_PD[0]["mStartValue"], _PD[0]["mEndValue"])              # 5 -> 30 by level (26.19; 26.20: 25)
P_DAMAGE_AD = _PD[1]["mCoefficient"]                                 # 25% bonus AD (26.19; 26.20: 20%)
_PE = _P["Calc_OnHit_Energy_Refund"]["mFormulaParts"][0]
P_REFUND = _PE["mLevel1Value"]                         # 40, +15 at levels 7 and 13
P_REFUND_STEPS = tuple(b["mLevel"] for b in _PE["mBreakpoints"])
P_REFUND_STEP = _PE["mBreakpoints"][0]["mAdditionalBonusAtThisLevel"]
# The empowered attacks are AmbessaBasicAttackChained*: uncancellable, with an overridden total attack time.
P_CHAIN_TIME = (0.35, -0.05, -0.05)                    # 0.35, -0.05 at levels 7 and 13 (mOverrideAttackTime)
P_CHAIN_STEPS = (7, 13)
BASE_WINDUP = 0.25 / 1.6                               # the record's mAttackCastTime / mAttackTotalTime

LOCKOUT = 0.225                                        # wiki: Q1, Q2, W and E lock out for 0.225 s
Q_SWAP = scalar(NAME, "Q", "Swap_Duration")            # Sundering Slam within 4 s of a Cunning Sweep hit
Q_SWAP_STATIC = scalar(NAME, "Q", "Swap_Static_Cooldown")       # 0.5
Q1_OUTER, Q1_INNER, Q1_NEAR = 400.0, 275.0, 135.0      # wiki: 180 degree cone; edge beyond 275 takes full damage
Q2_LENGTH, Q2_HALF_WIDTH = 650.0, 40.0                 # wiki: 650 x 80 (edge range)
Q_MIN_RATIO = scalar(NAME, "Q", "Damage_1_Min_Ratio")  # inner / secondary hits deal 50%
Q_MONSTER_FLAT = 75.0                                  # Calc_Damage_Monster_Flat_Bonus (26.10)
_QC = spell(NAME, "Q")["calculations"]["Calc_Damage_Monster_Percent_Cap"]["mFormulaParts"][0]
Q_MONSTER_CAP = (_QC["mStartValue"], _QC["mEndValue"])  # 100 -> 300 by level
Q1_PCT_AD = 0.0003                                     # Calc_Damage_1_Percent_Max: +3% max HP per 100 bonus AD
Q2_PCT_AD = 0.0004                                     # Calc_Damage_2_Percent_Max: +4% per 100 bonus AD

W_BRACE = scalar(NAME, "W", "Buff_Duration")           # 0.5
W_SHIELD_TIME = scalar(NAME, "W", "Shield_Duration")   # 1.5
_WS = spell(NAME, "W")["calculations"]["Calc_Shield"]["mFormulaParts"]
W_SHIELD = (_WS[0]["mStartValue"], _WS[0]["mEndValue"])  # 50 -> 320 by level
W_SHIELD_AD = _WS[1]["mCoefficient"]                   # 150% bonus AD
W_HIGH = scalar(NAME, "W", "HighDamageMultiplier")     # x1.5 after blocking a champion, large monster or turret
W_RADIUS = 325.0
E_RADIUS = 325.0
E_SLOW = scalar(NAME, "E", "Slow_Amount")              # 99%, decaying over 1 s
E_SLOW_TIME = scalar(NAME, "E", "Slow_Duration")       # 1
R_CAST = 0.7                                           # 26.9
R_LENGTH, R_HALF_WIDTH = 1250.0, 65.0                  # wiki: 1250 x 130 (edge range)
R_SUPPRESS = scalar(NAME, "R", "Suppress_Duration")    # 0.75
R_STUN = scalar(NAME, "R", "Stun_Duration")            # 0.4
R_HEAL_LS = scalar(NAME, "R", "Omnivamp_LifeStealScaling")      # + 50% of life steal
R_HEAL_SMALL = scalar(NAME, "R", "Omnivamp_MinionMod")          # 25% against minions and monsters

ABILITY = D.TAG_ACTIVE_SPELL
AOE = D.TAG_AOE | D.TAG_ACTIVE_SPELL
P_FLAGS = D.TAG_PROC | D.PROP_LIFESTEAL


class State(NamedTuple):
    stacks: Any             # (C,) int32 Medarda Maxim
    stacks_until: Any
    stack_at: Any           # (C,) a stack arrives when the current lockout ends
    armed: Any              # (C,) bool: the attack launched this tick is empowered
    win_from: Any           # (C,) Drakehound's Step window: commands from this time count ...
    win_lock_end: Any       # ... the dash starts when the lockout ends ...
    win_until: Any          # ... or until the forgiveness window closes
    win_kind: Any           # (C,) int32 ability that opened it: 0 Q, 1 W, 2 E, 3 R
    dashing: Any            # (C,) bool: a Drakehound's Step dash is under way
    dash_from: Any
    q_two_from: Any         # Sundering Slam allowed from ...
    q_two_until: Any        # ... until
    q_pending: Any
    q_fire_at: Any
    q_two: Any              # (C,) bool: the pending cast is Sundering Slam
    q_dx: Any
    q_dy: Any
    q_cast_id: Any
    w_smash_at: Any         # brace end
    w_blocked: Any          # (C,) bool: the shield absorbed champion / large monster / turret damage
    w_cast_id: Any
    e_pending: Any
    e_fire_at: Any
    e_cast_id: Any
    e_second: Any           # (C,) bool: the second spin comes at the end of the dash
    e_slow_from: Any        # (C, N) Lacerate slow start (decaying)
    r_pending: Any
    r_fire_at: Any
    r_dx: Any
    r_dy: Any
    r_cast_id: Any
    r_target: Any           # (C,) int32 seized champion, -1 none
    r_seq: Any
    r_slam_at: Any          # suppression end: the slam


def init(n_champions: int, n_units: int) -> State:
    c, n = n_champions, n_units
    z = jnp.zeros((c,), jnp.float32)
    f = jnp.zeros((c,), bool)
    i = jnp.zeros((c,), jnp.int32)
    return State(i, z - NEVER, z + NEVER, f, z - NEVER, z + NEVER, z - NEVER, i, f, z - NEVER,
                 z - NEVER, z - NEVER, f, z + NEVER, f, z, z, i,
                 z + NEVER, f, i,
                 f, z + NEVER, i, f, jnp.zeros((c, n), jnp.float32) - NEVER,
                 f, z + NEVER, z, z, i, i - 1, i, z + NEVER)


def _mine(kctx) -> Any:
    return kctx.champion_id == ID


def by_level(pair, level) -> Any:
    lv = jnp.clip(jnp.asarray(level, jnp.float32), 1.0, 18.0)
    return pair[0] + (pair[1] - pair[0]) * (lv - 1.0) / 17.0


def _steps(level, steps) -> Any:
    lv = jnp.asarray(level)
    return sum((lv >= s).astype(jnp.float32) for s in steps)


def dash_speed(level, move_speed) -> Any:
    """Drakehound's Step: 770, +60 at levels 6, 11 and 16, + move speed."""
    return P_SPEED + P_SPEED_STEP * _steps(level, P_SPEED_STEPS) + move_speed


def chain_time(level) -> Any:
    """Total attack time of an empowered (chained) attack: 0.35 s, 0.30 from level 7, 0.25 from 13."""
    return P_CHAIN_TIME[0] + P_CHAIN_TIME[1] * _steps(level, P_CHAIN_STEPS)


def energy_refund(level) -> Any:
    return P_REFUND + P_REFUND_STEP * _steps(level, P_REFUND_STEPS)


def _base_cd(kctx) -> Any:
    return jnp.where(_mine(kctx)[:, None], cooldown_row(NAME, kctx.ranks), 0.0)


def _unit_dir(dx, dy):
    d = jnp.sqrt(dx * dx + dy * dy)
    ok = d > 1e-3
    return jnp.where(ok, dx / jnp.maximum(d, 1e-3), 1.0), jnp.where(ok, dy / jnp.maximum(d, 1e-3), 0.0), d


def _open_window(state: State, kctx, opened, lockout, forgive, kind) -> State:
    now = f32(kctx.now)
    return state._replace(win_from=jnp.where(opened, now, state.win_from),
                          win_lock_end=jnp.where(opened, f32(now + lockout), state.win_lock_end),
                          win_until=jnp.where(opened, f32(now + lockout + forgive), state.win_until),
                          win_kind=jnp.where(opened, kind, state.win_kind).astype(jnp.int32),
                          stack_at=jnp.where(opened, f32(now + lockout), state.stack_at))


def cast(state: State, kctx, units, order) -> tuple[State, KitOut]:
    c, n = kctx.unit.shape[0], units.x.shape[0]
    m = _mine(kctx)
    r = kctx.ranks
    now = f32(kctx.now)
    free = m & kctx.alive & ~kctx.stunned & ~kctx.silenced & ~state.q_pending & ~state.e_pending \
        & ~state.r_pending & (state.r_target < 0)                    # not while seizing a target
    rooted = jnp.zeros_like(m) if kctx.rooted is None else kctx.rooted
    cost = mana_row(NAME, r)
    ready = (r > 0) & (kctx.cooldowns <= 0) & (kctx.mana[:, None] >= cost)
    want = lambda s: (order.slot == s) & free          # noqa: E731
    ux, uy, _ = _unit_dir(order.x - kctx.x, order.y - kctx.y)

    two_ok = (now >= state.q_two_from - 1e-6) & (now <= state.q_two_until)
    q2 = want(0) & (r[:, 0] > 0) & two_ok & (kctx.mana >= cost[:, 0])
    q1 = want(0) & ready[:, 0] & ~two_ok
    q = q1 | q2
    w = want(1) & ready[:, 1]
    e = want(2) & ready[:, 2]
    rr = want(3) & ready[:, 3] & ~rooted

    started = q | w | e | rr
    slot = jnp.where(q, 0, jnp.where(w, 1, jnp.where(e, 2, jnp.where(rr, 3, -1)))).astype(jnp.int32)
    cid = jnp.where(started, make_cast_id(kctx, jnp.maximum(slot, 0)), 0).astype(jnp.int32)
    # Repudiation's shield at cast.
    sh = jnp.where(w, by_level(W_SHIELD, kctx.level) + W_SHIELD_AD * kctx.bonus_ad, 0.0)[:, None]
    shield = ShieldGrant(f32(sh), jnp.zeros((c, 1), jnp.int32), jnp.where(w, W_SHIELD_TIME, 0.0)[:, None],
                         jnp.zeros((c, 1), jnp.float32))
    state = _open_window(state, kctx, q | e, LOCKOUT, P_WINDOW, jnp.where(q, 0, 2))
    state = _open_window(state, kctx, w, LOCKOUT, P_WINDOW_W, 1)
    state = state._replace(
        q_pending=state.q_pending | q, q_fire_at=jnp.where(q, later(kctx, LOCKOUT), state.q_fire_at),
        q_two=jnp.where(q, q2, state.q_two), q_dx=jnp.where(q, f32(ux), state.q_dx),
        q_dy=jnp.where(q, f32(uy), state.q_dy), q_cast_id=jnp.where(q, cid, state.q_cast_id),
        q_two_until=jnp.where(q2, -NEVER, state.q_two_until),
        w_smash_at=jnp.where(w, later(kctx, W_BRACE), state.w_smash_at), w_blocked=jnp.where(w, False, state.w_blocked),
        w_cast_id=jnp.where(w, cid, state.w_cast_id),
        e_pending=state.e_pending | e, e_fire_at=jnp.where(e, later(kctx, LOCKOUT), state.e_fire_at),
        e_cast_id=jnp.where(e, cid, state.e_cast_id), e_second=jnp.where(e, False, state.e_second),
        r_pending=state.r_pending | rr, r_fire_at=jnp.where(rr, later(kctx, R_CAST), state.r_fire_at),
        r_dx=jnp.where(rr, f32(ux), state.r_dx), r_dy=jnp.where(rr, f32(uy), state.r_dy),
        r_cast_id=jnp.where(rr, cid, state.r_cast_id))
    mana = jnp.sum(jnp.where(jnp.stack([q, w, e, rr], -1), cost, 0.0), axis=1)
    lock = jnp.where(q | w | e, LOCKOUT, jnp.where(rr, R_CAST, 0.0))
    cd = jnp.stack([q1, w, e, rr], -1)
    return state, out(c, n, mana_cost=f32(mana), shield=shield, cooldown_start=cd, base_cooldown=_base_cd(kctx),
                      cast_started=started, cast_slot=slot, cast_id=cid, cast_lockout=f32(lock))


def _q_raw(kctx, units, two) -> Any:
    """(C, N) full Cunning Sweep / Sundering Slam damage (outer edge / first target)."""
    r = kctx.ranks[:, 0]
    base = jnp.where(two, ranked(NAME, "Q", "Damage_2_Base", r), ranked(NAME, "Q", "Damage_1_Base", r)) \
        + jnp.where(two, ranked(NAME, "Q", "Damage_2_BAD_Ratio", r), ranked(NAME, "Q", "Damage_1_BAD_Ratio", r)) \
        * kctx.bonus_ad
    pct = jnp.where(two, ranked(NAME, "Q", "Damage_2_Percent", r) + Q2_PCT_AD * kctx.bonus_ad,
                    ranked(NAME, "Q", "Damage_1_Percent", r) + Q1_PCT_AD * kctx.bonus_ad)
    hp_part = pct[:, None] * units.max_hp[None, :]
    monster = (units.kind == KIND_MONSTER)[None, :]
    hp_part = jnp.where(monster, jnp.minimum(hp_part, by_level(Q_MONSTER_CAP, kctx.level)[:, None]), hp_part)
    return base[:, None] + hp_part + jnp.where(monster, Q_MONSTER_FLAT, 0.0)


def periodic(state: State, kctx, units) -> tuple[State, KitOut]:
    c, n = kctx.unit.shape[0], units.x.shape[0]
    m = _mine(kctx)
    alive = kctx.alive
    dead = ~alive
    r = kctx.ranks
    now = f32(kctx.now)
    cols = jnp.arange(n)[None, :]
    z = jnp.zeros((c, n), jnp.float32)
    foe = enemies(kctx, units)
    rad = units.radius[None, :]
    rx, ry = units.x[None, :] - kctx.x[:, None], units.y[None, :] - kctx.y[:, None]
    dist = jnp.sqrt(rx * rx + ry * ry)
    dashing_now = jnp.zeros_like(m) if kctx.dashing is None else kctx.dashing

    # Cunning Sweep / Sundering Slam at the end of the lockout, from where Ambessa is then.
    q_due = state.q_pending & (due(kctx, state.q_fire_at) | dead)
    q_fire = q_due & alive
    along = rx * state.q_dx[:, None] + ry * state.q_dy[:, None]
    perp = jnp.abs(ry * state.q_dx[:, None] - rx * state.q_dy[:, None])
    one = q_fire & ~state.q_two
    two = q_fire & state.q_two
    sweep = one[:, None] & foe & (((along + rad >= 0.0) & (dist <= Q1_OUTER + rad)) | (dist <= Q1_NEAR + rad))
    sweep_full = sweep & (dist > Q1_INNER) & (along + rad >= 0.0)
    slam = two[:, None] & foe & (along + rad >= 0.0) & (along - rad <= Q2_LENGTH) & (perp <= Q2_HALF_WIDTH + rad)
    first = jnp.argmin(jnp.where(slam, along, jnp.inf), axis=1)
    slam_full = slam & (cols == first[:, None])
    q_hit = sweep | slam
    q_mult = jnp.where(sweep_full | slam_full, 1.0, Q_MIN_RATIO)
    p_q = D.packets(q_hit, kctx.unit[:, None], cols, _q_raw(kctx, units, state.q_two) * q_mult, D.PHYSICAL,
                    AOE, cast_id=state.q_cast_id[:, None])
    q_two_from = jnp.where(one & jnp.any(sweep, axis=1), f32(now + Q_SWAP_STATIC), state.q_two_from)
    q_two_until = jnp.where(one & jnp.any(sweep, axis=1), f32(now + Q_SWAP), state.q_two_until)

    # Lacerate: the spin at the end of the lockout, again at the end of a Drakehound's Step dash.
    e_due = state.e_pending & (due(kctx, state.e_fire_at) | dead)
    e_fire = e_due & alive
    dash_done = state.dashing & ~dashing_now & (now > state.dash_from + 0.5 * kctx.dt)
    spin2 = dash_done & state.e_second & alive
    spin = e_fire | spin2
    e_hit = spin[:, None] & foe & (dist <= E_RADIUS + rad)
    e_raw = ranked(NAME, "E", "Damage_Flat_Base", r[:, 2]) + ranked(NAME, "E", "ADRatio", r[:, 2]) * kctx.bonus_ad
    e_id = jnp.where(spin2, make_cast_id(kctx, CODE_AMBESSA_SPIN2), state.e_cast_id)
    p_e = D.packets(e_hit, kctx.unit[:, None], cols, e_raw[:, None], D.PHYSICAL, AOE, cast_id=e_id[:, None])
    e_slow_from = jnp.where(e_hit, now, state.e_slow_from)
    # The decaying slow is re-applied every tick at its current strength (the world keeps the strongest slow).
    left = 1.0 - (now - e_slow_from) / E_SLOW_TIME
    slowing = m[:, None] & (left > 0.0) & units.alive[None, :]
    e_slow = jnp.where(slowing, E_SLOW * left, 0.0)
    e_slow_t = jnp.where(slowing, 1.5 * kctx.dt, 0.0)

    # Repudiation's smash at the end of the brace (at the end of a dash if the brace ends during it).
    w_due = m & due(kctx, state.w_smash_at) & ~(state.dashing & dashing_now)
    w_late = m & state.dashing & dash_done & (state.w_smash_at < NEVER / 2) & (now > state.w_smash_at)
    w_fire = (w_due | w_late) & alive
    w_hit = w_fire[:, None] & foe & (dist <= W_RADIUS + rad)
    w_raw = (ranked(NAME, "W", "BaseDamage", r[:, 1]) + ranked(NAME, "W", "DamageADRatio", r[:, 1]) * kctx.bonus_ad) \
        * jnp.where(state.w_blocked, W_HIGH, 1.0)
    p_w = D.packets(w_hit, kctx.unit[:, None], cols, w_raw[:, None], D.PHYSICAL, AOE, cast_id=state.w_cast_id[:, None])
    w_smash_at = jnp.where(w_fire | dead, NEVER, state.w_smash_at)

    # Public Execution: at the end of the cast time, seize the farthest enemy champion in the line.
    r_due = state.r_pending & (due(kctx, state.r_fire_at) | dead)
    r_fire = r_due & alive
    r_along = rx * state.r_dx[:, None] + ry * state.r_dy[:, None]
    r_perp = jnp.abs(ry * state.r_dx[:, None] - rx * state.r_dy[:, None])
    in_line = r_fire[:, None] & foe & (units.kind == KIND_CHAMPION)[None, :] & (r_along + rad >= 0.0) \
        & (r_along - rad <= R_LENGTH) & (r_perp <= R_HALF_WIDTH + rad)
    seized = jnp.any(in_line, axis=1)
    tgt = jnp.argmax(jnp.where(in_line, r_along, -jnp.inf), axis=1)
    tt = jnp.clip(tgt, 0, n - 1)
    gap = units.radius[tt] + 65.0
    bx = units.x[tt] + state.r_dx * gap
    by = units.y[tt] + state.r_dy * gap
    blink = Dash(seized, f32(bx), f32(by), jnp.full((c,), jnp.inf, jnp.float32), jnp.full((c,), -1, jnp.int32),
                 jnp.ones((c,), bool))
    supp_mask = seized[:, None] & (cols == tt[:, None])
    supp, sid = cc_matrix(supp_mask, R_SUPPRESS, state.r_cast_id[:, None])
    r_target = jnp.where(seized, tgt, jnp.where(r_fire, -1, state.r_target)).astype(jnp.int32)
    r_seq = jnp.where(seized, units.spawn_seq[tt], state.r_seq).astype(jnp.int32)
    r_slam_at = jnp.where(seized, f32(now + R_SUPPRESS), jnp.where(r_fire, NEVER, state.r_slam_at))
    # The slam at the end of the suppression; then the lockout ends and the Drakehound's Step window opens.
    st_ = jnp.clip(state.r_target, 0, n - 1)
    slam = m & (state.r_target >= 0) & due(kctx, state.r_slam_at) & alive & units.alive[st_] \
        & (units.spawn_seq[st_] == state.r_seq)
    slam_mask = slam[:, None] & (cols == st_[:, None])
    r_raw = ranked(NAME, "R", "Damage", r[:, 3]) + ranked(NAME, "R", "ADRatio", r[:, 3]) * kctx.bonus_ad
    p_r = D.packets(slam_mask, kctx.unit[:, None], cols, r_raw[:, None], D.PHYSICAL, ABILITY,
                    cast_id=state.r_cast_id[:, None])
    stun2, _ = cc_matrix(slam_mask, R_STUN, state.r_cast_id[:, None])
    r_end = m & (state.r_target >= 0) & due(kctx, state.r_slam_at)
    r_slam_at = jnp.where(r_end, NEVER, r_slam_at)
    r_target = jnp.where(r_end, -1, r_target).astype(jnp.int32)
    state = state._replace(r_target=r_target, r_seq=r_seq, r_slam_at=f32(r_slam_at))
    state = _open_window(state, kctx, r_fire & ~seized, 0.0, P_WINDOW, 3)
    state = _open_window(state, kctx, r_end, 0.0, P_WINDOW, 3)

    # Drakehound's Step: a move or attack command since the window opened dashes once the lockout ends.
    lock_over = now >= state.win_lock_end - 1e-6
    commanded = (kctx.last_order_at if kctx.last_order_at is not None else jnp.full((c,), -NEVER)) \
        >= state.win_from - 1e-6
    held = kctx.stunned | (jnp.zeros_like(m) if kctx.rooted is None else kctx.rooted)
    go = m & alive & lock_over & (now <= state.win_until + 1e-6) & commanded & ~held & ~state.r_pending
    at = jnp.full((c,), -1, jnp.int32) if kctx.attack_order is None else kctx.attack_order
    atc = jnp.clip(at, 0, n - 1)
    goal = jnp.zeros((c, 2)) if kctx.move_goal is None else kctx.move_goal
    gx = jnp.where(at >= 0, units.x[atc], goal[:, 0])
    gy = jnp.where(at >= 0, units.y[atc], goal[:, 1])
    gux, guy, gd = _unit_dir(gx - kctx.x, gy - kctx.y)
    gd = jnp.where(at >= 0, gd - units.radius[atc] - 65.0, gd)
    reach = jnp.clip(gd, P_DASH_MIN, P_DASH_MAX)
    ms = jnp.zeros((c,), jnp.float32) if kctx.move_speed is None else kctx.move_speed
    step = Dash(go, f32(kctx.x + gux * reach), f32(kctx.y + guy * reach), f32(dash_speed(kctx.level, ms)),
                jnp.full((c,), -1, jnp.int32), jnp.zeros((c,), bool))
    e_second = jnp.where(go & (state.win_kind == 2), True, jnp.where(spin2, False, state.e_second))

    # Medarda Maxim stacks arrive when the lockout ends.
    stack_now = m & (now >= state.stack_at - 1e-6)
    stacks = jnp.where(now < state.stacks_until, state.stacks, 0)
    stacks = jnp.where(stack_now, jnp.minimum(stacks + 1, P_STACKS), stacks)
    stacks_until = jnp.where(stack_now, f32(now + P_STACK_TIME), state.stacks_until)

    state = state._replace(
        stacks=stacks.astype(jnp.int32), stacks_until=f32(stacks_until),
        stack_at=jnp.where(stack_now | dead, NEVER, state.stack_at),
        win_until=jnp.where(go | dead, -NEVER, state.win_until),
        dashing=(state.dashing & ~dash_done) | go, dash_from=jnp.where(go, now, state.dash_from),
        q_pending=state.q_pending & ~q_due, q_two_from=f32(q_two_from), q_two_until=f32(q_two_until),
        e_pending=state.e_pending & ~e_due, e_second=e_second, e_slow_from=f32(e_slow_from),
        w_smash_at=f32(w_smash_at), r_pending=state.r_pending & ~r_due)
    cc = CCOut(f32(jnp.maximum(supp, stun2)), z, z, z, f32(e_slow), f32(e_slow_t),
               jnp.where(supp_mask, sid, jnp.where(e_hit, e_id[:, None], 0)).astype(jnp.int32))
    dash = Dash(*(jnp.where(blink.active, b, s_) for s_, b in zip(step, blink)))
    lock = jnp.where(seized, R_SUPPRESS, 0.0)
    return state, out(c, n, packets=D.concat_packets(p_q, p_e, p_w, p_r), cc=cc, dash=dash,
                      dash_stops_at_walls=go & ~seized, base_cooldown=_base_cd(kctx), cast_lockout=f32(lock))


def on_attack(state: State, kctx, units, launch) -> tuple[State, KitOut]:
    """Medarda Maxim: an attack launched with a stack is empowered and consumes it."""
    went = holder_rows(launch.launched, kctx) & _mine(kctx) & (holder_rows(launch.target, kctx) >= 0)
    armed = went & (state.stacks > 0) & (kctx.now < state.stacks_until)
    return state._replace(armed=armed, stacks=jnp.where(armed, state.stacks - 1, state.stacks).astype(jnp.int32)), \
        out(kctx.unit.shape[0], units.x.shape[0], base_cooldown=_base_cd(kctx))


def on_hit(state: State, kctx, units, launch, dodging) -> tuple[State, KitOut]:
    """The empowered attack's bonus physical damage (life steal) and energy refund, if it lands."""
    c, n = kctx.unit.shape[0], units.x.shape[0]
    target = holder_rows(launch.target, kctx)
    t = jnp.clip(target, 0, n - 1)
    hit = holder_rows(launch.launched, kctx) & _mine(kctx) & state.armed & (target >= 0)
    land = hit & ~gather(dodging, target)
    raw = by_level(P_DAMAGE, kctx.level) + P_DAMAGE_AD * kctx.bonus_ad
    p = D.packets(land, kctx.unit, t, raw, D.PHYSICAL, P_FLAGS, cast_id=make_cast_id(kctx, CODE_AMBESSA_P))
    refund = jnp.where(land, -energy_refund(kctx.level), 0.0)
    return state._replace(armed=state.armed & ~hit), out(c, n, packets=p, mana_cost=f32(refund),
                                                          base_cooldown=_base_cd(kctx))


def on_damage(state: State, kctx, units, report) -> tuple[State, KitOut]:
    """Repudiation: the shield absorbing champion / large monster / turret damage during the brace empowers the
    smash. Public Execution (passive): heals a share of post-mitigation active-ability damage (25% vs minions and
    monsters)."""
    c, n = kctx.unit.shape[0], units.x.shape[0]
    m = _mine(kctx)
    p = report.packets
    res = report.resolved
    absorbed = getattr(res, "absorbed", None)
    absorbed = jnp.zeros_like(p.raw) if absorbed is None else absorbed
    final = getattr(res, "final", None)
    final = getattr(res, "health_loss", p.raw) if final is None else final
    src = jnp.clip(p.src, 0, n - 1)
    dst = jnp.clip(p.dst, 0, n - 1)
    heavy = (units.kind[src] == KIND_CHAMPION) | (units.kind[src] == KIND_TURRET) \
        | (_large(units, src) & (units.kind[src] == KIND_MONSTER))
    to_me = p.valid[None, :] & (p.dst[None, :] == kctx.unit[:, None]) & (absorbed > 0)[None, :] & heavy[None, :]
    bracing = m & (kctx.now < state.w_smash_at)
    blocked = bracing & jnp.any(to_me, axis=1)
    rank = kctx.ranks[:, 3]
    ls = jnp.zeros((c,), jnp.float32) if kctx.life_steal is None else kctx.life_steal
    rate = jnp.where(rank > 0, ranked(NAME, "R", "Omnivamp", rank) + R_HEAL_LS * ls, 0.0)
    mine = p.valid[None, :] & (p.src[None, :] == kctx.unit[:, None]) & ((p.flags & D.TAG_ACTIVE_SPELL) != 0)[None, :]
    small = ((units.kind[dst] == KIND_MINION) | (units.kind[dst] == KIND_MONSTER))[None, :]
    heal = jnp.sum(jnp.where(mine, final[None, :] * jnp.where(small, R_HEAL_SMALL, 1.0), 0.0), axis=1) * rate
    return state._replace(w_blocked=state.w_blocked | blocked), out(c, n, heal=f32(jnp.where(m & kctx.alive, heal,
                                                                                               0.0)))


def on_takedown(state: State, kctx, units, kills) -> State:
    return state


def stats(state: State, kctx) -> ItemStats:
    """Public Execution (passive): armor penetration."""
    m = _mine(kctx)
    pen = jnp.where(m & (kctx.ranks[:, 3] > 0), ranked(NAME, "R", "Armor_Penetration", kctx.ranks[:, 3]), 0.0)
    return ItemStats(percent_armor_pen=f32(pen))


def defense(state: State, kctx) -> KitDefense:
    c = kctx.unit.shape[0]
    o = jnp.ones((c,), jnp.float32)
    return KitDefense(o, jnp.zeros((c,), bool), o, jnp.zeros((c,), jnp.float32))


def attack_mods(state: State, kctx) -> KitAttackMods:
    """Medarda Maxim: +75 range, uncancellable windup and the chained attack's short total attack time."""
    m = _mine(kctx)
    c = kctx.unit.shape[0]
    z = jnp.zeros((c,), jnp.float32)
    emp = m & (state.stacks > 0) & (kctx.now < state.stacks_until)
    period = jnp.where(emp, chain_time(kctx.level), 0.0)
    return KitAttackMods(f32(jnp.where(emp, P_RANGE, 0.0)), jnp.zeros((c,), bool),
                         m & (state.q_pending | state.e_pending | state.r_pending), jnp.zeros((c,), bool),
                         f32(period * BASE_WINDUP), f32(period), emp)


def debuffs(state: State, kctx, units) -> Debuffs:
    return neutral_debuffs(units.x.shape[0])


__all__ = ["State", "init", "cast", "periodic", "on_attack", "on_hit", "on_damage", "on_takedown", "stats",
           "defense", "attack_mods", "debuffs", "dash_speed", "chain_time", "energy_refund", "by_level"]
