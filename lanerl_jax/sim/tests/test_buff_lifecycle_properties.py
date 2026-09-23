"""`STRUCT-001` gate: buff/spell lifecycle properties over RANDOM order streams.

Every bug in ``SPELL-001..007`` survived because each test drove one scripted
path. This file drives the real ``apply_orders`` + ``tick`` with random
orders -- E and Q over-represented so re-presses inside open windows happen
constantly -- plus injected deaths mid-spin, and checks properties that must
hold on EVERY tick regardless of the order stream:

1. an active timed buff never outlives its duration by more than one tick,
   and one E spin never lasts longer than ``E_DURATION_S`` + one tick
   (``SPELL-001``: a re-cast reset the spin forever);
2. at most six E damage ticks per spin (``SPELL-003``: seven);
3. every E/Q buff end writes that spell's cooldown on the same step
   (``SPELL-001``'s "never started its cooldown") -- including a window whose
   owner DIED inside it (``SPELL-006``, fixed by ``STRUCT-001``: the server
   keeps ticking a corpse's buffs and their ``OnDeactivate`` writes the
   cooldown on schedule; the sim used to wipe them on death with none, which
   made death a free spell reset). Those are counted separately and pinned
   by ``test_an_e_or_q_ended_by_death_still_starts_its_cooldown``, which was
   a strict xfail until the fix;
4. every rise of ``spell_cooldown[E]`` pairs with an E end on the same step;
   every rise of ``spell_cooldown[Q]`` with a Q end or cast; ``[W]`` with a W
   cast; ``[R]`` with the R windup completing on the enemy's pending lane;
5. hp <= max_hp; level in [1, 18]; a dead unit holds no target and no swing
   target (``ENT-02``'s ``aa_target``); a dead unit is not mid-swing, not
   even on its death tick (``AA-006``, found by this file);
6. a champion's cs never rises on a tick where no enemy minion died, and its
   gold never rises unless an enemy minion or the enemy champion died (or
   ambient gold has started): ``ENT-01`` paid gold and CS for killing your
   OWN minion, so ATTACK orders here also target allied minions.

The stream is built from ``apply_orders`` followed by ``STEP_TICKS`` ticks per
decision -- exactly ``step_decision``'s body (which does not itself apply
orders) -- written out as a ``lax.scan`` so every intermediate state can be
recorded: per decision, a snapshot after the orders and one after each tick.
Every property is checked on each step between consecutive snapshots.

Harness edits, none inside the buff lifecycle: champions are placed at level 6
by XP so Q/W/E/R are all ranked (``spell_level`` is re-derived from level on
every tick, so a direct write would not survive); both champions' respawn
point is moved to the arena and a dead champion's respawn timer is capped at
1.5 s so deaths keep happening near the fight; a death is injected by setting
hp to a value no regen can lift above zero before the next tick's death check
(the literal "hp = 1" survives whenever nothing happens to hit it that tick);
and every ``REFILL`` decisions a small wave per team is spawned through the
sim's own ``spawn_minion`` so there is something to farm for the whole stream.
"""
from __future__ import annotations

import functools
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.data.patch import CONTENT_ROOT, load_patch
from lanerl_jax.sim.init import init_lane, lane_params, spawn_minion
from lanerl_jax.sim.orders import OrderKind, Orders, apply_orders
from lanerl_jax.sim.profiles import profile_id
from lanerl_jax.sim.rewards import AMBIENT_GOLD_DELAY_MS
from lanerl_jax.sim.spells import (
    E_DURATION_S,
    Q_BUFF_DURATION,
    R_CAST_TIME_S,
    RANKS_BY_LEVEL,
    Slot,
    q_haste_duration_at_rank,
    w_duration_at_rank,
)
from lanerl_jax.sim.state import Kind, Team
from lanerl_jax.sim.step import tick
from lanerl_jax.sim.targeting import MinionType

pytestmark = pytest.mark.skipif(
    not CONTENT_ROOT.exists(), reason="vendored Content tree not available"
)

TICK_S = 1.0 / 60.0
STEP_TICKS = 2
#: 100 s of sim time per seed. E's rank-3 cooldown is 11 s, so a shorter
#: stream lands too few spins to exercise anything (400 decisions = 13 s gave
#: six E casts per seed). One compiled scan serves every seed; ~10 s each.
N_DECISIONS = 3000
SEEDS = (0, 1, 2)
#: Alternating blocks: "spam" presses E/Q constantly (re-press inside open
#: windows); "calm" casts E but never re-presses it during its own spin, so
#: spins also run to their full 3 s (which is where a seventh tick would land).
BLOCK = 150
#: Harness wave: every REFILL decisions, 2 melee + 2 casters per team walk in,
#: so there is something to last-hit for the whole stream.
REFILL = 300
LEVEL = 6
#: Timed buff records: E, W, Q, Q haste, R pending. W-passive is infinite.
TIMED = ("e", "w", "q", "q_haste", "r_pending")
ARENA = ((5900.0, 6000.0), (6250.0, 6000.0))
KINDS = np.array([OrderKind.NOOP, OrderKind.MOVE, OrderKind.ATTACK,
                  OrderKind.CAST_E, OrderKind.CAST_Q, OrderKind.CAST_W,
                  OrderKind.CAST_R], np.int8)
#                 noop  move  attack  E     Q     W     R
P_SPAM = np.array([0.07, 0.12, 0.16, 0.30, 0.20, 0.07, 0.08])
P_CALM = np.array([0.12, 0.23, 0.33, 0.08, 0.12, 0.06, 0.06])
P_DEATH = 0.004      # per decision per champion, only while its E or Q is live
MAX_E_FIRES = 6      # GarenE: 0.0167, 0.533, 1.050, 1.567, 2.083, 2.600 s


def _arena(patch):
    s = init_lane(patch, include_all_turrets=False)
    kind = np.asarray(s.kind).copy(); team = np.asarray(s.team).copy()
    alive = np.asarray(s.alive).copy(); x = np.asarray(s.x).copy()
    y = np.asarray(s.y).copy(); hp = np.asarray(s.hp).copy()
    mhp = np.asarray(s.max_hp).copy(); model = np.asarray(s.model).copy()
    team[0], team[1] = Team.BLUE, Team.RED
    for c in (0, 1):
        x[c], y[c] = ARENA[c]
    minions = [(Team.BLUE, MinionType.MELEE, 5800, 5950), (Team.BLUE, MinionType.MELEE, 5820, 6080),
               (Team.BLUE, MinionType.CASTER, 5650, 6000), (Team.BLUE, MinionType.CASTER, 5680, 6120),
               (Team.RED, MinionType.MELEE, 6350, 5950), (Team.RED, MinionType.MELEE, 6330, 6070),
               (Team.RED, MinionType.CASTER, 6500, 6000), (Team.RED, MinionType.CASTER, 6480, 5880)]
    for j, (t, mt, mx, my) in enumerate(minions):
        i = 2 + j
        kind[i] = Kind.LANE_MINION
        team[i] = t
        alive[i] = True
        model[i] = profile_id(Kind.LANE_MINION, mt, t)
        x[i], y[i] = mx, my
        hp[i] = mhp[i] = 455.0 if mt == MinionType.MELEE else 290.0
    present = alive & (kind != Kind.NONE)
    lvl = np.asarray(s.spell_level).copy()
    lvl[0] = lvl[1] = RANKS_BY_LEVEL[LEVEL]
    xp = np.asarray(s.xp).copy()
    xp[0] = xp[1] = float(patch.xp_for_level(LEVEL)) + 1.0
    sx = np.asarray(s.spawn_x).copy(); sy = np.asarray(s.spawn_y).copy()
    sx[:2], sy[:2] = x[:2], y[:2]
    return s.replace(
        kind=jnp.asarray(kind), team=jnp.asarray(team), alive=jnp.asarray(alive),
        x=jnp.asarray(x), y=jnp.asarray(y), hp=jnp.asarray(hp),
        max_hp=jnp.asarray(mhp), model=jnp.asarray(model),
        collision_x=jnp.asarray(x), collision_y=jnp.asarray(y),
        collision_present=jnp.asarray(present),
        target=jnp.asarray(np.full(kind.shape[0], -1, np.int8)),
        spell_level=jnp.asarray(lvl), xp=jnp.asarray(xp, s.xp.dtype),
        spawn_x=jnp.asarray(sx, s.x.dtype), spawn_y=jnp.asarray(sy, s.y.dtype))


_WAVE = [(Team.BLUE, MinionType.MELEE, 5600.0, 5960.0),
         (Team.BLUE, MinionType.MELEE, 5600.0, 6060.0),
         (Team.BLUE, MinionType.CASTER, 5450.0, 5960.0),
         (Team.BLUE, MinionType.CASTER, 5450.0, 6060.0),
         (Team.RED, MinionType.MELEE, 6600.0, 5960.0),
         (Team.RED, MinionType.MELEE, 6600.0, 6060.0),
         (Team.RED, MinionType.CASTER, 6750.0, 5960.0),
         (Team.RED, MinionType.CASTER, 6750.0, 6060.0)]


def _refill(s, enabled):
    """Spawn a small wave per team just behind the fight through the sim's
    own ``spawn_minion`` (lowest free slot, every per-unit field reset)."""
    dummy_path = jnp.zeros((2, 2), s.x.dtype)
    for t, mt, x, y in _WAVE:
        s = spawn_minion(s, t, profile_id(Kind.LANE_MINION, mt, t),
                         455.0 if mt == MinionType.MELEE else 290.0,
                         dummy_path, enabled=enabled, spawn_xy=(x, y))
    return s


def _random_orders(s, key, spam):
    """One random order per champion, resolved against the live state."""
    n = s.kind.shape[0]
    idx = jnp.arange(n)
    probs = jnp.where(spam, jnp.asarray(P_SPAM), jnp.asarray(P_CALM))
    kinds, xs, ys, tgts = [], [], [], []
    for c in (0, 1):
        k = jax.random.fold_in(key, c)
        kk, kx, ky, kt, ka = jax.random.split(k, 5)
        kind = jnp.asarray(KINDS)[jax.random.choice(kk, len(KINDS), p=probs)]
        mx = s.x[c] + jax.random.uniform(kx, minval=-350.0, maxval=350.0)
        my = s.y[c] + jax.random.uniform(ky, minval=-350.0, maxval=350.0)
        # ATTACK: a random live unit, 80% a visible enemy, 20% an ALLIED
        # minion (ENT-01: the server holds an ally target and never swings).
        live = s.alive & (idx != c) & ((s.kind == Kind.CHAMPION) | (s.kind == Kind.LANE_MINION))
        enemy = live & (s.team != s.team[c]) & s.visible_to_enemy
        ally = live & (s.team == s.team[c]) & (s.kind == Kind.LANE_MINION)
        pick_ally = jax.random.uniform(ka) < 0.2
        elig = jnp.where(pick_ally, ally, enemy)
        score = jnp.where(elig, jax.random.uniform(kt, (n,)), -1.0)
        atk = jnp.where(jnp.any(elig), jnp.argmax(score), -1)
        # calm: no E re-press while this champion's own spin is live
        e_live = s.buffs.e.active[c]
        kind = jnp.where(~spam & e_live & (kind == OrderKind.CAST_E),
                         jnp.asarray(OrderKind.NOOP, kind.dtype), kind)
        tgt = jnp.where(kind == OrderKind.ATTACK, atk,
                        jnp.where(kind == OrderKind.CAST_R, 1 - c, -1))
        kinds.append(kind); xs.append(mx); ys.append(my); tgts.append(tgt)
    return Orders(kind=jnp.stack(kinds).astype(jnp.int8),
                  x=jnp.stack(xs).astype(jnp.float32),
                  y=jnp.stack(ys).astype(jnp.float32),
                  target=jnp.stack(tgts).astype(jnp.int8))


def _durations(b):
    """Each timed buff's duration, seconds, from its constant or its
    cast rank -- durations are not state (`STRUCT-001`)."""
    return {
        "e": jnp.full(b.e.active.shape, E_DURATION_S),
        "q": jnp.full(b.q.active.shape, Q_BUFF_DURATION),
        "w": w_duration_at_rank(b.w.rank),
        "q_haste": q_haste_duration_at_rank(b.q_haste.rank),
        "r_pending": jnp.full(b.r_pending.active.shape, R_CAST_TIME_S),
    }


def _snap(s):
    b = s.buffs
    dur = _durations(b)
    return dict(
        on={k: getattr(b, k).active[:2] for k in TIMED},
        el={k: getattr(b, k).elapsed_s[:2] for k in TIMED},
        dur={k: dur[k][:2] for k in TIMED},
        e_tick_acc_ms=b.e.tick_acc_ms[:2],
        cd=s.spell_cooldown[:2], hp=s.hp, max_hp=s.max_hp, alive=s.alive,
        kind=s.kind, team=s.team, cs=s.cs[:2], gold=s.gold[:2], level=s.level[:2],
        target=s.target, is_attacking=s.is_attacking, aa_target=s.aa_target,
        t_ms=s.t_ms)


@functools.lru_cache(maxsize=1)
def _runner():
    patch = load_patch()
    params = lane_params(patch)
    s0 = _arena(patch)

    def decision(carry, i):
        s, key = carry
        kd = jax.random.fold_in(key, i)
        k_ord, k_die = jax.random.split(kd)
        spam = (i // BLOCK) % 2 == 0
        o = _random_orders(s, k_ord, spam)
        s = apply_orders(s, o, params)
        snap_orders = _snap(s)
        # death injection: only while E or Q is live, so it lands mid-window
        live_window = s.buffs.e.active[:2] | s.buffs.q.active[:2]
        inject = (jax.random.uniform(k_die, (2,)) < P_DEATH) & live_window & s.alive[:2]
        s = s.replace(hp=s.hp.at[:2].set(jnp.where(inject, -1000.0, s.hp[:2])))

        def one(st, _):
            st = tick(st, params)
            # harness: bring a dead champion back within 1.5 s
            rt = st.respawn_ms[:2]
            st = st.replace(respawn_ms=st.respawn_ms.at[:2].set(
                jnp.where(rt > 1500.0, 1500.0, rt)))
            return st, _snap(st)
        s, snaps = jax.lax.scan(one, s, None, length=STEP_TICKS)
        s = _refill(s, (i % REFILL == REFILL - 1))
        return (s, key), (snap_orders, snaps, o.kind, inject)

    @jax.jit
    def run(seed):
        key = jax.random.PRNGKey(seed)
        _, out = jax.lax.scan(decision, (s0, key), jnp.arange(N_DECISIONS))
        return out
    return run


@functools.lru_cache(maxsize=None)
def _trace(seed):
    run = _runner()
    t0 = time.time()
    snap_orders, snaps, kinds, inject = jax.block_until_ready(run(seed))
    wall = time.time() - t0
    def cat(a, b):
        a = np.asarray(a)[:, None]
        b = np.asarray(b)
        return np.concatenate([a, b], axis=1).reshape((-1,) + a.shape[2:])
    flat = {}
    for k in snap_orders:
        if isinstance(snap_orders[k], dict):
            flat[k] = {r: cat(snap_orders[k][r], snaps[k][r]) for r in snap_orders[k]}
        else:
            flat[k] = cat(snap_orders[k], snaps[k])
    phase = np.tile(np.arange(STEP_TICKS + 1), N_DECISIONS)
    return flat, phase, np.asarray(kinds), np.asarray(inject), wall


def _analyse(seed):
    """Walk the timeline and collect violations and event counts."""
    f, phase, kinds, inject, wall = _trace(seed)
    T = len(phase)
    bad = {k: [] for k in ("timed", "spin_len", "fires", "end_no_cd", "cd_rise",
                           "hp", "level", "dead", "dead_stale", "dead_swing",
                           "cs", "gold")}
    ev = dict(e_casts=0, e_cancels=0, e_expiries=0, e_death_ends=0,
              q_casts=0, q_ends=0, q_death_ends=0, w_casts=0, r_landed=0,
              deaths=0, injected=int(inject.sum()), fires=0, cs_gain=0,
              e_death_ends_with_cd=0, q_death_ends_with_cd=0, dead_swinging=0,
              corpse_fires=0)
    tol = 1e-4
    on, el, dur, cd = f["on"], f["el"], f["dur"], f["cd"]
    alive, kind, team = f["alive"], f["kind"], f["team"]
    for c in (0, 1):
        spin_ticks, spin_fires = 0, 0
        # `SPELL-006`: a window whose owner died while it was live. Buffs are
        # not removed by death any more, so such a window ENDS later (on the
        # corpse, or after respawn) -- and that end must write the cooldown
        # like any other.
        e_saw_death = q_saw_death = False
        for t in range(T):
            where = f"seed {seed} snapshot {t} (decision {t // (STEP_TICKS + 1)}, phase {phase[t]}) champ {c}"
            # --- 1. timed buffs never overrun
            for rec in TIMED:
                if on[rec][t, c] and \
                        el[rec][t, c] > dur[rec][t, c] + TICK_S + tol:
                    bad["timed"].append(f"{where}: {rec} elapsed "
                                        f"{el[rec][t, c]:.4f} > duration {dur[rec][t, c]:.4f}")
            # --- 5. per-snapshot sanity
            if not (1 <= f["level"][t, c] <= 18):
                bad["level"].append(f"{where}: level {f['level'][t, c]}")
            if t == 0:
                continue
            p = t - 1
            e_prev, e_now = on["e"][p, c], on["e"][t, c]
            q_prev, q_now = on["q"][p, c], on["q"][t, c]
            died = alive[p, c] and not alive[t, c]
            ev["deaths"] += int(died)
            rise = cd[t, c] > cd[p, c] + 1e-6
            is_tick = phase[t] != 0
            if died and e_prev:
                e_saw_death = True
            if died and q_prev:
                q_saw_death = True
            # --- 1b/2. spin length and damage ticks (a tick that ENTERED
            # live, alive OR dead: a corpse's spin keeps running, SPELL-006)
            if e_now and not e_prev:
                spin_ticks, spin_fires = 0, 0
                ev["e_casts"] += 1
            if is_tick and e_prev:
                spin_ticks += 1
                if f["e_tick_acc_ms"][t, c] == 0.0:
                    spin_fires += 1
                    ev["fires"] += 1
                    ev["corpse_fires"] += int(not alive[p, c])
                if spin_ticks > round(E_DURATION_S / TICK_S) + 1:
                    bad["spin_len"].append(f"{where}: spin live for {spin_ticks} ticks")
                if spin_fires > MAX_E_FIRES:
                    bad["fires"].append(f"{where}: {spin_fires} E damage ticks in one spin")
            # --- 3. every E/Q end writes its cooldown -- a window its owner
            # died in included (counted separately: SPELL-006)
            if e_prev and not e_now:
                if e_saw_death:
                    ev["e_death_ends"] += 1
                    ev["e_death_ends_with_cd"] += int(rise[Slot.E])
                elif is_tick:
                    ev["e_expiries"] += 1
                else:
                    ev["e_cancels"] += 1
                if not rise[Slot.E]:
                    bad["end_no_cd"].append(f"{where}: E ended without a cooldown "
                                            f"({cd[p, c, Slot.E]:.3f} -> {cd[t, c, Slot.E]:.3f})"
                                            + (" [owner died in the spin]" if e_saw_death else ""))
                e_saw_death = False
            if q_prev and not q_now:
                if q_saw_death:
                    ev["q_death_ends"] += 1
                    ev["q_death_ends_with_cd"] += int(rise[Slot.Q])
                else:
                    ev["q_ends"] += 1
                if not rise[Slot.Q]:
                    bad["end_no_cd"].append(f"{where}: Q ended without a cooldown"
                                            + (" [owner died in the window]" if q_saw_death else ""))
                q_saw_death = False
            if q_now and not q_prev:
                ev["q_casts"] += 1
            # --- 4. every cooldown rise has its cause on the same step
            if rise[Slot.E] and not (e_prev and not e_now):
                bad["cd_rise"].append(f"{where}: cd[E] rose with no E end")
            if rise[Slot.Q] and not ((q_prev and not q_now) or (q_now and not q_prev)):
                bad["cd_rise"].append(f"{where}: cd[Q] rose with no Q end or cast")
            w_started = (on["w"][t, c]
                         and (not on["w"][p, c]
                              or el["w"][t, c] < el["w"][p, c]))
            ev["w_casts"] += int(w_started)
            if rise[Slot.W] and not w_started:
                bad["cd_rise"].append(f"{where}: cd[W] rose with no W cast")
            other = 1 - c
            r_fired = on["r_pending"][p, other] and not on["r_pending"][t, other]
            if rise[Slot.R]:
                ev["r_landed"] += 1
                if not r_fired:
                    bad["cd_rise"].append(f"{where}: cd[R] rose with no R windup ending")
            # --- 6. cs / gold only from kills
            if not is_tick:
                if f["cs"][t, c] != f["cs"][p, c] or f["gold"][t, c] != f["gold"][p, c]:
                    bad["gold"].append(f"{where}: cs/gold changed in the orders phase")
                continue
            minion_died = (kind == Kind.LANE_MINION) & alive[p] & ~alive[t] & (team[p] != team[p, c])
            enemy_minion_died = bool(minion_died.any())
            enemy_champ_died = bool(alive[p, other] and not alive[t, other])
            if f["cs"][t, c] > f["cs"][p, c]:
                ev["cs_gain"] += int(f["cs"][t, c] - f["cs"][p, c])
                if not enemy_minion_died:
                    bad["cs"].append(f"{where}: cs +{f['cs'][t, c] - f['cs'][p, c]} "
                                     f"with no enemy minion death")
            ambient = f["t_ms"][t] >= AMBIENT_GOLD_DELAY_MS
            if f["gold"][t, c] > f["gold"][p, c] + 1e-4 and not (
                    enemy_minion_died or enemy_champ_died or ambient):
                bad["gold"].append(f"{where}: gold +{f['gold'][t, c] - f['gold'][p, c]:.2f} "
                                   f"with no enemy death")
    # --- 5. all units
    present = kind != Kind.NONE
    over = present & (f["hp"] > f["max_hp"] + 1e-3)
    for t, u in zip(*np.nonzero(over)):
        bad["hp"].append(f"seed {seed} snapshot {t}: unit {u} hp {f['hp'][t, u]:.2f} "
                         f"> max {f['max_hp'][t, u]:.2f}")
    dead = present & ~alive
    held = dead & ((f["target"] != -1) | (f["aa_target"] != -1))
    for t, u in zip(*np.nonzero(held)):
        bad["dead"].append(f"seed {seed} snapshot {t} phase {phase[t]}: dead unit {u} "
                           f"target {f['target'][t, u]} aa_target {f['aa_target'][t, u]}")
    # `is_attacking` on a corpse, split out: see
    # test_a_dead_unit_is_not_mid_swing. `fresh` = the unit died on the last
    # TICK step (the orders phase that follows a death tick shows the same
    # state, so it counts as fresh too).
    swinging = dead & f["is_attacking"]
    last_tick = np.maximum.accumulate(np.where(phase != 0, np.arange(T), 0))
    for t, u in zip(*np.nonzero(swinging)):
        tt = t if phase[t] != 0 else last_tick[t]
        fresh = tt > 0 and alive[tt - 1, u]
        ev.setdefault("dead_swinging", 0)
        ev["dead_swinging"] += 1
        if not fresh:
            bad["dead_stale"].append(
                f"seed {seed} snapshot {t} phase {phase[t]}: unit {u} dead for more "
                f"than one tick and still is_attacking")
        else:
            bad["dead_swing"].append(
                f"seed {seed} snapshot {t} phase {phase[t]}: unit {u} "
                f"(kind {kind[t, u]}) died this tick with is_attacking=True")
    return bad, ev, wall


@functools.lru_cache(maxsize=None)
def _analysis(seed):
    return _analyse(seed)


def _report(bad, key, n=8):
    rows = bad[key]
    return f"{len(rows)} violations, first {min(n, len(rows))}:\n  " + "\n  ".join(rows[:n])


@pytest.mark.parametrize("seed", SEEDS)
def test_timed_buffs_never_outlive_their_duration(seed):
    bad, _, _ = _analysis(seed)
    assert not bad["timed"], _report(bad, "timed")
    assert not bad["spin_len"], _report(bad, "spin_len")


@pytest.mark.parametrize("seed", SEEDS)
def test_at_most_six_e_damage_ticks_per_spin(seed):
    bad, _, _ = _analysis(seed)
    assert not bad["fires"], _report(bad, "fires")


@pytest.mark.parametrize("seed", SEEDS)
def test_every_e_or_q_end_writes_its_cooldown(seed):
    """Property 3: expiry, cancel, Q consumed on hit -- and a window whose
    owner died inside it, whose end is ALSO counted and pinned separately
    below (`SPELL-006`)."""
    bad, _, _ = _analysis(seed)
    assert not bad["end_no_cd"], _report(bad, "end_no_cd")


@pytest.mark.parametrize("seed", SEEDS)
def test_every_cooldown_rise_has_its_cause_on_the_same_step(seed):
    bad, _, _ = _analysis(seed)
    assert not bad["cd_rise"], _report(bad, "cd_rise")


@pytest.mark.parametrize("seed", SEEDS)
def test_unit_sanity_on_every_snapshot(seed):
    """hp <= max_hp, level in [1, 18], a dead unit holds no target and no
    swing target, and a corpse's `is_attacking` never outlives its death tick
    (the death-tick case itself is `test_a_dead_unit_is_not_mid_swing`, AA-006)."""
    bad, _, _ = _analysis(seed)
    for k in ("hp", "level", "dead", "dead_stale"):
        assert not bad[k], _report(bad, k)


def test_a_dead_unit_is_not_mid_swing():
    rows = []
    for seed in SEEDS:
        bad, _, _ = _analysis(seed)
        rows += bad["dead_swing"]
    assert not rows, f"{len(rows)} deaths mid-swing left is_attacking=True:\n  " + \
        "\n  ".join(rows[:8])


@pytest.mark.parametrize("seed", SEEDS)
def test_cs_and_gold_only_rise_on_a_kill(seed):
    bad, _, _ = _analysis(seed)
    assert not bad["cs"], _report(bad, "cs")
    assert not bad["gold"], _report(bad, "gold")


def _totals():
    tot = {}
    for seed in SEEDS:
        _, ev, _ = _analysis(seed)
        for k, v in ev.items():
            tot[k] = tot.get(k, 0) + v
    return tot


def test_the_random_streams_exercise_every_path():
    """Coverage of the properties above: without these, a clean run could be
    a stream that never cancelled a spin, never let one expire, never died
    mid-window, etc. Printed with -s."""
    tot = _totals()
    walls = [(_analysis(s)[2]) for s in SEEDS]
    print(f"\n[lifecycle] events over {len(SEEDS)} seeds x {N_DECISIONS} decisions: {tot}")
    print(f"[lifecycle] scan wall per seed (s): {[round(w, 1) for w in walls]}")
    assert tot["e_casts"] >= 20
    assert tot["e_cancels"] >= 10, "re-press at >= 1 s never cancelled a spin"
    assert tot["e_expiries"] >= 6, "too few spins ran to their natural 3 s end"
    assert tot["fires"] >= 50
    assert tot["q_casts"] >= 10 and tot["q_ends"] >= 5
    assert tot["w_casts"] >= 2 and tot["r_landed"] >= 1
    assert tot["cs_gain"] >= 1
    assert tot["e_death_ends"] + tot["q_death_ends"] >= 2, \
        "no death landed inside an E or Q window: SPELL-006 is untested"


def test_an_e_or_q_ended_by_death_still_starts_its_cooldown():
    """`SPELL-006`, fixed. The server lets the buffs run out on the corpse
    and ``OnDeactivate`` sets the full cooldown; the sim used to wipe them on
    death with no cooldown (a free spell reset). A strict xfail until the
    fix. Requires the stream to have produced such deaths (asserted first, so
    a pass cannot come from zero samples)."""
    tot = _totals()
    n = tot["e_death_ends"] + tot["q_death_ends"]
    assert n >= 2, "coverage lost -- see test_the_random_streams_exercise_every_path"
    with_cd = tot["e_death_ends_with_cd"] + tot["q_death_ends_with_cd"]
    assert with_cd == n, f"{n - with_cd} of {n} E/Q ends by death wrote no cooldown"
