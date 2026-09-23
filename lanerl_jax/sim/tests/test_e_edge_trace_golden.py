"""`STRUCT-001` / `SPELL-001` gate: E-cooldown rising edges under a fixed stream.

The instrument that exposed `SPELL-001` was counting LANDED spins off the
E-cooldown rising edge rather than casts issued: sim 0 against server 44 over
300 s of a recorded policy stream. This file runs the analogous count on a
deterministic scripted stream whose answer can be derived by hand from the
constants in ``spells.py``, so it needs no server and no recording.

The stream is ``lanerl_jax/parity/record.py``'s E cadence (its
``if i % 23 == 0`` -> cast E line) with blue Garen otherwise idle (NOOP)
beside a red wave, for 300 s of sim time at 30 decisions/s (2 ticks each).
Garen stays at level 1, so E is rank 1 throughout (asserted).

Expected count, under the server's semantics as ``cast_e`` now implements them
(a re-press at spin ``elapsed < E_CANCEL_MIN_S`` is ignored; at ``>=`` it ENDS
the spin and starts the full rank cooldown; unpressed, the spin expires at
``E_DURATION_S`` and starts the same cooldown)::

    D   = 30 decisions/s                      (60 Hz tick / 2 ticks per decision)
    P   = 23 decisions between presses        (0.767 s)
    m   = ceil(E_CANCEL_MIN_S * D / P) = ceil(30/23) = 2
          -> the cancelling press is 2P = 46 decisions (1.533 s) after the cast,
             which is < E_DURATION_S * D = 90, so EVERY spin is cancelled
    cd  = ceil(E_COOLDOWNS[0] * D) = 390 decisions (13 s)
    cyc = ceil((2P + cd) / P) * P = ceil(436 / 23) * 23 = 19 * 23 = 437
          -> the first press after the cooldown has run out recasts
    N   = 300 s * D = 9000 decisions
    spins landed = #{k >= 0 : k * cyc + 2P < N} = floor((9000 - 1 - 46) / 437) + 1
                 = 20 + 1 = 21

The expected value is COMPUTED below from those constants, not typed in; the
arithmetic above is what the computation does. (The ledger's 44 is a different
stream -- a trained policy's orders, pressing E on 80% of decisions -- and is
not comparable.)

What the pre-fix bugs give on this stream, i.e. what this test catches:
``SPELL-001`` (re-cast resets the spin) -> the first spin would be reset by
every press and never end: 0 edges. The intermediate "always refuse the
re-cast" guard -> every spin runs 3 s: cyc = ceil((90+390)/23)*23 = 483,
floor((9000-1-90)/483)+1 = 19 edges, and every spin 3.0 s long.

Test harness choices, none of which touch the E lifecycle: the red minions are
given 1e6 HP so nothing dies (no XP, so no rank-up), and blue Garen's HP is
refilled before every tick so the wave cannot kill him in 300 s.
"""
from __future__ import annotations

import functools
import math
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.data.patch import CONTENT_ROOT, load_patch
from lanerl_jax.sim.init import init_lane, lane_params
from lanerl_jax.sim.orders import OrderKind, Orders, apply_orders
from lanerl_jax.sim.profiles import profile_id
from lanerl_jax.sim.spells import (E_BUFF_SLOT, E_CANCEL_MIN_S, E_COOLDOWNS,
                                   E_DURATION_S, RANKS_BY_LEVEL, BuffId,
                                   Slot)
from lanerl_jax.sim.state import Kind, Team
from lanerl_jax.sim.step import tick
from lanerl_jax.sim.targeting import MinionType

pytestmark = pytest.mark.skipif(
    not CONTENT_ROOT.exists(), reason="vendored Content tree not available"
)

TICKS_PER_S = 60
STEP_TICKS = 2
DECISIONS_PER_S = TICKS_PER_S // STEP_TICKS
PRESS_EVERY = 23                       # parity/record.py: `if i % 23 == 0`
SIM_SECONDS = 300
N_DECISIONS = SIM_SECONDS * DECISIONS_PER_S


def expected_schedule():
    """The hand arithmetic in the module docstring, from the constants."""
    P, D = PRESS_EVERY, DECISIONS_PER_S
    m = math.ceil(E_CANCEL_MIN_S * D / P)
    cancel_after = m * P
    full = int(round(E_DURATION_S * D))
    spin = min(cancel_after, full)     # cancelled if the press beats expiry
    cd = math.ceil(E_COOLDOWNS[0] * D)
    cycle = math.ceil((spin + cd) / P) * P
    edges = (N_DECISIONS - 1 - spin) // cycle + 1
    return dict(cancel_after=cancel_after, spin_decisions=spin, cd=cd,
                cycle=cycle, edges=edges, cancelled=cancel_after < full)


def _arena(patch):
    """Blue Garen at (6000, 6000); five red melee minions around him inside
    E's reach; red Garen parked far away. No turrets in reach."""
    s = init_lane(patch, include_all_turrets=False)
    kind = np.asarray(s.kind).copy(); team = np.asarray(s.team).copy()
    alive = np.asarray(s.alive).copy(); x = np.asarray(s.x).copy()
    y = np.asarray(s.y).copy(); hp = np.asarray(s.hp).copy()
    mhp = np.asarray(s.max_hp).copy(); model = np.asarray(s.model).copy()
    x[0], y[0] = 6000.0, 6000.0
    x[1], y[1] = 6000.0, 12000.0
    team[0], team[1] = Team.BLUE, Team.RED
    for j, (dx, dy) in enumerate([(150, 0), (120, 90), (120, -90),
                                  (200, 60), (200, -60)]):
        i = 2 + j
        kind[i] = Kind.LANE_MINION
        team[i] = Team.RED
        alive[i] = True
        model[i] = profile_id(Kind.LANE_MINION, MinionType.MELEE, Team.RED)
        x[i], y[i] = 6000.0 + dx, 6000.0 + dy
        hp[i] = mhp[i] = 1.0e6
    present = alive & (kind != Kind.NONE)
    # `init_lane` leaves `spell_level` all zero until the first tick derives it
    # from level, so without this the press on decision 0 is refused as an
    # unlearned E. Write what that first tick would write.
    lvl = np.asarray(s.spell_level).copy()
    lvl[0] = lvl[1] = RANKS_BY_LEVEL[1]
    return s.replace(spell_level=jnp.asarray(lvl),
        kind=jnp.asarray(kind), team=jnp.asarray(team), alive=jnp.asarray(alive),
        x=jnp.asarray(x), y=jnp.asarray(y), hp=jnp.asarray(hp),
        max_hp=jnp.asarray(mhp), model=jnp.asarray(model),
        collision_x=jnp.asarray(x), collision_y=jnp.asarray(y),
        collision_present=jnp.asarray(present),
        target=jnp.asarray(np.full(kind.shape[0], -1, np.int8)))


def _snap(s):
    return dict(e_on=s.buff_id[0, E_BUFF_SLOT] == BuffId.GAREN_E,
                cd_e=s.spell_cooldown[0, Slot.E],
                level=s.level[0], alive=s.alive[0], rank_e=s.spell_level[0, Slot.E])


@functools.lru_cache(maxsize=1)
def _trace():
    """Timeline of snapshots: per decision, [after orders, after tick 1,
    after tick 2]. Flattened to (3 * N_DECISIONS,)."""
    patch = load_patch()
    params = lane_params(patch)
    s0 = _arena(patch)

    def decision(s, i):
        press = (i % PRESS_EVERY) == 0
        o = Orders(kind=jnp.stack([jnp.where(press, OrderKind.CAST_E, OrderKind.NOOP),
                                   jnp.asarray(OrderKind.NOOP)]).astype(jnp.int8),
                   x=jnp.zeros(2, jnp.float32), y=jnp.zeros(2, jnp.float32),
                   target=jnp.asarray([-1, -1], jnp.int8))
        s = apply_orders(s, o, params)
        after_orders = _snap(s)

        def one(st, _):
            st = st.replace(hp=st.hp.at[0].set(st.max_hp[0]))   # harness: no death
            st = tick(st, params)
            return st, _snap(st)
        s, ticks = jax.lax.scan(one, s, None, length=STEP_TICKS)
        return s, (after_orders, ticks, press)

    run = jax.jit(lambda s: jax.lax.scan(decision, s, jnp.arange(N_DECISIONS)))
    t0 = time.time()
    _, (orders_snap, tick_snap, press) = run(s0)
    jax.block_until_ready(press)
    wall = time.time() - t0
    flat = {}
    for k in orders_snap:
        a = np.asarray(orders_snap[k])[:, None]
        b = np.asarray(tick_snap[k])
        flat[k] = np.concatenate([a, b], axis=1).reshape(-1)
    # phase of each snapshot: 0 = after orders, 1..STEP_TICKS = after tick
    phase = np.tile(np.arange(STEP_TICKS + 1), N_DECISIONS)
    decision_of = np.repeat(np.arange(N_DECISIONS), STEP_TICKS + 1)
    return flat, phase, decision_of, np.asarray(press), wall


def _spins(flat, phase):
    """(start_snapshot, end_snapshot, ticks_active) for each E spin.

    ``ticks_active`` counts the ticks that ENTERED with E live, i.e. the
    ``step_buffs`` calls that advanced the spin: a tick snapshot whose
    predecessor snapshot showed E on."""
    on = flat["e_on"]
    spins, start, ticks = [], None, 0
    for t in range(len(on)):
        if t > 0 and on[t - 1] and phase[t] != 0:
            ticks += 1
        if on[t] and (t == 0 or not on[t - 1]):
            start, ticks = t, 0
        if start is not None and not on[t] and on[t - 1]:
            spins.append((start, t, ticks))
            start = None
    return spins


def test_the_hand_arithmetic_is_what_the_docstring_says():
    """Pins the derivation, so a constant change shows up as a changed
    expectation here rather than as a mysterious count change below."""
    e = expected_schedule()
    assert (e["cancel_after"], e["cd"], e["cycle"], e["edges"]) == (46, 390, 437, 21)
    assert e["cancelled"]


def test_e_cooldown_rising_edges_match_the_server_semantics():
    flat, phase, decision_of, press, wall = _trace()
    exp = expected_schedule()
    assert np.all(flat["level"] == 1) and np.all(flat["rank_e"] == 1), \
        "the harness must keep E at rank 1 for the arithmetic to apply"
    assert np.all(flat["alive"])
    cd = flat["cd_e"]
    rises = np.flatnonzero(cd[1:] > cd[:-1] + 1e-6) + 1
    print(f"\n[E-edge golden] {len(rises)} spins landed in {SIM_SECONDS}s "
          f"(expected {exp['edges']}); scan wall {wall:.1f}s")
    assert len(rises) == exp["edges"], (
        f"{len(rises)} E-cooldown rising edges, expected {exp['edges']} "
        f"(first rises at decisions {decision_of[rises[:5]].tolist()})")
    # each rise is at an E press (the cancel), exactly `cancel_after` decisions
    # after a cast, and on the orders phase (cancel is an order, not a tick)
    rise_dec = decision_of[rises]
    assert np.all(phase[rises] == 0)
    assert np.all(press[rise_dec])
    starts = np.arange(exp["edges"]) * exp["cycle"]
    np.testing.assert_array_equal(rise_dec, starts + exp["spin_decisions"])
    # the cooldown written is the rank-1 table value (13 s)
    np.testing.assert_allclose(cd[rises], E_COOLDOWNS[0], atol=1e-6)


def test_every_spin_is_full_length_or_a_cancel_at_the_first_press_after_1s():
    flat, phase, decision_of, press, _ = _trace()
    exp = expected_schedule()
    spins = _spins(flat, phase)
    assert len(spins) >= exp["edges"], spins[:3]
    full_ticks = int(round(E_DURATION_S * TICKS_PER_S))
    cancel_ticks = exp["cancel_after"] * STEP_TICKS
    for start, end, ticks in spins:
        full = abs(ticks - full_ticks) <= 1
        # a cancel: ends on an orders snapshot, at a press, at the FIRST press
        # whose spin elapsed (ticks / 60) is >= E_CANCEL_MIN_S
        cancel = (phase[end] == 0 and press[decision_of[end]]
                  and ticks == cancel_ticks)
        assert full or cancel, (
            f"spin starting at decision {decision_of[start]} lasted {ticks} "
            f"ticks ({ticks / TICKS_PER_S:.3f}s), neither {full_ticks} (full) "
            f"nor {cancel_ticks} (cancel at the first press >= "
            f"{E_CANCEL_MIN_S}s)")
    # and the first press >= 1.0 s is the right one: the previous press fell
    # before E_CANCEL_MIN_S and must have been ignored
    assert (cancel_ticks - PRESS_EVERY * STEP_TICKS) / TICKS_PER_S < E_CANCEL_MIN_S
    assert cancel_ticks / TICKS_PER_S >= E_CANCEL_MIN_S
