"""`STRUCT-001` gate: the observation's spell-availability feature agrees with
what ``apply_orders`` actually does with the cast.

The policy is told whether a spell is available by the four cooldown features
of ``self_vec`` (``build_observation``: index ``6 + Slot.X``), which read 1.0
when the spell is unranked or ``cast_locked`` (Q's or E's buff is live) and
``spell_cooldown / base`` otherwise. So "the obs says LOCKED" here means that
feature is ``> 0``, and "AVAILABLE" means it is exactly 0.

The sim's truth is whether ``apply_orders`` with that cast changed the spell
state (the buff table, the cooldowns, R's cast timer, the recall channel).

The invariant this file enforces is one-directional and has no exceptions
except the one ``STRUCT-001`` records:

    obs LOCKED  =>  the sim refuses the cast.

The one standing exception is E during its spin at ``elapsed >= 1.0 s``: the
obs reports E LOCKED for the whole spin (``cast_locked[E]`` is "E's buff is
live"), while ``cast_e`` ACCEPTS a press from ``E_CANCEL_MIN_S`` on as a
CANCEL that ends the spin and starts the cooldown (``SPELL-001``, ``OBS-01``).
Those cases are ``xfail(strict=True)``. **When the buff/spell rewrite lands,
it must resolve that disagreement one way or the other -- and then the xfail
marks on ``E_SPIN_CANCELLABLE`` below MUST be removed** (strict xfail turns an
unexpected pass into a failure precisely so this cannot be forgotten).

The other direction (obs AVAILABLE but the sim refuses) is not a contradiction
in the same sense: silence, death, a recall wind-up, R's own cast lock and R's
range are real refusals that the cooldown feature does not encode (some are in
other features: ``is_dead``, the enemy champion's ``ds/dn``; silence and the
two wind-ups are in none). Those are listed in ``KNOWN_OBS_GAPS`` with the
reason, and pinned: a new "available but refused" case, or a documented one
whose behaviour changes, fails here and must be looked at.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.data.patch import CONTENT_ROOT, load_patch
from lanerl_jax.obs.builder import build_observation
from lanerl_jax.obs.frame import make_lane_frame
from lanerl_jax.sim.init import TOP_OUTER_TURRET, init_lane, lane_params
from lanerl_jax.sim.orders import OrderKind, Orders, apply_orders
from lanerl_jax.sim.spells import (
    E_BUFF_SLOT,
    E_CANCEL_MIN_S,
    E_DURATION_S,
    E_TICK_BUFF_SLOT,
    E_TICK_MS,
    Q_BUFF_DURATION,
    Q_BUFF_SLOT,
    Q_HASTE_BUFF_SLOT,
    R_CAST_RANGE,
    BuffId,
    Slot,
)
from lanerl_jax.sim.state import Team

pytestmark = pytest.mark.skipif(
    not CONTENT_ROOT.exists(), reason="vendored Content tree not available"
)

BLUE_NEXUS = (1131.8, 1426.3)
#: ``self_vec`` index of the Q cooldown feature; W/E/R follow in Slot order.
SELF_CD0 = 6
SELF_IS_DEAD = 14
SELF_RECALLING = 15

KIND_FOR_SLOT = {Slot.Q: OrderKind.CAST_Q, Slot.W: OrderKind.CAST_W,
                 Slot.E: OrderKind.CAST_E, Slot.R: OrderKind.CAST_R}


@functools.lru_cache(maxsize=1)
def _fns():
    patch = load_patch()
    params = lane_params(patch)
    frame = make_lane_frame(TOP_OUTER_TURRET[Team.BLUE],
                            TOP_OUTER_TURRET[Team.RED], BLUE_NEXUS)
    obs = jax.jit(lambda s: build_observation(s, 0, frame, params=params))
    orders = jax.jit(lambda s, o: apply_orders(s, o, params))
    return patch, obs, orders


@functools.lru_cache(maxsize=1)
def _base():
    """The isolated arena; ``_state`` places blue Garen at (6000, 6000) with
    every spell at rank 1 and the red Garen ``enemy_dx`` away (300 is inside
    R's 400 range). No turrets in reach. Cached: every case edits a copy."""
    patch, _, _ = _fns()
    s = init_lane(patch, include_all_turrets=False)
    return s


def _state(*, enemy_dx=300.0, ranks=(1, 1, 1, 1), **edits):
    s = _base()
    kind = np.asarray(s.kind).copy(); team = np.asarray(s.team).copy()
    alive = np.asarray(s.alive).copy(); x = np.asarray(s.x).copy()
    y = np.asarray(s.y).copy()
    x[0], y[0] = 6000.0, 6000.0
    x[1], y[1] = 6000.0 + enemy_dx, 6000.0
    team[0], team[1] = Team.BLUE, Team.RED
    lvl = np.asarray(s.spell_level).copy()
    lvl[0] = ranks
    lvl[1] = (1, 1, 1, 1)
    s = s.replace(x=jnp.asarray(x), y=jnp.asarray(y), kind=jnp.asarray(kind),
                  team=jnp.asarray(team), alive=jnp.asarray(alive),
                  collision_x=jnp.asarray(x), collision_y=jnp.asarray(y),
                  spell_level=jnp.asarray(lvl))
    return edits.get("edit", lambda st: st)(s)


def _with_buff(st, lane, bid, elapsed, duration, power=0.0):
    return st.replace(
        buff_id=st.buff_id.at[0, lane].set(bid),
        buff_elapsed=st.buff_elapsed.at[0, lane].set(elapsed),
        buff_duration=st.buff_duration.at[0, lane].set(duration),
        buff_power=st.buff_power.at[0, lane].set(power))


def _e_spin(elapsed):
    def edit(st):
        st = _with_buff(st, E_BUFF_SLOT, BuffId.GAREN_E, elapsed, E_DURATION_S, 30.0)
        # a mid-spin accumulator value, in ms (lane 6 is the E tick clock)
        return st.replace(buff_elapsed=st.buff_elapsed.at[0, E_TICK_BUFF_SLOT].set(
            E_TICK_MS / 3))
    return edit


def _q_window(st):
    st = _with_buff(st, Q_BUFF_SLOT, BuffId.GAREN_Q, 1.0, Q_BUFF_DURATION, 0.0)
    return _with_buff(st, Q_HASTE_BUFF_SLOT, BuffId.GAREN_Q_HASTE, 1.0, 1.5)


def _cooldown(slot, value):
    return lambda st: st.replace(spell_cooldown=st.spell_cooldown.at[0, slot].set(value))


def _field(name, value, unit=0):
    return lambda st: st.replace(**{name: getattr(st, name).at[unit].set(value)})


# (case id, slot cast, state edit, enemy offset)
CASES = [
    ("E_ready", Slot.E, None, 300.0),
    ("E_on_cooldown", Slot.E, _cooldown(Slot.E, 5.0), 300.0),
    ("E_spin_0.5s", Slot.E, _e_spin(0.5), 300.0),
    ("E_spin_0.98s", Slot.E, _e_spin(E_CANCEL_MIN_S - 1.0 / 60.0), 300.0),
    ("E_spin_1.0s", Slot.E, _e_spin(E_CANCEL_MIN_S), 300.0),
    ("E_spin_2.5s", Slot.E, _e_spin(2.5), 300.0),
    ("E_unranked", Slot.E, None, 300.0),
    ("Q_ready", Slot.Q, None, 300.0),
    ("Q_window_open", Slot.Q, _q_window, 300.0),
    ("Q_on_cooldown", Slot.Q, _cooldown(Slot.Q, 4.0), 300.0),
    ("W_ready", Slot.W, None, 300.0),
    ("W_on_cooldown", Slot.W, _cooldown(Slot.W, 10.0), 300.0),
    ("R_in_range", Slot.R, None, 300.0),
    ("R_out_of_range", Slot.R, None, R_CAST_RANGE + 200.0),
    ("R_on_cooldown", Slot.R, _cooldown(Slot.R, 60.0), 300.0),
    ("R_enemy_dead", Slot.R, _field("alive", False, unit=1), 300.0),
    ("Q_silenced", Slot.Q, _field("silenced_ms", 1000.0), 300.0),
    ("W_silenced", Slot.W, _field("silenced_ms", 1000.0), 300.0),
    ("E_silenced", Slot.E, _field("silenced_ms", 1000.0), 300.0),
    ("R_silenced", Slot.R, _field("silenced_ms", 1000.0), 300.0),
    ("E_dead", Slot.E, _field("alive", False), 300.0),
    ("Q_dead", Slot.Q, _field("alive", False), 300.0),
    ("E_recall_channel", Slot.E, _field("recall_channel_ms", 4000.0), 300.0),
    ("W_recall_channel", Slot.W, _field("recall_channel_ms", 4000.0), 300.0),
    ("E_recall_windup", Slot.E, _field("recall_windup_ms", 300.0), 300.0),
    ("E_during_own_R_cast", Slot.E, _field("r_cast_ms", 200.0), 300.0),
]

#: Obs says E LOCKED for the whole spin; `cast_e` cancels from 1.0 s on.
E_SPIN_CANCELLABLE = {"E_spin_1.0s", "E_spin_2.5s"}
_XFAIL_E_CANCEL = pytest.mark.xfail(
    strict=True,
    reason="STRUCT-001: obs says locked, cast_e cancels at >= 1 s")

#: Obs AVAILABLE (cooldown feature 0) but the sim refuses -- a real refusal
#: the cooldown feature does not encode. Pinned so the set cannot grow
#: silently. Value: where (if anywhere) the policy can see it.
KNOWN_OBS_GAPS = {
    "R_out_of_range": "range is only in the enemy entity's ds/dn",
    "R_enemy_dead": "the dead enemy is simply absent from the entity slots",
    "Q_silenced": "silence is in no feature",
    "W_silenced": "silence is in no feature",
    "E_silenced": "silence is in no feature",
    "R_silenced": "silence is in no feature",
    "E_dead": "self_vec is_dead",
    "Q_dead": "self_vec is_dead",
    "E_recall_windup": "the 0.5 s recall wind-up is in no feature "
                       "(`recalling` is the channel only)",
    "E_during_own_R_cast": "R's 0.435 s cast lock is in no feature",
}


def _evaluate(case_id, slot, edit, enemy_dx):
    _, obs_fn, orders_fn = _fns()
    ranks = (1, 1, 0, 1) if case_id == "E_unranked" else (1, 1, 1, 1)
    st = _state(enemy_dx=enemy_dx, ranks=ranks, edit=edit or (lambda s: s))
    ob = obs_fn(st)
    feature = float(ob.self_vec[SELF_CD0 + slot])
    target = 1 if slot == Slot.R else -1
    o = Orders(kind=jnp.asarray([KIND_FOR_SLOT[slot], OrderKind.NOOP], jnp.int8),
               x=jnp.zeros(2, jnp.float32), y=jnp.zeros(2, jnp.float32),
               target=jnp.asarray([target, -1], jnp.int8))
    after = orders_fn(st, o)
    changed = any(
        not np.array_equal(np.asarray(getattr(st, f)), np.asarray(getattr(after, f)))
        for f in ("buff_id", "buff_elapsed", "buff_duration", "buff_power",
                  "spell_cooldown", "r_cast_ms", "recall_channel_ms"))
    return feature, changed, ob


def _params(filter_fn=lambda c: True, xfail_ids=frozenset()):
    out = []
    for c in CASES:
        if not filter_fn(c):
            continue
        marks = [_XFAIL_E_CANCEL] if c[0] in xfail_ids else []
        out.append(pytest.param(*c, id=c[0], marks=marks))
    return out


@pytest.mark.parametrize("case_id,slot,edit,enemy_dx",
                         _params(xfail_ids=E_SPIN_CANCELLABLE))
def test_a_spell_the_obs_shows_locked_is_never_accepted(case_id, slot, edit, enemy_dx):
    feature, changed, _ = _evaluate(case_id, slot, edit, enemy_dx)
    if feature > 0.0:
        assert not changed, (
            f"{case_id}: obs cooldown feature {feature:.3f} (LOCKED) but "
            f"apply_orders ACCEPTED the cast")
    else:
        # Not a locked case: nothing to check in this direction -- but an
        # xfail case landing here means the obs changed, which the strict
        # xfail will surface as XPASS.
        pass


@pytest.mark.parametrize("case_id,slot,edit,enemy_dx", _params())
def test_a_spell_the_obs_shows_available_is_accepted_or_a_known_gap(
        case_id, slot, edit, enemy_dx):
    feature, changed, ob = _evaluate(case_id, slot, edit, enemy_dx)
    if feature > 0.0:
        return
    if case_id in KNOWN_OBS_GAPS:
        assert not changed, (
            f"{case_id} is documented as refused-while-shown-available "
            f"({KNOWN_OBS_GAPS[case_id]}) but the sim now accepts it: "
            f"update KNOWN_OBS_GAPS")
    else:
        assert changed, (
            f"{case_id}: obs shows the spell AVAILABLE (feature 0) but "
            f"apply_orders refused it, and this is not a documented gap")


def test_the_cases_cover_what_they_claim():
    """The fixture is doing what the case names say (so an agreement is not
    produced by, e.g., the champion being out of the state)."""
    got = {cid: _evaluate(cid, sl, ed, dx) for cid, sl, ed, dx in CASES}
    # every documented gap really is shown available and really is refused
    for cid in KNOWN_OBS_GAPS:
        feature, changed, _ = got[cid]
        assert feature == 0.0 and not changed, (cid, feature, changed)
    # the ready cases are accepted
    for cid in ("E_ready", "Q_ready", "W_ready", "R_in_range",
                "E_recall_channel", "W_recall_channel"):
        feature, changed, _ = got[cid]
        assert feature == 0.0 and changed, (cid, feature, changed)
    # below 1 s the spin is locked on both sides
    for cid in ("E_spin_0.5s", "E_spin_0.98s"):
        feature, changed, _ = got[cid]
        assert feature == 1.0 and not changed, (cid, feature, changed)
    # the side features the gaps point at are really set
    assert float(got["E_dead"][2].self_vec[SELF_IS_DEAD]) == 1.0
    assert float(got["E_recall_channel"][2].self_vec[SELF_RECALLING]) == 1.0
    assert float(got["E_recall_windup"][2].self_vec[SELF_RECALLING]) == 0.0
