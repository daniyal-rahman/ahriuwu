"""Ezreal 26.19 kit on the modern packet contract (champions, EZR.* in docs/modern/CHAMPIONS.md). The ranged attack,
the cooldown refund and the blink run through the compiled tick in test_ezreal_world."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.modern import champions as K
from lanerl_jax.modern.champions import ezreal as Z
from lanerl_jax.modern.core import damage as D
from lanerl_jax.modern.core.stat_pipeline import champion_base
from lanerl_jax.modern.core.types import CastOrder
from lanerl_jax.modern.data.champions import cooldowns, spell
from lanerl_jax.modern.tests.test_champions import DT, at, ctx, launch, units, valid

EZ, JAX, MINION_RED, MINION_RED2, TURRET_RED, MINION_BLUE = range(6)
AD, BONUS_AD, AP = 130.0, 30.0, 20.0                    # test_champions.ctx: base 100 + bonus 30, 20 AP


def kctx(**over):
    base = dict(ranks=jnp.asarray([[1, 1, 1, 1], [1, 1, 1, 0]], jnp.int32), level=jnp.full((2,), 6, jnp.int32),
                rooted=jnp.zeros((2,), bool), dashing=jnp.zeros((2,), bool))
    base.update(over)
    return ctx(ids=(81, 24), **base)


def order(slots, x=None, y=None):
    c = len(slots)
    f = lambda v: jnp.asarray([0.0] * c if v is None else v, jnp.float32)    # noqa: E731
    return CastOrder(jnp.asarray(slots, jnp.int32), jnp.full((c,), -1, jnp.int32), f(x), f(y))


def place(**over):
    """Ezreal at the origin; Jax, the minions and the turret off the +x axis unless placed."""
    base = dict(x=[0.0, 600.0, 400.0, 3200.0, 6000.0, -3000.0], y=[0.0, 0.0, 2000.0, 2000.0, 2000.0, 0.0])
    base.update(over)
    return units(**base)


def tick(st, k, u, t, o=None):
    k = at(k, t)
    st, a = K.cast(st, k, u, o if o is not None else order([-1, -1]))
    st, b = K.periodic(st, k, u)
    return st, a, b


def run(st, k, u, t0, seconds):
    outs, t = [], t0
    for _ in range(int(round(seconds / DT))):
        t += DT
        st, _, b = tick(st, k, u, t)
        outs.append((t, b))
    return st, outs, t


def fired(outs):
    return [(t, valid(b.packets), b) for t, b in outs if len(valid(b.packets)["raw"])]


def zs(st):
    return st.ezreal


# ---- data ------------------------------------------------------------------------

def test_record_and_tables():
    b = champion_base(["Ezreal"])
    np.testing.assert_allclose(float(b.attack_range[0]), 550.0)
    np.testing.assert_allclose(float(b.attack_speed[0]), 0.625, rtol=1e-6)
    assert cooldowns("Ezreal", "Q") == [5.5, 5.25, 5.0, 4.75, 4.5]
    assert cooldowns("Ezreal", "W") == [8.0] * 5
    assert cooldowns("Ezreal", "E") == [26.0, 23.0, 20.0, 17.0, 14.0]
    assert cooldowns("Ezreal", "R") == [120.0, 105.0, 90.0]
    assert spell("Ezreal", "Q")["mana"]["values"][:5] == [28.0, 31.0, 34.0, 37.0, 40.0]
    assert (Z.Q_AD, Z.Q_AP, Z.W_AD, Z.W_AP, Z.E_AD, Z.E_AP, Z.R_AD, Z.R_AP) == pytest.approx(
        (1.3, 0.4, 1.0, 0.9, 0.6, 0.75, 1.0, 1.1), rel=1e-6)
    assert (Z.R_MINION_AD, Z.R_MINION_AP, Z.Q_REFUND, Z.W_MANA, Z.E_BOLT_RANGE) == pytest.approx(
        (1.0, 1.1, 1.5, 60.0, 750.0), rel=1e-6)
    assert Z.SKILL_ORDER.count(0) == 5 and Z.SKILL_ORDER.count(3) == 3 and Z.SKILL_ORDER[:3] == (0, 2, 1)


# ---- Mystic Shot ------------------------------------------------------------------

def test_mystic_shot_hits_first_enemy_as_basic_damage_with_on_hit_and_refund():
    k = kctx()
    u = place()
    st, a, _ = tick(K.init(2, 6), k, u, 10.0, order([0, -1], x=[2000.0, 0.0]))
    assert bool(a.cast_started[0]) and int(a.cast_slot[0]) == 0 and bool(a.cooldown_start[0, 0])
    assert float(a.mana_cost[0]) == 28.0 and float(a.base_cooldown[0, 0]) == 5.5
    assert float(a.cast_lockout[0]) == pytest.approx(Z.Q_CAST)
    st, outs, _ = run(st, k, u, 10.0, 1.0)
    hits = fired(outs)
    assert len(hits) == 1
    t, p, b = hits[0]
    assert t - 10.0 == pytest.approx(Z.Q_CAST + (600.0 - 65.0) / Z.Q_SPEED, abs=1.5 * DT)
    assert set(p["dst"].tolist()) == {JAX}
    basic = p["flags"] & D.TAG_BASIC_ATTACK != 0
    np.testing.assert_allclose(p["raw"][basic], [20.0 + 1.3 * AD + 0.4 * AP], rtol=1e-5)
    assert int(p["dtype"][basic][0]) == D.PHYSICAL and int(p["flags"][basic][0]) & D.PROP_LIFESTEAL
    spell_pk = (p["flags"] & D.TAG_ACTIVE_SPELL != 0) & ~basic                  # wiki: + 0 spell damage
    np.testing.assert_allclose(p["raw"][spell_pk], [0.0])
    assert int(b.on_hit_target[0]) == JAX and bool(b.on_attack_proc[0])
    np.testing.assert_allclose(np.asarray(b.cooldown_refund)[0], [1.5] * 4)
    assert int(zs(st).p_stacks[0]) == 1
    np.testing.assert_allclose(float(K.stats(st, at(k, t + 1.0)).attack_speed[0]), 0.1, rtol=1e-5)


@pytest.mark.parametrize("x, hit", [(1210.0, True), (1230.0, False)])
def test_mystic_shot_reaches_1150_plus_the_hitbox(x, hit):
    k = kctx()
    u = place(x=[0.0, x, 400.0, 3200.0, 6000.0, -3000.0])
    st, _, _ = tick(K.init(2, 6), k, u, 10.0, order([0, -1], x=[2000.0, 0.0]))
    st, outs, _ = run(st, k, u, 10.0, 1.2)
    assert (len(fired(outs)) == 1) == hit
    assert not bool(zs(st).q_fly[0])


def test_mystic_shot_stops_on_a_minion_in_front():
    k = kctx()
    u = place(y=[0.0, 0.0, 0.0, 2000.0, 2000.0, 0.0])                       # red minion at 400 on the path
    st, _, _ = tick(K.init(2, 6), k, u, 10.0, order([0, -1], x=[2000.0, 0.0]))
    _, outs, _ = run(st, k, u, 10.0, 1.0)
    hits = fired(outs)
    assert len(hits) == 1 and set(hits[0][1]["dst"].tolist()) == {MINION_RED}


# ---- Essence Flux -----------------------------------------------------------------

def test_essence_flux_passes_minions_marks_the_champion_and_mystic_shot_detonates_it():
    k = kctx()
    u = place(y=[0.0, 0.0, 0.0, 2000.0, 2000.0, 0.0])                       # minion at 400, Jax at 600
    st, a, _ = tick(K.init(2, 6), k, u, 10.0, order([1, -1], x=[2000.0, 0.0]))
    assert float(a.mana_cost[0]) == 50.0 and float(a.base_cooldown[0, 1]) == 8.0
    st, outs, t = run(st, k, u, 10.0, 0.8)
    marks = fired(outs)
    assert len(marks) == 1
    p = marks[0][1]
    assert set(p["dst"].tolist()) == {JAX} and float(p["raw"][0]) == 0.0 and int(p["flags"][0]) == D.TAG_PROC
    assert marks[0][0] - 10.0 == pytest.approx(Z.W_CAST + (600.0 - 65.0) / Z.W_SPEED, abs=1.5 * DT)
    assert bool(zs(st).mark_on[0]) and int(zs(st).mark_unit[0]) == JAX
    # Mystic Shot (the minion moved off the path) detonates the mark: W damage, 60 + 28 mana.
    u = place()
    st, _, _ = tick(st, k, u, t + DT, order([0, -1], x=[2000.0, 0.0]))
    st, outs, _ = run(st, k, u, t + DT, 1.0)
    hits = fired(outs)
    assert len(hits) == 1
    p, b = hits[0][1], hits[0][2]
    det = (p["dtype"] == D.MAGIC)
    np.testing.assert_allclose(p["raw"][det], [80.0 + 1.0 * BONUS_AD + 0.9 * AP], rtol=1e-5)
    np.testing.assert_allclose(float(b.mana_gain[0]), 60.0 + 28.0)
    assert not bool(zs(st).mark_on[0])


def test_essence_flux_marks_a_turret_and_an_attack_detonates_it_without_mana():
    k = kctx()
    u = place(x=[0.0, 600.0, 400.0, 3200.0, 800.0, -3000.0], y=[0.0, 2000.0, 2000.0, 2000.0, 0.0, 0.0])
    st, _, _ = tick(K.init(2, 6), k, u, 10.0, order([1, -1], x=[2000.0, 0.0]))
    st, outs, t = run(st, k, u, 10.0, 0.8)
    assert int(zs(st).mark_unit[0]) == TURRET_RED and bool(zs(st).mark_on[0])
    st, o = K.on_hit(st, at(k, t), u, launch([True, False], [TURRET_RED, -1]))
    p = valid(o.packets)
    np.testing.assert_allclose(p["raw"][p["dst"] == TURRET_RED], [80.0 + BONUS_AD + 0.9 * AP], rtol=1e-5)
    assert o.mana_gain is None or float(o.mana_gain[0]) == 0.0
    assert not bool(zs(st).mark_on[0])


def test_essence_flux_mark_expires_after_four_seconds():
    k = kctx()
    u = place()
    st, _, _ = tick(K.init(2, 6), k, u, 10.0, order([1, -1], x=[2000.0, 0.0]))
    st, _, t = run(st, k, u, 10.0, 0.6)
    st, o = K.on_hit(st, at(k, t + 4.2), u, launch([True, False], [JAX, -1]))
    assert len(valid(o.packets)["raw"]) == 0


# ---- Arcane Shift -----------------------------------------------------------------

@pytest.mark.parametrize("tx, hop", [(2000.0, 475.0), (200.0, 200.0)])
def test_arcane_shift_blinks_up_to_475_at_the_end_of_the_cast(tx, hop):
    k = kctx()
    u = place()
    st, a, _ = tick(K.init(2, 6), k, u, 10.0, order([2, -1], x=[tx, 0.0]))
    assert float(a.mana_cost[0]) == 70.0 and float(a.base_cooldown[0, 2]) == 26.0
    assert not bool(a.dash.active[0])
    st, outs, _ = run(st, k, u, 10.0, 0.3)
    blinks = [(t, b) for t, b in outs if bool(b.dash.active[0])]
    assert len(blinks) == 1 and blinks[0][0] - 10.0 == pytest.approx(Z.E_CAST, abs=1.5 * DT)
    d = blinks[0][1].dash
    assert bool(d.blink[0]) and float(d.to_x[0]) == pytest.approx(hop) and np.isinf(float(d.speed[0]))


def test_arcane_shift_cannot_be_cast_rooted():
    k = kctx(rooted=jnp.asarray([True, False]))
    _, a, _ = tick(K.init(2, 6), k, place(), 10.0, order([2, -1], x=[2000.0, 0.0]))
    assert not bool(a.cast_started[0])


def _blink_then_bolt(st, k, u):
    st, _, _ = tick(st, k, u, 10.0, order([2, -1], x=[2000.0, 0.0]))
    st, _, t = run(st, k, u, 10.0, 7 * DT)                                # the blink tick (cast end)
    assert bool(zs(st).e_aim[0])
    k = k._replace(x=jnp.asarray([475.0, 600.0]))                          # landed
    return run(st, k, u, t, 0.5)


def test_arcane_shift_bolt_strikes_the_nearest_enemy_from_the_landing_point():
    k = kctx()
    u = place(x=[0.0, 600.0, 1100.0, 3200.0, 6000.0, -3000.0], y=[0.0, 0.0, 0.0, 2000.0, 2000.0, 0.0])
    _, outs, _ = _blink_then_bolt(K.init(2, 6), k, u)
    hits = fired(outs)
    assert len(hits) == 1 and set(hits[0][1]["dst"].tolist()) == {JAX}
    np.testing.assert_allclose(hits[0][1]["raw"], [80.0 + 0.6 * BONUS_AD + 0.75 * AP], rtol=1e-5)


def test_arcane_shift_bolt_prefers_the_marked_enemy():
    k = kctx()
    u = place(x=[0.0, 600.0, 1100.0, 3200.0, 6000.0, -3000.0], y=[0.0, 0.0, 0.0, 2000.0, 2000.0, 0.0])
    st = K.init(2, 6)
    st = st._replace(ezreal=zs(st)._replace(mark_on=jnp.asarray([True, False]),
                                            mark_unit=jnp.asarray([MINION_RED, -1], jnp.int32),
                                            mark_seq=jnp.asarray([3, 0], jnp.int32),
                                            mark_until=jnp.asarray([20.0, 0.0], jnp.float32)))
    _, outs, _ = _blink_then_bolt(st, k, u)
    hits = fired(outs)
    dsts = set(np.concatenate([h[1]["dst"] for h in hits]).tolist())
    assert dsts == {MINION_RED}                                             # 625 away, Jax 125
    np.testing.assert_allclose(float(hits[-1][2].mana_gain[0]), 60.0 + 70.0)  # bolt detonated the mark


def test_arcane_shift_bolt_needs_an_enemy_within_750():
    k = kctx()
    u = place(x=[0.0, 1300.0, 1400.0, 3200.0, 6000.0, -3000.0], y=[0.0, 0.0, 0.0, 2000.0, 2000.0, 0.0])
    st, outs, _ = _blink_then_bolt(K.init(2, 6), k, u)
    assert fired(outs) == [] and not bool(zs(st).e_bolt[0])


# ---- Trueshot Barrage -------------------------------------------------------------

def test_trueshot_barrage_strikes_each_enemy_once_with_less_to_minions():
    k = kctx()
    u = place(x=[0.0, 2000.0, 1000.0, 3200.0, 6000.0, -3000.0], y=[0.0, 0.0, 0.0, 2000.0, 2000.0, 0.0])
    st, a, _ = tick(K.init(2, 6), k, u, 10.0, order([3, -1], x=[5000.0, 0.0]))
    assert float(a.cast_lockout[0]) == pytest.approx(Z.R_CAST) and float(a.mana_cost[0]) == 100.0
    assert float(a.base_cooldown[0, 3]) == 120.0
    st, outs, _ = run(st, k, u, 10.0, 3.0)
    hits = fired(outs)
    got = {int(d): (t, float(r)) for t, p, _ in hits for d, r in zip(p["dst"], p["raw"])}
    assert sorted(got) == [JAX, MINION_RED] and sum(len(p["raw"]) for _, p, _ in hits) == 2
    assert got[MINION_RED][1] == pytest.approx(150.0 + BONUS_AD + 1.1 * AP, rel=1e-5)
    assert got[JAX][1] == pytest.approx(350.0 + BONUS_AD + 1.1 * AP, rel=1e-5)
    assert got[JAX][0] - 10.0 == pytest.approx(Z.R_CAST + (2000.0 - 65.0) / Z.R_SPEED, abs=1.5 * DT)
    p = hits[0][1]
    assert int(p["flags"][0]) & D.PROP_ULTIMATE and int(p["dtype"][0]) == D.MAGIC
    assert int(zs(st).p_stacks[0]) == 2


def test_rising_spell_force_caps_at_five_and_expires():
    k = kctx()
    st = K.init(2, 6)
    st = st._replace(ezreal=zs(st)._replace(p_stacks=jnp.asarray([5, 0], jnp.int32),
                                            p_until=jnp.asarray([16.0, 0.0], jnp.float32)))
    np.testing.assert_allclose(float(K.stats(st, at(k, 15.9)).attack_speed[0]), 0.5, rtol=1e-5)
    assert float(K.stats(st, at(k, 16.0)).attack_speed[0]) == 0.0
    np.testing.assert_allclose(float(K.attack_missile_speed(st, k)[0]), 2000.0)
    assert float(K.attack_missile_speed(st, k)[1]) == 0.0
