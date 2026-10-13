"""Ambessa 26.19 kit on the modern packet contract (champions, AMB.* in docs/modern/CHAMPIONS.md)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.modern import champions as K
from lanerl_jax.modern.champions import ambessa as AM
from lanerl_jax.modern.champions import core
from lanerl_jax.modern.core import damage as D
from lanerl_jax.modern.core.stat_pipeline import champion_base
from lanerl_jax.modern.core.types import KIND_CHAMPION, KIND_MINION, KIND_TURRET, CastOrder
from lanerl_jax.modern.data.champions import champion
from lanerl_jax.modern.items.effects.core import Report
from lanerl_jax.modern.tests.test_champions import DT, at, ctx, launch, units, valid

AMB, JAX, MINION_RED, MINION_RED2, TURRET_RED, MINION_BLUE = range(6)


def actx(**over):
    base = dict(ranks=jnp.asarray([[1, 1, 1, 1], [1, 1, 1, 0]], jnp.int32), move_speed=jnp.asarray([340.0, 340.0]),
                dashing=jnp.zeros((2,), bool), move_goal=jnp.zeros((2, 2), jnp.float32), moving=jnp.zeros((2,), bool),
                attack_order=jnp.full((2,), -1, jnp.int32), last_order_at=jnp.full((2,), -1e9, jnp.float32),
                life_steal=jnp.zeros((2,), jnp.float32), mana=jnp.asarray([200.0, 200.0]))
    base.update(over)
    return ctx(ids=(799, 24), **base)


def order(slots, x=None, y=None):
    c = len(slots)
    f = lambda v: jnp.asarray([0.0] * c if v is None else v, jnp.float32)    # noqa: E731
    return CastOrder(jnp.asarray(slots, jnp.int32), jnp.full((c,), -1, jnp.int32), f(x), f(y))


def place(**over):
    base = dict(x=[0.0, 350.0, 3100.0, 3200.0, 6000.0, -3000.0], y=[0.0] * 6)
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


def q1_full(k, max_hp):
    bad = float(k.bonus_ad[0])
    return 40.0 + 0.6 * bad + (0.04 + 0.0003 * bad) * max_hp


# ---- data ----------------------------------------------------------------------

def test_record_energy_and_windup():
    rec = champion("Ambessa")
    assert rec["slots"]["R"].endswith("/AmbessaR")
    r = jnp.asarray([[1, 1, 1, 1], [5, 5, 5, 3]], jnp.int32)
    np.testing.assert_allclose(np.asarray(core.cooldown_row("Ambessa", r)), [[14, 18, 13, 130], [10, 14, 9, 100]])
    np.testing.assert_allclose(np.asarray(core.mana_row("Ambessa", r))[:, :3], 70.0)
    base = champion_base(["Ambessa"])
    np.testing.assert_allclose([float(base.base_mana[0]), float(base.mana_regen[0])], [200.0, 10.0])
    np.testing.assert_allclose(float(base.windup_percent[0]), 0.25 / 1.6, rtol=1e-5)


def test_passive_curves():
    np.testing.assert_allclose([float(AM.dash_speed(l, 340.0)) for l in (1, 6, 11, 16)], [1110, 1170, 1230, 1290])
    np.testing.assert_allclose([float(AM.chain_time(l)) for l in (1, 7, 13)], [0.35, 0.30, 0.25], atol=1e-6)
    np.testing.assert_allclose([float(AM.energy_refund(l)) for l in (1, 7, 13)], [40, 55, 70])


# ---- Q -------------------------------------------------------------------------

def test_cunning_sweep_edge_inner_behind_and_slam_unlock():
    k = actx()
    # Jax 350 in front (edge), a minion 200 in front (inner), a minion 100 behind (inside 135), the blue minion far.
    u = place(x=[0.0, 350.0, 200.0, -100.0, 6000.0, -3000.0])
    st, a, _ = tick(K.init(2, 6), k, u, 10.0, order([0, -1], x=[1000.0, 0.0]))
    assert float(a.cast_lockout[0]) == pytest.approx(0.225) and float(a.mana_cost[0]) == 70.0
    assert bool(a.cooldown_start[0, 0])
    st, outs, t = run(st, k, u, 10.0, 0.3)
    hits = fired(outs)
    assert len(hits) == 1
    p = hits[0][1]
    raw = dict(zip(p["dst"].tolist(), p["raw"].tolist()))
    np.testing.assert_allclose(raw[JAX], q1_full(k, 1600.0), rtol=1e-5)
    np.testing.assert_allclose(raw[MINION_RED], 0.5 * q1_full(k, 477.0), rtol=1e-5)
    np.testing.assert_allclose(raw[MINION_RED2], 0.5 * q1_full(k, 477.0), rtol=1e-5)
    # Sundering Slam: 0.5 s after the sweep, within 4 s; no new cooldown; the first enemy takes full damage.
    k = k._replace(cooldowns=jnp.asarray([[13.0, 0, 0, 0], [0, 0, 0, 0]], jnp.float32))      # Q on cooldown
    st, a, _ = tick(st, k, u, t + 0.1, order([0, -1], x=[1000.0, 0.0]))
    assert not bool(a.cast_started[0])
    u2 = place(x=[0.0, 400.0, 200.0, 3200.0, 6000.0, -3000.0])
    st, a, _ = tick(st, k, u2, t + 0.6, order([0, -1], x=[1000.0, 0.0]))
    assert bool(a.cast_started[0]) and not bool(a.cooldown_start[0, 0])
    st, outs, _ = run(st, k, u2, t + 0.6, 0.3)
    p = fired(outs)[0][1]
    raw = dict(zip(p["dst"].tolist(), p["raw"].tolist()))
    bad = float(k.bonus_ad[0])
    full = lambda hp: 50.0 + 0.9 * bad + (0.04 + 0.0004 * bad) * hp       # noqa: E731
    np.testing.assert_allclose(raw[MINION_RED], full(477.0), rtol=1e-5)
    np.testing.assert_allclose(raw[JAX], 0.5 * full(1600.0), rtol=1e-5)


# ---- passive -------------------------------------------------------------------

def test_drakehounds_step_needs_a_command_and_grants_a_stack():
    k = actx()
    u = place(x=[0.0, 2000.0, 3100.0, 3200.0, 6000.0, -3000.0])
    st, _, _ = tick(K.init(2, 6), k, u, 10.0, order([2, -1]))
    st2, outs, _ = run(st, k, u, 10.0, 0.6)
    assert not any(bool(b.dash.active[0]) for _, b in outs)                # no command: no dash
    assert int(st2.ambessa.stacks[0]) == 1
    cmd = k._replace(last_order_at=jnp.asarray([10.1, -1e9]), move_goal=jnp.asarray([[0.0, 1000.0], [0.0, 0.0]]),
                     moving=jnp.asarray([True, False]))
    st3, outs, _ = run(st, cmd, u, 10.0, 0.4)
    dash = [(t, b) for t, b in outs if bool(b.dash.active[0])]
    assert len(dash) == 1 and dash[0][0] - 10.0 == pytest.approx(0.225, abs=1.5 * DT)
    b = dash[0][1]
    np.testing.assert_allclose([float(b.dash.to_x[0]), float(b.dash.to_y[0])], [0.0, 350.0], atol=1e-3)
    assert float(b.dash.speed[0]) == pytest.approx(830.0 + 340.0) and bool(b.dash_stops_at_walls[0])   # level 9


def test_medarda_maxim_empowered_attack():
    k = actx(level=jnp.full((2,), 7, jnp.int32))
    st = K.init(2, 6)
    st = st._replace(ambessa=st.ambessa._replace(stacks=jnp.asarray([2, 0]), stacks_until=jnp.asarray([20.0, 0.0])))
    mods = K.attack_mods(st, k)
    assert float(mods.extra_range[0]) == 75.0 and bool(mods.uncancellable[0])
    np.testing.assert_allclose([float(mods.period[0]), float(mods.windup[0])], [0.30, 0.30 * 0.15625], rtol=1e-5)
    u = place(x=[0.0, 150.0, 3100.0, 3200.0, 6000.0, -3000.0])
    st, _ = K.on_attack(st, k, u, launch([True, False], [JAX, -1]))
    st, o = K.on_hit(st, k, u, launch([True, False], [JAX, -1]))
    p = valid(o.packets)
    np.testing.assert_allclose(p["raw"], [5.0 + 25.0 * 6 / 17 + 0.25 * 30.0], rtol=1e-5)
    assert p["flags"][0] & D.PROP_LIFESTEAL and float(o.mana_cost[0]) == -55.0
    assert int(st.ambessa.stacks[0]) == 1


# ---- E -------------------------------------------------------------------------

def test_lacerate_spin_decaying_slow_and_second_spin():
    k = actx()
    u = place(x=[0.0, 300.0, 3100.0, 3200.0, 6000.0, -3000.0])
    st, _, _ = tick(K.init(2, 6), k, u, 10.0, order([2, -1]))
    st, outs, t = run(st, k, u, 10.0, 0.3)
    hits = fired(outs)
    assert len(hits) == 1
    np.testing.assert_allclose(hits[0][1]["raw"], [40.0 + 0.5 * 30.0], rtol=1e-5)
    t_hit = hits[0][0]
    st, outs, t = run(st, k, u, t, 0.5)
    slow = [(tt, float(b.cc.slow[0, JAX])) for tt, b in outs]
    tt, sv = slow[-1]
    np.testing.assert_allclose(sv, 0.99 * (1.0 - (tt - t_hit)), atol=1e-4)
    # A Drakehound's Step dash after the spin: a second spin when it ends.
    st, _, _ = tick(K.init(2, 6), k, u, 20.0, order([2, -1]))
    cmd = k._replace(last_order_at=jnp.asarray([20.1, -1e9]), move_goal=jnp.asarray([[0.0, 1000.0], [0.0, 0.0]]))
    st, outs, t = run(st, cmd, u, 20.0, 0.2)                                # the spin, lockout not over
    st, outs, t = run(st, cmd._replace(dashing=jnp.asarray([True, False])), u, t, 0.2)    # the dash starts
    assert bool(st.ambessa.dashing[0])
    st, outs, t = run(st, cmd, u, t, 0.1)                                   # it ends
    hits = fired(outs)
    assert len(hits) == 1 and hits[0][1]["dst"][0] == JAX


# ---- W -------------------------------------------------------------------------

def report(src, dst, absorbed):
    p = D.packets(jnp.asarray([True]), jnp.asarray([src]), jnp.asarray([dst]), jnp.asarray([absorbed]),
                  jnp.asarray([D.PHYSICAL]), jnp.asarray([D.BASIC_ATTACK]), cast_id=jnp.asarray([3]))
    f = jnp.asarray([absorbed], jnp.float32)
    z = jnp.zeros_like(f)
    return Report(p, D.Resolved(f, f, z, z, jnp.zeros((1,), bool), jnp.zeros(6), jnp.zeros(6), None,
                                jnp.zeros(6, bool), jnp.zeros(6), jnp.zeros(6, bool), jnp.int32(0)), None)


@pytest.mark.parametrize("blocked", [False, True])
def test_repudiation_shield_and_empowered_smash(blocked):
    k = actx(level=jnp.full((2,), 1, jnp.int32))
    u = place(x=[0.0, 300.0, 3100.0, 3200.0, 6000.0, -3000.0])
    st, a, _ = tick(K.init(2, 6), k, u, 10.0, order([1, -1]))
    np.testing.assert_allclose(float(jnp.sum(a.shield.amount[0])), 50.0 + 1.5 * 30.0, rtol=1e-5)
    if blocked:
        st, _ = K.on_damage(st, at(k, 10.2), u, report(JAX, AMB, 40.0))
    st, outs, t = run(st, k, u, 10.0, 0.6)
    hits = fired(outs)
    assert len(hits) == 1 and hits[0][0] - 10.0 == pytest.approx(0.5, abs=1.5 * DT)
    np.testing.assert_allclose(hits[0][1]["raw"], [(50.0 + 0.5 * 30.0) * (1.5 if blocked else 1.0)], rtol=1e-5)


# ---- R -------------------------------------------------------------------------

def test_public_execution_seize_suppress_slam():
    k = actx()
    u = place(x=[0.0, 900.0, 300.0, 3200.0, 6000.0, -3000.0])
    st, a, _ = tick(K.init(2, 6), k, u, 10.0, order([3, -1], x=[1000.0, 0.0]))
    assert float(a.cast_lockout[0]) == pytest.approx(0.7) and bool(a.cooldown_start[0, 3])
    st, outs, t = run(st, k, u, 10.0, 1.6)
    blink = [(tt, b) for tt, b in outs if bool(b.dash.active[0]) and bool(b.dash.blink[0])]
    assert len(blink) == 1 and blink[0][0] - 10.0 == pytest.approx(0.7, abs=1.5 * DT)
    assert float(blink[0][1].dash.to_x[0]) == pytest.approx(900.0 + 130.0)
    assert float(blink[0][1].cc.stun[0, JAX]) == pytest.approx(0.75)
    hits = fired(outs)
    assert len(hits) == 1 and hits[0][0] - 10.0 == pytest.approx(0.7 + 0.75, abs=2.5 * DT)
    np.testing.assert_allclose(hits[0][1]["raw"], [150.0 + 0.8 * 30.0], rtol=1e-5)
    assert float(hits[0][2].cc.stun[0, JAX]) == pytest.approx(0.4)


def test_public_execution_passive_heal_and_pen():
    k = actx(life_steal=jnp.asarray([0.1, 0.0]))
    np.testing.assert_allclose(float(K.stats(K.init(2, 6), k).percent_armor_pen[0]), 0.1, rtol=1e-6)
    p = D.packets(jnp.asarray([True, True]), jnp.asarray([AMB, AMB]), jnp.asarray([JAX, MINION_RED]),
                  jnp.asarray([100.0, 100.0]), jnp.asarray([D.PHYSICAL, D.PHYSICAL]),
                  jnp.asarray([D.TAG_ACTIVE_SPELL, D.TAG_ACTIVE_SPELL]), cast_id=jnp.asarray([3, 3]))
    f = jnp.asarray([100.0, 100.0])
    z = jnp.zeros_like(f)
    rep = Report(p, D.Resolved(f, z, z, f, jnp.zeros((2,), bool), jnp.zeros(6), jnp.zeros(6), None,
                               jnp.zeros(6, bool), jnp.zeros(6), jnp.zeros(6, bool), jnp.int32(0)), None)
    _, o = K.on_damage(K.init(2, 6), k, place(), rep)
    np.testing.assert_allclose(float(o.heal[0]), (0.15 + 0.5 * 0.1) * (100.0 + 25.0), rtol=1e-5)
