"""Yone 26.19 kit on the modern packet contract (champions, YON.* in docs/modern/CHAMPIONS.md)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.modern import champions as K
from lanerl_jax.modern.champions import core
from lanerl_jax.modern.champions import yone as YO
from lanerl_jax.modern.core import damage as D
from lanerl_jax.modern.core.stat_pipeline import champion_base
from lanerl_jax.modern.core.types import KIND_CHAMPION, KIND_MINION, CastOrder
from lanerl_jax.modern.data.champions import champion
from lanerl_jax.modern.items.effects.core import Report
from lanerl_jax.modern.tests.test_champions import DT, at, ctx, launch, units, valid

YON, JAX, MINION_RED, MINION_RED2, TURRET_RED, MINION_BLUE = range(6)


def yctx(**over):
    base = dict(rand=jnp.asarray([0.5, 0.5], jnp.float32), rand_n=jnp.ones((2, 6), jnp.float32),
                dashing=jnp.zeros((2,), bool), crit_damage=jnp.asarray([2.0, 2.0], jnp.float32),
                bonus_attack_speed=jnp.zeros((2,), jnp.float32))
    base.update(over)
    return ctx(ids=(777, 24), **base)


def order(slots, x=None, y=None):
    c = len(slots)
    f = lambda v: jnp.asarray([0.0] * c if v is None else v, jnp.float32)    # noqa: E731
    return CastOrder(jnp.asarray(slots, jnp.int32), jnp.full((c,), -1, jnp.int32), f(x), f(y))


def place(**over):
    """Yone at the origin, Jax 300 east; minions and the turret far away unless placed."""
    base = dict(x=[0.0, 300.0, 3100.0, 3200.0, 6000.0, -3000.0], y=[0.0] * 6)
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


# ---- data, passive -------------------------------------------------------------

def test_record_tables_no_mana_and_windup():
    rec = champion("Yone")
    assert rec["slots"]["E"].endswith("/YoneE")
    r = jnp.asarray([[1, 1, 1, 1], [5, 5, 5, 3]], jnp.int32)
    np.testing.assert_allclose(np.asarray(core.cooldown_row("Yone", r))[:, 2:], [[22, 120], [10, 80]])
    base = champion_base(["Yone"])
    assert float(base.base_mana[0]) == 0.0                              # Flow is not mana
    np.testing.assert_allclose(float(base.windup_percent[0]), 0.22, rtol=1e-5)


@pytest.mark.parametrize("crit,cc_add,ad", [(0.0, 0.0, 0.0), (0.25, 0.25, 0.0), (0.6, 0.6, 10.0), (1.0, 1.0, 50.0)])
def test_intent_crit_doubling_and_excess_ad(crit, cc_add, ad):
    k = yctx(crit_chance=jnp.asarray([crit, 0.0], jnp.float32))
    s = K.stats(K.init(2, 6), k)
    np.testing.assert_allclose(float(s.crit_chance[0]), cc_add, atol=1e-6)
    np.testing.assert_allclose(float(s.attack_damage[0]), ad, atol=1e-4)
    np.testing.assert_allclose(float(s.crit_damage[0]), -0.05 * 2.0, rtol=1e-5)    # 200% -> 190%


def test_as_scaling_and_haste_exemption():
    np.testing.assert_allclose(float(YO.as_scaled(4.0, 1.05, YO.Q_CD_PER_AS, YO.Q_CD_MAX)), 1.48, atol=0.01)
    np.testing.assert_allclose(float(YO.as_scaled(14.0, 2.0, YO.W_CD_PER_AS, YO.W_CD_MAX)), 6.0, atol=0.01)
    np.testing.assert_allclose(float(YO.as_scaled(0.35, 2.0, YO.Q_CAST_PER_AS, YO.Q_CAST_MAX)), 0.175, atol=1e-4)
    k = yctx(ability_haste=jnp.asarray([50.0, 0.0]), bonus_attack_speed=jnp.asarray([0.5, 0.0]))
    _, a, _ = tick(K.init(2, 6), k, place(), 10.0)
    np.testing.assert_allclose(float(a.base_cooldown[0, 0]), 4.0 * 0.7 * 1.5, rtol=1e-5)  # the world's haste cancels


@pytest.mark.parametrize("level,floor", [(1, 30), (8, 100), (9, 120), (13, 200), (14, 240), (18, 400)])
def test_spirit_cleave_minion_floor(level, floor):
    np.testing.assert_allclose(float(YO.w_minion_floor(level)), floor)


def test_swords_alternate_and_soul_unbound_resets():
    k = yctx()
    st = K.init(2, 6)
    assert float(K.attack_mods(st, k).magic_split[0]) == 0.0
    st, _ = K.on_attack(st, k, place(), launch([True, False], [JAX, -1]))
    assert float(K.attack_mods(st, k).magic_split[0]) == 0.5
    st, _, _ = tick(st, k, place(), 10.0, order([2, -1], x=[1000.0, 0.0]))
    assert float(K.attack_mods(st, k).magic_split[0]) == 0.0


# ---- Q -------------------------------------------------------------------------

def test_mortal_steel_thrust_stack_and_on_hit():
    k = yctx(ranks=jnp.asarray([[1, 1, 1, 1], [1, 1, 1, 0]], jnp.int32))
    u = place(x=[0.0, 300.0, 200.0, 3200.0, 6000.0, -3000.0], y=[0.0, 0.0, 30.0, 0.0, 0.0, 0.0])
    st, a, _ = tick(K.init(2, 6), k, u, 10.0, order([0, -1], x=[1000.0, 0.0]))
    assert float(a.cast_lockout[0]) == pytest.approx(0.35) and bool(a.cooldown_start[0, 0])
    st, outs, t = run(st, k, u, 10.0, 0.5)
    hits = fired(outs)
    assert len(hits) == 1 and hits[0][0] - 10.0 == pytest.approx(0.35, abs=1.5 * DT)
    p, b = hits[0][1], hits[0][2]
    np.testing.assert_allclose(sorted(p["raw"]), [25.0 + 1.1 * 130.0] * 2, rtol=1e-5)
    first = p["dst"] == MINION_RED                                     # the closer one takes basic damage
    assert p["flags"][first][0] & D.TAG_BASIC_ATTACK and not p["flags"][~first][0] & D.TAG_BASIC_ATTACK
    assert int(b.on_hit_target[0]) == MINION_RED and bool(b.on_attack_proc[0])
    assert int(st.yone.q_stacks[0]) == 1


def test_mortal_steel_crit_and_empowered_cast():
    k = yctx(ranks=jnp.asarray([[1, 1, 1, 1], [1, 1, 1, 0]], jnp.int32), crit_chance=jnp.asarray([0.5, 0.0]),
             rand_n=jnp.zeros((2, 6)), crit_damage=jnp.asarray([1.9, 2.0]))
    u = place()
    st = K.init(2, 6)
    st = st._replace(yone=st.yone._replace(q_stacks=jnp.asarray([2, 0]), q_stacks_until=jnp.asarray([20.0, 0.0])))
    st, a, _ = tick(st, k, u, 10.0, order([0, -1], x=[1000.0, 0.0]))
    assert bool(st.yone.q_three[0]) and int(st.yone.q_stacks[0]) == 0
    st, outs, t = run(st, k, u, 10.0, 1.0)
    dash = [b for _, b in outs if bool(b.dash.active[0])]
    assert len(dash) == 1 and float(dash[0].dash.to_x[0]) == pytest.approx(450.0)
    hits = fired(outs)
    assert len(hits) == 1                                              # Jax struck once (dash or whirlwind)
    p, b = hits[0][1], hits[0][2]
    np.testing.assert_allclose(p["raw"], [25.0 + 1.1 * 130.0 * 1.9], rtol=1e-5)   # crit with the 95% damage
    assert float(b.cc.knockup[0, JAX]) == pytest.approx(0.75)


# ---- W -------------------------------------------------------------------------

def test_spirit_cleave_cone_mixed_damage_and_shield():
    k = yctx(ranks=jnp.asarray([[1, 1, 1, 1], [1, 1, 1, 0]], jnp.int32), level=jnp.full((2,), 1, jnp.int32))
    u = place(x=[0.0, 300.0, -200.0, 3200.0, 6000.0, -3000.0])            # a minion behind Yone
    st, a, _ = tick(K.init(2, 6), k, u, 10.0, order([1, -1], x=[1000.0, 0.0]))
    assert float(a.cast_lockout[0]) == pytest.approx(0.5)
    st, outs, t = run(st, k, u, 10.0, 0.6)
    hits = fired(outs)
    assert len(hits) == 1
    p, b = hits[0][1], hits[0][2]
    assert set(p["dst"]) == {JAX}
    total = 10.0 + 0.08 * 1600.0
    np.testing.assert_allclose(sorted(p["raw"]), [total / 2] * 2, rtol=1e-5)
    assert sorted(p["dtype"]) == sorted([D.PHYSICAL, D.MAGIC])
    np.testing.assert_allclose(float(jnp.sum(b.shield.amount[0])), (40.0 + 0.65 * 30.0) * 2.0, rtol=1e-5)


# ---- E -------------------------------------------------------------------------

def report(src, dst, lost):
    p = D.packets(jnp.asarray([True]), jnp.asarray([src]), jnp.asarray([dst]), jnp.asarray([lost]),
                  jnp.asarray([D.PHYSICAL]), jnp.asarray([D.BASIC_ATTACK]), cast_id=jnp.asarray([3]))
    f = jnp.asarray([lost], jnp.float32)
    z = jnp.zeros_like(f)
    return Report(p, D.Resolved(f, z, z, f, jnp.zeros((1,), bool), jnp.zeros(6), jnp.zeros(6), None,
                                jnp.zeros(6, bool), jnp.zeros(6), jnp.zeros(6, bool), jnp.int32(0)), None)


def test_soul_unbound_marks_detonate_and_return():
    k = yctx(ranks=jnp.asarray([[1, 1, 1, 1], [1, 1, 1, 0]], jnp.int32))
    u = place()
    st, a, _ = tick(K.init(2, 6), k, u, 10.0, order([2, -1], x=[0.0, 0.0], y=[1000.0, 0.0]))
    assert bool(a.dash.active[0]) and float(a.dash.to_y[0]) == pytest.approx(300.0)
    assert float(a.dash.speed[0]) == 1200.0 and bool(K.ghosted(st, at(k, 10.1))[0])
    np.testing.assert_allclose([float(st.yone.e_body_x[0]), float(st.yone.e_body_y[0])], [0.0, -300.0], atol=1e-4)
    st, _ = K.on_damage(st, at(k, 11.0), u, report(YON, JAX, 200.0))
    np.testing.assert_allclose(float(st.yone.e_mark[0, JAX]), 50.0, rtol=1e-5)
    np.testing.assert_allclose(float(K.stats(st, at(k, 11.5)).percent_move_speed[0]), 0.1 + 0.2 * 0.5, rtol=1e-5)
    _, a, _ = tick(st, k, u, 10.3, order([2, -1]))
    assert not bool(a.cast_started[0])                                  # 0.5 s recast lockout
    st, a, _ = tick(st, k, u, 11.6, order([2, -1]))
    assert bool(a.cast_started[0]) and float(a.cast_lockout[0]) == pytest.approx(0.225)
    st, outs, t = run(st, k, u, 11.6, 0.3)
    hits = fired(outs)
    assert len(hits) == 1
    t, p, b = hits[0]
    np.testing.assert_allclose(p["raw"], [50.0], rtol=1e-5)
    assert p["dtype"][0] == D.TRUE and bool(b.cooldown_start[0, 2])
    assert bool(b.dash.active[0]) and float(b.dash.speed[0]) == pytest.approx(2950.0)
    assert float(b.dash.to_y[0]) == pytest.approx(-300.0)


def test_soul_unbound_returns_by_itself():
    k = yctx()
    st, _, _ = tick(K.init(2, 6), k, place(), 10.0, order([2, -1], x=[1000.0, 0.0]))
    st, outs, t = run(st, k, place(), 10.0, 5.5)
    back = [tt for tt, b in outs if bool(b.cooldown_start[0, 2])]
    assert len(back) == 1 and back[0] == pytest.approx(15.0 + 0.225, abs=2 * DT)
    assert not bool(st.yone.e_on[0])


# ---- R -------------------------------------------------------------------------

def test_fate_sealed_stun_blink_gust_pull():
    k = yctx(ranks=jnp.asarray([[1, 1, 1, 1], [1, 1, 1, 0]], jnp.int32))
    u = place(x=[0.0, 600.0, 300.0, 3200.0, 6000.0, -3000.0])
    st, a, _ = tick(K.init(2, 6), k, u, 10.0, order([3, -1], x=[1000.0, 0.0]))
    assert float(a.cast_lockout[0]) == pytest.approx(0.75) and bool(a.cooldown_start[0, 3])
    st, outs, t = run(st, k, u, 10.0, 1.3)
    stun = [(tt, b) for tt, b in outs if float(b.cc.stun[0, JAX]) > 0]
    assert len(stun) == 1 and stun[0][0] - 10.0 == pytest.approx(0.75, abs=1.5 * DT)
    assert float(stun[0][1].cc.stun[0, MINION_RED]) == pytest.approx(1.0)
    blink = [b for _, b in outs if bool(b.dash.active[0]) and bool(b.dash.blink[0])]
    assert len(blink) == 1 and float(blink[0].dash.to_x[0]) == pytest.approx(800.0)    # 200 beyond Jax
    hits = fired(outs)
    assert len(hits) == 1
    p = hits[0][1]
    np.testing.assert_allclose(p["raw"][p["dst"] == JAX], [100.0 + 0.4 * 30.0] * 2, rtol=1e-5)
    assert hits[0][0] - 10.0 == pytest.approx(0.75 + 0.3, abs=2.5 * DT)        # two chained timers
    act, x, _, v = K.displacement(st, at(k, hits[0][0]), 6)
    assert bool(act[MINION_RED]) and float(x[MINION_RED]) == pytest.approx(800.0) and float(v[MINION_RED]) == 3000.0
