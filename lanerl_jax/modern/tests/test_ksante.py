"""K'Sante 26.19 kit on the modern packet contract (champions, KSA.* in docs/modern/CHAMPIONS.md). The pull over
terrain needs real terrain and is exercised in test_ksante_world."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.modern import champions as K
from lanerl_jax.modern.champions import core
from lanerl_jax.modern.champions import ksante as KS
from lanerl_jax.modern.core import damage as D
from lanerl_jax.modern.core.stat_pipeline import champion_base
from lanerl_jax.modern.core.types import KIND_CHAMPION, KIND_MINION, KIND_MONSTER, CastOrder
from lanerl_jax.modern.data.champions import champion
from lanerl_jax.modern.tests.test_champions import DT, at, ctx, launch, units, valid

KSU, JAX, MINION_RED, MINION_RED2, TURRET_RED, MINION_BLUE = range(6)


def kctx(**over):
    base = dict(ranks=jnp.asarray([[1, 1, 1, 1], [1, 1, 1, 0]], jnp.int32), move_speed=jnp.asarray([330.0, 340.0]),
                dashing=jnp.zeros((2,), bool), kit_shield=jnp.zeros((2,), jnp.float32),
                level=jnp.full((2,), 1, jnp.int32), bonus_armor=jnp.zeros((2,), jnp.float32),
                bonus_mr=jnp.zeros((2,), jnp.float32), rooted=jnp.zeros((2,), bool))
    base.update(over)
    return ctx(ids=(897, 24), **base)


def order(slots, x=None, y=None, targets=None):
    c = len(slots)
    f = lambda v: jnp.asarray([0.0] * c if v is None else v, jnp.float32)    # noqa: E731
    return CastOrder(jnp.asarray(slots, jnp.int32), jnp.asarray([-1] * c if targets is None else targets, jnp.int32),
                     f(x), f(y))


def place(**over):
    base = dict(x=[0.0, 300.0, 3100.0, 3200.0, 6000.0, -3000.0], y=[0.0] * 6, radius=[80.0, 65.0, 48.0, 48.0, 88.0, 48.0])
    base.update(over)
    return units(**base)


def tick(st, k, u, t, o=None):
    k = at(k, t)
    st, a = K.cast(st, k, u, o if o is not None else order([-1, -1]))
    st, b = K.periodic(st, k, u)
    return st, a, b


def run(st, k, u, t0, seconds, raw=None):
    outs, t = [], t0
    for _ in range(int(round(seconds / DT))):
        t += DT
        kk = k if raw is None else k._replace(raw_cast_slot=jnp.asarray(raw(t), jnp.int32))
        st, _, b = tick(st, kk, u, t)
        outs.append((t, b))
    return st, outs, t


def fired(outs):
    return [(t, valid(b.packets), b) for t, b in outs if len(valid(b.packets)["raw"])]


# ---- data, curves ---------------------------------------------------------------

def test_record_tables_and_stats():
    rec = champion("KSante")
    assert rec["slots"]["W"].endswith("/KSanteW")
    r = jnp.asarray([[1, 1, 1, 1], [5, 5, 5, 3]], jnp.int32)
    np.testing.assert_allclose(np.asarray(core.cooldown_row("KSante", r)), [[3.5, 14, 10, 120], [3.5, 10, 8, 80]])
    np.testing.assert_allclose(np.asarray(core.mana_row("KSante", r)), [[20, 40, 45, 100], [20, 60, 65, 100]])
    b = champion_base(["KSante"])
    np.testing.assert_allclose(float(b.attack_range[0]), 150.0)
    np.testing.assert_allclose(float(b.attack_speed[0]), 0.688, rtol=1e-5)


@pytest.mark.parametrize("res,ct,cd,cd_ao", [(0, 0.45, 3.5, 3.5 * 2 / 3), (60, 0.40, 2.75, 2.75 * 2 / 3),
                                            (500, 0.35, 2.0, 2.0 * 2 / 3)])
def test_q_cast_time_and_cooldown(res, ct, cd, cd_ao):
    np.testing.assert_allclose(float(KS.q_cast_time(res)), ct, rtol=1e-5)
    np.testing.assert_allclose(float(KS.q_cooldown(res, False)), cd, rtol=1e-5)
    np.testing.assert_allclose(float(KS.q_cooldown(res, True)), cd_ao, rtol=1e-3)


@pytest.mark.parametrize("charge,length,stun,share", [(0.4, 100 + 350 * 0.4 / 0.9, 0.5, 0.1),
                                                      (0.65, 100 + 350 * 0.65 / 0.9, 1.125, 0.45),
                                                      (1.0, 450.0, 1.75, 0.8)])
def test_path_maker_curves(charge, length, stun, share):
    np.testing.assert_allclose(float(KS.w_length(charge)), length, rtol=1e-5)
    np.testing.assert_allclose(float(KS.w_stun(charge)), stun, rtol=1e-5)
    np.testing.assert_allclose(float(KS.w_true_share(charge)), share, rtol=1e-5)


def test_all_out_pull_speed_and_mark_damage():
    np.testing.assert_allclose([float(KS.r_pull_speed(d)) for d in (300.0, 1300.0, 2000.0, 4000.0)],
                               [900.0, 1300.0, 1700.0, 1700.0], rtol=1e-5)
    np.testing.assert_allclose(float(KS.mark_damage(1, 1000.0, KIND_CHAMPION)), 12 + 10, rtol=1e-5)
    np.testing.assert_allclose(float(KS.mark_damage(18, 1000.0, KIND_CHAMPION)), 12 + 20, rtol=1e-5)
    np.testing.assert_allclose(float(KS.mark_damage(1, 477.0, KIND_MINION)), 12 + 15, rtol=1e-5)     # minion floor
    np.testing.assert_allclose(float(KS.mark_damage(1, 3000.0, KIND_MONSTER)), 12 + 20, rtol=1e-5)   # monster cap
    np.testing.assert_allclose(float(KS.ao_bonus_share(100.0, 50.0)), 0.025, rtol=1e-5)


# ---- Q ---------------------------------------------------------------------------

def test_ntofo_strikes_hit_slow_stack_and_attack_order():
    k = kctx(bonus_armor=jnp.asarray([30.0, 0.0]), bonus_mr=jnp.asarray([30.0, 0.0]))
    u = place()
    st, a, _ = tick(K.init(2, 6), k, u, 10.0, order([0, -1], x=[1000.0, 0.0]))
    assert bool(a.cast_started[0]) and float(a.mana_cost[0]) == 20.0 and bool(a.cooldown_start[0, 0])
    np.testing.assert_allclose(float(a.cast_lockout[0]), 0.40, rtol=1e-5)
    np.testing.assert_allclose(float(a.base_cooldown[0, 0]), 2.75, rtol=1e-5)
    st, outs, t = run(st, k, u, 10.0, 0.5)
    hits = fired(outs)
    assert len(hits) == 1 and hits[0][0] - 10.0 == pytest.approx(0.40, abs=1.5 * DT)
    p, b = hits[0][1], hits[0][2]
    np.testing.assert_allclose(p["raw"][p["dst"] == JAX], [70.0 + 0.4 * 60.0], rtol=1e-5)
    assert float(b.cc.slow[0, JAX]) == pytest.approx(0.8) and float(b.cc.slow_duration[0, JAX]) == pytest.approx(0.5)
    assert int(b.attack_target[0]) == JAX and bool(b.attack_reset[0])
    assert int(st.ksante.q_stacks[0]) == 1 and float(st.ksante.mark_until[0, JAX]) > t


def test_shockwave_at_two_stacks_pulls_and_stuns():
    k = kctx()
    u = place(x=[0.0, 600.0, 3100.0, 3200.0, 6000.0, -3000.0])
    st = K.init(2, 6)
    st = st._replace(ksante=st.ksante._replace(q_stacks=jnp.asarray([2, 0], jnp.int32),
                                               q_stacks_until=jnp.asarray([20.0, -1e9], jnp.float32)))
    st, a, _ = tick(st, k, u, 10.0, order([0, -1], x=[1000.0, 0.0]))
    assert int(st.ksante.q_stacks[0]) == 0
    st, outs, t = run(st, k, u, 10.0, 1.2)
    hits = fired(outs)
    assert len(hits) == 1
    # Leaves after the 0.45 s cast time, 1600 units/s: Jax's edge (600 - 65) is reached ~0.33 s later.
    assert hits[0][0] - 10.0 == pytest.approx(0.45 + 535.0 / 1600.0, abs=2 * DT)
    b = hits[0][2]
    assert float(b.cc.stun[0, JAX]) == pytest.approx(0.8) and float(b.cc.knockup[0, JAX]) == pytest.approx(0.65)
    act, px, _, v = K.displacement(st, at(k, hits[0][0] + DT), 6)
    assert bool(act[JAX]) and float(v[JAX]) == pytest.approx(500.0)
    np.testing.assert_allclose(float(px[JAX]), 300.0, atol=1.0)          # 300 units toward K'Sante


# ---- W ---------------------------------------------------------------------------

def test_path_maker_charge_release_carry_and_stun():
    k = kctx()
    u = place(x=[0.0, 120.0, 3100.0, 3200.0, 6000.0, -3000.0])         # Jax touching K'Sante (80 + 65)
    st, a, _ = tick(K.init(2, 6), k, u, 10.0, order([1, -1], x=[1000.0, 0.0]))
    assert bool(a.cast_started[0]) and float(a.mana_cost[0]) == 40.0 and not bool(a.cooldown_start[0, 1])
    k1 = at(k, 10.2)
    assert float(K.defense(st, k1).received_mult[0]) == pytest.approx(0.7)
    assert bool(K.displacement_immune(st, k1)[0]) and bool(K.immobile(st, k1)[0])
    assert bool(K.attack_mods(st, k1).cannot_attack[0])
    # Recast at 20 ticks (2/3 s; raw slot: works while stunned).
    rel = 10.0 + 20 * DT
    st, outs, t = run(st, k._replace(stunned=jnp.asarray([True, False])), u, 10.0, 0.7,
                      raw=lambda t: [1 if abs(t - rel) < 1e-3 else -1, -1])
    go = [b for _, b in outs if bool(b.dash.active[0])]
    assert len(go) == 1
    charge = 20 * DT
    np.testing.assert_allclose(float(go[0].dash.to_x[0]), float(KS.w_length(charge)), rtol=1e-3)
    assert float(go[0].dash.speed[0]) == pytest.approx(1400.0)
    hits = fired(outs)
    assert hits and float(hits[0][2].cc.stun[0, JAX]) == pytest.approx(float(KS.w_stun(charge)), rel=1e-3)
    np.testing.assert_allclose(hits[0][1]["raw"][hits[0][1]["dst"] == JAX], [45.0 + 0.08 * 1600.0], rtol=1e-5)
    act, px, _, v = K.displacement(st, at(k, t), 6)
    assert bool(act[JAX]) and float(v[JAX]) == pytest.approx(1820.0)
    np.testing.assert_allclose(float(px[JAX]), float(KS.w_length(charge)) + 80 + 65, rtol=1e-3)
    # The cooldown starts when the dash ends.
    st, outs, t = run(st, k, u, t, 0.5)
    assert any(bool(b.cooldown_start[0, 1]) for _, b in outs)


# ---- E ---------------------------------------------------------------------------

def test_footwork_dash_and_shield_once():
    k = kctx(bonus_hp=jnp.asarray([400.0, 0.0]))
    st, a, _ = tick(K.init(2, 6), k, place(), 10.0, order([2, -1], x=[-1000.0, 0.0]))
    assert bool(a.dash.active[0]) and float(a.dash.to_x[0]) == pytest.approx(-250.0)
    assert float(a.dash.speed[0]) == pytest.approx(550.0 + 330.0) and bool(a.dash_stops_at_walls[0])
    np.testing.assert_allclose(float(jnp.sum(a.shield.amount[0])), 70.0 + 0.135 * 400.0, rtol=1e-5)
    k2 = at(k, 10.5)._replace(kit_shield=jnp.asarray([50.0, 0.0]))
    _, a2, _ = tick(st, k2, place(), 10.5, order([2, -1], x=[-1000.0, 0.0]))
    assert bool(a2.cast_started[0]) and float(jnp.sum(a2.shield.amount[0])) == 0.0   # the shield is still up


def test_footwork_to_an_allied_minion_crosses_terrain():
    k = kctx()
    u = place(x=[0.0, 300.0, 3100.0, 3200.0, 6000.0, -450.0])          # the allied minion 450 away
    st, a, _ = tick(K.init(2, 6), k, u, 10.0, order([2, -1], targets=[MINION_BLUE, -1]))
    assert bool(a.dash.active[0]) and int(a.dash.target[0]) == MINION_BLUE
    assert float(a.dash.speed[0]) == pytest.approx(1100.0 + 330.0) and not bool(a.dash_stops_at_walls[0])


# ---- R ---------------------------------------------------------------------------

def test_all_out_root_pull_blink_and_form():
    k = kctx(bonus_armor=jnp.asarray([100.0, 0.0]), bonus_mr=jnp.asarray([40.0, 0.0]), mana=jnp.asarray([300.0, 0.0]),
             max_mana=jnp.asarray([400.0, 0.0]), max_hp=jnp.asarray([2000.0, 1500.0]))
    u = place()
    st, a, _ = tick(K.init(2, 6), k, u, 10.0, order([3, -1], targets=[JAX, -1]))
    assert bool(a.cast_started[0]) and float(a.mana_cost[0]) == 100.0 and not bool(a.cooldown_start[0, 3])
    assert float(a.cc.root[0, JAX]) == pytest.approx(0.5) and bool(a.cc.tenacity_free[0, JAX])
    assert float(a.cast_lockout[0]) == pytest.approx(0.4) and bool(K.displacement_immune(st, at(k, 10.2))[0])
    k = k._replace(mana=jnp.asarray([200.0, 0.0]))
    st, outs, t = run(st, k, u, 10.0, 0.45)
    blinks = [(tt, b) for tt, b in outs if bool(b.dash.active[0]) and bool(b.dash.blink[0])]
    assert len(blinks) == 1 and blinks[0][0] - 10.0 == pytest.approx(0.4, abs=1.5 * DT)
    # No terrain: pulled 300 at 900 units/s; K'Sante lands 250 beyond.
    np.testing.assert_allclose(float(blinks[0][1].dash.to_x[0]), 300.0 + 300.0 + 250.0, rtol=1e-4)
    assert float(blinks[0][1].cc.knockup[0, JAX]) == pytest.approx(300.0 / 900.0, rel=1e-3)
    act, px, _, v = K.displacement(st, at(k, t), 6)
    assert bool(act[JAX]) and float(px[JAX]) == pytest.approx(600.0) and float(v[JAX]) == pytest.approx(900.0)
    # All Out.
    s = K.stats(st, at(k, t))
    assert float(s.attack_speed[0]) == pytest.approx(0.4) and float(s.omnivamp[0]) == pytest.approx(0.2)
    assert float(s.percent_bonus_armor_pen[0]) == pytest.approx(0.5)
    assert float(s.base_armor_loss[0]) == pytest.approx(85.0) and float(s.base_mr_loss[0]) == pytest.approx(34.0)
    assert float(K.hp_cap(st, at(k, t))[0]) == pytest.approx(0.65 * 2000.0)
    on, mana = K.mana_override(st, at(k, t))
    assert bool(on[0]) and float(mana[0]) == pytest.approx(400.0, rel=1e-2)
    # The damage near the end of the pull, then a 0.25 s stun.
    st, outs, t = run(st, k, u, t, 0.4)
    hits = fired(outs)
    assert hits and hits[0][1]["raw"][0] == pytest.approx(80.0 + (0.01 + 0.0001 * 140.0) * 1600.0, rel=1e-4)   # + All Out bonus
    assert any(float(b.cc.stun[0, JAX]) == pytest.approx(0.25) for _, b in outs)
    # Halfway: half the mana; free basic abilities; Q a third faster.
    on, mana = K.mana_override(st, at(k, 10.4 + 7.5))
    assert float(mana[0]) == pytest.approx(200.0, rel=2e-2)
    st2, a, _ = tick(st, k, u, 12.0, order([0, -1], x=[1000.0, 0.0]))
    assert float(a.mana_cost[0]) == 0.0
    # The end: the mana from before is restored and the cooldown starts.
    st, outs, t = run(st, k, u, 12.0, 13.5)
    ends = [b for _, b in outs if bool(b.cooldown_start[0, 3])]
    assert len(ends) == 1 and float(ends[0].base_cooldown[0, 3]) == 120.0
    on, mana = K.mana_override(st, at(k, t))
    assert float(K.hp_cap(st, at(k, t))[0]) == np.inf
    assert int(st.ksante.r_target[0]) == -1


# ---- passive ---------------------------------------------------------------------

def test_mark_consumed_by_an_attack_launched_while_marked():
    k = kctx(level=jnp.full((2,), 18, jnp.int32))
    u = place()
    st = K.init(2, 6)
    st = st._replace(ksante=st.ksante._replace(mark_until=st.ksante.mark_until.at[0, JAX].set(10.1)))
    st, _ = K.on_attack(st, at(k, 10.0), u, launch([True, False], [JAX, -1]))
    st, o = K.on_hit(st, at(k, 10.3), u, launch([True, False], [JAX, -1]))
    np.testing.assert_allclose(valid(o.packets)["raw"], [12.0 + 0.02 * 1600.0], rtol=1e-5)
    st, o = K.on_hit(st, at(k, 11.0), u, launch([True, False], [JAX, -1]))
    assert len(valid(o.packets)["raw"]) == 0                         # consumed
