"""Jayce 26.19 kit on the modern packet contract (champions, JAY.* in docs/modern/CHAMPIONS.md). Stance cooldown
swapping, ranged attacks and the 6-rank table run through the compiled tick in test_jayce_world."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.modern import champions as K
from lanerl_jax.modern.champions import jayce as J
from lanerl_jax.modern.core import damage as D
from lanerl_jax.modern.core.stat_pipeline import champion_base
from lanerl_jax.modern.core.types import KIND_CHAMPION, KIND_MINION, KIND_MONSTER, KIND_TURRET, CastOrder
from lanerl_jax.modern.data.champions import champion, cooldowns
from lanerl_jax.modern.tests.test_champions import DT, at, ctx, launch, units, valid

JAY, JAX, MINION_RED, MINION_RED2, TURRET_RED, MINION_BLUE = range(6)


def kctx(**over):
    base = dict(ranks=jnp.asarray([[1, 1, 1, 1], [1, 1, 1, 0]], jnp.int32), level=jnp.full((2,), 1, jnp.int32),
                rooted=jnp.zeros((2,), bool), dashing=jnp.zeros((2,), bool))
    base.update(over)
    return ctx(ids=(126, 24), **base)


def order(slots, x=None, y=None, targets=None):
    c = len(slots)
    f = lambda v: jnp.asarray([0.0] * c if v is None else v, jnp.float32)    # noqa: E731
    return CastOrder(jnp.asarray(slots, jnp.int32), jnp.asarray([-1] * c if targets is None else targets, jnp.int32),
                     f(x), f(y))


def place(**over):
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


def jstate(st):
    return st.jayce


# ---- data ------------------------------------------------------------------------

def test_record_rank_tables_and_stance_rows():
    rec = champion("Jayce")
    assert rec["slots"]["Q2"].endswith("/JayceShockBlast") and rec["slots"]["R2"].endswith("/JayceStanceGtH")
    b = champion_base(["Jayce"])
    np.testing.assert_allclose(float(b.attack_range[0]), 125.0)
    np.testing.assert_allclose(float(b.attack_speed[0]), 0.658, rtol=1e-5)
    np.testing.assert_allclose(float(b.windup_percent[0]), 0.3 - 0.205, rtol=1e-5)
    cap, innate = K.rank_tables(jnp.asarray([126, 86]))
    np.testing.assert_array_equal(np.asarray(cap), [[6, 6, 6, 1], [5, 5, 5, 3]])
    np.testing.assert_array_equal(np.asarray(innate), [[0, 0, 0, 1], [0, 0, 0, 0]])
    assert cooldowns("Jayce", "Q") == [16.0, 14.0, 12.0, 10.0, 8.0, 6.0]
    assert cooldowns("Jayce", "W") == [10.0] * 6                       # the BIN's default cooldown
    r = jnp.asarray([[1, 1, 1, 1], [6, 6, 6, 1]], jnp.int32)
    ham, can = J.cooldown_rows(r)
    np.testing.assert_allclose(np.asarray(ham), [[16, 10, 20, 6], [6, 10, 10, 6]])
    np.testing.assert_allclose(np.asarray(can), [[8, 13, 16, 6], [8, 5, 16, 6]], rtol=1e-6)
    hm, cm = J.mana_rows(r)
    np.testing.assert_allclose(np.asarray(hm), [[40, 40, 55, 0], [40, 40, 55, 0]])
    np.testing.assert_allclose(np.asarray(cm), [[55, 40, 50, 0], [80, 40, 50, 0]])
    lv = jnp.asarray([1, 5, 6, 11, 16, 18])
    np.testing.assert_allclose(np.asarray(J.by_level(J.R_RES, lv)), [5, 5, 12, 19, 26, 26])
    np.testing.assert_allclose(np.asarray(J.by_level(J.R_HIT, lv)), [25, 25, 60, 95, 130, 130])
    np.testing.assert_allclose(np.asarray(J.by_level(J.R_SHRED, lv)), [0.2, 0.2, 0.25, 0.3, 0.35, 0.35], rtol=1e-5)


# ---- Transform, stances -----------------------------------------------------------

def test_transform_swaps_stance_ranges_cooldowns_and_arms_the_hits():
    k = kctx()
    u = place()
    st = K.init(2, 6)
    np.testing.assert_allclose(np.asarray(K.unit_target_ranges(jnp.asarray([126, 24]), st))[0],
                               [J.Q_RANGE, 0.0, J.E_RANGE, 0.0])
    s0 = K.stats(st, at(k, 10.0))
    np.testing.assert_allclose(float(s0.armor[0]), 5.0 + 0.075 * 30.0, rtol=1e-5)         # Mercury Hammer
    assert float(s0.attack_range[0]) == 0.0
    st, a, _ = tick(st, k, u, 10.0, order([3, -1]))
    js = jstate(st)
    assert bool(js.cannon[0]) and bool(js.shred_ready[0]) and not bool(js.hammer_hit[0])
    assert bool(a.swap_cooldowns[0]) and bool(a.cooldown_start[0, 3]) and float(a.base_cooldown[0, 3]) == 6.0
    assert not bool(a.cast_started[0]) and float(a.mana_cost[0]) == 0.0      # not an ability activation
    s1 = K.stats(st, at(k, 10.0))
    assert float(s1.attack_range[0]) == 375.0 and float(s1.armor[0]) == 0.0
    assert float(s1.move_speed[0]) == 30.0 and bool(K.ghosted(st, at(k, 10.5))[0])
    assert not bool(K.ghosted(st, at(k, 10.8))[0])                                       # 0.75 s
    np.testing.assert_allclose(float(K.attack_missile_speed(st, at(k, 10.0))[0]), 2000.0)
    np.testing.assert_allclose(np.asarray(K.unit_target_ranges(jnp.asarray([126, 24]), st))[0], 0.0)
    # Back to Mercury Hammer: the hammer hit is armed, the shred is dropped.
    st, a, _ = tick(st, k._replace(cooldowns=jnp.zeros((2, 4))), u, 16.1, order([3, -1]))
    assert not bool(jstate(st).cannon[0]) and bool(jstate(st).hammer_hit[0]) and not bool(jstate(st).shred_ready[0])
    assert float(K.attack_missile_speed(st, at(k, 16.1))[0]) == 0.0


def test_mercury_hammer_hit_and_lightning_field_mana_on_hit():
    k = kctx(ranks=jnp.asarray([[1, 2, 1, 1], [1, 1, 1, 0]], jnp.int32))
    u = place()
    st = K.init(2, 6)
    st = st._replace(jayce=jstate(st)._replace(hammer_hit=jnp.asarray([True, False])))
    st, o = K.on_hit(st, at(k, 10.0), u, launch([True, False], [JAX, -1]))
    p = valid(o.packets)
    np.testing.assert_allclose(p["raw"][p["dst"] == JAX], [25.0 + 0.3 * 30.0], rtol=1e-5)
    assert int(p["dtype"][0]) == D.MAGIC
    np.testing.assert_allclose(float(o.mana_gain[0]), 17.0)                              # rank 2 (15 at rank 1)
    assert not bool(jstate(st).hammer_hit[0])
    st, o = K.on_hit(st, at(k, 11.0), u, launch([True, False], [TURRET_RED, -1]))
    assert len(valid(o.packets)["raw"]) == 0 and float(o.mana_gain[0]) == 17.0          # structures restore too


# ---- Mercury Hammer ----------------------------------------------------------------

def test_to_the_skies_leaps_strikes_around_the_target_and_slows():
    k = kctx()
    u = place(x=[0.0, 500.0, 600.0, 3200.0, 6000.0, -3000.0])
    st, a, _ = tick(K.init(2, 6), k, u, 10.0, order([0, -1], targets=[JAX, -1]))
    assert bool(a.dash.active[0]) and int(a.dash.target[0]) == -1 and bool(a.cooldown_start[0, 0])
    np.testing.assert_allclose(float(a.dash.to_x[0]), 500.0 - J.Q_STOP)                  # 225 short of Jax
    np.testing.assert_allclose(float(a.dash.speed[0]), (500.0 - J.Q_STOP) / J.Q_LEAP_TIME, rtol=1e-5)
    assert float(a.mana_cost[0]) == 40.0 and float(a.base_cooldown[0, 0]) == 16.0
    st, outs, _ = run(st, k, u, 10.0, 0.6)
    hits = fired(outs)
    assert len(hits) == 1 and hits[0][0] - 10.0 == pytest.approx(J.Q_LEAP_TIME, abs=1.5 * DT)
    p, b = hits[0][1], hits[0][2]
    assert set(p["dst"].tolist()) == {JAX, MINION_RED}                                  # 325 from the landing
    np.testing.assert_allclose(p["raw"][p["dst"] == JAX], [60.0 + 1.35 * 30.0], rtol=1e-5)
    assert float(b.cc.slow[0, JAX]) == pytest.approx(0.35) and float(b.cc.slow_duration[0, JAX]) == 2.0
    assert int(b.attack_target[0]) == JAX


def test_lightning_field_ticks_every_second_for_four_seconds():
    k = kctx(ap=jnp.asarray([40.0, 0.0]))
    u = place(x=[0.0, 300.0, 380.0, 3200.0, 6000.0, -3000.0])
    st, a, _ = tick(K.init(2, 6), k, u, 10.0, order([1, -1]))
    assert bool(a.cooldown_start[0, 1]) and float(a.base_cooldown[0, 1]) == 10.0 and float(a.mana_cost[0]) == 40.0
    st, outs, _ = run(st, k, u, 10.0, 5.0)
    hits = fired(outs)
    assert [round(t - 10.0, 1) for t, _, _ in hits] == [1.0, 2.0, 3.0, 4.0]
    for _, p, _ in hits:
        assert set(p["dst"].tolist()) == {JAX, MINION_RED}            # minion edge at 332 < 350
        np.testing.assert_allclose(p["raw"], (140.0 + 40.0) / 4.0, rtol=1e-5)


def test_thundering_blow_roots_then_strikes_knocks_back_and_locks_q():
    k = kctx()
    u = place()
    st, a, _ = tick(K.init(2, 6), k, u, 10.0, order([2, -1], targets=[JAX, -1]))
    assert float(a.cc.root[0, JAX]) == pytest.approx(J.E_CAST) and float(a.cast_lockout[0]) == pytest.approx(J.E_CAST)
    st, outs, t = run(st, k, u, 10.0, 0.4)
    hits = fired(outs)
    assert len(hits) == 1 and hits[0][0] - 10.0 == pytest.approx(J.E_CAST, abs=1.5 * DT)
    p, b = hits[0][1], hits[0][2]
    np.testing.assert_allclose(p["raw"], [0.08 * 1600.0 + 30.0], rtol=1e-5)
    assert int(p["dtype"][0]) == D.MAGIC and float(b.cc.knockup[0, JAX]) == pytest.approx(J.E_KNOCK_TIME)
    act, x, y, v = K.displacement(st, at(k, hits[0][0] + DT), 6)
    # From 300 away: pushed to max(475, 300 + 250) = 550 from Jayce over 0.35 s.
    assert bool(act[JAX]) and float(x[JAX]) == pytest.approx(550.0) and float(v[JAX]) == pytest.approx(
        250.0 / 0.35, rel=1e-4)
    _, a, _ = tick(st, k, u, hits[0][0] + 0.2, order([0, -1], targets=[JAX, -1]))
    assert not bool(a.dash.active[0])                                     # no To the Skies! for 0.4 s
    _, a, _ = tick(st, k, u, hits[0][0] + 0.45, order([0, -1], targets=[JAX, -1]))
    assert bool(a.dash.active[0])


def test_thundering_blow_pushes_close_targets_to_475():
    k = kctx()
    u = place(x=[0.0, 100.0, 3100.0, 3200.0, 6000.0, -3000.0])
    st, _, _ = tick(K.init(2, 6), k, u, 10.0, order([2, -1], targets=[JAX, -1]))
    st, outs, _ = run(st, k, u, 10.0, 0.3)
    act, x, _, v = K.displacement(st, at(k, 10.3), 6)
    assert float(x[JAX]) == pytest.approx(475.0) and float(v[JAX]) == pytest.approx(375.0 / 0.35, rel=1e-4)


def test_thundering_blow_damage_is_capped_on_monsters():
    k = kctx()
    u = place(kind=[KIND_CHAMPION, KIND_MONSTER, KIND_MINION, KIND_MINION, KIND_TURRET, KIND_MINION],
              max_hp=[1500.0, 9000.0, 477.0, 477.0, 5000.0, 477.0])
    st, _, _ = tick(K.init(2, 6), k, u, 10.0, order([2, -1], targets=[JAX, -1]))
    st, outs, _ = run(st, k, u, 10.0, 0.4)
    np.testing.assert_allclose(fired(outs)[0][1]["raw"], [200.0])


# ---- Mercury Cannon ----------------------------------------------------------------

def cannon_state():
    st = K.init(2, 6)
    return st._replace(jayce=jstate(st)._replace(cannon=jnp.asarray([True, False])))


def test_shock_blast_flies_and_detonates_on_the_first_enemy():
    k = kctx()
    u = place(x=[0.0, 800.0, 3100.0, 3200.0, 6000.0, -3000.0], y=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    u = u._replace(x=u.x.at[MINION_RED].set(850.0).at[MINION_RED2].set(800.0), y=u.y.at[MINION_RED2].set(200.0))
    st, a, _ = tick(cannon_state(), k, u, 10.0, order([0, -1], x=[2000.0, 0.0]))
    assert float(a.cast_lockout[0]) == pytest.approx(J.Q2_CAST) and float(a.mana_cost[0]) == 55.0
    assert float(a.base_cooldown[0, 0]) == 8.0
    st, outs, _ = run(st, k, u, 10.0, 1.2)
    hits = fired(outs)
    assert len(hits) == 1
    t_hit = hits[0][0] - 10.0
    expect = J.Q2_CAST + (800.0 - 65.0 - J.Q2_HALF) / J.Q2_SPEED
    assert t_hit == pytest.approx(expect, abs=1.5 * DT)
    p = hits[0][1]
    assert set(p["dst"].tolist()) == {JAX, MINION_RED}            # minion 50 beyond: in the blast; 200 aside: not
    np.testing.assert_allclose(p["raw"], 80.0 + 1.3 * 30.0, rtol=1e-5)


def test_shock_blast_detonates_at_range_and_through_the_gate():
    k = kctx()
    u = place(x=[0.0, 5000.0, 5100.0, 5200.0, 9000.0, -3000.0])
    st, _, _ = tick(cannon_state(), k, u, 10.0, order([0, -1], x=[3000.0, 0.0]))
    st, _, _ = run(st, k, u, 10.0, 1.2)
    js = jstate(st)
    assert not bool(js.m_fly[0]) and float(js.m_dist[0]) == pytest.approx(J.Q2_RANGE)
    # Through a gate 300 ahead: 1.4x, the enhanced blast, and the 1600 total range.
    k2 = k._replace(cooldowns=jnp.zeros((2, 4)))
    u2 = place(x=[0.0, 1500.0, 3100.0, 3200.0, 9000.0, -3000.0])
    st = cannon_state()
    st, a, _ = tick(st, k2, u2, 20.0, order([2, -1], x=[300.0, 0.0]))
    assert float(jstate(st).gate_x[0]) == pytest.approx(300.0) and bool(a.cooldown_start[0, 2])
    st, _, _ = tick(st, k2, u2, 20.0 + DT, order([0, -1], x=[3000.0, 0.0]))
    st, outs, _ = run(st, k2, u2, 20.0 + DT, 1.0)
    hits = fired(outs)
    assert len(hits) == 1 and bool(jstate(st).m_up[0])
    np.testing.assert_allclose(hits[0][1]["raw"], (80.0 + 1.3 * 30.0) * 1.4, rtol=1e-5)
    t_hit = hits[0][0] - (20.0 + DT)
    expect = J.Q2_CAST + (300.0 - J.E2_THICK) / J.Q2_SPEED + (1500.0 - 65.0 - J.Q2_HALF - 200.0) / J.Q2_SPEED_UP
    assert t_hit == pytest.approx(expect, abs=2.0 * DT)


def test_gate_hastes_jayce_and_the_haste_decays():
    k = kctx(ranks=jnp.asarray([[1, 1, 3, 1], [1, 1, 1, 0]], jnp.int32))
    u = place()
    st, _, _ = tick(cannon_state(), k, u, 10.0, order([2, -1], x=[50.0, 0.0]))       # on top of Jayce
    st, _, _ = tick(st, k, u, 10.0 + DT)
    np.testing.assert_allclose(float(J.gate_haste(jstate(st), at(k, 10.0 + DT))[0]), 0.45, rtol=1e-5)
    np.testing.assert_allclose(float(J.gate_haste(jstate(st), at(k, 10.0 + DT + 1.5))[0]), 0.225, rtol=1e-3)
    assert float(J.gate_haste(jstate(st), at(k, 10.0 + DT + 3.1))[0]) == 0.0


def test_hyper_charge_attack_speed_damage_and_post_effect_cooldown():
    k = kctx()
    u = place()
    st, a, _ = tick(cannon_state(), k, u, 10.0, order([1, -1]))
    assert bool(a.attack_reset[0]) and not bool(a.cooldown_start[0, 1]) and float(a.mana_cost[0]) == 40.0
    np.testing.assert_allclose(float(K.stats(st, at(k, 10.0)).attack_speed[0]), 3.6, rtol=1e-6)
    np.testing.assert_allclose(float(K.attack_mods(st, at(k, 10.0)).damage_mult[0]), 0.70, rtol=1e-5)
    np.testing.assert_allclose(float(K.attack_missile_speed(st, at(k, 10.0))[0]), 1800.0)
    for i in range(3):
        st, o = K.on_attack(st, at(k, 10.2 + 0.2 * i), u, launch([True, False], [JAX, -1]))
    assert int(jstate(st).w2_left[0]) == 0 and bool(o.cooldown_start[0, 1])
    np.testing.assert_allclose(float(o.base_cooldown[0, 1]), 13.0)
    # Expiry in Mercury Hammer stance starts the cannon W's cooldown on the inactive stance's row.
    st, _, _ = tick(cannon_state(), k, u, 20.0, order([1, -1]))
    st = st._replace(jayce=jstate(st)._replace(cannon=jnp.asarray([False, False])))
    st, outs, _ = run(st, k, u, 20.0, J.W2_TIME + 0.1)
    ends = [b for _, b in outs if bool(np.asarray(b.alt_cooldown_start)[0, 1])]
    assert len(ends) == 1 and float(ends[0].alt_base_cooldown[0, 1]) == 13.0


def test_cannon_shred_rides_the_attack_and_lands_on_hit():
    k = kctx(level=jnp.full((2,), 6, jnp.int32))
    u = place()
    st = cannon_state()
    st = st._replace(jayce=jstate(st)._replace(shred_ready=jnp.asarray([True, False])))
    st, _ = K.on_attack(st, at(k, 10.0), u, launch([True, False], [JAX, -1]))
    assert bool(jstate(st).shred_flying[0]) and not bool(jstate(st).shred_ready[0])
    st, _ = K.on_hit(st, at(k, 10.2), u, launch([True, False], [JAX, -1]))
    d = K.debuffs(st, at(k, 10.5), u)
    np.testing.assert_allclose(float(d.percent_armor_reduction[JAX]), 0.25, rtol=1e-5)
    np.testing.assert_allclose(float(d.percent_mr_reduction[JAX]), 0.25, rtol=1e-5)
    assert float(K.debuffs(st, at(k, 15.3), u).percent_armor_reduction[JAX]) == 0.0        # 5 s
    # Not on structures: kept for the next attack.
    st = cannon_state()
    st = st._replace(jayce=jstate(st)._replace(shred_ready=jnp.asarray([True, False])))
    st, _ = K.on_attack(st, at(k, 10.0), u, launch([True, False], [TURRET_RED, -1]))
    assert bool(jstate(st).shred_ready[0]) and not bool(jstate(st).shred_flying[0])


def test_merge_out_keeps_earlier_parts_swaps_alt_starts_and_mana():
    from lanerl_jax.modern.champions import core
    a = core.out(2, 6, swap_cooldowns=jnp.asarray([True, False]), mana_gain=jnp.asarray([5.0, 0.0]),
                 alt_cooldown_start=jnp.zeros((2, 3), bool).at[0, 1].set(True),
                 alt_base_cooldown=jnp.zeros((2, 3)).at[0, 1].set(13.0))
    b = core.out(2, 6, swap_cooldowns=jnp.zeros((2,), bool), mana_gain=jnp.asarray([2.0, 0.0]))
    m = core.merge_out([a, b, core.out(2, 6)], 2, 6)
    assert bool(m.swap_cooldowns[0]) and float(m.mana_gain[0]) == 7.0
    assert bool(m.alt_cooldown_start[0, 1]) and float(m.alt_base_cooldown[0, 1]) == 13.0
