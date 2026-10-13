"""Aatrox 26.19 kit on the modern packet contract (champions, AAT.* in docs/modern/CHAMPIONS.md)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.modern import champions as K
from lanerl_jax.modern.champions import aatrox as AA
from lanerl_jax.modern.champions import core
from lanerl_jax.modern.core import damage as D
from lanerl_jax.modern.core.stat_pipeline import champion_base
from lanerl_jax.modern.core.types import KIND_CHAMPION, KIND_MINION, KIND_MONSTER, CastOrder
from lanerl_jax.modern.data.champions import champion
from lanerl_jax.modern.items.effects.core import Kills, Report
from lanerl_jax.modern.tests.test_champions import DT, at, ctx, launch, units, valid

AAT, JAX, MINION_RED, MINION_RED2, TURRET_RED, MINION_BLUE = range(6)


def actx(**over):
    return ctx(ids=(266, 24), **over)


def order(slots, x=None, y=None, targets=None):
    c = len(slots)
    f = lambda v: jnp.asarray([0.0] * c if v is None else v, jnp.float32)    # noqa: E731
    return CastOrder(jnp.asarray(slots, jnp.int32), jnp.asarray([-1] * c if targets is None else targets, jnp.int32),
                     f(x), f(y))


def far_units(**over):
    """Everyone but Aatrox out of reach unless placed."""
    base = dict(x=[0.0, 3000.0, 3100.0, 3200.0, 6000.0, -3000.0])
    base.update(over)
    return units(**base)


def tick(st, k, u, t, o=None):
    """One world-order tick: cast then periodic at time ``t``."""
    k = at(k, t)
    st, a = K.cast(st, k, u, o if o is not None else order([-1, -1]))
    st, b = K.periodic(st, k, u)
    return st, a, b


def total_ad(k):
    return float(k.base_ad[0] + k.bonus_ad[0])


# ---- data ----------------------------------------------------------------------

def test_record_and_tables():
    rec = champion("Aatrox")
    assert rec["slots"]["Q"].endswith("/AatroxQ") and rec["slots"]["R"].endswith("/AatroxR")
    assert rec["character"]["pathfindingCollisionRadius"] == pytest.approx(35.0)
    assert "overrideGameplayCollisionRadius" not in rec["character"]
    r = jnp.asarray([[1, 1, 1, 1], [5, 5, 5, 3]], jnp.int32)
    np.testing.assert_allclose(np.asarray(core.cooldown_row("Aatrox", r)), [[14, 18, 9, 120], [6, 12, 5, 80]])
    np.testing.assert_allclose(np.asarray(core.mana_row("Aatrox", r)), 0.0)
    # Newer records give the windup as cast time / total time (wiki Attack speed): 0.3 / 1.52 = 19.737%.
    base = champion_base(["Aatrox", "Garen"])
    np.testing.assert_allclose(np.asarray(base.windup_percent), [0.3 / 1.52, 0.3 - 0.12], rtol=1e-5)


@pytest.mark.parametrize("level,pct,cool,cap", [(1, 0.04, 22.0, 100.0), (18, 0.10, 10.0, 320.0),
                                                 (9, 0.04 + 0.06 * 8 / 17, 22.0 - 12.0 * 8 / 17, 100 + 220 * 8 / 17)])
def test_passive_level_curves(level, pct, cool, cap):
    np.testing.assert_allclose(float(AA.by_level(AA.P_DAMAGE, level)), pct, rtol=1e-5)
    np.testing.assert_allclose(float(AA.by_level(AA.P_COOLDOWN, level)), cool, rtol=1e-5)
    np.testing.assert_allclose(float(AA.by_level(AA.P_MONSTER_CAP, level)), cap, rtol=1e-5)


@pytest.mark.parametrize("level,mult", [(1, 0.55), (5, 0.55), (6, 0.60), (11, 0.65), (16, 0.70), (18, 0.70)])
def test_q_minion_multiplier(level, mult):
    np.testing.assert_allclose(float(AA.minion_mult(jnp.int32(level))), mult, rtol=1e-5)


# ---- Q geometry ------------------------------------------------------------------

@pytest.mark.parametrize("index,along,perp,hit,sweet", [
    (0, 300.0, 0.0, True, False), (0, 500.0, 0.0, True, True), (0, 680.0, 0.0, True, True),   # edge reaches 690
    (0, 700.0, 0.0, False, False), (0, 500.0, 150.0, True, True), (0, 500.0, 160.0, False, False),
    (0, -60.0, 0.0, True, False), (0, -70.0, 0.0, False, False), (0, 440.0, 0.0, True, False),
    (1, -150.0, 0.0, True, False), (1, 0.0, 200.0, True, False), (1, 400.0, 300.0, True, True),
    (1, 540.0, 0.0, True, True), (1, 545.0, 0.0, False, False), (1, 290.0, 0.0, True, False),
    (2, 0.0, 0.0, True, False), (2, 200.0, 170.0, True, True), (2, 200.0, 190.0, True, False),
    (2, 500.0, 0.0, True, False), (2, 510.0, 0.0, False, False), (2, -110.0, 0.0, False, False)])
def test_q_regions(index, along, perp, hit, sweet):
    h, s = AA.q_regions(jnp.int32(index), jnp.float32(along), jnp.float32(perp), jnp.float32(65.0))
    assert (bool(h), bool(s)) == (hit, sweet)


# ---- Q casts ---------------------------------------------------------------------

def test_q_three_casts_damage_ramp_sweetspot_and_knockup():
    k = actx(ranks=jnp.asarray([[3, 1, 1, 1], [1, 1, 1, 0]], jnp.int32))
    st = K.init(2, 6)
    base = AA.ranked("Aatrox", "Q", "QBaseDamage", jnp.int32(3)) + AA.ranked("Aatrox", "Q", "QTotalADRatio",
                                                                             jnp.int32(3)) * total_ad(k)
    base = float(base)
    expected = []
    t = 10.0
    for i, (jx, sweet) in enumerate([(550.0, True), (200.0, False), (200.0, True)]):
        u = far_units(x=[0.0, jx, 3100.0, 3200.0, 6000.0, -3000.0])
        st, a, _ = tick(st, k, u, t, order([0, -1], x=[1000.0, 0.0]))
        assert bool(a.cast_started[0]) and int(a.cast_slot[0]) == 0 and bool(a.cooldown_start[0, 0])
        assert bool(K.immobile(st, at(k, t))[0]) and bool(K.attack_mods(st, at(k, t)).cannot_attack[0])
        packets = []
        for j in range(1, int(round(AA.Q_CAST_TIME / DT)) + 1):
            st, _, b = tick(st, k, u, t + j * DT)
            packets.append((t + j * DT, valid(b.packets), np.asarray(b.cc.knockup[0])))
        fired = [(tt, p, up) for tt, p, up in packets if len(p["raw"])]
        assert len(fired) == 1 and fired[0][0] == pytest.approx(t + AA.Q_CAST_TIME, abs=1.5 * DT)
        raw = base * (1.0 + 0.25 * i) * (1.75 if sweet else 1.0)
        np.testing.assert_allclose(fired[0][1]["raw"], [raw], rtol=1e-5)
        assert fired[0][1]["dst"][0] == JAX
        assert fired[0][2][JAX] == pytest.approx(0.25 if sweet else 0.0)
        expected.append(raw)
        t = t + 1.0
    # The sequence is spent: no fourth cast while the (restarted) cooldown runs.
    assert int(st.aatrox.q_stage[0]) == 0
    st, a, _ = tick(st, k._replace(cooldowns=jnp.asarray([[5.0, 0, 0, 0], [0, 0, 0, 0]], jnp.float32)),
                    far_units(), t, order([0, -1], x=[1000.0, 0.0]))
    assert not bool(a.cast_started[0])


def test_q_recast_needs_one_second_and_expires_after_four():
    k = actx()
    st = K.init(2, 6)
    u = far_units()
    st, a, _ = tick(st, k, u, 10.0, order([0, -1], x=[1.0, 0.0]))
    cd = k._replace(cooldowns=jnp.asarray([[9.0, 0, 0, 0], [0, 0, 0, 0]], jnp.float32))
    for t in np.arange(10.0 + DT, 10.95, DT):
        st, a, _ = tick(st, cd, u, float(t), order([0, -1], x=[1.0, 0.0]))
        assert not bool(a.cast_started[0])
    st, a, _ = tick(st, cd, u, 11.0, order([0, -1], x=[1.0, 0.0]))
    assert bool(a.cast_started[0]) and int(st.aatrox.q_index[0]) == 1
    for t in np.arange(11.0 + DT, 15.1, DT):
        st, _, _ = tick(st, cd, u, float(t))
    st, a, _ = tick(st, cd, u, 15.1, order([0, -1], x=[1.0, 0.0]))
    assert not bool(a.cast_started[0]) and int(st.aatrox.q_stage[0]) == 0


@pytest.mark.parametrize("kind,mult,up", [(KIND_MINION, 0.60, 0.25), (KIND_MONSTER, 1.0, 0.5)])
def test_q_minion_and_monster_rules(kind, mult, up):
    k = actx(ranks=jnp.asarray([[1, 1, 1, 1], [1, 1, 1, 0]], jnp.int32))           # level 9
    u = far_units(kind=[KIND_CHAMPION, KIND_CHAMPION, kind, KIND_MINION, 3, KIND_MINION],
                  x=[0.0, 3000.0, 520.0, 3200.0, 6000.0, -3000.0])
    st, _, _ = tick(K.init(2, 6), k, u, 10.0, order([0, -1], x=[1.0, 0.0]))
    for j in range(1, 19):
        st, _, b = tick(st, k, u, 10.0 + j * DT)
        if len(valid(b.packets)["raw"]):
            break
    base = 10.0 + 0.6 * total_ad(k) + (25.0 if kind == KIND_MONSTER else 0.0)
    np.testing.assert_allclose(valid(b.packets)["raw"], [base * 1.75 * mult], rtol=1e-5)
    assert float(b.cc.knockup[0, MINION_RED]) == pytest.approx(up)


def test_e_dash_clamped_attack_reset_and_castable_during_q():
    k = actx()
    u = far_units()
    st, a, _ = tick(K.init(2, 6), k, u, 10.0, order([0, -1], x=[1.0, 0.0]))
    st, a, _ = tick(st, k, u, 10.1, order([2, -1], x=[0.0, 0.0], y=[1000.0, 0.0]))
    assert bool(a.dash.active[0]) and bool(a.attack_reset[0]) and bool(a.cooldown_start[0, 2])
    np.testing.assert_allclose([float(a.dash.to_x[0]), float(a.dash.to_y[0]), float(a.dash.speed[0])],
                               [0.0, 300.0, 800.0], atol=1e-4)
    assert bool(st.aatrox.q_pending[0])                                # the Q wind-up survives the dash
    _, a, _ = tick(K.init(2, 6), k, u, 10.0, order([2, -1], x=[10.0, 0.0]))
    np.testing.assert_allclose(float(a.dash.to_x[0]), 75.0, atol=1e-4)                  # minimum range
    _, a, _ = tick(K.init(2, 6), k._replace(rooted=jnp.asarray([True, False])), u, 10.0, order([2, -1], x=[1.0, 0.0]))
    assert not bool(a.cast_started[0])


# ---- W -----------------------------------------------------------------------------

def run(st, k, u, t0, seconds):
    outs, t = [], t0
    for _ in range(int(round(seconds / DT))):
        t += DT
        st, _, b = tick(st, k, u, t)
        outs.append((t, b))
    return st, outs, t


def test_w_chain_first_enemy_slow_and_minion_double():
    k = actx(ranks=jnp.asarray([[1, 2, 1, 1], [1, 1, 1, 0]], jnp.int32))
    u = far_units(x=[0.0, 700.0, 500.0, 3200.0, 6000.0, -3000.0])         # minion in front of Jax
    st, a, _ = tick(K.init(2, 6), k, u, 10.0, order([1, -1], x=[1.0, 0.0]))
    assert bool(a.cooldown_start[0, 1]) and bool(K.immobile(st, at(k, 10.0))[0])
    st, outs, _ = run(st, k, u, 10.0, 0.9)
    hits = [(t, valid(b.packets), b) for t, b in outs if len(valid(b.packets)["raw"])]
    assert len(hits) == 1
    t, p, b = hits[0]
    assert p["dst"][0] == MINION_RED
    w = 40.0 + 0.4 * total_ad(k)
    np.testing.assert_allclose(p["raw"], [2 * w], rtol=1e-5)
    np.testing.assert_allclose(t - 10.0, 0.25 + (500.0 - 48.0) / 1800.0, atol=2 * DT)
    np.testing.assert_allclose(float(b.cc.slow[0, MINION_RED]), 0.275, rtol=1e-6)   # rank 2
    assert int(st.aatrox.w_target[0]) == -1                              # minions are not tethered


@pytest.mark.parametrize("move,pulled", [(0.0, True), (600.0, True), (700.0, False)])
def test_w_tether_pulls_back_from_inside_the_zone(move, pulled):
    k = actx(ranks=jnp.asarray([[1, 5, 1, 1], [1, 1, 1, 0]], jnp.int32))
    u = far_units(x=[0.0, 600.0, 3100.0, 3200.0, 6000.0, -3000.0])
    st, _, _ = tick(K.init(2, 6), k, u, 10.0, order([1, -1], x=[1.0, 0.0]))
    st, outs, t = run(st, k, u, 10.0, 0.6)
    assert int(st.aatrox.w_target[0]) == JAX
    cx = float(st.aatrox.w_cx[0])
    u2 = far_units(x=[0.0, 600.0 + move, 3100.0, 3200.0, 6000.0, -3000.0])
    st, outs, t = run(st, k, u2, t, 1.6)
    hits = [(tt, valid(b.packets), b) for tt, b in outs if len(valid(b.packets)["raw"])]
    assert len(hits) == (1 if pulled else 0)
    if pulled:
        tt, p, b = hits[0]
        np.testing.assert_allclose(p["raw"], [70.0 + 0.4 * total_ad(k)], rtol=1e-5)
        assert float(b.cc.knockup[0, JAX]) == pytest.approx(AA.W_AIRBORNE)
        a, x, _, v = K.displacement(st, at(k, tt), 6)
        assert bool(a[JAX]) and float(x[JAX]) == pytest.approx(cx) and float(v[JAX]) == pytest.approx(move / 0.25)


# ---- passive -----------------------------------------------------------------------

def report(packets, final):
    """A resolved report where every valid packet's post-mitigation damage is ``final`` (all of it health)."""
    f = jnp.asarray(final, jnp.float32)
    z = jnp.zeros_like(f)
    return Report(packets, D.Resolved(f, z, z, f, jnp.zeros(f.shape, bool), jnp.zeros(6), jnp.zeros(6), None,
                                      jnp.zeros(6, bool), jnp.zeros(6), jnp.zeros(6, bool), jnp.int32(0)), None)


def test_deathbringer_empowered_attack_heal_and_cooldown():
    k = actx(level=jnp.full((2,), 18, jnp.int32))
    u = far_units(x=[0.0, 150.0, 3100.0, 3200.0, 6000.0, -3000.0])
    st = K.init(2, 6)
    assert float(K.attack_mods(st, k).extra_range[0]) == 50.0
    st, _ = K.on_attack(st, k, u, launch([True, False], [JAX, -1]))
    st, o = K.on_hit(st, k, u, launch([True, False], [JAX, -1]))
    p = valid(o.packets)
    np.testing.assert_allclose(p["raw"], [0.10 * 1600.0], rtol=1e-5)
    assert p["dtype"][0] == D.MAGIC and p["flags"][0] & D.PROP_LIFESTEAL
    np.testing.assert_allclose(float(st.aatrox.p_ready_at[0]), 10.0 + 10.0, rtol=1e-6)
    assert float(K.attack_mods(st, k).extra_range[0]) == 0.0
    _, h = K.on_damage(st, k, u, report(o.packets, np.where(np.asarray(o.packets.valid), 100.0, 0.0)))
    # 100 passive heal + E passive 16% (+1.3% per 100 of 200 bonus health) of the 100 dealt to Jax
    np.testing.assert_allclose(float(h.heal[0]), 100.0 + 100.0 * (0.16 + 0.026), rtol=1e-5)
    # A later attack on a champion refunds 2 s.
    st, _ = K.on_attack(st, at(k, 11.0), u, launch([True, False], [JAX, -1]))
    st, o = K.on_hit(st, at(k, 11.0), u, launch([True, False], [JAX, -1]))
    assert not len(valid(o.packets)["raw"])
    np.testing.assert_allclose(float(st.aatrox.p_ready_at[0]), 18.0, rtol=1e-6)


def test_deathbringer_minions_monsters_structures():
    k = actx(level=jnp.full((2,), 1, jnp.int32))
    u = far_units(kind=[KIND_CHAMPION, KIND_CHAMPION, KIND_MONSTER, KIND_MINION, 3, KIND_MINION],
                  max_hp=[1500.0, 1600.0, 9000.0, 477.0, 5000.0, 477.0])
    st, _ = K.on_attack(K.init(2, 6), k, u, launch([True, False], [TURRET_RED, -1]))
    assert not bool(st.aatrox.p_armed[0])                               # not against structures
    st, _ = K.on_attack(st, k, u, launch([True, False], [MINION_RED, -1]))
    st, o = K.on_hit(st, k, u, launch([True, False], [MINION_RED, -1]))
    np.testing.assert_allclose(valid(o.packets)["raw"], [100.0], rtol=1e-5)  # 4% of 9000 capped at 100
    st = K.init(2, 6)
    st, _ = K.on_attack(st, k, u, launch([True, False], [MINION_RED2, -1]))
    st, o = K.on_hit(st, k, u, launch([True, False], [MINION_RED2, -1]))
    _, h = K.on_damage(st, k, u, report(o.packets, np.where(np.asarray(o.packets.valid), 19.0, 0.0)))
    np.testing.assert_allclose(float(h.heal[0]), 0.25 * 19.0, rtol=1e-5)


# ---- R -----------------------------------------------------------------------------

def test_world_ender_stats_decay_takedown_and_fear():
    k = actx(ranks=jnp.asarray([[1, 1, 1, 2], [1, 1, 1, 0]], jnp.int32))
    u = far_units(x=[0.0, 3000.0, 500.0, 3200.0, 6000.0, -3000.0])
    st, a, _ = tick(K.init(2, 6), k, u, 10.0, order([3, -1]))
    assert bool(a.cooldown_start[0, 3]) and not bool(K.immobile(st, at(k, 10.0))[0])
    st, outs, t = run(st, k, u, 10.0, 0.3)
    fear = [b for _, b in outs if float(b.cc.stun[0, MINION_RED]) > 0]
    assert len(fear) == 1 and float(fear[0].cc.stun[0, MINION_RED]) == pytest.approx(3.0)
    s = K.stats(st, at(k, t))
    np.testing.assert_allclose(float(s.attack_damage[0]), 0.3 * total_ad(k), rtol=1e-5)
    np.testing.assert_allclose(float(s.incoming_heal[0]), 0.75, rtol=1e-6)
    t0 = float(st.aatrox.r_ms_from[0])
    np.testing.assert_allclose(float(K.stats(st, at(k, t0 + 1.0)).percent_move_speed[0]), 0.8 * 0.9 ** 4, rtol=1e-5)
    assert bool(K.ghosted(st, at(k, t))[0]) and float(K.size(st, at(k, t))[0]) == pytest.approx(0.05)
    # A takedown 6 s in: +5 s, capped at 10 s remaining, and the speed refreshes.
    kills = Kills(jnp.asarray([1.0, 0.0]), jnp.zeros(2), jnp.zeros(2), jnp.zeros(2, bool), jnp.zeros((2, 6), bool))
    st = K.on_takedown(st, at(k, t0 + 6.0), u, kills)
    np.testing.assert_allclose(float(st.aatrox.r_until[0]), t0 + 6.0 + 9.0, rtol=1e-6)
    np.testing.assert_allclose(float(K.stats(st, at(k, t0 + 6.0)).percent_move_speed[0]), 0.8, rtol=1e-5)
    st = K.on_takedown(st, at(k, t0 + 6.5), u, kills)
    np.testing.assert_allclose(float(st.aatrox.r_until[0]), t0 + 6.5 + 10.0, rtol=1e-6)
    assert not bool(K.ghosted(st, at(k, t0 + 17.0))[0])
