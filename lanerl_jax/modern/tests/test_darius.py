"""Darius 26.19 kit on the modern packet contract (champions, DAR.* in docs/modern/CHAMPIONS.md)."""

from __future__ import annotations

import pytest as _pytest

_pytest.skip("Darius is not registered until every rule is exact "
             "(docs/modern/CHAMPIONS.md, Pending)", allow_module_level=True)

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.modern import champions as K
from lanerl_jax.modern.champions import core
from lanerl_jax.modern.champions import darius as DR
from lanerl_jax.modern.core import damage as D
from lanerl_jax.modern.core.types import KIND_CHAMPION, KIND_MINION, KIND_MONSTER, AttackLaunch, CastOrder
from lanerl_jax.modern.data.champions import champion, cooldowns
from lanerl_jax.modern.items.effects.core import Kills, Report
from lanerl_jax.modern.tests.test_champions import DT, at, ctx, jv, launch, units, valid

DAR, JAX, MINION_RED, MINION_RED2, TURRET_RED, MINION_BLUE = range(6)


def dctx(**over):
    return ctx(ids=(122, 24), **over)


def order(slots, targets=None, x=None, y=None):
    c = len(slots)
    f = lambda v: jnp.asarray([0.0] * c if v is None else v, jnp.float32)    # noqa: E731
    return CastOrder(jnp.asarray(slots, jnp.int32), jnp.asarray([-1] * c if targets is None else targets, jnp.int32),
                     f(x), f(y))


def kills(killed=(), champion_kill=0.0):
    ku = np.zeros((2, 6), bool)
    for u in killed:
        ku[0, u] = True
    return Kills(jnp.asarray([champion_kill, 0.0]), jnp.zeros(2), jnp.zeros(2), jnp.zeros(2, bool), jnp.asarray(ku))


def stacks(st, u):
    return int(st.darius.stacks[0, u])


def run_periodic(st, k, u, t0, seconds, **over):
    outs, t = [], t0
    for _ in range(int(round(seconds / DT))):
        t += DT
        st, o = K.periodic(st, at(k, t)._replace(**over), u)
        outs.append((t, o))
    return st, outs


# ---- data, passive -------------------------------------------------------------

def test_record_and_tables():
    rec = champion("Darius")
    assert rec["slots"]["Q"].endswith("/DariusCleave") and rec["slots"]["R"].endswith("/DariusExecute")
    assert rec["character"]["pathfindingCollisionRadius"] == pytest.approx(25.7666, rel=1e-4)
    assert "overrideGameplayCollisionRadius" not in rec["character"]                 # 65 since V14.9
    r = jnp.asarray([[1, 1, 1, 1], [5, 5, 5, 3]], jnp.int32)
    np.testing.assert_allclose(np.asarray(core.cooldown_row("Darius", r)), [[9, 5, 26, 120], [5, 5, 16, 80]])
    np.testing.assert_allclose(np.asarray(core.mana_row("Darius", r)), [[25, 40, 70, 100], [45, 40, 30, 0]])


@pytest.mark.parametrize("level,ad", [(1, 30), (6, 55), (10, 75), (11, 85), (13, 105), (14, 130), (18, 230)])
def test_noxian_might_ad(level, ad):
    np.testing.assert_allclose(float(DR.might_ad(jnp.int32(level))), ad)


def test_hemorrhage_stacks_ticks_and_noxian_might():
    k = dctx(level=jnp.full((2,), 1, jnp.int32))
    u = units()
    st = K.init(2, 6)
    for i in range(5):
        st, _ = K.on_hit(st, at(k, 10.0 + i * 0.1), u, launch([True, False], [JAX, -1]))
    assert stacks(st, JAX) == 5
    assert float(st.darius.might_until[0]) == pytest.approx(10.4 + 5.0, abs=1e-4)
    np.testing.assert_allclose(float(K.stats(st, at(k, 10.5)).attack_damage[0]), 30.0)
    st, outs = run_periodic(st, k, u, 10.4, 6.0)
    ticks = [(t, valid(o.packets)) for t, o in outs if len(valid(o.packets)["raw"])]
    assert len(ticks) == 4
    np.testing.assert_allclose([t - 10.0 for t, _ in ticks], [1.25, 2.5, 3.75, 5.0], atol=1.5 * DT)
    per = (13.0 + 0.3 * 30.0) / 4.0
    for _, p in ticks:
        np.testing.assert_allclose(p["raw"], [5 * per], rtol=1e-5)
        assert p["dst"][0] == JAX and p["flags"][0] & D.TAG_PERIODIC and p["dtype"][0] == D.PHYSICAL
    assert stacks(st, JAX) == 0


def test_hemorrhage_vs_monsters_and_not_structures():
    u = units(kind=[KIND_CHAMPION, KIND_CHAMPION, KIND_MONSTER, KIND_MINION, 3, KIND_MINION])
    k = dctx(level=jnp.full((2,), 1, jnp.int32))
    st, _ = K.on_hit(K.init(2, 6), k, u, launch([True, False], [MINION_RED, -1]))
    st, _ = K.on_hit(st, k, u, launch([True, False], [TURRET_RED, -1]))
    assert stacks(st, MINION_RED) == 1 and stacks(st, TURRET_RED) == 0
    _, outs = run_periodic(st, k, u, 10.0, 1.3)
    p = [valid(o.packets) for _, o in outs if len(valid(o.packets)["raw"])][0]
    np.testing.assert_allclose(p["raw"], [2.5 * (13.0 + 0.3 * 30.0) / 4.0], rtol=1e-5)


def test_armor_pen_from_e_rank():
    for rank, pen in ((0, 0.0), (1, 0.20), (5, 0.40)):
        k = dctx(ranks=jnp.asarray([[1, 1, rank, 0], [1, 1, 1, 0]], jnp.int32))
        np.testing.assert_allclose(float(K.stats(K.init(2, 6), k).percent_armor_pen[0]), pen, rtol=1e-6)


# ---- Q -------------------------------------------------------------------------

@pytest.mark.parametrize("rank", [1, 3, 5])
def test_q_decimate_blade_handle_heal(rank):
    k = dctx(ranks=jnp.asarray([[rank, 1, 1, 0], [1, 1, 1, 0]], jnp.int32))
    # Jax at 350 (blade), red minion at 100 (handle), red minion at 600 (outside 460 + 48).
    u = units(x=[0.0, 350.0, 100.0, 600.0, 3000.0, -100.0])
    st, out = K.cast(K.init(2, 6), k, u, order([0, -1]))
    np.testing.assert_allclose(float(out.mana_cost[0]), [25, 30, 35, 40, 45][rank - 1])
    assert not bool(out.cooldown_start[0, 0]) and bool(K.ghosted(st, k)[0])
    assert bool(K.attack_mods(st, k).cannot_attack[0])
    st, o = K.periodic(st, at(k, 10.75 - 2 * DT), u)
    assert not np.asarray(o.packets.valid).any()
    st, o = K.periodic(st, at(k, 10.75), u)
    p = valid(o.packets)
    full = jv("Darius", "Q", "BaseDamage", rank) + jv("Darius", "Q", "TotalADRatio", rank) / 100.0 * 130.0
    got = dict(zip(p["dst"].tolist(), p["raw"].tolist()))
    assert set(got) == {JAX, MINION_RED}
    np.testing.assert_allclose([got[JAX], got[MINION_RED]], [full, 0.35 * full], rtol=1e-5)
    np.testing.assert_allclose(float(o.heal[0]), 0.17 * (1500.0 - 1000.0), rtol=1e-5)
    assert bool(o.cooldown_start[0, 0]) and stacks(st, JAX) == 1 and stacks(st, MINION_RED) == 0
    assert not bool(K.ghosted(st, at(k, 11.05))[0])


# ---- W -------------------------------------------------------------------------

@pytest.mark.parametrize("rank", [1, 3, 5])
def test_w_crippling_strike(rank):
    k = dctx(ranks=jnp.asarray([[1, rank, 1, 0], [1, 1, 1, 0]], jnp.int32))
    u = units()
    st, out = K.cast(K.init(2, 6), k, u, order([1, -1]))
    assert bool(out.attack_reset[0]) and float(out.mana_cost[0]) == 40.0 and not bool(out.cooldown_start[0, 1])
    m = K.attack_mods(st, k)
    assert float(m.extra_range[0]) == 25.0 and bool(m.uncancellable[0])
    crit = AttackLaunch(jnp.asarray([True, False]), jnp.asarray([JAX, -1], jnp.int32), jnp.zeros(2, bool),
                        jnp.asarray([True, False]), jnp.asarray([7, 8], jnp.int32))
    st, out = K.on_hit(st, at(k, 10.3), u, crit)
    p = valid(out.packets)
    np.testing.assert_allclose(p["raw"], [(jv("Darius", "W", "ADRatio", rank) - 1.0) * 130.0 * 1.75], rtol=1e-5)
    np.testing.assert_allclose(float(out.cc.slow[0, JAX]), 0.9)
    np.testing.assert_allclose(float(out.cc.slow_duration[0, JAX]), 1.0)
    assert bool(out.cooldown_start[0, 1]) and stacks(st, JAX) == 1


def test_w_kill_refunds_mana_and_half_the_cooldown():
    k = dctx()
    u = units()
    st, _ = K.cast(K.init(2, 6), k, u, order([1, -1]))
    st, _ = K.on_hit(st, at(k, 10.3), u, launch([True, False], [MINION_RED, -1]))
    st = K.on_takedown(st, at(k, 10.3), u, kills([MINION_RED]))
    st, o = K.periodic(st, at(k, 10.3 + DT), u)
    np.testing.assert_allclose(float(o.mana_cost[0]), -40.0)
    assert bool(o.cooldown_start[0, 1])
    np.testing.assert_allclose(float(o.base_cooldown[0, 1]), 0.5 * cooldowns("Darius", "W")[2])
    st, o = K.periodic(st, at(k, 10.3 + 2 * DT), u)
    assert float(o.mana_cost[0]) == 0.0 and not bool(o.cooldown_start[0, 1])


# ---- E -------------------------------------------------------------------------

def test_e_apprehend_pulls_the_cone():
    k = dctx(ranks=jnp.asarray([[1, 1, 3, 0], [1, 1, 1, 0]], jnp.int32))
    # Jax at (450, 0) in the cone, minion 2 at (300, 300) outside it (45 deg), minion 3 at (520, 60) inside.
    u = units(x=[0.0, 450.0, 300.0, 520.0, 3000.0, -100.0], y=[0.0, 0.0, 300.0, 60.0, 0.0, 0.0])
    st, out = K.cast(K.init(2, 6), k, u, order([2, -1], x=[1000.0, 0.0], y=[0.0, 0.0]))
    np.testing.assert_allclose(float(out.mana_cost[0]), 50.0)
    np.testing.assert_allclose(float(out.cast_lockout[0]), 0.65)
    assert bool(out.cooldown_start[0, 2])
    st, o = K.periodic(st, at(k, 10.25), u)
    up = np.asarray(o.cc.knockup[0])
    assert up[JAX] > 0 and up[MINION_RED2] > 0 and up[MINION_RED] == 0 and up[MINION_BLUE] == 0
    np.testing.assert_allclose(up[JAX], (450.0 - 150.0) / 1500.0 + 0.5, rtol=1e-5)
    assert int(o.attack_target[0]) == JAX                           # the champion first
    act, tx, ty, sp = K.displacement(st, at(k, 10.3), 6)
    assert bool(act[JAX]) and bool(act[MINION_RED2]) and not bool(act[MINION_RED])
    np.testing.assert_allclose([float(tx[JAX]), float(ty[JAX])], [150.0, 0.0], atol=1e-3)
    np.testing.assert_allclose(float(sp[JAX]), 300.0 / up[JAX], rtol=1e-5)
    assert not bool(K.displacement(st, at(k, 10.25 + up[JAX] + 2 * DT), 6)[0][JAX])
    # The 40% slow starts when the rebound ends.
    st, outs = run_periodic(st, k, u, 10.25, 1.0)
    when = [t for t, o in outs if float(o.cc.slow[0, JAX]) > 0]
    assert len(when) == 1 and when[0] == pytest.approx(10.25 + up[JAX], abs=1.5 * DT)
    # Q and R are blocked for 0.4 s after the cast time.
    _, o = K.cast(st, at(k, 10.5), u, order([0, -1]))
    assert not bool(o.cast_started[0])


# ---- R -------------------------------------------------------------------------

@pytest.mark.parametrize("rank", [1, 2, 3])
def test_r_guillotine_damage_scales_with_stacks(rank):
    k = dctx(ranks=jnp.asarray([[1, 1, 1, rank], [1, 1, 1, 0]], jnp.int32))
    u = units()
    st = K.init(2, 6)
    for i in range(3):
        st, _ = K.on_hit(st, at(k, 9.0 + 0.1 * i), u, launch([True, False], [JAX, -1]))
    st, out = K.cast(st, k, u, order([3, -1], [JAX, -1]))
    assert bool(out.cast_started[0]) and bool(out.dash.active[0]) and int(out.dash.target[0]) == JAX
    np.testing.assert_allclose(float(out.cast_lockout[0]), 0.3667, rtol=1e-4)
    st, o = K.periodic(st, at(k, 10.3667), u)
    p = valid(o.packets)
    raw = (jv("Darius", "R", "RBaseDamage", rank) + 0.75 * 30.0) * (1.0 + 0.2 * 3)
    rp = p["raw"][p["dtype"] == D.TRUE]
    np.testing.assert_allclose(rp, [raw], rtol=1e-5)
    assert stacks(st, JAX) == 4                                    # stack applied after the damage


def test_r_needs_enemy_champion_in_range_and_not_rooted():
    _, out = K.cast(K.init(2, 6), dctx(), units(), order([3, -1], [MINION_RED, -1]))
    assert not bool(out.cast_started[0])
    far = units(x=[0.0, 600.0, 100.0, 250.0, 3000.0, -100.0])
    _, out = K.cast(K.init(2, 6), dctx(), far, order([3, -1], [JAX, -1]))
    assert not bool(out.cast_started[0])
    _, out = K.cast(K.init(2, 6), dctx(rooted=jnp.asarray([True, False])), units(), order([3, -1], [JAX, -1]))
    assert not bool(out.cast_started[0])


def test_r_execution_recast_window_and_rank3_reset():
    u = units()
    for rank in (1, 3):
        k = dctx(ranks=jnp.asarray([[1, 1, 1, rank], [1, 1, 1, 0]], jnp.int32))
        st, _ = K.cast(K.init(2, 6), k, u, order([3, -1], [JAX, -1]))
        st, _ = K.periodic(st, at(k, 10.3667), u)
        st = K.on_takedown(st, at(k, 10.3667), u, kills([JAX], 1.0))
        assert float(st.darius.might_until[0]) > 15.0                      # Noxian Might
        busy = at(k, 11.0)._replace(cooldowns=jnp.asarray([[0, 0, 0, 100.0], [0, 0, 0, 0]], jnp.float32),
                                    mana=jnp.asarray([0.0, 500.0], jnp.float32))
        st2, o = K.periodic(st, at(k, 10.3667 + DT), u)
        if rank == 3:
            assert bool(o.cooldown_start[0, 3]) and float(o.base_cooldown[0, 3]) == 0.0
        else:
            assert not bool(o.cooldown_start[0, 3])
            _, o = K.cast(st2, busy, u, order([3, -1], [JAX, -1]))
            assert bool(o.cast_started[0]) and float(o.mana_cost[0]) == 0.0 and not bool(o.cooldown_start[0, 3])
    # No execution: no recast.
    k = dctx()
    st, _ = K.cast(K.init(2, 6), k, u, order([3, -1], [JAX, -1]))
    st, _ = K.periodic(st, at(k, 10.3667), u)
    st = K.on_takedown(st, at(k, 10.7), u, kills([JAX], 1.0))             # dies 0.33 s later
    assert float(st.darius.r_recast_until[0]) < 0


# ---- jit -----------------------------------------------------------------------

def test_jit_all_hooks():
    k = dctx()
    u = units()
    rep = Report(D.packets(jnp.ones(2, bool), jnp.asarray([0, 2]), 1, 50.0, D.PHYSICAL, D.BASIC_ATTACK),
                 None, jnp.zeros(6))

    @jax.jit
    def tick(st, k):
        st, o1 = K.cast(st, k, u, order([2, 2], x=[300.0, 0.0], y=[0.0, 0.0]))
        st, o2 = K.periodic(st, k, u)
        st, o3 = K.on_attack(st, k, u, launch([True, True], [1, 0]))
        st, o4 = K.on_hit(st, k, u, launch([True, True], [1, 0]))
        st, o5 = K.on_damage(st, k, u, rep)
        st = K.on_takedown(st, k, u, kills())
        return st, (o1, o2, o3, o4, o5), K.stats(st, k), K.attack_mods(st, k), K.displacement(st, k, 6)

    st1, *_ = tick(K.init(2, 6), k)
    st2, *_ = tick(st1, at(k, 10.0 + DT))
    for a, b in zip(jax.tree_util.tree_leaves(st1), jax.tree_util.tree_leaves(st2)):
        assert a.dtype == b.dtype and a.shape == b.shape
    assert bool(st1.darius.e_pending[0]) and bool(st1.jax.e_on[1])
