"""Camille 26.19 kit on the modern packet contract (champions, CAM.* in docs/modern/CHAMPIONS.md). Hookshot needs
real terrain and is exercised in test_camille_world."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.modern import champions as K
from lanerl_jax.modern.champions import camille as CM
from lanerl_jax.modern.champions import core
from lanerl_jax.modern.core import damage as D
from lanerl_jax.modern.core.stat_pipeline import champion_base
from lanerl_jax.modern.core.types import KIND_CHAMPION, CastOrder
from lanerl_jax.modern.data.champions import champion
from lanerl_jax.modern.items.effects.core import Report
from lanerl_jax.modern.tests.test_champions import DT, at, ctx, launch, units, valid

CAM, JAX, MINION_RED, MINION_RED2, TURRET_RED, MINION_BLUE = range(6)


def cctx(**over):
    base = dict(ranks=jnp.asarray([[1, 1, 1, 1], [1, 1, 1, 0]], jnp.int32), move_speed=jnp.asarray([340.0, 340.0]),
                dashing=jnp.zeros((2,), bool), kit_shield=jnp.zeros((2,), jnp.float32), level=jnp.full((2,), 1, jnp.int32))
    base.update(over)
    return ctx(ids=(164, 24), **base)


def order(slots, x=None, y=None, targets=None):
    c = len(slots)
    f = lambda v: jnp.asarray([0.0] * c if v is None else v, jnp.float32)    # noqa: E731
    return CastOrder(jnp.asarray(slots, jnp.int32), jnp.asarray([-1] * c if targets is None else targets, jnp.int32),
                     f(x), f(y))


def place(**over):
    base = dict(x=[0.0, 150.0, 3100.0, 3200.0, 6000.0, -3000.0], y=[0.0] * 6)
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


def report(src, dst, final, dtype):
    p = D.packets(jnp.asarray([True]), jnp.asarray([src]), jnp.asarray([dst]), jnp.asarray([final]),
                  jnp.asarray([dtype]), jnp.asarray([D.BASIC_ATTACK]), cast_id=jnp.asarray([3]))
    f = jnp.asarray([final], jnp.float32)
    z = jnp.zeros_like(f)
    return Report(p, D.Resolved(f, z, z, f, jnp.zeros((1,), bool), jnp.zeros(6), jnp.zeros(6), None,
                                jnp.zeros(6, bool), jnp.zeros(6), jnp.zeros(6, bool), jnp.int32(0)), None)


# ---- data, passive -------------------------------------------------------------

def test_record_tables_and_windup():
    rec = champion("Camille")
    assert rec["slots"]["E"].endswith("/CamilleE")
    r = jnp.asarray([[1, 1, 1, 1], [5, 5, 5, 3]], jnp.int32)
    np.testing.assert_allclose(np.asarray(core.cooldown_row("Camille", r)), [[9, 12, 16, 140], [5, 10, 12, 90]])
    np.testing.assert_allclose(np.asarray(core.mana_row("Camille", r)), [[25, 50, 70, 100], [25, 70, 70, 100]])
    np.testing.assert_allclose(float(champion_base(["Camille"]).windup_percent[0]), 0.3 / 1.552, rtol=1e-5)


@pytest.mark.parametrize("level,shield,cd,conv", [(1, 0.10, 14, 0.40), (7, 0.15, 11, 0.64), (13, 0.20, 8, 0.88),
                                                  (19, 0.25, 8, 1.0)])
def test_passive_and_q_curves(level, shield, cd, conv):
    np.testing.assert_allclose(float(CM.shield_share(level)), shield, rtol=1e-5)
    np.testing.assert_allclose(float(CM.passive_cooldown(level)), cd, rtol=1e-5)
    np.testing.assert_allclose(float(CM.true_share(level)), conv, rtol=1e-5)


@pytest.mark.parametrize("magic", [False, True])
def test_adaptive_defenses_typed_shield_and_cooldown(magic):
    k = cctx()
    u = place()
    st = K.init(2, 6)
    if magic:                                                          # Jax has dealt magic damage to champions
        st, _ = K.on_damage(st, k, u, report(JAX, CAM, 300.0, D.MAGIC))
    st, o = K.on_hit(st, k, u, launch([True, False], [JAX, -1]))
    np.testing.assert_allclose(float(jnp.sum(o.shield.amount[0])), 0.10 * 1500.0, rtol=1e-5)
    kinds = np.asarray(o.shield.kind[0])[np.asarray(o.shield.amount[0]) > 0]
    assert kinds[0] % D.SOURCE_SHIFT == (D.SHIELD_MAGIC if magic else D.SHIELD_PHYSICAL)
    assert kinds[0] // D.SOURCE_SHIFT == D.SOURCE_KIT
    st, o = K.on_hit(st, at(k, 10.5), u, launch([True, False], [JAX, -1]))
    assert float(jnp.sum(o.shield.amount[0])) == 0.0                   # one at a time
    st, _, _ = run(st, k._replace(kit_shield=jnp.asarray([150.0, 0.0])), u, 10.0, 2.1)
    np.testing.assert_allclose(float(st.camille.p_ready_at[0]), 12.0 + 14.0, atol=2 * DT)


# ---- Q -------------------------------------------------------------------------

def test_precision_protocol_first_and_ramped_recast():
    k = cctx(level=jnp.full((2,), 10, jnp.int32))
    u = place()
    st, a, _ = tick(K.init(2, 6), k, u, 10.0, order([0, -1]))
    assert bool(a.attack_reset[0]) and float(a.mana_cost[0]) == 25.0 and not bool(a.cooldown_start[0, 0])
    mods = K.attack_mods(st, at(k, 10.0))
    assert float(mods.extra_range[0]) == 50.0 and bool(mods.cannot_crit[0]) and bool(mods.uncancellable[0])
    st, o = K.on_attack(st, at(k, 10.2), u, launch([True, False], [JAX, -1]))
    assert bool(o.cooldown_start[0, 0])
    st, o = K.on_hit(st, at(k, 10.2), u, launch([True, False], [JAX, -1]))
    np.testing.assert_allclose(valid(o.packets)["raw"][0], 0.20 * 130.0, rtol=1e-5)
    np.testing.assert_allclose(float(K.stats(st, at(k, 10.5)).percent_move_speed[0]), 0.25, rtol=1e-6)
    # Recast 1.6 s later: x2 bonus, 40% + 4%/level (76% at level 10) of the attack as true damage.
    st, a, _ = tick(st, k, u, 11.8, order([0, -1]))
    assert bool(a.cast_started[0]) and float(a.mana_cost[0]) == 0.0
    np.testing.assert_allclose(float(K.attack_mods(st, at(k, 11.8)).true_split[0]), 0.76, rtol=1e-5)
    st, _ = K.on_attack(st, at(k, 11.9), u, launch([True, False], [JAX, -1]))
    st, o = K.on_hit(st, at(k, 11.9), u, launch([True, False], [JAX, -1]))
    p = valid(o.packets)
    bonus = 2 * 0.20 * 130.0
    got = {int(d): float(r) for d, r in zip(p["dtype"], p["raw"])}
    np.testing.assert_allclose([got[D.PHYSICAL], got[D.TRUE]], [bonus * 0.24, bonus * 0.76], rtol=1e-4)


# ---- W -------------------------------------------------------------------------

def test_tactical_sweep_inner_outer_slow_and_heal():
    k = cctx()
    u = place(x=[0.0, 500.0, 200.0, 3200.0, 6000.0, -3000.0])         # Jax in the outer half, a minion inside
    st, a, _ = tick(K.init(2, 6), k, u, 10.0, order([1, -1], x=[1000.0, 0.0]))
    assert float(a.mana_cost[0]) == 50.0 and bool(a.cooldown_start[0, 1])
    assert bool(K.ghosted(st, at(k, 10.5))[0]) and bool(K.attack_mods(st, at(k, 10.5)).cannot_attack[0])
    st, outs, t = run(st, k, u, 10.0, 1.2)
    hits = fired(outs)
    assert len(hits) == 1 and hits[0][0] - 10.0 == pytest.approx(1.1, abs=1.5 * DT)
    p, b = hits[0][1], hits[0][2]
    base = 60.0 + 0.6 * 30.0
    jax_raw = sorted(p["raw"][p["dst"] == JAX])
    np.testing.assert_allclose(jax_raw, sorted([base, (0.07 + 30.0 / 40.0 / 100.0) * 1600.0]), rtol=1e-5)
    np.testing.assert_allclose(p["raw"][p["dst"] == MINION_RED], [base], rtol=1e-5)
    assert float(b.cc.slow[0, JAX]) == pytest.approx(0.8) and float(b.cc.slow[0, MINION_RED]) == 0.0
    outer_id = int(p["cast_id"][(p["dst"] == JAX) & (p["raw"] != base)][0])
    rep = Report(D.packets(jnp.asarray([True]), jnp.asarray([CAM]), jnp.asarray([JAX]), jnp.asarray([90.0]),
                           jnp.asarray([D.PHYSICAL]), jnp.asarray([D.TAG_ACTIVE_SPELL]), cast_id=jnp.asarray([outer_id])),
                 D.Resolved(jnp.asarray([90.0]), jnp.zeros(1), jnp.zeros(1), jnp.asarray([90.0]), jnp.zeros(1, bool),
                            jnp.zeros(6), jnp.zeros(6), None, jnp.zeros(6, bool), jnp.zeros(6), jnp.zeros(6, bool),
                            jnp.int32(0)), None)
    _, h = K.on_damage(st, at(k, hits[0][0]), u, rep)
    np.testing.assert_allclose(float(h.heal[0]), 90.0, rtol=1e-5)


def test_hookshot_without_a_wall_misses():
    k = cctx()
    st, a, _ = tick(K.init(2, 6), k, place(), 10.0, order([2, -1], x=[1000.0, 0.0]))
    assert bool(a.cast_started[0]) and bool(a.cooldown_start[0, 2]) and int(st.camille.e_stage[0]) == 0


# ---- R -------------------------------------------------------------------------

def test_hextech_ultimatum_leap_zone_leash_and_on_hit():
    k = cctx()
    u = place(x=[0.0, 400.0, 300.0, 3200.0, 6000.0, -3000.0])
    st, a, _ = tick(K.init(2, 6), k, u, 10.0, order([3, -1], targets=[JAX, -1]))
    assert bool(a.cooldown_start[0, 3]) and float(a.mana_cost[0]) == 100.0
    assert bool(a.dash.active[0]) and int(a.dash.target[0]) == JAX and float(a.dash.speed[0]) == pytest.approx(800.0)
    assert bool(K.untargetable(st, k.champion_id, jnp.float32(10.3))[0])
    landed = k._replace(x=jnp.asarray([330.0, 400.0]))
    st, outs, t = run(st, landed, u, 10.0, 0.6)
    act, cx, _, rad = K.leash(st, at(k, t), 6)
    assert bool(act[JAX]) and float(cx[JAX]) == pytest.approx(330.0) and float(rad[JAX]) == pytest.approx(425.0)
    knocked = [b for _, b in outs if float(b.cc.knockup[0, MINION_RED]) > 0]
    assert len(knocked) == 1
    st, o = K.on_hit(st, at(k, t), u, launch([True, False], [JAX, -1]))
    np.testing.assert_allclose(valid(o.packets)["raw"][-1], 0.04 * 1000.0, rtol=1e-5)
    assert not bool(K.untargetable(st, k.champion_id, jnp.float32(t))[0])
