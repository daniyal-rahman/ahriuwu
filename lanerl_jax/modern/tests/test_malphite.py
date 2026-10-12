"""Malphite 26.19 kit on the modern packet contract (champions, MAL.* in docs/modern/CHAMPIONS.md)."""
from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.modern import champions as K
from lanerl_jax.modern.champions import core
from lanerl_jax.modern.champions import malphite as MP
from lanerl_jax.modern.core import damage as D
from lanerl_jax.modern.core.types import KIND_CHAMPION, KIND_MINION, KIND_MONSTER, CastOrder
from lanerl_jax.modern.data.champions import champion
from lanerl_jax.modern.items.effects.core import Kills, Report
from lanerl_jax.modern.tests.test_champions import DT, at, ctx, jv, launch, units, valid

MAL, JAX, MINION_RED, MINION_RED2, TURRET_RED, MINION_BLUE = range(6)


def mctx(**over):
    over.setdefault("move_speed", jnp.full((2,), 335.0, jnp.float32))
    return ctx(ids=(54, 24), **over)


def order(slots, targets=None, x=None, y=None):
    c = len(slots)
    f = lambda v: jnp.asarray([0.0] * c if v is None else v, jnp.float32)    # noqa: E731
    return CastOrder(jnp.asarray(slots, jnp.int32), jnp.asarray([-1] * c if targets is None else targets, jnp.int32),
                     f(x), f(y))


def _report(dst, health_loss):
    p = D.packets(jnp.ones((1,), bool), jnp.asarray([JAX]), jnp.asarray([dst]), jnp.asarray([50.0]), D.PHYSICAL,
                  D.BASIC_ATTACK)
    return Report(p, SimpleNamespace(health_loss=jnp.asarray([health_loss], jnp.float32)), jnp.zeros(6))


def test_record_and_tables():
    rec = champion("Malphite")
    assert rec["slots"]["Q"].endswith("/SeismicShard") and rec["slots"]["Passive"].endswith("/MalphiteShield")
    r = jnp.asarray([[1, 1, 1, 1], [5, 5, 5, 3]], jnp.int32)
    np.testing.assert_allclose(np.asarray(core.cooldown_row("Malphite", r)), [[8, 10, 7, 130], [8, 8, 7, 100]])
    np.testing.assert_allclose(np.asarray(core.mana_row("Malphite", r)), [[70, 30, 50, 100], [90, 50, 70, 100]])


@pytest.mark.parametrize("level,cd", [(1, 8.0), (6, 8.0), (7, 7.0), (12, 7.0), (13, 6.0), (18, 6.0)])
def test_granite_shield_delay(level, cd):
    np.testing.assert_allclose(float(MP.shield_cooldown(jnp.int32(level))), cd)


def test_granite_shield_grants_refills_and_waits_after_health_loss():
    k = mctx(level=jnp.full((2,), 1, jnp.int32))
    u = units()
    st, out = K.periodic(K.init(2, 6), k, u)                                   # never hurt: shield at once
    np.testing.assert_allclose(float(out.shield.amount[0].sum()), 0.1 * 1500.0, rtol=1e-6)
    code = int(out.shield.kind[0][np.asarray(out.shield.amount[0]) > 0][0])
    assert code // D.SOURCE_SHIFT == D.SOURCE_KIT and code % D.SOURCE_SHIFT == D.SHIELD_ALL
    full = at(k, 10.0)._replace(kit_shield=jnp.asarray([150.0, 0.0], jnp.float32))
    _, out = K.periodic(st, full, u)
    assert float(out.shield.amount[0].sum()) == 0.0                            # already full
    st, _ = K.on_damage(st, k, u, _report(MAL, 30.0))                          # health lost at t=10
    part = lambda t: at(k, t)._replace(kit_shield=jnp.asarray([60.0, 0.0], jnp.float32))     # noqa: E731
    _, out = K.periodic(st, part(17.9), u)
    assert float(out.shield.amount[0].sum()) == 0.0
    _, out = K.periodic(st, part(18.0), u)
    np.testing.assert_allclose(float(out.shield.amount[0].sum()), 150.0, rtol=1e-6)
    st2, _ = K.on_damage(st, at(k, 12.0), u, _report(MAL, 0.0))               # absorbed: the clock keeps running
    assert float(st2.malphite.last_hurt[0]) == pytest.approx(10.0)


def test_sourced_grant_replaces_the_previous_kit_shield():
    sh = D.init_shields(3)
    sh = D.grant_shield(sh, 0, 80.0, D.SHIELD_ALL, 0.0, jnp.inf)                       # an item shield
    sh = D.grant_shield(sh, 0, 150.0, D.SHIELD_ALL, 0.0, jnp.inf, source=D.SOURCE_KIT)
    sh = sh._replace(amount=jnp.where(sh.source == D.SOURCE_KIT, 40.0, sh.amount))      # damaged to 40
    sh = D.grant_shield(sh, 0, 150.0, D.SHIELD_ALL, 1.0, jnp.inf, source=D.SOURCE_KIT)
    np.testing.assert_allclose(float(D.total_shield(sh, 1.0)[0]), 80.0 + 150.0)


def test_thunderclap_passive_armor_triples_under_the_shield():
    for shield, mult in ((0.0, 1.0), (50.0, 3.0)):
        k = mctx(ranks=jnp.asarray([[1, 3, 1, 0], [1, 1, 1, 0]], jnp.int32),
                 kit_shield=jnp.asarray([shield, 0.0], jnp.float32))
        np.testing.assert_allclose(float(K.stats(K.init(2, 6), k).percent_armor[0]),
                                   mult * jv("Malphite", "W", "BonusArmorPassive", 3), rtol=1e-6)


@pytest.mark.parametrize("rank", [1, 3, 5])
def test_seismic_shard_damage_slow_and_speed_steal(rank):
    k = mctx(ranks=jnp.asarray([[rank, 1, 1, 0], [1, 1, 1, 0]], jnp.int32))
    u = units(x=[0.0, 600.0, 100.0, 250.0, 3000.0, -100.0], move_speed=[335.0, 350.0] + [325.0] * 4)
    st, out = K.cast(K.init(2, 6), k, u, order([0, -1], [JAX, -1]))
    assert bool(out.cast_started[0]) and bool(out.cooldown_start[0, 0])
    np.testing.assert_allclose(float(out.cast_lockout[0]), 0.25)
    land = 10.0 + 0.25 + 600.0 / 1200.0
    st, o = K.periodic(st, at(k, land - 2 * DT), u)
    assert not np.asarray(o.packets.valid).any()
    st, o = K.periodic(st, at(k, land), u)
    np.testing.assert_allclose(valid(o.packets)["raw"], [jv("Malphite", "Q", "BaseDamage", rank) + 0.6 * 20.0],
                               rtol=1e-6)
    slow = jv("Malphite", "Q", "SpeedSteal", rank) / 100.0
    np.testing.assert_allclose(float(o.cc.slow[0, JAX]), slow, rtol=1e-6)
    np.testing.assert_allclose(float(o.cc.slow_duration[0, JAX]), 3.0)
    np.testing.assert_allclose(float(K.stats(st, at(k, land + 1.0)).move_speed[0]), slow * 350.0, rtol=1e-5)
    assert float(K.stats(st, at(k, land + 3.1)).move_speed[0]) == 0.0


def test_seismic_shard_needs_an_enemy_unit_in_range():
    _, out = K.cast(K.init(2, 6), mctx(), units(), order([0, -1], [TURRET_RED, -1]))
    assert not bool(out.cast_started[0])
    far = units(x=[0.0, 800.0, 100.0, 250.0, 3000.0, -100.0])
    _, out = K.cast(K.init(2, 6), mctx(), far, order([0, -1], [JAX, -1]))
    assert not bool(out.cast_started[0])
    _, out = K.cast(K.init(2, 6), mctx(), units(), order([0, -1], [MINION_RED, -1]))
    assert bool(out.cast_started[0])


@pytest.mark.parametrize("rank", [1, 3, 5])
def test_thunderclap_empowered_attack_and_aftershocks(rank):
    k = mctx(ranks=jnp.asarray([[1, rank, 1, 0], [1, 1, 1, 0]], jnp.int32))
    # Jax at 200 (target, in the cone), minion 2 at (50, 200) outside it (76 deg), minion 3 at 250 in it.
    u = units(x=[0.0, 200.0, 50.0, 250.0, 3000.0, -100.0], y=[0.0, 0.0, 200.0, 20.0, 0.0, 0.0])
    st, out = K.cast(K.init(2, 6), k, u, order([1, -1]))
    assert bool(out.attack_reset[0]) and bool(out.cooldown_start[0, 1])
    np.testing.assert_allclose(float(out.mana_cost[0]), [30, 35, 40, 45, 50][rank - 1])
    m = K.attack_mods(st, k)
    assert float(m.extra_range[0]) == 50.0 and bool(m.uncancellable[0])
    st, out = K.on_hit(st, at(k, 10.3), u, launch([True, False], [JAX, -1]))
    p = valid(out.packets)
    bonus = jv("Malphite", "W", "ThunderclapBaseDamage", rank) + 0.2 * 20.0 + 0.15 * 60.0
    splash = jv("Malphite", "W", "ThunderclapSplashDamage", rank) + 0.3 * 20.0 + 0.15 * 60.0
    got = sorted(zip(p["dst"].tolist(), p["raw"].tolist()))
    assert [d for d, _ in got] == [JAX, JAX, MINION_RED2]
    np.testing.assert_allclose(sorted(r for d, r in got if d == JAX), sorted([bonus, splash]), rtol=1e-5)
    # Later attacks within 5 s: aftershock only.
    st, out = K.on_hit(st, at(k, 12.0), u, launch([True, False], [JAX, -1]))
    np.testing.assert_allclose(sorted(valid(out.packets)["raw"].tolist()), [splash, splash], rtol=1e-5)
    _, out = K.on_hit(st, at(k, 15.1), u, launch([True, False], [JAX, -1]))
    assert not np.asarray(out.packets.valid).any()


def test_thunderclap_monster_bonus():
    u = units(kind=[KIND_CHAMPION, KIND_CHAMPION, KIND_MONSTER, KIND_MINION, 3, KIND_MINION],
              x=[0.0, 3000.0, 150.0, 3000.0, 3000.0, -100.0])
    k = mctx()
    st, _ = K.cast(K.init(2, 6), k, u, order([1, -1]))
    _, out = K.on_hit(st, k, u, launch([True, False], [MINION_RED, -1]))
    bonus = jv("Malphite", "W", "ThunderclapBaseDamage", 3) + 0.2 * 20.0 + 0.15 * 60.0
    splash = jv("Malphite", "W", "ThunderclapSplashDamage", 3) + 0.3 * 20.0 + 0.15 * 60.0
    np.testing.assert_allclose(sorted(valid(out.packets)["raw"].tolist()), sorted([1.8 * bonus, 1.8 * splash]),
                               rtol=1e-5)


@pytest.mark.parametrize("rank", [1, 3, 5])
def test_ground_slam_damage_and_cripple(rank):
    k = mctx(ranks=jnp.asarray([[1, 1, rank, 0], [1, 1, 1, 0]], jnp.int32))
    u = units()
    st, out = K.cast(K.init(2, 6), k, u, order([2, -1]))
    np.testing.assert_allclose(float(out.cast_lockout[0]), 0.2419, rtol=1e-4)
    st, o = K.periodic(st, at(k, 10.2419), u)
    p = valid(o.packets)
    assert sorted(p["dst"].tolist()) == [JAX, MINION_RED, MINION_RED2]
    np.testing.assert_allclose(p["raw"], jv("Malphite", "E", "BaseDamage", rank) + 0.4 * 60.0 + 0.6 * 20.0,
                               rtol=1e-6)
    db = K.debuffs(st, at(k, 12.0), u)
    np.testing.assert_allclose(float(db.attack_speed_cripple[JAX]), jv("Malphite", "E", "ASReduction", rank) / 100.0,
                               rtol=1e-6)
    assert float(K.debuffs(st, at(k, 13.4), u).attack_speed_cripple[JAX]) == 0.0


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_unstoppable_force(rank):
    k = mctx(ranks=jnp.asarray([[1, 1, 1, rank], [1, 1, 1, 0]], jnp.int32))
    # Cursor 2000 away: dash 1000; Jax at 900 and red minion 3 at 1250 (inside 325 + 48) get knocked up.
    u = units(x=[0.0, 900.0, 300.0, 1250.0, 3000.0, -100.0])
    st, out = K.cast(K.init(2, 6), k, u, order([3, -1], x=[2000.0, 0.0], y=[0.0, 0.0]))
    assert bool(out.dash.active[0]) and bool(out.cooldown_start[0, 3])
    np.testing.assert_allclose(float(out.dash.to_x[0]), 1000.0, atol=1e-3)
    np.testing.assert_allclose(float(out.dash.speed[0]), 1500.0 + 335.0)
    land = 10.0 + 1000.0 / 1835.0
    st, o = K.periodic(st, at(k, land)._replace(x=jnp.asarray([1000.0, 900.0], jnp.float32)), u)
    p = valid(o.packets)
    assert sorted(p["dst"].tolist()) == [JAX, MINION_RED2]
    np.testing.assert_allclose(p["raw"], jv("Malphite", "R", "BaseDamage", rank) + 0.9 * 20.0, rtol=1e-6)
    np.testing.assert_allclose(float(o.cc.knockup[0, JAX]), 1.5)
    assert float(o.cc.knockup[0, MINION_RED]) == 0.0


def test_jit_all_hooks():
    k = mctx()
    u = units()
    rep = Report(D.packets(jnp.ones(2, bool), jnp.asarray([1, 2]), 0, 50.0, D.PHYSICAL, D.BASIC_ATTACK),
                 None, jnp.zeros(6))
    kills = Kills(jnp.zeros(2), jnp.zeros(2), jnp.zeros(2), jnp.zeros(2, bool), jnp.zeros((2, 6), bool))

    @jax.jit
    def tick(st, k):
        st, o1 = K.cast(st, k, u, order([2, 2], x=[300.0, 0.0], y=[0.0, 0.0]))
        st, o2 = K.periodic(st, k, u)
        st, o3 = K.on_attack(st, k, u, launch([True, True], [1, 0]))
        st, o4 = K.on_hit(st, k, u, launch([True, True], [1, 0]))
        st, o5 = K.on_damage(st, k, u, rep)
        st = K.on_takedown(st, k, u, kills)
        return st, (o1, o2, o3, o4, o5), K.stats(st, k), K.attack_mods(st, k), K.debuffs(st, k, u)

    st1, *_ = tick(K.init(2, 6), k)
    st2, *_ = tick(st1, at(k, 10.0 + DT))
    for a, b in zip(jax.tree_util.tree_leaves(st1), jax.tree_util.tree_leaves(st2)):
        assert a.dtype == b.dtype and a.shape == b.shape
    assert bool(st1.malphite.e_pending[0]) and bool(st1.jax.e_on[1])
