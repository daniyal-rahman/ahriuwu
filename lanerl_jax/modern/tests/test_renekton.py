"""Renekton 26.19 kit on the modern packet contract (champions, REN.* in docs/modern/CHAMPIONS.md)."""

from __future__ import annotations

import pytest as _pytest

_pytest.skip("Renekton is not registered until every rule is exact "
             "(docs/modern/CHAMPIONS.md, Pending)", allow_module_level=True)

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.modern import champions as K
from lanerl_jax.modern.champions import core
from lanerl_jax.modern.champions import renekton as R
from lanerl_jax.modern.core import damage as D
from lanerl_jax.modern.core.types import KIND_CHAMPION, KIND_MINION, KIND_MONSTER, CastOrder
from lanerl_jax.modern.data.champions import champion, cooldowns
from lanerl_jax.modern.items.effects.core import Kills, Report
from lanerl_jax.modern.tests.test_champions import DT, at, ctx, jv, launch, units, valid

REN, JAX, MINION_RED, MINION_RED2, TURRET_RED, MINION_BLUE = range(6)


def rctx(**over):
    over.setdefault("move_speed", jnp.full((2,), 345.0, jnp.float32))
    return ctx(ids=(58, 24), **over)


def order(slots, targets=None, x=None, y=None):
    c = len(slots)
    f = lambda v: jnp.asarray([0.0] * c if v is None else v, jnp.float32)    # noqa: E731
    return CastOrder(jnp.asarray(slots, jnp.int32), jnp.asarray([-1] * c if targets is None else targets, jnp.int32),
                     f(x), f(y))


def with_fury(st, v):
    return st._replace(renekton=st.renekton._replace(fury=st.renekton.fury.at[0].set(v)))


def fury(st):
    return float(st.renekton.fury[0])


# ---- data ----------------------------------------------------------------------

def test_record_and_tables():
    rec = champion("Renekton")
    assert rec["slots"]["Q"].endswith("/RenektonCleave") and rec["slots"]["R"].endswith("/RenektonReignOfTheTyrant")
    assert rec["slots"]["Passive"].endswith("/RenektonPredator")
    r = jnp.asarray([[1, 1, 1, 1], [5, 5, 5, 3]], jnp.int32)
    np.testing.assert_allclose(np.asarray(core.cooldown_row("Renekton", r)), [[7, 16, 16, 120], [7, 8, 10, 80]])
    np.testing.assert_allclose(np.asarray(core.mana_row("Renekton", r)), 0.0)
    assert cooldowns("Renekton", "W") == [16.0, 14.0, 12.0, 10.0, 8.0]          # 26.17


# ---- Q -------------------------------------------------------------------------

@pytest.mark.parametrize("rank", [1, 3, 5])
def test_q_cleave_heal_and_fury(rank):
    k = rctx(ranks=jnp.asarray([[rank, 1, 1, 0], [1, 1, 1, 0]], jnp.int32))
    u = units()                                         # Jax at 200, red minions at 100 and 250: all inside 400
    st, out = K.cast(K.init(2, 6), k, u, order([0, -1]))
    p = valid(out.packets)
    assert sorted(p["dst"].tolist()) == [JAX, MINION_RED, MINION_RED2]
    np.testing.assert_allclose(p["raw"], jv("Renekton", "Q", "BaseDamage", rank) + 30.0, rtol=1e-6)
    champ = jv("Renekton", "Q", "ChampHeal", rank) + 0.17 * 30.0
    minion = jv("Renekton", "Q", "BaseMinionHeal", rank) + 0.02 * 30.0
    np.testing.assert_allclose(float(out.heal[0]), min(champ + 2 * minion, jv("Renekton", "Q", "BasicHealCap", rank)),
                               rtol=1e-5)
    np.testing.assert_allclose(fury(st), 10.0 + 2 * 2.5)
    assert bool(out.cooldown_start[0, 0]) and float(out.mana_cost[0]) == 0.0
    m = K.attack_mods(st, k)
    assert bool(m.cannot_attack[0]) and not bool(K.attack_mods(st, at(k, 10.26)).cannot_attack[0])


def test_q_fury_cap_low_health_bonus_and_empowered():
    many = units(kind=[KIND_CHAMPION, KIND_CHAMPION] + [KIND_MINION] * 4, team=[0, 1, 1, 1, 1, 1],
                 x=[0.0, 200.0, 100.0, 250.0, 300.0, -100.0])
    k = rctx(ranks=jnp.asarray([[5, 1, 1, 0], [1, 1, 1, 0]], jnp.int32))
    st, _ = K.cast(K.init(2, 6), k, many, order([0, -1]))
    np.testing.assert_allclose(fury(st), 10.0 + 4 * 2.5)                 # 20, under the cap of 30
    low = rctx(hp=jnp.full((2,), 600.0, jnp.float32))                    # below 50% of 1500
    st, _ = K.cast(K.init(2, 6), low, units(), order([0, -1]))
    np.testing.assert_allclose(fury(st), 1.5 * 15.0)
    st, out = K.cast(with_fury(K.init(2, 6), 70.0), k, units(), order([0, -1]))
    p = valid(out.packets)
    np.testing.assert_allclose(p["raw"], jv("Renekton", "Q", "EmpoweredBaseDamage", 5) + 1.4 * 30.0, rtol=1e-6)
    np.testing.assert_allclose(fury(st), 20.0)                           # -50, no Fury from empowered casts
    heal = 3 * (jv("Renekton", "Q", "ChampHeal", 5) + 0.17 * 30.0 + 2 * (jv("Renekton", "Q", "BaseMinionHeal", 5)
                                                                        + 0.02 * 30.0))
    np.testing.assert_allclose(float(out.heal[0]), min(heal, jv("Renekton", "Q", "EmpoweredHealCap", 5)), rtol=1e-5)


def test_q_radius_grows_during_dominus():
    np.testing.assert_allclose(np.asarray(R.q_radius(jnp.asarray([False, True]))), [400.0, 480.0], rtol=1e-6)


# ---- W -------------------------------------------------------------------------

@pytest.mark.parametrize("rank", [1, 3, 5])
def test_w_two_strikes_and_stun(rank):
    k = rctx(ranks=jnp.asarray([[1, rank, 1, 0], [1, 1, 1, 0]], jnp.int32))
    u = units()
    st, out = K.cast(K.init(2, 6), k, u, order([1, -1]))
    assert bool(out.attack_reset[0]) and not bool(out.cooldown_start[0, 1])
    assert float(K.attack_mods(st, k).extra_range[0]) == 50.0
    st, out = K.on_hit(st, at(k, 10.4), u, launch([True, False], [JAX, -1]))
    p = valid(out.packets)
    exp = 2 * (jv("Renekton", "W", "BaseDamagePerHit", rank) + 0.75 * 130.0) - 130.0
    np.testing.assert_allclose(p["raw"], [exp], rtol=1e-5)
    np.testing.assert_allclose(float(out.cc.stun[0, JAX]), 0.75)
    assert bool(out.cooldown_start[0, 1]) and not bool(st.renekton.w_on[0])
    np.testing.assert_allclose(float(out.cast_lockout[0]), 0.525)
    np.testing.assert_allclose(fury(st), 2 * 5.0 + 10.0)
    assert not np.asarray(out.strip_shields).any()
    np.testing.assert_allclose(float(out.base_cooldown[0, 1]), cooldowns("Renekton", "W")[rank - 1])


def test_w_empowered_three_strikes_strip_shields_no_fury():
    k = rctx()
    u = units()
    st, _ = K.cast(with_fury(K.init(2, 6), 60.0), k, u, order([1, -1]))
    np.testing.assert_allclose(fury(st), 10.0)
    st, out = K.on_hit(st, k, u, launch([True, False], [JAX, -1]))
    np.testing.assert_allclose(valid(out.packets)["raw"], [3 * (jv("Renekton", "W", "BaseDamagePerHit", 3)
                                                                + 0.75 * 130.0) - 130.0], rtol=1e-5)
    np.testing.assert_allclose(float(out.cc.stun[0, JAX]), 1.5)
    assert bool(out.strip_shields[0, JAX]) and int(np.asarray(out.strip_shields).sum()) == 1
    np.testing.assert_allclose(fury(st), 10.0)
    mon = units(kind=[KIND_CHAMPION, KIND_CHAMPION, KIND_MONSTER, KIND_MINION, 3, KIND_MINION])
    st, _ = K.cast(with_fury(K.init(2, 6), 60.0), k, mon, order([1, -1]))
    _, out = K.on_hit(st, k, mon, launch([True, False], [MINION_RED, -1]))
    assert not np.asarray(out.strip_shields).any()                         # never on monsters


def test_w_ignores_structures_and_expires():
    k = rctx()
    u = units()
    st, _ = K.cast(K.init(2, 6), k, u, order([1, -1]))
    st2, out = K.on_hit(st, k, u, launch([True, False], [TURRET_RED, -1]))
    assert bool(st2.renekton.w_on[0]) and not np.asarray(out.packets.valid).any() and fury(st2) == 0.0
    st3, out = K.periodic(st, at(k, 16.0), u)
    assert not bool(st3.renekton.w_on[0]) and bool(out.cooldown_start[0, 1])


def test_basic_attack_fury():
    k = rctx()
    st, _ = K.on_hit(K.init(2, 6), k, units(), launch([True, False], [MINION_RED, -1]))
    np.testing.assert_allclose(fury(st), 5.0)
    st, _ = K.on_hit(st, k, units(), launch([True, False], [TURRET_RED, -1]))
    np.testing.assert_allclose(fury(st), 5.0)                               # structures give none
    st = with_fury(K.init(2, 6), 99.0)
    st, _ = K.on_hit(st, k, units(), launch([True, False], [MINION_RED, -1]))
    np.testing.assert_allclose(fury(st), 100.0)


def test_fury_decays_after_twelve_seconds_out_of_combat():
    k = rctx()
    st, _ = K.on_hit(with_fury(K.init(2, 6), 40.0), k, units(), launch([True, False], [MINION_RED, -1]))
    st2, _ = K.periodic(st, at(k, 10.0 + 11.9), units())
    np.testing.assert_allclose(fury(st2), 45.0)
    st3 = st
    for i in range(int(round(1.0 / DT))):
        st3, _ = K.periodic(st3, at(k, 22.0 + i * DT), units())
    np.testing.assert_allclose(fury(st3), 45.0 - 4.0, atol=1.0)            # -1 per 0.25 s


# ---- E -------------------------------------------------------------------------

def test_e_slice_dash_hits_path_and_opens_dice():
    k = rctx(ranks=jnp.asarray([[1, 1, 3, 0], [1, 1, 1, 0]], jnp.int32))
    u = units(x=[0.0, 2000.0, 300.0, 600.0, 3000.0, 100.0], y=[0.0, 0.0, 30.0, 0.0, 0.0, 0.0])
    st, out = K.cast(K.init(2, 6), k, u, order([2, -1], x=[1000.0, 0.0], y=[0.0, 0.0]))
    assert bool(out.dash.active[0]) and int(out.dash.target[0]) == -1
    np.testing.assert_allclose([float(out.dash.to_x[0]), float(out.dash.to_y[0])], [450.0, 0.0], atol=1e-3)
    np.testing.assert_allclose(float(out.dash.speed[0]), 750.0 + 345.0)
    assert bool(out.cooldown_start[0, 2])
    hits, t, x = [], 10.0, 0.0
    while bool(st.renekton.e_dash[0]):
        t += DT
        x = min(450.0, x + 1095.0 * DT)
        st, o = K.periodic(st, at(k, t)._replace(x=jnp.asarray([x, 2000.0], jnp.float32)), u)
        hits += valid(o.packets)["dst"].tolist()
    assert hits == [MINION_RED]                                         # minion 3 at 600 is past the 450 dash
    np.testing.assert_allclose(fury(st), 2.0)
    assert float(st.renekton.e_dice_until[0]) == pytest.approx(t + 4.0, abs=1e-3)
    # Dice: recast inside the window, regardless of the cooldown; no new cooldown.
    k2 = at(k, t + 1.0)._replace(x=jnp.asarray([450.0, 2000.0], jnp.float32),
                                 cooldowns=jnp.asarray([[0, 0, 10.0, 0], [0, 0, 0, 0]], jnp.float32))
    st, out = K.cast(st, k2, u, order([2, -1], x=[0.0, 0.0], y=[0.0, 0.0]))
    assert bool(out.dash.active[0]) and not bool(out.cooldown_start[0, 2])
    np.testing.assert_allclose(float(out.dash.to_x[0]), 0.0, atol=1e-3)
    _, again = K.cast(st, at(k2, t + 1.5), u, order([2, -1]))
    assert not bool(again.cast_started[0])


def test_e_slice_without_a_hit_has_no_dice_and_empowered_dice_shreds():
    k = rctx()
    empty = units(x=[0.0, 3000.0, 3000.0, 3000.0, 3000.0, 100.0])
    st, _ = K.cast(K.init(2, 6), k, empty, order([2, -1], x=[1000.0, 0.0], y=[0.0, 0.0]))
    st, _ = K.periodic(st, at(k, 10.5)._replace(x=jnp.asarray([450.0, 3000.0], jnp.float32)), empty)
    assert not bool(st.renekton.e_dash[0]) and float(st.renekton.e_dice_until[0]) < 0
    # Empowered Dice: consumes 50 Fury, bonus damage and 4 s armor reduction.
    st = st._replace(renekton=st.renekton._replace(e_dice_until=st.renekton.e_dice_until.at[0].set(20.0),
                                                   fury=st.renekton.fury.at[0].set(55.0)))
    u = units(x=[450.0, 300.0, 3000.0, 3000.0, 3000.0, 100.0])
    kk = at(k, 11.0)._replace(x=jnp.asarray([450.0, 300.0], jnp.float32))
    st, _ = K.cast(st, kk, u, order([2, -1], x=[0.0, 0.0], y=[0.0, 0.0]))
    np.testing.assert_allclose(fury(st), 5.0)
    st, o = K.periodic(st, at(kk, 11.0 + 0.2)._replace(x=jnp.asarray([230.0, 300.0], jnp.float32)), u)
    p = valid(o.packets)
    np.testing.assert_allclose(p["raw"], [jv("Renekton", "E", "BaseDamage", 3) + 0.9 * 30.0
                                          + jv("Renekton", "E", "EnragedBaseDamage", 3) + 1.35 * 30.0], rtol=1e-5)
    np.testing.assert_allclose(float(K.debuffs(st, kk, u).percent_armor_reduction[JAX]),
                               jv("Renekton", "E", "EnragedArmorShred", 3) / 100.0, rtol=1e-6)
    assert float(K.debuffs(st, at(kk, 15.5), u).percent_armor_reduction[JAX]) == 0.0
    np.testing.assert_allclose(fury(st), 5.0)                                 # empowered: no Fury


def test_e_cannot_be_cast_rooted_or_during_q_lockout():
    st, out = K.cast(K.init(2, 6), rctx(rooted=jnp.asarray([True, False])), units(), order([2, -1], x=[500.0, 0.0]))
    assert not bool(out.cast_started[0])
    st, _ = K.cast(K.init(2, 6), rctx(), units(), order([0, -1]))
    _, out = K.cast(st, at(rctx(), 10.1), units(), order([2, -1], x=[500.0, 0.0]))
    assert not bool(out.cast_started[0])


# ---- R -------------------------------------------------------------------------

@pytest.mark.parametrize("rank", [1, 2, 3])
def test_r_dominus(rank):
    k = rctx(ranks=jnp.asarray([[1, 1, 1, rank], [1, 1, 1, 0]], jnp.int32))
    u = units()
    st, out = K.cast(K.init(2, 6), k, u, order([3, -1]))
    assert bool(out.cooldown_start[0, 3]) and float(out.cast_lockout[0]) == pytest.approx(0.25)
    st, _ = K.periodic(st, at(k, 10.25), u)
    assert bool(st.renekton.r_on[0])
    np.testing.assert_allclose(fury(st), 20.0)
    np.testing.assert_allclose(float(K.stats(st, k).health[0]), jv("Renekton", "R", "HealthGain", rank))
    assert float(K.attack_mods(st, k).extra_range[0]) == 25.0
    ticks, t = 0, 10.25
    for _ in range(int(round(15.5 / DT))):
        t += DT
        st, o = K.periodic(st, at(k, t), u)
        p = valid(o.packets)
        if len(p["raw"]):
            ticks += 1
            np.testing.assert_allclose(p["raw"], (jv("Renekton", "R", "AuraDamagePerSecond", rank)
                                                  + 0.1 * (20.0 + 30.0)) * 0.5, rtol=1e-5)
            assert sorted(p["dst"].tolist()) == [JAX, MINION_RED, MINION_RED2]
    assert ticks == 30 and not bool(st.renekton.r_on[0])
    np.testing.assert_allclose(fury(st), 75.0, atol=0.2)                   # +5 per second up to 75


def test_r_fury_generation_stops_at_75_but_keeps_more():
    k = rctx()
    st, _ = K.cast(with_fury(K.init(2, 6), 70.0), k, units(), order([3, -1]))
    st, _ = K.periodic(st, at(k, 10.25), units())
    np.testing.assert_allclose(fury(st), 90.0)
    st, _ = K.periodic(st, at(k, 11.0), units())
    np.testing.assert_allclose(fury(st), 90.0)


def test_resource_fraction_and_jit():
    k = rctx()
    u = units()
    st = with_fury(K.init(2, 6), 40.0)
    np.testing.assert_allclose(np.asarray(K.resource_fraction(st, k.champion_id, jnp.asarray([0.3, 0.6])))[[0, 1]], [0.4, 0.6])
    rep = Report(D.packets(jnp.ones(2, bool), jnp.asarray([0, 2]), 1, 50.0, D.PHYSICAL, D.BASIC_ATTACK),
                 None, jnp.zeros(6))
    kills = Kills(jnp.zeros(2), jnp.zeros(2), jnp.zeros(2), jnp.zeros(2, bool), jnp.zeros((2, 6), bool))

    @jax.jit
    def tick(st, k):
        st, o1 = K.cast(st, k, u, order([0, 2], x=[300.0, 0.0], y=[0.0, 0.0]))
        st, o2 = K.periodic(st, k, u)
        st, o3 = K.on_attack(st, k, u, launch([True, True], [1, 0]))
        st, o4 = K.on_hit(st, k, u, launch([True, True], [1, 0]))
        st, o5 = K.on_damage(st, k, u, rep)
        st = K.on_takedown(st, k, u, kills)
        return st, (o1, o2, o3, o4, o5), K.stats(st, k), K.attack_mods(st, k), K.debuffs(st, k, u)

    st1, *_ = tick(st, k)
    st2, *_ = tick(st1, at(k, 10.0 + DT))
    for a, b in zip(jax.tree_util.tree_leaves(st1), jax.tree_util.tree_leaves(st2)):
        assert a.dtype == b.dtype and a.shape == b.shape
    assert bool(st1.jax.e_on[1]) and not bool(st1.renekton.e_dash[1])
