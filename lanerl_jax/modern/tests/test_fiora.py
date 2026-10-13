"""Fiora 26.19 kit on the modern packet contract (champions, FIO.* in docs/modern/CHAMPIONS.md)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.modern import champions as K
from lanerl_jax.modern.champions import core
from lanerl_jax.modern.champions import fiora as FI
from lanerl_jax.modern.core import damage as D
from lanerl_jax.modern.core.stat_pipeline import champion_base
from lanerl_jax.modern.core.types import KIND_CHAMPION, KIND_MINION, CastOrder
from lanerl_jax.modern.data.champions import champion
from lanerl_jax.modern.items.effects.core import Report
from lanerl_jax.modern.tests.test_champions import DT, at, ctx, launch, units, valid

FIO, JAX, MINION_RED, MINION_RED2, TURRET_RED, MINION_BLUE = range(6)


def fctx(**over):
    base = dict(rand=jnp.asarray([0.25, 0.75], jnp.float32), move_speed=jnp.asarray([345.0, 345.0], jnp.float32),
                dashing=jnp.zeros((2,), bool))
    base.update(over)
    return ctx(ids=(114, 24), **base)


def order(slots, x=None, y=None, targets=None):
    c = len(slots)
    f = lambda v: jnp.asarray([0.0] * c if v is None else v, jnp.float32)    # noqa: E731
    return CastOrder(jnp.asarray(slots, jnp.int32), jnp.asarray([-1] * c if targets is None else targets, jnp.int32),
                     f(x), f(y))


def place(**over):
    """Fiora at the origin, Jax 300 north unless placed; minions and the turret far away."""
    base = dict(x=[0.0, 0.0, 3100.0, 3200.0, 6000.0, -3000.0], y=[0.0, 300.0, 0.0, 0.0, 0.0, 0.0])
    base.update(over)
    return units(**base)


def tick(st, k, u, t, o=None):
    k = at(k, t)
    st, a = K.cast(st, k, u, o if o is not None else order([-1, -1]))
    st, b = K.periodic(st, k, u)
    return st, a, b


def report(src, dst, final=100.0, flags=D.BASIC_ATTACK, cast_id=7):
    p = D.packets(jnp.asarray([True]), jnp.asarray([src]), jnp.asarray([dst]), jnp.asarray([final]),
                  jnp.asarray([D.PHYSICAL]), jnp.asarray([flags]), cast_id=jnp.asarray([cast_id]))
    f = jnp.asarray([final], jnp.float32)
    z = jnp.zeros_like(f)
    return Report(p, D.Resolved(f, z, z, f, jnp.zeros((1,), bool), jnp.zeros(6), jnp.zeros(6), None,
                                jnp.zeros(6, bool), jnp.zeros(6), jnp.zeros(6, bool), jnp.int32(0)), None)


# ---- data ----------------------------------------------------------------------

def test_record_tables_and_windup():
    rec = champion("Fiora")
    assert rec["slots"]["W"].endswith("/FioraW") and rec["character"]["acquisitionRange"] == 600.0
    r = jnp.asarray([[1, 1, 1, 1], [5, 5, 5, 3]], jnp.int32)
    np.testing.assert_allclose(np.asarray(core.cooldown_row("Fiora", r)), [[13, 24, 11, 110], [6, 16, 7, 70]])
    np.testing.assert_allclose(np.asarray(core.mana_row("Fiora", r)), [[20, 50, 40, 100], [20, 50, 40, 100]])
    np.testing.assert_allclose(float(champion_base(["Fiora"]).windup_percent[0]), 0.2 / 1.45, rtol=1e-5)


def test_passive_numbers_and_geometry():
    np.testing.assert_allclose(float(FI.vital_damage(100.0, 2000.0)), (0.03 + 0.04) * 2000.0, rtol=1e-5)
    np.testing.assert_allclose(float(FI.by_level(FI.P_HEAL, 1)), 35.0)
    np.testing.assert_allclose(float(FI.by_level(FI.P_HEAL, 18)), 100.0)
    # A source due north of the target is inside the north arc (45 degrees either side), not the east one.
    assert bool(FI.facing(jnp.float32(30.0), jnp.float32(100.0), jnp.int32(FI.N)))
    assert not bool(FI.facing(jnp.float32(101.0), jnp.float32(100.0), jnp.int32(FI.N)))
    assert bool(FI.facing(jnp.float32(-100.0), jnp.float32(10.0), jnp.int32(FI.W)))


@pytest.mark.parametrize("ms,speed", [(345.0, 1100.0), (445.0, 1450.0), (545.0, 1600.0), (245.0, 850.0),
                                      (295.0, 950.0)])
def test_lunge_speed(ms, speed):
    np.testing.assert_allclose(float(FI.lunge_speed(jnp.float32(ms))), speed, rtol=1e-6)


# ---- vitals --------------------------------------------------------------------

def test_vital_lifecycle_strike_heal_damage_and_next_direction():
    k = fctx(level=jnp.full((2,), 18, jnp.int32))
    u = place()
    st, _, _ = tick(K.init(2, 6), k, u, 10.0)
    assert int(st.fiora.v_dir[0, JAX]) == FI.S                         # first group south/west; rand 0.25 -> south
    np.testing.assert_allclose(float(st.fiora.v_from[0, JAX]), 10.0 + 1.75, rtol=1e-6)
    # Fiora is south of Jax (Jax at +300 y): a hit before the vital opens does nothing.
    st, h = K.on_damage(st, at(k, 11.0), u, report(FIO, JAX))
    assert float(h.heal[0]) == 0.0 and not bool(st.fiora.v_proc[0, JAX])
    st, h = K.on_damage(st, at(k, 12.0), u, report(FIO, JAX))
    np.testing.assert_allclose(float(h.heal[0]), 100.0, rtol=1e-6)
    np.testing.assert_allclose(float(K.stats(st, at(k, 13.0)).percent_move_speed[0]), 0.3 * 0.5, rtol=1e-5)  # R rank 1
    st, _, b = tick(st, k, u, 12.0 + DT)
    p = valid(b.packets)
    np.testing.assert_allclose(p["raw"], [FI.vital_damage(30.0, 1600.0)], rtol=1e-5)
    assert p["dtype"][0] == D.TRUE and p["dst"][0] == JAX
    assert int(st.fiora.v_dir[0, JAX]) in (FI.N, FI.E)                 # the next vital is north or east


def test_vital_wrong_side_and_reactive_damage_do_not_strike():
    k = fctx()
    u = place()
    st, _, _ = tick(K.init(2, 6), k, u, 10.0)
    k2 = at(k, 12.0)._replace(x=jnp.asarray([0.0, 0.0]), y=jnp.asarray([600.0, 0.0]))     # Fiora north of Jax
    st2, h = K.on_damage(st, k2, u, report(FIO, JAX))
    assert float(h.heal[0]) == 0.0
    st2, h = K.on_damage(st, at(k, 12.0), u, report(FIO, JAX, flags=D.PROP_REACTIVE))
    assert float(h.heal[0]) == 0.0


def test_vital_expires_out_of_range():
    k = fctx()
    st, _, _ = tick(K.init(2, 6), k, place(), 10.0)
    st, _, _ = tick(st, k, place(y=[0.0, 1500.0, 0.0, 0.0, 0.0, 0.0]), 11.0)
    assert int(st.fiora.v_dir[0, JAX]) == -1


# ---- Q -------------------------------------------------------------------------

def test_lunge_dash_stab_on_hit_and_half_cooldown():
    k = fctx(ranks=jnp.asarray([[1, 1, 1, 1], [1, 1, 1, 0]], jnp.int32))
    u = place(y=[0.0, 600.0, 0.0, 0.0, 0.0, 0.0])
    st, a, _ = tick(K.init(2, 6), k, u, 10.0, order([0, -1], x=[0.0, 0.0], y=[1000.0, 0.0]))
    assert bool(a.dash.active[0]) and float(a.dash.to_y[0]) == pytest.approx(400.0)
    assert float(a.dash.speed[0]) == pytest.approx(1100.0) and float(a.mana_cost[0]) == 20.0
    st, _, b = tick(st, k._replace(dashing=jnp.asarray([True, False])), u, 10.0 + DT)
    assert not len(valid(b.packets)["raw"])
    k2 = k._replace(y=jnp.asarray([400.0, 0.0]))
    st, _, b = tick(st, k2, u, 10.4)
    p = valid(b.packets)
    np.testing.assert_allclose(p["raw"], [70.0 + 0.9 * 30.0], rtol=1e-5)
    assert p["dst"][0] == JAX and p["flags"][0] & D.TAG_BASIC_ATTACK and p["flags"][0] & D.TAG_ACTIVE_SPELL
    assert int(b.on_hit_target[0]) == JAX and bool(b.cooldown_start[0, 0])
    np.testing.assert_allclose(float(b.base_cooldown[0, 0]), 13.0 * 0.5)


def test_lunge_prefers_a_killable_minion_close_by_over_a_champion():
    k = fctx(ranks=jnp.asarray([[1, 1, 1, 1], [1, 1, 1, 0]], jnp.int32))
    u = place(x=[0.0, 0.0, 100.0, 3200.0, 6000.0, -3000.0], y=[0.0, 300.0, 0.0, 0.0, 0.0, 0.0],
              hp=[1000.0, 1000.0, 50.0, 477.0, 5000.0, 477.0])
    st = K.init(2, 6)._replace()
    st = st._replace(fiora=st.fiora._replace(q_dash=jnp.asarray([True, False]), q_cast_at=jnp.asarray([9.5, 0.0])))
    _, _, b = tick(st, k, u, 10.0)
    assert valid(b.packets)["dst"][0] == MINION_RED


# ---- W -------------------------------------------------------------------------

@pytest.mark.parametrize("parried", [False, True])
def test_riposte_parry_window_and_stab(parried):
    k = fctx(ranks=jnp.asarray([[1, 1, 1, 1], [1, 1, 1, 0]], jnp.int32))
    u = place()
    st, a, _ = tick(K.init(2, 6), k, u, 10.0, order([1, -1], x=[0.0, 0.0], y=[1000.0, 0.0]))
    assert float(a.cast_lockout[0]) == pytest.approx(0.75) and float(a.mana_cost[0]) == 50.0
    assert bool(K.defense(st, at(k, 10.3)).parry[0]) and bool(K.cc_immune(st, at(k, 10.3))[0])
    if parried:
        st = K.on_parry(st, at(k, 10.2), jnp.asarray([True, False]))
    hits, cd_at = [], []
    for j in range(1, 30):
        st, _, b = tick(st, k, u, 10.0 + j * DT)
        if len(valid(b.packets)["raw"]):
            hits.append((10.0 + j * DT, valid(b.packets), b))
        if bool(b.cooldown_start[0, 1]):
            cd_at.append(10.0 + j * DT)
    assert len(hits) == 1
    t, p, b = hits[0]
    np.testing.assert_allclose(p["raw"], [110.0 + 20.0], rtol=1e-5)
    assert p["dtype"][0] == D.MAGIC
    np.testing.assert_allclose(t - 10.0, 0.5 + (300.0 - 65.0) / 3200.0, atol=1.5 * DT)
    if parried:
        assert float(b.cc.stun[0, JAX]) == pytest.approx(2.0) and float(b.cc.slow[0, JAX]) == 0.0
    else:
        assert float(b.cc.slow[0, JAX]) == pytest.approx(0.5) and float(b.cc.slow_duration[0, JAX]) == 2.0
        np.testing.assert_allclose(float(K.debuffs(st, at(k, t + 1.0), u).attack_speed_cripple[JAX]), 0.25)
    assert not bool(K.defense(st, at(k, 10.8)).parry[0])
    assert len(cd_at) == 1 and cd_at[0] == pytest.approx(10.75, abs=1.5 * DT)        # post-effect


# ---- E -------------------------------------------------------------------------

def test_bladework_two_attacks_slow_then_crit():
    k = fctx(ranks=jnp.asarray([[1, 1, 3, 1], [1, 1, 1, 0]], jnp.int32), crit_damage=jnp.asarray([2.4, 2.0]))
    u = place(y=[0.0, 150.0, 0.0, 0.0, 0.0, 0.0])
    st, a, _ = tick(K.init(2, 6), k, u, 10.0, order([2, -1]))
    assert bool(a.attack_reset[0]) and float(a.mana_cost[0]) == 40.0
    mods = K.attack_mods(st, k)
    assert float(mods.extra_range[0]) == 25.0 and bool(mods.cannot_crit[0]) and not bool(mods.force_crit[0])
    np.testing.assert_allclose(float(K.stats(st, k).attack_speed[0]), 0.7, rtol=1e-6)
    st, o = K.on_attack(st, k, u, launch([True, False], [JAX, -1]))
    assert float(o.cc.slow[0, JAX]) == pytest.approx(0.3) and float(o.cc.slow_duration[0, JAX]) == 1.0
    mods = K.attack_mods(st, k)
    assert bool(mods.force_crit[0]) and not bool(mods.cannot_crit[0])
    np.testing.assert_allclose(float(mods.crit_mult[0]), 1.8 + (2.4 - 2.0), rtol=1e-6)
    st, o = K.on_attack(st, k, u, launch([True, False], [JAX, -1]))
    assert bool(o.cooldown_start[0, 2]) and int(st.fiora.e_left[0]) == 0
    assert not bool(K.attack_mods(st, k).force_crit[0])


# ---- R -------------------------------------------------------------------------

def test_grand_challenge_four_vitals_victory_zone():
    k = fctx(ranks=jnp.asarray([[1, 1, 1, 2], [1, 1, 1, 0]], jnp.int32), level=jnp.full((2,), 11, jnp.int32))
    u = place()
    st, a, _ = tick(K.init(2, 6), k, u, 10.0, order([3, -1], targets=[JAX, -1]))
    assert bool(a.cast_started[0]) and bool(a.cooldown_start[0, 3]) and float(a.mana_cost[0]) == 100.0
    st, _, _ = tick(st, k, u, 10.3)
    st2, h = K.on_damage(st, at(k, 10.3), u, report(FIO, JAX))
    assert float(h.heal[0]) == 0.0                                      # 0.5 s delay
    heals = 0
    for t, (fx, fy) in zip((10.6, 10.8, 11.0, 11.2), ((0.0, 0.0), (300.0, 300.0), (0.0, 600.0), (-300.0, 300.0))):
        kk = at(k, t)._replace(x=jnp.asarray([fx, 0.0]), y=jnp.asarray([fy, 0.0]))
        st, _, _ = tick(st, kk, u, t)
        st, h = K.on_damage(st, kk, u, report(FIO, JAX))
        heals += float(h.heal[0]) > 0
    assert heals == 4 and bool(jnp.all(st.fiora.r_hit[0]))
    st, _, b = tick(st, at(k, 11.3)._replace(y=jnp.asarray([300.0, 0.0])), u, 11.3)
    assert float(st.fiora.r_zone_until[0]) == pytest.approx(11.3 + 5.0)
    np.testing.assert_allclose(float(b.heal[0]), (105.0 + 0.7 * 30.0) * 0.25, rtol=1e-5)
