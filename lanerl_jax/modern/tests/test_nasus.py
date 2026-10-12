"""Nasus 26.19 kit on the modern packet contract (champions, NAS.* in docs/modern/CHAMPIONS.md)."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.modern import champions as K
from lanerl_jax.modern.champions import core
from lanerl_jax.modern.champions import nasus as N
from lanerl_jax.modern.core import damage as D
from lanerl_jax.modern.core.types import KIND_CHAMPION, KIND_MINION, KIND_MONSTER, AttackLaunch, CastOrder
from lanerl_jax.modern.data.champions import champion, cooldowns, spell
from lanerl_jax.modern.items.effects.core import Kills, Report
from lanerl_jax.modern.jungle import objectives as OBJ
from lanerl_jax.modern.tests.test_champions import DT, at, ctx, jv, launch, units, valid

NASUS, JAX, MINION_RED, MINION_RED2, TURRET_RED, MINION_BLUE = range(6)


def nctx(**over):
    return ctx(ids=(75, 24), **over)


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


# ---- data ----------------------------------------------------------------------

def test_record_and_tables():
    rec = champion("Nasus")
    assert rec["slots"]["Q"].endswith("/NasusQ") and rec["slots"]["Passive"].endswith("/NasusPassive")
    assert rec["character"]["pathfindingCollisionRadius"] == 50.0 and rec["character"]["acquisitionRange"] == 350.0
    assert rec["character"]["overrideGameplayCollisionRadius"] == 80.0
    r = jnp.asarray([[1, 1, 1, 1], [5, 5, 5, 3]], jnp.int32)
    np.testing.assert_allclose(np.asarray(core.cooldown_row("Nasus", r)), [[7.5, 15, 12, 120], [3.5, 11, 12, 80]])
    np.testing.assert_allclose(np.asarray(core.mana_row("Nasus", r)), [[20, 80, 60, 100], [20, 80, 100, 100]])
    assert cooldowns("Nasus", "R") == [120.0, 100.0, 80.0]
    from lanerl_jax.modern.core.stat_pipeline import champion_base
    b = champion_base(["Garen", "Nasus"])
    np.testing.assert_allclose(np.asarray(b.gameplay_radius), [65.0, 80.0])
    np.testing.assert_allclose(np.asarray(b.pathing_radius), [35.0, 50.0])
    np.testing.assert_allclose(np.asarray(b.acquisition_range), [400.0, 350.0])


@pytest.mark.parametrize("level,ls", [(1, 0.10), (6, 0.10), (7, 0.15), (12, 0.15), (13, 0.20), (18, 0.20)])
def test_passive_life_steal(level, ls):
    k = nctx(level=jnp.full((2,), level, jnp.int32))
    s = K.stats(K.init(2, 6), k)
    np.testing.assert_allclose(float(s.life_steal[0]), ls, rtol=1e-6)
    assert float(s.life_steal[1]) == 0.0


# ---- Q -------------------------------------------------------------------------

@pytest.mark.parametrize("rank", [1, 3, 5])
def test_q_empowered_attack(rank):
    k = nctx(ranks=jnp.asarray([[rank, 1, 1, 0], [1, 1, 1, 0]], jnp.int32))
    u = units()
    st, out = K.cast(K.init(2, 6), k, u, order([0, -1]))
    assert bool(out.cast_started[0]) and bool(out.attack_reset[0]) and not bool(out.cooldown_start[0, 0])
    np.testing.assert_allclose(float(out.mana_cost[0]), 20.0)
    np.testing.assert_allclose(float(out.cast_lockout[0]), 0.0)
    m = K.attack_mods(st, k)
    assert bool(m.uncancellable[0]) and float(m.extra_range[0]) == 50.0 and not bool(m.uncancellable[1])
    _, again = K.cast(st, at(k, 10.2), u, order([0, -1]))
    assert not bool(again.cast_started[0])                       # no recast while the window is open
    st = st._replace(nasus=st.nasus._replace(q_stacks=st.nasus.q_stacks.at[0].set(37.0)))
    st, out = K.on_hit(st, at(k, 10.5), u, launch([True, False], [MINION_RED, -1]))
    p = valid(out.packets)
    assert len(p["raw"]) == 1 and p["dst"][0] == MINION_RED and p["dtype"][0] == D.PHYSICAL
    np.testing.assert_allclose(p["raw"][0], jv("Nasus", "Q", "BonusDamage", rank) + 37.0, rtol=1e-6)
    assert p["flags"][0] & D.PROP_LIFESTEAL and p["flags"][0] & D.TAG_ACTIVE_SPELL
    assert not p["flags"][0] & D.TAG_BASIC_ATTACK
    assert bool(out.cooldown_start[0, 0]) and not bool(st.nasus.q_on[0])
    np.testing.assert_allclose(float(out.base_cooldown[0, 0]), cooldowns("Nasus", "Q")[rank - 1])
    assert float(K.attack_mods(st, k).extra_range[0]) == 0.0


def test_q_flat_bonus_crits_but_stacks_do_not():
    k = nctx()
    st, _ = K.cast(K.init(2, 6), k, units(), order([0, -1]))
    st = st._replace(nasus=st.nasus._replace(q_stacks=st.nasus.q_stacks.at[0].set(100.0)))
    crit = AttackLaunch(jnp.asarray([True, False]), jnp.asarray([MINION_RED, -1], jnp.int32), jnp.zeros(2, bool),
                        jnp.asarray([True, False]), jnp.asarray([7, 8], jnp.int32))
    _, out = K.on_hit(st, k, units(), crit)
    np.testing.assert_allclose(valid(out.packets)["raw"][0], jv("Nasus", "Q", "BonusDamage", 3) * 1.75 + 100.0,
                               rtol=1e-6)


def test_q_window_expires_and_dodge_consumes():
    k = nctx()
    u = units()
    st, _ = K.cast(K.init(2, 6), k, u, order([0, -1]))
    st2, out = K.periodic(st, at(k, 10.0 + 10.0 - 2 * DT), u)
    assert bool(st2.nasus.q_on[0]) and not bool(out.cooldown_start[0, 0])
    st2, out = K.periodic(st, at(k, 20.0), u)
    assert not bool(st2.nasus.q_on[0]) and bool(out.cooldown_start[0, 0])
    dodging = jnp.zeros((6,), bool).at[JAX].set(True)
    st3, out = N.on_hit(st.nasus, k, u, launch([True, False], [JAX, -1]), dodging)
    assert not np.asarray(out.packets.valid).any() and bool(out.cooldown_start[0, 0]) and not bool(st3.q_on[0])


@pytest.mark.parametrize("victim,sub,champ_kill,gain", [
    (MINION_RED, 0, 0.0, 4.0), (MINION_RED, 1, 0.0, 4.0), (MINION_RED, 2, 0.0, 10.0), (MINION_RED, 3, 0.0, 10.0),
    (JAX, 0, 1.0, 10.0), (JAX, 0, 0.0, 0.0)])
def test_q_stacks_on_kills(victim, sub, champ_kill, gain):
    k = nctx()
    u = units(sub=[0, 0, sub, 0, 0, 0])
    st, _ = K.cast(K.init(2, 6), k, u, order([0, -1]))
    st, _ = K.on_hit(st, k, u, launch([True, False], [victim, -1]))
    st2 = K.on_takedown(st, k, u, kills([victim], champ_kill))
    assert float(st2.nasus.q_stacks[0]) == gain
    late = K.on_takedown(st, at(k, 10.0 + DT), u, kills([victim], champ_kill))       # died on a later tick
    assert float(late.nasus.q_stacks[0]) == 0.0
    other = K.on_takedown(st, k, u, kills([MINION_RED2], champ_kill))                 # a different unit died
    assert float(other.nasus.q_stacks[0]) == 0.0


@pytest.mark.parametrize("sub,large", [(0, True), (4, False), (8, False), (10, True),
                                       (OBJ.SUB_BASE + OBJ.T_BARON, True), (OBJ.SUB_BASE + OBJ.T_MITE, False)])
def test_q_large_monsters(sub, large):
    u = units(kind=[KIND_CHAMPION, KIND_CHAMPION, KIND_MONSTER, KIND_MINION, 3, KIND_MINION], sub=[0, 0, sub, 0, 0, 0])
    assert bool(N._large(u, jnp.int32(MINION_RED))) == large


# ---- W -------------------------------------------------------------------------

@pytest.mark.parametrize("rank", [1, 3, 5])
def test_w_ramping_slow_and_cripple(rank):
    k = nctx(ranks=jnp.asarray([[1, rank, 1, 0], [1, 1, 1, 0]], jnp.int32))
    u = units()
    st, out = K.cast(K.init(2, 6), k, u, order([1, -1], [JAX, -1]))
    assert bool(out.cast_started[0]) and bool(out.cooldown_start[0, 1])
    np.testing.assert_allclose(float(out.mana_cost[0]), 80.0)
    np.testing.assert_allclose(float(out.cast_lockout[0]), 0.25)
    st, out = K.periodic(st, at(k, 10.25 - 2 * DT), u)
    assert float(out.cc.slow.max()) == 0.0
    st, out = K.periodic(st, at(k, 10.25), u)                     # end of the cast time
    np.testing.assert_allclose(float(out.cc.slow[0, JAX]), 0.35, rtol=1e-6)
    np.testing.assert_allclose(float(out.cc.slow_duration[0, JAX]), 5.0)
    assert int(out.cc.cast_id[0, JAX]) != 0 and float(out.cc.slow[0].sum()) == float(out.cc.slow[0, JAX])
    per = jv("Nasus", "W", "SlowPerTick", rank) / 100.0
    np.testing.assert_allclose(float(K.debuffs(st, k, u).attack_speed_cripple[JAX]), 0.75 * 0.35, rtol=1e-6)
    seen = []
    t = 10.25
    for _ in range(int(round(6.0 / DT))):
        t += DT
        st, out = K.periodic(st, at(k, t), u)
        if float(out.cc.slow[0, JAX]) > 0:
            seen.append((round(t - 10.25, 3), float(out.cc.slow[0, JAX]), float(out.cc.slow_duration[0, JAX])))
        if 2.5 < t - 10.25 < 2.5 + DT:
            np.testing.assert_allclose(float(K.debuffs(st, k, u).attack_speed_cripple[JAX]),
                                       0.75 * (0.35 + 2 * per), rtol=1e-5)
    assert [s[2] for s in seen] == [4.0, 3.0, 2.0, 1.0]
    np.testing.assert_allclose([s[1] for s in seen], [0.35 + j * per for j in range(1, 5)], rtol=1e-5)
    np.testing.assert_allclose([s[0] for s in seen], [1.0, 2.0, 3.0, 4.0], atol=1e-3)
    np.testing.assert_allclose(0.35 + 4 * per, jv("Nasus", "W", "MaxSlowTooltipOnly", rank) / 100.0, rtol=1e-5)
    assert not bool(st.nasus.w_on[0]) and float(K.debuffs(st, k, u).attack_speed_cripple[JAX]) == 0.0


def test_w_ramp_scales_with_target_tenacity():
    k = nctx(tenacity=jnp.asarray([0.0, 0.5], jnp.float32))
    u = units()
    st, _ = K.cast(K.init(2, 6), k, u, order([1, -1], [JAX, -1]))
    st, _ = K.periodic(st, at(k, 10.25), u)
    np.testing.assert_allclose(float(st.nasus.w_scale[0]), 0.5)
    times, t = [], 10.25
    for _ in range(int(round(3.0 / DT))):
        t += DT
        st, out = K.periodic(st, at(k, t), u)
        if float(out.cc.slow[0, JAX]) > 0:
            times.append(t - 10.25)
    np.testing.assert_allclose(times, [0.5, 1.0, 1.5, 2.0], atol=1e-3)      # max reached at 4 x 0.5 s
    assert not bool(st.nasus.w_on[0])                                         # ends at 5 x 0.5 s


def test_w_needs_enemy_champion_in_range_and_ends_on_target_death():
    u = units()
    _, out = K.cast(K.init(2, 6), nctx(), u, order([1, -1], [MINION_RED, -1]))
    assert not bool(out.cast_started[0])
    far = units(x=[0.0, 800.0, 100.0, 250.0, 3000.0, -100.0])
    _, out = K.cast(K.init(2, 6), nctx(), far, order([1, -1], [JAX, -1]))
    assert not bool(out.cast_started[0])
    edge = units(x=[0.0, 760.0, 100.0, 250.0, 3000.0, -100.0])                # 700 + Jax radius 65
    k = nctx()
    st, out = K.cast(K.init(2, 6), k, edge, order([1, -1], [JAX, -1]))
    assert bool(out.cast_started[0])
    st, _ = K.periodic(st, at(k, 10.25), edge)
    assert bool(st.nasus.w_on[0])
    dead = units(alive=[True, False, True, True, True, True])
    st, out = K.periodic(st, at(k, 11.25), dead)
    assert not bool(st.nasus.w_on[0]) and float(out.cc.slow.max()) == 0.0


# ---- E -------------------------------------------------------------------------

@pytest.mark.parametrize("rank", [1, 3, 5])
def test_e_spirit_fire(rank):
    k = nctx(ranks=jnp.asarray([[1, 1, rank, 0], [1, 1, 1, 0]], jnp.int32))
    # Fire at (500, 0), radius 380 to the edge: Jax (200) and red minion 3 (600) inside, red minion 2 (50) outside,
    # the blue minion (450) is an ally.
    u = units(x=[0.0, 200.0, 50.0, 600.0, 3000.0, 450.0])
    st, out = K.cast(K.init(2, 6), k, u, order([2, -1], x=[500.0, 0.0], y=[0.0, 0.0]))
    assert bool(out.cast_started[0]) and bool(out.cooldown_start[0, 2])
    np.testing.assert_allclose(float(out.mana_cost[0]), spell("Nasus", "E")["mana"]["values"][rank - 1])
    np.testing.assert_allclose(float(out.cast_lockout[0]), 0.25)
    st, out = K.periodic(st, at(k, 10.5 - 2 * DT), u)
    assert not np.asarray(out.packets.valid).any()
    st, out = K.periodic(st, at(k, 10.5), u)
    p = valid(out.packets)
    assert sorted(p["dst"].tolist()) == [JAX, MINION_RED2]                    # not the ally minion or Nasus
    np.testing.assert_allclose(p["raw"], jv("Nasus", "E", "InitialHitDamage", rank) + 0.6 * 20.0, rtol=1e-6)
    assert np.all(p["dtype"] == D.MAGIC) and np.all(p["flags"] & D.TAG_AOE)
    shred = -jv("Nasus", "E", "ArmorShredPercent", rank)
    db = K.debuffs(st, k, u)
    np.testing.assert_allclose(np.asarray(db.percent_armor_reduction)[[JAX, MINION_RED2, MINION_RED]],
                               [shred, shred, 0.0], rtol=1e-6)
    ticks, t = [], 10.5
    for _ in range(int(round(6.0 / DT))):
        t += DT
        st, out = K.periodic(st, at(k, t), u)
        q = valid(out.packets)
        if len(q["raw"]):
            ticks.append(round(t - 10.5, 3))
            np.testing.assert_allclose(q["raw"], jv("Nasus", "E", "DamagePerTick", rank) + 0.12 * 20.0, rtol=1e-6)
            assert np.all(q["flags"] & D.TAG_PERIODIC) and len(q["raw"]) == 2
    np.testing.assert_allclose(ticks, [1.0, 2.0, 3.0, 4.0, 5.0], atol=1.5 * DT)
    assert not bool(st.nasus.e_on[0])
    # The reduction lingers 1 s after the fire is gone.
    np.testing.assert_allclose(float(K.debuffs(st, at(k, 15.5 + 0.9), u).percent_armor_reduction[JAX]), shred,
                               rtol=1e-6)
    assert float(K.debuffs(st, at(k, 15.5 + 1.1), u).percent_armor_reduction[JAX]) == 0.0


def test_e_cast_time_death_cancels_and_delay_does_not_block():
    k = nctx()
    u = units()
    st, _ = K.cast(K.init(2, 6), k, u, order([2, -1], x=[300.0, 0.0], y=[0.0, 0.0]))
    _, out = K.cast(st, at(k, 10.1), u, order([0, -1]))
    assert not bool(out.cast_started[0])                           # still casting
    st2, out = K.cast(st, at(k, 10.3), u, order([0, -1]))
    assert bool(out.cast_started[0])                               # the fire's delay does not block Q
    st2, out = K.periodic(st2, at(k, 10.5), u)
    assert sorted(valid(out.packets)["dst"].tolist()) == [JAX, MINION_RED, MINION_RED2]   # the fire still lands
    dead = at(k, 10.1)._replace(alive=jnp.asarray([False, True]))
    st3, _ = K.periodic(st, dead, u)
    assert not bool(st3.nasus.e_pending[0])
    _, out = K.periodic(st3, at(k, 10.5), u)
    assert not np.asarray(out.packets.valid).any()


def test_e_rejects_points_out_of_range():
    _, out = K.cast(K.init(2, 6), nctx(), units(), order([2, -1], x=[700.0, 0.0], y=[0.0, 0.0]))
    assert not bool(out.cast_started[0])
    assert np.asarray(K.point_target_ranges(jnp.asarray([75, 86])))[0, 2] == 650.0
    assert np.asarray(K.unit_target_ranges(jnp.asarray([75, 86])))[0, 1] == 700.0


# ---- R -------------------------------------------------------------------------

@pytest.mark.parametrize("rank", [1, 2, 3])
def test_r_fury_of_the_sands(rank):
    k = nctx(ranks=jnp.asarray([[3, 1, 1, rank], [1, 1, 1, 0]], jnp.int32), ap=jnp.full((2,), 100.0, jnp.float32))
    # Jax at 200 with 1600 max HP; minion 3 at 500 (beyond the storm); a 20k HP monster at 250 hits the cap.
    u = units(x=[0.0, 200.0, 250.0, 500.0, 3000.0, -100.0], kind=[KIND_CHAMPION, KIND_CHAMPION, KIND_MONSTER,
              KIND_MINION, 3, KIND_MINION], max_hp=[1500.0, 1600.0, 20000.0, 477.0, 5000.0, 477.0])
    st, out = K.cast(K.init(2, 6), k, u, order([3, -1]))
    assert bool(out.cooldown_start[0, 3]) and float(out.mana_cost[0]) == 100.0
    np.testing.assert_allclose(float(out.cast_lockout[0]), 0.2)
    st, _ = K.periodic(st, at(k, 10.2), u)
    assert bool(st.nasus.r_on[0])
    s = K.stats(st, k)
    np.testing.assert_allclose(float(s.health[0]), jv("Nasus", "R", "BonusHealth", rank))
    np.testing.assert_allclose(float(s.armor[0]), jv("Nasus", "R", "InitialResistGain", rank))
    np.testing.assert_allclose(float(s.magic_resist[0]), jv("Nasus", "R", "InitialResistGain", rank))
    assert float(K.attack_mods(st, k).extra_range[0]) == 50.0
    _, o = K.on_attack(st, k, u, launch([False, False], [-1, -1]))
    np.testing.assert_allclose(float(o.base_cooldown[0, 0]), 0.5 * cooldowns("Nasus", "Q")[2])
    radius = 400.0 + 80.0 * jv("Nasus", "R", "SizeIncreasePercent", rank)
    np.testing.assert_allclose(float(N.r_radius(jnp.int32(rank))), radius, rtol=1e-6)
    ticks, t = [], 10.2
    for _ in range(int(round(16.0 / DT))):
        t += DT
        st, out = K.periodic(st, at(k, t), u)
        q = valid(out.packets)
        if len(q["raw"]):
            ticks.append(t - 10.2)
            assert sorted(q["dst"].tolist()) == [JAX, MINION_RED]
            pct = (jv("Nasus", "R", "AOEDamagePercent", rank) + 0.01) * 0.5
            np.testing.assert_allclose(q["raw"][q["dst"] == JAX], pct * 1600.0, rtol=1e-5)
            np.testing.assert_allclose(q["raw"][q["dst"] == MINION_RED], 120.0, rtol=1e-6)   # 240 per second
            assert np.all(q["flags"] & D.PROP_ULTIMATE) and np.all(q["dtype"] == D.MAGIC)
    assert len(ticks) == 30
    np.testing.assert_allclose(ticks[:2], [0.5, 1.0], atol=1.5 * DT)
    assert not bool(st.nasus.r_on[0]) and float(K.stats(st, k).health[0]) == 0.0


def test_r_ends_on_death():
    k = nctx()
    st, _ = K.cast(K.init(2, 6), k, units(), order([3, -1]))
    st, _ = K.periodic(st, at(k, 10.2), units())
    st, _ = K.periodic(st, at(k, 11.0)._replace(alive=jnp.asarray([False, True])), units())
    assert not bool(st.nasus.r_on[0])


# ---- isolation, jit ------------------------------------------------------------

def test_isolation_and_jit_all_hooks():
    k = nctx()
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
        return st, (o1, o2, o3, o4, o5), K.stats(st, k), K.attack_mods(st, k), K.debuffs(st, k, u)

    st1, outs, *_ = tick(K.init(2, 6), k)
    st2, *_ = tick(st1, at(k, 10.0 + DT))
    for a, b in zip(jax.tree_util.tree_leaves(st1), jax.tree_util.tree_leaves(st2)):
        assert a.dtype == b.dtype and a.shape == b.shape
    assert bool(st1.nasus.e_pending[0]) and bool(st1.jax.e_on[1])
    assert not bool(st1.nasus.e_pending[1]) and not bool(st1.garen.e_on.any())
    # Jax's Counter Strike mana and Nasus's are charged per holder.
    np.testing.assert_allclose(np.asarray(outs[0].mana_cost), [60.0 + 10 * 2, spell("Jax", "E")["mana"]["values"][2]])
