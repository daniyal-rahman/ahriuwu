"""Garen/Jax 26.19 kits on the modern packet contract (modern_champions)."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.data.modern import cooldowns, spell, values
from lanerl_jax.sim import modern_champions as K
from lanerl_jax.sim import modern_damage as D
from lanerl_jax.sim.modern_champions import core, garen as G
from lanerl_jax.sim.modern_item_effects.core import Kills, Report
from lanerl_jax.sim.modern_world_types import (KIND_CHAMPION, KIND_MINION, KIND_TURRET, AttackLaunch, CastOrder,
                                               WorldUnits)

DT = 1.0 / 30.0
GAREN_UNIT, JAX_UNIT, MINION_RED, MINION_RED2, TURRET_RED, MINION_BLUE = range(6)


def units(**over):
    # 0 Garen (blue), 1 Jax (red), 2-3 red minions, 4 red turret, 5 blue minion.
    base = dict(
        kind=[KIND_CHAMPION, KIND_CHAMPION, KIND_MINION, KIND_MINION, KIND_TURRET, KIND_MINION],
        sub=[0] * 6, team=[0, 1, 1, 1, 1, 0], alive=[True] * 6, targetable=[True] * 6,
        x=[0.0, 200.0, 100.0, 250.0, 3000.0, -100.0], y=[0.0] * 6,
        radius=[65.0, 65.0, 48.0, 48.0, 88.0, 48.0],
        hp=[1000.0, 1000.0, 477.0, 477.0, 5000.0, 477.0], max_hp=[1500.0, 1600.0, 477.0, 477.0, 5000.0, 477.0],
        armor=[40.0] * 6, magic_resist=[30.0] * 6, attack_damage=[60.0] * 6, attack_range=[125.0] * 6,
        attack_speed=[0.6] * 6, move_speed=[340.0] * 6, spawn_seq=[1, 2, 3, 4, 5, 6], spawn_time=[0.0] * 6)
    base.update(over)
    dt = {"kind": jnp.int32, "sub": jnp.int32, "team": jnp.int32, "alive": bool, "targetable": bool,
          "spawn_seq": jnp.int32}
    return WorldUnits(**{k: jnp.asarray(v, dt.get(k, jnp.float32)) for k, v in base.items()})


def ctx(ids=(86, 24), **over):
    c = len(ids)
    f = lambda v: jnp.full((c,), v, jnp.float32)      # noqa: E731
    base = dict(
        unit=jnp.arange(c, dtype=jnp.int32), champion_id=jnp.asarray(ids, jnp.int32),
        team=jnp.asarray([0, 1][:c] + [0] * max(0, c - 2), jnp.int32), alive=jnp.ones((c,), bool),
        level=jnp.full((c,), 9, jnp.int32), ranks=jnp.asarray([[3, 3, 3, 1]] * c, jnp.int32),
        x=jnp.asarray([0.0, 200.0][:c] + [0.0] * max(0, c - 2), jnp.float32), y=f(0.0),
        mana=f(500.0), max_mana=f(600.0), hp=f(1000.0), max_hp=f(1500.0), base_ad=f(100.0), bonus_ad=f(30.0),
        ap=f(20.0), bonus_hp=f(200.0), armor=f(60.0), magic_resist=f(40.0), bonus_attack_speed=f(0.3),
        crit_chance=f(0.0), crit_damage=f(1.75), ability_haste=f(0.0), ultimate_haste=f(0.0),
        cooldowns=jnp.zeros((c, 4), jnp.float32), now=jnp.float32(10.0), dt=jnp.float32(DT),
        silenced=jnp.zeros((c,), bool), stunned=jnp.zeros((c,), bool), in_combat_ms_since_damaged=f(0.0))
    base.update(over)
    return core.KitCtx(**base)


def order(slots, targets=None):
    c = len(slots)
    t = [-1] * c if targets is None else targets
    return CastOrder(jnp.asarray(slots, jnp.int32), jnp.asarray(t, jnp.int32), jnp.zeros((c,), jnp.float32),
                     jnp.zeros((c,), jnp.float32))


def launch(launched, targets):
    c = len(launched)
    return AttackLaunch(jnp.asarray(launched, bool), jnp.asarray(targets, jnp.int32), jnp.zeros((c,), bool),
                        jnp.zeros((c,), bool), jnp.arange(c, dtype=jnp.int32) + 7)


def at(k, now):
    return k._replace(now=jnp.float32(now))


def valid(p):
    m = np.asarray(p.valid)
    return {f: np.asarray(getattr(p, f))[m] for f in D.Packets._fields}


def jv(name, slot, key, rank):
    return values(name, slot, key)[rank]


# ---- data tables -------------------------------------------------------------

@pytest.mark.parametrize("rank", [1, 3, 5])
def test_cooldown_and_mana_tables_match_json(rank):
    r = jnp.asarray([[rank, rank, rank, min(rank, 3)]], jnp.int32)
    for name in ("Garen", "Jax"):
        cd = np.asarray(core.cooldown_row(name, r))[0]
        exp = [cooldowns(name, s)[min(rank, 3 if s == "R" else 5) - 1] for s in "QWER"]
        np.testing.assert_allclose(cd, exp, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(core.mana_row("Garen", r))[0], 0.0)
    jm = np.asarray(core.mana_row("Jax", r))[0]
    np.testing.assert_allclose(jm, [50, 30, spell("Jax", "E")["mana"]["values"][rank - 1], 100])
    assert jm[2] == 40 + 10 * rank          # legacy modern.mana_cost


def test_base_cooldown_reported_per_holder():
    k = ctx()
    st = K.init(2, 6)
    _, out = K.cast(st, k, units(), order([-1, -1]))
    np.testing.assert_allclose(np.asarray(out.base_cooldown)[0], np.asarray(core.cooldown_row("Garen", k.ranks))[0])
    np.testing.assert_allclose(np.asarray(out.base_cooldown)[1], np.asarray(core.cooldown_row("Jax", k.ranks))[1])


# ---- Garen -------------------------------------------------------------------

@pytest.mark.parametrize("rank", [1, 3, 5])
def test_garen_q_empowered_attack_silence_and_reset(rank):
    k = ctx(ranks=jnp.asarray([[rank, 1, 1, 0], [1, 1, 1, 0]], jnp.int32))
    u = units()
    st = K.init(2, 6)
    st, out = K.cast(st, k, u, order([0, -1]))
    assert bool(out.cast_started[0]) and int(out.cast_slot[0]) == 0 and not bool(out.cast_started[1])
    assert bool(out.cooldown_start[0, 0]) and bool(out.attack_reset[0]) and bool(out.cleanse_slow[0])
    np.testing.assert_allclose(float(out.base_cooldown[0, 0]), 8.0)
    assert float(out.mana_cost[0]) == 0.0
    cid = int(out.cast_id[0])
    assert cid > 0
    mods = K.attack_mods(st, k)
    assert float(mods.extra_range[0]) == 50.0 and bool(mods.attack_reset[0]) and bool(mods.cannot_crit[0])
    np.testing.assert_allclose(float(K.stats(st, k).percent_move_speed[0]), 0.35, rtol=1e-6)
    st, out = K.on_hit(st, at(k, 10.5), u, launch([True, False], [JAX_UNIT, -1]))
    p = valid(out.packets)
    assert len(p["raw"]) == 1 and p["dst"][0] == JAX_UNIT and p["dtype"][0] == D.PHYSICAL
    # Total Q attack = BaseDamage + 1.5 total AD; the world's basic attack supplies 1.0 AD.
    np.testing.assert_allclose(p["raw"][0] + 130.0, jv("Garen", "Q", "BaseDamage", rank) + 1.5 * 130.0, rtol=1e-5)
    assert p["flags"][0] & D.TAG_ACTIVE_SPELL and not p["flags"][0] & D.TAG_BASIC_ATTACK
    assert p["cast_id"][0] == cid and p["item"][0] == 0
    np.testing.assert_allclose(float(out.cc.silence[0, JAX_UNIT]), 1.5)
    assert int(out.cc.cast_id[0, JAX_UNIT]) == cid
    assert float(out.cc.silence.sum()) == pytest.approx(1.5)
    assert not bool(st.garen.q_on[0])
    # Consumed: the next attack is plain.
    _, out = K.on_hit(st, at(k, 11.0), u, launch([True, False], [JAX_UNIT, -1]))
    assert not np.asarray(out.packets.valid).any()


def test_garen_q_haste_and_window_expire():
    k = ctx()
    st, _ = K.cast(K.init(2, 6), k, units(), order([0, -1]))
    dur = jv("Garen", "Q", "MovementSpeedDuration", 3)
    st, _ = K.periodic(st, at(k, 10.0 + dur - 2 * DT), units())
    assert float(K.stats(st, k).percent_move_speed[0]) > 0
    st, _ = K.periodic(st, at(k, 10.0 + dur), units())
    assert float(K.stats(st, k).percent_move_speed[0]) == 0
    assert bool(st.garen.q_on[0])
    st, _ = K.periodic(st, at(k, 14.5), units())
    assert not bool(st.garen.q_on[0])


def test_garen_q_dodged_by_jax_e_consumes_without_effect():
    k = ctx()
    u = units()
    st = K.init(2, 6)
    st, _ = K.cast(st, k, u, order([0, 2]))      # Garen Q, Jax E
    assert bool(K.defense(st, k).dodge_basic[1])
    st, out = K.on_hit(st, k, u, launch([True, False], [JAX_UNIT, -1]))
    assert not np.asarray(out.packets.valid).any()
    assert float(out.cc.silence.sum()) == 0.0
    assert not bool(st.garen.q_on[0])


@pytest.mark.parametrize("rank", [1, 4])
def test_garen_w_shield_dr_tenacity(rank):
    k = ctx(ranks=jnp.asarray([[1, rank, 1, 0], [1, 1, 1, 0]], jnp.int32))
    st, out = K.cast(K.init(2, 6), k, units(), order([1, -1]))
    np.testing.assert_allclose(float(out.shield.amount[0, 0]), jv("Garen", "W", "BaseShield", rank) + 0.18 * 200.0,
                               rtol=1e-5)
    np.testing.assert_allclose(float(out.shield.duration[0, 0]), 0.75)
    assert float(out.shield.amount[1].sum()) == 0.0
    assert bool(out.cooldown_start[0, 1])
    np.testing.assert_allclose(float(out.base_cooldown[0, 1]), cooldowns("Garen", "W")[rank - 1])
    d = K.defense(st, k)
    np.testing.assert_allclose(float(d.received_mult[0]), 1 - jv("Garen", "W", "DRPercent", rank), rtol=1e-6)
    np.testing.assert_allclose(float(d.tenacity_bonus[0]), 0.6, rtol=1e-6)
    assert float(d.received_mult[1]) == 1.0 and float(d.tenacity_bonus[1]) == 0.0
    st, _ = K.periodic(st, at(k, 10.75), units())
    d = K.defense(st, k)
    assert float(d.tenacity_bonus[0]) == 0.0 and float(d.received_mult[0]) < 1.0
    st, _ = K.periodic(st, at(k, 14.0), units())
    assert float(K.defense(st, k).received_mult[0]) == 1.0


def test_garen_w_dr_applies_through_damage_pipeline_not_true():
    k = ctx()
    st, _ = K.cast(K.init(2, 6), k, units(), order([1, -1]))
    d = K.defense(st, k)
    dfn = D.default_defense(2, unit_class=D.CLASS_CHAMPION)._replace(received_mult=d.received_mult)
    p = D.packets(jnp.asarray([True, True]), 1, 0, 100.0, jnp.asarray([D.PHYSICAL, D.TRUE]))
    final = np.asarray(D.premitigation_to_final(p, D.default_offense(2, unit_class=D.CLASS_CHAMPION), dfn))
    np.testing.assert_allclose(final, [100 * (1 - jv("Garen", "W", "DRPercent", 3)), 100.0], rtol=1e-5)


def test_garen_w_passive_stacks_on_takedown():
    k = ctx()
    st = K.init(2, 6)
    kills = Kills(jnp.asarray([1.0, 1.0]), jnp.zeros(2), jnp.asarray([4.0, 9.0]), jnp.zeros(2, bool),
                  jnp.zeros((2, 6), bool))
    st = K.on_takedown(st, k, units(), kills)
    s = K.stats(st, k)
    np.testing.assert_allclose(float(s.armor[0]), 5 * 0.2, rtol=1e-6)
    np.testing.assert_allclose(float(s.magic_resist[0]), 5 * 0.2, rtol=1e-6)
    assert float(s.armor[1]) == 0.0
    for _ in range(40):
        st = K.on_takedown(st, k, units(), kills)
    np.testing.assert_allclose(float(K.stats(st, k).armor[0]), 30.0, rtol=1e-6)


@pytest.mark.parametrize("rank,bonus_as", [(1, 0.0), (3, 0.3), (5, 0.6)])
def test_garen_e_spin_ticks_damage_and_ids(rank, bonus_as):
    k = ctx(ranks=jnp.asarray([[1, 1, rank, 0], [1, 1, 1, 0]], jnp.int32),
            bonus_attack_speed=jnp.full((2,), bonus_as, jnp.float32))
    # Keep Jax (unit 1) out of the circle; minions 2 (nearest) and 3 in it.
    u = units(x=[0.0, 2000.0, 100.0, 250.0, 3000.0, -100.0])
    st, out = K.cast(K.init(2, 6), k, u, order([2, -1]))
    assert bool(out.cast_started[0]) and not bool(out.cooldown_start[0, 2])
    assert bool(K.attack_mods(st, k).cannot_attack[0])
    per_tick = jv("Garen", "E", "BaseDamagePerTick", rank) + jv("Garen", "E", "ADRatioPerTick", rank) * 130.0
    n_ticks = 7 + int(np.floor(bonus_as / 0.25))
    ids, total, now, ended = [], 0, 10.0, None
    for i in range(int(3.2 / DT)):
        st, out = K.periodic(st, at(k, now), u)
        p = valid(out.packets)
        if len(p["raw"]):
            total += 1
            assert set(p["dst"]) == {MINION_RED, MINION_RED2}
            np.testing.assert_allclose(p["raw"][p["dst"] == MINION_RED], per_tick * 1.25, rtol=1e-5)
            np.testing.assert_allclose(p["raw"][p["dst"] == MINION_RED2], per_tick, rtol=1e-5)
            assert np.all(p["flags"] & D.TAG_AOE) and np.all(p["flags"] & D.TAG_ACTIVE_SPELL)
            assert np.all(p["dtype"] == D.PHYSICAL)
            assert len(set(p["cast_id"])) == 1 and p["cast_id"][0] > 0
            ids.append(p["cast_id"][0])
        if bool(out.cooldown_start[0, 2]) and ended is None:
            ended = now
        now += DT
    assert total == n_ticks
    assert len(set(ids)) == n_ticks       # one instance per tick (Conqueror stacks per tick)
    assert ended is not None and ended == pytest.approx(10.0 + 3.0 - DT, abs=DT)
    assert not bool(K.attack_mods(st, k).cannot_attack[0])


def test_garen_e_cancel_after_one_second_and_crit():
    k = ctx(crit_chance=jnp.ones((2,), jnp.float32))
    u = units(x=[0.0, 2000.0, 100.0, 2500.0, 3000.0, -100.0])
    st, _ = K.cast(K.init(2, 6), k, u, order([2, -1]))
    st, out = K.periodic(st, k, u)
    p = valid(out.packets)
    per_tick = jv("Garen", "E", "BaseDamagePerTick", 3) + jv("Garen", "E", "ADRatioPerTick", 3) * 130.0
    np.testing.assert_allclose(p["raw"], [per_tick * 1.25 * 1.3], rtol=1e-5)
    assert p["flags"][0] & D.PROP_CRIT
    st, out = K.cast(st, at(k, 10.5), u, order([2, -1]))
    assert bool(st.garen.e_on[0]) and not bool(out.cooldown_start[0, 2])
    st, out = K.cast(st, at(k, 11.0), u, order([2, -1]))
    assert not bool(st.garen.e_on[0]) and bool(out.cooldown_start[0, 2]) and not bool(out.cast_started[0])


def test_garen_e_shreds_champion_armor_at_six_hits():
    k = ctx()
    u = units(x=[0.0, 200.0, 2000.0, 2500.0, 3000.0, -100.0])
    st, _ = K.cast(K.init(2, 6), k, u, order([2, -1]))
    now = 10.0
    for i in range(6):
        assert float(K.debuffs(st, at(k, now), u).percent_armor_reduction[JAX_UNIT]) == 0.0
        while True:
            st, out = K.periodic(st, at(k, now), u)
            now += DT
            if np.asarray(out.packets.valid).any():
                break
    red = K.debuffs(st, at(k, now), u).percent_armor_reduction
    np.testing.assert_allclose(float(red[JAX_UNIT]), 0.25)
    assert float(red[MINION_RED]) == 0.0


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_garen_r_true_damage_after_cast_time(rank):
    k = ctx(ranks=jnp.asarray([[1, 1, 1, rank], [1, 1, 1, 1]], jnp.int32))
    u = units()
    st, out = K.cast(K.init(2, 6), k, u, order([3, -1], [JAX_UNIT, -1]))
    assert bool(out.cast_started[0]) and int(out.cast_slot[0]) == 3
    np.testing.assert_allclose(float(out.cast_lockout[0]), 0.435)
    assert bool(out.cooldown_start[0, 3])
    np.testing.assert_allclose(float(out.base_cooldown[0, 3]), cooldowns("Garen", "R")[rank - 1])
    cid = int(out.cast_id[0])
    assert bool(K.attack_mods(st, k).cannot_attack[0])
    # Silenced during the cast: no other cast.
    _, o2 = K.cast(st, k, u, order([0, -1]))
    assert not bool(o2.cast_started[0])
    st, out = K.periodic(st, at(k, 10.3), u)
    assert not np.asarray(out.packets.valid).any()
    u2 = u._replace(hp=u.hp.at[JAX_UNIT].set(400.0))
    st, out = K.periodic(st, at(k, 10.42), u2)
    p = valid(out.packets)
    assert len(p["raw"]) == 1 and p["dst"][0] == JAX_UNIT and p["dtype"][0] == D.TRUE
    np.testing.assert_allclose(p["raw"][0], jv("Garen", "R", "BaseDamage", rank)
                               + jv("Garen", "R", "ExecuteDamage", rank) * 1200.0, rtol=1e-5)
    assert p["flags"][0] & D.PROP_ULTIMATE and p["flags"][0] & D.TAG_ACTIVE_SPELL
    assert not p["flags"][0] & D.PROP_EXECUTE
    assert p["cast_id"][0] == cid
    assert not bool(st.garen.r_pending[0])


def test_garen_r_requires_enemy_champion_in_range():
    k = ctx()
    st = K.init(2, 6)
    _, out = K.cast(st, k, units(), order([3, -1], [MINION_RED, -1]))
    assert not bool(out.cast_started[0])
    far = units(x=[0.0, 600.0, 100.0, 250.0, 3000.0, -100.0])
    _, out = K.cast(st, k, far, order([3, -1], [JAX_UNIT, -1]))
    assert not bool(out.cast_started[0])
    edge = units(x=[0.0, 464.0, 100.0, 250.0, 3000.0, -100.0])       # 400 + 65 target radius
    _, out = K.cast(st, k, edge, order([3, -1], [JAX_UNIT, -1]))
    assert bool(out.cast_started[0])


def test_garen_passive_regen():
    k = ctx(level=jnp.asarray([7, 7], jnp.int32), in_combat_ms_since_damaged=jnp.asarray([9.0, 9.0], jnp.float32))
    _, out = K.periodic(K.init(2, 6), k, units())
    np.testing.assert_allclose(float(out.heal[0]), 1500.0 * 3.3 / 100 / 5 * DT, rtol=1e-5)
    assert float(out.heal[1]) == 0.0
    _, out = K.periodic(K.init(2, 6), k._replace(in_combat_ms_since_damaged=jnp.asarray([7.0, 9.0])), units())
    assert float(out.heal[0]) == 0.0
    np.testing.assert_allclose(np.asarray(G.regen_rate(jnp.asarray([1, 6, 13, 14, 18]))), [1.5, 2.5, 8.1, 8.5, 10.1],
                               rtol=1e-6)


# ---- Jax ---------------------------------------------------------------------

@pytest.mark.parametrize("rank", [1, 3, 5])
def test_jax_q_leap_damage_and_w_on_q(rank):
    k = ctx(ranks=jnp.asarray([[1, 1, 1, 0], [rank, 1, 1, 0]], jnp.int32))
    u = units()
    st, out = K.cast(K.init(2, 6), k, u, order([-1, 1]))          # W first
    wid = int(out.cast_id[1])
    assert bool(out.attack_reset[1]) and not bool(out.cooldown_start[1, 1])
    np.testing.assert_allclose(float(out.mana_cost[1]), 30.0)
    st, out = K.cast(st, at(k, 10.1), u, order([-1, 0], [-1, GAREN_UNIT]))
    assert bool(out.dash.active[1]) and int(out.dash.target[1]) == GAREN_UNIT
    np.testing.assert_allclose(float(out.dash.to_x[1]), 0.0)
    np.testing.assert_allclose(float(out.dash.speed[1]), 1400.0)
    assert not bool(out.dash.blink[1]) and not bool(out.dash.active[0])
    assert bool(out.cooldown_start[1, 0]) and float(out.mana_cost[1]) == 50.0
    qid = int(out.cast_id[1])
    assert bool(K.attack_mods(st, k).cannot_attack[1])
    # 200 units at 1400/s ~ 0.143 s.
    st, out = K.periodic(st, at(k, 10.1), u)
    assert not np.asarray(out.packets.valid).any()
    st, out = K.periodic(st, at(k, 10.1 + 0.143 - DT / 2), u)
    p = valid(out.packets)
    assert len(p["raw"]) == 2 and np.all(p["dst"] == GAREN_UNIT)
    phys = p["dtype"] == D.PHYSICAL
    np.testing.assert_allclose(p["raw"][phys], jv("Jax", "Q", "Damage", rank) + 30.0, rtol=1e-5)
    np.testing.assert_allclose(p["raw"][~phys], jv("Jax", "W", "Damage", 1) + 0.6 * 20.0, rtol=1e-5)
    assert p["cast_id"][phys][0] == qid and p["cast_id"][~phys][0] == wid
    assert np.all(p["flags"] & D.TAG_ACTIVE_SPELL) and not np.any(p["flags"] & D.TAG_AOE)
    assert bool(out.cooldown_start[1, 1]) and not bool(st.jax.w_on[1])


def test_jax_q_on_ally_dashes_without_damage_and_rejects_structures():
    k = ctx()
    u = units(team=[0, 1, 1, 1, 1, 1])
    st, out = K.cast(K.init(2, 6), k, u, order([-1, 0], [-1, MINION_BLUE]))   # unit 5 is an ally of Jax here
    assert bool(out.dash.active[1])
    st, out = K.periodic(st, at(k, 10.3), u)
    assert not np.asarray(out.packets.valid).any() and not bool(st.jax.q_pending[1])
    near_turret = units(x=[0.0, 200.0, 100.0, 250.0, 500.0, -100.0])
    _, out = K.cast(K.init(2, 6), k, near_turret, order([-1, 0], [-1, TURRET_RED]))
    assert not bool(out.cast_started[1])
    _, out = K.cast(K.init(2, 6), k, u, order([-1, 0], [-1, JAX_UNIT]))
    assert not bool(out.cast_started[1])


@pytest.mark.parametrize("rank", [1, 5])
def test_jax_w_empowered_attack(rank):
    k = ctx(ranks=jnp.asarray([[1, 1, 1, 0], [1, rank, 1, 0]], jnp.int32))
    u = units()
    st, out = K.cast(K.init(2, 6), k, u, order([-1, 1]))
    wid = int(out.cast_id[1])
    mods = K.attack_mods(st, k)
    assert float(mods.extra_range[1]) == 50.0 and bool(mods.attack_reset[1])
    _, again = K.cast(st, at(k, 10.1), u, order([-1, 1]))
    assert not bool(again.cast_started[1])
    st, out = K.on_hit(st, k, u, launch([False, True], [-1, GAREN_UNIT]))
    p = valid(out.packets)
    assert len(p["raw"]) == 1 and p["dtype"][0] == D.MAGIC and p["cast_id"][0] == wid
    np.testing.assert_allclose(p["raw"][0], jv("Jax", "W", "Damage", rank) + 0.6 * 20.0, rtol=1e-5)
    assert p["flags"][0] & D.TAG_ACTIVE_SPELL
    assert bool(out.cooldown_start[1, 1]) and not bool(st.jax.w_on[1])
    np.testing.assert_allclose(float(out.base_cooldown[1, 1]), cooldowns("Jax", "W")[rank - 1])
    # Structures take half.
    st, _ = K.cast(K.init(2, 6), k, u, order([-1, 1]))
    _, out = K.on_hit(st, k, u, launch([False, True], [-1, TURRET_RED]))
    np.testing.assert_allclose(valid(out.packets)["raw"][0], 0.5 * (jv("Jax", "W", "Damage", rank) + 12.0), rtol=1e-5)


def test_jax_w_expires_and_starts_cooldown():
    k = ctx()
    st, _ = K.cast(K.init(2, 6), k, units(), order([-1, 1]))
    st, out = K.periodic(st, at(k, 19.9), units())
    assert bool(st.jax.w_on[1]) and not bool(out.cooldown_start[1, 1])
    st, out = K.periodic(st, at(k, 20.0 - DT / 2), units())
    assert not bool(st.jax.w_on[1]) and bool(out.cooldown_start[1, 1])


@pytest.mark.parametrize("rank", [1, 3, 5])
def test_jax_e_dodge_counter_and_stun(rank):
    k = ctx(ranks=jnp.asarray([[1, 1, 1, 0], [1, 1, rank, 0]], jnp.int32))
    u = units()
    st, out = K.cast(K.init(2, 6), k, u, order([-1, 2]))
    eid = int(out.cast_id[1])
    np.testing.assert_allclose(float(out.mana_cost[1]), 40 + 10 * rank)
    assert not bool(out.cooldown_start[1, 2])
    d = K.defense(st, k)
    assert bool(d.dodge_basic[1]) and not bool(d.dodge_basic[0])
    np.testing.assert_allclose(float(d.aoe_received_mult[1]), 0.75)
    # The damage pipeline drops basic attacks on Jax and reduces AoE.
    dfn = D.default_defense(6, unit_class=D.CLASS_CHAMPION)._replace(
        dodge_basic=jnp.zeros(6, bool).at[1].set(d.dodge_basic[1]),
        aoe_received_mult=jnp.ones(6).at[1].set(d.aoe_received_mult[1]))
    hits = D.packets(jnp.ones(4, bool), jnp.asarray([0, 2, 0, 4]), 1, 100.0, D.PHYSICAL,
                     jnp.asarray([D.BASIC_ATTACK, D.BASIC_ATTACK, D.TAG_AOE, D.BASIC_ATTACK]),
                     cast_id=jnp.asarray([11, 12, 13, 14]))
    off = D.default_offense(6, unit_class=D.CLASS_CHAMPION)._replace(is_turret=jnp.zeros(6, bool).at[4].set(True))
    final = np.asarray(D.premitigation_to_final(hits, off, dfn))
    assert final[0] == 0 and final[1] == 0 and final[2] == pytest.approx(75.0) and final[3] > 0
    # Dodges counted per attack instance: Garen + minion attacks, not the AoE spell nor the turret.
    rep = Report(hits, None, jnp.zeros(6))
    st, _ = K.on_damage(st, k, u, rep)
    assert int(st.jax.e_dodges[1]) == 2
    for _ in range(4):
        st, _ = K.on_damage(st, k, u, rep)
    assert int(st.jax.e_dodges[1]) == 10
    # Recast before 1 s is ignored; after 1 s releases.
    st, out = K.cast(st, at(k, 10.5), u, order([-1, 2]))
    assert bool(st.jax.e_on[1]) and not np.asarray(out.packets.valid).any()
    st, out = K.cast(st, at(k, 11.0), u, order([-1, 2]))
    assert not bool(st.jax.e_on[1]) and bool(out.cooldown_start[1, 2]) and float(out.mana_cost[1]) == 0.0
    np.testing.assert_allclose(float(out.base_cooldown[1, 2]), cooldowns("Jax", "E")[rank - 1])
    p = valid(out.packets)
    assert set(p["dst"]) == {GAREN_UNIT, MINION_BLUE}           # enemies within 375 of Jax (x=200)
    base = jv("Jax", "E", "BaseDamage", rank) + 0.7 * 20.0
    mult = 1 + 0.2 * 5
    np.testing.assert_allclose(p["raw"][p["dst"] == GAREN_UNIT], (base + 0.04 * 1500.0) * mult, rtol=1e-5)
    np.testing.assert_allclose(p["raw"][p["dst"] == MINION_BLUE], (base + 0.04 * 477.0) * mult, rtol=1e-5)
    assert np.all(p["flags"] & D.TAG_AOE) and np.all(p["flags"] & D.TAG_ACTIVE_SPELL)
    assert np.all(p["dtype"] == D.MAGIC) and np.all(p["cast_id"] == eid)
    np.testing.assert_allclose(float(out.cc.stun[1, GAREN_UNIT]), 1.0)
    assert int(out.cc.cast_id[1, GAREN_UNIT]) == eid
    assert float(out.cc.stun[1, MINION_RED]) == 0.0           # ally of Jax
    assert not bool(K.defense(st, k).dodge_basic[1])


def test_jax_e_expiry_releases_and_death_cancels():
    k = ctx()
    u = units()
    st, _ = K.cast(K.init(2, 6), k, u, order([-1, 2]))
    s1, out = K.periodic(st, at(k, 12.0 - DT / 2), u)
    assert not bool(s1.jax.e_on[1]) and bool(out.cooldown_start[1, 2])
    assert float(out.cc.stun[1, GAREN_UNIT]) == 1.0
    dead = k._replace(alive=jnp.asarray([True, False]))
    s2, out = K.periodic(st, at(dead, 10.5), u)
    assert not bool(s2.jax.e_on[1]) and bool(out.cooldown_start[1, 2])
    assert not np.asarray(out.packets.valid).any()


def test_jax_passive_attack_speed_stacks():
    k = ctx(level=jnp.asarray([10, 10], jnp.int32))
    st = K.init(2, 6)
    u = units()
    for i in range(10):
        st, _ = K.on_attack(st, at(k, 10.0 + i * 0.5), u, launch([True, True], [JAX_UNIT, GAREN_UNIT]))
    per = 0.05 + 0.015 * 3
    np.testing.assert_allclose(float(K.stats(st, k).attack_speed[1]), 8 * per, rtol=1e-5)
    assert float(K.stats(st, k).attack_speed[0]) == 0.0
    # Expire 2.5 s after the last attack, then one stack every 0.35 s.
    last = 10.0 + 9 * 0.5
    now = last + DT
    while now < last + 2.5 + 0.35 * 2 + DT:
        st, _ = K.periodic(st, at(k, now), u)
        now += DT
    assert int(st.jax.stacks[1]) in (5, 6)


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_jax_r_swing_resists_and_passive(rank):
    k = ctx(ranks=jnp.asarray([[1, 1, 1, 0], [1, 1, 1, rank]], jnp.int32))
    u = units()
    st, out = K.cast(K.init(2, 6), k, u, order([-1, 3]))
    rid = int(out.cast_id[1])
    assert bool(out.cooldown_start[1, 3]) and float(out.mana_cost[1]) == 100.0
    np.testing.assert_allclose(float(out.base_cooldown[1, 3]), cooldowns("Jax", "R")[rank - 1])
    assert bool(K.attack_mods(st, k).cannot_attack[1])
    st, out = K.periodic(st, at(k, 10.1), u)
    assert not np.asarray(out.packets.valid).any()
    st, out = K.periodic(st, at(k, 10.25 - DT / 2), u)
    p = valid(out.packets)
    assert set(p["dst"]) == {GAREN_UNIT, MINION_BLUE}
    np.testing.assert_allclose(p["raw"], jv("Jax", "R", "SwingDamageBase", rank) + 20.0, rtol=1e-5)
    assert np.all(p["flags"] & D.PROP_ULTIMATE) and np.all(p["flags"] & D.TAG_AOE)
    assert np.all(p["flags"] & D.TAG_ACTIVE_SPELL) and np.all(p["cast_id"] == rid)
    armor = jv("Jax", "R", "BaseResists", rank) + 0.4 * 30.0
    s = K.stats(st, k)
    np.testing.assert_allclose(float(s.armor[1]), armor, rtol=1e-5)
    np.testing.assert_allclose(float(s.magic_resist[1]), 0.6 * armor, rtol=1e-5)
    # Under the buff every 2nd landed attack procs.
    procs = []
    for i in range(4):
        st, out = K.on_hit(st, at(k, 10.5 + i), u, launch([False, True], [-1, GAREN_UNIT]))
        procs.append(valid(out.packets))
    assert [len(p["raw"]) for p in procs] == [0, 1, 0, 1]
    pr = procs[1]
    np.testing.assert_allclose(pr["raw"][0], jv("Jax", "R", "PassiveBaseDamage", rank) + 0.6 * 20.0, rtol=1e-5)
    assert pr["flags"][0] & D.PROP_ULTIMATE and pr["flags"][0] & D.TAG_ON_HIT and pr["dtype"][0] == D.MAGIC
    assert pr["cast_id"][0] > 0 and pr["cast_id"][0] != procs[3]["cast_id"][0]


def test_jax_r_passive_every_third_attack_without_buff():
    k = ctx()
    st = K.init(2, 6)
    n = []
    for i in range(6):
        st, out = K.on_hit(st, at(k, 10.0 + i), units(), launch([False, True], [-1, MINION_BLUE]))
        n.append(int(np.asarray(out.packets.valid).sum()))
    assert n == [0, 0, 1, 0, 0, 1]
    # Counter resets 2.5 s after the last landed attack.
    st, _ = K.on_hit(st, at(k, 20.0), units(), launch([False, True], [-1, MINION_BLUE]))
    st, _ = K.periodic(st, at(k, 22.6), units())
    assert int(st.jax.r_hits[1]) == 0


# ---- isolation, jit ----------------------------------------------------------

def test_holder_isolation_two_garens():
    k = ctx(ids=(86, 86), team=jnp.asarray([0, 1], jnp.int32), x=jnp.asarray([0.0, 200.0], jnp.float32))
    u = units()
    st, out = K.cast(K.init(2, 6), k, u, order([0, 1]))
    assert bool(out.cast_started[0]) and bool(out.cast_started[1])
    assert int(out.cast_id[0]) != int(out.cast_id[1])
    assert bool(st.garen.q_on[0]) and not bool(st.garen.q_on[1])
    assert bool(st.garen.w_on[1]) and not bool(st.garen.w_on[0])
    assert float(out.shield.amount[0].sum()) == 0.0 and float(out.shield.amount[1].sum()) > 0
    # Jax kit state untouched by Garen holders.
    for a, b in zip(st.jax, K.init(2, 6).jax):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert not bool(out.dash.active.any()) and float(out.mana_cost.sum()) == 0.0


def test_holder_isolation_gates_and_cc():
    k = ctx(stunned=jnp.asarray([True, False]))
    st, out = K.cast(K.init(2, 6), k, units(), order([0, 1]))
    assert not bool(out.cast_started[0]) and bool(out.cast_started[1])
    k = ctx(silenced=jnp.asarray([False, True]))
    _, out = K.cast(K.init(2, 6), k, units(), order([0, 1]))
    assert bool(out.cast_started[0]) and not bool(out.cast_started[1])
    k = ctx(cooldowns=jnp.asarray([[1.0, 0, 0, 0], [0, 0, 0, 0]], jnp.float32))
    _, out = K.cast(K.init(2, 6), k, units(), order([0, 1]))
    assert not bool(out.cast_started[0])
    k = ctx(mana=jnp.asarray([0.0, 10.0], jnp.float32))
    _, out = K.cast(K.init(2, 6), k, units(), order([0, 1]))
    assert bool(out.cast_started[0]) and not bool(out.cast_started[1])     # Garen is manaless


def test_cast_ids_unique_across_ticks_and_holders():
    k = ctx()
    a = np.asarray(core.make_cast_id(k, 0))
    b = np.asarray(core.make_cast_id(at(k, 10.0 + DT), 0))
    c = np.asarray(core.make_cast_id(k, 3))
    ids = set(a) | set(b) | set(c)
    assert len(ids) == 6 and min(ids) > 0


def test_jit_all_hooks():
    k = ctx()
    u = units()
    st = K.init(2, 6)
    rep = Report(D.packets(jnp.ones(2, bool), jnp.asarray([0, 2]), 1, 50.0, D.PHYSICAL, D.BASIC_ATTACK),
                 None, jnp.zeros(6))
    kills = Kills(jnp.zeros(2), jnp.zeros(2), jnp.ones(2), jnp.zeros(2, bool), jnp.zeros((2, 6), bool))

    @jax.jit
    def tick(st, k):
        st, o1 = K.cast(st, k, u, order([2, 2]))
        st, o2 = K.periodic(st, k, u)
        st, o3 = K.on_attack(st, k, u, launch([True, True], [1, 0]))
        st, o4 = K.on_hit(st, k, u, launch([True, True], [1, 0]))
        st, o5 = K.on_damage(st, k, u, rep)
        st = K.on_takedown(st, k, u, kills)
        return st, (o1, o2, o3, o4, o5), K.stats(st, k), K.defense(st, k), K.attack_mods(st, k), K.debuffs(st, k, u)

    st1, outs, s, d, m, db = tick(st, k)
    st2, outs2, *_ = tick(st1, at(k, 10.0 + DT))
    assert jax.tree_util.tree_structure(st1) == jax.tree_util.tree_structure(st2)
    for a, b in zip(jax.tree_util.tree_leaves(st1), jax.tree_util.tree_leaves(st2)):
        assert a.dtype == b.dtype and a.shape == b.shape
    assert int(np.asarray(outs[1].packets.valid).sum()) > 0       # Garen E first tick
    assert bool(st1.jax.e_on[1])
