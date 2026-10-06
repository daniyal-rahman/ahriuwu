"""Resolve tree (8400): RUNES.md §6 rules and §13 fixtures F-13..F-19, F-30, F-31."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.modern.combat import combat_tick, init_combat
from lanerl_jax.modern.core import damage as D
from lanerl_jax.modern.items.catalog import zero_stats
from lanerl_jax.modern.items.effects import runtime as R
from lanerl_jax.modern.items.effects.core import CC, Cast
from lanerl_jax.modern.runes.effects import resolve as RS
from lanerl_jax.modern.runes.effects.core import rune_item
from lanerl_jax.modern.tests import item_harness as H
from lanerl_jax.modern.tests import rune_harness as RH
from lanerl_jax.modern.tests.rune_harness import world

GRASP, AFTERSHOCK, GUARDIAN, DEMOLISH = 8437, 8439, 8465, 8446
FONT, SHIELD_BASH, CONDITIONING, SECOND_WIND, BONE_PLATING = 8463, 8401, 8429, 8444, 8473
OVERGROWTH, REVITALIZE, UNFLINCHING = 8451, 8453, 8242
ALL = [GRASP, AFTERSHOCK, GUARDIAN, DEMOLISH, FONT, SHIELD_BASH, CONDITIONING, SECOND_WIND, BONE_PLATING,
       OVERGROWTH, REVITALIZE, UNFLINCHING]


def f(x, i=0):
    a = np.asarray(x)
    return float(a if a.ndim == 0 else a.reshape(a.shape[0], -1)[i, 0])


def no_attack():
    return H.attack(hit=(False, False))


def test_coverage_lists_every_resolve_rune():
    assert set(RS.COVERAGE) == set(ALL)


# ---- Grasp of the Undying (F-13, F-14) ----------------------------------------

def _prime(state, page, ctx, u, *, ticks, dt, combat_every_tick=True, t0=0.0):
    n = u.x.shape[0]
    lc = jnp.full((2,), -1e9, jnp.float32)
    for k in range(ticks):
        now = t0 + k * dt
        c = ctx._replace(now=jnp.float32(now), dt=jnp.float32(dt))
        state, _ = RS.periodic(state, page, c, u, RH.ev(c, n, clocks=RH.clocks(last_combat=lc)))
        if combat_every_tick or k == 0:
            lc = jnp.full((2,), now, jnp.float32)      # combat event this tick (seen next periodic)
    return state, lc


def test_grasp_f13_melee_proc_after_4s_of_combat():
    u = world()
    n = u.x.shape[0]
    page = RH.page(GRASP)
    ctx = H.ctx(base_hp=1000.0, max_hp=1000.0, hp=500.0)
    st, lc = _prime(RS.init(2, n), page, ctx, u, ticks=8, dt=0.5)       # t = 0 .. 3.5
    assert f(RS.grasp_stacks(st)) == 3.0
    c = ctx._replace(now=jnp.float32(4.0), dt=jnp.float32(0.5))
    st, _ = RS.periodic(st, page, c, u, RH.ev(c, n, clocks=RH.clocks(last_combat=3.5)))
    assert f(RS.grasp_stacks(st)) == 4.0
    c = ctx._replace(now=jnp.float32(4.2))
    st, eff = RS.on_hit(st, page, c, u, RH.ev(c, n, attack=H.attack(), clocks=RH.clocks(last_combat=4.0)))
    assert RH.total(eff.packets, rune=GRASP, dst=1) == pytest.approx(35.0, rel=1e-5)
    m = np.asarray(eff.packets.item) == rune_item(GRASP)
    assert np.all(np.asarray(eff.packets.dtype)[m & np.asarray(eff.packets.valid)] == D.MAGIC)
    flags = int(np.asarray(eff.packets.flags)[np.argmax(m & np.asarray(eff.packets.valid))])
    assert flags & D.TAG_PROC and flags & D.TAG_ON_HIT
    assert f(eff.heal) == pytest.approx(13.0, rel=1e-5)
    assert f(RS.stats(st, page, c, RH.ev(c, n)).health) == 5.0
    assert f(RS.grasp_stacks(st)) == 0.0
    # Holder isolation: holder 1 has no Grasp.
    assert f(eff.heal, 1) == 0.0 and RH.total(eff.packets, rune=GRASP, src=1) == 0.0


def test_grasp_f14_ranged_and_scaled_hp():
    u = world()
    n = u.x.shape[0]
    page = RH.page(GRASP)
    st = RS.init(2, n)._replace(grasp_acc=jnp.full((2,), 4.0, jnp.float32))
    c = H.ctx(base_hp=1000.0, max_hp=1000.0, ranged=True, now=1.0)
    st2, eff = RS.on_hit(st, page, c, u, RH.ev(c, n, attack=H.attack(), clocks=RH.clocks(last_combat=0.5)))
    assert RH.total(eff.packets, rune=GRASP) == pytest.approx(14.0, rel=1e-5)
    assert f(eff.heal) == pytest.approx(5.2, rel=1e-5)
    assert f(st2.grasp_hp) == 2.0
    # Melee at 2000 max HP after 30 procs: 70 damage, 26 heal.
    c = H.ctx(base_hp=1850.0, max_hp=2000.0, now=1.0)
    _, eff = RS.on_hit(st, page, c, u, RH.ev(c, n, attack=H.attack(), clocks=RH.clocks(last_combat=0.5)))
    assert RH.total(eff.packets, rune=GRASP) == pytest.approx(70.0, rel=1e-5)
    assert f(eff.heal) == pytest.approx(26.0, rel=1e-5)


def test_grasp_generation_stops_after_3s_and_decays_at_5s():
    u = world()
    n = u.x.shape[0]
    page = RH.page(GRASP)
    ctx = H.ctx()
    # One combat event at t=0, then none: generation runs until t=3 -> 3 stacks.
    st, _ = _prime(RS.init(2, n), page, ctx, u, ticks=10, dt=0.5, combat_every_tick=False)   # t = 0 .. 4.5
    assert f(RS.grasp_stacks(st)) == 3.0
    c = ctx._replace(now=jnp.float32(5.0), dt=jnp.float32(0.5))
    st, _ = RS.periodic(st, page, c, u, RH.ev(c, n, clocks=RH.clocks(last_combat=0.0)))
    assert f(RS.grasp_stacks(st)) == 0.0
    # Primed but not in combat for 5 s: no proc; vs a minion: no proc.
    st = st._replace(grasp_acc=jnp.full((2,), 4.0, jnp.float32))
    c = ctx._replace(now=jnp.float32(10.0))
    _, eff = RS.on_hit(st, page, c, u, RH.ev(c, n, attack=H.attack(), clocks=RH.clocks(last_combat=4.0)))
    assert RH.total(eff.packets, rune=GRASP) == 0.0
    u2 = world([dict(x=100, y=0, team=1)])
    st3 = RS.init(2, 3)._replace(grasp_acc=jnp.full((2,), 4.0, jnp.float32))
    _, eff = RS.on_hit(st3, page, c, u2, RH.ev(c, 3, attack=H.attack(target=(2, 0)), clocks=RH.clocks(last_combat=9.0)))
    assert RH.total(eff.packets, rune=GRASP) == 0.0


# ---- Second Wind (F-15) ---------------------------------------------------------

def test_second_wind_f15_continuous_regen():
    u = world()
    n = u.x.shape[0]
    page = RH.page(SECOND_WIND)
    ctx = H.ctx(base_hp=1000.0, max_hp=1000.0, hp=600.0)
    rep = RH.report(RH.hit_packet(1, 0, 50.0), u)
    st, _ = RS.on_damage(RS.init(2, n), page, ctx, u, RH.ev(ctx, n, report=rep))
    assert f(st.sw_until) == pytest.approx(10.0) and f(st.sw_until, 1) < 0
    dt = 1 / 30
    hp, healed, first = 600.0, 0.0, None
    for k in range(1, 400):
        c = ctx._replace(now=jnp.float32(k * dt), dt=jnp.float32(dt), hp=jnp.asarray([hp, 1000.0], jnp.float32))
        st, eff = RS.periodic(st, page, c, u, RH.ev(c, n))
        h = f(eff.heal_plain)
        hp += h
        healed += h
        if k == 30:
            first = healed
    assert first == pytest.approx(1.6, rel=5e-3)
    assert healed == pytest.approx(400 * (1 - np.exp(-0.04)), rel=1e-3)


def test_second_wind_needs_health_damage_from_champion():
    u = world([dict(x=100, y=0, team=1)])
    n = u.x.shape[0]
    page = RH.page(SECOND_WIND, BONE_PLATING)
    ctx = H.ctx()
    sh = D.grant_shield(D.init_shields(n), 0, 500.0, D.SHIELD_ALL, 0.0, 5.0)
    rep = RH.report(RH.hit_packet(1, 0, 50.0), u, shields=sh)          # fully absorbed
    st, _ = RS.on_damage(RS.init(2, n), page, ctx, u, RH.ev(ctx, n, report=rep))
    assert f(st.sw_until) < 0 and f(st.bp_until) < 0
    rep = RH.report(RH.hit_packet(2, 0, 50.0), u)                       # minion source
    st, _ = RS.on_damage(st, page, ctx, u, RH.ev(ctx, n, report=rep))
    assert f(st.sw_until) < 0 and f(st.bp_until) < 0


# ---- Bone Plating (F-16, F-17) -----------------------------------------------------

def _active_bp(n, level_until=1.5):
    return RS.init(2, n)._replace(bp_source=jnp.asarray([1, -1], jnp.int32),
                                  bp_until=jnp.asarray([level_until, -1e9], jnp.float32),
                                  bp_left=jnp.asarray([3, 0], jnp.int32),
                                  bp_cd_until=jnp.asarray([56.5, -1e9], jnp.float32))


def test_bone_plating_f17_floor_and_cast_instances():
    u = world()
    n = u.x.shape[0]
    page = RH.page(BONE_PLATING)
    ctx = H.ctx(level=18, now=0.5)
    st = _active_bp(n)
    p = RH.hit_packet(1, 0, 50.0)
    block = RS.packet_block(st, page, ctx, u, RH.ev(ctx, n), p)
    assert f(block) == pytest.approx(60.0)
    res = RH.report(p._replace(block=block), u).resolved
    assert f(res.final) == 0.0
    assert f(RS.bone_plating_block(H.ctx(level=9))) == pytest.approx(44.1176, abs=1e-4)
    # Two packets of one cast instance plus an unrelated instance: 2 blocks used.
    ctx1 = H.ctx(level=1, now=0.5)
    p3 = D.packets(jnp.ones(3, bool), 1, 0, jnp.asarray([50.0, 50.0, 50.0]), D.MAGIC, D.TAG_ACTIVE_SPELL,
                   cast_id=jnp.asarray([7, 7, 0]))
    jit_block = jax.jit(RS.packet_block)
    b3 = np.asarray(jit_block(st, page, ctx1, u, RH.ev(ctx1, n), p3))
    np.testing.assert_allclose(b3, [30.0, 0.0, 30.0])
    st2, _ = jax.jit(RS.on_damage)(st, page, ctx1, u, RH.ev(ctx1, n, report=RH.report(p3._replace(block=b3), u)))
    assert int(st2.bp_left[0]) == 1 and sorted(np.asarray(st2.bp_seen[0]).tolist()) == [0, 0, 7]
    # A later packet of cast 7 is not blocked again; another source is never blocked.
    later = D.packets(jnp.ones(2, bool), jnp.asarray([1, 0]), jnp.asarray([0, 1]), 50.0, D.MAGIC, cast_id=7)
    np.testing.assert_allclose(np.asarray(RS.packet_block(st2, page, ctx1, u, RH.ev(ctx1, n), later)), [0.0, 0.0])


def _e2e(page, *, dt=0.5, holder_hp=1000.0, extra=()):
    u = world(list(extra), x1=150.0)
    n = u.x.shape[0]
    hp = jnp.full((n,), 1000.0, jnp.float32).at[0].set(holder_hp)
    u = u._replace(hp=hp, max_hp=jnp.full((n,), 1000.0, jnp.float32))
    ctx0 = H.ctx(base_hp=1000.0, max_hp=1000.0, dt=dt, in_combat=True)
    dfn = D.default_defense(n)._replace(unit_class=u.cls)
    off = D.default_offense(n)._replace(unit_class=u.cls)
    own = H.own([], [])
    stats = zero_stats((2,))

    def step(cs, hp, max_hp, shields, status, now, base, attack):
        ctx = ctx0._replace(now=jnp.asarray(now, jnp.float32), hp=hp[:2], max_hp=max_hp[:2])
        units = u._replace(hp=hp, max_hp=max_hp)
        return combat_tick(cs, own, page, ctx, units, attack=attack, cast=H.cast(started=(False, False)),
                           request=jnp.zeros((2,), jnp.int32), base_packets=base, base_offense=off,
                           base_defense=dfn, hp=hp, max_hp=max_hp, shields=shields, status=status,
                           kills=H.kills(n), holder_stats=stats)
    return u, n, jax.jit(step)


def test_bone_plating_f16_end_to_end_combat_tick():
    page = RH.page(BONE_PLATING)
    u, n, step = _e2e(page)
    cs, hp, max_hp = init_combat(2, n), u.hp, u.max_hp
    shields, status = D.init_shields(n), R.init_status(n)
    taken = []
    for now, raw in ((0.0, 100.0), (0.4, 50.0), (0.8, 50.0), (1.2, 50.0), (1.4, 50.0)):
        base = D.packets(jnp.ones(1, bool), 1, 0, raw, D.PHYSICAL, D.BASIC_ATTACK)
        out = step(cs, hp, max_hp, shields, status, now, base, no_attack())
        taken.append(float(hp[0] - out.hp[0]))
        cs, hp, max_hp, shields, status = out.state, out.hp, out.max_hp, out.shields, out.status
    np.testing.assert_allclose(taken, [100.0, 20.0, 20.0, 20.0, 50.0], rtol=1e-5)
    rs = cs.runes.resolve
    assert float(rs.bp_cd_until[0]) == pytest.approx(1.2 + 55.0)
    assert int(rs.bp_left[1]) == 0                                      # holder 1 has no rune


def test_bone_plating_window_expiry_starts_cooldown():
    u = world()
    n = u.x.shape[0]
    page = RH.page(BONE_PLATING)
    ctx = H.ctx(now=0.0)
    st, _ = RS.on_damage(RS.init(2, n), page, ctx, u, RH.ev(ctx, n, report=RH.report(RH.hit_packet(1, 0, 100.0), u)))
    assert f(st.bp_cd_until) == pytest.approx(1.5 + 55.0) and int(st.bp_source[0]) == 1
    c = ctx._replace(now=jnp.float32(1.6))
    assert f(RS.packet_block(st, page, c, u, RH.ev(c, n), RH.hit_packet(1, 0, 50.0))) == 0.0
    c = ctx._replace(now=jnp.float32(30.0))
    st2, _ = RS.on_damage(st, page, c, u, RH.ev(c, n, report=RH.report(RH.hit_packet(1, 0, 100.0), u)))
    assert f(st2.bp_until) == pytest.approx(1.5)                         # still on cooldown


# ---- Grasp max HP through the runtime ---------------------------------------------

def test_grasp_end_to_end_max_hp_sync():
    page = RH.page(GRASP)
    u, n, step = _e2e(page, holder_hp=500.0)
    cs, hp, max_hp = init_combat(2, n), u.hp, u.max_hp
    shields, status = D.init_shields(n), R.init_status(n)
    poke = D.packets(jnp.ones(1, bool), 0, 1, 0.0, D.PHYSICAL, D.BASIC_ATTACK)   # 0-damage combat event
    for k in range(9):                                                         # t = 0 .. 4.0
        out = step(cs, hp, max_hp, shields, status, 0.5 * k, poke, no_attack())
        cs, hp, max_hp, shields, status = out.state, out.hp, out.max_hp, out.shields, out.status
    assert float(RS.grasp_stacks(cs.runes.resolve)[0]) == 4.0
    out = step(cs, hp, max_hp, shields, status, 4.5, poke, H.attack(raw=(0.0, 0.0)))
    assert float(hp[1] - out.hp[1]) == pytest.approx(35.0, rel=1e-5)
    assert float(out.hp[0]) == pytest.approx(513.0, rel=1e-5)
    cs, hp, max_hp, shields, status = out.state, out.hp, out.max_hp, out.shields, out.status
    out = step(cs, hp, max_hp, shields, status, 5.0, poke, no_attack())
    assert float(out.max_hp[0]) == pytest.approx(1005.0)
    assert float(out.hp[0]) == pytest.approx(518.0, rel=1e-5)
    assert float(out.max_hp[1]) == pytest.approx(1000.0)


# ---- Conditioning (F-18), Overgrowth (F-19) -----------------------------------------

def test_conditioning_f18():
    page = RH.page(CONDITIONING)
    st = RS.init(2, 2)
    ctx = H.ctx(base_armor=40.0, bonus_armor=20.0)
    for t, expect in ((719.9, 60.0), (720.0, 70.04)):
        s = RS.stats(st, page, ctx, RH.ev(ctx, 2, game_time=jnp.float32(t)))
        armor = (40.0 + 20.0 + f(s.armor)) * (1.0 + f(s.percent_armor))
        assert armor == pytest.approx(expect, abs=1e-3)
        assert f(s.armor, 1) == 0.0 and f(s.percent_magic_resist, 1) == 0.0
    assert f(s.magic_resist) == 8.0 and f(s.percent_magic_resist) == pytest.approx(0.03)


def test_overgrowth_f19_and_counting():
    page = RH.page(OVERGROWTH)
    ctx = H.ctx()
    for count, expect in ((119, 1542.0), (120, 1599.075)):
        st = RS.init(2, 2)._replace(og_count=jnp.asarray([count, 0], jnp.int32))
        s = RS.stats(st, page, ctx, RH.ev(ctx, 2))
        assert (1500.0 + f(s.health)) * (1.0 + f(s.percent_health)) == pytest.approx(expect, rel=1e-6)
    rows = [dict(x=1000, y=0, team=1), dict(x=1500, y=0, team=1), dict(x=500, y=0, team=0),
            dict(x=600, y=0, team=1, cls=D.CLASS_MONSTER), dict(x=700, y=0, team=1)]
    u = world(rows)
    n = u.x.shape[0]
    deaths = jnp.asarray([False, False, True, True, True, True, True])
    sight = jnp.ones((2, n), bool).at[0, 6].set(False)
    ctx = H.ctx(alive=False)                                          # counted while dead
    st, _ = RS.periodic(RS.init(2, n), page, ctx, u, RH.ev(ctx, n, deaths=deaths, sight=sight))
    # unit 2 (in range), 5 (monster) count; 3 out of range, 4 ally, 6 no sight.
    assert int(st.og_count[0]) == 2 and int(st.og_count[1]) == 0


# ---- Aftershock (F-30) ----------------------------------------------------------

@pytest.mark.parametrize("level,bonus_armor,expect,burst", [(1, 0.0, 45.0, 25.0), (1, 60.0, 80.0, 25.0),
                                                           (18, 200.0, 150.0, 120.0)])
def test_aftershock_f30(level, bonus_armor, expect, burst):
    rows = [dict(x=200, y=0, team=1, cls=D.CLASS_MONSTER), dict(x=100, y=0, team=1),
            dict(x=900, y=0, team=1, cls=D.CLASS_CHAMPION)]
    u = world(rows)
    n = u.x.shape[0]
    page = RH.page(AFTERSHOCK)
    ctx = H.ctx(level=level, bonus_armor=bonus_armor, base_hp=600.0, max_hp=700.0)
    imm = jnp.zeros((2, n), bool).at[0, 1].set(True)
    cc = CC(jnp.zeros((2, n), bool), imm)
    st, _ = RS.on_cc(RS.init(2, n), page, ctx, u, RH.ev(ctx, n, cc=cc))
    c = ctx._replace(now=jnp.float32(1.0))
    s = RS.stats(st, page, c, RH.ev(c, n))
    assert f(s.armor) == pytest.approx(expect, abs=1e-4) and f(s.magic_resist) == pytest.approx(45.0)
    assert f(s.armor, 1) == 0.0
    c = ctx._replace(now=jnp.float32(2.4))
    st, eff = RS.periodic(st, page, c, u, RH.ev(c, n))
    assert RH.total(eff.packets, rune=AFTERSHOCK) == 0.0
    c = ctx._replace(now=jnp.float32(2.5))
    assert f(RS.stats(st, page, c, RH.ev(c, n)).armor) == 0.0
    st, eff = RS.periodic(st, page, c, u, RH.ev(c, n))
    dmg = burst + 0.08 * 100.0
    assert H.packet_targets(eff.packets, item=rune_item(AFTERSHOCK)) == [1, 2]   # champion + monster only
    assert RH.total(eff.packets, rune=AFTERSHOCK, dst=1) == pytest.approx(dmg, rel=1e-5)
    # Cooldown 20 s from trigger; burst only once.
    _, eff = RS.periodic(st, page, ctx._replace(now=jnp.float32(3.0)), u, RH.ev(c, n))
    assert RH.total(eff.packets, rune=AFTERSHOCK) == 0.0
    st2, _ = RS.on_cc(st, page, ctx._replace(now=jnp.float32(19.0)), u, RH.ev(ctx, n, cc=cc))
    assert f(st2.as_cd_until) == pytest.approx(20.0)
    st3, _ = RS.on_cc(st, page, ctx._replace(now=jnp.float32(20.0)), u, RH.ev(ctx, n, cc=cc))
    assert f(st3.as_cd_until) == pytest.approx(40.0)


# ---- Demolish (F-31) ----------------------------------------------------------------

def test_demolish_f31_stacks_cooldown_and_global_lock():
    turret = dict(x=200, y=0, team=1, cls=D.CLASS_STRUCTURE, hp=5000, max_hp=5000)
    rows = [dict(x=0, y=0, team=0, cls=D.CLASS_CHAMPION), dict(x=50, y=0, team=0, cls=D.CLASS_CHAMPION), turret,
            dict(turret, x=4000)]
    u = H.units(rows)
    n = u.x.shape[0]
    page = RH.perks([DEMOLISH], [DEMOLISH])
    ctx = H.ctx(base_hp=1500.0, max_hp=1500.0, team=(0, 0))
    is_turret = jnp.asarray([False, False, True, True])
    st = RS.init(2, n)

    def hit(st, now, who=(True, False), tgt=(2, 2)):
        c = ctx._replace(now=jnp.float32(now))
        return RS.on_hit(st, page, c, u, RH.ev(c, n, attack=H.attack(hit=who, target=tgt), is_turret=is_turret))

    out = []
    for t in (0.0, 1.0, 2.0):
        st, eff = hit(st, t)
        out.append(RH.total(eff.packets, rune=DEMOLISH))
    assert out == [0.0, 0.0, pytest.approx(505.0, rel=1e-5)]
    flags = int(np.asarray(eff.packets.flags)[np.argmax(np.asarray(eff.packets.item) == rune_item(DEMOLISH))])
    assert flags & D.TAG_PROC and flags & D.TAG_BASIC_ATTACK
    # Holder 1 (same team) has 2 stacks on turret 2: the 3 s global lock blocks it.
    st = st._replace(demo_stacks=st.demo_stacks.at[1, 2].set(2))
    st, eff = hit(st, 3.0, who=(False, True))
    assert RH.total(eff.packets, rune=DEMOLISH) == 0.0 and int(st.demo_stacks[1, 2]) == 2
    st, eff = hit(st, 5.1, who=(False, True))
    assert RH.total(eff.packets, rune=DEMOLISH, src=1) == pytest.approx(505.0, rel=1e-5)
    # Holder 0: no stacks during its 30 s cooldown; the next consume >= 30 s later.
    for t in (10.0, 20.0, 31.0, 32.0, 33.0):
        st, eff = hit(st, t)
        assert RH.total(eff.packets, rune=DEMOLISH) == 0.0
    st, eff = hit(st, 34.0)
    assert RH.total(eff.packets, rune=DEMOLISH, src=0) == pytest.approx(505.0, rel=1e-5)


def test_demolish_ranged_and_consume_clears_other_turrets():
    turret = dict(x=200, y=0, team=1, cls=D.CLASS_STRUCTURE)
    u = world([turret, dict(turret, x=600)])
    n = u.x.shape[0]
    page = RH.page(DEMOLISH)
    ctx = H.ctx(base_hp=1000.0, max_hp=1000.0, ranged=True)
    st = RS.init(2, n)._replace(demo_stacks=jnp.zeros((2, n), jnp.int32).at[0, 2].set(2).at[0, 3].set(1))
    ev = RH.ev(ctx, n, attack=H.attack(target=(2, 0)), is_turret=jnp.asarray([False, False, True, True]))
    st, eff = RS.on_hit(st, page, ctx, u, ev)
    assert RH.total(eff.packets, rune=DEMOLISH) == pytest.approx(250.0, rel=1e-5)
    assert int(st.demo_stacks[0, 3]) == 0


# ---- Unflinching, Revitalize, Shield Bash, Font of Life (§6.5, §6.7) -----------------

def test_unflinching_silence_fixture():
    u = world()
    n = u.x.shape[0]
    page = RH.page(UNFLINCHING)
    ctx = H.ctx(dt=0.1)
    st = RS.init(2, n)
    active = {}
    for k in range(50):
        t = round(10.0 + 0.1 * k, 4)
        c = ctx._replace(now=jnp.float32(t))
        e = RH.ev(c, n, holder_cc_from_champion=jnp.asarray([t < 11.5, t < 11.5]))
        s = RS.stats(st, page, c, e)
        active[t] = (f(s.armor), f(s.magic_resist), f(s.armor, 1))
        st, _ = RS.periodic(st, page, c, u, e)
    assert active[10.0][0] == 0.0                         # damage resolves before the bonus
    assert active[10.1][:2] == (10.0, 10.0) and active[13.3][0] == 10.0
    assert active[13.5][0] == 0.0 and active[12.0][2] == 0.0


def test_revitalize_heal_power_and_low_hp_mult():
    page = RH.page(REVITALIZE)
    st = RS.init(2, 2)
    hi = H.ctx(base_hp=1000.0, hp=500.0)
    lo = H.ctx(base_hp=1000.0, hp=399.0)
    hsp = f(RS.stats(st, page, hi, RH.ev(hi, 2)).heal_shield_power)
    assert 80 * (1 + hsp) * f(RS.heal_mult(st, page, hi, RH.ev(hi, 2))) == pytest.approx(84.0, rel=1e-5)
    assert 80 * (1 + hsp) * f(RS.heal_mult(st, page, lo, RH.ev(lo, 2))) == pytest.approx(92.4, rel=1e-5)
    assert f(RS.heal_mult(st, page, lo, RH.ev(lo, 2)), 1) == 1.0


def test_shield_bash_empowers_next_attack_on_champion():
    u = world()
    n = u.x.shape[0]
    page = RH.page(SHIELD_BASH)
    ctx = H.ctx(base_hp=600.0, max_hp=1000.0)
    st = RS.post_tick(RS.init(2, n), page, ctx, u, RH.ev(ctx, n, shield_gained=jnp.asarray([100.0, 100.0])))
    st = RS.post_tick(st, page, ctx, u, RH.ev(ctx, n, shield_gained=jnp.asarray([40.0, 0.0])))   # smaller: kept 100
    c = ctx._replace(now=jnp.float32(1.0))
    st2, eff = RS.on_hit(st, page, c, u, RH.ev(c, n, attack=H.attack()))
    assert RH.total(eff.packets, rune=SHIELD_BASH) == pytest.approx(5.0 + 0.025 * 400 + 15.0, rel=1e-5)
    m = np.asarray(eff.packets.valid) & (np.asarray(eff.packets.item) == rune_item(SHIELD_BASH))
    assert np.asarray(eff.packets.dtype)[m][0] == D.PHYSICAL          # 0/0 tie -> adaptive type physical
    _, eff = RS.on_hit(st2, page, c, u, RH.ev(c, n, attack=H.attack()))
    assert RH.total(eff.packets, rune=SHIELD_BASH) == 0.0             # consumed
    late = ctx._replace(now=jnp.float32(4.1))
    _, eff = RS.on_hit(st, page, late, u, RH.ev(late, n, attack=H.attack()))
    assert RH.total(eff.packets, rune=SHIELD_BASH) == 0.0             # window over


def test_font_of_life_heals_self_and_ally_with_cooldown():
    rows = [dict(x=0, y=0, team=0, cls=D.CLASS_CHAMPION), dict(x=300, y=0, team=0, cls=D.CLASS_CHAMPION),
            dict(x=500, y=0, team=0, cls=D.CLASS_CHAMPION), dict(x=200, y=0, team=1, cls=D.CLASS_CHAMPION)]
    u = H.units(rows)
    n = u.x.shape[0]
    page = RH.perks([FONT], [], [])
    ctx = H.ctx(n=3, team=(0, 0, 0), x=jnp.asarray([0.0, 300.0, 500.0]), base_hp=1000.0,
                hp=jnp.asarray([1000.0, 700.0, 400.0]))
    slow = jnp.zeros((3, n), bool).at[0, 3].set(True)
    cc = CC(slow, jnp.zeros((3, n), bool))
    st, eff = RS.on_cc(RS.init(3, n), page, ctx, u, RH.ev(ctx, n, cc=cc))
    np.testing.assert_allclose(np.asarray(eff.heal), [10.0, 0.0, 10.0])   # full-HP self still heals
    _, eff = RS.on_cc(st, page, ctx._replace(now=jnp.float32(19.0)), u, RH.ev(ctx, n, cc=cc))
    assert float(jnp.sum(eff.heal)) == 0.0
    rng = H.ctx(n=3, team=(0, 0, 0), ranged=True, level=18, x=jnp.asarray([0.0, 300.0, 1500.0]))
    _, eff = RS.on_cc(RS.init(3, n), page, rng, u, RH.ev(rng, n, cc=cc))
    np.testing.assert_allclose(np.asarray(eff.heal), [35.0, 35.0, 0.0], rtol=1e-5)


def test_guardian_needs_ally_then_shields_both():
    u = world()
    n = u.x.shape[0]
    page = RH.perks([GUARDIAN], [GUARDIAN])
    ctx = H.ctx()
    rep = RH.report(RH.hit_packet(1, 0, 80.0), u)
    st, eff = RS.on_damage(RS.init(2, n), page, ctx, u, RH.ev(ctx, n, report=rep))
    assert float(jnp.sum(eff.shields.amount)) == 0.0                  # 1v1: never triggers
    rows = [dict(x=0, y=0, team=0, cls=D.CLASS_CHAMPION), dict(x=200, y=0, team=0, cls=D.CLASS_CHAMPION),
            dict(x=300, y=0, team=1, cls=D.CLASS_CHAMPION)]
    u = H.units(rows)
    n = u.x.shape[0]
    page = RH.page(GUARDIAN)
    ctx = H.ctx(team=(0, 0), x=jnp.asarray([0.0, 200.0]), base_hp=600.0, max_hp=1000.0)
    st = RS.init(2, n)
    p = RH.hit_packet(2, 1, 30.0)                                     # 30 on the ally: below 50
    st, eff = RS.on_damage(st, page, ctx, u, RH.ev(ctx, n, report=RH.report(p, u)))
    assert float(jnp.sum(eff.shields.amount)) == 0.0
    c = ctx._replace(now=jnp.float32(1.0))
    st, eff = RS.on_damage(st, page, c, u, RH.ev(c, n, report=RH.report(p, u)))   # 60 within 2.5 s
    shield = 40.0 + 0.06 * 400.0
    np.testing.assert_allclose(np.asarray(eff.shields.amount[0]), [shield, 0.0], rtol=1e-5)
    np.testing.assert_allclose(np.asarray(eff.shields.amount[1]), [0.0, shield], rtol=1e-5)
    assert f(eff.shields.duration) == pytest.approx(1.5)
    assert f(st.gd_cd_until) == pytest.approx(76.0)
    # Damage older than 2.5 s falls out of the window.
    st = RS.init(2, n)
    st, _ = RS.on_damage(st, page, ctx, u, RH.ev(ctx, n, report=RH.report(p, u)))
    c = ctx._replace(now=jnp.float32(3.0))
    _, eff = RS.on_damage(st, page, c, u, RH.ev(c, n, report=RH.report(p, u)))
    assert float(jnp.sum(eff.shields.amount)) == 0.0


def test_guardian_unit_targeted_cast_guards_distant_ally():
    rows = [dict(x=0, y=0, team=0, cls=D.CLASS_CHAMPION), dict(x=900, y=0, team=0, cls=D.CLASS_CHAMPION),
            dict(x=1000, y=0, team=1, cls=D.CLASS_CHAMPION)]
    u = H.units(rows)
    n = u.x.shape[0]
    page = RH.page(GUARDIAN)
    ctx = H.ctx(team=(0, 0), x=jnp.asarray([0.0, 900.0]))
    cast = Cast(jnp.asarray([True, False]), jnp.zeros((2,), jnp.int32), jnp.asarray([1, -1], jnp.int32))
    st, _ = RS.on_cast(RS.init(2, n), page, ctx, u, RH.ev(ctx, n, cast=cast))
    assert f(st.gd_guard_until[0, 1]) == pytest.approx(2.5)
    _, eff = RS.on_damage(st, page, ctx, u, RH.ev(ctx, n, report=RH.report(RH.hit_packet(2, 1, 60.0), u)))
    assert f(eff.shields.amount[1, 1]) > 0.0
