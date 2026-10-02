"""Fighter item passives (fighter.py): client values and ITEMS.md §17 fixtures."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.sim import modern_damage as D
from lanerl_jax.sim import modern_item_effects as E
from lanerl_jax.sim.modern_item_effects import fighter as F
from lanerl_jax.sim.tests import item_harness as H

N2 = 2


def _u(extra=()):
    return H.units(H.champions(x1=300.0) + list(extra))


def _fs(st):
    return st.fighter


def _report(p, u, **kw):
    rep, _ = H.resolve(p, u, **kw)
    return rep


def _pk(src, dst, raw, dtype=D.PHYSICAL, flags=D.BASIC_ATTACK, item=0):
    return D.packets(jnp.asarray([True] * len(src)), jnp.asarray(src), jnp.asarray(dst),
                     jnp.asarray(raw, jnp.float32), dtype, flags, item=item)


def test_coverage_and_no_stats_only_overlap():
    for iid in (2501, 2517, 2520, 3071, 3073, 3153, 3161, 3181, 3742, 6333, 6609, 6610, 6692, 3123,
                3033, 6694, 2019, 2020, 3134, 3091, 3302, 3139):
        assert iid in F.COVERAGE
        assert iid not in E.STATS_ONLY and iid not in E.DEFERRED


def test_overlord_f16():
    own = H.own([2501], [])
    # bonus HP 2000 (max 2600, base 600), other-source AD 200, HP 50%.
    ctx = H.ctx(base_ad=100., bonus_ad=100., max_hp=2600., hp=1300.)
    s = E.dynamic_stats(E.init(2, 2), own, ctx)
    tyr = 50.0
    ret = 0.12 * (0.5 / 0.7) * 250.0
    np.testing.assert_allclose(float(s.attack_damage[0]), tyr + ret, rtol=1e-4)
    assert float(s.attack_damage[1]) == 0.0
    # Full Retribution below 30% HP.
    s = E.dynamic_stats(E.init(2, 2), own, H.ctx(base_ad=100., bonus_ad=100., max_hp=2600., hp=500.))
    np.testing.assert_allclose(float(s.attack_damage[0]), 50 + 0.12 * 250, rtol=1e-4)


def test_endless_hunger_famine_and_feast():
    own = H.own([2517], [])
    s = E.dynamic_stats(E.init(2, 2), own, H.ctx(bonus_ad=100.))
    np.testing.assert_allclose(float(s.ability_haste[0]), 5 + 13, rtol=1e-5)
    s = E.dynamic_stats(E.init(2, 2), own, H.ctx(bonus_ad=100., ranged=True))
    np.testing.assert_allclose(float(s.ability_haste[0]), 5 + 10, rtol=1e-5)
    u = _u()
    st = E.init(2, 2)
    ctx = H.ctx()
    st, _ = E.on_damage(st, own, ctx, u, _report(_pk([0], [1], [100.]), u))
    ku = jnp.asarray([[False, True], [False, False]])
    st, _ = E.on_takedown(st, own, H.ctx(now=2.0), u, H.kills(2, champion_kill=(1, 0), killed_units=ku))
    assert float(E.dynamic_stats(st, own, H.ctx(now=9.9)).omnivamp[0]) == pytest.approx(0.15)
    assert float(E.dynamic_stats(st, own, H.ctx(now=10.1)).omnivamp[0]) == 0.0
    # Takedown more than 3 s after the last damage: no Feast.
    st2 = E.init(2, 2)
    st2, _ = E.on_damage(st2, own, ctx, u, _report(_pk([0], [1], [100.]), u))
    st2, _ = E.on_takedown(st2, own, H.ctx(now=3.5), u, H.kills(2, killed_units=ku))
    assert float(E.dynamic_stats(st2, own, H.ctx(now=4.0)).omnivamp[0]) == 0.0


def test_black_cleaver_carve_f9_and_icd():
    own = H.own([3071], [])
    u = _u()
    st = E.init(2, 2)
    for k in range(5):
        st, _ = E.on_damage(st, own, H.ctx(now=0.1 * k), u, _report(_pk([0], [1], [50.]), u))
    deb = E.target_debuffs(st, own, H.ctx(now=0.5), u)
    np.testing.assert_allclose(float(deb.percent_armor_reduction[1]), 0.30, rtol=1e-5)
    eff = D.effective_resist(100.0, 0.0, deb.percent_armor_reduction[1], 0.0, 0.0)
    np.testing.assert_allclose(float(eff), 70.0, rtol=1e-4)
    assert float(deb.percent_armor_reduction[0]) == 0.0
    # Expires after 6 s from the last stack.
    assert float(E.target_debuffs(st, own, H.ctx(now=6.5), u).percent_armor_reduction[1]) == 0.0
    # Three non-basic packets in one tick: one stack; a basic attack adds one more.
    st = E.init(2, 2)
    p = _pk([0, 0, 0, 0], [1, 1, 1, 1], [10., 10., 10., 10.],
            flags=jnp.asarray([D.TAG_ACTIVE_SPELL] * 3 + [D.BASIC_ATTACK]))
    st, _ = E.on_damage(st, own, H.ctx(), u, _report(p, u))
    assert float(_fs(st).carve[0, 1]) == 2.0
    # Magic damage does not Carve; Fervor from physical damage: +20 MS (ranged 10) 2 s.
    st = E.init(2, 2)
    st, _ = E.on_damage(st, own, H.ctx(), u, _report(_pk([0], [1], [10.], dtype=D.MAGIC), u))
    assert float(_fs(st).carve[0, 1]) == 0.0
    st, _ = E.on_damage(st, own, H.ctx(), u, _report(_pk([0], [1], [10.]), u))
    assert float(E.dynamic_stats(st, own, H.ctx(now=1.9)).move_speed[0]) == pytest.approx(20.0)
    assert float(E.dynamic_stats(st, own, H.ctx(now=1.9, ranged=True)).move_speed[0]) == pytest.approx(10.0)
    assert float(E.dynamic_stats(st, own, H.ctx(now=2.1)).move_speed[0]) == 0.0


def test_hexplate_overdrive():
    own = H.own([3073], [])
    u = _u()
    st = E.init(2, 2)
    s = E.dynamic_stats(st, own, H.ctx())
    assert float(s.ultimate_haste[0]) == 30.0 and float(s.attack_speed[0]) == 0.0
    st, _ = E.on_cast(st, own, H.ctx(), u, H.cast(slot=(3, 3)))
    s = E.dynamic_stats(st, own, H.ctx(now=7.9))
    assert float(s.attack_speed[0]) == pytest.approx(0.5) and float(s.percent_move_speed[0]) == pytest.approx(0.2)
    s = E.dynamic_stats(st, own, H.ctx(now=7.9, ranged=True))
    assert float(s.attack_speed[0]) == pytest.approx(0.35) and float(s.percent_move_speed[0]) == pytest.approx(0.14)
    assert float(E.dynamic_stats(st, own, H.ctx(now=8.1)).attack_speed[0]) == 0.0
    # Cooldown 30 s from cast; a Q does nothing.
    st, _ = E.on_cast(st, own, H.ctx(now=20.0), u, H.cast(slot=(3, 3)))
    assert float(E.dynamic_stats(st, own, H.ctx(now=21.0)).attack_speed[0]) == 0.0
    st, _ = E.on_cast(st, own, H.ctx(now=30.0), u, H.cast(slot=(3, 3)))
    assert float(E.dynamic_stats(st, own, H.ctx(now=31.0)).attack_speed[0]) == pytest.approx(0.5)


def test_botrk_mists_edge_and_clawing_shadows():
    own = H.own([3153], [])
    u = H.units(H.champions(x1=300., hp=2000., max_hp=3000.) + [dict(x=200, y=0, team=1, hp=5000., max_hp=5000.)])
    st = E.init(2, 3)
    st, eff = E.on_hit(st, own, H.ctx(), u, H.attack())
    assert H.packet_total(eff.packets, item=3153) == pytest.approx(0.09 * 2000.)
    sel = np.asarray(eff.packets.valid) & (np.asarray(eff.packets.item) == 3153)
    assert np.all(np.asarray(eff.packets.flags)[sel] & D.PROP_LIFESTEAL)
    _, eff = E.on_hit(E.init(2, 3), own, H.ctx(ranged=True), u, H.attack())
    assert H.packet_total(eff.packets, item=3153) == pytest.approx(0.06 * 2000.)
    # Cap 100 vs minions.
    _, eff = E.on_hit(E.init(2, 3), own, H.ctx(), u, H.attack(target=(2, 0)))
    assert H.packet_total(eff.packets, item=3153) == pytest.approx(100.)
    # Third hit on a champion within 6 s slows 30% for 1 s, then 15 s cd.
    st = E.init(2, 3)
    slows = []
    for k in range(4):
        st, eff = E.on_hit(st, own, H.ctx(now=float(k)), u, H.attack())
        slows.append(float(eff.slow[1]))
    assert slows == [0.0, 0.0, pytest.approx(0.3), 0.0]
    assert float(eff.slow_duration[1]) == 0.0
    # Counter expires after 6 s.
    st = E.init(2, 3)
    st, _ = E.on_hit(st, own, H.ctx(now=0.), u, H.attack())
    st, _ = E.on_hit(st, own, H.ctx(now=1.), u, H.attack())
    st, eff = E.on_hit(st, own, H.ctx(now=8.), u, H.attack())
    assert float(eff.slow[1]) == 0.0


def test_non_holder_isolation():
    own = H.own([], [3153, 3091, 3302, 3181, 3742, 6610])
    u = _u()
    st, eff = E.on_hit(E.init(2, 2), own, H.ctx(), u, H.attack())
    assert H.packet_total(eff.packets, src=0) == 0.0
    s = E.dynamic_stats(E.init(2, 2), H.own([], [2501, 2517, 3073]), H.ctx(bonus_ad=100., max_hp=2000.))
    assert float(s.attack_damage[0]) == 0.0 and float(s.ability_haste[0]) == 0.0


def test_shojin_stats_and_focused_will():
    own = H.own([3161], [])
    u = _u()
    st = E.init(2, 2)
    assert float(E.dynamic_stats(st, own, H.ctx()).basic_ability_haste[0]) == 25.0
    ab = _pk([0], [1], [50.], flags=D.TAG_ACTIVE_SPELL)
    st, _ = E.on_cast(st, own, H.ctx(), u, H.cast())
    st, _ = E.on_damage(st, own, H.ctx(), u, _report(ab, u))
    # Same cast instance, 0.5 s later: locked out.
    st, _ = E.on_damage(st, own, H.ctx(now=0.5), u, _report(ab, u))
    assert float(_fs(st).shojin[0]) == 1.0
    # New cast: stacks immediately.
    st, _ = E.on_cast(st, own, H.ctx(now=0.6), u, H.cast())
    st, _ = E.on_damage(st, own, H.ctx(now=0.6), u, _report(ab, u))
    assert float(_fs(st).shojin[0]) == 2.0
    for k in range(5):
        st, _ = E.on_damage(st, own, H.ctx(now=2.0 + k), u, _report(ab, u))
    assert float(_fs(st).shojin[0]) == 4.0
    np.testing.assert_allclose(float(F.shojin_ability_amp(_fs(st), own, H.ctx(now=6.5))[0]), 0.12, rtol=1e-5)
    np.testing.assert_allclose(float(F.shojin_ability_amp(_fs(st), own, H.ctx(now=6.5, ranged=True))[0]), 0.06, rtol=1e-5)
    assert float(F.shojin_ability_amp(_fs(st), own, H.ctx(now=13.0))[0]) == 0.0
    # Basic attacks do not stack.
    st2, _ = E.on_damage(E.init(2, 2), own, H.ctx(), u, _report(_pk([0], [1], [50.]), u))
    assert float(_fs(st2).shojin[0]) == 0.0


def test_hullbreaker_skipper_and_boarding_party():
    own = H.own([3181], [])
    u = H.units(H.champions(x1=300.) + [dict(x=200, y=0, team=1), dict(x=600, y=0, team=1, cls=D.CLASS_STRUCTURE),
                                        dict(x=500, y=0, team=0, siege=True), dict(x=5000, y=0, team=0, siege=True),
                                        dict(x=100, y=0, team=0)])
    ctx = H.ctx(base_ad=100., max_hp=2000.)
    st = E.init(2, 7)
    # 4 attacks on a minion build stacks, the 5th (on a champion) procs.
    for k in range(4):
        st, eff = E.on_hit(st, own, H.ctx(now=float(k), base_ad=100., max_hp=2000.), u, H.attack(target=(2, 0)))
        assert H.packet_total(eff.packets, item=3181) == 0.0
    st, eff = E.on_hit(st, own, H.ctx(now=4., base_ad=100., max_hp=2000.), u, H.attack())
    assert H.packet_total(eff.packets, item=3181) == pytest.approx(1.2 * 100 + 0.05 * 2000)
    assert float(_fs(st).hull[0]) == 0.0
    # Structures, ranged holder x0.7.
    st = _fs(st)._replace(hull=jnp.asarray([4., 0.]), hull_until=jnp.asarray([100., 0.]))
    st = E.init(2, 7)._replace(fighter=st)
    _, eff = E.on_hit(st, own, H.ctx(now=5., base_ad=100., max_hp=2000., ranged=True), u, H.attack(target=(3, 0)))
    assert H.packet_total(eff.packets, item=3181) == pytest.approx(0.7 * (3 * 100 + 0.1 * 2000), rel=1e-5)
    # Stacks expire after 10 s.
    st = E.init(2, 7)
    for k in range(4):
        st, _ = E.on_hit(st, own, ctx, u, H.attack(target=(2, 0)))
    _, eff = E.on_hit(st, own, H.ctx(now=10.5, base_ad=100., max_hp=2000.), u, H.attack())
    assert H.packet_total(eff.packets, item=3181) == 0.0
    # Boarding Party: level_bp(70, +6 at L>=9), ranged x0.5, only nearby allied siege minions.
    res = F.boarding_party_resists(own, H.ctx(level=12), u)
    r = np.asarray(res)
    assert r[4] == pytest.approx(70 + 6 * 4) and r[5] == 0.0 and r[6] == 0.0 and r[2] == 0.0
    assert float(F.boarding_party_resists(own, H.ctx(level=1, ranged=True), u)[4]) == pytest.approx(35.0)


def test_dead_mans_plate():
    own = H.own([3742], [])
    u = _u()
    st = E.init(2, 2)
    for k in range(60):     # 2 s moving at 30 Hz -> 50 stacks
        st, _ = E.periodic(st, own, H.ctx(now=k / 30, moved=10.), u)
    assert float(_fs(st).dmp[0]) == pytest.approx(50.0, rel=1e-4)
    assert float(E.dynamic_stats(st, own, H.ctx()).move_speed[0]) == pytest.approx(10.0, rel=1e-4)
    # dt-agnostic: one 4 s tick reaches 100.
    st2, _ = E.periodic(E.init(2, 2), own, H.ctx(dt=4.0, moved=10.), u)
    assert float(_fs(st2).dmp[0]) == pytest.approx(100.0)
    st, eff = E.on_hit(st, own, H.ctx(base_ad=100.), u, H.attack())
    assert H.packet_total(eff.packets, item=3742) == pytest.approx(0.5 * (100 + 40), rel=1e-4)
    sel = np.asarray(eff.packets.valid) & (np.asarray(eff.packets.item) == 3742)
    assert not np.any(np.asarray(eff.packets.flags)[sel] & D.PROP_LIFESTEAL)
    assert float(_fs(st).dmp[0]) == 0.0
    # Standing still gains nothing.
    st3, _ = E.periodic(E.init(2, 2), own, H.ctx(moved=0.), u)
    assert float(_fs(st3).dmp[0]) == 0.0


def test_deaths_dance_store_bleed_and_defy():
    own = H.own([6333], [])
    u = _u()
    st = E.init(2, 2)
    dfn = E.holder_defense(st, own, H.ctx())
    assert float(dfn.store_fraction[0]) == pytest.approx(0.3) and float(dfn.store_fraction[1]) == 0.0
    assert float(E.holder_defense(st, own, H.ctx(ranged=True)).store_fraction[0]) == pytest.approx(0.1)
    # Enemy deals 300 to the holder; 90 stored.
    n = 2
    full = D.default_defense(n)._replace(unit_class=u.cls, store_fraction=jnp.asarray([0.3, 0.0]))
    rep = _report(_pk([1], [0], [300.]), u, defense=full)
    assert float(rep.resolved.dd_pool_add[0]) == pytest.approx(90.)
    st, _ = E.on_damage(st, own, H.ctx(), u, rep)
    total = 0.0
    for k in range(1, 200):
        st, eff = E.periodic(st, own, H.ctx(now=k / 30), u)
        total += H.packet_total(eff.packets, item=6333, dst=0)
        if k == 30:
            assert total == pytest.approx(30.0, rel=1e-3)    # 1/3 per second
    assert total == pytest.approx(90.0, rel=1e-4)
    sel = np.asarray(eff.packets.item) == 6333
    flags = np.asarray(eff.packets.flags)[sel]
    assert np.all(flags & D.PROP_NO_OMNIVAMP) and np.all(flags & D.PROP_NO_DAMAGE_MOD)
    assert np.all(np.asarray(eff.packets.dtype)[sel] == D.TRUE)
    # Defy: holder damaged the enemy champion, enemy dies within 3 s -> pool cleared, heal 0.75 bAD over 2 s.
    st = E.init(2, 2)
    st, _ = E.on_damage(st, own, H.ctx(), u, rep)
    st, _ = E.on_damage(st, own, H.ctx(), u, _report(_pk([0], [1], [50.]), u))
    ku = jnp.asarray([[False, True], [False, False]])
    st, _ = E.on_takedown(st, own, H.ctx(now=1.0, bonus_ad=100.), u, H.kills(2, killed_units=ku))
    healed, bleed = 0.0, 0.0
    for k in range(30, 150):
        st, eff = E.periodic(st, own, H.ctx(now=k / 30), u)
        healed += float(eff.heal[0])
        bleed += H.packet_total(eff.packets, item=6333)
    assert bleed == 0.0
    assert healed == pytest.approx(75.0, rel=1e-3)


def test_grievous_wounds_family():
    u = H.units(H.champions() + [dict(x=200, y=0, team=1)])
    for iid in (3123, 3033, 6609):
        own = H.own([iid], [])
        _, eff = E.on_damage(E.init(2, 3), own, H.ctx(), u, _report(_pk([0, 0], [1, 2], [50., 50.]), u))
        assert float(eff.grievous[1]) == 3.0 and float(eff.grievous[2]) == 0.0   # champions only
        # Abilities count (any physical damage); magic does not.
        _, eff = E.on_damage(E.init(2, 3), own, H.ctx(), u,
                             _report(_pk([0], [1], [50.], flags=D.TAG_ACTIVE_SPELL), u))
        assert float(eff.grievous[1]) == 3.0
        _, eff = E.on_damage(E.init(2, 3), own, H.ctx(), u, _report(_pk([0], [1], [50.], dtype=D.MAGIC), u))
        assert float(eff.grievous[1]) == 0.0
    # Non-holder: nothing.
    _, eff = E.on_damage(E.init(2, 3), H.own([], [3123]), H.ctx(), u, _report(_pk([0], [1], [50.]), u))
    assert float(eff.grievous[1]) == 0.0


def test_sundered_sky():
    own = H.own([6610], [])
    u = H.units(H.champions() + [dict(x=200, y=0, team=1)])
    st = E.init(2, 3)
    m = E.attack_mods(st, own, H.ctx(), u, jnp.asarray([1, 0]))
    assert bool(m.force_crit[0]) and float(m.crit_scale[0]) == pytest.approx(0.8) and not bool(m.force_crit[1])
    assert not bool(E.attack_mods(st, own, H.ctx(), u, jnp.asarray([2, 0])).force_crit[0])   # minion
    ctx = H.ctx(base_ad=100., max_hp=2000., hp=1000.)
    st, eff = E.on_hit(st, own, ctx, u, H.attack())
    assert float(eff.heal[0]) == pytest.approx(90. + 40.)
    assert not bool(E.attack_mods(st, own, H.ctx(now=9.9), u, jnp.asarray([1, 0])).force_crit[0])
    assert bool(E.attack_mods(st, own, H.ctx(now=10.1), u, jnp.asarray([1, 0])).force_crit[0])
    _, eff = E.on_hit(E.init(2, 3), own, H.ctx(base_ad=100., max_hp=2000., hp=1000., ranged=True), u, H.attack())
    assert float(eff.heal[0]) == pytest.approx(45. + 40.)
    # Overheal -> temporary bonus health for 8 s.
    st, _ = E.on_hit(E.init(2, 3), own, H.ctx(base_ad=100., max_hp=2000., hp=1950.), u, H.attack())
    assert float(E.dynamic_stats(st, own, H.ctx(now=7.9)).health[0]) == pytest.approx(90. + 2. - 50.)
    assert float(E.dynamic_stats(st, own, H.ctx(now=8.1)).health[0]) == 0.0


def test_eclipse():
    own = H.own([6692], [])
    u = H.units(H.champions(max_hp=3000.))
    st = E.init(2, 2)
    p = _pk([0], [1], [50.])
    st, eff = E.on_damage(st, own, H.ctx(bonus_ad=100.), u, _report(p, u))
    assert H.packet_total(eff.packets, item=6692) == 0.0
    st, eff = E.on_damage(st, own, H.ctx(now=1.5, bonus_ad=100.), u, _report(p, u))
    assert H.packet_total(eff.packets, item=6692, dst=1) == pytest.approx(0.08 * 3000, rel=1e-5)
    assert float(jnp.sum(eff.shields.amount[0])) == pytest.approx(150 + 40)
    assert float(eff.shields.duration[0, 0]) == 2.0
    # Cooldown 6 s.
    st, eff = E.on_damage(st, own, H.ctx(now=2.0), u, _report(p, u))
    st, eff = E.on_damage(st, own, H.ctx(now=2.5), u, _report(p, u))
    assert H.packet_total(eff.packets, item=6692) == 0.0
    # Window 2 s: hits 2.5 s apart do not proc.
    st = E.init(2, 2)
    st, _ = E.on_damage(st, own, H.ctx(), u, _report(p, u))
    st, eff = E.on_damage(st, own, H.ctx(now=2.5), u, _report(p, u))
    assert H.packet_total(eff.packets, item=6692) == 0.0
    # Ranged: 5% max HP, half shield.
    st = E.init(2, 2)
    rc = H.ctx(ranged=True, bonus_ad=100.)
    st, _ = E.on_damage(st, own, rc, u, _report(p, u))
    _, eff = E.on_damage(st, own, rc._replace(now=jnp.float32(0.5)), u, _report(p, u))
    assert H.packet_total(eff.packets, item=6692) == pytest.approx(0.05 * 3000, rel=1e-5)
    assert float(jnp.sum(eff.shields.amount[0])) == pytest.approx(95.)


def test_serylda_bitter_cold():
    own = H.own([6694], [])
    u = H.units(H.champions(hp=600., max_hp=1000.))
    ab = _pk([0], [1], [150.], flags=D.TAG_ACTIVE_SPELL)
    _, eff = E.on_damage(E.init(2, 2), own, H.ctx(), u, _report(ab, u))
    assert float(eff.slow[1]) == pytest.approx(0.3) and float(eff.slow_duration[1]) == 1.0
    _, eff = E.on_damage(E.init(2, 2), own, H.ctx(), u, _report(_pk([0], [1], [50.], flags=D.TAG_ACTIVE_SPELL), u))
    assert float(eff.slow[1]) == 0.0           # still above 50%
    _, eff = E.on_damage(E.init(2, 2), own, H.ctx(), u, _report(_pk([0], [1], [150.]), u))
    assert float(eff.slow[1]) == 0.0           # basic attack


def test_wits_end_and_terminus():
    u = H.units(H.champions() + [dict(x=600, y=0, team=1, cls=D.CLASS_STRUCTURE)])
    _, eff = E.on_hit(E.init(2, 3), H.own([3091], []), H.ctx(), u, H.attack())
    assert H.packet_total(eff.packets, item=3091) == 45.0
    sel = np.asarray(eff.packets.valid) & (np.asarray(eff.packets.item) == 3091)
    assert np.all(np.asarray(eff.packets.dtype)[sel] == D.MAGIC)
    _, eff = E.on_hit(E.init(2, 3), H.own([3091], []), H.ctx(), u, H.attack(target=(2, 0)))
    assert H.packet_total(eff.packets, item=3091) == 0.0      # structures excluded
    own = H.own([3302], [])
    st = E.init(2, 3)
    st, eff = E.on_hit(st, own, H.ctx(bonus_ad=50., ap=100.), u, H.attack())
    assert H.packet_total(eff.packets, item=3302) == pytest.approx(30 + 5 + 10)
    # Light, Dark, Light, Dark, Light, Light...
    for k in range(1, 7):
        st, _ = E.on_hit(st, own, H.ctx(now=0.5 * k), u, H.attack())
    s = E.dynamic_stats(st, own, H.ctx(now=3.1, level=14))
    assert float(s.armor[0]) == pytest.approx(3 * 8) and float(s.magic_resist[0]) == pytest.approx(3 * 8)
    assert float(s.percent_armor_pen[0]) == pytest.approx(0.3) and float(s.percent_magic_pen[0]) == pytest.approx(0.3)
    s = E.dynamic_stats(st, own, H.ctx(now=3.1, level=10))
    assert float(s.armor[0]) == pytest.approx(18)
    assert float(E.dynamic_stats(st, own, H.ctx(now=9.0)).armor[0]) == 0.0


def test_bastionbreaker():
    own = H.own([2520], [])
    u = H.units(H.champions() + [dict(x=600, y=0, team=1, cls=D.CLASS_STRUCTURE, hp=5000., max_hp=5000.)])
    ab = _pk([0], [1], [50.], flags=D.TAG_ACTIVE_SPELL)
    ctx = H.ctx(lethality=22.)
    st, eff = E.on_damage(E.init(2, 3), own, ctx, u, _report(ab, u))
    assert H.packet_total(eff.packets, item=2520, dst=1) == pytest.approx(50 + 1.5 * 22)
    st, eff = E.on_damage(st, own, H.ctx(now=5., lethality=22.), u, _report(ab, u))
    assert H.packet_total(eff.packets, item=2520) == 0.0
    st, eff = E.on_damage(st, own, H.ctx(now=20., lethality=22.), u, _report(ab, u))
    assert H.packet_total(eff.packets, item=2520) > 0.0
    _, eff = E.on_damage(E.init(2, 3), own, H.ctx(lethality=22., ranged=True), u, _report(ab, u))
    assert H.packet_total(eff.packets, item=2520) == pytest.approx(0.5 * (50 + 33))
    # Basic attacks don't trigger Shaped Charge.
    _, eff = E.on_damage(E.init(2, 3), own, ctx, u, _report(_pk([0], [1], [50.]), u))
    assert H.packet_total(eff.packets, item=2520) == 0.0
    # Sabotage: takedown -> next structure attack burns 300 + 25 leth true over 3 s.
    ku = jnp.asarray([[False, True, False], [False, False, False]])
    st, _ = E.on_takedown(st, own, H.ctx(now=21., lethality=22.), u, H.kills(3, killed_units=ku))
    st, _ = E.on_hit(st, own, H.ctx(now=30., lethality=22.), u, H.attack(target=(2, 0)))
    total = 0.0
    for k in range(120):
        st, eff = E.periodic(st, own, H.ctx(now=30. + k / 30), u)
        total += H.packet_total(eff.packets, item=2520, dst=2)
    assert total == pytest.approx(300 + 25 * 22, rel=1e-4)


def test_jit_full_hooks():
    own = H.own([3071, 3153, 6333, 6692, 3742, 6610], [3181, 3302, 2501, 3033])
    u = H.units(H.champions() + [dict(x=200, y=0, team=1)])

    @jax.jit
    def step(st, ctx):
        st, e1 = E.on_hit(st, own, ctx, u, H.attack())
        p = _pk([0, 1], [1, 0], [80., 80.])
        st, e2 = E.on_damage(st, own, ctx, u, _report(p, u))
        st, e3 = E.periodic(st, own, ctx, u)
        s = E.dynamic_stats(st, own, ctx)
        d = E.target_debuffs(st, own, ctx, u)
        return st, e1.packets.raw.sum() + e2.packets.raw.sum(), s.attack_damage, d.percent_armor_reduction

    st = E.init(2, 3)
    st, tot, ad, arpen = step(st, H.ctx(now=0.0, max_hp=2000.))
    st, tot, ad, arpen = step(st, H.ctx(now=0.5, max_hp=2000.))
    assert float(arpen[1]) == pytest.approx(1 - (1 - 0.12)) and float(tot) > 0
    assert float(ad[1]) > 0.0
