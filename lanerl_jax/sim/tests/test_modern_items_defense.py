"""Defense item effects (modern_item_effects/defense.py), patch 26.19."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.sim import modern_damage as D
from lanerl_jax.sim.modern_item_effects import defense as M
from lanerl_jax.sim.modern_item_effects.core import dv
from lanerl_jax.sim.tests import item_harness as H

ASSIGNED = {2502, 2504, 2525, 3026, 3053, 3065, 3068, 3075, 3076, 3082, 3083, 3084, 3102, 3110, 3143,
            3155, 3156, 3211, 4401, 4632, 3814, 6660, 6664, 6665, 6673, 8020, 3140, 2420, 3157}


def fold(hd, u, *, armor=0.0, mr=0.0):
    """HolderDefense of holders 0..C-1 (= units 0..C-1) into a modern_damage.Defense."""
    n = u.x.shape[0]
    c = hd.received_mult.shape[0]
    dfn = D.default_defense(n, armor=armor, magic_resist=mr)._replace(unit_class=u.cls)
    put = lambda base, v: base.at[:c].set(v)
    return dfn._replace(**{f: put(getattr(dfn, f), getattr(hd, f)) for f in hd._fields})


def pk(src, dst, raw, dtype=D.TRUE, flags=0, item=0):
    return D.packets(jnp.asarray([True]), src, dst, raw, dtype, flags, item=item)


def test_coverage_matches_assignment():
    assert set(M.COVERAGE) == ASSIGNED
    for iid in (3140, 2420, 3157, 3143):
        assert "DEFERRED" in M.COVERAGE[iid]


def test_f7_steraks_claws_and_lifeline_absorbs_trigger_packet():
    u = H.units([dict(x=0, y=0, team=0, cls=D.CLASS_CHAMPION, hp=900., max_hp=2600.),
                 dict(x=300, y=0, team=1, cls=D.CLASS_CHAMPION)])
    own = H.own([3053], [])
    ctx = H.ctx(base_ad=70., base_hp=1600., max_hp=2600., hp=900.)
    st = M.init(2, 2)
    assert float(M.stats(st, own, ctx).attack_damage[0]) == pytest.approx(35.0)
    hd = M.defense(st, own, ctx)
    assert float(hd.lifeline_shield[0]) == pytest.approx(600.0)
    assert not bool(hd.lifeline_ready[1])
    rep, res = H.resolve(pk(1, 0, 200.), u, defense=fold(hd, u))
    assert bool(res.lifeline_fired[0])
    assert float(res.hp[0]) == pytest.approx(900.0)
    assert float(res.absorbed[0]) == pytest.approx(200.0)
    assert float(D.total_shield(res.shields, 0.0)[0]) == pytest.approx(400.0)
    st, _ = M.on_damage(st, own, ctx, u, rep)
    assert float(st.lifeline_cd[0]) == pytest.approx(90.0)
    assert not bool(M.defense(st, own, H.ctx(now=89.0, base_hp=1600., max_hp=2600.)).lifeline_ready[0])
    assert bool(M.defense(st, own, H.ctx(now=90.0, base_hp=1600., max_hp=2600.)).lifeline_ready[0])
    # Decay: hold 0.75 s, then linear to 0 at 4.5 s.
    assert float(D.total_shield(res.shields, 0.75)[0]) == pytest.approx(400.0)
    assert float(D.total_shield(res.shields, 4.5)[0]) == pytest.approx(0.0)


def test_f10_wardens_block_and_randuins_crit():
    u = H.units(H.champions())
    for loadout, flags, raw, expect in (([3082], D.BASIC_ATTACK, 50., 40.),
                                        ([3082], D.BASIC_ATTACK, 100., 85.),
                                        ([3143], D.BASIC_ATTACK | D.PROP_CRIT, 100., 70.),
                                        ([3143], D.BASIC_ATTACK, 100., 100.)):
        hd = M.defense(M.init(2, 2), H.own(loadout, []), H.ctx())
        _, res = H.resolve(pk(1, 0, raw, D.PHYSICAL, flags), u, defense=fold(hd, u))
        assert float(res.final[0]) == pytest.approx(expect)
    # Non-holder (unit 1) is unaffected.
    hd = M.defense(M.init(2, 2), H.own([3082], []), H.ctx())
    _, res = H.resolve(pk(0, 1, 50., D.PHYSICAL, D.BASIC_ATTACK), u, defense=fold(hd, u))
    assert float(res.final[0]) == pytest.approx(50.0)


def test_hexdrinker_magic_only_and_level_values():
    u = H.units([dict(x=0, y=0, team=0, cls=D.CLASS_CHAMPION, hp=400., max_hp=1000.),
                 dict(x=300, y=0, team=1, cls=D.CLASS_CHAMPION)])
    ctx = H.ctx(level=10, max_hp=1000., hp=400.)
    hd = M.defense(M.init(2, 2), H.own([3155], []), ctx)
    assert float(hd.lifeline_shield[0]) == pytest.approx(200.0)
    assert int(hd.lifeline_shield_kind[0]) == D.SHIELD_MAGIC
    _, res = H.resolve(pk(1, 0, 200., D.PHYSICAL), u, defense=fold(hd, u))
    assert not bool(res.lifeline_fired[0])
    _, res = H.resolve(pk(1, 0, 200., D.MAGIC), u, defense=fold(hd, u))
    assert bool(res.lifeline_fired[0]) and float(res.hp[0]) == pytest.approx(400.0)
    ranged = M.defense(M.init(2, 2), H.own([3155], []), H.ctx(level=10, ranged=True))
    assert float(ranged.lifeline_shield[0]) == pytest.approx(150.0)
    sb = M.defense(M.init(2, 2), H.own([6673], []), H.ctx(level=13))
    assert float(sb.lifeline_shield[0]) == pytest.approx(550.0)
    maw = M.defense(M.init(2, 2), H.own([3156], []), H.ctx(bonus_ad=60.))
    assert float(maw.lifeline_shield[0]) == pytest.approx(290.0)
    assert bool(maw.lifeline_magic_only[0])


def test_maw_omnivamp_after_lifeline():
    u = H.units([dict(x=0, y=0, team=0, cls=D.CLASS_CHAMPION, hp=300., max_hp=1000.),
                 dict(x=300, y=0, team=1, cls=D.CLASS_CHAMPION)])
    own, ctx = H.own([3156], []), H.ctx(max_hp=1000., hp=300.)
    st = M.init(2, 2)
    rep, res = H.resolve(pk(1, 0, 100., D.MAGIC), u, defense=fold(M.defense(st, own, ctx), u))
    st, _ = M.on_damage(st, own, ctx, u, rep)
    assert float(M.stats(st, own, H.ctx(now=4.9)).omnivamp[0]) == pytest.approx(0.1)
    assert float(M.stats(st, own, H.ctx(now=5.1)).omnivamp[0]) == 0.0


def test_protoplasm_bonus_health_and_heal_over_time_dt_agnostic():
    u = H.units([dict(x=0, y=0, team=0, cls=D.CLASS_CHAMPION, hp=400., max_hp=1000.),
                 dict(x=300, y=0, team=1, cls=D.CLASS_CHAMPION)])
    own = H.own([2525], [])
    ctx = H.ctx(level=18, max_hp=1000., hp=400., bonus_armor=40., bonus_mr=20.)
    st0 = M.init(2, 2)
    hd = M.defense(st0, own, ctx)
    rep, res = H.resolve(pk(1, 0, 200.), u, defense=fold(hd, u))
    assert bool(res.lifeline_fired[0])
    assert float(res.max_hp[0]) == pytest.approx(1300.0)
    assert float(res.hp[0]) == pytest.approx(500.0)
    st0, _ = M.on_damage(st0, own, ctx, u, rep)
    expect = 400. + 1.75 * 40. + 1.75 * 20.
    for dt in (1 / 30, 0.25, 0.7):
        st, total, t = st0, 0.0, 0.0
        while t < 6.0:
            t += dt
            st, e = M.periodic(st, own, H.ctx(now=t, dt=dt, level=18), u)
            total += float(e.heal[0])
        assert total == pytest.approx(expect, rel=1e-4)
    s = M.stats(st0, own, H.ctx(now=2.0))
    assert float(s.health[0]) == pytest.approx(300.0)
    assert float(s.tenacity[0]) == pytest.approx(0.25)
    assert float(s.percent_move_speed[0]) == pytest.approx(0.1)
    assert float(M.stats(st0, own, H.ctx(now=5.0)).health[0]) == 0.0


def test_guardian_angel_revive_and_cooldown():
    u = H.units([dict(x=0, y=0, team=0, cls=D.CLASS_CHAMPION, hp=100.),
                 dict(x=300, y=0, team=1, cls=D.CLASS_CHAMPION)])
    own = H.own([3026], [])
    ctx = H.ctx(base_hp=800., max_mana=500.)
    st = M.init(2, 2)
    rep, res = H.resolve(pk(1, 0, 500.), u)
    st, e = M.on_damage(st, own, ctx, u, rep)
    assert bool(e.revive[0]) and not bool(e.revive[1])
    assert float(e.revive_delay[0]) == pytest.approx(4.0)
    assert float(e.revive_hp[0]) == pytest.approx(400.0)
    st, e = M.periodic(st, own, H.ctx(now=4.0, max_mana=500.), u)
    assert float(e.mana[0]) == pytest.approx(500.0)
    _, e = M.on_damage(st, own, H.ctx(now=200.0, base_hp=800.), u, rep)
    assert not bool(e.revive[0])
    _, e = M.on_damage(st, own, H.ctx(now=304.0, base_hp=800.), u, rep)
    assert bool(e.revive[0])
    # Non-lethal: nothing.
    rep2, _ = H.resolve(pk(1, 0, 50.), u)
    _, e = M.on_damage(M.init(2, 2), own, ctx, u, rep2)
    assert not bool(e.revive[0])


def test_thorns_reactive_damage_and_grievous():
    u = H.units(H.champions())
    for loadout, ba, expect in (([3075], 50., 25.), ([3076], 50., 10.)):
        own = H.own(loadout, [])
        rep, _ = H.resolve(pk(1, 0, 100., D.PHYSICAL, D.BASIC_ATTACK), u)
        _, e = M.on_damage(M.init(2, 2), own, H.ctx(bonus_armor=ba), u, rep)
        assert H.packet_total(e.packets, src=0, dst=1) == pytest.approx(expect)
        assert np.all(np.asarray(e.packets.flags)[np.asarray(e.packets.valid)] & D.PROP_REACTIVE)
        assert np.asarray(e.packets.dtype)[np.asarray(e.packets.valid)].tolist() == [D.MAGIC]
        assert float(e.grievous[1]) == pytest.approx(3.0) and float(e.grievous[0]) == 0.0
    # Abilities and reactive packets do not trigger; non-holder isolation.
    own = H.own([3075], [])
    rep, _ = H.resolve(pk(1, 0, 100., D.PHYSICAL, D.TAG_ACTIVE_SPELL), u)
    _, e = M.on_damage(M.init(2, 2), own, H.ctx(), u, rep)
    assert H.packet_total(e.packets) == 0.0
    rep, _ = H.resolve(pk(0, 1, 100., D.PHYSICAL, D.BASIC_ATTACK), u)
    _, e = M.on_damage(M.init(2, 2), own, H.ctx(), u, rep)
    assert H.packet_total(e.packets) == 0.0
    # Minion attacker: damage but no Grievous Wounds.
    um = H.units(H.champions(x1=3000.) + [dict(x=200, y=0, team=1)])
    rep, _ = H.resolve(pk(2, 0, 20., D.PHYSICAL, D.BASIC_ATTACK), um)
    _, e = M.on_damage(M.init(2, 3), own, H.ctx(), um, rep)
    assert H.packet_total(e.packets, dst=2) == pytest.approx(20.0) and float(e.grievous[2]) == 0.0


def _immolate_run(item, dt, *, bonus_hp=1000.):
    u = H.units(H.champions(x1=300.) + [dict(x=-200, y=0, team=1), dict(x=0, y=600, team=1),
                                       dict(x=100, y=100, team=1, cls=D.CLASS_MONSTER)])
    own = H.own([item], [])
    ctx = lambda t: H.ctx(now=t, dt=dt, base_hp=1000., max_hp=1000. + bonus_hp)
    st = M.init(2, 5)
    rep, _ = H.resolve(pk(1, 0, 10., D.PHYSICAL), u)
    st, _ = M.on_damage(st, own, ctx(0.0), u, rep)
    totals = np.zeros(5)
    t, ticks = 0.0, []
    while t < 5.0:
        t += dt
        st, e = M.periodic(st, own, ctx(t), u)
        if H.packet_total(e.packets) > 0:
            ticks.append(round(t, 3))
        for k in range(5):
            totals[k] += H.packet_total(e.packets, dst=k, item=item)
    return totals, ticks


def test_immolate_ticks_values_and_dt_agnostic():
    tot, ticks = _immolate_run(3068, 0.05)
    per = 20. + 0.015 * 1000.
    assert ticks == [1.0, 2.0, 3.0]
    assert tot[1] == pytest.approx(3 * per)          # champion
    assert tot[2] == pytest.approx(3 * per * 1.5)    # minion
    assert tot[3] == 0.0                             # out of range
    assert tot[4] == pytest.approx(3 * per * 1.8)    # monster
    assert tot[0] == 0.0
    for dt in (1 / 30, 0.5, 1.5):
        assert _immolate_run(3068, dt)[0] == pytest.approx(tot, rel=1e-5)
    bami, _ = _immolate_run(6660, 0.05)
    assert bami[2] == pytest.approx(3 * 15 * 1.5) and bami[4] == pytest.approx(3 * 15 * 2.0)
    hr, _ = _immolate_run(6664, 0.05)
    assert hr[1] == pytest.approx(3 * 25.) and hr[2] == pytest.approx(3 * 25. * 1.25)


def test_immolate_needs_trigger_and_own_damage_does_not_refresh():
    u = H.units(H.champions(x1=300.))
    own = H.own([3068], [])
    st, e = M.periodic(M.init(2, 2), own, H.ctx(now=2.0), u)
    assert H.packet_total(e.packets) == 0.0
    rep, _ = H.resolve(pk(0, 1, 20., D.MAGIC, item=3068), u)
    st, _ = M.on_damage(M.init(2, 2), own, H.ctx(), u, rep)
    assert float(st.immo_until[0]) < 0


def test_hollow_radiance_desolate():
    u = H.units(H.champions(x1=300.) + [dict(x=600, y=0, team=1, alive=False), dict(x=800, y=0, team=1),
                                       dict(x=1200, y=0, team=1)])
    own = H.own([6664], [])
    ctx = H.ctx(base_hp=1000., max_hp=2000.)
    ku = np.zeros((2, 5), bool)
    ku[0, 2] = True
    _, e = M.on_takedown(M.init(2, 5), own, ctx, u, H.kills(5, killed_units=ku))
    assert H.packet_targets(e.packets) == [1, 3]
    assert H.packet_total(e.packets, dst=3) == pytest.approx(2 * 25.)
    # Champion takedown only counts within 3 s of damaging them.
    u2 = H.units([dict(x=0, y=0, team=0, cls=D.CLASS_CHAMPION),
                  dict(x=300, y=0, team=1, cls=D.CLASS_CHAMPION, alive=False), dict(x=700, y=0, team=1)])
    ku = np.zeros((2, 3), bool)
    ku[0, 1] = True
    st = M.init(2, 3)
    rep, _ = H.resolve(pk(0, 1, 50.), u2)
    st, _ = M.on_damage(st, own, ctx, u2, rep)
    _, e = M.on_takedown(st, own, H.ctx(now=2.0, base_hp=1000., max_hp=2000.), u2, H.kills(3, killed_units=ku))
    assert H.packet_total(e.packets, dst=2) == pytest.approx(4 * 25.)
    _, e = M.on_takedown(st, own, H.ctx(now=4.0, base_hp=1000., max_hp=2000.), u2, H.kills(3, killed_units=ku))
    assert H.packet_total(e.packets) == 0.0


def test_annul_spell_shield_cooldown_and_restart():
    u = H.units(H.champions())
    for item, cd in ((3102, 40.), (4632, 60.), (3814, 40.)):
        own = H.own([item], [])
        st = M.init(2, 2)
        hd = M.defense(st, own, H.ctx())
        assert bool(hd.spell_shield[0]) and not bool(hd.spell_shield[1])
        rep, res = H.resolve(pk(1, 0, 100., D.MAGIC, D.TAG_ACTIVE_SPELL), u,
                             defense=fold(hd, u), offense=D.default_offense(2)._replace(unit_class=u.cls))
        assert bool(res.spell_shield_popped[0]) and float(res.final[0]) == 0.0
        st, _ = M.on_damage(st, own, H.ctx(), u, rep)
        assert float(st.annul_ready[0]) == pytest.approx(cd)
        assert not bool(M.defense(st, own, H.ctx(now=10.)).spell_shield[0])
        rep2, _ = H.resolve(pk(1, 0, 30., D.PHYSICAL, D.BASIC_ATTACK), u)
        st, _ = M.on_damage(st, own, H.ctx(now=10.), u, rep2)
        assert float(st.annul_ready[0]) == pytest.approx(10. + cd)


def test_kaenic_shield_after_15s_and_reset():
    u = H.units(H.champions())
    own = H.own([2504], [])
    st = M.init(2, 2)
    rep, res = H.resolve(pk(1, 0, 50., D.MAGIC), u)
    st, _ = M.on_damage(st, own, H.ctx(now=1.0, max_hp=2000.), u, rep)
    _, e = M.periodic(st, own, H.ctx(now=15.0, max_hp=2000.), u)
    assert float(e.shields.amount.sum()) == 0.0
    st, e = M.periodic(st, own, H.ctx(now=16.0, max_hp=2000.), u)
    assert float(e.shields.amount[0, 0]) == pytest.approx(300.0)
    assert int(e.shields.kind[0, 0]) == D.SHIELD_MAGIC
    assert float(e.shields.amount[1, 0]) == 0.0
    _, e = M.periodic(st, own, H.ctx(now=40.0, max_hp=2000.), u)
    assert float(e.shields.amount.sum()) == 0.0
    # Partially broken: regrant tops up to 15% max HP.
    sh = D.grant_shield(D.init_shields(2), 0, 300., D.SHIELD_MAGIC, 16.0, M.KAENIC_DURATION)
    rep, res = H.resolve(pk(1, 0, 100., D.MAGIC), u, shields=sh, now=20.0)
    st, _ = M.on_damage(st, own, H.ctx(now=20.0, max_hp=2000.), u, rep)
    assert float(st.kaenic_left[0]) == pytest.approx(200.0)
    _, e = M.periodic(st, own, H.ctx(now=35.0, max_hp=2000.), u)
    assert float(e.shields.amount[0, 0]) == pytest.approx(100.0)


def test_unending_despair_pulse_and_drain_heal():
    u = H.units(H.champions(x1=500.) + [dict(x=100, y=0, team=1)])
    own = H.own([2502], [])
    ctx = lambda t: H.ctx(now=t, base_hp=1000., max_hp=2000.)
    st = M.init(2, 3)
    _, e = M.periodic(st, own, ctx(0.0), u)
    assert H.packet_total(e.packets) == 0.0      # not in champion combat
    rep, _ = H.resolve(pk(1, 0, 10., D.PHYSICAL), u)
    st, _ = M.on_damage(st, own, ctx(0.0), u, rep)
    st, e = M.periodic(st, own, ctx(0.1), u)
    assert H.packet_targets(e.packets, item=2502) == [1]
    assert H.packet_total(e.packets, dst=1) == pytest.approx(30.0)
    _, e2 = M.periodic(st, own, ctx(2.0), u)
    assert H.packet_total(e2.packets) == 0.0
    rep, _ = H.resolve(e.packets, u, mr=50.)
    _, eh = M.on_damage(st, own, ctx(0.1), u, rep)
    assert float(eh.heal[0]) == pytest.approx(2.5 * 20.0, rel=1e-5)


def test_frozen_heart_and_abyssal_debuffs():
    u = H.units(H.champions(x1=600.) + [dict(x=100, y=0, team=1),
                                       dict(x=900, y=0, team=1, cls=D.CLASS_CHAMPION)])
    d = M.debuffs(M.init(2, 4), H.own([3110, 8020], []), H.ctx(), u)
    assert np.allclose(np.asarray(d.attack_speed_cripple), [0., 0.2, 0., 0.])
    assert np.allclose(np.asarray(d.magic_received_amp), [0., 0.12, 0., 0.])
    d = M.debuffs(M.init(2, 4), H.own([], []), H.ctx(), u)
    assert float(d.attack_speed_cripple.sum()) == 0.0


def test_warmogs_vitality_and_heart():
    u = H.units(H.champions())
    own = H.own([3083, 3084, 1011], [])      # 1000 + 900 + 350 item HP
    st = M.init(2, 2)
    vit = 0.12 * 2250.
    s = M.stats(st, own, H.ctx(base_hp=1000., max_hp=3250.))
    assert float(s.health[0]) == pytest.approx(vit)
    ctx = lambda t, dt=0.1: H.ctx(now=t, dt=dt, base_hp=1000., max_hp=3250.)
    _, e = M.periodic(st, own, ctx(10.5), u)
    assert float(e.heal_plain[0]) == pytest.approx(0.015 * (3250. + vit))
    _, e = M.periodic(st, own, ctx(10.4), u)
    assert float(e.heal_plain[0]) == 0.0
    rep, _ = H.resolve(pk(1, 0, 10.), u)
    st2, _ = M.on_damage(st, own, ctx(10.0), u, rep)
    _, e = M.periodic(st2, own, ctx(17.5), u)
    assert float(e.heal_plain[0]) == 0.0
    _, e = M.periodic(st2, own, ctx(18.0), u)
    assert float(e.heal_plain[0]) > 0.0
    # Below the 2000 bonus-HP threshold: no heal.
    _, e = M.periodic(st, H.own([3083], []), H.ctx(now=10.5, dt=0.1, base_hp=1000., max_hp=2000.), u)
    assert float(e.heal_plain[0]) == 0.0


def test_heartsteel_charge_proc_and_cooldown():
    u = H.units(H.champions(x1=500.))
    own = H.own([3084], [])
    st = M.init(2, 2)
    t = 0.0
    while t < 2.9:
        t += 0.1
        st, _ = M.periodic(st, own, H.ctx(now=t, dt=0.1, max_hp=2000.), u)
    ctx = H.ctx(now=t, max_hp=2000.)
    _, e = M.on_hit(st, own, ctx, u, H.attack())
    assert H.packet_total(e.packets) == 0.0
    st, _ = M.periodic(st, own, H.ctx(now=3.0, dt=0.1, max_hp=2000.), u)
    st, e = M.on_hit(st, own, H.ctx(now=3.0, max_hp=2000.), u, H.attack())
    assert H.packet_total(e.packets, dst=1, item=3084) == pytest.approx(70. + 0.06 * 2000.)
    assert float(st.hs_hp[0]) == pytest.approx(19.0)
    assert float(M.stats(st, own, H.ctx()).health[0]) == pytest.approx(19.0)
    st, _ = M.periodic(st, own, H.ctx(now=10.0, dt=7.0, max_hp=2000.), u)
    _, e = M.on_hit(st, own, H.ctx(now=10.0, max_hp=2000.), u, H.attack())
    assert H.packet_total(e.packets) == 0.0       # per-target cooldown 30 s


def test_jaksho_after_five_seconds_of_champion_combat():
    u = H.units(H.champions())
    own = H.own([6665], [])
    st = M.init(2, 2)
    rep, _ = H.resolve(pk(1, 0, 10.), u)
    for t in (0.0, 2.0, 4.0, 5.0):
        st, _ = M.on_damage(st, own, H.ctx(now=t), u, rep)
    s = M.stats(st, own, H.ctx(now=4.9, bonus_armor=100., bonus_mr=50.))
    assert float(s.armor[0]) == 0.0
    s = M.stats(st, own, H.ctx(now=5.0, bonus_armor=100., bonus_mr=50.))
    assert float(s.armor[0]) == pytest.approx(30.0) and float(s.magic_resist[0]) == pytest.approx(15.0)
    assert float(M.stats(st, own, H.ctx(now=10.1, bonus_armor=100.)).armor[0]) == 0.0


def test_force_of_nature_stacks():
    u = H.units(H.champions())
    own = H.own([4401], [])
    st = M.init(2, 2)
    rep, _ = H.resolve(D.concat_packets(pk(1, 0, 10., D.MAGIC), pk(1, 0, 10., D.MAGIC)), u)
    for k in range(8):
        st, _ = M.on_damage(st, own, H.ctx(now=0.5 * k), u, rep)
    assert float(st.fon_stacks[0]) == 4.0          # one stack per source per second
    for k in range(4):
        st, _ = M.on_damage(st, own, H.ctx(now=4.0 + k), u, rep)
    assert float(st.fon_stacks[0]) == 8.0
    s = M.stats(st, own, H.ctx(now=7.5))
    assert float(s.magic_resist[0]) == pytest.approx(70.0) and float(s.percent_move_speed[0]) == pytest.approx(0.06)
    assert float(M.stats(st, own, H.ctx(now=14.1)).magic_resist[0]) == 0.0


def test_spirit_visage_stat_and_spectres_inert():
    s = M.stats(M.init(2, 2), H.own([3065], [3211]), H.ctx())
    assert float(s.incoming_heal[0]) == pytest.approx(0.25) and float(s.incoming_heal[1]) == 0.0


def test_hooks_jit():
    u = H.units(H.champions(x1=300.) + [dict(x=100, y=0, team=1)])
    own = H.own([3068, 3075, 3053, 2504], [3026, 3102])
    st = M.init(2, 3)
    rep, _ = H.resolve(pk(1, 0, 100., D.PHYSICAL, D.BASIC_ATTACK), u)
    ctx = H.ctx()
    st_j, e_j = jax.jit(M.on_damage)(st, own, ctx, u, rep)
    st_p, e_p = M.on_damage(st, own, ctx, u, rep)
    assert H.packet_total(e_j.packets) == pytest.approx(H.packet_total(e_p.packets))
    st2, e2 = jax.jit(M.periodic)(st_j, own, H.ctx(now=1.0), u)
    assert H.packet_total(e2.packets, item=3068) > 0.0
    jax.jit(M.defense)(st2, own, ctx)
    jax.jit(M.stats)(st2, own, ctx)
    jax.jit(M.debuffs)(st2, own, ctx, u)
    jax.jit(M.on_hit)(st2, own, ctx, u, H.attack())
