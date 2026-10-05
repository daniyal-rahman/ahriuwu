"""Marksman/lethality item effects (items.effects.marksman), client 16.19."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lanerl_jax.modern.items.effects as E
from lanerl_jax.modern.core import damage as D
from lanerl_jax.modern.items.effects import marksman as M
from lanerl_jax.modern.tests import item_harness as H


def _st(n):
    return E.init(2, n)


def _m(st):
    return st.marksman


def _world(extra=()):
    return H.units(H.champions(x1=300.0) + list(extra))


def test_coverage_complete():
    assigned = {1043, 2512, 2523, 3032, 3036, 3046, 3072, 3085, 3087, 3094, 3095, 3124, 3144, 6670, 6672,
                6675, 6676, 3142, 6697, 6696, 3179, 6695, 6699}
    assert set(M.COVERAGE) == assigned
    # coverage_report() needs every module filled in; check no double classification here.
    assert not assigned & E.STATS_ONLY
    assert not assigned & set(E.DEFERRED)


def test_recurve_and_guinsoo_on_hit_and_isolation():
    u = _world()
    own = H.own([1043, 3124], [])
    st, eff = E.on_hit(_st(2), own, H.ctx(), u, H.attack())
    p = eff.packets
    assert H.packet_total(p, item=1043) == pytest.approx(15.0)
    assert H.packet_total(p, item=3124) == pytest.approx(30.0)
    sel = np.asarray(p.valid) & (np.asarray(p.item) == 3124)
    assert np.all(np.asarray(p.dtype)[sel] == D.MAGIC)
    assert np.all((np.asarray(p.flags)[sel] & D.PROP_LIFESTEAL) != 0)
    # non-holder (holder 1 hits too) emits nothing
    st, eff = E.on_hit(_st(2), own, H.ctx(), u, H.attack(hit=(False, True), target=(1, 0)))
    assert H.packet_total(eff.packets) == 0.0


def test_guinsoo_stacks_phantom_and_expiry():
    u = _world()
    own = H.own([3124], [])
    st = _st(2)
    due = []
    for k in range(9):
        ctx = H.ctx(now=0.5 * k)
        st, _ = E.on_attack(st, own, ctx, u, H.attack())
        st, _ = E.on_hit(st, own, ctx, u, H.attack())
        due.append(bool(M.phantom_hit_due(_m(st), ctx)[0]))
    s = E.dynamic_stats(st, own, H.ctx(now=4.0))
    assert float(s.attack_speed[0]) == pytest.approx(0.32)
    assert float(s.attack_speed[1]) == 0.0
    # 4th attack reaches max and gives Phantom 1, 5th Phantom 2, 6th fires; then every third.
    assert due == [False, False, False, False, False, True, False, False, True]
    s = E.dynamic_stats(st, own, H.ctx(now=4.0 + 4.01))
    assert float(s.attack_speed[0]) == 0.0


def test_kraken_melee_every_third_hit_and_missing_hp():
    u = _world()
    u = u._replace(hp=u.hp.at[1].set(500.0))  # 50% missing
    own = H.own([6672], [])
    st = _st(2)
    totals = []
    for k in range(3):
        ctx = H.ctx(now=k * 1.0, level=12)
        st, _ = E.on_attack(st, own, ctx, u, H.attack())
        st, eff = E.on_hit(st, own, ctx, u, H.attack())
        totals.append(H.packet_total(eff.packets, item=6672))
    assert totals[:2] == [0.0, 0.0]
    assert totals[2] == pytest.approx(170.0 * (1 + 0.75 * 0.5))       # F15: L12 170


def test_kraken_ranged_and_stack_expiry():
    u = _world()
    own = H.own([6672], [])
    st = _st(2)
    totals = []
    for t in (0.0, 1.0, 6.0, 7.0, 8.0):     # stacks expire between 1.0 and 6.0
        ctx = H.ctx(now=t, ranged=True)
        st, _ = E.on_attack(st, own, ctx, u, H.attack())
        st, eff = E.on_hit(st, own, ctx, u, H.attack())
        totals.append(H.packet_total(eff.packets, item=6672))
    assert totals == [0.0, 0.0, 0.0, 0.0, pytest.approx(120.0)]       # 150 x 0.8


def test_energized_stacks_and_rfc_stormrazor():
    u = _world()
    own = H.own([3094, 3095], [])
    st = _st(2)
    ctx = H.ctx(moved=24.0 * 94)
    st, _ = E.periodic(st, own, ctx, u)
    assert float(_m(st).energy[0]) == pytest.approx(94.0)
    assert float(_m(st).energy[1]) == 0.0
    st, _ = E.on_attack(st, own, H.ctx(), u, H.attack())   # +6 -> 100, not energized
    st, eff = E.on_hit(st, own, H.ctx(), u, H.attack())
    assert H.packet_total(eff.packets, item=3094) == 0.0
    assert float(M.attack_range_bonus(_m(st), own, H.ctx(), 550.0)[0]) == pytest.approx(150.0)
    assert float(M.attack_range_bonus(_m(st), own, H.ctx(), 300.0)[0]) == pytest.approx(105.0)
    st, _ = E.on_attack(st, own, H.ctx(now=1.0), u, H.attack())
    st, eff = E.on_hit(st, own, H.ctx(now=1.0), u, H.attack())
    assert H.packet_total(eff.packets, item=3094) == pytest.approx(40.0)
    assert H.packet_total(eff.packets, item=3095) == pytest.approx(100.0)
    assert float(_m(st).energy[0]) == 0.0
    s = E.dynamic_stats(st, own, H.ctx(now=2.4))
    assert float(s.percent_move_speed[0]) == pytest.approx(0.45)
    s = E.dynamic_stats(st, own, H.ctx(now=2.6))
    assert float(s.percent_move_speed[0]) == 0.0


def test_statikk_chain_and_bounce_count():
    assert [float(M.statikk_bounces(L)) for L in (1, 6, 10, 14, 18, 20)] == [4, 5, 6, 7, 7, 8]
    minions = [dict(x=300.0 + 200 * i, y=0.0, team=1) for i in range(1, 7)]
    u = _world(minions)
    n = u.x.shape[0]
    own = H.own([3087], [])
    st = _st(n)
    st = st._replace(marksman=_m(st)._replace(energy=jnp.asarray([100.0, 0.0])))
    ctx = H.ctx()
    st, _ = E.on_attack(st, own, ctx, u, H.attack())
    st, eff = E.on_hit(st, own, ctx, u, H.attack())
    p = eff.packets
    assert H.packet_targets(p, item=3087) == [1, 2, 3, 4]       # 4 targets at L1
    assert H.packet_total(p, item=3087, dst=1) == pytest.approx(60.0)
    assert H.packet_total(p, item=3087, dst=2) == pytest.approx(90.0)
    extra = np.asarray(M.extra_on_hit_targets(_m(st), ctx))
    assert extra[0].nonzero()[0].tolist() == [2, 3, 4]
    # Statikk attack charge: 6 + 9
    st2, _ = E.on_attack(_st(n), own, ctx, u, H.attack())
    assert float(_m(st2).energy[0]) == pytest.approx(15.0)


def test_voltaic_firmament_and_galvanize():
    u = _world([dict(x=400.0, y=0.0, team=1, hp=5000.0, max_hp=5000.0)])
    own = H.own([6699], [])
    st = _st(3)
    st = st._replace(marksman=_m(st)._replace(energy=jnp.asarray([100.0, 0.0]), en_pending=jnp.asarray([True, False])))
    st2, eff = E.on_hit(st, own, H.ctx(), u, H.attack())
    assert H.packet_total(eff.packets, item=6699) == pytest.approx(90.0)     # 9% x 1000 champion
    s = E.dynamic_stats(st2, own, H.ctx(now=3.9))
    assert float(s.lethality[0]) == pytest.approx(15.0)
    st3, eff = E.on_hit(st, own, H.ctx(), u, H.attack(target=(2, 0)))
    assert H.packet_total(eff.packets, item=6699) == pytest.approx(200.0)    # cap vs minion
    # Galvanize: ability damage to a champion fires Energized
    st = _st(3)._replace(marksman=_m(_st(3))._replace(energy=jnp.asarray([100.0, 0.0])))
    p = D.packets(jnp.asarray([True]), 0, 1, 100.0, D.PHYSICAL, D.TAG_ACTIVE_SPELL)
    rep, _ = H.resolve(p, u)
    st, eff = E.on_damage(st, own, H.ctx(ranged=True), u, rep)
    assert H.packet_total(eff.packets, item=6699) == pytest.approx(70.0)
    assert float(_m(st).energy[0]) == 0.0


def test_yuntal_crit_and_flurry():
    u = _world()
    own = H.own([3032], [])
    st = _st(2)
    for k in range(70):
        ctx = H.ctx(now=0.1 * k)
        st, _ = E.on_attack(st, own, ctx, u, H.attack())
        st, _ = E.on_hit(st, own, ctx, u, H.attack())
    s = E.dynamic_stats(st, own, H.ctx(now=7.0))
    assert float(s.crit_chance[0]) == pytest.approx(0.25)
    st1, _ = E.on_attack(_st(2), own, H.ctx(ranged=True), u, H.attack())
    assert float(_m(st1).yt_crit[0]) == pytest.approx(0.002)
    s = E.dynamic_stats(st1, own, H.ctx(now=5.9))
    assert float(s.attack_speed[0]) == pytest.approx(0.30)
    assert float(_m(st1).yt_cd_until[0]) == pytest.approx(30.0)
    st1, _ = E.on_hit(st1, own, H.ctx(), u, H.attack(crit=(True, False)))
    assert float(_m(st1).yt_cd_until[0]) == pytest.approx(28.0)


def test_fiendhunter_barrage():
    u = _world()
    own = H.own([2512], [])
    s = E.dynamic_stats(_st(2), own, H.ctx())
    assert float(s.ultimate_haste[0]) == pytest.approx(30.0)
    st, _ = E.on_cast(_st(2), own, H.ctx(), u, H.cast(slot=(3, 0)))
    am = E.attack_mods(st, own, H.ctx(), u, jnp.asarray([1, 0]))
    assert bool(am.force_crit[0]) and not bool(am.force_crit[1])
    assert float(am.crit_scale[0]) == pytest.approx(0.8)
    s = E.dynamic_stats(st, own, H.ctx(now=1.0))
    assert float(s.attack_speed[0]) == pytest.approx(0.5)
    ctx = H.ctx(base_ad=100.0)
    # natural crit: raw 200 > forced 180 -> 30 true
    st, _ = E.on_attack(st, own, ctx, u, H.attack())
    st, eff = E.on_hit(st, own, ctx, u, H.attack(raw=(200.0, 0.0), crit=(True, False)))
    assert H.packet_total(eff.packets, item=2512) == pytest.approx(30.0)
    # forced crit: no true damage
    st, _ = E.on_attack(st, own, ctx, u, H.attack())
    st, eff = E.on_hit(st, own, ctx, u, H.attack(raw=(180.0, 0.0), crit=(True, False)))
    assert H.packet_total(eff.packets, item=2512) == 0.0
    st, _ = E.on_attack(st, own, ctx, u, H.attack())
    assert float(_m(st).fh_charges[0]) == 0.0
    am = E.attack_mods(st, own, ctx, u, jnp.asarray([1, 0]))
    assert not bool(am.force_crit[0])
    # cooldown 45 s from the cast
    st, _ = E.on_cast(st, own, H.ctx(now=10.0), u, H.cast(slot=(3, 0)))
    assert float(_m(st).fh_charges[0]) == 0.0


def test_runaans_bolts():
    minions = [dict(x=200.0, y=100.0, team=1), dict(x=250.0, y=-50.0, team=1), dict(x=-200.0, y=0.0, team=1),
               dict(x=500.0, y=0.0, team=1)]
    u = _world(minions)
    own = H.own([3085], [])
    ctx = H.ctx(ranged=True, base_ad=100.0, crit_damage=2.0)
    st, eff = E.on_attack(_st(u.x.shape[0]), own, ctx, u, H.attack(crit=(True, False)))
    assert H.packet_targets(eff.packets, item=3085) == [2, 3]     # nearest two in front, not primary/behind
    assert H.packet_total(eff.packets, item=3085, dst=2) == pytest.approx(130.0)
    assert np.asarray(M.extra_on_hit_targets(_m(st), ctx))[0].nonzero()[0].tolist() == [2, 3]


def test_ldr_giant_slayer_and_hexoptics_amp():
    u = H.units(H.champions(x1=630.0, bonus_hp=750.0) + [dict(x=100.0, y=0.0, team=1, bonus_hp=3000.0)])
    own = H.own([3036, 2523], [])
    amp = np.asarray(E.dealt_amp(_st(3), own, H.ctx(), u))
    assert amp[0, 1] == pytest.approx(0.075)
    assert amp[0, 2] == 0.0          # minion
    assert amp[1].sum() == 0.0
    hx = np.asarray(M.basic_attack_amp(_m(_st(3)), own, H.ctx(), u))
    assert hx[0, 1] == pytest.approx(0.10)              # edge 630-130 = 500
    assert hx[0, 2] == pytest.approx(0.0)


def test_phantom_dancer_ghosted():
    own = H.own([3046], [])
    assert np.asarray(E.status(_st(2), own, H.ctx()).ghosted).tolist() == [True, False]


def test_bloodthirster_ichorshield():
    u = _world()
    own = H.own([3072], [])
    p = D.packets(jnp.asarray([True]), 0, 1, 200.0, D.PHYSICAL, D.BASIC_ATTACK)
    rep, _ = H.resolve(p, u, life_steal=[1.0, 0.0])        # heal 200 at full HP
    ctx = H.ctx(level=9)
    st, eff = E.on_damage(_st(2), own, ctx, u, rep)
    assert float(eff.shields.amount[0].sum()) == pytest.approx(180.0)    # cap level_bp(165, 15 at 9)
    assert float(eff.shields.amount[1].sum()) == 0.0
    # tracked shield still full -> no further grant
    sh = D.grant_shield(D.init_shields(2), 0, 180.0, D.SHIELD_ALL, 0.0, 1e6)
    rep2, _ = H.resolve(p, u, life_steal=[1.0, 0.0], shields=sh)
    st, eff = E.on_damage(st, own, ctx, u, rep2)
    assert float(eff.shields.amount[0].sum()) == 0.0


def test_collector_execute_and_gold():
    u = _world()
    u = u._replace(hp=u.hp.at[1].set(100.0))
    own = H.own([6676], [])
    p = D.packets(jnp.asarray([True]), 0, 1, 60.0, D.TRUE)
    rep, res = H.resolve(p, u)                    # 40 hp left < 50
    st, eff = E.on_damage(_st(2), own, H.ctx(), u, rep)
    pe = eff.packets
    sel = np.asarray(pe.valid) & ((np.asarray(pe.flags) & D.PROP_EXECUTE) != 0)
    assert np.asarray(pe.dst)[sel].tolist() == [1]
    u2 = u._replace(hp=res.hp)
    res2 = D.resolve(pe, D.default_offense(2)._replace(unit_class=u.cls),
                     D.default_defense(2)._replace(unit_class=u.cls), res.hp, res.max_hp, D.init_shields(2), 0.0)
    assert float(res2.hp[1]) <= 0.0
    st, eff = E.on_takedown(st, own, H.ctx(), u2, H.kills(2, champion_kill=(1, 1)))
    assert np.asarray(eff.gold).tolist() == [25.0, 0.0]


def test_takedown_items_hubris_axiom_hexoptics():
    u = _world()
    own = H.own([6697, 6696, 2523], [])
    p = D.packets(jnp.asarray([True]), 0, 1, 50.0, D.PHYSICAL)
    rep, _ = H.resolve(p, u)
    st, _ = E.on_damage(_st(2), own, H.ctx(), u, rep)
    ku = jnp.asarray([[False, True], [False, False]])
    ctx = H.ctx(now=2.0, lethality=18.0)
    st, _ = E.on_takedown(st, own, ctx, u, H.kills(2, champion_kill=(1, 0), killed_units=ku))
    assert float(M.ult_refund_fraction(_m(st), ctx)[0]) == pytest.approx(0.145)
    assert float(M.ult_refund_fraction(_m(st), H.ctx(now=2.1))[0]) == 0.0
    s = E.dynamic_stats(st, own, H.ctx(now=50.0))
    assert float(s.attack_damage[0]) == pytest.approx(15.0)        # 12 + 3 x 1 stack
    assert float(M.attack_range_bonus(_m(st), own, H.ctx(now=9.9), 550.0)[0]) == pytest.approx(100.0)
    assert float(M.attack_range_bonus(_m(st), own, H.ctx(now=10.1), 550.0)[0]) == 0.0
    # takedown more than 3 s after damage does nothing
    st2, _ = E.on_damage(_st(2), own, H.ctx(), u, rep)
    st2, _ = E.on_takedown(st2, own, H.ctx(now=3.5), u, H.kills(2, killed_units=ku))
    assert float(E.dynamic_stats(st2, own, H.ctx(now=4.0)).attack_damage[0]) == 0.0


def test_scouts_slingshot_and_youmuu_navori():
    u = _world()
    own = H.own([3144, 3142, 6675], [])
    p = D.packets(jnp.asarray([True]), 0, 1, 50.0, D.PHYSICAL)
    rep, _ = H.resolve(p, u)
    s = E.dynamic_stats(_st(2), own, H.ctx(ranged=True))
    assert float(s.move_speed[0]) == pytest.approx(10.0)
    st, eff = E.on_damage(_st(2), own, H.ctx(), u, rep)
    assert H.packet_total(eff.packets, item=3144) == pytest.approx(40.0)
    assert float(_m(st).sling_cd_until[0]) == pytest.approx(40.0)
    assert float(E.dynamic_stats(st, own, H.ctx(now=2.9)).move_speed[0]) == 0.0
    assert float(E.dynamic_stats(st, own, H.ctx(now=3.0)).move_speed[0]) == pytest.approx(20.0)
    st, _ = E.on_attack(st, own, H.ctx(now=1.0), u, H.attack())
    assert float(_m(st).sling_cd_until[0]) == pytest.approx(39.0)
    assert np.asarray(M.basic_cooldown_scale(_m(st), H.ctx(now=1.0))).tolist() == pytest.approx([0.85, 1.0])
    st, eff = E.on_damage(st, own, H.ctx(now=5.0), u, rep)
    assert H.packet_total(eff.packets, item=3144) == 0.0


def test_serpents_fang_reaver():
    u = _world()
    own = H.own([6695], [])
    p = D.packets(jnp.asarray([True]), 0, 1, 50.0, D.PHYSICAL)
    rep, _ = H.resolve(p, u)
    st, _ = E.on_damage(_st(2), own, H.ctx(), u, rep)
    strength, fresh = M.shield_reaver(_m(st), own, H.ctx(), u)
    assert np.asarray(strength).tolist() == pytest.approx([0.0, 0.5])
    assert np.asarray(fresh).tolist() == [False, True]
    st, _ = E.on_damage(st, own, H.ctx(now=1.0, ranged=True), u, rep)
    _, fresh = M.shield_reaver(_m(st), own, H.ctx(now=1.0), u)
    assert not bool(fresh[1])                               # already afflicted
    strength, _ = M.shield_reaver(_m(st), own, H.ctx(now=3.9, ranged=True), u)   # refreshed to 4.0
    assert float(strength[1]) == pytest.approx(0.35)
    strength, _ = M.shield_reaver(_m(st), own, H.ctx(now=4.1), u)
    assert float(strength[1]) == 0.0


def test_jit_attack_cycle():
    u = _world()
    own = H.own([3124, 6672, 3094, 1043], [])

    @jax.jit
    def step(st, now):
        ctx = H.ctx(now=now, moved=10.0)
        st, _ = E.periodic(st, own, ctx, u)
        st, e1 = E.on_attack(st, own, ctx, u, H.attack())
        st, e2 = E.on_hit(st, own, ctx, u, H.attack())
        return st, jnp.sum(jnp.where(e2.packets.valid, e2.packets.raw, 0.0))

    st = _st(2)
    totals = []
    for k in range(3):
        st, t = step(st, jnp.float32(k * 0.5))
        totals.append(float(t))
    assert totals[0] == pytest.approx(45.0)
    assert totals[2] == pytest.approx(45.0 + 150.0)
