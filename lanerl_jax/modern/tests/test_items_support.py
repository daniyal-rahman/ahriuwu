"""Support items and support quest line (items.effects/support.py)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.modern.core import damage as D
from lanerl_jax.modern.items import effects as E
from lanerl_jax.modern.items.effects import support as S
from lanerl_jax.modern.tests import item_harness as H

ASSIGNED = (2065, 2524, 3050, 3107, 3109, 3190, 3222, 3504, 4005, 6616, 6617, 6620, 6621,
            3865, 3867, 3869, 3870, 3871, 3876)


def world(extra=(), x1=300.0):
    return H.units(H.champions(x1=x1) + list(extra))


def ally_row(x=200.0, hp=500.0):
    return dict(x=x, y=0.0, team=0, cls=D.CLASS_CHAMPION, radius=65.0, hp=hp, max_hp=1000.0)


def report_of(p, u, **kw):
    return H.resolve(p, u, **kw)[0]


def nocc(n):
    return jnp.zeros((2, n), bool)


def test_coverage_lists_every_assigned_item():
    assert set(S.COVERAGE) == set(ASSIGNED)
    assert not (set(ASSIGNED) & (E.STATS_ONLY | set(E.DEFERRED)))


def test_zekes_storm_ticks_slow_and_cooldown():
    u = world([dict(x=100, y=0, team=1)])                  # enemy minion inside the storm
    n = 4 - 1
    own = H.own([3050], [])
    ctx = H.ctx(now=0.0, dt=0.1)
    st = S.init(2, n)
    assert float(S.stats(st, own, ctx).ultimate_haste[0]) == 15 and float(S.stats(st, own, ctx).ultimate_haste[1]) == 0
    st, _ = S.on_cast(st, own, ctx, u, H.cast(started=(True, True), slot=(3, 3)))
    assert float(st.zeke_cd[0]) == 45 and float(st.zeke_cd[1]) < 0      # non-holder untouched
    per = jax.jit(S.periodic)
    total = 0.0
    slowed = False
    for k in range(80):                                    # 8 s at dt 0.1
        st, eff = per(st, own, H.ctx(now=k * 0.1, dt=0.1), u)
        total += H.packet_total(eff.packets, item=3050, dst=1)
        assert H.packet_total(eff.packets, item=3050, dst=2) == 0.0   # minions are not hit
        if k == 0:
            assert float(eff.slow[1]) == pytest.approx(0.30) and float(eff.slow[2]) == 0.0
            slowed = True
    assert slowed and total == pytest.approx(150.0)        # 30/s for 5 s
    # Recast inside the cooldown does not ready a new storm.
    st, _ = S.on_cast(st, own, H.ctx(now=10.0), u, H.cast(slot=(3, 3)))
    assert float(st.zeke_ready_until[0]) < 0


def test_zekes_storm_summons_at_window_end_without_contact():
    u = world(x1=2000.0)
    own = H.own([3050], [])
    st, _ = S.on_cast(S.init(2, 2), own, H.ctx(), u, H.cast(slot=(3, 0)))
    st, _ = S.periodic(st, own, H.ctx(now=4.9), u)
    assert float(st.zeke_storm_until[0]) < 0
    st, _ = S.periodic(st, own, H.ctx(now=5.0), u)
    assert float(st.zeke_storm_until[0]) == pytest.approx(10.0)


def test_bandlepipes_fanfare_melee_and_ranged():
    u = world()
    own = H.own([2524], [2524])
    st = S.init(2, 2)
    slowed = jnp.asarray([[False, True], [False, False]])
    st, _ = S.on_cc(st, own, H.ctx(), u, S.CC(slowed, nocc(2)))
    s = S.stats(st, own, H.ctx(now=7.9))
    assert float(s.attack_speed[0]) == pytest.approx(0.30) and float(s.move_speed[0]) == 20
    assert float(s.attack_speed[1]) == 0.0                 # holder 1 applied no CC
    assert float(S.stats(st, own, H.ctx(now=8.0)).move_speed[0]) == 0.0
    st, _ = S.on_cc(S.init(2, 2), own, H.ctx(ranged=True), u, S.CC(slowed, nocc(2)))
    s = S.stats(st, own, H.ctx(now=3.9, ranged=True))
    assert float(s.attack_speed[0]) == pytest.approx(0.30 * 0.667) and float(st.fanfare_until[0]) == 4.0
    # Ally aura
    u3 = world([ally_row()])
    aura = S.bandlepipes_aura(st, own, H.ctx(now=1.0, ranged=True), u3)
    assert float(aura[0, 2]) == pytest.approx(0.2001) and float(aura[0, 1]) == 0.0


def test_imperial_mandate_vulnerability_refreshes_not_stacks():
    u = world()
    own = H.own([4005], [])
    imm = jnp.asarray([[False, True], [False, False]])
    st, _ = S.on_cc(S.init(2, 2), own, H.ctx(), u, S.CC(nocc(2), imm))
    assert float(S.debuffs(st, own, H.ctx(now=3.9), u).received_amp[1]) == pytest.approx(0.07)
    st, _ = S.on_cc(st, own, H.ctx(now=3.0), u, S.CC(nocc(2), imm))
    assert float(S.debuffs(st, own, H.ctx(now=6.9), u).received_amp[1]) == pytest.approx(0.07)
    assert float(S.debuffs(st, own, H.ctx(now=7.0), u).received_amp[1]) == 0.0
    st2, _ = S.on_cc(S.init(2, 2), own, H.ctx(), u, S.CC(imm, nocc(2)))   # slow only: no mark
    assert float(S.debuffs(st2, own, H.ctx(), u).received_amp[1]) == 0.0
    assert np.allclose(S.mandate_immobilize_haste(own), [20, 0])


def test_solstice_sleigh_heal_ms_and_cooldown():
    u = world([ally_row(hp=300.0), ally_row(x=-300.0, hp=900.0)])
    own = H.own([3876], [])
    slowed = jnp.zeros((2, 4), bool).at[0, 1].set(True)
    st, eff = S.on_cc(S.init(2, 4), own, H.ctx(level=10), u, S.CC(slowed, jnp.zeros((2, 4), bool)))
    assert float(eff.heal[0]) == pytest.approx(50 + 15 * 4) and float(eff.heal[1]) == 0.0
    assert int(st.sleigh_ally[0]) == 2                     # most wounded ally
    assert float(S.stats(st, own, H.ctx(now=1.25)).percent_move_speed[0]) == pytest.approx(0.10)
    st, eff = S.on_cc(st, own, H.ctx(now=29.0), u, S.CC(slowed, jnp.zeros((2, 4), bool)))
    assert float(eff.heal[0]) == 0.0                       # cd 30
    _, eff = S.on_cc(st, own, H.ctx(now=30.0, level=1), u, S.CC(slowed, jnp.zeros((2, 4), bool)))
    assert float(eff.heal[0]) == 50


def test_ally_support_censer_flowing_echoes_moonstone_dream():
    u = world([ally_row(hp=800.0), ally_row(x=600.0, hp=200.0)])
    own = H.own([3504, 6616, 6620, 6617, 3870], [3504])
    st = S.init(2, 4)._replace(echoes_charges=jnp.asarray([70.0, 50.0]))
    ev = S.AllySupport(jnp.asarray([2, -1], jnp.int32), jnp.asarray([100.0, 0.0]), jnp.asarray([0.0, 0.0]))
    st, _, out = S.on_ally_support(st, own, H.ctx(level=7), u, ev)
    s = S.stats(st, own, H.ctx(now=5.9))
    assert float(s.attack_speed[0]) == pytest.approx(0.25) and float(s.attack_speed[1]) == 0.0
    assert float(s.ability_power[0]) == pytest.approx(40) and float(s.ability_haste[0]) == 15
    assert float(S.stats(st, own, H.ctx(now=6.0)).ability_power[0]) == 0.0
    assert float(out.echoes_heal[0]) == 70 and float(st.echoes_charges[0]) == 0.0
    assert float(st.echoes_charges[1]) == 50               # non-holder untouched
    assert int(out.moonstone_target[0]) == 3 and float(out.moonstone_heal[0]) == pytest.approx(30.0)
    assert bool(out.dream[0]) and float(out.dream_flat_dr[0]) == 62 and float(out.dream_proc[0]) == 50
    assert bool(out.censer[0]) and bool(out.flowing[0]) and not bool(out.censer[1])
    # Dream bubbles recharge 8 s; self-heal (target = holder) triggers nothing.
    _, _, out2 = S.on_ally_support(st, own, H.ctx(now=7.9), u, ev)
    assert not bool(out2.dream[0])
    self_ev = S.AllySupport(jnp.asarray([0, -1], jnp.int32), jnp.asarray([100.0, 0.0]), jnp.zeros(2))
    _, _, out3 = S.on_ally_support(S.init(2, 4), own, H.ctx(), u, self_ev)
    assert int(out3.target[0]) == -1 and not bool(out3.censer[0])
    # Censer on-hit while Sanctified.
    _, eff = S.on_hit(st, own, H.ctx(now=1.0), u, H.attack())
    assert H.packet_total(eff.packets, item=3504, dst=1) == 20


def test_moonstone_single_target_shield():
    u = world([ally_row()])
    own = H.own([6617], [])
    ev = S.AllySupport(jnp.asarray([2, -1], jnp.int32), jnp.zeros(2), jnp.asarray([200.0, 0.0]))
    _, _, out = S.on_ally_support(S.init(2, 3), own, H.ctx(), u, ev)
    assert int(out.moonstone_target[0]) == 2 and float(out.moonstone_shield[0]) == pytest.approx(70.0)


def test_echoes_charges_from_damage_and_cap():
    u = world([dict(x=100, y=0, team=1)])
    own = H.own([6620], [6620])
    p = D.packets(jnp.asarray([True, True, True]), jnp.asarray([0, 0, 1]), jnp.asarray([1, 2, 0]),
                  jnp.asarray([100.0, 500.0, 40.0]), D.MAGIC)
    rep = report_of(p, u)
    st, _ = S.on_damage(S.init(2, 3), own, H.ctx(level=1), u, rep)
    assert float(st.echoes_charges[0]) == pytest.approx(30.0)   # minion damage ignored
    assert float(st.echoes_charges[1]) == pytest.approx(12.0)
    for _ in range(3):
        st, _ = S.on_damage(st, own, H.ctx(level=1), u, rep)
    assert float(st.echoes_charges[0]) == pytest.approx(80.0)   # cap at L1
    assert float(S.echoes_cap(18)) == 250


def test_dawncore_first_light_scales_with_base_mana_regen():
    s = S.stats(S.init(2, 2), H.own([6621], [6621, 3222]), H.ctx())
    assert float(s.ability_power[0]) == pytest.approx(10.0) and float(s.heal_shield_power[0]) == pytest.approx(0.02)
    assert float(s.ability_power[1]) == pytest.approx(20.0) and float(s.heal_shield_power[1]) == pytest.approx(0.04)
    assert float(S.stats(S.init(2, 2), H.own([3222], []), H.ctx()).ability_power[0]) == 0.0


def test_support_line_gold_generation():
    own = H.own([3865], [3869])
    _, eff = S.periodic(S.init(2, 2), own, H.ctx(dt=1.0), world())
    assert np.allclose(eff.gold, [0.3, 0.9])
    _, eff = S.periodic(S.init(2, 2), H.own([3867], [3877]), H.ctx(dt=1.0), world())
    assert np.allclose(eff.gold, [0.5, 0.0])                 # Bloodsong gold is the spellblade module's


def test_world_atlas_charges_shared_riches_and_quest():
    u = world([ally_row()])
    own = H.own([3865], [])
    per = jax.jit(S.periodic)
    st = S.init(2, 3)
    st, _ = per(st, own, H.ctx(now=5.0), u)
    assert float(st.atlas_next_charge[0]) == 25.0 and float(st.atlas_charges[0]) == 0
    for t in (25.0, 45.0, 65.0, 85.0):
        st, _ = per(st, own, H.ctx(now=t), u)
    assert float(st.atlas_charges[0]) == 3                  # max 3
    p = D.packets(jnp.asarray([True]), 0, 1, 50.0, D.PHYSICAL, D.BASIC_ATTACK)
    rep = report_of(p, u)
    st1, eff = S.on_damage(st, own, H.ctx(now=86.0), u, rep)
    assert float(eff.gold[0]) == 18 and float(st1.atlas_charges[0]) == 2 and float(eff.gold[1]) == 0
    # No ally nearby: nothing.
    u_far = world([ally_row(x=3000.0)])
    _, eff = S.on_damage(st, own, H.ctx(now=86.0), u_far, rep)
    assert float(eff.gold[0]) == 0
    # Minion kills: cannon 20 + melee 18, two charges consumed, kill gold redirected.
    u4 = world([ally_row(), dict(x=100, y=0, team=1, siege=True), dict(x=120, y=0, team=1)])
    st4 = S.init(2, 5)._replace(atlas_charges=jnp.asarray([2.0, 0.0]))
    ku = jnp.zeros((2, 5), bool).at[0, 3].set(True).at[0, 4].set(True)
    st4, eff = S.on_takedown(st4, own, H.ctx(), u4, H.kills(5, killed_units=ku))
    assert float(eff.gold[0]) == 38 and float(st4.atlas_charges[0]) == 0 and float(st4.atlas_redirect[0]) == 2
    # Quest completes at 400 gold earned.
    stq = S.init(2, 3)._replace(atlas_gold=jnp.asarray([390.0, 0.0]), atlas_charges=jnp.asarray([1.0, 0.0]))
    stq, _ = S.on_damage(stq, own, H.ctx(), u, rep)
    assert bool(stq.atlas_done[0]) and not bool(stq.atlas_done[1])


def test_world_atlas_execute():
    u = world([ally_row(), dict(x=100, y=0, team=1, hp=540.0, max_hp=1000.0)])
    own = H.own([3865], [])
    st = S.init(2, 4)._replace(atlas_charges=jnp.asarray([1.0, 0.0]))
    atk = H.attack(target=(3, 0))
    _, eff = S.on_hit(st, own, H.ctx(base_ad=60.0), u, atk)    # 540 <= 0.5*1000 + 60
    assert H.packet_targets(eff.packets, item=3865) == [3]
    _, eff = S.on_hit(st, own, H.ctx(base_ad=30.0), u, atk)    # 540 > 530
    assert H.packet_targets(eff.packets, item=3865) == []
    _, eff = S.on_hit(st, own, H.ctx(base_ad=60.0, ranged=True), u, atk)   # 33.3%
    assert H.packet_targets(eff.packets, item=3865) == []


def test_celestial_opposition_blessing_cycle():
    u = world([dict(x=400, y=0, team=1)])
    own = H.own([3869], [])
    st = S.init(2, 3)
    assert np.allclose(S.celestial_champion_damage_mult(st, own, H.ctx()), [0.65, 1.0])
    assert float(S.celestial_champion_damage_mult(st, own, H.ctx(ranged=True))[0]) == pytest.approx(0.75)
    champ = report_of(D.packets(jnp.asarray([True]), 1, 0, 100.0, D.PHYSICAL, D.BASIC_ATTACK), u)
    minion = report_of(D.packets(jnp.asarray([True]), 2, 0, 100.0, D.PHYSICAL, D.BASIC_ATTACK), u)
    st, _ = S.on_damage(st, own, H.ctx(now=1.0), u, minion)
    assert not bool(st.cel_popped[0])                      # minion damage does not pop
    st, _ = S.on_damage(st, own, H.ctx(now=1.0), u, champ)
    assert bool(st.cel_popped[0]) and float(st.cel_linger_until[0]) == 3.0
    st, eff = S.periodic(st, own, H.ctx(now=2.9), u)
    assert float(eff.slow[1]) == 0.0 and float(S.celestial_champion_damage_mult(st, own, H.ctx(now=2.9))[0]) == pytest.approx(0.65)
    st, eff = S.periodic(st, own, H.ctx(now=3.0), u)
    assert float(eff.slow[1]) == pytest.approx(0.5) and float(eff.slow[2]) == pytest.approx(0.5)
    assert float(eff.slow_duration[1]) == 1.5 and float(st.cel_cd_until[0]) == 21.0
    assert float(S.celestial_champion_damage_mult(st, own, H.ctx(now=4.0))[0]) == 1.0
    st, _ = S.on_damage(st, own, H.ctx(now=10.0), u, champ)  # champion damage restarts the cd
    assert float(st.cel_cd_until[0]) == 28.0


def test_zazzak_void_explosion():
    u = world([dict(x=350, y=0, team=1, max_hp=1000.0)])
    own = H.own([3871], [])
    ability = report_of(D.packets(jnp.asarray([True]), 0, 1, 80.0, D.MAGIC, 0), u)
    attack = report_of(D.packets(jnp.asarray([True]), 0, 1, 80.0, D.PHYSICAL, D.BASIC_ATTACK), u)
    st, _ = S.on_damage(S.init(2, 3), own, H.ctx(ap=100.0), u, attack)
    assert np.isinf(float(st.zaz_at[0]))                    # basic attacks do not trigger
    st, _ = S.on_damage(st, own, H.ctx(ap=100.0), u, ability)
    assert float(st.zaz_at[0]) == 0.5 and float(st.zaz_cd[0]) == 10.0
    st, eff = S.periodic(st, own, H.ctx(now=0.4, ap=100.0), u)
    assert H.packet_total(eff.packets, item=3871) == 0.0
    st, eff = S.periodic(st, own, H.ctx(now=0.5, ap=100.0), u)
    # 10 + 15% of 100 AP + 3% of 1000 max HP, champion (x=300) and minion (x=350) in r250
    assert H.packet_total(eff.packets, item=3871, dst=1) == pytest.approx(55.0)
    assert H.packet_total(eff.packets, item=3871, dst=2) == pytest.approx(55.0)
    st, _ = S.on_damage(st, own, H.ctx(now=5.0, ap=100.0), u, ability)
    assert np.isinf(float(st.zaz_at[0]))                    # cd 10


def test_active_amount_helpers():
    assert float(S.locket_shield(8)) == 290 and float(S.locket_shield(18)) == 360
    assert float(S.redemption_heal(1)) == 150 and float(S.redemption_heal(18)) == 350


def test_jit_hooks_and_registry_dispatch():
    u = world()
    own = H.own([3050, 6620, 3869], [])
    st = E.init(2, 2)
    p = D.packets(jnp.asarray([True]), 0, 1, 100.0, D.MAGIC, 0)
    rep = report_of(p, u)
    st2, eff = jax.jit(E.on_damage)(st, own, H.ctx(), u, rep)
    assert float(st2.support.echoes_charges[0]) == pytest.approx(30.0)
    s = jax.jit(E.dynamic_stats)(st2, own, H.ctx())
    assert float(s.ultimate_haste[0]) == 15
    st3, eff = jax.jit(E.periodic)(st2, own, H.ctx(), u)
    assert eff.gold.shape == (2,)
