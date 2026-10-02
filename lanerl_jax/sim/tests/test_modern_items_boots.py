"""Boots passives (modern_item_effects.boots)."""
import jax
import jax.numpy as jnp
import pytest

from lanerl_jax.sim import modern_damage as D
from lanerl_jax.sim import modern_item_effects as E
from lanerl_jax.sim.modern_item_effects import boots as B
from lanerl_jax.sim.tests import item_harness as H


def world(extra=()):
    return H.units(H.champions(x1=300.0) + list(extra))


def defense_for(u, hd):
    """Pipeline Defense with holder 0's HolderDefense folded in at unit 0."""
    n = u.x.shape[0]
    dfn = D.default_defense(n)._replace(unit_class=u.cls)
    return dfn._replace(basic_attack_mult=dfn.basic_attack_mult.at[0].set(hd.basic_attack_mult[0]))


@pytest.mark.parametrize("item", [B.STEELCAPS, B.ARMORED])
def test_f10_plating(item):
    u = world([dict(x=0.0, y=500.0, team=1, cls=D.CLASS_STRUCTURE)])
    own = H.own([item], [])
    hd = B.defense(B.init(2, 3), own, H.ctx())
    assert float(hd.basic_attack_mult[0]) == pytest.approx(0.9)
    assert float(hd.basic_attack_mult[1]) == 1.0
    p = D.packets(jnp.array([True, True, True]), jnp.array([1, 2, 1]), 0, 100.0, D.PHYSICAL,
                  jnp.array([D.BASIC_ATTACK, D.BASIC_ATTACK, D.TAG_ON_HIT]))
    off = D.default_offense(3)._replace(unit_class=u.cls, is_turret=jnp.array([False, False, True]))
    _, res = H.resolve(p, u, defense=defense_for(u, hd), offense=off)
    assert [round(float(x), 4) for x in res.final] == [90.0, 100.0, 100.0]


def _report(u, src, dst, dtype, flags=D.BASIC_ATTACK, raw=100.0):
    p = D.packets(True, jnp.array([src]), jnp.array([dst]), raw, dtype, flags)
    rep, _ = H.resolve(p, u)
    return rep


def test_f15_armored_advance_shield_and_cooldown():
    u = world([dict(x=0.0, y=100.0, team=1)])
    own = H.own([B.ARMORED], [])
    c = lambda t: H.ctx(now=t, level=13, base_hp=600.0, max_hp=1600.0)
    st = B.init(2, 3)
    st, eff = B.on_damage(st, own, c(0.0), u, _report(u, 1, 0, D.PHYSICAL))
    assert float(eff.shields.amount[0, 0]) == pytest.approx(220.0, rel=1e-5)
    assert int(eff.shields.kind[0, 0]) == D.SHIELD_PHYSICAL
    assert float(eff.shields.duration[0, 0]) == pytest.approx(5.0)
    assert float(eff.shields.amount[1, 0]) == 0.0
    _, eff = B.on_damage(st, own, c(14.9), u, _report(u, 1, 0, D.PHYSICAL))
    assert float(eff.shields.amount[0, 0]) == 0.0
    _, eff = B.on_damage(st, own, c(15.0), u, _report(u, 1, 0, D.PHYSICAL))
    assert float(eff.shields.amount[0, 0]) == pytest.approx(220.0, rel=1e-5)
    fresh = B.init(2, 3)
    # Magic damage or minion damage does not trigger Armored Advance.
    _, eff = B.on_damage(fresh, own, c(0.0), u, _report(u, 1, 0, D.MAGIC))
    assert float(eff.shields.amount[0, 0]) == 0.0
    _, eff = B.on_damage(fresh, own, c(0.0), u, _report(u, 2, 0, D.PHYSICAL))
    assert float(eff.shields.amount[0, 0]) == 0.0


def test_d3_chainlaced_magic_shield_levels():
    u = world()
    own = H.own([B.CHAINLACED], [])
    for level, base in ((1, 90.0), (8, 90.0), (9, 100.0), (18, 190.0), (20, 210.0)):
        _, eff = B.on_damage(B.init(2, 2), own, H.ctx(level=level), u, _report(u, 1, 0, D.MAGIC))
        assert float(eff.shields.amount[0, 0]) == pytest.approx(base)
        assert int(eff.shields.kind[0, 0]) == D.SHIELD_MAGIC
    _, eff = B.on_damage(B.init(2, 2), own, H.ctx(level=1, max_hp=1100.0), u, _report(u, 1, 0, D.PHYSICAL))
    assert float(eff.shields.amount[0, 0]) == 0.0


def test_gluttonous_slay_stacks():
    u = world()
    own = H.own([B.GLUTTONOUS], [B.GLUTTONOUS])
    st = B.init(2, 2)
    st, _ = B.on_takedown(st, own, H.ctx(), u, H.kills(2, champion_kill=(1, 0), champion_assist=(2, 0),
                                                       minion_kill=(5, 5)))
    s = B.stats(st, own, H.ctx())
    assert float(s.omnivamp[0]) == pytest.approx(0.018) and float(s.omnivamp[1]) == 0.0
    st, _ = B.on_takedown(st, own, H.ctx(), u, H.kills(2, champion_kill=(9, 0)))
    assert float(B.stats(st, own, H.ctx()).omnivamp[0]) == pytest.approx(0.06)
    # Upgrade to Immortal Path keeps stacks; selling all Slay boots clears them.
    ip = H.own([B.IMMORTAL_PATH], [])
    st2, _ = B.periodic(st, ip, H.ctx(), u)
    assert float(B.stats(st2, ip, H.ctx()).omnivamp[0]) == pytest.approx(0.06)
    none = H.own([], [])
    st3, _ = B.periodic(st, none, H.ctx(), u)
    assert float(st3.slay_stacks[0]) == 0.0
    # Non-holder gains nothing.
    st4, _ = B.on_takedown(B.init(2, 2), none, H.ctx(), u, H.kills(2, champion_kill=(1, 1)))
    assert float(jnp.max(st4.slay_stacks)) == 0.0


def test_immortal_path_now_and_forever():
    u = world()
    own = H.own([B.IMMORTAL_PATH], [])
    st = B.init(2, 2)
    hi = H.ctx(max_hp=1000.0, hp=800.0)
    lo = H.ctx(max_hp=1000.0, hp=400.0)
    amp = B.dealt_amp(st, own, hi, u)
    assert amp.shape == (2, 2)
    assert float(amp[0, 1]) == pytest.approx(0.04) and float(amp[1, 0]) == 0.0
    assert float(B.dealt_amp(st, own, lo, u)[0, 1]) == 0.0
    assert float(B.stats(st, own, lo).incoming_heal[0]) == pytest.approx(0.12)
    assert float(B.stats(st, own, hi).incoming_heal[0]) == 0.0


def test_summoner_haste():
    st = B.init(2, 2)
    s = B.stats(st, H.own([B.IONIAN], [B.CRIMSON]), H.ctx())
    assert float(s.summoner_haste[0]) == pytest.approx(10.0)
    assert float(s.summoner_haste[1]) == pytest.approx(20.0)
    assert float(B.stats(st, H.own([B.BERSERKERS], []), H.ctx()).summoner_haste[0]) == 0.0


def test_crimson_lucidity_noxian_haste():
    u = world()
    own = H.own([B.CRIMSON], [])
    rep = _report(u, 0, 1, D.MAGIC, flags=D.TAG_ACTIVE_SPELL)
    for ranged, ms in ((False, 0.10), (True, 0.08)):
        st, _ = B.on_damage(B.init(2, 2), own, H.ctx(now=1.0, ranged=ranged), u, rep)
        assert float(B.stats(st, own, H.ctx(now=4.9, ranged=ranged)).percent_move_speed[0]) == pytest.approx(ms)
        assert float(B.stats(st, own, H.ctx(now=5.1, ranged=ranged)).percent_move_speed[0]) == 0.0
    st, _ = B.on_damage(B.init(2, 2), own, H.ctx(now=1.0), u, _report(u, 0, 1, D.PHYSICAL))
    assert float(B.stats(st, own, H.ctx(now=1.5)).percent_move_speed[0]) == 0.0


def test_swiftmarch_adaptive_force():
    st = B.init(2, 2)
    own = H.own([B.SWIFTMARCH], [])
    s = B.stats(st, own, H.ctx(move_speed=400.0, bonus_ad=10.0))
    assert float(s.attack_damage[0]) == pytest.approx(12.0) and float(s.ability_power[0]) == 0.0
    s = B.stats(st, own, H.ctx(move_speed=400.0, ap=100.0))
    assert float(s.ability_power[0]) == pytest.approx(20.0) and float(s.attack_damage[0]) == 0.0
    assert float(s.ability_power[1]) == 0.0


def test_coverage_lists_all_assigned():
    assigned = {3006, 3008, 3009, 3020, 3047, 3111, 3158, 3168, 3170, 3171, 3172, 3173, 3174, 3175}
    assert set(B.COVERAGE) == assigned
    assert not assigned & (E.STATS_ONLY | set(E.DEFERRED))


def test_jit_dispatch():
    u = world()
    own = H.own([B.ARMORED], [B.IMMORTAL_PATH])
    state = E.init(2, 2)
    c = H.ctx(level=13, base_hp=600.0, max_hp=1600.0)
    state, eff = jax.jit(E.on_damage)(state, own, c, u, _report(u, 1, 0, D.PHYSICAL))
    assert float(jnp.max(eff.shields.amount[0])) == pytest.approx(220.0, rel=1e-5)
    hd = jax.jit(E.holder_defense)(state, own, c)
    assert float(hd.basic_attack_mult[0]) == pytest.approx(0.9)
    amp = jax.jit(E.dealt_amp)(state, own, c, u)
    assert float(amp[1, 0]) == pytest.approx(0.04)
    jax.jit(E.on_takedown)(state, own, c, u, H.kills(2, champion_kill=(1, 0)))
    jax.jit(E.dynamic_stats)(state, own, c)
