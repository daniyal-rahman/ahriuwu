"""Catalog, shop, damage pipeline and Hydra-line tests (docs/modern/ITEMS.md §17)."""
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.sim import modern_damage as D
from lanerl_jax.sim import modern_inventory as I
from lanerl_jax.sim import modern_item_effects as E
from lanerl_jax.sim.modern_item_data import DATA_PATH, catalog, lerp_level, level_bp
from lanerl_jax.sim.tests import item_harness as H


def one(ids):
    inv = I.inventory_from_ids([ids])
    return I.Inventory(inv.item[0], inv.stack[0])


def ids_of(inv):
    return [catalog().ids[r] if r >= 0 else -1 for r in np.asarray(inv.item).tolist()]


def test_catalog_is_client_pinned_sr_pool():
    payload = json.loads(DATA_PATH.read_text())
    assert payload["client_build"] == "16.19.8230722" and payload["mode"] == "CLASSIC"
    cat = catalog()
    in_store = [s for s in cat.specs if s.in_store]
    assert len(in_store) == 210
    assert {3040, 3042, 3121, 2530} <= set(cat.ids)          # transforms present
    assert not (set(cat.ids) & {322065, 663039, 1105})       # DDragon mode mirrors absent
    assert cat[3077].stats.attack_damage == 25                # 26.16 Tiamat AD
    assert cat[1086].stats.attack_speed == pytest.approx(0.15)   # ITEMS.md D1: client 15%
    assert cat[3107].total == 2300                            # D2
    assert cat[3111].stats.tenacity == pytest.approx(0.30)
    assert cat[3134].stats.lethality == 10


def test_level_scaling_helpers():
    assert level_bp(400, 30, 9, 13) == pytest.approx(550)                 # Shieldbow L13 (F15)
    assert lerp_level(110, 280, 10) == pytest.approx(200)                 # Hexdrinker L10 (F15)
    assert lerp_level(10, 180, 20) == pytest.approx(200)                  # README X-1 extrapolation


def test_sell_values_and_recipe_purchase():
    for iid, value in ((3078, 2333), (1037, 613), (1054, 180), (3077, 840), (6631, 2310)):
        assert float(I.sell(one([iid]), 0.0, 0, can_shop=True).gold) == value      # F12
    buy = jax.jit(lambda inv, g, r: I.buy(inv, g, r, can_shop=True, level=1, is_ranged=False))
    held = one([3077, 1036])
    r = buy(held, 1500.0, catalog().row(3074))                                     # F13
    assert not bool(r.ok) and int(r.code) == I.ERR_GOLD
    r = buy(held, 1750.0, catalog().row(3074))
    assert bool(r.ok) and float(r.spent) == 1750 and ids_of(r.inv)[:2] == [3074, -1]


@pytest.mark.parametrize("held,want,ok", [
    ([3077], 6631, True),     # recipe consumes the Hydra member
    ([3748], 3077, False),    # Hydra max 1
    ([1055], 1054, False),    # one Doran's/starter
    ([1054], 1083, True),     # Cull is not a starter
    ([2003], 2031, False),    # Potion group
    ([2003], 2003, True),     # stacks
])
def test_item_groups(held, want, ok):
    r = I.buy(one(held), 5000.0, catalog().row(want), can_shop=True, level=10, is_ranged=False)
    assert bool(r.ok) == ok


def test_purchase_gates():
    row = catalog().row
    assert int(I.buy(one([]), 5000., row(2138), can_shop=True, level=8, is_ranged=False).code) == I.ERR_LEVEL
    r = I.buy(one([]), 5000., row(2138), can_shop=True, level=9, is_ranged=False, now=10.)
    assert bool(r.ok)
    r2 = I.buy(r.inv, 5000., row(2140), can_shop=True, level=9, is_ranged=False, now=12.,
               group_cd_until=r.group_cd_until)
    assert int(r2.code) == I.ERR_COOLDOWN                                          # F17
    assert int(I.buy(one([]), 5000., row(3085), can_shop=True, level=9, is_ranged=False).code) \
        == I.ERR_NOT_PURCHASABLE                                                   # Runaan's ranged only
    assert int(I.buy(one([]), 5000., row(3171), can_shop=True, level=9, is_ranged=False).code) \
        == I.ERR_NOT_PURCHASABLE                                                   # T3 boots need mid quest
    assert int(I.buy(one([]), 5000., row(1036), can_shop=False, level=9, is_ranged=False).code) \
        == I.ERR_NOT_IN_SHOP
    assert bool(I.in_shop_area(jnp.float32(412.9), jnp.float32(1400.), 0, False))
    assert not bool(I.in_shop_area(jnp.float32(412.9), jnp.float32(1430.), 0, False))
    full = one([1036] * 6)
    assert int(I.buy(full, 5000., row(1028), can_shop=True, level=1, is_ranged=False).code) == I.ERR_NO_SLOT
    assert bool(I.buy(full, 5000., row(1038), can_shop=True, level=1, is_ranged=True).ok) is False
    with pytest.raises(ValueError, match="group"):
        I.validate_item_loadout([3077, 6631])
    I.validate_item_loadout([2003, 2003, 1054])


def test_inventory_stat_stacking_rules():
    s = I.inventory_stats(I.inventory_from_ids([[3111, 3053], [1036, 1036, 3134]]))
    assert float(s.tenacity[0]) == pytest.approx(1 - 0.7 * 0.8)       # multiplicative
    assert float(s.attack_damage[1]) == pytest.approx(40)
    assert float(s.lethality[1]) == pytest.approx(10)


def test_resist_order_keeps_negative_resist():
    p = D.packets(jnp.ones(5, bool), 0, jnp.arange(1, 6), 500.0, D.PHYSICAL)
    n = 6
    dfn = D.default_defense(n)._replace(
        armor=jnp.asarray([0, 100, 100, 18, 10, 0.], jnp.float32),
        flat_armor_reduction=jnp.asarray([0, 0, 20, 30, 25, 0.], jnp.float32),
        percent_armor_reduction=jnp.asarray([0, 0, .3, .3, .5, 0.], jnp.float32))
    off = D.default_offense(n)._replace(lethality=jnp.full((n,), 10.), percent_armor_pen=jnp.full((n,), .3))
    final = D.premitigation_to_final(p, off, dfn)
    # 100 AR, 30% pen, 10 lethality -> 60; 100 AR -20 flat 30% red 30% pen -> 29.2; 18 AR -> -12; 10 -> -15.
    np.testing.assert_allclose(final, [500 * 100 / 160, 500 * 100 / 129.2, 500 * (2 - 100 / 112),
                                       500 * (2 - 100 / 115), 500.0], rtol=1e-5)


def test_dealt_amps_add_received_multiply_true_damage_rules():
    n = 2
    p = D.packets(jnp.ones(3, bool), 0, 1, 100.0, jnp.asarray([D.PHYSICAL, D.TRUE, D.TRUE]),
                  amp=jnp.asarray([.1, .1, .1]))
    dfn = D.default_defense(n)._replace(received_mult=jnp.full((n,), .8), received_amp=jnp.full((n,), .1))
    off = D.default_offense(n)._replace(dealt_reduction=jnp.full((n,), .35))
    final = D.premitigation_to_final(p, off, dfn)
    # physical: (1 + .1 - .35) * .8 * 1.1 = 66; true ignores Exhaust and DR but keeps both amps.
    np.testing.assert_allclose(final, [100 * .75 * .8 * 1.1, 100 * 1.1 * 1.1, 121.0], rtol=1e-5)


def test_minion_class_ratio_plating_and_warden():
    n = 3
    cls = jnp.asarray([D.CLASS_MINION, D.CLASS_CHAMPION, D.CLASS_CHAMPION], jnp.int32)
    p = D.packets(jnp.ones(2, bool), jnp.asarray([0, 1]), 2, jnp.asarray([100.0, 50.0]), D.PHYSICAL, D.BASIC_ATTACK)
    dfn = D.default_defense(n)._replace(unit_class=cls, basic_attack_mult=jnp.full((n,), .9),
                                         champion_attack_block=jnp.full((n,), 15.))
    off = D.default_offense(n)._replace(unit_class=cls)
    final = D.premitigation_to_final(p, off, dfn)
    # Minion: 100 * .55 * .9 = 49.5 (no Warden vs minions); champion: 45 - min(15, 9) = 36.
    np.testing.assert_allclose(final, [49.5, 36.0], rtol=1e-5)


def test_shields_typed_order_and_lifeline_absorbs_trigger():
    n = 2
    sh = D.init_shields(n)
    sh = D.grant_shield(sh, 1, 50.0, D.SHIELD_MAGIC, 0.0, 5.0)
    sh = D.grant_shield(sh, 1, 30.0, D.SHIELD_ALL, 0.0, 2.0)
    p = D.packets(jnp.ones(1, bool), 0, 1, 40.0, D.PHYSICAL)
    hp = jnp.asarray([1000., 1000.])
    r = D.resolve(p, D.default_offense(n), D.default_defense(n), hp, hp, sh, 0.0)
    assert float(r.absorbed[0]) == 30 and float(r.hp[1]) == 990            # magic shield ignored
    # Sterak's F7: HP 900/2600, 200 post-mitigation -> Lifeline 600 shield absorbs it.
    dfn = D.default_defense(n)._replace(lifeline_ready=jnp.asarray([False, True]),
                                         lifeline_shield=jnp.full((n,), 600.), lifeline_duration=jnp.full((n,), 4.5),
                                         lifeline_decay_hold=jnp.full((n,), .75))
    p = D.packets(jnp.ones(1, bool), 0, 1, 200.0, D.PHYSICAL)
    r = D.resolve(p, D.default_offense(n), dfn, jnp.asarray([1000., 900.]), jnp.asarray([1000., 2600.]),
                  D.init_shields(n), 0.0)
    assert bool(r.lifeline_fired[1]) and float(r.hp[1]) == 900
    assert float(D.total_shield(r.shields, 0.0)[1]) == pytest.approx(400)
    # Linear decay after the 0.75 s hold: at 2.625 s half the initial cap remains.
    assert float(D.total_shield(r.shields, 2.625)[1]) == pytest.approx(300)


def test_deaths_dance_store_spell_shield_execute():
    n = 2
    dfn = D.default_defense(n)._replace(store_fraction=jnp.asarray([0., .3]),
                                         spell_shield=jnp.asarray([False, True]),
                                         unit_class=jnp.full((n,), D.CLASS_CHAMPION, jnp.int32))
    off = D.default_offense(n)._replace(unit_class=jnp.full((n,), D.CLASS_CHAMPION, jnp.int32))
    p = D.packets(jnp.ones(2, bool), 0, 1, 100.0, D.PHYSICAL, jnp.asarray([D.BASIC_ATTACK, D.TAG_ACTIVE_SPELL]))
    hp = jnp.full((n,), 1000.)
    r = D.resolve(p, off, dfn, hp, hp, D.init_shields(n), 0.0)
    assert float(r.hp[1]) == 930 and float(r.dd_pool_add[1]) == 30 and bool(r.spell_shield_popped[1])
    sh = D.grant_shield(D.init_shields(n), 1, 500.0, D.SHIELD_ALL, 0.0, 5.0)
    p = D.packets(jnp.ones(1, bool), 0, 1, 0.0, D.TRUE, D.PROP_EXECUTE)
    r = D.resolve(p, off, D.default_defense(n), jnp.asarray([1000., 40.]), hp, sh, 0.0)
    assert bool(r.killed[0]) and float(D.total_shield(r.shields, 0.0)[1]) == 0


def test_vamp_ratios():
    cls = jnp.asarray([D.CLASS_CHAMPION, D.CLASS_MINION, D.CLASS_STRUCTURE], jnp.int32)
    p = D.packets(jnp.ones(4, bool), 0, jnp.asarray([1, 1, 2, 1]), 100.0, D.PHYSICAL,
                  jnp.asarray([D.BASIC_ATTACK, D.TAG_AOE, D.BASIC_ATTACK, D.PROP_NO_OMNIVAMP]))
    res = D.resolve(p, D.default_offense(3), D.default_defense(3)._replace(unit_class=cls),
                    jnp.full((3,), 1000.), jnp.full((3,), 1000.), D.init_shields(3), 0.0)
    heal = D.vamp_heal(p, res, D.Vamp(jnp.asarray([.1, 0, 0]), jnp.asarray([.1, 0, 0])), cls)
    # LS 10 (attack on minion) + OV 10 + OV 3.33 (AoE on minion); structures and NO_OMNIVAMP give nothing.
    assert float(heal[0]) == pytest.approx(10 + 10 + 3.33)
    assert float(D.heal_amount(100., source_power=.1, incoming=.25, grievous=True)) == pytest.approx(82.5)


def hydra_world(rows):
    return H.units(rows)


def test_cleave_radius_cap_and_structures():
    u = hydra_world(H.champions(x1=300.) + [dict(x=500, y=0, team=1), dict(x=640, y=0, team=1),
                                             dict(x=660, y=0, team=1, radius=0.)])
    ctx = H.ctx(base_ad=100., bonus_ad=50.)
    st, eff = E.on_hit(E.init(2, 5), H.own([3077], []), ctx, u, H.attack())
    assert H.packet_targets(eff.packets, item=3077) == [2, 3]                     # F1
    assert H.packet_total(eff.packets, dst=2) == pytest.approx(60)
    many = H.champions(x1=300.) + [dict(x=300 + 20 * k, y=50, team=1) for k in range(12)]
    st, eff = E.on_hit(E.init(2, 14), H.own([3077], []), ctx, H.units(many), H.attack())
    assert int(eff.packets.valid.sum()) == 10                                       # F2
    tower = H.champions(x1=3000.) + [dict(x=300, y=0, team=1, cls=D.CLASS_STRUCTURE), dict(x=400, y=0, team=1)]
    st, eff = E.on_hit(E.init(2, 4), H.own([3077], []), ctx, H.units(tower), H.attack(target=(2, 0)))
    assert int(eff.packets.valid.sum()) == 0
    ranged = ctx._replace(is_ranged=jnp.asarray([True, False]))
    st, eff = E.on_hit(E.init(2, 5), H.own([3074], []), ranged, u, H.attack())
    assert H.packet_total(eff.packets, dst=2) == pytest.approx(30)
    assert bool(np.all((np.asarray(eff.packets.flags)[np.asarray(eff.packets.valid)] & D.PROP_LIFESTEAL) != 0))


def test_crescent_offset_circle():
    u = hydra_world(H.champions(x1=2000.) + [dict(x=520, y=0, team=1, radius=0.), dict(x=-340, y=0, team=1, radius=0.),
                                              dict(x=100, y=440, team=1, radius=0.), dict(x=-360, y=0, team=1, radius=0.)])
    ctx = H.ctx(base_ad=100., bonus_ad=50., windup=0.)
    st, eff, out = E.active(E.init(2, 6), H.own([3077], []), ctx, u, jnp.asarray([3077, 0], jnp.int32))
    assert H.packet_targets(eff.packets, item=3077) == [2, 3, 4]                  # F3
    assert H.packet_total(eff.packets, dst=2) == pytest.approx(112.5)
    assert float(st.hydra.cd_until[0]) == pytest.approx(10.0)
    st, eff, out = E.active(st, H.own([3077], []), ctx._replace(now=jnp.float32(5.)), u,
                            jnp.asarray([3077, 0], jnp.int32))
    assert not bool(out.used[0])


def test_stridebreaker_and_profane_cooldowns_and_cast_time():
    rows = [dict(x=0, y=0, team=0, cls=0, radius=65.), dict(x=200, y=0, team=1, cls=0, radius=65.),
            dict(x=150, y=100, team=1, cls=0, radius=65.), dict(x=300, y=0, team=1),
            dict(x=100, y=-200, team=1), dict(x=50, y=50, team=1)]
    u = H.units(rows)
    own = H.own([6631], [])
    ctx = H.ctx(base_ad=180., windup=0.4)
    st, eff, out = E.active(E.init(2, 6), own, ctx, u, jnp.asarray([6631, 0], jnp.int32))
    assert float(out.cast_time[0]) == pytest.approx(0.25) and bool(out.can_move[0])
    assert int(eff.packets.valid.sum()) == 0
    late = ctx._replace(now=jnp.float32(0.25))
    st, eff, out = E.active(st, own, late, u, jnp.asarray([0, 0], jnp.int32))
    assert H.packet_targets(eff.packets, item=6631) == [1, 2, 3, 4, 5]             # F5
    assert H.packet_total(eff.packets, dst=1) == pytest.approx(144)
    np.testing.assert_allclose(eff.slow[1:], .35, rtol=1e-6)
    assert float(st.hydra.cd_until[0]) == pytest.approx(15.0)                       # from cast start
    ms = lambda t: float(E.dynamic_stats(st, own, late._replace(now=jnp.float32(t))).percent_move_speed[0])
    assert ms(0.25) == pytest.approx(0.70, rel=1e-5) and ms(1.75) == pytest.approx(0.35, rel=1e-5)
    assert ms(3.25) == 0.0


def test_titanic_on_hit_cone_and_empowered_reset():
    u = H.units(H.champions(x1=200.) + [dict(x=400, y=0, team=1), dict(x=200, y=600, team=1)])
    own = H.own([3748], [])
    ctx = H.ctx(base_hp=1000., max_hp=2500.)
    st, eff = E.on_hit(E.init(2, 4), own, ctx, u, H.attack())
    assert H.packet_total(eff.packets, dst=1) == pytest.approx(25)                 # F6
    assert H.packet_total(eff.packets, dst=2) == pytest.approx(75)
    assert H.packet_total(eff.packets, dst=3) == 0
    st, eff, out = E.active(st, own, ctx, u, jnp.asarray([3748, 0], jnp.int32))
    assert bool(out.attack_reset[0]) and bool(eff.attack_reset[0])
    st, eff = E.on_hit(st, own, ctx._replace(now=jnp.float32(1.)), u, H.attack())
    assert H.packet_total(eff.packets, dst=1) == pytest.approx(100)
    assert H.packet_total(eff.packets, dst=2) == pytest.approx(225)
    assert float(st.hydra.cd_until[0]) == pytest.approx(11.0)


def test_hydra_hooks_jit_and_isolation():
    u = H.units(H.champions(x1=300.) + [dict(x=500, y=0, team=1)])
    own = H.own([], [3077])
    st, eff = jax.jit(E.on_hit)(E.init(2, 3), own, H.ctx(), u, H.attack())
    assert int(eff.packets.valid.sum()) == 0


def test_parallel_resolution_for_non_champions_matches_emission_order():
    n = 4
    cls = jnp.asarray([D.CLASS_CHAMPION, D.CLASS_CHAMPION, D.CLASS_MINION, D.CLASS_MINION], jnp.int32)
    dfn = D.default_defense(n)._replace(unit_class=cls)
    off = D.default_offense(n)._replace(unit_class=cls)
    # Interleaved packets on two minions and one champion; minion 3 starts dead.
    p = D.packets(jnp.ones(5, bool), 0, jnp.asarray([2, 1, 2, 3, 2]), jnp.asarray([60., 50., 60., 10., 5.]), D.TRUE)
    hp = jnp.asarray([500., 500., 100., 0.])
    r = D.resolve(p, off, dfn, hp, hp, D.init_shields(n), 0.0)
    np.testing.assert_allclose(r.health_loss, [60., 50., 40., 0., 0.])
    np.testing.assert_array_equal(r.killed, [False, False, True, False, False])
    np.testing.assert_allclose(r.hp, [500., 450., -25., 0.])
    assert int(r.overflow) == 0
