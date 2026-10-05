"""Rune catalog, page legality, level formulas, shards and modifier ordering
(RUNES.md §1–2, §13; DAMAGE_AND_STATS.md §3–5, §8–11, §19)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.modern.core import damage as D
from lanerl_jax.modern.runes import catalog as RD
from lanerl_jax.modern.core import stat_pipeline as SP
from lanerl_jax.modern.core import stats as S
from lanerl_jax.modern.items.catalog import ItemStats, combine_stats
from lanerl_jax.modern.items.loadout import stat_shard_stats


# ---- catalog ----------------------------------------------------------------

def test_catalog_pins_the_2619_rune_set():
    cat = RD.rune_catalog()
    assert len(cat.runes) == 62 and len(cat.shards) == 7
    assert cat[8230].name == "Stormraider's Surge"          # DDragon key PhaseRush (D-10)
    assert cat[9105].name == "Legend: Haste"                  # hash-named slot entry {3ecd47e5}
    assert cat[8992].name == "Deathfire Touch" and cat[8992].slot_row == 0
    assert cat.style_row(RD.PRECISION, 0) == (8005, 8008, 8021, 8010)
    assert cat.style_row(RD.RESOLVE, 2) == (8429, 8444, 8473)
    assert cat.shard_slots == ((5008, 5005, 5007), (5008, 5010, 5001), (5011, 5013, 5001))
    for retired in (8124, 8136, 8120, 8138, 8134, 8359, 8339, 8472, 8430, 8435):
        assert retired in cat.not_selectable and retired not in cat
    assert RD.ea(8010, "MaxStacks") == 12 and RD.ea(8473, "Cooldown") == 55


# ---- legality and substitution (F-35) --------------------------------------

def test_page_legality_and_substitution():
    page = RD.GAREN_DEFAULT_PAGE
    RD.validate_page(page)
    bad = [
        RD.RunePage(RD.PRECISION, 8010, (9111, 9105, 8299), RD.SORCERY, (8224, 8226), page.shards),  # same row
        RD.RunePage(RD.PRECISION, 8010, (9111, 9105, 8299), RD.SORCERY, (8230, 8234), page.shards),  # keystone
        RD.RunePage(RD.PRECISION, 8010, (9111, 9105, 8299), RD.PRECISION, (9101, 8014), page.shards),
        RD.RunePage(RD.PRECISION, 8112, (9111, 9105, 8299), RD.SORCERY, (8224, 8234), page.shards),  # wrong tree
        RD.RunePage(RD.PRECISION, 8010, (9111, 9105, 8299), RD.SORCERY, (8224, 8234), (5001, 5008, 5011)),
        RD.RunePage(RD.PRECISION, 8010, (9111, 8014, 8299), RD.SORCERY, (8224, 8234), page.shards),  # row
    ]
    for p in bad:
        with pytest.raises(ValueError):
            RD.validate_page(p)
    # Shards: Adaptive twice (offense + flex) and scaling HP twice (flex + defense) are legal.
    RD.validate_page(RD.RunePage(RD.PRECISION, 8010, (9111, 9105, 8299), RD.SORCERY, (8224, 8234),
                                 (5005, 5001, 5001)))
    garen = RD.CHAMPION_TRAITS["Garen"]
    resolve = RD.RunePage(RD.RESOLVE, 8439, (8446, 8444, 8451), RD.PRECISION, (8009, 9105), page.shards)
    got = RD.prepare_page(resolve, garen)
    assert got.keystone == RD.GRASP and got.secondary[0] == RD.TRIUMPH
    sorc = RD.RunePage(RD.SORCERY, 8230, (8226, 8234, 8237), RD.INSPIRATION, (8306, 8347), page.shards)
    got = RD.prepare_page(sorc, RD.ChampionTraits(has_immobilize=True, resource="energy", flash_equipped=False))
    assert got.primary[0] == RD.AXIOM and got.secondary[0] == RD.CASH_BACK
    assert RD.prepare_page(sorc, RD.CHAMPION_TRAITS["Jax"]) == sorc
    counts = RD.page_counts([RD.prepare_page(page, garen), None])
    assert counts.shape == (2, len(RD.rune_catalog().ids)) and counts[0].sum() == 9 and counts[1].sum() == 0
    assert counts[0, RD.rune_catalog().row(5008)] == 2


# ---- level primitives -------------------------------------------------------

def test_level_scaling_primitives():
    levels = jnp.asarray([1., 6., 9., 13., 18.])
    np.testing.assert_allclose(RD.lin(1.8, 4.0, levels), [1.8, 2.4471, 2.8353, 3.3529, 4.0], atol=1e-4)  # F-4
    np.testing.assert_allclose(RD.lin_growth(15., 160., levels), [15., 48.691, 72.488, 108.397, 160.],
                               atol=1e-3)                                                                 # F-11
    bp = RD.breakpoints(1.0, 0.25, ((6, 1.0), (11, 2.0)), jnp.asarray([1., 5., 6., 10., 11., 18., 20.]))
    np.testing.assert_allclose(bp, [1, 2, 3, 7, 9, 23, 27])                                               # F-33
    pom = RD.rune_catalog()[8009].calculations["RegenAmount"]["mFormulaParts"][0]["values"]
    np.testing.assert_allclose(RD.level_table(pom, jnp.asarray([1, 18, 20])), [6., 44., 56.], atol=1e-5)
    assert float(RD.lin(25., 15., 20., scale_past_18=False)) == pytest.approx(15.)   # First Strike cd clamp
    assert float(RD.lin(10., 180., 20.)) == pytest.approx(200.)                     # U-01 extrapolation


def test_stat_shards_from_client_values():
    a = stat_shard_stats((5008, 5008, 5001), level=1)                                # F-1
    assert a.attack_damage == pytest.approx(10.8) and a.health == pytest.approx(10.)
    assert stat_shard_stats((5008, 5008, 5001), level=18).health == pytest.approx(180.)  # F-2
    f = stat_shard_stats((5005, 5010, 5013))                                         # F-3
    assert f.attack_speed == pytest.approx(0.10) and f.percent_move_speed == pytest.approx(0.025)
    assert f.tenacity == pytest.approx(0.15) and f.slow_resist == pytest.approx(0.15)
    u = stat_shard_stats(("adaptive", "adaptive", "health_flat"), adaptive_to_ad=None)
    assert u.adaptive_force == pytest.approx(18.) and u.attack_damage == 0


# ---- modifier ordering ------------------------------------------------------

def test_resist_order_keeps_negative_resist():
    # DAMAGE F2/F3: reduction can make resist negative; penetration then does nothing.
    r = S.armor_after_modifiers(18, flat_reduction=30, percent_reduction=.3, percent_penetration=.45, lethality=10)
    assert r == pytest.approx(-12)
    assert 500 * S.mitigation_multiplier(r) == pytest.approx(553.5714, rel=1e-6)
    assert S.armor_after_modifiers(10, flat_reduction=25, percent_reduction=.5, flat_penetration=40) == -15


def test_dealt_amps_add_received_multiply():
    # F-25 / U-19 default: PTA 8% + Coup de Grace 8% -> x1.16 (not 1.1664).
    assert S.apply_damage_modifiers(100., attacker_amp=0.16) == pytest.approx(116.)
    # DAMAGE F8: dealt +10% +8% summed; received 30% and 20% DR multiply.
    got = S.apply_damage_modifiers(500., attacker_amp=0.18, target_reduction=1 - 0.7 * 0.8)
    assert got * 100 / 150 == pytest.approx(220.2667, rel=1e-5)
    # Exhaust joins the dealt sum; true damage ignores Exhaust and DR, keeps amps (F6/F7).
    assert S.apply_damage_modifiers(100., attacker_amp=0.10, attacker_reduction=0.35) == pytest.approx(75.)
    assert S.apply_damage_modifiers(100., attacker_amp=0.10, attacker_reduction=0.35, target_reduction=0.3,
                                    is_true=True) == pytest.approx(110.)


def test_pipeline_sums_amps_in_one_dmg40_slot():
    n = 2
    off = D.default_offense(n, unit_class=D.CLASS_CHAMPION)
    dfn = D.default_defense(n, unit_class=D.CLASS_CHAMPION)
    p = D.packets(jnp.ones(2, bool), 0, 1, 100.0, D.TRUE, amp=jnp.asarray([0.08, 0.16]))
    np.testing.assert_allclose(D.premitigation_to_final(p, off, dfn), [108., 116.], rtol=1e-6)
    # DMG.70 per-packet block (Bone Plating) applies to true damage and floors at 0.
    p = D.packets(jnp.ones(2, bool), 0, 1, jnp.asarray([50., 20.]), D.TRUE, block=30.0)
    np.testing.assert_allclose(D.premitigation_to_final(p, off, dfn), [20., 0.])


def test_adaptive_choice_is_dynamic():
    ad, ap = S.resolve_adaptive(10., bonus_ad=40., ability_power=0.)
    assert (ad, ap) == (pytest.approx(6.), 0)
    ad, ap = S.resolve_adaptive(10., bonus_ad=40., ability_power=80.)
    assert (ad, ap) == (0, pytest.approx(10.))
    # Tie (including 0/0) goes to the champion's adaptive type.
    assert S.resolve_adaptive(10., 0., 0., adaptive_physical=False)[1] == pytest.approx(10.)
    assert S.resolve_adaptive(10., 0., 0., adaptive_physical=True)[0] == pytest.approx(6.)


# ---- stat composition (DAMAGE_AND_STATS §19.2–19.3) ------------------------

def garen():
    return SP.champion_base(["Garen"])


def test_compose_garen_growth_attack_timing():
    base = garen()
    hp = SP.compose(base, jnp.asarray([1., 2., 18., 20.]), ItemStats()).max_hp
    np.testing.assert_allclose(hp, [690., 760.56, 2356., 2617.17], atol=1e-2)            # F12
    s = SP.compose(base, jnp.asarray([6.]), ItemStats(attack_speed=0.25))                # F13
    assert float(s.attack_speed[0]) == pytest.approx(0.871359, rel=1e-5)
    assert float(s.attack_period[0]) == pytest.approx(1.147632, rel=1e-5)
    assert float(s.attack_windup[0]) == pytest.approx(0.247287, rel=1e-4)
    s1 = SP.compose(base, jnp.asarray([1.]), ItemStats())                                # F14
    assert float(s1.attack_period[0]) == pytest.approx(1.6) and float(s1.attack_windup[0]) == pytest.approx(0.288)
    capped = SP.compose(base, jnp.asarray([1.]), ItemStats(attack_speed=10.))            # F15
    assert float(capped.attack_speed[0]) == pytest.approx(3.003003, rel=1e-6)
    lifted = SP.compose(base, jnp.asarray([1.]), ItemStats(attack_speed=10., attack_speed_cap_lift=1.))
    assert float(lifted.attack_speed[0]) > 3.1


def test_compose_percent_stages_and_adaptive():
    base = garen()
    # Conditioning (F-18 shape): (base + bonus + 8) x 1.03 on total armor.
    s = SP.compose(base, jnp.asarray([1.]), ItemStats(armor=28., percent_armor=0.03))
    assert float(s.base_armor[0] + s.bonus_armor[0]) == pytest.approx((38 + 28) * 1.03, rel=1e-6)
    # Overgrowth x1.035 on all max HP; adaptive force split at STAT.50 from pre-adaptive stats.
    s = SP.compose(base, jnp.asarray([1.]), ItemStats(health=500., percent_health=0.035, adaptive_force=9.,
                                                      attack_damage=10.))
    assert float(s.max_hp[0]) == pytest.approx(1190 * 1.035, rel=1e-6)
    assert float(s.bonus_ad[0]) == pytest.approx(15.4)
    s = SP.compose(base, jnp.asarray([1.]), ItemStats(adaptive_force=9., ability_power=20.))
    assert float(s.ap[0]) == pytest.approx(29.) and float(s.bonus_ad[0]) == 0


def test_move_speed_haste_tenacity():
    ms = lambda **kw: float(SP.move_speed(**kw))
    # F19 inputs (340 + 45 boots, +35% additive): raw 519.75 -> 489.875 (the doc's 529.375 used 37.5%).
    assert ms(base_ms=340., flat=45., additive_pct=0.35) == pytest.approx(489.875)
    assert ms(base_ms=385., slow=0.4, slow_resist=0.15) == pytest.approx(254.1)              # F20
    assert ms(base_ms=340., slow=0.99) == pytest.approx(111.7)                               # F21
    assert ms(base_ms=450.) == pytest.approx(443.0)                                          # F22
    # Celerity: other bonus MS is 7% more effective (RUNES §5.6, U-11 clean default).
    assert ms(base_ms=300., flat=25., bonus_ms_amp=0.07) == pytest.approx(300 + 25 * 1.07)
    np.testing.assert_allclose(SP.cooldown(10., jnp.asarray([20., 500., 600.])), [8.3333, 1.6667, 1.6667],
                               atol=1e-4)                                                    # F16
    t = 1 - (1 - .30) * (1 - .20) * (1 - .15)
    assert float(SP.cc_duration(1.5, SP.tenacity_total(t))) == pytest.approx(0.714, abs=1e-3)  # F17
    assert float(SP.cc_duration(0.4, 0.9)) == pytest.approx(0.3)                             # F18
    assert float(SP.cc_duration(0.25, 0.9)) == pytest.approx(0.25)
    assert float(SP.cc_duration(1.0, -0.2)) == pytest.approx(1.2)
    hp, mx = SP.sync_max_health(500., 1000., 1030., heal_on_gain=0.)
    assert (float(hp), float(mx)) == (500., 1030.)


def test_compose_is_jittable():
    base = garen()
    f = jax.jit(lambda lv, b: SP.compose(base, lv, b))
    out = f(jnp.asarray([9.]), combine_stats(ItemStats(attack_damage=jnp.asarray([10.])), ItemStats(armor=5.)))
    assert out.max_hp.shape == (1,)
