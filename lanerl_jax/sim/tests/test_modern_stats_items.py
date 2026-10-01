import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.sim import modern_items as items
from lanerl_jax.sim import modern_stats as stats


def test_stat_composition_and_health_change():
    # Rune armor is flat bonus and percent base bonus does not multiply it.
    assert stats.stat_total(100, flat_bonus=9, percent_base_bonus=.2,
                            percent_bonus=.2) == pytest.approx(154.8)
    hp, maximum = stats.change_max_health(500, 1000, 1100)
    assert (hp, maximum) == (600, 1100)
    hp, maximum = stats.change_max_health(500, 1000, 900)
    assert (hp, maximum) == (500, 900)


def test_resistance_order_and_negative_armor():
    armor = stats.armor_after_modifiers(100, flat_reduction=20,
                                        percent_reduction=.3,
                                        percent_penetration=.25,
                                        lethality=10)
    assert armor == pytest.approx(32)
    assert stats.armor_after_modifiers(10, flat_reduction=25,
                                       percent_reduction=.5,
                                       flat_penetration=40) == 0
    assert stats.mitigation_multiplier(-50) == pytest.approx(1.3333333333)
    assert stats.post_mitigation_damage(100, 0) == 100
    with pytest.raises(ValueError, match="damage type"):
        stats.validate_damage_type(99)


def test_jax_jit_vmap_shared_damage_math():
    got = jax.jit(lambda r: stats.post_mitigation_damage(100.0, r, jnp))(
        jnp.array([-50., 0., 100.]))
    np.testing.assert_allclose(got, [133.33333, 100., 50.], rtol=1e-5)
    armor = jax.jit(jax.vmap(lambda a: stats.armor_after_modifiers(
        a, flat_reduction=10, percent_penetration=.2, lethality=5, xp=jnp)))(
            jnp.array([100., 50.]))
    np.testing.assert_allclose(armor, [67., 27.])


def test_adaptive_force_and_shards():
    ad, ap = stats.adaptive_force_total(9., converts_to_ad=True)
    assert ad == pytest.approx(5.4) and ap == 0
    shard = items.stat_shard_stats(level=18)
    assert shard.attack_damage == pytest.approx(10.8)
    assert shard.health == pytest.approx(65)
    scaling = items.stat_shard_stats(("adaptive", "health_scaling", "health_scaling"), level=20)
    assert scaling.health == pytest.approx(360)
    with pytest.raises(ValueError):
        items.stat_shard_stats(("armor", "adaptive", "health_flat"))
    shard = jax.jit(lambda level, adaptive: items.stat_shard_stats(
        ("adaptive", "health_scaling", "health_flat"), level=level,
        adaptive_to_ad=adaptive))
    compiled = shard(jnp.asarray(10.), jnp.asarray(False))
    assert compiled.ability_power == pytest.approx(9.0)
    assert compiled.health == pytest.approx(165.0)


def test_patch_pinned_items_and_strict_effect_gate():
    data = json.loads((items.DATA_DIR / "items.json").read_text())
    assert data["patch"] == "26.19" and data["game_version"] == "16.19.1"
    assert items.ITEMS[6631].name == "Stridebreaker"
    assert items.ITEMS[3077].stats.attack_damage == 25
    # Component and upgrade cannot both be equipped; exact SR item IDs matter.
    with pytest.raises(ValueError, match="Hydra"):
        items.validate_item_loadout([3077, 6631])
    with pytest.raises(NotImplementedError, match="effects"):
        items.item_loadout_stats([3071])  # Cleaver stacks its own armor shred.
    stats_out, unmodeled = items.item_loadout_stats([1036, 1036], strict_effects=True)
    assert stats_out.attack_damage == 20 and not unmodeled
    with pytest.raises(NotImplementedError, match="rune"):
        items.validate_rune_page([8010])


def test_supported_active_kernels_are_jittable():
    dmg, hit = jax.jit(lambda x: items.tiamat_crescent(
        100., x, jnp.zeros_like(x), 1., 0., xp=jnp))(
            jnp.array([100., 500., -200.]))
    np.testing.assert_allclose(dmg, [75., 0., 0.])
    np.testing.assert_array_equal(hit, [True, False, False])
    d, hit, slow, ms = jax.jit(lambda d, c: items.stridebreaker_active(
        100., d, champion_target=c, xp=jnp))(
            jnp.array([300., 451.]), jnp.array([True, True]))
    np.testing.assert_allclose(d, [80., 0.])
    np.testing.assert_array_equal(hit, [True, False])
    np.testing.assert_allclose(slow, [.35, 0.])
    assert ms == pytest.approx(.35)


def test_rune_catalog_is_patch_pinned():
    data = items.load_rune_data()
    assert data["patch"] == "26.19"
    assert data["trees"]
