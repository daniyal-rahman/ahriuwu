
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.modern.core import stats
from lanerl_jax.modern.items import loadout as items
from lanerl_jax.modern.items.catalog import catalog
from lanerl_jax.modern.items.inventory import inventory_from_ids, inventory_stats, validate_item_loadout


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
    # Negative armor from flat reduction survives % reduction and penetration (D1).
    assert stats.armor_after_modifiers(10, flat_reduction=25,
                                       percent_reduction=.5,
                                       flat_penetration=40) == pytest.approx(-15)
    assert stats.armor_after_modifiers(30, flat_penetration=40) == 0
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
    # Scaling health is 10 per level (180 at 18), extrapolated to level 20 (README X-1).
    scaling = items.stat_shard_stats(("adaptive", "health_scaling", "health_scaling"), level=20)
    assert scaling.health == pytest.approx(400)
    flex = items.stat_shard_stats(("attack_speed", "move_speed", "tenacity"))
    assert flex.percent_move_speed == pytest.approx(0.025)
    assert flex.tenacity == pytest.approx(0.15) and flex.slow_resist == pytest.approx(0.15)
    with pytest.raises(ValueError):
        items.stat_shard_stats(("armor", "adaptive", "health_flat"))
    shard = jax.jit(lambda level, adaptive: items.stat_shard_stats(
        ("adaptive", "health_scaling", "health_flat"), level=level,
        adaptive_to_ad=adaptive))
    compiled = shard(jnp.asarray(10.), jnp.asarray(False))
    assert compiled.ability_power == pytest.approx(9.0)
    assert compiled.health == pytest.approx(165.0)


def test_patch_pinned_items_and_rune_page_gate():
    assert catalog()[6631].name == "Stridebreaker"
    assert catalog()[3077].stats.attack_damage == 25
    with pytest.raises(ValueError, match="group"):          # component and upgrade cannot both be held
        validate_item_loadout([3077, 6631])
    s = inventory_stats(inventory_from_ids([[1036, 1036], [3071]], trinket=False))
    assert float(s.attack_damage[0]) == 20 and float(s.health[1]) == 400
    assert items.validate_rune_page([]) is None
    with pytest.raises(TypeError):
        items.validate_rune_page([8010])
