"""The patch table, and the asymmetry it exposed.

Parity-first applied to *data*: every number the sim uses comes from the
server's own Content tree, and these tests pin the ones the lane turns on so a
patch swap cannot silently change a mechanic.
"""
from __future__ import annotations

import pytest

from lanerl_jax.data.patch import (
    CONTENT_ROOT,
    UnitStats,
    flag,
    load_character,
    load_patch,
    num,
)

pytestmark = pytest.mark.skipif(
    not CONTENT_ROOT.exists(), reason="vendored Content tree not available"
)


@pytest.fixture(scope="module")
def patch():
    return load_patch()


def test_content_values_may_be_strings_and_are_coerced():
    """``"Armor": "0"`` and ``"ArmorPerLevel": 0`` live in the same file."""
    d = load_character("Garen")
    assert isinstance(d["AcquisitionRange"], str)     # stringly-typed float
    assert num(d, "AcquisitionRange") == 400.0
    assert isinstance(d["IsMelee"], str)              # stringly-typed bool
    assert flag(d, "IsMelee") is True


def test_a_missing_key_raises_rather_than_defaulting_to_zero():
    d = load_character("Garen")
    with pytest.raises(KeyError, match="no default"):
        num(d, "ThisStatDoesNotExist")
    assert num(d, "ThisStatDoesNotExist", 7.0) == 7.0


def test_garen_matches_the_literals_lanerl_rl_constants_carries(patch):
    """These are currently Python literals in ``lanerl_rl/constants.py``.

    The table is meant to replace them, so a disagreement means one of the two
    is stale -- which is the failure mode the whole patch-table design exists
    to remove.
    """
    g = patch.champion
    assert g.base_ad == pytest.approx(57.88)        # GAREN_BASE_AD
    assert g.ad_per_level == pytest.approx(3.5)     # GAREN_AD_PER_LEVEL
    assert g.move_speed == pytest.approx(345.0)     # GAREN_MOVE_SPEED
    assert g.attack_range == pytest.approx(125.0)   # AA_RANGE_GAREN
    assert g.attack_speed_per_level == pytest.approx(2.9)
    assert g.attack_delay_cast_offset_percent == pytest.approx(-0.091666667)
    assert g.pathfinding_radius == pytest.approx(35.0)
    assert g.is_melee is True
    assert patch.global_attack_delay == pytest.approx(1.6)            # GLOBAL_ATTACK_DELAY_S
    assert patch.global_attack_delay_cast_percent == pytest.approx(0.3)
    assert next(iter(patch.turrets.values())).attack_range == pytest.approx(750.0)


def test_the_two_sides_minions_are_NOT_mirror_images(patch):
    """**The lane is not symmetric, and the gold difference is the sharp one.**

    Measured from Content:

    ====================  =====  =====
    field                 blue   red
    ====================  =====  =====
    cannon AttackRange      300    280
    cannon GoldGivenOnDeath  35     30
    ====================  =====  =====

    You farm the *enemy's* minions, so the red champion kills blue cannons
    (35 gold) while the blue champion kills red cannons (30) -- a structural
    income edge to red, every cannon wave, for the whole game.

    This matters well beyond the JAX sim. ``lanerl_rl/obs.py`` canonicalises
    both sides into one lane frame on the stated ground that "blue and red see
    the same picture via the (s,n) -> (L-s,n) reflection", and the reward is
    gold-based. If the two sides do not in fact face the same game, a mirror
    self-play setup is not a mirror, and a policy can learn a side-specific
    habit while the observation insists the sides are equivalent.

    Pinned here rather than fixed: it is the server's data, parity comes first,
    and the JAX sim must reproduce it. Flagged for whoever owns the reward and
    the canonicalisation.
    """
    blue = UnitStats.from_content("Blue_Minion_MechCannon")
    red = UnitStats.from_content("Red_Minion_MechCannon")
    assert blue.attack_range == 300.0 and red.attack_range == 280.0
    assert blue.gold_given_on_death == 35.0 and red.gold_given_on_death == 30.0
    # HP and damage DO match, so the asymmetry is confined to range and bounty.
    assert blue.base_hp == red.base_hp
    assert blue.base_ad == red.base_ad


def test_minion_attack_windup_differs_by_side_on_three_of_four_types():
    """``AttackDelayCastOffsetPercent`` sets when a minion's damage lands.

    Last-hitting is timed against minion HP trajectories, so a windup that
    differs by side makes the last-hit *window* differ by side too -- a subtler
    version of the same asymmetry as the gold.
    """
    seen = {}
    for kind, (b, r) in (("melee", ("Blue_Minion_Basic", "Red_Minion_Basic")),
                         ("caster", ("Blue_Minion_Wizard", "Red_Minion_Wizard")),
                         ("cannon", ("Blue_Minion_MechCannon", "Red_Minion_MechCannon")),
                         ("super", ("Blue_Minion_MechMelee", "Red_Minion_MechMelee"))):
        vb = UnitStats.from_content(b).attack_delay_cast_offset_percent
        vr = UnitStats.from_content(r).attack_delay_cast_offset_percent
        seen[kind] = (vb, vr)
    assert seen["caster"][0] == seen["caster"][1], "casters were symmetric when measured"
    for kind in ("melee", "cannon", "super"):
        assert seen[kind][0] != seen[kind][1], f"{kind} windup was asymmetric when measured"


def test_champion_spawn_hp_exceeds_the_content_base_curve(patch):
    """Content is not the whole story: runes are applied at spawn.

    ``Garen.json`` gives ``BaseHP`` 616.28; the state dump reports 754.0 max HP
    at level 1. The 137.72 gap is the rune page
    (``LanerlEpisode.RestoreBaseline``, pinned by ``lanerl_rl/tests/test_runes.py``).
    So :meth:`UnitStats.hp_at_level` predicts the **base** curve only, and a sim
    that seeds champion HP from it alone starts every episode 137 HP light.
    """
    assert patch.champion.hp_at_level(1) == pytest.approx(616.28)
    observed_spawn_hp = 754.0        # from LANERL_STATEROW, 2026-09-16
    assert observed_spawn_hp - patch.champion.hp_at_level(1) == pytest.approx(137.72, abs=0.01)


def test_exp_curve_is_the_servers(patch):
    assert patch.xp_for_level(2) == 280.0
    assert patch.xp_for_level(6) == 2400.0
    assert patch.xp_for_level(18) == 18360.0
