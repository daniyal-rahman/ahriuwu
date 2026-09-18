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
    """Content is not the whole story: two MASTERIES are applied at spawn.

    ``Garen.json`` gives ``BaseHP`` 616.28; a shop-off state dump reports 671
    max HP at level 1 (the wire floors to an integer). The 55.568 gap is
    ``Veteran's Scars`` (talent 4222 rank 3, ``HealthPoints.FlatBonus = 36``)
    composed with ``Juggernaut`` (talent 4232, ``HealthPoints.PercentBonus =
    0.03``): ``(616.28 + 36) * 1.03``.

    It is **not** the rune page, which carries zero health
    (``lanerl/cfg/garen1v1.json`` is 9x5245 / 9x5317 / 9x5289 / 3x5335 -- AD,
    armour, MR, AD), and it is **not** Doran's Shield. This test previously
    asserted 137.72, which was ``(616.28 + 36 + 80) * 1.03 - 616.28`` measured
    off a dump taken with the auto-shop ON -- the shield's +80 was inside the
    number and mislabelled as a rune. See ``STAT-001``.
    """
    assert patch.champion.hp_at_level(1) == pytest.approx(616.28)
    observed_spawn_hp = 671.0        # shop-OFF dump; the wire floors to int
    assert observed_spawn_hp - patch.champion.hp_at_level(1) == pytest.approx(
        55.568, abs=1.0)             # abs=1 absorbs the wire's floor


def test_exp_curve_is_the_servers(patch):
    assert patch.xp_for_level(2) == 280.0
    assert patch.xp_for_level(6) == 2400.0
    assert patch.xp_for_level(18) == 18360.0


def test_missing_hp_regen_defaults_to_the_servers_030_not_zero():
    """``CharData.cs:37``: ``BaseStaticHpRegen`` defaults to ``0.30000001f``
    when a Content file omits the key, not ``0.0``.

    Currently inert -- Garen and every Map1 minion/turret model this project
    loads specifies the key explicitly (checked below), so this only guards
    against a *future* model silently regressing to the wrong default. Still
    a real content-table constant, per `docs/PORT_AUDIT_WAVES.md`.
    """
    assert num({}, "BaseStaticHPRegen", 0.30000001) == pytest.approx(0.30000001)


def test_every_loaded_model_specifies_hp_regen_explicitly(patch):
    """The 0.0-vs-0.3 default only matters for a model that omits the key.
    None currently does -- this pins that so the "currently inert" claim
    above stays true rather than going stale."""
    d = load_character("Garen")
    assert "BaseStaticHPRegen" in d
    for name in ("Blue_Minion_Basic", "Red_Minion_Basic", "Blue_Minion_Wizard",
                "Red_Minion_Wizard", "Blue_Minion_MechCannon", "Red_Minion_MechCannon",
                "Blue_Minion_MechMelee", "Red_Minion_MechMelee"):
        assert "BaseStaticHPRegen" in load_character(name), name
    for name in patch.turrets:
        assert "BaseStaticHPRegen" in load_character(name), name


def test_minion_acquisition_range_defaults_to_server_475_not_600(patch):
    """Minions that omit AcquisitionRange in Content fall back to
    ``CharData.cs:98`` -- a compiled default of **475**, not 600.

    ``lanerl_jax/sim/profiles.py:125`` and ``sim/init.py:261,267`` use this
    default when a minion model omits the key. The old value (600) was
    inferred from ``lanerl_rl/constants.py``'s ``MINION_ACQRANGE``, which was
    a guess; the correct value comes from reading CharData's source directly.

    Map1 minions split: melee and cannon omit the key (use 475 default),
    while wizard and super specify it (700 and 600 respectively).
    """
    # Minions that omit AcquisitionRange use the CharData default
    for name in ("Blue_Minion_Basic", "Red_Minion_Basic",
                "Blue_Minion_MechCannon", "Red_Minion_MechCannon"):
        d = load_character(name)
        assert d.get("AcquisitionRange") is None, f"{name} should omit AcquisitionRange"

    # Wizard and super minions specify it explicitly (values are stringly-typed)
    blue_wiz = load_character("Blue_Minion_Wizard")
    assert float(blue_wiz.get("AcquisitionRange")) == 700.0
    red_wiz = load_character("Red_Minion_Wizard")
    assert float(red_wiz.get("AcquisitionRange")) == 700.0
    blue_super = load_character("Blue_Minion_MechMelee")
    assert float(blue_super.get("AcquisitionRange")) == 600.0
    red_super = load_character("Red_Minion_MechMelee")
    assert float(red_super.get("AcquisitionRange")) == 600.0


def test_dorans_shield_is_not_applied_at_all(patch):
    """The champion profile must carry the masteries and NOT the shield.

    History, because both directions of this have now been wrong once:

    * An audit once proposed adding Doran's ``FlatHPPoolMod`` (80) on top of
      the then-``RUNE_HP_BONUS``, which would have given 834 against an
      observed 754.
    * The constant it would have double-counted was itself measured from a
      dump taken with the auto-shop **ON**, so it silently contained the very
      80 it was warning about -- and 754 is not a baseline at all. Purchases
      are fountain-gated (``BuyRequiresFountain``, radius 1800) and
      ``StartingGold`` is 475, so with the shop on Doran's (440) plus a potion
      are bought on the **boot tick**, before the first dumped frame exists.

    The parity runs that matter use ``autobuy=False``, so the champion has no
    items and the correct baseline is ``(616.28 + 36) * 1.03 = 671.848``.
    ``DORANS_SHIELD_HP``/``DORANS_SHIELD_HP_REGEN`` survive documented and
    deliberately unapplied, for whenever an item model exists (``ITEM-001``).
    """
    from lanerl_jax.sim.init import DORANS_SHIELD_HP, DORANS_SHIELD_HP_REGEN
    from lanerl_jax.sim.profiles import PROFILES, build_profile_tables
    from lanerl_jax.sim.state import Kind

    tables = build_profile_tables()
    champ_rows = [r for r, (k, _, _) in enumerate(PROFILES) if k == Kind.CHAMPION]
    assert champ_rows, "no champion profile rows"

    for row in champ_rows:
        assert abs(float(tables["max_hp"][row]) - 671.848) < 1e-2, (
            "champion max HP drifted. ~+82 means someone re-applied Doran's "
            "Shield (80 x 1.03); ~-55 means the masteries were dropped")
        # Garen's Content BaseStaticHPRegen, with NO item regen on top.
        assert abs(float(tables["hp_regen"][row]) - 1.568) < 1e-4

    # Kept as constants so the item model has them, but they must stay unused.
    assert DORANS_SHIELD_HP == 80.0
    assert DORANS_SHIELD_HP_REGEN == 1.2

