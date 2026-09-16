"""Combat formulas, cross-checked against three independent sources.

Nothing here needs a server: each formula is pinned against something the
project already trusts -- Map1's own `StatsProgression` table, the literals in
`lanerl_rl/constants.py`, or the C# source. Where the server is *wrong* relative
to real League, that is reproduced and labelled rather than corrected.
"""
from __future__ import annotations

import numpy as np
import pytest

from lanerl_jax.data.patch import CONTENT_ROOT, load_map_table, load_patch
from lanerl_jax.sim.combat import (
    attack_period,
    attack_speed_flat,
    attack_windup,
    growth_sum,
    level_up_factor,
    post_mitigation_damage,
    stat_at_level,
    stat_total,
)

pytestmark = pytest.mark.skipif(
    not CONTENT_ROOT.exists(), reason="vendored Content tree not available"
)


@pytest.fixture(scope="module")
def patch():
    return load_patch()


def test_growth_factor_reproduces_the_servers_own_table():
    """``GetLevelUpStatValue``'s ``0.65 + 0.035*Level`` IS ``PerLevelStatsFactor``.

    Two independent expressions of the same curve -- one computed in `Stats.cs`,
    one tabulated in `Maps/Map1/StatsProgression.json` -- so agreeing at every
    level is real evidence the formula was read correctly, not a tautology.
    """
    table = load_map_table("StatsProgression")["PerLevelStatsFactor"]
    for k, v in table.items():
        lvl = int(k.replace("Level", ""))
        if lvl == 1:
            continue          # the table stores "0"; there is no level-up into 1
        assert level_up_factor(lvl) == pytest.approx(float(v), abs=1e-6), f"level {lvl}"


def test_per_level_growth_is_not_linear(patch):
    """A linear reading is 10% heavy on Garen's HP by level 6 and worsens.

    Large enough to change every trade and every last-hit-under-tower
    calculation, which is why this has its own test rather than being folded
    into the stat accessor.
    """
    g = patch.champion
    assert g.hp_at_level(1) == pytest.approx(616.28)
    linear_l6 = g.base_hp + g.hp_per_level * 5
    assert g.hp_at_level(6) == pytest.approx(995.48, abs=0.01)
    assert linear_l6 == pytest.approx(1096.28, abs=0.01)
    assert linear_l6 > g.hp_at_level(6) * 1.09


def test_growth_sum_matches_an_explicit_loop():
    """The closed form must equal the accumulation it replaces."""
    for n in range(1, 19):
        loop = sum(level_up_factor(L) for L in range(2, n + 1))
        assert float(growth_sum(n)) == pytest.approx(loop, abs=1e-5), f"level {n}"


def test_stat_total_applies_bonuses_in_the_servers_order():
    """``((base + baseBonus) * (1 + pctBase) + flat) * (1 + pct)``.

    Flat lands after the percent-base multiplier and before the percent one;
    any other order gives a different number for the same item set.
    """
    got = stat_total(100.0, 10.0, 0.5, 20.0, 0.1)
    assert got == pytest.approx(((100.0 + 10.0) * 1.5 + 20.0) * 1.1)
    # and the orders really are distinguishable
    assert got != pytest.approx((100.0 + 10.0) * 1.5 * 1.1 + 20.0)


@pytest.mark.parametrize("armor,expected", [(0.0, 100.0), (100.0, 50.0), (27.536, 78.41)])
def test_mitigation_for_positive_resistance(armor, expected):
    got = float(post_mitigation_damage(np.float64(100.0), np.float64(armor)))
    assert got == pytest.approx(expected, abs=0.01)


def test_zero_or_negative_damage_is_zero():
    assert float(post_mitigation_damage(np.float64(0.0), np.float64(50.0))) == 0.0
    assert float(post_mitigation_damage(np.float64(-5.0), np.float64(50.0))) == 0.0


def test_the_negative_resistance_branch_is_the_servers_bug_reproduced():
    """**This is wrong, and reproducing it is the point.**

    ``Stats.GetPostMitigationDamage``::

        mitigationPercent = 100 / (100 + stat)
        if (stat < 0) mitigationPercent = 2 - mitigationPercent

    Real League amplifies damage against negative resistance:
    ``2 - 100/(100 - stat)``, which at -20 armour gives 1.167x. The server's
    version computes ``100/(100 + stat)`` -- note the **plus** -- so at -20 it
    gives 1.25, then ``2 - 1.25 = 0.75`` and the target takes *less* damage.
    At -50 armour it takes **zero**; below that, negative.

    Unreachable in a Garen 1v1: base armour is 27.5 and nothing in the lane
    shreds it, which is why this is pinned rather than escalated. It becomes
    live the moment armour reduction enters the kit, and it will be a
    modern-patch decision then.
    """
    assert float(post_mitigation_damage(np.float64(100.0), np.float64(-20.0))) == \
        pytest.approx(75.0)
    assert float(post_mitigation_damage(np.float64(100.0), np.float64(-50.0))) == \
        pytest.approx(0.0)
    real_league = 2.0 - 100.0 / (100.0 - (-20.0))
    assert real_league == pytest.approx(1.1667, abs=1e-4)


def test_attack_timing_matches_the_literals_lanerl_rl_derived_independently(patch):
    """`lanerl_rl/constants.py` derives these from the same C# by hand.

    Two independent derivations agreeing is the check; a disagreement means one
    of them is stale.
    """
    g = patch.champion
    flat = attack_speed_flat(patch.global_attack_delay, g.attack_delay_offset_percent)
    period = attack_period(flat)
    windup = attack_windup(period, patch.global_attack_delay_cast_percent,
                           g.attack_delay_cast_offset_percent)
    assert period == pytest.approx(1.6)            # GAREN_BASE_ATTACK_PERIOD_S
    assert windup == pytest.approx(1.6 * (0.3 - 0.091666667), abs=1e-6)  # GAREN_BASE_WINDUP_S
    assert flat == pytest.approx(0.625)


def test_attack_speed_multiplier_shortens_the_period(patch):
    g = patch.champion
    flat = attack_speed_flat(patch.global_attack_delay, g.attack_delay_offset_percent)
    assert attack_period(flat, 1.0) == pytest.approx(1.6)
    assert attack_period(flat, 2.0) == pytest.approx(0.8)


def test_the_formulas_are_array_shaped_for_the_sim():
    """One implementation serves the numpy reference and the JAX sim."""
    armor = np.array([0.0, 50.0, 100.0])
    out = post_mitigation_damage(np.full(3, 100.0), armor)
    assert out.shape == (3,)
    np.testing.assert_allclose(out, [100.0, 200.0 / 3.0, 50.0])
    assert stat_at_level(np.array([616.28, 455.0]), np.array([96.0, 0.0]),
                         np.array([6, 6])).shape == (2,)
