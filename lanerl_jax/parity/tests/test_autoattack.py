"""The auto-attack clock, and how far it can be checked without a better oracle.

The state dump carries no auto-attack state at all -- measured:
``cast_spell`` is ``"-"`` on all 15,162 champion snapshots of a run in which the
champion swung 32 times. So swings are observable *only* through the control
wire's ``atk`` flag, at the 33.3 ms decision rate rather than the 16.67 ms tick
rate. That is why ``auto_attack_cooldown`` and ``has_auto_attacked`` are in
``lanerl_jax.parity.diff.UNOBSERVABLE``.

What *can* be pinned exactly is the attack-speed stat itself, because the dump
does carry ``AttackSpeedMultiplier``.
"""
from __future__ import annotations

import numpy as np
import pytest

from lanerl_jax.data.patch import CONTENT_ROOT, load_patch
from lanerl_jax.sim.autoattack import ideal_attack_range, step_autoattack
from lanerl_jax.sim.combat import (
    attack_period,
    attack_speed_flat,
    attack_windup,
    level_up_factor,
    stat_total,
)

pytestmark = pytest.mark.skipif(
    not CONTENT_ROOT.exists(), reason="vendored Content tree not available"
)

#: Read off the state dump of a real run (2026-09-16), StatQ = 1/1024.
OBSERVED_ATTACK_SPEED_MULTIPLIER = {1: 1.0, 2: 1.020508, 3: 1.042969}


@pytest.fixture(scope="module")
def patch():
    return load_patch()


def test_attack_speed_multiplier_matches_the_dump_at_every_observed_level(patch):
    """A real end-to-end check of the stat machinery, on observed numbers.

    ``LevelUp`` feeds ``GrowthAttackSpeed / 100`` into the multiplier's
    ``PercentBaseBonus`` through the non-linear growth factor, and ``Stat.Total``
    composes it. Reproducing 1.020508 and 1.042969 to the dump's own
    quantisation exercises `level_up_factor`, `stat_total` and the Content
    value of `AttackSpeedPerLevel` together.
    """
    growth = patch.champion.attack_speed_per_level / 100.0      # 2.9 -> 0.029
    pct = 0.0
    for level in (1, 2, 3):
        if level > 1:
            pct += growth * level_up_factor(level)
        total = stat_total(1.0, 0.0, pct, 0.0, 0.0)
        quantised = round(total * 1024) / 1024
        assert quantised == pytest.approx(
            OBSERVED_ATTACK_SPEED_MULTIPLIER[level], abs=1e-6), f"level {level}"


def test_level_one_period_is_exactly_1600ms(patch):
    """The dump says the multiplier is exactly 1.0 at level 1, so the period is
    the bare ``gcd_AttackDelay``. Everything about swing timing is anchored here."""
    g = patch.champion
    flat = attack_speed_flat(patch.global_attack_delay, g.attack_delay_offset_percent)
    assert OBSERVED_ATTACK_SPEED_MULTIPLIER[1] == 1.0
    assert attack_period(flat, 1.0) * 1000.0 == pytest.approx(1600.0)


def test_ideal_range_adds_only_the_targets_radius(patch):
    """``Stats.Range.Total + TargetUnit.CollisionRadius`` -- edge to edge.

    The attacker's own radius is not added. The server's comment: "range is
    center -> center VS edge -> edge for attacks". Adding both would shift every
    last-hit by a champion radius.
    """
    assert ideal_attack_range(125.0, 48.0) == 173.0
    assert ideal_attack_range(patch.champion.attack_range, 0.0) == 125.0


def _swing_times(period, windup, ticks=600, dtype=np.float64):
    n = 1
    dt = 1000.0 / 60.0
    cd = np.zeros(n, dtype)
    wu = np.zeros(n, dtype)
    atk = np.zeros(n, bool)
    had = np.zeros(n, bool)
    hits = []
    for t in range(ticks):
        out = step_autoattack(
            cd, wu, atk, had,
            in_range=np.ones(n, bool), can_attack=np.ones(n, bool),
            has_target=np.ones(n, bool),
            attack_period=np.full(n, period, dtype),
            windup_time=np.full(n, windup, dtype),
            attack_damage=np.full(n, 60.0, dtype),
            target_resist=np.zeros(n, dtype), delta_ms=dt)
        cd, wu, atk, had = (out.aa_cooldown.astype(dtype), out.aa_windup.astype(dtype),
                            out.is_attacking, out.has_auto_attacked)
        if out.hit[0]:
            hits.append(t * dt)
    return hits


def test_the_first_hit_lands_after_exactly_one_windup(patch):
    g = patch.champion
    flat = attack_speed_flat(patch.global_attack_delay, g.attack_delay_offset_percent)
    period = attack_period(flat)
    windup = attack_windup(period, patch.global_attack_delay_cast_percent,
                           g.attack_delay_cast_offset_percent)
    hits = _swing_times(period, windup, ticks=40)
    assert hits, "no hit landed in 40 ticks"
    assert hits[0] == pytest.approx(windup * 1000.0, abs=1000.0 / 60.0)


def test_swings_are_periodic_to_within_one_tick(patch):
    g = patch.champion
    flat = attack_speed_flat(patch.global_attack_delay, g.attack_delay_offset_percent)
    period = attack_period(flat)
    windup = attack_windup(period, 0.3, g.attack_delay_cast_offset_percent)
    hits = _swing_times(period, windup, ticks=600)
    gaps = [hits[i + 1] - hits[i] for i in range(len(hits) - 1)]
    assert gaps, "fewer than two hits"
    for gp in gaps:
        # 97 ticks, not 96 -- see test_the_cooldown_drains_in_97_ticks_*
        assert gp == pytest.approx(97 * 1000.0 / 60.0, abs=1e-6)


#: Level-pinned measurement, 2026-09-16: both champions fought before the 90 s
#: first wave, so neither could gain XP and both stayed level 1 for all 2,600
#: decisions. 15 swings, 14 clean gaps.
OBSERVED_LEVEL1_SWING_GAP_MS = 1616.79


def test_the_cooldown_drains_in_97_ticks_and_the_server_agrees(patch):
    """**1616.67 ms, not the nominal 1600** -- and the server does the same.

    The cooldown is ``cd -= diff/1000`` per tick with every term a C# float.
    After 96 ticks of a 1.6 s cooldown the residue is ~1e-7 **positive**, so a
    97th tick is needed and the true period is 97/60 = 1616.67 ms. That is a 1%
    slower attack rate than the nominal 0.625/s, which is dozens of swings over
    a ten-minute lane.

    Getting the evidence took a level-pinned run. A first attempt measured a
    *median* of 1600 ms and looked like a 1-tick disagreement, but it was
    confounded twice over: the champion levelled 1->3 mid-run (periods 1600 /
    1567.9 / 1534 ms), and the control wire samples every 2 ticks so gaps
    quantise to 33.3 ms -- a true 1616.67 ms period is *observed* as a mix of
    1600 and 1633, never as 1616. Pinning the level and taking the **mean**
    instead of the median gives 1616.79 ms.

    Two lessons already recorded elsewhere both bite here: match the server's
    float32 (a float64 accumulation lands elsewhere), and read the oracle's
    sampling rate before trusting a statistic from it.
    """
    dt32 = np.float32(np.float32(1000.0 / 60.0) / np.float32(1000.0))
    for dtype in (np.float32, np.float64):
        cd = dtype(1.6)
        step = dtype(dt32)
        n = 0
        while cd > 0:
            cd = dtype(cd - step)
            n += 1
        assert n == 97, f"{dtype.__name__} drained in {n} ticks"

    assert 97 * 1000.0 / 60.0 == pytest.approx(1616.67, abs=0.01)
    assert OBSERVED_LEVEL1_SWING_GAP_MS == pytest.approx(97 * 1000.0 / 60.0, abs=0.5)
    # and the port's own swing spacing agrees with the server's measurement
    g = patch.champion
    flat = attack_speed_flat(patch.global_attack_delay, g.attack_delay_offset_percent)
    period = attack_period(flat)
    windup = attack_windup(period, 0.3, g.attack_delay_cast_offset_percent)
    hits = _swing_times(period, windup, ticks=900)
    gaps = [hits[i + 1] - hits[i] for i in range(len(hits) - 1)]
    assert sum(gaps) / len(gaps) == pytest.approx(
        OBSERVED_LEVEL1_SWING_GAP_MS, abs=1.0)


def test_no_swing_without_a_target_or_out_of_range():
    n = 3
    z = np.zeros(n)
    out = step_autoattack(
        z.copy(), z.copy(), np.zeros(n, bool), np.zeros(n, bool),
        in_range=np.array([True, False, True]),
        can_attack=np.array([True, True, False]),
        has_target=np.array([True, True, True]),
        attack_period=np.full(n, 1.6), windup_time=np.full(n, 0.33),
        attack_damage=np.full(n, 60.0), target_resist=z.copy())
    assert bool(out.is_attacking[0]) and not bool(out.is_attacking[1])
    assert not bool(out.is_attacking[2])


def test_damage_goes_through_mitigation():
    n = 1
    out = step_autoattack(
        np.zeros(n), np.array([1e-9]), np.ones(n, bool), np.zeros(n, bool),
        in_range=np.ones(n, bool), can_attack=np.ones(n, bool),
        has_target=np.ones(n, bool),
        attack_period=np.full(n, 1.6), windup_time=np.full(n, 0.33),
        attack_damage=np.full(n, 100.0), target_resist=np.full(n, 100.0))
    assert bool(out.hit[0])
    assert float(out.damage[0]) == pytest.approx(50.0)
