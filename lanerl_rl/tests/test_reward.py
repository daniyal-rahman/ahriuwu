"""The lane reward (correction 2).

The tests that matter here are the ones that would catch a *plausible* mistake,
not the ones that restate the code:

* the HP term is a difference of a potential, not a potential of a difference
  -- with the exact drift the wrong form produces, so the test fails loudly
  rather than by a tolerance;
* the money term does not pay for the passage of time;
* the reward is exactly antisymmetric at alpha = 1 and deliberately is not
  during the anneal;
* the last-hit shaping is a genuine Ng-et-al. potential term, i.e. it
  telescopes to zero around any loop;
* the absolute metrics are not the training reward.
"""

from __future__ import annotations

import math

import pytest

from lanerl_rl import constants as C
from lanerl_rl.reward import (
    AbsoluteLaneMetrics,
    LaneRewardConfig,
    RewardWeights,
    WinRateTracker,
    ZeroSumLaneReward,
    _wrong_hp_potential_of_delta,
    hp_potential,
    hp_potential_delta,
    lane_outcome,
    last_hit_potential,
)
from lanerl_rl.scenarios import make_frame, top_lane_scenario, unit


# --------------------------------------------------------------------------
# The HP potential
# --------------------------------------------------------------------------


def test_hp_potential_endpoints_and_curvature():
    assert hp_potential(0.0) == 0.0
    assert hp_potential(1.0) == 1.0
    # phi'(x) = (1 + 4(1-x)^3) / 2  ->  5x steeper at 0 hp than at full hp.
    eps = 1e-6
    d_low = (hp_potential(eps) - hp_potential(0.0)) / eps
    d_high = (hp_potential(1.0) - hp_potential(1.0 - eps)) / eps
    assert d_low / d_high == pytest.approx(5.0, rel=1e-3)


def test_hp_term_telescopes_to_zero_around_a_loop():
    """The correct form is a potential difference, so any cycle sums to 0."""
    path = [1.0, 0.98, 0.90, 0.55, 0.90, 0.98, 1.0]
    total = sum(hp_potential_delta(b, a) for a, b in zip(path, path[1:]))
    assert total == pytest.approx(0.0, abs=1e-12)


def test_the_wrong_form_leaks_the_documented_amount_per_oscillation():
    """phi(delta) instead of delta(phi): -0.0048 per 2% trade cycle, at w = 2.

    This is the pylol bug.  It is worth an exact number rather than a
    hand-wave, because -0.005 a cycle at 15 Hz is a large standing gradient
    away from ever trading.
    """
    w = RewardWeights().hp_point
    assert w == 2.0
    good = w * (hp_potential_delta(0.98, 1.00) + hp_potential_delta(1.00, 0.98))
    bad = w * (_wrong_hp_potential_of_delta(0.98, 1.00) + _wrong_hp_potential_of_delta(1.00, 0.98))
    assert good == pytest.approx(0.0, abs=1e-12)
    assert bad == pytest.approx(-0.0048, abs=1e-4)
    assert bad < -1e-3, "the wrong form must be measurably wrong, or the test is theatre"


def test_the_wrong_form_also_destroys_the_near_death_weighting():
    """The quartic's whole point is that HP is worth ~5x more near death."""
    w = RewardWeights().hp_point
    near_death = abs(w * hp_potential_delta(0.05, 0.15))
    near_full = abs(w * hp_potential_delta(0.90, 1.00))
    assert near_death / near_full > 2.5

    bad_near_death = abs(w * _wrong_hp_potential_of_delta(0.05, 0.15))
    bad_near_full = abs(w * _wrong_hp_potential_of_delta(0.90, 1.00))
    assert bad_near_death == pytest.approx(bad_near_full), (
        "phi(delta) cannot depend on the hp level at all -- that is the loss"
    )


# --------------------------------------------------------------------------
# Gold: earned, not accrued
# --------------------------------------------------------------------------


def test_ambient_gold_is_19_per_10_seconds():
    """Read out of Maps/Map1/Constants.json, not guessed."""
    assert C.AMBIENT_GOLD_AMOUNT == 9.5
    assert C.AMBIENT_GOLD_INTERVAL_S == 5.0
    assert C.AMBIENT_GOLD_PER_S * 10.0 == pytest.approx(19.0)
    assert C.AMBIENT_GOLD_DELAY_S == 90.0


def test_the_passive_trickle_is_about_45_percent_of_ten_minutes_of_gold():
    """The arithmetic that rules out rewarding delta(total gold)."""
    ambient = (600.0 - C.AMBIENT_GOLD_DELAY_S) * C.AMBIENT_GOLD_PER_S
    assert ambient == pytest.approx(969.0)
    earned_from_60_cs = 60 * 20.0
    assert ambient / (ambient + earned_from_60_cs) > 0.4


def _idle_lane_frames(n=120, dt_ms=1000, t0=100_000, gold0=600.0):
    """Both champions standing still; gold rises only from the ambient trickle."""
    frames = []
    for i in range(n):
        t = t0 + i * dt_ms
        gold = gold0 + C.AMBIENT_GOLD_PER_S * (t - t0) / 1000.0
        f = top_lane_scenario(t_ms=t, n_minions=0, blue_gold=gold, red_gold=gold)
        frames.append(f)
    return frames


def test_money_term_ignores_the_ambient_trickle():
    cfg = LaneRewardConfig(last_hit_shaping=False, subtract_ambient_gold=True)
    r = ZeroSumLaneReward(cfg=cfg)
    total = 0.0
    for f in _idle_lane_frames():
        _rew, info = r.step(f)
        total += info["terms"][C.TEAM_BLUE].get("money", 0.0)
    assert abs(total) < 1e-4, f"the trickle paid out {total}"


def test_without_the_correction_the_trickle_is_paid_for_doing_nothing():
    """Negative control: the subtraction must be doing real work."""
    cfg = LaneRewardConfig(last_hit_shaping=False, subtract_ambient_gold=False)
    r = ZeroSumLaneReward(cfg=cfg)
    total = sum(r.step(f)[1]["terms"][C.TEAM_BLUE].get("money", 0.0) for f in _idle_lane_frames())
    # 119 s of trickle at 1.9 gold/s, weighted 0.008.
    assert total == pytest.approx(119.0 * C.AMBIENT_GOLD_PER_S * 0.008, rel=1e-3)
    assert total > 1.5


# --------------------------------------------------------------------------
# Zero sum
# --------------------------------------------------------------------------


def test_reward_is_exactly_antisymmetric_at_alpha_one():
    cfg = LaneRewardConfig(
        last_hit_shaping=False, zero_sum_alpha_start=1.0, zero_sum_alpha_end=1.0
    )
    r = ZeroSumLaneReward(cfg=cfg)
    for i in range(30):
        f = top_lane_scenario(
            t_ms=100_000 + 100 * i,
            blue_hp_frac=max(0.3, 1.0 - 0.02 * i),
            red_hp_frac=max(0.4, 1.0 - 0.01 * i),
            blue_gold=600.0 + 7 * i,
            red_gold=600.0 + 3 * i,
        )
        rew, _ = r.step(f)
        assert rew[C.TEAM_BLUE] == pytest.approx(-rew[C.TEAM_RED], abs=1e-9)


def test_alpha_anneals_from_half_to_one():
    cfg = LaneRewardConfig(zero_sum_anneal_steps=1000)
    assert cfg.alpha(0) == pytest.approx(0.5)
    assert cfg.alpha(500) == pytest.approx(0.75)
    assert cfg.alpha(1000) == pytest.approx(1.0)
    assert cfg.alpha(10_000) == pytest.approx(1.0)


def test_during_the_anneal_the_reward_is_deliberately_not_antisymmetric():
    cfg = LaneRewardConfig(last_hit_shaping=False, zero_sum_anneal_steps=1_000_000)
    r = ZeroSumLaneReward(cfg=cfg)
    r.step(top_lane_scenario(t_ms=100_000))
    rew, info = r.step(top_lane_scenario(t_ms=100_100, blue_hp_frac=0.8))
    assert info["alpha"] == pytest.approx(0.5)
    assert rew[C.TEAM_BLUE] != pytest.approx(-rew[C.TEAM_RED])


def _death_frames():
    """Blue kills red between the second and third frame."""
    out = []
    for i, red_hp in enumerate((1.0, 0.2, 0.0)):
        out.append(top_lane_scenario(t_ms=100_000 + 100 * i, red_hp_frac=red_hp, n_minions=0))
    return out


def test_kill_and_death_weights_balance_to_plus_or_minus_half():
    """-0.5 kill and -1.0 death is what makes the pair antisymmetric."""
    w = RewardWeights()
    assert w.kill == -0.5 and w.death == -1.0
    cfg = LaneRewardConfig(
        last_hit_shaping=False,
        zero_sum_alpha_start=1.0,
        zero_sum_alpha_end=1.0,
        weights=RewardWeights(hp_point=0.0, tower_hp=0.0, money=0.0, exp=0.0, last_hit=0.0),
    )
    r = ZeroSumLaneReward(cfg=cfg)
    rew = None
    for f in _death_frames():
        rew, info = r.step(f)
    assert rew[C.TEAM_BLUE] == pytest.approx(0.5, abs=1e-9)
    assert rew[C.TEAM_RED] == pytest.approx(-0.5, abs=1e-9)


# --------------------------------------------------------------------------
# Potential-based last-hit shaping
# --------------------------------------------------------------------------


def _lane_with_minion(t_ms, minion_hp, minion_offset):
    a = C.TOP_OUTER_TURRET[C.TEAM_BLUE]
    return make_frame(
        t_ms,
        [
            unit(1001, "champion", C.TEAM_BLUE, a[0], a[1], hp=671, mhp=671,
                 gold=600.0, xp=0.0, lvl=1),
            unit(1002, "champion", C.TEAM_RED, 12000, 12000, hp=671, mhp=671,
                 gold=600.0, xp=0.0, lvl=1),
            unit(3001, "minion", C.TEAM_RED, a[0] + minion_offset, a[1],
                 hp=minion_hp, mhp=455),
        ],
    )


def test_shaping_potential_rises_as_a_killable_minion_comes_into_range():
    ad = C.garen_attack_damage(1)
    far = last_hit_potential(_lane_with_minion(0, 20, 900), None, C.TEAM_RED, ad)
    assert far == 0.0
    f = _lane_with_minion(0, 20, 100)
    near = last_hit_potential(f, f.champion_of_team(C.TEAM_BLUE), C.TEAM_RED, ad)
    assert near > 0.04  # c = 0.05 and one very killable minion
    healthy = _lane_with_minion(0, 400, 100)
    assert last_hit_potential(
        healthy, healthy.champion_of_team(C.TEAM_BLUE), C.TEAM_RED, ad
    ) < 1e-6


def test_shaping_telescopes_to_zero_around_a_loop_at_gamma_one():
    """Ng et al. (1999): F = gamma * Phi(s') - Phi(s) is policy-invariant.

    At gamma = 1 the sum of F around any cycle of states is exactly 0, which is
    the property that lets this term be tuned without changing the optimum.
    """
    cfg = LaneRewardConfig(
        gamma=1.0,
        weights=RewardWeights(hp_point=0.0, tower_hp=0.0, money=0.0, exp=0.0,
                              last_hit=0.0, death=0.0, kill=0.0),
    )
    r = ZeroSumLaneReward(cfg=cfg)
    offsets = [900, 400, 100, 100, 400, 900]
    total = 0.0
    for i, off in enumerate(offsets):
        _rew, info = r.step(_lane_with_minion(100_000 + 100 * i, 20, off))
        total += info["shaping"][C.TEAM_BLUE]
    assert total == pytest.approx(0.0, abs=1e-9)


def test_shaping_is_nonzero_in_between():
    """A telescoping sum of zeros would also pass the test above."""
    cfg = LaneRewardConfig(gamma=1.0)
    r = ZeroSumLaneReward(cfg=cfg)
    r.step(_lane_with_minion(100_000, 20, 900))
    _rew, info = r.step(_lane_with_minion(100_100, 20, 100))
    assert info["shaping"][C.TEAM_BLUE] > 0.01


def test_shaping_kappa_matches_the_observation_feature():
    """The reward's 'nearly dead' and the observation's must be the same curve."""
    assert LaneRewardConfig().shaping_kappa == C.AA_KILL_KAPPA_HP
    assert LaneRewardConfig().shaping_eps == C.AA_RANGE_EPS


def test_shaping_gamma_matches_the_trainer_gamma_by_default():
    """F = gamma*Phi' - Phi is only policy-invariant at the trainer's gamma.

    Two independently-defaulted gammas is exactly the sort of thing that drifts
    apart in a config refactor and quietly stops the shaping being free.
    """
    from lanerl_rl.ppo import PPOConfig

    assert LaneRewardConfig().gamma == pytest.approx(PPOConfig().gamma, rel=1e-12)


# --------------------------------------------------------------------------
# The reward wiring as a whole
# --------------------------------------------------------------------------


def test_weights_match_the_published_1v1_table():
    w = RewardWeights()
    assert w.hp_point == 2.0
    assert w.tower_hp == 10.0
    assert w.money == 0.008
    assert w.exp == 0.008
    assert w.death == -1.0
    assert w.kill == -0.5
    assert w.last_hit == 0.5
    assert w.mana == 0.0, "Garen has no mana bar; the deviation must be explicit"


def test_last_hits_are_rewarded():
    cfg = LaneRewardConfig(last_hit_shaping=False)
    r = ZeroSumLaneReward(cfg=cfg)
    a = C.TOP_OUTER_TURRET[C.TEAM_BLUE]

    def f(t_ms, with_minion):
        units = [
            unit(1001, "champion", C.TEAM_BLUE, a[0], a[1], hp=671, mhp=671,
                 gold=600.0, xp=0.0, lvl=1),
            unit(1002, "champion", C.TEAM_RED, 12000, 12000, hp=671, mhp=671,
                 gold=600.0, xp=0.0, lvl=1),
        ]
        if with_minion:
            units.append(unit(3001, "minion", C.TEAM_RED, a[0] + 100, a[1], hp=30, mhp=455))
        return make_frame(t_ms, units)

    r.step(f(100_000, True))
    _rew, info = r.step(f(100_100, False))
    assert info["terms"][C.TEAM_BLUE]["last_hit"] == pytest.approx(0.5)


def test_reward_is_finite_over_the_real_recording(recording_path):
    from lanerl_rl.frame import iter_jsonl
    import itertools

    r = ZeroSumLaneReward()
    for f in itertools.islice(iter_jsonl(recording_path), 900, 1400):
        rew, _info = r.step(f)
        for v in rew.values():
            assert math.isfinite(v)


# --------------------------------------------------------------------------
# Absolute evaluation
# --------------------------------------------------------------------------


def test_absolute_metrics_record_cs_at_ten_minutes(recording_path):
    from lanerl_rl.frame import iter_jsonl

    m = AbsoluteLaneMetrics(C.TEAM_BLUE)
    for f in iter_jsonl(recording_path):
        m.update(f)
    rep = m.report()
    assert "cs_at_10min" in rep and "cs_at_5min" in rep
    assert rep["duration_s"] > 400.0
    assert math.isfinite(rep["cs"])


def test_win_rate_is_nan_before_the_first_game_not_fifty_percent():
    """A 50% default would be indistinguishable from a real 50%."""
    t = WinRateTracker()
    assert math.isnan(t.rate())
    t.record("win")
    t.record("loss")
    assert t.rate() == pytest.approx(0.5)
    t.record("draw")
    assert t.rate() == pytest.approx(1.5 / 3)


def test_lane_outcome_prefers_a_destroyed_turret_then_gold():
    f = top_lane_scenario(blue_gold=1000.0, red_gold=600.0)
    assert lane_outcome(f, C.TEAM_BLUE, 50, 40) == "win"
    assert lane_outcome(f, C.TEAM_RED, 40, 50) == "loss"
    even = top_lane_scenario(blue_gold=700.0, red_gold=700.0)
    assert lane_outcome(even, C.TEAM_BLUE, 50, 40) == "win"
    assert lane_outcome(even, C.TEAM_BLUE, 40, 40) == "draw"
    dead_turret = top_lane_scenario(blue_gold=1000.0, red_gold=600.0)
    for u in dead_turret.units.values():
        if u.etype == "turret" and u.team == C.TEAM_BLUE:
            u.hp = 0.0
    assert lane_outcome(dead_turret, C.TEAM_BLUE, 50, 40) == "loss"


def test_absolute_metrics_are_not_the_training_reward():
    """A structural check: nothing in ZeroSumLaneReward touches the metrics."""
    import inspect

    from lanerl_rl import reward as R

    src = inspect.getsource(R.ZeroSumLaneReward)
    assert "AbsoluteLaneMetrics" not in src
    assert "WinRateTracker" not in src


# --------------------------------------------------------------------------
# Spending gold
# --------------------------------------------------------------------------


def _purchase_frames(cost=1000.0, t0=200_000, gold0=1200.0):
    """Wallet full, then an item is bought: the wallet DROPS by `cost`."""
    return [
        top_lane_scenario(t_ms=t0, n_minions=0, blue_gold=gold0, red_gold=gold0),
        top_lane_scenario(t_ms=t0 + 1000, n_minions=0,
                          blue_gold=gold0 - cost, red_gold=gold0),
    ]


def test_buying_an_item_is_not_punished():
    """`gold` is the WALLET, so a purchase makes it fall.

    Scoring the raw wallet delta paid `money * -1000` for a 1000g item, i.e. it
    penalised shopping and rewarded hoarding. Income is what `money` should
    measure, so a purchase must be neutral on that term.
    """
    cfg = LaneRewardConfig(last_hit_shaping=False, subtract_ambient_gold=False)
    r = ZeroSumLaneReward(cfg=cfg)
    money = 0.0
    for f in _purchase_frames():
        _rew, info = r.step(f)
        money += info["terms"][C.TEAM_BLUE].get("money", 0.0)
    assert money >= 0.0, f"buying an item was penalised: money term = {money}"


def test_spending_earns_a_small_bonus_worth_a_quarter_of_the_money_weight():
    cfg = LaneRewardConfig(last_hit_shaping=False, subtract_ambient_gold=False)
    w = cfg.weights
    assert 0.2 <= w.spend / w.money <= 0.3, "spend should be 0.2-0.3x money"
    r = ZeroSumLaneReward(cfg=cfg)
    spend = 0.0
    for f in _purchase_frames(cost=1000.0):
        _rew, info = r.step(f)
        spend += info["terms"][C.TEAM_BLUE].get("spend", 0.0)
    assert spend == pytest.approx(w.spend * 1000.0), f"spend term = {spend}"


def test_hoarding_earns_nothing_extra():
    """The bonus is for CONVERTING gold, not for having it."""
    cfg = LaneRewardConfig(last_hit_shaping=False, subtract_ambient_gold=False)
    r = ZeroSumLaneReward(cfg=cfg)
    spend = 0.0
    for f in _idle_lane_frames(n=30):
        _rew, info = r.step(f)
        spend += info["terms"][C.TEAM_BLUE].get("spend", 0.0)
    assert spend == 0.0, f"a wallet that only grew paid a spend bonus of {spend}"
