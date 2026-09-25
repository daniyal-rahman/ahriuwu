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
* the absolute metrics are not the training reward;
* **the BALANCE between the terms** -- see "The weight balance" at the bottom.
  Those are the ones that fail if a future edit silently re-inflates ``exp``
  back to the published 0.008, or quietly halves ``last_hit``.
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
    """phi(delta) instead of delta(phi): -0.0024 * w per 2% trade cycle.

    This is the pylol bug.  It is worth an exact number rather than a
    hand-wave, because -0.01 a cycle at 15 Hz is a large standing gradient
    away from ever trading -- and the drift scales with ``hp_point``, so
    raising that weight to make trading worth doing would have made this bug
    worse had it still been here.
    """
    w = RewardWeights().hp_point
    assert w == 4.0
    good = w * (hp_potential_delta(0.98, 1.00) + hp_potential_delta(1.00, 0.98))
    bad = w * (_wrong_hp_potential_of_delta(0.98, 1.00) + _wrong_hp_potential_of_delta(1.00, 0.98))
    assert good == pytest.approx(0.0, abs=1e-12)
    assert bad / w == pytest.approx(-0.0024, abs=1e-5)
    assert bad == pytest.approx(-0.0096, abs=1e-4)
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
        # lane_presence zeroed with the rest: this test is about the kill/death
        # PAIR being antisymmetric, and it asserts on the total reward, so any
        # other live term leaks in. It does not cancel under zero-sum either --
        # the dead champion is not in lane, so only the killer collects it.
        weights=RewardWeights(hp_point=0.0, tower_hp=0.0, money=0.0, exp=0.0,
                              last_hit=0.0, lane_presence=0.0),
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
            # ad is REQUIRED for the last-hit shaping potential: there is no
            # Python attack-damage derivation to fall back on any more, so a
            # champion with no `ad` has an unknown one and the potential is 0.
            # Real frames always carry it; a synthetic frame must too.
            unit(1001, "champion", C.TEAM_BLUE, a[0], a[1], hp=671, mhp=671,
                 gold=600.0, xp=0.0, lvl=1, ad=78.14),
            unit(1002, "champion", C.TEAM_RED, 12000, 12000, hp=671, mhp=671,
                 gold=600.0, xp=0.0, lvl=1, ad=78.14),
            unit(3001, "minion", C.TEAM_RED, a[0] + minion_offset, a[1],
                 hp=minion_hp, mhp=455),
        ],
    )


def test_shaping_potential_rises_as_a_killable_minion_comes_into_range():
    # The server's level-1 total for the configured rune + mastery page, as a
    # literal: there is deliberately no Python AD derivation any more, because
    # every version of it was wrong (see constants.py / test_runes.py).
    ad = 78.14
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


def test_weights_that_come_straight_from_the_published_1v1_table():
    """The four the retune did NOT touch.

    ``kill``/``death`` above all: they are a *pair*, and the module docstring's
    antisymmetry argument dies the moment one of them is rescaled alone.
    """
    w = RewardWeights()
    assert w.tower_hp == 10.0
    assert w.money == 0.008
    assert w.death == -1.0
    assert w.kill == -0.5


def test_every_deviation_from_the_published_table_is_deliberate():
    """The four that differ, each with its reason.

    This is a change-detector on purpose.  Every one of these numbers is
    argued with arithmetic in the module docstring, and a future edit that
    moves one without moving that argument should have to come through here.
    """
    w = RewardWeights()
    assert w.mana == 0.0, "Garen has no mana bar; the deviation must be explicit"
    assert w.exp == 0.001, "published 0.008: XP is paid for proximity, not skill"
    assert w.last_hit == 1.0, "published 0.5: this is the objective"
    assert w.hp_point == 4.0, "published 2.0: at 2.0 the agent never traded"
    assert w.spend == 0.0, "the purchase is scripted and lands on the respawn"


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
    assert info["terms"][C.TEAM_BLUE]["last_hit"] == pytest.approx(RewardWeights().last_hit)


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


def test_spending_pays_nothing_but_is_still_counted():
    """`spend` is a named zero, and the raw gold is still reported.

    Two different things used to be indistinguishable in a run log: "the term
    is weighted zero" and "the detector never fired".  The weight is zero AND
    the unweighted gold is surfaced, so a future run can tell them apart.
    """
    cfg = LaneRewardConfig(last_hit_shaping=False, subtract_ambient_gold=False)
    assert cfg.weights.spend == 0.0
    r = ZeroSumLaneReward(cfg=cfg)
    spend = 0.0
    info = None
    for f in _purchase_frames(cost=1000.0):
        _rew, info = r.step(f)
        spend += info["terms"][C.TEAM_BLUE].get("spend", 0.0)
    assert spend == 0.0, f"a scripted purchase paid {spend}"
    assert info["spent_gold"][C.TEAM_BLUE] == pytest.approx(1000.0), (
        "the detector must still SEE the drop, or a zero term is unreadable"
    )


def test_hoarding_earns_nothing_extra():
    """Neither having gold nor (now) converting it is worth anything."""
    cfg = LaneRewardConfig(last_hit_shaping=False, subtract_ambient_gold=False)
    r = ZeroSumLaneReward(cfg=cfg)
    spend = 0.0
    for f in _idle_lane_frames(n=30):
        _rew, info = r.step(f)
        spend += info["terms"][C.TEAM_BLUE].get("spend", 0.0)
    assert spend == 0.0, f"a wallet that only grew paid a spend bonus of {spend}"


def test_dying_is_never_profitable():
    """The bug that closed the `spend` term: AutoBuy shops on the RESPAWN.

    ``LanerlHooks.AutoBuyUndriven`` is fountain-gated and
    ``ShopState.BuyOutOnRespawn`` empties the bank on the walk back from a
    death, so the biggest wallet drop of an episode lands one step after the
    agent got killed.  At the old ``spend = 0.002`` a 1500g buy-out paid
    +3.00 against the -0.50 the kill/death pair charges for dying: the reward
    was strictly positive for feeding.

    Asserted on the agent's OWN raw reward (not the zero-sum difference) so it
    cannot be rescued by the opponent's term.
    """
    a = C.TOP_OUTER_TURRET[C.TEAM_BLUE]

    def f(t_ms, blue_hp, blue_gold):
        return make_frame(t_ms, [
            unit(1001, "champion", C.TEAM_BLUE, a[0], a[1],
                 hp=blue_hp, mhp=671, gold=blue_gold, xp=1000.0, lvl=6),
            unit(1002, "champion", C.TEAM_RED, 12000, 12000,
                 hp=671, mhp=671, gold=1500.0, xp=1000.0, lvl=6),
        ])

    cfg = LaneRewardConfig(last_hit_shaping=False, subtract_ambient_gold=False)
    r = ZeroSumLaneReward(cfg=cfg)
    own = 0.0
    # alive at 30% -> dead -> respawns at full hp and the shop empties 1500g.
    for t, hp, gold in ((0, 201, 1500.0), (100, 0, 1500.0), (15_000, 671, 0.0)):
        _rew, info = r.step(f(t, hp, gold))
        own += sum(info["terms"][C.TEAM_BLUE].values())
    assert info["spent_gold"][C.TEAM_BLUE] == pytest.approx(1500.0), (
        "the scenario must actually contain the buy-out, or it proves nothing"
    )
    assert own < 0.0, f"dying and then shopping was worth {own:+.3f}"
    # And specifically: the death is not cancelled by the shopping trip.
    assert own <= -1.0, f"a death should still hurt; got {own:+.3f}"


def test_a_respawn_is_not_a_heal():
    """The refund that made feeding profitable at the PUBLISHED weights too.

    `phi` is a potential on the hp LEVEL, so 0 -> full on respawn pays
    `w * phi(1) = w` while the death only charged `w * phi(h_at_death)`.  The
    difference is a rebate, and below ~55% hp it exceeds the -0.5 that the
    kill/death pair charges: at w = 2.0, dying at 30% came out at +0.44.
    Swept across hp so a future edit cannot fix one case and miss the rest.
    """
    a = C.TOP_OUTER_TURRET[C.TEAM_BLUE]

    def f(t_ms, blue_hp_frac):
        return make_frame(t_ms, [
            unit(1001, "champion", C.TEAM_BLUE, a[0], a[1],
                 hp=round(671 * blue_hp_frac), mhp=671, gold=600.0, xp=1000.0, lvl=6),
            unit(1002, "champion", C.TEAM_RED, 12000, 12000,
                 hp=671, mhp=671, gold=600.0, xp=1000.0, lvl=6),
        ])

    for h in (1.0, 0.6, 0.3, 0.05):
        cfg = LaneRewardConfig(
            last_hit_shaping=False, subtract_ambient_gold=False,
            zero_sum_alpha_start=1.0, zero_sum_alpha_end=1.0,
        )
        r = ZeroSumLaneReward(cfg=cfg)
        total = 0.0
        for t, hp in ((0, h), (100, 0.0), (15_000, 1.0)):
            rew, _info = r.step(f(t, hp))
            total += rew[C.TEAM_BLUE]
        assert total < 0.0, f"dying at {h:.0%} hp and respawning was worth {total:+.3f}"
        # The hp cost of a death is phi(h_at_death), never refunded, so a
        # death from FULL must cost strictly more than one from nearly dead.
        assert total <= -0.5, f"dying at {h:.0%} hp cost only {total:+.3f}"


def test_the_hp_cost_of_a_death_is_paid_once_whatever_the_path():
    """Going full -> dead costs `w` in total however slowly it happened.

    That is the property that lets the respawn suppression be a flat cut
    rather than a special case: the potential charges for the hp on the way
    down, so an agent cannot make dying cheaper by taking the damage first.
    """
    w = RewardWeights().hp_point
    straight = w * hp_potential_delta(0.0, 1.0)
    stepwise = sum(
        w * hp_potential_delta(b, a)
        for a, b in zip((1.0, 0.8, 0.5, 0.2), (0.8, 0.5, 0.2, 0.0))
    )
    assert straight == pytest.approx(stepwise, abs=1e-12)
    assert straight == pytest.approx(-w, abs=1e-12)


# --------------------------------------------------------------------------
# The weight balance
#
# The tests above check that each term is computed correctly.  These check the
# thing that was actually wrong: their RELATIVE SIZE.  Measured over a real
# 600 s episode (46 CS, 3 deaths) at the published weights, `exp` -- the term
# the policy least controls -- was the largest in the table, at +34.82 against
# `last_hit` +23.00, and carried 14.27 of the ~29 total |r| the episode moved.
# Every assertion below is two-sided on purpose: a future edit that re-inflates
# `exp`, or that over-corrects by drowning everything in `last_hit`, both fail.
# --------------------------------------------------------------------------

#: Server content, `LeagueSandbox-Default/Stats/Blue_Minion_*/*.json`.  Not in
#: `constants.py` because nothing else needs them; quoted here so the
#: arithmetic in the module docstring is checkable rather than asserted.
MELEE_XP, MELEE_GOLD = 77.0, 20.0
CASTER_XP, CASTER_GOLD = 51.0, 10.0
#: 3 melee + 3 caster, and `ai_MinionSpawnDelay` puts a wave in lane every 30 s.
WAVE_XP = 3 * MELEE_XP + 3 * CASTER_XP     # 384
WAVE_PERIOD_S = 30.0

#: The measured episode, as three numbers the weights are applied to.
#: 46 CS; 34.82 / 0.008 = 4353 XP; 5.27 / 0.008 = 659 net gold.
MEASURED_CS, MEASURED_XP, MEASURED_NET_GOLD = 46, 4352.5, 658.75
MEASURED_TOWER = 1.04
MEASURED_DEATHS = 3


def test_the_free_xp_radius_really_is_much_bigger_than_the_last_hit_reach():
    """Why `exp` is an ambient term: it is collected 8x further out.

    `AttackableUnit.Die` gives `ExpGivenOnDeath` to every enemy champion within
    `ai_ExpRadius2 = 1600` of the corpse, split among them and regardless of
    who killed it; gold goes through `Champion.OnKill`, to the killer only.
    """
    exp_radius = 1600.0                      # Maps/Map1/Constants.json
    reach = C.AA_RANGE_GAREN + C.TARGET_RADIUS["minion"] + C.AA_RANGE_EPS
    assert reach == pytest.approx(190.0)
    assert exp_radius / reach > 8.0


def test_a_minion_pays_mostly_for_the_last_hit_not_for_standing_there():
    """The headline ratio.  At the published weights it was 52 / 48.

    Per melee minion: being inside the 1600-unit XP radius pays
    `exp * 77`, landing the hit pays `last_hit + money * 20`.
    """
    w = RewardWeights()
    for xp, gold in ((MELEE_XP, MELEE_GOLD), (CASTER_XP, CASTER_GOLD)):
        free = w.exp * xp
        earned = w.last_hit + w.money * gold
        share = free / (free + earned)
        assert share < 0.10, (
            f"{share:.1%} of a minion's payout needs no skill "
            f"(free={free:.3f}, earned={earned:.3f}); it was 48% at exp=0.008"
        )
    # Two-sided: `exp` must not be deleted either -- it is the only dense term
    # that notices the agent leaving lane, being zoned, or lying dead.
    assert w.exp > 0.0
    # Pinned exactly, because this is the number the docstring argues from.
    assert w.exp * MELEE_XP == pytest.approx(0.077)
    assert w.last_hit + w.money * MELEE_GOLD == pytest.approx(1.16)


def test_one_last_hit_outvalues_a_minute_of_standing_next_to_the_wave():
    """The same thing as a rate, which is how the agent experiences it.

    A wave is 384 XP every 30 s = 12.8 XP/s of ambient income for a champion
    that is merely present.  At the published weights one melee last hit was
    worth 6.4 seconds of that, which is why the agent farmed like it did.
    """
    w = RewardWeights()
    ambient_per_s = w.exp * WAVE_XP / WAVE_PERIOD_S
    melee_last_hit = w.last_hit + w.money * MELEE_GOLD
    seconds = melee_last_hit / ambient_per_s
    assert seconds > 60.0, f"one last hit is worth only {seconds:.1f} s of loitering"
    assert seconds == pytest.approx(90.6, rel=0.02)

    published = RewardWeights(exp=0.008, last_hit=0.5)
    was = (published.last_hit + published.money * MELEE_GOLD) / (
        published.exp * WAVE_XP / WAVE_PERIOD_S
    )
    assert was == pytest.approx(6.4, rel=0.02), "the before-number in the docstring"


def _measured_episode_totals(w: RewardWeights):
    """The measured 600 s episode replayed through an arbitrary weight table."""
    return {
        "last_hit": MEASURED_CS * w.last_hit,
        "exp": MEASURED_XP * w.exp,
        "money": MEASURED_NET_GOLD * w.money,
        "tower_hp": MEASURED_TOWER,
        "death": MEASURED_DEATHS * w.death,
    }


def test_last_hit_dominates_the_measured_episode():
    """Composition of a real 600 s / 46 CS episode, before and after.

    before: exp 34.82 (54%), last_hit 23.00 (36%), money 5.27, tower 1.04
    after:  last_hit 46.00 (81%), exp 4.35 (8%), money 5.27, tower 1.04
    """
    after = _measured_episode_totals(RewardWeights())
    positive = sum(v for v in after.values() if v > 0)
    assert after["last_hit"] / positive > 0.60, (
        f"farming is only {after['last_hit'] / positive:.0%} of the return"
    )
    assert after["exp"] / positive < 0.15, (
        f"ambient XP is still {after['exp'] / positive:.0%} of the return"
    )
    assert after["last_hit"] > 3.0 * after["exp"]

    before = _measured_episode_totals(RewardWeights(exp=0.008, last_hit=0.5))
    before_pos = sum(v for v in before.values() if v > 0)
    assert before["exp"] > before["last_hit"], "the bug being fixed"
    assert before["exp"] / before_pos > 0.5

    # Scale is preserved to within 15%, so no PPO coefficient moves with this.
    assert 0.85 < positive / before_pos < 1.15, (
        f"the episode return changed scale by {positive / before_pos:.2f}x"
    )


def test_a_winning_trade_is_worth_a_fraction_of_a_cs():
    """`hp_point` against `last_hit`, run through the real reward.

    A clean level-6 trade -- 25% off them, 15% off me, both from full -- at
    the published hp_point = 2.0 paid 0.103, i.e. 9% of a last hit under the
    new `last_hit`: any trade costing one CS was a loss, and the measured
    episode shows an agent that never traded.  Bounded ABOVE as well: at
    hp_point = 8 a health bar is worth 6.9 CS and the agent learns to poke
    instead of farm, which is the opposite failure.
    """
    cfg = LaneRewardConfig(
        last_hit_shaping=False, subtract_ambient_gold=False,
        zero_sum_alpha_start=1.0, zero_sum_alpha_end=1.0,
    )
    r = ZeroSumLaneReward(cfg=cfg)
    r.step(top_lane_scenario(t_ms=200_000, n_minions=0))
    rew, _info = r.step(
        top_lane_scenario(t_ms=200_100, n_minions=0, blue_hp_frac=0.85, red_hp_frac=0.75)
    )
    trade = rew[C.TEAM_BLUE]
    w = RewardWeights()
    melee_last_hit = w.last_hit + w.money * MELEE_GOLD
    assert trade > 0.0, "winning a trade must pay something"
    assert trade == pytest.approx(0.0517 * w.hp_point, rel=0.02)
    ratio = trade / melee_last_hit
    assert 0.12 < ratio < 0.35, (
        f"a shallow trade is worth {ratio:.2f} CS; below 0.12 nobody trades, "
        f"above 0.35 nobody farms"
    )

    # And the quartic still does its job: the SAME 25% taken off a champion
    # already at 40% -- the chunk that sets up a kill -- is worth five times
    # as much, which is the entire reason for the quartic.
    chunk = w.hp_point * (hp_potential(0.40) - hp_potential(0.15))
    assert chunk / trade == pytest.approx(6.2, rel=0.05)
    assert chunk == pytest.approx(1.285, rel=0.02)
    assert 0.9 < chunk / melee_last_hit < 1.5, (
        f"the kill-setup chunk is worth {chunk / melee_last_hit:.2f} CS"
    )

    # A whole health bar, i.e. what killing from full is worth on this term.
    assert w.hp_point / melee_last_hit == pytest.approx(3.45, rel=0.02)


def test_the_terms_the_policy_does_not_control_stay_small_together():
    """The invariant behind all of the above, stated once.

    Three things pay out without the policy doing anything: ambient XP
    (`exp`), the ambient-gold subtraction residual, and a scripted purchase
    (`spend`).  Summed over the measured episode they must stay well under
    the term that requires a skill.  Stated as a SUM, not per-term, so the
    budget cannot be re-spread across three small inflations.
    """
    w = RewardWeights()
    after = _measured_episode_totals(w)
    # The ambient-gold residual nets to zero over a window by construction
    # (test_money_term_ignores_the_ambient_trickle); `spend` is a named zero.
    assert w.spend == 0.0
    uncontrolled = after["exp"] + 0.0 + w.spend * 1500.0
    assert uncontrolled < 0.25 * after["last_hit"], (
        f"uncontrolled income is {uncontrolled:.2f} against "
        f"{after['last_hit']:.2f} of farming"
    )
