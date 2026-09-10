"""Decision rate and discount horizon (correction 1).

Two claims are under test:

1. The server ticks at 60 Hz and emits an observation every
   ``LANERL_STEP_TICKS`` ticks, so the only expressible decision rates are
   ``60 / k``.  50 Hz is 1.2 ticks per decision and does not exist; asking for
   it must be an error, not a silent rounding.
2. The discount is configured as a horizon in SECONDS.  A raw gamma means a
   different amount of game time at every rate, so copying one across a rate
   change silently changes the objective -- which is exactly the mistake that
   made 50 Hz look harmless.
"""

from __future__ import annotations

import pytest

from lanerl_rl import constants as C
from lanerl_rl.ppo import PPOConfig


# --------------------------------------------------------------------------
# Legal rates
# --------------------------------------------------------------------------


def test_the_stack_runs_at_30_hz():
    """The ONE place the chosen rate is pinned.

    30 Hz was chosen over 15 because the eventual pixel shim reads screen
    capture at 30-60 FPS, so decisions faster than the frame rate would have no
    new observation behind them.  Everything else in this file derives from
    C.DECISION_HZ, so changing the rate is this assertion plus constants.py.
    """
    assert C.SERVER_TICK_HZ == 60.0
    assert C.STEP_TICKS == 2
    assert C.DECISION_HZ == 30.0
    assert C.DECISION_DT_MS == pytest.approx(1000.0 / 30.0)


@pytest.mark.parametrize("hz,ticks", [(60, 1), (30, 2), (20, 3), (15, 4), (12, 5), (10, 6), (5, 12)])
def test_legal_rates_are_exactly_60_over_k(hz, ticks):
    assert C.legal_step_ticks(hz) == ticks


@pytest.mark.parametrize("hz", [50, 45, 25, 7, 100, 0.7])
def test_illegal_rates_are_rejected_loudly(hz):
    """50 Hz in particular: 60/50 = 1.2 ticks, which the server cannot do."""
    with pytest.raises(ValueError, match="not expressible"):
        C.legal_step_ticks(hz)


def test_ppo_config_rejects_an_illegal_decision_rate():
    with pytest.raises(ValueError, match="not expressible"):
        PPOConfig(decision_hz=50.0)


# --------------------------------------------------------------------------
# Horizon <-> gamma
# --------------------------------------------------------------------------


def test_gamma_comes_from_a_horizon_in_seconds():
    assert C.gamma_for_horizon(30.0, 15.0) == pytest.approx(0.9977778, abs=1e-7)
    assert C.gamma_for_horizon(45.0, 15.0) == pytest.approx(0.9985185, abs=1e-7)


def test_horizon_and_gamma_round_trip():
    for h in (10.0, 30.0, 45.0, 120.0):
        g = C.gamma_for_horizon(h, C.DECISION_HZ)
        assert C.horizon_for_gamma(g, C.DECISION_HZ) == pytest.approx(h, rel=1e-9)


def test_the_same_horizon_gives_different_gammas_at_different_rates():
    """The point of configuring seconds: 30 s is not one number."""
    g15 = C.gamma_for_horizon(30.0, 15.0)
    g60 = C.gamma_for_horizon(30.0, 60.0)
    assert g60 > g15
    # And the same gamma means a different horizon: this is the trap.
    assert C.horizon_for_gamma(g15, 60.0) == pytest.approx(7.5)


def test_default_ppo_horizon_is_in_the_30_to_45_second_band():
    cfg = PPOConfig()
    assert 30.0 == pytest.approx(cfg.effective_horizon_s(), rel=1e-9)
    # gamma follows the rate: 1 - 1/(H*f). At 30 Hz that is 0.998889.
    assert cfg.gamma == pytest.approx(1.0 - 1.0 / (30.0 * C.DECISION_HZ), abs=1e-9)
    assert cfg.step_ticks == C.STEP_TICKS


def test_explicit_gamma_overrides_the_horizon():
    cfg = PPOConfig(horizon_s=30.0, gamma=0.99)
    assert cfg.gamma == 0.99
    # ... and effective_horizon_s tells you what you actually asked for --
    # 3.3 s at 30 Hz, i.e. an explicit gamma silently buys a MUCH shorter
    # horizon than the 30 s default, which is the point of reporting it.
    assert cfg.effective_horizon_s() == pytest.approx(1.0 / (0.01 * C.DECISION_HZ))


def test_horizon_shorter_than_one_step_is_rejected():
    with pytest.raises(ValueError):
        C.gamma_for_horizon(0.01, 15.0)


# --------------------------------------------------------------------------
# The measurement behind the rate choice
# --------------------------------------------------------------------------


def test_last_hit_is_a_prediction_problem_not_a_reaction_problem():
    """854 ms median window against a 333 ms windup, at a 66.7 ms tick.

    The decision that lands a last hit is taken ~0.3 s before the kill, and the
    window is 12.8 decisions wide at 15 Hz.  Nothing about this task needs
    50 Hz, and AlphaStar measured -129 Elo for doubling the action rate.
    """
    windup_ms = 1000.0 * C.garen_attack_windup(1)
    assert windup_ms == pytest.approx(333.3, abs=1.0)
    decisions_in_window = C.LAST_HIT_WINDOW_MEDIAN_MS / C.DECISION_DT_MS
    assert decisions_in_window > 10.0
    # Even the windup alone is several decisions long, so the agent can react
    # inside it -- there is no reaction-speed argument for a faster rate.
    assert windup_ms / C.DECISION_DT_MS > 4.0
