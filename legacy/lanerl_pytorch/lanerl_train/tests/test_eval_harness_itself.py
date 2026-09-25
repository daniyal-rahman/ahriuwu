"""Test the measuring instrument, not just the thing measured.

Three separate times this project reported a confident, smooth, entirely
fake number: an eval that scored an argmax policy which never moved, a CS
curve that was servers wearing out, and instances silently scoring 0 for
champions nobody was driving. In each case the harness was trusted and the
policy was blamed.

So: the eval harness gets tested against answers that are known a priori,
with no reference to any policy's actual skill.

    identical policies      -> ~0.5
    a policy that cannot lose -> 1.0, and one that cannot win -> 0.0
    a deathless lane        -> exactly 0.5, not a coin flip
    the Elo fit             -> orders three policies it is given in order

These are cheap, pure-Python, and they are the checks that would have made
the earlier fakes impossible to report.
"""
from __future__ import annotations

import math

import pytest

from lanerl_train.anchor_eval import score_for_deaths, score_for_reason
from lanerl_rl import constants as C


# -- the score function itself ---------------------------------------------


def test_identical_policies_score_a_draw():
    """A mirror match is 0.5 by construction, whatever the death count.

    If this ever returns something else, every self-play rating is measuring
    an asymmetry in the harness rather than in the policies.
    """
    for n in (0, 1, 3, 7):
        assert score_for_deaths(n, n) == 0.5, n


def test_a_policy_that_only_dies_scores_zero_and_vice_versa():
    assert score_for_deaths(3, 0) == 0.0
    assert score_for_deaths(0, 3) == 1.0
    # ...and the ordering is monotone in the differential, not just the ends
    worse = score_for_deaths(4, 1)
    same = score_for_deaths(2, 2)
    better = score_for_deaths(1, 4)
    assert worse < same < better, (worse, same, better)


def test_a_deathless_lane_is_a_draw_not_a_coin_flip():
    """The common case. If a deathless game scored anything but 0.5 the
    ladder would drift on games where nothing happened."""
    assert score_for_deaths(0, 0) == 0.5


def test_the_two_scorers_agree_on_a_game_that_ended_at_the_first_death():
    """score_for_reason (ends at first death) and score_for_deaths (plays
    out) must not disagree on the case they overlap on, or the ladder shifts
    silently the day end_on_death is toggled -- which has happened."""
    assert score_for_reason(f"death_team_{C.TEAM_RED}", C.TEAM_BLUE) == \
        score_for_deaths(0, 1)
    assert score_for_reason(f"death_team_{C.TEAM_BLUE}", C.TEAM_BLUE) == \
        score_for_deaths(1, 0)


# -- the rating fit --------------------------------------------------------


def test_bradley_terry_orders_policies_it_is_given_in_order():
    """A rating fit that cannot recover a planted ordering cannot detect a
    real one. Uses the project's own fitter, not a reimplementation."""
    from lanerl_train.eval import MatchRecord, PairTable, bradley_terry

    # strong beats middle beats weak, consistently
    table = PairTable()
    for _ in range(20):
        for a, b in (("strong", "middle"), ("middle", "weak"), ("strong", "weak")):
            table.add(MatchRecord(agent_a=a, agent_b=b, score_a=1.0))
    ratings = bradley_terry(table)
    assert ratings["strong"] > ratings["middle"] > ratings["weak"], ratings


def test_bradley_terry_calls_a_symmetric_league_level():
    """Everyone drawing with everyone must not produce a spread. A fitter
    that invents an ordering out of draws would rank noise."""
    from lanerl_train.eval import MatchRecord, PairTable, bradley_terry

    table = PairTable()
    for a, b in (("a", "b"), ("b", "c"), ("a", "c")):
        for _ in range(20):
            table.add(MatchRecord(agent_a=a, agent_b=b, score_a=0.5))
    ratings = bradley_terry(table)
    spread = max(ratings.values()) - min(ratings.values())
    assert spread < 1e-6, f"draws produced a {spread:.3f} rating spread: {ratings}"
