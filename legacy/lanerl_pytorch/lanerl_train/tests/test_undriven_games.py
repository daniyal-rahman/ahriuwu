"""A champion nobody drove is not a data point.

``rl-screen-bc-0914c`` reported this CS@10 curve against scripted_bronze::

    update    400    800   1200   1600   2000
    CS@10    22.6   12.3   11.6    9.0    3.2

and it read exactly like a policy collapsing away from its 36.0 CS BC prior.
Broken down by the anchor instance the game ran on, with the SAME policy in
every game:

    update 400    inst0 32.7   inst1 31.3   inst2 13.4   inst3 11.8
    update 1200   inst0 26.7   inst1  7.4   inst2 10.9   inst3  0.0 (0/8)
    update 2000   inst0  0.0   inst1 12.4   inst2  0.0   inst3  0.0

Instances drop to exactly zero and stay there. The server's own rows for a
dead one, at ten minutes::

    brian8544 team=100 cs=0 gold=603 lvl=1 hp=754/754 deaths=0

Level 1, untouched, 603 gold (starting gold plus the ambient trickle): a
champion that never left the fountain. It hit the scripted bot as often as
the agent, so it is not the policy. Nothing errored -- servers alive at
6.65x, all 18,001 decisions sent, no NetId complaint, no instance death, no
restart. The zeros simply went into the mean.

So the rule: if a champion never got UNDRIVEN_MOVE_EPS from where it started
over an entire game, that game's CS@10 is None, not 0, and the discard is
logged loudly. A wrong number that looks like a trend is worse than a gap.
"""
from __future__ import annotations

import math

import pytest

from lanerl_train.anchor_eval import UNDRIVEN_MIN_DECISIONS, UNDRIVEN_MOVE_EPS


def test_the_threshold_separates_bad_play_from_no_play():
    """6,835 units is the walk from the fountain to the lane corridor, so any
    champion that played at all clears this by more than an order of
    magnitude. The threshold only has to catch a champion sitting still."""
    assert 0.0 < UNDRIVEN_MOVE_EPS < 1000.0
    assert UNDRIVEN_MOVE_EPS < 6835.0 / 10


def test_an_episode_result_can_carry_the_discard_reason():
    from lanerl_train.run import EpisodeResult

    ok = EpisodeResult(agent="a", opponent_id="b",
                       opponent_category="anchor", score=0.5)
    assert ok.undriven is None, "default must be None or every game looks broken"

    bad = EpisodeResult(agent="a", opponent_id="b", opponent_category="anchor",
                        score=0.5, cs_at_10=None, undriven="AGENT (moved 3u)")
    assert bad.cs_at_10 is None, (
        "a discarded game must not carry a CS of 0 -- 0 is a measurement and "
        "this is the absence of one"
    )
    assert "AGENT" in bad.undriven


def test_a_short_game_is_never_judged_idle():
    """The gate that stops the fix becoming the same bug facing the other way.
    A game that ended on an early death, or a 7-step fixture, has not had the
    chance to move -- calling those 'not a measurement' would silently DROP
    real games instead of silently counting fake ones."""
    assert UNDRIVEN_MIN_DECISIONS >= 900, (
        "at 30 Hz this must cover the ~20 s walk from fountain to lane"
    )
    # the wedged games that motivated all this ran the full 18,001 decisions,
    # so the gate is nowhere near them
    assert UNDRIVEN_MIN_DECISIONS < 18_001


@pytest.mark.parametrize(
    "moved,expected_undriven",
    [(0.0, True), (3.0, True), (499.0, True), (501.0, False), (6835.0, False)],
)
def test_the_rule_as_applied(moved, expected_undriven):
    assert (moved < UNDRIVEN_MOVE_EPS) is expected_undriven


def test_a_discarded_game_is_excluded_from_the_mean_not_counted_as_zero():
    """The whole point. Averaging the zeros produced a 22.6 -> 3.2 'decline'
    that was three servers going quiet."""
    from lanerl_train.run import EpisodeResult

    played = [EpisodeResult(agent="a", opponent_id="b",
                            opponent_category="anchor", score=0.5, cs_at_10=v)
              for v in (32.0, 31.0, 30.0)]
    idle = [EpisodeResult(agent="a", opponent_id="b", opponent_category="anchor",
                          score=0.5, cs_at_10=None,
                          undriven="AGENT (moved 0u)") for _ in range(9)]
    got = [e.cs_at_10 for e in played + idle if e.cs_at_10 is not None]
    assert len(got) == 3
    assert math.isclose(sum(got) / len(got), 31.0)

    as_zeros = [0.0 if e.cs_at_10 is None else e.cs_at_10 for e in played + idle]
    assert sum(as_zeros) / len(as_zeros) < 8.0, (
        "this is the number the old code reported: 7.75 instead of 31.0"
    )
