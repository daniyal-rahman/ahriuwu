"""Deaths must be reported WITH a location.

A run that prints ``deaths=4`` tells you nothing about the failure. Four
deaths under the enemy turret and four deaths to the wave in your own half
are the same number and completely different bugs, and the only way we could
tell them apart was to re-run games under a side script that recorded 18,000
positions -- which meant the map existed only for games somebody thought to
reproduce, and never for the games that actually happened during training.

Three floats per death is cheap enough to leave on forever.

The failure mode this guards is silence: ``death_pos`` is read out of a dict
by key in two places, so a rename or a changed team-key type degrades to an
empty list rather than an error, and the metric just quietly stops appearing.
"""
from __future__ import annotations

import math

import pytest

from lanerl_rl import constants as C
from lanerl_rl.reward import LaneRewardConfig, ZeroSumLaneReward
from lanerl_rl.scenarios import make_frame, unit

BLUE, RED = C.TEAM_BLUE, C.TEAM_RED
BX, BY = 2000.0, 11500.0
RX, RY = 3600.0, 13200.0


def _frame(t_ms, blue_alive=True, blue_xy=(BX, BY)):
    # Unit.alive is DERIVED from hp > 0, not a field; setting hp is the only
    # way to kill one, and passing alive= silently used to be a TypeError.
    return make_frame(t_ms, [
        unit(1, "champion", BLUE, blue_xy[0], blue_xy[1],
             hp=600.0 if blue_alive else 0.0, mhp=600.0),
        unit(2, "champion", RED, RX, RY, hp=600.0, mhp=600.0),
    ])


def test_a_death_reports_where_it_happened():
    r = ZeroSumLaneReward(cfg=LaneRewardConfig())
    r.step(_frame(1000))
    _, info = r.step(_frame(2000, blue_alive=False, blue_xy=(3500.0, 13100.0)))

    assert info["died"][BLUE] is True
    assert BLUE in info["death_pos"], (
        "a death was resolved but carried no position -- the dict key or the "
        "team key type has drifted, and this degrades to silence"
    )
    t, x, y = info["death_pos"][BLUE]
    assert (t, x, y) == (2000.0, 3500.0, 13100.0)


def test_no_death_reports_no_position():
    """An entry on a live tick would put a phantom marker on every map."""
    r = ZeroSumLaneReward(cfg=LaneRewardConfig())
    r.step(_frame(1000))
    _, info = r.step(_frame(2000))
    assert info["death_pos"] == {}


def test_the_survivor_is_not_given_a_death_position():
    r = ZeroSumLaneReward(cfg=LaneRewardConfig())
    r.step(_frame(1000))
    _, info = r.step(_frame(2000, blue_alive=False))
    assert RED not in info["death_pos"]


def test_the_position_is_the_corpse_not_the_respawn():
    """The tick alive flips is the last one with the champion where it fell.
    Reading a tick later would put every death marker on the fountain, which
    is exactly the wrong answer and a plausible-looking one."""
    r = ZeroSumLaneReward(cfg=LaneRewardConfig())
    r.step(_frame(1000))
    _, info = r.step(_frame(2000, blue_alive=False, blue_xy=(3500.0, 13100.0)))
    fountain = C.NEXUS_POSITION[BLUE]
    _, x, y = info["death_pos"][BLUE]
    assert math.hypot(x - fountain[0], y - fountain[1]) > 1000.0, (
        "the death was recorded at the fountain -- it is being read after the "
        "respawn teleport, so every map will show deaths in the base"
    )
