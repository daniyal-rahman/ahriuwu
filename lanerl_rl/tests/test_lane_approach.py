"""The potential that pays for WALKING TO LANE.

Why this term exists at all is a measurement, not a theory. Run
``rl-screen-0914`` was the first from-scratch run in this project's history --
every earlier run that farmed was initialised from the BC policy, which
supplied the walk to lane as a prior. Across 164 episodes it produced:

    gametime    mean CS    mean level
      0-210 s     0.00        1.00
        300 s     0.00        1.50
        570 s     0.00        1.41

A champion that reaches lane is level 2 off the first wave at ~90 s. Level
1.00 flat through 210 s is a champion that never left its own base, and 0.00
CS across every bucket is what "no reward gradient anywhere" looks like from
the outside.

``lane_presence`` was supposed to cover this and cannot: it is an indicator
that pays only once the agent is ALREADY inside the corridor, so from the
fountain its gradient is exactly zero in every direction. The corridor is
6,835 units from the fountain (measured, not assumed -- an earlier note in
this repo carried 11,866, which is the distance to a different anchor) and a
decision lasts 33 ms (~11 units of travel), so re-drawing a direction at
30 Hz is a random walk covering ~11*sqrt(9000) ~ 1.1k units over a 300 s
episode. It never arrives, so it never sees a minion, so there is
nothing to learn from. The two terms are complementary, not redundant: the
potential carries the agent to the corridor, the indicator holds it there.

The tests below are the ones that would catch a plausible mistake:

* it is a genuine Ng-et-al. potential, so it telescopes to zero around any
  closed loop -- the property that makes it safe to add at all, since it
  changes which policies are FOUND and never which policy is OPTIMAL;
* it saturates exactly where ``lane_presence`` starts paying, so the two hand
  off instead of double-counting;
* distance is to the corridor RECTANGLE, not to the infinite axis, so sitting
  in your own base at n = 0 is correctly "far";
* dying is still never profitable, which is the one way this term could do
  real damage (a death teleports the champion to the fountain, which is a
  step DOWN in the potential).
"""
from __future__ import annotations

import math

import pytest

from lanerl_rl import constants as C
from lanerl_rl.frame import LaneFrame
from lanerl_rl.reward import lane_approach_potential
from lanerl_rl.scenarios import unit

TEAM = C.TEAM_BLUE
ENEMY = C.TEAM_RED


@pytest.fixture()
def lane() -> LaneFrame:
    return LaneFrame(
        C.TOP_OUTER_TURRET[TEAM],
        C.TOP_OUTER_TURRET[ENEMY],
        C.NEXUS_POSITION[TEAM],
    )


def _at(lane: LaneFrame, s: float, n: float):
    """A champion at lane coordinates (s, n), given back in world space."""
    ox, oy = lane.origin
    ax, ay = lane.axis
    nx, ny = lane.normal
    return unit(1, "champion", TEAM, ox + ax * s + nx * n, oy + ay * s + ny * n)


def _phi(lane: LaneFrame, u, per_1000: float = 0.07) -> float:
    return lane_approach_potential(
        lane, u, per_1000=per_1000, corridor=C.LANE_HALF_WIDTH
    )


# -- the hand-off with lane_presence ---------------------------------------


def test_it_is_zero_exactly_where_lane_presence_starts_paying(lane):
    """Saturation point == the indicator's corridor, or they double-count."""
    mid = lane.length / 2
    assert _phi(lane, _at(lane, mid, 0.0)) == 0.0
    assert _phi(lane, _at(lane, mid, C.LANE_HALF_WIDTH - 1.0)) == 0.0
    # and immediately outside it is not zero
    assert _phi(lane, _at(lane, mid, C.LANE_HALF_WIDTH + 500.0)) < 0.0


def test_being_far_off_axis_costs_the_perpendicular_excess_only(lane):
    mid = lane.length / 2
    off = 3000.0
    got = _phi(lane, _at(lane, mid, off))
    want = -(0.07 / 1000.0) * (off - C.LANE_HALF_WIDTH)
    assert got == pytest.approx(want, rel=1e-9)


def test_sitting_in_your_own_base_is_far_even_at_n_zero(lane):
    """Distance is to the RECTANGLE. Measuring to the infinite axis instead
    would score the fountain as 'in lane' because the lane axis, extended
    backwards, runs straight through it -- and the whole term would be dead
    for exactly the state it exists to fix."""
    base = _phi(lane, _at(lane, -6000.0, 0.0))
    assert base < 0.0
    assert base == pytest.approx(-(0.07 / 1000.0) * (6000.0 - C.LANE_HALF_WIDTH))


def test_the_gradient_points_at_the_lane_the_whole_way_in(lane):
    """Monotone, so every step toward lane pays. A non-monotone potential
    would have a basin the agent could get stuck in short of the corridor."""
    phis = [_phi(lane, _at(lane, -s, 0.0)) for s in range(8000, 0, -500)]
    assert all(b >= a for a, b in zip(phis, phis[1:])), phis
    # ...and STRICTLY increasing for every step still outside the corridor,
    # where a flat stretch would be a basin to get stuck in. Inside, it is
    # flat at zero by design -- that is the hand-off to lane_presence.
    outside = [p for p in phis if p < 0.0]
    assert len(outside) >= 10
    assert all(b > a for a, b in zip(outside, outside[1:])), outside


def test_the_whole_walk_pays_about_half_a_last_hit(lane):
    """Scale check. Big enough to be a signal against per-step noise, small
    enough that the death transient (see below) stays under the 1.0 death
    weight."""
    spawn = _phi(lane, unit(1, "champion", TEAM, *C.NEXUS_POSITION[TEAM]))
    arrived = _phi(lane, _at(lane, lane.length / 2, 0.0))
    assert arrived == 0.0
    # 6,835 units from the fountain to the corridor, measured, times
    # 0.07/1000. Pinned tightly: this is the number the death transient below
    # is sized against, so it should not drift silently.
    assert (arrived - spawn) == pytest.approx(0.478, abs=0.01), arrived - spawn


# -- the property that makes it safe ---------------------------------------


def test_it_telescopes_to_zero_around_a_closed_loop(lane):
    """The Ng-et-al. property, at gamma = 1: sum of gamma*phi(s') - phi(s)
    around any loop is zero, so no cycle can be farmed for reward. This is
    what separates a potential from a 'closer than last tick' bonus, which
    would pay an agent to oscillate toward and away from the lane forever."""
    loop = [(-6000.0, 0.0), (-3000.0, 2000.0), (0.0, 4000.0),
            (lane.length / 2, 0.0), (-2000.0, -3000.0), (-6000.0, 0.0)]
    total = 0.0
    for (s0, n0), (s1, n1) in zip(loop, loop[1:]):
        total += _phi(lane, _at(lane, s1, n1)) - _phi(lane, _at(lane, s0, n0))
    assert total == pytest.approx(0.0, abs=1e-12)


def test_a_death_costs_the_walk_back_and_no_more(lane):
    """A death teleports the champion to the fountain: a real step DOWN in
    phi. It is recovered by walking back, so the SUM over the death-and-return
    cycle is zero and dying stays strictly unprofitable -- but the transient
    must stay under the 1.0 death weight, or this term would quietly become
    the dominant death penalty."""
    in_lane = _phi(lane, _at(lane, lane.length / 2, 0.0))
    fountain = _phi(lane, unit(1, "champion", TEAM, *C.NEXUS_POSITION[TEAM]))
    transient = fountain - in_lane
    assert transient < 0.0
    assert abs(transient) < 1.0, (
        f"the death transient is {transient:.3f}, which rivals the -1.0 death "
        f"weight -- lower RewardWeights.lane_approach"
    )
    assert (in_lane - fountain) + transient == pytest.approx(0.0, abs=1e-12)


def test_a_missing_champion_is_not_an_error(lane):
    """Fog, or a frame taken while dead. Raising here would kill the episode."""
    assert _phi(lane, None) == 0.0


def test_zero_weight_disables_it_completely(lane):
    """The ablation has to actually ablate."""
    assert _phi(lane, _at(lane, -6000.0, 0.0), per_1000=0.0) == 0.0
