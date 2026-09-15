"""Side symmetry: the harness must not turn which side you played into skill.

A policy playing ITSELF should score 0.5 overall. If it does not, something
in the pipeline advantages one side, and every self-play rating and every
league Elo is measuring that asymmetry rather than skill.

I expected top lane to be materially asymmetric and to need the scripted
bot's own blue/red split as the reference. Measured, it is not: the two
fountains sit 10,570 and 10,574 units from the lane midpoint, a difference
of 0.04%. So 0.5 is a fair null after all. (Geometry is not everything --
terrain and brush are not measured -- but the big obvious asymmetry is
absent.)

WHAT WOULD MAKE THIS FAIL
-------------------------
* the lane-frame reflection applied to one side and not the other (already
  guarded by the mirror tests, but those check the observation, not outcomes)
* a reward term with the wrong sign for red
* CS or deaths attributed to the wrong team -- this has happened here: the
  CS@10 readout was once a mean over every team in the frame, so a strong
  scripted bot read as a strong agent
* an action decoded in world space for one side and lane space for the other

WHY IT IS STRUCTURED AS SCORE ARITHMETIC
----------------------------------------
The expensive version plays hundreds of real games from both sides. That is
worth doing before a headline claim, but it is not a test -- it takes hours
and its answer is a confidence interval. What IS testable cheaply is that
the SCORING is side-symmetric: given mirrored outcomes, the harness must
produce mirrored scores. A harness that cannot do that arithmetic correctly
will never produce a trustworthy number however many games it plays.
"""
from __future__ import annotations

import pytest

from lanerl_rl import constants as C
from lanerl_train.anchor_eval import score_for_deaths, score_for_reason
from lanerl_train.eval import MatchRecord, PairTable, bradley_terry


def test_the_score_is_antisymmetric_under_swapping_the_sides():
    """score(a,b) + score(b,a) == 1 for every outcome, exactly.

    If this drifts, playing the same matchup from both sides does not
    cancel, and a seed-balanced evaluation is biased rather than fair.
    """
    for da in range(4):
        for db in range(4):
            s = score_for_deaths(da, db)
            swapped = score_for_deaths(db, da)
            assert s + swapped == pytest.approx(1.0), (da, db, s, swapped)


def test_the_end_reason_scorer_is_also_antisymmetric():
    for winner, loser in ((C.TEAM_BLUE, C.TEAM_RED), (C.TEAM_RED, C.TEAM_BLUE)):
        reason = f"death_team_{loser}"
        assert score_for_reason(reason, winner) + score_for_reason(reason, loser) \
            == pytest.approx(1.0), reason


def test_a_policy_that_plays_itself_from_both_sides_rates_level():
    """Same policy, every seed played from both sides: rating spread must be
    zero. A nonzero spread here is the harness inventing skill out of side."""
    table = PairTable()
    # 40 seeds, each played both ways, with a deliberately LOPSIDED per-seed
    # outcome: blue wins every seed when playing as "a", red wins every seed
    # when playing as "a". Balanced pairing must cancel it exactly.
    for _ in range(40):
        table.add(MatchRecord(agent_a="a", agent_b="b", score_a=1.0))
        table.add(MatchRecord(agent_a="b", agent_b="a", score_a=1.0))
    ratings = bradley_terry(table)
    spread = abs(ratings["a"] - ratings["b"])
    assert spread < 1e-6, (
        f"playing both sides did not cancel: spread {spread:.4f}. Either the "
        f"pairing is not balanced or the fit is side-sensitive, and every "
        f"self-play Elo is reporting side advantage as skill."
    )


def test_an_unbalanced_pairing_does_NOT_cancel():
    """The guard on the test above. If lopsided results still rated level,
    the previous test would pass for the wrong reason."""
    table = PairTable()
    for _ in range(40):
        table.add(MatchRecord(agent_a="a", agent_b="b", score_a=1.0))
    ratings = bradley_terry(table)
    assert ratings["a"] - ratings["b"] > 100.0, (
        "40-0 produced no rating separation, so the fit cannot detect a real "
        "difference and the cancellation test proves nothing"
    )


def test_the_lane_is_geometrically_symmetric_enough_that_0_5_is_the_null():
    """Measured, because I assumed the opposite and was wrong.

    I wrote this file expecting top lane to be materially asymmetric, so
    that a real side-swap check would have to compare against the scripted
    bot's own blue/red split rather than against 0.5. The geometry says
    otherwise:

        fountain -> lane midpoint    blue 10570   red 10574   diff    4.1
        fountain -> own outer turret blue  8812   red  8871   diff   59.6

    4 units in 10,570 is 0.04%. So 0.5 IS the right null for a side-swap
    check, and the protocol is simply: play every seed from both sides, and
    expect ~0.5 overall.

    HONEST LIMIT: equal distances are not full symmetry. Terrain, brush
    placement and which side the wave meets first are not measured here, and
    any of them could favour a side in a way this does not see. What this
    rules out is the big, obvious geometric asymmetry -- enough to make 0.5
    a reasonable null rather than a guess, not enough to make a measured
    deviation automatically a bug.
    """
    bf, rf = C.NEXUS_POSITION[C.TEAM_BLUE], C.NEXUS_POSITION[C.TEAM_RED]
    bt, rt = C.TOP_OUTER_TURRET[C.TEAM_BLUE], C.TOP_OUTER_TURRET[C.TEAM_RED]
    mid = tuple((a + b) / 2 for a, b in zip(bt, rt))

    def dist(p, q):
        return sum((a - b) ** 2 for a, b in zip(p, q)) ** 0.5

    to_mid = abs(dist(bf, mid) - dist(rf, mid))
    to_turret = abs(dist(bf, bt) - dist(rf, rt))
    assert to_mid < 50.0, (
        f"fountain-to-midpoint differs by {to_mid:.0f} units between sides. "
        f"If that grows, 0.5 stops being the right null for a side-swap "
        f"check and the scripted bot's own split has to become the reference."
    )
    assert to_turret < 200.0, f"fountain-to-own-turret differs by {to_turret:.0f}"
