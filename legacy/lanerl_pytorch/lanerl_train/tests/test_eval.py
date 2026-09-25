"""The metrics that stand in for a win rate that is pinned at 50% by symmetry.

The Elo tests are the load-bearing ones: a rating system that is subtly wrong
still produces a plausible-looking rising curve, which is exactly the artefact
this whole module exists to distinguish from real improvement.  So the fit is
checked against a synthetic tournament whose true ordering *and* true rating
gaps are known in advance.
"""

from __future__ import annotations

import json
import math
import random

import pytest

from lanerl_train.eval import (
    ANCHOR_EPISODE_SHARE,
    CsTracker,
    Evaluator,
    MatchRecord,
    PairTable,
    anchor_episode_budget,
    bradley_terry,
    default_anchors,
    elo_from_ratings,
    expected_score,
)


def table_from(records):
    t = PairTable()
    t.extend(records)
    return t


# -- pairwise bookkeeping --------------------------------------------------


def test_a_match_is_recorded_symmetrically():
    t = table_from([MatchRecord("a", "b", 1.0)])
    assert t.win_rate("a", "b") == 1.0
    assert t.win_rate("b", "a") == 0.0
    assert t.games("a", "b") == t.games("b", "a") == 1


def test_draws_are_half_a_win_to_each_side():
    t = table_from([MatchRecord("a", "b", 0.5)] * 4)
    assert t.win_rate("a", "b") == 0.5 and t.win_rate("b", "a") == 0.5


def test_a_self_match_is_rejected():
    with pytest.raises(ValueError, match="self-match"):
        MatchRecord("a", "a", 0.5)


def test_a_score_outside_zero_one_is_rejected():
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        MatchRecord("a", "b", 1.5)


# -- Bradley-Terry / Elo ---------------------------------------------------


def test_a_transitive_sweep_recovers_the_exact_ordering():
    recs = []
    for _ in range(30):
        recs += [MatchRecord("A", "B", 1.0), MatchRecord("B", "C", 1.0), MatchRecord("A", "C", 1.0)]
    elo = bradley_terry(table_from(recs))
    assert [p for p, _ in elo_from_ratings(elo)] == ["A", "B", "C"]


def test_an_undefeated_player_gets_a_finite_rating():
    """Without the prior this diverges -- and a fresh checkpoint sweeping the
    bronze anchor is a routine occurrence, not an edge case."""
    recs = [MatchRecord("A", "B", 1.0) for _ in range(200)]
    elo = bradley_terry(table_from(recs))
    assert all(math.isfinite(v) for v in elo.values())
    assert elo["A"] > elo["B"]


def test_a_zero_prior_is_refused_rather_than_diverging():
    with pytest.raises(ValueError, match="prior_games"):
        bradley_terry(table_from([MatchRecord("A", "B", 1.0)]), prior_games=0.0)


def test_synthetic_tournament_recovers_the_true_ratings():
    true = {"p0": -400.0, "p1": -200.0, "p2": 0.0, "p3": 200.0, "p4": 400.0}
    rng = random.Random(20260909)
    names = sorted(true)
    recs = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            p = expected_score(true[a], true[b])
            for _ in range(600):
                recs.append(MatchRecord(a, b, 1.0 if rng.random() < p else 0.0))
    elo = bradley_terry(table_from(recs))
    # exact ordering
    assert [p for p, _ in elo_from_ratings(elo)] == ["p4", "p3", "p2", "p1", "p0"]
    # and the gaps, not just the order (true ratings already sum to zero, and
    # bradley_terry mean-centres by default)
    for name, want in true.items():
        assert abs(elo[name] - want) < 40.0, f"{name}: {elo[name]:.1f} vs {want}"
    # a fitted rating difference must reproduce the observed win rate
    t = table_from(recs)
    got = t.win_rate("p4", "p0")
    pred = expected_score(elo["p4"], elo["p0"])
    assert abs(got - pred) < 0.03


def test_anchoring_pins_the_scale_so_a_growing_pool_does_not_screen_yero():
    recs = []
    for _ in range(40):
        recs += [MatchRecord("agent", "scripted_gold", 1.0), MatchRecord("agent", "x", 0.5)]
    elo = bradley_terry(table_from(recs), anchor="scripted_gold", anchor_elo=0.0)
    assert elo["scripted_gold"] == pytest.approx(0.0, abs=1e-9)
    assert elo["agent"] > 0.0
    # adding a new, much weaker player must not move the anchored player
    recs += [MatchRecord("agent", "weakling", 1.0) for _ in range(40)]
    elo2 = bradley_terry(table_from(recs), anchor="scripted_gold", anchor_elo=0.0)
    assert elo2["scripted_gold"] == pytest.approx(0.0, abs=1e-9)


def test_anchoring_on_an_unplayed_agent_is_an_error_not_a_silent_fallback():
    with pytest.raises(KeyError, match="cannot pin the scale"):
        bradley_terry(table_from([MatchRecord("a", "b", 1.0)]), anchor="nobody")


def test_expected_score_is_the_standard_elo_curve():
    assert expected_score(0, 0) == pytest.approx(0.5)
    assert expected_score(400, 0) == pytest.approx(10 / 11, abs=1e-9)
    assert expected_score(0, 400) == pytest.approx(1 / 11, abs=1e-9)


# -- the rot signature -----------------------------------------------------


def test_min_win_rate_vs_past_finds_the_worst_opponent():
    ev = Evaluator(anchors=[], min_games_for_min_winrate=5)
    for _ in range(20):
        ev.record_match(MatchRecord("latest", "old1", 0.9))
        ev.record_match(MatchRecord("latest", "old2", 0.4))  # the rot
        ev.record_match(MatchRecord("latest", "old3", 0.7))
    worst, considered, skipped = ev.min_win_rate_vs_past("latest", ["old1", "old2", "old3"])
    assert worst is not None
    assert worst[0] == "old2"
    assert worst[1] == pytest.approx(0.4)
    assert (considered, skipped) == (3, 0)


def test_an_opponent_with_too_few_games_is_skipped_and_counted_not_treated_as_zero():
    ev = Evaluator(anchors=[], min_games_for_min_winrate=10)
    for _ in range(20):
        ev.record_match(MatchRecord("latest", "old1", 0.8))
    ev.record_match(MatchRecord("latest", "fresh", 0.0))  # one unlucky game
    worst, considered, skipped = ev.min_win_rate_vs_past("latest", ["old1", "fresh"])
    assert worst[0] == "old1", "a single game must not raise a false rot alarm"
    assert (considered, skipped) == (1, 1)


def test_rot_warning_fires_when_the_minimum_drops_below_half(caplog):
    ev = Evaluator(anchors=[], min_games_for_min_winrate=5)
    for _ in range(20):
        ev.record_match(MatchRecord("latest", "old", 0.46))  # AlphaStar's number
    report = ev.report(1000, "latest", ["old"])
    assert report.rot_warning is True
    assert "ROT WARNING" in caplog.text
    assert "ROT WARNING" in report.summary()


def test_no_rot_warning_when_the_agent_still_beats_its_past():
    ev = Evaluator(anchors=[], min_games_for_min_winrate=5)
    for _ in range(20):
        ev.record_match(MatchRecord("latest", "old", 0.8))
    assert ev.report(10, "latest", ["old"]).rot_warning is False


def test_rising_elo_with_a_falling_minimum_is_visible_in_one_report():
    """The exact naive-self-play failure: the ordering looks great, the floor rots."""
    ev = Evaluator(anchors=[], min_games_for_min_winrate=5, elo_anchor=None)
    for _ in range(30):
        ev.record_match(MatchRecord("latest", "recent", 0.85))
        ev.record_match(MatchRecord("latest", "ancient", 0.40))
        ev.record_match(MatchRecord("recent", "ancient", 0.85))
    r = ev.report(500, "latest", ["recent", "ancient"])
    assert r.elo["latest"] == max(r.elo.values())  # top of the ladder
    assert r.rot_warning is True  # and rotting anyway


# -- CS@10 -----------------------------------------------------------------


def test_cs_tracker_reports_mean_and_spread():
    cs = CsTracker(window=100)
    for v in [30, 32, 34, 28]:
        cs.add("a", v)
    mean, sd, n = cs.stats("a")
    assert mean == pytest.approx(31.0) and n == 4 and sd > 0


def test_cs_tracker_is_a_window_not_a_lifetime_mean():
    cs = CsTracker(window=10)
    for _ in range(50):
        cs.add("a", 5.0)
    for _ in range(10):
        cs.add("a", 35.0)
    mean, _, n = cs.stats("a")
    assert n == 10 and mean == pytest.approx(35.0)


def test_unknown_agent_has_no_cs_rather_than_zero():
    assert CsTracker().stats("nobody") is None


def test_negative_cs_is_rejected():
    with pytest.raises(ValueError):
        CsTracker().add("a", -1)


# -- anchors ---------------------------------------------------------------


def test_the_four_permanent_anchors_are_the_three_difficulties_plus_bc():
    ids = [a.id for a in default_anchors()]
    assert ids == ["scripted_bronze", "scripted_gold", "scripted_diamond", "bc_policy"]


def test_scripted_anchor_configs_exist_and_carry_their_measured_cs():
    by_id = {a.id: a for a in default_anchors()}
    for aid, cs in [("scripted_bronze", 16.7), ("scripted_gold", 29.5), ("scripted_diamond", 35.2)]:
        a = by_id[aid]
        assert a.exists(), f"{aid} config missing at {a.resource}"
        assert a.reference_cs_at_10 == cs


def test_anchor_budget_is_about_five_percent_and_never_rounds_to_zero():
    assert ANCHOR_EPISODE_SHARE == 0.05
    assert anchor_episode_budget(1000) == 50
    assert anchor_episode_budget(3) == 1  # a tiny run still gets a frozen rung
    assert anchor_episode_budget(0) == 0


# -- report / persistence --------------------------------------------------


def test_report_carries_every_metric_self_play_cannot_provide():
    ev = Evaluator(anchors=default_anchors(), min_games_for_min_winrate=2, elo_anchor=None)
    for _ in range(10):
        ev.record_match(MatchRecord("latest", "scripted_bronze", 1.0))
        ev.record_match(MatchRecord("latest", "scripted_gold", 0.6))
        ev.record_match(MatchRecord("latest", "scripted_diamond", 0.2))
        ev.record_match(MatchRecord("latest", "snap@1", 0.55))
        ev.record_cs("latest", 30.0)
    r = ev.report(42, "latest", ["snap@1"])
    assert r.cs_at_10[0] == pytest.approx(30.0)
    assert r.win_rate_vs_anchor["scripted_bronze"] == (1.0, 10)
    assert r.win_rate_vs_anchor["scripted_diamond"][0] == pytest.approx(0.2)
    assert r.win_rate_vs_anchor["scripted_diamond"][1] == 10
    assert r.win_rate_vs_anchor["bc_policy"] == (None, 0)  # never played, not 0%
    assert r.min_win_rate_vs_past[0] == "snap@1"
    assert r.elo["latest"] > r.elo["scripted_bronze"]
    assert r.elo["scripted_diamond"] > r.elo["latest"]
    assert json.loads(r.to_json())["kind"] == "eval"


def test_a_metrics_log_replays_into_an_identical_evaluator(tmp_path):
    src = Evaluator(anchors=[], elo_anchor=None)
    lines = []
    for i in range(20):
        rec = MatchRecord("latest", "snap@1", 1.0 if i % 2 else 0.0, step=i)
        src.record_match(rec)
        lines.append(json.dumps({"kind": "match", "agent_a": rec.agent_a, "agent_b": rec.agent_b,
                                 "score_a": rec.score_a, "step": rec.step}))
        src.record_cs("latest", 25.0 + i)
        lines.append(json.dumps({"kind": "episode", "agent": "latest", "cs_at_10": 25.0 + i}))
    path = tmp_path / "metrics.jsonl"
    path.write_text("\n".join(lines) + "\n")

    dst = Evaluator(anchors=[], elo_anchor=None)
    assert dst.load_jsonl(path) == 20
    assert dst.table.win_rate("latest", "snap@1") == src.table.win_rate("latest", "snap@1")
    assert dst.cs.stats("latest") == src.cs.stats("latest")


def test_a_corrupt_metrics_line_raises_instead_of_half_reading(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"kind":"match","agent_a":"a","agent_b":"b","score_a":1.0}\nnot json\n')
    with pytest.raises(ValueError, match="is not JSON"):
        Evaluator(anchors=[]).load_jsonl(path)


# -- CS@10: which opponent was it against? ---------------------------------
#
# ``TrainingLoop.record_episode`` files every episode's CS under the run's own
# ``agent@N``, whatever it was played against, so ``runs/rl-bc4-0912`` put 144
# self-play games (mean 36.4) and 6 anchor games (mean 0.0) in the same bucket.
# The pooled mean is then a blend of a mirror match and a scripted-bot match
# whose weights move with the eval cadence -- and the absolute number, the one
# against an opponent that does not move, is the one that gets hidden.


def test_anchor_cs_and_self_play_cs_do_not_pool_into_one_number():
    ev = Evaluator(anchors=[], elo_anchor=None)
    for _ in range(10):
        ev.record_cs("agent@0", 36.0, category="self")
    for _ in range(2):
        ev.record_cs("agent@0", 0.0, category="anchor")
    r = ev.report(0, "agent@0", [])
    assert r.cs_at_10_by_category["self"][0] == pytest.approx(36.0)
    assert r.cs_at_10_by_category["self"][2] == 10
    assert r.cs_at_10_by_category["anchor"][0] == pytest.approx(0.0)
    assert r.cs_at_10_by_category["anchor"][2] == 2
    # the pooled number still exists, and is exactly the misleading blend
    assert r.cs_at_10[0] == pytest.approx(30.0)


def test_a_pooled_cs_number_admits_in_the_notes_that_it_is_a_blend():
    """A reader who sees one CS number has no way to know it is two populations."""
    ev = Evaluator(anchors=[], elo_anchor=None)
    ev.record_cs("agent@0", 36.0, category="self")
    ev.record_cs("agent@0", 0.0, category="anchor")
    notes = " ".join(ev.report(0, "agent@0", []).notes)
    assert "cs_at_10 pools" in notes
    assert "self" in notes and "anchor" in notes


def test_a_cs_reading_filed_without_a_category_says_so():
    """Silence here is what made the conflation invisible for a whole run."""
    ev = Evaluator(anchors=[], elo_anchor=None)
    ev.record_cs("agent@0", 36.0)
    r = ev.report(0, "agent@0", [])
    assert r.cs_at_10[0] == pytest.approx(36.0)  # back-compatible
    assert r.cs_at_10_by_category == {}
    assert any("no opponent_category breakdown" in n for n in r.notes)


def test_the_opponents_own_cs_from_the_same_game_reaches_the_report():
    """Replaces the stale hardcoded 16.7 / 29.5 / 35.2, measured with no rune page."""
    ev = Evaluator(anchors=default_anchors(), elo_anchor=None)
    for cs in (46.0, 50.0):
        ev.record_opponent_cs("scripted_bronze", cs)
    r = ev.report(0, "agent@0", [])
    assert r.opponent_cs_at_10["scripted_bronze"][0] == pytest.approx(48.0)
    assert r.opponent_cs_at_10["scripted_bronze"][2] == 2
    # and the point of measuring it: the bronze bot really farms ABOVE the
    # number the curriculum calls diamond
    stale = {a.id: a.reference_cs_at_10 for a in default_anchors()}
    assert r.opponent_cs_at_10["scripted_bronze"][0] > stale["scripted_diamond"]


def test_the_split_and_the_opponent_cs_are_visible_in_the_one_line_summary():
    ev = Evaluator(anchors=[], elo_anchor=None)
    ev.record_cs("agent@0", 36.0, category="self")
    ev.record_cs("agent@0", 12.0, category="anchor")
    ev.record_opponent_cs("scripted_bronze", 48.0)
    s = ev.report(0, "agent@0", []).summary()
    assert "cs@10[self]=36.0" in s
    assert "cs@10[anchor]=12.0" in s
    assert "cs@10[scripted_bronze itself]=48.0" in s


def test_a_metrics_log_replays_the_category_and_the_opponents_cs(tmp_path):
    """An offline replay must reach the same conclusion as the live run."""
    lines = [
        json.dumps({"kind": "episode", "agent": "agent@0", "cs_at_10": 36.0,
                    "opponent": "self", "opponent_category": "self"}),
        json.dumps({"kind": "episode", "agent": "agent@0", "cs_at_10": 12.0,
                    "opponent": "scripted_bronze", "opponent_category": "anchor",
                    "opponent_cs_at_10": 48.0}),
    ]
    path = tmp_path / "metrics.jsonl"
    path.write_text("\n".join(lines) + "\n")
    ev = Evaluator(anchors=[], elo_anchor=None)
    ev.load_jsonl(path)
    r = ev.report(0, "agent@0", [])
    assert r.cs_at_10_by_category["self"][0] == pytest.approx(36.0)
    assert r.cs_at_10_by_category["anchor"][0] == pytest.approx(12.0)
    assert r.opponent_cs_at_10["scripted_bronze"][0] == pytest.approx(48.0)
    assert json.loads(r.to_json())["cs_at_10_by_category"]["anchor"][2] == 1
