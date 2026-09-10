"""Opponent sampling must actually produce the mixture it claims.

The mixture is the whole design: 40% latest self / 40% PFSP / 15% uniform / 5%
frozen anchors.  A sampler that quietly drifts -- an empty category absorbing
mass, a PFSP weight that is really uniform -- would look exactly like a working
one from the outside, and would remove the anti-forgetting slice that keeps
min-win-rate-vs-past from collapsing.  So the distribution is measured, not
asserted by reading the code.
"""

from __future__ import annotations

import math
import random

import pytest

from lanerl_train.eval import AnchorSpec
from lanerl_train.league import (
    LATEST,
    CheckpointPool,
    LeagueConfig,
    LeagueConfigError,
    OpponentSampler,
    Snapshot,
    WinRateTracker,
    expected_pfsp_share,
)

DRAWS = 200_000
#: ~5 standard errors at N=200k for the largest slice.
TOL = 0.006


def make_pool(n=8, cfg=None):
    pool = CheckpointPool(cfg or LeagueConfig())
    for i in range(n):
        pool.add(Snapshot(id=f"snap@{i}", step=i * 100))
    return pool


def anchors(n=4):
    # kind="policy" with no resource is "usable" -- exists() only gates scripted
    # anchors, whose config file must really be there.
    return [AnchorSpec(f"anchor{i}", "policy", None) for i in range(n)]


# -- configuration ---------------------------------------------------------


def test_mixture_must_sum_to_one():
    with pytest.raises(LeagueConfigError, match="must sum to 1.0"):
        LeagueConfig(p_latest=0.4, p_pfsp=0.4, p_uniform=0.15, p_anchor=0.10)


def test_default_mixture_is_the_documented_one():
    cfg = LeagueConfig()
    assert cfg.weights() == {"latest": 0.40, "pfsp": 0.40, "uniform": 0.15, "anchor": 0.05}


def test_zero_prior_is_rejected():
    with pytest.raises(LeagueConfigError, match="Beta prior"):
        LeagueConfig(prior_wins=0.0)


# -- the mixture -----------------------------------------------------------


def test_category_frequencies_match_the_target_over_many_draws():
    cfg = LeagueConfig()
    sampler = OpponentSampler(
        make_pool(12, cfg), WinRateTracker(cfg), anchors(), cfg, random.Random(0)
    )
    specs = sampler.sample_many(DRAWS)
    counts = {k: 0 for k in cfg.weights()}
    for s in specs:
        counts[s.category] += 1
    realised = {k: v / DRAWS for k, v in counts.items()}
    for k, target in cfg.weights().items():
        assert abs(realised[k] - target) < TOL, f"{k}: {realised[k]:.4f} vs {target}"
    # the sampler's own bookkeeping must agree with an independent count
    assert sampler.realised_mixture() == pytest.approx(realised, abs=1e-12)
    assert max(abs(v) for v in sampler.mixture_drift().values()) < TOL


def test_latest_means_the_live_weights_on_both_sides():
    cfg = LeagueConfig()
    sampler = OpponentSampler(
        make_pool(4, cfg), WinRateTracker(cfg), anchors(2), cfg, random.Random(1)
    )
    latest = [s for s in sampler.sample_many(2000) if s.category == "latest"]
    assert latest, "the latest slice must be reachable"
    assert all(s.id == LATEST and s.is_latest and s.snapshot is None for s in latest)


def test_pfsp_draws_follow_one_minus_p_squared():
    cfg = LeagueConfig(halflife_games=None)  # no decay: exact arithmetic
    pool = make_pool(8, cfg)
    rates = WinRateTracker(cfg)
    # A spread of win rates, including one the agent is losing to badly.
    scripted = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.1]
    for snap, target in zip(pool.snapshots, scripted):
        for _ in range(50):
            rates.record(snap.id, target)  # fractional scores are legal (draws)
    sampler = OpponentSampler(pool, rates, anchors(2), cfg, random.Random(7))

    ps = {s.id: rates.p(s.id) for s in pool.snapshots}
    expected = expected_pfsp_share(ps, cfg.pfsp_exponent)
    # sanity: the reference and the sampler's own weights agree
    assert sampler.pfsp_weights() == pytest.approx(expected, abs=1e-12)

    specs = [s for s in sampler.sample_many(DRAWS) if s.category == "pfsp"]
    n = len(specs)
    assert n > 50_000
    counts = {sid: 0 for sid in ps}
    for s in specs:
        counts[s.id] += 1
    for sid, share in expected.items():
        se = math.sqrt(max(share * (1 - share), 1e-9) / n)
        got = counts[sid] / n
        assert abs(got - share) < 5 * se + 1e-3, f"{sid}: {got:.4f} vs {share:.4f}"
    # the opponent being lost to hardest must be the most-played one
    hardest = min(ps, key=lambda k: ps[k])
    assert max(counts, key=lambda k: counts[k]) == hardest


def test_uniform_slice_is_actually_uniform():
    cfg = LeagueConfig(halflife_games=None)
    pool = make_pool(6, cfg)
    rates = WinRateTracker(cfg)
    for s in pool.snapshots[:3]:
        for _ in range(100):
            rates.record(s.id, 1.0)  # PFSP would abandon these entirely
    sampler = OpponentSampler(pool, rates, anchors(1), cfg, random.Random(3))
    specs = [s for s in sampler.sample_many(DRAWS) if s.category == "uniform"]
    counts = {s.id: 0 for s in pool.snapshots}
    for s in specs:
        counts[s.id] += 1
    share = 1.0 / len(pool)
    for sid, c in counts.items():
        assert abs(c / len(specs) - share) < 0.01, f"{sid}: {c/len(specs):.4f}"
    # anti-forgetting: the beaten snapshots still get played
    assert all(counts[s.id] > 0 for s in pool.snapshots[:3])


def test_pfsp_falls_back_to_uniform_when_every_weight_is_zero(caplog):
    """"I beat everything" and "results are not being recorded" look identical."""

    class Unbeaten(WinRateTracker):
        def p(self, opponent_id: str) -> float:
            return 1.0

    cfg = LeagueConfig(halflife_games=None)
    pool = make_pool(3, cfg)
    sampler = OpponentSampler(pool, Unbeaten(cfg), (), cfg, random.Random(0))
    w = sampler.pfsp_weights()
    assert w == pytest.approx({s.id: 1 / 3 for s in pool.snapshots})
    assert "falling back to uniform" in caplog.text


def test_an_empty_pool_renormalises_onto_latest_and_anchors(caplog):
    cfg = LeagueConfig()
    sampler = OpponentSampler(CheckpointPool(cfg), WinRateTracker(cfg), anchors(2), cfg,
                              random.Random(5))
    specs = sampler.sample_many(20_000)
    cats = {s.category for s in specs}
    assert cats == {"latest", "anchor"}
    share_latest = sum(s.category == "latest" for s in specs) / len(specs)
    assert share_latest == pytest.approx(0.40 / 0.45, abs=0.02)
    assert "checkpoint pool is empty" in caplog.text


def test_no_opponent_at_all_is_fatal_not_silent():
    cfg = LeagueConfig(p_latest=0.0, p_pfsp=0.5, p_uniform=0.5, p_anchor=0.0)
    sampler = OpponentSampler(CheckpointPool(cfg), WinRateTracker(cfg), (), cfg)
    with pytest.raises(LeagueConfigError, match="nothing to play against"):
        sampler.sample()


# -- the pool --------------------------------------------------------------


def test_pool_is_bounded_and_keeps_the_origin_and_the_newest():
    cfg = LeagueConfig(pool_max=30, keep_recent=5)
    pool = CheckpointPool(cfg)
    for i in range(120):
        pool.add(Snapshot(id=f"snap@{i}", step=i * 10))
    ids = pool.ids()
    assert len(pool) <= cfg.pool_max
    assert len(pool) >= cfg.pool_min
    assert "snap@0" in ids, "the origin is the longest lever on the rot signal"
    for i in range(115, 120):
        assert f"snap@{i}" in ids
    steps = [s.step for s in pool.snapshots]
    assert steps == sorted(steps)


def test_pool_rejects_a_duplicate_id():
    pool = CheckpointPool()
    pool.add(Snapshot(id="a", step=1))
    with pytest.raises(LeagueConfigError, match="already in the pool"):
        pool.add(Snapshot(id="a", step=2))


def test_pool_round_trips_through_json():
    pool = make_pool(5)
    again = CheckpointPool.from_dict(pool.to_dict())
    assert again.ids() == pool.ids()


# -- win rates -------------------------------------------------------------


def test_unplayed_opponent_starts_at_one_half():
    assert WinRateTracker().p("never-played") == pytest.approx(0.5)


def test_win_rate_moves_toward_the_evidence():
    t = WinRateTracker(LeagueConfig(halflife_games=None))
    for _ in range(100):
        t.record("x", 1.0)
    assert t.p("x") > 0.95 and t.n("x") == 100
    for _ in range(100):
        t.record("y", 0.0)
    assert t.p("y") < 0.05


def test_decay_forgets_old_results():
    fast = WinRateTracker(LeagueConfig(halflife_games=5.0))
    slow = WinRateTracker(LeagueConfig(halflife_games=None))
    for t in (fast, slow):
        for _ in range(50):
            t.record("x", 1.0)
        for _ in range(10):
            t.record("x", 0.0)
    assert fast.p("x") < slow.p("x"), "a decayed tracker must react faster to a reversal"


def test_draws_are_representable():
    t = WinRateTracker(LeagueConfig(halflife_games=None))
    for _ in range(200):
        t.record("x", 0.5)
    assert t.p("x") == pytest.approx(0.5, abs=0.01)


def test_sampler_state_round_trips():
    cfg = LeagueConfig()
    pool = make_pool(4, cfg)
    rates = WinRateTracker(cfg)
    rates.record("snap@1", 1.0)
    s = OpponentSampler(pool, rates, anchors(1), cfg, random.Random(0))
    s.sample_many(50)
    blob = s.to_dict()
    other = OpponentSampler(CheckpointPool(cfg), WinRateTracker(cfg), anchors(1), cfg)
    other.load_dict(blob)
    assert other.pool.ids() == pool.ids()
    assert other.win_rates.p("snap@1") == pytest.approx(rates.p("snap@1"))
    assert other.counts == s.counts
