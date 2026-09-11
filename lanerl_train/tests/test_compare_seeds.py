"""Cross-run comparison and multi-seed aggregation, on synthetic run dirs.

Synthetic rather than fixtures copied from a real run: every property asserted
here is one a real run can violate, and the only way to test "absent is not
zero" is to write a metrics file with the field genuinely absent.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from lanerl_train.compare import render
from lanerl_train.runstats import (
    RESUME_GAP_S,
    TAIL_WINDOW,
    config_differences,
    flatten_config,
    load_run,
    load_runs,
)
from lanerl_train.seeds import (
    MIN_SEEDS_FOR_A_USABLE_CI,
    aggregate,
    bootstrap_interval,
    plan_seeds,
    render_aggregate,
    t_interval,
)


def write_run(
    root: Path,
    name: str,
    *,
    config: dict | None = None,
    updates: int = 100,
    wall0: float = 1_000_000.0,
    update_dt: float = 2.0,
    entropy=lambda i: 8.8,
    ep_returns=None,
    cs_at_10=None,
    throughput: bool = True,
    resume_after: int | None = None,
    anchor_scores=None,
) -> Path:
    """One run directory, written the way TrainingLoop writes one."""
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    if config is not None:
        (d / "resolved_config.json").write_text(json.dumps(config, indent=2))
    lines = []
    wall = wall0
    for i in range(updates):
        wall += update_dt
        if resume_after is not None and i == resume_after:
            wall += 3600.0  # a requeue: an hour of queued time, not training
        rec = {
            "kind": "update",
            "wall": wall,
            "update": i + 1,
            "steps": 255,
            "parallel_envs": 8,
            "loss/entropy": entropy(i),
            "loss/value_loss": 0.001,
            "loss/approx_kl": 0.02,
            "loss/clip_frac": 0.3,
            "loss/epochs_run": 1.0,
        }
        if throughput:
            rec.update(
                {
                    "throughput/updates_per_s": 1.0 / update_dt,
                    "throughput/env_steps_per_s": 255.0 / update_dt,
                    "throughput/decisions_per_s": 255.0 * 8 / update_dt,
                    "throughput/learner_frac": 0.05,
                    "gpu/util_pct": 5.0,
                    "gpu/mem_allocated_mb": 128.0,
                }
            )
        lines.append(json.dumps(rec))
    for j, r in enumerate(ep_returns or []):
        ep = {
            "kind": "episode",
            "wall": wall0 + j,
            "agent": "self",
            "opponent": "self",
            "opponent_category": "self",
            "score": 0.5,
            "ep_return": r,
            "cs_at_10": (cs_at_10[j] if cs_at_10 else None),
        }
        lines.append(json.dumps(ep))
    for j, sc in enumerate(anchor_scores or []):
        lines.append(
            json.dumps(
                {
                    "kind": "episode",
                    "wall": wall0 + j,
                    "agent": "agent@1",
                    "opponent": "scripted_bronze",
                    "opponent_category": "anchor",
                    "score": sc,
                    "ep_return": None,
                    "cs_at_10": None,
                }
            )
        )
    (d / "metrics.jsonl").write_text("\n".join(lines) + "\n")
    return d


BASE_CONFIG = {
    "args": {"seed": 0, "run_name": "x", "port_base": 21000},
    "ppo_config": {"lr": 3e-4, "entropy_coef": 0.01, "target_kl": 0.02},
    "run_config": {"num_actors": 3, "seed": 0, "run_dir": "/x"},
}


# -- runstats --------------------------------------------------------------


def test_flatten_names_the_leaf_not_the_subtree():
    flat = flatten_config(BASE_CONFIG)
    assert flat["ppo_config.lr"] == 3e-4
    assert "ppo_config" not in flat


def test_a_key_one_run_lacks_is_reported_separately_from_a_changed_value(tmp_path):
    a = write_run(tmp_path, "a", config={"ppo_config": {"lr": 3e-4, "target_kl": 0.02}})
    b = write_run(tmp_path, "b", config={"ppo_config": {"lr": 1e-4}})
    differing, missing = config_differences(load_runs([a, b]))
    assert differing == ["ppo_config.lr"]
    assert missing == ["ppo_config.target_kl"]


def test_absent_ep_return_is_none_with_n_zero_never_zero(tmp_path):
    """The whole first run had no ep_return; reporting 0.0 would look like failure."""
    d = write_run(tmp_path, "old", config=BASE_CONFIG, ep_returns=None)
    s = load_run(d)
    assert s.ep_return_final is None
    assert s.ep_return_n == 0
    assert any("ep_return" in n for n in s.notes)


def test_ep_return_final_is_the_tail_mean_and_best_is_the_best_window(tmp_path):
    # Rises, then collapses: "final" must show the collapse and "best" the peak.
    rets = [float(i) for i in range(200)] + [0.0] * TAIL_WINDOW
    d = write_run(tmp_path, "r", config=BASE_CONFIG, ep_returns=rets)
    s = load_run(d)
    assert s.ep_return_final == pytest.approx(0.0)
    assert s.ep_return_best == pytest.approx(sum(range(150, 200)) / TAIL_WINDOW)
    assert s.ep_return_n == len(rets)


def test_a_resume_gap_is_not_counted_as_training_time(tmp_path):
    d = write_run(tmp_path, "r", config=BASE_CONFIG, updates=100, update_dt=2.0,
                  resume_after=50)
    s = load_run(d)
    # 99 inter-update gaps of 2 s, of which the one straddling the resume (2 s
    # of training plus 3600 s of queue) is dropped whole: 98 x 2 s survives.
    assert s.wall_h == pytest.approx(98 * 2.0 / 3600.0)
    assert any("resume seam" in n for n in s.notes)
    assert RESUME_GAP_S < 3600.0


def test_throughput_is_derived_for_runs_that_predate_the_telemetry(tmp_path):
    d = write_run(tmp_path, "old", config=BASE_CONFIG, updates=50, update_dt=4.0,
                  throughput=False)
    s = load_run(d)
    assert s.updates_per_s == pytest.approx(50 / (49 * 4.0))
    assert s.learner_frac is None
    assert any("predates throughput" in n for n in s.notes)


def test_throughput_is_read_straight_out_when_present(tmp_path):
    d = write_run(tmp_path, "new", config=BASE_CONFIG, updates=50, update_dt=4.0)
    s = load_run(d)
    assert s.decisions_per_s == pytest.approx(255 * 8 / 4.0)
    assert s.learner_frac == pytest.approx(0.05)
    assert s.gpu_util_pct == pytest.approx(5.0)
    assert not any("predates throughput" in n for n in s.notes)


def test_an_entropy_that_never_falls_shows_a_flat_slope(tmp_path):
    flat = write_run(tmp_path, "flat", config=BASE_CONFIG, updates=1000)
    falling = write_run(
        tmp_path, "fall", config=BASE_CONFIG, updates=1000,
        entropy=lambda i: 8.8 - 0.002 * i,
    )
    sf, sd = load_run(flat), load_run(falling)
    assert sf.entropy_slope_per_1k == pytest.approx(0.0, abs=1e-9)
    assert sd.entropy_slope_per_1k == pytest.approx(-2.0, rel=1e-6)


def test_a_run_with_no_anchor_games_says_so(tmp_path):
    d = write_run(tmp_path, "selfplay", config=BASE_CONFIG)
    s = load_run(d)
    assert s.anchor_win_rate is None
    assert any("frozen anchor" in n for n in s.notes)


def test_anchor_episodes_become_a_win_rate(tmp_path):
    d = write_run(tmp_path, "anchored", config=BASE_CONFIG,
                  anchor_scores=[1.0, 0.0, 0.5, 1.0])
    s = load_run(d)
    assert s.anchor_games == 4
    assert s.anchor_win_rate == pytest.approx(0.625)


def test_a_torn_last_line_is_counted_not_fatal(tmp_path):
    d = write_run(tmp_path, "killed", config=BASE_CONFIG, updates=10)
    with (d / "metrics.jsonl").open("a") as fh:
        fh.write('{"kind":"update","wall":1,')  # SIGKILL mid-write
    s = load_run(d)
    assert s.updates == 10
    assert any("unparseable" in n for n in s.notes)


def test_a_run_that_died_before_its_first_update_still_summarises(tmp_path):
    d = tmp_path / "stillborn"
    d.mkdir()
    (d / "resolved_config.json").write_text(json.dumps(BASE_CONFIG))
    s = load_run(d)
    assert s.updates is None
    assert any("no metrics.jsonl" in n for n in s.notes)


# -- compare ---------------------------------------------------------------


def test_compare_shows_only_the_differing_keys_by_default(tmp_path):
    a = write_run(tmp_path, "a", config={"args": {"lr": 3e-4, "seed": 0, "epochs": 4}})
    b = write_run(tmp_path, "b", config={"args": {"lr": 1e-4, "seed": 0, "epochs": 4}})
    text = render(load_runs([a, b]))
    assert "args.lr" in text
    assert "args.epochs" not in text
    assert "args.seed" not in text
    assert "3e-04" in text or "0.0003" in text

    everything = render(load_runs([a, b]), all_config=True)
    assert "args.epochs" in everything


def test_compare_prints_the_headline_metrics_side_by_side(tmp_path):
    a = write_run(tmp_path, "a", config=BASE_CONFIG, updates=100,
                  ep_returns=[1.0] * 60)
    b = write_run(tmp_path, "b", config=BASE_CONFIG, updates=200,
                  ep_returns=[2.0] * 60)
    text = render(load_runs([a, b]))
    line = next(ln for ln in text.splitlines() if "ep_return (last" in ln)
    assert "1.0000" in line and "2.0000" in line
    assert "100" in text and "200" in text


def test_compare_says_identical_when_the_configs_match(tmp_path):
    a = write_run(tmp_path, "a", config=BASE_CONFIG)
    b = write_run(tmp_path, "b", config=BASE_CONFIG)
    assert "the resolved configs are identical" in render(load_runs([a, b]))


# -- intervals -------------------------------------------------------------


def test_t_interval_matches_the_textbook_formula():
    xs = [1.0, 2.0, 3.0, 4.0]
    iv = t_interval(xs)
    mean, sd = 2.5, math.sqrt(sum((x - 2.5) ** 2 for x in xs) / 3)
    half = 3.182 * sd / 2.0
    assert iv.mean == pytest.approx(mean)
    assert iv.lo == pytest.approx(mean - half)
    assert iv.hi == pytest.approx(mean + half)
    assert not iv.unusable


def test_n_equals_one_has_no_interval_and_says_so():
    iv = t_interval([3.0])
    assert iv.mean == 3.0 and iv.lo is None and iv.hi is None
    assert iv.unusable and "n=1" in iv.note


def test_n_equals_two_produces_an_interval_flagged_unusable():
    iv = t_interval([1.0, 2.0])
    assert iv.lo is not None
    assert iv.unusable, "a t multiplier of 12.7 must not be presented as a measurement"
    assert iv.n == 2 < MIN_SEEDS_FOR_A_USABLE_CI


def test_bootstrap_is_deterministic_and_flags_its_own_quantisation():
    xs = [1.0, 2.0, 3.0]
    a = bootstrap_interval(xs, resamples=2000, seed=7)
    b = bootstrap_interval(xs, resamples=2000, seed=7)
    assert (a.lo, a.hi) == (b.lo, b.hi)
    assert a.unusable and "distinct resamples" in a.note
    big = bootstrap_interval([float(i) for i in range(12)], resamples=2000, seed=7)
    assert not big.unusable
    assert big.lo < big.mean < big.hi


def test_a_metric_no_seed_recorded_aggregates_to_n_zero(tmp_path):
    runs = [write_run(tmp_path, f"s{i}", config=BASE_CONFIG) for i in range(3)]
    ivs = aggregate(load_runs(runs))
    assert ivs["ep_return_final"].n == 0
    assert ivs["ep_return_final"].mean is None
    assert ivs["updates"].n == 3


# -- seed planning ---------------------------------------------------------


def test_plan_gives_each_seed_its_own_name_dir_and_port_block(tmp_path):
    plans = plan_seeds("abl", [0, 1, 2], ["--lr", "1e-4"], port_base=21000,
                       runs_root=tmp_path)
    assert [p.run_name for p in plans] == ["abl-s0", "abl-s1", "abl-s2"]
    assert len({p.run_dir for p in plans}) == 3
    bases = [int(p.argv[p.argv.index("--port-base") + 1]) for p in plans]
    assert len(set(bases)) == 3 and sorted(bases) == bases
    assert all(p.argv[-2:] == ["--lr", "1e-4"] for p in plans)


def test_plan_refuses_to_let_extra_args_own_the_seed():
    with pytest.raises(ValueError, match="--seed"):
        plan_seeds("abl", [0, 1, 2], ["--seed", "7"])
    with pytest.raises(ValueError, match="--run-name"):
        plan_seeds("abl", [0, 1, 2], ["--run-name=other"])


def test_plan_refuses_duplicate_seeds():
    with pytest.raises(ValueError, match="duplicate seeds"):
        plan_seeds("abl", [0, 1, 1])


def test_nothing_is_launched_without_execute(tmp_path):
    from lanerl_train.seeds import launch, render_plan

    plans = plan_seeds("abl", [0, 1], runs_root=tmp_path)
    text = render_plan(plans, python="/usr/bin/python3")
    assert "Nothing has been launched" in text
    handles = launch(plans, execute=False)
    assert all("-m lanerl_train" in h for h in handles)
    assert not (tmp_path / "abl-s0").exists()


# -- the aggregate report --------------------------------------------------


def test_the_report_shouts_when_n_is_too_small(tmp_path):
    runs = [write_run(tmp_path, f"s{i}", config=BASE_CONFIG) for i in range(2)]
    summaries = load_runs(runs)
    text = render_aggregate(summaries, aggregate(summaries), "t")
    assert "NOT a measurement" in text
    assert "n=2" in text


def test_the_report_is_quiet_at_a_usable_n(tmp_path):
    runs = [write_run(tmp_path, f"s{i}", config=BASE_CONFIG) for i in range(5)]
    summaries = load_runs(runs)
    text = render_aggregate(summaries, aggregate(summaries), "t")
    assert "NOT a measurement" not in text
    assert "UNUSABLE" not in text


def test_a_sweep_whose_members_differ_in_config_is_called_out(tmp_path):
    """Three runs with different lr is not a seed sweep, and its mean is a lie."""
    runs = [
        write_run(tmp_path, f"s{i}", config={"args": {"seed": i}, "ppo_config": {"lr": lr}})
        for i, lr in enumerate([3e-4, 1e-4, 5e-4])
    ]
    summaries = load_runs(runs)
    text = render_aggregate(summaries, aggregate(summaries), "t")
    assert "do NOT share a config" in text
    assert "ppo_config.lr" in text
    # args.seed and run_name legitimately differ and must NOT be flagged.
    assert "!!   args.seed" not in text


def test_a_real_three_seed_sweep_reports_a_mean_and_an_interval(tmp_path):
    runs = [
        write_run(tmp_path, f"s{i}", config={"args": {"seed": i}, "ppo_config": {"lr": 3e-4}},
                  ep_returns=[r] * 60)
        for i, r in enumerate([1.0, 2.0, 3.0])
    ]
    summaries = load_runs(runs)
    ivs = aggregate(summaries)
    assert ivs["ep_return_final"].n == 3
    assert ivs["ep_return_final"].mean == pytest.approx(2.0)
    assert ivs["ep_return_final"].lo == pytest.approx(2.0 - 4.303 * 1.0 / math.sqrt(3))
    text = render_aggregate(summaries, ivs, "t")
    assert "do NOT share a config" not in text
    assert "NOT a measurement" not in text
