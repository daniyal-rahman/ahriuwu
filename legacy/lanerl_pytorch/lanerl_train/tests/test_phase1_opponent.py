"""The reward scale must hold still, and phase 1 must be reachable.

Two configuration failures, both of which have already cost a run.

**The objective moved mid-run with no way to stop it.**
``LaneRewardConfig`` anneals the zero-sum ``alpha`` from 0.5 to 1.0 over
2,000,000 rows of the anneal clock, ``__main__`` built one with defaults, and
there was no flag.  At ~255 rows per update that anneal is ~7,800 updates,
i.e. ~6.7 h, so a six-hour run spent its whole life on a changing reward
definition.  Measured in ``runs/rl-bc4-0912``, ep_return by 500-update bucket
ran 30.0 -> 26.6 -> 24.4 -> 34.9 -> 18.7 -> 6.5 while CS@10 stayed flat at
~36: the first bucket matches ``raw_blue - 0.5 * raw_red`` exactly and alpha
had reached ~0.67 by the end.  The return curve was measuring the reward, not
the policy.

**Every run was self-play, and every instance faced the same bot.**
``__main__`` always built ``SideAssignment(blue=SELF, red=SELF)`` with the
default launch spec.  In a mirror the opponent's farming stream is
uncontrollable noise in every advantage, and the training opponent is not the
yardstick the evaluation uses -- so CS against a fixed opponent arrived at
~1 sample per 400 updates from the anchor ladder instead of ~35 per 200 from
training itself.  And ``LANERL_BOT_SEED`` was never set by a training env, so
every instance inherited the server's default of 1234 (``LanerlConfig.cs:57``)
and N-fold parallelism bought N copies of one game.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from lanerl_rl.reward import LaneRewardConfig
from lanerl_train.eval import Evaluator
from lanerl_train.league import LeagueConfig
from lanerl_train.ports import InstancePorts
from lanerl_train.run import EpisodeResult, Rollout, RunConfig, TrainingLoop

from .fakes import FakeLearner

_REPO = Path(__file__).resolve().parents[2]


def _loop(run_dir: Path) -> TrainingLoop:
    """A learner-only loop: no actors, no eval cadence, driven by submit()."""
    cfg = RunConfig(
        run_dir=run_dir, num_actors=0, queue_capacity=4, checkpoint_every=0,
        snapshot_every=0, eval_every=0, stall_timeout_s=5.0, league=LeagueConfig(),
    )
    return TrainingLoop(cfg, FakeLearner(), evaluator=Evaluator(anchors=[]))


def _rows(run_dir: Path, kind: str):
    lines = (run_dir / "metrics.jsonl").read_text().splitlines()
    return [r for r in (json.loads(l) for l in lines) if r["kind"] == kind]


# -- A. alpha holds still ---------------------------------------------------


def test_the_alpha_flags_exist_and_are_documented():
    from lanerl_train.__main__ import build_argparser

    text = build_argparser().format_help()
    for flag in ("--alpha", "--alpha-anneal-steps"):
        assert flag in text


def test_the_reward_default_no_longer_moves_under_the_run():
    """The DEFAULT has to be constant, not merely constant when asked for.

    A flag that has to be passed to stop the objective drifting is a flag that
    will be forgotten on the run that matters -- and the drift is invisible in
    every metric except the return curve it corrupts.
    """
    from lanerl_train.__main__ import build_argparser, reward_config

    args = build_argparser().parse_args(["--run-name", "x"])
    cfg = reward_config(0.99, args.alpha, args.alpha_anneal_steps)
    early, mid, late = cfg.alpha(0), cfg.alpha(2_000_000), cfg.alpha(10 ** 9)
    assert early == mid == late, (
        f"alpha still moves under the default flags: {early} -> {mid} -> {late}"
    )
    # ...and it is the value every measured run actually spent its first bucket
    # at, so nothing about the reward's SCALE changes with this -- only that it
    # stops moving.
    assert early == pytest.approx(0.5)


def test_the_underlying_config_still_anneals_which_is_why_the_flag_exists():
    """Pins the thing being defended against, so this file cannot go stale."""
    bare = LaneRewardConfig()
    assert bare.alpha(0) != bare.alpha(bare.zero_sum_anneal_steps), (
        "LaneRewardConfig no longer anneals by default; if that changed "
        "deliberately, the --alpha default above should be revisited with it"
    )


def test_an_anneal_is_still_reachable_when_it_is_asked_for():
    from lanerl_train.__main__ import reward_config

    cfg = reward_config(0.99, 1.0, 2_000)
    assert cfg.alpha(0) == pytest.approx(LaneRewardConfig().zero_sum_alpha_start)
    assert cfg.alpha(1_000) == pytest.approx(0.75)
    assert cfg.alpha(2_000) == pytest.approx(1.0)
    assert cfg.alpha(10 ** 9) == pytest.approx(1.0), "the anneal must saturate, not overshoot"


def test_the_alpha_schedule_sets_both_ends_not_just_one():
    """``end`` alone looks exactly like a working --alpha while 0.5 -> 1.0 runs on.

    ``LaneRewardConfig.alpha`` interpolates from ``_start`` whenever
    ``anneal_steps > 0``, so "constant at x" is only true if the steps are
    zeroed as well.
    """
    from lanerl_train.__main__ import reward_config

    cfg = reward_config(0.99, 0.25, 0)
    assert cfg.zero_sum_anneal_steps == 0
    assert cfg.zero_sum_alpha_start == cfg.zero_sum_alpha_end == pytest.approx(0.25)


@pytest.mark.parametrize("alpha", [-0.1, 1.5])
def test_an_alpha_outside_the_unit_interval_is_refused(alpha):
    from lanerl_train.__main__ import reward_config

    with pytest.raises(SystemExit):
        reward_config(0.99, alpha, 0)


# -- B. training against the scripted bot ----------------------------------


def test_the_opponent_flag_exists_and_is_documented():
    from lanerl_train.__main__ import build_argparser

    text = build_argparser().format_help()
    assert "--opponent" in text
    assert "scripted" in text


def test_self_play_is_still_the_default_and_still_a_mirror():
    from lanerl_train.__main__ import build_argparser, build_training_specs

    args = build_argparser().parse_args(["--run-name", "x"])
    specs, labels, red_is_ours = build_training_specs(args.opponent, args.seed, 0, 2)
    assert red_is_ours, "the default must remain policy-vs-policy"
    assert labels == {}
    assert all(s.bot_teams == "none" for s in specs), (
        "a self-play run with LANERL_BOT set would have the in-server bot fighting "
        "the network for the same champion"
    )


def test_a_scripted_opponent_hands_red_to_the_in_server_bot():
    from lanerl_train.__main__ import build_training_specs

    specs, labels, red_is_ours = build_training_specs("scripted:bronze", 0, 0, 2)
    assert not red_is_ours, (
        "red must be assigned to NO policy: omitting the key is how the bot's own "
        "orders stand (SideAssignment)"
    )
    assert all(s.bot_teams == "purple" for s in specs)
    assert all(s.bot_config.name == "anchor_bronze.json" for s in specs)
    assert set(labels.values()) == {"train:scripted_bronze"}


def test_every_instance_of_the_run_faces_its_own_bot_seed():
    """Otherwise N parallel envs are N replays of one game.

    ``LANERL_BOT_SEED`` is read once at process start, so this is the level at
    which it can vary; ``LanerlBot.OnEpisodeReset`` deliberately does not
    re-seed the RNG, so successive episodes within one process continue the
    stream and do differ -- but two processes on the same seed do not.
    """
    from lanerl_train.__main__ import build_training_specs

    seeds = []
    for actor in range(3):
        specs, _, _ = build_training_specs("scripted:bronze", 0, actor, 4)
        seeds.extend(s.bot_seed for s in specs)
    assert all(s is not None for s in seeds), "the server default of 1234 is not a seed choice"
    assert len(set(seeds)) == len(seeds), f"two instances share a bot seed: {seeds}"


def test_a_different_run_seed_moves_every_bot_seed():
    from lanerl_train.__main__ import build_training_specs

    a, _, _ = build_training_specs("scripted:bronze", 0, 0, 4)
    b, _, _ = build_training_specs("scripted:bronze", 1, 0, 4)
    assert not (set(s.bot_seed for s in a) & set(s.bot_seed for s in b)), (
        "--seed must move the bot population too, or two 'independent' runs face "
        "the same opponent stream"
    )


def test_the_bot_seed_and_config_reach_the_process_environment():
    """The flag is only real if it arrives as an env var the server reads."""
    from lanerl_train.__main__ import build_training_specs

    specs, _, _ = build_training_specs("scripted:gold", 0, 0, 1)
    env = specs[0].environment(InstancePorts(0, 21000, 21001))
    assert env["LANERL_BOT"] == "purple"
    assert env["LANERL_BOT_SEED"] == str(specs[0].bot_seed)
    assert env["LANERL_BOT_CONFIG"].endswith("anchor_gold.json")


def test_population_play_is_opt_in_and_balanced_across_envs():
    """One name is one opponent; several is a population, evenly spread.

    Per INSTANCE rather than per episode because ``LANERL_BOT_CONFIG`` is read
    once at process start and nothing reloads it on reset -- see
    ``build_training_specs``. Round-robin rather than a random draw because a
    draw over 4 envs routinely leaves a difficulty unsampled.
    """
    from lanerl_train.__main__ import build_training_specs

    _, one, _ = build_training_specs("scripted:bronze", 0, 0, 4)
    assert set(one.values()) == {"train:scripted_bronze"}, "one name must not randomise"

    drawn = []
    for actor in range(2):
        _, labels, _ = build_training_specs("scripted:bronze,gold,diamond", 0, actor, 3)
        drawn.extend(labels[i] for i in sorted(labels))
    assert set(drawn) == {
        "train:scripted_bronze", "train:scripted_gold", "train:scripted_diamond",
    }, f"the population is not represented across envs: {drawn}"


def test_an_unknown_difficulty_is_fatal_rather_than_a_weaker_bot():
    """LANERL_BOT_CONFIG pointing nowhere leaves LanerlConfig on its defaults.

    That is a different and much weaker opponent, and nothing downstream can
    tell it apart from a difficulty somebody chose.
    """
    from lanerl_train.__main__ import build_training_specs

    with pytest.raises(SystemExit):
        build_training_specs("scripted:platinum", 0, 0, 2)
    with pytest.raises(SystemExit):
        build_training_specs("bronze", 0, 0, 2)  # missing the 'scripted:' kind


def test_the_cli_rejects_a_bad_opponent_before_it_boots_a_server():
    """Resolved in main(), not on the actor thread that first needs a driver.

    build_driver_for_actor is called lazily from inside ActorLoop, so a typo
    would otherwise surface as an actor dying minutes into a run that has
    already spawned its servers.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "lanerl_train", "--opponent", "scripted:platinum",
         "--run-name", "should-never-start"],
        cwd=_REPO, capture_output=True, text=True, timeout=300,
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0
    # Not argparse refusing an unknown flag -- that is what it did before
    # --opponent existed, and it would make this test pass vacuously.
    assert "unrecognized arguments" not in out, "--opponent is not a real flag"
    assert "platinum" in out


# -- C. the diagnostics have to reach the log -------------------------------


def test_the_episode_row_carries_every_per_episode_diagnostic(run_dir):
    """A field on EpisodeResult that metrics.jsonl drops is not a diagnostic.

    ``opponent_cs_at_10`` is the case in point: it was computed, logged to INFO
    by AnchorEvaluator, and never written -- so the only durable record of how
    the agent did relative to what it played was a hardcoded constant measured
    on a different bot.
    """
    loop = _loop(run_dir)
    loop.record_episode(EpisodeResult(
        agent="self", opponent_id="train:scripted_bronze", opponent_category="scripted",
        score=0.0, cs_at_10=21.0, opponent_cs_at_10=48.0, length_steps=18_000,
        reason="time", instance=1, ep_return=-3.5, kills=1, deaths=2,
        first_frame_ad=78.14, first_frame_mhp=672.0,
        reward_terms={"last_hit": 21.0, "death": -2.0, "shaping": 0.4},
    ))
    row = _rows(run_dir, "episode")[0]
    assert row["opponent_cs_at_10"] == 48.0
    assert row["kills"] == 1 and row["deaths"] == 2
    assert row["first_frame_ad"] == pytest.approx(78.14)
    assert row["first_frame_mhp"] == pytest.approx(672.0)
    assert row["reward_terms"]["last_hit"] == pytest.approx(21.0)
    assert row["reward_terms"]["death"] == pytest.approx(-2.0)


def test_the_live_report_and_a_replay_of_its_own_log_agree(run_dir):
    """``load_jsonl`` reads the category and the opponent's CS; the live path did not.

    So a report rebuilt offline from a run's own metrics.jsonl carried numbers
    the live report -- the one anybody actually watches -- did not have.
    """
    loop = _loop(run_dir)
    for cs in (20.0, 22.0):
        loop.record_episode(EpisodeResult(
            agent="self", opponent_id="train:scripted_bronze",
            opponent_category="scripted", score=0.5,
            cs_at_10=cs, opponent_cs_at_10=48.0,
        ))
    replayed = Evaluator(anchors=[])
    replayed.load_jsonl(run_dir / "metrics.jsonl")
    live, offline = loop.evaluator, replayed
    assert live.opponent_cs.stats("train:scripted_bronze") is not None, (
        "the live run never recorded the opponent's CS at all"
    )
    assert (live.opponent_cs.stats("train:scripted_bronze")
            == offline.opponent_cs.stats("train:scripted_bronze"))
    assert (live.cs_by_category["scripted"].stats(loop.agent_id())
            == offline.cs_by_category["scripted"].stats(loop.agent_id())), (
        "the live path pooled every category into one mean, so the split only "
        "existed offline"
    )


def test_the_update_row_carries_the_action_marginals(run_dir):
    """Per update, because it is only meaningful as a time series."""
    loop = _loop(run_dir)
    loop.submit(Rollout(
        actor_id=0, param_version=loop.state.param_version, steps=4,
        action_marginals={"noop": 0.9, "move": 0.1},
    ))
    assert loop.step_once(timeout=1.0)
    row = _rows(run_dir, "update")[0]
    assert row["actions/noop"] == pytest.approx(0.9)
    assert row["actions/move"] == pytest.approx(0.1)


# -- the launch specs that carry all of the above --------------------------


def test_each_instance_can_be_launched_from_its_own_spec():
    """One shared spec is why every instance had the same bot and the same seed."""
    from lanerl_train.vec import ServerLaunchSpec, VecLaneEnv
    from lanerl_train.tests.fakes import FakeInstance

    ports = [InstancePorts(i, 46000 + 2 * i, 46000 + 2 * i + 1) for i in range(2)]
    a = ServerLaunchSpec(bot_teams="purple", bot_seed=11)
    b = ServerLaunchSpec(bot_teams="purple", bot_seed=22)
    env = VecLaneEnv(2, specs=[a, b], ports=ports, factory=lambda i, p: FakeInstance(i))
    # The real factory is what a production run uses; build the handles it would
    # build without spawning anything (start() is what spawns).
    assert env._default_factory(0, ports[0]).spec is a
    assert env._default_factory(1, ports[1]).spec is b


def test_spec_and_specs_together_is_refused():
    from lanerl_train.vec import ServerLaunchSpec, VecLaneEnv
    from lanerl_train.tests.fakes import FakeInstance

    ports = [InstancePorts(0, 46100, 46101)]
    with pytest.raises(ValueError):
        VecLaneEnv(1, spec=ServerLaunchSpec(), specs=[ServerLaunchSpec()],
                   ports=ports, factory=lambda i, p: FakeInstance(i))


def test_a_spec_per_instance_means_exactly_one_per_instance():
    from lanerl_train.vec import ServerLaunchSpec, VecLaneEnv
    from lanerl_train.tests.fakes import FakeInstance

    ports = [InstancePorts(i, 46200 + 2 * i, 46200 + 2 * i + 1) for i in range(2)]
    with pytest.raises(ValueError):
        VecLaneEnv(2, specs=[ServerLaunchSpec()], ports=ports,
                   factory=lambda i, p: FakeInstance(i))
