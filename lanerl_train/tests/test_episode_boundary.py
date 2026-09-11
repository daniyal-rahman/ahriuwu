"""What happens at an episode boundary, and what used to be lost there.

``VecDriver.step()`` ends an episode and starts the next one inside a single
call: it detects the done, sends ``{"cmd":"reset"}``, reads the post-reset
observation and hands that back as ``result.obs[i]``.  That is the right
protocol -- the server blocks on a read, so the reset line *is* that step's
action -- but it used to mean the observation that ENDED the episode was
overwritten and then unreachable.  Three things were read off the wrong frame
as a result, and each one is a test below:

* ``cs_at_10`` was read from ``env.last_obs`` *after* the reset, i.e. from a
  frame where the champion has 0 CS.  Every completed episode in
  ``runs/rl-overnight-0911-0608`` recorded ``cs_at_10: 0.0`` -- 298 of them in
  the only segment where the field was populated at all.
* the reward of the final transition was computed from that same post-reset
  frame, where ``InstanceRewardContext`` is deliberately invalid, so it was
  written as exactly 0.0.  With ``end_on_death=True`` the frame that ends the
  episode is the frame the champion dies on, so the ``death`` term (-1.0, the
  largest single weight in ``RewardWeights``) could never reach the learner.
* ``done`` in the rollout buffer was taken from the step that *followed* the
  boundary, so GAE cut the bootstrap one row late -- through the terminal
  transition and into the first transition of the next episode.

A fourth: an episode that ended on the first iteration of a rollout was never
recorded at all, and its step counter was never cleared, so the next episode
reported double length.  ``17994`` (exactly 2 x 8997) appears in that run.
"""

from __future__ import annotations

from typing import List

import pytest
import torch

from lanerl_rl.model import LanePolicy, ModelConfig
from lanerl_train.lane_wiring import (
    LanePolicyActor,
    collect_rollout,
    make_lane_adapters,
)
from lanerl_train.tests.fakes import FakeAdapter, FakeEncoder, FakeInstance, FakePolicy
from lanerl_train.vec import (
    EpisodeSpec,
    SideAssignment,
    VecDriver,
    VecLaneEnv,
)

SELF = "self"


def build_env_with(instances: List[FakeInstance]) -> VecLaneEnv:
    from lanerl_train.ports import InstancePorts

    n = len(instances)
    ports = [InstancePorts(i, 44000 + 2 * i, 44000 + 2 * i + 1) for i in range(n)]
    return VecLaneEnv(
        n,
        ports=ports,
        factory=lambda i, p: instances[i],
        step_timeout_s=2.0,
        auto_restart=False,
    )


# -- the driver keeps the frame that ended the episode ---------------------


def test_the_observation_that_ended_the_episode_survives_the_reset():
    """Without this the terminal frame is unreachable from outside the driver."""
    insts = [FakeInstance(0, step_ms=200_000, cs_per_step=3)]
    env = build_env_with(insts)
    pol = FakePolicy("p")
    driver = VecDriver(
        env,
        policies={"p": pol},
        adapter_factory=lambda i, side: FakeAdapter(i, side),
        encoder=FakeEncoder(),
        assignments=[SideAssignment(blue="p", red=None)],
        episode=EpisodeSpec(max_game_ms=600_000),
    )
    driver.start()
    for _ in range(4):
        result, dones = driver.step()
        if dones:
            break
    assert dones == {0: "time"}, "the fake should have crossed ten game-minutes"
    assert 0 in result.terminal_obs, "the frame that ended the episode was not kept"
    assert result.terminal_obs[0]["t"] >= 600_000
    # ...and the frame handed back for the next decision is the post-reset one.
    assert result.obs[0]["t"] < 600_000
    assert result.obs[0] is not result.terminal_obs[0]


def test_no_terminal_obs_when_nothing_ended():
    insts = [FakeInstance(0)]
    env = build_env_with(insts)
    pol = FakePolicy("p")
    driver = VecDriver(
        env,
        policies={"p": pol},
        adapter_factory=lambda i, side: FakeAdapter(i, side),
        encoder=FakeEncoder(),
        assignments=[SideAssignment(blue="p", red=None)],
        episode=EpisodeSpec(max_game_ms=10**9),
    )
    driver.start()
    result, dones = driver.step()
    assert dones == {}
    assert result.terminal_obs == {}


# -- the real wiring: what the collector records ---------------------------


def small_policy() -> LanePolicy:
    """A real ``LanePolicy``, shrunk everywhere the observation does not pin it."""
    return LanePolicy(
        ModelConfig(core_dim=32, d_model=32, ffn_dim=32, n_layers=1, n_heads=2, mlp_hidden=32)
    )


def rollout_over(insts, episode: EpisodeSpec, num_steps: int, train_step: int = 0):
    env = build_env_with(insts)
    policy = small_policy()
    actor = LanePolicyActor(policy)
    adapters = make_lane_adapters(train_step_source=lambda: train_step)
    driver = VecDriver(
        env,
        policies={SELF: actor},
        adapter_factory=adapters.adapter_factory,
        encoder=adapters.encoder,
        assignments=[SideAssignment(blue=SELF, red=SELF) for _ in insts],
        episode=episode,
    )
    driver.start()
    rollout = collect_rollout(
        driver, actor, adapters.reward_contexts, SELF,
        num_steps=num_steps, gamma=0.99, gae_lambda=0.95,
    )
    return rollout, driver, adapters


def test_cs_at_10_comes_from_the_final_frame_not_the_post_reset_one():
    """The whole run recorded cs_at_10: 0.0 because it read env.last_obs."""
    # 6 decisions of 100_000 ms each reaches 600_000 on the 6th; 3 CS a decision.
    insts = [FakeInstance(0, step_ms=100_000, cs_per_step=3)]
    rollout, driver, _ = rollout_over(
        insts, EpisodeSpec(max_game_ms=600_000), num_steps=12
    )
    timed = [e for e in rollout.episodes if e.reason == "time"]
    assert timed, f"no episode ended on the clock: {[e.reason for e in rollout.episodes]}"
    assert timed[0].cs_at_10 is not None
    assert timed[0].cs_at_10 > 0.0, (
        "cs_at_10 was read from the post-reset frame, where CS is 0 by construction"
    )


def test_the_death_that_ends_an_episode_reaches_the_learner():
    """end_on_death makes the death frame the terminal frame; it was discarded."""
    insts = [FakeInstance(0, step_ms=1_000, champ_dies_at_ms=4_000)]
    rollout, driver, adapters = rollout_over(
        insts, EpisodeSpec(max_game_ms=10**9, end_on_death=True), num_steps=12
    )
    deaths = [e for e in rollout.episodes if e.reason.startswith("death")]
    assert deaths, f"no death episode: {[e.reason for e in rollout.episodes]}"
    rewards = rollout.data.rewards[: rollout.data.step]
    assert float(rewards.abs().max()) > 0.0, (
        "every reward in the rollout is exactly 0: the frame carrying the death "
        "was replaced by the post-reset frame before any adapter saw it"
    )


def test_done_marks_the_transition_that_ended_the_episode():
    """GAE must cut the bootstrap AT the terminal row, not one row later."""
    insts = [FakeInstance(0, step_ms=100_000)]
    rollout, driver, _ = rollout_over(
        insts, EpisodeSpec(max_game_ms=600_000), num_steps=12
    )
    buf = rollout.data
    dones = buf.dones[: buf.step, 0]
    resets = buf.resets[: buf.step, 0]
    done_rows = [int(t) for t in range(buf.step) if float(dones[t]) > 0.5]
    reset_rows = [int(t) for t in range(buf.step) if float(resets[t]) > 0.5]
    assert done_rows, "no terminal row was marked at all"
    # The row AFTER a terminal row is the first decision of the next episode,
    # which is exactly the row the policy is told to reset its state on.
    for t in done_rows:
        if t + 1 < buf.step:
            assert t + 1 in reset_rows, (
                f"row {t} is marked done but row {t+1} is not a fresh-episode row; "
                f"done={done_rows} reset={reset_rows}"
            )


def test_an_episode_that_ends_on_the_first_iteration_is_still_recorded():
    """Otherwise its length is silently added to the next episode's (17994)."""
    # start_t_ms puts the fake one decision short of the limit, so the very
    # first iteration of the rollout crosses it.
    insts = [FakeInstance(0, step_ms=100_000, start_t_ms=550_000)]
    rollout, driver, _ = rollout_over(
        insts, EpisodeSpec(max_game_ms=600_000), num_steps=14
    )
    reasons = [e.reason for e in rollout.episodes]
    assert reasons.count("time") >= 2, (
        f"the boundary on the first iteration was dropped: {reasons}"
    )
    lengths = [e.length_steps for e in rollout.episodes if e.reason == "time"]
    assert all(1 <= n <= 7 for n in lengths), (
        f"an episode length is the sum of two episodes: {lengths}"
    )


def test_episode_length_counts_decisions_and_survives_a_rollout_boundary():
    insts = [FakeInstance(0, step_ms=100_000)]
    env = build_env_with(insts)
    policy = small_policy()
    actor = LanePolicyActor(policy)
    adapters = make_lane_adapters(train_step_source=lambda: 0)
    driver = VecDriver(
        env,
        policies={SELF: actor},
        adapter_factory=adapters.adapter_factory,
        encoder=adapters.encoder,
        assignments=[SideAssignment(blue=SELF, red=SELF)],
        episode=EpisodeSpec(max_game_ms=600_000),
    )
    driver.start()
    episodes = []
    for _ in range(3):
        r = collect_rollout(
            driver, actor, adapters.reward_contexts, SELF,
            num_steps=4, gamma=0.99, gae_lambda=0.95,
        )
        episodes.extend(r.episodes)
    assert episodes, "three 4-step rollouts should span at least one boundary"
    # 600_000 / 100_000 = 6 decisions per episode, and the rollout is 4 long,
    # so a correct counter can only get there by carrying across rollouts.
    assert [e.length_steps for e in episodes] == [6] * len(episodes), (
        f"episode lengths {[e.length_steps for e in episodes]} are rollout-local, "
        f"not per-episode"
    )


def test_rollout_reports_rows_and_the_slot_width_they_cover():
    """`steps` is buffer ROWS; decisions are rows x parallel_envs."""
    insts = [FakeInstance(i) for i in range(2)]
    rollout, driver, _ = rollout_over(insts, EpisodeSpec(max_game_ms=10**9), num_steps=6)
    assert rollout.steps == 5, "num_steps decisions yield num_steps-1 rows"
    assert rollout.parallel_envs == 4, "2 instances x 2 sides are all one row"
