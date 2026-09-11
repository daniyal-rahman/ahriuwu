"""End to end: one real server instance, the real policy, the real PPO
learner, wired through ``lanerl_train.lane_wiring`` -- never ``FakeInstance``.

Before this, ``TrainingLoop`` (``lanerl_train/tests/test_run.py``) and the
real environment (``lanerl_bot``'s CS baseline, ``lanerl_rl``'s control smoke
test) were each proven independently, but nothing had ever driven the real
server, the real observation/action pipeline, the real policy, and a real PPO
update in the same process. That is exactly where an observation-shape
mismatch, a policy whose gradient never reaches part of the network, or a
reward wired to the wrong frame would hide -- each piece looks fine alone.

The reward wiring was independently verified by hand to fire correctly at
t=90.1s (League's real passive-gold timer) over a ~1350-decision run; this
test does not reproduce that (it would make every CI run ~90 real seconds
slower for a fact already established), and instead asserts the mechanics
that must hold on *any* window: a finite loss and a gradient that reaches
every parameter tensor.
"""

from __future__ import annotations

import math
import re

import pytest
import torch

from lanerl_rl import constants as C
from lanerl_rl.model import LanePolicy, ModelConfig
from lanerl_rl.ppo import DualClipPPO, PPOConfig
from lanerl_train import paths
from lanerl_train.lane_wiring import LanePolicyActor, collect_rollout, make_lane_adapters
from lanerl_train.vec import (
    EpisodeSpec,
    ServerLaunchSpec,
    SideAssignment,
    VecDriver,
    VecLaneEnv,
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not paths.server_available(),
        reason="server build not available (or LANERL_SKIP_SERVER_TESTS=1)",
    ),
]

SELF = "self"


@pytest.fixture
def real_driver(tmp_path):
    policy = LanePolicy(ModelConfig())
    actor = LanePolicyActor(policy)
    adapters = make_lane_adapters(train_step_source=lambda: 0)
    env = VecLaneEnv(n=1, log_dir=tmp_path / "logs")
    driver = VecDriver(
        env=env,
        policies={SELF: actor},
        adapter_factory=adapters.adapter_factory,
        encoder=adapters.encoder,
        assignments=[SideAssignment(blue=SELF, red=SELF)],
        episode=EpisodeSpec(max_game_ms=600_000),
    )
    driver.start()
    try:
        yield driver, policy, actor, adapters
    finally:
        env.close()


def test_real_rollout_feeds_a_finite_ppo_update_that_touches_every_parameter(real_driver):
    driver, policy, actor, adapters = real_driver
    ppo = DualClipPPO(policy, PPOConfig(chunk_len=4, burn_in=0, minibatch_chunks=2, epochs=1))

    rollout = collect_rollout(
        driver, actor, adapters.reward_contexts, SELF,
        num_steps=16, gamma=0.99, gae_lambda=0.95,
    )
    assert rollout.steps > 0, "collect_rollout produced an empty buffer against a real server"

    before = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    stats = ppo.update(rollout.data)

    assert math.isfinite(stats["loss"]), f"loss is not finite: {stats}"
    assert math.isfinite(stats["grad_norm"]), f"grad norm is not finite: {stats}"

    unchanged = [
        k for k, v in policy.state_dict().items()
        if torch.equal(v, before[k]) and before[k].numel() > 0
    ]
    assert not unchanged, (
        f"{len(unchanged)}/{len(before)} parameter tensors did NOT change after a real "
        f"PPO update -- the loss is not reaching them: {unchanged[:5]}"
    )


def test_real_rollout_reward_alignment_does_not_crash_across_a_reset(real_driver):
    """A short rollout long enough to hit the server's own episode boundary
    logic at least once should not desync the buffer or crash the collector.

    ``EpisodeSpec.max_steps`` default is 20,000 decisions, so this does not
    naturally hit a real episode end; it exists to catch a regression in the
    reset-flag/off-by-one plumbing (`InstanceRewardContext`, `_pending_resets`
    snapshot timing) rather than to force an actual death/timeout boundary,
    which real_driver's short window cannot reach.
    """
    driver, policy, actor, adapters = real_driver
    rollout = collect_rollout(
        driver, actor, adapters.reward_contexts, SELF,
        num_steps=24, gamma=0.99, gae_lambda=0.95,
    )
    assert rollout.steps == 23  # num_steps - 1: see collect_rollout's docstring
    assert rollout.data.rewards.shape[0] == rollout.steps


# -- episode boundaries, against the server's own reset log ----------------

RESET_RE = re.compile(r"LANERL_RESET .*t_before=(\d+) t_after=(\d+)")

#: Short enough that three of them run in seconds of wall clock, long enough
#: that a spurious boundary is unmistakable: 20 game-seconds is 600 decisions
#: at ``C.DECISION_HZ``.
SHORT_EPISODE_MS = 20_000


def test_recorded_episodes_match_the_servers_own_resets(tmp_path):
    """The claim the audit made, decided against the server instead of inferred.

    The audit of ``runs/rl-overnight-0911-0608`` read 1,372 of 1,670 completed
    episodes as having lasted under 1,000 decisions and concluded that the
    in-process reset was failing to rewind the clock, so ``episode_done``
    re-fired "time" on every following step.  It could not run a server to
    check.  This is that check: drive a real one with a deliberately tiny
    ``max_game_ms``, then compare the episodes the collector recorded against
    the ``LANERL_RESET`` lines the server printed for itself.

    Two numbers have to agree and one has to be plausible:

    * one recorded episode per server-side reset -- a boundary the trainer
      invented would show up here as an extra episode with no reset behind it,
      and a boundary it missed as a reset with no episode;
    * every episode's decision count within a decision of
      ``max_game_ms / (1000 / DECISION_HZ)``.  A re-fire produces episodes of
      1-3 decisions, which is exactly the shape the audit reported.
    """
    log_dir = tmp_path / "logs"
    env = VecLaneEnv(n=1, spec=ServerLaunchSpec(bot_teams="both"), log_dir=log_dir)
    policy = LanePolicy(
        ModelConfig(core_dim=32, d_model=32, ffn_dim=32, n_layers=1, n_heads=2, mlp_hidden=32)
    )
    actor = LanePolicyActor(policy)
    adapters = make_lane_adapters(train_step_source=lambda: 0)
    driver = VecDriver(
        env=env,
        policies={SELF: actor},
        adapter_factory=adapters.adapter_factory,
        encoder=adapters.encoder,
        assignments=[SideAssignment(blue=SELF, red=SELF)],
        episode=EpisodeSpec(max_game_ms=SHORT_EPISODE_MS, max_steps=10**9),
    )
    driver.start()
    episodes = []
    try:
        # Long enough for three episodes plus slack; stops as soon as it has them.
        for _ in range(12):
            rollout = collect_rollout(
                driver, actor, adapters.reward_contexts, SELF,
                num_steps=256, gamma=0.99, gae_lambda=0.95,
            )
            episodes.extend(rollout.episodes)
            if len(episodes) >= 3:
                break
    finally:
        env.close()

    text = (log_dir / "instance000.log").read_text(errors="replace")
    resets = RESET_RE.findall(text)
    assert len(resets) >= 3, f"the server logged only {len(resets)} resets"
    for t_before, t_after in resets:
        assert int(t_before) >= SHORT_EPISODE_MS, (
            f"the server reset at t={t_before}, before the episode was over"
        )
        assert int(t_after) == 0, f"the clock did not rewind: t_after={t_after}"

    assert len(episodes) == len(resets), (
        f"{len(episodes)} episodes recorded against {len(resets)} server-side resets; "
        f"a mismatch is either an invented boundary or a dropped one"
    )
    expected = round(SHORT_EPISODE_MS / (1000.0 / C.DECISION_HZ))
    for ep in episodes:
        assert ep.reason == "time", f"unexpected end reason {ep.reason!r}"
        assert abs(ep.length_steps - expected) <= 2, (
            f"episode ran {ep.length_steps} decisions, expected ~{expected}; "
            f"all lengths: {[e.length_steps for e in episodes]}"
        )


def test_the_terminal_frame_is_the_one_past_the_limit_not_the_reset_frame(tmp_path):
    """The frame CS@10 and the final reward are read from, against a real server."""
    log_dir = tmp_path / "logs"
    env = VecLaneEnv(n=1, spec=ServerLaunchSpec(bot_teams="both"), log_dir=log_dir)
    policy = LanePolicy(
        ModelConfig(core_dim=32, d_model=32, ffn_dim=32, n_layers=1, n_heads=2, mlp_hidden=32)
    )
    actor = LanePolicyActor(policy)
    adapters = make_lane_adapters(train_step_source=lambda: 0)
    driver = VecDriver(
        env=env,
        policies={SELF: actor},
        adapter_factory=adapters.adapter_factory,
        encoder=adapters.encoder,
        assignments=[SideAssignment(blue=SELF, red=SELF)],
        episode=EpisodeSpec(max_game_ms=SHORT_EPISODE_MS, max_steps=10**9),
    )
    driver.start()
    try:
        seen = None
        for _ in range(800):
            result, dones = driver.step()
            if dones:
                seen = result
                break
    finally:
        env.close()
    assert seen is not None, "no episode boundary in 800 decisions"
    assert 0 in seen.terminal_obs
    assert int(seen.terminal_obs[0]["t"]) >= SHORT_EPISODE_MS
    # ...and what the driver hands back for the next decision is the other one.
    assert int(seen.obs[0]["t"]) < SHORT_EPISODE_MS
    assert int(driver.env.last_obs[0]["t"]) < SHORT_EPISODE_MS
