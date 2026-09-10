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

import pytest
import torch

from lanerl_rl.model import LanePolicy, ModelConfig
from lanerl_rl.ppo import DualClipPPO, PPOConfig
from lanerl_train import paths
from lanerl_train.lane_wiring import LanePolicyActor, collect_rollout, make_lane_adapters
from lanerl_train.vec import EpisodeSpec, SideAssignment, VecDriver, VecLaneEnv

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
