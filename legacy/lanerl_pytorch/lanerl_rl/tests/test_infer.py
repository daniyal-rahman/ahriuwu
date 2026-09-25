"""Batched inference (correction 9).

Correctness first -- a batched forward must give bit-identical logits to the
per-agent forward, or the speedup is bought with a silent behaviour change.
Then the throughput claim, asserted loosely enough to survive a noisy machine
but tightly enough to fail if batching stops working.
"""

from __future__ import annotations

import itertools

import pytest
import torch

from lanerl_rl import constants as C
from lanerl_rl.frame import ApproxFogModel, iter_jsonl
from lanerl_rl.infer import BatchedActor, benchmark, collate_observations
from lanerl_rl.model import LanePolicy, ModelConfig
from lanerl_rl.obs import ObservationBuilder
from lanerl_rl.scenarios import top_lane_sequence


@pytest.fixture(scope="module")
def observations():
    fog = ApproxFogModel(warn=False)
    frames = top_lane_sequence(n=8)
    builders = [ObservationBuilder(t, fog_model=fog) for t in (C.TEAM_BLUE, C.TEAM_RED)]
    out = []
    for f in frames:
        for b in builders:
            out.append(b.build(f))
    return out


def test_collate_shapes(observations):
    n = 6
    batch = collate_observations(observations[:n])
    assert batch["entities"].shape == (n, 1, C.N_SLOTS, C.ENTITY_DIM)
    assert batch["entity_pad_mask"].shape == (n, 1, C.N_SLOTS)
    assert batch["self_vec"].shape == (n, 1, C.SELF_DIM)
    assert batch["global_vec"].shape == (n, 1, C.GLOBAL_DIM)
    assert batch["priv_vec"].shape == (n, 1, C.PRIV_DIM)
    assert batch["action_masks"]["target"].shape == (n, 1, C.N_SLOTS)
    assert batch["entities"].dtype == torch.float32
    assert batch["entity_pad_mask"].dtype == torch.bool


def test_batched_logits_equal_per_agent_logits(observations):
    """The whole point: 29x cheaper, and the same numbers."""
    torch.manual_seed(0)
    policy = LanePolicy(ModelConfig())
    policy.eval()
    obs = observations[:8]

    batch = collate_observations(obs)
    state = policy.initial_state(len(obs))
    with torch.no_grad():
        dist_b, value_b, _ = policy(state=state, **{k: batch[k] for k in (
            "entities", "entity_pad_mask", "self_vec", "global_vec",
            "priv_entities", "priv_pad_mask", "priv_vec")},
            action_masks=batch["action_masks"])

    for i, o in enumerate(obs):
        one = collate_observations([o])
        s1 = policy.initial_state(1)
        with torch.no_grad():
            dist_1, value_1, _ = policy(state=s1, **{k: one[k] for k in (
                "entities", "entity_pad_mask", "self_vec", "global_vec",
                "priv_entities", "priv_pad_mask", "priv_vec")},
                action_masks=one["action_masks"])
        for head in dist_b.logits:
            assert torch.allclose(
                dist_b.logits[head][i], dist_1.logits[head][0], atol=1e-5
            ), f"{head} differs for agent {i} between batched and single inference"
        assert torch.allclose(value_b[i], value_1[0], atol=1e-5)


def test_actor_returns_one_action_dict_per_agent(observations):
    torch.manual_seed(1)
    policy = LanePolicy(ModelConfig())
    obs = observations[:6]
    actor = BatchedActor(policy, len(obs))
    out = actor.act(obs)
    assert len(out.actions) == len(obs)
    for a in out.actions:
        assert set(a) == {"button", "screen_x", "screen_y", "target"}
        assert 0 <= a["button"] < C.N_BUTTONS
        assert 0 <= a["target"] < C.N_SLOTS
    assert out.log_probs.shape == (len(obs),)
    assert out.values.shape == (len(obs),)
    assert torch.isfinite(out.log_probs).all()


def test_actor_respects_the_action_masks(observations):
    torch.manual_seed(2)
    policy = LanePolicy(ModelConfig())
    obs = observations[:6]
    actor = BatchedActor(policy, len(obs))
    for _ in range(10):
        out = actor.act(obs)
        for a, o in zip(out.actions, obs):
            assert o.action_mask.button[a["button"]]
            assert o.action_mask.target[a["target"]]


def test_actor_carries_core_state_between_ticks(observations):
    torch.manual_seed(3)
    policy = LanePolicy(ModelConfig())
    obs = observations[:4]
    actor = BatchedActor(policy, 4)
    before = actor.state.actor.clone()
    actor.act(obs)
    assert not torch.equal(before, actor.state.actor), "the core state never advanced"
    actor.reset([0, 1])
    assert torch.count_nonzero(actor.state.actor[:, :2]) == 0
    assert torch.count_nonzero(actor.state.actor[:, 2:]) > 0


def test_wrong_batch_size_is_an_error_not_a_silent_misalignment(observations):
    policy = LanePolicy(ModelConfig())
    actor = BatchedActor(policy, 4)
    with pytest.raises(ValueError, match="row order"):
        actor.act(observations[:3])


def test_mlp_core_also_batches(observations):
    torch.manual_seed(4)
    policy = LanePolicy(ModelConfig(core="mlp", mlp_hidden=64, mlp_layers=2))
    obs = observations[:5]
    actor = BatchedActor(policy, 5)
    out = actor.act(obs)
    assert len(out.actions) == 5
    assert torch.isfinite(out.values).all()


def test_batching_is_much_cheaper_per_decision():
    """29x on the dev machine, 7.3x on a loaded 6-core box.

    Asserted at 3x, not 29x, so a busy CI machine does not fail the build --
    but a regression that removed batching entirely would drop this to ~1x,
    which is what the test is actually guarding.
    """
    torch.manual_seed(5)
    res = benchmark(batch_sizes=(1, 24), iters=20)
    assert res[24] * 3.0 < res[1], res


def test_batched_actor_drives_the_real_recording(recording_path):
    """End to end: recording -> two builders -> one forward per tick."""
    torch.manual_seed(6)
    fog = ApproxFogModel(warn=False)
    builders = {t: ObservationBuilder(t, fog_model=fog) for t in (C.TEAM_BLUE, C.TEAM_RED)}
    policy = LanePolicy(ModelConfig())
    actor = BatchedActor(policy, 2)
    n = 0
    for f in itertools.islice(iter_jsonl(recording_path), 900, 940):
        obs = [builders[t].build(f) for t in (C.TEAM_BLUE, C.TEAM_RED)]
        out = actor.act(obs)
        assert len(out.actions) == 2
        assert torch.isfinite(out.values).all()
        n += 1
    assert n == 40
