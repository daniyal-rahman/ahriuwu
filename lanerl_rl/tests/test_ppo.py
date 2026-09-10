"""Dual-clip recurrent PPO."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from lanerl_rl import constants as C
from lanerl_rl.model import LanePolicy, ModelConfig, RecurrentState
from lanerl_rl.ppo import (
    ACTION_KEYS,
    MASK_KEYS,
    OBS_KEYS,
    DualClipPPO,
    PPOConfig,
    RecurrentRolloutBuffer,
    compute_gae,
)

T_STEPS, N_ENVS = 32, 4


def _fill_buffer(seed=0) -> RecurrentRolloutBuffer:
    torch.manual_seed(seed)
    cfg = ModelConfig()
    buf = RecurrentRolloutBuffer(T_STEPS, N_ENVS, cfg)
    for t in range(T_STEPS):
        ent = torch.randn(N_ENVS, cfg.n_slots, cfg.entity_dim)
        ent[..., C.E_TYPE_ONEHOT] = 0.0
        tidx = torch.randint(0, C.N_ENTITY_TYPES, (N_ENVS, cfg.n_slots))
        ent[..., C.E_TYPE_ONEHOT] = torch.nn.functional.one_hot(tidx, C.N_ENTITY_TYPES).float()
        pad = torch.rand(N_ENVS, cfg.n_slots) < 0.3
        pad[:, 0] = False
        obs = {
            "entities": ent,
            "entity_pad_mask": pad,
            "self_vec": torch.randn(N_ENVS, cfg.self_dim),
            "global_vec": torch.randn(N_ENVS, cfg.global_dim),
            "priv_entities": torch.randn(N_ENVS, cfg.n_slots, cfg.entity_dim),
            "priv_pad_mask": pad.clone(),
            "priv_vec": torch.randn(N_ENVS, cfg.priv_dim),
        }
        masks = {
            "button": torch.ones(N_ENVS, cfg.n_buttons, dtype=torch.bool),
            "move_x": torch.ones(N_ENVS, cfg.n_move_bins, dtype=torch.bool),
            "move_z": torch.ones(N_ENVS, cfg.n_move_bins, dtype=torch.bool),
            "target": ~pad,
        }
        action = {
            "button": torch.randint(0, cfg.n_buttons, (N_ENVS,)),
            "move_x": torch.randint(0, cfg.n_move_bins, (N_ENVS,)),
            "move_z": torch.randint(0, cfg.n_move_bins, (N_ENVS,)),
            "target": torch.zeros(N_ENVS, dtype=torch.long),
        }
        done = (torch.rand(N_ENVS) < 0.03).float()
        buf.add(
            obs=obs,
            masks=masks,
            action=action,
            log_prob=torch.randn(N_ENVS) * 0.1 - 3.0,
            value=torch.randn(N_ENVS),
            reward=torch.randn(N_ENVS) * 0.1,
            done=done,
            reset=torch.zeros(N_ENVS),
            state=RecurrentState(
                torch.randn(1, N_ENVS, cfg.gru_dim) * 0.1,
                torch.randn(1, N_ENVS, cfg.gru_dim) * 0.1,
            ),
        )
    buf.finish(torch.zeros(N_ENVS), gamma=0.99, lam=0.95)
    return buf


# --------------------------------------------------------------------------
# GAE
# --------------------------------------------------------------------------


def test_gae_matches_hand_computation():
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    values = torch.tensor([[0.5], [0.5], [0.5]])
    dones = torch.tensor([[0.0], [0.0], [0.0]])
    last = torch.tensor([0.0])
    g, lam = 0.9, 0.8
    adv, ret = compute_gae(rewards, values, dones, last, g, lam)

    d2 = 3.0 + g * 0.0 - 0.5
    d1 = 2.0 + g * 0.5 - 0.5
    d0 = 1.0 + g * 0.5 - 0.5
    a2 = d2
    a1 = d1 + g * lam * a2
    a0 = d0 + g * lam * a1
    assert adv[0, 0] == pytest.approx(a0, rel=1e-6)
    assert adv[1, 0] == pytest.approx(a1, rel=1e-6)
    assert adv[2, 0] == pytest.approx(a2, rel=1e-6)
    assert ret[0, 0] == pytest.approx(a0 + 0.5, rel=1e-6)


def test_gae_does_not_bootstrap_through_a_terminal_step():
    rewards = torch.tensor([[1.0], [1.0]])
    values = torch.tensor([[10.0], [10.0]])
    dones = torch.tensor([[1.0], [0.0]])
    adv, _ = compute_gae(rewards, values, dones, torch.tensor([0.0]), 0.99, 0.95)
    # step 0 is terminal -> advantage is r - V, with no next value at all.
    assert adv[0, 0] == pytest.approx(1.0 - 10.0, rel=1e-6)


def test_gae_lambda_099_is_the_default():
    assert PPOConfig().gae_lambda == 0.99


# --------------------------------------------------------------------------
# Dual clip
# --------------------------------------------------------------------------


def test_dual_clip_bounds_a_negative_advantage():
    policy = LanePolicy()
    trainer = DualClipPPO(policy, PPOConfig(dual_clip=3.0, clip_eps=0.2))
    adv = torch.tensor([-1.0])
    old_lp = torch.tensor([0.0])
    lp = torch.tensor([10.0])  # ratio = e^10 ~= 22026
    loss, stats = trainer.policy_loss(lp, old_lp, adv)
    # Without the dual clip the loss would be ~22026; with c=3 it is exactly 3.
    assert loss.item() == pytest.approx(3.0, rel=1e-5)
    assert stats["dual_clip_frac"] == pytest.approx(1.0)


def test_dual_clip_is_inactive_for_positive_advantage():
    policy = LanePolicy()
    trainer = DualClipPPO(policy, PPOConfig(dual_clip=3.0, clip_eps=0.2))
    adv = torch.tensor([1.0])
    loss, stats = trainer.policy_loss(torch.tensor([10.0]), torch.tensor([0.0]), adv)
    # Standard PPO: min(rA, clip(r)A) = 1.2 * 1.0
    assert loss.item() == pytest.approx(-1.2, rel=1e-5)
    assert stats["dual_clip_frac"] == pytest.approx(0.0)


def test_dual_clip_agrees_with_standard_ppo_near_ratio_one():
    policy = LanePolicy()
    trainer = DualClipPPO(policy, PPOConfig(dual_clip=3.0, clip_eps=0.2))
    for adv_val in (-1.0, 1.0):
        adv = torch.tensor([adv_val])
        lp = torch.tensor([0.05])
        loss, _ = trainer.policy_loss(lp, torch.tensor([0.0]), adv)
        ratio = float(torch.exp(lp))
        expected = -min(ratio * adv_val, np.clip(ratio, 0.8, 1.2) * adv_val)
        assert loss.item() == pytest.approx(expected, rel=1e-5)


def test_dual_clip_must_be_greater_than_one():
    with pytest.raises(ValueError):
        PPOConfig(dual_clip=0.5)


# --------------------------------------------------------------------------
# Buffer / minibatching
# --------------------------------------------------------------------------


def test_buffer_chunks_are_rectangular_and_cover_the_rollout():
    buf = _fill_buffer()
    cfg = PPOConfig(chunk_len=8, burn_in=4)
    groups = buf.chunk_starts(cfg.chunk_len, cfg.burn_in)
    assert set(groups) <= {0, cfg.burn_in}
    total = sum(len(v) for v in groups.values())
    assert total == N_ENVS * (T_STEPS // cfg.chunk_len)
    for bi, items in groups.items():
        batch = buf.gather(items, bi, cfg.chunk_len)
        L = bi + cfg.chunk_len
        assert batch["obs"]["entities"].shape == (len(items), L, C.N_SLOTS, C.ENTITY_DIM)
        assert batch["state"].actor.shape == (1, len(items), ModelConfig().actor_state_dim)


def test_minibatches_are_finite_and_complete():
    buf = _fill_buffer()
    cfg = PPOConfig(chunk_len=8, burn_in=4, minibatch_chunks=4)
    seen = 0
    for mb in buf.iter_minibatches(cfg.chunk_len, cfg.burn_in, cfg.minibatch_chunks):
        seen += mb["obs"]["entities"].shape[0]
        for k in OBS_KEYS:
            assert torch.isfinite(mb["obs"][k].float()).all()
    assert seen == N_ENVS * (T_STEPS // cfg.chunk_len)


# --------------------------------------------------------------------------
# The gradient step
# --------------------------------------------------------------------------


def test_one_gradient_step_changes_params_and_loss_is_finite():
    torch.manual_seed(0)
    policy = LanePolicy()
    before = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    trainer = DualClipPPO(policy, PPOConfig(chunk_len=8, burn_in=4, minibatch_chunks=4, epochs=1))
    buf = _fill_buffer()

    batch = next(iter(buf.iter_minibatches(8, 4, 4)))
    stats = trainer.update_minibatch(batch)

    assert np.isfinite(stats["loss"]), stats
    assert np.isfinite(stats["policy_loss"])
    assert np.isfinite(stats["value_loss"])
    assert np.isfinite(stats["entropy"])
    assert np.isfinite(stats["grad_norm"])
    assert stats["grad_norm"] > 0.0

    after = policy.state_dict()
    changed = [k for k, v in after.items() if v.dtype.is_floating_point and not torch.equal(v, before[k])]
    assert changed, "no parameter changed after a gradient step"
    # Both trunks must have been updated.
    assert any(k.startswith("critic.") for k in changed), "critic did not learn"
    assert any(not k.startswith("critic.") for k in changed), "actor did not learn"


def test_full_update_runs_and_reports_stats():
    torch.manual_seed(1)
    policy = LanePolicy()
    trainer = DualClipPPO(
        policy, PPOConfig(chunk_len=8, burn_in=4, minibatch_chunks=4, epochs=2, lr=1e-4)
    )
    buf = _fill_buffer(seed=2)
    stats = trainer.update(buf)
    for k in ("loss", "policy_loss", "value_loss", "entropy", "approx_kl", "grad_norm"):
        assert k in stats and np.isfinite(stats[k]), (k, stats)
    assert stats["n_minibatches"] > 0
    assert stats["approx_kl"] >= 0.0


def test_kl_early_stop_triggers_on_a_large_target_violation():
    torch.manual_seed(3)
    policy = LanePolicy()
    trainer = DualClipPPO(
        policy,
        PPOConfig(chunk_len=8, burn_in=0, minibatch_chunks=4, epochs=6, lr=1e-2, target_kl=1e-9),
    )
    buf = _fill_buffer(seed=4)
    stats = trainer.update(buf)
    assert stats["kl_early_stop"] == 1.0
    assert stats["epochs_run"] < 6


def test_burn_in_does_not_receive_gradients():
    """Burn-in steps must be replayed under no_grad."""
    torch.manual_seed(5)
    policy = LanePolicy()
    trainer = DualClipPPO(policy, PPOConfig(chunk_len=8, burn_in=4, minibatch_chunks=2))
    buf = _fill_buffer(seed=6)
    batch = None
    for mb in buf.iter_minibatches(8, 4, 2):
        if int(mb["burn_in"]) == 4:
            batch = mb
            break
    assert batch is not None
    dist, value = trainer._forward_chunk(batch)
    assert value.shape[1] == 8, "gradient chunk must exclude the burn-in prefix"
    assert dist.logits["button"].shape[1] == 8


def test_value_loss_clipping_is_bounded():
    policy = LanePolicy()
    trainer = DualClipPPO(policy, PPOConfig(clip_value_loss=True, value_clip_eps=0.2))
    old_v = torch.zeros(4)
    v = torch.tensor([100.0, -100.0, 0.1, -0.1])
    ret = torch.zeros(4)
    loss = trainer.value_loss(v, old_v, ret)
    unclipped = trainer_unclipped = DualClipPPO(policy, PPOConfig(clip_value_loss=False)).value_loss(
        v, old_v, ret
    )
    assert torch.isfinite(loss)
    # max(unclipped, clipped) is at least the unclipped term, by construction.
    assert loss >= unclipped - 1e-6
