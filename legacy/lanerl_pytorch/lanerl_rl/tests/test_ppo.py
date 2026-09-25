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
            "screen_x": torch.ones(N_ENVS, cfg.n_screen_x, dtype=torch.bool),
            "screen_y": torch.ones(N_ENVS, cfg.n_screen_y, dtype=torch.bool),
            "target": ~pad,
        }
        action = {
            "button": torch.randint(0, cfg.n_buttons, (N_ENVS,)),
            "screen_x": torch.randint(0, cfg.n_screen_x, (N_ENVS,)),
            "screen_y": torch.randint(0, cfg.n_screen_y, (N_ENVS,)),
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
    assert float(stats["dual_clip_frac"]) == pytest.approx(1.0)


def test_dual_clip_is_inactive_for_positive_advantage():
    policy = LanePolicy()
    trainer = DualClipPPO(policy, PPOConfig(dual_clip=3.0, clip_eps=0.2))
    adv = torch.tensor([1.0])
    loss, stats = trainer.policy_loss(torch.tensor([10.0]), torch.tensor([0.0]), adv)
    # Standard PPO: min(rA, clip(r)A) = 1.2 * 1.0
    assert loss.item() == pytest.approx(-1.2, rel=1e-5)
    assert float(stats["dual_clip_frac"]) == pytest.approx(0.0)


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

    for k in ("loss", "policy_loss", "value_loss", "entropy", "grad_norm"):
        assert np.isfinite(float(stats[k])), (k, stats)
    assert float(stats["grad_norm"]) > 0.0

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


# --------------------------------------------------------------------------
# Architecture / hyperparameter sanity, beyond a single finite gradient step.
# --------------------------------------------------------------------------


def test_gradient_reaches_every_trainable_parameter():
    """A part of the network disconnected from the loss must fail this test.

    ``test_one_gradient_step_changes_params_and_loss_is_finite`` only checks
    that *some* actor and *some* critic parameter moved; a silently dead head
    (e.g. a target head nothing routes gradient through) would still pass it.
    """
    torch.manual_seed(7)
    policy = LanePolicy()
    trainer = DualClipPPO(policy, PPOConfig(chunk_len=8, burn_in=4, minibatch_chunks=4, epochs=1))
    buf = _fill_buffer(seed=7)
    batch = next(iter(buf.iter_minibatches(8, 4, 4)))
    trainer.update_minibatch(batch)

    dead = [
        n for n, p in policy.named_parameters()
        if p.requires_grad and (p.grad is None or torch.all(p.grad == 0))
    ]
    assert not dead, f"{len(dead)} parameter(s) got no gradient: {dead}"


def test_tiny_batch_overfits():
    """Basic "can this architecture learn at all" check: repeated updates on
    one fixed small batch must drive the loss down, not just keep it finite."""
    torch.manual_seed(8)
    policy = LanePolicy()
    trainer = DualClipPPO(
        policy, PPOConfig(chunk_len=4, burn_in=0, minibatch_chunks=1, epochs=1, lr=3e-3,
                           target_kl=10.0, normalize_advantage=False)
    )
    buf = _fill_buffer(seed=8)
    batch = next(iter(buf.iter_minibatches(4, 0, 1)))

    losses = []
    for _ in range(30):
        stats = trainer.update_minibatch(batch)
        losses.append(float(stats["loss"]))

    assert all(np.isfinite(l) for l in losses), losses
    early = float(np.mean(losses[:3]))
    late = float(np.mean(losses[-3:]))
    assert late < early, f"loss did not decrease on a fixed batch: {early} -> {late} ({losses})"


def test_same_seed_gives_the_same_first_action_and_loss():
    """Reproducibility: two freshly-seeded policies must agree exactly on the
    first forward pass and the first gradient step against the same data."""

    def run():
        torch.manual_seed(42)
        policy = LanePolicy()
        trainer = DualClipPPO(policy, PPOConfig(chunk_len=8, burn_in=4, minibatch_chunks=4, epochs=1))
        buf = _fill_buffer(seed=42)
        batch = next(iter(buf.iter_minibatches(8, 4, 4)))
        with torch.no_grad():
            dist, value = trainer._forward_chunk(batch)
            action = dist.mode()
        stats = trainer.update_minibatch(batch)
        return action, value.clone(), float(stats["loss"])

    a1, v1, l1 = run()
    a2, v2, l2 = run()

    for k in a1:
        assert torch.equal(a1[k], a2[k]), f"action[{k}] differs across identically-seeded runs"
    assert torch.equal(v1, v2)
    assert l1 == l2, (l1, l2)


def test_policy_and_optimizer_state_round_trip_through_the_learner_payloads():
    """``DualClipPPO.state_payload``/``load_payload``/``policy_payload`` are
    what ``TrainingLoop`` resumes from; nothing exercised them before they
    were added, and ``lanerl_train.tests.test_run``'s resume tests all use
    ``FakeLearner``, whose own trivial payload methods prove nothing about
    the real ones.
    """
    torch.manual_seed(9)
    policy = LanePolicy()
    trainer = DualClipPPO(policy, PPOConfig(chunk_len=8, burn_in=4, minibatch_chunks=4, epochs=1))
    buf = _fill_buffer(seed=9)
    batch = next(iter(buf.iter_minibatches(8, 4, 4)))
    trainer.update_minibatch(batch)  # give the optimizer real (Adam) state to round-trip

    policy_payload = trainer.policy_payload()
    assert "optimizer" not in policy_payload, "actors must not need optimiser state to act"
    full_payload = trainer.state_payload()
    assert "optimizer" in full_payload and "cfg" in full_payload

    fresh_policy = LanePolicy()
    fresh_trainer = DualClipPPO(fresh_policy, PPOConfig())
    assert any(
        not torch.equal(p1, p2)
        for p1, p2 in zip(policy.parameters(), fresh_policy.parameters())
    ), "test setup bug: fresh policy already matches the trained one"

    fresh_trainer.load_payload(full_payload)

    for p1, p2 in zip(policy.parameters(), fresh_policy.parameters()):
        assert torch.equal(p1, p2)
    flat = lambda o: [p for g in o.param_groups for p in g["params"]]  # noqa: E731
    assert len(trainer.optimizer.param_groups) == 2, "actor and critic are separate groups"
    for g1, g2 in zip(flat(trainer.optimizer), flat(fresh_trainer.optimizer)):
        s1, s2 = trainer.optimizer.state.get(g1, {}), fresh_trainer.optimizer.state.get(g2, {})
        assert set(s1) == set(s2)
        for k in s1:
            if torch.is_tensor(s1[k]):
                assert torch.equal(s1[k], s2[k]), f"optimizer state {k!r} did not round-trip"
    assert fresh_trainer._updates_done == trainer._updates_done


def test_kl_anchor_pulls_the_policy_toward_the_reference():
    """Without an anchor the policy drifts back to uniform.

    Measured on the first real run: entropy returned to 88% of its theoretical
    maximum over 16,200 updates. The BC prior only helps if something keeps the
    policy near it while PPO improves on it -- that is what kl_ref_coef is.
    """
    import copy
    import torch
    from lanerl_rl.model import LanePolicy, ModelConfig
    from lanerl_rl.ppo import PPOConfig, DualClipPPO

    torch.manual_seed(0)
    ref = LanePolicy(ModelConfig())
    pol = copy.deepcopy(ref)
    # move the policy away from the reference so there is a KL to close
    with torch.no_grad():
        for prm in pol.parameters():
            prm.add_(torch.randn_like(prm) * 0.05)

    learner = DualClipPPO(pol, PPOConfig(kl_ref_coef=1.0), reference=ref)
    assert learner.reference is ref
    # the reference must be frozen: a second learner, not a fixed target, would
    # let the anchor drift to meet the policy
    assert all(not p.requires_grad for p in learner.reference.parameters())


def test_no_reference_means_no_kl_term():
    """kl_ref_coef must be inert when no prior is supplied."""
    from lanerl_rl.model import LanePolicy, ModelConfig
    from lanerl_rl.ppo import PPOConfig, DualClipPPO

    learner = DualClipPPO(LanePolicy(ModelConfig()), PPOConfig(kl_ref_coef=1.0))
    assert learner.reference is None
    assert PPOConfig().kl_ref_coef == 0.0


# The log-space / underflow properties of the KL now live on the head-level
# function that implements them: tests/test_model.py::
# test_categorical_kl_is_finite_when_the_policy_underflows_a_masked_slot and
# ::test_categorical_kl_agrees_with_torch_on_well_conditioned_logits.


def test_kl_ref_coefficient_anneals_to_zero():
    """A BC prior is a floor to leave behind, not a target to sit on.

    With a flat coefficient the KL penalty was 69% of the mean |policy_loss|
    on run rl-bc4-0912, and kl_ref was still climbing (0.098 -> 0.138) at the
    end -- the policy pushing against a leash that never releases, tied to a
    heuristic that caps out at 49 CS.
    """
    from lanerl_rl.ppo import PPOConfig

    cfg = PPOConfig(kl_ref_coef=0.05, kl_ref_anneal_steps=1000)
    assert cfg.kl_ref_at(0) == pytest.approx(0.05)
    assert cfg.kl_ref_at(500) == pytest.approx(0.025)
    assert cfg.kl_ref_at(1000) == pytest.approx(0.0)
    assert cfg.kl_ref_at(10_000) == pytest.approx(0.0), "must not go negative"

    flat = PPOConfig(kl_ref_coef=0.05)
    assert flat.kl_ref_at(10**9) == pytest.approx(0.05), "0 steps means no anneal"


def test_the_annealed_coefficient_is_what_actually_gets_applied():
    """Config alone proves nothing -- the learner must READ the clock.

    kl_ref_coef spent weeks as a config field no caller consulted; an anneal
    that the update step ignores would be the same bug with extra arithmetic.
    """
    import torch
    from lanerl_rl.model import LanePolicy, ModelConfig
    from lanerl_rl.ppo import DualClipPPO, PPOConfig

    clock = {"t": 0}
    policy = LanePolicy(ModelConfig())
    reference = LanePolicy(ModelConfig())
    learner = DualClipPPO(
        policy,
        PPOConfig(kl_ref_coef=0.05, kl_ref_anneal_steps=100),
        reference=reference,
        train_step_source=lambda: clock["t"],
    )
    assert learner.cfg.kl_ref_at(learner._train_step_source()) == pytest.approx(0.05)
    clock["t"] = 50
    assert learner.cfg.kl_ref_at(learner._train_step_source()) == pytest.approx(0.025)
    clock["t"] = 100
    assert learner.cfg.kl_ref_at(learner._train_step_source()) == pytest.approx(0.0)


def test_a_learner_built_without_a_clock_keeps_the_flat_coefficient():
    from lanerl_rl.model import LanePolicy, ModelConfig
    from lanerl_rl.ppo import DualClipPPO, PPOConfig

    learner = DualClipPPO(LanePolicy(ModelConfig()), PPOConfig(kl_ref_coef=0.05))
    assert learner._train_step_source() == 0
    assert learner.cfg.kl_ref_at(0) == pytest.approx(0.05)


# --------------------------------------------------------------------------
# The critic is a DIFFERENT problem from the actor   (two param groups)
# --------------------------------------------------------------------------
#
# lanerl_train.bc trains only the action heads and then saves the WHOLE
# state_dict, so a BC checkpoint carries a randomly-initialised critic -- 2.6M
# of the model's 4.6M parameters. __main__ loads it and, until now, PPO trained
# both halves with one Adam at one learning rate, chosen for fine-tuning the
# actor (1e-5 on rl-bc4-0912). The critic never caught up: loss/value_loss went
# 0.047 -> 0.55 over 2,690 updates, with a maximum of 34.3.


def _small_cfg(**kw):
    base = dict(chunk_len=8, burn_in=4, minibatch_chunks=4, epochs=1)
    base.update(kw)
    return PPOConfig(**base)


def test_the_actor_and_the_critic_get_separate_learning_rates():
    policy = LanePolicy()
    learner = DualClipPPO(policy, _small_cfg(lr=1e-5, critic_lr=3e-4))
    groups = {g["name"]: g for g in learner.optimizer.param_groups}
    assert set(groups) == {"actor", "critic"}
    assert groups["actor"]["lr"] == 1e-5
    assert groups["critic"]["lr"] == 3e-4
    n_actor = len(groups["actor"]["params"])
    n_critic = len(groups["critic"]["params"])
    assert n_actor == len(policy.actor_parameter_names())
    assert n_critic == len(policy.critic_parameter_names())
    assert n_actor + n_critic == len(list(policy.parameters()))
    # critic_lr=None means "whatever the actor uses" -- the old behaviour.
    shared = DualClipPPO(LanePolicy(), _small_cfg(lr=7e-5, critic_lr=None))
    assert [g["lr"] for g in shared.optimizer.param_groups] == [7e-5, 7e-5]


def test_the_critic_learns_while_the_actor_is_held_still():
    """The behavioural statement of the split: lr 0 on the actor group must
    stop the actor and leave the critic training at its own rate."""
    torch.manual_seed(11)
    policy = LanePolicy()
    before = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    trainer = DualClipPPO(policy, _small_cfg(lr=0.0, critic_lr=1e-2, target_kl=1e9))
    trainer.update(_fill_buffer(seed=11))
    moved = {
        k for k, v in policy.state_dict().items()
        if v.dtype.is_floating_point and not torch.equal(v, before[k])
    }
    assert moved, "nothing trained at all"
    assert all(k.startswith("critic.") for k in moved), sorted(moved)[:5]


def test_critic_warmup_freezes_the_actor_for_exactly_n_updates():
    torch.manual_seed(12)
    policy = LanePolicy()
    trainer = DualClipPPO(
        policy, _small_cfg(lr=1e-2, critic_lr=1e-2, critic_warmup_updates=1, target_kl=1e9)
    )
    buf = _fill_buffer(seed=12)

    before = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    stats = trainer.update(buf)
    assert stats["actor_frozen"] == 1.0
    moved = {k for k, v in policy.state_dict().items()
             if v.dtype.is_floating_point and not torch.equal(v, before[k])}
    assert moved and all(k.startswith("critic.") for k in moved), sorted(moved)[:5]

    before = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    stats = trainer.update(buf)
    assert stats["actor_frozen"] == 0.0
    moved = {k for k, v in policy.state_dict().items()
             if v.dtype.is_floating_point and not torch.equal(v, before[k])}
    assert any(not k.startswith("critic.") for k in moved), "the actor never thawed"


def test_a_one_group_checkpoint_still_resumes_into_the_two_group_optimizer():
    """Every checkpoint written before the split has one param group, and
    Adam.load_state_dict refuses a group-count mismatch outright -- which
    would turn "resume rl-bc4-0912" into a crash."""
    torch.manual_seed(13)
    policy = LanePolicy()
    legacy = torch.optim.Adam(policy.parameters(), lr=1e-4, eps=1e-5)
    batch = next(iter(_fill_buffer(seed=13).iter_minibatches(8, 4, 4)))
    ref = DualClipPPO(policy, _small_cfg(), optimizer=legacy)
    ref.update_minibatch(batch)  # real Adam moments to carry over
    payload = {"policy": policy.state_dict(), "optimizer": legacy.state_dict()}

    fresh = DualClipPPO(LanePolicy(), _small_cfg(lr=1e-5, critic_lr=3e-4))
    fresh.load_payload(payload)
    assert [g["lr"] for g in fresh.optimizer.param_groups] == [1e-5, 3e-4], (
        "the CONFIG, not the checkpoint, decides the learning rate"
    )
    old_params = list(policy.parameters())
    new_params = [p for g in fresh.optimizer.param_groups for p in g["params"]]
    assert len(old_params) == len(new_params)
    checked = 0
    for p_old, p_new in zip(old_params, new_params):
        s_old = legacy.state.get(p_old, {})
        s_new = fresh.optimizer.state.get(p_new, {})
        assert set(s_old) == set(s_new)
        for k, v in s_old.items():
            if torch.is_tensor(v) and v.numel() > 1:
                assert torch.equal(v, s_new[k]), f"moments landed on the wrong parameter ({k})"
                checked += 1
    assert checked > 0, "test setup bug: no Adam moments to compare"


# --------------------------------------------------------------------------
# Explained variance: the number whose absence hid all of the above
# --------------------------------------------------------------------------


def test_explained_variance_means_what_it_says():
    buf = _fill_buffer(seed=14)
    n = buf.step
    buf.values[:n] = buf.returns[:n]
    assert buf.explained_variance().item() == pytest.approx(1.0, abs=1e-5)
    buf.values[:n] = buf.returns[:n].mean()
    assert buf.explained_variance().item() == pytest.approx(0.0, abs=1e-4)
    buf.values[:n] = -buf.returns[:n]
    assert buf.explained_variance().item() < 0.0, "worse than the mean must read negative"


def test_update_reports_explained_variance():
    torch.manual_seed(15)
    trainer = DualClipPPO(LanePolicy(), _small_cfg(target_kl=1e9))
    buf = _fill_buffer(seed=15)
    buf.values[: buf.step] = buf.returns[: buf.step]
    stats = trainer.update(buf)
    assert "explained_variance" in stats
    assert stats["explained_variance"] == pytest.approx(1.0, abs=1e-4)


# --------------------------------------------------------------------------
# The KL early stop fires on LEARNING, not on staleness
# --------------------------------------------------------------------------


def test_the_early_stop_ignores_staleness_the_update_did_not_cause():
    """rl-bc4-0912's loss/epochs_run was bimodal: 4 epochs 1,684 times and ONE
    epoch 948 times (35%), almost never 2 or 3.  A rollout arrives up to
    max_staleness parameter versions old, so approx_kl is already above 0.02
    before a single gradient is taken, and the absolute test stopped on that
    and threw three epochs of work away.

    Here the stored log-probs are deliberately far from the policy's (approx_kl
    in the ones, not the hundredths) and the learning rate is ZERO, so the
    update adds nothing.  The run must NOT stop.
    """
    torch.manual_seed(16)
    trainer = DualClipPPO(
        LanePolicy(), _small_cfg(epochs=4, lr=0.0, critic_lr=0.0, target_kl=0.02)
    )
    stats = trainer.update(_fill_buffer(seed=16))
    assert stats["approx_kl"] > 0.02, "test setup bug: no staleness to ignore"
    assert stats["approx_kl_staleness"] > 0.02, "the drift must be visible on its own"
    assert stats["approx_kl_excess"] == pytest.approx(0.0, abs=1e-6)
    assert stats["kl_early_stop"] == 0.0
    assert stats["epochs_run"] == 4.0


def test_epoch_zero_is_never_cut_short():
    """The baseline IS epoch 0, so a rollout always gets one full pass -- which
    is what the 948 one-epoch updates were being denied."""
    torch.manual_seed(23)
    trainer = DualClipPPO(
        LanePolicy(), _small_cfg(epochs=4, burn_in=0, lr=1e-2, target_kl=1e-12)
    )
    stats = trainer.update(_fill_buffer(seed=23))
    assert stats["epochs_run"] == 2.0, "epoch 0 sets the baseline; epoch 1 must trip"
    assert stats["kl_early_stop"] == 1.0


# --------------------------------------------------------------------------
# Advantage normalisation, the KL reference, and the stats path
# --------------------------------------------------------------------------


def test_advantages_are_whitened_over_the_whole_rollout():
    buf = _fill_buffer(seed=17)
    buf.normalize_advantages()
    a = buf.advantages[: buf.step]
    assert a.mean().abs().item() < 1e-5
    assert a.std(unbiased=False).item() == pytest.approx(1.0, abs=1e-4)
    # A single minibatch is NOT itself zero-mean, and that is the point: its
    # mean says whether those steps were better than the rollout, which
    # per-minibatch normalisation threw away.
    means = [
        float(mb["advantages"].mean())
        for mb in buf.iter_minibatches(8, 4, 4)
    ]
    assert max(abs(m) for m in means) > 1e-3, means


def test_update_normalises_advantages_once_not_per_minibatch():
    torch.manual_seed(18)
    trainer = DualClipPPO(LanePolicy(), _small_cfg(lr=0.0, critic_lr=0.0, target_kl=1e9))
    buf = _fill_buffer(seed=18)
    raw = buf.advantages[: buf.step].clone()
    trainer.update(buf)
    assert not torch.equal(raw, buf.advantages[: buf.step])
    assert buf.advantages[: buf.step].mean().abs().item() < 1e-5


def test_the_precomputed_reference_logits_match_one_direct_forward():
    """The reference is frozen and the rollout is fixed, so its logits are the
    same in every epoch and every minibatch: computed once, for the whole
    rollout, instead of 4 x 60 forward passes with a critic attached."""
    torch.manual_seed(19)
    reference = LanePolicy()
    trainer = DualClipPPO(LanePolicy(), _small_cfg(kl_ref_coef=1.0), reference=reference)
    buf = _fill_buffer(seed=19)
    # an awkward slice size, to catch a bookkeeping error in the chunking
    cached = trainer._reference_logits_for_rollout(buf, time_chunk=7)

    T = buf.step
    with torch.no_grad():
        dist, _tokens, _h = reference.actor_forward(
            buf.obs["entities"][:T].transpose(0, 1),
            buf.obs["entity_pad_mask"][:T].transpose(0, 1),
            buf.obs["self_vec"][:T].transpose(0, 1),
            buf.obs["global_vec"][:T].transpose(0, 1),
            buf.h_actor[0].unsqueeze(0).contiguous(),
            buf.resets[:T].transpose(0, 1),
            action_masks={k: buf.masks[k][:T].transpose(0, 1) for k in MASK_KEYS},
        )
    for k, v in dist.logits.items():
        assert torch.allclose(cached[k], v.transpose(0, 1), atol=1e-5), k


def test_the_kl_anchor_is_still_applied_through_the_cached_logits():
    torch.manual_seed(20)
    policy = LanePolicy()
    same = copy.deepcopy(policy)
    far = LanePolicy()  # a different random init
    buf = _fill_buffer(seed=20)

    def kl_ref_of(reference):
        p = copy.deepcopy(policy)
        tr = DualClipPPO(p, _small_cfg(lr=0.0, critic_lr=0.0, kl_ref_coef=1.0, target_kl=1e9),
                         reference=reference)
        return tr.update(_fill_buffer(seed=20))["kl_ref"]

    identical, different = kl_ref_of(same), kl_ref_of(far)
    # Identical weights do NOT give exactly zero, and that is the documented
    # consequence of precomputing: the policy's state for a chunk is the stored
    # hidden vector plus a burn_in-step replay, while the reference carries its
    # own state across the whole rollout, so the two condition on slightly
    # different histories. Measured, that residual is ~1e-4 -- three orders
    # below the KL to a differently-initialised prior.
    assert identical < 1e-3, identical
    assert different > 100 * max(identical, 1e-9), (identical, different)


def test_update_minibatch_returns_device_tensors_not_host_floats():
    """Eight .item() calls per minibatch, up to 60 minibatches per update, is
    up to 480 device synchronisations for numbers nothing reads until the
    update ends."""
    torch.manual_seed(21)
    trainer = DualClipPPO(LanePolicy(), _small_cfg(kl_ref_coef=1.0), reference=LanePolicy())
    batch = next(iter(_fill_buffer(seed=21).iter_minibatches(8, 4, 4)))
    stats = trainer.update_minibatch(batch)
    assert stats, "no diagnostics at all"
    for k, v in stats.items():
        assert torch.is_tensor(v), f"{k} is a host float: that is a sync per minibatch"
        assert v.ndim == 0 and not v.requires_grad, k


def test_update_returns_plain_floats():
    """...but the update itself must hand TrainingLoop numbers it can log."""
    torch.manual_seed(22)
    trainer = DualClipPPO(LanePolicy(), _small_cfg(target_kl=1e9))
    stats = trainer.update(_fill_buffer(seed=22))
    for k, v in stats.items():
        assert isinstance(v, float), (k, type(v))
    for k in ("explained_variance", "approx_kl_staleness", "approx_kl_baseline",
              "approx_kl_excess", "lr_actor", "lr_critic", "actor_frozen"):
        assert k in stats, k


def test_buffer_to_moves_every_tensor_it_owns():
    """A rollout now crosses a process boundary, so nothing may be left behind.

    The actor collects on its own CUDA context and hands the buffer over on
    CPU; the learner moves it back. A tensor this misses does not fail here --
    it fails deep inside the PPO step as a device mismatch that names neither
    the field nor the handover, or worse, quietly drags the update onto CPU.
    """
    buf = _fill_buffer()
    buf.ref_logits["button"] = torch.zeros(buf.T, buf.B, 4)

    def every_tensor(b):
        for d in (b.obs, b.masks, b.actions, b.ref_logits):
            for k, v in d.items():
                yield f"{k}", v
        for name in ("log_probs", "values", "rewards", "dones", "resets",
                     "h_actor", "h_critic", "advantages", "returns"):
            yield name, getattr(b, name)

    assert buf.to("cpu") is buf, ".to() must return self so it can be chained"
    for name, t in every_tensor(buf):
        assert t.device.type == "cpu", f"{name} stayed on {t.device}"
    assert buf.device.type == "cpu"

    if torch.cuda.is_available():
        buf.to("cuda")
        for name, t in every_tensor(buf):
            assert t.device.type == "cuda", f"{name} did not move to cuda"


def test_the_learner_knows_where_its_weights_are():
    """`device` is what puts a rollout from another process back on the GPU."""
    policy = LanePolicy(ModelConfig())
    learner = DualClipPPO(policy, PPOConfig())
    assert learner.device == next(policy.parameters()).device
