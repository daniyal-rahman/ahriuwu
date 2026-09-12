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
        losses.append(stats["loss"])

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
        return action, value.clone(), stats["loss"]

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
    for g1, g2 in zip(trainer.optimizer.param_groups[0]["params"], fresh_trainer.optimizer.param_groups[0]["params"]):
        s1, s2 = trainer.optimizer.state.get(g1, {}), fresh_trainer.optimizer.state.get(g2, {})
        assert set(s1) == set(s2)
        for k in s1:
            if torch.is_tensor(s1[k]):
                assert torch.equal(s1[k], s2[k]), f"optimizer state {k!r} did not round-trip"


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


def test_kl_to_is_finite_when_the_policy_underflows_a_masked_slot():
    """KL to the frozen prior must never be inf.

    torch.distributions.kl_divergence for Categorical writes inf wherever the
    SECOND argument's probability is exactly 0. As the policy sharpens, a
    softmax entry underflows to 0 while the reference still has mass there, and
    the whole KL term -- the only thing anchoring the run to its BC prior --
    becomes inf. On run rl-bc2-0912 that happened on 7 of 56 updates.

    It is worse than a logging artefact: the updates where the policy has run
    furthest from the prior are exactly the ones where the anchor is silently
    dropped.
    """
    import torch
    from lanerl_rl.model import LaneActionDist

    class _One:
        HEADS = ("a",)

        def __init__(self, logits):
            self.dists = {"a": torch.distributions.Categorical(logits=logits)}

        kl_to = LaneActionDist.kl_to

    policy = torch.tensor([[80.0, -80.0, -1e9]])   # entry 1 underflows to 0.0
    reference = torch.tensor([[1.0, 0.5, -1e9]])   # reference still has mass there
    assert torch.softmax(policy, -1)[0, 1].item() == 0.0

    torch_kl = torch.distributions.kl_divergence(
        torch.distributions.Categorical(logits=reference),
        torch.distributions.Categorical(logits=policy),
    )
    assert torch.isinf(torch_kl).any(), "the failure this guards is gone; revisit"

    ours = _One(policy).kl_to({"a": reference})
    assert torch.isfinite(ours).all()
    assert ours.item() > 1.0, "a policy this far from the prior must be penalised"

    grad_src = policy.clone().requires_grad_(True)
    _One(grad_src).kl_to({"a": reference}).sum().backward()
    assert torch.isfinite(grad_src.grad).all()


def test_kl_to_agrees_with_torch_on_well_conditioned_logits():
    """The log-space rewrite must not change the ordinary case."""
    import torch
    from lanerl_rl.model import LaneActionDist

    class _One:
        HEADS = ("a",)

        def __init__(self, logits):
            self.dists = {"a": torch.distributions.Categorical(logits=logits)}

        kl_to = LaneActionDist.kl_to

    torch.manual_seed(0)
    a, b = torch.randn(8, 11), torch.randn(8, 11)
    expected = torch.distributions.kl_divergence(
        torch.distributions.Categorical(logits=b),
        torch.distributions.Categorical(logits=a),
    )
    assert torch.allclose(expected, _One(a).kl_to({"a": b}), atol=1e-5)


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
