"""The policy network: shapes, masking, permutation equivariance, asymmetry."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lanerl_rl import constants as C
from lanerl_rl.model import (
    USES_MOVE_HEAD,
    USES_TARGET_HEAD,
    LaneActionDist,
    LanePolicy,
    ModelConfig,
    RecurrentState,
    apply_action_mask,
    assert_actor_critic_disjoint,
    categorical_kl,
    frame_stack_with_resets,
    masked_pool,
)
from lanerl_rl.obs import ObservationBuilder
from lanerl_rl.scenarios import top_lane_sequence

B, T = 3, 5


@pytest.fixture(scope="module")
def policy():
    torch.manual_seed(0)
    return LanePolicy(ModelConfig())


def _random_batch(b=B, t=T, seed=0, all_valid=False):
    g = torch.Generator().manual_seed(seed)
    c = ModelConfig()
    ent = torch.randn(b, t, c.n_slots, c.entity_dim, generator=g)
    # Force a legal one-hot in the type block so the type embedding index is sane.
    ent[..., C.E_TYPE_ONEHOT] = 0.0
    tidx = torch.randint(0, C.N_ENTITY_TYPES, (b, t, c.n_slots), generator=g)
    ent[..., C.E_TYPE_ONEHOT] = torch.nn.functional.one_hot(tidx, C.N_ENTITY_TYPES).float()
    if all_valid:
        pad = torch.zeros(b, t, c.n_slots, dtype=torch.bool)
    else:
        pad = torch.rand(b, t, c.n_slots, generator=g) < 0.3
        pad[..., 0] = False  # keep at least one valid slot
    return {
        "entities": ent,
        "entity_pad_mask": pad,
        "self_vec": torch.randn(b, t, c.self_dim, generator=g),
        "global_vec": torch.randn(b, t, c.global_dim, generator=g),
        "priv_entities": torch.randn(b, t, c.n_slots, c.entity_dim, generator=g),
        "priv_pad_mask": pad.clone(),
        "priv_vec": torch.randn(b, t, c.priv_dim, generator=g),
    }


def test_forward_shapes(policy):
    batch = _random_batch()
    state = policy.initial_state(B)
    dist, value, new_state = policy(state=state, **batch)
    assert dist.logits["button"].shape == (B, T, C.N_BUTTONS)
    assert dist.logits["screen_x"].shape == (B, T, C.N_SCREEN_X)
    assert dist.logits["screen_y"].shape == (B, T, C.N_SCREEN_Y)
    assert dist.logits["target"].shape == (B, T, C.N_SLOTS)
    assert value.shape == (B, T)
    assert new_state.actor.shape == (1, B, policy.cfg.gru_dim)
    assert new_state.critic.shape == (1, B, policy.cfg.gru_dim)
    assert torch.isfinite(value).all()
    for v in dist.logits.values():
        assert torch.isfinite(v).all()


def test_tokens_are_prepool_and_correctly_shaped(policy):
    batch = _random_batch()
    state = policy.initial_state(B)
    dist, tokens, _ = policy.actor_forward(
        batch["entities"],
        batch["entity_pad_mask"],
        batch["self_vec"],
        batch["global_vec"],
        state.actor,
    )
    assert tokens.shape == (B, T, C.N_SLOTS, policy.cfg.d_model)
    assert dist.logits["target"].shape[-1] == C.N_SLOTS


def test_target_head_is_permutation_equivariant(policy):
    """Swapping two slots must swap their target logits and change nothing else.

    This is the real test that the target head reads the PRE-POOL tokens: a head
    computed from the pooled vector alone could not track a slot swap, and a
    head that had a slot-index feature would not be equivariant.
    """
    batch = _random_batch(b=1, t=1, seed=7, all_valid=True)
    state = policy.initial_state(1)
    with torch.no_grad():
        d0, _, _ = policy.actor_forward(
            batch["entities"], batch["entity_pad_mask"], batch["self_vec"],
            batch["global_vec"], state.actor,
        )
        perm = list(range(C.N_SLOTS))
        i, j = 3, 11
        perm[i], perm[j] = perm[j], perm[i]
        ent2 = batch["entities"][:, :, perm]
        pad2 = batch["entity_pad_mask"][:, :, perm]
        d1, _, _ = policy.actor_forward(
            ent2, pad2, batch["self_vec"], batch["global_vec"], state.actor
        )

    t0 = d0.logits["target"][0, 0]
    t1 = d1.logits["target"][0, 0]
    assert torch.allclose(t0[perm], t1, atol=1e-5), "target logits are not permutation equivariant"
    for head in ("button", "screen_x", "screen_y"):
        assert torch.allclose(d0.logits[head], d1.logits[head], atol=1e-5), (
            f"{head} logits changed under a slot permutation"
        )


def test_action_masking_zeroes_illegal_choices(policy):
    batch = _random_batch()
    state = policy.initial_state(B)
    masks = {
        "button": torch.zeros(B, T, C.N_BUTTONS, dtype=torch.bool),
        "screen_x": torch.ones(B, T, C.N_SCREEN_X, dtype=torch.bool),
        "screen_y": torch.ones(B, T, C.N_SCREEN_Y, dtype=torch.bool),
        "target": ~batch["entity_pad_mask"],
    }
    masks["button"][..., 0] = True
    masks["button"][..., 3] = True
    dist, _, _ = policy(state=state, action_masks=masks, **batch)
    probs = torch.softmax(dist.logits["button"], dim=-1)
    illegal = ~masks["button"]
    assert probs[illegal].max() < 1e-6
    tprobs = torch.softmax(dist.logits["target"], dim=-1)
    assert tprobs[batch["entity_pad_mask"]].max() < 1e-6
    for _ in range(20):
        a = dist.sample()
        assert bool(masks["button"].gather(-1, a["button"].unsqueeze(-1)).all())


def test_fully_masked_row_does_not_produce_nan():
    logits = torch.randn(2, 4)
    mask = torch.zeros(2, 4, dtype=torch.bool)
    out = apply_action_mask(logits, mask)
    probs = torch.softmax(out, dim=-1)
    assert torch.isfinite(probs).all()
    assert probs[:, 0].min() > 0.99


def test_fully_padded_entity_row_does_not_produce_nan(policy):
    batch = _random_batch(b=2, t=2, seed=3)
    batch["entity_pad_mask"] = torch.ones_like(batch["entity_pad_mask"])
    batch["priv_pad_mask"] = torch.ones_like(batch["priv_pad_mask"])
    state = policy.initial_state(2)
    dist, value, _ = policy(state=state, **batch)
    assert torch.isfinite(value).all()
    for v in dist.logits.values():
        assert torch.isfinite(v).all()


def test_masked_pool_ignores_padded_slots():
    x = torch.randn(2, 5, 4)
    valid = torch.tensor([[True, True, False, False, False], [True, False, False, False, False]])
    pooled = masked_pool(x, valid)
    assert pooled.shape == (2, 8)
    expected_mean_0 = x[0, :2].mean(dim=0)
    assert torch.allclose(pooled[0, 4:], expected_mean_0, atol=1e-6)
    expected_max_1 = x[1, 0]
    assert torch.allclose(pooled[1, :4], expected_max_1, atol=1e-6)


def test_padded_entity_values_do_not_affect_output(policy):
    """Garbage in a masked slot must be inert."""
    batch = _random_batch(b=2, t=2, seed=11)
    state = policy.initial_state(2)
    with torch.no_grad():
        d0, v0, _ = policy(state=state, **batch)
        poisoned = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
        pad = batch["entity_pad_mask"]
        noise = torch.randn_like(poisoned["entities"]) * 50.0
        noise[..., C.E_TYPE_ONEHOT] = 0.0
        poisoned["entities"] = torch.where(pad.unsqueeze(-1), noise, poisoned["entities"])
        # Keep the one-hot legal in the poisoned slots.
        poisoned["entities"][..., C.E_TYPE_ONEHOT] = batch["entities"][..., C.E_TYPE_ONEHOT]
        d1, v1, _ = policy(state=state, **poisoned)
    assert torch.allclose(d0.logits["button"], d1.logits["button"], atol=1e-4)
    assert torch.allclose(v0, v1, atol=1e-4)


def test_actor_and_critic_do_not_share_parameters(policy):
    assert_actor_critic_disjoint(policy)
    actor = policy.actor_parameter_names()
    critic = policy.critic_parameter_names()
    assert actor and critic
    assert not (actor & critic)


def test_actor_output_is_independent_of_privileged_input(policy):
    """Structural proof of asymmetry: change priv_*, the actor must not move."""
    batch = _random_batch(seed=5)
    state = policy.initial_state(B)
    with torch.no_grad():
        d0, v0, _ = policy(state=state, **batch)
        b2 = dict(batch)
        b2["priv_vec"] = torch.randn_like(batch["priv_vec"]) * 10
        b2["priv_entities"] = torch.randn_like(batch["priv_entities"]) * 10
        d1, v1, _ = policy(state=state, **b2)
    for head in d0.logits:
        assert torch.equal(d0.logits[head], d1.logits[head]), f"{head} moved with privileged input"
    assert not torch.equal(v0, v1), "critic ignored its privileged input"


def test_gru_resets_zero_the_hidden_state(policy):
    batch = _random_batch(b=1, t=4, seed=13)
    state = policy.initial_state(1)
    resets = torch.zeros(1, 4)
    with torch.no_grad():
        _, v_no_reset, _ = policy(state=state, resets=resets, **batch)
        resets2 = resets.clone()
        resets2[0, 2] = 1.0
        _, v_reset, _ = policy(state=state, resets=resets2, **batch)
    assert torch.allclose(v_no_reset[:, :2], v_reset[:, :2], atol=1e-6)
    assert not torch.allclose(v_no_reset[:, 2:], v_reset[:, 2:], atol=1e-6)


# --------------------------------------------------------------------------
# The asymmetric critic must be V(h, s), not V(s)   (correction 5)
# --------------------------------------------------------------------------


def test_critic_value_depends_on_the_actor_history():
    """Baisero & Amato (arXiv:2105.11674): a state-only critic biases the PG.

    Feed the critic two different actor histories with identical privileged
    state and identical critic state.  If the value does not move, the critic
    is V(s) and the advantage it produces is not the actor's.
    """
    torch.manual_seed(21)
    p = LanePolicy(ModelConfig())
    batch = _random_batch(b=2, t=3, seed=21)
    state = p.initial_state(2)
    with torch.no_grad():
        _d0, v0, _ = p(state=state, **batch)
        # Change ONLY the actor's inputs; every critic input is untouched.
        b2 = dict(batch)
        b2["entities"] = batch["entities"].clone()
        b2["entities"][..., C.E_DS] += 3.0
        b2["entities"][..., C.E_DN] -= 2.0
        _d1, v1, _ = p(state=state, **b2)
    assert not torch.allclose(v0, v1, atol=1e-6), "the critic ignored the actor's history"


def test_the_critic_can_be_configured_back_to_state_only():
    """The flag exists so the bias can be ablated, not so it can be forgotten."""
    torch.manual_seed(22)
    p = LanePolicy(ModelConfig(critic_sees_actor_core=False))
    batch = _random_batch(b=2, t=3, seed=22)
    state = p.initial_state(2)
    with torch.no_grad():
        _d0, v0, _ = p(state=state, **batch)
        b2 = dict(batch)
        b2["entities"] = batch["entities"].clone()
        b2["entities"][..., C.E_DS] += 3.0
        _d1, v1, _ = p(state=state, **b2)
    assert torch.allclose(v0, v1, atol=1e-6)


def test_value_loss_does_not_reach_the_actor_trunk_by_default():
    """The actor's core is fed to the critic DETACHED."""
    torch.manual_seed(23)
    p = LanePolicy(ModelConfig())
    batch = _random_batch(b=2, t=2, seed=23)
    state = p.initial_state(2)
    _dist, value, _ = p(state=state, **batch)
    value.sum().backward()
    actor_grads = [
        n
        for n, q in p.named_parameters()
        if not n.startswith("critic.") and q.grad is not None and q.grad.abs().sum() > 0
    ]
    assert not actor_grads, f"value loss leaked into the actor trunk: {actor_grads[:5]}"


def test_undetached_mode_does_reach_the_actor_trunk():
    """Negative control for the test above."""
    torch.manual_seed(24)
    p = LanePolicy(ModelConfig(detach_actor_core_for_critic=False))
    batch = _random_batch(b=2, t=2, seed=24)
    state = p.initial_state(2)
    _dist, value, _ = p(state=state, **batch)
    value.sum().backward()
    assert any(
        not n.startswith("critic.") and q.grad is not None and q.grad.abs().sum() > 0
        for n, q in p.named_parameters()
    )


def test_actor_and_critic_are_still_parameter_disjoint_with_the_shared_history():
    for cfg in (ModelConfig(), ModelConfig(core="mlp", mlp_hidden=64, mlp_layers=2)):
        assert_actor_critic_disjoint(LanePolicy(cfg))


# --------------------------------------------------------------------------
# The swappable core   (correction 4)
# --------------------------------------------------------------------------


def test_frame_stack_slides_a_window():
    x = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    h = torch.zeros(1, 2, 3 * 3)
    out, hn = frame_stack_with_resets(x, h, None, 4)
    assert out.shape == (2, 5, 4 * 3)
    # The last frame of every window is the current step.
    assert torch.equal(out[:, :, -3:], x)
    # The first three windows are zero-padded from the incoming (zero) state.
    assert torch.equal(out[0, 0, :9], torch.zeros(9))
    assert torch.equal(out[0, 3, :3], x[0, 0])
    assert hn.shape == (1, 2, 9)
    assert torch.equal(hn.reshape(2, 3, 3), x[:, -3:])


def test_frame_stack_honours_resets():
    """Nothing from before an episode boundary may be in the stack after it."""
    x = torch.ones(1, 6, 2)
    for t in range(6):
        x[0, t] = t + 1
    h = torch.zeros(1, 1, 3 * 2)
    resets = torch.zeros(1, 6)
    resets[0, 3] = 1.0
    out, _hn = frame_stack_with_resets(x, h, resets, 4)
    win3 = out[0, 3].reshape(4, 2)
    assert torch.equal(win3[3], x[0, 3]), "the current frame must survive its own reset"
    assert torch.equal(win3[:3], torch.zeros(3, 2)), "pre-reset frames leaked across the boundary"
    win4 = out[0, 4].reshape(4, 2)
    assert torch.equal(win4[2:], x[0, 3:5])
    assert torch.equal(win4[:2], torch.zeros(2, 2))
    win6 = out[0, 5].reshape(4, 2)
    assert torch.equal(win6[1:], x[0, 3:6])


def test_frame_stack_reset_state_does_not_leak_into_the_next_chunk():
    """A reset near the end of a chunk must be reflected in the carried state.

    With a reset at the last step, the state handed to the next chunk keeps the
    post-reset frame and nothing before it -- otherwise the first steps of the
    next chunk would silently see the previous episode.
    """
    x = torch.arange(1 * 4 * 2, dtype=torch.float32).reshape(1, 4, 2) + 1.0
    h = torch.zeros(1, 1, 3 * 2)
    resets = torch.zeros(1, 4)
    resets[0, 3] = 1.0
    _out, hn = frame_stack_with_resets(x, h, resets, 4)
    carried = hn.reshape(3, 2)
    assert torch.equal(carried[:2], torch.zeros(2, 2)), "pre-reset frames survived"
    assert torch.equal(carried[2], x[0, 3]), "the post-reset frame was dropped"

    # Without the reset, all three previous frames carry over.
    _out2, hn2 = frame_stack_with_resets(x, h, torch.zeros(1, 4), 4)
    assert torch.equal(hn2.reshape(3, 2), x[0, 1:])


@pytest.mark.parametrize("core", ["gru", "mlp"])
def test_both_cores_produce_the_right_shapes(core):
    torch.manual_seed(31)
    cfg = ModelConfig(core=core, mlp_hidden=64, mlp_layers=2)
    p = LanePolicy(cfg)
    batch = _random_batch(b=2, t=4, seed=31)
    state = p.initial_state(2)
    dist, value, new_state = p(state=state, **batch)
    assert dist.logits["button"].shape == (2, 4, C.N_BUTTONS)
    assert dist.logits["target"].shape == (2, 4, C.N_SLOTS)
    assert value.shape == (2, 4)
    assert new_state.actor.shape == (1, 2, cfg.actor_state_dim)
    assert new_state.critic.shape == (1, 2, cfg.critic_state_dim)
    assert torch.isfinite(value).all()


@pytest.mark.parametrize("core", ["gru", "mlp"])
def test_both_cores_train(core):
    torch.manual_seed(32)
    cfg = ModelConfig(core=core, mlp_hidden=64, mlp_layers=2)
    p = LanePolicy(cfg)
    before = {k: v.detach().clone() for k, v in p.state_dict().items()}
    batch = _random_batch(b=2, t=4, seed=32)
    state = p.initial_state(2)
    dist, value, _ = p(state=state, **batch)
    action = dist.sample()
    loss = -(dist.log_prob(action).mean()) + value.pow(2).mean()
    loss.backward()
    opt = torch.optim.Adam(p.parameters(), lr=1e-2)
    opt.step()
    changed = [
        k for k, v in p.state_dict().items()
        if v.dtype.is_floating_point and not torch.equal(v, before[k])
    ]
    assert any(k.startswith("critic.") for k in changed), "critic did not learn"
    assert any(not k.startswith("critic.") for k in changed), "actor did not learn"


def test_mlp_state_is_a_frame_stack_not_a_hidden_vector():
    cfg = ModelConfig(core="mlp", frame_stack=4)
    assert cfg.actor_state_dim == 3 * cfg.actor_core_input_dim
    assert cfg.critic_state_dim == 3 * cfg.critic_core_input_dim
    gru = ModelConfig(core="gru")
    assert gru.actor_state_dim == gru.core_dim == gru.gru_dim


def test_mlp_core_has_no_recurrence_so_a_single_step_is_memoryless():
    """A frame stack of 1 is a pure feed-forward policy; useful as a control."""
    torch.manual_seed(33)
    cfg = ModelConfig(core="mlp", frame_stack=1, mlp_hidden=64, mlp_layers=2)
    p = LanePolicy(cfg)
    assert cfg.actor_state_dim == 0
    batch = _random_batch(b=2, t=3, seed=33)
    state = p.initial_state(2)
    dist, value, new_state = p(state=state, **batch)
    assert new_state.actor.numel() == 0
    assert torch.isfinite(value).all()


def test_mlp_core_reset_changes_the_output():
    torch.manual_seed(34)
    p = LanePolicy(ModelConfig(core="mlp", mlp_hidden=64, mlp_layers=2))
    batch = _random_batch(b=1, t=4, seed=34)
    # Warm the state so there is real history to lose.
    state = p.initial_state(1)
    with torch.no_grad():
        _d, _v, state = p(state=state, **batch)
        _d, v_no_reset, _ = p(state=state, resets=torch.zeros(1, 4), **batch)
        r = torch.zeros(1, 4)
        r[0, 1] = 1.0
        _d, v_reset, _ = p(state=state, resets=r, **batch)
    assert torch.allclose(v_no_reset[:, :1], v_reset[:, :1], atol=1e-6)
    assert not torch.allclose(v_no_reset[:, 1:], v_reset[:, 1:], atol=1e-6)


def test_bad_core_name_is_rejected():
    with pytest.raises(ValueError, match="core must be"):
        ModelConfig(core="lstm")


# --------------------------------------------------------------------------
# Head masking: the action the ENVIRONMENT sees is not all four heads
# --------------------------------------------------------------------------
#
# The old LaneActionDist scored button + screen_x + screen_y + target on every
# step, with a docstring claiming that conditioning on the button "makes the
# PPO ratio depend on the sampled button, which is a well known source of
# silent bias".  That is backwards: the mask is a deterministic function of the
# sampled button, the button is part of the action, and the masked product is a
# normalised distribution over the actions the environment can tell apart.
# These tests enumerate that space and prove it.

N_M, N_T = 3, 4  # small enough to enumerate exhaustively


def _effective_actions(n_x=N_M, n_target=N_T, n_y=None):
    """Every action the environment can distinguish, as (b, mx, mz, t)."""
    out = []
    for b in range(C.N_BUTTONS):
        moves = (
            [(x, z) for x in range(n_x) for z in range(n_y if n_y is not None else n_x)]
            if USES_MOVE_HEAD[b]
            else [(0, 0)]
        )
        targets = list(range(n_target)) if USES_TARGET_HEAD[b] else [0]
        out += [(b, x, z, t) for (x, z) in moves for t in targets]
    return out


def _logits(seed, n_x=N_M, n_target=N_T, n_y=None):
    g = torch.Generator().manual_seed(seed)
    return {
        "button": torch.randn(C.N_BUTTONS, generator=g),
        "screen_x": torch.randn(n_x, generator=g),
        "screen_y": torch.randn(n_y if n_y is not None else n_x, generator=g),
        "target": torch.randn(n_target, generator=g),
    }


def _batched(logits, n):
    return LaneActionDist({k: v.unsqueeze(0).expand(n, -1) for k, v in logits.items()})


def _action_tensors(actions):
    cols = list(zip(*actions))
    return {
        k: torch.tensor(c, dtype=torch.long)
        for k, c in zip(("button", "screen_x", "screen_y", "target"), cols)
    }


def test_masked_log_prob_is_normalised_over_the_effective_action_space():
    """sum_a pi(a|s) == 1 -- the masked product really is a distribution."""
    actions = _effective_actions()
    dist = _batched(_logits(0), len(actions))
    total = dist.log_prob(_action_tensors(actions)).exp().sum()
    assert total.item() == pytest.approx(1.0, abs=1e-5), total.item()


def test_the_importance_ratio_is_unbiased_under_head_masking():
    """``E_{a~pi_old}[pi_new(a)/pi_old(a)] == 1``, exactly.

    This is the property the old docstring said masking would break.  It is
    the whole content of "the PPO ratio is unbiased": the estimator
    ``ratio * f(a)`` has expectation ``E_{pi_new}[f]`` for every ``f`` iff the
    ratio integrates to 1 under the behaviour policy.
    """
    actions = _effective_actions()
    a = _action_tensors(actions)
    old = _batched(_logits(1), len(actions)).log_prob(a)
    new = _batched(_logits(2), len(actions)).log_prob(a)
    expected_ratio = (old.exp() * (new - old).exp()).sum()
    assert expected_ratio.item() == pytest.approx(1.0, abs=1e-5)


def test_two_draws_differing_only_in_an_unused_head_are_ONE_action():
    """They must score identically -- that is why integrating them out is exact."""
    dist = _batched(_logits(3), 2)
    noop, move = C.BUTTONS.index("noop"), C.BUTTONS.index("move")
    a = {
        "button": torch.tensor([noop, noop]),
        "screen_x": torch.tensor([0, 2]),
        "screen_y": torch.tensor([1, 0]),
        "target": torch.tensor([0, 3]),
    }
    lp = dist.log_prob(a)
    assert lp[0].item() == pytest.approx(lp[1].item(), abs=1e-6)
    # move DOES use the move heads, so those two must NOT collapse
    a2 = dict(a, button=torch.tensor([move, move]))
    lp2 = dist.log_prob(a2)
    assert lp2[0].item() != pytest.approx(lp2[1].item(), abs=1e-6)
    # ...but its target head is still ignored
    a3 = dict(a2, screen_x=torch.tensor([0, 0]), screen_y=torch.tensor([0, 0]))
    lp3 = dist.log_prob(a3)
    assert lp3[0].item() == pytest.approx(lp3[1].item(), abs=1e-6)


def test_masked_entropy_equals_the_entropy_of_the_enumerated_joint():
    actions = _effective_actions()
    dist = _batched(_logits(4), len(actions))
    lp = dist.log_prob(_action_tensors(actions))
    brute = -(lp.exp() * lp).sum()
    assert dist.entropy()[0].item() == pytest.approx(brute.item(), rel=1e-5)


def test_masked_kl_equals_the_kl_of_the_enumerated_joint():
    actions = _effective_actions()
    a = _action_tensors(actions)
    q_logits, p_logits = _logits(5), _logits(6)
    q = _batched(q_logits, len(actions))
    p_lp = _batched(p_logits, len(actions)).log_prob(a)
    q_lp = q.log_prob(a)
    brute = (p_lp.exp() * (p_lp - q_lp)).sum()
    ours = q.kl_to({k: v.unsqueeze(0).expand(len(actions), -1) for k, v in p_logits.items()})
    assert ours[0].item() == pytest.approx(brute.item(), rel=1e-5)


def test_the_entropy_numbers_in_the_docstring_are_the_real_ones():
    """``loss/entropy`` is not comparable across this change, so the two
    reference points a reader needs are pinned here rather than only asserted
    in prose."""
    import math

    zeros = {
        "button": torch.zeros(1, C.N_BUTTONS),
        "screen_x": torch.zeros(1, C.N_SCREEN_X),
        "screen_y": torch.zeros(1, C.N_SCREEN_Y),
        "target": torch.zeros(1, C.N_SLOTS),
    }
    uniform = LaneActionDist(zeros).entropy().item()
    assert uniform == pytest.approx(10.661, abs=1e-3)

    n_eff = len(_effective_actions(C.N_SCREEN_X, C.N_SLOTS, C.N_SCREEN_Y))
    assert n_eff == 834_626
    assert math.log(n_eff) == pytest.approx(13.635, abs=1e-3)
    assert uniform < math.log(n_eff), "uniform is no longer the maximum-entropy policy"
    # what the unconditional sum over four heads had instead
    assert math.log(C.N_BUTTONS * C.N_SCREEN_X * C.N_SCREEN_Y * C.N_SLOTS) == pytest.approx(14.099, abs=1e-3)


def test_head_usage_matches_what_decode_action_puts_on_the_wire(quiet_fog):
    """The mask is only unbiased if it agrees with the ENVIRONMENT.

    Masking a head the server does read would make two genuinely different
    actions share a log-probability, and then the ratio really would be wrong.
    So the table is checked against the only thing that decides it: whether
    changing that head changes the order on the wire.  Note that
    ``constants.TARGETED_BUTTONS`` / ``MOVE_BUTTONS`` do NOT agree with this
    (they are referenced by nothing) -- a cast sends both ``id`` and ``x``/``y``.
    """
    from lanerl_rl.env import decode_action, order_for_command

    frame = top_lane_sequence(n=1)[0]
    builder = ObservationBuilder(C.TEAM_BLUE, fog_model=quiet_fog)
    ob = builder.build(frame)
    me = frame.champion_of_team(C.TEAM_BLUE)
    netids = [1000 + i for i in range(C.N_SLOTS)]
    valid = [i for i in range(C.N_SLOTS) if ob.entities[i, C.E_VALID] > 0.5]
    empty = [i for i in range(C.N_SLOTS) if ob.entities[i, C.E_VALID] < 0.5]
    assert len(valid) >= 2 and empty, "scenario cannot exercise the target head"

    def order(b, mx, mz, t):
        return order_for_command(
            decode_action(
                {"button": b, "screen_x": mx, "screen_y": mz, "target": t},
                builder, ob, me, netids,
            )
        )

    mid, hi = C.N_SCREEN_Y // 2, C.N_SCREEN_Y - 1
    for b, name in enumerate(C.BUTTONS):
        # attack_move only falls back to a positional move when the chosen
        # slot is empty, so "does the move head matter" is asked of both.
        move_matters = any(
            order(b, 0, 0, t) != order(b, hi, hi, t) for t in (valid[0], empty[0])
        )
        target_matters = order(b, mid, mid, valid[0]) != order(b, mid, mid, valid[1])
        assert move_matters == bool(USES_MOVE_HEAD[b]), f"{name}: move head"
        assert target_matters == bool(USES_TARGET_HEAD[b]), f"{name}: target head"


def test_masked_logits_do_not_poison_an_unused_head():
    """An illegal slot is -1e9; on a button that ignores the target head that
    must contribute exactly 0, not -1e9 and not NaN."""
    logits = _logits(7, n_target=4)
    logits["target"] = torch.tensor([0.0, -1e9, -1e9, -1e9])
    dist = _batched(logits, 1)
    a = {k: torch.tensor([v]) for k, v in
         zip(("button", "screen_x", "screen_y", "target"), (C.BUTTONS.index("move"), 1, 1, 2))}
    lp = dist.log_prob(a)
    assert torch.isfinite(lp).all() and lp.item() > -20.0


def test_categorical_kl_is_finite_when_the_policy_underflows_a_masked_slot():
    """KL to the frozen prior must never be inf.

    ``torch.distributions.kl_divergence`` for Categorical writes inf wherever
    the SECOND argument's probability is exactly 0.  As the policy sharpens, a
    softmax entry underflows to 0 while the reference still has mass there,
    and the whole KL term -- the only thing anchoring the run to its BC prior
    -- becomes inf.  On run rl-bc2-0912 that happened on 7 of 56 updates, and
    the updates where the policy has run furthest from the prior are exactly
    the ones where the anchor was silently dropped.
    """
    policy = torch.tensor([[80.0, -80.0, -1e9]])   # entry 1 underflows to 0.0
    reference = torch.tensor([[1.0, 0.5, -1e9]])   # reference still has mass there
    assert torch.softmax(policy, -1)[0, 1].item() == 0.0

    torch_kl = torch.distributions.kl_divergence(
        torch.distributions.Categorical(logits=reference),
        torch.distributions.Categorical(logits=policy),
    )
    assert torch.isinf(torch_kl).any(), "the failure this guards is gone; revisit"

    ours = categorical_kl(reference, policy)
    assert torch.isfinite(ours).all()
    assert ours.item() > 1.0, "a policy this far from the prior must be penalised"

    grad_src = policy.clone().requires_grad_(True)
    categorical_kl(reference, grad_src).sum().backward()
    assert torch.isfinite(grad_src.grad).all()


def test_categorical_kl_agrees_with_torch_on_well_conditioned_logits():
    """The log-space rewrite must not change the ordinary case."""
    torch.manual_seed(0)
    a, b = torch.randn(8, 11), torch.randn(8, 11)
    expected = torch.distributions.kl_divergence(
        torch.distributions.Categorical(logits=b),
        torch.distributions.Categorical(logits=a),
    )
    assert torch.allclose(expected, categorical_kl(b, a), atol=1e-5)


def test_the_whole_policy_still_produces_a_normalised_masked_log_prob(policy):
    """The enumeration tests above build logits by hand; this one runs the
    real network at the real head sizes and integrates over the 834,626
    effective actions it can emit."""
    batch = _random_batch(b=1, t=1, seed=77, all_valid=True)
    state = policy.initial_state(1)
    with torch.no_grad():
        dist, _tokens, _h = policy.actor_forward(
            batch["entities"], batch["entity_pad_mask"], batch["self_vec"],
            batch["global_vec"], state.actor,
        )
    flat = {k: v.reshape(1, -1) for k, v in dist.logits.items()}
    actions = _effective_actions(C.N_SCREEN_X, C.N_SLOTS, C.N_SCREEN_Y)
    n = len(actions)
    d = LaneActionDist({k: v.expand(n, -1) for k, v in flat.items()})
    total = d.log_prob(_action_tensors(actions)).exp().sum()
    assert total.item() == pytest.approx(1.0, abs=1e-4), (n, total.item())


def test_end_to_end_from_real_observations(policy, quiet_fog):
    """Feed the builder's actual output through the network."""
    frames = top_lane_sequence(n=6)
    builder = ObservationBuilder(C.TEAM_BLUE, fog_model=quiet_fog)
    obs = [builder.build(f) for f in frames]

    def stack(attr):
        return torch.from_numpy(np.stack([getattr(o, attr) for o in obs])).unsqueeze(0)

    masks = {
        "button": torch.from_numpy(np.stack([o.action_mask.button for o in obs])).unsqueeze(0),
        "screen_x": torch.from_numpy(np.stack([o.action_mask.screen_x for o in obs])).unsqueeze(0),
        "screen_y": torch.from_numpy(np.stack([o.action_mask.screen_y for o in obs])).unsqueeze(0),
        "target": torch.from_numpy(np.stack([o.action_mask.target for o in obs])).unsqueeze(0),
    }
    state = policy.initial_state(1)
    dist, value, _ = policy(
        entities=stack("entities"),
        entity_pad_mask=stack("entity_pad_mask"),
        self_vec=stack("self_vec"),
        global_vec=stack("global_vec"),
        priv_entities=stack("priv_entities"),
        priv_pad_mask=stack("priv_pad_mask"),
        priv_vec=stack("priv_vec"),
        state=state,
        action_masks=masks,
    )
    assert value.shape == (1, len(obs))
    assert torch.isfinite(value).all()
    action = dist.sample()
    lp = dist.log_prob(action)
    assert torch.isfinite(lp).all()
