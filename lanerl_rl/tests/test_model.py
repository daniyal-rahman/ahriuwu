"""The policy network: shapes, masking, permutation equivariance, asymmetry."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lanerl_rl import constants as C
from lanerl_rl.model import (
    LanePolicy,
    ModelConfig,
    RecurrentState,
    apply_action_mask,
    assert_actor_critic_disjoint,
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
    assert dist.logits["move_x"].shape == (B, T, C.N_MOVE_BINS)
    assert dist.logits["move_z"].shape == (B, T, C.N_MOVE_BINS)
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
    for head in ("button", "move_x", "move_z"):
        assert torch.allclose(d0.logits[head], d1.logits[head], atol=1e-5), (
            f"{head} logits changed under a slot permutation"
        )


def test_action_masking_zeroes_illegal_choices(policy):
    batch = _random_batch()
    state = policy.initial_state(B)
    masks = {
        "button": torch.zeros(B, T, C.N_BUTTONS, dtype=torch.bool),
        "move_x": torch.ones(B, T, C.N_MOVE_BINS, dtype=torch.bool),
        "move_z": torch.ones(B, T, C.N_MOVE_BINS, dtype=torch.bool),
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


def test_end_to_end_from_real_observations(policy, quiet_fog):
    """Feed the builder's actual output through the network."""
    frames = top_lane_sequence(n=6)
    builder = ObservationBuilder(C.TEAM_BLUE, fog_model=quiet_fog)
    obs = [builder.build(f) for f in frames]

    def stack(attr):
        return torch.from_numpy(np.stack([getattr(o, attr) for o in obs])).unsqueeze(0)

    masks = {
        "button": torch.from_numpy(np.stack([o.action_mask.button for o in obs])).unsqueeze(0),
        "move_x": torch.from_numpy(np.stack([o.action_mask.move_x for o in obs])).unsqueeze(0),
        "move_z": torch.from_numpy(np.stack([o.action_mask.move_z for o in obs])).unsqueeze(0),
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
