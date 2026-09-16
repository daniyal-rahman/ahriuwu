"""The training loop's invariants.

The headline one is **actor/learner agreement**: the log-probs the rollout
recorded must equal the ones the update recomputes from the same parameters.
`lanerl_train` keeps its own version of this check (`test_actor_learner_agree`)
because the class of bug it catches is silent -- the importance ratio is
`exp(new - old)`, so if the two paths disagree about what the policy did, every
ratio is wrong by a constant the optimiser then chases. Curves stay smooth.

Under Anakin the two paths are the *same code* on the same device, which removes
most of the ways they can drift (there is no serialisation, no parameter
version, no separate process). It does not remove all of them: a different mask,
a different observation for the same state, or a head read in a different order
would still do it.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.obs.builder import build_observation
from lanerl_jax.obs.frame import make_lane_frame
from lanerl_jax.sim.init import TOP_OUTER_TURRET, init_lane
from lanerl_jax.sim.state import Team
from lanerl_jax.train.policy import LanePolicy, PolicyConfig
from lanerl_jax.train.ppo import factored_log_prob
from lanerl_jax.train.trainer import TrainConfig, make_train

SMALL = TrainConfig(n_envs=4, rollout_steps=8, n_updates=2, n_minibatches=2)


@pytest.fixture(scope="module")
def trained():
    train = jax.jit(make_train(SMALL))
    return train(jax.random.key(0))


def test_the_loop_runs_and_produces_the_expected_metrics(trained):
    _, m = trained
    for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac",
              "dual_clip_frac", "reward", "cs"):
        assert k in m, k
        assert np.isfinite(np.asarray(m[k])).all(), f"{k} went non-finite"
    assert np.asarray(m["policy_loss"]).shape[0] == SMALL.n_updates


def test_the_actor_and_the_learner_agree_on_log_probs():
    """Recomputing a stored action's log-prob under the SAME parameters must
    reproduce the stored value.

    Built by hand rather than reaching into the loop, because the thing under
    test is that the two code paths -- sample-time and update-time -- compute
    the same quantity.
    """
    frame = make_lane_frame(TOP_OUTER_TURRET[Team.BLUE],
                            TOP_OUTER_TURRET[Team.RED], (1131.8, 1426.3))
    policy = LanePolicy(PolicyConfig())
    state = init_lane()
    obs = jax.vmap(lambda i: build_observation(state, i, frame))(jnp.arange(2))
    params = policy.init(jax.random.key(0), obs.entities, obs.entity_pad_mask,
                         obs.self_vec, obs.global_vec)

    logits = policy.apply(params, obs.entities, obs.entity_pad_mask,
                          obs.self_vec, obs.global_vec)
    lg = (logits.button, logits.screen_x, logits.screen_y, logits.target)
    keys = jax.random.split(jax.random.key(7), 4)
    action = tuple(jax.random.categorical(k, l) for k, l in zip(keys, lg))
    at_sample = factored_log_prob(lg, action)

    # the update path: same params, same observation, recomputed
    again = policy.apply(params, obs.entities, obs.entity_pad_mask,
                         obs.self_vec, obs.global_vec)
    at_update = factored_log_prob(
        (again.button, again.screen_x, again.screen_y, again.target), action)

    np.testing.assert_allclose(np.asarray(at_update), np.asarray(at_sample),
                               rtol=0, atol=1e-6)


def test_a_sampled_action_never_lands_on_a_masked_slot():
    """A masked slot must be unreachable by SAMPLING, not merely improbable.

    Targeting an entity that is not there is the action-space version of the
    fog hallucination `obs.py` guards against -- and it would be scored as a
    real action by the update.
    """
    frame = make_lane_frame(TOP_OUTER_TURRET[Team.BLUE],
                            TOP_OUTER_TURRET[Team.RED], (1131.8, 1426.3))
    policy = LanePolicy(PolicyConfig())
    state = init_lane()
    obs = jax.vmap(lambda i: build_observation(state, i, frame))(jnp.arange(2))
    params = policy.init(jax.random.key(0), obs.entities, obs.entity_pad_mask,
                         obs.self_vec, obs.global_vec)
    logits = policy.apply(params, obs.entities, obs.entity_pad_mask,
                          obs.self_vec, obs.global_vec)

    pad = np.asarray(obs.entity_pad_mask)
    assert pad.any(), "fixture has no masked slots, so this proves nothing"
    for i in range(256):
        tgt = np.asarray(jax.random.categorical(
            jax.random.key(i), logits.target))
        for row in range(tgt.shape[0]):
            assert not pad[row, tgt[row]], f"sampled masked slot {tgt[row]}"


def test_reset_is_a_where_against_a_constant(trained):
    """D11: reset must stay pure array initialisation.

    The auto-reset pattern runs the reset path every step and discards >99% of
    it, which is affordable only while that path is a constant write. If a
    procedural reset ever lands, this is the assumption it breaks.
    """
    import inspect

    from lanerl_jax.train import trainer

    src = inspect.getsource(trainer)
    assert "fresh = init_lane()" in src
    assert "jnp.where(done, b, a)" in src, (
        "reset is no longer a where against a constant pytree")


def test_training_changes_the_parameters(trained):
    """A loop that runs and learns nothing looks identical to one that works."""
    runner, _ = trained
    # the pad mask is BOOL -- the policy inverts it, and a float mask fails with
    # "not does not accept dtype float64" rather than anything about masking
    fresh = LanePolicy(PolicyConfig()).init(
        jax.random.key(0), jnp.zeros((2, 32, 16)), jnp.zeros((2, 32), bool),
        jnp.zeros((2, 16)), jnp.zeros((2, 6)))
    moved = [not np.allclose(np.asarray(a), np.asarray(b), atol=1e-9)
             for a, b in zip(jax.tree.leaves(runner.params),
                             jax.tree.leaves(fresh))]
    assert any(moved), "no parameter moved across two updates"


def test_episode_length_is_not_the_discount_horizon():
    """They are different numbers, and conflating them made the task impossible.

    With the episode clamped to PPO's 120 s discount horizon, the arithmetic is
    fatal: the champion spawns 13,532 units from the wave meeting point (39 s of
    walking), the first wave spawns at 90 s and reaches the middle around 120 s,
    and the episode ends at 120 s. Farming is possible for ~0 seconds.

    CS was exactly 0.00000 across 400 updates and 26 million champion-decisions
    and could not have been anything else -- the last-hit term can never fire.
    That looks exactly like a policy that has not learned.
    """
    cfg = TrainConfig()
    assert cfg.episode_s == 600.0
    assert cfg.ppo.horizon_s == 120.0
    assert cfg.episode_s > cfg.ppo.horizon_s
    # long enough for the first wave (90 s) plus the walk to lane (~39 s)
    assert cfg.episode_s > 90.0 + 39.0 + 60.0
    assert cfg.episode_steps == 18_000
