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
from lanerl_jax.sim.init import TOP_OUTER_TURRET, init_lane, lane_params
from lanerl_jax.sim.orders import OrderKind
from lanerl_jax.sim.state import Team
from lanerl_jax.train.policy import LanePolicy, PolicyConfig
from lanerl_jax.train.ppo import factored_log_prob
from lanerl_jax.train.trainer import TrainConfig, _orders_from, make_train

SMALL = TrainConfig(n_envs=4, rollout_steps=8, n_updates=2, n_minibatches=2)


def test_policy_targets_are_observation_slots_not_simulator_unit_indices():
    """Slot 13 is an enemy-minion slot; it need not be unit 13."""
    state = init_lane()
    slots = jnp.full((2, 32), -1, jnp.int32)
    slots = slots.at[0, 13].set(41).at[1, 19].set(57)

    attack_r = _orders_from(
        (jnp.asarray([2, 6]), jnp.asarray([0, 0]), jnp.asarray([0, 0]),
         jnp.asarray([13, 19])), state, slots)
    np.testing.assert_array_equal(np.asarray(attack_r.target), [41, 57])
    np.testing.assert_array_equal(np.asarray(attack_r.kind),
                                  [OrderKind.ATTACK, OrderKind.CAST_R])

    q_w = _orders_from(
        (jnp.asarray([3, 4]), jnp.asarray([0, 0]), jnp.asarray([0, 0]),
         jnp.asarray([13, 19])), state, slots)
    np.testing.assert_array_equal(np.asarray(q_w.kind),
                                  [OrderKind.CAST_Q, OrderKind.CAST_W])
    e_recall = _orders_from(
        (jnp.asarray([5, 7]), jnp.asarray([0, 0]), jnp.asarray([0, 0]),
         jnp.asarray([13, 19])), state, slots)
    np.testing.assert_array_equal(np.asarray(e_recall.kind),
                                  [OrderKind.CAST_E, OrderKind.RECALL])

    no_target = _orders_from(
        (jnp.asarray([2, 1]), jnp.asarray([0, 0]), jnp.asarray([0, 0]),
         jnp.asarray([0, 0])), state, slots)
    np.testing.assert_array_equal(np.asarray(no_target.kind),
                                  [OrderKind.MOVE, OrderKind.MOVE])
    np.testing.assert_array_equal(np.asarray(no_target.target), [-1, -1])


@pytest.fixture(scope="module")
def trained():
    train = jax.jit(make_train(SMALL))
    return train(jax.random.key(0))


def test_the_loop_runs_and_produces_the_expected_metrics(trained):
    _, m = trained
    for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac",
              "dual_clip_frac", "reward", "lane_dist", "route_nonready"):
        assert k in m, k
        assert np.isfinite(np.asarray(m[k])).all(), f"{k} went non-finite"
    assert np.asarray(m["policy_loss"]).shape[0] == SMALL.n_updates


def test_cs_at_10min_is_nan_until_an_episode_actually_ends(trained):
    """An unfinished episode must report NOTHING, not zero.

    This is the metric that cost a 400-update run. The old `cs` metric sampled
    `env_state.cs` at the rollout boundary, which reports 0.0 both when the
    agent has farmed nothing and when no episode has finished yet -- and those
    demand opposite responses. NaN is unmistakable; 0.0 is a lie that looks
    like data.

    Every env resets on the same step (`done` is a pure function of `t_ms`),
    so the old metric was also a sawtooth whose value depended on where the
    rollout boundary fell rather than on how well the agent played.
    """
    _, m = trained
    assert "cs_at_10min" in m
    # 4 envs x 8 steps is 32 decisions against an 18,000-decision episode
    assert np.isnan(np.asarray(m["cs_at_10min"])).all(), (
        "cs_at_10min reported a number when no episode had ended")


def test_lane_distance_starts_at_the_fountain(trained):
    """The leading indicator, and a check that the potential is wired live.

    ~8,000 units is where both champions spawn relative to the lane corridor.
    If this read 0 the potential would be silently inactive and the walk would
    again pay nothing.
    """
    _, m = trained
    d = float(np.asarray(m["lane_dist"])[0])
    assert 7_000 < d < 9_000, f"lane_dist started at {d:.0f}, expected ~8,000"


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
    params_tbl = lane_params()
    obs = jax.vmap(
        lambda i: build_observation(state, i, frame, params=params_tbl)
    )(jnp.arange(2))
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
    params_tbl = lane_params()
    obs = jax.vmap(
        lambda i: build_observation(state, i, frame, params=params_tbl)
    )(jnp.arange(2))
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


def test_episode_phases_are_staggered_not_lockstep():
    """Envs must not all reset on the same update.

    With `done = t_ms >= episode_s * 1000` and every env broadcast from the
    same constant pytree, all `n_envs` episodes ran in lockstep: a rollout was
    `rollout_steps / decision_hz` seconds of ONE moment of the game repeated
    `n_envs` times, and cs@10min arrived in a single burst every
    `episode_steps / rollout_steps` updates (141 at the production config).
    Both failures are invisible in the metrics, so they are asserted here.
    """
    cfg = TrainConfig(n_envs=16, rollout_steps=4, n_updates=1, n_minibatches=2,
                      episode_s=4.0)
    built = make_train(cfg)
    runner = built.initial_runner(jax.random.PRNGKey(0))

    d = np.asarray(runner.deadline_ms)
    assert d.shape == (16,)
    assert len(np.unique(d)) > 1, "deadlines identical -- envs are in lockstep"
    full = cfg.episode_s * 1000.0
    assert d.max() <= full and d.min() > 0
    # Floored at two rollouts so nothing resets inside its own first rollout.
    assert d.min() >= min(2.0 * cfg.rollout_steps / cfg.decision_hz * 1000.0,
                          full) - 1e-3

    # Run past the longest first episode; the clocks must then be spread out.
    runner, _ = jax.jit(built.run_chunk, static_argnums=1)(
        runner, int(cfg.episode_steps / cfg.rollout_steps) + 4)
    t = np.asarray(runner.env_state.t_ms)
    assert len(np.unique(np.round(t, 3))) > 1, (
        "game clocks converged -- the phase offset did not survive reset")
    # Every deadline is back to the full length: only the FIRST episode is cut.
    np.testing.assert_allclose(np.asarray(runner.deadline_ms), full, rtol=0,
                               atol=1e-3)


def test_partial_first_episode_is_excluded_from_cs():
    """`cs_at_10min` must average full games only.

    The stagger makes each env's first episode short by construction. Counting
    those would read as a CS collapse over the first ~141 updates -- a metric
    artefact indistinguishable from the agent getting worse.
    """
    cfg = TrainConfig(n_envs=8, rollout_steps=4, n_updates=1, n_minibatches=2,
                      episode_s=4.0)
    built = make_train(cfg)
    runner = built.initial_runner(jax.random.PRNGKey(1))
    n = int(cfg.episode_steps / cfg.rollout_steps)
    # First pass: partial episodes end here, so `done` fires and `done_full`
    # must not.
    _, metrics = jax.jit(built.run_chunk, static_argnums=1)(runner, n)
    eps = np.asarray(metrics["cs_episodes"])
    assert eps.sum() == 0.0, (
        "a partial first episode was counted as a cs@10min sample")


def test_target_kl_is_enforced_rather_than_merely_configured():
    """A `target_kl` of 0 must stop every minibatch after the first.

    `critic_lr` and `target_kl` were both declared in `PPOConfig`, recorded in
    run manifests, and read by nothing. This is the regression test for the
    half of that which is behavioural.
    """
    import dataclasses  # noqa: F401  (NamedTuple._replace is the real tool)

    cfg = SMALL._replace(ppo=SMALL.ppo._replace(target_kl=0.0))
    built = make_train(cfg)
    _, metrics = jax.jit(built.run_chunk, static_argnums=1)(
        built.initial_runner(jax.random.PRNGKey(2)), 1)
    # approx_kl is >= 0 and is > 0 for any nonzero step, so the stop latches on
    # the first minibatch and every later one is skipped.
    assert float(np.asarray(metrics["kl_stopped"])[0]) > 0.0

    loose = SMALL._replace(ppo=SMALL.ppo._replace(target_kl=1e9))
    _, m2 = jax.jit(make_train(loose).run_chunk, static_argnums=1)(
        make_train(loose).initial_runner(jax.random.PRNGKey(2)), 1)
    assert float(np.asarray(m2["kl_stopped"])[0]) == 0.0


def test_critic_head_runs_at_critic_lr():
    """The value readout must be optimised at `critic_lr`, not `lr`.

    A single `adam(lr)` trained it thirty times slower than the rate the config
    advertised, which is the leading candidate for `value_loss` reaching 542.7
    in the RL-002 run. Asserted through the optimiser's own hyperparameters so
    the test fails if the label tree stops matching the module name.
    """
    from lanerl_jax.train.policy import VALUE_HEAD_NAME

    cfg = SMALL._replace(ppo=SMALL.ppo._replace(lr=1e-5, critic_lr=3e-4))
    built = make_train(cfg)
    r0 = built.initial_runner(jax.random.PRNGKey(3))
    names = jax.tree_util.tree_flatten_with_path(r0.params)[0]
    assert any(VALUE_HEAD_NAME in jax.tree_util.keystr(p) for p, _ in names), (
        f"no {VALUE_HEAD_NAME!r} subtree -- the label tree cannot be matching")

    r1, _ = jax.jit(built.run_chunk, static_argnums=1)(r0, 1)
    moved = jax.tree_util.tree_map(
        lambda a, b: float(jnp.abs(a - b).max()), r0.params, r1.params)
    flat = {jax.tree_util.keystr(p): v
            for p, v in jax.tree_util.tree_flatten_with_path(moved)[0]}
    critic = [v for k, v in flat.items() if VALUE_HEAD_NAME in k]
    actor = [v for k, v in flat.items() if VALUE_HEAD_NAME not in k]
    assert critic and actor
    # Adam's first step is ~lr in magnitude regardless of gradient scale, so a
    # 30x lr ratio shows up directly as a step-size ratio.
    assert max(critic) > 10.0 * max(actor), (
        f"critic step {max(critic):.2e} vs actor {max(actor):.2e} -- the "
        "critic is not on its own learning rate")
