"""Opt-in history migration, nonzero-feature likelihood, reset and resume."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from lanerl_jax.obs.visible_history import (
    VISIBLE_HISTORY_INTERFACE, HISTORY_ENTITY_DIM, HISTORY_FEATURES)
from lanerl_jax.train.policy import LanePolicy, PolicyConfig, merge_visible_history_params


def small_config():
    return PolicyConfig(core='gru', core_norm=True, core_residual=True, d_model=16,
        n_heads=4, n_layers=1, ffn_dim=32, ctx_dim=32, core_dim=32, mlp_hidden=32, mlp_layers=1)


def test_visible_history_migration_preserves_outputs_and_has_gradient():
    old = LanePolicy(small_config())
    new = LanePolicy(small_config()._replace(observation_interface=VISIBLE_HISTORY_INTERFACE,
                                            entity_dim=HISTORY_ENTITY_DIM))
    e = jax.random.normal(jax.random.key(1), (2, 32, 16), dtype=jnp.float32)
    history = jax.random.normal(jax.random.key(2), (2, 32, HISTORY_FEATURES), dtype=jnp.float32)
    mask = jnp.zeros((2, 32), bool)
    sv = jax.random.normal(jax.random.key(3), (2, 16), dtype=jnp.float32)
    gv = jax.random.normal(jax.random.key(4), (2, 6), dtype=jnp.float32)
    carry = jax.random.normal(jax.random.key(5), (2, 32), dtype=jnp.float32)
    before_args = (e, mask, sv, gv, carry)
    after_args = (jnp.concatenate([e, history], -1), mask, sv, gv, carry)
    p = old.init(jax.random.key(6), *before_args)
    initialized = new.init(jax.random.key(6), *after_args)
    q = merge_visible_history_params(initialized, p)
    ref = jax.jit(old.apply)(p, *before_args)
    got = jax.jit(new.apply)(q, *after_args)
    for a, b in zip(jax.tree.leaves(ref), jax.tree.leaves(got)):
        np.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-6)
    for name, value in p['params'].items():
        for a, b in zip(jax.tree.leaves(value), jax.tree.leaves(q['params'][name])):
            np.testing.assert_array_equal(a, b)
    grad = jax.grad(lambda w: new.apply(w, *after_args)[0].button[:, 2].sum())(q)
    assert float(jnp.abs(grad['params']['visible_history']['kernel']).sum()) > 0
    # A learned history projection remains exactly inert when its inputs are zero.
    q['params']['visible_history']['kernel'] = jnp.ones_like(q['params']['visible_history']['kernel'])
    ablated = new.apply(q, jnp.concatenate([e, jnp.zeros_like(history)], -1), mask, sv, gv, carry)
    for a, b in zip(jax.tree.leaves(ref), jax.tree.leaves(ablated)):
        np.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-6)


def test_visible_history_actor_learner_reset_and_resume():
    from lanerl_jax.sim.config import SimConfig
    from lanerl_jax.sim.init import lane_params
    from lanerl_jax.train.wave_scenario import raw_state
    from lanerl_jax.train.vec_train import VecConfig, make_vec_train
    from lanerl_jax.train.ppo import factored_log_prob
    from lanerl_jax.train.replay_audit import serialize_replay_state, restore_replay_state

    pc = small_config()._replace(observation_interface=VISIBLE_HISTORY_INTERFACE, entity_dim=HISTORY_ENTITY_DIM)
    cfg = VecConfig(n_envs=2, rollout_steps=4, n_minibatches=2, opponent='afk', policy=pc,
                    episode_s=120.6, stagger_initial=False)
    states = [raw_state(SimpleNamespace(params=lane_params()), seed) for seed in (0, 1)]
    bank = jax.tree.map(lambda a,b: jnp.stack([a,b]), *states)
    built = make_vec_train(cfg, SimConfig.training().replace(step_ticks=6), bank)
    r = built['initial_runner'](jax.random.key(0))
    p = jax.tree.map(lambda x: x, r.params)
    kernel = p['params']['visible_history']['kernel']
    p['params']['visible_history']['kernel'] = .02*jax.random.normal(jax.random.key(7), kernel.shape, dtype=kernel.dtype)
    r = r._replace(params=p)
    rollout = jax.jit(built['rollout'])
    r2, tr, batch = rollout(r)
    assert np.asarray(tr.obs_entities[..., 16:]).any(), 'history never entered actor inputs'
    assert np.asarray(r2.visible_history.known).any(), 'memory was not retained across rollout'
    lg = built['loss'].forward(r.params, batch)
    lp = factored_log_prob((lg.button, lg.screen_x, lg.screen_y), batch['action'], batch['uses_screen'])
    np.testing.assert_allclose(lp, batch['log_prob'], atol=1e-5, rtol=1e-4)
    fields = ('env_state', 'carry', 'rng', 'deadline_ms', 'visible_history')
    template = {k: getattr(r2, k) for k in fields}
    restored = restore_replay_state(template, serialize_replay_state(template))
    resumed = r2._replace(**jax.tree.map(jnp.asarray, restored))
    r3, tr2, _ = rollout(r2)
    rr, rt, _ = rollout(resumed)
    for a,b in zip(jax.tree.leaves(tr2), jax.tree.leaves(rt)):
        np.testing.assert_array_equal(a, b)
    for a,b in zip(jax.tree.leaves(r3.visible_history), jax.tree.leaves(rr.visible_history)):
        np.testing.assert_array_equal(a, b)
    done = np.asarray(tr2.done[:, :, 0])
    assert done.any()
    for t, env in zip(*np.nonzero(done)):
        if t+1 < len(done):
            assert not np.asarray(tr2.obs_entities[t+1, env, :, :, 16:]).any(), 'history crossed reset'
    trained, metrics = jax.jit(built['run_chunk'], static_argnums=1)(r2, 1)
    assert np.isfinite(np.asarray(metrics['policy_loss'])).all()
    assert int(trained.step) == cfg.n_envs*cfg.learn_agents*cfg.rollout_steps
