import jax
import jax.numpy as jnp
import numpy as np

from lanerl_jax.modern.bc.event_policy import EventPolicy, EventPolicyConfig


def test_conditional_target_heads_do_not_leak_the_human_button_into_state():
    c = EventPolicyConfig(d_model=16, n_heads=2, n_layers=1, ffn_dim=32, ctx_dim=16,
                          mlp_hidden=32, mlp_layers=1, core_dim=16, target_dim=16, pointer_dim=8)
    policy = EventPolicy(c)
    args = (jnp.zeros((32, c.entity_dim)), jnp.arange(32) > 2, jnp.full((c.self_dim,), jnp.nan),
            jnp.zeros(c.global_dim), jnp.zeros(32, jnp.int32), jnp.asarray(1), jnp.asarray(2),
            jnp.full((4, c.metadata_dim), jnp.nan), jnp.full((4, c.metadata_dim), jnp.nan),
            jnp.asarray(0), policy.initial_carry())
    params = policy.init(jax.random.key(0), *args, target_button=jnp.asarray(1))
    a, ca = policy.apply(params, *args, target_button=jnp.asarray(1))
    b, cb = policy.apply(params, *args, target_button=jnp.asarray(3))
    np.testing.assert_array_equal(a.button, b.button)
    np.testing.assert_array_equal(ca, cb)
    assert not np.allclose(a.grid, b.grid)
    assert np.isfinite(a.grid).all() and (np.asarray(a.pointer)[3:] < -1000).all()


def test_own_age_ablation_does_not_remove_other_history_or_change_parameter_shapes():
    c = EventPolicyConfig(d_model=16, n_heads=2, n_layers=1, ffn_dim=32, ctx_dim=16,
                          mlp_hidden=32, mlp_layers=1, core_dim=16, target_dim=16, pointer_dim=8,
                          own_cast_history=False)
    policy = EventPolicy(c)
    args = [jnp.zeros((32, c.entity_dim)), jnp.arange(32) > 2, jnp.zeros(c.self_dim),
            jnp.zeros(c.global_dim), jnp.zeros(32, jnp.int32), jnp.asarray(1), jnp.asarray(2),
            jnp.zeros((4, c.metadata_dim)), jnp.zeros((4, c.metadata_dim)), jnp.asarray(0),
            policy.initial_carry()]
    params = policy.init(jax.random.key(1), *args)
    before, state = policy.apply(params, *args)
    args[2] = args[2].at[24:28].set(.9)
    after, state_after = policy.apply(params, *args)
    np.testing.assert_array_equal(before.button, after.button)
    np.testing.assert_array_equal(state, state_after)
    enabled = EventPolicy(c._replace(own_cast_history=True))
    changed, _ = enabled.apply(params, *args)
    args[2] = args[2].at[24:28].set(0.)
    original, _ = enabled.apply(params, *args)
    assert not np.allclose(original.button, changed.button)
    args[2] = args[2].at[28:].set(.9)
    other_history, _ = policy.apply(params, *args)
    assert not np.allclose(before.button, other_history.button)
