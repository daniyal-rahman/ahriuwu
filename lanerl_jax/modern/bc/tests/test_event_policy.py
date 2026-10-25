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


def test_relative_angle_rotation_keeps_probability_mass_and_full_direction_support():
    from lanerl_jax.modern.bc.event_policy import rotate_angle_logits
    logits = jnp.full((3, 72), -10.).at[:, 36].set(10.)
    rotated = jax.nn.softmax(rotate_angle_logits(logits, jnp.array([0., jnp.pi/2, -jnp.pi/2])), -1)
    np.testing.assert_array_equal(rotated.argmax(-1), [36, 54, 18])
    np.testing.assert_allclose(rotated.sum(-1), 1., atol=1e-6)
    assert np.asarray(rotated).min() > 0
    uniform = rotate_angle_logits(jnp.zeros((2, 72)), jnp.array([.13, -2.]))
    np.testing.assert_allclose(jax.nn.softmax(uniform, -1), 1/72, atol=1e-6)


def test_missing_opponent_relative_angle_equals_global_head_without_new_parameters():
    c = EventPolicyConfig(d_model=16, n_heads=2, n_layers=1, ffn_dim=32, ctx_dim=16,
                          mlp_hidden=32, mlp_layers=1, core_dim=16, target_dim=16, pointer_dim=8)
    absolute, relative = EventPolicy(c), EventPolicy(c._replace(opponent_relative_angle=True))
    args = [jnp.zeros((32, c.entity_dim)), jnp.ones(32, bool), jnp.zeros(c.self_dim),
            jnp.zeros(c.global_dim), jnp.zeros(32, jnp.int32), jnp.asarray(1), jnp.asarray(2),
            jnp.zeros((4, c.metadata_dim)), jnp.zeros((4, c.metadata_dim)), jnp.asarray(0),
            absolute.initial_carry()]
    params = absolute.init(jax.random.key(2), *args)
    a, _ = absolute.apply(params, *args)
    b, _ = relative.apply(params, *args)
    np.testing.assert_allclose(jax.nn.softmax(a.angle), jax.nn.softmax(b.angle), atol=1e-6)
    np.testing.assert_array_equal(a.button, b.button)
    np.testing.assert_array_equal(a.grid, b.grid)


def test_previous_event_ablation_preserves_own_and_enemy_cast_history():
    c = EventPolicyConfig(d_model=16, n_heads=2, n_layers=1, ffn_dim=32, ctx_dim=16,
                          mlp_hidden=32, mlp_layers=1, core_dim=16, target_dim=16, pointer_dim=8,
                          previous_event_history=False)
    policy = EventPolicy(c)
    args = [jnp.zeros((32, c.entity_dim)), jnp.arange(32) > 2, jnp.zeros(c.self_dim),
            jnp.zeros(c.global_dim), jnp.zeros(32, jnp.int32), jnp.asarray(1), jnp.asarray(2),
            jnp.zeros((4, c.metadata_dim)), jnp.zeros((4, c.metadata_dim)), jnp.asarray(0),
            policy.initial_carry()]
    params = policy.init(jax.random.key(3), *args)
    before, state = policy.apply(params, *args)
    args[0] = args[0].at[..., 24:].set(.9)
    args[2] = args[2].at[28:].set(.9)
    after, state_after = policy.apply(params, *args)
    np.testing.assert_array_equal(before.button, after.button)
    np.testing.assert_array_equal(state, state_after)
    enabled = EventPolicy(c._replace(previous_event_history=True))
    changed, _ = enabled.apply(params, *args)
    args[0] = args[0].at[..., 24:].set(0.)
    args[2] = args[2].at[28:].set(0.)
    original, _ = enabled.apply(params, *args)
    assert not np.allclose(original.button, changed.button)
    args[2] = args[2].at[24:28].set(.9)
    own, _ = policy.apply(params, *args)
    assert not np.allclose(before.button, own.button)
    args[2] = args[2].at[24:28].set(0.)
    args[0] = args[0].at[..., 20:24].set(.9)
    enemy, _ = policy.apply(params, *args)
    assert not np.allclose(before.button, enemy.button)


def test_next_event_outputs_are_independent_of_teacher_aim_button():
    c = EventPolicyConfig(d_model=16, n_heads=2, n_layers=1, ffn_dim=32, ctx_dim=16,
                          mlp_hidden=32, mlp_layers=1, core_dim=16, target_dim=16, pointer_dim=8,
                          next_event_horizon=20)
    policy = EventPolicy(c)
    args = (jnp.zeros((32, c.entity_dim)), jnp.arange(32) > 2, jnp.zeros(c.self_dim),
            jnp.zeros(c.global_dim), jnp.zeros(32, jnp.int32), jnp.asarray(1), jnp.asarray(2),
            jnp.zeros((4, c.metadata_dim)), jnp.zeros((4, c.metadata_dim)), jnp.asarray(0),
            policy.initial_carry())
    params = policy.init(jax.random.key(4), *args, target_button=jnp.asarray(1))
    a, ca = policy.apply(params, *args, target_button=jnp.asarray(1))
    b, cb = policy.apply(params, *args, target_button=jnp.asarray(3))
    np.testing.assert_array_equal(a.button, b.button)
    np.testing.assert_array_equal(a.delay, b.delay)
    np.testing.assert_array_equal(ca, cb)
    assert a.delay.shape == (8, 21)
    from lanerl_jax.modern.bc.event_next import immediate_logits
    h, _, _ = policy.apply(params, *args, method=policy.encode)
    emitted = policy.apply(params, h, method=policy.buttons)
    np.testing.assert_allclose(emitted, immediate_logits(a.button, a.delay), atol=1e-6)
