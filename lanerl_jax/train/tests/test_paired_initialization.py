"""Check initialization against the actual policy schema and forward pass."""
import numpy as np
import jax
import jax.numpy as jnp
from flax.serialization import to_bytes
from lanerl_jax.train.policy import LanePolicy, PolicyConfig
from lanerl_jax.train.paired_vec_train import adjust_button_bias


def test_button_shift_preserves_farming_conditional_and_other_weights():
    cfg = PolicyConfig(d_model=16, n_heads=2, ffn_dim=32, ctx_dim=16,
                       core_dim=16, mlp_hidden=32, core='gru',
                       core_norm=True, core_residual=True)
    policy = LanePolicy(cfg)
    args = (jnp.zeros((1, cfg.n_slots, cfg.entity_dim)),
            jnp.zeros((1, cfg.n_slots), bool),
            jnp.zeros((1, cfg.self_dim)), jnp.zeros((1, cfg.global_dim)),
            policy.initial_carry((1,)))
    params = policy.init(jax.random.key(0), *args)
    original = to_bytes(params)
    altered = adjust_button_bias(params, {'e': 10.}, cfg)
    old, carry = policy.apply(params, *args)
    new, new_carry = policy.apply(altered, *args)
    expected = np.zeros((1, cfg.n_buttons)); expected[:, 5] = 10.
    np.testing.assert_allclose(new.button-old.button, expected, atol=1e-6)
    for a, b in ((old.screen_x,new.screen_x), (old.screen_y,new.screen_y),
                 (old.value,new.value), (carry,new_carry)):
        np.testing.assert_array_equal(a,b)
    a = np.asarray(jax.nn.softmax(old.button)); b = np.asarray(jax.nn.softmax(new.button))
    np.testing.assert_allclose(a[:,1]/a[:,2], b[:,1]/b[:,2], rtol=1e-6)
    assert to_bytes(params) == original
    for path, value in jax.tree_util.tree_flatten_with_path(params)[0]:
        other = altered
        for entry in path: other = other[entry.key]
        if tuple(entry.key for entry in path) != ('params','Dense_7','bias'):
            np.testing.assert_array_equal(value, other)
