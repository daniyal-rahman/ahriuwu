"""The masked joint click distribution (INT-001 principled fix)."""
import jax, jax.numpy as jnp, numpy as np
from lanerl_jax.train.ppo import factored_log_prob, joint_click_logits, joint_click_entropy, _entropy
from lanerl_jax.train.trainer import _sample
from lanerl_jax.train.policy import ActionLogits

N_X, N_Y = 96, 54

def _logits(key, n=64):
    kb, kx, ky = jax.random.split(key, 3)
    return ActionLogits(button=jax.random.normal(kb, (n, 8)), screen_x=jax.random.normal(kx, (n, N_X)) * 2,
                        screen_y=jax.random.normal(ky, (n, N_Y)) * 2, value=jnp.zeros((n,)))

def test_all_true_mask_equals_factored():
    lg = _logits(jax.random.key(0)); mask = jnp.ones((64, N_X, N_Y), bool)
    a = (jnp.full((64,), 1), jnp.arange(64) % N_X, jnp.arange(64) % N_Y)
    lp_f = factored_log_prob((lg.button, lg.screen_x, lg.screen_y), a, jnp.ones(64), jnp.zeros(64))
    lp_m = factored_log_prob((lg.button, lg.screen_x, lg.screen_y), a, jnp.ones(64), jnp.zeros(64), click_mask=mask)
    np.testing.assert_allclose(np.asarray(lp_f), np.asarray(lp_m), atol=1e-4)
    ent_f = _entropy(lg.screen_x) + _entropy(lg.screen_y)
    np.testing.assert_allclose(np.asarray(ent_f), np.asarray(joint_click_entropy(lg.screen_x, lg.screen_y, mask)), atol=1e-3)

def test_masked_sampling_never_picks_a_masked_cell_and_logprob_matches():
    lg = _logits(jax.random.key(1)); rng = np.random.default_rng(0)
    mask = jnp.asarray(rng.random((64, N_X, N_Y)) < 0.3)
    a, lp, usage = _sample(lg, jax.random.key(2), None, click_mask=mask)
    a = tuple(np.asarray(x) for x in a)
    assert all(bool(mask[i, a[1][i], a[2][i]]) for i in range(64))
    lp2 = factored_log_prob((lg.button, lg.screen_x, lg.screen_y), tuple(jnp.asarray(x) for x in a), *usage, click_mask=mask)
    np.testing.assert_allclose(np.asarray(lp), np.asarray(lp2), atol=1e-5)
    # masked distribution has no mass off the mask and lower entropy than the unmasked one
    p = jnp.exp(jax.nn.log_softmax(joint_click_logits(lg.screen_x, lg.screen_y, mask), -1)).reshape(64, N_X, N_Y)
    assert float((p * ~mask).sum()) == 0.0
    assert float(joint_click_entropy(lg.screen_x, lg.screen_y, mask).mean()) < float((_entropy(lg.screen_x) + _entropy(lg.screen_y)).mean())

def test_click_mask_from_position_shapes():
    from lanerl_jax.train.actions import click_mask_from_position
    m = click_mask_from_position(jnp.float32(1500.), jnp.float32(12800.), jnp.asarray([0.697, 0.717]), jnp.asarray([-0.717, 0.697]))
    assert m.shape == (N_X, N_Y) and 0.2 < float(m.mean()) < 0.9

def test_masked_loss_gradient_is_finite():
    """The E20 canary: entropy/log-prob gradients through the mask must be finite."""
    from lanerl_jax.train.ppo import joint_click_log_prob
    lg = _logits(jax.random.key(3), n=16); rng = np.random.default_rng(1)
    mask = jnp.asarray(rng.random((16, N_X, N_Y)) < 0.3)
    a_x = jnp.argmax(jnp.where(mask.any(-1), 1.0, 0.0), -1); a_y = jnp.argmax(mask[jnp.arange(16), a_x], -1)
    def loss(lx, ly):
        return (joint_click_entropy(lx, ly, mask) + joint_click_log_prob(lx, ly, mask, a_x, a_y)).sum()
    gx, gy = jax.grad(loss, argnums=(0, 1))(lg.screen_x, lg.screen_y)
    assert bool(jnp.isfinite(gx).all()) and bool(jnp.isfinite(gy).all())
