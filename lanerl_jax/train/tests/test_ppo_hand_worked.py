"""Independent hand numbers retained through PPO-17.

Four-head likelihood/entropy and KL-stop tests retired: those paths no
longer exist. Reference/update regressions live in test_reference_ppo.py.
"""
import jax.numpy as jnp
import numpy as np
import pytest
from lanerl_jax.train.ppo import PPOConfig, gae, value_loss

# ------------------------------------------------------------------ GAE ----
def test_gae_hand_worked_reset_in_the_middle_and_last_step_bootstrap():
    """gamma = lam = 0.5, two envs, T = 4, ``last_value = 10``.

    rewards [1, 2, 3, 4], values [0.5, 1, 1.5, 2] in both envs.

    env 0, dones [0, 1, 0, 0] -- an episode ends at t=1, and t=3 is NOT
    terminal, so the last step bootstraps from ``last_value``::

        t=3  delta = 4 + .5*10  - 2   = 7      A = 7
        t=2  delta = 3 + .5*2   - 1.5 = 2.5    A = 2.5 + .25*7 = 4.25
        t=1  delta = 2 + 0      - 1   = 1      A = 1      (chain cut)
        t=0  delta = 1 + .5*1   - .5  = 1      A = 1 + .25*1 = 1.25
        returns = A + V = [1.75, 2, 5.75, 9]

    env 1, dones [0, 0, 0, 1] -- the last step is terminal, so ``last_value``
    must be IGNORED::

        t=3  delta = 4 - 2 = 2                 A = 2
        t=2  delta = 3 + .5*2   - 1.5 = 2.5    A = 2.5 + .25*2 = 3
        t=1  delta = 2 + .5*1.5 - 1   = 1.75   A = 1.75 + .25*3 = 2.5
        t=0  delta = 1 + .5*1   - .5  = 1      A = 1 + .25*2.5 = 1.625
        returns = [2.125, 3.5, 4.5, 4]
    """
    r = jnp.asarray([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    v = jnp.asarray([[0.5, 0.5], [1.0, 1.0], [1.5, 1.5], [2.0, 2.0]])
    d = jnp.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    adv, ret = gae(r, v, d, jnp.asarray([10.0, 10.0]), 0.5, 0.5)
    np.testing.assert_allclose(np.asarray(adv[:, 0]), [1.25, 1.0, 4.25, 7.0],
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(ret[:, 0]), [1.75, 2.0, 5.75, 9.0],
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(adv[:, 1]), [1.625, 2.5, 3.0, 2.0],
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(ret[:, 1]), [2.125, 3.5, 4.5, 4.0],
                               rtol=0, atol=1e-6)


def test_value_loss_clipped_hand_numbers():
    v, old, ret = map(jnp.asarray, ([1.5, 0.0], [1., 1.], [2., 1.]))
    assert float(value_loss(v, old, ret, PPOConfig())) == pytest.approx(.41, abs=1e-6)
