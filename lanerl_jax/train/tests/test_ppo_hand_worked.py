"""PPO correctness against HAND-WORKED numbers (`docs/EXPERIMENT_METHOD.md` s3).

`test_ppo.py` checks the port against the torch implementation it came from,
which catches a transcription error but not a shared misreading. These check
each mechanism against numbers worked by hand in the docstrings (or, for the
log-prob gradient, against the closed form ``onehot(a) - softmax(z)`` in plain
numpy), so an implementation and its reference cannot agree by sharing a bug.

Needs no torch.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from lanerl_rl import constants as C

from lanerl_jax.train.ppo import (
    PPOConfig,
    factored_log_prob,
    gae,
    kl_stopped_epochs,
    summarise_minibatches,
    value_loss,
)


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


# ------------------------------------------- masked factored log-prob -----
#: Which auxiliary heads each button puts on the wire, written out from
#: `train/actions.orders_from` by hand -- NOT read from `ppo.USES_*`, which
#: is what is under test.
_USES = {"noop": (0, 0), "recall": (0, 0), "q": (0, 0), "w": (0, 0),
         "e": (0, 0), "move": (1, 0), "attack_move": (1, 1), "r": (0, 1)}


def _np_log_softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def test_masked_log_prob_and_its_gradient_match_numpy_per_button_class():
    """For each of the 8 buttons (4 classes: none / screen / screen+target /
    target) the log-prob is ``lp_b + s*(lp_x + lp_y) + t*lp_t`` and its
    gradient w.r.t. each head's logits is ``u * (onehot(a) - softmax(z))``
    with ``u`` = 1 for the button and the head's usage bit otherwise -- ZERO
    on a head the button does not use. Screen/target widths are shrunk to
    3/4/5; the masking does not depend on them.
    """
    rng = np.random.default_rng(11)
    widths = (len(C.BUTTONS), 3, 4, 5)
    n = len(C.BUTTONS)
    z = [rng.normal(size=(n, k)).astype(np.float64) for k in widths]
    a = [np.arange(n)] + [rng.integers(0, k, size=n) for k in widths[1:]]
    use = np.zeros((n, 4))
    use[:, 0] = 1.0
    for name, b in C.BUTTON_INDEX.items():
        s, t = _USES[name]
        use[b, 1] = use[b, 2] = s
        use[b, 3] = t
    assert {tuple(u) for u in use} == {(1, 0, 0, 0), (1, 1, 1, 0),
                                       (1, 1, 1, 1), (1, 0, 0, 1)}

    want_lp = sum(use[:, h] * _np_log_softmax(z[h])[np.arange(n), a[h]]
                  for h in range(4))
    want_grad = []
    for h in range(4):
        p = np.exp(_np_log_softmax(z[h]))
        onehot = np.eye(widths[h])[a[h]]
        want_grad.append(use[:, h:h + 1] * (onehot - p))

    zj = [jnp.asarray(x, jnp.float32) for x in z]
    aj = [jnp.asarray(x) for x in a]
    got_lp = factored_log_prob(zj, aj)
    got_grad = jax.grad(lambda zz: factored_log_prob(zz, aj).sum())(zj)

    for name, b in C.BUTTON_INDEX.items():
        assert float(got_lp[b]) == pytest.approx(want_lp[b], abs=1e-5), name
        for h, head in enumerate(("button", "screen_x", "screen_y", "target")):
            np.testing.assert_allclose(
                np.asarray(got_grad[h][b]), want_grad[h][b], rtol=0, atol=1e-5,
                err_msg=f"d logp / d {head} logits for button {name}")


# ------------------------------------------------------- value loss -------
def test_value_loss_clipped_and_unclipped_hand_numbers():
    """v = [1.5, 0.0], old = [1.0, 1.0], returns = [2.0, 1.0], eps = 0.2.

    unclipped squared errors: (1.5-2)^2 = 0.25, (0-1)^2 = 1
        OFF: 0.5 * mean(0.25, 1) = 0.3125
    clipped predictions: 1 + clip(0.5) = 1.2, 1 + clip(-1) = 0.8
        clipped squared errors: (1.2-2)^2 = 0.64, (0.8-1)^2 = 0.04
        ON:  0.5 * mean(max(.25, .64), max(1, .04)) = 0.5 * 0.82 = 0.41
    """
    v = jnp.asarray([1.5, 0.0])
    old = jnp.asarray([1.0, 1.0])
    ret = jnp.asarray([2.0, 1.0])
    on = PPOConfig(value_clip_eps=0.2, clip_value_loss=True)
    off = on._replace(clip_value_loss=False)
    assert float(value_loss(v, old, ret, on)) == pytest.approx(0.41, abs=1e-6)
    assert float(value_loss(v, old, ret, off)) == pytest.approx(0.3125, abs=1e-6)


# ------------------------------------------------ KL early stop -----------
def _toy(kl_of_p):
    """One scalar parameter ``p`` pulled to 1 by ``0.5 (p - 1)^2``; the
    minibatch data is identical everywhere, so the shuffle cannot matter."""
    def loss_fn(params, b):
        p = params["p"][0]
        loss = 0.5 * (p - 1.0) ** 2 + 0.0 * b["x"].sum()
        return loss, {"approx_kl": kl_of_p(p)}
    return loss_fn


def test_kl_stop_freezes_params_and_opt_state_at_the_first_over_target_minibatch():
    """2 epochs x 2 minibatches, Adam(lr 0.5, b1 .9, b2 .999), p0 = 0,
    ``approx_kl = p^2`` (measured before the step), ``target_kl = 0.3``.

        mb (0,0)  p = 0      kl 0      applied  g = -1
                  m = -.1, v = .001; mhat = -1, vhat = 1 -> step .5 -> p = .5
        mb (0,1)  p = .5     kl .25    applied  g = -.5
                  m = .9(-.1) + .1(-.5) = -.14
                  v = .999(.001) + .001(.25) = .001249
                  mhat = -.14/.19 = -.736842, vhat = .001249/.001999 = .624812
                  step = .5 * .736842 / sqrt(.624812) = .466091 -> p = .966091
        mb (1,0)  p = .966   kl .933   WITHHELD (first over target)
        mb (1,1)  withheld (latched)

    So p = .966091 and Adam's state is exactly its 2-step state
    (count 2, mu -.14, nu .001249). Applying the over-target minibatch
    (the `PPO-05` bug) gives count 3 and p > .966; not stopping at all gives
    count 4.

    The summary (`PPO-11`): means over the 2 APPLIED minibatches --
    approx_kl (0 + .25)/2 = .125, grad_norm (1 + .5)/2 = .75, grad_clipped
    at max_grad_norm .8 is (1 + 0)/2 = .5 -- while the plain 4-way mean would
    report approx_kl (0 + .25 + .933 + .933)/4 = .529. kl_stopped = .5.
    """
    tx = optax.adam(0.5, b1=0.9, b2=0.999, eps=1e-8)
    params = {"p": jnp.zeros((1,), jnp.float32)}
    opt0 = tx.init(params)
    p, opt, _, info = kl_stopped_epochs(
        _toy(lambda p: p ** 2), tx, params, opt0,
        {"x": jnp.zeros((4,))}, jax.random.key(0),
        epochs=2, n_minibatches=2, target_kl=0.3, max_grad_norm=0.8)

    assert float(p["p"][0]) == pytest.approx(0.966091, abs=1e-5)
    adam = opt[0]
    assert int(adam.count) == 2
    assert float(adam.mu["p"][0]) == pytest.approx(-0.14, abs=1e-6)
    assert float(adam.nu["p"][0]) == pytest.approx(0.001249, abs=1e-8)
    np.testing.assert_array_equal(np.asarray(info["applied"]),
                                  [[1.0, 1.0], [0.0, 0.0]])

    s = summarise_minibatches(info)
    assert float(s["approx_kl"]) == pytest.approx(0.125, abs=1e-6)
    assert float(s["grad_norm"]) == pytest.approx(0.75, abs=1e-6)
    assert float(s["grad_clipped"]) == pytest.approx(0.5, abs=1e-6)
    assert float(s["kl_stopped"]) == pytest.approx(0.5, abs=1e-6)
    assert float(s["loss_nonfinite"]) == 0.0


def test_a_nan_kl_withholds_everything_and_is_reported():
    """``NaN > target`` is False; the stop is written ``~(kl <= target)`` so a
    NaN stops. Nothing is applied: params and Adam state are exactly the
    initial ones, the applied-only means are NaN (no applied minibatch), and
    ``loss_nonfinite`` -- over ALL minibatches -- is 1, which is what the
    divergence guard reads."""
    tx = optax.adam(0.5)
    params = {"p": jnp.zeros((1,), jnp.float32)}
    opt0 = tx.init(params)
    p, opt, _, info = kl_stopped_epochs(
        _toy(lambda p: jnp.nan * p), tx, params, opt0,
        {"x": jnp.zeros((4,))}, jax.random.key(0),
        epochs=2, n_minibatches=2, target_kl=0.3, max_grad_norm=1.0)
    assert float(p["p"][0]) == 0.0
    for got, want in zip(jax.tree.leaves(opt), jax.tree.leaves(opt0)):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    s = summarise_minibatches(info)
    assert np.isnan(float(s["approx_kl"]))
    assert float(s["kl_stopped"]) == 1.0
    assert float(s["loss_nonfinite"]) == 1.0
