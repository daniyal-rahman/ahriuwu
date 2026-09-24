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
    expected_head_usage,
    factored_entropy,
    factored_log_prob,
    gae,
    head_usage,
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
#: Which auxiliary heads a sample puts on the wire, written out from
#: `train/actions.orders_from` by hand -- NOT read from `ppo.head_usage`,
#: which is what is under test. PER SAMPLE (`PPO-14`): attack_move reaches
#: the target head when its slot holds a unit (ATTACK) and the screen heads
#: when it does not (the fallback MOVE); r reaches the target head only when
#: its slot holds a unit (an empty slot sends target -1 whatever was picked).
def _np_usage(name, slot_ok):
    if name == "move":
        return (1, 0)
    if name == "attack_move":
        return (0, 1) if slot_ok else (1, 0)
    if name == "r":
        return (0, 1) if slot_ok else (0, 0)
    return (0, 0)


#: The per-BUTTON rule `PPO-01` shipped: attack_move counted screen AND
#: target "conservatively", r always target. `PPO-14` measured attack_move
#: decoding to ATTACK in 100.0% of 5.76 M decisions, so its screen term was
#: pure noise. Kept to show the new tests reject it.
def _np_usage_ppo01(name, slot_ok):
    return {"move": (1, 0), "attack_move": (1, 1), "r": (0, 1)}.get(name, (0, 0))


def _np_log_softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def _check_log_prob_and_grad(usage_rule):
    """Every button x {slot holds a unit, slot empty}: the log-prob is
    ``lp_b + s*(lp_x + lp_y) + t*lp_t`` and its gradient w.r.t. each head's
    logits is ``u * (onehot(a) - softmax(z))`` with ``u`` = 1 for the button
    and the SAMPLE's usage bit otherwise -- ZERO on a head the sample did not
    put on the wire. ``usage_rule`` is the numpy reference; the
    implementation is `ppo.head_usage` + `ppo.factored_log_prob`.
    Screen/target widths are shrunk to 3/4/5; the masking does not depend on
    them.
    """
    rng = np.random.default_rng(11)
    widths = (len(C.BUTTONS), 3, 4, 5)
    nb = len(C.BUTTONS)
    buttons = np.concatenate([np.arange(nb), np.arange(nb)])
    slot_ok = np.repeat([True, False], nb)
    n = len(buttons)
    z = [rng.normal(size=(n, k)).astype(np.float64) for k in widths]
    a = [buttons] + [rng.integers(0, k, size=n) for k in widths[1:]]
    valid = np.ones((n, widths[3]), bool)
    valid[np.arange(n), a[3]] = slot_ok
    use = np.zeros((n, 4))
    use[:, 0] = 1.0
    for i, b in enumerate(buttons):
        s, t = usage_rule(C.BUTTONS[b], bool(slot_ok[i]))
        use[i, 1] = use[i, 2] = s
        use[i, 3] = t

    want_lp = sum(use[:, h] * _np_log_softmax(z[h])[np.arange(n), a[h]]
                  for h in range(4))
    want_grad = []
    for h in range(4):
        p = np.exp(_np_log_softmax(z[h]))
        onehot = np.eye(widths[h])[a[h]]
        want_grad.append(use[:, h:h + 1] * (onehot - p))

    zj = [jnp.asarray(x, jnp.float32) for x in z]
    aj = [jnp.asarray(x) for x in a]
    masks = head_usage(aj[0], aj[3], jnp.asarray(valid))
    got_lp = factored_log_prob(zj, aj, *masks)
    got_grad = jax.grad(
        lambda zz: factored_log_prob(zz, aj, *masks).sum())(zj)

    for i, b in enumerate(buttons):
        tag = f"{C.BUTTONS[b]} slot_ok={bool(slot_ok[i])}"
        assert float(got_lp[i]) == pytest.approx(want_lp[i], abs=1e-5), tag
        for h, head in enumerate(("button", "screen_x", "screen_y", "target")):
            np.testing.assert_allclose(
                np.asarray(got_grad[h][i]), want_grad[h][i], rtol=0,
                atol=1e-5, err_msg=f"d logp / d {head} logits for {tag}")
    return use


def test_masked_log_prob_and_its_gradient_match_numpy_per_sample():
    """The per-sample rule (`PPO-14`), all four usage classes present:
    none (noop/recall/q/w/e, and r on an empty slot), screen (move, and
    attack_move on an empty slot), target (attack_move and r on a unit).
    attack_move is never screen+target any more."""
    use = _check_log_prob_and_grad(_np_usage)
    assert {tuple(u) for u in use} == {(1, 0, 0, 0), (1, 1, 1, 0),
                                       (1, 0, 0, 1)}


def test_the_PPO01_per_button_rule_fails_the_per_sample_test():
    """The rule the trainer used until 2026-09-24 must not pass the test
    above: its attack_move-on-a-unit samples carry the screen log-prob and a
    nonzero screen-head gradient the implementation (correctly) does not."""
    with pytest.raises(AssertionError, match="attack_move slot_ok=True"):
        _check_log_prob_and_grad(_np_usage_ppo01)


def _wire_distribution(z, valid):
    """Enumerate the joint ``(b, x, y, t)`` space of ONE observation and
    aggregate it by what `orders_from` puts on the wire (the button is always
    on it). Returns ``(rows, p_joint, keys, p_wire)``."""
    lp = [_np_log_softmax(h) for h in z]
    rows, pj, keys = [], [], []
    for b in range(z[0].shape[0]):
        name = C.BUTTONS[b]
        for x in range(z[1].shape[0]):
            for y in range(z[2].shape[0]):
                for t in range(z[3].shape[0]):
                    p = np.exp(lp[0][b] + lp[1][x] + lp[2][y] + lp[3][t])
                    if name == "move" or (name == "attack_move"
                                          and not valid[t]):
                        key = (b, "point", x, y)
                    elif name in ("attack_move", "r"):
                        key = (b, "unit", t if valid[t] else -1)
                    else:
                        key = (b,)
                    rows.append((b, x, y, t))
                    pj.append(p)
                    keys.append(key)
    p_wire = {}
    for k, p in zip(keys, pj):
        p_wire[k] = p_wire.get(k, 0.0) + p
    return np.asarray(rows), np.asarray(pj), keys, p_wire


@pytest.mark.parametrize("valid", [
    np.asarray([True, False, True, False, False]),   # some slot visible
    np.zeros(5, bool),                               # nothing visible
], ids=["some-visible", "none-visible"])
def test_log_prob_and_entropy_are_the_exact_wire_action_distribution(valid):
    """The derivation in `ppo`'s docstring, by brute force.

    Enumerate every joint sample (8 x 3 x 4 x 5 = 480), merge the samples
    `orders_from` cannot tell apart -- the screen point is irrelevant to an
    ATTACK, the slot to a MOVE, an empty slot to r -- and compute the exact
    probability of each wire action and the exact entropy of that
    distribution. The target logits carry the policy's -1e9 on empty slots.

    * ``factored_log_prob`` under `head_usage`'s per-sample masks equals
      ``log P(wire action)`` for every sample with nonzero probability;
    * ``factored_entropy`` under `expected_head_usage` equals the wire
      entropy -- with the target head's valid-slot mass ``q``, not a
      constant.

    The `PPO-01` rule (attack_move = screen + target, r = target) fails
    both whenever a slot is visible.
    """
    rng = np.random.default_rng(5)
    widths = (len(C.BUTTONS), 3, 4, 5)
    z = [rng.normal(size=k) for k in widths]
    z[3] = np.where(valid, z[3], -1e9)
    rows, pj, keys, p_wire = _wire_distribution(z, valid)
    live = pj > 0
    exact_lp = np.log([p_wire[k] for k, ok in zip(keys, live) if ok])
    probs = np.asarray(list(p_wire.values()))
    probs = probs[probs > 0]
    exact_h = float(-(probs * np.log(probs)).sum())

    n = int(live.sum())
    zj = [jnp.broadcast_to(jnp.asarray(h, jnp.float32), (n, h.shape[0]))
          for h in z]
    aj = [jnp.asarray(rows[live, h]) for h in range(4)]
    vj = jnp.broadcast_to(jnp.asarray(valid), (n, len(valid)))
    got_lp = np.asarray(factored_log_prob(zj, aj, *head_usage(aj[0], aj[3], vj)))
    np.testing.assert_allclose(got_lp, exact_lp, rtol=0, atol=1e-4)
    got_h = float(factored_entropy(
        [h[:1] for h in zj], *expected_head_usage(zj[0][:1], zj[3][:1],
                                                   vj[:1]))[0])
    assert got_h == pytest.approx(exact_h, abs=1e-4)

    # The PPO-01 rule, in numpy: fails wherever a slot is visible.
    lps = [_np_log_softmax(h) for h in z]
    old_use = np.asarray([_np_usage_ppo01(C.BUTTONS[b], True)
                          for b in rows[live, 0]], float)
    r = rows[live]
    old_lp = (lps[0][r[:, 0]] + old_use[:, 0] * (lps[1][r[:, 1]] + lps[2][r[:, 2]])
              + old_use[:, 1] * lps[3][r[:, 3]])
    p_b = np.exp(lps[0])
    ent = [float(-(np.exp(h) * h).sum()) for h in lps]
    am, rr, mv = (C.BUTTON_INDEX[k] for k in ("attack_move", "r", "move"))
    old_h = (ent[0] + (p_b[mv] + p_b[am]) * (ent[1] + ent[2])
             + (p_b[am] + p_b[rr]) * ent[3])
    if valid.any():
        assert not np.allclose(old_lp, exact_lp, rtol=0, atol=1e-4)
        assert old_h != pytest.approx(exact_h, abs=1e-3)


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
