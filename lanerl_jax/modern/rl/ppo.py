"""PPO equations for the three-head screen-click distribution.

Adapted from Chris Lu's PureJaxRL ppo_rnn.py at 31756b197773a52db763fdbe6d635e4b46522a73 (Apache-2.0):
GAE with post-action dones, clipped surrogate and value losses, nested epoch/minibatch scans over
agent-major trajectories.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax


class PPOConfig(NamedTuple):
    decision_hz: float = 10.0
    discount: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01 / 3      # PureJaxRL's 0.01 spread over the three heads
    kl_prior_coef: float = 0.0
    max_grad_norm: float = 0.5
    lr: float = 2.5e-4
    epochs: int = 4
    n_minibatches: int = 4
    normalize_advantage: bool = True


def gae(rewards, values, dones, last_value, gamma: float, lam: float):
    """Reverse-scan GAE over [T, ...]; ``dones[t]`` ends action t and masks the bootstrap."""
    resets = jnp.concatenate((jnp.zeros_like(dones[:1]), dones[:-1]), axis=0)

    def step(carry, transition):
        adv, next_value, next_done = carry
        done, value, reward = transition
        delta = reward + gamma * next_value * (1 - next_done) - value
        adv = delta + gamma * lam * (1 - next_done) * adv
        return (adv, value, done), adv

    _, advantages = jax.lax.scan(step, (jnp.zeros_like(last_value), last_value, dones[-1]),
                                 (resets, values, rewards), reverse=True, unroll=16)
    return advantages, advantages + values


def _chosen(lg, a):
    lp = jax.nn.log_softmax(lg, axis=-1)
    return jnp.take_along_axis(lp, a[..., None].astype(jnp.int32), axis=-1)[..., 0]


def _entropy(lg):
    lp = jax.nn.log_softmax(lg, axis=-1)
    return -jnp.sum(jnp.exp(lp) * lp, axis=-1)


def factored_log_prob(logits, actions, uses_screen):
    """Button log-prob plus the click log-prob where the button uses the screen."""
    return _chosen(logits[0], actions[0]) + uses_screen * (_chosen(logits[1], actions[1])
                                                           + _chosen(logits[2], actions[2]))


def factored_entropy(logits):
    """Unconditional sum of the head entropies."""
    return _entropy(logits[0]) + (_entropy(logits[1]) + _entropy(logits[2]))


def policy_loss(log_prob, old_log_prob, adv, cfg: PPOConfig):
    """Clipped surrogate (advantages normalised within the minibatch) and its KL/clip diagnostics."""
    ratio = jnp.exp(log_prob - old_log_prob)
    if cfg.normalize_advantage:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    loss = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv)
    logr = log_prob - old_log_prob
    return loss.mean(), {"approx_kl": ((ratio - 1.0) - logr).mean(),
                         "clip_frac": (jnp.abs(ratio - 1.0) > cfg.clip_eps).mean()}


def value_loss(value, old_value, returns, cfg: PPOConfig):
    """Clipped value loss with the actor's ``clip_eps``."""
    clipped = old_value + (value - old_value).clip(-cfg.clip_eps, cfg.clip_eps)
    return 0.5 * jnp.maximum(jnp.square(value - returns), jnp.square(clipped - returns)).mean()


def update_epochs(loss_fn, tx, params, opt_state, batch, rng, *, epochs: int, n_minibatches: int,
                  max_grad_norm: float):
    """``epochs`` passes of shuffled minibatches over the leading (row) axis; diagnostics averaged."""
    n = batch["value"].shape[0]
    if epochs < 1 or n_minibatches < 1 or n % n_minibatches:
        raise ValueError("positive epochs/minibatches required; minibatches must divide batch rows")

    def epoch(state, _):
        params, opt_state, rng = state
        rng, key = jax.random.split(rng)
        perm = jax.random.permutation(key, n)
        mbs = jax.tree.map(lambda x: jnp.reshape(jnp.take(x, perm, axis=0), (n_minibatches, -1) + x.shape[1:]),
                           batch)

        def minibatch(state, mb):
            params, opt_state = state
            (total, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, mb)
            grad_norm = optax.global_norm(grads)
            updates, opt_state = tx.update(grads, opt_state, params)
            return (optax.apply_updates(params, updates), opt_state), {
                **info, "grad_norm": grad_norm, "grad_clipped": (grad_norm > max_grad_norm).astype(jnp.float32),
                "loss_nonfinite": (~jnp.isfinite(total) | ~jnp.isfinite(grad_norm)).astype(jnp.float32)}

        (params, opt_state), info = jax.lax.scan(minibatch, (params, opt_state), mbs)
        return (params, opt_state, rng), info

    (params, opt_state, rng), info = jax.lax.scan(epoch, (params, opt_state, rng), None, length=epochs)
    return params, opt_state, rng, jax.tree.map(jnp.mean, info)
