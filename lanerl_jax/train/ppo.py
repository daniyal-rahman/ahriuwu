"""Dual-clip PPO, ported from ``lanerl_rl/ppo.py``.

The pieces, and why each is the way it is
-----------------------------------------

**Dual clip.** Standard PPO's clipped surrogate is unbounded below when the
importance ratio explodes on a *negative* advantage: ``min(rA, clip(r)A)``
equals ``rA`` there and ``r`` can be arbitrarily large, so one bad minibatch
produces an enormous gradient. Ye et al. (2020), *Mastering Complex Control in
MOBA Games* (JueWu), add a second clip for that case::

    A >= 0:  L = min(rA, clip(r, 1-e, 1+e) A)
    A <  0:  L = max( min(rA, clip(r, 1-e, 1+e) A), cA )      c = 3.0

``c > 1`` is required for the bound to be a relaxation rather than a constraint.

**The discount is a horizon in seconds, not a raw gamma.** A raw gamma means a
different amount of *game time* at every decision rate, so copying one across a
rate change silently changes the objective. ``gamma = 1 - 1/(horizon_s *
decision_hz)``: at 30 Hz, 120 s gives 0.999722, and the same 120 s at 15 Hz
would give 0.999444. The source file records that its own worked example said
15 Hz long after the stack moved to 30 -- exactly the silent change it warns
about.

**The action distribution is factored** over four independent heads (button,
screen_x, screen_y, target), so log-probs and entropies **sum**. Maximum
factored entropy is ``ln 8 + ln 96 + ln 54 + ln 32`` = **14.099 nats**.

Do not reuse the 9.940 that appears in `lanerl_rl/ppo.py`'s notes. That figure
is correct for the run it cites (`rl-overnight-0911-0608`, 2026-09-11) under the
action space of the time -- 8 buttons, a **9x9** move grid, 32 targets, which is
exactly ``ln 8 + ln 9 + ln 9 + ln 32 = 9.940``. The screen-space action landed
on 2026-09-14 (`cf63786`, "the policy clicks a point, not a direction") and
moved the ceiling to 14.099. The note is a historical record rather than a stale
constant, but reading it against a *current* run silently inflates the number:
the 8.876 it reports was 89% of the old maximum and would be **63%** of this
one.

**Early stopping is on the EXCESS KL, not the absolute.** An off-policy rollout
already carries staleness drift before a single gradient is taken, so stopping
on the absolute value stops on that. The reference is epoch 0's own mean.
"""
from __future__ import annotations

from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp

__all__ = [
    "PPOConfig", "gamma_for_horizon", "gae", "factored_log_prob",
    "factored_entropy", "policy_loss", "value_loss", "MAX_FACTORED_ENTROPY",
]


def gamma_for_horizon(horizon_s: float, decision_hz: float) -> float:
    """``1 - 1/(horizon_s * decision_hz)``. See the module docstring."""
    return 1.0 - 1.0 / (horizon_s * decision_hz)


class PPOConfig(NamedTuple):
    """Defaults are `runs/rl-league-0915e/resolved_config.json`."""

    horizon_s: float = 120.0
    decision_hz: float = 30.0
    gae_lambda: float = 0.99
    clip_eps: float = 0.2
    dual_clip: float = 3.0
    value_coef: float = 0.5
    value_clip_eps: float = 0.2
    clip_value_loss: bool = True
    entropy_coef: float = 0.001
    max_grad_norm: float = 1.0
    target_kl: float = 0.02
    lr: float = 1e-5
    critic_lr: float = 3e-4
    epochs: int = 4
    normalize_advantage: bool = True

    @property
    def gamma(self) -> float:
        return gamma_for_horizon(self.horizon_s, self.decision_hz)


#: ln 8 + ln 96 + ln 54 + ln 32 -- a uniform policy over the four heads.
MAX_FACTORED_ENTROPY = float(
    jnp.log(8.0) + jnp.log(96.0) + jnp.log(54.0) + jnp.log(32.0))


def gae(rewards, values, dones, last_value, gamma: float, lam: float):
    """Generalised advantage estimation over ``(T, ...)`` arrays.

    ``dones[t] == 1`` means step ``t`` is terminal, so the value of ``t+1`` must
    not be bootstrapped through it.

    Written as a reverse ``scan`` rather than a Python loop, which is what lets
    the whole update live inside one ``jit``. Returns ``(advantages, returns)``.
    """
    def step(carry, xs):
        gae_t, next_value = carry
        reward, value, done = xs
        nonterminal = 1.0 - done
        delta = reward + gamma * next_value * nonterminal - value
        gae_t = delta + gamma * lam * nonterminal * gae_t
        return (gae_t, value), gae_t

    (_, _), adv = jax.lax.scan(
        step, (jnp.zeros_like(last_value), last_value),
        (rewards, values, dones), reverse=True)
    return adv, adv + values


def factored_log_prob(logits, actions) -> jax.Array:
    """Sum of per-head log-probs. ``logits``/``actions`` are matching pytrees."""
    total = None
    for lg, a in zip(logits, actions):
        lp = jax.nn.log_softmax(lg, axis=-1)
        chosen = jnp.take_along_axis(lp, a[..., None].astype(jnp.int32), axis=-1)[..., 0]
        total = chosen if total is None else total + chosen
    return total


def factored_entropy(logits) -> jax.Array:
    """Sum of per-head entropies, in nats. Max is :data:`MAX_FACTORED_ENTROPY`."""
    total = None
    for lg in logits:
        lp = jax.nn.log_softmax(lg, axis=-1)
        h = -jnp.sum(jnp.exp(lp) * lp, axis=-1)
        total = h if total is None else total + h
    return total


def policy_loss(log_prob, old_log_prob, adv, cfg: PPOConfig):
    """Dual-clip surrogate. Returns ``(loss, diagnostics)``."""
    ratio = jnp.exp(log_prob - old_log_prob)
    surr1 = ratio * adv
    surr2 = jnp.clip(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
    inner = jnp.minimum(surr1, surr2)
    # bound the objective below by c*A when A < 0
    dual = jnp.maximum(inner, cfg.dual_clip * adv)
    obj = jnp.where(adv < 0.0, dual, inner)

    logr = log_prob - old_log_prob
    return -obj.mean(), {
        # the Schulman k3 estimator, which is what the PyTorch side uses
        "approx_kl": ((jnp.exp(logr) - 1.0) - logr).mean(),
        "clip_frac": (jnp.abs(ratio - 1.0) > cfg.clip_eps).mean(),
        "dual_clip_frac": ((adv < 0.0) & (cfg.dual_clip * adv > inner)).mean(),
    }


def value_loss(value, old_value, returns, cfg: PPOConfig):
    """Clipped value loss, matching ``DualClipPPO.value_loss``."""
    unclipped = (value - returns) ** 2
    if not cfg.clip_value_loss:
        return 0.5 * unclipped.mean()
    clipped_v = old_value + jnp.clip(
        value - old_value, -cfg.value_clip_eps, cfg.value_clip_eps)
    return 0.5 * jnp.maximum(unclipped, (clipped_v - returns) ** 2).mean()
