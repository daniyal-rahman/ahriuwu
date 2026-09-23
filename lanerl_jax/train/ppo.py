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

**The action distribution is factored** over four heads (button, screen_x,
screen_y, target), but only the heads THE CHOSEN BUTTON PUTS ON THE WIRE
count. `train/actions.orders_from` is the only thing that decides that:
noop/recall/q/w/e read nothing beyond the button, move reads the screen
heads, attack_move reads the target head (and falls back to the screen
point when the slot is empty, so conservatively both), r reads the target
head. The log-prob is ``lp_b + uses_screen[b]*(lp_x+lp_y) +
uses_target[b]*lp_t`` and the entropy is ``H_b + P(uses_screen)*(H_x+H_y) +
P(uses_target)*H_t`` -- exactly the torch reference's `_head_usage`. The
port summed all four unconditionally (`PPO-01`, 2026-09-23): with E cast on
80% of decisions, 80% of the screen/target samples were pure noise in the
ratio -- zero-mean gradient of variance ~A^2, spurious clipping that also
cut the button's gradient, an inflated `approx_kl` feeding `target_kl`,
and an entropy figure that counted heads the behaviour never used.

The maximum of the masked entropy is ``ln(sum_b exp(c_b))`` where ``c_b`` is
the auxiliary entropy button ``b`` unlocks (``ln 96 + ln 54`` for move, that
plus ``ln 32`` for attack_move, ``ln 32`` for r, 0 otherwise): **12.050
nats**, attained by ``softmax(c)`` over the buttons, NOT by a uniform button
(which gives 5.08). The old 14.099 counted every head regardless.

**Even that is a ceiling, not an achievable value.** The target head is masked to
the *visible* entity slots, so its share of the budget is ``ln(n_visible)``, not
``ln 32``. At episode start only four units exist, so that head contributes
**zero** entropy -- there is nothing to be uncertain about -- and the whole
policy reads 10.63 even when the other three heads are exactly uniform
(2.079 + 4.564 + 3.989 + 0.000). Read "entropy as a fraction of maximum"
against the *achievable* maximum for the observation, or it will report
collapse where there is only an empty lane.

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
import numpy as np

from lanerl_rl.constants import BUTTON_INDEX, BUTTONS, N_SCREEN_X, N_SCREEN_Y, N_SLOTS

__all__ = [
    "PPOConfig", "gamma_for_horizon", "gae", "factored_log_prob",
    "factored_entropy", "policy_loss", "value_loss", "MAX_FACTORED_ENTROPY",
    "USES_SCREEN_HEADS", "USES_TARGET_HEAD",
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


def _head_usage():
    """Which auxiliary heads each button puts on the wire (module docstring).
    Read off `train/actions.orders_from`; if that decoder changes, this must.
    """
    uses_screen = np.zeros(len(BUTTONS), dtype=bool)
    uses_target = np.zeros(len(BUTTONS), dtype=bool)
    uses_screen[BUTTON_INDEX["move"]] = True
    # attack_move: the target head when the chosen slot holds a visible
    # unit, the screen point otherwise -- a function of the observation,
    # not of the button, so both may reach the wire.
    uses_screen[BUTTON_INDEX["attack_move"]] = True
    uses_target[BUTTON_INDEX["attack_move"]] = True
    uses_target[BUTTON_INDEX["r"]] = True
    return (jnp.asarray(uses_screen, jnp.float32),
            jnp.asarray(uses_target, jnp.float32))


USES_SCREEN_HEADS, USES_TARGET_HEAD = _head_usage()

#: ln(sum_b exp(c_b)) with c_b the auxiliary entropy button b unlocks -- the
#: supremum of the masked factored entropy, attained at softmax(c) over the
#: buttons with every auxiliary head uniform. 12.050 nats.
MAX_FACTORED_ENTROPY = float(jax.nn.logsumexp(
    USES_SCREEN_HEADS * (jnp.log(float(N_SCREEN_X)) + jnp.log(float(N_SCREEN_Y)))
    + USES_TARGET_HEAD * jnp.log(float(N_SLOTS))))


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


def _chosen(lg, a):
    lp = jax.nn.log_softmax(lg, axis=-1)
    return jnp.take_along_axis(lp, a[..., None].astype(jnp.int32), axis=-1)[..., 0]


def _entropy(lg):
    lp = jax.nn.log_softmax(lg, axis=-1)
    return -jnp.sum(jnp.exp(lp) * lp, axis=-1)


def factored_log_prob(logits, actions) -> jax.Array:
    """Log-prob of the joint action, counting only the heads the chosen
    button puts on the wire (module docstring). ``logits``/``actions`` are
    ``(button, screen_x, screen_y, target)`` sequences; a single-head call
    (``[button]``) is the button alone.
    """
    lg_b, a_b = logits[0], actions[0]
    total = _chosen(lg_b, a_b)
    if len(logits) == 1:
        return total
    b = a_b.astype(jnp.int32)
    lg_x, lg_y, lg_t = logits[1], logits[2], logits[3]
    a_x, a_y, a_t = actions[1], actions[2], actions[3]
    return (total
            + USES_SCREEN_HEADS[b] * (_chosen(lg_x, a_x) + _chosen(lg_y, a_y))
            + USES_TARGET_HEAD[b] * _chosen(lg_t, a_t))


def factored_entropy(logits) -> jax.Array:
    """Entropy of the joint action in nats, each auxiliary head weighted by
    the probability the button unlocks it. Sup is :data:`MAX_FACTORED_ENTROPY`.
    A single-head call (``[head]``) is that head's own entropy.
    """
    if len(logits) == 1:
        return _entropy(logits[0])
    lg_b, lg_x, lg_y, lg_t = logits[0], logits[1], logits[2], logits[3]
    p_b = jax.nn.softmax(lg_b, axis=-1)
    p_screen = jnp.sum(p_b * USES_SCREEN_HEADS, axis=-1)
    p_target = jnp.sum(p_b * USES_TARGET_HEAD, axis=-1)
    return (_entropy(lg_b)
            + p_screen * (_entropy(lg_x) + _entropy(lg_y))
            + p_target * _entropy(lg_t))


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
