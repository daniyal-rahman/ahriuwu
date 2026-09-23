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

**Early stopping is on the ABSOLUTE per-minibatch KL**, measured before the
minibatch's step (:func:`kl_stopped_epochs`). The torch stack stopped on the
EXCESS over epoch 0's mean because its off-policy rollouts already carried
staleness drift; under Anakin the rollout and the update use the same
parameters, so the first minibatch's KL is zero up to roundoff and the
absolute value is the right quantity (`PPO-05`).

Removed for the baseline (2026-09-23)
-------------------------------------
Nothing in `PPOConfig` was unread (`PPO-12`). `decision_hz` stays here
because ``gamma`` is defined by it; `TrainConfig.decision_hz` is now a
read-only view of this field instead of a second copy that could disagree.
Return normalisation is deliberately NOT here: it is the first candidate
experiment (`PPO-02`), not baseline. The zero-sum alpha anneal, the unported
reward terms and `PolicyConfig.frame_stack` were removed from `reward.py`,
`policy.py` and `trainer.py`; all are recoverable from commit ``490bb38``.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from lanerl_rl.constants import BUTTON_INDEX, BUTTONS, N_SCREEN_X, N_SCREEN_Y, N_SLOTS

__all__ = [
    "PPOConfig", "gamma_for_horizon", "gae", "factored_log_prob",
    "factored_entropy", "policy_loss", "value_loss", "MAX_FACTORED_ENTROPY",
    "USES_SCREEN_HEADS", "USES_TARGET_HEAD", "kl_stopped_epochs",
    "summarise_minibatches",
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
    #: 1e-5, not 3e-4: sweep B (2026-09-23, corrected sim, one seed, 300
    #: updates) put the 3e-4 critic at 0.03 CS against 23.3 with the critic
    #: at the actor's 1e-5 -- and at every actor lr the fast critic lost
    #: (a2 9.9, a3 5.3, a7 no-clip 0.7). Seeds are running; until they land
    #: this is the best single measurement, not a settled number.
    critic_lr: float = 1e-5
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


def kl_stopped_epochs(loss_fn, tx, params, opt_state, batch, rng, *,
                      epochs: int, n_minibatches: int, target_kl: float,
                      max_grad_norm: float):
    """``epochs`` passes of shuffled minibatch steps with the KL early stop.

    ``loss_fn(params, minibatch) -> (loss, info)``; ``info`` must carry
    ``approx_kl``. Returns ``(params, opt_state, rng, info)`` with every
    ``info`` leaf shaped ``(epochs, n_minibatches)``.

    The stop is MASKED rather than branched, because this runs inside
    ``scan``. The KL is measured on the params a minibatch starts from, and
    the first minibatch whose KL is not ``<= target_kl`` is WITHHELD, as is
    every later one, across epochs as well as minibatches. A withheld
    minibatch keeps both params AND optimiser state, so it is a true no-op
    rather than a zero-gradient Adam step that would still decay the moments
    (`RL-004`). Latching before the step is `PPO-05` (SB3 checks before
    stepping); ``~(kl <= target)`` rather than ``kl > target`` so a NaN KL
    stops too.

    Each minibatch reports ``applied`` (1 if its step was kept) and
    ``loss_nonfinite`` (1 if its loss or KL was not finite) alongside the
    loss's own ``info`` and ``grad_norm``/``grad_clipped``. Aggregate with
    :func:`summarise_minibatches`, which excludes the withheld ones.
    """
    n = jax.tree.leaves(batch)[0].shape[0]

    def epoch(carry, _):
        params, opt_state, rng, stopped = carry
        rng, pk = jax.random.split(rng)
        perm = jax.random.permutation(pk, n)
        mb = jax.tree.map(lambda x: x[perm].reshape(
            n_minibatches, -1, *x.shape[1:]), batch)

        def minibatch(carry, b):
            params, opt_state, stopped = carry
            (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                params, b)
            # Under ADAM an always-active clip still steps ~lr (Adam is
            # invariant to gradient scale), so `grad_clipped` near 1 does not
            # mean an lr sweep measured nothing (`PPO-10`). What clipping
            # changes is the relative weight of the updates where it is
            # intermittent, which is why the fraction is logged.
            gnorm = optax.global_norm(grads)
            info = {**info, "grad_norm": gnorm,
                    "grad_clipped": (gnorm > max_grad_norm).astype(jnp.float32)}
            updates, new_opt_state = tx.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            kl = info["approx_kl"]
            stopped = stopped | ~(kl <= target_kl)
            keep = ~stopped
            params = jax.tree.map(lambda new, old: jnp.where(keep, new, old),
                                  new_params, params)
            opt_state = jax.tree.map(
                lambda new, old: jnp.where(keep, new, old),
                new_opt_state, opt_state)
            return (params, opt_state, stopped), {
                **info,
                "applied": keep.astype(jnp.float32),
                "loss_nonfinite": (~jnp.isfinite(loss) | ~jnp.isfinite(kl)
                                   ).astype(jnp.float32)}

        (params, opt_state, stopped), info = jax.lax.scan(
            minibatch, (params, opt_state, stopped), mb)
        return (params, opt_state, rng, stopped), info

    (params, opt_state, rng, _), info = jax.lax.scan(
        epoch, (params, opt_state, rng, jnp.asarray(False)), None,
        length=epochs)
    return params, opt_state, rng, info


def summarise_minibatches(info) -> dict:
    """Per-update scalars from :func:`kl_stopped_epochs`'s per-minibatch info.

    Every loss/gradient statistic is the mean over the minibatches whose step
    was APPLIED. The plain mean used to include the KL-stopped ones, so
    ``grad_norm``/``grad_clipped`` averaged in gradients that were never
    applied (`PPO-11`). NaN if no minibatch was applied, which requires the
    very first minibatch -- measured on the rollout's own params -- to be
    over ``target_kl``: an actor/learner disagreement or a NaN, both of which
    the divergence guard should see.

    Two scalars are over ALL minibatches, and are named for it:
    ``kl_stopped`` (the fraction withheld) and ``loss_nonfinite`` (the
    fraction whose loss or KL was not finite, withheld or not). The
    divergence guard reads the latter, because a NaN-KL minibatch is by
    construction a withheld one and the applied-only means cannot show it.
    """
    applied = info["applied"]
    n = applied.sum()
    out = {}
    for k, v in info.items():
        if k in ("applied", "loss_nonfinite"):
            continue
        tot = jnp.where(applied > 0, v, 0.0).sum()
        out[k] = jnp.where(n > 0, tot / jnp.maximum(n, 1.0), jnp.nan)
    out["kl_stopped"] = 1.0 - applied.mean()
    out["loss_nonfinite"] = info["loss_nonfinite"].mean()
    return out
