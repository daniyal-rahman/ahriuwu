"""Dual-clip PPO, ported from ``lanerl_rl/ppo.py``.

Current production actions are screen-click-v2: button, screen X, screen Y.
``screen_head_usage`` and the three-head branches below define their likelihood
and entropy. Four-head functions and derivations are retained only as
historical numerical controls for PPO-14; they do not describe the current
policy. See the fidelity ledger's screen-click contract before changing this.

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
screen_y, target), but only the heads THE SAMPLED ACTION PUTS ON THE WIRE
count. `train/actions.orders_from` is the only thing that decides that:
noop/recall/q/w/e read nothing beyond the button, move reads the screen
heads, attack_move reads the target head when the sampled slot holds a unit
(ATTACK) and the screen point when it does not (the fallback MOVE), and r
reads the target head when the slot holds a unit (an empty slot sends
target -1 whichever slot was picked, so the pointer did not reach the wire).
The attack_move and r usage is a property of the SAMPLE -- the button, the
sampled target slot and the observation's slot validity -- so the usage is a
per-sample mask
computed at sampling time (:func:`head_usage`), stored in the rollout's
`Transition`, and passed to :func:`factored_log_prob` by both the rollout and
the loss. The stored and the recomputed log-prob therefore use the SAME mask
by construction. The log-prob is ``lp_b + uses_screen*(lp_x+lp_y) +
uses_target*lp_t``.

History. `PPO-01` (2026-09-23): the port summed all four heads for every
sample; with E cast on 80% of decisions, 80% of the screen/target samples
were pure noise in the ratio -- zero-mean gradient of variance ~A^2, spurious
clipping that also cut the button's gradient, an inflated `approx_kl` feeding
`target_kl`, and an entropy that counted heads the behaviour never used. Its
fix used a per-BUTTON table and marked attack_move as using both screen and
target, "conservatively". `PPO-14` (2026-09-24): over 10 checkpoints and
5.76 M decisions attack_move decoded to a unit ATTACK in 100.0% of cases --
the own turret is always visible, so a valid slot always exists and the
masked pointer always hits one -- so the screen heads took pure-noise policy
gradient on the 22-75% of decisions that were attack_move, and sat at 88-96%
of their maximum entropy in every run.

**The entropy, derived.** Given attack_move, the wire action is a mixture:
with probability ``q = sum_{valid s} softmax(target)_s`` the pointer picks a
unit and the order is ATTACK(t); with ``1 - q`` it picks an empty slot and
the order is MOVE(x, y). The exact entropy of that is
``H2(q) + q*H(t | valid) + (1-q)*(H_x + H_y)``. The policy masks empty slots
with a -1e9 logit, so in float32 ``q`` is exactly 1 whenever ANY slot is
valid and exactly 0 (uniform over empty slots) when none is. At those two
values ``H2(q) = 0`` and ``q*H_t = q*H(t | valid)``, so with the expected
usage (:func:`expected_head_usage`)

    p_screen = p(move) + p(attack_move) * (1 - q)
    p_target = (p(r) + p(attack_move)) * q

``H_b + p_screen*(H_x+H_y) + p_target*H_t`` (:func:`factored_entropy`) IS the
exact wire-action entropy (r is the same argument with an empty slot sending
nothing the pointer chose). The same float argument makes the per-sample
log-prob the exact wire log-likelihood: the ATTACK case's ``lp_t`` already
contains ``log q``, and the MOVE case's omitted ``log(1-q)`` is ``log 1 = 0``
whenever that case has nonzero probability
(`test_log_prob_and_entropy_are_the_exact_wire_action_distribution`
enumerates it). ``q`` rather than a constant keeps the entropy a continuous
function of the logits and right for an observation with no visible unit.

**The ceiling changed with `PPO-14`.** The supremum of the masked entropy is
``ln(sum_b exp(c_b))`` where ``c_b`` is the auxiliary entropy button ``b``
unlocks, attained by ``softmax(c)`` over the buttons with uniform auxiliary
heads -- NOT by a uniform button. ``c_b`` now depends on the observation:
move always unlocks ``ln 96 + ln 54 = 8.553``, and attack_move unlocks the
target head OR the screen heads, never both.

* With a visible slot (``c``: move 8.553, attack_move ``ln 32``, r ``ln 32``,
  0 for the five others) the ceiling is
  :data:`MAX_FACTORED_ENTROPY_TARGET_VISIBLE` = **8.567 nats**. Every
  observation the trainer produces is this case (the own turret is always
  slotted), so read ``entropy`` against it. A uniform button gives 4.015.
* With no visible slot (move 8.553, attack_move 8.553, r 0) it is
  ``ln(6 + 2 * 5184)`` = **9.247 nats**.
* :data:`MAX_FACTORED_ENTROPY` is the larger, the observation-free
  supremum (9.247): a true upper bound no sample can exceed.

The `PPO-01` ceiling was 12.050 (attack_move at 8.553 + 3.466), the unmasked
sum 14.099. Entropy figures from before 2026-09-24 are not comparable with
those after.

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
    "MAX_FACTORED_ENTROPY_TARGET_VISIBLE", "head_usage",
    "expected_head_usage", "kl_stopped_epochs",
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

    @classmethod
    def standard(cls, decision_hz: float = 10.0, **overrides) -> "PPOConfig":
        """Known PPO defaults (CleanRL `ppo`/`ppo_lstm`, Schulman et al. 2017),
        adopted 2026-09-26 so no arm depends on a value we invented: lr 2.5e-4
        (actor and critic, linear anneal to 0 is the launcher's job), GAE
        lambda 0.95, clip 0.2, entropy 0.01, value coef 0.5, max grad norm 0.5,
        4 epochs, per-minibatch advantage normalisation, no KL early stop.
        gamma stays horizon-based (120 s) because the task is 600 s at 10 Hz;
        OpenAI Five used 180-360 s horizons for a 45-minute game."""
        # entropy 0.01 is CleanRL's value for ONE discrete head; our regulariser
        # is the SUM of three head entropies (button + x + y, up to 10.6 nats),
        # so the same push per head is 0.01/3. E05 ran 0.01 on the sum: entropy
        # pinned at 9.0, frozen CS 6.5/10.0 after 2.3M decisions (untrained: 8).
        base = dict(lr=2.5e-4, critic_lr=2.5e-4, gae_lambda=0.95, clip_eps=0.2,
                    entropy_coef=0.01 / 3, value_coef=0.5, max_grad_norm=0.5, epochs=4,
                    normalize_advantage=True, target_kl=float("inf"),
                    decision_hz=decision_hz)
        base.update(overrides)
        return cls(**base)


_MOVE = BUTTON_INDEX["move"]
_ATTACK_MOVE = BUTTON_INDEX["attack_move"]
_R = BUTTON_INDEX["r"]


def screen_head_usage(button):
    """Three-head interface: ground position matters for move, attack and R."""
    used = ((button == _MOVE) | (button == _ATTACK_MOVE) | (button == _R))
    return used.astype(jnp.float32), jnp.zeros_like(button, dtype=jnp.float32)


def expected_screen_usage(button_logits):
    p = jax.nn.softmax(button_logits, axis=-1)
    return p[..., _MOVE] + p[..., _ATTACK_MOVE] + p[..., _R]


def head_usage(button, target_slot, slot_valid):
    """Per-sample ``(uses_screen, uses_target)``, float32, shaped like
    ``button``: which auxiliary heads THIS sample put on the wire.

    Read off `train/actions.orders_from`; if that decoder changes, this must
    (`test_head_usage_matches_what_orders_from_puts_on_the_wire`).
    ``slot_valid`` is ``(..., n_slots)`` bool, True where the observation's
    slot holds a unit: ``~entity_pad_mask``, i.e. ``slot_unit >= 0``.

    * move: screen.
    * attack_move: target if the sampled slot is valid (ATTACK), else screen
      (the fallback MOVE).
    * r: target if the sampled slot is valid; an empty slot sends target -1
      whichever slot was picked.
    * noop/recall/q/w/e: neither.

    A function of the observation and the sampled action only, so it is legal
    inside the log-prob (module docstring, `PPO-14`).
    """
    b = button.astype(jnp.int32)
    n_slots = slot_valid.shape[-1]
    t = jnp.clip(target_slot.astype(jnp.int32), 0, n_slots - 1)
    has_target = jnp.take_along_axis(slot_valid, t[..., None], axis=-1)[..., 0]
    is_am = b == _ATTACK_MOVE
    uses_screen = (b == _MOVE) | (is_am & ~has_target)
    uses_target = (is_am | (b == _R)) & has_target
    return uses_screen.astype(jnp.float32), uses_target.astype(jnp.float32)


def expected_head_usage(button_logits, target_logits, slot_valid):
    """``(p_screen, p_target)``: the probability under the policy that the
    screen heads / the target head reach the wire -- :func:`head_usage`'s
    expectation over the button AND the target slot. With
    ``q = sum_{valid s} softmax(target)_s``::

        p_screen = p(move) + p(attack_move) * (1 - q)
        p_target = (p(r) + p(attack_move)) * q

    The module docstring derives why this makes :func:`factored_entropy` the
    exact wire-action entropy.
    """
    p_b = jax.nn.softmax(button_logits, axis=-1)
    q = jnp.sum(jax.nn.softmax(target_logits, axis=-1)
                * slot_valid.astype(target_logits.dtype), axis=-1)
    p_am = p_b[..., _ATTACK_MOVE]
    return (p_b[..., _MOVE] + p_am * (1.0 - q),
            (p_b[..., _R] + p_am) * q)


def _ceiling(target_visible: bool) -> float:
    """``ln(sum_b exp(c_b))``: the masked entropy's supremum over the logits
    for an observation with (or without) a visible slot (module docstring)."""
    ln_screen = np.log(N_SCREEN_X) + np.log(N_SCREEN_Y)
    c = np.zeros(len(BUTTONS))
    c[_MOVE] = ln_screen
    c[_ATTACK_MOVE] = np.log(N_SLOTS) if target_visible else ln_screen
    c[_R] = np.log(N_SLOTS) if target_visible else 0.0
    return float(np.log(np.exp(c).sum()))


#: The ceiling for any observation with a visible slot -- every training
#: observation, since the own turret is always slotted. 8.567 nats. Read
#: ``entropy`` against this one.
MAX_FACTORED_ENTROPY_TARGET_VISIBLE = _ceiling(True)
#: The observation-free supremum of the masked factored entropy, 9.247 nats,
#: reached only by an observation with NO visible unit (attack_move then
#: unlocks the screen heads). No sample can exceed it. Was 12.050 before
#: `PPO-14` counted attack_move's screen and target as alternatives.
MAX_FACTORED_ENTROPY = max(_ceiling(True), _ceiling(False))

# Current three-head contract: move, attack and R consume coordinates.
MAX_SCREEN_CLICK_ENTROPY = float(np.log(3 * N_SCREEN_X * N_SCREEN_Y + len(BUTTONS) - 3))


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


def factored_log_prob(logits, actions, uses_screen=None,
                      uses_target=None) -> jax.Array:
    """Log-prob of the joint action, counting only the heads THIS sample put
    on the wire: ``lp_b + uses_screen*(lp_x+lp_y) + uses_target*lp_t``.

    ``logits``/``actions`` are ``(button, screen_x, screen_y, target)``
    sequences; ``uses_screen``/``uses_target`` are the per-sample masks from
    :func:`head_usage` and are REQUIRED for the four-head form. The rollout
    computes them at sampling time and stores them, and the loss passes the
    stored ones, so both log-probs use the same mask (`PPO-14`). A
    single-head call (``[button]``) is the button alone and takes no masks.
    """
    lg_b, a_b = logits[0], actions[0]
    total = _chosen(lg_b, a_b)
    if len(logits) == 1:
        return total
    if len(logits) == 3:
        used, _ = screen_head_usage(actions[0])
        return total + used * (_chosen(logits[1], actions[1]) + _chosen(logits[2], actions[2]))
    if uses_screen is None or uses_target is None:
        raise TypeError("the four-head factored_log_prob needs the per-sample "
                        "uses_screen/uses_target masks (ppo.head_usage)")
    lg_x, lg_y, lg_t = logits[1], logits[2], logits[3]
    a_x, a_y, a_t = actions[1], actions[2], actions[3]
    return (total
            + uses_screen * (_chosen(lg_x, a_x) + _chosen(lg_y, a_y))
            + uses_target * _chosen(lg_t, a_t))


def factored_entropy(logits, p_screen=None, p_target=None) -> jax.Array:
    """Entropy of the sampled factored action representation in nats,
    ``H_b + p_screen*(H_x+H_y) + p_target*H_t``, where ``p_screen``/
    ``p_target`` are the per-sample expected usage from
    :func:`expected_head_usage` (REQUIRED for the four-head form; they carry
    the observation's slot validity). Sup is :data:`MAX_FACTORED_ENTROPY`,
    and :data:`MAX_FACTORED_ENTROPY_TARGET_VISIBLE` with a visible slot. A
    single-head call (``[head]``) is that head's own entropy.

    The environment can collapse distinct samples to the same wire command
    (for example every unranked R cursor becomes NOOP). This is not entropy
    over those resolved commands. Coordinate usage is differentiable, so
    this regularizer also favors buttons with coordinate heads.
    """
    if len(logits) == 1:
        return _entropy(logits[0])
    if len(logits) == 3:
        return (_entropy(logits[0]) + expected_screen_usage(logits[0])
                * (_entropy(logits[1]) + _entropy(logits[2])))
    if p_screen is None or p_target is None:
        raise TypeError("the four-head factored_entropy needs the per-sample "
                        "expected usage (ppo.expected_head_usage)")
    lg_b, lg_x, lg_y, lg_t = logits[0], logits[1], logits[2], logits[3]
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
