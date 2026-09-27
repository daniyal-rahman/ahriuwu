"""PureJaxRL PPO equations with the lane policy's three-head distribution.

Transcribed/adapted from Chris Lu's purejaxrl/ppo_rnn.py at
31756b197773a52db763fdbe6d635e4b46522a73 (Apache-2.0; see
licenses/PureJaxRL-LICENSE). Layout adapters preserve our post-action dones
and agent-major trajectories; no dual clipping or KL early stopping.
"""
from __future__ import annotations
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
import optax
from lanerl_rl.constants import BUTTON_INDEX, BUTTONS, N_SCREEN_X, N_SCREEN_Y


def gamma_for_horizon(horizon_s: float, decision_hz: float) -> float:
    return 1.0 - 1.0 / (horizon_s * decision_hz)


class PPOConfig(NamedTuple):
    """Legacy task settings; standard() selects the reference defaults."""
    horizon_s: float = 120.0
    decision_hz: float = 30.0
    discount: float | None = None
    gae_lambda: float = 0.99
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.001
    kl_prior_coef: float = 0.0
    max_grad_norm: float = 1.0
    lr: float = 1e-5
    # Compatibility for existing commands. A distinct rate is rejected.
    critic_lr: float | None = None
    epochs: int = 4
    n_minibatches: int = 4
    normalize_advantage: bool = True

    @property
    def gamma(self) -> float:
        return (self.discount if self.discount is not None else
                gamma_for_horizon(self.horizon_s, self.decision_hz))

    @classmethod
    def standard(cls, decision_hz: float = 10.0, **overrides) -> "PPOConfig":
        """PureJaxRL defaults; entropy coefficient distributed over 3 heads."""
        base = dict(lr=2.5e-4, discount=0.99, gae_lambda=0.95, clip_eps=0.2,
                    entropy_coef=0.01 / 3, value_coef=0.5, max_grad_norm=0.5,
                    epochs=4, n_minibatches=4, normalize_advantage=True,
                    decision_hz=decision_hz)
        base.update(overrides)
        return cls(**base)


def screen_head_usage(button):
    """Return coordinate usage plus a zero placeholder for collector storage."""
    used = ((button == BUTTON_INDEX["move"]) |
            (button == BUTTON_INDEX["attack_move"]) | (button == BUTTON_INDEX["r"]))
    return used.astype(jnp.float32), jnp.zeros_like(button, dtype=jnp.float32)


MAX_SCREEN_CLICK_ENTROPY = float(np.log(len(BUTTONS)) + np.log(N_SCREEN_X) + np.log(N_SCREEN_Y))


def gae(rewards, values, dones, last_value, gamma: float, lam: float):
    """Reference reverse scan, adapted from pre-observation to post-action done.

    Inputs are [T, ...]. dones[t] ends action t; PureJaxRL transition.done
    instead resets observation t. The final done masks the bootstrap value.
    """
    resets = jnp.concatenate((jnp.zeros_like(dones[:1]), dones[:-1]), axis=0)

    def _get_advantages(carry, transition):
        gae, next_value, next_done = carry
        done, value, reward = transition
        delta = reward + gamma * next_value * (1 - next_done) - value
        gae = delta + gamma * lam * (1 - next_done) * gae
        return (gae, value, done), gae

    _, advantages = jax.lax.scan(
        _get_advantages, (jnp.zeros_like(last_value), last_value, dones[-1]),
        (resets, values, rewards), reverse=True, unroll=16)
    return advantages, advantages + values


def _chosen(lg, a):
    lp = jax.nn.log_softmax(lg, axis=-1)
    return jnp.take_along_axis(lp, a[..., None].astype(jnp.int32), axis=-1)[..., 0]


def _entropy(lg):
    lp = jax.nn.log_softmax(lg, axis=-1)
    return -jnp.sum(jnp.exp(lp) * lp, axis=-1)


def joint_click_logits(lg_x, lg_y, click_mask):
    """Masked joint click distribution over the 96 x 54 screen cells:
    ``softmax(lg_x[i] + lg_y[j] + log(mask[i, j]))``. The SAME two factored
    heads (no new parameters; E15-era checkpoints load unchanged), but the
    probability is renormalised over walkable cells only -- invalid-action
    masking (Huang & Ontanon 2020), INT-001's principled fix. Returns
    ``(..., n_x * n_y)`` logits with ``-inf`` on masked cells."""
    joint = lg_x[..., :, None] + lg_y[..., None, :]
    # A large FINITE penalty, not -inf: with -inf the entropy's p * log p is
    # 0 * -inf = NaN on masked cells and its gradient poisons the update even
    # under jnp.where (E20 canary: "nonfinite learner"). exp(-1e4) underflows
    # to exactly 0 in float32, so the masked mass is zero either way.
    joint = jnp.where(click_mask, joint, -1e4)
    return joint.reshape(joint.shape[:-2] + (-1,))


def joint_click_log_prob(lg_x, lg_y, click_mask, a_x, a_y):
    lp = jax.nn.log_softmax(joint_click_logits(lg_x, lg_y, click_mask), axis=-1)
    idx = a_x * lg_y.shape[-1] + a_y
    return jnp.take_along_axis(lp, idx[..., None], axis=-1)[..., 0]


def joint_click_entropy(lg_x, lg_y, click_mask):
    lp = jax.nn.log_softmax(joint_click_logits(lg_x, lg_y, click_mask), axis=-1)
    return -jnp.sum(jnp.exp(lp) * lp, axis=-1)


def factored_log_prob(logits, actions, uses_screen=None,
                      uses_target=None, click_mask=None):
    """Button likelihood plus click likelihood for coordinate-using buttons.

    uses_target is an unused collector compatibility argument, not a head.
    Accept the collector's tuple of heads or a stacked [..., 3] action array.
    """
    if len(logits) not in (1, 3):
        raise ValueError("expected one categorical head or three screen-click heads")
    if not isinstance(actions, (tuple, list)):
        actions = tuple(actions[..., i] for i in range(len(logits)))
    total = _chosen(logits[0], actions[0])
    if len(logits) == 1:
        return total
    used = screen_head_usage(actions[0])[0] if uses_screen is None else uses_screen
    if click_mask is not None:
        click_lp = joint_click_log_prob(logits[1], logits[2], click_mask, actions[1], actions[2])
    else:
        click_lp = _chosen(logits[1], actions[1]) + _chosen(logits[2], actions[2])
    return total + used * click_lp


def factored_entropy(logits, click_mask=None):
    """Unconditional sum of head entropies (PPO-15)."""
    if len(logits) == 1:
        return _entropy(logits[0])
    if len(logits) != 3:
        raise ValueError("expected one categorical head or three screen-click heads")
    click = (joint_click_entropy(logits[1], logits[2], click_mask)
             if click_mask is not None else _entropy(logits[1]) + _entropy(logits[2]))
    return _entropy(logits[0]) + click


def policy_loss(log_prob, old_log_prob, adv, cfg: PPOConfig):
    """Reference clipped surrogate; normalise within THIS minibatch."""
    ratio = jnp.exp(log_prob - old_log_prob)
    if cfg.normalize_advantage:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    loss_actor1 = ratio * adv
    loss_actor2 = jnp.clip(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
    logr = log_prob - old_log_prob
    return loss_actor.mean(), {
        "approx_kl": ((ratio - 1.0) - logr).mean(),
        "clip_frac": (jnp.abs(ratio - 1.0) > cfg.clip_eps).mean(),
    }


def value_loss(value, old_value, returns, cfg: PPOConfig):
    """Reference value clipping uses the SAME CLIP_EPS as the actor."""
    value_pred_clipped = old_value + (value - old_value).clip(-cfg.clip_eps, cfg.clip_eps)
    value_losses = jnp.square(value - returns)
    value_losses_clipped = jnp.square(value_pred_clipped - returns)
    return 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()


def update_epochs(loss_fn, tx, params, opt_state, batch, rng, *,
                  epochs: int, n_minibatches: int, max_grad_norm: float):
    """Reference nested scans; shuffle whole trajectories on our agent axis.

    Agent-major [N,T,...] and carry0 [N,H] have the same leading axis, so
    the reference's axis-1 take/reshape/swap becomes axis-0 take/reshape.
    Feed-forward batches use that axis for individual samples instead.
    All minibatches are applied and all diagnostic entries are averaged.
    """
    n = batch["value"].shape[0]
    if epochs < 1 or n_minibatches < 1 or n % n_minibatches:
        raise ValueError("positive epochs/minibatches required; minibatches must divide batch rows")

    def _update_epoch(update_state, unused):
        params, opt_state, rng = update_state
        rng, _rng = jax.random.split(rng)
        permutation = jax.random.permutation(_rng, n)
        shuffled_batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=0), batch)
        minibatches = jax.tree.map(
            lambda x: jnp.reshape(x, (n_minibatches, -1) + x.shape[1:]), shuffled_batch)

        def _update_minbatch(train_state, batch_info):
            params, opt_state = train_state
            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (total_loss, info), grads = grad_fn(params, batch_info)
            grad_norm = optax.global_norm(grads)
            updates, opt_state = tx.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), {
                **info, "grad_norm": grad_norm,
                "grad_clipped": (grad_norm > max_grad_norm).astype(jnp.float32),
                "loss_nonfinite": (~jnp.isfinite(total_loss) | ~jnp.isfinite(grad_norm)).astype(jnp.float32),
            }

        (params, opt_state), info = jax.lax.scan(_update_minbatch, (params, opt_state), minibatches)
        return (params, opt_state, rng), info

    (params, opt_state, rng), info = jax.lax.scan(
        _update_epoch, (params, opt_state, rng), None, length=epochs)
    return params, opt_state, rng, jax.tree.map(jnp.mean, info)
