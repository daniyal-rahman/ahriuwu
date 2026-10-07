"""The learner half of the scan PPO trainer: optimizer and loss (``make_learner``), the rollout record
(``Transition``, ``VecRunner``), rollout -> batch layout (``make_batch_fn``) and one PPO update with
its metrics (``ppo_learn``).

Batches are flat ``[B, ...]`` for the MLP core, or agent-major sequences ``[N, T, ...]`` with
``carry0 [N, core]`` and ``done [N, T]`` for the GRU (truncated BPTT over the rollout, carry reset after
terminal steps, as the rollout ran it). Optimizer and loss after PureJaxRL ppo_rnn.py (Apache-2.0).
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from .ppo import factored_entropy, factored_log_prob, gae, policy_loss, update_epochs, value_loss


def make_learner(policy, ppo, *, anneal_steps: int = 0, prior_params=None):
    """``(tx, loss)``: clipped Adam (linear LR decay over ``anneal_steps`` optimizer steps, constant within
    an update) and ``loss(params, batch, ppo) -> (total, info)`` with ``loss.forward``. A positive
    ``ppo.kl_prior_coef`` adds KL(prior || current) to frozen ``prior_params``."""
    if ppo.kl_prior_coef > 0 and prior_params is None:
        raise ValueError("kl_prior_coef requires frozen prior_params")
    steps_per_update = ppo.epochs * ppo.n_minibatches

    def schedule(count):
        return ppo.lr * (1.0 - (count // steps_per_update) / (anneal_steps / steps_per_update))

    tx = optax.chain(optax.clip_by_global_norm(ppo.max_grad_norm),
                     optax.adam(schedule if anneal_steps else ppo.lr, eps=1e-5))
    recurrent = policy.cfg.core == "gru"

    def forward(params, batch):
        """Logits for every sample, in the batch's own layout."""
        if not recurrent:
            return policy.apply(params, batch["entities"], batch["mask"], batch["self"], batch["global"])
        tm = lambda x: jnp.swapaxes(x, 0, 1)                                    # noqa: E731
        ent, mask, sv, gv, done = (tm(batch[k]) for k in ("entities", "mask", "self", "global", "done"))

        def step(carry, xs):
            e, m, s, g, d_prev = xs
            carry = jnp.where(d_prev[:, None], policy.initial_carry((carry.shape[0],)), carry)
            logits, new_carry = policy.apply(params, e, m, s, g, carry)
            return new_carry.astype(carry.dtype), logits
        d_prev = jnp.concatenate([jnp.zeros_like(done[:1]), done[:-1]], axis=0)
        _, logits = jax.lax.scan(step, batch["carry0"], (ent, mask, sv, gv, d_prev))
        return jax.tree.map(tm, logits)

    def kl(p_lg, q_lg):
        lp, lq = jax.nn.log_softmax(p_lg, -1), jax.nn.log_softmax(q_lg, -1)
        return jnp.sum(jnp.exp(lp) * (lp - lq), axis=-1)

    def kl_to_prior(params, batch):
        cur = forward(params, batch)
        pri = jax.lax.stop_gradient(forward(prior_params, batch))
        return (kl(pri.button, cur.button) + kl(pri.screen_x, cur.screen_x) + kl(pri.screen_y, cur.screen_y)).mean()

    def loss(params, batch, cfg_ppo):
        logits = forward(params, batch)
        lg = (logits.button, logits.screen_x, logits.screen_y)
        log_prob = factored_log_prob(lg, batch["action"], batch["uses_screen"])
        entropy = factored_entropy(lg).mean()
        pl, stats = policy_loss(log_prob, batch["log_prob"], batch["adv"], cfg_ppo)
        vl = value_loss(logits.value, batch["value"], batch["returns"], cfg_ppo)
        total = pl + cfg_ppo.value_coef * vl - cfg_ppo.entropy_coef * entropy
        info = {"policy_loss": pl, "value_loss": vl, "entropy": entropy, **stats}
        if prior_params is not None and cfg_ppo.kl_prior_coef > 0:
            klp = kl_to_prior(params, batch)
            total = total + cfg_ppo.kl_prior_coef * klp
            info["kl_prior"] = klp
        return total, info

    loss.forward = forward
    return tx, loss


class VecRunner(NamedTuple):
    params: dict
    opt_state: optax.OptState
    env_state: object
    carry: jax.Array          # (n_envs, 2, core_dim) GRU carry; (n_envs, 2, 0) for the MLP core
    rng: jax.Array
    step: jax.Array
    deadline_ms: jax.Array    # (n_envs,) game ms at which each env's current episode ends


class Transition(NamedTuple):
    """One decision of every env, (n_envs, 2, ...) per field."""
    obs_entities: jax.Array
    obs_mask: jax.Array
    obs_self: jax.Array
    obs_global: jax.Array
    action: tuple
    log_prob: jax.Array
    uses_screen: jax.Array
    value: jax.Array
    reward: jax.Array
    done: jax.Array
    reward_terms: dict
    cs: jax.Array             # at the end of a full episode (else 0), like gold and xp
    gold: jax.Array
    xp: jax.Array
    done_full: jax.Array      # the episode ran to ``episode_s`` (not a staggered first episode)
    deaths: jax.Array
    lane_dist: jax.Array
    overflow: jax.Array       # dropped packets/missiles/rays in the decision's ticks (must stay 0)


def make_batch_fn(cfg, n_learn: int, recurrent: bool, core_dim: int):
    """``batch(tr, adv, returns, carry0)``: rollout ``[T, n_envs, 2, ...]`` -> learner batch of the
    learning agents (agent-major ``[N, T, ...]``, flattened to ``[N*T, ...]`` for the MLP)."""
    n_rows = cfg.n_envs * n_learn

    def rows(x):
        # The agent axis must sit beside envs before folding, or the fold interleaves time and agents.
        x = jnp.moveaxis(x[:, :, :n_learn], 0, 2)
        x = x.reshape((n_rows, cfg.rollout_steps) + x.shape[3:])
        return x if recurrent else x.reshape((n_rows * cfg.rollout_steps,) + x.shape[2:])

    def batch(tr: Transition, adv, returns, carry0):
        b = {"entities": rows(tr.obs_entities), "mask": rows(tr.obs_mask), "self": rows(tr.obs_self),
             "global": rows(tr.obs_global), "action": tuple(rows(a) for a in tr.action),
             "log_prob": rows(tr.log_prob), "uses_screen": rows(tr.uses_screen), "value": rows(tr.value),
             "adv": rows(adv), "returns": rows(returns)}
        if recurrent:
            b["done"] = rows(tr.done)
            b["carry0"] = carry0[:, :n_learn].reshape(n_rows, core_dim)
        return b

    return batch


def ppo_learn(runner: VecRunner, tr: Transition, carry0, last_value, *, cfg, tx, loss, batch_fn, n_learn: int,
              buttons):
    """GAE, the PPO epochs and the update's metrics, given the bootstrap value."""
    adv, returns = gae(tr.reward, tr.value, tr.done, last_value, cfg.ppo.discount, cfg.ppo.gae_lambda)
    batch = batch_fn(tr, adv, returns, carry0)
    params, opt_state, rng, metrics = update_epochs(
        lambda p, b: loss(p, b, cfg.ppo), tx, runner.params, runner.opt_state, batch, runner.rng,
        epochs=cfg.ppo.epochs, n_minibatches=cfg.n_minibatches, max_grad_norm=cfg.ppo.max_grad_norm)
    new_lg = loss.forward(params, batch)
    new_lp = factored_log_prob((new_lg.button, new_lg.screen_x, new_lg.screen_y), batch["action"],
                               batch["uses_screen"])
    metrics["post_kl"] = jnp.mean(batch["log_prob"] - new_lp)       # drift over the rollout it trained on
    r_var = batch["returns"].var()
    metrics["explained_variance"] = jnp.where(
        r_var > 0, 1.0 - (batch["returns"] - batch["value"]).var() / r_var, jnp.nan)
    learn = lambda x: x[:, :, :n_learn]                                       # noqa: E731
    metrics["reward"] = learn(tr.reward).mean()
    for k, v in tr.reward_terms.items():
        metrics[f"reward_{k}"] = learn(v).mean()
    n_done = learn(tr.done_full).sum()
    for name, v in (("cs_at_10min", tr.cs), ("gold_at_10min", tr.gold), ("xp_at_10min", tr.xp)):
        metrics[name] = jnp.where(n_done > 0, learn(v).sum() / jnp.maximum(n_done, 1), jnp.nan)
    metrics["cs_episodes"] = n_done.astype(jnp.float32)
    metrics["deaths_per_episode"] = learn(tr.deaths).mean() * cfg.episode_s * cfg.decision_hz
    metrics["lane_dist"] = learn(tr.lane_dist).mean()
    metrics["sim_overflow_max"] = tr.overflow.max()
    for i, b in enumerate(buttons):
        metrics[f"button_{b}"] = (learn(tr.action[0]) == i).mean()
    return runner._replace(params=params, opt_state=opt_state, rng=rng,
                           step=runner.step + cfg.n_envs * n_learn * cfg.rollout_steps), metrics
