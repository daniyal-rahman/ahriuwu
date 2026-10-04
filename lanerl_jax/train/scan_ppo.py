"""Env-agnostic pieces of the on-device scan PPO trainers.

`vec_train` (legacy world) and `modern_vec_train` (26.19 modern world) both
collect a rollout with a `lax.scan` over vmapped envs into a `Transition`
and update with `ppo_learn`. Kept here so the modern trainer does not import
the legacy simulator (which loads the C# navgrid at import time).
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from lanerl_rl.constants import BUTTONS

from .ppo import factored_log_prob, gae, update_epochs

__all__ = ["VecRunner", "Transition", "make_batch_fn", "ppo_learn"]


class VecRunner(NamedTuple):
    params: dict
    opt_state: optax.OptState
    env_state: object
    #: (n_envs, 2, core_dim) GRU carry, or a (n_envs, 2, 0) placeholder for mlp.
    carry: jax.Array
    rng: jax.Array
    step: jax.Array
    #: (n_envs,) game-ms at which each env's CURRENT episode ends; the first
    #: episode is cut short at random to stagger phases (see `trainer.py`).
    deadline_ms: jax.Array


class Transition(NamedTuple):
    obs_entities: jax.Array
    obs_mask: jax.Array
    obs_self: jax.Array
    obs_global: jax.Array
    action: tuple
    log_prob: jax.Array
    uses_screen: jax.Array
    uses_target: jax.Array
    value: jax.Array
    reward: jax.Array
    done: jax.Array
    reward_terms: dict
    cs: jax.Array
    gold: jax.Array
    xp: jax.Array
    done_full: jax.Array
    deaths: jax.Array
    lane_dist: jax.Array
    click_mask: jax.Array | None
    hp_at_end: jax.Array
    tower_damage_at_end: jax.Array
    kills_at_end: jax.Array


def make_batch_fn(cfg, n_learn: int, recurrent: bool, use_mask: bool, core_dim: int):
    """`_batch(tr, adv, returns, carry0)`: rollout [T, n_envs, 2, ...] -> learner batch
    (learning agents only, agent-major [N, T, ...] for the GRU). Env-agnostic:
    shared by this trainer and `modern_vec_train`."""
    n_rows = cfg.n_envs * n_learn

    def _batch(tr: Transition, adv, returns, carry0):
        # [T, n_envs, 2, ...] -> learning agents only -> agent-major [N, T, ...]
        def rows(x):
            # [T, n_envs, A, ...] -> [n_envs, A, T, ...] -> [n_envs * A, T, ...].
            # The agent axis must sit beside envs BEFORE the fold: a
            # swapaxes(0, 1) alone gave [n_envs, T, A] and the fold then
            # interleaved time and agent, scrambling every GRU sequence
            # (caught by test_vec_train's actor/learner agreement).
            x = jnp.moveaxis(x[:, :, :n_learn], 0, 2)
            x = x.reshape((n_rows, cfg.rollout_steps) + x.shape[3:])
            return x if recurrent else x.reshape((n_rows * cfg.rollout_steps,) + x.shape[2:])
        b = {"entities": rows(tr.obs_entities), "mask": rows(tr.obs_mask),
             "self": rows(tr.obs_self), "global": rows(tr.obs_global),
             "action": tuple(rows(a) for a in tr.action), "log_prob": rows(tr.log_prob),
             "uses_screen": rows(tr.uses_screen), "uses_target": rows(tr.uses_target),
             "value": rows(tr.value), "adv": rows(adv), "returns": rows(returns)}
        if use_mask:
            b["click_mask"] = rows(tr.click_mask)
        if recurrent:
            b["done"] = rows(tr.done)
            b["carry0"] = carry0[:, :n_learn].reshape(n_rows, core_dim)
        return b

    return _batch


def ppo_learn(runner: VecRunner, tr: Transition, carry0, last_value, *, cfg, tx, loss, batch_fn,
              n_learn: int, buttons=BUTTONS):
    """The PPO update half given the bootstrap value: GAE, epochs, diagnostics.
    Env-agnostic (`buttons` names the `button_*` metrics; `obs_self[..., 14]` is
    the is-dead column in both the legacy and the modern self block)."""
    n_rows = cfg.n_envs * n_learn
    adv, returns = gae(tr.reward, tr.value, tr.done, last_value,
                       cfg.ppo.gamma, cfg.ppo.gae_lambda)
    batch = batch_fn(tr, adv, returns, carry0)
    params, opt_state, rng, metrics = update_epochs(
        lambda p, b: loss(p, b, cfg.ppo), tx, runner.params, runner.opt_state,
        batch, runner.rng, epochs=cfg.ppo.epochs, n_minibatches=cfg.n_minibatches,
        max_grad_norm=cfg.ppo.max_grad_norm)
    # post_kl: the updated policy's drift over the rollout it was trained on.
    new_lg = loss.forward(params, batch)
    new_lp = factored_log_prob((new_lg.button, new_lg.screen_x, new_lg.screen_y),
                               batch["action"], batch["uses_screen"],
                               click_mask=batch.get("click_mask"))
    metrics["post_kl"] = jnp.mean(batch["log_prob"] - new_lp)
    r_var = batch["returns"].var()
    metrics["explained_variance"] = jnp.where(
        r_var > 0, 1.0 - (batch["returns"] - batch["value"]).var() / r_var, jnp.nan)
    learn = lambda x: x[:, :, :n_learn]
    metrics["reward"] = learn(tr.reward).mean()
    for k, v in tr.reward_terms.items():
        metrics[f"reward_{k}"] = learn(v).mean()
    n_done = learn(tr.done_full).sum()
    for name, v in (("cs_at_10min", tr.cs), ("gold_at_10min", tr.gold), ("xp_at_10min", tr.xp)):
        metrics[name] = jnp.where(n_done > 0, learn(v).sum() / jnp.maximum(n_done, 1), jnp.nan)
    metrics["cs_episodes"] = n_done.astype(jnp.float32)
    metrics["deaths_per_episode"] = learn(tr.deaths).mean() * cfg.episode_s * cfg.decision_hz
    metrics["lane_dist"] = learn(tr.lane_dist).mean()
    for i, b in enumerate(buttons):
        metrics[f"button_{b}"] = (learn(tr.action[0]) == i).mean()
    alive = learn(tr.obs_self[..., 14]) < 0.5  # observation S_IS_DEAD
    spell = (learn(tr.action[0]) >= 3) & (learn(tr.action[0]) <= 6)
    metrics['alive_spell_decisions'] = (alive & spell).sum().astype(jnp.float32)
    metrics['alive_spell_fraction'] = (alive & spell).sum() / jnp.maximum(alive.sum(), 1)
    runner = runner._replace(params=params, opt_state=opt_state, rng=rng,
                             step=runner.step + n_rows * cfg.rollout_steps)
    return runner, metrics
