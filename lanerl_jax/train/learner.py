"""Shared PPO optimizer/loss for JAX and source-server rollout collectors.

No environment initialization, dynamics, routing, or server I/O lives here.
"""
import jax
import optax
from .policy import VALUE_HEAD_NAME
from .ppo import (factored_log_prob, factored_entropy, expected_screen_usage,
                  policy_loss, value_loss)


def make_learner(policy, ppo):
    def _label(params):
        return jax.tree_util.tree_map_with_path(
            lambda path, _: ("critic" if any(
                getattr(k, "key", None) == VALUE_HEAD_NAME for k in path)
                else "actor"),
            params)

    tx = optax.chain(
        optax.clip_by_global_norm(ppo.max_grad_norm),
        optax.multi_transform(
            # eps=1e-5 (CleanRL, Huang et al. detail #3), not optax's 1e-8:
            # fresh Adam with 1e-8 steps ~lr*sign(g) on every parameter,
            # including those whose gradient is ~0, and the lr-3e-4 arms
            # blew the critic up in chunk 0 (`PPO-04`).
            {"actor": optax.adam(ppo.lr, eps=1e-5),
             "critic": optax.adam(ppo.critic_lr, eps=1e-5)},
            _label),
    )

    def _loss(params, batch, cfg_ppo):
        logits = policy.apply(params, batch["entities"], batch["mask"],
                              batch["self"], batch["global"])
        lg = (logits.button, logits.screen_x, logits.screen_y)
        # The SAME per-sample head masks the rollout computed its log-prob
        # under (`PPO-14`); the entropy's expected usage reads the stored
        # three-head screen usage (`ppo.expected_screen_usage`).
        log_prob = factored_log_prob(lg, batch["action"], batch["uses_screen"],
                                     batch["uses_target"])
        entropy = factored_entropy(
            lg, expected_screen_usage(logits.button)).mean()
        pl, stats = policy_loss(log_prob, batch["log_prob"], batch["adv"], cfg_ppo)
        vl = value_loss(logits.value, batch["value"], batch["returns"], cfg_ppo)
        total = pl + cfg_ppo.value_coef * vl - cfg_ppo.entropy_coef * entropy
        return total, {"policy_loss": pl, "value_loss": vl, "entropy": entropy,
                       **stats}

    return tx, _loss
