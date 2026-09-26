"""Shared PPO optimizer/loss for JAX and source-server rollout collectors.

No environment initialization, dynamics, routing, or server I/O lives here.
Two batch layouts: flat ``[B, ...]`` for the feed-forward policy, and
agent-major sequences ``[N, T, ...]`` plus ``carry0 [N, core]`` and
``done [N, T]`` for the GRU policy (truncated BPTT over the rollout, carry
reset after every terminal step, exactly as the rollout ran it).
"""
import jax
import jax.numpy as jnp
import optax
from .policy import VALUE_HEAD_NAME
from .ppo import (factored_log_prob, _entropy, policy_loss, value_loss)


def make_learner(policy, ppo, *, anneal_steps: int = 0):
    """``anneal_steps`` > 0: linear lr decay to zero over that many optimizer
    steps (updates x epochs x minibatches), the CleanRL default."""
    def _label(params):
        return jax.tree_util.tree_map_with_path(
            lambda path, _: ("critic" if any(
                getattr(k, "key", None) == VALUE_HEAD_NAME for k in path)
                else "actor"),
            params)

    def sched(lr):
        return optax.linear_schedule(lr, 0.0, anneal_steps) if anneal_steps else lr

    tx = optax.chain(
        optax.clip_by_global_norm(ppo.max_grad_norm),
        optax.multi_transform(
            # eps=1e-5 (CleanRL, Huang et al. detail #3), not optax's 1e-8:
            # fresh Adam with 1e-8 steps ~lr*sign(g) on every parameter,
            # including those whose gradient is ~0, and the lr-3e-4 arms
            # blew the critic up in chunk 0 (`PPO-04`).
            {"actor": optax.adam(sched(ppo.lr), eps=1e-5),
             "critic": optax.adam(sched(ppo.critic_lr), eps=1e-5)},
            _label),
    )
    recurrent = getattr(policy.cfg, "core", "mlp") == "gru"

    def forward(params, batch):
        """Logits for every sample of the batch, in the batch's own layout."""
        if not recurrent:
            return policy.apply(params, batch["entities"], batch["mask"],
                                batch["self"], batch["global"])
        # [N, T, ...] -> scan over T with the carry reset after terminal steps.
        tm = lambda x: jnp.swapaxes(x, 0, 1)
        ent, mask, sv, gv, done = (tm(batch[k]) for k in ("entities", "mask", "self", "global", "done"))
        def step(carry, xs):
            e, m, s, g, d_prev = xs
            carry = jnp.where(d_prev[:, None], 0.0, carry)
            logits, carry = policy.apply(params, e, m, s, g, carry)
            return carry, logits
        d_prev = jnp.concatenate([jnp.zeros_like(done[:1]), done[:-1]], axis=0)
        _, logits = jax.lax.scan(step, batch["carry0"], (ent, mask, sv, gv, d_prev))
        return jax.tree.map(tm, logits)          # back to [N, T, ...]

    def _loss(params, batch, cfg_ppo):
        logits = forward(params, batch)
        lg = (logits.button, logits.screen_x, logits.screen_y)
        # The SAME per-sample head masks the rollout computed its log-prob
        # under (`PPO-14`).
        log_prob = factored_log_prob(lg, batch["action"], batch["uses_screen"],
                                     batch["uses_target"])
        # UNCONDITIONAL per-head entropy, H_b + H_x + H_y (`PPO-15`): the
        # usage-weighted form favoured coordinate buttons.
        entropy = (_entropy(lg[0]) + _entropy(lg[1]) + _entropy(lg[2])).mean()
        pl, stats = policy_loss(log_prob, batch["log_prob"], batch["adv"], cfg_ppo)
        vl = value_loss(logits.value, batch["value"], batch["returns"], cfg_ppo)
        total = pl + cfg_ppo.value_coef * vl - cfg_ppo.entropy_coef * entropy
        return total, {"policy_loss": pl, "value_loss": vl, "entropy": entropy,
                       **stats}

    _loss.forward = forward
    return tx, _loss
