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
from .ppo import (factored_log_prob, _entropy, policy_loss, value_loss, joint_click_entropy)


def make_learner(policy, ppo, *, anneal_steps: int = 0, prior_params=None):
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
            logits, new_carry = policy.apply(params, e, m, s, g, carry)
            return new_carry.astype(carry.dtype), logits
        d_prev = jnp.concatenate([jnp.zeros_like(done[:1]), done[:-1]], axis=0)
        _, logits = jax.lax.scan(step, batch["carry0"], (ent, mask, sv, gv, d_prev))
        return jax.tree.map(tm, logits)          # back to [N, T, ...]

    def _head_kl(p_lg, q_lg):
        """KL(p || q) per sample for one head of logits."""
        lp, lq = jax.nn.log_softmax(p_lg, -1), jax.nn.log_softmax(q_lg, -1)
        return jnp.sum(jnp.exp(lp) * (lp - lq), axis=-1)

    def kl_to_prior(params, batch):
        """Mean KL(prior || current) over the batch, summed over the heads
        (masked joint click when a click mask is present). The prior's
        forward pass runs under stop_gradient: it is a fixed reference."""
        cur = forward(params, batch)
        pri = jax.lax.stop_gradient(forward(prior_params, batch))
        cm = batch.get("click_mask")
        kl = _head_kl(pri.button, cur.button)
        if cm is not None:
            from .ppo import joint_click_logits
            kl = kl + _head_kl(joint_click_logits(pri.screen_x, pri.screen_y, cm),
                               joint_click_logits(cur.screen_x, cur.screen_y, cm))
        else:
            kl = kl + _head_kl(pri.screen_x, cur.screen_x) + _head_kl(pri.screen_y, cur.screen_y)
        return kl.mean()

    def _loss(params, batch, cfg_ppo):
        logits = forward(params, batch)
        lg = (logits.button, logits.screen_x, logits.screen_y)
        # The SAME per-sample head masks the rollout computed its log-prob
        # under (`PPO-14`).
        click_mask = batch.get("click_mask")
        log_prob = factored_log_prob(lg, batch["action"], batch["uses_screen"],
                                     batch["uses_target"], click_mask=click_mask)
        # UNCONDITIONAL per-head entropy, H_b + H_x + H_y (`PPO-15`): the
        # usage-weighted form favoured coordinate buttons. With a click mask
        # the click entropy is that of the masked joint distribution.
        if click_mask is not None:
            entropy = (_entropy(lg[0]) + joint_click_entropy(lg[1], lg[2], click_mask)).mean()
        else:
            entropy = (_entropy(lg[0]) + _entropy(lg[1]) + _entropy(lg[2])).mean()
        pl, stats = policy_loss(log_prob, batch["log_prob"], batch["adv"], cfg_ppo)
        vl = value_loss(logits.value, batch["value"], batch["returns"], cfg_ppo)
        total = pl + cfg_ppo.value_coef * vl - cfg_ppo.entropy_coef * entropy
        info = {"policy_loss": pl, "value_loss": vl, "entropy": entropy, **stats}
        if prior_params is not None and cfg_ppo.kl_prior_coef > 0:
            klp = kl_to_prior(params, batch)
            total = total + cfg_ppo.kl_prior_coef * klp
            info["kl_prior"] = klp
        return total, info

    _loss.forward = forward

    def trunk_grad_norms(params, batch, cfg_ppo):
        """Diagnostic: the gradient norm each loss term sends into the TRUNK
        (everything except the value head), and the cosine between the policy
        and value terms there. Which term is steering the shared features?"""
        def terms(q):
            logits = forward(q, batch)
            lg = (logits.button, logits.screen_x, logits.screen_y)
            cm = batch.get("click_mask")
            lp = factored_log_prob(lg, batch["action"], batch["uses_screen"], batch["uses_target"], click_mask=cm)
            pl, _ = policy_loss(lp, batch["log_prob"], batch["adv"], cfg_ppo)
            vl = cfg_ppo.value_coef * value_loss(logits.value, batch["value"], batch["returns"], cfg_ppo)
            ent = -cfg_ppo.entropy_coef * ((_entropy(lg[0]) + (joint_click_entropy(lg[1], lg[2], cm) if cm is not None
                                             else _entropy(lg[1]) + _entropy(lg[2]))).mean())
            return pl, vl, ent
        def trunk(g):
            return jax.tree_util.tree_map_with_path(
                lambda path, x: jnp.zeros_like(x) if any(getattr(k, "key", None) == VALUE_HEAD_NAME for k in path) else x, g)
        g_pl = trunk(jax.grad(lambda q: terms(q)[0])(params))
        g_vl = trunk(jax.grad(lambda q: terms(q)[1])(params))
        g_en = trunk(jax.grad(lambda q: terms(q)[2])(params))
        dot = sum(jnp.vdot(a, b) for a, b in zip(jax.tree.leaves(g_pl), jax.tree.leaves(g_vl)))
        n_pl, n_vl, n_en = (optax.global_norm(g) for g in (g_pl, g_vl, g_en))
        return {"g_trunk_pg": n_pl, "g_trunk_value": n_vl, "g_trunk_entropy": n_en,
                "cos_pg_value": dot / (n_pl * n_vl + 1e-12)}
    _loss.trunk_grad_norms = trunk_grad_norms
    return tx, _loss
