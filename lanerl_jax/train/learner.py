"""Shared PPO optimizer/loss for JAX and source-server rollout collectors.

No environment initialization, dynamics, routing, or server I/O lives here.
Two batch layouts: flat ``[B, ...]`` for the feed-forward policy, and
agent-major sequences ``[N, T, ...]`` plus ``carry0 [N, core]`` and
``done [N, T]`` for the GRU policy (truncated BPTT over the rollout, carry
reset after every terminal step, exactly as the rollout ran it).

Optimizer and loss adapted from Chris Lu, PureJaxRL ppo_rnn.py, revision
31756b197773a52db763fdbe6d635e4b46522a73 (Apache-2.0; licenses/PureJaxRL-LICENSE).
"""
import jax
import jax.numpy as jnp
import optax
from .policy import VALUE_HEAD_NAME
from .ppo import factored_log_prob, factored_entropy, policy_loss, value_loss


def make_learner(policy, ppo, *, anneal_steps: int = 0, prior_params=None):
    """Reference clipped Adam; anneal_steps counts optimizer minibatch steps.

    PureJaxRL's schedule is constant inside an update and drops at its next
    boundary. The caller supplies the total number of optimizer steps and
    config's n_minibatches describes how many belong to each update.
    """
    if ppo.critic_lr is not None and ppo.critic_lr != ppo.lr:
        raise ValueError("reference PPO uses one Adam learning rate; critic_lr must equal lr")
    if ppo.kl_prior_coef > 0 and prior_params is None:
        raise ValueError("kl_prior_coef requires frozen prior_params")
    steps_per_update = ppo.epochs * ppo.n_minibatches

    def linear_schedule(count):
        frac = 1.0 - (count // steps_per_update) / (anneal_steps / steps_per_update)
        return ppo.lr * frac

    tx = optax.chain(
        optax.clip_by_global_norm(ppo.max_grad_norm),
        optax.adam(linear_schedule if anneal_steps else ppo.lr, eps=1e-5),
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
            carry = jnp.where(d_prev[:, None], policy.initial_carry((carry.shape[0],)), carry)
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
        if cm is not None or getattr(cur, 'click_logits', None) is not None:
            from .ppo import joint_click_logits
            kl = kl + _head_kl(joint_click_logits(pri.screen_x, pri.screen_y, cm, getattr(pri, 'click_logits', None)),
                               joint_click_logits(cur.screen_x, cur.screen_y, cm, getattr(cur, 'click_logits', None)))
        else:
            kl = kl + _head_kl(pri.screen_x, cur.screen_x) + _head_kl(pri.screen_y, cur.screen_y)
        return kl.mean()

    def _loss(params, batch, cfg_ppo):
        logits = forward(params, batch)
        lg = (logits.button, logits.screen_x, logits.screen_y)
        # Preserve the coordinate usage stored by the collector.
        click_mask = batch.get("click_mask")
        joint = getattr(logits, 'click_logits', None)
        log_prob = factored_log_prob(lg, batch["action"], batch.get("uses_screen"), click_mask=click_mask, click_logits=joint)
        entropy = factored_entropy(lg, click_mask=click_mask, click_logits=joint).mean()
        pl, stats = policy_loss(log_prob, batch["log_prob"], batch["adv"], cfg_ppo)
        vl = value_loss(logits.value, batch["value"], batch["returns"], cfg_ppo)
        total = pl + cfg_ppo.value_coef * vl - cfg_ppo.entropy_coef * entropy
        info = {"policy_loss": pl, "value_loss": vl, "entropy": entropy, **stats}
        # Marginal categorical entropies; total may use a masked joint click.
        for name, head in zip(('button', 'screen_x', 'screen_y'), lg):
            lp = jax.nn.log_softmax(head)
            info['entropy_' + name] = -(jnp.exp(lp) * lp).sum(-1).mean()
        if joint is not None:
            # Report actual mixture marginals, not the old ground-head entropy.
            from .ppo import joint_click_logits
            probabilities = jax.nn.softmax(joint_click_logits(logits.screen_x, logits.screen_y, click_mask, joint))
            probabilities = probabilities.reshape(probabilities.shape[:-1] + (logits.screen_x.shape[-1], logits.screen_y.shape[-1]))
            for name, marginal in (('screen_x', probabilities.sum(-1)), ('screen_y', probabilities.sum(-2))):
                info['entropy_' + name] = -(marginal * jnp.log(jnp.maximum(marginal, 1e-30))).sum(-1).mean()
            if getattr(logits, 'proposal_mass', None) is not None:
                info['proposal_mass_mean'] = logits.proposal_mass.mean()
        if prior_params is not None and cfg_ppo.kl_prior_coef > 0:
            klp = kl_to_prior(params, batch)
            total = total + cfg_ppo.kl_prior_coef * klp
            info["kl_prior"] = klp
        return total, info

    _loss.forward = forward
    _loss.kl_to_prior = kl_to_prior

    def trunk_grad_norms(params, batch, cfg_ppo):
        """Diagnostic: the gradient norm each loss term sends into the TRUNK
        (everything except the value head), and the cosine between the policy
        and value terms there. Which term is steering the shared features?"""
        def terms(q):
            logits = forward(q, batch)
            lg = (logits.button, logits.screen_x, logits.screen_y)
            cm = batch.get("click_mask")
            joint = getattr(logits, 'click_logits', None)
            lp = factored_log_prob(lg, batch["action"], batch.get("uses_screen"), click_mask=cm, click_logits=joint)
            pl, _ = policy_loss(lp, batch["log_prob"], batch["adv"], cfg_ppo)
            vl = cfg_ppo.value_coef * value_loss(logits.value, batch["value"], batch["returns"], cfg_ppo)
            ent = -cfg_ppo.entropy_coef * factored_entropy(lg, click_mask=cm, click_logits=joint).mean()
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


def make_update(tx, loss, cfg):
    """Build update(params, opt_state, batch, key), shared with the trainer."""
    from .ppo import update_epochs

    @jax.jit
    def update(params, opt_state, batch, key):
        return update_epochs(lambda q, b: loss(q, b, cfg), tx, params,
                             opt_state, batch, key, epochs=cfg.epochs,
                             n_minibatches=cfg.n_minibatches,
                             max_grad_norm=cfg.max_grad_norm)
    return update
