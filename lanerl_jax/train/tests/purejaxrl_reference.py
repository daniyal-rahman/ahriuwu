"""Vendored test oracle: unmodified function bodies from PureJaxRL ppo_rnn.py.
Copyright 2023 Chris Lu. Apache-2.0 (../licenses/PureJaxRL-LICENSE).
Source: https://github.com/luchris429/purejaxrl/blob/31756b197773a52db763fdbe6d635e4b46522a73/purejaxrl/ppo_rnn.py
Only enclosing closures are adapted: config/network supplied to this factory.
This is test support, never production code.
"""
import jax
import jax.numpy as jnp


def reference_functions(config, network):
    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
        # RERUN NETWORK
        _, pi, value = network.apply(
            params, init_hstate[0], (traj_batch.obs, traj_batch.done)
        )
        log_prob = pi.log_prob(traj_batch.action)

        # CALCULATE VALUE LOSS
        value_pred_clipped = traj_batch.value + (
            value - traj_batch.value
        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
        value_losses = jnp.square(value - targets)
        value_losses_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = (
            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
        )

        # CALCULATE ACTOR LOSS
        ratio = jnp.exp(log_prob - traj_batch.log_prob)
        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
        loss_actor1 = ratio * gae
        loss_actor2 = (
            jnp.clip(
                ratio,
                1.0 - config["CLIP_EPS"],
                1.0 + config["CLIP_EPS"],
            )
            * gae
        )
        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
        loss_actor = loss_actor.mean()
        entropy = pi.entropy().mean()

        total_loss = (
            loss_actor
            + config["VF_COEF"] * value_loss
            - config["ENT_COEF"] * entropy
        )
        return total_loss, (value_loss, loss_actor, entropy)

    def _calculate_gae(traj_batch, last_val, last_done):
        def _get_advantages(carry, transition):
            gae, next_value, next_done = carry
            done, value, reward = transition.done, transition.value, transition.reward
            delta = reward + config["GAMMA"] * next_value * (1 - next_done) - value
            gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - next_done) * gae
            return (gae, value, done), gae
        _, advantages = jax.lax.scan(_get_advantages, (jnp.zeros_like(last_val), last_val, last_done), traj_batch, reverse=True, unroll=16)
        return advantages, advantages + traj_batch.value
    return _loss_fn, _calculate_gae
