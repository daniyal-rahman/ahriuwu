"""The whole training loop inside one ``jit`` -- the Anakin shape.

What this deletes
-----------------
The production stack is a hand-rolled **Sebulba**: process actors, a rollout
queue, ``max_staleness``, param versions, latest-wins parameter broadcast, a GIL
workaround. Every operational bug this project has paid for most lives in that
machinery -- undriven champions, silent instance death, a frozen opponent's
experience reaching the learner, a dropped param version killing a run.

Under **Anakin** (Hessel et al. 2021) the environment, action selection and the
parameter update are one XLA program: no host-to-device transfers, no
latency-critical work outside XLA, no Python between steps. There is no queue to
go stale, because acting and learning are the same computation. That is not a
performance note -- it is why the failure modes above cannot occur here.

Reset is a constant write, deliberately
---------------------------------------
The standard auto-reset pattern computes ``where(done, reset, step)`` **every**
step, so the reset path runs on every tick and >99% of its output is discarded.
That is a documented cost sink for long episodes, and ours are long: a 120 s
horizon at 30 Hz is 3,600 steps. It is affordable only while reset is pure
array initialisation from a *constant* pytree, which is what ``init_lane``
produces. Keep it that way: no procedural generation, no host round-trip, no
data-dependent work in reset.

Both champions act under the same parameters
--------------------------------------------
That is the mirror self-play setup, and it makes the zero-sum reward meaningful
-- the opponent improves exactly as fast as the agent. A frozen-opponent league
is a later step and is a different object (a second parameter set carried on the
env axis, not a second policy here).

What is NOT here yet
--------------------
The KL-to-reference term (there is no BC prior to anchor to yet), league/PFSP
opponent sampling, and the recurrent core. `PPOConfig` carries the first two so
they are a wiring change rather than a redesign.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..obs.builder import build_observation
from ..obs.frame import make_lane_frame
from ..sim.init import TOP_LANE_PATH, TOP_OUTER_TURRET, init_lane, lane_params
from ..sim.orders import OrderKind, Orders, apply_orders
from ..sim.state import Team
from ..sim.step import step_decision
from .policy import LanePolicy, PolicyConfig
from .ppo import (
    PPOConfig,
    factored_entropy,
    factored_log_prob,
    gae,
    policy_loss,
    value_loss,
)
from .reward import RewardConfig, lane_reward, reward_init

__all__ = ["TrainConfig", "RunnerState", "make_train"]

BLUE_NEXUS = (1131.8, 1426.3)
SCREEN_RADIUS = 1800.0


class TrainConfig(NamedTuple):
    n_envs: int = 512
    rollout_steps: int = 128
    n_updates: int = 10
    n_minibatches: int = 4
    horizon_s: float = 120.0
    decision_hz: float = 30.0
    ppo: PPOConfig = PPOConfig()
    reward: RewardConfig = RewardConfig()

    @property
    def episode_steps(self) -> int:
        return int(self.horizon_s * self.decision_hz)


class RunnerState(NamedTuple):
    params: dict
    opt_state: optax.OptState
    env_state: object
    reward_state: object
    rng: jax.Array
    step: jax.Array


class Transition(NamedTuple):
    obs_entities: jax.Array
    obs_mask: jax.Array
    obs_self: jax.Array
    obs_global: jax.Array
    action: tuple
    log_prob: jax.Array
    value: jax.Array
    reward: jax.Array
    done: jax.Array


def _sample(logits, key):
    kb, kx, ky, kt = jax.random.split(key, 4)
    a = (jax.random.categorical(kb, logits.button),
         jax.random.categorical(kx, logits.screen_x),
         jax.random.categorical(ky, logits.screen_y),
         jax.random.categorical(kt, logits.target))
    lg = (logits.button, logits.screen_x, logits.screen_y, logits.target)
    return a, factored_log_prob(lg, a)


def _orders_from(action, state, cfg_x=96, cfg_y=54):
    button, sx, sy, tgt = action
    nx = (sx + 0.5) / cfg_x * 2.0 - 1.0
    ny = (sy + 0.5) / cfg_y * 2.0 - 1.0
    # BUTTONS = (noop, move, attack_move, q, w, e, r, recall). Only the three
    # the sim implements are decoded; the rest fall through to noop, which is
    # action-repeat and therefore a real choice rather than a dropped one.
    kind = jnp.where(button == 1, OrderKind.MOVE,
                     jnp.where(button == 2, OrderKind.ATTACK,
                               jnp.where(button == 5, OrderKind.CAST_E,
                                         OrderKind.NOOP)))
    return Orders(kind=kind.astype(jnp.int8),
                  x=state.x[:2] + nx * SCREEN_RADIUS,
                  y=state.y[:2] + ny * SCREEN_RADIUS,
                  target=tgt.astype(jnp.int8))


def make_train(cfg: TrainConfig = TrainConfig()):
    """Build the jittable training function. Returns ``train(rng) -> (state, metrics)``."""
    params_tbl = lane_params()
    path = jnp.asarray(np.array(TOP_LANE_PATH, np.float32))
    frame = make_lane_frame(TOP_OUTER_TURRET[Team.BLUE],
                            TOP_OUTER_TURRET[Team.RED], BLUE_NEXUS)
    policy = LanePolicy(PolicyConfig())
    fresh = init_lane()                    # the constant pytree reset writes
    fresh_reward = reward_init(fresh)
    dt_s = 1.0 / cfg.decision_hz
    n_batch = cfg.rollout_steps * cfg.n_envs * 2      # two champions per env

    tx = optax.chain(
        optax.clip_by_global_norm(cfg.ppo.max_grad_norm),
        optax.adam(cfg.ppo.lr),
    )

    def _obs(state):
        return jax.vmap(lambda i: build_observation(state, i, frame))(jnp.arange(2))

    def _env_step(runner: RunnerState, _):
        rng, sk = jax.random.split(runner.rng)

        def one(state, rstate, key):
            obs = _obs(state)
            logits = policy.apply(runner.params, obs.entities,
                                  obs.entity_pad_mask, obs.self_vec,
                                  obs.global_vec)
            action, log_prob = _sample(logits, key)
            nxt = step_decision(apply_orders(state, _orders_from(action, state)),
                                params_tbl, lane_path=path)
            reward, rstate = lane_reward(nxt, rstate, dt_s, cfg.reward,
                                         runner.step)
            done = nxt.t_ms >= cfg.horizon_s * 1000.0
            # reset is a WHERE against a constant pytree -- see the module docstring
            nxt = jax.tree.map(lambda a, b: jnp.where(done, b, a), nxt, fresh)
            rstate = jax.tree.map(lambda a, b: jnp.where(done, b, a),
                                  rstate, fresh_reward)
            t = Transition(
                obs.entities, obs.entity_pad_mask, obs.self_vec, obs.global_vec,
                action, log_prob, logits.value, reward,
                jnp.broadcast_to(done, reward.shape))
            return nxt, rstate, t

        keys = jax.random.split(sk, cfg.n_envs)
        env_state, reward_state, tr = jax.vmap(one)(
            runner.env_state, runner.reward_state, keys)
        return runner._replace(env_state=env_state, reward_state=reward_state,
                               rng=rng), tr

    def _loss(params, batch, cfg_ppo):
        logits = policy.apply(params, batch["entities"], batch["mask"],
                              batch["self"], batch["global"])
        lg = (logits.button, logits.screen_x, logits.screen_y, logits.target)
        log_prob = factored_log_prob(lg, batch["action"])
        entropy = factored_entropy(lg).mean()
        pl, stats = policy_loss(log_prob, batch["log_prob"], batch["adv"], cfg_ppo)
        vl = value_loss(logits.value, batch["value"], batch["returns"], cfg_ppo)
        total = pl + cfg_ppo.value_coef * vl - cfg_ppo.entropy_coef * entropy
        return total, {"policy_loss": pl, "value_loss": vl, "entropy": entropy,
                       **stats}

    def _update(runner: RunnerState, _):
        runner, tr = jax.lax.scan(_env_step, runner, None,
                                  length=cfg.rollout_steps)

        last_obs = jax.vmap(_obs)(runner.env_state)
        last_v = policy.apply(runner.params, last_obs.entities,
                              last_obs.entity_pad_mask, last_obs.self_vec,
                              last_obs.global_vec).value

        adv, returns = gae(tr.reward, tr.value, tr.done, last_v,
                           cfg.ppo.gamma, cfg.ppo.gae_lambda)

        flat = {
            "entities": tr.obs_entities.reshape(n_batch, *tr.obs_entities.shape[-2:]),
            "mask": tr.obs_mask.reshape(n_batch, -1),
            "self": tr.obs_self.reshape(n_batch, -1),
            "global": tr.obs_global.reshape(n_batch, -1),
            "action": tuple(a.reshape(n_batch) for a in tr.action),
            "log_prob": tr.log_prob.reshape(n_batch),
            "value": tr.value.reshape(n_batch),
            "adv": adv.reshape(n_batch),
            "returns": returns.reshape(n_batch),
        }
        if cfg.ppo.normalize_advantage:
            a = flat["adv"]
            flat["adv"] = (a - a.mean()) / (a.std() + 1e-8)

        def epoch(carry, _):
            params, opt_state, rng = carry
            rng, pk = jax.random.split(rng)
            perm = jax.random.permutation(pk, n_batch)
            mb = jax.tree.map(lambda x: x[perm].reshape(
                cfg.n_minibatches, -1, *x.shape[1:]), flat)

            def minibatch(carry, b):
                params, opt_state = carry
                (loss, info), grads = jax.value_and_grad(_loss, has_aux=True)(
                    params, b, cfg.ppo)
                updates, opt_state = tx.update(grads, opt_state, params)
                return (optax.apply_updates(params, updates), opt_state), info

            (params, opt_state), info = jax.lax.scan(
                minibatch, (params, opt_state), mb)
            return (params, opt_state, rng), info

        (params, opt_state, rng), info = jax.lax.scan(
            epoch, (runner.params, runner.opt_state, runner.rng), None,
            length=cfg.ppo.epochs)

        metrics = jax.tree.map(lambda x: x.mean(), info)
        metrics["reward"] = tr.reward.mean()
        metrics["cs"] = runner.env_state.cs[:, :2].mean()
        runner = runner._replace(params=params, opt_state=opt_state, rng=rng,
                                 step=runner.step + n_batch)
        return runner, metrics

    def train(rng):
        rng, ik = jax.random.split(rng)
        obs0 = _obs(fresh)
        params = policy.init(ik, obs0.entities, obs0.entity_pad_mask,
                             obs0.self_vec, obs0.global_vec)
        env_state = jax.tree.map(
            lambda a: jnp.broadcast_to(a, (cfg.n_envs,) + a.shape), fresh)
        reward_state = jax.tree.map(
            lambda a: jnp.broadcast_to(a, (cfg.n_envs,) + jnp.shape(a)),
            fresh_reward)
        runner = RunnerState(params, tx.init(params), env_state, reward_state,
                             rng, jnp.asarray(0, jnp.int32))
        return jax.lax.scan(_update, runner, None, length=cfg.n_updates)

    return train
