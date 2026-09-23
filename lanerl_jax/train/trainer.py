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
from ..sim.orders import apply_orders
from ..sim.orders import OrderKind
from ..sim.state import Team
from ..sim.step import step_decision
from .policy import VALUE_HEAD_NAME, LanePolicy, PolicyConfig
from .actions import orders_from
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
RED_NEXUS = (12760.9, 13026.1)


class TrainConfig(NamedTuple):
    """Episode length and discount horizon are DIFFERENT NUMBERS.

    Conflating them is a bug I shipped, and it made the task impossible rather
    than merely hard. ``PPOConfig.horizon_s`` (120 s) is the **discount**
    horizon: it sets ``gamma = 1 - 1/(horizon_s * decision_hz)`` and says how
    far ahead the agent is asked to care. ``episode_s`` is how long an episode
    actually runs before reset.

    With both at 120 s the arithmetic is fatal:

        blue spawn -> wave meeting point   13,532 units = 39 s of walking
        first wave spawns                  90 s
        minions reach the middle           ~120 s
        episode ends                       120 s
        farming possible for               ~0 s

    CS was exactly 0.00000 across 400 updates and 26 million champion-decisions,
    and it could not have been anything else: the last-hit reward term can never
    fire, so the agent was being asked to learn from a signal the environment
    could not produce. That reads identically to a policy that has not learned.

    600 s matches the evaluation protocol the existing stack reports against
    (`cs_at_10min`), and `lanerl_train`'s own `EpisodeSpec` uses step limits of
    6,000-20,000 at 30 Hz, i.e. 200-667 s -- never 120.
    """

    n_envs: int = 512
    rollout_steps: int = 128
    n_updates: int = 10
    n_minibatches: int = 4
    #: how long an episode runs. NOT the discount horizon.
    episode_s: float = 600.0
    decision_hz: float = 30.0
    ppo: PPOConfig = PPOConfig()
    reward: RewardConfig = RewardConfig()

    @property
    def episode_steps(self) -> int:
        return int(self.episode_s * self.decision_hz)


class RunnerState(NamedTuple):
    params: dict
    opt_state: optax.OptState
    env_state: object
    reward_state: object
    rng: jax.Array
    step: jax.Array
    #: ``(n_envs,)`` game-ms at which each env's CURRENT episode ends.
    #:
    #: Why this is not just ``episode_s``: ``done`` used to be
    #: ``t_ms >= episode_s * 1000`` with every env initialised from the same
    #: constant pytree, so all ``n_envs`` episodes ran in LOCKSTEP. Two
    #: consequences, both bad and both invisible in the metrics.
    #:
    #: (1) A rollout of 128 decisions at 30 Hz is a 4.27 s window of game
    #: time, and every env was in the same 4.27 s of its game. So a
    #: 256-env batch held 256 samples of one moment -- the fountain walk, or
    #: the 3rd wave, never both -- and the critic was asked to fit a value
    #: function from one time-slice per update.
    #: (2) CS@10min arrives only on the update where ``done`` fires, which is
    #: every ``episode_s * decision_hz / rollout_steps`` = 141 updates, all
    #: 256 envs at once. A 300-update run yields TWO samples of the headline
    #: metric, which is why RL-001 could not distinguish "not learning" from
    #: "not enough episodes".
    #:
    #: Each env's FIRST episode is therefore cut short at a random point, and
    #: from then on the phases stay spread: CS samples arrive on ~every update
    #: and each batch spans the whole game. The first episode's CS is NOT
    #: comparable (it is a partial game) and is excluded by ``done_full``.
    deadline_ms: jax.Array


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
    #: Diagnostics. Not consumed by the loss -- see `_update`'s metrics block
    #: for why CS alone is not a readable training signal.
    cs: jax.Array
    #: One where this step ended a FULL-LENGTH episode, so its `cs` is a real
    #: cs@10min. The staggered first episode per env is short by construction
    #: and its CS would drag the mean down for the first ~141 updates.
    done_full: jax.Array
    lane_dist: jax.Array
    #: Fractional diagnostic input to the rollout aggregate: one for a
    #: semantic Move whose local route table returned a non-ready status.
    route_nonready: jax.Array


def _sample(logits, key):
    kb, kx, ky, kt = jax.random.split(key, 4)
    a = (jax.random.categorical(kb, logits.button),
         jax.random.categorical(kx, logits.screen_x),
         jax.random.categorical(ky, logits.screen_y),
         jax.random.categorical(kt, logits.target))
    lg = (logits.button, logits.screen_x, logits.screen_y, logits.target)
    return a, factored_log_prob(lg, a)


# Kept as the local name for callers/tests that used the rollout helper before
# decoding was shared with the benchmark path.
_orders_from = orders_from


def make_train(cfg: TrainConfig = TrainConfig(), *, route_table=None,
               terrain=None):
    """Build the jittable training function. Returns ``train(rng) -> (state, metrics)``."""
    if route_table is not None and terrain is None:
        raise ValueError("route_table requires a TerrainGrid")
    params_tbl = lane_params()
    path = jnp.asarray(np.array(TOP_LANE_PATH, np.float32))
    frame = make_lane_frame(TOP_OUTER_TURRET[Team.BLUE],
                            TOP_OUTER_TURRET[Team.RED], BLUE_NEXUS)
    red_frame = make_lane_frame(TOP_OUTER_TURRET[Team.RED],
                                TOP_OUTER_TURRET[Team.BLUE], RED_NEXUS)
    policy = LanePolicy(PolicyConfig())
    fresh = init_lane()                    # the constant pytree reset writes
    fresh_reward = reward_init(fresh, cfg.reward)
    dt_s = 1.0 / cfg.decision_hz
    n_batch = cfg.rollout_steps * cfg.n_envs * 2      # two champions per env

    # `critic_lr` (3e-4) and `lr` (1e-5) are BOTH honoured. They were not:
    # the optimiser was a single `adam(cfg.ppo.lr)`, so `critic_lr` sat in the
    # config -- and in every run manifest, reading as if it were in effect --
    # while the critic actually trained at the actor's 1e-5, thirty times
    # slower than its declared rate. That is the leading candidate for the
    # `value_loss` blow-up observed in the RL-002 run (0.0024 -> 2.0032 ->
    # 542.7 near update 597): as the policy starts earning gold the return
    # scale grows by orders of magnitude, and a linear readout at 1e-5 cannot
    # rescale to follow it.
    #
    # The split is the value READOUT only. The trunk is shared with the actor
    # and stays at `lr` by design -- running shared features at the critic's
    # rate would drag the policy along with them, which is the failure mode
    # `value_coef` exists to balance instead.
    def _label(params):
        return jax.tree_util.tree_map_with_path(
            lambda path, _: ("critic" if any(
                getattr(k, "key", None) == VALUE_HEAD_NAME for k in path)
                else "actor"),
            params)

    tx = optax.chain(
        optax.clip_by_global_norm(cfg.ppo.max_grad_norm),
        optax.multi_transform(
            {"actor": optax.adam(cfg.ppo.lr),
             "critic": optax.adam(cfg.ppo.critic_lr)},
            _label),
    )

    def _obs(state):
        blue = build_observation(state, 0, frame, params=params_tbl)
        red = build_observation(state, 1, red_frame, params=params_tbl)
        return jax.tree.map(lambda a, b: jnp.stack([a, b]), blue, red)

    full_ms = jnp.asarray(cfg.episode_s * 1000.0, jnp.float32)

    def _env_step(runner: RunnerState, _):
        rng, sk = jax.random.split(runner.rng)

        def one(state, rstate, deadline, key):
            obs = _obs(state)
            logits = policy.apply(runner.params, obs.entities,
                                  obs.entity_pad_mask, obs.self_vec,
                                  obs.global_vec)
            action, log_prob = _sample(logits, key)
            orders = _orders_from(action, state, obs.slot_unit, frame)
            ordered = apply_orders(
                state, orders, params_tbl,
                route_table=route_table, terrain=terrain)
            route_nonready = ((orders.kind == OrderKind.MOVE)
                              & (ordered.route_status[:2] != 0))
            # Exact per-neighbour terrain repair lowers through nested dynamic
            # control flow and misses J1's throughput gate by two orders of
            # magnitude. Training uses the documented deferred repair: every
            # individual terrain query is preserved, after the dynamic sweep.
            nxt = step_decision(
                ordered, params_tbl, lane_path=path,
                collision_terrain=False, defer_collision_terrain=True)
            # gamma is the TRAINER's gamma, threaded through deliberately:
            # the shaping potential is policy-invariant only under the same
            # discount the advantage estimator uses.
            reward, rstate = lane_reward(nxt, rstate, dt_s, cfg.reward,
                                         runner.step, gamma=cfg.ppo.gamma)
            # Phi is read BEFORE the reset masks it back to the fountain value.
            phi = rstate.phi
            done = nxt.t_ms >= deadline
            # A full-length episode is one whose deadline was never shortened
            # for phase staggering -- see `RunnerState.deadline_ms`. After
            # reset every deadline is `full_ms`, so this is true for all but
            # each env's first episode.
            done_full = done & (deadline >= full_ms)
            deadline = jnp.where(done, full_ms, deadline)
            # CS is read BEFORE the reset zeroes it. episode_s is 600 s, so
            # this is literally cs@10min -- the headline absolute metric.
            cs_at_done = jnp.where(done_full, nxt.cs[:2].astype(jnp.float32), 0.0)
            # reset is a WHERE against a constant pytree -- see the module docstring
            nxt = jax.tree.map(lambda a, b: jnp.where(done, b, a), nxt, fresh)
            rstate = jax.tree.map(lambda a, b: jnp.where(done, b, a),
                                  rstate, fresh_reward)
            # Distance from the lane corridor, in game units, recovered from
            # the potential. This is the diagnostic that actually moves: CS
            # cannot change until a champion has walked 13,532 units AND the
            # first wave has spawned at 90 s, so it says nothing for the first
            # ~140 updates. Lane distance responds inside a single update.
            per_1000 = cfg.reward.weights.lane_approach
            lane_dist = -phi * (1000.0 / per_1000)
            t = Transition(
                obs.entities, obs.entity_pad_mask, obs.self_vec, obs.global_vec,
                action, log_prob, logits.value, reward,
                jnp.broadcast_to(done, reward.shape),
                cs=cs_at_done,
                done_full=jnp.broadcast_to(done_full, reward.shape),
                lane_dist=lane_dist,
                route_nonready=route_nonready.astype(jnp.float32))
            return nxt, rstate, deadline, t

        keys = jax.random.split(sk, cfg.n_envs)
        env_state, reward_state, deadline_ms, tr = jax.vmap(one)(
            runner.env_state, runner.reward_state, runner.deadline_ms, keys)
        return runner._replace(env_state=env_state, reward_state=reward_state,
                               deadline_ms=deadline_ms, rng=rng), tr

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

        # `target_kl` is ENFORCED, over epochs and minibatches both. It was
        # not: like `critic_lr` it was declared in `PPOConfig`, recorded in
        # every manifest, and read by nothing -- so all four epochs ran
        # unconditionally however far the policy had already moved. Under a
        # trust-region method that is the difference between a step and a leap.
        #
        # Masked rather than branched, because this is inside `scan`: once the
        # k3 KL estimate crosses the target, every later minibatch keeps its
        # params AND its optimiser state, so a stopped epoch is a true no-op
        # and not a zero-gradient Adam step (which would still decay the
        # moments). `kl_stopped` is logged as the fraction of minibatches
        # skipped -- a run sitting near 1.0 is a run whose lr is too high.
        def epoch(carry, _):
            params, opt_state, rng, stopped = carry
            rng, pk = jax.random.split(rng)
            perm = jax.random.permutation(pk, n_batch)
            mb = jax.tree.map(lambda x: x[perm].reshape(
                cfg.n_minibatches, -1, *x.shape[1:]), flat)

            def minibatch(carry, b):
                params, opt_state, stopped = carry
                (loss, info), grads = jax.value_and_grad(_loss, has_aux=True)(
                    params, b, cfg.ppo)
                updates, new_opt_state = tx.update(grads, opt_state, params)
                new_params = optax.apply_updates(params, updates)
                keep = ~stopped
                params = jax.tree.map(
                    lambda new, old: jnp.where(keep, new, old),
                    new_params, params)
                opt_state = jax.tree.map(
                    lambda new, old: jnp.where(keep, new, old),
                    new_opt_state, opt_state)
                stopped = stopped | (info["approx_kl"] > cfg.ppo.target_kl)
                return ((params, opt_state, stopped),
                        {**info, "kl_stopped": stopped.astype(jnp.float32)})

            (params, opt_state, stopped), info = jax.lax.scan(
                minibatch, (params, opt_state, stopped), mb)
            return (params, opt_state, rng, stopped), info

        (params, opt_state, rng, _stopped), info = jax.lax.scan(
            epoch,
            (runner.params, runner.opt_state, runner.rng, jnp.asarray(False)),
            None, length=cfg.ppo.epochs)

        metrics = jax.tree.map(lambda x: x.mean(), info)
        metrics["reward"] = tr.reward.mean()
        # cs@10min, averaged over the episodes that actually ENDED in this
        # rollout. Sampling `env_state.cs` at the end of the rollout instead
        # gives a sawtooth: every env resets on the same step (done is a pure
        # function of t_ms), so that number climbs through an episode and
        # drops to zero together, and its value depends on where the rollout
        # boundary happens to fall rather than on how well the agent plays.
        # Counted over FULL-LENGTH episodes only (`done_full`), not `done`:
        # each env's first episode is deliberately cut short to stagger the
        # phases, and averaging those partial games in would have made the
        # first ~141 updates read as a CS collapse.
        n_done = tr.done_full.sum()
        metrics["cs_at_10min"] = jnp.where(
            n_done > 0, tr.cs.sum() / jnp.maximum(n_done, 1), jnp.nan)
        #: How many full episodes the mean above is over. A metric whose
        #: denominator is invisible is how "BC = 37.3" got quoted off three
        #: games; with staggered phases this should be ~n_envs/141 per update,
        #: and a persistent 0 means the stagger stopped working.
        metrics["cs_episodes"] = n_done.astype(jnp.float32)
        # SCALE diagnostics, because `value_loss` reached 542.7 in the RL-002
        # run and a loss number alone cannot say whether the critic diverged
        # or the targets grew. These say which: `returns_absmax` climbing with
        # `value_absmax` flat is a critic that has fallen behind; both
        # climbing together is a reward-scale problem.
        metrics["returns_absmax"] = jnp.abs(returns).max()
        metrics["value_absmax"] = jnp.abs(tr.value).max()
        metrics["adv_absmax"] = jnp.abs(adv).max()
        # How far from the lane corridor the champions sat, in game units.
        # ~7,981 at spawn, 0 anywhere in lane. This is the leading indicator.
        metrics["lane_dist"] = tr.lane_dist.mean()
        metrics["route_nonready"] = tr.route_nonready.mean()
        runner = runner._replace(params=params, opt_state=opt_state, rng=rng,
                                 step=runner.step + n_batch)
        return runner, metrics

    def initial_runner(rng) -> RunnerState:
        """The state a run starts from. Factored out of `train` so a caller can
        drive the loop in CHUNKS and see it in flight.

        The whole update loop used to live inside one `jax.jit` call, so nothing
        -- no metric, no checkpoint -- was observable until it finished. A
        45-minute run was therefore all-or-nothing, and a node failure mid-run
        lost everything (it did, once). Exposing the carry lets `run_train` scan
        N updates, come back to Python to log and checkpoint, and scan on. The
        jitted function is called repeatedly with identical shapes, so there is
        exactly one compile.
        """
        rng, ik = jax.random.split(rng)
        obs0 = _obs(fresh)
        params = policy.init(ik, obs0.entities, obs0.entity_pad_mask,
                             obs0.self_vec, obs0.global_vec)
        env_state = jax.tree.map(
            lambda a: jnp.broadcast_to(a, (cfg.n_envs,) + a.shape), fresh)
        reward_state = jax.tree.map(
            lambda a: jnp.broadcast_to(a, (cfg.n_envs,) + jnp.shape(a)),
            fresh_reward)
        # The phase stagger, applied ONCE at init: each env's first episode
        # ends at a uniformly random point of a full episode, and every episode
        # after that runs the full `episode_s`. So after one episode the envs
        # are permanently spread over the game clock -- see
        # `RunnerState.deadline_ms` for the two failures that fixes. Floored at
        # two rollouts so no env is resetting inside its own first rollout,
        # which would produce a `done` on step 0 with no preceding transition.
        rng, dk = jax.random.split(rng)
        floor = 2.0 * cfg.rollout_steps / cfg.decision_hz * 1000.0
        deadline_ms = jax.random.uniform(
            dk, (cfg.n_envs,), jnp.float32,
            minval=jnp.minimum(floor, cfg.episode_s * 1000.0),
            maxval=cfg.episode_s * 1000.0)
        return RunnerState(params, tx.init(params), env_state, reward_state,
                           rng, jnp.asarray(0, jnp.int32), deadline_ms)

    def run_chunk(runner: RunnerState, n: int):
        """`n` updates from `runner`. `n` is static -- one compile per value."""
        return jax.lax.scan(_update, runner, None, length=n)

    def train(rng):
        return run_chunk(initial_runner(rng), cfg.n_updates)

    train.initial_runner = initial_runner
    train.run_chunk = run_chunk
    return train
