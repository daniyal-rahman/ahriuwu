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

What is NOT here
----------------
The KL-to-reference term (there is no BC prior to anchor to yet), league/PFSP
opponent sampling, and a recurrent core or frame stack. None of them has a
config field: each is an experiment with its own contract
(`docs/EXPERIMENT_METHOD.md`), not a dormant knob. (This docstring used to
say `PPOConfig` carried the first two; it never did.)

Removed for the baseline (2026-09-23)
-------------------------------------
Recoverable from commit ``490bb38``:

* the ``zero_sum_alpha`` metric -- alpha is now the constant 1
  (`reward.py`), and a constant logged every update is noise;
* ``TrainConfig.decision_hz`` as a FIELD: it duplicated
  ``PPOConfig.decision_hz``, and the two could disagree silently (dt and the
  episode clock from one, gamma from the other, `PPO-12`). It is now a
  property reading ``ppo.decision_hz``;
* ``make_train``'s hard-coded ``PolicyConfig()``: the policy config is now
  ``TrainConfig.policy``, so the manifest records the one that ran.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from lanerl_rl.constants import BUTTON_INDEX

from ..obs.builder import build_observation
from ..obs.frame import make_lane_frame
from ..sim.config import SimConfig
from ..sim.init import TOP_OUTER_TURRET, init_lane
from ..sim.local_pathing import route_is_server_exact
from ..sim.orders import OrderKind
from ..sim.state import Kind, Team
from ..sim.step import env_advance, env_apply
from .policy import LanePolicy, PolicyConfig
from .learner import make_learner
from .actions import orders_from
from .ppo import (
    PPOConfig,
    expected_head_usage,
    factored_entropy,
    factored_log_prob,
    gae,
    head_usage, screen_head_usage, expected_screen_usage,
    kl_stopped_epochs,
    policy_loss,
    summarise_minibatches,
    value_loss,
)
from .reward import (RewardConfig, lane_corridor_distance, lane_reward,
                     reward_init)

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
    ppo: PPOConfig = PPOConfig()
    reward: RewardConfig = RewardConfig()
    policy: PolicyConfig = PolicyConfig()

    @property
    def decision_hz(self) -> float:
        """ONE decision rate: gamma, dt and the episode clock all read
        ``ppo.decision_hz``. A second field here could disagree (`PPO-12`)."""
        return self.ppo.decision_hz

    @property
    def episode_steps(self) -> int:
        return int(self.episode_s * self.decision_hz)


class RunnerState(NamedTuple):
    params: dict
    opt_state: optax.OptState
    env_state: object
    reward_state: object
    rng: jax.Array
    #: Champion-decisions so far (``n_batch`` per update). Read only for the
    #: resume RNG fold-in and checkpoint names since the alpha anneal was
    #: removed. int32: it wraps after ~32k updates at 256 envs x 128 steps.
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
    #: Per-sample head masks (`ppo.head_usage`, `PPO-14`): 1.0 where this
    #: sample's screen point reached the wire. uses_target is always zero. Computed at
    #: sampling time and consumed by the loss, so the stored `log_prob` and
    #: the loss's recomputation mask the same heads.
    uses_screen: jax.Array
    uses_target: jax.Array
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
    #: Per-weight reward breakdown, summing to `reward` exactly. Only the TOTAL
    #: used to be logged, so when cs@10min moved there was no way to say which
    #: term moved it -- and the last reward change (`RL-002`) was a reweighting.
    reward_terms: dict
    #: Diagnostic: `ATTACK_CLASSES` code per champion-decision (0 = not an
    #: attack_move), for the attack-by-target-type metrics.
    attack_class: jax.Array


def _sample(logits, key, slot_valid):
    """Sample button and screen coordinates, with matching PPO likelihood.

    slot_valid is an unused compatibility argument. uses_target stays zero
    in the transition schema; there is no actor entity-pointer head.
    """
    kb, kx, ky = jax.random.split(key, 3)
    a = (jax.random.categorical(kb, logits.button),
         jax.random.categorical(kx, logits.screen_x),
         jax.random.categorical(ky, logits.screen_y))
    lg = (logits.button, logits.screen_x, logits.screen_y)
    usage = screen_head_usage(a[0])
    return a, factored_log_prob(lg, a, *usage), usage


#: `attack_class` codes (the R3 dashboard metric, `NONFARMING_FAILURES.md`):
#: what each champion-decision's attack_move turned into on the wire.
ATTACK_CLASSES = {
    "attack_enemy_minion": 1,
    "attack_enemy_champion": 2,
    "attack_enemy_turret": 3,
    #: an ATTACK on an allied unit (own turret, allied minion): held with no
    #: swing, chase or hold (`ENT-01`) -- the free "stop" / do-nothing.
    "attack_ally": 4,
    #: attack_move whose sampled slot was empty: the screen-point MOVE (or the
    #: NOOP a minimap click is suppressed to, `actions.orders_from`).
    "attack_move_fallback": 5,
}


def _attack_class(button, orders, state):
    """Per-champion `ATTACK_CLASSES` code, 0 for any other decision.

    Read from the decoded ``orders`` and the PRE-order ``state`` (the one the
    observation was built from), so it classifies the unit the order named.
    """
    is_am = button == BUTTON_INDEX["attack_move"]
    attack = orders.kind == OrderKind.ATTACK
    tgt = jnp.clip(orders.target.astype(jnp.int32), 0, state.kind.shape[0] - 1)
    kind = state.kind[tgt]
    ally = state.team[tgt] == state.team[:2]
    hostile = attack & ~ally
    cls = jnp.where(attack & ally, 4, 0)
    cls = jnp.where(hostile & (kind == Kind.LANE_MINION), 1, cls)
    cls = jnp.where(hostile & (kind == Kind.CHAMPION), 2, cls)
    cls = jnp.where(hostile & (kind == Kind.TURRET), 3, cls)
    cls = jnp.where(is_am & ~attack, 5, cls)
    return cls.astype(jnp.int8)


# Kept as the local name for callers/tests that used the rollout helper before
# decoding was shared with the benchmark path.
_orders_from = orders_from


def make_train(cfg: TrainConfig = TrainConfig(), *, route_table=None,
               terrain=None, sim_config: SimConfig | None = None):
    """Build the jittable training function. Returns ``train(rng) -> (state, metrics)``.

    The sim is stepped under ONE :class:`~lanerl_jax.sim.config.SimConfig`
    (`STRUCT-003`): ``sim_config`` if given, else
    ``SimConfig.training(route_artifact=None, route_table=..., terrain=...)``
    -- the training configuration, routed iff a table is passed. It is exposed
    as ``train.sim_config`` so a manifest and the gates can read the exact
    object rather than a copy of its flags.
    """
    if sim_config is None:
        sim_config = SimConfig.training(route_artifact=None,
                                        route_table=route_table,
                                        terrain=terrain)
    elif route_table is not None or terrain is not None:
        raise ValueError("pass route_table/terrain inside sim_config, not both")
    sim = sim_config
    params_tbl = sim.params
    frame = make_lane_frame(TOP_OUTER_TURRET[Team.BLUE],
                            TOP_OUTER_TURRET[Team.RED], BLUE_NEXUS)
    red_frame = make_lane_frame(TOP_OUTER_TURRET[Team.RED],
                                TOP_OUTER_TURRET[Team.BLUE], RED_NEXUS)
    policy = LanePolicy(cfg.policy)
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
    tx, _loss = make_learner(policy, cfg.ppo)

    def _obs(state):
        # The clock feature is t/episode_s; the builder's default 600 was a
        # second copy of `episode_s` that `--episode-s` did not move (`OBS-10`).
        blue = build_observation(state, 0, frame, params=params_tbl,
                                 horizon_s=cfg.episode_s, vision=sim.vision)
        red = build_observation(state, 1, red_frame, params=params_tbl,
                                 horizon_s=cfg.episode_s, vision=sim.vision)
        return jax.tree.map(lambda a, b: jnp.stack([a, b]), blue, red)

    full_ms = jnp.asarray(cfg.episode_s * 1000.0, jnp.float32)

    def _env_step(runner: RunnerState, _):
        rng, sk = jax.random.split(runner.rng)

        def one(state, rstate, deadline, key):
            obs = _obs(state)
            logits = policy.apply(runner.params, obs.entities,
                                  obs.entity_pad_mask, obs.self_vec,
                                  obs.global_vec)
            action, log_prob, (uses_screen, uses_target) = _sample(
                logits, key, ~obs.entity_pad_mask)
            orders = _orders_from(action, state, obs.slot_unit, frame, vision=sim.vision)
            attack_class = _attack_class(action[0], orders, state)
            # `env_step`'s two halves, split only to read the post-order
            # route status. The step mode (deferred terrain repair, routed
            # Moves, TOP lane waves) is `SimConfig.training`'s -- see its
            # docstring for why training defers the repair.
            ordered = env_apply(state, orders, sim)
            # SERVER_NULL (`PATH-008`) is the server's own two-point walk,
            # not an approximation, so it is not counted as non-ready.
            route_nonready = ((orders.kind == OrderKind.MOVE)
                              & ~route_is_server_exact(ordered.route_status[:2]))
            nxt = env_advance(ordered, sim)
            # gamma is the TRAINER's gamma, threaded through deliberately:
            # the shaping potential is policy-invariant only under the same
            # discount the advantage estimator uses.
            reward, rstate, rterms = lane_reward(
                nxt, rstate, dt_s, cfg.reward,
                gamma=cfg.ppo.gamma, return_terms=True)
            # Distance from the lane corridor, in game units, read BEFORE the
            # reset moves the champions back to the fountain. This is the
            # diagnostic that actually moves: CS cannot change until a
            # champion has walked 13,532 units AND the first wave has spawned
            # at 90 s, so it says nothing for the first ~140 updates. Lane
            # distance responds inside a single update. Read from positions,
            # NOT recovered from the potential by dividing out its weight,
            # which is 0 * inf = NaN at weight 0 (`REW-09`).
            lane_dist = lane_corridor_distance(nxt.x[:2], nxt.y[:2])
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
            t = Transition(
                obs.entities, obs.entity_pad_mask, obs.self_vec, obs.global_vec,
                action, log_prob, uses_screen, uses_target, logits.value,
                reward, jnp.broadcast_to(done, reward.shape),
                cs=cs_at_done,
                done_full=jnp.broadcast_to(done_full, reward.shape),
                lane_dist=lane_dist,
                route_nonready=route_nonready.astype(jnp.float32),
                reward_terms=rterms,
                attack_class=attack_class)
            return nxt, rstate, deadline, t

        keys = jax.random.split(sk, cfg.n_envs)
        env_state, reward_state, deadline_ms, tr = jax.vmap(one)(
            runner.env_state, runner.reward_state, runner.deadline_ms, keys)
        return runner._replace(env_state=env_state, reward_state=reward_state,
                               deadline_ms=deadline_ms, rng=rng), tr

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
            "uses_screen": tr.uses_screen.reshape(n_batch),
            "uses_target": tr.uses_target.reshape(n_batch),
            "value": tr.value.reshape(n_batch),
            "adv": adv.reshape(n_batch),
            "returns": returns.reshape(n_batch),
        }
        if cfg.ppo.normalize_advantage:
            a = flat["adv"]
            flat["adv"] = (a - a.mean()) / (a.std() + 1e-8)

        # `target_kl` is ENFORCED, over epochs and minibatches both
        # (`RL-004`); see `ppo.kl_stopped_epochs` for the masked stop and
        # why a withheld minibatch keeps its optimiser state too.
        params, opt_state, rng, info = kl_stopped_epochs(
            lambda p, b: _loss(p, b, cfg.ppo), tx, runner.params,
            runner.opt_state, flat, runner.rng, epochs=cfg.ppo.epochs,
            n_minibatches=cfg.n_minibatches, target_kl=cfg.ppo.target_kl,
            max_grad_norm=cfg.ppo.max_grad_norm)

        # Loss/gradient metrics are means over the APPLIED minibatches only;
        # `kl_stopped` and `loss_nonfinite` are over all of them (`PPO-11`,
        # `ppo.summarise_minibatches`). `kl_stopped` near 1.0 is a run whose
        # lr is too high.
        metrics = summarise_minibatches(info)
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
        # IT LAGS THE PARAMETERS (`NONFARMING_FAILURES.md` R7): an episode that
        # ended in this rollout was PLAYED over the previous ~141 updates
        # (`episode_steps / rollout_steps` at the production config), so this
        # number describes a mixture of the last ~141 updates' policies, not
        # the parameters of the update it is logged against. A fresh rollout of
        # the final parameters scored higher than the update-280 readout in 7
        # of 9 checkpoints (e.g. c13s0a 1.86 -> 9.88). Read a trend, and score
        # a checkpoint by rolling it out, not by this row.
        n_done = tr.done_full.sum()
        metrics["cs_at_10min"] = jnp.where(
            n_done > 0, tr.cs.sum() / jnp.maximum(n_done, 1), jnp.nan)
        #: The denominator of the mean above, in CHAMPION-episodes: `done_full`
        #: is broadcast to both champions, so one finished env-episode counts
        #: TWICE, matching `cs_at_10min` being CS per champion. With staggered
        #: phases this is ~2 * n_envs / 141 per update at the production
        #: config (the comment used to say n_envs/141, `PPO-11`); a persistent
        #: 0 means the stagger stopped working. A metric whose denominator is
        #: invisible is how "BC = 37.3" got quoted off three games.
        metrics["cs_episodes"] = n_done.astype(jnp.float32)
        # SCALE diagnostics, because `value_loss` reached 542.7 in the RL-002
        # run and a loss number alone cannot say whether the critic diverged
        # or the targets grew. These say which: `returns_absmax` climbing with
        # `value_absmax` flat is a critic that has fallen behind; both
        # climbing together is a reward-scale problem.
        metrics["returns_absmax"] = jnp.abs(returns).max()
        metrics["value_absmax"] = jnp.abs(tr.value).max()
        metrics["adv_absmax"] = jnp.abs(adv).max()
        # EXPLAINED VARIANCE, the scale-free version of the value diagnostic.
        # `value_loss` is unnormalised, so 542.7 could be a broken critic or a
        # large return scale; this says which without needing the absmaxes.
        # 1.0 is a perfect critic, 0.0 is no better than predicting the mean,
        # and NEGATIVE is worse than that -- which is the reading that matters.
        r_var = returns.var()
        metrics["value_explained_var"] = jnp.where(
            r_var > 0, 1.0 - (returns - tr.value).var() / r_var, jnp.nan)
        # Which reward term is actually driving the total.
        # The zero-summed terms (`r_self - r_other`) average to EXACTLY zero
        # over the two champions, so their plain mean -- what this logged
        # until 2026-09-24 -- was 0.0 in every row of every run and said
        # nothing (`BASELINE_AUDIT.md`). What matters is how much of the
        # learning signal each term carries: its mean ABSOLUTE contribution.
        # Shaping is per-agent and not zero-summed, so it keeps its mean.
        for k, v in tr.reward_terms.items():
            metrics[f"reward_{k}"] = (v.mean() if k == "shaping"
                                        else jnp.abs(v).mean())
        # How far from the lane corridor the champions sat, in game units.
        # ~7,981 at spawn, 0 anywhere in lane. This is the leading indicator.
        metrics["lane_dist"] = tr.lane_dist.mean()
        metrics["route_nonready"] = tr.route_nonready.mean()
        # What attack_move actually did, as fractions of ALL champion-
        # decisions (`NONFARMING_FAILURES.md` R3): "attack_move %" is not
        # "attacking" -- an ATTACK on an allied unit (usually the own turret)
        # is a free, always-available stop (`ENT-01`) and took 56-85% of the
        # pointer in the non-farmers. The five sum to p(attack_move).
        for k, code in ATTACK_CLASSES.items():
            metrics[k] = (tr.attack_class == code).mean()
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

    def rollout(runner: RunnerState):
        """One rollout exactly as `_update` collects it: ``(runner, Transition)``
        with leaves ``(rollout_steps, n_envs, ...)``. For the actor/learner
        agreement test, which needs the batch the update will see."""
        return jax.lax.scan(_env_step, runner, None, length=cfg.rollout_steps)

    train.initial_runner = initial_runner
    train.run_chunk = run_chunk
    train.rollout = rollout
    train.sim_config = sim
    return train
