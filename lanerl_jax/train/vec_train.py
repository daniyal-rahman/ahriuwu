"""Vectorised on-device PPO for the lane sim: the JAX-native trainer.

`jax_train.py` drives the sim from a Python loop (`JaxFarmCollector`): one
host round-trip per decision, ~230 champion-decisions/s at 16 envs and
GIL-bound. `trainer.make_train` showed the alternative -- the whole rollout is
a `jax.lax.scan` over vmapped envs, so the GPU never waits for Python -- but
it was the fountain-start / MLP / `lane_reward` research trainer. This module
is that scan trainer with the E31/E32 task on it:

* GRU policy with the carry threaded through the scan and reset on `done`
  (`learner.make_learner`'s agent-major [N, T] layout, truncated BPTT over
  the rollout, exactly as the collector path trains).
* NEAR-WAVE START. The collector's setup (walk to the wave-start point, hold
  until `start_ms`, spend the rank NOOP) is data-dependent host work, so it
  cannot run inside the scan. Instead a BANK of `bank_size` post-setup states
  is prepared ONCE by `JaxFarmCollector` itself (same seeds, same legs, same
  `START_JITTER` protocol, `setup.jsonl` written under the run) and a reset is
  a gather from that constant bank -- the pure-array reset the scan needs
  (see `trainer.py`'s module docstring for why reset must be cheap).
* RELATIVE reward (`server_train.relative_reward`, REW-12): zero-summed
  gold and XP deltas plus the lane-keep potential, no death term.
* Dropped or masked unwalkable clicks (INT-001), `--init-from`, a scripted
  opponent, RunDir checkpoints that `jax_eval` / `ops/jax_periodic_eval.sh`
  load unchanged.

Decisions/s: 254 at 16 envs, 983 at 64 (beside two other jobs), scaling
linearly in env count -- the step is latency-bound, so envs are ~free until
the GPU saturates (`lanerl_jax/probes/vec_bench.py`).
"""
from __future__ import annotations

import argparse
import json
import shlex
import shutil
import sys
import time
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from lanerl_rl.constants import BUTTONS

from ..obs.builder import build_observation
from ..sim.config import DEFAULT_ROUTE_ARTIFACT, SimConfig
from ..sim.step import env_step
from .actions import click_mask_from_position, orders_from
from .learner import make_learner
from .policy import LanePolicy, PolicyConfig
from .ppo import PPOConfig, factored_log_prob, gae, update_epochs
from .reward import lane_corridor_distance

__all__ = ["VecConfig", "VecRunner", "make_vec_train", "main"]


class VecConfig(NamedTuple):
    n_envs: int = 256
    rollout_steps: int = 128
    n_updates: int = 1000
    n_minibatches: int = 4
    episode_s: float = 600.0
    step_ticks: int = 6
    #: distinct prepared start states; each reset draws one uniformly.
    bank_size: int = 32
    start_jitter_s: float = 0.0
    #: "mirror" (shared params, both champions learn), "lasthit" / "brawler"
    #: (scripted red, only blue learns).
    opponent: str = "mirror"
    unwalkable_click: str = "noop"
    gold_scale: float = 20.0
    xp_scale: float = 0.008
    enemy_scale: float = 1.0
    lr_anneal: bool = False
    ppo: PPOConfig = PPOConfig()
    policy: PolicyConfig = PolicyConfig()

    @property
    def decision_hz(self) -> float:
        return 60.0 / self.step_ticks

    @property
    def learn_agents(self) -> int:
        return 2 if self.opponent == "mirror" else 1


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


def _relative_reward(prev, nxt, cfg: VecConfig):
    """`server_train.relative_reward` on device: (2,) per champion, terms
    under the farm names every consumer reads (cs -> gold term)."""
    d_gold = nxt.gold[:2] - prev.gold[:2]
    d_xp = nxt.xp[:2] - prev.xp[:2]
    pot_p = -lane_corridor_distance(prev.x[:2], prev.y[:2]) / 10000.0
    pot_n = -lane_corridor_distance(nxt.x[:2], nxt.y[:2]) / 10000.0
    es = cfg.enemy_scale
    gold = (d_gold - es * d_gold[::-1]) / cfg.gold_scale
    xp = cfg.xp_scale * (d_xp - es * d_xp[::-1])
    shaping = 5.0 * (pot_n - pot_p)
    return gold + xp + shaping, {"cs": gold, "death": jnp.zeros_like(gold),
                                 "approach": shaping, "xp": xp}


def prepare_bank(cfg: VecConfig, sim: SimConfig, out: Path, seed: int):
    """`bank_size` post-setup states from the collector's own near-wave setup.
    Returns the stacked LaneState pytree (leading axis bank_size)."""
    from .jax_farm import JaxFarmCollector
    from .server_train import START_JITTER
    START_JITTER.update(max_s=cfg.start_jitter_s, seed=seed)
    col = JaxFarmCollector(cfg.bank_size, out, cfg.episode_s, True, cfg.step_ticks,
                           seed=seed, sim_config=sim, batch_mode="auto", teams=(0, 1),
                           drop_unwalkable_moves=(cfg.unwalkable_click == "noop"))
    states = col.states
    col.close()
    return states


def make_vec_train(cfg: VecConfig, sim: SimConfig, bank, *, prior_params=None):
    """Build the jittable pieces. `bank` is the stacked reset pytree."""
    from ..parity.policy_driver import _lane_frames
    from .trainer import _sample
    if cfg.opponent not in ("mirror", "lasthit", "brawler"):
        raise ValueError(f"unknown opponent {cfg.opponent!r}")
    if cfg.unwalkable_click not in ("noop", "resolve"):
        raise ValueError("unwalkable_click must be noop or resolve")
    frames = _lane_frames()
    policy = LanePolicy(cfg.policy)
    recurrent = cfg.policy.core == "gru"
    use_mask = bool(cfg.policy.click_mask)
    scripted = None
    if cfg.opponent != "mirror":
        from .scripted_policy import PLAYERS
        scripted = PLAYERS[cfg.opponent]
    n_learn = cfg.learn_agents
    n_rows = cfg.n_envs * n_learn
    if n_rows % cfg.n_minibatches:
        raise ValueError("minibatches must divide envs x learning agents")
    full_ms = jnp.asarray(cfg.episode_s * 1000.0, jnp.float32)
    K = jax.tree.leaves(bank)[0].shape[0]
    core_dim = cfg.policy.core_dim if recurrent else 0
    tx, loss = make_learner(
        policy, cfg.ppo._replace(n_minibatches=cfg.n_minibatches),
        anneal_steps=(cfg.n_updates * cfg.ppo.epochs * cfg.n_minibatches) if cfg.lr_anneal else 0,
        prior_params=prior_params)

    def _obs(state):
        per = [build_observation(state, t, frames[t], params=sim.params,
                                 horizon_s=cfg.episode_s, vision=sim.vision) for t in (0, 1)]
        return jax.tree.map(lambda a, b: jnp.stack([a, b]), *per)

    def _apply(params, obs, carry):
        if recurrent:
            return policy.apply(params, obs.entities, obs.entity_pad_mask,
                                obs.self_vec, obs.global_vec, carry)
        return policy.apply(params, obs.entities, obs.entity_pad_mask,
                            obs.self_vec, obs.global_vec), carry

    def _click_mask(state):
        if not use_mask:
            return None
        return jnp.stack([click_mask_from_position(state.x[t], state.y[t], frames[t].axis,
                                                   frames[t].normal) for t in (0, 1)])

    def _env_step(runner: VecRunner, _):
        rng, sk = jax.random.split(runner.rng)

        def one(state, carry, deadline, key):
            k_act, k_reset = jax.random.split(key)
            obs = _obs(state)
            logits, new_carry = _apply(runner.params, obs, carry)
            cm = _click_mask(state)
            action, log_prob, (uses_screen, uses_target) = _sample(
                logits, k_act, ~obs.entity_pad_mask, cm)
            if scripted is not None:
                red = scripted(jax.tree.map(lambda a: a[1], obs), k_act)
                action = tuple(a.at[1].set(jnp.asarray(r, a.dtype)) for a, r in zip(action, red))
            orders = orders_from(action, state, None, frames[0], snap_moves=False,
                                 params=sim.params, vision=sim.vision,
                                 drop_unwalkable_moves=(cfg.unwalkable_click == "noop"))
            nxt = env_step(state, orders, sim)
            reward, terms = _relative_reward(state, nxt, cfg)
            done = nxt.t_ms >= deadline
            done_full = done & (deadline >= full_ms)
            deadline = jnp.where(done, full_ms, deadline)
            cs_at_done = jnp.where(done_full, nxt.cs[:2].astype(jnp.float32), 0.0)
            gold_at_done = jnp.where(done_full, nxt.gold[:2].astype(jnp.float32), 0.0)
            xp_at_done = jnp.where(done_full, nxt.xp[:2].astype(jnp.float32), 0.0)
            deaths = (nxt.deaths[:2] - state.deaths[:2]).astype(jnp.float32)
            lane_dist = lane_corridor_distance(nxt.x[:2], nxt.y[:2])
            idx = jax.random.randint(k_reset, (), 0, K)
            fresh = jax.tree.map(lambda b: b[idx], bank)
            nxt = jax.tree.map(lambda a, b: jnp.where(done, b, a), nxt, fresh)
            new_carry = jnp.where(done, jnp.zeros_like(new_carry), new_carry)
            t = Transition(obs.entities, obs.entity_pad_mask, obs.self_vec, obs.global_vec,
                           action, log_prob, uses_screen, uses_target, logits.value,
                           reward, jnp.broadcast_to(done, reward.shape), terms,
                           cs_at_done, gold_at_done, xp_at_done,
                           jnp.broadcast_to(done_full, reward.shape), deaths, lane_dist, cm)
            return nxt, new_carry, deadline, t

        keys = jax.random.split(sk, cfg.n_envs)
        env_state, carry, deadline_ms, tr = jax.vmap(one)(
            runner.env_state, runner.carry, runner.deadline_ms, keys)
        return runner._replace(env_state=env_state, carry=carry,
                               deadline_ms=deadline_ms, rng=rng), tr

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

    def collect(runner: VecRunner):
        carry0 = runner.carry
        runner, tr = jax.lax.scan(_env_step, runner, None, length=cfg.rollout_steps)
        return runner, tr, carry0

    def learn(runner: VecRunner, tr: Transition, carry0):
        """The unchanged update half, exposed for separate timing/memory checks."""
        last_obs = jax.vmap(_obs)(runner.env_state)
        last_logits, _ = jax.vmap(lambda o, c: _apply(runner.params, o, c))(last_obs, runner.carry)
        adv, returns = gae(tr.reward, tr.value, tr.done, last_logits.value,
                           cfg.ppo.gamma, cfg.ppo.gae_lambda)
        batch = _batch(tr, adv, returns, carry0)
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
        for i, b in enumerate(BUTTONS):
            metrics[f"button_{b}"] = (learn(tr.action[0]) == i).mean()
        alive = learn(tr.obs_self[..., 14]) < 0.5  # observation S_IS_DEAD
        spell = (learn(tr.action[0]) >= 3) & (learn(tr.action[0]) <= 6)
        metrics['alive_spell_decisions'] = (alive & spell).sum().astype(jnp.float32)
        metrics['alive_spell_fraction'] = (alive & spell).sum() / jnp.maximum(alive.sum(), 1)
        runner = runner._replace(params=params, opt_state=opt_state, rng=rng,
                                 step=runner.step + n_rows * cfg.rollout_steps)
        return runner, metrics

    def _update(runner: VecRunner, _):
        return learn(*collect(runner))

    def init_params(rng):
        obs0 = _obs(jax.tree.map(lambda b: b[0], bank))
        carry = policy.initial_carry((2,)) if recurrent else None
        return policy.init(rng, obs0.entities, obs0.entity_pad_mask, obs0.self_vec,
                           obs0.global_vec, carry) if recurrent else \
            policy.init(rng, obs0.entities, obs0.entity_pad_mask, obs0.self_vec, obs0.global_vec)

    def initial_runner(rng, params=None) -> VecRunner:
        rng, ik, bk, dk = jax.random.split(rng, 4)
        if params is None:
            params = init_params(ik)
        idx = jax.random.randint(bk, (cfg.n_envs,), 0, K)
        env_state = jax.tree.map(lambda b: b[idx], bank)
        carry = jnp.zeros((cfg.n_envs, 2, core_dim), jnp.float32)
        start_ms = jnp.max(env_state.t_ms)
        floor = start_ms + 2.0 * cfg.rollout_steps / cfg.decision_hz * 1000.0
        deadline_ms = jax.random.uniform(dk, (cfg.n_envs,), jnp.float32,
                                         minval=jnp.minimum(floor, full_ms), maxval=full_ms)
        return VecRunner(params, tx.init(params), env_state, carry, rng,
                         jnp.asarray(0, jnp.int32), deadline_ms)

    def run_chunk(runner: VecRunner, n: int):
        return jax.lax.scan(_update, runner, None, length=n)

    def rollout(runner: VecRunner):
        """One rollout as `_update` collects it, plus the learner batch built
        from it (for the actor/learner agreement test)."""
        carry0 = runner.carry
        runner, tr = jax.lax.scan(_env_step, runner, None, length=cfg.rollout_steps)
        zeros = jnp.zeros_like(tr.reward)
        return runner, tr, _batch(tr, zeros, zeros, carry0)

    return dict(initial_runner=initial_runner, run_chunk=run_chunk, rollout=rollout,
                init_params=init_params, policy=policy, loss=loss,
                collect=collect, learn=learn)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--envs", type=int, default=256)
    p.add_argument("--rollout", type=int, default=128)
    p.add_argument("--updates", type=int, default=1000)
    p.add_argument("--minibatches", type=int, default=4)
    p.add_argument("--episode-s", type=float, default=600.0)
    p.add_argument("--step-ticks", type=int, default=6)
    p.add_argument("--bank-size", type=int, default=32)
    p.add_argument("--start-jitter-s", type=float, default=0.0)
    p.add_argument("--opponent", choices=("mirror", "lasthit", "brawler"), default="mirror")
    p.add_argument("--unwalkable-click", choices=("noop", "resolve"), default="noop")
    p.add_argument("--gold-scale", type=float, default=20.0)
    p.add_argument("--xp-scale", type=float, default=0.008)
    p.add_argument("--enemy-scale", type=float, default=1.0)
    p.add_argument("--core", choices=("mlp", "gru"), default="gru")
    p.add_argument("--core-norm", action="store_true")
    p.add_argument("--core-residual", action="store_true")
    p.add_argument("--click-mask", action="store_true")
    p.add_argument("--detach-critic", action="store_true")
    p.add_argument("--preset", choices=("standard", "legacy"), default="standard")
    p.add_argument("--fine-tune", action="store_true",
                   help="standard preset with lr 1e-5, entropy 0 (E32 recipe)")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--entropy-coef", type=float, default=None)
    p.add_argument("--kl-prior", type=float, default=0.0)
    p.add_argument("--lr-anneal", action="store_true")
    p.add_argument("--init-from", type=Path, default=None, help="params only")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt-every", type=int, default=50)
    p.add_argument("--chunk", type=int, default=1, help="updates per device call")
    p.add_argument("--route-artifact", type=Path, default=DEFAULT_ROUTE_ARTIFACT)
    p.add_argument("--out", type=Path, default=Path("lanerl_jax/runs/vec"))
    p.add_argument("--notes", default="")
    return p


def main(argv=None) -> None:
    from flax.serialization import from_state_dict, msgpack_restore
    from .run_manifest import RunDir, file_sha256
    p = build_parser()
    a = p.parse_args(argv)
    if a.start_jitter_s < 0 or a.episode_s * 1000 <= 120_000:
        p.error("episode must end after 120 s; jitter must be non-negative")
    hz = 60.0 / a.step_ticks
    if a.preset == "standard":
        over = dict(lr=1e-5, entropy_coef=0.0) if a.fine_tune else {}
        if a.lr is not None:
            over["lr"] = a.lr
        if a.entropy_coef is not None:
            over["entropy_coef"] = a.entropy_coef
        ppo = PPOConfig.standard(decision_hz=hz, kl_prior_coef=a.kl_prior, **over)
    else:
        ppo = PPOConfig(decision_hz=hz, lr=a.lr or 1e-5,
                        entropy_coef=0.001 if a.entropy_coef is None else a.entropy_coef,
                        kl_prior_coef=a.kl_prior)
    pcfg = PolicyConfig(core=a.core, core_norm=a.core_norm, core_residual=a.core_residual,
                        click_mask=a.click_mask, detach_critic=a.detach_critic)
    if a.init_from is not None and (a.init_from.parent / "manifest.json").exists():
        saved = json.loads((a.init_from.parent / "manifest.json").read_text()).get("config", {}).get("train", {}).get("policy", {})
        if saved:
            pcfg = PolicyConfig(**{**saved, "click_mask": a.click_mask or saved.get("click_mask", False),
                                   "detach_critic": a.detach_critic or saved.get("detach_critic", False)})
    cfg = VecConfig(n_envs=a.envs, rollout_steps=a.rollout, n_updates=a.updates,
                    n_minibatches=a.minibatches, episode_s=a.episode_s, step_ticks=a.step_ticks,
                    bank_size=a.bank_size, start_jitter_s=a.start_jitter_s, opponent=a.opponent,
                    unwalkable_click=a.unwalkable_click, gold_scale=a.gold_scale,
                    xp_scale=a.xp_scale, enemy_scale=a.enemy_scale, lr_anneal=a.lr_anneal,
                    ppo=ppo, policy=pcfg)
    jax.config.update("jax_default_matmul_precision", "highest")
    command = shlex.join([sys.executable, "-m", "lanerl_jax.train.vec_train", *sys.argv[1:]])
    sim = SimConfig.training(route_artifact=a.route_artifact.resolve()).replace(step_ticks=a.step_ticks)
    collector_cfg = {**vars(a), "start_near_wave": True, "environment": "jax-vectorised"}
    run = RunDir(a.out, f"vec-s{a.seed}", {
        "train": {"policy": pcfg._asdict()}, "command": command, "cwd": str(Path.cwd()),
        "ppo": ppo._asdict(), "vec": {k: v for k, v in cfg._asdict().items() if k not in ("ppo", "policy")},
        "collector": collector_cfg, "environment": "jax-vectorised",
        "observation": "viewport+map-fog+structured-HUD",
        "opponent": {"mirror": "mirror-self-play", "lasthit": "scripted lasthit red",
                     "brawler": "scripted brawler red"}[a.opponent],
        "reward": f"relative: (d_gold - {a.enemy_scale}*enemy)/{a.gold_scale} + {a.xp_scale}*(d_xp - enemy) + 5*d_potential",
        "initialization": str(a.init_from) if a.init_from else "random; no prior"}, notes=a.notes)
    route_manifest = a.route_artifact / "manifest.json"
    shutil.copyfile(route_manifest, run.path / "route-manifest.json")
    run.manifest["sim"] = {"resolved": sim.describe(), "fingerprint": sim.fingerprint(),
                           "route_manifest_sha256": file_sha256(route_manifest)}
    run.write()
    print(f"run dir {run.path}", flush=True)

    t0 = time.perf_counter()
    bank = prepare_bank(cfg, sim, run.path / "bank", a.seed)
    print(f"bank of {cfg.bank_size} prepared states in {time.perf_counter() - t0:.0f}s "
          f"(t_ms {float(jnp.min(bank.t_ms)):.0f}-{float(jnp.max(bank.t_ms)):.0f})", flush=True)

    built = make_vec_train(cfg, sim, bank)
    params = None
    if a.init_from is not None:
        params = built["init_params"](jax.random.key(a.seed))
        params = from_state_dict(params, msgpack_restore(a.init_from.read_bytes())["params"])
        run.manifest["init_from"] = {"checkpoint": str(a.init_from), "sha256": file_sha256(a.init_from)}
        run.write()
    if a.kl_prior > 0:
        built = make_vec_train(cfg, sim, bank, prior_params=params)
    runner = built["initial_runner"](jax.random.key(a.seed), params)
    step_fn = jax.jit(built["run_chunk"], static_argnums=1)
    n_dec_update = cfg.n_envs * cfg.learn_agents * cfg.rollout_steps

    t0 = time.perf_counter()
    upd = 0
    while upd < cfg.n_updates:
        n = min(a.chunk, cfg.n_updates - upd)
        t1 = time.perf_counter()
        runner, m = step_fn(runner, n)
        jax.block_until_ready(m)
        dt = time.perf_counter() - t1
        upd += n
        m = jax.tree.map(np.asarray, m)
        cs_n = m["cs_episodes"].sum()
        row = {k: float(np.nanmean(v)) if np.isfinite(v).any() else float("nan") for k, v in m.items()}
        for name in ("cs_at_10min", "gold_at_10min", "xp_at_10min"):
            row[name] = float((np.nan_to_num(m[name]) * m["cs_episodes"]).sum() / cs_n) if cs_n > 0 else float("nan")
        row["cs_episodes"] = float(cs_n)
        mem = jax.local_devices()[0].memory_stats() or {}
        row.update(update=upd, step=int(runner.step), wall_s=round(time.perf_counter() - t0, 1),
                   dec_per_s=round(n * n_dec_update / dt), update_s=round(dt / n, 2),
                   peak_gb=round(mem.get("peak_bytes_in_use", 0) / 1e9, 2))
        run.log(row)
        print(f"upd {upd:>5} reward {row['reward']:+.4f} ent {row['entropy']:.3f} kl {row['approx_kl']:.4f} "
              f"post_kl {row['post_kl']:.4f} ev {row['explained_variance']:.2f} "
              f"cs {row['cs_at_10min']:.1f} (n={cs_n:.0f}) lane {row['lane_dist']:.0f} "
              f"{row['dec_per_s']} dec/s {row['update_s']}s/upd peak {row['peak_gb']}GB", flush=True)
        bad = not np.all(np.isfinite(m["policy_loss"])) or np.any(m["loss_nonfinite"] > 0)
        finite = all(bool(np.all(np.isfinite(np.asarray(x)))) for x in jax.tree.leaves(runner.params))
        if bad or not finite:
            run.save(int(runner.step), upd, {"params": runner.params, "opt_state": runner.opt_state,
                                             "step": runner.step}, latest=False)
            run.set_results(status="diverged", diverged_at_update=upd)
            print(f"DIVERGED at update {upd}", flush=True)
            break
        if a.ckpt_every and upd % a.ckpt_every == 0:
            run.save(int(runner.step), upd, {"params": runner.params, "opt_state": runner.opt_state,
                                             "step": runner.step})
    else:
        run.save(int(runner.step), upd, {"params": runner.params, "opt_state": runner.opt_state,
                                         "step": runner.step})
        run.set_results(status="finished", updates=upd)
    run.close()
    print(f"run dir {run.path}", flush=True)


if __name__ == "__main__":
    main()
