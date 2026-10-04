"""Vectorised on-device PPO on the 26.19 modern world (``sim.modern_step``).

TOOL (not yet run for an experiment). The scan trainer of ``vec_train`` with a
modern-world environment adapter; the learner half is shared by import
(``scan_ppo.make_batch_fn`` / ``ppo_learn`` / ``Transition`` / ``VecRunner``,
``learner.make_learner``, ``ppo``), so only the environment differs:

* RESET BANK. ``prepare_modern_bank`` runs ``modern_step.step`` from
  ``init_state(cfg)`` for ``start_s`` of game time (default 60 s: the first
  wave spawns at 30 s and reaches top lane about then) with one scripted move
  order per champion to a hold point on the top lane (``hold_offset`` game
  units along the lane path from its own outer turret, +-``hold_jitter``
  uniform per bank entry). Each entry has its own PRNG key (crit rolls and
  everything ``step`` draws), so entries diverge. A reset is a gather from the
  bank, as in ``vec_train``.
* DECISION. Every ``step_ticks`` world ticks (30 Hz; default 3 -> 10 Hz, the
  legacy rate): both teams observe (``build_modern_observation`` with
  ``modern_frames``), the policy samples ``(button, sx, sy)``,
  ``modern_orders_from`` decodes, the orders apply on the first tick and the
  remaining ticks run with ``no_orders`` (move / attack / attack-move goals
  persist in the world), exactly as the legacy ``env_step`` applies once and
  advances.
* BUTTONS. ``LanePolicy`` has no ``choice`` head, so ``buy`` / ``sell`` /
  ``use_item`` cannot name a catalog row or slot; they are masked off by
  default (``MaskedLanePolicy``: logit -1e4, zero probability, finite
  entropy), together with ``level_q..r`` because the loadouts use
  ``auto_skill``. Champions keep their starting items (Doran's Blade, Health
  Potion) and never shop. ``--buttons-off`` changes the set.
* REWARD. ``modern_relative_reward``: ``vec_train._relative_reward`` on the
  modern state with the same term names (``cs`` = the gold term, ``xp``,
  ``approach``, ``death`` = 0, optional ``health`` / ``tower``). Gold is
  ``econ.gold_total`` (lifetime earned, so buying is not a loss); the
  lane-keep potential is the distance to ``cfg.lane_path`` clipped between
  the two top outer turrets, minus the legacy 1400-unit corridor; tower damage
  is the HP lost by the enemy top outer turret (team credit; the modern world
  has no per-champion tower attribution, so ``tower_damage_personal`` is not
  offered).
* OPPONENT. ``mirror`` (shared params, both learn), ``afk`` (red always
  ``noop``; red still respawns/auto-levels), ``frozen`` (red from
  ``--opponent-from``).

    python -m lanerl_jax.train.modern_vec_train --envs 64 --rollout 128 --updates 2 \
        --opponent mirror --out /mnt/nfs/checkpoints/lanerl-jax/modern-smoke

SIGTERM/SIGINT/SIGUSR1 (gpup pre-emption gives SIGTERM 60 s ahead): the loop
finishes the current chunk, saves a checkpoint and records
``status=preempted``. Keep ``--chunk`` small enough that one chunk fits in
that window.
"""
from __future__ import annotations

import argparse
import json
import shlex
import signal
import sys
import time
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..obs import modern_builder as OB
from ..sim import modern_world_types as W
from .learner import make_learner
from .modern_actions import MODERN_BUTTON_INDEX, MODERN_BUTTONS, modern_orders_from, screen_usage
from .policy import LanePolicy, PolicyConfig
from .ppo import PPOConfig, factored_log_prob
from .reward import LANE_HALF_WIDTH
from .scan_ppo import Transition, VecRunner, make_batch_fn, ppo_learn

__all__ = ["ModernVecConfig", "ModernEnvState", "MaskedLanePolicy", "DEFAULT_BUTTONS_OFF", "TICK_HZ",
           "default_loadouts", "make_env", "prepare_modern_bank", "modern_relative_reward",
           "modern_policy_config", "make_modern_vec_train", "main"]

TICK_HZ = 30
DEFAULT_BUTTONS_OFF = ("buy", "sell", "use_item", "level_q", "level_w", "level_e", "level_r")


class ModernVecConfig(NamedTuple):
    n_envs: int = 256
    rollout_steps: int = 128
    n_updates: int = 1000
    n_minibatches: int = 4
    #: game clock (s) at which an episode ends; episodes start at ``start_s``.
    episode_s: float = 600.0
    start_s: float = 60.0
    step_ticks: int = 3
    observation_horizon_s: float | None = None
    stagger_initial: bool = True
    bank_size: int = 16
    hold_offset: float = 800.0
    hold_jitter: float = 300.0
    opponent: str = "mirror"
    gold_scale: float = 20.0
    xp_scale: float = 0.008
    enemy_scale: float = 1.0
    health_loss_gold: float = 0.0
    tower_damage_gold: float = 0.0
    lr_anneal: bool = False
    buttons_off: tuple = DEFAULT_BUTTONS_OFF
    ppo: PPOConfig = PPOConfig()
    policy: PolicyConfig = PolicyConfig()

    @property
    def decision_hz(self) -> float:
        return TICK_HZ / self.step_ticks

    @property
    def learn_agents(self) -> int:
        return 2 if self.opponent == "mirror" else 1


class ModernEnvState(NamedTuple):
    world: object           # sim.modern_step.ModernState
    kills: jax.Array        # (2,) int32 enemy-champion deaths this episode
    tower_damage: jax.Array  # (2,) HP the enemy top outer turret lost this episode (team credit)


class MaskedLanePolicy(LanePolicy):
    """``LanePolicy`` with some buttons masked off (logit -1e4). Same parameter
    tree as ``LanePolicy``, so checkpoints load into either."""
    button_off: tuple = ()

    def __call__(self, *args, **kwargs):
        out = super().__call__(*args, **kwargs)
        gru = self.cfg.core == "gru"
        logits, carry = out if gru else (out, None)
        if self.button_off:
            keep = jnp.asarray([i not in self.button_off for i in range(self.cfg.n_buttons)])
            logits = logits._replace(button=jnp.where(keep, logits.button, -1e4))
        return (logits, carry) if gru else logits


def modern_policy_config(**kw) -> PolicyConfig:
    """``PolicyConfig`` with the modern-world-v1 widths (32x20 entities, 32 self, 6 global, 19 buttons)."""
    return PolicyConfig(**{**dict(entity_dim=OB.MODERN_ENTITY_DIM, self_dim=OB.MODERN_WORLD_SELF_DIM,
                                  n_buttons=len(MODERN_BUTTONS)), **kw})


def default_loadouts(names=("Garen", "Jax")):
    """The test/bench loadouts: Doran's Blade + Health Potion, Garen's default page, the Jax
    Precision/Resolve page of ``modern_world_harness``; auto skill order."""
    from ..sim import modern_rune_data as RD
    from ..sim import modern_world as MW
    pages = {"Garen": RD.GAREN_DEFAULT_PAGE,
             "Jax": RD.RunePage(RD.PRECISION, 8010, (9111, 9104, 8299), RD.RESOLVE, (8444, 8451),
                                (5005, 5008, 5001))}
    return tuple(MW.Loadout(n, items=(1055, 2003), rune_page=pages[n]) for n in names)


# ---- static geometry ----------------------------------------------------------------

def _outer_turrets(wcfg) -> np.ndarray:
    """(2,) unit index of the blue / red top outer turret (as ``modern_frames``)."""
    kind, team, sub, lane = (np.asarray(a) for a in (wcfg.unit_kind, wcfg.unit_team, wcfg.unit_sub,
                                                       wcfg.unit_lane))
    return np.asarray([np.flatnonzero((kind == W.KIND_TURRET) & (team == t) & (sub == 0) & (lane == 2))[0]
                       for t in (0, 1)], np.int32)


def _arc_position(path: np.ndarray, p: np.ndarray) -> float:
    """Arc length along ``path`` of the closest point to ``p``."""
    a, b = path[:-1], path[1:]
    ab = b - a
    seg = np.hypot(ab[:, 0], ab[:, 1])
    t = np.clip(np.sum((p - a) * ab, -1) / np.maximum(seg ** 2, 1e-9), 0, 1)
    d = np.hypot(*(a + t[:, None] * ab - p).T)
    i = int(np.argmin(d))
    return float(np.concatenate([[0.0], np.cumsum(seg)])[i] + t[i] * seg[i])


def _point_at(path: np.ndarray, s: float) -> np.ndarray:
    seg = np.hypot(*np.diff(path, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    s = float(np.clip(s, 0.0, cum[-1]))
    i = int(np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg) - 1))
    return path[i] + (s - cum[i]) / max(seg[i], 1e-9) * (path[i + 1] - path[i])


def lane_segment(path, start_xy, end_xy) -> np.ndarray:
    """The part of polyline ``path`` between the projections of two points (in path order)."""
    path = np.asarray(path, np.float64)
    s0, s1 = sorted((_arc_position(path, np.asarray(start_xy, np.float64)),
                     _arc_position(path, np.asarray(end_xy, np.float64))))
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))])
    inner = path[(cum > s0) & (cum < s1)]
    return np.concatenate([_point_at(path, s0)[None], inner, _point_at(path, s1)[None]]).astype(np.float32)


def lane_path_distance(x, y, segment, corridor: float = LANE_HALF_WIDTH):
    """Distance from each point to the lane corridor around ``segment`` (0 inside), the
    modern counterpart of ``reward.lane_corridor_distance``."""
    a, b = segment[:-1], segment[1:]
    ab = b - a
    p = jnp.stack([x, y], -1)[..., None, :]
    t = jnp.clip(jnp.sum((p - a) * ab, -1) / jnp.maximum(jnp.sum(ab * ab, -1), 1e-6), 0.0, 1.0)
    q = a + t[..., None] * ab
    d = jnp.min(jnp.sqrt(jnp.sum((p - q) ** 2, -1)), -1)
    return jnp.maximum(d - corridor, 0.0)


class ModernEnv(NamedTuple):
    wcfg: object
    frames: tuple
    outer: jax.Array          # (2,) blue / red top outer turret unit
    segment: jax.Array        # (P, 2) top lane between the outer turrets, blue -> red


def make_env(wcfg) -> ModernEnv:
    if abs(wcfg.dt * TICK_HZ - 1.0) > 1e-6:
        raise ValueError(f"modern trainer assumes a {TICK_HZ} Hz world (dt={wcfg.dt})")
    outer = _outer_turrets(wcfg)
    xy = np.stack([np.asarray(wcfg.unit_x), np.asarray(wcfg.unit_y)], -1)
    seg = lane_segment(np.asarray(wcfg.lane_path), xy[outer[0]], xy[outer[1]])
    if np.hypot(*(seg[0] - xy[outer[0]])) > np.hypot(*(seg[0] - xy[outer[1]])):
        seg = seg[::-1].copy()
    return ModernEnv(wcfg, OB.modern_frames(wcfg), jnp.asarray(outer), jnp.asarray(seg))


# ---- reward ---------------------------------------------------------------------------

def modern_relative_reward(prev, nxt, cfg: ModernVecConfig, env: ModernEnv):
    """``vec_train._relative_reward`` on two ``ModernState``: (2,) per champion and the
    same term names. Returns ``(total, terms, tower_hp_lost (2,) credited per champion)``."""
    d_gold = nxt.econ.gold_total[:2] - prev.econ.gold_total[:2]
    d_xp = nxt.econ.xp[:2] - prev.econ.xp[:2]
    pot_p = -lane_path_distance(prev.x[:2], prev.y[:2], env.segment) / 10000.0
    pot_n = -lane_path_distance(nxt.x[:2], nxt.y[:2], env.segment) / 10000.0
    es = cfg.enemy_scale
    gold = (d_gold - es * d_gold[::-1]) / cfg.gold_scale
    xp = cfg.xp_scale * (d_xp - es * d_xp[::-1])
    shaping = 5.0 * (pot_n - pot_p)
    total = gold + xp + shaping
    terms = {"cs": gold, "death": jnp.zeros_like(gold), "approach": shaping, "xp": xp}
    if cfg.health_loss_gold:
        lost = jnp.maximum((nxt.max_hp[:2] - nxt.hp[:2]) - (prev.max_hp[:2] - prev.hp[:2]), 0.0)
        health = -cfg.health_loss_gold / cfg.gold_scale * lost / jnp.maximum(prev.max_hp[:2], 1.0) * prev.alive[:2]
        total = total + health
        terms["health"] = health
    idx = env.outer
    damage = jnp.clip(prev.hp[idx] - nxt.hp[idx], 0.0, prev.hp[idx]) * prev.alive[idx]
    credited = damage[::-1]
    if cfg.tower_damage_gold:
        tower = cfg.tower_damage_gold / cfg.gold_scale * credited / jnp.maximum(prev.max_hp[idx][::-1], 1.0)
        total = total + tower
        terms["tower"] = tower
    return total, terms, credited


# ---- environment ------------------------------------------------------------------------

def _advance(env: ModernEnv, world, orders, ticks: int):
    """``ticks`` world ticks: ``orders`` on the first, ``no_orders`` after (one ``step`` call site)."""
    from ..sim import modern_step as MS
    idle = MS.no_orders()

    def body(s, i):
        o = jax.tree.map(lambda a, b: jnp.where(i == 0, a, b), orders, idle)
        return MS.step(s, o, env.wcfg)[0], None
    return jax.lax.scan(body, world, jnp.arange(ticks))[0]


def hold_points(env: ModernEnv, cfg: ModernVecConfig, k: int, seed: int) -> np.ndarray:
    """(k, 2, 2) per-entry, per-champion hold point on the top lane."""
    seg = np.asarray(env.segment, np.float64)
    length = float(np.sum(np.hypot(*np.diff(seg, axis=0).T)))
    jit = np.random.default_rng(seed).uniform(-cfg.hold_jitter, cfg.hold_jitter, (k, 2))
    off = np.clip(cfg.hold_offset + jit, 0.0, length)
    return np.asarray([[_point_at(seg, off[i, 0]), _point_at(seg, length - off[i, 1])] for i in range(k)],
                      np.float32)


def prepare_modern_bank(cfg: ModernVecConfig, env: ModernEnv, seed: int, out: Path | None = None):
    """``bank_size`` start states at ``start_s``: stacked ``ModernEnvState`` (leading axis K)."""
    from ..sim import modern_step as MS
    k = cfg.bank_size
    s0 = MS.init_state(env.wcfg, seed=seed)
    batch = jax.tree.map(lambda a: jnp.broadcast_to(a, (k,) + jnp.shape(a)), s0)
    batch = batch._replace(key=jax.random.split(jax.random.PRNGKey(seed), k))
    targets = jnp.asarray(hold_points(env, cfg, k, seed))
    ticks = int(round(cfg.start_s * TICK_HZ))

    def one(s, tgt):
        o = MS.no_orders()._replace(move=jnp.ones((2,), bool), move_x=tgt[:, 0], move_y=tgt[:, 1])
        return _advance(env, s, o, ticks)
    worlds = jax.jit(jax.vmap(one))(batch, targets)
    bank = ModernEnvState(worlds, jnp.zeros((k, 2), jnp.int32), jnp.zeros((k, 2), jnp.float32))
    xy = np.stack([np.asarray(worlds.x[:, :2]), np.asarray(worlds.y[:, :2])], -1)
    miss = np.hypot(*(xy - np.asarray(targets)).transpose(2, 0, 1))          # (k, 2)
    if (miss > 300.0).any():
        print(f"WARNING: bank champions did not reach their hold points (max miss blue "
              f"{miss[:, 0].max():.0f}, red {miss[:, 1].max():.0f} units); check routing", flush=True)
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        rows = [{"entry": i, "seed": seed, "t": float(worlds.t[i]), "hold": np.asarray(targets[i]).tolist(),
                 "xy": xy[i].tolist(), "miss": miss[i].tolist()} for i in range(k)]
        (out / "setup.json").write_text(json.dumps(rows, indent=2))
    return bank


# ---- trainer ----------------------------------------------------------------------------

def _sample(logits, key):
    kb, kx, ky = jax.random.split(key, 3)
    a = (jax.random.categorical(kb, logits.button), jax.random.categorical(kx, logits.screen_x),
         jax.random.categorical(ky, logits.screen_y))
    used = screen_usage(a[0])
    lp = factored_log_prob((logits.button, logits.screen_x, logits.screen_y), a, used)
    return a, lp, (used, jnp.zeros_like(used))


def make_modern_vec_train(cfg: ModernVecConfig, env: ModernEnv, bank, *, prior_params=None,
                          opponent_params=None):
    """Same contract as ``vec_train.make_vec_train`` (``initial_runner``, ``run_chunk``, ``rollout``,
    ``init_params``, ``policy``, ``loss``, ``collect``, ``learn``)."""
    if cfg.opponent not in ("mirror", "frozen", "afk"):
        raise ValueError(f"unknown opponent {cfg.opponent!r}")
    if cfg.opponent == "frozen" and opponent_params is None:
        raise ValueError("frozen opponent requires opponent_params")
    if cfg.policy.click_mask:
        raise ValueError("click_mask is a legacy-map feature; not available on the modern world")
    pc = cfg.policy
    want = (OB.MODERN_ENTITY_DIM, OB.MODERN_WORLD_SELF_DIM, len(MODERN_BUTTONS))
    if (pc.entity_dim, pc.self_dim, pc.n_buttons) != want:
        raise ValueError(f"policy widths {(pc.entity_dim, pc.self_dim, pc.n_buttons)} != modern {want}; "
                         "use modern_policy_config")
    unknown = set(cfg.buttons_off) - set(MODERN_BUTTONS)
    if unknown:
        raise ValueError(f"unknown buttons {sorted(unknown)}")
    off = tuple(sorted(MODERN_BUTTON_INDEX[b] for b in cfg.buttons_off))
    if MODERN_BUTTON_INDEX["noop"] in off:
        raise ValueError("noop cannot be masked")
    policy = MaskedLanePolicy(pc, off)
    recurrent = pc.core == "gru"
    n_learn = cfg.learn_agents
    n_rows = cfg.n_envs * n_learn
    if n_rows % cfg.n_minibatches:
        raise ValueError("minibatches must divide envs x learning agents")
    if cfg.episode_s <= cfg.start_s:
        raise ValueError("episode_s must exceed start_s")
    full_ms = jnp.asarray(cfg.episode_s * 1000.0, jnp.float32)
    K = jax.tree.leaves(bank)[0].shape[0]
    core_dim = pc.core_dim if recurrent else 0
    tx, loss = make_learner(
        policy, cfg.ppo._replace(n_minibatches=cfg.n_minibatches),
        anneal_steps=(cfg.n_updates * cfg.ppo.epochs * cfg.n_minibatches) if cfg.lr_anneal else 0,
        prior_params=prior_params)
    horizon = cfg.observation_horizon_s or cfg.episode_s

    def _obs(world):
        per = [OB.build_modern_observation(world, t, env.frames[t], env.wcfg, horizon_s=horizon) for t in (0, 1)]
        return jax.tree.map(lambda a, b: jnp.stack([a, b]), *per)

    def _apply(params, obs, carry):
        if recurrent:
            return policy.apply(params, obs.entities, obs.entity_pad_mask, obs.self_vec, obs.global_vec, carry)
        return policy.apply(params, obs.entities, obs.entity_pad_mask, obs.self_vec, obs.global_vec), carry

    def _env_step(runner: VecRunner, _):
        rng, sk = jax.random.split(runner.rng)

        def one(es: ModernEnvState, carry, deadline, key):
            k_act, k_reset = jax.random.split(key)
            state = es.world
            obs = _obs(state)
            logits, new_carry = _apply(runner.params, obs, carry)
            if cfg.opponent == "frozen":
                fixed, fixed_carry = _apply(opponent_params, obs, carry)
                logits = jax.tree.map(lambda a, b: a.at[1].set(b[1]), logits, fixed)
                new_carry = new_carry.at[1].set(fixed_carry[1])
            action, log_prob, (uses_screen, uses_target) = _sample(logits, k_act)
            if cfg.opponent == "afk":
                action = tuple(a.at[1].set(0) for a in action)
            orders = modern_orders_from(action, state, env.frames)
            nxt = _advance(env, state, orders, cfg.step_ticks)
            reward, terms, tower_hp = modern_relative_reward(state, nxt, cfg, env)
            died = (nxt.econ.dead[:2] & ~state.econ.dead[:2])
            kills = es.kills + died[::-1].astype(jnp.int32)
            tower_damage = es.tower_damage + tower_hp
            done = (nxt.t * 1000.0 >= deadline) | nxt.game_over
            done_full = done & (deadline >= full_ms)
            deadline = jnp.where(done, full_ms, deadline)
            at_end = lambda v: jnp.where(done_full, v, jnp.zeros_like(v))
            t = Transition(obs.entities, obs.entity_pad_mask, obs.self_vec, obs.global_vec,
                           action, log_prob, uses_screen, uses_target, logits.value,
                           reward, jnp.broadcast_to(done, reward.shape), terms,
                           at_end(nxt.champ.cs[:2].astype(jnp.float32)),
                           at_end(nxt.econ.gold_total[:2].astype(jnp.float32)),
                           at_end(nxt.econ.xp[:2].astype(jnp.float32)),
                           jnp.broadcast_to(done_full, reward.shape), died.astype(jnp.float32),
                           lane_path_distance(nxt.x[:2], nxt.y[:2], env.segment), None,
                           at_end(nxt.hp[:2] / jnp.maximum(nxt.max_hp[:2], 1.0)), at_end(tower_damage),
                           at_end(kills))
            nes = ModernEnvState(nxt, kills, tower_damage)
            idx = jax.random.randint(k_reset, (), 0, K)
            fresh = jax.tree.map(lambda b: b[idx], bank)
            nes = jax.tree.map(lambda a, b: jnp.where(done, b, a), nes, fresh)
            new_carry = jnp.where(done, jnp.zeros_like(new_carry), new_carry)
            return nes, new_carry, deadline, t

        keys = jax.random.split(sk, cfg.n_envs)
        env_state, carry, deadline_ms, tr = jax.vmap(one)(runner.env_state, runner.carry, runner.deadline_ms, keys)
        return runner._replace(env_state=env_state, carry=carry, deadline_ms=deadline_ms, rng=rng), tr

    _batch = make_batch_fn(cfg, n_learn, recurrent, False, core_dim)
    # ``deaths_per_episode`` is per played episode (start_s .. episode_s).
    metric_cfg = cfg._replace(episode_s=cfg.episode_s - cfg.start_s)

    def collect(runner: VecRunner):
        carry0 = runner.carry
        runner, tr = jax.lax.scan(_env_step, runner, None, length=cfg.rollout_steps)
        return runner, tr, carry0

    def learn(runner: VecRunner, tr: Transition, carry0):
        last_obs = jax.vmap(_obs)(runner.env_state.world)
        last_logits, _ = jax.vmap(lambda o, c: _apply(runner.params, o, c))(last_obs, runner.carry)
        return ppo_learn(runner, tr, carry0, last_logits.value, cfg=metric_cfg, tx=tx, loss=loss,
                         batch_fn=_batch, n_learn=n_learn, buttons=MODERN_BUTTONS)

    def _update(runner: VecRunner, _):
        return learn(*collect(runner))

    def init_params(rng):
        obs0 = _obs(jax.tree.map(lambda b: b[0], bank.world))
        if recurrent:
            return policy.init(rng, obs0.entities, obs0.entity_pad_mask, obs0.self_vec, obs0.global_vec,
                               policy.initial_carry((2,)))
        return policy.init(rng, obs0.entities, obs0.entity_pad_mask, obs0.self_vec, obs0.global_vec)

    def initial_runner(rng, params=None) -> VecRunner:
        rng, ik, bk, dk = jax.random.split(rng, 4)
        if params is None:
            params = init_params(ik)
        idx = jax.random.randint(bk, (cfg.n_envs,), 0, K)
        env_state = jax.tree.map(lambda b: b[idx], bank)
        carry = jnp.zeros((cfg.n_envs, 2, core_dim), jnp.float32)
        start_ms = jnp.max(env_state.world.t) * 1000.0
        floor = start_ms + 2.0 * cfg.rollout_steps / cfg.decision_hz * 1000.0
        deadline_ms = jax.random.uniform(dk, (cfg.n_envs,), jnp.float32,
                                         minval=jnp.minimum(floor, full_ms), maxval=full_ms)
        if not cfg.stagger_initial:
            deadline_ms = jnp.full((cfg.n_envs,), full_ms)
        return VecRunner(params, tx.init(params), env_state, carry, rng, jnp.asarray(0, jnp.int32), deadline_ms)

    def run_chunk(runner: VecRunner, n: int):
        return jax.lax.scan(_update, runner, None, length=n)

    def rollout(runner: VecRunner):
        carry0 = runner.carry
        runner, tr = jax.lax.scan(_env_step, runner, None, length=cfg.rollout_steps)
        zeros = jnp.zeros_like(tr.reward)
        return runner, tr, _batch(tr, zeros, zeros, carry0)

    return dict(initial_runner=initial_runner, run_chunk=run_chunk, rollout=rollout, init_params=init_params,
                policy=policy, loss=loss, collect=collect, learn=learn, obs=_obs)


# ---- CLI --------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--envs", "--n-envs", dest="envs", type=int, default=256)
    p.add_argument("--rollout", "--rollout-steps", dest="rollout", type=int, default=128)
    p.add_argument("--updates", "--n-updates", dest="updates", type=int, default=1000)
    p.add_argument("--minibatches", type=int, default=4)
    p.add_argument("--episode-s", type=float, default=600.0, help="game clock at which episodes end")
    p.add_argument("--start-s", type=float, default=60.0, help="game clock of the reset bank")
    p.add_argument("--step-ticks", type=int, default=3, help="30 Hz world ticks per decision")
    p.add_argument("--bank-size", type=int, default=16)
    p.add_argument("--hold-offset", type=float, default=800.0)
    p.add_argument("--hold-jitter", type=float, default=300.0)
    p.add_argument("--champions", default="Garen,Jax", help="BLUE,RED")
    p.add_argument("--fog", choices=("rays", "fast", "off"), default="rays")
    p.add_argument("--lanes", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--no-jungle", action="store_true")
    p.add_argument("--no-objectives", action="store_true")
    p.add_argument("--opponent", choices=("mirror", "afk", "frozen"), default="mirror")
    p.add_argument("--opponent-from", type=Path, default=None, help="checkpoint for --opponent frozen")
    p.add_argument("--buttons-off", default=",".join(DEFAULT_BUTTONS_OFF),
                   help="comma list of masked buttons ('' = none)")
    p.add_argument("--gold-scale", type=float, default=20.0)
    p.add_argument("--xp-scale", type=float, default=0.008)
    p.add_argument("--enemy-scale", type=float, default=1.0)
    p.add_argument("--health-loss-gold", type=float, default=0.0)
    p.add_argument("--tower-damage-gold", type=float, default=0.0)
    p.add_argument("--core", choices=("mlp", "gru"), default="gru")
    p.add_argument("--core-norm", action="store_true")
    p.add_argument("--core-residual", action="store_true")
    p.add_argument("--detach-critic", action="store_true")
    p.add_argument("--small-policy", action="store_true", help="tiny widths (smoke tests only)")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--entropy-coef", type=float, default=None)
    p.add_argument("--kl-prior", type=float, default=0.0)
    p.add_argument("--lr-anneal", action="store_true")
    p.add_argument("--init-from", type=Path, default=None, help="params only")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt-every", type=int, default=50)
    p.add_argument("--chunk", type=int, default=1, help="updates per device call")
    p.add_argument("--out", type=Path, default=Path("lanerl_jax/runs/modern-vec"))
    p.add_argument("--notes", default="")
    return p


SMALL_POLICY = dict(d_model=32, n_layers=1, ffn_dim=32, ctx_dim=32, core_dim=32, mlp_hidden=32, mlp_layers=1)


def main(argv=None) -> None:
    from flax.serialization import from_state_dict, msgpack_restore
    from ..jax_cache import enable_compile_cache
    from ..sim import modern_world as MW
    from .run_manifest import RunDir, file_sha256
    p = build_parser()
    a = p.parse_args(argv)
    if a.opponent == "frozen" and a.opponent_from is None:
        p.error("--opponent frozen needs --opponent-from")
    stopping: list[int] = []
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, lambda signum, frame: stopping.append(signum))
    cache = enable_compile_cache()
    jax.config.update("jax_default_matmul_precision", "highest")
    hz = TICK_HZ / a.step_ticks
    over = {k: v for k, v in (("lr", a.lr), ("entropy_coef", a.entropy_coef)) if v is not None}
    ppo = PPOConfig.standard(decision_hz=hz, kl_prior_coef=a.kl_prior, **over)
    pcfg = modern_policy_config(core=a.core, core_norm=a.core_norm, core_residual=a.core_residual,
                                detach_critic=a.detach_critic, **(SMALL_POLICY if a.small_policy else {}))
    if a.init_from is not None and (a.init_from.parent / "manifest.json").exists():
        saved = json.loads((a.init_from.parent / "manifest.json").read_text()).get("config", {}).get(
            "train", {}).get("policy", {})
        if saved:
            pcfg = PolicyConfig(**saved)
    buttons_off = tuple(b for b in a.buttons_off.split(",") if b)
    cfg = ModernVecConfig(n_envs=a.envs, rollout_steps=a.rollout, n_updates=a.updates, n_minibatches=a.minibatches,
                          episode_s=a.episode_s, start_s=a.start_s, step_ticks=a.step_ticks, bank_size=a.bank_size,
                          hold_offset=a.hold_offset, hold_jitter=a.hold_jitter, opponent=a.opponent,
                          gold_scale=a.gold_scale, xp_scale=a.xp_scale, enemy_scale=a.enemy_scale,
                          health_loss_gold=a.health_loss_gold, tower_damage_gold=a.tower_damage_gold,
                          lr_anneal=a.lr_anneal, buttons_off=buttons_off, ppo=ppo, policy=pcfg)
    names = tuple(a.champions.split(","))
    fog = False if a.fog == "off" else a.fog
    t0 = time.perf_counter()
    wcfg = MW.build_config(default_loadouts(names), fog=fog, lanes=tuple(a.lanes), jungle=not a.no_jungle,
                           objectives=not a.no_objectives)
    env = make_env(wcfg)
    command = shlex.join([sys.executable, "-m", "lanerl_jax.train.modern_vec_train", *sys.argv[1:]])
    world_desc = {"ruleset": "modern 26.19 (sim.modern_step)", "champions": names, "fog": a.fog,
                  "lanes": a.lanes, "jungle": not a.no_jungle, "objectives": not a.no_objectives,
                  "loadouts": [repr(lo) for lo in wcfg.loadouts], "profile": wcfg.profile,
                  "tick_hz": TICK_HZ}
    run = RunDir(a.out, f"modern-vec-s{a.seed}", {
        "train": {"policy": pcfg._asdict()}, "command": command, "cwd": str(Path.cwd()),
        "ppo": ppo._asdict(), "vec": {k: v for k, v in cfg._asdict().items() if k not in ("ppo", "policy")},
        "environment": "jax-vectorised-modern-world", "world": world_desc,
        "observation": OB.PROFILE, "action": "modern-world-v1 (button, sx, sy); choice heads absent",
        "buttons_off": buttons_off,
        "opponent": {"mirror": "mirror-self-play", "afk": "afk red (noop)", "frozen": f"frozen {a.opponent_from}"}[
            a.opponent],
        "reward": (f"relative: (d_gold_total - {a.enemy_scale}*enemy)/{a.gold_scale} + {a.xp_scale}*(d_xp - enemy) "
                   f"+ 5*d_potential(top lane path) + health {a.health_loss_gold} + tower {a.tower_damage_gold}"),
        "initialization": str(a.init_from) if a.init_from else "random; no prior",
        "compile_cache": cache}, notes=a.notes)
    print(f"run dir {run.path}", flush=True)

    bank = prepare_modern_bank(cfg, env, a.seed, run.path / "bank")
    jax.block_until_ready(bank.world.t)
    print(f"bank of {cfg.bank_size} start states at t={float(jnp.min(bank.world.t)):.1f}s "
          f"in {time.perf_counter() - t0:.0f}s", flush=True)
    # Parameter templates come from a mirror build (a frozen build needs the opponent first).
    template = make_modern_vec_train(cfg._replace(opponent="mirror"), env, bank)["init_params"]
    params = None
    if a.init_from is not None:
        params = from_state_dict(template(jax.random.key(a.seed)), msgpack_restore(a.init_from.read_bytes())["params"])
        run.manifest["init_from"] = {"checkpoint": str(a.init_from), "sha256": file_sha256(a.init_from)}
    opponent_params = None
    if a.opponent == "frozen":
        opponent_params = from_state_dict(template(jax.random.key(a.seed)),
                                          msgpack_restore(a.opponent_from.read_bytes())["params"])
        run.manifest["opponent_from"] = {"checkpoint": str(a.opponent_from),
                                         "sha256": file_sha256(a.opponent_from)}
    run.write()
    built = make_modern_vec_train(cfg, env, bank, prior_params=params if a.kl_prior > 0 else None,
                                  opponent_params=opponent_params)
    runner = built["initial_runner"](jax.random.key(a.seed), params)
    step_fn = jax.jit(built["run_chunk"], static_argnums=1)
    n_dec_update = cfg.n_envs * cfg.learn_agents * cfg.rollout_steps
    payload = lambda r: {"params": r.params, "opt_state": r.opt_state, "step": r.step}

    t0 = time.perf_counter()
    upd = 0
    status = "finished"
    while upd < cfg.n_updates:
        if stopping:
            status = "preempted"
            break
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
              f"ev {row['explained_variance']:.2f} cs {row['cs_at_10min']:.1f} (n={cs_n:.0f}) "
              f"lane {row['lane_dist']:.0f} {row['dec_per_s']} dec/s {row['update_s']}s/upd "
              f"peak {row['peak_gb']}GB", flush=True)
        bad = not np.all(np.isfinite(m["policy_loss"])) or np.any(m["loss_nonfinite"] > 0)
        if bad or not all(bool(np.all(np.isfinite(np.asarray(x)))) for x in jax.tree.leaves(runner.params)):
            run.save(int(runner.step), upd, payload(runner), latest=False)
            run.set_results(status="diverged", diverged_at_update=upd)
            print(f"DIVERGED at update {upd}", flush=True)
            run.close()
            return
        if a.ckpt_every and upd % a.ckpt_every == 0:
            run.save(int(runner.step), upd, payload(runner))
    run.save(int(runner.step), upd, payload(runner))
    run.set_results(status=status, updates=upd, **({"signal": int(stopping[0])} if stopping else {}))
    run.close()
    print(f"{status} after {upd} updates; run dir {run.path}", flush=True)


if __name__ == "__main__":
    main()
