"""Experimental JAX counterpart to the source-server farming collector.

Blue receives structured screen/fog observations and emits screen clicks; red
receives NOOP. No learner, demonstrations, teleports or scripted farming live
here. Native JAX spell ranks change during a tick, whereas source rank commands
consume control decisions. Extra per-environment NOOP decisions mirror that elapsed time but
cannot reproduce the exact within-tick ordering. Map/dynamics fidelity remains
subject to the ledger; this collector is not evidence of matched convergence.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from lanerl_rl.constants import BUTTON_INDEX, BUTTONS, SCREEN_X_VALUES, SCREEN_Y_VALUES
from ..obs.builder import build_observation
from ..obs.frame import make_lane_frame
from ..sim.config import SimConfig
from ..sim.init import TOP_OUTER_TURRET, init_lane
from ..sim.orders import OrderKind, Orders
from ..sim.state import Team
from ..sim.step import env_step
from .actions import orders_from
from .policy import PolicyConfig
from .ppo import PPOConfig
from .reward import lane_corridor_distance
from .server_train import WAVE_START_MS, WAVE_START_POS, RED_WAVE_START_POS, farm_reward
from ..parity.policy_driver import _lane_frames

GROUND_CLICK_NORMALIZATION = 'raw-screen-projection; environment-terrain-exit; no-decoder-EDT'


WAVE_START = {0: (WAVE_START_POS, 100.), 1: (RED_WAVE_START_POS, 250.)}


def validate_wave_start(state, team=0):
    """Check actual reset trajectory outcome, without repairing its state."""
    goal, tol = WAVE_START[team]
    position = np.array([state.x[team], state.y[team]])
    if not bool(state.alive[team]) or int(state.deaths[team]) or int(state.cs[team]):
        raise RuntimeError('wave-start setup died or farmed before policy control')
    if np.linalg.norm(position - goal) > tol:
        raise RuntimeError(f'wave-start setup missed {goal}: {position.tolist()}')
    if float(state.t_ms) < WAVE_START_MS:
        raise RuntimeError('wave-start setup returned before 120 game seconds')


class JaxFarmCollector:
    """Same host-facing API as ServerCollector; only blue observations returned.

    ``sim_config`` is an explicit diagnostic override. The default loads the
    production route artifact, top waves, normal turrets and map vision. A
    no-route override is recorded as a control, never silently promoted.
    ``states`` is privileged diagnostic state and is never an actor input.
    """
    def __init__(self, n, out, episode_s=600., start_near_wave=False, step_ticks=2,
                 *, seed=0, sim_config=None, batch_mode="auto", teams=(0,)):
        if n <= 0 or episode_s <= 0 or step_ticks <= 0 or int(step_ticks) != step_ticks:
            raise ValueError('n, episode_s and integer step_ticks must be positive')
        if start_near_wave and episode_s * 1000 <= WAVE_START_MS:
            raise ValueError('wave-start episode must end after 120 game seconds')
        contract = PolicyConfig()
        if (contract.action_interface != 'screen-click-v2'
                or contract.observation_interface != 'viewport-structured-v3'):
            raise ValueError('farming collector requires screen-click-v2 / viewport-structured-v3')
        if batch_mode not in ('auto', 'map', 'vmap'):
            raise ValueError('batch_mode must be auto, map or vmap')
        self.platform = jax.default_backend()
        self.batch_mode = ('map' if self.platform == 'cpu' else 'vmap') if batch_mode == 'auto' else batch_mode
        # Agent rows are env-major then team, exactly as ServerCollector.
        self.teams = tuple(int(t) for t in teams)
        self.T = len(self.teams)
        self.n_envs, self.out, self.episode_s = int(n), Path(out), float(episode_s)
        self.n = self.n_envs * self.T
        self.out.mkdir(parents=True, exist_ok=True)
        self.seed, self.start_near_wave = int(seed), bool(start_near_wave)
        self.sim = (SimConfig.training() if sim_config is None else sim_config).replace(
            step_ticks=int(step_ticks))
        if self.sim.vision is None or self.sim.lane_path is None:
            raise ValueError('farming requires map visibility and top-lane waves')
        self.episodes = [0] * self.n
        self._compiled = {}
        self._executions = {}
        # Large immutable map tables are runtime arguments, not duplicated
        # constants embedded into every idle/policy/warmup executable.
        self._model = (self.sim.params, self.sim.route_table, self.sim.terrain, self.sim.vision)
        self.rank_decisions = 0
        self.closed = False
        self.last_orders = None  # Privileged replay diagnostic, never an actor input.
        self.frame = make_lane_frame(TOP_OUTER_TURRET[Team.BLUE],
                                     TOP_OUTER_TURRET[Team.RED], (1131.8, 1426.3))
        self.frames = _lane_frames()
        self.noop = Orders(jnp.array([OrderKind.NOOP]*2, jnp.int8),
                           jnp.zeros(2, jnp.float32), jnp.zeros(2, jnp.float32), jnp.full(2, -1, jnp.int8),
                           clear_target=jnp.zeros(2, bool))
        self._noop_orders = self._batch_orders(self.noop)
        def decode_one(state, agent_actions, model):
            # agent_actions: (T, 3). A team without a policy row gets NOOP.
            sim = self._config(model)
            rows = {t: agent_actions[k] for k, t in enumerate(self.teams)}
            noop = jnp.asarray([BUTTON_INDEX['noop'], 0, 0], jnp.int32)
            per_team = [rows.get(t, noop) for t in (0, 1)]
            actions = tuple(jnp.stack([per_team[0][i], per_team[1][i]]) for i in range(3))
            return orders_from(actions, state, None, self.frame,
                               snap_moves=False, params=sim.params, vision=sim.vision)
        self._decode = jax.jit(jax.vmap(decode_one, in_axes=(0, 0, None)))
        def step_one(state, orders, enabled, model):
            sim = self._config(model)
            return jax.lax.cond(enabled, lambda q: env_step(q, orders, sim),
                                lambda q: q, state)
        if self.batch_mode == 'map':
            # CPU scalar control flow must survive across environments. Outer
            # vmap changes conditional branches into selection and executes
            # expensive inactive minion/terrain branches on ordinary ticks.
            self._step = jax.jit(lambda states, orders, enabled, model: jax.lax.map(
                lambda row: step_one(row[0], row[1], row[2], model),
                (states, orders, enabled)))
        else:
            self._step = jax.jit(jax.vmap(step_one, in_axes=(0, 0, 0, None)))
        def observe_all(s, model):
            per_team = [build_observation(s, t, self.frames[t], params=model[0],
                                          horizon_s=self.episode_s, vision=model[3])
                        for t in self.teams]
            return jax.tree.map(lambda *x: jnp.stack(x, axis=0), *per_team)   # (T, ...)
        self._observe = jax.jit(jax.vmap(observe_all, in_axes=(0, None)))
        def stats_all(s):
            pot = -lane_corridor_distance(s.x[:2], s.y[:2]) / 10000.
            return jnp.stack([jnp.stack([s.cs[t].astype(jnp.float32), s.alive[t].astype(jnp.float32),
                                         pot[t], s.xp[t].astype(jnp.float32)]) for t in self.teams])   # (T, 4)
        self._stats = jax.jit(jax.vmap(stats_all))
        self.states = jax.tree.map(lambda *a: jnp.stack(a),
                                   *(self._fresh(i) for i in range(self.n_envs)))
        self._write_metadata()
        self._initialize(np.ones(self.n_envs, bool))

    def _batch_orders(self, orders):
        return jax.tree.map(lambda a: jnp.broadcast_to(a, (self.n_envs,) + a.shape), orders)

    def _step_states(self, states, orders, enabled):
        # The simulation uses explicit float32 even when reference tests enable
        # x64 globally. Setup and decoded orders must share one executable ABI.
        orders = orders._replace(x=orders.x.astype(states.x.dtype),
                                  y=orders.y.astype(states.y.dtype))
        return self._call('step', self._step, states, orders, jnp.asarray(enabled), self._model)

    def _config(self, model):
        params, routes, terrain, vision = model
        return self.sim.replace(params=params, route_table=routes,
                                terrain=terrain, vision=vision)

    def _event(self, **row):
        with (self.out / 'timing.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)

    def _call(self, name, fn, *args):
        if name not in self._compiled:
            self._event(kernel=name, phase='lower-start')
            started = time.monotonic()
            lowered = fn.lower(*args)
            self._event(kernel=name, phase='lower', wall_s=time.monotonic()-started)
            self._event(kernel=name, phase='compile-start')
            started = time.monotonic()
            self._compiled[name] = lowered.compile()
            self._event(kernel=name, phase='compile', wall_s=time.monotonic()-started)
        started = time.monotonic()
        out = self._compiled[name](*args)
        count = self._executions.get(name, 0)
        if count < 3 or count in (10, 20, 50, 100, 1000):
            jax.block_until_ready(out)
            self._event(kernel=name, phase='execute', invocation=count,
                        wall_s=time.monotonic()-started)
        self._executions[name] = count + 1
        return out

    def _fresh(self, i):
        # Separate deterministic streams; episode reset preserves all base stats.
        return init_lane(seed=self.seed + i + self.n_envs * self.episodes[i * self.T])

    def _rank_time(self, previous_ranks):
        """Account for rank-command time only in environments with pending ranks.

        JAX has already applied ranks in the preceding tick. This is timing
        accounting, not exact source spell-rank event ordering.
        """
        ts = list(self.teams)
        pending = np.maximum(np.asarray(self.states.spell_level[:, ts]) - previous_ranks, 0).sum(axis=(1, 2))
        for _ in range(18):
            if not np.any(pending):
                return
            before = np.asarray(self.states.spell_level[:, ts])
            self.states = self._step_states(self.states, self._noop_orders, pending > 0)
            self.rank_decisions += 1
            gained = np.maximum(np.asarray(self.states.spell_level[:, ts]) - before, 0).sum(axis=(1, 2))
            pending = np.maximum(pending - 1, 0) + gained
        raise RuntimeError('automatic skill progression failed to settle')

    def _initialize(self, mask):
        # Source initialization spends a rank-up command before first observe.
        # JAX's first NOOP decision installs its native level-one ranks.
        self.states = self._step_states(self.states, self._noop_orders, mask)
        self.rank_decisions += 1
        if self.start_near_wave:
            self._prepare(mask)

    def _prepare(self, mask):
        red = 1 in self.teams
        move = Orders(jnp.array([OrderKind.MOVE, OrderKind.MOVE if red else OrderKind.NOOP], jnp.int8),
                      jnp.array([WAVE_START_POS[0], RED_WAVE_START_POS[0] if red else 0.], jnp.float32),
                      jnp.array([WAVE_START_POS[1], RED_WAVE_START_POS[1] if red else 0.], jnp.float32),
                      jnp.full(2, -1, jnp.int8), clear_target=jnp.zeros(2, bool))
        # Reuse the policy dynamics executable during setup. A separate JIT
        # around this loop would compile the full routing graph a second time.
        self.states = self._step_states(self.states, self._batch_orders(move), mask)
        max_decisions = int(np.ceil(WAVE_START_MS / (self.sim.step_ticks * self.sim.delta_ms))) + 1
        started = time.monotonic()
        for count in range(max_decisions):
            ts = list(self.teams)
            failed = mask & ((~np.asarray(self.states.alive[:, ts])).any(axis=1)
                            | (np.asarray(self.states.deaths[:, ts]) > 0).any(axis=1)
                            | (np.asarray(self.states.cs[:, ts]) != 0).any(axis=1))
            if failed.any():
                break
            active = mask & (np.asarray(self.states.t_ms) < WAVE_START_MS)
            if not active.any():
                break
            self.states = self._step_states(self.states, self._noop_orders, active)
        self._event(kernel='prepare', phase='execute', decisions=count+1,
                    wall_s=time.monotonic()-started)
        for i in np.flatnonzero(mask):
            state = jax.tree.map(lambda a: a[i], self.states)
            with (self.out / 'setup.jsonl').open('a') as f:
                f.write(json.dumps({'env': int(i), 'episode': self.episodes[i * self.T],
                    't_ms': float(state.t_ms),
                    'champions': [{'team': t, 'x': float(state.x[t]), 'y': float(state.y[t]),
                                   'cs': int(state.cs[t]), 'deaths': int(state.deaths[t]),
                                   'alive': bool(state.alive[t]), 'goal': WAVE_START[t][0]}
                                  for t in self.teams],
                    'setup': 'one routed move then idle; no teleports'}) + '\n')
            for t in self.teams:
                validate_wave_start(state, t)

    def _write_metadata(self):
        data = {'environment': 'jax-experimental-farming', 'seed': self.seed,
                'n': self.n, 'envs': self.n_envs, 'teams': list(self.teams),
                'episode_s': self.episode_s, 'start_near_wave': self.start_near_wave,
                'platform': self.platform, 'batch_mode': self.batch_mode,
                'ground_click_normalization': GROUND_CLICK_NORMALIZATION,
                'sim': self.sim.describe(), 'sim_fingerprint': self.sim.fingerprint(),
                'compilation_cache': jax.config.jax_compilation_cache_dir,
                'action_interface': 'screen-click-v2', 'observation_interface': 'viewport-structured-v3',
                'opponent': 'mirror self-play' if 1 in self.teams else 'idle red fountain', 'items': 'none',
                'rank_timing': 'native within-tick ranks; extra per-environment NOOP decisions after policy level gains',
                'warmup_rank_timing': 'native automatic ranks; no extra decision per warmup rank',
                'reset_timing': 'only reset environments advance during initial rank and warmup',
                'control': self.sim.route_table is None}
        (self.out / 'collector.json').write_text(json.dumps(data, indent=2) + '\n')

    def observe(self):
        self._check_open()
        obs = self._call('observe', self._observe, self.states, self._model)   # (n_envs, T, ...)
        obs = jax.tree.map(lambda x: x.reshape((self.n,) + x.shape[2:]), obs)
        stats = np.asarray(self._call('stats', self._stats, self.states)).reshape(self.n, 4)
        return obs, stats

    def spell_ranks(self):
        """Own HUD ranks per agent row for diagnostics; never an actor input."""
        self._check_open()
        return np.asarray(self.states.spell_level[:, list(self.teams)]).reshape(self.n, 4)

    def step(self, actions):
        self._check_open()
        actions = np.asarray(actions)
        if actions.shape != (self.n, 3) or not np.issubdtype(actions.dtype, np.integer):
            raise ValueError(f'actions must be integer ({self.n}, 3) button/x/y')
        limits = np.array([len(BUTTONS), len(SCREEN_X_VALUES), len(SCREEN_Y_VALUES)])
        if np.any(actions < 0) or np.any(actions >= limits):
            raise ValueError('action outside screen-click-v2 grid')
        ranks = np.asarray(self.states.spell_level[:, list(self.teams)])
        orders = self._call('decode', self._decode, self.states,
                            jnp.asarray(actions).reshape(self.n_envs, self.T, 3), self._model)
        self.last_orders = orders
        self.states = self._step_states(self.states, orders, np.ones(self.n_envs, bool))
        self._rank_time(ranks)
        return np.repeat(np.asarray(self.states.t_ms >= self.episode_s * 1000.), self.T)

    def restart_done(self, done):
        self._check_open()
        done = np.asarray(done, bool)
        if done.shape != (self.n,):
            raise ValueError('done must have one entry per agent row')
        if not done.any():
            return
        envs = np.zeros(self.n_envs, bool)
        for a in np.flatnonzero(done):
            envs[int(a) // self.T] = True
        for i in np.flatnonzero(envs):
            for k in range(self.T):
                self.episodes[i * self.T + k] += 1
            fresh = self._fresh(i)
            self.states = jax.tree.map(lambda a, b: a.at[i].set(b), self.states, fresh)
        self._initialize(envs)

    def _check_open(self):
        if self.closed:
            raise RuntimeError('collector is closed')

    def close(self):
        self.closed = True


def main():
    """Bounded task smoke, never a training or convergence claim."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--envs', type=int, default=1)
    parser.add_argument('--episode-s', type=float, default=.05)
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--step-ticks', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--start-near-wave', action='store_true')
    parser.add_argument('--no-route-table', action='store_true', help='Explicit diagnostic control')
    parser.add_argument('--batch-mode', choices=['auto', 'map', 'vmap'], default='auto')
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error('--steps must be positive')
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    from .run_manifest import file_sha256, git_environment
    names = subprocess.check_output(['git', 'ls-files', '-z', '--cached', '--others',
        '--exclude-standard'], cwd=root, env=git_environment(root)).decode().split('\0')
    with tarfile.open(args.out / 'source.tar.gz', 'w:gz') as archive:
        for name in sorted(set(names)):
            p = root / name
            if name and p.is_file() and p.suffix in ('.py', '.json', '.md', '.toml', '.patch', '.sh'):
                archive.add(p, arcname=name)
    (args.out / 'command.txt').write_text(shlex.join([sys.executable, '-m',
        'lanerl_jax.train.jax_farm', *sys.argv[1:]]) + '\n')
    result = {'seed': args.seed, 'status': 'running',
              'source_sha256': file_sha256(args.out / 'source.tar.gz')}
    started = time.monotonic()
    collector = None
    try:
        sim = SimConfig.training(route_artifact=None) if args.no_route_table else None
        collector = JaxFarmCollector(args.envs, args.out, args.episode_s,
            args.start_near_wave, args.step_ticks, seed=args.seed, sim_config=sim, batch_mode=args.batch_mode)
        actions = np.zeros((args.envs, 3), np.int32); actions[:, 0] = BUTTON_INDEX['noop']
        discount = PPOConfig(decision_hz=60. / args.step_ticks).gamma
        for _ in range(args.steps):
            obs, before = collector.observe()
            done = collector.step(actions)
            _, after = collector.observe()
            reward, _ = farm_reward(before[:, 0], after[:, 0], before[:, 1].astype(bool),
                after[:, 1].astype(bool), before[:, 2], after[:, 2], done, discount)
            if not all(np.isfinite(np.asarray(a)).all() for a in jax.tree.leaves(obs)):
                raise RuntimeError('nonfinite observation')
            if not np.isfinite(np.asarray(reward)).all():
                raise RuntimeError('nonfinite reward')
            collector.restart_done(done)
        result.update(status='complete', episodes=collector.episodes,
                      rank_decisions=collector.rank_decisions)
    except BaseException as exc:
        result.update(status='failed', error=repr(exc))
        raise
    finally:
        if collector is not None:
            collector.close()
        result['wall_s'] = time.monotonic() - started
        (args.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
