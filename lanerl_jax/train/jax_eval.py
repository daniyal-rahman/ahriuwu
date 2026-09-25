"""Frozen blue-only JAX farming evaluation, with optional diagnostic replay.

Uses the farming collector's setup, screen actions and rank timing. No updates,
resets, demonstrations or second actor. This is not evidence of server parity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace

import numpy as np


def run_episode(collector, choose, on_transition, max_decisions):
    """Stop at the first terminal transition; never reset into another episode."""
    for count in range(max_decisions):
        obs, _ = collector.observe()
        before = collector.states
        action = np.asarray(choose(obs), dtype=np.int32)
        done = np.asarray(collector.step(action), dtype=bool)
        on_transition(before, action, collector.last_orders)
        if done.shape != (1,):
            raise RuntimeError('frozen evaluator requires exactly one environment')
        if done[0]:
            return count + 1
    raise RuntimeError('collector did not terminate within the game-time decision bound')


def resolve_task(args, manifest):
    saved = manifest.get('config', {}).get('collector', {})
    seconds = args.seconds if args.seconds is not None else saved.get('episode_s', 600.)
    step_ticks = args.step_ticks if args.step_ticks is not None else saved.get('step_ticks', 2)
    near = args.start_near_wave if args.start_near_wave is not None else saved.get('start_near_wave', False)
    resolved = manifest.get('sim', {}).get('resolved', {})
    route = args.route_artifact or resolved.get('route_artifact') or saved.get('route_artifact')
    if not math.isfinite(seconds) or seconds <= 0 or step_ticks < 1 or int(step_ticks) != step_ticks:
        raise ValueError('positive finite seconds and positive integer step_ticks required')
    if near and seconds <= 120:
        raise ValueError('near-wave evaluation must end after 120 seconds')
    return dict(episode_s=float(seconds), step_ticks=int(step_ticks),
                start_near_wave=bool(near), route_artifact=route)


def evaluate(args):
    import jax
    import jax.numpy as jnp
    from flax.serialization import to_bytes
    from lanerl_rl.constants import BUTTON_INDEX, N_SCREEN_X, N_SCREEN_Y
    from ..parity.policy_driver import load_params
    from ..replay import state_snapshot, render_metadata, summarize
    from ..sim.config import SimConfig, DEFAULT_ROUTE_ARTIFACT
    from .actions import _screen_to_centred_lane
    from .jax_farm import JaxFarmCollector, GROUND_CLICK_NORMALIZATION
    from .run_manifest import file_sha256, git_environment, git_provenance
    from .trainer import _sample

    checkpoint = args.checkpoint.resolve()
    manifest_path = checkpoint.parent / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    task = resolve_task(args, manifest)
    route = Path(task.pop('route_artifact') or DEFAULT_ROUTE_ARTIFACT).resolve()
    if not math.isfinite(args.replay_hz) or not 0 < args.replay_hz <= 60:
        raise ValueError('replay-hz must be in (0, 60]')
    args.out.mkdir(parents=True, exist_ok=False)
    # Freeze the actual weights and manifest before loading; do not follow a
    # training process that might rewrite the supplied checkpoint path.
    inputs = args.out / 'input'
    inputs.mkdir()
    shutil.copyfile(checkpoint, inputs / 'checkpoint.msgpack')
    shutil.copyfile(manifest_path, inputs / 'manifest.json')
    command = shlex.join([sys.executable, '-m', 'lanerl_jax.train.jax_eval', *sys.argv[1:]])
    (args.out / 'command.txt').write_text(command + '\n')
    root = Path(__file__).resolve().parents[2]
    source = git_provenance()
    names = subprocess.check_output(['git', 'ls-files', '-z', '--cached', '--others',
        '--exclude-standard'], cwd=root, env=git_environment(root)).decode().split('\0')
    with tarfile.open(args.out / 'source.tar.gz', 'w:gz') as archive:
        for name in sorted(set(names)):
            path = root / name
            if name and path.is_file() and path.suffix in {'.py', '.sh', '.html', '.toml', '.patch', '.md', '.json'}:
                archive.add(path, arcname=name)
    source['snapshot_sha256'] = file_sha256(args.out / 'source.tar.gz')
    result = dict(status='running', checkpoint=str(checkpoint),
        ground_click_normalization=GROUND_CLICK_NORMALIZATION,
        checkpoint_ground_click_normalization=manifest.get('config', {}).get(
            'ground_click_normalization', 'unspecified; inspect frozen source'),
        checkpoint_sha256=file_sha256(inputs / 'checkpoint.msgpack'),
        checkpoint_manifest_sha256=file_sha256(inputs / 'manifest.json'), source=source,
        seed=args.seed, environment='jax-experimental-farming', opponent='idle-fountain',
        task=task, route_artifact=str(route), decision_hz=60. / task['step_ticks'],
        action_interface='screen-click-v2', observation_interface='viewport-structured-v3',
        limitations=['JAX/source dynamics parity is not established; see fidelity ledger.',
                    'Native ranks plus collector NOOP rank timing; warmup is not replayed.',
                    'Replay is omniscient diagnostic state, never actor input.'])
    (args.out / 'evaluation.json').write_text(json.dumps(result, indent=2)+'\n')
    collector, logs = None, []
    started = time.monotonic()
    try:
        sim = SimConfig.training(route_artifact=route).replace(step_ticks=task['step_ticks'])
        if sim.route_table is None or sim.vision is None or sim.lane_path is None:
            raise RuntimeError('evaluation requires production routes, map visibility and waves')
        shutil.copyfile(route / 'manifest.json', args.out / 'route-manifest.json')
        result['simulation'] = dict(resolved=sim.describe(), fingerprint=sim.fingerprint(),
            route_manifest_sha256=file_sha256(args.out / 'route-manifest.json'))
        collector = JaxFarmCollector(1, args.out, seed=args.seed, sim_config=sim, **task)
        policy, params, _ = load_params(str(inputs / 'checkpoint.msgpack'))
        initial_parameter_hash = hashlib.sha256(to_bytes(params)).hexdigest()
        key = jax.random.key(args.seed)

        @jax.jit
        def sample(obs, rng):
            logits = policy.apply(params, obs.entities, obs.entity_pad_mask, obs.self_vec, obs.global_vec)
            action, _, _ = _sample(logits, rng, ~obs.entity_pad_mask)
            finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(x))
                                        for x in jax.tree.leaves(logits)]))
            return jnp.stack(action, axis=-1), finite

        def choose(obs):
            nonlocal key
            if not all(np.isfinite(np.asarray(a)).all() for a in jax.tree.leaves(obs)):
                raise RuntimeError('nonfinite actor observation')
            key, action_key = jax.random.split(key)
            action, finite = sample(obs, action_key)
            if not bool(finite):
                raise RuntimeError('nonfinite frozen policy output')
            return np.asarray(action)

        @jax.jit
        def snapshot(state, orders, blue):
            action = tuple(jnp.stack([blue[i], jnp.int32(
                BUTTON_INDEX['noop'] if i == 0 else 0)]) for i in range(3))
            # Reuse projection alone; diagnostic raw cursors do not rerun hit
            # tests or select targets. Resolved orders come from collector.step.
            ds, dn = _screen_to_centred_lane((action[1]+.5)/N_SCREEN_X,
                                             (action[2]+.5)/N_SCREEN_Y)
            side = jnp.where(state.team[:2] == 0, 1., -1.)
            axis, normal = collector.frame.axis, collector.frame.normal
            cursor = SimpleNamespace(x=state.x[:2]+side*ds*axis[0]+dn*normal[0],
                                     y=state.y[:2]+side*ds*axis[1]+dn*normal[1])
            return state_snapshot(state, orders, action, cursor)

        next_sample_ms = float(collector.states.t_ms[0])
        result['policy_start_ms'] = next_sample_ms
        with (args.out / 'actions.jsonl').open('w') as action_log:
            def record(before, action, orders):
                nonlocal next_sample_ms
                t_ms = float(before.t_ms[0])
                action_log.write(json.dumps(dict(t_ms=t_ms, blue=action[0].tolist()))+'\n')
                if args.replay and t_ms >= next_sample_ms:
                    state = jax.tree.map(lambda a: a[0], before)
                    order = jax.tree.map(lambda a: a[0], orders)
                    logs.append(jax.tree.map(np.asarray, snapshot(state, order, action[0])))
                    next_sample_ms = t_ms + 1000. / args.replay_hz - 1.
            bound = math.ceil(task['episode_s']*1000. / (sim.delta_ms*sim.step_ticks)) + 1
            decisions = run_episode(collector, choose, record, bound)
        final = jax.tree.map(lambda a: a[0], collector.states)
        final_ms = float(final.t_ms)
        if final_ms < task['episode_s']*1000:
            raise RuntimeError('collector reported termination before evaluation horizon')
        if hashlib.sha256(to_bytes(params)).hexdigest() != initial_parameter_hash:
            raise RuntimeError('frozen evaluation modified policy parameters')
        result.update(status='complete', policy_decisions=decisions, final_t_ms=final_ms,
            requested_seconds=task['episode_s'], cs={str(100+100*i):int(final.cs[i]) for i in range(2)},
            deaths={str(100+100*i):int(final.deaths[i]) for i in range(2)},
            parameters_unchanged=True, parameter_sha256=initial_parameter_hash,
            rank_decisions=collector.rank_decisions, episodes=collector.episodes)
        if args.replay:
            # Retain terminal totals without taking an extra action or resetting.
            terminal_action = jnp.array([BUTTON_INDEX['noop'], 0, 0], jnp.int32)
            logs.append(jax.tree.map(np.asarray, snapshot(final, collector.noop, terminal_action)))
            data = {k:np.stack([row[k] for row in logs]) for k in logs[0]}
            terrain = sim.terrain
            label = args.label or f'Frozen JAX farming seed {args.seed}'
            if task['start_near_wave']:
                label += ' (setup omitted)'
            metadata = dict(label=label,
                controller='frozen blue policy; red idle', environment='jax-experimental-farming',
                **render_metadata(), **{k:result[k] for k in ('checkpoint', 'checkpoint_sha256', 'seed',
                    'simulation', 'source', 'action_interface', 'observation_interface',
                    'ground_click_normalization', 'checkpoint_ground_click_normalization')},
                view='omniscient diagnostic; not actor input', hz=args.replay_hz,
                seconds=task['episode_s'], start_seconds=result['policy_start_ms']/1000.,
                action_alignment='pre-action samples; final frame is terminal state with unsent NOOP placeholder',
                terrain=dict(min_x=float(terrain.min_x), min_y=float(terrain.min_y), cell_size=float(terrain.cell_size)))
            metadata['summary'] = summarize(data)
            np.savez_compressed(args.out/'trace.npz', **data, walkable=np.asarray(terrain.walkable),
                                metadata=np.asarray(json.dumps(metadata)))
            (args.out/'trace.json').write_text(json.dumps(metadata, indent=2)+'\n')
            result['replay'] = 'trace.npz'
    except BaseException as exc:
        result.update(status='failed', error=repr(exc))
        raise
    finally:
        if collector is not None:
            collector.close()
        result['wall_s'] = time.monotonic()-started
        (args.out/'evaluation.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--seconds', type=float, help='Default: checkpoint collector episode_s')
    p.add_argument('--step-ticks', type=int, help='Default: checkpoint collector step_ticks')
    p.add_argument('--start-near-wave', action=argparse.BooleanOptionalAction, default=None)
    p.add_argument('--route-artifact', type=Path)
    p.add_argument('--replay', action='store_true')
    p.add_argument('--replay-hz', type=float, default=10.)
    p.add_argument('--label')
    evaluate(p.parse_args())


if __name__ == '__main__':
    main()
