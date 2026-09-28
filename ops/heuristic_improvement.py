#!/usr/bin/env python3
"""TOOL: score E24's predeclared frozen E25/E26/E27 comparisons, read-only.

Run from the repo root under ops/login_capped.sh. Never launches jobs or
selects checkpoints. Missing, duplicate, incomplete, or mismatched cohorts
fail closed. One successful training seed is preliminary evidence only.
"""
import argparse
import json
from pathlib import Path
import numpy as np

ROLES = {'clone': 'E25_dagger_teacher_baseline',
         'teacher': 'E26_heuristic_teacher_baseline',
         'ppo': 'E27_reference_teacher_final'}
CONTRACT = ('envs', 'opponent', 'start_near_wave', 'start_jitter_s', 'step_ticks',
            'episode_s', 'reward', 'gold_scale', 'xp_scale', 'enemy_scale',
            'no_snap_clicks', 'unwalkable_click', 'deterministic')


def paired_interval(candidate, baseline):
    delta = np.asarray(candidate, float) - np.asarray(baseline, float)
    if delta.size < 16 or not np.isfinite(delta).all():
        raise ValueError('need at least 16 finite paired games')
    rng = np.random.default_rng(240027)
    draws = delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(1)
    return {'mean': float(delta.mean()), 'ci95': np.quantile(draws, [.025, .975]).tolist()}


def verdict(cohorts):
    keys = sorted(cohorts['teacher'])
    if len(keys) != 16 or any(set(rows) != set(keys) for rows in cohorts.values()):
        raise ValueError('require identical 16 (seed, env, team, episode) pairs across all three policies')
    result = {'scope': 'one training seed; preliminary frozen-policy evidence', 'comparisons': {}}
    passed = True
    for baseline in ('clone', 'teacher'):
        comparisons = {}
        for metric in ('reward_return', 'gold_diff', 'cs'):
            comparisons[metric] = paired_interval(
                [cohorts['ppo'][k][metric] for k in keys], [cohorts[baseline][k][metric] for k in keys])
        passed &= (comparisons['reward_return']['ci95'][0] > 0 and
                   comparisons['gold_diff']['ci95'][0] > 0 and comparisons['cs']['mean'] >= 0)
        result['comparisons'][baseline] = comparisons
    result['improved_beyond_teacher_and_clone'] = bool(passed)
    return result


def load_cohorts(root):
    cohorts, contract = {}, None
    clone_checkpoint = None
    for role, experiment in ROLES.items():
        rows = {}
        checkpoint_hashes = set()
        for seed in (2, 3):
            manifests = sorted((root / experiment / f'seed{seed}').glob('server-farm-*/manifest.json'))
            if len(manifests) != 1:
                raise ValueError(f'{experiment} seed {seed}: expected exactly one run, found {len(manifests)}')
            data = json.loads(manifests[0].read_text())
            config, results = data['config']['collector'], data['results']
            if results.get('status') != 'complete' or config['seed'] != seed or config['eval_episodes'] != 1:
                raise ValueError(f'incomplete or wrong evaluation: {manifests[0]}')
            if config['opponent'] != 'scripted' or config['reward'] != 'relative' or config.get('deterministic'):
                raise ValueError('requires sampled policy vs fixed teacher with relative reward')
            current = ({k: config.get(k) for k in CONTRACT}, data['vendor']['binary_sha256'], data['vendor']['config_sha256'])
            if contract is not None and current != contract:
                raise ValueError('evaluation settings/server binary differ between cohorts')
            contract = current
            if role == 'teacher':
                if config.get('scripted') != 'lasthit': raise ValueError('teacher must be scripted lasthit')
            else:
                if config.get('scripted'): raise ValueError('network cohort cannot use scripted learner')
                checkpoint = Path(str(results['evaluated_checkpoint']).replace('/mnt/nfs/', '/srv/nfs/', 1))
                checkpoint_hashes.add(results['checkpoint_sha256'])
                if role == 'clone':
                    clone_checkpoint = checkpoint.resolve()
                if role == 'ppo':
                    manifest = json.loads((checkpoint.parent / 'manifest.json').read_text())
                    matches = [c for c in manifest['checkpoints'] if c.get('file') == checkpoint.name]
                    if not matches or matches[0].get('update') != 400:
                        raise ValueError('PPO evaluation must use the fixed u400 checkpoint, not latest/best')
                    train_cfg = manifest['config']['collector']
                    init = Path(str(train_cfg['init_from']).replace('/mnt/nfs/', '/srv/nfs/', 1)).resolve()
                    if init != clone_checkpoint or train_cfg['opponent'] != 'scripted':
                        raise ValueError('PPO must start from the evaluated clone and train against the teacher')
                    if train_cfg['updates'] != 400 or train_cfg['seed'] != 0 or train_cfg['reward'] != 'relative':
                        raise ValueError('unexpected training budget/seed/reward')
            if len(results['episodes']) != 8: raise ValueError('need eight games per evaluation seed')
            for row in results['episodes']:
                key = (seed, row['env'], row['team'], row['episode'])
                if key in rows or row['team'] != row['env'] % 2: raise ValueError('duplicate or mislabeled episode')
                rows[key] = row
        if role != 'teacher' and len(checkpoint_hashes) != 1:
            raise ValueError('each network cohort must evaluate one frozen checkpoint')
        cohorts[role] = rows
    return cohorts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, default=Path('lanerl_jax/runs'))
    args = p.parse_args()
    try:
        result = verdict(load_cohorts(args.runs))
    except (ValueError, KeyError, OSError) as exc:
        p.exit(2, f'NOT SCORED: {exc}\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
