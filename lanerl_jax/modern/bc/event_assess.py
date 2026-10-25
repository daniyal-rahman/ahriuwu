"""Assess a saved event learner without retraining or tuning on scored matches."""
import argparse
import json
from pathlib import Path

from flax.serialization import msgpack_restore
import jax

from .event_batches import load_event_games, ahri_angle_targets
from .event_eval import evaluate_games
from .event_policy import EventPolicy, EventPolicyConfig


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--split', choices=('train', 'calibration', 'val'), default='val')
    p.add_argument('--sample-seed', type=int)
    p.add_argument('--limit', type=int, default=0)
    a = p.parse_args()
    if jax.default_backend() != 'gpu':
        raise RuntimeError('assessment requires a Slurm GPU allocation')
    saved = msgpack_restore(a.checkpoint.read_bytes())
    contract = json.loads(saved['contract_json'])
    policy = EventPolicy(EventPolicyConfig(**contract['config']))
    games = load_event_games(a.dataset, split=a.split, champion=contract['champion'])
    if contract.get('ahri_angle'):
        ahri_angle_targets(games)
    games = sorted(games, key=lambda g: g['row']['base'])
    if a.limit:
        games = games[:a.limit]
    a.output.mkdir(parents=True, exist_ok=False)
    reports = evaluate_games(jax.device_put(saved['params']), policy, games, seed=a.sample_seed,
                             prediction_dir=a.output/'predictions')
    (a.output/'validation.json').write_text(json.dumps(reports, indent=2))
    (a.output/'results.json').write_text(json.dumps(dict(passed=True, split=a.split,
        checkpoint=str(a.checkpoint), step=saved['step'], sample_seed=a.sample_seed,
        perspectives=len(reports), emitter='greedy' if a.sample_seed is None else 'categorical-temperature1',
        limits='offline replay-conditioned history, not closed-loop play; no calibration fitted here'), indent=2))


if __name__ == '__main__':
    main()
