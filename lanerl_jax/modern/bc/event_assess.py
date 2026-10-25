"""Assess a saved event learner without retraining or tuning on scored matches."""
import argparse
import hashlib
import json
from pathlib import Path

from flax.serialization import msgpack_restore
import jax
import numpy as np

from .event_batches import load_event_games, ahri_angle_targets
from .event_eval import evaluate_games
from .event_policy import EventPolicy, EventPolicyConfig


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--split', choices=('train', 'calibration', 'val'), default='val')
    p.add_argument('--calibration', type=Path)
    p.add_argument('--sample-seed', type=int)
    p.add_argument('--reset-every', type=int, default=0)
    p.add_argument('--limit', type=int, default=0)
    a = p.parse_args()
    if a.reset_every < 0 or a.reset_every % 128:
        p.error('reset-every must be zero or a positive multiple of128')
    if jax.default_backend() != 'gpu':
        raise RuntimeError('assessment requires a Slurm GPU allocation')
    checkpoint_bytes = a.checkpoint.read_bytes()
    saved = msgpack_restore(checkpoint_bytes)
    contract = json.loads(saved['contract_json'])
    policy = EventPolicy(EventPolicyConfig(**contract['config']))
    games = load_event_games(a.dataset, split=a.split, champion=contract['champion'])
    if contract.get('ahri_angle'):
        ahri_angle_targets(games)
    games = sorted(games, key=lambda g: g['row']['base'])
    if a.limit:
        games = games[:a.limit]
    bias, temperature = None, 1.
    if a.calibration:
        calibration = json.loads(a.calibration.read_text())
        if not calibration['passed'] or calibration['checkpoint_sha256'] != hashlib.sha256(checkpoint_bytes).hexdigest():
            raise ValueError('calibration checkpoint fingerprint mismatch')
        if calibration['dataset_manifest_sha256'] != hashlib.sha256((a.dataset/'manifest.json').read_bytes()).hexdigest():
            raise ValueError('calibration dataset fingerprint mismatch')
        if a.split == 'calibration':
            raise ValueError('scored assessment must be separate from calibration')
        bias, temperature = np.asarray(calibration['bias'], np.float32), float(calibration['temperature'])
    a.output.mkdir(parents=True, exist_ok=False)
    reports = evaluate_games(jax.device_put(saved['params']), policy, games, seed=a.sample_seed,
                             prediction_dir=a.output/'predictions', bias=bias, temperature=temperature,
                             reset_every=a.reset_every or None)
    (a.output/'validation.json').write_text(json.dumps(reports, indent=2))
    (a.output/'results.json').write_text(json.dumps(dict(passed=True, split=a.split,
        checkpoint=str(a.checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint_bytes).hexdigest(),
        dataset_manifest_sha256=hashlib.sha256((a.dataset/'manifest.json').read_bytes()).hexdigest(),
        model_contract=contract, step=saved['step'], sample_seed=a.sample_seed, temperature=temperature,
        calibration_sha256=hashlib.sha256(a.calibration.read_bytes()).hexdigest() if a.calibration else None,
        perspectives=len(reports), reset_every=a.reset_every, emitter='greedy' if a.sample_seed is None else 'categorical',
        limits='offline replay-conditioned history, not closed-loop play; any calibration frozen on separate calibration matches'), indent=2))


if __name__ == '__main__':
    main()
