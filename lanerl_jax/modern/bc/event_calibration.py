"""Freeze temperature and rate biases using only the separate calibration matches."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

from .event_metrics import valid_segments
from .event_timing_null import counts, f1
from .events import CASTS


def calibrated_probabilities(probabilities, temperature, bias):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be finite and positive')
    return softmax(np.log(np.maximum(probabilities, 1e-30))/temperature + bias, axis=-1)


def fit_rate_bias(probabilities, labels, temperature):
    probabilities, labels = np.asarray(probabilities, np.float64), np.asarray(labels)
    if probabilities.ndim != 2 or labels.shape != probabilities.shape[:1] or not len(labels):
        raise ValueError('calibration needs aligned nonempty probabilities and labels')
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or not np.allclose(probabilities.sum(-1), 1.):
        raise ValueError('invalid calibration probabilities')
    if not np.issubdtype(labels.dtype, np.integer) or (labels < 0).any() or (labels >= probabilities.shape[1]).any():
        raise ValueError('invalid calibration buttons')
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('invalid temperature')
    frequency = np.bincount(labels, minlength=probabilities.shape[1])/len(labels)
    active = np.flatnonzero(frequency > 0)
    free = active[1:]
    bias = np.full(probabilities.shape[1], -1e4)
    bias[active] = 0.
    score = np.log(np.maximum(probabilities, 1e-30))/temperature

    def objective(v):
        beta = bias.copy()
        beta[free] = v
        logits = score+beta
        normalizer = logsumexp(logits, axis=-1)
        loss = float(np.mean(normalizer-logits[np.arange(len(labels)), labels]))
        grad = (np.exp(logits-normalizer[:, None]).mean(0)-frequency)[free]
        return loss, grad

    if len(free):
        result = minimize(objective, np.zeros(len(free)), jac=True, method='L-BFGS-B',
                          bounds=[(-40., 40.)]*len(free), options=dict(maxiter=300, ftol=1e-12, gtol=1e-9))
        bias[free] = result.x
    fitted = calibrated_probabilities(probabilities, temperature, bias).mean(0)
    if np.max(np.abs(fitted[active]/frequency[active]-1)) > .01:
        raise ValueError('rate-bias fit did not meet1% calibration count tolerance')
    return bias


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--assessment', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    metadata = json.loads((a.assessment/'results.json').read_text())
    if not metadata['passed'] or metadata['split'] != 'calibration':
        raise ValueError('fit only an explicit completed calibration assessment')
    games = []
    for path in sorted((a.assessment/'predictions').glob('*.npz')):
        with np.load(path) as z:
            valid = z['valid'].copy()
            valid[:32] = False
            games.append(dict(game=path.stem, t=z['t'], label=z['label_button'],
                              probability=z['probabilities'], valid=valid))
    probability = np.concatenate([g['probability'][g['valid']] for g in games])
    labels = np.concatenate([g['label'][g['valid']] for g in games])
    trials = []
    # Predeclared choices; select by calibration event F1, never validation labels.
    for temperature in (1., .7, .5):
        bias = fit_rate_bias(probability, labels, temperature)
        seed_scores, seed_counts = [], []
        for seed in (0, 1, 2):
            total = np.zeros((probability.shape[1], 3), np.int64)
            for game in games:
                game_seed = int(hashlib.sha256(game['game'].encode()).hexdigest()[:8], 16)
                rng = np.random.default_rng(np.random.SeedSequence([seed, game_seed]))
                calibrated = calibrated_probabilities(game['probability'], temperature, bias)
                draw = rng.random(len(game['t']))
                emitted = np.minimum((draw[:, None] > calibrated.cumsum(-1)).sum(-1), probability.shape[1]-1)
                for lo, hi in valid_segments(game['valid']):
                    total += counts(game['t'][lo:hi], game['label'][lo:hi], emitted[lo:hi])
            supported = [b for b in CASTS if total[b, 0] > 0]
            seed_scores.append(float(np.mean(f1(total)[supported])))
            seed_counts.append(total.tolist())
        trials.append(dict(temperature=temperature, bias=bias.tolist(), scores=seed_scores,
                           score=float(np.mean(seed_scores)), counts=seed_counts))
    selected = max(trials, key=lambda x: x['score'])
    result = dict(passed=True, temperature=selected['temperature'], bias=selected['bias'],
        checkpoint_sha256=metadata['checkpoint_sha256'],
        dataset_manifest_sha256=metadata['dataset_manifest_sha256'],
        calibration_games=[g['game'] for g in games], trials=trials,
        selection='mean QWER eventF1@0.5s across3 fixed calibration sampling seeds; biases fit expected counts',
        limits='no validation fitting; unsupported calibration classes disabled; no future input, cooldown mask or closed-loop claim')
    a.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
