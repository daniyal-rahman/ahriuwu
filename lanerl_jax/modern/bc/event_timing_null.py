"""Rate-preserving within-valid-segment permutation control for tolerant event F1."""
import argparse
import json
from pathlib import Path

import numpy as np

from .event_metrics import valid_segments
from .events import BUTTONS, event_matches


def counts(t, received, predicted):
    result = np.zeros((len(BUTTONS), 3), np.int64)
    for b in range(1, len(BUTTONS)):
        truth, emitted = t[received == b], t[predicted == b]
        result[b] = (len(truth), len(emitted), event_matches(emitted, truth)['true_positive'])
    return result


def f1(values):
    return 2*values[:, 2]/np.maximum(values[:, 0]+values[:, 1], 1)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--samples', type=int, default=32)
    a = p.parse_args()
    rng = np.random.default_rng(0)
    cases = []
    observed = np.zeros((len(BUTTONS), 3), np.int64)
    for path in sorted(a.predictions.glob('*.npz')):
        with np.load(path) as z:
            valid = z['valid'].copy()
            valid[:32] = False
            t, label, emitted = z['t'], z['label_button'], z['button']
            for lo, hi in valid_segments(valid):
                case = (t[lo:hi], label[lo:hi], emitted[lo:hi])
                cases.append(case)
                observed += counts(*case)
    if not cases or a.samples < 1:
        raise ValueError('need predictions and a positive null budget')
    null = []
    for _ in range(a.samples):
        total = np.zeros_like(observed)
        for t, truth, emitted in cases:
            total += counts(t, truth, rng.permutation(emitted))
        # The null preserves every button count in every valid segment.
        np.testing.assert_array_equal(total[:, :2], observed[:, :2])
        null.append(f1(total))
    null = np.asarray(null)
    report = dict(passed=True, samples=a.samples, buttons={})
    for b, name in enumerate(BUTTONS[1:], 1):
        report['buttons'][name] = dict(received=int(observed[b, 0]), predicted=int(observed[b, 1]),
            observed_f1=float(f1(observed)[b]), null_median=float(np.median(null[:, b])),
            null95=list(map(float, np.quantile(null[:, b], [.025, .975]))))
    report['limits'] = 'fixed policy/seed, timing only; permutation preserves segment counts but destroys autocorrelation; no matched-target or closed-loop claim'
    a.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
