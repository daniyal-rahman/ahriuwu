"""Aggregate replay evaluation and bootstrap uncertainty by held-out match."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from .events import BUTTONS


def summarize(reports):
    result = {}
    for button in BUTTONS[1:]:
        rows = [r['metrics']['classes'][button] for r in reports]
        n = sum(r['received'] for r in rows)
        p = sum(r['predicted'] for r in rows)
        tp = sum(r['true_positive'] for r in rows)
        expected = sum(r['expected_count'] for r in rows)
        values = dict(received=n, predicted=p, true_positive=tp, f1=2*tp/(n+p) if n+p else None,
                      precision=tp/p if p else 0., recall=tp/n if n else None,
                      expected_ratio=expected/n if n else None, emitted_ratio=p/n if n else None)
        for name in ('exact_frame_ap', 'near_frame_ap'):
            supported = [r[name] for r in rows if r[name] is not None]
            values['macro_'+name] = float(np.mean(supported)) if supported else None
        for prefix in ('actual', 'teacher'):
            for mode in ('point', 'unit', 'angle'):
                target_rows = [r['metrics'].get('targets_by_button', {}).get(prefix+'_'+mode, {}).get(button)
                               for r in reports]
                target_rows = [r for r in target_rows if r is not None]
                for metric in ('accuracy', 'within2', 'joint_success'):
                    weighted = [(r[metric], r['eligible'] if metric == 'joint_success' else r['count'])
                                for r in target_rows if metric in r]
                    denom = sum(w for _, w in weighted)
                    if denom:
                        values[prefix+'_'+mode+'_'+metric] = sum(v*w for v, w in weighted)/denom
                for metric in ('median_cells', 'median_degrees'):
                    medians = [r[metric] for r in target_rows if metric in r]
                    if medians:
                        values[prefix+'_'+mode+'_macro_'+metric] = float(np.mean(medians))
        result[button] = values
    return result


def report(reports, *, samples=500, seed=0, reference=None):
    grouped = defaultdict(list)
    for row in reports:
        grouped[row['base']].append(row)
    bases = sorted(grouped)
    if not bases:
        raise ValueError('no held-out matches')
    reference_groups = defaultdict(list)
    if reference is not None:
        for row in reference:
            reference_groups[row['base']].append(row)
        if set(reference_groups) != set(grouped):
            raise ValueError('paired comparison requires exactly the same held-out matches')
    point = summarize(reports)
    reference_point = summarize(reference) if reference is not None else None
    draws = defaultdict(list)
    rng = np.random.default_rng(seed)
    for _ in range(samples):
        chosen = rng.integers(len(bases), size=len(bases))
        selected = [r for i in chosen for r in grouped[bases[i]]]
        estimate = summarize(selected)
        control = summarize([r for i in chosen for r in reference_groups[bases[i]]]) if reference is not None else None
        for button, metrics in estimate.items():
            for metric, value in metrics.items():
                if metric in ('received', 'predicted', 'true_positive') or value is None:
                    continue
                if control is not None:
                    other = control[button].get(metric)
                    if other is None:
                        continue
                    value -= other
                draws[(button, metric)].append(value)
    intervals = {button: {} for button in point}
    for (button, metric), values in draws.items():
        intervals[button][metric] = list(map(float, np.quantile(values, [.025, .975])))
    return dict(matches=len(bases), perspectives=len(reports), metrics=point,
                reference=reference_point, confidence95=intervals, bootstrap_samples=samples,
                limits='match bootstrap; paired delta intervals when reference supplied; macro frame AP and mean per-match median target errors are not pooled AP/medians')


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('validation', type=Path)
    p.add_argument('--reference', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    reference = json.loads(a.reference.read_text()) if a.reference else None
    result = report(json.loads(a.validation.read_text()), reference=reference)
    a.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result['metrics'], indent=2))


if __name__ == '__main__':
    main()
