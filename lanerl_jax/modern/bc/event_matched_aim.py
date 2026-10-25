"""Matched-event direction agreement and observed-opponent aiming diagnostics."""
import argparse
import json
from pathlib import Path

import numpy as np

from ..screen import screen_to_lane
from .event_metrics import valid_segments
from .events import BUTTON_INDEX as B, event_matches


def direction_error(predicted, received):
    return np.abs((np.asarray(predicted)-received+np.pi) % (2*np.pi)-np.pi)*180/np.pi


def matched_indices(t, label, emitted, valid, button):
    pairs = []
    for lo, hi in valid_segments(valid):
        p = np.flatnonzero(emitted[lo:hi] == button)+lo
        q = np.flatnonzero(label[lo:hi] == button)+lo
        pairs.extend((int(p[i]), int(q[j])) for i, j in event_matches(t[p], t[q])['pairs'])
    return np.asarray(pairs, np.int64).reshape(-1, 2)


def grid_direction(cell):
    ds, dn = screen_to_lane((cell % 96+.5)/96, (cell // 96+.5)/54)
    return np.arctan2(np.asarray(dn), np.asarray(ds))


def error_summary(values):
    a = np.asarray(values)
    return dict(count=len(a), median_degrees=float(np.median(a)) if len(a) else None,
                within10=float((a <= 10).mean()) if len(a) else None)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--grid-predictions', type=Path, required=True)
    p.add_argument('--angle-predictions', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    paths = dict(grid=a.grid_predictions, angle=a.angle_predictions)
    names = sorted(x.name for x in a.grid_predictions.glob('*.npz'))
    assert names and names == sorted(x.name for x in a.angle_predictions.glob('*.npz'))
    report = {arm: {b: dict(received=0, predicted=0, matched=0, qualified_received=0,
        matched_errors=[], teacher_visible_errors=[], opponent_visible_errors=[])
        for b in ('q', 'e')} for arm in paths}
    for name in names:
        with np.load(a.dataset/'games'/name) as z:
            t, label = z['t'], z['button']
            valid = z['valid'].copy()
            valid[:32] = False
            eligible = valid & z['aim_valid'] & np.isfinite(z['angle'])
            truth = z['angle']
            xy = z['entities'][:, 0, 1:3].astype(np.float64)
            visible = ~z['pad'][:, 0] & np.isfinite(xy).all(-1)
            opponent = np.arctan2(xy[:, 1], xy[:, 0])
        for arm, path in paths.items():
            with np.load(path/name) as z:
                np.testing.assert_array_equal(t, z['t'])
                np.testing.assert_array_equal(label, z['label_button'])
                emitted = z['button']
                if arm == 'angle':
                    actual = (z['actual_angle']+.5)*2*np.pi/72-np.pi
                    teacher = (z['teacher_angle']+.5)*2*np.pi/72-np.pi
                else:
                    actual, teacher = grid_direction(z['actual_grid']), grid_direction(z['teacher_grid'])
            for button in ('q', 'e'):
                b, result = B[button], report[arm][button]
                result['received'] += int((valid & (label == b)).sum())
                result['predicted'] += int((valid & (emitted == b)).sum())
                target = eligible & (label == b)
                result['qualified_received'] += int(target.sum())
                pair = matched_indices(t, label, emitted, valid, b)
                result['matched'] += len(pair)
                pair = pair[eligible[pair[:, 1]]]
                result['matched_errors'].extend(direction_error(actual[pair[:, 0]], truth[pair[:, 1]]).tolist())
                mask = target & visible
                result['teacher_visible_errors'].extend(direction_error(teacher[mask], truth[mask]).tolist())
                result['opponent_visible_errors'].extend(direction_error(opponent[mask], truth[mask]).tolist())
    for arm in report.values():
        for result in arm.values():
            errors = result.pop('matched_errors')
            result['matched_aim'] = error_summary(errors)
            result['joint_known_target_recall'] = sum(e <= 10 for e in errors)/max(result['qualified_received'], 1)
            result['event_f1'] = 2*result['matched']/max(result['received']+result['predicted'], 1)
            for prefix in ('teacher_visible', 'opponent_visible'):
                result[prefix] = error_summary(result.pop(prefix+'_errors'))
    output = dict(passed=True, perspectives=len(names), arms=report,
        limits='one sampled seed; matched-angle agreement to human direction within0.5s, not hit rate; visible-opponent heuristic only a diagnostic, never supervision')
    a.output.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2), flush=True)


if __name__ == '__main__':
    main()
