"""Match-local event and target metrics; no thresholds fitted on scored games."""
import numpy as np

from .event_labels import UNIT_TARGET, POINT_TARGET, ANGLE_TARGET
from .events import BUTTONS, event_matches


def average_precision(truth, score):
    truth, score = np.asarray(truth, bool), np.asarray(score, np.float64)
    if truth.shape != score.shape or truth.ndim != 1 or not np.isfinite(score).all():
        raise ValueError('AP needs aligned finite vectors')
    if not truth.any():
        return None
    order = np.argsort(-score, kind='stable')
    y, s = truth[order], score[order]
    ends = np.r_[np.flatnonzero(s[1:] != s[:-1]), len(s) - 1]
    tp = np.cumsum(y)[ends]
    return float(np.sum(np.diff(np.r_[0, tp]) * tp / (ends + 1)) / truth.sum())


def valid_segments(valid):
    padded = np.r_[False, np.asarray(valid, bool), False].astype(np.int8)
    return list(zip(np.flatnonzero(np.diff(padded) == 1), np.flatnonzero(np.diff(padded) == -1)))


def sequence_metrics(arrays, predictions, *, burn=32, tolerance=.5):
    t, button = arrays['t'], arrays['button']
    valid = arrays['valid'].copy()
    valid[:burn] = False
    probabilities = np.asarray(predictions['probabilities'])
    emitted = np.asarray(predictions['button'])
    if probabilities.shape != (len(t), len(BUTTONS)) or emitted.shape != button.shape:
        raise ValueError('prediction shape mismatch')
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(-1), 1., atol=1e-5):
        raise ValueError('probabilities must be finite normalized distributions')
    segments = valid_segments(valid)
    minutes = float(valid.sum()) * .1 / 60.
    classes = {}
    for b, name in enumerate(BUTTONS):
        received = (button == b) & valid
        predicted = (emitted == b) & valid
        tp = 0
        for lo, hi in segments:
            matched = event_matches(t[lo:hi][predicted[lo:hi]], t[lo:hi][received[lo:hi]], tolerance)
            tp += matched['true_positive']
        n, m = int(received.sum()), int(predicted.sum())
        near = np.zeros(len(t), bool)
        radius = int(round(tolerance / .1))
        for lo, hi in segments:
            for i in np.flatnonzero(received[lo:hi]) + lo:
                near[max(lo, i-radius):min(hi, i+radius+1)] = True
        classes[name] = dict(received=n, predicted=m, true_positive=tp,
            precision=tp/m if m else 0., recall=tp/n if n else None,
            f1=2*tp/(n+m) if n+m else None,
            received_per_minute=n/minutes if minutes else None,
            predicted_per_minute=m/minutes if minutes else None,
            expected_count=float(probabilities[valid, b].sum()),
            calibration_ratio=float(probabilities[valid, b].sum())/n if n else None,
            exact_frame_ap=average_precision(received[valid], probabilities[valid, b]),
            near_frame_ap=average_precision(near[valid], probabilities[valid, b]))
    result = dict(valid_frames=int(valid.sum()), minutes=minutes, classes=classes,
                  button_accuracy=float(np.mean(emitted[valid] == button[valid])) if valid.any() else None)
    targets = {}
    for prefix in ('actual', 'teacher'):
        for name, mode in [('unit', UNIT_TARGET), ('point', POINT_TARGET), ('angle', ANGLE_TARGET)]:
            mask = valid & arrays['aim_valid'] & (arrays['aim_mode'] == mode)
            if prefix == 'actual':
                mask &= emitted == button
            count = int(mask.sum())
            item = dict(count=count)
            if count:
                if mode == UNIT_TARGET:
                    correct = predictions[prefix+'_pointer'][mask] == arrays['target_row'][mask]
                    item['accuracy'] = float(correct.mean())
                elif mode == POINT_TARGET:
                    cell = predictions[prefix+'_grid'][mask]
                    dx, dy = cell % 96 - arrays['screen_x'][mask], cell // 96 - arrays['screen_y'][mask]
                    error = np.hypot(dx, dy)
                    item.update(median_cells=float(np.median(error)), p90_cells=float(np.quantile(error, .9)),
                                within2=float(((np.abs(dx) <= 2) & (np.abs(dy) <= 2)).mean()))
                else:
                    angle = (predictions[prefix+'_angle'][mask] + .5) * (2*np.pi/72) - np.pi
                    error = np.abs((angle - arrays['angle'][mask] + np.pi) % (2*np.pi) - np.pi) * 180/np.pi
                    item.update(median_degrees=float(np.median(error)), p90_degrees=float(np.quantile(error, .9)))
            targets[prefix+'_'+name] = item
    result['targets'] = targets
    result['limits'] = 'replay-conditioned observations; actual target metrics conditional on exact correct button; teacher targets diagnostic'
    return result
