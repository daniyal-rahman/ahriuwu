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
    def target_metrics(prefix, mode, eligible):
        mask = eligible & ((emitted == button) if prefix == 'actual' else True)
        count, total = int(mask.sum()), int(eligible.sum())
        item = dict(count=count, eligible=total)
        if not count:
            if prefix == 'actual' and total:
                item['joint_success'] = 0.
            return item
        if mode == UNIT_TARGET:
            correct = predictions[prefix+'_pointer'][mask] == arrays['target_row'][mask]
            item['accuracy'] = float(correct.mean())
        elif mode == POINT_TARGET:
            cell = predictions[prefix+'_grid'][mask]
            dx, dy = cell % 96 - arrays['screen_x'][mask], cell // 96 - arrays['screen_y'][mask]
            error = np.hypot(dx, dy)
            correct = (np.abs(dx) <= 2) & (np.abs(dy) <= 2)
            item.update(median_cells=float(np.median(error)), p90_cells=float(np.quantile(error, .9)),
                        within2=float(correct.mean()))
        else:
            angle = (predictions[prefix+'_angle'][mask] + .5) * (2*np.pi/72) - np.pi
            error = np.abs((angle - arrays['angle'][mask] + np.pi) % (2*np.pi) - np.pi) * 180/np.pi
            correct = error <= 10.
            item.update(median_degrees=float(np.median(error)), p90_degrees=float(np.quantile(error, .9)))
        if mode == POINT_TARGET and 'angle' in arrays:
            from ..screen import screen_to_lane
            ds, dn = screen_to_lane((cell % 96 + .5)/96, (cell // 96 + .5)/54)
            predicted_angle = np.arctan2(np.asarray(dn), np.asarray(ds))
            actual_angle = arrays['angle'][mask]
            known = np.isfinite(actual_angle)
            if known.any():
                angle_error = np.abs((predicted_angle[known] - actual_angle[known] + np.pi) % (2*np.pi) - np.pi)*180/np.pi
                item['direction_count'] = int(known.sum())
                item['median_degrees'] = float(np.median(angle_error))
                item['p90_degrees'] = float(np.quantile(angle_error, .9))
                item['within10'] = float((angle_error <= 10).mean())
        if mode == ANGLE_TARGET:
            item['direction_count'] = count
            item['within10'] = float(correct.mean())
        if prefix == 'actual':
            item['joint_success'] = float(correct.sum())/total
        return item

    result['targets'], result['targets_by_button'] = {}, {}
    for prefix in ('actual', 'teacher'):
        for name, mode in [('unit', UNIT_TARGET), ('point', POINT_TARGET), ('angle', ANGLE_TARGET)]:
            eligible = valid & arrays['aim_valid'] & (arrays['aim_mode'] == mode)
            key = prefix+'_'+name
            result['targets'][key] = target_metrics(prefix, mode, eligible)
            result['targets_by_button'][key] = {
                bname: target_metrics(prefix, mode, eligible & (button == b))
                for b, bname in enumerate(BUTTONS) if np.any(eligible & (button == b))}
    result['limits'] = 'replay-conditioned observations; actual target metrics conditional on exact correct button; teacher targets diagnostic'
    return result
