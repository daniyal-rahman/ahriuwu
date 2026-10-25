"""Match-local event and target metrics; no thresholds fitted on scored games."""
import math

import numpy as np

from .event_labels import UNIT_TARGET, POINT_TARGET, ANGLE_TARGET
from .events import BUTTONS, event_matches
from .labels import lane_to_screen
from ..screen import FLOOR_Y, CAM_Y, FOV_V_DEG, TILT_DEG, RESOLUTION


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
    classes, event_pairs = {}, {}
    for b, name in enumerate(BUTTONS):
        received = (button == b) & valid
        predicted = (emitted == b) & valid
        tp, pairs = 0, []
        for lo, hi in segments:
            matched = event_matches(t[lo:hi][predicted[lo:hi]], t[lo:hi][received[lo:hi]], tolerance)
            tp += matched['true_positive']
            pi, ri = np.flatnonzero(predicted[lo:hi]) + lo, np.flatnonzero(received[lo:hi]) + lo
            pairs.extend((int(pi[p]), int(ri[r])) for p, r in matched['pairs'])
        event_pairs[b] = pairs
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
    chosen_probability = probabilities[np.arange(len(t)), button]
    result = dict(valid_frames=int(valid.sum()), minutes=minutes, classes=classes,
                  button_nll=float(-np.log(np.maximum(chosen_probability[valid], 1e-30)).mean()) if valid.any() else None,
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
            selected = predictions[prefix+'_pointer'][mask]
            correct = (selected >= 0) & (selected == arrays['target_row'][mask])
            item['accuracy'] = float(correct.mean())
        elif mode == POINT_TARGET:
            cell = predictions[prefix+'_grid'][mask]
            dx, dy = cell % 96 - arrays['screen_x'][mask], cell // 96 - arrays['screen_y'][mask]
            error = np.hypot(dx, dy)
            correct = (cell >= 0) & (cell < 96*54) & (np.abs(dx) <= 2) & (np.abs(dy) <= 2)
            item.update(median_cells=float(np.median(error)), p90_cells=float(np.quantile(error, .9)),
                        within2=float(correct.mean()))
        else:
            angle = (predictions[prefix+'_angle'][mask] + .5) * (2*np.pi/72) - np.pi
            error = np.abs((angle - arrays['angle'][mask] + np.pi) % (2*np.pi) - np.pi) * 180/np.pi
            correct = (predictions[prefix+'_angle'][mask] >= 0) & (error <= 10.)
            item.update(median_degrees=float(np.median(error)), p90_degrees=float(np.quantile(error, .9)))
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
    result['target_coverage_by_button'] = {}
    result['matched_event_targets'] = {}
    for b, name in enumerate(BUTTONS):
        rows = valid & (button == b)
        required = rows & (arrays['aim_mode'] != 0)
        eligible = required & arrays['aim_valid']
        total = int(eligible.sum())
        result['target_coverage_by_button'][name] = dict(events=int(rows.sum()), required=int(required.sum()),
            eligible=total, clipped=int((required & arrays.get('clipped', np.zeros(len(t), bool))).sum()),
            coverage=total/int(required.sum()) if required.any() else None)
        successes, scored, unavailable = 0, 0, 0
        for pi, ri in event_pairs[b]:
            if not eligible[ri]:
                continue
            mode = arrays['aim_mode'][ri]
            if mode == UNIT_TARGET and 'unit_ids' in arrays:
                target = int(predictions['actual_pointer'][pi])
                correct = (0 <= target < arrays['unit_ids'].shape[1] and
                    arrays['unit_ids'][pi, target] >= 0 and
                    arrays['unit_ids'][pi, target] == arrays['unit_ids'][ri, arrays['target_row'][ri]])
            elif mode == POINT_TARGET and 'anchor_xy' in arrays:
                cell = int(predictions['actual_grid'][pi])
                world = cell_world(cell, arrays['anchor_xy'][pi], arrays['frame_axis'], arrays['frame_normal'])
                offset = world - arrays['anchor_xy'][ri]
                x, y = lane_to_screen(offset @ arrays['frame_axis'], offset @ arrays['frame_normal'])
                dx, dy = x*96 - .5 - arrays['screen_x'][ri], y*54 - .5 - arrays['screen_y'][ri]
                correct = 0 <= cell < 96*54 and abs(dx) <= 2 and abs(dy) <= 2
            elif mode == ANGLE_TARGET:
                angle = (predictions['actual_angle'][pi] + .5) * (2*np.pi/72) - np.pi
                error = abs((angle - arrays['angle'][ri] + np.pi) % (2*np.pi) - np.pi)
                correct = predictions['actual_angle'][pi] >= 0 and error <= np.pi/18
            else:
                unavailable += 1
                continue
            scored += 1
            successes += int(correct)
        result['matched_event_targets'][name] = dict(eligible=total, matched_scored=scored,
            coordinate_unavailable=unavailable, successes=successes,
            joint_success=successes/total if total and not unavailable else None)
    result['limits'] = 'replay-conditioned history; targets use exact buttons; matched_event_targets uses tolerant pairs with identity/world projection; teacher targets diagnostic'
    return result


def cell_world(cell, anchor, axis, normal):
    """NumPy form of screen.screen_to_lane for host-side matched-event scoring."""
    sx, sy = (cell % 96 + .5)/96, (cell // 96 + .5)/54
    ct, st = math.cos(math.radians(TILT_DEG)), math.sin(math.radians(TILT_DEG))
    dy = FLOOR_Y - CAM_Y
    tv = math.tan(math.radians(FOV_V_DEG)/2)
    kv = (.5 - sy)*2*tv
    dz = dy*(ct + kv*st)/(kv*ct - st)
    vz = -dy*st + dz*ct
    ds, dn = (sx - .5)*2*tv*(RESOLUTION[0]/RESOLUTION[1])*vz, dy*ct/st + dz
    return np.asarray(anchor) + ds*np.asarray(axis) + dn*np.asarray(normal)
