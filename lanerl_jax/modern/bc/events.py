"""Lane BC v2 event targets and evaluation; docs/modern/BC.md section 15."""
from __future__ import annotations

import numpy as np


BUTTONS = ("continue", "move", "attack", "q", "w", "e", "r", "recall", "stop")
BUTTON_INDEX = {name: i for i, name in enumerate(BUTTONS)}
CASTS = tuple(BUTTON_INDEX[name] for name in ("q", "w", "e", "r"))


def ordinary_move(action):
    return (action.kind == 'path' and action.n_waypoints > 1
            and not getattr(action, 'special_motion', False) and not (getattr(action, 'motion_flags', 0) & 0xff01))


def modeled_events(actions):
    """Shared receipt interpretation; None invalidates history/supervision, not a stop key.

    After a one-point/special path a same-target execution cannot distinguish an
    uninterrupted attack from a new input. Preserve that ambiguity until an order
    or a changed target supplies evidence (BC.md section15).
    """
    active, interrupted = None, False
    previous = None
    for i, a in enumerate(actions):
        key = (a.t, a.source_order)
        if previous is not None and key < previous:
            raise ValueError('action receipts must retain native time/order')
        previous = key
        if a.kind == 'path':
            if ordinary_move(a):
                active, interrupted = None, False
                yield i, BUTTON_INDEX['move']
            else:
                interrupted = True
                yield i, None
        elif a.kind == 'attack':
            if a.target is None or (interrupted and a.target == active):
                yield i, None
            elif a.target != active:
                interrupted = False
                yield i, BUTTON_INDEX['attack']
            active = a.target
        elif a.kind == 'cast':
            if a.slot in range(4):
                yield i, BUTTON_INDEX['q'] + a.slot
            elif a.slot == 13:
                active, interrupted = None, False
                yield i, BUTTON_INDEX['recall']
            else:
                yield i, None


def smooth_cast_targets(button, valid, kernel=(0.2, 0.5, 1.0, 0.5, 0.2)):
    """Move one unit of each cast's mass into neighboring continue frames.

    Other events are immutable. Nearest-cast ownership makes neighborhoods
    disjoint, preserving per-button mass even for adjacent casts and boundaries.
    These are training targets, never observation features or availability masks.
    """
    button, valid = np.asarray(button), np.asarray(valid, bool)
    kernel = np.asarray(kernel, np.float64)
    if button.ndim != 1 or valid.shape != button.shape:
        raise ValueError("button and valid must be aligned vectors")
    if not np.issubdtype(button.dtype, np.integer) or np.any((button < 0) | (button >= len(BUTTONS))):
        raise ValueError("unknown event button")
    if (kernel.ndim != 1 or not len(kernel) or len(kernel) % 2 == 0
            or not np.isfinite(kernel).all() or (kernel < 0).any() or kernel[len(kernel) // 2] <= 0):
        raise ValueError("kernel must be finite, nonnegative, odd and positive at its center")
    target = np.eye(len(BUTTONS), dtype=np.float64)[button] * valid[:, None]
    events = np.flatnonzero(valid & np.isin(button, CASTS))
    if not len(events):
        return target.astype(np.float32)
    radius = len(kernel) // 2
    segment = np.cumsum(~valid)
    owner = np.full(len(button), -1, np.int64)
    distance = np.full(len(button), radius + 1, np.int64)
    for event in events:
        rows = np.arange(max(0, event - radius), min(len(button), event + radius + 1))
        delta = np.abs(rows - event)
        allowed = valid[rows] & (segment[rows] == segment[event]) & ((button[rows] == 0) | (rows == event))
        use = allowed & (delta < distance[rows]) & (kernel[rows - event + radius] > 0)
        owner[rows[use]], distance[rows[use]] = event, delta[use]
    for event in events:
        rows = np.flatnonzero(owner == event)
        mass = kernel[rows - event + radius]
        mass = mass / mass.sum()
        target[rows] = 0.0
        target[rows, 0] = 1.0 - mass
        target[rows, button[event]] = mass
    return target.astype(np.float32)


def event_matches(predicted, received, tolerance=0.5):
    """Maximum-cardinality one-to-one matching in ordered time, per button."""
    predicted, received = (np.asarray(v, np.float64) for v in (predicted, received))
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    if any(v.ndim != 1 or not np.isfinite(v).all() or (np.diff(v) < 0).any()
           for v in (predicted, received)):
        raise ValueError("event times must be finite ordered vectors")
    pairs, i, j = [], 0, 0
    while i < len(predicted) and j < len(received):
        if predicted[i] < received[j] - tolerance:
            i += 1
        elif received[j] < predicted[i] - tolerance:
            j += 1
        else:
            pairs.append((i, j))
            i, j = i + 1, j + 1
    tp = len(pairs)
    return dict(pairs=pairs, true_positive=tp, false_positive=len(predicted) - tp,
                false_negative=len(received) - tp,
                precision=tp / max(len(predicted), 1), recall=tp / max(len(received), 1),
                f1=2 * tp / max(len(predicted) + len(received), 1))
