"""Event-only replay supervision; Lane BC v2 (BC.md section 15)."""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .events import BUTTON_INDEX as B, modeled_events
from .labels import screen_cell

NO_TARGET, UNIT_TARGET, POINT_TARGET, ANGLE_TARGET = range(4)


class EventLabels(NamedTuple):
    button: np.ndarray
    valid: np.ndarray
    aim_mode: np.ndarray
    aim_valid: np.ndarray
    target_row: np.ndarray
    screen_x: np.ndarray
    screen_y: np.ndarray
    angle: np.ndarray
    clipped: np.ndarray
    source: np.ndarray
    collisions: dict


def label_events(t, actions, own_observed_xy, slot_ids, axis, normal, valid, me):
    """Keep event and target masks independent; no held destination supervision.

    ``slot_ids`` and the anchor are the delayed observation's identities/position.
    Attack records are execution proxies: only a changed target or an attack after
    a received movement order starts a new attack event. One-point path stops
    remain unqualified here rather than inventing an S-key press.
    """
    t, own = np.asarray(t, np.float64), np.asarray(own_observed_xy)
    ids, valid = np.asarray(slot_ids), np.asarray(valid, bool).copy()
    if t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all() or not np.allclose(np.diff(t), .1):
        raise ValueError("event grid must be finite ordered 10Hz times")
    if own.shape != (len(t), 2) or ids.shape != (len(t), 32) or valid.shape != t.shape:
        raise ValueError("event observations must align with the decision grid")
    button = np.zeros(len(t), np.int32)
    mode = np.zeros(len(t), np.int8)
    aim = np.zeros(len(t), bool)
    target = np.full(len(t), -1, np.int32)
    sx, sy = np.zeros((2, len(t)), np.int32)
    angle = np.full(len(t), np.nan, np.float32)
    clipped = np.zeros(len(t), bool)
    source = np.full(len(t), -1, np.int64)
    selected, bins = {}, {}
    for i, event_button in modeled_events(actions):
        action = actions[i]
        if not np.isfinite(action.t):
            raise ValueError("nonfinite action receipt")
        k = int(np.floor((action.t - t[0]) / .1 + 1e-9))
        kind = 'move' if action.kind == 'path' else action.kind
        if event_button == B['recall']:
            kind = 'recall'
        rank = {'path': 0, 'attack': 1, 'cast': 3}[action.kind]
        if event_button is None:
            kind = 'unsupported'
            rank = 2 if action.kind == 'cast' else rank
        if not 0 <= k < len(t):
            continue
        bins.setdefault(k, []).append(kind)
        if k not in selected or rank >= selected[k][0]:
            selected[k] = (rank, i, event_button or 0, kind)
    for k, (_, i, b, kind) in selected.items():
        a = actions[i]
        source[k], button[k] = i, b
        if kind == "unsupported":
            valid[k] = False
            continue
        if kind == "recall" or (kind == "cast" and (a.mode == "self" or a.target == me)):
            continue
        if a.target is not None:
            mode[k] = UNIT_TARGET
            rows = np.flatnonzero(ids[k] == a.target)
            if len(rows) == 1:
                target[k], aim[k] = rows[0], True
        elif a.point is not None and np.isfinite(a.point).all() and np.isfinite(own[k]).all():
            mode[k] = POINT_TARGET
            delta = np.asarray(a.point) - own[k]
            ds, dn = float(delta @ axis), float(delta @ normal)
            sx[k], sy[k], clipped[k] = screen_cell(ds, dn)
            if ds * ds + dn * dn > 1e-8:
                angle[k] = np.arctan2(dn, ds)
            # Preserve the event but do not call a clipped cursor an exact label.
            aim[k] = not clipped[k]
        elif kind in ("cast", "move"):
            mode[k] = POINT_TARGET
    collisions = dict(event_bins=len(bins), multi_event_bins=sum(len(v) > 1 for v in bins.values()),
                      cast_move_bins=sum("cast" in v and "move" in v for v in bins.values()),
                      unsupported_bins=sum("unsupported" in v for v in bins.values()))
    return EventLabels(button, valid, mode, aim & valid, target, sx, sy, angle, clipped, source, collisions)
