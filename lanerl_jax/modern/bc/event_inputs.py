"""Causal v2 identity/history features. Every history value is a received fact."""
from __future__ import annotations

import numpy as np

from .events import BUTTON_INDEX as B, modeled_events
from .history import received_cast_ages


SELF_FIELDS = ("s", "n", "hp", "level", "gold", "cs", "ad", "ap", "armor", "mr", "dead",
               "mana", "shield", "in_combat", "ms", "range", "rank_q", "rank_w", "rank_e", "rank_r",
               "role_top", "role_mid", "role_bot", "role_support")
ABILITY_FIELDS = ("min_range", "max_range", "width", "cast_time", "projectile_speed", "base_cooldown",
                  "physical_damage", "magic_damage", "true_damage", "recast", "charges")
ENTITY_DIM = 25                         # v4's20 + received cast ages4 + previous target match
SELF_DIM = len(SELF_FIELDS) + 4 + len(B) + 3
GLOBAL_DIM = 3                         # clock, paired opponent visible, team


def encode_ages(seconds):
    """Preserve long-ult history and unknowns; this never asserts readiness."""
    return np.log1p(np.minimum(seconds, 600.)) / np.log1p(600.)


def previous_received_events(t, actions, own_xy, slot_ids, axis, normal, scale=3000., reset_times=()):
    """Previous modeled receipt at/before each observation's timestamp, not current label."""
    t, own_xy, slot_ids = np.asarray(t), np.asarray(own_xy), np.asarray(slot_ids)
    events = [(actions[i].t, b, actions[i].target, actions[i].point)
              for i, b in modeled_events(actions)]
    features = np.full((len(t), len(B) + 3), np.nan, np.float32)
    target_match = np.zeros_like(slot_ids, np.float32)
    if not events:
        return features, target_match
    times = np.asarray([e[0] for e in events])
    if not np.isfinite(times).all() or (np.diff(times) < 0).any():
        raise ValueError('received events must be finite and ordered')
    index = np.searchsorted(times, t, side='right') - 1
    reset_times = np.asarray(reset_times, np.float64)
    reset_index = np.searchsorted(reset_times, t, side='right') - 1
    for k in np.flatnonzero(index >= 0):
        tc, b, target, point = events[index[k]]
        if b is None or (reset_index[k] >= 0 and tc < reset_times[reset_index[k]]):
            continue
        features[k, :len(B)] = 0.
        features[k, b] = 1.
        features[k, -1] = encode_ages(t[k] - tc)
        if point is not None and np.isfinite(point).all():
            delta = np.asarray(point) - own_xy[k]
            features[k, len(B):len(B) + 2] = [delta @ axis / scale, delta @ normal / scale]
        if target is not None:
            target_match[k] = (slot_ids[k] == target) & (slot_ids[k] >= 0)
    return features, target_match


def visible_cast_ages(g, selected, visible, observation_index):
    """Return lagged per-row history, using visibility at the cast receipt."""
    ids = np.asarray([u.id for u in g.units])
    lookup = {u.id: i for i, u in enumerate(g.units)}
    own_team = g.units[g.index(g.me)].team
    accepted = {}
    for tc, uid, slot in g.champion_casts:
        if uid not in lookup or slot not in range(4):
            continue
        unit = lookup[uid]
        k = np.searchsorted(g.t, tc, side='right') - 1
        allied = g.units[unit].team == own_team
        if allied or (k >= 0 and visible[k, unit]):
            accepted.setdefault(uid, []).append((tc, slot))
    rows = selected[observation_index]
    result = np.full(rows.shape + (4,), np.nan, np.float32)
    for uid, casts in accepted.items():
        age = received_cast_ages(g.t, casts)[:, :4][observation_index]
        match = (rows >= 0) & (ids[np.maximum(rows, 0)] == uid)
        result[match] = np.broadcast_to(age[:, None, :], result.shape)[match]
    return encode_ages(result)
