"""Approved approximate team fog over replay units; docs/modern/BC.md section 11."""
from __future__ import annotations

import numpy as np

# Deliberate heuristic parameters, not decoded sight ranges or simulator constants.
PROFILE = "replay-team-circles-v1"
SIGHT_RADIUS = {"champion": 1350.0, "minion": 1200.0, "turret": 1350.0}


def provenance():
    return dict(profile=PROFILE, source="user-approved heuristic", decoded=False,
                sight_radius=dict(SIGHT_RADIUS), sources="live replay allies with finite positions",
                omitted=["walls", "brush", "wards", "stealth", "temporary reveals",
                         "units absent from the replay reader"])


def team_visibility(units, x, y, alive, unit_of_slot):
    """(T, 2, N) approximate visibility for replay indices in ``unit_of_slot``.

    -1 slots have no replay evidence and never become visible. All replay allies
    can supply sight, including champions omitted from the policy entity slots.
    Frames are independent: neither future positions nor stale sightings reveal
    enemies. Work per frame avoids a large T x sources x targets distance tensor.
    """
    x, y, alive = np.asarray(x), np.asarray(y), np.asarray(alive, bool)
    slots = np.asarray(unit_of_slot)
    if x.ndim != 2 or x.shape != y.shape or x.shape != alive.shape or x.shape[1] != len(units):
        raise ValueError("fog requires aligned replay unit positions and life")
    if slots.ndim != 2 or slots.shape[0] != len(x):
        raise ValueError("fog requires per-frame replay slot indices")
    if np.any((slots < -1) | (slots >= len(units))):
        raise ValueError("fog slot index outside replay units")
    result = np.zeros((len(x), 2, slots.shape[1]), bool)
    teams = np.asarray([u.team for u in units])
    radius = np.asarray([SIGHT_RADIUS.get(u.family, 0.0) for u in units])
    known = alive & np.isfinite(x) & np.isfinite(y)
    for k, row in enumerate(slots):
        dest = np.flatnonzero(row >= 0)
        target = row[dest]
        dest, target = dest[known[k, target]], target[known[k, target]]
        for team in (0, 1):
            sources = np.flatnonzero(known[k] & (teams == team) & (radius > 0))
            if not len(sources) or not len(target):
                continue
            dx = x[k, target, None] - x[k, sources][None, :]
            dy = y[k, target, None] - y[k, sources][None, :]
            result[k, team, dest] = np.any(dx * dx + dy * dy <= radius[sources] ** 2, axis=1)
    return result
