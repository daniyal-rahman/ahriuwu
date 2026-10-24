"""Received own-cast history, distinct from cooldown duration or readiness."""
import numpy as np


def received_cast_ages(t, casts, summoner_slot=None):
    """Elapsed seconds since each own QWERDF receipt; unknown before its first receipt."""
    t = np.asarray(t, np.float64)
    if t.ndim != 1 or not np.isfinite(t).all() or (np.diff(t) < 0).any():
        raise ValueError("Decision times must be finite and ordered")
    mapping = {4: 0, 5: 1} if summoner_slot is None else summoner_slot
    if set(mapping) != {4, 5} or set(mapping.values()) != {0, 1}:
        raise ValueError("Summoner mapping must map slots4/5 to canonical D/F")
    events = [[] for _ in range(6)]
    for tc, slot in casts:
        if not np.isfinite(tc):
            raise ValueError("Cast receipt time must be finite")
        if slot in range(6):
            canonical = slot if slot < 4 else 4 + mapping[slot]
            events[canonical].append(float(tc))
    ages = np.full((len(t), 6), np.nan, np.float32)
    for slot, times in enumerate(events):
        if not times:
            continue
        times = np.sort(times)
        index = np.searchsorted(times, t, side="right") - 1
        known = index >= 0
        ages[known, slot] = t[known] - times[index[known]]
    return ages
