"""Map11 lane geometry from ``data/26.19/geometry.json``: ``LANE_PATHS[team, lane]`` (Chaos walks the Order path
reversed; padded with the last point), ``LANE_PATH_LEN[lane]`` and ``BARRACKS[team, lane]`` spawn points.
"""
from __future__ import annotations

import json

import numpy as np

from ..data import PATCH_DIR

LANE_BOT, LANE_MID, LANE_TOP = 0, 1, 2
LANE_NAMES = ("bot", "mid", "top")      # geometry.json lane ids 0, 1, 2


def _lane_tables():
    geometry = json.loads((PATCH_DIR / "geometry.json").read_text())
    raw = [np.asarray(geometry["lane_paths"][name], np.float32) for name in LANE_NAMES]
    length = max(len(p) for p in raw)
    paths = np.zeros((2, 3, length, 2), np.float32)
    for lane, p in enumerate(raw):
        for team, q in enumerate((p, p[::-1])):          # Order->Chaos; Chaos reversed
            paths[team, lane, :len(q)] = q
            paths[team, lane, len(q):] = q[-1]
    barracks = np.zeros((2, 3, 2), np.float32)
    for b in geometry["barracks"]:
        barracks[b["team"], b["lane"]] = b["position"]
    return paths, np.asarray([len(p) for p in raw], np.int32), barracks


LANE_PATHS, LANE_PATH_LEN, BARRACKS = _lane_tables()
