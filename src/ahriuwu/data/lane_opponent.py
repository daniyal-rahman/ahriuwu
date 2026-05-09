"""Lane-opponent detection from labels.json.

Two heuristics over the visible_heroes block:

1. **Team assignment** by spawn-side. Each hero's earliest recorded position
   is at-or-near their fountain (blue ~(400, 400), red ~(14400, 14400))
   before they walk out, so the minimum-coord-sum sample identifies team.

2. **Lane opponent** = enemy with the smallest mean distance to the focus
   champion during the laning phase (gt 60-300s).

Both run on labels.json alone and don't require additional metadata.

Future cleanup: pipeline.py could write the resolved lane opponent into
labels.json once at post-process time so consumers don't re-derive.
"""

from __future__ import annotations

import math
from typing import Optional


_BLUE_SPAWN_MAX = 3000.0  # x and z below this at game start = blue side


def identify_teams(frames: list[dict], max_seconds: float = 30.0) -> dict[str, str]:
    """Return {hero_name: 'blue' | 'red'}.

    Looks at the first ``max_seconds`` of labeled frames and assigns each hero
    to a team from its earliest (lowest-coord-sum) world position.
    """
    earliest: dict[str, tuple[float, float]] = {}
    if not frames:
        return {}

    gt0: Optional[float] = None
    for fr in frames:
        lab = fr.get("label")
        if not lab:
            continue
        if gt0 is None:
            gt0 = fr["gt"]
        if fr["gt"] - gt0 > max_seconds:
            break
        for vh in lab.get("visible_heroes") or []:
            name = vh.get("name")
            world = vh.get("world")
            if not (name and world and len(world) == 2):
                continue
            cur = earliest.get(name)
            if cur is None or world[0] + world[1] < cur[0] + cur[1]:
                earliest[name] = (world[0], world[1])

    return {
        name: ("blue" if (x < _BLUE_SPAWN_MAX and z < _BLUE_SPAWN_MAX) else "red")
        for name, (x, z) in earliest.items()
    }


def find_lane_opponent(
    frames: list[dict],
    focus_name: str,
    enemy_names: list[str],
    gt_lo: float = 60.0,
    gt_hi: float = 300.0,
) -> Optional[str]:
    """Return the enemy with min mean distance to focus during laning.

    Returns None when no enemy has any sample in the [gt_lo, gt_hi] window.
    """
    sums: dict[str, float] = {n: 0.0 for n in enemy_names}
    counts: dict[str, int] = {n: 0 for n in enemy_names}

    for fr in frames:
        gt = fr.get("gt")
        if gt is None or gt < gt_lo or gt > gt_hi:
            continue
        lab = fr.get("label")
        if not lab:
            continue
        focus_w: Optional[tuple[float, float]] = None
        enemy_w: dict[str, tuple[float, float]] = {}
        for vh in lab.get("visible_heroes") or []:
            name = vh.get("name")
            world = vh.get("world")
            if not (name and world and len(world) == 2):
                continue
            if name == focus_name:
                focus_w = (world[0], world[1])
            elif name in enemy_names:
                enemy_w[name] = (world[0], world[1])
        if focus_w is None:
            continue
        for name, w in enemy_w.items():
            sums[name] += math.hypot(w[0] - focus_w[0], w[1] - focus_w[1])
            counts[name] += 1

    best: Optional[str] = None
    best_mean = float("inf")
    for name in enemy_names:
        if counts[name] == 0:
            continue
        mean = sums[name] / counts[name]
        if mean < best_mean:
            best, best_mean = name, mean
    return best


def resolve_lane_opponent(labels: dict) -> Optional[str]:
    """End-to-end: from a loaded labels.json dict, return the lane opponent's name.

    Prefers ``labels["lane_opponent"]`` when the pipeline already persisted it
    (recordings post commit `<lane_opponent persistence>`). Otherwise re-derives
    from team detection + lane-opponent search.

    Returns None when no team is detectable, no enemies are visible in the
    laning-phase window, or labels lacks a champion field.
    """
    cached = labels.get("lane_opponent")
    if cached:
        return cached

    frames = labels.get("frames", [])
    focus_name = labels.get("champion")
    if not focus_name:
        return None

    teams = identify_teams(frames)
    focus_team = labels.get("team") or teams.get(focus_name)
    if focus_team not in ("blue", "red"):
        return None

    enemy_team = "red" if focus_team == "blue" else "blue"
    enemy_names = [n for n, t in teams.items() if t == enemy_team]
    if not enemy_names:
        return None

    return find_lane_opponent(frames, focus_name, enemy_names)
