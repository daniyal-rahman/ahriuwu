"""Per-frame reward for Garen TOP RL training.

Reward is the sum of four components, computed once per episode from a
loaded labels.json plus the win/loss outcome:

  Dense   - per-frame Δ(focus.gold_total - lane_opp.gold_total). Primary
            laning signal: positive when Garen out-earns the lane opponent
            this frame, negative when they out-earn Garen.
  Backup  - per-frame max(0, Δ focus.gold_total). Small weight; only matters
            when both teams gain together (the relative term gives 0).
  Event   - one-shot penalty the frame Garen's hp transitions to 0.
  Anchor  - one-shot ±lane_anchor at the frame closest to lane_checkpoint_gt
            (default 14:00) based on gold/level diff thresholds.
  Outcome - one-shot ±outcome_weight at the last labeled frame.

Why this shape: with γ=0.997 and DreamerV4's ~192-frame imagination horizon,
a purely-terminal reward at frame 36000 has effectively zero gradient at
early-game frames. The dense components keep the value head training every
imagination window; the anchors pin it at known points.

Public entry point: ``compute_episode_reward(labels, garen_won, config)``
returns a (T,) tensor, where T = ``len(labels["frames"])``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from ..data.lane_opponent import resolve_lane_opponent


@dataclass
class RewardConfig:
    """Reward weights and thresholds.

    Calibration intent (typical 30-min Garen game):
      Dense gold-diff term integrates to ≈ ±0.5 over the game.
      Death events take ~−0.6 cumulative across 3 deaths.
      Lane anchor adds ±0.5 at 14:00.
      Outcome adds ±1.0 at game end.
      Total per-game reward range ≈ [-2.5, +2.5].
    """

    gold_diff_scale: float = 5e-5
    gold_self_scale: float = 1e-5
    death_penalty: float = -0.2
    lane_checkpoint_gt: float = 14 * 60.0
    lane_gold_threshold: float = 1500.0
    lane_level_threshold: int = 2
    lane_anchor_weight: float = 0.5
    outcome_weight: float = 1.0


def compute_episode_reward(
    labels: dict,
    garen_won: bool,
    config: Optional[RewardConfig] = None,
) -> torch.Tensor:
    """Return the per-frame reward tensor for an episode.

    Args:
        labels: parsed labels.json (must have ``champion`` and ``frames``).
        garen_won: True if the focus champion's team won this game.
        config: reward weights/thresholds; defaults to ``RewardConfig()``.

    Returns:
        A (T,) float32 tensor where T = len(labels["frames"]).
        Unlabeled frames receive 0.
    """
    cfg = config or RewardConfig()
    frames = labels.get("frames") or []
    T = len(frames)
    if T == 0:
        return torch.zeros(0, dtype=torch.float32)

    focus_name = labels.get("champion")
    if not focus_name:
        raise ValueError("labels.champion is required for reward computation")

    opp_name = resolve_lane_opponent(labels)

    rewards = torch.zeros(T, dtype=torch.float32)
    rewards += _dense_gold_diff(frames, opp_name, cfg)
    rewards += _dense_gold_self(frames, cfg)
    rewards += _death_event(frames, cfg)
    if opp_name is not None:
        rewards += _lane_anchor(frames, opp_name, cfg)
    rewards += _outcome(frames, garen_won, cfg)
    return rewards


# ─────────────────────────── components ──────────────────────────────


def _focus_stats(label: Optional[dict]) -> Optional[dict]:
    """Pull the focus champion's per-frame stats dict, or None if unlabeled.

    Tolerates both the current schema (``champion_stats``) and the legacy
    pre-multi-champion schema (``garen_stats``) so old labels still load.
    """
    if not label:
        return None
    return label.get("champion_stats") or label.get("garen_stats")


def _hero_stat(label: Optional[dict], hero_name: str, key: str) -> Optional[float]:
    """Read ``key`` for a non-focus hero from ``visible_heroes``, or None."""
    if not label:
        return None
    for vh in label.get("visible_heroes") or []:
        if vh.get("name") == hero_name:
            v = vh.get(key)
            return float(v) if v is not None else None
    return None


def _dense_gold_diff(
    frames: list[dict],
    opp_name: Optional[str],
    cfg: RewardConfig,
) -> torch.Tensor:
    """β · Δ(focus.gold_total - opp.gold_total) per frame.

    Resets across unlabeled gaps (a missing frame breaks the delta chain to
    avoid charging a fictitious jump when labels resume).
    """
    T = len(frames)
    out = torch.zeros(T, dtype=torch.float32)
    if not opp_name:
        return out

    prev_diff: Optional[float] = None
    for i, fr in enumerate(frames):
        cs = _focus_stats(fr.get("label"))
        opp_gold = _hero_stat(fr.get("label"), opp_name, "gold_total")
        if cs is None or opp_gold is None:
            prev_diff = None
            continue
        focus_gold = float(cs.get("gold_total", 0.0) or 0.0)
        diff = focus_gold - opp_gold
        if prev_diff is not None:
            out[i] = cfg.gold_diff_scale * (diff - prev_diff)
        prev_diff = diff
    return out


def _dense_gold_self(frames: list[dict], cfg: RewardConfig) -> torch.Tensor:
    """β' · max(0, Δ focus.gold_total) per frame. Positive-only."""
    T = len(frames)
    out = torch.zeros(T, dtype=torch.float32)
    prev_gold: Optional[float] = None
    for i, fr in enumerate(frames):
        cs = _focus_stats(fr.get("label"))
        if cs is None:
            prev_gold = None
            continue
        gold = float(cs.get("gold_total", 0.0) or 0.0)
        if prev_gold is not None:
            delta = gold - prev_gold
            if delta > 0.0:
                out[i] = cfg.gold_self_scale * delta
        prev_gold = gold
    return out


def _death_event(frames: list[dict], cfg: RewardConfig) -> torch.Tensor:
    """Penalty the frame focus.hp transitions from positive to 0.

    Fires on each death (so a game with 3 deaths emits 3 penalties).
    Resets across unlabeled gaps to avoid spurious transitions.
    """
    T = len(frames)
    out = torch.zeros(T, dtype=torch.float32)
    prev_hp: Optional[float] = None
    for i, fr in enumerate(frames):
        cs = _focus_stats(fr.get("label"))
        if cs is None:
            prev_hp = None
            continue
        hp = float(cs.get("hp", 0.0) or 0.0)
        if prev_hp is not None and prev_hp > 0.0 and hp <= 0.0:
            out[i] = cfg.death_penalty
        prev_hp = hp
    return out


def _lane_anchor(
    frames: list[dict],
    opp_name: str,
    cfg: RewardConfig,
) -> torch.Tensor:
    """One-shot ±lane_anchor_weight at the frame closest to lane_checkpoint_gt."""
    T = len(frames)
    out = torch.zeros(T, dtype=torch.float32)

    best_idx: Optional[int] = None
    best_gap = float("inf")
    for i, fr in enumerate(frames):
        gt = fr.get("gt")
        if gt is None or fr.get("label") is None:
            continue
        gap = abs(gt - cfg.lane_checkpoint_gt)
        if gap < best_gap:
            best_gap, best_idx = gap, i

    # Skip if the game ended before the checkpoint or labels stop short of it.
    if best_idx is None or best_gap > 5.0:
        return out

    label = frames[best_idx]["label"]
    cs = _focus_stats(label)
    if cs is None:
        return out
    focus_gold = float(cs.get("gold_total", 0.0) or 0.0)
    focus_level = int(cs.get("level", 0) or 0)
    opp_gold = _hero_stat(label, opp_name, "gold_total")
    opp_level = _hero_stat(label, opp_name, "level")
    if opp_gold is None or opp_level is None:
        return out

    gold_diff = focus_gold - opp_gold
    level_diff = focus_level - int(opp_level)

    ahead = gold_diff > cfg.lane_gold_threshold or level_diff >= cfg.lane_level_threshold
    behind = gold_diff < -cfg.lane_gold_threshold or level_diff <= -cfg.lane_level_threshold
    if ahead and not behind:
        out[best_idx] = cfg.lane_anchor_weight
    elif behind and not ahead:
        out[best_idx] = -cfg.lane_anchor_weight
    return out


def _outcome(
    frames: list[dict],
    garen_won: bool,
    cfg: RewardConfig,
) -> torch.Tensor:
    """One-shot ±outcome_weight at the last labeled frame."""
    T = len(frames)
    out = torch.zeros(T, dtype=torch.float32)
    for i in range(T - 1, -1, -1):
        if frames[i].get("label") is not None:
            out[i] = cfg.outcome_weight if garen_won else -cfg.outcome_weight
            return out
    return out
