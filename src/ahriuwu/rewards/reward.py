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

import warnings
from dataclasses import dataclass
from typing import Optional

import torch

from ..data.lane_opponent import resolve_lane_opponent


@dataclass
class RewardConfig:
    """Reward weights and thresholds.

    Calibration intent (typical 30-min Garen game):
      Dense gold-diff term telescopes — its sum equals
      gold_diff_scale · (final_gold_diff - initial_gold_diff). At a typical
      ±2k final lane gap that's ≈ ±0.1; a stomp at ±10k integrates to ±0.5.
      Dense gold-self adds ≈ +0.12 (one-sided, ~12k cumulative gold × β').
      Death events take ~−0.6 cumulative across 3 deaths.
      Lane anchor adds ±0.5 at 14:00.
      Outcome adds ±1.0 at game end.
      Total per-game reward magnitude typically lives in roughly [-2, +2].
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
    if opp_name is None:
        warnings.warn(
            f"compute_episode_reward: no lane opponent for "
            f"match={labels.get('match_id')!r} (focus={focus_name!r}); "
            "dense gold-diff and lane-anchor terms will be 0 for this episode"
        )

    rewards = torch.zeros(T, dtype=torch.float32)
    rewards += _dense_gold_diff(frames, opp_name, cfg)
    rewards += _dense_gold_self(frames, cfg)
    rewards += _death_event(frames, cfg)
    if opp_name is not None:
        rewards += _lane_anchor(frames, opp_name, cfg)
    rewards += _outcome(frames, garen_won, cfg)

    # Loud-fail on numerical pathology — gold_total can return garbage from a
    # stale memory read, which the dense Δ would amplify into a giant spike.
    if not torch.isfinite(rewards).all():
        n_bad = int((~torch.isfinite(rewards)).sum().item())
        raise ValueError(
            f"compute_episode_reward produced {n_bad} non-finite values for "
            f"match={labels.get('match_id')!r} — likely garbage gold_total reads"
        )

    # No-coverage warning — should only fire if labels has no usable
    # champion_stats / opponent / outcome plumbing.
    if (rewards == 0).all():
        warnings.warn(
            f"compute_episode_reward returned all-zero rewards for "
            f"match={labels.get('match_id')!r} (focus={focus_name!r}, "
            f"opp={opp_name!r}, T={T}). Check labels.json schema."
        )
    return rewards


# ─────────────────────────── components ──────────────────────────────


def _focus_stats(label: Optional[dict]) -> Optional[dict]:
    """Pull the focus champion's per-frame stats dict, or None if unlabeled."""
    if not label:
        return None
    return label.get("champion_stats")


def _hero_stat(label: Optional[dict], hero_name: str, key: str) -> Optional[float]:
    """Read ``key`` for a non-focus hero from ``visible_heroes``, or None."""
    if not label:
        return None
    for vh in label.get("visible_heroes") or []:
        if vh.get("name") == hero_name:
            v = vh.get(key)
            return float(v) if v is not None else None
    return None


def _safe_float(d: dict, key: str) -> Optional[float]:
    """Fetch a numeric stat from a stats dict. Returns None on missing/null —
    NEVER zero-fills, since a stale-read None silently treated as 0 would
    create a fictitious gold-delta spike on the next frame."""
    v = d.get(key)
    return float(v) if v is not None else None


def _dense_gold_diff(
    frames: list[dict],
    opp_name: Optional[str],
    cfg: RewardConfig,
) -> torch.Tensor:
    """β · Δ(focus.gold_total - opp.gold_total) per frame.

    Resets across unlabeled gaps and missing-stat frames so we never charge a
    fictitious jump when valid data resumes.
    """
    T = len(frames)
    out = torch.zeros(T, dtype=torch.float32)
    if not opp_name:
        return out

    prev_diff: Optional[float] = None
    for i, fr in enumerate(frames):
        cs = _focus_stats(fr.get("label"))
        if cs is None:
            prev_diff = None
            continue
        focus_gold = _safe_float(cs, "gold_total")
        opp_gold = _hero_stat(fr.get("label"), opp_name, "gold_total")
        if focus_gold is None or opp_gold is None:
            prev_diff = None
            continue
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
        gold = _safe_float(cs, "gold_total")
        if gold is None:
            prev_gold = None
            continue
        if prev_gold is not None:
            delta = gold - prev_gold
            if delta > 0.0:
                out[i] = cfg.gold_self_scale * delta
        prev_gold = gold
    return out


def _death_event(frames: list[dict], cfg: RewardConfig) -> torch.Tensor:
    """Penalty the frame focus.hp transitions from positive to 0.

    Fires on each death. Resets prev_hp across unlabeled gaps and missing-stat
    frames so a transient stale-read of None doesn't masquerade as a death.
    """
    T = len(frames)
    out = torch.zeros(T, dtype=torch.float32)
    prev_hp: Optional[float] = None
    for i, fr in enumerate(frames):
        cs = _focus_stats(fr.get("label"))
        if cs is None:
            prev_hp = None
            continue
        hp = _safe_float(cs, "hp")
        if hp is None:
            prev_hp = None
            continue
        if prev_hp is not None and prev_hp > 0.0 and hp <= 0.0:
            out[i] = cfg.death_penalty
        prev_hp = hp
    return out


def _lane_anchor(
    frames: list[dict],
    opp_name: str,
    cfg: RewardConfig,
) -> torch.Tensor:
    """One-shot ±lane_anchor_weight at the frame closest to lane_checkpoint_gt.

    Returns 0 when ahead-by-gold and behind-by-level (or vice versa) both
    fire — those are conflicting signals about lane state, treat as neutral.
    """
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
    focus_gold = _safe_float(cs, "gold_total")
    focus_level = _safe_float(cs, "level")
    opp_gold = _hero_stat(label, opp_name, "gold_total")
    opp_level = _hero_stat(label, opp_name, "level")
    if focus_gold is None or focus_level is None or opp_gold is None or opp_level is None:
        return out

    gold_diff = focus_gold - opp_gold
    level_diff = focus_level - opp_level

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
