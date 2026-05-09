"""Replay-data dataset: latents + memory-derived actions + computed rewards.

Each match contributes three on-disk artifacts produced by
``scripts/aggregation/pipeline.py``:

* Packed latents (``<latents_dir>/<match_id>.pt``) — same format the YT
  pretokenize pipeline emits.
* ``labels.json`` — per-frame memory snapshots (gold/level/hp, action,
  visible_heroes, movement).
* ``clicks.json`` — gt-tagged click and cast events (used for binary
  ability-press flags; one frame per cast).

The dataset emits per sequence:

* ``latents``: (T, C, H, W)
* ``actions``: ``{movement: (T, 2), Q W E R D F item B: (T,) long}``
* ``rewards``: (T,)

Action mapping
--------------
* **Q/W/E/R** — set on the frame matching the cast's ``game_t`` for any
  cast event whose ``slot`` is one of those letters.
* **D/F** — set the same way; if ``labels.summoner_slots`` is present we
  trust it; otherwise we fall back to the cast's ``slot`` field.
* **B** — set on cast events with ``slot == "B"`` (the recall stream
  pipeline already emits). Falls back to ``label.action.type == "recall"``
  if no cast event lands on that frame.
* **item** — not tracked by the pipeline; constant 0.
* **movement** (x, y) — ``label.movement.heading_screen`` normalized by
  ``labels.screen_resolution``; falls back to (0.5, 0.5) when the player
  is stationary or the frame is unlabeled.

Rewards come from :func:`ahriuwu.rewards.reward.compute_episode_reward`.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from ..constants import ABILITY_KEYS
from ..rewards.reward import RewardConfig, compute_episode_reward


_DEFAULT_SCREEN = (1280, 720)


def load_outcomes(manifest_path: Path | str) -> dict[str, bool]:
    """Read ``garen_win`` per ``match_id`` from a manifest JSON."""
    with open(manifest_path) as f:
        manifest = json.load(f)
    out: dict[str, bool] = {}
    for m in manifest.get("matches", []):
        mid = m.get("match_id")
        if mid is not None:
            out[mid] = bool(m.get("garen_win", False))
    return out


class ReplayLatentSequenceDataset(Dataset):
    """Sequences of (latents, actions, rewards) drawn from replay matches."""

    def __init__(
        self,
        latents_dir: Path | str,
        labels_root: Path | str,
        outcomes: dict[str, bool] | None = None,
        manifest_path: Path | str | None = None,
        sequence_length: int = 64,
        stride: int = 1,
        reward_config: RewardConfig | None = None,
        max_cache_size: int = 5,
    ):
        if outcomes is None and manifest_path is None:
            raise ValueError("Provide either `outcomes` dict or `manifest_path`")
        self.latents_dir = Path(latents_dir)
        self.labels_root = Path(labels_root)
        self.sequence_length = sequence_length
        self.stride = stride
        self.reward_config = reward_config or RewardConfig()
        self.max_cache_size = max_cache_size

        self.outcomes: dict[str, bool] = (
            dict(outcomes) if outcomes is not None else load_outcomes(manifest_path)
        )

        # LRU cache for packed latent arrays (heavy)
        self._latent_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._latent_cache_order: list[str] = []

        # Per-match parsed actions + rewards (light, kept for all loaded matches)
        self.match_data: dict[str, dict] = {}

        # Precomputed sequence index built at construction time
        self.sequences: list[dict] = []

        self._index()

    # ───────────────────────── indexing ─────────────────────────

    def _index(self) -> None:
        """Walk available matches, parse labels/clicks once, build sequence list."""
        index_path = self.latents_dir / "index.pt"
        if index_path.exists():
            index = torch.load(index_path, weights_only=True)
            match_ids = list(index.keys())
            frame_indices_by_match = {mid: index[mid].numpy() for mid in match_ids}
        else:
            match_ids = []
            frame_indices_by_match = {}
            for path in sorted(self.latents_dir.glob("*.pt")):
                mid = path.stem
                if mid == "index":
                    continue
                data = torch.load(path, weights_only=True)
                frame_indices_by_match[mid] = data["frame_indices"].numpy()
                match_ids.append(mid)

        skipped: list[tuple[str, str]] = []
        for mid in match_ids:
            if mid not in self.outcomes:
                skipped.append((mid, "not in outcomes manifest"))
                continue
            labels_path = self.labels_root / mid / "labels.json"
            if not labels_path.exists():
                skipped.append((mid, "missing labels.json"))
                continue

            md = self._parse_match(mid, labels_path)
            if md is None:
                skipped.append((mid, "labels parse returned no frames"))
                continue
            self.match_data[mid] = md
            self._index_match(mid, frame_indices_by_match[mid], md["frame_count"])

        n_matches = len(self.match_data)
        n_seqs = len(self.sequences)
        print(f"ReplayLatentSequenceDataset: {n_matches} matches, {n_seqs} sequences "
              f"(seq_len={self.sequence_length}, stride={self.stride})")
        for mid, why in skipped[:5]:
            print(f"  skipped {mid}: {why}")
        if len(skipped) > 5:
            print(f"  ... and {len(skipped) - 5} more skipped")

    def _index_match(
        self,
        match_id: str,
        frame_indices: np.ndarray,
        n_match_frames: int,
    ) -> None:
        """Append (start_frame, start_idx) entries for all valid windows in a match."""
        if len(frame_indices) < self.sequence_length:
            return
        usable = min(len(frame_indices), n_match_frames)
        frame_to_idx = {int(frame_indices[i]): i for i in range(len(frame_indices))}
        frame_nums = sorted(int(f) for f in frame_indices[:usable])

        # Walk contiguous runs and emit windows.
        run_start = frame_nums[0]
        run_len = 1
        for i in range(1, len(frame_nums)):
            if frame_nums[i] == frame_nums[i - 1] + 1:
                run_len += 1
            else:
                self._emit_windows(match_id, run_start, run_len, frame_to_idx)
                run_start = frame_nums[i]
                run_len = 1
        self._emit_windows(match_id, run_start, run_len, frame_to_idx)

    def _emit_windows(
        self,
        match_id: str,
        run_start: int,
        run_len: int,
        frame_to_idx: dict[int, int],
    ) -> None:
        if run_len < self.sequence_length:
            return
        for off in range(0, run_len - self.sequence_length + 1, self.stride):
            start_frame = run_start + off
            self.sequences.append({
                "video_id": match_id,
                "start_frame": start_frame,
                "start_idx": frame_to_idx[start_frame],
            })

    # ───────────────────────── per-match parsing ─────────────────────────

    def _parse_match(self, match_id: str, labels_path: Path) -> Optional[dict]:
        with open(labels_path) as f:
            labels = json.load(f)
        frames = labels.get("frames") or []
        T = len(frames)
        if T == 0:
            return None

        garen_won = self.outcomes[match_id]
        rewards = compute_episode_reward(labels, garen_won, self.reward_config)
        movement = self._parse_movement(labels, frames)
        abilities = self._parse_abilities(labels, frames, labels_path.parent)

        return {
            "rewards": rewards,
            "movement": movement,
            "abilities": abilities,
            "frame_count": T,
        }

    def _parse_movement(self, labels: dict, frames: list[dict]) -> torch.Tensor:
        """Per-frame (x, y) ∈ [0, 1] from movement.heading_screen, default (0.5, 0.5)."""
        T = len(frames)
        screen = labels.get("screen_resolution") or list(_DEFAULT_SCREEN)
        screen_w, screen_h = float(screen[0]), float(screen[1])
        movement = torch.full((T, 2), 0.5, dtype=torch.float32)
        for i, fr in enumerate(frames):
            lab = fr.get("label")
            if not lab:
                continue
            mv = lab.get("movement") or {}
            hs = mv.get("heading_screen")
            if hs and len(hs) == 2 and screen_w > 0 and screen_h > 0:
                movement[i, 0] = max(0.0, min(1.0, hs[0] / screen_w))
                movement[i, 1] = max(0.0, min(1.0, hs[1] / screen_h))
        return movement

    def _parse_abilities(
        self,
        labels: dict,
        frames: list[dict],
        match_dir: Path,
    ) -> dict[str, torch.Tensor]:
        """Per-frame binary ability flags. Prefers clicks.json cast events; falls
        back to labels.json action.type for recall when clicks.json is absent.
        """
        T = len(frames)
        abilities: dict[str, torch.Tensor] = {
            k: torch.zeros(T, dtype=torch.long) for k in ABILITY_KEYS
        }
        if T == 0:
            return abilities

        gt0 = frames[0].get("gt") or 0.0
        fps = float(labels.get("fps") or 20)
        step = 1.0 / fps

        clicks_path = match_dir / "clicks.json"
        casts: list[dict] = []
        if clicks_path.exists():
            with open(clicks_path) as f:
                clicks_data = json.load(f)
            casts = clicks_data.get("casts") or []

        if casts:
            for c in casts:
                gt = c.get("game_t")
                if gt is None:
                    gt = c.get("game_time")
                slot = c.get("slot")
                if gt is None or slot not in ABILITY_KEYS:
                    continue
                i = int((gt - gt0) / step)
                if 0 <= i < T:
                    abilities[slot][i] = 1
        else:
            # Fallback for matches recorded before clicks.json was emitted: surface
            # transitions in label.action so at least recalls and ability casts
            # produce a binary signal (lossy on D vs F — both flagged together).
            warnings.warn(
                f"No clicks.json at {clicks_path}; falling back to labels.action "
                "transitions for ability flags (lossy)."
            )
            self._fill_abilities_from_action_transitions(frames, abilities, labels)

        return abilities

    @staticmethod
    def _fill_abilities_from_action_transitions(
        frames: list[dict],
        abilities: dict[str, torch.Tensor],
        labels: dict,
    ) -> None:
        focus = labels.get("champion") or ""
        prev_spell: Optional[str] = None
        for i, fr in enumerate(frames):
            lab = fr.get("label")
            if not lab:
                prev_spell = None
                continue
            action = lab.get("action") or {}
            atype = action.get("type")
            spell = action.get("spell") or ""
            if spell == prev_spell:
                continue
            if atype == "ability" and focus and spell.startswith(focus):
                tail = spell[len(focus):]
                slot = tail[:1] if tail else ""
                if slot in ("Q", "W", "E", "R"):
                    abilities[slot][i] = 1
            elif atype == "summoner":
                # Ambiguous without summoner_slots metadata — flag both.
                abilities["D"][i] = 1
                abilities["F"][i] = 1
            elif atype == "recall":
                abilities["B"][i] = 1
            prev_spell = spell

    # ───────────────────────── latent loading ─────────────────────────

    def _load_latents(self, match_id: str) -> tuple[np.ndarray, np.ndarray]:
        if match_id in self._latent_cache:
            self._latent_cache_order.remove(match_id)
            self._latent_cache_order.append(match_id)
            return self._latent_cache[match_id]

        while len(self._latent_cache) >= self.max_cache_size:
            evicted = self._latent_cache_order.pop(0)
            del self._latent_cache[evicted]

        pt_path = self.latents_dir / f"{match_id}.pt"
        if pt_path.exists():
            data = torch.load(pt_path, weights_only=True)
            latents = data["latents"].numpy()
            frame_indices = data["frame_indices"].numpy()
        else:
            npz_path = self.latents_dir / f"{match_id}.npz"
            with np.load(npz_path, mmap_mode=None) as data:
                latents = data["latents"].copy()
                frame_indices = data["frame_indices"].copy()

        self._latent_cache[match_id] = (latents, frame_indices)
        self._latent_cache_order.append(match_id)
        return latents, frame_indices

    # ───────────────────────── Dataset API ─────────────────────────

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        seq = self.sequences[idx]
        match_id = seq["video_id"]
        start_frame = seq["start_frame"]
        start_idx = seq["start_idx"]
        T = self.sequence_length

        latents_arr, _ = self._load_latents(match_id)
        latents = torch.from_numpy(latents_arr[start_idx:start_idx + T].copy())

        md = self.match_data[match_id]
        sl = slice(start_frame, start_frame + T)
        rewards = md["rewards"][sl]
        movement = md["movement"][sl]
        actions = {"movement": movement}
        for k in ABILITY_KEYS:
            actions[k] = md["abilities"][k][sl]

        return {
            "latents": latents,
            "actions": actions,
            "rewards": rewards,
            "video_id": match_id,
            "start_frame": start_frame,
        }
