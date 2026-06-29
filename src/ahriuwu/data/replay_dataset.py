"""Replay-data dataset: latents + memory-derived actions + computed rewards.

Each match contributes three on-disk artifacts produced by
``scripts/aggregation/pipeline.py``:

* Packed latents (``<latents_dir>/<match_id>.pt``) — same format the YT
  pretokenize pipeline emits.
* ``labels.json`` — per-frame memory snapshots (gold/level/hp, action,
  visible_heroes, movement).
* ``clicks.json`` — gt-tagged click and cast events (used for binary
  ability-press flags; one frame per cast).

Contract with the pretokenize step (NOT runtime-checked):
  ``frame_indices`` MUST be sorted strictly ascending (no duplicates,
  no permutation). The slice-based latent loading in ``__getitem__``
  assumes ``frame_to_idx[run_start + k] == frame_to_idx[run_start] + k``.
  Pretokenize writes in PNG-number order, so this holds for any pack
  produced by our own tooling.

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

# Garen v1 action mapping: clicks.json cast spell_name (or label.action.spell)
# -> action key. TP / super-recall / any unmapped spell are intentionally ignored.
_SPELL_TO_KEY = {
    "GarenQ": "Q", "GarenW": "W", "GarenR": "R",
    # GarenE and GarenECancel are inconsistent per-match aliases for "E used"
    # (verified mutually exclusive across matches) -> both map to E.
    "GarenE": "E", "GarenECancel": "E",
    "SummonerFlash": "Flash", "SummonerDot": "Ignite",
    "recall": "Recall",
}
_STRIDE_ITEM_ID = 6631  # Stridebreaker — the one item-active the labels log (sparse).


def load_outcomes(manifest_path: Path | str) -> dict[str, bool]:
    """Read ``garen_win`` per ``match_id`` from a manifest JSON.

    Raises if any match entry is missing ``garen_win`` — silently defaulting
    a missing outcome to False would flip the sign of the terminal reward.
    """
    with open(manifest_path) as f:
        manifest = json.load(f)
    out: dict[str, bool] = {}
    missing: list[str] = []
    for m in manifest.get("matches", []):
        mid = m.get("match_id")
        if mid is None:
            continue
        if "garen_win" not in m:
            missing.append(mid)
            continue
        out[mid] = bool(m["garen_win"])
    if missing:
        raise ValueError(
            f"manifest at {manifest_path} is missing `garen_win` for "
            f"{len(missing)} match(es): {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
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
        max_cache_size: int = 2,
    ):
        # max_cache_size is per-worker — DataLoader fork-spawns each worker
        # with its own cache copy. With VideoShuffleSampler the access
        # pattern is roughly linear within each video, so a 2-deep LRU
        # gives near-100% hit rate without ballooning RAM at high
        # num_workers (each .pt file is ~210MB).
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
            seen: set[str] = set()
            # .pt is preferred (raw tensors); .npz only consulted for matches
            # that don't have a .pt next to them.
            for path in sorted(self.latents_dir.glob("*.pt")):
                mid = path.stem
                if mid == "index" or mid in seen:
                    continue
                data = torch.load(path, weights_only=True)
                frame_indices_by_match[mid] = data["frame_indices"].numpy()
                match_ids.append(mid)
                seen.add(mid)
            for path in sorted(self.latents_dir.glob("*.npz")):
                mid = path.stem
                if mid in seen:
                    continue
                with np.load(path) as data:
                    frame_indices_by_match[mid] = data["frame_indices"].copy()
                match_ids.append(mid)
                seen.add(mid)

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
            n_latent = len(frame_indices_by_match[mid])
            n_label = md["frame_count"]
            if n_latent != n_label:
                warnings.warn(
                    f"{mid}: latent frame count ({n_latent}) != label frame count "
                    f"({n_label}); using min({n_latent}, {n_label}) and dropping the rest"
                )
            self.match_data[mid] = md
            self._index_match(mid, frame_indices_by_match[mid], n_label)

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

        # Pipeline invariant: labels.frames[i].frame == i for every i. We rely
        # on this so that start_frame indexes both label-derived tensors and
        # latent frame numbers consistently. Hard-fail if it's violated.
        for i in range(min(T, 64)):  # spot-check the first 64 to avoid full O(T) cost
            f_idx = frames[i].get("frame")
            if f_idx is not None and f_idx != i:
                raise ValueError(
                    f"{match_id}: labels.frames[{i}].frame={f_idx!r} (expected {i}). "
                    "Pipeline invariant violated; dataset slicing assumes 1:1."
                )

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
        """Per-frame cursor (x, y) ∈ [0, 1] from label.cursor.screen.

        ``cursor.screen`` is the most-recent issued-command location: ability
        target if a spell is active this frame, AA target if attacking, else
        the most-recent movement click from clicks.json — held forward through
        idle frames. Pipeline emits None before any input or when off-screen;
        we hold the previous in-frame value, fall back to (0.5, 0.5) only if
        no cursor info exists yet (e.g., very early-game frames).

        Backwards compat: older labels (pre-cursor) keep ``movement.heading_screen``;
        we fall back to that when ``cursor`` is absent.
        """
        T = len(frames)
        screen_w, screen_h = labels["screen_resolution"]
        screen_w, screen_h = float(screen_w), float(screen_h)
        movement = torch.full((T, 2), 0.5, dtype=torch.float32)
        last_xy: Optional[tuple[float, float]] = None
        for i, fr in enumerate(frames):
            lab = fr.get("label")
            if not lab:
                if last_xy is not None:
                    movement[i, 0], movement[i, 1] = last_xy
                continue
            cs = (lab.get("cursor") or {}).get("screen")
            if cs and len(cs) == 2:
                last_xy = (cs[0] / screen_w, cs[1] / screen_h)
            else:
                # Old-schema fallback. Drop once all data is re-recorded /
                # backfilled to write `cursor`.
                hs = (lab.get("movement") or {}).get("heading_screen")
                if hs and len(hs) == 2:
                    last_xy = (hs[0] / screen_w, hs[1] / screen_h)
            if last_xy is not None:
                movement[i, 0], movement[i, 1] = last_xy
        return movement

    def _parse_abilities(
        self,
        labels: dict,
        frames: list[dict],
        match_dir: Path,
    ) -> dict[str, torch.Tensor]:
        """Per-frame binary action flags for the Garen v1 action space.

        clicks.json casts map by spell_name -> key (Q/W/E/Ecancel/R/Flash/Ignite/
        Recall); AA from label.action.type transitions; Stride from inventory
        `lf` jumps. Falls back to label.action.spell when clicks.json is absent.
        Unmapped spells (TP, super-recall, ...) are ignored.
        """
        T = len(frames)
        abilities: dict[str, torch.Tensor] = {
            k: torch.zeros(T, dtype=torch.long) for k in ABILITY_KEYS
        }
        if T == 0:
            return abilities

        # gt0 is the timebase reference. Pipeline writes gt for every frame;
        # cast game_t is anchored to it.
        gt0 = float(frames[0]["gt"])
        step = 1.0 / float(labels["fps"])

        clicks_path = match_dir / "clicks.json"
        casts: list[dict] = []
        if clicks_path.exists():
            with open(clicks_path) as f:
                casts = json.load(f).get("casts") or []

        if casts:
            n_no_time = n_unmapped = n_out_of_range = 0
            for c in casts:
                # Explicit None-check, not `or` — game_t == 0.0 is valid.
                gt = c.get("game_t")
                if gt is None:
                    gt = c.get("game_time")
                if gt is None:
                    n_no_time += 1
                    continue
                key = _SPELL_TO_KEY.get(c.get("spell_name"))
                if key is None:
                    n_unmapped += 1  # TP / super-recall / etc. — intentionally dropped
                    continue
                # round, not int-truncate, for symmetric frame-boundary quantization
                i = int(round((float(gt) - gt0) / step))
                if 0 <= i < T:
                    abilities[key][i] = 1
                else:
                    n_out_of_range += 1
            if n_no_time or n_out_of_range:
                warnings.warn(
                    f"{match_dir.name}: {n_no_time} no-time, {n_out_of_range} "
                    f"out-of-range casts dropped ({n_unmapped} unmapped ignored)"
                )
        else:
            warnings.warn(
                f"{match_dir.name}: no clicks.json at {clicks_path}; mapping "
                "QWER/Flash/Ignite/Recall from label.action.spell (lossy)."
            )
            self._fill_from_action_spell(frames, abilities)

        # AA — attack-move / auto-attack initiation isn't in the cast stream;
        # take it from label.action.type transitions into "attack".
        self._fill_aa_from_attack(frames, abilities)
        # Stride active — sparse signal from the item's `lf` (last-fired) jumps.
        self._fill_stride_from_inventory(frames, abilities)
        return abilities

    @staticmethod
    def _fill_aa_from_attack(
        frames: list[dict],
        abilities: dict[str, torch.Tensor],
    ) -> None:
        """abilities['AA'][i] = 1 on the frame label.action.type enters 'attack'."""
        prev_type: Optional[str] = None
        for i, fr in enumerate(frames):
            lab = fr.get("label")
            if not lab:
                prev_type = None
                continue
            atype = (lab.get("action") or {}).get("type")
            if atype == "attack" and prev_type != "attack":
                abilities["AA"][i] = 1
            prev_type = atype

    @staticmethod
    def _fill_stride_from_inventory(
        frames: list[dict],
        abilities: dict[str, torch.Tensor],
    ) -> None:
        """abilities['Stride'][i] = 1 when Stridebreaker's `lf` (last-fired
        game-time) jumps up — i.e. the active was used. Sparse (~2-18/game) but
        the only item-active the labels reliably log (pots/tiamat/ward have no
        usable signal). `lf` is held across unlabeled gaps so a gap isn't read
        as a use.
        """
        prev_lf: Optional[float] = None
        for i, fr in enumerate(frames):
            lab = fr.get("label")
            lf = None
            if lab:
                for it in (lab.get("inventory") or []):
                    if it and it.get("id") == _STRIDE_ITEM_ID:
                        lf = it.get("lf")
                        break
            if lf is not None and prev_lf is not None and lf > prev_lf + 1e-6:
                abilities["Stride"][i] = 1
            if lf is not None:
                prev_lf = lf

    @staticmethod
    def _fill_from_action_spell(
        frames: list[dict],
        abilities: dict[str, torch.Tensor],
    ) -> None:
        """Fallback (no clicks.json): map label.action.spell -> key on entry."""
        prev_spell: Optional[str] = None
        for i, fr in enumerate(frames):
            lab = fr.get("label")
            if not lab:
                prev_spell = None
                continue
            spell = (lab.get("action") or {}).get("spell")
            if spell and spell != prev_spell:
                key = _SPELL_TO_KEY.get(spell)
                if key:
                    abilities[key][i] = 1
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
