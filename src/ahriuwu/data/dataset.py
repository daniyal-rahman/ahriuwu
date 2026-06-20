"""PyTorch datasets for LoL frame sequences.

Two label-free datasets used for tokenizer training and YT pretraining:

* :class:`SingleFrameDataset` — flat list of frames.
* :class:`FrameSequenceDataset` — fixed-length contiguous frame windows.

Plus generic samplers that group sequences by video for cache locality.

Action- and reward-aware loading lives in
:mod:`ahriuwu.data.replay_dataset` (replay matches with labels.json).
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


# 352x352. Divisible by 16 (22x22 = 484 patches). 352 > 256 keeps more detail.
TARGET_SIZE = (352, 352)


class SingleFrameDataset(Dataset):
    """Flat dataset of individual frames for tokenizer training."""

    def __init__(
        self,
        frames_dir: Path | str,
        target_size: tuple[int, int] = TARGET_SIZE,
        file_ext: str = "jpg",
        transform=None,
    ):
        self.frames_dir = Path(frames_dir)
        self.target_size = target_size
        self.file_ext = file_ext
        self.transform = transform

        self.frame_paths: list[Path] = []
        for video_dir in self.frames_dir.iterdir():
            if not video_dir.is_dir():
                continue
            self.frame_paths.extend(sorted(video_dir.glob(f"frame_*.{self.file_ext}")))
        print(f"Indexed {len(self.frame_paths)} frames from {self.frames_dir}")

    def __len__(self) -> int:
        return len(self.frame_paths)

    def __getitem__(self, idx: int) -> dict:
        path = self.frame_paths[idx]
        frame = cv2.imread(str(path))
        if frame is None:
            raise FileNotFoundError(f"Failed to load frame: {path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)
        if self.transform:
            frame = self.transform(frame)
        else:
            frame = torch.from_numpy(frame).float() / 255.0
            frame = frame.permute(2, 0, 1)  # HWC -> CHW
        return {"frame": frame, "path": str(path)}


class FrameSequenceDataset(Dataset):
    """Fixed-length contiguous frame windows for world-model pretraining.

    Globs ``*.{file_ext}`` so both YT layout (``frame_NNNNNN.jpg``) and replay
    layout (``NNNNNN.png``) work. Stores the sorted per-video frame list and
    indexes into it at ``__getitem__`` time — no printf-format reconstruction
    that would assume the ``frame_`` prefix.
    """

    def __init__(
        self,
        frames_dir: Path | str,
        sequence_length: int = 192,
        stride: int = 1,
        target_size: tuple[int, int] = TARGET_SIZE,
        file_ext: str = "jpg",
        transform=None,
        skip_resize: bool = False,
        augment: bool = False,
        aug_brightness: float = 0.10,
        aug_contrast: float = 0.10,
        aug_saturation: float = 0.10,
        aug_hue: float = 0.05,
        aug_gamma: float = 0.15,
        aug_noise_std: float = 0.01,
    ):
        """
        Args:
            skip_resize: if True, skip cv2.resize at load time (assumes frames
                are already at target_size). ~0.5ms/frame saved; useful with a
                pre-resized corpus on /scratch. Falls back to resize for any
                frame that doesn't match target_size (defensive).
            augment: if True, apply on-the-fly augmentation (color jitter,
                gamma, mild Gaussian noise) to make the tokenizer robust to
                player display calibration / video filter differences. Aug
                params are SAMPLED ONCE per sequence (constant across all T
                frames) so temporal consistency is preserved.
            aug_*: per-augmentation magnitudes. Defaults are mild (±10-15%
                brightness/contrast/saturation/gamma, ±5° hue, σ=0.01 noise).
        """
        self.frames_dir = Path(frames_dir)
        self.sequence_length = sequence_length
        self.stride = stride
        self.target_size = target_size
        self.file_ext = file_ext
        self.transform = transform
        self.skip_resize = skip_resize
        self.augment = augment
        self.aug_brightness = aug_brightness
        self.aug_contrast = aug_contrast
        self.aug_saturation = aug_saturation
        self.aug_hue = aug_hue
        self.aug_gamma = aug_gamma
        self.aug_noise_std = aug_noise_std

        self.sequences: list[dict] = []
        for video_dir in self.frames_dir.iterdir():
            if not video_dir.is_dir():
                continue
            frames = sorted(video_dir.glob(f"*.{self.file_ext}"))
            if len(frames) < self.sequence_length:
                continue
            for start_idx in range(0, len(frames) - self.sequence_length + 1, self.stride):
                self.sequences.append({
                    "video_id": video_dir.name,
                    "start_idx": start_idx,
                    "frame_paths": frames,
                })

    def __len__(self) -> int:
        return len(self.sequences)

    def _apply_augment(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply color/gamma/noise augmentation. Params sampled once per
        sequence so the T frames share the same augmentation (temporal
        consistency preserved — the model shouldn't learn from non-physical
        per-frame flicker).

        Input/output shape: (T, 3, H, W) in [0, 1].
        """
        from torchvision.transforms.v2.functional import (
            adjust_brightness, adjust_contrast, adjust_saturation,
            adjust_hue, adjust_gamma,
        )
        # Sample ONCE for the sequence
        b = 1.0 + (torch.empty(1).uniform_(-self.aug_brightness, self.aug_brightness).item())
        c = 1.0 + (torch.empty(1).uniform_(-self.aug_contrast, self.aug_contrast).item())
        s = 1.0 + (torch.empty(1).uniform_(-self.aug_saturation, self.aug_saturation).item())
        h = torch.empty(1).uniform_(-self.aug_hue, self.aug_hue).item()
        g = 1.0 + (torch.empty(1).uniform_(-self.aug_gamma, self.aug_gamma).item())
        # Functional ops broadcast over leading dims; (T, 3, H, W) works.
        frames = adjust_brightness(frames, b)
        frames = adjust_contrast(frames, c)
        frames = adjust_saturation(frames, s)
        frames = adjust_hue(frames, h)
        frames = adjust_gamma(frames, max(g, 0.05))
        if self.aug_noise_std > 0:
            frames = frames + torch.randn_like(frames) * self.aug_noise_std
        return frames.clamp(0.0, 1.0)

    def __getitem__(self, idx: int) -> dict:
        seq = self.sequences[idx]
        start_idx = seq["start_idx"]
        paths = seq["frame_paths"][start_idx:start_idx + self.sequence_length]

        frames = []
        for path in paths:
            # 1-retry-with-backoff: a single transient cv2.imread None return
            # (NFS attr cache, rsync write contention) shouldn't kill a 40h run.
            # Observed once in ~130k reads on /scratch under load.
            frame = cv2.imread(str(path))
            if frame is None:
                import time as _t; _t.sleep(0.05)
                frame = cv2.imread(str(path))
            if frame is None:
                raise FileNotFoundError(f"Failed to load frame: {path}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Skip cv2.resize if input is already target_size (pre-resized
            # corpus case). Defensive fallback if size mismatch.
            if not self.skip_resize or (frame.shape[1], frame.shape[0]) != self.target_size:
                frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)
            if self.transform:
                frame = self.transform(frame)
            else:
                frame = torch.from_numpy(frame).float() / 255.0
                frame = frame.permute(2, 0, 1)
            frames.append(frame)

        stacked = torch.stack(frames)
        if self.augment:
            stacked = self._apply_augment(stacked)

        return {
            "frames": stacked,
            "video_id": seq["video_id"],
            "start_idx": start_idx,
        }


class VideoGroupedSampler(Sampler):
    """Yield sequences grouped by video. Shuffles video order each epoch and
    sequence order within each video, but keeps a video's sequences contiguous
    in the batch stream so cache hits dominate.
    """

    def __init__(self, dataset):
        groups: dict = defaultdict(list)
        for idx, seq in enumerate(dataset.sequences):
            groups[seq["video_id"]].append(idx)
        self.video_groups = list(groups.values())
        self._len = sum(len(g) for g in self.video_groups)

    def __iter__(self):
        order = torch.randperm(len(self.video_groups)).tolist()
        for v in order:
            group = self.video_groups[v]
            perm = torch.randperm(len(group)).tolist()
            yield from (group[i] for i in perm)

    def __len__(self) -> int:
        return self._len


class VideoShuffleSampler(Sampler):
    """Like :class:`VideoGroupedSampler` but seedable for reproducibility."""

    def __init__(self, dataset, seed: int | None = None):
        groups: dict = defaultdict(list)
        for idx, seq in enumerate(dataset.sequences):
            groups[seq["video_id"]].append(idx)
        self.video_to_indices = dict(groups)
        self.video_ids = list(self.video_to_indices.keys())
        self.seed = seed
        total = sum(len(v) for v in self.video_to_indices.values())
        print(f"VideoShuffleSampler: {len(self.video_ids)} videos, {total} sequences")

    def __iter__(self):
        rng = random.Random(self.seed)
        videos = self.video_ids.copy()
        rng.shuffle(videos)
        for v in videos:
            indices = self.video_to_indices[v].copy()
            rng.shuffle(indices)
            yield from indices

    def __len__(self) -> int:
        return sum(len(v) for v in self.video_to_indices.values())


class PackedLatentSequenceDataset(Dataset):
    """Label-free packed-latent sequences for dynamics training.

    Reads packed per-video files — ``<latents_dir>/<video_id>.pt`` (preferred)
    or ``<video_id>.npz`` (fallback) — each holding ``latents`` of shape
    ``(N, C, H, W)`` and ``frame_indices`` of shape ``(N,)``, and emits
    fixed-length windows over *contiguous* frame runs (no gaps within a window).

    Latents only — no actions/rewards.  The action/reward-conditioned path lives
    in :class:`ahriuwu.data.replay_dataset.ReplayLatentSequenceDataset`, which
    consumes the same packed format plus ``labels.json``/``clicks.json``.

    The channel count ``C`` (a.k.a. latent_dim) is whatever the packed files
    hold — never hardcoded — so the same code serves the dim-48 (old tokenizer)
    and dim-32 (frozen v7) latents unchanged.  Each item is
    ``{"latents": (T, C, H, W) float tensor, "video_id": str, "start_frame": int}``.
    """

    def __init__(
        self,
        latents_dir: Path | str,
        sequence_length: int = 64,
        stride: int = 1,
        max_cache_size: int = 5,
        load_actions: bool = False,
        features_dir=None,  # accepted for call-site compatibility; unused
    ):
        if load_actions:
            raise ValueError(
                "PackedLatentSequenceDataset is latents-only. Use "
                "ahriuwu.data.replay_dataset.ReplayLatentSequenceDataset for "
                "action/reward conditioning."
            )
        self.latents_dir = Path(latents_dir)
        self.sequence_length = int(sequence_length)
        self.stride = int(stride)
        self.max_cache_size = int(max_cache_size)

        # Per-worker LRU cache of packed arrays (heavy: ~200MB/video).
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._cache_order: list[str] = []
        self.sequences: list[dict] = []
        self._index()

    # ───────────────────────── indexing ─────────────────────────

    def _index(self) -> None:
        pt_files = sorted(self.latents_dir.glob("*.pt"))
        seen = {p.stem for p in pt_files}
        files = [p for p in pt_files if p.stem != "index"]
        files += [p for p in sorted(self.latents_dir.glob("*.npz")) if p.stem not in seen]
        for path in files:
            self._index_video(path.stem, self._read_frame_indices(path))
        if not self.sequences:
            raise RuntimeError(
                f"No contiguous sequences of length {self.sequence_length} found "
                f"under {self.latents_dir} ({len(files)} packed files scanned)."
            )
        print(
            f"PackedLatentSequenceDataset: {len(self.sequences)} sequences from "
            f"{len(files)} videos (T={self.sequence_length}, stride={self.stride})"
        )

    @staticmethod
    def _read_frame_indices(path: Path) -> np.ndarray:
        if path.suffix == ".pt":
            # mmap=True so indexing reads only the small frame_indices tensor,
            # not the ~200MB latents tensor in each packed file.
            return torch.load(path, weights_only=True, mmap=True)["frame_indices"].numpy()
        with np.load(path) as data:
            return data["frame_indices"].copy()

    def _index_video(self, video_id: str, frame_indices: np.ndarray) -> None:
        n = len(frame_indices)
        if n < self.sequence_length:
            return
        frame_to_idx = {int(frame_indices[i]): i for i in range(n)}
        frame_nums = sorted(frame_to_idx)
        run_start, run_len = frame_nums[0], 1
        for i in range(1, len(frame_nums)):
            if frame_nums[i] == frame_nums[i - 1] + 1:
                run_len += 1
            else:
                self._emit(video_id, run_start, run_len, frame_to_idx)
                run_start, run_len = frame_nums[i], 1
        self._emit(video_id, run_start, run_len, frame_to_idx)

    def _emit(self, video_id: str, run_start: int, run_len: int, frame_to_idx: dict) -> None:
        if run_len < self.sequence_length:
            return
        for off in range(0, run_len - self.sequence_length + 1, self.stride):
            start_frame = run_start + off
            self.sequences.append({
                "video_id": video_id,
                "start_frame": start_frame,
                "start_idx": frame_to_idx[start_frame],
            })

    # ───────────────────────── loading ─────────────────────────

    def _load(self, video_id: str) -> tuple[np.ndarray, np.ndarray]:
        if video_id in self._cache:
            self._cache_order.remove(video_id)
            self._cache_order.append(video_id)
            return self._cache[video_id]
        while len(self._cache) >= self.max_cache_size:
            del self._cache[self._cache_order.pop(0)]
        pt_path = self.latents_dir / f"{video_id}.pt"
        if pt_path.exists():
            data = torch.load(pt_path, weights_only=True)
            arr = (data["latents"].numpy(), data["frame_indices"].numpy())
        else:
            with np.load(self.latents_dir / f"{video_id}.npz") as data:
                arr = (data["latents"].copy(), data["frame_indices"].copy())
        self._cache[video_id] = arr
        self._cache_order.append(video_id)
        return arr

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        seq = self.sequences[idx]
        latents_arr, _ = self._load(seq["video_id"])
        s = seq["start_idx"]
        latents = torch.from_numpy(latents_arr[s:s + self.sequence_length].copy())
        return {
            "latents": latents,            # (T, C, H, W)
            "video_id": seq["video_id"],
            "start_frame": seq["start_frame"],
        }
