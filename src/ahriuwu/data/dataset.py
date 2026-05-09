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
    """Fixed-length contiguous frame windows for world-model pretraining."""

    def __init__(
        self,
        frames_dir: Path | str,
        sequence_length: int = 192,
        stride: int = 1,
        target_size: tuple[int, int] = TARGET_SIZE,
        file_ext: str = "jpg",
        transform=None,
    ):
        self.frames_dir = Path(frames_dir)
        self.sequence_length = sequence_length
        self.stride = stride
        self.target_size = target_size
        self.file_ext = file_ext
        self.transform = transform

        self.sequences: list[dict] = []
        for video_dir in self.frames_dir.iterdir():
            if not video_dir.is_dir():
                continue
            frames = sorted(video_dir.glob(f"frame_*.{self.file_ext}"))
            if len(frames) < self.sequence_length:
                continue
            for start_idx in range(0, len(frames) - self.sequence_length + 1, self.stride):
                self.sequences.append({
                    "video_id": video_dir.name,
                    "start_idx": start_idx,
                    "video_dir": video_dir,
                })

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        seq = self.sequences[idx]
        video_dir = seq["video_dir"]
        start_idx = seq["start_idx"]

        frames = []
        for i in range(self.sequence_length):
            path = video_dir / f"frame_{start_idx + i:06d}.{self.file_ext}"
            frame = cv2.imread(str(path))
            if frame is None:
                raise FileNotFoundError(f"Failed to load frame: {path}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)
            if self.transform:
                frame = self.transform(frame)
            else:
                frame = torch.from_numpy(frame).float() / 255.0
                frame = frame.permute(2, 0, 1)
            frames.append(frame)

        return {
            "frames": torch.stack(frames),
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
