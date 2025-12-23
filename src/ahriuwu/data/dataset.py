"""PyTorch datasets for LoL frame sequences."""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Default target resolution for world model
TARGET_SIZE = (256, 256)


class SingleFrameDataset(Dataset):
    """Dataset of individual frames for tokenizer training.

    Loads frames from video directories, resizes to target size.
    """

    def __init__(
        self,
        frames_dir: Path | str,
        target_size: tuple[int, int] = TARGET_SIZE,
        file_ext: str = "jpg",
        transform=None,
    ):
        """Initialize dataset.

        Args:
            frames_dir: Directory containing video subdirs with frames
            target_size: (width, height) to resize frames to
            file_ext: Frame file extension (jpg or png)
            transform: Optional torchvision transform
        """
        self.frames_dir = Path(frames_dir)
        self.target_size = target_size
        self.file_ext = file_ext
        self.transform = transform

        # Index all frames
        self.frame_paths = []
        self._index_frames()

    def _index_frames(self):
        """Build index of all frames."""
        for video_dir in self.frames_dir.iterdir():
            if not video_dir.is_dir():
                continue

            frames = sorted(video_dir.glob(f"frame_*.{self.file_ext}"))
            self.frame_paths.extend(frames)

        print(f"Indexed {len(self.frame_paths)} frames from {self.frames_dir}")

    def __len__(self) -> int:
        return len(self.frame_paths)

    def __getitem__(self, idx: int) -> dict:
        frame_path = self.frame_paths[idx]

        # Load and resize
        frame = cv2.imread(str(frame_path))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)

        if self.transform:
            frame = self.transform(frame)
        else:
            # Default: normalize to [0, 1] and convert to tensor
            frame = torch.from_numpy(frame).float() / 255.0
            frame = frame.permute(2, 0, 1)  # HWC -> CHW

        return {
            "frame": frame,  # (C, H, W)
            "path": str(frame_path),
        }


class FrameSequenceDataset(Dataset):
    """Dataset of frame sequences for world model training.

    Each item is a sequence of consecutive frames.
    """

    def __init__(
        self,
        frames_dir: Path | str,
        sequence_length: int = 192,  # 9.6 seconds at 20 FPS
        stride: int = 1,
        target_size: tuple[int, int] = TARGET_SIZE,
        file_ext: str = "jpg",
        transform=None,
    ):
        """Initialize dataset.

        Args:
            frames_dir: Directory containing video subdirs with frames
            sequence_length: Number of frames per sequence
            stride: Step between sequence start indices
            target_size: (width, height) to resize frames to
            file_ext: Frame file extension (jpg or png)
            transform: Optional torchvision transform
        """
        self.frames_dir = Path(frames_dir)
        self.sequence_length = sequence_length
        self.stride = stride
        self.target_size = target_size
        self.file_ext = file_ext
        self.transform = transform

        # Index all videos and their frames
        self.sequences = []
        self._index_frames()

    def _index_frames(self):
        """Build index of all valid sequences."""
        for video_dir in self.frames_dir.iterdir():
            if not video_dir.is_dir():
                continue

            frames = sorted(video_dir.glob(f"frame_*.{self.file_ext}"))
            if len(frames) < self.sequence_length:
                continue

            video_id = video_dir.name
            num_frames = len(frames)

            # Create sequences with stride
            for start_idx in range(0, num_frames - self.sequence_length + 1, self.stride):
                self.sequences.append({
                    "video_id": video_id,
                    "start_idx": start_idx,
                    "video_dir": video_dir,
                })

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        seq_info = self.sequences[idx]
        video_dir = seq_info["video_dir"]
        start_idx = seq_info["start_idx"]

        frames = []
        for i in range(self.sequence_length):
            frame_path = video_dir / f"frame_{start_idx + i:06d}.{self.file_ext}"
            frame = cv2.imread(str(frame_path))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)

            if self.transform:
                frame = self.transform(frame)
            else:
                # Default: normalize to [0, 1] and convert to tensor
                frame = torch.from_numpy(frame).float() / 255.0
                frame = frame.permute(2, 0, 1)  # HWC -> CHW

            frames.append(frame)

        return {
            "frames": torch.stack(frames),  # (T, C, H, W)
            "video_id": seq_info["video_id"],
            "start_idx": start_idx,
        }


class FrameWithStateDataset(Dataset):
    """Dataset of frames with game state and rewards.

    For behavioral cloning and reward model training.
    """

    def __init__(
        self,
        frames_dir: Path | str,
        states_dir: Path | str,
        sequence_length: int = 192,
        stride: int = 1,
        target_size: tuple[int, int] = TARGET_SIZE,
        file_ext: str = "jpg",
        transform=None,
    ):
        """Initialize dataset.

        Args:
            frames_dir: Directory containing video subdirs with frames
            states_dir: Directory containing game state JSON files
            sequence_length: Number of frames per sequence
            stride: Step between sequence start indices
            target_size: (width, height) to resize frames to
            file_ext: Frame file extension (jpg or png)
            transform: Optional torchvision transform
        """
        self.frames_dir = Path(frames_dir)
        self.states_dir = Path(states_dir)
        self.sequence_length = sequence_length
        self.stride = stride
        self.target_size = target_size
        self.file_ext = file_ext
        self.transform = transform

        self.sequences = []
        self._index_frames()

    def _index_frames(self):
        """Build index of sequences that have both frames and states."""
        for video_dir in self.frames_dir.iterdir():
            if not video_dir.is_dir():
                continue

            video_id = video_dir.name
            states_file = self.states_dir / f"{video_id}_states.json"

            if not states_file.exists():
                continue

            frames = sorted(video_dir.glob(f"frame_*.{self.file_ext}"))
            states = json.loads(states_file.read_text())

            # Use minimum of frames and states
            num_items = min(len(frames), len(states))
            if num_items < self.sequence_length:
                continue

            for start_idx in range(0, num_items - self.sequence_length + 1, self.stride):
                self.sequences.append({
                    "video_id": video_id,
                    "start_idx": start_idx,
                    "video_dir": video_dir,
                    "states": states,
                })

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        seq_info = self.sequences[idx]
        video_dir = seq_info["video_dir"]
        start_idx = seq_info["start_idx"]
        states = seq_info["states"]

        frames = []
        gold = []
        cs = []
        player_health = []
        rewards = []

        for i in range(self.sequence_length):
            frame_idx = start_idx + i

            # Load frame
            frame_path = video_dir / f"frame_{frame_idx:06d}.{self.file_ext}"
            frame = cv2.imread(str(frame_path))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)

            if self.transform:
                frame = self.transform(frame)
            else:
                frame = torch.from_numpy(frame).float() / 255.0
                frame = frame.permute(2, 0, 1)

            frames.append(frame)

            # Load state
            state = states[frame_idx] if frame_idx < len(states) else {}
            gold.append(state.get("gold", 0) or 0)
            cs.append(state.get("cs", 0) or 0)
            player_health.append(state.get("player_health", 0.5) or 0.5)
            rewards.append(state.get("reward", 0.0) or 0.0)

        return {
            "frames": torch.stack(frames),
            "gold": torch.tensor(gold, dtype=torch.float32),
            "cs": torch.tensor(cs, dtype=torch.float32),
            "player_health": torch.tensor(player_health, dtype=torch.float32),
            "rewards": torch.tensor(rewards, dtype=torch.float32),
            "video_id": seq_info["video_id"],
            "start_idx": start_idx,
        }


class LatentSequenceDataset(Dataset):
    """Dataset of pre-tokenized latent sequences for dynamics model training.

    Loads latent vectors from .pt files instead of raw frames.
    Much faster I/O than loading and resizing JPEGs.
    """

    def __init__(
        self,
        latents_dir: Path | str,
        sequence_length: int = 64,
        stride: int = 1,
    ):
        """Initialize dataset.

        Args:
            latents_dir: Directory containing video subdirs with latent .pt files
            sequence_length: Number of frames per sequence
            stride: Step between sequence start indices
        """
        self.latents_dir = Path(latents_dir)
        self.sequence_length = sequence_length
        self.stride = stride

        self.sequences = []
        self._index_latents()

    def _index_latents(self):
        """Build index of all valid sequences."""
        for video_dir in sorted(self.latents_dir.iterdir()):
            if not video_dir.is_dir():
                continue

            latent_files = sorted(video_dir.glob("latent_*.pt"))
            if len(latent_files) < self.sequence_length:
                continue

            video_id = video_dir.name

            # Extract frame numbers and sort
            frame_nums = []
            for f in latent_files:
                try:
                    num = int(f.stem.split("_")[1])
                    frame_nums.append(num)
                except (ValueError, IndexError):
                    continue

            frame_nums.sort()

            # Find contiguous sequences
            # We need sequence_length consecutive frames
            if not frame_nums:
                continue

            # Build list of start indices for contiguous sequences
            contiguous_start = frame_nums[0]
            contiguous_count = 1

            for i in range(1, len(frame_nums)):
                if frame_nums[i] == frame_nums[i - 1] + 1:
                    contiguous_count += 1
                else:
                    # End of contiguous block - add sequences
                    if contiguous_count >= self.sequence_length:
                        for start_offset in range(0, contiguous_count - self.sequence_length + 1, self.stride):
                            self.sequences.append({
                                "video_id": video_id,
                                "start_frame": contiguous_start + start_offset,
                                "video_dir": video_dir,
                            })
                    # Reset for new block
                    contiguous_start = frame_nums[i]
                    contiguous_count = 1

            # Handle last contiguous block
            if contiguous_count >= self.sequence_length:
                for start_offset in range(0, contiguous_count - self.sequence_length + 1, self.stride):
                    self.sequences.append({
                        "video_id": video_id,
                        "start_frame": contiguous_start + start_offset,
                        "video_dir": video_dir,
                    })

        print(f"Indexed {len(self.sequences)} latent sequences from {self.latents_dir}")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        seq_info = self.sequences[idx]
        video_dir = seq_info["video_dir"]
        start_frame = seq_info["start_frame"]

        latents = []
        for i in range(self.sequence_length):
            frame_num = start_frame + i
            latent_path = video_dir / f"latent_{frame_num:06d}.pt"
            latent = torch.load(latent_path, weights_only=True)
            latents.append(latent)

        return {
            "latents": torch.stack(latents),  # (T, C, H, W) = (T, 256, 16, 16)
            "video_id": seq_info["video_id"],
            "start_frame": start_frame,
        }
