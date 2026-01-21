"""PyTorch datasets for LoL frame sequences."""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .actions import ABILITY_KEYS, collate_actions

# Default target resolution for world model
TARGET_SIZE = (256, 256)


class SingleFrameDataset(Dataset):
    """Dataset of individual frames for tokenizer training.

    Loads frames from video directories, resizes to target size.
    Supports optional pre-cached .npy files for faster loading.
    """

    def __init__(
        self,
        frames_dir: Path | str,
        target_size: tuple[int, int] = TARGET_SIZE,
        file_ext: str = "jpg",
        transform=None,
        cache_dir: Path | str | None = None,
    ):
        """Initialize dataset.

        Args:
            frames_dir: Directory containing video subdirs with frames
            target_size: (width, height) to resize frames to
            file_ext: Frame file extension (jpg or png)
            transform: Optional torchvision transform
            cache_dir: Optional directory with pre-cached .npy files
                       (auto-detected as frames_dir/../frames_cache if exists)
        """
        self.frames_dir = Path(frames_dir)
        self.target_size = target_size
        self.file_ext = file_ext
        self.transform = transform

        # Cache disabled for stability - use raw JPEG loading
        self.cache_dir = None

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

    def _get_cache_path(self, frame_path: Path) -> Path | None:
        """Get the cache file path for a frame."""
        if self.cache_dir is None:
            return None
        # Cache structure mirrors frames structure: cache_dir/video_id/frame_XXXXXX.npy
        video_id = frame_path.parent.name
        cache_path = self.cache_dir / video_id / f"{frame_path.stem}.npy"
        return cache_path if cache_path.exists() else None

    def __len__(self) -> int:
        return len(self.frame_paths)

    def __getitem__(self, idx: int) -> dict:
        frame_path = self.frame_paths[idx]

        # Try loading from cache first
        cache_path = self._get_cache_path(frame_path)
        loaded_from_cache = False
        if cache_path is not None:
            try:
                # Load pre-cached uint8 numpy array
                frame = np.load(cache_path)
                frame = torch.from_numpy(frame).float() / 255.0
                frame = frame.permute(2, 0, 1)  # HWC -> CHW
                loaded_from_cache = True
            except (EOFError, ValueError):
                # Corrupted cache file, fall back to JPEG
                pass

        if not loaded_from_cache:
            # Fall back to loading from JPEG
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


class PackedLatentSequenceDataset(Dataset):
    """Dataset of packed latent sequences for dynamics model training.

    Loads latent vectors from packed .npz files (one per video) instead of
    individual per-frame .npy files. This dramatically reduces I/O overhead
    by eliminating file open/close operations and enabling sequential reads.

    Expected speedup: 10-50x compared to per-frame loading.

    Optionally loads action labels from features.json per video.
    """

    def __init__(
        self,
        latents_dir: Path | str,
        sequence_length: int = 64,
        stride: int = 1,
        load_actions: bool = False,
        features_dir: Path | str | None = None,
    ):
        """Initialize dataset.

        Args:
            latents_dir: Directory containing packed .npz files (video_id.npz)
            sequence_length: Number of frames per sequence
            stride: Step between sequence start indices
            load_actions: Whether to load action labels from features.json
            features_dir: Directory containing features.json per video
        """
        self.latents_dir = Path(latents_dir)
        self.sequence_length = sequence_length
        self.stride = stride
        self.load_actions = load_actions
        self.features_dir = Path(features_dir) if features_dir else self.latents_dir.parent.parent / "data" / "processed"

        # Cache for loaded video data: video_id -> (latents, frame_indices)
        self.video_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        # Action labels per video (if load_actions=True)
        self.action_labels: dict[str, list[dict]] = {}

        self.sequences = []
        self._index_packed_latents()

        # Load action labels after indexing
        if load_actions:
            self._load_action_labels()

    def _load_video(self, video_id: str) -> tuple[np.ndarray, np.ndarray]:
        """Load packed video data, using cache if available."""
        if video_id not in self.video_cache:
            npz_path = self.latents_dir / f"{video_id}.npz"
            data = np.load(npz_path)
            self.video_cache[video_id] = (data['latents'], data['frame_indices'])
        return self.video_cache[video_id]

    def _index_packed_latents(self):
        """Build index of all valid sequences from packed files."""
        npz_files = sorted(self.latents_dir.glob("*.npz"))

        for npz_path in npz_files:
            video_id = npz_path.stem

            # Load metadata (just frame indices, not full latents yet)
            data = np.load(npz_path)
            frame_indices = data['frame_indices']
            num_frames = len(frame_indices)

            if num_frames < self.sequence_length:
                continue

            # Build frame number to array index mapping
            frame_to_idx = {int(frame_indices[i]): i for i in range(num_frames)}

            # Find contiguous sequences in original frame numbers
            frame_nums = sorted(frame_to_idx.keys())

            contiguous_start = frame_nums[0]
            contiguous_count = 1

            for i in range(1, len(frame_nums)):
                if frame_nums[i] == frame_nums[i - 1] + 1:
                    contiguous_count += 1
                else:
                    # End of contiguous block - add sequences
                    if contiguous_count >= self.sequence_length:
                        for start_offset in range(0, contiguous_count - self.sequence_length + 1, self.stride):
                            start_frame = contiguous_start + start_offset
                            # Store array index for fast access
                            start_idx = frame_to_idx[start_frame]
                            self.sequences.append({
                                "video_id": video_id,
                                "start_frame": start_frame,
                                "start_idx": start_idx,  # Index into packed array
                            })
                    # Reset for new block
                    contiguous_start = frame_nums[i]
                    contiguous_count = 1

            # Handle last contiguous block
            if contiguous_count >= self.sequence_length:
                for start_offset in range(0, contiguous_count - self.sequence_length + 1, self.stride):
                    start_frame = contiguous_start + start_offset
                    start_idx = frame_to_idx[start_frame]
                    self.sequences.append({
                        "video_id": video_id,
                        "start_frame": start_frame,
                        "start_idx": start_idx,
                    })

        print(f"Indexed {len(self.sequences)} packed latent sequences from {self.latents_dir}")

    def _load_action_labels(self):
        """Load action labels from features.json for each video."""
        video_ids = set(seq["video_id"] for seq in self.sequences)
        loaded_count = 0

        for video_id in video_ids:
            features_path = self.features_dir / video_id / "features.json"
            if features_path.exists():
                with open(features_path) as f:
                    data = json.load(f)
                    self.action_labels[video_id] = data.get("frames", [])
                loaded_count += 1

        print(f"Loaded action labels for {loaded_count}/{len(video_ids)} videos")

    def _get_actions(self, video_id: str, start_frame: int) -> dict[str, torch.Tensor] | None:
        """Get action tensors for a sequence."""
        if video_id not in self.action_labels:
            return None

        labels = self.action_labels[video_id]
        actions = {
            'movement': [],
            **{k: [] for k in ABILITY_KEYS}
        }

        for t in range(start_frame, start_frame + self.sequence_length):
            if t < len(labels):
                entry = labels[t]
                actions['movement'].append(entry.get('movement_slice', 0))
                actions['Q'].append(int(entry.get('ability_q', False)))
                actions['W'].append(int(entry.get('ability_w', False)))
                actions['E'].append(int(entry.get('ability_e', False)))
                actions['R'].append(int(entry.get('ability_r', False)))
                actions['D'].append(int(entry.get('summoner_d', False)))
                actions['F'].append(int(entry.get('summoner_f', False)))
                actions['item'].append(int(entry.get('item_used', False)))
                actions['B'].append(int(entry.get('recall_b', False)))
            else:
                actions['movement'].append(0)
                for k in ABILITY_KEYS:
                    actions[k].append(0)

        return {k: torch.tensor(v, dtype=torch.long) for k, v in actions.items()}

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        seq_info = self.sequences[idx]
        video_id = seq_info["video_id"]
        start_frame = seq_info["start_frame"]
        start_idx = seq_info["start_idx"]

        # Load from packed file (cached)
        latents_array, _ = self._load_video(video_id)

        # Slice contiguous sequence - very fast!
        latents = latents_array[start_idx:start_idx + self.sequence_length]
        latents = torch.from_numpy(latents.copy())  # Copy to avoid numpy mmap issues

        result = {
            "latents": latents,  # (T, C, H, W)
            "video_id": video_id,
            "start_frame": start_frame,
        }

        if self.load_actions:
            actions = self._get_actions(video_id, start_frame)
            result["actions"] = actions

        return result


class LatentSequenceDataset(Dataset):
    """Dataset of pre-tokenized latent sequences for dynamics model training.

    Loads latent vectors from .npy files instead of raw frames.
    Much faster I/O than loading and resizing JPEGs.

    NOTE: For better performance, use PackedLatentSequenceDataset with packed
    .npz files created by scripts/pack_latents.py.

    Optionally loads action labels from features.json per video.
    """

    def __init__(
        self,
        latents_dir: Path | str,
        sequence_length: int = 64,
        stride: int = 1,
        load_actions: bool = False,
        features_dir: Path | str | None = None,
    ):
        """Initialize dataset.

        Args:
            latents_dir: Directory containing video subdirs with latent .npy files
            sequence_length: Number of frames per sequence
            stride: Step between sequence start indices
            load_actions: Whether to load action labels from features.json
            features_dir: Directory containing features.json per video (default: data/processed)
        """
        self.latents_dir = Path(latents_dir)
        self.sequence_length = sequence_length
        self.stride = stride
        self.load_actions = load_actions
        self.features_dir = Path(features_dir) if features_dir else self.latents_dir.parent.parent / "data" / "processed"

        # Action labels per video (if load_actions=True)
        self.action_labels: dict[str, list[dict]] = {}

        self.sequences = []
        self._index_latents()

        # Load action labels after indexing
        if load_actions:
            self._load_action_labels()

    def _index_latents(self):
        """Build index of all valid sequences."""
        for video_dir in sorted(self.latents_dir.iterdir()):
            if not video_dir.is_dir():
                continue

            latent_files = sorted(video_dir.glob("latent_*.npy"))
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

    def _load_action_labels(self):
        """Load action labels from features.json for each video."""
        video_ids = set(seq["video_id"] for seq in self.sequences)
        loaded_count = 0

        for video_id in video_ids:
            features_path = self.features_dir / video_id / "features.json"
            if features_path.exists():
                with open(features_path) as f:
                    data = json.load(f)
                    # features.json has {"frames": [...]} structure
                    self.action_labels[video_id] = data.get("frames", [])
                loaded_count += 1

        print(f"Loaded action labels for {loaded_count}/{len(video_ids)} videos")

    def _get_actions(self, video_id: str, start_frame: int) -> dict[str, torch.Tensor] | None:
        """Get action tensors for a sequence.

        Args:
            video_id: Video identifier
            start_frame: Starting frame index

        Returns:
            Dict with 'movement' and ability keys as (T,) tensors,
            or None if no action labels for this video.
        """
        if video_id not in self.action_labels:
            return None

        labels = self.action_labels[video_id]
        actions = {
            'movement': [],
            **{k: [] for k in ABILITY_KEYS}
        }

        for t in range(start_frame, start_frame + self.sequence_length):
            if t < len(labels):
                entry = labels[t]
                # Map features.json keys to our action space
                actions['movement'].append(entry.get('movement_slice', 0))
                actions['Q'].append(int(entry.get('ability_q', False)))
                actions['W'].append(int(entry.get('ability_w', False)))
                actions['E'].append(int(entry.get('ability_e', False)))
                actions['R'].append(int(entry.get('ability_r', False)))
                actions['D'].append(int(entry.get('summoner_d', False)))
                actions['F'].append(int(entry.get('summoner_f', False)))
                actions['item'].append(int(entry.get('item_used', False)))
                actions['B'].append(int(entry.get('recall_b', False)))
            else:
                # Padding with "no action"
                actions['movement'].append(0)
                for k in ABILITY_KEYS:
                    actions[k].append(0)

        return {k: torch.tensor(v, dtype=torch.long) for k, v in actions.items()}

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        seq_info = self.sequences[idx]
        video_dir = seq_info["video_dir"]
        start_frame = seq_info["start_frame"]
        video_id = seq_info["video_id"]

        latents = []
        for i in range(self.sequence_length):
            frame_num = start_frame + i
            latent_path = video_dir / f"latent_{frame_num:06d}.npy"
            latent = np.load(latent_path)
            latents.append(torch.from_numpy(latent))

        result = {
            "latents": torch.stack(latents),  # (T, C, H, W) = (T, 256, 16, 16)
            "video_id": video_id,
            "start_frame": start_frame,
        }

        # Add actions if loading them
        if self.load_actions:
            actions = self._get_actions(video_id, start_frame)
            result["actions"] = actions  # dict of (T,) tensors or None

        return result
