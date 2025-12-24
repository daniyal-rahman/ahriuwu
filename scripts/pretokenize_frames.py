#!/usr/bin/env python3
"""Pre-tokenize all frames to latent vectors for faster dynamics training.

Converts 256×256 JPEG frames to 16×16×256 latent tensors using trained tokenizer.
This reduces I/O from ~115GB JPEGs to ~11GB latents (10x speedup).

Usage:
    python scripts/pretokenize_frames.py --checkpoint checkpoints/tokenizer_best.pt
    python scripts/pretokenize_frames.py --checkpoint checkpoints/tokenizer_best.pt --batch-size 64
    python scripts/pretokenize_frames.py --checkpoint checkpoints/tokenizer_best.pt --resume

Output:
    data/processed/latents/{video_id}/latent_{frame:06d}.npy
    Each file contains a (256, 16, 16) float16 numpy array
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from ahriuwu.models import create_tokenizer


class FrameDatasetForEncoding(Dataset):
    """Dataset that yields frames with their output paths for encoding."""

    def __init__(
        self,
        frames_dir: Path,
        output_dir: Path,
        target_size: tuple[int, int] = (256, 256),
        file_ext: str = "jpg",
        skip_existing: bool = False,
    ):
        self.frames_dir = Path(frames_dir)
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        self.file_ext = file_ext
        self.skip_existing = skip_existing

        self.frame_items = []
        self._index_frames()

    def _index_frames(self):
        """Build index of all frames to process."""
        for video_dir in sorted(self.frames_dir.iterdir()):
            if not video_dir.is_dir():
                continue

            video_id = video_dir.name
            output_video_dir = self.output_dir / video_id

            frames = sorted(video_dir.glob(f"frame_*.{self.file_ext}"))

            for frame_path in frames:
                # Extract frame number
                frame_num = int(frame_path.stem.split("_")[1])
                output_path = output_video_dir / f"latent_{frame_num:06d}.npy"

                # Skip if already exists and resume mode
                if self.skip_existing and output_path.exists():
                    continue

                self.frame_items.append({
                    "frame_path": frame_path,
                    "output_path": output_path,
                    "video_id": video_id,
                    "frame_num": frame_num,
                })

        print(f"Found {len(self.frame_items)} frames to process")

    def __len__(self) -> int:
        return len(self.frame_items)

    def __getitem__(self, idx: int) -> dict:
        item = self.frame_items[idx]

        # Load and preprocess frame
        frame = cv2.imread(str(item["frame_path"]))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)

        # Normalize to [0, 1] and convert to tensor
        frame = torch.from_numpy(frame).float() / 255.0
        frame = frame.permute(2, 0, 1)  # HWC -> CHW

        return {
            "frame": frame,
            "output_path": str(item["output_path"]),
            "video_id": item["video_id"],
        }


def load_tokenizer(checkpoint_path: Path, device: str):
    """Load trained tokenizer from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Get model size from checkpoint args
    args = checkpoint.get("args", {})
    model_size = args.get("model_size", "small")

    model = create_tokenizer(model_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model


def process_batch(model, batch, device):
    """Encode a batch of frames and save latents."""
    frames = batch["frame"].to(device)
    output_paths = batch["output_path"]

    with torch.no_grad():
        with torch.amp.autocast(device_type=device.split(":")[0], dtype=torch.float16):
            latents = model.encode(frames)

    # Save each latent as numpy (much smaller files than torch.save)
    latents = latents.cpu().numpy().astype(np.float16)
    for i, output_path in enumerate(output_paths):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, latents[i])


def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize frames to latents")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to tokenizer checkpoint",
    )
    parser.add_argument(
        "--frames-dir",
        type=str,
        default="data/processed/frames",
        help="Directory containing video subdirs with frames",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/latents",
        help="Directory to save latent tensors",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for encoding",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip frames that already have latents saved",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        help="Device to run encoding on",
    )
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    output_dir = Path(args.output_dir)
    checkpoint_path = Path(args.checkpoint)

    print("=" * 60)
    print("Frame Pre-tokenization")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Frames dir: {frames_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    print(f"Resume: {args.resume}")
    print("=" * 60)

    # Load tokenizer
    print("\nLoading tokenizer...")
    model = load_tokenizer(checkpoint_path, args.device)
    print(f"Loaded tokenizer with {model.get_num_params():,} parameters")

    # Create dataset
    print(f"\nIndexing frames from {frames_dir}...")
    dataset = FrameDatasetForEncoding(
        frames_dir=frames_dir,
        output_dir=output_dir,
        skip_existing=args.resume,
    )

    if len(dataset) == 0:
        print("No frames to process!")
        return

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Process all frames
    print(f"\nProcessing {len(dataset)} frames...")
    start_time = time.time()

    for batch in tqdm(dataloader, desc="Encoding"):
        process_batch(model, batch, args.device)

    elapsed = time.time() - start_time
    fps = len(dataset) / elapsed

    print("\n" + "=" * 60)
    print("Pre-tokenization Complete")
    print("=" * 60)
    print(f"Frames processed: {len(dataset)}")
    print(f"Time elapsed: {elapsed / 60:.1f} minutes")
    print(f"Speed: {fps:.1f} frames/sec")
    print(f"Output saved to: {output_dir}")

    # Calculate storage savings
    num_videos = len(list(output_dir.iterdir())) if output_dir.exists() else 0
    print(f"Videos processed: {num_videos}")

    # Estimate size
    sample_latent_size = 256 * 16 * 16 * 2  # float16 = 2 bytes
    total_size_mb = len(dataset) * sample_latent_size / (1024 * 1024)
    print(f"Estimated output size: {total_size_mb / 1024:.1f} GB")
    print("=" * 60)


if __name__ == "__main__":
    main()
