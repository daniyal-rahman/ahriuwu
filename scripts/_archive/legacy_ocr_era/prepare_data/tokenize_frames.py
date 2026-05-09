#!/usr/bin/env python3
"""Unified frame tokenization for both YouTube and replay sources.

Tokenizes 352x352 masked frames → latent tensors using the trained
transformer tokenizer. Handles both data sources identically since
masking and resizing are done upstream.

Output: packed .npz or .pt files per video, matching the dynamics
training pipeline's expected format.

Usage:
    # Tokenize YouTube frames
    python scripts/prepare_data/tokenize_frames.py \
        --frames-dir /mnt/storage/ahriuwu-data/frames/frames \
        --output-dir /mnt/storage/ahriuwu-data/latents/youtube \
        --checkpoint /mnt/storage/ahriuwu-data/checkpoints/transformer_tokenizer_latest.pt

    # Tokenize replay frames
    python scripts/prepare_data/tokenize_frames.py \
        --frames-dir /mnt/storage/ahriuwu-data/frames/replay_frames \
        --output-dir /mnt/storage/ahriuwu-data/latents/replays \
        --checkpoint /mnt/storage/ahriuwu-data/checkpoints/transformer_tokenizer_latest.pt

    # Both in one run
    python scripts/prepare_data/tokenize_frames.py \
        --frames-dir /mnt/storage/ahriuwu-data/frames/frames \
                     /mnt/storage/ahriuwu-data/frames/replay_frames \
        --output-dir /mnt/storage/ahriuwu-data/latents/all \
        --checkpoint /mnt/storage/ahriuwu-data/checkpoints/transformer_tokenizer_latest.pt
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class FrameDataset(Dataset):
    """Dataset that loads pre-extracted 352x352 frames from JPG files."""

    def __init__(self, video_dir: Path):
        self.frames = sorted(video_dir.glob("frame_*.jpg"))
        self.video_id = video_dir.name

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = cv2.imread(str(self.frames[idx]))
        if frame is None:
            raise FileNotFoundError(f"Failed to load: {self.frames[idx]}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Normalize to [0, 1] float tensor, (C, H, W)
        tensor = torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0
        return {
            "frame": tensor,
            "frame_idx": idx,
        }


def load_tokenizer(checkpoint_path: str, device: str):
    """Load tokenizer from checkpoint, auto-detecting legacy vs unified."""
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    tok_args = ckpt.get("args", {})
    model_size = tok_args.get("model_size", "medium")

    sd = ckpt["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}

    # Auto-detect: try unified first, fall back to legacy
    try:
        from ahriuwu.models import create_transformer_tokenizer
        tok = create_transformer_tokenizer(model_size).to(device).float()
        missing, unexpected = tok.load_state_dict(sd, strict=False)
        # If too many missing weights (not just buffers), it's a legacy checkpoint
        missing_weights = [k for k in missing if not any(
            x in k for x in ['inv_freq', 'positions', 'causal_mask', 'rope_indices', 'space_mask']
        )]
        if len(missing_weights) > 0:
            raise ValueError(f"Unified tokenizer: {len(missing_weights)} missing weights")
        print(f"Loaded unified tokenizer ({model_size})")
    except (ValueError, RuntimeError):
        from ahriuwu.models.transformer_tokenizer_legacy import create_transformer_tokenizer
        tok = create_transformer_tokenizer(model_size).to(device).float()
        tok.load_state_dict(sd, strict=False)
        print(f"Loaded legacy tokenizer ({model_size})")

    tok.eval()
    return tok


def tokenize_video(
    tokenizer,
    video_dir: Path,
    output_path: Path,
    device: str,
    batch_size: int = 16,
    num_workers: int = 4,
):
    """Tokenize all frames in a video directory → single .pt file.

    Output: {video_id}.pt containing:
        'latents': (N, latent_dim, 16, 16) float16 tensor
        'frame_indices': (N,) int32 tensor of frame numbers
    """
    dataset = FrameDataset(video_dir)
    if len(dataset) == 0:
        return 0

    dataloader = DataLoader(
        dataset, batch_size=batch_size,
        num_workers=num_workers, pin_memory=True,
    )

    all_latents = []
    all_indices = []

    with torch.no_grad():
        for batch in dataloader:
            frames = batch["frame"].to(device)  # (B, 3, 352, 352)
            indices = batch["frame_idx"]

            # Tokenizer expects (B, T, C, H, W) — add T=1 dim
            frames_5d = frames.unsqueeze(1)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = tokenizer.encode(frames_5d)
                latents = output["latent"]  # (B, 256, latent_dim)

            # Reshape: (B, 256, D) → (B, D, 16, 16)
            B = latents.shape[0]
            latent_dim = latents.shape[-1]
            latents = latents.view(B, 16, 16, latent_dim).permute(0, 3, 1, 2)

            all_latents.append(latents.half().cpu())
            all_indices.append(indices)

    latents_cat = torch.cat(all_latents, dim=0)  # (N, D, 16, 16)
    indices_cat = torch.cat(all_indices, dim=0).int()  # (N,)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "latents": latents_cat,
        "frame_indices": indices_cat,
    }, output_path)

    return len(dataset)


def main():
    parser = argparse.ArgumentParser(description="Tokenize frames to latents")
    parser.add_argument("--frames-dir", type=str, nargs="+", required=True,
                        help="One or more directories containing video subdirs with frame_*.jpg")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for .pt latent files")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Tokenizer checkpoint path")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-tokenized videos")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer(args.checkpoint, device)
    output_dir = Path(args.output_dir)

    # Collect all video directories from all frame sources
    video_dirs = []
    for frames_dir in args.frames_dir:
        frames_path = Path(frames_dir)
        subdirs = sorted([d for d in frames_path.iterdir() if d.is_dir()])
        video_dirs.extend(subdirs)
        print(f"Source: {frames_dir} → {len(subdirs)} videos")

    print(f"Total: {len(video_dirs)} videos to tokenize")

    total_frames = 0
    for video_dir in tqdm(video_dirs, desc="Tokenizing"):
        video_id = video_dir.name
        output_path = output_dir / f"{video_id}.pt"

        if args.resume and output_path.exists():
            continue

        n = tokenize_video(
            tokenizer, video_dir, output_path,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        total_frames += n

    print(f"\nDone. {total_frames} frames tokenized → {output_dir}")


if __name__ == "__main__":
    main()
