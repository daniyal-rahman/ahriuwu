#!/usr/bin/env python3
"""Quick evaluation of transformer tokenizer reconstruction quality."""

import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from ahriuwu.models import create_transformer_tokenizer, psnr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--frames-dir", type=str, default="data/processed/frames")
    parser.add_argument("--output-dir", type=str, default="eval_results/transformer_tokenizer")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_random_frames(frames_dir: str, num_frames: int, device: str):
    """Load random frames for evaluation."""
    frames_path = Path(frames_dir)

    # Get all video directories
    video_dirs = [d for d in frames_path.iterdir() if d.is_dir()]

    frames = []
    paths = []

    for _ in range(num_frames):
        # Random video
        video_dir = np.random.choice(video_dirs)
        frame_files = sorted(list(video_dir.glob("*.png")) + list(video_dir.glob("*.jpg")))

        if len(frame_files) == 0:
            continue

        # Random frame
        frame_file = np.random.choice(frame_files)

        # Load and preprocess
        img = Image.open(frame_file).convert("RGB")
        img = img.resize((256, 256), Image.LANCZOS)

        frame = torch.from_numpy(np.array(img)).float() / 255.0
        frame = frame.permute(2, 0, 1)  # HWC -> CHW
        frames.append(frame)
        paths.append(str(frame_file))

    return torch.stack(frames).to(device), paths


def main():
    args = parse_args()

    print("=" * 60)
    print("Transformer Tokenizer Evaluation")
    print("=" * 60)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Get model size from checkpoint args
    ckpt_args = checkpoint.get("args", {})
    model_size = ckpt_args.get("model_size", "tiny")
    print(f"Model size: {model_size}")

    # Create model
    model = create_transformer_tokenizer(model_size)

    # Load weights (handle missing keys gracefully)
    state_dict = checkpoint["model_state_dict"]
    model_state = model.state_dict()

    for key in state_dict:
        if key in model_state:
            if model_state[key].shape == state_dict[key].shape:
                model_state[key] = state_dict[key]

    model.load_state_dict(model_state)
    model = model.to(args.device)
    model.eval()

    print(f"Parameters: {model.get_num_params():,}")

    # Load test frames
    print(f"\nLoading {args.num_samples} random frames...")
    frames, paths = load_random_frames(args.frames_dir, args.num_samples, args.device)
    print(f"Loaded {len(frames)} frames")

    # Evaluate reconstruction
    print("\nEvaluating reconstruction (no masking)...")
    with torch.no_grad():
        output = model(frames.unsqueeze(1), mask_ratio=0.0)  # Add time dim
        recon = output["reconstruction"].squeeze(1)  # Remove time dim

    # Compute metrics
    psnr_values = []
    for i in range(len(frames)):
        p = psnr(recon[i:i+1], frames[i:i+1]).item()
        psnr_values.append(p)
        print(f"  Frame {i}: PSNR = {p:.2f} dB")

    avg_psnr = np.mean(psnr_values)
    print(f"\nAverage PSNR: {avg_psnr:.2f} dB")

    # Save comparison images
    print(f"\nSaving comparison images to {output_dir}...")

    for i in range(min(4, len(frames))):
        # Original
        orig = (frames[i].cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

        # Reconstruction
        rec = (recon[i].cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

        # Side by side
        combined = np.concatenate([orig, rec], axis=1)

        save_path = output_dir / f"comparison_{i:02d}.png"
        Image.fromarray(combined).save(save_path)
        print(f"  Saved {save_path}")

    # Also test with masking (MAE style)
    print("\nEvaluating with 75% masking (MAE reconstruction)...")
    with torch.no_grad():
        output_masked = model(frames.unsqueeze(1), mask_ratio=0.75)
        recon_masked = output_masked["reconstruction"].squeeze(1)

    psnr_masked = []
    for i in range(len(frames)):
        p = psnr(recon_masked[i:i+1], frames[i:i+1]).item()
        psnr_masked.append(p)

    avg_psnr_masked = np.mean(psnr_masked)
    print(f"Average PSNR (75% masked): {avg_psnr_masked:.2f} dB")

    # Save masked comparison
    for i in range(min(4, len(frames))):
        orig = (frames[i].cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        rec = (recon_masked[i].cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        combined = np.concatenate([orig, rec], axis=1)

        save_path = output_dir / f"masked_comparison_{i:02d}.png"
        Image.fromarray(combined).save(save_path)

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  No masking PSNR: {avg_psnr:.2f} dB")
    print(f"  75% masked PSNR: {avg_psnr_masked:.2f} dB")
    print(f"  Checkpoint step: {checkpoint.get('global_step', 'unknown')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
