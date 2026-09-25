#!/usr/bin/env python3
"""Quick rollout test for transformer tokenizer — reconstructs video sequences."""

import torch
import numpy as np
from pathlib import Path
from PIL import Image
from ahriuwu.models import create_transformer_tokenizer
from ahriuwu.data.dataset import FrameSequenceDataset
from ahriuwu.utils.training import load_checkpoint
from torch.amp import autocast


def main():
    device = "cuda"
    checkpoint_path = Path("checkpoints/transformer_tokenizer_latest.pt")
    frames_dir = Path("/mnt/storage/ahriuwu-data/frames")
    out_dir = Path("eval_results/rollout_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading model...")
    model = create_transformer_tokenizer("medium", use_rope=True).to(device)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    step = ckpt.get("global_step", "?")
    epoch = ckpt.get("epoch", "?")
    print(f"Loaded checkpoint: epoch {epoch}, step {step}")

    # Load dataset
    print("Loading dataset...")
    dataset = FrameSequenceDataset(frames_dir, sequence_length=16, stride=64)
    print(f"Dataset: {len(dataset)} sequences")

    # Pick a few diverse samples
    np.random.seed(42)
    indices = np.random.choice(len(dataset), size=8, replace=False)

    psnrs = []
    with torch.no_grad():
        for sample_i, idx in enumerate(indices):
            batch = dataset[int(idx)]
            frames = batch["frames"].unsqueeze(0).to(device)  # (1, T, C, H, W)
            video_id = batch.get("video_id", "unknown")

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                # No masking — pure reconstruction
                output = model(frames, mask_ratio=0.0)
                recon = output["reconstruction"]

            # Compute PSNR
            mse = ((frames.float() - recon.float()) ** 2).mean().item()
            p = 10 * np.log10(1 / (mse + 1e-8))
            psnrs.append(p)

            # Save: top row = original, bottom row = reconstruction (all 16 frames)
            T = frames.shape[1]
            gt_frames = frames[0].cpu().float()       # (T, C, H, W)
            rc_frames = recon[0].cpu().float().clamp(0, 1)

            rows = []
            for t in range(T):
                gt_np = (gt_frames[t].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                rc_np = (rc_frames[t].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                rows.append(np.concatenate([gt_np, rc_np], axis=0))  # stack vertically

            # Concatenate all frames horizontally
            grid = np.concatenate(rows, axis=1)
            Image.fromarray(grid).save(out_dir / f"rollout_{sample_i}_{video_id}_psnr{p:.1f}.png")
            print(f"Sample {sample_i}: video={video_id} PSNR={p:.2f} dB")

    print(f"\nAverage PSNR: {np.mean(psnrs):.2f} dB")
    print(f"Saved {len(indices)} rollouts to {out_dir}/")


if __name__ == "__main__":
    main()
