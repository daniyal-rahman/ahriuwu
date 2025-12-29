#!/usr/bin/env python3
"""Quick evaluation of transformer tokenizer."""

import torch
from pathlib import Path
from PIL import Image
import numpy as np
from ahriuwu.models import create_transformer_tokenizer
from ahriuwu.data import SingleFrameDataset


def main():
    device = "cuda"

    print("Loading model...")
    model = create_transformer_tokenizer("small").to(device)
    ckpt = torch.load("checkpoints/transformer_tokenizer_latest.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    step = ckpt.get("global_step", "unknown")
    print(f"Loaded step {step}")

    print("Loading test images...")
    dataset = SingleFrameDataset("data/processed/frames")
    indices = [0, 1000, 5000, 10000]

    out_dir = Path("eval_results/transformer_tokenizer")
    out_dir.mkdir(parents=True, exist_ok=True)

    psnrs = []
    with torch.no_grad():
        for i, idx in enumerate(indices):
            frame = dataset[idx]["frame"].unsqueeze(0).to(device)

            # Reconstruct (no masking)
            output = model(frame, mask_ratio=0.0)
            recon = output["reconstruction"]

            # Also test with 75% masking
            output_masked = model(frame, mask_ratio=0.75)
            recon_masked = output_masked["reconstruction"]

            # Convert to images
            gt_np = (frame[0].cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            recon_np = (recon[0].cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            masked_np = (recon_masked[0].cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

            # Side by side: GT | Recon | Masked Recon
            combined = np.concatenate([gt_np, recon_np, masked_np], axis=1)
            Image.fromarray(combined).save(out_dir / f"compare_{i}.png")

            # Compute PSNR
            mse = ((frame - recon) ** 2).mean().item()
            psnr = 10 * np.log10(1 / (mse + 1e-8))
            psnrs.append(psnr)
            print(f"Sample {i}: PSNR={psnr:.2f} dB")

    print(f"\nAverage PSNR: {np.mean(psnrs):.2f} dB")
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
