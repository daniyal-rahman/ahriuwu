#!/usr/bin/env python3
"""Quick 1-step evaluation of dynamics model."""

import torch
import numpy as np
from pathlib import Path
from PIL import Image

from ahriuwu.models import create_dynamics, create_tokenizer, DiffusionSchedule
from ahriuwu.data import LatentSequenceDataset


def main():
    print("Loading dynamics...")
    dynamics = create_dynamics("small", latent_dim=256).cuda()
    ckpt = torch.load("checkpoints/dynamics_step_0140000.pt", map_location="cuda", weights_only=False)
    dynamics.load_state_dict(ckpt["model_state_dict"])
    dynamics.eval()
    print(f"Loaded step {ckpt.get('global_step', 'unknown')}")

    print("\nLoading tokenizer...")
    tokenizer = create_tokenizer("small")
    tok_ckpt = torch.load("checkpoints/tokenizer_best.pt", map_location="cuda", weights_only=False)
    tokenizer.load_state_dict(tok_ckpt["model_state_dict"])
    tokenizer = tokenizer.cuda().eval()
    print("Tokenizer loaded")

    print("\nLoading test sequence...")
    dataset = LatentSequenceDataset("data/processed/latents", sequence_length=32, stride=100)
    sample = dataset[0]
    z_0 = sample["latents"].unsqueeze(0).cuda()  # (1, 32, 256, 16, 16)
    print(f"Loaded sequence shape: {z_0.shape}")

    print("\nTesting 1-step prediction (tau=0.5)...")
    schedule = DiffusionSchedule(device="cuda")

    # Add noise at tau=0.5
    tau = torch.full((1, 32), 0.5, device="cuda")
    z_noisy, noise = schedule.add_noise(z_0, tau)

    with torch.no_grad():
        z_pred = dynamics(z_noisy, tau)

    # Check if predictions vary
    pred_std = z_pred.std().item()
    input_diff = (z_0[0, 0] - z_0[0, 16]).abs().mean().item()
    pred_diff = (z_pred[0, 0] - z_pred[0, 16]).abs().mean().item()

    print(f"\nPrediction stats:")
    print(f"  Pred std: {pred_std:.4f} (should be > 0.1)")
    print(f"  Input frame diff (0 vs 16): {input_diff:.4f}")
    print(f"  Pred frame diff (0 vs 16): {pred_diff:.4f} (should be similar)")

    # PSNR in latent space
    mse = ((z_pred - z_0) ** 2).mean().item()
    psnr = 10 * np.log10(1 / (mse + 1e-8))
    print(f"  Latent PSNR: {psnr:.2f} dB")

    # Check mode collapse: are different frames producing different predictions?
    if pred_diff < 0.001:
        print("\n⚠️  WARNING: Predictions may be mode-collapsed (frames look identical)")
    else:
        print("\n✓ Predictions vary between frames (no mode collapse)")

    # Decode first and last frame to pixels
    print("\nDecoding to pixels...")

    with torch.no_grad():
        # Ensure float32 for tokenizer decode
        z_0_f32 = z_0.float()
        z_pred_f32 = z_pred.float()

        # Decode ground truth
        recon_0 = tokenizer.decode(z_0_f32[:, 0])
        recon_16 = tokenizer.decode(z_0_f32[:, 16])

        # Decode predictions
        pred_0 = tokenizer.decode(z_pred_f32[:, 0])
        pred_16 = tokenizer.decode(z_pred_f32[:, 16])

    # Save comparison
    print("Saving comparison images...")
    out_dir = Path("eval_results/dynamics_quick")
    out_dir.mkdir(parents=True, exist_ok=True)

    def save_img(tensor, path):
        img = (tensor[0].cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img).save(path)

    save_img(recon_0, out_dir / "gt_frame0.png")
    save_img(pred_0, out_dir / "pred_frame0.png")
    save_img(recon_16, out_dir / "gt_frame16.png")
    save_img(pred_16, out_dir / "pred_frame16.png")

    # Side-by-side comparison
    def make_comparison(gt, pred, path):
        gt_np = (gt[0].cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        pred_np = (pred[0].cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        combined = np.concatenate([gt_np, pred_np], axis=1)
        Image.fromarray(combined).save(path)

    make_comparison(recon_0, pred_0, out_dir / "compare_frame0.png")
    make_comparison(recon_16, pred_16, out_dir / "compare_frame16.png")

    print(f"\nDone! Images saved to {out_dir}/")
    print("  compare_frame0.png  - GT (left) vs Pred (right) for frame 0")
    print("  compare_frame16.png - GT (left) vs Pred (right) for frame 16")


if __name__ == "__main__":
    main()
