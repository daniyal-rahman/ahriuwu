#!/usr/bin/env python3
"""Full rollout evaluation of dynamics model with iterative denoising."""

import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from ahriuwu.models import create_dynamics, create_tokenizer, DiffusionSchedule
from ahriuwu.data import LatentSequenceDataset


def main():
    device = "cuda"
    num_denoise_steps = 64
    context_frames = 4
    predict_frames = 12

    print("=" * 60)
    print("Full Dynamics Rollout Evaluation")
    print("=" * 60)
    print(f"Denoise steps: {num_denoise_steps}")
    print(f"Context frames: {context_frames}")
    print(f"Predict frames: {predict_frames}")

    print("\nLoading dynamics...")
    dynamics = create_dynamics("small", latent_dim=256).to(device)
    ckpt = torch.load("checkpoints/dynamics_step_0140000.pt", map_location=device, weights_only=False)
    dynamics.load_state_dict(ckpt["model_state_dict"])
    dynamics.eval()
    print(f"Loaded step {ckpt.get('global_step', 'unknown')}")

    print("\nLoading tokenizer...")
    tokenizer = create_tokenizer("small")
    tok_ckpt = torch.load("checkpoints/tokenizer_best.pt", map_location=device, weights_only=False)
    tokenizer.load_state_dict(tok_ckpt["model_state_dict"])
    tokenizer = tokenizer.to(device).eval()

    print("\nLoading test sequence...")
    dataset = LatentSequenceDataset("data/processed/latents", sequence_length=32, stride=500)
    sample = dataset[0]
    z_full = sample["latents"].unsqueeze(0).to(device)  # (1, 32, 256, 16, 16)

    # Split into context and ground truth
    z_context = z_full[:, :context_frames]  # (1, 4, 256, 16, 16)
    z_gt = z_full[:, context_frames:context_frames+predict_frames]  # (1, 12, 256, 16, 16)

    print(f"Context shape: {z_context.shape}")
    print(f"Ground truth shape: {z_gt.shape}")

    schedule = DiffusionSchedule(device=device)

    # Context tau - match training distribution (model never sees tau=0)
    context_tau = 0.1  # Small noise on context, like training

    print(f"\nGenerating {predict_frames} frames with {num_denoise_steps}-step denoising...")
    print(f"Context tau: {context_tau} (matching training distribution)")

    z_predicted = []
    z_current = z_context.clone()

    for frame_idx in tqdm(range(predict_frames), desc="Generating frames"):
        # Start from pure noise for the next frame
        z_next_noisy = torch.randn(1, 1, 256, 16, 16, device=device)

        # Iterative denoising
        for step in range(num_denoise_steps):
            tau_val = 1.0 - step / num_denoise_steps

            # Add small noise to context to match training distribution
            context_noise = torch.randn_like(z_current)
            z_context_noisy = (1 - context_tau) * z_current + context_tau * context_noise

            # Concatenate noisy context with noisy prediction
            z_input = torch.cat([z_context_noisy, z_next_noisy], dim=1)  # (1, T+1, 256, 16, 16)

            # Tau: context frames at context_tau, prediction frame at current noise level
            tau = torch.full((1, z_input.shape[1]), context_tau, device=device)
            tau[:, -1] = tau_val

            with torch.no_grad():
                z_pred = dynamics(z_input, tau)

            # Take prediction for the last frame
            z_next_pred = z_pred[:, -1:]

            # For next iteration, interpolate toward prediction
            if step < num_denoise_steps - 1:
                next_tau = 1.0 - (step + 1) / num_denoise_steps
                noise = torch.randn_like(z_next_pred)
                z_next_noisy = (1 - next_tau) * z_next_pred + next_tau * noise
            else:
                z_next_noisy = z_next_pred

        # Store prediction
        z_predicted.append(z_next_noisy)

        # Slide context window: drop oldest, add prediction
        z_current = torch.cat([z_current[:, 1:], z_next_noisy], dim=1)

    z_predicted = torch.cat(z_predicted, dim=1)  # (1, predict_frames, 256, 16, 16)

    # Compute metrics
    print("\nComputing metrics...")
    mse = ((z_predicted - z_gt) ** 2).mean().item()
    psnr_latent = 10 * np.log10(1 / (mse + 1e-8))
    print(f"Latent MSE: {mse:.6f}")
    print(f"Latent PSNR: {psnr_latent:.2f} dB")

    # Decode to pixels
    print("\nDecoding to pixels...")
    out_dir = Path("eval_results/dynamics_full")
    out_dir.mkdir(parents=True, exist_ok=True)

    frames_gt = []
    frames_pred = []

    with torch.no_grad():
        for i in range(predict_frames):
            gt_frame = tokenizer.decode(z_gt[:, i].float())
            pred_frame = tokenizer.decode(z_predicted[:, i].float())

            gt_np = (gt_frame[0].cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            pred_np = (pred_frame[0].cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

            frames_gt.append(gt_np)
            frames_pred.append(pred_np)

    # Save individual comparisons
    print("Saving comparison images...")
    for i in range(predict_frames):
        combined = np.concatenate([frames_gt[i], frames_pred[i]], axis=1)
        Image.fromarray(combined).save(out_dir / f"compare_frame{i:02d}.png")

    # Save as grid (4 columns)
    print("Saving grid...")
    cols = 4
    rows = (predict_frames + cols - 1) // cols

    grid_gt = []
    grid_pred = []

    for r in range(rows):
        row_gt = []
        row_pred = []
        for c in range(cols):
            idx = r * cols + c
            if idx < predict_frames:
                row_gt.append(frames_gt[idx])
                row_pred.append(frames_pred[idx])
            else:
                row_gt.append(np.zeros_like(frames_gt[0]))
                row_pred.append(np.zeros_like(frames_pred[0]))
        grid_gt.append(np.concatenate(row_gt, axis=1))
        grid_pred.append(np.concatenate(row_pred, axis=1))

    grid_gt = np.concatenate(grid_gt, axis=0)
    grid_pred = np.concatenate(grid_pred, axis=0)
    grid_combined = np.concatenate([grid_gt, grid_pred], axis=1)

    Image.fromarray(grid_gt).save(out_dir / "grid_gt.png")
    Image.fromarray(grid_pred).save(out_dir / "grid_pred.png")
    Image.fromarray(grid_combined).save(out_dir / "grid_combined.png")

    print(f"\nDone! Results saved to {out_dir}/")
    print(f"  grid_combined.png - GT (left half) vs Pred (right half)")
    print(f"  compare_frameXX.png - Individual frame comparisons")


if __name__ == "__main__":
    main()
