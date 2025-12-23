#!/usr/bin/env python3
"""Evaluate trained dynamics model on test sequences.

Tests rollout quality:
1. Given context frames, predict future frames
2. Decode latents back to pixels using tokenizer
3. Compare to ground truth

Usage:
    python scripts/eval_dynamics.py --dynamics-checkpoint checkpoints/dynamics_best.pt --tokenizer-checkpoint checkpoints/tokenizer_best.pt
    python scripts/eval_dynamics.py --dynamics-checkpoint checkpoints/dynamics_best.pt --tokenizer-checkpoint checkpoints/tokenizer_best.pt --num-samples 8
"""

import argparse
import json
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from ahriuwu.data import LatentSequenceDataset
from ahriuwu.models import (
    create_dynamics,
    create_tokenizer,
    DiffusionSchedule,
    psnr,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate dynamics model")
    parser.add_argument(
        "--dynamics-checkpoint",
        type=str,
        required=True,
        help="Path to dynamics model checkpoint",
    )
    parser.add_argument(
        "--tokenizer-checkpoint",
        type=str,
        required=True,
        help="Path to tokenizer checkpoint",
    )
    parser.add_argument(
        "--latents-dir",
        type=str,
        default="data/processed/latents",
        help="Directory containing pre-tokenized latents",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results/dynamics",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number of sample rollouts to generate",
    )
    parser.add_argument(
        "--context-frames",
        type=int,
        default=16,
        help="Number of context frames to condition on",
    )
    parser.add_argument(
        "--predict-frames",
        type=int,
        default=16,
        help="Number of frames to predict",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=64,
        help="Number of diffusion steps for sampling",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        help="Device to run evaluation on",
    )
    return parser.parse_args()


def load_dynamics(checkpoint_path: Path, device: str):
    """Load dynamics model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    args = checkpoint.get("args", {})

    model_size = args.get("model_size", "small")
    latent_dim = args.get("latent_dim", 256)

    model = create_dynamics(model_size, latent_dim=latent_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, checkpoint


def load_tokenizer(checkpoint_path: Path, device: str):
    """Load tokenizer from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    args = checkpoint.get("args", {})

    model_size = args.get("model_size", "small")

    model = create_tokenizer(model_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model


def rollout_predictions(
    dynamics: torch.nn.Module,
    schedule: DiffusionSchedule,
    context_latents: torch.Tensor,
    num_predict: int,
    num_steps: int,
    device: str,
) -> torch.Tensor:
    """Generate rollout predictions given context.

    Args:
        dynamics: Dynamics model
        schedule: Diffusion schedule
        context_latents: (B, T_ctx, C, H, W) context frames
        num_predict: Number of frames to predict
        num_steps: Diffusion sampling steps
        device: Device to run on

    Returns:
        predicted: (B, num_predict, C, H, W) predicted latents
    """
    B, T_ctx, C, H, W = context_latents.shape

    # Start with noise for frames to predict
    predicted = torch.randn(B, num_predict, C, H, W, device=device)

    # Simple single-step prediction for each frame
    # (More sophisticated: predict all at once with masked attention)
    with torch.no_grad():
        for frame_idx in range(num_predict):
            # Concatenate context with predicted so far
            if frame_idx == 0:
                full_seq = context_latents
            else:
                full_seq = torch.cat([context_latents, predicted[:, :frame_idx]], dim=1)

            # Predict next frame using diffusion sampling
            # Start from noise and denoise
            z_t = torch.randn(B, 1, C, H, W, device=device)

            step_size = 1.0 / num_steps
            for i in range(num_steps):
                tau = 1.0 - i * step_size
                tau_tensor = torch.full((B,), tau, device=device)

                # Concatenate context + current noisy frame
                input_seq = torch.cat([full_seq, z_t], dim=1)

                # Predict (model outputs full sequence, we take last frame)
                z_pred = dynamics(input_seq, tau_tensor)
                z_0_pred = z_pred[:, -1:, ...]

                # Euler step
                if i < num_steps - 1:
                    next_tau = tau - step_size
                    z_t = (1 - next_tau) * z_0_pred + next_tau * torch.randn_like(z_t)
                else:
                    z_t = z_0_pred

            predicted[:, frame_idx:frame_idx+1] = z_t

    return predicted


def latents_to_frames(
    tokenizer: torch.nn.Module,
    latents: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """Decode latents to pixel frames.

    Args:
        tokenizer: Tokenizer with decode method
        latents: (B, T, C, H, W) latents
        device: Device

    Returns:
        frames: (B, T, 3, 256, 256) decoded frames
    """
    B, T, C, H, W = latents.shape

    # Reshape to (B*T, C, H, W) for batched decoding
    latents_flat = latents.view(B * T, C, H, W)

    with torch.no_grad():
        frames_flat = tokenizer.decode(latents_flat)

    # Reshape back
    frames = frames_flat.view(B, T, 3, 256, 256)
    return frames


def create_comparison_video(
    gt_frames: torch.Tensor,
    pred_frames: torch.Tensor,
    output_path: Path,
    context_len: int,
):
    """Create side-by-side comparison image grid.

    Args:
        gt_frames: (T, 3, 256, 256) ground truth
        pred_frames: (T, 3, 256, 256) predictions
        output_path: Path to save
        context_len: Number of context frames (for labeling)
    """
    T = gt_frames.shape[0]

    # Select subset of frames for visualization
    frame_indices = list(range(0, T, max(1, T // 8)))[:8]

    # Create grid
    frame_h, frame_w = 256, 256
    padding = 4
    num_cols = len(frame_indices)

    grid_w = num_cols * frame_w + (num_cols + 1) * padding
    grid_h = 2 * frame_h + 3 * padding  # GT row + Pred row

    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 40  # Dark gray

    for col, t in enumerate(frame_indices):
        x = padding + col * (frame_w + padding)

        # Ground truth row
        y = padding
        gt = (gt_frames[t].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        grid[y:y+frame_h, x:x+frame_w] = gt

        # Prediction row
        y = 2 * padding + frame_h
        if t >= context_len:
            pred = (pred_frames[t - context_len].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            pred = np.clip(pred, 0, 255)
            grid[y:y+frame_h, x:x+frame_w] = pred
        else:
            # Context frame - just show black
            grid[y:y+frame_h, x:x+frame_w] = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(output_path)


def compute_rollout_metrics(
    dynamics: torch.nn.Module,
    tokenizer: torch.nn.Module,
    schedule: DiffusionSchedule,
    dataloader: DataLoader,
    args,
    device: str,
):
    """Compute metrics over dataset.

    Returns:
        metrics: Dict with average metrics
    """
    total_psnr = 0.0
    total_mse = 0.0
    num_samples = 0

    context_frames = args.context_frames
    predict_frames = args.predict_frames
    total_seq_len = context_frames + predict_frames

    for batch in dataloader:
        latents = batch["latents"][:, :total_seq_len].to(device)
        B = latents.shape[0]

        if latents.shape[1] < total_seq_len:
            continue

        # Split into context and target
        context = latents[:, :context_frames]
        target = latents[:, context_frames:total_seq_len]

        # Predict
        predicted = rollout_predictions(
            dynamics, schedule, context, predict_frames,
            args.num_steps, device
        )

        # Decode to pixels
        target_frames = latents_to_frames(tokenizer, target, device)
        pred_frames = latents_to_frames(tokenizer, predicted, device)

        # Compute PSNR
        for b in range(B):
            for t in range(predict_frames):
                p = psnr(pred_frames[b, t:t+1], target_frames[b, t:t+1]).item()
                m = torch.nn.functional.mse_loss(
                    pred_frames[b, t], target_frames[b, t]
                ).item()
                total_psnr += p
                total_mse += m
                num_samples += 1

        if num_samples >= args.num_samples * predict_frames:
            break

    return {
        "psnr": total_psnr / num_samples if num_samples > 0 else 0,
        "mse": total_mse / num_samples if num_samples > 0 else 0,
        "num_samples": num_samples,
    }


def generate_sample_rollouts(
    dynamics: torch.nn.Module,
    tokenizer: torch.nn.Module,
    schedule: DiffusionSchedule,
    dataloader: DataLoader,
    args,
    output_dir: Path,
    device: str,
):
    """Generate visual sample rollouts."""
    context_frames = args.context_frames
    predict_frames = args.predict_frames
    total_seq_len = context_frames + predict_frames

    sample_idx = 0
    for batch in dataloader:
        latents = batch["latents"][:, :total_seq_len].to(device)
        B = latents.shape[0]

        if latents.shape[1] < total_seq_len:
            continue

        context = latents[:, :context_frames]
        target = latents[:, context_frames:total_seq_len]

        # Predict
        predicted = rollout_predictions(
            dynamics, schedule, context, predict_frames,
            args.num_steps, device
        )

        # Decode
        full_gt_frames = latents_to_frames(tokenizer, latents, device)
        pred_frames = latents_to_frames(tokenizer, predicted, device)

        for b in range(B):
            if sample_idx >= args.num_samples:
                return

            output_path = output_dir / f"rollout_{sample_idx:03d}.png"
            create_comparison_video(
                full_gt_frames[b], pred_frames[b],
                output_path, context_frames
            )
            print(f"Saved rollout visualization to {output_path}")
            sample_idx += 1


def main():
    args = parse_args()

    dynamics_path = Path(args.dynamics_checkpoint)
    tokenizer_path = Path(args.tokenizer_checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Dynamics Model Evaluation")
    print("=" * 60)
    print(f"Dynamics checkpoint: {dynamics_path}")
    print(f"Tokenizer checkpoint: {tokenizer_path}")
    print(f"Device: {args.device}")
    print(f"Context frames: {args.context_frames}")
    print(f"Predict frames: {args.predict_frames}")
    print(f"Diffusion steps: {args.num_steps}")
    print("=" * 60)

    # Load models
    print("\nLoading models...")
    dynamics, dynamics_ckpt = load_dynamics(dynamics_path, args.device)
    tokenizer = load_tokenizer(tokenizer_path, args.device)

    epoch = dynamics_ckpt.get("epoch", "?")
    step = dynamics_ckpt.get("global_step", "?")
    print(f"Dynamics: epoch {epoch}, step {step}")
    print(f"Dynamics params: {dynamics.get_num_params():,}")
    print(f"Tokenizer params: {tokenizer.get_num_params():,}")

    # Create diffusion schedule
    schedule = DiffusionSchedule(device=args.device)

    # Load dataset
    print(f"\nLoading latents from {args.latents_dir}...")
    total_seq_len = args.context_frames + args.predict_frames
    dataset = LatentSequenceDataset(
        latents_dir=args.latents_dir,
        sequence_length=total_seq_len,
        stride=total_seq_len,  # Non-overlapping for evaluation
    )
    print(f"Found {len(dataset)} sequences")

    if len(dataset) == 0:
        print("ERROR: No sequences found!")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    # Compute metrics
    print("\nComputing rollout metrics...")
    metrics = compute_rollout_metrics(
        dynamics, tokenizer, schedule, dataloader, args, args.device
    )

    print(f"\n{'='*40}")
    print("Rollout Metrics")
    print("=" * 40)
    print(f"PSNR: {metrics['psnr']:.2f} dB")
    print(f"MSE:  {metrics['mse']:.6f}")
    print(f"Samples evaluated: {metrics['num_samples']}")
    print("=" * 40)

    # Quality interpretation
    if metrics["psnr"] >= 25:
        quality = "Good (>25 dB) - predictions maintain structure"
    elif metrics["psnr"] >= 20:
        quality = "Acceptable (20-25 dB) - some blurring"
    elif metrics["psnr"] >= 15:
        quality = "Poor (15-20 dB) - significant degradation"
    else:
        quality = "Very poor (<15 dB) - model needs more training"
    print(f"Quality: {quality}")

    # Save metrics
    metrics_path = output_dir / "rollout_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "dynamics_checkpoint": str(dynamics_path),
            "tokenizer_checkpoint": str(tokenizer_path),
            "context_frames": args.context_frames,
            "predict_frames": args.predict_frames,
            "num_steps": args.num_steps,
            **metrics,
        }, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    # Generate sample rollouts
    print("\nGenerating sample rollouts...")
    generate_sample_rollouts(
        dynamics, tokenizer, schedule, dataloader, args, output_dir, args.device
    )

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
