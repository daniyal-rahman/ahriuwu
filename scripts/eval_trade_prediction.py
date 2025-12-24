#!/usr/bin/env python3
"""Evaluate dynamics model prediction during trading sequences.

Finds frames where health bars change and compares predicted vs actual frames.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from ahriuwu.models import create_tokenizer
from ahriuwu.models.dynamics import create_dynamics


def load_frame(frame_path: Path, size: tuple = (256, 256)) -> np.ndarray:
    """Load and resize a frame."""
    frame = cv2.imread(str(frame_path))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    return frame


def frame_to_tensor(frame: np.ndarray) -> torch.Tensor:
    """Convert numpy frame to tensor."""
    tensor = torch.from_numpy(frame).float() / 255.0
    tensor = tensor.permute(2, 0, 1)  # HWC -> CHW
    return tensor


def detect_health_change(frames: list[np.ndarray], threshold: float = 500) -> list[int]:
    """Detect frames where health bars change significantly.

    Health bars are in top portion of frame. Look for red/green pixel changes.
    """
    change_indices = []

    for i in range(1, len(frames)):
        # Look at top 15% of frame where health bars are
        h = frames[i].shape[0]
        top_region_curr = frames[i][:int(h*0.15), :, :]
        top_region_prev = frames[i-1][:int(h*0.15), :, :]

        # Calculate difference
        diff = np.abs(top_region_curr.astype(float) - top_region_prev.astype(float))
        total_diff = diff.sum()

        if total_diff > threshold:
            change_indices.append(i)

    return change_indices


def predict_next_frame(
    model,
    context_latents: torch.Tensor,
    num_denoise_steps: int = 4,
    tau_ctx: float = 0.1,
) -> torch.Tensor:
    """Predict the next frame using diffusion forcing.

    Args:
        model: Dynamics model
        context_latents: Clean context frames (B, T, C, H, W)
        num_denoise_steps: Number of denoising iterations (K=4 in paper)
        tau_ctx: Small noise level for context frames

    Returns:
        Predicted next frame latent (B, 1, C, H, W)
    """
    device = next(model.parameters()).device
    B, T, C, H, W = context_latents.shape

    # Start with pure noise for the next frame
    z_next = torch.randn(B, 1, C, H, W, device=device)

    # Denoising schedule for the target frame: 1.0 -> 0.0
    taus = torch.linspace(1.0, 0.0, num_denoise_steps + 1, device=device)

    with torch.no_grad():
        for i in range(num_denoise_steps):
            tau_target = taus[i]

            # Concatenate context (slightly noised) + current noisy prediction
            noise_ctx = torch.randn_like(context_latents)
            z_ctx_noisy = (1 - tau_ctx) * context_latents + tau_ctx * noise_ctx

            z_full = torch.cat([z_ctx_noisy, z_next], dim=1)  # (B, T+1, C, H, W)

            # Create per-timestep tau: context gets tau_ctx, target gets tau_target
            tau = torch.full((B, T + 1), tau_ctx, device=device)
            tau[:, -1] = tau_target

            # Predict clean frames
            z_pred = model(z_full, tau)

            # Extract prediction for next frame
            z_next_pred = z_pred[:, -1:]

            if i < num_denoise_steps - 1:
                # Re-noise for next iteration
                tau_next = taus[i + 1]
                noise = torch.randn_like(z_next_pred)
                z_next = (1 - tau_next) * z_next_pred + tau_next * noise
            else:
                z_next = z_next_pred

    return z_next


def main():
    parser = argparse.ArgumentParser(description="Evaluate trade prediction")
    parser.add_argument("--frames-dir", type=str, required=True, help="Directory with frames")
    parser.add_argument("--tokenizer-checkpoint", type=str, default="checkpoints/tokenizer_best.pt")
    parser.add_argument("--dynamics-checkpoint", type=str, default="checkpoints/dynamics_best.pt")
    parser.add_argument("--start-frame", type=int, default=2400, help="Start frame (2min at 20fps)")
    parser.add_argument("--end-frame", type=int, default=14400, help="End frame (12min at 20fps)")
    parser.add_argument("--context-length", type=int, default=8, help="Context frames for prediction")
    parser.add_argument("--predict-ahead", type=int, default=4, help="Frames to predict ahead")
    parser.add_argument("--output-dir", type=str, default="eval_results/trades")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of trade sequences to visualize")
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    print(f"Using device: {device}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer_ckpt = torch.load(args.tokenizer_checkpoint, map_location="cpu")
    tokenizer = create_tokenizer(tokenizer_ckpt.get("args", {}).get("model_size", "small"))
    tokenizer.load_state_dict(tokenizer_ckpt["model_state_dict"])
    tokenizer = tokenizer.to(device).eval()

    # Load dynamics model
    print("Loading dynamics model...")
    dynamics_ckpt = torch.load(args.dynamics_checkpoint, map_location="cpu")
    dynamics = create_dynamics("small")
    dynamics.load_state_dict(dynamics_ckpt["model_state_dict"])
    dynamics = dynamics.to(device).eval()

    # Load frames in range
    print(f"Loading frames {args.start_frame} to {args.end_frame}...")
    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    frame_paths = [p for p in frame_paths if args.start_frame <= int(p.stem.split("_")[1]) <= args.end_frame]

    # Sample every 10th frame to find trades faster
    sample_paths = frame_paths[::10]
    sample_frames = [load_frame(p) for p in sample_paths]

    print(f"Loaded {len(sample_frames)} sampled frames, detecting health changes...")

    # Find health change moments
    change_indices = detect_health_change(sample_frames, threshold=1000)
    print(f"Found {len(change_indices)} potential trade moments")

    if len(change_indices) == 0:
        print("No trades detected, using random samples instead")
        change_indices = [len(sample_frames) // 4, len(sample_frames) // 2, 3 * len(sample_frames) // 4]

    # Take first N trade moments
    trade_moments = change_indices[:args.num_samples]

    for trade_idx, sample_idx in enumerate(trade_moments):
        print(f"\nProcessing trade {trade_idx + 1}/{len(trade_moments)}...")

        # Get actual frame index
        actual_idx = args.start_frame + sample_idx * 10

        # Load context frames and future frames
        context_start = max(0, actual_idx - args.context_length)
        future_end = min(len(frame_paths) + args.start_frame, actual_idx + args.predict_ahead + 1)

        context_frames = []
        for i in range(context_start, actual_idx):
            fp = frames_dir / f"frame_{i:06d}.jpg"
            if fp.exists():
                context_frames.append(load_frame(fp))

        future_frames = []
        for i in range(actual_idx, future_end):
            fp = frames_dir / f"frame_{i:06d}.jpg"
            if fp.exists():
                future_frames.append(load_frame(fp))

        if len(context_frames) < 4 or len(future_frames) < 2:
            print(f"  Skipping - not enough frames")
            continue

        # Encode context to latents
        context_tensors = torch.stack([frame_to_tensor(f) for f in context_frames]).unsqueeze(0).to(device)
        B, T, C, H, W = context_tensors.shape

        with torch.no_grad():
            # Encode each frame
            context_latents = []
            for t in range(T):
                latent = tokenizer.encode(context_tensors[:, t])
                context_latents.append(latent)
            context_latents = torch.stack(context_latents, dim=1)  # (1, T, C, H, W)

            # Predict next frames autoregressively using diffusion forcing
            predicted_latents = []
            current_context = context_latents.clone()

            for step in range(min(args.predict_ahead, len(future_frames))):
                # Predict next frame using proper diffusion forcing
                z_next = predict_next_frame(
                    dynamics,
                    current_context,
                    num_denoise_steps=4,
                    tau_ctx=0.1,
                )
                predicted_latents.append(z_next[:, 0])

                # Shift context window: drop oldest, add prediction
                current_context = torch.cat([current_context[:, 1:], z_next], dim=1)

            # Decode predictions
            predicted_frames = []
            for latent in predicted_latents:
                decoded = tokenizer.decode(latent)
                decoded = decoded.squeeze(0).permute(1, 2, 0).cpu().numpy()
                decoded = (decoded * 255).clip(0, 255).astype(np.uint8)
                predicted_frames.append(decoded)

        # Create comparison visualization
        num_compare = min(len(future_frames), len(predicted_frames), 4)
        fig, axes = plt.subplots(3, num_compare, figsize=(4 * num_compare, 12))

        # Row 1: Last context frame (repeated for reference)
        for i in range(num_compare):
            axes[0, i].imshow(context_frames[-1])
            axes[0, i].set_title(f"Context (t={actual_idx-1})")
            axes[0, i].axis("off")

        # Row 2: Actual future frames
        for i in range(num_compare):
            axes[1, i].imshow(future_frames[i])
            axes[1, i].set_title(f"Actual (t+{i+1})")
            axes[1, i].axis("off")

        # Row 3: Predicted frames
        for i in range(num_compare):
            axes[2, i].imshow(predicted_frames[i])
            axes[2, i].set_title(f"Predicted (t+{i+1})")
            axes[2, i].axis("off")

        plt.suptitle(f"Trade Sequence {trade_idx + 1} (frame {actual_idx})", fontsize=14)
        plt.tight_layout()

        output_path = output_dir / f"trade_{trade_idx:02d}_frame_{actual_idx}.png"
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"  Saved: {output_path}")

    print(f"\nDone! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
