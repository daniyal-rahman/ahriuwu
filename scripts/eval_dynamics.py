#!/usr/bin/env python3
"""Evaluate trained dynamics model on test sequences.

Tests rollout quality:
1. Given context frames, predict future frames
2. Decode latents back to pixels using tokenizer
3. Compare to ground truth

Can filter for "interesting" sequences:
- Laning phase (2-12 min game time)
- Post-laning (12+ min)
- Fight sequences (health decreases) if OCR states available

Usage:
    python scripts/eval_dynamics.py --dynamics-checkpoint checkpoints/dynamics_best.pt --tokenizer-checkpoint checkpoints/tokenizer_best.pt
    python scripts/eval_dynamics.py --dynamics-checkpoint checkpoints/dynamics_best.pt --tokenizer-checkpoint checkpoints/tokenizer_best.pt --filter laning
    python scripts/eval_dynamics.py --dynamics-checkpoint checkpoints/dynamics_best.pt --tokenizer-checkpoint checkpoints/tokenizer_best.pt --filter fights
"""

import argparse
import json
import random
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Subset

from ahriuwu.data import LatentSequenceDataset
from ahriuwu.models import (
    create_dynamics,
    create_tokenizer,
    DiffusionSchedule,
    psnr,
)

# At 20 FPS
FPS = 20
LANING_START_MIN = 2
LANING_END_MIN = 12
LANING_START_FRAME = LANING_START_MIN * 60 * FPS  # 2400
LANING_END_FRAME = LANING_END_MIN * 60 * FPS  # 14400


def load_ocr_states(states_dir: Path) -> dict:
    """Load all OCR state files into a dict keyed by video_id.

    Returns:
        Dict mapping video_id -> list of (frame_index, garen_health_pct) tuples
    """
    states = {}
    states_dir = Path(states_dir)

    if not states_dir.exists():
        return states

    for state_file in states_dir.glob("*_states.json"):
        with open(state_file) as f:
            data = json.load(f)

        video_id = data.get("video_id")
        if not video_id:
            continue

        # Extract frame_index and health for each state
        frame_health = []
        for s in data.get("states", []):
            frame_idx = s.get("frame_index")
            health_pct = s.get("garen_health_pct")
            if frame_idx is not None and health_pct is not None:
                frame_health.append((frame_idx, health_pct))

        if frame_health:
            states[video_id] = sorted(frame_health, key=lambda x: x[0])

    return states


def find_fight_sequences(
    ocr_states: dict,
    sequence_length: int,
    health_drop_threshold: float = 0.1,
) -> list[tuple[str, int]]:
    """Find sequences where Garen's health drops significantly.

    Args:
        ocr_states: Dict from load_ocr_states
        sequence_length: Length of sequences to find
        health_drop_threshold: Minimum health drop to count as fight (0.1 = 10%)

    Returns:
        List of (video_id, start_frame) tuples where fights occur
    """
    fight_sequences = []

    for video_id, frame_health in ocr_states.items():
        if len(frame_health) < 2:
            continue

        # Find frames where health dropped
        for i in range(1, len(frame_health)):
            prev_frame, prev_health = frame_health[i - 1]
            curr_frame, curr_health = frame_health[i]

            health_drop = prev_health - curr_health

            if health_drop >= health_drop_threshold:
                # Found a fight! Use the frame before the drop as sequence start
                # Go back enough frames to have full context
                start_frame = max(0, prev_frame - sequence_length)
                fight_sequences.append((video_id, start_frame))

    return fight_sequences


def filter_dataset_indices(
    dataset: LatentSequenceDataset,
    filter_type: str,
    ocr_states: dict = None,
    sequence_length: int = 64,
) -> list[int]:
    """Get filtered indices based on filter type.

    Args:
        dataset: The latent sequence dataset
        filter_type: "none", "laning", "post-laning", or "fights"
        ocr_states: OCR state data (required for "fights" filter)
        sequence_length: Sequence length for fight detection

    Returns:
        List of valid dataset indices
    """
    if filter_type == "none":
        return list(range(len(dataset)))

    valid_indices = []

    # Access dataset's internal sequences list
    # Each entry has: video_id, start_frame, video_dir
    for idx, seq_info in enumerate(dataset.sequences):
        video_id = seq_info["video_id"]
        frame_num = seq_info["start_frame"]

        if filter_type == "laning":
            # Laning phase: 2-12 minutes
            if LANING_START_FRAME <= frame_num < LANING_END_FRAME:
                valid_indices.append(idx)

        elif filter_type == "post-laning":
            # Post-laning: 12+ minutes
            if frame_num >= LANING_END_FRAME:
                valid_indices.append(idx)

        elif filter_type == "fights":
            if ocr_states is None or len(ocr_states) == 0:
                print("WARNING: No OCR states loaded, cannot filter for fights")
                print("Run: python scripts/extract_ocr_states.py first")
                return list(range(len(dataset)))

            # Check if this sequence overlaps with a known fight
            if video_id in ocr_states:
                frame_health = ocr_states[video_id]
                seq_start = frame_num
                seq_end = frame_num + sequence_length

                # Check for health drops within this sequence
                for i in range(1, len(frame_health)):
                    prev_frame, prev_health = frame_health[i - 1]
                    curr_frame, curr_health = frame_health[i]

                    # Is this health change within our sequence?
                    if seq_start <= curr_frame < seq_end:
                        health_drop = prev_health - curr_health
                        if health_drop >= 0.1:  # 10% health drop
                            valid_indices.append(idx)
                            break

    return valid_indices


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
        default=48,
        help="Number of context frames to condition on (48 = 2.4 sec at 20 FPS)",
    )
    parser.add_argument(
        "--predict-frames",
        type=int,
        default=16,
        help="Number of frames to predict (16 = 0.8 sec at 20 FPS)",
    )
    parser.add_argument(
        "--filter",
        type=str,
        choices=["none", "laning", "post-laning", "fights"],
        default="none",
        help="Filter sequences: laning (2-12 min), post-laning (12+ min), fights (health drops)",
    )
    parser.add_argument(
        "--states-dir",
        type=str,
        default="data/processed/states",
        help="Directory containing OCR state files (for fight detection)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=64,
        help="Number of diffusion steps for sampling",
    )
    parser.add_argument(
        "--tau-ctx",
        type=float,
        default=0.1,
        help="Noise level for context frames (matches training, 0.1 recommended)",
    )
    parser.add_argument(
        "--use-shortcut",
        action="store_true",
        help="Use shortcut forcing for few-step inference (requires model trained with --shortcut-forcing)",
    )
    parser.add_argument(
        "--shortcut-k-max",
        type=int,
        default=64,
        help="k_max for shortcut forcing (must match training)",
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
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if missing:
        print(f"Warning: Missing keys in checkpoint (using random init): {missing}")
    if unexpected:
        print(f"Warning: Unexpected keys in checkpoint (ignored): {unexpected}")
    model = model.to(device)
    model.eval()

    return model, checkpoint


def load_tokenizer(checkpoint_path: Path, device: str):
    """Load tokenizer from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    args = checkpoint.get("args", {})

    model_size = args.get("model_size", "small")

    model = create_tokenizer(model_size)
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if missing:
        print(f"Warning: Missing keys in tokenizer checkpoint (using random init): {missing}")
    if unexpected:
        print(f"Warning: Unexpected keys in tokenizer checkpoint (ignored): {unexpected}")
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
    tau_ctx: float = 0.1,
    use_shortcut: bool = False,
    k_max: int = 64,
) -> torch.Tensor:
    """Generate rollout predictions given context.

    Args:
        dynamics: Dynamics model
        schedule: Diffusion schedule
        context_latents: (B, T_ctx, C, H, W) context frames
        num_predict: Number of frames to predict
        num_steps: Diffusion sampling steps
        device: Device to run on
        tau_ctx: Noise level for context frames (matches training, default 0.1)
        use_shortcut: Whether to use shortcut forcing for few-step inference
        k_max: Maximum step size for shortcut forcing (must match training)

    Returns:
        predicted: (B, num_predict, C, H, W) predicted latents
    """
    B, T_ctx, C, H, W = context_latents.shape

    # Add slight noise to context frames to match training distribution
    # During training, context frames have tau=0.1, so we do the same at inference
    # z_tau = (1 - tau) * z_0 + tau * noise  (from flow matching formulation)
    if tau_ctx > 0:
        context_noise = torch.randn_like(context_latents)
        context_latents = (1 - tau_ctx) * context_latents + tau_ctx * context_noise

    # Compute step size for shortcut forcing
    # If using 4-step sampling with k_max=64: step_size_int = 64/4 = 16
    step_size_int = k_max // num_steps if use_shortcut else None
    step_size_norm = torch.full((B,), step_size_int / k_max, device=device) if use_shortcut else None

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
                # Also add slight noise to predicted frames used as context
                pred_context = predicted[:, :frame_idx]
                if tau_ctx > 0:
                    pred_noise = torch.randn_like(pred_context)
                    pred_context = (1 - tau_ctx) * pred_context + tau_ctx * pred_noise
                full_seq = torch.cat([context_latents, pred_context], dim=1)

            # Predict next frame using diffusion sampling
            # Start from noise and denoise
            z_t = torch.randn(B, 1, C, H, W, device=device)

            # Stop at tau_ctx (0.1) not 0.0 - model never saw τ < 0.1 during training
            tau_start = 1.0
            tau_end = tau_ctx  # match training minimum
            step_size = (tau_start - tau_end) / num_steps

            for i in range(num_steps):
                tau = tau_start - i * step_size
                tau_tensor = torch.full((B,), tau, device=device)

                # Concatenate context + current noisy frame
                input_seq = torch.cat([full_seq, z_t], dim=1)

                # Predict (model outputs full sequence, we take last frame)
                if use_shortcut:
                    z_pred = dynamics(input_seq, tau_tensor, step_size=step_size_norm)
                else:
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
    batch_size: int = 8,
) -> torch.Tensor:
    """Decode latents to pixel frames.

    Args:
        tokenizer: Tokenizer with decode method
        latents: (B, T, C, H, W) latents
        device: Device
        batch_size: Decode batch size to avoid OOM

    Returns:
        frames: (B, T, 3, 256, 256) decoded frames
    """
    B, T, C, H, W = latents.shape

    # Reshape to (B*T, C, H, W) for batched decoding
    latents_flat = latents.reshape(B * T, C, H, W)

    # Decode in batches to avoid OOM
    frames_list = []
    with torch.no_grad():
        for i in range(0, B * T, batch_size):
            batch = latents_flat[i:i + batch_size].float()
            decoded = tokenizer.decode(batch)
            frames_list.append(decoded.cpu())  # Move to CPU to free GPU memory

    frames_flat = torch.cat(frames_list, dim=0).to(device)

    # Reshape back
    frames = frames_flat.reshape(B, T, 3, 256, 256)
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
            args.num_steps, device, tau_ctx=args.tau_ctx,
            use_shortcut=args.use_shortcut, k_max=args.shortcut_k_max
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
            args.num_steps, device, tau_ctx=args.tau_ctx,
            use_shortcut=args.use_shortcut, k_max=args.shortcut_k_max
        )

        # Decode
        full_gt_frames = latents_to_frames(tokenizer, latents, device)
        pred_frames = latents_to_frames(tokenizer, predicted, device)

        for b in range(B):
            if sample_idx >= args.num_samples:
                return

            # Get video info from batch
            video_id = batch["video_id"][b] if "video_id" in batch else "unknown"
            start_frame = batch["start_frame"][b].item() if "start_frame" in batch else 0
            game_time_sec = start_frame / FPS
            game_time_min = game_time_sec / 60

            output_path = output_dir / f"rollout_{sample_idx:03d}.png"
            create_comparison_video(
                full_gt_frames[b], pred_frames[b],
                output_path, context_frames
            )
            print(f"Saved: {output_path}")
            print(f"  Video: {video_id}, Frame: {start_frame} (~{game_time_min:.1f} min)")
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
    print(f"Context frames: {args.context_frames} ({args.context_frames/FPS:.1f} sec)")
    print(f"Predict frames: {args.predict_frames} ({args.predict_frames/FPS:.1f} sec)")
    print(f"Diffusion steps: {args.num_steps}")
    print(f"Context noise (tau_ctx): {args.tau_ctx}")
    if args.use_shortcut:
        print(f"Shortcut forcing: enabled (k_max={args.shortcut_k_max})")
    print(f"Filter: {args.filter}")
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

    # Load OCR states if needed for fight detection
    ocr_states = {}
    if args.filter == "fights":
        print(f"\nLoading OCR states from {args.states_dir}...")
        ocr_states = load_ocr_states(args.states_dir)
        print(f"Loaded OCR data for {len(ocr_states)} videos")

    # Load dataset
    print(f"\nLoading latents from {args.latents_dir}...")
    total_seq_len = args.context_frames + args.predict_frames
    dataset = LatentSequenceDataset(
        latents_dir=args.latents_dir,
        sequence_length=total_seq_len,
        stride=total_seq_len,  # Non-overlapping for evaluation
    )
    print(f"Found {len(dataset)} total sequences")

    if len(dataset) == 0:
        print("ERROR: No sequences found!")
        return

    # Apply filtering
    if args.filter != "none":
        print(f"\nFiltering for '{args.filter}' sequences...")
        valid_indices = filter_dataset_indices(
            dataset, args.filter, ocr_states, total_seq_len
        )
        print(f"Found {len(valid_indices)} matching sequences")

        if len(valid_indices) == 0:
            print("ERROR: No sequences match filter!")
            return

        # Random sample from valid indices
        random.shuffle(valid_indices)
        subset_indices = valid_indices[:min(len(valid_indices), args.num_samples * 10)]
        dataset = Subset(dataset, subset_indices)
        print(f"Using {len(dataset)} sequences for evaluation")

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
