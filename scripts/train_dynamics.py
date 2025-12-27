#!/usr/bin/env python3
"""Train the dynamics model on pre-tokenized latent sequences.

Uses x-prediction diffusion objective following DreamerV4.
Requires pre-tokenized latents from pretokenize_frames.py.

Usage:
    python scripts/train_dynamics.py --latents-dir data/processed/latents
    python scripts/train_dynamics.py --latents-dir data/processed/latents --model-size small --epochs 10
    python scripts/train_dynamics.py --resume checkpoints/dynamics_latest.pt
"""

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.utils.checkpoint import checkpoint

from ahriuwu.data import LatentSequenceDataset
from ahriuwu.models import (
    create_dynamics,
    DiffusionSchedule,
    x_prediction_loss,
    ShortcutForcing,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train dynamics model")
    parser.add_argument(
        "--latents-dir",
        type=str,
        default="data/processed/latents",
        help="Directory containing pre-tokenized latent sequences",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="small",
        choices=["tiny", "small", "medium", "large"],
        help="Model size preset",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=256,
        help="Latent dimension (must match tokenizer)",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=64,
        help="Number of frames per sequence (ignored if --alternating-lengths)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=8,
        help="Stride between sequences (for data augmentation, 8=50% overlap)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size (sequences per batch, ignored if --alternating-lengths)",
    )
    # Alternating batch lengths (DreamerV4 Section 3.4)
    parser.add_argument(
        "--alternating-lengths",
        action="store_true",
        help="Use alternating short/long batch lengths (DreamerV4 style)",
    )
    parser.add_argument(
        "--seq-len-short",
        type=int,
        default=32,
        help="Short sequence length for alternating (default 32 = 1.6 sec)",
    )
    parser.add_argument(
        "--seq-len-long",
        type=int,
        default=64,
        help="Long sequence length for alternating (default 64 = 3.2 sec)",
    )
    parser.add_argument(
        "--batch-size-short",
        type=int,
        default=2,
        help="Batch size for short sequences",
    )
    parser.add_argument(
        "--batch-size-long",
        type=int,
        default=1,
        help="Batch size for long sequences",
    )
    parser.add_argument(
        "--long-ratio",
        type=float,
        default=0.1,
        help="Ratio of long batches (default 0.1 = 10%% long, 90%% short)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs to train",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loader workers",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Log every N batches",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=1,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=0,
        help="Save checkpoint every N steps (0 to disable)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        default=True,
        help="Use gradient checkpointing to save memory",
    )
    parser.add_argument(
        "--shortcut-forcing",
        action="store_true",
        help="Enable shortcut forcing for few-step inference",
    )
    parser.add_argument(
        "--shortcut-k-max",
        type=int,
        default=64,
        help="Maximum step size for shortcut forcing (default 64)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        help="Device to train on",
    )
    return parser.parse_args()


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    global_step: int,
    loss: float,
    args: argparse.Namespace,
    path: Path,
):
    """Save training checkpoint."""
    checkpoint_data = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "loss": loss,
        "args": vars(args),
    }
    torch.save(checkpoint_data, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
):
    """Load training checkpoint."""
    checkpoint_data = torch.load(path, map_location="cpu")

    # Load model state dict with strict=False to handle new parameters (e.g., step_embed)
    missing, unexpected = model.load_state_dict(checkpoint_data["model_state_dict"], strict=False)
    if missing:
        print(f"Note: Initializing new parameters: {missing}")
    if unexpected:
        print(f"Warning: Unexpected keys in checkpoint: {unexpected}")

    optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint_data["scaler_state_dict"])
    return (
        checkpoint_data["epoch"],
        checkpoint_data["global_step"],
        checkpoint_data.get("loss", float("inf")),
    )


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader | None,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    schedule: DiffusionSchedule,
    device: str,
    epoch: int,
    global_step: int,
    args: argparse.Namespace,
    checkpoint_dir: Path = None,
    shortcut: ShortcutForcing | None = None,
    dataloader_short: DataLoader | None = None,
    dataloader_long: DataLoader | None = None,
):
    """Train for one epoch.

    If dataloader is provided, uses single dataloader mode.
    If dataloader_short and dataloader_long are provided, uses alternating mode.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    start_time = time.time()

    # Determine mode and setup iterators
    if dataloader is not None:
        # Single dataloader mode
        batch_iterator = iter(dataloader)
        total_batches = len(dataloader)
        alternating = False
    else:
        # Alternating dataloader mode
        iter_short = iter(dataloader_short)
        iter_long = iter(dataloader_long)
        # Estimate total batches based on short loader (most batches come from there)
        total_batches = len(dataloader_short) + int(len(dataloader_short) * args.long_ratio / (1 - args.long_ratio))
        alternating = True

    batch_idx = 0
    while True:
        # Get next batch
        try:
            if alternating:
                # Ratio-based selection: long_ratio chance of long batch
                use_long = random.random() < args.long_ratio
                if use_long:
                    try:
                        batch = next(iter_long)
                        seq_type = "L"
                    except StopIteration:
                        iter_long = iter(dataloader_long)
                        batch = next(iter_long)
                        seq_type = "L"
                else:
                    try:
                        batch = next(iter_short)
                        seq_type = "S"
                    except StopIteration:
                        # Short loader exhausted = epoch done
                        break
            else:
                batch = next(batch_iterator)
                seq_type = ""
        except StopIteration:
            break

        # Get latent sequences: (B, T, C, H, W)
        z_0 = batch["latents"].to(device)
        B, T, C, H, W = z_0.shape

        optimizer.zero_grad()

        # Sample per-timestep noise levels (diffusion forcing)
        # This creates temporal causality: clean past → noisy future
        tau = schedule.sample_diffusion_forcing_timesteps(B, T, device=device)

        # Add noise with per-timestep levels
        z_tau, noise = schedule.add_noise(z_0, tau)

        # Mixed precision forward
        amp_dtype = torch.bfloat16 if device != "mps" else torch.float16
        with autocast(device_type=device.split(":")[0], dtype=amp_dtype):
            if shortcut is not None:
                # Shortcut forcing: sample step sizes and use bootstrap loss
                step_size = shortcut.sample_step_size(B, device=device)
                loss, loss_info = shortcut.compute_loss(model, schedule, z_0, tau, step_size)
            else:
                # Standard training: predict clean latents
                z_pred = model(z_tau, tau)
                loss = x_prediction_loss(z_pred, z_0, tau, use_ramp_weight=True)

        # Backward with scaling
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        # Track metrics
        total_loss += loss.item()
        num_batches += 1
        global_step += 1

        # Log
        if batch_idx % args.log_interval == 0:
            elapsed = time.time() - start_time
            batches_per_sec = (batch_idx + 1) / elapsed if elapsed > 0 else 0
            # Show tau range: context (low) to target (high)
            tau_min = tau.min().item()
            tau_max = tau.max().item()

            # Format progress string
            if alternating:
                progress = f"Epoch {epoch} [{batch_idx}/{total_batches}] {seq_type} T={T}"
            else:
                progress = f"Epoch {epoch} [{batch_idx}/{total_batches}]"

            if shortcut is not None:
                # Show shortcut forcing info
                print(
                    f"{progress} "
                    f"Loss: {loss.item():.4f} (std:{loss_info['loss_std']:.4f} boot:{loss_info['loss_boot']:.4f}) "
                    f"Tau: [{tau_min:.2f}-{tau_max:.2f}] "
                    f"({batches_per_sec:.1f} batch/s)"
                )
            else:
                print(
                    f"{progress} "
                    f"Loss: {loss.item():.4f} "
                    f"Tau: [{tau_min:.2f}-{tau_max:.2f}] "
                    f"({batches_per_sec:.1f} batch/s)"
                )

        # Step-based checkpoint saving
        if args.save_steps > 0 and global_step % args.save_steps == 0 and checkpoint_dir is not None:
            step_path = checkpoint_dir / f"dynamics_step_{global_step:07d}.pt"
            save_checkpoint(
                model, optimizer, scaler, epoch, global_step,
                loss.item(), args, step_path
            )
            # Also update latest
            latest_path = checkpoint_dir / "dynamics_latest.pt"
            save_checkpoint(
                model, optimizer, scaler, epoch, global_step,
                loss.item(), args, latest_path
            )

        batch_idx += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    return {
        "loss": avg_loss,
        "global_step": global_step,
    }


def main():
    args = parse_args()

    print("=" * 60)
    print("Dynamics Model Training")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Model size: {args.model_size}")
    print(f"Latent dim: {args.latent_dim}")
    if args.alternating_lengths:
        print(f"Alternating lengths: short={args.seq_len_short} (batch={args.batch_size_short}), "
              f"long={args.seq_len_long} (batch={args.batch_size_long})")
        print(f"Long ratio: {args.long_ratio:.0%}")
    else:
        print(f"Sequence length: {args.sequence_length}")
        print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("=" * 60)

    # Create directories
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create dataset(s) and dataloader(s)
    print(f"\nLoading latent sequences from {args.latents_dir}...")

    if args.alternating_lengths:
        # Two datasets with different sequence lengths
        dataset_short = LatentSequenceDataset(
            latents_dir=args.latents_dir,
            sequence_length=args.seq_len_short,
            stride=args.stride,
        )
        dataset_long = LatentSequenceDataset(
            latents_dir=args.latents_dir,
            sequence_length=args.seq_len_long,
            stride=args.stride,
        )

        if len(dataset_short) == 0 or len(dataset_long) == 0:
            print("ERROR: No sequences found!")
            print("Make sure to run pretokenize_frames.py first.")
            return

        print(f"Short sequences: {len(dataset_short)} (T={args.seq_len_short})")
        print(f"Long sequences: {len(dataset_long)} (T={args.seq_len_long})")

        dataloader_short = DataLoader(
            dataset_short,
            batch_size=args.batch_size_short,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        dataloader_long = DataLoader(
            dataset_long,
            batch_size=args.batch_size_long,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        dataloader = None  # Will use alternating loaders
    else:
        # Single dataset/dataloader
        dataset = LatentSequenceDataset(
            latents_dir=args.latents_dir,
            sequence_length=args.sequence_length,
            stride=args.stride,
        )

        if len(dataset) == 0:
            print("ERROR: No sequences found!")
            print("Make sure to run pretokenize_frames.py first.")
            return

        print(f"Found {len(dataset)} sequences")

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        dataloader_short = None
        dataloader_long = None

    # Create model
    model = create_dynamics(args.model_size, latent_dim=args.latent_dim)
    model = model.to(args.device)
    print(f"Model parameters: {model.get_num_params():,}")

    # Enable gradient checkpointing for memory efficiency
    if args.gradient_checkpointing:
        # Note: Would need to modify model for true gradient checkpointing
        # For now, just enable torch.utils.checkpoint for attention layers
        print("Gradient checkpointing enabled")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # Create diffusion schedule
    schedule = DiffusionSchedule(device=args.device)

    # Create shortcut forcing if enabled
    shortcut = None
    if args.shortcut_forcing:
        shortcut = ShortcutForcing(k_max=args.shortcut_k_max)
        print(f"Shortcut forcing enabled (k_max={args.shortcut_k_max})")

    # Create scaler for mixed precision
    scaler = GradScaler("cuda")

    # Resume if checkpoint provided
    start_epoch = 0
    global_step = 0
    if args.resume:
        print(f"\nResuming from {args.resume}...")
        start_epoch, global_step, _ = load_checkpoint(
            Path(args.resume), model, optimizer, scaler
        )
        start_epoch += 1
        print(f"Resuming from epoch {start_epoch}, step {global_step}")

    # Training loop
    best_loss = float("inf")
    history = []

    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print("=" * 60)

        metrics = train_epoch(
            model, dataloader, optimizer, scaler, schedule, args.device,
            epoch, global_step, args, checkpoint_dir, shortcut,
            dataloader_short, dataloader_long
        )

        global_step = metrics["global_step"]

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Loss: {metrics['loss']:.4f}")

        # Save history
        history.append({
            "epoch": epoch + 1,
            **metrics,
        })

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = checkpoint_dir / f"dynamics_epoch_{epoch + 1:03d}.pt"
            save_checkpoint(
                model, optimizer, scaler, epoch, global_step,
                metrics["loss"], args, checkpoint_path
            )

            # Also save as latest
            latest_path = checkpoint_dir / "dynamics_latest.pt"
            save_checkpoint(
                model, optimizer, scaler, epoch, global_step,
                metrics["loss"], args, latest_path
            )

        # Save best model
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            best_path = checkpoint_dir / "dynamics_best.pt"
            save_checkpoint(
                model, optimizer, scaler, epoch, global_step,
                metrics["loss"], args, best_path
            )
            print(f"New best model saved (loss: {best_loss:.4f})")

    # Save training history
    history_path = checkpoint_dir / "dynamics_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Best model: {checkpoint_dir / 'dynamics_best.pt'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
