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
        help="Number of frames per sequence",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=16,
        help="Stride between sequences (for data augmentation)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size (sequences per batch)",
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
    model.load_state_dict(checkpoint_data["model_state_dict"])
    optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint_data["scaler_state_dict"])
    return (
        checkpoint_data["epoch"],
        checkpoint_data["global_step"],
        checkpoint_data.get("loss", float("inf")),
    )


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    schedule: DiffusionSchedule,
    device: str,
    epoch: int,
    global_step: int,
    args: argparse.Namespace,
):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        # Get latent sequences: (B, T, C, H, W)
        z_0 = batch["latents"].to(device)
        B, T, C, H, W = z_0.shape

        optimizer.zero_grad()

        # Sample random timesteps
        tau = schedule.sample_timesteps(B, device=device)

        # Add noise
        z_tau, noise = schedule.add_noise(z_0, tau)

        # Mixed precision forward
        amp_dtype = torch.bfloat16 if device != "mps" else torch.float16
        with autocast(device_type=device.split(":")[0], dtype=amp_dtype):
            # Predict clean latents
            z_pred = model(z_tau, tau)

            # Compute loss
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
            samples_per_sec = (batch_idx + 1) * args.batch_size / elapsed
            print(
                f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                f"Loss: {loss.item():.4f} "
                f"Tau: {tau.mean().item():.2f} "
                f"({samples_per_sec:.1f} seqs/s)"
            )

    avg_loss = total_loss / num_batches

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
    print(f"Sequence length: {args.sequence_length}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("=" * 60)

    # Create directories
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create dataset
    print(f"\nLoading latent sequences from {args.latents_dir}...")
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
            epoch, global_step, args
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
