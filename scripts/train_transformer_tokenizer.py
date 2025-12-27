#!/usr/bin/env python3
"""Train the transformer tokenizer with MAE (Masked Autoencoder) training.

This implements DreamerV4 Section 3.1 tokenizer training:
- Block-causal transformer encoder/decoder
- MAE masking with p ~ U(0, 0.9)
- Loss: MSE + 0.2 * LPIPS

Usage:
    python scripts/train_transformer_tokenizer.py --frames-dir data/frames --epochs 50
    python scripts/train_transformer_tokenizer.py --resume checkpoints/transformer_tokenizer_latest.pt
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

from ahriuwu.data.dataset import SingleFrameDataset
from ahriuwu.models import create_transformer_tokenizer, MAELoss, psnr


def parse_args():
    parser = argparse.ArgumentParser(description="Train transformer tokenizer with MAE")
    parser.add_argument(
        "--frames-dir",
        type=str,
        default="data/frames",
        help="Directory containing video subdirs with frames",
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
        "--batch-size",
        type=int,
        default=8,
        help="Batch size (smaller due to transformer memory)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
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
        "--lpips-weight",
        type=float,
        default=0.2,
        help="Weight for LPIPS perceptual loss (paper uses 0.2)",
    )
    parser.add_argument(
        "--mask-ratio-min",
        type=float,
        default=0.0,
        help="Minimum mask ratio (paper uses 0)",
    )
    parser.add_argument(
        "--mask-ratio-max",
        type=float,
        default=0.9,
        help="Maximum mask ratio (paper uses 0.9)",
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
        default=5,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--sample-dir",
        type=str,
        default="samples/transformer_tokenizer",
        help="Directory to save sample reconstructions",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        help="Device to train on",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=1,
        help="Number of gradient accumulation steps",
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
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "loss": loss,
        "args": vars(args),
        "model_type": "transformer_tokenizer",
    }
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, scaler: GradScaler):
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return checkpoint["epoch"], checkpoint["global_step"], checkpoint.get("loss", float("inf"))


def save_samples(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    sample_dir: Path,
    epoch: int,
    num_samples: int = 4,
):
    """Save sample reconstructions showing original, masked, and reconstructed."""
    import torchvision.utils as vutils
    from PIL import Image
    import numpy as np

    model.eval()
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Get a batch
    batch = next(iter(dataloader))
    frames = batch["frame"][:num_samples].to(device)

    with torch.no_grad():
        device_type = device.split(":")[0]
        dtype = torch.bfloat16 if device_type == "cuda" else torch.float16

        with autocast(device_type=device_type, dtype=dtype):
            # Get reconstruction without masking for comparison
            output_no_mask = model(frames, mask_ratio=0.0)
            recon_no_mask = output_no_mask["reconstruction"]

            # Get reconstruction with masking (MAE style)
            output_masked = model(frames, mask_ratio=0.75)
            recon_masked = output_masked["reconstruction"]

    # Create side-by-side comparison: original | no-mask recon | masked recon
    comparison = torch.cat([frames, recon_no_mask, recon_masked], dim=3)

    # Save grid
    grid = vutils.make_grid(comparison.cpu(), nrow=1, padding=2, normalize=False)
    save_path = sample_dir / f"epoch_{epoch:03d}.png"

    grid_np = (grid.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(grid_np).save(save_path)
    print(f"Saved samples to {save_path}")

    model.train()


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    loss_fn: MAELoss,
    device: str,
    epoch: int,
    global_step: int,
    args: argparse.Namespace,
):
    """Train for one epoch with MAE training."""
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_lpips = 0.0
    total_psnr = 0.0
    num_batches = 0

    start_time = time.time()
    accumulation_steps = args.gradient_accumulation

    device_type = device.split(":")[0]
    dtype = torch.bfloat16 if device_type == "cuda" else torch.float16

    for batch_idx, batch in enumerate(dataloader):
        frames = batch["frame"].to(device)

        # Sample mask ratio from U(min, max) for each batch
        mask_ratio = random.uniform(args.mask_ratio_min, args.mask_ratio_max)

        # Mixed precision forward
        with autocast(device_type=device_type, dtype=dtype):
            output = model(frames, mask_ratio=mask_ratio)
            recon = output["reconstruction"]
            mask_indices = output.get("mask_indices", None)

            # Compute MAE loss
            losses = loss_fn(recon, frames, mask_indices=mask_indices)
            loss = losses["loss"] / accumulation_steps

        # Backward with scaling
        scaler.scale(loss).backward()

        # Gradient accumulation
        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Track metrics (unscaled loss)
        total_loss += losses["loss"].item()
        total_mse += losses["mse"].item()
        total_lpips += losses["lpips"].item()

        with torch.no_grad():
            batch_psnr = psnr(recon, frames).item()
            total_psnr += batch_psnr

        num_batches += 1
        global_step += 1

        # Log
        if batch_idx % args.log_interval == 0:
            elapsed = time.time() - start_time
            samples_per_sec = (batch_idx + 1) * args.batch_size / elapsed
            print(
                f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                f"Loss: {losses['loss'].item():.4f} "
                f"MSE: {losses['mse'].item():.4f} "
                f"LPIPS: {losses['lpips'].item():.4f} "
                f"PSNR: {batch_psnr:.2f} dB "
                f"Mask: {mask_ratio:.2f} "
                f"({samples_per_sec:.1f} samples/s)"
            )

    # Epoch averages
    avg_loss = total_loss / num_batches
    avg_mse = total_mse / num_batches
    avg_lpips = total_lpips / num_batches
    avg_psnr = total_psnr / num_batches

    return {
        "loss": avg_loss,
        "mse": avg_mse,
        "lpips": avg_lpips,
        "psnr": avg_psnr,
        "global_step": global_step,
    }


def main():
    args = parse_args()

    print("=" * 60)
    print("Transformer Tokenizer Training (MAE)")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Model size: {args.model_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation}")
    print(f"Learning rate: {args.lr}")
    print(f"LPIPS weight: {args.lpips_weight}")
    print(f"Mask ratio: U({args.mask_ratio_min}, {args.mask_ratio_max})")
    print("=" * 60)

    # Create directories
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = Path(args.sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Create dataset
    print(f"Loading dataset from {args.frames_dir}...")
    dataset = SingleFrameDataset(args.frames_dir)
    print(f"Found {len(dataset)} frames")

    if len(dataset) == 0:
        print("ERROR: No frames found! Check --frames-dir path.")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Create model
    model = create_transformer_tokenizer(args.model_size)
    model = model.to(args.device)
    print(f"Model parameters: {model.get_num_params():,}")

    # Create loss function
    loss_fn = MAELoss(mse_weight=1.0, lpips_weight=args.lpips_weight)
    loss_fn = loss_fn.to(args.device)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # Create scaler for mixed precision
    device_type = args.device.split(":")[0]
    scaler = GradScaler(device_type)

    # Resume if checkpoint provided
    start_epoch = 0
    global_step = 0
    if args.resume:
        print(f"Resuming from {args.resume}...")
        start_epoch, global_step, _ = load_checkpoint(
            Path(args.resume), model, optimizer, scaler
        )
        start_epoch += 1  # Start from next epoch
        print(f"Resuming from epoch {start_epoch}, step {global_step}")

    # Training loop
    best_loss = float("inf")
    history = []

    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print("=" * 60)

        metrics = train_epoch(
            model, dataloader, optimizer, scaler, loss_fn, args.device,
            epoch, global_step, args
        )

        global_step = metrics["global_step"]

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  MSE: {metrics['mse']:.4f}")
        print(f"  LPIPS: {metrics['lpips']:.4f}")
        print(f"  PSNR: {metrics['psnr']:.2f} dB")

        # Save history
        history.append({
            "epoch": epoch + 1,
            **metrics,
        })

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = checkpoint_dir / f"transformer_tokenizer_epoch_{epoch + 1:03d}.pt"
            save_checkpoint(model, optimizer, scaler, epoch, global_step, metrics["loss"], args, checkpoint_path)

            # Also save as latest
            latest_path = checkpoint_dir / "transformer_tokenizer_latest.pt"
            save_checkpoint(model, optimizer, scaler, epoch, global_step, metrics["loss"], args, latest_path)

        # Save best model
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            best_path = checkpoint_dir / "transformer_tokenizer_best.pt"
            save_checkpoint(model, optimizer, scaler, epoch, global_step, metrics["loss"], args, best_path)
            print(f"New best model saved (loss: {best_loss:.4f})")

        # Save sample reconstructions
        save_samples(model, dataloader, args.device, sample_dir, epoch + 1)

    # Save training history
    history_path = checkpoint_dir / "transformer_tokenizer_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Best model: {checkpoint_dir / 'transformer_tokenizer_best.pt'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
