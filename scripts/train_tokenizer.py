#!/usr/bin/env python3
"""Train the vision tokenizer on extracted frames.

Usage:
    python scripts/train_tokenizer.py --frames-dir data/frames --epochs 10
    python scripts/train_tokenizer.py --resume checkpoints/tokenizer_latest.pt
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

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

from ahriuwu.data.dataset import SingleFrameDataset
from ahriuwu.models import create_tokenizer, TokenizerLoss, psnr, RunningRMS
from ahriuwu.utils.logging import add_wandb_args, init_wandb, log_step, log_images, finish_wandb


def parse_args():
    parser = argparse.ArgumentParser(description="Train vision tokenizer")
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
        default=16,
        help="Batch size for training",
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
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.1,
        help="Weight decay for AdamW",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=2000,
        help="Number of warmup steps for WSD LR schedule",
    )
    parser.add_argument(
        "--decay-steps",
        type=int,
        default=0,
        help="Number of decay steps for WSD LR schedule (0 = no decay, just warmup + hold)",
    )
    parser.add_argument(
        "--use-8bit-adam",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use 8-bit AdamW from bitsandbytes (default: True, --no-use-8bit-adam to disable)",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile",
    )
    parser.add_argument(
        "--lpips-weight",
        type=float,
        default=0.1,
        help="Weight for perceptual loss",
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
        default=100,
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
        "--sample-dir",
        type=str,
        default="samples/tokenizer",
        help="Directory to save sample reconstructions",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        help="Device to train on",
    )
    add_wandb_args(parser)
    return parser.parse_args()


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    scheduler,
    rms_trackers: dict,
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
        "scheduler_state_dict": scheduler.state_dict(),
        "rms_state": {k: v.state_dict() for k, v in rms_trackers.items()},
        "epoch": epoch,
        "global_step": global_step,
        "loss": loss,
        "args": vars(args),
    }
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, scaler: GradScaler, scheduler=None, rms_trackers: dict = None):
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if rms_trackers is not None and "rms_state" in checkpoint:
        for k, state in checkpoint["rms_state"].items():
            if k in rms_trackers:
                rms_trackers[k].load_state_dict(state)
    return checkpoint["epoch"], checkpoint["global_step"], checkpoint.get("loss", float("inf"))


def save_samples(model: nn.Module, dataloader: DataLoader, device: str, sample_dir: Path, epoch: int, num_samples: int = 4):
    """Save sample reconstructions for visual inspection."""
    import torchvision.utils as vutils

    model.eval()
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Get a batch
    batch = next(iter(dataloader))
    frames = batch["frame"][:num_samples].to(device)

    with torch.no_grad():
        with autocast(device_type=device.split(":")[0], dtype=torch.bfloat16 if device != "mps" else torch.float16):
            output = model(frames)
            recon = output["reconstruction"]

    # Concatenate original and reconstruction side by side
    comparison = torch.cat([frames, recon], dim=3)  # Side by side

    # Save grid
    grid = vutils.make_grid(comparison.cpu(), nrow=1, padding=2, normalize=False)
    save_path = sample_dir / f"epoch_{epoch:03d}.png"

    # Convert to PIL and save
    from PIL import Image
    import numpy as np

    grid_np = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img = Image.fromarray(grid_np)
    img.save(save_path)
    print(f"Saved samples to {save_path}")

    # Log to wandb
    log_images({"samples/reconstruction": img}, step=epoch)

    model.train()


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    scheduler,
    loss_fn: TokenizerLoss,
    rms_trackers: dict,
    device: str,
    epoch: int,
    global_step: int,
    args: argparse.Namespace,
):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_lpips = 0.0
    total_psnr = 0.0
    num_batches = 0

    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        frames = batch["frame"].to(device)

        optimizer.zero_grad()

        # Mixed precision forward
        with autocast(device_type=device.split(":")[0], dtype=torch.bfloat16 if device != "mps" else torch.float16):
            output = model(frames)
            recon = output["reconstruction"]
            losses = loss_fn(recon, frames)

            # Normalize MSE and LPIPS separately via RunningRMS
            mse_norm = rms_trackers["mse"].update(losses["mse"])
            lpips_norm = rms_trackers["lpips"].update(losses["lpips"])
            loss = mse_norm + args.lpips_weight * lpips_norm

        # Backward with scaling
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Track metrics
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
            current_lr = scheduler.get_last_lr()[0]

            log_step({
                "train/loss": losses["loss"].item(),
                "train/mse": losses["mse"].item(),
                "train/lpips": losses["lpips"].item(),
                "train/psnr": batch_psnr,
                "train/lr": current_lr,
                "train/samples_per_sec": samples_per_sec,
                "train/epoch": epoch,
            }, step=global_step)

            print(
                f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                f"Loss: {losses['loss'].item():.4f} "
                f"MSE: {losses['mse'].item():.4f} "
                f"LPIPS: {losses['lpips'].item():.4f} "
                f"PSNR: {batch_psnr:.2f} dB "
                f"LR: {current_lr:.2e} "
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
    print("Vision Tokenizer Training")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Model size: {args.model_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"LPIPS weight: {args.lpips_weight}")
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
    model = create_tokenizer(args.model_size)
    model = model.to(args.device)
    print(f"Model parameters: {model.get_num_params():,}")

    # torch.compile the model (but NOT the LPIPS/VGG loss model)
    if not args.no_compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Create loss function (NOT compiled - frozen LPIPS/VGG)
    loss_fn = TokenizerLoss(mse_weight=1.0, lpips_weight=args.lpips_weight)
    loss_fn = loss_fn.to(args.device)

    # Create optimizer
    if args.use_8bit_adam and HAS_BNB:
        print("Using 8-bit AdamW (bitsandbytes)")
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        if args.use_8bit_adam and not HAS_BNB:
            print("WARNING: bitsandbytes not installed, falling back to standard AdamW")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # WSD Learning Rate Schedule
    total_steps = args.epochs * len(dataloader)

    def wsd_schedule(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        if args.decay_steps > 0 and step >= total_steps - args.decay_steps:
            decay_progress = (total_steps - step) / max(1, args.decay_steps)
            return max(0.0, decay_progress)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=wsd_schedule)

    # Create scaler for mixed precision
    scaler = GradScaler(args.device.split(":")[0])

    # Initialize wandb
    wandb_run = init_wandb(args, job_type="tokenizer", extra_config={
        "num_params": model.get_num_params() if not hasattr(model, '_orig_mod') else model._orig_mod.get_num_params(),
    })

    # RMS trackers for loss normalization
    rms_trackers = {
        "mse": RunningRMS(),
        "lpips": RunningRMS(),
    }

    # Resume if checkpoint provided
    start_epoch = 0
    global_step = 0
    if args.resume:
        print(f"Resuming from {args.resume}...")
        start_epoch, global_step, _ = load_checkpoint(
            Path(args.resume), model, optimizer, scaler, scheduler, rms_trackers
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
            model, dataloader, optimizer, scaler, scheduler, loss_fn, rms_trackers,
            args.device, epoch, global_step, args
        )

        global_step = metrics["global_step"]

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  MSE: {metrics['mse']:.4f}")
        print(f"  LPIPS: {metrics['lpips']:.4f}")
        print(f"  PSNR: {metrics['psnr']:.2f} dB")

        log_step({
            "epoch/loss": metrics["loss"],
            "epoch/mse": metrics["mse"],
            "epoch/lpips": metrics["lpips"],
            "epoch/psnr": metrics["psnr"],
            "epoch/epoch": epoch + 1,
        }, step=global_step)

        # Save history
        history.append({
            "epoch": epoch + 1,
            **metrics,
        })

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = checkpoint_dir / f"tokenizer_epoch_{epoch + 1:03d}.pt"
            save_checkpoint(model, optimizer, scaler, scheduler, rms_trackers, epoch, global_step, metrics["loss"], args, checkpoint_path)

            # Also save as latest
            latest_path = checkpoint_dir / "tokenizer_latest.pt"
            save_checkpoint(model, optimizer, scaler, scheduler, rms_trackers, epoch, global_step, metrics["loss"], args, latest_path)

        # Save best model
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            best_path = checkpoint_dir / "tokenizer_best.pt"
            save_checkpoint(model, optimizer, scaler, scheduler, rms_trackers, epoch, global_step, metrics["loss"], args, best_path)
            print(f"New best model saved (loss: {best_loss:.4f})")

        # Save sample reconstructions
        save_samples(model, dataloader, args.device, sample_dir, epoch + 1)

    # Save training history
    history_path = checkpoint_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Best model: {checkpoint_dir / 'tokenizer_best.pt'}")
    print("=" * 60)

    finish_wandb()


if __name__ == "__main__":
    main()
