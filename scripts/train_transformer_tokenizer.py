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
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from ahriuwu.data.dataset import SingleFrameDataset
from ahriuwu.models import create_transformer_tokenizer, MAELoss, psnr, RunningRMS
from ahriuwu.utils.logging import add_wandb_args, init_wandb, log_step, log_images, finish_wandb
from ahriuwu.utils.training import add_training_args, create_optimizer, create_wsd_schedule, save_checkpoint, load_checkpoint

_preempt = threading.Event()


def _install_preemption_handler():
    """Install SIGTERM handler for graceful slurm preemption."""
    def _handler(signum, frame):
        if not _preempt.is_set():
            print("\nSIGTERM received — will save checkpoint and exit.", flush=True)
            _preempt.set()
    signal.signal(signal.SIGTERM, _handler)


def parse_args():
    parser = argparse.ArgumentParser(description="Train transformer tokenizer with MAE")
    add_training_args(parser)
    parser.add_argument(
        "--frames-dir",
        type=str,
        default="data/frames",
        help="Directory containing video subdirs with frames",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="small",
        choices=["tiny", "small", "medium", "large"],
        help="Model size preset",
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
        default=0.1,
        help="Minimum mask ratio when masking (default 0.1, avoids near-zero-but-nonzero masking)",
    )
    parser.add_argument(
        "--mask-ratio-max",
        type=float,
        default=0.9,
        help="Maximum mask ratio (paper uses 0.9)",
    )
    parser.add_argument(
        "--p-zero-mask",
        type=float,
        default=0.1,
        help="Probability of using mask_ratio=0.0 (full reconstruction, no masking)",
    )
    parser.add_argument(
        "--mask-warmup-steps",
        type=int,
        default=50000,
        help="Steps to linearly ramp mask_ratio_max from 0 to target (curriculum learning)",
    )
    parser.add_argument(
        "--step-save-interval",
        type=int,
        default=20000,
        help="Save checkpoint every N steps (0 to disable)",
    )
    parser.add_argument(
        "--sample-dir",
        type=str,
        default="samples/transformer_tokenizer",
        help="Directory to save sample reconstructions",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=1,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--use-rope",
        action="store_true",
        help="Use RoPE (Rotary Position Embeddings) instead of additive position embeddings",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Use gradient checkpointing for memory efficiency (recommended for batch>2 or model>100M)",
    )
    parser.set_defaults(batch_size=8, epochs=50, save_interval=5)
    add_wandb_args(parser)
    return parser.parse_args()


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
    loss_fn: MAELoss,
    rms_trackers: dict,
    device: str,
    epoch: int,
    global_step: int,
    args: argparse.Namespace,
    checkpoint_dir: Path = None,
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

        # Curriculum learning: ramp up mask_ratio_max over warmup steps
        if args.mask_warmup_steps > 0 and global_step < args.mask_warmup_steps:
            warmup_progress = global_step / args.mask_warmup_steps
            current_mask_max = args.mask_ratio_max * warmup_progress
        else:
            current_mask_max = args.mask_ratio_max

        # With p_zero_mask probability, use mask_ratio=0.0 (full reconstruction)
        # Otherwise sample from U(mask_ratio_min, current_max) avoiding near-zero masking
        if random.random() < args.p_zero_mask:
            mask_ratio = 0.0
        else:
            mask_ratio = random.uniform(args.mask_ratio_min, current_mask_max)

        # Mixed precision forward
        with autocast(device_type=device_type, dtype=dtype):
            output = model(frames, mask_ratio=mask_ratio)
            recon = output["reconstruction"]
            mask_indices = output.get("mask_indices", None)

            # Compute MAE loss (raw components)
            losses = loss_fn(recon, frames, mask_indices=mask_indices)

            # Normalize MSE and LPIPS separately via RunningRMS
            mse_norm = rms_trackers["mse"].update(losses["mse"])
            lpips_norm = rms_trackers["lpips"].update(losses["lpips"])
            loss = (mse_norm + args.lpips_weight * lpips_norm) / accumulation_steps

        # Backward with scaling
        scaler.scale(loss).backward()

        # Gradient accumulation
        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
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

        # Step-based checkpoint saving
        if checkpoint_dir and args.step_save_interval > 0 and global_step % args.step_save_interval == 0:
            step_path = checkpoint_dir / f"transformer_tokenizer_step_{global_step:07d}.pt"
            save_checkpoint(step_path, model, optimizer, scaler, epoch, global_step, losses["loss"].item(), args, scheduler=scheduler, rms_trackers=rms_trackers, extra={"model_type": "transformer_tokenizer"})
            # Also save as latest
            latest_path = checkpoint_dir / "transformer_tokenizer_latest.pt"
            save_checkpoint(latest_path, model, optimizer, scaler, epoch, global_step, losses["loss"].item(), args, scheduler=scheduler, rms_trackers=rms_trackers, extra={"model_type": "transformer_tokenizer"})

        # Preemption: save checkpoint immediately and break
        if _preempt.is_set() and checkpoint_dir:
            print(f"Preemption: saving checkpoint at step {global_step}...", flush=True)
            latest_path = checkpoint_dir / "transformer_tokenizer_latest.pt"
            save_checkpoint(latest_path, model, optimizer, scaler, epoch, global_step, losses["loss"].item(), args, scheduler=scheduler, rms_trackers=rms_trackers, extra={"model_type": "transformer_tokenizer"})
            print("Checkpoint saved. Exiting.", flush=True)
            return {
                "loss": total_loss / max(num_batches, 1),
                "mse": total_mse / max(num_batches, 1),
                "lpips": total_lpips / max(num_batches, 1),
                "psnr": total_psnr / max(num_batches, 1),
                "global_step": global_step,
                "preempted": True,
            }

        # Log
        if batch_idx % args.log_interval == 0:
            elapsed = time.time() - start_time
            samples_per_sec = (batch_idx + 1) * args.batch_size / elapsed
            mask_info = f"Mask: {mask_ratio:.2f}"
            if args.mask_warmup_steps > 0 and global_step < args.mask_warmup_steps:
                mask_info += f" (max: {current_mask_max:.2f})"
            current_lr = scheduler.get_last_lr()[0]

            log_step({
                "train/loss": losses["loss"].item(),
                "train/mse": losses["mse"].item(),
                "train/lpips": losses["lpips"].item(),
                "train/psnr": batch_psnr,
                "train/lr": current_lr,
                "train/mask_ratio": mask_ratio,
                "train/mask_max": current_mask_max,
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
                f"{mask_info} "
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
        "preempted": False,
    }


def main():
    args = parse_args()
    _install_preemption_handler()

    # Create run ID with timestamp
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("Transformer Tokenizer Training (MAE)")
    print("=" * 60)
    print(f"Run ID: {run_id}")
    print(f"Device: {args.device}")
    print(f"Model size: {args.model_size}")
    print(f"Use RoPE: {args.use_rope}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation}")
    print(f"Learning rate: {args.lr}")
    print(f"LPIPS weight: {args.lpips_weight}")
    print(f"Mask ratio: {args.p_zero_mask:.0%} p=0, else U({args.mask_ratio_min}, {args.mask_ratio_max})")
    print(f"Mask warmup steps: {args.mask_warmup_steps} (curriculum learning)")
    print(f"Step save interval: {args.step_save_interval}")
    print("=" * 60)

    # Create directories with run_id
    checkpoint_dir = Path(args.checkpoint_dir) / f"run_{run_id}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = Path(args.sample_dir) / f"run_{run_id}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Sample dir: {sample_dir}")

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
        prefetch_factor=4 if args.num_workers > 0 else None,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    # Create model
    model = create_transformer_tokenizer(
        args.model_size,
        use_rope=args.use_rope,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    model = model.to(args.device)
    print(f"Model parameters: {model.get_num_params():,}")
    print(f"Gradient checkpointing: {'ENABLED' if args.gradient_checkpointing else 'DISABLED'}")

    # torch.compile the model (but NOT the LPIPS/VGG loss model)
    if not args.no_compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Create loss function (NOT compiled - frozen LPIPS/VGG)
    loss_fn = MAELoss(mse_weight=1.0, lpips_weight=args.lpips_weight)
    loss_fn = loss_fn.to(args.device)

    # Create optimizer
    optimizer = create_optimizer(model.parameters(), args.lr, args.weight_decay, use_8bit=args.use_8bit_adam, betas=(0.9, 0.95))

    # WSD Learning Rate Schedule
    # Estimate total optimizer steps accounting for gradient accumulation
    steps_per_epoch = len(dataloader) // args.gradient_accumulation
    total_steps = args.epochs * steps_per_epoch
    scheduler = create_wsd_schedule(optimizer, total_steps, args.warmup_steps, args.decay_steps)

    # Create scaler for mixed precision
    device_type = args.device.split(":")[0]
    scaler = GradScaler(device_type)

    # Initialize wandb
    wandb_run = init_wandb(args, job_type="transformer_tokenizer", extra_config={
        "num_params": model.get_num_params() if not hasattr(model, '_orig_mod') else model._orig_mod.get_num_params(),
        "checkpoint_dir": str(checkpoint_dir),
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
            Path(args.resume), model, optimizer, scaler, scheduler=scheduler, rms_trackers=rms_trackers
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
            args.device, epoch, global_step, args, checkpoint_dir
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
            checkpoint_path = checkpoint_dir / f"transformer_tokenizer_epoch_{epoch + 1:03d}.pt"
            save_checkpoint(checkpoint_path, model, optimizer, scaler, epoch, global_step, metrics["loss"], args, scheduler=scheduler, rms_trackers=rms_trackers, extra={"model_type": "transformer_tokenizer"})

            # Also save as latest
            latest_path = checkpoint_dir / "transformer_tokenizer_latest.pt"
            save_checkpoint(latest_path, model, optimizer, scaler, epoch, global_step, metrics["loss"], args, scheduler=scheduler, rms_trackers=rms_trackers, extra={"model_type": "transformer_tokenizer"})

        # Save best model
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            best_path = checkpoint_dir / "transformer_tokenizer_best.pt"
            save_checkpoint(best_path, model, optimizer, scaler, epoch, global_step, metrics["loss"], args, scheduler=scheduler, rms_trackers=rms_trackers, extra={"model_type": "transformer_tokenizer"})
            print(f"New best model saved (loss: {best_loss:.4f})")

        if metrics.get("preempted"):
            print("Preempted during epoch — exiting training loop.", flush=True)
            break

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

    finish_wandb()


if __name__ == "__main__":
    main()
