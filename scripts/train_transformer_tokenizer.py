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

from ahriuwu.data.dataset import FrameSequenceDataset
from ahriuwu.models import create_transformer_tokenizer, MAELoss, psnr, RunningRMS
from ahriuwu.utils.logging import add_wandb_args, init_wandb, log_step, log_images, finish_wandb
from ahriuwu.utils.training import (
    add_training_args, create_optimizer, create_wsd_schedule,
    save_checkpoint, load_checkpoint,
    PreemptionState, install_preemption_handlers, compute_dynamic_save_interval,
)

_preempt = PreemptionState()


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
        default="medium",
        choices=["tiny", "small", "medium", "large"],
        help="Model size preset (medium=~130M: dim=768, 8+8 layers, 12 heads)",
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
        default=15000,
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
        "--sequence-length",
        type=int,
        default=16,
        help="Number of consecutive frames per sequence (enables block-causal temporal attention)",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Use gradient checkpointing for memory efficiency (recommended for batch>2 or model>100M)",
    )
    parser.set_defaults(batch_size=2, epochs=50, save_interval=5)
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
    """Save sample reconstructions showing original, masked, and reconstructed.

    For sequence inputs (B, T, C, H, W), visualizes the middle frame of each sequence.
    """
    import torchvision.utils as vutils
    from PIL import Image
    import numpy as np

    model.eval()
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Get a batch
    batch = next(iter(dataloader))
    sequences = batch["frames"][:num_samples].to(device)  # (B, T, C, H, W)

    with torch.no_grad():
        device_type = device.split(":")[0]
        dtype = torch.bfloat16 if device_type == "cuda" else torch.float16

        with autocast(device_type=device_type, dtype=dtype):
            # Get reconstruction without masking for comparison
            output_no_mask = model(sequences, mask_ratio=0.0)
            recon_no_mask = output_no_mask["reconstruction"]  # (B, T, C, H, W)

            # Get reconstruction with masking (MAE style)
            output_masked = model(sequences, mask_ratio=0.75)
            recon_masked = output_masked["reconstruction"]  # (B, T, C, H, W)

    # Visualize the middle frame of each sequence
    mid = sequences.shape[1] // 2
    orig_frames = sequences[:, mid]  # (B, C, H, W)
    recon_no_mask_frames = recon_no_mask[:, mid]
    recon_masked_frames = recon_masked[:, mid]

    # Create side-by-side comparison: original | no-mask recon | masked recon
    comparison = torch.cat([orig_frames, recon_no_mask_frames, recon_masked_frames], dim=3)

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
    base_checkpoint_dir: Path = None,
):
    """Train for one epoch with MAE training."""
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_lpips = 0.0
    num_lpips_batches = 0
    total_psnr = 0.0
    num_batches = 0

    start_time = time.time()
    accumulation_steps = args.gradient_accumulation

    device_type = device.split(":")[0]
    dtype = torch.bfloat16 if device_type == "cuda" else torch.float16

    for batch_idx, batch in enumerate(dataloader):
        frames = batch["frames"].to(device)  # (B, T, C, H, W)

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

        # Generate mask outside torch.compile boundary
        base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        mask_indices = base_model.make_mask(
            frames.shape[0], frames.shape[1], mask_ratio, frames.device
        )

        # Mixed precision forward
        with autocast(device_type=device_type, dtype=dtype):
            output = model(frames, mask_indices=mask_indices)
            recon = output["reconstruction"]
            mask_indices = output.get("mask_indices", None)

            # Compute MAE loss — run LPIPS every 4th optimizer step only
            use_lpips = (global_step % 4 == 0)
            losses = loss_fn(recon, frames, mask_indices=mask_indices, skip_lpips=not use_lpips)

            # Normalize MSE and LPIPS separately via RunningRMS
            mse_norm = rms_trackers["mse"].update(losses["mse"])
            if use_lpips:
                lpips_norm = rms_trackers["lpips"].update(losses["lpips"])
                loss = (mse_norm + args.lpips_weight * lpips_norm) / accumulation_steps
            else:
                loss = mse_norm / accumulation_steps

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
            global_step += 1

        # Track metrics (unscaled loss)
        total_loss += losses["loss"].item()
        total_mse += losses["mse"].item()
        if use_lpips:
            total_lpips += losses["lpips"].item()
            num_lpips_batches += 1

        with torch.no_grad():
            # Flatten (B, T, C, H, W) -> (B*T, C, H, W) for PSNR
            recon_flat = recon.view(-1, *recon.shape[-3:]) if recon.dim() == 5 else recon
            frames_flat = frames.view(-1, *frames.shape[-3:]) if frames.dim() == 5 else frames
            batch_psnr = psnr(recon_flat, frames_flat).item()
            total_psnr += batch_psnr

        num_batches += 1

        # Compute effective save interval (dynamic or explicit)
        if args.step_save_interval > 0:
            effective_save_interval = args.step_save_interval
        else:
            effective_save_interval = compute_dynamic_save_interval(
                global_step, time.time() - start_time, args.checkpoint_minutes
            )

        # Step-based checkpoint saving (skip step 0 to avoid spam)
        save_at_boundary = (
            checkpoint_dir
            and global_step > 0
            and effective_save_interval > 0
            and global_step % effective_save_interval == 0
        )
        if save_at_boundary:
            step_path = checkpoint_dir / f"transformer_tokenizer_step_{global_step:07d}.pt"
            save_checkpoint(step_path, model, optimizer, scaler, epoch, global_step, losses["loss"].item(), args, scheduler=scheduler, rms_trackers=rms_trackers, extra={"model_type": "transformer_tokenizer"})
            # Also save as latest (both run dir and base dir for sbatch resume)
            latest_path = checkpoint_dir / "transformer_tokenizer_latest.pt"
            save_checkpoint(latest_path, model, optimizer, scaler, epoch, global_step, losses["loss"].item(), args, scheduler=scheduler, rms_trackers=rms_trackers, extra={"model_type": "transformer_tokenizer"})
            if base_checkpoint_dir and base_checkpoint_dir != checkpoint_dir:
                base_latest = base_checkpoint_dir / "transformer_tokenizer_latest.pt"
                save_checkpoint(base_latest, model, optimizer, scaler, epoch, global_step, losses["loss"].item(), args, scheduler=scheduler, rms_trackers=rms_trackers, extra={"model_type": "transformer_tokenizer"})

        # Preemption: save and exit on SIGTERM (Slurm preemption/time limit) or SIGUSR1 (manual)
        if _preempt.should_save_now() and checkpoint_dir and global_step > 0:
            if not save_at_boundary:
                latest_path = checkpoint_dir / "transformer_tokenizer_latest.pt"
                save_checkpoint(latest_path, model, optimizer, scaler, epoch, global_step, losses["loss"].item(), args, scheduler=scheduler, rms_trackers=rms_trackers, extra={"model_type": "transformer_tokenizer"})
                if base_checkpoint_dir and base_checkpoint_dir != checkpoint_dir:
                    base_latest = base_checkpoint_dir / "transformer_tokenizer_latest.pt"
                    save_checkpoint(base_latest, model, optimizer, scaler, epoch, global_step, losses["loss"].item(), args, scheduler=scheduler, rms_trackers=rms_trackers, extra={"model_type": "transformer_tokenizer"})
            print(f"Preempted at step {global_step}. Exiting.", flush=True)
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

            log_dict = {
                "train/loss": losses["loss"].item(),
                "train/mse": losses["mse"].item(),
                "train/psnr": batch_psnr,
                "train/lr": current_lr,
                "train/mask_ratio": mask_ratio,
                "train/mask_max": current_mask_max,
                "train/samples_per_sec": samples_per_sec,
                "train/epoch": epoch,
            }
            if use_lpips:
                log_dict["train/lpips"] = losses["lpips"].item()
            log_step(log_dict, step=global_step)

            lpips_str = f"LPIPS: {losses['lpips'].item():.4f} " if use_lpips else ""
            print(
                f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                f"Loss: {losses['loss'].item():.4f} "
                f"MSE: {losses['mse'].item():.4f} "
                f"{lpips_str}"
                f"PSNR: {batch_psnr:.2f} dB "
                f"LR: {current_lr:.2e} "
                f"{mask_info} "
                f"({samples_per_sec:.1f} samples/s)"
            )

    # Epoch averages
    avg_loss = total_loss / num_batches
    avg_mse = total_mse / num_batches
    avg_lpips = total_lpips / max(num_lpips_batches, 1)
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
    install_preemption_handlers(_preempt)

    # Create run ID with timestamp
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("Transformer Tokenizer Training (MAE)")
    print("=" * 60)
    print(f"Run ID: {run_id}")
    print(f"Device: {args.device}")
    print(f"Model size: {args.model_size}")
    print(f"Use RoPE: {args.use_rope}")
    print(f"Sequence length: {args.sequence_length}")
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
    base_checkpoint_dir = Path(args.checkpoint_dir)
    base_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        # When resuming, reuse the same directory as the checkpoint
        checkpoint_path = Path(args.resume)
        if checkpoint_path.parent.name.startswith("run_"):
            checkpoint_dir = checkpoint_path.parent
        else:
            # Legacy checkpoint in flat directory - create new run dir
            checkpoint_dir = base_checkpoint_dir / f"run_{run_id}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
    else:
        checkpoint_dir = base_checkpoint_dir / f"run_{run_id}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    sample_dir = Path(args.sample_dir) / f"run_{run_id}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Sample dir: {sample_dir}")

    # Create dataset
    print(f"Loading dataset from {args.frames_dir}...")
    dataset = FrameSequenceDataset(
        args.frames_dir,
        sequence_length=args.sequence_length,
        stride=8,
    )
    print(f"Found {len(dataset)} sequences (T={args.sequence_length})")

    if len(dataset) == 0:
        print("ERROR: No sequences found! Check --frames-dir path and --sequence-length.")
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
            args.device, epoch, global_step, args, checkpoint_dir, base_checkpoint_dir
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

            # Also save as latest (both run dir and base dir for sbatch resume)
            latest_path = checkpoint_dir / "transformer_tokenizer_latest.pt"
            save_checkpoint(latest_path, model, optimizer, scaler, epoch, global_step, metrics["loss"], args, scheduler=scheduler, rms_trackers=rms_trackers, extra={"model_type": "transformer_tokenizer"})
            if base_checkpoint_dir != checkpoint_dir:
                base_latest = base_checkpoint_dir / "transformer_tokenizer_latest.pt"
                save_checkpoint(base_latest, model, optimizer, scaler, epoch, global_step, metrics["loss"], args, scheduler=scheduler, rms_trackers=rms_trackers, extra={"model_type": "transformer_tokenizer"})

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
