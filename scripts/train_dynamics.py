#!/usr/bin/env python3
"""Train the dynamics model on pre-tokenized latent sequences.

Uses x-prediction diffusion objective following DreamerV4.
Requires pre-tokenized latents from pretokenize_frames.py.

Usage:
    python scripts/train_dynamics.py --latents-dir data/processed/latents
    python scripts/train_dynamics.py --latents-dir data/processed/latents --model-size small --epochs 10
    python scripts/train_dynamics.py --resume checkpoints/dynamics_latest.pt

Checkpoint directories are automatically organized as:
    checkpoints/run_YYYYMMDD_HHMMSS_dynamics_{model_size}/
"""

import argparse
import json
import random
import signal
import threading
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from ahriuwu.data import LatentSequenceDataset, PackedLatentSequenceDataset
from ahriuwu.models import (
    create_dynamics,
    DiffusionSchedule,
    x_prediction_loss,
    ShortcutForcing,
    RunningRMS,
)
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


def create_run_directory(base_dir: Path, args: argparse.Namespace) -> Path:
    """Create a timestamped run directory with model info in the name.

    Format: run_YYYYMMDD_HHMMSS_dynamics_{tokenizer_type}{latent_dim}_{model_size}

    Example: run_20260119_143000_dynamics_cnn256_small
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tokenizer_type = getattr(args, 'tokenizer_type', 'unknown')
    run_name = f"run_{timestamp}_dynamics_{tokenizer_type}{args.latent_dim}_{args.model_size}"

    run_dir = base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_run_config(run_dir: Path, args: argparse.Namespace, model_params: int):
    """Save run configuration as a JSON manifest file."""
    config = {
        "run_type": "dynamics_training",
        "timestamp": datetime.now().isoformat(),
        "model": {
            "type": "dynamics_transformer",
            "size": args.model_size,
            "latent_dim": args.latent_dim,
            "num_parameters": model_params,
            "use_actions": args.use_actions,
            "use_agent_tokens": args.use_agent_tokens,
            "use_qk_norm": not args.no_qk_norm,
            "soft_cap": args.soft_cap if args.soft_cap > 0 else None,
            "num_register_tokens": args.num_register_tokens,
            "num_kv_heads": args.num_kv_heads,
        },
        "tokenizer": {
            "type": getattr(args, 'tokenizer_type', 'unknown'),
            "latent_dim": args.latent_dim,
        },
        "training": {
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "gradient_checkpointing": args.gradient_checkpointing,
            "shortcut_forcing": args.shortcut_forcing,
            "independent_frame_ratio": args.independent_frame_ratio,
            "gradient_accumulation": args.gradient_accumulation,
        },
        "data": {
            "latents_dir": str(args.latents_dir),
            "stride": args.stride,
        },
        "device": args.device,
    }

    # Add batch/sequence config based on mode
    if args.alternating_lengths:
        accum_short = args.gradient_accumulation_short or args.gradient_accumulation
        accum_long = args.gradient_accumulation_long or args.gradient_accumulation
        config["training"]["alternating_lengths"] = True
        config["training"]["seq_len_short"] = args.seq_len_short
        config["training"]["seq_len_long"] = args.seq_len_long
        config["training"]["batch_size_short"] = args.batch_size_short
        config["training"]["batch_size_long"] = args.batch_size_long
        config["training"]["long_ratio"] = args.long_ratio
        config["training"]["gradient_accumulation_short"] = accum_short
        config["training"]["gradient_accumulation_long"] = accum_long
    else:
        config["training"]["sequence_length"] = args.sequence_length
        config["training"]["batch_size"] = args.batch_size

    config_path = run_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved run config to {config_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train dynamics model")
    add_training_args(parser)
    parser.add_argument(
        "--latents-dir",
        type=str,
        default="data/processed/latents",
        help="Directory containing pre-tokenized latent sequences",
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
        default=32,
        help="Latent dimension per token (must match tokenizer: tiny=16, small=32, medium=48, large=64)",
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
        help="Stride between sequences (for data augmentation, 8=50%% overlap)",
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
        "--save-steps",
        type=int,
        default=0,
        help="Save checkpoint every N steps (0 to disable)",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use gradient checkpointing to save memory (--no-gradient-checkpointing to disable)",
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
    # Action conditioning
    parser.add_argument(
        "--use-actions",
        action="store_true",
        help="Enable action conditioning (requires features.json per video)",
    )
    parser.add_argument(
        "--features-dir",
        type=str,
        default="data/processed",
        help="Directory containing features.json per video",
    )
    # Agent tokens (for Phase 2+)
    parser.add_argument(
        "--use-agent-tokens",
        action="store_true",
        help="Enable agent tokens for policy/reward heads",
    )
    parser.add_argument(
        "--tokenizer-type",
        type=str,
        default="transformer",
        help="Type of tokenizer used to generate latents (for labeling runs)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom run name (overrides auto-generated name)",
    )
    parser.add_argument(
        "--packed",
        action="store_true",
        help="Use packed latent format (.npz files) for faster I/O",
    )
    # DreamerV4 stability features
    parser.add_argument(
        "--no-qk-norm",
        action="store_true",
        help="Disable QKNorm for attention stability (enabled by default)",
    )
    parser.add_argument(
        "--soft-cap",
        type=float,
        default=50.0,
        help="Attention logit soft cap value (0 to disable, default 50.0)",
    )
    parser.add_argument(
        "--num-register-tokens",
        type=int,
        default=8,
        help="Number of register tokens for information routing (0 to disable)",
    )
    parser.add_argument(
        "--num-kv-heads",
        type=int,
        default=None,
        help="Number of KV heads for GQA (None = same as Q heads, no GQA)",
    )
    parser.add_argument(
        "--independent-frame-ratio",
        type=float,
        default=0.3,
        help="Ratio of batches using independent frame mode (no temporal attention)",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=1,
        help="Number of batches to accumulate gradients over before stepping "
             "(used for non-alternating mode, and as default for alternating)",
    )
    parser.add_argument(
        "--gradient-accumulation-short",
        type=int,
        default=None,
        help="Gradient accumulation for short sequences in alternating mode "
             "(defaults to --gradient-accumulation)",
    )
    parser.add_argument(
        "--gradient-accumulation-long",
        type=int,
        default=None,
        help="Gradient accumulation for long sequences in alternating mode "
             "(defaults to --gradient-accumulation)",
    )
    add_wandb_args(parser)
    return parser.parse_args()


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader | None,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    scheduler,
    schedule: DiffusionSchedule,
    device: str,
    epoch: int,
    global_step: int,
    args: argparse.Namespace,
    checkpoint_dir: Path = None,
    shortcut: ShortcutForcing | None = None,
    dataloader_short: DataLoader | None = None,
    dataloader_long: DataLoader | None = None,
    use_actions: bool = False,
    independent_frame_ratio: float = 0.3,
    rms_dict: dict[str, RunningRMS] | None = None,
    accumulation_steps: int = 1,
    accumulation_steps_long: int | None = None,
):
    """Train for one epoch.

    If dataloader is provided, uses single dataloader mode.
    If dataloader_short and dataloader_long are provided, uses alternating mode.

    In alternating mode with per-length accumulation, each accumulation window
    processes batches of a single type (all short or all long) so the loss scaling
    is consistent within the window.
    """
    model.train()
    total_loss = 0.0
    total_grad_norm = 0.0
    total_grad_norm_count = 0
    total_pred_std = 0.0
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

    # Per-length accumulation: in alternating mode, each accumulation window
    # processes only one batch type (all short or all long) for consistent
    # loss scaling. At the start of each window (micro_count==0), we pick the
    # batch type and set current_accum accordingly.
    accum_short = accumulation_steps
    accum_long = accumulation_steps_long if accumulation_steps_long is not None else accumulation_steps
    micro_count = 0
    current_accum = accumulation_steps  # default for non-alternating
    use_long = False

    batch_idx = 0
    optimizer.zero_grad()
    while True:
        # At start of accumulation window, decide batch type
        if micro_count == 0 and alternating:
            use_long = random.random() < args.long_ratio
            current_accum = accum_long if use_long else accum_short

        # Get next batch
        try:
            if alternating:
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

        # Get actions if available
        actions = None
        if use_actions and "actions" in batch and batch["actions"] is not None:
            actions = {k: v.to(device) for k, v in batch["actions"].items()}

        # Sample per-timestep noise levels (diffusion forcing)
        # This creates temporal causality: clean past → noisy future
        tau = schedule.sample_diffusion_forcing_timesteps(B, T, device=device)

        # Add noise with per-timestep levels
        z_tau, noise = schedule.add_noise(z_0, tau)

        # Sample whether to use independent frames mode (30% by default)
        use_independent = random.random() < independent_frame_ratio

        # Mixed precision forward
        amp_dtype = torch.bfloat16 if device != "mps" else torch.float16
        with autocast(device_type=device.split(":")[0], dtype=amp_dtype):
            if shortcut is not None:
                # Shortcut forcing: sample step sizes and use bootstrap loss
                # Disable gradient checkpointing — autocast inserts hidden cast ops
                # that change the saved tensor list, so checkpoint recomputation
                # misaligns tensor indices and raises CheckpointError. ~2.5 GB extra
                # VRAM for the student forward pass, well within 16 GB headroom.
                gc_was_enabled = getattr(model, 'gradient_checkpointing', False)
                if gc_was_enabled:
                    model.gradient_checkpointing = False
                step_size = shortcut.sample_step_size(B, device=device)
                raw_loss, loss_info = shortcut.compute_loss(model, schedule, z_0, tau, step_size, actions=actions)
                if gc_was_enabled:
                    model.gradient_checkpointing = True
                # Normalize via RunningRMS (loss_info values are floats from .item())
                # Update component trackers, then normalize the combined loss tensor
                if rms_dict is not None:
                    loss_std_t = torch.tensor(loss_info["loss_std"], device=device)
                    loss_boot_t = torch.tensor(loss_info["loss_boot"], device=device)
                    rms_dict["std"].update(loss_std_t)
                    rms_dict["boot"].update(loss_boot_t)
                    # Weighted RMS scale from component trackers (detached, preserves grad graph)
                    n_std, n_boot = loss_info["n_std"], loss_info["n_boot"]
                    total = max(n_std + n_boot, 1)
                    rms_scale = torch.sqrt(
                        (n_std * rms_dict["std"].rms + n_boot * rms_dict["boot"].rms) / total
                    ) + 1e-8
                    loss = raw_loss / rms_scale
                else:
                    loss = raw_loss
                raw_loss_val = raw_loss.item()
            else:
                # Standard training: predict clean latents
                z_pred = model(z_tau, tau, actions=actions, independent_frames=use_independent)
                raw_loss = x_prediction_loss(z_pred, z_0, tau, use_ramp_weight=True)
                # Normalize via RunningRMS
                if rms_dict is not None:
                    loss = rms_dict["x_pred"].update(raw_loss)
                else:
                    loss = raw_loss
                raw_loss_val = raw_loss.item()

        # Scale loss for gradient accumulation and backward
        scaled_loss = loss / current_accum
        scaler.scale(scaled_loss).backward()
        micro_count += 1

        # Step optimizer when accumulation window is full
        if micro_count >= current_accum:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            micro_count = 0
            did_step = True
        else:
            grad_norm = torch.tensor(0.0)
            did_step = False

        # Track metrics (use raw unscaled loss)
        total_loss += raw_loss_val
        if did_step:
            total_grad_norm += grad_norm.item() if torch.isfinite(grad_norm) else 0.0
            total_grad_norm_count += 1
        num_batches += 1
        global_step += 1

        # Track prediction stats for mode collapse detection (every 10 batches)
        if batch_idx % 10 == 0:
            with torch.no_grad():
                if shortcut is None:
                    pred_std = z_pred.std().item()
                else:
                    # For shortcut forcing, we need to get prediction from the model
                    pred_std = 0.0  # Skip for now
                total_pred_std += pred_std

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

            # Compute running averages (approximate step count)
            num_steps = max(1, total_grad_norm_count)
            avg_grad_norm = total_grad_norm / num_steps
            pred_std_samples = (batch_idx // 10) + 1
            avg_pred_std = total_pred_std / pred_std_samples if pred_std_samples > 0 else 0

            # Warning flags
            warnings = []
            if raw_loss_val > 1.0:
                warnings.append("HIGH_LOSS")
            if did_step and grad_norm.item() > 0.99:  # Near clip threshold
                warnings.append("GRAD_CLIP")
            if not torch.isfinite(torch.tensor(raw_loss_val)):
                warnings.append("NAN/INF")
            warning_str = f" ⚠️  {' '.join(warnings)}" if warnings else ""

            # Wandb per-step logging
            wandb_metrics = {
                "train/loss": raw_loss_val,
                "train/grad_norm": grad_norm.item() if did_step else 0.0,
                "train/tau_min": tau_min,
                "train/tau_max": tau_max,
                "train/lr": scheduler.get_last_lr()[0],
                "train/batches_per_sec": batches_per_sec,
                "train/epoch": epoch,
            }
            if shortcut is not None:
                wandb_metrics["train/loss_std"] = loss_info["loss_std"]
                wandb_metrics["train/loss_boot"] = loss_info["loss_boot"]
            else:
                wandb_metrics["train/pred_std"] = pred_std
            log_step(wandb_metrics, step=global_step)

            if shortcut is not None:
                # Show shortcut forcing info
                print(
                    f"{progress} "
                    f"Loss: {raw_loss_val:.4f} (std:{loss_info['loss_std']:.4f} boot:{loss_info['loss_boot']:.4f}) "
                    f"Tau: [{tau_min:.2f}-{tau_max:.2f}] "
                    f"GradN: {grad_norm.item():.3f} "
                    f"({batches_per_sec:.1f} batch/s){warning_str}"
                )
            else:
                print(
                    f"{progress} "
                    f"Loss: {raw_loss_val:.4f} "
                    f"Tau: [{tau_min:.2f}-{tau_max:.2f}] "
                    f"GradN: {grad_norm.item():.3f} "
                    f"PredStd: {pred_std:.4f} "
                    f"({batches_per_sec:.1f} batch/s){warning_str}"
                )

        # Step-based checkpoint saving
        if args.save_steps > 0 and global_step % args.save_steps == 0 and checkpoint_dir is not None:
            step_path = checkpoint_dir / f"dynamics_step_{global_step:07d}.pt"
            save_checkpoint(
                step_path, model, optimizer, scaler, epoch, global_step,
                raw_loss_val, args, scheduler=scheduler, rms_trackers=rms_dict
            )
            # Also update latest
            latest_path = checkpoint_dir / "dynamics_latest.pt"
            save_checkpoint(
                latest_path, model, optimizer, scaler, epoch, global_step,
                raw_loss_val, args, scheduler=scheduler, rms_trackers=rms_dict
            )

        # Preemption: save checkpoint immediately and break
        if _preempt.is_set() and checkpoint_dir is not None:
            print(f"Preemption: saving checkpoint at step {global_step}...", flush=True)
            latest_path = checkpoint_dir / "dynamics_latest.pt"
            save_checkpoint(
                latest_path, model, optimizer, scaler, epoch, global_step,
                raw_loss_val, args, scheduler=scheduler, rms_trackers=rms_dict
            )
            print("Checkpoint saved. Exiting.", flush=True)
            return {
                "loss": total_loss / max(num_batches, 1),
                "grad_norm": total_grad_norm / max(total_grad_norm_count, 1),
                "pred_std": 0.0,
                "global_step": global_step,
                "preempted": True,
            }

        batch_idx += 1

    # Flush remaining accumulated gradients at end of epoch
    if micro_count > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    # Approximate optimizer steps (exact count depends on short/long mix)
    avg_accum = (accum_short + accum_long) / 2 if alternating and accum_short != accum_long else accumulation_steps
    num_steps = max(1, int(num_batches / avg_accum))
    avg_grad_norm = total_grad_norm / num_steps
    pred_std_samples = (num_batches // 10) + 1
    avg_pred_std = total_pred_std / pred_std_samples if pred_std_samples > 0 else 0.0

    return {
        "loss": avg_loss,
        "grad_norm": avg_grad_norm,
        "pred_std": avg_pred_std,
        "global_step": global_step,
        "preempted": False,
    }


def main():
    args = parse_args()
    _install_preemption_handler()

    print("=" * 60)
    print("Dynamics Model Training")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Model size: {args.model_size}")
    print(f"Latent dim: {args.latent_dim}")
    print(f"Tokenizer type: {args.tokenizer_type}")
    if args.alternating_lengths:
        accum_short = args.gradient_accumulation_short or args.gradient_accumulation
        accum_long = args.gradient_accumulation_long or args.gradient_accumulation
        print(f"Alternating lengths: short={args.seq_len_short} (batch={args.batch_size_short}), "
              f"long={args.seq_len_long} (batch={args.batch_size_long})")
        print(f"Long ratio: {args.long_ratio:.0%}")
        if accum_short > 1 or accum_long > 1:
            print(f"Gradient accumulation: short={accum_short}, long={accum_long}")
            print(f"Effective batch size: short={args.batch_size_short * accum_short}, "
                  f"long={args.batch_size_long * accum_long}")
    else:
        print(f"Sequence length: {args.sequence_length}")
        print(f"Batch size: {args.batch_size}")
        if args.gradient_accumulation > 1:
            print(f"Gradient accumulation: {args.gradient_accumulation} steps")
            print(f"Effective batch size: {args.batch_size * args.gradient_accumulation}")
    print(f"Learning rate: {args.lr}")
    print("=" * 60)

    # Create run directory with descriptive name
    base_checkpoint_dir = Path(args.checkpoint_dir)
    base_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        # When resuming, use the same directory as the checkpoint
        checkpoint_path = Path(args.resume)
        if checkpoint_path.parent.name.startswith("run_"):
            checkpoint_dir = checkpoint_path.parent
        else:
            # Legacy checkpoint in flat directory - create new run dir
            checkpoint_dir = create_run_directory(base_checkpoint_dir, args)
    elif args.run_name:
        # Custom run name provided
        checkpoint_dir = base_checkpoint_dir / args.run_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Auto-generate run directory name
        checkpoint_dir = create_run_directory(base_checkpoint_dir, args)

    print(f"Checkpoint directory: {checkpoint_dir}")

    # Create dataset(s) and dataloader(s)
    print(f"\nLoading latent sequences from {args.latents_dir}...")

    # Select dataset class based on --packed flag
    DatasetClass = PackedLatentSequenceDataset if args.packed else LatentSequenceDataset
    if args.packed:
        print("Using PACKED latent format (fast I/O)")
    else:
        print("Using per-frame latent format (consider --packed for faster I/O)")

    if args.alternating_lengths:
        # Two datasets with different sequence lengths
        dataset_short = DatasetClass(
            latents_dir=args.latents_dir,
            sequence_length=args.seq_len_short,
            stride=args.stride,
            load_actions=args.use_actions,
            features_dir=args.features_dir if args.use_actions else None,
        )
        dataset_long = DatasetClass(
            latents_dir=args.latents_dir,
            sequence_length=args.seq_len_long,
            stride=args.stride,
            load_actions=args.use_actions,
            features_dir=args.features_dir if args.use_actions else None,
        )

        if len(dataset_short) == 0 or len(dataset_long) == 0:
            print("ERROR: No sequences found!")
            if args.packed:
                print("Make sure to run pack_latents.py first.")
            else:
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
        dataset = DatasetClass(
            latents_dir=args.latents_dir,
            sequence_length=args.sequence_length,
            stride=args.stride,
            load_actions=args.use_actions,
            features_dir=args.features_dir if args.use_actions else None,
        )

        if len(dataset) == 0:
            print("ERROR: No sequences found!")
            if args.packed:
                print("Make sure to run pack_latents.py first.")
            else:
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
    model = create_dynamics(
        args.model_size,
        latent_dim=args.latent_dim,
        use_actions=args.use_actions,
        use_agent_tokens=args.use_agent_tokens,
        use_qk_norm=not args.no_qk_norm,
        soft_cap=args.soft_cap if args.soft_cap > 0 else None,
        num_register_tokens=args.num_register_tokens,
        num_kv_heads=args.num_kv_heads,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    model = model.to(args.device)
    num_params = model.get_num_params()
    print(f"Model parameters: {num_params:,}")
    if args.use_actions:
        print("Action conditioning: ENABLED")
    if args.use_agent_tokens:
        print("Agent tokens: ENABLED")
    print(f"QKNorm: {'ENABLED' if not args.no_qk_norm else 'DISABLED'}")
    print(f"Soft cap: {args.soft_cap if args.soft_cap > 0 else 'DISABLED'}")
    print(f"Register tokens: {args.num_register_tokens}")
    if args.num_kv_heads:
        print(f"GQA: {args.num_kv_heads} KV heads")
    print(f"Gradient checkpointing: {'ENABLED' if args.gradient_checkpointing else 'DISABLED'}")
    print(f"Independent frame ratio: {args.independent_frame_ratio:.0%}")

    # torch.compile the model
    if not args.no_compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Save run configuration
    save_run_config(checkpoint_dir, args, num_params)

    # Initialize wandb
    wandb_run = init_wandb(args, job_type="dynamics", extra_config={
        "num_params": num_params,
        "checkpoint_dir": str(checkpoint_dir),
    })

    # Create optimizer
    optimizer = create_optimizer(model.parameters(), args.lr, args.weight_decay, use_8bit=args.use_8bit_adam, betas=(0.9, 0.95))

    # WSD Learning Rate Schedule
    # Estimate total optimizer steps accounting for gradient accumulation
    if dataloader is not None:
        steps_per_epoch = len(dataloader) // args.gradient_accumulation
    else:
        accum_s = args.gradient_accumulation_short or args.gradient_accumulation
        accum_l = args.gradient_accumulation_long or args.gradient_accumulation
        # Weighted average: (1-long_ratio) of batches use accum_s, long_ratio use accum_l
        n_short = len(dataloader_short)
        n_long = int(n_short * args.long_ratio / (1 - args.long_ratio))
        steps_per_epoch = n_short // accum_s + n_long // accum_l
    total_steps = args.epochs * steps_per_epoch
    scheduler = create_wsd_schedule(optimizer, total_steps, args.warmup_steps, args.decay_steps)

    # Create diffusion schedule
    schedule = DiffusionSchedule(device=args.device)

    # Create shortcut forcing if enabled
    shortcut = None
    if args.shortcut_forcing:
        shortcut = ShortcutForcing(k_max=args.shortcut_k_max)
        print(f"Shortcut forcing enabled (k_max={args.shortcut_k_max})")

    # Create scaler for mixed precision
    scaler = GradScaler(args.device.split(":")[0])

    # RMS trackers for loss normalization
    if args.shortcut_forcing:
        rms_dict = {
            "std": RunningRMS(),
            "boot": RunningRMS(),
        }
    else:
        rms_dict = {
            "x_pred": RunningRMS(),
        }

    # Resume if checkpoint provided
    start_epoch = 0
    global_step = 0
    if args.resume:
        print(f"\nResuming from {args.resume}...")
        start_epoch, global_step, _ = load_checkpoint(
            Path(args.resume), model, optimizer, scaler, scheduler=scheduler, rms_trackers=rms_dict, strict=False
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

        # Resolve per-length accumulation
        if args.alternating_lengths:
            accum_s = args.gradient_accumulation_short or args.gradient_accumulation
            accum_l = args.gradient_accumulation_long or args.gradient_accumulation
        else:
            accum_s = args.gradient_accumulation
            accum_l = None

        metrics = train_epoch(
            model, dataloader, optimizer, scaler, scheduler, schedule, args.device,
            epoch, global_step, args, checkpoint_dir, shortcut,
            dataloader_short, dataloader_long, use_actions=args.use_actions,
            independent_frame_ratio=args.independent_frame_ratio,
            rms_dict=rms_dict,
            accumulation_steps=accum_s,
            accumulation_steps_long=accum_l,
        )

        global_step = metrics["global_step"]

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Grad Norm (avg): {metrics['grad_norm']:.4f}")
        print(f"  Pred Std (avg): {metrics['pred_std']:.4f}")

        # Wandb epoch summary
        log_step({
            "epoch/loss": metrics["loss"],
            "epoch/grad_norm": metrics["grad_norm"],
            "epoch/pred_std": metrics["pred_std"],
            "epoch/epoch": epoch + 1,
        }, step=global_step)

        # Mode collapse warning
        if metrics['pred_std'] < 0.01:
            print("  ⚠️  WARNING: Low prediction variance - possible mode collapse!")

        # Save history
        history.append({
            "epoch": epoch + 1,
            **metrics,
        })

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = checkpoint_dir / f"dynamics_epoch_{epoch + 1:03d}.pt"
            save_checkpoint(
                checkpoint_path, model, optimizer, scaler, epoch, global_step,
                metrics["loss"], args, scheduler=scheduler, rms_trackers=rms_dict
            )

            # Also save as latest
            latest_path = checkpoint_dir / "dynamics_latest.pt"
            save_checkpoint(
                latest_path, model, optimizer, scaler, epoch, global_step,
                metrics["loss"], args, scheduler=scheduler, rms_trackers=rms_dict
            )

        # Save best model
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            best_path = checkpoint_dir / "dynamics_best.pt"
            save_checkpoint(
                best_path, model, optimizer, scaler, epoch, global_step,
                metrics["loss"], args, scheduler=scheduler, rms_trackers=rms_dict
            )
            print(f"New best model saved (loss: {best_loss:.4f})")

        if metrics.get("preempted"):
            print("Preempted during epoch — exiting training loop.", flush=True)
            break

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

    finish_wandb()


if __name__ == "__main__":
    main()
