"""Shared training utilities for all training scripts.

Provides common infrastructure:
- Argument parsing helpers (add_training_args)
- Optimizer creation with bitsandbytes fallback (create_optimizer)
- WSD (Warmup-Stable-Decay) learning rate schedule (create_wsd_schedule)
- Checkpoint save/load (save_checkpoint, load_checkpoint)
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler

try:
    import bitsandbytes as bnb

    HAS_BNB = True
except ImportError:
    HAS_BNB = False


def add_training_args(parser: argparse.ArgumentParser) -> None:
    """Add common training arguments shared across all training scripts.

    Scripts can override defaults after calling this via parser.set_defaults().
    """
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
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
        "--device",
        type=str,
        default="cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu",
        help="Device to train on",
    )


def create_optimizer(
    params,
    lr: float,
    weight_decay: float,
    use_8bit: bool = True,
    betas: tuple = (0.9, 0.999),
):
    """Create AdamW optimizer with optional bitsandbytes 8-bit variant."""
    if use_8bit and HAS_BNB:
        print("Using 8-bit AdamW (bitsandbytes)")
        return bnb.optim.AdamW8bit(
            params, lr=lr, weight_decay=weight_decay, betas=betas
        )
    if use_8bit and not HAS_BNB:
        print("WARNING: bitsandbytes not installed, falling back to standard AdamW")
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)


def create_wsd_schedule(
    optimizer, total_steps: int, warmup_steps: int, decay_steps: int
):
    """Create Warmup-Stable-Decay (WSD) learning rate schedule."""

    def wsd_schedule(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        if decay_steps > 0 and step >= total_steps - decay_steps:
            decay_progress = (total_steps - step) / max(1, decay_steps)
            return max(0.0, decay_progress)
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=wsd_schedule)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    global_step: int,
    loss: float,
    args: argparse.Namespace,
    scheduler=None,
    rms_trackers: dict = None,
    extra: dict = None,
):
    """Save training checkpoint.

    Args:
        extra: Additional key-value pairs to include in the checkpoint
               (e.g. {"model_type": "transformer_tokenizer"}).
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "loss": loss,
        "args": vars(args),
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if rms_trackers is not None:
        checkpoint["rms_state"] = {
            k: v.state_dict() for k, v in rms_trackers.items()
        }
    if extra is not None:
        checkpoint.update(extra)
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    scheduler=None,
    rms_trackers: dict = None,
    strict: bool = True,
):
    """Load training checkpoint.

    Args:
        strict: If False, allows missing/unexpected keys in model state dict
                (useful when loading checkpoints with new parameters).
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if strict:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        missing, unexpected = model.load_state_dict(
            checkpoint["model_state_dict"], strict=False
        )
        if missing:
            print(f"Note: Initializing new parameters: {missing}")
        if unexpected:
            print(f"Warning: Unexpected keys in checkpoint: {unexpected}")
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if rms_trackers is not None and "rms_state" in checkpoint:
        for k, state in checkpoint["rms_state"].items():
            if k in rms_trackers:
                rms_trackers[k].load_state_dict(state)
    return (
        checkpoint["epoch"],
        checkpoint["global_step"],
        checkpoint.get("loss", float("inf")),
    )
