"""Shared training utilities for all training scripts.

Provides common infrastructure:
- Argument parsing helpers (add_training_args)
- Optimizer creation with bitsandbytes fallback (create_optimizer)
- WSD (Warmup-Stable-Decay) learning rate schedule (create_wsd_schedule)
- Checkpoint save/load (save_checkpoint, load_checkpoint)
- Slurm cooperative yielding (should_yield_to_queue)
- Two-mode preemption system (install_preemption_handlers, PreemptionState)
"""

import argparse
import os
import signal
import threading
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler

try:
    import bitsandbytes as bnb

    HAS_BNB = True
except ImportError:
    HAS_BNB = False


class PreemptionState:
    """Two-mode preemption state for training scripts.

    Immediate mode (SIGUSR1): Save checkpoint after the current/next optimizer
    step and exit. Used for manual preemption when you want a fast stop.

    Delayed mode (SIGTERM): Save checkpoint at the next scheduled checkpoint
    boundary and exit. Used by Slurm preemption and voluntary queue yielding.
    """

    def __init__(self):
        self.immediate = threading.Event()
        self.at_checkpoint = threading.Event()

    def is_immediate(self) -> bool:
        return self.immediate.is_set()

    def is_at_checkpoint(self) -> bool:
        return self.at_checkpoint.is_set()

    def should_save_now(self) -> bool:
        """Check if we should save and exit immediately (every optimizer step)."""
        return self.immediate.is_set()

    def should_save_at_boundary(self) -> bool:
        """Check if we should exit at a checkpoint boundary."""
        return self.at_checkpoint.is_set()


def install_preemption_handlers(state: PreemptionState) -> None:
    """Install signal handlers for preemption.

    Both SIGUSR1 and SIGTERM trigger immediate save+exit.
    SIGTERM: sent by Slurm on preemption or time limit (--signal=B:TERM@120).
    SIGUSR1: manual trigger (scancel --signal=USR1 <jobid>).
    """
    def _handler(signum, frame):
        name = "SIGTERM" if signum == signal.SIGTERM else "SIGUSR1"
        if not state.immediate.is_set():
            print(
                f"\n{name} received -- will save checkpoint and exit.",
                flush=True,
            )
            state.immediate.set()

    signal.signal(signal.SIGUSR1, _handler)
    signal.signal(signal.SIGTERM, _handler)


def compute_dynamic_save_interval(
    global_step: int,
    elapsed_seconds: float,
    checkpoint_minutes: float,
) -> int:
    """Compute step_save_interval to target approximately checkpoint_minutes between saves.

    Returns 0 if not enough data to compute (global_step == 0 or elapsed_seconds == 0).
    """
    if global_step <= 0 or elapsed_seconds <= 0:
        return 0
    steps_per_second = global_step / elapsed_seconds
    interval = int(checkpoint_minutes * 60 * steps_per_second)
    return max(1, interval)


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
        "--checkpoint-minutes",
        type=float,
        default=60,
        help="Target minutes between checkpoints when using dynamic interval "
             "(used when step-save-interval <= 0, default: 60)",
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
    # Normalize state dict: strip _orig_mod. prefix if present
    state_dict = checkpoint["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    # Load into unwrapped model if torch.compile'd
    target = model._orig_mod if hasattr(model, "_orig_mod") else model
    if strict:
        target.load_state_dict(state_dict)
    else:
        missing, unexpected = target.load_state_dict(
            state_dict, strict=False
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


def is_queue_yield_enabled(checkpoint_dir: str | Path | None = None) -> bool:
    """Deprecated: Slurm QOS preemption handles this now. Always returns False."""
    return False
