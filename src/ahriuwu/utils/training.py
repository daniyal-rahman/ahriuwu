"""Shared training utilities for all training scripts.

Provides common infrastructure:
- Argument parsing helpers (add_training_args)
- Optimizer creation with bitsandbytes fallback (create_optimizer)
- WSD (Warmup-Stable-Decay) learning rate schedule (create_wsd_schedule)
- Checkpoint save/load (save_checkpoint, load_checkpoint)
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

    File-based trigger: ``touch checkpoint-now`` in the repo root triggers an
    immediate checkpoint save (without exiting). The file is deleted after the
    checkpoint is written so the trigger is one-shot.
    """

    def __init__(self):
        self.immediate = threading.Event()
        self.at_checkpoint = threading.Event()
        # Resolved once by install_preemption_handlers; stays None if not set.
        self._checkpoint_now_path: Path | None = None

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

    def check_checkpoint_now(self) -> bool:
        """Check if the ``checkpoint-now`` file exists (file-based trigger).

        Unlike signal-based preemption this does NOT cause an exit -- it only
        requests an immediate checkpoint save.  The caller is responsible for
        deleting the file after saving (see :meth:`clear_checkpoint_now`).
        """
        if self._checkpoint_now_path is not None and self._checkpoint_now_path.exists():
            return True
        return False

    def clear_checkpoint_now(self) -> None:
        """Remove the ``checkpoint-now`` sentinel file after saving."""
        if self._checkpoint_now_path is not None and self._checkpoint_now_path.exists():
            self._checkpoint_now_path.unlink()
            print("Cleared checkpoint-now sentinel file.", flush=True)


def install_preemption_handlers(
    state: PreemptionState,
    repo_dir: str | Path | None = None,
) -> None:
    """Install signal handlers for preemption and set up file-based trigger.

    Both SIGUSR1 and SIGTERM trigger immediate save+exit.
    SIGTERM: sent by Slurm on preemption or time limit (--signal=B:TERM@120).
    SIGUSR1: manual trigger (scancel --signal=USR1 <jobid>).

    If *repo_dir* is given (or auto-detected via git), the ``checkpoint-now``
    sentinel file is resolved relative to the repo root so that
    ``touch checkpoint-now`` triggers an immediate checkpoint save.
    """
    # Resolve repo root for checkpoint-now file
    if repo_dir is not None:
        state._checkpoint_now_path = Path(repo_dir) / "checkpoint-now"
    else:
        # Auto-detect: walk up from this file to find the repo root
        _here = Path(__file__).resolve()
        for parent in _here.parents:
            if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
                state._checkpoint_now_path = parent / "checkpoint-now"
                break

    if state._checkpoint_now_path is not None:
        print(f"checkpoint-now trigger: touch {state._checkpoint_now_path}", flush=True)

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
        "--lr-schedule",
        type=str,
        default="wsd",
        choices=["wsd", "cosine"],
        help="LR schedule: wsd (warmup→stable→decay) or cosine (warmup→cosine-to-0, paper-faithful)",
    )
    parser.add_argument(
        "--adam-betas",
        type=float,
        nargs=2,
        default=(0.9, 0.999),
        metavar=("BETA1", "BETA2"),
        help="AdamW (beta1, beta2). DreamerV4 paper uses defaults (0.9, 0.999).",
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


def create_cosine_schedule(optimizer, total_steps: int, warmup_steps: int):
    """Linear warmup + cosine decay to 0 (DreamerV4 §3.4 recipe)."""
    import math

    def cosine_schedule(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_schedule)


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
    # Unwrap torch.compile so the saved keys are the canonical module names
    # (no ``_orig_mod.`` prefix). Downstream consumers (eval, ports, hub
    # uploads) shouldn't have to know about the compile wrapper. The load
    # side still strips defensively, but new checkpoints are now clean.
    inner = model._orig_mod if hasattr(model, "_orig_mod") else model

    # Resolved model config (latent_dim, embed_dim, etc.) as set by the
    # ``create_*`` factory. CLI args alone are not enough — the same
    # ``--model-size small`` can mean different latent_dim values across
    # commits. Capturing the factory-resolved config makes the checkpoint
    # self-describing.
    model_config = getattr(inner, "config", None)

    checkpoint = {
        "model_state_dict": inner.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "loss": loss,
        "args": vars(args),
        "model_config": model_config,
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
        checkpoint.get("batch_idx", 0),
    )


