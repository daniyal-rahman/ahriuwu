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
import os
import random
import atexit
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.amp import GradScaler, autocast

from ahriuwu.data.dataset import FrameSequenceDataset
from ahriuwu.models import create_transformer_tokenizer, MAELoss, psnr, RunningRMS
from ahriuwu.utils.logging import add_wandb_args, init_wandb, log_step, log_images, finish_wandb
from ahriuwu.utils.training import (
    add_training_args, create_optimizer, create_wsd_schedule, create_cosine_schedule,
    save_checkpoint, load_checkpoint,
    PreemptionState, install_preemption_handlers,
)

_preempt = PreemptionState()

# DDP: True on rank 0 (and always in single-GPU). Side-effecting helpers
# (checkpoint writes) early-return when False so only rank 0 touches disk.
_IS_MAIN = True

# Rank-0-gated logging. We shadow print() in THIS module only (NOT the global
# builtin): library output, warnings, and tracebacks from EVERY rank still
# surface, so a silently-failing non-main rank can't hide — the failure mode of
# the previous `builtins.print = lambda: None` monkey-patch.
_builtin_print = print
def print(*args, **kwargs):  # noqa: A001 — intentional module-scoped rank gate
    if _IS_MAIN:
        _builtin_print(*args, **kwargs)


def get_base(m):
    """Peel torch.compile (._orig_mod) and DDP (.module) wrappers -> raw model."""
    while True:
        if hasattr(m, "_orig_mod"):
            m = m._orig_mod
        elif hasattr(m, "module"):
            m = m.module
        else:
            return m


# ---------------------------------------------------------------------------
# Checkpoint helper — eliminates 4x duplicated save_checkpoint calls
# ---------------------------------------------------------------------------

def _save_tokenizer_checkpoints(
    checkpoint_dir: Path,
    base_checkpoint_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    global_step: int,
    loss: float,
    args: argparse.Namespace,
    scheduler=None,
    rms_trackers: dict = None,
    *,
    step_path: Path | None = None,
) -> None:
    """Save checkpoint to up to 3 locations in one call.

    Always saves ``transformer_tokenizer_latest.pt`` in both the run dir and
    (if different) the base checkpoint dir. Optionally saves to *step_path*
    for periodic numbered snapshots.
    """
    if not _IS_MAIN:
        return  # under DDP only rank 0 writes checkpoints
    extra = {"model_type": "transformer_tokenizer"}
    common = dict(
        model=model, optimizer=optimizer, scaler=scaler,
        epoch=epoch, global_step=global_step, loss=loss, args=args,
        scheduler=scheduler, rms_trackers=rms_trackers, extra=extra,
    )

    if step_path is not None:
        save_checkpoint(step_path, **common)

    latest = checkpoint_dir / "transformer_tokenizer_latest.pt"
    save_checkpoint(latest, **common)

    if base_checkpoint_dir is not None and base_checkpoint_dir != checkpoint_dir:
        save_checkpoint(base_checkpoint_dir / "transformer_tokenizer_latest.pt", **common)


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
        "--lpips-frame-subsample",
        type=int,
        default=None,
        help="If set, compute LPIPS on this many random frames of B*T per step "
             "to cap VRAM. RMS norm absorbs the magnitude shift. "
             "Recommended K=16 for B=2 T=16 on 16GB GPUs.",
    )
    parser.add_argument(
        "--use-lpips-lib",
        action="store_true",
        default=True,
        help="Use lpips library (Zhang et al. 2018, paper-faithful). Default on.",
    )
    parser.add_argument(
        "--no-lpips-lib",
        action="store_false",
        dest="use_lpips_lib",
        help="Use custom VGGPerceptualLoss instead of lpips library.",
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
        help="Save checkpoint every N optimizer steps (fixed interval; "
             "0 disables step-based saving). No dynamic/wall-clock interval — "
             "set this to roughly target_minutes / sec_per_optstep.",
    )
    parser.add_argument(
        "--checkpoint-warn-minutes",
        type=float,
        default=60.0,
        help="Warn once at startup if the measured step rate implies the "
             "checkpoint cadence (step-save-interval x sec/optstep) exceeds "
             "this many minutes — flags crash-loss risk at health checks.",
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
        "--max-steps",
        type=int,
        default=0,
        help="If >0, stop training after this many optimizer steps (sanity runs). 0 = no cap.",
    )
    parser.add_argument(
        "--file-ext",
        type=str,
        default="jpg",
        choices=["jpg", "png"],
        help="Frame file extension. YT uses jpg, action-labeled replay uses png.",
    )
    parser.add_argument(
        "--skip-resize",
        action="store_true",
        help="Skip cv2.resize at load time. Use when frames-dir is pre-resized "
             "to target res (defensive fallback to resize if any frame mismatches).",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable color/gamma/noise augmentation on input frames (params "
             "sampled once per sequence for temporal consistency). Mild "
             "defaults: ±10%% brightness/contrast/saturation, ±5° hue, ±15%% "
             "gamma, σ=0.01 noise. Robust to player display calibration drift.",
    )
    parser.add_argument(
        "--use-rope",
        action="store_true",
        help="Use RoPE (Rotary Position Embeddings) instead of additive position embeddings",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=None,
        help="Override the preset's bottleneck per-token width (paper-faithful=32). "
             "Lets you pin a small bottleneck on a non-matching width preset, e.g. "
             "--model-size medium --latent-dim 32 for 115M params @ paper compression.",
    )
    parser.add_argument(
        "--num-latents",
        type=int,
        default=None,
        help="Override the preset's number of bottleneck tokens (paper-faithful=256).",
    )
    parser.add_argument(
        "--embed-dim", type=int, default=None,
        help="Override preset width (e.g. 1024). Used for width-vs-depth experiments.",
    )
    parser.add_argument(
        "--num-heads", type=int, default=None,
        help="Override preset head count (typically embed_dim/64).",
    )
    parser.add_argument(
        "--num-encoder-layers", type=int, default=None,
        help="Override preset encoder depth. Combine with --num-decoder-layers for depth sweeps.",
    )
    parser.add_argument(
        "--num-decoder-layers", type=int, default=None,
        help="Override preset decoder depth.",
    )
    parser.add_argument(
        "--temporal-every", type=int, default=None,
        help="Layer i is temporal iff i %% temporal_every == temporal_every-1. Default 2 "
             "(every-2 = 50%%, pre-2026-05-28 default). Pass 4 for paper Fig. 2 (~25%%).",
    )
    parser.add_argument(
        "--mse-on-full-frame",
        action="store_true",
        help="DENOISING-AE objective: compute MSE on the FULL frame (all patches) while "
             "still masking the INPUT. Default (off) is MAE-style masked-only MSE, which "
             "structurally caps m=0 reconstruction fidelity (verified 2026-06-03: masked-only "
             "20.4 dB vs full-frame 31.2 dB at m=0 on identical arch/clips, +10.8 dB). Since "
             "this tokenizer DEPLOYS at m=0 (frozen encoder feeds dynamics on full frames), "
             "full-frame MSE is the correct target; input masking is kept as augmentation.",
    )
    parser.add_argument(
        "--reset-schedule",
        action="store_true",
        help="When resuming, load model+optimizer+RMS weights but RESET the LR "
             "schedule clock to 0 (ignore the saved scheduler). Use to CONTINUE "
             "a run that already finished its WSD decay (LR~0) for more epochs: "
             "the fresh schedule re-warms then holds flat, instead of inheriting "
             "the dead LR tail. Optimizer momentum is kept (true continuation).",
    )
    parser.add_argument(
        "--tube-masking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="TUBE masking (default on): mask the SAME spatial patch locations across all "
             "frames of a clip. Blocks the temporal-copy shortcut where per-frame-independent "
             "masking lets the model fill a masked patch by copying the same location from an "
             "adjacent frame where it's visible (temporal redundancy => no real learning). "
             "Standard for video MAE. Use --no-tube-masking for the old per-frame-independent "
             "behavior (ablation only).",
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
    is_main: bool = True,
    ddp_model=None,
):
    """Train for one epoch with MAE training.

    Computes LPIPS every step for consistent loss magnitude. Uses GradScaler
    only when dtype is float16 (disabled for bfloat16 to avoid spurious step skips).
    """
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

    # Checkpoint-cadence tracking. We use a FIXED step interval (no dynamic
    # drift). last_saved_step guards against the modulo being true for all
    # `accumulation_steps` micro-batches at a given global_step. The cadence
    # warning measures sec/optstep AFTER warmup (torch.compile + cudnn autotune
    # settle in the first ~15 steps) and flags if the implied wall-time between
    # checkpoints exceeds the warn threshold — so a health check surfaces a
    # too-slow save cadence instead of silently risking large progress loss.
    last_saved_step = -1
    rate_warmup_steps = 15        # ignore these (compile/autotune slow)
    rate_t0 = None               # wall-clock at end of warmup
    rate_step0 = None            # global_step at end of warmup
    cadence_warned = False
    warn_seconds = getattr(args, "checkpoint_warn_minutes", 60) * 60

    for batch_idx, batch in enumerate(dataloader):
        frames = batch["frames"].to(device)  # (B, T, C, H, W)

        # Curriculum learning: ramp up mask_ratio_max over warmup steps
        if args.mask_warmup_steps > 0 and global_step < args.mask_warmup_steps:
            warmup_progress = global_step / args.mask_warmup_steps
            current_mask_max = args.mask_ratio_max * warmup_progress
            # Ensure current_mask_max >= mask_ratio_min so U(min, max) is valid
            current_mask_max = max(current_mask_max, args.mask_ratio_min)
        else:
            current_mask_max = args.mask_ratio_max

        # With p_zero_mask probability, use mask_ratio=0.0 (full reconstruction)
        # Otherwise sample from U(mask_ratio_min, current_max) avoiding near-zero masking
        if random.random() < args.p_zero_mask:
            mask_ratio = 0.0
        else:
            mask_ratio = random.uniform(args.mask_ratio_min, current_mask_max)

        # Generate mask outside torch.compile boundary
        base_model = get_base(model)
        mask_indices = base_model.make_mask(
            frames.shape[0], frames.shape[1], mask_ratio, frames.device,
            tube=args.tube_masking,
        )

        # Mixed precision forward
        with autocast(device_type=device_type, dtype=dtype):
            output = model(frames, mask_indices=mask_indices)
            recon = output["reconstruction"]
            mask_indices = output.get("mask_indices", None)

            # MSE loss support: full-frame (denoising-AE) vs masked-only (MAE).
            # Input is always masked above (representation benefit); this only
            # controls which pixels the MSE is computed on. None => all patches.
            loss_mask = None if args.mse_on_full_frame else mask_indices

            # Compute MAE loss — LPIPS every step for consistent loss magnitude
            losses = loss_fn(recon, frames, mask_indices=loss_mask, skip_lpips=False)

            # Normalize MSE and LPIPS separately via RunningRMS
            mse_norm = rms_trackers["mse"].update(losses["mse"])
            lpips_norm = rms_trackers["lpips"].update(losses["lpips"])
            loss = (mse_norm + args.lpips_weight * lpips_norm) / accumulation_steps

        # DDP: mask_embed (the MAE mask token) is UNUSED when mask_ratio==0
        # (make_mask returns None -> encode passes mask_embed=None), so it
        # receives no gradient and DDP's reducer raises "Expected to have
        # finished reduction in the prior iteration...". This is silent on
        # single-GPU (no grad sync) but fatal under DDP, and mask_ratio==0 is
        # common here (p_zero_mask + mask warmup + --mse-on-full-frame). Tap it
        # with a zero-valued term so every parameter participates in EVERY
        # backward — keeps find_unused_parameters=False (required by
        # torch.compile's DDPOptimizer) at ~zero cost.
        if ddp_model is not None and hasattr(base_model, "mask_embed"):
            loss = loss + 0.0 * base_model.mask_embed.sum()

        # Backward with scaling. Under DDP, suppress the gradient all-reduce on
        # non-boundary micro-steps (no_sync) so it fires ONCE per optimizer step
        # instead of every micro-step — critical for multi-GPU efficiency.
        is_boundary = (batch_idx + 1) % accumulation_steps == 0
        sync_ctx = ddp_model.no_sync() if (ddp_model is not None and not is_boundary) else nullcontext()
        with sync_ctx:
            scaler.scale(loss).backward()

        # Gradient accumulation
        if is_boundary:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if args.max_steps > 0 and global_step >= args.max_steps:
                if checkpoint_dir:
                    _save_tokenizer_checkpoints(
                        checkpoint_dir, base_checkpoint_dir, model, optimizer, scaler,
                        epoch, global_step, losses["loss"].item(), args,
                        scheduler=scheduler, rms_trackers=rms_trackers,
                    )
                print(f"max_steps={args.max_steps} reached. Exiting.", flush=True)
                return {
                    "loss": total_loss / max(num_batches, 1),
                    "mse": total_mse / max(num_batches, 1),
                    "lpips": total_lpips / max(num_batches, 1),
                    "psnr": total_psnr / max(num_batches, 1),
                    "global_step": global_step,
                    "preempted": True,
                }

        # Track metrics (unscaled loss)
        total_loss += losses["loss"].item()
        total_mse += losses["mse"].item()
        total_lpips += losses["lpips"].item()

        with torch.no_grad():
            # Flatten (B, T, C, H, W) -> (B*T, C, H, W) for PSNR
            recon_flat = recon.view(-1, *recon.shape[-3:]) if recon.dim() == 5 else recon
            frames_flat = frames.view(-1, *frames.shape[-3:]) if frames.dim() == 5 else frames
            batch_psnr = psnr(recon_flat, frames_flat).item()
            total_psnr += batch_psnr

        num_batches += 1

        # Cadence warning: once, after warmup, estimate sec/optstep over a
        # measurement window and flag if the implied checkpoint wall-time is
        # too slow. Surfaces a misconfigured --step-save-interval at health
        # checks instead of silently risking large progress loss.
        if not cadence_warned and args.step_save_interval > 0:
            if rate_t0 is None and global_step >= rate_warmup_steps:
                rate_t0 = time.time()
                rate_step0 = global_step
            elif rate_t0 is not None and global_step - rate_step0 >= 20:
                sec_per_optstep = (time.time() - rate_t0) / max(global_step - rate_step0, 1)
                cadence_sec = args.step_save_interval * sec_per_optstep
                msg = (f"checkpoint cadence ~{cadence_sec/60:.0f} min "
                       f"({sec_per_optstep:.1f}s/optstep x {args.step_save_interval} steps)")
                if cadence_sec > warn_seconds:
                    print(f"WARNING: {msg} EXCEEDS {warn_seconds/60:.0f} min threshold — "
                          f"lower --step-save-interval to reduce crash-loss risk.", flush=True)
                else:
                    print(f"checkpoint cadence OK: {msg}", flush=True)
                cadence_warned = True

        # Fixed-interval checkpoint. The last_saved_step guard prevents the
        # modulo from firing on all `accumulation_steps` micro-batches that
        # share the same global_step.
        save_at_boundary = (
            checkpoint_dir
            and args.step_save_interval > 0
            and global_step > 0
            and global_step % args.step_save_interval == 0
            and global_step != last_saved_step
        )
        if save_at_boundary:
            step_path = checkpoint_dir / f"transformer_tokenizer_step_{global_step:07d}.pt"
            _save_tokenizer_checkpoints(
                checkpoint_dir, base_checkpoint_dir, model, optimizer, scaler,
                epoch, global_step, losses["loss"].item(), args,
                scheduler=scheduler, rms_trackers=rms_trackers, step_path=step_path,
            )
            last_saved_step = global_step

        # checkpoint-now file trigger (save but do NOT exit)
        if _preempt.check_checkpoint_now() and checkpoint_dir and global_step > 0:
            print(f"checkpoint-now trigger at step {global_step}.", flush=True)
            if not save_at_boundary:
                _save_tokenizer_checkpoints(
                    checkpoint_dir, base_checkpoint_dir, model, optimizer, scaler,
                    epoch, global_step, losses["loss"].item(), args,
                    scheduler=scheduler, rms_trackers=rms_trackers,
                )
            _preempt.clear_checkpoint_now()

        # Preemption: save and exit on SIGTERM/SIGUSR1
        if _preempt.should_save_now() and checkpoint_dir and global_step > 0:
            if not save_at_boundary:
                _save_tokenizer_checkpoints(
                    checkpoint_dir, base_checkpoint_dir, model, optimizer, scaler,
                    epoch, global_step, losses["loss"].item(), args,
                    scheduler=scheduler, rms_trackers=rms_trackers,
                )
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
                "train/lpips": losses["lpips"].item(),
                "train/psnr": batch_psnr,
                "train/lr": current_lr,
                "train/mask_ratio": mask_ratio,
                "train/mask_max": current_mask_max,
                "train/samples_per_sec": samples_per_sec,
                "train/epoch": epoch,
            }
            log_step(log_dict, step=global_step)

            lpips_str = f"LPIPS: {losses['lpips'].item():.4f} "
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

    # ---- DDP setup (backward-compatible: single-GPU when not under torchrun) ----
    global _IS_MAIN
    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if ddp:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        args.device = f"cuda:{local_rank}"
    else:
        local_rank, rank, world_size = 0, 0, 1
    is_main = rank == 0
    _IS_MAIN = is_main  # gates the module-scoped print() above (no builtin override)
    if ddp:
        # Tear down the process group on ANY exit (incl. exceptions) so sibling
        # ranks don't hang and NCCL doesn't warn at shutdown.
        atexit.register(lambda: dist.is_initialized() and dist.destroy_process_group())
        # Keep the effective batch invariant: split accumulation across ranks so
        # eff = world_size * batch_size * grad_accum stays the configured value.
        if args.gradient_accumulation % world_size != 0:
            print(f"WARNING: --gradient-accumulation {args.gradient_accumulation} is not "
                  f"divisible by world_size {world_size}; effective batch will drift from the "
                  f"configured {args.batch_size * args.gradient_accumulation} (not a clean split).")
        args.gradient_accumulation = max(1, args.gradient_accumulation // world_size)

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
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation * world_size}"
          f"  (world_size={world_size}, per-rank accum={args.gradient_accumulation})")
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
        file_ext=args.file_ext,
        skip_resize=args.skip_resize,
        augment=args.augment,
    )
    print(f"Found {len(dataset)} sequences (T={args.sequence_length})")

    if len(dataset) == 0:
        print("ERROR: No sequences found! Check --frames-dir path and --sequence-length.")
        return

    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                           shuffle=True, drop_last=True)
        if ddp else None
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
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
        latent_dim=args.latent_dim,
        num_latents=args.num_latents,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        temporal_every=args.temporal_every,
    )
    model = model.to(args.device)
    print(f"Model parameters: {model.get_num_params():,}")
    print(f"Gradient checkpointing: {'ENABLED' if args.gradient_checkpointing else 'DISABLED'}")

    # DDP wrap BEFORE compile so torch.compile's DDPOptimizer manages the
    # bucketed all-reduce. Keep the DDP handle for no_sync() during accumulation.
    ddp_model = None
    if ddp:
        model = DDP(model, device_ids=[local_rank], gradient_as_bucket_view=True)
        ddp_model = model

    # torch.compile the model (but NOT the LPIPS/VGG loss model)
    if not args.no_compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Create loss function (NOT compiled - frozen LPIPS/VGG)
    loss_fn = MAELoss(
        mse_weight=1.0,
        lpips_weight=args.lpips_weight,
        use_lpips_lib=args.use_lpips_lib,
        lpips_frame_subsample=args.lpips_frame_subsample,
    )
    loss_fn = loss_fn.to(args.device)
    print(f"Loss: use_lpips_lib={args.use_lpips_lib}  "
          f"lpips_frame_subsample={args.lpips_frame_subsample}")

    # Create optimizer
    optimizer = create_optimizer(model.parameters(), args.lr, args.weight_decay, use_8bit=args.use_8bit_adam, betas=tuple(args.adam_betas))

    # WSD Learning Rate Schedule
    # Estimate total optimizer steps accounting for gradient accumulation
    steps_per_epoch = len(dataloader) // args.gradient_accumulation
    total_steps = args.epochs * steps_per_epoch
    # Honor --max-steps as the true horizon for the LR scheduler so WSD decay
    # actually fires before the run stops. Otherwise epochs=999 gives a huge
    # total and the decay branch (which triggers at total - decay_steps) is
    # never reached. Cosine likewise needs the right horizon to land at 0.
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    if args.lr_schedule == "cosine":
        scheduler = create_cosine_schedule(optimizer, total_steps, args.warmup_steps)
        print(f"LR schedule: cosine  warmup={args.warmup_steps}  total={total_steps}")
    else:
        scheduler = create_wsd_schedule(optimizer, total_steps, args.warmup_steps, args.decay_steps)
        print(f"LR schedule: wsd  warmup={args.warmup_steps}  decay={args.decay_steps}  total={total_steps}")

    # Create scaler for mixed precision
    # GradScaler is a no-op with bfloat16 and can spuriously skip optimizer steps.
    # Only enable for float16; disable for bfloat16.
    device_type = args.device.split(":")[0]
    use_fp16 = device_type != "cuda"  # bfloat16 on CUDA, float16 on CPU
    scaler = GradScaler(device_type, enabled=use_fp16)

    # Initialize wandb — include the factory-resolved model config so each run
    # records the exact numeric dimensions (latent_dim, embed_dim, num_layers,
    # …). Without this, args.model_size='small' is the only identifier and
    # silently lies if the preset table changes between commits.
    _inner = get_base(model)
    wandb_run = init_wandb(args, job_type="transformer_tokenizer", extra_config={
        "num_params": _inner.get_num_params(),
        "checkpoint_dir": str(checkpoint_dir),
        "model_config": getattr(_inner, "config", {}),
    }) if is_main else None

    # RMS trackers for loss normalization
    rms_trackers = {
        "mse": RunningRMS(),
        "lpips": RunningRMS(),
    }

    # Resume if checkpoint provided
    start_epoch = 0
    global_step = 0
    if args.resume:
        print(f"Resuming from {args.resume}..."
              + ("  [--reset-schedule: fresh LR schedule from step 0]" if args.reset_schedule else ""))
        start_epoch, global_step, _, _ = load_checkpoint(
            Path(args.resume), model, optimizer, scaler, scheduler=scheduler,
            rms_trackers=rms_trackers, reset_schedule=args.reset_schedule,
        )
        # Resume at the same epoch (don't increment — the epoch may not be complete)
        print(f"Resuming from epoch {start_epoch}, step {global_step}")

    # Training loop
    best_loss = float("inf")
    history = []

    # Zero gradients before training to avoid stale grads on first batch
    optimizer.zero_grad()

    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)  # reshuffle the per-rank shards each epoch
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print("=" * 60)

        metrics = train_epoch(
            model, dataloader, optimizer, scaler, scheduler, loss_fn, rms_trackers,
            args.device, epoch, global_step, args, checkpoint_dir, base_checkpoint_dir,
            is_main=is_main, ddp_model=ddp_model,
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
            epoch_path = checkpoint_dir / f"transformer_tokenizer_epoch_{epoch + 1:03d}.pt"
            _save_tokenizer_checkpoints(
                checkpoint_dir, base_checkpoint_dir, model, optimizer, scaler,
                epoch, global_step, metrics["loss"], args,
                scheduler=scheduler, rms_trackers=rms_trackers, step_path=epoch_path,
            )

        # Save best model (rank 0 only)
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            if is_main:
                best_path = checkpoint_dir / "transformer_tokenizer_best.pt"
                save_checkpoint(
                    best_path, model, optimizer, scaler, epoch, global_step,
                    metrics["loss"], args, scheduler=scheduler, rms_trackers=rms_trackers,
                    extra={"model_type": "transformer_tokenizer"},
                )
                print(f"New best model saved (loss: {best_loss:.4f})")

        if metrics.get("preempted"):
            print("Preempted during epoch — exiting training loop.", flush=True)
            break

        # Save sample reconstructions (rank 0 only)
        if is_main:
            save_samples(model, dataloader, args.device, sample_dir, epoch + 1)

    # Save training history (rank 0 only)
    if is_main:
        history_path = checkpoint_dir / "transformer_tokenizer_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"\nTraining history saved to {history_path}")

        print("\n" + "=" * 60)
        print("Training complete!")
        print(f"Best loss: {best_loss:.4f}")
        print(f"Best model: {checkpoint_dir / 'transformer_tokenizer_best.pt'}")
        print("=" * 60)

    finish_wandb()  # self-no-ops on non-main (wandb.run is None)
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
