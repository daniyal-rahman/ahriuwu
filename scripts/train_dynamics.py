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
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from ahriuwu.data import PackedLatentSequenceDataset
from ahriuwu.models import (
    create_dynamics,
    DiffusionSchedule,
    x_prediction_loss,
    ShortcutForcing,
    RunningRMS,
)
from ahriuwu.data.dataset import VideoGroupedSampler
from ahriuwu.utils.logging import add_wandb_args, init_wandb, log_step, log_images, finish_wandb
from ahriuwu.utils.training import (
    add_training_args, create_optimizer, create_wsd_schedule,
    save_checkpoint, load_checkpoint,
    PreemptionState, install_preemption_handlers, compute_dynamic_save_interval,
)

_preempt = PreemptionState()


# ---------------------------------------------------------------------------
# Dataclasses to reduce train_epoch parameter count
# ---------------------------------------------------------------------------

@dataclass
class TrainingState:
    """Mutable training state passed through the training loop."""
    model: nn.Module
    optimizer: torch.optim.Optimizer
    scaler: GradScaler
    scheduler: object  # LambdaLR
    global_step: int
    rms_dict: dict[str, RunningRMS] | None = None
    shortcut: ShortcutForcing | None = None


@dataclass
class DataConfig:
    """Dataloader configuration for single or alternating-length training.

    Always loads max batch size. Shortcut steps slice to smaller batch.
    """
    dataloader: DataLoader | None = None
    dataloader_short: DataLoader | None = None
    dataloader_long: DataLoader | None = None
    accumulation_steps: int = 1
    accumulation_steps_long: int | None = None
    shortcut_batch_short: int = 2  # Slice to this for shortcut on short batches
    shortcut_batch_long: int = 1   # Slice to this for shortcut on long batches
    long_ratio: float = 0.1


@dataclass
class CheckpointConfig:
    """Paths for checkpoint saving and the model name prefix."""
    checkpoint_dir: Path | None = None
    base_checkpoint_dir: Path | None = None
    name_prefix: str = "dynamics"


# ---------------------------------------------------------------------------
# Helpers: batch acquisition, checkpoint saving
# ---------------------------------------------------------------------------

def get_next_batch(
    alternating: bool,
    use_long: bool,
    *,
    batch_iterator=None,
    iter_short=None,
    iter_long=None,
    dataloader_long: DataLoader | None = None,
):
    """Fetch the next batch from the appropriate dataloader.

    Returns ``(batch, seq_type, new_iter_long)`` where *new_iter_long* may be
    a freshly rewound iterator (long loader cycles, short loader ends the
    epoch).  Raises ``StopIteration`` when the epoch is done.
    """
    if alternating:
        if use_long:
            try:
                batch = next(iter_long)
            except StopIteration:
                iter_long = iter(dataloader_long)
                batch = next(iter_long)
            return batch, "L", iter_long
        else:
            # Short loader exhausted = epoch done (raises StopIteration)
            batch = next(iter_short)
            return batch, "S", iter_long
    else:
        batch = next(batch_iterator)
        return batch, "", iter_long


def save_checkpoints(
    ckpt_cfg: CheckpointConfig,
    ts: TrainingState,
    epoch: int,
    loss: float,
    args: argparse.Namespace,
    *,
    step_path: Path | None = None,
    extra: dict | None = None,
) -> None:
    """Save checkpoint to up to 3 locations in one call.

    Always saves ``<name_prefix>_latest.pt`` in both the run dir and (if
    different) the base checkpoint dir.  Optionally saves to *step_path* for
    periodic numbered snapshots.

    This replaces the 3x duplicated ``save_checkpoint(...)`` blocks that
    previously appeared at every save site.
    """
    cd = ckpt_cfg.checkpoint_dir
    if cd is None:
        return

    common = dict(
        model=ts.model,
        optimizer=ts.optimizer,
        scaler=ts.scaler,
        epoch=epoch,
        global_step=ts.global_step,
        loss=loss,
        args=args,
        scheduler=ts.scheduler,
        rms_trackers=ts.rms_dict,
        extra=extra,
    )

    if step_path is not None:
        save_checkpoint(step_path, **common)

    latest = cd / f"{ckpt_cfg.name_prefix}_latest.pt"
    save_checkpoint(latest, **common)

    bcd = ckpt_cfg.base_checkpoint_dir
    if bcd is not None and bcd != cd:
        save_checkpoint(bcd / f"{ckpt_cfg.name_prefix}_latest.pt", **common)


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


def _pick_holdout(dataset, n):
    """N whole videos reserved for eval (deterministic: last N by sorted id)."""
    if n <= 0:
        return set()
    vids = sorted({s["video_id"] for s in dataset.sequences})
    return set(vids[-n:])


def _build_holdout_batch(dataset, holdout_vids, batch_size):
    """Collate up to batch_size sequences drawn from the held-out videos."""
    from torch.utils.data import default_collate
    idxs = [i for i, s in enumerate(dataset.sequences) if s["video_id"] in holdout_vids]
    if not idxs:
        return None
    return default_collate([dataset[i] for i in idxs[:batch_size]])


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
        default=8,
        help="Batch size for short sequences (max, used for standard steps)",
    )
    parser.add_argument(
        "--batch-size-long",
        type=int,
        default=4,
        help="Batch size for long sequences",
    )
    parser.add_argument(
        "--shortcut-batch-short",
        type=int,
        default=2,
        help="Batch size for shortcut steps on short sequences (sliced from full batch). "
             "Shortcut needs 3 forward passes without GC, so smaller batch.",
    )
    parser.add_argument(
        "--shortcut-batch-long",
        type=int,
        default=1,
        help="Batch size for shortcut steps on long sequences.",
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
    parser.add_argument(
        "--bootstrap-weight",
        type=float,
        default=10.0,
        help="Weight multiplier for bootstrap loss (default 10). "
             "Needed because bootstrap MSE is naturally smaller than standard MSE.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=200,
        help="Eval every N optimizer steps (default 200, ~5 min)",
    )
    # Action conditioning (replay actions: movement + Q/W/E/R/Flash/Ignite/AA/Recall/Stride)
    parser.add_argument(
        "--use-actions",
        action="store_true",
        help="Enable action conditioning via ReplayLatentSequenceDataset (needs --labels-root)",
    )
    parser.add_argument(
        "--labels-root",
        type=str,
        default=None,
        help="Dir of <match_id>/labels.json + clicks.json (replay action labels)",
    )
    parser.add_argument(
        "--features-dir",
        type=str,
        default="data/processed",
        help="Legacy OCR features dir (unused by the replay action path)",
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
    parser.add_argument(
        "--holdout-videos", type=int, default=0,
        help="Reserve N whole videos (excluded from training) for a clean "
             "held-out eval batch used by the rollout/denoising eval. 0 (default) "
             "uses a training-drawn val batch — what job 124 uses, so its data is "
             "unchanged on requeue.",
    )
    return parser.parse_args()


@torch.no_grad()
def eval_denoising_psnr(
    model: nn.Module,
    schedule: "DiffusionSchedule",
    val_batch: dict,
    device: str,
    tau_levels: list[float] = [0.1, 0.3, 0.5, 0.7, 0.9],
    shortcut: "ShortcutForcing | None" = None,
) -> dict:
    """1-step denoising PSNR on a validation batch.

    For each tau level, corrupts clean latents, predicts clean, and measures PSNR.
    Also tests K=4 shortcut denoising if shortcut is provided.
    """
    model.eval()
    z_0 = val_batch["latents"].to(device)
    B, T = z_0.shape[:2]
    results = {}

    for tau_val in tau_levels:
        tau = torch.full((B, T), tau_val, device=device)
        z_tau, _ = schedule.add_noise(z_0, tau)

        amp_dtype = torch.bfloat16 if device != "mps" else torch.float16
        with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype):
            z_pred = model(z_tau, tau, step_size=torch.ones(B, dtype=torch.long, device=device))

        # PSNR: 10 * log10(max_val² / MSE)
        mse = ((z_pred.float() - z_0.float()) ** 2).mean().item()
        max_val = z_0.abs().max().item()
        psnr = 10 * torch.log10(torch.tensor(max_val ** 2 / max(mse, 1e-10))).item()
        results[f"eval/psnr_tau{tau_val:.1f}"] = psnr

        if psnr < 10.0:
            pred_std = z_pred.float().std().item()
            pred_mean = z_pred.float().mean().item()
            print(f"    [EVAL WARN] tau={tau_val:.1f} PSNR={psnr:.1f} dB | "
                  f"pred mean={pred_mean:.4f} std={pred_std:.4f} | "
                  f"MSE={mse:.4f} | z_0 range=[{z_0.min():.3f}, {z_0.max():.3f}]")

    # Shortcut denoising: test d=16 (K=4 inference step size) as a denoiser.
    # Compare to d=1 at the same tau to measure shortcut quality gap.
    # NOTE: The old K4 eval started from pure noise and ran 4 Euler steps,
    # then compared to z_0 — but that's unconditional generation, not denoising.
    # The output has no connection to z_0, so PSNR was always ~3 dB (random).
    if shortcut is not None:
        K = 4
        d_shortcut = shortcut.k_max // K  # d=16 for K=4

        for tau_val in tau_levels:
            tau = torch.full((B, T), tau_val, device=device)
            z_tau, _ = schedule.add_noise(z_0, tau)

            with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype):
                # d=16 prediction (what K4 inference uses)
                d_t = torch.full((B,), d_shortcut, dtype=torch.long, device=device)
                z_pred_d16 = model(z_tau, tau, step_size=d_t)

            mse = ((z_pred_d16.float() - z_0.float()) ** 2).mean().item()
            psnr = 10 * torch.log10(torch.tensor(max_val ** 2 / max(mse, 1e-10))).item()
            results[f"eval/psnr_d16_tau{tau_val:.1f}"] = psnr

        # Summary: gap between d=1 and d=16 at tau=0.5
        gap = results.get("eval/psnr_tau0.5", 0) - results.get("eval/psnr_d16_tau0.5", 0)
        results["eval/shortcut_gap_db"] = gap

    model.train()
    return results


@torch.no_grad()
def eval_rollout_psnr(
    model: nn.Module,
    schedule: "DiffusionSchedule",
    val_batch: dict,
    device: str,
    ctx_frames: int = 16,
    rollout_frames: int = 32,
    num_steps: int = 16,
    k_max: int = 16,
    tau_ctx: float = 0.1,
    use_actions: bool = False,
    horizons: tuple = (1, 2, 4, 8, 16, 32),
) -> dict:
    """Free-running autoregressive rollout PSNR -- the DEPLOY regime.

    ``eval_denoising_psnr`` is teacher-forced 1-step (corrupt clean -> predict
    clean), which cannot see error ACCUMULATION across autoregressive steps --
    exactly the metric/objective/deploy-regime decoupling that hid the tokenizer
    bug for a week. This prefills ``ctx_frames`` of real context, then generates
    the next ``rollout_frames`` ONE AT A TIME via the KV-cached shortcut rollout
    (the real inference path), feeding its own predictions back, and reports PSNR
    per horizon offset so drift is visible.

    Base flow-matching model (no shortcut): pass ``num_steps == k_max`` (d=1, the
    standard multi-step ODE). Shortcut-distilled model: pass the deploy config
    (``num_steps=4, k_max=64`` -> d=16, the real-time path).
    """
    net = getattr(model, "_orig_mod", model)  # unwrap torch.compile if present
    z = val_batch["latents"].to(device)        # (B, T, C, H, W)
    B, T = z.shape[:2]
    H = min(rollout_frames, T - ctx_frames)
    if H < 1:
        return {}
    context = z[:, :ctx_frames]
    gt = z[:, ctx_frames:ctx_frames + H].float()

    actions_context = actions_future = None
    if use_actions and val_batch.get("actions") is not None:
        a = {k: v.to(device) for k, v in val_batch["actions"].items()}
        actions_context = {k: v[:, :ctx_frames] for k, v in a.items()}
        actions_future = {k: v[:, ctx_frames:ctx_frames + H] for k, v in a.items()}

    amp_dtype = torch.bfloat16 if device != "mps" else torch.float16
    with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype):
        pred = net.rollout(                    # (B, H, C, H, W); manages eval/train itself
            context, predict_frames=H, num_steps=num_steps, k_max=k_max,
            tau_ctx=tau_ctx, actions_context=actions_context,
            actions_future=actions_future, device=device,
        ).float()

    max_val = z.float().abs().max().item()

    def _psnr(a, b):
        mse = ((a - b) ** 2).mean().item()
        return 10 * torch.log10(torch.tensor(max_val ** 2 / max(mse, 1e-10))).item()

    results = {}
    for off in sorted(set(list(horizons) + [H])):
        if 1 <= off <= H:
            results[f"eval_rollout/psnr_h{off}"] = _psnr(pred[:, off - 1], gt[:, off - 1])
    results["eval_rollout/psnr_mean"] = _psnr(pred, gt)
    results["eval_rollout/horizon"] = float(H)
    return results


def train_epoch(
    ts: TrainingState,
    data_cfg: DataConfig,
    ckpt_cfg: CheckpointConfig,
    schedule: DiffusionSchedule,
    device: str,
    epoch: int,
    args: argparse.Namespace,
    *,
    val_batch: dict | None = None,
    eval_interval: int = 1000,
    skip_batches: int = 0,
):
    """Train for one epoch.

    Args:
        ts: Mutable training state (model, optimizer, scaler, scheduler, etc.).
        data_cfg: Dataloader configuration (single or alternating mode).
        ckpt_cfg: Checkpoint directory paths and naming.
        schedule: Diffusion noise schedule.
        device: Device string (e.g. "cuda:0").
        epoch: Current epoch number.
        args: Full argument namespace (for hyperparams and logging config).
        val_batch: Optional fixed validation batch for periodic eval.
        eval_interval: Steps between eval runs.
        skip_batches: Number of batches to fast-forward on resume.

    In alternating mode with per-length accumulation, each accumulation window
    processes batches of a single type (all short or all long) so the loss
    scaling is consistent within the window.
    """
    model = ts.model
    model.train()
    total_loss = 0.0
    total_grad_norm = 0.0
    total_grad_norm_count = 0
    total_pred_std = 0.0
    pred_std = 0.0  # last sampled prediction std (for logging)
    num_batches = 0
    last_grad_norm = 0.0
    start_time = time.time()
    # Steps already completed when THIS epoch window began. The dynamic save
    # interval must be derived from the rate over this window, not from the
    # cumulative global_step -- dividing the cumulative step count by per-epoch
    # elapsed inflates the rate ~Nx by epoch N (and explodes it to tens of
    # millions of steps right after a resume), which silently stops periodic
    # checkpointing.
    step_at_epoch_start = ts.global_step

    # Determine mode and setup iterators
    alternating = data_cfg.dataloader is None
    if not alternating:
        batch_iterator = iter(data_cfg.dataloader)
        total_batches = len(data_cfg.dataloader)
        iter_short = iter_long = None
    else:
        batch_iterator = None
        iter_short = iter(data_cfg.dataloader_short)
        iter_long = iter(data_cfg.dataloader_long)
        total_batches = len(data_cfg.dataloader_short) + int(
            len(data_cfg.dataloader_short) * data_cfg.long_ratio / (1 - data_cfg.long_ratio)
        )

    # Per-length accumulation setup
    accum_short = data_cfg.accumulation_steps
    accum_long = (
        data_cfg.accumulation_steps_long
        if data_cfg.accumulation_steps_long is not None
        else data_cfg.accumulation_steps
    )
    micro_count = 0
    current_accum = data_cfg.accumulation_steps
    use_long = False

    batch_idx = 0
    ts.optimizer.zero_grad()

    # Skip batches on resume (fast-forward through the dataloader)
    if skip_batches > 0:
        print(f"Skipping {skip_batches} batches to resume position...")
        for _ in range(skip_batches):
            try:
                if alternating:
                    if random.random() < data_cfg.long_ratio:
                        next(iter_long)
                    else:
                        next(iter_short)
                else:
                    next(batch_iterator)
            except StopIteration:
                break
        batch_idx = skip_batches
        print(f"Resumed at batch {batch_idx}")

    while True:
        # --- Decide batch type at start of accumulation window ---
        if micro_count == 0 and alternating:
            use_long = random.random() < data_cfg.long_ratio
            current_accum = accum_long if use_long else accum_short

        # --- Get next batch (always max batch size) ---
        try:
            batch, seq_type, iter_long = get_next_batch(
                alternating, use_long,
                batch_iterator=batch_iterator,
                iter_short=iter_short,
                iter_long=iter_long,
                dataloader_long=data_cfg.dataloader_long,
            )
        except StopIteration:
            break

        # --- Unpack batch ---
        z_0 = batch["latents"].to(device)
        B, T, C, H, W = z_0.shape

        actions = None
        if args.use_actions and "actions" in batch and batch["actions"] is not None:
            actions = {k: v.to(device) for k, v in batch["actions"].items()}

        use_independent = random.random() < args.independent_frame_ratio

        # --- Decide whether to use shortcut on this step ---
        # Skip shortcut on long batches (OOMs without GC).
        use_shortcut = ts.shortcut is not None and not use_long

        # Slice batch for shortcut steps (3 forward passes need less VRAM)
        if use_shortcut:
            sc_b = data_cfg.shortcut_batch_short if not use_long else data_cfg.shortcut_batch_long
            if B > sc_b:
                z_0 = z_0[:sc_b]
                if actions is not None:
                    actions = {k: v[:sc_b] for k, v in actions.items()}
                B = sc_b

        # --- Forward pass (mixed precision) ---
        amp_dtype = torch.bfloat16 if device != "mps" else torch.float16
        with autocast(device_type=device.split(":")[0], dtype=amp_dtype):
            if use_shortcut:
                raw_loss, loss_info, tau = _forward_shortcut(
                    model, ts.shortcut, schedule, z_0, B, T, device, actions,
                    global_step=ts.global_step,
                )
                loss = raw_loss
            else:
                raw_loss, loss_info, tau, z_pred = _forward_standard(
                    model, schedule, z_0, B, T, device, actions,
                    use_independent, ts.rms_dict,
                )
                loss = loss_info["loss_normed"]

            raw_loss_val = raw_loss.item()

        # --- Backward + optional optimizer step ---
        scaled_loss = loss / current_accum
        ts.scaler.scale(scaled_loss).backward()
        micro_count += 1

        if micro_count >= current_accum:
            ts.scaler.unscale_(ts.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # On the bf16 path GradScaler is a no-op, so scaler.step() does NOT
            # skip non-finite grads on its own. Gate the step ourselves: a single
            # NaN/Inf gradient must never be written into the weights -- once it
            # is, every later step and the RunningRMS are poisoned to NaN
            # permanently, with no crash, silently wasting the rest of the run.
            if torch.isfinite(grad_norm):
                ts.scaler.step(ts.optimizer)
                ts.scaler.update()
                ts.scheduler.step()
                ts.optimizer.zero_grad()
                micro_count = 0
                did_step = True
                ts.global_step += 1
                last_grad_norm = grad_norm.item()
            else:
                ts.scaler.update()
                ts.optimizer.zero_grad()
                micro_count = 0
                did_step = False
                last_grad_norm = 0.0
                print(
                    f"[WARN] non-finite grad_norm ({grad_norm}) at step "
                    f"{ts.global_step} (batch {batch_idx}); optimizer step "
                    f"skipped, gradients discarded.",
                    flush=True,
                )
        else:
            grad_norm = torch.tensor(0.0)
            did_step = False

        # --- Track metrics ---
        total_loss += raw_loss_val
        if did_step:
            total_grad_norm += grad_norm.item() if torch.isfinite(grad_norm) else 0.0
            total_grad_norm_count += 1
        num_batches += 1

        if batch_idx % 10 == 0:
            with torch.no_grad():
                pred_std = z_pred.std().item() if ts.shortcut is None else 0.0
                total_pred_std += pred_std

        # --- Logging ---
        if batch_idx % args.log_interval == 0:
            _log_train_step(
                batch_idx, total_batches, epoch, alternating, seq_type, T,
                raw_loss_val, loss_info, last_grad_norm, tau, ts, start_time,
                pred_std, did_step, grad_norm,
            )

        # --- Checkpoint saving ---
        if args.save_steps > 0:
            effective_save_interval = args.save_steps
        else:
            effective_save_interval = compute_dynamic_save_interval(
                ts.global_step - step_at_epoch_start, time.time() - start_time,
                args.checkpoint_minutes,
            )

        save_at_boundary = (
            ckpt_cfg.checkpoint_dir is not None
            and ts.global_step > 0
            and effective_save_interval > 0
            and ts.global_step % effective_save_interval == 0
        )
        if save_at_boundary:
            step_path = ckpt_cfg.checkpoint_dir / f"{ckpt_cfg.name_prefix}_step_{ts.global_step:07d}.pt"
            save_checkpoints(
                ckpt_cfg, ts, epoch, raw_loss_val, args,
                step_path=step_path, extra={"batch_idx": batch_idx},
            )

        # --- Eval ---
        if val_batch is not None and did_step and ts.global_step > 0 and ts.global_step % eval_interval == 0:
            eval_results = eval_denoising_psnr(
                model, schedule, val_batch, device, shortcut=ts.shortcut,
            )
            log_step(eval_results, step=ts.global_step)
            psnr_strs = " ".join(f"{k.split('/')[-1]}={v:.1f}" for k, v in eval_results.items())
            print(f"  [EVAL step {ts.global_step}] {psnr_strs}")
            # Deploy-regime rollout eval (autoregressive + KV-cached): exposes the
            # error accumulation the teacher-forced PSNR above is blind to.
            rs, rk = (4, ts.shortcut.k_max) if ts.shortcut is not None else (16, 16)
            roll_results = eval_rollout_psnr(
                model, schedule, val_batch, device,
                num_steps=rs, k_max=rk, use_actions=args.use_actions,
            )
            if roll_results:
                log_step(roll_results, step=ts.global_step)
                roll_strs = " ".join(f"{k.split('/')[-1]}={v:.1f}"
                                     for k, v in roll_results.items() if "psnr" in k)
                print(f"  [EVAL-ROLLOUT step {ts.global_step}] {roll_strs}")

        # --- checkpoint-now file trigger (save but do NOT exit) ---
        if _preempt.check_checkpoint_now() and ckpt_cfg.checkpoint_dir is not None and ts.global_step > 0:
            print(f"checkpoint-now trigger at step {ts.global_step}.", flush=True)
            if not save_at_boundary:
                save_checkpoints(ckpt_cfg, ts, epoch, raw_loss_val, args,
                                 extra={"batch_idx": batch_idx})
            _preempt.clear_checkpoint_now()

        # --- Preemption: signal-based (save and EXIT) ---
        if _preempt.should_save_now() and ckpt_cfg.checkpoint_dir is not None and ts.global_step > 0:
            if not save_at_boundary:
                save_checkpoints(ckpt_cfg, ts, epoch, raw_loss_val, args,
                                 extra={"batch_idx": batch_idx})
            print(f"Immediate preemption at step {ts.global_step}. Exiting.", flush=True)
            return _make_epoch_result(total_loss, total_grad_norm, total_grad_norm_count, num_batches, ts.global_step, preempted=True)

        if save_at_boundary and _preempt.should_save_at_boundary():
            print(f"Yielding at step {ts.global_step} (checkpoint saved). Exiting.", flush=True)
            return _make_epoch_result(total_loss, total_grad_norm, total_grad_norm_count, num_batches, ts.global_step, preempted=True)

        batch_idx += 1

    # Flush remaining accumulated gradients at end of epoch
    if micro_count > 0:
        ts.scaler.unscale_(ts.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if torch.isfinite(grad_norm):
            ts.scaler.step(ts.optimizer)
            ts.scaler.update()
            ts.scheduler.step()
            ts.optimizer.zero_grad()
            ts.global_step += 1
        else:
            ts.scaler.update()
            ts.optimizer.zero_grad()
            print(
                "[WARN] non-finite grad_norm at end-of-epoch flush; "
                "optimizer step skipped.",
                flush=True,
            )

    avg_accum = (
        (accum_short + accum_long) / 2
        if alternating and accum_short != accum_long
        else data_cfg.accumulation_steps
    )
    pred_std_samples = (num_batches // 10) + 1
    return {
        "loss": total_loss / max(num_batches, 1),
        "grad_norm": total_grad_norm / max(1, int(num_batches / avg_accum)),
        "pred_std": total_pred_std / max(pred_std_samples, 1),
        "global_step": ts.global_step,
        "preempted": False,
    }


# ---------------------------------------------------------------------------
# Private helpers for train_epoch (keep the main loop readable)
# ---------------------------------------------------------------------------

def _forward_shortcut(model, shortcut, schedule, z_0, B, T, device, actions,
                      global_step: int = 0):
    """Shortcut forcing forward pass. Returns (raw_loss, loss_info, tau).

    Uses progressive step size curriculum: starts with d∈{1,2} (where teacher
    d=1 is well-trained), then gradually adds larger d as training progresses.
    This breaks the bootstrap trap where teacher ≈ student for untrained d.

    Schedule: d∈{1,2} for steps 0-2k, add d=4 at 2k, d=8 at 4k,
              d=16 at 6k, d=32 at 8k, d=64 at 10k.
    """
    _inner = getattr(model, '_orig_mod', model)
    gc_was_enabled = getattr(_inner, 'gradient_checkpointing', False)
    if gc_was_enabled:
        _inner.gradient_checkpointing = False
    try:
        # Progressive curriculum: max_step_idx increases every 2k steps
        # idx 0=d1, 1=d2, 2=d4, 3=d8, 4=d16, 5=d32, 6=d64
        max_idx = min(1 + global_step // 2000, len(shortcut.step_sizes) - 1)
        step_size = shortcut.sample_step_size(B, device=device, max_step_idx=max_idx)
        tau = shortcut.sample_tau_for_step_size_2d(step_size, T, device=device)
        z_tau, noise = schedule.add_noise(z_0, tau)
        raw_loss, loss_info = shortcut.compute_loss(
            model, schedule, z_0, tau, step_size, actions=actions,
        )
    finally:
        if gc_was_enabled:
            _inner.gradient_checkpointing = True
    return raw_loss, loss_info, tau


def _forward_standard(model, schedule, z_0, B, T, device, actions,
                       use_independent, rms_dict):
    """Standard x-prediction forward pass. Returns (raw_loss, info, tau, z_pred)."""
    tau = schedule.sample_diffusion_forcing_timesteps(B, T, device=device)
    z_tau, noise = schedule.add_noise(z_0, tau)
    z_pred = model(z_tau, tau, actions=actions, independent_frames=use_independent)
    raw_loss = x_prediction_loss(z_pred, z_0, tau, use_ramp_weight=True)
    if rms_dict is not None:
        loss_normed = rms_dict["x_pred"].update(raw_loss)
    else:
        loss_normed = raw_loss
    loss_info = {"loss_normed": loss_normed}
    return raw_loss, loss_info, tau, z_pred


def _log_train_step(
    batch_idx, total_batches, epoch, alternating, seq_type, T,
    raw_loss_val, loss_info, last_grad_norm, tau, ts, start_time,
    pred_std, did_step, grad_norm,
):
    """Emit console + wandb logs for the current training step."""
    elapsed = time.time() - start_time
    batches_per_sec = (batch_idx + 1) / elapsed if elapsed > 0 else 0
    tau_min = tau.min().item()
    tau_max = tau.max().item()

    progress = (
        f"Epoch {epoch} [{batch_idx}/{total_batches}] {seq_type} T={T}"
        if alternating
        else f"Epoch {epoch} [{batch_idx}/{total_batches}]"
    )

    warnings = []
    if raw_loss_val > 1.0:
        warnings.append("HIGH_LOSS")
    if did_step and grad_norm.item() > 0.99:
        warnings.append("GRAD_CLIP")
    if not torch.isfinite(torch.tensor(raw_loss_val)):
        warnings.append("NAN/INF")
    warning_str = f" !!  {' '.join(warnings)}" if warnings else ""

    wandb_metrics = {
        "train/loss": raw_loss_val,
        "train/grad_norm": last_grad_norm,
        "train/tau_min": tau_min,
        "train/tau_max": tau_max,
        "train/lr": ts.scheduler.get_last_lr()[0],
        "train/batches_per_sec": batches_per_sec,
        "train/epoch": epoch,
    }
    if "loss_std" in loss_info:
        wandb_metrics["train/loss_std"] = loss_info["loss_std"]
        wandb_metrics["train/loss_boot"] = loss_info["loss_boot"]
        wandb_metrics["train/loss_boot_weighted"] = loss_info.get("loss_boot_weighted", 0.0)
    else:
        wandb_metrics["train/pred_std"] = pred_std
    log_step(wandb_metrics, step=ts.global_step)

    if "loss_std" in loss_info:
        print(
            f"{progress} "
            f"Loss: {raw_loss_val:.4f} (std:{loss_info['loss_std']:.4f} boot:{loss_info['loss_boot']:.4f}) "
            f"Tau: [{tau_min:.2f}-{tau_max:.2f}] "
            f"GradN: {last_grad_norm:.3f} "
            f"({batches_per_sec:.1f} batch/s){warning_str}"
        )
    else:
        print(
            f"{progress} "
            f"Loss: {raw_loss_val:.4f} "
            f"Tau: [{tau_min:.2f}-{tau_max:.2f}] "
            f"GradN: {last_grad_norm:.3f} "
            f"PredStd: {pred_std:.4f} "
            f"({batches_per_sec:.1f} batch/s){warning_str}"
        )


def _make_epoch_result(total_loss, total_grad_norm, total_grad_norm_count,
                        num_batches, global_step, *, preempted=False):
    """Build the return dict for train_epoch."""
    return {
        "loss": total_loss / max(num_batches, 1),
        "grad_norm": total_grad_norm / max(total_grad_norm_count, 1),
        "pred_std": 0.0,
        "global_step": global_step,
        "preempted": preempted,
    }


def main():
    args = parse_args()
    install_preemption_handlers(_preempt)

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
        checkpoint_path = Path(args.resume)
        if checkpoint_path.parent.name.startswith("run_"):
            checkpoint_dir = checkpoint_path.parent
        else:
            checkpoint_dir = create_run_directory(base_checkpoint_dir, args)
    elif args.run_name:
        checkpoint_dir = base_checkpoint_dir / args.run_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    else:
        checkpoint_dir = create_run_directory(base_checkpoint_dir, args)

    print(f"Checkpoint directory: {checkpoint_dir}")

    # Create dataset(s) and dataloader(s)
    print(f"\nLoading latent sequences from {args.latents_dir}...")

    # The per-frame LatentSequenceDataset was removed in the OCR->replay
    # migration (commit 3f1d40e); only the packed format is supported now.
    if not args.packed:
        raise SystemExit(
            "train_dynamics requires packed latents (pass --packed). Per-frame "
            "LatentSequenceDataset was removed; run scripts/pack_latents.py first."
        )
    if args.use_actions:
        # Action-conditioned: ReplayLatentSequenceDataset pairs packed latents
        # with labels.json/clicks.json-derived actions. The dynamics x-prediction
        # loss ignores rewards, and garen_win for these matches isn't in any
        # manifest, so pass dummy outcomes (all False) — they feed only the unused
        # reward channel. Imported here to dodge the circular import that keeps
        # ReplayLatentSequenceDataset out of ahriuwu.data's __init__.
        import glob as _glob
        from ahriuwu.data.replay_dataset import ReplayLatentSequenceDataset
        if not args.labels_root:
            raise SystemExit("--use-actions requires --labels-root (dir of <match>/labels.json)")
        _mids = [Path(p).stem for p in _glob.glob(str(Path(args.latents_dir) / "*.pt"))
                 if Path(p).stem != "index"]
        _dummy_outcomes = {m: False for m in _mids}
        print(f"Action-conditioned via ReplayLatentSequenceDataset | labels_root={args.labels_root} "
              f"| {len(_mids)} latent matches (rewards unused -> dummy outcomes)")

        def make_dataset(seq_len):
            return ReplayLatentSequenceDataset(
                latents_dir=args.latents_dir, labels_root=args.labels_root,
                outcomes=_dummy_outcomes, sequence_length=seq_len, stride=args.stride,
            )
    else:
        print("Using PACKED latent format, actions OFF (PackedLatentSequenceDataset)")

        def make_dataset(seq_len):
            return PackedLatentSequenceDataset(
                latents_dir=args.latents_dir, sequence_length=seq_len, stride=args.stride,
            )

    holdout_val_batch = None
    if args.alternating_lengths:
        dataset_short = make_dataset(args.seq_len_short)
        dataset_long = make_dataset(args.seq_len_long)
        holdout_vids = _pick_holdout(dataset_short, args.holdout_videos)
        if holdout_vids:
            print(f"Held-out eval videos ({len(holdout_vids)}): {sorted(holdout_vids)}")
            holdout_val_batch = _build_holdout_batch(dataset_short, holdout_vids, args.batch_size_short)

        if len(dataset_short) == 0 or len(dataset_long) == 0:
            print("ERROR: No sequences found!")
            if args.packed:
                print("Make sure to run pack_latents.py first.")
            else:
                print("Make sure to run pretokenize_frames.py first.")
            return

        # Verify action loading matches expectation
        if args.use_actions:
            sample = dataset_short[0]
            assert "actions" in sample and sample["actions"] is not None, \
                "use_actions=True but dataset not loading actions. " \
                "Check: dataset initialization must have load_actions=True"

        print(f"Short sequences: {len(dataset_short)} (T={args.seq_len_short})")
        print(f"Long sequences: {len(dataset_long)} (T={args.seq_len_long})")

        # Shortcut-step dataloaders (smaller batch, GC disabled during shortcut)
        dataloader_short = DataLoader(
            dataset_short,
            batch_size=args.batch_size_short,
            sampler=VideoGroupedSampler(dataset_short, exclude_videos=holdout_vids),
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        dataloader_long = DataLoader(
            dataset_long,
            batch_size=args.batch_size_long,
            sampler=VideoGroupedSampler(dataset_long, exclude_videos=holdout_vids),
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        dataloader = None
    else:
        dataset = make_dataset(args.sequence_length)
        holdout_vids = _pick_holdout(dataset, args.holdout_videos)
        if holdout_vids:
            print(f"Held-out eval videos ({len(holdout_vids)}): {sorted(holdout_vids)}")
            holdout_val_batch = _build_holdout_batch(dataset, holdout_vids, args.batch_size)

        if len(dataset) == 0:
            print("ERROR: No sequences found!")
            if args.packed:
                print("Make sure to run pack_latents.py first.")
            else:
                print("Make sure to run pretokenize_frames.py first.")
            return

        # Verify action loading matches expectation
        if args.use_actions:
            sample = dataset[0]
            assert "actions" in sample and sample["actions"] is not None, \
                "use_actions=True but dataset not loading actions. " \
                "Check: dataset initialization must have load_actions=True"

        print(f"Found {len(dataset)} sequences")

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=VideoGroupedSampler(dataset, exclude_videos=holdout_vids),
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

    if not args.no_compile:
        print("Compiling model with torch.compile(dynamic=True)...")
        model = torch.compile(model, dynamic=True)

    save_run_config(checkpoint_dir, args, num_params)

    wandb_run = init_wandb(args, job_type="dynamics", extra_config={
        "num_params": num_params,
        "checkpoint_dir": str(checkpoint_dir),
    })

    # Create optimizer + scheduler
    optimizer = create_optimizer(model.parameters(), args.lr, args.weight_decay, use_8bit=args.use_8bit_adam, betas=(0.9, 0.95))

    if dataloader is not None:
        steps_per_epoch = len(dataloader) // args.gradient_accumulation
    else:
        accum_s = args.gradient_accumulation_short or args.gradient_accumulation
        accum_l = args.gradient_accumulation_long or args.gradient_accumulation
        n_short = len(dataloader_short)
        n_long = int(n_short * args.long_ratio / (1 - args.long_ratio))
        steps_per_epoch = n_short // accum_s + n_long // accum_l
    total_steps = args.epochs * steps_per_epoch
    scheduler = create_wsd_schedule(optimizer, total_steps, args.warmup_steps, args.decay_steps)

    # Diffusion schedule + shortcut forcing
    schedule = DiffusionSchedule(device=args.device)

    shortcut = None
    if args.shortcut_forcing:
        shortcut = ShortcutForcing(k_max=args.shortcut_k_max, bootstrap_weight=args.bootstrap_weight)
        print(f"Shortcut forcing enabled (k_max={args.shortcut_k_max}, bootstrap_weight={args.bootstrap_weight})")

    # GradScaler: only useful for float16, no-op for bfloat16
    amp_dtype = torch.bfloat16 if args.device != "mps" else torch.float16
    scaler = GradScaler(args.device.split(":")[0], enabled=(amp_dtype == torch.float16))

    # RMS trackers (only for standard training, not shortcut forcing)
    rms_dict = None if args.shortcut_forcing else {"x_pred": RunningRMS()}

    # Resume
    start_epoch = 0
    global_step = 0
    resume_skip_batches = 0
    if args.resume:
        print(f"\nResuming from {args.resume}...")
        saved_epoch, global_step, _, saved_batch_idx = load_checkpoint(
            Path(args.resume), model, optimizer, scaler, scheduler=scheduler, rms_trackers=rms_dict, strict=False
        )
        start_epoch = saved_epoch
        resume_skip_batches = saved_batch_idx
        print(f"Resuming from epoch {start_epoch}, batch {resume_skip_batches}, global_step {global_step}")

    # Fixed validation batch for eval metrics
    val_batch = holdout_val_batch
    if val_batch is not None:
        print(f"Held-out val batch: {val_batch['latents'].shape}")
    elif dataloader_short is not None:
        val_batch = next(iter(dataloader_short))
    elif dataloader is not None:
        val_batch = next(iter(dataloader))
    if val_batch is not None and holdout_val_batch is None:
        print(f"Validation batch (training-drawn): {val_batch['latents'].shape}")

    # --- Build dataclasses for train_epoch ---
    ts = TrainingState(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=scheduler,
        global_step=global_step,
        rms_dict=rms_dict,
        shortcut=shortcut,
    )

    if args.alternating_lengths:
        accum_s = args.gradient_accumulation_short or args.gradient_accumulation
        accum_l = args.gradient_accumulation_long or args.gradient_accumulation
    else:
        accum_s = args.gradient_accumulation
        accum_l = None

    data_cfg = DataConfig(
        dataloader=dataloader,
        dataloader_short=dataloader_short,
        dataloader_long=dataloader_long,
        accumulation_steps=accum_s,
        accumulation_steps_long=accum_l,
        shortcut_batch_short=args.shortcut_batch_short,
        shortcut_batch_long=args.shortcut_batch_long,
        long_ratio=args.long_ratio,
    )

    ckpt_cfg = CheckpointConfig(
        checkpoint_dir=checkpoint_dir,
        base_checkpoint_dir=base_checkpoint_dir,
        name_prefix="dynamics",
    )

    # --- Training loop ---
    best_psnr = -float("inf")
    history = []

    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print("=" * 60)

        metrics = train_epoch(
            ts, data_cfg, ckpt_cfg, schedule, args.device, epoch, args,
            val_batch=val_batch,
            eval_interval=args.eval_interval,
            skip_batches=resume_skip_batches,
        )
        resume_skip_batches = 0  # only skip on first epoch after resume

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Grad Norm (avg): {metrics['grad_norm']:.4f}")
        print(f"  Pred Std (avg): {metrics['pred_std']:.4f}")

        log_step({
            "epoch/loss": metrics["loss"],
            "epoch/grad_norm": metrics["grad_norm"],
            "epoch/pred_std": metrics["pred_std"],
            "epoch/epoch": epoch + 1,
        }, step=ts.global_step)

        if metrics['pred_std'] < 0.01:
            print("  WARNING: Low prediction variance - possible mode collapse!")

        history.append({"epoch": epoch + 1, **metrics})

        # Epoch checkpoint (uses save_checkpoints helper)
        if (epoch + 1) % args.save_interval == 0:
            epoch_path = checkpoint_dir / f"dynamics_epoch_{epoch + 1:03d}.pt"
            # Store epoch+1 (the next epoch to run), batch_idx defaulting to 0, so
            # a resume from this boundary checkpoint continues at the next epoch
            # instead of silently re-training the epoch that just completed.
            save_checkpoints(ckpt_cfg, ts, epoch + 1, metrics["loss"], args, step_path=epoch_path)

        # Save best model based on validation PSNR
        if val_batch is not None:
            eval_results = eval_denoising_psnr(
                model, schedule, val_batch, args.device, shortcut=shortcut,
            )
            psnr_keys = [k for k in eval_results if k.startswith("eval/psnr_tau")]
            mean_psnr = sum(eval_results[k] for k in psnr_keys) / max(len(psnr_keys), 1)
            log_step({"epoch/mean_psnr": mean_psnr}, step=ts.global_step)
            # Deploy-regime rollout eval per epoch (logged for the drift curve;
            # best.pt still tracks the teacher-forced mean for stability).
            _rs, _rk = (4, shortcut.k_max) if shortcut is not None else (16, 16)
            _roll = eval_rollout_psnr(
                model, schedule, val_batch, args.device,
                num_steps=_rs, k_max=_rk, use_actions=args.use_actions,
            )
            if _roll:
                log_step({"epoch/rollout_psnr_mean": _roll["eval_rollout/psnr_mean"]},
                         step=ts.global_step)

            if mean_psnr > best_psnr:
                best_psnr = mean_psnr
                best_path = checkpoint_dir / "dynamics_best.pt"
                save_checkpoint(
                    best_path, model, optimizer, scaler, epoch, ts.global_step,
                    metrics["loss"], args, scheduler=scheduler, rms_trackers=rms_dict,
                )
                print(f"New best model saved (PSNR: {best_psnr:.2f} dB)")

        if metrics.get("preempted"):
            print("Preempted during epoch -- exiting training loop.", flush=True)
            break

    # Save training history
    history_path = checkpoint_dir / "dynamics_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")

    print("\n" + "=" * 60)
    print("Training complete!")
    if best_psnr > -float("inf"):
        print(f"Best PSNR: {best_psnr:.2f} dB")
    print(f"Best model: {checkpoint_dir / 'dynamics_best.pt'}")
    print("=" * 60)

    finish_wandb()


if __name__ == "__main__":
    main()
