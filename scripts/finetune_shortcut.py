#!/usr/bin/env python3
"""Phase B: Shortcut finetuning.

Freezes flow matching weights, trains only step_size conditioning + output
projection using bootstrap loss with large step sizes (d ∈ {8, 16, 32, 64}).

Usage:
    python scripts/finetune_shortcut.py \
        --checkpoint /mnt/storage/ahriuwu-data/checkpoints/dynamics_latest.pt \
        --latents-dir /opt/ahriuwu/latents_pt \
        --steps 5000 --lr 1e-3
"""

import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader

from ahriuwu.models import create_dynamics, DiffusionSchedule
from ahriuwu.models.diffusion import ShortcutForcing
from ahriuwu.data import PackedLatentSequenceDataset, VideoGroupedSampler
from ahriuwu.utils.training import save_checkpoint, load_checkpoint
from ahriuwu.utils.logging import init_wandb, log_step


def eval_k4(model, schedule, shortcut, val_batch, device):
    """Quick K=4 PSNR eval."""
    model.eval()
    z_0 = val_batch["latents"].to(device)
    B, T = z_0.shape[:2]

    # Seeded noise for reproducibility
    rng = torch.cuda.get_rng_state()
    torch.cuda.manual_seed(42)
    z_t = torch.randn_like(z_0)
    torch.cuda.set_rng_state(rng)
    z_noise = z_t.clone()

    K = 4
    eps = 1e-3
    step_size = (1.0 - eps) / K

    with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16):
        for i in range(K):
            tau_val = eps + i * step_size
            tau = torch.full((B, T), tau_val, device=device)
            d = torch.full((B,), shortcut.k_max // K, dtype=torch.long, device=device)
            z_pred = model(z_t, tau, step_size=d)
            if isinstance(z_pred, tuple):
                z_pred = z_pred[0]
            if i < K - 1:
                next_tau = tau_val + step_size
                z_t = next_tau * z_pred + (1 - next_tau) * z_noise
            else:
                z_t = z_pred

    mse = ((z_t.float() - z_0.float()) ** 2).mean().item()
    max_val = z_0.abs().max().item()
    psnr = 10 * torch.log10(torch.tensor(max_val ** 2 / max(mse, 1e-10))).item()

    # Also eval single-step at tau=0.9 for reference
    tau_09 = torch.full((B, T), 0.9, device=device)
    z_tau, _ = schedule.add_noise(z_0, tau_09)
    with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16):
        z_pred_1step = model(z_tau, tau_09, step_size=torch.ones(B, dtype=torch.long, device=device))
        if isinstance(z_pred_1step, tuple):
            z_pred_1step = z_pred_1step[0]
    mse_1step = ((z_pred_1step.float() - z_0.float()) ** 2).mean().item()
    psnr_1step = 10 * torch.log10(torch.tensor(max_val ** 2 / max(mse_1step, 1e-10))).item()

    model.train()
    return psnr, psnr_1step


def main():
    parser = argparse.ArgumentParser(description="Phase B: shortcut finetuning")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--latents-dir", type=str, default="/opt/ahriuwu/latents_pt")
    parser.add_argument("--output-dir", type=str, default="/mnt/storage/ahriuwu-data/checkpoints")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--k-max", type=int, default=64)
    parser.add_argument("--model-size", type=str, default="medium")
    parser.add_argument("--latent-dim", type=int, default=48)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--num-register-tokens", type=int, default=8)
    parser.add_argument("--soft-cap", type=float, default=50.0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="ahriuwu")
    args = parser.parse_args()

    device = "cuda"
    schedule = DiffusionSchedule(device=device)
    shortcut = ShortcutForcing(k_max=args.k_max)

    # Large step sizes only — these are what K=4 inference uses
    large_step_sizes = [d for d in shortcut.step_sizes if d >= 8]
    print(f"Phase B: shortcut finetuning with d ∈ {large_step_sizes}")

    # Create and load model
    model = create_dynamics(
        args.model_size, latent_dim=args.latent_dim,
        num_kv_heads=args.num_kv_heads, num_register_tokens=args.num_register_tokens,
        soft_cap=args.soft_cap, gradient_checkpointing=False,
    ).to(device)

    ckpt = torch.load(args.checkpoint, weights_only=False, map_location=device)
    sd = ckpt["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    print(f"Loaded from {args.checkpoint} (step {ckpt.get('global_step', '?')})")

    # Freeze everything except step_size conditioning + output projection
    trainable_names = []
    for name, param in model.named_parameters():
        if "step_embed" in name or "cond_proj" in name or "output_proj" in name:
            param.requires_grad = True
            trainable_names.append(name)
        else:
            param.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params/1e3:.1f}K / {total_params/1e6:.1f}M ({100*trainable_params/total_params:.1f}%)")
    print(f"Trainable layers: {trainable_names}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.0,
    )

    # Data
    dataset = PackedLatentSequenceDataset(
        latents_dir=args.latents_dir, sequence_length=args.seq_len,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=VideoGroupedSampler(dataset),
        num_workers=4, pin_memory=True, drop_last=True,
    )
    data_iter = iter(dataloader)

    # Val batch
    val_batch = next(iter(dataloader))

    # Wandb
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, name="shortcut_finetune",
                   config=vars(args))

    # Training
    model.train()
    best_k4_psnr = -float("inf")
    start_time = time.time()

    print(f"\nStarting Phase B: {args.steps} steps, lr={args.lr}")
    print("=" * 60)

    for step in range(args.steps):
        # Get batch (cycle through data)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        z_0 = batch["latents"].to(device)
        B, T = z_0.shape[:2]

        # Sample ONLY large step sizes
        d_idx = random.randint(0, len(large_step_sizes) - 1)
        step_size = torch.full((B,), large_step_sizes[d_idx], dtype=torch.long, device=device)

        # Grid-aligned tau
        tau = shortcut.sample_tau_for_step_size_2d(step_size, T, device=device)

        # Bootstrap loss only (no standard flow loss)
        optimizer.zero_grad()
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, info = shortcut.compute_loss(model, schedule, z_0, tau, step_size)

        if torch.isfinite(loss) and loss.item() > 0:
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()
        else:
            grad_norm = torch.tensor(0.0)

        # Logging
        if step % 10 == 0:
            elapsed = time.time() - start_time
            d_val = step_size[0].item()
            print(f"  Step {step:5d}/{args.steps} | d={d_val:2d} | "
                  f"loss={loss.item():.6f} (std:{info['loss_std']:.4f} boot:{info['loss_boot']:.4f}) | "
                  f"GradN={grad_norm.item():.4f} | {step/(elapsed+1e-6):.1f} step/s")

            if args.wandb:
                log_step({
                    "shortcut/loss": loss.item(),
                    "shortcut/loss_std": info["loss_std"],
                    "shortcut/loss_boot": info["loss_boot"],
                    "shortcut/grad_norm": grad_norm.item(),
                    "shortcut/step_size": d_val,
                }, step=step)

        # Eval
        if step % args.eval_interval == 0:
            k4_psnr, single_psnr = eval_k4(model, schedule, shortcut, val_batch, device)
            print(f"  [EVAL step {step}] K4={k4_psnr:.1f} dB | 1-step(τ=0.9)={single_psnr:.1f} dB")

            if args.wandb:
                log_step({
                    "shortcut/psnr_K4": k4_psnr,
                    "shortcut/psnr_1step": single_psnr,
                }, step=step)

            if k4_psnr > best_k4_psnr:
                best_k4_psnr = k4_psnr
                # Save best shortcut model
                torch.save(model.state_dict(), Path(args.output_dir) / "dynamics_shortcut_best.pt")
                print(f"  New best K4: {best_k4_psnr:.1f} dB")

    # Final eval
    k4_psnr, single_psnr = eval_k4(model, schedule, shortcut, val_batch, device)
    print(f"\nFinal: K4={k4_psnr:.1f} dB | 1-step={single_psnr:.1f} dB | Best K4={best_k4_psnr:.1f} dB")

    # Save final
    torch.save(model.state_dict(), Path(args.output_dir) / "dynamics_shortcut_final.pt")
    print(f"Saved to {args.output_dir}/dynamics_shortcut_final.pt")


if __name__ == "__main__":
    main()
