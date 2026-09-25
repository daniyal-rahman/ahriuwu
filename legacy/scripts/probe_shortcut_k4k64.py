#!/usr/bin/env python3
"""Task 3: measure the x-space bootstrap deviation's impact on K=4 quality.

Trains a dynamics model with shortcut forcing (the CURRENT x-space bootstrap
loss in ``ShortcutForcing.compute_loss`` — UNCHANGED) on real latents, then
compares K=4 (d=k_max/4 shortcut) vs K=64 (d=1 full diffusion) *next-frame*
generation in LATENT space: from pure noise, conditioned on clean context
frames, denoise the next frame in K steps and measure latent PSNR vs ground
truth. If K=4 tracks K=64, the deviation is benign; a large gap flags it.

The step-size curriculum is accelerated vs the production trainer (which only
introduces d=16 at step 6000) so the large step sizes get meaningfully trained
within the probe budget.
"""
import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ahriuwu.data import PackedLatentSequenceDataset, VideoGroupedSampler
from ahriuwu.models import create_dynamics, DiffusionSchedule, x_prediction_loss
from ahriuwu.models.diffusion import ShortcutForcing


@torch.no_grad()
def gen_next_frame(model, context, gt_next, K, k_max, dev, noise0):
    """Generate the frame after `context` from noise in K steps; return latent PSNR.

    context: (B, C, Cc, 16, 16) clean latents (tau=1). The new frame starts as
    the GIVEN pure-noise tensor `noise0` (tau~0) and is Euler-denoised over K
    steps with shortcut step size d=k_max//K, while the context stays clean.
    Caller passes the SAME noise0 to the K=4 and K=64 calls so the comparison
    is paired (no noise-seed variance in the gap).
    """
    B, C = context.shape[:2]
    Cc = context.shape[2]
    eps = 1e-3
    step = (1.0 - eps) / K
    d = max(1, k_max // K)
    tgt = noise0.clone()
    for i in range(K):
        tau_t = eps + i * step
        seq = torch.cat([context, tgt], dim=1)
        tau = torch.ones(B, C + 1, device=dev)
        tau[:, -1] = tau_t  # context clean, target at current signal level
        dd = torch.full((B,), d, dtype=torch.long, device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            zp = model(seq, tau, step_size=dd)
        if isinstance(zp, tuple):
            zp = zp[0]
        z0t = zp[:, -1:].float()
        if i < K - 1:
            nt = tau_t + step
            tgt = nt * z0t + (1 - nt) * noise0
        else:
            tgt = z0t
    mse = ((tgt.float() - gt_next.float()) ** 2).mean().item()
    mx = gt_next.abs().max().item()
    return 10 * torch.log10(torch.tensor(mx ** 2 / max(mse, 1e-10))).item()


def evaluate(model, vb, k_max, dev, ctx=12, n_seeds=4):
    """Paired K=4 vs K=64: same seeded noise0 for both K, averaged over n_seeds."""
    model.eval()
    z0 = vb.to(dev)
    context = z0[:, :ctx]
    gt_next = z0[:, ctx:ctx + 1]
    B, Cc = z0.shape[0], z0.shape[2]
    p4s, p64s = [], []
    for s in range(n_seeds):
        g = torch.Generator(device=dev).manual_seed(1234 + s)
        noise0 = torch.randn(B, 1, Cc, 16, 16, device=dev, generator=g)
        p4s.append(gen_next_frame(model, context, gt_next, 4, k_max, dev, noise0))
        p64s.append(gen_next_frame(model, context, gt_next, 64, k_max, dev, noise0))
    model.train()
    return sum(p4s) / n_seeds, sum(p64s) / n_seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents-dir", required=True)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--latent-dim", type=int, default=48)
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--k-max", type=int, default=64)
    ap.add_argument("--size", default="small")
    ap.add_argument("--accel", type=int, default=400,
                    help="steps between curriculum stage bumps (smaller = faster)")
    ap.add_argument("--eval-every", type=int, default=1000)
    args = ap.parse_args()
    dev = "cuda"

    ds = PackedLatentSequenceDataset(args.latents_dir, sequence_length=args.seq_len, stride=8)
    dl = DataLoader(ds, batch_size=args.batch_size, sampler=VideoGroupedSampler(ds),
                    num_workers=4, drop_last=True, pin_memory=True)
    sched = DiffusionSchedule(device=dev)
    sc = ShortcutForcing(k_max=args.k_max, bootstrap_weight=3.0)
    vb = next(iter(dl))["latents"]

    m = create_dynamics(args.size, latent_dim=args.latent_dim, num_kv_heads=4,
                        num_register_tokens=8, gradient_checkpointing=True).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    nstage = len(sc.step_sizes) - 1

    print(f"# Task3 K4-vs-K64: size={args.size} latent_dim={args.latent_dim} "
          f"k_max={args.k_max} steps={args.steps} accel={args.accel} | {len(ds)} seqs")
    print(f"# K4 uses d={args.k_max//4} (shortcut, x-space bootstrap); K64 uses d=1 (full diffusion)")
    m.train()
    it = iter(dl)
    t0 = time.time()
    for step in range(args.steps):
        try:
            b = next(it)
        except StopIteration:
            it = iter(dl)
            b = next(it)
        z0 = b["latents"].to(dev)
        B, T = z0.shape[:2]
        if torch.rand(1).item() < 0.5:
            # standard diffusion-forcing x-prediction
            tau = sched.sample_diffusion_forcing_timesteps(B, T, device=dev)
            zt, _ = sched.add_noise(z0, tau)
            ui = torch.rand(1).item() < 0.3
            with torch.autocast("cuda", dtype=torch.bfloat16):
                zp = m(zt, tau, independent_frames=ui)
                raw = x_prediction_loss(zp, z0, tau, use_ramp_weight=True)
        else:
            # shortcut forcing with accelerated curriculum (3 fwd -> disable GC)
            max_idx = min(1 + step // args.accel, nstage)
            d = sc.sample_step_size(B, device=dev, max_step_idx=max_idx)
            tau = sc.sample_tau_for_step_size_2d(d, T, device=dev)
            m.gradient_checkpointing = False
            with torch.autocast("cuda", dtype=torch.bfloat16):
                raw, info = sc.compute_loss(m, sched, z0, tau, d)
            m.gradient_checkpointing = True
        if not torch.isfinite(raw):
            continue
        opt.zero_grad()
        raw.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if step % args.eval_every == 0 or step == args.steps - 1:
            p4, p64 = evaluate(m, vb, args.k_max, dev)
            ms = (time.time() - t0) / max(step, 1) * 1000
            print(f"step {step:5d}: K4={p4:6.2f} dB  K64={p64:6.2f} dB  "
                  f"gap(K64-K4)={p64 - p4:5.2f} dB  ({ms:.0f} ms/step)", flush=True)
    print("done")


if __name__ == "__main__":
    main()
