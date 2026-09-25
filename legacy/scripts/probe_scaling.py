#!/usr/bin/env python3
"""Short-horizon scaling sweep: loss/denoising-PSNR vs params on REAL latents.

Trains tiny/small/medium/large for a fixed step budget on the SAME data
(same seed, same batches) and reports final train loss + 1-step denoising
PSNR. Uses real packed latents (structured) because a loss-vs-params curve on
synthetic Gaussian latents is uninformative — every size converges to the same
closed-form Gaussian-denoising floor. This is an *indicative* short-horizon
trend, not a converged scaling law (which needs the production data + long
training on the cloud).
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ahriuwu.data import PackedLatentSequenceDataset, VideoGroupedSampler
from ahriuwu.models import create_dynamics, DiffusionSchedule, x_prediction_loss, RunningRMS
from torch.utils.data import DataLoader


@torch.no_grad()
def eval_psnr(model, sched, vb, dev):
    model.eval()
    z0 = vb.to(dev)
    B, T = z0.shape[:2]
    out = {}
    for tv in (0.5, 0.9):
        tau = torch.full((B, T), tv, device=dev)
        zt, _ = sched.add_noise(z0, tau)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            zp = model(zt, tau, step_size=torch.ones(B, dtype=torch.long, device=dev))
        if isinstance(zp, tuple):
            zp = zp[0]
        mse = ((zp.float() - z0.float()) ** 2).mean().item()
        mx = z0.abs().max().item()
        out[tv] = 10 * torch.log10(torch.tensor(mx ** 2 / max(mse, 1e-10))).item()
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents-dir", required=True)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--latent-dim", type=int, default=48)
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--sizes", default="tiny,small,medium,large")
    args = ap.parse_args()
    dev = "cuda"

    ds = PackedLatentSequenceDataset(args.latents_dir, sequence_length=args.seq_len, stride=8)
    dl = DataLoader(ds, batch_size=args.batch_size, sampler=VideoGroupedSampler(ds),
                    num_workers=4, drop_last=True, pin_memory=True)
    sched = DiffusionSchedule(device=dev)
    vb = next(iter(dl))["latents"]  # fixed val batch shared across sizes

    print(f"# scaling sweep: {args.steps} steps, T={args.seq_len}, B={args.batch_size}, "
          f"latent_dim={args.latent_dim}, {len(ds)} seqs")
    print(f"{'size':7} {'params':>8} {'loss_last50':>11} {'psnr_t0.5':>9} {'psnr_t0.9':>9} {'min/k_steps':>11} {'sec':>6}")
    for size in [s.strip() for s in args.sizes.split(",")]:
        torch.manual_seed(0)
        m = create_dynamics(size, latent_dim=args.latent_dim, num_kv_heads=4,
                            num_register_tokens=8, gradient_checkpointing=True).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
        rms = RunningRMS()
        m.train()
        it = iter(dl)
        t0 = time.time()
        losses = []
        for step in range(args.steps):
            try:
                b = next(it)
            except StopIteration:
                it = iter(dl)
                b = next(it)
            z0 = b["latents"].to(dev)
            B, T = z0.shape[:2]
            tau = sched.sample_diffusion_forcing_timesteps(B, T, device=dev)
            zt, _ = sched.add_noise(z0, tau)
            ui = torch.rand(1).item() < 0.3
            with torch.autocast("cuda", dtype=torch.bfloat16):
                zp = m(zt, tau, independent_frames=ui)
                raw = x_prediction_loss(zp, z0, tau, use_ramp_weight=True)
            ln = rms.update(raw)
            opt.zero_grad()
            ln.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            losses.append(raw.item())
        ev = eval_psnr(m, sched, vb, dev)
        dt = time.time() - t0
        print(f"{size:7} {m.get_num_params()/1e6:7.1f}M {sum(losses[-50:])/50:11.5f} "
              f"{ev[0.5]:9.2f} {ev[0.9]:9.2f} {dt/args.steps*1000:11.1f} {dt:6.0f}", flush=True)
        del m, opt
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
