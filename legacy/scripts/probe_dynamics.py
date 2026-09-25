#!/usr/bin/env python3
"""GPU probes for the dynamics model: VRAM limits and inference latency.

Uses synthetic (random) latents at a fixed ``--latent-dim`` — no dataset
needed — so it characterises the model independent of the data pipeline. Run
on the desktop 5080 via srun.

Modes
-----
``--mode vram``      Max *trainable* config per size (real forward+backward+
                     AdamW step, bf16 autocast): max seq-len at B=1, max batch
                     at T=64, and peak GB at a B=1,T=64 reference, with and
                     without gradient checkpointing.
``--mode latency``   Inference forward-pass latency -> autoregressive rollout
                     FPS. Per *new* frame the diffusion-forcing rollout runs K
                     denoising steps, each a full forward over the context
                     window (no KV cache in the current code), so
                     per-frame ms ~= K * forward(T=context). Reports K=4
                     (shortcut) and K=64 (full diffusion).
``--mode both``      Run latency then vram.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ahriuwu.models.dynamics import create_dynamics
from ahriuwu.models.diffusion import x_prediction_loss
from ahriuwu.models.layers import Attention, _FLEX_AVAILABLE


def set_flex(model, flag: bool) -> int:
    """Toggle the fused flex_attention backend on every Attention module.

    Dynamics ``TransformerBlock`` builds attention with ``allow_flex=False``
    (manual matmul) — fine for grad-checkpointed training, but it also forces
    the slow path at *inference*. This flips it so we can measure the real
    achievable inference latency.
    """
    n = 0
    for m in model.modules():
        if isinstance(m, Attention):
            m.allow_flex = bool(flag) and _FLEX_AVAILABLE
            n += 1
    return n

SIZES = ["tiny", "small", "medium", "large"]
SPATIAL = 16  # 16x16 = 256 latent tokens


def _is_oom(err: Exception) -> bool:
    return isinstance(err, torch.cuda.OutOfMemoryError) or (
        isinstance(err, RuntimeError) and "out of memory" in str(err).lower()
    )


def build(size, latent_dim, gc=False, kv=4, device="cuda"):
    return create_dynamics(
        size, latent_dim=latent_dim, num_kv_heads=kv,
        num_register_tokens=8, gradient_checkpointing=gc,
    ).to(device)


def _gpu_header(latent_dim):
    p = torch.cuda.get_device_properties(0)
    return (f"GPU={p.name}  {p.total_memory/1024**3:.1f} GB  "
            f"torch={torch.__version__}  latent_dim={latent_dim}  spatial={SPATIAL}x{SPATIAL}")


# --------------------------------------------------------------------------- #
# VRAM
# --------------------------------------------------------------------------- #

def train_step_peak(model, B, T, C, device):
    """One real train step (fwd+bwd+AdamW) at (B,T). Returns peak GB or None on OOM."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    try:
        z0 = torch.randn(B, T, C, SPATIAL, SPATIAL, device=device)
        tau = torch.rand(B, T, device=device)
        te = tau[..., None, None, None]
        ztau = te * z0 + (1 - te) * torch.randn_like(z0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(ztau, tau)
            loss = x_prediction_loss(pred, z0, tau)
        loss.backward()
        opt.step()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1024**3
    except Exception as e:  # noqa: BLE001
        if _is_oom(e):
            model.zero_grad(set_to_none=True)
            del opt
            torch.cuda.empty_cache()
            return None
        raise


def vram_mode(latent_dim, device):
    print(f"# VRAM probe  ({_gpu_header(latent_dim)})")
    print("# max trainable config: forward+backward+AdamW step, bf16 autocast, GQA kv=4")
    print(f"{'size':7} {'params':>8} {'GC':>4} {'maxT@B1':>8} {'maxB@T64':>9} {'peakGB@B1,T64':>14}")
    for size in SIZES:
        for gc in (False, True):
            model = build(size, latent_dim, gc=gc, device=device)
            model.train()
            params = model.get_num_params() / 1e6
            maxT = 0
            for T in [16, 32, 64, 128, 192, 256]:  # max_seq_len=256 is the arch cap
                if train_step_peak(model, 1, T, latent_dim, device) is not None:
                    maxT = T
                else:
                    break
            maxB = 0
            for B in [1, 2, 4, 8, 16, 32, 64]:
                if train_step_peak(model, B, 64, latent_dim, device) is not None:
                    maxB = B
                else:
                    break
            peak = train_step_peak(model, 1, 64, latent_dim, device)
            peak_s = f"{peak:.2f}" if peak is not None else "OOM"
            print(f"{size:7} {params:7.1f}M {('on' if gc else 'off'):>4} "
                  f"{maxT:>8} {maxB:>9} {peak_s:>14}")
            del model
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Latency
# --------------------------------------------------------------------------- #

@torch.no_grad()
def forward_ms(model, T, latent_dim, device, iters=20, warmup=3):
    z = torch.randn(1, T, latent_dim, SPATIAL, SPATIAL, device=device)
    tau = torch.full((1, T), 0.5, device=device)
    d = torch.full((1,), 16, dtype=torch.long, device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(warmup):
            out = model(z, tau, step_size=d)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            out = model(z, tau, step_size=d)
        torch.cuda.synchronize()
    if isinstance(out, tuple):
        out = out[0]
    return (time.perf_counter() - t0) / iters * 1000.0


def latency_mode(latent_dim, device, contexts=(32, 64, 128, 192)):
    print(f"# Latency probe  ({_gpu_header(latent_dim)})  flex_available={_FLEX_AVAILABLE}")
    print("# per-new-frame rollout ms ~= K x forward(T=context)  [no KV cache]")
    print("# real-time target: >=20 FPS  (per-frame <= 50 ms)")
    print("# backend: manual = eager matmul (dynamics default, allow_flex=False); "
          "flex = fused flex_attention (must be enabled for inference)")
    print(f"{'size':7} {'params':>8} {'backend':>7} {'T':>4} {'fwd_ms':>8} "
          f"{'K4_ms':>8} {'K4_fps':>7} {'K64_ms':>9} {'K64_fps':>8}")
    for size in SIZES:
        model = build(size, latent_dim, device=device).eval()
        params = model.get_num_params() / 1e6
        for backend in ("manual", "flex"):
            n = set_flex(model, backend == "flex")
            if backend == "flex" and n == 0:
                continue
            for T in contexts:
                try:
                    fwd = forward_ms(model, T, latent_dim, device)
                except Exception as e:  # noqa: BLE001
                    print(f"{size:7} {params:7.1f}M {backend:>7} {T:>4}  ERROR: {type(e).__name__}: {str(e)[:60]}")
                    continue
                k4, k64 = 4 * fwd, 64 * fwd
                print(f"{size:7} {params:7.1f}M {backend:>7} {T:>4} {fwd:8.2f} "
                      f"{k4:8.1f} {1000/k4:7.1f} {k64:9.1f} {1000/k64:8.2f}")
        del model
        torch.cuda.empty_cache()


@torch.no_grad()
def rollout_latency(latent_dim, device, ctx_frames=128, contexts=(4, 64)):
    """Marginal per-frame latency of the *cached* autoregressive rollout.

    Times rollout for P=2 and P=22 generated frames (after a fixed ctx prefill)
    and reports the marginal per-frame cost (t22-t2)/20, which cancels the
    one-time prefill + warmup. This is the real steady-state real-time number.
    """
    print(f"# Cached-rollout latency  ({_gpu_header(latent_dim)})")
    print(f"# prefill {ctx_frames} ctx frames; marginal per-frame over a 20-frame gen window")
    print(f"# real-time target: >=20 FPS (per-frame <= 50 ms)")
    print(f"{'size':7} {'params':>8} {'K':>4} {'per_frame_ms':>12} {'fps':>8}")

    def _time(model, ctx, K, P):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model.rollout(ctx, predict_frames=P, num_steps=K, k_max=64)
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    for size in SIZES:
        model = build(size, latent_dim, device=device).eval()
        params = model.get_num_params() / 1e6
        ctx = torch.randn(1, ctx_frames, latent_dim, SPATIAL, SPATIAL, device=device)
        for K in contexts:
            _time(model, ctx, K, 2)            # warmup
            t2 = _time(model, ctx, K, 2)
            t22 = _time(model, ctx, K, 22)
            per = (t22 - t2) / 20 * 1000
            print(f"{size:7} {params:7.1f}M {K:>4} {per:12.2f} {1000/max(per,1e-6):8.1f}")
        del model
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description="Dynamics VRAM + latency probes")
    ap.add_argument("--mode", choices=["vram", "latency", "rollout", "both"], default="both")
    ap.add_argument("--latent-dim", type=int, default=32)
    ap.add_argument("--ctx-frames", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "needs CUDA"
    torch.manual_seed(0)
    if args.mode == "rollout":
        rollout_latency(args.latent_dim, args.device, ctx_frames=args.ctx_frames)
        return
    if args.mode in ("latency", "both"):
        latency_mode(args.latent_dim, args.device)
        print()
    if args.mode in ("vram", "both"):
        vram_mode(args.latent_dim, args.device)


if __name__ == "__main__":
    main()
