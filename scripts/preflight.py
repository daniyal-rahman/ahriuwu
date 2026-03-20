#!/usr/bin/env python3
"""Pre-flight check: run BEFORE any training to catch bugs early.

Tests every code path that training will hit, with the actual model config.
Every test must pass or training WILL fail hours in.

Usage:
    srun --mem=14G --gres=gpu:1 --qos=short --time=00:30:00 python scripts/preflight.py --model-size medium
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from ahriuwu.models import create_dynamics, DiffusionSchedule
from ahriuwu.models.diffusion import ShortcutForcing, x_prediction_loss, ramp_weight
from ahriuwu.data import PackedLatentSequenceDataset, VideoGroupedSampler
from ahriuwu.constants import MOVEMENT_DIM, ABILITY_KEYS

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
failures = []


def test(name):
    """Decorator for test functions."""
    def decorator(fn):
        def wrapper(*args, **kwargs):
            try:
                fn(*args, **kwargs)
                print(f"  [{PASS}] {name}")
            except Exception as e:
                failures.append((name, str(e)))
                print(f"  [{FAIL}] {name}: {e}")
                traceback.print_exc()
        wrapper.__name__ = fn.__name__
        return wrapper
    return decorator


def run_preflight(args):
    device = "cuda"
    schedule = DiffusionSchedule(device=device)
    shortcut = ShortcutForcing(k_max=args.shortcut_k_max)

    print("=" * 60)
    print(f"PRE-FLIGHT CHECK: {args.model_size} model, latent_dim={args.latent_dim}")
    print("=" * 60)

    # ===== Section 1: Model creation and init =====
    print("\n--- Model Creation ---")

    model = None

    @test("Model creates without error")
    def _():
        nonlocal model
        model = create_dynamics(
            args.model_size, latent_dim=args.latent_dim,
            num_kv_heads=args.num_kv_heads, num_register_tokens=args.num_register_tokens,
            soft_cap=args.soft_cap, gradient_checkpointing=args.gradient_checkpointing,
        ).to(device)
    _()

    if model is None:
        print("\nModel creation failed. Aborting.")
        return False

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {num_params/1e6:.1f}M")

    @test("Tau embeddings are diverse (not identical)")
    def _():
        w = model.tau_embed.weight.data
        cos_sim = nn.functional.cosine_similarity(w[6:7], w[57:58]).item()
        assert abs(cos_sim) < 0.5, f"tau embeddings too similar: cos_sim={cos_sim:.3f}"
    _()

    @test("Step embeddings are diverse")
    def _():
        w = model.step_embed.weight.data
        cos_sim = nn.functional.cosine_similarity(w[0:1], w[3:4]).item()
        assert abs(cos_sim) < 0.5, f"step embeddings too similar: cos_sim={cos_sim:.3f}"
    _()

    # ===== Section 2: Forward pass =====
    print("\n--- Forward Pass ---")

    B_short, T_short = args.batch_size_short, args.seq_len_short
    B_long, T_long = args.batch_size_long, args.seq_len_long

    @test(f"Standard forward B={B_short} T={T_short}")
    def _():
        z = torch.randn(B_short, T_short, args.latent_dim, 16, 16, device=device)
        tau = torch.full((B_short, T_short), 0.5, device=device)
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(z, tau, step_size=torch.ones(B_short, dtype=torch.long, device=device))
        assert out.shape == z.shape, f"Shape mismatch: {out.shape} vs {z.shape}"
        assert not torch.isnan(out).any(), "NaN in output"
    _()

    @test(f"Standard forward B={B_long} T={T_long} (long batch)")
    def _():
        z = torch.randn(B_long, T_long, args.latent_dim, 16, 16, device=device)
        tau = torch.full((B_long, T_long), 0.5, device=device)
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(z, tau, step_size=torch.ones(B_long, dtype=torch.long, device=device))
        assert out.shape == z.shape
    _()

    # ===== Section 3: Tau conditioning =====
    print("\n--- Tau Conditioning ---")

    @test("Output changes with tau (conditioning is alive)")
    def _():
        z = torch.randn(1, 8, args.latent_dim, 16, 16, device=device)
        outputs = {}
        with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16):
            for tv in [0.1, 0.9]:
                tau = torch.full((1, 8), tv, device=device)
                outputs[tv] = model(z, tau, step_size=torch.ones(1, dtype=torch.long, device=device)).float()
        diff = (outputs[0.1] - outputs[0.9]).abs().mean().item()
        assert diff > 1e-5, f"Outputs identical for tau=0.1 vs 0.9: diff={diff}"
    _()

    @test("Output changes with step_size (shortcut conditioning alive)")
    def _():
        z = torch.randn(1, 8, args.latent_dim, 16, 16, device=device)
        tau = torch.full((1, 8), 0.5, device=device)
        outputs = {}
        with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16):
            for d in [1, 4]:
                outputs[d] = model(z, tau, step_size=torch.full((1,), d, dtype=torch.long, device=device)).float()
        diff = (outputs[1] - outputs[4]).abs().mean().item()
        assert diff > 1e-5, f"Outputs identical for d=1 vs d=4: diff={diff}"
    _()

    @test("Gradients flow through tau_embed")
    def _():
        model.train()
        model.zero_grad()
        z_0 = torch.randn(1, 8, args.latent_dim, 16, 16, device=device)
        tau = torch.full((1, 8), 0.5, device=device)
        z_tau, _ = schedule.add_noise(z_0, tau)
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(z_tau, tau, step_size=torch.ones(1, dtype=torch.long, device=device))
            loss = ((out.float() - z_0.float()) ** 2).mean()
        loss.backward()
        g = model.tau_embed.weight.grad
        assert g is not None and g.norm().item() > 1e-8, f"tau_embed grad is zero: {g.norm().item() if g is not None else 'None'}"
    _()

    @test("Gradients flow through cond_proj")
    def _():
        g = model.cond_proj.weight.grad
        assert g is not None and g.norm().item() > 1e-8, f"cond_proj grad is zero"
    _()

    # ===== Section 4: Loss computation =====
    print("\n--- Loss Computation ---")

    @test("x_prediction_loss produces finite values")
    def _():
        pred = torch.randn(2, 8, args.latent_dim, 16, 16, device=device)
        target = torch.randn_like(pred)
        tau = torch.full((2, 8), 0.5, device=device)
        loss = x_prediction_loss(pred, target, tau, use_ramp_weight=True)
        assert torch.isfinite(loss), f"Loss not finite: {loss.item()}"
        assert loss.item() > 0, f"Loss is zero"
    _()

    @test("Ramp weight convention correct (w(1.0)=1.0, w(0.0)=0.1)")
    def _():
        assert abs(ramp_weight(torch.tensor(1.0)).item() - 1.0) < 1e-6
        assert abs(ramp_weight(torch.tensor(0.0)).item() - 0.1) < 1e-6
    _()

    @test("add_noise convention (tau=0 noise, tau=1 clean)")
    def _():
        clean = torch.ones(1, 1, 1, 1, 1, device=device) * 10
        noise = torch.zeros(1, 1, 1, 1, 1, device=device)
        z0, _ = schedule.add_noise(clean, torch.tensor([0.0], device=device), noise=noise)
        z1, _ = schedule.add_noise(clean, torch.tensor([1.0], device=device), noise=noise)
        assert abs(z0.item() - 0.0) < 1e-5, f"tau=0 should give noise, got {z0.item()}"
        assert abs(z1.item() - 10.0) < 1e-5, f"tau=1 should give clean, got {z1.item()}"
    _()

    # ===== Section 5: Shortcut forcing =====
    print("\n--- Shortcut Forcing ---")

    @test(f"Shortcut compute_loss B={B_short} T={T_short}")
    def _():
        model.train()
        z_0 = torch.randn(B_short, T_short, args.latent_dim, 16, 16, device=device)
        step_size = shortcut.sample_step_size(B_short, device=device)
        tau = shortcut.sample_tau_for_step_size_2d(step_size, T_short, device=device)
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, info = shortcut.compute_loss(model, schedule, z_0, tau, step_size)
        assert torch.isfinite(loss), f"Shortcut loss not finite"
        assert info["n_std"] + info["n_boot"] > 0, "No samples in either path"
    _()

    @test(f"Shortcut compute_loss B={B_long} T={T_long} (long batch)")
    def _():
        z_0 = torch.randn(B_long, T_long, args.latent_dim, 16, 16, device=device)
        step_size = shortcut.sample_step_size(B_long, device=device)
        tau = shortcut.sample_tau_for_step_size_2d(step_size, T_long, device=device)
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, info = shortcut.compute_loss(model, schedule, z_0, tau, step_size)
        assert torch.isfinite(loss), f"Shortcut loss not finite"
    _()

    @test("Shortcut backward produces gradients (GC disabled)")
    def _():
        model.zero_grad()
        # Disable GC for shortcut (matches training — autocast+GC incompatible)
        gc_was = model.gradient_checkpointing
        model.gradient_checkpointing = False
        z_0 = torch.randn(B_short, T_short, args.latent_dim, 16, 16, device=device)
        step_size = shortcut.sample_step_size(B_short, device=device)
        tau = shortcut.sample_tau_for_step_size_2d(step_size, T_short, device=device)
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, info = shortcut.compute_loss(model, schedule, z_0, tau, step_size)
        loss.backward()
        model.gradient_checkpointing = gc_was
        total_grad = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        assert total_grad > 0, "No gradients after backward"
    _()

    # ===== Section 6: VRAM profiling =====
    print("\n--- VRAM Profiling ---")

    vram_results = {}
    for step_type in ["standard", "shortcut"]:
        for T, B, label in [(T_short, B_short, "short"), (T_long, B_long, "long")]:
            for test_B in [B, B * 2, B * 4, B * 8]:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                test_model = create_dynamics(
                    args.model_size, latent_dim=args.latent_dim,
                    num_kv_heads=args.num_kv_heads, num_register_tokens=args.num_register_tokens,
                    soft_cap=args.soft_cap, gradient_checkpointing=args.gradient_checkpointing,
                ).to(device)
                opt = torch.optim.AdamW(test_model.parameters(), lr=3e-4)

                z_0 = torch.randn(test_B, T, args.latent_dim, 16, 16, device=device)
                try:
                    if step_type == "standard":
                        tau = torch.full((test_B, T), 0.5, device=device)
                        z_tau, _ = schedule.add_noise(z_0, tau)
                        with autocast("cuda", dtype=torch.bfloat16):
                            pred = test_model(z_tau, tau, step_size=torch.ones(test_B, dtype=torch.long, device=device))
                            loss = ((pred.float() - z_0.float()) ** 2).mean()
                        loss.backward()
                        opt.step()
                    else:
                        # Disable GC for shortcut (matches training behavior)
                        test_model.gradient_checkpointing = False
                        ss = shortcut.sample_step_size(test_B, device=device)
                        tau = shortcut.sample_tau_for_step_size_2d(ss, T, device=device)
                        with autocast("cuda", dtype=torch.bfloat16):
                            loss, _ = shortcut.compute_loss(test_model, schedule, z_0, tau, ss)
                        loss.backward()
                        opt.step()

                    peak = torch.cuda.max_memory_allocated() / 1e9
                    total = torch.cuda.get_device_properties(0).total_memory / 1e9
                    status = "OK" if peak < total * 0.9 else "TIGHT"
                    vram_results[(step_type, label, test_B)] = peak
                    print(f"  {step_type:10s} T={T:2d} B={test_B:2d} -> {peak:5.1f}/{total:.0f} GB  {status}")
                except torch.cuda.OutOfMemoryError:
                    print(f"  {step_type:10s} T={T:2d} B={test_B:2d} -> OOM")
                    test_model = opt = z_0 = None
                    torch.cuda.empty_cache()
                    break

                test_model = opt = z_0 = None
                torch.cuda.empty_cache()

    # ===== Section 7: Overfit test =====
    print("\n--- Overfit Test (50 steps) ---")

    @test("Model can overfit single batch")
    def _():
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        z_0 = torch.randn(1, 8, args.latent_dim, 16, 16, device=device)
        tau = torch.full((1, 8), 0.3, device=device)
        z_tau, _ = schedule.add_noise(z_0, tau)
        for i in range(50):
            opt.zero_grad()
            with autocast("cuda", dtype=torch.bfloat16):
                pred = model(z_tau, tau, step_size=torch.ones(1, dtype=torch.long, device=device))
                loss = ((pred.float() - z_0.float()) ** 2).mean()
            loss.backward()
            opt.step()
        assert loss.item() < 0.05, f"Failed to overfit: loss={loss.item():.4f}"
    _()

    @test("After overfit: tau=0.9 PSNR > tau=0.1 PSNR")
    def _():
        model.eval()
        z_0 = torch.randn(1, 8, args.latent_dim, 16, 16, device=device)
        psnrs = {}
        with torch.no_grad(), autocast("cuda", dtype=torch.bfloat16):
            for tv in [0.1, 0.9]:
                tau = torch.full((1, 8), tv, device=device)
                z_tau, _ = schedule.add_noise(z_0, tau)
                pred = model(z_tau, tau, step_size=torch.ones(1, dtype=torch.long, device=device))
                mse = ((pred.float() - z_0.float()) ** 2).mean().item()
                max_val = z_0.abs().max().item()
                psnrs[tv] = 10 * torch.log10(torch.tensor(max_val ** 2 / max(mse, 1e-10))).item()
        print(f"    tau=0.1: {psnrs[0.1]:.1f} dB, tau=0.9: {psnrs[0.9]:.1f} dB")
        assert psnrs[0.9] > psnrs[0.1], f"tau=0.9 ({psnrs[0.9]:.1f}) should beat tau=0.1 ({psnrs[0.1]:.1f})"
    _()

    # ===== Section 8: Data loading =====
    print("\n--- Data Loading ---")

    @test("PackedLatentSequenceDataset loads from .pt files")
    def _():
        ds = PackedLatentSequenceDataset(latents_dir=args.latents_dir, sequence_length=T_short)
        assert len(ds) > 0, "Dataset is empty"
        batch = ds[0]
        assert "latents" in batch, "Missing 'latents' key"
        assert batch["latents"].shape[0] == T_short, f"Wrong T: {batch['latents'].shape[0]}"
        print(f"    {len(ds)} sequences, shape={batch['latents'].shape}")
    _()

    @test("VideoGroupedSampler produces valid indices")
    def _():
        ds = PackedLatentSequenceDataset(latents_dir=args.latents_dir, sequence_length=T_short)
        sampler = VideoGroupedSampler(ds)
        indices = list(sampler)[:100]
        assert len(indices) == 100
        assert all(0 <= i < len(ds) for i in indices), "Invalid index from sampler"
    _()

    # ===== Summary =====
    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED: {len(failures)} test(s)")
        for name, err in failures:
            print(f"  - {name}: {err}")
        return False
    else:
        print("ALL TESTS PASSED")
        print("\nRecommended batch sizes from VRAM profiling:")
        for (stype, label, B), peak in sorted(vram_results.items()):
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            if peak < total * 0.85:
                print(f"  {stype:10s} {label}: B={B} ({peak:.1f} GB)")
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", default="medium")
    parser.add_argument("--latent-dim", type=int, default=48)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--num-register-tokens", type=int, default=8)
    parser.add_argument("--soft-cap", type=float, default=50.0)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--shortcut-k-max", type=int, default=64)
    parser.add_argument("--batch-size-short", type=int, default=2)
    parser.add_argument("--batch-size-long", type=int, default=1)
    parser.add_argument("--seq-len-short", type=int, default=32)
    parser.add_argument("--seq-len-long", type=int, default=64)
    parser.add_argument("--latents-dir", type=str, default="/opt/ahriuwu/latents_pt")
    args = parser.parse_args()

    success = run_preflight(args)
    sys.exit(0 if success else 1)
