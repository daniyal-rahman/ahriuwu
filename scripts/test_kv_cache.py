#!/usr/bin/env python3
"""Equivalence test for the dynamics KV-cache rollout.

The cached temporal path must produce numerically identical next-frame
predictions to the full non-cached forward over the same sequence. Runs on CPU
in fp32 for a tight tolerance. This is the correctness gate for the KV cache.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import torch
from ahriuwu.models.dynamics import create_dynamics

TOL = 1e-4


def build_model():
    torch.manual_seed(0)
    return create_dynamics("tiny", latent_dim=16, num_kv_heads=2,
                           num_register_tokens=4).float().eval()


def cache_slots(m):
    n = sum(1 for b in m.blocks if b.attn_type == "temporal")
    return [{"k": None, "v": None, "pos": 0} for _ in range(n)]


@torch.no_grad()
def test_prefill_decode():
    m = build_model()
    B, Cc, Ctx = 2, 16, 6
    z_ctx = torch.randn(B, Ctx, Cc, 16, 16)
    z_tgt = torch.randn(B, 1, Cc, 16, 16)
    tau_ctx = torch.full((B, Ctx), 0.95)
    tau_tgt = torch.full((B, 1), 0.5)
    one = torch.ones(B, dtype=torch.long)
    ref = m(torch.cat([z_ctx, z_tgt], 1), torch.cat([tau_ctx, tau_tgt], 1),
            step_size=one)[:, Ctx:Ctx + 1]
    c = cache_slots(m)
    m._run_blocks(m._build_tokens(z_ctx, tau_ctx, one), caches=c, append=True)
    xt = m._run_blocks(m._build_tokens(z_tgt, tau_tgt, one), caches=c, append=False)
    cached = m._project_out(xt, B, 1, Cc, 16, 16)
    d = (ref - cached).abs().max().item()
    print(f"prefill+decode         max|Δ| = {d:.2e}")
    assert d < TOL, "FAILED prefill+decode equivalence"


@torch.no_grad()
def test_commit():
    m = build_model()
    B, Cc, Ctx = 2, 16, 5
    z_ctx = torch.randn(B, Ctx, Cc, 16, 16)
    z_a = torch.randn(B, 1, Cc, 16, 16)  # committed clean frame
    z_b = torch.randn(B, 1, Cc, 16, 16)  # decoded frame
    tau_ctx = torch.full((B, Ctx), 0.95)
    tau_a = torch.full((B, 1), 1.0)
    tau_b = torch.full((B, 1), 0.5)
    one = torch.ones(B, dtype=torch.long)
    ref = m(torch.cat([z_ctx, z_a, z_b], 1), torch.cat([tau_ctx, tau_a, tau_b], 1),
            step_size=one)[:, Ctx + 1:Ctx + 2]
    c = cache_slots(m)
    m._run_blocks(m._build_tokens(z_ctx, tau_ctx, one), caches=c, append=True)
    m._run_blocks(m._build_tokens(z_a, tau_a, one), caches=c, append=True)  # commit
    xt = m._run_blocks(m._build_tokens(z_b, tau_b, one), caches=c, append=False)
    cached = m._project_out(xt, B, 1, Cc, 16, 16)
    d = (ref - cached).abs().max().item()
    print(f"prefill+commit+decode  max|Δ| = {d:.2e}")
    assert d < TOL, "FAILED commit equivalence"


@torch.no_grad()
def test_rollout_runs():
    m = build_model()
    B, Cc, Ctx = 2, 16, 8
    ctx = torch.randn(B, Ctx, Cc, 16, 16)
    out = m.rollout(ctx, predict_frames=5, num_steps=4, k_max=64)
    assert out.shape == (B, 5, Cc, 16, 16), out.shape
    print(f"rollout                -> {tuple(out.shape)} ok")


if __name__ == "__main__":
    test_prefill_decode()
    test_commit()
    test_rollout_runs()
    print("ALL KV-CACHE TESTS PASSED")
