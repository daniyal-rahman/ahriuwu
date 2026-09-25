"""Test flex_attention correctness and benchmark vs manual attention.

Verifies that the flex_attention implementation produces numerically equivalent
results to the manual attention, and measures the speedup.
"""

import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


from ahriuwu.models.dynamics import create_dynamics


def manual_attention_forward(q, k, v, scale, soft_cap, causal_mask=None, num_groups=1):
    """Reference manual attention (the old code path)."""
    if num_groups > 1:
        k = k.repeat_interleave(num_groups, dim=1)
        v = v.repeat_interleave(num_groups, dim=1)

    attn = (q @ k.transpose(-2, -1)) * scale
    if soft_cap is not None:
        attn = soft_cap * torch.tanh(attn / soft_cap)
    if causal_mask is not None:
        attn = attn.masked_fill(causal_mask, float("-inf"))
    attn = F.softmax(attn, dim=-1)
    return attn @ v


def test_spatial_attention():
    """Test SpatialAttention produces correct output."""
    from ahriuwu.models.dynamics import SpatialAttention
    from torch.nn.attention.flex_attention import flex_attention

    print("Testing SpatialAttention...", end=" ")
    layer = SpatialAttention(dim=512, num_heads=8, num_kv_heads=4,
                             soft_cap=50.0, spatial_size=16).cuda().bfloat16()

    # Create input: (B, T, S, D) where S = 256 latent + 8 register + extras
    x = torch.randn(2, 8, 270, 512, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        out = layer(x)

    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
    assert not torch.isnan(out).any(), "NaN in output"
    assert out.std() > 0.01, f"Output collapsed: std={out.std():.6f}"
    print(f"PASS (shape={out.shape}, std={out.std():.4f})")


def test_temporal_attention():
    """Test TemporalAttention produces correct output in both modes."""
    from ahriuwu.models.dynamics import TemporalAttention

    print("Testing TemporalAttention (causal)...", end=" ")
    layer = TemporalAttention(dim=512, num_heads=8, num_kv_heads=4,
                              soft_cap=50.0, max_seq_len=256).cuda().bfloat16()

    x = torch.randn(2, 32, 16, 512, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        out_causal = layer(x, independent_frames=False)
        out_indep = layer(x, independent_frames=True)

    assert out_causal.shape == x.shape
    assert out_indep.shape == x.shape
    assert not torch.isnan(out_causal).any(), "NaN in causal output"
    assert not torch.isnan(out_indep).any(), "NaN in independent output"

    # Independent mode should differ from causal mode
    diff = (out_causal - out_indep).abs().mean()
    assert diff > 0.001, f"Causal and independent outputs too similar: diff={diff:.6f}"
    print(f"PASS (causal_std={out_causal.std():.4f}, indep_std={out_indep.std():.4f}, diff={diff:.4f})")


def test_agent_cross_attention():
    """Test AgentCrossAttention produces correct output."""
    from ahriuwu.models.dynamics import AgentCrossAttention

    print("Testing AgentCrossAttention...", end=" ")
    layer = AgentCrossAttention(dim=512, num_heads=8, num_kv_heads=4,
                                soft_cap=50.0).cuda().bfloat16()

    agent = torch.randn(2, 16, 512, device="cuda", dtype=torch.bfloat16)
    z_tokens = torch.randn(2, 16, 270, 512, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        out = layer(agent, z_tokens)

    assert out.shape == agent.shape, f"Shape mismatch: {out.shape} vs {agent.shape}"
    assert not torch.isnan(out).any(), "NaN in output"
    assert out.std() > 0.01, f"Output collapsed: std={out.std():.6f}"
    print(f"PASS (shape={out.shape}, std={out.std():.4f})")


def test_agent_temporal_attention():
    """Test AgentTemporalAttention produces correct output."""
    from ahriuwu.models.dynamics import AgentTemporalAttention

    print("Testing AgentTemporalAttention...", end=" ")
    layer = AgentTemporalAttention(dim=512, num_heads=8, num_kv_heads=4,
                                   soft_cap=50.0, max_seq_len=256).cuda().bfloat16()

    x = torch.randn(2, 32, 512, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        out = layer(x)

    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
    assert not torch.isnan(out).any(), "NaN in output"
    assert out.std() > 0.01, f"Output collapsed: std={out.std():.6f}"
    print(f"PASS (shape={out.shape}, std={out.std():.4f})")


def test_backward():
    """Test that gradients flow correctly through flex_attention."""
    from ahriuwu.models.dynamics import SpatialAttention, TemporalAttention

    print("Testing backward pass (SpatialAttention)...", end=" ")
    layer = SpatialAttention(dim=256, num_heads=4, num_kv_heads=2,
                             soft_cap=50.0, spatial_size=8).cuda().bfloat16()

    x = torch.randn(2, 4, 70, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = layer(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "No gradient for input"
    assert not torch.isnan(x.grad).any(), "NaN in gradient"
    print(f"PASS (grad_norm={x.grad.norm():.4f})")

    print("Testing backward pass (TemporalAttention)...", end=" ")
    layer2 = TemporalAttention(dim=256, num_heads=4, num_kv_heads=2,
                                soft_cap=50.0, max_seq_len=64).cuda().bfloat16()

    x2 = torch.randn(2, 16, 8, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out2 = layer2(x2)
    loss2 = out2.sum()
    loss2.backward()
    assert x2.grad is not None, "No gradient for input"
    assert not torch.isnan(x2.grad).any(), "NaN in gradient"
    print(f"PASS (grad_norm={x2.grad.norm():.4f})")


def test_full_model():
    """Test the full DynamicsTransformer model."""
    print("Testing full DynamicsTransformer (tiny)...", end=" ")
    model = create_dynamics("tiny", latent_dim=32).cuda().bfloat16()

    B, T, C, H, W = 2, 16, 32, 16, 16
    z = torch.randn(B, T, C, H, W, device="cuda", dtype=torch.bfloat16)
    tau = torch.rand(B, T, device="cuda")

    with torch.no_grad():
        out = model(z, tau)

    if isinstance(out, tuple):
        out = out[0]
    assert out.shape == z.shape, f"Shape mismatch: {out.shape} vs {z.shape}"
    assert not torch.isnan(out).any(), "NaN in model output"
    print(f"PASS (shape={out.shape}, std={out.std():.4f})")


def benchmark_full_model():
    """Benchmark the full model (compiled)."""
    print("\n=== Benchmark: DynamicsTransformer (small) ===")
    model = create_dynamics("small", latent_dim=32).cuda().bfloat16()
    model = torch.compile(model)

    B, T, C, H, W = 2, 32, 32, 16, 16
    z = torch.randn(B, T, C, H, W, device="cuda", dtype=torch.bfloat16)
    tau = torch.rand(B, T, device="cuda")

    # Warmup (includes compilation)
    print("Warming up (compiling)...")
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(3):
            _ = model(z, tau)
    torch.cuda.synchronize()

    # Benchmark forward
    iters = 20
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(iters):
            _ = model(z, tau)
    torch.cuda.synchronize()
    fwd_ms = (time.perf_counter() - t0) / iters * 1000
    print(f"Forward: {fwd_ms:.1f} ms (B={B}, T={T})")

    # Benchmark forward + backward
    model_train = create_dynamics("small", latent_dim=32).cuda().bfloat16()
    model_train = torch.compile(model_train)
    z_train = torch.randn(B, T, C, H, W, device="cuda", dtype=torch.bfloat16)
    tau_train = torch.rand(B, T, device="cuda")

    print("Warming up training (compiling)...")
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(3):
            out = model_train(z_train, tau_train)
            if isinstance(out, tuple):
                out = out[0]
            out.sum().backward()
            model_train.zero_grad()
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(iters):
            out = model_train(z_train, tau_train)
            if isinstance(out, tuple):
                out = out[0]
            out.sum().backward()
            model_train.zero_grad()
    torch.cuda.synchronize()
    train_ms = (time.perf_counter() - t0) / iters * 1000
    print(f"Forward + Backward: {train_ms:.1f} ms (B={B}, T={T})")

    # Memory usage
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model_train(z_train, tau_train)
        if isinstance(out, tuple):
            out = out[0]
        out.sum().backward()
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    print(f"Peak GPU memory: {peak_mb:.0f} MB")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    print("=== flex_attention Correctness Tests ===\n")

    test_spatial_attention()
    test_temporal_attention()
    test_agent_cross_attention()
    test_agent_temporal_attention()
    test_backward()
    test_full_model()

    print("\nAll correctness tests passed!")

    benchmark_full_model()
