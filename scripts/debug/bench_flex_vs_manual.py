"""A/B benchmark: flex_attention vs manual attention.

Creates two models - one with flex_attention (current), one with manual attention (baseline) -
and measures forward+backward throughput.
"""

import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/mnt/storage/ahriuwu/repo/src")

from ahriuwu.models.layers import soft_cap_attention, apply_rotary_emb


class ManualSpatialAttention(nn.Module):
    """Old manual attention for comparison."""
    def __init__(self, dim, num_heads=8, num_kv_heads=None, head_dim=None,
                 soft_cap=50.0, spatial_size=16):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.soft_cap = soft_cap
        self.num_groups = num_heads // self.num_kv_heads
        self.spatial_tokens = spatial_size * spatial_size

        from ahriuwu.models.layers import QKNorm, RotaryEmbedding2D
        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(dim, q_dim, bias=False)
        self.k_proj = nn.Linear(dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(dim, kv_dim, bias=False)
        self.out_proj = nn.Linear(q_dim, dim, bias=False)
        self.qk_norm = QKNorm(self.head_dim)
        self.rope_2d = RotaryEmbedding2D(self.head_dim, spatial_size)
        self.dropout = nn.Dropout(0.0)

    def forward(self, x):
        B, T, S, D = x.shape
        x = x.view(B * T, S, D)
        q = self.q_proj(x).view(B * T, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B * T, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B * T, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = self.qk_norm(q, k)
        Nz = self.spatial_tokens
        if Nz > 0 and Nz <= S:
            cos, sin = self.rope_2d.get_rotary_emb(Nz, x.device)
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
            q_l = apply_rotary_emb(q[:, :, :Nz, :], cos, sin)
            k_l = apply_rotary_emb(k[:, :, :Nz, :], cos, sin)
            q = torch.cat([q_l, q[:, :, Nz:, :]], dim=2)
            k = torch.cat([k_l, k[:, :, Nz:, :]], dim=2)
        if self.num_groups > 1:
            k = k.repeat_interleave(self.num_groups, dim=1)
            v = v.repeat_interleave(self.num_groups, dim=1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.soft_cap is not None:
            attn = soft_cap_attention(attn, self.soft_cap)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).to(v.dtype)  # cast back from float32 (RoPE upcast)
        out = out.transpose(1, 2).reshape(B * T, S, -1)
        out = self.out_proj(out)
        return out.view(B, T, S, D)


class ManualTemporalAttention(nn.Module):
    """Old manual temporal attention for comparison."""
    def __init__(self, dim, num_heads=8, num_kv_heads=None, head_dim=None,
                 soft_cap=50.0, max_seq_len=256):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.soft_cap = soft_cap
        self.num_groups = num_heads // self.num_kv_heads

        from ahriuwu.models.layers import QKNorm, RotaryEmbedding1D
        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(dim, q_dim, bias=False)
        self.k_proj = nn.Linear(dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(dim, kv_dim, bias=False)
        self.out_proj = nn.Linear(q_dim, dim, bias=False)
        self.qk_norm = QKNorm(self.head_dim)
        self.rope_1d = RotaryEmbedding1D(self.head_dim, max_seq_len)
        self.register_buffer("causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool())

    def forward(self, x):
        B, T, S, D = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * S, T, D)
        q = self.q_proj(x).view(B * S, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B * S, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B * S, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = self.qk_norm(q, k)
        cos, sin = self.rope_1d.get_rotary_emb(T, x.device)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        if self.num_groups > 1:
            k = k.repeat_interleave(self.num_groups, dim=1)
            v = v.repeat_interleave(self.num_groups, dim=1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.soft_cap is not None:
            attn = soft_cap_attention(attn, self.soft_cap)
        mask = self.causal_mask[:T, :T]
        attn = attn.masked_fill(mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).to(v.dtype)  # cast back from float32 (RoPE upcast)
        out = out.transpose(1, 2).reshape(B * S, T, -1)
        out = self.out_proj(out)
        return out.view(B, S, T, D).permute(0, 2, 1, 3)


def bench(name, fn, warmup=5, iters=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / iters * 1000
    print(f"  {name}: {elapsed:.2f} ms")
    return elapsed


def benchmark_layer(name, ManualClass, FlexClass, init_kwargs, input_shape, compile_model=True):
    print(f"\n=== {name} (input: {input_shape}) ===")

    manual = ManualClass(**init_kwargs).cuda().bfloat16()
    flex = FlexClass(**init_kwargs).cuda().bfloat16()
    # Copy weights so outputs are comparable
    flex.load_state_dict(manual.state_dict(), strict=False)

    if compile_model:
        manual_c = torch.compile(manual)
        flex_c = torch.compile(flex)
    else:
        manual_c, flex_c = manual, flex

    x = torch.randn(*input_shape, device="cuda", dtype=torch.bfloat16)

    # Forward only
    manual_fwd = bench("Manual forward", lambda: manual_c(x))
    flex_fwd = bench("Flex forward", lambda: flex_c(x))
    print(f"  Forward speedup: {manual_fwd / flex_fwd:.2f}x")

    # Forward + backward
    def train_step(model):
        out = model(x)
        out.sum().backward()
        model.zero_grad()

    manual_train = bench("Manual fwd+bwd", lambda: train_step(manual_c))
    flex_train = bench("Flex fwd+bwd", lambda: train_step(flex_c))
    print(f"  Train speedup: {manual_train / flex_train:.2f}x")

    # Memory comparison
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    out = manual_c(x)
    out.sum().backward()
    manual_mem = torch.cuda.max_memory_allocated() / 1024**2
    manual.zero_grad()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    out = flex_c(x)
    out.sum().backward()
    flex_mem = torch.cuda.max_memory_allocated() / 1024**2

    print(f"  Memory: Manual={manual_mem:.0f} MB, Flex={flex_mem:.0f} MB ({manual_mem/flex_mem:.2f}x)")


def benchmark_full_model():
    """Compare full model performance."""
    from ahriuwu.models.dynamics import create_dynamics

    print("\n=== Full DynamicsTransformer (small, compiled) ===")

    model = create_dynamics("small", latent_dim=32).cuda().bfloat16()
    model_c = torch.compile(model)

    B, T, C, H, W = 2, 32, 32, 16, 16
    z = torch.randn(B, T, C, H, W, device="cuda", dtype=torch.bfloat16)
    tau = torch.rand(B, T, device="cuda")

    # Warmup
    print("Compiling...")
    for _ in range(3):
        out = model_c(z, tau)
        if isinstance(out, tuple):
            out = out[0]
        out.sum().backward()
        model.zero_grad()
    torch.cuda.synchronize()

    # Benchmark
    def train_step():
        out = model_c(z, tau)
        if isinstance(out, tuple):
            out = out[0]
        out.sum().backward()
        model.zero_grad()

    elapsed = bench("Flex model fwd+bwd", train_step, warmup=3, iters=20)

    # Memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    train_step()
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    print(f"  Peak memory: {peak_mb:.0f} MB")
    print(f"  Throughput: {2 * 32 / (elapsed / 1000):.1f} frames/sec")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    from ahriuwu.models.dynamics import SpatialAttention, TemporalAttention

    # Spatial attention: (B=2, T=32, S=270, D=512)
    benchmark_layer(
        "SpatialAttention (GQA 8/4)",
        ManualSpatialAttention, SpatialAttention,
        dict(dim=512, num_heads=8, num_kv_heads=4, soft_cap=50.0, spatial_size=16),
        (2, 32, 270, 512),
    )

    # Temporal attention: (B=2, T=32, S=270, D=512)
    benchmark_layer(
        "TemporalAttention (GQA 8/4, causal)",
        ManualTemporalAttention, TemporalAttention,
        dict(dim=512, num_heads=8, num_kv_heads=4, soft_cap=50.0, max_seq_len=256),
        (2, 32, 270, 512),
    )

    # Full model
    benchmark_full_model()
