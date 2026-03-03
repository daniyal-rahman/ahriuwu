#!/usr/bin/env python3
"""Benchmark inference speed for live play pipeline.

Tests:
1. Transformer tokenizer encode speed
2. Dynamics forward pass speed
3. Full pipeline (tokenize + dynamics + policy)

Target: < 50ms total for 20 FPS live play
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ahriuwu.models import create_dynamics, PolicyHead
from ahriuwu.models.transformer_tokenizer import create_transformer_tokenizer


def benchmark_fn(fn, warmup: int = 5, runs: int = 50, name: str = ""):
    """Benchmark a function with warmup."""
    # Warmup
    for _ in range(warmup):
        fn()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()

    # Benchmark
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif torch.backends.mps.is_available():
            torch.mps.synchronize()
        times.append((time.perf_counter() - start) * 1000)  # ms

    avg = sum(times) / len(times)
    min_t = min(times)
    max_t = max(times)

    print(f"{name:40s} avg: {avg:6.2f}ms  min: {min_t:6.2f}ms  max: {max_t:6.2f}ms")
    return avg


def main():
    # Device selection
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    if device == "cuda":
        print(f"CUDA: {torch.cuda.get_device_name()}")
    print()

    # Test input: single frame (batch=1) for live inference
    frame = torch.randn(1, 3, 352, 352, device=device)
    frames_8 = torch.randn(8, 3, 352, 352, device=device)

    print("=" * 70)
    print("TRANSFORMER TOKENIZER (single frame, batch=1)")
    print("=" * 70)

    tokenizer = create_transformer_tokenizer("small").to(device).eval()
    params = sum(p.numel() for p in tokenizer.parameters())
    print(f"\nTransformer Tokenizer (small): {params:,} params")

    with torch.no_grad():
        encode_time = benchmark_fn(
            lambda: tokenizer.encode(frame)["latent"],
            name="  Encode (1 frame)"
        )

        batch_time = benchmark_fn(
            lambda: tokenizer.encode(frames_8)["latent"],
            name="  Encode (8 frames batched)"
        )

    with torch.no_grad():
        latent = tokenizer.encode(frame)["latent"]
    print(f"  Latent shape: {latent.shape}")  # (1, 256, 32)

    print()
    print("=" * 70)
    print("DYNAMICS MODEL (tiny for live inference)")
    print("=" * 70)

    # Use tiny dynamics for live inference (speed over quality)
    dynamics = create_dynamics(
        size="tiny",
        latent_dim=32,  # Matches transformer tokenizer
        use_agent_tokens=True,
        use_qk_norm=True,
        soft_cap=50.0,
        num_register_tokens=4,
    ).to(device).eval()

    dyn_params = dynamics.get_num_params()
    print(f"\nDynamics (tiny): {dyn_params:,} params")

    # Reshape latent for dynamics: (1, 256, 32) -> (1, 32, 16, 16) -> (1, 8, 32, 16, 16)
    with torch.no_grad():
        z = latent.transpose(1, 2).view(1, 32, 16, 16)
    latent_seq = z.unsqueeze(1).expand(-1, 8, -1, -1, -1)  # (1, 8, 32, 16, 16)
    tau = torch.zeros(1, 8, device=device)

    with torch.no_grad():
        dynamics_time = benchmark_fn(
            lambda: dynamics(latent_seq, tau),
            name="  Dynamics forward (seq_len=8)"
        )

    print()
    print("=" * 70)
    print("POLICY HEAD")
    print("=" * 70)

    policy = PolicyHead(
        input_dim=dynamics.model_dim,
        action_dim=128,
        hidden_dim=256,
        mtp_length=1,
    ).to(device).eval()

    policy_params = sum(p.numel() for p in policy.parameters())
    print(f"\nPolicy Head: {policy_params:,} params")

    with torch.no_grad():
        z_pred, agent_out = dynamics(latent_seq, tau)
    print(f"  Agent output shape: {agent_out.shape}")

    with torch.no_grad():
        policy_time = benchmark_fn(
            lambda: policy(agent_out[:, -1:, :]),
            name="  Policy forward (last frame)"
        )

    print()
    print("=" * 70)
    print("FULL PIPELINE")
    print("=" * 70)

    def full_pipeline():
        z_new = tokenizer.encode(frame)["latent"]  # (1, 256, 32)
        z_reshaped = z_new.transpose(1, 2).view(1, 32, 16, 16).unsqueeze(1)
        z_seq = z_reshaped.expand(-1, 8, -1, -1, -1)
        tau_zeros = torch.zeros(1, 8, device=device)
        z_pred, agent_out = dynamics(z_seq, tau_zeros)
        action_logits, _ = policy(agent_out[:, -1:, :])
        return action_logits.argmax(-1)

    print()
    with torch.no_grad():
        total = benchmark_fn(full_pipeline, name="Full pipeline (Transformer)")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"Target for 20 FPS: < 50ms per frame")
    print()
    print(f"Transformer pipeline: {total:6.2f}ms  {'OK' if total < 50 else 'TOO SLOW'}")
    print()

    if total < 50:
        print("Meets real-time requirements!")
        print(f"Headroom: {50 - total:.1f}ms for screen capture + action sending")
    else:
        print("WARNING: Does not meet 20 FPS target on this device.")
        print("Consider:")
        print("  - Using 'tiny' model sizes")
        print("  - Running on GPU")
        print("  - Reducing sequence length")
        print("  - Using lower inference frequency (10 FPS)")


if __name__ == "__main__":
    main()
