#!/usr/bin/env python3
"""Benchmark inference speed for live play pipeline.

Tests:
1. CNN tokenizer encode speed
2. Transformer tokenizer encode speed
3. Dynamics forward pass speed
4. Full pipeline (tokenize + dynamics + policy)

Target: < 50ms total for 20 FPS live play
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ahriuwu.models import create_tokenizer, create_dynamics, PolicyHead
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
    frame = torch.randn(1, 3, 256, 256, device=device)

    # Also test with sequence for context (e.g., last 8 frames)
    frame_seq = torch.randn(1, 8, 3, 256, 256, device=device)

    print("=" * 70)
    print("TOKENIZER COMPARISON (single frame, batch=1)")
    print("=" * 70)

    # CNN Tokenizer
    cnn_tokenizer = create_tokenizer("small").to(device).eval()
    cnn_params = sum(p.numel() for p in cnn_tokenizer.parameters())
    print(f"\nCNN Tokenizer (small): {cnn_params:,} params")

    with torch.no_grad():
        cnn_encode_time = benchmark_fn(
            lambda: cnn_tokenizer.encode(frame),
            name="  CNN encode (1 frame)"
        )

        # Test 8 frames (batched)
        frames_8 = torch.randn(8, 3, 256, 256, device=device)
        cnn_batch_time = benchmark_fn(
            lambda: cnn_tokenizer.encode(frames_8),
            name="  CNN encode (8 frames batched)"
        )

    # Get latent shape from CNN
    with torch.no_grad():
        cnn_latent = cnn_tokenizer.encode(frame)
    print(f"  Latent shape: {cnn_latent.shape}")  # (1, 256, 16, 16)

    # Transformer Tokenizer
    print()
    trans_tokenizer = create_transformer_tokenizer("small").to(device).eval()
    trans_params = sum(p.numel() for p in trans_tokenizer.parameters())
    print(f"Transformer Tokenizer (small): {trans_params:,} params")

    with torch.no_grad():
        trans_encode_time = benchmark_fn(
            lambda: trans_tokenizer.encode(frame)["latent"],
            name="  Transformer encode (1 frame)"
        )

        trans_batch_time = benchmark_fn(
            lambda: trans_tokenizer.encode(frames_8)["latent"],
            name="  Transformer encode (8 frames batched)"
        )

    with torch.no_grad():
        trans_latent = trans_tokenizer.encode(frame)["latent"]
    print(f"  Latent shape: {trans_latent.shape}")  # (1, 256, 32)

    print()
    print(f"CNN vs Transformer speedup (1 frame): {trans_encode_time / cnn_encode_time:.1f}x slower")
    print(f"CNN vs Transformer speedup (8 frames): {trans_batch_time / cnn_batch_time:.1f}x slower")

    print()
    print("=" * 70)
    print("DYNAMICS MODEL (tiny for live inference)")
    print("=" * 70)

    # Use tiny dynamics for live inference (speed over quality)
    dynamics = create_dynamics(
        size="tiny",
        latent_dim=256,
        use_agent_tokens=True,
        use_qk_norm=True,
        soft_cap=50.0,
        num_register_tokens=4,  # Reduced for speed
    ).to(device).eval()

    dyn_params = dynamics.get_num_params()
    print(f"\nDynamics (tiny): {dyn_params:,} params")

    # Prepare latent input - reshape CNN latent to match expected format
    # CNN: (B, 256, 16, 16) -> (B, T, 256, 16, 16) for dynamics
    latent_seq = cnn_latent.unsqueeze(1).expand(-1, 8, -1, -1, -1)  # (1, 8, 256, 16, 16)
    tau = torch.zeros(1, 8, device=device)  # Noise level 0 for inference

    with torch.no_grad():
        dynamics_time = benchmark_fn(
            lambda: dynamics(latent_seq, tau),
            name="  Dynamics forward (seq_len=8)"
        )

    print()
    print("=" * 70)
    print("POLICY HEAD")
    print("=" * 70)

    # Policy head
    policy = PolicyHead(
        input_dim=dynamics.model_dim,
        action_dim=128,
        hidden_dim=256,
        mtp_length=1,  # Only predict next action for live play
    ).to(device).eval()

    policy_params = sum(p.numel() for p in policy.parameters())
    print(f"\nPolicy Head: {policy_params:,} params")

    # Get agent tokens from dynamics (need to run with agent tokens)
    with torch.no_grad():
        z_pred, agent_out = dynamics(latent_seq, tau)
        # agent_out shape: (B, T, model_dim)

    print(f"  Agent output shape: {agent_out.shape}")

    with torch.no_grad():
        policy_time = benchmark_fn(
            lambda: policy(agent_out[:, -1:, :]),  # Just last timestep
            name="  Policy forward (last frame)"
        )

    print()
    print("=" * 70)
    print("FULL PIPELINE COMPARISON")
    print("=" * 70)

    # Full pipeline with CNN tokenizer
    def full_pipeline_cnn():
        # Encode single new frame
        z_new = cnn_tokenizer.encode(frame)  # (1, 256, 16, 16)
        # Add to sequence (simulated - in practice would maintain buffer)
        z_seq = z_new.unsqueeze(1).expand(-1, 8, -1, -1, -1)
        tau_zeros = torch.zeros(1, 8, device=device)
        # Run dynamics
        z_pred, agent_out = dynamics(z_seq, tau_zeros)
        # Get action
        action_logits = policy(agent_out[:, -1:, :])
        return action_logits.argmax(-1)

    print()
    with torch.no_grad():
        total_cnn = benchmark_fn(full_pipeline_cnn, name="Full pipeline (CNN tokenizer)")

    # Full pipeline with Transformer tokenizer
    # Need to handle different latent shape
    dynamics_trans = create_dynamics(
        size="tiny",
        latent_dim=32,  # Transformer small uses latent_dim=32
        use_agent_tokens=True,
        use_qk_norm=True,
        soft_cap=50.0,
        num_register_tokens=4,
    ).to(device).eval()

    policy_trans = PolicyHead(
        input_dim=dynamics_trans.model_dim,
        action_dim=128,
        hidden_dim=256,
        mtp_length=1,
    ).to(device).eval()

    def full_pipeline_transformer():
        # Encode single new frame
        z_new = trans_tokenizer.encode(frame)["latent"]  # (1, 256, 32)
        # Reshape for dynamics: (B, num_latents, latent_dim) -> (B, T, num_latents, sqrt, sqrt)
        # Actually transformer outputs (B, 256, 32), dynamics expects (B, T, C, H, W)
        # Need to reshape: (1, 256, 32) -> (1, 1, 32, 16, 16)
        z_reshaped = z_new.transpose(1, 2).view(1, 32, 16, 16).unsqueeze(1)
        z_seq = z_reshaped.expand(-1, 8, -1, -1, -1)  # (1, 8, 32, 16, 16)
        tau_zeros = torch.zeros(1, 8, device=device)
        # Run dynamics
        z_pred, agent_out = dynamics_trans(z_seq, tau_zeros)
        # Get action
        action_logits = policy_trans(agent_out[:, -1:, :])
        return action_logits.argmax(-1)

    with torch.no_grad():
        total_trans = benchmark_fn(full_pipeline_transformer, name="Full pipeline (Transformer tokenizer)")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"Target for 20 FPS: < 50ms per frame")
    print()
    print(f"CNN tokenizer pipeline:         {total_cnn:6.2f}ms  {'OK' if total_cnn < 50 else 'TOO SLOW'}")
    print(f"Transformer tokenizer pipeline: {total_trans:6.2f}ms  {'OK' if total_trans < 50 else 'TOO SLOW'}")
    print()

    if total_cnn < 50:
        print("CNN tokenizer meets real-time requirements!")
        print(f"Headroom: {50 - total_cnn:.1f}ms for screen capture + action sending")
    elif total_trans < 50:
        print("Transformer tokenizer meets real-time requirements!")
        print(f"Headroom: {50 - total_trans:.1f}ms for screen capture + action sending")
    else:
        print("WARNING: Neither tokenizer meets 20 FPS target on this device.")
        print("Consider:")
        print("  - Using 'tiny' model sizes")
        print("  - Running on GPU")
        print("  - Reducing sequence length")
        print("  - Using lower inference frequency (10 FPS)")


if __name__ == "__main__":
    main()
