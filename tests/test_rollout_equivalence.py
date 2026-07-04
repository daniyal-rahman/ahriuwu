"""Equivalence tests for the autoregressive rollout machinery.

Two load-bearing claims the code makes about itself, now actually verified (the
``forward_temporal_cached`` docstring previously said "verified by the rollout
equivalence test" — this is that test):

1. KV-cache temporal attention == parallel causal temporal attention. Committing
   frames one at a time into the cache (the rollout path) must produce exactly the
   same per-frame outputs as running the whole sequence through the parallel
   causal forward at once. If it didn't, the rollout would silently drift from
   what training optimized.

2. ``euler_renoise_step`` stays on the true noising trajectory. When the model's
   clean prediction is exact, re-noising from τ to next_τ must preserve the
   underlying noise direction — i.e. land exactly on the interpolant
   next_τ·z0 + (1-next_τ)·ε. The old sampler reused the frozen initial noise and
   failed this, which is why multi-step denoising diverged.

Run:  PYTHONPATH=src python tests/test_rollout_equivalence.py
      (or: pytest tests/test_rollout_equivalence.py)
"""
import torch

from ahriuwu.models.layers import Attention
from ahriuwu.models.diffusion import euler_renoise_step


def test_kv_cache_matches_full_forward():
    torch.manual_seed(0)
    B, T, D = 2, 12, 32
    attn = Attention(
        dim=D, num_heads=4, head_dim=8, mode="temporal",
        max_seq_len=16, use_qk_norm=True, soft_cap=50.0, allow_flex=False,
    ).eval()

    x = torch.randn(B, T, D)

    # Parallel causal forward (what training uses).
    with torch.no_grad():
        parallel = attn._forward_temporal(x, independent_frames=False)

    # Incremental cache: commit one frame at a time (what rollout uses).
    cache = {"k": None, "v": None, "pos": 0}
    incr = []
    with torch.no_grad():
        for t in range(T):
            out_t = attn.forward_temporal_cached(x[:, t:t + 1], cache, append=True)
            incr.append(out_t)
    incr = torch.cat(incr, dim=1)

    max_diff = (parallel - incr).abs().max().item()
    assert max_diff < 1e-5, f"KV-cache diverged from full forward: max|Δ|={max_diff:.2e}"
    print(f"OK: KV-cache == full causal forward (max|Δ|={max_diff:.2e}).")


def test_euler_renoise_preserves_true_trajectory():
    torch.manual_seed(0)
    z0 = torch.randn(4, 32)
    eps = torch.randn(4, 32)  # the true underlying noise

    # A state on the true trajectory at level tau, and the EXACT clean prediction.
    for tau, next_tau in [(0.01, 0.2), (0.2, 0.5), (0.5, 0.9), (0.9, 0.999)]:
        z_t = tau * z0 + (1.0 - tau) * eps
        z_next = euler_renoise_step(z_t, z0, tau, next_tau)
        expected = next_tau * z0 + (1.0 - next_tau) * eps  # same eps preserved
        max_diff = (z_next - expected).abs().max().item()
        assert max_diff < 1e-5, (
            f"renoise left the true trajectory at τ={tau}->{next_tau}: {max_diff:.2e}"
        )

    # Contrast: the OLD frozen-initial-noise scheme would only match by luck.
    # Use a DIFFERENT initial noise than the true eps and show it diverges.
    frozen = torch.randn(4, 32)
    tau, next_tau = 0.2, 0.5
    z_t = tau * z0 + (1.0 - tau) * eps
    buggy = next_tau * z0 + (1.0 - next_tau) * frozen  # old behaviour
    correct = euler_renoise_step(z_t, z0, tau, next_tau)
    assert (buggy - correct).abs().max().item() > 1e-2, "expected old scheme to differ"
    print("OK: euler_renoise_step preserves the true noise trajectory (old scheme did not).")


if __name__ == "__main__":
    test_kv_cache_matches_full_forward()
    test_euler_renoise_preserves_true_trajectory()
