"""Regression test for the diffusion-forcing τ-schedule (paper-faithful, 2026-07-04).

Original bug: ``sample_diffusion_forcing_timesteps`` tied each frame's noise level τ
to a sampled horizon and to its ABSOLUTE position (a ramp normalized by the remaining
window). A frame at position p only ever trained on a narrow, position-dependent τ band
— e.g. position 16 in a T=128 sequence never saw τ below ~0.74. But the autoregressive
rollout denoises *every* generated frame from τ≈0 up to τ=1 regardless of position, so
the (position, low-τ) pairs it queries were never trained — a silent train/inference gap
the teacher-forced eval could not see.

Fix = the paper's law (Diffusion Forcing / DreamerV4 Sec 2): sample each frame's τ
i.i.d. from U(tau_min, 1), with NO horizon/context ramp. This test asserts the schedule
is now position-independent and uniform: every position sees the full [tau_min, 1] range
with ~uniform density, so the rollout's (position, τ) queries are all covered.

Run:  PYTHONPATH=src python tests/test_diffusion_forcing_schedule.py
      (or: pytest tests/test_diffusion_forcing_schedule.py)
"""
import torch
from ahriuwu.models.diffusion import DiffusionSchedule


def test_tau_is_iid_uniform_per_frame():
    s = DiffusionSchedule(device="cpu")
    T, B, tau_min = 128, 40000, 0.0
    torch.manual_seed(0)
    tau = s.sample_diffusion_forcing_timesteps(B, T, device="cpu", tau_min=tau_min)

    assert tau.shape == (B, T)
    assert tau.min() >= tau_min - 1e-6 and tau.max() <= 1.0 + 1e-6

    # Uniform per column: mean ~0.5, and full low-τ coverage at EVERY position
    # (the pre-fix schedule had frac(τ<0.15)==0.0000 at pos 16).
    for pos in (0, 16, 24, 32, 47, 64, 96, 127):
        col = tau[:, pos]
        frac_low = (col < 0.15).float().mean().item()   # U(0,1) -> ~0.15
        assert 0.12 < frac_low < 0.18, f"pos {pos}: frac(τ<0.15)={frac_low:.3f} not ~uniform"
        assert abs(col.mean().item() - 0.5) < 0.02, f"pos {pos}: mean {col.mean():.3f} != ~0.5"

    # Position-independence: per-position means are all statistically ~equal.
    col_means = tau.mean(dim=0)
    assert (col_means.max() - col_means.min()).item() < 0.03, "τ distribution varies with position"

    # Independence across frames within a sequence: near-zero lag-1 correlation.
    a = tau[:, :-1].reshape(-1)
    b = tau[:, 1:].reshape(-1)
    corr = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    assert abs(corr) < 0.02, f"adjacent-frame τ correlated ({corr:.3f}) — not i.i.d."

    print("OK: τ is i.i.d. U(0,1) per frame — full low-τ coverage at every position.")


if __name__ == "__main__":
    test_tau_is_iid_uniform_per_frame()
