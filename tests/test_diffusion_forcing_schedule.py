"""Regression test for the diffusion-forcing τ-schedule position bug (fixed 2026-07-04).

Bug: ``sample_diffusion_forcing_timesteps`` tied each *target* frame's noise level τ
to its absolute position via a ramp normalized by the remaining window. A frame at
position p therefore only ever trained on a narrow, position-dependent τ band — e.g.
position 16 in a T=128 sequence never saw τ below ~0.74. But the autoregressive rollout
(``DynamicsModel.rollout``) denoises *every* generated frame from τ≈0 up to τ=1
regardless of position, so ~75% of the denoising steps for the first generated frame
fell below the minimum τ ever trained at that position — a silent train/inference gap
that the teacher-forced 1-step PSNR eval could not see. Fix: sample each target frame's
τ from the full U(tau_min, 1.0), independent of position.

This test asserts (a) context frames stay near-clean, (b) target τ covers the whole
[tau_min, 1] range at *every* position (the coverage the rollout needs), and (c) the
per-position minimum τ is low everywhere — i.e. the position tie is gone.

Run:  PYTHONPATH=src python tests/test_diffusion_forcing_schedule.py
      (or: pytest tests/test_diffusion_forcing_schedule.py)
"""
import torch
from ahriuwu.models.diffusion import DiffusionSchedule


def test_target_tau_is_position_independent():
    s = DiffusionSchedule(device="cpu")
    T, B, tau_ctx, tau_min = 128, 40000, 0.9, 0.0
    torch.manual_seed(0)
    tau = s.sample_diffusion_forcing_timesteps(
        B, T, device="cpu", tau_ctx=tau_ctx, tau_min=tau_min
    )
    assert tau.shape == (B, T)
    assert tau.min() >= tau_min - 1e-6 and tau.max() <= 1.0 + 1e-6

    # Reconstruct which frames were context vs target (same RNG draw as the fn).
    torch.manual_seed(0)
    horizon = torch.randint(1, T, (B,))
    positions = torch.arange(T).unsqueeze(0)
    is_context = positions < horizon.unsqueeze(1)

    # (a) context frames are near-clean: every context τ >= tau_ctx.
    ctx_tau = tau[is_context]
    assert ctx_tau.min() >= tau_ctx - 1e-6, f"context τ dipped to {ctx_tau.min():.3f}"

    # (b)+(c) at EVERY position the model can generate (>= earliest possible horizon=1),
    # the target-frame τ reaches deep into the noisy regime. The pre-fix schedule had
    # frac(τ<0.15)==0.0000 at pos 16; require real low-τ coverage at several positions.
    for pos in (16, 24, 32, 47, 64, 96):
        col = tau[:, pos]
        target_col = col[~is_context[:, pos]]  # only frames that were targets at this pos
        frac_low = (target_col < 0.15).float().mean().item()
        assert frac_low > 0.10, (
            f"pos {pos}: only {frac_low:.4f} of target frames saw τ<0.15 "
            f"(position tie still present)"
        )
        assert target_col.min() < 0.05, (
            f"pos {pos}: target τ never reached noise (min={target_col.min():.3f})"
        )

    print("OK: target τ covers [0,1] independent of position; context stays near-clean.")


if __name__ == "__main__":
    test_target_tau_is_position_independent()
