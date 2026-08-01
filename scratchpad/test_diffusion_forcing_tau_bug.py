"""PROOF: sample_diffusion_forcing_timesteps couples tau to absolute frame
position, creating a train/inference distribution gap that corrupts the
autoregressive rollout (the deploy regime) while the teacher-forced 1-step
eval stays blind to it.

Run: PYTHONPATH=src /home/dani/miniconda3/envs/ml/bin/python scratchpad/test_diffusion_forcing_tau_bug.py
"""
import torch
from ahriuwu.models.diffusion import DiffusionSchedule


def reconstruct_horizon(seed, B, T):
    torch.manual_seed(seed)
    return torch.randint(1, T, (B,))


def main():
    s = DiffusionSchedule(device="cpu")
    T = 128  # job 124 short config (--seq-len-short 128)
    B = 50000

    torch.manual_seed(0)
    tau = s.sample_diffusion_forcing_timesteps(B, T, device="cpu")

    # ---- CLAIM 1: the horizon frame (first frame to predict) is ALWAYS ~0.9 ----
    horizon = reconstruct_horizon(0, B, T)  # same RNG draw as inside the fn
    first_tgt = tau[torch.arange(B), horizon]
    print(f"[1] First target frame tau: mean={first_tgt.mean():.4f} "
          f"std={first_tgt.std():.4f} min={first_tgt.min():.4f} max={first_tgt.max():.4f}")
    assert first_tgt.mean() > 0.85, "first target frame is NOT near-clean?!"
    print("    -> the frame the model starts PREDICTING is trained ~90% clean.\n")

    # ---- CLAIM 2: tau is (near) deterministic in absolute position ----
    print("[2] Per-position tau min over 50k samples (rollout generates pos>=16):")
    for pos in [16, 24, 32, 47]:
        tp = tau[:, pos]
        print(f"    pos {pos:3d}: tau in [{tp.min():.3f}, {tp.max():.3f}], "
              f"frac(tau<0.15)={(tp < 0.15).float().mean():.4f}")
    pos16_min = tau[:, 16].min().item()
    assert pos16_min > 0.5, "pos16 unexpectedly saw low tau"
    print(f"    -> position 16 NEVER trained below tau={pos16_min:.2f}.\n")

    # ---- CLAIM 3: rollout queries those positions at tau=eps (pure noise) ----
    ctx, num_steps, eps = 16, 16, 1e-3
    step = (1.0 - eps) / num_steps
    infer_taus = [eps + i * step for i in range(num_steps)]
    oos = sum(1 for t in infer_taus if t < pos16_min)
    print(f"[3] Rollout denoises the pos-16 frame across tau={[round(t,2) for t in infer_taus]}")
    print(f"    {oos}/{num_steps} = {100*oos//num_steps}% of those steps are BELOW the "
          f"min tau (={pos16_min:.2f}) training ever showed at pos 16.")
    assert oos >= num_steps // 2, "expected majority of inference steps off-distribution"
    print("    -> majority of the actual inference denoising trajectory is off-distribution.\n")

    # ---- CLAIM 4: a position-decoupled schedule would close the gap ----
    # A correct diffusion-forcing target schedule samples the target frames'
    # tau ~ U(tau_min, 1) INDEPENDENT of position (each generated frame denoises
    # from noise regardless of where it sits). Show that closes the coverage gap.
    torch.manual_seed(0)
    fixed_target_tau = torch.rand(B)  # U(0,1), position-independent
    print(f"[4] Position-decoupled target tau ~ U(0,1): "
          f"frac(<0.15)={(fixed_target_tau < 0.15).float().mean():.3f} "
          f"(vs 0.000 at pos 16 today). Gap closes.\n")

    print("VERDICT: sample_diffusion_forcing_timesteps (diffusion.py:89-144) ties the "
          "denoising level to absolute frame position. Rollout (dynamics.py:873-896) "
          "denoises every generated frame from tau=eps regardless of position, so the "
          "(position, tau) pairs it needs were never trained. Teacher-forced "
          "eval_denoising_psnr uses uniform per-frame tau on the REAL corrupted latent, "
          "so it does NOT expose this. CORRUPTS ROLLOUT.")


if __name__ == "__main__":
    main()
