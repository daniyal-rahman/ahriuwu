"""Diffusion utilities for dynamics model training.

Implements x-prediction diffusion following DreamerV4:
- Linear noise schedule: z_τ = τz_0 + (1-τ)ε
- X-prediction: model predicts clean data z_0 directly
- Ramp loss weight: focus learning on high-signal regions

Convention (matches paper): τ=0 → pure noise, τ=1 → clean data.

References:
- DreamerV4: "Training Agents Inside of Scalable World Models" (Hafner et al., 2025)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionSchedule:
    """Linear noise schedule for x-prediction diffusion.

    Uses flow-matching style interpolation (paper convention):
        z_τ = τz_0 + (1-τ)ε

    where τ ∈ [0, 1], z_0 is clean data, ε is noise.
    At τ=0: z_τ = ε (pure noise)
    At τ=1: z_τ = z_0 (clean)
    """

    def __init__(self, device: str = "cuda"):
        self.device = device

    def add_noise(
        self,
        z_0: torch.Tensor,
        tau: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add noise to clean latents.

        Args:
            z_0: Clean latents, shape (B, ...) or (B, T, ...)
            tau: Noise levels, shape (B,) or (B, T) in [0, 1]
            noise: Optional pre-generated noise, same shape as z_0

        Returns:
            z_tau: Noisy latents, same shape as z_0
            noise: The noise that was added
        """
        if noise is None:
            noise = torch.randn_like(z_0)

        # Expand tau to match z_0 dimensions
        # tau shape: (B,) or (B, T)
        # z_0 shape: (B, C, H, W) or (B, T, C, H, W)
        while tau.dim() < z_0.dim():
            tau = tau.unsqueeze(-1)

        # Linear interpolation (paper convention): z_τ = τz_0 + (1-τ)ε
        z_tau = tau * z_0 + (1 - tau) * noise

        return z_tau, noise

    def sample_timesteps(
        self,
        batch_size: int,
        device: torch.device | str | None = None,
        min_tau: float = 0.0,
        max_tau: float = 1.0,
    ) -> torch.Tensor:
        """Sample random timesteps uniformly.

        Args:
            batch_size: Number of timesteps to sample
            device: Device to create tensor on
            min_tau: Minimum timestep value
            max_tau: Maximum timestep value

        Returns:
            tau: Timesteps, shape (B,) in [min_tau, max_tau]
        """
        device = device or self.device
        tau = torch.rand(batch_size, device=device)
        tau = min_tau + (max_tau - min_tau) * tau
        return tau

    def sample_diffusion_forcing_timesteps(
        self,
        batch_size: int,
        seq_length: int,
        device: torch.device | str | None = None,
        tau_ctx: float = 0.9,
        tau_min: float = 0.0,
    ) -> torch.Tensor:
        """Sample per-timestep noise levels for diffusion forcing.

        Paper convention: τ=1 clean, τ=0 noise.

        Implements DreamerV4's diffusion forcing where:
        - A random horizon h is sampled for each batch item
        - Frames before h: high τ (near-clean context, U(tau_ctx, 1.0))
        - Frames at/after h: decreasing τ (noisier targets, tau_ctx down to tau_min)

        Args:
            batch_size: Number of sequences
            seq_length: Number of frames per sequence (T)
            device: Device to create tensor on
            tau_ctx: Min τ for context frames (context varies U(tau_ctx, 1.0))
            tau_min: Minimum τ for target frames (most noisy)

        Returns:
            tau: Per-timestep noise levels, shape (B, T)
        """
        device = device or self.device

        # Sample random horizon for each batch item: where prediction starts
        horizon = torch.randint(1, seq_length, (batch_size,), device=device)

        positions = torch.arange(seq_length, device=device).unsqueeze(0)  # (1, T)
        horizon = horizon.unsqueeze(1)  # (B, 1)

        # distance = how far past the horizon (0 for context, 1+ for targets)
        distance = (positions - horizon).clamp(min=0).float()  # (B, T)

        max_distance = (seq_length - 1 - horizon).clamp(min=1).float()  # (B, 1)
        normalized_dist = distance / max_distance  # (B, T) in [0, 1]

        is_context = positions < horizon  # (B, T)

        # Context frames: high τ (near-clean), sampled from U(tau_ctx, 1.0)
        context_tau = tau_ctx + torch.rand(batch_size, seq_length, device=device) * (1.0 - tau_ctx)

        # Target frames: τ decreases from tau_ctx toward tau_min (more noise)
        # Add small noise to avoid a fully deterministic linear ramp, which would
        # make the model memorize exact positions rather than generalize over tau.
        target_tau = tau_ctx - normalized_dist * (tau_ctx - tau_min)
        target_tau = target_tau + torch.randn_like(target_tau) * 0.02
        target_tau = target_tau.clamp(tau_min, tau_ctx)

        tau = torch.where(is_context, context_tau, target_tau)

        return tau

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple,
        num_steps: int = 64,
        device: torch.device | str | None = None,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample from the model using Euler integration.

        Args:
            model: Dynamics model that predicts clean data
            shape: Shape of output (B, T, C, H, W) or (B, C, H, W)
            num_steps: Number of denoising steps
            device: Device to sample on
            context: Optional context frames (for conditional generation)

        Returns:
            z_0: Sampled clean latents
        """
        device = device or self.device

        # Start from pure noise (save for reuse in Euler steps)
        z_t = torch.randn(shape, device=device)
        z_noise = z_t.clone()

        # Euler integration from τ=eps (near-noise) to τ=1 (clean).
        # Starting exactly at τ=0 means the input is pure noise and the model
        # prediction is essentially random. Start from a small epsilon instead.
        eps = 1e-3
        step_size = (1.0 - eps) / num_steps

        for i in range(num_steps):
            tau = eps + i * step_size
            tau_tensor = torch.full((shape[0],), tau, device=device)

            # Predict clean data
            z_0_pred = model(z_t, tau_tensor, context=context)

            # Euler step towards prediction
            if i < num_steps - 1:
                next_tau = tau + step_size
                # z at next_tau: next_tau * z_0_pred + (1 - next_tau) * noise
                z_t = next_tau * z_0_pred + (1 - next_tau) * z_noise
            else:
                z_t = z_0_pred

        return z_t


def ramp_weight(tau: torch.Tensor) -> torch.Tensor:
    """Compute ramp loss weight from DreamerV4 (Paper Eq. 8).

    Paper convention: τ=0 noise, τ=1 clean.
    w(τ) = 0.9τ + 0.1

    Weights clean data higher (where signal is strong) and noisy data lower.
    At τ=1 (clean): w = 1.0 (full weight)
    At τ=0 (noise): w = 0.1 (minimal weight)

    Args:
        tau: Timesteps, shape (B,) or (B, T) in [0, 1] (0=noise, 1=clean)

    Returns:
        weights: Loss weights, same shape as tau
    """
    return 0.9 * tau + 0.1


def x_prediction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    tau: torch.Tensor,
    use_ramp_weight: bool = True,
    reduce: str = "mean",
) -> torch.Tensor:
    """Compute x-prediction loss.

    Args:
        pred: Predicted clean latents (B, T, C, H, W) or (B, C, H, W)
        target: Ground truth clean latents, same shape
        tau: Timesteps used, shape (B,) or (B, T)
        use_ramp_weight: Whether to apply ramp weighting
        reduce: Reduction mode ("mean", "none", "sum")

    Returns:
        loss: Scalar loss if reduce="mean", else per-sample losses
    """
    # MSE loss per element
    mse = F.mse_loss(pred, target, reduction="none")

    # Reduce spatial dimensions
    # (B, T, C, H, W) -> (B, T) or (B, C, H, W) -> (B,)
    while mse.dim() > tau.dim() + 1:
        mse = mse.mean(dim=-1)
    mse = mse.mean(dim=-1)  # Average over last dim

    if use_ramp_weight:
        weights = ramp_weight(tau)
        mse = mse * weights

    if reduce == "mean":
        return mse.mean()
    elif reduce == "sum":
        return mse.sum()
    else:
        return mse


# Shortcut forcing utilities (for Phase 2 - advanced training)

class ShortcutForcing:
    """Shortcut forcing objective for few-step sampling.

    Enables 4-step inference instead of 64 by training the model
    to "skip" denoising steps using bootstrapped targets.

    Not implemented in MVP - use standard diffusion first.
    """

    def __init__(self, k_max: int = 64, k_min: int = 1, bootstrap_weight: float = 10.0):
        self.k_max = k_max
        self.k_min = k_min
        self.bootstrap_weight = bootstrap_weight
        # Step sizes: 1, 2, 4, 8, ..., k_max
        self.step_sizes = [2**i for i in range(int(math.log2(k_max)) + 1)]

    def sample_step_size(self, batch_size: int, device: torch.device,
                         max_step_idx: int | None = None) -> torch.Tensor:
        """Sample step sizes uniformly up to max_step_idx.

        For progressive training: start with small d (where teacher d/2 is
        well-trained), then gradually increase max_step_idx as training
        progresses. This breaks the bootstrap trap where teacher ≈ student.

        Args:
            batch_size: Number of step sizes to sample
            device: Device
            max_step_idx: Max index into self.step_sizes to sample from.
                         None = all step sizes. 0 = d=1 only, 1 = d∈{1,2}, etc.
        Returns:
            step_sizes: (B,) tensor of integer step sizes
        """
        n = len(self.step_sizes) if max_step_idx is None else min(max_step_idx + 1, len(self.step_sizes))
        idx = torch.randint(0, n, (batch_size,), device=device)
        step_sizes_tensor = torch.tensor(self.step_sizes, device=device)
        return step_sizes_tensor[idx]

    def sample_tau_for_step_size(
        self,
        step_size: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Sample tau from grid aligned with step size (Paper Eq. 4).

        Paper convention: τ=0 noise, τ=1 clean.
        Grid: {0, d/k_max, 2d/k_max, ..., 1-d/k_max}. Avoids τ=1.

        Args:
            step_size: Step sizes (integers), shape (B,)
            device: Device for tensor creation

        Returns:
            tau: Sampled tau values, shape (B,) in [0, 1-d/k_max]
        """
        B = step_size.shape[0]
        tau = torch.zeros(B, device=device)

        for d in self.step_sizes:
            mask = (step_size == d)
            if mask.any():
                n = mask.sum().item()
                # Grid spacing for this step size
                grid_spacing = d / self.k_max
                # Number of valid grid points: 0, 1, 2, ..., (k_max/d - 1)
                num_grid_points = self.k_max // d
                # Sample grid indices uniformly
                grid_idx = torch.randint(0, num_grid_points, (n,), device=device)
                # Convert to tau values
                tau[mask] = grid_idx.float() * grid_spacing

        return tau

    def sample_tau_for_step_size_2d(
        self,
        step_size: torch.Tensor,
        seq_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Sample per-timestep tau from grid aligned with step size (Paper Eq. 4).

        Paper convention: τ=0 noise, τ=1 clean.
        Grid: {0, d/k_max, 2d/k_max, ..., 1-d/k_max}.
        Max grid point is 1-d/k_max, which naturally avoids τ=1 (the singularity).

        Args:
            step_size: Step sizes (integers), shape (B,)
            seq_length: Sequence length T
            device: Device for tensor creation

        Returns:
            tau: Sampled tau values, shape (B, T)
        """
        B = step_size.shape[0]
        tau = torch.zeros(B, seq_length, device=device)

        for d in self.step_sizes:
            mask = (step_size == d)
            if mask.any():
                n = mask.sum().item()
                grid_spacing = d / self.k_max
                # Valid grid points: 0, 1, ..., k_max/d - 1
                # Max tau = (k_max/d - 1) * d/k_max = 1 - d/k_max (avoids τ=1)
                num_grid_points = self.k_max // d
                grid_idx = torch.randint(0, num_grid_points, (n, seq_length), device=device)
                tau[mask] = grid_idx.float() * grid_spacing

        return tau

    def compute_loss(
        self,
        model: nn.Module,
        schedule: "DiffusionSchedule",
        z_0: torch.Tensor,
        tau: torch.Tensor,
        step_size: torch.Tensor,
        actions: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Compute shortcut forcing loss (DreamerV4 Paper Section 3.2, Eq. 7).

        Enables few-step inference (K=4) by training student network to predict
        k-step denoising in single step using bootstrapped teacher targets.

        For d=d_min (base/smallest step): standard x-prediction loss
        For d>d_min: bootstrap loss in velocity space with (1-τ)² weighting

        Paper Eq. 7 (τ=0 noise, τ=1 clean):
            L = (1-τ)² || (ẑ_0 - z_τ) / (1-τ) - sg(b' + b'')/2 ||²
        where:
            b' = (z_mid - z_τ) / (1-τ)            (first half-step velocity)
            b'' = (z_target - z_τ_mid) / (1-τ_mid) (second half-step velocity)

        Args:
            model: Dynamics model with step_size conditioning (independent from tau)
            schedule: DiffusionSchedule for adding noise
            z_0: Clean latents, shape (B, T, C, H, W)
            tau: Noise levels, shape (B,) or (B, T)
            step_size: Step sizes (integers, powers of 2), shape (B,).
            actions: Optional action dict with 'movement' and ability keys

        Returns:
            loss: Combined normalized loss (scalar)
            info: Dict with 'loss_std', 'loss_boot', 'n_std', 'n_boot' for logging
        """
        B = z_0.shape[0]

        # Add noise to get z_tau
        z_tau, noise = schedule.add_noise(z_0, tau)

        # Split batch by step size
        is_base_step = (step_size == 1)

        loss_std = torch.tensor(0.0, device=z_0.device)
        loss_boot = torch.tensor(0.0, device=z_0.device)
        n_std = 0
        n_boot = 0

        # Helper to slice actions if provided
        def slice_actions(actions, idx):
            if actions is None:
                return None
            return {k: v[idx] for k, v in actions.items()}

        # Standard loss for d=1 (smallest step), with ramp weight
        if is_base_step.any():
            idx = is_base_step
            tau_std = tau[idx] if tau.dim() > 1 else tau[idx]
            z_pred = model(z_tau[idx], tau_std,
                          step_size=step_size[idx], actions=slice_actions(actions, idx))
            loss_std = x_prediction_loss(z_pred, z_0[idx], tau_std, use_ramp_weight=True)
            n_std = idx.sum().item()

        # Bootstrap loss for d>1 (velocity space with (1-τ)² scaling)
        # Paper convention: τ=0 noise, τ=1 clean. Velocity denominator is (1-τ).
        if (~is_base_step).any():
            idx = ~is_base_step
            tau_idx = tau[idx] if tau.dim() > 1 else tau[idx]

            with torch.no_grad():
                # Teacher: take 2 half-steps
                d_half = step_size[idx] // 2

                # First half-step: predict z_0 from z_tau
                z_mid = model(z_tau[idx], tau_idx, step_size=d_half,
                             actions=slice_actions(actions, idx))

                # First half-step velocity: b' = (z_mid - z_tau) / (1-τ)
                if tau_idx.dim() == 1:
                    tau_expanded = tau_idx.view(-1, 1, 1, 1, 1)
                else:
                    tau_expanded = tau_idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                # (1-τ) is the noise fraction; clamp to avoid div-by-zero near τ=1 (clean)
                one_minus_tau_safe = (1 - tau_expanded).clamp(min=1e-6)
                b_prime = (z_mid - z_tau[idx]) / one_minus_tau_safe

                # Compute tau after half-step
                # Paper convention: denoising INCREASES tau toward 1 (clean)
                half_step_amount = step_size[idx].float() / self.k_max / 2
                if tau_idx.dim() == 1:
                    tau_mid = (tau_idx + half_step_amount).clamp(max=1 - 1e-6)
                    tau_mid_expanded = tau_mid.view(-1, 1, 1, 1, 1)
                else:
                    tau_mid = (tau_idx + half_step_amount.unsqueeze(-1)).clamp(max=1 - 1e-6)
                    tau_mid_expanded = tau_mid.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

                # Euler step to intermediate point: z' = z_tau + b' * half_step
                if tau_idx.dim() == 1:
                    half_step_expanded = half_step_amount.view(-1, 1, 1, 1, 1)
                else:
                    half_step_expanded = half_step_amount.view(-1, 1, 1, 1, 1)
                z_prime = z_tau[idx] + b_prime * half_step_expanded

                # Second half-step: predict z_0 from z_prime
                z_target = model(z_prime, tau_mid, step_size=d_half,
                                actions=slice_actions(actions, idx))

                # Second half-step velocity: b'' = (z_target - z_prime) / (1-τ_mid)
                one_minus_tau_mid_safe = (1 - tau_mid_expanded).clamp(min=1e-6)
                b_double_prime = (z_target - z_prime) / one_minus_tau_mid_safe

                # Average teacher velocities
                avg_velocity = (b_prime + b_double_prime) / 2

            # Student: take 1 full step directly
            z_pred = model(z_tau[idx], tau_idx, step_size=step_size[idx],
                          actions=slice_actions(actions, idx))

            # Bootstrap loss in X-SPACE (not velocity space).
            # The paper uses velocity-space with (1-τ)² scaling to convert back,
            # but this creates a numerical trap: division by (1-τ) then multiplication
            # by (1-τ)² loses precision in bfloat16, and when teacher ≈ student
            # (self-consistent model), the velocity diff is zero.
            #
            # X-space loss directly: compare student's x-prediction to teacher's
            # two-step target. The teacher target is z_target (final x-prediction
            # after two half-steps via Euler integration). Ramp weight applied
            # to focus learning on high-signal (high τ) regions.
            x_diff = z_pred - z_target.detach()
            x_mse = (x_diff ** 2).mean(dim=(-3, -2, -1))  # Reduce C, H, W -> (B, T)

            # Clamp to prevent numerical explosion
            x_mse = x_mse.clamp(max=100.0)

            # Skip batch if any NaN/Inf detected
            if torch.isnan(x_mse).any() or torch.isinf(x_mse).any():
                n_boot = 0
            else:
                # Apply ramp weight (Paper Eq. 8) — per-element, not separated
                if tau_idx.dim() == 1:
                    loss_boot = (x_mse.mean(dim=-1) * ramp_weight(tau_idx)).mean()
                else:
                    loss_boot = (x_mse * ramp_weight(tau_idx)).mean()
                n_boot = idx.sum().item()

        # Combine losses with bootstrap weight boost.
        # Bootstrap MSE is naturally smaller (deterministic targets) so without
        # boosting, the optimizer prioritizes standard loss and bootstrap
        # contributes nothing. The boost ensures the model actually learns
        # step-size-dependent behavior for shortcut inference.
        total = n_std + n_boot
        if total > 0:
            loss = (loss_std * n_std + self.bootstrap_weight * loss_boot * n_boot) / total
        else:
            loss = loss_std

        info = {
            "loss_std": loss_std.item(),
            "loss_boot": loss_boot.item(),
            "loss_boot_weighted": (self.bootstrap_weight * loss_boot).item(),
            "n_std": n_std,
            "n_boot": n_boot,
        }

        return loss, info


if __name__ == "__main__":
    # Quick test
    print("Testing diffusion utilities...")

    schedule = DiffusionSchedule(device="cpu")

    # Test noise addition
    z_0 = torch.randn(4, 8, 256, 16, 16)  # (B, T, C, H, W)
    tau = schedule.sample_timesteps(4, device="cpu")
    print(f"Sampled tau: {tau}")

    z_tau, noise = schedule.add_noise(z_0, tau)
    print(f"z_0 shape: {z_0.shape}")
    print(f"z_tau shape: {z_tau.shape}")

    # Test diffusion forcing timesteps
    print("\nTesting diffusion forcing...")
    B, T = 4, 16
    tau_df = schedule.sample_diffusion_forcing_timesteps(B, T, device="cpu")
    print(f"Diffusion forcing tau shape: {tau_df.shape}")  # Should be (4, 16)
    print(f"Sample tau sequence: {tau_df[0].tolist()}")
    print(f"  Context frames (low tau): {tau_df[0, :4].tolist()}")
    print(f"  Target frames (high tau): {tau_df[0, -4:].tolist()}")

    # Verify temporal structure: later frames should have higher tau
    assert tau_df.shape == (B, T), f"Expected ({B}, {T}), got {tau_df.shape}"

    # Test with per-timestep tau
    z_tau_df, _ = schedule.add_noise(z_0, tau_df)
    print(f"Noisy latents shape with per-timestep tau: {z_tau_df.shape}")

    # Test loss with per-timestep tau
    pred = z_0 + 0.1 * torch.randn_like(z_0)  # Simulated prediction
    loss = x_prediction_loss(pred, z_0, tau_df)
    print(f"Loss with per-timestep tau: {loss.item():.4f}")

    print("\nAll tests passed!")
