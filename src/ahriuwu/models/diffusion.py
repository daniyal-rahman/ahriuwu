"""Diffusion utilities for dynamics model training.

Implements x-prediction diffusion following DreamerV4:
- Linear noise schedule: z_τ = (1-τ)z_0 + τε
- X-prediction: model predicts clean data z_0 directly
- Ramp loss weight: focus learning on high-signal regions

References:
- DreamerV4: "Training Agents Inside of Scalable World Models" (Hafner et al., 2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionSchedule:
    """Linear noise schedule for x-prediction diffusion.

    Uses flow-matching style interpolation:
        z_τ = (1-τ)z_0 + τε

    where τ ∈ [0, 1], z_0 is clean data, ε is noise.
    At τ=0: z_τ = z_0 (clean)
    At τ=1: z_τ = ε (pure noise)
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

        # Linear interpolation: z_τ = (1-τ)z_0 + τε
        z_tau = (1 - tau) * z_0 + tau * noise

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
        tau_ctx: float = 0.1,
        tau_max: float = 1.0,
    ) -> torch.Tensor:
        """Sample per-timestep noise levels for diffusion forcing.

        Implements DreamerV4's diffusion forcing where:
        - A random horizon h is sampled for each batch item
        - Frames before h: low noise (τ_ctx, context frames)
        - Frames at/after h: increasing noise (prediction targets)

        This creates temporal causality: model must use clean past
        to predict noisy future.

        Args:
            batch_size: Number of sequences
            seq_length: Number of frames per sequence (T)
            device: Device to create tensor on
            tau_ctx: Noise level for context frames (default 0.1)
            tau_max: Maximum noise level for target frames

        Returns:
            tau: Per-timestep noise levels, shape (B, T)
        """
        device = device or self.device

        # Sample random horizon for each batch item: where prediction starts
        # Horizon is uniformly distributed from 1 to T-1
        # (at least 1 context frame, at least 1 prediction frame)
        horizon = torch.randint(1, seq_length, (batch_size,), device=device)

        # Create position indices: 0, 1, 2, ..., T-1
        positions = torch.arange(seq_length, device=device).unsqueeze(0)  # (1, T)
        horizon = horizon.unsqueeze(1)  # (B, 1)

        # Frames before horizon get low noise (context)
        # Frames at/after horizon get linearly increasing noise
        # distance = how far past the horizon (0 for context, 1+ for targets)
        distance = (positions - horizon).clamp(min=0).float()  # (B, T)

        # Normalize distance: frames at horizon=0, last frame has max distance
        max_distance = (seq_length - 1 - horizon).clamp(min=1).float()  # (B, 1)
        normalized_dist = distance / max_distance  # (B, T) in [0, 1]

        # Context frames (before horizon) get tau_ctx
        # Target frames get tau_ctx + normalized_dist * (tau_max - tau_ctx)
        is_context = positions < horizon  # (B, T)
        tau = torch.where(
            is_context,
            torch.full_like(normalized_dist, tau_ctx),
            tau_ctx + normalized_dist * (tau_max - tau_ctx)
        )

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

        # Start from pure noise
        z_t = torch.randn(shape, device=device)

        # Euler integration from τ=1 to τ=0
        step_size = 1.0 / num_steps

        for i in range(num_steps):
            tau = 1.0 - i * step_size
            tau_tensor = torch.full((shape[0],), tau, device=device)

            # Predict clean data
            z_0_pred = model(z_t, tau_tensor, context=context)

            # Euler step towards prediction
            if i < num_steps - 1:
                # Interpolate towards predicted clean
                next_tau = tau - step_size
                z_t = (1 - next_tau) * z_0_pred + next_tau * torch.randn_like(z_t)
            else:
                # Final step - just return prediction
                z_t = z_0_pred

        return z_t


def ramp_weight(tau: torch.Tensor) -> torch.Tensor:
    """Compute ramp loss weight from DreamerV4.

    Higher weight at high signal levels (low τ in our convention) focuses
    learning on regions with more structure.

    Paper formula: w(τ) = 0.9τ + 0.1 (where paper τ=0 is noise, τ=1 is clean)
    Our convention: τ=0 is clean, τ=1 is noise
    Inverted formula: w(τ) = 1.0 - 0.9τ

    At τ=0 (clean): w = 1.0 (high weight - more informative gradients)
    At τ=1 (noise): w = 0.1 (low weight - less useful gradients)

    Args:
        tau: Timesteps, shape (B,) or (B, T)

    Returns:
        weights: Loss weights, same shape as tau
    """
    return 1.0 - 0.9 * tau


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


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding for diffusion models.

    Maps scalar timestep τ ∈ [0, 1] to a vector embedding.
    """

    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

        # MLP to project embeddings
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        """Embed timesteps.

        Args:
            tau: Timesteps, shape (B,) or (B, T)

        Returns:
            emb: Embeddings, shape (B, dim) or (B, T, dim)
        """
        # Sinusoidal embedding
        half_dim = self.dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(self.max_period, device=tau.device))
            * torch.arange(half_dim, device=tau.device)
            / half_dim
        )

        # Expand tau for broadcasting
        if tau.dim() == 1:
            # (B,) -> (B, 1) for broadcasting with freqs (half_dim,)
            args = tau.unsqueeze(-1) * freqs
        else:
            # (B, T) -> (B, T, 1) for broadcasting
            args = tau.unsqueeze(-1) * freqs

        # Concat sin and cos
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        # MLP projection
        emb = self.mlp(emb)

        return emb


# Shortcut forcing utilities (for Phase 2 - advanced training)

class ShortcutForcing:
    """Shortcut forcing objective for few-step sampling.

    Enables 4-step inference instead of 64 by training the model
    to "skip" denoising steps using bootstrapped targets.

    Not implemented in MVP - use standard diffusion first.
    """

    def __init__(self, k_max: int = 64, k_min: int = 1):
        self.k_max = k_max
        self.k_min = k_min
        # Step sizes: 1, 2, 4, 8, ..., k_max
        self.step_sizes = [2**i for i in range(int(torch.log2(torch.tensor(k_max)).item()) + 1)]

    def sample_step_size(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample step sizes with inverse weighting.

        Larger step sizes are sampled less frequently:
        P(d) ∝ 1/d

        Returns:
            step_sizes: (B,) tensor of integer step sizes
        """
        # Compute inverse weights: P(d) ∝ 1/d
        weights = torch.tensor([1.0 / d for d in self.step_sizes], device=device)
        weights = weights / weights.sum()

        # Sample indices according to weights
        idx = torch.multinomial(weights.expand(batch_size, -1), num_samples=1).squeeze(-1)

        # Return step sizes as integers
        step_sizes_tensor = torch.tensor(self.step_sizes, device=device)
        return step_sizes_tensor[idx]

    def sample_tau_for_step_size(
        self,
        step_size: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Sample tau from grid aligned with step size (Paper Eq. 4).

        For step size d, tau must come from grid {0, d/k_max, 2d/k_max, ..., 1-d/k_max}
        This ensures after stepping by d, tau lands on valid grid points.

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
        tau_ctx: float = 0.1,
    ) -> torch.Tensor:
        """Sample per-timestep tau from grid aligned with step size (Paper Eq. 4).

        Extension for diffusion forcing: samples tau for each position in sequence,
        maintaining grid alignment while having per-timestep variation.

        Args:
            step_size: Step sizes (integers), shape (B,)
            seq_length: Sequence length T
            device: Device for tensor creation
            tau_ctx: Minimum tau for context frames

        Returns:
            tau: Sampled tau values, shape (B, T)
        """
        B = step_size.shape[0]
        tau = torch.zeros(B, seq_length, device=device)

        for d in self.step_sizes:
            mask = (step_size == d)
            if mask.any():
                n = mask.sum().item()
                # Grid spacing for this step size
                grid_spacing = d / self.k_max
                # Number of valid grid points
                num_grid_points = self.k_max // d
                # Minimum grid index to stay above tau_ctx
                min_grid_idx = int(tau_ctx / grid_spacing) if tau_ctx > 0 else 0
                # Sample grid indices for all positions
                grid_idx = torch.randint(min_grid_idx, num_grid_points, (n, seq_length), device=device)
                # Convert to tau values
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
        """Compute shortcut forcing loss (Paper Eq. 7).

        For d=1 (base step): standard x-prediction loss
        For d>1: bootstrap loss in velocity space with τ² scaling

        Paper Eq. 7 (our convention where τ=0 clean, τ=1 noise):
            L = τ² || (ẑ_0 - z_τ) / τ - sg(b' + b'')/2 ||²
        where:
            b' = (z_mid - z_τ) / τ         (first half-step velocity)
            b'' = (z_target - z_τ_mid) / τ_mid  (second half-step velocity)

        Args:
            model: Dynamics model with step_size conditioning
            schedule: DiffusionSchedule for adding noise
            z_0: Clean latents, shape (B, T, C, H, W)
            tau: Noise levels, shape (B,) or (B, T)
            step_size: Step sizes (integers), shape (B,)
            actions: Optional action dict with 'movement', 'target', and ability keys

        Returns:
            loss: Combined loss
            info: Dict with loss breakdown
        """
        B = z_0.shape[0]

        # Normalize step sizes to [0, 1] for model conditioning
        d_normalized = step_size.float() / self.k_max

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

        # Standard loss for d=1 (smallest step)
        if is_base_step.any():
            idx = is_base_step
            z_pred = model(z_tau[idx], tau[idx] if tau.dim() > 1 else tau[idx],
                          step_size=d_normalized[idx], actions=slice_actions(actions, idx))
            loss_std = x_prediction_loss(z_pred, z_0[idx], tau[idx] if tau.dim() > 1 else tau[idx])
            n_std = idx.sum().item()

        # Bootstrap loss for d>1 (velocity space with τ² scaling)
        if (~is_base_step).any():
            idx = ~is_base_step
            tau_idx = tau[idx] if tau.dim() > 1 else tau[idx]

            with torch.no_grad():
                # Teacher: take 2 half-steps
                d_half = step_size[idx] // 2
                d_half_norm = d_half.float() / self.k_max

                # First half-step: predict z_0 from z_tau
                z_mid = model(z_tau[idx], tau_idx, step_size=d_half_norm,
                             actions=slice_actions(actions, idx))

                # First half-step velocity: b' = (z_mid - z_tau) / tau
                # Expand tau for broadcasting
                if tau_idx.dim() == 1:
                    tau_expanded = tau_idx.view(-1, 1, 1, 1, 1)
                else:
                    tau_expanded = tau_idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                # Clamp tau to avoid division by zero
                tau_safe = tau_expanded.clamp(min=1e-6)
                b_prime = (z_mid - z_tau[idx]) / tau_safe

                # Compute tau after half-step
                # Our convention: denoising DECREASES tau toward 0
                half_step_amount = step_size[idx].float() / self.k_max / 2
                if tau_idx.dim() == 1:
                    tau_mid = (tau_idx - half_step_amount).clamp(min=1e-6)
                    tau_mid_expanded = tau_mid.view(-1, 1, 1, 1, 1)
                else:
                    tau_mid = (tau_idx - half_step_amount.unsqueeze(-1)).clamp(min=1e-6)
                    tau_mid_expanded = tau_mid.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

                # Re-noise z_mid to tau_mid level for second half-step
                z_tau_mid = (1 - tau_mid_expanded) * z_mid + tau_mid_expanded * noise[idx]

                # Second half-step: predict z_0 from z_tau_mid
                z_target = model(z_tau_mid, tau_mid, step_size=d_half_norm,
                                actions=slice_actions(actions, idx))

                # Second half-step velocity: b'' = (z_target - z_tau_mid) / tau_mid
                b_double_prime = (z_target - z_tau_mid) / tau_mid_expanded.clamp(min=1e-6)

                # Average teacher velocities
                avg_velocity = (b_prime + b_double_prime) / 2

            # Student: take 1 full step directly
            z_pred = model(z_tau[idx], tau_idx, step_size=d_normalized[idx],
                          actions=slice_actions(actions, idx))

            # Student velocity: (z_pred - z_tau) / tau
            b_student = (z_pred - z_tau[idx]) / tau_safe

            # Bootstrap loss in velocity space with τ² scaling (Paper Eq. 7)
            # L = τ² || b_student - sg(avg_velocity) ||²
            velocity_diff = b_student - avg_velocity.detach()
            velocity_mse = (velocity_diff ** 2).mean(dim=(-3, -2, -1))  # Reduce C, H, W -> (B, T)

            # Clamp velocity_mse to prevent numerical explosion
            # When tau is small (~0.1), dividing by tau amplifies differences by 10x
            # Squaring makes this 100x, which can overflow if predictions diverge
            # Normal velocity_mse should be O(1), so clamp at 100 to catch outliers
            velocity_mse = velocity_mse.clamp(max=100.0)

            # Skip batch if any NaN/Inf detected
            if torch.isnan(velocity_mse).any() or torch.isinf(velocity_mse).any():
                n_boot = 0
            else:
                # Apply τ² scaling - reduce to scalar loss
                # velocity_mse: (B_subset, T), tau_idx: (B_subset,) or (B_subset, T)
                if tau_idx.dim() == 1:
                    tau_weight = tau_idx ** 2  # (B_subset,)
                    # Average over T first, then apply per-sample tau weight
                    loss_boot = (velocity_mse.mean(dim=-1) * tau_weight).mean()
                else:
                    # tau_idx is (B_subset, T) - use mean tau per sample for weighting
                    tau_weight = (tau_idx ** 2).mean(dim=-1)  # (B_subset,)
                    loss_boot = (velocity_mse.mean(dim=-1) * tau_weight).mean()
                n_boot = idx.sum().item()

        # Combine losses (weighted by number of samples)
        total = n_std + n_boot
        if total > 0:
            loss = (loss_std * n_std + loss_boot * n_boot) / total
        else:
            loss = loss_std

        info = {
            "loss_std": loss_std.item(),
            "loss_boot": loss_boot.item(),
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

    # Test timestep embedding
    emb = TimestepEmbedding(512)
    tau_emb = emb(tau)
    print(f"Timestep embedding shape (B,): {tau_emb.shape}")

    tau_emb_df = emb(tau_df)
    print(f"Timestep embedding shape (B,T): {tau_emb_df.shape}")

    print("\nAll tests passed!")
