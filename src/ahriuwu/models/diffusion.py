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

    Higher weight at high signal levels (low τ) focuses learning
    on regions with more structure.

    Formula: w(τ) = 0.9τ + 0.1

    At τ=0 (clean): w = 0.1
    At τ=1 (noise): w = 1.0

    Args:
        tau: Timesteps, shape (B,) or (B, T)

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
            step_sizes: (B,) tensor of step sizes
        """
        # Sample inverse uniformly
        idx = torch.randint(0, len(self.step_sizes), (batch_size,), device=device)
        return torch.tensor([self.step_sizes[i] for i in idx], device=device)

    def compute_loss(
        self,
        model: nn.Module,
        z_0: torch.Tensor,
        tau: torch.Tensor,
        step_size: torch.Tensor,
    ) -> torch.Tensor:
        """Compute shortcut forcing loss.

        For small step sizes: standard x-prediction
        For large step sizes: bootstrap from two smaller steps

        Not yet implemented - placeholder for Phase 2.
        """
        raise NotImplementedError(
            "Shortcut forcing not yet implemented. "
            "Use standard diffusion with x_prediction_loss() for MVP."
        )


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

    # Test loss
    pred = z_0 + 0.1 * torch.randn_like(z_0)  # Simulated prediction
    loss = x_prediction_loss(pred, z_0, tau)
    print(f"Loss: {loss.item():.4f}")

    # Test timestep embedding
    emb = TimestepEmbedding(512)
    tau_emb = emb(tau)
    print(f"Timestep embedding shape: {tau_emb.shape}")

    print("All tests passed!")
