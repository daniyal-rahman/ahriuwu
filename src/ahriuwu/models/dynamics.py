"""Dynamics Transformer for world model.

Predicts future latent frames using diffusion with factorized attention.
Architecture follows DreamerV4 with simplifications for MVP:

- Factorized attention: spatial within frames, temporal across frames
- Temporal attention every 4th layer (efficiency optimization)
- X-prediction objective (predicts clean data directly)
- RMSNorm, SwiGLU, learned positional embeddings

References:
- DreamerV4: "Training Agents Inside of Scalable World Models" (Hafner et al., 2025)
"""

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .diffusion import TimestepEmbedding


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network.

    SwiGLU(x) = (xW₁ ⊙ Swish(xW₂))W₃
    """

    def __init__(self, dim: int, hidden_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 8 / 3)
        # Round to multiple of 64 for efficiency
        hidden_dim = ((hidden_dim + 63) // 64) * 64

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class SpatialAttention(nn.Module):
    """Self-attention within each frame.

    Attends over 256 spatial tokens (16×16 grid) independently for each frame.
    Uses standard multi-head attention.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim or dim // num_heads
        self.scale = self.head_dim ** -0.5

        inner_dim = self.num_heads * self.head_dim

        self.qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.out_proj = nn.Linear(inner_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, T, S, D) where S=256 spatial tokens

        Returns:
            (B, T, S, D) attended features
        """
        B, T, S, D = x.shape

        # Reshape to (B*T, S, D) for efficient batched attention
        x = x.view(B * T, S, D)

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B * T, S, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B*T, heads, S, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B * T, S, -1)
        out = self.out_proj(out)

        # Reshape back to (B, T, S, D)
        out = out.view(B, T, S, D)
        return out


class TemporalAttention(nn.Module):
    """Causal self-attention across frames.

    Attends over T frames at each spatial position independently.
    Uses causal mask so frame t can only attend to frames 0..t.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: int | None = None,
        dropout: float = 0.0,
        max_seq_len: int = 256,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim or dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.max_seq_len = max_seq_len

        inner_dim = self.num_heads * self.head_dim

        self.qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.out_proj = nn.Linear(inner_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Pre-compute causal mask
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, T, S, D) where T is sequence length

        Returns:
            (B, T, S, D) attended features
        """
        B, T, S, D = x.shape

        # Reshape to (B*S, T, D) for efficient batched attention over time
        x = x.permute(0, 2, 1, 3).reshape(B * S, T, D)

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B * S, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B*S, heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention with causal mask
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Apply causal mask
        mask = self.causal_mask[:T, :T]
        attn = attn.masked_fill(mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B * S, T, -1)
        out = self.out_proj(out)

        # Reshape back to (B, T, S, D)
        out = out.view(B, S, T, D).permute(0, 2, 1, 3)
        return out


class TransformerBlock(nn.Module):
    """Transformer block with either spatial or temporal attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: int | None = None,
        attn_type: Literal["spatial", "temporal"] = "spatial",
        dropout: float = 0.0,
        max_seq_len: int = 256,
    ):
        super().__init__()
        self.attn_type = attn_type

        self.norm1 = RMSNorm(dim)
        if attn_type == "spatial":
            self.attn = SpatialAttention(dim, num_heads, head_dim, dropout)
        else:
            self.attn = TemporalAttention(dim, num_heads, head_dim, dropout, max_seq_len)

        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, dropout=dropout)

        # AdaLN modulation for timestep conditioning
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6),
        )

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, T, S, D) features
            time_emb: (B, D) or (B, T, D) timestep embedding for conditioning

        Returns:
            (B, T, S, D) transformed features
        """
        if time_emb is not None:
            # AdaLN modulation
            # Expand time_emb to match x shape: (B, D) -> (B, 1, 1, D*6)
            if time_emb.dim() == 2:
                modulation = self.adaLN_modulation(time_emb).unsqueeze(1).unsqueeze(1)
            else:
                # (B, T, D) -> (B, T, 1, D*6)
                modulation = self.adaLN_modulation(time_emb).unsqueeze(2)

            shift1, scale1, gate1, shift2, scale2, gate2 = modulation.chunk(6, dim=-1)

            # Attention with modulation
            h = self.norm1(x)
            h = h * (1 + scale1) + shift1
            h = self.attn(h)
            x = x + gate1 * h

            # FFN with modulation
            h = self.norm2(x)
            h = h * (1 + scale2) + shift2
            h = self.ffn(h)
            x = x + gate2 * h
        else:
            # Standard pre-norm transformer
            x = x + self.attn(self.norm1(x))
            x = x + self.ffn(self.norm2(x))

        return x


class DynamicsTransformer(nn.Module):
    """Dynamics model for world model training.

    Predicts clean latent frames from noisy inputs using diffusion.
    Uses factorized attention with temporal attention every Nth layer.

    Architecture:
    - Input projection: latent tokens to model dim
    - Spatial + temporal position embeddings
    - Timestep embedding (diffusion)
    - Transformer blocks with factorized attention
    - Output projection back to latent dim
    """

    def __init__(
        self,
        latent_dim: int = 256,
        spatial_size: int = 16,  # 16×16 = 256 spatial tokens
        model_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        head_dim: int | None = None,
        temporal_every: int = 4,  # Temporal attention every N layers
        dropout: float = 0.0,
        max_seq_len: int = 256,  # Max frames in sequence
    ):
        """Initialize dynamics transformer.

        Args:
            latent_dim: Dimension of input latent tokens (from tokenizer)
            spatial_size: Size of spatial grid (16 for 16×16)
            model_dim: Model hidden dimension
            num_layers: Number of transformer blocks
            num_heads: Number of attention heads
            head_dim: Dimension per head (default: model_dim // num_heads)
            temporal_every: Add temporal attention every N layers
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.spatial_size = spatial_size
        self.spatial_tokens = spatial_size * spatial_size  # 256
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.temporal_every = temporal_every

        # Input projection: (B, T, C, H, W) -> (B, T, S, D)
        self.input_proj = nn.Linear(latent_dim, model_dim)

        # Positional embeddings
        self.spatial_pos = nn.Parameter(
            torch.randn(1, 1, self.spatial_tokens, model_dim) * 0.02
        )
        self.temporal_pos = nn.Parameter(
            torch.randn(1, max_seq_len, 1, model_dim) * 0.02
        )

        # Timestep embedding for diffusion
        self.time_embed = TimestepEmbedding(model_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            # Temporal attention every temporal_every layers (on the last of each group)
            is_temporal = (i % temporal_every == temporal_every - 1)
            attn_type = "temporal" if is_temporal else "spatial"

            self.blocks.append(
                TransformerBlock(
                    dim=model_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    attn_type=attn_type,
                    dropout=dropout,
                    max_seq_len=max_seq_len,
                )
            )

        # Output projection
        self.norm_out = RMSNorm(model_dim)
        self.output_proj = nn.Linear(model_dim, latent_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p, gain=0.02)
            elif "bias" in name:
                nn.init.zeros_(p)

        # Zero-init output projection for residual
        nn.init.zeros_(self.output_proj.weight)

    def forward(
        self,
        z_tau: torch.Tensor,
        tau: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass: predict clean latents from noisy input.

        Args:
            z_tau: Noisy latents, shape (B, T, C, H, W)
            tau: Diffusion timesteps, shape (B,) or (B, T)
            context: Optional context frames (not used in MVP)

        Returns:
            z_0_pred: Predicted clean latents, shape (B, T, C, H, W)
        """
        B, T, C, H, W = z_tau.shape
        assert H == W == self.spatial_size, f"Expected {self.spatial_size}×{self.spatial_size}, got {H}×{W}"

        # Reshape to (B, T, S, C) where S = H*W
        x = z_tau.view(B, T, C, -1).permute(0, 1, 3, 2)  # (B, T, S, C)

        # Project to model dim
        x = self.input_proj(x)  # (B, T, S, D)

        # Add positional embeddings
        x = x + self.spatial_pos[:, :, :self.spatial_tokens, :]
        x = x + self.temporal_pos[:, :T, :, :]

        # Get timestep embedding
        time_emb = self.time_embed(tau)  # (B, D) or (B, T, D)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, time_emb)

        # Output projection
        x = self.norm_out(x)
        x = self.output_proj(x)  # (B, T, S, C)

        # Reshape back to (B, T, C, H, W)
        x = x.permute(0, 1, 3, 2).view(B, T, C, H, W)

        return x

    def get_num_params(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_dynamics(
    size: str = "small",
    latent_dim: int = 256,
) -> DynamicsTransformer:
    """Create dynamics model with preset sizes.

    Args:
        size: One of "tiny", "small", "medium", "large"
        latent_dim: Dimension of latent tokens (must match tokenizer)

    Returns:
        DynamicsTransformer instance
    """
    configs = {
        "tiny": {
            "model_dim": 256,
            "num_layers": 6,
            "num_heads": 4,
        },
        "small": {
            "model_dim": 512,
            "num_layers": 12,
            "num_heads": 8,
        },
        "medium": {
            "model_dim": 768,
            "num_layers": 18,
            "num_heads": 12,
        },
        "large": {
            "model_dim": 512,
            "num_layers": 24,
            "num_heads": 8,
        },
    }

    if size not in configs:
        raise ValueError(f"Unknown size: {size}. Choose from {list(configs.keys())}")

    return DynamicsTransformer(
        latent_dim=latent_dim,
        **configs[size],
    )


if __name__ == "__main__":
    # Quick test
    print("Testing dynamics transformer...")

    model = create_dynamics("small", latent_dim=256)
    print(f"Parameters: {model.get_num_params():,}")

    # Test forward pass
    B, T, C, H, W = 2, 8, 256, 16, 16
    z_tau = torch.randn(B, T, C, H, W)
    tau = torch.rand(B)

    z_pred = model(z_tau, tau)
    print(f"Input shape: {z_tau.shape}")
    print(f"Output shape: {z_pred.shape}")
    print(f"Tau shape: {tau.shape}")

    # Test with sequence tau
    tau_seq = torch.rand(B, T)
    z_pred_seq = model(z_tau, tau_seq)
    print(f"Sequence tau shape: {tau_seq.shape}")
    print(f"Output shape: {z_pred_seq.shape}")

    print("All tests passed!")
