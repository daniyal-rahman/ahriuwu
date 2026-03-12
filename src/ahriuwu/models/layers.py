"""Shared model layers used across dynamics and tokenizer models.

Contains: RMSNorm, QKNorm, SwiGLU, soft_cap_attention,
          RotaryEmbedding1D, RotaryEmbedding2D, apply_rotary_emb.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class QKNorm(nn.Module):
    """Query-Key Normalization for attention stability.

    Normalizes Q and K independently before computing attention scores.
    This prevents attention logits from growing too large with scale.

    Reference: Gemma 2, DreamerV4
    """

    def __init__(self, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.q_norm = RMSNorm(head_dim, eps)
        self.k_norm = RMSNorm(head_dim, eps)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize Q and K.

        Args:
            q: (..., seq, head_dim)
            k: (..., seq, head_dim)

        Returns:
            Normalized (q, k)
        """
        return self.q_norm(q), self.k_norm(k)


def soft_cap_attention(logits: torch.Tensor, cap: float = 50.0) -> torch.Tensor:
    """Apply soft capping to attention logits.

    Prevents extreme attention scores using tanh squashing.
    logits = cap * tanh(logits / cap)

    Reference: Gemma 2 uses cap=50.0

    Args:
        logits: Attention logits of any shape
        cap: Soft cap value (default 50.0 from Gemma 2)

    Returns:
        Soft-capped logits
    """
    return cap * torch.tanh(logits / cap)


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


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate consecutive pairs: [a, b, c, d, ...] -> [-b, a, -d, c, ...]."""
    x = x.reshape(*x.shape[:-1], -1, 2)
    x = torch.stack([-x[..., 1], x[..., 0]], dim=-1)
    return x.reshape(*x.shape[:-2], -1)


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary positional embedding to tensor x.

    Args:
        x: (..., seq, dim) tensor
        cos: (..., seq, dim) cosine frequencies
        sin: (..., seq, dim) sine frequencies

    Returns:
        Rotated tensor of same shape as x
    """
    return (x * cos) + (_rotate_half(x) * sin)


class RotaryEmbedding1D(nn.Module):
    """1D Rotary Position Embedding for temporal positions (0..T-1)."""

    def __init__(self, dim: int, max_seq_len: int = 256, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def get_rotary_emb(self, seq_len: int, device: torch.device):
        """Get cos/sin embeddings for positions 0..seq_len-1.

        Returns:
            cos, sin each of shape (seq_len, dim)
        """
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(t, self.inv_freq)  # (T, dim/2)
        freqs = freqs.repeat_interleave(2, dim=-1)  # (T, dim)
        return freqs.cos(), freqs.sin()


class RotaryEmbedding2D(nn.Module):
    """2D Rotary Position Embedding (RoPE) for image patches.

    Applies separate rotations for x and y coordinates using axial decomposition.
    RoPE encodes relative position in the attention dot product.

    Head dim is split in half: first half for y-axis, second half for x-axis.
    Each half is further split for the rotation pairs.
    """

    def __init__(self, dim: int, grid_size: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.grid_size = grid_size
        self.dim_per_axis = dim // 2  # Half for y, half for x

        # Compute frequencies for each axis (half the per-axis dim for pairs)
        inv_freq = 1.0 / (base ** (torch.arange(0, self.dim_per_axis, 2).float() / self.dim_per_axis))
        self.register_buffer('inv_freq', inv_freq)

        # Precompute position grid
        y_pos = torch.arange(grid_size).float()
        x_pos = torch.arange(grid_size).float()

        # Create meshgrid and flatten
        grid_y, grid_x = torch.meshgrid(y_pos, x_pos, indexing='ij')
        positions_y = grid_y.reshape(-1)  # (grid_size^2,)
        positions_x = grid_x.reshape(-1)  # (grid_size^2,)

        self.register_buffer('positions_y', positions_y)
        self.register_buffer('positions_x', positions_x)

    def get_rotary_emb(self, seq_len: int, device: torch.device):
        """Get sin/cos embeddings for rotary application.

        Returns cos, sin each of shape (seq_len, dim) where:
        - dims 0:dim/2 encode y position
        - dims dim/2:dim encode x position
        """
        # Compute frequencies for each position
        freqs_y = torch.outer(self.positions_y[:seq_len], self.inv_freq)  # (seq, dim/4)
        freqs_x = torch.outer(self.positions_x[:seq_len], self.inv_freq)  # (seq, dim/4)

        # Repeat each frequency for the pair
        freqs_y = freqs_y.repeat_interleave(2, dim=-1)  # (seq, dim/2)
        freqs_x = freqs_x.repeat_interleave(2, dim=-1)  # (seq, dim/2)

        # Combine y and x: [y_freqs, x_freqs]
        freqs = torch.cat([freqs_y, freqs_x], dim=-1)  # (seq, dim)

        return freqs.cos(), freqs.sin()
