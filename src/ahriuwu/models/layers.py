"""Shared model layers used across dynamics and tokenizer models.

Contains: RMSNorm, QKNorm, SwiGLU, soft_cap_attention.
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
