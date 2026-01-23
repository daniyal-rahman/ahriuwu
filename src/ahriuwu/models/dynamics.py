"""Dynamics Transformer for world model.

Predicts future latent frames using diffusion with factorized attention.
Architecture follows DreamerV4 with simplifications for MVP:

- Factorized attention: spatial within frames, temporal across frames
- Temporal attention every 4th layer (efficiency optimization)
- X-prediction objective (predicts clean data directly)
- RMSNorm, SwiGLU, learned positional embeddings
- Agent tokens for policy/reward prediction (Phase 2+)

References:
- DreamerV4: "Training Agents Inside of Scalable World Models" (Hafner et al., 2025)
"""

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .diffusion import TimestepEmbedding

# Action space constants (must match data/actions.py)
MOVEMENT_CLASSES = 18  # 0-17 = directions (20° apart)
ABILITY_KEYS = ['Q', 'W', 'E', 'R', 'D', 'F', 'item', 'B']


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


class SpatialAttention(nn.Module):
    """Self-attention within each frame.

    Attends over 256 spatial tokens (16×16 grid) independently for each frame.
    Supports GQA (Grouped Query Attention), QKNorm, and soft capping.

    Reference: DreamerV4 Section 3.2
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        dropout: float = 0.0,
        use_qk_norm: bool = True,
        soft_cap: float | None = 50.0,
    ):
        """Initialize spatial attention.

        Args:
            dim: Model dimension
            num_heads: Number of query heads
            num_kv_heads: Number of KV heads for GQA (None = same as num_heads)
            head_dim: Dimension per head (default: dim // num_heads)
            dropout: Dropout probability
            use_qk_norm: Whether to use QK normalization
            soft_cap: Soft cap value for attention logits (None = no capping)
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_qk_norm = use_qk_norm
        self.soft_cap = soft_cap

        # GQA: num_heads must be divisible by num_kv_heads
        assert num_heads % self.num_kv_heads == 0, \
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"
        self.num_groups = num_heads // self.num_kv_heads

        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim

        self.q_proj = nn.Linear(dim, q_dim, bias=False)
        self.k_proj = nn.Linear(dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(dim, kv_dim, bias=False)
        self.out_proj = nn.Linear(q_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        # QKNorm
        if use_qk_norm:
            self.qk_norm = QKNorm(self.head_dim)
        else:
            self.qk_norm = None

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
        q = self.q_proj(x).view(B * T, S, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B * T, S, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B * T, S, self.num_kv_heads, self.head_dim)

        # Transpose for attention: (B*T, heads, S, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply QKNorm if enabled
        if self.qk_norm is not None:
            q, k = self.qk_norm(q, k)

        # GQA: repeat KV heads to match Q heads
        if self.num_groups > 1:
            # (B*T, num_kv_heads, S, head_dim) -> (B*T, num_heads, S, head_dim)
            k = k.repeat_interleave(self.num_groups, dim=1)
            v = v.repeat_interleave(self.num_groups, dim=1)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Apply soft capping if enabled
        if self.soft_cap is not None:
            attn = soft_cap_attention(attn, self.soft_cap)

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
    Supports GQA, QKNorm, and soft capping.

    Reference: DreamerV4 Section 3.2
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        dropout: float = 0.0,
        max_seq_len: int = 256,
        use_qk_norm: bool = True,
        soft_cap: float | None = 50.0,
    ):
        """Initialize temporal attention.

        Args:
            dim: Model dimension
            num_heads: Number of query heads
            num_kv_heads: Number of KV heads for GQA (None = same as num_heads)
            head_dim: Dimension per head (default: dim // num_heads)
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
            use_qk_norm: Whether to use QK normalization
            soft_cap: Soft cap value for attention logits (None = no capping)
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.max_seq_len = max_seq_len
        self.use_qk_norm = use_qk_norm
        self.soft_cap = soft_cap

        # GQA: num_heads must be divisible by num_kv_heads
        assert num_heads % self.num_kv_heads == 0, \
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"
        self.num_groups = num_heads // self.num_kv_heads

        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim

        self.q_proj = nn.Linear(dim, q_dim, bias=False)
        self.k_proj = nn.Linear(dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(dim, kv_dim, bias=False)
        self.out_proj = nn.Linear(q_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        # QKNorm
        if use_qk_norm:
            self.qk_norm = QKNorm(self.head_dim)
        else:
            self.qk_norm = None

        # Pre-compute causal mask
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool(),
        )

    def forward(
        self,
        x: torch.Tensor,
        independent_frames: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, T, S, D) where T is sequence length
            independent_frames: If True, treat frames as independent (no temporal attention)

        Returns:
            (B, T, S, D) attended features
        """
        B, T, S, D = x.shape

        # Reshape to (B*S, T, D) for efficient batched attention over time
        x = x.permute(0, 2, 1, 3).reshape(B * S, T, D)

        # Compute Q, K, V
        q = self.q_proj(x).view(B * S, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B * S, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B * S, T, self.num_kv_heads, self.head_dim)

        # Transpose for attention: (B*S, heads, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply QKNorm if enabled
        if self.qk_norm is not None:
            q, k = self.qk_norm(q, k)

        # GQA: repeat KV heads to match Q heads
        if self.num_groups > 1:
            k = k.repeat_interleave(self.num_groups, dim=1)
            v = v.repeat_interleave(self.num_groups, dim=1)

        # Scaled dot-product attention with causal mask
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Apply soft capping if enabled
        if self.soft_cap is not None:
            attn = soft_cap_attention(attn, self.soft_cap)

        # Apply causal mask (or independent frame mask)
        if independent_frames:
            # Each frame only attends to itself (diagonal mask)
            # Create identity-like mask: only diagonal elements allowed
            diag_mask = ~torch.eye(T, dtype=torch.bool, device=x.device)
            attn = attn.masked_fill(diag_mask, float("-inf"))
        else:
            # Standard causal mask
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
    """Transformer block with either spatial or temporal attention.

    Supports GQA, QKNorm, soft capping, and independent frame mode.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        attn_type: Literal["spatial", "temporal"] = "spatial",
        dropout: float = 0.0,
        max_seq_len: int = 256,
        use_qk_norm: bool = True,
        soft_cap: float | None = 50.0,
    ):
        """Initialize transformer block.

        Args:
            dim: Model dimension
            num_heads: Number of query heads
            num_kv_heads: Number of KV heads for GQA (None = same as num_heads)
            head_dim: Dimension per head
            attn_type: "spatial" or "temporal"
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
            use_qk_norm: Whether to use QK normalization
            soft_cap: Soft cap value for attention logits
        """
        super().__init__()
        self.attn_type = attn_type

        self.norm1 = RMSNorm(dim)
        if attn_type == "spatial":
            self.attn = SpatialAttention(
                dim, num_heads, num_kv_heads, head_dim, dropout,
                use_qk_norm=use_qk_norm, soft_cap=soft_cap
            )
        else:
            self.attn = TemporalAttention(
                dim, num_heads, num_kv_heads, head_dim, dropout, max_seq_len,
                use_qk_norm=use_qk_norm, soft_cap=soft_cap
            )

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
        independent_frames: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, T, S, D) features
            time_emb: (B, D) or (B, T, D) timestep embedding for conditioning
            independent_frames: If True, treat frames as independent (temporal attention only)

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
            if self.attn_type == "temporal":
                h = self.attn(h, independent_frames=independent_frames)
            else:
                h = self.attn(h)
            x = x + gate1 * h

            # FFN with modulation
            h = self.norm2(x)
            h = h * (1 + scale2) + shift2
            h = self.ffn(h)
            x = x + gate2 * h
        else:
            # Standard pre-norm transformer
            if self.attn_type == "temporal":
                x = x + self.attn(self.norm1(x), independent_frames=independent_frames)
            else:
                x = x + self.attn(self.norm1(x))
            x = x + self.ffn(self.norm2(x))

        return x


class AgentCrossAttention(nn.Module):
    """Cross-attention where agent tokens query z tokens.

    Implements the asymmetric attention pattern from DreamerV4:
    - Agent tokens can attend to all z tokens
    - Z tokens cannot attend back to agent tokens (handled by keeping them separate)

    Supports GQA, QKNorm, and soft capping.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        dropout: float = 0.0,
        use_qk_norm: bool = True,
        soft_cap: float | None = 50.0,
    ):
        """Initialize agent cross-attention.

        Args:
            dim: Model dimension
            num_heads: Number of query heads
            num_kv_heads: Number of KV heads for GQA (None = same as num_heads)
            head_dim: Dimension per head
            dropout: Dropout probability
            use_qk_norm: Whether to use QK normalization
            soft_cap: Soft cap value for attention logits
        """
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_qk_norm = use_qk_norm
        self.soft_cap = soft_cap

        # GQA setup
        assert num_heads % self.num_kv_heads == 0
        self.num_groups = num_heads // self.num_kv_heads

        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim

        # Agent token query projection
        self.q_proj = nn.Linear(dim, q_dim, bias=False)
        # Z token key/value projections
        self.k_proj = nn.Linear(dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(dim, kv_dim, bias=False)
        self.out_proj = nn.Linear(q_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        # QKNorm
        if use_qk_norm:
            self.qk_norm = QKNorm(self.head_dim)
        else:
            self.qk_norm = None

    def forward(
        self,
        agent_tokens: torch.Tensor,
        z_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-attention from agent tokens to z tokens.

        Args:
            agent_tokens: (B, T, D) one agent token per frame
            z_tokens: (B, T, S, D) spatial tokens per frame

        Returns:
            (B, T, D) attended agent token features
        """
        B, T, D = agent_tokens.shape
        _, _, S, _ = z_tokens.shape

        # Agent queries: (B, T, num_heads, 1, head_dim)
        q = self.q_proj(agent_tokens).view(B, T, self.num_heads, 1, self.head_dim)

        # Z key/values: (B, T, num_kv_heads, S, head_dim)
        z_flat = z_tokens.view(B * T, S, D)
        k = self.k_proj(z_flat).view(B, T, S, self.num_kv_heads, self.head_dim)
        v = self.v_proj(z_flat).view(B, T, S, self.num_kv_heads, self.head_dim)

        # Transpose: (B, T, num_kv_heads, S, head_dim)
        k = k.permute(0, 1, 3, 2, 4)
        v = v.permute(0, 1, 3, 2, 4)

        # Apply QKNorm if enabled
        if self.qk_norm is not None:
            q, k = self.qk_norm(q, k)

        # GQA: repeat KV heads
        if self.num_groups > 1:
            k = k.repeat_interleave(self.num_groups, dim=2)
            v = v.repeat_interleave(self.num_groups, dim=2)

        # Attention: agent attends to all spatial tokens
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, T, heads, 1, S)

        # Apply soft capping if enabled
        if self.soft_cap is not None:
            attn = soft_cap_attention(attn, self.soft_cap)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention
        out = (attn @ v).squeeze(-2)  # (B, T, heads, head_dim)
        out = out.view(B, T, -1)
        out = self.out_proj(out)

        return out


class AgentTemporalAttention(nn.Module):
    """Causal self-attention for agent tokens across time.

    Agent tokens attend to themselves and past agent tokens.
    Supports GQA, QKNorm, and soft capping.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        dropout: float = 0.0,
        max_seq_len: int = 256,
        use_qk_norm: bool = True,
        soft_cap: float | None = 50.0,
    ):
        """Initialize agent temporal attention.

        Args:
            dim: Model dimension
            num_heads: Number of query heads
            num_kv_heads: Number of KV heads for GQA (None = same as num_heads)
            head_dim: Dimension per head
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
            use_qk_norm: Whether to use QK normalization
            soft_cap: Soft cap value for attention logits
        """
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_qk_norm = use_qk_norm
        self.soft_cap = soft_cap

        # GQA setup
        assert num_heads % self.num_kv_heads == 0
        self.num_groups = num_heads // self.num_kv_heads

        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim

        self.q_proj = nn.Linear(dim, q_dim, bias=False)
        self.k_proj = nn.Linear(dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(dim, kv_dim, bias=False)
        self.out_proj = nn.Linear(q_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        # QKNorm
        if use_qk_norm:
            self.qk_norm = QKNorm(self.head_dim)
        else:
            self.qk_norm = None

        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, T, D) agent tokens

        Returns:
            (B, T, D) attended agent tokens
        """
        B, T, D = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim)

        # Transpose: (B, heads, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply QKNorm if enabled
        if self.qk_norm is not None:
            q, k = self.qk_norm(q, k)

        # GQA: repeat KV heads
        if self.num_groups > 1:
            k = k.repeat_interleave(self.num_groups, dim=1)
            v = v.repeat_interleave(self.num_groups, dim=1)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Apply soft capping if enabled
        if self.soft_cap is not None:
            attn = soft_cap_attention(attn, self.soft_cap)

        attn = attn.masked_fill(self.causal_mask[:T, :T], float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, -1)
        out = self.out_proj(out)

        return out


class AgentTokenBlock(nn.Module):
    """Processing block for agent tokens.

    1. Cross-attention to z tokens (agent sees everything)
    2. Self-attention across time (causal)
    3. FFN

    Supports GQA, QKNorm, and soft capping.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        dropout: float = 0.0,
        max_seq_len: int = 256,
        use_qk_norm: bool = True,
        soft_cap: float | None = 50.0,
    ):
        """Initialize agent token block.

        Args:
            dim: Model dimension
            num_heads: Number of query heads
            num_kv_heads: Number of KV heads for GQA
            head_dim: Dimension per head
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
            use_qk_norm: Whether to use QK normalization
            soft_cap: Soft cap value for attention logits
        """
        super().__init__()

        self.norm1 = RMSNorm(dim)
        self.cross_attn = AgentCrossAttention(
            dim, num_heads, num_kv_heads, head_dim, dropout,
            use_qk_norm=use_qk_norm, soft_cap=soft_cap
        )

        self.norm2 = RMSNorm(dim)
        self.self_attn = AgentTemporalAttention(
            dim, num_heads, num_kv_heads, head_dim, dropout, max_seq_len,
            use_qk_norm=use_qk_norm, soft_cap=soft_cap
        )

        self.norm3 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, dropout=dropout)

    def forward(
        self,
        agent_tokens: torch.Tensor,
        z_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Process agent tokens.

        Args:
            agent_tokens: (B, T, D) agent tokens
            z_tokens: (B, T, S, D) spatial z tokens

        Returns:
            (B, T, D) processed agent tokens
        """
        # Cross-attention to z tokens
        agent_tokens = agent_tokens + self.cross_attn(self.norm1(agent_tokens), z_tokens)

        # Self-attention across time
        agent_tokens = agent_tokens + self.self_attn(self.norm2(agent_tokens))

        # FFN
        agent_tokens = agent_tokens + self.ffn(self.norm3(agent_tokens))

        return agent_tokens


class DynamicsTransformer(nn.Module):
    """Dynamics model for world model training.

    Predicts clean latent frames from noisy inputs using diffusion.
    Uses factorized attention with temporal attention every Nth layer.

    Architecture:
    - Input projection: latent tokens to model dim
    - Register tokens for improved information flow
    - Spatial + temporal position embeddings
    - Timestep embedding (diffusion)
    - Transformer blocks with factorized attention (GQA, QKNorm, soft capping)
    - Output projection back to latent dim

    Reference: DreamerV4 Section 3.2
    """

    def __init__(
        self,
        latent_dim: int = 256,
        spatial_size: int = 16,  # 16×16 = 256 spatial tokens
        model_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        temporal_every: int = 4,  # Temporal attention every N layers
        dropout: float = 0.0,
        max_seq_len: int = 256,  # Max frames in sequence
        # Stability and efficiency features
        use_qk_norm: bool = True,
        soft_cap: float | None = 50.0,
        # Register tokens
        num_register_tokens: int = 8,
        # Agent token settings (Phase 2+)
        use_agent_tokens: bool = False,
        num_tasks: int = 1,  # For multi-task conditioning
        agent_layers: int = 4,  # Number of agent token processing layers
        # Action conditioning
        use_actions: bool = False,
    ):
        """Initialize dynamics transformer.

        Args:
            latent_dim: Dimension of input latent tokens (from tokenizer)
            spatial_size: Size of spatial grid (16 for 16×16)
            model_dim: Model hidden dimension
            num_layers: Number of transformer blocks
            num_heads: Number of attention heads (query heads)
            num_kv_heads: Number of KV heads for GQA (None = same as num_heads, MHA)
            head_dim: Dimension per head (default: model_dim // num_heads)
            temporal_every: Add temporal attention every N layers
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
            use_qk_norm: Whether to use QK normalization for attention stability
            soft_cap: Soft cap value for attention logits (None = no capping)
            num_register_tokens: Number of register tokens (0 = disabled)
            use_agent_tokens: Enable agent tokens for Phase 2+
            num_tasks: Number of tasks for multi-task conditioning
            agent_layers: Number of agent token processing layers
            use_actions: Enable action conditioning with factorized embeddings
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.spatial_size = spatial_size
        self.spatial_tokens = spatial_size * spatial_size  # 256
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.temporal_every = temporal_every
        self.use_agent_tokens = use_agent_tokens
        self.use_actions = use_actions
        self.num_register_tokens = num_register_tokens
        self.use_qk_norm = use_qk_norm
        self.soft_cap = soft_cap

        # Total spatial tokens including registers
        self.total_spatial_tokens = self.spatial_tokens + num_register_tokens

        # Input projection: (B, T, C, H, W) -> (B, T, S, D)
        self.input_proj = nn.Linear(latent_dim, model_dim)

        # Register tokens (learnable, shared across all frames)
        if num_register_tokens > 0:
            self.register_tokens = nn.Parameter(
                torch.randn(1, 1, num_register_tokens, model_dim) * 0.02
            )
        else:
            self.register_tokens = None

        # Positional embeddings (for original spatial tokens only, registers have none)
        self.spatial_pos = nn.Parameter(
            torch.randn(1, 1, self.spatial_tokens, model_dim) * 0.02
        )
        self.temporal_pos = nn.Parameter(
            torch.randn(1, max_seq_len, 1, model_dim) * 0.02
        )

        # Timestep embedding for diffusion
        self.time_embed = TimestepEmbedding(model_dim)
        # Step size embedding for shortcut forcing
        self.step_embed = TimestepEmbedding(model_dim)

        # Factorized action embeddings
        if use_actions:
            self.action_embed = nn.ModuleDict({
                'movement': nn.Embedding(MOVEMENT_CLASSES, model_dim),
                **{k: nn.Embedding(2, model_dim) for k in ABILITY_KEYS}
            })
            # Learned "no action" embedding for unlabeled videos
            self.no_action_embed = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02)

        # Transformer blocks with GQA, QKNorm, and soft capping
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            # Temporal attention every temporal_every layers (on the last of each group)
            is_temporal = (i % temporal_every == temporal_every - 1)
            attn_type = "temporal" if is_temporal else "spatial"

            self.blocks.append(
                TransformerBlock(
                    dim=model_dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    attn_type=attn_type,
                    dropout=dropout,
                    max_seq_len=max_seq_len,
                    use_qk_norm=use_qk_norm,
                    soft_cap=soft_cap,
                )
            )

        # Output projection
        self.norm_out = RMSNorm(model_dim)
        self.output_proj = nn.Linear(model_dim, latent_dim)

        # Agent token components (Phase 2+)
        if use_agent_tokens:
            # Learnable agent token (one per frame, initialized from parameters)
            self.agent_token = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02)

            # Task embedding for multi-task conditioning
            self.task_embed = nn.Embedding(num_tasks, model_dim)

            # Agent token temporal position embedding
            self.agent_temporal_pos = nn.Parameter(
                torch.randn(1, max_seq_len, model_dim) * 0.02
            )

            # Agent token processing blocks with GQA, QKNorm, soft capping
            self.agent_blocks = nn.ModuleList([
                AgentTokenBlock(
                    dim=model_dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    max_seq_len=max_seq_len,
                    use_qk_norm=use_qk_norm,
                    soft_cap=soft_cap,
                )
                for _ in range(agent_layers)
            ])

            # Output normalization for agent tokens
            self.agent_norm_out = RMSNorm(model_dim)

        # Initialize weights
        self._init_weights()

    def embed_actions(self, actions: dict[str, torch.Tensor]) -> torch.Tensor:
        """Sum factorized action embeddings.

        Args:
            actions: Dict with keys 'movement' and ability keys.
                     Each value is a (B, T) tensor of class indices.

        Returns:
            (B, T, D) summed action embedding
        """
        # Start with movement embedding
        emb = self.action_embed['movement'](actions['movement'])  # (B, T, D)

        # Add all ability key embeddings
        for key in ABILITY_KEYS:
            emb = emb + self.action_embed[key](actions[key])

        return emb

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
        step_size: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        task_id: torch.Tensor | None = None,
        actions: dict[str, torch.Tensor] | None = None,
        independent_frames: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: predict clean latents from noisy input.

        Args:
            z_tau: Noisy latents, shape (B, T, C, H, W)
            tau: Diffusion timesteps, shape (B,) or (B, T)
            step_size: Optional step size for shortcut forcing, shape (B,)
                       Normalized to [0, 1] where 1.0 = k_max steps
            context: Optional context frames (not used in MVP)
            task_id: Optional task ID for multi-task conditioning, shape (B,)
            actions: Optional action dict with 'movement', 'target', and ability keys.
                     Each value is (B, T) tensor of class indices.
            independent_frames: If True, treat frames as independent (no temporal context).
                               Use with 30% probability during training to prevent
                               temporal shortcut learning (DreamerV4 Section 3.2).

        Returns:
            If use_agent_tokens=False:
                z_0_pred: Predicted clean latents, shape (B, T, C, H, W)
            If use_agent_tokens=True:
                tuple of:
                    z_0_pred: Predicted clean latents, shape (B, T, C, H, W)
                    agent_out: Agent token outputs, shape (B, T, D) for heads
        """
        B, T, C, H, W = z_tau.shape
        assert H == W == self.spatial_size, f"Expected {self.spatial_size}×{self.spatial_size}, got {H}×{W}"

        # Reshape to (B, T, S, C) where S = H*W
        x = z_tau.view(B, T, C, -1).permute(0, 1, 3, 2)  # (B, T, S, C)

        # Project to model dim
        x = self.input_proj(x)  # (B, T, S, D)

        # Add positional embeddings to spatial tokens
        x = x + self.spatial_pos[:, :, :self.spatial_tokens, :]
        x = x + self.temporal_pos[:, :T, :, :]

        # Add register tokens if enabled
        if self.register_tokens is not None:
            # Expand register tokens to match batch and time: (1, 1, R, D) -> (B, T, R, D)
            registers = self.register_tokens.expand(B, T, -1, -1)
            # Concatenate: (B, T, S+R, D)
            x = torch.cat([x, registers], dim=2)

        # Add action conditioning (broadcast to all spatial tokens including registers)
        if self.use_actions:
            if actions is not None:
                action_emb = self.embed_actions(actions)  # (B, T, D)
            else:
                action_emb = self.no_action_embed.expand(B, T, -1)
            x = x + action_emb.unsqueeze(2)  # (B, T, 1, D) broadcast to (B, T, S+R, D)

        # Get timestep embedding
        time_emb = self.time_embed(tau)  # (B, D) or (B, T, D)

        # Add step size embedding for shortcut forcing
        if step_size is not None:
            step_emb = self.step_embed(step_size)  # (B, D)
            # Handle broadcasting when tau is per-timestep (B, T)
            if time_emb.dim() == 3:  # (B, T, D)
                step_emb = step_emb.unsqueeze(1)  # (B, 1, D) for broadcasting
            time_emb = time_emb + step_emb  # additive combination

        # Transformer blocks (z tokens + registers - agent tokens processed separately)
        for block in self.blocks:
            x = block(x, time_emb, independent_frames=independent_frames)

        # Strip register tokens before output projection
        if self.register_tokens is not None:
            x_spatial = x[:, :, :self.spatial_tokens, :]  # (B, T, S, D)
        else:
            x_spatial = x

        # Output projection for z prediction
        z_out = self.norm_out(x_spatial)
        z_out = self.output_proj(z_out)  # (B, T, S, C)

        # Reshape back to (B, T, C, H, W)
        z_0_pred = z_out.permute(0, 1, 3, 2).view(B, T, C, H, W)

        # Process agent tokens if enabled
        if self.use_agent_tokens:
            # Initialize agent tokens: expand to (B, T, D)
            agent_tokens = self.agent_token.expand(B, T, -1).clone()

            # Add task embedding if provided
            if task_id is not None:
                task_emb = self.task_embed(task_id)  # (B, D)
                agent_tokens = agent_tokens + task_emb.unsqueeze(1)

            # Add temporal position embedding
            agent_tokens = agent_tokens + self.agent_temporal_pos[:, :T, :]

            # Process through agent blocks
            # Agent tokens attend to z tokens (x includes registers), but z tokens don't see agent tokens
            for agent_block in self.agent_blocks:
                agent_tokens = agent_block(agent_tokens, x)

            # Output normalization
            agent_out = self.agent_norm_out(agent_tokens)

            return z_0_pred, agent_out

        return z_0_pred

    def get_num_params(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_dynamics(
    size: str = "small",
    latent_dim: int = 256,
    use_agent_tokens: bool = False,
    num_tasks: int = 1,
    agent_layers: int = 4,
    use_actions: bool = False,
    # New DreamerV4 features
    use_qk_norm: bool = True,
    soft_cap: float | None = 50.0,
    num_register_tokens: int = 8,
    num_kv_heads: int | None = None,
) -> DynamicsTransformer:
    """Create dynamics model with preset sizes.

    Args:
        size: One of "tiny", "small", "medium", "large"
        latent_dim: Dimension of latent tokens (must match tokenizer)
        use_agent_tokens: Enable agent tokens for Phase 2+
        num_tasks: Number of tasks for multi-task conditioning
        agent_layers: Number of agent token processing layers
        use_actions: Enable action conditioning with factorized embeddings
        use_qk_norm: Whether to use QK normalization for attention stability
        soft_cap: Soft cap value for attention logits (None = no capping)
        num_register_tokens: Number of register tokens (0 = disabled)
        num_kv_heads: Number of KV heads for GQA (None = MHA, same as num_heads)

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
        use_agent_tokens=use_agent_tokens,
        num_tasks=num_tasks,
        agent_layers=agent_layers,
        use_actions=use_actions,
        use_qk_norm=use_qk_norm,
        soft_cap=soft_cap,
        num_register_tokens=num_register_tokens,
        num_kv_heads=num_kv_heads,
        **configs[size],
    )


if __name__ == "__main__":
    # Quick test
    print("Testing dynamics transformer...")

    model = create_dynamics("small", latent_dim=256)
    print(f"Parameters (base): {model.get_num_params():,}")
    print(f"  QKNorm: {model.use_qk_norm}")
    print(f"  Soft cap: {model.soft_cap}")
    print(f"  Register tokens: {model.num_register_tokens}")

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

    print("\n--- Testing independent frames mode (30% training) ---")
    z_pred_indep = model(z_tau, tau, independent_frames=True)
    print(f"Output shape (independent frames): {z_pred_indep.shape}")

    print("\n--- Testing with GQA (4 KV heads vs 8 Q heads) ---")
    model_gqa = create_dynamics("small", latent_dim=256, num_kv_heads=4)
    print(f"Parameters (GQA): {model_gqa.get_num_params():,}")
    z_pred_gqa = model_gqa(z_tau, tau)
    print(f"Output shape (GQA): {z_pred_gqa.shape}")

    print("\n--- Testing with actions ---")
    model_actions = create_dynamics("small", latent_dim=256, use_actions=True)
    print(f"Parameters (with actions): {model_actions.get_num_params():,}")

    # Create mock actions
    actions = {
        'movement': torch.randint(0, MOVEMENT_CLASSES, (B, T)),
        **{k: torch.randint(0, 2, (B, T)) for k in ABILITY_KEYS}
    }

    z_pred_actions = model_actions(z_tau, tau, actions=actions)
    print(f"Output shape (with actions): {z_pred_actions.shape}")

    # Test without actions (should use no_action_embed)
    z_pred_no_actions = model_actions(z_tau, tau, actions=None)
    print(f"Output shape (no actions): {z_pred_no_actions.shape}")

    print("\n--- Testing with agent tokens ---")
    model_agent = create_dynamics("small", latent_dim=256, use_agent_tokens=True)
    print(f"Parameters (with agent tokens): {model_agent.get_num_params():,}")

    z_pred_agent, agent_out = model_agent(z_tau, tau)
    print(f"Output z shape: {z_pred_agent.shape}")
    print(f"Agent output shape: {agent_out.shape}")

    print("\n--- Testing with both actions and agent tokens ---")
    model_both = create_dynamics("small", latent_dim=256, use_actions=True, use_agent_tokens=True)
    print(f"Parameters (both): {model_both.get_num_params():,}")

    z_pred_both, agent_both = model_both(z_tau, tau, actions=actions)
    print(f"Output z shape: {z_pred_both.shape}")
    print(f"Agent output shape: {agent_both.shape}")

    print("\n--- Testing with no register tokens ---")
    model_no_reg = create_dynamics("small", latent_dim=256, num_register_tokens=0)
    print(f"Parameters (no registers): {model_no_reg.get_num_params():,}")
    z_pred_no_reg = model_no_reg(z_tau, tau)
    print(f"Output shape (no registers): {z_pred_no_reg.shape}")

    print("\nAll tests passed!")
