"""Dynamics Transformer for world model.

Predicts future latent frames using diffusion with factorized attention.
Architecture follows DreamerV4:

- Factorized attention: spatial within frames, temporal across frames
- Temporal attention every 4th layer (efficiency optimization)
- X-prediction objective (predicts clean data directly)
- RMSNorm, SwiGLU, RoPE (2D spatial, 1D temporal)
- Action and conditioning as explicit sequence tokens (not broadcast/AdaLN)
- Agent tokens for policy/reward prediction (Phase 2+)

References:
- DreamerV4: "Training Agents Inside of Scalable World Models" (Hafner et al., 2025)
"""

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .layers import (
    RMSNorm, QKNorm, SwiGLU, soft_cap_attention,
    RotaryEmbedding1D, RotaryEmbedding2D, apply_rotary_emb,
)

# Action space constants (must match data/actions.py)
MOVEMENT_DIM = 2  # Continuous (x, y) in [0, 1]
MOVEMENT_CLASSES = 18  # Legacy, deprecated
ABILITY_KEYS = ['Q', 'W', 'E', 'R', 'D', 'F', 'item', 'B']


def _checkpoint_block_forward(block, x, independent_frames):
    """Wrapper for gradient checkpointing that passes independent_frames as keyword arg."""
    return block(x, independent_frames=independent_frames)


class SpatialAttention(nn.Module):
    """Self-attention within each frame with 2D RoPE.

    Attends over spatial tokens (latent + register + action + condition tokens)
    independently for each frame. RoPE is applied only to latent tokens that
    have 2D spatial positions.

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
        spatial_size: int = 16,
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
            spatial_size: Grid size for 2D RoPE (e.g. 16 for 16x16)
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_qk_norm = use_qk_norm
        self.soft_cap = soft_cap
        self.spatial_tokens = spatial_size * spatial_size  # Number of tokens that get RoPE

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

        # 2D RoPE for spatial positions (only applied to latent tokens)
        self.rope_2d = RotaryEmbedding2D(self.head_dim, spatial_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, T, S, D) where S = latent_tokens + register_tokens + action_token + cond_token
               The first self.spatial_tokens entries are latent tokens with 2D spatial structure.

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

        # Apply QKNorm BEFORE RoPE (paper: QKNorm + RoPE together)
        if self.qk_norm is not None:
            q, k = self.qk_norm(q, k)

        # Apply 2D RoPE to latent tokens only (first spatial_tokens positions)
        Nz = self.spatial_tokens
        if Nz > 0 and Nz <= S:
            cos, sin = self.rope_2d.get_rotary_emb(Nz, x.device)
            # cos, sin: (Nz, head_dim) -> (1, 1, Nz, head_dim)
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)

            # Apply RoPE only to latent token positions
            q_latent = apply_rotary_emb(q[:, :, :Nz, :], cos, sin)
            k_latent = apply_rotary_emb(k[:, :, :Nz, :], cos, sin)

            q = torch.cat([q_latent, q[:, :, Nz:, :]], dim=2)
            # For GQA, k has num_kv_heads which may differ from num_heads
            k = torch.cat([k_latent, k[:, :, Nz:, :]], dim=2)

        # GQA: repeat KV heads to match Q heads
        if self.num_groups > 1:
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
    """Causal self-attention across frames with 1D RoPE.

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

        # 1D RoPE for temporal positions
        self.rope_1d = RotaryEmbedding1D(self.head_dim, max_seq_len)

        # Pre-compute causal mask (True = masked out)
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

        # Apply QKNorm BEFORE RoPE
        if self.qk_norm is not None:
            q, k = self.qk_norm(q, k)

        # Apply 1D RoPE for temporal positions
        cos, sin = self.rope_1d.get_rotary_emb(T, x.device)
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

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
    """Standard pre-norm transformer block with either spatial or temporal attention.

    Uses RoPE instead of AdaLN for position/conditioning information.
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
        spatial_size: int = 16,
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
            spatial_size: Grid size for 2D spatial RoPE
        """
        super().__init__()
        self.attn_type = attn_type

        self.norm1 = RMSNorm(dim)
        if attn_type == "spatial":
            self.attn = SpatialAttention(
                dim, num_heads, num_kv_heads, head_dim, dropout,
                use_qk_norm=use_qk_norm, soft_cap=soft_cap,
                spatial_size=spatial_size,
            )
        else:
            self.attn = TemporalAttention(
                dim, num_heads, num_kv_heads, head_dim, dropout, max_seq_len,
                use_qk_norm=use_qk_norm, soft_cap=soft_cap
            )

        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        independent_frames: bool = False,
    ) -> torch.Tensor:
        """Forward pass (standard pre-norm transformer).

        Args:
            x: (B, T, S, D) features
            independent_frames: If True, treat frames as independent (temporal attention only)

        Returns:
            (B, T, S, D) transformed features
        """
        # Standard pre-norm transformer (no AdaLN)
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
        self.norm_kv = RMSNorm(dim)
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
        agent_tokens = agent_tokens + self.cross_attn(self.norm1(agent_tokens), self.norm_kv(z_tokens))

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
    - 2D RoPE for spatial attention, 1D RoPE for temporal attention
    - Action and conditioning (tau + step_size) as explicit sequence tokens
    - Transformer blocks with factorized attention (GQA, QKNorm, soft capping)
    - Output projection back to latent dim

    Spatial sequence per time step:
        [latent_tokens (with 2D RoPE), register_tokens (no RoPE),
         action_token (no RoPE), condition_token (no RoPE)]

    Reference: DreamerV4 Section 3.2
    """

    def __init__(
        self,
        latent_dim: int = 32,
        spatial_size: int = 16,  # 16x16 = 256 spatial tokens
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
        # Memory efficiency
        gradient_checkpointing: bool = False,
    ):
        """Initialize dynamics transformer.

        Args:
            latent_dim: Dimension of input latent tokens (from tokenizer)
            spatial_size: Size of spatial grid (16 for 16x16)
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
            gradient_checkpointing: Use gradient checkpointing to save memory (~2x reduction)
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
        self.gradient_checkpointing = gradient_checkpointing

        # Count extra tokens appended after latent+register tokens
        # Always have 1 conditioning token (tau + step_size)
        self.num_extra_tokens = 1  # condition token
        if use_actions:
            self.num_extra_tokens += 1  # action token

        # Total spatial tokens per time step: latent + register + action + condition
        self.total_spatial_tokens = self.spatial_tokens + num_register_tokens + self.num_extra_tokens

        # Input projection: (B, T, C, H, W) -> (B, T, S, D)
        self.input_proj = nn.Linear(latent_dim, model_dim)

        # Register tokens (learnable, shared across all frames)
        if num_register_tokens > 0:
            self.register_tokens = nn.Parameter(
                torch.randn(1, 1, num_register_tokens, model_dim) * 0.02
            )
        else:
            self.register_tokens = None

        # Discrete embeddings for tau and step_size (DreamerV4 uses lookup tables)
        # tau lives on a grid of k_max levels: {0, 1/k_max, ..., (k_max-1)/k_max}
        # step_size is power of 2: {1, 2, 4, ..., k_max} -> index by log2(d)
        self.k_max = 64
        self.num_tau_levels = self.k_max  # 64 discrete levels
        self.num_step_sizes = int(math.log2(self.k_max)) + 1  # 7: d=1,2,4,8,16,32,64
        self.tau_embed = nn.Embedding(self.num_tau_levels, model_dim)
        self.step_embed = nn.Embedding(self.num_step_sizes, model_dim)
        # Project concatenated [tau_emb, step_emb] to model_dim
        self.cond_proj = nn.Linear(model_dim * 2, model_dim)

        # Factorized action embeddings -> produces one action token per time step
        if use_actions:
            self.action_embed = nn.ModuleDict({
                'movement': nn.Linear(MOVEMENT_DIM, model_dim),  # continuous (x, y) -> D
                **{k: nn.Embedding(2, model_dim) for k in ABILITY_KEYS}
            })
            # Learned "no action" embedding for unlabeled videos
            self.no_action_embed = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02)

        # Transformer blocks with GQA, QKNorm, soft capping, and RoPE
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
                    spatial_size=spatial_size,
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
                     'movement' is (B, T, 2) float tensor of (x, y) coordinates.
                     Ability keys are (B, T) long tensors.

        Returns:
            (B, T, D) summed action embedding
        """
        # Movement: linear projection from continuous (x, y)
        emb = self.action_embed['movement'](actions['movement'])  # (B, T, 2) -> (B, T, D)

        # Add all ability key embeddings
        for key in ABILITY_KEYS:
            emb = emb + self.action_embed[key](actions[key])

        return emb

    def _build_condition_token(
        self,
        tau: torch.Tensor,
        step_size: torch.Tensor | None,
        B: int,
        T: int,
    ) -> torch.Tensor:
        """Build conditioning token from discrete tau and step_size embeddings.

        Args:
            tau: Diffusion timesteps, shape (B,) or (B, T), values in [0, 1)
                 Must be grid-aligned: multiples of 1/k_max
            step_size: Step sizes as integers {1, 2, 4, ..., k_max}, shape (B,) or (B, T).
                       None defaults to d=1 (finest).
            B: Batch size
            T: Sequence length

        Returns:
            (B, T, D) conditioning token
        """
        # Convert continuous tau to discrete index: tau * k_max -> integer in [0, k_max)
        tau_idx = (tau * self.k_max).long().clamp(0, self.num_tau_levels - 1)
        tau_emb = self.tau_embed(tau_idx)  # (B, D) or (B, T, D)

        # Convert step_size (integer power of 2) to index via log2
        if step_size is not None:
            step_idx = torch.log2(step_size.float()).long().clamp(0, self.num_step_sizes - 1)
        else:
            step_idx = torch.zeros(B, dtype=torch.long, device=tau.device)  # d=1 -> log2(1)=0
        step_emb = self.step_embed(step_idx)  # (B, D) or (B, T, D)

        # Expand to (B, T, D) if needed
        if tau_emb.dim() == 2:
            tau_emb = tau_emb.unsqueeze(1).expand(-1, T, -1)
        if step_emb.dim() == 2:
            step_emb = step_emb.unsqueeze(1).expand(-1, T, -1)

        # Concatenate and project: [tau_emb, step_emb] -> (B, T, D)
        cond = torch.cat([tau_emb, step_emb], dim=-1)  # (B, T, 2*D)
        cond = self.cond_proj(cond)  # (B, T, D)

        return cond

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
        task_id: torch.Tensor | None = None,
        actions: dict[str, torch.Tensor] | None = None,
        independent_frames: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: predict clean latents from noisy input.

        Args:
            z_tau: Noisy latents, shape (B, T, C, H, W)
            tau: Diffusion timesteps, shape (B,) or (B, T)
            step_size: Optional step size for shortcut forcing, shape (B,) or (B, T).
                       Integer powers of 2: {1, 2, 4, 8, 16, 32, 64}. None defaults to d=1.
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
        assert H == W == self.spatial_size, f"Expected {self.spatial_size}x{self.spatial_size}, got {H}x{W}"

        # Reshape to (B, T, S, C) where S = H*W
        x = z_tau.view(B, T, C, -1).permute(0, 1, 3, 2)  # (B, T, S, C)

        # Project to model dim
        x = self.input_proj(x)  # (B, T, S, D) where S = spatial_tokens

        # Add register tokens if enabled
        if self.register_tokens is not None:
            # Expand register tokens to match batch and time: (1, 1, R, D) -> (B, T, R, D)
            registers = self.register_tokens.expand(B, T, -1, -1)
            # Concatenate: (B, T, S+R, D)
            x = torch.cat([x, registers], dim=2)

        # Build action token and append to sequence
        if self.use_actions:
            if actions is not None:
                action_token = self.embed_actions(actions)  # (B, T, D)
            else:
                action_token = self.no_action_embed.expand(B, T, -1)
            # (B, T, D) -> (B, T, 1, D) and concatenate
            x = torch.cat([x, action_token.unsqueeze(2)], dim=2)

        # Build conditioning token (tau + step_size) and append to sequence
        cond_token = self._build_condition_token(tau, step_size, B, T)  # (B, T, D)
        x = torch.cat([x, cond_token.unsqueeze(2)], dim=2)

        # x is now (B, T, total_spatial_tokens, D)
        # Layout: [latent_tokens, register_tokens, action_token?, cond_token]

        # Transformer blocks
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(
                    _checkpoint_block_forward,
                    block, x, independent_frames,
                    use_reentrant=False,
                )
            else:
                x = block(x, independent_frames=independent_frames)

        # Strip extra tokens (register, action, condition) before output projection
        # Only keep latent tokens
        x_spatial = x[:, :, :self.spatial_tokens, :]  # (B, T, S, D)

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
            # Agent tokens attend to z tokens (x includes all tokens), but z tokens don't see agent tokens
            for agent_block in self.agent_blocks:
                if self.gradient_checkpointing and self.training:
                    agent_tokens = checkpoint(
                        agent_block,
                        agent_tokens, x,
                        use_reentrant=False,
                    )
                else:
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
    latent_dim: int = 32,
    use_agent_tokens: bool = False,
    num_tasks: int = 1,
    agent_layers: int = 4,
    use_actions: bool = False,
    # New DreamerV4 features
    use_qk_norm: bool = True,
    soft_cap: float | None = 50.0,
    num_register_tokens: int = 8,
    num_kv_heads: int | None = None,
    # Memory efficiency
    gradient_checkpointing: bool = False,
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
        gradient_checkpointing: Use gradient checkpointing to save memory (~2x reduction)

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
            "model_dim": 1024,
            "num_layers": 24,
            "num_heads": 16,
        },
    }

    if size not in configs:
        raise ValueError(f"Unknown size: {size}. Choose from {list(configs.keys())}")

    return DynamicsTransformer(
        latent_dim=latent_dim,
        spatial_size=16,  # 256 latent tokens = 16x16 spatial grid
        use_agent_tokens=use_agent_tokens,
        num_tasks=num_tasks,
        agent_layers=agent_layers,
        use_actions=use_actions,
        use_qk_norm=use_qk_norm,
        soft_cap=soft_cap,
        num_register_tokens=num_register_tokens,
        num_kv_heads=num_kv_heads,
        gradient_checkpointing=gradient_checkpointing,
        **configs[size],
    )


if __name__ == "__main__":
    # Quick test
    print("Testing dynamics transformer...")

    model = create_dynamics("small", latent_dim=32)
    print(f"Parameters (base): {model.get_num_params():,}")
    print(f"  QKNorm: {model.use_qk_norm}")
    print(f"  Soft cap: {model.soft_cap}")
    print(f"  Register tokens: {model.num_register_tokens}")

    # Test forward pass
    B, T, C, H, W = 2, 8, 32, 16, 16
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
    model_gqa = create_dynamics("small", latent_dim=32, num_kv_heads=4)
    print(f"Parameters (GQA): {model_gqa.get_num_params():,}")
    z_pred_gqa = model_gqa(z_tau, tau)
    print(f"Output shape (GQA): {z_pred_gqa.shape}")

    print("\n--- Testing with actions ---")
    model_actions = create_dynamics("small", latent_dim=32, use_actions=True)
    print(f"Parameters (with actions): {model_actions.get_num_params():,}")

    # Create mock actions
    actions = {
        'movement': torch.rand(B, T, MOVEMENT_DIM),  # continuous (x, y) in [0, 1]
        **{k: torch.randint(0, 2, (B, T)) for k in ABILITY_KEYS}
    }

    z_pred_actions = model_actions(z_tau, tau, actions=actions)
    print(f"Output shape (with actions): {z_pred_actions.shape}")

    # Test without actions (should use no_action_embed)
    z_pred_no_actions = model_actions(z_tau, tau, actions=None)
    print(f"Output shape (no actions): {z_pred_no_actions.shape}")

    print("\n--- Testing with agent tokens ---")
    model_agent = create_dynamics("small", latent_dim=32, use_agent_tokens=True)
    print(f"Parameters (with agent tokens): {model_agent.get_num_params():,}")

    z_pred_agent, agent_out = model_agent(z_tau, tau)
    print(f"Output z shape: {z_pred_agent.shape}")
    print(f"Agent output shape: {agent_out.shape}")

    print("\n--- Testing with both actions and agent tokens ---")
    model_both = create_dynamics("small", latent_dim=32, use_actions=True, use_agent_tokens=True)
    print(f"Parameters (both): {model_both.get_num_params():,}")

    z_pred_both, agent_both = model_both(z_tau, tau, actions=actions)
    print(f"Output z shape: {z_pred_both.shape}")
    print(f"Agent output shape: {agent_both.shape}")

    print("\n--- Testing with no register tokens ---")
    model_no_reg = create_dynamics("small", latent_dim=32, num_register_tokens=0)
    print(f"Parameters (no registers): {model_no_reg.get_num_params():,}")
    z_pred_no_reg = model_no_reg(z_tau, tau)
    print(f"Output shape (no registers): {z_pred_no_reg.shape}")

    print("\nAll tests passed!")
