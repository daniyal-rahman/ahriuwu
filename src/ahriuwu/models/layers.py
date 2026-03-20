"""Shared model layers used across dynamics and tokenizer models.

Contains: RMSNorm, QKNorm, SwiGLU, soft_cap_attention,
          RotaryEmbedding1D, RotaryEmbedding2D, apply_rotary_emb,
          Attention (unified spatial/temporal attention).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# flex_attention is optional — requires PyTorch 2.5+ with a CUDA backend.
# When unavailable or explicitly disabled, we fall back to manual matmul attention.
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    _FLEX_AVAILABLE = True
except ImportError:
    _FLEX_AVAILABLE = False


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


# ---------------------------------------------------------------------------
# flex_attention score_mod / mask_mod helpers (used only on the flex path)
# ---------------------------------------------------------------------------

def _soft_cap_score_mod(score, b, h, q_idx, kv_idx):
    """Soft cap attention logits at 50.0 (DreamerV4 paper value)."""
    return 50.0 * torch.tanh(score / 50.0)


def _causal_mask_mod(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def _diagonal_mask_mod(b, h, q_idx, kv_idx):
    return q_idx == kv_idx


# ---------------------------------------------------------------------------
# Unified Attention
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """Unified multi-head attention supporting spatial and temporal modes.

    Consolidates the attention implementations previously duplicated across
    dynamics.py (SpatialAttention, TemporalAttention) and
    transformer_tokenizer.py (MultiHeadAttention, TemporalAttentionTok).

    Features:
        - **Spatial mode** (mode="spatial"): Full (non-causal) self-attention
          with 2D RoPE applied to a subset of tokens (latent/patch tokens that
          have spatial positions).  Supports an explicit boolean mask and
          selective RoPE via ``rope_indices``.
        - **Temporal mode** (mode="temporal"): Causal self-attention across
          frames with 1D RoPE.  Supports ``independent_frames`` (diagonal mask)
          for the DreamerV4 "no temporal context" training mode.
        - **GQA** (grouped query attention) via ``num_kv_heads``.
        - **QKNorm** — when enabled the 1/sqrt(d) scaling is removed (Gemma 2
          convention: Q and K are already unit-RMS after normalization).
        - **Soft capping** of attention logits (Gemma 2, DreamerV4).
        - **flex_attention** backend for fused kernels when available.  Disabled
          automatically when ``allow_flex=False`` (e.g. inside gradient
          checkpointing where flex + GC causes OOM).

    Backend selection (flex vs manual):
        ``flex_attention`` compiles CUDA kernels and caches them.  When combined
        with gradient checkpointing the double-forward amplifies peak memory and
        can cause OOM on smaller GPUs.  The ``allow_flex`` flag (settable at
        init or at runtime via ``self.allow_flex``) controls this:

        * ``allow_flex=True``  (default for dynamics) — use flex when available.
        * ``allow_flex=False`` (default for tokenizer) — always use manual path.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        mode: str = "spatial",
        dropout: float = 0.0,
        use_qk_norm: bool = True,
        soft_cap: float | None = 50.0,
        # Spatial-mode options
        spatial_size: int = 16,
        # Temporal-mode options
        max_seq_len: int = 256,
        # Backend control
        allow_flex: bool = True,
    ):
        """Initialize unified attention.

        Args:
            dim: Model dimension.
            num_heads: Number of query heads.
            num_kv_heads: Number of KV heads for GQA (None = MHA).
            head_dim: Per-head dimension (default: dim // num_heads).
            mode: "spatial" or "temporal".
            dropout: Dropout probability.
            use_qk_norm: Apply QKNorm before RoPE.  When True the 1/sqrt(d)
                scaling is removed (Gemma 2 convention).
            soft_cap: Soft cap value for attention logits (None = disabled).
            spatial_size: Grid size for 2D RoPE (spatial mode only).
            max_seq_len: Maximum sequence length (temporal mode only).
            allow_flex: If True *and* flex_attention is importable, use the fused
                flex_attention kernel.  Set False to force the manual matmul path
                (required when running inside gradient checkpointing to avoid OOM).
        """
        super().__init__()
        assert mode in ("spatial", "temporal"), f"mode must be 'spatial' or 'temporal', got '{mode}'"

        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or dim // num_heads
        self.mode = mode
        self.use_qk_norm = use_qk_norm
        self.soft_cap = soft_cap
        self.allow_flex = allow_flex and _FLEX_AVAILABLE

        # When QKNorm is enabled, Q and K are already normalized to unit RMS,
        # so the 1/sqrt(d) scaling is redundant and compresses attention logits.
        # Following Gemma 2 convention, we set scale=1.0 with QKNorm.
        self.scale = 1.0 if use_qk_norm else self.head_dim ** -0.5

        # GQA validation
        assert self.num_heads % self.num_kv_heads == 0, (
            f"num_heads ({self.num_heads}) must be divisible by "
            f"num_kv_heads ({self.num_kv_heads})"
        )
        self.num_groups = self.num_heads // self.num_kv_heads

        # Projections
        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim

        self.q_proj = nn.Linear(dim, q_dim, bias=False)
        self.k_proj = nn.Linear(dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(dim, kv_dim, bias=False)
        self.out_proj = nn.Linear(q_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        # QKNorm
        self.qk_norm = QKNorm(self.head_dim) if use_qk_norm else None

        # Mode-specific components
        if mode == "spatial":
            self.spatial_tokens = spatial_size * spatial_size
            # spatial_size=0 disables RoPE (tokenizer without RoPE)
            self.rope = RotaryEmbedding2D(self.head_dim, spatial_size) if spatial_size > 0 else None
        else:  # temporal
            self.rope = RotaryEmbedding1D(self.head_dim, max_seq_len)
            # Precompute causal mask (True = masked out) for manual path
            self.register_buffer(
                "causal_mask",
                torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool(),
            )

    # ------------------------------------------------------------------
    # flex_attention path
    # ------------------------------------------------------------------

    def _forward_flex(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask_mod=None,
    ) -> torch.Tensor:
        """Attention via flex_attention (fused CUDA kernel).

        Args:
            q: (B, H_q, S, D)
            k: (B, H_kv, S, D)
            v: (B, H_kv, S, D)
            mask_mod: Optional flex mask_mod function for temporal masking.

        Returns:
            (B, H_q, S, D) attended values.
        """
        score_mod = _soft_cap_score_mod if self.soft_cap is not None else None

        block_mask = None
        if mask_mod is not None:
            S = q.shape[2]
            block_mask = create_block_mask(
                mask_mod, B=None, H=None, Q_LEN=S, KV_LEN=S, device=q.device,
            )

        return flex_attention(
            q, k, v,
            score_mod=score_mod,
            block_mask=block_mask,
            scale=self.scale,
            enable_gqa=(self.num_groups > 1),
        )

    # ------------------------------------------------------------------
    # Manual matmul path (fallback)
    # ------------------------------------------------------------------

    def _forward_manual(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attention via explicit matmul (no CUDA kernel compilation).

        Args:
            q: (B, H_q, S, D)
            k: (B, H_kv, S, D)
            v: (B, H_kv, S, D)
            mask: Optional boolean mask.  Shape (S, S) or (B, S, S) or
                (B, 1, S, S).  True = **can attend**, False = masked out.

        Returns:
            (B, H_q, S, D) attended values.
        """
        # Expand KV heads for GQA if needed
        if self.num_groups > 1:
            # (B, H_kv, S, D) -> (B, H_kv, G, S, D) -> (B, H_q, S, D)
            k = k.unsqueeze(2).expand(-1, -1, self.num_groups, -1, -1)
            k = k.reshape(k.shape[0], self.num_heads, k.shape[3], k.shape[4])
            v = v.unsqueeze(2).expand(-1, -1, self.num_groups, -1, -1)
            v = v.reshape(v.shape[0], self.num_heads, v.shape[3], v.shape[4])

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if self.soft_cap is not None:
            attn = soft_cap_attention(attn, self.soft_cap)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, S, S)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (B, 1, S, S)
            attn = attn.masked_fill(~mask, float('-inf'))

        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(v.dtype)
        attn = self.dropout(attn)
        return torch.matmul(attn, v)

    # ------------------------------------------------------------------
    # RoPE helpers
    # ------------------------------------------------------------------

    def _apply_rope_spatial_prefix(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply 2D RoPE to the first ``spatial_tokens`` positions only.

        Used by the dynamics model where the token layout is:
            [latent (spatial_tokens) | register | action | cond]
        and only the latent tokens have 2D grid positions.
        """
        Nz = self.spatial_tokens
        S = q.shape[2]
        if Nz <= 0 or Nz > S:
            return q, k

        cos, sin = self.rope.get_rotary_emb(Nz, device)
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, Nz, D)
        sin = sin.unsqueeze(0).unsqueeze(0)

        q_latent = apply_rotary_emb(q[:, :, :Nz, :], cos, sin)
        k_latent = apply_rotary_emb(k[:, :, :Nz, :], cos, sin)

        q = torch.cat([q_latent, q[:, :, Nz:, :]], dim=2)
        k = torch.cat([k_latent, k[:, :, Nz:, :]], dim=2)
        return q, k

    def _apply_rope_spatial_indexed(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        rope_indices: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply 2D RoPE selectively based on ``rope_indices``.

        Used by the tokenizer where patches get grid indices (>= 0) and
        latent query tokens get index -1 (no rotation).
        """
        cos, sin = self.rope.get_rotary_emb(
            self.rope.grid_size ** 2, device,
        )

        valid_mask = rope_indices >= 0  # (N,)
        if not valid_mask.any():
            return q, k

        safe_indices = rope_indices.clamp(min=0)
        cos_sel = cos[safe_indices].unsqueeze(0).unsqueeze(0)  # (1, 1, N, D)
        sin_sel = sin[safe_indices].unsqueeze(0).unsqueeze(0)

        q_rotated = apply_rotary_emb(q, cos_sel, sin_sel)
        k_rotated = apply_rotary_emb(k, cos_sel, sin_sel)

        valid_mask_exp = valid_mask.view(1, 1, -1, 1).expand_as(q)
        q = torch.where(valid_mask_exp, q_rotated, q)
        k = torch.where(valid_mask_exp, k_rotated, k)
        return q, k

    def _apply_rope_temporal(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        T: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply 1D RoPE for temporal positions 0..T-1."""
        cos, sin = self.rope.get_rotary_emb(T, device)
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        return q, k

    # ------------------------------------------------------------------
    # Public forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        rope_indices: torch.Tensor | None = None,
        independent_frames: bool = False,
    ) -> torch.Tensor:
        """Unified attention forward pass.

        **Spatial mode** expects ``x`` of shape ``(B, S, D)`` — a batch of
        per-frame token sequences.  The caller is responsible for reshaping
        ``(B, T, S, D) -> (B*T, S, D)`` before calling.

        When ``rope_indices`` is provided (tokenizer path), 2D RoPE is applied
        selectively to tokens whose index >= 0.  Otherwise 2D RoPE is applied
        to the first ``spatial_tokens`` positions (dynamics path).

        ``mask`` is an optional boolean attend mask — True means "can attend".
        Shape: ``(S, S)`` or ``(B, S, S)``.

        **Temporal mode** expects ``x`` of shape ``(B, T, D)`` — a batch of
        temporal token sequences.  The caller reshapes as needed (e.g.
        ``(B, T, S, D) -> (B*S, T, D)``).

        ``independent_frames`` replaces the causal mask with a diagonal mask
        (each frame only attends to itself).

        Args:
            x: Input tensor.  Shape depends on mode (see above).
            mask: Boolean attend mask for spatial mode (True = can attend).
            rope_indices: Per-token RoPE grid indices for spatial mode.
                Values >= 0 are grid positions; -1 means no rotation.
            independent_frames: Diagonal mask for temporal mode.

        Returns:
            Output tensor, same shape as ``x``.
        """
        if self.mode == "spatial":
            return self._forward_spatial(x, mask=mask, rope_indices=rope_indices)
        else:
            return self._forward_temporal(x, independent_frames=independent_frames)

    # ------------------------------------------------------------------
    # Spatial forward
    # ------------------------------------------------------------------

    def _forward_spatial(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        rope_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Spatial attention: (B, S, D) -> (B, S, D)."""
        B, S, D = x.shape

        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # QKNorm before RoPE
        if self.qk_norm is not None:
            q, k = self.qk_norm(q, k)

        # Apply 2D RoPE (skipped when self.rope is None, i.e. spatial_size=0)
        if self.rope is not None:
            if rope_indices is not None:
                q, k = self._apply_rope_spatial_indexed(q, k, rope_indices, x.device)
            else:
                q, k = self._apply_rope_spatial_prefix(q, k, x.device)

        # Cast to common dtype (RoPE may upcast to float32)
        q, k = q.to(v.dtype), k.to(v.dtype)

        # Dispatch to backend
        if self.allow_flex and mask is None:
            # flex_attention: no explicit mask needed for full spatial attention
            out = self._forward_flex(q, k, v)
        else:
            # Manual path (supports arbitrary boolean masks)
            out = self._forward_manual(q, k, v, mask=mask)

        out = out.transpose(1, 2).reshape(B, S, -1)
        return self.out_proj(out)

    # ------------------------------------------------------------------
    # Temporal forward
    # ------------------------------------------------------------------

    def _forward_temporal(
        self,
        x: torch.Tensor,
        independent_frames: bool = False,
    ) -> torch.Tensor:
        """Temporal attention: (B, T, D) -> (B, T, D)."""
        B, T, D = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # QKNorm before RoPE
        if self.qk_norm is not None:
            q, k = self.qk_norm(q, k)

        # 1D RoPE
        q, k = self._apply_rope_temporal(q, k, T, x.device)

        # Cast to common dtype
        q, k = q.to(v.dtype), k.to(v.dtype)

        # Dispatch to backend
        if self.allow_flex:
            mask_fn = _diagonal_mask_mod if independent_frames else _causal_mask_mod
            out = self._forward_flex(q, k, v, mask_mod=mask_fn)
        else:
            # Manual path: build causal or diagonal mask
            if independent_frames:
                mask = torch.eye(T, device=x.device, dtype=torch.bool)
            else:
                # causal_mask buffer stores True = masked *out*, invert for attend mask
                mask = ~self.causal_mask[:T, :T]
            out = self._forward_manual(q, k, v, mask=mask)

        out = out.transpose(1, 2).contiguous().reshape(B, T, -1)
        return self.out_proj(out)
