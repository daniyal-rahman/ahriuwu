"""Factored Space-Time Transformer Tokenizer for video frame compression.

Implements DreamerV4 Section 3.1 tokenizer with factored space-time attention:
- Spatial attention: (B*T, N, D) per-frame, 740×740 mask
- Temporal attention: (B*N, T, D) causal across frames with 1D RoPE, 16×16
- Alternating s/t blocks every layer (s, t, s, t, ...)
- Bottleneck: linear → tanh → reshape (512×D → 256×32)
- MAE training with p ~ U(0, 0.9)
- Loss: MSE + 0.2 * LPIPS
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .layers import (
    RMSNorm, QKNorm, SwiGLU, soft_cap_attention,
    RotaryEmbedding1D, RotaryEmbedding2D, apply_rotary_emb,
)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    """Generate 2D sinusoidal position embeddings.

    Args:
        embed_dim: Embedding dimension (must be divisible by 4)
        grid_size: Size of the grid (e.g., 16 for 16x16 patches)

    Returns:
        (grid_size*grid_size, embed_dim) position embeddings
    """
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2D sin/cos"

    # Create grid
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_h, grid_w, indexing='ij')
    grid = torch.stack(grid, dim=0)  # (2, H, W)
    grid = grid.reshape(2, 1, grid_size, grid_size)

    # Sinusoidal embeddings
    embed_dim_per_axis = embed_dim // 2
    omega = torch.arange(embed_dim_per_axis // 2, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / (embed_dim_per_axis // 2)))

    # Height embeddings
    pos_h = grid[0].reshape(-1)  # (H*W,)
    out_h = torch.outer(pos_h, omega)  # (H*W, D/4)
    emb_h = torch.cat([torch.sin(out_h), torch.cos(out_h)], dim=1)  # (H*W, D/2)

    # Width embeddings
    pos_w = grid[1].reshape(-1)  # (H*W,)
    out_w = torch.outer(pos_w, omega)  # (H*W, D/4)
    emb_w = torch.cat([torch.sin(out_w), torch.cos(out_w)], dim=1)  # (H*W, D/2)

    # Concatenate height and width embeddings
    pos_embed = torch.cat([emb_h, emb_w], dim=1)  # (H*W, D)

    return pos_embed


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional masking, RoPE, QKNorm, and soft capping."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_qk_norm: bool = True,
        soft_cap: float | None = 50.0,
    ):
        """Initialize multi-head attention.

        Args:
            dim: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            use_qk_norm: Whether to use QK normalization
            soft_cap: Soft cap value for attention logits (None = no capping)
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_qk_norm = use_qk_norm
        self.soft_cap = soft_cap

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        # QKNorm
        if use_qk_norm:
            self.qk_norm = QKNorm(self.head_dim)
        else:
            self.qk_norm = None

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        rope: Optional[RotaryEmbedding2D] = None,
        rope_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, D = x.shape

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply QKNorm before RoPE (normalization must not undo rotations)
        if self.qk_norm is not None:
            q, k = self.qk_norm(q, k)

        # Apply RoPE if provided (only to tokens with valid indices >= 0)
        # Latent tokens use index -1 and don't get RoPE rotation
        if rope is not None and rope_indices is not None:
            cos, sin = rope.get_rotary_emb(rope.grid_size ** 2, x.device)

            # Create mask for valid positions (patches have indices 0 to grid_size^2-1)
            valid_mask = rope_indices >= 0  # (N,)

            if valid_mask.any():
                # Safe indexing: clamp to valid range for indexing, mask handles selection
                safe_indices = rope_indices.clamp(min=0)
                cos_sel = cos[safe_indices]  # (N, head_dim)
                sin_sel = sin[safe_indices]  # (N, head_dim)

                # Expand for broadcasting: (1, 1, N, head_dim)
                cos_sel = cos_sel.unsqueeze(0).unsqueeze(0)
                sin_sel = sin_sel.unsqueeze(0).unsqueeze(0)

                # Apply rotation
                q_rotated = apply_rotary_emb(q, cos_sel, sin_sel)
                k_rotated = apply_rotary_emb(k, cos_sel, sin_sel)

                # Only update valid tokens (patches), keep latents unchanged
                valid_mask_exp = valid_mask.view(1, 1, N, 1).expand_as(q)
                q = torch.where(valid_mask_exp, q_rotated, q)
                k = torch.where(valid_mask_exp, k_rotated, k)

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply soft capping if enabled
        if self.soft_cap is not None:
            attn = soft_cap_attention(attn, self.soft_cap)

        if mask is not None:
            # mask: (N, N) or (B, N, N), True = attend, False = mask
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (B, 1, N, N)
            attn = attn.masked_fill(~mask, float('-inf'))

        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(v.dtype)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)


class TemporalAttentionTok(nn.Module):
    """Temporal attention with 1D RoPE and causal masking.

    Operates on (B_eff, T, D) where B_eff = B*N (each spatial token attends across time).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_qk_norm: bool = True,
        soft_cap: float | None = 50.0,
        max_seq_len: int = 256,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.soft_cap = soft_cap

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        if use_qk_norm:
            self.qk_norm = QKNorm(self.head_dim)
        else:
            self.qk_norm = None

        self.rope = RotaryEmbedding1D(self.head_dim, max_seq_len)

        # Precompute causal mask: True = can attend
        causal = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer('causal_mask', causal)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if self.qk_norm is not None:
            q, k = self.qk_norm(q, k)

        # Apply 1D RoPE
        cos, sin = self.rope.get_rotary_emb(T, x.device)
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if self.soft_cap is not None:
            attn = soft_cap_attention(attn, self.soft_cap)

        # Apply causal mask
        mask = self.causal_mask[:T, :T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
        attn = attn.masked_fill(~mask, float('-inf'))

        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(v.dtype)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


class SpaceTimeTransformerBlock(nn.Module):
    """Factored space-time transformer block.

    attn_type="spatial": reshape to (B*T, N, D), apply MultiHeadAttention with space mask + 2D RoPE
    attn_type="temporal": reshape to (B*N, T, D), apply TemporalAttentionTok with causal + 1D RoPE
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_qk_norm: bool = True,
        soft_cap: float | None = 50.0,
        attn_type: str = "spatial",
        max_time: int = 256,
    ):
        super().__init__()
        self.attn_type = attn_type
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim)

        if attn_type == "spatial":
            self.attn = MultiHeadAttention(dim, num_heads, dropout, use_qk_norm, soft_cap)
        else:
            self.attn = TemporalAttentionTok(dim, num_heads, dropout, use_qk_norm, soft_cap, max_time)

    def forward(
        self,
        x: torch.Tensor,
        T: int,
        space_mask: Optional[torch.Tensor] = None,
        rope: Optional[RotaryEmbedding2D] = None,
        rope_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, D)
            T: number of frames
            space_mask: (N, N) for spatial attention
            rope: 2D RoPE for spatial attention
            rope_indices: (N,) patch grid indices for spatial RoPE
        """
        B, _, N, D = x.shape

        if self.attn_type == "spatial":
            x_s = x.reshape(B * T, N, D)
            x_s = x_s + self.attn(self.norm1(x_s), mask=space_mask, rope=rope, rope_indices=rope_indices)
            x_s = x_s + self.ffn(self.norm2(x_s))
            return x_s.reshape(B, T, N, D)
        else:
            x_t = x.permute(0, 2, 1, 3).reshape(B * N, T, D)
            x_t = x_t + self.attn(self.norm1(x_t))
            x_t = x_t + self.ffn(self.norm2(x_t))
            return x_t.reshape(B, N, T, D).permute(0, 2, 1, 3).contiguous()


class PatchEmbed(nn.Module):
    """Convert image to patch embeddings."""

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 512,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # 256 for 256/16

        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) images
        Returns:
            (B, num_patches, embed_dim) patch embeddings
        """
        x = self.proj(x)  # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, D)
        return x


class TransformerEncoder(nn.Module):
    """Factored space-time transformer encoder.

    Takes patches, concatenates learned latent tokens, applies alternating
    spatial/temporal attention blocks. Outputs latent tokens only.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        num_patches: int = 256,
        num_latents: int = 256,
        dropout: float = 0.0,
        use_sincos_pos: bool = True,
        use_rope: bool = False,
        use_qk_norm: bool = True,
        soft_cap: float | None = 50.0,
        gradient_checkpointing: bool = False,
        max_time: int = 256,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.num_latents = num_latents
        self.use_sincos_pos = use_sincos_pos
        self.use_rope = use_rope
        self.gradient_checkpointing = gradient_checkpointing
        self.grid_size = int(math.sqrt(num_patches))
        N = num_patches + num_latents

        # Learned latent tokens (one set, repeated per frame)
        self.latent_tokens = nn.Parameter(torch.randn(1, num_latents, embed_dim) * 0.02)

        # RoPE for spatial position encoding in attention
        if use_rope:
            head_dim = embed_dim // num_heads
            self.rope = RotaryEmbedding2D(head_dim, self.grid_size)
            self.patch_pos_embed = None
        else:
            self.rope = None
            if use_sincos_pos:
                patch_pos = get_2d_sincos_pos_embed(embed_dim, self.grid_size)
                self.register_buffer('patch_pos_embed', patch_pos.unsqueeze(0))
            else:
                self.patch_pos_embed = nn.Parameter(
                    torch.randn(1, num_patches, embed_dim) * 0.02
                )

        # Latent position embeddings (always learned)
        self.latent_pos_embed = nn.Parameter(
            torch.randn(1, num_latents, embed_dim) * 0.02
        )

        # Alternating spatial/temporal blocks (every 2nd = temporal)
        self.blocks = nn.ModuleList([
            SpaceTimeTransformerBlock(
                embed_dim, num_heads, dropout, use_qk_norm, soft_cap,
                attn_type="temporal" if i % 2 == 1 else "spatial",
                max_time=max_time,
            )
            for i in range(num_layers)
        ])
        self.norm = RMSNorm(embed_dim)

        # Precompute encoder space mask (N, N): patches see patches, latents see all
        P = num_patches
        enc_mask = torch.zeros(N, N, dtype=torch.bool)
        enc_mask[:P, :P] = True    # patches see patches
        enc_mask[P:, :] = True     # latents see all
        self.register_buffer('space_mask', enc_mask)

        # Precompute RoPE indices for spatial attention: (N,)
        # patches get grid position 0..P-1, latents get -1
        if use_rope:
            rope_idx = torch.zeros(N, dtype=torch.long)
            rope_idx[:P] = torch.arange(P)
            rope_idx[P:] = -1
            self.register_buffer('rope_indices', rope_idx)
        else:
            self.rope_indices = None

    def forward(
        self,
        patches: torch.Tensor,
        num_frames: int,
        mask_indices: Optional[torch.Tensor] = None,
        mask_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = patches.shape[0]
        T = num_frames

        # Reshape patches to per-frame: (B, T, P, D)
        patches = patches.reshape(B, T, self.num_patches, self.embed_dim)

        # Apply MAE masking if provided
        if mask_indices is not None and mask_embed is not None:
            mask_indices = mask_indices.reshape(B, T, self.num_patches)
            patches = patches.clone()
            patches[mask_indices] = mask_embed.to(patches.dtype)

        # Add position embeddings to patches (only if not using RoPE)
        if self.patch_pos_embed is not None:
            patches = patches + self.patch_pos_embed

        # Create latent tokens for each frame: (B, T, L, D)
        latents = self.latent_tokens.expand(B, -1, -1).unsqueeze(1).expand(-1, T, -1, -1).contiguous()
        latents = latents + self.latent_pos_embed

        # Build (B, T, N, D) from patches + latents
        x = torch.cat([patches, latents], dim=2)

        # Apply transformer blocks — each handles its own reshape
        gc = self.gradient_checkpointing
        for block in self.blocks:
            if gc and self.training:
                x = checkpoint(
                    block, x, T, self.space_mask, self.rope, self.rope_indices,
                    use_reentrant=False,
                )
            else:
                x = block(x, T, space_mask=self.space_mask, rope=self.rope, rope_indices=self.rope_indices)

        x = self.norm(x.reshape(B, T * (self.num_patches + self.num_latents), self.embed_dim))
        x = x.reshape(B, T, self.num_patches + self.num_latents, self.embed_dim)

        # Extract latents: (B, T, L, D) → (B, T*L, D)
        latents = x[:, :, self.num_patches:, :].contiguous()
        return latents.reshape(B, T * self.num_latents, self.embed_dim)


class TransformerDecoder(nn.Module):
    """Factored space-time transformer decoder.

    Takes latent tokens, adds fresh learned patch queries, applies alternating
    spatial/temporal attention blocks. Outputs reconstructed patches.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        num_patches: int = 256,
        num_latents: int = 256,
        dropout: float = 0.0,
        use_sincos_pos: bool = True,
        use_rope: bool = False,
        use_qk_norm: bool = True,
        soft_cap: float | None = 50.0,
        gradient_checkpointing: bool = False,
        max_time: int = 256,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.num_latents = num_latents
        self.use_sincos_pos = use_sincos_pos
        self.use_rope = use_rope
        self.gradient_checkpointing = gradient_checkpointing
        self.grid_size = int(math.sqrt(num_patches))
        N = num_patches + num_latents

        # Fresh learned patch queries (NOT from encoder)
        self.patch_queries = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)

        # RoPE for spatial position encoding in attention
        if use_rope:
            head_dim = embed_dim // num_heads
            self.rope = RotaryEmbedding2D(head_dim, self.grid_size)
            self.patch_pos_embed = None
        else:
            self.rope = None
            if use_sincos_pos:
                patch_pos = get_2d_sincos_pos_embed(embed_dim, self.grid_size)
                self.register_buffer('patch_pos_embed', patch_pos.unsqueeze(0))
            else:
                self.patch_pos_embed = nn.Parameter(
                    torch.randn(1, num_patches, embed_dim) * 0.02
                )

        # Latent position embeddings (always learned)
        self.latent_pos_embed = nn.Parameter(
            torch.randn(1, num_latents, embed_dim) * 0.02
        )

        # Alternating spatial/temporal blocks (every 2nd = temporal)
        self.blocks = nn.ModuleList([
            SpaceTimeTransformerBlock(
                embed_dim, num_heads, dropout, use_qk_norm, soft_cap,
                attn_type="temporal" if i % 2 == 1 else "spatial",
                max_time=max_time,
            )
            for i in range(num_layers)
        ])
        self.norm = RMSNorm(embed_dim)

        # Precompute decoder space mask (N, N): patches see all, latents see latents only
        P = num_patches
        dec_mask = torch.zeros(N, N, dtype=torch.bool)
        dec_mask[:P, :] = True     # patches see all (patches + latents)
        dec_mask[P:, P:] = True    # latents see latents only
        self.register_buffer('space_mask', dec_mask)

        # Precompute RoPE indices for spatial attention: (N,)
        if use_rope:
            rope_idx = torch.zeros(N, dtype=torch.long)
            rope_idx[:P] = torch.arange(P)
            rope_idx[P:] = -1
            self.register_buffer('rope_indices', rope_idx)
        else:
            self.rope_indices = None

    def forward(self, latents: torch.Tensor, num_frames: int) -> torch.Tensor:
        B = latents.shape[0]
        T = num_frames

        # Reshape latents to per-frame: (B, T, L, D)
        latents = latents.reshape(B, T, self.num_latents, self.embed_dim)
        latents = latents + self.latent_pos_embed

        # Create fresh patch queries for each frame: (B, T, P, D)
        patches = self.patch_queries.expand(B, -1, -1).unsqueeze(1).expand(-1, T, -1, -1).contiguous()

        # Add position embeddings to patches (only if not using RoPE)
        if self.patch_pos_embed is not None:
            patches = patches + self.patch_pos_embed

        # Build (B, T, N, D) from patches + latents
        x = torch.cat([patches, latents], dim=2)

        # Apply transformer blocks — each handles its own reshape
        gc = self.gradient_checkpointing
        for block in self.blocks:
            if gc and self.training:
                x = checkpoint(
                    block, x, T, self.space_mask, self.rope, self.rope_indices,
                    use_reentrant=False,
                )
            else:
                x = block(x, T, space_mask=self.space_mask, rope=self.rope, rope_indices=self.rope_indices)

        x = self.norm(x.reshape(B, T * (self.num_patches + self.num_latents), self.embed_dim))
        x = x.reshape(B, T, self.num_patches + self.num_latents, self.embed_dim)

        # Extract patches: (B, T, P, D) → (B, T*P, D)
        patches = x[:, :, :self.num_patches, :].contiguous()
        return patches.reshape(B, T * self.num_patches, self.embed_dim)


class Bottleneck(nn.Module):
    """Bottleneck: per-token linear → tanh.

    Compresses each latent token from embed_dim to latent_dim.
    Paper: each of 256 tokens projected from D → latent_dim, then tanh.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_latents_in: int = 256,
        num_latents_out: int = 256,
        latent_dim: int = 32,
    ):
        super().__init__()
        self.num_latents_in = num_latents_in
        self.num_latents_out = num_latents_out
        self.latent_dim = latent_dim

        # Per-token projection: embed_dim -> latent_dim
        # Applied to each latent token independently
        self.proj = nn.Linear(embed_dim, latent_dim)

    def forward(self, x: torch.Tensor, num_frames: int) -> torch.Tensor:
        """
        Args:
            x: (B, T*num_latents_in, embed_dim)
            num_frames: T

        Returns:
            (B, T*num_latents_out, latent_dim) bottlenecked latents
        """
        B = x.shape[0]

        # Reshape to (B, T, num_latents_in, embed_dim)
        x = x.reshape(B, num_frames, self.num_latents_in, -1)

        # Per-token projection and tanh
        x = self.proj(x)  # (B, T, num_latents, latent_dim)
        x = torch.tanh(x)

        # Flatten temporal
        x = x.reshape(B, num_frames * self.num_latents_out, self.latent_dim)

        return x


class BottleneckInverse(nn.Module):
    """Inverse bottleneck: per-token linear.

    Expands each latent token from latent_dim back to embed_dim.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_latents_in: int = 256,  # from bottleneck
        num_latents_out: int = 256,  # to decoder
        latent_dim: int = 32,
    ):
        super().__init__()
        self.num_latents_in = num_latents_in
        self.num_latents_out = num_latents_out
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim

        # Per-token projection: latent_dim -> embed_dim
        self.proj = nn.Linear(latent_dim, embed_dim)

    def forward(self, x: torch.Tensor, num_frames: int) -> torch.Tensor:
        """
        Args:
            x: (B, T*num_latents_in, latent_dim) bottlenecked latents
            num_frames: T

        Returns:
            (B, T*num_latents_out, embed_dim) expanded latents for decoder
        """
        B = x.shape[0]

        # Reshape to (B, T, num_latents_in, latent_dim)
        x = x.reshape(B, num_frames, self.num_latents_in, self.latent_dim)

        # Per-token projection
        x = self.proj(x)  # (B, T, num_latents, embed_dim)

        # Flatten temporal
        x = x.reshape(B, num_frames * self.num_latents_out, self.embed_dim)

        return x


class PatchUnembed(nn.Module):
    """Convert patch embeddings back to image."""

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        out_channels: int = 3,
        embed_dim: int = 512,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches_side = img_size // patch_size

        # Project to patch pixels
        self.proj = nn.Linear(embed_dim, patch_size * patch_size * out_channels)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_patches, embed_dim)
        Returns:
            (B, C, H, W) reconstructed image
        """
        B = x.shape[0]

        x = self.proj(x)  # (B, num_patches, P*P*C)

        # Reshape to image
        x = x.view(
            B,
            self.num_patches_side,
            self.num_patches_side,
            self.patch_size,
            self.patch_size,
            self.out_channels
        )
        # (B, H_patches, W_patches, P, P, C) -> (B, C, H, W)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B, self.out_channels, self.img_size, self.img_size)

        return x


class TransformerTokenizer(nn.Module):
    """Block-causal transformer tokenizer for video frames.

    Architecture (DreamerV4 Section 3.1):
    - Encoder: patches + latent tokens with block-causal attention
    - Bottleneck: linear → tanh → reshape
    - Decoder: fresh patch queries + latents with block-causal attention
    - Output: reconstructed frames

    Supports QKNorm and soft capping for training stability.

    For dynamics model integration:
    - encode() returns bottlenecked latents (256 tokens × 32 dims per frame)
    - These feed directly into dynamics model
    """

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        embed_dim: int = 512,
        latent_dim: int = 32,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_latents: int = 256,
        dropout: float = 0.0,
        use_sincos_pos: bool = True,
        use_rope: bool = False,
        use_qk_norm: bool = True,
        soft_cap: float | None = 50.0,
        gradient_checkpointing: bool = False,
        max_time: int = 256,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_patches = (img_size // patch_size) ** 2
        self.num_latents = num_latents
        self.use_qk_norm = use_qk_norm
        self.soft_cap = soft_cap
        self.gradient_checkpointing = gradient_checkpointing

        # Patch embedding and unembedding
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        self.patch_unembed = PatchUnembed(img_size, patch_size, 3, embed_dim)

        # Encoder and decoder with factored space-time attention
        self.encoder = TransformerEncoder(
            embed_dim, num_heads, num_encoder_layers,
            self.num_patches, num_latents, dropout, use_sincos_pos, use_rope,
            use_qk_norm, soft_cap, gradient_checkpointing, max_time,
        )
        self.decoder = TransformerDecoder(
            embed_dim, num_heads, num_decoder_layers,
            self.num_patches, num_latents, dropout, use_sincos_pos, use_rope,
            use_qk_norm, soft_cap, gradient_checkpointing, max_time,
        )

        # Bottleneck (encoder latents → compact form for dynamics)
        self.bottleneck = Bottleneck(
            embed_dim, num_latents, num_latents, latent_dim
        )
        self.bottleneck_inv = BottleneckInverse(
            embed_dim, num_latents, num_latents, latent_dim
        )

        # MAE mask embedding
        self.mask_embed = nn.Parameter(torch.randn(embed_dim) * 0.02)

        # Output activation
        self.output_act = nn.Sigmoid()

    def make_mask(
        self,
        B: int,
        T: int,
        mask_ratio: float,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Generate MAE mask outside of torch.compile boundary.

        Args:
            B: batch size
            T: sequence length
            mask_ratio: fraction of patches to mask
            device: tensor device

        Returns:
            (B, T*num_patches) boolean mask or None if mask_ratio == 0
        """
        num_masked = round(mask_ratio * self.num_patches)
        if num_masked == 0:
            return None
        mask_indices = torch.zeros(
            B, T, self.num_patches, dtype=torch.bool, device=device
        )
        for b in range(B):
            for t in range(T):
                perm = torch.randperm(self.num_patches, device=device)
                mask_indices[b, t, perm[:num_masked]] = True
        return mask_indices.reshape(B, T * self.num_patches)

    def encode(
        self,
        x: torch.Tensor,
        mask_indices: torch.Tensor | None = None,
    ) -> dict:
        """Encode frames to latent tokens.

        Args:
            x: (B, T, C, H, W) or (B, C, H, W) input frames in [0, 1]
            mask_indices: (B, T*num_patches) boolean mask from make_mask(), or None

        Returns:
            dict with:
                - latent: (B, T*num_latents, latent_dim) bottlenecked latents
                - mask_indices: passed through if provided
        """
        # Handle single frame
        if x.dim() == 4:
            x = x.unsqueeze(1)

        B, T, C, H, W = x.shape

        # Flatten batch and temporal
        x_flat = x.reshape(B * T, C, H, W)

        # Patch embed
        patches = self.patch_embed(x_flat)  # (B*T, num_patches, D)
        patches = patches.reshape(B, T * self.num_patches, self.embed_dim)

        # Encode
        latents = self.encoder(
            patches, T,
            mask_indices=mask_indices,
            mask_embed=self.mask_embed if mask_indices is not None else None,
        )

        # Bottleneck
        latents = self.bottleneck(latents, T)

        result = {"latent": latents}
        if mask_indices is not None:
            result["mask_indices"] = mask_indices

        return result

    def decode(self, latents: torch.Tensor, num_frames: int) -> torch.Tensor:
        """Decode latent tokens to frames.

        Args:
            latents: (B, T*num_latents, latent_dim) bottlenecked latents
            num_frames: T

        Returns:
            (B, T, C, H, W) reconstructed frames in [0, 1]
        """
        B = latents.shape[0]

        # Inverse bottleneck
        latents = self.bottleneck_inv(latents, num_frames)

        # Decode
        patches = self.decoder(latents, num_frames)

        # Unembed patches to images
        patches = patches.reshape(B * num_frames, self.num_patches, self.embed_dim)
        recon = self.patch_unembed(patches)  # (B*T, C, H, W)
        recon = self.output_act(recon)

        recon = recon.reshape(B, num_frames, 3, self.img_size, self.img_size)

        return recon

    def forward(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.0,
        mask_indices: torch.Tensor | None = None,
    ) -> dict:
        """Forward pass: encode then decode.

        Args:
            x: (B, T, C, H, W) or (B, C, H, W) input frames in [0, 1]
            mask_ratio: MAE mask ratio — used only if mask_indices is None
            mask_indices: pre-computed mask from make_mask() (preferred)

        Returns:
            dict with:
                - reconstruction: (B, T, C, H, W) reconstructed frames
                - latent: (B, T*num_latents, latent_dim) bottlenecked latents
                - mask_indices: (B, T*num_patches) if masking was applied
        """
        # Handle single frame
        single_frame = x.dim() == 4
        if single_frame:
            x = x.unsqueeze(1)

        B, T = x.shape[:2]

        # Generate mask outside compiled boundary if not provided
        if mask_indices is None and mask_ratio > 0:
            mask_indices = self.make_mask(B, T, mask_ratio, x.device)

        # Encode
        enc_out = self.encode(x, mask_indices)
        latents = enc_out["latent"]

        # Decode
        recon = self.decode(latents, T)

        if single_frame:
            recon = recon.squeeze(1)

        result = {
            "reconstruction": recon,
            "latent": latents,
        }
        if "mask_indices" in enc_out:
            result["mask_indices"] = enc_out["mask_indices"]

        return result

    def get_num_params(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_transformer_tokenizer(
    size: str = "small",
    use_rope: bool = True,
    use_qk_norm: bool = True,
    soft_cap: float | None = 50.0,
    gradient_checkpointing: bool = False,
) -> TransformerTokenizer:
    """Create transformer tokenizer with preset sizes.

    Args:
        size: One of "tiny", "small", "medium", "large"
        use_rope: If True, use RoPE for position encoding instead of additive embeddings
        use_qk_norm: Whether to use QK normalization for attention stability
        soft_cap: Soft cap value for attention logits (None = no capping)
        gradient_checkpointing: Use gradient checkpointing to save memory (~2x reduction)

    Returns:
        TransformerTokenizer instance
    """
    configs = {
        "tiny": {
            "embed_dim": 256,
            "latent_dim": 16,
            "num_heads": 4,
            "num_encoder_layers": 4,
            "num_decoder_layers": 4,
            "num_latents": 256,
        },
        "small": {
            "embed_dim": 512,
            "latent_dim": 32,
            "num_heads": 8,
            "num_encoder_layers": 8,
            "num_decoder_layers": 8,
            "num_latents": 256,
        },
        "medium": {
            "embed_dim": 768,
            "latent_dim": 48,
            "num_heads": 12,
            "num_encoder_layers": 8,
            "num_decoder_layers": 8,
            "num_latents": 256,
        },
        "large": {
            "embed_dim": 1024,
            "latent_dim": 64,
            "num_heads": 16,
            "num_encoder_layers": 12,
            "num_decoder_layers": 12,
            "num_latents": 256,
        },
    }

    if size not in configs:
        raise ValueError(f"Unknown size: {size}. Choose from {list(configs.keys())}")

    return TransformerTokenizer(
        img_size=352,  # Match dataset TARGET_SIZE=(352, 352)
        **configs[size],
        use_rope=use_rope,
        use_qk_norm=use_qk_norm,
        soft_cap=soft_cap,
        gradient_checkpointing=gradient_checkpointing,
    )


if __name__ == "__main__":
    # Quick test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_transformer_tokenizer("tiny").to(device)
    print(f"Parameters: {model.get_num_params():,}")

    # Test single frame
    x = torch.randn(2, 3, 256, 256, device=device)
    out = model(x)
    print(f"Single frame input: {x.shape}")
    print(f"Single frame latent: {out['latent'].shape}")
    print(f"Single frame recon: {out['reconstruction'].shape}")

    # Test video (multiple frames)
    x = torch.randn(2, 4, 3, 256, 256, device=device)
    out = model(x, mask_ratio=0.75)
    print(f"\nVideo input: {x.shape}")
    print(f"Video latent: {out['latent'].shape}")
    print(f"Video recon: {out['reconstruction'].shape}")
    print(f"Mask indices: {out['mask_indices'].shape}")
    print(f"Masked patches: {out['mask_indices'].sum().item()} / {out['mask_indices'].numel()}")
