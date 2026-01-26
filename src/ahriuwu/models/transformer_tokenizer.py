"""Block-Causal Transformer Tokenizer for video frame compression.

Implements DreamerV4 Section 3.1 tokenizer architecture:
- Encoder: patches + latent tokens with block-causal attention
- Decoder: fresh patch queries + latents with block-causal attention
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

        # For each axis, we need cos and sin interleaved in pairs
        # RoPE rotation operates on consecutive pairs: [d0,d1], [d2,d3], ...
        # So we need: [cos(θ_0), cos(θ_0), cos(θ_1), cos(θ_1), ...]

        # Repeat each frequency for the pair
        freqs_y = freqs_y.repeat_interleave(2, dim=-1)  # (seq, dim/2)
        freqs_x = freqs_x.repeat_interleave(2, dim=-1)  # (seq, dim/2)

        # Combine y and x: [y_freqs, x_freqs]
        freqs = torch.cat([freqs_y, freqs_x], dim=-1)  # (seq, dim)

        return freqs.cos(), freqs.sin()

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate pairs of dimensions.

        For input [..., d0, d1, d2, d3, ...], returns [..., -d1, d0, -d3, d2, ...]
        This implements complex multiplication when combined with cos/sin.
        """
        # Split y and x halves to rotate them separately
        y_part = x[..., :self.dim_per_axis]
        x_part = x[..., self.dim_per_axis:]

        # Rotate each half's pairs: [a, b] -> [-b, a]
        def rotate_pairs(t):
            t = t.reshape(*t.shape[:-1], -1, 2)  # (..., dim/4, 2)
            t = torch.stack([-t[..., 1], t[..., 0]], dim=-1)  # swap and negate
            return t.reshape(*t.shape[:-2], -1)  # (..., dim/2)

        y_rotated = rotate_pairs(y_part)
        x_rotated = rotate_pairs(x_part)

        return torch.cat([y_rotated, x_rotated], dim=-1)

    def apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotary embedding to x."""
        return (x * cos) + (self.rotate_half(x) * sin)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class QKNorm(nn.Module):
    """Query-Key Normalization for attention stability.

    Normalizes Q and K independently before computing attention scores.
    Reference: Gemma 2, DreamerV4
    """

    def __init__(self, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.q_norm = RMSNorm(head_dim, eps)
        self.k_norm = RMSNorm(head_dim, eps)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q_norm(q), self.k_norm(k)


def soft_cap_attention(logits: torch.Tensor, cap: float = 50.0) -> torch.Tensor:
    """Apply soft capping to attention logits.

    Prevents extreme attention scores using tanh squashing.
    Reference: Gemma 2 uses cap=50.0
    """
    return cap * torch.tanh(logits / cap)


class SwiGLU(nn.Module):
    """SwiGLU activation function."""

    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 8 / 3)
        # Round to nearest multiple of 64 for efficiency
        hidden_dim = ((hidden_dim + 63) // 64) * 64

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


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
                q_rotated = rope.apply_rotary(q, cos_sel, sin_sel)
                k_rotated = rope.apply_rotary(k, cos_sel, sin_sel)

                # Only update valid tokens (patches), keep latents unchanged
                valid_mask_exp = valid_mask.view(1, 1, N, 1).expand_as(q)
                q = torch.where(valid_mask_exp, q_rotated, q)
                k = torch.where(valid_mask_exp, k_rotated, k)

        # Apply QKNorm if enabled
        if self.qk_norm is not None:
            q, k = self.qk_norm(q, k)

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

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm, QKNorm, and soft capping."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_qk_norm: bool = True,
        soft_cap: float | None = 50.0,
    ):
        """Initialize transformer block.

        Args:
            dim: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            use_qk_norm: Whether to use QK normalization
            soft_cap: Soft cap value for attention logits
        """
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout, use_qk_norm, soft_cap)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        rope: Optional[RotaryEmbedding2D] = None,
        rope_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask, rope, rope_indices)
        x = x + self.ffn(self.norm2(x))
        return x


def create_encoder_mask(
    num_patches: int,
    num_latents: int,
    num_frames: int,
    device: torch.device,
) -> torch.Tensor:
    """Create block-causal encoder attention mask.

    Encoder attention pattern:
    - Patches ONLY see same-frame patches (block diagonal for patches)
    - Latents see ALL tokens causally (patches + latents from current and past frames)

    Token layout per frame: [patches..., latents...]
    Total tokens: num_frames * (num_patches + num_latents)

    Returns:
        mask: (N, N) boolean tensor, True = can attend
    """
    tokens_per_frame = num_patches + num_latents
    total_tokens = num_frames * tokens_per_frame

    mask = torch.zeros(total_tokens, total_tokens, dtype=torch.bool, device=device)

    for t in range(num_frames):
        frame_start = t * tokens_per_frame
        patch_start = frame_start
        patch_end = frame_start + num_patches
        latent_start = frame_start + num_patches
        latent_end = frame_start + tokens_per_frame

        # Patches only see same-frame patches (block diagonal)
        mask[patch_start:patch_end, patch_start:patch_end] = True

        # Latents see all tokens from current and previous frames (causal)
        causal_end = latent_end  # up to and including current frame
        mask[latent_start:latent_end, :causal_end] = True

    return mask


def create_decoder_mask(
    num_patches: int,
    num_latents: int,
    num_frames: int,
    device: torch.device,
) -> torch.Tensor:
    """Create block-causal decoder attention mask.

    Decoder attention pattern:
    - Patches see own-frame patches + ALL latents (cross-frame via latents)
    - Latents ONLY see latents (no patches)

    Token layout per frame: [patches..., latents...]
    Total tokens: num_frames * (num_patches + num_latents)

    Returns:
        mask: (N, N) boolean tensor, True = can attend
    """
    tokens_per_frame = num_patches + num_latents
    total_tokens = num_frames * tokens_per_frame

    mask = torch.zeros(total_tokens, total_tokens, dtype=torch.bool, device=device)

    # First, let all latents see all latents (causally across frames)
    for t in range(num_frames):
        latent_start = t * tokens_per_frame + num_patches
        latent_end = latent_start + num_latents
        # Latents see all latents up to and including current frame
        for t2 in range(t + 1):
            src_latent_start = t2 * tokens_per_frame + num_patches
            src_latent_end = src_latent_start + num_latents
            mask[latent_start:latent_end, src_latent_start:src_latent_end] = True

    # Patches see own-frame patches + ALL latents
    for t in range(num_frames):
        patch_start = t * tokens_per_frame
        patch_end = patch_start + num_patches

        # Own-frame patches
        mask[patch_start:patch_end, patch_start:patch_end] = True

        # ALL latents from all frames
        for t2 in range(num_frames):
            src_latent_start = t2 * tokens_per_frame + num_patches
            src_latent_end = src_latent_start + num_latents
            mask[patch_start:patch_end, src_latent_start:src_latent_end] = True

    return mask


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
    """Block-causal transformer encoder.

    Takes patches, concatenates learned latent tokens, applies block-causal attention.
    Outputs latent tokens only (discards patches).
    Supports QKNorm and soft capping for stability.
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
    ):
        """Initialize transformer encoder.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            num_patches: Number of patch tokens per frame
            num_latents: Number of latent tokens per frame
            dropout: Dropout probability
            use_sincos_pos: Use sinusoidal position embeddings
            use_rope: Use rotary position embeddings
            use_qk_norm: Whether to use QK normalization
            soft_cap: Soft cap value for attention logits
            gradient_checkpointing: Use gradient checkpointing to save memory
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.num_latents = num_latents
        self.use_sincos_pos = use_sincos_pos
        self.use_rope = use_rope
        self.gradient_checkpointing = gradient_checkpointing
        self.grid_size = int(math.sqrt(num_patches))

        # Learned latent tokens (one set, repeated per frame)
        self.latent_tokens = nn.Parameter(torch.randn(1, num_latents, embed_dim) * 0.02)

        # RoPE for spatial position encoding in attention
        if use_rope:
            head_dim = embed_dim // num_heads
            self.rope = RotaryEmbedding2D(head_dim, self.grid_size)
            # No additive position embeddings for patches when using RoPE
            self.patch_pos_embed = None
        else:
            self.rope = None
            # Position embeddings for patches (additive)
            if use_sincos_pos:
                # Sinusoidal (fixed) - better spatial structure
                patch_pos = get_2d_sincos_pos_embed(embed_dim, self.grid_size)
                self.register_buffer('patch_pos_embed', patch_pos.unsqueeze(0))
            else:
                # Learned
                self.patch_pos_embed = nn.Parameter(
                    torch.randn(1, num_patches, embed_dim) * 0.02
                )

        # Latent position embeddings (always learned - no spatial structure)
        self.latent_pos_embed = nn.Parameter(
            torch.randn(1, num_latents, embed_dim) * 0.02
        )

        # Transformer blocks with QKNorm and soft capping
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout, use_qk_norm, soft_cap)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(embed_dim)

    def forward(
        self,
        patches: torch.Tensor,
        num_frames: int,
        mask_indices: Optional[torch.Tensor] = None,
        mask_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            patches: (B, T*num_patches, D) patch embeddings for all frames
            num_frames: T, number of frames
            mask_indices: (B, T*num_patches) boolean, True = masked (for MAE)
            mask_embed: (D,) learned mask embedding

        Returns:
            latents: (B, T*num_latents, D) latent tokens
        """
        B = patches.shape[0]
        device = patches.device

        # Reshape patches to per-frame
        patches = patches.reshape(B, num_frames, self.num_patches, self.embed_dim)

        # Apply MAE masking if provided
        if mask_indices is not None and mask_embed is not None:
            mask_indices = mask_indices.reshape(B, num_frames, self.num_patches)
            patches = patches.clone()
            patches[mask_indices] = mask_embed.to(patches.dtype)

        # Add position embeddings to patches (only if not using RoPE)
        if self.patch_pos_embed is not None:
            patches = patches + self.patch_pos_embed

        # Create latent tokens for each frame
        latents = self.latent_tokens.expand(B, -1, -1)  # (B, num_latents, D)
        latents = latents.unsqueeze(1).expand(-1, num_frames, -1, -1)  # (B, T, num_latents, D)
        latents = latents.contiguous()  # Make contiguous after expand
        latents = latents + self.latent_pos_embed

        # Interleave patches and latents per frame: [patches_t, latents_t, patches_t+1, ...]
        # Shape: (B, T, num_patches + num_latents, D)
        x = torch.cat([patches, latents], dim=2)
        x = x.reshape(B, num_frames * (self.num_patches + self.num_latents), self.embed_dim)

        # Create encoder mask
        mask = create_encoder_mask(
            self.num_patches, self.num_latents, num_frames, device
        )

        # Create RoPE indices if using RoPE
        # Token layout per frame: [patch_0, patch_1, ..., patch_255, latent_0, ..., latent_255]
        # Patches get their grid position (0-255), latents get -1 (no RoPE)
        rope_indices = None
        if self.use_rope:
            tokens_per_frame = self.num_patches + self.num_latents
            rope_indices = torch.zeros(num_frames * tokens_per_frame, dtype=torch.long, device=device)
            for t in range(num_frames):
                frame_start = t * tokens_per_frame
                # Patches: indices 0 to num_patches-1 (their grid position)
                rope_indices[frame_start:frame_start + self.num_patches] = torch.arange(
                    self.num_patches, device=device
                )
                # Latents: index -1 (no RoPE)
                rope_indices[frame_start + self.num_patches:frame_start + tokens_per_frame] = -1

        # Apply transformer blocks
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                # Use gradient checkpointing to save memory during training
                x = checkpoint(
                    block,
                    x, mask, self.rope, rope_indices,
                    use_reentrant=False,
                )
            else:
                x = block(x, mask, self.rope, rope_indices)

        x = self.norm(x)

        # Extract latents only
        x = x.reshape(B, num_frames, self.num_patches + self.num_latents, self.embed_dim)
        latents = x[:, :, self.num_patches:, :].contiguous()  # (B, T, num_latents, D)
        latents = latents.reshape(B, num_frames * self.num_latents, self.embed_dim)

        return latents


class TransformerDecoder(nn.Module):
    """Block-causal transformer decoder.

    Takes latent tokens, adds fresh learned patch queries, applies block-causal attention.
    Outputs reconstructed patches.
    Supports QKNorm and soft capping for stability.
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
    ):
        """Initialize transformer decoder.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            num_patches: Number of patch tokens per frame
            num_latents: Number of latent tokens per frame
            dropout: Dropout probability
            use_sincos_pos: Use sinusoidal position embeddings
            use_rope: Use rotary position embeddings
            use_qk_norm: Whether to use QK normalization
            soft_cap: Soft cap value for attention logits
            gradient_checkpointing: Use gradient checkpointing to save memory
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.num_latents = num_latents
        self.use_sincos_pos = use_sincos_pos
        self.use_rope = use_rope
        self.gradient_checkpointing = gradient_checkpointing
        self.grid_size = int(math.sqrt(num_patches))

        # Fresh learned patch queries (NOT from encoder)
        self.patch_queries = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)

        # RoPE for spatial position encoding in attention
        if use_rope:
            head_dim = embed_dim // num_heads
            self.rope = RotaryEmbedding2D(head_dim, self.grid_size)
            # No additive position embeddings for patches when using RoPE
            self.patch_pos_embed = None
        else:
            self.rope = None
            # Position embeddings for patches (additive)
            if use_sincos_pos:
                # Sinusoidal (fixed) - better spatial structure
                patch_pos = get_2d_sincos_pos_embed(embed_dim, self.grid_size)
                self.register_buffer('patch_pos_embed', patch_pos.unsqueeze(0))
            else:
                # Learned
                self.patch_pos_embed = nn.Parameter(
                    torch.randn(1, num_patches, embed_dim) * 0.02
                )

        # Latent position embeddings (always learned)
        self.latent_pos_embed = nn.Parameter(
            torch.randn(1, num_latents, embed_dim) * 0.02
        )

        # Transformer blocks with QKNorm and soft capping
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout, use_qk_norm, soft_cap)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(embed_dim)

    def forward(self, latents: torch.Tensor, num_frames: int) -> torch.Tensor:
        """
        Args:
            latents: (B, T*num_latents, D) latent tokens
            num_frames: T

        Returns:
            patches: (B, T*num_patches, D) reconstructed patch embeddings
        """
        B = latents.shape[0]
        device = latents.device

        # Reshape latents to per-frame
        latents = latents.reshape(B, num_frames, self.num_latents, self.embed_dim)
        latents = latents + self.latent_pos_embed

        # Create fresh patch queries for each frame
        patches = self.patch_queries.expand(B, -1, -1)  # (B, num_patches, D)
        patches = patches.unsqueeze(1).expand(-1, num_frames, -1, -1)  # (B, T, num_patches, D)
        patches = patches.contiguous()  # Make contiguous after expand

        # Add position embeddings to patches (only if not using RoPE)
        if self.patch_pos_embed is not None:
            patches = patches + self.patch_pos_embed

        # Interleave patches and latents
        x = torch.cat([patches, latents], dim=2)
        x = x.reshape(B, num_frames * (self.num_patches + self.num_latents), self.embed_dim)

        # Create decoder mask
        mask = create_decoder_mask(
            self.num_patches, self.num_latents, num_frames, device
        )

        # Create RoPE indices if using RoPE
        # Token layout per frame: [patch_0, ..., patch_255, latent_0, ..., latent_255]
        # Patches get their grid position (0-255), latents get -1 (no RoPE)
        rope_indices = None
        if self.use_rope:
            tokens_per_frame = self.num_patches + self.num_latents
            rope_indices = torch.zeros(num_frames * tokens_per_frame, dtype=torch.long, device=device)
            for t in range(num_frames):
                frame_start = t * tokens_per_frame
                # Patches: indices 0 to num_patches-1 (their grid position)
                rope_indices[frame_start:frame_start + self.num_patches] = torch.arange(
                    self.num_patches, device=device
                )
                # Latents: index -1 (no RoPE)
                rope_indices[frame_start + self.num_patches:frame_start + tokens_per_frame] = -1

        # Apply transformer blocks
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                # Use gradient checkpointing to save memory during training
                x = checkpoint(
                    block,
                    x, mask, self.rope, rope_indices,
                    use_reentrant=False,
                )
            else:
                x = block(x, mask, self.rope, rope_indices)

        x = self.norm(x)

        # Extract patches only
        x = x.reshape(B, num_frames, self.num_patches + self.num_latents, self.embed_dim)
        patches = x[:, :, :self.num_patches, :].contiguous()  # (B, T, num_patches, D)
        patches = patches.reshape(B, num_frames * self.num_patches, self.embed_dim)

        return patches


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
    ):
        """Initialize transformer tokenizer.

        Args:
            img_size: Input image size (square)
            patch_size: Size of each patch
            embed_dim: Embedding dimension
            latent_dim: Bottleneck dimension per token
            num_heads: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            num_latents: Number of latent tokens per frame
            dropout: Dropout probability
            use_sincos_pos: Use sinusoidal position embeddings
            use_rope: Use rotary position embeddings
            use_qk_norm: Whether to use QK normalization
            soft_cap: Soft cap value for attention logits
            gradient_checkpointing: Use gradient checkpointing to save memory
        """
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

        # Encoder and decoder with QKNorm, soft capping, and gradient checkpointing
        self.encoder = TransformerEncoder(
            embed_dim, num_heads, num_encoder_layers,
            self.num_patches, num_latents, dropout, use_sincos_pos, use_rope,
            use_qk_norm, soft_cap, gradient_checkpointing
        )
        self.decoder = TransformerDecoder(
            embed_dim, num_heads, num_decoder_layers,
            self.num_patches, num_latents, dropout, use_sincos_pos, use_rope,
            use_qk_norm, soft_cap, gradient_checkpointing
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

    def encode(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.0,
    ) -> dict:
        """Encode frames to latent tokens.

        Args:
            x: (B, T, C, H, W) or (B, C, H, W) input frames in [0, 1]
            mask_ratio: fraction of patches to mask (for MAE training)

        Returns:
            dict with:
                - latent: (B, T*num_latents, latent_dim) bottlenecked latents
                - mask_indices: (B, T*num_patches) boolean mask (if mask_ratio > 0)
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

        # MAE masking
        mask_indices = None
        if mask_ratio > 0:
            mask_indices = torch.rand(B, T * self.num_patches, device=x.device) < mask_ratio

        # Encode
        latents = self.encoder(
            patches, T,
            mask_indices=mask_indices,
            mask_embed=self.mask_embed if mask_ratio > 0 else None,
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
    ) -> dict:
        """Forward pass: encode then decode.

        Args:
            x: (B, T, C, H, W) or (B, C, H, W) input frames in [0, 1]
            mask_ratio: MAE mask ratio (0 = no masking, train mode uses ~0.75)

        Returns:
            dict with:
                - reconstruction: (B, T, C, H, W) reconstructed frames
                - latent: (B, T*num_latents, latent_dim) bottlenecked latents
                - mask_indices: (B, T*num_patches) if mask_ratio > 0
        """
        # Handle single frame
        single_frame = x.dim() == 4
        if single_frame:
            x = x.unsqueeze(1)

        B, T = x.shape[:2]

        # Encode
        enc_out = self.encode(x, mask_ratio)
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
    use_rope: bool = False,
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
            "num_encoder_layers": 6,
            "num_decoder_layers": 6,
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
