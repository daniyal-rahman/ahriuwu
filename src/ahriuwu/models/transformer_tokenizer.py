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


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


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
    """Multi-head attention with optional masking."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, D = x.shape

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

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
    """Transformer block with pre-norm."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
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
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        num_patches: int = 256,
        num_latents: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.num_latents = num_latents

        # Learned latent tokens (one set, repeated per frame)
        self.latent_tokens = nn.Parameter(torch.randn(1, num_latents, embed_dim) * 0.02)

        # Position embeddings for patches and latents
        self.patch_pos_embed = nn.Parameter(
            torch.randn(1, num_patches, embed_dim) * 0.02
        )
        self.latent_pos_embed = nn.Parameter(
            torch.randn(1, num_latents, embed_dim) * 0.02
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout)
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

        # Add position embeddings to patches
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

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)

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
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        num_patches: int = 256,
        num_latents: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.num_latents = num_latents

        # Fresh learned patch queries (NOT from encoder)
        self.patch_queries = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)

        # Position embeddings
        self.patch_pos_embed = nn.Parameter(
            torch.randn(1, num_patches, embed_dim) * 0.02
        )
        self.latent_pos_embed = nn.Parameter(
            torch.randn(1, num_latents, embed_dim) * 0.02
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout)
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
        patches = patches + self.patch_pos_embed

        # Interleave patches and latents
        x = torch.cat([patches, latents], dim=2)
        x = x.reshape(B, num_frames * (self.num_patches + self.num_latents), self.embed_dim)

        # Create decoder mask
        mask = create_decoder_mask(
            self.num_patches, self.num_latents, num_frames, device
        )

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        x = self.norm(x)

        # Extract patches only
        x = x.reshape(B, num_frames, self.num_patches + self.num_latents, self.embed_dim)
        patches = x[:, :, :self.num_patches, :].contiguous()  # (B, T, num_patches, D)
        patches = patches.reshape(B, num_frames * self.num_patches, self.embed_dim)

        return patches


class Bottleneck(nn.Module):
    """Bottleneck: linear → tanh → reshape.

    Compresses 512 latent tokens with D dims to 256 latent tokens with latent_dim dims.
    Paper: 512×D → linear → tanh → reshape to 256×32
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

        # Project each latent token to smaller dim, then reshape
        # Input: (B, T, num_latents_in, embed_dim)
        # We project to total of num_latents_out * latent_dim values
        # Then reshape to (B, T, num_latents_out, latent_dim)
        total_in = num_latents_in * embed_dim
        total_out = num_latents_out * latent_dim

        self.proj = nn.Linear(total_in, total_out)

    def forward(self, x: torch.Tensor, num_frames: int) -> torch.Tensor:
        """
        Args:
            x: (B, T*num_latents_in, embed_dim)
            num_frames: T

        Returns:
            (B, T*num_latents_out, latent_dim) bottlenecked latents
        """
        B = x.shape[0]

        # Reshape to (B, T, num_latents_in * embed_dim)
        x = x.reshape(B, num_frames, -1)

        # Project and apply tanh
        x = self.proj(x)  # (B, T, num_latents_out * latent_dim)
        x = torch.tanh(x)

        # Reshape to (B, T, num_latents_out, latent_dim)
        x = x.reshape(B, num_frames, self.num_latents_out, self.latent_dim)

        # Flatten temporal
        x = x.reshape(B, num_frames * self.num_latents_out, self.latent_dim)

        return x


class BottleneckInverse(nn.Module):
    """Inverse bottleneck: reshape → linear.

    Expands 256 latent tokens with latent_dim to encoder's format.
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

        total_in = num_latents_in * latent_dim
        total_out = num_latents_out * embed_dim

        self.proj = nn.Linear(total_in, total_out)

    def forward(self, x: torch.Tensor, num_frames: int) -> torch.Tensor:
        """
        Args:
            x: (B, T*num_latents_in, latent_dim) bottlenecked latents
            num_frames: T

        Returns:
            (B, T*num_latents_out, embed_dim) expanded latents for decoder
        """
        B = x.shape[0]

        # Reshape to (B, T, num_latents_in * latent_dim)
        x = x.reshape(B, num_frames, -1)

        # Project
        x = self.proj(x)  # (B, T, num_latents_out * embed_dim)

        # Reshape to (B, T, num_latents_out, embed_dim)
        x = x.reshape(B, num_frames, self.num_latents_out, self.embed_dim)

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
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_patches = (img_size // patch_size) ** 2
        self.num_latents = num_latents

        # Patch embedding and unembedding
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        self.patch_unembed = PatchUnembed(img_size, patch_size, 3, embed_dim)

        # Encoder and decoder
        self.encoder = TransformerEncoder(
            embed_dim, num_heads, num_encoder_layers,
            self.num_patches, num_latents, dropout
        )
        self.decoder = TransformerDecoder(
            embed_dim, num_heads, num_decoder_layers,
            self.num_patches, num_latents, dropout
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


def create_transformer_tokenizer(size: str = "small") -> TransformerTokenizer:
    """Create transformer tokenizer with preset sizes.

    Args:
        size: One of "tiny", "small", "medium", "large"

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

    return TransformerTokenizer(**configs[size])


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
