"""Vision Tokenizer for frame compression.

Simple CNN autoencoder that compresses 256×256 frames to 16×16 latent space.
This is Phase 1 of the world model - learn to reconstruct frames before
adding temporal dynamics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual block with optional downsampling."""

    def __init__(self, in_channels: int, out_channels: int, downsample: bool = False):
        super().__init__()
        stride = 2 if downsample else 1

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        if downsample or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.gelu(out + identity)


class ResBlockUp(nn.Module):
    """Residual block with upsampling."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.skip = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.upsample(x)
        out = F.gelu(self.bn1(self.conv1(out)))
        out = self.bn2(self.conv2(out))
        return F.gelu(out + identity)


class Encoder(nn.Module):
    """CNN encoder: 256×256×3 → 16×16×latent_dim"""

    def __init__(self, latent_dim: int = 256, base_channels: int = 64):
        super().__init__()
        # 256×256×3 → 256×256×base
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, 7, stride=1, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
        )

        # Downsampling stages: 256 → 128 → 64 → 32 → 16
        self.stage1 = ResBlock(base_channels, base_channels * 2, downsample=True)  # 128
        self.stage2 = ResBlock(base_channels * 2, base_channels * 4, downsample=True)  # 64
        self.stage3 = ResBlock(base_channels * 4, base_channels * 8, downsample=True)  # 32
        self.stage4 = ResBlock(base_channels * 8, latent_dim, downsample=True)  # 16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to latent.

        Args:
            x: (B, 3, 256, 256) input image

        Returns:
            (B, latent_dim, 16, 16) latent representation
        """
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x


class Decoder(nn.Module):
    """CNN decoder: 16×16×latent_dim → 256×256×3"""

    def __init__(self, latent_dim: int = 256, base_channels: int = 64):
        super().__init__()
        # Upsampling stages: 16 → 32 → 64 → 128 → 256
        self.stage1 = ResBlockUp(latent_dim, base_channels * 8)  # 32
        self.stage2 = ResBlockUp(base_channels * 8, base_channels * 4)  # 64
        self.stage3 = ResBlockUp(base_channels * 4, base_channels * 2)  # 128
        self.stage4 = ResBlockUp(base_channels * 2, base_channels)  # 256

        # Final conv to RGB
        self.head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, 3, 3, padding=1),
            nn.Sigmoid(),  # Output in [0, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to image.

        Args:
            z: (B, latent_dim, 16, 16) latent representation

        Returns:
            (B, 3, 256, 256) reconstructed image
        """
        x = self.stage1(z)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.head(x)
        return x


class VisionTokenizer(nn.Module):
    """Vision Tokenizer: compress and reconstruct frames.

    Architecture:
    - Encoder: CNN with residual blocks, 256×256 → 16×16×latent_dim
    - Decoder: Transposed CNN with residual blocks, 16×16×latent_dim → 256×256

    The 16×16 spatial grid gives 256 "tokens" that can later be fed to
    the dynamics model transformer.
    """

    def __init__(self, latent_dim: int = 256, base_channels: int = 64):
        """Initialize tokenizer.

        Args:
            latent_dim: Number of channels in latent space
            base_channels: Base channel count (scaled up in deeper layers)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim, base_channels)
        self.decoder = Decoder(latent_dim, base_channels)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to latent representation."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to image."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> dict:
        """Forward pass: encode then decode.

        Args:
            x: (B, 3, H, W) input images, normalized to [0, 1]

        Returns:
            dict with:
                - reconstruction: (B, 3, H, W) reconstructed images
                - latent: (B, latent_dim, 16, 16) latent representation
        """
        z = self.encode(x)
        recon = self.decode(z)
        return {"reconstruction": recon, "latent": z}

    def get_num_params(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_tokenizer(
    size: str = "small",
) -> VisionTokenizer:
    """Create tokenizer with preset sizes.

    Args:
        size: One of "tiny", "small", "medium", "large"

    Returns:
        VisionTokenizer instance
    """
    configs = {
        "tiny": {"latent_dim": 128, "base_channels": 32},  # ~1.5M params
        "small": {"latent_dim": 256, "base_channels": 64},  # ~6M params
        "medium": {"latent_dim": 384, "base_channels": 96},  # ~14M params
        "large": {"latent_dim": 512, "base_channels": 128},  # ~25M params
    }

    if size not in configs:
        raise ValueError(f"Unknown size: {size}. Choose from {list(configs.keys())}")

    return VisionTokenizer(**configs[size])


if __name__ == "__main__":
    # Quick test
    model = create_tokenizer("small")
    print(f"Parameters: {model.get_num_params():,}")

    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Latent shape: {out['latent'].shape}")
    print(f"Reconstruction shape: {out['reconstruction'].shape}")
