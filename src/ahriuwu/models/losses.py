"""Loss functions for tokenizer training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class VGGPerceptualLoss(nn.Module):
    """Perceptual loss using VGG16 features.

    Compares feature activations at multiple layers for perceptually
    meaningful similarity (less blurry reconstructions than pure MSE).
    """

    def __init__(self, layer_weights: dict | None = None):
        """Initialize VGG perceptual loss.

        Args:
            layer_weights: Dict mapping layer names to weights.
                          Default uses conv1_2, conv2_2, conv3_3, conv4_3.
        """
        super().__init__()

        # Load pretrained VGG16
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        # Extract feature layers
        self.blocks = nn.ModuleList()
        self.layer_names = []

        # Layer indices for VGG16 features
        # conv1_2: 0-3, conv2_2: 4-8, conv3_3: 9-15, conv4_3: 16-22
        block_indices = [
            (0, 4, "conv1_2"),
            (4, 9, "conv2_2"),
            (9, 16, "conv3_3"),
            (16, 23, "conv4_3"),
        ]

        prev_end = 0
        for start, end, name in block_indices:
            block = nn.Sequential(*list(vgg.features.children())[prev_end:end])
            self.blocks.append(block)
            self.layer_names.append(name)
            prev_end = end

        # Freeze VGG weights
        for param in self.parameters():
            param.requires_grad = False

        # Layer weights
        if layer_weights is None:
            # Default: equal weight, scaled by layer depth
            layer_weights = {
                "conv1_2": 1.0,
                "conv2_2": 1.0,
                "conv3_3": 1.0,
                "conv4_3": 1.0,
            }
        self.layer_weights = layer_weights

        # ImageNet normalization (VGG expects this)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input for VGG (assumes input in [0, 1])."""
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss.

        Args:
            pred: Predicted images (B, 3, H, W) in [0, 1]
            target: Target images (B, 3, H, W) in [0, 1]

        Returns:
            Scalar perceptual loss
        """
        pred = self.normalize(pred)
        target = self.normalize(target)

        loss = 0.0
        pred_feat = pred
        target_feat = target

        for block, name in zip(self.blocks, self.layer_names):
            pred_feat = block(pred_feat)
            target_feat = block(target_feat)

            # L1 loss on features (more robust than L2)
            layer_loss = F.l1_loss(pred_feat, target_feat)
            loss = loss + self.layer_weights.get(name, 1.0) * layer_loss

        return loss


class TokenizerLoss(nn.Module):
    """Combined loss for tokenizer training.

    Loss = mse_weight * MSE + lpips_weight * LPIPS
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        lpips_weight: float = 0.1,
    ):
        """Initialize combined loss.

        Args:
            mse_weight: Weight for MSE reconstruction loss
            lpips_weight: Weight for perceptual loss
        """
        super().__init__()
        self.mse_weight = mse_weight
        self.lpips_weight = lpips_weight
        self.lpips = VGGPerceptualLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> dict:
        """Compute combined loss.

        Args:
            pred: Predicted images (B, 3, H, W) in [0, 1]
            target: Target images (B, 3, H, W) in [0, 1]

        Returns:
            Dict with total loss and components
        """
        mse_loss = F.mse_loss(pred, target)
        lpips_loss = self.lpips(pred, target)

        total_loss = self.mse_weight * mse_loss + self.lpips_weight * lpips_loss

        return {
            "loss": total_loss,
            "mse": mse_loss,
            "lpips": lpips_loss,
        }


def psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute Peak Signal-to-Noise Ratio.

    Args:
        pred: Predicted images (B, C, H, W) in [0, 1]
        target: Target images (B, C, H, W) in [0, 1]

    Returns:
        PSNR in dB (higher is better, >25 is decent, >30 is good)
    """
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return torch.tensor(float("inf"))
    return 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse)


if __name__ == "__main__":
    # Quick test
    loss_fn = TokenizerLoss()

    pred = torch.rand(2, 3, 256, 256)
    target = torch.rand(2, 3, 256, 256)

    losses = loss_fn(pred, target)
    print(f"Total loss: {losses['loss']:.4f}")
    print(f"MSE loss: {losses['mse']:.4f}")
    print(f"LPIPS loss: {losses['lpips']:.4f}")
    print(f"PSNR: {psnr(pred, target):.2f} dB")
