#!/usr/bin/env python3
"""Evaluate trained tokenizer on test frames.

Computes metrics (PSNR, MSE) and generates visual comparisons.

Usage:
    python scripts/eval_tokenizer.py --checkpoint checkpoints/tokenizer_best.pt
    python scripts/eval_tokenizer.py --checkpoint checkpoints/tokenizer_best.pt --num-samples 16
    python scripts/eval_tokenizer.py --checkpoint checkpoints/tokenizer_best.pt --plot-history
"""

import argparse
import json
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from ahriuwu.data.dataset import SingleFrameDataset
from ahriuwu.models import create_tokenizer, psnr


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate vision tokenizer")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--frames-dir",
        type=str,
        default="data/frames",
        help="Directory containing video subdirs with frames",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Number of sample reconstructions to generate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for metric computation",
    )
    parser.add_argument(
        "--plot-history",
        action="store_true",
        help="Plot training history from checkpoint directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        help="Device to run evaluation on",
    )
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: str):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Get model size from checkpoint args
    args = checkpoint.get("args", {})
    model_size = args.get("model_size", "small")

    model = create_tokenizer(model_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, checkpoint


def compute_metrics(model, dataloader, device: str, max_batches: int = None):
    """Compute metrics over dataset.

    Args:
        model: Tokenizer model
        dataloader: DataLoader for test data
        device: Device to run on
        max_batches: Max batches to evaluate (None = all)

    Returns:
        Dict with metrics
    """
    total_psnr = 0.0
    total_mse = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break

            frames = batch["frame"].to(device)

            output = model(frames)
            recon = output["reconstruction"]

            # PSNR
            batch_psnr = psnr(recon, frames).item()
            total_psnr += batch_psnr * frames.shape[0]

            # MSE
            batch_mse = torch.nn.functional.mse_loss(recon, frames).item()
            total_mse += batch_mse * frames.shape[0]

            num_samples += frames.shape[0]

            if (batch_idx + 1) % 10 == 0:
                print(f"  Evaluated {num_samples} samples...")

    return {
        "psnr": total_psnr / num_samples,
        "mse": total_mse / num_samples,
        "num_samples": num_samples,
    }


def generate_comparison_grid(
    model,
    dataloader,
    device: str,
    num_samples: int,
    output_path: Path,
):
    """Generate side-by-side comparison of original vs reconstruction.

    Args:
        model: Tokenizer model
        dataloader: DataLoader
        device: Device
        num_samples: Number of samples to include
        output_path: Path to save image
    """
    # Collect samples
    samples = []
    with torch.no_grad():
        for batch in dataloader:
            frames = batch["frame"].to(device)
            output = model(frames)
            recon = output["reconstruction"]

            for i in range(frames.shape[0]):
                if len(samples) >= num_samples:
                    break
                samples.append({
                    "original": frames[i].cpu(),
                    "recon": recon[i].cpu(),
                })

            if len(samples) >= num_samples:
                break

    if not samples:
        print("No samples found!")
        return

    # Create grid: rows of [original | reconstruction]
    sample_h, sample_w = 256, 256
    padding = 4

    # 2 columns (original, recon) x num_samples rows
    grid_w = 2 * sample_w + 3 * padding
    grid_h = num_samples * sample_h + (num_samples + 1) * padding

    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 40  # Dark gray background

    for row, sample in enumerate(samples):
        y = padding + row * (sample_h + padding)

        # Original
        orig = (sample["original"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        x = padding
        grid[y:y+sample_h, x:x+sample_w] = orig

        # Reconstruction
        recon = (sample["recon"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        recon = np.clip(recon, 0, 255)
        x = 2 * padding + sample_w
        grid[y:y+sample_h, x:x+sample_w] = recon

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(output_path)
    print(f"Saved comparison grid to {output_path}")


def generate_error_heatmap(
    model,
    dataloader,
    device: str,
    output_path: Path,
):
    """Generate heatmap showing reconstruction error distribution.

    Args:
        model: Tokenizer model
        dataloader: DataLoader
        device: Device
        output_path: Path to save image
    """
    # Get a single sample
    batch = next(iter(dataloader))
    frames = batch["frame"][:1].to(device)

    with torch.no_grad():
        output = model(frames)
        recon = output["reconstruction"]

    # Compute per-pixel error
    error = torch.abs(frames - recon).mean(dim=1)  # Average across RGB
    error = error[0].cpu().numpy()

    # Normalize to 0-255
    error_normalized = (error / error.max() * 255).astype(np.uint8)

    # Apply colormap (hot)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    colored = cm.hot(error_normalized / 255.0)[:, :, :3]
    colored = (colored * 255).astype(np.uint8)

    # Create figure with original, recon, and error map
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    orig = (frames[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    rec = (recon[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    rec = np.clip(rec, 0, 255)

    axes[0].imshow(orig)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(rec)
    axes[1].set_title("Reconstruction")
    axes[1].axis("off")

    im = axes[2].imshow(error, cmap="hot")
    axes[2].set_title("Error Heatmap")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved error heatmap to {output_path}")


def plot_training_history(checkpoint_dir: Path, output_path: Path):
    """Plot training history from JSON file.

    Args:
        checkpoint_dir: Directory containing training_history.json
        output_path: Path to save plot
    """
    history_path = checkpoint_dir / "training_history.json"
    if not history_path.exists():
        print(f"No training history found at {history_path}")
        return

    with open(history_path) as f:
        history = json.load(f)

    if not history:
        print("Training history is empty")
        return

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    losses = [h["loss"] for h in history]
    mses = [h["mse"] for h in history]
    lpips_vals = [h["lpips"] for h in history]
    psnrs = [h["psnr"] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Total loss
    axes[0, 0].plot(epochs, losses, "b-", linewidth=2)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Total Loss")
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].grid(True, alpha=0.3)

    # MSE
    axes[0, 1].plot(epochs, mses, "g-", linewidth=2)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("MSE")
    axes[0, 1].set_title("MSE Loss")
    axes[0, 1].grid(True, alpha=0.3)

    # LPIPS
    axes[1, 0].plot(epochs, lpips_vals, "r-", linewidth=2)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("LPIPS")
    axes[1, 0].set_title("Perceptual Loss (LPIPS)")
    axes[1, 0].grid(True, alpha=0.3)

    # PSNR
    axes[1, 1].plot(epochs, psnrs, "m-", linewidth=2)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("PSNR (dB)")
    axes[1, 1].set_title("PSNR (higher is better)")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training history plot to {output_path}")


def main():
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Tokenizer Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {args.device}")
    print(f"Output dir: {output_dir}")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    model, checkpoint = load_model(checkpoint_path, args.device)

    epoch = checkpoint.get("epoch", "?")
    step = checkpoint.get("global_step", "?")
    print(f"Loaded checkpoint from epoch {epoch}, step {step}")
    print(f"Model parameters: {model.get_num_params():,}")

    # Load dataset
    print(f"\nLoading dataset from {args.frames_dir}...")
    dataset = SingleFrameDataset(args.frames_dir)
    print(f"Found {len(dataset)} frames")

    if len(dataset) == 0:
        print("ERROR: No frames found!")
        return

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(model, dataloader, args.device, max_batches=100)

    print(f"\n{'='*40}")
    print("Metrics")
    print("=" * 40)
    print(f"PSNR: {metrics['psnr']:.2f} dB")
    print(f"MSE:  {metrics['mse']:.6f}")
    print(f"Samples evaluated: {metrics['num_samples']}")
    print("=" * 40)

    # Quality interpretation
    if metrics["psnr"] >= 30:
        quality = "Excellent (>30 dB)"
    elif metrics["psnr"] >= 25:
        quality = "Good (25-30 dB)"
    elif metrics["psnr"] >= 20:
        quality = "Acceptable (20-25 dB)"
    else:
        quality = "Poor (<20 dB) - consider scaling up model"
    print(f"Quality: {quality}")

    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "checkpoint": str(checkpoint_path),
            "epoch": epoch,
            "step": step,
            **metrics,
        }, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")

    # Generate comparison grid
    print("\nGenerating comparison grid...")
    comparison_path = output_dir / "comparison_grid.png"
    generate_comparison_grid(model, dataloader, args.device, args.num_samples, comparison_path)

    # Generate error heatmap
    print("\nGenerating error heatmap...")
    heatmap_path = output_dir / "error_heatmap.png"
    generate_error_heatmap(model, dataloader, args.device, heatmap_path)

    # Plot training history if requested
    if args.plot_history:
        print("\nPlotting training history...")
        checkpoint_dir = checkpoint_path.parent
        history_path = output_dir / "training_history.png"
        plot_training_history(checkpoint_dir, history_path)

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
