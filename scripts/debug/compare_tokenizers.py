#!/usr/bin/env python3
"""Compare different tokenizer models (CNN vs Transformer) visually and quantitatively."""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from PIL import Image

from ahriuwu.models import (
    create_tokenizer,
    create_transformer_tokenizer,
    psnr,
    LPIPSLoss,
)


def load_cnn_tokenizer(checkpoint_path: str, device: str):
    """Load CNN tokenizer from checkpoint."""
    model = create_tokenizer()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    return model, ckpt.get("args", {})


def load_transformer_tokenizer(checkpoint_path: str, device: str):
    """Load transformer tokenizer from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = ckpt.get("args", {})
    model_size = args.get("model_size", "small")
    use_rope = args.get("use_rope", False)

    model = create_transformer_tokenizer(model_size, use_rope=use_rope)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    return model, args


def load_frame(frame_path: str, target_size: tuple = (256, 256)) -> torch.Tensor:
    """Load and preprocess a frame."""
    frame = cv2.imread(frame_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
    frame = torch.from_numpy(frame).float() / 255.0
    frame = frame.permute(2, 0, 1)  # HWC -> CHW
    return frame


def reconstruct_cnn(model, frame: torch.Tensor, device: str) -> torch.Tensor:
    """Reconstruct frame using CNN tokenizer."""
    with torch.no_grad():
        frame = frame.unsqueeze(0).to(device)
        with autocast("cuda", dtype=torch.bfloat16):
            latent = model.encode(frame)
            recon = model.decode(latent)
        return recon.squeeze(0).float().cpu()


def reconstruct_transformer(model, frame: torch.Tensor, device: str) -> torch.Tensor:
    """Reconstruct frame using transformer tokenizer (no masking)."""
    with torch.no_grad():
        frame = frame.unsqueeze(0).to(device)
        with autocast("cuda", dtype=torch.bfloat16):
            output = model(frame, mask_ratio=0.0)
            recon = output["reconstruction"]
        return recon.squeeze(0).float().cpu()


def compute_metrics(original: torch.Tensor, recon: torch.Tensor, lpips_fn) -> dict:
    """Compute quality metrics between original and reconstruction."""
    # Ensure same device
    original = original.unsqueeze(0)
    recon = recon.unsqueeze(0)

    # PSNR
    psnr_val = psnr(recon, original).item()

    # MSE
    mse_val = F.mse_loss(recon, original).item()

    # LPIPS
    with torch.no_grad():
        lpips_val = lpips_fn(recon.cuda(), original.cuda()).item()

    return {
        "psnr": psnr_val,
        "mse": mse_val,
        "lpips": lpips_val,
    }


def create_comparison_image(
    original: torch.Tensor,
    recons: dict[str, torch.Tensor],
    metrics: dict[str, dict],
) -> np.ndarray:
    """Create side-by-side comparison image with metrics."""
    # Convert tensors to numpy
    def to_numpy(t):
        return (t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

    orig_np = to_numpy(original)

    # Create comparison: Original | Model1 | Model2 | Model3
    images = [orig_np]
    labels = ["Original"]

    for name, recon in recons.items():
        images.append(to_numpy(recon))
        m = metrics[name]
        labels.append(f"{name}\nPSNR:{m['psnr']:.1f} LPIPS:{m['lpips']:.3f}")

    # Stack horizontally
    comparison = np.hstack(images)

    # Add labels at the top
    h, w = comparison.shape[:2]
    labeled = np.zeros((h + 40, w, 3), dtype=np.uint8)
    labeled[40:] = comparison

    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    img_width = orig_np.shape[1]
    for i, label in enumerate(labels):
        x = i * img_width + 5
        for j, line in enumerate(label.split('\n')):
            cv2.putText(labeled, line, (x, 15 + j*15), font, 0.4, (255, 255, 255), 1)

    return labeled


def main():
    parser = argparse.ArgumentParser(description="Compare tokenizer models")
    parser.add_argument(
        "--frames-dir",
        type=str,
        default="data/processed/frames",
        help="Directory containing frames",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="samples/tokenizer_comparison",
        help="Output directory for comparisons",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of sample frames to compare",
    )
    parser.add_argument(
        "--cnn-checkpoint",
        type=str,
        default="checkpoints/tokenizer_epoch_010.pt",
        help="CNN tokenizer checkpoint",
    )
    parser.add_argument(
        "--transformer-dynamics",
        type=str,
        default="checkpoints/run_20260115_195915/transformer_tokenizer_best.pt",
        help="Transformer tokenizer used for dynamics training",
    )
    parser.add_argument(
        "--transformer-current",
        type=str,
        default="checkpoints/run_20260117_165401/transformer_tokenizer_latest.pt",
        help="Current transformer tokenizer",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for frame selection",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Tokenizer Comparison")
    print("=" * 60)

    # Load models
    print("\nLoading models...")

    print("  Loading CNN tokenizer...")
    cnn_model, cnn_args = load_cnn_tokenizer(args.cnn_checkpoint, args.device)
    cnn_params = sum(p.numel() for p in cnn_model.parameters())
    print(f"    Params: {cnn_params:,}")

    print("  Loading Transformer (dynamics)...")
    trans_dyn_model, trans_dyn_args = load_transformer_tokenizer(
        args.transformer_dynamics, args.device
    )
    trans_dyn_params = sum(p.numel() for p in trans_dyn_model.parameters())
    print(f"    Params: {trans_dyn_params:,}")

    print("  Loading Transformer (current)...")
    trans_cur_model, trans_cur_args = load_transformer_tokenizer(
        args.transformer_current, args.device
    )
    trans_cur_params = sum(p.numel() for p in trans_cur_model.parameters())
    print(f"    Params: {trans_cur_params:,}")

    # Load LPIPS
    print("  Loading LPIPS...")
    lpips_fn = LPIPSLoss().to(args.device)

    # Collect all frames
    print(f"\nIndexing frames from {args.frames_dir}...")
    frames_dir = Path(args.frames_dir)
    all_frames = list(frames_dir.glob("*/frame_*.jpg"))
    print(f"  Found {len(all_frames):,} frames")

    # Sample frames
    sample_frames = random.sample(all_frames, min(args.num_samples, len(all_frames)))

    # Run comparison
    print(f"\nComparing on {len(sample_frames)} frames...")

    all_metrics = {
        "cnn": [],
        "trans_dyn": [],
        "trans_cur": [],
    }

    for i, frame_path in enumerate(sample_frames):
        print(f"  [{i+1}/{len(sample_frames)}] {frame_path.name}")

        # Load frame
        frame = load_frame(str(frame_path))

        # Reconstruct with each model
        recons = {
            "CNN": reconstruct_cnn(cnn_model, frame, args.device),
            "Trans-Dyn": reconstruct_transformer(trans_dyn_model, frame, args.device),
            "Trans-Cur": reconstruct_transformer(trans_cur_model, frame, args.device),
        }

        # Compute metrics
        metrics = {}
        for name, recon in recons.items():
            metrics[name] = compute_metrics(frame, recon, lpips_fn)

        all_metrics["cnn"].append(metrics["CNN"])
        all_metrics["trans_dyn"].append(metrics["Trans-Dyn"])
        all_metrics["trans_cur"].append(metrics["Trans-Cur"])

        # Create comparison image
        comparison = create_comparison_image(frame, recons, metrics)

        # Save
        save_path = output_dir / f"comparison_{i:03d}.png"
        Image.fromarray(comparison).save(save_path)

    # Compute aggregate metrics
    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)

    results = {}
    for model_name, model_metrics in [
        ("CNN (13M params)", all_metrics["cnn"]),
        ("Transformer-Dynamics (40M)", all_metrics["trans_dyn"]),
        ("Transformer-Current (40M)", all_metrics["trans_cur"]),
    ]:
        avg_psnr = np.mean([m["psnr"] for m in model_metrics])
        avg_mse = np.mean([m["mse"] for m in model_metrics])
        avg_lpips = np.mean([m["lpips"] for m in model_metrics])
        std_psnr = np.std([m["psnr"] for m in model_metrics])

        print(f"\n{model_name}:")
        print(f"  PSNR:  {avg_psnr:.2f} ± {std_psnr:.2f} dB")
        print(f"  MSE:   {avg_mse:.5f}")
        print(f"  LPIPS: {avg_lpips:.4f}")

        results[model_name] = {
            "psnr_mean": avg_psnr,
            "psnr_std": std_psnr,
            "mse_mean": avg_mse,
            "lpips_mean": avg_lpips,
        }

    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Create summary grid
    print(f"\nComparison images saved to {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
