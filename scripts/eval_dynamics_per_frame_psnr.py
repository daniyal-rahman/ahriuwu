#!/usr/bin/env python3
"""120-frame autoregressive rollout with per-frame PSNR curve.

Wraps ``scripts/eval_dynamics.py``'s ``rollout_predictions`` and replaces
its averaged-PSNR reducer with a per-frame accumulator + matplotlib plot.

Outputs (under ``--output-dir``):
  per_frame_psnr.csv   — frame_idx, mean_psnr, std_psnr, n
  per_frame_psnr.png   — line plot vs frame index
  rollout_grid.png     — first-batch GT vs predicted side-by-side
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Reuse helpers from the existing evaluator.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))
from eval_dynamics import (  # type: ignore
    rollout_predictions,
    latents_to_frames,
    load_dynamics,
)
from ahriuwu.data import PackedLatentSequenceDataset
from ahriuwu.models import DiffusionSchedule, psnr
# Legacy module — the trained transformer_tokenizer_latest.pt predates the
# attention-unification refactor and only loads cleanly here. PROGRESS_LOG
# documents this shim.
from ahriuwu.models.transformer_tokenizer_legacy import (
    create_transformer_tokenizer as create_transformer_tokenizer_legacy,
)


def load_tokenizer_with_prefix_strip(ckpt_path, device, size_override: str):
    """Strips ``_orig_mod.`` prefix (torch.compile artifact) and instantiates
    the legacy tokenizer at the size that actually matches the saved weights.

    The checkpoint's ``args.model_size='small'`` is misleading — actual
    ``latent_dim=48`` matches the legacy ``medium`` preset. Caller passes the
    correct size via ``size_override``.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    use_rope = any("rope" in k for k in sd)
    model = create_transformer_tokenizer_legacy(size=size_override, use_rope=use_rope)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[tokenizer] missing {len(missing)} keys; first 3: {missing[:3]}", flush=True)
    if unexpected:
        print(f"[tokenizer] unexpected {len(unexpected)} keys; first 3: {unexpected[:3]}", flush=True)
    model = model.to(device).eval()
    return model


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dynamics-checkpoint", required=True)
    ap.add_argument("--tokenizer-checkpoint", required=True)
    ap.add_argument("--latents-dir", default="data/processed/latents")
    ap.add_argument("--output-dir", default="eval_results/per_frame_psnr")
    ap.add_argument("--context-frames", type=int, default=128)
    ap.add_argument("--predict-frames", type=int, default=120)
    ap.add_argument("--num-steps", type=int, default=64)
    ap.add_argument("--tau-ctx", type=float, default=0.1)
    ap.add_argument("--use-shortcut", action="store_true")
    ap.add_argument("--shortcut-k-max", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-batches", type=int, default=4,
                    help="Number of batches to average PSNR over")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save-grid", action="store_true",
                    help="Also dump a GT/pred side-by-side grid for the first batch")
    ap.add_argument("--tokenizer-size", default="medium",
                    choices=["tiny", "small", "medium", "large"],
                    help="Tokenizer size preset (legacy module). Default medium "
                         "(matches transformer_tokenizer_latest.pt latent_dim=48)")
    return ap.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = args.device
    total_seq_len = args.context_frames + args.predict_frames

    print(f"[load] dynamics={args.dynamics_checkpoint}", flush=True)
    dynamics, _ = load_dynamics(Path(args.dynamics_checkpoint), device)

    print(f"[load] tokenizer={args.tokenizer_checkpoint} size={args.tokenizer_size}", flush=True)
    tokenizer = load_tokenizer_with_prefix_strip(
        Path(args.tokenizer_checkpoint), device, size_override=args.tokenizer_size,
    )

    schedule = DiffusionSchedule()

    print(f"[data] {args.latents_dir} seq_len={total_seq_len}", flush=True)
    dataset = PackedLatentSequenceDataset(args.latents_dir, sequence_length=total_seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # Per-frame accumulators
    psnrs = [[] for _ in range(args.predict_frames)]
    n_done = 0
    grid_saved = False

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= args.num_batches:
            break
        latents = batch["latents"][:, :total_seq_len].to(device)
        if latents.shape[1] < total_seq_len:
            continue
        context = latents[:, :args.context_frames]
        target = latents[:, args.context_frames:total_seq_len]

        print(f"[rollout] batch {batch_idx+1}/{args.num_batches} "
              f"B={latents.shape[0]} ctx={args.context_frames} pred={args.predict_frames}",
              flush=True)
        predicted = rollout_predictions(
            dynamics, schedule, context, args.predict_frames,
            args.num_steps, device, tau_ctx=args.tau_ctx,
            use_shortcut=args.use_shortcut, k_max=args.shortcut_k_max,
        )

        target_frames = latents_to_frames(tokenizer, target, device)
        pred_frames = latents_to_frames(tokenizer, predicted, device)

        B = pred_frames.shape[0]
        for b in range(B):
            for t in range(args.predict_frames):
                p = psnr(pred_frames[b, t:t+1], target_frames[b, t:t+1]).item()
                psnrs[t].append(p)
        n_done += B

        if args.save_grid and not grid_saved:
            # Cheap inline grid: every Nth frame, GT row over pred row.
            from PIL import Image as _Image
            stride = max(1, args.predict_frames // 12)
            cols = list(range(0, args.predict_frames, stride))
            fh, fw = pred_frames.shape[-2], pred_frames.shape[-1]
            grid = np.zeros((fh * 2 + 4, (fw + 2) * len(cols), 3), dtype=np.uint8)
            for ci, t in enumerate(cols):
                x = ci * (fw + 2)
                gt_img = (target_frames[0, t].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                pr_img = (pred_frames[0, t].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                grid[:fh, x:x + fw] = gt_img
                grid[fh + 4:fh + 4 + fh, x:x + fw] = pr_img
            _Image.fromarray(grid).save(out / "rollout_grid.png")
            grid_saved = True

    # Reduce
    rows = []
    for t in range(args.predict_frames):
        vals = np.array(psnrs[t], dtype=np.float64)
        rows.append({
            "frame_idx": t,
            "mean_psnr": float(vals.mean()) if vals.size else float("nan"),
            "std_psnr": float(vals.std()) if vals.size else float("nan"),
            "n": int(vals.size),
        })

    csv_path = out / "per_frame_psnr.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame_idx", "mean_psnr", "std_psnr", "n"])
        w.writeheader()
        w.writerows(rows)
    print(f"[write] {csv_path}", flush=True)

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = np.array([r["frame_idx"] for r in rows])
        ys = np.array([r["mean_psnr"] for r in rows])
        es = np.array([r["std_psnr"] for r in rows])
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(xs, ys, label="mean PSNR (dB)")
        ax.fill_between(xs, ys - es, ys + es, alpha=0.2, label="±1 std")
        ax.set_xlabel("predicted frame index (0 = first frame after context)")
        ax.set_ylabel("pixel PSNR (dB)")
        ax.set_title(f"Per-frame PSNR — {args.predict_frames}-frame rollout "
                     f"(n={n_done} sequences, num_steps={args.num_steps}"
                     f"{', shortcut' if args.use_shortcut else ''})")
        ax.grid(True, alpha=0.3)
        ax.legend()
        png = out / "per_frame_psnr.png"
        fig.tight_layout()
        fig.savefig(png, dpi=120)
        print(f"[write] {png}", flush=True)
    except Exception as e:
        print(f"[warn] matplotlib plot failed: {e!r}", flush=True)

    # Summary
    head = np.mean([r["mean_psnr"] for r in rows[:10]])
    tail = np.mean([r["mean_psnr"] for r in rows[-10:]])
    print(f"\n[summary] {n_done} sequences over {args.predict_frames} frames", flush=True)
    print(f"  first-10-frame mean PSNR: {head:.2f} dB", flush=True)
    print(f"  last-10-frame  mean PSNR: {tail:.2f} dB", flush=True)
    print(f"  degradation: {head - tail:+.2f} dB", flush=True)


if __name__ == "__main__":
    main()
