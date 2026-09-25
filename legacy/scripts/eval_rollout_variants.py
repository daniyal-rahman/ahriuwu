#!/usr/bin/env python3
"""Compare per-frame PSNR across rollout variants on the SAME fixed-seed
sequences. Produces a single overlaid trajectory plot.

Variants:
  gt_decode  — Tokenizer.decode(true_latents). No dynamics involvement.
               This is the **ceiling**: any rollout PSNR above this would
               be magic.
  k4         — Dynamics rollout, num_steps=4 shortcut forcing (deployment
               inference).
  k64        — Dynamics rollout, num_steps=64 full diffusion (slowest
               but should be the best dynamics quality).
  k4_tau0    — Same as k4 but tau_ctx=0.0 (no context noise). Tests
               whether context-noise injection is helping or hurting.

Each variant is evaluated over the SAME N sequences sampled deterministically
from PackedLatentSequenceDataset using ``--seed``. Same sequences across
variants means paired comparison — std bars across sequences reflect
sequence-difficulty variance, not sampling variance.

Outputs:
  <output-dir>/per_frame_psnr_compare.csv   — long format (variant, frame_idx, mean, std, n)
  <output-dir>/per_frame_psnr_compare.png   — overlaid trajectories with std bands
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

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
from ahriuwu.models.transformer_tokenizer_legacy import (
    create_transformer_tokenizer as create_transformer_tokenizer_legacy,
)


def load_tokenizer_with_prefix_strip(ckpt_path, device, size_override):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    use_rope = any("rope" in k for k in sd)
    model = create_transformer_tokenizer_legacy(size=size_override, use_rope=use_rope)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    n_buf_missing = sum(1 for k in missing if any(s in k for s in ("rope.", "rope_indices")))
    n_critical_missing = len(missing) - n_buf_missing
    if n_critical_missing > 0:
        print(f"[tokenizer] WARNING: {n_critical_missing} non-buffer keys missing", flush=True)
    if unexpected:
        print(f"[tokenizer] {len(unexpected)} unexpected (ignored)", flush=True)
    return model.to(device).eval()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dynamics-checkpoint", required=True)
    ap.add_argument("--tokenizer-checkpoint", required=True)
    ap.add_argument("--tokenizer-size", default="medium")
    ap.add_argument("--latents-dir", default="data/processed/latents")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--context-frames", type=int, default=128)
    ap.add_argument("--predict-frames", type=int, default=120)
    ap.add_argument("--shortcut-k-max", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-sequences", type=int, default=16,
                    help="Total sequences evaluated per variant (paired)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=20260510)
    ap.add_argument("--variants", default="gt_decode,k4,k4_tau0,k64",
                    help="Comma-separated variant names")
    return ap.parse_args()


def pick_indices(dataset_len: int, n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return rng.sample(range(dataset_len), min(n, dataset_len))


def variant_config(name: str) -> dict:
    if name == "gt_decode":
        return {"kind": "gt_decode"}
    if name == "k4":
        return {"kind": "rollout", "num_steps": 4, "use_shortcut": True, "tau_ctx": 0.1}
    if name == "k4_tau0":
        return {"kind": "rollout", "num_steps": 4, "use_shortcut": True, "tau_ctx": 0.0}
    if name == "k8":
        return {"kind": "rollout", "num_steps": 8, "use_shortcut": True, "tau_ctx": 0.1}
    if name == "k16":
        return {"kind": "rollout", "num_steps": 16, "use_shortcut": True, "tau_ctx": 0.1}
    if name == "k32":
        return {"kind": "rollout", "num_steps": 32, "use_shortcut": True, "tau_ctx": 0.1}
    if name == "k64":
        return {"kind": "rollout", "num_steps": 64, "use_shortcut": False, "tau_ctx": 0.1}
    raise ValueError(f"unknown variant: {name}")


@torch.no_grad()
def evaluate_variant(
    name: str,
    cfg: dict,
    dataset,
    indices: list[int],
    dynamics,
    tokenizer,
    schedule,
    device: str,
    context_frames: int,
    predict_frames: int,
    batch_size: int,
    shortcut_k_max: int,
) -> list[list[float]]:
    total_seq_len = context_frames + predict_frames
    psnrs = [[] for _ in range(predict_frames)]
    n_done = 0

    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=2)

    for batch_idx, batch in enumerate(loader):
        latents = batch["latents"][:, :total_seq_len].to(device)
        if latents.shape[1] < total_seq_len:
            continue
        target = latents[:, context_frames:total_seq_len]
        target_frames = latents_to_frames(tokenizer, target, device)

        if cfg["kind"] == "gt_decode":
            # Decode the SAME ground-truth latents twice — once as "target",
            # once as "predicted". PSNR should be infinity (or near it) for
            # this comparison. Useful as a sanity check that the decode is
            # deterministic.
            pred_frames = target_frames
        else:
            context = latents[:, :context_frames]
            predicted = rollout_predictions(
                dynamics, schedule, context, predict_frames,
                cfg["num_steps"], device,
                tau_ctx=cfg["tau_ctx"],
                use_shortcut=cfg["use_shortcut"],
                k_max=shortcut_k_max,
            )
            pred_frames = latents_to_frames(tokenizer, predicted, device)

        B = pred_frames.shape[0]
        for b in range(B):
            for t in range(predict_frames):
                p = psnr(pred_frames[b, t:t+1], target_frames[b, t:t+1]).item()
                psnrs[t].append(p)
        n_done += B
        print(f"[{name}] batch {batch_idx+1} ({n_done}/{len(indices)})", flush=True)

    return psnrs


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = args.device
    total_seq_len = args.context_frames + args.predict_frames

    print(f"[load] dynamics={args.dynamics_checkpoint}", flush=True)
    dynamics, _ = load_dynamics(Path(args.dynamics_checkpoint), device)
    print(f"[load] tokenizer={args.tokenizer_checkpoint} size={args.tokenizer_size}", flush=True)
    tokenizer = load_tokenizer_with_prefix_strip(
        Path(args.tokenizer_checkpoint), device, args.tokenizer_size,
    )
    schedule = DiffusionSchedule()

    print(f"[data] {args.latents_dir} seq_len={total_seq_len}", flush=True)
    dataset = PackedLatentSequenceDataset(args.latents_dir, sequence_length=total_seq_len)
    indices = pick_indices(len(dataset), args.num_sequences, args.seed)
    print(f"[data] picked {len(indices)} sequences (seed={args.seed})", flush=True)

    variant_names = [v.strip() for v in args.variants.split(",") if v.strip()]
    print(f"[run] variants = {variant_names}", flush=True)

    results: dict[str, list[list[float]]] = {}
    for name in variant_names:
        cfg = variant_config(name)
        print(f"\n=== variant: {name}  cfg={cfg} ===", flush=True)
        psnrs = evaluate_variant(
            name, cfg, dataset, indices, dynamics, tokenizer, schedule, device,
            args.context_frames, args.predict_frames,
            args.batch_size, args.shortcut_k_max,
        )
        results[name] = psnrs

    # CSV (long format)
    csv_path = out / "per_frame_psnr_compare.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "frame_idx", "mean_psnr", "std_psnr", "n"])
        for name, psnrs in results.items():
            for t, vals in enumerate(psnrs):
                a = np.array(vals, dtype=np.float64)
                if a.size == 0:
                    continue
                # gt_decode can produce +inf — clip for sanity
                a = a[np.isfinite(a)]
                if a.size == 0:
                    w.writerow([name, t, float("nan"), float("nan"), 0])
                else:
                    w.writerow([name, t, float(a.mean()), float(a.std()), int(a.size)])
    print(f"[write] {csv_path}", flush=True)

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {
        "gt_decode": "#888888",
        "k4": "#1f77b4",
        "k4_tau0": "#2ca02c",
        "k64": "#d62728",
    }
    for name in variant_names:
        psnrs = results[name]
        means = np.array([np.mean([v for v in row if np.isfinite(v)]) if row else np.nan for row in psnrs])
        stds = np.array([np.std([v for v in row if np.isfinite(v)]) if row else np.nan for row in psnrs])
        xs = np.arange(len(means))
        ax.plot(xs, means, label=name, color=colors.get(name, None), linewidth=1.5)
        ax.fill_between(xs, means - stds, means + stds, alpha=0.15, color=colors.get(name, None))

    ax.set_xlabel("predicted frame index (0 = first frame after context)")
    ax.set_ylabel("pixel PSNR (dB)")
    ax.set_title(
        f"Per-frame PSNR — {args.predict_frames}-frame rollout "
        f"(n={args.num_sequences} paired sequences, seed={args.seed})"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    png = out / "per_frame_psnr_compare.png"
    fig.savefig(png, dpi=120)
    print(f"[write] {png}", flush=True)

    print("\n=== Summary (mean over all predicted frames) ===")
    for name in variant_names:
        psnrs = results[name]
        flat = [v for row in psnrs for v in row if np.isfinite(v)]
        if flat:
            print(f"  {name:>10}: {np.mean(flat):.2f} dB  (n={len(flat)})")


if __name__ == "__main__":
    main()
