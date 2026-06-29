#!/usr/bin/env python3
"""Pretokenize the action-labeled REPLAY frames with the frozen v7 tokenizer.

Encodes ``<frames-root>/<match_id>/NNNNNN.png`` through the frozen v7 transformer
tokenizer (512 latents x 16 dim) and writes ONE packed file per match:

    <out>/<match_id>.pt = {"latents": (N, 32, 16, 16) float16,
                           "frame_indices": (N,) int32}

which is the exact format ``PackedLatentSequenceDataset`` and
``ReplayLatentSequenceDataset`` read. The 512x16 bottleneck is folded to a
16x16x32 grid (``view(16,16,-1)``), giving dynamics latent_dim 32 — the same
512-bottleneck -> 256-spatial reshape DreamerV4 uses.

IMPORTANT: the tokenizer is built from the checkpoint's *saved model_config*
(exact architecture), NOT a size preset — the 'large' preset is latent_dim 64 /
num_latents 256, which does NOT match v7 (latent_dim 16 / num_latents 512), so a
preset-based load would silently produce garbage via strict=False.

This script is dynamics-only and does NOT modify the YT pipeline's
scripts/pretokenize_frames.py.
"""
import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ahriuwu.models.transformer_tokenizer import TransformerTokenizer


def load_v7(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = {k: v for k, v in ck["model_config"].items() if k != "size_preset"}
    model = TransformerTokenizer(**cfg)
    sd = ck["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad_missing = [k for k in missing if "rope" not in k.lower()]
    if bad_missing or unexpected:
        raise RuntimeError(
            f"tokenizer load mismatch — missing(non-rope)={bad_missing[:8]} "
            f"unexpected={unexpected[:8]}"
        )
    return model.to(device).eval(), cfg, ck.get("global_step")


def load_frame(path, size):
    im = cv2.imread(str(path))
    if im is None:
        raise RuntimeError(f"cv2 could not read {path}")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if im.shape[:2] != (size, size):
        im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(im).float().div_(255.0).permute(2, 0, 1)  # (3,H,W) in [0,1]


def match_pngs(d):
    return sorted(d.glob("*.png"), key=lambda p: int(p.stem))


class _FrameDS(Dataset):
    """Per-match frame dataset — lets a DataLoader decode + prefetch frames in
    parallel worker processes. Serial cv2.imread left the GPU at ~30-60% util."""

    def __init__(self, pngs, size):
        self.pngs = pngs
        self.size = size

    def __len__(self):
        return len(self.pngs)

    def __getitem__(self, i):
        return load_frame(self.pngs[i], self.size)  # (3, H, W) in [0,1]


@torch.no_grad()
def encode_match(model, pngs, device, batch_size, amp_dtype, size, num_workers):
    dl = DataLoader(
        _FrameDS(pngs, size), batch_size=batch_size, num_workers=num_workers,
        pin_memory=True, drop_last=False,
    )
    out = []
    for batch in dl:                                       # (B, 3, H, W)
        batch = batch.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            lat = model.encode(batch)["latent"]            # (B, num_latents, latent_dim)
        B = lat.shape[0]
        lat = lat.view(B, 16, 16, -1).permute(0, 3, 1, 2)  # (B, 32, 16, 16)
        out.append(lat.float().cpu().to(torch.float16))
    return torch.cat(out, 0)


@torch.no_grad()
def recon_psnr(model, pngs, device, amp_dtype, size, n=8):
    x = torch.stack([load_frame(p, size) for p in pngs[:n]]).to(device)
    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        lat = model.encode(x)["latent"]
        recon = model.decode(lat, num_frames=1)[:, 0]  # (B,3,H,W)
    mse = ((recon.float().clamp(0, 1) - x.float()) ** 2).mean().item()
    return 10 * np.log10(1.0 / max(mse, 1e-10)), tuple(lat.shape)


def write_info(out: Path, args, cfg, step, n_matches, total_frames):
    info = out / "INFO.md"
    ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    dyn_dim = cfg["num_latents"] * cfg["latent_dim"] // 256
    info.write_text(f"""# Dynamics training latents — replay corpus, frozen v7 tokenizer

**What this is.** Pre-tokenized latents for the DreamerV4-style **dynamics model**,
produced by encoding the action-labeled **replay** frames through the **frozen v7
transformer tokenizer**. One packed file per match.

Generated: {ts} by `scripts/pretokenize_replay_v7.py`.

## Source
- Frames: `{args.frames_root}` (action-labeled replay, 352x352 PNG, `NNNNNN.png`).
- Tokenizer checkpoint: `{args.checkpoint}` (global_step={step}).
- Tokenizer config (exact, from checkpoint `model_config`):
  embed_dim={cfg['embed_dim']}, num_latents={cfg['num_latents']}, latent_dim={cfg['latent_dim']},
  {cfg['num_encoder_layers']} enc / {cfg['num_decoder_layers']} dec layers, use_rope={cfg.get('use_rope')},
  soft_cap={cfg.get('soft_cap')}, temporal_every={cfg.get('temporal_every')}.

## Format (one file per match)
`<match_id>.pt` = a torch dict:
- `latents`: float16 tensor `(N, 32, 16, 16)` — N frames, dynamics latent_dim=32.
- `frame_indices`: int32 tensor `(N,)` — original PNG frame numbers (strictly ascending).

The tokenizer bottleneck is **{cfg['num_latents']} latents x {cfg['latent_dim']} dim**;
`encode()` returns `(N, {cfg['num_latents']}, {cfg['latent_dim']})` which is folded via
`view(16,16,-1)` to a 16x16 grid of **{dyn_dim}** channels (the dynamics latent_dim) —
the same 512-bottleneck -> 256-spatial reshape DreamerV4 uses.

## Contents
- {n_matches} matches, {total_frames:,} frames total.

## How to use
```python
from ahriuwu.data import PackedLatentSequenceDataset            # latents only
from ahriuwu.data.replay_dataset import ReplayLatentSequenceDataset  # + actions/rewards
ds = PackedLatentSequenceDataset("{out}", sequence_length=128)
```
Training: `scripts/train_dynamics.py --latents-dir {out} --packed --latent-dim {dyn_dim} ...`

## Caveats
- **No action/reward labels here** — these are latents only. The action-conditioned
  path needs `labels.json`/`clicks.json` + a `garen_win` manifest per match (not yet
  materialized). Base/world-model pretraining (actions OFF) works now.
- This pairs ONLY with the v7 tokenizer above. Latents from any other tokenizer
  (e.g. the older dim-48 YouTube latents under `/mnt/storage/data/ahriuwu/latents_pt`)
  are NOT interchangeable.
""")
    (out / "provenance.json").write_text(json.dumps({
        "generated": ts, "script": "scripts/pretokenize_replay_v7.py",
        "frames_root": str(args.frames_root), "checkpoint": str(args.checkpoint),
        "tokenizer_global_step": step, "tokenizer_config": cfg,
        "n_matches": n_matches, "total_frames": total_frames,
        "dynamics_latent_dim": dyn_dim,
    }, indent=2, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default="/mnt/storage/data/ahriuwu-checkpoints/tokenizer_v7/transformer_tokenizer_latest.pt")
    ap.add_argument("--frames-root", default="/scratch/ahriuwu/action_labeled_352png_train_flat")
    ap.add_argument("--out", default="/scratch/ahriuwu/dynamics_replay_latents_v7_dim32")
    ap.add_argument("--batch-size", type=int, default=32)  # 64 OOMs the v7 'large' tokenizer on 16GB
    ap.add_argument("--num-workers", type=int, default=6)   # parallel decode is the speedup, not batch size
    ap.add_argument("--max-matches", type=int, default=0, help="0=all; >0 for a quick check")
    ap.add_argument("--resume", action="store_true", help="skip matches that already have a .pt")
    ap.add_argument("--verify", action="store_true", help="report recon PSNR on the first match, then exit unless full run")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "needs CUDA"
    device = "cuda"
    amp_dtype = torch.bfloat16

    model, cfg, step = load_v7(args.checkpoint, device)
    size = int(cfg.get("img_size", 352))
    print(f"loaded v7 (step {step}): num_latents={cfg['num_latents']} latent_dim={cfg['latent_dim']} "
          f"-> dynamics dim {cfg['num_latents']*cfg['latent_dim']//256}", flush=True)

    frames_root = Path(args.frames_root)
    matches = sorted([d for d in frames_root.iterdir() if d.is_dir()])
    if args.max_matches:
        matches = matches[:args.max_matches]

    if args.verify:
        pngs = match_pngs(matches[0])
        psnr, shape = recon_psnr(model, pngs, device, amp_dtype, size)
        print(f"VERIFY: {matches[0].name} encode latent {shape} | recon PSNR {psnr:.2f} dB "
              f"({'OK' if psnr > 25 else 'SUSPECT — load may be wrong'})", flush=True)
        if not args.max_matches:  # verify-only run
            return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import time
    t0 = time.time()
    total_frames = 0
    done = 0
    for mi, d in enumerate(matches):
        outp = out / f"{d.name}.pt"
        if args.resume and outp.exists():
            done += 1
            continue
        pngs = match_pngs(d)
        if not pngs:
            print(f"[{mi+1}/{len(matches)}] {d.name}: no PNGs, skipped", flush=True)
            continue
        lat = encode_match(model, pngs, device, args.batch_size, amp_dtype, size, args.num_workers)
        idxs = torch.tensor([int(p.stem) for p in pngs], dtype=torch.int32)
        tmp = outp.with_suffix(".pt.tmp")
        torch.save({"latents": lat, "frame_indices": idxs}, tmp)
        tmp.replace(outp)  # atomic
        total_frames += lat.shape[0]
        done += 1
        rate = total_frames / max(time.time() - t0, 1e-6)
        print(f"[{mi+1}/{len(matches)}] {d.name}: {tuple(lat.shape)} -> {outp.name} "
              f"| {total_frames:,} frames, {rate:.0f} f/s", flush=True)

    # Re-scan for the final tallies (covers resumed runs) and write INFO.
    all_pts = sorted(out.glob("*.pt"))
    grand_total = 0
    for p in all_pts:
        try:
            grand_total += int(torch.load(p, map_location="cpu", weights_only=True)["frame_indices"].shape[0])
        except Exception:
            pass
    write_info(out, args, cfg, step, len(all_pts), grand_total)
    print(f"DONE: {len(all_pts)} matches, {grand_total:,} frames -> {out}", flush=True)


if __name__ == "__main__":
    main()
