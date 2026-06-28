#!/usr/bin/env python3
"""Before/after rollout for the v7 transformer tokenizer.

Reconstructs a fixed set of held-out clips (full-frame, no masking) with a given
checkpoint and saves GT-vs-recon image grids + a PSNR summary. Run it twice on the
SAME --frames-dir (once per checkpoint) to get a before/after comparison.
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.amp import autocast
from ahriuwu.models import create_transformer_tokenizer
from ahriuwu.data.dataset import FrameSequenceDataset


def load_model(ckpt_path, device):
    # v7 architecture (resume-critical, must match training)
    m = create_transformer_tokenizer(
        "large", use_rope=True, gradient_checkpointing=False,
        latent_dim=16, num_latents=512,
        num_encoder_layers=8, num_decoder_layers=8, temporal_every=4,
    ).to(device)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sd.items()}
    m.load_state_dict(sd)
    m.eval()
    return m, ck.get("global_step", "?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num-clips", type=int, default=6)
    ap.add_argument("--seq-len", type=int, default=20)
    ap.add_argument("--file-ext", default="jpg")
    ap.add_argument("--label", default="rollout")
    a = ap.parse_args()
    device = "cuda"
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    m, step = load_model(a.checkpoint, device)
    ds = FrameSequenceDataset(a.frames_dir, sequence_length=a.seq_len, stride=a.seq_len,
                              file_ext=a.file_ext, skip_resize=True, augment=False)
    n = len(ds)
    if n == 0:
        raise SystemExit(f"no clips in {a.frames_dir}")
    idxs = np.linspace(0, n - 1, min(a.num_clips, n)).astype(int)

    psnrs = []
    with torch.no_grad():
        for i, idx in enumerate(idxs):
            frames = ds[int(idx)]["frames"].unsqueeze(0).to(device)  # (1,T,C,H,W)
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                recon = m(frames, mask_indices=None)["reconstruction"].float()
            mse = ((frames.float() - recon) ** 2).mean().item()
            psnr = 10 * np.log10(1.0 / (mse + 1e-8)); psnrs.append(psnr)
            T = frames.shape[1]
            rows = []
            for t in [0, T // 2, T - 1]:
                gt = (frames[0, t].cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                rc = (recon[0, t].cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                rows.append(np.concatenate([gt, rc], axis=0))      # GT over recon
            Image.fromarray(np.concatenate(rows, axis=1)).save(out / f"{a.label}_clip{i}_psnr{psnr:.1f}.png")
            print(f"clip {i} (idx {idx}): PSNR={psnr:.2f} dB", flush=True)

    summary = {"label": a.label, "checkpoint": a.checkpoint, "step": step,
               "mean_psnr": float(np.mean(psnrs)), "psnrs": [float(p) for p in psnrs]}
    (out / f"{a.label}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{a.label}: mean PSNR={np.mean(psnrs):.2f} dB over {len(psnrs)} clips (step={step})")


if __name__ == "__main__":
    main()
