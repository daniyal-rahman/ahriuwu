#!/usr/bin/env python3
"""Quantitative held-out eval of v7 tokenizer checkpoints (run on danilogin).

Reconstructs N evenly-sampled full-frame clips (no masking) per checkpoint over the
SAME clip indices for every checkpoint, and reports PSNR (+ MSE, and SSIM/LPIPS if
those libs are importable) as mean/std/median/min/max, plus pairwise deltas.

Usage:
  python scripts/eval_tokenizer.py --checkpoints seed=/p/seed.pt run2=/p/run2.pt run3=/p/run3.pt \
    --frames-dir /mnt/nfs/datasets/ahriuwu_yt_holdout --num-clips 200 --out eval.json
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from torch.amp import autocast
from ahriuwu.models import create_transformer_tokenizer
from ahriuwu.data.dataset import FrameSequenceDataset


def load_model(ckpt, device):
    m = create_transformer_tokenizer(
        "large", use_rope=True, gradient_checkpointing=False,
        latent_dim=16, num_latents=512, num_encoder_layers=8, num_decoder_layers=8, temporal_every=4,
    ).to(device)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in ck["model_state_dict"].items()}
    m.load_state_dict(sd); m.eval()
    return m, ck.get("global_step", "?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True, help="name=path pairs")
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--num-clips", type=int, default=200)
    ap.add_argument("--seq-len", type=int, default=20)
    ap.add_argument("--file-ext", default="jpg")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="/mnt/nfs/checkpoints/ahriuwu_run3/eval_holdout.json")
    a = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    use_bf16 = device == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8
    # optional perceptual metrics (graceful if absent)
    try:
        import lpips as _lp; lpips_fn = _lp.LPIPS(net="alex").to(device).eval(); have_lpips = True
    except Exception as e:
        lpips_fn = None; have_lpips = False; print("LPIPS unavailable:", e, flush=True)
    try:
        from skimage.metrics import structural_similarity as ssim_fn; have_ssim = True
    except Exception as e:
        ssim_fn = None; have_ssim = False; print("SSIM unavailable:", e, flush=True)

    ds = FrameSequenceDataset(a.frames_dir, sequence_length=a.seq_len, stride=a.seq_len,
                              file_ext=a.file_ext, skip_resize=True, augment=False)
    n = len(ds)
    if n == 0:
        raise SystemExit(f"no clips in {a.frames_dir}")
    idxs = np.linspace(0, n - 1, min(a.num_clips, n)).astype(int)
    print(f"dataset n={n}  eval clips={len(idxs)}  device={device}  bf16={use_bf16} "
          f"ssim={have_ssim} lpips={have_lpips}", flush=True)

    cks = [kv.split("=", 1) for kv in a.checkpoints]
    results = {}
    for name, path in cks:
        m, step = load_model(path, device)
        psnrs, mses, ssims, lps = [], [], [], []
        t0 = time.time()
        with torch.no_grad():
            for j, idx in enumerate(idxs):
                frames = ds[int(idx)]["frames"].unsqueeze(0).to(device)  # (1,T,C,H,W)
                if use_bf16:
                    with autocast(device_type="cuda", dtype=torch.bfloat16):
                        recon = m(frames, mask_indices=None)["reconstruction"].float()
                else:
                    recon = m(frames, mask_indices=None)["reconstruction"].float()
                f = frames.float(); r = recon.clamp(0, 1)
                mse = ((f - r) ** 2).mean().item(); mses.append(mse)
                psnrs.append(10 * np.log10(1.0 / (mse + 1e-8)))
                T = f.shape[1]; sub = sorted(set([0, T // 2, T - 1]))  # subsample frames for perceptual
                if have_ssim:
                    fs = f[0].cpu().numpy(); rs = r[0].cpu().numpy()
                    ssims.append(float(np.mean([ssim_fn(fs[t].transpose(1, 2, 0), rs[t].transpose(1, 2, 0),
                                                        channel_axis=2, data_range=1.0) for t in sub])))
                if have_lpips:
                    fl = f[0, sub] * 2 - 1; rl = r[0, sub] * 2 - 1
                    lps.append(lpips_fn(fl, rl).mean().item())
                if (j + 1) % 25 == 0:
                    print(f"  {name}: {j+1}/{len(idxs)} running PSNR={np.mean(psnrs):.3f}", flush=True)
        agg = {"step": step, "n_clips": len(psnrs),
               "psnr_mean": float(np.mean(psnrs)), "psnr_std": float(np.std(psnrs)),
               "psnr_median": float(np.median(psnrs)), "psnr_min": float(np.min(psnrs)),
               "psnr_max": float(np.max(psnrs)), "mse_mean": float(np.mean(mses)),
               "sec": round(time.time() - t0, 1)}
        if have_ssim: agg["ssim_mean"] = float(np.mean(ssims))
        if have_lpips: agg["lpips_mean"] = float(np.mean(lps))
        agg["psnrs"] = [float(p) for p in psnrs]  # per-clip, for paired deltas
        results[name] = agg
        line = (f"== {name} (step {step}): PSNR {agg['psnr_mean']:.3f}±{agg['psnr_std']:.3f} dB "
                f"(med {agg['psnr_median']:.2f})")
        if have_ssim: line += f"  SSIM {agg['ssim_mean']:.4f}"
        if have_lpips: line += f"  LPIPS {agg['lpips_mean']:.4f}"
        print(line + f"  [{agg['sec']}s]", flush=True)
        del m
        if device == "cuda": torch.cuda.empty_cache()

    out = {"frames_dir": a.frames_dir, "n_dataset": n, "num_clips": len(idxs),
           "clip_indices": [int(i) for i in idxs], "device": device, "checkpoints": results}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
