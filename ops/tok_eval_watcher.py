#!/usr/bin/env python3
"""Fixed held-out eval for the v7-yt plateau test (runs on the LOGIN 1060, fp32).

Polls data/tokenizer_v7_yt/transformer_tokenizer_latest.pt; on a new global_step,
evals m=0 single-frame reconstruction PSNR on two FIXED sets:
  - replay: 400 frames from held-out matches (full-frame PSNR — continuity with
    the historical ~29 dB numbers)
  - yt: 400 frames from 4 never-trained YT games (PSNR on non-constant-black
    pixels via the per-set black mask, plus full-frame for reference)
Appends scratchpad/tok_yt_eval.csv. The null slope to beat: v7c's +0.03 dB/day.
Single-frame protocol mirrors pretokenize's recon_psnr (known-good path, tiny
VRAM — coexists with the 1060 BC job). First row evals the gs6000 NFS baseline.

  setsid nohup /home/dani/miniconda3/envs/ml/bin/python scratchpad/tok_eval_watcher.py \
      > scratchpad/tok_eval_watcher.log 2>&1 &
"""
import csv
import glob
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from pretokenize_replay_v7 import load_v7

CKPT = Path("data/tokenizer_v7_yt/transformer_tokenizer_latest.pt")
BASELINE = "rollout_stage/transformer_tokenizer_latest.pt"     # v7 gs6000 anchor
CSV = Path("scratchpad/tok_yt_eval.csv")
REPLAY_MATCHES = ["NA1_5549981347", "NA1_5550450386", "NA1_5551132630"]  # held-out
REPLAY_ROOT = "/srv/nfs/datasets/lol_replays_16_9_772"
YT_EVAL_ROOT = "/srv/nfs/datasets/yt_eval_frames_352"
N_PER_SET, BATCH, POLL_S = 400, 4, 300


def fixed_frames():
    torch.manual_seed(0)
    rep = []
    for m in REPLAY_MATCHES:
        fs = sorted(glob.glob(f"{REPLAY_ROOT}/{m}/frames/*.png"))
        idx = np.linspace(2000, len(fs) - 1, N_PER_SET // len(REPLAY_MATCHES)).astype(int)
        rep += [fs[i] for i in idx]
    yts = []
    vids = sorted(glob.glob(f"{YT_EVAL_ROOT}/yt_*"))
    for v in vids:
        fs = sorted(glob.glob(f"{v}/*.jpg")) or sorted(glob.glob(f"{v}/*.png"))
        if not fs:
            continue
        idx = np.linspace(len(fs) // 10, len(fs) - 1, N_PER_SET // max(len(vids), 1)).astype(int)
        yts += [fs[i] for i in idx]
    return rep, yts


def load_imgs(paths):
    out = []
    for p in paths:
        im = cv2.imread(p)
        if im is None:
            continue
        if im.shape[:2] != (352, 352):
            im = cv2.resize(im, (352, 352), interpolation=cv2.INTER_AREA)
        out.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    x = torch.from_numpy(np.stack(out)).float().div_(255).permute(0, 3, 1, 2)
    return x  # (N,3,352,352)


@torch.no_grad()
def eval_ckpt(path, xr, xy, yt_black):
    model, cfg, step = load_v7(str(path), "cuda")
    model = model.float()

    def recon(x):
        outs = []
        for i in range(0, x.shape[0], BATCH):
            b = x[i:i + BATCH].cuda()
            lat = model.encode(b)["latent"]
            r = model.decode(lat, num_frames=1)
            r = r[:, 0] if r.dim() == 5 else r
            outs.append(r.float().clamp(0, 1).cpu())
        return torch.cat(outs)

    def psnr(a, b, mask=None):
        se = (a - b) ** 2
        if mask is not None:
            mse = (se * mask).sum() / (mask.sum() * 3 * a.shape[0]).clamp_min(1)
        else:
            mse = se.mean()
        return float(10 * np.log10(1.0 / max(mse.item(), 1e-10)))

    rr = recon(xr)
    ry = recon(xy)
    res = {
        "step": step,
        "psnr_replay": psnr(rr, xr),
        "psnr_yt_valid": psnr(ry, xy, yt_black),
        "psnr_yt_full": psnr(ry, xy),
    }
    del model
    torch.cuda.empty_cache()
    return res


def main():
    os.chdir("/srv/nfs/projects/ahriuwu")
    while True:
        rep_paths, yt_paths = fixed_frames()
        if len(yt_paths) >= 100:                 # staging extracts eval tars last
            break
        print(f"waiting for YT eval frames ({len(yt_paths)} so far)...", flush=True)
        time.sleep(120)
    print(f"fixed sets: {len(rep_paths)} replay / {len(yt_paths)} yt frames", flush=True)
    xr, xy = load_imgs(rep_paths), load_imgs(yt_paths)
    # constant-black mask over the YT SET (pixel dark in >=95% of frames = HUD black)
    dark = (xy.max(dim=1).values <= 0.02).float().mean(0)          # (H,W)
    yt_black = (dark < 0.95).float()[None, None]                    # 1=valid
    print(f"yt blacked-out fraction: {(1 - yt_black).mean():.1%}", flush=True)

    new = not CSV.exists()
    f = open(CSV, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(["time", "ckpt", "step", "psnr_replay", "psnr_yt_valid", "psnr_yt_full"])
        r = eval_ckpt(BASELINE, xr, xy, yt_black)
        w.writerow([time.strftime("%m-%d %H:%M"), "v7-gs6000-baseline", r["step"],
                    f"{r['psnr_replay']:.3f}", f"{r['psnr_yt_valid']:.3f}", f"{r['psnr_yt_full']:.3f}"])
        f.flush()
        print(f"baseline gs{r['step']}: replay={r['psnr_replay']:.2f} yt_valid={r['psnr_yt_valid']:.2f}", flush=True)

    last_step = -1
    while True:
        if CKPT.exists():
            try:
                r = eval_ckpt(CKPT, xr, xy, yt_black)
                if r["step"] != last_step:
                    last_step = r["step"]
                    w.writerow([time.strftime("%m-%d %H:%M"), "v7-yt", r["step"],
                                f"{r['psnr_replay']:.3f}", f"{r['psnr_yt_valid']:.3f}", f"{r['psnr_yt_full']:.3f}"])
                    f.flush()
                    print(f"step {r['step']}: replay={r['psnr_replay']:.2f} "
                          f"yt_valid={r['psnr_yt_valid']:.2f} yt_full={r['psnr_yt_full']:.2f}", flush=True)
            except Exception as e:
                print(f"eval failed ({type(e).__name__}: {e}); retrying next poll", flush=True)
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
