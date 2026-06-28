#!/usr/bin/env python3
"""Targeted rollout: reconstruct ONE specific 20-frame window with the v7 tokenizer.
Saves a side-by-side GT|recon animated GIF + a static multi-frame grid + PSNR.
Frames are read in sorted order from --frames-dir (put exactly the window there)."""
import argparse, glob
from pathlib import Path
import numpy as np, torch
from PIL import Image
from torch.amp import autocast
from ahriuwu.models import create_transformer_tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--frames-dir", required=True)   # contains exactly the window jpgs
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--seq-len", type=int, default=20)
    ap.add_argument("--label", default="clip")
    a = ap.parse_args()
    dev = "cuda"
    m = create_transformer_tokenizer("large", use_rope=True, gradient_checkpointing=False,
        latent_dim=16, num_latents=512, num_encoder_layers=8, num_decoder_layers=8, temporal_every=4).to(dev)
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    sd = {k.replace("_orig_mod.","").replace("module.",""): v for k,v in ck["model_state_dict"].items()}
    m.load_state_dict(sd); m.eval()

    files = sorted(glob.glob(f"{a.frames_dir}/*.jpg"))[:a.seq_len]
    imgs = [Image.open(f).convert("RGB") for f in files]
    arr = np.stack([np.asarray(im, dtype=np.float32)/255.0 for im in imgs])  # (T,H,W,C)
    frames = torch.from_numpy(arr).permute(0,3,1,2).unsqueeze(0).to(dev)     # (1,T,C,H,W)
    with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16):
        recon = m(frames, mask_indices=None)["reconstruction"].float().clamp(0,1)
    mse = ((frames-recon)**2).mean().item(); psnr = 10*np.log10(1/(mse+1e-8))

    gt = (frames[0].cpu().permute(0,2,3,1).numpy()*255).astype(np.uint8)     # (T,H,W,C)
    rc = (recon[0].cpu().permute(0,2,3,1).numpy()*255).astype(np.uint8)
    H = gt.shape[1]
    # side-by-side GT | recon frames (with a 4px divider)
    pair = [Image.fromarray(np.concatenate([gt[t], np.full((H,4,3),60,np.uint8), rc[t]],axis=1)) for t in range(len(gt))]
    gif = f"{a.out_prefix}_GTvsRECON.gif"
    pair[0].save(gif, save_all=True, append_images=pair[1:], duration=80, loop=0)
    # static grid: 5 evenly-spaced timesteps, GT row over recon row
    ts = np.linspace(0,len(gt)-1,5).astype(int)
    top = np.concatenate([gt[t] for t in ts],axis=1); bot = np.concatenate([rc[t] for t in ts],axis=1)
    Image.fromarray(np.concatenate([top,bot],axis=0)).save(f"{a.out_prefix}_grid.png")
    print(f"{a.label}: PSNR={psnr:.2f} dB  -> {gif}")


if __name__ == "__main__":
    main()
