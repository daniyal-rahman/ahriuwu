#!/usr/bin/env python3
"""Encode the held-out laning-phase frame slice to v7 latents on the 1060 (fp32 —
Pascal has no bf16, and pretokenize_replay_v7 hardcodes bf16 autocast). Reuses the
exact-config tokenizer load + output format from pretokenize_replay_v7."""
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
from pretokenize_replay_v7 import load_v7, match_pngs, _FrameDS

MATCH = "NA1_5549981347"
PNGDIR = Path(f"/srv/nfs/datasets/lol_replays_16_9_772_heldout_slice/{MATCH}/frames")
OUT = Path("/srv/nfs/datasets/replay_latents_v7_heldout")
BATCH = 4

def main():
    device = "cuda"
    model, cfg, step = load_v7("rollout_stage/transformer_tokenizer_latest.pt", device)
    size = int(cfg.get("img_size", 352))
    pngs = match_pngs(PNGDIR)
    print(f"{MATCH}: {len(pngs)} pngs, tokenizer step {step}, fp32 batch {BATCH}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    dl = DataLoader(_FrameDS(pngs, size), batch_size=BATCH, num_workers=4, pin_memory=True)
    out, t0, n = [], time.time(), 0
    with torch.no_grad():
        for batch in dl:
            lat = model.encode(batch.to(device, non_blocking=True))["latent"]
            B = lat.shape[0]
            out.append(lat.view(B, 16, 16, -1).permute(0, 3, 1, 2).cpu().to(torch.float16))
            n += B
            if n % 200 < BATCH:
                print(f"  {n}/{len(pngs)} ({n/(time.time()-t0):.1f} f/s)", flush=True)
    lat = torch.cat(out, 0)
    idxs = torch.tensor([int(p.stem) for p in pngs], dtype=torch.int32)
    tmp = OUT / f"{MATCH}.pt.tmp"
    torch.save({"latents": lat, "frame_indices": idxs}, tmp)
    tmp.replace(OUT / f"{MATCH}.pt")
    print(f"DONE {tuple(lat.shape)} -> {OUT}/{MATCH}.pt ({len(pngs)/(time.time()-t0):.1f} f/s)", flush=True)

if __name__ == "__main__":
    main()
