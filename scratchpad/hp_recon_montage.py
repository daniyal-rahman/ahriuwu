#!/usr/bin/env python3
"""Side-by-side GT | tokenizer-reconstruction stills for judging whether Garen's
floating HP bar (+ level chip) survives the v7 tokenizer. Rows span HP 1.0 -> 0.30
on the held-out match; each row = GT full | recon full | GT crop x3 | recon crop x3
(crop centered just above the champion, where the HP bar sits).

Run on the 5080 (bf16):
  ssh desktop 'cd /mnt/nfs/projects/ahriuwu && PYTHONPATH=src:scripts \
    /home/dani/miniconda3/envs/ml/bin/python scratchpad/hp_recon_montage.py'
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
from pretokenize_replay_v7 import load_v7

MATCH = "NA1_5549981347"
ROOT = Path("/mnt/nfs/datasets/lol_replays_16_9_772") / MATCH
CKPT = "rollout_stage/transformer_tokenizer_latest.pt"
OUT = Path("scratchpad/hp_recon_stills")
CROP = 120          # crop side in 352-space, centered above champ
ZOOM = 3
LABEL_SPACE = (1280.0, 720.0)  # champion_screen coordinate space


def pick_frames():
    lab = json.load(open(ROOT / "labels.json"))
    rows = []
    for f in lab["frames"]:
        l = f.get("label") or {}
        cs = l.get("champion_stats") or {}
        hp, hpm = cs.get("hp"), cs.get("hp_max")
        scr = l.get("champion_screen")
        if hp is not None and hpm and isinstance(scr, list):
            rows.append((f["frame"], hp / hpm, scr[0], scr[1]))
    a = np.array(rows)
    picks, seen = [], set()
    for tgt in (1.0, 0.8, 0.6, 0.45, 0.3, None):
        cand = a[a[:, 0] > 1500]  # skip fountain/loading
        i = int(cand[:, 1].argmin() if tgt is None else np.abs(cand[:, 1] - tgt).argmin())
        fr = int(cand[i, 0])
        if fr not in seen:
            seen.add(fr)
            picks.append((fr, float(cand[i, 1]), float(cand[i, 2]), float(cand[i, 3])))
    return picks


def main():
    dev = "cuda"
    model, cfg, step = load_v7(CKPT, dev)
    size = int(cfg.get("img_size", 352))
    print(f"tokenizer step {step}, img {size}")
    OUT.mkdir(parents=True, exist_ok=True)

    rows_img = []
    for fr, hp, sx, sy in pick_frames():
        im = cv2.imread(str(ROOT / "frames" / f"{fr:06d}.png"))
        assert im is not None, fr
        gt = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(gt).float().div_(255).permute(2, 0, 1)[None].to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            lat = model.encode(x)["latent"]
            rec = model.decode(lat, num_frames=1)[:, 0]
        rec = (rec.float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        cx = int(sx / LABEL_SPACE[0] * size)
        cy = int(sy / LABEL_SPACE[1] * size) - 15   # HP bar floats above the champ
        x0 = np.clip(cx - CROP // 2, 0, size - CROP)
        y0 = np.clip(cy - CROP // 2, 0, size - CROP)

        def crop(img):
            c = img[y0:y0 + CROP, x0:x0 + CROP]
            return cv2.resize(c, (CROP * ZOOM, CROP * ZOOM), interpolation=cv2.INTER_NEAREST)

        H = CROP * ZOOM
        cells = [cv2.resize(gt, (H, H)), cv2.resize(rec, (H, H)), crop(gt), crop(rec)]
        names = [f"GT f={fr} hp={hp:.2f}", "RECON", "GT zoom", "RECON zoom"]
        for c, n in zip(cells, names):
            cv2.putText(c, n, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)
        rows_img.append(np.concatenate(cells, axis=1))
        print(f"frame {fr}: hp={hp:.2f} crop=({x0},{y0})", flush=True)

    mont = np.concatenate(rows_img, axis=0)
    outp = OUT / "hp_recon_montage.png"
    cv2.imwrite(str(outp), cv2.cvtColor(mont, cv2.COLOR_RGB2BGR))
    print(f"wrote {outp} {mont.shape}")


if __name__ == "__main__":
    main()
