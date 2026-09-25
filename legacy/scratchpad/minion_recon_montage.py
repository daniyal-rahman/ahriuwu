#!/usr/bin/env python3
"""Expert-review risk test (i): are MINION HP bars legible in v7 reconstructions?
Laning-phase frames from the held-out game, wide crop around the wave, 2x zoom.

  ssh desktop 'cd /mnt/nfs/projects/ahriuwu && PYTHONPATH=src:scripts \
    /home/dani/miniconda3/envs/ml/bin/python scratchpad/minion_recon_montage.py'
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
FRAMES = [2400, 2900, 3400, 3900, 4400, 4800]   # laning phase, wave on screen
CROP, ZOOM = 176, 2
LABEL_SPACE = (1280.0, 720.0)


def main():
    dev = "cuda"
    model, cfg, step = load_v7("rollout_stage/transformer_tokenizer_latest.pt", dev)
    size = int(cfg.get("img_size", 352))
    lab = json.load(open(ROOT / "labels.json"))
    pos = {}
    for f in lab["frames"]:
        l = f.get("label") or {}
        scr = l.get("champion_screen")
        if isinstance(scr, list):
            pos[f["frame"]] = scr
    out_rows = []
    for fr in FRAMES:
        im = cv2.imread(str(ROOT / "frames" / f"{fr:06d}.png"))
        gt = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(gt).float().div_(255).permute(2, 0, 1)[None].to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            rec = model.decode(model.encode(x)["latent"], num_frames=1)[:, 0]
        rec = (rec.float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        sx, sy = pos.get(fr, (640, 360))
        cx = int(sx / LABEL_SPACE[0] * size)
        cy = int(sy / LABEL_SPACE[1] * size)
        x0 = int(np.clip(cx - CROP // 2, 0, size - CROP))
        y0 = int(np.clip(cy - CROP // 2, 0, size - CROP))

        def crop(img):
            c = img[y0:y0 + CROP, x0:x0 + CROP]
            return cv2.resize(c, (CROP * ZOOM, CROP * ZOOM), interpolation=cv2.INTER_NEAREST)

        cells = [crop(gt), crop(rec)]
        for c, n in zip(cells, [f"GT wave f={fr}", "RECON"]):
            cv2.putText(c, n, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)
        out_rows.append(np.concatenate(cells, axis=1))
    mont = np.concatenate(out_rows, axis=0)
    outp = Path("scratchpad/hp_recon_stills/minion_recon_montage.png")
    cv2.imwrite(str(outp), cv2.cvtColor(mont, cv2.COLOR_RGB2BGR))
    print(f"wrote {outp} {mont.shape}")


if __name__ == "__main__":
    main()
