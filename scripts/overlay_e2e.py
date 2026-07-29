#!/usr/bin/env python3
"""E2E overlay video: decode a replay segment and draw GROUND-TRUTH vs PREDICTED
actions on each frame. GREEN = human (logged), RED = agent (policy).

  movement : a target dot at the click/cursor location (green=human, red=agent argmax)
  ability  : text — human's actual cast, and the agent's TOP-logit ability + its prob
             (shown even below the 0.5 cast threshold, so you can see what it "wants")

    PYTHONPATH=src python scripts/overlay_e2e.py --start 2000 --frames 400 --out e2e.mp4
"""
import argparse
import glob
import os
import sys
import tempfile

import cv2
import numpy as np
import torch

sys.path.insert(0, "scripts")
from agent_infer import GarenAgent
from ahriuwu.constants import ABILITY_KEYS


def unfold(z):  # (1,32,16,16) -> (1,512,16) tokenizer layout for decode
    return z.permute(0, 2, 3, 1).reshape(1, 512, 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase2-ckpt", default="data/phase2_bc_garen/agent_finetune_latest.pt")
    ap.add_argument("--tokenizer-ckpt", default="rollout_stage/transformer_tokenizer_latest.pt")
    ap.add_argument("--match", default="NA1_5549995114")
    ap.add_argument("--latents-dir", default="rollout_stage")
    ap.add_argument("--labels-root", default="/srv/nfs/datasets/lol_replays_16_9_772")
    ap.add_argument("--start", type=int, default=2000)
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--context", type=int, default=16)
    ap.add_argument("--frames-dir", default=None,
                    help="Dir of ORIGINAL sharp PNGs (frames/, named <frame>.png). If given, "
                         "render on those instead of the (blurry) tokenizer decode.")
    ap.add_argument("--out", default="e2e_overlay.mp4")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    from ahriuwu.data.replay_dataset import ReplayLatentSequenceDataset

    tmp = tempfile.mkdtemp()
    os.symlink(os.path.abspath(glob.glob(f"{args.latents_dir}/{args.match}.pt")[0]), f"{tmp}/{args.match}.pt")
    tot = args.start + args.frames + 2
    ds = ReplayLatentSequenceDataset(latents_dir=tmp, labels_root=args.labels_root,
                                     outcomes={args.match: False}, sequence_length=tot, stride=tot)
    s = ds[0]
    lat = s["latents"].float()
    gm = s["actions"]["movement"].float().numpy()
    ga = torch.stack([s["actions"][k].float() for k in ABILITY_KEYS], -1).numpy().astype(bool)
    start_idx = ds.sequences[0]["start_idx"]
    fidx = torch.load(glob.glob(f"{args.latents_dir}/{args.match}.pt")[0],
                      map_location="cpu", weights_only=True)["frame_indices"].numpy()

    ag = GarenAgent(args.phase2_ckpt, tokenizer_ckpt=args.tokenizer_ckpt, context=args.context, device=dev)
    ag.reset()
    for t in range(max(0, args.start - args.context + 1), args.start):   # warm the context buffer
        ag.buf.append(lat[t:t + 1].to(dev))

    W = H = 512
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    n_written = 0
    for t in range(args.start, args.start + args.frames):
        img = None
        if args.frames_dir:
            fnum = int(fidx[start_idx + t])
            img = cv2.imread(f"{args.frames_dir}/{fnum:06d}.png")     # sharp original (BGR)
        if img is None:                                              # decode fallback (soft)
            with torch.no_grad(), ag._ac():
                rec = ag.tok.decode(unfold(lat[t:t + 1].to(dev)), 1)
            img = cv2.cvtColor((rec.squeeze().permute(1, 2, 0).clamp(0, 1).float().cpu().numpy() * 255)
                               .astype(np.uint8), cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (W, H))

        ag.buf.append(lat[t:t + 1].to(dev))
        w = list(ag.buf)
        while len(w) < ag.context:
            w.insert(0, w[0])
        z0 = torch.stack(w, dim=1).squeeze(2).to(dev).float()
        B, T = z0.shape[:2]
        tau = ag.tau_ctx + torch.rand(B, T, device=dev) * (1 - ag.tau_ctx)
        z_tau, _ = ag.sched.add_noise(z0, tau)
        d1 = torch.ones(B, dtype=torch.long, device=dev)
        with torch.no_grad(), ag._ac():
            _, agent_out = ag.dyn(z_tau, tau, step_size=d1, actions=None)
            al, ml = ag.policy(agent_out[:, -1:, :])
        noff = 1 if ag.mtp > 1 else 0
        al = al[0, 0, noff, :].float().cpu().numpy()
        ml = ml[0, 0, noff].float().cpu().numpy()
        pmx, pmy = ml[0].argmax() / (ml.shape[1] - 1), ml[1].argmax() / (ml.shape[1] - 1)
        gx, gy = gm[t + 1]
        gab = [ABILITY_KEYS[i] for i in range(len(ABILITY_KEYS)) if ga[t + 1, i]]
        topi = int(al.argmax()); topp = 1 / (1 + np.exp(-al[topi]))

        cv2.circle(img, (int(gx * W), int(gy * H)), 11, (0, 220, 0), 2)      # human = green ring
        cv2.circle(img, (int(pmx * W), int(pmy * H)), 7, (0, 0, 235), -1)    # agent = red dot
        cv2.putText(img, f"HUMAN move({gx:.2f},{gy:.2f}) cast:{','.join(gab) or '-'}",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1, cv2.LINE_AA)
        cv2.putText(img, f"AGENT move({pmx:.2f},{pmy:.2f}) top:{ABILITY_KEYS[topi]} p={topp:.2f}",
                    (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 235), 1, cv2.LINE_AA)
        cv2.putText(img, f"frame {t}", (8, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        vw.write(img)
        n_written += 1
    vw.release()
    print(f"wrote {args.out}: {n_written} frames @ {args.fps}fps ({args.match} {args.start}-{args.start+args.frames})")


if __name__ == "__main__":
    main()
