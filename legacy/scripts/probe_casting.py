#!/usr/bin/env python3
"""Casting calibration probe — the decisive test for the ability weakness.

For each ability, collect the policy's raw pre-sigmoid LOGIT at every frame and ask:
does the logit at real human-cast frames stand ABOVE the logit elsewhere?
  AUC(logit -> is-cast) >> 0.5  => the signal IS there, it's a threshold/calibration
                                    bug (cheap fix: per-ability threshold / focal loss).
  AUC ~ 0.5                     => the head never learned cast timing (needs data/arch).

Also reports logit@cast vs logit@non-cast and where a threshold would need to sit.

    PYTHONPATH=src python scripts/probe_casting.py --frames 8000 --labels-root <...>
"""
import argparse
import glob
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, "scripts")
from agent_infer import GarenAgent
from ahriuwu.constants import ABILITY_KEYS
from ahriuwu.data.replay_dataset import ReplayLatentSequenceDataset


def auc(scores, labels):
    """Rank-based (Mann-Whitney) AUC; labels bool."""
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.concatenate([pos, neg]).argsort(kind="mergesort")
    ranks = np.empty(len(order), float)
    ranks[order] = np.arange(1, len(order) + 1)
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase2-ckpt", default="data/phase2_bc_garen/agent_finetune_latest.pt")
    ap.add_argument("--match", default="NA1_5549995114")
    ap.add_argument("--latents-dir", default="rollout_stage")
    ap.add_argument("--labels-root", default="/srv/nfs/datasets/lol_replays_16_9_772")
    ap.add_argument("--frames", type=int, default=8000)
    ap.add_argument("--context", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device

    tmp = tempfile.mkdtemp()
    os.symlink(os.path.abspath(glob.glob(f"{args.latents_dir}/{args.match}.pt")[0]), f"{tmp}/{args.match}.pt")
    ds = ReplayLatentSequenceDataset(latents_dir=tmp, labels_root=args.labels_root,
                                     outcomes={args.match: False},
                                     sequence_length=args.frames + 1, stride=args.frames + 1)
    s = ds[0]
    lat = s["latents"].float()
    N = min(args.frames, lat.shape[0] - 1)
    abil_gt = torch.stack([s["actions"][k].float() for k in ABILITY_KEYS], -1).numpy().astype(bool)

    ag = GarenAgent(args.phase2_ckpt, tokenizer_ckpt=None, context=args.context, device=dev)
    ag.reset()
    logits = np.zeros((N, len(ABILITY_KEYS)), np.float32)
    for t in range(N):
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
            a_logits, _ = ag.policy(agent_out[:, -1:, :])       # (1,1,L,A)
        n = 1 if ag.mtp > 1 else 0
        logits[t] = a_logits[0, 0, n, :].float().cpu().numpy()

    ga = abil_gt[1:N + 1]                                        # next-action GT (BC offset 1)
    print(f"\nprobe: {args.match}, {N} frames, ctx={args.context}")
    print("ability | human | logit@cast | logit@else |  AUC  | verdict")
    for i, k in enumerate(ABILITY_KEYS):
        c = ga[:, i]
        sup = int(c.sum())
        if sup == 0:
            print(f"  {k:7s} {sup:5d}   (no in-range casts — no support)")
            continue
        lc, le = logits[c, i].mean(), logits[~c, i].mean()
        a = auc(logits[:, i], c)
        v = "SIGNAL (threshold fix)" if a >= 0.70 else ("weak signal" if a >= 0.58 else "NO SIGNAL")
        print(f"  {k:7s} {sup:5d}   {lc:+7.2f}   {le:+7.2f}   {a:.3f}  {v}")
    print("\nAUC >> 0.5 -> the cast signal exists, it's a calibration/threshold problem.")


if __name__ == "__main__":
    main()
