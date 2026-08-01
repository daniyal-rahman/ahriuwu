#!/usr/bin/env python3
"""IDM value test — is an inverse-dynamics model good enough to pseudo-label YT?

An IDM reads two consecutive latent frames (z_t, z_{t+1}) and predicts the action
the player took at t. If it recovers MOVEMENT well on held-out replays, we can run
it over the 454h of unlabeled YT to turn no-action video into action-conditioned
data (upgrading YT from no_action_embed to real movement conditioning). Abilities
are a bonus (sparse). Trains on replays (we have ground-truth actions), evaluates
on a held-out game.

    PYTHONPATH=<repo>/src python scripts/idm_value_test.py --latents-dir ... --labels-root ...
"""
import argparse
import glob
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "src")
from ahriuwu.constants import ABILITY_KEYS
from ahriuwu.data.replay_dataset import ReplayLatentSequenceDataset


class IDM(nn.Module):
    def __init__(self, c=32, d=256):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(2 * c, d, 3, 2, 1), nn.GELU(),      # 16->8
            nn.Conv2d(d, d, 3, 2, 1), nn.GELU(),          # 8->4
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.move = nn.Linear(d, 2)
        self.abil = nn.Linear(d, len(ABILITY_KEYS))

    def forward(self, z_t, z_n):
        h = self.enc(torch.cat([z_t, z_n], 1))
        return torch.sigmoid(self.move(h)), self.abil(h)


def build_pairs(ds, idxs):
    Zt, Zn, M, A = [], [], [], []
    for i in idxs:
        s = ds[i]
        lat = s["latents"].float()                        # (T,C,16,16)
        mv = s["actions"]["movement"].float()             # (T,2)
        ab = torch.stack([s["actions"][k].float() for k in ABILITY_KEYS], -1)  # (T,9)
        Zt.append(lat[:-1]); Zn.append(lat[1:]); M.append(mv[:-1]); A.append(ab[:-1])
    return (torch.cat(Zt), torch.cat(Zn), torch.cat(M), torch.cat(A))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents-dir", default="/mnt/nfs/datasets/replay_latents_v7_bc")
    ap.add_argument("--labels-root", default="/mnt/nfs/datasets/lol_replays_16_9_772")
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device

    mids = [p.split("/")[-1][:-3] for p in sorted(glob.glob(f"{args.latents_dir}/NA1_*.pt"))[:args.games]]
    outc = {m: False for m in mids}
    ds = ReplayLatentSequenceDataset(latents_dir=args.latents_dir, labels_root=args.labels_root,
                                     outcomes=outc, sequence_length=32, stride=32)
    # split sequences by game: last game held out
    test_game = mids[-1]
    tr = [i for i, s in enumerate(ds.sequences) if s["video_id"] != test_game]
    te = [i for i, s in enumerate(ds.sequences) if s["video_id"] == test_game]
    print(f"IDM value test: {len(mids)} games, train seq {len(tr)}, test seq {len(te)} (held-out {test_game})")
    Ztr, Zntr, Mtr, Atr = build_pairs(ds, tr)        # kept on CPU
    Zte, Znte, Mte, Ate = build_pairs(ds, te)
    print(f"pairs: train {Ztr.shape[0]}, test {Zte.shape[0]} (CPU-resident, minibatched to GPU)")

    net = IDM(c=Ztr.shape[1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    pw = torch.tensor([5.0] * len(ABILITY_KEYS), device=dev)     # up-weight rare casts
    bs = 256

    def eval_full():
        net.eval()
        pm_all, pa_all = [], []
        with torch.no_grad():
            for i in range(0, Zte.shape[0], 2048):
                pm, pa = net(Zte[i:i + 2048].to(dev), Znte[i:i + 2048].to(dev))
                pm_all.append(pm.cpu()); pa_all.append(pa.cpu())
        net.train()
        pm, pa = torch.cat(pm_all), torch.cat(pa_all)
        mae = (pm - Mte).abs().mean().item()
        base = (Mte - 0.5).abs().mean().item()
        binacc = (torch.round(pm * 20) == torch.round(Mte * 20)).all(1).float().mean().item()
        moved = (Mte - 0.5).abs().sum(1) > 0.02
        mae_moved = (pm - Mte).abs()[moved].mean().item() if moved.any() else float("nan")
        pred = torch.sigmoid(pa) > 0.5
        f1s = []
        for j in range(len(ABILITY_KEYS)):
            tp = (pred[:, j] & (Ate[:, j] > 0.5)).sum().item()
            fp = (pred[:, j] & (Ate[:, j] < 0.5)).sum().item()
            fn = (~pred[:, j] & (Ate[:, j] > 0.5)).sum().item()
            if tp + fn > 0:
                p = tp / (tp + fp) if tp + fp else 0.0
                r = tp / (tp + fn)
                f1s.append(2 * p * r / (p + r) if p + r else 0.0)
        return mae, base, binacc, mae_moved, (np.mean(f1s) if f1s else 0.0)

    t0 = time.time()
    for step in range(args.steps):
        idx = torch.randint(0, Ztr.shape[0], (bs,))
        pm, pa = net(Ztr[idx].to(dev), Zntr[idx].to(dev))
        loss = ((pm - Mtr[idx].to(dev)) ** 2).mean() + \
            nn.functional.binary_cross_entropy_with_logits(pa, Atr[idx].to(dev), pos_weight=pw)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 300 == 0 or step == args.steps - 1:
            mae, base, binacc, mae_moved, f1 = eval_full()
            print(f"  step {step:4d} loss={loss.item():.3f} | move MAE={mae:.3f} "
                  f"(center {base:.3f}) bin-acc={binacc:.1%} MAE|moved={mae_moved:.3f} | "
                  f"mean ability F1={f1:.2f}")
    dt = time.time() - t0
    verdict = ("WORTH IT (movement recoverable -> pseudo-label YT movement)"
               if mae_moved < 0.12 and binacc > 0.25
               else "MARGINAL — IDM movement weak; YT stays no_action_embed")
    print(f"\n{dt:.0f}s. VERDICT: {verdict}")
    print("(threshold: movement MAE|moved<0.12 and bin-acc>25% => IDM good enough to pseudo-label)")


if __name__ == "__main__":
    main()
