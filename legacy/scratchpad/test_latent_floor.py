#!/usr/bin/env python3
"""Two WM-independent tests of the latent space.

TEST C — predictability floor: ridge from PCA features of the past k latents ->
  next latent (full 8192-d), PSNR in the eval's units, vs persistence. Trained on
  4 games, evaluated on a HELD-OUT game (no in-sample advantage). If context
  lifts prediction well above persistence, the latents ARE predictable and the
  'noise floor' story is wrong. k=1 vs k=4 shows how much history helps linearly.

TEST D — encoder jitter: for consecutive frame pairs, compare PIXEL delta
  (352x352 grayscale RMS) vs LATENT delta. Report latent delta in the lowest
  pixel-delta decile (near-static frames). If near-identical frames still get
  large latent jumps, encoder jitter is real (measured, not assumed).
"""
import glob
import sys

import numpy as np
import torch

LATDIR = "/srv/nfs/datasets/replay_latents_v7_bc"
FRROOT = "/srv/nfs/datasets/lol_replays_16_9_772"
GAMES = sorted(glob.glob(f"{LATDIR}/NA1_*.pt"))[:5]
STEP = 2


def load(pt):
    d = torch.load(pt, weights_only=True)
    return d["latents"].float(), d["frame_indices"].numpy()


def psnr(a, b, mx):
    mse = ((a - b) ** 2).mean().item()
    return 10 * np.log10(mx ** 2 / max(mse, 1e-10))


def main():
    print("== TEST C: cross-game latent predictability (linear, with context) ==")
    zs = [load(p)[0][::STEP] for p in GAMES]
    mx = max(z.abs().max().item() for z in zs)
    # PCA basis from train games only
    Xtr_full = torch.cat(zs[:4]).reshape(-1, 8192)
    mu = Xtr_full.mean(0)
    _, _, V = torch.pca_lowrank(Xtr_full - mu, q=256, niter=4)

    def feats(z, k):
        F = (z.reshape(len(z), -1) - mu) @ V                      # (N,256)
        cols = [F[i:len(F) - k + i] for i in range(k)]            # k past frames
        return torch.cat(cols, 1)                                 # (N-k, 256k)

    te = zs[4].reshape(len(zs[4]), -1)
    print(f"train 4 games ({Xtr_full.shape[0]} fr), test {GAMES[4].split('/')[-1]} ({len(te)} fr), stride {STEP}")
    print(f"  persistence     : {psnr(te[:-1], te[1:], mx):.1f} dB")
    for k in (1, 4):
        Xtr = torch.cat([feats(z, k) for z in zs[:4]])
        Ytr = torch.cat([z.reshape(len(z), -1)[k:] for z in zs[:4]])
        W = torch.linalg.solve(Xtr.T @ Xtr + 10.0 * torch.eye(Xtr.shape[1]), Xtr.T @ Ytr)
        Xte = feats(zs[4], k)
        pred = Xte @ W
        print(f"  ridge ctx k={k}    : {psnr(pred, te[k:], mx):.1f} dB")
    # persistence + linear-delta hybrid: predict the CHANGE from context
    for k in (4,):
        Xtr = torch.cat([feats(z, k) for z in zs[:4]])
        Ytr = torch.cat([(z.reshape(len(z), -1)[k:] - z.reshape(len(z), -1)[k - 1:-1]) for z in zs[:4]])
        W = torch.linalg.solve(Xtr.T @ Xtr + 10.0 * torch.eye(Xtr.shape[1]), Xtr.T @ Ytr)
        pred = te[k - 1:-1] + feats(zs[4], k) @ W
        print(f"  persist+Δridge k={k}: {psnr(pred, te[k:], mx):.1f} dB")
    print("  (WM sampled h1 = 21.6 dB, for reference. delta-at-stride-2 doubles the gap vs stride-1 numbers.)")

    print("\n== TEST D: encoder jitter on near-static frames ==")
    import cv2
    z, fi = load(GAMES[0])
    zf = z.reshape(len(z), -1)
    fdir = f"{FRROOT}/{GAMES[0].split('/')[-1][:-3]}/frames"
    idx = np.linspace(0, len(fi) - 2, 400).astype(int)             # 400 consecutive pairs
    pix, lat = [], []
    for i in idx:
        a = cv2.imread(f"{fdir}/{int(fi[i]):06d}.png", cv2.IMREAD_GRAYSCALE)
        b = cv2.imread(f"{fdir}/{int(fi[i + 1]):06d}.png", cv2.IMREAD_GRAYSCALE)
        if a is None or b is None:
            continue
        a = cv2.resize(a, (352, 352)).astype(np.float32) / 255
        b = cv2.resize(b, (352, 352)).astype(np.float32) / 255
        pix.append(float(np.sqrt(((a - b) ** 2).mean())))
        lat.append(float((zf[i + 1] - zf[i]).pow(2).mean().sqrt()))
    pix, lat = np.array(pix), np.array(lat)
    print(f"pairs={len(pix)}  corr(pixelΔ, latentΔ)={np.corrcoef(pix, lat)[0, 1]:.2f}")
    q = np.quantile(pix, [0.1, 0.5])
    lo, hi = lat[pix <= q[0]], lat[pix >= q[1]]
    print(f"latentΔ RMS | near-static pairs (lowest 10% pixelΔ, pixΔ<={q[0]:.4f}): {lo.mean():.4f}")
    print(f"latentΔ RMS | median-motion pairs                                  : {lat[(pix >= q[0]) & (pix <= q[1])].mean():.4f}")
    print(f"latentΔ RMS | high-motion  pairs (top half)                        : {hi.mean():.4f}")
    print(f"latent RMS (signal scale)                                          : {zf.pow(2).mean().sqrt():.4f}")
    print("read: near-static >> 0 and comparable to motion pairs => encoder jitter real;")
    print("      near-static ~ 0 => encoder temporally consistent, jitter story dead.")


if __name__ == "__main__":
    main()
