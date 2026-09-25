#!/usr/bin/env python3
"""Dream-quality evaluation — the paper's actual bar (distributional), not PSNR-to-GT.

Runs all three measurements from the diagnosis:
 1. LONG DREAMS: generate `--dream-frames` latents from real context at ctx=16 and
    ctx=224 (paper protocol scaled to our 256 cap), decode to mp4 side by side with
    ground truth.
 2. POOR-MAN'S FVD: embed 16-frame chunks of dream vs real latents (features =
    per-chunk latent stats: spatial-mean/std pooled over 16 frames) and compute a
    Frechet distance between the two feature distributions. Lower = dreams live on
    the real-gameplay manifold. Reported per context length + for a real-vs-real
    control split (the floor).
 3. SHARPNESS-OVER-HORIZON: decoded-frame Laplacian variance vs horizon, alongside
    the same for ground-truth frames. Distinguishes sharp-but-divergent (healthy
    stochastic WM) from blur-collapse (conditional-mean disease).

    PYTHONPATH=src python scripts/eval_dream_quality.py \
        --ckpt <dynamics.pt> --tokenizer-ckpt <tok.pt> --match NA1_5550067582
"""
import argparse
import contextlib
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")


def load_models(ckpt_path, tok_path, dev, model_size="medium", use_actions=True):
    """Raw dynamics + tokenizer load, mirroring rollout_check.py."""
    from ahriuwu.models.dynamics import create_dynamics
    from ahriuwu.models.transformer_tokenizer import TransformerTokenizer
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"dynamics ckpt: global_step={ck.get('global_step')} epoch={ck.get('epoch')}")
    net = create_dynamics(model_size, latent_dim=32, use_actions=use_actions, num_kv_heads=4,
                          num_register_tokens=8, soft_cap=50.0, use_qk_norm=True).to(dev).eval()
    sd = ck["model_state_dict"]
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    miss, unexp = net.load_state_dict(sd, strict=False)
    miss = [m for m in miss if "rope" not in m.lower()]
    assert len(miss) + len(unexp) <= 10, f"ARCH MISMATCH miss={miss[:5]} unexp={unexp[:5]}"

    tk = torch.load(tok_path, map_location="cpu", weights_only=False)
    cfg = {k: v for k, v in tk["model_config"].items() if k != "size_preset"}
    tok = TransformerTokenizer(**cfg)
    tsd = {k.replace("_orig_mod.", ""): v for k, v in tk["model_state_dict"].items()}
    tok.load_state_dict(tsd, strict=False)
    tok = tok.to(dev).eval()
    amp_ok = dev == "cpu" or (dev.startswith("cuda") and torch.cuda.get_device_capability(0)[0] >= 8)
    ac = (lambda: torch.autocast(dev.split(":")[0], dtype=torch.bfloat16)) if amp_ok \
        else contextlib.nullcontext
    return net, tok, ac


def frechet(a, b):
    """Frechet distance between two gaussian fits (diagonal-free, full cov)."""
    mu1, mu2 = a.mean(0), b.mean(0)
    c1 = np.cov(a, rowvar=False) + 1e-6 * np.eye(a.shape[1])
    c2 = np.cov(b, rowvar=False) + 1e-6 * np.eye(b.shape[1])
    from scipy import linalg
    csqrt, _ = linalg.sqrtm(c1 @ c2, disp=False)
    if np.iscomplexobj(csqrt):
        csqrt = csqrt.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(c1 + c2 - 2 * csqrt))


def chunk_feats(z, chunk=16):
    """(N,C,H,W) latents -> per-chunk features: [spatial-mean, spatial-std] per C,
    averaged over the chunk + temporal-delta RMS per C. Shape (n_chunks, 3C)."""
    N, C = z.shape[:2]
    n = N // chunk
    z = z[:n * chunk].reshape(n, chunk, C, -1)
    m = z.mean(dim=(1, 3))                                   # (n, C)
    s = z.std(dim=3).mean(1)                                 # (n, C)
    d = (z[:, 1:] - z[:, :-1]).pow(2).mean(dim=(1, 3)).sqrt()  # (n, C)
    return torch.cat([m, s, d], 1).numpy()


def lap_var(img):
    """Sharpness: variance of the Laplacian (grayscale)."""
    import cv2
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer-ckpt", required=True)
    ap.add_argument("--latents-dir", default="/mnt/nfs/datasets/replay_latents_v7_bc")
    ap.add_argument("--match", default="NA1_5550067582")
    ap.add_argument("--start", type=int, default=3000)
    ap.add_argument("--dream-frames", type=int, default=128)
    ap.add_argument("--num-steps", type=int, default=4)
    ap.add_argument("--contexts", type=int, nargs="+", default=[16, 224])
    ap.add_argument("--out-dir", default="scratchpad/dreamq")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    os.makedirs(args.out_dir, exist_ok=True)

    net, tok, ac = load_models(args.ckpt, args.tokenizer_ckpt, dev)

    z_all = torch.load(glob.glob(f"{args.latents_dir}/{args.match}.pt")[0],
                       weights_only=True)["latents"].float()
    print(f"match {args.match}: {z_all.shape[0]} latent frames; dreaming {args.dream_frames} "
          f"@ K={args.num_steps} from ctx {args.contexts}")

    H = args.dream_frames
    gt = z_all[args.start:args.start + H]                                  # (H,C,16,16)
    results = {}
    dreams = {}
    for ctx in args.contexts:
        c = z_all[args.start - ctx:args.start].unsqueeze(0).to(dev)        # (1,ctx,C,H,W)
        with torch.no_grad(), ac():
            pred = net.rollout(c, predict_frames=H, num_steps=args.num_steps,
                               k_max=64, device=dev)                       # (1,H,C,16,16)
        dreams[ctx] = pred.squeeze(0).float().cpu()
        print(f"  ctx={ctx}: dreamed {H} frames")

    # ---- 2. poor-man's FVD ----
    print("\n== poor-man's FVD (chunked latent-feature Frechet; lower=better) ==")
    # control: real-vs-real from two disjoint parts of the same game (the floor)
    zA = z_all[1000:1000 + 1024]
    zB = z_all[6000:6000 + 1024]
    fA, fB = chunk_feats(zA), chunk_feats(zB)
    ctrl = frechet(fA, fB)
    ref = chunk_feats(z_all[args.start - 1024:args.start + 1024])          # real, near the dream
    print(f"  real-vs-real control : {ctrl:8.2f}   <- the floor")
    for ctx, dz in dreams.items():
        fd = frechet(chunk_feats(dz), ref)
        results[f"fvd_ctx{ctx}"] = fd
        print(f"  dream(ctx={ctx:3d}) vs real: {fd:8.2f}")
    fgt = frechet(chunk_feats(gt), ref)
    print(f"  GT-window vs real    : {fgt:8.2f}   <- sanity (should be ~control)")

    # ---- 3. sharpness over horizon + 1. decode to mp4 ----
    print("\n== sharpness over horizon (Laplacian var of decoded frames) ==")
    import cv2

    def decode(z):                                                        # (n,C,16,16) -> uint8 RGB
        with torch.no_grad(), ac():
            r = tok.decode(z.to(dev).permute(0, 2, 3, 1).reshape(1, -1, 16), z.shape[0])
        r = r.squeeze(0) if r.dim() == 5 else r
        return (r.float().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

    hs = [1, 4, 8, 16, 32, 64, 96, 127]
    gt_img = decode(gt[torch.tensor(hs)])
    gt_sharp = [lap_var(f) for f in gt_img]
    print(f"  {'h':>4} | GT     " + " ".join(f"ctx{c:<4d}" for c in dreams))
    dream_imgs = {c: decode(dreams[c][torch.tensor(hs)]) for c in dreams}
    sharp = {c: [lap_var(f) for f in dream_imgs[c]] for c in dreams}
    for i, h in enumerate(hs):
        row = " ".join(f"{sharp[c][i]:7.0f}" for c in dreams)
        print(f"  {h:>4} | {gt_sharp[i]:6.0f} {row}")

    # side-by-side video: GT | dream(ctx16) | dream(ctx224)
    W = 352
    vw = cv2.VideoWriter(f"{args.out_dir}/dream_vs_gt.mp4",
                         cv2.VideoWriter_fourcc(*"mp4v"), 10, (W * (1 + len(dreams)), W))
    B = 8                                                                 # decode in chunks
    for s in range(0, H, B):
        cols = [decode(gt[s:s + B])]
        cols += [decode(dreams[c][s:s + B]) for c in dreams]
        for j in range(cols[0].shape[0]):
            row = np.concatenate([c[j] for c in cols], axis=1)
            row = cv2.cvtColor(row, cv2.COLOR_RGB2BGR)
            h_abs = s + j
            for k, name in enumerate(["GT"] + [f"dream ctx={c}" for c in dreams]):
                cv2.putText(row, f"{name} h={h_abs}", (k * W + 6, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            vw.write(row)
    vw.release()
    print(f"\nwrote {args.out_dir}/dream_vs_gt.mp4  (GT | " +
          " | ".join(f"ctx{c}" for c in dreams) + ")")
    print("\nVERDICT GUIDE: fvd(dream)~control + sharpness~GT  => dreams on-manifold (healthy);")
    print("fvd >> control and/or sharpness decaying with h     => blur-collapse (fix sampling).")


if __name__ == "__main__":
    main()
