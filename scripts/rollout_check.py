#!/usr/bin/env python3
"""Rollout check + dream visualization for the dynamics world model.

Loads a dynamics checkpoint, rolls it out AUTOREGRESSIVELY (the deploy regime)
on a real match with the real recorded actions, reports per-horizon PSNR, and
with --decode renders the ground-truth vs rolled-out latents back through the v7
tokenizer into a side-by-side REAL|DREAM mp4 + a montage PNG so you can WATCH it.

    python scripts/rollout_check.py [--checkpoint dyn.pt] [--decode]

Defaults point at the desktop paths (job-124 checkpoint, /scratch latents). Runs
on CPU by default so it can overlap the live training without touching the GPU.
"""
import argparse, glob, os, sys, tempfile
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torch.utils.data import default_collate
import train_dynamics as td
from ahriuwu.models.dynamics import create_dynamics
from ahriuwu.models.diffusion import DiffusionSchedule
from ahriuwu.data.replay_dataset import ReplayLatentSequenceDataset


def _psnr(a, b, max_val):
    mse = ((a - b) ** 2).mean().item()
    return 10 * np.log10(max_val ** 2 / max(mse, 1e-10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="/mnt/storage/data/ahriuwu/checkpoints/dynamics_v7_replay/dynamics_latest.pt")
    ap.add_argument("--latents-dir", default="/scratch/ahriuwu/dynamics_replay_latents_v7_dim32")
    ap.add_argument("--labels-root", default="/mnt/nfs/datasets/lol_replays_16_9_772")
    ap.add_argument("--match", default=None, help="match id (default: first latent file)")
    ap.add_argument("--model-size", default="medium")
    ap.add_argument("--ctx", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--num-steps", type=int, default=6, help="rollout denoise steps/frame (base model -> d=1)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--decode", action="store_true", help="render REAL vs DREAM via the v7 tokenizer")
    ap.add_argument("--tokenizer", default="/mnt/storage/data/ahriuwu-checkpoints/tokenizer_v7/transformer_tokenizer_latest.pt")
    ap.add_argument("--out-mp4", default="/mnt/nfs/projects/ahriuwu/dream_check.mp4")
    ap.add_argument("--out-png", default="/mnt/nfs/projects/ahriuwu/dream_check.png")
    args = ap.parse_args()
    dev = args.device
    amp = torch.bfloat16 if dev != "mps" else torch.float16

    # ---- dynamics model ----
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"checkpoint: global_step={ck.get('global_step')} epoch={ck.get('epoch')} loss={ck.get('loss')}", flush=True)
    model = create_dynamics(args.model_size, latent_dim=32, use_actions=True, num_kv_heads=4,
                            num_register_tokens=8, soft_cap=50.0, use_qk_norm=True).to(dev).eval()
    sd = ck["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    miss, unexp = model.load_state_dict(sd, strict=False)
    miss = [m for m in miss if "rope" not in m.lower()]
    print(f"load: missing(non-rope)={len(miss)} unexpected={len(unexp)}", flush=True)
    assert len(miss) + len(unexp) <= 10, f"ARCH MISMATCH miss={miss[:5]} unexp={unexp[:5]}"

    # ---- one-match dataset, pick the most ACTIVE sequence (real actions) ----
    match = args.match or os.path.basename(sorted(glob.glob(f"{args.latents_dir}/*.pt"))[0])[:-3]
    tmp = tempfile.mkdtemp()
    os.symlink(f"{args.latents_dir}/{match}.pt", f"{tmp}/{match}.pt")
    seq_len = args.ctx + args.horizon
    ds = ReplayLatentSequenceDataset(latents_dir=tmp, labels_root=args.labels_root,
                                     outcomes={match: False}, sequence_length=seq_len, stride=seq_len)
    best, best_score = ds[0], -1.0
    for i in range(0, len(ds), max(1, len(ds) // 30)):
        s = ds[i]
        a = s["actions"]
        score = a["movement"].std().item() + sum(int(a[k].sum()) for k in a if k != "movement")
        if score > best_score:
            best, best_score = s, score
    vb = default_collate([best])
    a = vb["actions"]
    presses = {k: int(a[k].sum()) for k in a if k != "movement" and int(a[k].sum())}
    print(f"{match}: {len(ds)} seqs | chosen movement_std={a['movement'].std():.3f} presses={presses}", flush=True)

    # ---- teacher-forced (contrast) + autoregressive rollout (capture latents) ----
    sched = DiffusionSchedule(device=dev)
    tf = td.eval_denoising_psnr(model, sched, vb, dev)
    print("TEACHER-FORCED 1-step:", {k.split('/')[-1]: round(v, 1) for k, v in tf.items() if "psnr" in k}, flush=True)

    z = vb["latents"].to(dev)
    ctx, gt = z[:, :args.ctx], z[:, args.ctx:args.ctx + args.horizon].float()
    ac = {k: v[:, :args.ctx] for k, v in a.items()}
    af = {k: v[:, args.ctx:args.ctx + args.horizon] for k, v in a.items()}
    with torch.no_grad(), torch.autocast(device_type=dev.split(":")[0], dtype=amp):
        pred = model.rollout(ctx, predict_frames=args.horizon, num_steps=args.num_steps,
                             k_max=args.num_steps, tau_ctx=0.1, actions_context=ac,
                             actions_future=af, device=dev).float()
    mv = z.float().abs().max().item()
    hs = sorted({1, 2, 4, args.horizon})
    print("ROLLOUT (autoregressive, real actions):",
          {f"h{o}": round(_psnr(pred[:, o - 1], gt[:, o - 1], mv), 1) for o in hs if o <= args.horizon},
          "| mean", round(_psnr(pred, gt, mv), 1), flush=True)

    if not args.decode:
        print("DONE (no decode)", flush=True)
        return

    # ---- decode REAL vs DREAM through the v7 tokenizer ----
    import cv2
    from ahriuwu.models.transformer_tokenizer import TransformerTokenizer
    tk = torch.load(args.tokenizer, map_location="cpu", weights_only=False)
    cfg = {k: v for k, v in tk["model_config"].items() if k != "size_preset"}
    tok = TransformerTokenizer(**cfg)
    tsd = tk["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in tsd):
        tsd = {k.replace("_orig_mod.", ""): v for k, v in tsd.items()}
    tok.load_state_dict(tsd, strict=False)
    tok = tok.to(dev).eval()
    NL, LD = cfg["num_latents"], cfg["latent_dim"]

    def decode(zt):  # (n,32,16,16) dynamics latents -> (n,H,W,3) uint8 BGR frames
        n = zt.shape[0]
        lat = zt.permute(0, 2, 3, 1).reshape(n, NL, LD).to(dev)  # inverse of the pretokenize fold
        with torch.no_grad():
            fr = tok.decode(lat.float(), num_frames=1)[:, 0]      # (n,3,H,W) in [0,1]
        fr = fr.float().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
        return np.ascontiguousarray(fr[..., ::-1] * 255).astype(np.uint8)  # RGB->BGR

    ctx_f, gt_f, dr_f = decode(ctx[0]), decode(gt[0]), decode(pred[0])
    H, W = gt_f.shape[1:3]

    def lab(img, txt, color=(0, 255, 0)):
        img = img.copy()
        cv2.putText(img, txt, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return img

    # mp4: context (shared) then REAL | DREAM
    vw = cv2.VideoWriter(args.out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), 6, (2 * W, H))
    ok = vw.isOpened()
    if ok:
        for f in ctx_f:
            vw.write(np.hstack([lab(f, "context"), lab(f, "context")]))
        for i, (g, d) in enumerate(zip(gt_f, dr_f)):
            vw.write(np.hstack([lab(g, "REAL"), lab(d, f"DREAM +{i+1}", (0, 200, 255))]))
        vw.release()
        print(f"DREAM mp4: {args.out_mp4} ({len(ctx_f)+len(gt_f)} frames @6fps, {W}x{H}/side)", flush=True)
    else:
        print("WARN: mp4 writer failed to open (codec) — PNG montage only", flush=True)

    # montage PNG: rows REAL/DREAM x cols [ctx-end, +1, +mid, +last]
    cols = sorted({1, max(2, args.horizon // 2), args.horizon})
    real_row = [lab(ctx_f[-1], "ctx end")] + [lab(gt_f[c - 1], f"REAL +{c}") for c in cols]
    dream_row = [lab(ctx_f[-1], "ctx end")] + [lab(dr_f[c - 1], f"DREAM +{c}", (0, 200, 255)) for c in cols]
    montage = np.vstack([np.hstack(real_row), np.hstack(dream_row)])
    cv2.imwrite(args.out_png, montage)
    print(f"DREAM montage: {args.out_png} ({montage.shape[1]}x{montage.shape[0]})", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
