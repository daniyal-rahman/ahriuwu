#!/usr/bin/env python3
"""Rollout check + dream visualization for the dynamics world model.

Loads a dynamics checkpoint, rolls it out AUTOREGRESSIVELY (the deploy regime)
on a real match with the real recorded actions, reports per-horizon PSNR, and
with --decode renders a 3-row comparison through the v7 tokenizer:

    GROUND TRUTH  - the original recorded frames (true game)
    TOKENIZER     - decode(GT latent): the tokenizer's reconstruction = the
                    ceiling the dynamics can possibly hit (tokenizer loss)
    DREAM         - decode(autoregressive rollout): what the world model imagines

Output: a side-by-side mp4 + a montage PNG. Gap GT->TOKENIZER is tokenizer loss;
TOKENIZER->DREAM is rollout drift. Runs on CPU so it can overlap live training.

    python scripts/rollout_check.py --decode [--horizon 18] [--checkpoint dyn.pt]
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


def _psnr(a, b, mv):
    mse = ((a - b) ** 2).mean().item()
    return 10 * np.log10(mv ** 2 / max(mse, 1e-10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="/mnt/storage/data/ahriuwu/checkpoints/dynamics_v7_replay/dynamics_latest.pt")
    ap.add_argument("--latents-dir", default="/scratch/ahriuwu/dynamics_replay_latents_v7_dim32")
    ap.add_argument("--labels-root", default="/mnt/nfs/datasets/lol_replays_16_9_772")
    ap.add_argument("--match", default=None)
    ap.add_argument("--model-size", default="medium")
    ap.add_argument("--ctx", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--num-steps", type=int, default=6, help="rollout denoise steps/frame (base model -> d=1)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--decode", action="store_true")
    ap.add_argument("--tokenizer", default="/mnt/storage/data/ahriuwu-checkpoints/tokenizer_v7/transformer_tokenizer_latest.pt")
    ap.add_argument("--out-mp4", default="/mnt/nfs/projects/ahriuwu/dream_check.mp4")
    ap.add_argument("--out-png", default="/mnt/nfs/projects/ahriuwu/dream_check.png")
    ap.add_argument("--out-plot", default="/mnt/nfs/projects/ahriuwu/dream_psnr.png")
    args = ap.parse_args()
    dev = args.device
    amp = torch.bfloat16 if dev != "mps" else torch.float16
    # Pascal (cap<8) has no native bf16; its emulation compounds error across the
    # multi-step rollout so drift looks far worse than reality. Use fp32 there;
    # bf16 only on CPU or a bf16-native GPU (Ampere+, like the deploy 5080/5090).
    amp_ok = dev == "cpu" or (dev.startswith("cuda") and torch.cuda.get_device_capability(0)[0] >= 8)
    print(f"device={dev} bf16_autocast={amp_ok}", flush=True)

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

    match = args.match or os.path.basename(sorted(glob.glob(f"{args.latents_dir}/*.pt"))[0])[:-3]
    tmp = tempfile.mkdtemp()
    os.symlink(os.path.abspath(f"{args.latents_dir}/{match}.pt"), f"{tmp}/{match}.pt")
    seq_len = args.ctx + args.horizon
    ds = ReplayLatentSequenceDataset(latents_dir=tmp, labels_root=args.labels_root,
                                     outcomes={match: False}, sequence_length=seq_len, stride=seq_len)
    best_i, best_score = 0, -1.0
    for i in range(0, len(ds), max(1, len(ds) // 30)):
        a = ds[i]["actions"]
        score = a["movement"].std().item() + sum(int(a[k].sum()) for k in a if k != "movement")
        if score > best_score:
            best_i, best_score = i, score
    best = ds[best_i]
    start_idx = ds.sequences[best_i]["start_idx"]
    vb = default_collate([best])
    a = vb["actions"]
    presses = {k: int(a[k].sum()) for k in a if k != "movement" and int(a[k].sum())}
    print(f"{match}: {len(ds)} seqs | chosen movement_std={a['movement'].std():.3f} presses={presses}", flush=True)

    sched = DiffusionSchedule(device=dev)
    tf = td.eval_denoising_psnr(model, sched, vb, dev)
    print("TEACHER-FORCED 1-step:", {k.split('/')[-1]: round(v, 1) for k, v in tf.items() if "psnr" in k}, flush=True)

    z = vb["latents"].to(dev).float()  # latents stored fp16; fp32 for the no-autocast path
    a = {k: (v.to(dev).float() if k == "movement" else v.to(dev)) for k, v in a.items()}
    ctx, gt = z[:, :args.ctx], z[:, args.ctx:args.ctx + args.horizon].float()
    ac = {k: v[:, :args.ctx] for k, v in a.items()}
    af = {k: v[:, args.ctx:args.ctx + args.horizon] for k, v in a.items()}
    with torch.no_grad():
        if amp_ok:
            with torch.autocast(device_type=dev.split(":")[0], dtype=amp):
                pred = model.rollout(ctx, predict_frames=args.horizon, num_steps=args.num_steps,
                                     k_max=args.num_steps, tau_ctx=0.1, actions_context=ac,
                                     actions_future=af, device=dev).float()
        else:
            pred = model.rollout(ctx, predict_frames=args.horizon, num_steps=args.num_steps,
                                 k_max=args.num_steps, tau_ctx=0.1, actions_context=ac,
                                 actions_future=af, device=dev).float()
    mv = z.float().abs().max().item()
    hs = sorted({1, 2, 4, 8, args.horizon})
    print("ROLLOUT (autoregressive, real actions):",
          {f"h{o}": round(_psnr(pred[:, o - 1], gt[:, o - 1], mv), 1) for o in hs if o <= args.horizon},
          "| mean", round(_psnr(pred, gt, mv), 1), flush=True)

    if not args.decode:
        print("DONE (no decode)", flush=True)
        return

    # rollout is done -> free the dynamics model before loading the big tokenizer
    # (keeps both off a small GPU at once).
    del model
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()

    # ---- 3-row decode: GROUND TRUTH (png) / TOKENIZER (decode GT latent) / DREAM ----
    import cv2
    from ahriuwu.models.transformer_tokenizer import TransformerTokenizer
    fi = torch.load(f"{tmp}/{match}.pt", weights_only=True)["frame_indices"].tolist()
    fdir = f"{args.labels_root}/{match}/frames"

    def png(num):
        im = cv2.imread(f"{fdir}/{int(num):06d}.png")
        return im if im is not None else np.zeros((352, 352, 3), np.uint8)

    ctx_png = [png(fi[start_idx + j]) for j in range(args.ctx)]
    gt_png = [png(fi[start_idx + args.ctx + j]) for j in range(args.horizon)]

    tk = torch.load(args.tokenizer, map_location="cpu", weights_only=False)
    cfg = {k: v for k, v in tk["model_config"].items() if k != "size_preset"}
    tok = TransformerTokenizer(**cfg)
    tsd = tk["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in tsd):
        tsd = {k.replace("_orig_mod.", ""): v for k, v in tsd.items()}
    tok.load_state_dict(tsd, strict=False)
    tok = tok.to(dev).eval()
    NL, LD = cfg["num_latents"], cfg["latent_dim"]

    def decode(zt, chunk=4):  # (n,32,16,16) -> list of (H,W,3) uint8 BGR; chunked for small VRAM
        out = []
        for s in range(0, zt.shape[0], chunk):
            z = zt[s:s + chunk]
            lat = z.permute(0, 2, 3, 1).reshape(z.shape[0], NL, LD).to(dev)
            with torch.no_grad():
                fr = tok.decode(lat.float(), num_frames=1)[:, 0].float().clamp(0, 1)
            fr = fr.permute(0, 2, 3, 1).cpu().numpy()
            out.extend(np.ascontiguousarray(fr[..., ::-1] * 255).astype(np.uint8))
        return out

    tok_gt = decode(gt[0])     # tokenizer recon of GT future
    dream = decode(pred[0])    # dream
    H, W = gt_png[0].shape[:2]

    # ---- per-frame PIXEL PSNR vs the true frame: tokenizer ceiling vs dynamics dream ----
    def px_psnr(a, b):
        mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
        return 10 * np.log10(255.0 ** 2 / max(mse, 1e-10))
    tok_curve = [px_psnr(tok_gt[t], gt_png[t]) for t in range(args.horizon)]
    dyn_curve = [px_psnr(dream[t], gt_png[t]) for t in range(args.horizon)]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = list(range(1, args.horizon + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(xs, tok_curve, "o-", color="#f0a000", label="Tokenizer ceiling (decode of TRUE latent)")
    plt.plot(xs, dyn_curve, "o-", color="#00b4c8", label="Dynamics dream (decode of rollout)")
    plt.fill_between(xs, dyn_curve, tok_curve, color="gray", alpha=0.12)
    plt.xlabel("frames into the future (autoregressive rollout)")
    plt.ylabel("PSNR vs the true frame (dB, pixels)")
    plt.title(f"Rollout quality per frame — job 124 @ step {ck.get('global_step')}\nmatch {match}")
    plt.xticks(xs)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(args.out_plot, dpi=120)
    print(f"plot: {args.out_plot} | tok {tok_curve[0]:.1f}->{tok_curve[-1]:.1f}  "
          f"dyn {dyn_curve[0]:.1f}->{dyn_curve[-1]:.1f}", flush=True)

    def lab(im, t, c=(0, 255, 0)):
        im = im.copy(); cv2.putText(im, t, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2); return im

    GREEN, CYAN, YEL = (0, 255, 0), (0, 200, 255), (0, 255, 255)
    vw = cv2.VideoWriter(args.out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), 6, (3 * W, H))
    if vw.isOpened():
        for f in ctx_png:  # context: same real frame across all 3 panels
            vw.write(np.hstack([lab(f, "GROUND TRUTH"), lab(f, "TOKENIZER"), lab(f, "DREAM")]))
        for j in range(args.horizon):
            vw.write(np.hstack([lab(gt_png[j], "GROUND TRUTH", GREEN),
                                lab(tok_gt[j], "TOKENIZER", YEL),
                                lab(dream[j], f"DREAM +{j+1}", CYAN)]))
        vw.release()
        print(f"mp4: {args.out_mp4} ({args.ctx+args.horizon} frames @6fps, {3*W}x{H})", flush=True)
    else:
        print("WARN: mp4 writer failed (codec) — PNG only", flush=True)

    cols = sorted({1, max(2, args.horizon // 2), args.horizon})
    rows = [
        [lab(ctx_png[-1], "ctx end")] + [lab(gt_png[c - 1], f"GT +{c}", GREEN) for c in cols],
        [lab(ctx_png[-1], "ctx end")] + [lab(tok_gt[c - 1], f"TOK +{c}", YEL) for c in cols],
        [lab(ctx_png[-1], "ctx end")] + [lab(dream[c - 1], f"DREAM +{c}", CYAN) for c in cols],
    ]
    cv2.imwrite(args.out_png, np.vstack([np.hstack(r) for r in rows]))
    print(f"montage: {args.out_png}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
