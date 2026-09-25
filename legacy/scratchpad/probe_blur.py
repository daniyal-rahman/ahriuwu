"""Blur / mean-collapse diagnostic for the dynamics rollout.

A diffusion-forcing world model is supposed to sample a *mode* (sharp future),
not regress the conditional mean (blurry, frozen-looking frame). This probe
measures three independent signatures of the failure on a staged checkpoint:

  1. SHARPNESS  — Laplacian variance of the decoded dream vs the tokenizer's
     decode of the TRUE latent. dream/GT << 1  =>  the dream is low-pass (blur).
  2. MOTION     — how far each predicted frame moves from the last real context
     frame, vs how far the TRUE frame moves. dream_motion/true_motion << 1  =>
     the model is under-moving (frozen-frame / persistence collapse).
  3. NOISE-SENS — run the rollout twice with different seeds; latent divergence
     between the two dreams. ~0 => the model ignores its noise input (pure mean
     regression). NB: for an action-conditioned world model the true future is
     mostly determined, so *low* sensitivity is partly expected — read this
     alongside sharpness/motion, not alone.

Run on the login GPU (fp32 on the 1060):
  PYTHONPATH=src python scratchpad/probe_blur.py --checkpoint rollout_stage/dynamics_latest.pt \
     --latents-dir rollout_stage --tokenizer rollout_stage/transformer_tokenizer_latest.pt \
     --labels-root /srv/nfs/datasets/lol_replays_16_9_772 --device cuda --ctx 6 --horizon 12
"""
import argparse, glob, os, tempfile
import numpy as np
import torch
from torch.utils.data import default_collate

from ahriuwu.models.dynamics import create_dynamics
from ahriuwu.models.diffusion import DiffusionSchedule
from ahriuwu.data.replay_dataset import ReplayLatentSequenceDataset


def lap_var(g):
    # variance of the discrete Laplacian = classic focus/sharpness measure (numpy)
    lap = (-4 * g[1:-1, 1:-1]
           + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
    return lap.var()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--latents-dir", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--labels-root", required=True)
    ap.add_argument("--match", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ctx", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--num-steps", type=int, default=6)
    args = ap.parse_args()
    dev = args.device

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"checkpoint: step={ck.get('global_step')} epoch={ck.get('epoch')} loss={ck.get('loss'):.4f}")
    model = create_dynamics("medium", latent_dim=32, use_actions=True, num_kv_heads=4,
                            num_register_tokens=8, soft_cap=50.0, use_qk_norm=True).to(dev).eval()
    sd = ck["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)

    match = args.match or os.path.basename(sorted(glob.glob(f"{args.latents_dir}/*.pt"))[0])[:-3]
    tmp = tempfile.mkdtemp()
    os.symlink(os.path.abspath(f"{args.latents_dir}/{match}.pt"), f"{tmp}/{match}.pt")
    seq_len = args.ctx + args.horizon
    ds = ReplayLatentSequenceDataset(latents_dir=tmp, labels_root=args.labels_root,
                                     outcomes={match: False}, sequence_length=seq_len, stride=seq_len)
    # pick the most dynamic clip (max movement + presses), same rule as rollout_check
    best_i, best = 0, -1.0
    for i in range(0, len(ds), max(1, len(ds) // 30)):
        a = ds[i]["actions"]
        s = a["movement"].std().item() + sum(int(a[k].sum()) for k in a if k != "movement")
        if s > best:
            best_i, best = i, s
    vb = default_collate([ds[best_i]])
    a = vb["actions"]
    z = vb["latents"].to(dev).float()
    a = {k: (v.to(dev).float() if k == "movement" else v.to(dev)) for k, v in a.items()}
    ctx, gt = z[:, :args.ctx], z[:, args.ctx:args.ctx + args.horizon]
    ac = {k: v[:, :args.ctx] for k, v in a.items()}
    af = {k: v[:, args.ctx:args.ctx + args.horizon] for k, v in a.items()}
    last_ctx = ctx[:, -1:]  # (1,1,32,16,16) the held frame

    def roll(seed):
        torch.manual_seed(seed)
        with torch.no_grad():
            return model.rollout(ctx, predict_frames=args.horizon, num_steps=args.num_steps,
                                 k_max=args.num_steps, tau_ctx=0.1, actions_context=ac,
                                 actions_future=af, device=dev).float()

    dreamA, dreamB = roll(0), roll(1)

    # ---------- latent-space metrics (no decode) ----------
    H = args.horizon
    print("\nframe |  MOTION dream/true  |  latent_std dream/true  |  err_to_true err_to_held")
    mot_r, std_r = [], []
    for t in range(H):
        d_mot = (dreamA[:, t] - last_ctx[:, 0]).pow(2).mean().sqrt().item()
        t_mot = (gt[:, t] - last_ctx[:, 0]).pow(2).mean().sqrt().item()
        r = d_mot / max(t_mot, 1e-6)
        sd_r = dreamA[:, t].std().item() / max(gt[:, t].std().item(), 1e-6)
        e_true = (dreamA[:, t] - gt[:, t]).pow(2).mean().item()
        e_held = (dreamA[:, t] - last_ctx[:, 0]).pow(2).mean().item()
        mot_r.append(r); std_r.append(sd_r)
        print(f"  {t+1:3d} |      {r:5.2f}          |        {sd_r:5.2f}            |   {e_true:.4f}    {e_held:.4f}")
    print(f"\nMOTION ratio (dream vs true displacement from held frame): mean {np.mean(mot_r):.2f} "
          f"(1.0=moves like reality, <<1=frozen/under-moving)")
    print(f"LATENT-STD ratio (dream/true): mean {np.mean(std_r):.2f} (1.0=matched, <<1=variance-collapsed=mean regression)")

    # noise sensitivity: how different are two seeds' dreams vs the real motion?
    div = (dreamA - dreamB).pow(2).mean().sqrt().item()
    realmot = (gt[:, 1:] - gt[:, :-1]).pow(2).mean().sqrt().item()
    print(f"NOISE-SENS: ||dreamA-dreamB||={div:.4f}  vs real frame-to-frame motion={realmot:.4f}  "
          f"ratio={div/max(realmot,1e-6):.2f}")

    # ---------- pixel sharpness (decode) ----------
    from ahriuwu.models.transformer_tokenizer import TransformerTokenizer
    del model
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()
    tk = torch.load(args.tokenizer, map_location="cpu", weights_only=False)
    cfg = {k: v for k, v in tk["model_config"].items() if k != "size_preset"}
    tok = TransformerTokenizer(**cfg)
    tsd = tk["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in tsd):
        tsd = {k.replace("_orig_mod.", ""): v for k, v in tsd.items()}
    tok.load_state_dict(tsd, strict=False)
    tok = tok.to(dev).eval()
    NL, LD = cfg["num_latents"], cfg["latent_dim"]

    def decode(zt):
        lat = zt.permute(0, 2, 3, 1).reshape(zt.shape[0], NL, LD).to(dev)
        with torch.no_grad():
            fr = tok.decode(lat.float(), num_frames=1)[:, 0].float().clamp(0, 1)
        return fr.permute(0, 2, 3, 1).cpu().numpy()  # (n,H,W,3) in [0,1]

    tok_gt = decode(gt[0])
    dream = decode(dreamA[0])

    def gray(x):
        return x @ np.array([0.299, 0.587, 0.114])

    print("\nframe |  sharpness dream/GT-decode  (Laplacian-var ratio; <<1 = blurry)")
    ratios = []
    for t in range(H):
        sg = lap_var(gray(tok_gt[t]))
        sd_ = lap_var(gray(dream[t]))
        r = sd_ / max(sg, 1e-9)
        ratios.append(r)
        print(f"  {t+1:3d} |   dream {sd_:8.2f}  /  GT {sg:8.2f}  =  {r:5.2f}")
    print(f"\nSHARPNESS ratio (dream/GT-decode): mean {np.mean(ratios):.2f}  "
          f"(1.0=as sharp as tokenizer can do, <<1=blurry/low-pass)")

    print("\n==== VERDICT GUIDE ====")
    print("blur/mean-collapse  => SHARPNESS<<1 and LATENT-STD<<1")
    print("frozen/copy-collapse=> MOTION<<1 and err_to_held << err_to_true")
    print("healthy sampling    => SHARPNESS~1, MOTION~1, dream tracks true (err_to_true < err_to_held)")


if __name__ == "__main__":
    main()
