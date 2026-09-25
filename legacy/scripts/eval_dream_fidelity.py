#!/usr/bin/env python3
"""Open-loop dream-fidelity evaluation for the Phase-1 world model.

Answers: how long do imagined rollouts stay faithful to reality?

Measures the EXACT path scripts/train_imagination.py:imagine() uses -- repeated
DynamicsTransformer.rollout(predict_frames=1) on a GROWING window, gen K=4
shortcut steps, k_max=64, tau_ctx=0.1 -- but feeds the REAL recorded actions
instead of policy samples, so the only error source is the world model.

Variants rolled per start point (all share the same real context + actions):
  dream        the real thing: dreamed frames are fed back into the window
  tf           teacher-forced: the REAL frame is fed back, so every step is a
               fresh 1-step prediction. This is the no-compounding FLOOR.
  copy         predict z_{t+k} = last real context frame (the "do nothing"
               baseline; also measures how fast the real game actually moves)
  shuffle      a random real frame from elsewhere in the same game -- garbage
               but ON-manifold
  noise        gaussian matched to the per-channel latent moments -- garbage and
               OFF-manifold

Action-conditioning probe (--action-ab): roll the same context under (a) the real
actions, (b) mirrored actions, (c) the real actions again under a different noise
seed. If d(a,b) ~ d(a,c), the dreams do not depend on the action.
"""
import argparse
import contextlib
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from ahriuwu.constants import ABILITY_KEYS, MOVEMENT_DIM  # noqa: E402


# --------------------------------------------------------------------------
# model loading
# --------------------------------------------------------------------------
def load_dynamics(ckpt_path, dev):
    from ahriuwu.models.dynamics import create_dynamics
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["model_config"]
    net = create_dynamics(
        size=cfg["size_preset"], latent_dim=cfg["latent_dim"],
        use_agent_tokens=cfg["use_agent_tokens"], num_tasks=cfg["num_tasks"],
        agent_layers=cfg["agent_layers"], use_actions=cfg["use_actions"],
        use_game_time=cfg["use_game_time"], use_qk_norm=cfg["use_qk_norm"],
        soft_cap=cfg["soft_cap"], num_register_tokens=cfg["num_register_tokens"],
        num_kv_heads=cfg["num_kv_heads"], k_max=cfg["k_max"],
        gradient_checkpointing=False,
    ).to(dev).eval()
    sd = {k.replace("_orig_mod.", ""): v for k, v in ck["model_state_dict"].items()}
    miss, unexp = net.load_state_dict(sd, strict=False)
    miss = [m for m in miss if "rope" not in m.lower()]
    assert not miss and not unexp, f"ARCH MISMATCH miss={miss[:5]} unexp={unexp[:5]}"
    nparam = sum(p.numel() for p in net.parameters())
    net.requires_grad_(False)
    print(f"[dynamics] step={ck.get('global_step')} epoch={ck.get('epoch')} "
          f"cfg={cfg['size_preset']} dim={cfg['model_dim']} layers={cfg['num_layers']} "
          f"use_actions={cfg['use_actions']} params={nparam/1e6:.0f}M")
    return net, cfg


def load_tokenizer(tok_path, dev):
    from ahriuwu.models.transformer_tokenizer import TransformerTokenizer
    tk = torch.load(tok_path, map_location="cpu", weights_only=False)
    cfg = {k: v for k, v in tk["model_config"].items() if k != "size_preset"}
    tok = TransformerTokenizer(**cfg)
    sd = {k.replace("_orig_mod.", ""): v for k, v in tk["model_state_dict"].items()}
    miss, unexp = tok.load_state_dict(sd, strict=False)
    miss = [m for m in miss if "rope" not in m.lower()]
    assert not miss and not unexp, f"TOK MISMATCH miss={miss[:5]} unexp={unexp[:5]}"
    tok = tok.to(dev).eval()
    tok.requires_grad_(False)
    print(f"[tokenizer] step={tk.get('global_step')} latent_dim={cfg.get('latent_dim')}")
    return tok


def amp_ctx(dev):
    ok = dev.startswith("cuda") and torch.cuda.get_device_capability(0)[0] >= 8
    return (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if ok else contextlib.nullcontext


def decode_latents(tok, z, ac, chunk=8):
    """(N,32,16,16) dynamics latents -> (N,3,352,352) RGB in [0,1].

    Inverts scripts/agent_infer._dyn_from_tok: (N,32,16,16) -> (N,16,16,32)
    -> (N,512,16), which is the tokenizer's own latent layout.
    """
    outs = []
    for s in range(0, z.shape[0], chunk):
        zz = z[s:s + chunk]
        n = zz.shape[0]
        lat = zz.permute(0, 2, 3, 1).reshape(1, n * 512, 16)
        with torch.no_grad(), ac():
            r = tok.decode(lat, n)              # (1, n, 3, 352, 352)
        outs.append(r.squeeze(0).float().clamp(0, 1).cpu())
    return torch.cat(outs, 0)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_game(latents_dir, labels_root, match, device):
    """Whole-game latents (kept on CPU, fp16) + per-frame real action arrays.

    Uses ReplayLatentSequenceDataset._parse_match directly (the ab_checkpoints.py
    route) so we never trigger the whole-directory index scan.
    """
    import pathlib
    from ahriuwu.data.replay_dataset import ReplayLatentSequenceDataset
    ds = ReplayLatentSequenceDataset.__new__(ReplayLatentSequenceDataset)
    ds.outcomes = {match: False}          # only feeds the reward channel, unused here
    ds.movement_source = "clicks"
    ds.prefirst_mode = "sentinel"
    ds.movement_interp = False
    ds.reward_config = None
    md = ds._parse_match(match, pathlib.Path(f"{labels_root}/{match}/labels.json"))
    pack = torch.load(f"{latents_dir}/{match}.pt", weights_only=True)
    fi = pack["frame_indices"].numpy()
    assert fi[0] == 0 and np.all(np.diff(fi) == 1), f"{match}: non-contiguous frame_indices"
    z = pack["latents"]                                  # (N,32,16,16) fp16 CPU
    N = min(z.shape[0], md["movement"].shape[0])
    acts = {"movement": md["movement"][:N].float()}      # (N,2) in [0,1]
    for k in ABILITY_KEYS:
        acts[k] = md["abilities"][k][:N].long()          # (N,)
    acts["cursor_valid"] = torch.ones(N, dtype=torch.bool)
    return z[:N], acts, md


def slice_actions(acts, idx, device):
    """idx: LongTensor (B, L) absolute frame indices -> batched action dict."""
    out = {"movement": acts["movement"][idx].to(device)}
    for k in ABILITY_KEYS:
        out[k] = acts[k][idx].to(device)
    out["cursor_valid"] = acts["cursor_valid"][idx].to(device)
    return out


def cat_actions(a, b):
    return {k: torch.cat([a[k], b[k]], dim=1) for k in a}


# --------------------------------------------------------------------------
# the rollout under test
# --------------------------------------------------------------------------
@torch.no_grad()
def phase3_rollout(net, z_ctx, acts_ctx, acts_future, horizon, *,
                   gen_steps=4, k_max=64, tau_ctx=0.1, device="cuda",
                   feedback="dream", z_true=None, tf_period=0,
                   window_cap=0, reproject=None, cached=False):
    """Roll `horizon` frames the way train_imagination.imagine() does.

    imagine() calls DynamicsTransformer.rollout(predict_frames=1) once per step on
    a window that GROWS by one frame each step, so the KV cache is rebuilt (and the
    whole window re-corrupted at tau ~ U(1-tau_ctx, 1)) every step. That is the
    default here. `cached=True` instead issues ONE rollout(predict_frames=horizon),
    which keeps a persistent cache and commits each dreamed frame CLEAN -- a cheaper
    path Phase 3 could use, measured as an alternative.

    feedback:
      'dream'  dreamed frame goes back into the window (what Phase 3 does)
      'tf'     the REAL frame goes back in -> every step is a fresh 1-step
               prediction (the no-compounding floor)
      'periodic' dream, but every `tf_period` steps the window is reset to real
    window_cap: if >0, keep only the last `window_cap` frames in the window.
    reproject:  optional fn(z)->z applied to a dreamed frame before feedback
                (tokenizer decode->encode = projection back onto the real manifold).
    Returns (B, horizon, C, H, W).
    """
    B = z_ctx.shape[0]
    if cached:
        return net.rollout(context=z_ctx, predict_frames=horizon, num_steps=gen_steps,
                           k_max=k_max, tau_ctx=tau_ctx, actions_context=acts_ctx,
                           actions_future=acts_future, device=device)
    z_window = z_ctx
    a_window = acts_ctx
    preds = []
    for t in range(horizon):
        a_step = {k: v[:, t:t + 1] for k, v in acts_future.items()}
        z_next = net.rollout(context=z_window, predict_frames=1, num_steps=gen_steps,
                             k_max=k_max, tau_ctx=tau_ctx, actions_context=a_window,
                             actions_future=a_step, device=device)      # (B,1,C,H,W)
        preds.append(z_next)
        if feedback == "tf":
            fed = z_true[:, t:t + 1]
        elif feedback == "periodic" and tf_period > 0 and (t + 1) % tf_period == 0:
            fed = z_true[:, t:t + 1]
        else:
            fed = reproject(z_next) if reproject is not None else z_next
        z_window = torch.cat([z_window, fed], dim=1)
        a_window = cat_actions(a_window, a_step)
        if window_cap and z_window.shape[1] > window_cap:
            z_window = z_window[:, -window_cap:]
            a_window = {k: v[:, -window_cap:] for k, v in a_window.items()}
    return torch.cat(preds, dim=1)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def latent_metrics(pred, true, ref):
    """pred/true: (B,H,C,Hh,Ww). ref: (B,1,C,Hh,Ww) last real context frame.

    Returns dict of (B,H) arrays.
      nmse    ||pred-true||^2 / ||true - mean(true)||^2   (1.0 == predicting the mean)
      cos     cosine similarity of the flattened latents
      d_ref   ||pred - last_ctx||  -- how far the prediction moved from the context
      d_true_ref ||true - last_ctx||  -- how far REALITY moved (the scale)
      std     per-frame latent std (collapse / blur indicator)
    """
    B, H = pred.shape[:2]
    p = pred.reshape(B, H, -1).float()
    t = true.reshape(B, H, -1).float()
    r = ref.reshape(B, 1, -1).float()
    # denominator: variance of the TRUE latents about their own per-sample mean,
    # so nmse==1 means "no better than predicting the constant mean latent".
    tm = t.mean(dim=(1, 2), keepdim=True)
    denom = (t - tm).pow(2).mean(dim=2) + 1e-8
    return {
        "nmse": ((p - t).pow(2).mean(2) / denom).cpu().numpy(),
        "mse": (p - t).pow(2).mean(2).cpu().numpy(),
        "cos": torch.nn.functional.cosine_similarity(p, t, dim=2).cpu().numpy(),
        "d_ref": (p - r).pow(2).mean(2).sqrt().cpu().numpy(),
        "d_true_ref": (t - r).pow(2).mean(2).sqrt().cpu().numpy(),
        "std": p.std(dim=2).cpu().numpy(),
        "std_true": t.std(dim=2).cpu().numpy(),
    }


def pixel_metrics(tok, pred, true, ac, chunk=8):
    """Decode both and return per-(B,H) PSNR + Laplacian sharpness of the dream."""
    import cv2
    B, H = pred.shape[:2]
    C, Hh, Ww = pred.shape[2:]
    dp = decode_latents(tok, pred.reshape(B * H, C, Hh, Ww), ac, chunk)
    dt = decode_latents(tok, true.reshape(B * H, C, Hh, Ww), ac, chunk)
    mse = (dp - dt).pow(2).mean(dim=(1, 2, 3)).clamp_min(1e-10)
    psnr = (-10.0 * torch.log10(mse)).reshape(B, H).numpy()
    sharp = np.zeros((B * H,), dtype=np.float32)
    sharp_t = np.zeros((B * H,), dtype=np.float32)
    for i in range(B * H):
        g = cv2.cvtColor((dp[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        sharp[i] = cv2.Laplacian(g, cv2.CV_32F).var()
        g = cv2.cvtColor((dt[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        sharp_t[i] = cv2.Laplacian(g, cv2.CV_32F).var()
    return {"psnr": psnr, "sharp": sharp.reshape(B, H), "sharp_true": sharp_t.reshape(B, H)}


# --------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------
def build_arms(names, horizon):
    """name -> (rollout kwargs, action transform). Action transform maps the real
    future action dict to what the rollout is fed."""
    P3 = dict(gen_steps=4, k_max=64, tau_ctx=0.1)     # exactly train_imagination defaults
    A = {
        # --- the thing under test + its brackets ---
        "dream":      (dict(**P3, feedback="dream"), "real"),
        "tf":         (dict(**P3, feedback="tf"), "real"),
        # --- sampler / shortcut-step ablation (d = k_max // gen_steps) ---
        "d16_K4":     (dict(gen_steps=4,  k_max=64, tau_ctx=0.1, feedback="dream"), "real"),
        "d1_K4":      (dict(gen_steps=4,  k_max=4,  tau_ctx=0.1, feedback="dream"), "real"),
        "d1_K16":     (dict(gen_steps=16, k_max=16, tau_ctx=0.1, feedback="dream"), "real"),
        "d1_K64":     (dict(gen_steps=64, k_max=64, tau_ctx=0.1, feedback="dream"), "real"),
        "d4_K16":     (dict(gen_steps=16, k_max=64, tau_ctx=0.1, feedback="dream"), "real"),
        "d1_K4_tf":   (dict(gen_steps=4,  k_max=4,  tau_ctx=0.1, feedback="tf"), "real"),
        "d1_K64_tf":  (dict(gen_steps=64, k_max=64, tau_ctx=0.1, feedback="tf"), "real"),
        # --- context corruption ---
        "tau0":       (dict(gen_steps=4, k_max=64, tau_ctx=0.0, feedback="dream"), "real"),
        "tau03":      (dict(gen_steps=4, k_max=64, tau_ctx=0.3, feedback="dream"), "real"),
        "tau05":      (dict(gen_steps=4, k_max=64, tau_ctx=0.5, feedback="dream"), "real"),
        # tau_ctx=1.0 -> context tau ~ U(0,1): EXACTLY the training marginal. Training
        # samples tau i.i.d. U(0,1) per frame (diffusion.py:123), so the all-clean
        # context the rollout actually uses has probability ~1e-16 under training.
        "tau10":      (dict(gen_steps=4, k_max=64, tau_ctx=1.0, feedback="dream"), "real"),
        # best inference-time configuration: trained shortcut row (d=1), converged
        # sampler (K=64), perfectly clean context.
        "best":       (dict(gen_steps=64, k_max=64, tau_ctx=0.0, feedback="dream"), "real"),
        "best_cached": (dict(gen_steps=64, k_max=64, tau_ctx=0.0, feedback="dream",
                             cached=True), "real"),
        "best_tf":    (dict(gen_steps=64, k_max=64, tau_ctx=0.0, feedback="tf"), "real"),
        # --- action conditioning ---
        "act_mirror": (dict(**P3, feedback="dream"), "mirror"),
        "act_frozen": (dict(**P3, feedback="dream"), "frozen"),
        "act_none":   (dict(**P3, feedback="dream"), "none"),
        "dream_seed2": (dict(**P3, feedback="dream"), "real"),   # same actions, new noise
        # --- mechanism ---
        "cached":     (dict(**P3, feedback="dream", cached=True), "real"),
        # cached variants of the action A/B -- rollout() keeps one persistent KV
        # cache instead of re-prefilling per step. Measured within ~10% of the
        # Phase-3 path at 23x less compute, and the action test is a within-path
        # ratio, so the cheap path is the right one for a multi-arm sweep.
        "cachedA_seed2": (dict(**P3, feedback="dream", cached=True), "real"),
        "cachedA_mirror": (dict(**P3, feedback="dream", cached=True), "mirror"),
        "cachedA_frozen": (dict(**P3, feedback="dream", cached=True), "frozen"),
        "cachedA_none": (dict(**P3, feedback="dream", cached=True), "none"),
        "cachedA_best": (dict(gen_steps=16, k_max=16, tau_ctx=0.0, feedback="dream",
                              cached=True), "real"),
        "cachedA_tau0": (dict(gen_steps=4, k_max=4, tau_ctx=0.0, feedback="dream",
                              cached=True), "real"),
        "cachedA_d1": (dict(gen_steps=4, k_max=4, tau_ctx=0.1, feedback="dream",
                            cached=True), "real"),
        "cap16":      (dict(**P3, feedback="dream", window_cap=16), "real"),
        "cap32":      (dict(**P3, feedback="dream", window_cap=32), "real"),
        "tf_p4":      (dict(**P3, feedback="periodic", tf_period=4), "real"),
        "tf_p8":      (dict(**P3, feedback="periodic", tf_period=8), "real"),
        "tf_p16":     (dict(**P3, feedback="periodic", tf_period=16), "real"),
        "reproject":  (dict(**P3, feedback="dream"), "real"),    # handled in main
        "d1K64_reproj": (dict(gen_steps=64, k_max=64, tau_ctx=0.1, feedback="dream"), "real"),
    }
    for n in names:
        if n not in A:
            raise SystemExit(f"unknown arm {n}; have {sorted(A)}")
    return {n: A[n] for n in names}


def transform_actions(acts, how):
    if how == "real":
        return acts
    out = {k: v.clone() for k, v in acts.items()}
    if how == "mirror":
        out["movement"] = 1.0 - out["movement"]
    elif how == "frozen":
        out["movement"] = out["movement"][:, :1].expand_as(out["movement"]).clone()
        for k in ABILITY_KEYS:
            out[k] = torch.zeros_like(out[k])
    elif how == "none":
        out["cursor_valid"] = torch.zeros_like(out["cursor_valid"])
        for k in ABILITY_KEYS:
            out[k] = torch.zeros_like(out[k])
    else:
        raise ValueError(how)
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
HELD_OUT = ["NA1_5549995114", "NA1_5550417257", "NA1_5551063460",
            "NA1_5551782551", "NA1_5552261591", "NA1_5552945604"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="rollout_stage/desktop_resume_8775_stripped.pt")
    ap.add_argument("--tokenizer-ckpt", default="rollout_stage/transformer_tokenizer_latest.pt")
    ap.add_argument("--latents-dir", default="/mnt/nfs/datasets/replay_latents_v7_bc")
    ap.add_argument("--labels-root", default="/mnt/nfs/datasets/lol_replays_16_9_772")
    ap.add_argument("--games", nargs="+", default=HELD_OUT)
    ap.add_argument("--arms", nargs="+", default=["dream", "tf"])
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--context", type=int, default=16)
    ap.add_argument("--starts-per-game", type=int, default=8)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pixels", action="store_true", help="also decode + score PSNR/sharpness")
    ap.add_argument("--save-latents", action="store_true", help="dump dreamed latents (for the probe)")
    ap.add_argument("--out", default="scratchpad/dreamfid/run.npz")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    net, _ = load_dynamics(args.ckpt, dev)
    ac = amp_ctx(dev)
    tok = load_tokenizer(args.tokenizer_ckpt, dev) if (args.pixels or "reproject" in args.arms
                                                       or "d1K64_reproj" in args.arms) else None

    rng = np.random.default_rng(args.seed)
    Ctx, H = args.context, args.horizon

    # ---- pick start points (same across arms) ----
    games, samples = {}, []
    for g in args.games:
        z, acts, md = load_game(args.latents_dir, args.labels_root, g, dev)
        games[g] = (z, acts)
        ev = md["movement_event"].numpy()
        first = int(np.argmax(ev)) if ev.any() else 0
        lo, hi = max(Ctx + 1, first + 200), z.shape[0] - H - 2
        assert hi > lo, f"{g}: game too short"
        st = rng.choice(np.arange(lo, hi), size=args.starts_per_game, replace=False)
        samples += [(g, int(s)) for s in sorted(st)]
        print(f"[data] {g}: {z.shape[0]} frames, first click @{first}, "
              f"{args.starts_per_game} starts in [{lo},{hi}]")
    print(f"[data] {len(samples)} rollouts total | ctx={Ctx} horizon={H}")

    arms = build_arms(args.arms, H)
    results = {}
    saved_lat = {}
    for arm_name, (kw, act_how) in arms.items():
        seed = args.seed + (1000 if arm_name in ("dream_seed2", "cachedA_seed2") else 0)
        reproj = None
        if arm_name in ("reproject", "d1K64_reproj"):
            def reproj(z, _tok=tok, _ac=ac):
                B1, T1, C1, H1, W1 = z.shape
                lat = z.reshape(B1 * T1, C1, H1, W1).permute(0, 2, 3, 1).reshape(B1 * T1, 512, 16)
                with torch.no_grad(), _ac():
                    img = _tok.decode(lat, 1).squeeze(1).clamp(0, 1)
                    # byte-quantise, exactly DIAMOND denoiser.py:83
                    img = (img * 255).round() / 255.0
                    re = _tok.encode(img)["latent"].float()
                return re.reshape(B1 * T1, 16, 16, 32).permute(0, 3, 1, 2).reshape(B1, T1, C1, H1, W1)
        acc = {}
        t0 = __import__("time").time()
        for b0 in range(0, len(samples), args.batch):
            chunk = samples[b0:b0 + args.batch]
            zc, zt, ac_ctx_idx, ac_fut_idx, gnames = [], [], [], [], []
            for g, s in chunk:
                z, _ = games[g]
                zc.append(z[s - Ctx:s].float())
                zt.append(z[s:s + H].float())
                ac_ctx_idx.append(torch.arange(s - Ctx, s))
                ac_fut_idx.append(torch.arange(s, s + H))
                gnames.append(g)
            z_ctx = torch.stack(zc).to(dev)
            z_true = torch.stack(zt).to(dev)
            # actions are per-game, so slice per sample then stack
            a_ctx = {k: torch.stack([games[g][1][k][i] for (g, _), i in zip(chunk, ac_ctx_idx)]).to(dev)
                     for k in ["movement", "cursor_valid"] + ABILITY_KEYS}
            a_fut = {k: torch.stack([games[g][1][k][i] for (g, _), i in zip(chunk, ac_fut_idx)]).to(dev)
                     for k in ["movement", "cursor_valid"] + ABILITY_KEYS}
            a_fut_used = transform_actions(a_fut, act_how)
            torch.manual_seed(seed + b0)
            with ac():
                pred = phase3_rollout(net, z_ctx, a_ctx, a_fut_used, H, device=dev,
                                      z_true=z_true, reproject=reproj, **kw).float()
            m = latent_metrics(pred, z_true, z_ctx[:, -1:])
            if args.pixels:
                m.update(pixel_metrics(tok, pred, z_true, ac))
            for k, v in m.items():
                acc.setdefault(k, []).append(v)
            if args.save_latents:
                saved_lat.setdefault(arm_name, []).append(pred.half().cpu().numpy())
            print(f"  [{arm_name}] {b0 + len(chunk)}/{len(samples)} "
                  f"nmse@1={m['nmse'][:,0].mean():.4f} nmse@8={m['nmse'][:,min(7,H-1)].mean():.4f} "
                  f"nmse@{H}={m['nmse'][:,-1].mean():.4f}", flush=True)
        results[arm_name] = {k: np.concatenate(v, 0) for k, v in acc.items()}
        dt = __import__("time").time() - t0
        print(f"[arm {arm_name}] done in {dt:.0f}s "
              f"({dt/max(1,len(samples)):.1f}s/rollout)", flush=True)

    # ---- non-model controls, computed once ----
    ctrl = {}
    zc_all = torch.stack([games[g][0][s - 1].float() for g, s in samples])          # last ctx frame
    zt_all = torch.stack([games[g][0][s:s + H].float() for g, s in samples])        # (n,H,...)
    ref = zc_all.unsqueeze(1)
    ctrl["copy"] = latent_metrics(zc_all.unsqueeze(1).expand(-1, H, -1, -1, -1).contiguous(),
                                  zt_all, ref)
    shuf = torch.stack([games[g][0][rng.integers(0, games[g][0].shape[0] - H)
                                    :][:H].float() for g, s in samples])
    ctrl["shuffle"] = latent_metrics(shuf, zt_all, ref)
    gm, gs = zt_all.mean(), zt_all.std()
    ctrl["noise"] = latent_metrics(torch.randn_like(zt_all) * gs + gm, zt_all, ref)
    if args.pixels:
        for cname, cz in [("copy", zc_all.unsqueeze(1).expand(-1, H, -1, -1, -1).contiguous()),
                          ("shuffle", shuf)]:
            ctrl[cname].update(pixel_metrics(tok, cz.to(dev), zt_all.to(dev), ac))
    results.update({f"CTRL_{k}": v for k, v in ctrl.items()})

    out = {}
    for arm, d in results.items():
        for k, v in d.items():
            out[f"{arm}/{k}"] = v
    out["_games"] = np.array([g for g, _ in samples])
    out["_starts"] = np.array([s for _, s in samples])
    np.savez_compressed(args.out, **out)
    print(f"\nwrote {args.out}  ({len(samples)} rollouts x H={H})")
    if args.save_latents:
        for arm, chunks in saved_lat.items():
            p = args.out.replace(".npz", f".lat_{arm}.npy")
            np.save(p, np.concatenate(chunks, 0))
            print(f"wrote {p}")

    # ---- console summary ----
    hs = [h for h in [1, 2, 4, 8, 12, 16, 24, 32, 48, 64] if h <= H]
    print(f"\n{'arm':<14} " + " ".join(f"h={h:<5d}" for h in hs) + "   (latent NMSE, 1.0 = predict-the-mean)")
    for arm, d in results.items():
        row = " ".join(f"{d['nmse'][:, h-1].mean():<7.4f}" for h in hs)
        print(f"{arm:<14} {row}")


if __name__ == "__main__":
    main()
