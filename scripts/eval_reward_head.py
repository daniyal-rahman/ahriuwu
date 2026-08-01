#!/usr/bin/env python3
"""Can the Phase-2 reward head actually read solo-gold reward off the latents?

Go/no-go for Phase-3 imagination: in dreams, reward exists ONLY as the reward
head's prediction from agent tokens. Gold is NOT on the Garen-POV screen, so the
head must infer Δgold from visual correlates (last-hit events, wave state, gold
popups). This runs the frozen Phase-2 dynamics+reward head over real latent
windows (same tau regime as training) and scores predictions against the true
solo-gold reward from labels.json.

Metrics:
  corr/R2       overall predicted-vs-true per-frame reward
  event AUC     can predictions rank last-hit frames (Δgold>=10) above
                no-income frames? (the decision-relevant signal)
  means         predicted reward on event vs non-event frames

    PYTHONPATH=src python scripts/eval_reward_head.py \
        --phase2-ckpt data/phase2_bc_garen/agent_finetune_latest.pt \
        --matches NA1_5549995114 --latents-dir /srv/nfs/datasets/replay_latents_v7_bc
"""
import argparse
import glob
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, "src")
from ahriuwu.constants import ABILITY_KEYS
from ahriuwu.models import create_dynamics, RewardHead, DiffusionSchedule
from ahriuwu.data.replay_dataset import ReplayLatentSequenceDataset


def load_phase2(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    cfg = ck.get("dynamics_config") or {}
    dyn_state = ck["dynamics_state_dict"]
    if any(k.startswith("_orig_mod.") for k in dyn_state):
        dyn_state = {k.replace("_orig_mod.", ""): v for k, v in dyn_state.items()}
    use_actions = cfg.get("use_actions", any("action_embed." in k for k in dyn_state))
    dyn = create_dynamics(
        size=a.get("model_size", "medium"), latent_dim=cfg.get("latent_dim", a.get("latent_dim", 32)),
        use_agent_tokens=True, use_actions=use_actions, num_tasks=1,
        agent_layers=a.get("agent_layers", 4), use_qk_norm=not a.get("no_qk_norm", False),
        soft_cap=a.get("soft_cap", 50.0) or None,
        num_register_tokens=a.get("num_register_tokens", 8),
        num_kv_heads=a.get("num_kv_heads", None),
    ).to(device)
    miss, unexp = dyn.load_state_dict(dyn_state, strict=False)
    miss = [m for m in miss if "rope" not in m.lower()]
    assert len(miss) + len(unexp) <= 10, f"ARCH MISMATCH miss={miss[:5]} unexp={unexp[:5]}"
    dyn.eval().requires_grad_(False)
    rh = RewardHead(input_dim=dyn.model_dim, hidden_dim=a.get("hidden_dim", 256),
                    num_buckets=a.get("num_buckets", 255),
                    mtp_length=a.get("mtp_length", 9)).to(device)
    rh.load_state_dict(ck["reward_head_state_dict"])
    rh.eval().requires_grad_(False)
    print(f"phase2 {path}: gs={ck.get('global_step')} epoch={ck.get('epoch')} "
          f"use_actions={use_actions} tau_ctx={a.get('tau_ctx', 0.9)}")
    return dyn, rh, a.get("tau_ctx", 0.9)


def auc(pos, neg):
    """Rank AUC: P(pred_pos > pred_neg)."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(np.float64)
    rp = ranks[: len(pos)]
    return float((rp.sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase2-ckpt", default="data/phase2_bc_garen/agent_finetune_latest.pt")
    ap.add_argument("--matches", nargs="+", default=["NA1_5549995114"])
    ap.add_argument("--latents-dir", default="/srv/nfs/datasets/replay_latents_v7_bc")
    ap.add_argument("--labels-root", default="/srv/nfs/datasets/lol_replays_16_9_772")
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--max-windows", type=int, default=30, help="per match")
    ap.add_argument("--event-thresh", type=float, default=0.010,
                    help="true reward >= this counts as an income event (gold>=10)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    torch.manual_seed(0)

    dyn, rh, tau_ctx = load_phase2(args.phase2_ckpt, dev)
    schedule = DiffusionSchedule(device=dev)

    preds, trues = [], []
    for match in args.matches:
        tmp = tempfile.mkdtemp()
        src = glob.glob(f"{args.latents_dir}/{match}.pt")[0]
        os.symlink(os.path.abspath(src), f"{tmp}/{match}.pt")
        ds = ReplayLatentSequenceDataset(
            latents_dir=tmp, labels_root=args.labels_root, outcomes={match: False},
            sequence_length=args.seq_len, stride=args.seq_len)
        idx = np.linspace(0, len(ds) - 1, min(args.max_windows, len(ds))).astype(int)
        for i in idx:
            s = ds[int(i)]
            z0 = s["latents"].float().unsqueeze(0).to(dev)          # (1,T,C,H,W)
            r_true = s["rewards"].float()                            # (T,)
            T = z0.shape[1]
            tau = tau_ctx + torch.rand(1, T, device=dev) * (1.0 - tau_ctx)
            z_noisy, _ = schedule.add_noise(z0, tau)
            actions = None
            if dyn.use_actions:
                actions = {"movement": s["actions"]["movement"].float().unsqueeze(0).to(dev)}
                for k in ABILITY_KEYS:
                    actions[k] = s["actions"][k].long().unsqueeze(0).to(dev)
            d_one = torch.ones(1, dtype=torch.long, device=dev)
            with torch.no_grad():
                _, agent_out = dyn(z_noisy, tau, step_size=d_one, actions=actions)
                pred = rh.predict(agent_out)[0, :, 0]                # offset n=0, orig scale
            preds.append(pred.float().cpu().numpy())
            trues.append(r_true.numpy())
        print(f"  {match}: {len(idx)} windows x {args.seq_len} frames", flush=True)

    p = np.concatenate(preds)
    t = np.concatenate(trues)
    ev = t >= args.event_thresh
    nv = t <= 0.002
    corr = float(np.corrcoef(p, t)[0, 1]) if p.std() > 0 else 0.0
    ss = 1 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum()
    a = auc(p[ev], p[nv])
    print(f"\nframes={len(t)}  events(gold>={args.event_thresh*1000:.0f})={ev.sum()} "
          f"({ev.mean():.1%})  no-income={nv.sum()}")
    print(f"true reward:  mean={t.mean():.5f}  p50={np.median(t):.5f}  p99={np.quantile(t,0.99):.5f}")
    print(f"pred reward:  mean={p.mean():.5f}  std={p.std():.5f}")
    print(f"corr(pred,true)      = {corr:+.3f}")
    print(f"R2 (vs predict-mean) = {ss:+.3f}")
    print(f"event AUC            = {a:.3f}   (0.5=blind, >0.75=usable, >0.9=strong)")
    print(f"mean pred | event    = {p[ev].mean() if ev.any() else float('nan'):.5f}")
    print(f"mean pred | no-event = {p[nv].mean() if nv.any() else float('nan'):.5f}")
    print("\nread: AUC~0.5 => reward head is blind to income events -> Phase-3 imagination")
    print("has no learning signal (fix reward pathway first). High AUC => reward is readable;")
    print("Phase-3 credit assignment can work.")


if __name__ == "__main__":
    main()
