"""Counterfactual-action probe: does the world model actually OBEY actions?

The whole agent phase rides on the dream reacting correctly when the policy
changes its action. Our replay action labels are lossy, so the model could have
learned to ignore the action token. This isolates action-effect from noise:

  fix the noise seed, run the SAME context through the rollout under different
  action_future sets:
    - REAL   : the actions that actually happened
    - ZERO   : no movement, no presses
    - FLIP   : movement negated (champion should go the other way)
    - WRONG  : a different clip's real actions
  and a NOISE baseline: REAL actions, a *different* seed.

Metric (per predicted frame, latent RMS from the REAL/seed0 dream):
  action_effect = ||dream(REAL) - dream(COUNTERFACTUAL)||   (same noise, diff actions)
  noise_effect  = ||dream(REAL,seed0) - dream(REAL,seed1)|| (same actions, diff noise)
Verdict: action_effect / noise_effect >> 1  => the model responds to actions.
         ~1 or less                          => it ignores them (agent phase blocked).

Run:  PYTHONPATH=src python scratchpad/probe_action.py --checkpoint <ckpt> \
        --latents-dir rollout_stage --labels-root /srv/nfs/datasets/lol_replays_16_9_772 --device cuda
"""
import argparse, glob, os, tempfile
import numpy as np
import torch
from torch.utils.data import default_collate

from ahriuwu.models.dynamics import create_dynamics
from ahriuwu.data.replay_dataset import ReplayLatentSequenceDataset


def rms(a, b):
    return (a - b).pow(2).mean().sqrt().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--latents-dir", required=True)
    ap.add_argument("--labels-root", required=True)
    ap.add_argument("--match", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ctx", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--num-steps", type=int, default=6)
    args = ap.parse_args()
    dev = args.device

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"checkpoint step={ck.get('global_step')} loss={ck.get('loss'):.4f}")
    m = create_dynamics("medium", latent_dim=32, use_actions=True, num_kv_heads=4,
                        num_register_tokens=8, soft_cap=50.0, use_qk_norm=True).to(dev).eval()
    sd = ck["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    miss, unexp = m.load_state_dict(sd, strict=False)
    print(f"load: missing(non-rope)={len([x for x in miss if 'rope' not in x.lower()])} unexpected={len(unexp)}")

    match = args.match or os.path.basename(sorted(glob.glob(f"{args.latents_dir}/*.pt"))[0])[:-3]
    tmp = tempfile.mkdtemp()
    os.symlink(os.path.abspath(f"{args.latents_dir}/{match}.pt"), f"{tmp}/{match}.pt")
    seq = args.ctx + args.horizon
    ds = ReplayLatentSequenceDataset(latents_dir=tmp, labels_root=args.labels_root,
                                     outcomes={match: False}, sequence_length=seq, stride=seq)
    # pick the most ACTIVE clip (most movement + presses) — where actions should matter most
    best_i, best = 0, -1.0
    for i in range(0, len(ds), max(1, len(ds) // 40)):
        a = ds[i]["actions"]
        s = a["movement"].std().item() + sum(int(a[k].sum()) for k in a if k != "movement")
        if s > best:
            best_i, best = i, s
    vb = default_collate([ds[best_i]])
    # a DIFFERENT clip for the WRONG-actions counterfactual
    other = default_collate([ds[(best_i + len(ds) // 2) % len(ds)]])

    z = vb["latents"].to(dev).float()
    A = {k: (v.to(dev).float() if k == "movement" else v.to(dev)) for k, v in vb["actions"].items()}
    Ao = {k: (v.to(dev).float() if k == "movement" else v.to(dev)) for k, v in other["actions"].items()}
    ctx = z[:, :args.ctx]
    ac = {k: v[:, :args.ctx] for k, v in A.items()}
    H = args.horizon
    presses = {k: int(A[k][:, args.ctx:args.ctx + H].sum()) for k in A if k != "movement"}
    presses = {k: v for k, v in presses.items() if v}
    print(f"{match} clip: movement_std={A['movement'][:, args.ctx:].std():.3f} future_presses={presses}")

    def fut(d, k):
        return d[k][:, args.ctx:args.ctx + H]

    real = {k: fut(A, k) for k in A}
    zero = {k: torch.zeros_like(v) for k, v in real.items()}
    flip = {**real, "movement": -real["movement"]}
    wrong = {k: fut(Ao, k) for k in Ao}

    def roll(af, seed):
        torch.manual_seed(seed)
        with torch.no_grad():
            return m.rollout(ctx, predict_frames=H, num_steps=args.num_steps, k_max=args.num_steps,
                             tau_ctx=0.1, actions_context=ac, actions_future=af, device=dev).float()

    d_real0 = roll(real, 0)
    d_real1 = roll(real, 1)     # noise baseline (same actions, diff noise)
    d_zero = roll(zero, 0)      # same noise, no actions
    d_flip = roll(flip, 0)      # same noise, movement reversed
    d_wrong = roll(wrong, 0)    # same noise, a different clip's actions

    print(f"\nframe |  noise  | act:ZERO | act:FLIPmove | act:WRONG   (latent RMS from REAL/seed0)")
    accN = accZ = accF = accW = 0.0
    for t in range(H):
        n = rms(d_real0[:, t], d_real1[:, t])
        z_ = rms(d_real0[:, t], d_zero[:, t])
        f = rms(d_real0[:, t], d_flip[:, t])
        w = rms(d_real0[:, t], d_wrong[:, t])
        accN += n; accZ += z_; accF += f; accW += w
        print(f"  {t+1:3d} | {n:.4f} |  {z_:.4f}  |    {f:.4f}    |  {w:.4f}")
    n, z_, f, w = accN / H, accZ / H, accF / H, accW / H
    print(f"\nmean  | {n:.4f} |  {z_:.4f}  |    {f:.4f}    |  {w:.4f}")
    print(f"\nACTION/NOISE ratio:  ZERO={z_/max(n,1e-9):.2f}x  FLIP={f/max(n,1e-9):.2f}x  WRONG={w/max(n,1e-9):.2f}x")
    print("  >>1  => the dream changes MORE from actions than from noise = model OBEYS actions.")
    print("  ~1   => actions are drowned by noise / weakly used.")
    print("  <1   => actions barely register = world model IGNORES actions (agent phase blocked).")


if __name__ == "__main__":
    main()
