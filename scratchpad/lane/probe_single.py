#!/usr/bin/env python3
"""Single-shot probe: what direction does the policy command for THIS window?

Under movement_action_mode='none' the movement action never enters the model, so
there is NO recurrence through the policy's own output -- one forward per sampled
frame is the complete answer, and windows from many games/times batch together.
That buys ~50x the samples of a self-fed rollout for the same GPU time, and makes
the INTERVENTION test trivial: hold the action history fixed and swap the latent
window (does the output track pixels?), or hold the latents and swap the history
(does it track its own past?).

History modes:
  none         cursor_valid all False -- the movement action never reaches it
  sentinel     movement=(0.5,0.5) valid -- EXACTLY what BC teacher-forced on
               every pre-first-click frame (clicks.json starts at ~60s)
  const:X,Y    movement=(X,Y) held on every slot, valid -- the intervention arm
"""
import argparse, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from agent_infer import GarenAgent                                    # noqa: E402
from ahriuwu.constants import ABILITY_KEYS                             # noqa: E402


def build_windows(lat, ts, T):
    """lat (Nfr, C,16,16) -> (len(ts), T, C,16,16); left-pad with the oldest frame
    exactly as GarenAgent does when the buffer is not yet full."""
    out = []
    for t in ts:
        i0 = t - T + 1
        if i0 >= 0:
            w = lat[i0:t + 1]
        else:
            w = torch.cat([lat[0:1].repeat(-i0, 1, 1, 1), lat[0:t + 1]], 0)
        out.append(w)
    return torch.stack(out)


@torch.no_grad()
def run(agent, win, hist, temperature=0.0):
    """win (B,T,C,16,16) float32 on device. Returns movement logits + gate logit."""
    dev = agent.device
    B, T = win.shape[:2]
    actions = None
    if agent.use_actions:
        if hist == "none":
            mv = torch.full((B, T, 2), 0.5, device=dev)
            cv = torch.zeros(B, T, dtype=torch.bool, device=dev)
        elif hist == "sentinel":
            mv = torch.full((B, T, 2), 0.5, device=dev)
            cv = torch.ones(B, T, dtype=torch.bool, device=dev)
        elif hist.startswith("const:"):
            x, y = (float(v) for v in hist.split(":")[1].split(","))
            mv = torch.tensor([x, y], device=dev).view(1, 1, 2).expand(B, T, 2).contiguous()
            cv = torch.ones(B, T, dtype=torch.bool, device=dev)
        else:
            raise ValueError(hist)
        actions = {"movement": mv, "cursor_valid": cv}
        for k in ABILITY_KEYS:
            actions[k] = torch.zeros(B, T, dtype=torch.long, device=dev)
    tau = agent.tau_ctx + torch.rand(B, T, device=dev) * (1.0 - agent.tau_ctx)
    z_tau, _ = agent.sched.add_noise(win, tau)
    d_one = torch.ones(B, dtype=torch.long, device=dev)
    with agent._ac():
        _, ag = agent.dyn(z_tau, tau, step_size=d_one, actions=actions)
        h = ag[:, -1:, :]
        al, ml = agent.policy(h)
        gl = agent.policy.gate_logits(h)
    n = 1 if agent.mtp > 1 else 0
    return (ml[:, 0, n].float().cpu().numpy(), gl[:, 0, n].float().cpu().numpy(),
            al[:, 0, n].float().cpu().numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/mnt/nfs/projects/ahriuwu/data/phase2_bc_clicks/agent_finetune_latest.pt")
    ap.add_argument("--latents-dir", default="/mnt/nfs/datasets/replay_latents_v7_bc")
    ap.add_argument("--matches", default=None)
    ap.add_argument("--times", default="30,80,140,200,260,320,400,500,600,700,800,900,1000,1100,1190",
                    help="frame indices to probe (walk-out grid by default)")
    ap.add_argument("--hists", default="none,sentinel,const:0.9,0.5,const:0.1,0.5")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="walkout")
    ap.add_argument("--out", default="/mnt/nfs/projects/ahriuwu/scratchpad/lane/probe")
    ap.add_argument("--movement-action-mode", choices=["held", "event_only", "none"],
                    default=None,
                    help="Override the checkpoint's movement-action input. 'none' cuts "
                         "the crutch so the direction number reflects PIXELS rather than "
                         "the model copying its own last order.")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    agent = GarenAgent(args.ckpt, context=16, device=args.device)
    # Override the checkpoint's trained action mode. Without this the probe always
    # scores the model WITH its movement-action crutch, which is the one condition
    # under which the direction number is meaningless -- a blind copier beats the
    # measured permutation null by 26 sigma. The chained direction eval passed this
    # flag to a script that did not accept it and crashed on all four A/B arms,
    # producing no direction number at all.
    if getattr(args, "movement_action_mode", None):
        agent.movement_action_mode = args.movement_action_mode
        print(f"[probe] movement_action_mode -> {args.movement_action_mode}", flush=True)
    T = agent.context
    ts = [int(v) for v in args.times.split(",")]
    # 'const:X,Y' contains a comma, so split the hist list on ';' if present
    hists = args.hists.split(";") if ";" in args.hists else \
        [h for h in _split_hists(args.hists)]
    mids = ([m.strip() for m in args.matches.split(",")] if args.matches
            else sorted(p.stem for p in Path(args.latents_dir).glob("*.pt")))

    res = {h: [] for h in hists}
    keep_mid, keep_t = [], []
    for m in mids:
        p = Path(args.latents_dir) / f"{m}.pt"
        lat = torch.load(p, map_location="cpu", weights_only=True)["latents"]
        good = [t for t in ts if t < lat.shape[0]]
        if not good:
            continue
        win = build_windows(lat, good, T)
        keep_mid += [m] * len(good); keep_t += good
        for h in hists:
            mls, gls, als = [], [], []
            for s in range(0, len(good), args.batch):
                w = win[s:s + args.batch].to(args.device, torch.float32)
                ml, gl, al = run(agent, w, h)
                mls.append(ml); gls.append(gl); als.append(al)
            res[h].append((np.concatenate(mls), np.concatenate(gls), np.concatenate(als)))
        print(f"  {m}: {len(good)} windows x {len(hists)} hists", flush=True)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    save = dict(match=np.array(keep_mid), t=np.array(keep_t))
    for h in hists:
        save[f"ml__{h}"] = np.concatenate([r[0] for r in res[h]])
        save[f"gl__{h}"] = np.concatenate([r[1] for r in res[h]])
        save[f"al__{h}"] = np.concatenate([r[2] for r in res[h]])
    f = Path(args.out) / f"probe_{args.tag}.npz"
    np.savez_compressed(f, **save)
    print("wrote", f, len(keep_mid), "windows")


def _split_hists(s):
    """split a comma list while keeping 'const:X,Y' together."""
    parts, buf = [], []
    for tok in s.split(","):
        if buf:
            buf.append(tok); parts.append(",".join(buf)); buf = []
        elif tok.startswith("const:"):
            buf = [tok]
        else:
            parts.append(tok)
    if buf:
        parts.append(",".join(buf))
    return parts


if __name__ == "__main__":
    main()
