#!/usr/bin/env python3
"""Offline inference engine for the Garen agent (Phase-2 BC policy).

Clean modern replacement for the archived scripts/_archive/.../play_live.py, wired
to the CURRENT stack: v7 tokenizer (load_v7) -> dynamics with agent tokens -> the
factorized PolicyHead. Screen capture / input injection are intentionally OUT of
scope here (that's the Windows layer); this module is the pure, testable core:

    frame (or precomputed latent) -> GarenAgent.act(...) -> action dict

Inference contract (matches BC training exactly, train_agent_finetune.run_step):
  * keep a rolling window of the last `context` observed latents;
  * corrupt the window to near-clean tau ~ U(tau_ctx, 1) (tau_ctx=0.9), the same
    in-distribution regime the frozen denoiser was finetuned under;
  * ONE dynamics forward with agent tokens on (no future-frame denoising loop —
    BC reads the CURRENT window's agent token, it does not roll out);
  * read the policy at MTP offset 1 (BC trains offsets n>=1; n=0 is the dropped
    same-frame leak), i.e. the action to take NEXT.

Test offline on a real replay (no screen needed):
    PYTHONPATH=src python scripts/agent_infer.py --test-latents \
        --phase2-ckpt data/phase2_bc_garen/agent_finetune_latest.pt \
        --latents /scratch/ahriuwu/dynamics_replay_latents_v7_dim32/NA1_5549995114.pt \
        --labels-root /srv/nfs/datasets/lol_replays_16_9_772 --frames 300
"""
import argparse
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from ahriuwu.constants import ABILITY_KEYS, MOVEMENT_DIM
from ahriuwu.models import create_dynamics, PolicyHead, RewardHead, DiffusionSchedule


def _dyn_from_tok(latent_256):
    """(B, num_latents=256, latent_dim) -> (B, latent_dim, 16, 16) dynamics grid."""
    B, N, C = latent_256.shape
    return latent_256.view(B, 16, 16, C).permute(0, 3, 1, 2).contiguous()


class GarenAgent:
    """Load a Phase-2 checkpoint and turn observed frames/latents into actions."""

    def __init__(self, phase2_ckpt, tokenizer_ckpt=None, context=32, tau_ctx=0.9,
                 device="cuda", init_only=False):
        self.device = device
        self.context = context
        self.tau_ctx = tau_ctx
        self.buf = deque(maxlen=context)
        self.sched = DiffusionSchedule(device=device)

        ck = torch.load(phase2_ckpt, map_location="cpu", weights_only=False) if not init_only else {}
        a = ck.get("args", {}) if isinstance(ck.get("args", {}), dict) else vars(ck.get("args"))
        self.latent_dim = a.get("latent_dim", 32)
        self.mtp = a.get("mtp_length", 9)
        self.movement_bins = a.get("movement_bins", 21)
        hidden = a.get("hidden_dim", 256)
        num_buckets = a.get("num_buckets", 255)
        size = a.get("model_size", "medium")

        # dynamics backbone with agent tokens (weights incl. trained agent blocks)
        self.dyn = create_dynamics(
            size=size, latent_dim=self.latent_dim, use_agent_tokens=True,
            use_actions=a.get("use_actions", False), num_tasks=1,
            agent_layers=a.get("agent_layers", 4), use_qk_norm=not a.get("no_qk_norm", False),
            soft_cap=a.get("soft_cap", 50.0) if a.get("soft_cap", 50.0) > 0 else None,
            num_register_tokens=a.get("num_register_tokens", 8),
            num_kv_heads=a.get("num_kv_heads", 4), gradient_checkpointing=False,
        ).to(device).eval()
        model_dim = self.dyn.model_dim

        self.policy = PolicyHead(input_dim=model_dim, num_abilities=len(ABILITY_KEYS),
                                 hidden_dim=hidden, mtp_length=self.mtp,
                                 movement_dim=MOVEMENT_DIM, movement_bins=self.movement_bins).to(device).eval()
        self.reward = RewardHead(input_dim=model_dim, hidden_dim=hidden,
                                 num_buckets=num_buckets, mtp_length=self.mtp).to(device).eval()

        if not init_only:
            self.dyn.load_state_dict(ck["dynamics_state_dict"], strict=False)
            self.policy.load_state_dict(ck["policy_head_state_dict"])
            if "reward_head_state_dict" in ck:
                self.reward.load_state_dict(ck["reward_head_state_dict"])
        for m in (self.dyn, self.policy, self.reward):
            m.requires_grad_(False)

        self.tok = None
        if tokenizer_ckpt:
            import sys
            sys.path.insert(0, "scripts")
            from pretokenize_replay_v7 import load_v7
            self.tok, _, _ = load_v7(tokenizer_ckpt, device)
            self.tok.requires_grad_(False)

        gs = ck.get("global_step") if not init_only else None
        print(f"[GarenAgent] loaded phase2 (step {gs}) | size={size} latent_dim={self.latent_dim} "
              f"mtp={self.mtp} bins={self.movement_bins} tokenizer={'yes' if self.tok else 'no'}")

    @torch.no_grad()
    def encode_frame(self, frame_rgb01):
        """(H,W,3) RGB in [0,1] -> (1, latent_dim, 16, 16). Requires a tokenizer."""
        assert self.tok is not None, "no tokenizer loaded; use act_from_latent()"
        import cv2
        f = cv2.resize(frame_rgb01, (352, 352), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(f).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        lat = self.tok.encode(x)["latent"]           # (1, 256, latent_dim)
        return _dyn_from_tok(lat)                     # (1, latent_dim, 16, 16)

    def reset(self):
        self.buf.clear()

    @torch.no_grad()
    def act_from_latent(self, latent, temperature=0.0):
        """latent: (1, latent_dim, 16, 16) for the newest observed frame -> action dict.
        temperature 0 = greedy (deterministic, best for a demo)."""
        self.buf.append(latent)
        window = list(self.buf)
        while len(window) < self.context:            # left-pad with the oldest frame
            window.insert(0, window[0])
        z0 = torch.stack(window, dim=1).squeeze(2) if window[0].dim() == 4 else torch.stack(window, dim=1)
        z0 = z0.to(self.device).float()              # (1, T, C, 16, 16)
        B, T = z0.shape[:2]

        # near-clean corruption, one forward, agent tokens on (BC regime)
        tau = self.tau_ctx + torch.rand(B, T, device=self.device) * (1.0 - self.tau_ctx)
        z_tau, _ = self.sched.add_noise(z0, tau)
        d_one = torch.ones(B, dtype=torch.long, device=self.device)
        _, agent_out = self.dyn(z_tau, tau, step_size=d_one, actions=None)
        h = agent_out[:, -1:, :]                      # newest frame's agent token (1,1,D)

        abil, move, _ = self.policy.sample(h, temperature=temperature)  # (1,1,L,A),(1,1,L,2)
        n = 1 if self.mtp > 1 else 0                  # MTP offset 1 = trained "next action"
        abilities = {k: bool(abil[0, 0, n, i].item() > 0.5) for i, k in enumerate(ABILITY_KEYS)}
        movement = tuple(float(v) for v in move[0, 0, n].tolist())
        rew = float(self.reward.predict(h)[0, 0, n].item())
        return {"abilities": abilities, "movement": movement, "reward_pred": rew}


# --------------------------------------------------------------------------- #
# Offline test: run the agent over a real replay's precomputed latents
# --------------------------------------------------------------------------- #

def test_latents(args):
    dev = args.device
    agent = GarenAgent(args.phase2_ckpt, tokenizer_ckpt=None, context=args.context,
                       device=dev, init_only=args.init_only)
    d = torch.load(args.latents, map_location="cpu", weights_only=True)
    lat = d["latents"].float()                        # (N, latent_dim, 16, 16)
    N = min(args.frames, lat.shape[0])
    print(f"replay {Path(args.latents).stem}: {lat.shape[0]} frames, running {N}")

    presses = {k: 0 for k in ABILITY_KEYS}
    moves = []
    t0 = time.perf_counter()
    agent.reset()
    for i in range(N):
        a = agent.act_from_latent(lat[i:i+1].to(dev), temperature=args.temperature)
        for k, v in a["abilities"].items():
            presses[k] += int(v)
        moves.append(a["movement"])
    dt = (time.perf_counter() - t0) / max(N, 1) * 1000
    moves = np.array(moves)
    print(f"  {dt:.1f} ms/frame ({1000/dt:.1f} fps) on {dev}")
    print(f"  ability presses / {N}: " + ", ".join(f"{k}={presses[k]}" for k in ABILITY_KEYS if presses[k]))
    print(f"  movement x mean/std={moves[:,0].mean():.3f}/{moves[:,0].std():.3f} "
          f"y mean/std={moves[:,1].mean():.3f}/{moves[:,1].std():.3f} "
          f"(unique cells={len(set(map(tuple, np.round(moves,2))))})")
    print("PLUMBING OK — encode(latent)->dynamics->agent_token->policy->action decoded end to end")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase2-ckpt", default="data/phase2_bc_garen/agent_finetune_latest.pt")
    ap.add_argument("--latents", default="/scratch/ahriuwu/dynamics_replay_latents_v7_dim32/NA1_5549995114.pt")
    ap.add_argument("--labels-root", default="/srv/nfs/datasets/lol_replays_16_9_772")
    ap.add_argument("--context", type=int, default=32)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--test-latents", action="store_true")
    ap.add_argument("--init-only", action="store_true",
                    help="Build with fresh (untrained) heads to validate plumbing before BC finishes.")
    args = ap.parse_args()
    test_latents(args)


if __name__ == "__main__":
    main()
