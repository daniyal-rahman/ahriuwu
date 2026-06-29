#!/usr/bin/env python3
"""Phase 2: Agent Finetuning - Behavior Cloning + Reward Modeling.

DreamerV4 Section 3.3 "Behavior cloning and reward model". The dynamics model
(Phase 1) is the FROZEN backbone here: we run it (with agent tokens + action
conditioning) on real replay latents to get one agent token per frame, then
train two small heads on those tokens:

- ``PolicyHead`` via behavioral cloning on the NEXT-frame action — both the 9
  independent ability Bernoullis AND the binned (x, y) movement categoricals,
  by log-probability. (See the label-leakage note below for why NEXT-frame.)
- ``RewardHead`` via twohot multi-token-prediction (MTP) of the solo-gold
  reward at offsets n = 0..L-1.

Loss = bc_loss + reward_loss (each normalized by its running RMS, then summed).

LABEL-LEAKAGE FIX (vs the archived trainer): the dynamics is *action-conditioned*
— the agent token at frame t is built from a window whose frame-t input already
contains action a_t. So predicting a_t from agent_out[:, t] (the old n=0 BC term)
trivially leaks. BC here predicts the NEXT actions: MTP head n (n >= 1) predicts
a_{t+n}, and the n=0 term is dropped. The reward target is never a model input,
so the reward MTP keeps the full n = 0..L-1.

Usage (real run on the GPU node):
    PYTHONPATH=src python scripts/train_agent_finetune.py \
        --dynamics-checkpoint checkpoints/dynamics_best.pt \
        --latents-dir /opt/ahriuwu/latents_pt \
        --labels-root /mnt/storage/ahriuwu-data/replays \
        --epochs 1

Smoke test (CPU, synthetic, no checkpoint/data needed):
    PYTHONPATH=src python scripts/train_agent_finetune.py --smoke-test

Reference: DreamerV4 Section 3.3 "Behavior cloning and reward model".
"""

import argparse
import glob
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from ahriuwu.constants import ABILITY_KEYS, MOVEMENT_DIM
from ahriuwu.models import (
    create_dynamics,
    RewardHead,
    PolicyHead,
    DiffusionSchedule,
    symlog,
    twohot_loss,
    RunningRMS,
)
from ahriuwu.data.dataset import VideoGroupedSampler
from ahriuwu.utils.logging import add_wandb_args, init_wandb, log_step, finish_wandb
from ahriuwu.utils.training import (
    add_training_args, create_optimizer, create_wsd_schedule,
)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2: Agent Finetuning (BC + Reward)")
    add_training_args(parser)
    parser.add_argument(
        "--dynamics-checkpoint", type=str, default=None,
        help="Path to the Phase 1 dynamics checkpoint (frozen backbone). "
             "Omit only with --init-dynamics / --smoke-test.",
    )
    parser.add_argument(
        "--init-dynamics", action="store_true",
        help="Build a fresh (untrained) dynamics backbone instead of loading a "
             "checkpoint. For wiring/smoke tests only — produces garbage tokens.",
    )
    parser.add_argument(
        "--latents-dir", type=str, default=None,
        help="Dir of packed per-match latents (<match>.pt) — same format Phase 1 uses.",
    )
    parser.add_argument(
        "--labels-root", type=str, default=None,
        help="Dir of <match>/labels.json + clicks.json (replay action + reward labels).",
    )
    parser.add_argument(
        "--manifest", type=str, default=None,
        help="Optional outcomes manifest (garen_win per match). If omitted, dummy "
             "outcomes (all False) are used — fine for solo-gold reward, which "
             "ignores win/loss.",
    )
    parser.add_argument("--model-size", type=str, default="small",
                        choices=["tiny", "small", "medium", "large"])
    parser.add_argument("--latent-dim", type=int, default=32,
                        help="Latent dim per token (must match the tokenizer/dynamics ckpt).")
    parser.add_argument("--seq-len", type=int, default=32, help="Frames per sequence.")
    parser.add_argument("--stride", type=int, default=8, help="Stride between windows.")
    parser.add_argument("--mtp-length", type=int, default=9,
                        help="MTP heads (paper Eq 9: n=0..L with L=8 -> 9).")
    parser.add_argument("--num-buckets", type=int, default=255,
                        help="Twohot buckets for the reward head.")
    parser.add_argument("--movement-bins", type=int, default=21,
                        help="Per-axis movement bins in the policy head.")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Head MLP width.")
    # Dynamics architecture flags (must match the checkpoint's build).
    parser.add_argument("--no-qk-norm", action="store_true")
    parser.add_argument("--soft-cap", type=float, default=50.0)
    parser.add_argument("--num-register-tokens", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=None)
    parser.add_argument("--agent-layers", type=int, default=4)
    parser.add_argument("--tau-ctx", type=float, default=0.9,
                        help="Near-clean context corruption: per-frame tau ~ U(tau_ctx, 1).")
    # Smoke test
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run a tiny synthetic CPU train step end-to-end + assert "
                             "movement_heads receive BC gradient. No data/ckpt needed.")
    parser.set_defaults(num_workers=0, wandb=False)
    add_wandb_args(parser)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Backbone loading
# ---------------------------------------------------------------------------

def build_dynamics(args, *, use_actions: bool, device: str):
    """Create a dynamics backbone (agent tokens ON) matching the CLI arch flags."""
    return create_dynamics(
        size=args.model_size,
        latent_dim=args.latent_dim,
        use_agent_tokens=True,
        use_actions=use_actions,
        num_tasks=1,
        agent_layers=args.agent_layers,
        use_qk_norm=not args.no_qk_norm,
        soft_cap=args.soft_cap if args.soft_cap > 0 else None,
        num_register_tokens=args.num_register_tokens,
        num_kv_heads=args.num_kv_heads,
        gradient_checkpointing=False,
    ).to(device)


def load_frozen_dynamics(args, device: str):
    """Load the Phase 1 dynamics as a FROZEN backbone with agent tokens enabled.

    The Phase 1 checkpoint usually has no agent-token / reward-head weights (those
    are trained here), so non-matching keys are loaded non-strictly: the agent
    blocks start from their init and get trained, the diffusion backbone is the
    pretrained one. Returns (dynamics, use_actions).
    """
    if args.init_dynamics or args.dynamics_checkpoint is None:
        # Fresh backbone (smoke / wiring). Enable actions so the action path is exercised.
        dyn = build_dynamics(args, use_actions=True, device=device)
        print("  [init] fresh untrained dynamics backbone (no checkpoint loaded)")
    else:
        ckpt = torch.load(args.dynamics_checkpoint, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        if any(k.startswith("_orig_mod.") for k in state):
            state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        # Honor the checkpoint's own resolved config where available.
        cfg = ckpt.get("model_config") or {}
        latent_dim = cfg.get("latent_dim", args.latent_dim)
        if latent_dim != args.latent_dim:
            print(f"  [ckpt] overriding --latent-dim {args.latent_dim} -> {latent_dim} (from ckpt config)")
            args.latent_dim = latent_dim
        use_actions = cfg.get("use_actions", any("action_embed" in k for k in state))
        dyn = build_dynamics(args, use_actions=use_actions, device=device)
        missing, unexpected = dyn.load_state_dict(state, strict=False)
        print(f"  [ckpt] loaded {args.dynamics_checkpoint}")
        print(f"  [ckpt] use_actions={use_actions}; {len(missing)} missing "
              f"(agent/new) / {len(unexpected)} unexpected keys")

    dyn.eval()
    dyn.requires_grad_(False)  # FROZEN backbone
    return dyn, dyn.use_actions


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_dataset(args):
    """ReplayLatentSequenceDataset over real replay latents (Phase-1 style).

    Imported lazily to dodge the circular import that keeps it out of
    ahriuwu.data's __init__.
    """
    from ahriuwu.data.replay_dataset import ReplayLatentSequenceDataset

    if not args.latents_dir or not args.labels_root:
        raise SystemExit("Real run needs --latents-dir and --labels-root "
                         "(or use --smoke-test).")
    outcomes = None
    if args.manifest:
        outcomes = None  # let the dataset read it
    else:
        mids = [Path(p).stem for p in glob.glob(str(Path(args.latents_dir) / "*.pt"))
                if Path(p).stem != "index"]
        outcomes = {m: False for m in mids}  # solo-gold ignores win/loss
        print(f"  [data] {len(mids)} matches; dummy outcomes (solo-gold reward ignores them)")
    return ReplayLatentSequenceDataset(
        latents_dir=args.latents_dir,
        labels_root=args.labels_root,
        outcomes=outcomes,
        manifest_path=args.manifest,
        sequence_length=args.seq_len,
        stride=args.stride,
    )


def actions_to_device(actions: dict, device: str) -> dict:
    """Move the dataset action dict (movement + per-ability) to ``device``."""
    return {k: v.to(device) for k, v in actions.items()}


def stack_ability_targets(actions: dict, device: str) -> torch.Tensor:
    """(B, T, num_abilities) float {0,1} from the per-ability action tensors."""
    return torch.stack(
        [actions[k].to(device).float() for k in ABILITY_KEYS], dim=-1
    )


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def reward_mtp_loss(reward_logits, rewards, bucket_centers, mtp_length):
    """Twohot MTP reward loss over offsets n = 0..L-1 (Eq 9).

    reward_logits: (B, T, L, num_buckets); rewards: (B, T) raw scale.
    Reward is a TARGET (never a model input) so n=0 is legitimate here.
    """
    B, T = rewards.shape
    targets = symlog(rewards)  # (B, T)
    loss = torch.zeros((), device=rewards.device)
    n_terms = 0
    for n in range(mtp_length):
        if T - n <= 0:
            break
        pred = reward_logits[:, :T - n, n, :]      # predict reward at t+n from token t
        tgt = targets[:, n:]                        # (B, T-n)
        loss = loss + twohot_loss(pred, tgt, bucket_centers)
        n_terms += 1
    return loss / max(n_terms, 1)


def bc_next_action_loss(policy_head, agent_out, ability_targets, movement_targets,
                        mtp_length):
    """Behavior-cloning negative log-likelihood of the NEXT actions.

    MTP head n (n >= 1) predicts the action at t+n from the token at t; n=0 is
    dropped to avoid the action-conditioning label leak. Returns the mean NLL of
    the factorized policy (abilities + binned movement) and a (split) breakdown.

    ability_targets: (B, T, num_abilities) {0,1}; movement_targets: (B, T, 2) xy.
    """
    ability_logits, movement_logits = policy_head(agent_out)
    # ability_logits:  (B, T, L, A)
    # movement_logits: (B, T, L, move_dim, bins)
    B, T = ability_targets.shape[0], ability_targets.shape[1]

    import torch.nn.functional as F
    move_idx_full = policy_head.discretize_movement(movement_targets)  # (B, T, 2)

    ability_nll = torch.zeros((), device=agent_out.device)
    move_nll = torch.zeros((), device=agent_out.device)
    n_terms = 0
    for n in range(1, mtp_length):  # n >= 1: predict the NEXT actions only
        if T - n <= 0:
            break
        # token positions 0..T-1-n predict action at +n
        a_logits = ability_logits[:, :T - n, n, :]          # (B, T-n, A)
        a_tgt = ability_targets[:, n:, :]                    # (B, T-n, A)
        ability_nll = ability_nll + F.binary_cross_entropy_with_logits(a_logits, a_tgt)

        m_logits = movement_logits[:, :T - n, n, :, :]       # (B, T-n, move_dim, bins)
        m_idx = move_idx_full[:, n:, :]                      # (B, T-n, move_dim)
        # cross-entropy per axis (flatten axes into the batch dim of CE)
        move_nll = move_nll + F.cross_entropy(
            m_logits.reshape(-1, m_logits.shape[-1]),
            m_idx.reshape(-1),
        )
        n_terms += 1

    n_terms = max(n_terms, 1)
    ability_nll = ability_nll / n_terms
    move_nll = move_nll / n_terms
    return ability_nll + move_nll, {"bc_ability": ability_nll, "bc_movement": move_nll}


# ---------------------------------------------------------------------------
# One training step
# ---------------------------------------------------------------------------

def run_step(batch, dynamics, reward_head, policy_head, schedule, args, device,
             amp_dtype, rms):
    """Forward + loss for one batch. Returns (total_loss, info_dict).

    The frozen dynamics runs under no_grad (it's the backbone); gradients flow
    only into the two heads via ``agent_out``.
    """
    z0 = batch["latents"].to(device)                  # (B, T, C, H, W)
    rewards = batch["rewards"].to(device)             # (B, T)
    actions = actions_to_device(batch["actions"], device)
    ability_targets = stack_ability_targets(batch["actions"], device)  # (B,T,A)
    movement_targets = actions["movement"]            # (B, T, 2)
    B, T = rewards.shape

    actions_dict = actions if dynamics.use_actions else None

    # Near-clean context corruption so the frozen denoiser sees in-distribution
    # inputs (it was trained on noised latents): per-frame tau ~ U(tau_ctx, 1).
    tau = args.tau_ctx + torch.rand(B, T, device=device) * (1.0 - args.tau_ctx)
    with torch.no_grad():
        z_noisy, _ = schedule.add_noise(z0, tau)
        d_one = torch.ones(B, dtype=torch.long, device=device)
        _, agent_out = dynamics(z_noisy, tau, step_size=d_one, actions=actions_dict)
    agent_out = agent_out.detach()  # backbone frozen; heads own all gradients

    with autocast(device_type=device.split(":")[0], dtype=amp_dtype,
                  enabled=(amp_dtype == torch.bfloat16 or amp_dtype == torch.float16) and device != "cpu"):
        reward_logits = reward_head(agent_out)
        reward_loss = reward_mtp_loss(
            reward_logits, rewards, reward_head.bucket_centers, args.mtp_length
        )
        bc_loss, bc_info = bc_next_action_loss(
            policy_head, agent_out, ability_targets, movement_targets, args.mtp_length
        )

        bc_n = rms["bc"].update(bc_loss)
        rew_n = rms["reward"].update(reward_loss)
        total = bc_n + rew_n

    info = {
        "loss": total.detach(),
        "bc_loss": bc_loss.detach(),
        "reward_loss": reward_loss.detach(),
        "bc_ability": bc_info["bc_ability"].detach(),
        "bc_movement": bc_info["bc_movement"].detach(),
    }
    return total, info


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def smoke_test(args):
    """Tiny synthetic CPU run: forward+backward+optimizer step end-to-end, and
    PROVE the movement_heads receive a real BC gradient (the headline fix)."""
    print("=" * 60)
    print("PHASE 2 BC SMOKE TEST (synthetic, CPU)")
    print("=" * 60)
    torch.manual_seed(0)
    device = "cpu"
    args.model_size = "tiny"
    args.latent_dim = 16
    args.mtp_length = 4
    args.num_buckets = 41
    args.movement_bins = 11
    args.hidden_dim = 32
    args.num_register_tokens = 2
    args.tau_ctx = 0.9

    B, T, C, S = 2, 6, args.latent_dim, 16
    dynamics = build_dynamics(args, use_actions=True, device=device)
    dynamics.eval()
    dynamics.requires_grad_(False)
    model_dim = dynamics.model_dim

    reward_head = RewardHead(input_dim=model_dim, hidden_dim=args.hidden_dim,
                             num_buckets=args.num_buckets, mtp_length=args.mtp_length).to(device)
    policy_head = PolicyHead(input_dim=model_dim, num_abilities=len(ABILITY_KEYS),
                             hidden_dim=args.hidden_dim, mtp_length=args.mtp_length,
                             movement_dim=MOVEMENT_DIM, movement_bins=args.movement_bins).to(device)

    # Synthetic batch matching the dataset contract.
    batch = {
        "latents": torch.randn(B, T, C, S, S),
        "rewards": torch.randn(B, T) * 0.01,
        "actions": {
            "movement": torch.rand(B, T, MOVEMENT_DIM),
            **{k: torch.randint(0, 2, (B, T)) for k in ABILITY_KEYS},
        },
    }

    schedule = DiffusionSchedule(device=device)
    rms = {"bc": RunningRMS(), "reward": RunningRMS()}
    params = list(reward_head.parameters()) + list(policy_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-3)

    optimizer.zero_grad()
    total, info = run_step(batch, dynamics, reward_head, policy_head, schedule,
                           args, device, torch.float32, rms)
    total.backward()

    # --- GRAD-FLOW PROOF: movement_heads must receive a nonzero BC gradient ---
    move_grads = [h.weight.grad for h in policy_head.movement_heads if h.weight.grad is not None]
    assert move_grads, "movement_heads received NO gradient (grad is None) under BC!"
    move_grad_norm = sum(g.norm().item() for g in move_grads)
    assert move_grad_norm > 0, f"movement_heads gradient is exactly zero ({move_grad_norm})!"

    ability_grads = [h.weight.grad for h in policy_head.heads if h.weight.grad is not None]
    ability_grad_norm = sum(g.norm().item() for g in ability_grads)
    reward_grad_norm = sum(
        p.grad.norm().item() for p in reward_head.parameters() if p.grad is not None
    )
    # Dynamics is frozen: it must have NO grads.
    dyn_with_grad = [n for n, p in dynamics.named_parameters() if p.grad is not None]
    assert not dyn_with_grad, f"frozen dynamics got gradients: {dyn_with_grad[:3]}"

    torch.nn.utils.clip_grad_norm_(params, 1.0)
    optimizer.step()

    print(f"  total_loss          = {info['loss'].item():.4f}")
    print(f"  bc_loss             = {info['bc_loss'].item():.4f} "
          f"(ability={info['bc_ability'].item():.4f}, movement={info['bc_movement'].item():.4f})")
    print(f"  reward_loss         = {info['reward_loss'].item():.4f}")
    print(f"  GRAD movement_heads = {move_grad_norm:.6e}  (PROOF: > 0 under BC)")
    print(f"  GRAD ability heads  = {ability_grad_norm:.6e}")
    print(f"  GRAD reward head    = {reward_grad_norm:.6e}")
    print(f"  frozen dynamics grads: {len(dyn_with_grad)} (must be 0)")
    print("  optimizer.step() OK")
    print("SMOKE TEST PASSED")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    if args.smoke_test:
        smoke_test(args)
        return

    print("=" * 60)
    print("Phase 2: Agent Finetuning (BC + Reward)")
    print("=" * 60)
    device = args.device
    print(f"Device: {device} | model_size={args.model_size} | latent_dim={args.latent_dim}")
    print(f"seq_len={args.seq_len} mtp={args.mtp_length} movement_bins={args.movement_bins}")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading frozen dynamics backbone...")
    dynamics, use_actions = load_frozen_dynamics(args, device)
    model_dim = dynamics.model_dim
    print(f"  dynamics params: {sum(p.numel() for p in dynamics.parameters()):,} (frozen)")
    print(f"  action conditioning: {use_actions}")

    reward_head = RewardHead(
        input_dim=model_dim, hidden_dim=args.hidden_dim,
        num_buckets=args.num_buckets, mtp_length=args.mtp_length,
    ).to(device)
    policy_head = PolicyHead(
        input_dim=model_dim, num_abilities=len(ABILITY_KEYS),
        hidden_dim=args.hidden_dim, mtp_length=args.mtp_length,
        movement_dim=MOVEMENT_DIM, movement_bins=args.movement_bins,
    ).to(device)
    print(f"  reward head: {sum(p.numel() for p in reward_head.parameters()):,}")
    print(f"  policy head: {sum(p.numel() for p in policy_head.parameters()):,}")

    print(f"\nLoading replay data from {args.latents_dir}...")
    dataset = build_dataset(args)
    if len(dataset) == 0:
        raise SystemExit("No sequences found. Check --latents-dir / --labels-root / --seq-len.")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=VideoGroupedSampler(dataset),
        num_workers=args.num_workers, pin_memory=(device != "cpu"), drop_last=True,
    )

    schedule = DiffusionSchedule(device=device)
    rms = {"bc": RunningRMS(), "reward": RunningRMS()}
    params = list(reward_head.parameters()) + list(policy_head.parameters())
    optimizer = create_optimizer(params, args.lr, args.weight_decay,
                                 use_8bit=args.use_8bit_adam, betas=tuple(args.adam_betas))
    total_steps = args.epochs * max(1, len(dataloader))
    scheduler = create_wsd_schedule(optimizer, total_steps, args.warmup_steps, args.decay_steps)
    amp_dtype = torch.bfloat16 if device != "mps" else torch.float16
    scaler = GradScaler(device.split(":")[0], enabled=(amp_dtype == torch.float16))

    init_wandb(args, job_type="agent_finetune", extra_config={
        "reward_head_params": sum(p.numel() for p in reward_head.parameters()),
        "policy_head_params": sum(p.numel() for p in policy_head.parameters()),
    })

    print("\n" + "=" * 60)
    print("Starting BC + reward training...")
    print("=" * 60)
    global_step = 0
    for epoch in range(args.epochs):
        reward_head.train()
        policy_head.train()
        t0 = time.time()
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            total, info = run_step(batch, dynamics, reward_head, policy_head,
                                   schedule, args, device, amp_dtype, rms)
            if not torch.isfinite(total):
                print(f"[WARN] non-finite loss at step {global_step}; skipping.")
                continue
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            if batch_idx % args.log_interval == 0:
                sps = (batch_idx + 1) * args.batch_size / max(time.time() - t0, 1e-6)
                log_step({
                    "train/loss": info["loss"].item(),
                    "train/bc_loss": info["bc_loss"].item(),
                    "train/bc_ability": info["bc_ability"].item(),
                    "train/bc_movement": info["bc_movement"].item(),
                    "train/reward_loss": info["reward_loss"].item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/epoch": epoch,
                }, step=global_step)
                print(f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                      f"loss={info['loss'].item():.4f} "
                      f"bc={info['bc_loss'].item():.4f} "
                      f"(abil={info['bc_ability'].item():.3f} move={info['bc_movement'].item():.3f}) "
                      f"rew={info['reward_loss'].item():.4f} ({sps:.1f} samp/s)")

        ckpt_path = checkpoint_dir / f"agent_finetune_epoch_{epoch + 1:03d}.pt"
        save_phase2_checkpoint(ckpt_path, dynamics, reward_head, policy_head,
                               optimizer, scheduler, rms, epoch + 1, global_step, args)
        save_phase2_checkpoint(checkpoint_dir / "agent_finetune_latest.pt", dynamics,
                               reward_head, policy_head, optimizer, scheduler, rms,
                               epoch + 1, global_step, args)

    print("\nPhase 2 training complete.")
    finish_wandb()


def save_phase2_checkpoint(path, dynamics, reward_head, policy_head, optimizer,
                           scheduler, rms, epoch, global_step, args):
    inner = getattr(dynamics, "_orig_mod", dynamics)
    ckpt = {
        "dynamics_state_dict": inner.state_dict(),
        "dynamics_config": getattr(inner, "config", None),
        "reward_head_state_dict": reward_head.state_dict(),
        "policy_head_state_dict": policy_head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "rms_state": {k: v.state_dict() for k, v in rms.items()},
        "epoch": epoch,
        "global_step": global_step,
        "args": vars(args),
        "phase": "agent_finetune",
    }
    tmp = Path(str(path) + ".tmp")
    torch.save(ckpt, tmp)
    import os
    os.replace(tmp, path)
    print(f"Saved checkpoint to {path}")


if __name__ == "__main__":
    main()
