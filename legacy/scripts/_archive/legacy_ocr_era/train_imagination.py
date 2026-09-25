#!/usr/bin/env python3
"""Phase 3: Imagination Training - RL inside the world model.

Implements DreamerV4 Section 3.3 "Reinforcement learning":
- Freeze dynamics transformer (optionally finetune with extra losses)
- Initialize value head + frozen behavioral prior (copy of BC policy)
- Generate imagined rollouts: sample z from dynamics, actions from policy
- Train value head with TD(λ) on imagined trajectories (Eq 10)
- Train policy head with PMPO objective (Eq 11)

Key design decisions (from paper):
- One rollout per context (diversity > length)
- PMPO uses sign-only advantages (no normalization needed)
- Reverse KL to prior: KL[π_θ || π_prior] constrains to reasonable behaviors
- α=0.5 (balance D+/D- equally), β=0.3 (weak prior)
- γ=0.997, λ=0.95

Usage:
    python scripts/train_imagination.py \
        --agent-checkpoint checkpoints/agent_finetune_latest.pt \
        --tokenizer-checkpoint checkpoints/tokenizer_best.pt \
        --latents-dir data/latents \
        --epochs 10

Reference: DreamerV4 Section 3.3 "Reinforcement learning"
"""

import argparse
import copy
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from ahriuwu.models import (
    create_transformer_tokenizer,
    create_dynamics,
    RewardHead,
    PolicyHead,
    ValueHead,
    DiffusionSchedule,
    ShortcutForcing,
    symlog,
    twohot_loss,
    compute_lambda_returns,
    compute_pmpo_loss,
    RunningRMS,
)
from ahriuwu.data.dataset import PackedLatentSequenceDataset, RewardMixtureSampler
from ahriuwu.utils.logging import add_wandb_args, init_wandb, log_step, finish_wandb
from ahriuwu.utils.training import add_training_args, create_optimizer, create_wsd_schedule


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 3: Imagination Training")
    add_training_args(parser)
    parser.add_argument(
        "--agent-checkpoint", type=str, required=True,
        help="Path to Phase 2 agent finetuning checkpoint",
    )
    parser.add_argument(
        "--tokenizer-checkpoint", type=str, required=True,
        help="Path to pretrained tokenizer checkpoint",
    )
    parser.add_argument(
        "--latents-dir", type=str, default=None,
        help="Directory containing pre-tokenized latents for context frames",
    )
    parser.add_argument(
        "--features-dir", type=str, default=None,
        help="Directory containing features.json per video",
    )
    parser.add_argument(
        "--model-size", type=str, default="small",
        choices=["tiny", "small", "medium", "large"],
    )
    parser.add_argument("--seq-len", type=int, default=32, help="Context sequence length")
    parser.add_argument("--horizon", type=int, default=16, help="Imagination rollout horizon")
    parser.add_argument("--action-dim", type=int, default=128)
    parser.add_argument("--mtp-length", type=int, default=8)
    parser.add_argument("--num-buckets", type=int, default=255)
    # RL hyperparameters (paper defaults)
    parser.add_argument("--gamma", type=float, default=0.997, help="Discount factor")
    parser.add_argument("--lambda_", type=float, default=0.95, help="TD(λ) parameter")
    parser.add_argument("--pmpo-alpha", type=float, default=0.5, help="PMPO D+/D- balance")
    parser.add_argument("--pmpo-beta", type=float, default=0.3, help="PMPO prior KL weight")
    parser.add_argument("--temperature", type=float, default=1.0, help="Policy sampling temperature")
    # Model config (should match Phase 2)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--no-qk-norm", action="store_true")
    parser.add_argument("--soft-cap", type=float, default=50.0)
    parser.add_argument("--num-register-tokens", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=None)
    parser.add_argument("--shortcut-k-max", type=int, default=64)
    parser.add_argument("--shortcut-k-steps", type=int, default=4,
                        help="Number of denoising steps K for generation (paper uses 4)")
    parser.set_defaults(num_workers=0)
    add_wandb_args(parser)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_phase2_checkpoint(checkpoint_path: str, model_size: str, device: str, args):
    """Load Phase 2 checkpoint: dynamics, reward head, policy head."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_args = checkpoint.get("args", {})

    latent_dim = saved_args.get("latent_dim", 32)

    # Detect action conditioning from dynamics weights
    dyn_state = checkpoint["dynamics_state_dict"]
    use_actions = any("action_embed" in k for k in dyn_state.keys())

    # Reconstruct dynamics
    dynamics = create_dynamics(
        size=model_size,
        latent_dim=latent_dim,
        use_agent_tokens=True,
        use_actions=use_actions,
        num_tasks=1,
        agent_layers=4,
        use_qk_norm=not args.no_qk_norm,
        soft_cap=args.soft_cap if args.soft_cap > 0 else None,
        num_register_tokens=args.num_register_tokens,
        num_kv_heads=args.num_kv_heads,
    )
    dynamics.load_state_dict(dyn_state)
    dynamics = dynamics.to(device)
    dynamics.eval()
    for p in dynamics.parameters():
        p.requires_grad = False

    model_dim = dynamics.model_dim

    # Reconstruct reward head
    mtp_length = saved_args.get("mtp_length", args.mtp_length)
    num_buckets = saved_args.get("num_buckets", args.num_buckets)
    reward_head = RewardHead(
        input_dim=model_dim, hidden_dim=256,
        num_buckets=num_buckets, mtp_length=mtp_length,
    )
    reward_head.load_state_dict(checkpoint["reward_head_state_dict"])
    reward_head = reward_head.to(device)
    reward_head.eval()
    for p in reward_head.parameters():
        p.requires_grad = False

    # Reconstruct policy head
    action_dim = saved_args.get("action_dim", args.action_dim)
    policy_head = PolicyHead(
        input_dim=model_dim, action_dim=action_dim,
        hidden_dim=256, mtp_length=mtp_length,
    )
    policy_head.load_state_dict(checkpoint["policy_head_state_dict"])
    policy_head = policy_head.to(device)

    print(f"Loaded Phase 2 checkpoint from {checkpoint_path}")
    print(f"  Dynamics: {sum(p.numel() for p in dynamics.parameters()):,} params (frozen)")
    print(f"  Reward head: {sum(p.numel() for p in reward_head.parameters()):,} params (frozen)")
    print(f"  Policy head: {sum(p.numel() for p in policy_head.parameters()):,} params (trainable)")
    if use_actions:
        print("  Action conditioning: detected")

    return dynamics, reward_head, policy_head, model_dim, use_actions


def load_tokenizer(checkpoint_path: str, device: str):
    """Load pretrained transformer tokenizer (frozen)."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    tok_args = checkpoint.get("args", {})

    from ahriuwu.models import create_transformer_tokenizer
    model_size = tok_args.get("model_size", "small")
    use_rope = tok_args.get("use_rope", True)
    tokenizer = create_transformer_tokenizer(size=model_size, use_rope=use_rope)
    tokenizer.load_state_dict(state_dict, strict=False)
    tokenizer = tokenizer.to(device)
    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad = False
    return tokenizer


# ---------------------------------------------------------------------------
# Imagination rollout
# ---------------------------------------------------------------------------

@torch.no_grad()
def imagine_rollout(
    dynamics: nn.Module,
    policy_head: PolicyHead,
    reward_head: RewardHead,
    value_head: ValueHead,
    z_context: torch.Tensor,
    horizon: int,
    schedule: DiffusionSchedule,
    shortcut_k_steps: int = 4,
    shortcut_k_max: int = 64,
    temperature: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Generate one imagined rollout per context.

    Unrolls the dynamics model autoregressively, sampling actions from
    the policy head and annotating with rewards and values.

    Paper: "we start only one rollout from each context, prioritizing
    data diversity and reducing memory consumption."

    Args:
        dynamics: Frozen dynamics transformer with agent tokens
        policy_head: Current policy (will be differentiated through later)
        reward_head: Frozen reward predictor
        value_head: Current value predictor
        z_context: (B, T_ctx, C, H, W) context latent frames
        horizon: Number of steps to imagine
        schedule: DiffusionSchedule for generation
        shortcut_k_steps: K denoising steps per generated frame
        shortcut_k_max: Maximum shortcut step size
        temperature: Policy sampling temperature

    Returns:
        Dict with:
            agent_outs: (B, H, D) agent token outputs per imagined step
            actions: (B, H) sampled discrete action indices
            rewards: (B, H) predicted rewards (original scale)
            values: (B, H) predicted values (original scale)
    """
    B = z_context.shape[0]
    device = z_context.device

    # We'll collect agent outputs for each imagined step
    all_agent_outs = []
    all_actions = []
    all_rewards = []
    all_values = []

    # Start from the last context frame
    # For autoregressive generation, we maintain a rolling window
    z_window = z_context  # (B, T_ctx, C, H, W)

    for t in range(horizon):
        # Run dynamics on current window to get agent output for last frame
        # Use τ_ctx = 0.1 for slightly noisy context (paper inference setting)
        T_win = z_window.shape[1]
        tau = torch.full((B, T_win), 0.1, device=device)

        # Slightly corrupt context
        z_noisy, _ = schedule.add_noise(z_window, tau)

        # Step size for shortcut: use d=1/K (finest step for context processing)
        d_norm = torch.full((B,), 1.0 / shortcut_k_max, device=device)

        # Forward pass — get agent tokens for the last frame
        z_pred, agent_out = dynamics(z_noisy, tau, step_size=d_norm)
        # agent_out: (B, T_win, D) — take last timestep
        h_t = agent_out[:, -1, :]  # (B, D)

        all_agent_outs.append(h_t)

        # Sample action from policy (use MTP offset n=0 = current timestep)
        ability_logits, movement_pred = policy_head(h_t.unsqueeze(1))
        # ability_logits: (B, 1, L, action_dim)
        # Take n=0 offset
        logits_t = ability_logits[:, 0, 0, :]  # (B, action_dim)
        if temperature == 0:
            action_t = logits_t.argmax(dim=-1)
        else:
            probs = F.softmax(logits_t / temperature, dim=-1)
            action_t = torch.multinomial(probs, num_samples=1).squeeze(-1)
        all_actions.append(action_t)

        # Predict reward (MTP offset n=0)
        reward_t = reward_head.predict(h_t.unsqueeze(1))[:, 0, 0]  # (B,)
        all_rewards.append(reward_t)

        # Predict value
        value_t = value_head.predict(h_t.unsqueeze(1))[:, 0]  # (B,)
        all_values.append(value_t)

        # Generate next frame using shortcut model (K denoising steps)
        # Start from pure noise
        C, H, W = z_window.shape[2:]
        z_noise = torch.randn(B, 1, C, H, W, device=device)

        # Simple iterative denoising with K steps
        z_t = z_noise
        step_size = 1.0 / shortcut_k_steps
        for k in range(shortcut_k_steps):
            tau_k = torch.full((B, 1), 1.0 - k * step_size, device=device)
            d_k = torch.full((B,), step_size, device=device)

            # Concatenate context + current noisy frame
            z_input = torch.cat([z_window, z_t], dim=1)
            tau_input = torch.cat([
                torch.full((B, T_win), 0.1, device=device),  # context slightly noisy
                tau_k,
            ], dim=1)

            z_full_pred, _ = dynamics(z_input, tau_input, step_size=d_k)
            # Take prediction for the last (new) frame
            z_t = z_full_pred[:, -1:, :]

        # Advance window: drop oldest frame, append generated frame
        z_window = torch.cat([z_window[:, 1:, :], z_t], dim=1)

    return {
        "agent_outs": torch.stack(all_agent_outs, dim=1),  # (B, H, D)
        "actions": torch.stack(all_actions, dim=1),         # (B, H)
        "rewards": torch.stack(all_rewards, dim=1),         # (B, H)
        "values": torch.stack(all_values, dim=1),           # (B, H)
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(
    dynamics: nn.Module,
    reward_head: nn.Module,
    policy_head: nn.Module,
    policy_prior: nn.Module,
    value_head: nn.Module,
    schedule: DiffusionSchedule,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    scheduler,
    rms_trackers: dict,
    device: str,
    epoch: int,
    args: argparse.Namespace,
):
    """Train one epoch of Phase 3 imagination training."""
    # Dynamics and reward head stay frozen
    dynamics.eval()
    reward_head.eval()
    policy_prior.eval()
    # Policy and value heads are trained
    policy_head.train()
    value_head.train()

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_loss = 0.0
    total_mean_reward = 0.0
    total_mean_value = 0.0
    total_pos_frac = 0.0
    num_batches = 0
    start_time = time.time()

    device_type = device.split(":")[0]
    amp_dtype = torch.bfloat16 if device_type == "cuda" else torch.float16

    for batch_idx, batch in enumerate(dataloader):
        z_context = batch["latents"].to(device)  # (B, T_ctx, C, H, W)
        B = z_context.shape[0]

        # --- Phase 1: Imagined rollout (no gradients for dynamics/reward) ---
        rollout = imagine_rollout(
            dynamics=dynamics,
            policy_head=policy_head,
            reward_head=reward_head,
            value_head=value_head,
            z_context=z_context,
            horizon=args.horizon,
            schedule=schedule,
            shortcut_k_steps=args.shortcut_k_steps,
            shortcut_k_max=args.shortcut_k_max,
            temperature=args.temperature,
        )

        agent_outs = rollout["agent_outs"]  # (B, H, D)
        actions = rollout["actions"]         # (B, H)
        rewards = rollout["rewards"]         # (B, H) original scale
        values = rollout["values"].detach()  # (B, H) original scale, detached

        # --- Phase 2: Compute returns and advantages ---
        with autocast(device_type=device_type, dtype=amp_dtype):
            # No terminal states in imagination
            continues = torch.ones_like(rewards)

            # λ-returns (Eq 10)
            returns = compute_lambda_returns(
                rewards, values, continues,
                gamma=args.gamma, lambda_=args.lambda_,
            )  # (B, H)

            # Raw advantages (NOT normalized — PMPO uses sign only)
            advantages = returns - values  # (B, H)

            # --- Value loss: TD(λ) with twohot (Eq 10) ---
            value_logits = value_head(agent_outs)  # (B, H, num_buckets)
            value_targets = symlog(returns.detach())  # (B, H) in symlog space
            value_loss = twohot_loss(value_logits, value_targets, value_head.bucket_centers)

            # --- Policy loss: PMPO (Eq 11) ---
            # Get log-probs from current policy and frozen prior
            # We need to re-forward through policy head WITH gradients
            log_probs = policy_head.log_prob(
                agent_outs, actions.unsqueeze(-1).expand(-1, -1, args.mtp_length)
            )[:, :, 0]  # (B, H) using MTP offset 0

            with torch.no_grad():
                log_probs_prior = policy_prior.log_prob(
                    agent_outs, actions.unsqueeze(-1).expand(-1, -1, args.mtp_length)
                )[:, :, 0]  # (B, H)

            # Flatten for PMPO
            policy_loss = compute_pmpo_loss(
                log_probs=log_probs.reshape(-1),
                advantages=advantages.reshape(-1),
                log_probs_prior=log_probs_prior.reshape(-1),
                alpha=args.pmpo_alpha,
                beta=args.pmpo_beta,
            )

            # --- Total loss ---
            policy_loss_norm = rms_trackers["policy"].update(policy_loss)
            value_loss_norm = rms_trackers["value"].update(value_loss)
            total_loss_batch = policy_loss_norm + value_loss_norm

        # --- Backward (only policy + value heads) ---
        scaler.scale(total_loss_batch).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(policy_head.parameters()) + list(value_head.parameters()),
            max_norm=1.0,
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()

        # --- Track metrics ---
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_loss += total_loss_batch.item()
        total_mean_reward += rewards.mean().item()
        total_mean_value += values.mean().item()
        pos_frac = (advantages >= 0).float().mean().item()
        total_pos_frac += pos_frac
        num_batches += 1

        if batch_idx % args.log_interval == 0:
            elapsed = time.time() - start_time
            samples_per_sec = (batch_idx + 1) * args.batch_size / elapsed

            log_step({
                "train/loss": total_loss_batch.item(),
                "train/policy_loss": policy_loss.item(),
                "train/value_loss": value_loss.item(),
                "train/mean_reward": rewards.mean().item(),
                "train/mean_value": values.mean().item(),
                "train/mean_return": returns.mean().item(),
                "train/pos_advantage_frac": pos_frac,
                "train/lr": scheduler.get_last_lr()[0],
                "train/samples_per_sec": samples_per_sec,
                "train/epoch": epoch,
            }, step=batch_idx + epoch * len(dataloader))

            print(
                f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                f"Loss: {total_loss_batch.item():.4f} "
                f"π: {policy_loss.item():.4f} "
                f"V: {value_loss.item():.4f} "
                f"R̄: {rewards.mean().item():.3f} "
                f"V̄: {values.mean().item():.3f} "
                f"A+: {pos_frac:.1%} "
                f"({samples_per_sec:.1f} s/s)"
            )

    n = max(num_batches, 1)
    return {
        "loss": total_loss / n,
        "policy_loss": total_policy_loss / n,
        "value_loss": total_value_loss / n,
        "mean_reward": total_mean_reward / n,
        "mean_value": total_mean_value / n,
        "pos_advantage_frac": total_pos_frac / n,
    }


# ---------------------------------------------------------------------------
# Dataset (reuses latent dataset for context frames)
# ---------------------------------------------------------------------------

class ContextDataset(torch.utils.data.Dataset):
    """Dataset that provides context frames for imagination rollouts.

    Each item is a sequence of latent frames from the dataset.
    The imagination loop will generate future frames from these contexts.
    """

    def __init__(self, latents_dir: str, seq_len: int = 32, features_dir: str | None = None):
        self.latent_dataset = PackedLatentSequenceDataset(
            latents_dir=latents_dir,
            sequence_length=seq_len,
            stride=seq_len // 2,
            load_actions=False,
            features_dir=features_dir,
        )
        print(f"Loaded {len(self.latent_dataset)} context sequences")

    def __len__(self):
        return len(self.latent_dataset)

    def __getitem__(self, idx):
        data = self.latent_dataset[idx]
        return {"latents": data["latents"]}


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    dynamics_state: dict,
    reward_head_state: dict,
    policy_head: nn.Module,
    value_head: nn.Module,
    policy_prior: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    scheduler,
    rms_trackers: dict,
    epoch: int,
    metrics: dict,
    args: argparse.Namespace,
    path: Path,
):
    """Save Phase 3 checkpoint."""
    checkpoint = {
        "dynamics_state_dict": dynamics_state,
        "reward_head_state_dict": reward_head_state,
        "policy_head_state_dict": policy_head.state_dict(),
        "value_head_state_dict": value_head.state_dict(),
        "policy_prior_state_dict": policy_prior.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "rms_state": {k: v.state_dict() for k, v in rms_trackers.items()},
        "epoch": epoch,
        "metrics": metrics,
        "args": vars(args),
        "phase": "imagination",
    }
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print("=" * 60)
    print("Phase 3: Imagination Training (PMPO + Value)")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Horizon: {args.horizon}")
    print(f"γ={args.gamma}, λ={args.lambda_}")
    print(f"PMPO: α={args.pmpo_alpha}, β={args.pmpo_beta}")
    print(f"Temperature: {args.temperature}")
    print(f"Shortcut K={args.shortcut_k_steps}")
    print("=" * 60)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load Phase 2 models
    print("\nLoading Phase 2 checkpoint...")
    dynamics, reward_head, policy_head, model_dim, use_actions = load_phase2_checkpoint(
        args.agent_checkpoint, args.model_size, args.device, args,
    )

    # Create frozen behavioral prior (deep copy of BC-trained policy)
    print("Creating frozen behavioral prior (copy of BC policy)...")
    policy_prior = copy.deepcopy(policy_head)
    policy_prior.eval()
    for p in policy_prior.parameters():
        p.requires_grad = False

    # Create value head (initialized fresh for Phase 3)
    print("Creating value head...")
    value_head = ValueHead(
        input_dim=model_dim,
        hidden_dim=256,
        num_buckets=args.num_buckets,
    ).to(args.device)
    print(f"  Value head: {sum(p.numel() for p in value_head.parameters()):,} params")

    # Diffusion schedule
    schedule = DiffusionSchedule(device=args.device)

    # Dataset
    print(f"\nLoading context sequences from {args.latents_dir}...")
    dataset = ContextDataset(
        latents_dir=args.latents_dir,
        seq_len=args.seq_len,
        features_dir=args.features_dir,
    )

    sampler = RewardMixtureSampler(dataset.latent_dataset) if hasattr(dataset.latent_dataset, 'sequences') else None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Optimizer: only policy + value heads
    trainable_params = list(policy_head.parameters()) + list(value_head.parameters())
    optimizer = create_optimizer(trainable_params, args.lr, args.weight_decay, use_8bit=args.use_8bit_adam)

    total_steps = args.epochs * len(dataloader)
    scheduler = create_wsd_schedule(optimizer, total_steps, args.warmup_steps, args.decay_steps)

    device_type = args.device.split(":")[0]
    scaler = GradScaler(device_type)

    rms_trackers = {
        "policy": RunningRMS(),
        "value": RunningRMS(),
    }

    wandb_run = init_wandb(args, job_type="imagination", extra_config={
        "horizon": args.horizon,
        "gamma": args.gamma,
        "lambda": args.lambda_,
        "pmpo_alpha": args.pmpo_alpha,
        "pmpo_beta": args.pmpo_beta,
    })

    # Save frozen model states for checkpointing (don't re-save every epoch)
    dynamics_state = {k: v.cpu() for k, v in dynamics.state_dict().items()}
    reward_head_state = {k: v.cpu() for k, v in reward_head.state_dict().items()}

    # Training loop
    print("\n" + "=" * 60)
    print("Starting imagination training...")
    print("=" * 60)

    history = []
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)

        metrics = train_epoch(
            dynamics=dynamics,
            reward_head=reward_head,
            policy_head=policy_head,
            policy_prior=policy_prior,
            value_head=value_head,
            schedule=schedule,
            dataloader=dataloader,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            rms_trackers=rms_trackers,
            device=args.device,
            epoch=epoch,
            args=args,
        )

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Total Loss: {metrics['loss']:.4f}")
        print(f"  Policy Loss: {metrics['policy_loss']:.4f}")
        print(f"  Value Loss: {metrics['value_loss']:.4f}")
        print(f"  Mean Reward: {metrics['mean_reward']:.4f}")
        print(f"  Mean Value: {metrics['mean_value']:.4f}")
        print(f"  Positive Advantage %: {metrics['pos_advantage_frac']:.1%}")

        log_step({
            "epoch/loss": metrics["loss"],
            "epoch/policy_loss": metrics["policy_loss"],
            "epoch/value_loss": metrics["value_loss"],
            "epoch/mean_reward": metrics["mean_reward"],
            "epoch/mean_value": metrics["mean_value"],
            "epoch/pos_advantage_frac": metrics["pos_advantage_frac"],
            "epoch/epoch": epoch + 1,
        }, step=(epoch + 1) * len(dataloader))

        history.append({"epoch": epoch + 1, **metrics})

        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(
                dynamics_state, reward_head_state,
                policy_head, value_head, policy_prior,
                optimizer, scaler, scheduler, rms_trackers,
                epoch, metrics, args,
                checkpoint_dir / f"imagination_epoch_{epoch + 1:03d}.pt",
            )
            save_checkpoint(
                dynamics_state, reward_head_state,
                policy_head, value_head, policy_prior,
                optimizer, scaler, scheduler, rms_trackers,
                epoch, metrics, args,
                checkpoint_dir / "imagination_latest.pt",
            )

    history_path = checkpoint_dir / "imagination_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print("Imagination training complete!")
    print(f"Checkpoints saved to {checkpoint_dir}")
    print("=" * 60)

    finish_wandb()


if __name__ == "__main__":
    main()
