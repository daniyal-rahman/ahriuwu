#!/usr/bin/env python3
"""Phase 2: Agent Finetuning - BC + Reward Modeling.

This implements DreamerV4 Section 3.3 "Behavior cloning and reward model":
- Continue training dynamics model
- Add agent tokens to dynamics
- Train reward head (symexp twohot)
- Train policy head (behavioral cloning with MTP)

Loss = dynamics_loss + reward_loss + bc_loss (normalized by RMS)

Usage:
    python scripts/train_agent_finetune.py \
        --dynamics-checkpoint checkpoints/dynamics_best.pt \
        --tokenizer-checkpoint checkpoints/tokenizer_best.pt \
        --data-dir data/replays \
        --epochs 10

Reference: DreamerV4 Section 3.3 "Behavior cloning and reward model"
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from ahriuwu.models import (
    create_tokenizer,
    create_dynamics,
    RewardHead,
    PolicyHead,
    ShortcutForcing,
    symlog,
    twohot_loss,
    RunningRMS,
)
from ahriuwu.data.dataset import LatentSequenceDataset, RewardMixtureSampler
from ahriuwu.data.actions import encode_action


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2: Agent Finetuning")
    parser.add_argument(
        "--dynamics-checkpoint",
        type=str,
        required=True,
        help="Path to pretrained dynamics checkpoint",
    )
    parser.add_argument(
        "--tokenizer-checkpoint",
        type=str,
        required=True,
        help="Path to pretrained tokenizer checkpoint",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/replays",
        help="Directory containing replay data with actions and rewards (for raw frames)",
    )
    parser.add_argument(
        "--latents-dir",
        type=str,
        default=None,
        help="Directory containing pre-tokenized latents (if provided, skips tokenization)",
    )
    parser.add_argument(
        "--features-dir",
        type=str,
        default=None,
        help="Directory containing features.json per video (default: data/processed)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        default=128,
        help="Number of discrete actions",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="small",
        choices=["tiny", "small", "medium", "large"],
        help="Dynamics model size",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=32,
        help="Sequence length (frames)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--mtp-length",
        type=int,
        default=8,
        help="Multi-token prediction length (paper uses 8)",
    )
    parser.add_argument(
        "--num-buckets",
        type=int,
        default=255,
        help="Number of twohot buckets for reward prediction",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Log every N batches",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=1,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        help="Device to train on",
    )
    parser.add_argument(
        "--use-reward-mixture",
        action="store_true",
        help="Use 50/50 reward mixture sampling (50% uniform, 50% reward-containing)",
    )
    return parser.parse_args()


class ReplayDataset(torch.utils.data.Dataset):
    """Dataset for replay data with latents, actions, and rewards.

    Wraps LatentSequenceDataset to provide properly formatted data:
    - latents: (T, C, H, W) pre-tokenized latent vectors
    - actions: (T,) discrete action indices
    - rewards: (T,) reward values

    When use_latents=True, uses pre-computed latents from LatentSequenceDataset.
    Otherwise falls back to placeholder for raw frame mode (not yet implemented).
    """

    def __init__(
        self,
        data_dir: str,
        seq_len: int = 32,
        latents_dir: str | None = None,
        features_dir: str | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.use_latents = latents_dir is not None

        if self.use_latents:
            # Use LatentSequenceDataset for pre-tokenized data
            self.latent_dataset = LatentSequenceDataset(
                latents_dir=latents_dir,
                sequence_length=seq_len,
                stride=seq_len // 2,  # 50% overlap for more sequences
                load_actions=True,
                load_rewards=True,
                features_dir=features_dir,
            )
            print(f"Loaded {len(self.latent_dataset)} latent sequences with actions and rewards")
        else:
            # Placeholder for raw frame mode
            self.latent_dataset = None
            replay_files = list(self.data_dir.glob("*.pt")) + list(self.data_dir.glob("*.npz"))
            print(f"Found {len(replay_files)} replay files (raw frame mode - not yet implemented)")
            self.replay_files = replay_files

    def __len__(self):
        if self.use_latents:
            return len(self.latent_dataset)
        return len(self.replay_files) * 10  # Placeholder

    def __getitem__(self, idx):
        if self.use_latents:
            # Get data from LatentSequenceDataset
            data = self.latent_dataset[idx]

            latents = data["latents"]  # (T, C, H, W) = (T, 256, 16, 16)

            # Convert actions dict to single discrete action tensor
            actions_dict = data.get("actions")
            if actions_dict is not None:
                # Encode actions using the action encoding function
                actions = []
                for t in range(self.seq_len):
                    action = encode_action(
                        movement=actions_dict['movement'][t].item(),
                        abilities={
                            'Q': bool(actions_dict['Q'][t].item()),
                            'W': bool(actions_dict['W'][t].item()),
                            'E': bool(actions_dict['E'][t].item()),
                            'R': bool(actions_dict['R'][t].item()),
                            'D': bool(actions_dict['D'][t].item()),
                            'F': bool(actions_dict['F'][t].item()),
                            'item': bool(actions_dict['item'][t].item()),
                            'B': bool(actions_dict['B'][t].item()),
                        }
                    )
                    actions.append(action)
                actions = torch.tensor(actions, dtype=torch.long)
            else:
                # No actions available - use zeros
                actions = torch.zeros(self.seq_len, dtype=torch.long)

            # Get rewards (or zeros if not available)
            rewards = data.get("rewards")
            if rewards is None:
                rewards = torch.zeros(self.seq_len, dtype=torch.float32)

            return {
                "latents": latents,  # (T, C, H, W) pre-tokenized
                "actions": actions,  # (T,)
                "rewards": rewards,  # (T,)
            }
        else:
            # Placeholder for raw frame mode
            T = self.seq_len
            C, H, W = 3, 256, 256
            return {
                "frames": torch.rand(T, C, H, W),
                "actions": torch.randint(0, 128, (T,)),
                "rewards": torch.randn(T) * 10,
            }


def load_pretrained_dynamics(checkpoint_path: str, model_size: str, device: str):
    """Load pretrained dynamics and upgrade to use agent tokens."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Create new dynamics with agent tokens enabled
    dynamics = create_dynamics(
        size=model_size,
        latent_dim=256,
        use_agent_tokens=True,
        num_tasks=1,
        agent_layers=4,
    )

    # Load pretrained weights (excluding new agent token components)
    pretrained_state = checkpoint["model_state_dict"]
    model_state = dynamics.state_dict()

    # Copy matching weights
    for name, param in pretrained_state.items():
        if name in model_state and model_state[name].shape == param.shape:
            model_state[name] = param
            print(f"  Loaded: {name}")
        else:
            print(f"  Skipped: {name} (new or shape mismatch)")

    dynamics.load_state_dict(model_state)
    dynamics = dynamics.to(device)

    return dynamics, checkpoint.get("args", {})


def load_tokenizer(checkpoint_path: str, device: str):
    """Load pretrained tokenizer."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    tokenizer = create_tokenizer()
    tokenizer.load_state_dict(checkpoint["model_state_dict"])
    tokenizer = tokenizer.to(device)
    tokenizer.eval()

    # Freeze tokenizer
    for param in tokenizer.parameters():
        param.requires_grad = False

    return tokenizer


def train_epoch(
    dynamics: nn.Module,
    reward_head: nn.Module,
    policy_head: nn.Module,
    tokenizer: nn.Module,
    shortcut: ShortcutForcing,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    rms_trackers: dict,
    device: str,
    epoch: int,
    args: argparse.Namespace,
):
    """Train for one epoch."""
    dynamics.train()
    reward_head.train()
    policy_head.train()

    total_loss = 0.0
    total_dynamics_loss = 0.0
    total_reward_loss = 0.0
    total_bc_loss = 0.0
    num_batches = 0
    start_time = time.time()

    device_type = device.split(":")[0]
    dtype = torch.bfloat16 if device_type == "cuda" else torch.float16

    for batch_idx, batch in enumerate(dataloader):
        # Handle both latents (pre-tokenized) and frames (need tokenization)
        if "latents" in batch:
            z = batch["latents"].to(device)  # (B, T, C, H, W) already tokenized
        else:
            frames = batch["frames"].to(device)  # (B, T, C, H, W)
            # Encode frames to latents
            with torch.no_grad():
                B_frames, T_frames = frames.shape[:2]
                frames_flat = frames.view(B_frames * T_frames, *frames.shape[2:])
                z = tokenizer.encode(frames_flat)  # (B*T, C, H, W) latent
                z = z.view(B_frames, T_frames, *z.shape[1:])  # (B, T, C, H, W)

        actions = batch["actions"].to(device)  # (B, T)
        rewards = batch["rewards"].to(device)  # (B, T)

        B, T = actions.shape

        with autocast(device_type=device_type, dtype=dtype):

            # Dynamics forward with shortcut forcing
            z_noisy, tau, step_size, z_target = shortcut.add_noise(z)

            # Forward through dynamics with agent tokens
            z_pred, agent_out = dynamics(z_noisy, tau, step_size=step_size)

            # Dynamics loss (x-prediction)
            dynamics_loss = F.mse_loss(z_pred, z_target)

            # Reward prediction loss
            reward_logits = reward_head(agent_out)  # (B, T, L, num_buckets)
            reward_targets = symlog(rewards)  # (B, T)

            # MTP reward loss: predict rewards at t+1, t+2, ..., t+L
            reward_loss = 0.0
            L = args.mtp_length
            for offset in range(L):
                if offset < T - 1:
                    # Target is reward at t + offset + 1
                    target_idx = min(offset + 1, T - 1)
                    target = reward_targets[:, target_idx:]  # (B, T - target_idx)
                    pred = reward_logits[:, :T - target_idx, offset, :]  # (B, T - target_idx, buckets)

                    if pred.shape[1] > 0:
                        reward_loss = reward_loss + twohot_loss(
                            pred, target,
                            reward_head.bucket_centers
                        )

            reward_loss = reward_loss / L

            # Behavioral cloning loss
            action_logits = policy_head(agent_out)  # (B, T, L, action_dim)

            # MTP BC loss: predict actions at t+1, t+2, ..., t+L
            bc_loss = 0.0
            for offset in range(L):
                if offset < T - 1:
                    target_idx = min(offset + 1, T - 1)
                    target = actions[:, target_idx:]  # (B, T - target_idx)
                    pred = action_logits[:, :T - target_idx, offset, :]  # (B, T - target_idx, action_dim)

                    if pred.shape[1] > 0:
                        bc_loss = bc_loss + F.cross_entropy(
                            pred.reshape(-1, pred.shape[-1]),
                            target.reshape(-1),
                        )

            bc_loss = bc_loss / L

            # Normalize losses by running RMS
            dynamics_loss_norm = rms_trackers["dynamics"].update(dynamics_loss)
            reward_loss_norm = rms_trackers["reward"].update(reward_loss)
            bc_loss_norm = rms_trackers["bc"].update(bc_loss)

            total_loss_batch = dynamics_loss_norm + reward_loss_norm + bc_loss_norm

        # Backward
        scaler.scale(total_loss_batch).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(dynamics.parameters()) + list(reward_head.parameters()) + list(policy_head.parameters()),
            max_norm=1.0
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # Track metrics
        total_loss += total_loss_batch.item()
        total_dynamics_loss += dynamics_loss.item()
        total_reward_loss += reward_loss.item()
        total_bc_loss += bc_loss.item()
        num_batches += 1

        # Log
        if batch_idx % args.log_interval == 0:
            elapsed = time.time() - start_time
            samples_per_sec = (batch_idx + 1) * args.batch_size / elapsed
            print(
                f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                f"Loss: {total_loss_batch.item():.4f} "
                f"Dyn: {dynamics_loss.item():.4f} "
                f"Rew: {reward_loss.item():.4f} "
                f"BC: {bc_loss.item():.4f} "
                f"({samples_per_sec:.1f} samples/s)"
            )

    return {
        "loss": total_loss / num_batches,
        "dynamics_loss": total_dynamics_loss / num_batches,
        "reward_loss": total_reward_loss / num_batches,
        "bc_loss": total_bc_loss / num_batches,
    }


def save_checkpoint(
    dynamics: nn.Module,
    reward_head: nn.Module,
    policy_head: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    metrics: dict,
    args: argparse.Namespace,
    path: Path,
):
    """Save training checkpoint."""
    checkpoint = {
        "dynamics_state_dict": dynamics.state_dict(),
        "reward_head_state_dict": reward_head.state_dict(),
        "policy_head_state_dict": policy_head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "args": vars(args),
        "phase": "agent_finetune",
    }
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def main():
    args = parse_args()

    print("=" * 60)
    print("Phase 2: Agent Finetuning (BC + Reward)")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Model size: {args.model_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_len}")
    print(f"MTP length: {args.mtp_length}")
    print(f"Action dim: {args.action_dim}")
    print(f"Reward mixture: {args.use_reward_mixture}")
    print("=" * 60)

    # Create directories
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load pretrained models
    print("\nLoading pretrained tokenizer...")
    tokenizer = load_tokenizer(args.tokenizer_checkpoint, args.device)
    print(f"Tokenizer loaded from {args.tokenizer_checkpoint}")

    print("\nLoading pretrained dynamics (upgrading to agent tokens)...")
    dynamics, pretrained_args = load_pretrained_dynamics(
        args.dynamics_checkpoint, args.model_size, args.device
    )
    print(f"Dynamics loaded from {args.dynamics_checkpoint}")
    print(f"  Total parameters: {dynamics.get_num_params():,}")

    # Get model dimension from dynamics
    model_dim = dynamics.model_dim

    # Create heads
    print("\nCreating reward and policy heads...")
    reward_head = RewardHead(
        input_dim=model_dim,
        hidden_dim=256,
        num_buckets=args.num_buckets,
        mtp_length=args.mtp_length,
    ).to(args.device)

    policy_head = PolicyHead(
        input_dim=model_dim,
        action_dim=args.action_dim,
        hidden_dim=256,
        mtp_length=args.mtp_length,
    ).to(args.device)

    print(f"Reward head parameters: {sum(p.numel() for p in reward_head.parameters()):,}")
    print(f"Policy head parameters: {sum(p.numel() for p in policy_head.parameters()):,}")

    # Create shortcut forcing
    shortcut = ShortcutForcing(k_max=128).to(args.device)

    # Create dataset and dataloader
    if args.latents_dir:
        print(f"\nLoading latent sequences from {args.latents_dir}...")
    else:
        print(f"\nLoading data from {args.data_dir}...")
    dataset = ReplayDataset(
        args.data_dir,
        seq_len=args.seq_len,
        latents_dir=args.latents_dir,
        features_dir=args.features_dir,
    )

    # Create sampler for 50/50 reward mixture (DreamerV4 strategy)
    sampler = None
    shuffle = True
    if args.use_reward_mixture and dataset.use_latents and dataset.latent_dataset is not None:
        print("\nCreating 50/50 reward mixture sampler...")
        sampler = RewardMixtureSampler(dataset.latent_dataset)
        shuffle = False  # Sampler handles shuffling

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )

    # Create optimizer
    all_params = (
        list(dynamics.parameters()) +
        list(reward_head.parameters()) +
        list(policy_head.parameters())
    )
    optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=0.01)

    # Create scaler
    device_type = args.device.split(":")[0]
    scaler = GradScaler(device_type)

    # RMS trackers for loss normalization
    rms_trackers = {
        "dynamics": RunningRMS(),
        "reward": RunningRMS(),
        "bc": RunningRMS(),
    }

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    history = []
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)

        metrics = train_epoch(
            dynamics=dynamics,
            reward_head=reward_head,
            policy_head=policy_head,
            tokenizer=tokenizer,
            shortcut=shortcut,
            dataloader=dataloader,
            optimizer=optimizer,
            scaler=scaler,
            rms_trackers=rms_trackers,
            device=args.device,
            epoch=epoch,
            args=args,
        )

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Total Loss: {metrics['loss']:.4f}")
        print(f"  Dynamics Loss: {metrics['dynamics_loss']:.4f}")
        print(f"  Reward Loss: {metrics['reward_loss']:.4f}")
        print(f"  BC Loss: {metrics['bc_loss']:.4f}")

        history.append({"epoch": epoch + 1, **metrics})

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = checkpoint_dir / f"agent_finetune_epoch_{epoch + 1:03d}.pt"
            save_checkpoint(
                dynamics, reward_head, policy_head,
                optimizer, scaler, epoch, metrics, args, checkpoint_path
            )

            # Also save as latest
            latest_path = checkpoint_dir / "agent_finetune_latest.pt"
            save_checkpoint(
                dynamics, reward_head, policy_head,
                optimizer, scaler, epoch, metrics, args, latest_path
            )

    # Save training history
    history_path = checkpoint_dir / "agent_finetune_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Checkpoints saved to {checkpoint_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
