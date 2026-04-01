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
import random
import time
from datetime import datetime
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
    DiffusionSchedule,
    x_prediction_loss,
    symlog,
    twohot_loss,
    RunningRMS,
)
from ahriuwu.data.dataset import PackedLatentSequenceDataset, RewardMixtureSampler, VideoShuffleSampler
from ahriuwu.data.actions import encode_action
from ahriuwu.utils.logging import add_wandb_args, init_wandb, log_step, log_images, finish_wandb
from ahriuwu.utils.training import add_training_args, create_optimizer, create_wsd_schedule


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2: Agent Finetuning")
    add_training_args(parser)
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
        "--seq-len",
        type=int,
        default=32,
        help="Sequence length (frames)",
    )
    parser.add_argument(
        "--mtp-length",
        type=int,
        default=9,
        help="Multi-token prediction length (paper Eq 9: n=0..L with L=8, so 9 predictions)",
    )
    parser.add_argument(
        "--num-buckets",
        type=int,
        default=255,
        help="Number of twohot buckets for reward prediction",
    )
    parser.add_argument(
        "--use-reward-mixture",
        action="store_true",
        help="Use 50/50 reward mixture sampling (50%% uniform, 50%% reward-containing)",
    )
    # DreamerV4 stability features (should match dynamics training)
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=256,
        help="Latent dimension (must match tokenizer)",
    )
    parser.add_argument(
        "--no-qk-norm",
        action="store_true",
        help="Disable QKNorm (enabled by default)",
    )
    parser.add_argument(
        "--soft-cap",
        type=float,
        default=50.0,
        help="Attention logit soft cap (0 to disable, default 50.0)",
    )
    parser.add_argument(
        "--num-register-tokens",
        type=int,
        default=8,
        help="Number of register tokens (0 to disable)",
    )
    parser.add_argument(
        "--num-kv-heads",
        type=int,
        default=None,
        help="Number of KV heads for GQA (None = no GQA)",
    )
    parser.add_argument(
        "--independent-frame-ratio",
        type=float,
        default=0.3,
        help="Ratio of batches using independent frame mode",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        default=True,
        help="Use gradient checkpointing for memory efficiency (default: enabled)",
    )
    parser.set_defaults(num_workers=0)
    add_wandb_args(parser)
    return parser.parse_args()


class ReplayDataset(torch.utils.data.Dataset):
    """Dataset for replay data with latents, actions, and rewards.

    Wraps PackedLatentSequenceDataset to provide properly formatted data:
    - latents: (T, C, H, W) pre-tokenized latent vectors
    - actions: (T,) discrete action indices
    - rewards: (T,) reward values

    When use_latents=True, uses pre-computed latents from PackedLatentSequenceDataset.
    Otherwise falls back to placeholder for raw frame mode (not yet implemented).
    """

    def __init__(
        self,
        data_dir: str,
        seq_len: int = 32,
        latents_dir: str | None = None,
        features_dir: str | None = None,
        gold_scale: float = 0.01,
        death_penalty: float = -10.0,
    ):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.use_latents = latents_dir is not None
        self.features_dir = Path(features_dir) if features_dir else Path("data/processed")
        self.gold_scale = gold_scale
        self.death_penalty = death_penalty

        # Reward data per video: video_id -> list of reward values per frame
        self.reward_data: dict[str, list[float]] = {}
        # Death/terminal data per video: video_id -> list of bool per frame
        self.death_data: dict[str, list[bool]] = {}

        if self.use_latents:
            # Use PackedLatentSequenceDataset for pre-tokenized data
            self.latent_dataset = PackedLatentSequenceDataset(
                latents_dir=latents_dir,
                sequence_length=seq_len,
                stride=seq_len // 2,  # 50% overlap for more sequences
                load_actions=True,
                features_dir=features_dir,
            )
            print(f"Loaded {len(self.latent_dataset)} latent sequences with actions")

            # Load rewards from features.json
            self._load_rewards()

            # Summarize missing data
            all_video_ids = set(s['video_id'] for s in self.latent_dataset.sequences)
            videos_with_rewards = set(self.reward_data.keys()) & all_video_ids
            # Check which videos have actions by sampling the first sequence per video
            videos_with_actions = set()
            seen_videos = set()
            for seq_info in self.latent_dataset.sequences:
                vid = seq_info['video_id']
                if vid not in seen_videos:
                    seen_videos.add(vid)
                    if seq_info.get('has_actions', True):  # assume True if not tracked
                        videos_with_actions.add(vid)
            self._n_videos_missing_actions = len(all_video_ids) - len(videos_with_actions)
            self._n_videos_missing_rewards = len(all_video_ids) - len(videos_with_rewards)
            print(f"Loaded reward data for {len(videos_with_rewards)}/{len(all_video_ids)} videos")
            print(f"Missing data summary: {self._n_videos_missing_actions} videos missing actions, "
                  f"{self._n_videos_missing_rewards} videos missing rewards")
        else:
            # Placeholder for raw frame mode
            self.latent_dataset = None
            replay_files = list(self.data_dir.glob("*.pt")) + list(self.data_dir.glob("*.npz"))
            print(f"Found {len(replay_files)} replay files (raw frame mode - not yet implemented)")
            self.replay_files = replay_files

    def _load_rewards(self):
        """Load reward and terminal data from features.json files.

        Populates self.reward_data with per-frame reward values, and
        self.death_data with per-frame boolean death flags used to
        construct the ``continues`` tensor (1.0 = alive, 0.0 = dead).

        Death detection uses health bar presence/absence from the feature
        extraction pipeline (``health_bar_x`` field).  The ``is_dead`` field
        is NOT written by the extraction pipeline, so we replicate the
        lookback/lookahead logic from ``PackedLatentSequenceDataset._get_rewards``:

        1. If the health bar was present in >= 3 of the previous 5 frames
           but is absent on the current frame, this is a *candidate* death.
        2. Confirm by checking that the health bar stays absent for >= 2 of
           the next 3 frames (filters out single-frame flicker/occlusion).
        """
        video_ids = set(s['video_id'] for s in self.latent_dataset.sequences)
        context_frames = 5  # frames to look back for health bar history

        for video_id in video_ids:
            features_path = self.features_dir / video_id / "features.json"
            if not features_path.exists():
                continue

            with open(features_path) as f:
                features = json.load(f)

            frames = features.get("frames", [])
            rewards = []
            deaths = []
            num_frames = len(frames)

            for frame_idx, frame in enumerate(frames):
                reward = frame.get("gold_gained", 0) * self.gold_scale

                # Death detection via health bar presence/absence
                is_dead = False
                curr_hb = frame.get("health_bar_x") is not None

                if not curr_hb:
                    # Count how many of the previous frames had a health bar
                    prev_hb_count = 0
                    for lookback in range(1, context_frames + 1):
                        prev_idx = frame_idx - lookback
                        if 0 <= prev_idx < num_frames:
                            if frames[prev_idx].get("health_bar_x") is not None:
                                prev_hb_count += 1

                    # Health bar was present recently but now gone
                    if prev_hb_count >= 3:
                        # Confirm it stays gone (not just a flicker)
                        gone_count = 0
                        for lookahead in range(1, 4):
                            next_idx = frame_idx + lookahead
                            if next_idx < num_frames:
                                if frames[next_idx].get("health_bar_x") is None:
                                    gone_count += 1
                        if gone_count >= 2:
                            is_dead = True
                            reward += self.death_penalty

                rewards.append(reward)
                deaths.append(is_dead)

            self.reward_data[video_id] = rewards
            self.death_data[video_id] = deaths

    def __len__(self):
        if self.use_latents:
            return len(self.latent_dataset)
        return len(self.replay_files) * 10  # Placeholder

    def __getitem__(self, idx):
        if self.use_latents:
            # Get data from PackedLatentSequenceDataset
            data = self.latent_dataset[idx]

            latents = data["latents"]  # (T, C, H, W) = (T, 32, 16, 16) for transformer tokenizer

            # Get sequence info to look up rewards
            seq_info = self.latent_dataset.sequences[idx]
            video_id = seq_info['video_id']
            # Use start_frame (actual frame number), not start_idx (array position)
            # to correctly align rewards with frames
            start_frame = seq_info['start_frame']

            # Extract both the factorized action dict (for dynamics model) and
            # the encoded discrete action tensor (for BC loss)
            raw_actions_dict = data.get("actions")
            if raw_actions_dict is not None:
                # Movement is now (T, 2) float tensor of (x, y) coordinates
                movement_targets = raw_actions_dict['movement']  # (T, 2) float

                # Encode discrete ability actions (movement=0 placeholder since
                # movement is now continuous and handled separately)
                actions = []
                for t in range(self.seq_len):
                    action = encode_action(
                        movement=0,  # movement is now continuous, not used here
                        abilities={
                            'Q': bool(raw_actions_dict['Q'][t].item()),
                            'W': bool(raw_actions_dict['W'][t].item()),
                            'E': bool(raw_actions_dict['E'][t].item()),
                            'R': bool(raw_actions_dict['R'][t].item()),
                            'D': bool(raw_actions_dict['D'][t].item()),
                            'F': bool(raw_actions_dict['F'][t].item()),
                            'item': bool(raw_actions_dict['item'][t].item()),
                            'B': bool(raw_actions_dict['B'][t].item()),
                        }
                    )
                    actions.append(action)
                actions = torch.tensor(actions, dtype=torch.long)

                # Build factorized dict for dynamics model (movement + per-ability tensors)
                factorized_actions = {
                    'movement': raw_actions_dict['movement'],  # (T, 2)
                    'Q': raw_actions_dict['Q'],       # (T,)
                    'W': raw_actions_dict['W'],
                    'E': raw_actions_dict['E'],
                    'R': raw_actions_dict['R'],
                    'D': raw_actions_dict['D'],
                    'F': raw_actions_dict['F'],
                    'item': raw_actions_dict['item'],
                    'B': raw_actions_dict['B'],
                }
            else:
                # No actions available - use zeros / center
                actions = torch.zeros(self.seq_len, dtype=torch.long)
                movement_targets = torch.full((self.seq_len, 2), 0.5, dtype=torch.float32)
                # Default factorized actions: center movement, no abilities
                factorized_actions = {
                    'movement': torch.full((self.seq_len, 2), 0.5, dtype=torch.float32),
                    'Q': torch.zeros(self.seq_len, dtype=torch.long),
                    'W': torch.zeros(self.seq_len, dtype=torch.long),
                    'E': torch.zeros(self.seq_len, dtype=torch.long),
                    'R': torch.zeros(self.seq_len, dtype=torch.long),
                    'D': torch.zeros(self.seq_len, dtype=torch.long),
                    'F': torch.zeros(self.seq_len, dtype=torch.long),
                    'item': torch.zeros(self.seq_len, dtype=torch.long),
                    'B': torch.zeros(self.seq_len, dtype=torch.long),
                }

            # Get rewards and continues from our loaded reward/death data
            if video_id in self.reward_data:
                video_rewards = self.reward_data[video_id]
                end_idx = min(start_frame + self.seq_len, len(video_rewards))
                rewards = video_rewards[start_frame:end_idx]
                if len(rewards) < self.seq_len:
                    rewards = rewards + [0.0] * (self.seq_len - len(rewards))
                rewards = torch.tensor(rewards, dtype=torch.float32)
            else:
                rewards = torch.zeros(self.seq_len, dtype=torch.float32)

            # Build continues tensor: 0.0 for terminal (death) frames, 1.0 otherwise
            if video_id in self.death_data:
                video_deaths = self.death_data[video_id]
                end_idx = min(start_frame + self.seq_len, len(video_deaths))
                deaths = video_deaths[start_frame:end_idx]
                if len(deaths) < self.seq_len:
                    deaths = deaths + [False] * (self.seq_len - len(deaths))
                continues = torch.tensor(
                    [0.0 if d else 1.0 for d in deaths], dtype=torch.float32
                )
            else:
                continues = torch.ones(self.seq_len, dtype=torch.float32)

            return {
                "latents": latents,  # (T, C, H, W) pre-tokenized
                "actions": actions,  # (T,) discrete ability actions
                "movement_targets": movement_targets,  # (T, 2) continuous (x, y)
                "rewards": rewards,  # (T,)
                "continues": continues,  # (T,) 1.0=alive, 0.0=dead
                "factorized_actions": factorized_actions,  # dict for dynamics model
                "has_actions": raw_actions_dict is not None,
                "has_rewards": video_id in self.reward_data,
            }
        else:
            # Placeholder for raw frame mode
            T = self.seq_len
            C, H, W = 3, 352, 352
            return {
                "frames": torch.rand(T, C, H, W),
                "actions": torch.randint(0, 128, (T,)),
                "movement_targets": torch.full((T, 2), 0.5),
                "rewards": torch.randn(T) * 10,
                "continues": torch.ones(T),
            }


def load_pretrained_dynamics(
    checkpoint_path: str,
    model_size: str,
    device: str,
    latent_dim: int = 32,
    use_qk_norm: bool = True,
    soft_cap: float | None = 50.0,
    num_register_tokens: int = 8,
    num_kv_heads: int | None = None,
    gradient_checkpointing: bool = True,
):
    """Load pretrained dynamics and upgrade to use agent tokens."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Get latent_dim from checkpoint args (default 32 for transformer tokenizer)
    pretrained_args = checkpoint.get("args", {})
    latent_dim = pretrained_args.get("latent_dim", 32)

    # Auto-detect use_actions from pretrained checkpoint weights
    pretrained_state = checkpoint["model_state_dict"]
    use_actions = any("action_embed" in k for k in pretrained_state.keys())
    if use_actions:
        print("  Detected action conditioning in pretrained checkpoint")

    # Create new dynamics with agent tokens enabled
    dynamics = create_dynamics(
        size=model_size,
        latent_dim=latent_dim,
        use_agent_tokens=True,
        use_actions=use_actions,
        num_tasks=1,
        agent_layers=4,
        use_qk_norm=use_qk_norm,
        soft_cap=soft_cap,
        num_register_tokens=num_register_tokens,
        num_kv_heads=num_kv_heads,
        gradient_checkpointing=gradient_checkpointing,
    )
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
    """Load pretrained transformer tokenizer."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    state_dict = checkpoint["model_state_dict"]
    args = checkpoint.get("args", {})

    model_size = args.get("model_size", "small")
    use_rope = args.get("use_rope", True)
    print(f"  Transformer tokenizer: size={model_size}, use_rope={use_rope}")
    tokenizer = create_transformer_tokenizer(size=model_size, use_rope=use_rope)
    tokenizer.load_state_dict(state_dict, strict=False)

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
    diffusion: DiffusionSchedule,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    scheduler,
    rms_trackers: dict,
    device: str,
    epoch: int,
    args: argparse.Namespace,
):
    """Train for one epoch."""
    from ahriuwu.models.diffusion import ShortcutForcing
    _shortcut = ShortcutForcing(k_max=64)

    dynamics.train()
    reward_head.train()
    policy_head.train()

    total_loss = 0.0
    total_dynamics_loss = 0.0
    total_reward_loss = 0.0
    total_bc_loss = 0.0
    total_correct_top1 = 0
    total_correct_top5 = 0
    total_predictions = 0
    num_batches = 0
    total_has_actions = 0
    total_has_rewards = 0
    total_samples = 0
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

        # Track data completeness
        if "has_actions" in batch:
            total_has_actions += batch["has_actions"].sum().item()
            total_has_rewards += batch["has_rewards"].sum().item()
            total_samples += batch["has_actions"].numel()

        actions = batch["actions"].to(device)  # (B, T) discrete ability actions
        movement_targets = batch["movement_targets"].to(device)  # (B, T, 2) continuous (x, y)
        rewards = batch["rewards"].to(device)  # (B, T)
        # continues is loaded here for Phase 3 lambda returns (not used in Phase 2 BC training)
        continues = batch["continues"].to(device)  # (B, T) 1.0=alive, 0.0=dead

        B, T = actions.shape

        with autocast(device_type=device_type, dtype=dtype):

            # Sample diffusion timesteps (per-timestep for diffusion forcing)
            tau = diffusion.sample_diffusion_forcing_timesteps(B, T, device=device)

            # Add noise to clean latents
            z_noisy, _ = diffusion.add_noise(z, tau)
            z_target = z  # X-prediction: predict clean from noisy

            # Independent frame mode: 30% of batches disable temporal attention
            # to prevent temporal shortcut learning (DreamerV4 Section 3.2)
            use_independent = random.random() < args.independent_frame_ratio

            # Build factorized action dict for dynamics model if it uses actions
            actions_dict = None
            _inner = getattr(dynamics, '_orig_mod', dynamics)
            has_actions = _inner.use_actions
            if has_actions and "factorized_actions" in batch:
                # Move each tensor in the factorized dict to device
                actions_dict = {
                    k: v.to(device) for k, v in batch["factorized_actions"].items()
                }

            # Sample step_size to preserve shortcut forcing during Phase 2
            # (Paper: "continue applying the dynamics loss" — includes multi-scale shortcuts)
            step_size = _shortcut.sample_step_size(B, device=device)

            # Forward through dynamics with agent tokens, actions, and step_size
            z_pred, agent_out = dynamics(
                z_noisy, tau,
                step_size=step_size,
                actions=actions_dict,
                independent_frames=use_independent,
                task_id=None,  # TODO: pass task_ids from batch when multi-task is enabled
            )

            # Dynamics loss (x-prediction with ramp weighting, matching Phase 1)
            dynamics_loss = x_prediction_loss(z_pred, z_target, tau, use_ramp_weight=True)

            # Reward prediction loss
            reward_logits = reward_head(agent_out)  # (B, T, L, num_buckets)
            reward_targets = symlog(rewards)  # (B, T)

            # MTP reward loss: predict rewards at t+n for n=0..L-1 (Eq. 9)
            reward_loss = torch.tensor(0.0, device=device)
            L = args.mtp_length
            for offset in range(L):
                if T - offset > 0:
                    # Target is reward at t + offset (n=0 → current timestep)
                    target = reward_targets[:, offset:]  # (B, T - offset)
                    pred = reward_logits[:, :T - offset, offset, :]  # (B, T - offset, buckets)

                    if pred.shape[1] > 0:
                        reward_loss = reward_loss + twohot_loss(
                            pred, target,
                            reward_head.bucket_centers
                        )

            reward_loss = reward_loss / L

            # Behavioral cloning loss (vectorized binary abilities + continuous movement)
            ability_logits, movement_pred = policy_head(agent_out)
            # ability_logits: (B, T, L, num_abilities) — independent binary logits
            # movement_pred: (B, T, L, 2)

            # Build binary ability targets from factorized_actions: (B, T, 8)
            from ahriuwu.constants import ABILITY_KEYS
            ability_binary = torch.stack(
                [batch["factorized_actions"][k].to(device).float() for k in ABILITY_KEYS],
                dim=-1,
            )  # (B, T, 8)

            # MTP BC loss: predict actions at t+n for n=0..L-1 (Eq. 9)
            bc_loss_ability = torch.tensor(0.0, device=device)
            bc_loss_movement = torch.tensor(0.0, device=device)
            for offset in range(L):
                if T - offset > 0:
                    # Vectorized binary ability loss (independent BCE per ability)
                    ability_target = ability_binary[:, offset:]  # (B, T - offset, 8)
                    ability_pred = ability_logits[:, :T - offset, offset, :]  # (B, T - offset, 8)

                    if ability_pred.shape[1] > 0:
                        bc_loss_ability = bc_loss_ability + F.binary_cross_entropy_with_logits(
                            ability_pred, ability_target,
                        )

                    # Continuous movement loss (MSE)
                    move_target = movement_targets[:, offset:]  # (B, T - offset, 2)
                    move_pred = movement_pred[:, :T - offset, offset, :]  # (B, T - offset, 2)

                    if move_pred.shape[1] > 0:
                        bc_loss_movement = bc_loss_movement + F.mse_loss(move_pred, move_target)

            bc_loss = (bc_loss_ability + bc_loss_movement) / L

            # Compute action prediction accuracy (per-ability, offset=0)
            with torch.no_grad():
                pred_t0 = ability_logits[:, :, 0, :]  # (B, T, 8)
                target_t0 = ability_binary  # (B, T, 8)

                # Per-ability accuracy: predicted binary matches target
                pred_binary = (pred_t0 > 0).float()  # threshold logits at 0
                correct = (pred_binary == target_t0).sum().item()
                total_correct_top1 += correct
                total_correct_top5 += correct  # same metric for binary
                total_predictions += target_t0.numel()

            # Normalize losses by running RMS
            dynamics_loss_norm = rms_trackers["dynamics"].update(dynamics_loss)
            reward_loss_norm = rms_trackers["reward"].update(reward_loss)
            bc_loss_norm = rms_trackers["bc"].update(bc_loss)

            total_loss_batch = dynamics_loss_norm + reward_loss_norm + bc_loss_norm

        if torch.isnan(total_loss_batch) or torch.isinf(total_loss_batch):
            print(f"[WARN] NaN/Inf loss at batch {batch_idx}! "
                  f"dynamics={dynamics_loss.item():.4f} reward={reward_loss.item() if isinstance(reward_loss, torch.Tensor) else reward_loss} "
                  f"bc={bc_loss.item() if isinstance(bc_loss, torch.Tensor) else bc_loss}")
            optimizer.zero_grad()
            continue

        # Backward
        scaler.scale(total_loss_batch).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(dynamics.parameters()) + list(reward_head.parameters()) + list(policy_head.parameters()),
            max_norm=1.0
        )
        # Log gradient norms per component
        grad_norms = {}
        for name, mod in [("dynamics", dynamics), ("reward", reward_head), ("policy", policy_head)]:
            total_norm = sum(p.grad.norm().item()**2 for p in mod.parameters() if p.grad is not None)**0.5
            grad_norms[name] = total_norm

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
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
            acc_top1 = 100 * total_correct_top1 / total_predictions if total_predictions > 0 else 0
            acc_top5 = 100 * total_correct_top5 / total_predictions if total_predictions > 0 else 0

            log_step({
                "train/loss": total_loss_batch.item(),
                "train/dynamics_loss": dynamics_loss.item(),
                "train/reward_loss": reward_loss.item() if isinstance(reward_loss, torch.Tensor) else reward_loss,
                "train/bc_loss": bc_loss.item() if isinstance(bc_loss, torch.Tensor) else bc_loss,
                "train/acc_top1": acc_top1,
                "train/acc_top5": acc_top5,
                "train/lr": scheduler.get_last_lr()[0],
                "train/samples_per_sec": samples_per_sec,
                "train/epoch": epoch,
                "train/grad_norm_dynamics": grad_norms["dynamics"],
                "train/grad_norm_reward": grad_norms["reward"],
                "train/grad_norm_policy": grad_norms["policy"],
                "train/frac_has_actions": total_has_actions / total_samples if total_samples > 0 else 0,
                "train/frac_has_rewards": total_has_rewards / total_samples if total_samples > 0 else 0,
            }, step=batch_idx + epoch * len(dataloader))

            print(
                f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                f"Loss: {total_loss_batch.item():.4f} "
                f"Dyn: {dynamics_loss.item():.4f} "
                f"Rew: {reward_loss.item():.4f} "
                f"BC: {bc_loss.item():.4f} "
                f"Acc@1: {acc_top1:.1f}% Acc@5: {acc_top5:.1f}% "
                f"GradNorm[D:{grad_norms['dynamics']:.2f} R:{grad_norms['reward']:.2f} P:{grad_norms['policy']:.2f}] "
                f"({samples_per_sec:.1f} samples/s)"
            )

    final_acc_top1 = 100 * total_correct_top1 / total_predictions if total_predictions > 0 else 0
    final_acc_top5 = 100 * total_correct_top5 / total_predictions if total_predictions > 0 else 0
    return {
        "loss": total_loss / num_batches,
        "dynamics_loss": total_dynamics_loss / num_batches,
        "reward_loss": total_reward_loss / num_batches,
        "bc_loss": total_bc_loss / num_batches,
        "acc_top1": final_acc_top1,
        "acc_top5": final_acc_top5,
    }


def save_checkpoint(
    dynamics: nn.Module,
    reward_head: nn.Module,
    policy_head: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    scheduler,
    rms_trackers: dict,
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
        "scheduler_state_dict": scheduler.state_dict(),
        "rms_state": {k: v.state_dict() for k, v in rms_trackers.items()},
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
    print(f"Latent dim: {args.latent_dim}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_len}")
    print(f"MTP length: {args.mtp_length}")
    print(f"Action dim: {args.action_dim}")
    print(f"Reward mixture: {args.use_reward_mixture}")
    print(f"QKNorm: {'ENABLED' if not args.no_qk_norm else 'DISABLED'}")
    print(f"Soft cap: {args.soft_cap if args.soft_cap > 0 else 'DISABLED'}")
    print(f"Register tokens: {args.num_register_tokens}")
    if args.num_kv_heads:
        print(f"GQA: {args.num_kv_heads} KV heads")
    print(f"Independent frame ratio: {args.independent_frame_ratio:.0%}")
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
        args.dynamics_checkpoint,
        args.model_size,
        args.device,
        latent_dim=args.latent_dim,
        use_qk_norm=not args.no_qk_norm,
        soft_cap=args.soft_cap if args.soft_cap > 0 else None,
        num_register_tokens=args.num_register_tokens,
        num_kv_heads=args.num_kv_heads,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    print(f"Dynamics loaded from {args.dynamics_checkpoint}")
    print(f"  Total parameters: {dynamics.get_num_params():,}")
    print(f"  Gradient checkpointing: {'ENABLED' if args.gradient_checkpointing else 'DISABLED'}")

    # Extract model attributes BEFORE torch.compile (avoids fragile _orig_mod access)
    model_dim = dynamics.model_dim
    dynamics_use_actions = dynamics.use_actions

    # torch.compile the dynamics model
    if not getattr(args, 'no_compile', False):
        print("Compiling dynamics model with torch.compile...")
        dynamics = torch.compile(dynamics)

    # model_dim and dynamics_use_actions already extracted above

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
        num_abilities=8,  # Q/W/E/R/D/F/item/B — vectorized binary (DreamerV4 paper)
        hidden_dim=256,
        mtp_length=args.mtp_length,
    ).to(args.device)

    print(f"Reward head parameters: {sum(p.numel() for p in reward_head.parameters()):,}")
    print(f"Policy head parameters: {sum(p.numel() for p in policy_head.parameters()):,}")

    # Create diffusion schedule for noise
    diffusion = DiffusionSchedule(device=args.device)

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

    # Create sampler - reward mixture (default) or video shuffle
    sampler = None
    shuffle = False  # Always use sampler, not shuffle
    if dataset.use_latents and dataset.latent_dataset is not None:
        if args.use_reward_mixture:
            # Use 50/50 reward mixture sampler (DreamerV4 paper default)
            print("\nCreating 50/50 reward mixture sampler...")
            sampler = RewardMixtureSampler(dataset.latent_dataset)
        else:
            # Use video shuffle sampler for cache efficiency
            print("\nCreating video shuffle sampler...")
            sampler = VideoShuffleSampler(dataset.latent_dataset, filter_empty_rewards=True)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Create optimizer
    all_params = (
        list(dynamics.parameters()) +
        list(reward_head.parameters()) +
        list(policy_head.parameters())
    )
    optimizer = create_optimizer(all_params, args.lr, args.weight_decay, use_8bit=args.use_8bit_adam)

    # WSD Learning Rate Schedule
    total_steps = args.epochs * len(dataloader)
    scheduler = create_wsd_schedule(optimizer, total_steps, args.warmup_steps, args.decay_steps)

    # Create scaler
    device_type = args.device.split(":")[0]
    scaler = GradScaler(device_type)

    # RMS trackers for loss normalization
    rms_trackers = {
        "dynamics": RunningRMS(),
        "reward": RunningRMS(),
        "bc": RunningRMS(),
    }

    # Resume from Phase 2 checkpoint if provided
    start_epoch = 0
    if args.resume:
        print(f"\nResuming from {args.resume}...")
        ckpt = torch.load(args.resume, weights_only=False, map_location=args.device)
        # Load model states
        dyn_sd = ckpt["dynamics_state_dict"]
        if any(k.startswith("_orig_mod.") for k in dyn_sd):
            dyn_sd = {k.replace("_orig_mod.", ""): v for k, v in dyn_sd.items()}
        _inner = getattr(dynamics, '_orig_mod', dynamics)
        _inner.load_state_dict(dyn_sd, strict=False)
        reward_head.load_state_dict(ckpt["reward_head_state_dict"])
        policy_head.load_state_dict(ckpt["policy_head_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "rms_state" in ckpt:
            for k, state in ckpt["rms_state"].items():
                if k in rms_trackers:
                    rms_trackers[k].load_state_dict(state)
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resumed from epoch {start_epoch}")

    # Initialize wandb
    wandb_run = init_wandb(args, job_type="agent_finetune", extra_config={
        "dynamics_params": dynamics.get_num_params() if hasattr(dynamics, 'get_num_params') else sum(p.numel() for p in dynamics.parameters()),
        "reward_head_params": sum(p.numel() for p in reward_head.parameters()),
        "policy_head_params": sum(p.numel() for p in policy_head.parameters()),
    })

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    history = []
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)

        metrics = train_epoch(
            dynamics=dynamics,
            reward_head=reward_head,
            policy_head=policy_head,
            tokenizer=tokenizer,
            diffusion=diffusion,
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
        print(f"  Dynamics Loss: {metrics['dynamics_loss']:.4f}")
        print(f"  Reward Loss: {metrics['reward_loss']:.4f}")
        print(f"  BC Loss: {metrics['bc_loss']:.4f}")
        print(f"  Action Acc@1: {metrics['acc_top1']:.1f}%  Acc@5: {metrics['acc_top5']:.1f}%")

        log_step({
            "epoch/loss": metrics["loss"],
            "epoch/dynamics_loss": metrics["dynamics_loss"],
            "epoch/reward_loss": metrics["reward_loss"],
            "epoch/bc_loss": metrics["bc_loss"],
            "epoch/acc_top1": metrics["acc_top1"],
            "epoch/acc_top5": metrics["acc_top5"],
            "epoch/epoch": epoch + 1,
        }, step=(epoch + 1) * len(dataloader))

        history.append({"epoch": epoch + 1, **metrics})

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = checkpoint_dir / f"agent_finetune_epoch_{epoch + 1:03d}.pt"
            save_checkpoint(
                dynamics, reward_head, policy_head,
                optimizer, scaler, scheduler, rms_trackers, epoch, metrics, args, checkpoint_path
            )

            # Also save as latest
            latest_path = checkpoint_dir / "agent_finetune_latest.pt"
            save_checkpoint(
                dynamics, reward_head, policy_head,
                optimizer, scaler, scheduler, rms_trackers, epoch, metrics, args, latest_path
            )

    # Save training history
    history_path = checkpoint_dir / "agent_finetune_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Checkpoints saved to {checkpoint_dir}")
    print("=" * 60)

    finish_wandb()


if __name__ == "__main__":
    main()
