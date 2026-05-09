#!/usr/bin/env python3
"""Debug reward model training - comprehensive diagnostics."""

import sys
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ahriuwu.models import (
    create_dynamics,
    create_transformer_tokenizer,
    RewardHead,
    PolicyHead,
    DiffusionSchedule,
    symlog,
    twohot_loss,
)
from ahriuwu.data.dataset import PackedLatentSequenceDataset
from torch.utils.data import DataLoader

print("=" * 70)
print("REWARD MODEL TRAINING DIAGNOSTICS")
print("=" * 70)

# Load checkpoints
dynamics_ckpt = torch.load("checkpoints/dynamics_best.pt", map_location="cpu", weights_only=False)
tokenizer_ckpt = torch.load("checkpoints/transformer_tokenizer_best.pt", map_location="cpu", weights_only=False)

print("\n## 1. DATA COMPATIBILITY CHECK")
print("-" * 50)

# Check dynamics input projection shape
dynamics_state = dynamics_ckpt["model_state_dict"]
if "input_proj.weight" in dynamics_state:
    input_proj_shape = dynamics_state["input_proj.weight"].shape
    print(f"Dynamics input_proj.weight shape: {input_proj_shape}")
    print(f"  -> Expects input features: {input_proj_shape[1]}")
else:
    print("No input_proj.weight found in dynamics checkpoint")
    # Check what keys exist
    proj_keys = [k for k in dynamics_state.keys() if "proj" in k.lower() or "input" in k.lower()]
    print(f"  Related keys: {proj_keys[:5]}")

# Check dynamics args
dynamics_args = dynamics_ckpt.get("args", {})
print(f"Dynamics checkpoint args: latent_dim={dynamics_args.get('latent_dim')}")

# Load dataset and check latent shapes
print("\nLoading dataset...")
dataset = PackedLatentSequenceDataset(
    latents_dir="data/processed/latents_packed",
    sequence_length=32,
    stride=16,
    load_actions=True,
    features_dir="data/processed",
)

# Get one sample
sample = dataset[0]
latents = sample["latents"]
print(f"Batch latents shape: {latents.shape}")
print(f"  -> (T, C, H, W) = {latents.shape}")

# Check compatibility
if "input_proj.weight" in dynamics_state:
    expected_features = input_proj_shape[1]
    actual_features = latents.shape[1] * latents.shape[2] * latents.shape[3]  # C*H*W
    print(f"\nCompatibility check:")
    print(f"  Dynamics expects: {expected_features} features")
    print(f"  Latents provide: C*H*W = {latents.shape[1]}*{latents.shape[2]}*{latents.shape[3]} = {actual_features}")
    if expected_features != actual_features:
        print(f"  *** FATAL MISMATCH! ***")
    else:
        print(f"  OK - shapes match")

print("\n## 2. REWARD SIGNAL CHECK")
print("-" * 50)

# Create dataloader
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Sample 10 batches worth of rewards
total_frames = 0
nonzero_frames = 0
all_rewards = []

# Need to also load rewards from ReplayDataset
import json
from ahriuwu.data.actions import encode_action

class SimpleReplayDataset(torch.utils.data.Dataset):
    def __init__(self, latent_dataset, features_dir, gold_scale=0.01, death_penalty=-10.0):
        self.latent_dataset = latent_dataset
        self.features_dir = Path(features_dir)
        self.gold_scale = gold_scale
        self.death_penalty = death_penalty
        self.reward_data = {}
        self._load_rewards()

    def _load_rewards(self):
        video_ids = set(s['video_id'] for s in self.latent_dataset.sequences)
        for video_id in video_ids:
            features_path = self.features_dir / video_id / "features.json"
            if features_path.exists():
                with open(features_path) as f:
                    features = json.load(f)
                frames = features.get("frames", [])
                rewards = []
                for frame in frames:
                    gold = frame.get("gold_gained", 0) * self.gold_scale
                    death = self.death_penalty if frame.get("is_dead", False) else 0.0
                    rewards.append(gold + death)
                self.reward_data[video_id] = rewards

    def __len__(self):
        return len(self.latent_dataset)

    def __getitem__(self, idx):
        data = self.latent_dataset[idx]
        seq_info = self.latent_dataset.sequences[idx]
        video_id = seq_info['video_id']
        start_idx = seq_info['start_idx']

        seq_len = self.latent_dataset.sequence_length
        if video_id in self.reward_data:
            video_rewards = self.reward_data[video_id]
            end_idx = min(start_idx + seq_len, len(video_rewards))
            rewards = video_rewards[start_idx:end_idx]
            if len(rewards) < seq_len:
                rewards = rewards + [0.0] * (seq_len - len(rewards))
            rewards = torch.tensor(rewards, dtype=torch.float32)
        else:
            rewards = torch.zeros(seq_len, dtype=torch.float32)

        return {"latents": data["latents"], "rewards": rewards}

replay_dataset = SimpleReplayDataset(dataset, "data/processed")
replay_loader = DataLoader(replay_dataset, batch_size=1, shuffle=False)

print("Sampling 100 batches...")
for i, batch in enumerate(replay_loader):
    if i >= 100:
        break
    rewards = batch["rewards"]
    total_frames += rewards.numel()
    nonzero = (rewards != 0).sum().item()
    nonzero_frames += nonzero
    all_rewards.extend(rewards.flatten().tolist())

print(f"Total frames sampled: {total_frames}")
print(f"Non-zero reward frames: {nonzero_frames}")
print(f"Percentage: {100 * nonzero_frames / total_frames:.2f}%")

if nonzero_frames == 0:
    print("*** CRITICAL: No non-zero rewards found! Reward loading may be broken ***")

print("\n## 3. REWARD VALUES CHECK")
print("-" * 50)

all_rewards_np = np.array(all_rewards)
nonzero_rewards = all_rewards_np[all_rewards_np != 0]

if len(nonzero_rewards) > 0:
    print(f"Non-zero reward stats:")
    print(f"  Min: {nonzero_rewards.min():.4f}")
    print(f"  Max: {nonzero_rewards.max():.4f}")
    print(f"  Mean: {nonzero_rewards.mean():.4f}")
    print(f"  Std: {nonzero_rewards.std():.4f}")

    # Check for deaths vs gold
    deaths = nonzero_rewards[nonzero_rewards < 0]
    golds = nonzero_rewards[nonzero_rewards > 0]
    print(f"\nDeaths (negative rewards): {len(deaths)}")
    if len(deaths) > 0:
        print(f"  Values: {deaths[:5]}")
    print(f"Gold gains (positive rewards): {len(golds)}")
    if len(golds) > 0:
        print(f"  Sample values: {golds[:10]}")
else:
    print("No non-zero rewards to analyze!")

print("\n## 4. OVERSAMPLING STATUS")
print("-" * 50)

# Check what percentage of sequences have rewards
seqs_with_rewards = 0
for i in range(min(1000, len(dataset))):
    if dataset.has_nonzero_reward(i):
        seqs_with_rewards += 1

print(f"Sequences with non-zero reward (first 1000): {seqs_with_rewards}/1000 ({seqs_with_rewards/10:.1f}%)")
print(f"Current training uses: shuffle=False, no RewardMixtureSampler")
print(f"*** Without oversampling, model trains on ~{100-seqs_with_rewards/10:.0f}% zero rewards ***")

print("\n## 5. LOSS FUNCTION CHECK")
print("-" * 50)

# Create reward head to check params
model_dim = 512  # small model
reward_head = RewardHead(
    input_dim=model_dim,
    hidden_dim=256,
    num_buckets=255,
    mtp_length=8,
)

print(f"RewardHead parameters:")
print(f"  num_buckets: {reward_head.num_buckets}")
print(f"  bucket_low: {reward_head.bucket_low}")
print(f"  bucket_high: {reward_head.bucket_high}")
print(f"  bucket_centers shape: {reward_head.bucket_centers.shape}")
print(f"  bucket_centers range: [{reward_head.bucket_centers[0]:.4f}, {reward_head.bucket_centers[-1]:.4f}]")

# Test twohot encoding for sample rewards
test_rewards = torch.tensor([0.0, 0.14, 0.21, 1.0, -10.0])  # typical values
symlog_rewards = symlog(test_rewards)
print(f"\nTest rewards: {test_rewards.tolist()}")
print(f"Symlog transformed: {symlog_rewards.tolist()}")

# Find which buckets these map to
bucket_centers = reward_head.bucket_centers
for r, sr in zip(test_rewards.tolist(), symlog_rewards.tolist()):
    # Find closest bucket
    diffs = (bucket_centers - sr).abs()
    closest_idx = diffs.argmin().item()
    print(f"  {r:.2f} -> symlog {sr:.4f} -> bucket {closest_idx} (center={bucket_centers[closest_idx]:.4f})")

print("\n## 6. ARCHITECTURE CHECK")
print("-" * 50)

# Check dynamics model dim
dynamics = create_dynamics(
    size="small",
    latent_dim=32,
    use_agent_tokens=True,
)
print(f"Dynamics model_dim: {dynamics.model_dim}")
print(f"RewardHead input_dim: {reward_head.input_dim}")
print(f"MTP length: {reward_head.mtp_length}")

# Check if reward head gets right input
print(f"\nArchitecture flow check:")
print(f"  Latents: (B, T, 32, 16, 16)")
print(f"  -> Flattened to (B, T, 32*16*16) = (B, T, 8192)?")
print(f"  -> Or kept as (B, T, 32, 16, 16)?")

# Check dynamics input projection
print(f"\nDynamics input_proj:")
if hasattr(dynamics, 'input_proj'):
    print(f"  input_proj: {dynamics.input_proj}")
    if hasattr(dynamics.input_proj, 'weight'):
        print(f"  weight shape: {dynamics.input_proj.weight.shape}")

print("\n## 7. GRADIENT CHECK")
print("-" * 50)

# Quick forward/backward pass
device = "cuda" if torch.cuda.is_available() else "cpu"
reward_head = reward_head.to(device)

# Simulate input from dynamics (B, T, model_dim)
fake_agent_out = torch.randn(1, 32, model_dim, device=device, requires_grad=True)
reward_logits = reward_head(fake_agent_out)  # (B, T, L, num_buckets)
print(f"Reward head output shape: {reward_logits.shape}")

# Compute loss with fake target
fake_targets = torch.zeros(1, 32, device=device)
loss = 0
for offset in range(8):
    if offset < 31:
        target = fake_targets[:, offset + 1:]
        pred = reward_logits[:, :31 - offset, offset, :]
        if pred.shape[1] > 0:
            loss = loss + twohot_loss(pred, target, reward_head.bucket_centers.to(device))
loss = loss / 8

print(f"Test loss value: {loss.item():.4f}")
loss.backward()

grad_norm = fake_agent_out.grad.norm().item()
print(f"Gradient norm through reward_head: {grad_norm:.6f}")
if grad_norm == 0:
    print("*** CRITICAL: Zero gradients! Something is disconnected ***")
elif np.isnan(grad_norm):
    print("*** CRITICAL: NaN gradients! ***")
else:
    print("OK - gradients are flowing")

# Check reward head weight gradients
for name, param in reward_head.named_parameters():
    if param.grad is not None:
        print(f"  {name}: grad_norm={param.grad.norm().item():.6f}")

print("\n" + "=" * 70)
print("DIAGNOSTICS COMPLETE")
print("=" * 70)
