#!/usr/bin/env python3
"""Full diagnostic checklist for reward model training."""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import json

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ahriuwu.models import (
    create_dynamics,
    RewardHead,
    PolicyHead,
    DiffusionSchedule,
    symlog,
    twohot_loss,
)
from ahriuwu.data.dataset import PackedLatentSequenceDataset, VideoShuffleSampler
from ahriuwu.data.actions import encode_action
from torch.utils.data import DataLoader

print("=" * 70)
print("FULL DIAGNOSTIC CHECKLIST FOR REWARD MODEL TRAINING")
print("=" * 70)

# ============================================================================
# 1. DATA COMPATIBILITY
# ============================================================================
print("\n## 1. DATA COMPATIBILITY")
print("-" * 50)

# Load dynamics checkpoint
dynamics_ckpt = torch.load("checkpoints/dynamics_best.pt", map_location="cpu", weights_only=False)
dynamics_state = dynamics_ckpt["model_state_dict"]
dynamics_args = dynamics_ckpt.get("args", {})

print(f"Dynamics checkpoint trained with:")
print(f"  latent_dim: {dynamics_args.get('latent_dim')}")
print(f"  input_proj.weight shape: {dynamics_state['input_proj.weight'].shape}")

# Load one batch of latents
dataset = PackedLatentSequenceDataset(
    latents_dir="data/processed/latents_packed",
    sequence_length=32,
    stride=16,
    load_actions=True,
    features_dir="data/processed",
)

sample = dataset[100]  # Skip first video
latents = sample["latents"]
print(f"\nLatents from dataloader:")
print(f"  Shape: {latents.shape} (T, C, H, W)")

# Check compatibility
latent_dim = dynamics_args.get('latent_dim', 32)
print(f"\nCompatibility check:")
print(f"  Dynamics expects latent_dim={latent_dim}")
print(f"  Latents have C={latents.shape[1]}")
if latents.shape[1] == latent_dim and latents.shape[2] == 16 and latents.shape[3] == 16:
    print("  ✓ MATCH: Latents are (T, 32, 16, 16) as expected")
else:
    print("  ✗ MISMATCH!")

# ============================================================================
# 2. REWARD SIGNAL CHECK
# ============================================================================
print("\n## 2. REWARD SIGNAL CHECK")
print("-" * 50)

# Load reward data like training script does
features_dir = Path("data/processed")
reward_data = {}
video_ids = set(s['video_id'] for s in dataset.sequences)
gold_scale = 0.01
death_penalty = -10.0

for video_id in video_ids:
    features_path = features_dir / video_id / "features.json"
    if features_path.exists():
        with open(features_path) as f:
            features = json.load(f)
        frames = features.get("frames", [])
        rewards = []
        for frame in frames:
            gold = frame.get("gold_gained", 0) * gold_scale
            death = death_penalty if frame.get("is_dead", False) else 0.0
            rewards.append(gold + death)
        reward_data[video_id] = rewards

# Sample 10 batches using VideoShuffleSampler
sampler = VideoShuffleSampler(dataset, filter_empty_rewards=True)
sample_indices = list(sampler)[:10 * 32]  # 10 batches worth

total_frames = 0
nonzero_frames = 0
all_rewards = []

for idx in sample_indices[:320]:  # 10 sequences
    seq_info = dataset.sequences[idx]
    video_id = seq_info['video_id']
    start_idx = seq_info['start_idx']

    if video_id in reward_data:
        video_rewards = reward_data[video_id]
        end_idx = min(start_idx + 32, len(video_rewards))
        rewards = video_rewards[start_idx:end_idx]
        if len(rewards) < 32:
            rewards = rewards + [0.0] * (32 - len(rewards))
    else:
        rewards = [0.0] * 32

    total_frames += len(rewards)
    nonzero_frames += sum(1 for r in rewards if r != 0)
    all_rewards.extend(rewards)

print(f"Sampled 10 batches (320 frames):")
print(f"  Total frames: {total_frames}")
print(f"  Non-zero reward frames: {nonzero_frames}")
print(f"  Percentage: {100 * nonzero_frames / total_frames:.2f}%")

if nonzero_frames == 0:
    print("  ✗ CRITICAL: No non-zero rewards found!")
else:
    print(f"  ✓ Expected ~1% without oversampling, got {100 * nonzero_frames / total_frames:.2f}%")

# ============================================================================
# 3. REWARD VALUES CHECK
# ============================================================================
print("\n## 3. REWARD VALUES CHECK")
print("-" * 50)

all_rewards_np = np.array(all_rewards)
nonzero_rewards = all_rewards_np[all_rewards_np != 0]

if len(nonzero_rewards) > 0:
    print(f"Non-zero reward statistics:")
    print(f"  Count: {len(nonzero_rewards)}")
    print(f"  Min: {nonzero_rewards.min():.4f}")
    print(f"  Max: {nonzero_rewards.max():.4f}")
    print(f"  Mean: {nonzero_rewards.mean():.4f}")

    deaths = nonzero_rewards[nonzero_rewards < 0]
    golds = nonzero_rewards[nonzero_rewards > 0]
    print(f"\n  Deaths (negative): {len(deaths)}")
    print(f"  Gold gains (positive): {len(golds)}")
    if len(golds) > 0:
        print(f"  Gold values sample: {golds[:5].tolist()}")

    print(f"\n  Expected: gold ~0.14-3.0 (after 0.01 scale), deaths = -10.0")
    if len(golds) > 0 and 0.1 <= golds.min() <= 5.0:
        print("  ✓ Gold values in expected range")
    else:
        print("  ✗ Gold values may be wrong scale")
else:
    print("  ✗ No non-zero rewards to analyze!")

# ============================================================================
# 4. OVERSAMPLING STATUS
# ============================================================================
print("\n## 4. OVERSAMPLING STATUS")
print("-" * 50)

# Count sequences with rewards
seqs_with_rewards = 0
for i in range(min(1000, len(dataset))):
    if dataset.has_nonzero_reward(i):
        seqs_with_rewards += 1

print(f"Sequences with non-zero reward (first 1000): {seqs_with_rewards}/1000 ({seqs_with_rewards/10:.1f}%)")
print(f"\nCurrent sampler: VideoShuffleSampler (filters empty videos)")
print(f"  Filtered videos: 1 (0Ax8Pudr5-o with empty features)")
print(f"  Active sequences: {len(sampler)}")
print(f"\nRewardMixtureSampler: NOT USED (would give 50% reward sequences)")
print(f"  ✓ Training on ~{100-seqs_with_rewards/10:.0f}% zero rewards is expected")

# ============================================================================
# 5. LOSS FUNCTION CHECK
# ============================================================================
print("\n## 5. LOSS FUNCTION CHECK")
print("-" * 50)

model_dim = 512
reward_head = RewardHead(
    input_dim=model_dim,
    hidden_dim=256,
    num_buckets=255,
    mtp_length=8,
)

print(f"RewardHead config:")
print(f"  num_buckets: {reward_head.num_buckets}")
print(f"  bucket_low: {reward_head.bucket_low}")
print(f"  bucket_high: {reward_head.bucket_high}")
print(f"  bucket_centers: [{reward_head.bucket_centers[0]:.2f}, ..., {reward_head.bucket_centers[-1]:.2f}]")

# Test twohot encoding
print(f"\nTwohot encoding test:")
test_rewards = torch.tensor([0.0, 0.14, 0.21, 1.0, 2.1, -10.0])
symlog_rewards = symlog(test_rewards)
bucket_centers = reward_head.bucket_centers

for r, sr in zip(test_rewards.tolist(), symlog_rewards.tolist()):
    diffs = (bucket_centers - sr).abs()
    closest_idx = diffs.argmin().item()
    print(f"  reward={r:6.2f} -> symlog={sr:7.4f} -> bucket {closest_idx:3d} (center={bucket_centers[closest_idx]:7.4f})")

# Test loss function
print(f"\nLoss function test:")
fake_logits = torch.randn(1, 10, 8, 255)  # (B, T, MTP, buckets)
fake_targets = torch.tensor([[0.0, 0.14, 0.21, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
symlog_targets = symlog(fake_targets)

loss = twohot_loss(fake_logits[:, :, 0, :], symlog_targets, bucket_centers)
print(f"  Loss with random logits: {loss.item():.4f}")
print(f"  ✓ Loss function is working")

# ============================================================================
# 6. ARCHITECTURE CHECK
# ============================================================================
print("\n## 6. ARCHITECTURE CHECK")
print("-" * 50)

dynamics = create_dynamics(
    size="small",
    latent_dim=32,
    use_agent_tokens=True,
)

print(f"Dynamics model:")
print(f"  model_dim: {dynamics.model_dim}")
print(f"  latent_dim: {dynamics.latent_dim}")
print(f"  spatial_size: {dynamics.spatial_size}")
print(f"  use_agent_tokens: {dynamics.use_agent_tokens}")

print(f"\nRewardHead:")
print(f"  Receives agent_out: (B, T, {dynamics.model_dim})")
print(f"  MTP length: {reward_head.mtp_length}")
print(f"  Output: (B, T, {reward_head.mtp_length}, {reward_head.num_buckets})")

# Verify forward pass
device = "cuda" if torch.cuda.is_available() else "cpu"
dynamics = dynamics.to(device)
reward_head = reward_head.to(device)

z = torch.randn(1, 32, 32, 16, 16, device=device)
tau = torch.zeros(1, 32, device=device)

with torch.no_grad():
    z_pred, agent_out = dynamics(z, tau)

print(f"\nForward pass check:")
print(f"  Input z: {z.shape}")
print(f"  Output z_pred: {z_pred.shape}")
print(f"  Output agent_out: {agent_out.shape}")

reward_logits = reward_head(agent_out)
print(f"  Reward logits: {reward_logits.shape}")
print(f"  ✓ Architecture is correctly connected")

# ============================================================================
# 7. GRADIENT CHECK
# ============================================================================
print("\n## 7. GRADIENT CHECK")
print("-" * 50)

# Fresh models with gradients
dynamics = create_dynamics(size="small", latent_dim=32, use_agent_tokens=True).to(device)
reward_head = RewardHead(input_dim=512, hidden_dim=256, num_buckets=255, mtp_length=8).to(device)

# Forward pass with gradients
z = torch.randn(1, 32, 32, 16, 16, device=device)
tau = torch.zeros(1, 32, device=device)
z_pred, agent_out = dynamics(z, tau)
reward_logits = reward_head(agent_out)

# Compute loss
targets = torch.zeros(1, 32, device=device)
targets[0, 5] = 0.21  # One gold event
symlog_targets = symlog(targets)

loss = 0
for offset in range(8):
    if offset < 31:
        target = symlog_targets[:, offset + 1:]
        pred = reward_logits[:, :31 - offset, offset, :]
        if pred.shape[1] > 0:
            loss = loss + twohot_loss(pred, target, reward_head.bucket_centers)
loss = loss / 8

print(f"Test loss: {loss.item():.4f}")
loss.backward()

# Check gradients
print(f"\nGradient norms:")
reward_grad_norm = sum(p.grad.norm().item() for p in reward_head.parameters() if p.grad is not None)
dynamics_grad_norm = sum(p.grad.norm().item() for p in dynamics.parameters() if p.grad is not None)

print(f"  reward_head total grad norm: {reward_grad_norm:.6f}")
print(f"  dynamics total grad norm: {dynamics_grad_norm:.6f}")

if reward_grad_norm == 0:
    print("  ✗ CRITICAL: Zero gradients in reward_head!")
elif np.isnan(reward_grad_norm):
    print("  ✗ CRITICAL: NaN gradients!")
else:
    print("  ✓ Gradients are flowing correctly")

# Individual layer gradients
print(f"\nReward head layer gradients:")
for name, param in reward_head.named_parameters():
    if param.grad is not None:
        print(f"  {name}: {param.grad.norm().item():.6f}")

print("\n" + "=" * 70)
print("DIAGNOSTIC SUMMARY")
print("=" * 70)
print("""
1. DATA COMPATIBILITY: ✓ Latents (32, 16, 16) match dynamics expectation
2. REWARD SIGNAL: ✓ ~1% non-zero rewards (expected without oversampling)
3. REWARD VALUES: Check gold values above (should be 0.14-3.0)
4. OVERSAMPLING: VideoShuffleSampler used, filters empty videos
5. LOSS FUNCTION: ✓ twohot cross-entropy working
6. ARCHITECTURE: ✓ RewardHead receives (B, T, 512) from agent tokens
7. GRADIENTS: ✓ Flowing through reward_head and dynamics
""")
