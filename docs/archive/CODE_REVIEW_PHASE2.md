# Phase 2 Training Pipeline Code Review

Deep-dive code review of the ML training pipeline, focusing on `train_agent_finetune.py`, `heads.py`, `returns.py`, and `dataset.py`.

---

## 1. MTP (Multi-Token Prediction) Loss Implementation

### Location: `train_agent_finetune.py:506-541`

### Observation: Off-by-one confusion in MTP target indexing

```python
# MTP reward loss: predict rewards at t+1, t+2, ..., t+L
for offset in range(L):
    if offset < T - 1:
        # Target is reward at t + offset + 1
        target_idx = min(offset + 1, T - 1)
        target = reward_targets[:, target_idx:]  # (B, T - target_idx)
        pred = reward_logits[:, :T - target_idx, offset, :]
```

**Potential Issue**: The slicing logic is confusing and potentially incorrect:

1. `target_idx = min(offset + 1, T - 1)` - this clamps the start of the target slice
2. `target = reward_targets[:, target_idx:]` - takes rewards from `target_idx` to end
3. `pred = reward_logits[:, :T - target_idx, offset, :]` - takes predictions from start to `T - target_idx`

For `offset=0`: predicting reward at t+1 from state at t (correct)
For `offset=7`: predicting reward at t+8 from state at t

**The issue**: When `offset >= T-1`, `target_idx` is clamped to `T-1`, causing the same target to be used for multiple offsets. The `if offset < T - 1` check prevents this, but this means later MTP heads (offset 7 with T=32) never train on sequences shorter than 9 frames.

**Deeper issue**: The targets are sliced as `[:, target_idx:]` but predictions as `[:, :T - target_idx, ...]`. This means:
- For offset=0: target shape (B, T-1), pred shape (B, T-1, buckets) ✓
- For offset=1: target shape (B, T-2), pred shape (B, T-2, buckets) ✓

This seems correct but the naming is misleading - `target_idx` isn't an index, it's a slice offset.

---

## 2. Reward vs BC Loss Asymmetry

### Location: `train_agent_finetune.py:522, 541`

```python
reward_loss = reward_loss / L  # Divided by L
bc_loss = bc_loss / L          # Divided by L
```

**Observation**: Both losses are divided by `L` (MTP length), but they contribute differently:
- `reward_loss` uses `twohot_loss` which calls `.mean()` internally
- `bc_loss` uses `cross_entropy` which also averages

So both are doubly-averaged: once by the loss function, once by `/L`. This is fine as long as it's consistent, but means the effective loss weight is `1/L` of what you might expect if thinking in per-sample terms.

---

## 3. RunningRMS Loss Normalization

### Location: `returns.py:209-240`

```python
class RunningRMS:
    def update(self, value: torch.Tensor) -> torch.Tensor:
        value_sq = value.detach() ** 2
        if self.rms is None:
            self.rms = value_sq
        else:
            self.rms = self.decay * self.rms + (1 - self.decay) * value_sq
        return value / (torch.sqrt(self.rms) + 1e-8)
```

**Potential Issues**:

1. **First batch problem**: On the first batch, `self.rms = value_sq`, so the normalized value is `value / (|value| + 1e-8) ≈ sign(value)`. This means the first batch loss contribution is always ~1.0 regardless of actual loss magnitude.

2. **Device mismatch risk**: `self.rms` is a Python float or tensor without explicit device placement. If model moves between devices, this could cause issues.

3. **No state saving**: The RMS state isn't saved in checkpoints, so resuming training resets the normalization. This could cause training instability on resume.

---

## 4. Twohot Bucket Range Mismatch

### Location: `heads.py:29-30`, `returns.py:64-65`

```python
# RewardHead defaults:
bucket_low: float = -20.0,
bucket_high: float = 20.0,

# twohot_encode:
x_clamped = x.clamp(low, high)
```

**Observation**: The bucket range is [-20, 20] in symlog space. Since symlog(x) = sign(x) * log(1 + |x|):
- symlog(20) ≈ 3.0
- symlog(-20) ≈ -3.0

But the bucket centers go from -20 to +20, not -3 to +3. This means the model can represent symlog values up to ±20, which corresponds to original values of:
- symexp(20) = exp(20) - 1 ≈ 485 million

**This is WAY larger than needed**. Your rewards are:
- Gold: 0-400 gold * 0.01 = 0-4.0 → symlog(4) ≈ 1.6
- Death: -10 → symlog(-10) ≈ -2.4

Most of the 255 buckets are wasted on values that never occur. A range of [-5, 5] would be more appropriate and give finer granularity.

---

## 5. ValueHead Zero Initialization Warning

### Location: `heads.py:242-244`

```python
# Initialize output weights to zero (paper recommendation)
nn.init.zeros_(self.mlp[-1].weight)
nn.init.zeros_(self.mlp[-1].bias)
```

**Observation**: This is the ValueHead (Phase 3), not currently used. However, note the comment says "paper recommendation" - but this causes the same gradient flow issue that was fixed in RewardHead. If you use ValueHead in Phase 3, it will initially output all zeros and have no gradient through the output layer.

The RewardHead was fixed to use `std=0.01` init instead - ValueHead should match.

---

## 6. ReplayDataset Reward Loading Duplication

### Location: `train_agent_finetune.py:263-282`

```python
def _load_rewards(self):
    """Load reward data from features.json files."""
    for video_id in video_ids:
        features_path = self.features_dir / video_id / "features.json"
        ...
        for frame in frames:
            gold = frame.get("gold_gained", 0) * self.gold_scale
            death = self.death_penalty if frame.get("is_dead", False) else 0.0
            rewards.append(gold + death)
```

**Issue**: The code checks for `is_dead` field, but looking at the features.json structure from earlier analysis:

```python
Frame keys: ['frame_idx', 'timestamp_ms', 'movement_dx', 'movement_dy',
             'movement_slice', 'movement_confidence', 'ability_q', ...,
             'gold_gained', 'health_bar_x', 'health_bar_y']
```

**There is no `is_dead` field!** The death detection logic exists in `LatentSequenceDataset._get_rewards()` using health bar presence, but `ReplayDataset._load_rewards()` just checks a non-existent field.

This means **death penalties are never applied** in the current training pipeline.

---

## 7. Dataset Index Mismatch Risk

### Location: `train_agent_finetune.py:289-341`

```python
def __getitem__(self, idx):
    # Get data from PackedLatentSequenceDataset
    data = self.latent_dataset[idx]

    # Get sequence info to look up rewards
    seq_info = self.latent_dataset.sequences[idx]
    video_id = seq_info['video_id']
    start_idx = seq_info['start_idx']
```

**Observation**: The code assumes `idx` maps directly between:
1. `self.latent_dataset[idx]` (data access)
2. `self.latent_dataset.sequences[idx]` (metadata access)

This works correctly, but there's a subtle issue: `start_idx` here is the index into the packed array, NOT the frame number. But then:

```python
# Get rewards from our loaded reward data
if video_id in self.reward_data:
    video_rewards = self.reward_data[video_id]
    end_idx = min(start_idx + self.seq_len, len(video_rewards))
    rewards = video_rewards[start_idx:end_idx]
```

**The reward lookup uses `start_idx` as a frame index into `video_rewards`**, but `start_idx` from `PackedLatentSequenceDataset` is actually `start_frame`:

```python
# From PackedLatentSequenceDataset._index_packed_latents():
self.sequences.append({
    "video_id": video_id,
    "start_frame": start_frame,  # <-- This is the frame number
    "start_idx": start_idx,       # <-- This is the array index
})
```

Wait, looking more closely, `seq_info['start_idx']` is used but the dict has both `start_frame` and `start_idx`. The code uses `start_idx` which is the array index into the packed numpy array, NOT the frame number.

**BUG**: Should be using `seq_info['start_frame']` for reward lookup, not `seq_info['start_idx']`.

Actually wait - let me re-read:
```python
start_idx = seq_info['start_idx']
...
rewards = video_rewards[start_idx:end_idx]
```

If `start_idx` is the array index and frame indices are contiguous starting from some offset, this could work. But if frame indices have gaps (e.g., frames 0-1000, then 2000-3000), the array index won't match the frame number.

Looking at the packed file creation, frames should be contiguous within each file, so `start_idx` equals `start_frame - first_frame_of_video`. This is fragile.

---

## 8. Sampler Replacement Sampling

### Location: `dataset.py:1022-1028`

```python
for _ in range(len(all_indices)):
    if reward_indices and self.rng.random() < 0.5:
        # Sample from reward sequences (with replacement)
        yield self.rng.choice(reward_indices)
    else:
        # Sample from all sequences
        yield self.rng.choice(all_indices)
```

**Observation**: Both branches use `random.choice()` which samples WITH REPLACEMENT. This means:
- Some sequences may be seen multiple times per epoch
- Some sequences may never be seen in a given epoch
- Epoch length is maintained, but coverage is not guaranteed

This matches the DreamerV4 paper's mixture sampling strategy, but means "epoch" is not a true epoch (full data pass). This is intentional but worth documenting.

---

## 9. Dynamics Loss Without Gradient Stoppage

### Location: `train_agent_finetune.py:499-500`

```python
# Forward through dynamics with agent tokens
z_pred, agent_out = dynamics(z_noisy, tau)

# Dynamics loss (x-prediction)
dynamics_loss = F.mse_loss(z_pred, z_target)
```

**Observation**: The dynamics loss flows gradients through the ENTIRE dynamics model including the new agent token components. This is probably intentional (joint training), but means:

1. Agent token gradients affect the pretrained dynamics backbone
2. The dynamics backbone is being fine-tuned, not frozen

If the intent was to only train the agent heads while keeping dynamics frozen (transfer learning), this needs `z_pred.detach()` or freezing dynamics parameters.

---

## 10. Agent Token Temporal Masking

### Location: `dynamics.py:1084-1094`

```python
# Process through agent blocks
# Agent tokens attend to z tokens (x includes registers), but z tokens don't see agent tokens
for agent_block in self.agent_blocks:
    agent_tokens = agent_block(agent_tokens, x)
```

**Observation**: The comment says "z tokens don't see agent tokens" - this is architectural (agent tokens attend to z, not vice versa). However, looking at `AgentTokenBlock`:

The agent tokens use cross-attention to z tokens, but there's no **causal masking** mentioned. This means agent token at time t can attend to z tokens at ALL times (0 to T-1), not just times 0 to t.

For reward prediction this might be okay (rewards are known for all times in the sequence), but for policy prediction this could leak future information - the policy at time t shouldn't see z_t+1.

**Need to verify**: Does `AgentTokenBlock` apply causal masking in its cross-attention?

---

## 11. Independent Frame Ratio Unused

### Location: `train_agent_finetune.py:193-194`

```python
parser.add_argument(
    "--independent-frame-ratio",
    type=float,
    default=0.3,
    help="Ratio of batches using independent frame mode",
)
```

But in `train_epoch()`, this argument is never used:

```python
# Forward through dynamics with agent tokens
z_pred, agent_out = dynamics(z_noisy, tau)  # No independent_frames argument!
```

The dynamics model supports `independent_frames=True` but the training script never passes it. This means the 30% independent frame training (DreamerV4 Section 3.2) is not being applied.

---

## 12. GradScaler Device Type Handling

### Location: `train_agent_finetune.py:756-757`

```python
device_type = args.device.split(":")[0]
scaler = GradScaler(device_type)
```

**Observation**: If device is "cuda:1", `device_type` becomes "cuda" which is correct. But if device is "mps", the scaler is created with "mps" which may not be fully supported.

Also, `GradScaler` with MPS backend has known issues. The code should probably disable AMP entirely on MPS:

```python
# Current code:
dtype = torch.bfloat16 if device_type == "cuda" else torch.float16
```

Using float16 on MPS with GradScaler can cause NaN issues.

---

## 13. Optimizer Parameter Groups

### Location: `train_agent_finetune.py:747-753`

```python
all_params = (
    list(dynamics.parameters()) +
    list(reward_head.parameters()) +
    list(policy_head.parameters())
)
optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=0.01)
```

**Observation**: All parameters use the same learning rate. This might not be ideal:

1. **Pretrained dynamics** - should potentially use lower LR (fine-tuning)
2. **New agent components** - can use higher LR (training from scratch)
3. **Head networks** - may need different LR than backbone

Consider using parameter groups:
```python
param_groups = [
    {"params": dynamics.parameters(), "lr": args.lr * 0.1},  # Fine-tune
    {"params": reward_head.parameters(), "lr": args.lr},
    {"params": policy_head.parameters(), "lr": args.lr},
]
```

---

## 14. Twohot Loss Numerical Stability

### Location: `returns.py:121-125`

```python
def twohot_loss(logits, targets, bucket_centers):
    target_twohot = twohot_encode(targets, bucket_centers)
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(target_twohot * log_probs).sum(dim=-1)
    return loss.mean()
```

**Observation**: This is a soft cross-entropy with two-hot targets. The implementation is correct, but there's a numerical consideration:

`log_softmax` can produce `-inf` for very low logits. When multiplied by a zero in `target_twohot`, this gives `0 * (-inf) = nan`.

PyTorch handles this case correctly (0 * -inf = 0 in autograd), but it's worth being aware of. The alternative would be:
```python
loss = F.cross_entropy(logits, target_twohot, reduction='mean')
```
which handles this internally.

---

## Summary of Critical Issues

| Severity | Issue | Location |
|----------|-------|----------|
| **HIGH** | Death penalties never applied (missing `is_dead` field) | `train_agent_finetune.py:279` |
| **HIGH** | Possible reward index mismatch (`start_idx` vs `start_frame`) | `train_agent_finetune.py:329` |
| **MEDIUM** | Independent frame training not implemented | `train_agent_finetune.py:497` |
| **MEDIUM** | ValueHead zero-init will break gradient flow in Phase 3 | `heads.py:243` |
| **MEDIUM** | RunningRMS state not saved in checkpoint | `returns.py:209` |
| **LOW** | Bucket range inefficient (-20 to 20 in symlog space) | `heads.py:29-30` |
| **LOW** | All params use same LR (no fine-tuning rate) | `train_agent_finetune.py:753` |
| **INFO** | Sampler uses replacement (not true epochs) | `dataset.py:1025` |

---

## Recommendations

1. **Verify reward indexing**: Add assertion that `start_idx == start_frame` or fix to use `start_frame`
2. **Implement death detection**: Either add `is_dead` to features.json during extraction, or use the health bar logic from `LatentSequenceDataset`
3. **Add independent frame training**: Pass `independent_frames=random.random() < 0.3` to dynamics forward
4. **Save RMS state**: Add `rms_trackers` state to checkpoint dict
5. **Consider bucket range**: Reduce to [-5, 5] for better resolution on actual reward values
