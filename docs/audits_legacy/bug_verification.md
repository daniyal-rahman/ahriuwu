# Bug Verification Report

**Date:** 2026-03-02
**Verifier:** Claude Opus 4.6 (code verification against actual source)

---

## TRAINING-BREAKING (silently wrong results)

### Bug 1: Reward misalignment — start_idx vs start_frame
**Status: CONFIRMED**

**File:** `scripts/train_agent_finetune.py`, line 335, 365-366

**Evidence:** `ReplayDataset.__getitem__()` uses `start_idx` (the array position in the packed .npz) to index into `reward_data` (a list built sequentially from features.json frames). Meanwhile, `PackedLatentSequenceDataset._get_actions()` correctly uses `start_frame` (the actual frame number). When the packed array has gaps or doesn't start at frame 0, `start_idx != start_frame`, and rewards are fetched for the wrong frames.

```python
# Line 335: uses start_idx (array position)
start_idx = seq_info['start_idx']
# Line 365-366: indexes reward list with array position instead of frame number
rewards = video_rewards[start_idx:end_idx]
```

**Fix:** Use `start_frame = seq_info['start_frame']` and index with `start_frame`.

---

### Bug 2: Agent finetuning drops actions
**Status: CONFIRMED**

**Files:** `scripts/train_agent_finetune.py`, lines 409-420, 538

**Evidence:** Two separate issues:
1. `load_pretrained_dynamics()` creates dynamics without `use_actions=True` (line 409-420). If the pretrained checkpoint had action embeddings, they're logged as "Skipped" and lost.
2. `train_epoch()` forward call at line 538 doesn't pass `actions=` argument:
   ```python
   z_pred, agent_out = dynamics(z_noisy, tau, independent_frames=use_independent)
   ```

**Fix:** (1) Auto-detect `use_actions` from checkpoint keys. (2) Pass actions to forward call.

---

### Bug 3: Phase 2 dynamics loss uses plain MSE
**Status: CONFIRMED**

**File:** `scripts/train_agent_finetune.py`, line 541

**Evidence:** Phase 1 uses `x_prediction_loss(z_pred, z_0, tau, use_ramp_weight=True)` (train_dynamics.py). Phase 2 uses:
```python
dynamics_loss = F.mse_loss(z_pred, z_target)
```
This ignores the per-timestep tau weighting entirely. The model was trained to focus on high-signal regions (low tau); Phase 2 equally weights all noise levels.

**Fix:** Replace with `x_prediction_loss(z_pred, z_target, tau, use_ramp_weight=True)`.

---

### Bug 4: Action direction off by half bin
**Status: PARTIAL**

**Files:** `actions.py:73` vs `keylog_extractor.py:1313`

**Evidence:** Two different binning conventions exist:
- `ActionSpace.angle_to_direction()`: centered buckets `int((angle + 10) / 20) % 18`
- `MousePositionEstimator.angle_to_slice()`: edge-aligned `int(angle / self.slice_angle) % self.num_slices`

However, during training, only `angle_to_slice` is used (via features.json → dataset → model). The model trains on edge-aligned bins consistently. The `angle_to_direction` function is not called during training.

**Impact:** Internally consistent during training. Bug manifests at inference/deployment when interpreting model outputs via `ActionSpace`. Less severe than claimed — not "training-breaking", but a real inconsistency that would cause 10° errors at deployment.

---

### Bug 5: Decoder causal leak
**Status: CONFIRMED**

**File:** `src/ahriuwu/models/transformer_tokenizer.py`, lines 425-429

**Evidence:** The `create_decoder_mask` function has patches attending to ALL latents:
```python
for t2 in range(num_frames):  # Should be range(t + 1)
    src_latent_start = t2 * tokens_per_frame + num_patches
    src_latent_end = src_latent_start + num_latents
    mask[patch_start:patch_end, src_latent_start:src_latent_end] = True
```

For multi-frame (T>1) training, patches at frame t can see latents from future frames. For T=1 (current usage), this has no effect.

**Fix:** Change `range(num_frames)` to `range(t + 1)`.

---

### Bug 6: eval_dynamics ignores arch settings
**Status: CONFIRMED**

**File:** `scripts/eval_dynamics.py`, line 291

**Evidence:** `load_dynamics()` creates model with defaults only:
```python
model = create_dynamics(model_size, latent_dim=latent_dim, use_actions=use_actions)
```
Does not pass `use_qk_norm`, `soft_cap`, `num_register_tokens`, `num_kv_heads`. With `strict=False` on line 292, mismatched keys silently get random init. The checkpoint's `args` dict (which contains all settings) is never read for architecture construction.

**Fix:** Read architecture args from `checkpoint.get("args", {})` and pass to `create_dynamics`.

---

### Bug 7: Euler sampler uses fresh noise per step
**Status: CONFIRMED (but lower severity than claimed)**

**File:** `src/ahriuwu/models/diffusion.py`, line 193

**Evidence:**
```python
z_t = (1 - next_tau) * z_0_pred + next_tau * torch.randn_like(z_t)
```
Fresh noise is sampled at every step instead of re-using the initial noise or using the ODE velocity formulation. This makes sampling stochastic rather than deterministic.

**Impact:** Only affects sampling/inference quality, NOT training. The sampler is called in `eval_dynamics.py` and `DiffusionSchedule.sample()`, not during training. Training correctly uses `add_noise()` with properly generated noise. The claim of "training-breaking" is overstated — this only makes eval predictions noisier.

**Fix:** Save initial noise and re-use: `z_t = (1 - next_tau) * z_0_pred + next_tau * z_noise_initial`

---

## CRASH BUGS

### Bug 8: train_tokenizer.py wrong argument order
**Status: FALSE_POSITIVE**

**Files:** `scripts/train_tokenizer.py`, lines 414-416, 436, 399-401

**Evidence:** The tier0 changes (documented in `docs/audits/tier0_training_changes.md`) already fixed this. The current code has correct argument order at all call sites:
- `train_epoch()` call (line 414): passes `scheduler, loss_fn, rms_trackers` in correct order
- `save_checkpoint()` calls (line 436): passes `scheduler, rms_trackers` correctly
- `load_checkpoint()` call (line 399): passes `scheduler, rms_trackers` correctly

The function signatures and call sites match perfectly in the current codebase.

---

### Bug 9: pretokenize uses fp16 while training uses bf16
**Status: PARTIAL**

**File:** `scripts/pretokenize_frames.py`, line 210

**Evidence:** The autocast uses `torch.float16`:
```python
with torch.amp.autocast(device_type=device.split(":")[0], dtype=torch.float16):
```
Training scripts use `torch.bfloat16` on CUDA. The mismatch is real but severity is lower than claimed:

1. Latents are saved as `np.float16` (line 227) regardless, so final precision is the same
2. The tanh bottleneck bounds values to [-1, 1], so fp16 overflow/NaN is extremely unlikely
3. Intermediate computation differences between fp16 and bf16 would produce slightly different latent values, but both are quantized to float16 on save

**Impact:** Very unlikely to crash. May produce slightly different latent values vs training. Worth fixing for consistency.

**Fix:** Use `amp_dtype = torch.bfloat16 if device.split(":")[0] == "cuda" else torch.float16`

---

### Bug 10: RunningRMS device mismatch on resume
**Status: CONFIRMED**

**File:** `src/ahriuwu/models/returns.py`, line 250

**Evidence:**
```python
self.rms = torch.tensor(rms_val) if rms_val is not None else None
```
`torch.tensor(rms_val)` creates a CPU tensor. On next `update()` with a GPU tensor:
```python
self.rms = self.decay * self.rms + (1 - self.decay) * value_sq  # CPU * GPU = crash
```

**Fix:** Move rms to correct device in `update()`:
```python
if self.rms is not None and self.rms.device != value.device:
    self.rms = self.rms.to(value.device)
```

---

### Bug 11: Feature pipeline calls nonexistent method
**Status: CONFIRMED**

**File:** `src/ahriuwu/data/feature_extraction_pipeline.py`, line 248

**Evidence:** `movement_tracker` is `GarenHUDTracker` (created at line 201). `GarenHUDTracker` (keylog_extractor.py:132) only has `detect_movement()` and `infer_wasd()`. The method `detect_ability_usage()` is defined on `AbilityBarDetector` (keylog_extractor.py:314). This will raise `AttributeError`.

**Fix:** Create an `AbilityBarDetector` instance and call its method instead.

---

### Bug 12: RewardMixtureSampler crash — reward_indices never initialized
**Status: CONFIRMED (but low practical severity)**

**File:** `src/ahriuwu/data/dataset.py`, lines 627-667

**Evidence:** `LatentSequenceDataset.__init__()` never initializes `self.reward_indices` or calls `self._precompute_reward_indices()`. Compare with `PackedLatentSequenceDataset.__init__()` (lines 373-375) which correctly does both.

However, `RewardMixtureSampler` is only used with `PackedLatentSequenceDataset` in practice (train_agent_finetune.py:787). `LatentSequenceDataset` is used in eval_dynamics.py without a sampler. So this doesn't crash in current usage.

**Fix:** Add `self.reward_indices: list[int] = []` and conditional `_precompute_reward_indices()` call to `LatentSequenceDataset.__init__()`.

---

## Summary

| # | Bug | Verdict | Severity |
|---|-----|---------|----------|
| 1 | Reward misalignment (start_idx vs start_frame) | **CONFIRMED** | Training-breaking |
| 2 | Agent finetuning drops actions | **CONFIRMED** | Training-breaking |
| 3 | Phase 2 dynamics loss uses plain MSE | **CONFIRMED** | Training-breaking |
| 4 | Action direction off by half bin | **PARTIAL** | Deployment issue, not training-breaking |
| 5 | Decoder causal leak | **CONFIRMED** | Training-breaking for T>1 |
| 6 | eval_dynamics ignores arch settings | **CONFIRMED** | Eval-breaking (not training) |
| 7 | Euler sampler fresh noise | **CONFIRMED** | Eval quality only (not training) |
| 8 | train_tokenizer.py wrong args | **FALSE_POSITIVE** | Already fixed by tier0 |
| 9 | pretokenize fp16 vs bf16 | **PARTIAL** | Consistency issue, unlikely to crash |
| 10 | RunningRMS device mismatch | **CONFIRMED** | Crash on resume |
| 11 | Feature pipeline nonexistent method | **CONFIRMED** | Crash at runtime |
| 12 | RewardMixtureSampler reward_indices | **CONFIRMED** | API bug, not hit in practice |

**Confirmed critical (fix immediately):** 1, 2, 3, 5, 10, 11
**Confirmed important (fix soon):** 6, 7, 9, 12
**Partial (real but less severe):** 4, 9
**False positive:** 8
