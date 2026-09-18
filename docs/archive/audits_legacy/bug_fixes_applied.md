# Bug Fixes Applied

**Date:** 2026-03-02
**Applied by:** Claude Opus 4.6 (automated bug fixing from verification report)

---

## Fix 1: Reward misalignment — start_idx vs start_frame
**File:** `scripts/train_agent_finetune.py`
**Bug:** Used `start_idx` (array position) to index rewards instead of `start_frame` (actual frame number), causing wrong rewards when packed arrays have gaps.

**Change:** Replaced `start_idx = seq_info['start_idx']` with `start_frame = seq_info['start_frame']` and updated all downstream indexing to use `start_frame`.

---

## Fix 2: Agent finetuning drops actions
**File:** `scripts/train_agent_finetune.py`
**Bug:** `load_pretrained_dynamics()` always created dynamics without `use_actions=True`, silently discarding action embeddings from checkpoint.

**Change:** Auto-detect `use_actions` by scanning checkpoint keys for `"action_embed"`, and pass the detected value to `create_dynamics()`.

---

## Fix 3: Phase 2 dynamics loss uses plain MSE
**File:** `scripts/train_agent_finetune.py`
**Bug:** Phase 2 used `F.mse_loss(z_pred, z_target)` instead of the ramp-weighted x-prediction loss used in Phase 1.

**Change:** Added import of `x_prediction_loss` from `ahriuwu.models.diffusion` and replaced plain MSE with `x_prediction_loss(z_pred, z_target, tau, use_ramp_weight=True)`.

---

## Fix 5: Decoder causal leak
**File:** `src/ahriuwu/models/transformer_tokenizer.py`
**Bug:** In `create_decoder_mask`, patches at frame `t` could attend to latents from ALL frames (`range(num_frames)`) instead of only past/current frames.

**Change:** Changed `for t2 in range(num_frames):` to `for t2 in range(t + 1):` to enforce causal masking.

---

## Fix 6: eval_dynamics ignores arch settings
**File:** `scripts/eval_dynamics.py`
**Bug:** `load_dynamics()` created model with default architecture settings, ignoring checkpoint args like `use_qk_norm`, `soft_cap`, `num_register_tokens`, `num_kv_heads`. With `strict=False`, mismatched keys got random init.

**Change:** Read architecture args from `checkpoint.get("args", {})` and pass `use_qk_norm`, `soft_cap`, `num_register_tokens`, `num_kv_heads` to `create_dynamics()`.

---

## Fix 7: Euler sampler uses fresh noise per step
**Files:** `src/ahriuwu/models/diffusion.py`, `scripts/eval_dynamics.py`
**Bug:** Both `DiffusionSchedule.sample()` and `rollout_predictions()` sampled fresh `torch.randn_like(z_t)` noise at every Euler step, making sampling stochastic instead of following the ODE trajectory.

**Change:** Save initial noise as `z_noise = z_t.clone()` at the start and reuse it in all Euler steps: `z_t = (1 - next_tau) * z_0_pred + next_tau * z_noise`.

---

## Fix 9: pretokenize uses fp16 while training uses bf16
**File:** `scripts/pretokenize_frames.py`
**Bug:** Autocast used `torch.float16` unconditionally, while training scripts use `torch.bfloat16` on CUDA.

**Change:** Select dtype based on device: `torch.bfloat16` for CUDA, `torch.float16` otherwise. Matches training script convention.

---

## Fix 10: RunningRMS device mismatch on resume
**File:** `src/ahriuwu/models/returns.py`
**Bug:** `load_state_dict()` creates RMS as CPU tensor. On next `update()` with GPU tensor, the EMA computation crashes with device mismatch.

**Change:** Added device check in `update()`: if `self.rms.device != value_sq.device`, move rms to the correct device before the EMA update.

---

## Fix 11: Feature pipeline calls nonexistent method
**File:** `src/ahriuwu/data/feature_extraction_pipeline.py`
**Bug:** Called `movement_tracker.detect_ability_usage()` but `GarenHUDTracker` doesn't have that method — it's on `AbilityBarDetector`.

**Change:** Added `AbilityBarDetector` to imports, created an instance in `process_video()`, and changed the call to use the correct instance.

---

## Fix 12: LatentSequenceDataset missing reward_indices
**File:** `src/ahriuwu/data/dataset.py`
**Bug:** `LatentSequenceDataset.__init__()` never initialized `self.reward_indices` or called `_precompute_reward_indices()`, unlike `PackedLatentSequenceDataset`.

**Change:** Added `self.reward_indices: list[int] = []` initialization and conditional `self._precompute_reward_indices()` call when `load_rewards=True`.

---

## Bugs NOT fixed (by design)

### Bug 4: Action direction off by half bin
**Status:** PARTIAL — inconsistency between `angle_to_direction` and `angle_to_slice`, but training is internally consistent. Only affects deployment inference. Requires design decision on which convention to standardize.

### Bug 8: train_tokenizer.py wrong argument order
**Status:** FALSE_POSITIVE — already fixed by tier0 changes.

---

## Summary

| # | Bug | Fix Applied |
|---|-----|-------------|
| 1 | Reward misalignment | start_idx → start_frame |
| 2 | Agent finetuning drops actions | Auto-detect use_actions from checkpoint |
| 3 | Phase 2 plain MSE | x_prediction_loss with ramp weight |
| 5 | Decoder causal leak | range(num_frames) → range(t + 1) |
| 6 | eval_dynamics ignores arch | Read args from checkpoint |
| 7 | Euler sampler fresh noise | Reuse initial noise |
| 9 | pretokenize fp16 vs bf16 | Use bf16 on CUDA |
| 10 | RunningRMS device mismatch | Device check in update() |
| 11 | Feature pipeline method | Use AbilityBarDetector |
| 12 | Missing reward_indices | Add init + precompute call |

**Total: 10 fixes applied, 1 partial (deferred), 1 false positive (no action needed)**
