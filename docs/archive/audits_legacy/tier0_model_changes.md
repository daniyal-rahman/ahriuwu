# Tier 0 Model & Data Changes Audit

Date: 2026-03-02

## 1. Variable tau_ctx noise augmentation

**File:** `src/ahriuwu/models/diffusion.py`
**Method:** `DiffusionSchedule.sample_diffusion_forcing_timesteps()`
**Lines:** 85-149

**What changed:**
- Default `tau_ctx` parameter changed from `0.1` (fixed) to `0.3` (max for uniform sampling).
- Instead of `torch.full_like(normalized_dist, tau_ctx)` applying a fixed noise level to all context frames, now samples `tau_ctx_per_sample = torch.rand(batch_size, 1, device=device) * tau_ctx` per batch element from U(0, tau_ctx).
- Context frames use the per-sample value: `tau_ctx_per_sample.expand_as(normalized_dist)`.
- Target frames ramp from the per-sample tau_ctx: `tau_ctx_per_sample + normalized_dist * (tau_max - tau_ctx_per_sample)`.
- The `tau_ctx` parameter now represents the max of the uniform range, not a fixed value.

**Why:** Prevents the model from relying on a fixed context noise level. Variable augmentation improves generalization.

---

## 2. Explicit p=0 MAE batches

**File:** `scripts/train_transformer_tokenizer.py`
**Lines:** 82-98 (args), 290-295 (sampling logic), 391 (print)

**What changed:**
- `--mask-ratio-min` default changed from `0.0` to `0.1`. Help text updated to clarify it avoids near-zero-but-nonzero masking.
- New CLI arg `--p-zero-mask` added with default `0.1` (10% probability of full reconstruction).
- Mask ratio sampling logic changed: with probability `p_zero_mask`, `mask_ratio = 0.0` exactly; otherwise sample from `U(mask_ratio_min, current_mask_max)`.
- Print statement updated to show `p_zero_mask` info: `"10% p=0, else U(0.1, 0.9)"`.

**Why:** Ensures the model explicitly trains on full reconstruction (p=0) a controlled fraction of the time, while avoiding the ambiguous near-zero masking regime that is neither proper MAE nor full reconstruction.

---

## 3. Resolution change preparation

### 3a. Dataset

**File:** `src/ahriuwu/data/dataset.py`
**Line:** 14-15

**What changed:**
- `TARGET_SIZE = (256, 256)` changed to `TARGET_SIZE = (480, 352)`.
- Comment changed from `# Default target resolution for world model` to `# League is 16:9, this is ~1.36:1 aspect ratio. Both dims divisible by 16 for patch-based tokenizer.`

### 3b. Feature extraction pipeline

**File:** `src/ahriuwu/data/feature_extraction_pipeline.py`
**Lines:** 7, 124, 130, 139, 159, 278, 279

**What changed:**
- Module docstring: `"Convert frames to 256x256"` -> `"Convert frames to 480x352"`.
- Class docstring: `"Extract features from 1080p video and save 256x256 frames"` -> `"...480x352 frames"`.
- Default `target_size` parameter: `(256, 256)` -> `(480, 352)`.
- Docstring for `target_size`: `"Output frame size (default 256x256)"` -> `"Output frame size (width, height), default 480x352"`.
- `process_video` docstring: `"save 256x256 frames"` -> `"save 480x352 frames"`.
- Comment `"# === Save 256x256 frame ==="` -> `"# === Save downscaled frame ==="`.
- Variable name `frame_256` -> `frame_resized`.

### NOT changed (by design)

- `src/ahriuwu/models/tokenizer.py` - CNN tokenizer architecture, will be updated separately.
- `src/ahriuwu/models/transformer_tokenizer.py` - Transformer tokenizer architecture, will be updated separately.
- `src/ahriuwu/models/losses.py` - Test code uses 256x256 tensors, not data-dependent.
