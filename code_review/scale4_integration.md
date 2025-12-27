# DreamerV4 Implementation Integration Review

**Review Date:** 2025-12-27
**Reviewer:** Claude (Automated Code Review)
**Scope:** File wiring and integration issues between tokenizer, dynamics, datasets, and training/evaluation scripts

---

## Executive Summary

This review examines the integration between key components of the DreamerV4 world model implementation. Several issues were identified ranging from **Critical** (will cause runtime failures) to **Low** (code quality concerns). The most significant issues involve checkpoint loading during evaluation and latent dimension configuration mismatches.

---

## 1. Tokenizer -> Dynamics Data Flow

### 1.1 Latent Dimension Mismatch Risk

**Severity:** Medium
**Files Involved:**
- `/Users/danirahman/Repos/ahriuwu/src/ahriuwu/models/tokenizer.py` (lines 201-206)
- `/Users/danirahman/Repos/ahriuwu/src/ahriuwu/models/dynamics.py` (lines 421-463)
- `/Users/danirahman/Repos/ahriuwu/scripts/train_dynamics.py` (lines 56-60)

**Description:**
The tokenizer and dynamics model have independent `latent_dim` configurations that must be manually synchronized. The `create_tokenizer()` function encodes `latent_dim` into the model size presets, while `create_dynamics()` accepts `latent_dim` as an explicit parameter defaulting to 256.

```python
# tokenizer.py - latent_dim embedded in size presets
configs = {
    "tiny": {"latent_dim": 128, "base_channels": 32},
    "small": {"latent_dim": 256, "base_channels": 64},  # 256
    "medium": {"latent_dim": 384, "base_channels": 96},  # 384
    "large": {"latent_dim": 512, "base_channels": 128},  # 512
}

# dynamics.py - latent_dim is a separate parameter
def create_dynamics(size: str = "small", latent_dim: int = 256):
```

**How to Reproduce:**
1. Train tokenizer with `--model-size medium` (latent_dim=384)
2. Run `pretokenize_frames.py` with this tokenizer
3. Train dynamics with default `--latent-dim 256`
4. The dynamics model will fail to process the 384-channel latents

**Recommendation:**
Either extract `latent_dim` from tokenizer checkpoint when training dynamics, or add validation that checks the actual latent file dimensions against `--latent-dim`.

---

### 1.2 Tokenizer Output Normalization

**Severity:** Low (Informational)
**Files Involved:**
- `/Users/danirahman/Repos/ahriuwu/src/ahriuwu/models/tokenizer.py` (lines 67-99)
- `/Users/danirahman/Repos/ahriuwu/src/ahriuwu/models/dynamics.py`

**Description:**
The tokenizer encoder outputs raw feature maps without explicit normalization (no LayerNorm, no clamping). The dynamics model receives these raw values and projects them via a Linear layer. This design is acceptable but may cause training instability if latent magnitudes grow large.

The encoder uses BatchNorm and GELU activations, which provide some implicit normalization. The dynamics model's `input_proj` Linear layer can learn to scale appropriately.

**Current Flow:**
```
Encoder: Conv -> BN -> GELU -> ... -> output (no final norm)
Dynamics: input_proj(latent) -> adds positional embeddings
```

**Observation:** No immediate issue, but monitoring latent statistics during training is recommended.

---

### 1.3 Pre-tokenized .npy File Format Compatibility

**Severity:** Low
**Files Involved:**
- `/Users/danirahman/Repos/ahriuwu/scripts/pretokenize_frames.py` (lines 126-131)
- `/Users/danirahman/Repos/ahriuwu/src/ahriuwu/data/dataset.py` (lines 378-383)

**Description:**
The pretokenizer saves latents as `float16` numpy arrays, and the dataset loads them correctly:

```python
# pretokenize_frames.py
latents = latents.cpu().numpy().astype(np.float16)
np.save(output_path, latents[i])  # Shape: (256, 16, 16)

# dataset.py
latent = np.load(latent_path)
latents.append(torch.from_numpy(latent))  # Implicit conversion to float32 tensor
```

**Observation:** The conversion from float16 numpy to PyTorch tensor results in a float32 tensor by default. This is actually correct behavior for training stability, though it does mean GPU memory usage is 2x what the file size suggests.

---

## 2. Configuration Consistency

### 2.1 Model Size Configs Between Tokenizer and Dynamics

**Severity:** Low (Design Concern)
**Files Involved:**
- `/Users/danirahman/Repos/ahriuwu/src/ahriuwu/models/tokenizer.py` (lines 201-206)
- `/Users/danirahman/Repos/ahriuwu/src/ahriuwu/models/dynamics.py` (lines 434-455)

**Description:**
Both models support "tiny", "small", "medium", "large" sizes, but they configure different parameters and don't share a common config:

| Size | Tokenizer latent_dim | Dynamics model_dim | Dynamics layers |
|------|---------------------|-------------------|-----------------|
| tiny | 128 | 256 | 6 |
| small | 256 | 512 | 12 |
| medium | 384 | 768 | 18 |
| large | 512 | 512 | 24 |

**Issue:** The "large" dynamics model has the same `model_dim=512` as "small", which is unusual. This may be intentional (depth over width) but could be confusing.

**Observation:** The latent_dim -> model_dim relationship is:
- tiny: 128 -> 256 (2x projection)
- small: 256 -> 512 (2x projection)
- medium: 384 -> 768 (2x projection)
- large: 512 -> 512 (1x projection) <-- Different ratio

---

### 2.2 Default Model Size Inconsistency

**Severity:** Low
**Files Involved:**
- `/Users/danirahman/Repos/ahriuwu/scripts/train_dynamics.py` (line 54)
- `/Users/danirahman/Repos/ahriuwu/scripts/eval_dynamics.py` (lines 282-283)

**Description:**
`train_dynamics.py` defaults to `--model-size small` and `--latent-dim 256`. The evaluation script correctly extracts these from the checkpoint:

```python
# eval_dynamics.py
args = checkpoint.get("args", {})
model_size = args.get("model_size", "small")  # Fallback to small
latent_dim = args.get("latent_dim", 256)      # Fallback to 256
```

**Observation:** The fallback values match the training defaults, so this works correctly. However, if `args` is missing from an older checkpoint, it will silently use defaults.

---

## 3. Checkpoint Loading

### 3.1 Missing step_embed Weights in Old Checkpoints

**Severity:** Medium
**Files Involved:**
- `/Users/danirahman/Repos/ahriuwu/scripts/train_dynamics.py` (lines 221-236)
- `/Users/danirahman/Repos/ahriuwu/src/ahriuwu/models/dynamics.py` (lines 323-325)

**Description:**
The `step_embed` layer was added for shortcut forcing support. The training script handles this gracefully with `strict=False`:

```python
# train_dynamics.py
missing, unexpected = model.load_state_dict(checkpoint_data["model_state_dict"], strict=False)
if missing:
    print(f"Note: Initializing new parameters: {missing}")
```

**Issue:** The `step_embed` weights will be randomly initialized when loading old checkpoints. If the model is then used with `--shortcut-forcing`, it will produce incorrect outputs until fine-tuned.

**How to Reproduce:**
1. Train dynamics model without `--shortcut-forcing` (old checkpoint format)
2. Resume training with `--shortcut-forcing`
3. The `step_embed` weights are randomly initialized, causing loss spike

**Observation:** The code correctly warns about missing parameters, but users may not understand the implications.

---

### 3.2 Evaluation Script Missing step_embed Handling

**Severity:** Critical
**Files Involved:**
- `/Users/danirahman/Repos/ahriuwu/scripts/eval_dynamics.py` (lines 277-290)

**Description:**
The evaluation script loads checkpoints with default `strict=True`, which will fail for checkpoints with missing or extra keys:

```python
# eval_dynamics.py - load_dynamics function
def load_dynamics(checkpoint_path: Path, device: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    args = checkpoint.get("args", {})

    model_size = args.get("model_size", "small")
    latent_dim = args.get("latent_dim", 256)

    model = create_dynamics(model_size, latent_dim=latent_dim)
    model.load_state_dict(checkpoint["model_state_dict"])  # <-- STRICT=True by default
```

**How to Reproduce:**
1. Train dynamics model without shortcut forcing (missing `step_embed`)
2. Run eval_dynamics.py
3. If the current model definition has `step_embed` but checkpoint doesn't, it fails:
   ```
   RuntimeError: Error(s) in loading state_dict:
       Missing key(s) in state_dict: "step_embed.mlp.0.weight", ...
   ```

**Recommendation:** Add `strict=False` and warning similar to training script:
```python
missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
if missing:
    print(f"Warning: Missing keys (using random init): {missing}")
```

---

### 3.3 Tokenizer Checkpoint Loading in Eval (Potential Mismatch)

**Severity:** Medium
**Files Involved:**
- `/Users/danirahman/Repos/ahriuwu/scripts/eval_dynamics.py` (lines 293-305)
- `/Users/danirahman/Repos/ahriuwu/scripts/pretokenize_frames.py` (lines 101-114)

**Description:**
Both scripts extract `model_size` from checkpoint args to create the tokenizer:

```python
args = checkpoint.get("args", {})
model_size = args.get("model_size", "small")
model = create_tokenizer(model_size)
```

**Issue:** If `args` is missing from the checkpoint (older format), it defaults to "small". This could cause a mismatch if a different size tokenizer was used.

**How to Reproduce:**
1. Train tokenizer with `--model-size medium`
2. Save checkpoint without `args` field (older save format)
3. Run eval_dynamics.py with this checkpoint
4. It creates "small" tokenizer instead of "medium", causing state_dict mismatch

**Observation:** Modern checkpoints include `args`, so this is a legacy concern. The `load_state_dict` call would catch size mismatches via shape errors.

---

## 4. Training Script Integration

### 4.1 DataLoader Configuration

**Severity:** Low
**Files Involved:**
- `/Users/danirahman/Repos/ahriuwu/scripts/train_dynamics.py` (lines 447-462, 479-486)

**Description:**
DataLoader configuration appears correct:

```python
DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,  # default=4
    pin_memory=True,
    drop_last=True,
)
```

**Observations:**
- `pin_memory=True` is appropriate for CUDA training
- `drop_last=True` prevents variable batch sizes at epoch end
- `num_workers=4` is reasonable default
- `shuffle=True` is correct for training

**Potential Issue:** On MPS (Apple Silicon), `pin_memory=True` may cause warnings but won't break functionality.

---

### 4.2 Alternating Batch Lengths Iterator Cycling

**Severity:** Low
**Files Involved:**
- `/Users/danirahman/Repos/ahriuwu/scripts/train_dynamics.py` (lines 279-304)

**Description:**
The alternating batch length implementation cycles the long dataloader when exhausted but ends the epoch when the short dataloader is exhausted:

```python
if use_long:
    try:
        batch = next(iter_long)
    except StopIteration:
        iter_long = iter(dataloader_long)  # Recycle long loader
        batch = next(iter_long)
else:
    try:
        batch = next(iter_short)
    except StopIteration:
        break  # Epoch ends when short loader exhausted
```

**Observation:** This design is intentional - short sequences dominate (90% by default), so the short loader defines epoch length. The long loader is recycled to ensure long sequences are seen throughout the epoch. This is correct behavior.

**Edge Case:** If the long loader has very few sequences, the same long sequences may repeat many times per epoch. This is acceptable for training but worth noting.

---

### 4.3 Loss Logging with Shortcut Forcing

**Severity:** Low
**Files Involved:**
- `/Users/danirahman/Repos/ahriuwu/scripts/train_dynamics.py` (lines 360-374)

**Description:**
When shortcut forcing is enabled, the logged loss includes both standard and bootstrap components:

```python
if shortcut is not None:
    print(
        f"{progress} "
        f"Loss: {loss.item():.4f} (std:{loss_info['loss_std']:.4f} boot:{loss_info['loss_boot']:.4f}) "
        ...
    )
```

**Observation:** This correctly logs the breakdown. However, `loss_info['loss_std']` and `loss_info['loss_boot']` may both be 0.0 in certain batches (all base steps or all bootstrap steps), which could be confusing.

---

### 4.4 GradScaler Device Mismatch

**Severity:** Medium
**Files Involved:**
- `/Users/danirahman/Repos/ahriuwu/scripts/train_dynamics.py` (line 519)

**Description:**
The GradScaler is always initialized with "cuda" regardless of the actual device:

```python
scaler = GradScaler("cuda")  # Hard-coded "cuda"
```

**How to Reproduce:**
1. Run `train_dynamics.py --device mps`
2. GradScaler("cuda") may cause issues or warnings on MPS

**Observation:** On MPS, GradScaler is less well-supported. The autocast context uses the correct device (`device.split(":")[0]`), but the scaler initialization doesn't follow this pattern.

**Recommendation:** Use:
```python
scaler = GradScaler(device_type=args.device.split(":")[0])
```

---

## 5. Evaluation Script Integration

### 5.1 tau_ctx Application During Inference

**Severity:** Low (Informational)
**Files Involved:**
- `/Users/danirahman/Repos/ahriuwu/scripts/eval_dynamics.py` (lines 336-343, 360-365)

**Description:**
The evaluation script correctly applies `tau_ctx=0.1` to context frames during rollout:

```python
# Add slight noise to context frames to match training distribution
if tau_ctx > 0:
    context_noise = torch.randn_like(context_latents)
    context_latents = (1 - tau_ctx) * context_latents + tau_ctx * context_noise
```

This matches the training behavior where context frames have `tau=0.1`.

**Also applied to predicted frames used as context:**
```python
if tau_ctx > 0:
    pred_noise = torch.randn_like(pred_context)
    pred_context = (1 - tau_ctx) * pred_context + tau_ctx * pred_noise
```

**Observation:** Correctly implemented to match training distribution.

---

### 5.2 Rollout Function Parameter Completeness

**Severity:** Low
**Files Involved:**
- `/Users/danirahman/Repos/ahriuwu/scripts/eval_dynamics.py` (lines 308-399)

**Description:**
The `rollout_predictions` function receives all necessary parameters:

```python
def rollout_predictions(
    dynamics: torch.nn.Module,
    schedule: DiffusionSchedule,
    context_latents: torch.Tensor,
    num_predict: int,
    num_steps: int,
    device: str,
    tau_ctx: float = 0.1,
    use_shortcut: bool = False,
    k_max: int = 64,
) -> torch.Tensor:
```

**Observation:** All parameters are passed through from calling code. The function is complete.

---

### 5.3 Decoded Frame Resolution

**Severity:** Low (Informational)
**Files Involved:**
- `/Users/danirahman/Repos/ahriuwu/scripts/eval_dynamics.py` (lines 402-436)

**Description:**
The `latents_to_frames` function decodes to the tokenizer's native resolution:

```python
# Returns:
#     frames: (B, T, 3, 256, 256) decoded frames
frames = frames_flat.reshape(B, T, 3, 256, 256)
```

The hard-coded `256, 256` matches the tokenizer's design (see tokenizer.py docstring: "256x256 frames to 16x16 latent space").

**Observation:** Correct, but the magic number could be extracted from the tokenizer or made configurable.

---

### 5.4 Rollout Tau End Value

**Severity:** Low (Potential Optimization)
**Files Involved:**
- `/Users/danirahman/Repos/ahriuwu/scripts/eval_dynamics.py` (lines 371-374)

**Description:**
The rollout denoises from tau=1.0 to tau=tau_ctx (0.1) instead of tau=0.0:

```python
# Stop at tau_ctx (0.1) not 0.0 - model never saw tau < 0.1 during training
tau_start = 1.0
tau_end = tau_ctx  # match training minimum
step_size = (tau_start - tau_end) / num_steps
```

**Observation:** This is correct given the diffusion forcing training setup, where context frames have tau=0.1. The model was never trained to predict from tau < 0.1 to tau=0, so stopping at 0.1 is appropriate.

However, this means the final prediction still has 10% noise mixed in. For evaluation metrics like PSNR, this may slightly reduce scores.

---

## 6. Dataset Pipeline

### 6.1 LatentSequenceDataset Stride/Sequence Length Interaction

**Severity:** Low
**Files Involved:**
- `/Users/danirahman/Repos/ahriuwu/src/ahriuwu/data/dataset.py` (lines 338-366)

**Description:**
The `LatentSequenceDataset` correctly handles stride and sequence length:

```python
for start_offset in range(0, contiguous_count - self.sequence_length + 1, self.stride):
    self.sequences.append({
        "video_id": video_id,
        "start_frame": contiguous_start + start_offset,
        "video_dir": video_dir,
    })
```

**Example:** With `sequence_length=64` and `stride=8`:
- Contiguous block of 100 frames
- Valid starts: 0, 8, 16, 24, 32 (last valid is 36 = 100-64)
- Creates 5 sequences

**Observation:** Correctly implemented. The `stride=8` with `sequence_length=64` gives 87.5% overlap between consecutive sequences, providing good data augmentation.

---

### 6.2 Video ID Boundary Respect (Cross-Video Prevention)

**Severity:** Low
**Files Involved:**
- `/Users/danirahman/Repos/ahriuwu/src/ahriuwu/data/dataset.py` (lines 311-366)

**Description:**
The dataset correctly processes each video directory independently and finds contiguous sequences within each video:

```python
for video_dir in sorted(self.latents_dir.iterdir()):
    if not video_dir.is_dir():
        continue

    latent_files = sorted(video_dir.glob("latent_*.npy"))
    # ... process only within this video_dir
```

**Also handles gaps within videos:**
```python
# Find contiguous sequences
if frame_nums[i] == frame_nums[i - 1] + 1:
    contiguous_count += 1
else:
    # End of contiguous block
```

**Observation:** Correctly prevents cross-video sequences. Also handles frame gaps within a single video (e.g., missing frames due to extraction errors).

---

### 6.3 Frame Indexing Consistency

**Severity:** Low
**Files Involved:**
- `/Users/danirahman/Repos/ahriuwu/src/ahriuwu/data/dataset.py` (lines 373-389)
- `/Users/danirahman/Repos/ahriuwu/scripts/pretokenize_frames.py` (lines 62-64)

**Description:**
Both scripts use consistent 6-digit zero-padded frame numbering:

```python
# pretokenize_frames.py
output_path = output_video_dir / f"latent_{frame_num:06d}.npy"

# dataset.py
latent_path = video_dir / f"latent_{frame_num:06d}.npy"
```

**Observation:** Consistent naming convention. Frame numbers are extracted from the original frame filenames, preserving the original indexing.

---

## 7. Import Structure

### 7.1 __init__.py Exports

**Severity:** Low (Informational)
**Files Involved:**
- `/Users/danirahman/Repos/ahriuwu/src/ahriuwu/models/__init__.py`
- `/Users/danirahman/Repos/ahriuwu/src/ahriuwu/data/__init__.py`

**Description:**
Both `__init__.py` files correctly export the necessary classes and functions:

**models/__init__.py:**
```python
from .tokenizer import VisionTokenizer, create_tokenizer
from .losses import TokenizerLoss, VGGPerceptualLoss, psnr
from .diffusion import (
    DiffusionSchedule,
    TimestepEmbedding,
    x_prediction_loss,
    ramp_weight,
    ShortcutForcing,
)
from .dynamics import DynamicsTransformer, create_dynamics
```

**data/__init__.py:**
```python
from .dataset import (
    SingleFrameDataset,
    FrameSequenceDataset,
    FrameWithStateDataset,
    LatentSequenceDataset,
)
```

**Observation:** All necessary exports are present. Scripts can use clean imports like:
```python
from ahriuwu.models import create_dynamics, DiffusionSchedule
from ahriuwu.data import LatentSequenceDataset
```

---

### 7.2 Circular Import Risk Assessment

**Severity:** Low (No Issues Found)
**Files Involved:**
- All reviewed files

**Description:**
The import structure follows a clean hierarchy:

```
data/
  dataset.py       <- No model imports
  __init__.py      <- Re-exports dataset

models/
  tokenizer.py     <- Base, no cross-imports
  diffusion.py     <- TimestepEmbedding only
  dynamics.py      <- Imports from diffusion (TimestepEmbedding)
  losses.py        <- Standalone
  __init__.py      <- Re-exports all

scripts/
  train_dynamics.py   <- Imports from ahriuwu.data and ahriuwu.models
  eval_dynamics.py    <- Imports from ahriuwu.data and ahriuwu.models
  pretokenize_frames.py <- Imports from ahriuwu.models only
```

**Observation:** No circular imports detected. The dependency graph is a clean DAG.

---

## Summary of Issues by Severity

### Critical (1)
1. **Evaluation script missing step_embed handling** - Will fail loading newer checkpoints or older checkpoints depending on model version

### Medium (3)
1. **Latent dimension mismatch risk** - Manual sync required between tokenizer and dynamics configs
2. **Missing step_embed weights in old checkpoints** - Random initialization on resume
3. **GradScaler device mismatch** - Hard-coded "cuda" may cause issues on other devices

### Low (12)
1. Tokenizer output normalization (informational)
2. Pre-tokenized .npy float16->float32 conversion (informational)
3. Model size configs differ between tokenizer and dynamics
4. Default model size inconsistency (handled via fallbacks)
5. DataLoader configuration (potential MPS warning)
6. Alternating batch lengths iterator cycling (edge case)
7. Loss logging with shortcut forcing (minor UX)
8. tau_ctx application during inference (informational, correct)
9. Rollout function parameter completeness (informational, correct)
10. Decoded frame resolution (magic number)
11. Rollout tau end value (potential optimization)
12. Import structure (informational, clean)

---

## Recommended Priority Actions

1. **[Critical]** Add `strict=False` to `load_dynamics()` in `eval_dynamics.py` with appropriate warning message

2. **[Medium]** Fix GradScaler initialization in `train_dynamics.py` to use correct device type

3. **[Medium]** Consider adding a validation check in `train_dynamics.py` that loads a sample latent file and verifies dimensions match `--latent-dim`

4. **[Low]** Extract the `256, 256` frame resolution constant to a shared config or derive from tokenizer properties
