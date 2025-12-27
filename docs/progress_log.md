# Progress Log

## 2024-12-25: Stride vs Alternating Batch Lengths

### Decision
Using overlapping stride (stride=8, seq_len=16) instead of DreamerV4's alternating batch lengths

### Why diverging from DreamerV4
DreamerV4 uses alternating batch lengths (T1=64 short, T2=256 long) bc:
- their sequences are 192+ frames
- attention is O(n^2), long seqs = slow
- short batches speed up training, occasional long batches for temporal consistency

### Why stride is ok for us
- our seq_len=16 is short, no quadratic blowup
- stride=8 gives 2x data augmentation (50% overlap)
- simpler impl, same effect for short sequences
- model is 60M params vs their 1.6B, dont need the complexity yet

### When to switch to alternating
- if seq_len goes to 64+ frames
- if training becomes slow

### Risk
- might miss some temporal coherence benefits from occasional long batches
- prob fine bc our sequences are already short enough that full sequence fits in one forward pass

---

## 2024-12-25: Diffusion Forcing Fix

### Problem
Predictions were completely blurry except static HUD. Model was trained as denoiser (same tau for all frames) not predictor.

### Fix
Added `sample_diffusion_forcing_timesteps(B, T)` - per-timestep noise levels:
- context frames: tau=0.1 (mostly clean)
- target frames: tau increasing toward 1.0

Training now shows `Tau: [0.10-1.00]` confirming fix working.

### Result
After 10 epochs (loss 0.0092): can see garen, tower, minions in predictions. still blurry but much better than before.

---

## 2024-12-25: Scaling Up Data

### Current
- 10 videos (~5 hrs)
- 344K tokenized frames
- 21K sequences (stride=16)

### Target
- 100 videos (~50 hrs)
- downloading from domisumReplay-Garen channel
- with stride=8 will get ~4x more sequences

### DreamerV4 comparison
- they used 2541 hrs (500x more than us)
- but their model is 1.6B dynamics, ours is 60M
- 50 hrs should be reasonable for our scale

---

## 2024-12-26: Transformer Sequence Length Flexibility

### Finding
Same 60M dynamics model can handle variable sequence lengths without architecture changes.

### Why it works
1. **Attention is length-agnostic** - computes pairwise relationships regardless of sequence length
2. **Autoregressive rollout** - predict one frame at a time, concat context + predicted
3. **Relative position learning** - model learns "frame after frame", not absolute positions

### Practical implications
- Epoch 1 trained with seq_len=32 (1.6 sec)
- Epoch 2+ training with seq_len=64 (3.2 sec) - same model, just more context
- Eval can use different context lengths (e.g., 48 context + 16 predict)
- Only constraint is VRAM (attention is O(n^2))

### Memory scaling
- seq_len=32, batch=2: ~10GB VRAM
- seq_len=64, batch=1: ~10GB VRAM
- seq_len=100, batch=1: OOM (>16GB)

### Context comparison with DreamerV4
- DreamerV4: 320 frames context (16 sec) -> predict 16 frames (0.8 sec)
- Us (epoch 1): 16 frames context (0.8 sec) -> predict 16 frames (0.8 sec)
- Us (epoch 2+): 48 frames context (2.4 sec) -> predict 16 frames (0.8 sec)

More context = better long-horizon prediction, less error accumulation.

---

## 2024-12-26: Epoch 1 Evaluation Results

### Setup
- 16 context frames, 16 predict frames
- 64 diffusion steps

### Results
- PSNR: 21.00 dB ("Acceptable - some blurring")
- MSE: 0.013

### Observations
- Frames 1-2: Good quality, recognizable scene
- Frames 3-4: Significant smearing, error accumulation visible
- Garen appears to "spin" in predictions - could be error accumulation OR model learned his E animation

### Interpretation
Error accumulation: each predicted frame has small errors, fed back as input for next prediction, errors compound exponentially. More training + more context should help.

---

## 2024-12-26: Context Corruption Fix (Train/Test Mismatch)

### The Bug
During training, context frames have τ=0.1 (slight noise via diffusion forcing).
During eval, context frames were clean (τ=0).

This is a train/test distribution mismatch - model trained expecting slightly noisy context, but got clean context at inference.

### The Fix
Added `tau_ctx` parameter to `rollout_predictions()`:
```python
# Add slight noise to context to match training
if tau_ctx > 0:
    context_noise = torch.randn_like(context_latents)
    context_latents = (1 - tau_ctx) * context_latents + tau_ctx * context_noise
```

Also apply to predicted frames when using them as context for subsequent predictions.

### Why This Matters
- Model learned to denoise from τ=0.1, not τ=0
- Clean inputs might produce overconfident/wrong predictions
- Could explain rapid degradation at frames 3-4

### Usage
```bash
python scripts/eval_dynamics.py ... --tau-ctx 0.1  # default, matches training
python scripts/eval_dynamics.py ... --tau-ctx 0.0  # disable for comparison
```

---

## 2024-12-26: Noise Schedule Analysis

### Current Sampling (64-step Euler)
Linear schedule from τ=1.0 to τ=0.0:
```python
step_size = 1.0 / num_steps  # = 0.015625
for i in range(num_steps):
    tau = 1.0 - i * step_size  # τ: 1.0 → 0.984 → 0.969 → ... → 0.015 → 0.0
```

This is **flow matching** style interpolation, not DDPM cosine schedule.

### Flow Matching Formulation
```
z_tau = (1 - tau) * z_0 + tau * noise
```
- τ=1.0: pure noise
- τ=0.0: clean signal
- τ=0.5: 50/50 mixture

### Euler Sampling Step
```python
z_0_pred = model(z_t, tau)  # model predicts clean signal (x-prediction)
z_next = (1 - next_tau) * z_0_pred + next_tau * noise  # interpolate toward clean
```

### Future: Shortcut Forcing (DreamerV4)
Requires training model to take larger steps directly:
- Add step size `d` as conditioning input alongside τ
- Bootstrap loss: distill 2 small steps into 1 big step
- Train on d ∈ {1/4, 1/8, 1/16, ...}
- At inference: use d=1/4 for 4-step sampling (16x speedup)

Not just "use fewer steps" - model must learn to skip intermediate states.

---

## 2024-12-27: Shortcut Forcing Implementation

### What It Does
Enables 4-step inference instead of 64 steps (16x speedup) by training model to take large denoising steps directly.

### How It Works
1. **Step size conditioning**: Model receives normalized step size `d/k_max` alongside τ
2. **Bootstrap loss**: For large steps, teacher takes 2 half-steps, student learns to match in 1 step
3. **Inverse sampling**: `P(d) ∝ 1/d` so small steps train more often (they're the foundation)

### Key Implementation Details

**Step sizes**: Powers of 2 from 1 to k_max
```python
step_sizes = [1, 2, 4, 8, 16, 32, 64]  # for k_max=64
```

**Bootstrap loss (d > 1)**:
```python
# Teacher: 2 half-steps
z_mid = model(z_tau, tau, step_size=d/2)
tau_mid = tau - d/2  # halfway denoised
z_tau_mid = (1-tau_mid)*z_mid + tau_mid*noise
z_target = model(z_tau_mid, tau_mid, step_size=d/2)

# Student: 1 full step, match teacher
z_pred = model(z_tau, tau, step_size=d)
loss = mse(z_pred, z_target)
```

### Files Changed
- `dynamics.py`: Added `step_embed` embedding, `step_size` param to forward
- `diffusion.py`: Implemented `ShortcutForcing.compute_loss()` with bootstrap
- `train_dynamics.py`: Added `--shortcut-forcing`, `--shortcut-k-max`
- `eval_dynamics.py`: Added `--use-shortcut`, `--shortcut-k-max`

### Usage

**Training**:
```bash
python scripts/train_dynamics.py --shortcut-forcing --shortcut-k-max 64
```

**Inference (4 steps = 16x faster)**:
```bash
python scripts/eval_dynamics.py --use-shortcut --num-steps 4
```

### Backward Compatibility
- `step_size=None` uses existing behavior
- Old checkpoints work with new code
- New checkpoints require `--use-shortcut` at inference

### Convention Note
Our codebase: τ=0 clean, τ=1 noise (denoising decreases τ)
Paper: τ=0 noise, τ=1 clean (denoising increases τ)
Implementation accounts for this difference.

---

## 2024-12-27: Alternating Batch Lengths

### What It Does
Follows DreamerV4 Section 3.4: "alternate training on many short batches and occasional long batches"

Short batches = faster training, long batches = better temporal consistency.

### Configuration

| | Short | Long |
|---|---|---|
| Paper | T=64 | T=256 |
| Us (16GB) | T=32 | T=64 |
| Batch size | 2 | 1 |

### Implementation
Two separate dataloaders, ratio-based random selection:
```python
# 90% short, 10% long (configurable via --long-ratio)
use_long = random.random() < args.long_ratio
if use_long:
    batch = next(iter_long)  # T=64, B=1
else:
    batch = next(iter_short)  # T=32, B=2
```

### Why Two Dataloaders
- Simplest approach: no collate function changes, no padding
- Each dataloader indexes same underlying latent files but with different sequence lengths
- Long dataloader recycles when exhausted (fewer long sequences available)
- Epoch ends when short dataloader exhausted

### Usage
```bash
python scripts/train_dynamics.py \
  --alternating-lengths \
  --seq-len-short 32 \
  --seq-len-long 64 \
  --batch-size-short 2 \
  --batch-size-long 1 \
  --long-ratio 0.1
```

### Log Output
Shows batch type and sequence length:
```
Epoch 1 [50/1000] S T=32 Loss: 0.0123 Tau: [0.10-1.00] (2.5 batch/s)
Epoch 1 [51/1000] L T=64 Loss: 0.0134 Tau: [0.10-1.00] (2.5 batch/s)
```

### Next Steps
- Test if T=64 still OOMs or if we can push to T=80/100 for long batches
- Experiment with different long ratios (5%, 10%, 20%)

---

## 2024-12-27: Ready for Desktop Testing

### Changes Ready to Test
1. **Shortcut forcing** - 4-step inference (16x speedup)
2. **Alternating batch lengths** - short T=32 / long T=64
3. **Context corruption fix** - tau_ctx=0.1 at inference

### Test Commands
```bash
# Smoke test imports
python -c "from ahriuwu.models import ShortcutForcing, create_dynamics; print('OK')"

# Alternating lengths (find VRAM limit)
python scripts/train_dynamics.py \
  --alternating-lengths \
  --seq-len-short 32 \
  --seq-len-long 64 \
  --batch-size-short 2 \
  --batch-size-long 1 \
  --long-ratio 0.1 \
  --epochs 1

# If T=64 works, try T=80 or T=100 for --seq-len-long
```
