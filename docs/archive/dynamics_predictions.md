# Dynamics Model Training Predictions

## Pre-Training Checklist (Blockers)

### 1. Latent Data Exists and is Valid
```bash
# Verify latents exist
ls data/processed/latents/ | head -5

# Check latent dimensions (should be 256x16x16)
python -c "import numpy as np; x=np.load('data/processed/latents/<video>/latent_000000.npy'); print(x.shape, x.dtype)"
# Expected: (256, 16, 16) float32
```

**Potential Issue:** Latent dimension mismatch with model
- Training script default: `--latent-dim 256`
- If CNN tokenizer outputs different dim, will get shape error immediately

### 2. Sequence Length vs Data Length
- Need at least `sequence_length` consecutive frames per video
- With stride=8, seq_len=64: need ~72 frames minimum per video
- Check: `ls data/processed/latents/<video>/*.npy | wc -l`

### 3. VRAM Budget
| Config | Approx VRAM |
|--------|-------------|
| seq_len=32, batch=2, small model | ~10GB |
| seq_len=64, batch=1, small model | ~12GB |
| seq_len=64, batch=2, small model | OOM on 16GB |

---

## Red Flags During Training

### HIGH PRIORITY - Stop Training Immediately

#### 1. Loss Explodes (NaN or Inf)
**Symptom:** `Loss: nan` or `Loss: inf` or loss > 10
**Likely Causes:**
- Learning rate too high (try 1e-5 instead of 1e-4)
- Gradient explosion (check if grad_norm is very high)
- Division by zero in timestep embedding or ramp weight

**Fix:** Lower LR, check for zero tau values in diffusion forcing

#### 2. Loss Immediately Plateaus at High Value
**Symptom:** Loss starts at ~0.5+ and doesn't decrease in first 1000 steps
**Likely Causes:**
- Model not learning at all (mode collapse potential)
- Wrong tau convention (model trained with opposite meaning)
- Output projection not connected properly

**Fix:** Check model outputs vary with different inputs (like we did for tokenizer)

#### 3. CUDA OOM
**Symptom:** `CUDA out of memory`
**Fix:**
- Reduce batch_size
- Reduce sequence_length
- Enable gradient checkpointing
- Use alternating lengths with smaller long batches

### MEDIUM PRIORITY - Monitor Closely

#### 4. Loss Decreasing but Predictions Blurry
**Symptom:** Loss goes down but eval shows blurry/smeared predictions
**Likely Causes:**
- Not enough temporal attention (every 4th layer may be too sparse)
- Diffusion forcing not creating enough clean→noisy gradient
- Need more training

**Diagnostic:** Save sample predictions periodically, check if later frames are worse than early frames

#### 5. Loss Variance Very High
**Symptom:** Loss oscillates wildly (e.g., 0.01 → 0.05 → 0.008 → 0.06)
**Likely Causes:**
- Batch size too small (high gradient variance)
- Alternating lengths causing distribution shift
- Shortcut forcing with large step sizes

**Fix:** Increase batch size, use gradient accumulation

#### 6. Temporal Inconsistency
**Symptom:** Individual frames look good but motion is jerky/inconsistent
**Likely Causes:**
- Temporal attention not strong enough
- Short sequence lengths during training
- Not enough long batches in alternating mode

**Diagnostic:** Check if temporal attention weights are uniform (like tokenizer issue)

### LOW PRIORITY - Expected Behaviors

#### 7. Early Loss Spike Then Recovery
**Expected:** Loss may spike in first few hundred steps then settle
**Why:** AdamW warmup, model finding good initialization
**Action:** Only worry if spike doesn't recover within 1k steps

#### 8. Loss Different for Short vs Long Sequences
**Expected:** Long sequences (T=64) may have higher loss than short (T=32)
**Why:** More frames to predict = harder task
**Action:** Compare within same sequence length, not across

---

## Metrics to Log (Enhanced Verbosity)

### Must Have (Already Logged)
- [x] Total loss
- [x] Tau range [min-max]
- [x] Batches per second
- [x] Shortcut loss breakdown (loss_std, loss_boot)

### Should Add
- [ ] **Gradient norm** - catches explosion before NaN
- [ ] **Per-timestep loss** - are later frames harder?
- [ ] **Prediction variance** - is output varying with input?
- [ ] **Max/min prediction values** - sanity check outputs
- [ ] **Memory usage** - track VRAM over time

### Nice to Have
- [ ] Attention entropy (like tokenizer diagnostic)
- [ ] Sample predictions every N steps
- [ ] Learning rate (if using scheduler)

---

## Verification Tests Before Long Run

### Quick Sanity Check (~5 min)
```bash
# Run 100 steps, verify:
# 1. Loss decreases
# 2. No NaN/Inf
# 3. No OOM
python scripts/train_dynamics.py \
  --epochs 1 \
  --save-interval 999 \
  --log-interval 10
```

### Input/Output Variation Check
```python
# After 1k steps, verify model isn't mode collapsing
model.eval()
x1 = torch.randn(1, 16, 256, 16, 16).cuda()
x2 = torch.randn(1, 16, 256, 16, 16).cuda()
tau = torch.tensor([0.5]).cuda()

y1 = model(x1, tau)
y2 = model(x2, tau)

diff = (y1 - y2).abs().mean()
print(f"Output diff: {diff:.6f}")  # Should be > 0.1, not near 0
```

---

## Recommended Training Commands

### Phase 1: Smoke Test (5 min)
```bash
python scripts/train_dynamics.py \
  --model-size small \
  --sequence-length 32 \
  --batch-size 2 \
  --epochs 1 \
  --log-interval 10 \
  --save-steps 0
```

### Phase 2: Quick Validation (1 hour)
```bash
python scripts/train_dynamics.py \
  --model-size small \
  --alternating-lengths \
  --seq-len-short 32 \
  --seq-len-long 64 \
  --batch-size-short 2 \
  --batch-size-long 1 \
  --epochs 1 \
  --save-steps 5000 \
  --log-interval 50
```

### Phase 3: Full Training
```bash
python scripts/train_dynamics.py \
  --model-size small \
  --alternating-lengths \
  --seq-len-short 32 \
  --seq-len-long 64 \
  --batch-size-short 2 \
  --batch-size-long 1 \
  --epochs 10 \
  --save-steps 10000 \
  --shortcut-forcing
```

---

## Historical Issues (From Tokenizer)

### Mode Collapse Pattern
- **Symptom:** Output identical regardless of input
- **Root Cause:** Attention became uniform, averaging all tokens
- **Detection:** Check output diff between different inputs < 0.001
- **Fix for Dynamics:** Watch gradient norm, ensure temporal attention is learning

The tokenizer collapsed because MAE let it output mean image. Dynamics is harder to collapse since:
1. Diffusion forcing requires predicting across noise levels
2. Temporal attention must vary by frame position
3. No masking shortcut available

BUT still possible if:
- Model just predicts noise level, ignores content
- Temporal attention collapses to uniform
- Output projection gets stuck at initialization

---

## When to Alert

**Immediate (stop & debug):**
- Loss NaN/Inf
- Loss > 1.0 after 1000 steps
- OOM error
- "Output diff" < 0.001 (mode collapse)

**Within 1 hour (investigate):**
- Loss not decreasing after 5000 steps
- Loss variance > 50% of mean
- PSNR < 15 dB on eval

**End of run (review):**
- Final loss not < initial loss by 50%+
- Eval predictions blurry beyond frame 4
- Temporal consistency poor
