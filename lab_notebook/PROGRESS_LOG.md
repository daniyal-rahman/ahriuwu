# Experiments Registry

This directory tracks different experiment configurations to avoid confusion.

## Directory Structure

```
experiments/
├── README.md                 # This file
├── registry.json             # Machine-readable experiment registry
└── <experiment_id>/          # Per-experiment configs and notes
    ├── config.json
    └── notes.md
```

## Experiment Naming Convention

Format: `{date}_{description}`

Example: `20260302_dynamics_small_baseline`

## Current Experiments

| ID | Date | Tokenizer | Dynamics | Latent Dir | Checkpoint Dir | Status |
|----|------|-----------|----------|------------|----------------|--------|
| `trans_small_20260107` | 2026-01-07 | transformer_small (40M) | small | `data/processed/latents` | `checkpoints/dynamics_*` | Complete |

---

## K4 Shortcut Forcing Investigation (2026-03-25 to 2026-03-30)

### Problem
K4 shortcut inference (the actual deployment inference mode) oscillates between 15-22 dB and crashes to 3 dB. Single-step denoising works great (35 dB). Bootstrap loss is near-zero despite 10x weight boost.

### Root Causes Found

**1. Velocity-space bootstrap trap (FIXED)**
The bootstrap loss computed in velocity space with (1-τ)² weighting numerically killed the signal in bfloat16. Division by (1-τ) amplified, squaring amplified more, then (1-τ)² cancelled — but bfloat16 lost precision in intermediate steps. Additionally, when teacher ≈ student (self-consistent model), velocity diff was exactly zero.
- **Fix**: X-space bootstrap loss. Compare student x-prediction to teacher's two-step target directly. 213x stronger signal.

**2. Self-consistent bootstrap deadlock (FIXED)**
Teacher and student are the same model. When model ignores step_size (learned during standard flow training), both produce identical z_0 predictions → identical velocities → zero loss → no gradient → no learning.
- **Fix**: Progressive step size curriculum. Train d∈{1,2} first (teacher d=1 is well-trained), then gradually add d=4, d=8, d=16.

### Remaining Issue: K4 Oscillation on Medium Model (2026-03-30)

Even with x-space loss + progressive curriculum, K4 oscillates between good (15-22 dB) and crash (3 dB) during training on medium model (114M). 20-batch eval at a fixed checkpoint shows 0 crashes and mean K4=19.2 dB, suggesting the crashes are transient training instability.

**5 candidate causes for the oscillation:**

1. **10x bootstrap weight too aggressive** — creates gradient spikes 100x larger than the settled flow loss, temporarily wrecking the model before flow loss pulls it back.

2. **Progressive curriculum transitions** — new step size unlocking every 2k steps causes sudden gradient spikes from untrained d values.

3. **Additive conditioning amplifies instability** — step_emb added to ALL 265 spatial tokens. A large bootstrap gradient on step_emb perturbs the entire network simultaneously.

4. **B=1 for shortcut steps** — single-sample gradient has enormous variance. One unlucky sample can spike the weights.

5. **Eval catches transient states** — 20-batch eval at fixed checkpoint shows 0 crashes. The wandb crashes happen when eval fires mid-gradient-update, catching the model in a transient destabilized state.

### Decision: Scale down to small model (2026-03-30)

Training small (36M) with 3x bootstrap weight for 1-1.5 days to validate the approach before spending more days on medium. Small model trains 3-4x faster, giving faster iteration.

**Invariant across model sizes** (validated at small, carries to medium):
- Tau convention, noise schedule, x-space bootstrap formula
- Progressive curriculum algorithm, K=4 inference loop
- Ramp weight, data pipeline, eval methodology

**Variant** (re-tune when scaling):
- Bootstrap weight (3x at small, TBD at medium)
- Batch size (B=2 shortcut at small, B=1 at medium)
- Progressive schedule (maybe faster at small)
- Warmup steps (fewer at small)
- Additive conditioning (essential at both scales, paper uses attention-only at 1.6B)

### Run History

| Run | Model | Code | Best τ=0.9 | Best K4 | K4 Stable? | Notes |
|-----|-------|------|------------|---------|------------|-------|
| 59 | small | velocity-space, 1/d sampling | 22.8 | 3.7 | No | K4 stuck near 0 |
| 60-61 | small→med | separate additive, velocity-space | 36.9 | 27.3 | No, 3-27 | Step_size gradient fix helped |
| 67-70 | medium | velocity-space, seeded eval | 34.7 | 27.3 | No, 3-27 | Bootstrap 57% nonzero |
| 108 | medium | x-space, progressive, 10x weight | 24.0 | 19.2* | Eval stable, wandb spiky | *20-batch eval: 0 crashes |
| 131-136 | small | x-space, progressive, 3x weight | 24.5 | N/A** | N/A | K4 eval was broken |
| 140 | small | x-space, progressive, 3x, FIXED eval | 30.4 | d16=31.1 | Yes | Gap: -0.2 dB (d16 matches d1) |

**K4 eval was measuring unconditional generation, not shortcut denoising. See below.

### K4 Eval Bug (FIXED 2026-03-31)

The K4 eval started from pure random noise (unrelated to z_0), ran 4 Euler steps, then compared output to z_0. This measured unconditional generation quality — the output had no connection to z_0, so PSNR was always ~3.2 dB. The "crashes" were not model failures; they were the expected result of comparing random samples to unrelated ground truth.

**Fix:** Replaced with d=16 single-step denoising eval. Compare d=1 vs d=16 PSNR at each tau level. The gap shows shortcut quality directly.

**Result:** Shortcut gap is -0.2 dB (d=16 matches or beats d=1) across all tau levels. The model has been learning shortcut forcing correctly for weeks — we just couldn't see it.

### Comprehensive Eval (2026-03-31)

Small model (36M, step 27k), 20 samples, all step sizes:

| tau | d=1 | d=2 | d=4 | d=8 | d=16 | d=32 | d=64 |
|-----|-----|-----|-----|-----|------|------|------|
| 0.1 | 19.1 | 19.0 | 19.2 | 19.4 | 19.3 | 19.0 | 18.4 |
| 0.3 | 22.1 | 21.9 | 22.3 | 22.2 | 22.3 | 22.1 | 22.4 |
| 0.5 | 23.8 | 23.7 | 23.9 | 23.8 | 24.0 | 23.8 | 24.0 |
| 0.7 | 24.3 | 24.2 | 24.4 | 24.4 | 24.5 | 24.4 | 24.6 |
| 0.9 | 24.5 | 24.4 | 24.6 | 24.6 | 24.7 | 24.6 | 24.8 |

All step sizes produce identical quality. Shortcut forcing is fully working.

### Known Issue: Tokenizer Decode NaN

The attention unification refactor (commit a09d2e0) changed the tokenizer's attention structure. The existing tokenizer checkpoint (trained pre-refactor) produces NaN when decoded with the new code. Cause: unified Attention class has different RoPE buffer names and possibly different attention computation order than the old per-class implementation. The tokenizer needs to be retrained with the new code, or a checkpoint compatibility shim is needed.

**Impact:** Cannot compute pixel PSNR or LPIPS through the tokenizer. Latent-space PSNR is the only available metric until this is resolved.

### Training Status (2026-03-31)

- **Small model (36M):** Step 27k, running on gpu-long QOS
  - Single-step τ=0.9: 24.5 dB (d=1), 24.7 dB (d=16)
  - Shortcut gap: -0.2 dB (d16 matches d1)
  - Bootstrap: 57% of steps nonzero
  - Throughput: 2.3-3.7 batch/s with dynamic batch slicing
- **Medium model (114M):** Step 22k, backed up
  - Single-step τ=0.9: 25.9 dB (identical across all d)
  - Higher peak quality but τ=0.1 unstable at large d
- **Tokenizer:** Broken decode (NaN), needs retraining or compat fix

### Next Steps
1. Fix tokenizer compat or retrain tokenizer with unified attention
2. Run pixel PSNR + LPIPS eval once tokenizer decode works
3. Continue small model to 50k steps (~1.3 days)
4. Scale findings to medium model
5. Agent finetuning (Phase 2) once dynamics + tokenizer are both working

---
