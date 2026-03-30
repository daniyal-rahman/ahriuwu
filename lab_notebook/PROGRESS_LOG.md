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
| next | small | x-space, progressive, 3x weight | TBD | TBD | TBD | Validation run |

---
