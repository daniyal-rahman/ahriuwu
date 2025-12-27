# DreamerV4 Implementation - Comprehensive Code Review Summary

**Date:** 2025-12-27
**Codebase:** `/Users/danirahman/Repos/ahriuwu`
**Reference:** DreamerV4 Paper - "Training Agents Inside of Scalable World Models" (Hafner et al., 2025)

---

## Executive Summary

This multi-scale code review analyzed the DreamerV4 implementation across four review dimensions:
1. **Scale 1: Equation Correctness** - Mathematical formulas vs. paper
2. **Scale 2: Architecture Deviations** - Model design vs. paper specifications
3. **Scale 3: Code-Level Bugs** - Runtime issues and edge cases
4. **Scale 4: File Wiring/Integration** - Component interconnections

**Overall Assessment:** The implementation captures the core DreamerV4 concepts correctly (factorized attention, x-prediction, diffusion forcing) but has several critical equation bugs that need immediate attention before meaningful training.

---

## Critical Issues (Must Fix Before Training)

### 1. Ramp Weight Formula Inverted
**File:** `src/ahriuwu/models/diffusion.py:209`
**Review:** [scale1_diffusion_equations.md](scale1_diffusion_equations.md#finding-2-ramp-weight-function---critical-bug)

**The Bug:**
```python
# Current (WRONG for implementation's tau convention)
def ramp_weight(tau):
    return 0.9 * tau + 0.1  # HIGH weight at noise (tau=1), LOW at clean (tau=0)

# Should be (given tau=0 clean, tau=1 noise)
def ramp_weight(tau):
    return 1.0 - 0.9 * tau  # HIGH weight at clean (tau=0), LOW at noise (tau=1)
```

**Impact:** The model is being trained to focus on noisy examples (least useful gradients) instead of clean examples (most informative gradients). This likely degrades model quality significantly.

**Paper Reference:** Equation 8, Section 3.3 states the ramp weight should "focus the model capacity on signal levels with the most learning signal" (high signal = clean data).

---

### 2. Missing Tau Grid Sampling for Shortcut Forcing
**File:** `scripts/train_dynamics.py:314`
**Review:** [scale1_shortcut_forcing.md](scale1_shortcut_forcing.md#finding-1-missing-tau-grid-sampling-eq-4)

**The Bug:**
```python
# Current: tau sampled independently of step size
tau = schedule.sample_diffusion_forcing_timesteps(B, T, device=device)

# Paper Eq. 4: tau must come from grid aligned to step size d
# tau in {0, 1/d, 2/d, ..., 1 - 1/d}
```

**Impact:** Without grid-aligned tau, the bootstrap learning signal is inconsistent - after taking a step of size d, tau won't land on valid grid points.

---

### 3. Missing Velocity-Space Bootstrap Loss
**Files:** `src/ahriuwu/models/diffusion.py:419-423`
**Review:** [scale1_shortcut_forcing.md](scale1_shortcut_forcing.md#finding-3-missing-velocity-conversion-in-bootstrap-eq-7)

**The Bug:**
```python
# Current: Simple MSE in x-space
loss_boot = F.mse_loss(z_pred, z_target.detach())

# Paper Eq. 7: Loss in velocity space with scaling
# L = tau^2 * ||b_student - sg(avg_velocity)||^2
# where velocities: b = (prediction - noisy) / tau
```

**Impact:** The shortcut forcing mechanism won't work as intended. The paper's formulation averages teacher velocities and applies τ² scaling for proper gradient weighting.

---

### 4. Missing (1-τ)² / τ² Scaling Factor
**File:** `src/ahriuwu/models/diffusion.py:423`
**Review:** [scale1_shortcut_forcing.md](scale1_shortcut_forcing.md#finding-2-missing-1-tau2-scaling-factor-eq-7)

**Paper Eq. 7:** `L = (1-tau)^2 * || ... ||^2` (paper convention)
**Our convention:** τ² scaling (since our tau=0 is clean, tau=1 is noise)

**Impact:** Without this scaling, gradient magnitudes are wrong for different noise levels.

---

### 5. Evaluation Script Will Crash on Old Checkpoints
**File:** `scripts/eval_dynamics.py:277-290`
**Review:** [scale4_integration.md](scale4_integration.md#32-evaluation-script-missing-step_embed-handling)

**The Bug:**
```python
# Current: strict=True (default)
model.load_state_dict(checkpoint["model_state_dict"])

# Should be:
missing, _ = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
if missing:
    print(f"Warning: Missing keys: {missing}")
```

**Impact:** RuntimeError when loading checkpoints without `step_embed` weights (added for shortcut forcing).

---

## High Priority Issues

### 6. step_size Broadcast Mismatch
**File:** `src/ahriuwu/models/dynamics.py:400-401`
**Review:** [scale3_bugs.md](scale3_bugs.md#bug-82-step_size-broadcast-mismatch-in-shortcutforcing)

When using per-timestep tau `(B, T)` with step_size `(B,)`:
- `time_emb`: `(B, T, D)`
- `step_emb`: `(B, D)`
- Addition `time_emb + step_emb` causes shape mismatch error

**Fix:** Add `.unsqueeze(1)` to step_emb before adding.

---

### ~~7. Step Size Sampling Uses Wrong Distribution~~ (RETRACTED)
**File:** `src/ahriuwu/models/diffusion.py:321-339`

**Status:** ACTUALLY CORRECT

Paper Eq. 4 notation `d ~ 1/U({1, 2, 4, ...})` means P(d) ∝ 1/d, which is exactly what the implementation does. Larger step sizes are sampled less frequently. The current inverse weighting is correct.

---

### 8. tau_mid Edge Case
**File:** `src/ahriuwu/models/diffusion.py:402-406`
**Review:** [scale3_bugs.md](scale3_bugs.md#bug-24-missing-clamping-in-shortcut-forcing-half-step-calculation)

When tau is already low and step_size is large, `tau_mid` clamps to 0, causing the second half-step to operate on clean data instead of noisy data.

---

## Medium Priority Issues

### Architecture Deviations from Paper

| Feature | Paper | Implementation | File | Priority |
|---------|-------|----------------|------|----------|
| Position Encoding | RoPE | Learned embeddings | dynamics.py:314-319 | MEDIUM |
| Tanh Bottleneck | Yes (Eq. 2) | No | tokenizer.py:162-168 | MEDIUM |
| QKNorm | Yes (Section 3.4) | No | dynamics.py:104-107 | MEDIUM |
| Attention Soft Capping | Yes (Section 3.4) | No | dynamics.py:106 | MEDIUM |
| Tokenizer Architecture | Block-causal Transformer | CNN ResNet | tokenizer.py:67-99 | MEDIUM |
| MAE Training | Yes (Section 3.1) | No | tokenizer.py | MEDIUM |
| GQA | Yes (Section 3.4) | Standard MHA | dynamics.py:81-82 | LOW |
| Register Tokens | Yes (Section 3.2) | No | dynamics.py | LOW |

**Details:** [scale2_architecture.md](scale2_architecture.md), [scale2_attention.md](scale2_attention.md)

---

### Integration Issues

| Issue | File | Severity |
|-------|------|----------|
| GradScaler hard-coded "cuda" | train_dynamics.py:519 | MEDIUM |
| Latent dim mismatch risk | tokenizer.py/dynamics.py | MEDIUM |
| Default seq lengths differ from paper (32/64 vs 64/256) | train_dynamics.py:88-97 | MEDIUM |
| Missing step_embed in old checkpoints | train_dynamics.py:221-236 | MEDIUM |

**Details:** [scale4_integration.md](scale4_integration.md)

---

## What's Implemented Correctly

### Core Architecture
- Factorized space-time attention pattern
- Temporal attention every 4th layer
- Causal masking on temporal attention only
- Pre-norm transformer with RMSNorm and SwiGLU
- AdaLN modulation (6-parameter: shift, scale, gate x2)
- Zero-initialized output projection

### Diffusion Components
- Linear noise schedule (correct formula)
- X-prediction (predicting clean data)
- Diffusion forcing (per-timestep noise levels)
- Euler sampling (tau=1 to tau=tau_ctx)
- tau_ctx=0.1 for context frames

### Training Infrastructure
- Alternating batch lengths
- Gradient clipping
- Mixed precision training (autocast)
- Checkpoint saving with args

**Details:** [scale2_attention.md](scale2_attention.md#9-summary)

---

## Tau Convention Reference

The codebase uses an **inverted tau convention** from the paper:

| Value | Paper Convention | Implementation |
|-------|-----------------|----------------|
| tau=0 | Full noise | Clean data |
| tau=1 | Clean data | Full noise |

This affects how formulas must be translated. The code correctly handles this in most places EXCEPT the ramp weight function.

---

## Recommended Fix Order

### Phase 1: Critical Math Fixes (Before Training)
1. Fix `ramp_weight()` - change to `1.0 - 0.9 * tau`
2. Add tau grid sampling for shortcut forcing
3. Implement velocity-space bootstrap loss with τ² scaling
4. Add `strict=False` to eval_dynamics.py checkpoint loading

### Phase 2: High Priority Bug Fixes
5. Fix step_emb broadcast mismatch (add `.unsqueeze(1)`)
6. Change step size sampling to uniform
7. Add tau_mid minimum threshold

### Phase 3: Stability Features (For Scale)
8. Add tanh bottleneck to tokenizer
9. Implement RoPE position embeddings
10. Add QKNorm and attention soft capping
11. Fix GradScaler device initialization

### Phase 4: Scale-Up Features (Optional)
12. Implement MAE training for tokenizer
13. Add GQA to attention layers
14. Add register tokens
15. Consider transformer tokenizer migration

---

## Detailed Review Files

| Scale | Focus | File |
|-------|-------|------|
| 1 | Diffusion Equations | [scale1_diffusion_equations.md](scale1_diffusion_equations.md) |
| 1 | Shortcut Forcing | [scale1_shortcut_forcing.md](scale1_shortcut_forcing.md) |
| 1 | Training Math | [scale1_training_math.md](scale1_training_math.md) |
| 2 | Model Architecture | [scale2_architecture.md](scale2_architecture.md) |
| 2 | Attention Mechanisms | [scale2_attention.md](scale2_attention.md) |
| 3 | Code-Level Bugs | [scale3_bugs.md](scale3_bugs.md) |
| 4 | Integration | [scale4_integration.md](scale4_integration.md) |

---

## Issue Count Summary

| Severity | Count | Key Examples |
|----------|-------|--------------|
| CRITICAL | 5 | Ramp weight inverted, missing tau grid, missing velocity bootstrap |
| HIGH | 2 | step_size broadcast, tau_mid edge case |
| MEDIUM | 10+ | Missing RoPE, QKNorm, tanh bottleneck, GradScaler device |
| LOW | 12 | Edge cases, informational, code quality |

---

*This review was conducted by launching 7 specialized subagents across 4 review scales, referencing the DreamerV4 paper (arXiv:2509.24527v1).*
