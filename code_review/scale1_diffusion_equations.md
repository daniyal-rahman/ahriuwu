# DreamerV4 Diffusion Equation Review

**Reviewed File**: `/Users/danirahman/Repos/ahriuwu/src/ahriuwu/models/diffusion.py`
**Review Date**: 2025-12-27
**Reviewer**: Claude Code (Opus 4.5)

---

## Executive Summary

The implementation uses an **inverted tau convention** compared to the DreamerV4 paper:
- **Paper**: tau=0 is NOISE, tau=1 is CLEAN
- **Implementation**: tau=0 is CLEAN, tau=1 is NOISE

This inversion affects how several equations must be interpreted. After careful analysis, **the ramp weight function has a critical bug** where the weight is applied in the opposite direction from paper intent.

---

## Convention Reference

| tau value | Paper interpretation | Implementation interpretation |
|-----------|---------------------|------------------------------|
| tau = 0   | Pure noise          | Clean data (z_0)             |
| tau = 1   | Clean data (z_1)    | Pure noise (epsilon)         |

The implementation explicitly documents this in the docstring (lines 17-26):
```python
"""Linear noise schedule for x-prediction diffusion.
...
At tau=0: z_tau = z_0 (clean)
At tau=1: z_tau = epsilon (pure noise)
"""
```

---

## Finding 1: Noise Schedule (Linear Interpolation)

### Paper Reference
DreamerV4 Equation 1: `x_tau = (1 - tau) x_0 + tau x_1`

Where in paper convention:
- x_0 is noise (tau=0 is noise)
- x_1 is clean data (tau=1 is clean)

### Implementation (lines 57-58)
```python
# Linear interpolation: z_tau = (1-tau)z_0 + tau*epsilon
z_tau = (1 - tau) * z_0 + tau * noise
```

### Analysis
The implementation formula produces:
- At tau=0: `z_tau = z_0` (clean)
- At tau=1: `z_tau = noise` (pure noise)

**Verdict**: CORRECT - The formula is mathematically correct given the inverted convention. The paper's `x_tau = (1-tau)x_0 + tau*x_1` with paper's convention (x_0=noise, x_1=clean) becomes `x_tau = (1-tau)*noise + tau*clean`, which at paper tau=1 gives clean data. The implementation with inverted convention achieves the same interpolation behavior.

| Location | File:Line |
|----------|-----------|
| Function | `add_noise()` |
| File | `/Users/danirahman/Repos/ahriuwu/src/ahriuwu/models/diffusion.py:57-58` |

**Severity**: N/A - Not a bug

---

## Finding 2: Ramp Weight Function - CRITICAL BUG

### Paper Reference
DreamerV4 Equation 8 (Section 3.3 - Ramp Loss Weight):

The paper states the ramp weight is designed to **focus learning on high-signal regions** (clean data) because:
> "Training on clean examples provides a better gradient signal since the model makes smaller errors on clean data where the task is easiest."

Paper formula: `w(tau) = 0.9*tau + 0.1`

In **paper convention** (tau=1 is clean):
- w(0) = 0.1 (low weight at noise, tau=0)
- w(1) = 1.0 (high weight at clean, tau=1)

This gives HIGH weight at HIGH signal (clean data), as intended.

### Implementation (lines 192-209)
```python
def ramp_weight(tau: torch.Tensor) -> torch.Tensor:
    """Compute ramp loss weight from DreamerV4.

    Higher weight at high signal levels (low tau) focuses learning
    on regions with more structure.

    Formula: w(tau) = 0.9*tau + 0.1

    At tau=0 (clean): w = 0.1    # <-- LOW weight at clean!
    At tau=1 (noise): w = 1.0    # <-- HIGH weight at noise!
    """
    return 0.9 * tau + 0.1
```

### Analysis
The implementation **directly copies the paper's formula without inverting it for the tau convention**.

In **implementation convention** (tau=0 is clean, tau=1 is noise):
- w(0) = 0.1 (low weight at clean, tau=0)
- w(1) = 1.0 (high weight at noise, tau=1)

**This is BACKWARDS from the paper's intent!**

The paper wants high weight on clean data to get better gradients. The implementation gives high weight on noisy data.

The docstring even acknowledges the intention ("Higher weight at high signal levels") but the implementation does the opposite.

### Correct Implementation Should Be
```python
def ramp_weight(tau: torch.Tensor) -> torch.Tensor:
    """Compute ramp loss weight from DreamerV4.

    Higher weight at high signal levels (low tau in our convention).

    At tau=0 (clean): w = 1.0 (high weight)
    At tau=1 (noise): w = 0.1 (low weight)
    """
    return 0.9 * (1 - tau) + 0.1  # Inverted for our convention
    # Or equivalently: 1.0 - 0.9 * tau
```

| Location | File:Line |
|----------|-----------|
| Function | `ramp_weight()` |
| File | `/Users/danirahman/Repos/ahriuwu/src/ahriuwu/models/diffusion.py:192-209` |

**Severity**: CRITICAL

**Impact**: Training loss is weighted incorrectly. The model is being trained to focus on noisy examples (where gradients are least useful) instead of clean examples (where gradients are most informative). This could significantly degrade model quality and convergence.

---

## Finding 3: X-Prediction Loss

### Paper Reference
X-prediction means predicting clean data z_0 directly from noisy input z_tau.
Loss is MSE between prediction and target, optionally weighted by ramp weight.

### Implementation (lines 212-249)
```python
def x_prediction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    tau: torch.Tensor,
    use_ramp_weight: bool = True,
    reduce: str = "mean",
) -> torch.Tensor:
    # MSE loss per element
    mse = F.mse_loss(pred, target, reduction="none")

    # Reduce spatial dimensions
    while mse.dim() > tau.dim() + 1:
        mse = mse.mean(dim=-1)
    mse = mse.mean(dim=-1)

    if use_ramp_weight:
        weights = ramp_weight(tau)
        mse = mse * weights
```

### Analysis
The loss computation itself is correct:
1. Computes element-wise MSE
2. Reduces spatial dimensions appropriately
3. Applies ramp weight per sample/timestep

**Verdict**: The loss computation is correct, but it inherits the bug from `ramp_weight()` when weighting is enabled.

| Location | File:Line |
|----------|-----------|
| Function | `x_prediction_loss()` |
| File | `/Users/danirahman/Repos/ahriuwu/src/ahriuwu/models/diffusion.py:212-249` |

**Severity**: HIGH (due to dependency on buggy ramp_weight)

---

## Finding 4: Diffusion Forcing Timesteps

### Paper Reference
DreamerV4 Section 3.4 describes diffusion forcing:
- Context frames have low noise (high signal)
- Target frames have increasing noise as they get further from context
- Creates temporal causality for autoregressive prediction

### Implementation (lines 85-142)
```python
def sample_diffusion_forcing_timesteps(
    self,
    batch_size: int,
    seq_length: int,
    device: torch.device | str | None = None,
    tau_ctx: float = 0.1,
    tau_max: float = 1.0,
) -> torch.Tensor:
    """Sample per-timestep noise levels for diffusion forcing.

    - Frames before h: low noise (tau_ctx, context frames)
    - Frames at/after h: increasing noise (prediction targets)
    """
```

### Analysis
The implementation correctly handles the convention:

1. **Context frames get tau_ctx=0.1** (lines 133-138):
   - In implementation convention, tau=0.1 means "slightly noisy but mostly clean"
   - This is correct - context should be nearly clean

2. **Target frames get increasing tau toward tau_max=1.0** (lines 127-142):
   - Target frames at horizon get tau_ctx (nearly clean)
   - Target frames further out get higher tau (more noise)
   - Final frame approaches tau_max=1.0 (full noise)
   - This is correct for the convention

3. **Random horizon sampling** (line 118):
   - Horizon uniformly sampled from [1, seq_length-1]
   - Ensures at least 1 context frame and 1 prediction frame
   - This matches paper intent

**Verdict**: CORRECT - The diffusion forcing implementation correctly handles the tau convention.

| Location | File:Line |
|----------|-----------|
| Function | `sample_diffusion_forcing_timesteps()` |
| File | `/Users/danirahman/Repos/ahriuwu/src/ahriuwu/models/diffusion.py:85-142` |

**Severity**: N/A - Not a bug

---

## Finding 5: Euler Sampling

### Paper Reference
Sampling goes from noise (high tau in impl) to clean (low tau in impl).

### Implementation (lines 144-189)
```python
def sample(self, model, shape, num_steps=64, ...):
    # Start from pure noise
    z_t = torch.randn(shape, device=device)

    # Euler integration from tau=1 to tau=0
    step_size = 1.0 / num_steps

    for i in range(num_steps):
        tau = 1.0 - i * step_size  # tau: 1.0 -> 0.0
        # ...
        if i < num_steps - 1:
            next_tau = tau - step_size
            z_t = (1 - next_tau) * z_0_pred + next_tau * torch.randn_like(z_t)
```

### Analysis
1. Starts at tau=1.0 (pure noise in impl convention) - CORRECT
2. Steps toward tau=0.0 (clean in impl convention) - CORRECT
3. Euler step: `z_t = (1 - next_tau) * z_0_pred + next_tau * noise`
   - As next_tau approaches 0, z_t approaches z_0_pred (clean prediction)
   - CORRECT

**Verdict**: CORRECT

| Location | File:Line |
|----------|-----------|
| Function | `sample()` |
| File | `/Users/danirahman/Repos/ahriuwu/src/ahriuwu/models/diffusion.py:144-189` |

**Severity**: N/A - Not a bug

---

## Finding 6: Eval Script Sampling Range

### Reference
The eval script (`eval_dynamics.py`) samples from tau=1.0 down to tau=tau_ctx (0.1) instead of tau=0.0.

### Implementation (eval_dynamics.py lines 371-374)
```python
# Stop at tau_ctx (0.1) not 0.0 - model never saw tau < 0.1 during training
tau_start = 1.0
tau_end = tau_ctx  # match training minimum
step_size = (tau_start - tau_end) / num_steps
```

### Analysis
This is a **deliberate design choice**, not a bug:
- During training with diffusion forcing, context frames have tau=0.1 (not tau=0)
- The model never sees tau < 0.1 during training
- At inference, stopping at tau=0.1 matches the training distribution
- This avoids train/test distribution mismatch

**Verdict**: CORRECT - This is a thoughtful design decision documented in progress_log.md

| Location | File:Line |
|----------|-----------|
| Function | `rollout_predictions()` |
| File | `/Users/danirahman/Repos/ahriuwu/scripts/eval_dynamics.py:371-374` |

**Severity**: N/A - Not a bug (intentional design)

---

## Finding 7: Shortcut Forcing Bootstrap Loss

### Paper Reference
DreamerV4 Section 3.5 describes shortcut forcing:
- For d=1: standard x-prediction loss
- For d>1: bootstrap loss - student takes 1 large step, teacher takes 2 half-steps

### Implementation (lines 341-440)
```python
def compute_loss(self, model, schedule, z_0, tau, step_size):
    # Denoising DECREASES tau toward 0
    half_step_amount = step_size[idx].float() / self.k_max / 2
    if tau.dim() > 1:
        tau_mid = (tau[idx] - half_step_amount.unsqueeze(-1)).clamp(min=0)
    else:
        tau_mid = (tau[idx] - half_step_amount).clamp(min=0)
```

### Analysis
The comment "Denoising DECREASES tau toward 0" is correct for the implementation convention:
- In impl convention: tau=1 is noise, tau=0 is clean
- Denoising goes from high tau to low tau
- The half-step calculation correctly subtracts from tau

**Verdict**: CORRECT - The shortcut forcing implementation handles the tau convention correctly.

| Location | File:Line |
|----------|-----------|
| Class | `ShortcutForcing.compute_loss()` |
| File | `/Users/danirahman/Repos/ahriuwu/src/ahriuwu/models/diffusion.py:341-440` |

**Severity**: N/A - Not a bug

---

## Summary Table

| Finding | Component | Equation | Severity | Status |
|---------|-----------|----------|----------|--------|
| 1 | add_noise() | Linear interpolation | N/A | Correct |
| 2 | ramp_weight() | Eq. 8 | **CRITICAL** | **BUG - inverted** |
| 3 | x_prediction_loss() | MSE + weight | HIGH | Correct (but uses buggy weight) |
| 4 | sample_diffusion_forcing_timesteps() | Diffusion forcing | N/A | Correct |
| 5 | sample() | Euler sampling | N/A | Correct |
| 6 | rollout_predictions() (eval) | Sampling range | N/A | Correct (intentional) |
| 7 | ShortcutForcing.compute_loss() | Bootstrap | N/A | Correct |

---

## Recommended Fix

**For Finding 2 (ramp_weight bug)**:

Change line 209 from:
```python
return 0.9 * tau + 0.1
```

To:
```python
return 0.9 * (1 - tau) + 0.1  # Inverted for impl convention (tau=0 clean, tau=1 noise)
```

Or equivalently:
```python
return 1.0 - 0.9 * tau
```

This gives:
- w(0) = 1.0 (high weight at clean, tau=0 in impl convention)
- w(1) = 0.1 (low weight at noise, tau=1 in impl convention)

Which matches the paper's intent of focusing learning on high-signal (clean) regions.

---

## Appendix: Convention Verification

The progress_log.md confirms the convention (line 238-241):
```
### Convention Note
Our codebase: tau=0 clean, tau=1 noise (denoising decreases tau)
Paper: tau=0 noise, tau=1 clean (denoising increases tau)
Implementation accounts for this difference.
```

The implementation mostly handles the convention correctly, but the ramp_weight function was a direct copy from the paper without inversion.
