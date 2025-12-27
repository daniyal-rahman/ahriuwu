# ShortcutForcing Code Review - DreamerV4 Equation Verification

**Reviewer**: Claude Code Review
**Date**: 2024-12-27
**File**: `/Users/danirahman/Repos/ahriuwu/src/ahriuwu/models/diffusion.py`
**Class**: `ShortcutForcing` (lines 306-440)

---

## Convention Reference

**Paper Convention**: tau=0 is NOISE, tau=1 is CLEAN (denoising INCREASES tau)
**Implementation Convention**: tau=0 is CLEAN, tau=1 is NOISE (denoising DECREASES tau)

This convention inversion is acknowledged in the progress log and must be accounted for in all equations.

---

## Finding 1: Missing Tau Grid Sampling (Eq. 4)

**Severity**: CRITICAL

**Paper Reference**: Equation 4 - Shortcut Forcing Sampling
```
d ~ 1/U({1, 2, 4, 8, ..., K_max})     -- step size with inverse weighting
tau ~ U({0, 1/d, 2/d, ..., 1 - 1/d}) -- tau from GRID based on step size
```

**Issue Description**:
The paper specifies that tau must be sampled from a discrete grid aligned with the step size `d`. Specifically, tau can only take values `{0, 1/d, 2/d, ..., 1 - 1/d}`. This ensures that after taking a step of size `d`, tau lands exactly on another valid grid point.

The current implementation does NOT enforce this constraint. In `train_dynamics.py` (line 314), tau is sampled using `sample_diffusion_forcing_timesteps()` which produces continuous/arbitrary tau values. The `compute_loss()` method receives these arbitrary tau values without grid alignment.

**Code Location**:
- `diffusion.py:341-348` - `compute_loss()` accepts arbitrary tau
- `train_dynamics.py:314` - tau sampled without grid constraint

**Current Code**:
```python
# train_dynamics.py:314
tau = schedule.sample_diffusion_forcing_timesteps(B, T, device=device)
# ...
loss, loss_info = shortcut.compute_loss(model, schedule, z_0, tau, step_size)
```

**Expected Behavior**:
Tau should be sampled as:
```python
# For each step_size d (normalized to d_norm = d/k_max):
# tau in {0, d_norm, 2*d_norm, ..., 1 - d_norm}
grid_points = int(1.0 / d_norm)  # e.g., d=4, k_max=64 -> d_norm=1/16 -> 16 grid points
tau_idx = torch.randint(0, grid_points, (B,))
tau = tau_idx * d_norm
```

**Suggested Fix**:
Add a `sample_tau_for_step_size()` method in `ShortcutForcing` that samples tau from the appropriate grid given the step size, or modify `compute_loss()` to snap tau to the nearest grid point before use.

---

## Finding 2: Missing (1-tau)^2 Scaling Factor (Eq. 7)

**Severity**: CRITICAL

**Paper Reference**: Equation 7 - Bootstrap Loss
```
L = (1-tau)^2 || (z_hat_1 - z_tilde) / (1-tau) - sg(b' + b'')/2 ||^2
```

In the paper's convention (tau=0 noise, tau=1 clean), this is `(1-tau)^2`.
In our convention (tau=0 clean, tau=1 noise), this becomes `tau^2` scaling.

**Issue Description**:
The bootstrap loss should be weighted by `tau^2` (in our convention) to focus learning on noisier samples. The current implementation uses simple MSE without this scaling factor.

**Code Location**: `diffusion.py:423`

**Current Code**:
```python
# Bootstrap loss: match the teacher's final prediction
loss_boot = F.mse_loss(z_pred, z_target.detach())
```

**Expected Behavior**:
```python
# With tau^2 weighting (our convention)
tau_weight = tau[idx] ** 2  # or (1 - tau[idx]) ** 2 for paper convention equivalent
mse = F.mse_loss(z_pred, z_target.detach(), reduction='none')
# Reduce spatial dims and apply weight
loss_boot = (mse.mean(dim=(-3,-2,-1)) * tau_weight).mean()
```

**Suggested Fix**:
Apply the appropriate tau-dependent scaling factor to the bootstrap loss.

---

## Finding 3: Missing Velocity Conversion in Bootstrap (Eq. 7)

**Severity**: HIGH

**Paper Reference**: Equation 7 defines bootstrap targets using velocities:
```
b' = (f_theta(z_tilde, tau, d/2, a) - z_tau) / (1 - tau)
b'' = (f_theta(z', tau + d/2, d/2, a) - z') / (1 - (tau + d/2))
```

The paper computes VELOCITY targets (prediction - noisy state divided by 1-tau), averages them, then converts back.

**Issue Description**:
The current implementation directly uses x-predictions as targets, bypassing the velocity conversion step. This is a significant deviation from the paper's formulation.

In x-prediction, the model outputs clean data estimates. The paper converts these to velocity space, averages, then the student tries to match. The current code just takes the teacher's final x-prediction directly.

**Code Location**: `diffusion.py:391-423`

**Current Code**:
```python
with torch.no_grad():
    # First half-step: predict z_0 from z_tau
    z_mid = model(z_tau[idx], tau[idx] if tau.dim() > 1 else tau[idx], step_size=d_half_norm)
    # ...
    # Second half-step: predict z_0 from z_tau_mid
    z_target = model(z_tau_mid, tau_mid, step_size=d_half_norm)

# Student: take 1 full step directly
z_pred = model(z_tau[idx], tau[idx] if tau.dim() > 1 else tau[idx], step_size=d_normalized[idx])

# Bootstrap loss: match the teacher's final prediction
loss_boot = F.mse_loss(z_pred, z_target.detach())
```

**Expected Behavior (Paper Eq. 7)**:
```python
# First half-step velocity (our convention: divide by tau, not 1-tau)
b_prime = (z_mid - z_tau[idx]) / tau[idx]  # velocity at first step

# Second half-step velocity
b_double_prime = (z_target - z_tau_mid) / tau_mid  # velocity at second step

# Average velocities and create target
avg_velocity = (b_prime + b_double_prime) / 2

# Student velocity prediction
z_pred = model(z_tau[idx], ...)
b_student = (z_pred - z_tau[idx]) / tau[idx]

# Loss in velocity space with (tau)^2 scaling
loss_boot = (tau[idx]**2) * ||b_student - sg(avg_velocity)||^2
```

**Suggested Fix**:
Rewrite the bootstrap loss to operate in velocity space as specified in the paper.

---

## Finding 4: Intermediate State Calculation (Half-Step)

**Severity**: MEDIUM

**Paper Reference**: Section discussing half-step intermediate state:
```
z' = z_tau + b' * (d/2)
```

Where `b'` is the velocity after the first half-step.

**Issue Description**:
The current implementation re-noises `z_mid` (the x-prediction) to create the intermediate state for the second half-step. However, the paper formulation uses the velocity to step from the current noisy state.

**Code Location**: `diffusion.py:409-414`

**Current Code**:
```python
# Re-noise z_mid to tau_mid level for second half-step
# z_tau_mid = (1 - tau_mid) * z_mid + tau_mid * noise
if tau_mid.dim() == 1:
    tau_mid_expanded = tau_mid.view(-1, 1, 1, 1, 1)
else:
    tau_mid_expanded = tau_mid.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
z_tau_mid = (1 - tau_mid_expanded) * z_mid + tau_mid_expanded * noise[idx]
```

**Analysis**:
The current approach creates the intermediate state by:
1. Getting x-prediction `z_mid` (estimate of clean data)
2. Re-noising it to `tau_mid` level

The paper formulation would be:
1. Get velocity `b' = (z_mid - z_tau) / tau`
2. Step: `z' = z_tau + b' * (d/2)` where d/2 is half the step size

These are mathematically related but not identical due to the stochastic vs deterministic nature of the step.

**Suggested Fix**:
Consider implementing the deterministic velocity-based step as in the paper, or document why the stochastic re-noising approach is an intentional deviation.

---

## Finding 5: tau_mid Calculation Sign

**Severity**: MEDIUM

**Paper Reference**: After taking half-step of size d/2:
- Paper: `tau' = tau + d/2` (tau increases toward clean)
- Our convention: `tau_mid = tau - d/2` (tau decreases toward clean)

**Issue Description**:
The code calculates:
```python
half_step_amount = step_size[idx].float() / self.k_max / 2
tau_mid = (tau[idx] - half_step_amount).clamp(min=0)
```

This appears correct for our convention (tau=0 clean, tau=1 noise; denoising decreases tau).

**Code Location**: `diffusion.py:402-406`

**Analysis**:
The calculation is:
- `half_step_amount = d / k_max / 2` - This is d/(2*k_max), which is the normalized half-step size
- `tau_mid = tau - half_step_amount` - Subtracting moves toward clean (correct for our convention)

However, there's a potential issue: the paper uses unnormalized step sizes in units of tau. If `d=4` and `k_max=64`, then:
- Paper: take a step of d/k_max = 4/64 = 0.0625 in tau space
- Code: half_step_amount = 4/64/2 = 0.03125

This seems correct for a HALF step. But verify the full step size semantics match the paper.

**Status**: LIKELY CORRECT, but warrants verification against paper.

---

## Finding 6: Step Size Grid

**Severity**: LOW

**Paper Reference**: Equation 4 specifies step sizes as powers of 2:
```
d in {1, 2, 4, 8, ..., K_max}
```

**Code Location**: `diffusion.py:318-319`

**Current Code**:
```python
# Step sizes: 1, 2, 4, 8, ..., k_max
self.step_sizes = [2**i for i in range(int(torch.log2(torch.tensor(k_max)).item()) + 1)]
```

**Analysis**:
For k_max=64:
- `log2(64) = 6`
- Range: `[0, 1, 2, 3, 4, 5, 6]` (7 values)
- Step sizes: `[1, 2, 4, 8, 16, 32, 64]`

This matches the paper.

**Status**: CORRECT

---

## Finding 7: Inverse Weighting for Step Size Sampling

**Severity**: LOW

**Paper Reference**: Equation 4: `d ~ 1/U({1, 2, 4, 8, ..., K_max})` means P(d) proportional to 1/d.

**Code Location**: `diffusion.py:330-339`

**Current Code**:
```python
# Compute inverse weights: P(d) ∝ 1/d
weights = torch.tensor([1.0 / d for d in self.step_sizes], device=device)
weights = weights / weights.sum()

# Sample indices according to weights
idx = torch.multinomial(weights.expand(batch_size, -1), num_samples=1).squeeze(-1)
```

**Analysis**:
For step_sizes = [1, 2, 4, 8, 16, 32, 64]:
- Unnormalized weights: [1, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]
- Sum = 1.984375
- Normalized: [0.504, 0.252, 0.126, 0.063, 0.0315, 0.0158, 0.0079]

This correctly implements P(d) proportional to 1/d.

**Status**: CORRECT

---

## Summary

| Finding | Description | Severity | Status |
|---------|-------------|----------|--------|
| 1 | Missing tau grid sampling | CRITICAL | Needs fix |
| 2 | Missing (1-tau)^2 / tau^2 scaling | CRITICAL | Needs fix |
| 3 | Missing velocity space conversion | HIGH | Needs fix |
| 4 | Intermediate state via re-noising vs velocity step | MEDIUM | Review needed |
| 5 | tau_mid sign convention | MEDIUM | Likely correct |
| 6 | Step size grid powers of 2 | LOW | Correct |
| 7 | Inverse weighting P(d) proportional to 1/d | LOW | Correct |

---

## Recommendations

### Priority 1 (CRITICAL)
1. Implement tau grid sampling aligned with step size
2. Add velocity-space bootstrap loss with proper scaling factor

### Priority 2 (HIGH)
3. Convert bootstrap loss to operate in velocity space as per Eq. 7

### Priority 3 (MEDIUM)
4. Review intermediate state calculation for consistency with paper
5. Add unit tests verifying tau convention handling

---

## Additional Notes

### On the Implementation Approach

The current implementation takes a simplified approach:
- Teacher: make 2 x-predictions
- Student: match final x-prediction directly

The paper approach is:
- Teacher: compute 2 velocities, average them
- Student: match averaged velocity with scaled loss

The simplified approach may still work empirically but diverges from the paper's mathematical formulation. Consider testing both approaches.

### On Convention Consistency

The codebase correctly documents the tau convention inversion in the progress log. However, ensure ALL equations are properly translated:
- Paper's `(1-tau)` becomes our `tau`
- Paper's `tau` becomes our `(1-tau)`

This affects:
- Velocity calculation: `(pred - noisy) / tau` (our convention)
- Scaling factor: `tau^2` instead of `(1-tau)^2`
