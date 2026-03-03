# DreamerV4 Math Audit: Diffusion, Losses, Returns, and Heads

**Date:** 2026-03-02
**Reviewer:** Claude Opus 4.6 (automated audit)
**Scope:** `src/ahriuwu/models/diffusion.py`, `losses.py`, `returns.py`, `heads.py`
**Method:** Line-by-line comparison against DreamerV4 paper equations

---

## Severity Legend

- **BUG** -- Incorrect math that will produce wrong gradients or values at runtime
- **WARNING** -- Likely correct but fragile, or deviates from paper in a way that may cause subtle issues
- **STYLE** -- No functional impact, but confusing or could lead to future bugs

---

## 1. Flow Matching Convention (Eq. 1)

**Paper:** `x_tau = (1 - tau) * x_0 + tau * x_1`, where x_0 is noise, x_1 is data.
tau=0 is pure noise, tau=1 is clean data.

**Code convention (diffusion.py):** tau=0 is clean, tau=1 is noise. The code uses `z_0` for clean data and `noise` for N(0,I). The interpolation at line 58 is:

```python
z_tau = (1 - tau) * z_0 + tau * noise
```

This is a **flipped convention** from the paper (tau=0 gives clean, tau=1 gives noise). The code is internally consistent -- the ramp weight, loss, shortcut forcing, and sampler all use this flipped convention correctly. The docstrings explicitly document this choice (line 208-209).

| File | Line | Severity | Issue |
|------|------|----------|-------|
| `diffusion.py` | 4-6, 21-26, 207-209 | **STYLE** | Convention is inverted from paper (tau=0 clean vs paper tau=0 noise). Fully documented and internally consistent, but anyone reading the paper side-by-side will be confused. Consider a prominent warning at the module level mapping paper symbols to code symbols. |

---

## 2. Ramp Weight (Eq. 8)

**Paper:** `w(tau) = 0.9 * tau + 0.1` (paper convention: tau=1 is clean).
At tau=1 (clean): w=1.0. At tau=0 (noise): w=0.1.

**Code (line 220):**
```python
return 1.0 - 0.9 * tau
```
In code convention (tau=0 clean, tau=1 noise):
At tau=0 (clean): w=1.0. At tau=1 (noise): w=0.1.

This correctly maps the paper's intent: high weight at high signal, low weight at noise.

| File | Line | Severity | Issue |
|------|------|----------|-------|
| `diffusion.py` | 201-220 | OK | Ramp weight correctly adapts Eq. 8 to the inverted convention. No issues. |

---

## 3. x-prediction Loss

**Paper:** The base flow-matching loss is MSE between predicted clean data and actual clean data, weighted by w(tau).

**Code (lines 223-260):**
```python
mse = F.mse_loss(pred, target, reduction="none")
# Reduce spatial dims ...
if use_ramp_weight:
    weights = ramp_weight(tau)
    mse = mse * weights
```

| File | Line | Severity | Issue |
|------|------|----------|-------|
| `diffusion.py` | 246-249 | **WARNING** | **Spatial reduction may be off-by-one.** The while loop `while mse.dim() > tau.dim() + 1` followed by `mse = mse.mean(dim=-1)` reduces spatial dims. For a 5D tensor (B,T,C,H,W) with 2D tau (B,T): loop runs twice reducing H and W, then line 249 reduces C. This leaves (B,T) matching tau. For a 4D tensor (B,C,H,W) with 1D tau (B,): loop runs twice reducing H and W, then line 249 reduces C. This leaves (B,) matching tau. **Correct behavior.** However, the logic is fragile: it depends on the caller providing tau with the right dimensionality. If tau is accidentally (B,1) instead of (B,), the spatial reduction will stop one dim too early. |
| `diffusion.py` | 252-253 | **STYLE** | Ramp weight is applied per-sample (or per-sample-per-timestep), not per-spatial-element. This matches the paper since w(tau) is a scalar per noise level. Correct. |

---

## 4. Euler Sampling (lines 153-198)

**Code:**
```python
for i in range(num_steps):
    tau = 1.0 - i * step_size          # starts at 1.0 (noise)
    z_0_pred = model(z_t, tau_tensor)   # predict clean data
    next_tau = tau - step_size
    z_t = (1 - next_tau) * z_0_pred + next_tau * torch.randn_like(z_t)
```

| File | Line | Severity | Issue |
|------|------|----------|-------|
| `diffusion.py` | 193 | **BUG** | **Re-noising with fresh noise at each step.** The Euler step computes `z_t = (1 - next_tau) * z_0_pred + next_tau * torch.randn_like(z_t)`. This adds **fresh random noise** at each step rather than interpolating along the ODE trajectory. In standard flow matching / rectified flow, the Euler step should be: `z_t = z_t + step_size * velocity` where velocity = `(z_0_pred - z_t) / tau`. The current code essentially does **stochastic sampling** (like DDPM) rather than **deterministic ODE** (like flow matching Euler). This will work but produces noisier, less deterministic samples and requires more steps to converge. For x-prediction, the correct Euler update is: `z_t = z_t + step_size * (z_0_pred - z_t) / tau`, or equivalently, just re-interpolate the prediction at the next tau using the **same** noise that was used to create the current z_t (not fresh noise). |
| `diffusion.py` | 182-183 | **WARNING** | **First step starts at tau=1.0 exactly.** At tau=1.0 the input is pure noise and the model has zero signal -- this is the hardest prediction. Most implementations start at tau=1-eps or skip the first step. Not technically wrong, but the model's prediction at tau=1.0 may be very poor, wasting a step. |

**Suggested fix for the Euler ODE sampler:**
```python
# Correct ODE Euler for x-prediction:
z_t = torch.randn(shape, device=device)
for i in range(num_steps):
    tau = 1.0 - i * step_size
    tau_tensor = torch.full((shape[0],), tau, device=device)
    z_0_pred = model(z_t, tau_tensor, context=context)
    # Velocity: v = (x_1 - x_tau) / (1 - tau) in paper convention
    # In our convention: v = (z_0_pred - z_t) / tau
    if i < num_steps - 1:
        velocity = (z_0_pred - z_t) / max(tau, 1e-6)
        z_t = z_t - step_size * velocity  # step toward clean
    else:
        z_t = z_0_pred
```

Or equivalently, re-interpolate with the **original** noise (not fresh):
```python
z_noise = torch.randn(shape, device=device)
z_t = z_noise  # start at tau=1
for i in range(num_steps):
    tau = 1.0 - i * step_size
    z_0_pred = model(z_t, ...)
    next_tau = tau - step_size
    if next_tau > 0:
        z_t = (1 - next_tau) * z_0_pred + next_tau * z_noise  # SAME noise
    else:
        z_t = z_0_pred
```

---

## 5. Shortcut Forcing (Eq. 3/7)

### 5.1 Step Size Sampling (Eq. 4)

**Paper:** d ~ 1/U({1, 2, 4, 8, ..., K_max})

**Code (lines 341-350):**
```python
weights = torch.tensor([1.0 / d for d in self.step_sizes])
weights = weights / weights.sum()
idx = torch.multinomial(weights.expand(batch_size, -1), ...)
```

| File | Line | Severity | Issue |
|------|------|----------|-------|
| `diffusion.py` | 342-343 | **WARNING** | **Inverse weighting does not match paper.** The paper says `d ~ 1/U({1,2,4,...,K})` which means d is the **reciprocal** of a uniform draw from the set. If U is uniform over {1,2,4,...,64}, then P(d=1) = P(d=2) = ... = P(d=64) = 1/7 (there are 7 values: 1,2,4,8,16,32,64). The code instead uses P(d) proportional to 1/d, which gives P(d=1)=0.504, P(d=64)=0.008. **This dramatically over-samples small step sizes.** Re-reading the paper more carefully: "d is sampled from 1/U({1,2,...,K_max})" -- this actually means d = 1/k where k ~ Uniform({1,2,...,K_max}), so d takes values {1, 1/2, 1/3, ..., 1/K_max}. But since d must be on the grid of powers of 2, this likely means d is uniform over the set {1/K, 2/K, 4/K, ..., 1}. **The paper's notation is ambiguous.** The current 1/d weighting is a reasonable interpretation that samples smaller steps more often (which makes sense for progressive distillation), but may not match the paper's intent exactly. |

### 5.2 Tau Grid Sampling

**Paper:** tau ~ U({0, d/K, 2d/K, ..., 1-d/K}) for step size d.

**Code (lines 371-383):**
```python
grid_spacing = d / self.k_max
num_grid_points = self.k_max // d
grid_idx = torch.randint(0, num_grid_points, (n,), device=device)
tau[mask] = grid_idx.float() * grid_spacing
```

For d=1, k_max=64: grid_spacing = 1/64, num_grid_points = 64, tau in {0, 1/64, ..., 63/64}. Correct.
For d=64, k_max=64: grid_spacing = 1, num_grid_points = 1, tau = {0}. Correct.

| File | Line | Severity | Issue |
|------|------|----------|-------|
| `diffusion.py` | 371-383 | OK | Grid sampling is correct. Tau values land on the proper grid for each step size. |

### 5.3 Bootstrap Two-Step Computation

**Paper Eq. 7:** Take two half-steps with the teacher, average their velocities, use as target for the student.

**Code (lines 495-569):** Inside `with torch.no_grad():` block.

| File | Line | Severity | Issue |
|------|------|----------|-------|
| `diffusion.py` | 495-535 | OK | The `torch.no_grad()` block covers the teacher computation. The `avg_velocity.detach()` at line 546 provides an additional explicit detach. Both are correct -- the stop-gradient is properly applied. |
| `diffusion.py` | 500-512 | **BUG** | **First half-step: model is called as x-predictor but velocity is computed as if model outputs a denoised sample.** The code calls `z_mid = model(z_tau[idx], tau_idx, step_size=d_half_norm, ...)` -- this is the model's x-prediction (predicted clean data). Then `b_prime = (z_mid - z_tau[idx]) / tau_safe`. This computes `(x_hat_0 - x_tau) / tau`, which in the code's convention (tau=0 clean, tau=1 noise) is `(x_hat_clean - x_noisy) / tau`. This is the **velocity toward clean data** in the flow-matching sense. The paper's velocity (Eq. 7) is `v = (x_1 - x_tau) / (1 - tau)` (paper convention). Converting: paper's (1-tau) = code's tau. So the code's formula `(z_mid - z_tau) / tau` correctly maps to the paper's `(x_1 - x_tau) / (1-tau)`. **Correct.** |
| `diffusion.py` | 516-525 | **WARNING** | **Re-noising for second half-step uses original noise.** `z_tau_mid = (1 - tau_mid_expanded) * z_mid + tau_mid_expanded * noise[idx]`. This re-creates the noisy latent at the midpoint by interpolating the x-prediction `z_mid` with the **original noise** `noise[idx]` (from the initial `add_noise` call). This is correct -- it places the midpoint sample on the correct interpolation path assuming the teacher's prediction is perfect. |
| `diffusion.py` | 542 | **BUG** | **Student velocity reuses `tau_safe` from the teacher block.** `b_student = (z_pred - z_tau[idx]) / tau_safe` -- but `tau_safe` was computed inside the `with torch.no_grad()` block (line 511) and has `requires_grad=False`. Since `z_pred` requires grad (student output) and `z_tau[idx]` also doesn't (it's derived from data+noise), the gradient will flow through `z_pred` correctly via the numerator. The denominator `tau_safe` is a constant w.r.t. model parameters, so this is actually fine. **Not a bug after closer inspection.** However, there's a subtle issue: `tau_safe` was created inside `no_grad()`, so it won't be on the computation graph. But since tau is not a model parameter and we only need gradients w.r.t. model weights (through z_pred), this is correct. |
| `diffusion.py` | 560-568 | **BUG** | **tau-squared weight uses wrong exponent for inverted convention.** The paper says the bootstrap loss has a `(1-tau)^2` multiplier (paper Eq. 7), where paper's tau=0 is noise. In the code's convention (tau=0 clean, tau=1 noise), this should be `tau^2` (since code's tau corresponds to paper's (1-tau)). The code uses `tau_weight = tau_idx ** 2` which is correct. **However**, the paper's Eq. 7 actually says `(1 - sigma_tau)^2` where `sigma_tau` is the signal level. In the code's convention where tau IS the noise level (opposite of signal), `(1 - sigma_tau)^2 = tau^2`. So `tau_idx ** 2` is correct. **No bug.** |
| `diffusion.py` | 566-568 | **WARNING** | **2D tau case uses mean tau^2 per sample for weighting.** When tau is (B, T), the code computes `tau_weight = (tau_idx ** 2).mean(dim=-1)` and applies it as a per-sample scalar. This averages the tau^2 weight across timesteps rather than applying it per-token. The paper applies w(tau) per-token. This means tokens with low tau (near clean) get more weight than they should (pulled up by the average), and tokens with high tau (noisy) get less. |

### 5.4 Loss Combination

| File | Line | Severity | Issue |
|------|------|----------|-------|
| `diffusion.py` | 571-576 | **WARNING** | **Weighting base vs bootstrap loss by sample count.** `loss = (loss_std * n_std + loss_boot * n_boot) / total`. This weights the two losses proportionally to how many samples fell in each category. Since `loss_std` and `loss_boot` are already means over their respective subsets, this is equivalent to a weighted average, which is correct. However, it means the effective learning signal for shortcut forcing depends on the random split of the batch. With 1/d weighting, most samples will be d=1 (base step), so the bootstrap loss gets very little weight. This may slow shortcut distillation. |

---

## 6. Shortcut Forcing: Missing d_min Check

**Paper Eq. 3/7:** At d = d_min, use the standard flow-matching loss `||x_hat_1 - x_1||^2`. For d > d_min, use the bootstrap loss.

**Code:** `is_base_step = (step_size == 1)` at line 469. This checks for step_size == 1, which is d_min = 1.

| File | Line | Severity | Issue |
|------|------|----------|-------|
| `diffusion.py` | 469 | **WARNING** | Hardcoded `== 1` instead of `== self.k_min`. The `k_min` parameter exists (line 327) but is never used in `compute_loss`. If `k_min` is changed, this will silently break. Should be `step_size == self.k_min`. |

---

## 7. TimestepEmbedding (lines 263-312)

| File | Line | Severity | Issue |
|------|------|----------|-------|
| `diffusion.py` | 291-296 | **STYLE** | The sinusoidal embedding divides by `half_dim` (integer division). When `dim` is odd, `half_dim = dim // 2` and the final embedding has `2 * half_dim` dimensions, which is `dim - 1` if dim is odd. The MLP at line 275 expects `dim` input features. This would crash for odd dim. Should add an assertion or handle it. In practice, dim is always even (256, 512, etc.), so this is a latent bug. |

---

## 8. Tokenizer Loss (Eq. 5)

**Paper:** `L = L_MSE + 0.2 * L_LPIPS`

**Code (losses.py, lines 157-218):**
```python
self.mse_weight = mse_weight      # default 1.0
self.lpips_weight = lpips_weight   # default 0.2
total_loss = self.mse_weight * mse_loss + self.lpips_weight * lpips_loss
```

| File | Line | Severity | Issue |
|------|------|----------|-------|
| `losses.py` | 157-218 | OK | Weights match paper exactly: MSE + 0.2 * LPIPS. |
| `losses.py` | 53-55 | OK | LPIPS rescaling from [0,1] to [-1,1] is correct for the `lpips` library which expects [-1,1]. |
| `losses.py` | 204-208 | **STYLE** | Video input (B,T,C,H,W) is flattened to (B*T,C,H,W) before computing MSE and LPIPS. This means the loss is averaged over all frames equally, which is correct. |

---

## 9. VGGPerceptualLoss (lines 63-154)

| File | Line | Severity | Issue |
|------|------|----------|-------|
| `losses.py` | 95-100 | **BUG** | **VGG block extraction uses cumulative slicing but indices are absolute.** The code sets `prev_end = 0` and then for each `(start, end, name)` it does `nn.Sequential(*list(vgg.features.children())[prev_end:end])` and updates `prev_end = end`. But the `start` values (0, 4, 9, 16) are never used -- it always slices from `prev_end`. For the first block: `prev_end=0`, slices `[0:4]` -- correct (matches start=0, end=4). Second block: `prev_end=4`, slices `[4:9]` -- correct. Third: `prev_end=9`, slices `[9:16]` -- correct. Fourth: `prev_end=16`, slices `[16:23]` -- correct. **Actually correct**, but the `start` variable in the tuple is dead code. The cumulative approach works because the blocks are sequential. |
| `losses.py` | 96-100 | **STYLE** | The `start` variable from `block_indices` is unused. Remove it or assert `start == prev_end` to guard against mismatched indices. |

---

## 10. MAELoss (lines 221-327)

| File | Line | Severity | Issue |
|------|------|----------|-------|
| `losses.py` | 316-319 | **WARNING** | **LPIPS computed on full images even when MSE is patch-only.** The comment says "patches are too small for perceptual loss", which is reasonable. But this means the LPIPS loss includes both masked and unmasked regions, while MSE only counts masked patches. The gradients from LPIPS will push the model to reconstruct unmasked patches too (which it can already copy). This dilutes the LPIPS signal on masked regions. Consider masking the unmasked regions to neutral gray before LPIPS, or weighting LPIPS down when mask_indices is provided. |

---

## 11. symlog / symexp (returns.py, lines 15-41)

**Paper:**
- `symlog(x) = sign(x) * ln(|x| + 1)`
- `symexp(x) = sign(x) * (exp(|x|) - 1)`

**Code:**
```python
def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))

def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)
```

| File | Line | Severity | Issue |
|------|------|----------|-------|
| `returns.py` | 15-41 | OK | Matches paper exactly. `log1p` is used for numerical stability (good). These are exact inverses: `symexp(symlog(x)) = sign(x) * (exp(ln(|x|+1)) - 1) = sign(x) * (|x|+1-1) = sign(x)*|x| = x`. Verified. |

---

## 12. Twohot Encoding/Decoding (returns.py, lines 44-125)

### 12.1 Encoding

| File | Line | Severity | Issue |
|------|------|----------|-------|
| `returns.py` | 60-87 | OK | Twohot encoding distributes weight between two adjacent bins proportionally. Clamps to bucket range. Uses scatter for efficient GPU implementation. Mathematically correct. |
| `returns.py` | 68 | **STYLE** | `bucket_width = (high - low) / (num_buckets - 1)` assumes uniformly spaced buckets. This is correct given the linear spacing from `torch.linspace`. |

### 12.2 Decoding

| File | Line | Severity | Issue |
|------|------|----------|-------|
| `returns.py` | 90-104 | OK | Decoding computes weighted sum: `(softmax(logits) * bucket_centers).sum(-1)`. Standard and correct. |

### 12.3 Loss

| File | Line | Severity | Issue |
|------|------|----------|-------|
| `returns.py` | 107-125 | OK | Cross-entropy against soft two-hot targets. `-(target_twohot * log_softmax(logits)).sum(-1).mean()`. This is the standard soft cross-entropy loss. Correct. |

### 12.4 Bucket Configuration

| File | Line | Severity | Issue |
|------|------|----------|-------|
| `heads.py` | 27-30 | OK | 255 buckets from -20 to +20. These are in **symlog space**, so the actual value range is `symexp([-20, 20]) = [-exp(20)+1, exp(20)-1]` which is approximately +/- 485 million. This is more than sufficient for any reward scale. Paper uses 255 buckets. Matches. |

---

## 13. Lambda Returns (Eq. 10)

**Paper:** `R_t^lambda = r_t + gamma * c_t * [(1-lambda) * v_{t+1} + lambda * R_{t+1}^lambda]`
`R_T^lambda = v_T` (bootstrap)

**Code (lines 128-179):**
```python
next_return = values[:, -1]  # bootstrap
for t in range(T - 1, -1, -1):
    if t < T - 1:
        next_value = values[:, t + 1]
    else:
        next_value = values[:, t]  # Bootstrap

    next_return = rewards[:, t] + gamma * continues[:, t] * (
        (1 - lambda_) * next_value + lambda_ * next_return
    )
    returns[:, t] = next_return
```

| File | Line | Severity | Issue |
|------|------|----------|-------|
| `returns.py` | 161 | OK | Bootstrap: `next_return = values[:, -1]`. This sets `R_T = v_T`. Correct. |
| `returns.py` | 164-177 | **BUG** | **Last timestep (t=T-1) uses `next_value = values[:, t]` instead of `values[:, T]`.** At t=T-1, the paper wants `v_{t+1} = v_T`. The code uses `values[:, t] = values[:, T-1]` as the "bootstrap" value. But `values[:, T-1]` IS `v_T` if the sequence is 0-indexed and has T timesteps (indices 0 to T-1). So `values[:, T-1]` is the value at the last timestep. **However**, this means at t=T-1, the recursion computes: `R_{T-1} = r_{T-1} + gamma * c_{T-1} * [(1-lambda)*v_{T-1} + lambda*v_{T-1}]` = `r_{T-1} + gamma * c_{T-1} * v_{T-1}`. But the paper would have `R_{T-1} = r_{T-1} + gamma * c_{T-1} * [(1-lambda)*v_T + lambda*R_T]`. Since the code's `next_return` is initialized to `values[:, -1] = v_{T-1}` and `next_value = values[:, T-1] = v_{T-1}`, these are the same value: `v_{T-1}`. **The issue is that there's no separate `v_T` -- the sequence only has T timesteps, so `v_{T-1}` is the last available value prediction.** This is actually the standard Dreamer convention where the value at the last timestep serves as the bootstrap. **Not a bug per se**, but the comment "# Bootstrap" on line 169 is misleading -- it's doing exactly the right thing by using the last available value. |
| `returns.py` | 170-171 | **STYLE** | The `td_target` variable on line 171 is computed but **never used**. It's dead code. The actual return computation on lines 174-176 is the correct lambda-return formula. Remove the dead `td_target` computation. |

---

## 14. RunningRMS (returns.py, lines 209-251)

**Paper:** Losses are normalized by their running RMS.

**Code:**
```python
class RunningRMS:
    def __init__(self, decay=0.99):
        self.rms = None

    def update(self, value):
        value_sq = value.detach() ** 2
        if self.rms is None:
            self.rms = value_sq
        else:
            self.rms = self.decay * self.rms + (1 - self.decay) * value_sq
        return value / (torch.sqrt(self.rms) + 1e-8)
```

| File | Line | Severity | Issue |
|------|------|----------|-------|
| `returns.py` | 233 | OK | `value.detach() ** 2` -- detaching before squaring means the RMS statistics don't affect the gradient. Correct. |
| `returns.py` | 238 | OK | EMA formula: `rms = decay * rms + (1-decay) * value^2`. Standard exponential moving average. Correct. |
| `returns.py` | 240 | OK | Normalization: `value / (sqrt(rms) + eps)`. This divides by the root mean square with epsilon protection. Correct. |
| `returns.py` | 240 | **WARNING** | **The returned value retains its gradient** (the original `value` is divided, not `value.detach()`). This means the gradient is scaled by `1/rms`, which is the desired behavior -- normalized loss produces normalized gradients. Correct, but worth noting that the gradient magnitude depends on the RMS estimate. |
| `returns.py` | 235-236 | **WARNING** | **First-step initialization.** On the first call, `self.rms = value_sq`, so the normalization is `value / (|value| + eps) ~= sign(value)`. This means the first step's loss is effectively 1.0 regardless of actual magnitude. For the very first training step this may cause an unusually large or small gradient. Consider initializing with a reasonable default (e.g., 1.0) instead of None. |
| `returns.py` | 250 | **WARNING** | **Device mismatch on load.** `self.rms = torch.tensor(rms_val)` creates a CPU tensor. If training is on GPU, the next `update()` call will fail with a device mismatch when computing `self.decay * self.rms + (1 - self.decay) * value_sq`. Should use `torch.tensor(rms_val, device=value.device)` at update time, or save device info. |

---

## 15. Advantage Computation (returns.py, lines 182-206)

| File | Line | Severity | Issue |
|------|------|----------|-------|
| `returns.py` | 199-205 | **WARNING** | **Advantage normalization uses global mean/std.** `advantages = (advantages - mean) / std`. DreamerV4 normalizes advantages differently: per-batch mean and std, but importantly the paper uses **RunningRMS** to normalize advantages (or per-batch percentile normalization). Using batch-level mean/std is a common approach but may not match the paper exactly. Also, this normalizes to zero-mean which changes the sign of small advantages, potentially flipping positive to negative and vice versa. This matters for PMPO which splits on advantage sign. |

---

## 16. RewardHead (heads.py, lines 14-101)

| File | Line | Severity | Issue |
|------|------|----------|-------|
| `heads.py` | 63-65 | **WARNING** | **Output weights initialized with `std=0.01` but comment on line 62 says "True zero init breaks gradient flow."** The original DreamerV4 audit (line 269) says "Zero-init output weights: Matches". The paper uses zero initialization for the output layer. The comment claims zero init breaks gradient flow, but for a linear layer `y = Wx + b` with W=0 and b=0, the gradient `dL/dW = dL/dy * x^T` is nonzero as long as `dL/dy != 0` and `x != 0`. Zero init does NOT break gradient flow for linear layers (unlike zero init for all layers in a deep MLP, where it causes symmetry). The `std=0.01` is fine and nearly equivalent, but the comment is misleading. |
| `heads.py` | 73-86 | OK | MTP: stacks outputs from L separate heads. Shape (B,T,L,num_buckets). Matches paper. |
| `heads.py` | 97-101 | OK | Prediction: twohot_decode then symexp. Correct pipeline. |

---

## 17. PolicyHead (heads.py, lines 104-199)

| File | Line | Severity | Issue |
|------|------|----------|-------|
| `heads.py` | 104-199 | OK | Standard categorical policy with MTP. `log_prob` uses gather correctly. `sample` uses multinomial correctly. |
| `heads.py` | 174 | **STYLE** | `temperature == 0` uses `argmax`, but float comparison with 0 is fragile. Consider `temperature < 1e-8` or a dedicated greedy flag. |

---

## 18. ValueHead (heads.py, lines 202-277)

| File | Line | Severity | Issue |
|------|------|----------|-------|
| `heads.py` | 243-245 | **WARNING** | Same "zero init" comment issue as RewardHead. `std=0.01` with zeros bias. The actual paper may use true zero init; the deviation is small. |

---

## 19. PMPO (Eq. 11) -- NOT IMPLEMENTED

The PMPO objective is not present in any of the reviewed files. Per the existing audit document (DREAMERV4_AUDIT.md line 328), this is a known gap tagged as "Missing".

| File | Line | Severity | Issue |
|------|------|----------|-------|
| N/A | N/A | **WARNING** | PMPO is not implemented. When implementing, key details to get right: (1) Split on advantage sign, (2) alpha=0.5, beta=0.3, (3) KL direction is KL[pi || pi_prior] (reverse KL / mode-seeking), (4) pi_prior is a frozen copy of the behavioral policy from Phase 2. |

---

## 20. Division by (1-tau) Near tau=1

**Paper Eq. 7:** The velocity conversion involves dividing by tau (in code convention). When tau approaches 0 (clean data in code convention), division by near-zero occurs.

| File | Line | Severity | Issue |
|------|------|----------|-------|
| `diffusion.py` | 511 | OK | `tau_safe = tau_expanded.clamp(min=1e-6)` provides numerical protection. |
| `diffusion.py` | 553 | OK | `velocity_mse = velocity_mse.clamp(max=100.0)` provides additional overflow protection. |
| `diffusion.py` | 556-558 | OK | NaN/Inf check skips the entire bootstrap batch if detected. This is a reasonable safety measure. |

However, there is a subtlety:

| File | Line | Severity | Issue |
|------|------|----------|-------|
| `diffusion.py` | 383 | **WARNING** | **Tau grid can include tau=0.** For d=1: `grid_idx` ranges from 0 to 63, and `tau[mask] = grid_idx.float() * (1/64)`. Index 0 gives tau=0. At tau=0 (clean data), the velocity `(z_pred - z_tau) / tau` requires dividing by 0. The `tau_safe` clamp at 1e-6 handles this, but the resulting velocity at tau~0 is `(z_pred - z_0) / 1e-6`, which is enormous (since z_pred should be very close to z_0 at low noise). The velocity_mse clamp at 100 saves it, but this is a lossy workaround. Consider excluding tau=0 from the grid (start grid_idx at 1), or handling the d=1/tau=0 case separately. For the base step (d=1), tau=0 means "predict clean from clean" which is trivial and contributes no useful gradient. |

---

## Summary Table

| # | File | Line(s) | Severity | Short Description |
|---|------|---------|----------|-------------------|
| 1 | diffusion.py | 193 | **BUG** | Euler sampler uses fresh noise each step instead of ODE trajectory |
| 2 | diffusion.py | 342-343 | **WARNING** | Step size 1/d weighting may not match paper's sampling distribution |
| 3 | diffusion.py | 469 | **WARNING** | Hardcoded `== 1` instead of `== self.k_min` |
| 4 | diffusion.py | 566-568 | **WARNING** | 2D tau: mean tau^2 per sample instead of per-token weighting |
| 5 | diffusion.py | 383 | **WARNING** | Tau grid includes 0, causing velocity blowup (clamped but lossy) |
| 6 | diffusion.py | 182-183 | **WARNING** | Sampler starts at tau=1.0 exactly (hardest prediction, wasted step) |
| 7 | diffusion.py | 291-296 | **STYLE** | TimestepEmbedding breaks for odd dim (latent bug) |
| 8 | diffusion.py | 4-6 | **STYLE** | Inverted tau convention vs paper (documented but confusing) |
| 9 | losses.py | 95-100 | **STYLE** | `start` variable in VGG block indices is unused dead code |
| 10 | losses.py | 316-319 | **WARNING** | MAE LPIPS on full images dilutes gradient signal for masked regions |
| 11 | returns.py | 170-171 | **STYLE** | Dead `td_target` variable never used |
| 12 | returns.py | 235-236 | **WARNING** | RunningRMS first-step init makes first loss always ~1.0 |
| 13 | returns.py | 250 | **WARNING** | RunningRMS load_state_dict creates CPU tensor, device mismatch risk |
| 14 | returns.py | 199-205 | **WARNING** | Advantage normalization may not match paper; interacts with PMPO sign split |
| 15 | heads.py | 62-65 | **WARNING** | Comment claims zero init breaks gradient flow (incorrect for linear layers) |
| 16 | heads.py | 174 | **STYLE** | Float comparison `temperature == 0` is fragile |

### Critical Items (Fix Before Training)

1. **Euler sampler** (Issue #1): The stochastic re-noising approach will produce noisier samples and waste denoising steps. For evaluation/generation quality, switch to deterministic ODE sampling.

2. **RunningRMS device mismatch** (Issue #13): Will crash on checkpoint resume if training on GPU.

### Items to Fix Before Shortcut Forcing

3. **tau=0 in grid** (Issue #5): Exclude or handle specially to avoid velocity blowup.
4. **k_min hardcode** (Issue #3): Use `self.k_min` for future flexibility.
5. **Per-token tau^2 weighting** (Issue #4): Apply weight per-token for 2D tau, not averaged.

### Items to Fix Before Phase 3 (RL)

6. **Advantage normalization** (Issue #14): Align with paper's normalization before implementing PMPO.
7. **PMPO** (not implemented): Entire objective needs to be written.
