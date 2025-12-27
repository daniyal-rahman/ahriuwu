# DreamerV4 Training Loop Code Review

## Summary

This document reviews the training loop implementation in `/Users/danirahman/Repos/ahriuwu/scripts/train_dynamics.py` for correctness against the DreamerV4 paper "Training Agents Inside of Scalable World Models" (Hafner et al., 2025).

---

## 1. Diffusion Forcing Implementation

### 1.1 Per-Timestep Noise Levels

**Status: IMPLEMENTED CORRECTLY**

The implementation correctly applies per-timestep noise levels via diffusion forcing.

**Paper Reference:** Section 3.2 "Shortcut forcing", Equation (6)
> "The dynamics model takes the interleaved sequence of actions a = {a_t}, discrete signal levels tau = {tau_t} and step sizes d = {d_t}, and corrupted representations z_tilde = {z_t^(tau_t)} as input and predicts the clean representations z_1 = {z_t^1}"

**Code Location:** `/Users/danirahman/Repos/ahriuwu/scripts/train_dynamics.py:314`
```python
tau = schedule.sample_diffusion_forcing_timesteps(B, T, device=device)
```

**Code Location:** `/Users/danirahman/Repos/ahriuwu/src/ahriuwu/models/diffusion.py:85-142`

The `sample_diffusion_forcing_timesteps()` function correctly:
- Samples a random horizon h for each batch item
- Assigns low noise (tau_ctx=0.1) to context frames (before horizon)
- Assigns linearly increasing noise to target frames (from tau_ctx to tau_max)

---

### 1.2 Context Frame Noise Level (tau_ctx)

**Status: CORRECT - MATCHES PAPER**

**Paper Reference:** Section 3.2, end of section
> "We slightly corrupt the past inputs to the dynamics model to signal level tau_ctx = 0.1 to make the model robust to small imperfections in its generations."

**Code Location:** `/Users/danirahman/Repos/ahriuwu/src/ahriuwu/models/diffusion.py:90-91`
```python
tau_ctx: float = 0.1,
tau_max: float = 1.0,
```

**Severity: N/A (Correct)**

---

## 2. Loss Computation

### 2.1 X-Prediction Loss

**Status: IMPLEMENTED CORRECTLY**

The implementation uses x-prediction (predicting clean data) rather than v-prediction (predicting velocity).

**Paper Reference:** Section 3.2, Equation (7)
> "Instead, we found that parameterizing the network to predict clean representations, called x-prediction, enables high-quality rollouts of arbitrary length."

**Code Location:** `/Users/danirahman/Repos/ahriuwu/scripts/train_dynamics.py:328-329`
```python
z_pred = model(z_tau, tau)
loss = x_prediction_loss(z_pred, z_0, tau, use_ramp_weight=True)
```

**Severity: N/A (Correct)**

---

### 2.2 Ramp Loss Weight

**Status: IMPLEMENTED WITH INVERTED FORMULA**

**Issue Description:** The ramp weight formula in the implementation appears inverted compared to the paper's intent.

**Paper Reference:** Section 3.2, Equation (8)
> "w(tau) = 0.9*tau + 0.1"
> "Low signal levels contain less learning signal... To focus the model capacity on signal levels with the most learning signal, we propose a ramp loss weight that linearly increases with the signal level tau, where tau = 0 corresponds to full noise and tau = 1 to clean data"

**CRITICAL NOTE ON CONVENTION MISMATCH:**

The paper states:
- tau = 0 corresponds to **full noise**
- tau = 1 corresponds to **clean data**

But the implementation uses the OPPOSITE convention:
- tau = 0 corresponds to **clean data** (z_tau = z_0)
- tau = 1 corresponds to **full noise** (z_tau = noise)

**Code Location:** `/Users/danirahman/Repos/ahriuwu/src/ahriuwu/models/diffusion.py:24-26`
```python
# At tau=0: z_tau = z_0 (clean)
# At tau=1: z_tau = epsilon (pure noise)
```

**Code Location:** `/Users/danirahman/Repos/ahriuwu/src/ahriuwu/models/diffusion.py:192-209`
```python
def ramp_weight(tau: torch.Tensor) -> torch.Tensor:
    """...
    At tau=0 (clean): w = 0.1
    At tau=1 (noise): w = 1.0
    """
    return 0.9 * tau + 0.1
```

**Analysis:** Given the implementation's convention (tau=0 is clean, tau=1 is noise):
- The paper wants HIGH weight at HIGH signal (clean data)
- In implementation's convention, high signal = low tau
- So weight should be HIGH when tau is LOW
- Current formula: w(0) = 0.1 (low weight at clean) - THIS IS WRONG
- Should be: w(tau) = 0.9*(1-tau) + 0.1 = 1.0 - 0.9*tau

**Severity:** CRITICAL

**Suggested Fix:** Either:
1. Invert the tau convention throughout the codebase to match the paper, OR
2. Change the ramp weight formula to `w(tau) = 1.0 - 0.9 * tau` to give high weight to low-tau (clean) samples under the current convention

---

### 2.3 Loss Reduction

**Status: CORRECT**

The loss uses mean reduction which is standard.

**Code Location:** `/Users/danirahman/Repos/ahriuwu/src/ahriuwu/models/diffusion.py:244-245`
```python
if reduce == "mean":
    return mse.mean()
```

**Severity: N/A (Correct)**

---

## 3. Alternating Batch Lengths

### 3.1 Implementation Structure

**Status: IMPLEMENTED WITH DIFFERENT PARAMETERS**

**Paper Reference:** Section 3.4 "Efficient Transformer" - "Sequence length"
> "To support efficient training, we alternate training on many short batches and occasional long batches"

**Paper Reference:** Appendix A - Minecraft VPT dataset settings
> "We train the dynamics model with N_z = 256 spatial tokens, context length C = 192, and batch lengths T_1 = 64 and T_2 = 256."

**Code Location:** `/Users/danirahman/Repos/ahriuwu/scripts/train_dynamics.py:88-97`
```python
parser.add_argument(
    "--seq-len-short",
    type=int,
    default=32,  # Paper uses T_1=64
    help="Short sequence length for alternating (default 32 = 1.6 sec)",
)
parser.add_argument(
    "--seq-len-long",
    type=int,
    default=64,  # Paper uses T_2=256
    help="Long sequence length for alternating (default 64 = 3.2 sec)",
)
```

**Severity:** MEDIUM

**Issue Description:** Default sequence lengths differ from paper:
- Short: Implementation uses 32, paper uses 64
- Long: Implementation uses 64, paper uses 256

**Suggested Fix:** Update defaults to match paper values:
- `--seq-len-short` default = 64
- `--seq-len-long` default = 256

---

### 3.2 Long Batch Ratio

**Status: NO PAPER SPECIFICATION**

**Paper Reference:** Section 3.4 does not specify exact ratio

**Code Location:** `/Users/danirahman/Repos/ahriuwu/scripts/train_dynamics.py:110-115`
```python
parser.add_argument(
    "--long-ratio",
    type=float,
    default=0.1,
    help="Ratio of long batches (default 0.1 = 10%% long, 90%% short)",
)
```

**Severity:** LOW

**Issue Description:** The paper mentions "many short batches and occasional long batches" but doesn't specify exact ratio. The 10% default seems reasonable but cannot be verified against paper.

---

### 3.3 Alternating Selection Logic

**Status: IMPLEMENTED WITH RANDOM SELECTION**

**Code Location:** `/Users/danirahman/Repos/ahriuwu/scripts/train_dynamics.py:282-299`
```python
# Ratio-based selection: long_ratio chance of long batch
use_long = random.random() < args.long_ratio
```

**Severity:** LOW

**Issue Description:** The implementation uses random selection based on ratio. The paper doesn't specify whether alternating should be deterministic (e.g., every 10th batch is long) or probabilistic. Current probabilistic approach is acceptable.

---

## 4. Shortcut Forcing Integration

### 4.1 Step Size Sampling

**Status: CORRECT**

**Paper Reference:** Section 2 "Background" - Shortcut models, Equation (4)
> "d ~ 1/U({1, 2, 4, 8, ..., K_max})"

**Code Location:** `/Users/danirahman/Repos/ahriuwu/src/ahriuwu/models/diffusion.py:321-339`
```python
def sample_step_size(self, batch_size: int, device: torch.device) -> torch.Tensor:
    """Sample step sizes with inverse weighting.

    Larger step sizes are sampled less frequently:
    P(d) proportional to 1/d
    """
    # Compute inverse weights: P(d) proportional to 1/d
    weights = torch.tensor([1.0 / d for d in self.step_sizes], device=device)
```

**Severity:** N/A (Correct)

**Analysis:** The paper notation `d ~ 1/U({1, 2, 4, ...})` means "d is distributed as 1 over a uniform sample", which mathematically gives P(d) ∝ 1/d. The implementation correctly uses inverse weighting - larger step sizes are sampled less frequently.

---

### 4.2 Timestep Sampling for Shortcut Forcing

**Status: NOT FULLY IMPLEMENTED**

**Paper Reference:** Section 2, Equation (4)
> "tau ~ U({0, 1/d, ..., 1 - 1/d})"

The signal level tau should be sampled from a grid that depends on the step size d.

**Code Location:** `/Users/danirahman/Repos/ahriuwu/scripts/train_dynamics.py:314`
```python
tau = schedule.sample_diffusion_forcing_timesteps(B, T, device=device)
```

**Severity:** HIGH

**Issue Description:** When shortcut forcing is enabled, the tau sampling doesn't follow the grid-based sampling from Equation (4). The diffusion forcing timesteps are sampled independently of step size.

**Suggested Fix:** When shortcut forcing is enabled, tau should be sampled from the grid {0, 1/d, 2/d, ..., 1-1/d} where d is the step size.

---

### 4.3 Bootstrap Loss Computation

**Status: IMPLEMENTED WITH FORMULA DIFFERENCES**

**Paper Reference:** Section 3.2, Equation (7)
> "L(theta) = ||\hat{z}_1 - z_1||^2 if d = d_min"
> "L(theta) = (1-tau)^2 ||(\hat{z}_1 - \tilde{z})/(1-tau) - sg(b' + b'')/2||^2 else"

**Code Location:** `/Users/danirahman/Repos/ahriuwu/src/ahriuwu/models/diffusion.py:419-423`
```python
# Bootstrap loss: match the teacher's final prediction
loss_boot = F.mse_loss(z_pred, z_target.detach())
```

**Severity:** HIGH

**Issue Description:** The bootstrap loss in the implementation is a simple MSE between student prediction and teacher final prediction. However, the paper's formula (Equation 7) shows a more complex computation that:
1. Converts to v-space: (z_hat_1 - z_tilde)/(1-tau)
2. Targets the average of two velocities: sg(b' + b'')/2
3. Scales by (1-tau)^2

The implementation simplifies this to direct x-space MSE matching, which may not be equivalent.

**Suggested Fix:** Implement the bootstrap loss exactly as in Equation (7), accounting for the v-space conversion and (1-tau)^2 scaling factor.

---

## 5. Optimizer Settings

### 5.1 Optimizer Configuration

**Status: REASONABLE DEFAULTS**

**Code Location:** `/Users/danirahman/Repos/ahriuwu/scripts/train_dynamics.py:502-507`
```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.weight_decay,
    betas=(0.9, 0.95),
)
```

**Default values:**
- lr=1e-4
- weight_decay=0.01
- betas=(0.9, 0.95)

**Severity:** LOW

**Issue Description:** The paper doesn't specify optimizer hyperparameters. The values used are reasonable and commonly used for transformer training.

---

### 5.2 Gradient Clipping

**Status: IMPLEMENTED CORRECTLY**

**Code Location:** `/Users/danirahman/Repos/ahriuwu/scripts/train_dynamics.py:335-336`
```python
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Severity: N/A (Correct)**

Gradient clipping with max_norm=1.0 is standard practice for stable training.

---

## 6. Additional Observations

### 6.1 Noise Schedule Convention Mismatch

**Status: NEEDS ATTENTION**

Throughout the codebase, the noise convention is INVERTED from the paper:
- **Implementation:** tau=0 is clean, tau=1 is noise
- **Paper:** tau=0 is noise, tau=1 is clean

This affects multiple formulas:
1. Ramp weight (Section 2.2 above)
2. Noise interpolation formula
3. Bootstrap loss scaling

**Severity:** CRITICAL (architectural decision with cascading effects)

**Code Locations:**
- `/Users/danirahman/Repos/ahriuwu/src/ahriuwu/models/diffusion.py:21-26`
- `/Users/danirahman/Repos/ahriuwu/src/ahriuwu/models/diffusion.py:57-58`

**Current implementation:**
```python
# z_tau = (1 - tau) * z_0 + tau * noise
```

**Paper (Equation 1):**
```python
# x_tau = (1 - tau) * x_0 + tau * x_1  where x_1 is data, x_0 is noise
```

**Suggested Fix:** This is a fundamental architectural decision. Either:
1. Swap the convention throughout the entire codebase (significant refactor), OR
2. Keep current convention but carefully adjust all formulas that depend on it (ramp weight, bootstrap scaling, etc.)

---

### 6.2 Missing RMS Loss Normalization

**Status: NOT IMPLEMENTED**

**Paper Reference:** Section 3, Algorithm 1
> "To train a single dynamics transformer with multiple modalities and output heads, we normalize all loss terms by running estimates of their root-mean-square (RMS)."

**Code Location:** Loss normalization is not present in `train_dynamics.py`

**Severity:** LOW (for current single-loss training)

**Issue Description:** The paper mentions RMS normalization for multi-loss training. Since the current implementation only uses x-prediction loss, this is not critical. However, it will be needed when adding behavior cloning, reward modeling, etc.

**Suggested Fix:** Implement running RMS estimates for loss normalization when expanding to multi-loss training.

---

## Summary Table

| Finding | Severity | Paper Section | Code Location |
|---------|----------|---------------|---------------|
| Ramp weight formula inverted due to tau convention | CRITICAL | 3.2, Eq. 8 | diffusion.py:192-209 |
| Tau convention opposite from paper | CRITICAL | 2, Eq. 1 | diffusion.py:21-26, 57-58 |
| ~~Shortcut step size sampling~~ | ~~HIGH~~ | ~~2, Eq. 4~~ | ACTUALLY CORRECT - P(d)∝1/d matches paper |
| Bootstrap loss simplified (not matching Eq. 7) | HIGH | 3.2, Eq. 7 | diffusion.py:419-423 |
| Tau sampling doesn't use grid for shortcut forcing | HIGH | 2, Eq. 4 | train_dynamics.py:314 |
| Default sequence lengths differ from paper | MEDIUM | Appendix A | train_dynamics.py:88-97 |
| Missing RMS loss normalization | LOW | 3, Alg. 1 | train_dynamics.py |
| Long batch ratio not specified in paper | LOW | 3.4 | train_dynamics.py:110-115 |

---

## Recommendations

1. **Highest Priority:** Resolve the tau convention mismatch. The ramp weight formula gives LOW weight to clean data (which should get HIGH weight). This likely hurts training quality significantly.

2. **High Priority:** Fix shortcut forcing implementation:
   - Change step size sampling to uniform
   - Implement grid-based tau sampling
   - Match bootstrap loss formula exactly

3. **Medium Priority:** Update default sequence lengths to match paper (64/256 instead of 32/64)

4. **Low Priority:** Add RMS loss normalization infrastructure for future multi-loss training
