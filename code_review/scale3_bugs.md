# Code-Level Bug Review: DreamerV4 Implementation

**Files Reviewed:**
- `/Users/danirahman/Repos/ahriuwu/src/ahriuwu/models/diffusion.py`
- `/Users/danirahman/Repos/ahriuwu/src/ahriuwu/models/dynamics.py`
- `/Users/danirahman/Repos/ahriuwu/src/ahriuwu/models/losses.py`

---

## 1. Tensor Shape Issues

### Bug 1.1: `add_noise()` tau dimension expansion is correct but fragile
**File:** `diffusion.py:54-55`
**Severity:** LOW

```python
while tau.dim() < z_0.dim():
    tau = tau.unsqueeze(-1)
```

**Description:** The while loop correctly expands tau dimensions, but it assumes tau always has fewer dimensions than z_0. If accidentally passed a pre-expanded tau with more dimensions than z_0, this becomes a no-op and the broadcasting will fail silently or produce wrong results.

**Reproduction Scenario:**
```python
z_0 = torch.randn(4, 8, 256, 16, 16)  # (B, T, C, H, W)
tau = torch.rand(4, 8, 1, 1, 1, 1)  # Already over-expanded
z_tau, _ = schedule.add_noise(z_0, tau)  # Will broadcast incorrectly
```

**Impact:** Incorrect noise application due to broadcasting misalignment.

---

### Bug 1.2: Reshape in `dynamics.forward()` assumes specific dimension ordering
**File:** `dynamics.py:386`
**Severity:** LOW

```python
x = z_tau.view(B, T, C, -1).permute(0, 1, 3, 2)  # (B, T, S, C)
```

**Description:** This correctly handles (B, T, C, H, W) -> (B, T, S, C), but `-1` inference requires H*W to equal `spatial_tokens`. The assertion on line 383 validates H==W==spatial_size, but if latent_dim was misconfigured, the view would silently succeed with wrong semantics.

**Reproduction Scenario:** Edge case where spatial_size is wrong but H*W happens to match expected token count.

---

### Bug 1.3: `x_prediction_loss` dimension reduction is order-dependent
**File:** `diffusion.py:236-238`
**Severity:** MEDIUM

```python
while mse.dim() > tau.dim() + 1:
    mse = mse.mean(dim=-1)
mse = mse.mean(dim=-1)  # Average over last dim
```

**Description:** This reduction assumes a specific structure. For (B, T, C, H, W) with tau (B, T), it does:
- (B, T, C, H, W) -> (B, T, C, H) -> (B, T, C) -> (B, T) via while loop
- Then final mean reduces (B, T) to (B,) which is WRONG - we want (B, T) to stay for per-timestep weighting.

Wait, actually checking: if tau.dim() is 2 (B, T), then tau.dim() + 1 = 3, so we reduce until mse.dim() > 3 is false, i.e., mse.dim() <= 3. So (B,T,C,H,W) dim=5 -> mean(-1) -> (B,T,C,H) dim=4 -> mean(-1) -> (B,T,C) dim=3. Then final mean reduces (B,T,C) -> (B,T). This is CORRECT.

For tau.dim() = 1 (B,): tau.dim() + 1 = 2, so we reduce until dim <= 2. (B,T,C,H,W) -> ... -> (B,T). Then final mean -> (B,). But tau is (B,), so weight shape (B,) * mse shape (B,) works.

Actually this is correct. Removing this bug.

---

## 2. Numerical Stability Issues

### Bug 2.1: No epsilon in softmax - potential overflow with large attention scores
**File:** `dynamics.py:106, 172, 178`
**Severity:** MEDIUM

```python
attn = F.softmax(attn, dim=-1)
```

**Description:** Standard softmax without numerical stability measures. While PyTorch's softmax implementation subtracts the max internally, extremely large attention scores (possible with head_dim=64, scale=0.125) could still cause issues in mixed precision training (bfloat16 has limited range).

**Reproduction Scenario:**
- Large batch with many spatial tokens
- Early training with uninitialized weights producing extreme values
- bfloat16 training with gradient scaling

**Impact:** NaN gradients during early training phases.

---

### Bug 2.2: RMSNorm epsilon may be too small for bfloat16
**File:** `dynamics.py:28, 34`
**Severity:** LOW

```python
def __init__(self, dim: int, eps: float = 1e-6):
    ...
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
```

**Description:** eps=1e-6 is typical for float32. For bfloat16, the minimum positive value is ~1e-38 but the precision is only ~7 bits of mantissa. An epsilon of 1e-6 should be safe, but 1e-8 (common in some implementations) would be problematic.

**Impact:** This is actually fine as-is. No action needed.

---

### Bug 2.3: Division by zero potential in diffusion forcing timesteps
**File:** `diffusion.py:130-131`
**Severity:** LOW

```python
max_distance = (seq_length - 1 - horizon).clamp(min=1).float()  # (B, 1)
normalized_dist = distance / max_distance  # (B, T) in [0, 1]
```

**Description:** The code correctly clamps max_distance to min=1, preventing division by zero. However, when horizon = seq_length - 1 (max possible value), max_distance becomes 1, and all target frames get the same normalized distance. This is technically correct behavior but may be surprising.

**Impact:** No bug, just edge case behavior note.

---

### Bug 2.4: Missing clamping in shortcut forcing half-step calculation
**File:** `diffusion.py:402-406`
**Severity:** HIGH

```python
half_step_amount = step_size[idx].float() / self.k_max / 2
if tau.dim() > 1:
    tau_mid = (tau[idx] - half_step_amount.unsqueeze(-1)).clamp(min=0)
else:
    tau_mid = (tau[idx] - half_step_amount).clamp(min=0)
```

**Description:** The clamp(min=0) is correct, but when tau is already low and step_size is large, tau_mid could become 0 for many samples. At tau=0, the re-noising operation at line 414:
```python
z_tau_mid = (1 - tau_mid_expanded) * z_mid + tau_mid_expanded * noise[idx]
```
becomes just `z_mid` (clean data), which may not be the intended behavior for the second half-step.

**Impact:** Training instability when bootstrap targets are computed from nearly-clean data.

---

### Bug 2.5: Potential -inf in causal mask with softmax
**File:** `dynamics.py:176-178`
**Severity:** LOW

```python
attn = attn.masked_fill(mask, float("-inf"))
attn = F.softmax(attn, dim=-1)
```

**Description:** When T=1 (single frame), the causal mask is 1x1 with no masked positions (diagonal=1 means upper triangle starting from diagonal+1). This is correct. However, if all positions in a row were masked (which can't happen with proper causal mask), softmax(-inf) would produce NaN.

**Impact:** Not a bug in this specific implementation since the causal mask always allows self-attention.

---

## 3. Gradient Flow Issues

### Bug 3.1: Missing `.detach()` on teacher predictions in bootstrap
**File:** `diffusion.py:417-423`
**Severity:** CRITICAL (but code is correct)

```python
with torch.no_grad():
    ...
    z_target = model(z_tau_mid, tau_mid, step_size=d_half_norm)

# ... later
loss_boot = F.mse_loss(z_pred, z_target.detach())
```

**Description:** The code uses BOTH `torch.no_grad()` for the teacher computation AND `.detach()` on z_target. This is redundant but safe. The `.detach()` is necessary because even though z_target was computed in no_grad, it may still hold references to the graph through `noise[idx]` slicing.

**Impact:** No bug - the implementation is correct and safe.

---

### Bug 3.2: Zero-init output projection prevents gradient flow initially
**File:** `dynamics.py:361`
**Severity:** LOW

```python
nn.init.zeros_(self.output_proj.weight)
```

**Description:** Zero-initializing the output projection is a common technique for residual blocks to start as identity. However, in this case, `output_proj` is NOT in a residual connection - the forward pass is:
```python
x = self.output_proj(x)  # (B, T, S, C)
```
The output is directly returned, not added as a residual. This means the model initially outputs zeros regardless of input, which requires other parts of the training (e.g., the loss) to provide strong gradients.

**Reproduction Scenario:**
```python
model = create_dynamics("small")
z_tau = torch.randn(2, 8, 256, 16, 16)
z_pred = model(z_tau, torch.rand(2))
print(z_pred.abs().mean())  # Will be ~0
```

**Impact:** Slow initial training, though the gradients will eventually update the projection. This is a design choice, not necessarily a bug, but worth noting.

---

### Bug 3.3: Positional embeddings may have gradient issues with slicing
**File:** `dynamics.py:392-393`
**Severity:** LOW

```python
x = x + self.spatial_pos[:, :, :self.spatial_tokens, :]
x = x + self.temporal_pos[:, :T, :, :]
```

**Description:** The slicing operations `:self.spatial_tokens` and `:T` create views of the parameter tensors. Gradients flow correctly through these slices in PyTorch. However, if `T > max_seq_len`, line 393 would silently truncate without error (since Python slicing doesn't raise on out-of-bounds).

**Reproduction Scenario:**
```python
model = DynamicsTransformer(max_seq_len=64)
z_tau = torch.randn(2, 128, 256, 16, 16)  # T=128 > max_seq_len=64
tau = torch.rand(2)
z_pred = model(z_tau, tau)  # Silently uses only first 64 positions
```

**Impact:** Model silently handles sequences longer than max_seq_len but with incorrect positional information.

---

## 4. Edge Cases

### Bug 4.1: `randint(1, seq_length)` excludes seq_length
**File:** `diffusion.py:118`
**Severity:** LOW

```python
horizon = torch.randint(1, seq_length, (batch_size,), device=device)
```

**Description:** `torch.randint(low, high, ...)` samples from [low, high), so horizon is in [1, seq_length-1]. This is intentional (ensures at least 1 context and 1 target frame), but the comment could be clearer.

When seq_length=2:
- horizon can only be 1
- Frame 0 is context, Frame 1 is target
- This is the minimum valid case

When seq_length=1:
- `randint(1, 1)` raises an error: "high must be greater than low"

**Reproduction Scenario:**
```python
schedule = DiffusionSchedule()
tau = schedule.sample_diffusion_forcing_timesteps(4, 1, device="cpu")  # CRASHES
```

**Impact:** Runtime error when using diffusion forcing with single-frame sequences.

---

### Bug 4.2: tau=0 and tau=1 edge cases
**File:** `diffusion.py:57-58`
**Severity:** LOW

```python
z_tau = (1 - tau) * z_0 + tau * noise
```

**Description:**
- At tau=0: z_tau = z_0 (clean data) - correct
- At tau=1: z_tau = noise (pure noise) - correct

These edge cases are handled correctly by the formula.

---

### Bug 4.3: batch_size=1 handling
**File:** `diffusion.py:335`
**Severity:** LOW

```python
idx = torch.multinomial(weights.expand(batch_size, -1), num_samples=1).squeeze(-1)
```

**Description:** When batch_size=1, the expand and multinomial work correctly. The squeeze(-1) converts (1, 1) -> (1,), which is correct.

**Impact:** No bug.

---

### Bug 4.4: Empty batch (batch_size=0) handling
**File:** Multiple locations
**Severity:** LOW

**Description:** Most operations would fail with batch_size=0, but this is an unrealistic scenario. PyTorch generally handles empty tensors, but the code doesn't explicitly check for it.

**Impact:** No practical impact - empty batches indicate data pipeline bugs.

---

## 5. Device/Dtype Consistency

### Bug 5.1: Tensor creation without explicit device in TimestepEmbedding
**File:** `diffusion.py:281-284`
**Severity:** MEDIUM

```python
freqs = torch.exp(
    -torch.log(torch.tensor(self.max_period, device=tau.device))
    * torch.arange(half_dim, device=tau.device)
    / half_dim
)
```

**Description:** The code correctly places tensors on `tau.device`. However, the `half_dim` divisor is a Python int, which is fine. The dtype of `freqs` will be float32 regardless of tau's dtype, which could cause dtype mismatches in mixed precision.

**Reproduction Scenario:**
```python
tau = torch.rand(4, dtype=torch.bfloat16, device="cuda")
emb = TimestepEmbedding(512)
emb = emb.to(dtype=torch.bfloat16, device="cuda")
result = emb(tau)  # freqs is float32, tau is bfloat16, may cause issues
```

**Impact:** Potential dtype mismatch in mixed precision training. The multiplication `tau.unsqueeze(-1) * freqs` would upcast to float32.

---

### Bug 5.2: VGGPerceptualLoss buffers dtype
**File:** `losses.py:64-69`
**Severity:** MEDIUM

```python
self.register_buffer(
    "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
)
self.register_buffer(
    "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
)
```

**Description:** These buffers are created as float32. If the model is moved to bfloat16 via `.to(dtype=...)`, these buffers should convert automatically. However, if only using `.half()` or manual dtype setting, mismatches could occur.

**Impact:** Low - PyTorch's `.to()` method handles registered buffers.

---

### Bug 5.3: ShortcutForcing loss tensor creation
**File:** `diffusion.py:376-377`
**Severity:** LOW

```python
loss_std = torch.tensor(0.0, device=z_0.device)
loss_boot = torch.tensor(0.0, device=z_0.device)
```

**Description:** These tensors are created as float32 regardless of z_0's dtype. When combined with actual losses (which may be bfloat16), the result will be upcast to float32. This is actually fine for loss computation (losses should be accumulated in higher precision).

**Impact:** No bug - this is good practice.

---

## 6. Off-by-One Errors

### Bug 6.1: Causal mask diagonal is correct
**File:** `dynamics.py:147-150`
**Severity:** None (confirmed correct)

```python
self.register_buffer(
    "causal_mask",
    torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool(),
)
```

**Description:** `torch.triu(..., diagonal=1)` keeps elements on and above the 1st diagonal (i.e., strictly upper triangular). This masks out future positions, allowing position i to attend to positions 0..i. This is the correct causal mask.

Example for 4x4:
```
[[0, 1, 1, 1],
 [0, 0, 1, 1],
 [0, 0, 0, 1],
 [0, 0, 0, 0]]
```
Position 0 can only attend to 0, position 1 to 0-1, etc. Correct.

---

### Bug 6.2: Position embedding indexing
**File:** `dynamics.py:392-393`
**Severity:** LOW

```python
x = x + self.spatial_pos[:, :, :self.spatial_tokens, :]
x = x + self.temporal_pos[:, :T, :, :]
```

**Description:** `self.spatial_pos` has shape (1, 1, spatial_tokens, model_dim), so `:self.spatial_tokens` is always the full tensor. `self.temporal_pos` has shape (1, max_seq_len, 1, model_dim), and `:T` correctly slices for the current sequence length.

**Impact:** No bug - slicing is correct.

---

## 7. Memory Leaks

### Bug 7.1: No tensor accumulation in loops
**File:** Multiple
**Severity:** None (no issues found)

**Description:** The code does not accumulate tensors in loops that would cause memory leaks:
- `diffusion.py` sampling loop (lines 173-188) uses in-place-ish reassignment
- Transformer blocks iterate over modules without accumulation

---

### Bug 7.2: Potential memory growth in `compute_loss`
**File:** `diffusion.py:341-440`
**Severity:** LOW

**Description:** The `compute_loss` method creates intermediate tensors for both standard and bootstrap loss computation. These are properly scoped and will be garbage collected. However, calling this method repeatedly without proper context management could cause memory fragmentation on GPU.

**Impact:** No leak, but potential fragmentation in long training runs.

---

## 8. Logic Errors

### Bug 8.1: ShortcutForcing step_sizes initialization
**File:** `diffusion.py:319`
**Severity:** MEDIUM

```python
self.step_sizes = [2**i for i in range(int(torch.log2(torch.tensor(k_max)).item()) + 1)]
```

**Description:** For k_max=64, log2(64)=6, so range(7) = [0,1,2,3,4,5,6], giving step_sizes=[1,2,4,8,16,32,64]. This is correct.

However, if k_max is not a power of 2, this breaks:
- k_max=50: log2(50)=5.64, int=5, range(6), step_sizes=[1,2,4,8,16,32]
- This misses step sizes between 32 and 50.

**Reproduction Scenario:**
```python
sf = ShortcutForcing(k_max=50)
print(sf.step_sizes)  # [1, 2, 4, 8, 16, 32] - missing larger steps
```

**Impact:** For non-power-of-2 k_max, the largest step sizes are not trainable.

---

### Bug 8.2: step_size broadcast mismatch in ShortcutForcing
**File:** `diffusion.py:400-401`
**Severity:** HIGH

```python
if step_size is not None:
    step_emb = self.step_embed(step_size)  # (B, D)
    time_emb = time_emb + step_emb  # additive combination
```

**Description:** When tau has shape (B, T), time_emb has shape (B, T, D). But step_size has shape (B,), so step_emb has shape (B, D). The addition `time_emb + step_emb` broadcasts (B, T, D) + (B, D), which requires step_emb to be (B, 1, D).

This will raise a runtime error:
```
RuntimeError: The size of tensor a (T) must match the size of tensor b (D) at non-singleton dimension 1
```

Wait, let me verify:
- time_emb: (B, T, D)
- step_emb: (B, D)

PyTorch broadcasting: (B, T, D) + (B, D) tries to align from the right:
- D matches D
- T vs B - misaligned!

This WILL cause a shape mismatch error when using per-timestep tau with step_size.

**Reproduction Scenario:**
```python
model = create_dynamics("small")
z_tau = torch.randn(2, 8, 256, 16, 16)  # (B, T, C, H, W)
tau = torch.rand(2, 8)  # Per-timestep tau
step_size = torch.rand(2)  # Step size per batch
z_pred = model(z_tau, tau, step_size=step_size)  # CRASHES
```

**Impact:** Shortcut forcing cannot work with per-timestep tau (diffusion forcing mode).

---

### Bug 8.3: VGGPerceptualLoss block accumulation
**File:** `losses.py:41-46`
**Severity:** MEDIUM

```python
prev_end = 0
for start, end, name in block_indices:
    block = nn.Sequential(*list(vgg.features.children())[prev_end:end])
    self.blocks.append(block)
    self.layer_names.append(name)
    prev_end = end
```

**Description:** The variable `start` is never used - the code always slices from `prev_end:end`. Looking at block_indices:
```python
block_indices = [
    (0, 4, "conv1_2"),    # Slice [0:4]
    (4, 9, "conv2_2"),    # Slice [4:9]
    (9, 16, "conv3_3"),   # Slice [9:16]
    (16, 23, "conv4_3"),  # Slice [16:23]
]
```

The slicing [prev_end:end] produces [0:4], [4:9], [9:16], [16:23], which matches the intended ranges. So the code is correct despite the unused `start` variable.

**Impact:** No bug, but dead code (unused `start` variable).

---

## Summary

| Severity | Count | Key Issues |
|----------|-------|------------|
| CRITICAL | 0 | - |
| HIGH | 2 | step_size broadcast mismatch (#8.2), ShortcutForcing tau_mid edge case (#2.4) |
| MEDIUM | 4 | softmax overflow potential (#2.1), TimestepEmbedding dtype (#5.1), VGG buffers dtype (#5.2), non-power-of-2 k_max (#8.1) |
| LOW | 10 | Various edge cases and minor issues |

### Priority Fixes:
1. **Bug 8.2 (HIGH)**: Add `.unsqueeze(1)` to step_emb before adding to per-timestep time_emb
2. **Bug 2.4 (HIGH)**: Consider adding a minimum tau threshold for bootstrap targets
3. **Bug 4.1 (LOW but crashable)**: Add check for seq_length > 1 in diffusion forcing

### Code Quality Notes:
- The codebase is generally well-structured with good documentation
- Most numerical stability considerations are handled correctly
- The main issues are around the interaction between shortcut forcing and diffusion forcing (per-timestep tau)
