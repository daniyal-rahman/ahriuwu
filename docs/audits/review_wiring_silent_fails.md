# Wiring & Silent Failure Audit

**Date:** 2026-03-02
**Scope:** DreamerV4 replication -- end-to-end data flow, loss wiring, freeze/unfreeze, numerical safety, checkpoint compatibility.
**Files audited:** dynamics.py, diffusion.py, train_dynamics.py, transformer_tokenizer.py, train_transformer_tokenizer.py, dataset.py, pretokenize_frames.py, train_agent_finetune.py, heads.py, returns.py, eval_dynamics.py, actions.py

---

## 1. REWARD INDEXING IS WRONG IN AGENT FINETUNING

**Severity:** BUG
**Files:** `scripts/train_agent_finetune.py` (line 300-331), `src/ahriuwu/data/dataset.py` (line 434-438)

**Description:**
In `ReplayDataset.__getitem__`, rewards are looked up using `start_idx` (the array index into the packed `.npz` file), but `self.reward_data` is keyed by original frame number. `start_idx` is the position in the numpy array, while `start_frame` is the actual frame number from the video. These are only equal when the packed array starts at frame 0 and has no gaps.

```python
# train_agent_finetune.py line 300
start_idx = seq_info['start_idx']   # <-- array index, e.g. 0, 1, 2...
...
# line 330-331
end_idx = min(start_idx + self.seq_len, len(video_rewards))
rewards = video_rewards[start_idx:end_idx]  # WRONG: should use start_frame
```

Meanwhile, `PackedLatentSequenceDataset._get_actions()` correctly uses `start_frame`:
```python
# dataset.py line 567
for t in range(start_frame, start_frame + self.sequence_length):
```

**Impact:** Actions and latents are correctly aligned, but rewards are misaligned with the corresponding frames. If video frames start at frame 100, then the rewards for frame 100 come from `video_rewards[0]` (the wrong frame's rewards). This silently trains the reward head on wrong targets.

**Fix:**
```python
# Change line 300 from:
start_idx = seq_info['start_idx']
# To:
start_frame = seq_info['start_frame']
# And use start_frame everywhere rewards are sliced (lines 330-331)
```

---

## 2. IMAGE SIZE MISMATCH: TOKENIZER TRAINING vs PRETOKENIZATION

**Severity:** BUG
**Files:** `scripts/train_transformer_tokenizer.py` (line 446), `src/ahriuwu/data/dataset.py` (line 15), `scripts/pretokenize_frames.py` (line 89), `src/ahriuwu/models/transformer_tokenizer.py` (line 941)

**Description:**
The transformer tokenizer is hardcoded to `img_size=256` (square). Three different image sizes are used across the pipeline:

| Stage | Size | Source |
|-------|------|--------|
| Tokenizer training | (480, 352) | `SingleFrameDataset` default `TARGET_SIZE` |
| Pretokenize | (256, 256) | `FrameDatasetForEncoding` hardcoded `target_size` |
| Tokenizer model | 256x256 | `TransformerTokenizer(img_size=256)` |

The tokenizer training uses `SingleFrameDataset(args.frames_dir)` with no override of `target_size`, so it defaults to `(480, 352)`. But the `TransformerTokenizer` expects 256x256 input (it does `(img_size // patch_size) ** 2` to compute num_patches). Feeding 480x352 images into a model expecting 256x256 will either crash (dimension mismatch at patch embedding) or silently produce wrong results if the model happens to accept the flattened tensor shape.

**Impact:** If training actually ran without error, it means patches are computed from the wrong spatial layout, corrupting the spatial structure of latents. If it crashed, this is just a configuration bug.

**Fix:**
Pass explicit `target_size=(256, 256)` when creating `SingleFrameDataset` in `train_transformer_tokenizer.py`:
```python
dataset = SingleFrameDataset(args.frames_dir, target_size=(256, 256))
```
Or change `TARGET_SIZE` in dataset.py to `(256, 256)` and make the pretokenize script consistent.

---

## 3. AGENT FINETUNING DOES NOT PASS ACTIONS TO DYNAMICS MODEL

**Severity:** BUG
**Files:** `scripts/train_agent_finetune.py` (line 502)

**Description:**
The dynamics model forward call in agent finetuning does NOT pass actions, even though the model supports action conditioning:

```python
# line 502
z_pred, agent_out = dynamics(z_noisy, tau, independent_frames=use_independent)
```

No `actions=` argument is passed. Meanwhile, Phase 1 dynamics training does pass actions (line 523 of `train_dynamics.py`):
```python
z_pred = model(z_tau, tau, actions=actions, independent_frames=use_independent)
```

If the dynamics model was pretrained WITH action conditioning (`use_actions=True`), it expects actions during finetuning to produce correct predictions. Without actions, it falls back to `no_action_embed` for every sample, which means the model can no longer leverage action information it learned during Phase 1 pretraining.

**Impact:** The dynamics model effectively forgets its action conditioning during agent finetuning. The dynamics loss degrades because the model is no longer given the information it was trained to rely on.

**Fix:**
Pass actions to the dynamics model. The `ReplayDataset` already provides a single-int action tensor. Convert it back to the factorized dict format or adapt the dynamics model to accept both formats.

---

## 4. DYNAMICS LOSS IN AGENT FINETUNING LACKS RAMP WEIGHTING

**Severity:** WARNING
**Files:** `scripts/train_agent_finetune.py` (line 505)

**Description:**
Phase 1 dynamics training uses `x_prediction_loss()` which applies ramp weighting (higher weight at low tau / high signal):
```python
# train_dynamics.py line 524
raw_loss = x_prediction_loss(z_pred, z_0, tau, use_ramp_weight=True)
```

But Phase 2 agent finetuning uses plain MSE:
```python
# train_agent_finetune.py line 505
dynamics_loss = F.mse_loss(z_pred, z_target)
```

This means:
1. No per-timestep weighting (all tau values weighted equally)
2. The tau tensor is not used in the loss at all, even though it has per-timestep structure from `sample_diffusion_forcing_timesteps()`

**Impact:** The dynamics loss gradient distribution changes between Phase 1 and Phase 2, potentially degrading the pretrained dynamics model's quality. The model was trained to focus on high-signal regions; now it equally weights pure-noise predictions.

**Fix:**
```python
from ahriuwu.models import x_prediction_loss
dynamics_loss = x_prediction_loss(z_pred, z_target, tau, use_ramp_weight=True)
```

---

## 5. EVAL_DYNAMICS USES SCALAR TAU FOR ALL FRAMES

**Severity:** WARNING
**Files:** `scripts/eval_dynamics.py` (lines 403-414)

**Description:**
During evaluation rollouts, the same scalar tau is used for ALL frames in the sequence:
```python
tau_tensor = torch.full((B,), tau, device=device)
# ...
z_pred = dynamics(input_seq, tau_tensor)
```

But during training, the model sees per-timestep tau values via `sample_diffusion_forcing_timesteps()`, which produces shape `(B, T)`. When tau is `(B,)`, the `TimestepEmbedding` produces `(B, D)` and this is broadcast identically across all frames. When tau is `(B, T)`, it produces `(B, T, D)` giving each frame its own conditioning.

The model is trained with per-timestep tau but evaluated with uniform tau. This means the model receives context frames (which should have low tau ~0.1) with the same tau as the predicted frame (which should have high tau ~1.0 at start). The model never saw this distribution during training.

**Impact:** Suboptimal evaluation quality. The model may produce blurrier or more confused predictions because the conditioning signal doesn't match training.

**Fix:**
Construct a proper per-timestep tau tensor: low tau for context frames, high tau for the frame being denoised:
```python
T_input = full_seq.shape[1] + 1
tau_per_frame = torch.full((B, T_input), tau_ctx, device=device)
tau_per_frame[:, -1] = tau  # only last frame gets current denoising tau
```

---

## 6. EVAL_DYNAMICS DOES NOT RESTORE QK_NORM, SOFT_CAP, REGISTER TOKENS

**Severity:** BUG
**Files:** `scripts/eval_dynamics.py` (line 291)

**Description:**
When loading the dynamics model for evaluation, the script uses default arguments:
```python
model = create_dynamics(model_size, latent_dim=latent_dim, use_actions=use_actions)
```

This uses default `use_qk_norm=True`, `soft_cap=50.0`, `num_register_tokens=8`. If the model was trained with different settings (e.g., `--no-qk-norm`, `--soft-cap 0`, or `--num-register-tokens 0`), the architecture won't match. Thanks to `strict=False`, mismatched keys silently get random initialization.

The checkpoint's `args` dict already contains all these settings but they are never read.

**Impact:** If training used non-default settings, the loaded model has wrong architecture + randomly initialized components. Results are meaningless but no error is raised.

**Fix:**
Read all architecture arguments from the checkpoint:
```python
args = checkpoint.get("args", {})
model = create_dynamics(
    model_size,
    latent_dim=latent_dim,
    use_actions=use_actions,
    use_qk_norm=args.get("use_qk_norm", True),
    soft_cap=args.get("soft_cap", 50.0),
    num_register_tokens=args.get("num_register_tokens", 8),
    num_kv_heads=args.get("num_kv_heads", None),
)
```

---

## 7. `strict=False` USED PERVASIVELY, MASKING LOADING ERRORS

**Severity:** WARNING
**Files:** `scripts/pretokenize_frames.py` (line 195), `scripts/train_agent_finetune.py` (line 421), `scripts/eval_dynamics.py` (lines 292, 324), `scripts/train_dynamics.py` (line 381)

**Description:**
Every checkpoint loading site uses `strict=False`. While this is needed for legitimate architecture upgrades (adding `step_embed`, `agent_tokens`, etc.), it also silently masks:
- Misspelled parameter names in saved checkpoints
- Accidentally removed components
- Architecture mismatches between train/eval (see issue #6)

The existing code does print warnings for missing/unexpected keys, but only in some sites. In `pretokenize_frames.py` and `train_agent_finetune.py`, the return values of `load_state_dict` are not checked at all.

**Impact:** If a checkpoint is incompatible (wrong model size, missing components), training/eval continues with randomly initialized parameters. The only symptom would be bad metrics, which is hard to diagnose.

**Fix:**
At minimum, add warnings at every site. Better: use `strict=True` by default and only fall back to `strict=False` with explicit expected missing/unexpected key lists.

---

## 8. OUTPUT PROJECTION ZERO-INITIALIZED BREAKS GRADIENT AT INIT

**Severity:** WARNING
**Files:** `src/ahriuwu/models/dynamics.py` (line 971)

**Description:**
```python
# Zero-init output projection for residual
nn.init.zeros_(self.output_proj.weight)
```

The output projection is zero-initialized. Since this is the final linear layer (not inside a residual block), at initialization the model predicts all zeros regardless of input. The first forward pass produces a loss, and gradients flow back through the zero weight matrix to update it. However, the initial gradient signal from this layer to earlier layers is zero (since forward activations are multiplied by zero weights), which means:

1. First few gradient updates only affect `output_proj.weight` itself
2. All earlier layers get zero gradient through this path

The `_init_weights` method runs AFTER this (since it's in `__init__`), and it does `xavier_uniform_(p, gain=0.02)` for all 2D weights -- but then `nn.init.zeros_` at line 971 overrides this for `output_proj`. Actually, looking more carefully, `_init_weights` runs first during `__init__`, then `nn.init.zeros_` is part of `_init_weights` itself.

Wait, looking again: `_init_weights` at line 962-971 does xavier for everything, then zero-inits output_proj. This means gradients WILL flow through the skip connections and other paths, just not through output_proj at first. This is actually a standard technique (zero-init residual) but here output_proj is NOT inside a residual -- it's the final projection. This means the model starts by predicting zeros, and the loss is dominated by reconstructing the mean.

**Impact:** Slow initial training. The model initially predicts zeros, which for centered latents (from tanh bottleneck, range [-1, 1]) gives high initial loss. Not catastrophic but delays convergence.

**Fix:** Use small random init instead of zeros:
```python
nn.init.normal_(self.output_proj.weight, std=0.01)
```

---

## 9. AGENT FINETUNING CREATES DYNAMICS WITH `use_actions=False`

**Severity:** WARNING
**Files:** `scripts/train_agent_finetune.py` (line 374)

**Description:**
The `load_pretrained_dynamics` function creates a new dynamics model but does NOT pass `use_actions`:
```python
dynamics = create_dynamics(
    size=model_size,
    latent_dim=latent_dim,
    use_agent_tokens=True,
    num_tasks=1,
    agent_layers=4,
    # ... no use_actions=True!
)
```

If the pretrained checkpoint was trained with `--use-actions`, the new model won't have action embedding layers. The pretrained `action_embed.*` weights will be logged as "Skipped" (line 397) and lost. This is related to issue #3 but is the root cause.

**Impact:** Action conditioning from Phase 1 is silently discarded during Phase 2.

**Fix:**
Auto-detect `use_actions` from the pretrained checkpoint:
```python
use_actions = any("action_embed" in k for k in pretrained_state.keys())
```
And pass it to `create_dynamics`.

---

## 10. MISSING `gradient_checkpointing` DEFAULT DISCREPANCY

**Severity:** WARNING
**Files:** `scripts/train_dynamics.py` (line 247-249), `scripts/train_agent_finetune.py` (line 206-208)

**Description:**
Both scripts set `gradient_checkpointing` as a `store_true` action with `default=True`:
```python
parser.add_argument(
    "--gradient-checkpointing",
    action="store_true",
    default=True,
)
```

`action="store_true"` combined with `default=True` means the flag is ALWAYS True regardless of command-line arguments. The user cannot disable gradient checkpointing. This is a Python argparse misuse: `store_true` sets the value to `True` when the flag is present, and `default=True` means it's also `True` when absent.

**Impact:** Users cannot disable gradient checkpointing even if they have enough GPU memory and want faster training. Not a correctness issue, but a usability bug.

**Fix:** Either:
- `action="store_true"` with `default=False` (disabled by default, enabled with flag)
- Or use `store_false` with `--no-gradient-checkpointing` to disable

---

## 11. RunningRMS STATE DEVICE MISMATCH ON RESUME

**Severity:** WARNING
**Files:** `src/ahriuwu/models/returns.py` (line 250)

**Description:**
When restoring `RunningRMS` state from a checkpoint:
```python
def load_state_dict(self, state: dict):
    rms_val = state.get("rms")
    self.rms = torch.tensor(rms_val) if rms_val is not None else None
```

The `torch.tensor(rms_val)` creates a CPU tensor. But during training, `update()` receives GPU tensors:
```python
def update(self, value: torch.Tensor) -> torch.Tensor:
    value_sq = value.detach() ** 2
    if self.rms is None:
        self.rms = value_sq
    else:
        self.rms = self.decay * self.rms + (1 - self.decay) * value_sq
```

After resuming, `self.rms` is on CPU but `value_sq` is on GPU. The expression `self.decay * self.rms + (1 - self.decay) * value_sq` will trigger a device mismatch error on the first training step after resume.

**Impact:** Crash on first step after resuming training (if RMS state was saved).

**Fix:**
```python
def load_state_dict(self, state: dict):
    rms_val = state.get("rms")
    self.rms = torch.tensor(rms_val) if rms_val is not None else None
    # Device will be fixed on first update() call
```
Better: move rms to the correct device in `update()`:
```python
if self.rms is not None and self.rms.device != value.device:
    self.rms = self.rms.to(value.device)
```

---

## 12. `x_prediction_loss` SHAPE REDUCTION OVER-REDUCES FOR 4D INPUT

**Severity:** WARNING
**Files:** `src/ahriuwu/models/diffusion.py` (lines 243-249)

**Description:**
The shape reduction logic is:
```python
while mse.dim() > tau.dim() + 1:
    mse = mse.mean(dim=-1)
mse = mse.mean(dim=-1)
```

For 5D input `(B, T, C, H, W)` with 2D tau `(B, T)`:
- mse starts at 5 dims, tau.dim()+1 = 3
- loop: mean(-1) -> 4 dims, then mean(-1) -> 3 dims, then exits (3 > 3 is false)
- final mean(-1): 3 -> 2 dims = `(B, T)` -- CORRECT

For 4D input `(B, C, H, W)` with 1D tau `(B,)`:
- mse starts at 4 dims, tau.dim()+1 = 2
- loop: mean(-1) -> 3 dims, then mean(-1) -> 2 dims, then exits
- final mean(-1): 2 -> 1 dim = `(B,)` -- CORRECT

This actually works correctly. But the final `mse.mean(dim=-1)` is somewhat confusing because for the 5D case it averages over the C dimension (last remaining spatial dim after H, W are reduced), and for the 4D case it averages over the H dimension. The semantic meaning differs.

**Impact:** Functionally correct but could produce unexpected weighting if dimensions change. No immediate bug.

---

## 13. TOKENIZER NOT FROZEN DURING DYNAMICS TRAINING (BY DESIGN)

**Severity:** STYLE (no issue)
**Files:** `scripts/train_dynamics.py`

**Description:**
During dynamics training, the tokenizer is not present at all -- latents are pre-computed offline by `pretokenize_frames.py`. The tokenizer runs in eval mode with `torch.no_grad()` during pretokenization. No gradients flow back through the tokenizer. This is correct.

During agent finetuning (`train_agent_finetune.py`), the tokenizer IS loaded and explicitly frozen:
```python
tokenizer.eval()
for param in tokenizer.parameters():
    param.requires_grad = False
```

This is also correct. When encoding raw frames during finetuning, `torch.no_grad()` wraps the encode call (line 477). No issues here.

---

## 14. ACTION ENCODING PRODUCES >128 VALUES BUT action_dim DEFAULTS TO 128

**Severity:** WARNING
**Files:** `src/ahriuwu/data/actions.py` (lines 125-162), `scripts/train_agent_finetune.py` (line 92)

**Description:**
The `encode_action()` function computes: `movement + 18 * (1 + priority_index)`. With 8 ability keys and 18 movement directions:
- Maximum unclamped value: `17 + 18 * (1 + 7) = 17 + 144 = 161`
- Clamped: `min(161, 127) = 127`

The clamping at 127 means different (movement, ability) combinations map to the same action index. For example:
- movement=1, ability=B (priority 7): `1 + 18*8 = 145 -> 127`
- movement=17, ability=item (priority 6): `17 + 18*7 = 143 -> 127`
- Any movement >= 2 with ability priority >= 6 maps to 127

The `PolicyHead` has `action_dim=128` (default from `--action-dim 128`), so valid indices are 0-127. The encoding technically fits, but many distinct actions collide at index 127.

**Impact:** The policy head cannot distinguish between different high-index ability+movement combinations. The behavioral cloning loss has aliased targets for these actions.

**Fix:** Either increase `action_dim` to 162 (18 * 9), or use a cleaner encoding like `movement * 9 + ability_index` which gives 18 * 9 = 162 distinct actions. Update `--action-dim` default to match.

---

## 15. `compute_lambda_returns` BOOTSTRAP AT TERMINAL STATE

**Severity:** WARNING
**Files:** `src/ahriuwu/models/returns.py` (lines 161-169)

**Description:**
```python
# Bootstrap from final value
next_return = values[:, -1]

# Compute returns backwards in time
for t in range(T - 1, -1, -1):
    if t < T - 1:
        next_value = values[:, t + 1]
    else:
        next_value = values[:, t]  # Bootstrap
```

When `t == T-1` (last timestep), `next_value = values[:, T-1]` (the same timestep's value). This means the lambda return at the last timestep is:
```
R_{T-1} = r_{T-1} + gamma * c_{T-1} * ((1-lambda) * V_{T-1} + lambda * V_{T-1})
        = r_{T-1} + gamma * c_{T-1} * V_{T-1}
```

This is a standard bootstrap from the value function. However, `continues[:, T-1]` should typically be 1.0 for non-terminal states, which means the bootstrapped value at the boundary always contributes. This is fine for the DreamerV3/V4 formulation but the variable `td_target` computed at line 171 is never used (it's computed but then the lambda-return overwrites `next_return` at line 174). The `td_target` variable is dead code.

**Impact:** Dead code, no functional impact. The lambda-return computation itself is correct.

---

## 16. SHORTCUT FORCING BOOTSTRAP USES SAME NOISE FOR RE-NOISING

**Severity:** WARNING
**Files:** `src/ahriuwu/models/diffusion.py` (line 525)

**Description:**
In the bootstrap loss computation:
```python
# Re-noise z_mid to tau_mid level for second half-step
z_tau_mid = (1 - tau_mid_expanded) * z_mid + tau_mid_expanded * noise[idx]
```

This re-uses the original noise (`noise[idx]`) to create `z_tau_mid` for the second half-step. The original noise was used to create `z_tau` from `z_0`. Using the same noise for re-noising creates correlation between the two half-steps -- the noise structure at `tau_mid` is not independent of the prediction at `tau`.

In the original shortcut forcing paper (Eq. 7), the re-noising at the midpoint should ideally use fresh noise to match the forward diffusion process. Using the same noise biases the velocity estimate.

**Impact:** Biased bootstrap targets for shortcut forcing. The velocity estimate from the two half-steps is correlated with the original noise, which may slow convergence of shortcut forcing but is unlikely to cause divergence.

**Fix:** Use fresh noise for re-noising:
```python
fresh_noise = torch.randn_like(z_mid)
z_tau_mid = (1 - tau_mid_expanded) * z_mid + tau_mid_expanded * fresh_noise
```

---

## 17. GradScaler HARDCODED TO "cuda" IN DYNAMICS TRAINING

**Severity:** WARNING
**Files:** `scripts/train_dynamics.py` (line 835)

**Description:**
```python
scaler = GradScaler("cuda")
```

But the device could be "mps" or "cpu". When running on MPS (Apple Silicon) or CPU, the GradScaler initialized with "cuda" will either error or silently do nothing useful.

Meanwhile, `train_transformer_tokenizer.py` correctly uses:
```python
scaler = GradScaler(device_type)
```

And `train_agent_finetune.py` also correctly uses:
```python
scaler = GradScaler(device_type)
```

**Impact:** Crash or silent performance issue when training dynamics on non-CUDA devices.

**Fix:**
```python
device_type = args.device.split(":")[0]
scaler = GradScaler(device_type)
```

---

## 18. EVAL ROLLOUT APPLIES CONTEXT NOISE TO PREDICTED FRAMES

**Severity:** STYLE
**Files:** `scripts/eval_dynamics.py` (lines 387-391)

**Description:**
During autoregressive rollout, previously predicted frames are re-noised before being used as context:
```python
pred_context = predicted[:, :frame_idx]
if tau_ctx > 0:
    pred_noise = torch.randn_like(pred_context)
    pred_context = (1 - tau_ctx) * pred_context + tau_ctx * pred_noise
```

This adds noise to already-denoised predictions. While this matches the training distribution (context frames always have tau_ctx noise), it also degrades the signal in an accumulating way over long rollouts: each frame's noise gets baked into subsequent predictions. The default `tau_ctx=0.1` means 10% noise is added at each step.

**Impact:** Quality degrades over long rollouts faster than necessary. This is a design choice, not a bug, but could be improved by reducing tau_ctx for predicted context frames.

---

## 19. REWARD HEAD AND POLICY HEAD BACKPROP THROUGH DYNAMICS

**Severity:** STYLE (matches DreamerV4 paper)
**Files:** `scripts/train_agent_finetune.py` (lines 502-570)

**Description:**
In agent finetuning, the reward and policy heads receive `agent_out` from the dynamics model:
```python
z_pred, agent_out = dynamics(z_noisy, tau, ...)
reward_logits = reward_head(agent_out)
action_logits = policy_head(agent_out)
```

All three losses (dynamics, reward, BC) are summed and backpropagated together:
```python
total_loss_batch = dynamics_loss_norm + reward_loss_norm + bc_loss_norm
scaler.scale(total_loss_batch).backward()
```

This means the reward and BC losses DO flow gradients back through the dynamics transformer (via `agent_out` -> agent blocks -> z tokens). In DreamerV4, this is intentional -- the dynamics model is jointly trained with the heads during Phase 2. The agent tokens attend to z tokens, so reward/BC gradients flow into the dynamics backbone.

**Impact:** This is correct per the DreamerV4 paper. The dynamics model adapts during finetuning, which is the intended behavior.

---

## 20. MTP REWARD/BC LOSS INDEXING HAS OFF-BY-ONE PATTERN

**Severity:** WARNING
**Files:** `scripts/train_agent_finetune.py` (lines 512-547)

**Description:**
The MTP (Multi-Token Prediction) loop for rewards:
```python
for offset in range(L):
    if offset < T - 1:
        target_idx = min(offset + 1, T - 1)
        target = reward_targets[:, target_idx:]  # (B, T - target_idx)
        pred = reward_logits[:, :T - target_idx, offset, :]
```

When `offset=0`: `target_idx = 1`, target = `reward_targets[:, 1:]`, pred = `reward_logits[:, :T-1, 0, :]`.
This means "head 0 at position t predicts reward at t+1". Correct for next-step prediction.

When `offset=1`: `target_idx = 2`, target = `reward_targets[:, 2:]`, pred = `reward_logits[:, :T-2, 1, :]`.
This means "head 1 at position t predicts reward at t+2". Correct.

When `offset=7` (L-1 for L=8): `target_idx = min(8, T-1)`. For T=32, target_idx=8.
This is correct as long as T > L.

However, the `min(offset + 1, T - 1)` clamp means that when `T` is small (e.g., T=8 and L=8), the last few heads predict the same target (the last frame's reward). This is wasteful but not wrong.

**Impact:** No bug, but for short sequences relative to `mtp_length`, later MTP heads all predict the same target, providing no additional training signal.

---

## 21. PRETOKENIZE USES float16 WHILE DYNAMICS EXPECTS float32

**Severity:** STYLE
**Files:** `scripts/pretokenize_frames.py` (line 227), `src/ahriuwu/data/dataset.py` (line 600)

**Description:**
Latents are saved as float16:
```python
latents = latents.cpu().numpy().astype(np.float16)
np.save(output_path, latents[i])
```

And loaded without explicit dtype conversion:
```python
latents = latents_array[start_idx:start_idx + self.sequence_length]
latents = torch.from_numpy(latents.copy())
```

When loaded from float16 numpy arrays, PyTorch creates float16 tensors. These are then moved to GPU and used in the dynamics model, which runs in mixed precision (bfloat16 on CUDA, float16 on MPS). The input projection `nn.Linear` will auto-cast in the autocast context.

**Impact:** No functional issue -- autocast handles the conversion. But float16 has less precision than float32 for the stored latents, which means about 0.1% quantization noise is baked into every latent. For the transformer tokenizer with tanh bottleneck (values in [-1, 1]), float16 has ~10 bits of mantissa, giving ~0.001 precision. This is acceptable.

---

## 22. DIFFUSION SAMPLING EULER STEP ADDS FRESH NOISE AT EACH STEP

**Severity:** STYLE
**Files:** `src/ahriuwu/models/diffusion.py` (line 193)

**Description:**
In the `DiffusionSchedule.sample()` method:
```python
z_t = (1 - next_tau) * z_0_pred + next_tau * torch.randn_like(z_t)
```

Fresh noise is sampled at each denoising step. This is the stochastic Euler sampler. The deterministic alternative (DDIM-style) would re-use the initial noise. The stochastic version adds diversity but slightly lower quality per step.

**Impact:** Design choice. The eval script also uses this stochastic approach. No bug.

---

## Summary by Severity

### BUG (3 issues -- will cause wrong training results)
1. **#1 - Reward indexing uses `start_idx` instead of `start_frame`**: Rewards are misaligned with frames during agent finetuning
2. **#2 - Image size mismatch**: Tokenizer trained on 480x352 but used on 256x256 (or vice versa)
3. **#6 - Eval ignores checkpoint architecture args**: Model loaded with wrong QKNorm/soft_cap/register settings

### WARNING (11 issues -- suboptimal but won't crash or silently corrupt)
4. **#3 - Agent finetuning drops action conditioning**: Dynamics model loses action information
5. **#4 - Missing ramp weighting in agent finetuning dynamics loss**
6. **#5 - Eval uses scalar tau instead of per-timestep**
7. **#7 - `strict=False` everywhere masks loading errors**
8. **#8 - Zero-init output projection slows convergence**
9. **#9 - `use_actions` not auto-detected from checkpoint in finetuning**
10. **#10 - `gradient_checkpointing` argparse default means it can't be disabled**
11. **#11 - RunningRMS CPU/GPU device mismatch on resume**
12. **#14 - Action encoding collision at index 127**
13. **#16 - Shortcut forcing reuses noise for re-noising**
14. **#17 - GradScaler hardcoded to "cuda" in dynamics training**

### STYLE (5 issues -- cosmetic or by-design)
15. **#12 - x_prediction_loss reduction semantics**
16. **#15 - Dead code in compute_lambda_returns**
17. **#18 - Eval noise accumulation in autoregressive rollout**
18. **#21 - float16 latent storage**
19. **#22 - Stochastic vs deterministic sampling**
