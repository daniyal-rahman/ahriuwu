# Training Loop Audit - DreamerV4 Replication

**Date:** 2026-03-02
**Scope:** 5 training/preprocessing scripts
**Status:** READ-ONLY audit, no files modified

---

## Summary

| Severity | Count |
|----------|-------|
| BUG      | 7     |
| WARNING  | 14    |
| STYLE    | 5     |

---

## 1. `scripts/train_tokenizer.py`

### BUG-1: `train_epoch()` called with wrong arguments (will crash at runtime)

**File:** `scripts/train_tokenizer.py`, lines 384-387
**Severity:** BUG

The `train_epoch` function signature on line 220 expects 11 positional arguments:
```python
def train_epoch(model, dataloader, optimizer, scaler, scheduler, loss_fn, rms_trackers, device, epoch, global_step, args):
```

But `main()` calls it on line 384 without `scheduler` or `rms_trackers`:
```python
metrics = train_epoch(
    model, dataloader, optimizer, scaler, loss_fn, args.device,
    epoch, global_step, args
)
```

This means `loss_fn` is bound to the `scheduler` parameter, `args.device` (a string) to `loss_fn`, `epoch` to `rms_trackers`, etc. The script will crash immediately with a `TypeError` or produce nonsensical results.

**Suggested fix:** Pass all arguments correctly:
```python
metrics = train_epoch(
    model, dataloader, optimizer, scaler, scheduler, loss_fn, rms_trackers,
    args.device, epoch, global_step, args
)
```

### BUG-2: `save_checkpoint()` called with wrong arguments (will crash at runtime)

**File:** `scripts/train_tokenizer.py`, lines 406, 410, 416
**Severity:** BUG

The `save_checkpoint` function on line 143 expects 10 positional arguments including `scheduler` and `rms_trackers`:
```python
def save_checkpoint(model, optimizer, scaler, scheduler, rms_trackers, epoch, global_step, loss, args, path):
```

But all calls in `main()` omit `scheduler` and `rms_trackers`:
```python
save_checkpoint(model, optimizer, scaler, epoch, global_step, metrics["loss"], args, checkpoint_path)
```

This binds `epoch` (int) to `scheduler`, `global_step` (int) to `rms_trackers`, etc. The call will crash when it tries to call `.state_dict()` on an int.

**Suggested fix:** Pass `scheduler` and `rms_trackers` in all three call sites:
```python
save_checkpoint(model, optimizer, scaler, scheduler, rms_trackers, epoch, global_step, metrics["loss"], args, checkpoint_path)
```

### BUG-3: `load_checkpoint()` on resume doesn't restore scheduler or RMS state

**File:** `scripts/train_tokenizer.py`, lines 397-400
**Severity:** BUG

When resuming, `scheduler` and `rms_trackers` are not passed to `load_checkpoint`:
```python
start_epoch, global_step, _ = load_checkpoint(
    Path(args.resume), model, optimizer, scaler
)
```

The function signature supports optional `scheduler` and `rms_trackers` args. Without passing them, the scheduler restarts from scratch (wrong LR on resume) and RMS normalization is reset (loss spikes on resume).

**Suggested fix:**
```python
start_epoch, global_step, _ = load_checkpoint(
    Path(args.resume), model, optimizer, scaler, scheduler, rms_trackers
)
```

### WARNING-1: `GradScaler("cuda")` hardcoded -- breaks on MPS/CPU

**File:** `scripts/train_tokenizer.py`, line 386
**Severity:** WARNING

```python
scaler = GradScaler("cuda")
```

The device string should be derived from `args.device`, not hardcoded. Other scripts (e.g., `train_transformer_tokenizer.py` line 451) do this correctly:
```python
scaler = GradScaler(device_type)
```

### WARNING-2: `scheduler.step()` called per micro-batch (correct here, but fragile)

**File:** `scripts/train_tokenizer.py`, line 268
**Severity:** WARNING

`scheduler.step()` is called every batch, which is correct for a per-step `LambdaLR`. However, there is no gradient accumulation in this script, so this is fine now. If gradient accumulation is added later (as in the other scripts), `scheduler.step()` would need to move inside the `if (batch_idx + 1) % accumulation_steps == 0` block.

### WARNING-3: Weight decay hardcoded in optimizer, ignoring `args.weight_decay`

**File:** `scripts/train_tokenizer.py`, line 359
**Severity:** WARNING

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
```

The weight decay is hardcoded to `0.01` while `args.weight_decay` is parsed as `0.1`. The argument is ignored. If using 8-bit Adam (line 366) it correctly uses `args.weight_decay`, creating an inconsistency depending on which optimizer branch executes.

---

## 2. `scripts/train_dynamics.py`

### WARNING-4: `GradScaler("cuda")` hardcoded

**File:** `scripts/train_dynamics.py`, line 835
**Severity:** WARNING

Same issue as WARNING-1. Should use `args.device.split(":")[0]`.

### WARNING-5: GradScaler used with bfloat16 is a no-op (not a bug, but wasteful)

**File:** `scripts/train_dynamics.py`, lines 498-499, 534, 835
**Severity:** WARNING

When `amp_dtype = torch.bfloat16` (the CUDA path), `GradScaler` is unnecessary. With bf16, gradients don't suffer from the dynamic range issues that fp16 has. The scaler will be a no-op (it detects bf16 autocast and skips scaling internally in recent PyTorch), but:
1. It adds code complexity with `scaler.scale()`, `scaler.unscale_()`, `scaler.step()`, `scaler.update()` calls.
2. On older PyTorch versions, the scaler might interfere incorrectly.

This applies to all scripts using bf16 + GradScaler: `train_dynamics.py`, `train_transformer_tokenizer.py`, `train_agent_finetune.py`.

**Suggested fix:** For bf16, skip the scaler entirely and just call `loss.backward()`, `optimizer.step()` directly. Or use `GradScaler(enabled=False)` to make the no-op explicit.

### WARNING-6: No LR scheduler -- constant LR throughout training

**File:** `scripts/train_dynamics.py` (entire file)
**Severity:** WARNING

Unlike `train_tokenizer.py` which has a WSD schedule, the dynamics training uses a constant learning rate with no warmup or decay. DreamerV4 Section 3.2 suggests using a warmup schedule. For long training runs this may lead to instability in early steps.

### WARNING-7: No validation/eval loop

**File:** `scripts/train_dynamics.py` (entire file)
**Severity:** WARNING

There is no validation split and no evaluation during training. The `best_loss` is tracked on training loss only. This makes it impossible to detect overfitting. Given the small dataset (~18 replays), overfitting risk is high.

### WARNING-8: Checkpoint does not save LR scheduler state (none exists)

**File:** `scripts/train_dynamics.py`, lines 343-367
**Severity:** WARNING (informational)

The `save_checkpoint` and `load_checkpoint` functions don't handle a scheduler because none is used. If a scheduler is added later, the checkpoint code will need updating. This is consistent with the current code but noted for future work.

### WARNING-9: `global_step` increments per micro-batch, not per optimizer step

**File:** `scripts/train_dynamics.py`, line 553
**Severity:** WARNING

```python
global_step += 1  # Increments every micro-batch
```

With `gradient_accumulation > 1`, `global_step` counts micro-batches rather than optimizer steps. This affects:
1. **Step-based checkpoint saving** (line 615): `global_step % args.save_steps` triggers at micro-batch frequency, not optimizer step frequency.
2. **Logging**: The step count in checkpoints doesn't reflect actual optimizer updates.

DreamerV4's convention is to count optimizer steps. With `--gradient-accumulation 4`, the effective step count is 4x inflated.

**Suggested fix:** Only increment `global_step` when `did_step` is True, or track `optimizer_step` separately.

### STYLE-1: Alternating lengths uses random sampling, not deterministic alternating

**File:** `scripts/train_dynamics.py`, lines 454-476
**Severity:** STYLE

The DreamerV4 paper says "80% short + 20% long". The implementation uses `random.random() < args.long_ratio` per batch (line 456), which gives the correct expected ratio over many batches but does not guarantee deterministic alternation. This is actually fine -- stochastic selection with the right probability is equivalent in expectation and simpler to implement.

The default `long_ratio=0.1` (10% long) differs from the paper's 20%. This is configurable so not a bug, just worth noting.

### STYLE-2: Long dataloader silently resets on exhaustion

**File:** `scripts/train_dynamics.py`, lines 461-464
**Severity:** STYLE

When the long dataloader is exhausted, it silently resets (`iter_long = iter(dataloader_long)`). The epoch ends only when the short loader is exhausted. This means long sequences may be seen more than once per epoch if there are few of them. This is intentional behavior but could surprise users.

### WARNING-10: Potential stale `z_pred` reference in prediction std logging

**File:** `scripts/train_dynamics.py`, lines 556-563
**Severity:** WARNING

```python
if batch_idx % 10 == 0:
    with torch.no_grad():
        if shortcut is None:
            pred_std = z_pred.std().item()
```

`z_pred` is defined inside the `autocast` block (line 523) only in the non-shortcut path. When `shortcut is not None`, this code correctly skips it. However, `z_pred` could also reference the previous iteration's tensor if the current batch took the shortcut path but a prior batch didn't. In practice this won't happen because `shortcut` is constant per run, but the code is fragile.

---

## 3. `scripts/train_transformer_tokenizer.py`

### WARNING-11: Loss division by `accumulation_steps` is inside autocast

**File:** `scripts/train_transformer_tokenizer.py`, line 309
**Severity:** WARNING

```python
with autocast(device_type=device_type, dtype=dtype):
    ...
    loss = (mse_norm + args.lpips_weight * lpips_norm) / accumulation_steps
```

The division by `accumulation_steps` happens inside the `autocast` context. While this is mathematically correct (division is fine in reduced precision), some frameworks recommend keeping the scaling outside autocast for clarity. In this case it works because the values are scalars and bf16 has sufficient range.

### WARNING-12: No end-of-epoch gradient flush for accumulation

**File:** `scripts/train_transformer_tokenizer.py`, lines 280-360
**Severity:** WARNING

Unlike `train_dynamics.py` (lines 630-636), this script does NOT flush remaining accumulated gradients at the end of the epoch:

```python
# train_dynamics.py has this, train_transformer_tokenizer.py does not:
if batch_idx % accumulation_steps != 0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

If the number of batches is not divisible by `accumulation_steps`, the last few micro-batches' gradients are silently discarded. With `drop_last=True` in the DataLoader and typical batch counts, this could lose 1-3 micro-batches per epoch.

**Suggested fix:** Add the same flush logic at the end of `train_epoch()`.

### WARNING-13: `optimizer.zero_grad()` not called at start of epoch

**File:** `scripts/train_transformer_tokenizer.py`, line 280
**Severity:** WARNING

Unlike `train_dynamics.py` (line 450) which calls `optimizer.zero_grad()` before the training loop, this script relies on the optimizer being zero'd from the previous epoch's last step or from initialization. If the previous epoch ended with an incomplete accumulation cycle, stale gradients from the previous epoch will leak into the first micro-batch of the new epoch.

**Suggested fix:** Add `optimizer.zero_grad()` before the batch loop.

### STYLE-3: No random seed set for reproducibility

**File:** `scripts/train_transformer_tokenizer.py` (entire file)
**Severity:** STYLE

No `torch.manual_seed()`, `random.seed()`, or `np.random.seed()` is called. The mask ratio sampling (line 295) and DataLoader shuffling are non-reproducible. This applies to all 5 scripts.

---

## 4. `scripts/train_agent_finetune.py`

### BUG-4: No gradient accumulation support

**File:** `scripts/train_agent_finetune.py`, lines 438-615
**Severity:** BUG (design gap)

Unlike the other training scripts, `train_agent_finetune.py` has no `--gradient-accumulation` argument and no accumulation logic. The dynamics + reward + BC heads are all trained end-to-end with the full batch, which may cause OOM for larger models. This is more of a missing feature than a bug, but given that the dynamics model requires gradient accumulation, the finetuning will likely also need it.

### BUG-5: MTP reward loss indexing is incorrect

**File:** `scripts/train_agent_finetune.py`, lines 511-527
**Severity:** BUG

The MTP (Multi-Token Prediction) reward loss loop has a subtle indexing issue:

```python
for offset in range(L):
    if offset < T - 1:
        target_idx = min(offset + 1, T - 1)
        target = reward_targets[:, target_idx:]     # (B, T - target_idx)
        pred = reward_logits[:, :T - target_idx, offset, :]  # (B, T - target_idx, buckets)
```

The variable `target_idx = min(offset + 1, T - 1)` is meant to represent the prediction offset. For `offset=0`, `target_idx=1` (predict next timestep). For `offset=1`, `target_idx=2`, etc. This is correct.

However, for `offset >= T - 1`, the guard `if offset < T - 1` skips the iteration. With `L=8` and `T=32`, this is fine (8 < 31). But if `T <= L` (e.g., `T=8, L=8`), only 7 of 8 offsets contribute, and the final division by `L` over-penalizes.

**Suggested fix:** Divide by the actual number of contributing offsets, not `L`:
```python
num_offsets = min(L, T - 1)
reward_loss = reward_loss / max(num_offsets, 1)
```

### BUG-6: Same MTP averaging bug in BC loss

**File:** `scripts/train_agent_finetune.py`, lines 532-546
**Severity:** BUG

Identical issue to BUG-5 for the behavioral cloning loss. `bc_loss / L` should be `bc_loss / max(min(L, T - 1), 1)`.

### WARNING-14: `total_loss_batch.item()` called on a tensor that may retain gradients in logging

**File:** `scripts/train_agent_finetune.py`, line 584
**Severity:** WARNING

```python
total_loss += total_loss_batch.item()
```

`.item()` correctly detaches and converts to Python float, so this is fine for the accumulator. However, on line 597:

```python
f"Loss: {total_loss_batch.item():.4f} "
```

This re-calls `.item()` on the same tensor that was just used for `.backward()`. While `.item()` doesn't hold a reference, having `total_loss_batch` still in scope means the entire computation graph is alive until the next iteration. This is not a memory leak (Python GC handles it) but is worth noting.

### WARNING-15: No LR scheduler

**File:** `scripts/train_agent_finetune.py` (entire file)
**Severity:** WARNING

Similar to `train_dynamics.py`, no learning rate scheduler is used. For finetuning where you're adding new heads (reward_head, policy_head) to a pretrained backbone, a warmup for the new parameters is particularly important to avoid destabilizing the pretrained weights.

### WARNING-16: Checkpoint does not save `global_step`

**File:** `scripts/train_agent_finetune.py`, lines 618-644
**Severity:** WARNING

The `save_checkpoint` function does not save or restore a `global_step` counter. There is no `global_step` tracked in `train_epoch` either. If step-based logging or scheduling is added later, resume will have no way to know how many steps have elapsed.

### STYLE-4: `dynamics_loss` uses `F.mse_loss` instead of `x_prediction_loss`

**File:** `scripts/train_agent_finetune.py`, line 505
**Severity:** STYLE

```python
dynamics_loss = F.mse_loss(z_pred, z_target)
```

The Phase 1 dynamics training uses the custom `x_prediction_loss` function (which includes ramp weighting by tau). The finetuning uses plain MSE. This means the finetuning doesn't weight the loss by noise level, which is a deliberate simplification but diverges from the Phase 1 training objective and may cause the model to spend equal effort on all noise levels rather than focusing on high-noise predictions.

---

## 5. `scripts/pretokenize_frames.py`

### BUG-7: Uses `torch.float16` autocast instead of `torch.bfloat16` on CUDA

**File:** `scripts/pretokenize_frames.py`, line 210
**Severity:** BUG

```python
with torch.amp.autocast(device_type=device.split(":")[0], dtype=torch.float16):
```

All training scripts use `torch.bfloat16` on CUDA. Using `torch.float16` during pretokenization means the latents are encoded with different numerical precision than what the model was trained with. While the final `.astype(np.float16)` save truncates to float16 anyway, intermediate computations differ.

More importantly, if the tokenizer model contains operations that are unstable in fp16 (e.g., large attention logits that overflow fp16's range of [-65504, 65504]), the pretokenized latents could contain NaN/Inf values that silently corrupt downstream dynamics training.

**Suggested fix:**
```python
amp_dtype = torch.bfloat16 if device.split(":")[0] == "cuda" else torch.float16
with torch.amp.autocast(device_type=device.split(":")[0], dtype=amp_dtype):
```

### WARNING-17: Latents saved as float16, losing precision from bfloat16 training

**File:** `scripts/pretokenize_frames.py`, line 227
**Severity:** WARNING

```python
latents = latents.cpu().numpy().astype(np.float16)
```

The tokenizer was trained in bfloat16. NumPy's float16 is IEEE fp16, which has a different exponent range than bf16 (fp16: 5 exponent bits, bf16: 8 exponent bits). Values that fit in bf16 but exceed fp16 range will overflow to Inf. Values that are small in bf16 but below fp16 subnormal range will underflow to 0.

In practice, latent values are typically in the range [-5, 5], so this is unlikely to cause issues. But it's a precision mismatch worth documenting.

---

## 6. Cross-cutting Issues

### No random seed anywhere

**File:** All 5 scripts
**Severity:** STYLE (STYLE-5)

None of the scripts set `torch.manual_seed()`, `random.seed()`, or `numpy.random.seed()`. Training runs are not reproducible. For debugging and comparing runs, seeding is essential.

### No train/val split in any script

**File:** All training scripts
**Severity:** WARNING (already noted as WARNING-7)

No script creates a validation split. All metrics are computed on training data. With ~18 replays, the risk of overfitting is significant and there's no way to detect it.

### VideoShuffleSampler seed is deterministic across epochs

**File:** `src/ahriuwu/data/dataset.py`, lines 1086-1100
**Severity:** WARNING

```python
def __iter__(self):
    rng = random.Random(self.seed)
    videos = self.video_ids.copy()
    rng.shuffle(videos)
```

If `seed` is provided, the same shuffle order is used every epoch because `rng` is re-created from the same seed each time `__iter__` is called. This means the model sees data in the exact same video order every epoch. If `seed=None` (the default in the agent finetuning script), Python's `Random(None)` seeds from system entropy, so this is fine by default.

### RewardMixtureSampler samples with replacement within videos

**File:** `src/ahriuwu/data/dataset.py`, lines 1022-1028
**Severity:** WARNING

```python
for _ in range(len(all_indices)):
    if reward_indices and self.rng.random() < 0.5:
        yield self.rng.choice(reward_indices)
    else:
        yield self.rng.choice(all_indices)
```

Both branches use `random.choice()`, which samples **with replacement**. This means:
1. Some sequences may appear multiple times in the same epoch.
2. Some sequences may never appear in an epoch.
3. The DataLoader's `batch_size` collation can produce batches with duplicate sequences.

For the 50/50 reward mixture, this is a deliberate design choice (matching DreamerV4's stochastic mixture), but the within-video replacement could lead to less diverse training. With very few reward-containing sequences per video, the same reward sequence could be sampled many times.

---

## Severity Legend

| Severity | Meaning |
|----------|---------|
| **BUG** | Will crash at runtime or produce incorrect results |
| **WARNING** | Won't crash but may degrade training quality or cause subtle issues |
| **STYLE** | Code quality, maintainability, or documentation concern |

---

## Priority Fixes

1. **Immediate (will crash):** BUG-1, BUG-2, BUG-3 in `train_tokenizer.py` -- function call signatures are wrong. The script cannot run at all.
2. **High (silent correctness issue):** BUG-7 in `pretokenize_frames.py` -- fp16 vs bf16 mismatch could produce NaN latents. BUG-5/BUG-6 in `train_agent_finetune.py` -- MTP loss averaging is wrong when T is small.
3. **Medium (training quality):** WARNING-9 (global_step counting), WARNING-12/WARNING-13 (gradient accumulation edge cases), WARNING-7 (no validation).
4. **Low (robustness):** WARNING-1/WARNING-4 (hardcoded GradScaler device), WARNING-5 (GradScaler with bf16), STYLE items.
