# Tier 0 Training Changes Audit

All changes applied to the four training scripts for DreamerV4 replication alignment.

## Files Modified

- `scripts/train_tokenizer.py`
- `scripts/train_transformer_tokenizer.py`
- `scripts/train_dynamics.py`
- `scripts/train_agent_finetune.py`

---

## 1. WSD Learning Rate Schedule (all 4 scripts)

**CLI args added:**
- `--warmup-steps` (default 2000): linear ramp from 0 to peak LR
- `--decay-steps` (default 0): linear decay to 0 at end. 0 means no decay (warmup + hold only)

**Implementation:**
- Lambda scheduler via `torch.optim.lr_scheduler.LambdaLR` with `wsd_schedule` function
- Three phases: warmup (linear 0 -> 1), stable (constant 1.0), decay (linear 1 -> 0)
- `scheduler.step()` called once per **optimizer step** (not per micro-batch) -- critical for gradient accumulation correctness
- In `train_tokenizer.py`: called after every batch (no gradient accumulation)
- In `train_transformer_tokenizer.py` and `train_dynamics.py`: called inside the `(batch_idx + 1) % accumulation_steps == 0` block
- In `train_agent_finetune.py`: called after every optimizer step

**Checkpoint integration:**
- `scheduler.state_dict()` saved in all checkpoints as `"scheduler_state_dict"`
- Restored on resume via `scheduler.load_state_dict()` with backward-compat guard (`if "scheduler_state_dict" in checkpoint`)

**Changes by file:**
| File | save_checkpoint | load_checkpoint | train_epoch | main |
|------|----------------|-----------------|-------------|------|
| train_tokenizer.py | +scheduler param, +scheduler_state_dict | +scheduler param, +restore logic | +scheduler param, +scheduler.step(), +LR logging | +scheduler creation, +wsd_schedule lambda |
| train_transformer_tokenizer.py | +scheduler param, +scheduler_state_dict | +scheduler param, +restore logic | +scheduler param, +scheduler.step() in accum block, +LR logging | +scheduler creation, +wsd_schedule lambda |
| train_dynamics.py | +scheduler kwarg, +scheduler_state_dict | +scheduler kwarg, +restore logic | +scheduler param, +scheduler.step() in accum block | +scheduler creation, +wsd_schedule lambda |
| train_agent_finetune.py | +scheduler param, +scheduler_state_dict | N/A (no load_checkpoint for finetune) | +scheduler param, +scheduler.step() | +scheduler creation, +wsd_schedule lambda |

---

## 2. 8-bit AdamW (all 4 scripts)

**Import:**
```python
try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False
```

**CLI arg:** `--use-8bit-adam` (default True, `--no-use-8bit-adam` to disable via `BooleanOptionalAction`)

**Optimizer creation:** `bnb.optim.AdamW8bit(...)` if `args.use_8bit_adam and HAS_BNB`, else falls back to `torch.optim.AdamW(...)` with a warning.

**Changes by file:**
| File | Lines changed |
|------|--------------|
| train_tokenizer.py | import block (lines 20-24), CLI arg (lines 87-92), optimizer creation (lines 363-370) |
| train_transformer_tokenizer.py | import block, CLI arg, optimizer creation |
| train_dynamics.py | import block, CLI arg, optimizer creation |
| train_agent_finetune.py | import block, CLI arg, optimizer creation (uses `all_params` list) |

---

## 3. torch.compile (all 4 scripts)

**CLI arg:** `--no-compile` (store_true, default False = compile enabled)

**Placement:** After `model.to(device)`, BEFORE optimizer creation.

**Tokenizer scripts:** LPIPS/VGG loss model is NOT compiled (comment added: "NOT compiled - frozen LPIPS/VGG").

**train_agent_finetune.py special handling:** After `torch.compile(dynamics)`, `model_dim` access uses `dynamics._orig_mod.model_dim` fallback since compiled models wrap the original.

**Changes by file:**
| File | What gets compiled |
|------|-------------------|
| train_tokenizer.py | tokenizer model only (not TokenizerLoss/LPIPS) |
| train_transformer_tokenizer.py | transformer tokenizer model only (not MAELoss/LPIPS) |
| train_dynamics.py | dynamics model |
| train_agent_finetune.py | dynamics model only (not reward_head, policy_head, tokenizer) |

---

## 4. Weight Decay Default Change (all 4 scripts)

Changed default from `0.01` to `0.1`.

| File | Before | After |
|------|--------|-------|
| train_tokenizer.py | hardcoded `weight_decay=0.01` | CLI arg `--weight-decay` default=0.1 |
| train_transformer_tokenizer.py | `--weight-decay` default=0.01 | default=0.1 |
| train_dynamics.py | `--weight-decay` default=0.01 | default=0.1 |
| train_agent_finetune.py | hardcoded `weight_decay=0.01` in optimizer | CLI arg `--weight-decay` default=0.1, used as `args.weight_decay` |

---

## 5. Learning Rate Default Change (all 4 scripts)

Changed default from `1e-4` to `3e-4`.

| File | Before | After |
|------|--------|-------|
| train_tokenizer.py | `--lr` default=1e-4 | default=3e-4 |
| train_transformer_tokenizer.py | `--lr` default=1e-4 | default=3e-4 |
| train_dynamics.py | `--lr` default=1e-4 | default=3e-4 |
| train_agent_finetune.py | `--lr` default=1e-4 | default=3e-4 |

---

## 6. Gradient Clipping in train_tokenizer.py

**Before:** No gradient clipping. `scaler.scale(loss).backward()` then `scaler.step(optimizer)` directly.

**After:**
```python
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
```

This matches the pattern already used in `train_transformer_tokenizer.py` and `train_dynamics.py`.

---

## 7. RunningRMS in train_tokenizer.py

**Before:** Used `losses["loss"]` directly (weighted combination computed inside `TokenizerLoss`).

**After:**
- Import `RunningRMS` from `ahriuwu.models`
- Create RMS trackers: `{"mse": RunningRMS(), "lpips": RunningRMS()}`
- Normalize losses before combining:
  ```python
  mse_norm = rms_trackers["mse"].update(losses["mse"])
  lpips_norm = rms_trackers["lpips"].update(losses["lpips"])
  loss = mse_norm + args.lpips_weight * lpips_norm
  ```
- RMS state saved/loaded in checkpoints as `"rms_state"`
- Pattern matches `train_transformer_tokenizer.py` exactly

---

## Summary of New CLI Args (all 4 scripts)

| Arg | Type | Default | Purpose |
|-----|------|---------|---------|
| `--warmup-steps` | int | 2000 | WSD warmup phase length |
| `--decay-steps` | int | 0 | WSD decay phase length (0=none) |
| `--use-8bit-adam` / `--no-use-8bit-adam` | bool | True | Toggle bitsandbytes 8-bit optimizer |
| `--no-compile` | flag | False | Disable torch.compile |
| `--weight-decay` | float | 0.1 | AdamW weight decay (new CLI arg in tokenizer + agent_finetune) |

## Backward Compatibility

- Old checkpoints without `scheduler_state_dict` or `rms_state` keys are handled gracefully (guarded by `if key in checkpoint`)
- `--no-use-8bit-adam --no-compile` reproduces pre-change behavior (except LR/WD defaults)
