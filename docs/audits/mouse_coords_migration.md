# Migration: Slice-Based Movement -> Continuous Mouse Coordinates

## Summary

Replaced the 18-bucket directional movement system (inferred from optical flow)
with continuous normalized mouse (x, y) coordinates in [0, 1]. This change
spans the data layer, dynamics model, policy head, and training script.

## Motivation

The old system discretized movement into 18 directional slices (20 degrees each)
from optical flow. This was lossy and imprecise -- optical flow from replay
footage is noisy, and 18 buckets is very coarse for actual mouse positioning.

Continuous (x, y) coordinates from replay files provide:
- Exact mouse position (no quantization loss)
- Natural regression target (MSE) instead of forced classification
- Direct compatibility with replay-extracted cursor data

## Changes

### Data Layer

**`src/ahriuwu/data/actions.py`**
- Added `MOVEMENT_DIM = 2` constant
- `MOVEMENT_CLASSES = 18` kept but marked as deprecated
- `ActionDict` type hint updated: movement is now `list[float]` (x, y)
- `ActionSpace.to_tensor_dict()`: movement returns `(2,)` float tensor
- `ActionSpace.empty_action()`: returns `[0.5, 0.5]` for center
- `collate_actions()`: updated docstring for `(B, T, 2)` movement shape
- Legacy methods (`angle_to_direction`, `direction_to_angle`) kept for backward compat

**`src/ahriuwu/data/dataset.py`**
- `TARGET_SIZE` changed from `(480, 352)` to `(352, 352)` (separate fix, square for tokenizers)
- `PackedLatentSequenceDataset._get_actions()`: loads `mouse_x`/`mouse_y` from features.json,
  returns movement as `(T, 2)` float tensor. Defaults to `(0.5, 0.5)` when absent.
- `LatentSequenceDataset._get_actions()`: same changes as above

**`src/ahriuwu/data/feature_extraction_pipeline.py`**
- `FrameFeatures` dataclass: added `mouse_x: float = 0.5` and `mouse_y: float = 0.5` fields
- Target size references updated to 352x352

**`src/ahriuwu/data/keylog_extractor.py`**
- No changes. `MousePositionEstimator` and `angle_to_slice` kept for backward compat.

### Model Layer

**`src/ahriuwu/models/dynamics.py`**
- Added `MOVEMENT_DIM = 2` constant
- `action_embed['movement']`: changed from `nn.Embedding(18, model_dim)` to
  `nn.Linear(2, model_dim)` -- takes continuous (x, y) and projects to model dim
- `embed_actions()`: movement input is now `(B, T, 2)` float, passed through Linear
- Test code at bottom updated with `torch.rand(B, T, MOVEMENT_DIM)` instead of `randint`

**`src/ahriuwu/models/heads.py`**
- `PolicyHead` now returns a tuple: `(ability_logits, movement_pred)`
- Added `self.movement_heads`: `nn.ModuleList` of `nn.Linear(hidden_dim, 2)` with sigmoid
- `forward()` returns `(B, T, L, action_dim)` ability logits + `(B, T, L, 2)` movement
- `sample()` returns both discrete action indices and movement coordinates
- `log_prob()` operates on discrete ability logits only

### Training Script

**`scripts/train_agent_finetune.py`**
- `ReplayDataset.__getitem__()`: now returns `movement_targets` as `(T, 2)` float tensor
  alongside discrete `actions` tensor
- Training loop extracts `movement_targets` from batch
- BC loss split into:
  - `bc_loss_ability`: cross-entropy for discrete ability actions
  - `bc_loss_movement`: MSE for continuous (x, y) prediction
  - Combined: `bc_loss = (bc_loss_ability + bc_loss_movement) / L`
- Accuracy metrics (top-1, top-5) only computed for discrete ability predictions

## Backward Compatibility

- Old features.json files without `mouse_x`/`mouse_y` default to `(0.5, 0.5)` center
- `MOVEMENT_CLASSES = 18` and all slice-based utilities kept but deprecated
- Old checkpoints will have shape mismatch on `action_embed.movement` (Embedding vs Linear) --
  these keys will be logged as "skipped (shape mismatch)" on load, which is expected

## Data Format

The `features.json` per video now supports:
```json
{
  "frames": [
    {
      "mouse_x": 0.65,
      "mouse_y": 0.42,
      "movement_slice": 3,
      "movement_dx": 2.1,
      "movement_dy": -1.5,
      ...
    }
  ]
}
```

`mouse_x` and `mouse_y` are normalized screen coordinates in [0, 1].
The replay-extracted coordinates will come from a separate pipeline that reads
actual cursor positions from replay files.
