# Data Pipeline Audit: ahriuwu DreamerV4 Replication

**Date:** 2026-03-02
**Scope:** `src/ahriuwu/data/dataset.py`, `keylog_extractor.py`, `actions.py`, `feature_extraction_pipeline.py`
**Auditor:** Claude Opus 4.6 (code review mode)

---

## Summary

Reviewed ~3,238 lines across 4 files. Found **5 BUGs** (including 1 crash-at-runtime), **16 WARNINGs**, and **7 STYLE** issues. The most critical finding is that the feature extraction pipeline calls a method that does not exist on its target class, which will crash at runtime. There are also angle-binning inconsistencies between the action space and keylog extractor, a missing `reward_indices` attribute on `LatentSequenceDataset`, and a massive memory duplication issue in `FrameWithStateDataset`.

---

## BUG severity issues

### BUG-1: `FeatureExtractionPipeline` calls nonexistent method on `GarenHUDTracker`

**File:** `/Users/dani/Repos/ahriuwu/src/ahriuwu/data/feature_extraction_pipeline.py`, line 248
**Severity:** BUG (crash at runtime)

```python
# Line 201-205: creates a GarenHUDTracker
movement_tracker = GarenHUDTracker(
    normalized_regions=normalized_regions,
    frame_width=frame_width,
    frame_height=frame_height,
)

# Line 248: calls detect_ability_usage which does NOT exist on GarenHUDTracker
abilities = movement_tracker.detect_ability_usage(frame, absolute_frame)
```

`GarenHUDTracker` (keylog_extractor.py:132-244) only has `detect_movement()` and `infer_wasd()`. The method `detect_ability_usage()` lives on `AbilityBarDetector` (keylog_extractor.py:314). This will raise `AttributeError` whenever `process_video()` is called.

**Suggested fix:** Create an `AbilityBarDetector` instance in the pipeline and call its `detect_ability_usage()`:

```python
ability_detector = AbilityBarDetector(
    normalized_regions=normalized_regions,
    frame_width=frame_width,
    frame_height=frame_height,
    fps=input_fps,
)
# ...
abilities = ability_detector.detect_ability_usage(frame, absolute_frame)
```

---

### BUG-2: `LatentSequenceDataset` missing `reward_indices` attribute

**File:** `/Users/dani/Repos/ahriuwu/src/ahriuwu/data/dataset.py`, lines 627-667 vs 744-754
**Severity:** BUG (crash when used with `RewardMixtureSampler`)

`LatentSequenceDataset.__init__()` never initializes `self.reward_indices` and never calls `self._precompute_reward_indices()`. However, the method `_precompute_reward_indices()` exists on lines 744-754 and tries to append to `self.reward_indices`. If a `RewardMixtureSampler` is constructed with a `LatentSequenceDataset`, it accesses `dataset.reward_indices` (line 986) which will raise `AttributeError`.

Compare with `PackedLatentSequenceDataset.__init__()` (lines 373-375) which correctly initializes and calls it:
```python
self.reward_indices: list[int] = []
self._precompute_reward_indices()
```

**Suggested fix:** Add these two lines to `LatentSequenceDataset.__init__()` after the feature data loading block (after line 667):

```python
self.reward_indices: list[int] = []
if load_rewards:
    self._precompute_reward_indices()
```

---

### BUG-3: `visualize_regions()` references nonexistent fields on `HUDRegions`

**File:** `/Users/dani/Repos/ahriuwu/src/ahriuwu/data/keylog_extractor.py`, lines 1465-1497
**Severity:** BUG (crash at runtime)

The function references these fields which do not exist on `HUDRegions` (lines 87-95):
- `regions.ability_bar_full` (line 1480)
- `regions.garen_level_box` (line 1484)
- `regions.garen_name_area` (line 1485)
- `regions.garen_health_bar` (line 1486)
- `regions.garen_portrait` (line 1488)
- `regions.stats_area` (line 1489)

`HUDRegions` only has: `ability_q`, `ability_w`, `ability_e`, `ability_r`, `summoner_d`, `summoner_f`, `game_area`. This function is dead code that was not updated when `HUDRegions` was refactored.

**Suggested fix:** Either remove the function or update it to only reference fields that exist on `HUDRegions`.

---

### BUG-4: Angle binning inconsistency between `ActionSpace.angle_to_direction` and `MousePositionEstimator.angle_to_slice`

**File:** `/Users/dani/Repos/ahriuwu/src/ahriuwu/data/actions.py`, line 73 vs `/Users/dani/Repos/ahriuwu/src/ahriuwu/data/keylog_extractor.py`, line 1313
**Severity:** BUG (silent data corruption -- training on misaligned labels)

Two different methods convert angles to the 0-17 direction bins, but they use different algorithms:

`ActionSpace.angle_to_direction()` (actions.py:73):
```python
bucket = int((angle + 10) / 20) % 18  # Buckets CENTERED on 0, 20, 40...
```
Bucket 0 covers [-10, 10), bucket 1 covers [10, 30), etc.

`MousePositionEstimator.angle_to_slice()` (keylog_extractor.py:1313):
```python
return int(angle / self.slice_angle) % self.num_slices  # Buckets STARTING at 0, 20, 40...
```
Bucket 0 covers [0, 20), bucket 1 covers [20, 40), etc.

For angle = 10 degrees:
- `angle_to_direction(10)` = `int((10 + 10) / 20) % 18` = 1
- `angle_to_slice(10)` = `int(10 / 20) % 18` = 0

This means the movement_slice values stored in features.json (produced by `angle_to_slice`) are systematically shifted by half a bucket compared to what the `ActionSpace` expects. During training, the model learns movement labels that are misaligned with the canonical action space.

**Suggested fix:** Use the same binning logic everywhere. The centered-bucket approach in `angle_to_direction` is the standard practice (each bin represents the direction nearest its center). Apply the same formula in `MousePositionEstimator.angle_to_slice`:

```python
def angle_to_slice(self, angle_degrees: float) -> int:
    angle = angle_degrees % 360
    return int((angle + self.slice_angle / 2) / self.slice_angle) % self.num_slices
```

---

### BUG-5: `encode_action` can silently clamp to 127, losing information

**File:** `/Users/dani/Repos/ahriuwu/src/ahriuwu/data/actions.py`, lines 125-162
**Severity:** BUG (silent data corruption)

```python
action = movement + 18 * (1 + i)
return min(action, 127)
```

For priority index i >= 6 (i.e., `item` and `B`):
- `item` (i=6): movement + 18*7 = movement + 126. For movement >= 2, action = 128+, clamped to 127.
- `B` (i=7): movement + 18*8 = movement + 144, always clamped to 127.

This means for `item` with movement >= 2 and for all `B` + any movement, the action is clamped to 127, making it impossible to decode the original movement direction. Additionally, `decode_action(127)` would give `ability_idx = (127 // 18) - 1 = 6` and `movement = 127 % 18 = 1`, which is incorrect for all clamped values.

The docstring claims "fits in 128 actions" and the code claims "18 * 7 = 126 actions, plus 2 reserved" but the actual range is 18 * 9 = 162 (movement * (1 + 8 priorities)).

**Suggested fix:** Either increase the action space size to accommodate all combinations (162), or use a different encoding scheme. Given the factorized action representation used elsewhere (movement + 8 binary), consider whether `encode_action`/`decode_action` are actually used -- if only the factorized dict representation is used during training, these functions may be dead code.

---

## WARNING severity issues

### WARN-1: No "stationary" class in movement encoding

**File:** `/Users/dani/Repos/ahriuwu/src/ahriuwu/data/actions.py`, lines 31, 39-48
**Severity:** WARNING

Movement has 18 classes (0-17) representing directions, but there is no explicit "stationary" class. When there is no movement, `movement_to_slice` returns 0 (East), and the dataset `_get_actions` pads with 0 meaning "no action is East." This means the model cannot distinguish "moving East" from "not moving."

In `MousePositionEstimator.movement_to_slice()` (keylog_extractor.py:1320-1321):
```python
if abs(dx) < 1 and abs(dy) < 1:
    return 0  # No movement, default to right
```

**Suggested fix:** Add a 19th class (class 0 = stationary, classes 1-18 = directions) or use a separate binary "is_moving" flag.

---

### WARN-2: `FrameWithStateDataset` stores entire states JSON in every sequence entry

**File:** `/Users/dani/Repos/ahriuwu/src/ahriuwu/data/dataset.py`, lines 260-266
**Severity:** WARNING (memory)

Every sequence entry stores a reference to the same `states` list. In Python, lists are reference-counted, so this is the same object for sequences within one video. However, the pattern is confusing and fragile -- if any code mutates `states`, all sequences are affected. More importantly, the entire states file is loaded into RAM during `_index_frames()` and kept alive for the lifetime of the dataset.

Compare with `LatentSequenceDataset` and `PackedLatentSequenceDataset`, which store feature data in a per-video dict, not per-sequence.

**Suggested fix:** Store states in a `self.video_states: dict[str, list]` and reference by video_id in `__getitem__`.

---

### WARN-3: `SingleFrameDataset` / `FrameSequenceDataset` no error handling for missing frames

**File:** `/Users/dani/Repos/ahriuwu/src/ahriuwu/data/dataset.py`, lines 97, 182
**Severity:** WARNING

```python
frame = cv2.imread(str(frame_path))        # Returns None if file missing
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Crashes on None
```

If a frame file is missing or corrupt, `cv2.imread` returns `None` and the next line crashes with an unhelpful error. The `FrameSequenceDataset` constructs frame paths via f-string (line 181) assuming contiguous numbering, so a gap in frame numbers would hit this.

**Suggested fix:** Add a check after `cv2.imread` and either skip the frame or raise a clear error with the path.

---

### WARN-4: `PackedLatentSequenceDataset` LRU cache is not thread-safe

**File:** `/Users/dani/Repos/ahriuwu/src/ahriuwu/data/dataset.py`, lines 377-399
**Severity:** WARNING

The LRU cache (`video_cache`, `cache_order`) is a plain dict/list with no locking. With `num_workers > 0` in DataLoader, multiple worker processes will each get a copy (fork), so this is safe with forked workers. However, if anyone uses threading or `spawn` start method (default on macOS/Windows), concurrent access could corrupt the cache.

The `cache_order.remove(video_id)` on line 381 is O(n) which degrades performance. More critically, if two threads call `_load_video` simultaneously for the same video_id, the cache_order could be corrupted.

**Suggested fix:** Document that this dataset requires `fork` start method, or add a threading lock. Consider using `functools.lru_cache` or an `OrderedDict` for O(1) LRU operations.

---

### WARN-5: `PackedLatentSequenceDataset._index_packed_latents` loads all .npz metadata eagerly

**File:** `/Users/dani/Repos/ahriuwu/src/ahriuwu/data/dataset.py`, lines 401-455
**Severity:** WARNING (startup latency)

Line 409: `data = np.load(npz_path)` loads the file (at least partially) for every npz file during `__init__`. For a large dataset with many videos, this could cause significant startup delay. The `frame_indices` array is loaded and then the npz handle is not explicitly closed (relying on GC).

**Suggested fix:** Use `np.load(npz_path, mmap_mode='r')` for the index pass, or load only the `frame_indices` key explicitly. Also consider wrapping in a `with` statement or calling `data.close()`.

---

### WARN-6: No NaN/inf checking in latent loading

**File:** `/Users/dani/Repos/ahriuwu/src/ahriuwu/data/dataset.py`, lines 878, 599-600
**Severity:** WARNING

Neither `LatentSequenceDataset.__getitem__` nor `PackedLatentSequenceDataset.__getitem__` checks for NaN or inf values in loaded latents. If a corrupt or partially-written `.npy`/`.npz` file contains NaN, it will silently propagate through training, potentially causing loss to go to NaN without an obvious error message.

**Suggested fix:** Add an assertion or check after loading:
```python
assert torch.isfinite(latents).all(), f"NaN/inf in latents for {video_id} frame {start_frame}"
```

---

### WARN-7: `RewardMixtureSampler` is not epoch-deterministic

**File:** `/Users/dani/Repos/ahriuwu/src/ahriuwu/data/dataset.py`, lines 1011-1028
**Severity:** WARNING

`RewardMixtureSampler.__iter__` uses `self.rng` which is a persistent `random.Random` instance. Each call to `__iter__` (i.e., each epoch) advances the RNG state, producing a different order. This is intentional for shuffling, but it means:
1. Resuming training from a checkpoint will not reproduce the exact same sampling order unless the RNG state is saved/restored.
2. There is no `set_epoch()` method (common in distributed training samplers).

Compare with `VideoShuffleSampler` which creates a new `random.Random(self.seed)` each epoch (line 1088) -- this makes it deterministic across epochs (same order every time), which is wrong in a different way: repeated identical ordering across epochs.

**Suggested fix:** For `RewardMixtureSampler`, add a `set_epoch(epoch)` method that reseeds with `seed + epoch`. For `VideoShuffleSampler`, do the same instead of recreating the RNG with the same seed each time.

---

### WARN-8: Feature data loaded twice in `PackedLatentSequenceDataset`

**File:** `/Users/dani/Repos/ahriuwu/src/ahriuwu/data/dataset.py`, lines 368-371
**Severity:** WARNING (minor inefficiency)

When `load_actions=True`, both `_load_action_labels()` (which loads `features.json` into `self.action_labels`) and `_load_feature_data()` (which loads `features.json` into `self.feature_data`) are called. Both open and parse the same JSON files. This doubles memory usage and initialization time for the feature data.

**Suggested fix:** Unify into a single load that populates both `action_labels` and `feature_data` from one parse, or share the parsed data.

---

### WARN-9: `_get_actions` in both datasets silently pads when features.json is shorter than latent sequence

**File:** `/Users/dani/Repos/ahriuwu/src/ahriuwu/data/dataset.py`, lines 567-583 and 776-794
**Severity:** WARNING

When `t >= len(labels)`, actions are padded with zeros (movement=0 i.e. "East", all abilities=0). This silent padding means:
1. If features.json has fewer frames than the latent data (e.g., due to a pipeline bug), training proceeds with garbage labels -- no warning is emitted.
2. Movement 0 means "East" rather than "unknown" (see WARN-1).

**Suggested fix:** Log a warning when padding occurs, or raise an error if the mismatch is beyond a small tolerance.

---

### WARN-10: `recall_b` field never written by the feature extraction pipeline

**File:** `/Users/dani/Repos/ahriuwu/src/ahriuwu/data/dataset.py`, lines 578, 788
**Severity:** WARNING

Both dataset classes read `entry.get('recall_b', False)` from features.json. However, `FeatureExtractionPipeline` (feature_extraction_pipeline.py) never writes a `recall_b` field to the features JSON. The `FrameFeatures` dataclass (line 92-120) has no `recall_b` field. This means `recall_b` will always be `False`/0 in practice.

**Suggested fix:** Either add recall detection to the pipeline, or document that recall detection is not yet implemented and remove it from the action space if it will always be zero during training.

---

### WARN-11: Death detection uses health bar disappearance which may fire for other reasons

**File:** `/Users/dani/Repos/ahriuwu/src/ahriuwu/data/dataset.py`, lines 836-858
**Severity:** WARNING

Death is detected by health bar disappearing for 3+ frames after being present for 3+ frames. However, health bar can also disappear when:
- Camera lock temporarily breaks during replay
- Garen enters fog of war (unlikely with locked camera, but possible in replays)
- HUD overlay appears (e.g., tab scoreboard)
- Color detection fails transiently

The 3-frame lookback + 2-frame lookahead window is very short (150-250ms at 20 FPS). A transient detection failure could trigger a -10.0 death penalty, which is a very large reward signal.

**Suggested fix:** Increase the lookback/lookahead window, or cross-reference with gold events (kills often come with gold). Also consider using the respawn timer as confirmation.

---

### WARN-12: `_find_runs` in `GoldTextDetector` is Python-loop over every pixel in a row

**File:** `/Users/dani/Repos/ahriuwu/src/ahriuwu/data/keylog_extractor.py`, lines 803-823
**Severity:** WARNING (performance)

This function iterates over every pixel in a binary row using a Python for-loop. It is called for every row in the search area for every frame. At 1080p with `ga_h = int(0.63 * 1080) = 680` rows, this is 680 calls per frame, each iterating over ~1152 pixels.

**Suggested fix:** Replace with numpy operations:
```python
def _find_runs(self, binary_row: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([0], binary_row, [0]))
    diff = np.diff(padded)
    starts = np.where(diff == 255)[0]
    ends = np.where(diff == -255)[0]
    return list(zip(starts.tolist(), ends.tolist()))
```

---

### WARN-13: Gold filter discards values 100-148, potentially losing legitimate gold amounts

**File:** `/Users/dani/Repos/ahriuwu/src/ahriuwu/data/keylog_extractor.py`, lines 1222-1227
**Severity:** WARNING

```python
if amount < 10:
    continue
# Values like 140, 142 are likely OCR errors for 14
if amount > 100 and amount < 149:
    continue
```

This filter discards gold values in [101, 148]. While the comment says these are "likely OCR errors for 14", there are legitimate gold amounts in this range:
- Cannon minion gold: 60-90 (scales with game time, can exceed 100 late game)
- Tower plating: 120 gold
- Some assist gold amounts

**Suggested fix:** Reconsider this filter or make it configurable. The consensus mechanism should already handle OCR noise adequately.

---

### WARN-14: `GarenHUDTracker` optical flow does not account for skipped frames

**File:** `/Users/dani/Repos/ahriuwu/src/ahriuwu/data/keylog_extractor.py`, lines 170-222
**Severity:** WARNING

The tracker stores `self.prev_gray` from the last call. In the pipeline (feature_extraction_pipeline.py:238), frames are only processed every `frame_skip` frames. But `prev_gray` is updated every call, so when the pipeline skips frames and then calls `detect_movement`, `prev_gray` is from the last *processed* frame, not the previous *source* frame. This means the optical flow is computed across `frame_skip` frames worth of motion, making the displacement values larger and less reliable.

However, looking more carefully, the pipeline reads every frame from the video (line 231 `ret, frame = cap.read()`) but only processes every Nth frame (line 238 `if frame_count % frame_skip == 0`). The `detect_movement` is only called on processed frames, so `prev_gray` is indeed from `frame_skip` frames ago. This means movement magnitudes depend on `frame_skip` -- higher skip = larger displacements. The `movement_threshold` of 0.5 pixels was tuned for 60fps and may not work correctly at 20fps extraction (3x larger displacements).

**Suggested fix:** Either normalize displacement by `frame_skip`, or call `detect_movement` on every frame (updating `prev_gray`) and only record the result on sampled frames.

---

### WARN-15: `_is_enemy_bar` and `_check_name_is_garen` are dead code

**File:** `/Users/dani/Repos/ahriuwu/src/ahriuwu/data/keylog_extractor.py`, lines 900-958
**Severity:** WARNING

`_is_enemy_bar` (line 900) and `_check_name_is_garen` (line 929) are defined on `GoldTextDetector` but never called anywhere in the codebase. They appear to be remnants of an earlier detection strategy.

**Suggested fix:** Remove dead code or add calls where appropriate.

---

### WARN-16: `collate_actions` imported but never used

**File:** `/Users/dani/Repos/ahriuwu/src/ahriuwu/data/dataset.py`, line 12
**Severity:** WARNING

```python
from .actions import ABILITY_KEYS, collate_actions
```

`collate_actions` is imported but never used in dataset.py. It also appears unused in any other file (only defined in actions.py:187 and imported here).

**Suggested fix:** Remove unused import.

---

## STYLE severity issues

### STYLE-1: Inconsistent docstring for `FeatureExtractionPipeline`

**File:** `/Users/dani/Repos/ahriuwu/src/ahriuwu/data/feature_extraction_pipeline.py`, lines 6-8, 124, 139, 159
**Severity:** STYLE

The module docstring (line 4) says "Convert frames to 480x352 for training" and the default `target_size` is `(480, 352)`. The CLI (line 384) constructs the pipeline without passing `target_size`, so it uses the default `(480, 352)`. But the `process_video` docstring (line 159) still says "save 256x256 frames" and the class docstring (line 124) says "save 480x352 frames" -- inconsistent.

Also, the frame save code (line 279) uses variable name `frame_resized` which doesn't match the `frame_256` mentioned in the comment on line 278 ("Save 256x256 frame"). The code actually says `frame_256 = cv2.resize(...)` on line 279 -- the variable is literally named `frame_256` but the size is `(480, 352)`.

**Suggested fix:** Rename variable to `frame_resized` and update all docstrings to say "480x352".

---

### STYLE-2: `FrameWithStateDataset` not used for world model training

**File:** `/Users/dani/Repos/ahriuwu/src/ahriuwu/data/dataset.py`, lines 202-315
**Severity:** STYLE

This class loads raw frames (not latents) and has separate states files. The DreamerV4 pipeline uses `PackedLatentSequenceDataset` for dynamics training. This class appears to be from an earlier design iteration and may be dead code.

---

### STYLE-3: Duplicated contiguous-sequence-finding logic

**File:** `/Users/dani/Repos/ahriuwu/src/ahriuwu/data/dataset.py`
- Lines 420-453 (`PackedLatentSequenceDataset._index_packed_latents`)
- Lines 697-724 (`LatentSequenceDataset._index_latents`)
**Severity:** STYLE

The contiguous frame detection algorithm is duplicated between the two dataset classes. Both use identical logic to find contiguous blocks and create sequence entries.

**Suggested fix:** Extract to a shared utility function.

---

### STYLE-4: `print` statements for logging throughout

**File:** Multiple locations in all 4 files
**Severity:** STYLE

All logging uses bare `print()` calls instead of Python's `logging` module. This makes it impossible to control verbosity or redirect output.

---

### STYLE-5: `ActionSpace` has both static methods and module-level constants

**File:** `/Users/dani/Repos/ahriuwu/src/ahriuwu/data/actions.py`
**Severity:** STYLE

`MOVEMENT_CLASSES` and `ABILITY_KEYS` exist as both module-level constants (lines 31-32) and as class attributes on `ActionSpace` (lines 56-57). The `encode_action` and `decode_action` functions are module-level but logically belong to the `ActionSpace` class. The priority order in `encode_action` (line 152: `['R', 'Q', 'E', 'W', 'D', 'F', 'item', 'B']`) differs from `ABILITY_KEYS` order (line 32: `['Q', 'W', 'E', 'R', 'D', 'F', 'item', 'B']`), which is confusing.

---

### STYLE-6: Magic numbers in health bar detection

**File:** `/Users/dani/Repos/ahriuwu/src/ahriuwu/data/keylog_extractor.py`
**Severity:** STYLE

Various magic numbers throughout health bar detection:
- Line 672: `min_width = int(40 * (w / 1920))`
- Line 728: `min_gray_width = max(3, int(25 * w / 1920))`
- Line 764: `min_cluster_size = max(3, int(10 * h / 1080))`
- Line 990: `max_health_width = int(120 * (w / 1920))`
- Line 1098: `hb_y = garen_y - int(10 * (h / 1080))`

These should be named constants or configuration parameters.

---

### STYLE-7: `_red_health_lower`/`_red_health_upper` on `GoldTextDetector` is confusing with team-based health

**File:** `/Users/dani/Repos/ahriuwu/src/ahriuwu/data/keylog_extractor.py`, lines 520-543
**Severity:** STYLE

The `_is_enemy_bar` method (dead code per WARN-15) uses `self.red_health_lower` which is the team-side-independent "red health lower" threshold. But `self.ally_health_lower` is set conditionally based on team side. If the red team ally detection is used, it uses the same thresholds as enemy detection, making `_is_enemy_bar` unreliable. This logic is tangled.

---

## Detailed analysis by component

### Dataset classes

**LatentSequenceDataset -- Consecutive frame handling:** Correctly implemented. The `_index_latents` method (lines 669-724) finds contiguous blocks of frame numbers and only creates sequences within those blocks. Gaps in frame numbering cause sequence boundaries -- sequences never span gaps. This is correct.

**PackedLatentSequenceDataset -- Packing logic:** Correct. The packed dataset stores latents in a contiguous array per video. The `_index_packed_latents` method correctly builds a mapping from frame numbers to array indices and finds contiguous blocks. The `__getitem__` method slices the array directly (line 599: `latents_array[start_idx:start_idx + self.sequence_length]`), which is efficient and correct because contiguous frame numbers map to contiguous array indices.

**Cross-video sequences:** Impossible by design. Both datasets iterate per-video and only create sequences within a single video's frames. The contiguous block detection resets at each video boundary. This is correct.

**Off-by-one errors in frame indexing:** No off-by-one errors found in the sequence indexing. The range `range(0, contiguous_count - self.sequence_length + 1, self.stride)` is correct -- the last sequence starts at `contiguous_count - sequence_length` and ends at `contiguous_count - 1` inclusive.

**Data types:** Latents are loaded from numpy (typically float32 from the tokenizer) and converted via `torch.from_numpy()` which preserves dtype. The `.copy()` call (line 600) is correct to avoid sharing memory with the cached numpy array.

### Attention masks / Packing

**No packed-batch attention masking:** Neither dataset class produces attention masks. If sequences from different videos were packed into a single batch element (as in some LLM training), attention masks would be needed to prevent cross-sequence attention. However, each batch element is a single contiguous sequence, so no attention masking is needed. This is correct for the current architecture.

### Action encoding

**18-direction encoding:** The movement encoding uses 18 bins of 20 degrees each. The binning inconsistency (BUG-4) is the main issue. The factorized representation (separate movement + 8 binary abilities) used in `_get_actions()` is sound for the dynamics model's multi-head action embedding.

**Multi-component action sum:** The dynamics model (referenced in models/dynamics.py:951) sums action embeddings from different components. The dataset provides separate tensors per component (movement, Q, W, E, R, D, F, item, B), each as `(T,)` long tensors. The model embeds each and sums. This is correct.

**Temporal alignment:** Actions from features.json are indexed by the same frame index as latents. The `_get_actions` method uses `start_frame + t` to index into the features list, matching the same frame that the latent was encoded from. This is correct.

### Samplers

**VideoShuffleSampler correctness:** Correctly shuffles video order while keeping sequences within a video together. Within each video, sequences are also shuffled. The total count in `__len__` matches the number of yielded indices. Correct.

**RewardMixtureSampler correctness:** The 50/50 mixture is implemented as: for each slot, flip a coin -- heads picks from reward-containing sequences (with replacement), tails picks from all sequences (with replacement). This means reward sequences are oversampled relative to their natural frequency, which is the intended behavior. However, using with-replacement sampling means some sequences may be seen multiple times per epoch while others are never seen. This is a deliberate design choice matching DreamerV4's priority replay.

### Feature extraction pipeline

**Frame extraction rate:** `frame_skip = max(1, int(input_fps / self.target_fps))` with integer division. For 30fps input targeting 20fps: `frame_skip = 1`, yielding 30fps (not 20fps). For 60fps input targeting 20fps: `frame_skip = 3`, yielding 20fps. The integer division means the effective output FPS is `input_fps / frame_skip`, which may not exactly equal `target_fps`.

**Resolution handling:** The pipeline default is `(480, 352)` but the dataset default `TARGET_SIZE` is also `(480, 352)`, so they are consistent. The old 256x256 references are only in comments/variable names (STYLE-1).

### No `worker_init_fn` anywhere

**File:** All training scripts
**Severity:** WARNING (noted in report, not a data pipeline issue per se)

No training script sets `worker_init_fn` for DataLoader. With `num_workers > 0`, each worker process inherits the same numpy/random seed from the parent. For the dataset classes here, this is not a problem because they do not use random operations in `__getitem__`. However, if any transform with random augmentation is added later, this would cause all workers to produce identical augmentations.

---

## Issue summary table

| ID | Severity | File | Line(s) | Summary |
|---|---|---|---|---|
| BUG-1 | BUG | feature_extraction_pipeline.py | 248 | Calls `detect_ability_usage` on `GarenHUDTracker` which lacks that method |
| BUG-2 | BUG | dataset.py | 627-667 | `LatentSequenceDataset` missing `reward_indices` init and `_precompute_reward_indices` call |
| BUG-3 | BUG | keylog_extractor.py | 1465-1497 | `visualize_regions` references 6 nonexistent fields on `HUDRegions` |
| BUG-4 | BUG | actions.py:73, keylog_extractor.py:1313 | 73, 1313 | Inconsistent angle-to-bin conversion (centered vs edge-aligned buckets) |
| BUG-5 | BUG | actions.py | 125-162 | `encode_action` silently clamps to 127, corrupting item/B actions |
| WARN-1 | WARNING | actions.py | 31 | No "stationary" class; movement=0 conflates with East |
| WARN-2 | WARNING | dataset.py | 260-266 | `FrameWithStateDataset` stores entire states list per sequence entry |
| WARN-3 | WARNING | dataset.py | 97, 182 | No null check after `cv2.imread` |
| WARN-4 | WARNING | dataset.py | 377-399 | LRU cache not thread-safe; O(n) remove |
| WARN-5 | WARNING | dataset.py | 401-455 | All .npz files partially loaded at init; no explicit close |
| WARN-6 | WARNING | dataset.py | 878, 599-600 | No NaN/inf checking on loaded latents |
| WARN-7 | WARNING | dataset.py | 1011-1028, 1086-1100 | Sampler determinism issues across epochs/checkpoints |
| WARN-8 | WARNING | dataset.py | 368-371 | Feature data loaded/stored twice when load_actions=True |
| WARN-9 | WARNING | dataset.py | 567-583, 776-794 | Silent padding when features.json is shorter than latent sequence |
| WARN-10 | WARNING | dataset.py | 578, 788 | `recall_b` field never written by pipeline; always False |
| WARN-11 | WARNING | dataset.py | 836-858 | Death detection via health bar disappearance is fragile |
| WARN-12 | WARNING | keylog_extractor.py | 803-823 | `_find_runs` uses Python loop over every pixel |
| WARN-13 | WARNING | keylog_extractor.py | 1222-1227 | Gold filter discards legitimate values in 101-148 range |
| WARN-14 | WARNING | keylog_extractor.py | 170-222 | Optical flow magnitude depends on frame_skip, thresholds may be miscalibrated |
| WARN-15 | WARNING | keylog_extractor.py | 900-958 | `_is_enemy_bar` and `_check_name_is_garen` are dead code |
| WARN-16 | WARNING | dataset.py | 12 | `collate_actions` imported but unused |
| STYLE-1 | STYLE | feature_extraction_pipeline.py | 124, 159, 279 | Stale "256x256" references in docstrings and variable names |
| STYLE-2 | STYLE | dataset.py | 202-315 | `FrameWithStateDataset` may be dead code |
| STYLE-3 | STYLE | dataset.py | 420-453, 697-724 | Duplicated contiguous-sequence logic |
| STYLE-4 | STYLE | all files | multiple | Bare `print()` instead of `logging` |
| STYLE-5 | STYLE | actions.py | multiple | Inconsistent organization of constants/functions |
| STYLE-6 | STYLE | keylog_extractor.py | multiple | Magic numbers in health bar detection |
| STYLE-7 | STYLE | keylog_extractor.py | 520-543 | Confusing red health threshold naming |

---

## Priority recommendations

1. **Fix BUG-1 immediately** -- the feature extraction pipeline cannot run at all.
2. **Fix BUG-4** -- angle binning mismatch means all existing movement labels in features.json are systematically off by half a bin. Either fix the extractor and re-extract, or fix the consumer to match.
3. **Fix BUG-2** -- if `LatentSequenceDataset` is still in use, it will crash with `RewardMixtureSampler`.
4. **Address WARN-1** -- adding a stationary class (19 total) is a design decision that affects the entire model. Better to do it now before training at scale.
5. **Address WARN-7** -- sampler determinism matters for reproducibility and checkpoint resumption.
