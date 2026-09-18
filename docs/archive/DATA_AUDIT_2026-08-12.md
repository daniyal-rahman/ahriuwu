# Data / label / preprocessing audit — 2026-08-12

Scope: the 125 replay matches that have both frames and latents
(`scratchpad/valid_games.json:both`), **3,554,768 label frames / 3,381,817 latent
frames / 49.4 h of game time**. Everything below was measured against the files
on disk, not read out of comments. Scripts and raw outputs are under
`scratchpad/audit_*`; the per-match distilled arrays are cached in
`scratchpad/audit_cache/<match>.npz` (regenerate with
`scratchpad/audit_scan_labels.py`).

Reproduce:

| script | what it measures |
| --- | --- |
| `scratchpad/audit_scan_labels.py` | distills all 125 `labels.json` + `clicks.json` into npz |
| `scratchpad/audit_analyze.py` | corpus basics, corrupt stats, reward, attack, casts, aux targets |
| `scratchpad/audit_cursor.py` | cursor.screen presence + drift-vs-command decomposition |
| `scratchpad/audit_movement_exact.py` | exact `_parse_movement` + `discretize_movement` replay, transition attribution |
| `scratchpad/audit_align2.py` | frame↔label offset from teleport events |
| `scratchpad/audit_projection.py`, `audit_projection2.py` | `project()` calibration vs measured optical scroll (**misleading — see R6**) |
| `scratchpad/audit_overlay_offcentre.png` etc. | `project()` verified against rendered champion models |
| `scratchpad/audit_b_*.py`, `audit_c_*.py`, `audit_d_*.py` | latent index invariants, dupes, HUD mask |
| `scratchpad/audit_aspect/*.py` | minion HP-bar geometry, squish vs letterbox |

---

## Summary table (ranked by estimated impact on model quality)

| # | Finding | Measured | Verdict |
| --- | --- | --- | --- |
| 1 | BC movement target: 43.2% of its transitions are camera drift, not commands | 239,069 / 552,909 | **CONFIRMED, new** |
| 2 | 24.1% of frames have no movement target; the label pipeline drops off-screen commands | 857,520 frames | **CONFIRMED, new** |
| 3 | No held-out set during training; dynamics "eval" is a training batch | `--holdout-videos 0` | **CONFIRMED, new** |
| 4 | `enemy_visible` aux target is wrong on 29.6% of frames; `enemy_hp_frac` supervised off-screen | 54.5% of positives wrong | **CONFIRMED, new** |
| 5 | 37.7% of genuine commands are quantized away by the 21-bin grid | 190,168 / 504,008 | **CONFIRMED (lead 2, restated)** |
| 6 | Phantom cursor jump at 89.7% of attack/ability endings | 19,852 / 22,122 | **CONFIRMED, new** |
| 7 | 4.9% of label frames have no latent; 9.3% of captured game time has no frames | 172,951 + 5.0 h | **CONFIRMED, new** |
| 8 | 11.0% of ability casts land past the end of the frame record and are dropped | 3,353 / 30,338 | **CONFIRMED, new** |
| 9 | Frames lag labels by a fixed +1.47 frames | median +1, no drift | **CONFIRMED, minor** |
| 10 | `GarenQAttack` (≈21% of autos) is labeled `ability`, not `AA` | 226 vs 179 episodes/game | **PARTIAL (lead 3)** |
| 11 | `level` reads 19–20 → aux target > 1.0 | 59,412 frames, 25 matches | **CONFIRMED, minor** |
| 12 | Reward head wastes 85% of its bucket range | 39 / 255 buckets used | **CONFIRMED, minor** |
| 13 | Movement bins are anisotropic (64 px in x, 36 px in y) | — | **CONFIRMED, minor** |
| 14 | Train/deploy gap: training frames have **no HUD**, live capture does | 0 static pixels | **CONFIRMED, new** |
| 15 | `visible_heroes[*].screen != None` means "in frustum", **not** "rendered" — fog-of-war units get coords | verified visually | **CONFIRMED, new** |
| 16 | `champion_world` repeats on 26.6% of consecutive frames (25 Hz mem → 20 Hz grid) | 942,332 pairs | **CONFIRMED, minor** |
| — | *Aspect squish destroys minion HP bars* | 9 px squish vs 9 px letterbox | **REFUTED (lead 1)** |
| — | *`project()` is miscalibrated* | lands on the model at 319 px off-centre | **REFUTED (see R6)** |
| — | *Chained autos are never re-marked* | 99–100% of episodes captured | **REFUTED (lead 3)** |
| — | *Corrupt gold poisons the reward* | `gold_total` 0.000% corrupt | **REFUTED (lead 4)** |
| — | *latent↔frame_indices misalignment* | 0 of 422,527 windows | **REFUTED** |
| — | *HUD mask damages replay frames* | mask is YT-gated | **REFUTED** |
| — | *Duplicate / dropped video frames* | 0 exact dupes | **REFUTED** |

---

# TIER 1 — these cap model quality now

## 1. The BC movement target is ~43% pure label noise (camera drift)

**What the code assumes.** `ReplayLatentSequenceDataset._parse_movement`
(`src/ahriuwu/data/replay_dataset.py:379-422`) treats `label.cursor.screen` as
"the most-recent issued-command location", and treats any frame-to-frame move
larger than a 1% dead-band as a *new command*. `bc_next_action_loss`
(`scripts/train_agent_finetune.py:352-366`) then defines a **transition** as
"the 21-bin cell changed vs the previous frame", and the sticky-categorical gate
is trained to fire exactly on those transitions.

**What the data actually is.** `scripts/aggregation/pipeline.py:1560-1569`
builds it as

```python
cursor_world  = cast_target if cast_target is not None else last_click_world
cursor_screen = project(cursor_world[0], cursor_world[1], cx, cy, cz)
```

`cursor_world` is *held* between commands, but it is re-projected every frame
through the **current** camera. The camera is champion-locked, so a held world
point sweeps across the screen while the champion walks. `cursor.screen` moves
with no player input at all.

**Measured** (`scratchpad/audit_movement_exact.py`, all 125 matches, exact
replication of `_parse_movement` schema-3 + `discretize_movement(bins=21)`):

```
BC movement-target bin transitions : 552,909  (15.55% of frames)
  from a genuinely new command     : 313,840  (56.8%)
  from camera-projection DRIFT     : 239,069  (43.2%)   <-- pure label noise
new commands producing NO transition: 190,168 (37.7% of 504,008 accepted updates)
```

The 1% dead-band was tuned right at the edge of the drift distribution, so it
cannot separate the two:

```
|delta cursor.screen| on HOLD frames (no new command): p50 0.417%  p90 0.972%  p99 1.667%  max 34.0%
|delta cursor.screen| on COMMAND frames             : p50 7.266%  p90 24.30%  p99 43.61%
```

p90 of the drift is 0.972%, i.e. *just* under the 1.0% dead-band — 8.34% of
hold frames leak through, and 12.3% of hold frames change bin.

**Impact.** The gate head is trained on a transition signal whose base rate is
15.55% and whose precision is 56.8%. Nearly half of every "issue a new movement
command now" gradient points at a frame where the player did nothing. The
categorical is simultaneously trained to point at a *stale* world location that
happens to have drifted into a new cell. This is the single largest label-quality
defect found.

**Fix.** `clicks.json` already contains the exact ground truth and is currently
unused by the dataset: **359,349 movement clicks** across the corpus, each with a
`game_t` and a world `(x, z)` — 2.02 clicks/s, i.e. a real command every 9.9
frames. Build the movement target from clicks + casts as *events*, not from a
held-and-reprojected position:

* transition ⇔ a click/cast event lands on this frame (from `clicks.json`), full
  stop — delete the dead-band heuristic;
* the target position is that event's world point projected with *that frame's*
  camera;
* between events the target is "hold" by construction, with no drift.

This yields ~359k clean transitions instead of 553k with 43% noise, and removes
the need for the schema-3 dead-band entirely.

---

## 2. 24.1% of frames have no movement target at all — off-screen commands are silently dropped

**What the code assumes.** `_parse_movement`'s docstring says `cursor.screen` is
`None` only "before any input or when off-screen", and the handling is to hold
the previous value. `__getitem__` then marks every replay frame
`cursor_valid=True` (`replay_dataset.py:610-612`), so the action-conditioned
dynamics is told the held value is a real action.

**What the data actually is.** `project()`
(`scripts/aggregation/pipeline.py:817-825`) returns `None` for any point outside
`[0,1280) x [0,720)`.

**Measured** (`scratchpad/audit_cursor.py`):

```
frames                                : 3,554,768
cursor.screen present                 : 2,697,248 (75.88%)
cursor.screen None                    :   857,520 (24.12%)
  ...of which cursor.world WAS known  :   701,750 (81.8% of the Nones)
held-forward (0.5,0.5) before first cursor : 146,839 frames (4.13%)
```

So on **19.7% of all frames** the pipeline knew exactly where the player had
commanded — it just projected outside the viewport and threw the number away.
Off-screen commands are not an edge case in League; they are edge-of-screen
movement and minimap clicks, i.e. exactly the rotation/repositioning decisions.

A further 146,839 frames (4.13%) get the `(0.5, 0.5)` fallback, which in 21 bins
is bin (10,10) — the *modal* cell (20.5% of all frames). The fallback is
indistinguishable from a real centre command.

**Impact.** ~a quarter of the movement supervision is a stale hold marked as
valid. Combined with finding 1, the effective usable movement supervision is a
minority of frames.

**Fix.** Do not clamp to the viewport in the label pipeline. Emit the raw
projected coordinate (which can be < 0 or > 1) plus an `on_screen` flag; let the
dataset decide. For genuinely off-screen targets, either emit the clipped
direction with a distinct "off-screen" class, or mask the frame out of the
movement loss (`cursor_valid=False`) instead of feeding a stale hold.

---

## 3. Nothing is held out during training — every training-time eval number is in-sample

**Measured** (see `scratchpad/audit_b_*`):

* **BC** (`scripts/train_agent_finetune.py`) has no `--val`/`--holdout`
  argument at all. Its built index proves it trains on everything:
  `data/phase2_bc_gate1060/dataset_cache.pt` → `125 matches, 422,527 sequences,
  125 distinct video_ids`.
* **Dynamics** (`scripts/train_dynamics.py:527`) defaults `--holdout-videos 0`,
  and no production launcher overrides it. The fallback at `:1507-1513` is
  `val_batch = next(iter(dataloader_short))` — **a training batch**. The BC
  policy's own backbone confirms it: `rollout_stage/desktop_resume_8775_stripped.pt`
  carries `args['holdout_videos'] = 0`. Every `eval/psnr_tau*` and
  `eval_rollout/psnr_h*` on that lineage is in-sample.
* `_pick_holdout` (`train_dynamics.py:278`) takes `sorted(vids)[-n:]`, which on
  the mixed corpus returns **YouTube** ids (`VJE_9YpwNXw`, `VVlLa1p7aCQ`) — so
  `--holdout-videos 2` never holds out a replay game.
* `scripts/eval_bc_sim.py:37` defaults `--match NA1_5549995114`, which is in the
  training set.

**The good news.** The nominated holdout is genuinely clean. The tokenizer's
`action_labeled_352png_holdout_flat` = 5 matches
(`NA1_5549981347`, `NA1_5550450386`, `NA1_5551132630`, `NA1_5551825358`,
`NA1_5552703163`); all 5 are in `valid_games.json:frames_only`, none has a latent
pack, and their intersection with the 125 BC/dynamics matches is **empty**. Any
number quoted from those games is trustworthy.

**But** all 125 BC games *were* seen by the tokenizer (142 of 147 matches were in
`action_labeled_352png_train_flat`). For a reconstruction model this is mild, but
it means "held-out latents" from the v7 tokenizer are not held out at the pixel
level.

**Impact.** An apparent "ceiling" measured on a training batch is not a ceiling —
it is a fit. Any conclusion of the form "the model plateaus at X" drawn from
training-time metrics on this lineage has to be re-measured.

**Fix.** Set `--holdout-videos` to a nonzero value *and* fix `_pick_holdout` to
sample from `NA1_*` only; add a `--val-matches` list to `train_agent_finetune.py`
and reserve the 5 tokenizer-holdout games (they already have frames; they need a
latent pack from the same tokenizer).

---

## 4. `enemy_visible` is wrong on 29.6% of frames; `enemy_hp_frac` is supervised on invisible enemies

**What the code assumes.** `_parse_state`
(`replay_dataset.py:336-377`) sets `enemy_visible = 1` iff the lane opponent has
an entry in `label.visible_heroes`, and unmasks `enemy_hp_frac` whenever that
entry carries hp.

**What the data actually is.** `pipeline.py:1533-1537` is explicit:

> "Include ALL heroes (not just camera-visible). Off-screen heroes get
> `screen=None` but their stats … are still captured every frame"

So `visible_heroes` is a *memory* list, not a visibility list. The only real
visibility signal is `screen != None`.

**Measured** (`scratchpad/audit_analyze.py` §6):

```
lane opponent present in visible_heroes : 1,932,037 frames (54.4%)
lane opponent actually ON SCREEN        :   879,289 frames (24.7%)
=> enemy_visible target = 1 while the enemy is NOT in the frame:
     1,052,748 frames = 29.6% of ALL frames = 54.5% of the positive labels
```

**Impact.** The `enemy_visible` head is trained to say "visible" more than half
the time it says so incorrectly — it can at best learn the *memory list*
membership, which is unlearnable from pixels. Worse, `enemy_hp_frac` is
supervised on all 54.4% including the 29.6% where the enemy is nowhere in the
input; at `--aux-state-weight 0.5` this drives a large unlearnable residual into
the shared agent tokens.

**Fix.** The minimal change is `seen = (vh.get("screen") is not None)`, keeping
`mask[i, 3] = 1` and setting `mask[i, 2] = 1` only when the enemy is on-screen.
That removes the 29.6% of clearly-unlearnable frames — **but see finding 15: a
screen coordinate still does not guarantee the enemy is rendered**, so a residual
fog-of-war subset survives even after the fix. The fully correct signal needs a
visibility/fog flag that `pipeline.py` does not currently read from memory.

---

## 5. The 21-bin movement grid quantizes away 37.7% of real commands (lead 2 — confirmed, restated)

The original framing ("what fraction of movement is smaller than one bin") does
not apply, because the target is an **absolute screen position**, not a delta.
The correct measurement is: of the command updates that the dataset accepts, how
many fail to change the bin?

**Measured**, at the production default `--movement-bins 21`
(`train_agent_finetune.py:108`; the `11` at `:475` is smoke-test only):

```
bin width = 1/20 = 5.0% of screen = 64 px in x, 36 px in y
accepted command updates              : 504,008
  ...producing no bin change (lost)   : 190,168  (37.7%)
target distribution: 441/441 cells used, top cell 20.5% of frames,
                     entropy 4.453 nats = 85.9 effective cells (uniform = 6.089 nats)
```

At 41 bins the loss falls to 31.3% but drift noise rises to 55.6% of transitions;
at 11 bins the loss is 48.4%. **There is no bin count that fixes this while the
target is a drifting held position** — the noise and the quantization loss trade
off against each other. Fixing finding 1 first (event-based targets) is what
makes a finer grid pay off.

Note also the *decision* resolution: 5% of screen is 64 px horizontally at 720p.
A minion HP bar is ~33 px wide natively (finding: aspect). One movement bin is
wider than two minions. The policy cannot express "click this minion" — only
"click roughly over there".

---

## 6. A phantom movement command fires at 89.7% of attack/ability endings

**What the data actually is.** `cursor_world` switches *source* when an action
starts and ends: `cast_target` while casting/attacking, `last_click_world`
otherwise. When an attack ends, the target teleports from the attacked unit back
to whatever movement click was issued before the attack began.

**Measured** (all 125 matches):

```
attack/ability end events                                   : 22,122
cursor target jumps > 1% of screen within 3 frames          : 21,287 (96.2%)
cursor target jumps > one full 21-bin cell (>5% of screen)  : 19,852 (89.7%)
```

**Impact.** ~19.9k of the 553k bin transitions (3.6%) are artifacts, and they are
*structured*: they fire immediately after every trade and every last-hit —
precisely the frames where movement matters most. The policy is taught to fling
the cursor back to a stale point after each attack.

**Fix.** Same as finding 1 — make the target event-driven. If the held-position
formulation is kept, at minimum do not fall back to `last_click_world` after a
cast; hold the cast target until the *next real click*.

---

# TIER 2 — real data loss, correctable

## 7. 4.9% of label frames have no latent; 9.3% of captured game time has no frames

**Truncated latent packs.** 14 of 125 `.pt` files are shorter than their
`labels.json` (`scratchpad/audit_b_counts.json`). This is what
`replay_dataset.py:241-245`'s "using min(...) and dropping the rest" warning has
been reporting all along:

```
             match    N_lat  n_label  missing  frac_lost
    NA1_5551763045     1078    34114    33036      0.968
    NA1_5551743405    10503    38732    28229      0.729
    NA1_5551715757    17871    38738    20867      0.539
    NA1_5551736612    17290    36912    19622      0.532
    NA1_5551627612    19950    38574    18624      0.483
    NA1_5552670437    20765    38620    17855      0.462
    NA1_5552652535    25603    38613    13010      0.337
    ... (7 more)
TOTAL  3,381,817 latents vs 3,554,768 labels -> 172,951 frames (4.9%) unusable
```

Consistent with an interrupted `pretokenize_replay_v7.py` run. `--resume` skips
any match that already has a `.pt`, so a re-run will **not** repair these.

**Truncated frame recordings.** Separately, 19 matches' PNG record stops long
before the game ends (measured as label span vs the span of `clicks.json` casts):

```
frame-record coverage of the game: p05 0.35  p25 1.00  p50 1.00  mean 0.909
matches with coverage < 0.75 : 19
game-seconds recorded   : 49.4 h
game-seconds lost after the record ended: 5.0 h (9.3%)
worst: NA1_5553385868 (0.18), NA1_5553395019 (0.29), NA1_5550013959 (0.29)
```

These are the same 19–26 matches that appear as "short games" (duration < 15 min)
in the corpus stats — they are not short games, they are truncated recordings.
The corpus is therefore biased toward early game, and any terminal/outcome signal
on those matches is anchored to the wrong frame.

**Fix.** Delete and re-tokenize the 14 short packs (`--resume` will not do it);
either re-record or explicitly flag the 19 truncated matches so outcome/terminal
terms are disabled for them.

## 8. 11.0% of ability casts are dropped as out-of-range

`_parse_abilities` (`replay_dataset.py:470-474`) computes
`i = round((gt - gt0)/step)` and drops anything outside `[0, T)`, warning but
continuing.

**Measured:**

```
cast events total                        : 30,338
out-of-range                             :  3,353 (11.05%)  -- ALL past the end, none before
max seconds past the end of frame record : 1,375 s  (22 matches affected)
per-key label counts after the drop:
   Q 11,530 of 12,942 GarenQ   (10.9% lost)
   E 10,899 of 12,236 GarenE+GarenECancel (10.9% lost)
   W  2,040   Recall 1,103   R 470   Ignite 401   Flash 311
unmapped and intentionally dropped: 245 (0.8%)  -- SummonerTeleport 100,
   SuperRecall 81, SummonerExhaust 55, SummonerHaste 9
label collisions (two casts of a key rounding to one frame): 2
```

This is a *symptom* of finding 7, not an independent bug: the memory/click
recorder outlives the frame recorder. Once truncation is fixed the loss goes to
~0. No action needed beyond fixing 7, but the sparse classes (R at 0.013% of
frames, Flash at 0.009%) can least afford an 11% cut.

## 9. Frames lag labels by a fixed +1.47 frames

`gt` is **synthetic** — `pipeline.py` builds it as `gt = rec_start + fi/FPS` and
the code comment concedes it "assumes the engine writes PNG #0 at exactly
rec_start". Verified with a sharp event test (`scratchpad/audit_align2.py`):
locate the frames where `champion_world` teleports > 1500 units (recall / death /
Flash) and find the corresponding visual cut in the PNGs.

```
156 teleport events across 20 matches
offset (PNG index - label index): mean +1.47  median +1  std 0.68  range 0..+3
histogram: {0: 5, 1: 85, 2: 54, 3: 12}
corr(offset, position-in-game) = +0.072   <-- NO accumulating drift
```

Cross-checked with a whole-signal correlation of label velocity vs frame motion
(`scratchpad/audit_align.py`), which also peaks at +1..+2.

**Impact.** Labels *lead* the pixels by ~1.5 frames (≈75 ms). Because BC scores
MTP offsets `n >= 1`, the effective horizon is `n + 1.5` frames rather than `n`,
and the action-conditioned dynamics receives `a_t` about 1.5 frames before the
pixels that action produced. Modest, and — importantly — it is a **fixed** offset
with no drift, so it is correctable by a constant shift.

**Fix.** Shift the label↔frame association by 1 or 2 (i.e. pair `labels[i]` with
`frames[i+1]`), or better, record real PNG timestamps instead of synthesising
`gt`. Verify by re-running `audit_align2.py` and checking the mode moves to 0.

## 10. Attack labels: chaining is fine, but Q-empowered autos are misfiled (lead 3)

**The chained-auto hypothesis is REFUTED.** `label.action.type` derives from the
per-frame memory `spell` field, which goes null between attacks, so each auto
produces its own state transition:

```
attack-run lengths: p50 6 frames (0.30 s) p90 10  mean 8.2   -- i.e. one windup per run
```

Direct comparison against `raw_mem.json` at native sample rate, restricted to the
labelled time window:

| match | raw_mem BasicAttack episodes | labels AA transitions | capture |
| --- | --- | --- | --- |
| NA1_5550045094 | 179 | 179 | **100%** |
| NA1_5550073400 | 190 | 188 | **99%** |
| NA1_5551243795 | 72 | 72 | **100%** |

**What IS lost.** `classify_spell` (`pipeline.py:825-845`) tests
`"basicattack" in name` first, so `GarenBasicAttack`/`GarenBasicAttack2` → attack,
but:

* `GarenQAttack` — the Q-empowered auto-attack, a real right-click — starts with
  `garen` and has suffix `q`, so it is classified **`ability`**, not `attack`.
* `GarenCritAttack` — matches neither branch, falls through to **`other`**.

Counting all `*Attack` episodes instead of just `BasicAttack`:

```
NA1_5550045094: 226 any-Attack episodes vs 179 labeled AA  -> 79% captured
NA1_5550073400: 241 vs 188 -> 78%
NA1_5551243795:  86 vs  72 -> 84%
```

So the AA label undercounts real attack commands by **~21%**, not by the 5–10×
the lead hypothesised. Corpus-wide AA base rate: 18,713 labels = 0.526% of frames.

**Fix.** In `classify_spell`, check `"attack" in nl` (after the recall/summoner
branches) and return `attack` for any `*Attack` suffix; keep the separate ability
label for the Q press itself, which already comes from `clicks.json`.

## 11. `level` reads 19 and 20

```
level histogram: ... 17: 180,862  18: 94,263  19: 49,698  20: 9,714
59,412 frames (1.673%) out of [1,18], in 25 of 125 matches
```

League caps at 18. `_parse_state` writes `state[i,1] = level/18.0`, so the aux
target is 1.056–1.111 for those frames — outside any bounded head's range, giving
a permanent loss floor, and implying the level read is biased +1/+2 in those 25
matches. `own_hp_frac` is clean (`[0,1]`, 0.000% out of range).

**Fix.** Clamp to `[1,18]` and mask frames that were out of range, or re-derive
level from the XP offset. Low effort, small payoff.

## 12. Reward head wastes 85% of its output range

```
per-frame reward (gold_scale 1e-3 * delta gold_total, + death penalty):
   exactly 0 on 89.9% of frames   mean 3.1e-4   p99 0.0011   max 1.090   min -0.200
episode return (sum of dense term): median 10.82   max 18.40
head: 255 twohot buckets over symlog +-3  (= raw +-19)
   observed target span: buckets -7.7 .. +31.2  ->  39 of 255 buckets ever touched
   99.36% of targets lie within HALF a bucket of zero
```

The `±3` bound was chosen for *returns* (median 10.8, which fits), but the head is
trained on **per-frame reward** (`reward_mtp_loss` applies `symlog(rewards)` to
the raw per-frame tensor), whose range is `[-0.20, +1.09]`. At 255 buckets the
resolution is still adequate (a 21-gold last-hit ≈ 0.88 of a bucket), so this is
inefficiency rather than breakage.

**Fix.** Either narrow `bucket_low/high` to about `±1.2` for the reward head
(keep `±3` for a value head), or raise `gold_scale` so a last-hit lands ~1 bucket
under a narrower range. Not urgent.

## 13. Movement bins are anisotropic

`_parse_movement` normalises x by 1280 and y by 720, then `discretize_movement`
applies the same 21 bins to both axes. One bin is **64 px horizontally, 36 px
vertically**. The x-marginal is correspondingly more peaked (36.1% in the centre
cell vs 28.8% for y). A single isotropic grid (e.g. 21 in y and 37 in x) would
make a bin square in screen space.

## 14. Train/deploy gap: training frames have no HUD, the live capture does

Verified empirically across 8 matches (500-frame variance maps) and visually
(`scratchpad/audit_frames_multi.png`, `scratchpad/audit_mask/hud_region_zoom.png`):

```
zero-variance pixels in replay frames: 0 / 123,904  (0.0000)
always-black pixel fraction          : 0.0000
minimap slot, HP/ability slot        : plain terrain
```

The replays were recorded with the in-client HUD disabled. Consequences:

* The world model / BC policy **never see** own HP, mana, cooldowns, item slots,
  gold, level, the scoreboard, or the minimap. All of that reaches the model only
  through `labels.json` memory reads (which is why `probe_hp_fulldim.py` and
  `validate_hp_reader.py` exist). This is an observability limit of the corpus,
  not a bug — but it invalidates any assumption that the agent can read its own
  cooldowns or the map.
* `scripts/agent_infer.py:132` at inference does nothing but
  `cv2.resize(frame, (352,352), INTER_AREA)` on the captured screen. If the live
  game runs with its HUD on, roughly a fifth of the model's input is a region
  type it has literally never seen. (`scripts/play_live.py:86` notes that HUD
  masking at inference "did not change the policy's behavior" — consistent with a
  policy that is not using that region either way, but the domain gap is real.)

**Fix.** Play with the HUD hidden (League: `Ctrl+U` / the `HideUI` binding) so the
live pixels match training, or mask/black the HUD regions in `agent_infer.py`
before the resize — but note the training distribution has *terrain* there, not
black, so hiding the HUD is the correct match.

## 15. `screen != None` means "inside the frustum", not "visible in the frame"

`project()` returns a coordinate for any world point inside the viewport,
regardless of fog of war. Verified directly
(`scratchpad/audit_overlay_check.png`, `audit_overlay_near.png`): in
`NA1_5550968662` the lane opponent Nasus carries a screen coordinate while
sitting only **934–1308 world units** from Garen — close enough to fill a large
part of the frame — and there is no Nasus model anywhere in the picture. He is in
fog. The Garen marker in the same frames lands exactly on the Garen model, so
this is not a projection error; it is what `project()` means.

**Impact.** It compounds finding 4: even the "obvious" fix (`screen is not None`)
still labels fogged enemies as visible. It also means any downstream consumer
reading `visible_heroes[*].screen` as a detection target (overlays, casting
probes, an IDM) is training on partially phantom boxes.

**Fix.** Capture a per-hero visibility/fog flag in `pipeline.py` alongside `pos`,
or gate on `screen is not None AND hero is within the champion's vision radius`
as a cheap approximation.

## 16. `champion_world` repeats on 26.6% of consecutive frames

`raw_mem.json` is sampled at ~25 Hz in game time; `pipeline.py` maps it onto the
20 Hz frame grid with `_nearest(...)` (a nearest-neighbour lookup within
`MAX_MEM_GAP = 0.1 s`), so some frames reuse the previous memory sample and
others skip one.

```
consecutive frames with IDENTICAL champion_world : 942,332 / 3,548,533 = 26.6%
per-frame |d world| when nonzero                 : p50 18.5, p90 35.2 units
```

Any per-frame velocity derived from labels therefore alternates between 0 and
~2x the true step. `label.movement.heading_*` is computed with a 10-frame
lookahead so it is unaffected; anything that differentiates `champion_world`
frame-to-frame is not. Also note this is why the AA state can never be finer than
~40 ms and why the near-frozen-frame analysis (finding 17) should not be blamed
on labels.

**Fix.** Interpolate the memory stream to the frame grid instead of
nearest-neighbour, or sample memory at ≥ 40 Hz.

## 17. Near-frozen frame runs

0 exact duplicates and 0 pixel-identical pairs across 23,992 consecutive pairs
(8 matches) — **no dropped or repeated video frames**. But 11.2% of consecutive
pairs have mean-abs-diff < 1/255, and these sit in *contiguous* runs, not
scattered: worst match 24.5% still, 19 runs, max run 121 frames (~6 s), 713 of 734
still frames in runs ≥ 10. A world model trained on such windows gets blocks of
trivially predictable targets, and any rollout PSNR sampled from one is inflated.

---

# TIER 3 — leads that turned out to be non-issues

## R1. Aspect squish is NOT what is killing minion HP bars (lead 1 — REFUTED)

The frames are a straight `cv2.resize(img, (352,352), INTER_AREA)` from 1280x720
(`pipeline.py:1327`) — full-bleed squish, no letterbox, verified by measuring the
top/bottom rows (mean intensity 51.0 / 55.0, not 0). Letterboxing was never
implemented (0 hits for `letterbox|copyMakeBorder|pad_to_square`).

Measured minion HP-bar geometry (n = 3,133 components over 12 matches × 41 frames;
hand-verified against `NA1_5549995114/frames/005000.png`):

| | squish-352 (shipped) | letterbox-352 | native 1280x720 |
| --- | --- | --- | --- |
| minion HP bar | **9.0 x 1.0 px** | 9.0 x **0.56** px | 32.7 x 2.05 px |
| champion HP bar | 16 px | 16 px | 58.2 px |
| HP quantization | **11.1% per px** | 11.1% per px | 3.1% per px |

**The premise of the lead is arithmetically wrong.** At a fixed 352-wide canvas,
horizontal sampling is 352/1280 = 0.275 px/px under *both* squish and letterbox —
the bar is the same 9 px wide either way. What differs is *vertical*: 0.489
(squish) vs 0.275 (letterbox). Since HP bars are thin horizontal marks, the squish
is the pipeline that *preserves* them; letterboxing collapses the 1 px bar to
0.56 px. A synthetic test on native-geometry 33x2 bars at 10 fill levels:

```
squish-352    : 9/10 bars detected, 8/10 distinct widths
letterbox-352 : 4/10 bars detected, 4/10 distinct widths
squish-512    : 10/10, 10/10
native        : 10/10, 10/10
```

The real cost is **total resolution**, not aspect: 1280 -> 352 is a 3.64x
downscale, and that alone sets the 9 px bar and its 11.1%-per-pixel HP
quantization. To get ≥16 px bars you need a 626x626 square input (or
626x353 aspect-preserved). The one genuine aspect cost is modest: at an equal
pixel budget, a non-square 469x264 gives 12.0 px bars (8.3% quantization) — +33%
horizontal detail bought by halving vertical.

**Can we re-extract at native resolution? No.** All 147 matches / 769 GiB are
352x352; `pipeline.py:1779` deletes each `.rofl` after processing, and
`scripts/yt_pipeline.py:8-9` already admits the corpus is "locked at 352x352 since
its originals were deleted". `ssh windows` did not respond. Two partial escape
hatches on `ssh desktop`, both with **zero match-ID overlap** with the trained
corpus: 17 deprecated 1920x1080 `.avi` games (HUD-on, so not distribution-matched)
and 98 raw 1920x1080 YouTube videos (62 overlap `yt_pretrain_garen`, fully
re-extractable for the pretrain side).

**Where to actually look.** `scratchpad/hp_recon_stills/minion_recon_montage.png`
already shows the v7 tokenizer smearing minion bars into blobs while champion bars
survive. A 9x1 px bar is 3.5% of one 16x16 patch at `patch_size=16` — the
bottleneck is far more likely the tokenizer's patch size / latent budget than the
resize.

## R2. Corrupt stats do NOT reach the reward (lead 4 — REFUTED)

The `-3.77e22` gold is real and widespread — but `gold` is not consumed anywhere:

```
champion_stats.gold  : 871,635 frames corrupt (24.55%) in 30/125 matches,
                       |value| ranging 6.47e+03 .. 2.47e+35
champion_stats.gold_total : 0 corrupt (0.0000%), range 500 .. 18,920
                            0 decreases in 3,548,533 consecutive pairs (monotone, as assumed)
hp / hp_max          : 0 out of range
champion_screen      : 99.0% present, 0.00% out of bounds
champion_world       : 100% present, 0.00% out of bounds, x[52,14602] y[193,14656]
```

`pipeline.py:1546` already flags it: `"gold": ... # known broken offset on 16.9`.
`_dense_solo_gold` reads **`gold_total`**, which is clean and monotone; the
`RewardConfig` guards (`_safe_float` never zero-filling, gap resets, the
`isfinite` hard-fail) are correct and, on this corpus, never fire. **No guard is
missing and no reward is poisoned.** The only residual risk is that a future
consumer reads `gold` by mistake — worth deleting the field or renaming it
`gold_BROKEN`.

## R3. The latent ↔ frame_indices contract HOLDS (REFUTED)

All 125 packs scanned: strictly ascending 125/125, 0 duplicates, `min == 0`
125/125, `frame_indices == arange(N)` on 121/125. Crucially, `_index_match`
does **not** assume `frame_indices[i] == i` — it stores a genuine
`frame_to_idx[start_frame]` lookup and only emits windows inside runs of
*consecutive* frame numbers, so position and frame number advance in lockstep.
Proved exhaustively by replaying `_index_match` at the production
`seq_len=16, stride=8`:

```
total windows across 125 matches : 422,527   (matches the real dataset_cache.pt exactly)
MISALIGNED windows               : 0
latent out-of-bounds windows     : 0
label out-of-bounds windows      : 0
```

Residual risk worth closing cheaply: the invariant is unenforced. A one-line
`assert (np.diff(fi) > 0).all()` in `_index` would make it impossible to
regress. Separately, `_cache_meta` (`replay_dataset.py:153-158`) keys the index
cache on `(latents_dir, seq_len, stride, schema)` and **not on the match list**,
so changing which `.pt` files are in the directory silently reuses a stale index.

## R4. The HUD mask does not damage replay frames (REFUTED)

`scratchpad/hud_valid_mask_352.pt` masks 34.09% of the frame (top rows 0-31, left
cols 0-17, right cols 335-351, bottom rows 280-351). On YouTube it is a correct,
conservative superset (IoU 0.848 with the actually-black region; 0.0% of black
pixels escape it). On replays it has no relationship to the data at all (IoU
0.000; 100% of masked pixels carry live variance). It is only ever applied under
`--pixel-hud-loss` **and** the `is_yt` guard at `train_dynamics.py:798-800,1104`;
`pretokenize_replay_v7.py` applies no mask. Had it leaked onto replays it would
have deleted a third of the playfield — worth an explicit assertion, but it does
not currently fire.

## R5. No duplicate or dropped frames (REFUTED)

0 exact md5 duplicates and 0 pixel-identical pairs in 23,992 consecutive pairs
across 8 matches; 0 entirely-black frames (min per-frame mean 38-49/255). See
finding 17 for the near-frozen runs, which are a different phenomenon.

## R6. `project()` is NOT miscalibrated (REFUTED — and a warning about the test)

Worth recording because the first test said the opposite. Every screen coordinate
in `labels.json` comes from `project()` with hard-coded, "empirically fit"
constants (`FOV_V = 40°`, `TILT = 56°`, `FLOOR_Y = 52`, `CAM_Y = 1912`,
`CAM_Z_OFFSET = -1292`, `pipeline.py:50-63,817-825`). Since `cursor.screen` is the
only movement target, a scale error there would be catastrophic — so it was worth
checking.

**Test 1 (misleading).** `scratchpad/audit_projection.py`: on frames where
`cursor.world` is unchanged, `cursor.screen` moves only because the camera moved,
so its displacement is the predicted optical scroll of a fixed world point;
compare against `cv2.phaseCorrelate` of the PNG pair. Result over 1,969 pairs:

```
x: measured = 0.503 * predicted (r = 0.639)
y: measured = 0.473 * predicted (r = 0.603)
lag sweep (MSE): -2: 6.87  -1: 6.18  0: 4.94  +1: 4.18  +2: 4.66   <- confirms the +1 offset
```

An apparent 2x error. **It is an artifact.** In a champion-locked camera a large
fraction of the frame does *not* scroll with the terrain — the champion model, its
floating HP bar, and every other unit's HP bar/nameplate move with their units,
and nearby minions advance with the wave. Phase correlation returns the dominant
shift over that mixture and is attenuated toward zero. Template matching
(`audit_projection2.py` and a direct `matchTemplate` run) was worse still, because
Summoner's Rift terrain is highly self-similar — shifts scattered over ±130 px
with match scores of 0.26–0.82.

**Test 2 (decisive).** Draw the label's own `champion_screen` on the frame at
moments when the champion is *far from screen centre* — exactly where a scale
error would show, and where the frustum-centre degeneracy does not hide it.
`scratchpad/audit_overlay_offcentre.png` shows 6 frames at 95, 104, 113, 135, 181
and 319 px off-centre: **the marker lands on the champion model every time.**
`audit_overlay_near.png` and `audit_overlay_check.png` show the same for Garen
alongside other heroes.

**Verdict: the projection is correct.** Methodological note for future audits —
do not use global phase correlation as ground truth for camera motion in a
character-locked game; verify against rendered object positions instead.

---

# Recommended order of work

1. **Rebuild the movement target from `clicks.json` events** (findings 1, 2, 5, 6).
   One change fixes the largest defect, the 24% missing targets, the phantom
   post-attack jump, and makes a finer bin grid worth having. The ground truth is
   already on disk and unused.
2. **Turn on a real holdout** for dynamics and BC and re-measure the "ceiling"
   (finding 3). Everything else is guesswork until the metric is out-of-sample.
3. **One-line fix to `enemy_visible` / `enemy_hp_frac`** (finding 4) — currently
   the aux loss is chasing an unlearnable target on ~30% of frames.
4. **Re-tokenize the 14 short packs and flag the 19 truncated recordings**
   (findings 7, 8).
5. Constant frame↔label shift of +1 (finding 9), `classify_spell` attack fix
   (finding 10), `level` clamp (finding 11).
6. If pursuing resolution: the answer is a **larger square input and/or smaller
   tokenizer patches**, not letterboxing (R1). 626x626 is the threshold for a
   16 px minion bar, and the originals for these 147 matches are gone — new
   recordings would be required.
