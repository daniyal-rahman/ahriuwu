# A direction acceptance metric for the movement retrain, and a cursor head

**Date:** 2026-09-02 · Adds `dir_octant_*` to `scripts/train_agent_finetune.py`'s
held-out eval, and an optional cursor regression head behind `--cursor-weight`.

Two things ship here. The first is the number the movement retrain will be
judged by. The second is a second spatial output channel.

The headline result is not either of them. It is that **the metric as specified
would have passed the deployed copier**, and the check that caught it is the one
the task mandated: score the deployed checkpoint, which demonstrably copies, and
refuse to believe a good number from it.

---

## 1. Why `move_event_ce` cannot judge this retrain

`move_event_ce` compares the head's likelihood against a blind
`p(next bin | prev bin)` table worth **0.832 bits**. It is a *ratio test against
a crutch*, so it cannot separate

* "the representation is empty" from
* "the representation's signal is smaller than the crutch",

which is exactly the question the retrain asks. Worse, the retrain's whole point
is to take the movement action *out of the model's input*. That necessarily makes
`move_event_ce` worse — the crutch leaves the input, so nothing can reach the
table's score — whether or not behaviour improved.

It is kept. It is still the sharpest copy-detector in the file. It just does not
get a vote on this run.

## 2. What the new metric is

On **`movement_event` frames only**, never before a game's first real click:

| number | meaning |
|---|---|
| `dir_octant_acc` | 8-way octant accuracy of the **commanded** direction about the champion |
| `dir_persist_acc` | **THE BAR** — the same octant for the *previous* order, re-aimed from the current position. Blind, no pixels |
| `dir_null_mean/sd`, `dir_z` | null **measured** by within-game label permutation (default 200 draws) |
| `dir_med_deg` | median angular error, the interpretable number |
| `dir_med_deg_const` | best constant direction on *this* split, recomputed where it is used |
| `dir_n` | events scored |

The commanded direction is read off the movement **categorical argmax** with the
gate (or the `NO_OP` class) excluded — the same readout `move_event_ce` uses, and
the action the policy would issue at temperature 0 on a frame where it fires.

It rides on the eval's existing forward passes, so it costs no GPU time, and it
reads `labels.json` for the val games only. **No dataset change, no cache schema
bump** (see §6).

### 2.1 Geometry: three traps, each measured

The probe in `docs/DECISION0_REPLICATION.md` works in **world** space
(`atan2(z − hero_z, x − hero_x)`). The trainer only has **screen**-space targets.
Three corrections turn one into the other, each verified on real click events.

**1. The champion is not at screen centre.** Camera-locked but loosely: median
**0.041** of the screen from `(0.5, 0.5)`, 99th pct 0.132. Using the centre as
the origin flips the octant on **30.1%** of all events (16.2% of far ones). So
the real `champion_screen` is used.

**2. Screen space is a sheared view of world space.** The ground plane is
compressed vertically by camera tilt and the 16:9 horizontal FOV; a raw screen
angle disagrees with the world octant on **20.1%** of events. One constant undoes
it, *derived* rather than fitted:

```
dx_screen ∝ dx_world / tan_h        dy_screen ∝ −dz_world · sin(tilt) / tan_v
  =>  world angle = atan2( −dy_screen · tan_v/(sin(tilt)·tan_h),  dx_screen )
```

With the shipped projection (fov_v 40°, tilt 56°, 16:9) that is **0.679**.
Grid-searching the constant against true world angles independently finds
**0.69**. It is taken from each game's own `projection` block, not hardcoded.

**3. The pre-first-click window.** `clicks.json` never starts before ~60 s, and
`--prefirst-mode heading` *synthesizes* movement events in that window from
`heading_screen` — a consequence of the action, not the command. Every frame
before a game's first real click is dropped.

**Validation of the shipped class** against world-space ground truth
(`clicks.json`'s own `hero_x/hero_z`), the 6 val games, n = 14,589 events:

| rows | octant agreement with world truth | median &#124;angle diff&#124; |
|---|---|---|
| all events | **93.95%** | **0.514°** |
| dist > 250 world units | **97.13%** | **0.404°** |

That is the probe's quantity, computed from the batch.

## 3. Known-good validation (before any checkpoint was read)

Four synthetic heads, in `--smoke-test` so they are permanently guarded. The
champion positions, first-click frames and y-scale are synthetic too, so the
geometry is exercised, and the direction marginal is deliberately skewed.

```
  DIRECTION metric KNOWN-ANSWER CHECK (the retrain's acceptance test):
      PERFECT head   -> octant 100.00%  median 0.000deg   (must be 100% / 0deg)
      UNIFORM head   -> octant  12.46%  vs MEASURED null 12.48+-0.69%  z=-0.03  median 93.2deg (must be ~90)
      MARGINAL-only  -> octant  21.01%  vs MEASURED null 19.87+-0.73%  z=+1.57 (must be ~0)
      ^ that head knows only the direction marginal, nothing from pixels. Against an ASSUMED 1/8 null it would have scored z=+11.7 -- PROOF the measured null is load-bearing
      COPIER head    -> octant  58.27%  == persistence BAR 58.27% on the same rows, z=+53.1 vs the null
      ^ a copier BEATS the permutation null by 53 sd. The null cannot catch it (it shuffles away time); the persistence bar can. Judge a run on the BAR.
      pre-first-click cut dropped 220 events, kept 2480
```

Read those four rows in order. The first two are the checks the task asked for.
The third shows why the null is *measured*: a head that has learned only each
game's direction marginal — no pixels — lands on its measured null at z ≈ 0, but
against an **assumed 1/8** it would have read **z = +11.7**, a large fake
discovery. The fourth is §4.

## 4. The deployed checkpoint, and the bar it forced into existence

The task's instruction was explicit: the deployed checkpoint copies, so it should
land near the null; if it scores well, the metric is wrong.

**It scored well.** First measurement, `agent_finetune_latest.pt` at step
102,420, 6 held-out games:

```
dir_octant = 33.60%  vs MEASURED null 14.83 +- 2.20%  (z = +8.5)   n = 247
dir_median_angle = 41.4 deg   (best constant on this split 79.4 deg)
```

33.6% against a 15% null, and a median angular error of 41.4° — statistically
indistinguishable from the frozen-latent ridge probe (37.8%, 43.0°). For a model
that matches a *no-pixel lookup table to 0.0002 nats* on `move_event_ce`.

At the larger sample the resemblance gets *worse*, not better — n = 1,313 events:
**37.70%**, z = **+26.4**, median **35.4°**. That is the probe's 37.8% to two
decimal places. A metric reporting only accuracy-vs-null would have declared the
copier's direction sense equal to the frozen-latent probe's.

The metric was not arithmetically wrong. It was **incomplete**, and the doc
already said why:

> | direction to the **held** previous target, re-aimed from the current position | **0.5479** |

**A blind copier scores 54.79% on this exact quantity.** Consecutive clicks are
spatially autocorrelated, so the direction to the *old* target is already most of
the direction to the new one. The permutation null cannot catch that: shuffling
labels *within a game* destroys temporal order, so a copier clears the null by
tens of sigma while knowing nothing about pixels. Reported alone, `33.6% / z=+8.5`
reads as vision — the same class of error as the ratio test it replaces.

So **persistence, computed on the identical rows, is the bar**, exactly as
`blind_table_bar` is the bar for `move_event_ce`. Both numbers are reported, and
the printed verdict keys off the bar, not the null:

The deployed checkpoint, re-scored with the bar, n = 1,313 events:

```
move_event_ce = 4.1161  acc=8.76%  n=1313   [bar 4.1214, deployed ref 4.1212]
      dir_octant=37.70%  [BAR: blind copier 45.85% on these rows]  vs MEASURED null 13.86+-0.90% (z=+26.4)  n=1313
      -> above null but BELOW the blind copier (45.8%) - NOT evidence of vision
      dir_median_angle=35.4deg  (copier 22.6deg, best constant on THIS split 86.0deg; ...)
```

**37.70% against a 45.85% bar.** The model is *worse at direction than repeating
the order already sitting in its own input* — median 35.4° against the copier's
22.6°. That is the honest description of a 146M-parameter network that matches a
no-pixel table on `move_event_ce`, and it is only visible because the bar is
printed next to the accuracy.

The two numbers that look identical mean opposite things, and this is the trap
worth remembering:

| | probe (DECISION0) | deployed checkpoint |
|---|---|---|
| octant accuracy | 37.8% | 37.70% |
| access to the previous click | **none** | **in its input** |
| what a copier gets on its rows | 54.79% (n/a, probe can't copy) | **45.85%** |
| verdict | signal read from **pixels** | **below** what copying alone gives |

Read that way, the deployed checkpoint is what it always was: a **degraded
copier** — above the null because copying carries real directional information,
well below what copying alone achieves.

### The bar reproduces the doc's number, on the doc's rows

The bar is recomputed on whatever rows the eval scored, so it must be checked
against the doc on the doc's rowset. Shipped `DirectionReference`, same 6 val
games, n = 14,558 events:

| rowset | shipped bar | doc |
|---|---|---|
| all post-first-click events | 48.17% | — |
| dist > 250 world units (75.4% of events) | **55.95%** | **54.79%** |

The whole gap is the **rowset**, not a bug: near clicks have noisy directions and
this trainer deliberately scores all of them. Camera drift between the two clicks
is *not* a factor — re-projecting the previous **world** point through the
**current** camera (the doc's own construction) gives 55.46% against the frozen
screen coord's 55.95%, a 0.5-point difference. The frozen coord is also the
operationally correct one: it is literally what `actions["movement"]` holds, so
it is the score a model copying its own input actually gets.

The hardcoded `DIR_OCTANT_PERSIST = 0.5479` is therefore a **sanity reference,
never the bar**, and is labelled as belonging to the doc's far rowset.

### Why the bar survives the retrain

`--movement-action-mode none/event_only` only touches `cursor_valid`; it never
alters `actions["movement"]`. So `dir_tgt_xy` and `dir_prev_xy` are identical in
all three modes. With the crutch removed from the *input*, persistence remains
exactly the right blind opponent — and the model can no longer reach it by
copying, which is the point.

**Acceptance rule for the retrain:** beat the **persistence bar** printed beside
the accuracy, and get `dir_med_deg` down toward the probe's 43°. Clearing the
null alone means nothing — the deployed copier clears it by 26 sigma.

## 5. The cursor head

`actions["cursor"]` (schema 6) is a second spatial channel: 28.9% exact-repeat
and 0.0059 median step vs `movement`'s 91.2% / 0.1075. Added behind
`--cursor-weight` (default **0**, head not even built, so nothing changes and
every existing checkpoint still loads).

**Regression, not bins — and the measurement decides it.** At the movement head's
own `movement_bins=21`, over 66k frames of 4 games:

| bins | width | quantization RMSE | median step / width | consecutive frames in the SAME bin |
|---|---|---|---|---|
| 11 | 0.100 | 0.0362 | 0.05 | 88.5% |
| **21** | **0.050** | **0.0193** | **0.10** | **81.7%** |
| 41 | 0.025 | 0.0097 | 0.20 | 71.8% |
| 101 | 0.010 | 0.0038 | 0.50 | 51.8% |

A 21-bin cursor head carries a quantization RMSE **3.9× the median inter-frame
step** and freezes **81.7%** of consecutive frames into the same bin. That is
manufacturing the ~90%-repeat target that turned the movement head into a copier,
in the one channel added *because* it moves continuously. A continuous head is
the only one that can represent the motion.

**Huber (`beta=0.05`), not MSE.** Normal pointer motion (median step 0.0050, p90
0.022) is far below the knee, so it stays in the quadratic regime where precision
is the point. Above it sit the anchor teleports: `label.cursor.world` is sampled
sparsely and **1% of steps exceed 0.25**, with a measured max of **1.41** — the
full screen diagonal. MSE would let those dominate the gradient.

**MTP offsets n ≥ 1**, mirroring movement. Not for a label leak — the cursor
never enters the model — but for a **crutch**, and this one is worth stating
plainly because it is measured:

| offset | cursor_{t+n} within 0.05 of the movement action already in the input | median dist |
|---|---|---|
| n=0 | 65.1% | 0.027 |
| n=1 | 64.4% | 0.029 |
| n=3 | 55.2% | 0.044 |

Under `--movement-action-mode held` the cursor head can copy the movement action
on ~2/3 of frames. `n ≥ 1` does not remove that. **A good `bc_cursor` under
`held` is not evidence of perception.** Reported baseline: predicting the corpus
mean every frame gives RMSE **0.2734**.

It is a **prediction target only**. `DynamicsModel.embed_actions` reads
`movement`, `cursor_valid` and the ability keys and nothing else — verified in
source and asserted in the smoke test — so the cursor cannot become action
conditioning. Weighted and RMS-normalized **separately** from `bc_loss`, like the
aux state loss, so `bc_loss` keeps meaning what it meant in every earlier run.

## 6. Cache schema: no bump, and why that is correct

The rule is that anything changing `_parse_match`'s **output** goes in
`_cache_meta`. Neither new consumer does:

* `--cursor-weight` only **reads** `md["cursor"]`, which schema 6 already
  produces unconditionally. `run_step` hard-fails with a rebuild instruction if
  the key is absent, rather than training a head on nothing.
* `--direction-metric` / `--direction-perms` read `labels.json` directly, outside
  the dataset entirely.

Schema stays at **6**. Threading `champion_screen` through `_parse_match` was
considered and rejected: it would force every index cache to be rebuilt (a
re-read of ~26 GB of latent packs, there being no `index.pt`) for a number no
training step consumes. The stale-comment ordering left in `_cache_meta` by
`ed0fec3` is fixed, with the read-vs-write distinction written down.

## 7. How it was verified

* **Known-good, before any checkpoint** — §3, four synthetic heads, in
  `--smoke-test`.
* **Geometry against world truth** — §2.1, 14,589 real click events, 6 val games.
* **Deployed checkpoint** — §4, the check that found the missing bar.
* **The bar against the doc** — §4, 14,558 events, agrees to ~1 point on the
  doc's rowset.
* **End-to-end in the real trainer** — a bounded `--max-steps 40` run on the
  login 1060 over 12 staged games (6 val / 6 train) with `--cursor-weight 0.2`,
  exercising dataset → cursor targets → cursor gradient → val eval → checkpoint
  → clean self-termination. Not a training run; scratch checkpoint dir. Its
  step-20 eval, on a head 20 optimizer steps old:

  ```
    [VAL @ step 20] loss=2.4069 bc=1.9731 (abil=0.693 move=1.280) rew=5.5402 aux=0.4576  [6 held-out games]
        dir_octant=11.63%  [BAR: blind copier 58.14% on these rows]  vs MEASURED null 18.53+-2.42% (z=-2.8)  n=86
        dir_median_angle=88.5deg  (copier 10.8deg, best constant on THIS split 63.4deg; ...)
        cursor huber=0.11344 rmse=0.2385 (constant-mean baseline 0.2734; weight 0.2)
  ```

  Exactly the right reading for an untrained head: **at the null**, median angle
  **88.5°** — the best-constant level — and nowhere near the 58.14% bar. The
  cursor head is already slightly under the constant-mean baseline (0.2385 vs
  0.2734) after 20 steps, which is what a working regression head does.

## 8. Scoring an existing checkpoint

`scripts/score_checkpoint_direction.py` runs the trainer's own
`evaluate()` / `run_step()` / `DirectionReference` path over a checkpoint's own
recorded val games — no reimplementation, so it prints what a run would print:

```
PYTHONPATH=src python scripts/score_checkpoint_direction.py \
    --checkpoint data/phase2_bc_clicks/agent_finetune_latest.pt
```

It symlinks only the val packs into `--stage`, because the dataset index
`torch.load`s every `*.pt` just for `frame_indices` and there is no `index.pt`.
Paths resolve `/srv/nfs` vs `/mnt/nfs` at runtime; nothing is hardcoded.

## 9. Reference values

| quantity | value | source |
|---|---|---|
| ridge on one frozen latent frame → click octant | 37.8% | DECISION0 §4 |
| its measured null | 15.4% | DECISION0 §4 |
| uniform chance (**wrong** null) | 12.5% | — |
| best constant per game (oracle) | 20.3% | DECISION0 §4 |
| **held previous target, re-aimed — THE BAR** | **54.79%** | DECISION0 §4 |
| previous-click octant table | 56.34% | DECISION0 §4 |
| median angular error: best constant | 87.9° | DECISION0 §4 |
| median angular error: ridge | 43.0° | DECISION0 §4 |
| cursor constant-mean baseline RMSE | 0.2734 | measured here |
| **deployed checkpoint, this metric** | **37.70%**, bar **45.85%**, median 35.4°, n=1,313 | measured here |
