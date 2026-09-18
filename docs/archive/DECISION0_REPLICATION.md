# Decision 0, independent replication: is the click signal in the frozen v7 latents?

**Date:** 2026-09-02 · **Verdict: A — the signal IS present in the frozen v7 latents,
at high confidence.** The frozen representation carries **0.31 bits** about the human's
next-click direction with **no access to the previous click in any form**, on six
held-out games, against a *measured* null. The deployed movement head extracts
essentially none of it (4.1212 vs a blind table's 4.1214).

This is a from-scratch replication. It shares no code with the nested-probe work: the
event index, the labels, the features, the estimator and the null were all rebuilt from
`clicks.json` + the raw `.pt` latents. Scripts live in the session scratchpad
(`rep/{build_events,traj,extract,probe,probe2,pos_control,cnn,validate,final}.py`), not
in the repo.

---

## 1. Design, and why it fails differently from CE-vs-table

The deployed test asks: *does the model's cross-entropy beat a blind previous-click
transition table's?* That is a **ratio test against a crutch**. Reproduced here in
direction space, it gives the same verdict the deployed test gave:

| octant NLL on the 6 val games (nats; uniform = 2.0794) | |
|---|---|
| blind previous-click table | **1.4795** |
| frozen latent, no previous click | **1.7271** |

A CE-vs-table comparison therefore reports *"the pixels are worse than a blind table"* —
which is **true**, and which is **completely compatible with the pixels carrying 0.31
bits**. The test cannot separate *"the representation is empty"* from *"the
representation's information is smaller than the crutch's."* Those are exactly options B
and A.

So this replication never compares against the crutch. Instead:

1. **Direction, not cell.** Collapse the target to 8 world-space octants of
   `atan2(click_z − hero_z, click_x − hero_x)`. This deletes the radius-jitter nuisance
   dimension that dominates a 256-cell likelihood, and it is the quantity that decides
   whether the champion walks to the right place.
2. **A measured null, not an assumed one.** Chance is estimated by **within-game label
   permutation** (300 draws), which preserves each game's own direction marginal. Uniform
   chance is 12.5%; the measured null is 15.4%; the best constant-per-game oracle is 20.3%.
3. **A discriminative probe, not the trained model.** A fresh ridge (and a small CNN) is
   fit on the frozen 8192-d latent of 105 other games and read out on the 6 held-out
   games. This removes the training objective from the question entirely, which is the
   only way to test "is it *reachable*" as opposed to "did this run use it".
4. **Validated in both directions before interpreting anything** (§3).
5. **Baselines reported separately**, never merged into the null: marginal, per-game
   oracle constant, map-position table, previous-click table, held-target-re-aimed.

This design fails when the representation is genuinely empty — §3's shuffled-label
refit shows exactly what that looks like (17.4% vs a 15.9% null, 0.007 bits) — and it
cannot be fooled by the crutch being large.

## 2. What the probe sees, exactly

* **Input:** one frame's frozen v7 latent, `latents[t] ∈ R^{32×16×16}` from
  `/srv/nfs/datasets/replay_latents_v7_bc`, flattened to 8192 and standardised with
  train-fold statistics. Nothing else. **No previous click, no held movement target,
  no action embedding, no game id, no time, no champion position feature.**
* **Target:** the octant of the *new* click's world-space direction from the champion,
  taken from `clicks.json` (`atan2(z − hero_z, x − hero_x)`), scored **only** on frames
  where a new click landed.
* **Splits:** 111 games with a usable click stream (`NA1_5553776931` dropped:
  `champion_screen` null on 85% of frames). Held-out = the 6 named games; λ chosen on 12
  further games held out of the ridge fit; the remaining 93 games train the probe.
  **314,917** click events total.
* **Traps handled:** scored only on click events (never on held frames); every event is
  after the first click of its game, so the (0.5, 0.5) pre-first-click sentinel window is
  structurally excluded; `t−10` and `t+20` are clipped inside each game and the latent
  packs are shorter than `labels.total_frames` on 11 games, so the row table is bounded
  by `min(T_labels, T_latent)`.
* **Independent-derivation check:** the event index, rebuilt from `clicks.json` alone with
  `round((game_t − gt0)·fps)`, reproduces **exactly n = 14,564** movement-event frames on
  the 6 val games — the same number the production loader reports. The champion positions
  it implies match `clicks.json`'s own `hero_x/hero_z` at p50 = 8.58 world units, matching
  the loader's documented 8.6.

Primary rowset filters to clicks farther than **250 world units** from the champion
(75% of events), where direction is behaviourally meaningful; the unfiltered result is in
§6 and is the same.

## 3. Estimator validation (done *before* reading the result)

| control | result |
|---|---|
| **C1** ridge, same latents → champion **world position** | R² = **0.910** (x) / **0.938** (z), RMSE 1246 units |
| **V1** ridge, same 8-way machinery → **known-good** spatial target (octant of hero position about map centre) | **97.0%** vs null 51.3% |
| **V2** ridge, **train labels shuffled within game**, evaluated against real val labels | **17.4%** vs null 15.9%, MI **0.007 bits** |

V1 proves the 8-class readout can express a direction from these latents at
near-ceiling. V2 proves the whole pipeline returns a null when there is nothing to find.
A null result would have been believable; there was none.

## 4. Result

Six held-out games, **n = 10,982** click events, dist > 250 units.

| probe (input → target) | acc | measured null | z | MI (debiased) |
|---|---|---|---|---|
| **ridge, latent[t] → click octant** | **0.3776** | 0.1539 ± 0.0033 | **68.4** | **0.306 bits** |
| CNN, latent[t] → click octant | **0.3819** | 0.1470 ± 0.0032 | **72.8** | **0.334 bits** |
| ridge, latent[**t−10**] → click octant (0.5 s *before* the click) | 0.3163 | 0.1581 | 50.9 | 0.187 |
| ridge, latent[t+20] → click octant (non-causal) | 0.3408 | 0.1565 | 54.3 | 0.232 |
| ridge, latent[t] → **realised** displacement octant over [t, t+20] | 0.3996 | 0.1561 | 65.9 | 0.352 |
| ridge, latent[t+20] → realised displacement octant (non-causal) | 0.4361 | 0.1533 | 76.7 | 0.447 |

Blind baselines, no pixels at all, same rows:

| baseline | acc | MI |
|---|---|---|
| train-marginal mode (constant) | 0.1842 | 0 |
| best constant **per game** (oracle) | 0.2027 | — |
| champion **map position** table, 32×32 bins | 0.2732 | 0.168 |
| **previous-click octant** table (= persistence; the table's argmax is always the diagonal) | **0.5634** | **0.832** |
| direction to the **held** previous target, re-aimed from the current position | 0.5479 | 0.817 |

Angular readout of the same ridge (regression to (cos, sin)):
**median error 43.0°** vs **87.9°** for the best constant direction; **51.6%** within 45°
vs 21.3%.

### Per game — all six independently significant

| game | n | acc | its own null | z | median angular err | best-constant err |
|---|---|---|---|---|---|---|
| NA1_5549995114 | 711 | 0.2813 | 0.1384 | 10.7 | 49.7° | 84.5° |
| NA1_5550417257 | 794 | 0.2809 | 0.1651 | 9.1 | 58.9° | 78.2° |
| NA1_5551063460 | 2852 | 0.4400 | 0.1662 | 41.7 | 38.7° | 91.9° |
| NA1_5551782551 | 2383 | 0.3542 | 0.1477 | 30.2 | 43.8° | 78.1° |
| NA1_5552261591 | 1744 | 0.3813 | 0.1685 | 25.8 | 44.2° | 90.3° |
| NA1_5552945604 | 2498 | 0.3827 | 0.1370 | 35.7 | 40.4° | 75.8° |

Treating the game as the unit (5 df): lift over own null **0.1996 ± 0.0602**,
t(5) = 8.12, one-sided **p = 2.3 × 10⁻⁴**. Per-game spread is real (0.28 → 0.44) but the
sign is not in doubt.

### The signal is not just "where am I on the map"

Champion position is trivially decodable (R² ≈ 0.92) and is itself a decent direction
prior, so it has to be subtracted:

| octant NLL, val (nats) | |
|---|---|
| train marginal | 2.0501 |
| map-position table (32×32) | 1.8653 |
| **frozen latent** | **1.7271** |
| map position + latent | 1.6933 |
| previous-click table | 1.4795 |
| map position + previous click | 1.4217 |
| **previous click + latent** | **1.4136** |
| map position + previous click + latent | **1.3869** |

The latent beats a position lookup by **10.4 accuracy points** and **0.14 nats**, and
still adds **0.18 nats** *on top of* position. It is reading the scene, not the map square.

### It is not the click leaking into its own frame

Frame `t` is 0–50 ms after the click, so a rendered destination marker would be leakage.
Ruled out: the latent **0.5 s before** the click (`t−10`) still scores 0.3163 at z = 50.9,
above the position table and far above any null.

### It adds information the previous click does not contain

I(latent prediction ; true octant **| previous octant**), permutation-debiased within
(game × previous octant) strata: **0.075 bits**, null 0.030 ± 0.002, **z = 36.2**.

Matched-estimator stack (both logistic, fit on the 12 λ-fold games, read out on the 6):

| features | val octant NLL |
|---|---|
| previous octant only | 1.5007 |
| previous octant + latent scores | **1.4136** |

Gain **0.083 ± 0.016 nats**, positive in all six games, t(5) = 12.9,
**p = 2.5 × 10⁻⁵**.

## 5. What this does and does not license

**Does:** the frozen v7 latents demonstrably contain the human's next-click direction.
Option B — "not reachable from these latents, rebuild the tokenizer / Phase 1" — is
excluded. A *linear* readout of a *single* frame already recovers it; the CNN adds only
0.4 points, so this is not a capacity artefact either way, and a transformer over a
context window can only see more.

**Does not:** it does not promise a good movement policy once the crutch is removed.
The crutch is worth 0.83 bits; single-frame perception is worth 0.31 bits marginally and
**0.075 bits conditional on the crutch**. A pixel-only movement head should be expected to
land near **38% octant accuracy / 43° median error**, not near the 56% that "repeat the
last click" gets for free. The right reading of the numbers is: *the model is currently
extracting ~0 of a real 0.31 bits, and fixing that is a days-scale objective/data change —
but it buys a modest, not a transformative, movement head.*

**Untested caveat.** This probe reads the **raw** v7 latent. The deployed BC movement
head reads the frozen dynamics trunk's hidden state *derived from* those latents. If that
trunk destroys the direction signal, that is a different failure from the one tested
here — but it is still a Phase-2 failure (unfreeze or replace the trunk), not a
tokenizer/Phase-1 rebuild, so the A/B decision is unchanged. Confirming it is a
one-afternoon check and is the sibling investigation's territory.

## 6. Robustness

Dropping the distance filter (all 14,105 scorable val events, 97% of the 14,564; the
remainder lose `t−10`/`t+20` context or have zero radius):

| | acc | null | z | MI |
|---|---|---|---|---|
| ridge latent[t] → click octant | 0.3479 | 0.1513 | 68.5 | 0.243 |
| ridge latent[t−10] → click octant | 0.2835 | 0.1561 | 45.6 | 0.133 |
| previous-click table | 0.4970 | 0.1390 | 125.9 | 0.623 |

Same conclusion; short-radius clicks have noisier direction and dilute every estimate
uniformly.

## 7. Verdict

**A. High confidence.** The signal is present in the frozen v7 latents; the training
objective is not extracting it. Six of six held-out games agree, the estimator is
validated in both directions, the result survives the map-position control, the
click-leakage control and the previous-click control, and it is the same at every radius
cut tried. Confidence that B (rebuild the tokenizer) is the right call: **low** —
I would not spend weeks on Phase 1 on the strength of the CE-vs-table result, because
that test cannot tell B apart from a crutch that is merely bigger than the signal, and
here it demonstrably is.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
