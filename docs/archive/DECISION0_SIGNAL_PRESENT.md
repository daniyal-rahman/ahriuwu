# DECISION 0 — the signal IS present and reachable. Verdict: **A**.

**Date:** 2026-09-02. **Checkpoint:** `data/phase2_bc_clicks/agent_finetune_latest.pt`
(step 102,420, the deployed one). **No training run; frozen features only.**
Code: `scratchpad/decision0/` (extract.py, fit.py, ceiling.py, head_walkout.py).

## Verdict

**A. The signal is there and the objective never forced the model to use it.**
The work stays on the BC side (~days). It is NOT tokenizer/Phase-1 work.

With the movement action ablated (`cursor_valid=False`), the frozen features predict
the human's next movement command **2–5x better than a probe on GROUND-TRUTH perfect
state** — true map position, every visible hero's position and hp, own hp/level, side,
game time — on every task, in both windows, on games no readout ever saw. And the
SHIPPED head, with no retraining, already emits the right walk-out direction once the
action channel is cut.

The previous verdict ("the signal is ABSENT from the features", CORRECTION section of
`MOVEMENT_HEAD_BLIND_2026-08-26.md`) was a **probe-power artifact**: it fitted its
readout on 5 games. Refitting the identical feature on 30 games more than doubles the
effect (−0.052 → −0.126 nats), and the ablated feature then beats the deployed head.

---

## Why the earlier probes could not answer this

Three defects, each fixed here.

1. **The readout was fitted on FIVE games.** A BC retrain sees 119. "A 5-game probe
   found nothing" and "the information is not there" are different statements. Fixed:
   the readout is fitted on 106 games (walk-out) / 30 games (in-lane) and read on the
   6 BC held-out games, plus grouped k-fold so every game is scored by a readout that
   never saw it.
2. **Every probe scored POST-first-click rows only.** The walk-out — the behaviour
   that is actually broken — was never probed on frozen features at all.
3. **The ceiling was a CHEATING oracle** (the champion's own future path). That bounds
   TOTAL predictability, the human's private intent included, so "the probe recovers
   5% of the oracle" cannot separate "the features are blind" from "the target is not
   a function of what is on screen". Fixed: `ORACLE_state`, a ground-truth
   **perception** ceiling, on identical rows.

---

## Method

- **Rows.** `walkout` = every frame before a game's first click (112 games with a click
  stream; first click p50 = frame 1223 = 61.2 s @ 20 fps). `lane` = post-first-click;
  the click tasks score **only** frames whose successor is a `movement_event`
  (13,627 rows over the 6 held-out games), never a hold frame, never a pre-first-click
  sentinel frame.
- **Ablation.** Every forward is PAIRED — `act` (cursor_valid=True) and `noact`
  (cursor_valid=False, `embed_actions` swaps in the learned `no_action_embed`) in one
  batch, same latents, same τ/noise draw. Verified to reproduce `scratchpad/probe2`'s
  stored agent tokens to 0.008 (bf16 noise).
- **Features.** Agent token (768), block-17 and block-11 pooled spatial tokens, 2×2
  quadrant pooling — LATENT tokens only, the action slot never pooled in — and the raw
  v7 latents.
- **Estimator.** Nested: the readout's output layer is zero-initialised and ADDED to a
  blind baseline's log-probs, so the baseline is candidate #0 and only genuine
  cross-GAME generalising gain can select the probe away from it. Early stopping and
  weight-decay selection hold out whole GAMES from the fit set (a row-level inner split
  shares game identity and cannot see cross-game overfitting). Splits are always by
  whole game.
- **Blind baseline.** The transition table `p(next bin | prev bin)`, fitted on the
  TRAIN games of the split in use — `blind_table_bar()`'s construction.

---

## 1. WALK-OUT — the signal is there, and richer than ground truth

The walk-out is **structurally leak-free**: `_parse_movement_clicks` sets the movement
input to the constant `(0.5, 0.5)` sentinel for every frame of every game before the
first click (verified: exactly one unique value over all rows). There is no held target
to copy. Target = the human's own walk direction, world displacement over the next 1 s,
8-way. Blue mean +84.7° (R = 0.987 across games) / red −172.9° (R = 0.994), separation
102.3° over 112 games — replicating the 2026-08-27 HUMAN row (+87.6 / −171.7, 100.8°)
on 2.8× the games.

**Moving frames.** `heldout6` = readout fitted on 106 games, read on the 6 BC held-out
games (2,782 rows). `cv5` = every one of the 112 games scored by a readout that never
saw it.

| predictor | CE heldout6 | CE cv5 | Δ vs table (cv5) | t (112 g) |
|---|---|---|---|---|
| uniform | 2.0794 | | | |
| marginal | 1.7136 | 1.7476 | +0.604 | +29.0 |
| **blind persistence table** | **1.2825** | **1.1438** | 0 | |
| ORACLE side + game time | 1.2859 | 1.3269 | −0.065 | −15.1 |
| ORACLE true map position | 1.1848 | 1.2376 | −0.109 | −20.0 |
| **ORACLE full ground-truth state** | **1.2195** | **1.2062** | **−0.113** | **−18.0** |
| agent token, movement ABLATED | 1.0193 | 0.9586 | −0.151 | −9.5 |
| block-17 tokens, ABLATED | 0.9395 | 0.8717 | −0.216 | −14.3 |
| **block-11 tokens, ABLATED** | **0.9406** | **0.8610** | **−0.233** | **−16.5** |

Held-out-6 deltas vs the table: block-11 −0.2208 (t = −4.78, negative in 6/6 games),
agent token −0.1594 (t = −5.06), ground-truth state −0.1223. Restricting to window
positions ≥ 9 frames of context changes nothing (−0.207 / −0.128 / −0.103).

**Against the marginal** — the deployment-relevant framing, since a self-fed rollout's
own action history is worthless — the frozen ablated features carry **0.887 nats**
about the human's walk direction, versus 0.604 for the persistence table and 0.541 for
perfect ground-truth perception. Median angular error of the probe's mean direction:
**7.0°**, versus 47.8° for the baseline.

**From a standstill** (past 1 s displacement < 40 units — in the fountain, where there
is no momentum to extrapolate and persistence is worthless), marginal baseline:

| | Δ (cv5, 112 g) | t |
|---|---|---|
| blind persistence table | −0.018 | −2.05 |
| agent token, ABLATED | −0.166 | −3.01 |
| ORACLE full state | −0.222 | −5.38 |
| ORACLE true position | −0.236 | −6.93 |
| **block-17 tokens, ABLATED** | **−0.267** | **−5.78** |

**2–3 s ahead** (decorrelated from the champion's current velocity, which the features
are already known to carry): block-17 ablated −0.233 (t = −4.13, heldout6), ground-truth
state −0.198. Same ordering.

## 2. The SHIPPED head already does it — cut the action and nothing else

Open-loop, teacher-forced on real replay latents, no rollout, no retraining, all 112
games. Direction of the head's expected commanded point relative to the champion, in
screen space, against `label.movement.heading_screen` (the pipeline's own walk-direction
readout, good to 11.5°).

| | per-frame median err | frac < 90° | per-game mean-direction err | blue/red separation |
|---|---|---|---|---|
| HUMAN | 0 | 1.000 | 0 | 101.6° |
| **movement action CUT (`noact`)** | **29.9°** | **0.844** | **10.9°** (chance 48.6°, p < 2.5e−4) | 79.5° |
| as deployed (`act`) | 72.7° | 0.641 | 38.2° (chance 100.6°) | 105.6°, pointing wrong |
| CENTER control | 124.4° | 0.339 | 104.7° (chance 116.9°, **p = 0.55**) | 15.5° |

- **CENTER control**: the champion is camera-locked but not pinned — `champion_screen`
  drifts opposite the walk while the camera catches up, so "direction from the champion
  to a FIXED screen point" would already carry the walk direction. It does not: it is
  at chance and mildly anti-correlated. The effect is the head's, not the camera's.
- **Within-side tracking** (does it follow the individual game, or only blue-vs-red?):
  `noact` r = **+0.386** blue (p = 0.005, n = 53) and **+0.287** red (p = 0.016, n = 59).
  `act` r = +0.06 / −0.13, both n.s. Cutting the action is what makes it track the game.
- **In the fountain (frames 0–200):** `noact` 37.4° vs chance 100.6° (p < 2.5e−4);
  `act` 111.1° vs chance 114.4° (**p = 0.06 — at chance**).
- **Stationary frames only:** `noact` 24.7° vs chance 62.0° (p < 0.001), frac < 90° =
  0.949, blue/red separation 98.6° against the human's 101.6°.

## 3. IN-LANE — the 4.1212 bar, refitted at scale, also moves

Next-click 21×21 cell CE at event frames, 13,627 rows over the 6 held-out games,
readout fitted on 30 train games (every previous probe used 5).

| predictor | CE | Δ vs table | t (6 g) |
|---|---|---|---|
| uniform | 6.0890 | | |
| marginal | 5.2574 | +0.995 | |
| blind table (fit on the 30 train games) | 4.1333 | 0 | |
| ORACLE true map position | 4.0891 | −0.046 | −16.3 |
| **ORACLE full ground-truth state** | **4.0916** | **−0.043** | **−17.6** |
| DEPLOYED head, as shipped | 4.0503 | −0.095 | −3.76 |
| block-17 tokens, ABLATED | 4.0317 | −0.109 | −12.8 |
| **agent token, ABLATED (leak-free-by-baseline)** | **4.0081** | **−0.126** | **−23.8** |
| ORACLE future path (CHEATS) | 3.1022 | −1.079 | −20.8 |

Two things to read off this table.

- **The ablated feature beats the deployed head** (−0.126 vs −0.095). A probe with no
  copy channel extracts more from vision than the 146M stack extracts with the crutch
  available. The prior probe on the *same* feature with a 5-game fit got −0.052.
- **The cell-CE acceptance test is a bad bar.** A ground-truth perfect-perception oracle
  beats the table by only **0.043 nats** on it. The metric's entire visually available
  headroom is ~0.13 nats; the remaining 0.95 of the cheating oracle's 1.08 is the
  human's private intent. A bar written in this metric cannot distinguish a blind model
  from a sighted one. **Replace it with a direction bar.**

Same rows, direction instead of cell (8-way, next click relative to the champion):
table 1.6481 → ablated block-17 **1.5586** (−0.087, t = −13.4), ground-truth state
1.6296 (−0.018), cheating oracle 0.6612.

Champion's own in-lane walk direction (8-way, 15,504 rows): table 1.8324 → ablated
block-11 **1.5977** (−0.233, t = −27.2, median error 30.6°), ground-truth state 1.7862
(−0.049, median 40.1°).

The shipped head in lane, action cut, open loop (36 games): median per-frame direction
error **55.6°**, frac < 90° = 0.659, per-game mean direction 24.6° from the human
(chance 63.5°). With the action left in it scores 22.3° — but that is the crutch, not
skill: the held click target is where the champion is walking.

## 4. Controls

Every control the brief demanded, plus two the earlier work did not run.

| control | result |
|---|---|
| PLANTED signal | 0.0001 (walk-out) / 0.0002 (cell) — the estimator finds signal when it is there |
| NOISE features | exactly the baseline, every arm |
| **Labels shuffled GLOBALLY** | **every arm exactly at the marginal**, both windows |
| Labels shuffled WITHIN game | arms retain −0.13 (walk-out): the game/region direction prior survives that permutation, so this is a decomposition, not a null — see caveat 4 |
| Feature power (champion map position, ablated features) | 0.23 nats walk-out / 0.66 in-lane, vs marginal 4.63 / 4.98 |
| Leak, un-ablated direction | held click target decodes from `agent_act` at **1.09** nats (marginal 5.30) — the leak is real and large |
| Leak, ablated direction | held target still decodes at **4.22** — the ablation is NOT complete; see caveat 1 |
| Context position (window pos ≥ 9) | every headline delta unchanged |
| Per-game | headline arms negative in **6/6** held-out games in both windows; in CV, 105/112 (walk-out block-11) and 35/36 (in-lane agent token) |
| Deployed-head anchor | 4.0503 on these rows, matching the 4.050 already on record |

---

## What this licenses, and what it does not

**Do:** the cheap fork. Retrain BC with the movement action removed from the policy
path (or run-level block dropout — per-frame p = 0.15 leaves the shortcut available on
98.13% of frames) **AND** `prefirst_mode='heading'`. Both, as the earlier doc said: the
channel fix alone leaves the walk-out unsupervised, and the label fix alone leaves the
copy channel dominant. Free today, no retraining: `--movement-action-mode none`.

**Change the acceptance test.** Event-frame cell CE cannot see any of this (perfect
perception moves it 0.043 nats). Bar the direction instead: per-game commanded direction
vs the champion's own heading, reported separately for blue and red, with the CENTER
control alongside.

**Do not** expect CS. The reachable signal is route-level. Even the best probe leaves
~30° median direction error in lane, and the shipped head with the action cut leaves
55.6°. Walking out, arriving in the right lane and holding a region should follow. Fine
target selection — last-hitting, standing on the wave — is not in this result and should
not be promised by it.

## What I could not rule out

1. **The ablation is not complete.** The held click target still decodes from the
   ablated features at 4.22 nats (marginal 5.30) — better even than from the champion's
   own actual future path (4.50). Some of that is legitimate (the click destination is
   on screen and the champion is walking to it), but a residual movement-token signal
   smeared through 18 layers of spatial attention cannot be excluded. Two reasons it
   does not explain the results: (a) the nested baseline is the table evaluated at the
   **true** previous bin, so recovering the previous bin adds nothing — only information
   *beyond* it can score; (b) the entire walk-out result is immune, because the movement
   input there is a constant sentinel. A sub-bin-resolution residual channel could still
   contribute a little to the in-lane cell number. Note the earlier "ablation is
   complete in both directions" claim was itself a 5-game-probe artifact (it measured
   4.90/5.58 where 30 games measure 4.22).
2. **"Raw v7 latents carry nothing" is NOT established.** The raw arm scores −0.019
   (t = −0.27) — nothing — but it reads a SINGLE frame while the dynamics features carry
   16. The comparison is confounded and I did not run a multi-lag raw arm. What *is*
   established: the dynamics is a function of the v7 latent sequence and the (ablated)
   actions and nothing else, so the information is provably present in the v7 latent
   sequence. Whether a single frame suffices is open.
3. **`noact` is out of distribution** for this checkpoint — BC saw `cursor_valid=False`
   at p = 0.15 per frame, never for a whole window. Every open-loop head number in §2 is
   measured in that OOD regime. It is evidence the competence exists, not that this
   checkpoint is deployable as-is.
4. **How much is route prior vs frame-by-frame navigation.** Shuffling labels within a
   game leaves the arms −0.13 nats ahead of the table, because a game's direction
   histogram survives that permutation. Of the 0.774 nats the walk-out probe extracts
   over the marginal (held-out 6), roughly 0.32 is that region-level prior and ~0.45 is
   frame-specific. Consistent with the surviving caveat in the earlier doc (heading
   tracking r = +0.104): it knows the route, it does not turn where the lane turns.
5. **Six held-out games is 5 df.** The powered numbers come from k-fold over 112 / 36
   games; those games were seen by BC and the tokenizer, though never by the readout.
   The two agree throughout, which is the reason to believe them.
6. The walk-out target uses displacement up to 1 s (3 s for the far variant) past the
   row's frame, so ~5% of rows near the end of the window have targets influenced by
   the game's first real command.

## Reproduce

```
scratchpad/decision0/extract.py --rows walkout|lane     # frozen features, paired act/noact
scratchpad/decision0/run_walkout.sh                     # walk-out probes + controls
scratchpad/decision0/run_lane.sh                        # in-lane probes + controls
scratchpad/decision0/head_walkout.py f_walkout|f_lane   # the shipped head, open loop
scratchpad/decision0/summarize.py scratchpad/decision0/ce_*.json
```
