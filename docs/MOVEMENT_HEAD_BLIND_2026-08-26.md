# The movement head is blind — it never learned to use vision

**Date:** 2026-08-26. **Checkpoint:** `data/phase2_bc_clicks/agent_finetune_latest.pt`
(step 102,420, axis+gate, `movement_source=clicks`, frozen backbone).
**Trigger:** Garen does not walk to top lane out of the fountain — a behaviour that
occurs in ~100% of replays and should be the easiest signal in the dataset.

## The finding

Event-frame categorical cross-entropy (nats, 2 axes summed), 1,800 teacher-forced
frames over 3 held-out games at MTP offset n=1:

| predictor | CE |
|---|---|
| uniform | 6.089 |
| corpus marginal | 5.216 |
| trained model, **movement action input removed** | 5.450 |
| **blind 21x21 lookup table `p(next bin \| prev bin)`, no pixels** | **4.357** |
| **trained model (146M world model + policy head)** | **4.365** |

**The 146M stack matches a count table to 0.008 nats (0.2%).** The table is fitted on
5 games, evaluated leave-one-game-out, and never sees a pixel. Remove the action
input from the model and it scores *worse than the corpus marginal*.

Corroborating, on the frames where a new command is actually issued:
- p(previous held bin) = **0.0611** vs p(true new bin) = **0.0335** — twice as much
  mass on the target it is already holding
- argmax orbits the input: mean |argmax - prev| = **1.460** bins vs
  |argmax - true| = **1.548** bins
- ablating the action input: argmax error 15.3deg -> **71.6deg**; sampled 45.0deg -> **89.5deg** (chance)
- *shuffling* the action history within the window LOWERS the loss (0.7208 vs
  0.7565) — a permutation can move the t+1 target into a causally visible slot and
  the head finds it. Direct proof it reads the target out of the action tokens.

**Mechanism.** `embed_actions` appends the action as a token in frame t's set
(`dynamics.py:776`); the agent token cross-attends to exactly that set
(`dynamics.py:192-245`). BC predicts a_{t+n} for n>=1, and on a hold frame
**a_{t+1} is bit-for-bit a_t** — 89.9% of frames. So the head learns a lookup on its
own input rather than a visual policy.

**Why it destroys deployment.** `agent_infer` builds the action history from the
agent's own emissions. A lookup keyed on the previous target is worthless when the
previous target is your own output: the head re-emits a ~1.5-bin perturbation of its
own standing order, so a self-fed rollout locks onto whatever direction it drifted
into (measured directional concentration R = 0.643 / 0.739 — one direction per game,
unrelated to the human).

## Why the training loss looked healthy

`bc_movement 0.737 vs uniform 6.089` is not a valid comparison — 6.089 is the
**no-gate** reference against a **gated** loss.

| predictor (under the gated objective as coded) | bc_movement |
|---|---|
| uniform, no gate (the number that was quoted) | 6.089 |
| **best blind stateless predictor** (copy-prev + marginal + constant gate) | **0.7716** |
| **trained model** | **0.737** |
| **oracle gate + uniform categorical** (knows WHEN, nothing about WHERE) | **0.6196** |
| oracle gate + perfect categorical | 0 |

The model is **0.035 nats better than blind** and **0.117 worse than knowing only
when to click.** The apparent 8x improvement was the gate discovering that 90% of
frames need no prediction. Hold frames contribute at most 14.1% of the loss
(algebraic cap: `l_hold <= -log(1-g)` = 0.116 at the trained g=0.1095).

## Why `--action-dropout 0.15` did not prevent it

Dropout is applied **per frame**, but the held value is constant across a whole hold
run and temporal attention is causal over the action-token slot — so the answer is
hidden only if EVERY frame since the last click is dropped: `P = p^(j+1)`.

| p | shortcut still available |
|---|---|
| **0.15 (shipped)** | **98.13%** |
| 0.50 | 90.04% |
| 0.95 | 45.85% |

No per-frame rate fixes this. It requires **run-level (block) dropout**. Worse,
`no_action_embed` is a single learned vector, so a dropped frame is *identifiable* —
the model knows to look at a neighbour instead of being forced to guess.

Note also `evaluate()` sets `action_dropout=0`, so every reported val number was
measured with the shortcut fully available.

## The fix, with an acceptance test

1. **Block-level dropout of the movement action input** (drop whole hold runs), or
   remove the movement action from the BC path entirely while keeping it for the
   world-model loss.
2. **ACCEPTANCE TEST: event-frame categorical CE must fall clearly below 4.357 nats.**
   At or above that, the head has learned nothing from pixels. No checkpoint we have
   ever produced clears this bar.
3. **Log the blind baselines next to `bc_movement`** (0.7716 blind, 0.6196
   oracle-gate). Against 6.089 the number is misleading and hid this for months.
4. Exclude the pre-first-click sentinel frames (the `(0.5,0.5)` "your own feet"
   window, 4.2% of corpus) from the sequence index — see the label bug below.
5. Delete `train_agent_finetune.py:698` (dead) and pass `movement_event` explicitly
   to `gated_movement_log_prob` instead of encoding it as a fake +1 bin difference.

## Related label bug (separate, also confirmed)

`_parse_movement_clicks` defaults the target to `(0.5,0.5)` until the first click.
Clicks do not start until ~60s in ANY game (the memory extractor takes that long to
lock on), and the champion is camera-locked to screen centre — so `(0.5,0.5)` means
"your move order is your own feet". **159,330 frames = 4.2% of corpus, ~61s of each
of 130 games**, and they are the only fountain/base frames in the dataset.
`heading_screen` (already in `labels.json`, currently unread) recovers the true walk
direction to **11.5deg**.

## Retracted

An earlier claim in this investigation that the policy is "worse than chance"
(131.9deg median vs 90deg) does **not** hold. That measurement was taken entirely
inside the pre-first-click window, where there are zero events and the target is the
sentinel, while its control came from post-click frames of a different game —
different populations. The metric is also bimodal (median jumps between ~22deg and
~111deg with horizontal-sign agreement) and n_effective ~= 2, since each game
contributes essentially one direction. The defensible statement is **at chance**.

## Metric hygiene (earned)

- Never score movement before the first `movement_event`.
- `movement` is HELD between clicks (median run 5 frames, p99 67, max 728), so any
  lag comparison inside a hold window has identical ground truth and is degenerate:
  at |k|=1, **98.8%** of holds are longer.
- Use frac<90 / frac<45 with a per-game block bootstrap, not a median angle on n<500.
- Validate every metric on the human's own labels before interpreting a model score.

---

# CORRECTION (same day, 13,627 event rows vs the original 1,800 frames)

An independent probe with ~8x the data corrects two claims above and reverses the
document's practical conclusion.

## 1. "Matches a no-pixel table to 0.008 nats" does NOT replicate at scale

On 13,627 event rows over 6 held-out games, the head is **0.099 nats BETTER** than a
properly-fitted blind table (fit on 24 TRAINING games), paired t = -2.31 (5 df),
negative in 5/6 games:

| predictor | CE |
|---|---|
| blind table fit on 5 held-out games | 4.183 |
| blind table fit on 24 TRAINING games | 4.123 |
| **deployed head** | **4.050** |

The original tie came from a 1,800-frame subset scored against a 5-game table. The
head is not literally a lookup table -- it is marginally above one.

## 2. The 4.357 acceptance bar is TOO WEAK — the broken checkpoint already passes it

A 24-game count table scores 4.123 and the SHIPPED checkpoint scores 4.050. The bar as
written is cleared by the model it was meant to reject. Any future bar must be stated
against a table fitted on the TRAINING corpus, and on the same val split it will be
read on (the 6-game split gives blind 4.1214 vs deployed 4.1212 on 14,564 events).

## 3. THE REAL FINDING: the signal is ABSENT from the features

Nested probes, each initialised AT the blind table so the crutch is free and only real
gain registers (same rows, same folds):

| probe | delta vs table | note |
|---|---|---|
| + raw v7 latents, linear | -0.002 | |
| + game state (hp/level/enemy) | -0.005 | |
| + raw v7 latents, MLP | -0.016 | |
| + oracle champion map position | -0.020 | |
| + raw v7 latents, CNN | -0.024 | |
| **+ agent token, movement ablated (leak-free)** | **-0.052** | not robust: -0.019 (t=-1.50) at context >=9 |
| **+ ORACLE future walk (cheats)** | **-1.058** | proves the estimator finds a nat when present |

Everything perceptual recovers 2-5% of what the cheating oracle recovers.

**Leak control verified in BOTH directions:** the held target decodes from the normal
token at 2.365 nats and at 5.578 (worse than marginal) once ablated -- the ablation is
complete. Champion map position still decodes at 2.687 from the ablated token -- it is
not garbage. **The latents encode the state; they do not determine the action.**

## 4. Switching the target to `heading` will not fix perception

At event frames the champion's own actual next-0.5s walk octant scores 1.753/1.859
against a past-heading table of 1.909 -- the same ~0.04-nat leak-free gain. Heading and
click carry the same information at the frames that matter, and neither is in the
features. (The pre-first-click label fix is still correct for COVERAGE -- those frames
had no supervision at all -- but it will not make the model see.)

## Consequence

**The bottleneck is perception/representation, not the objective.** Action dropout,
`event_only`, block masking and the structural mask all operate on a head that has
nothing to read. The planned T1 training program was cancelled before it consumed GPU.

Open question, and the cheap-vs-expensive fork: the probe read the AGENT TOKEN and the
RAW LATENTS, not the dynamics' INTERNAL SPATIAL TOKENS. If the signal is there but the
agent token fails to surface it, the fix is architectural (agent-block depth,
cross-attention) and cheap. If absent there too, it is tokenizer/Phase-1 work.
