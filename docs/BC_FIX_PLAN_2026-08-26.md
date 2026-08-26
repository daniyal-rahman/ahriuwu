# BC movement fix — options, tests, costs

**Problem** (see `MOVEMENT_HEAD_BLIND_2026-08-26.md`): the movement head learned zero
visual information. Event-frame categorical CE 4.365 nats vs a **no-pixel lookup
table's 4.357**. It reads the previous target out of the action tokens and perturbs
it ~1.5 bins. Self-fed at inference -> walks one arbitrary direction forever.

**THE BAR: event-frame categorical CE < 4.357 nats.** Everything below is judged on it.

---

## 0. Instrument first (PREREQUISITE, ~30 min, no GPU)

`bc_movement` is the GATED loss and is dominated by the gate, not the categorical.
It fell 0.871 -> 0.727 while the head learned nothing. **Watching it cannot tell us
whether a fix worked.** Before any run:

- log **event-frame categorical CE** (n=1, argmax + NLL at the true cell)
- log the **blind-table baseline** (4.357) and **oracle-gate** (0.6196) next to it
- log **CE with the movement action ablated** — if that number is close to the normal
  one, the head is NOT using the crutch; if it collapses, it is

Without this, a 25h run ends and we still don't know. This is the single highest-value
change in the document.

---

## The options

### A. Hide the movement action from the agent tokens (architectural)
The agent token cross-attends to frame t's token set, which contains the action token
(`dynamics.py:776`, `192-245`). Mask the movement action out of **the agent
cross-attention only**; the world model still sees it, so the video loss (7) is
untouched.

- closest to the paper: their Eq 9 sums from n=0, which is vacuous unless h_t excludes a_t
- removes BOTH the a_t leak AND the "copy a_{t-1}" route
- **risk:** also removes legitimate context. Defensible -- the policy's job is pi(a|s)
  and the screen already shows the consequences of past actions
- **strongest ablation: if this cannot clear 4.357, nothing softer will, and the
  problem is perception not objective.** Test this FIRST.

### B. Run-level (block) action dropout
Current `--action-dropout 0.15` is per-frame; the held value spans a whole run and
attention over the action slot is causal, so the answer hides only if EVERY frame
since the click is dropped: p^(j+1) = **1.87%**. Shortcut available on 98.1% of frames.

Drop **whole hold runs** instead. At run-level p, the shortcut is gone on exactly p of
runs. Keeps action conditioning for the frames that do have it.

- much smaller change than A
- but leaves the crutch on (1-p) of runs; unclear whether partial removal is enough
- also fix: `evaluate()` forces action_dropout=0, so every val number to date was
  measured with the shortcut fully open

### C. Reproject the held target every frame (changes what is copyable)
Today the held value is a FROZEN screen coordinate; the camera keeps moving, so it is
also wrong as the dynamics' action input (audit 1.4: off by >=1 of 21 cells on 46.6%
of hold frames). Re-projecting the held WORLD point through the CURRENT camera each
frame makes a_{t+1} != a_t even during a hold.

- fixes a known-wrong input independently of the shortcut
- copying the previous value is no longer exactly right, so the crutch degrades
- but the value is still highly autocorrelated -- this WEAKENS the shortcut, does not
  remove it. Not sufficient alone.
- already agreed as correct on its own merits

### D. Movement loss on event frames only
The gate already routes ~90% of categorical gradient to event frames (measured
w_hold = 0.0132, hold share 10.1%), so this changes little. **The copy is learned
through the INPUT, not through the loss weighting.** Listed for completeness; low value.

### E. Unfreeze the backbone
Audit 1.1: the frozen `action_embed` was fitted to the legacy `cursor` target and
cannot adapt. Paper parity also wants (7)+(9) on the world model. Does NOT address the
leak -- an unfrozen model can learn the same shortcut. Orthogonal; already run once
(the parity run, val move 0.727).

### F. Fix the pre-first-click labels
159,330 frames (4.2%, ~61s x 130 games) labelled "move to your own feet"; the only
fountain/base frames in the corpus. `heading_screen` recovers the true walk direction
to 11.5deg. **Necessary for the walk-to-lane symptom specifically, but useless alone**
-- the head cannot act on targets it already has.

---

## MEASURED: representation change ALONE does not remove the crutch

The shortcut only works if the answer is REACHABLE inside the 16-frame context.
Measured over 5,513 event-to-event gaps (3 held-out games, median gap 5 frames):

| context T | prev click in-window |
|---|---|
| 8 | 74.8% |
| **16 (ours)** | **93.3%** |
| 32 | 97.3% |
| 64 | 98.5% |

- **2a (held target)**: answer sits in the CURRENT token -> 100% reachable
- **2e (NO_OP input)**: answer sits at the last click -> **93.3% reachable**

So switching to NO_OP moves the answer from this token to one a few frames back and
the model simply attends to it. **2e alone is nearly useless.**

### But 2e is what makes DROPOUT work

Count how many tokens carry the answer:

- **2a**: replicated across EVERY frame of the run -> must drop all of them,
  `p^(j+1)` = **1.87%** at p=0.15. Dropout is structurally defeated.
- **2e**: exists in EXACTLY ONE token (the last click) -> dropping it once at rate
  `p` hides it with probability **p**.

**2e makes per-frame dropout ~30,000x more effective at the same rate.** The
representation change and the dropout are only worth anything TOGETHER.

### Revised ranking

| option | crutch reachable | verdict |
|---|---|---|
| 2a + per-frame dropout 0.15 (current) | 98.1% | hopeless |
| 2a + run-level dropout (1b) | 1-p | works, needs block machinery |
| 2e alone | 93.3% | nearly useless |
| **2e + per-frame dropout p=0.5** | ~50% | **cheap and principled** |
| **1d structural mask** | 0% | **guaranteed, upper bound** |
| shorter context T=8 | 74.8% | costs perception, not worth it |

---

## Test plan (cheapest discriminator first)

| # | test | cost | decides |
|---|---|---|---|
| 0 | instrument the trainer | 30 min, no GPU | makes everything else readable |
| 1 | **frozen-feature probe**: can ANY probe read the next click from the agent tokens / raw latents? | ~1 h, no training | signal PRESENT (fix the objective) vs ABSENT (perception problem -- no BC change helps) |
| 2 | **short run, option A** (hardest ablation), 5k steps | 2.2 h | if CE stays >= 4.357, nothing softer works |
| 3 | short runs, B and C, 5k steps each | 2.2 h each | whether a softer fix suffices |
| 4 | full train of the winner | ~25 h/epoch | the actual checkpoint |

**Test 1 before test 2.** If a supervised probe on frozen features cannot beat the
blind table either, then the tokenizer latents do not carry the information and every
option above is wasted effort. That probe needs no training run and answers the
expensive question first.

**Sequencing note:** desktop is one GPU. Tests 2-3 are ~7h total and can run back to
back overnight. Test 4 only after one variant clears the bar at 5k steps.

## Costs (measured, RTX 5080)

- 1 BC epoch = **55,216 steps = 24.6 h** (unfrozen, bs2 x accum8, 0.62 steps/s)
- 5,000 steps = **2.2 h**;  10,000 steps = **4.5 h**
- val `move` history: 0.871 -> 0.811 -> 0.798 -> 0.777 -> 0.758 -> 0.750 early, then
  crawls to 0.727 over the remaining ~50k. **Most learning is in the first few
  thousand steps**, which is why 5k-step runs discriminate.
