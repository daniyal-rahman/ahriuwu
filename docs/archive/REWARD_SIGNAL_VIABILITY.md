# Is the solo-gold reward worth doing RL against?

**Date:** 2026-08-27. **Scope:** the reward *signal*, not its twohot representation
(the bucketing defect in `docs/archive/WIRING_AUDIT_2026-08-20.md` §1.6 is a separate item and
is assumed fixed throughout). **Question:** if Phase 3 ran perfectly tomorrow against
this reward, would it learn to play Garen better?

**Corpus:** all 146 labeled games in `/srv/nfs/datasets/lol_replays_16_9_772`,
**4,176,465 label frames = 58.0 game-hours** at 20 fps. Every number below is measured
on that corpus unless marked otherwise. Extraction + analysis scripts are in the
session scratchpad (`rw/extract.py`, `rw/analyze*.py`, `rw/probe.py`); they only read
`labels.json` and the packed v7 latents.

**Status key:** CONFIRMED (measured here) · LIKELY (measured but confounded) ·
SPECULATIVE (reasoned).

---

## Verdict

**Qualified yes — but not as configured, and not at H=8.**

The reward is *not* the degenerate time-accumulator it was suspected of being. 71.4% of
its magnitude is discrete gameplay events, its credit-assignment delay is ~0.2 s
(comfortably inside the imagination horizon), and at the episode level it ranks games by
lane outcome at r = +0.90. Those are the three things a reward has to get right, and it
gets all three. The reward is also **readable off the frozen latents** (§4), which was the
one finding that could have made this whole question moot.

What it gets wrong is that it is **not zero-sum**. It is blind to the opponent by
construction, and the measurement is unambiguous: at *every* timescale from 0.4 s to
300 s, `corr(own gold gained, lane gold-diff change) = 0.697–0.708`, which is exactly
the arithmetic null for two independent equal-variance streams. Half the variance in
"did I gain an advantage over this window" is invisible to the reward at every timescale
RL acts on. Adding the opponent term is a **one-line config change** (`use_solo_gold=False`)
whose stated blocker — that the opponent is often not visible — is **false**: the lane
opponent's `gold_total` is present in the labels on **99.8%** of frames.

The two changes that would have to happen before Phase 3 is worth GPU:

1. **Zero-sum it** (§7.1). One flag plus a `gold_diff_scale` retune (5e-5 → 1e-3, which is
   also what the published gold : death ratio says). Recovers the missing half of the
   signal, cancels passive income exactly, and gives the reward a **negative direction**
   for the first time: measured, 13.1% of 0.4 s windows go negative under gold-diff versus
   **0.1%** today.
2. **Lengthen the horizon** (§7.2). At H=8 (0.4 s) the *only* learnable behaviour is
   last-hit timing — 95.0% of imagined rollouts contain no reward event at all, so PMPO's
   sign-of-advantage split is a coin flip on 95% of samples. Trades need H≈40–100.

And one thing that turned out **not** to be a blocker:

3. ~~Confirm the reward head can read it~~ — **checked, and it passes** (§4). A probe
   straight off the frozen v7 latents reads "gold event within the next 0.5 s" at
   **held-out AUC 0.778** across 8 unseen games, with world-position controls at R² 0.91.
   The `DESIGN_DECISIONS.md` §1 claim that the reward head is blind (AUC 0.431) is
   **overturned** — independently, and by the parallel `REWARD_MODEL_INVESTIGATION.md`
   (0.904 through the trained head). What is *not* readable is the reward's scalar
   magnitude (R² < 0), which is a target-design problem, not a perception one.

---

## 0. What the reward actually is

`src/ahriuwu/rewards/reward.py`, `RewardConfig` defaults, which is what every trainer
uses — **no CLI flag anywhere overrides it** (`grep RewardConfig scripts/*.py` → only
scratchpad probes). CONFIRMED.

```
r_t = 1e-3 · Δ(Garen's own gold_total)        # ≥ 0, monotone accumulator
    + (-0.2 if hp crossed >0 → ≤0 this frame) # own death, one-shot
```

`use_solo_gold=False` (gold **diff** vs the lane opponent) and `use_outcome` (±win/loss)
exist but are off. `use_outcome` is also *unusable*: there is no outcomes manifest
anywhere on disk, so `garen_win` is a dummy `False` for every match
(`train_agent_finetune.py --manifest` is never passed). CONFIRMED.

The `visible_heroes` block that the enemy-mode terms need is **not** a visibility-gated
view — it is a full memory read of all ten heroes (`raw_mem.json` → `labels.json`). See §7.1.

---

## 1. Characterisation

### 1.1 Sparsity and magnitude — CONFIRMED

| quantity | value |
|---|---|
| frames with `r != 0` | **10.07%** |
| frames with `r < 0` (deaths) | 0.01% |
| mean `r` | 0.000314 |
| std `r` | 0.006070 |
| median / p90 / p99 `r` | 0 / 0.0010 / 0.0011 |
| p99.9 / p99.99 `r` | 0.053 / 0.322 |
| min / max `r` | −0.2000 / +1.0900 |
| frames with a gold lump ≥10 g | 0.63% |

90% of frames are exactly zero. The 10% that are non-zero are almost all the **passive
income tick**: 1.0–1.1 gold arriving ~2×/s, i.e. `r = 0.001`. Real gameplay events are
0.63% of frames.

### 1.2 The lump spectrum is interpretable — CONFIRMED

Per-frame Δgold ≥ 3 g, 27,091 events over 146 games:

| Δgold bucket | count | share of events | gold |
|---|---|---|---|
| 3–12 | 814 | 3.0% | 5k |
| **12–16** (caster minion, 14 g) | 9,084 | **33.5%** | 127k |
| **16–22** (melee minion, 21 g) | 8,786 | **32.4%** | 167k |
| 22–45 (2 minions in one frame) | 3,356 | 12.4% | 113k |
| 45–90 (3 minions / cannon) | 3,220 | 11.9% | 205k |
| 90–250 (plates, multi-kill waves) | 1,259 | 4.6% | 176k |
| **250–700** (champion kills, ~300 g) | 553 | **2.0%** | 208k |
| >700 | 19 | 0.07% | 35k |

The reward's content is: **two-thirds last-hits, then cannon/plate-sized lumps, then
champion kills.** That is a real description of laning, not an artefact.

### 1.3 Passive income is 28.6% of the *magnitude* and 0.1% of the *variance* — CONFIRMED

Splitting each game's positive Δgold at 3 g:

| | per game | share |
|---|---|---|
| drip (<3 g steps) | 2,748 g | **28.6%** |
| lumps (≥3 g) | 6,857 g | **71.4%** |
| total earned | 9,605 g | |

Measured drip rate **1.897 ± 0.074 g/s** — LoL's base passive income is 2.04 g/s after
110 s, so this is exactly the game clock and nothing else.

**The "reward is elapsed time" hypothesis is FALSE at the decision timescale, and the
naive test that appears to support it is invalid.** Cumulative reward regresses on frame
index at R² = 0.976 (mean over games) and `corr(episode reward, duration) = 0.890` — but
*any* non-negative accumulator does that, so those numbers carry no information. The
decision-relevant decomposition is of the **variance of the discounted return over the
horizon the policy actually optimises** (H=8, γ=0.997):

| component | var share of the 8-frame return |
|---|---|
| gameplay lumps | **88.4%** |
| own death | 12.0% |
| **passive drip** | **0.1%** |

(n = 4,175,297 windows, one per frame.)

Over 0.4 s the drip is a near-constant offset that any value function absorbs. It does
not destroy the advantage signal. It *is* a 28.6% dead-weight bias in the episode return,
which matters for the value head's range, not for the policy gradient.

### 1.4 Concentration and placement — CONFIRMED

- Top **0.1%** of frames carry **37.2%** of episode reward; top **1%** carry **76.0%**;
  top 5% carry 91.1%.
- Episode reward: mean **8.98**, std 4.28, range [1.20, 18.00] (n=146).
- Only **42.1%** of reward falls in the first 14 minutes. Minutes 20–30 carry **31.3%**.
  Gold rate climbs monotonically from 5.7 g/s at minute 2 to 9.6 g/s at minute 28. The
  project targets *laning*, but the reward's mass is mid-game.

### 1.5 The reward is temporally white — CONFIRMED

Autocorrelation of `r_t`, pooled over games:

| lag (frames) | 1 | 8 | 20 | 40 | 100 | 400 |
|---|---|---|---|---|---|---|
| ρ | −0.003 | +0.006 | **+0.015** | +0.004 | +0.003 | +0.001 |

*Control (the metric validated on known-good signals, same code path):* own hp fraction
ρ(1)=+1.000, ρ(20)=+0.989, ρ(400)=+0.475; level ρ(400)=+0.891; synthetic white noise
ρ(20)=+0.008. The ACF is measuring correctly; the reward really is a near-white point
process. Nothing about the *previous* reward predicts the next one — there is no temporal
smoothing for the reward head to exploit.

### 1.6 Event arrival rate — CONFIRMED

Inter-event gaps (Δgold ≥ 3 g, n=26,945): median **40 frames (2.0 s)**, mean 145 frames
(7.24 s), p90 411 frames (20.6 s). **8.3 events per game-minute.**

---

## 2. Does it discriminate good play from bad?

### 2.1 At the episode level: YES, strongly — CONFIRMED

n=87 games with a resolvable lane opponent. Raw `corr(episode reward, duration) = +0.93`,
so duration must be partialled out (a longer game trivially accumulates more of a
non-negative reward). After partialling out duration:

| episode reward R vs | partial r \| duration |
|---|---|
| **final gold diff vs lane opponent** | **+0.899**  (bootstrap 95% CI [+0.845, +0.936]) |
| kill − death differential | +0.808 |
| mean level diff vs opponent | +0.745 |
| opponent's final gold | **−0.623** |

And the extremes line up with what a League player would say:

- 5 worst games by gold/min (188–229 g/min): final gold diff **−764 to −1,592**
- 5 best games by gold/min (502–530 g/min): final gold diff **+3,548 to +6,748**

The mechanism is real, not arithmetic: `partial corr(own final gold, opponent final gold
| duration) = −0.584`. Within a fixed game length, the lane's minion pool is shared — when
Garen farms more, his opponent farms less. **Own gold is a genuinely good whole-game proxy
for lane advantage.**

### 2.2 At every timescale RL actually acts on: NO — CONFIRMED

The same comparison, windowed, pooled over games. `null = 0.707` is what you get
arithmetically if own and opponent gold are independent with equal variance:

| window | n windows | corr(own, diff) | corr(own, opp) | within-game demeaned |
|---|---|---|---|---|
| 0.4 s (H=8) | 273,862 | **0.697** | 0.008 | 0.697 |
| 1 s | 109,516 | 0.700 | 0.009 | 0.700 |
| 2 s | 54,736 | 0.705 | 0.002 | 0.705 |
| 5 s | 21,870 | 0.704 | 0.006 | 0.703 |
| 10 s | 10,910 | 0.697 | 0.013 | 0.695 |
| 30 s | 3,607 | 0.702 | 0.037 | 0.697 |
| 60 s | 1,788 | 0.708 | 0.044 | 0.701 |
| 180 s | 493 | 0.708 | 0.076 | 0.688 |
| 300 s | 276 | 0.706 | 0.115 | 0.689 |
| whole game (dur-partialled) | 87 | **0.899** | −0.584 | — |

**Flat at the null from 0.4 s to 300 s.** The solo reward carries no information about the
opponent at any sub-game timescale; it only becomes a good proxy once a whole game's
shared minion pool has been exhausted. Concretely: farming 400 g while the opponent farms
0 and farming 400 g while the opponent farms 800 produce **identical reward**. R² of the
reward against "did I gain lane advantage this window" is **0.49** at every RL timescale.

### 2.3 Deaths are underweighted, and non-absorbing — CONFIRMED

- 3.14 own deaths/game over 146 games (6.1% of frames spent dead); on the 87-game
  opponent-resolvable subset, 2.56 own vs **3.63 opponent** deaths per game.
- Total death term is **−0.63 per game = 6.5%** of the gold term, but **12.0%** of the
  0.4 s return variance. It is not negligible at the decision timescale.
- **Measured opportunity cost of a death** (gold earned in the window after a death vs.
  a duration-matched random alive window, n = 412–456 deaths depending on window):

| window after death | earned | baseline | opportunity cost |
|---|---|---|---|
| 10 s | 50.4 g | 73.0 g | 23 g |
| 30 s | 101.1 g | 207.6 g | **107 g** |
| 60 s | 225.6 g | 417.3 g | **192 g** |

  The configured −0.2 = 200 gold-equivalent, which is roughly the *own* opportunity cost.
  It does **not** price the ~300 g bounty handed to the enemy, because the reward is not
  zero-sum. True zero-sum cost of a death ≈ 480–500 g ⇒ the penalty is light by ~2.4×.
- Worse, `train_imagination.py:378` sets `continues = torch.ones_like(rewards)` — **death
  is not terminal in imagination**, so the dream keeps paying after it. And gold keeps
  accruing while dead: **3.15 g/s dead vs 6.99 g/s alive** (assists + passive), i.e. the
  reward literally pays `r = 0.00016`/frame for being dead.

### 2.4 Implied exchange rates — CONFIRMED

| event | r | in "own death = 1" units |
|---|---|---|
| passive tick (1 g) | +0.0010 | 0.005 |
| caster minion (14 g) | +0.0140 | 0.07 |
| melee minion (21 g) | +0.0210 | 0.105 |
| cannon (~75 g) | +0.0750 | 0.375 |
| turret plate (160 g) | +0.1600 | 0.80 |
| **champion kill (300 g)** | **+0.3000** | **1.50** |
| **own death** | **−0.2000** | **−1.00** |

The reward says **a 1-for-1 trade is worth +0.10** — kill 1.5× death. Under a zero-sum
reward the two are symmetric by construction (your kill *is* the enemy's death), which is
why none of the published MOBA agents (§5) has this asymmetry. As configured, ours is a
standing instruction to all-in on coin-flip trades.

### 2.5 How much reward arrives without acting — LIKELY

- 43.6% of lump *events* (40.7% of lump *gold*) have a logged `attack` action within the
  preceding 1 s, against a base rate of 13.4% — a **3.3× enrichment**, so the association
  is real (146 games, 27,091 lump events).
- The remaining 59% is unattributable with current labels. `label.action.type` never
  carries `cast` (distribution over all 4,176,465 frames: idle 88.73%, recall 4.38%,
  attack 4.33%, none 2.56%), so Garen's ability farming (E, Q) is invisible to this
  test. The residual is some mixture of ability kills, assists, plates and ally-driven
  gold. **Cannot separate them with these labels.**
- Gold rate while standing still ≥2 s (8.8% of alive frames): **3.21 g/s vs 6.99 g/s
  alive** — a do-nothing policy collects ~46% of the human's reward rate. Confounded
  (humans stand still at particular moments), so LIKELY, not CONFIRMED. The unconfounded
  floor is the 1.90 g/s drip = 27% of the rate; over a 0.4 s horizon the drip alone is 33%
  of the mean discounted return.

---

## 3. Credit assignment — the best news in this document

**CONFIRMED, and it overturns the prior worry.** Rate of the `attack` action label in a
±3 s window around minion-sized gold lumps (12–95 g); 9,187 events over the first 60
games, base rate 4.41%:

| lag | −2.0 s | −1.0 s | −0.5 s | **−0.30 s** | **−0.20 s** | **−0.10 s** | 0 | +0.5 s | +2.0 s |
|---|---|---|---|---|---|---|---|---|---|
| P(attack) | 0.104 | 0.060 | 0.052 | **0.265** | **0.334** | **0.285** | 0.017 | 0.032 | 0.079 |

A sharp, clean 7.6× spike at **−4 frames (−0.2 s)** that collapses to below base rate the
instant the gold lands. The causal action for the dominant reward component sits **inside
the H=8 (0.4 s) horizon**. Credit assignment for last-hitting is not the problem.

**But the horizon covers nothing else.** Discounted-return reachability, measured:

| H (frames) | seconds | P(≥1 lump in window) | P(≥1 death) | P(return is drip-only) |
|---|---|---|---|---|
| **8** | **0.4** | **0.050** | 0.0009 | **0.950** |
| 16 | 0.8 | 0.091 | 0.0018 | 0.909 |
| 20 | 1.0 | 0.109 | 0.0022 | 0.891 |
| 40 | 2.0 | 0.183 | 0.0044 | 0.817 |
| 60 | 3.0 | 0.243 | 0.0066 | 0.757 |
| 100 | 5.0 | 0.340 | 0.0110 | 0.660 |
| 200 | 10.0 | 0.511 | 0.0221 | 0.489 |
| 400 | 20.0 | 0.707 | 0.0443 | 0.293 |

**95.0% of imagined rollouts at H=8 contain no reward event at all**, and 22.7% have a
discounted return of exactly 0.0. In those rollouts `A = R^λ − v` is determined entirely by
value-head error and dreamed-state noise, and PMPO uses **only the sign of A**
(`returns.py:322`) — so the D+/D− split in 95% of samples is a coin flip. The reward is
not too *small* for PMPO (it is scale-invariant); it is too *rare*.

Also: γ=0.997 at 20 fps gives an 11.5 s half-life, λ=0.95 gives a 1.0 s mixing length, and
the rollout is 0.4 s. Only **3.8%** of a γ-consistent (333-frame) discounted return lies
inside H=8; the other 96.2% is whatever the value head guesses. This is already recorded
as MIS-TUNED in `DESIGN_DECISIONS.md` §6; the measurements above quantify it.

---

## 4. Can the reward head even read it? — YES

In imagination, reward exists **only** as `reward_head.predict(h_t)` on a dreamed agent
token (`train_imagination.py:318`). Gold is not rendered — the corpus is HUD-disabled, so
there is no gold counter on screen. Unit health bars *are* rendered (verified by eye on
`NA1_5549995114/frames/004999.png`), so "a minion I was hitting just died" is in principle
visible at 352×352 — but it is a few pixels of a bar over a ~10 px minion.

`DESIGN_DECISIONS.md` §1 records a falsifier having fired on the trained head — **AUC
0.431 / 0.29 for "will this swing last-hit", worse than chance**. That has since been
**overturned**: the parallel investigation in `REWARD_MODEL_INVESTIGATION.md` §5 measures
the step-99,421 head at **held-out income-event AUC 0.9041**, with the right controls
(true reward 1.000, noise 0.506, game-time confound 0.5006, passive-tick confound 0.485)
and a 10× separation between predicted reward on event vs. passive frames. Its caveat is
real and should be carried: **n = 34 events on a single held-out game.**

**An independent, larger check run here.** A probe straight off the *frozen v7 latents*
(`replay_latents_v7_bc`, 32×16×16 = 8,192 dims/frame), bypassing the trained head
entirely, so it measures whether the signal is in the observation stream at all.
**16 train games / 8 held-out games**, 43,046 train and 28,254 held-out frames sampled
balanced on the target (≈14.1k positives held out). Linear and one-hidden-layer (512)
probes, AdamW, 60 epochs. Numbers are (train, **held-out games**):

| target | linear | MLP-512 |
|---|---|---|
| CONTROL champion world **x** — R² | 0.782 / **0.710** | 0.996 / **0.908** |
| CONTROL champion world **y** — R² | 0.893 / **0.805** | 0.994 / **0.895** |
| CONTROL dead (`hp==0`) — AUC | 1.000 / **0.698** | 1.000 / **0.858** |
| CONTROL own hp fraction — R² | 0.387 / **−0.367** | 0.984 / **+0.060** |
| **TARGET gold lump this frame — AUC** | 0.952 / **0.611** | 0.996 / **0.754** |
| **TARGET gold lump within next 10 f (0.5 s) — AUC** | 0.948 / **0.738** | 0.999 / **0.778** |
| TARGET reward magnitude `r_t` — R² | −0.074 / **−0.916** | 0.718 / **−1.066** |

Read it in this order:

1. **The pipeline is validated on a known-good signal.** World position comes off the
   latents at R² 0.91 cross-game — terrain is visible, the probe finds it. Anything the
   latents encode, this probe can find.
2. **The reward's *occurrence* is present.** "A gold event lands within the next 0.5 s"
   reads out at **held-out AUC 0.778**, on 8 games the probe never saw. That is the
   quantity the reward head has to produce, and it is in the latents. Independent
   agreement with `REWARD_MODEL_INVESTIGATION.md`'s 0.904, on ~400× the events.
3. **The reward's *magnitude* is not.** `r_t` regression is R² < 0 held-out, i.e. worse
   than predicting the mean. Expected: `r_t` is 90% a 0.5 s passive clock tick (genuinely
   unpredictable from pixels) plus a rare heavy tail. **A reward head should be trained to
   predict the event, not the scalar** — or the drip should be removed from the target,
   which §7.1's zero-sum switch does for free.
4. **HP is not readable cross-game** (R² +0.06), reproducing `DESIGN_DECISIONS.md` §7's
   0.16. Any future HP/trade reward term (§7.4) would have to come from labels or a
   CV reader, not from the tokenizer.

The train/held-out gaps are large (0.95 → 0.61 for the linear lump probe) — an 8,192-dim
probe on 43k samples overfits; only the held-out column is meaningful.

**Conclusion for §4: observability is NOT the blocker.** The reward is readable off the
frozen latents at useful discrimination. That was the one finding that could have made the
whole reward question moot, and it did not fire.

---

## 5. What the published MOBA agents use

The three independent, well-documented shaped rewards for MOBAs, normalised to
**own death = −1** so they can be compared to ours:

| signal | OpenAI Five (Dota, arXiv:1912.06680 Tab. 6) | JueWu 1v1 (arXiv:1912.09729 Tab. 6) | JueWu 5v5 (arXiv:2011.12692 Tab. 4) | **ahriuwu** |
|---|---|---|---|---|
| gold gained (per gold) | 0.006 | 0.008 | 0.005 | **0.005** |
| XP / level gained | 0.002 | 0.008 | 0.001 | — |
| own HP (frac of max) | 2 (quartic warp) | 2.0 | 3 (4th power) | — |
| own mana | 0.75 | 0.8 | 0.05 | — |
| own death | −1 | −1.0 | −1 | **−1** |
| kill enemy hero | **−0.6** (offset) | **−0.5** (offset) | +1 | **+1.5** |
| last hit | −0.16 (offset) | +0.5 | +0.2 | (via gold) |
| deny | +0.15 | — | — | — |
| tower / plate | 2.25–6 | 10.0 | 1 | (via gold) |
| terminal win | 5 | — | 2.5 | — (unusable) |
| wrong-lane / idle | −0.15/s | — | −0.00001/step | — |
| **zero-sum vs opponent** | **yes** | **yes** | (n/s) | **NO** |
| reward normalisation | running std | — | — | none |
| game-time decay | 0.6^(T/10min) | — | — | none |
| γ | 0.9993 (H=180 s @ 0.133 s/step) | 0.997 (46 s half-life) | 0.998 | 0.997 (11.5 s half-life @ 0.05 s/step) |
| GAE/λ horizon | 45–840 s (annealed) | — | — | **0.4 s** |

Three things fall out of this table.

1. **Our two guessed constants are right.** `gold_scale/|death_penalty| = 0.005` per gold
   is within 20% of OpenAI Five (0.006) and identical to JueWu 5v5 (0.005). The
   `DESIGN_DECISIONS.md` note that they were "guessed before any return was observed" is
   true, but the guess landed on the published value.
2. **What we are missing is every other row.** In particular XP, the HP/trade term, and
   the zero-sum subtraction. Note also that both OpenAI Five and JueWu 1v1 put a
   *negative* offset on hero kills specifically because gold+XP already over-pay for them.
   We apply no such correction, and after their zero-sum step a kill and a death are
   symmetric for them where ours are 1.5 : 1.
3. **Zero-sum is how they all handle passive income.** No paper excludes passive gold
   explicitly. They do not have to: it is symmetric across the two laners, so subtracting
   the opponent's reward cancels it exactly. This is the same fix as our §2.2 finding and
   §7.1 recommendation.

The structural principle underneath (Ng, Harada & Russell 1999): shaping that is not of
the form `F(s,s') = γΦ(s') − Φ(s)` has no policy-invariance guarantee. The canonical
failure is Randløv & Alstrøm's bicycle: reward the positive part of a difference and not
the negative, and the agent rides in circles. Our reward is exactly that shape —
`Δgold ≥ 0` always, with the only negative term being a one-shot −0.2 that is not even
terminal. The gold accumulator can't be cycled (you can't un-earn gold), so it is not the
literal bicycle bug, but it is the same family: **the only negative feedback in the whole
reward is a one-shot −0.2 that fires on 0.01% of frames.** Nothing else the agent can do
is ever punished.

Also relevant: OpenAI Five's sparse-reward ablation (win/loss only) still beat a scripted
bot, at a large sample-efficiency cost; VPT's RL fine-tune only works *from a BC prior with
a KL leash* (from scratch it "fails to get almost any reward"; without the KL it
catastrophically forgets). Both are arguments that a mediocre reward on top of a decent BC
prior is not automatically destructive — provided the KL term is doing real work. As of
`af72fa2` it does: the MTP-offset-0 defect (`WIRING_AUDIT` §2.1), which made the prior a
zero-init head and the KL a pull toward uniform, is fixed and Phase 3 now samples and
regularises at offset 1. That is a precondition for anything in this document to matter.

---

## 6. What Phase 3 would learn tomorrow

Assume everything mechanical is fixed (buckets, MTP offset, checkpoint loading) and only
the reward is as configured. Then:

**It would learn a last-hit reflex, and nothing else.** That is the one behaviour whose
causal action (−0.2 s) and payoff (immediate) both fit inside a 0.4 s horizon, and it is
where 66% of the reward's *events* (29% of its lump gold) live. SPECULATIVE but well-supported.

**On 95% of samples it would learn noise.** No event in the window ⇒ advantage is
value-head error ⇒ PMPO's sign split is a coin flip ⇒ half the sampled actions get pushed
up and half get pushed down at random, restrained only by the KL to the BC prior (which,
post-`af72fa2`, is at least a real prior). CONFIRMED (the 95% figure) + SPECULATIVE (the consequence).

**Where it would drift, if it drifted anywhere:**

- **Toward greed.** kill:death = 1.5:1, no bounty term, death non-terminal, gold keeps
  paying while dead. Tower-diving and coin-flip all-ins are positively rewarded.
- **Toward ignoring the opponent entirely.** Zoning, denying, freezing, poking, and
  winning trades produce literally zero reward. Losing lane 0/5 while farming under tower
  scores the same as winning it.
- **Never toward wave management, recall timing, plate timing, or objectives.** All resolve
  at 10–60 s, i.e. 25–150× the horizon.

**Would it beat the BC policy?** SPECULATIVE, and the honest answer is *probably not
measurably, in either direction* — the KL leash plus a mostly-noise gradient plus 95%
uninformative samples is a recipe for a policy that moves slowly and randomly around the
prior. The failure mode is not "learns something bad"; it is "burns GPU and lands
statistically indistinguishable from where it started". Given the movement head is already
blind (`MOVEMENT_HEAD_BLIND_2026-08-26.md`), the reward would not be the binding
constraint anyway.

---

## 7. What would have to change

### 7.1 Zero-sum the reward — the single highest-value change

`RewardConfig(use_solo_gold=False)` already implements `gold_diff_scale · Δ(own − opponent
gold_total)`. **The reason recorded for not using it is factually wrong.**
`DESIGN_DECISIONS.md` §1 says option C "needs the opponent resolved *and* visible, which
fails exactly when the opponent leaves screen". Measured, over 146 games:

| | coverage |
|---|---|
| own `gold_total` present | 99.86% of frames |
| **lane opponent `gold_total` present** | **99.81% of frames** (n=87 games) |
| all 10 heroes' gold/hp/level/world present | 99.86% of frames |
| lane opponent actually **on screen** | 49.1% of frames |

`visible_heroes` is a **full memory read** (it comes from `raw_mem.json`, which lists every
hero's position/hp/gold/level every frame regardless of fog). The name is misleading. The
opponent is on screen half the time; his *labels* are there essentially always. CONFIRMED.

**Benefit:** recovers the 50% of lane-advantage variance that §2.2 shows is currently
invisible, cancels passive income exactly, and makes denying, zoning and killing rewarded
for the first time. **Cost:** one flag, plus retuning `gold_diff_scale` (currently 5e-5,
i.e. 20× smaller than `gold_scale` — a ratio that was never measured against anything; see
below).

**Landmine, CONFIRMED by reading `replay_dataset.py:265-275, 446-490`.**
`ReplayLatentSequenceDataset._cache_meta` keys the index cache on
`(latents_dir, seq_len, stride, movement_source, matches, schema)` — **the reward config is
not in that key**, and `_parse_match` stores the computed rewards inside the cache.
Flipping the flag with `data/phase2_bc_clicks/dataset_cache.pt` (431 MB) on disk would
silently train on the *old* rewards. Bump `schema` or delete the cache.

**What the switch actually does to the reward, measured on the same 87 games
(2,191,208 frames), death term held at −0.2:**

| | solo, `gold_scale=1e-3` | diff, `gold_diff_scale=5e-5` |
|---|---|---|
| frames with `r != 0` | 10.00% | **3.63%** |
| frames with `r > 0` | 9.99% | 1.77% |
| **frames with `r < 0`** | **0.01%** | **1.86%** |
| std `r` | 0.00573 | 0.00215 |
| H=8 return exactly 0 | 23.2% | 74.4% |
| **H=8 return < 0** | **0.1%** | **13.1%** |
| episode sum, mean ± std | +7.55 ± 4.52 | −0.51 ± 0.51 |

Three readings:

- **The passive drip cancels exactly**, as the MOBA papers assume it does: non-zero frames
  drop from 10.0% to 3.6%, and what is left is events only.
- **The reward acquires a negative direction for the first time.** 1.86% of frames and
  13.1% of 0.4 s windows go negative, against 0.01% / 0.1% today. That is the single
  structural thing solo-gold cannot have.
- **`gold_diff_scale = 5e-5` is the wrong constant.** The dense term telescopes to
  `scale × final gold diff`, and final gold diff is 13 ± 2,588 g — so at 5e-5 the whole
  episode's lane outcome is worth ±0.13 while deaths alone are −0.51. The episode return
  would be *dominated by the death count*. Set it to **1e-3**, the same as `gold_scale`:
  that is exactly the published gold : death ratio (1e-3/0.2 = 0.005/gold, vs OpenAI Five
  0.006 and JueWu 5v5 0.005), and it puts lane outcome at ±2.6 against −0.51 of deaths.

**Cross-check with `REWARD_MODEL_INVESTIGATION.md` §6.2**, which recommends raising
`gold_scale` 1e-3 → 1e-2 to use more of the symlog bucket budget. That is a
*representation* argument and it is sound, but it silently multiplies the gold : death
ratio by 10 (0.05 per gold in death units, ~10× every published MOBA agent). **If
`gold_scale` goes to 1e-2, `death_penalty` must go to −2.0** to keep the behavioural
trade-off where it is; symlog(−2.0) = −1.10, comfortably inside ±3.

### 7.2 Set the horizon from the decision timescale

H=8 (0.4 s) covers last-hitting only. Trades resolve in 2–5 s = 40–100 frames; at H=40 the
event coverage rises from 5.0% to 18.3%, at H=100 to 34.0%. λ=0.95 (1.0 s mixing) should
move to ~0.99 to match. Published MOBA agents use 45–180 s GAE horizons at a 0.133 s tick. Note
this is a compute decision, not a reward decision — it is the thing that determines what
Phase 3 is *capable* of learning.

### 7.3 Price death correctly and make it terminal

- Measured own opportunity cost: 107 g @30 s, 192 g @60 s.
- Bounty to the enemy: ~300 g, currently uncounted (fixed for free by §7.1).
- `continues` should be 0 on death in imagination. Note this is **not** a one-liner: there
  is no continue/termination head, so nothing in a dream knows a death happened. Either add
  one (Dreamer has it; we dropped it) or fold the expected post-death loss into the penalty
  — measured, that is ~190 g of foregone farm plus the ~300 g bounty, i.e. ≈ −0.5 not −0.2.
- Under a zero-sum reward the enemy's death becomes a positive automatically, which
  removes the need for the ad-hoc +1.5:−1 kill:death asymmetry.

### 7.4 Terms the labels already support, in priority order

Everything below is present on 99.86% of frames for all ten heroes and needs no new data:

| term | published analogue | our label source |
|---|---|---|
| **Δ(own − opp gold)** | OF gold + zero-sum | `champion_stats.gold_total`, `visible_heroes[opp].gold_total` |
| **Δ(own − opp level)** as an XP proxy | OF XP 0.002 | `level` (integer, so it is a coarse step function — LIKELY adequate) |
| **Δ(own hp frac) − Δ(opp hp frac)**, quartic-warped | OF Health Changed 2 | `hp / hp_max` for both |
| kill / death events, both sides | OF ±1 | `hp` crossing 0 for any hero |
| positional: distance to opponent, to own tower, lane deviation | OF Lane Assign −0.15/s | `champion_world`, `visible_heroes[*].world` |
| items completed | OF Gold Spent 0.0006 | `inventory` (present on only **59.5%** of frames — weakest of the set) |

**Not derivable from these labels:** minion positions/HP (so no wave state, no true CS
count, no denies), turret HP, ward state, XP itself, cooldowns, objective timers. Wave
management — the thing that actually separates Emerald from Gold — remains out of reach
without new extraction. CONFIRMED.

### 7.5 Two mechanisms worth copying that cost nothing

- **Running-std normalisation of the reward** before the value loss (OpenAI Five, Table 2).
  Interacts directly with the twohot range item.
- **Multi-head value decomposition** (JueWu 5v5): group reward terms into farm / trade /
  objective heads with a value head each. Published Elo win, and it makes the value
  function debuggable — you can see *which* head is mispredicting.

---

## 8. What I could not rule out

1. **How well the reward head reads the signal in a *dream*, not on real latents.** §4
   measures readability on real frozen latents (held-out AUC 0.778) and
   `REWARD_MODEL_INVESTIGATION.md` measures the trained head on real windows (0.904).
   Neither measures the head applied to a *generated* latent 8 steps into an imagined
   rollout, which is what Phase 3 actually does. If dreamed latents are off-distribution
   the head's predictions there could be arbitrary, and nothing here rules that out.
2. **Game outcome.** There is no `garen_win` manifest on disk, so I could not correlate the
   reward with *winning*, only with lane state. All of §2.1 is "did Garen beat his lane
   opponent", not "did Garen's team win".
3. **True CS.** No minion data in the labels. The 12–22 g lump bands are almost certainly
   minion last-hits, but I inferred that from the gold spectrum, not from a CS counter.
4. **The 59% of lump gold with no preceding attack label** (§2.5). Ability kills, assists,
   plates and ally gold are not separable with current labels, so I cannot say how much of
   the reward is genuinely Garen's own doing.
5. **Whether a zero-sum reward would actually train better.** §2.2 shows it carries 2× the
   relevant variance and §7.1 shows it is the only version with a negative direction, but
   it is also 2× noisier per window (`var(d_diff)/var(d_own) = 2.02`) and leaves 74% of
   0.4 s windows at exactly zero instead of 23%,
   and no one has run the comparison. It is a hypothesis with a strong prior from three
   published MOBA agents, not a measured result here.
6. **Whether any of this matters yet.** With the movement head blind and the Phase-2 →
   Phase-3 handoff discarding the BC policy, reward quality may not be the binding
   constraint on Phase 3's outcome at all.
7. **Per-game variance.** n=146 games, but only n=87 have a resolvable lane opponent, and
   the §2.1 correlations are over those 87. Bootstrap CIs are given for the headline
   number only.
