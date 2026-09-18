# The reward buckets are not the defect — the reward head works

**Date:** 2026-08-27. **Trigger:** `WIRING_AUDIT_2026-08-20.md` §1.6 was listed as a
Phase-3 blocker ("reward twohot buckets sized for returns, fed per-frame rewards").
**Verdict: §1.6's measurements replicate exactly; its diagnosis does not survive
them.** The bucket grid is already 6.7x finer than DreamerV3's default *on our own
data*, and the trained reward head measurably reads income events off the latents.

---

## 0. Summary

| claim | status |
|---|---|
| §1.6's raw numbers (3,554,768 targets, ~99.8% in one interval, 34/254 occupied, width 0.0236, median nonzero 0.001) | **CONFIRMED — reproduced to the frame** |
| "buckets sized for returns" (i.e. too coarse for rewards) | **REFUTED.** Ours are 6.7x *finer* than DreamerV3's default and carry 4.4x more learnable nats on this data |
| "ValueHead ... roughly correct — the two heads should not share it" | **REFUTED in the stated direction.** ±3 is near-optimal for the value head (1.95 learnable nats) and merely conservative for the reward head. Splitting buys ~1.5-2.3x, not a fix |
| "RL against a degenerate reward is pointless" (the framing that made this a blocker) | **REFUTED.** Held-out income-event AUC **0.904**, with clean controls |
| the reward is ~29% a clock (passive gold) | **NEW — CONFIRMED**, and harmless: a constant-rate term cancels out of PMPO advantages |
| the real per-frame signal is 0.63% of frames, not 10% | **NEW — CONFIRMED** |

**Do not resize the buckets to "fix" §1.6.** There is a real but modest improvement
available (§6); it is a config change to fold into the next Phase-2 run, not a blocker.

---

## 1. What the code actually does (verified by reading, then by running)

- `RewardHead` (`src/ahriuwu/models/heads.py:48-140`): 255 buckets,
  `bucket_centers = linspace(-3, 3, 255)` **in symlog space**, width **0.0236220**,
  raw coverage `±symexp(3) = ±19.09`. Zero-init output heads. 9 MTP heads.
- Target (`scripts/train_agent_finetune.py:759`): `targets = symlog(rewards)`, raw
  per-frame reward. Loss is the **mean** over offsets n=0..8 (`:769`).
- Decode (`heads.py:138`): `symexp(twohot_decode(logits, bucket_centers))` —
  encode and decode are in the *same* space. This is the DreamerV3-2023 formulation
  implemented consistently. (Contrast `vijayabhaskar-ev/dreamer_v4` `heads.py:197`
  vs `:233`, which encodes in symlog space and decodes in raw space — a real bug we
  do **not** have.)
- `ValueHead` (`heads.py:507-586`): same 255 / ±3 grid; target is
  `symlog(lambda_returns)` (`train_imagination.py:385`).
- So §3 of the brief is confirmed: **reward head → per-frame reward, value head →
  returns.** The formulation is right. The shared range is the only coupling.

---

## 2. Re-measuring §1.6

Over the exact BC training corpus (`data/phase2_bc_clicks/dataset_cache.pt`,
125 matches, **n = 3,554,768** — the audit's number to the frame):

| §1.6 | re-measured |
|---|---|
| 99.79% in one bucket | **99.79%** by floor-into-interval; **99.36%** by nearest-centre (bin 127) |
| 34 of 254 intervals occupied | **35 of 255 bins** carry hard mass |
| width 0.0236 | 0.0236220 |
| median nonzero reward 0.001 | 0.001 |

All four replicate. The two "disagreements" are only assignment conventions.

**But 35 occupied bins is the correct answer, not a symptom.** 35 bins spans symlog
0.83 = **1,290 gold in a single frame**; the largest reward in the corpus is 1.09
(1,090 gold). The occupied range covers the data exactly; the rest is headroom, which
under twohot costs nothing but a slightly diffuse softmax normaliser.

---

## 3. What our reward actually is

`RewardConfig` (`src/ahriuwu/rewards/reward.py:32-59`), default solo-gold mode:
`gold_scale · Δ(Garen's own gold_total)` per frame, `+ death_penalty = -0.2` on each
hp>0 → hp<=0 transition. `gold_scale = 1e-3`, flagged in its own docstring as
"a rough default — TUNE once real return magnitudes are observed". **It was never
tuned.** Enemy gold-diff, lane anchor and win/loss are all off by default.

Decomposition over all 3,554,768 frames (n and gold-share of all positive reward):

| band | frames | % of frames | share of positive reward |
|---|---|---|---|
| exactly 0 | 3,196,873 | 89.932% | — |
| **passive tick** (<2.1 gold) | 334,584 | 9.4123% | **28.74%** |
| 2.1-10 gold | 680 | 0.0191% | 0.32% |
| 10-25 g (melee minion) | 15,201 | 0.4276% | 21.96% |
| 25-45 g (caster) | 2,790 | 0.0785% | 7.64% |
| 45-100 g (cannon/plate) | 2,832 | 0.0797% | 14.75% |
| >100 g (kill/turret) | 1,413 | 0.0397% | 26.59% |
| deaths (negative) | 395 | 0.0111% | sum −78.86 |

Per game (n=125): median 31,920 frames, **199 income events** (>=10 gold, p10 63,
p90 253), 3,028 passive-tick frames, 3 deaths (20/125 games have none). Event gold
is a median **69.1%** of positive reward per game (range 42.6-79.7%).

**The "median nonzero reward = 0.001" that made §1.6 alarming is the passive-gold
tick** — LoL's flat 2.04 g/s income, quantised by the memory read into +1.0/+1.1
steps. It is 93.5% of nonzero frames and only 28.7% of the reward. It is also
**action-independent**: a constant-rate term adds the same constant to every state's
value and cancels out of `A_t = R_t − v_t`, so it is inert for PMPO — noise in the
head's training target, not a corruption of the RL objective.

The behaviourally meaningful reward is the other 0.63% of frames, and those are
**not** near a bucket edge:

| event | r | symlog | ours: bins from 0 (twohot weight) | DreamerV3 default |
|---|---|---|---|---|
| passive tick (1 g) | 0.0010 | 0.001000 | 0.042 (w=0.042) | 0.0063 |
| melee minion (14 g) | 0.0140 | 0.013903 | **0.589** (w=0.589) | 0.0883 |
| melee minion late (21 g) | 0.0210 | 0.020783 | **0.880** | 0.1320 |
| caster (40 g) | 0.0400 | 0.039221 | 1.660 | 0.2491 |
| cannon (53 g) | 0.0530 | 0.051643 | 2.186 | 0.3279 |
| plate (120 g) | 0.1200 | 0.113329 | 4.798 | 0.7196 |
| champion kill (300 g) | 0.3000 | 0.262364 | **11.107** | 1.6660 |

A minion last hit moves 59-88% of the twohot mass to the next bin. That is a strong,
well-resolved target. Under DreamerV3's own default grid it would move 9-13%.

---

## 4. The cross-entropy budget — the number §1.6 never computed

Twohot cross-entropy is a proper scoring rule; concentration causes no bias and no
loss of resolution (`E[centers]` reconstructs the target exactly — already verified in
the audit's §5). What concentration costs is **nats available to learn**. The right
comparison is against the *optimal constant predictor* — a head that learned nothing:

Over the whole corpus, offset n=0:

| grid | constant-CE | perfect-CE | **learnable gap** |
|---|---|---|---|
| **ours `[-3,3]x255`** | 0.059928 | 0.020280 | **0.039649 nats** |
| DreamerV3 default `[-20,20]x255` | 0.015409 | 0.006319 | 0.009090 |
| edwhu/dreamer4-jax `[-3,3]x101` | 0.031337 | 0.011704 | 0.019632 |
| `[-1.5,1.5]x255` | 0.091877 | 0.031281 | 0.060596 |
| `[-0.75,0.75]x255` | 0.136684 | 0.047221 | 0.089463 |
| `[-3,3]x255`, reward x10 | 0.226439 | 0.067906 | 0.158534 |
| `[-3,3]x255`, reward x100 | 0.403618 | 0.027071 | 0.376547 |

**Our grid carries 4.4x more learnable signal than DreamerV3's default would on this
data.** §1.6's premise — that the grid is mis-sized *upward* — is backwards.

And the value head, whose targets are full-episode discounted returns
(gamma=0.997 → 16.7 s effective horizon; measured p50 **0.0837**, p99 0.463,
max 1.381, min −0.186):

| grid | constant-CE | perfect-CE | **learnable gap** |
|---|---|---|---|
| **ours `[-3,3]x255`** | 2.454063 | 0.501447 | **1.952616 nats** |
| DreamerV3 `[-20,20]x255` | 0.987629 | 0.554116 | 0.433513 |

±3 is *near-optimal for the value head* — 4.5x better than DreamerV3's default. The
audit's "ValueHead ... roughly correct" is right; its "the two heads should not share
it" points the wrong way. It is the reward head that would prefer a narrower grid,
and only by ~1.5-2.3x.

*(The docstrings on both heads claim "O(0.5-1) discounted returns". Measured median
is 0.084 — wrong by ~6-10x. Documentation error, not a defect.)*

---

## 4b. What the references actually do

Verified by cloning and reading each repo at the SHA cited.

| repo | reward-head parameterisation | bins | range | bins uniform in | target |
|---|---|---|---|---|---|
| danijar/dreamerv3 **2023** (`8fa35f8`) | `DiscDist` | 255 | `[-20,20]` | symlog | `symlog(r)` |
| danijar/dreamerv3 **2025** (`e3f0224`) | `symexp_twohot` | 255 | `symexp([-20,20])` = ±4.85e8 | raw | **raw** `r` |
| NM512/dreamerv3-torch (`6ef8646`) | `DiscDist` | 255 (hardcoded) | `[-20,20]` | symlog | `symlog(r)` |
| lucidrains/dreamer4 (`aef0d73`) | **HL-Gauss** by default | 255 | `(-20,20)` **raw** | raw | raw |
| edwhu/dreamer4-jax (`753d650`) | twohot | 101 | **`[-3,3]`** in both training scripts | symlog | `symlog(r)` |
| vijayabhaskar-ev/dreamer_v4 (`484759e`) | twohot | 255 | `[-20,20]` | symlog | `symlog(r)` |
| **ahriuwu (this repo)** | twohot | 255 | **`[-3,3]`** | symlog | `symlog(r)` |

- **Ours is the DreamerV3-2023 formulation, implemented consistently.** The 2023 tree
  and the 2025 rewrite are near-equivalent reparameterisations of the same thing:
  2023 interpolates in symlog space and applies `symexp` to the decoded expectation;
  2025 moves the bins to raw space via `symexp` and interpolates there. The paper
  (Eq. 10, `B ≐ symexp([−20 … +20])`) matches the 2025 code.
- **`[-3,3]` is not exotic.** `edwhu/dreamer4-jax`'s own training scripts
  (`scripts/train_bc_rew_heads.py:106-108`) override to `[-3, 3]` with the comment
  "tune per dataset". We are 2.5x finer than that.
- **No implementation surveyed normalises the reward-head target.** DreamerV3's
  `retnorm` (`perc_ema`, 5/95, `configs.yaml:94`) and NM512's misleadingly named
  `reward_EMA` (`models.py:11-27`, applied at `:404-408`) both scale **actor
  returns**. Critic targets are raw λ-returns in both (2023 `agent.py:343`; 2025
  `valnorm: {impl: none}`). `clip_reward: False` is the Atari default. The "Reward
  normalization Yes / clipping 10" row in the DreamerV3 paper is **Table 1's PPO
  baseline**, not Dreamer.
- **Dreamer 4 (arXiv 2509.24527) specifies nothing quantitative.** L288-290: "Following
  Dreamer 3, the reward head is parameterized as a symexp twohot output". No bin count,
  no range anywhere in the paper. Its rewards are **sparse binary** (L520: "we annotate
  the tasks and their sparse binary rewards"), i.e. magnitude 1, and its answer to
  sparsity is a **50/50 data mixture** (L521-525), not anything in the head. It
  explicitly removes normalisation because PMPO is sign-only (L1278-1279).
- **A bug we do not have:** `vijayabhaskar-ev` encodes the target in symlog space
  (`heads.py:197`) and decodes against raw bins (`heads.py:233`). Since `symexp` is
  convex, `Σ wᵢ symexp(bᵢ) ≠ symexp(Σ wᵢ bᵢ)` — its predictions are systematically
  inflated. Ours encodes and decodes in the same space.

**Bearing on §1.6:** the "correct" reference grid is 6.7x *coarser* than ours, would
carry 4.4x fewer learnable nats on our data, and no reference does anything about
tiny-magnitude rewards because none of their benchmarks has any — Atari points,
Crafter +1, DMC [0,1], Dreamer 4's binary flags are all O(1). **The one thing every
reference does that we do not is hand the head an O(1) reward.** That, not the grid,
is the actionable difference (§7).

## 5. What the trained head actually predicts

Checkpoint `data/phase2_from_vast/agent_finetune_latest.pt` (step 99,421), frozen
dynamics at the training tau regime (`tau ~ U(0.9, 1)` — identical to
`train_imagination.py`'s `tau_ctx_forward`, checked), reward-head offset n=0.

Two evaluations. `NA1_5549981347`'s latents live in a separate pack
(`replay_latents_v7_heldout`) and are not in the BC latents dir; the 8-match set is
from `replay_latents_v7_bc` and is very likely train (the checkpoint records no
`val_matches`, so this cannot be established — see §8).

| | held-out, 1 match | 8 BC matches |
|---|---|---|
| frames | 3,968 | 23,040 |
| income events (>=10 gold) | 34 | 146 |
| CE(head) | 0.053420 | 0.051374 |
| CE(optimal constant) — *learned nothing* | 0.061466 | 0.059586 |
| CE(perfect predictor) | 0.020712 | 0.019909 |
| **fraction of the learnable gap closed** | **+0.197** | **+0.207** |
| **event AUC** | **0.9041** | **0.8862** |

**Metric validated on known-good signals before any model number was interpreted:**

| predictor | held-out | 8 BC |
|---|---|---|
| CONTROL: the true reward (must be 1.0) | **1.0000** | **1.0000** |
| CONTROL: random noise (must be ~0.5) | 0.5062 | 0.4750 |
| CONFOUND: game time / window index | **0.5006** | 0.5615 |
| CONFOUND: passive-tick indicator | 0.4853 | 0.5966 |
| **trained reward head** | **0.9041** | **0.8862** |

Per-match AUC over the 8: 0.755, 0.803, 0.811, 0.917, 0.934, 0.945, 0.970, 0.976
(median 0.926); the matching per-match time-confound AUCs are 0.53-0.73. Every game
beats its own clock by a wide margin.

Separation: mean pred | event **0.002656** vs | passive 0.000203 vs | zero 0.000216 —
**13x**. pred std 0.001075 on mean 0.000230, so the head is state-dependent, not a
constant emitter.

**The reward head is not blind, and it is not a clock.** This is the opposite result
from the movement head (`MOVEMENT_HEAD_BLIND_2026-08-26.md`), which matched a
no-pixel lookup table to 0.008 nats. The train and held-out numbers agree to within
0.01 on both metrics, so there is no visible overfit either.

## 5b. Detector or predictor? (Phase-3 credit assignment)

Phase 3 dreams forward and needs the reward head to see income *coming*. Held-out
match, 62 windows, offset n predicting `r_{t+n}` from `h_t`:

| n | lead | events | AUC | mean pred \| event | mean \| no-event | ratio |
|---|---|---|---|---|---|---|
| 0 | 0.00 s | 34 | **0.9041** | 0.002897 | 0.000282 | 10.3x |
| 1 | 0.05 s | 34 | 0.8553 | 0.001771 | 0.000245 | 7.2x |
| 2 | 0.10 s | 32 | 0.8268 | 0.001415 | 0.000219 | 6.5x |
| 3 | 0.15 s | 31 | 0.8503 | 0.001345 | 0.000219 | 6.2x |
| 4 | 0.20 s | 31 | 0.8833 | 0.001580 | 0.000238 | 6.6x |
| 5 | 0.25 s | 29 | 0.8601 | 0.002057 | 0.000278 | 7.4x |
| 6 | 0.30 s | 28 | 0.8608 | 0.002378 | 0.000326 | 7.3x |
| 7 | 0.35 s | 28 | 0.7942 | 0.001792 | 0.000313 | 5.7x |
| 8 | 0.40 s | 27 | 0.7340 | 0.000753 | 0.000286 | 2.6x |

Every MTP head carries signal out to the full 0.4 s window. Whether that is real
anticipation or just temporal clustering of last hits is settled by scoring the
**offset-0** head against a target shifted by k frames:

| k | −4 | −2 | −1 | **0** | **+1** | +2 | +3 | +4 | +5 |
|---|---|---|---|---|---|---|---|---|---|
| AUC | 0.557 | 0.748 | 0.832 | **0.904** | **0.914** | 0.837 | 0.768 | 0.656 | 0.539 |

Offset-0 is a **sharp detector**, half-width ~±0.1 s, at chance by ±0.25 s. So the
offset-4 head's 0.883 at 0.2 s lead is **not** available by shifting offset-0's output
(0.656) — the dedicated MTP heads have learned genuine ~0.2-0.3 s anticipation.

Repeating on **4 BC matches** (180 windows, 73 events at n=0) gives the same shape,
higher throughout: AUC by offset 0.956, 0.949, 0.941, 0.919, 0.924, 0.926, 0.928,
0.886, 0.854; lead/lag peak 0.956 at k=0, floor ~0.73-0.77 by ±0.2 s.

Two notes:
- The held-out lead/lag peaks at **k=+1** (0.914 vs 0.904 at k=0) but the BC set peaks
  at k=0 (0.956 vs 0.940). A one-frame label lag behind the pixels is plausible but is
  **not** established by this — the two sets disagree, and both differences are inside
  the noise at n=34/73.
- Anticipation reaches ~0.2-0.3 s. `--horizon 8` is 0.4 s, so the reward head can see
  roughly the whole imagination window — but 0.4 s is far short of the seconds of lead
  a last-hit decision needs (§6.4).

## 5c. Hypotheses tested and REFUTED

**Numerical precision at this reward scale.** twohot encode → softmax → decode,
relative error of the recovered reward:

| r | fp32 | logits rounded to bf16 |
|---|---|---|
| 0.001 | 4.7e-05 | 6.1e-03 |
| 0.0019 | 2.2e-05 | 5.4e-03 |
| 0.014 | 4.2e-06 | 1.1e-03 |
| 1.09 | 1.4e-07 | 9.1e-06 |

Catastrophic cancellation in `sum(probs * centers)` was a plausible story for
1e-3-scale values against ±3 bin centres. It does not happen. `imagine()` carries no
autocast at all (`train_imagination.py:298-320`), so `r_t`/`v_t` are fp32; `run_step`
is bf16 but only for the head Linears. **Refuted.**

**The RMS normaliser starving the reward loss.** `run_step` does
`bc_n = norm("bc", bc_loss); rew_n = norm("reward", reward_loss); total = bc_n + rew_n`
(`train_agent_finetune.py:1076-1079`). Each term is divided by its own running RMS, so
the reward term contributes ~1.0 regardless of how few nats it carries — the reward
gradient is *amplified* ~17x relative to raw (`sqrt(EMA(L²))` = 0.0549-0.0766 across
the five checkpoints on disk, vs 0.746-1.199 for BC). `MIN_RMS` (1e-4) is nowhere near
binding. **Refuted.**

**Train/imagine tau mismatch.** BC uses `tau ~ U(0.9, 1)` (`tau_ctx=0.9`);
`imagine()` uses `U(tau_ctx_forward, 1)` with `tau_ctx_forward = 1 - 0.1 = 0.9`
(`train_imagination.py:555`). Identical. **Refuted.**

**Reward/latent slice misalignment.** `latents[start_idx:start_idx+T]` vs
`rewards[start_frame:start_frame+T]` (`replay_dataset.py:1025-1029`) — different
variables, but the audit's §5 established `frame_indices == arange(N)` on all 125
packs, and the lead/lag scan peaks at k=0/+1 rather than at some large offset.
**Refuted.**

## 5d. What the loss *does* cost: an equal-budget learnability test

Isolating the loss from perception: 64 synthetic "situations" with a random 128-d
embedding each, reward per situation drawn to match the real corpus (89.9% zero, the
passive band, the event bands, a death, a kill), reweighted to the real frequencies.
A linear head **can** fit this exactly. Zero-init, Adam 3e-4, batch 512, **800 steps
for every config** — an equal-budget speed comparison, not a converged one:

| config | constant-CE | learned CE | distance to floor, in units of its own learnable gap | wMAE of decoded reward | event-AUC |
|---|---|---|---|---|---|
| **ours `[-3,3]x255`** | 0.0621 | 0.9979 | **−22.5** | 0.00179 | 0.864 |
| DreamerV3 `[-20,20]x255` | 0.0247 | 0.9960 | **−52.1** | 0.00159 | 0.810 |
| edwhu `[-3,3]x101` | 0.0392 | 0.6734 | −22.9 | 0.00149 | 0.895 |
| **reward x100, `[-3,3]x255`** | 0.3766 | 0.7442 | **−1.02** | **0.00087** | **1.000** |
| reward x1000, `[-3,3]x255` | 0.4197 | 0.7893 | −1.04 | 0.00078 | 0.883 |

Read it this way: a zero-init softmax over 255 bins starts at `ln(255) = 5.54` nats.
**The optimizer must burn 5.54 nats reaching the marginal before it can spend anything
on the 0.040 nats of real signal — a 138:1 ratio.** Rescaling the reward by 100 makes
that ratio 14.7:1, and at equal budget the head lands **22x closer to its floor** and
halves the decoded-reward error. This is the mechanism behind the §7 recommendation,
and it is a *speed* result — the real trainer runs 100k+ steps and does reach the
floor (measured `reward_loss` ~0.059 vs a 0.0599 constant floor).

## 6. Where the reward model *is* weak

1. **It closes only ~20% of the available CE gap.** Not degenerate, not good.
2. **Untuned `gold_scale`.** At 1e-3 the whole corpus lives in ±0.74 symlog. Raising
   it to 1e-2 quadruples the learnable budget (0.0396 → 0.1585 nats) with no
   clipping (reward max 10.9, return max 13.8, both inside ±3). This is the single
   cheapest real improvement, and it is *exactly equivalent* to shrinking the grid
   near zero — where our mass is — without clipping the tail, which is why it beats
   narrowing the range (0.159 vs 0.089 nats).
3. **Phase-3 cold start.** 98.4% of the head's predictions on real frames are
   positive, and 98.0% of 8-frame blocks sum positive. The value head is zero-init,
   so at step 0 `A_t = R_t − 0 > 0` almost everywhere: PMPO's negative set is nearly
   empty and `compute_pmpo_loss` reduces to "raise log-prob of everything sampled".
   Self-correcting once the value head fits, but it is a real transient.
4. **Horizon/discount mismatch in Phase 3** (`train_imagination.py:103,110`):
   `--horizon 8` (0.4 s at 20 fps) against `gamma=0.997` (16.7 s effective). The
   discount says you care about 333 frames; the rollout gives you 8. A last-hit
   decision has seconds of lead time. This is a credit-assignment problem, not a
   bucket problem, and it is untested because Phase 3 cannot run (§2.1/§2.2).

---

## 7. Recommendation

**Strike §1.6 from the blocker list.** Do not resize the buckets on its reasoning:
narrowing them is a ~2x improvement, not a fix, and widening them (which "sized for
returns, fed per-frame rewards" implies) makes things 4.4x worse.

Fold into the next Phase-2 run — config only, no restructuring:

1. `RewardConfig.gold_scale`: `1e-3` → `1e-2`. Measured **4x** more learnable nats
   (0.0396 → 0.1585), no clipping in either head (reward max 10.9, return max 13.8,
   both inside ±3). This beats narrowing the reward grid (0.159 vs 0.089 nats)
   because rescaling expands resolution where the mass is without compressing the
   tail. PMPO is sign-only, so nothing downstream shifts. §5d measures the effect at
   equal optimisation budget: 22x closer to the floor, half the decoded-reward error.
2. **Required companion fix, or (1) silently does nothing.**
   `ReplayLatentSequenceDataset._cache_meta` (`replay_dataset.py:265-275`) keys the
   dataset cache on `(latents_dir, seq_len, stride, movement_source, matches,
   schema)` — **no reward-config field**. Every Phase-2 dir on disk has a
   `dataset_cache.pt`. Changing `gold_scale` with one present reuses the old rewards
   with no warning. Add the reward config to the cache key (this is convention #2
   in the audit's §4: one contract, checked at load).
3. Optional: give `RewardHead` its own `bucket_low/high = ±1.5` (raw ±3.48, 3.2x
   headroom over the observed max 1.09). Worth ~1.5x on top. **Leave `ValueHead` at
   ±3** — measured, that is the right grid for returns.

### Cost

- **Not** Phase 1, not the tokenizer, not re-tokenization. `gold_scale` is applied at
  label-parse time.
- **Cheap validation first (a few GPU-hours):** freeze everything except
  `reward_head` and re-fit it on a fixed Phase-2 checkpoint's agent tokens. The
  frozen-backbone Phase-2 path already trains only the agent blocks + the two heads
  (`train_agent_finetune.py:486-496`), so a reward-head-only refit is a small edit
  and answers "does x10 actually help?" without a BC rerun.
- **Full adoption:** one Phase-2 BC rerun (the reward head reads agent tokens that
  Phase 2 also trains, so the two cannot be permanently decoupled).

### Acceptance test

Run `scripts/eval_reward_head.py` (extended with the CE-gap metric used here) on
held-out matches, at reward-head MTP offset n=0, requiring **n >= 200 income events**
(>=10 gold). The corpus has **21 labeled matches with no latents yet**
(`lol_replays_16_9_772` has 148, `replay_latents_v7_bc` 125, `_heldout` 1) — tokenize
5-6 of them to reach n >= 200.

Gates, all of which must hold:

| # | gate | today |
|---|---|---|
| G0 | CONTROL true-reward AUC == 1.000; random AUC in [0.45, 0.55]; game-time-confound AUC <= 0.70 | **passes** (1.000 / 0.475-0.506 / 0.50-0.56) |
| G1 | **fraction of learnable CE gap closed >= 0.40** | **FAILS — 0.197 held-out, 0.207 train** |
| G2 | event AUC >= 0.90 (non-regression guard) | marginal — 0.904 held-out, 0.886 train |

**G1 is the number that says "fixed" and fails today.** G0 is not optional: a
reward-head AUC is uninterpretable without its controls — the game-time confound sits
at 0.50-0.73 on these same frames, and "the head learned a clock" would have looked
identical without it.

§1.6 reached the opposite conclusion because it measured only the *target*
distribution and never ran the head. A degenerate target histogram and a working
reward model are entirely compatible under twohot, which is the whole point of the
parameterisation.

## 8. Conclusions, ranked by confidence

**CONFIRMED (I ran it):**

1. §1.6's four measurements replicate exactly on the same n = 3,554,768.
2. Our grid is 6.7x finer than DreamerV3's default and carries **4.4x more learnable
   nats** on this data (0.0396 vs 0.0091). "Sized for returns" is backwards.
3. ±3 is near-optimal for the **value** head (1.95 learnable nats vs DreamerV3's
   0.43). Splitting the ranges is worth ~1.5-2.3x for the reward head only.
4. The trained reward head reads income events: **AUC 0.886-0.904** held-out/train at
   offset 0 (0.956 on a 4-match BC subset), 10-13x mean separation, with
   true-reward / random / game-time confound controls all clean.
4b. Every MTP offset carries signal across the full 0.4 s window (AUC 0.73-0.96), and
   the dedicated offset-n heads beat shifting offset-0's output — genuine ~0.2-0.3 s
   anticipation, not just temporal clustering of last hits.
5. The reward is 89.9% exact zero, 9.4% passive-gold clock ticks (28.7% of reward),
   and **0.63% real income events (71% of reward)**. Median 199 events/game.
6. Four candidate mechanisms refuted by measurement (§5c): fp32/bf16 cancellation at
   this reward scale, RMS-normaliser starvation, BC/imagination tau mismatch, and
   reward/latent slice misalignment.
6b. Reward-head and value-head roles are wired correctly (per-frame vs lambda-returns),
   symlog encode/decode spaces are consistent, tau regimes match between BC and
   imagination. No code bug found in the reward path.
7. The dataset cache key omits the reward config — changing `gold_scale` today is a
   silent no-op wherever a `dataset_cache.pt` exists.
8. 98.4% of the head's predictions are positive and 98.0% of 8-frame blocks sum
   positive, so at Phase-3 step 0 (zero-init value head) PMPO's negative set is
   nearly empty.

**LIKELY (read, not run):**

9. No DreamerV3/V4 implementation surveyed normalises the reward-head target;
   DreamerV3's `retnorm` (perc_ema 5/95) and NM512's misleadingly named `reward_EMA`
   both act on **actor returns**, not on rewards. Dreamer 4 removes normalisation
   entirely because PMPO is sign-only, and answers sparsity with a **50/50 data
   mixture**, not with the head.
10. Dreamer 4's rewards are sparse **binary** (magnitude 1). Nothing in either paper
    addresses tiny-magnitude rewards; DreamerV3 is simply never handed any.
11. `--horizon 8` (0.4 s) against `gamma=0.997` (16.7 s effective) is a
    credit-assignment mismatch for a last-hit decision that takes seconds to set up.

**SPECULATIVE:**

12. Raising `gold_scale` will translate its 4x nats into a materially better *trained*
    head. The nats are measured and the equal-budget synthetic result (§5d) is
    strongly in favour, but that test uses perfect features, a linear head and 800
    steps; the real trainer has imperfect features and runs 100k+ steps, where the
    speed advantage may wash out.
13. The remaining ~80% of the CE gap is perception-limited rather than loss-limited
    (the v7 latents are known to erase the "+14" gold popup on decode).

## 9. What I could NOT rule out

- **Whether the 8 BC matches are train or val.** No checkpoint on disk records
  `val_matches`, and only one latent pack exists outside the BC dir. The train/held-out
  agreement (0.207 vs 0.197) is reassuring but is one game against eight.
- **n is small.** 34 held-out events, 146 across 8 matches. Per-match AUC ranges
  0.755-0.976. Every AUC here has a wide interval.
- **How far the anticipation really reaches.** §5b shows ~0.2-0.3 s of genuine lead,
  but at n=27-73 events per offset the per-offset AUCs are not separable from one
  another. The claim "the MTP heads anticipate" is solid; "by exactly 0.3 s" is not.
- **Anything measured inside imagination.** Phase 3 cannot run (audit §2.1/§2.2), so
  the reward head has never been evaluated on *dreamed* agent tokens. Every number
  here is on real latents. Distribution shift between real and dreamed tokens is
  entirely unmeasured and is the single largest remaining unknown for Phase 3.
- **Whether the reward signal is action-sensitive through the dynamics** — whether
  changing the policy's action changes the reward head's prediction at all. That is
  the actual precondition for a policy gradient, and it needs a working imagination
  loop to measure.
- **Whether solo-gold is the *right* reward** (does it reward walking to lane?). A
  sibling investigation is on signal viability; this document only establishes that
  the signal is present and readable, not that maximising it produces good play.
- **The death penalty's calibration.** −0.2 = 200 gold, 395 deaths in 3.55M frames,
  −78.9 total against ~+1,300 positive. Not analysed.

---

## Reproduction

Scripts used (scratchpad, not committed — each is <150 lines and re-derivable from the
numbers above):

| what | how |
|---|---|
| corpus reward distribution, bucket occupancy, CE budget | reads `data/phase2_bc_clicks/dataset_cache.pt` (`match_data[m]["rewards"]`, all 125 matches, already reward-parsed — 3.4 GB of `labels.json` need not be re-read) |
| head CE / AUC / controls | `scripts/eval_reward_head.py`'s `load_phase2` + a twohot-CE and control-AUC block |
| MTP offset sweep + lead/lag | same loader, scoring `rh.predict(ao)[0][:, n]` against `rewards[n:]` |
| equal-budget learnability | synthetic 64-situation linear probe under `twohot_loss` |

`scripts/eval_reward_head.py` already exists and already computes the AUC; it is
missing only the CE-gap metric and the control/confound rows. Adding them there is the
natural home for the §7 acceptance test.
