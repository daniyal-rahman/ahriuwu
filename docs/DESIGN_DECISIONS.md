# DESIGN_DECISIONS.md — where Dreamer 4 is silent, what we chose, and why

Companion to [`PAPER_DEVIATIONS.md`](PAPER_DEVIATIONS.md). That document answers *"where do we differ
from the paper?"*. This one answers the question the paper **cannot**: for every design point Dreamer 4
leaves unspecified, **what was the option space, which option did we take, and why is it right for
this use case?**

The distinction matters. A deviation is measured against a fixed reference; a decision is measured
against the alternatives we *could* have picked. Most of what determines whether this agent plays
League well lives in the second category, because Dreamer 4 is a general method paper and specifies
almost nothing about action spaces, rewards, or discretization.

**Use case, fixed for all of the below.** Single champion (Garen), top lane, 1v1 laning phase.
20 fps, 352×352 frames, 512×16 latents. 125 action-labeled Masters+ replays (≈3.55M frames,
≈177k game-seconds) + 453 unlabeled YouTube games. One RTX 5080 for training. Target: hold its own
against League's built-in heuristic bots, then Emerald-level laning.

Each decision below carries: **Paper** (what it says, with anchor, or *silent*) · **Options** ·
**Chosen** · **Why here** · **Cost** · **Falsifier** — the observation that would change the choice.

Status key: **SOUND** (analysis still holds) · **MIS-TUNED** (right choice, wrong constant) ·
**SUSPECT** (measurement argues against it) · **UNEXAMINED** (chosen by default, never reasoned).

---

## 1. Reward definition — **SUSPECT**

**Paper.** Silent on reward. Dreamer 4 uses environment reward; §3.3 only says the reward head is
trained with symexp twohot. For Minecraft it inherits the benchmark's reward. **League has no
environment reward signal available from a replay** — this is entirely ours.

**Options.**

| Option | Density | Requires | Aligns with |
|---|---|---|---|
| A. Terminal win/loss | 1 per game | outcomes manifest | winning; useless for laning credit |
| B. Δ own `gold_total` | ~1 per last-hit | replay labels only | farming |
| C. Gold **diff** vs lane opponent | dense | resolved opponent + visibility | winning lane |
| D. Hand-shaped (CS + trades + plates + wave state) | dense | heavy label engineering | designer's idea of good play |
| E. Potential-based shaping over B | dense | a potential function | B, provably policy-invariant |

**Chosen: B**, `gold_scale=1e-3 · Δ(own gold_total)` + `death_penalty=-0.2`
(`src/ahriuwu/rewards/reward.py:46-48`). C exists behind `use_solo_gold=False` and is off; A exists
behind `use_outcome` and is off.

**Why here.** B is the only dense signal recoverable from replay labels alone with no visibility
assumptions — C needs the opponent resolved *and* visible, which fails exactly when the opponent
leaves screen, i.e. the frames where the reward would matter most. B telescopes cleanly
(Σ Δgold = total gold), so returns have an interpretable scale. D was rejected as the thing the
paper exists to avoid: hand-specifying good play defeats the point of learning it.

**Cost.** B says nothing about wave state, plate timing, map position, or objectives — the agent is
told "get gold" and nothing else. Everything separating Emerald from Gold in lane (wave management,
freeze/slow-push, recall timing) is invisible to it. This is the single narrowest assumption in the
stack.

**Falsifier — already fired.** The reward head should predict imminent gold. Measured AUC for "will
this swing last-hit" is **0.431 / 0.29 — worse than chance**; probing shows it behaves as an
attack-windup detector. Either the signal isn't in the latents (see §7) or B is too sparse at the
frame level to learn from. **Do not treat the current reward head as a reward model.**

---

## 2. Twohot bucket range — **MIS-TUNED**

**Paper.** Silent. §3.3 says *"symexp twohot"* citing Dreamer 3. **±20 is a Dreamer 3 number, not a
Dreamer 4 one** — any doc claiming the paper specifies a range is wrong.

**Options.** ±20 (Dreamer 3 inheritance) · ±3 · ±0.5 · percentile-adaptive from observed returns ·
learned bucket edges.

**Chosen: ±3, 255 buckets** (`heads.py`), after ±1.5 (`7013f3a`) → ±3.0 (`25bed26`).

**Why here.** With `gold_scale=1e-3`, a full laning phase of ~8k gold gives returns O(1–8), and
symlog compresses the tail. ±3 was picked to leave headroom for kill/streak spikes without
saturating. The reasoning was sound; the constant was guessed before any return was observed, and
the docstring says so: *"TUNE once real return magnitudes are seen."*

**Cost — measured.** **39 of 255 buckets are ever used.** 85% of the head's capacity is dead, and
the live buckets are packed so tightly that the twohot interpolation is doing the work a plain
scalar regression would do. The range is too **wide**, not too narrow.

**What to do.** Set the range from the empirical return distribution (e.g. 1st/99th percentile of
realized λ-returns) rather than a guess; or drop to ±0.5 which covers the observed mass. This is a
one-line change and should happen before Phase 3, since the value head inherits the same range.

---

## 3. Movement action representation — **SOUND, with a now-available better option**

**Paper.** Models per-frame actions directly (§3.3). Silent on how a mouse-driven action space
should be encoded, because Minecraft's action space is keyboard + camera delta.

**The problem this must solve.** Humans issue ~2 movement commands/s; we sample at 20 fps. So **~90%
of frames are "the previous command is still in effect."** A naive per-frame target teaches the model
to copy its own last action — measured previously as BC collapsing to a 1.8% action-change rate.

**Options.**

| Option | Camera-invariant? | Handles 90% holds? |
|---|---|---|
| A. Per-frame continuous (x,y) regression | no | no — regresses to the held value, learns copy |
| B. Per-frame categorical over a screen grid | no | no — same copy shortcut |
| C. **Sticky gate + categorical** (fire/hold mixture) | no | yes — models onset separately from target |
| D. Velocity / delta targets | partly | partly — still per-frame |
| E. **World-space target** (unproject to game coords) | **yes** | needs C on top anyway |

**Chosen: C.** One gate logit per MTP offset; mixture NLL
`transition → log g + log p_cat`, `hold → logaddexp(log(1−g), log g + log p_cat)`.

**Why here.** C is the only option that separates *when* a command is issued from *where* it points,
which is the actual structure of the data. It was the direct response to a measured pathology, not a
guess.

**Cost.** Largest architectural addition with no paper analogue; still **untested against an ungated
baseline**; blocks Phase 3 (PMPO's `log_prob` needs the previous action threaded through the imagined
rollout).

**E is now available and wasn't before.** Fixing the movement labels required building an invertible
camera model (`_Projection` in `replay_dataset.py`, verified to ≤1 px on 10,725 reprojections). That
same inversion makes **world-space movement targets** feasible for the first time. Screen-space
targets are expressed in a moving frame — which is precisely how the original 47.5%-camera-drift bug
became possible. World-space targets are camera-invariant by construction and cannot regress that
way. **Recommended for v2: C's gate + E's coordinate frame.**

---

## 4. Movement discretization — 21×21 — **UNEXAMINED**

**Paper.** Silent.

**Options.** 11×11 · 21×21 · 41×41 · continuous regression · **polar (angle × distance)** ·
champion-relative offsets.

**Chosen: 21×21 = 441 bins** over the screen. (Note `train_agent_finetune.py:699` sets
`movement_bins = 11` in the **smoke-test** path only; production is 21.)

**Why here.** No recorded rationale — this is a default that was never reasoned about, which is why
it is tagged UNEXAMINED.

**Cost — measured.** 18.6% of real commands land in the same bin as the previous command (down from
38.0% under the old labels). Those are commands the categorical cannot express; only the gate
records that anything happened. 7.3% more are off-viewport and clamped to the screen edge, which
preserves direction and destroys distance.

**Why polar deserves a look.** In League, *direction* matters far more than *distance* — "walk toward
the enemy" and "walk 200 units toward the enemy" are nearly the same decision, while a 10° heading
error is a missed skillshot dodge. An angle×distance factorization spends bins where the decision
actually lives, and makes the edge-clamp lossless (clamping distance, keeping angle exactly).

**Falsifier.** If per-bin confusion is concentrated among angular neighbours, the grid is wasting
capacity on distance; if it's concentrated in radius, 21×21 is fine.

---

## 5. Ability representation — multi-hot Bernoulli — **SOUND for Garen**

**Paper.** Silent. Minecraft actions are discrete keypresses; League casts have positions.

**Options.** A. multi-hot independent Bernoulli per key · B. single categorical over
{none, Q, W, E, R, …} · C. (key, aim-point) tuple with a second spatial head ·
D. multi-hot keys **+ the existing cursor channel supplying the aim point implicitly**.

**Chosen: A.**

**Why here.** Garen's kit is almost entirely self-centred: Q is a self-buff, W a shield, E a
point-blank AoE. Only R is champion-targeted. So an aim point is nearly unnecessary *for this
champion*, and A keeps the head small and the labels trivially recoverable.

**Correction to an earlier claim.** A previous version of the assumptions ledger called untargeted
abilities KNOWN-VIOLATED, on the grounds that we emit "press Q" and not "Q at (x,y)". That
overstates it: **the cursor position at the moment of the keypress *is* the aim point**, and it is
present in the data at every frame — so option D is available essentially for free, and the model can
learn "Q while pointing there" as a joint relationship without a second head.

**The real risk is that we may have just removed that signal.** The legacy movement target was
`cursor.screen` per frame — the mouse position. The new target is the *click destination*, held
constant between clicks. These are different quantities, and the aim point lives in the former.
**For v2, feed both channels: click-target for movement, per-frame cursor for aim.** Garen masks the
consequence today; any skillshot champion would expose it immediately.

---

## 6. Discount, λ, and imagination horizon — **MIS-TUNED (internally inconsistent)**

**Paper.** γ=0.997 is stated and we match. **λ is never stated. The imagination horizon is never
stated.** Both are ours.

**Chosen.** γ=0.997, λ=0.95, `--horizon 8` (`train_imagination.py:91,98,99`).

**The inconsistency, in frames.** At 20 fps:

| quantity | value | effective horizon | in seconds |
|---|---|---|---|
| γ = 0.997 | discount | 1/(1−γ) ≈ 333 frames | **16.7 s** |
| λ = 0.95 | bootstrap mix | 1/(1−λ) = 20 frames | **1.0 s** |
| horizon | rollout length | 8 frames | **0.4 s** |

**The rollout is shorter than both constants that are supposed to act over it.** Over 8 steps,
γ discounts by only 0.997⁸ = 0.976 — effectively 1.0, so γ does nothing. λ=0.95 over 8 steps puts
~34% of the weight on the final bootstrap, so the λ-return is dominated by **the value head's
estimate at 0.4 s** — a head that has never been trained. In other words, at H=8 the entire
imagination objective rests on an untrained value function, and the two constants we chose are
inert.

**Why the constants aren't crazy in isolation.** γ=0.997 is the paper's. λ=0.95 is the standard
Dreamer-lineage value, chosen by convention rather than analysis — which is the problem: at 20 fps
it means 1 second, and almost no League decision resolves in 1 second. Last-hit timing is ~0.5–1 s
(the only thing it covers), trades are 2–5 s, wave management is 30 s+.

**What to do.** Pick the horizon from the decision timescale you want to learn, then set λ to match:
trades need H ≈ 40–100 frames and λ ≈ 0.99; wave management needs H in the hundreds, which is
probably out of reach on one GPU. H=8 can only ever learn the shortest tactical reflexes. This
should be settled **before** Phase 3 burns compute, since it determines what Phase 3 is even capable
of learning.

---

## 7. Aux state head — 4 targets — **SUSPECT**

**Paper.** No aux head at all. Entirely ours.

**Options.** none · 4 scalars (HP, level, enemy HP, enemy visible) · richer state (minion counts,
cooldowns, gold, wave position) · state as a direct policy **input** rather than an aux target.

**Chosen:** 4 scalars, masked MSE, `--aux-state-weight 0.5`.

**Why here.** The v7 tokenizer preserves HUD detail too weakly for probes (cross-game HP R² ≈ 0.16),
so the intent was to force game semantics into the trainable agent blocks via gradient rather than
to read state out.

**Cost.** Two independent reviews call this the wrong layer — an aux target *cannot exceed what the
frozen latents contain*, so if HP isn't in the latents the head cannot invent it. The alternative
they recommend (scalar state as a direct input, e.g. from the CV HP-bar reader) sidesteps the
tokenizer entirely and is strictly more informative.

**Falsifier.** The oracle-state ablation in §8 answers this directly.

---

## 8. The measurement that would settle "tokenizer vs dynamics"

Both candidate bottlenecks have supporting evidence, and arguing from evidence has not resolved it.
The decisive experiment is an **oracle-state ablation**: train the policy on ground-truth state from
replay labels (own/enemy HP, level, positions) *instead of* latents.

- Plays materially better → **perception is the bottleneck**; tokenizer work is justified, and the
  CV HP-bar reader becomes a shortcut worth shipping.
- No better → the bottleneck is downstream (policy, reward, or the never-run Phase 3), and a better
  tokenizer buys nothing.

Hours, not days, and it gates a multi-week decision. **Run this before any tokenizer retrain.**

---

## 9. Decisions inherited without examination

Listed so they are not mistaken for reasoned choices. Each was a default that no recorded analysis
supports: `seq_len=16` / `stride=8`; `--action-dropout 0.15`; `--aux-state-weight 0.5`;
`num_register_tokens=8`; `agent_layers=4`; the 3-gold last-hit threshold; `death_penalty=-0.2`;
`gold_scale=1e-3`. Any of these could be right — none has been tested, and several
(`gold_scale` especially, since it sets where returns land in §2's buckets) interact with decisions
that measurement has already shown to be mis-tuned.
