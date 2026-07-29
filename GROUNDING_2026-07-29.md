# Grounding the gaps — measured, not hand-waved (2026-07-29)

Answers to: is the tokenizer too low-res? does the WM need more training or is it plateaued?
how far is our data from the paper? how much does each link lag? Plus efficiency ideas.

## 1. Tokenizer fidelity — measured with a latent probe
Train a probe (latent -> game-state from labels), test how well it reads HP/gold/level/position.

| split | champ_hp_frac | gold | level | screen pos | world pos |
|---|---|---|---|---|---|
| within-game, random | 0.82 | 0.83 | 0.82 | 0.75–0.79 | 0.98–0.99 |
| within-game, temporal-block | −0.61 | 0.01 | 0.11 | −0.1/−0.84 | 0.69/0.82 |
| **cross-game (linear+MLP, PCA-256)** | −0.05/−0.08 | ≤0 | ≤0 | 0.12/0.20 | 0.15–0.23 |
| **cross-game, full 8192-d** | **0.09** | — | — | ~0.00 | — |

**Verdict:** the latent encodes the *scene* richly (within a game it tracks state as it
changes — 0.82) but does **NOT** expose fine game-state as a clean, transferable feature.
Even reading all 8192 dims, HP-fraction is unreadable across games (R²=0.09). So the
tokenizer is a good *scene compressor*, not a *game-state extractor*. Fine HUD numerics
(HP, gold, cooldowns) are effectively blurred away.

**But it is NOT the current bottleneck:** the world model teacher-forces at ~25 dB, which
is ~9 dB **below** the tokenizer's own recon ceiling (28–29 dB). The WM is failing to use
detail the tokenizer already provides. A sharper tokenizer buys nothing until the WM closes
that 9 dB. **Order of operations: fix the WM first.**

**Cheap consequence now:** don't rely on the tokenizer to hand the policy crisp HP/mana/
cooldowns — feed those explicitly (HUD-to-policy, from labels/OCR). The probe quantifies why.

## 2. Does the WM need more training, or is it plateaued?
Job 135 (action-conditioned, best case) eval trajectory: tau0.9 = 24.9 → 25.6 → **25.6** (flat).
It **plateaued**. More training of the same config = diminishing returns. The gap is a
capability/data/objective limit, not an optimization-time limit — the loss flattened, so the
model has extracted what it can from *this* data + arch + objective. Levers that move it:
(a) action-conditioning on the *mixed* corpus (the paper recipe — never actually done: 179
was unconditioned, 135 replays-only), (b) more/better data, (c) capacity, (d) structured
priors (§5). "Just train longer" will not close it.

## 3. Data gap vs the paper (DreamerV4: ~100h action + ~2500h video)
- action-labeled replays: **49.4h → ~49% of the paper.** Not that far.
- unlabeled YT: 906 games × ~36k frames ≈ **454h → ~18% of the paper.** The bigger gap.
Your intuition was right on both. **IDM pseudo-labeling of YT converts the 454h unlabeled
into action-labeled — closing both gaps at once** (and helps rare casts: ults/summoners have
only 300–600 examples vs Q/E/AA 10–18k).

## 4. How much does each link lag "good"? (ranked)
1. **World model — dominant, order-of-magnitude gap.** 9 dB below its own ceiling; dream beats
   persistence by only ~2 dB; drifts to 9–12 dB latent by h32. Blocks imagination RL entirely.
2. **Policy (BC) — moderate, cheap.** Movement works (57% bin-acc, beats baseline). Casting is
   calibration (probe AUC 0.77–0.89, just mis-thresholded) + can't see its own HUD.
3. **Tokenizer — moderate, not binding yet** (§1).
4. **Reward — unknown, untuned placeholder;** only matters once imagination runs.
5. **Imagination RL — 0% (doesn't run), but gated on the WM,** not itself broken.

## 5. Efficiency ideas (web) + your bootstrap idea
- **Predict in latent space, not pixels (JEPA / V-JEPA 2 / DINO-WM).** Non-generative latent
  prediction is more data/compute-efficient because the encoder drops nuisance pixel detail.
  V-JEPA 2 = action-conditioned latent world model for planning. Our WM already predicts
  latents (good); the *tokenizer* is a pixel-MAE — a JEPA-style objective would give a more
  predictable latent. (This is also the lesson of our HUD-pixel-loss bug.)
- **DreamerV3 robustness tricks** (symlog, two-hot, free-bits, KL-balance, %-return norm) —
  we already use two-hot; the rest are cheap small-data robustness wins.
- **Non-curated offline data (experience rehearsal + execution guidance)** — reported +100%
  RL sample-efficiency using exactly our kind of uncurated corpus (YT).
- **Your bootstrap idea = distillation / structural-prior pretraining.** It's a real, named
  family (sim-distillation, self-distillation, subnetwork+distill → up to 5× fewer FLOPs).
  Cheapest high-leverage form for us: **auxiliary label-supervision** — give the WM an extra
  head that predicts HP/gold/position from the privileged labels during pretraining. That
  injects a strong game-semantics prior into the latent/WM (directly fixing what §1 showed is
  missing) and makes dynamics learning far more data-efficient — no simulator to build.

## The plan (updated)
1. **Action-conditioned WM retrain on MIXED data (paper recipe) + auxiliary label-supervision
   head** — the critical path fused with the highest-leverage prior.
2. **IDM pseudo-label YT** → close both data gaps + rare casts.
3. **Short imagination horizon (H≈8)** where the dream is still 16–22 dB.
4. **Cheap parallel wins:** HUD-to-policy (now quantitatively justified), cast calibration.
5. **Tokenizer: defer.** If revisited later, a JEPA-style latent objective, not just more pixels.

**One line:** the tokenizer blurs fine HUD detail (real, but fixable by feeding the HUD to the
policy) yet is not the bottleneck; the world model is — it's plateaued 9 dB short of what the
tokenizer already gives, and the untried levers are action-conditioned mixed training + a
label-supervision prior + IDM data, not more epochs.
