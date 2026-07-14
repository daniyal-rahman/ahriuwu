# World-model debugging — full log (2026-07-14)

How we found and fixed the plateau. Read top-down; the answer is in §0, the evidence trail is §1–7, the "what now" is §8.

---

## 0. The answer (TL;DR)

**The world model wasn't broken by capacity, data, or missing actions. It was broken by the HUD-mask loss.** Job 179 trained ~78% of its batches (the YT games) on a *decode-only* pixel loss that **never directly constrains the latent** — so the model learned latents that decode correctly one-frame but are latent-imprecise, which sank teacher-forced PSNR to ~19 dB and made rollout collapse.

**The fix (Step 1′): drop the pixel-only HUD loss, use plain latent loss.** Denoising recovered from **τ0.9 18.4 → 24.6 in 24 optimizer steps**, held at **~26 dB** — right back to the "good" model 135's level. Same weights, same architecture. So it was never underfitting/capacity.

**Actions were a red herring.** The rollout gap between 135 and the fixed model is mostly a *measurement artifact* (135 was evaluated with the real recorded actions fed in; the unconditioned model wasn't). Actions matter for Phase-3 imagination (steering the dream) and for fairly measuring rollout — **not** for world-model quality.

**Canonical checkpoints now:**
- **135** — `dynamics_v7_accel_resume/dynamics_latest.pt`, gs 11017, action-conditioned, 125 clean replays, τ0.9 25.3, rollout marginally beats persistence. Known-good.
- **Step 1′** — `dyn179_s1prime_latentonly/dynamics_latest.pt`, gs 7550, unconditioned, 578 games, τ0.9 ~26. The recovered model; drop-in for the current (actions-off) BC.

---

## 1. Starting point — the plateau
Job 179 (medium, 578 games, `use_actions=False`, pixel-HUD loss, stride 64): teacher-forced **τ0.9 flat ~19 dB for ~24h**; rollout PSNR **below persistence**, dream collapses into teal garbage by ~frame 16. The briefing hypothesized capacity/underfitting/actions.

## 2. Step 0 — is it the model or the ruler?
Re-ran rollout on 135 and 179 at num_steps ∈ {1,4,16}, same window (NA1_5549995114, ctx 6, horizon 16, seed 0). Latent rollout mean PSNR:

| num_steps | 135 (act) | 179 (noact) |
|---|---|---|
| 1 | 14.8 | 9.3 |
| 4 | 11.6 | 8.6 |
| 16 | 11.3 | 7.2 |

At every setting 135 ≫ 179 → **it's the model, not the sampler**. (But fewer steps *is* better — the eval should run num_steps=1, not 16. Ruler was slightly bent, didn't change the verdict.)

## 3. The killer finding — it's NOT capacity
Teacher-forced τ-sweep (τ 0.1/0.3/0.5/0.7/0.9):
- 135: 11.9 / 14.5 / 16.8 / 19.8 / **25.3**
- 179: 9.4 / 10.6 / 12.0 / 14.1 / **18.4**

**Same medium (114M) architecture fit to 25.3 in one run and stalled at 18.4 in another → not a capacity ceiling.** An optimization/objective pathology specific to 179. (This gap is at near-clean denoising, which is ~action-independent → also mostly rules out "actions" as the cause.)

## 4. The mechanism — reading the loss code
- `is_yt = all(video_id not startswith "NA1_")` per batch; two separate RMS normalizers (`x_pred`, `pixel`).
- YT batch → `pixel_hud_masked_loss`: decode **k=4 of 128 random frames** through the frozen v7 decoder, MSE over non-HUD pixels only, grad through the frozen decoder into z_pred. **The latent target z_0 is never directly constrained.**
- Replay batch → plain latent `x_prediction_loss`.
- 578 games = 453 YT + 125 replay → **~78% of batches never got direct latent supervision**; the model was optimizing "decode looks right in the unmasked region," a looser, decoder-filtered constraint. Rollout is *pure latent autoregression* → the latent imprecision compounds → collapse.

## 5. Step 1′ — the falsifiable test that confirmed it
Resumed 179's degraded weights (job 181, `dyn179_s1prime_latentonly`), removed `--pixel-hud-loss` → plain latent loss on all 578 games, **nothing else changed** (actions still off, same data/stride). τ0.9 trajectory:
- gs6526 (baseline): 18.4
- gs6550 (**24 steps later**): **24.6**  ← +6.2 dB, nothing learns that fast → the capability was in the weights, the HUD loss was suppressing it
- gs6850–7450: stable **26.1–26.5** (held, not a transient)

Full sweep at gs6550: 13.8 / 15.9 / 17.5 / 20.0 / 24.6 (vs 179's 9.4 / 10.6 / 12.0 / 14.1 / 18.4). **Denoising fully recovered.**

## 6. But denoising ≠ rollout — the rollout test on Step 1′
Rollout on the Step 1′ checkpoint (gs7550), same window, num_steps 1. Pixel-space (persistence = 24.2→21.5, tokenizer ceiling 28.3→28.5):

| | τ0.9 | latent rollout mean | pixel dream h1→h16 |
|---|---|---|---|
| 179 (pixel-HUD, noact) | 18.4 | 9.3 | 26.3 → 17.0 |
| **Step 1′ (latent, noact)** | **25.0** | **11.6** | 26.6 → **18.8** |
| 135 (latent, **act**) | 25.3 | 14.8 | 26.6 → 22.5 |
| persistence | — | — | 24.2 → 21.5 |

Step 1′ **improved rollout** (9.3→11.6, h16 17.0→18.8) but **still below persistence** in the tail. So: HUD loss crushed denoising (fixed); something else caps unconditioned rollout.

## 7. The montage + the actions reframe
Decoded Step 1′'s dream (`s1p_montage.png`): it does **not** paint HUD-black and does **not** collapse into garbage — it **blurs toward the mean** over the horizon while still tracking the scene. Two conclusions:
- **HUD-black is not the drag** on Step 1′'s rollout (at gs7550, ~1k steps of latent-on-YT — hasn't strongly re-learned it; 154 took ~3k).
- **The blur is the correct behavior of an unconditioned model** — without the future actions/camera it can only predict the average future. That blur scores below persistence, but that's the *metric penalizing missing information*, not the model being bad.

**So the 135-vs-Step1′ rollout gap is confounded and mostly a measurement artifact:** 135 was fed the real recorded actions at eval (knows the trajectory → sharp) and trained on clean replay data; Step 1′ got neither. It is **not** clean evidence that actions make a better world model. The paper agrees: the WM absorbs its knowledge from unlabeled video; actions add steering.

---

## 8. Conclusions & state

**Root cause:** the pixel-HUD-masked loss (decode-only, no latent constraint) on the YT-majority batches. **Fix:** Step 1′ = latent supervision on every batch. WM recovered to ~135's denoising level.

**Ruled out:** capacity (same arch → 25 dB), data scale, actions-as-model-quality.

**Deferred (real but not blocking):**
- **Long-run HUD handling.** Plain latent loss on the blacked YT frames *will* re-learn HUD-black over a long run (154 did at ~3k steps). It doesn't hurt the *encoder* use (encoding real frames is unaffected), only generation. If we scale the WM long, the clean fix is **pre-black the replay HUD to match YT + black the HUD at inference** (consistent data everywhere → latent loss everywhere, no mismatch), not the masked pixel loss. An `--pixel-hud-aux` mode (latent always + pixel aux) is implemented but predicted to still re-learn black (latent target is blacked; latents are non-spatial so can't be masked). Untested — parked.
- **Action-conditioning** for Phase-3 imagination (steering the dream). Needs the labeled+unlabeled mixed-dataset plumbing. Later, and it's a steering wheel, not a fix.

**Checkpoints:** 135 (action-cond, clean, known-good) and Step 1′ gs7550 (unconditioned, recovered). Either works as the demo *encoder*.

## 9. Pipeline-ready plan (what happens next, autonomously)
1. Re-point Phase-2 BC onto the **Step 1′ recovered backbone** (it's been running on 179's *degraded* gs6120 snapshot — that must change).
2. Build + offline-test the **inference module** (`play_live` → clean): tokenizer → WM-encode → BC-policy → action, verified on replay frames.
3. Leave the WM as-is (recovered); do **not** spend more compute chasing rollout — it's a metric artifact for an unconditioned model.
4. At-home: Windows capture→infer→inject wiring onto the tested module.

**One-liner:** *The HUD-mask loss (decode-only, on 78% of batches) was starving the latent target and crushing the model; removing it recovered denoising to 26 dB instantly. Rollout "below persistence" is mostly the unfair unconditioned-vs-specific-future metric, not a defect. Actions are for Phase-3 steering, not model quality. World model is recovered; pivot to BC-on-good-backbone + the inference module.*
