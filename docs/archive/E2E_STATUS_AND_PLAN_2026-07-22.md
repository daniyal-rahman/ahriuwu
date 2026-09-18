# Garen agent — e2e status, what's weak, and how to fix it (2026-07-22)

## 1. Do we have a full e2e? YES — and here's the working test

Simulated end-to-end (no live game): the trained BC policy driven frame-by-frame over a real replay's latents, its predicted NEXT action scored against the **logged human action** (`scripts/eval_bc_sim.py`). Full path exercised: **v7 tokenizer-latents → dynamics(agent tokens) → PolicyHead → action**, on the 5080 in bf16 at **26 fps** (inside the 20fps budget).

| Component | Metric (greedy, 800 frames, replay NA1_5549995114) | Verdict |
|---|---|---|
| **Movement** | **bin-accuracy 71.1%** (random ~0.2%), MAE 0.018 < center-baseline 0.032, MAE\|moved 0.028 | **GOOD — real imitation** |
| **Abilities** | **F1 = 0.00**; greedy never casts; temp=1 casts at ~marginal rate but 0 temporal alignment with human | **BAD — the weak link** |
| Speed | 26 fps bf16 ctx=16 | fits budget |

**So: the pipeline is real and working.** The agent moves like the human 71% of the time (bin-exact). The one clearly-broken component is **casting/abilities**.

(Separately, the *world model as a dreamer* is weak — rollout < persistence — but that's irrelevant to this demo, which uses the WM only as a frame *encoder*. It matters only for Phase-3 imagination.)

## 2. My best guesses (before deeper tests) — why casting is bad, ranked

1. **Sparse-action collapse (primary).** Casts are ~1 in hundreds of frames. Even with `pos_weight=5`, BCE learns the *marginal cast rate*, not the *state-conditional "cast now."* The dense movement signal dominates the gradient. → the head predicts near-base-rate everywhere.
2. **Greedy threshold destroys a weak-but-present signal.** At temp=0 the sigmoid rarely clears 0.5 for rare abilities → *never casts*. The signal may live in the **logit spikes** near real casts; the fixed 0.5 threshold throws it away. (Testable — see §3.2.)
3. **Cast-data scarcity.** 125 replays, mostly laning; few, low-diversity cast examples. Not enough for BC to learn the visual precondition of a cast (enemy in range, ability off cooldown).
4. **No value signal on casting.** BC only imitates; it never learns "cast → gold/kill." The reward head exists but doesn't shape the policy (that's Phase-3's job, currently blocked).
5. **The policy can't see cooldowns/mana.** Those live in the HUD — which we *mask out* for the world model. The policy has no idea what it's *allowed* to cast. (Big, under-appreciated — see §6.5.)

**Why movement is good:** it's dense (every frame has a target), smooth, and balanced in the loss → BC learns it cleanly. This is the classic imitation pattern: dense-continuous easy, sparse-discrete hard.

## 3. Extensive testing plan — find + confirm what's weak

Run in order; each is cheap on the 5080.

1. **Held-out generalization** (the honest number): run `eval_bc_sim` on games **not in BC training**. Gate: movement bin-acc should stay ~>50% (else it's overfitting, not learning). Confirms the split is real.
2. **Ability calibration probe** *(the key isolator)*: for the 9 abilities, plot the raw logits in a ±window around true-cast frames vs elsewhere. Two outcomes:
   - logits **spike** near real casts → **signal is there, threshold/calibration is the bug** → cheap fix (per-ability threshold tuned on val, or temperature to match the human cast rate).
   - logits **flat** → the head **never learned cast timing** → needs more data / better loss.
3. **Movement deep-dive**: bin-acc stratified by move magnitude, error-vs-frame (does it drift late-game?), per-game variance.
4. **Encoder ablation**: BC-eval with the Step 1′ encoder vs the action-conditioned 135 encoder → does the encoder change move/ability accuracy? Isolates encoder contribution.
5. **Context-length ablation**: eval at ctx 8/16/32 → does more history improve *cast timing* (casts are event-triggered, may need more context)?
6. **Cast-count census**: count Q/W/E/R/etc. presses across all 125 replays. Quantifies the scarcity (if < ~1–2k cast events, data is the wall).
7. **World-model rollout** (Phase-3 readiness only): horizon-PSNR vs persistence (already known bad → Phase-3 blocked until the WM dreams).

## 4. Isolation methodology (why the bad part is bad)

- **Ability = calibration vs no-signal**: §3.2 is decisive. Spiking logits ⇒ threshold/loss fix (hours). Flat logits ⇒ data/architecture fix (weeks).
- **Then quantify the cause**: §3.6 (data scarcity) + a `pos_weight`/focal-loss sweep on a short BC run isolates the *loss* contribution; §3.4 isolates the *encoder*; §3.5 isolates *context*.
- **Movement (if held-out drops)**: §3.1 separates overfitting from genuine learning.

## 5. Roadmap review (`lab_notebook/EXPERIMENTS_AND_ROADMAP.md`) — what actually helps casting

- **1.C Inverse Dynamics Model (IDM) pipeline — THE relevant one.** Train an IDM on the 125 labeled replays (frame_t, frame_t+1 → action), pseudo-label the **906 YT games**, giving 10–50× more (incl. cast) examples. This *directly* attacks the cast-scarcity root cause (#3). Highest-leverage roadmap item for this problem. Caveat: the IDM must detect **cast animations** visually, which is exactly the sparse/hard bit — validate IDM cast-recall on held-out replay before trusting YT pseudo-labels.
- **3.L Policy distillation from pros** — same family (more/better action labels).
- **1.A Mamba hybrid / 1.D ProMAG / 1.B MeanFlow / 2.x** — these improve the **world model's rollout/efficiency**, i.e. they unblock **Phase-3 imagination**, which would then reward-shape casting. Indirect: they help casting *only* by making the dreamer usable. Not the fast path.
- **0.1 variable τ_ctx** — rollout stability; matters for Phase-3, not the encoder path.

**Net:** of the roadmap, only **1.C (IDM)** attacks the casting problem head-on. The rest route through fixing the world-model dreamer (Phase-3), which is the longer road.

## 6. My own ideas (beyond the roadmap) — cheapest first

1. **Focal loss + calibrated inference threshold.** Swap ability BCE→focal loss (down-weights the flood of easy negatives, focuses on rare positives). At inference, drop the 0.5 threshold — tune a per-ability threshold on val to maximize F1, or set the temperature so the agent's cast *rate* matches the human's. If §3.2 shows spiking logits, this alone likely unlocks casting. **~a day.**
2. **Decompose casting into gate + selector.** A dense binary "**is a cast imminent?**" head (learns the rare-event *timing* with the full frame budget) + a "**which ability**" head trained *only on cast frames* (concentrates the sparse signal). Turns one impossible 9-way-sparse problem into two learnable ones.
3. **Oversample cast frames in BC.** Weight/oversample cast frames (and a short pre-cast window) so each epoch sees far more cast examples from the *same* data. Cheap, directly fights scarcity. **~hours.**
4. **Reward-weighted BC (poor-man's RL).** Weight the BC loss by the reward/advantage (upweight actions that preceded gold/kills). A cheap stand-in for Phase-3's value signal that biases toward *good* casts — bridges BC→RL without the (blocked) imagination loop.
5. **Feed the HUD to the *policy* (not the world model).** We mask the HUD for the WM, but the policy needs it — cooldowns, mana, ability availability are all there. Add a small HUD-crop (or the parsed cooldown/mana from labels) as an extra policy input. The agent currently casts blind to what's off cooldown; this could be the single biggest casting unlock. **~1–2 days.**
6. **Cast-conditioned context probe → then longer context for casting only.** If §3.5 shows casts need more history, run the policy at ctx=32 for the *cast* head and ctx=16 for movement (movement is fine short; casting may need the setup).
7. **Action-space audit for AA.** The temp=1 run showed the agent AA-ing 17× vs human 9× but 0 alignment — auto-attack (right-click) timing is its own sub-problem; consider modeling AA as part of the movement/right-click stream rather than a discrete ability.

## 7. Recommended sequence
1. §3.2 calibration probe (decides everything about casting — hours).
2. If logits spike → idea #1 (focal + calibrated threshold) + #5 (HUD to policy). Cheap, high-upside.
3. If logits flat → idea #3 (oversample) + roadmap **1.C IDM** (more cast data), then re-probe.
4. In parallel, §3.1 held-out eval to lock in the honest movement number.
5. Longer game: fix the world-model dreamer (1.A/1.D) → Phase-3 imagination → reward-shaped casting (the "great agent" path).

**One-liner:** *Full e2e works — movement imitation is genuinely good (71% bin-exact), casting is the weak link (never casts at greedy, uncorrelated when sampled). Best guess: sparse-action collapse + a 0.5-threshold that kills a weak signal + the policy being blind to cooldowns (HUD masked). The decisive test is the ability-logit calibration probe; the cheap fixes are focal-loss + calibrated threshold + feeding the HUD to the policy; the data fix is the IDM pipeline (roadmap 1.C).*
