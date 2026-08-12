# Third-party review: ahriuwu (Garen laning agent) — 2026-08-02

*Independent senior-ML review from a neutral fact-brief (no recommendations included in the brief).*

## 1. Top critiques

**1a. The build order is inverted: there is no closed loop and no environment in which the goal is even measurable.** The single most important artifact — capture → encode → policy → inject on the Windows box — does not exist. Everything measured so far is an offline proxy, and the project's own history shows its offline proxies are unreliable (two automated dream metrics rated a poisoned model above the clean one). A loop-first build (even driving a random or scripted policy) would have surfaced, months ago: real capture latency, live-rig encode speed, input-injection feasibility under Vanguard, and the replay-renderer-vs-live-client domain shift that the frozen tokenizer has never been tested against. Related and unaddressed: "Platinum" implies ranked ladder, which means botting live games — a ToS violation, a kernel-anticheat risk, and an eval you can't run repeatedly anyway. The goal as stated is currently unfalsifiable.

**1b. Perception is structurally broken at the tokenizer, and the mitigation cannot fix it.** A 352×352 downsample folded to a 16×16 grid gives ~22px cells; an on-screen HP bar is a few pixels tall and a rounding error in a reconstruction loss — the probe result (MLP R² 0.16 on champion HP, level lost entirely) was predictable from first principles, and minion HP bars are smaller still. This is not a side issue: last-hitting is the core laning mechanic, its reward trigger is minion HP crossing a damage threshold, and that variable is almost certainly illegible in the latents. That silently caps BC (the policy cannot time last-hits it cannot see), decouples the reward head from causality (AUC 0.956 is probably keyed on death animations / after-the-fact correlates), and makes the dream simulator blind to the thing the reward depends on. The frozen-tokenizer decision has hardened into an architecture constraint the whole stack is contorting around.

**1c. The action representation contradicts the known data-generating process, and the BC numbers are the symptom.** The human emits 2–5 discrete events/second; it is modeled as 20Hz per-frame held actions, so 77% of the gradient teaches "copy the last action." The measured results are exactly what that predicts: below-repeat-last accuracy, worse-than-freeze MAE on transitions, and a self-fed collapse to 1.8% that is textbook causal confusion — the policy learned to condition on its own action history as the answer key. The claim that live play is safe because "history stays consistent" is wrong in the direction that matters: a history-copying policy that drifts into a bad action is conditioned to *persist* in it. The Bernoulli-ability thresholding saga is the same disease in a second location.

**1d. Phase-3 is over-engineered relative to its evidence base.** Full PMPO + λ-returns + twohot value + factorized KL machinery was built before either prerequisite was demonstrated: (i) dreams respond causally to policy actions, and (ii) the reward head fires coherently *inside dreams*. The shortcut fine-tune — which DreamerV4 uses precisely because imagination at K=64 is unaffordable — is filed under "later." Phase-3 *is* dreaming; at K=64, H=8–10, O(H²) on one 5080 you get a few thousand gradient steps of RL against a 0.5s-coherent simulator. That is noise injection, not policy improvement.

**1e. Data strategy: 49h labeled, 450h discarded over a fixable artifact.** The paper recipe is ~25:1 unlabeled:labeled; you're at 0:1 after discarding 90% of the corpus over a black HUD rectangle — a preprocessing problem (mask or crop the HUD region; HUD-mask machinery already exists in the repo). The "no eval gain" justification leans on evals admitted to be weak. Under the current budget the exclusion is moot in practice, but it's recorded as a principled decision when it's actually a deferred bug.

**What I'd have done differently, one line each:** wire the live loop before training anything; read scalar state (HP/gold/level, own *and* enemy) from pixels/API/labels as a direct side-channel input from day one; model actions as events; crop the HUD out of *all* frames so replays and YT unify; treat imagination RL as a stretch goal gated on demonstrated dream controllability.

Fairness note: the measurement hygiene — held-out protocols, repeat-last baselines, self-fed evals, human-verified recons — is well above solo-project norm. The instruments are good; the conclusions drawn from them are lagging the readings.

## 2. Specific design choices

**(a) Per-frame Bernoulli abilities + greedy/threshold.** Wrong on principle twice over. Greedy decoding of a 0.1–1% base-rate Bernoulli never fires by construction; a global threshold trades never-fires for 50× over-fires. The right model is event-level: a hazard head, P(cast event in next window | state), with the ability as a conditional categorical, decoded by *sampling* (or argmax over a short window with refractory suppression). Per-frame categorical over {no-op, Q, W, E, R, …} with sampling is an acceptable cheap middle ground. Cast *timing* error vs the human event is the metric, not per-frame accuracy.

**(b) Frame-level 20fps cursor BC.** Indefensible given the measurements. Restructure as gate + location: per frame predict P(new movement command) and, conditional on a command, the cursor bin; train the location loss only on transition frames (or heavily upweighted); drop or corrupt the own-action-history input (scheduled dropout) to kill the copy shortcut. Success = transition MAE beating the freeze baseline and self-fed within shouting distance of teacher-forced.

**(c) Aux state head.** Right instinct, wrong layer. A supervised head over a frozen backbone and frozen tokenizer cannot recover information the tokenizer destroyed — the probe R² 0.16 is approximately an upper bound for enemy HP. The correct move: scalar state as a *direct input* — own stats from labels/Live Client API, enemy HP from a trivially small supervised reader on the HP-bar pixel crop (reading a rendered UI element; should hit R² > 0.95). Concatenate as tokens into the agent blocks. Keep the aux head only as a diagnostic.

**(d) Imagination RL on 0.5s dreams with value bootstrap.** As specced, cannot work. At H=8–10 with γ=0.997, ~97% of the λ-return is the bootstrapped value — trained inside the same 0.5s dreams, from a reward head with R² 0.06 magnitude calibration, on a simulator that cannot represent minion HP and has never demonstrated causal action response. The death penalty will essentially never materialize inside a 10-frame dream. Two cheap gates before any GPU-hour: (i) counterfactual controllability — same context, "move left" vs "move right", measure decoded champion displacement divergence by h=8; (ii) reward-in-dream sanity. If either fails, Phase-3 is off. If run, the shortcut fine-tune is a prerequisite.

**(e) Excluding 450h YouTube.** Wrong as a permanent decision, accidentally right this quarter. Contamination is preprocessing-fixable (HUD masking/cropping, which also shrinks train/live domain shift). Keep excluded *for now*; fix the HUD at the data level when next touching tokenizer/dynamics; stop citing the mixed-retrain eval as evidence.

## 3. Highest-leverage plan (ordered, with gates)

1. **Close the loop** (1–2 weeks, $0): Windows capture → encode → current BC policy → inject; per-stage latency on the live rig; decide the eval environment (Practice Tool + customs, not ladder). *Gate:* 20fps sustained, photon-to-click < 150ms, 10 min crash-free.
2. **Perception side-channel** ($0, days): pixel-crop HP-bar readers (own + enemy + lowest-minion-in-range) trained on labels; scalars as input tokens. *Gate:* enemy-HP R² > 0.9; "dies to one AA/Q" classification > 0.85.
3. **Action-model rewrite + BC retrain** (biggest visible skill gain): gate+location movement, hazard casts with sampling, action-history dropout, transition-weighted loss. *Gate:* transition MAE < freeze baseline; cast-timing F1 within ±150ms at human-comparable rates; self-fed > 60% of teacher-forced.
4. **Mechanics benchmark in-game**: Practice Tool free-farm CS 2:00–10:00; then vs Intermediate bot. *Gate:* ≥60% of available CS (→80% = the Platinum-mechanics bar); vs bot CS@10 ≥ 55, deaths ≤ 1.
5. **Offline advantage-weighted BC instead of imagination**: use the reward head on *real* latents (where it's validated) for AWR/IQL-style filtered BC. No dreaming. *Gate:* beats plain BC on the Step-4 benchmarks.
6. **Phase-3 only if the 2(d) gates pass**, after the shortcut fine-tune. Bonus, not plan.

## 4. Milestone ladder (replaces rank-talk)

- **L0 System:** live loop, 20fps, <150ms, 30-min session, zero crashes.
- **L1 Mechanics (Practice Tool):** free-farm CS ≥ 60% → 80% → 90%; cast rates within 2× of human statistics.
- **L2 Vs bot laner:** CS@10 ≥ 55 → 70, deaths ≤ 1, positive gold diff@10.
- **L3 Vs humans (arranged customs, laning-only):** positive gold+XP diff@10 vs Silver, then Gold, then Plat volunteers, Bo5 lanes each. *This* is the honest operationalization of "Platinum-level laning."
- Component gates underneath: transition-MAE vs freeze, self-fed retention, cast-timing F1, dream counterfactual divergence, reward-in-dream AUC. Nothing graduates on offline metrics alone.

## 5. Biggest risk + cheapest test

**Risk:** the pixels→latent pipeline cannot see minion and champion HP — the load-bearing state of laning. Kills the goal through three channels (BC, reward causality, dream blindness); no downstream RL polish repairs it.

**Cheapest test (one afternoon, $0):** (i) recon-montage on minion-wave frames — are minion HP bars legible by eye? (ii) pixel-crop lowest-minion-HP probe vs the same probe on v7 latents; (iii) current BC agent in Practice Tool, 10 min free farm, count CS. If the latent probe is junk while the pixel-crop probe is strong: route scalar state around the tokenizer. If free-farm CS < ~40% even with that fix: the problem is deeper than perception; revisit before building more architecture.

---

**Bottom line:** the world-model framing is defensible and the measurement discipline is genuinely good, but the project is polishing its most speculative component (imagination RL) while its two actually-binding constraints — an action representation that contradicts the human event process, and a perception stack blind to HP — sit measured, documented, and unfixed, and the loop that would prove any of it live has never been closed. Fix those three in that order; the Dreamer machinery can wait.
