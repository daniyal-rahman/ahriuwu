# Architecture literature review for lanerl_jax (2026-09-23)

Scope: the design questions in `docs/ROADMAP_CHAMPIONS.md` item 6, checked against what the top systems published. The baseline is the current stack. It has an entity-attention encoder over 32 slots and a 4x1024 MLP core with no memory. The `PolicyConfig.frame_stack = 4` field exists but nothing reads it. There are four independent heads (button 8, screen_x 96, screen_y 54, pointer target 32), and log-prob and entropy are gated by head usage. Training is dual-clip PPO inside one jit (Anakin) with mirror self-play and a zero-sum reward. There is no return normalisation, the value loss is clipped at 0.2 in raw reward units, and the decision rate is 30 Hz over 600 s (18,000-step) episodes.

Every factual claim below cites a primary source. Where I am reasoning rather than reporting, the text says so.

---

## 1. Memory core

**What the top systems actually ran.** None of the flagship game-playing RL agents used a transformer as the memory over time. Where transformers appear, they encode entities or sets within a single timestep.

| System | Temporal core | Transformer used for |
|---|---|---|
| AlphaStar ([Vinyals et al., Nature 2019](https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf)) | "deep LSTM" (3x384 per the released detailed architecture, summarised in [mini-AlphaStar, 2021](https://arxiv.org/abs/2104.06890)) | entity encoder (self-attention over units) |
| OpenAI Five ([Berner et al., 2019](https://arxiv.org/abs/1912.06680)) | single-layer **4096**-unit LSTM, 84% of parameters. The 1024 figure is the June 2018 version ([OpenAI blog, 2018](https://openai.com/index/openai-five/)); it grew 2048 to 4096 by surgery | none (max-pool over units) |
| JueWu / Tencent Solo 1v1 ([Ye et al., AAAI 2020](https://arxiv.org/abs/1912.09729)) and full game ([Ye et al., NeurIPS 2020](https://arxiv.org/abs/2011.12692)) | LSTM ("for learning skill combos"); ablation: +LSTM alone gave a 73% win rate vs base | target attention |
| R2D2 / Agent57 ([Kapturowski et al., 2019](https://openreview.net/forum?id=r1lyTjAqYX); [Badia et al., 2020](https://arxiv.org/abs/2003.13350)) | LSTM with stored state + burn-in | none |
| GT Sophy ([Wurman et al., Nature 2022](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/nature22.pdf)) | **none**: 4x2048 MLP on near-full state | none |
| GT7 vision agent ([Sony AI, 2025](https://arxiv.org/abs/2504.09021)) | **recurrent** actor added once the observation became partial (vision), plus an asymmetric critic | none |
| DeepNash ([Perolat et al., Science 2022](https://arxiv.org/abs/2206.15378)) | **none**: U-Net torso over a hand-built history tensor (last 40 moves, 82 stacked frames) | none |
| DreamerV3 ([Hafner et al., 2023](https://arxiv.org/abs/2301.04104)) | GRU-based RSSM | none |
| VPT ([Baker et al., 2022](https://arxiv.org/abs/2206.11795)) | transformer over a 128-frame context | memory (BC-pretrained, then RL fine-tuned) |
| AdA ([Adaptive Agents Team, 2023](https://arxiv.org/abs/2301.07608)) | Transformer-XL | memory across multi-trial meta-RL episodes |

The two transformer-over-time agents (VPT, AdA) were either BC-pretrained on huge datasets or needed cross-episode memory for meta-RL. Neither is a real-time competitive agent trained from on-policy RL.

**Recent evidence (2022-2026).** GTrXL ([Parisotto et al., 2019](https://arxiv.org/abs/1910.06764)) did make transformers trainable in RL and beat LSTMs on memory-heavy DMLab. On a broad POMDP suite, however, POPGym found that "the GRU is the best general-purpose memory model" ([Morad et al., ICLR 2023](https://arxiv.org/abs/2303.01859)). [Ni et al. (NeurIPS 2023)](https://arxiv.org/abs/2307.03864) separate memory from credit assignment. Transformers extend *memory* (up to 1500 steps back) but "do not improve long-term credit assignment". The transformer-in-RL survey ([Li et al., TMLR 2023](https://arxiv.org/abs/2301.03044)) lists non-stationarity, design sensitivity and compute cost as the obstacles in online RL. Our open problem is credit assignment: CS falls away from the BC prior, and the reward horizon is long at 30 Hz. The evidence says a transformer core does not address that.

**Autonomous driving.** The owner's premise is only half right. Transformers dominate the *imitation and forecasting* models: Waymo's Wayformer ([2022](https://arxiv.org/abs/2207.05844)) and MotionLM ([2023](https://arxiv.org/abs/2309.16534)) are supervised motion forecasters, and MotionLM decodes motion tokens autoregressively. EMMA ([Waymo, 2024](https://arxiv.org/abs/2410.23262)) is a supervised Gemini-based model. Wayve's GAIA-1 ([2023](https://arxiv.org/abs/2309.17080)) is a generative world model. NVIDIA's Alpamayo-R1 ([2025](https://arxiv.org/abs/2511.00088)) is a VLA trained with SFT and then RL post-training. Tesla has published nothing primary. Secondary reporting describes FSD v12 as end-to-end nets trained on human video, with later release notes mentioning an RL stage ([thinkautonomous, 2024](https://www.thinkautonomous.ai/blog/tesla-end-to-end-deep-learning/)), and that is unverifiable. Where driving uses RL, the architectures are small. Waymo's BC-SAC combined imitation and RL for a 38% failure reduction on hard scenarios ([Lu et al., 2022](https://arxiv.org/abs/2212.11419)). The strongest pure-RL driving result, GIGAFLOW (1.6 billion km of self-play, SOTA on three benchmarks, no human data), uses a **6M-parameter feed-forward Deep-Sets policy** ([Cusumano-Towner et al., 2025](https://arxiv.org/abs/2502.03349)). So in AV, "transformer" usually means a large supervised sequence model, not a recurrent core for RL.

**Why memory matters here specifically.** The builder docstring (`lanerl_jax/obs/builder.py`) says the velocity and hp-delta fields were removed because they are things "a GRU exists to compute". No GRU exists. The policy sees each minion's `hp_frac` now, not how fast it is falling. That rate is the main last-hit signal. Dactyl measured the same trade-off: the LSTM policy beat a feed-forward one under domain randomisation, and its hidden state predicted the randomised physics ([OpenAI, 2018](https://arxiv.org/abs/1808.00177)). That implicit system identification is what sim-to-server transfer needs.

A naive frame stack is wrong for this observation. Slots are ordered by distance, so slot *k* at *t-1* is not the same unit as slot *k* at *t*.

> **Recommendation for lanerl_jax.** (a) Add a **GRU (512)** between the MLP trunk and the heads. Carry `h` in `RunnerState` and store `h_0` at each rollout start. In the update, re-unroll the 128-step rollout from the stored `h_0` (truncated BPTT over 4.3 s, which is Five's scheme at 16 steps). Minibatch over the **env axis**, never over time. Reset `h` to zero only where `done` fires, i.e. at episode boundaries. Do not reset on death: respawn and wave timing are exactly what memory should carry. The forward state carries information across all 18,000 steps; only the *gradient* is truncated. If staleness across the 4 epochs matters, add R2D2-style burn-in (e.g. 32 steps). (b) Independently, and first, restore **per-unit deltas in the builder, keyed by unit index**: hp change per tick and velocity for each visible unit. This is cheap, it is stable under slot reordering, and it is what DeepNash and GT Sophy did instead of recurrence. (c) Delete the dead `frame_stack` field. (d) Do **not** build a transformer-over-time now. Revisit it as a GTrXL ablation only if a probe shows a need for memory beyond ~10 s that the GRU fails. **Cost: (b) small, (a) medium.**

---

## 2. Autoregressive single action space

**What the top systems did.**

- **AlphaStar** is fully autoregressive: action type, then delay, queued, selected units (a recurrent pointer network), and finally target unit or location. Each head consumes an "autoregressive embedding" updated by the previous choice ([Vinyals et al., 2019](https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf); order per [mini-AlphaStar, 2021](https://arxiv.org/abs/2104.06890)).
- **OpenAI Five** was *partly* autoregressive. The primary action is a dot product over the embeddings of the *available* actions. Only the target unit is conditioned on it, via "a learned per-action mask based on the sampled action" over unit keys. **Offset and delay are plain linear projections of the LSTM state, not conditioned.** Heads the action ignores are masked out of the loss "since their gradients would be pure noise" ([Berner et al., 2019](https://arxiv.org/abs/1912.06680), Fig. 18 and Appx F). Our `PPO-01` fix reproduces exactly that last point.
- **JueWu 1v1**, the closest published analogue to our task (MOBA 1v1, pro level), went the *other way*. "Control dependency decoupling" treats every label independently in a multi-label PPO objective, and puts button-argument correlations into an action mask built from game knowledge ([Ye et al., 2020](https://arxiv.org/abs/1912.09729)). The full-game system later used hierarchical what→who→how heads ([Ye et al., 2020b](https://arxiv.org/abs/2011.12692)).
- **Single-token-stream approaches** such as Gato ([Reed et al., 2022](https://arxiv.org/abs/2205.06175)), Decision Transformer ([Chen et al., 2021](https://arxiv.org/abs/2106.01345)) and action chunking ([ACT, Zhao et al., 2023](https://arxiv.org/abs/2304.13705)) are all *supervised*. Chunking fixes compounding BC error at low demo counts. None is evidence for online RL at 30 Hz.

**Honest reading.** Full autoregression is not required to reach pro level in MOBA 1v1: JueWu did not use it. What it buys *us* is concrete, though. The x/y factorisation cannot express a joint mask; `actions.py` currently decodes minimap-rectangle clicks to NOOP after sampling because of that. The target can be made to depend on the button (R should point only at the enemy champion). It also gives a clean place to hang per-champion argument semantics (section 4).

> **Recommendation for lanerl_jax.** Replace the four heads with one decoder, ordered **button → target (pointer) → screen_x → screen_y**. The first factor is the verb. Target comes second because for attack-move the chosen unit decides whether the point is used at all. y is conditioned on x so the joint (x, y) mask becomes expressible. Conditioning follows AlphaStar: `z_1 = h + E_button[b]`, then `z_2 = z_1 + W·token[t]`, where the chosen entity token is projected, and then `z_3 = z_2 + E_x[x]`. **Keep the pointer target head**: it is what keeps the target permutation-equivariant (Five and AlphaStar both point). The log-prob stays `Σ_i uses_i(b)·log π(a_i | a_<i)`, exactly the current gating. For entropy, compute the button level exactly: evaluate the next factor's logits for all 8 buttons in one batched call, which is cheap, giving `H(b) + Σ_b p(b)·uses(b)·H(next|b)`. Estimate deeper levels on the sampled path, which is unbiased for the expected conditional entropy (my derivation, not from a paper). The `MAX_FACTORED_ENTROPY` bookkeeping must be rewritten to match. Do this **when the second champion lands**, not before: for Garen alone the gain in CS is probably small. **Cost: medium.**

---

## 3. Invalid action masking

**Evidence.** Masking logits before the softmax gives a valid policy gradient and scales to large invalid-action spaces where penalties fail. Train-with-mask/deploy-without collapses: episodic return drops from 40 to 33.5 (4x4 map) and to 17.4 (24x24) ([Huang & Ontañón, 2020/FLAIRS 2022](https://arxiv.org/abs/2006.14171)). Five masks the primary action to the available set ([Berner et al., 2019](https://arxiv.org/abs/1912.06680)). JueWu's action mask cut time-to-converge from 80 h to 65 h at equal strength ([Ye et al., 2020](https://arxiv.org/abs/1912.09729)). For the transfer question, [Zabounidis et al. (2026)](https://arxiv.org/abs/2603.09090) identify *valid action suppression*: unmasked gradients at states where an action is invalid suppress it at states where it is valid. They propose an auxiliary **feasibility classifier**, trained with the mask on, that substitutes for the oracle mask at deployment. I found no paper that directly measures a *mismatched* availability rule between training and deployment. That exact question is unstudied.

> **Recommendation for lanerl_jax.** Mask Q/W/E/R where the **observation's** cooldown feature is `> 0`. `builder.py` already writes 1.0 for rank 0 or cast-locked and the fraction for on-cooldown, so availability is exactly `cd == 0`. The mask must be a function of the observation, not of sim state, so the deployed memory-read builder produces it identically and the mask transfers by construction. Three requirements come with it. (i) Apply the mask identically in rollout, loss, entropy and the KL-to-BC term, or the ratio and the KL go to inf. (ii) **Clean the BC labels**: humans press keys that are on cooldown. Map those labels to NOOP or drop them, or the BC log-likelihood of a masked action is −inf. (iii) For robustness to server-side disagreement, randomise a small rate of silently-failed casts in the sim so the policy has seen a press that did nothing. Add the feasibility head ("did the cast take effect") as an auxiliary loss. **Cost: small.**

---

## 4. Multi-champion single policy

**Evidence.**

- **Five** ran one network for 17 heroes. Each replica is told which hero it controls by appending that hero's unit embedding to the LSTM input. In early training, larger pools (40, 80) slowed learning "only slightly" ([Berner et al., 2019](https://arxiv.org/abs/1912.06680), Appx P). That is early training only.
- **AlphaStar** trained **one policy per race** ([Vinyals et al., 2019](https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf)).
- **JueWu full game** found that randomly mixing hero combinations causes "learning collapse". It notes that OpenAI's attempt at 25 heroes gave "unacceptably slow training". The fix was fixed-lineup teachers, then student-driven distillation, then merged training. The result was one shared model for 40 heroes ([Ye et al., 2020b](https://arxiv.org/abs/2011.12692)).
- **Honor of Kings Arena** shows that a policy trained on one hero-vs-hero pairing **does not generalise** to other opponents or to other controlled heroes. Multi-task training and distillation over 5 tasks only partly close the gap ([Wei et al., NeurIPS 2022](https://arxiv.org/abs/2209.08483)).
- **Outside MOBAs**, SIMA's multi-game BC agent beat environment specialists by 67% on average ([SIMA Team, 2024](https://arxiv.org/abs/2404.10179)). GIGAFLOW drives trucks to bicycles with one conditioned feed-forward policy ([2025](https://arxiv.org/abs/2502.03349)).

Summary: one conditioned policy is the right end state and has worked at 40 heroes. Zero-shot generalisation to champions the policy has not trained on is **not** supported. "KL on a handful, and the model generalises to the rest" will not come for free. Every champion needs RL experience, and the pool needs a curriculum.

> **Recommendation for lanerl_jax.** Use one policy with (a) a learned champion-ID embedding for self and for the opponent entity, plus (b) **per-ability descriptor features** for each slot: cast type (self / unit-targeted / direction skillshot / ground-targeted / dash), range, cooldown, cost, and damage scalings. Descriptors give the network a chance to share across champions; the ID alone cannot. Button slots stay Q/W/E/R. The **cast type of the chosen slot** (data, not a constant) decides which argument factors the autoregressive decoder emits. The constant `USES_SCREEN_HEADS`/`USES_TARGET_HEAD` tables become a per-champion `(n_champ, 8)` lookup. Skillshots and dashes use the screen point as a target location, read as a direction or destination relative to self. Add champions incrementally, 2 to 4 at a time. Check that CS on the existing champions does not regress. If it does, fall back to JueWu's per-champion teachers plus distillation. **Cost: large (sim work dominates), policy side medium.**

---

## 5. Human prior with a diminishing KL

**Evidence.**

- **AlphaStar** initialised from supervised learning *and* "continually minimise[d] the KL divergence between the supervised and current policy", plus z-statistic pseudo-rewards. The paper describes **no annealing** of that KL. Ablation (test Elo): no human data 149, supervised 936, human init 1020, **+ supervised KL 1400**, + statistics 1540 ([Vinyals et al., 2019](https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf)).
- **OpenAI Five** used no human data.
- **VPT** is the closest match to the owner's plan. It initialised from BC and added a KL to the frozen BC policy **in place of the entropy bonus**, with coefficient ρ = 0.2 **decayed ×0.9995 per iteration**. Without the KL, progress stalled from catastrophic forgetting, and the learning rate had to drop to 3e-6 before learning worked at all. The authors argue the KL is what permits the larger learning rate ([Baker et al., 2022](https://arxiv.org/abs/2206.11795), Appx G).
- [Wołczyk et al. (ICML 2024)](https://arxiv.org/abs/2402.02868) show that fine-tuning forgets pretrained skills in parts of the state space not yet revisited. They recommend knowledge-retention methods (BC replay, kickstarting, EWC) routinely.
- **Human-regularised self-play**: HR-PPO holds a fixed λ·KL(τ‖π) to a BC policy ([Cornelisse & Vinitsky, RLC 2024](https://arxiv.org/abs/2403.19648)). Its 2026 follow-up gets human-compatible driving from 30 minutes of demonstrations used as a regulariser on top of a minimal reward ([Cornelisse et al., 2026](https://arxiv.org/abs/2606.19370)). Diplomacy's RL-DiL-piKL samples λ from a distribution rather than fixing it ([Bakhtin et al., 2022](https://arxiv.org/abs/2210.05492)).
- **DeepNash** regularises toward a reference policy that is *periodically replaced by the current one* ([Perolat et al., 2022](https://arxiv.org/abs/2206.15378)). That is another way to let the anchor diminish.

This matches the project's own findings: lr 3e-4 destroyed the BC prior, an entropy of 0.01 diffused it, and kl_ref was the useful drift signal.

> **Recommendation for lanerl_jax.** **Initialise from BC and also regularise toward it.** Every system that used human data did both. Use a per-state analytic KL over the *masked* action distribution on the agent's own states, and **replace** the entropy bonus with it for champions that have a reference (VPT). Schedule: hold ρ constant while the critic warms up, i.e. until value explained variance is clearly positive. The critic starts from nothing even when the actor starts from BC. After that, decay exponentially with a half-life set in *updates-to-plateau* terms, with a **floor** (e.g. 10-20% of ρ₀) rather than zero. AlphaStar never switched its KL off, and Wołczyk shows the forgetting risk persists. For champions without replays, use entropy as now. An alternative that also covers the multi-champion case is DeepNash-style anchor replacement: every N updates, the anchor becomes a mixture of BC and the latest checkpoint. That is more design work, so treat it as a later experiment. **Cost: small to medium** (`PPOConfig` already reserves the wiring).

---

## 6. Asymmetric critic / centralised value

**Evidence.** AlphaStar's value used opponent observations during training only. In its ablation, average win rate rose from **22% without opponent info to 82% with it** ([Vinyals et al., 2019](https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf)). JueWu's full-game value net takes "invisible opponent information" and says the design follows AlphaStar. It also splits the value into heads per reward group ([Ye et al., 2020b](https://arxiv.org/abs/2011.12692)). Dactyl ([2018](https://arxiv.org/abs/1808.00177)) and Sony's GT7 vision agent ([2025](https://arxiv.org/abs/2504.09021)) both use asymmetric critics ([Pinto et al., 2017](https://arxiv.org/abs/1710.06542)). MAPPO's suggestion 2 is to include both agent-specific local features and global features in the value input ([Yu et al., 2022](https://arxiv.org/abs/2103.01955)). The caveat: a critic on state *alone* is biased for a history-dependent policy. The unbiased form conditions on **history and state** ([Baisero & Amato, AAMAS 2022](https://arxiv.org/abs/2105.11674)).

> **Recommendation for lanerl_jax.** Both champions' observations are already built every step (`trainer.py` stacks blue and red), so a privileged critic costs almost nothing. The value head gets `[h_actor (history, from the GRU), enc(opponent's observation)]` plus any unfogged enemy state: HP, cooldowns, gold, position when fogged. Stop gradients from the privileged branch into the actor trunk so nothing privileged can leak into acting. Optionally enforce the zero-sum structure at α = 1 with `V_blue = f(a,b) − f(b,a)`. That antisymmetric form is my suggestion and has no citation. **Cost: small to medium.**

---

## 7. Population and league

**Evidence.** Five played the latest policy 80% of the time and past versions 20%. Past versions were sampled by softmax over quality scores, with a snapshot every 10 iterations and q_i lowered when the current agent wins ([Berner et al., 2019](https://arxiv.org/abs/1912.06680), Appx N). AlphaStar's naive self-play had high Elo (1519) but was "more forgetful": 46% minimum win rate against past versions, against 71% for PFSP+SP. Main agents trained 35% SP / 50% PFSP / 15% against forgotten players. PFSP weights opponents by `(1−x)^p` (hardest first) or `x(1−x)` (even matches). Main and league exploiters raised main-agent Elo from 1540 to 1824 ([Vinyals et al., 2019](https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf)). On whether a symmetric 1v1 will cycle, self-play *can* cycle in non-transitive games ([Balduzzi et al., 2019](https://arxiv.org/abs/1901.08106)). Real games look like "spinning tops": non-transitivity is concentrated at *intermediate* skill, and populations matter most there ([Czarnecki et al., 2020](https://arxiv.org/abs/2004.09468)). No paper measures cycling in MOBA laning specifically. My own expectation is plausible cycles once trading exists (all-in vs. farm-and-freeze), and mostly transitive progress at the current skill level.

> **Recommendation for lanerl_jax.** Keep mirror self-play for most games, but add a Five-style pool on-device: **K = 16 frozen snapshots** carried as a stacked param pytree, one added every ~50 updates, evicting the oldest non-protected one. Protect the BC policy and the best-by-CS snapshot permanently. At episode reset, sample the opponent per env: **80% latest, 20% pool** via PFSP `x(1−x)`, where x is the lane-outcome win rate. Train only the learner's side in pool games. Log the **minimum win rate vs. the pool**, AlphaStar's forgetting proxy, alongside CS@10. Exploiters come after champion #2. **Cost: medium.**

---

## 8. Things the top systems did that we do not

1. **Return/value normalisation.** Five normalised rewards by a running std, with the value weight applied after normalisation ([Berner et al., 2019](https://arxiv.org/abs/1912.06680), Table 2). PopArt keeps value learning scale-invariant ([Hessel et al., 2018](https://arxiv.org/abs/1809.04474)). MAPPO's suggestion 1 is value normalisation ([Yu et al., 2022](https://arxiv.org/abs/2103.01955)). [Andrychowicz et al. (2020)](https://arxiv.org/abs/2006.05990) found that **PPO-style value clipping hurt**. Ours has neither normalisation nor sensible clipping. `trainer.py` records value_loss reaching 542.7 as returns grew, and the value clip is 0.2 in raw units. This probably also confounds the sweep-B finding that a fast critic lost: a 3e-4 readout chasing unnormalised, growing targets through a 0.2 clip is not a clean lr test. **Small cost, high priority.**
2. **Reaction time / observation latency.** Five reacts in 217 ms on average. JueWu's online response is 193 ms (133 ms observation delay + ~60 ms processing) ([Ye et al., 2020b](https://arxiv.org/abs/2011.12692)). AlphaStar has ~110 ms latency plus self-chosen delays ([Vinyals et al., 2019](https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf)). Dactyl randomised action delays in simulation for transfer ([OpenAI, 2018](https://arxiv.org/abs/1808.00177)). Our sim acts on the current tick. The deployed memory-read and client path does not. **Train with an observation delay sampled from the measured deployment latency distribution** (a ring buffer in env state). This is the most direct sim-to-server lever. It depends on #2 in section 1, because memory is how a policy compensates for latency. **Small to medium.**
3. **Previous action in the observation.** Five fed "my previous action". AdA fed previous action and reward ([2023](https://arxiv.org/abs/2301.07608)). Ours feeds neither. **Small.**
4. **Decision rate.** Every MOBA system acts at ~7.5 Hz: Five uses frameskip 4 plus a delay head that it "did not learn to" use, and JueWu uses 133 ms. We act at 30 Hz, which puts 4x more steps between an action and its consequence. Our GAE λ = 0.99 at 30 Hz is ≈ 0.95 at 7.5 Hz in game time (0.95^(1/4) ≈ 0.987), so that part is consistent. The credit-assignment burden is the issue, and a transformer core does not fix it (Ni et al.). The existing 15 Hz fallback (memory note) is supported by this evidence. Test it as an ablation after the critic fixes, not before.
5. **Schedules.** Five annealed entropy 0.01→0.001, learning rate 5e-5→5e-6, and the GAE horizon 180→360 s (840 s final) ([Berner et al., 2019](https://arxiv.org/abs/1912.06680), Table 2 / Fig. 7). Andrychowicz rates linear lr decay as "of secondary importance". A **horizon curriculum** (start ~30-60 s, grow to 120 s+) is the schedule most relevant to us. **Small.**
6. **Auxiliary losses / multi-head value.** Five trained win-probability and other auxiliary predictions, mostly with stop-gradient. JueWu splits the value by reward group (farming, KDA, damage, pushing, win) ([Ye et al., 2020b](https://arxiv.org/abs/2011.12692)). For us, a per-term value head decomposes the exp-vs-gold question directly. An auxiliary "this minion dies within k ticks" prediction would give the last-hit perception its own gradient, the lesson from the blind movement head. **Small to medium.**
7. **Large-batch PPO settings.** Five ran 1-3M-step batches with sample reuse ~1 and found staleness and reuse both hurt ([Berner et al., 2019](https://arxiv.org/abs/1912.06680), Fig. 5). MAPPO says ≤10-15 epochs and "avoid splitting data into mini-batches" ([Yu et al., 2022](https://arxiv.org/abs/2103.01955)). Ours is 131k samples × 4 epochs × 4 minibatches, which is in range. Anakin has no staleness. Nothing urgent.

---

## Ranked changes

Ordered by expected impact on CS@10 and on sim-to-server transfer, then by cost. The "impact" column is my judgement from the evidence above, not a measurement.

| # | Change | Impact (CS@10 / transfer) | Cost |
|---|---|---|---|
| 1 | Return normalisation (running-std or PopArt); drop raw-unit value clipping (§8.1) | High / Med. The critic is the prerequisite for every advantage | small |
| 2 | Per-unit hp-delta and velocity features in the builder, keyed by unit index (§1b) | High / Med. Restores the last-hit timing signal | small |
| 3 | Cast-button masking from observation cooldowns + BC label cleaning + previous action in obs (§3, §8.3) | Med / High. The mask transfers by construction | small |
| 4 | BC init + KL-to-BC replacing entropy, hold, then exponential decay to a floor (§5) | High / High. Human-like play is the transfer prior | small-med |
| 5 | GRU core with stored-state truncated BPTT, env-axis minibatching, reset at episode end only (§1a) | Med-High / High. Latency and physics compensation | medium |
| 6 | Observation-latency randomisation matched to the deployed path; silent-cast-failure randomisation (§8.2, §3) | Low / High | small-med |
| 7 | Privileged critic: history plus opponent observation, stop-grad into the actor (§6) | Med / Low | small-med |
| 8 | Multi-head value per reward term + last-hit and feasibility auxiliary heads (§8.6, §3) | Med / Med | small-med |
| 9 | GAE horizon curriculum; 15 Hz ablation (§8.4-5) | Med / Low | small |
| 10 | Opponent pool: 80/20, PFSP `x(1−x)`, K = 16, min-win-rate-vs-pool metric (§7) | Low now, rising with skill / Med | medium |
| 11 | Autoregressive decoder button→target→x→y, keeping the pointer (§2) | Low for Garen; required for champion #2 | medium |
| 12 | Champion-ID + ability-descriptor conditioning, per-champion cast-type argument table, incremental pool (§4) | none for Garen-vs-Garen; the roadmap's step 4 | large |
| 13 | Transformer-over-time core | Not recommended now: no evidence it helps a 1v1 reactive task, and credit assignment is the bottleneck | large |

## References

- Andrychowicz et al. 2020, What Matters in On-Policy RL. https://arxiv.org/abs/2006.05990
- Badia et al. 2020, Agent57. https://arxiv.org/abs/2003.13350
- Baisero & Amato 2022, Unbiased Asymmetric RL under Partial Observability. https://arxiv.org/abs/2105.11674
- Baker et al. 2022, Video PreTraining (VPT). https://arxiv.org/abs/2206.11795
- Bakhtin et al. 2022, No-Press Diplomacy via Human-Regularized RL and Planning. https://arxiv.org/abs/2210.05492
- Balduzzi et al. 2019, Open-ended Learning in Symmetric Zero-sum Games. https://arxiv.org/abs/1901.08106
- Berner et al. (OpenAI) 2019, Dota 2 with Large Scale Deep RL. https://arxiv.org/abs/1912.06680
- Chen et al. 2021, Decision Transformer. https://arxiv.org/abs/2106.01345
- Cornelisse & Vinitsky 2024, Human-compatible driving partners through data-regularized self-play RL. https://arxiv.org/abs/2403.19648
- Cornelisse et al. 2026, Human-like autonomy emerges from self-play and a pinch of human data. https://arxiv.org/abs/2606.19370
- Cusumano-Towner et al. 2025, Robust Autonomy Emerges from Self-Play (GIGAFLOW). https://arxiv.org/abs/2502.03349
- Czarnecki et al. 2020, Real World Games Look Like Spinning Tops. https://arxiv.org/abs/2004.09468
- Hafner et al. 2023, Mastering Diverse Domains through World Models (DreamerV3). https://arxiv.org/abs/2301.04104
- Hessel et al. 2018, Multi-task Deep RL with PopArt. https://arxiv.org/abs/1809.04474
- Hu et al. (Wayve) 2023, GAIA-1. https://arxiv.org/abs/2309.17080
- Huang & Ontañón 2020 (FLAIRS 2022), A Closer Look at Invalid Action Masking. https://arxiv.org/abs/2006.14171
- Hwang et al. (Waymo) 2024, EMMA. https://arxiv.org/abs/2410.23262
- Kapturowski et al. 2019, R2D2. https://openreview.net/forum?id=r1lyTjAqYX
- Li et al. 2023, A Survey on Transformers in RL. https://arxiv.org/abs/2301.03044
- Liu et al. 2021, An Introduction of mini-AlphaStar (AlphaStar architecture details). https://arxiv.org/abs/2104.06890
- Lu et al. (Waymo) 2022, Imitation Is Not Enough (BC-SAC). https://arxiv.org/abs/2212.11419
- Morad et al. 2023, POPGym. https://arxiv.org/abs/2303.01859
- Nayakanti et al. (Waymo) 2022, Wayformer. https://arxiv.org/abs/2207.05844
- NVIDIA 2025, Alpamayo-R1. https://arxiv.org/abs/2511.00088
- Ni et al. 2023, When Do Transformers Shine in RL? https://arxiv.org/abs/2307.03864
- OpenAI 2018, OpenAI Five (blog). https://openai.com/index/openai-five/
- OpenAI et al. 2018, Learning Dexterous In-Hand Manipulation. https://arxiv.org/abs/1808.00177
- Parisotto et al. 2019, Stabilizing Transformers for RL (GTrXL). https://arxiv.org/abs/1910.06764
- Perolat et al. 2022, Mastering Stratego (DeepNash). https://arxiv.org/abs/2206.15378
- Pinto et al. 2017, Asymmetric Actor Critic. https://arxiv.org/abs/1710.06542
- Reed et al. 2022, A Generalist Agent (Gato). https://arxiv.org/abs/2205.06175
- Seff et al. (Waymo) 2023, MotionLM. https://arxiv.org/abs/2309.16534
- SIMA Team 2024, Scaling Instructable Agents Across Many Simulated Worlds. https://arxiv.org/abs/2404.10179
- Adaptive Agents Team (DeepMind) 2023, Human-Timescale Adaptation (AdA). https://arxiv.org/abs/2301.07608
- Sony AI 2025, Champion-level Vision-based RL Agent for GT7. https://arxiv.org/abs/2504.09021
- thinkautonomous 2024 (secondary), Tesla's FSD Architecture. https://www.thinkautonomous.ai/blog/tesla-end-to-end-deep-learning/
- Vinyals et al. 2019, Grandmaster level in StarCraft II (AlphaStar). https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf
- Wei et al. 2022, Honor of Kings Arena. https://arxiv.org/abs/2209.08483
- Wołczyk et al. 2024, Fine-tuning RL Models is Secretly a Forgetting Mitigation Problem. https://arxiv.org/abs/2402.02868
- Wurman et al. 2022, Outracing champion Gran Turismo drivers (GT Sophy). https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/nature22.pdf
- Ye et al. 2020, Mastering Complex Control in MOBA Games (JueWu 1v1). https://arxiv.org/abs/1912.09729
- Ye et al. 2020b, Towards Playing Full MOBA Games with Deep RL. https://arxiv.org/abs/2011.12692
- Yu et al. 2022, The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (MAPPO). https://arxiv.org/abs/2103.01955
- Zabounidis et al. 2026, Overcoming Valid Action Suppression in Unmasked Policy Gradient Algorithms. https://arxiv.org/abs/2603.09090
- Zhao et al. 2023, ACT (action chunking). https://arxiv.org/abs/2304.13705
