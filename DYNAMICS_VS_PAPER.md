# Dreamer 4 (paper) vs. ahriuwu dynamics — exhaustive replication reference

**Paper.** Danijar Hafner\*, Wilson Yan\*, Timothy Lillicrap (2025), *"Training Agents Inside of
Scalable World Models"* (Dreamer 4), **arXiv 2509.24527v1**. All blockquotes below are transcribed
**verbatim** from the paper PDF (32 pp; extracted with `pypdf`). PDF text-extraction drops some
inter-word spaces; where that happened the spacing has been restored but no words were changed.
Equation numbers, section names, and figures/tables are the paper's own.

**Background papers Dreamer 4 builds on** (Sec 2, refs [14] and [21]):
- Diffusion Forcing — Chen, Monsó, Du, Simchowitz, Tedrake, Sitzmann, *NeurIPS 2024*, arXiv 2407.01392 (ref [14]).
- One-Step Diffusion via Shortcut Models — Frans, Hafner, Levine, Abbeel, 2024, arXiv 2410.12557 (ref [21]).

**Our code.** Paths are `/srv/nfs/projects/ahriuwu/...`; `file:line` cites are against the repo and
the companion `DYNAMICS_REVIEW.md`. The active training run referenced throughout is **Slurm job
124** (medium, ~114.6M params, x-prediction diffusion forcing, shortcut **off**).

**Paper section map** (for cites): **Sec 2 Background** (Flow matching / Shortcut models /
Diffusion forcing, Eqs 1–4, p3). **Sec 3.1 Causal Tokenizer** (Eq 5, p5). **Sec 3.2 Interactive
Dynamics** (Eqs 6–8, pp5–6). **Sec 3.3 Imagination Training** (Eqs 9–11, pp6–8). **Sec 3.4
Efficient Transformer** (p8). **Sec 4** experiments incl. the 30%-separate-images setup (p9) and
**Table 2** ablation cascade (p15). **Fig 8** DF-vs-shortcut sampling-steps curve (p16).

> **Convention note (critical, used everywhere).** The paper's signal level is **τ=1 → clean,
> τ=0 → pure noise**. Our code uses the *same* convention (`diffusion.py:8, 24-30`). Every τ
> statement below is in this convention.

---

# PART 1 — Paper (verbatim) vs. our replication, detail by detail

## 1.1 Flow matching (Sec 2, Eq 1)

> "Our world model is based on the paradigm of diffusion models, where the network `f_θ` is trained
> to restore the a data point `x1` given a corrupted version `xτ`. The signal level `τ ∈ [0,1]`
> determines the mixture of noise and data and is randomized during training, where `τ=0`
> corresponds to pure noise and `τ=1` means clean data. We build on the flow matching formulation
> because of its simplicity, where the network predicts the velocity vector `v = x1 − x0` that
> points towards the clean data:" — Sec 2, Background

> `xτ = (1−τ)x0 + τ x1 ,  x0 ∼ N(0,𝕀),  x1 ∼ D,  τ ∼ p(τ)`
> `L(θ) = ‖ f_θ(xτ, τ) − (x1 − x0) ‖²`  — **Eq 1**

> "The signal level is typically sampled from a uniform distribution or a logit-normal
> distribution." — Sec 2 (immediately after Eq 1)

**Our replication.** Same linear interpolation, `z_τ = τ·z_0 + (1−τ)·ε` (`diffusion.py:62`,
`DiffusionSchedule.add_noise` `:35-64`); convention documented `diffusion.py:8, 24-30`. **But** our
world model does **x-prediction, not the Eq-1 v-prediction** — the net outputs clean `z_0` (see
§1.6). The generic uniform sampler `sample_timesteps` (`diffusion.py:66-87`) exists but is **not**
what training uses; training uses the custom diffusion-forcing schedule (§1.5).

**Verdict: MATCH** on the interpolation/convention; the *prediction target* differs by design (Eq 1
is v-pred; Dreamer 4 itself switches to x-pred in Sec 3.2, and so do we — see §1.6).

## 1.2 Flow-matching inference / Euler integration (Sec 2, Eq 2)

> "At inference time, the sampling process starts with a pure noise vector `x0` and iteratively
> transforms it into a clean data point `x1` over `K` sampling steps with step size `d = 1/K`:"
> — Sec 2

> `x_{τ+d} = xτ + f_θ(xτ, τ) d ,  x0 ∼ N(0,𝕀)`  — **Eq 2**

**Our replication.** Euler integration in `DiffusionSchedule.sample` (`diffusion.py:146-199`) and in
the rollout inner loop (`dynamics.py:879-891`). Because we do **x-prediction**, the Euler update is
written in x-form: `z_next = τ_next·ẑ0 + (1−τ_next)·ε̂` rather than `xτ + v·d`; the two are
algebraically equivalent for a correct noise estimate. **This is the exact spot fixed today** — see
PART 3 (the frozen-noise bug made this update non-equivalent for >1 step).

**Verdict: MATCH** (post-fix). The many-step Euler sampler is faithful; step size `d=1/K`
reproduced via `step=(1−eps)/num_steps` (`dynamics.py:870`).

## 1.3 Shortcut models (Sec 2, Eqs 3–4)

> "Shortcut models condition the neural network not only on the signal level `τ` but also on the
> requested step size `d`. This allows them to choose the step size at inference time and generate
> data points using only a few sampling steps and forward passes of the neural network. For the
> finest step size `d_min`, shortcut models are trained using the flow matching loss. For larger
> step sizes `d_min < d ≤ 1`, shortcut models are trained using a bootstrap loss that distills two
> smaller steps, where `sg(·)` stops the gradient:" — Sec 2

> `x0 ∼ N(0,I),  x1 ∼ D,  τ, d ∼ p(τ, d)`
> `b′ = f_θ(xτ, τ, d/2) ,  b″ = f_θ(x′, τ+d/2, d/2) ,  x′ = xτ + b′ d/2`
> `L(θ) = ‖ f_θ(xτ, τ, d) − v_target ‖²`,
> `v_target = { x1 − x0 if d = d_min ;  sg(b′ + b″)/2 else }`  — **Eq 3**

> "The step size is sampled uniformly as a power of two, based on the maximum number of sampling
> steps `K_max`, which defines the finest step size `d_min = 1/K_max`. The signal level is sampled
> uniformly over the grid that is reached by the current step size:" — Sec 2

> `d ∼ 1/U({1, 2, 4, 8, …, K_max}) ,  τ ∼ U({0, 1/d, …, 1−1/d})`  — **Eq 4**

> "At inference time, one can condition the model on a step size `d = 1/K` to target `K` sampling
> steps, without suffering from discretization error because the model has learned to predict the
> end point of each step. In practice, shortcut models generate high-quality samples with 2 or 4
> sampling steps, compared to 64 or more steps for typical diffusion models." — Sec 2

**Our replication.** `ShortcutForcing` (`diffusion.py:263-535`) implements this in full but it is
**not active in job 124**. Step-size set `{1,2,…,k_max}` (`diffusion.py:277`); power-of-two sampling
`sample_step_size` (`:279-298`) with a **progressive `max_step_idx` curriculum** (start `d=1` only,
grow) — an addition beyond Eq 4, to escape the teacher≈student bootstrap trap (`:281-298`). Grid τ
sampling `sample_tau_for_step_size` / `_2d` reproduces Eq 4 exactly: `τ ∈ {0, d/k_max, …, 1−d/k_max}`
(`:300-369`; comment cites "Paper Eq. 4" `:305, :341`). Bootstrap two-half-step teacher `compute_loss`
(`:371-535`; comment cites "Paper Section 3.2, Eq. 7" `:380`). Model consumes `step_size` via a
discrete `log2(d)` embedding, `num_step_sizes=7` for d∈{1,…,64} (`dynamics.py:448-452, 769-773`).

> **Correction to `DYNAMICS_REVIEW.md`:** the `ShortcutForcing` **class docstring**
> (`diffusion.py:270`) still reads *"Not implemented in MVP - use standard diffusion first."* This
> is **stale** — the real `compute_loss` below it (`:371-535`) is fully implemented. Read the code,
> not the docstring.

**Verdict: DEFERRED.** Faithful implementation, but shortcut is **off** for job 124 (Phase-B), so
Eqs 3–4 are inactive in the run under review. One deliberate deviation *within* the shortcut path —
the bootstrap is computed in **x-space, not v-space** — is detailed in §1.7.

## 1.4 Diffusion forcing (Sec 2)

> "For sequential data, diffusion forcing assigns a different signal level to each time step of the
> data sequence, producing a corrupted sequence. This allows applying loss terms to all time steps
> in the sequence, where each time step serves both as denoising task and as history context for
> later time steps. At inference time, diffusion forcing supports flexible noise patterns, such as
> generating the next frame given clean or lightly noised history." — Sec 2, Background

**Our replication.** Per-frame τ (a *different* level per time step) is realized by
`sample_diffusion_forcing_timesteps` returning shape `(B, T)` (`diffusion.py:89-144`) and applied
per-frame in `add_noise` (`:58-62`). Loss is applied to **all** frames (`x_prediction_loss` reduces
per-frame then means over B,T, `:240-258`). Inference "clean/lightly-noised history" is the rollout
KV-cache context corruption (§1.13). **The precise per-frame *pattern* we sample is repo-invented and
not the paper's** — see §1.5.

**Verdict: MATCH on the mechanism** (per-frame levels, loss on all steps); **DEVIATE on the specific
noise pattern** (§1.5).

## 1.5 Per-frame noise / τ schedule + the τ grid + context-corruption τ_ctx

**What the paper actually prescribes for *training* τ.** The paper gives **only two** sampling laws:
- Flow-matching: *"typically sampled from a uniform distribution or a logit-normal distribution"* (Sec 2, after Eq 1).
- Shortcut grid: `τ ∼ U({0, 1/d, …, 1−1/d})`, i.i.d. per frame over the reachable grid (**Eq 4**).

There is **no** "sample a horizon `h`; context frames = `U(0.9,1)`; target frames ramp toward 0 past
`h`" scheme anywhere in the paper. The only "context vs. next-frame" statement is about **inference**:

> "…diffusion forcing supports flexible noise patterns, such as generating the next frame given clean
> or lightly noised history." — Sec 2

And the inference context-corruption level:

> "We slightly corrupt the past inputs to the dynamics model to signal level `τ_ctx = 0.1` to make
> the model robust to small imperfections in its generations." — Sec 3.2

**Our replication (training schedule).** `sample_diffusion_forcing_timesteps`
(`diffusion.py:89-144`), defaults `tau_ctx=0.9, tau_min=0.0`:
- per-sequence horizon `h ~ randint(1, T)` (`:119`);
- **context** frames (`pos < h`): `τ ~ U(0.9, 1.0)` (`:132-133`);
- **target** frames (`pos ≥ h`): deterministic linear ramp `0.9 → 0` in normalized distance past `h`,
  **plus `N(0, 0.02²)` jitter**, clamped `[0, 0.9]` (`:135-140`);
- used by `_forward_standard` (`train_dynamics.py:980`).

Consequences: only ~10% of the τ range (0.9–1.0) is ever assigned to context; deep-target τ is
near-deterministic in position; the pure-noise end (τ≈0, the regime rollout runs in) gets thin,
position-correlated coverage. `DYNAMICS_REVIEW.md` §4.1/§6.7 flags this as intentional-but-context-heavy.

**Our replication (context-corruption τ_ctx at rollout).** `rollout(tau_ctx=0.1)` uses `tau_ctx` as a
**width**: context `τ ~ U(1−tau_ctx, 1) = U(0.9, 1.0)` — i.e. near-**clean** (`dynamics.py:849, 865`).
This matches the paper's *intent* (context sits near clean; the paper's "signal level 0.1" reads as a
small corruption amount).

**Verdict: DEVIATE (training τ schedule)** — repo-invented, context-heavy, not in the paper.
**MATCH (rollout context corruption)** — near-clean context reproduced.

## 1.6 x-prediction vs. v-prediction (Sec 3.2, Eq 6)

> "The dynamics model takes the interleaved sequence of actions `a = {aₜ}`, discrete signal levels
> `τ = {τₜ}` and step sizes `d = {dₜ}`, and corrupted representations `z̃ = {zₜ^(τₜ)}` as input and
> predicts the clean representations `z1 = {zₜ¹}`. Note that `t ∈ [1, T]` is the sequence timestep
> while `τₜ ∈ [0,1]` is the signal level at that step." — Sec 3.2

> `z0 ∼ N(0,1),  z1 ∼ D,  τ, d ∼ p(τ, d),  τ, d ∈ [0,1]^T`
> `ẑ1 = f_θ(z̃, τ, d, a) ,  z̃ = (1−τ)z0 + τ z1`  — **Eq 6**

> "Shortcut models parameterize the network to predict velocities `v = x1 − x0`, called
> v-prediction. This approach excels when generating the output jointly as one block, such as for
> image or video generation models. However, v-prediction trains the network to produce
> high-frequency outputs. When iteratively generating long videos frame by frame, this can cause
> subtle errors that accumulate over time. Instead, we found that parameterizing the network to
> predict clean representations, called x-prediction, enables high-quality rollouts of arbitrary
> length. Computing the flow loss term in x-space is straightforward." — Sec 3.2

And from the ablation discussion (Sec 4.4):

> "The complete model achieves an FVD of 57 compared to 306 for the naive diffusion forcing
> transformer baseline and 124 for the complete architecture with v-space prediction and losses."
> — Sec 4.4 (immediately after Table 2 discussion)

**Our replication.** x-prediction throughout: net outputs clean `z_0` (`diffusion.py:5`, output head
`dynamics.py:816-821`); flow loss in x-space (`x_prediction_loss`, `diffusion.py:221-258`). Docstring
cites the same rationale (`diffusion.py:3-6`).

**Verdict: MATCH** (exact, and the paper's stated motivation — "high-quality rollouts of arbitrary
length" — is exactly the property we want).

## 1.7 Shortcut-forcing objective + bootstrap term & its space (Sec 3.2, Eq 7 + footnote)

> "To compute the bootstrap loss term, we convert the network output into v-space and scale the
> resulting loss back into x-space∗:" — Sec 3.2

> `b′ = (f_θ(z̃, τ, d/2, a) − z_τ)/(1−τ) ,  z′ = z̃ + b′ d/2`
> `b″ = (f_θ(z′, τ+d/2, d/2, a) − z′)/(1−(τ+d/2))`
> `L(θ) = { ‖ẑ1 − z1‖²₂        if d = d_min ;`
> `        (1−τ)² ‖ (ẑ1 − z̃)/(1−τ) − sg(b1 + b2)/2 ‖²₂   else }`  — **Eq 7**

> **Footnote ∗:** "The network output is converted as `v̂τ = (x̂1 − xτ)/(1−τ)`. The MSE in x-space
> and v-space is related by `‖x̂1 − x1‖²₂ = (1−τ)² ‖v̂τ − vτ‖²₂`, motivating a `(1−τ)²` multiplier to
> bring the bootstrap loss into a range similar to the x-space flow loss." — Sec 3.2 footnote

> "Low signal levels contain less learning signal, because the flow matching term degenerates to
> predicting the dataset mean, while the bootstrap term is generally easier to optimize because it
> has deterministic targets compared to the noisy flow matching term." — Sec 3.2

**Our replication.** `ShortcutForcing.compute_loss` (`diffusion.py:371-535`): base step `d=1` uses the
x-prediction flow loss with ramp weight (`:426-432`); `d>1` builds the two-half-step teacher `b′, b″`
(`:440-483`) exactly as Eq 7. **Deliberate deviation:** the bootstrap loss is computed **directly in
x-space** — `x_diff = z_pred − z_target.detach()`, MSE over C,H,W, **clamped to 100**, ramp-weighted
(`:489-514`) — *not* the paper's v-space `(1−τ)²` form. Rationale in-code (`:489-498`): the paper's
`÷(1−τ)` then `×(1−τ)²` loses precision in bf16, and when teacher≈student the v-diff collapses to
zero. A `bootstrap_weight=10.0` boost (`:272, 516-523`) compensates for the smaller deterministic-target
MSE. `DYNAMICS_REVIEW.md` §6.1 rates this benign at small scale (K4 tracks K64 within ~1 dB).

**Verdict: DEFERRED / DEVIATE-within-shortcut.** Faithful two-step bootstrap *structure*; the
x-space-vs-v-space computation deviates by design. **Irrelevant to job 124** (shortcut off).

## 1.8 Ramp loss weight (Sec 3.2, Eq 8)

> "To focus the model capacity on signal levels with the most learning signal, we propose a ramp
> loss weight that linearly increases with the signal level `τ`, where `τ=0` corresponds to full
> noise and `τ=1` to clean data:" — Sec 3.2

> `w(τ) = 0.9τ + 0.1`  — **Eq 8**

And its ablation value (Table 2): adding the ramp weight moves **FVD 151 → 102**. And Fig 8 caption
context:

> "Figure 8 compares the visual quality of shortcut forcing to diffusion forcing—both using the
> x-space loss with ramp weight—for a wider range of sampling steps." — Sec 4.4 / Fig 8

**Our replication.** `ramp_weight(τ) = 0.9*τ + 0.1` **verbatim** (`diffusion.py:202-218`; cites "Paper
Eq. 8" `:203`), applied in `x_prediction_loss` (`:249-251`) and inside the shortcut bootstrap
(`:509-513`).

**Verdict: MATCH** (exact). **Caveat (see PART 3, deviation #3):** in the paper the low-τ down-weight
is *compensated* by the bootstrap term (Eq 7, which lives at low τ). With shortcut **off**, job 124
down-weights low-τ x-prediction to 0.1 with **nothing** filling that role → high-noise / far-horizon
learning is doubly starved (thin schedule coverage from §1.5 × 0.1 weight here).

## 1.9 Independent frames / 30%-separate-images (Sec 4 setup)

> "To improve generations without context, we treat 30% of the videos in the batch as separate
> images, effectively training the dynamics model to generate start frames." — Sec 4 (experimental
> setup, p9)

(Note: the paper's phrase for the temporal-every-4 benefit — *"possibly because of the inductive bias
of spatial attention that focuses computation on the current frame"* (Sec 4.4) — is about the layer
ratio, **not** about disabling temporal attention. The paper never describes a diagonal temporal mask.)

**Our replication.** `use_independent = random.random() < independent_frame_ratio` — **one boolean per
micro-batch** (`train_dynamics.py:741`, default ratio **0.3** `:448-452`), passed as
`independent_frames=` into `model(...)` (`:982`). Inside the model this swaps the temporal mask for a
**diagonal** mask so **every frame in the whole T=128/256 sequence attends only to itself**
(`dynamics.py:683-686, 703`; `layers.py:234-235` `_diagonal_mask_mod: q_idx == kv_idx`; manual path
`torch.eye`). Docstring miscites this as "DreamerV4 Section 3.2" (`dynamics.py:685`).

**The mismatch.** Paper = 30% of **batch elements** are genuine **single images** (start-frame
generation); the other 70% remain full temporal sequences, so the temporal-prediction pathway trains
on 70% of examples **every** batch. Ours = 30% of **steps** strip temporal context from **entire
multi-frame sequences**, so on those steps the temporal layers are trained to be a **cross-frame
no-op**, and they receive prediction gradient on only ~70% of steps. Same "0.3", wrong unit.

**Verdict: DEVIATE (misapplication).** High-suspicion training-side deviation (see PART 3, #1).

## 1.10 Space–time attention factorization / temporal every 4th layer (Sec 3.4)

> "The architecture is a 2D transformer with time and space dimensions. To support interactive
> generation, the attention is masked to be causal in time, so that all tokens within a time step
> can attend to each other and to the past." — Sec 3.4

> "First, we break up the cost of dense attention over all video tokens by using separate space-only
> and time-only attention layers. Second, we find that only a relatively small number of temporal
> layers are needed and only use temporal attention once every 4 layers, in line with recent
> findings. Third, we apply GQA to all attention layers in the dynamics, where multiple query heads
> attend to the same key-value heads to reduce the KV cache size further." — Sec 3.4

> "Using temporal attention only every 4 layers not only speeds up training and inference but also
> improves generation quality, possibly because of the inductive bias of spatial attention that
> focuses computation on the current frame. GQA further accelerates generations without degrading
> performance." — Sec 4.4

Table 2 rows: "+ Long context every 4 layers" (FVD 102 → **70**, and FPS 9.1 → 18.9) and "+ GQA"
(FVD 70 → **71**, FPS → 23.2).

**Our replication.** `is_temporal = (i % temporal_every == temporal_every-1)`, `temporal_every=4`
(`dynamics.py:354, 478-496`) → **14 spatial / 4 temporal** of 18 layers (temporal at 0-idx 3,7,11,15).
Spatial block: full non-causal attention within each frame's 266 tokens (`dynamics.py:119-123`).
Temporal block: **causal** across frames per spatial position (`:124-131`; masks `layers.py:230-231,
355-358`). GQA via `num_kv_heads=4` (12 query / 4 KV heads, `dynamics.py:158`, `layers.py`). Soft-cap
50 and QKNorm applied (`layers.py:438-450, 322-325`).

**Verdict: MATCH** (ratio, causality-in-time, GQA all reproduced).

## 1.11 Conditioning token(s) (Sec 3.2)

> "The representations are linearly projected into `S_z` spatial tokens and concatenated with `S_r`
> learned register tokens and a single token for the shortcut signal level and step size. Since the
> signal level and step size are discrete, we encode each with a discrete embedding lookup and
> concatenate their channels." — Sec 3.2

> "Actions can contain multiple components, such as mouse and keyboard. We encode each action
> component separately into `S_a` tokens and sum the results together with a learned embedding.
> Continuous actions components are linearly projected and categorical or binary components use an
> embedding lookup. When training unlabeled videos, only the learned embedding is used." — Sec 3.2

**Our replication.** Per-frame token layout `[256 latent | 8 register | 1 action | 1 condition]` = 266
(`dynamics.py:742-795`). Register tokens: 8 learned, shared (`:438-441, 752-754`). Action token: sum of
factorized embeddings — `Linear(2→768)` movement + per-key binary `Embedding`, summed (`embed_actions`
`:536-570`); unlabeled → learned `no_action_embed` (`:760`), matching *"only the learned embedding is
used."* Condition token: discrete `tau_embed` + `step_embed`, concatenated then projected
`Linear(2·768→768)` (`_build_condition_token` `:572-616`); discrete τ index `tau_idx =
(τ·k_max).long().clamp(0,63)`, `num_tau_levels=64`, `num_step_sizes=7` (`:448-452, 596, 765`) —
matches *"discrete embedding lookup and concatenate their channels."*

**Deviation:** we **additionally** inject τ+step **additively onto all 266 tokens**
(`x = x + tau_emb + step_emb`, `dynamics.py:763-776`) on top of the appended condition token
(`:793-794`). The paper describes **only** the single appended token. `DYNAMICS_REVIEW.md` §6.3.

**Verdict: DEVIATE (superset).** Appended-token conditioning is faithful; the extra additive injection
is beyond the paper. Benign-to-helpful (stronger τ signal), low suspicion.

## 1.12 Alternating batch lengths + length generalization (Sec 3.4)

> "Increasing spatial tokens directly improves visual quality, whereas increasing temporal tokens
> allows training longer context lengths for more temporally consistent generation. To support
> efficient training, we alternate training on many short batches and occasional long batches, and
> finetune the model on only long batches afterwards. Alternating batch lengths produces
> intermediate training metrics and generations that are more indicative of final model performance
> than training only on short batches. The batch lengths need to be longer than the context length
> of the model to prevent the transformer from overfitting to always seeing a start frame at the
> beginning of its context, enabling length generalization to arbitrary generation lengths." — Sec 3.4

> "Training on alternating batch lengths is similar to progressive training and speeds up learning
> while allowing to generate long videos for inspection throughout training." — Sec 4.4

Table 2: "+ Alternating batch lengths" moves train-step 9.8s → **1.5s** (FVD 102 → 80).

**Our replication.** `--alternating-lengths --seq-len-short 128 --seq-len-long 256 --long-ratio 0.1`
(`slurm_dyn_train.sbatch`; loop `train_dynamics.py:297-345, 716-719`). `max_seq_len=256`
(`dynamics.py:356`).

**Deviation (subtle):** the paper requires **batch length > context length**. Ours sets batch length
**==** model context length (128/256) with no separate longer batch window, so the paper's explicit
guard against "always seeing a start frame at position 0" is **not** satisfied. Also there is no
documented long-only finetune phase for job 124.

**Verdict: MATCH (alternating scheme) / DEVIATE (batch-len == ctx-len, no length-gen margin).** Minor
contributor to weak far-horizon rollout.

## 1.13 Inference / rollout — K=4 shortcut steps, τ_ctx=0.1, per-frame generation (Sec 3.2)

> "At inference time, the dynamics model supports different noise patterns. We sample autoregressively
> in time and generate the representations of each frame using the shortcut model with `K=4` sample
> steps with corresponding step size `d = 1/4`. We slightly corrupt the past inputs to the dynamics
> model to signal level `τ_ctx = 0.1` to make the model robust to small imperfections in its
> generations." — Sec 3.2

Also (Sec 3.2 header): *"It is trained using a shortcut forcing objective to enable fast interactive
inference with `K=4` forward passes per generated frame."* And Fig 8 shows shortcut forcing holding
FVD roughly flat as sampling steps drop, whereas diffusion forcing degrades.

**Our replication.** `DynamicsTransformer.rollout` (`dynamics.py:823-899`): one KV cache per temporal
block (`:860-861`); **prefill** context near-clean `τ ~ U(0.9,1)` with `append=True` to fill caches
(`:864-867`); **per future frame**, start from Gaussian noise, denoise in `num_steps` Euler steps with
`append=False` (transient, must not pollute cache), then commit the clean frame at τ=1 with
`append=True` (`:869-895`). Shortcut step size `d = max(1, k_max // num_steps)` (`:871`). The
**base/job-124 model rolls out at d=1** (`num_steps == k_max`, e.g. eval passes `num_steps=16,
k_max=16`, `train_dynamics.py:875`; `rollout_check.py:108`), i.e. the many-step regime — the K=4
shortcut path is meaningful only after Phase-B distillation. Numerical equivalence of cached vs.
uncached temporal forward on an empty cache (`layers.py:687-690`); token-build + output-projection are
byte-shared between `forward` and `rollout` (`dynamics.py:698-704, 738-741`).

**Verdict: MATCH (autoregressive KV-cache rollout, near-clean context).** We run at **d=1** rather than
the paper's headline **K=4**, by design (shortcut deferred) — legitimate, since d=1 is the many-step
regime the shortcut merely accelerates.

## 1.14 FVD ablation cascade — Table 2 (Sec 4.4), every row

> **Table 2 caption:** "Cascade of model design choices. Dreamer 4 is based on a shortcut forcing
> objective and an efficient transformer architecture, combining a range of known techniques to
> achieve accurate and fast interleaved generation. Starting from a naive diffusion forcing
> transformer with `N_z = 64` spatial tokens and `K = 64` sampling steps, we apply the objective and
> architecture modifications, and increase the number of spatial tokens once feasible. Inference
> speed measured on a single H100 GPU."

Columns: **Train step (s)** · **Inference FPS (↑)** · **Quality FVD (↓)**. Verbatim rows:

| Row (cumulative) | Train step (s) | FPS ↑ | FVD ↓ |
|---|---|---|---|
| Diffusion Forcing Transformer | 9.8 | 0.8 | **306** |
| + Fewer sampling steps (K=4) | 9.8 | 9.1 | **875** |
| + Shortcut model | 9.8 | 9.1 | **329** |
| + X-Prediction | 9.8 | 9.1 | **326** |
| + X-Loss | 9.8 | 9.1 | **151** |
| + Ramp weight | 9.8 | 9.1 | **102** |
| + Alternating batch lengths | 1.5 | 9.1 | **80** |
| + Long context every 4 layers | 0.6 | 18.9 | **70** |
| + GQA | 0.5 | 23.2 | **71** |
| + Time factorized long context | 0.4 | 30.1 | **91** |
| + Register tokens | 0.5 | 28.9 | **91** |
| + More spatial tokens (N_z=128) | 0.8 | 25.7 | **66** |
| + More spatial tokens (N_z=256) | 1.7 | 21.4 | **57** |

**Load-bearing reading of Table 2 (for the "is shortcut required" question).** The **K=64 naive
diffusion-forcing transformer is already a good model** (FVD 306, but slow at 0.8 FPS). Collapse to
FVD 875 happens **only** when dropping to K=4 *without* shortcut; the shortcut model exists to
**restore quality at K=4 for real-time speed** (875 → 329), not to make multi-step rollouts possible.
The big *quality* wins are the **objective** changes that are independent of shortcut and that we
already replicate: **X-Loss (326 → 151)** and **Ramp weight (151 → 102)**. Therefore omitting shortcut
while rolling out at **d=1** (job 124) is **not** a departure that would by itself wreck rollouts.

**Our replication of the cascade.** We match: X-Prediction (§1.6), X-Loss/ramp (§1.7–1.8), alternating
lengths (§1.12), temporal-every-4 + GQA (§1.10), register tokens (§1.11). We **defer** shortcut/K=4
(§1.3). Our spatial tokens `N_z = 256` (16×16 grid, `dynamics.py:410, 971`) matches the final row.

## 1.15 Standard-transformer specifics (Sec 3.4)

> "We start from a standard transformer with pre-layer RMSNorm, RoPE, and SwiGLU. We employ QKNorm
> and attention logit soft capping to increase training stability." — Sec 3.4

**Our replication.** RMSNorm everywhere (eps 1e-6, Gemma-2; `layers.py:23-44`); RoPE — 2D over the 256
latent tokens spatially (`layers.py:458-484`), 1D temporally (`:518-531`); SwiGLU FFN hidden
`int(dim·8/3)`→mult-of-64 = 2048 (`layers.py:101-119`); QKNorm per-head, scale forced to 1.0
(`:47-80, 322-325`); soft-cap 50 on logits (`:83-98, 438-450`). Register tokens present (§1.11).

**Deviations noted in `DYNAMICS_REVIEW.md` §6.6:** RMSNorm eps hardcoded 1e-6 (not a config knob),
QKNorm scale hardcoded 1.0. Both deliberate Gemma-2 conventions.

**Verdict: MATCH** (all named components present).

## 1.16 Representation / tokenizer assumptions the world model relies on (Sec 3.1)

> "The tokenizer compresses raw video into a sequence of continuous representations for the dynamics
> model to consume and generate. It consists of an encoder and a decoder with a bottleneck in
> between. Both components are causal in time, enabling temporal compression while maintaining the
> ability to decode frame by frame for interactive inference." — Sec 3.1

> "After applying the encoder, the representations are read out of the latent tokens using a linear
> projection to a smaller channel dimension followed by a `tanh` activation." — Sec 3.1

> "We train the tokenizer using a straightforward reconstruction objective, consisting of mean
> squared error and LPIPS loss. To simplify weighing the two loss terms, we employ loss
> normalization as explained later." — Sec 3.1

> `L(θ) = L_MSE(θ) + 0.2 L_LPIPS(θ)`  — **Eq 5**

> "We drop out input patches to the encoder to improve its representations using masked autoencoding.
> The dropout probability is randomized across images as `p ∼ U(0, 0.9)`. … We found MAE training to
> improve the spatial consistency of videos generated by the dynamics model." — Sec 3.1

**Our replication (tokenizer v7, frozen for dynamics).** Transformer image tokenizer, **512 latents ×
16 dim** bottleneck (`pretokenize_replay_v7.py:5-6`); per-frame latent `(512,16)` folded to a **16×16
grid of 32 channels** `(32,16,16)` (`pretokenize_replay_v7.py:86-97`), which is the 256-spatial-token
layout the dynamics model consumes (`latent_dim=32`, spatial 16×16). Frozen during dynamics
(`DYNAMICS_REVIEW.md` §1, §2.2). The dynamics input projection is `Linear(32→768)` per latent token
(`dynamics.py:435, 749-750`).

**Deviations / notes.** ahriuwu's tokenizer is a **separate v7 checkpoint**, not re-derived here; the
paper's causal-in-time temporal-compression tokenizer, its `tanh` bottleneck read-out, MSE+0.2·LPIPS
objective (Eq 5), and MAE patch-dropout `p∼U(0,0.9)` are **tokenizer-stage** properties not verified in
this dynamics review (dynamics consumes the frozen latents regardless). The 512→256-spatial reshape
matches the paper's spatial-token treatment.

**Verdict: MATCH (dynamics-side contract: continuous per-frame latents on a spatial grid, frozen
tokenizer).** Tokenizer-internal recipe (Eq 5, `tanh`, MAE, causal temporal compression) is
**out-of-scope / CAN'T-VERIFY here** — it's Phase-0, not the dynamics run under review.

## 1.17 Imagination-training hooks the world model must support (Sec 3.3) — informational

Not part of job 124, but the paper's world model is built to accept agent tokens later:

> "…we insert agent tokens as an additional modality into the world model transformer and interleave
> it with the image representations, actions, and register tokens. … While the agent tokens attend to
> themselves and all other modalities, no other modalities can attend back to the agent tokens. This
> is crucial for avoiding causal confusion of the world model—its future predictions can only be
> directly influenced by actions, not by the current task." — Sec 3.3

Reward/value/policy heads (Eqs 9–11): MTP length `L=8` (Eq 9); symexp-twohot reward and value heads;
`γ=0.997` λ-returns (Eq 10); PMPO policy loss with `α=0.5, β=0.3` (Eq 11).

**Our replication.** `use_agent_tokens` path exists but is **OFF** for job 124: `AgentCrossAttention`
(agent→z cross-attn, z cannot see agent, matching the paper's one-way rule) + causal temporal
self-attn (`dynamics.py:137-321`); heads in `heads.py`. `DYNAMICS_REVIEW.md` §3.6, §6.5.

**Verdict: DEFERRED** (Phase 2+). The one-way agent-token masking rule is implemented consistently with
the paper. Reward/value twohot ranges deviate (±3 vs Dreamer's ±20; `DYNAMICS_REVIEW.md` §6.5) but are
unused by the dynamics x-pred loss.

---

# PART 2 — Our replication, full detail (self-contained)

Condensed from `DYNAMICS_REVIEW.md`; every claim `file:line`-cited there.

## 2.1 Goal & pipeline
DreamerV4-style **latent world model for League of Legends**, single champion **Garen TOP**. Predicts
future latent frames from past latents + actions. Phases: (0) frozen transformer tokenizer "v7"; (1)
**dynamics — this doc, currently training as Slurm job 124**; (2) agent BC; (3) imagination RL.
`dynamics.py:1-15` cites Dreamer 4. Proof-of-concept scale: ~125–146 replay matches.

## 2.2 Data & latents
Source `/srv/nfs/datasets/lol_replays_16_9_772`: 147 matches, 352×352 PNG frames, 20 fps, per-frame
`labels.json` (champion stats, inventory, visible heroes, action, cursor, movement)
(`DYNAMICS_REVIEW.md` §2.1). Tokenizer v7 bottleneck **512×16**, folded to **(32,16,16)** dynamics
latents; one packed file per match `{"latents":(N,32,16,16) fp16, "frame_indices":(N,) int32}`
(`pretokenize_replay_v7.py:86-97, 219-221`). **Action space**: 9 binary keys `['Q','W','E','R','Flash',
'Ignite','AA','Recall','Stride']` + 2-D movement in `[0,1]` (`constants.py:14, 29`), parsed from
clicks/labels (`replay_dataset.py:317-427`). Sequences via `ReplayLatentSequenceDataset` over
contiguous frame runs, `VideoGroupedSampler` for LRU-cache-friendly ordering
(`replay_dataset.py:96, 205-245, 480-504`; `dataset.py:213-237`).

## 2.3 Model (medium, exact config, measured)
`create_dynamics("medium", latent_dim=32, use_actions=True, num_kv_heads=4, num_register_tokens=8,
soft_cap=50.0, use_qk_norm=True, gradient_checkpointing=True)` → **model_dim 768, 18 layers, 12 query /
4 KV heads, head_dim 64, spatial 16×16=256, temporal_every=4, max_seq_len 256, SwiGLU hidden 2048,
~114.6M params** (`dynamics.py:943-991`). Per-frame tokens `[256 latent(2D-RoPE) | 8 register | 1
action | 1 condition] = 266` (`dynamics.py:742-795`). Block stack
`[S S S T]×4 [S S]` = 14 spatial / 4 temporal (`dynamics.py:478-496`). τ/step conditioning injected
**twice** (additive to all tokens `:763-776` + appended token `:793-794`); discrete τ (64 levels),
step (7 sizes) (`:448-452`). RMSNorm/QKNorm/SwiGLU/soft-cap/2D+1D-RoPE per §1.10/§1.15. Output head
strips non-latent tokens, RMSNorm, `Linear(768→32)`, reshape `(B,T,32,16,16)` (`:816-821`). Manual
attention path (`allow_flex=False`, `dynamics.py:93`) for GC compatibility.

## 2.4 Objective & training config
**x-prediction diffusion forcing.** `z_τ = τ·z_0 + (1−τ)·ε` (`diffusion.py:62`); per-frame τ from the
context-heavy horizon schedule (§1.5, `diffusion.py:89-144`); loss `x_prediction_loss` = per-frame MSE
× `ramp_weight(τ)=0.9τ+0.1` (`diffusion.py:221-258`); **RMS-normalized** loss backpropagated
(`RunningRMS` decay 0.99, `returns.py:334-395`, `train_dynamics.py:984-988`). **independent_frames** at
**0.3** per-batch (whole-sequence diagonal mask, §1.9). **Stability:** grad-clip 1.0, NaN-skip guard on
non-finite grad-norm, DDP zero-loss tap over all params (`train_dynamics.py:784-818`). **Optim:** AdamW
`betas=(0.9,0.95)`, wd 0.1, 8-bit Adam if bitsandbytes present, **WSD** schedule lr 3e-4 / warmup 3000
/ no decay, **50 epochs** (`training.py:181-299`, `train_dynamics.py:1273`). **Batching:** alternating
128/256, batch 2/1, long-ratio 0.1, grad-accum 8/16 (eff. batch 16), bf16 autocast, single RTX 5080,
**shortcut off**, **compile off** (`slurm_dyn_train.sbatch`). **Evals every 200 steps:** teacher-forced
1-step denoising PSNR (best-ckpt metric) + free-running rollout PSNR (`train_dynamics.py:485-618,
865-884`).

## 2.5 Rollout
KV-cached autoregressive rollout (§1.13): prefill near-clean context → per future frame denoise from
noise in `num_steps` Euler steps (transient, `append=False`) → commit clean frame at τ=1
(`dynamics.py:823-899`). Base model runs **d=1** (`num_steps==k_max`). Eval harness `rollout_check.py`
picks the most dynamic window, runs teacher-forced 1-step PSNR + autoregressive rollout with real
recorded actions, and (with `--decode`) a 3-row GT/recon/dream video + per-frame pixel-PSNR plot
including a **persistence baseline** (hold last real frame) (`rollout_check.py:83-195`).

---

# PART 3 — Changes made today (2026-07-04)

## 3.1 SAMPLER FIX — committed `3f69de8` (INFERENCE-only; training was always correct)

**Bug.** The multi-step Euler denoiser in **both** `DynamicsTransformer.rollout` (`dynamics.py`,
rollout inner loop) **and** `DiffusionSchedule.sample` (`diffusion.py`) re-noised each intermediate
Euler step with the **frozen initial noise** (`noise0` / `z_noise`) instead of the **implied** noise
consistent with the current x-prediction:

`ε̂ = (z_t − τ·ẑ0) / max(1−τ, 1e-3)`  (DDIM / flow-matching update).

Because x-prediction changes each step, holding `ε` frozen makes the update **not** an Euler step of
the learned flow; error compounds with step count. Measured on a good checkpoint: **num_steps 1 → ~18
dB, 4 → ~16, 16 → ~12, 64 → ~7** (monotonic divergence; **1-step fine**). Since
`eval_rollout_psnr` calls the base model at **num_steps=16** (`train_dynamics.py:875`), a **competent
predictor** (1-step beats a persistence baseline on **22–24 / 24** frames on training clips) was
rendered as **garbage in every rollout eval** — i.e. the eval measured a broken sampler, not the model.

**Fix.** Switch both samplers to the **implied-noise (DDIM / flow-matching) Euler** update
(`ε̂` recomputed each step from `(z_t, τ, ẑ0)`), so multi-step denoising is a consistent integration
of the learned x-prediction flow. Training was **never** affected (it does a single forward per frame
with the diffusion-forcing schedule — no iterative sampler).

**Verification** (same checkpoint **step 18204**, same sequence): rollout-vs-persistence PSNR crossover
moved from **~+14 frames → ~+3 frames** at `num_steps=4`; at **`num_steps=1`** the dream **beats
persistence across all 20 frames** (pixel PSNR dynamic clip **24.6 → 19.2** dream vs **24.2 → 17.4**
persistence).

**How it was found.** Adversarial code review that **empirically falsified 6 hypotheses** —
non-causal attention, ignores-context, dead-actions, ramp-weight, independent_frames, KV-cache — via
perturbation + update-rule-swap tests, isolating the frozen-noise Euler update as the sole cause.

## 3.2 DEVIATIONS IDENTIFIED (not yet fixed — all training-side; require a job-124 restart or
apply to the next / cloud run)

1. **independent_frames unit mismatch (highest suspicion).** We apply a **whole-sequence diagonal
   temporal mask on 30% of *steps*** (`train_dynamics.py:741`, `dynamics.py:683-686`,
   `layers.py:234-235`); the paper treats **30% of *videos* as single images** (start-frame
   generation) while keeping the other 70% as full temporal sequences (Sec 4, §1.9 above). Our version
   trains the temporal layers to be a cross-frame no-op 30% of the time and gives them prediction
   gradient on only ~70% of steps → strong per-frame denoiser, weak forward predictor (the observed
   symptom). **Fix:** make 30% of *batch elements* length-1/start-frame items; never disable temporal
   context on multi-frame sequences.

2. **Repo-invented context-heavy τ schedule.** `sample_diffusion_forcing_timesteps`
   (`diffusion.py:89-144`) confines context to `τ∈[0.9,1]` (~10% of range) and makes deep-target τ
   near-deterministic in position, starving the pure-noise (τ≈0) regime that rollout runs in. Paper
   prescribes **uniform / logit-normal** (Eq 1) or the **uniform grid** (Eq 4). **Fix:** replace with
   i.i.d. uniform (or Eq-4 grid) per-frame τ, or at minimum widen the context band and de-correlate
   target τ from position.

3. **Ramp weight de-weights low-τ with no bootstrap to compensate (amplifies #2).** `w(τ)=0.9τ+0.1`
   exactly matches Eq 8 (`diffusion.py:202-218`), but the paper's low-τ down-weight is covered by the
   **bootstrap term** (Eq 7, present at low τ). With shortcut **off**, low-τ x-prediction is
   down-weighted to 0.1 with nothing filling the gap. **Fix:** enable shortcut, or temporarily flatten
   the ramp for the plain-DF base run.

4. **Rollout-eval step count.** Drop `eval_rollout_psnr` base **num_steps 16 → ~4**
   (`train_dynamics.py:875`) — **eval-only**, now especially safe post-sampler-fix (fewer steps = less
   accumulated integration error, closer to the intended shortcut regime), and it makes the live
   rollout metric track predictor quality rather than sampler length.

**Benign / deferred (do NOT chase for the current symptom):**
- **Double τ/step conditioning** (additive + appended token, §1.11) — superset of the paper; benign.
- **Batch-length == context-length** (§1.12) — violates the paper's "batch length > context length"
  length-generalization margin; minor.
- **x-space shortcut bootstrap** (§1.7) — deliberate bf16-precision deviation; **Phase-B only**
  (shortcut off in job 124).
- **Shortcut omission** (§1.3, §1.14) — by-design **speed** mechanism per Table 2; the K=64/d=1
  many-step regime we run is the paper's own strong baseline (FVD 306). **Not** the cause of the poor
  rollout.
