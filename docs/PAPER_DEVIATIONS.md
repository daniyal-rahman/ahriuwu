# PAPER_DEVIATIONS.md — where ahriuwu differs from Dreamer 4, and why

**Living reference. Base pass verified 2026-08-12 against the code on disk, the production
checkpoint args, and the paper text. Revised 2026-08-13** — sections carrying a dated
`UPDATE`/`verified` note were re-checked then; everything else still dates from the 08-12 pass.

**Scope note.** §1–§6 record *deviations from the paper*. **[§3A](#3a-assumptions-ledger--what-each-head-takes-on-faith)
records something the paper cannot help with: the modelling assumptions each head asserts on its
own** — no paper counterpart, so they can never show up as "deviations", but they decide whether the
design works. Read §3A before trusting any head's numbers.

**When code and logs disagree, trust the logs.** Resume semantics have changed under this repo more
than once (see §1.2), so arithmetic derived from today's source can misdescribe what a historical
run actually did. Two claims in this document were wrongly "corrected" from current code before the
logs were consulted.

**The paper.** Hafner\*, Yan\*, Lillicrap (2025), *Training Agents Inside of Scalable World
Models* (Dreamer 4), arXiv **2509.24527v1**. Repo copy: `Dreamerv4_paper.pdf`; extracted text:
`scratchpad/dreamer4_text.txt` (grep it with **python `re`**, not `grep` — encoding issues).
Section/equation numbers below are the paper's own.

## How to read this

Every paper claim carries a locatable anchor (section, equation, table, or a quoted phrase).
Claims made elsewhere in this repo that the paper does **not** support are marked
**UNVERIFIED** and collected in [§7](#7-unverified--fabricated-paper-claims-found-in-this-repo).
This matters: the repo has a documented history of invented paper claims (see §7).

| Tag | Meaning |
|---|---|
| **DELIBERATE** | We chose to differ; reason recorded. |
| **FORCED** | Compute, data, or domain forced it. |
| **ACCIDENTAL** | Drift, stale code, or a bug — nobody chose it. |
| **REVERTED** | Tried and undone. Kept here so it isn't re-litigated. |
| **NOT-A-DEVIATION** | Previously documented as a deviation; verification says we match. |

Impact = likely effect on final agent quality: **HIGH** / **MED** / **LOW** / **N/A** (stage not
yet trained).

---

## Summary table

| # | Stage | Deviation | Type | Impact |
|---|---|---|---|---|
| 1.1 | Tokenizer | **Tube masking** — one spatial mask reused across all 20 frames of a clip. Paper masks per image. | DELIBERATE | **HIGH** |
| 1.2 | Tokenizer | **Mask-ratio curriculum** (0→0.9 over 2 000 steps) + 10% forced-zero spike. Paper: flat `p~U(0,0.9)`. Measured mean input mask ≈ 0.1–0.2, never the paper regime. | ACCIDENTAL | **HIGH** |
| 1.3 | Tokenizer | Mask ratio drawn **once per micro-batch**, not per image. | ACCIDENTAL | LOW |
| 1.4 | Tokenizer | Resolution **352×352 → 484 patches** vs paper 360×640 → 960 patches (1.98× fewer tokens, 3.6× horizontal downsample). | FORCED | **HIGH** |
| 1.5 | Tokenizer | **206 M params** vs paper 400 M; **~0.8 epoch** of a ~54 h corpus vs 2 541 h on 256–1024 TPU-v5p. | FORCED | **HIGH** |
| 1.6 | Tokenizer | LPIPS on a random **16-of-20-frame subsample**; no such subsampling in the paper. | DELIBERATE | LOW |
| 1.7 | Tokenizer | `--exclude-blacked-regions` valid-mask (YT black-HUD fix). No paper equivalent. | DELIBERATE | LOW |
| 1.8 | Tokenizer | No GQA in the tokenizer. Paper only requires GQA in the *dynamics*. | NOT-A-DEVIATION | — |
| 1.9 | Tokenizer | Bottleneck **512×16 + tanh**, reshaped 16×16×32 downstream. | NOT-A-DEVIATION | — |
| 1.10 | Tokenizer | `256×32` bottleneck trial and its "+7 dB" claim. | REVERTED | — |
| 2.1 | Dynamics | **Shortcut forcing OFF.** Production ckpt `shortcut_forcing: False`. Plain diffusion forcing, rollout at K=64/d=1. | DELIBERATE (FORCED by VRAM) | MED |
| 2.2 | Dynamics | **τ ~ iid U(0,1) per frame, continuous.** Paper's *shortcut* τ is a discrete grid tied to `d` (Eq 4). Our τ matches the plain-DF reading (§2 "uniform distribution"). | NOT-A-DEVIATION *while shortcut is off* | LOW |
| 2.3 | Dynamics | Shortcut **bootstrap loss computed in x-space**, target = second half-step x-prediction. Paper Eq 7 is v-space with `(1−τ)²`, target `sg(b'+b'')/2`. Also `bootstrap_weight=10.0` and a clamp at 100. | DELIBERATE | N/A (unused) |
| 2.4 | Dynamics | τ + step-size injected **additively onto all tokens AND as an appended token**. Paper: appended token only. | DELIBERATE | LOW |
| 2.5 | Dynamics | Context length / batch length **128 & 256** with `max_seq_len == batch length`. Paper Minecraft: C=192, T₁=64 / T₂=256, and requires **batch length > context length**. | ACCIDENTAL | MED |
| 2.6 | Dynamics | **~115 M params** (dim 768, 18 layers) vs paper 1.6 B. `global_step 8775`, `epoch 0` — under one epoch. | FORCED | **HIGH** |
| 2.7 | Dynamics | **WSD schedule with `decay_steps=0`** (warmup → flat forever). Paper states **no** LR schedule at all. | DELIBERATE | LOW |
| 2.8 | Dynamics | Extra **`use_game_time` conditioning modality** (bucketed game clock + dropout). No paper equivalent. OFF in the production ckpt. | DELIBERATE | LOW |
| 2.9 | Dynamics | Latent dim 32 on a **fake** 16×16 spatial grid (512 perceiver latents are global readers; the grid is a reshape fiction). Paper's `N_z=256×32` is also a reshape, so shape matches; *spatial semantics* do not. | NOT-A-DEVIATION (shape) / caveat | MED |
| 2.10 | Dynamics | `holdout_videos: 0` — no held-out eval during dynamics training. | ACCIDENTAL | MED |
| 2.11 | Dynamics | τ **noised continuously** but **conditioned through a 64-bin lookup** — up to 1/64 signal-level ambiguity by construction. | DELIBERATE | LOW |
| 2.12 | Dynamics | `step_embed` rows for d>1 are **never trained** (shortcut off ⇒ `step_size` never passed ⇒ only index 0 gets gradient). | ACCIDENTAL consequence of 2.1 | MED (blocks 5.2) |
| 2.13 | Dynamics | Unlabeled/no-action masking is **movement-only**: ability embeddings are still summed in on `cursor_valid=False` frames. Paper: *"only the learned embedding is used."* | ACCIDENTAL | MED **if YT is re-admitted** |
| 2.14 | Dynamics | Dormant deviations inside the *shortcut path*: RMS loss normalization **disabled** when shortcut is on; `independent_frames` **never passed**; step size sampled **per-example** not per-frame. | ACCIDENTAL (dormant) | N/A now, HIGH at Phase B |
| 2.15 | Dynamics | Extra **`pixel_hud_loss`** term (HUD-masked pixel loss through the decoder) in the 168/179/s1prime lineage. Not in the paper; **not in the production ckpt**. | DELIBERATE | LOW |
| 2.16 | Dynamics | 30%-independent-frames applied to whole sequences per *step* instead of per *example*. | REVERTED (fixed `8212365`) | — |
| 2.17 | Dynamics | Horizon-ramped, context-heavy τ schedule. | REVERTED (fixed `8212365`) | — |
| 3.1 | Heads | Action space: **9 ability Bernoullis + 2 × 21-bin screen-position categoricals**. Paper: 23 keyboard binaries + 121-class foveated mouse categorical. | FORCED (domain) | MED |
| 3.2 | Heads | **Sticky "movement gate"** — a per-offset Bernoulli mixing "repeat previous bin" vs "fresh categorical". **No paper equivalent at all.** | DELIBERATE | **HIGH** |
| 3.3 | Heads | **n=0 BC term dropped** (label-leak fix). Paper Eq 9 sums n=0..L. Reward MTP keeps n=0. | DELIBERATE | LOW |
| 3.4 | Heads | Twohot buckets **±3 symlog**, 255 bins. Paper gives no range; Dreamer 3's is ±20. Audit measures **39/255 buckets used**. | DELIBERATE (mis-tuned) | MED |
| 3.5 | Heads | **`StateHead`** — aux supervised HP/level/visibility regression from agent tokens, weight 0.5. **Entirely ours.** | DELIBERATE | MED |
| 3.6 | Heads | `--ability-pos-weight` default **5.0**; every production run passes **1.0**. | REVERTED | — |
| 3.7 | Heads | MTP length 9 heads (L=8, n=0..8). | NOT-A-DEVIATION | — |
| 3.8 | Heads | Threshold-based greedy cast decoding at inference. | REVERTED (sampling is the live default) | — |
| 4.1 | Phase 2 | **Backbone frozen**; only 31.7 M agent-token blocks + heads train. Paper finetunes the whole world model. | DELIBERATE (FORCED) | **HIGH** |
| 4.2 | Phase 2 | **Video-prediction loss dropped.** Paper: *"we continue to apply the video prediction loss."* | DELIBERATE (follows from 4.1) | **HIGH** |
| 4.3 | Phase 2 | Latents noised only in the narrow band τ~U(0.9, 1.0), `d=1`. Paper reuses the full pretraining noising. | DELIBERATE | MED |
| 4.4 | Phase 2 | **Action-history dropout** p=0.15 on the *movement* channel only. No paper equivalent. | DELIBERATE | MED |
| 4.5 | Phase 2 | **No task conditioning** — `num_tasks=1`, `task_id` never passed, `task_embed` frozen at random init. Paper is multi-task with 20 tasks. | FORCED (single-task domain) | LOW |
| 4.6 | Phase 2 | **No 50/50 uniform-vs-relevant data mixture.** | ACCIDENTAL | MED |
| 4.7 | Phase 2 | Agent tokens are a separate residual stream; z never sees them. | NOT-A-DEVIATION | — |
| 5.1 | Phase 3 | **Never actually trained.** One 11-batch smoke run, 2026-07-31, degenerate (R=0, A⁺=100%, KL≈0). | FORCED | N/A |
| 5.2 | Phase 3 | **K=64 dreaming** (`--gen-steps 64` recommended) because the backbone never got the shortcut finetune. Paper dreams at K=4. O(H²) per rollout. | DELIBERATE (follows 2.1) | **HIGH** if run |
| 5.3 | Phase 3 | `λ = 0.95`, `horizon = 8`. **Paper states neither value.** | UNVERIFIED baseline | MED |
| 5.4 | Phase 3 | `continues = ones` — no terminal handling, no continue head. | DELIBERATE | LOW |
| 5.5 | Phase 3 | Bootstrap at the horizon uses `v_{T-1}` in place of `v_T`. | DELIBERATE | LOW |
| 5.6 | Phase 3 | Per-loss `RunningRMS` normalization of the PMPO and value losses. Not in Eq 11. | DELIBERATE | LOW |
| 5.7 | Phase 3 | **Gated policy heads cannot enter Phase 3** — `log_prob()` hard-raises; `PolicyHead` rebuilt without `movement_gate` then `load_state_dict(strict=True)`. Every current Phase-2 ckpt is gated. | ACCIDENTAL (blocking bug) | **HIGH** |
| 5.8 | Phase 3 | PMPO α=0.5, β=0.3, reverse KL, frozen prior, 1 rollout/context, no entropy bonus, no advantage normalization. | NOT-A-DEVIATION | — |
| 6.1 | Data | **49.4 h** action-labeled vs paper's 100 h-of-actions experiment. | FORCED | MED |
| 6.2 | Data | **~450 h YouTube EXCLUDED** from dynamics training vs paper's 2 441 h unlabeled. Unlabeled:labeled is **0:1** vs the paper's ~25:1. | DELIBERATE (contested) | **HIGH** |
| 6.3 | Data | **HUD-off replays**; live capture has a HUD. Train/deploy domain gap. | ACCIDENTAL | **HIGH** |
| 6.4 | Data | 352×352 squished from 1280×720 (aspect not preserved). | FORCED | MED |

---

## 1. Tokenizer

### What the paper specifies

| Item | Paper | Anchor |
|---|---|---|
| Objective | `L(θ) = L_MSE(θ) + 0.2·L_LPIPS(θ)` — **reconstruction only** | §3.1, **Eq 5** |
| Loss weighting | *"To simplify weighing the two loss terms, we employ loss normalization"*; all loss terms normalized by *"running estimates of their root-mean-square (RMS)"* | §3.1; §3 intro |
| Bottleneck | linear projection to a smaller channel dim **followed by a `tanh`**; `(N_b=512)×(D_b=16)`, reshaped to `(N_z=256)×32` for the dynamics | §3.1; **Appendix A** |
| Masking | *"The dropout probability is randomized **across images** as p ∼ U(0,0.9). Patches of **each image** are replaced with a learned embedding with this probability, so that the tokenizer is sometimes trained on the p=0 case used during inference."* | §3.1 |
| Resolution | 360×640, zero-padded to 384×640, patch 16×16 → **960 tokens**, 20 FPS | Appendix A |
| Scale | **400 M** params; 2 541 h; 256–1024 TPU-v5p | §4 |

There is **no** predictability regularizer, no latent smoothness term, no GAN, no KL, no VQ.
Any claim to the contrary is fabricated — see §7.

### What we do

Production tokenizer = **v7**, frozen. Config is `scripts/v7_train_args.sh` (single source of
truth, sourced by both launchers).

| Item | ahriuwu v7 | Verdict |
|---|---|---|
| Loss | `mse_norm + 0.2 * lpips_norm`, each term RMS-normalized separately (`scripts/train_transformer_tokenizer.py:523-529`; `RunningRMS` decay 0.99, `src/ahriuwu/models/returns.py:334-396`) | **MATCH** |
| Extra loss terms | **None.** No GAN/discriminator/KL/VQ/commitment/latent-reg/smoothness anywhere in `losses.py` or the trainer | **MATCH** |
| MSE domain | `--mse-on-full-frame` (v7 passes it). Code default is masked-patches-only (measured 20.4 dB vs 31.2 dB at m=0). Paper does not restrict MSE to masked patches, so full-frame is the faithful reading | **MATCH** |
| Bottleneck | `nn.Linear(1024→16)` then `torch.tanh` (`src/ahriuwu/models/transformer_tokenizer.py:465,483`), **512 latents × 16 dim**; folded to `(32,16,16)` in `scripts/pretokenize_replay_v7.py:96` | **MATCH** — bottleneck shape, tanh, and the 256×32 reshape all match Appendix A |
| Masking | see 1.1–1.3 below | **DEVIATE** |
| Resolution | 352×352, patch 16 → **484** patch tokens (`src/ahriuwu/data/dataset.py:27`; `transformer_tokenizer.py:599,636`), 20 FPS, T=20 clips, stride 16 | **DEVIATE** (1.4) |
| Scale | **206,090,000** params (training logs; docs elsewhere say 208.4 M — logs win), 6 000 optsteps ≈ **0.8 epoch** of ~54 h replay-only frames, single GPU, eff. batch 64 clips | **DEVIATE** (1.5) |
| Arch | RMSNorm(eps 1e-6) + SwiGLU(8/3, mult-64) + QKNorm(scale 1.0) + soft-cap 50 + 2D/1D RoPE; temporal attention every 4th layer (8+8 layers → temporal at idx 3, 7) | **MATCH** |
| Optimizer | AdamW fp32, betas (0.9, 0.999), lr 1e-4, wd 0.1, WSD warmup 500 / decay 1500, grad-clip 1.0, bf16 autocast, **no EMA** | see 2.7 |

#### 1.1 Tube masking — DELIBERATE, impact **HIGH**

`--tube-masking` is on (`v7_train_args.sh:40`) and is also the code default
(`transformer_tokenizer.py:713-769`): one random spatial mask per **batch element**, `expand`ed
identically across all T=20 frames. The paper masks **per image** (§3.1: *"Patches of **each
image** are replaced…"*). No paper support for tube masking exists — the phrase does not appear
in the paper.

**Why it matters.** `docs/TOKENIZER_REVIEW_2026-08-02.md:33` measured the consequence: a tube-masked
HP bar is unpredictable from *any* frame's context, so the loss-optimal output is a generic mean
bar — the observed artifact. Per-image masking would leave the same region visible in ~most frames
of a clip, making the reconstruction an interpolation rather than a hallucination.

**Reason recorded?** No rationale for choosing tube masking is recorded anywhere. Treat as an
unexamined default.

#### 1.2 Mask-ratio curriculum — ACCIDENTAL, impact **HIGH**

```
current_mask_max = mask_ratio_max * (global_step / mask_warmup_steps)   # linear ramp
mask_ratio = 0.0 with prob p_zero_mask(=0.1), else U(mask_ratio_min, current_mask_max)
```
(`scripts/train_transformer_tokenizer.py:479-493`). v7 uses `--mask-ratio-min 0.0
--mask-warmup-steps 2000`.

The paper has **no curriculum**: `p ~ U(0,0.9)` flat from step 0. Worse, the measured effect is
that v7 largely never reached the paper's masking regime — training logs show `Mask: 0.26
(max: 0.34)` around step 750, and `verify/VERIFICATION_REPORT.md` records mean input mask ≈ 0.1–0.2.
A `--reset-schedule` resume re-zeroes `global_step` and therefore **re-ramps the curriculum**
(`train_transformer_tokenizer.py:312-320` + `:480`).

**Exactly which runs re-ramped (verified 2026-08-13).** This matters, because a naive reading of the
code suggests the ramp runs once ever — `load_checkpoint` restores `global_step`
(`train_transformer_tokenizer.py:936`), so a clean resume continues the clock, and the lifetime mean
mask for a single uninterrupted 5 970-step run would be ≈ **0.34**, not 0.1–0.2. The measured 0.1–0.2
is nevertheless the right number, because of the resume path:

| launcher | `RESET_SCHEDULE` | effect on the curriculum |
|---|---|---|
| `v7_train_args.sh` (Slurm autoresume) | default **0** — strict opt-in | ramp runs once |
| `run_ddp_tok.sh` | default **1** (`:41`) | **re-ramps on every launch/resume** |
| `slurm_tok_train_v7_cont/_yt.sbatch` | on **first launch only** (seed) | re-ramps once per new run |

…and, decisively, a **historical bug**: the opt-in test used to be `[ -n "$RESET_SCHEDULE" ]`, which
is true for the string `"0"`, so **every resume wrongly reset `global_step`** until it was changed to
`[ "${RESET_SCHEDULE:-0}" = "1" ]` (`v7_train_args.sh:67-71`, comment records the fix). While that
bug was live, each requeue restarted the ramp from 0. v7 trained across many requeues.

**Rule: trust the logged mask values over any derivation from the current code**, because the code
that produced those logs had a different resume semantics. The continuation runs are worse still —
`tokenizer_v7_yt` died at `gs≈780`, i.e. **entirely inside the 2 000-step ramp** (mean mask ≈ 0.08).

Two consequences: (a) the shipped `tokenizer_v7_step5970_FINAL` is the *least* affected artifact and
may sit nearer 0.34; (b) `run_ddp_tok.sh` still defaults to re-ramping and is a live trap for any
future tokenizer run.

Net: **the paper's stated MAE benefit — *"MAE training improves the spatial consistency of videos
generated by the dynamics model"* (§3.1) — was largely not purchased.** This is the single most
under-appreciated tokenizer deviation.

The 10% forced-`p=0` spike is *consistent with* the paper's stated intent (*"so that the tokenizer
is sometimes trained on the p=0 case used during inference"*), which `U(0,0.9)` already achieves in
the limit; the explicit spike is an addition, not a contradiction.

#### 1.3 Mask ratio per micro-batch, not per image — ACCIDENTAL, impact LOW

`random.uniform(...)` is called once per dataloader batch (`:493`). At v7's `--batch-size 1` this
is one ratio per clip, i.e. per 20 images. Paper randomizes *"across images"*. Low impact because
batch size is 1, but it does not survive a batch-size increase.

#### 1.4 Resolution — FORCED, impact **HIGH**

352×352 → 484 patch tokens vs the paper's 384×640 → 960. `docs/DATA_AUDIT_2026-08-12.md:611`
measures the real cost as **total resolution**, not aspect: 1280→352 is a 3.64× horizontal
downsample. `docs/EXPERT_REVIEW_2026-08-02.md:1b` argues this is why HP bars are illegible; the
tokenizer review counters that at 352² the bars are still human-legible in GT and the binding
constraint is the objective's area weighting (`TOKENIZER_REVIEW:35`). Both agree the tokenizer
cannot see minion HP; they disagree on the mechanism.

#### 1.5 Scale and training budget — FORCED, impact **HIGH**

206 M vs 400 M params; ~0.8 epoch of ~54 h vs 2 541 h. Measured plateau
(`TOKENIZER_REVIEW:123-126`): a 9-day continuation bought **+0.28 dB**; effective rank of the
bottleneck is **≈31 of 8 192** dims, i.e. the bottleneck is nowhere near full — *"the plateau is
objective/data-limited, not capacity-limited."*

#### 1.6 LPIPS frame subsample — DELIBERATE, impact LOW

`--lpips-frame-subsample 16` picks 16 of the clip's 20 frames per step
(`src/ahriuwu/models/losses.py:340`). Compute saving; no paper equivalent; unbiased in expectation.

#### 1.7 Blacked-region exclusion — DELIBERATE, impact LOW

`--exclude-blacked-regions` (default on) builds a `valid_mask` excluding pixels ≤ 0.02 across the
whole clip, applied to MSE as a normalized masked mean and to LPIPS by substituting `pred := target`
in excluded pixels (`losses.py:268,290-291,331-333`). This is the fix for the YouTube black-HUD
rectangle, which otherwise collapsed edge quality by −14.5 dB. It is only wired on the full-frame
MSE path. Domain-specific; no paper analogue.

#### 1.8, 1.9 — NOT deviations

Covered by the table above and listed only so they stop being re-raised:
- **1.8 No GQA in the tokenizer.** §3.4 scopes GQA to *"all attention layers in the **dynamics**"*.
  The tokenizer runs plain MHA. Correct.
- **1.9 Bottleneck 512×16 + tanh.** Exactly Appendix A, including the 16×16×32 reshape done
  downstream in `pretokenize_replay_v7.py`. See 2.9 for the one caveat that *does* matter.

#### 1.10 The 256×32 bottleneck trial — REVERTED

Ran as `slurm/slurm_v7_trial.sbatch` (3 000-optstep health check, `--latent-dim 32 --num-latents
256`). `slurm/slurm_tok_train_v7.sbatch:35` records *"user chose 512×16 despite Step-5 overfit
showing 256×32 was ~7 dB better at matched params"*. **Both halves of that note are now retracted**
by `docs/TOKENIZER_REVIEW_2026-08-02.md:114-122`:

1. 512×16 **is** the paper shape (Appendix A) — the "faithfulness override" framing was backwards.
2. The "+7 dB" was an overfit-regime artifact; matched-sample re-analysis gives ~**+1 dB** early
   training (25.75 vs 24.8 dB at ~96–98 k samples).

Do not resurrect this. The evidence-backed levers are **data** and **objective**, not shape.

---

## 2. Dynamics / world model

### What the paper specifies

| Item | Paper | Anchor |
|---|---|---|
| Objective | Shortcut forcing = diffusion forcing + shortcut models, **x-prediction**, x-space flow loss | §3.2, **Eq 6–7** |
| Flow τ | *"typically sampled from a uniform distribution or a logit-normal distribution"* | §2, after Eq 1 |
| Shortcut τ/d | `d ∼ 1/U({1,2,4,…,K_max})`, `τ ∼ U({0, 1/d, …, 1−1/d})` — a **discrete grid tied to d** | §2, **Eq 4** |
| Bootstrap | v-space, `(1−τ)²` multiplier, target `sg(b′+b″)/2` | **Eq 7** + footnote ∗ |
| Ramp weight | `w(τ) = 0.9τ + 0.1` | **Eq 8** |
| τ/d conditioning | *"a **single token** for the shortcut signal level and step size … discrete embedding lookup and concatenate their channels"* | §3.2 |
| Actions | per-component encodings summed with a learned embedding; unlabeled videos use *"only the learned embedding"* | §3.2 |
| Inference | **K=4** steps, `d=1/4`; past context corrupted to `τ_ctx = 0.1` | §3.2 |
| Architecture | pre-layer RMSNorm, RoPE, SwiGLU, QKNorm, attention soft capping; space-only + time-only layers; temporal attention **every 4 layers**; **GQA on all dynamics attention**; `S_r` register tokens | §3.4 |
| Sequence | Minecraft: `N_z=256`, **C=192**, **T₁=64 / T₂=256**; robotics & kitchens: `N_z=512`, C=96, T₁=32 / T₂=128. *"batch lengths need to be longer than the context length"* | §4; Appendix A; §3.4 |
| Start frames | *"we treat **30% of the videos in the batch** as separate images"* | §4 |
| Scale | **1.6 B** params | §4 |
| LR schedule | **not specified anywhere in the paper** | — |

### What we do

Production checkpoint: `rollout_stage/desktop_resume_8775_stripped.pt` (a.k.a. `dynamics_accel`
gs8775). Its embedded `args` and `model_config` are the ground truth below.

```
model_config: size_preset medium, model_dim 768, num_layers 18, num_heads 12, num_kv_heads 4,
              latent_dim 32, spatial_size 16, num_register_tokens 8, k_max 64, soft_cap 50.0,
              use_qk_norm True, use_actions True, use_agent_tokens False, use_game_time False
args:         shortcut_forcing FALSE, lr 3e-4, lr_schedule wsd, warmup_steps 3000, decay_steps 0,
              weight_decay 0.1, adam_betas [0.9, 0.999], use_8bit_adam True,
              alternating_lengths True, seq_len_short 128, seq_len_long 256, long_ratio 0.1,
              batch_size_short 2, batch_size_long 1, grad_accum 2/4,
              independent_frame_ratio 0.3, holdout_videos 0
global_step:  8775   epoch: 0   params: 114,854,528
```

Decoded: 18 blocks, `head_dim 64`, **GQA 12 query / 4 KV heads**, temporal attention at layer indices
**3, 7, 11, 15** (4 temporal / 14 spatial — the stack ends on spatial layers), per-frame tokens
`[256 latent (2D RoPE) | 8 register | 1 action | 1 condition] = 266`, SwiGLU hidden 2048, manual
attention path (`allow_flex=False`, no SDPA/flash), `--no-compile`, bf16 autocast with a no-op
GradScaler, grad-clip 1.0 with a non-finite skip gate, no EMA.

The saved accumulation values are **post-division** — `train_dynamics.py:1262-1267` divides by
`WORLD_SIZE` at startup, so `2/4` implies configured `8/16` on **world_size 4** (the 4×4090 Vast
box). Effective batch: short = 4×2×2 = **16 sequences × 128 frames**; long = 4×1×4 = 16 × 256.

#### 2.1 Shortcut forcing OFF — DELIBERATE (VRAM-FORCED), impact MED

`shortcut_forcing: False` in the shipped checkpoint. The whole `ShortcutForcing` machinery exists
(`src/ahriuwu/models/diffusion.py:263-535`) and `scripts/finetune_shortcut.py` is the intended
Phase-B distillation, never run. Reason recorded in `slurm/slurm_dyn_train.sbatch:29-32`: shortcut's
3-forward-pass steps with gradient-checkpointing disabled OOM `medium@128` on the 5080.

**Why the impact is only MED, not HIGH.** Paper **Table 2** shows the naive diffusion-forcing
transformer at K=64 is already a strong model (FVD **306**); the collapse to FVD 875 happens only
when you drop to K=4 *without* the shortcut. Shortcut restores quality **at K=4 for real-time
speed** (875 → 329). The big *quality* wins in the cascade — X-Loss (326→151) and ramp weight
(151→102) — are shortcut-independent and we have both. Rolling out at d=1 is the paper's own strong
baseline.

**Verified, not assumed.** `shortcut_forcing: False` in **all five** dynamics checkpoints on disk
(`rollout_stage/{dynamics_accel_latest, desktop_resume_8775, dynamics_168_latest}.pt`,
`scratchpad/{dyn179_gs6120, dyn_s1prime_gs7550_backbone}.pt`). No launcher in `scripts/dyn_train_args*.sh`
or `slurm/slurm_dyn_*.sbatch` contains the string `--shortcut-forcing`. `finetune_shortcut.py` writes
`dynamics_shortcut_{best,final}.pt`; **neither file exists anywhere in the repo.**

**Where it does bite: Phase 3** (5.2), and it leaves `step_embed` rows 1–6 untrained (2.12).

**One more gate to know about:** even with `--shortcut-forcing` on, the shortcut loss is skipped on
long (T=256) batches — `use_shortcut = ts.shortcut is not None and not use_long`
(`train_dynamics.py:853`).

#### 2.2 τ schedule — NOT-A-DEVIATION while shortcut is off, impact LOW

`sample_diffusion_forcing_timesteps` returns per-frame iid `U(tau_min, 1)`, `(B,T)`
(`src/ahriuwu/models/diffusion.py:89-124`). This matches the paper's plain flow-matching law
(§2: *"typically sampled from a uniform distribution"*) and diffusion forcing's per-timestep
requirement. It does **not** match Eq 4's shortcut grid — but Eq 4 only applies when the shortcut
bootstrap is active, which it is not.

Caveat noted in `DYNAMICS_VS_PAPER.md` §1.8 and still live: the ramp weight down-weights low-τ to
0.1, and in the paper that region is carried by the **bootstrap term** (which lives at low τ).
With shortcut off, low-τ x-prediction is down-weighted with nothing filling the gap.

#### 2.3 Bootstrap in x-space — DELIBERATE, impact N/A (path unused)

Paper Eq 7 for `d > d_min`:
`L = (1−τ)² ‖ (ẑ₁ − z̃)/(1−τ) − sg(b′+b″)/2 ‖²`

Ours (`diffusion.py:489-514`):
```python
x_diff = z_pred - z_target.detach()        # z_target = the SECOND half-step's x-prediction
x_mse  = (x_diff ** 2).mean(dim=(-3,-2,-1)).clamp(max=100.0)
loss_boot = (x_mse * ramp_weight(tau_idx)).mean()
```
Two differences, not one:
1. **Space.** x-space instead of v-space with `(1−τ)²`. Reason recorded in-code: the paper's
   `÷(1−τ)` followed by `×(1−τ)²` loses precision in bf16, and when teacher ≈ student the velocity
   difference collapses to zero.
2. **Target.** The paper averages the two half-step velocities; we use only `z_target`, the second
   half-step's x-output. `avg_velocity` is computed at `:483` and then **never used** — a dead
   variable. This looks like unintended drift rather than a decision.

`bootstrap_weight=10.0` compensates for the smaller deterministic-target MSE. `x_mse.clamp(max=100)`
and a NaN-skip are stability guards with no paper analogue. Re-derive this before any shortcut
finetune; it has never been validated at production scale.

#### 2.4 Double τ/step conditioning — DELIBERATE, impact LOW

Per-frame token layout is `[256 latent | 8 register | 1 action | 1 condition] = 266`. The condition
token is `Linear(concat(tau_embed, step_embed))` — faithful to §3.2. **In addition**, τ and step
embeddings are added to *every* token:

```python
# dynamics.py:781-783
# tau + step_size injected additively to all tokens (strong conditioning
# signal) and also as an appended token below.
tau_emb, step_emb = self._embed_tau_step(tau, step_size, B, T)
x = x + tau_emb.unsqueeze(2) + step_emb.unsqueeze(2)
```

It is a **plain additive bias** — not adaLN, not modulation, no scale/shift, no gating. The additive
term lands on latent + register + action tokens but not on the cond token itself.

**Reason recorded, in git**: added as `3eba0e1` *"Add additive conditioning to fix tau gradient
bottleneck"*, reverted in `eb38863`, re-added in `48541ba`, split per-signal in `7ae518d`. So this
was tried, undone, and deliberately restored. Superset of the paper; benign-to-helpful; keep, but
know it is ours.

#### 2.5 Context length / batch length — ACCIDENTAL, impact MED

We train `--alternating-lengths --seq-len-short 128 --seq-len-long 256 --long-ratio 0.1`, with
`max_seq_len = 256`. The alternating scheme is faithful (§3.4). But:

- Paper Minecraft is C=192 with T₁=64 / T₂=256; ours is effectively **C = batch length**.
- §3.4 explicitly requires *"batch lengths … longer than the context length of the model to prevent
  the transformer from overfitting to always seeing a start frame at the beginning of its context,
  enabling length generalization to arbitrary generation lengths."* We violate this. Expect weaker
  length generalization in long rollouts — which is the observed symptom (dreams hold ~10 frames).
- No long-only finetune phase was run either.

Note that 128/256 frames at 20 fps = **6.4 s / 12.8 s**, comparable to the paper's 9.6 s context.
The problem is the missing margin, not the absolute length.

#### 2.6 Model size and training budget — FORCED, impact **HIGH**

**115 M vs 1.6 B**, and the shipped checkpoint is at `epoch 0, global_step 8775` — i.e. **less than
one pass over the corpus**. Nothing else on this list plausibly outweighs this.

#### 2.7 LR schedule — DELIBERATE, impact LOW; **the "paper cosine" framing is UNVERIFIED**

We use WSD with `decay_steps=0` (warmup 3 000 → flat forever). `08fc474` records the switch:
*"slurm v6: constant LR (WSD, decay-steps=0) — drop cosine"*.

**The paper specifies no learning rate, no optimizer, no schedule, no warmup, no weight decay, no
betas, and no EMA.** A python regex sweep over the full extracted text returns **zero** hits for
`cosine`, `learning rate`, `warmup`, `weight decay`; the only `Adam` hits are author names in the
bibliography. So "WSD vs the paper's cosine" is a false contrast — see §7.

#### 2.8 `use_game_time` conditioning — DELIBERATE, impact LOW

`dynamics.py` supports a bucketed game-clock embedding (`game_time_bucket_seconds 30.0`,
`game_time_num_buckets 120`, `gt_dropout 0.1`) added to all tokens. No paper equivalent. **OFF in the
production checkpoint** (`use_game_time: False`), so currently inert.

#### 2.9 The 16×16 grid is a reshape fiction — caveat, impact MED

Paper Appendix A also reshapes (512×16 → 256×32), so the *shape* matches exactly. But
`docs/TOKENIZER_REVIEW_2026-08-02.md:23` measured that our 512 latents are **global perceiver
readers with no spatial locality** — *"the 16×16×32 grid is a reshape fiction — no latent-space
region masking possible."* Consequence: the dynamics' **2D spatial RoPE and spatial attention are
applied over an axis with no spatial meaning.** The paper never claims its bottleneck latents are
spatially localized either, so this is not strictly a deviation — but it is a load-bearing
assumption the repo's architecture makes and the data does not support.

#### 2.10 No held-out eval — ACCIDENTAL, impact MED

`holdout_videos: 0`. `docs/DATA_AUDIT_2026-08-12.md` finding 3: *"No held-out set during training;
dynamics 'eval' is a training batch."*

#### 2.11 Continuous τ, quantized τ conditioning — DELIBERATE, impact LOW

The noise uses the exact continuous τ, but the conditioning quantizes it:

```python
# dynamics.py:595
tau_idx = (tau * self.k_max).long().clamp(0, self.num_tau_levels - 1)   # 64 bins
```

so the network is told `floor(τ·64)/64` while the input was corrupted at exact τ. The docstring
(`dynamics.py:590-593`) calls this intentional, citing the paper's *"discrete signal levels"*. That
citation is *nearly* right: the paper's τ genuinely **is** discrete in the shortcut regime (Eq 4
samples it on a grid), so there is no ambiguity there. Ours samples continuously and then rounds,
which the paper never does. Up to 1/64 signal-level noise in the conditioning. Small, but it is a
deviation, not a match.

#### 2.12 `step_embed` rows d>1 are untrained — ACCIDENTAL consequence, impact MED

`num_step_sizes = log2(k_max)+1 = 7` embeddings for d ∈ {1,2,4,…,64}. With shortcut off, the standard
path never passes `step_size`, so `step_idx` is always 0 and **only row 0 receives gradient**. Rows
1–6 are still at init in `desktop_resume_8775.pt`.

**Consequence:** the shipped model does not merely "lack shortcut training" — its step-size
conditioning is *uninitialized*. Any future `finetune_shortcut.py` run starts those 6 rows cold.
Budget for that.

#### 2.13 Partial no-action masking — ACCIDENTAL, impact MED **if YT is re-admitted**

Paper §3.2: *"When training unlabeled videos, **only** the learned embedding is used."*

Two granularities exist in `dynamics.py`:
- **Whole-batch** (`actions is None` → `no_action_embed`, `:777-778`) — faithful. This is the path
  taken by every unlabeled run (154/168/178/179/s1prime).
- **Per-frame** `cursor_valid` mask (`:566-576`) — swaps `no_action_embed` in for the **movement**
  embedding only; the 9 ability-key embeddings are **still summed in** (`:581-583`). So a frame
  marked "no action" still carries ability information.

This is exactly the path that mixed labeled/unlabeled training uses
(`scripts/dyn_train_args_action.sh` + `scripts/gen_yt_placeholder_labels.py`, where YT clips get
placeholder labels parsing to `cursor_valid=False`). It is also the path Phase-2 action-history
dropout (4.4) rides on. Currently inert because gs8775 is replays-only with actions on — but this is
a live bug the moment §6.2 is acted on.

#### 2.14 Dormant deviations inside the shortcut path — ACCIDENTAL, impact N/A now / **HIGH at Phase B**

Three things are wrong in `_forward_shortcut` and will bite the moment shortcut is enabled:

1. **RMS loss normalization is turned off**: `rms_dict = None if args.shortcut_forcing else {...}`
   (`scripts/train_dynamics.py:1536`). The shortcut path returns its raw combined loss. The paper
   normalizes **all** loss terms by running RMS (§3 intro) — precisely so a two-term objective can be
   balanced. Enabling shortcut silently deletes that.
2. **`independent_frames` is never passed** (`train_dynamics.py:1058-1086`), so shortcut steps would
   run 100% full-temporal-context — the opposite of 2.16's fix.
3. **Step size is sampled per-example, not per-frame** (`diffusion.py:286-305`). Paper Eq 6 declares
   `τ, d ∈ [0,1]^T`, i.e. **both** vary per timestep.

Also: `train_dynamics.py:1079` computes `z_tau` and discards it — `ShortcutForcing.compute_loss`
re-noises internally with a *different* random draw (`diffusion.py:416`). Wasted work, and the two
noisings are inconsistent.

#### 2.15 `pixel_hud_loss` — DELIBERATE, impact LOW

The 168 / 179 / s1prime lineage adds a second loss term: a HUD-masked pixel loss computed by decoding
the predicted latents through the frozen tokenizer. Introduced in `79d90cc` to kill the
"black-HUD shortcut". No paper equivalent; the paper's dynamics loss is purely in latent space.
**Not present in the production `gs8775` checkpoint** (its `rms_state` has only `x_pred`).

#### Inference regime (matches the paper) + a `tau_ctx` naming footgun

`DynamicsTransformer.rollout` (`dynamics.py:831-907`) reproduces §3.2 faithfully: context corrupted
to `ctx_tau ~ U(1−tau_ctx, 1)` with **`tau_ctx = 0.1`** (a *width*, matching the paper's stated 0.1),
each new frame denoised from pure noise via the implied-noise Euler helper, then committed clean at
τ=1 into the temporal KV cache. **The only inference deviation is K** (2.1 / 5.2): in-training rollout
eval runs `num_steps=16, k_max=16 ⇒ d=1`; the README prescribes K=64 for this lineage.

**Footgun:** `tau_ctx` means two opposite things in this codebase.
- `dynamics.rollout(tau_ctx=0.1)` — a **width**: `U(1−0.1, 1)`.
- `train_agent_finetune.py --tau-ctx 0.9` and `agent_infer.py` — a **floor**: `U(0.9, 1)`.

Same distribution, inverted parameterization. Reading `0.1` and `0.9` as the same thing is correct;
reading either as "the paper's τ_ctx" without checking which file you're in is not.

#### 2.16 / 2.17 REVERTED (both fixed in `8212365`)

Two deviations catalogued at length in `DYNAMICS_VS_PAPER.md` §1.5 and §1.9 have since been fixed;
**that document is stale on these two points.**

- **Independent frames.** Was: one boolean per micro-batch, applying a whole-sequence diagonal
  temporal mask on 30% of *steps*. Now: `independent_frames = torch.rand(B) < 0.3` — a per-**example**
  bool (`scripts/train_dynamics.py:1140`; `dynamics.py:112-113` docstring now cites *"the paper's
  30%-of-videos-as-images setup"*). Matches §4.
- **τ schedule.** Was: a repo-invented horizon ramp confining context to τ∈[0.9,1]. Now: iid
  `U(0,1)` per frame (`diffusion.py:89-124`, with the old scheme's failure documented in the
  docstring and `tests/test_diffusion_forcing_schedule.py`).

The gs8775 checkpoint post-dates both fixes.

---

## 3. Heads

*The developer specifically asked about this section.* All four heads live in
`src/ahriuwu/models/heads.py`.

### What the paper specifies

| Item | Paper | Anchor |
|---|---|---|
| Head form | *"small MLPs with one output layer per MTP distance"* | §3.3 |
| MTP | `L = 8`, loss sums `n = 0..L` for **both** actions and rewards → 9 terms | §3.3, **Eq 9** |
| Reward head | *"symexp twohot output"*, *"Following Dreamer 3"* | §3.3 |
| Value head | symexp twohot; initialized at Phase 3 | §3.3 |
| Policy head | *"categorical or vectorized binary distribution, depending on the action space"* | §3.3 |
| Action space (Minecraft) | *"23 binary distributions"* keyboard + *"a categorical with 121 classes using foveated discretization"* mouse | §4.1 |
| Twohot bucket range | **NOT STATED.** Only *"Following Dreamer 3"* | — |
| Aux state head | **does not exist** | — |
| Movement gate | **does not exist** | — |

### 3.1 Action space — FORCED, impact MED

| | Paper (Minecraft) | ahriuwu (LoL) |
|---|---|---|
| Discrete keys | 23 Bernoullis | **9 Bernoullis**: `Q W E R Flash Ignite AA Recall Stride` (`src/ahriuwu/constants.py:29`) |
| Pointer | 1 categorical, 121 classes (11×11 μ-law foveated **camera delta**) | **2 independent categoricals, 21 bins each** over absolute screen position ∈ [0,1]² (`heads.py:174-247`) |

Two structural differences beyond the count:

- **Absolute position vs relative delta.** The paper's mouse action is a camera *delta*; ours is an
  absolute screen *target*. Domain-correct (LoL is click-to-move) but it means our action is not
  translation-invariant and inherits the camera's frame of reference.
- **Linear bins, not foveated.** Paper: μ-law/foveated so small deltas get fine bins. Ours: uniform
  `linspace(0,1,21)`. `docs/DATA_AUDIT_2026-08-12.md` findings 5 and 13 measure the cost:
  **37.7% of genuine commands are quantized away** by the 21-bin grid (190 168 / 504 008), and the
  bins are **anisotropic** (64 px in x, 36 px in y) because the grid is square but the source frame
  is not.

### 3.2 The sticky movement gate — DELIBERATE, impact **HIGH**, **no paper equivalent**

```python
# heads.py:217-232 (abridged)
# Optional STICKY-CATEGORICAL movement (the action-model rewrite):
# humans issue ~2-5 discrete movement commands/s but the data is 20fps
# per-frame held targets, so ~77% of frames are "repeat the previous
# action" and a plain categorical mostly learns to copy. With the gate,
# movement is a MIXTURE: with prob (1-g) repeat the previous bin, with
# prob g draw a fresh bin from the categorical.
```

One gate logit per MTP offset. Log-prob is the mixture (`gated_movement_log_prob`, `heads.py:299-319`):
`transition → log g + log p_cat`; `hold → logaddexp(log(1−g), log g + log p_cat)`.

**Nothing in Dreamer 4 resembles this.** The paper models per-frame actions directly. The gate is
this repo's response to `docs/EXPERT_REVIEW_2026-08-02.md` §2(b) — *"restructure as gate + location"* —
and is the correct response to a real measured pathology (77% held frames, self-fed BC collapse to
1.8%).

**Why it's HIGH impact and needs watching:**
- It is the largest single architectural addition on this list, and it is **untested against the
  ungated baseline** on a held-out metric. This remains the single biggest open question about the
  design and is *not* answered by the 2026-08-13 run (which is gated-only).
- It **blocks Phase 3** (see 5.7).
- Its supervision (`movement_event`) came from a target measured as **47.5% camera drift, not
  commands** on the legacy `cursor` source (`docs/DATA_AUDIT_2026-08-12.md` finding 1 says 43.2%;
  the direct full-corpus count on 2026-08-12 was 47.5% — use the latter). `--movement-source clicks`
  fixes this; **verify which source each checkpoint used before comparing runs.**

**UPDATE 2026-08-13 — the collapse was a label artifact, and it is fixed.** Under the old labels the
trained gate sat near **−5.0 logit ≈ 0.7 %/frame ≈ 0.2 commands/s** against the 2–5/s it models, and
inference needed a hand-calibrated `gate_bias` (`scripts/agent_infer.py:236-242`). Retrained on click
labels (`data/phase2_bc_clicks`), the gate tracks the true rate immediately:

| quantity | old (cursor labels) | new (click labels), epoch 0 |
|---|---|---|
| label event rate (`base`) | — | 0.11–0.13 /frame = **2.2–2.6 cmd/s** (matches 2.02 measured independently) |
| gate on transitions (`t`) | ≈0.007 | 0.15 |
| gate on holds (`h`) | ≈0.007 | 0.12 |

**A NEW problem replaced the old one.** The gate no longer under-fires — but `t − h ≈ 0.03` means it
has learned the *base rate* and barely discriminates "a command happens here" from "nothing happens
here." **`t − h` is the metric that decides whether this design earns its place.** If it does not
widen materially with training, then *when* a human issues a move command may not be predictable
from these features at all, which is a deeper finding than bad labels and argues for the ungated
baseline. Watch it in `ops/bc_clicks_5080.log` (`gate[t=… h=… base=…]`).

### 3.3 Dropped `n=0` BC term — DELIBERATE, impact LOW

Paper Eq 9 sums `n = 0..L` for actions. We sum `n = 1..L` (`scripts/train_agent_finetune.py:409`:
`for n in range(1, mtp_length):`). Reason recorded (`train_agent_finetune.py:17-22`):

> the dynamics is *action-conditioned* — the agent token at frame t is built from a window whose
> frame-t input already contains action a_t. So predicting a_t from `agent_out[:, t]` (the old n=0
> BC term) trivially leaks.

This is a genuine label leak that the paper does not have to worry about in the same way, and the
fix is right. **The reward MTP correctly keeps n=0** (`:356-373`) — reward is a target, never an
input.

Minor inconsistency to clean up: the arg help says *"n=0..L with L=8 → 9"* while the loss docstring
says *"n = 0..L−1"*. The code is `range(9)` = 0..8. Both descriptions are of the head count, not
the loss range.

### 3.4 Twohot bucket range ±3 — DELIBERATE but mis-tuned, impact MED

`RewardHead` and `ValueHead` both default to `bucket_low=-3.0, bucket_high=3.0, num_buckets=255`
(`heads.py:62-63, 439-440`). Docstring reason:

> Solo-gold reward (Δ own gold_total) gives tiny per-frame values and O(0.5-1) discounted returns;
> ±3 symlog leaves headroom for gold_scale tuning + kill/streak spikes without saturating.
> **TUNE once real return magnitudes are seen.**

Lineage: `7013f3a` set ±1.5 → `25bed26` set ±3.0.

**The paper gives no bucket range** — it says only *"symexp twohot output¹"* citing Dreamer 3.
Dreamer 3's ±20 is a **Dreamer 3** fact, not a Dreamer 4 one. Any doc that says "paper uses −20..+20"
is wrong on the paper (see §7).

**Where it actually hurts:** `docs/DATA_AUDIT_2026-08-12.md` finding 12 measures **39 of 255 buckets
used** — 85% of the head's capacity is wasted. The range is too *wide* for the realized reward
distribution, not too narrow. The docstring's own "TUNE once real return magnitudes are seen" is now
actionable.

### 3.5 `StateHead` — entirely ours, DELIBERATE, impact MED

`heads.py:16-45`. A 2-layer SiLU MLP off the agent token predicting 4 scalars in [0,1] at offset
n=0 only: own HP fraction, own level/18, lane-opponent HP fraction, opponent visibility. Trained
with masked MSE, RMS-normalized, `--aux-state-weight 0.5` in every production run.

Rationale in the docstring: *"The point is the GRADIENT, not the readout: the v7 tokenizer preserves
HUD detail too weakly for probes (cross-game HP R²~0.16), so this forces game semantics into the
trainable agent blocks straight from replay labels."*

**Two independent reviews rate this a diagnostic, not a fix:**
- `docs/TOKENIZER_REVIEW_2026-08-02.md` §3(G): *"Cannot exceed what frozen latents contain… do not
  read it as 'state recovered.'"*
- `docs/EXPERT_REVIEW_2026-08-02.md` §2(c): *"Right instinct, wrong layer… scalar state as a direct
  input."*

**Additionally, its targets are partly wrong.** `docs/DATA_AUDIT_2026-08-12.md` finding 4:
`enemy_visible` is wrong on **29.6%** of frames (54.5% of positives wrong) and `enemy_hp_frac` is
supervised off-screen; finding 11: `level` reads 19–20 → the level target exceeds 1.0 on 59 412
frames. Treat aux numbers from any pre-2026-08-13 checkpoint as unreliable.

**UPDATE 2026-08-13.** `enemy_visible` was fixed (gated on `screen is not None`), flipping 1 052 748
frames 1→0. It still means **"inside the camera frustum", not "rendered"** — fog-of-war false
positives remain (audit finding 15), documented in the docstring. The `level > 18` clamp
(finding 11) is **still open**.

**Silent-invalidation hazard, now guarded.** `num_targets` is `len(STATE_TARGETS)` at import time and
was recorded nowhere. Changing the *count* raises on shape, but **redefining a target at the same
width is silent** — which is exactly what `enemy_visible` just did. Checkpoints now record
`state_targets` and a resume refuses when the list differs
(`train_agent_finetune.py`, save block + resume guard). Checkpoints written before 2026-08-13 have no
`state_targets` key and get a warning instead: their `enemy_visible` column means the *old* thing.

### 3.6 `--ability-pos-weight 5.0` — REVERTED

The CLI **default is still 5.0** (`train_agent_finetune.py:136-138`) but **every production launcher
passes `--ability-pos-weight 1.0`** (`scratchpad/launch_bc_gate_1060.sh:19`,
`ops/bc5080_gate_watchdog.sh:28`). The 5.0 era is recorded in
`E2E_STATUS_AND_PLAN_2026-07-22.md:19`: *"Even with pos_weight=5, BCE learns the marginal cast rate,
not the state-conditional 'cast now.'"*

**Action item:** flip the default to 1.0 so the default and the practice agree. Right now a naive
re-launch silently changes the objective.

### 3.7 MTP length — NOT-A-DEVIATION

`--mtp-length 9` = 9 heads = `n ∈ {0..8}` = paper's `L=8`. Correct. (`docs/DREAMERV4_AUDIT.md`'s
claim of `mtp_length=8 default` is stale.)

### 3.8 Threshold cast decoding — REVERTED / superseded

Added in `0b9d2a6` ("casting: probe proves it's a calibration bug + add threshold fix"). It survives
only on the `temperature == 0` greedy branch of `agent_infer.py:229-230` with `ability_thresh=0.0`
(= sigmoid 0.5). **The live entrypoint defaults to `--temperature 1.0`**
(`scripts/play_live.py:309`), i.e. Bernoulli sampling — which is what
`docs/EXPERT_REVIEW_2026-08-02.md` §2(a) recommended. The threshold path is a demo/debug artifact.

---

## 3A. Assumptions ledger — what each head takes on faith

Sections 1–3 record where we differ from the paper. This section records something different and
easier to lose: **modelling assumptions that are nobody's deviation** — choices with no paper
counterpart at all, which are simply asserted by the code and would change the design if false.
Each row says what is assumed, whether it has been checked, and what breaks if it is wrong.

Status key: **MEASURED** (checked against data) · **UNTESTED** (plausible, never verified) ·
**KNOWN-VIOLATED** (data contradicts it and we ship anyway).

### 3A.1 RewardHead

| # | Assumption | Status | If wrong |
|---|---|---|---|
| R1 | Δ own `gold_total` is a sufficient reward for laning | **UNTESTED** | The agent optimizes farm and ignores everything else (waves, plates, map). This is the whole reward definition — nothing else enters. |
| R2 | Returns land inside twohot **±3** | **KNOWN-VIOLATED** | Measured **39 of 255 buckets used** (audit finding 12) — 85% of head capacity wasted. Range is too *wide*, not too narrow. |
| R3 | Reward at offsets n=0..8 is predictable from the agent token | **MEASURED — negative** | "Will this swing last-hit" AUC **0.431/0.29**, *worse than chance*. The head is an attack-windup detector, not a reward model. |
| R4 | γ=0.997 suits ~30-min games at 20 fps | **UNTESTED** | Horizon ≈ 333 frames ≈ 17 s. Wave-level credit assignment (30 s+) is outside it. |
| R5 | A 3-gold threshold cleanly separates last-hits from passive income | **UNTESTED** | Ambient gold leaks into the "event" signal. |

### 3A.2 PolicyHead — movement

| # | Assumption | Status | If wrong |
|---|---|---|---|
| P1 | The champion is camera-centred, so a **screen-space** target determines a move | **UNTESTED** | Movement is stored in screen coords; the camera is a moving frame. This is precisely how the old label bug arose (47.5% drift). Clicks fixed the *label*, not the representation. |
| P2 | A **21×21** grid resolves a move command | **MEASURED** | Still swallows **18.6%** of commands into the same bin as the previous one (down from 38.0%). The gate now fires on them regardless. |
| P3 | Off-screen commands can be clamped to the screen edge | **UNTESTED** | 7.3% of commands are clamped; direction is kept, distance is destroyed. |
| P4 | Command onsets are Bernoulli, independent per MTP offset | **UNTESTED** | Real commands are bursty and correlated (kiting, retreats). One gate logit per offset cannot express that. |
| P5 | Humans issue ~2 commands/s | **MEASURED ✓** | 2.027 cmd/s from clicks; two independent methods agree (2.02). |
| P6 | Movement is predictable at all from frozen latents | **OPEN — under test** | This is the `t − h` question in §3.2. If onsets are unpredictable, the gate cannot beat predicting the base rate. |

### 3A.3 PolicyHead — abilities

| # | Assumption | Status | If wrong |
|---|---|---|---|
| A1 | Abilities are **independent Bernoulli** (multi-hot) | **UNTESTED** | No combo structure representable (Q→AA→E). League play is combo-shaped. |
| A2 | Abilities are untargeted — one bit each, no aim point | **KNOWN-VIOLATED** | Real casts have positions. We emit "press Q", not "Q at (x,y)". A skillshot champion would be unplayable; Garen hides this because his kit is self/point-blank. |
| A3 | `--ability-pos-weight 1.0` is calibrated | **MEASURED** | At 5.0 the BCE learned the marginal cast rate, not state-conditional casting. Default is still 5.0 — **launchers override it**; a naive re-launch silently changes the objective (§3.6). |

### 3A.4 StateHead

| # | Assumption | Status | If wrong |
|---|---|---|---|
| S1 | All 4 targets lie in [0,1] | **KNOWN-VIOLATED** | `level` reads 19–20 on 59 412 frames → target > 1.0. Clamp still open. |
| S2 | `enemy_visible` means the enemy is **visible** | **KNOWN-VIOLATED** | It means *in the camera frustum*. Fog-of-war false positives survive the 2026-08-13 fix. |
| S3 | Aux gradient helps the agent blocks even when the readout is poor | **UNTESTED** | Two reviews call it a diagnostic, not a fix: it cannot exceed what frozen latents contain. |
| S4 | Target **order** is stable across checkpoints | **GUARDED** | Positional columns; a redefinition at equal width is silent. Now recorded + refused on mismatch. |

### 3A.5 ValueHead / Phase 3

| # | Assumption | Status | If wrong |
|---|---|---|---|
| V1 | λ=0.95 | **UNTESTED — and not from the paper** | The paper never states λ. Any doc claiming it does is wrong (§7). |
| V2 | Value shares the reward head's ±3 bucket range | **UNTESTED** | Inherits R2's mis-tuning. |
| V3 | Imagination is trainable from a frozen world model | **UNTESTED** | Never run (§5.1). The paper finetunes the world model and keeps the video loss; we do neither. |

### 3A.6 Cross-cutting

| # | Assumption | Status | If wrong |
|---|---|---|---|
| X1 | Phase-1 backbone features suffice; freezing loses nothing | **UNTESTED** | The paper finetunes the whole model (§4.1/4.2). Largest compute-forced bet in the stack. |
| X2 | Actions are resolvable at 20 fps (50 ms) | **UNTESTED** | Sub-50ms inputs (fast combos, animation cancels) are unrepresentable. |
| X3 | The action channel matters to the world model | **MEASURED ✓** | Removing actions costs **+16.4%** prediction loss; swapping cursor→click labels costs **+0.13% (t=0.80, n.s.)**. The backbone uses actions *coarsely*. |
| X4 | Masters+ replays are expert demonstrations worth cloning | **UNTESTED** | BC ceiling is the demonstrator. Emerald-level play needs the data to contain it. |
| X5 | `task_embed` is unused | **MEASURED ✓** | Nothing passes `task_id`; the branch never runs and `num_tasks=1`. It is random-init **and** frozen — harmless only because it is dead. Listed in `AGENT_ABSENT_PREFIXES`, deliberately not trainable. |
| X6 | 13/125 matches without `clicks.json` can fall back to legacy cursor labels | **KNOWN-VIOLATED, accepted** | 10.4% of frames train on drift-contaminated movement targets. Named loudly at startup and excluded from val. |

---

## 4. Phase 2 — behavior cloning + reward

### What the paper specifies

| Item | Paper | Anchor |
|---|---|---|
| What trains | *"Finetune **world model** with task inputs for policy and reward heads using (7) and (9)"* — the whole model | **Algorithm 1**, Phase 2 |
| Losses | Eq 9 (BC + reward MTP) **plus** Eq 7 (the video-prediction loss) | Algorithm 1; §3.3 |
| Noising | *"we reuse the pretraining setting with this additional loss function, so the representations are noisy and we continue to apply the video prediction loss"* | §3.3 |
| Agent tokens | attend to everything; *"no other modalities can attend back… crucial for avoiding causal confusion"* | §3.3 |
| Task conditioning | 20 one-hot tasks | §4.1, Table 4 |
| Data mixture | *"50% uniform sequences and 50% relevant sequences… BC loss applied only on the relevant fraction, dynamics loss only on the uniform sequences"* | §4.1 |

### 4.1 Frozen backbone — DELIBERATE (compute-FORCED), impact **HIGH**

```
[freeze] diffusion backbone FROZEN; 79 agent tensors (31,668,736 params) TRAINABLE
dynamics params: 146,254,368 (frozen)  |  reward 852,471  policy 382,932  state 263,684
```
(`scratchpad/bc_gate_5080.log`; `train_agent_finetune.py:193-221`.)

Only `agent_token`, `agent_temporal_pos`, `agent_blocks`, `agent_norm_out` + heads train. The paper
finetunes the whole transformer.

**Note this is a defensible variant**, not a mistake: the paper's own Fig 4 / Table 7 reports a
"WM+BC" ablation that outperforms both a from-scratch BC agent and a Gemma-3 VLA, showing world-model
representations transfer. But the paper's WM+BC *did* update the backbone. Ours is closer to linear
probing on a 0.8-epoch backbone.

**Correct sub-decision:** the agent blocks themselves **must** be trained here, because Phase-1
pretraining never touches them (`agent_out` is a side readout, absent from the denoising loss, so
the agent blocks only ever receive the DDP zero-grad tap and stay at random init). That reasoning is
recorded verbatim at `train_agent_finetune.py:193-198` and is correct.

**Stale docstring to fix:** `train_agent_finetune.py:483-484` claims *"The frozen dynamics runs
under `no_grad`… gradients flow only into the heads"* — false, and the code at `:205-211, 514-517`
deliberately does the opposite.

### 4.2 Video-prediction loss dropped — DELIBERATE, impact **HIGH**

Total loss is `bc_n + rew_n + aux_state_weight * aux_n` (`train_agent_finetune.py:539-545`). The
denoising output is explicitly discarded: `_, agent_out = dynamics(...)` (`:517`). No `z_0_pred`
term anywhere in the file.

This follows mechanically from 4.1 — with the backbone frozen there is nothing for the video loss
to preserve. But it means the paper's stated purpose (*"To preserve existing capabilities…"*) is met
by freezing instead of by regularizing, and there is **no signal at all keeping the world model
useful for the agent's own dreams**. This is why Phase 3 sees a backbone that was never trained with
the agent in the loop.

### 4.3 Narrow noise band — DELIBERATE, impact MED

```python
# train_agent_finetune.py:508-513
tau = args.tau_ctx + torch.rand(B, T, device=device) * (1.0 - args.tau_ctx)   # tau_ctx = 0.9
z_noisy, _ = schedule.add_noise(z0, tau)     # under no_grad
step_size = d_one                            # d = 1 pinned
```
τ ~ U(0.9, 1.0) = 0–10% noise, vs the paper's *"reuse the pretraining setting"* (full iid U(0,1)).
Reason recorded: *"Near-clean context corruption so the frozen denoiser sees in-distribution inputs."*
Sensible for a frozen backbone and matches the live-inference regime — but the agent tokens are then
**never trained on the noisy latents Phase 3 will hand them**, which is a real train/dream gap.

### 4.4 Action-history dropout — DELIBERATE, impact MED, no paper equivalent

`--action-dropout 0.15` in both production launchers. Implementation (`:494-500`) flips
`cursor_valid` off with probability 0.15 per frame, routing the movement input to `no_action_embed`.
**Movement only — ability history is never dropped.** Targets untouched. Off during validation.

Reason recorded: *"Breaks the learned copy-of-own-history shortcut so self-fed inference doesn't
collapse"* — the fix for the measured 1.8% self-fed collapse
(`docs/EXPERT_REVIEW_2026-08-02.md` §1c). Not in the paper.

### 4.5 No task conditioning — FORCED, impact LOW

`num_tasks=1` hardcoded (`:183`). `dynamics.task_embed` exists but `task_id` is **never passed** by
any trainer or by `agent_infer.py`, and `task_embed` is not in `AGENT_PARAM_PREFIXES`, so it sits
frozen at random init (inert, since it is never read). Single-task domain (Garen top lane) — the
paper's 20-task setup has no analogue here. Low impact now; will need building if the agent ever
needs steering.

### 4.6 No 50/50 data mixture — ACCIDENTAL, impact MED

One `ReplayLatentSequenceDataset`, fixed-stride windows, `VideoGroupedSampler` shuffling only for
cache locality. No relevance weighting, no episode filtering. Paper §4.1's mixture exists precisely
to *"amplify the signal in the dataset during behavior cloning, reward modeling, and reinforcement
learning"* — and our rewards are sparse (last-hits). `docs/DREAMERV4_AUDIT.md` has flagged this as
missing since January; it is still missing.

Held-out split *is* present and correct: whole-game, deterministic, `--val-games 6` default.

### 4.7 Agent-token attention — NOT-A-DEVIATION

Structural rather than mask-based, but the paper's rule holds exactly: the z stack runs to
completion first, then agent tokens cross-attend into the finished `x`
(`dynamics.py:738-747`); `AgentCrossAttention` takes q from agent tokens and k/v from z tokens only.
z can never see agent tokens. Agent self-attention is causal in time.

### Phase-2 hyperparameters (no paper counterpart — the paper specifies none)

AdamW (8-bit unless `--no-use-8bit-adam`), betas (0.9, 0.999), wd 0.1, lr 3e-4, WSD warmup 2 000 /
decay 0, grad-clip 1.0, batch 16 (5080) or 4 (1060), `seq_len 16 / stride 8`, 10 epochs planned,
bf16 autocast **only around the heads** (the backbone forward runs in fp32), no EMA, per-loss
`RunningRMS`.

---

## 5. Phase 3 — imagination training

### 5.1 It has never been trained — FORCED

Exactly one execution, a smoke test on 2026-07-31 (`scratchpad/eval_queue_0731.sh:24-33`):
`--horizon 4 --gen-steps 8 --batch-size 2 --epochs 1` on **1 match / 23 sequences / 11 batches**.
Log (`scratchpad/eval_round2.log:19-52`):

```
Epoch 0 [10/11] loss=2.0000 pi=6.1638 V=5.5412 KL=1.731e-07 R=0.000 A+=100%
```

Degenerate by construction — it used the *old non-action* backbone, so dreams could not respond to
the policy's actions. Artifacts: `scratchpad/imag_smoke/imagination_{epoch_001,latest}.pt` (policy +
value heads only). **No other Phase-3 checkpoint exists.** Everything below is code review, not
measured behaviour.

### 5.2 K=64 dreaming instead of K=4 — DELIBERATE (follows 2.1), impact **HIGH if run**

`--gen-steps` defaults to 4 but the README instructs `--gen-steps 64` for this lineage, because the
backbone never got the shortcut finetune (README:20, 97). Paper §3.2 dreams at **K=4, d=1/4**.

`imagine()` (`train_imagination.py:199-310`) additionally re-runs a **full forward over the whole
window** *and* a fresh `rollout()` prefill at every step, i.e. **O(H²)** in the horizon.

`docs/EXPERT_REVIEW_2026-08-02.md` §1d states the consequence bluntly: *"at K=64, H=8–10, O(H²) on
one 5080 you get a few thousand gradient steps of RL against a 0.5s-coherent simulator. That is noise
injection, not policy improvement."* **The shortcut finetune is a prerequisite for Phase 3, not an
optimization.**

### 5.3 λ and horizon — UNVERIFIED baseline, impact MED

`--gamma 0.997` ✅ matches Eq 10 verbatim. But:

- **`--lambda_ 0.95`**: the paper writes `R^λ` in Eq 10 and **never states λ**. 0.95 is Dreamer 3's
  value. Not wrong — just not a paper number. Any doc claiming "λ=0.95 matches paper" is unsupported.
- **`--horizon 8`**: the paper never states an imagination horizon either.

The interaction is the problem: at H=8 with γ=0.997, ≈97% of the λ-return is the **bootstrapped
value**, trained inside the same 0.5 s dreams, from a reward head whose magnitude calibration is
R²≈0.06. Sparse events (deaths) essentially never materialize inside an 8-frame dream. Flagged in
`docs/EXPERT_REVIEW_2026-08-02.md` §2(d).

### 5.4–5.6 Smaller deviations

- **`continues = torch.ones_like(rewards)`** (`train_imagination.py:332`) — no terminals, no continue
  head. The paper's Eq 10 carries `c_t` for non-terminal states. In LoL laning there is no episode
  end inside an 8-frame dream, so this is defensible; deaths enter only as a reward penalty.
- **Horizon bootstrap uses `v_{T-1}` for `v_T`** (`returns.py:164-169`), so the final reward is not
  silently dropped. Deliberate, documented, minor.
- **Per-loss `RunningRMS`** on the PMPO and value losses (`train_imagination.py:365-367`). Not part
  of Eq 11 — and it is why the logged `loss` pins to exactly 2.0000. Cosmetically confusing; the
  paper's own §3 intro does normalize all loss terms by running RMS, so this is arguably in-spirit.

### 5.7 Gated policy heads cannot enter Phase 3 — ACCIDENTAL, impact **HIGH** (blocking)

Every current Phase-2 checkpoint is trained with `--movement-gate`. Two hard blocks:

1. `train_imagination.py:163-166` builds `PolicyHead(...)` **without** `movement_gate`, then
   `load_state_dict(...)` with default `strict=True` → unexpected `gate_heads.*` keys.
2. `PolicyHead.log_prob` raises on gated heads (`heads.py:392-399`): *"wire prev-action plumbing into
   the imagination path before running PMPO on a gated head."*

The 2026-07-31 smoke used an *ungated* checkpoint, so this has never been hit. It will be the first
thing to break when Phase 3 is attempted for real.

### 5.8 What actually matches the paper — NOT-A-DEVIATION

| Item | Paper | Ours |
|---|---|---|
| PMPO α, β | 0.5, 0.3 (Eq 11) | 0.5, 0.3 (`train_imagination.py:99-100`) |
| PMPO form | `(1−α)·mean_{D⁻} ln π − α·mean_{D⁺} ln π + β·mean KL` | identical (`returns.py:292-331`) |
| Sign-only advantages | yes | yes, magnitudes unused |
| KL direction | *"reverse direction for the prior KL"*, `KL[π_θ ‖ π_prior]` (Eq 11) | `KL[π_θ ‖ π_prior]`, correctly factorized over 9 Bernoullis + 2 categoricals (`returns.py:232-289`) |
| Frozen prior | *"a frozen copy of the policy head"* | one-time `deepcopy`, `requires_grad_(False)` (`:512-514`) |
| What's frozen | *"only update the policy and value heads and keep the transformer frozen"* | dynamics + agent blocks frozen, reward head frozen, policy + value trainable |
| Rollouts per context | *"only one rollout from each context"* | exactly 1 |
| Value head | symexp twohot on λ-returns, TD | `twohot_loss(value_logits, symlog(returns))` (`:337-340`) |
| No entropy bonus | *"Dreamer 4 uses PMPO with a KL to the BC prior… where no normalization is needed"* (App. F) | none present |
| No advantage normalization | as above | `advantages = returns - values  # raw` (`:335`); `compute_advantages(normalize=True)` exists but is **never called** |
| γ | 0.997 (Eq 10) | 0.997 |

One documented mismatch in the codebase: `heads.py:581-598`'s `freeze_for_imagination` docstring says
*"paper trains reward head during imagination too."* The paper says the opposite — *"We **only**
update the policy and value heads"* (§3.3). The **trainer** freezes the reward head correctly
(`train_imagination.py:160-161`); only the helper's docstring is wrong. See §7.

---

## 6. Data and scale

### What the paper specifies

| Item | Paper | Anchor |
|---|---|---|
| Corpus | **2 541 h** VPT contractor gameplay, 90/10 split, 360p @ 20 FPS | §4; Appendix A |
| Action labels | Full 2 541 h available; the scaling study shows **100 h of actions** reaches 85% PSNR / 100% SSIM of the all-actions model; 10 h reaches 53% / 75% | §4.3, **Fig 7** |
| Unlabeled leverage | *"world models absorb the majority of their knowledge from unlabeled videos, and require only a small amount of actions"*; unlabeled videos use *"only the learned embedding"* | §4.3; §3.2 |
| Resolution | 360×640 | Table 3 |
| Compute | 2 B params on 256–1024 TPU-v5p | §4 |

### What we do

| Item | ahriuwu | Verdict |
|---|---|---|
| Action-labeled | **49.4 h** / 125 matches with both frames and latents (3 554 768 label frames; `docs/DATA_AUDIT_2026-08-12.md` scope line). Full replay corpus is 147 games / ~58 h | 6.1 |
| Unlabeled | **~450 h YouTube available, 0 h used** in dynamics training. Tokenizer v7 is also replays-only (~54 h) | 6.2 |
| Ratio | unlabeled : labeled = **0 : 1** vs the paper's ≈ **25 : 1** | 6.2 |
| Resolution | 352×352 squished from 1280×720, 20 FPS | 6.4 |
| Compute | one RTX 5080 (+ a GTX 1060, + short Vast rentals) | — |

#### 6.1 49 h labeled — FORCED, impact MED

Below the paper's 100 h data point but above its 10 h one, so by Fig 7's curve the action
conditioning should be *learnable* — **if** there were a large unlabeled corpus underneath it. Which
brings us to:

#### 6.2 450 h YouTube excluded — DELIBERATE, contested, impact **HIGH**

This is the deviation that most directly inverts the paper's central claim. Dreamer 4's headline
result (§4.3, Fig 7) is that **the world model gets most of its knowledge from unlabeled video** and
needs only a sliver of actions. We run at 0:1.

Recorded reason: a YT-mixed retrain re-introduced black-HUD contamination and was discarded
(README:32). Two independent reviews call the *permanence* of that decision wrong:

- `docs/EXPERT_REVIEW_2026-08-02.md` §1e: *"Wrong as a permanent decision, accidentally right this
  quarter. Contamination is preprocessing-fixable (HUD masking/cropping)… stop citing the mixed-retrain
  eval as evidence."*
- `docs/TOKENIZER_REVIEW_2026-08-02.md:131-135`: DATA is one of only two evidence-backed levers left
  (the other is the objective); *"the 450h YT never entered tokenizer training (needs the HUD masking
  fix)."*

The `--exclude-blacked-regions` machinery (see 1.7) is the fix, and a mixed run
(`slurm/slurm_tok_train_v7_yt.sbatch`, 142 replay matches + 250 YT games, ~181 h of clips) **was
launched with it** and cancelled at ~step 800 on 2026-08-09. So this is a **deferred fix, not a
principled exclusion** — and the doc record should say so.

#### 6.3 HUD-off replays vs HUD-on live capture — ACCIDENTAL, impact **HIGH**

`docs/DATA_AUDIT_2026-08-12.md` finding 14: *"Train/deploy gap: training frames have **no HUD**, live
capture does"* — 0 static pixels in training frames. The tokenizer has never seen the HUD it will be
handed at inference, and the HUD occupies 34.09% of the frame per `scratchpad/hud_valid_mask_352.pt`.
Nothing about this is a paper deviation *per se* (the paper's train and eval distributions match) —
it is a domain-transfer defect that the paper's setup simply doesn't have.

#### 6.4 352×352 squish — FORCED, impact MED

Aspect-destroying resize from 1280×720. `docs/DATA_AUDIT_2026-08-12.md` **refutes** the theory that
the squish is what kills minion HP bars (squish-352: 9/10 bars detected vs letterbox-352: 4/10); the
real cost is total resolution (3.64× horizontal downsample), which is 1.4 above.

---

## 7. UNVERIFIED / fabricated paper claims found in this repo

These statements appear in this repo's code or docs and are **not supported by the paper text.**
A python-regex sweep of `scratchpad/dreamer4_text.txt` was used for each.

| Claim | Where | Status |
|---|---|---|
| *"Linear warmup + cosine decay to 0 (**DreamerV4 §3.4 recipe**)"* | `src/ahriuwu/utils/training.py:303` | **UNVERIFIED.** §3.4 is "Efficient Transformer" and contains no LR schedule. The words `cosine`, `learning rate`, `warmup`, `schedule` (as an LR schedule) do not appear in the paper at all. |
| *"cosine (warmup→cosine-to-0, **paper-faithful**)"* | `src/ahriuwu/utils/training.py:204` | **UNVERIFIED.** Same. |
| *"AdamW (beta1, beta2). **DreamerV4 paper uses defaults (0.9, 0.999)**."* | `src/ahriuwu/utils/training.py:212` | **UNVERIFIED.** The paper never names an optimizer. The only `Adam` matches in the text are bibliography author names. |
| *"Bucket range −20 to +20 (255 buckets) ✅ **Matches**"* | `docs/DREAMERV4_AUDIT.md:268` | **UNVERIFIED as a paper claim** (±20 is Dreamer **3**) **and factually wrong about the code** (we use ±3). |
| *"λ-returns with γ=0.997, **λ=0.95**" listed under "Matches Paper"* | `docs/DREAMERV4_AUDIT.md:431` | **UNVERIFIED.** The paper never states λ. (The same doc correctly lists λ under "Paper Unknowns" at `:287` — it contradicts itself.) |
| *"lambda_: TD(λ) parameter (**paper uses 0.95**)"* | `src/ahriuwu/models/returns.py:155` | **UNVERIFIED.** Same. |
| *"freeze_reward: If False (default), reward head stays unfrozen (**paper trains reward head during imagination too**)"* | `src/ahriuwu/models/heads.py:596-597` | **CONTRADICTED.** §3.3: *"We **only** update the policy and value heads and keep the transformer frozen."* The trainer does the right thing; only the docstring is wrong. |
| *"tokenizer regularized for predictability"* | historical claim, per the task brief | **CONTRADICTED.** Eq 5 is reconstruction only: MSE + 0.2·LPIPS. No latent regularizer of any kind exists in the paper or in `losses.py`. |
| *"3 space layers per 1 time layer"* attributed to the paper | `docs/DREAMERV4_AUDIT.md:164` | **PARAPHRASE, not a quote.** The paper says *"only use temporal attention once every 4 layers"* (§3.4) — same ratio, different framing. Harmless but don't quote it as paper text. |
| *"Ramp weight… `ramp_weight(tau) = 1.0 − 0.9 * tau` (inverted convention) ✅ Matches"* | `docs/DREAMERV4_AUDIT.md:212` | **STALE + WRONG.** Current code is `0.9 * tau + 0.1` (`diffusion.py:218`) — literally Eq 8. |
| *"τ² scaling \| `tau_weight = tau_idx ** 2` \| ✅ Matches"* | `docs/DREAMERV4_AUDIT.md:228` | **STALE + WRONG.** Paper Eq 7 uses `(1−τ)²`, not `τ²`, and the current code uses neither (x-space, see 2.3). |

### Paper facts that are genuinely unspecified (don't invent them)

The paper specifies **no** learning rate, optimizer, betas, weight decay, LR schedule, warmup, EMA,
gradient clipping, dropout, twohot bucket range, λ for λ-returns, imagination horizon, PMPO
temperature, or per-phase step counts. Every one of those in this repo is a repo choice. That is
fine — just never label them "paper-faithful."

---

## 8. Reverted / abandoned — do not re-litigate

| Thing | Verdict | Where |
|---|---|---|
| **256×32 bottleneck** and its "+7 dB" evidence | Retracted. Paper shape **is** 512×16 (Appendix A); the +7 dB was an overfit artifact, real gap ~+1 dB early-training | `docs/TOKENIZER_REVIEW_2026-08-02.md:114-122`; `slurm/slurm_v7_trial.sbatch` |
| **`--ability-pos-weight 5.0`** | Superseded by 1.0 in every launcher; 5.0 is still the CLI default (fix this) | §3.6 |
| **Threshold-based greedy cast decoding** | Superseded by Bernoulli sampling (`--temperature 1.0` is the live default) | §3.8 |
| **Horizon-ramped context-heavy τ schedule** | Fixed in `8212365`; now iid U(0,1) per frame | §2.17 |
| **Per-step whole-sequence `independent_frames`** | Fixed in `8212365`; now per-example | §2.16 |
| **Cosine LR schedule** | Dropped for WSD constant in `08fc474`; the "paper-faithful" label on the cosine path is wrong either way | §7 |
| **Twohot range ±1.5** | Widened to ±3.0 in `25bed26`; measured usage is 39/255 buckets, so ±3 is now too wide | §3.4 |
| **Additive τ/step conditioning** | Added `3eba0e1`, **reverted `eb38863`**, re-added `48541ba`, split per-signal `7ae518d`. Currently ON, deliberately | §2.4 |
| **CNN tokenizer baseline** | Deleted in `737444b`; transformer won | `docs/TOKENIZER_REVIEW_2026-08-02.md:11` |
| **YT-mixed dynamics retrain** | Discarded over black-HUD contamination; the `--exclude-blacked-regions` fix now exists and the exclusion should be revisited, not treated as settled | §6.2 |

---

## 9. Which docs to trust

| Doc | Status |
|---|---|
| **This file** | Current. Verified 2026-08-12 against code, checkpoint args, and the paper text. |
| `DYNAMICS_VS_PAPER.md` | **Best-cited paper transcription in the repo** — use it for verbatim paper quotes. **Stale on two points**: its §1.5 (τ schedule) and §1.9 (independent frames) describe pre-`8212365` behaviour, both since fixed. Its "job 124" config also predates the shipped `gs8775` checkpoint (which uses betas 0.9/0.999 and 8-bit Adam, not 0.9/0.95). |
| `docs/TOKENIZER_REVIEW_2026-08-02.md` | Current and measurement-backed. Read the 2026-08-05 correction addendum — it retracts §1's "faithfulness override" framing and Option D. |
| `docs/EXPERT_REVIEW_2026-08-02.md` | Current opinion piece; its measured claims (77% held frames, R² 0.16, 49h/450h) all check out. |
| `docs/DATA_AUDIT_2026-08-12.md` | Current, the most rigorously measured doc here. Supersedes older claims about the movement target, aux targets, bucket usage, and aspect ratio. |
| `docs/DREAMERV4_AUDIT.md` | **STALE — do not cite.** Written 2026-01-28. Wrong on: action space (says 18 directions / 8 abilities; actual 21×21 bins / 9 abilities), resolution (256×256 vs 352×352), bucket range (−20..+20 vs ±3), `mtp_length` default (8 vs 9), ramp-weight sign, bootstrap τ² scaling, and "PMPO / behavioral prior / rollouts not implemented" (all implemented in `d63bcbe`). Its §7 fabrications are catalogued above. |
| `DYNAMICS_REVIEW.md` | Mostly current on architecture; its §6.7 "context-heavy τ schedule" is stale (fixed), and §4.3's betas (0.9, 0.95) disagree with the shipped checkpoint (0.9, 0.999). |
| `README.md` | Current as of 2026-08-01 and honest about measurement quality. |

---

## 10. If you change one thing

Ranked by (likely quality gain) ÷ (cost), from the deviations above:

1. **1.2 — the mask curriculum.** The tokenizer was never trained in the paper's MAE regime. Fixing
   this is a one-line change (`--mask-warmup-steps 0`), and the paper attributes concrete
   downstream value to MAE (*"improve the spatial consistency of videos generated by the dynamics
   model"*). Any future tokenizer run should also drop `--tube-masking` (1.1).
2. **6.2 — re-admit the YouTube corpus** with `--exclude-blacked-regions`. This is the paper's own
   headline lever (0:1 vs 25:1) and the fix is already written and already launched once.
3. **5.2 / 2.1 — run the shortcut finetune before Phase 3.** It is a prerequisite, not an
   optimization, at H=8 on one GPU. **Fix 2.14 first** — enabling `--shortcut-forcing` today silently
   disables RMS loss normalization, drops `independent_frames`, and samples `d` per-example instead
   of per-frame. Also budget for 2.12: `step_embed` rows 1–6 are still at init.
4. **5.7 — unblock gated heads in `train_imagination.py`** (strict=False + prev-action plumbing), or
   Phase 3 cannot start at all.
5. **3.4 — retune the twohot range** to the measured reward distribution (39/255 buckets used).
6. **3.6 — flip `--ability-pos-weight` default to 1.0** so the default matches every production run.
7. **2.5 — set batch length > context length** on the next dynamics run, per §3.4's explicit
   length-generalization requirement.
