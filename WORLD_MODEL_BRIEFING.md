# ahriuwu World Model — Problem Briefing & Full Handoff

*Written 2026-07-13. Purpose: everything a person needs to get fully up to speed on why the League-of-Legends world model plateaued as a forward-predictor, what's been tried, how it maps to the Dreamer 4 paper, and where the solution likely is. Self-contained — you shouldn't need to read the code first, but every claim is traceable to a file or a run.*

---

## 0. TL;DR — the core problem (read this first)

We're building a **Dreamer 4–style world model for League of Legends (Garen, top lane)**. Pipeline: frozen **tokenizer** → **dynamics** (world model) → **agent heads** (BC + reward, then imagination RL). The dynamics model is the keystone: the RL that makes an agent actually *good* (Phase 3, "imagination training") runs entirely *inside* the world model's dreamed rollouts. **If the world model can't predict the future accurately, the whole agent stack is capped.**

**The symptom:** the current world model (job **179**) is a **good single-frame denoiser but a bad multi-step predictor**. Its autoregressive "dream" degrades *below the trivial persistence baseline* (just holding the last real frame) and visually collapses into garbage within ~16 frames. Its teacher-forced denoising PSNR has been **flat at ~19 dB for ~24h of training** — a plateau.

**The key insight (corrected from an earlier misdiagnosis):** we *already had* a world model that predicted well. Job **135** — same architecture but **action-conditioned** on the 125 action-labeled replays — **beat persistence across all 16 rollout frames** (dream 25.6→22.4 dB vs persist 24.2→21.5). The regression to "worse than persistence" happened when we **turned action-conditioning OFF** (`use_actions=False`) to scale training data up to 578 games including unlabeled YouTube. So the plateau is **largely self-inflicted: we're running the paper's worst-case "no actions" ablation.** The paper trains the world model with *mixed* action-conditioning (labeled frames get real actions; unlabeled frames get a learned placeholder embedding) — we skipped that and ran fully unconditioned.

**Not the cause (ruled out this session):** it is **not** tokenizer-limited (the tokenizer faithfully decodes every frame at ~28–29 dB, far above the model's 24.6 dB 1-step); it is **not** data-limited (~19 dB is measured on *training* data → it's underfitting, not overfitting); the τ-schedule and independent-frames deviations that once hurt it were **already fixed** in July and are what let 135 succeed.

**The most likely fix:** re-introduce action-conditioning the way the paper does — one world model trained on a **mix** of action-labeled replays (real actions) + unlabeled YT (placeholder embedding). Secondary levers: model capacity (paper says WMs need high capacity; ours is "medium" 114M), more labeled data, and cloud scale. There's also a measurement caveat (see §6.4) worth clearing first.

---

## 1. Project goal & the Dreamer 4 pipeline

- **Goal:** an RL agent that plays Garen in a real LoL game, learned mostly from video. End-to-end target: screen capture → tokenize → world model → policy → inject inputs.
- **Reference:** Hafner\*, Yan\*, Lillicrap (2025), *"Training Agents Inside of Scalable World Models"* (Dreamer 4), arXiv **2509.24527v1**. PDF is in the repo (`Dreamerv4_paper.pdf`, 32pp). An exhaustive line-by-line paper-vs-code comparison already exists at `DYNAMICS_VS_PAPER.md` (632 lines) — this briefing summarizes and updates it.
- **The three phases (Algorithm 1 in the paper):**
  1. **World-model pretraining** — train tokenizer on video; train dynamics on tokenized video **and optionally actions**.
  2. **Agent finetuning** — freeze the world model, insert *agent tokens*, train a **policy head** (behavior cloning) and **reward head**.
  3. **Imagination training** — RL (PMPO policy + value head) *inside* rollouts dreamed by the world model. **This is what makes the agent good, and it requires a world model that rolls out well.**

Convention (used everywhere): signal level **τ=1 → clean, τ=0 → pure noise** (paper's convention; our code matches).

---

## 2. Architecture (what each component is)

### 2.1 Tokenizer "v7" (frozen — not the problem)
- Transformer/perceiver image tokenizer. Bottleneck = **512 latent tokens × 16 dim** with a tanh readout.
- For the dynamics model, each frame's `(512,16)` is folded `view(16,16,-1)` → **(32, 16, 16)** = **latent_dim 32 on a 16×16 spatial grid** (256 spatial tokens). This is the paper's "spatial tokens" layout.
- Frozen throughout dynamics/agent training. Replay recon ~26.8 dB; measured pixel **ceiling ~28–29 dB** this session (decode of *true* latents — the max any dynamics could hit).

### 2.2 Dynamics model (the world model — the problem)
- `create_dynamics("medium", latent_dim=32, num_kv_heads=4, num_register_tokens=8, soft_cap=50.0, use_qk_norm=True)`:
  **model_dim 768, 18 layers, 12 query / 4 KV heads (GQA), head_dim 64, spatial 16×16=256, temporal attention every 4th layer (→ 14 spatial / 4 temporal), SwiGLU hidden 2048, max_seq_len 256, ~114.6M params.**
- Per-frame token layout: **`[256 latent (2D-RoPE) | 8 register | 1 action | 1 condition]` = 266 tokens.**
- Standard-transformer specifics (all present, match paper Sec 3.4): pre-RMSNorm (eps 1e-6), RoPE (2D spatial + 1D temporal), SwiGLU, QKNorm (scale 1.0), attention logit soft-cap 50. Spatial attention is full within a frame; temporal attention is **causal** across frames per spatial position.
- **Objective: x-prediction diffusion forcing.** `z_τ = τ·z_0 + (1−τ)·ε`; the net predicts **clean z_0** (x-prediction, not v-prediction — paper's choice for stable long rollouts). Loss = per-frame MSE × **ramp weight `w(τ)=0.9τ+0.1`** (Eq 8), then **RMS-normalized**.
- **Action conditioning mechanism (this is the crux):** actions are embedded (continuous movement via `Linear`, binary keys via `Embedding`), summed with a learned embedding, and appended as the "action" token. **For unlabeled frames, only a learned placeholder `no_action_embed` is used** — exactly the paper's recipe. *This mechanism exists in the code but 179 has it turned off.*
- **Shortcut forcing** (fast K=4 inference) is implemented but **OFF** for these runs; we roll out at `d=1` (the many-step regime), which is the paper's own strong baseline (FVD 306). Not the cause of the plateau.
- **Agent tokens** (Phase 2+): one-way cross-attention (agent tokens see everything; nothing attends back — paper's anti-causal-confusion rule). See §5 for the Phase-2 fix made this session.

### 2.3 Heads
- **RewardHead**: twohot multi-token-prediction (MTP length 9), symexp buckets ±3 symlog.
- **PolicyHead**: factorized — 9 independent **Bernoulli abilities** + **per-axis binned movement** (movement_bins=21) with real categorical log-prob.
- **ValueHead** (Phase 3 only): twohot, ±3 symlog.
- **Action space (v1):** 9 binary keys `[Q, W, E, R, Flash, Ignite, AA, Recall, Stride]` + continuous 2-D movement `(x,y)∈[0,1]`. Parsed from `labels.json`/`clicks.json`. **NB: no camera** (camera *is* recorded in `raw_cam.json` as `cx,cy,cz` per frame, but is not in the action space; adding it was considered and rejected).
- **Reward = solo Garen gold**: dense `gold_scale·Δ(own gold_total)` + own-death penalty. `gold_scale=1e-3` is a placeholder to tune on real returns. Ignores win/loss (works on every match).

---

## 3. Full hyperparameters (current run 179)

**World-model training (`slurm/slurm_dyn_yt578.sbatch`):**
| Param | Value | Notes |
|---|---|---|
| model-size | medium (114.6M) | |
| latent-dim | 32 | matches tokenizer v7 |
| **use-actions** | **FALSE (`--packed`)** | ← the regression vs 135 |
| num-kv-heads | 4 | GQA |
| num-register-tokens | 8 | |
| soft-cap | 50.0 | |
| alternating-lengths | short 128 / long 256 | long-ratio 0.1 |
| batch-size | short 2 / long 1 | grad-accum 8/8/16 (eff. ~16) |
| **stride** | **64** | was 8 (changed 2026-07-11 for data efficiency; ~50% overlap on 128-clips) |
| gradient-checkpointing | ON | |
| compile | OFF | 5080 is Blackwell sm_120 → compiled kernels hit illegal-memory-access |
| independent-frame-ratio | 0.3 | per-example (post-fix), = paper's "30% videos as images" |
| τ schedule | i.i.d. U(0,1) per frame | post-fix (was a context-heavy repo invention) |
| pixel-hud-loss | ON, K=4 frames | masks blacked YT HUD out of loss (see §4) |
| lr / schedule | 3e-4 / **WSD, warmup 3000, decay-steps 0** | **⇒ constant LR forever, never anneals** |
| optim | AdamW β(0.9,0.999), wd 0.1 | |
| epochs | 50 | eval every 200 steps |
| eval | teacher-forced τ-sweep PSNR + free-running rollout PSNR | rollout eval uses num_steps≈16 (see §6.4) |

**Phase-2 BC (`scripts/launch_bc_1060.sh`, currently running on the 1060):** frozen 179 backbone + trainable agent blocks + reward/policy heads; seq-len 32, stride 8, batch 6, lr 3e-4, warmup 2000, ability-pos-weight 5.0, checkpoint every 20 min, `--resume auto`, dataset-index cached. (See §5.)

**Phase-3 imagination (`scripts/train_imagination.py`, not yet run for real):** horizon 8, γ=0.997, λ=0.95, PMPO α=0.5 β=0.3, gen K=4 / k_max 64, temperature 1.0. **Blocked** on an action-conditioned world model (with `use_actions=False` the dreamed frames don't respond to the policy's actions → no learning signal).

---

## 4. Training-run history & checkpoint lineage

All checkpoints under `/mnt/storage/data/ahriuwu/checkpoints/…` (desktop-local HDD). Metric shorthand: **tf τ0.9** = teacher-forced 1-step denoising PSNR at τ=0.9 (near-clean); **rollout** = free-running autoregressive dream vs persistence.

| Job | What | Data | Actions | Result |
|---|---|---|---|---|
| **124** | first medium run (`dynamics_v7_replay`) | 125 replays | on | competent 1-step predictor; rollout eval looked like garbage until the **sampler bug** was found (see below) |
| **131** | fresh run after the τ-fix | 125 replays | on | tf τ0.9 23.6 @ 3020 |
| **135** | **the good one** — resumed 131 via a **4×4090 Vast** accel (step 5215→8775, ~$7.67) then continued on desktop | 125 replays | **on** | **tf τ0.9 25.5; rollout BEATS persistence across all 16 frames** (dream 25.6→22.4 vs persist 24.2→21.5). Checkpoint `dynamics_v7_accel_resume/`, also on R2. |
| 154 | first 578-game **unlabeled** scale-up | 453 YT + 125 replay | **off** | learned the **HUD-black shortcut** (painted black onto clean replay dreams) — YT frames have a fixed blacked HUD, the model learned to reproduce it |
| **168** | HUD-fix: pixel-space masked loss on YT | 578 | off | HUD-blacking **fixed & verified** (dream bottom-brightness/GT 0.71→0.91); tf τ0.9 ~25 early |
| 178 | continuation of 168 | 578 | off | — |
| **179** | current: 178 + **stride 8→64** | 578 | **off** | **PLATEAUED: tf τ0.9 flat ~19 dB for ~24h; rollout ~7–8 dB mean, below persistence.** Paused at gs≈6526, epoch 3. |

**Key changes between checkpoints that matter:**
1. **Actions ON→OFF** (135 → 154+): the single biggest change. 135 (on) beat persistence; 179 (off) doesn't.
2. **Data 125 → 578 games** (added 453 unlabeled YT, tokenized via `pretokenize_yt_v7.py`). More data, but only useful for unconditioned prediction.
3. **HUD-mask pixel loss** (154 → 168): for YT batches, decode `z_pred`+`z_0` through the frozen v7 decoder frame-by-frame and take MSE over **non-HUD pixels only** (`scratchpad/hud_valid_mask_352.pt`). Replays keep the latent loss. Fixed the black-painting.
4. **stride 8 → 64** (178 → 179): more distinct frames per step; did *not* lift the plateau.
5. **Two earlier fixes that WORKED (2026-07-04/05), applied from 131 on:** (a) **sampler fix** — the multi-step Euler denoiser was re-noising each step with the *frozen* initial noise instead of the implied noise, so multi-step rollouts diverged even for a good model (committed `3f69de8`); (b) **τ-schedule + independent-frames fixes** — training τ made i.i.d. U(0,1) (was a context-heavy repo invention that starved the τ≈0 regime rollout runs in), and independent-frames made per-*example* (was one coin-flip per whole batch, which trained the temporal layers to be a cross-frame no-op). These two are exactly why 135 rolls out well; **the current code already has them.**

---

## 5. A separate bug fixed this session (Phase-2 BC)

Not the world-model problem, but relevant to the handoff. The Phase-2 trainer (`train_agent_finetune.py`) froze the **entire** dynamics and detached the agent-token output — but the **agent-token blocks (~32M params) are new in Phase 2 and were never trained in Phase 1** (agent_out is a side readout, absent from the denoising loss; 179's checkpoint has 0 agent keys). So BC was learning a policy from a **random projection**. Fixed: freeze only the diffusion backbone, keep the agent blocks trainable, let gradient flow. Verified (BC loss now drops on real data). Also added: time-based checkpointing + `--resume` + a watchdog + heartbeat + an ability class-weight (casts are sparse → BCE was collapsing to "never press") + a dataset-index cache (label/reward parse was ~40 min per start → now ~0s on cache hit, 486× faster).

---

## 6. The diagnosis — evidence, with numbers

### 6.1 The rollout decay graph (this session, 179 @ gs6526, `rollout_check.py --decode --horizon 32`)
Per-frame **pixel** PSNR vs the true frame, autoregressive:
- **Tokenizer ceiling** (decode of true latents): **~28–29 dB, flat** across all 32 frames.
- **Dynamics dream**: **24.6 → 16 dB** — starts ~ok, collapses.
- **Persistence** (hold last real frame): **~25 → 18 dB**.
- ⇒ the dream is **below persistence for nearly the whole horizon**, and the montage shows it degenerating into a teal blocky hallucination by ~frame 16 (classic autoregressive error-snowball).
- ⇒ **NOT tokenizer-limited** (ceiling ≫ dream). Files: `dream_psnr_plateau.png`, `dream_plateau.png`.

### 6.2 Denoising τ-sweep (179, teacher-forced 1-step)
`τ0.1=8.1, 0.3=11.1, 0.5=12.9, 0.7=15.4, 0.9=19.4`. The τ0.9 point (~19 dB latent) has been flat within noise (17–22) for the whole stride-64 run.

### 6.3 Train vs. val → underfitting
179's in-trainer eval uses `num-eval-videos 0`, i.e. it evaluates on **training** data (no held-out set exists as latents — the holdout dirs are PNGs only). So ~19 dB is **train** PSNR: the model can't fit its own training data past ~19 dB → **underfitting, not overfitting → more data won't fix it.** Bottleneck is capacity / optimization / (the missing) action-conditioning, not dataset size.

### 6.4 Measurement caveat (clear this before over-interpreting rollout)
There's a documented **rollout-eval step-count sensitivity**: post-sampler-fix, fewer denoise steps = less accumulated integration error. On a good checkpoint (pre-fix numbers): num_steps 1→~18 dB, 4→~16, 16→~12, 64→~7. The rollout evals (and my `--num-steps 8` run) use *more* steps than the intended K≈4 regime, so **part of the "below persistence" gap is sampler integration error, not pure model error.** Deviation #4 in `DYNAMICS_VS_PAPER.md` recommends dropping eval `num_steps` 16→~4. **TODO before big decisions: re-run 179 rollout at num_steps=1 and 4** to separate sampler error from model error. (Even so, the *clean* comparison — 135 action-conditioned beat persistence, 179 unconditioned didn't, at similar eval configs — still points at actions.)

### 6.5 Camera observation (recorded, unused — noted, user rejected adding it)
The dominant frame-to-frame change in LoL is the **camera panning**. Camera position is recorded per frame in `raw_cam.json` (`cx,cy,cz`, ~60k frames/match) but is **not** in the action space. So an unconditioned model literally can't know where the view will move. This is a candidate lever but **the user has decided against adding camera** — leaving it here only as a known fact for whoever picks this up.

---

## 7. Paper vs. ours — mirrored vs. different

Condensed from `DYNAMICS_VS_PAPER.md` (which cites the paper verbatim + our `file:line`). "Fixed since" = the 2026-07-04 audit doc flagged it and it was subsequently fixed.

**MIRRORED (faithful):**
- x-prediction diffusion forcing (Eq 6); flow loss in x-space; ramp weight `0.9τ+0.1` (Eq 8) — exact.
- Efficient transformer: temporal attention every 4th layer, causal-in-time, GQA, register tokens, RMSNorm/RoPE/SwiGLU/QKNorm/soft-cap (Sec 3.4) — all present.
- Alternating short/long batch lengths (Sec 3.4).
- Conditioning token: discrete τ + step-size embeddings, appended token (Sec 3.2).
- Action mechanism incl. **`no_action_embed` placeholder for unlabeled frames** — the code matches *"when training unlabeled videos, only the learned embedding is used."*
- Autoregressive KV-cached rollout with near-clean context corruption (τ_ctx≈0.1) (Sec 3.2).
- One-way agent-token masking (Sec 3.3). Twohot reward/value + λ-returns (γ=0.997) + PMPO (α=0.5,β=0.3) math — verified correct to ~1e-5.

**DIFFERENT (deviations):**
1. **We ran the world model with NO actions at all (`use_actions=False`)** — the paper trains *mixed* (small action-paired fraction + unlabeled placeholder). **This is the big one.** The paper shows action-conditioned generation is markedly better than the no-action ablation, and that only ~10–100h of actions (of ~2500h video) already recovers 53–85% of full-action PSNR. Ours is the worst-case ablation.
2. **Capacity:** paper (Sec 4.4) says "world models require high model capacity"; their models are larger. Ours is "medium" (114M) — possibly undersized for LoL's complexity.
3. **Shortcut forcing OFF** — deferred by design; we roll out at d=1 (the paper's strong K=64 baseline). *Not* the cause. When enabled, our bootstrap loss is computed in x-space, not the paper's v-space `(1−τ)²` form (deliberate bf16-precision choice).
4. **τ/step conditioning injected twice** (additive to all tokens *and* appended token; paper = token only) — benign superset.
5. **Batch length == context length** (128/256) — violates the paper's "batch length > context length" length-generalization margin; minor contributor to weak far-horizon rollout.
6. **LR schedule:** WSD with `decay-steps 0` = constant LR forever (paper uses cosine; `--lr-schedule cosine` is available). A constant high LR is consistent with the noisy, non-converging plateau — an anneal would likely settle it and squeeze a little more denoising.
7. **Rollout-eval num_steps** too high (§6.4).

**Deviations that were flagged then FIXED (so *not* current issues):** the context-heavy τ schedule (now i.i.d. U(0,1)); independent-frames as a per-batch coin-flip (now per-example); the frozen-noise Euler sampler bug (now implied-noise Euler everywhere).

---

## 8. The path to "great" — candidate solutions, ranked

1. **Re-introduce action-conditioning the paper's way (highest confidence).** One world model, `use_actions=True`, trained on a **mix**: the 125 labeled replays carry real actions; the 453 YT games use `no_action_embed`. This is the paper recipe and is directly supported by our evidence (135 action-conditioned beat persistence; 179 unconditioned regressed). **Blocker to wire:** the data path — `ReplayLatentSequenceDataset` (has actions) and `PackedLatentSequenceDataset` (YT, no actions) need to be *mixed in one run* so a batch can contain both, with YT routed to the placeholder. This "mix labeled + unlabeled" tweak is the concrete next engineering task; it's been flagged repeatedly as "a separate phase."
2. **Clear the measurement caveat first (cheap):** re-run 179's rollout at num_steps ∈ {1,4} and set eval num_steps≈4, so the metric reflects the model, not sampler drift. Do this before spending compute, so you're optimizing the right number.
3. **Capacity + scale (likely needed for *great*, needs cloud):** medium may be too small; a great WM is probably larger + longer-trained + more data than the 5080 (16 GB) can hold. The Vast/DDP cloud path is already built and validated on 2×4090 (see §9).
4. **LR anneal (small, quick win for the encoder):** schedule a decay tail (or drop to ~3e-5) to settle the constant-LR plateau — worth ~1–3 dB of denoising for the demo encoder, won't fix prediction.
5. **More labeled replays** — only 125 games carry actions; the paper needs relatively little, but more helps action grounding.
6. **(Rejected) camera in the action space** — see §6.5.

**Gate for the whole imagination track:** after (1), re-check *does rollout beat persistence?* If yes, Phase-3 imagination becomes viable and you scale (3). If no, escalate capacity/data or reconsider whether LoL is tractable at this scale.

---

## 9. Infrastructure — where everything lives

**Compute (self-hosted cluster):**
- **Login node `danilogin`** — GTX **1060 6GB** (Pascal, fp32). Sees repo at `/srv/nfs/projects/ahriuwu`. Currently running Phase-2 BC.
- **Slurm node `desktop`** — RTX **5080 16GB** (Blackwell sm_120; `torch.compile` broken → `--no-compile`; bf16 fine). Repo at `/mnt/nfs/projects/ahriuwu`. Runs the world-model training. `srun -p gpup --gres=gpu:1` or `sbatch`. `/home/dani` is **per-node** (not shared).
- Python: `/home/dani/miniconda3/envs/ml/bin/python` (desktop = torch 2.10/cu128; login = torch 2.5.1).

**Storage:**
- Repo/NFS: `/srv/nfs` (login) = `/mnt/nfs` (desktop), shared.
- `/srv/nfs/datasets` = 8TB, 5.8TB free — perma copies. Replays: `/srv/nfs/datasets/lol_replays_16_9_772/<match>/{labels.json,clicks.json,raw_cam.json,raw_mem.json}` (147 matches, 146 labeled). BC latents staged to `/srv/nfs/datasets/replay_latents_v7_bc` (125).
- `/mnt/storage` = 3.6TB HDD, **desktop-local** (not visible from login) — checkpoints + tokenizer live here; repo `checkpoints/` symlink → `/mnt/storage/...` (dangling on login, so login jobs must use a real NFS path).
- `/scratch` = NVMe, desktop-local — training copies (`/scratch/ahriuwu/dynamics_all_v7` = 578 combined latents; `dynamics_replay_latents_v7_dim32` = 125 replay).
- Latent format: `{"latents": (N,32,16,16) fp16, "frame_indices": (N,) int32}` per match.

**Cloud (built + validated, mostly idle):**
- **R2** bucket `r2:ahriuwu-yt-pretrain`: dim-32 v7 latents at `dynamics_replay_latents_v7_tok6000_clean/` (125, 51.6 GiB); HUD-fix seed/tokenizer/mask/80-YT-subset backups; 135's accel checkpoint. rclone binary `/home/dani/bin/rclone` (not on PATH), config `~/.config/rclone/rclone.conf`.
- **Vast.ai** DDP path validated on 2×4090 (`scripts/run_ddp_dyn.sh`, `dyn_train_args.sh`, `stage_dyn_latents.sh`, `vast_supervised_dyn.sh`). 4090 = Ada sm_89 (compile works, unlike the 5080). CLI `/home/dani/.vastcli/bin/vastai`.
- **Gotchas:** box needs `opencv-python-headless` (dataset imports cv2 unconditionally) + `wandb` + `rclone`. 8-bit-Adam resume: bitsandbytes optimizer state loads into standard AdamW but dies at `optimizer.step()` with `KeyError: exp_avg` → strip optimizer from the seed. DDP: the short/long alternating choice must be rank-synced. A `0.0*sum(p.sum())` zero-loss tap connects unused params so `find_unused_parameters=False` (forced by compile) doesn't crash.

**Key files:**
- World model: `scripts/train_dynamics.py`, `src/ahriuwu/models/dynamics.py`, `diffusion.py`, `layers.py`; run `slurm/slurm_dyn_yt578.sbatch`.
- Heads/agent: `scripts/train_agent_finetune.py` (Phase 2), `train_imagination.py` (Phase 3), `src/ahriuwu/models/heads.py`, `returns.py`.
- Data: `src/ahriuwu/data/replay_dataset.py` (actions), `dataset.py` (PackedLatentSequenceDataset, YT).
- Eval: `scripts/rollout_check.py` (the decay/ceiling/persistence graph).
- Docs: `Dreamerv4_paper.pdf`, `DYNAMICS_VS_PAPER.md` (paper-vs-code, line-cited), `DYNAMICS_REVIEW.md`, `docs/DREAMERV4_AUDIT.md`.

---

## 10. Open questions / what to test next
1. **Wire the mixed labeled+unlabeled action-conditioned dataset**, run a controlled `use_actions=True` world-model run (start at medium to validate the recipe lifts rollout above persistence — the gate), then scale on cloud.
2. **Re-measure rollout at num_steps 1/4** to separate sampler error from model error (§6.4) and fix the eval default.
3. **Is medium enough capacity?** — the honest unknown; test a larger model on cloud once (1) confirms the recipe.
4. **Camera** — recorded but user-rejected; parked.
5. **LR anneal** — quick encoder win for the demo; separate from the prediction fix.
6. **gold_scale (1e-3)** — placeholder; tune once real return magnitudes are seen (matters for Phase 3, not the WM).

**One-line summary for someone in a hurry:** *We had a world model that predicted well when it could see the player's actions (job 135); we then turned actions off to scale up on unlabeled video and it regressed to worse-than-persistence. The fix is the paper's mixed action-conditioning (labeled actions + unlabeled placeholder), which needs a labeled+unlabeled dataset-mixing tweak; capacity and cloud scale are the follow-ons.*
