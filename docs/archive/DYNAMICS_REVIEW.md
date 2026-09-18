# ahriuwu Dynamics Pipeline — Technical Specification for Review

**Scope of this document.** A fact-based, code-grounded description of the *dynamics* stage of
the ahriuwu DreamerV4-style League-of-Legends world model, written for a PI / external reviewer
with zero prior knowledge. Every non-obvious claim is cited as `file:line` against the actual
source in `/srv/nfs/projects/ahriuwu`. Numbers were read from the code or produced by importing
the model with `PYTHONPATH=src /home/dani/miniconda3/envs/ml/bin/python`; where a value is
computed at import time it is marked **(measured)**.

**Reproduction environment.** `PYTHONPATH=src /home/dani/miniconda3/envs/ml/bin/python`.
Repo git HEAD on the login node at time of writing: `d9d03ce` on branch `main`. Note the
currently-running training job (Slurm 124) was launched from an *earlier* commit and has been
running ~5 days; the code below is the current `main` and may differ in minor ways from the exact
bytes job 124 launched. Checkpoints are self-describing (`model.config` is saved, see
`dynamics.py:988-990`), so architecture is recoverable regardless.

> **A note on paths.** The login node this review was written on does **not** mount `/scratch`
> or `/mnt/storage`; those are local to the Slurm GPU node (`desktop`). The source dataset
> `/srv/nfs/datasets/lol_replays_16_9_772` **is** visible here and was inspected directly. The
> sbatch file uses a `/mnt/nfs/...` spelling of the repo/dataset that is the GPU node's mount of
> the same NFS export seen here as `/srv/nfs/...`.

---

## 1. Goal & Scope

- **What it is.** A DreamerV4-style latent world model for League of Legends, single champion
  **Garen, TOP lane**. The model predicts *future latent frames* of the game given past latent
  frames and the player's actions. `dynamics.py:1-15` cites DreamerV4 (Hafner et al., 2025,
  "Training Agents Inside of Scalable World Models") as the reference architecture.
- **Proof-of-concept / debug scale.** The corpus is ~125–146 replay matches (see §2), a debug /
  bring-up run, not a production-scale training set.
- **Pipeline phases** (this document is Phase 1 only):
  1. **Tokenizer (frozen)** — a transformer image tokenizer ("v7") that encodes each 352×352 frame
     to a latent. Frozen for the dynamics stage.
  2. **Dynamics (this document)** — the latent-space world model. **This is what is currently
     training.**
  3. **Agent BC (Phase 2)** — behavior-cloning a policy/value head on agent tokens. Code exists
     (`use_agent_tokens` path) but is not the active training.
  4. **Imagination / RL (Phase 3)** — training the agent inside the learned world model.
- **Currently training.** Slurm **job 124** (`squeue`: `JOBID 124, PARTITION gpup, NAME dyn-trai,
  ST R`, ~5 days elapsed on node `desktop`), the **medium** model (~114.6M params, §3), launched by
  `slurm/slurm_dyn_train.sbatch`.

---

## 2. Data

### 2.1 Source matches

- **Location:** `/srv/nfs/datasets/lol_replays_16_9_772`.
- **Counts (measured on disk):** 147 match directories (`NA1_*`), **146** with `labels.json`,
  236 casts in the sampled `clicks.json`. Prior project notes put "~125 with latents" — the
  pretokenized latents live on the GPU node's `/scratch` (not visible here) so the exact
  latent-match count could not be re-counted in this pass; the *authoritative* count is whatever
  `provenance.json` in the latents dir records (written by `pretokenize_replay_v7.py:160-166`).
- **Champion:** every sampled match is **Garen** (checked 20 matches; `labels.json.champion ==
  "Garen"` for all). Example: `NA1_5549981347` is Garen, team red, TOP (slot 0), lane opponent
  Kayle.
- **Frames:** `<match>/frames/NNNNNN.png`, **352×352×3** PNG (measured), **20 fps**
  (`labels.json.fps == 20`, measured). One match (`NA1_5549981347`) has 11,202 frames.
- **`labels.json` schema (measured).** Top-level keys: `match_id, champion, team, slot,
  lane_opponent, fps, screen_resolution, frame_resolution, total_frames, projection,
  action_distribution, frames`. Notable values: `screen_resolution == [1280, 720]`,
  `frame_resolution == [352, 352]`. Each element of `frames` has keys `{frame, gt, label}` where
  `frame` is the integer PNG index, `gt` is game-time seconds, and `label` (nullable) has:
  - `champion_screen`, `champion_world`
  - `champion_stats`: `{hp, hp_max, gold, gold_total, level}`
  - `inventory`: length-7 array, each slot `null` or `{id, uc, lf}` (`lf` = last-fired game-time)
  - `visible_heroes`: array of `{name, screen, world, hp, hp_max, gold, gold_total, level}`
  - `action`: `{type, spell, screen}` (`type ∈ {idle, attack, ability, recall, other}`,
    from `action_distribution`)
  - `cursor`: `{world, screen}` (most-recent issued-command location; nullable early-game)
  - `movement`: `{heading_world, heading_screen, speed}` or `null`
- **`clicks.json` schema (measured).** Top-level keys `{match_id, champion, clicks, casts,
  watched}`. Each `cast`: `{game_t, slot, spell_name, hero_x, hero_z, cd_expire, total_cd}`.
  In the sampled match, `spell_name` histogram: `GarenQ 109, GarenE 94, GarenW 14, recall 9,
  GarenR 5, SummonerDot 3, SummonerFlash 2`.

### 2.2 Tokenizer v7 (frozen) and the fold to dynamics latents

- **Tokenizer bottleneck:** **512 latents × 16 dim** (`pretokenize_replay_v7.py:5-6, docstring`).
  The script loads the tokenizer from the checkpoint's *saved `model_config`*, **not** a size
  preset, precisely because the `large` preset (latent_dim 64 / num_latents 256) does **not** match
  v7 (`pretokenize_replay_v7.py:15-21`, `load_v7` at `:38-52`).
- **Encode + fold.** Per frame, `model.encode(...)["latent"]` is `(B, 512, 16)`, then folded:
  `lat.view(B, 16, 16, -1).permute(0, 3, 1, 2)` → **`(B, 32, 16, 16)`**
  (`pretokenize_replay_v7.py:94-97`). So the dynamics latent is a **16×16 grid of 32 channels**
  (latent_dim 32), which is the same 512-bottleneck → 256-spatial reshape DreamerV4 uses
  (`pretokenize_replay_v7.py:11-13`). Encoding runs under `torch.autocast(..., bfloat16)` on CUDA.
- **Output format (one file per match):**
  `<out>/<match>.pt = {"latents": (N, 32, 16, 16) float16, "frame_indices": (N,) int32}`
  (`pretokenize_replay_v7.py:7-8, 219-221`). Default out dir:
  `/scratch/ahriuwu/dynamics_replay_latents_v7_dim32` (`pretokenize_replay_v7.py:174`).
- **Latents are strictly-ascending PNG-numbered** (written in PNG order, `:214, :219`); the
  dataset relies on this contiguity contract (`replay_dataset.py:13-19`).

### 2.3 Action space

- **9 binary keys + 2-D movement.** `constants.ABILITY_KEYS = ['Q','W','E','R','Flash','Ignite',
  'AA','Recall','Stride']` (`constants.py:29`) and `MOVEMENT_DIM = 2` (continuous cursor `(x,y)` in
  `[0,1]` screen coords, `constants.py:14`).
- **How each key is parsed** (`replay_dataset.py:317-385`, mapping table `_SPELL_TO_KEY` at
  `:59-66`):
  - **Q/W/R:** `clicks.json` cast with `spell_name` `GarenQ/GarenW/GarenR`, set on the frame whose
    time matches the cast (`round((game_t - gt0)/step)`, `:363`).
  - **E:** `GarenE` **and** `GarenECancel` both map to `E` — they are per-match aliases for
    "E used", verified mutually exclusive across matches, so there is no separate cancel key
    (`constants.py:19-21`, `replay_dataset.py:62-64`).
  - **Flash / Ignite:** `SummonerFlash` / `SummonerDot`.
  - **Recall:** cast `spell_name == "recall"`, with fallback to `label.action.type == "recall"`.
  - **AA:** not in the cast stream — set on frames where `label.action.type` *transitions into*
    `"attack"` (`_fill_aa_from_attack`, `:387-402`).
  - **Stride:** Stridebreaker (item id **6631**) active, detected when the item's `lf` (last-fired
    game-time) jumps up (`_fill_stride_from_inventory`, `:404-427`). Sparse (~2–18/game).
  - **Dropped** (no clean signal): pots, tiamat, ward, TP, super-recall (`constants.py:22-24`).
  - **Movement (x,y):** from `label.cursor.screen` normalized by `labels.screen_resolution`,
    held-forward through idle/unlabeled frames, falling back to `(0.5, 0.5)` if no cursor info yet
    (`_parse_movement`, `:280-315`). Older labels fall back to `movement.heading_screen`.
- **NaN-safe embedding.** `embed_actions` supports an optional `cursor_valid` mask that swaps the
  movement projection for a learned `no_action_embed` on frames with invalid (NaN) movement
  (`dynamics.py:536-570`). **The replay dataset does not emit `cursor_valid`** (`__getitem__`
  returns only `movement` + ability keys, `replay_dataset.py:494-497`), so at train time the
  "legacy / fully-labeled" branch runs (`dynamics.py:562-564`) and `no_action_embed` is never used
  by the forward — which is exactly why the DDP zero-tap exists (§4).

### 2.4 Sequence construction

- **Action-conditioned path (what job 124 uses): `ReplayLatentSequenceDataset`**
  (`replay_dataset.py:96`). Constructed with `sequence_length`, `stride`, `labels_root`, and
  dummy `outcomes` (all False — see §4). It:
  - Indexes windows over **contiguous frame runs only** (no gaps inside a window),
    `stride`-spaced (`_index_match` / `_emit_windows`, `:205-245`).
  - Enforces `labels.frames[i].frame == i` (spot-checks first 64, hard-fails otherwise,
    `:260-266`), and warns if latent frame-count ≠ label frame-count, using the min (`:186-192`).
  - `__getitem__` returns per sequence:
    `{"latents": (T, C, H, W) float, "actions": {"movement": (T,2), <9 ability keys>: (T,) long},
    "rewards": (T,), "video_id": str, "start_frame": int}` (`:480-504`).
  - Has a 2-deep per-worker LRU latent cache (`max_cache_size=2`, `:456-473`); each packed file is
    ~200 MB.
- **Latents-only path (actions OFF): `PackedLatentSequenceDataset`** (`dataset.py:266`), same
  packed format, emits `{"latents", "video_id", "start_frame"}`, channel count `C` never
  hardcoded (`dataset.py:279-282`). Job 124 does **not** use this (it passes `--use-actions`).
- **Sampler: `VideoGroupedSampler`** (`dataset.py:213-237`) — shuffles video order each epoch and
  sequence order within a video, but keeps a video's sequences contiguous so the LRU cache hits.
  Supports `exclude_videos` for held-out eval. (There is also a seedable `VideoShuffleSampler`,
  `:240-263`, unused by the training loop.)

---

## 3. Dynamics Model Architecture

Source: `src/ahriuwu/models/dynamics.py` (model) and `src/ahriuwu/models/layers.py` (attention /
norms / RoPE). All shapes below **measured** by importing the exact medium config job 124 uses.

### 3.1 Size preset (medium) — exact config

`create_dynamics("medium", latent_dim=32, use_actions=True, num_kv_heads=4,
num_register_tokens=8, soft_cap=50.0, use_qk_norm=True, gradient_checkpointing=True)` yields
(`dynamics.py:943-991`, preset table `:943-964`):

| Field | Value | Source |
|---|---|---|
| `model_dim` (embed_dim) | **768** | preset `medium` `dynamics.py:954-958` (measured) |
| `num_layers` | **18** | same (measured) |
| `num_heads` (query heads) | **12** | same (measured) |
| `num_kv_heads` (GQA) | **4** | CLI `--num-kv-heads 4` (measured) |
| `head_dim` | **64** (= 768/12) | `layers.py:316` (measured) |
| `latent_dim` | **32** | CLI `--latent-dim 32` |
| spatial grid | **16×16 = 256** latent tokens | `dynamics.py:410, 971` |
| `num_register_tokens` | **8** | CLI (measured) |
| `temporal_every` | **4** | default `dynamics.py:354` |
| `soft_cap` | **50.0** | CLI / Gemma-2 convention |
| `use_qk_norm` | **True** | default |
| `max_seq_len` (temporal cap) | **256** | `dynamics.py:356` |
| attention `scale` | **1.0** | QKNorm ⇒ 1/√d dropped (`layers.py:322-325`) (measured) |
| SwiGLU hidden dim | **2048** | `layers.py:107-111` (768·8/3 → round-64) (measured) |
| **Total trainable params** | **114,584,864 (~114.6M)** | (measured) |

For reference (measured): medium **without** actions = 114,567,968; the action embeddings add
**16,896** params.

### 3.2 Token layout (per frame)

`_build_tokens` (`dynamics.py:742-795`) builds a per-frame token sequence, concatenated along the
spatial axis in this order:

```
[ 256 latent tokens (2D RoPE) | 8 register tokens | 1 action token | 1 condition token ]
```

- Total spatial tokens per frame = `256 + 8 + 1(action) + 1(cond) = 266` (measured;
  `num_extra_tokens == 2` because actions are on, `dynamics.py:427-432`).
- **Input projection:** `Linear(32 → 768)` applied to each of the 256 latent tokens
  (`dynamics.py:435, 749-750`).
- **Register tokens:** 8 learnable vectors shared across all frames (`dynamics.py:438-441,
  752-754`).
- **Action token:** one token/frame = **sum** of factorized action embeddings — `Linear(2→768)`
  for movement plus a 2-way `Embedding` per ability key, all summed (`embed_actions`,
  `:536-570`; module dict at `:457-463`). Appended at `:756-761`.
- **Condition token:** one token/frame from `_build_condition_token` (`:572-616`): `tau` and
  `step_size` each mapped to a discrete `Embedding(768)`, concatenated and projected
  `Linear(2·768 → 768)`. Appended at `:793-794`.

### 3.3 tau / step-size conditioning — injected **twice**

Conditioning is added **both** additively to every spatial token **and** as the appended condition
token:

- **Additive to all tokens** (`_build_tokens`, `dynamics.py:763-776`):
  `x = x + tau_emb.unsqueeze(2) + step_emb.unsqueeze(2)` — the `tau_embed` and `step_embed`
  lookups broadcast onto all 266 tokens of that frame.
- **Appended condition token** (`:793-794`) — the projected `[tau_emb ‖ step_emb]`.

`tau` is quantized to a discrete index by `tau_idx = (tau * k_max).long().clamp(0, 63)`
(`:596, 765`), with **`k_max = 64`**, `num_tau_levels = 64`, `num_step_sizes = 7`
(d ∈ {1,2,4,8,16,32,64}, indexed by `log2(d)`) (`dynamics.py:448-452`, measured). This discrete
lookup is intentional (DreamerV4 "discrete signal levels", `:592-595`).

(A `use_game_time` conditioning path exists — additive per-frame bucket embedding with train-time
dropout — but is **off** for job 124: the sbatch passes no `--use-game-time`, and `create_dynamics`
defaults `use_game_time=False`, `dynamics.py:913`. Details `:465-474, 778-791`.)

### 3.4 Factored space-time attention

Blocks alternate spatial and temporal attention; **temporal attention is every 4th layer**
(`is_temporal = (i % temporal_every == temporal_every - 1)`, `dynamics.py:478-496`). With 18 layers
and `temporal_every=4`, the block types are (measured):

```
[S S S T  S S S T  S S S T  S S S T  S S]   → 14 spatial, 4 temporal
```

(temporal at 0-indexed layers 3, 7, 11, 15).

- **Spatial block** (`TransformerBlock`, attn_type `"spatial"`): reshapes `(B,T,S,D) → (B*T,S,D)`
  and runs **full, non-causal** self-attention over the 266 tokens **within each frame,
  independently** (`dynamics.py:119-123`). 2D RoPE is applied to the **first 256 tokens only**
  (the latent grid); register/action/cond tokens get no rotation
  (`_apply_rope_spatial_prefix`, `layers.py:458-484`).
- **Temporal block** (attn_type `"temporal"`): reshapes `(B,T,S,D) → (B*S,T,D)` and runs
  **causal** self-attention **across frames** for each of the 266 spatial positions
  (`dynamics.py:124-131`). 1D RoPE over temporal positions (`layers.py:518-531`).
- **CAUSALITY (read precisely).** Temporal attention is **causal**: frame `t` attends to frames
  `≤ t`. Two code paths implement this identically:
  - flex path: `mask_mod = _causal_mask_mod` = `q_idx >= kv_idx` (`layers.py:230-231, 649-651`).
  - manual path: `mask = ~causal_mask[:T,:T]` where `causal_mask = triu(ones, diagonal=1)`
    (upper triangle above the diagonal masked out) (`layers.py:355-358, 653-659`).
  During training, with probability `independent_frame_ratio` (default **0.3**, §4) the temporal
  mask is replaced by a **diagonal** mask (`_diagonal_mask_mod` / `torch.eye`, each frame attends
  only to itself), the DreamerV4 "no temporal context" mode (`layers.py:234-235, 654-655`;
  `dynamics.py:683-686`).
- **Backend.** `TransformerBlock` builds its `Attention` with `allow_flex=False`
  (`dynamics.py:93`) so blocks always use the manual matmul path — flex + gradient checkpointing
  can OOM (`layers.py:262-273`). QKNorm is applied before RoPE; soft-cap 50 is applied to the
  logits (`layers.py:438-450`, `soft_cap_attention` `:83-98`).

### 3.5 Normalization / FFN / init

- **RMSNorm** everywhere (eps 1e-6, Gemma-2 default; hardcoded, `layers.py:23-44`).
- **QKNorm** per-head over head_dim before RoPE, scale forced to 1.0 (`layers.py:47-80, 322-325`).
- **SwiGLU** FFN, hidden `int(dim·8/3)` rounded to multiple of 64 (`layers.py:101-119`).
- **Weight init** (`_init_weights`, `dynamics.py:631-656`): xavier-uniform linears, normal(0.02)
  embeddings, residual out-projections (attn `out_proj`, SwiGLU `w3`) scaled by
  `1/sqrt(2·num_layers)`, and the final `output_proj` scaled by gain **0.02** (near-identity
  residual output).
- **Output head.** After the block stack, `_project_out` strips the non-latent tokens (keeps first
  256), RMSNorm, `Linear(768→32)`, reshapes back to `(B,T,32,16,16)` (`dynamics.py:816-821`).

### 3.6 Agent tokens (Phase 2+, OFF here)

`use_agent_tokens` (default False, `dynamics.py:363`) adds a per-frame agent token with
cross-attention to z-tokens + causal temporal self-attention (`AgentCrossAttention` `:137-236`,
`AgentTokenBlock` `:239-321`). Job 124 does **not** enable it (no `--use-agent-tokens` flag), so
these params exist only if built; they are not built in job 124's model.

---

## 4. Training Objective & Config

Source: `src/ahriuwu/models/diffusion.py` (objective), `scripts/train_dynamics.py` (loop),
`slurm/slurm_dyn_train.sbatch` (launched config).

### 4.1 Diffusion forcing (the objective actually used)

- **Flow-matching / x-prediction noise schedule** (`DiffusionSchedule`, `diffusion.py:21-64`):
  `z_τ = τ·z_0 + (1−τ)·ε`, convention **τ=1 clean, τ=0 pure noise** (`:24-30, 61-62`). The model
  predicts the **clean latent z_0 directly** (x-prediction).
- **Per-frame τ schedule** (`sample_diffusion_forcing_timesteps`, `diffusion.py:89-144`), with
  default `tau_ctx=0.9`, `tau_min=0.0`:
  - A random **horizon** `h ~ randint(1, T)` per sequence (`:119`).
  - **Context frames** (`positions < h`): near-clean, `τ ~ U(tau_ctx, 1.0) = U(0.9, 1.0)`
    (`:132-133`).
  - **Target frames** (`positions ≥ h`): `τ` ramps down from `tau_ctx` toward `tau_min` with
    distance past the horizon, **plus** `N(0, 0.02²)` jitter, clamped to `[tau_min, tau_ctx]`
    (`:135-140`). So targets span `[0, 0.9]` and context is `[0.9, 1.0]`. (By construction the
    context band is only ~10% of the τ range; project notes flag this as a "context-heavy" schedule
    that is by design.)
- **`add_noise`** produces `z_τ` and returns the noise (`diffusion.py:35-64`).
- **x-prediction loss** (`x_prediction_loss`, `diffusion.py:221-258`): per-element MSE, reduced over
  channel+spatial to per-frame, times a **ramp weight**.
- **Ramp weight** (`ramp_weight`, `diffusion.py:202-218`): **`w(τ) = 0.9·τ + 0.1`** — clean frames
  weighted 1.0, pure noise 0.1 (DreamerV4 Eq. 8, paper convention).
- **Forward at train time** (`_forward_standard`, `train_dynamics.py:977-989`): samples the
  per-frame τ, adds noise, calls `model(z_τ, τ, actions=actions,
  independent_frames=use_independent)`, computes the ramp-weighted x-pred loss.
- **RMS loss normalization** (`RunningRMS`, `returns.py:334-395`): the raw loss is divided by its
  EMA-RMS (decay 0.99, min-RMS 1e-4 floor) and that **normalized** loss is what is backpropagated
  (`train_dynamics.py:984-988, 770`). A non-finite loss is *not* folded into the RMS statistic
  (`returns.py:368-371`) — a guard added specifically because a single NaN would pin the RMS to NaN
  forever.

### 4.2 Stability guards

- **Grad clip 1.0** (`clip_grad_norm_(..., max_norm=1.0)`, `train_dynamics.py:792, 911`).
- **NaN-skip guard.** On the bf16 path `GradScaler` is a no-op, so the code explicitly gates the
  optimizer step on `torch.isfinite(grad_norm)`; a non-finite grad-norm ⇒ step skipped, grads
  zeroed, warning printed (`train_dynamics.py:798-818`). Same at the end-of-epoch flush (`:912-925`).
- **DDP / zero-loss tap.** Before backward, `scaled_loss += 0.0 * Σ p.sum()` over all trainable
  params (`train_dynamics.py:784-787`). This (a) connects conditionally-unused params
  (`no_action_embed` — never used because `cursor_valid` is not emitted, §2.3 — plus `no_gt_embed`
  and agent-token params) so a multi-GPU DDP reducer with `find_unused_parameters=False` doesn't
  crash, and (b) makes a zero shortcut loss backward-able. Zero contribution ⇒ gradients unchanged.
- **`independent_frames`** applied per-batch with probability `--independent-frame-ratio` (default
  **0.3**, `train_dynamics.py:447-452, 741`).

### 4.3 Optimizer & schedule

- **Optimizer: AdamW, `betas=(0.9, 0.95)` — hardcoded at the call site**
  (`train_dynamics.py:1273`: `create_optimizer(..., betas=(0.9, 0.95))`). Note this **overrides**
  the `create_optimizer` default `(0.9, 0.999)` (`training.py:273`) and the unused
  `--adam-betas` CLI default `(0.9, 0.999)` (`training.py:206-213`). **weight_decay = 0.1**
  (default, `training.py:181-186`; sbatch does not override).
- **8-bit Adam:** `--use-8bit-adam` defaults **True** (`training.py:214-219`), and the sbatch does
  **not** pass `--no-use-8bit-adam`. So *if* bitsandbytes is installed on the GPU node, the
  optimizer is `bnb.optim.AdamW8bit`; otherwise it silently falls back to `torch.optim.AdamW`
  (`training.py:276-283`). (bitsandbytes is **not** installed on this login node — measured — so
  the GPU-node state governs which path job 124 took.)
- **LR schedule: WSD (Warmup-Stable-Decay)**, `create_wsd_schedule` (`training.py:286-299`).
  With sbatch args: **lr = 3e-4**, **warmup = 3000 steps**, **decay-steps = 0** (default,
  `training.py:193-198`; sbatch does not pass `--decay-steps`). So the schedule is linear warmup to
  full LR over 3000 steps then **held flat** (no decay phase) for the run.
- **Epochs = 50** (`--epochs 50`). `total_steps` for the schedule is derived from the alternating
  short/long dataloader lengths (`train_dynamics.py:1276-1284`).

### 4.4 Batch / sequence config (exact, from `slurm/slurm_dyn_train.sbatch`)

The launched command (`slurm_dyn_train.sbatch:33-63`):

- `--model-size medium --latent-dim 32 --tokenizer-type transformer`
- `--use-actions --labels-root /mnt/nfs/datasets/lol_replays_16_9_772`
- **Alternating context lengths:** `--alternating-lengths --seq-len-short 128 --seq-len-long 256`
  (DreamerV4 §3.4 style, `train_dynamics.py:297-345`).
- **Batch sizes:** `--batch-size-short 2 --batch-size-long 1`.
- **Long ratio:** `--long-ratio 0.1` (10% of accumulation windows use the 256-length loader,
  `train_dynamics.py:717-719`).
- **Grad accumulation:** `--gradient-accumulation 8`, `--gradient-accumulation-short 8`,
  `--gradient-accumulation-long 16`. Effective batch: short = 2×8 = **16 sequences**, long =
  1×16 = **16 sequences**. Per-length accumulation windows process a single type each
  (`train_dynamics.py:684-694, 716-719`).
- **`--gradient-checkpointing`** (enabled; ~2× activation-memory reduction, `dynamics.py:797-814`).
- **`--no-compile`** — **torch.compile is OFF** (`train_dynamics.py:1261-1263`).
- **`--num-kv-heads 4 --num-register-tokens 8 --soft-cap 50.0`.**
- **`--lr 3e-4 --warmup-steps 3000 --eval-interval 200 --checkpoint-minutes 60 --epochs 50`.**
- **Mixed precision:** `bfloat16` autocast (`train_dynamics.py:757, 1295`); `GradScaler` enabled
  only for fp16, i.e. a no-op here (`:1296`).
- **Shortcut forcing is NOT passed** (`--shortcut-forcing` absent). See §6.
- **Single GPU** (`--gres=gpu:1`, `slurm_dyn_train.sbatch:5`), an **RTX 5080** per project notes;
  partition `gpup`, 4 CPUs, 18 GB RAM, 14-day walltime, `--requeue` with a TERM-at-120s preemption
  signal (`:5-13`).
- **Resume:** the sbatch appends `--resume $CHECKPOINT_DIR/dynamics_latest.pt` if it exists
  (`:64-66`).

### 4.5 Eval during training

Two evals run every `--eval-interval` (200) steps (`train_dynamics.py:865-884`):

1. **Teacher-forced 1-step denoising PSNR** (`eval_denoising_psnr`, `:485-552`): for τ ∈
   {0.1,0.3,0.5,0.7,0.9}, corrupt clean latents to `z_τ`, predict clean in one step, PSNR vs z_0.
2. **Free-running autoregressive rollout PSNR** (`eval_rollout_psnr`, `:555-618`): prefills 16
   real context frames, rolls out 32 via the KV-cached rollout (§5), reports **per-horizon** PSNR.
   For the base (no-shortcut) model it uses `num_steps=16, k_max=16` (d=1) (`:875`). Project notes
   caution that absolute rollout PSNR reads ~13 dB even for a collapsed predictor, so the
   **per-horizon curve** is the signal, not the absolute number.

Best-checkpoint selection tracks the **teacher-forced** mean PSNR (`train_dynamics.py:1402-1428`).

---

## 5. Inference / Rollout

Source: `DynamicsTransformer.rollout` (`dynamics.py:823-899`) + `Attention.forward_temporal_cached`
(`layers.py:675-722`). Eval harness: `scripts/rollout_check.py`.

- **KV-cached autoregressive rollout.** One cache dict `{k, v, pos}` per **temporal** block
  (`dynamics.py:860-861`). Spatial attention is per-frame and cache-free; only the 4 temporal blocks
  read/write caches (`_run_blocks`, `:797-814`, dispatches temporal blocks to
  `forward_temporal_cached`).
- **Prefill.** Context latents get τ ~ `U(1−tau_ctx, 1)` (near-clean; `tau_ctx=0.1` in the eval
  harness ⇒ `U(0.9,1.0)`), tokens are built and run through the stack with `append=True` so the
  caches fill; the output is discarded (`dynamics.py:864-867`).
- **Per future frame** (`dynamics.py:873-895`): start from Gaussian noise, denoise in `num_steps`
  Euler steps from τ=eps to τ≈1 (`step = (1−eps)/num_steps`, `eps=1e-3`, `:869-870, 879-891`).
  Each sub-step calls the stack with `append=False` (transient — the in-progress noisy frame must
  not pollute the cache). After the last step the clean frame is **committed** with τ=1 and
  `append=True`, writing its K/V into every temporal cache (`:892-895`).
- **Shortcut step size.** `d = max(1, k_max // num_steps)` (`dynamics.py:871`). **The base model
  uses d=1** — i.e. `num_steps == k_max` (e.g. the eval harness passes `num_steps=16, k_max=16`,
  `train_dynamics.py:875`; `rollout_check.py` passes `k_max=args.num_steps`, `:108-110`). The
  shortcut regime (`num_steps=4, k_max=64 ⇒ d=16`) is only meaningful once shortcut forcing has been
  trained (§6).
- **Numerical equivalence.** With an empty cache and `append=True`, the cached temporal forward is
  numerically identical to the standard causal temporal forward (`layers.py:687-690`); the token
  builder and output projection are shared byte-for-byte between `forward` and `rollout`
  (`dynamics.py:698-704, 738-741`).
- **How rollout differs from training.** Training gives every frame its own τ from the
  diffusion-forcing schedule (context high-τ, targets low-τ) and does a **single forward** over the
  full sequence with the true (clean) latents present as context. Rollout instead (a) treats the
  past as **committed, near-clean context** in a KV cache, (b) generates each future frame from pure
  noise via an iterative Euler denoise, and (c) feeds its **own predictions** back as context for
  subsequent frames — so errors can **accumulate**, which teacher-forced training never exercises.
- **Eval harness `scripts/rollout_check.py`.** Loads a dynamics checkpoint
  (`create_dynamics("medium", latent_dim=32, use_actions=True, num_kv_heads=4,
  num_register_tokens=8, soft_cap=50.0)`, `:67-68`), picks the most *dynamic* window of a match
  (max movement-std + button presses, `:83-89`), runs teacher-forced 1-step PSNR (`:97-98`) and the
  autoregressive rollout with the **real recorded actions** (`:105-119`), and with `--decode`
  renders a 3-row GT / tokenizer-reconstruction / dream comparison video + a per-frame pixel-PSNR
  plot that includes a **persistence baseline** (hold last real frame, `:175, 183`). Runs on CPU by
  default so it overlaps live training. Default `--ctx 6 --horizon 8 --num-steps 6` (`:43-45`); a
  Pascal-bf16 guard forces fp32 on old GPUs to avoid emulated-bf16 drift (`:60-63`).

---

## 6. Known Deviations from DreamerV4

All grounded in code/comments; these are deliberate and documented in-repo.

1. **Shortcut bootstrap loss in x-space, not paper v-space.** The paper's Eq. 7 bootstrap target is
   in velocity space with `(1−τ)²` weighting; this repo computes the bootstrap loss **directly in
   x-space** (`x_diff = z_pred − z_target.detach()`, clamped to 100, ramp-weighted), explicitly to
   avoid the bf16 `÷(1−τ)` then `×(1−τ)²` precision trap and the teacher≈student zero-velocity trap
   (`diffusion.py:489-514`). Project notes rate this benign at small scale (K4 tracks K64 within
   ~1 dB) but flag it for re-test at production scale.
2. **Shortcut forcing OMITTED from this base run.** Job 124 is standard x-prediction diffusion
   forcing only; the sbatch header states shortcut's GC-disabled 3-forward-pass steps OOM
   medium@128 on the 5080, so shortcut distillation is deferred to a **separate Phase B**
   (`finetune_shortcut.py`) (`slurm_dyn_train.sbatch:29-32`). The whole `ShortcutForcing` machinery
   (`diffusion.py:263-535`, progressive `max_step_idx` curriculum in `_forward_shortcut`,
   `train_dynamics.py:946-974`) is present but inactive here.
3. **τ / step conditioning added to ALL tokens *and* as an appended token.** The paper uses a
   conditioning token only; this repo does both (§3.3, `dynamics.py:763-776` additive + `:793-794`
   appended).
4. **Temporal attention only every 4th layer** (14 spatial : 4 temporal). This matches the repo's
   stated efficiency choice (`dynamics.py:6-8, 478-480`) and is the DreamerV4 factorization, but the
   sparsity of temporal layers is a knob a reviewer should weigh against forward-prediction quality
   (see §7).
5. **Reward / value twohot ranges** are narrower than Dreamer's `[−20,20]` — project notes record a
   move to ±3.0 symlog under a *solo-Garen-gold* reward (`heads.py`, per memory). This is a Phase 2+
   concern (rewards are unused by the dynamics x-pred loss; dummy `outcomes=False` are passed,
   `train_dynamics.py:1124-1145`) and was not re-verified in this pass.
6. **RMSNorm eps hardcoded 1e-6** (not a config knob) and **QKNorm scale=1.0** (Gemma-2), a
   deliberate choice documented at `layers.py:23-44, 322-325`.
7. **Context-heavy τ schedule** — only ~10% of the τ mass is the near-clean context band by
   construction (`diffusion.py:132-140`); flagged in project notes as intentional but worth a look.

---

## 7. Current Empirical State (as reported)

> The training node's wandb/scratch/checkpoints are not mounted on the login node, so the numbers
> in this section are **as reported by the task brief and corroborating project notes**, not
> re-measured here. The *mechanism* each number comes from is cited.

- **Job 124 is live**: `squeue` shows it Running, ~5 days elapsed, node `desktop`.
- **Teacher-forced denoising PSNR ~25 dB at τ=0.9**, climbing but converging across steps
  (~1.8k → ~18k). This is the `eval_denoising_psnr` τ=0.9 metric (`train_dynamics.py:485-524`).
- **Training loss ~0.005** (the raw x-pred loss `train/loss`, `train_dynamics.py:1019`).
- **Grad-norm settling ~1.5–2.5** (`train/grad_norm`, clipped at 1.0 only when it exceeds it;
  note the console `GRAD_CLIP` warning fires at >0.99, `train_dynamics.py:1012`).
- **PredStd ~0.6** (std of the model's clean-latent prediction, sampled every 10 batches,
  `train_dynamics.py:830-833`) — well above the mode-collapse alarm at <0.01 (`:1389-1390`).
- **THE DEBUG SYMPTOM.** On a **dynamic clip that is in the training set**, the **autoregressive
  rollout falls below a persistence baseline** (hold the last real frame) for ~13 frames. I.e. the
  model is a **good 1-step denoiser but a poor forward predictor**, on its own training data. This
  is exactly the metric/regime decoupling `eval_rollout_psnr` and `rollout_check.py`'s persistence
  curve were built to expose (`train_dynamics.py:555-582`; `rollout_check.py:169-195`), and mirrors
  the project's "eval measured the wrong regime" tokenizer lesson (teacher-forced 1-step vs
  free-running K-step rollout).

### 7.1 Reviewer hooks (where to look / re-measure)

- Re-run the deploy-regime eval to reproduce the symptom:
  `PYTHONPATH=src python scripts/rollout_check.py --decode --horizon 18` (uses `dynamics_latest.pt`
  + the v7 tokenizer; emits per-horizon PSNR incl. the persistence baseline).
- Candidate causes a reviewer would naturally probe, given the code: (i) temporal attention is only
  4 of 18 layers (§3.4); (ii) the τ schedule is context-heavy so the model sees relatively little
  training signal for *deep* target frames far past the horizon (§4.1); (iii) `independent_frames`
  at 0.3 deliberately trains 30% of batches with **no** temporal context, which strengthens the
  per-frame denoiser but not the temporal predictor (§3.4/§4.2); (iv) rollout feeds back the model's
  own (tokenizer-lossy, denoiser-imperfect) latents, unlike the clean-context training regime (§5).
  These are hypotheses the *structure* of the code motivates — not asserted conclusions.

---

### Appendix: file map

| Concern | File |
|---|---|
| Model (transformer, rollout, factory) | `src/ahriuwu/models/dynamics.py` |
| Attention / RoPE / RMSNorm / QKNorm / SwiGLU | `src/ahriuwu/models/layers.py` |
| Diffusion schedule, x-pred loss, ramp weight, shortcut | `src/ahriuwu/models/diffusion.py` |
| Training loop / evals / CLI | `scripts/train_dynamics.py` |
| Optimizer, WSD schedule, checkpoint I/O | `src/ahriuwu/utils/training.py` |
| RunningRMS | `src/ahriuwu/models/returns.py` |
| Action-conditioned dataset | `src/ahriuwu/data/replay_dataset.py` |
| Latents-only dataset + samplers | `src/ahriuwu/data/dataset.py` |
| Action space constants | `src/ahriuwu/constants.py` |
| Pretokenization (frozen v7 → dim-32 latents) | `scripts/pretokenize_replay_v7.py` |
| Launched training config | `slurm/slurm_dyn_train.sbatch` |
| Rollout / dream eval harness | `scripts/rollout_check.py` |
