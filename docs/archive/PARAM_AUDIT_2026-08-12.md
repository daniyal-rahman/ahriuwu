# Model-hyperparameter construction audit — 2026-08-12

Scope: every site in `scripts/`, `scripts/probes/`, `src/`, `slurm/`, `ops/` that builds
`create_dynamics` / `DynamicsTransformer`, `TransformerTokenizer` / `create_transformer_tokenizer`,
or the heads in `models/heads.py`, and every site that loads a checkpoint into one.

Method: ground truth read from the shipped checkpoints' own saved config
(`torch.load(..., map_location="meta", mmap=True)`), then **each construction site's exact
argument set was rebuilt on the meta device and its `state_dict()` diffed against the
checkpoint's** — so every "N tensors dropped" number below is measured, not inferred.
Probe script: `scratchpad/param_audit_probe.py` (read-only, no GPU).

Line numbers are as of this audit (`scripts/agent_infer.py` and `scripts/train_agent_finetune.py`
have uncommitted working-tree changes that shifted them).

---

## 0. Ground truth — what the shipped checkpoints actually are

| checkpoint | source key | architecture |
|---|---|---|
| `rollout_stage/transformer_tokenizer_latest.pt` | `model_config` | img 352, patch 16, **embed 1024**, **latent_dim 16**, heads 16, **8+8 layers**, **num_latents 512**, **temporal_every 4**, rope, qk_norm, soft_cap 50, max_time 256, `size_preset='large'` |
| `rollout_stage/desktop_resume_8775_stripped.pt` | `model_config` | medium (768/18/12), latent_dim 32, spatial 16, **num_kv_heads 4**, reg 8, **qk_norm True**, soft_cap 50, **k_max 64**, **use_actions True**, use_agent_tokens False, use_game_time False |
| `data/phase2_bc_gate1060/agent_finetune_latest.pt` | `dynamics_config` | same as above **+ use_agent_tokens True**, agent_layers 4 |
| — same, heads | `args` | hidden 256, num_buckets 255, mtp 9, movement_bins 21, **movement_gate True**, StateHead num_targets 4 |

Two structural facts that drive most of the findings:

1. **`args` and `model_config` are different dictionaries with different keys.** The dynamics
   trainer writes `no_qk_norm` into `args`; it never writes `use_qk_norm`. It derives
   `use_actions` into `model_config`; `args.use_actions` also exists but `args.use_agent_tokens`
   is the CLI flag, not the resolved value. Reading an arch flag from `args` when the authority
   is `model_config` is the single most repeated defect here.
2. **`load_state_dict(strict=False)` only tolerates missing/unexpected KEYS. Shape mismatches
   raise `RuntimeError` regardless of `strict`.** So the danger ranking of a parameter is
   determined by whether getting it wrong changes a *tensor's shape* (loud) or changes
   *which tensors exist* / *pure numerics* (silent).

---

## 1. Ground truth — every constructor parameter and its danger class

### 1a. `create_dynamics` → `DynamicsTransformer`

| param | factory default | what it changes | failure mode when omitted/wrong |
|---|---|---|---|
| `size` | `"small"` (512/12/8) | every weight shape | **LOUD** (shape) |
| `latent_dim` | `32` | `input_proj`, `output_proj` | **LOUD** (shape) |
| `use_qk_norm` | `True` | 36 `qk_norm.*` tensors at medium **and** attn `scale` 1.0 ↔ `head_dim**-0.5` | **SILENT + catastrophic** |
| `soft_cap` | `50.0` | pure numerics, **zero tensors** | **TOTALLY SILENT.** `0.0` → `0*tanh(x/0)` → all logits 0 (uniform attention) or NaN |
| `use_actions` | `False` | 12 `action_embed.*`/`no_action_embed` tensors **and** `num_extra_tokens` 2→1, i.e. sequence length 266→265 | **SILENT** |
| `use_agent_tokens` | `False` | 88 agent tensors (`agent_token`, `task_embed`, `agent_temporal_pos`, `agent_blocks.*`, `agent_norm_out`) | **SILENT** drop; forward return arity changes so the *call site* usually crashes |
| `agent_layers` | `4` | number of agent blocks | **SILENT** (missing/unexpected) |
| `use_game_time` | `False` | `gt_embed` + `no_gt_embed` and the additive game-time conditioning | **SILENT** |
| `game_time_bucket_seconds` | `30.0` | seconds→bucket mapping, numerics only | **SILENT** |
| `gt_dropout` | `0.1` | train-time only | silent, train-only |
| `num_register_tokens` | `8` | `register_tokens` param + sequence length | **SILENT** when `0` vs `8` (unexpected key); LOUD when `4` vs `8` (shape) |
| `num_kv_heads` | `None` (=MHA) | `k_proj`/`v_proj` shape | **LOUD** (shape). Note `0` is *falsy* → `num_kv_heads or num_heads` silently gives MHA |
| `k_max` | `64` | `tau_embed` rows (=k_max), `step_embed` rows (=log2+1), τ quantization | LOUD (shape) — **but its required equality with `ShortcutForcing.k_max` is unchecked and SILENT** |
| `num_tasks` | `1` | `task_embed` rows | LOUD (shape), only when agent tokens on |
| `game_time_num_buckets` | `120` | `gt_embed` rows | LOUD, only when `use_game_time` |
| `gradient_checkpointing` | `False` | nothing persistent | safe |
| `spatial_size` | **hardcoded 16 in the factory** | token count, RoPE grid | not overridable; forward `assert` (loud) |
| `temporal_every` | **not a factory param** (4) | which blocks are temporal vs spatial → different buffer sets (`causal_mask`, `rope.positions_*`) and `rope.inv_freq` shape | LOUD (shape) if it ever drifted |
| `max_seq_len` | **not a factory param** (256) | `agent_temporal_pos` rows, `causal_mask` size | LOUD (shape) |
| `head_dim`, `dropout` | **not factory params** | head_dim → all attn shapes | LOUD |

> **`model.config` (saved as `model_config`/`dynamics_config`) does NOT record
> `temporal_every`, `max_seq_len`, `head_dim`, or `dropout`.** The dynamics checkpoint is
> therefore *not fully self-describing*: a future change to any of those four is unrecoverable
> from the checkpoint alone. `TransformerTokenizer.config` records every constructor arg and
> does not have this hole.

### 1b. `create_transformer_tokenizer` → `TransformerTokenizer`

The factory's `resolved` dict **omits** `patch_size` (16), `dropout` (0.0), `use_sincos_pos`
(True), `max_time` (256), and **`temporal_every` (2)**, and **hardcodes `img_size=352`**.

| param | factory default | failure mode |
|---|---|---|
| `embed_dim` / `num_heads` / `num_*_layers` / `num_latents` / `latent_dim` | from `size` preset | LOUD (shape) — presets are `tiny 256/16`, `small 512/32`, `medium 768/48`, `large 1024/64`, all `num_latents 256` |
| **`temporal_every`** | **2** (v7 trained at **4**) | LOUD — measured: 4 shape mismatches (`*.attn.rope.inv_freq` 32 vs 16), 8 dropped `rope.positions_*`, 4 missing `causal_mask` |
| `img_size` | hardcoded 352 | a 256-px checkpoint cannot be rebuilt by the factory at all |
| `use_rope`, `use_qk_norm`, `soft_cap` | True/True/50.0 | rope → shape (LOUD); qk_norm → SILENT; soft_cap → TOTALLY SILENT |

**Key consequence:** the v7 tokenizer's real architecture (`latent_dim 16`, `num_latents 512`,
`8+8`, `temporal_every 4`) matches **no** size preset. Any script that rebuilds it from a
preset now fails loudly on shape — but only by luck of the arch having diverged.

### 1c. Heads

| head | param | default | failure mode |
|---|---|---|---|
| `PolicyHead` | **`movement_gate`** | **`False`** (BC ckpt trained `True`) | **SILENT** under `strict=False`: 18 `gate_heads.*` tensors dropped |
| | `mtp_length` | 9 | **SILENT** (missing/unexpected heads) |
| | `num_abilities` / `hidden_dim` / `movement_dim` / `movement_bins` | 9/256/2/21 | LOUD (shape) |
| `RewardHead` | `mtp_length` | 9 | **SILENT** |
| | `num_buckets` / `hidden_dim` | 255/256 | LOUD (shape) |
| | `bucket_low` / `bucket_high` | -3.0/3.0 | **SAFE when omitted** — `bucket_centers` is a persistent buffer restored by the load, and every consumer (`train_agent_finetune:555`, `train_imagination:340`, `returns.py`) reads `head.bucket_centers`, never a literal |
| `ValueHead` | same as RewardHead | | same |
| `StateHead` | `num_targets` | 4, sourced from `len(STATE_TARGETS)` at import | LOUD, but **not recorded in any checkpoint** — growing `STATE_TARGETS` silently invalidates every existing Phase-2 checkpoint's state head |

---

## 2. Construction-site matrix

Legend: **OK** = passed correctly · `-safe` = omitted, default matches truth · **OMITTED-WRONG** =
omitted and default ≠ truth · **SRC** = read from the wrong dict · **HC** = hardcoded literal ·
**STALE** = literal that no longer matches the stack.

### Dynamics sites

| site | size | latent_dim | use_actions | use_qk_norm | soft_cap | reg | kv | k_max | agent | guard on load |
|---|---|---|---|---|---|---|---|---|---|---|
| `scripts/train_dynamics.py:1461` | args | args | args | args (`not no_qk_norm`) | args | args | args | args ✅ | args | strict resume ✅ |
| `scripts/train_agent_finetune.py:205` (`build_dynamics`) | args | **cfg-corrected** | cfg ✅ | args | args | args | args | -safe | HC True | `load_state_dict_guarded` ✅ |
| `scripts/agent_infer.py:111` | **SRC** args | **SRC** args | cfg ✅ | **SRC** args | **SRC** args | **SRC** args | **SRC** args | -safe | HC True | `_load_state_dict_guarded` ✅ |
| `scripts/eval_reward_head.py:44` | **SRC** args | cfg ✅ | cfg ✅ | **SRC** args | **SRC** args | **SRC** args | **SRC** args | -safe | `assert miss+unexp<=10` ⚠ |
| `scripts/rollout_check.py:69` | args | **HC 32** | CLI | **HC True** | **HC 50** | **HC 8** | **HC 4** | -safe | off | `assert miss+unexp<=10` ⚠ |
| `scripts/eval_dream_quality.py:39` | CLI | **HC 32** | CLI | **HC True** | **HC 50** | **HC 8** | **HC 4** | -safe | off | `assert miss+unexp<=10` ⚠ |
| **`scripts/eval_dynamics.py:298`** | args | args | detect | **OMITTED-WRONG / SRC** | args (dflt `0.0`!) | args (dflt `0`!) | args (dflt `0`!) | -safe | **OMITTED** | print-only ❌ |
| **`scripts/eval_trade_prediction.py:162`** | args | args | **OMITTED-WRONG** | **OMITTED-WRONG / SRC** | args (dflt `0.0`) | args (dflt `0`) | args (dflt `0`) | -safe | **OMITTED** | none ❌ |
| **`scripts/finetune_shortcut.py:107`** | CLI | CLI **STALE 48** | **OMITTED-WRONG** | -safe | CLI | CLI | CLI | **OMITTED** (≠ `--k-max`) | **OMITTED** | none ❌ |
| `scripts/train_imagination.py:126` | **CLI dflt `small`** | cfg ✅ | cfg ✅ | CLI | CLI | CLI | **CLI dflt `None`** | **OMITTED** (≠ `--k-max`) | HC True | `strict=False`, no check ❌ (masked by shape crash) |
| `scripts/preflight.py:64,246` | CLI | CLI **STALE 48** | **OMITTED** | **OMITTED** | CLI | CLI | CLI | **OMITTED** (≠ `--shortcut-k-max`) | **OMITTED** | n/a (fresh) |
| `scripts/probe_scaling.py:67` | CLI | CLI **STALE 48** | **OMITTED** | -safe | -safe | HC 8 | HC 4 | -safe | — | n/a (fresh) |
| `scripts/probe_shortcut_k4k64.py:105` | CLI | CLI **STALE 48** | **OMITTED** | -safe | -safe | HC 8 | HC 4 | **OMITTED** (≠ `--k-max`) | — | n/a (fresh) |
| `scripts/probe_dynamics.py:64` | CLI | CLI | **OMITTED** | -safe | -safe | HC 8 | CLI | -safe | — | n/a (fresh) |
| `scripts/vram_sweep.py:53` | HC medium | HC 32 | **HC False** | HC True | HC 50 | HC 8 | HC 4 | -safe | — | n/a (fresh) |
| `scripts/test_kv_cache.py:20` | HC tiny | HC 16 | — | -safe | -safe | HC 4 | HC 2 | -safe | — | n/a (unit test) |

`scripts/eval_dynamics_per_frame_psnr.py` and `scripts/eval_rollout_variants.py` both
`from eval_dynamics import load_dynamics` — **they inherit the `eval_dynamics.py:298` defect verbatim.**

### Tokenizer sites

| site | source of arch | verdict |
|---|---|---|
| `scripts/pretokenize_replay_v7.py:41` (`load_v7`) | **`ck["model_config"]`** ✅ + raises on non-rope missing/unexpected | **CLEAN — this is the canonical pattern.** Used by `pretokenize_yt_v7`, `bench_encode`, `vram_sweep`, `train_dynamics:1624`, `agent_infer`, `ops/tok_eval_watcher`, `probes/bar_*` |
| `scripts/eval_dream_quality.py:49`, `scripts/rollout_check.py:150` | `ck["model_config"]` ✅ | good source, but `strict=False` with **no missing/unexpected check** |
| `scripts/eval_tokenizer.py:22`, `rollout_v7.py:20`, `rollout_clip.py:22` | **HC full v7 arch** (`large`+`latent_dim 16`+`num_latents 512`+`8/8`+`temporal_every 4`) | measured **CLEAN**; strict load → loud if v7 is ever retrained differently |
| **`scripts/eval_dynamics.py:325`** | **SIZE PRESET** + `strict=False`, print-only | broken vs v7 |
| **`scripts/pretokenize_frames.py:150`** | **SIZE PRESET** + `strict=False`, **no check at all** | broken vs v7 — and this is a **corpus writer** |
| **`scripts/eval_trade_prediction.py:151`** | **SIZE PRESET** + `strict=False`, no check | the original 112-random-tensor bug, unfixed |
| `scripts/rollout_tokenizer.py:23` | **HC `medium`** | dead; strict load → loud |
| `scripts/eval_transformer_tokenizer.py:16` | **HC `small`** | dead; strict load → loud |
| `scripts/eval_dynamics_per_frame_psnr.py:55`, `eval_rollout_variants.py:58` | legacy module + `--tokenizer-size` override | legacy-only shim, documented |
| `scripts/train_transformer_tokenizer.py:847` | all 9 arch axes from CLI incl. `temporal_every` ✅ | correct |

### Head sites

| site | mtp | buckets | hidden | movement_bins | **movement_gate** | num_targets |
|---|---|---|---|---|---|---|
| `train_agent_finetune.py:664/666/670`, `826/830/842` | args | args | args | args | args ✅ | `len(STATE_TARGETS)` |
| `agent_infer.py:121/125` | args | args | args | args | args ✅ | n/a |
| `eval_reward_head.py:56` | args | args | args | — | n/a | n/a |
| **`train_imagination.py:163`, `:421`** | args | args | args | args | **OMITTED-WRONG** | n/a |
| `train_imagination.py:437/517` (ValueHead) | n/a | args | args | — | n/a | n/a |

---

## 3. Findings, ranked by (silent × affects live/training/corpus)

### 🔴 P0 — SILENT, and it is the eval path everyone reads numbers from

**F1. `scripts/eval_dynamics.py:302` — `args.get("use_qk_norm", False)`. Still live.**
The dynamics trainer writes `no_qk_norm` into `args` and never writes `use_qk_norm`, so the
key does not exist and the `False` default always wins.

*Measured against `desktop_resume_8775_stripped.pt`:*
- **36 trained `qk_norm.{q,k}_norm.weight` tensors dropped** (unexpected keys),
- attention `scale` flips **1.0 → 0.125** in all 18 blocks,
- **0 missing, 0 shape mismatches** → nothing raises. Only two `print("Warning: ...")` lines.

The correct read is `use_qk_norm = not args.get("no_qk_norm", False)`, or better
`checkpoint["model_config"]["use_qk_norm"]`. Inherited verbatim by
`eval_dynamics_per_frame_psnr.py` and `eval_rollout_variants.py`.

**F2. Same site — three defaults that are wrong for *any* checkpoint whose `args` predate the flag.**
- `soft_cap=args.get("soft_cap", 0.0)` → `soft_cap_attention` computes `0.0 * tanh(x/0.0)`,
  i.e. **every attention logit becomes 0 (uniform attention) or NaN**. Purely numeric: **no
  tensor is missing, nothing raises, ever.** This is the most undetectable failure in the repo.
- `num_register_tokens=args.get("num_register_tokens", 0)` → drops `register_tokens` (silent
  unexpected key) and shortens the sequence 266 → 258.
- `num_kv_heads=args.get("num_kv_heads", 0)` → `0` is falsy, `num_kv_heads or num_heads`
  silently yields MHA. Measured: 36 shape mismatches (`k_proj` 768×768 vs 256×768) → this one
  *is* loud, by accident.

**F3. `scripts/eval_trade_prediction.py:162` — `use_actions` never passed, plus F1.**
*Measured:* **48 trained tensors dropped** (36 `qk_norm` + 12 `action_embed.*`/`no_action_embed`),
sequence length 265 vs the trained 266, scale 0.125. Zero shape mismatches → **runs to
completion and prints plausible trade predictions.** Exactly the historical failure mode.

**F4. `scripts/eval_trade_prediction.py:151` / `pretokenize_frames.py:150` / `eval_dynamics.py:325`
— tokenizer rebuilt from a SIZE PRESET, `model_config` ignored.**
Against the v7 checkpoint this now fails loudly (measured: 14 shape mismatches incl.
`bottleneck.proj.weight` 64×1024 vs 16×1024, 96 non-rope missing, 8 dropped) — but *only*
because v7's arch drifted off every preset. For any checkpoint whose dims happen to match a
preset, the omitted **`temporal_every`** (factory default 2, v7 trained 4) is the residual trap.
`pretokenize_frames.py` is the worst of the three because it is a **corpus writer** with
`strict=False` and **no missing/unexpected check whatsoever** — it would write a whole latent
corpus from a partly-random tokenizer. `slurm/slurm_pretokenize.sbatch` still invokes it.

### 🟠 P1 — SILENT, affects training/fine-tuning

**F5. `scripts/finetune_shortcut.py:107` — `use_actions` never passed.**
*Measured:* **12 action tensors dropped**, sequence 265 vs 266, no shape mismatch, `strict=False`
with no check → silent. This script fine-tunes the *shipped, action-conditioned* `gs8775`
backbone (README names it as the path to K=4 sampling). Also: `--latent-dim` default is **48**
(stale; the stack is 32) — that one crashes loudly, which is the only thing currently masking F5.
Also: `--k-max` is passed to `ShortcutForcing` but **not** to `create_dynamics`, so any
`--k-max != 64` silently mismaps τ indices and step embeddings — the exact hazard the
`k_max` docstring in `dynamics.py` warns about.

**F6. `scripts/train_imagination.py:163,421` — `PolicyHead` built without `movement_gate`.**
*Measured:* **18 `gate_heads.*` tensors** in the BC checkpoint have nowhere to go. Line 166 uses
a **strict** load, so this raises — Phase 3 is currently *broken but loud* against every gated
BC checkpoint. `PolicyHead.log_prob()` also hard-raises for gated heads. Fix is to read
`movement_gate` from `ckpt["args"]` the way `agent_infer.py:124` does.

**F7. `train_imagination.py:126` reads only `latent_dim` + `use_actions` from `dynamics_config`;
`model_size` and `num_kv_heads` come from CLI defaults (`small`, `None`).**
*Measured against the BC checkpoint:* 193 shape mismatches, 83 dropped → loud. Safe today only
because the size mismatch is a shape mismatch. `dynamics.load_state_dict(dyn_state, strict=False)`
at :152 has **no missing/unexpected check**, so any future arch difference that is
key-shaped rather than shape-shaped goes silent.

### 🟡 P2 — silent measurement corruption / stale launchers

**F8. `scripts/preflight.py:64,246` measures the wrong model.** `--latent-dim` default **48**
(stale), and `use_actions`, `use_agent_tokens`, `use_qk_norm`, and `k_max` are never passed.
`--shortcut-k-max` *is* parsed and handed to `ShortcutForcing` but not to `create_dynamics`.
Nothing crashes — it just reports VRAM/throughput for a model that is not the one being trained.

**F9. `slurm/slurm_agent_finetune.sbatch:36-37` — `--model-size small --latent-dim 48`.**
This is the stale launcher the brief predicted. The current stack is **medium / 32**.
`train_agent_finetune` auto-corrects `latent_dim` from `model_config` (:271) but **not**
`model_size`, so this crashes loudly (512 vs 768) rather than mis-training. It also omits
`--movement-gate`, `--movement-bins`, `--num-buckets`, `--hidden-dim`, `--agent-layers`.
The working launchers are `scripts/launch_bc_1060.sh` and `scratchpad/launch_bc_gate_1060.sh`
(`--model-size medium --num-kv-heads 4 --num-register-tokens 8 --soft-cap 50.0 --movement-gate`).

**F10. Stale `--latent-dim 48` defaults in `probe_scaling.py:49`, `probe_shortcut_k4k64.py:87`,
`finetune_shortcut.py:90`, `preflight.py:366`.** These are fresh-init training probes; a 48-dim
`input_proj` against 32-channel latents raises, so they are loud — but four copies of a stale
default is a standing trap.

**F11. `slurm/slurm_dyn_{s1prime,auxloss,yt578,anneal_probe}.sbatch` omit `--use-actions`;
only `slurm_dyn_train.sbatch` passes it.** Correct for latent-only/unlabeled runs, but there is
nothing tying the flag to the checkpoint being resumed. `train_dynamics` resumes with
`strict=True` by default (`--loose-resume` to opt out), which is the guard that saves this.

**F12. `create_transformer_tokenizer` cannot express a 256-px tokenizer** (`img_size` hardcoded
352), and its `resolved` dict silently drops `temporal_every`, `patch_size`, `dropout`,
`use_sincos_pos`, and `max_time`.

**F13. `dynamics_config`/`model_config` for the dynamics model does not record `temporal_every`,
`max_seq_len`, `head_dim`, or `dropout`.** Dynamics checkpoints are not fully self-describing.

**F14. The tokenizer→dynamics latent fold `view(B, 16, 16, -1).permute(0,3,1,2)` is
copy-pasted, unasserted, across 8 files** (`pretokenize_replay_v7.py:96`,
`pretokenize_frames.py:173`, `agent_infer.py:64`, `eval_trade_prediction.py:232`,
`eval_dynamics.py:349`, `probes/timing_audit.py:63`, and as the literal `reshape(1, 512, 16)`
in `overlay_e2e.py:27` and `test_agent_infer.py:38`). The literal `(1, 512, 16)` form is the
dangerous one: a tokenizer with `num_latents=256, latent_dim=32` has the *same element count*
(8192), so the reshape succeeds and silently produces a differently-ordered token grid.
`pretokenize_replay_v7.py:114` is the only place that derives it (`num_latents*latent_dim//256`).

### ⚪ Correct-by-construction (keep as templates)

- `scripts/pretokenize_replay_v7.py:38` `load_v7` — builds from `model_config`, raises on any
  non-rope missing/unexpected. Measured **CLEAN**.
- `scripts/agent_infer.py` `_load_state_dict_guarded` and
  `scripts/train_agent_finetune.py:176` `load_state_dict_guarded` — `strict=False` that
  *raises* on anything unexplained, with an explicit `allow_missing` allowlist. Measured
  **CLEAN** against the BC checkpoint. These two are the right shape for a shared loader.
- `scripts/train_dynamics.py:1461` — the only site that threads `k_max` to *both*
  `create_dynamics` and `ShortcutForcing`.
- `scripts/eval_tokenizer.py:22`, `rollout_v7.py:20`, `rollout_clip.py:22` — hardcoded but
  *complete* v7 arch + strict load. Measured **CLEAN**.

---

## 4. Recommended canonical construction

Add **one** loader module (e.g. `src/ahriuwu/models/loading.py`) and route every non-trainer
site through it. Rules:

1. **Architecture comes from the checkpoint, never from `args`, never from a size preset.**
   ```
   dynamics : ckpt["model_config"]  (Phase-1)  or  ckpt["dynamics_config"]  (Phase-2)
   tokenizer: ckpt["model_config"]
   heads    : ckpt["args"]  ← the only legitimate use of args; see (5)
   ```
   Build with `DynamicsTransformer(**{k: v for k, v in cfg.items() if k != "size_preset"})`
   and `TransformerTokenizer(**cfg)` — i.e. the `load_v7` pattern, not the `create_*` factory.
   The factories are for *new* models from CLI flags; they must not be used to reconstruct a
   trained one.

2. **Fallback only when `model_config` is absent (pre-2026 checkpoints), and then explicitly:**
   `use_qk_norm = not args["no_qk_norm"]`, `use_actions = any("action_embed." in k)`,
   `use_agent_tokens = any("agent_token" in k)`, `num_kv_heads` inferred from
   `k_proj.shape[0] // head_dim`. Log loudly that inference was used.

3. **Every load goes through a guarded `strict=False`** — the
   `train_agent_finetune.load_state_dict_guarded` implementation, promoted into the module.
   Never a bare `strict=False`, never print-only warnings. Rope buffers may be allowlisted by
   name; nothing else may.

4. **Reject `soft_cap == 0.0` at construction.** `create_dynamics` should raise on a
   non-positive, non-`None` `soft_cap` rather than produce `0*tanh(x/0)`. Same for
   `num_kv_heads == 0` (currently silently means MHA).

5. **Record head config in the checkpoint.** Add a `head_config` dict (`hidden_dim`,
   `num_buckets`, `mtp_length`, `movement_bins`, `movement_gate`, `num_targets`,
   `bucket_low/high`) next to `dynamics_config`, so heads stop being reconstructed from CLI
   `args`. Today `StateHead.num_targets` is recoverable only from `len(STATE_TARGETS)` at
   import time.

6. **Close the two self-describing-checkpoint holes:** add `temporal_every`, `max_seq_len`,
   `head_dim`, `dropout` to `create_dynamics`'s `resolved` dict, and stop having the tokenizer
   factory drop `temporal_every`/`img_size` from its `resolved` dict.

7. **One `latent_grid_fold(latent, cfg)` helper** that asserts
   `cfg["num_latents"] * cfg["latent_dim"] == spatial_tokens * dynamics_latent_dim`, replacing
   the eight copy-pasted `view(B,16,16,-1)` / `reshape(1,512,16)` sites.

8. **Assert `model.k_max == shortcut.k_max`** wherever both exist
   (`finetune_shortcut.py`, `preflight.py`, `probe_shortcut_k4k64.py`).
