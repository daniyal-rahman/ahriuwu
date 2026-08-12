# Code Quality Review — 2026-08-06

Repo/file-level structural review of `ahriuwu` at HEAD `93c47f0`. Read-only pass: tree walked,
line counts measured, duplication grepped, git history sized. Not a line-by-line bug hunt.

**Context for calibration:** this is a solo-dev research repo with live training jobs
(Slurm `tok-v7-y` on `desktop`, `train_agent_finetune.py` + `scratchpad/tok_eval_watcher.py`
on this login box). The review optimizes for *not breaking those* and *preventing the next
shipped bug*, not for aesthetic purity.

---

## (a) Scorecard

| Area | Grade | One-line verdict |
|---|---|---|
| `src/ahriuwu` library core | **B+** | 28 files / 8.2k lines, well-commented, checkpoints are self-describing (`model_config` + `args` + `git_info`). The good bones are here. |
| Layout & layering | **C** | Library vs scripts split is real, but scripts/ is a flat 67-entry pile, scratchpad/ (119 entries, 69 tracked) contains production infra, and the repo root has 12 status .md + 12 png + a file literally named `"C:\Users\daniz\replay_pipeline.bat"`. |
| Duplication | **C-** | The dynamics-checkpoint load block exists in ~15 scripts (16 non-archive files carry `strict=False`), tokenizer `model_config` load in 6, PSNR in 17, tmp-symlink dataset setup in 6+, arch flags in 30 files. One consolidation module kills most of it. |
| Config sprawl | **C-** | `medium/latent32/kv4/reg8/softcap50` re-typed in 24 scripts/launchers + 6 sbatches. Checkpoints already carry the truth; consumers re-derive by hand. This lineage **already shipped the `use_actions` bug** (agent_infer.py:81–88). One stale `--latent-dim 48` sbatch still live in slurm/. |
| Dead/stale code | **C+** | `scripts/_archive` exists and is used (158 files — good discipline), but 8+ superseded eval/probe scripts and ~10 finished-run sbatches still sit in the active namespace with no marker of which is current. |
| Tests | **C** | 4 real pytest files (319 lines) + 2 scripts/test_*.py + trainer `--smoke-test` flags. What exists is well-chosen (DDP sampler, τ schedule, HUD mask, rollout equivalence) but the highest-risk contracts (checkpoint load-path, cache schema, fold/unfold, head interfaces) are untested. |
| Hygiene | **D+** | 85 MB OpenAI codex binary tracked in `node_modules/`; 38 MB of tracked pngs/media; `.git` = 342 MB (deleted 31 MB eval pngs still in history); `.gitignore` misses `checkpoints`/`wandb` symlinks, `*.log.prev`, `*.out`; tracked logs and status txt in scratchpad; CLAUDE.md both tracked and gitignored. |
| Docs | **B-** | README (2026-08-01) is genuinely accurate and measured — rare and valuable. But 10 dated status/briefing .md files litter the root, and docs/ has no index saying which of the 14 docs are current vs `audits_legacy`-grade. |

---

## (b) Top-10 findings, ranked by risk to the project

### 1. Checkpoint-consumption is hand-rolled in ~15 places — this class already shipped a bug
**Risk: HIGH (recurrence is near-certain).** The shipped instance: `agent_infer.py` read
`use_actions` from the *args* dict when it only exists in `dynamics_config` — the derived dict —
so `a.get("use_actions")` silently built an **action-less** backbone for an action-conditioned
checkpoint (documented in the fix comment at `scripts/agent_infer.py:81–88`, fixed in `33b9ad9`/`7fff912`).
The same read-the-wrong-dict / hardcode-the-shape pattern is live today:

- `create_dynamics(...)` + `load_state_dict(strict=False)` + rope-key filter, copy-pasted:
  `rollout_check.py`, `eval_dream_quality.py`, `eval_reward_head.py`, `eval_dynamics.py`,
  `eval_trade_prediction.py`, `agent_infer.py`, `train_agent_finetune.py`, `finetune_shortcut.py`,
  `probe_dynamics.py`, `test_kv_cache.py`, `vram_sweep.py`, + 4 scratchpad probes.
- `latent_dim=32` hardcoded at the call site in 8 files; `use_actions` derived 3 different ways
  (`cfg.get(...)`, `any("action_embed." in k ...)`, `not args.no_actions`) across
  `eval_reward_head.py:43`, `eval_dynamics.py:294`, `rollout_check.py:69`, `train_agent_finetune.py:228`.
- Tokenizer side: `{k: v for k, v in tk["model_config"].items() if k != "size_preset"}` + `strict=False`
  in 6 files (`rollout_check.py:149`, `eval_dream_quality.py:48`, `pretokenize_replay_v7.py`,
  `train_agent_finetune.py`, `train_transformer_tokenizer.py`, `scratchpad/probe_blur.py`).

The irony: `src/ahriuwu/utils/training.py:374` already saves a **self-describing** checkpoint
(`model_config` = factory-resolved constructor kwargs, exactly so "CLI args alone are not enough").
Nobody consumes it generically.

**Fix sketch — `src/ahriuwu/utils/loading.py` (~120 lines):**
```python
def load_dynamics(ckpt_path, device="cpu", overrides=None):
    """create_dynamics from ckpt['model_config'] (fallback: args + state-dict probe),
    strict=False load, rope-filter, assert unexpected==[]. Returns (model, ckpt)."""

def load_tokenizer(tok_path, device="cpu"):
    """TransformerTokenizer from ckpt['model_config'] minus size_preset; frozen+eval."""

def resolve_use_actions(ckpt) -> bool:
    """THE one place: dynamics_config['use_actions'] if present,
    else any('action_embed.' in k for k in state_dict). Never reads args."""
```
Migrate consumers opportunistically (each next time a script is touched); migrate
`agent_infer.py`, `rollout_check.py`, `eval_reward_head.py`, `train_agent_finetune.py` immediately
since they're the production path. **Effort: 0.5 day for the module + 4 priority call sites.**

### 2. Arch hyperparameters re-specified in 30 files; one stale dim-48 launcher is still armed
**Risk: HIGH.** `--num-kv-heads 4 --num-register-tokens 8 --soft-cap 50.0` (plus
`--model-size medium --latent-dim 32 --shortcut-k-max 64`) appears in 24 scripts/launchers + 6
sbatches. `scripts/dyn_train_args.sh` is titled "SINGLE SOURCE OF TRUTH" but its two forks
(`dyn_train_args_action.sh`, `dyn_train_args_hudfix.sh`) re-paste the entire resume-critical arch
block, and `slurm/slurm_dyn_train.sbatch`/`slurm_dyn_s1prime.sbatch` re-specify it independently
of all three. Meanwhile **`slurm/slurm_agent_finetune.sbatch` still says `--model-size small
--latent-dim 48`** — the stale pre-v7 shape (memory notes flag exactly this hazard); running it
against any current checkpoint produces a shape mismatch or a silent `strict=False` disaster.

**Fix sketch:** one `scripts/arch_v7.sh` exporting `ARCH_ARGS=(--model-size medium --latent-dim 32
--num-kv-heads 4 --num-register-tokens 8 --soft-cap 50.0 --shortcut-k-max 64)`; the three
`dyn_train_args*.sh` and every sbatch source it and append run-specific args. For *resume/eval*
paths, finding #1 makes the flags unnecessary entirely (read `model_config`). Archive or fix the
dim-48 sbatch **today** — it costs one `git mv`. **Effort: 2–3 h.**

### 3. Production infrastructure lives in scratchpad/
**Risk: HIGH (operational).** Right now, on live paths:
- `scratchpad/tok_eval_watcher.py` — **running as a process on this machine now**.
- `scratchpad/bc5080_watchdog.sh` — supervises the act8775 BC run on the desktop 5080.
- `scratchpad/encode_heldout_slice.py`, `scratchpad/eval_queue_0731.sh`, `scratchpad/build_hud_mask.py`
  (produced `hud_valid_mask_352.pt`, a training input of the hudfix recipe), `scratchpad/run_e2e.sh`.

A directory named "scratchpad" is, by social contract, deletable. It is not — and worse, 69 of its
119 entries are now git-tracked, including logs, so the delete-vs-keep signal is gone entirely.
Note `scripts/bc_watchdog.sh` (the 1060 twin of `bc5080_watchdog.sh`, 43 lines each, near-identical)
was already promoted — the pattern is half-applied.

**Fix sketch:** promote the 6 files above into `scripts/infra/` and `scripts/eval/` (watchdogs can
take LOG/STATUS/CKPT/STOP as env vars and collapse to ONE parameterized `bc_watchdog.sh`). Do the
`git mv` **without touching the running copies' paths** — i.e., move at the next natural restart of
each job, or leave a symlink `scratchpad/tok_eval_watcher.py -> ../scripts/infra/tok_eval_watcher.py`
so live invocations keep working. Then un-track everything else in scratchpad
(`git rm -r --cached scratchpad/` + gitignore, re-add only the promoted files at their new home).
**Effort: 2 h (+ symlink care).**

### 4. 85 MB OpenAI codex binary and 38 MB of pngs are tracked; .git is 342 MB
**Risk: MEDIUM (bleeds time on every clone/pull to Vast boxes; will only grow).**
- `node_modules/@openai/codex-darwin-arm64/.../codex` — an 81 MB **macOS arm64** binary, in a Linux
  repo, tracked. `node_modules/` isn't in `.gitignore` at all.
- 38 MB of tracked pngs/media: 12 root-level `dream_*.png`/`s1p_*.png` (~20 MB), scratchpad stills,
  duplicated 3.5 MB `hp_recon_montage.png` (both `docs/assets/` and `scratchpad/hp_recon_stills/`).
- History carries deleted `eval_results/ocr_analysis/*.png` blobs of 31 MB, 28 MB, 11 MB.

**Fix sketch:** `git rm -r --cached node_modules && echo node_modules/ >> .gitignore` (keep
`package.json` if the CLI is wanted per-machine). `git rm --cached` the root pngs — the ones that
matter are already curated into `docs/assets/` (README references those, not the root copies).
History rewrite (`git filter-repo --strip-blobs-bigger-than 5M`) is optional; do it only at a quiet
moment since every clone (desktop, Vast snapshots) must re-clone — **defer until between runs**.
**Effort: 30 min now; 1–2 h for the history rewrite later.**

### 5. No test on the exact contracts that break silently
**Risk: MEDIUM-HIGH.** The existing 4 pytest files are good (they encode *lessons*: DDP sampler
equality, τ-schedule paper-faithfulness, HUD-mask loss, KV-cache rollout equivalence). The gaps are
precisely the interfaces where past bugs lived:

1. **Load-path contract** (the #1 class): build a tiny `create_dynamics("tiny"…)`, save via
   `save_checkpoint`, reload via `utils/loading.load_dynamics`, assert config round-trip,
   `unexpected == []`, and `resolve_use_actions` correct for both action/actionless states. ~60 lines.
2. **Fold/unfold**: `agent_infer._dyn_from_tok` vs `pretokenize_replay_v7`'s
   `view(B,16,16,-1).permute(0,3,1,2)` — currently duplicated by *comment agreement only*
   ("folded EXACTLY as pretokenize_replay_v7"). Move the fold into `src/ahriuwu` (e.g.
   `models/__init__.py` or `utils/latents.py`), import it in both, one 10-line test. A layout drift
   here poisons every downstream number *silently* — PSNR still computes, dreams still render.
3. **Dataset cache schema**: `replay_dataset._cache_meta` uses a hand-bumped `"schema": 2` with a
   comment "bump it on any md-dict change" — a test that hashes the md-dict keys of a synthetic
   parse and compares against a golden list turns "forgot to bump" into a red test instead of a
   stale-cache training run. ~30 lines.
4. **Head interfaces**: PolicyHead with/without `movement_gate` (output shapes, `gate_logits`
   raising without the flag), StateHead/RewardHead/ValueHead forward shapes at (B,T,·). ~50 lines.

That's it. Four files, ~200 lines, no test theater; everything else (visual dream quality, probe R²)
is legitimately eval-scripts-and-eyeballs territory and should stay that way.
**Effort: 0.5–1 day** (halved if #1's loading module lands first).

### 6. scripts/ is a flat pile of 67 entries with no current/superseded signal
**Risk: MEDIUM.** 51 .py + 16 .sh at top level, plus 6 subdirs of mixed vintage (`aggregation/`
`debug/` — now only a `__pycache__` — `keysender/`, `prepare_data/`, `rofl_decode/`, `_archive/`).
Four generations of dynamics eval coexist with no marker: `eval_dynamics.py` (910 lines, oldest) →
`eval_dynamics_per_frame_psnr.py` (wraps it; its own docstring defaults to "legacy
`latent_dim=48`"!) → `rollout_check.py` + `eval_rollout_variants.py` (current, actively churning)
→ `eval_dream_quality.py` (distributional; README notes automated metrics were caught rating a
poisoned model above the clean one — trust accordingly). Same story for probes:
`probe_latents.py` → `probe_latents_xgame.py` (cross-game, the honest protocol) and
`probe_hp_fulldim.py` → `probe_hp_mlp.py` (the definitive HP answer, per README's probe_r2 chart).

**Fix sketch:** subdirectories by *lifecycle role*, not topic:
```
scripts/
  train/     train_dynamics.py train_transformer_tokenizer.py train_agent_finetune.py
             train_imagination.py finetune_shortcut.py + *_args.sh + run_ddp_*.sh
  eval/      rollout_check.py eval_dream_quality.py eval_rollout_variants.py eval_bc_sim.py
             eval_reward_head.py overlay_e2e.py probe_latents_xgame.py probe_hp_mlp.py probe_casting.py
  data/      pretokenize_*.py pack_latents.py stage_*.sh yt_pipeline.py download_* gen_dataset_manifest.py
  infra/     watchdogs, launch_*.sh, preflight.py, boot_windows_vanguard.sh, keysender/
  live/      agent_infer.py play_live.py test_agent_infer.py
  _archive/  (add: eval_dynamics.py, eval_dynamics_per_frame_psnr.py, probe_latents.py,
              probe_hp_fulldim.py, rollout_v7.py-era tools, rofl_decode/ correlate_v1/v2)
```
Do it **between runs** (sbatches and watchdogs reference `scripts/...` paths) and grep
slurm/ + scratchpad/ + docs for every moved path in the same commit. If that's too disruptive now,
the 80/20 version is: archive the superseded 8 files + add a 20-line `scripts/README.md` mapping
"question → current tool". **Effort: 0.5 day full; 1 h for the 80/20.**

### 7. slurm/ carries ~10 sbatches for finished runs, one of them wrong
**Risk: MEDIUM.** 20 sbatches; `slurm_tok_train.sbatch` v1–v4 + `slurm_v7_trial` are tokenizer
archaeology; `slurm_dyn_yt578`, `slurm_dyn_s1prime`, `slurm_dyn_anneal_probe`, `slurm_dyn_auxloss`
are one-shot experiments that concluded; `slurm_agent_finetune.sbatch` is the dim-48 landmine
(finding #2). The live ones are `slurm_tok_train_v7_yt.sbatch` (job 208 running NOW),
`slurm_tok_train_v7[_cont]`, `slurm_dyn_train`, `slurm_pretokenize_replay_v7`, `slurm_pack_latents`.
**Fix sketch:** `slurm/_archive/` + move the 10, with a one-line "superseded by X / run concluded
YYYY-MM-DD" header appended. Keeps `sbatch slurm/<tab>` honest. **Effort: 30 min.**

### 8. .gitignore doesn't match the repo's actual shape
**Risk: LOW-MEDIUM (it's how the binary/log tracking happened).** Specific holes:
- `checkpoints/` and `wandb/` (trailing slash) don't match the **symlinks** at repo root — hence
  the perpetual `?? checkpoints` / `?? wandb` status noise. Use `/checkpoints` and `/wandb`.
- `*.log` is ignored but `*.log.prev`, `*.out`, and status `.txt` aren't →
  `scratchpad/bc_1060.log.prev`, `vast_*_driver.out`, `bc_status.txt`, `bc5080_status.txt` are tracked.
- No `node_modules/` (finding #4). No `/*.png` guard for the root.
- `CLAUDE.md` and `.claude/` are listed in .gitignore **and tracked** — gitignore is a no-op for
  already-tracked files. Decide: either un-track them or (better, since CLAUDE.md is genuinely
  project config) remove them from .gitignore so the file's intent is coherent.
- The tracked file `"C:\Users\daniz\replay_pipeline.bat"` (backslashes-in-filename, created by a
  Windows-path mishap) should be `git mv`'d into `scripts/aggregation/` next to its siblings.
**Effort: 20 min.**

### 9. Run/lineage naming has no scheme — bc_1060 vs bc_5080_act8775 vs phase2_bc_garen vs phase2_bc_gate1060
**Risk: LOW-MEDIUM (it's how you grab the wrong checkpoint at 2 a.m.).** The same lineage is named
by GPU (`bc_1060.log`), by GPU+base-ckpt (`bc_5080_act8775.log`), by phase+champion
(`data/phase2_bc_garen`), and by phase+feature+GPU (`data/phase2_bc_gate1060`) — four axes, no
order. The hardware axis is the least meaningful one (the 5080 run is defined by *act8775-backbone*,
not by the card).
**Fix sketch:** adopt `p2bc_<backbone>_<variant>` (e.g. `p2bc_act8775_base`, `p2bc_old135_gate`)
for *new* run dirs/logs/wandb names, recorded in a 10-line `docs/RUNS.md` ledger (name → data → base
ckpt → where it lives). Don't rename existing dirs — live jobs write into them. **Effort: 15 min +
discipline.**

### 10. Root-level markdown sprawl hides the good docs
**Risk: LOW.** 12 .md at root (`DYNAMICS_REVIEW`, `DYNAMICS_VS_PAPER`, `WM_DEBUG_LOG_2026-07-14`,
`E2E_STATUS_AND_PLAN_2026-07-22`, `GAPS_TO_GOOD_2026-07-22`, `GROUNDING_2026-07-29`, …) — all
dated working notes, several superseded by the (excellent, measured) README. docs/ meanwhile mixes
current (`VAST.md`, `DATASETS.md`, `PREEMPTION.md`) with OCR-era (`click_detection_test_guide.md`,
`QUICK_START_CLICK_TEST.md`, `replay_movement_extraction.md`) with no index.
**Fix sketch:** root keeps `README.md`, `CLAUDE.md`, `INFERENCE_RUNBOOK.md`, `pyproject.toml`;
everything dated moves to `docs/notes/` (they're history — the verify/ report and audits_legacy
pattern shows you already know how to do this); add a 15-line index to `docs/` separating
"current ops" from "historical". Also: `pyproject.toml` still depends on `easyocr` (OCR era) —
drop it. `src/ahriuwu/vision/hp_bars.py` + `data/hud_regions.py` have **zero** non-`__init__`
importers (OCR-era remnants — archive/delete); `data/lane_opponent.py` is **live**
(replay_dataset + rewards import it) and `live/client_api.py` is the planned port-2999 path
(keep both). **Effort: 45 min.**

---

## (c) Cleanup plan

### Do this weekend (< 1 day total, safe with jobs running)
| # | Action | Time |
|---|---|---|
| 1 | Archive `slurm/slurm_agent_finetune.sbatch` (dim-48) + the 9 finished-run sbatches to `slurm/_archive/` with a header note | 30 min |
| 2 | `.gitignore` fixes: `node_modules/`, `/checkpoints`, `/wandb`, `*.log.prev`, `*.out`, `/*.png`; `git rm -r --cached node_modules` (−85 MB tracked) and the 12 root pngs + scratchpad logs/status/`.log.prev` | 45 min |
| 3 | Write `src/ahriuwu/utils/loading.py` (`load_dynamics` / `load_tokenizer` / `resolve_use_actions`) + the load-path contract test; port `rollout_check.py` and `eval_reward_head.py` as proof | 3–4 h |
| 4 | Extract `ARCH_ARGS` into `scripts/arch_v7.sh`, source from the three `dyn_train_args*.sh` (do **not** touch a launcher mid-run — this only affects the next launch) | 1 h |
| 5 | Move the fold helper (`_dyn_from_tok`) into `src/ahriuwu` + 10-line equivalence test against the pretokenize fold | 45 min |
| 6 | `git mv "C:\Users\daniz\replay_pipeline.bat"` to `scripts/aggregation/replay_pipeline.bat`; delete empty `scripts/debug/` | 5 min |
| 7 | `scripts/README.md`: 20-line "question → current tool" map (which eval/probe supersedes which) | 30 min |

### Do eventually (between runs / next quiet week)
- Promote scratchpad production tools to `scripts/infra/` + `scripts/eval/` with symlinks left behind for live invocations; un-track the rest of scratchpad (finding #3).
- Full `scripts/` subdirectory reorg (train/eval/data/infra/live) + fix every referencing path in slurm/, docs/, watchdogs in one commit (finding #6).
- Port the remaining ~11 checkpoint-load call sites to `utils/loading.py`; delete the per-script copies.
- Tests 3–4 from finding #5 (cache-schema golden test, head-interface shapes).
- Root .md → `docs/notes/`; docs index; drop `easyocr` from pyproject; archive `vision/hp_bars.py`, `data/hud_regions.py`, OCR-era docs.
- `git filter-repo --strip-blobs-bigger-than 5M` + re-clone everywhere (342 MB → ~50 MB); coordinate with desktop + Vast snapshot.
- Consolidate `bc_watchdog.sh` / `bc5080_watchdog.sh` into one parameterized watchdog; adopt the run-naming scheme + `docs/RUNS.md`.
- Consolidate PSNR into `src/ahriuwu/utils/metrics.py` and the tmp-symlink single-match dataset setup into a `single_match_dataset(latents_dir, match, labels_root, seq_len)` helper (6+ call sites each) — do it lazily, as each script gets touched.

### Sketch: the consolidation module APIs
```python
# src/ahriuwu/utils/loading.py
load_dynamics(ckpt_path, device="cpu", overrides: dict | None = None) -> tuple[Dynamics, dict]
load_tokenizer(tok_path, device="cpu", frozen=True) -> TransformerTokenizer
resolve_use_actions(ckpt: dict) -> bool          # dynamics_config first, state-dict probe second
load_phase2(ckpt_path, device) -> tuple[Dynamics, PolicyHead, RewardHead | None, dict]

# src/ahriuwu/utils/latents.py
fold_tok_to_dyn(latent: Tensor) -> Tensor        # (B,512,16) -> (B,32,16,16), THE layout
unfold_dyn_to_tok(grid: Tensor) -> Tensor

# src/ahriuwu/utils/metrics.py
psnr(a, b, max_val=1.0) -> float

# src/ahriuwu/data/__init__.py
single_match_dataset(latents_dir, match, labels_root, seq_len, stride=None) -> ReplayLatentSequenceDataset
```

---

## (d) DO-NOT-BREAK list

Live jobs and their dependencies as of this review (verified via `ps`/`squeue`):

| Path | Why |
|---|---|
| `scratchpad/tok_eval_watcher.py` | Running as a process on the login box **right now**. Do not move/rename without leaving a symlink; do not delete its log. |
| `scratchpad/bc5080_watchdog.sh`, `scratchpad/bc5080_status.txt`, `scratchpad/bc_5080_act8775.log`, `scratchpad/bc5080.stop` sentinel path | Supervises the act8775 BC run on the desktop 5080 (note: it cd's to `/mnt/nfs/projects/ahriuwu` — desktop mount of this same tree; edits here ARE edits there). |
| `scripts/train_agent_finetune.py` + `src/ahriuwu/**` | A `train_agent_finetune.py --dynamics-checkpoint …` process is running on this machine and imports the library live off NFS. No refactors of imported modules while it runs. |
| `slurm/slurm_tok_train_v7_yt.sbatch`, `scripts/train_transformer_tokenizer.py`, `scripts/v7_train_args.sh` | Slurm job 208 `tok-v7-y` RUNNING on `desktop`; Slurm autoresume re-reads the sbatch on requeue. Also: these files (+ `losses.py`, `run_ddp_dyn.sh`, `rollout_login.sh`) carry uncommitted changes from a parallel session — hands off the working tree. |
| `data/tokenizer_v7_yt/`, `data/phase2_bc_garen*/`, `data/phase2_bc_gate1060/` | Active checkpoint dirs written by the jobs above (`data/` is untracked; nothing here should be "cleaned"). |
| `checkpoints`, `wandb` symlinks at root | Targets on `/mnt/storage`; ignore-rule fixes must use `/checkpoints` (no trailing slash), never delete the links. |
| Dataset cache schema (`replay_dataset._cache_meta`, `"schema": 2`) + `dataset_cache.pt` files in run dirs | Any change to `_parse_match` output requires the schema bump; deleting a live run's `dataset_cache.pt` forces an expensive rebuild mid-run. |
| `scratchpad/hud_valid_mask_352.pt`, `scripts/dyn_train_args_hudfix.sh` `HUD_MASK` contract | Training input of the hudfix recipe — the mask file is referenced by env var at launch. |
| `scratchpad/bc.stop` semantics | `touch scratchpad/bc.stop` is the *clean-stop* switch for the 1060 watchdog — never create it as a side effect of scripted cleanup. |

---

*Method note: counts in this review are measured, not estimated — 51 scripts/*.py (12,840 lines),
16 scripts/*.sh, 28 src files (8,217 lines), 20 sbatches, 119 scratchpad entries (69 tracked),
158 archived scripts, 449 tracked files (134 MB), .git 342 MB. Duplication counts from grep over
non-archive paths.*

---
*Correction (same day): `src/ahriuwu/vision/hp_bars.py` is NOT dead OCR-era code — it is the new
CV HP-bar reader created 2026-08-06 (consumed by `scripts/validate_hp_reader.py`, under active
refinement for the perception side-channel). `data/hud_regions.py` + the `easyocr` dependency
remain valid dead-code findings.*
