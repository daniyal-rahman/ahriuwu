# Vibe-Code Audit — ahriuwu — 2026-08-12

Audit of a ~15k-line, largely LLM-written ML repo (frozen v7 tokenizer → dynamics
transformer → BC policy heads → live inference), hunting **silent correctness bugs**:
code that runs, emits plausible numbers, and is wrong.

Read-only audit. Nothing under `src/` or `scripts/` was modified. Two reproducible
probe scripts were added under `scratchpad/`.

---

## Part A — Phase 1: what the field actually does

### A.1 The core empirical claim: AI code fails by *duplication and plausibility*, not by crashing

GitClear's longitudinal analysis of **211M changed lines (2020–2024)** across Google,
Microsoft, Meta and enterprise repos is the most-cited quantitative evidence:

- code cloning rose **8.3% → 12.3%** (2021→2024), ~4x growth in duplicate blocks
- refactoring ("moved" lines) fell from **25% → under 10%** of changed lines
- **copy/paste exceeded moved code for the first time in recorded history**

Sources: [GitClear 2025 AI code quality research](https://www.gitclear.com/ai_assistant_code_quality_2025_research),
[GitClear "The Maintainability Gap" 2026](https://www.gitclear.com/the_ai_code_quality_maintainability_gap),
[DevClass summary](https://www.devclass.com/ai-ml/2025/02/20/ai-is-eroding-code-quality-states-new-in-depth-report/).

**The operative consequence for an auditor:** the highest-yield search is not "find bad
code", it is **"find the same logic in N places and diff the copies"**. Divergence
between clones is where silent bugs live, because one copy gets fixed and the others
don't.

### A.2 Why review doesn't catch it: the perception gap

METR's RCT (16 experienced OSS devs, 246 real issues, randomized AI-allowed vs not)
found devs were **19% slower** with AI tools while believing they had been **20% faster**
— a 39-point gap between perceived and measured effect.
[METR study](https://metr.org/blog/2025-07-10-early-2025-ai-experienced-os-dev-study/)

This is the evidence base for the central audit stance: **do not trust that reviewed
code was reviewed.** Prefer techniques that produce a machine-checkable artifact
(a diff, a coverage report, a numerical divergence) over re-reading.

Corroborating: an industry figure widely repeated is that **~85% of developers ship AI
code on intuition rather than review** ([usekyros](https://usekyros.ai/blog/vibe-coding-crisis-ai-technical-debt))
— treat this one as *folklore*, it is a vendor blog with no published methodology.

### A.3 Taxonomy of LLM bug patterns (evidence-backed)

Tambon et al., *Bugs in Large Language Models Generated Code: An Empirical Study*
(333 bugs across three LLMs, validated by a 34-practitioner survey) —
[arXiv:2403.08937](https://arxiv.org/abs/2403.08937). Ten patterns:

Misinterpretation · Syntax Error · Silly Mistake · Prompt-biased code · Missing Corner
Case · Wrong Input Type · **Hallucinated Object** · **Wrong Attribute** · Incomplete
Generation · Non-Prompted Consideration.

The two that matter most for a silent-bug hunt are **Wrong Attribute** and
**Hallucinated Object** — reading a key/field that doesn't exist. In Python, with
`dict.get(k, default)` and `getattr(o, k, default)`, a wrong attribute does not raise;
it returns the default. *(This audit's #1 finding is exactly this pattern.)*

Also relevant:
- [arXiv:2607.20852](https://arxiv.org/abs/2607.20852) — of LLM solutions passing public
  tests, **23,081 of 43,677 failed hidden tests**. Passing the tests you have is weak
  evidence of correctness.
- [arXiv:2607.02333](https://arxiv.org/abs/2607.02333) — failures stem from
  "underspecified requirements and **subtle semantic deviations**".
- [arXiv:2508.00700](https://arxiv.org/abs/2508.00700) — LLM code had *fewer* bugs
  overall but introduced **structural issues in complex scenarios** absent from
  human-written code. Matches what we found: the local code is fine, the *seams* are wrong.

### A.4 Silent-failure antipatterns specifically attributed to AI agents

[urgentry: agent-introduced bugs / swallowed exceptions](https://urgentry.com/guides/ai-agents/agent-introduced-bugs-swallowed-exceptions/)
names four shapes: empty except block · **bare exception with plausible default return**
· silent retry exhaustion · 200-OK-with-error-body. Detection: ruff `BLE001`
(bare/broad except), ruff `S110` (`except: pass`).

The ML analogue of "returns 0 on a failed balance lookup" is "returns 0.0 for a failed
loss term" or "leaves a tensor randomly initialized" — same shape, but the wrongness is
laundered into a metric instead of a crash.

### A.5 The ML-specific literature: this repo's failure modes were named in 2015

Sculley et al., *Hidden Technical Debt in Machine Learning Systems*, NeurIPS 2015 —
[paper](https://proceedings.neurips.cc/paper_files/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf).
Categories that landed directly on findings below:

- **Configuration debt** — "ML systems often lack principled approaches to managing
  hyperparameters and configuration settings"; without versioning it is hard to know
  which settings correspond to a deployed model. → findings **S1, S3, S4**
- **Dead experimental codepaths** — abandoned-experiment code left in place; devs must
  work out which paths are live. → findings **S9, S10**
- **Glue code** (5–10x the ML code, obscures the algorithm) and **pipeline jungles**
  → findings **S3, S5**
- **Undeclared consumers** — downstream users break silently. → **S3**
- **Entanglement / CACE** ("Changing Anything Changes Everything").

Kapoor & Narayanan, *Leakage and the Reproducibility Crisis in ML-based Science* —
[arXiv:2207.07048](https://arxiv.org/abs/2207.07048) — catalogue **8 leakage types
affecting 329 papers across 17 fields**, and note the critical property:
**"none of these errors could have been caught by reading the papers."** Leakage is
invisible at the reporting layer; you must inspect the split mechanics. → finding **S4**

Karpathy, *A Recipe for Training Neural Networks* —
[karpathy.github.io](https://karpathy.github.io/2019/04/25/recipe/) — the standard
practitioner checklist. Techniques mapping to silent bugs: *verify loss @ init*
(wrong init), *overfit one batch* (architecture/wiring bug), *input-independent baseline*
(model ignoring its input), *visualize just before the net* (preprocessing/shape bugs),
*use backprop to chart dependencies* (information leaking across the batch/time axis),
*fix random seed*. Folklore-but-universally-adopted; not an RCT, but battle-tested.

Training/serving skew: the standard framing is that skew arises from **duplicate logic
implemented twice** (training path vs inference path), and the standard fix is a single
shared implementation. → findings **S3, S6**

### A.6 The practitioner audit playbooks

Best-structured of the blog-tier sources: [Vibe Coding Tech Debt: Audit & Refactor
AI Code](https://www.kunalganglani.com/blog/vibe-coding-tech-debt-audit) — 5 debt types
(hallucinated APIs · **copy-paste duplication from context-window patterns: "LLMs lack
project-wide memory, generating 4+ subtly different implementations of the same logic"**
· missing edge cases · confident-looking security holes · undocumented architecture) and
an 8-step checklist: locate AI hotspots in git history → `vulture`/`ts-prune` dead code →
`jscpd`/PMD-CPD duplicate detection at 50-token threshold → grep empty catch blocks →
verify every external API call against docs → Semgrep for LLM antipatterns → map
decisions to ADRs → mutation-test the tests.

From [Ask HN: How do you audit LLM code in languages you don't know?](https://news.ycombinator.com/item?id=46992895)
— practitioner consensus is **behavioral validation over line-by-line reading**:
max-warnings compilation, adversarial second-LLM review, black-box functional testing,
sandboxing/blast-radius limits, and static analysis for **dead & unreachable code
(catalogued as CWE-561)** which functional tests structurally cannot catch.

One notable observation from [aiinsightsnews](https://aiinsightsnews.net/vibe-coding-technical-debt/):
generated-code audits differ from normal review because **"comments describe intention
rather than actual behavior"** — which promotes *comment-vs-code diffing* from a style
check to a correctness technique. → finding **S11**

Tooling aimed at LLM output: [Semgrep Guardian](https://semgrep.dev/products/product-updates/detect-risks-in-ai-generated-code-with-semgrep-guardian/),
[semgrep-rules](https://github.com/semgrep/semgrep-rules),
[fettle](https://github.com/MilindGaharwar/fettle) (intercepts Claude Code mutations,
runs ruff + semgrep + "incident-derived LLM-antipattern rules"),
[CodeDrift](https://www.npmjs.com/package/codedrift),
`vulture` + [`vulturecov`](https://github.com/) (coverage-informed dead-code detection).

### A.7 Evidence grading

| Technique | Status |
|---|---|
| Clone-and-diff (duplication is *the* AI failure mode) | **Evidence-backed** (GitClear, 211M lines) |
| Don't trust that review happened | **Evidence-backed** (METR RCT) |
| Wrong-attribute / hallucinated-key bug pattern | **Evidence-backed** (arXiv:2403.08937, n=333) |
| Passing tests ≠ correct | **Evidence-backed** (arXiv:2607.20852) |
| Leakage invisible in reports; inspect split mechanics | **Evidence-backed** (arXiv:2207.07048, 329 papers) |
| ML config/dead-codepath debt taxonomy | **Evidence-backed** (Sculley, NeurIPS'15) |
| Karpathy training checklist | Folklore, battle-tested |
| Comment-vs-code diffing as correctness check | Folklore, but mechanically sound here |
| "85% ship without review", "1.7x more issues/PR" | **Unverified vendor marketing** — do not cite as fact |

---

## Part B — Techniques applied, and what each found

Four techniques selected for highest silent-bug yield per Part A.

### T1 — End-to-end value trace (config-key contract checking)
*Per Sculley "configuration debt" + arXiv:2403.08937 "Wrong Attribute".*

Method: pick a value that must mean the same thing at every hop, then grep every
producer and consumer and compare — **including against the real artifact on disk**,
not just the source. Traced: `latent_dim` / `num_latents` / packing layout,
`ABILITY_KEYS`, `use_qk_norm`, `soft_cap`, movement coordinates.

The decisive move was reading the actual `.pt` files instead of trusting the code:
the checkpoint's own `model_config` is ground truth and disagreed with what eval code
reconstructs. → **S1, S2, S3, S7, S12**

### T2 — Clone-and-diff for semantic drift
*Per GitClear + kunalganglani "4+ subtly different implementations".*

Method: enumerate near-duplicate files and repeated function names, then `diff` and
discard formatting noise, keeping only differing constants/formulas/defaults/keys.
Then determine which copy actually wins at import time (`sys.path`, `PYTHONPATH`,
`__init__.py`, `conftest.py`).

Found 4 latent-packing copies in two incompatible conventions, 4 `psnr()` variants over
two value ranges, 2 `load_tokenizer()` copies (one dropped its error check), 2 `auc()`
copies with different tie-breaking, and 3 rollout implementations that disagree on
whether context is noised. → **S3, S5, S8, S9, S13**

Import-shadowing was **cleared**: `_ahriuwu_patch_pickup/` (4 stale modules) is never on
`sys.path`, has no `__init__.py`, is referenced only by `.gitignore`, and would
`ImportError` on `from .layers` if forced. It is a *human* hazard (wrong file edited),
not a runtime one.

### T3 — Dead / never-executed path detection
*Per Sculley "dead experimental codepaths" + CWE-561.*

Method: reachability of imports, flags nobody sets, branches that can't be taken.
Found two eval scripts dead on import, one flag whose default silently disables the
held-out set, and overwritten-before-use initializations. → **S4, S9, S10**

### T4 — Silent-failure grep + differential execution
*Per urgentry antipatterns (ruff BLE001/S110) — extended with actual execution.*

Method: enumerate `except: pass`, `strict=False`, `.get(k, default)`,
`ImportError` fallbacks, NaN-masking. Then — the step that turns suspicion into a
finding — **build the model both ways and measure the numerical divergence.**

Two probe scripts written (both CPU-only, no GPU touched):
- `scratchpad/audit_strict_false_probe.py`
- `scratchpad/audit_qknorm_divergence.py`

---

## Part C — Findings, ranked by silent correctness risk

Legend: **[VERIFIED]** = reproduced by executing a probe against real artifacts.
**[CONFIRMED]** = confirmed by reading code + on-disk data. **[LATENT]** = correct today,
fires on a plausible near-future change.

---

### S1 — `use_qk_norm` is read from a key the trainer never writes → every dynamics eval runs a materially different model **[VERIFIED]**

**Where:** `scripts/eval_dynamics.py:302`, duplicated at `scripts/eval_trade_prediction.py:164`
```python
use_qk_norm=args.get("use_qk_norm", False),
```
**Why it's wrong:** `scripts/train_dynamics.py:476` defines the flag as `--no-qk-norm`, so
the saved `args` dict contains **`no_qk_norm`**, never `use_qk_norm`. The `.get` default
therefore *always* wins. Measured on `rollout_stage/desktop_resume_8775.pt`:

```
model_config['use_qk_norm'] = True      <- ground truth, present in the same checkpoint
'use_qk_norm' in args        = False    <- key eval reads: ABSENT
'no_qk_norm'  in args        = True (value False)
```

Two things break at once, because `src/ahriuwu/models/dynamics.py:176` couples them:
```python
self.scale = 1.0 if use_qk_norm else self.head_dim ** -0.5
```
- 36 trained QKNorm tensors (2,304 params) become "unexpected" and are dropped by
  `strict=False` at `eval_dynamics.py:307` (a printed warning only)
- attention scale silently flips **1.0 → 0.125**

Measured output divergence on identical input:

| | correct (`model_config`) | as eval builds it |
|---|---|---|
| attn scale | 1.000000 | 0.125000 |
| dropped trained params | 0 | 2,304 (36 tensors) |
| **relative L2 error** | — | **86.5%** |
| **cosine similarity** | — | **0.5354** |

**Blast radius:** every PSNR / rollout / reconstruction number produced by
`eval_dynamics.py`, plus `eval_rollout_variants.py` (which imports `load_dynamics` from
it) and `eval_trade_prediction.py`. Live since **2026-03-02** (`b9c1ce7`) — over five
months of eval numbers.

**How to verify:** `CUDA_VISIBLE_DEVICES="" python scratchpad/audit_qknorm_divergence.py`

**Fix shape:** read `checkpoint["model_config"]` (which is complete and correct) instead
of re-deriving from `args`; make the loader assert `unexpected == []`.

---

### S2 — `eval_trade_prediction.py` rebuilds the tokenizer from a size preset, leaving 112 tensors randomly initialized **[VERIFIED]**

**Where:** `scripts/eval_trade_prediction.py:149-152`
```python
model_size = tokenizer_args.get("model_size", "small")
use_rope   = tokenizer_args.get("use_rope", True)
tokenizer  = create_transformer_tokenizer(model_size, use_rope=use_rope)
tokenizer.load_state_dict(tokenizer_ckpt["model_state_dict"], strict=False)
```
**Why it's wrong:** the v7 tokenizer is preset `large` **with overrides**
(`latent_dim=16`, `num_latents=512`, `num_encoder_layers=8`, `temporal_every=4`). The
preset alone gives a different architecture. Reconstructing from `model_size` alone
discards every override, and `strict=False` swallows the mismatch. Measured against the
real checkpoint:

```
checkpoint tensors: 245 (208.4M params)   model tensors: 349
MISSING in ckpt (stay RANDOM):     112     <-- entire encoder blocks 8+
UNEXPECTED in ckpt (DISCARDED):      8
SHAPE MISMATCH (DISCARDED):         14
  encoder.latent_tokens:    ckpt(1,512,1024) vs model(1,256,1024)
  encoder.rope.inv_freq:    ckpt(16,)        vs model(32,)
```
**112 randomly-initialized tensors** and the script runs to completion and prints trade
predictions.

The correct pattern already exists in the same repo —
`scripts/pretokenize_replay_v7.py:38-52` (`load_v7`) builds from `**model_config` and
**raises** on any non-RoPE key mismatch. Same probe against `load_v7`: `0 / 0 / 0`.
The header comment at `pretokenize_replay_v7.py:14-20` even documents this exact trap.
The knowledge existed; it just never propagated to the sibling call site.

**Note the partial-fix signature:** eight lines below, at `eval_trade_prediction.py:161`,
sits a comment `# FIX #7: Read model config from checkpoint args instead of hardcoding`
— applied to the *dynamics* loader and not to the *tokenizer* loader directly above it.
This is the canonical LLM copy-paste divergence: one of two sibling sites fixed.

**How to verify:** `CUDA_VISIBLE_DEVICES="" python scratchpad/audit_strict_false_probe.py`

---

### S3 — Two incompatible latent packing conventions produce byte-identical shapes **[CONFIRMED]**

Every latent file on disk is `(T, 32, 16, 16) float16`. **That shape is ambiguous.**

| convention | producer | meaning of the axes |
|---|---|---|
| **v7** — 512 tokens × 16 dim | `scripts/pretokenize_replay_v7.py:96` | `(32,16,16)` is a *repacking* of a 512-token sequence; the 16×16 is **not spatial** |
| **legacy** — 256 tokens × 32 dim | `scripts/pretokenize_frames.py:169-174` | `(32,16,16)` is genuinely `(latent_dim, H, W)` |

Both do `latents.view(B, 16, 16, -1).permute(0, 3, 1, 2)`; both yield `(B,32,16,16)`;
`8192 = 512×16 = 256×32`. **Nothing in the file records which produced it.**
`pretokenize_frames.py:171` still asserts in a comment `= (B, 256, 32)`.

Confirmed the live tokenizer is the **512×16** one, from its own checkpoint:
`latent_dim=16, num_latents=512, size_preset=large`.

Consumers hardcode the v7 convention with no assertion tying it to the loaded tokenizer:
- `scripts/train_dynamics.py:1048` `z.permute(0,1,3,4,2).reshape(B, T*512, 16)`
- `scripts/overlay_e2e.py:27`, `scripts/test_agent_infer.py:38` `reshape(1, 512, 16)`
- `scripts/agent_infer.py:43`

Worst case, `scripts/eval_trade_prediction.py`: its own fold (`:232` `view(1,16,16,-1)`)
and unfold (`:257` `permute(0,2,3,1).reshape(1,256,-1)`) are **not inverses** under v7.
Element count matches, so nothing raises — `decode()` just receives a scrambled token
layout and returns a plausible blurry frame.

Compounding this, `scripts/train_dynamics.py:334` help text is wrong:
> `"Latent dimension per token (must match tokenizer: tiny=16, small=32, medium=48, large=64)"`

The live tokenizer is preset `large` with `latent_dim=16`. Anyone following this help
text passes 64.

**How to verify:** `python -c` load any latent `.pt`, confirm `(T,32,16,16)`; then load
the tokenizer ckpt and read `model_config['num_latents']`/`['latent_dim']`. If they are
512/16, `pretokenize_frames.py` outputs are incompatible and must not be mixed in.

**Fix shape:** write `num_latents`/`latent_dim` into the latent `.pt` alongside
`frame_indices`, and assert on load.

---

### S4 — Dynamics "validation" loss is computed on training data by default **[CONFIRMED]**

**Where:** `scripts/train_dynamics.py:527` + `:1506-1515`
```python
parser.add_argument("--holdout-videos", type=int, default=0, ...)
...
val_batch = holdout_val_batch          # None when --holdout-videos is 0
if val_batch is not None: ...
elif dataloader_short is not None:
    val_batch = next(iter(dataloader_short))   # <-- TRAINING loader
elif dataloader is not None:
    val_batch = next(iter(dataloader))         # <-- TRAINING loader
```
`_pick_holdout` (`:278-283`) returns an empty set for `n <= 0`, so no video is excluded
from the sampler either.

Of the launch scripts, only `scripts/dyn_train_args_action.sh:30`
(`--holdout-videos 2`) and `scratchpad/smoke_action_mixed.sh:44` (`1`) set it.
**None of the `slurm/*.sbatch` dynamics jobs pass it** — so those runs reported
train-set metrics under a `val/` label.

This is corroborated by the repo's own prior notes (`WM_UPDATE_2026-07-13.md:36`,
`WORLD_MODEL_BRIEFING.md:133`) — though both refer to the flag as `--num-eval-videos`,
which **does not exist**; the flag was renamed and the docs weren't. A reader following
those docs cannot fix the problem.

Per Kapoor & Narayanan: this class of error is undetectable from the reported numbers.
The run prints a plausible val PSNR that is really train PSNR.

**Related, separately documented:** `verify/VERIFICATION_REPORT.md:9,30` already flags
that the 6 YT eval-holdout videos sit *inside* the tokenizer training corpus with no
exclusion. Not re-reported here; noting it compounds S4.

**How to verify:** grep the launcher for `--holdout-videos`; if absent, `val_batch` came
from `dataloader`. Cross-check by confirming `_pick_holdout` returned `set()`.

---

### S5 — `strict=False` with no missing/unexpected check, on paths that write the corpus and drive live play **[CONFIRMED]**

The repo's own known-bug history includes "`strict=False` hid dropped trained weights".
The pattern is still live at sites with no guard at all:

| site | consequence |
|---|---|
| `scripts/pretokenize_frames.py:151` | **worst** — this writes the *entire latent corpus*. A partly-random tokenizer emits statistically plausible latents that silently poison every downstream dynamics and BC run. Return value discarded entirely. |
| `scripts/agent_infer.py:108` | the **live play** path; a dropped agent block just plays badly |
| `scripts/train_imagination.py:152` | the frozen "pretrained" world model can be part-random; imagined returns are measured against it |
| `scripts/finetune_shortcut.py:117` | then freezes everything except 3 modules, so a silently-random frozen tensor is never trained out |
| `scripts/eval_trade_prediction.py:152,170` | see S2 |

Fuzzy variant — `scripts/eval_dream_quality.py:44-45`, `scripts/rollout_check.py:75-77`:
```python
assert len(miss) + len(unexp) <= 10, ...
```
tolerates up to 10 randomly-initialized tensors (a whole attention block is q/k/v/out +
2 norms = 6). `eval_dream_quality.py:51` loads the tokenizer with **no** check.

**Template to copy:** `scripts/pretokenize_replay_v7.py:45-51` raises `RuntimeError` on
any non-RoPE mismatch.

---

### S6 — Dataset cache key omits the reward config and outcomes manifest **[scan-reported, code-confirmed]**

**Where:** `src/ahriuwu/data/replay_dataset.py:153-158`
```python
return {"latents_dir": str(self.latents_dir),
        "seq_len": self.sequence_length, "stride": self.stride,
        "schema": 3}
```
The cached blob contains precomputed `md["rewards"]`, built from `self.reward_config`
and the win/loss `outcomes` dict — **neither is in the key**, and `schema` is a
hand-bumped integer. Change `gold_scale` / `death_penalty` / lane thresholds, or fix a
wrong win-label, and training silently reuses the **old** reward targets while printing
`[dscache] HIT`. The reward head converges beautifully on the previous reward function.

**How to verify:** hash `reward_config` + outcomes into the key and confirm the cache
misses on a config change.

---

### S7 — `soft_cap=0.0` means "cap at zero", not "disabled" **[LATENT — not firing on current checkpoints]**

**Where:** `scripts/eval_dynamics.py:303`, `scripts/eval_trade_prediction.py:165`
```python
soft_cap=args.get("soft_cap", 0.0),
```
The trainer (`train_dynamics.py:232`) stores `args.soft_cap if args.soft_cap > 0 else None`
— so `None` means disabled. The eval's fallback is `0.0`, and `0.0 is not None`, so
`src/ahriuwu/models/layers.py:458-459` fires `soft_cap_attention(attn, 0.0)`:
`0.0 * tanh(logits/0.0)` → **0.0 for every logit** (NaN for exactly-zero logits).
Attention collapses to uniform over all keys, with zero missing/unexpected keys and no
warning.

**Verified not currently firing:** `desktop_resume_8775.pt` has `args['soft_cap'] = 50.0`
present. This fires only for checkpoints saved without the key, or trained with
`--soft-cap 0`. The two sides disagree on the meaning of the same value — a live
landmine.

---

### S8 — Context-noise τ desync: the model is told a noise level different from the one applied **[scan-reported, code-confirmed]**

`scripts/eval_dynamics.py:434-438` fixed this and documents it:
> *"FIX #6: Noise context once and store tau values to avoid desync. Re-sampling context
> noise each denoising step changes the tau values reported to the model vs the actual
> noise level applied."*

`scripts/eval_trade_prediction.py:95-103` does exactly what that comment forbids:
```python
 96:  ctx_tau      = (1.0 - tau_ctx) + torch.rand(B, T, 1, 1, 1, ...) * tau_ctx
 97:  z_ctx_noisy  = ctx_tau * context_latents + (1 - ctx_tau) * noise_ctx
102:  ctx_tau_flat = (1.0 - tau_ctx) + torch.rand(B, T, ...) * tau_ctx   # SECOND independent draw
103:  tau = torch.cat([ctx_tau_flat, tau_target.expand(B, 1)], dim=1)
```
Two independent draws: latents corrupted at `ctx_tau`, model told `ctx_tau_flat`, and
both re-randomized every denoising step. Same fix-didn't-propagate signature as S2.

---

### S9 — Fourth inline copy of the Euler renoise step reuses frozen initial noise **[scan-reported, code-confirmed]**

`src/ahriuwu/models/diffusion.py:196-202` docstring:
> *"Using the implied ε̂ — rather than the frozen initial noise the sampler started from —
> is what makes multi-step (K>1) denoising converge instead of diverge. This was a real
> bug; **every sampler (rollout + all eval scripts) must go through this helper so the
> fix cannot drift back out of sync.**"*

`scripts/probe_shortcut_k4k64.py:46-47` does not:
```python
nt  = tau_t + step
tgt = nt * z0t + (1 - nt) * noise0      # frozen initial noise, reused every step
```
That script exists specifically to compare K=4 vs K=64 — the exact compounding-error
regime the docstring warns about. Its K=64 numbers are not comparable to
`dynamics.rollout()`.

Three rollout implementations also disagree on whether context is noised at all
(`dynamics.rollout()` labels context with `ctx_tau` but feeds it **clean**;
`eval_dynamics.rollout_predictions()` actually noises it;
`probe_shortcut_k4k64.py` keeps it clean at τ=1). PSNR from these three families is
mutually incomparable.

---

### S10 — Two eval scripts are dead on import **[scan-reported]**

`scripts/eval_rollout_variants.py:47` and `scripts/eval_dynamics_per_frame_psnr.py:37`:
```python
from ahriuwu.models.transformer_tokenizer_legacy import (...)
```
`src/ahriuwu/models/transformer_tokenizer_legacy.py` was deleted in `3f1d40e`. The
orphan `__pycache__/transformer_tokenizer_legacy.cpython-311.pyc` **cannot** satisfy the
import (PEP 3147 requires sourceless `.pyc` at the source location). These raise
`ModuleNotFoundError`. Ironically these are the two scripts that anchor `sys.path`
robustly to `REPO_ROOT`.

*This is the same class as the repo's known `import sys` bug — a documented-as-working
path that has never run.*

---

### S11 — Module docstring documents an action space two renames stale **[CONFIRMED]**

`src/ahriuwu/data/replay_dataset.py:23`
```
* ``actions``: ``{movement: (T, 2), Q W E R D F item B: (T,) long}``
```
Actual (`src/ahriuwu/constants.py:29`):
`['Q','W','E','R','Flash','Ignite','AA','Recall','Stride']` — 9 keys, not 8.

`item` was dropped **two** renames ago; git shows
`['Q','W','E','R','D','F','item','B']` → `[...,'D','F','B','C']` → current (`7013f3a`,
2026-06-29). The "Action mapping" section (`:27-38`) still documents `D/F` summoner-slot
resolution, `B`-slot recall, and `item` — none of which exist.

Same block, `:36`: movement documented as `label.movement.heading_screen` normalized by
`labels.screen_resolution`. Actual behavior (`_parse_movement`, `:379-422`) reads
`label.cursor.screen` and only *falls back* to `heading_screen`.

Per A.6: in LLM-written code, comments describe intention, not behavior. This one
would actively mislead anyone debugging the action pipeline.

**Checked and cleared:** the `ABILITY_KEYS` renumbering (index 6 changed Recall→AA,
7 changed AA→Recall) predates every checkpoint on disk — all are ≥ 2026-07-14, the
change was 2026-06-29. No live checkpoint has permuted ability semantics.

---

### S12 — Action-conditioning validity keyed on a hardcoded match-ID prefix **[LATENT]**

`src/ahriuwu/data/replay_dataset.py:610-612`
```python
actions["cursor_valid"] = torch.full((T,), match_id.startswith("NA1_"), dtype=torch.bool)
```
Add EUW1/KR/EUN1 replays and every real labeled action is flagged **invalid**;
`embed_actions` (`dynamics.py:576`) then substitutes `no_action_embed` for all of it.
The model trains as if action-conditioned while receiving no action signal, loss looks
normal, and action-conditioning ablations conclude "actions don't help".

Verified currently safe: all match IDs on disk are `NA1_*`.

Related train/serve skew: `scripts/agent_infer.py:175` builds `actions = {"movement": mv}`
with **no** `cursor_valid`, so `dynamics.py:566-578` takes the legacy branch and treats
padded/cursor-less early frames as *real* clicks — exactly the frames trained with
`no_action_embed`.

---

### S13 — Metrics that report a healthy number when the thing being measured failed **[scan-reported]**

- `scripts/train_dynamics.py:893` — `total_grad_norm += grad_norm.item() if torch.isfinite(grad_norm) else 0.0`.
  Exploding steps contribute **0**, so `train/grad_norm` under-reports in exactly the
  regime it exists to detect. Instability reads as "norms near zero".
- `src/ahriuwu/models/diffusion.py:510-521` — on NaN, `n_boot = 0` and `loss_boot`
  stays at its `0.0` init, which is what `info["loss_boot"]` logs. A permanently
  diverging shortcut branch reports `train/loss_boot = 0.0` — indistinguishable from
  perfect convergence. The neighbouring `x_mse.clamp(max=100.0)` separately flattens a
  diverging curve into a plateau.
- `scripts/eval_rollout_variants.py:218,240-241` — NaN rollouts are filtered out before
  averaging. A variant that diverges on half its sequences is scored on the surviving
  half, so the **less stable variant can win**. The CSV keeps an `n` column; the plot
  humans actually look at drops it.
- `scripts/train_agent_finetune.py:706-708` — non-finite batches `continue`, skipping
  `global_step += 1` and `scheduler.step()`. No rate counter, so a 90%-skipped run and a
  0%-skipped run produce identical-looking logs (with a desynced LR schedule).

---

### S14 — Environment divergence between launchers **[CONFIRMED]**

Launch scripts reference **two different Python environments**:
`source ~/Repos/ahriuwu/.venv/bin/activate` vs `conda activate ml` /
`/home/dani/miniconda3/envs/ml/bin/python`. Neither is pinned in `pyproject.toml`.
This is the same shape as the repo's known bitsandbytes-mismatch bug.

Compounding: `scripts/train_dynamics.py:304` defaults `--tokenizer-ckpt` to
`/mnt/storage/data/ahriuwu-checkpoints/tokenizer_v7/transformer_tokenizer_latest.pt`,
and `src/ahriuwu/constants.py` defaults `DATA_ROOT=/mnt/storage/ahriuwu-data`,
`LATENTS_DIR=/opt/ahriuwu/latents_pt`. **All three paths are absent on this host.**
Silent-failure grade is low (these fail loudly), but they mean the checked-in defaults
describe no machine that currently exists.

Also: most `sys.path.insert` calls use *relative* strings (`"src"`, `"scripts"`,
`"scratchpad"`), so they only resolve when cwd is the repo root.

---

### S15 — Cleared hypotheses (checked, not bugs)

Recorded so they aren't re-audited:

- **Stale-module shadowing** — `_ahriuwu_patch_pickup/` (4 modules, ~3 months stale)
  is **not reachable**: no `__init__.py`, never on `sys.path`/`PYTHONPATH`, referenced
  only by `.gitignore:61`, and would `ImportError` on `from .layers` (it imports
  `_soft_cap_score_mod`, renamed to `make_soft_cap_score_mod`). It carries real
  divergences (hardcoded `soft_cap=50.0`, hardcoded `k_max=64`, no tube masking,
  non-atomic checkpoint writes, compile-only unwrap that breaks under DDP) — so it is a
  **human hazard** (wrong file edited), not a runtime one. Recommend deleting.
- `scripts/_archive/` (250+ files) — `sys.path.insert(0, "scripts")` does not reach
  subdirectories; live `scripts/*.py` always wins.
- `/mnt/nfs/projects/ahriuwu` and `/srv/nfs/projects/ahriuwu` are the **same directory**
  (identical device/inode), not a duplicate checkout.
- `ABILITY_KEYS` ordering is consistently derived from the constant at every use site
  (no hardcoded index literals), and the renumbering predates all checkpoints.
- `build_dynamics()` in `train_agent_finetune.py` vs `train_imagination.py` — identical.
- **Good patterns worth preserving:** `pretokenize_replay_v7.py:45-51` (hard-fail on key
  mismatch), `rewards/reward.py:147-151` (`_safe_float` returns `None`, not 0),
  `returns.py:363-371` (skips NaN RMS updates), `train_dynamics.py:865-885` (explicit
  non-finite gradient gate), `utils/training.py:393-402` (atomic checkpoint write).

---

## Recommended order of work

1. **S1** — one-line class of fix (`model_config` over `args`), invalidates months of
   eval numbers. Re-run the headline evals after fixing.
2. **S4** — decide whether any reported dynamics val metric is trustworthy.
3. **S2 / S5** — make `load_v7`'s guard the only way to load a checkpoint anywhere.
4. **S3** — stamp latent geometry into the `.pt` and assert on load.
5. **S6** — hash reward config into the cache key.
6. **S13** — stop logging 0.0 for "failed"; log a NaN or a separate failure-rate counter.

**Systemic:** S1, S2, S8, S9 are all the same failure — *a fix applied to one of N
copies*. The single highest-leverage structural change is to delete the duplicate
loaders/samplers and route every script through one shared, guarded implementation,
which is what `diffusion.py:196-202` already asks for in writing.

---

### Probe scripts added

- `scratchpad/audit_qknorm_divergence.py` — builds the dynamics model both ways from one
  checkpoint, reports dropped tensors, attention scale, and output divergence.
- `scratchpad/audit_strict_false_probe.py` — reports how many trained tensors each
  tokenizer-construction path silently discards or leaves random.

Both are CPU-only (`CUDA_VISIBLE_DEVICES=""`) and touch no GPU or running process.
