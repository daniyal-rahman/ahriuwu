# CODEBASE_COMPARISON.md — ahriuwu against the public world-model implementations

**Written 2026-08-27.** Two questions, answered separately:

1. **Part 1** — how does this codebase compare, technically and as engineering, to the public
   implementations we might learn from?
2. **Part 2** — for each of those external repos, how much should we trust it? Before we copy a
   pattern out of someone's repo, what is the evidence it was ever *run*?

**Companion documents.** [`PAPER_DEVIATIONS.md`](PAPER_DEVIATIONS.md) measures us against the paper;
[`DESIGN_DECISIONS.md`](DESIGN_DECISIONS.md) measures our choices against the alternatives;
[`WIRING_AUDIT_2026-08-20.md`](WIRING_AUDIT_2026-08-20.md) and
[`MOVEMENT_HEAD_BLIND_2026-08-26.md`](MOVEMENT_HEAD_BLIND_2026-08-26.md) are the defect record. This
document measures us against *other people's code*, which none of those do. Known defects from those
four are not re-reported here; where this pass adds something to one of them, §1.6 says so explicitly.

**Not covered here:** action-conditioning mechanics. A parallel survey
(`DREAMER_IMPLEMENTATIONS_SURVEY.md`) owns that topic in depth. Where action conditioning appears
below it is only as a row in a comparison table.

## How to read the evidence labels

| Label | Means |
|---|---|
| **HARD** | Verified in this session at the cited `file:line`. Re-runnable. |
| **REPORTED** | Read by a delegated reviewer at the cited `file:line`, not independently re-checked by me. Treat as reliable-but-single-sourced. |
| **INFERENCE** | A judgement I reasoned to. The reasoning is stated so you can disagree with it. |

Where I could not establish something, it says so rather than guessing. See
[Appendix A](#appendix-a--what-i-could-not-verify).

---

## 0. The comparison set

The brief named `danijar/dreamerv3` and `NM512/dreamerv3-torch`. Both were surveyed. But the useful
result of this pass is that **neither is our reference class**, and that two categories the brief
did not name are (§1.1). The full set examined:

| Repo | What it is | ★ | Reference value to us |
|---|---|---|---|
| [`danijar/dreamerv3`](https://github.com/danijar/dreamerv3) | Official JAX DreamerV3 (RSSM) | 3.7k | Hyperparameter source; twohot/return-norm reference |
| [`NM512/dreamerv3-torch`](https://github.com/NM512/dreamerv3-torch) | PyTorch DreamerV3 port | 887 | Clearest PyTorch statement of twohot / KL balancing |
| [`edwhu/dreamer4-jax`](https://github.com/edwhu/dreamer4-jax) | DreamerV4, full 4-phase, toy scale | 102 | **Closest structural match to our pipeline** |
| [`lucidrains/dreamer4`](https://github.com/lucidrains/dreamer4) | DreamerV4 as a library, full pipeline | 213 | Loss normaliser, MTP heads, PMPO |
| [`vijayabhaskar-ev/dreamer_v4`](https://github.com/vijayabhaskar-ev/dreamer_v4) | DreamerV4, 1 DMC task, measured | 38 | Only repo with closed-loop numbers + error bars |
| [`nicklashansen/dreamer4`](https://github.com/nicklashansen/dreamer4) | DreamerV4 world model only | 386 | Best-validated PyTorch WM; released weights |
| [`next-state/open-dreamer`](https://github.com/next-state/open-dreamer) | DreamerV4 WM at Minecraft scale | 370 | Scale reference. **All-Rights-Reserved licence** |
| [`kwsong0113/diffusion-forcing-transformer`](https://github.com/kwsong0113/diffusion-forcing-transformer) (DFoT) | Frozen VAE + latent diffusion forcing + DiT | 708 | **Closest architecture match outside Dreamer** |
| [`buoyancy99/diffusion-forcing`](https://github.com/buoyancy99/diffusion-forcing) | Diffusion Forcing (our objective's ancestor) | 1.3k | The objective's origin |
| [`eloialonso/diamond`](https://github.com/eloialonso/diamond) | Pixel-space diffusion world model | 2.1k | Rollout-drift mitigations |
| [`eloialonso/iris`](https://github.com/eloialonso/iris) | Transformer WM, discrete tokenizer | 900 | Frozen-vs-joint tokenizer question |

**There is no official DreamerV4 code and there will not be one.** REPORTED — Hafner on TalkRL,
2025-11-09: *"Probably not. I would love to release code and checkpoints, but I think getting the
paper out was already exceptional enough in the current climate."* Corroborating: all 99 repos under
`github.com/danijar` enumerated, no dreamer4; `danijar.com/project/dreamer4/` has no "Code" button
(the DreamerV3 project page does). **Every DreamerV4 implementation in the table, including ours, is
an independent reading of the same 32-page paper.**

That fact is load-bearing for the rest of this document. Where four independent readings agree with
each other and we disagree with all four, that is worth more than any single repo's authority.

---

# Part 1 — ahriuwu against the references

## 1.1 We have been comparing ourselves against the wrong repos

The two DreamerV3 repos are **RSSM** implementations: a recurrent latent-state model with a stochastic
posterior, trained with KL balancing and free bits, on an online replay buffer fed by a live
environment. We are a **frozen tokenizer → latent diffusion-forcing transformer** trained on an
offline video corpus with no environment at all.

Concretely, the following DreamerV3 concepts have no counterpart in our code and should not be
"ported": free bits (`kl_free: 1.0`), KL balancing (`dyn_scale: 0.5 / rep_scale: 0.1`), posterior
vs prior, `unimix`, `train_ratio` / replay ratio, the replay buffer itself, and the recurrent carry.
HARD: grep for `free_bits|kl_free|kl_balance` across `src/` and `scripts/` returns zero hits, correctly.

What *does* transfer from the DreamerV3 lineage is narrow and specific: **twohot/symexp**, **λ-returns**,
**γ=0.997**, and the **slow-critic** idea. Everything else about the training loop is a different
problem.

The repos that actually match our shape are:

- **`edwhu/dreamer4-jax`** — tokenizer → dynamics → agent tokens + MTP heads → PMPO. Same four phases.
- **`kwsong0113/…/DFoT`** — frozen pretrained VAE, offline precomputed latents, diffusion forcing,
  transformer backbone, external conditioning with dropout. REPORTED (`algorithms/dfot/dfot_video.py:174-186`
  freezes the VAE; `:275-279, :372-378` is the offline-precomputed-latents path). Every leg of our stack.
- **`lucidrains/dreamer4`** and **`vijayabhaskar-ev/dreamer_v4`** for the head/Phase-3 half.

**Recommendation.** When a design question comes up, check `edwhu/dreamer4-jax` and DFoT before
either DreamerV3 repo. Clones are already on disk — see [Appendix B](#appendix-b--local-clones).

## 1.2 Architecture fidelity

### 1.2.1 Tokenizer

| | ahriuwu | Paper | edwhu / nicklashansen | DFoT | IRIS | DIAMOND |
|---|---|---|---|---|---|---|
| Type | Block-causal ViT MAE, 512×16 bottleneck | same | same | pretrained VAE | VQ-VAE (512×512 codebook) | **none** (pixel diffusion) |
| Frozen for dynamics | yes | yes | yes | yes (`freeze_model`) | yes (`torch.no_grad()` wall) | n/a |
| Objective | MSE + 0.2·LPIPS | MSE + 0.2·LPIPS (Eq 5) | same | — | L1 + LPIPS + VQ (β=1.0) | — |
| Latent regulariser | none | none | none | per-channel stats | commitment | n/a |

**We match the paper here, and we match the field.** HARD (paper, re-extracted with pypdf to
`scratchpad/d4.txt`): Eq 5 is reconstruction only — MSE + 0.2·LPIPS, no latent regulariser of any
kind. Our `TokenizerLoss` (`src/ahriuwu/models/losses.py:157`) is the same.

**The one thing worth knowing:** REPORTED — *no public reference trains the tokenizer jointly with
the dynamics model.* IRIS trains them concurrently but with a hard `torch.no_grad()` gradient wall
(`src/models/world_model.py:99-100`) plus a 20-epoch stagger; DFoT freezes a pretrained VAE outright.
Our frozen-v7 decision is not a compromise — it is what everyone does.

**Where we diverge from the paper**, per `PAPER_DEVIATIONS` §1.1/§1.2: tube masking and a mask-ratio
curriculum meant the paper's MAE regime was never actually trained (measured mean input mask 0.1–0.2
vs the paper's ~0.45). **This is now fixed in code** — HARD: `make_mask(tube=False)` is the default
at `src/ahriuwu/models/transformer_tokenizer.py:719`, and its docstring records why. The commit is
`fd7632f` (2026-08-20). But the *deployed v7 checkpoint* was trained under the old regime, so the
deviation is still baked into the latents everything downstream reads. See §1.7 on the doc staleness
this creates.

### 1.2.2 Dynamics / shortcut forcing

| | ahriuwu | Paper | edwhu | nicklashansen | lucidrains | vijayabhaskar |
|---|---|---|---|---|---|---|
| Parameterisation | x-prediction, flow-matching interp | same | same | same | same | same |
| Interpolant | `τ·z₀ + (1−τ)·ε`, τ=1 clean | same | same | same | same | same |
| Ramp weight | `0.9τ + 0.1` | Eq 8 | ✓ | ✓ | ✓ | ✓ |
| Shortcut forcing | **OFF** | ON, K=4 | ON | ON | ON | ON |
| Per-frame i.i.d. τ | yes | yes | yes | yes | yes | yes |
| Space/time factorised, temporal every 4 | yes | yes | yes | yes | yes | yes |
| GQA / QKNorm / soft-cap / RMSNorm / SwiGLU / RoPE | yes | yes | yes | yes | yes | yes |

HARD (ours): `diffusion.py:62` (interpolant), `diffusion.py:224` (ramp), `diffusion.py:227`
(x-prediction loss), `dynamics.py:363` (`temporal_every=4`), `layers.py` (QKNorm/soft-cap/RoPE).

**Shortcut forcing is the one big architectural thing we do not have, and five independent
implementations agree on how to do it.** REPORTED — the self-distillation loss was read in
`nicklashansen/dreamer4/dreamer4/train_dynamics.py:238-256`, `lucidrains/dreamer4.py:8053-8095`,
`edwhu/scripts/train_dynamics.py:288-307`, `open-dreamer/dreamer/training.py:244-275`, and
`vijayabhaskar/dynamics/trainer.py:1084-1111` — same algebra in all five, with all four load-bearing
details present: step size conditioned through a *separate* discrete embedding from the noise level;
target = two chained half-steps, averaged, stop-gradiented; `(1−τ)²` weighting mapping x-space error
into velocity space; and σ/d sampled at shape `(B,T)` — per-frame independent, which is the
*forcing* half of *shortcut forcing*.

**Our dormant shortcut path deviates from all five** in exactly the way `PAPER_DEVIATIONS` §2.3
records (bootstrap computed in x-space, not velocity space, and `bootstrap_weight=10.0`). Because
the path is off, this costs nothing today — but §2.14's "dormant deviations" become live the moment
anyone runs `finetune_shortcut.py`. **Before that happens, diff `diffusion.py:387-540` against
`edwhu/scripts/train_dynamics.py:288-307`.** Five agreeing references is the cheapest correctness
check available to us.

Two smaller notes:
- REPORTED — `nicklashansen` builds and discards autograd graphs for the two extra teacher forwards;
  `vijayabhaskar` and `open-dreamer` wrap them in `no_grad`. Numerically identical, materially cheaper.
- REPORTED — `open-dreamer` uses a separate **EMA `bootstrap_model`** as the teacher, "for more stable
  bootstrap targets". Beyond the paper, and directly aimed at the bootstrap-trap problem our
  `ShortcutForcing` docstring (`diffusion.py:292`) says it is working around with a τ-schedule instead.

### 1.2.3 Agent tokens — **the clearest divergence in the stack**

The paper (HARD, `scratchpad/d4.txt`, §3.3):

> *"we insert agent tokens as an additional modality into the world model transformer and
> **interleave** it with the image representations, actions, and register tokens. … While the agent
> tokens attend to themselves and all other modalities, no other modalities can attend back to the
> agent tokens. This is crucial for avoiding causal confusion of the world model."*

**Everyone else puts agent tokens in the sequence and enforces the firewall with an attention mask.**

- HARD — `edwhu_dreamer4-jax/dreamer/models.py:680`:
  `toks = [action_tokens, signal_tok, step_tok, spatial_tokens, register_tokens, agent_tokens]`,
  run through all `depth` blocks, then `models.py:689` slices `h_t = x[:, :, self.agent_slice, :]`.
  The firewall is the mask (`space_mode="wm_agent"` / `"wm_agent_isolated"`), and it is **tested by
  exact bit-equality**: `models.py:1071-1107`,
  `assert jnp.allclose(x1_a, x1_b, atol=0, rtol=0), "x1_hat changed with agent tokens—firewall broken"`.
- HARD — `lucidrains/dreamer4/dreamer4/dreamer4.py:7919`:
  `pack([flow_token, space_tokens, proprio_token, state_pred_token, registers, action_tokens, reward_tokens, maybe_aug_token, agent_tokens], 'b t * d')`.
- HARD — `vijayabhaskar-ev/dreamer_v4/dynamics/dynamic_model.py:186` inserts `agent_tok` into the
  per-frame token set, and `:83-84` **rebuilds spatial RoPE with `+num_agent_tokens` positions** —
  i.e. the agent token occupies a real position in the sequence.

**We do it differently.** HARD — `src/ahriuwu/models/dynamics.py:719-748`: the full 18-block stack
runs first (`x = self._run_blocks(...)`, `:720`), and only then do four `AgentTokenBlock`s
cross-attend from a *separate residual stream* into that final-layer `x`:

```python
for agent_block in self.agent_blocks:
    agent_tokens = agent_block(agent_tokens, x)      # dynamics.py:739-747
```

`_build_tokens` (`dynamics.py:760-802`) never contains an agent token; the sequence is
`[latents, registers, action, condition]` only.

**Is this a mistake?** The firewall property is satisfied — structurally, in fact, which is stronger
than a mask and cannot be broken by a mask bug. `PAPER_DEVIATIONS` §4.7 calls it NOT-A-DEVIATION on
exactly that basis, and for the firewall it is right. **But the firewall is not the only property.**
"Interleave into the transformer" means agent tokens are refined *alongside* the representation
through all 18 layers; ours read one fixed summary produced by a stack that was trained (Phase 1)
without them and is frozen (Phase 2). Our agent tokens have **no access to any intermediate-layer
feature**.

INFERENCE, and I think a strong one: this is the mechanism that produces exactly the symptom
`MOVEMENT_HEAD_BLIND_2026-08-26.md` ends on. That document's closing question is:

> *"the probe read the AGENT TOKEN and the RAW LATENTS, not the dynamics' INTERNAL SPATIAL TOKENS.
> If the signal is there but the agent token fails to surface it, the fix is architectural
> (agent-block depth, cross-attention) and cheap."*

A four-block cross-attention read of layer-18 output is precisely the configuration in which "the
signal is in the model but not in the agent token" is possible. **This is a concrete, testable
architectural hypothesis for the open question, and it was not on the candidate list.** §1.6-A
below states the cheap version of the test.

### 1.2.4 MTP heads

| | ahriuwu | Paper (Eq 9) | edwhu | lucidrains | vijayabhaskar |
|---|---|---|---|---|---|
| MTP length | 9 (L=8, n=0..8) | L=8 | ✓ | ✓ | ✓ |
| Action term covers n=0 | **no** | yes | yes | yes | yes |
| Reward term covers n=0 | yes | yes | yes | yes | yes |
| One output layer per offset | yes | yes | ✓ | ✓ | ✓ |

HARD (ours): `heads.py:238-241` (nine `movement_heads`), `train_agent_finetune.py:739-747`
(reward covers n=0), and the BC loop drops n=0.

Dropping n=0 for actions is a deliberate label-leak fix and is defensible in isolation. Its
consequence is not: `WIRING_AUDIT` §2.1 measured `heads.0.weight` and `movement_heads.0.weight` at
**exactly zero norm** after 99,421 steps, while every Phase-3 read is offset 0. That is already on
the fix list; this pass adds only that **no other implementation has this hazard**, because none of
them drop n=0 — so there is no external precedent to copy a fix from. The fix has to be ours:
either train n=0 with the action input masked, or make Phase 3 read the first *trained* offset
(which `train_imagination.py` already parameterises as `MTP_OFFSET`).

### 1.2.5 Loss normalisation (RMS)

The paper (HARD): *"To train a single dynamics transformer with multiple modalities and output
heads, we normalize all loss terms by running estimates of their root-mean-square (RMS)."*

**We match, and so do the two DreamerV4 repos that implement it.** But the details differ in ways
that are worth stealing:

| | ahriuwu `RunningRMS` | lucidrains `LossNormalizer` | vijayabhaskar `RMSNormalizer` |
|---|---|---|---|
| Location | `returns.py:334-395` | `dreamer4.py:688-726` | `dynamics/trainer.py:140-141, 212-214` |
| Decay | 0.99 | **0.95** | 0.99 |
| Init | `None`, takes first value | **`ones(num_losses)` buffer** | — |
| Storage | plain attr + manual `state_dict` | **`register_buffer`** | `state_dict` |
| Cold-start guard | `MIN_RMS = 1e-4` clamp (`:383`) | not needed (starts at 1) | — |
| Device fix on resume | **manual** (`:377-379`) | not needed (buffer) | — |
| Eval-mode behaviour | hand-rolled `_rms_normalize` in the trainer | `update_ema = default(update_ema, self.training)` | — |

HARD, both sides. Ours (`returns.py:334-395`) is functionally correct but carries three workarounds
that lucidrains' design does not need: the `MIN_RMS` clamp exists because `rms` starts at `None` and
takes the first observed loss; the device-mismatch fix at `:377-379` exists because it is a plain
attribute rather than a buffer; and the validation path is a hand-copied duplicate
(`train_agent_finetune.py:926-934`, whose own comment says *"Mirrors RunningRMS.update's tail
exactly (it clamps rms, not sqrt(rms))"* — a clone that must be kept in sync by hand, which is the
duplication smell `VIBE_AUDIT_2026-08-12.md` §T2 is about).

**Actionable, small:** make `rms` a registered buffer initialised to `ones`, and gate the EMA update
on `self.training`. That deletes the clamp, the device fix, and the duplicated validation path.

**Two real problems in our RMS wiring, both HARD:**

1. **RMS is switched off entirely when shortcut forcing is on.** `train_dynamics.py:1568`:
   ```python
   rms_dict = None if args.shortcut_forcing else {"x_pred": RunningRMS(), "pixel": RunningRMS()}
   ```
   This is `PAPER_DEVIATIONS` §2.14, correctly flagged as dormant. The external precedent settles the
   fix: `vijayabhaskar` keeps a **separate** `rms_bootstrap` tracker alongside `rms_flow`
   (`trainer.py:140-141`) — REPORTED. Two terms, two trackers, which is what the paper's sentence
   actually asks for. Do that rather than disabling normalisation.
2. **We RMS-normalise the Phase-3 losses, and the paper says not to.** HARD — `train_imagination.py:588`
   creates `{"value": RunningRMS(), "policy": RunningRMS()}`. The paper (HARD) says of PMPO: *"uses
   the sign of the advantages and ignores their magnitude. This property **alleviates the need for
   normalizing returns or advantages**"* and *"We find the scaling between the three objective terms
   to be highly robust in practice **as they are all measured in nats**."* The RMS sentence is
   explicitly about *"a single dynamics transformer with multiple modalities and output heads"* —
   Phase 1/2, not Phase 3. `PAPER_DEVIATIONS` §5.6 rates this DELIBERATE/LOW; I'd keep the rating but
   note the paper gives a *reason* it is unnecessary, so the burden is on us to show it helps.

### 1.2.6 twohot / symexp — and a genuine split in the field

| | Bins | Range | Interpolation space | Decode |
|---|---|---|---|---|
| **ahriuwu** | 255 | **±3** (symlog) | symlog | `symexp(Σp·b)` |
| NM512 | 255 | ±20 (symlog) | symlog | `symexp(Σp·b)` |
| DreamerV3 paper | 255 | ±20 (symlog) | symlog | `symexp(Σp·b)` |
| **danijar HEAD** | 255 | **±4.85e8 raw** | **raw** | `Σp·b` (raw) |

HARD, all four rows. Ours: `heads.py:107-109` and `:558-560`,
`torch.linspace(bucket_low, bucket_high, num_buckets)` with `low=-3, high=3`; targets symlogged at
`train_agent_finetune.py:737` and `train_imagination.py:385`. NM512: `tools.py:456-464`,
`torch.linspace(-20, 20, steps=255)` with `transfwd=symlog`. danijar: `embodied/jax/heads.py:136-139`,
`bins = symexp(linspace(-20, 0, …))` mirrored — i.e. **symexp-spaced bins in raw reward space**,
with `TwoHot(..., squash=None)` so the interpolation weights and the expectation are computed in raw
space, not symlog space.

**So the official JAX repo at HEAD does not implement the formulation its own paper describes.** That
is not a bug (the two are close, and the code is what produced the numbers), but it is a trap:
*"port the code, not the paper"* — REPORTED, and I verified the bin construction myself.

**Our ±3 is right in kind and wrong in size, in the opposite direction from what one might guess.**
`WIRING_AUDIT` §1.6 measured 99.79% of all 3,554,768 reward targets landing in **one** bucket, and
only 34 of 254 intervals ever occupied — because per-frame `gold_scale=1e-3` rewards are ~0.001 while
the bucket width is 0.0236. The DreamerV3-lineage answer (±20) would make this **6.7× worse**. The
fix is a much narrower reward range, set from the empirical distribution, and a **separate** range
for the value head, where ±3 is roughly correct. `DESIGN_DECISIONS` §2 already says this; the
external comparison confirms there is no precedent for sharing one range between the two heads —
danijar and NM512 both give `rewhead` and `value` independent head configs.

## 1.3 Training loop

| | ahriuwu | Paper | danijar dv3 | NM512 | DIAMOND | DFoT | edwhu |
|---|---|---|---|---|---|---|---|
| Optimiser | AdamW / **8-bit AdamW** | *unspecified* | **LaProp + AGC** | Adam | AdamW | AdamW | Adam |
| LR | 3e-4 | *unspecified* | 4e-5 | 1e-4 / 3e-5 | 1e-4 | — | 1e-3 |
| Betas | (0.9, 0.999) | *unspecified* | (0.9, 0.999) | — | — | (0.9, 0.99) | β₂=0.9 |
| Warmup | 3000 (dyn) / 2000 (BC) | *unspecified* | 1000 linear | **none** | 100 linear | 5000 | none |
| Schedule after warmup | **flat forever** (WSD, `decay_steps=0`) | *unspecified* | constant | none | constant | constant | constant |
| Grad clip | global norm 1.0 | *unspecified* | AGC 0.3 (per-tensor) | 1000 / 100 | 1.0 | — | **none** |
| Weight decay | configurable | *unspecified* | 0.0 (off) | 0.0 (off) | 1e-2 | 1e-3 | — |
| Precision | **bf16** autocast | *unspecified* | **bf16** | fp16 AMP, **off by default** | **fp32** | fp16-mixed | fp32 |
| Grad checkpointing | **yes** (Phase 1) | *unspecified* | **none**, XLA remat disabled | none | none | — | — |
| EMA of weights | **none** | *unspecified* | slow critic only | slow critic (Polyak 0.02) | **none** | — | — |
| Multi-GPU | DDP (2× validated) | FSDP, 256–1024 TPU | sharded | **none** | DDP (2× validated) | — | single |
| Batch × seq | 2×128 / 1×256, accum 8/16 | T₁=64, T₂=256, C=192 | 16×64 | 16×64 | 32×6 | — | — |

HARD (ours): `utils/training.py:268-283` (optimiser), `:196` (`--decay-steps` default 0),
`:286-300` (WSD), `dyn_train_args.sh:21-22, 31-34` (LR/warmup/shapes),
`train_dynamics.py:932, 1051` (clip), `:1564-1565` (bf16 + GradScaler-only-for-fp16),
`dynamics.py:812-816` (grad checkpointing). HARD (paper): a full-text regex sweep of the re-extracted
32-page PDF returns **zero** hits for `learning rate`, `cosine`, `optimizer`, `warmup`,
`weight decay`, `hyperparam`, `gradient clip`; the only `Adam` hits are bibliography author names.

Three observations:

1. **The paper specifies none of this, and the field does not converge.** The optimiser column alone
   spans LaProp+AGC, Adam, AdamW and 8-bit AdamW. Anything in our repo labelled "paper-faithful" in
   this table is mislabelled — `PAPER_DEVIATIONS` §7 already catalogues four such labels in
   `utils/training.py` and `DREAMERV4_AUDIT.md`. **Those strings are still in the source.** Fixing
   them is a comment-only change and prevents the next reader re-deriving a wrong constant from them.
2. **We are the only repo here using 8-bit Adam** (`bitsandbytes AdamW8bit`, default `use_8bit=True`,
   `utils/training.py:277-280`). Justified by our VRAM budget, but it is an unvalidated axis: no
   reference has run this recipe, and optimiser-state quantisation interacts with the very small
   gradients a frozen-backbone Phase 2 produces. INFERENCE: worth one A/B at some point, low priority.
3. **Nobody uses an EMA of the world-model weights.** REPORTED, and consistent across DIAMOND, IRIS,
   Diffusion Forcing and both DreamerV3 repos (which have a slow *critic* but no model EMA). Our
   absence of one is normal, not a gap. If dream rollouts prove unstable, it is the cheapest untried
   knob in the field.

**One thing we do that is better than most:** REPORTED — `danijar/dreamerv3` has **no gradient
checkpointing at all**, and explicitly disables XLA's own remat pass
(`--xla_disable_hlo_passes=rematerialization`, `internal.py:64, 82`); NM512, DIAMOND and IRIS have
none either. We have it, it is real on the Phase-1 path (`dynamics.py:812-816`, guarded on
`self.training`), and `WIRING_AUDIT` §1.5's finding that it was inert in Phase 2 has been fixed —
HARD, `train_agent_finetune.py:294` now enables it conditionally on `--unfreeze-backbone` and
`:459-461` records why the old `.eval()` made it a no-op.

## 1.4 What we do that nobody else does

| Thing | Where | Verdict |
|---|---|---|
| **Sticky movement gate** (Bernoulli mixture: repeat-previous-bin vs fresh categorical) | `heads.py:242-256` | **Justified by domain.** No reference has an action space where ~90% of frames are "the previous order is still executing". No precedent to copy — and no precedent to check against, which is why it is still untested against an ungated baseline. It also hard-blocks Phase 3 (`heads.py:457-463` raises in `log_prob`). |
| **`joint_noop`** — one categorical over bins²+1 with an explicit NO_OP class | `heads.py:206-217` | **Justified**, and the better of the two: it needs no previous action, so PMPO's `log_prob` works. This is the one that should win. |
| **`StateHead`** — aux supervised HP/level/visibility regression | `heads.py:16-45` | **Suspect, per two prior reviews.** An aux target cannot exceed what the frozen latents contain. Nobody else has one because nobody else has replay labels; but nobody else needs one either. |
| **`use_game_time`** — bucketed game-clock embedding with dropout | `dynamics.py:487-489` | Harmless; OFF in the production checkpoint, so currently inert. |
| **`pixel_hud_loss`** — HUD-masked pixel loss through the decoder | dynamics lineage | Domain-specific, not in the production checkpoint. |
| **8-bit AdamW** | `utils/training.py:277-280` | Forced by VRAM. Unvalidated axis (§1.3). |
| **Live deployment path** (HID gadget, preflight, provenance stamping) | `scripts/play_live.py`, `docs/archive/DEMO_RUNBOOK.md` | **Genuinely ahead of the field.** No reference repo has a deployment story at all. See §1.8. |

The first two exist because League's action space is *event-driven at 20 fps* — a structural property
of the domain that Minecraft, Atari and DMC do not have. INFERENCE: these are the right kind of
deviation, and the fact that no reference implementation has anything analogous is evidence that the
problem is real, not that our solution is wrong.

## 1.5 What everybody else does that we don't

Ordered by how much I think it matters.

1. **Shortcut forcing** (five references, unanimous algebra). Ours is off, our dormant path deviates,
   and `step_embed` rows for d>1 are consequently untrained (`PAPER_DEVIATIONS` §2.12). The paper's
   own Table 2 ablation cascade (HARD, `scratchpad/d4.txt`) quantifies what we are giving up:
   "Diffusion Forcing Transformer" runs at **0.8 fps**; "+ Fewer sampling steps (K=4)" gets 9.1 fps
   but the third column degrades 306 → 875; "+ Shortcut model" restores it to 329 at 9.1 fps. In
   other words the shortcut objective is exactly what buys K=4 without a quality collapse. This is
   Phase-3 cost, not live-inference cost (live play never dreams) — but at H=8 and K=64 we are paying
   **16× the paper's forward passes per imagined frame**.
2. **Conditioning dropout on the action channel, as a first-class feature.** REPORTED — DFoT has
   `external_cond_dropout` built into the backbone (`backbones/base_backbone.py:41`) with an
   `external_cond_mask`. We have `--action-dropout` (`train_agent_finetune.py:178`) but it defaults to
   **0.0** and every launcher sets 0.15 by hand. Given that `MOVEMENT_HEAD_BLIND` traced the whole
   copy-shortcut pathology to the model reading its own held action, a reference implementation
   treating conditioning dropout as architecture rather than a training flag is a direct
   endorsement of the fix that document lands on.
3. **A separation between context length and batch length.** See §1.6-B — this is a real defect.
4. **Agent tokens inside the transformer** (§1.2.3, §1.6-A).
5. **Real tests.** `lucidrains/dreamer4` has 61 test functions / 202 asserts (REPORTED);
   `edwhu` has the bit-equality firewall test (HARD). We have 5 test files / 629 lines. §1.8.
6. **Sampling-time tricks for autoregressive drift.** REPORTED, DIAMOND: train at σ_max=20 but
   *sample* from σ_max=5.0 (a train/sample config gap); SEPARATELY, the author notes in issue #40
   that "starting from a lower variance noise helps to mitigate the autoregressive drift" — which is
   about UNSCALED INITIAL NOISE, not about that config gap. This sentence originally welded the two
   into one claim; corrected 2026-09-02. Also: byte-quantise every generated frame before
   it re-enters the conditioning buffer (`denoiser.py:83`); and overwrite ground-truth conditioning
   frames with the model's own one-step output during *training* (`denoiser.py:119`). We already do a
   version of the first (`tau_ctx=0.9` context noising), nothing like the third. Given dreams hold to
   h≈10 and then ghost, the training-time autoregressive rollout is the most directly relevant idea
   in this document that we have not tried.

## 1.6 Three findings this pass turned up

These are not in `WIRING_AUDIT`, `MOVEMENT_HEAD_BLIND`, `DESIGN_DECISIONS` or `PAPER_DEVIATIONS`.
Each is HARD in the code and INFERENCE in the consequence.

### A. Agent tokens read only the final layer — and that is a candidate answer to the open question

Stated in full at §1.2.3. HARD: `dynamics.py:719-748` vs the paper's "interleave", vs
`edwhu/models.py:680`, `lucidrains/dreamer4.py:7919`, `vijayabhaskar/dynamic_model.py:186`.

**The cheap test.** `MOVEMENT_HEAD_BLIND`'s CORRECTION section already built the nested-probe
machinery: probes initialised at the blind table, scored on the same rows and folds, so only real
gain registers. Everything perceptual recovered 2–5% of what the cheating oracle recovered — but
every probe read either the raw latents or the *final* agent token. **Re-run the same probe against
the dynamics' intermediate spatial tokens** — e.g. layer 6, 12, 18 of `_run_blocks`. It needs one
forward hook and no training.

- If an intermediate layer carries the signal and layer 18 does not, the fix is architectural and
  cheap (deeper agent blocks, or cross-attention into several layers, or moving the agent token into
  the sequence) and the perception verdict is wrong.
- If no layer carries it, the perception verdict stands and this hypothesis is closed for good.

Either way it is hours, and it discriminates between "cheap architectural fix" and "tokenizer
retrain" — the exact fork that document says it cannot resolve.

### B. There is no context length. Batch length *is* the context, which the paper explicitly warns against

HARD, paper (`scratchpad/dreamer4_text.txt`; §3.4 Training Recipe, not §3.2 — verified 2026-09-02, the quote is present and the file `d4.txt` does not exist):

> *"The batch lengths need to be **longer than the context length** of the model to prevent the
> transformer from overfitting to always seeing a start frame at the beginning of its context,
> enabling length generalization to arbitrary generation lengths."*

The paper's Minecraft config: `Nz=256`, **context length C=192**, batch lengths **T₁=64, T₂=256**.

HARD, ours: there is no windowing anywhere. `layers.py:263-264` is a plain unbounded causal mask
(`q_idx >= kv_idx`), and a grep for `window|sliding|context_len|context_length` across
`src/ahriuwu/models/` returns **zero hits**. `max_seq_len` defaults to 256 (`dynamics.py:365`) and
`create_dynamics` never passes it (`dynamics.py:914-995`), so it stays 256 while
`dyn_train_args.sh:31` trains at seq 128 and 256. **Every frame's context therefore always includes
frame 0 of the window.**

`PAPER_DEVIATIONS` §2.5 records this as "ACCIDENTAL, impact MED". INFERENCE: **MED is too low.** The
paper states the failure mode and its consequence — loss of length generalisation — in the same
sentence as the requirement, and length generalisation is precisely what a long autoregressive
rollout needs. Our dreams hold entity coherence to h≈10 and then ghost, and we currently attribute
that to model size and training budget (§2.6, FORCED/HIGH). This is a second, cheaper candidate
explanation that costs one attention-mask change to test: add a sliding causal window of C frames
with C < T, retrain briefly, and measure rollout PSNR past h=10 against the current lineage.

I have **not** verified that the reference implementations do implement a windowed context —
Appendix A.

### C. `agent_temporal_pos` rows past `seq_len` are at random init, and Phase 3 indexes straight into them

HARD:

- `dynamics.py:526-528` — `self.agent_temporal_pos = nn.Parameter(torch.randn(1, max_seq_len, model_dim) * 0.02)`,
  a **learned absolute** position embedding of length 256, while the rest of the model uses RoPE.
- `dynamics.py:735` — consumed as `agent_tokens + self.agent_temporal_pos[:, :T, :]`, always from index 0.
- `train_agent_finetune.py:304` — it *is* trainable in Phase 2 (`AGENT_PARAM_PREFIXES`).
- `launch_bc_1060.sh:25` — but BC trains at `--seq-len 16`. **So rows 16..255 never receive a
  gradient and remain at `randn * 0.02`.**
- `train_imagination.py:301` — imagination grows the window every step: `Tw = z_window.shape[1]`,
  starting at the context length and adding one dreamed frame per horizon step. With ctx=16 and
  `--horizon 8` (`train_imagination.py:103`), T runs 16 → 24.
- `train_imagination.py:311` — and it reads `h_t = agent_out[:, -1:, :]`, the **last** position, at
  every step.

So from imagination step 1 onward, the agent token is being positioned by an untrained random vector,
and read at a position BC never supervised. `agent_infer.py:278-282` already documents half of this
for *live* inference (*"at seq_len 16 it only ever supervises agent-token positions 0..14. Position
15 (=-1) receives BC gradient NEVER"*) and offers `read_pos=-2` as the workaround — but nothing
connects it to Phase 3, and the position-embedding half is not recorded anywhere.

This compounds `WIRING_AUDIT` §2.1 (BC never trains MTP offset 0; Phase 3 reads only offset 0) rather
than duplicating it: §2.1 is about the *head*, this is about the *token* feeding it. Both must be
fixed before a Phase-3 number means anything.

**Cheapest fixes**, in order: (i) train BC at a seq_len ≥ ctx + horizon; (ii) drop the learned
absolute table in favour of the RoPE the rest of the model already uses — REPORTED,
`vijayabhaskar/dynamic_model.py:83-84` extends spatial RoPE to cover agent positions rather than
adding a separate learned table; (iii) at minimum, assert at Phase-3 startup that
`ctx + horizon <= trained_seq_len` and fail loudly. (iii) is three lines and is the
`WIRING_AUDIT` §4.10 convention ("a test that every parser `dest` is read") generalised to shapes.

## 1.7 Is `PAPER_DEVIATIONS.md` trustworthy?

The brief said to check it critically rather than trust it. I re-extracted the paper independently
(pypdf, 32 pages, 80,956 chars → `scratchpad/d4.txt`) and checked eleven claims against both the
paper text and the code.

**Verdict: it is the most reliable document in this repo, and every claim I checked held.** Verified
HARD:

| § | Claim | Result |
|---|---|---|
| 2.4 | τ/step injected additively to all tokens **and** as an appended token | ✅ `dynamics.py:783-784` + `:801` |
| 2.7 | Paper specifies no LR, optimiser, schedule, warmup, weight decay | ✅ zero regex hits across the full re-extracted text |
| 2.13 | Paper: *"When training unlabeled videos, only the learned embedding is used"* | ✅ verbatim, §3.2 |
| 2.14 | RMS normalisation disabled when shortcut is on | ✅ `train_dynamics.py:1568` |
| 3.3 | Paper Eq 9 sums n=0..L for **both** actions and rewards | ✅ verbatim |
| 3.6 | `--ability-pos-weight` default 5.0, every production launcher passes 1.0 | ✅ `train_agent_finetune.py:188` vs `ops/bc5080_*_watchdog.sh` |
| 4.1 | Paper Phase 2 finetunes the whole world model | ✅ *"Finetune world model with task inputs"*, Alg. 1 |
| 4.2 | Paper: *"we continue to apply the video prediction loss"* | ✅ verbatim, §3.3 |
| 4.7 | Agent-token firewall is not a deviation | ✅ for the firewall — but see §1.2.3, it is incomplete |
| 5.7 | Gated policy heads hard-raise in `log_prob` | ✅ `heads.py:457-463` |
| 7 | Four "paper-faithful" labels in the source are unsupported | ✅ still present in `utils/training.py` |

**Two corrections to make:**

1. **§1.1 and §1.2 are STALE.** Both describe tube masking and the mask-ratio curriculum as live
   deviations rated HIGH. They were fixed in commit `fd7632f` (2026-08-20); the document's last
   revision is 2026-08-13. HARD: `transformer_tokenizer.py:719` now defaults `tube=False` with a
   docstring citing §3.1. **The nuance that keeps them from being simply wrong** is that the deployed
   v7 checkpoint was trained under the old regime — so the summary table describes the *checkpoint*
   accurately and the *code* inaccurately. Worth splitting those two columns, since every other row
   describes code.
2. **§4.7 is right about the firewall and silent on the interleaving** (§1.2.3). Not an error, but
   the row currently reads as "agent tokens: no deviation", which is not what the evidence supports.

**One rating I would raise:** §2.5 (context/batch length) from MED to HIGH — see §1.6-B.

## 1.8 Engineering practice

Applying the same criteria used in Part 2 to ourselves, honestly.

| | ahriuwu | danijar dv3 | NM512 | DIAMOND | IRIS | lucidrains d4 | edwhu |
|---|---|---|---|---|---|---|---|
| Tests | 5 files / 629 lines, regression-shaped | 6 modules, **55 of 196 fail, 2 uncollectable** | **none, ever** | **none** | **none** | **61 fns / 202 asserts** | firewall + shape tests |
| CI | **none** | **none, ever** | **none** | **none** | **none** | none | none |
| Config as code | shell arg files, single source of truth | YAML + typed `elements.Config`, **rejects unknown keys** | YAML + argparse, **silently accepts unknown keys** | Hydra | Hydra | — | — |
| Checkpoint self-describing | **args + resolved `model_config` + git commit/branch/dirty** | params + counters | state dicts only | — | — | — | — |
| Atomic checkpoint write | **yes** (tmp + `os.replace`) | — | **no** (`torch.save` to `latest.pt`) | — | — | — | — |
| Checkpoint rotation | yes | every 900 s | **single `latest.pt`, overwritten** | — | — | — | — |
| Dep pinning | **lower bounds only, no lockfile** | `requirements.txt` pinned | pinned | pinned | pinned | — | — |
| Deployment provenance | **commit + ckpt shas + resolved geometry in `meta.json`; preflight refuses unidentifiable code** | n/a | n/a | n/a | n/a | n/a | n/a |
| Dead config knobs | **1** (`--features-dir`, self-labelled) | ≥7, incl. a dead 15-flag XLA block | ≥6, incl. **4 silent typos that change training** | 2 | 1 | many (library posture) | 0 |

HARD on the ahriuwu column throughout: `utils/training.py:341-404` (checkpoint contents, `_git_info`,
atomic tmp+`os.replace` at `:399-402` with the 2026-06-03 corruption it was written for);
`pyproject.toml` (lower bounds only); `docs/archive/DEMO_RUNBOOK.md:111, 238, 254` (provenance gates);
and a repo-wide scan of every `add_argument` dest against its readers, which found exactly one
never-read knob, `train_dynamics.py:494-497`, whose own help string says *"Legacy OCR features dir
(unused by the replay action path)"*.

**Where we are clearly ahead:**

- **Checkpoint discipline.** We stamp the git commit, branch and dirty flag into every checkpoint
  (`utils/training.py:315-340`) alongside the factory-*resolved* model config, with the rationale
  recorded: *"Architecture is defined by code, not just config — two checkpoints with identical
  model_config can be mutually incompatible if the code changed."* REPORTED: NM512 saves a single
  overwritten `latest.pt` with no config snapshot and recovers the step counter by **parsing replay
  filenames**; danijar has no checkpoint format versioning. Nobody else does provenance.
- **Deployment.** No reference repo has one. Ours refuses `--inject hid` without a resolvable commit.
- **Comment style.** Ours is unusually good at the WHY/bug-recording kind that Part 2 treats as a
  positive signal. `dataset.py:231-235` is the archetype: *"`__iter__` used to draw from the global
  torch RNG, which a fresh process re-seeds identically, so every restart replayed the SAME batch
  order from the top of the epoch … measured 2026-08-15 as ~2/3 of epoch 1 unseen across three
  restarts."* Argparse help strings carry measured numbers rather than descriptions
  (`train_agent_finetune.py:139-144, 155-177`).
- **Adversarial self-audit.** `VIBE_AUDIT_2026-08-12.md`, `WIRING_AUDIT_2026-08-20.md` and the two
  correction passes in `MOVEMENT_HEAD_BLIND_2026-08-26.md` have no analogue in any repo surveyed.

**Where we are behind:**

- **Tests.** Five files. Every one is a real regression test with recorded provenance
  (`test_diffusion_forcing_schedule.py:1-17` documents the position-dependent-τ bug it exists to
  prevent; `test_rollout_equivalence.py:1-21` is the test a docstring had been *claiming* existed) —
  the quality is right, the coverage is not. There is no test on the RMS normaliser, the twohot
  round-trip, MTP slicing, or the movement binning, all of which `WIRING_AUDIT` §5 verified by hand
  and none of which is protected against regression. **`edwhu`'s firewall test is the model to
  copy** (HARD, `models.py:1071-1107`): it asserts an *architectural invariant* by bit-equality, in
  ~30 lines. Our equivalent — "agent tokens cannot influence `z_0_pred`" — is currently guaranteed by
  construction and would be silently lost the day anyone tries §1.6-A's fix.
- **No CI.** Every repo here shares this, so it is not a differentiator, but our five tests would
  cost minutes to run on push.
- **No dependency pinning.** `pyproject.toml` has lower bounds and no lockfile, on a project that has
  already been bitten by version-specific behaviour (`torch.compile` on Blackwell sm_120,
  `docs/archive/VAST.md` §4). danijar's issue #175 is the cautionary tale: an `ale-py` minor-version bump
  silently broke reproduction across 26 games and took an external contributor a full sweep to find.
- **Test-in-`scripts/`.** `scripts/test_agent_infer.py` and `scripts/test_kv_cache.py` are tests
  living outside `tests/`, so `pytest` never collects them.

---

# Part 2 — how much should we trust each external repo?

## 2.1 Method

The criteria are the brief's, and they line up with the house methodology already in
`VIBE_AUDIT_2026-08-12.md` §A: AI-written code fails by *plausibility and duplication*, not by
crashing, so the highest-yield question is not "does this look good" but **"what is the artifact
proving it was run?"**

Weighted roughly in this order:

1. **Evidence of execution** — released checkpoints, committed training curves that changed over
   time, benchmark scorecards, third-party reproductions, hardware-specific workarounds.
2. **Commit-history shape** — incremental bugfix commits vs a single finished drop. *Weak on its
   own*: see §2.3.
3. **Comment style** — WHY / recorded-bug vs generic restatement. Docstring saturation is a
   *negative* signal for human authorship.
4. **Tests** — real invariants vs none vs broken.
5. **README claims vs demonstrated evidence** — especially whether the repo admits to losing.
6. **Dead code and never-read knobs.**
7. **Whether maintainers answer issues with real understanding of their own code.**

"Written with AI assistance" is not a defect. The question is only whether it was **validated**.

## 2.2 Ratings

### `danijar/dreamerv3` — **LIKELY HUMAN-AUTHORED & VALIDATED** (high confidence) · *but see the caveats*

**Evidence it was run — very strong.** REPORTED: `scores/` holds 28 gzipped JSON scorecards covering
**3,238 individual training runs** with full learning curves, DreamerV3 plus a dozen baseline
families (e.g. `atari57-dreamerv3` = 299 runs / 57 tasks / 3–6 seeds / 149,245 datapoints). Plus
`entrypoint.sh` querying GCP instance metadata, and workaround comments that only exist after
profiling — `nets.py:305-306`: *"Manual implementation of fractionally strided convolution because
the cuDNN implementation used by XLA has bugs and performance issues."*

**Human authorship — confident.** REPORTED: 0.2% docstring coverage (2 of 831 functions), zero
`TODO`/`FIXME`, 3 lines over 79 chars in 10.4k lines, single-letter tensor dims, style identical
across the 2023 initial commit and 2026 HEAD and across four separate personal packages. Every marker
points away from generation.

**Caveats that matter to us:**
- **Commit history is not a development history.** 31 squashed drops, messages like `Update`
  (one is 177 files / +10,432 / −6,009). `git blame` is useless here. INFERENCE: for this repo the
  commit-shape signal is simply unavailable, which is why it is weighted low in §2.1.
- **The scorecards predate the code.** REPORTED: `scores/` was committed in `423291a` (2023-05-06),
  before both rewrites (2024-04, 2024-12). The README's *"tested to reproduce the official results"*
  is not backed by anything in the current tree.
- **Reproduction of the current code is only partial and the gap is open.** REPORTED, issue #175: a
  26-game × 5-seed Atari100k sweep found the current `atari.py` underperforms the 2023 one; the
  maintainer confirmed an `ale-py` regression and pinned `ale_py==0.9.0` (`7949c3c`), but the
  residual gap was never explained and the issue is still open.
- **The test suite is broken.** REPORTED, and the reviewer ran it: 141 passed, **55 failed**, 2
  modules uncollectable — `import zerofun` (renamed to `portal` long ago) and 43 tests calling a
  `Replay.dataset()` that was commented out at `core/replay.py:237-253`. Zero coverage of the agent,
  RSSM, losses, optimizer or twohot.
- **Confirmed live defects.** HARD (I verified two myself): `configs.yaml:73` sets
  `jax.platform: cuda` while `embodied/jax/internal.py:54` gates on `platform == 'gpu'`, so the
  entire 15-flag XLA GPU performance block never applies; and the twohot bin construction at
  `embodied/jax/heads.py:136-139` is symexp-spaced in raw space, not the paper's formulation.
  REPORTED for the rest: `run.from_checkpoint` crashes on a `from_checkpoint_regex` key that exists
  in no config; `agent.opt.momentum` is accepted and never read; `mets.update(mets)` at
  `agent.py:260` is a self-update no-op that discards the report metrics; prioritized replay is
  half-wired (the producing code is commented out at `agent.py:151-152`).

**Verdict for us: excellent as a hyperparameter source and a reference for the *algorithm*; not
something to depend on.** Fork-and-own, and remember it is a different model family (§1.1).

### `NM512/dreamerv3-torch` — **LIKELY HUMAN-AUTHORED & VALIDATED** (high confidence) · **but ARCHIVED and algorithmically stale**

**Lead finding (REPORTED, and checkable): the repo is archived and deprecated by its own author.**
GitHub API `"archived": true`; the README carries a maintainer notice (commit `6253f98`, 2026-02-21)
saying it *"was implemented prior to major updates to DreamerV3 and does not reflect those changes,
which accounts for several GitHub Issues in this repository"*, redirecting to
[`NM512/r2dreamer`](https://github.com/NM512/r2dreamer). Root cause, per the author in issue #64: it
implements DreamerV3 **arXiv v1** (Jan 2023); the paper was revised April 2024 with a unified
REINFORCE actor loss this code does not have.

**Evidence it was run — strong, and unusually honest.** REPORTED: DMC Vision (20 tasks × 1M),
Atari100k (26 games × 400k), DMC Proprio and Crafter curves, each overlaying this repo against the
JAX reference with variance bands. **It loses on a meaningful minority of tasks** — Hero, Qbert,
Kangaroo, Seaquest, Private Eye, Road Runner, Demon Attack, Crazy Climber, Gopher, Chopper Command,
Finger Spin. Fabricated results are uniformly flattering. The curve PNGs were regenerated **eight
separate times** across 2023–2024 tracking bugfixes.

**Human authorship — confident.** REPORTED: consistent characteristic misspellings propagated into
the API (`stickey` in `configs.yaml:139` *and* its consumer `dreamer.py:165`; `gumble`, `necesarry`,
`varibs`), commented-out scaffolding left in place, one docstring in the whole repo — and it is
*wrong* (`models.py:12` says "running mean and std" over a percentile implementation).

**Maintainer understanding — demonstrated.** REPORTED, issue #16: an external contributor found an
off-by-one in the return computation; the author confirmed it, explained why the impact was small,
conceded a second bug in `trunc_normal`, **volunteered that he had been training with 1 env instead
of the paper's 4**, fixed it, and re-benchmarked.

**Real defects.** HARD — I verified the config typos myself in the clone. `configs.yaml:116` and
`:150` write `step:` where every other section writes `steps:` (lines 10/97/106/133/173), and `:127`
and `:165` write `value:` where the defaults define `critic:` (`:51`). `grep` for `config.step` /
`config.value` finds no reader. Consequence: **Minecraft trains 100× short of its intended `1e8`
steps, and the Crafter/Minecraft critic silently stays at `layers: 2` while every other head gets 5.**
The enabler is `recursive_update` at `dreamer.py:355` accepting any key — the exact opposite of
danijar's `elements.Config`, which raises on unknown keys. REPORTED for the rest: no tests have ever
existed (verified against the full 125-commit history), `DiscDist.log_prob_target` calls
`super().logits` on a class with no base class, `nadam` returns `NotImplemented(...)`, and the
`"huber"` and `"onehot_gumble"` branches both raise on use.

**Verdict for us: the clearest PyTorch statement of twohot/symlog, KL balancing and percentile return
normalisation that exists — read it for those three things. Do not build on it: archived, stale, zero
tests, and four silent config typos that change training.**

### `edwhu/dreamer4-jax` — **LIKELY HUMAN-AUTHORED & VALIDATED** (high confidence, small scale)

**The most useful repo in this survey for us**, despite being the smallest.

**Evidence it was run.** REPORTED: 53 commits with 7 explicitly bugfix-shaped (`fix rew pred bug,
offby one`; `debugged dynamics model and fixed sampling bugs`), real curves in `docs/`, and
`docs/NOTES.md` is an honest checklist **with unchecked boxes**. The maintainer answers with
operational specifics (*"We used H100s for ~24 hours… increase LR to 1e-3 and set adam beta2 to
0.9"*), and a third party independently reproduced 34 PSNR on 4×A100.

**Tests — the best in the set, and the reason to read it.** HARD, verified myself:
`dreamer/models.py:1071-1107` asserts the paper's causal firewall by exact bit-equality
(`atol=0, rtol=0`). That is an architectural invariant test, and it is the single most transferable
engineering idea in this document (§1.8).

**Caveats.** REPORTED: toy scale only (32×32 bouncing squares — it has *not* been shown to work at
video scale); fp32, single-GPU, constant LR, no gradient clipping; loss weights are fixed scalars
(all 1.0) rather than the paper's RMS normalisation; **no licence file**, which is a real problem if
we copy code rather than ideas; development stopped 2025-11-28, with the author pointing to
open-dreamer as successor.

**Verdict: safe to learn from, and the first place to look for DreamerV4 structure. Check the licence
before copying more than a pattern.**

### `lucidrains/dreamer4` — **LIKELY HUMAN-AUTHORED, PARTIALLY VALIDATED (toy scale only)**

**Commit history is the strongest in the survey.** REPORTED: 406 commits over 11 months with
messages that can only come from running the thing — *"fix an issue where tokenizer was not brought
back to training mode after the eval phase, and MAE may have been turned off erroneously"*,
*"correct an issue with reward prediction head off the agent embed lacking action conditioning"*.
Outside contributors have merged fixes for an off-by-one and for PMPO loss normalisation.

**Tests — real.** REPORTED: 61 test functions / 202 asserts / 3,085 lines. No CI; run locally.

**Comments cite the paper by section.** HARD, verified: `dreamer4.py:690` — *"the authors mentioned
the need for loss normalization in the dynamics transformer"*; `:711` — *"get the rms value - as
mentioned at the end of section 3 in the paper"*.

**The weakness is scope, and it is stated plainly rather than hidden.** REPORTED: **no demonstrated
results at all** — no curves, no checkpoints, no benchmark numbers. Validation is toy-scale
(cartpole, 4×4 snake, moving MNIST, HalfCheetah), which the commit messages confirm actually
happened. Plus an 8,496-line single file and heavy accretion of speculative extras beyond the paper
(evolutionary policy optimization, LAPO, BYOL, TEM, latent-AR); the `WorldModelLosses` namedtuple has
17 fields. INFERENCE: the extras are the main hazard — it is easy to copy a block and not notice it
depends on a non-paper option.

**Verdict: safe to read for individual components (`LossNormalizer`, the MTP heads, the shortcut
loss), all of which are corroborated elsewhere. Do not assume any of it has been run at video scale.
Prefer it as a second opinion, not a primary source.**

### `vijayabhaskar-ev/dreamer_v4` — **MIXED AUTHORSHIP, UNUSUALLY WELL VALIDATED** (for its scope)

**This is the instructive case for the whole exercise**, because the surface signals and the
underlying reality point in opposite directions.

Surface signals that look bad: heavy LLM-flavoured README prose, and literally an *"em-dash pass"*
commit. **Under it is the best-validated work in the entire survey.** REPORTED:

- Commit history is unmistakable real-hardware work: *"Fix TPU/XLA host OOM: reduce compile count,
  CPU-offload model stats"*, *"Work around torch_xla 2.5 MpDeviceLoader thread leak with timed
  auto-exit"*, *"Fix XLA compile creep: on-device curriculum, hang watchdog, leak tracers"*, then a
  full *"Remove TPU/XLA code; keep CUDA path"* migration.
- `dynamics/trainer.py:1124-1136` documents a bug found empirically on a named checkpoint:
  *"(Was `agent_out.detach()`, which severed the only gradient path to agent_embedding — freezing it
  at its randn\*0.02 init, so the heads decoded from a fixed **untrained** random query. Confirmed
  empirically: `exp_avg_sq==0` for agent_embedding in the e040 checkpoint.)"*
- **The README supersedes its own earlier public claims.** It flags that a previous version — and an
  accompanying Reddit/X post — reported a single-run n=50 result that *did not survive replication*,
  then reports 6 runs × 500 seeded episodes with CIs and sign tests, states that **3 of 6 runs
  individually look like nothing**, and commits 25 directories of raw per-episode data.

**Caveats.** Scope is **one DMC task** (`ball_in_cup_catch`), not Minecraft. "Tests" are `assert`-free
custom harnesses with `sys.exit(1)`. Git history starts with a 28-file drop, so pre-history is
invisible. Almost no external engagement (3 issues).

**Verdict: the prose is AI-assisted; the science is not fabricated. Safe to learn from — and the only
repo here whose Phase-3/PMPO path has published closed-loop numbers with error bars. Note that its
scope does not include video-scale anything.**

### `nicklashansen/dreamer4` — **LIKELY HUMAN-AUTHORED & VALIDATED** · *world model only*

REPORTED: commit shape is a single 30-file drop plus README commits, which would normally be a red
flag — but the external evidence is decisive. Committed training curves
(`assets/{tokenizer,dynamics}/*.png`), **real HuggingFace checkpoints** (tokenizer 90k steps,
dynamics 40k steps, 128×128, 30 tasks), a real 3.6M-frame dataset, and concrete hardware claims
("~24h on 8× RTX 3090"). An automated audit of all 79 argparse flags across both training scripts
found **zero** never-read knobs.

**Scope honesty is exemplary and directly relevant to us.** The agent-token path is **dead code** —
REPORTED, `agent_tokens=None` at every call site (`train_dynamics.py:356, 770, 843, 855`;
`interactive.py:143`), no policy/value/reward head exists — and it says so in the README, in issue #5,
and in the maintainer's own reply: *"Correct! Perhaps it would be better to remove this logic
altogether."* **Dead code that is known, documented and acknowledged is a completely different
artifact from dead code that is hidden** — that distinction is worth carrying into our own audits.

**Weakness: zero tests.**

**Verdict: safe to learn from for the tokenizer and dynamics. It has nothing to say about Phases 2–3.**

### `next-state/open-dreamer` — **MIXED (validated at scale, but opaque and not open-source)**

REPORTED: Minecraft/VPT scale, real-time playable demo, correct shortcut forcing with an EMA
bootstrap-target refinement worth stealing. But **8 commits, all within 3 days** — a pure release
drop with no debugging narrative to inspect — and no agent tokens, policy, MTP or imagination RL.
**The licence is All-Rights-Reserved despite "Open Dreamer" in the name.**

**Verdict: read the shortcut-forcing implementation as a fifth corroborating vote. Do not copy code
— the licence forbids it.**

### `eloialonso/diamond` — **LIKELY HUMAN-AUTHORED & VALIDATED** (high confidence)

REPORTED: NeurIPS 2024 spotlight, pretrained Atari-100k weights on HuggingFace, a separately trained
CSGO world model ("12 days on an RTX 4090"), 27 commits / 3 contributors. Issue #29 is the strongest
single piece of evidence: a user reported OOM, the maintainer **reproduced it**, root-caused it
("we originally trained the dynamics and upsampling models separately, but combined them in the
current training code"), pushed a fix (`a4396eb`), and the reporter confirmed. Issues #36 and #40 are
paragraph-length technically-correct answers including LaTeX for a modified objective.

A nice negative control: PR #45 was an actual AI-slop PR (branch name `Claude/checkout dop 011 c ut
vmu…`, dumping `node_modules/` and an unrelated React app into the repo). **It was closed.** The
maintainers filter.

**Two real defects, both benign in the shipped Atari config, both live on the CSGO branch** —
REPORTED: `trainer.py:366` calls `loss.backward()` with no `/ grad_acc_steps`, so effective LR scales
with accumulation (dead at Atari's `grad_acc_steps: 1`, **8× on CSGO's 8**); and
`diffusion_sampler.py:44-45` passes `sigma` where EDM Alg. 2 wants `sigma_hat` after churn (dead at
`s_churn: 0.0`, live on CSGO's `s_churn: 10.0`). **If you port that sampler, port the fix.**

**Verdict: safe to learn from. Different architecture (pixel-space, no tokenizer), but the
drift-mitigation tricks are the most transferable ideas here.**

### `eloialonso/iris` — **LIKELY HUMAN-AUTHORED & VALIDATED** (high confidence)

REPORTED: ICLR 2023, 12 commits over 2 years (low churn reading as *finished*, not abandoned),
checkpoints released, 28 issues essentially all closed substantively. **Issue #10 is the gold
standard for maintainer honesty:** the author explained a `sampling_weights` feature, then six months
later returned to say *"Sorry for the misunderstanding, this was indeed a bug (reported again
in #18). We removed this feature as it was not used during our experiments"* → commit `ac6be40`. A
maintainer telling you **which parts of the released code the paper's numbers did not depend on** is
rarer and more valuable than a green test suite.

**Caveats: zero tests; GPL-3.0** (unlike DIAMOND's MIT) — relevant if we copy code.

**Verdict: safe to learn from, specifically for the frozen-vs-joint tokenizer question (§1.2.1).**

### `buoyancy99/diffusion-forcing` — **MIXED. Its `main` branch shipped code that could not train.**

This one deserves the most caution of any repo here, and its story is the best argument in this
document for why Part 2 exists.

Human authorship is not in doubt: 47 commits, NeurIPS 2024, checkpoints released, and a README that
is unusually candid about its own negative results (*"we don't recommend this because diffusing
quantity and its derivative together creates some bad optimization landscape"*).

**But `main` is not the paper's code, and it was demonstrably not validated at release.** REPORTED:
the repo ships two branches, `paper` (the RNN version that produced the numbers) and `main` (a
temporal-attention re-implementation). Issue #13 — *"Video UNet not learning?"* — is an external user
spending three days proving the `main`-branch video model did not train, cross-checking against the
`paper` branch and against DFoT. The bug, commit `1762031`: **a missing `return output_dict` in
`DiffusionForcingVideo.training_step`** (`df_video.py:41`). Under Lightning, returning `None` from
`training_step` skips the optimisation step. **The released `main` branch could not train a video
model for roughly six weeks** (2024-07-03 release → 2024-08-16 fix), and it took an outside user to
notice.

Also REPORTED: `use_snr` and `use_cum_snr` in `df_base.yaml:30-31` are never read anywhere; the
stabilization math changed twice after release (`e4d62c4`, `2519217`), so current `main` no longer
matches the released checkpoints' sampling behaviour; and **action conditioning is hard-blocked** in
the video model (`models/unet3d.py:51-52` raises `NotImplementedError`, while `forward` accepts
`external_cond` and silently ignores it).

**Verdict: this is the direct ancestor of our training objective, so read it — but read the `paper`
branch or DFoT, not `main`. `main` is a reference for the math, not a known-good recipe. And take the
general lesson: published + starred + checkpointed ≠ the released training path was ever run.**

### `kwsong0113/diffusion-forcing-transformer` (DFoT) — **LIKELY HUMAN-AUTHORED & VALIDATED** (medium confidence — not deeply audited)

REPORTED: ICML 2025, 708 stars, authored by MIT-lab authors including the Diffusion Forcing author
himself; the DF README points video users here. 11 commits including a real bugfix with issue
linkage (`2afc548` "Fix history guidance not being applied during rollout (#6)"). No tests. Confidence
is *medium* only because it was not audited as deeply as the primaries.

**Verdict: architecturally the closest public analogue to our stack** — frozen VAE, offline
precomputed latents, diffusion forcing, transformer backbone, external conditioning with dropout.
**Read it before the DreamerV3 repos.**

## 2.3 Two signals that turned out to be misleading

Worth recording, because both would have led us wrong:

1. **"A few enormous drops of finished code" is a weak signal on its own.** It fires on
   `danijar/dreamerv3` (31 squashed drops), `nicklashansen/dreamer4` (one 30-file drop) and
   `next-state/open-dreamer` (8 commits in 3 days) — all three of which are backed by released
   weights, committed curves, or 3,238 benchmark runs. It also fires on genuine stubs. **The signal
   that actually discriminates is the presence of a costly artifact** — checkpoints, multi-seed
   curves regenerated over time, hardware-specific workarounds. A drop plus artifacts is a private
   repo made public; a drop without artifacts is unvalidated.
2. **LLM-flavoured prose says nothing about whether the work was done.**
   `vijayabhaskar-ev/dreamer_v4` has an *"em-dash pass"* commit and README prose that trips every
   stylistic detector, and it is the most rigorously measured repo in the survey — six runs, 500
   seeded episodes, confidence intervals, sign tests, and a public retraction of its own earlier
   n=50 claim. Conversely `danijar/dreamerv3` reads as maximally human and has 55 failing tests and
   six confirmed dead knobs. **Style is evidence about authorship. It is not evidence about
   validation, and validation is the thing we actually care about.**

## 2.4 Verdict — what to copy, and what to verify first

**Safe to learn from, and to copy patterns from with normal care:**

| Repo | Copy this |
|---|---|
| `edwhu/dreamer4-jax` | Overall DreamerV4 structure; **the bit-equality firewall test**; shortcut-forcing algebra. *(No licence file — copy ideas, not lines.)* |
| DFoT | Frozen-VAE + offline-latents wiring; conditioning dropout as architecture; history guidance |
| `nicklashansen/dreamer4` | Tokenizer and dynamics training scripts; scope honesty |
| `NM512/dreamerv3-torch` | twohot/symlog, KL balancing, percentile return norm — **as reading material only** |
| `danijar/dreamerv3` | Hyperparameter values; `elements.Config`'s reject-unknown-keys discipline |
| `eloialonso/diamond` | Autoregressive-drift mitigations (train-time self-rollout; sample from lower σ than trained) |
| `eloialonso/iris` | The frozen-vs-joint tokenizer answer *(GPL-3.0 — ideas only)* |
| `lucidrains/dreamer4` | `LossNormalizer`; MTP head layout — **as a second opinion, corroborated elsewhere** |
| `vijayabhaskar-ev/dreamer_v4` | Per-term RMS trackers including the bootstrap loss; its evaluation methodology |

**Do not copy without independent verification:**

| Repo | Why |
|---|---|
| **`buoyancy99/diffusion-forcing` `main` branch** | Shipped a video training path that could not train, for six weeks. Current `main` no longer matches its own checkpoints. Use the `paper` branch or DFoT. |
| **`NM512/dreamerv3-torch` configs** | Four silent typos (`step`/`steps`, `value`/`critic`) that change training; `recursive_update` accepts any key. Archived — no fix will ever land. |
| **`danijar/dreamerv3` test suite and infra glue** | 55 of 196 tests fail; `run.from_checkpoint` crashes; the GPU XLA flag block is dead. The *algorithm* is fine; the scaffolding is not. |
| **`eloialonso/diamond` `csgo` branch** | Two defects that are dead on Atari and live on CSGO: missing `/ grad_acc_steps`, and `sigma` vs `sigma_hat` after churn. |
| **`next-state/open-dreamer`** | All-Rights-Reserved licence. Read only. |

**The rule that falls out of this.** Every one of the eleven repos here has zero or broken CI, and
eight have no tests at all. **Nothing in this field is protected against regression by its own
authors.** So the only durable defence when we take a pattern is to bring a test for the invariant
with it — which is exactly what `edwhu/models.py:1071-1107` does in thirty lines, and what we should
copy first.

---

## Appendix A — what I could not verify

- **I ran nothing external.** Every benchmark number, PSNR, star count and reproduction claim is read,
  not reproduced. The one exception: a delegated reviewer did install and execute
  `danijar/dreamerv3`'s test suite (141 passed / 55 failed / 2 uncollectable).
- **§1.6-B is one-sided.** I established HARD that *we* have no sliding-window context. I did **not**
  verify that the reference implementations implement one. The paper's requirement is quoted
  verbatim and is unambiguous, but "everyone else does it" is not something I checked.
- **Paper text.** Re-extracted with pypdf, so a hyperparameter appendix rendered as an image would
  not appear. The zero-hit result for `learning rate` / `optimizer` / `cosine` / `warmup` is
  consistent across two independent extractions (mine and the existing
  `scratchpad/dreamer4_text.txt`), which is good but not proof.
- **Hafner's TalkRL quote** was verified from a transcript by a delegated reviewer; I did not fetch
  the audio. His X post could not be fetched (x.com returns HTTP 402).
- **DFoT** was assessed from a partial read, not a full audit. Medium confidence only.
- **`james0248/visionary`** (47★, most actively developed DreamerV4-architecture repo, pushed
  2026-08-27) was found late and not inspected at all.
- **Whether `IamCreateAI/Dreamerv4-MC`'s released weights were trained with shortcut forcing** —
  the architecture carries step-size conditioning, which is suggestive, but the objective is not in
  the repo. INFERENCE only.
- **GitHub search undercounts.** Name search returns ~29 `dreamer4` repos; two of the most
  substantial (`next-state/open-dreamer`, `james0248/visionary`) do not have "dreamer4" in the name
  and surfaced only via code search for `"shortcut forcing"`. There are probably more.

## Appendix B — local clones

Cloned during this pass to
`/tmp/claude-1000/-srv-nfs-projects-ahriuwu/8ca13f00-a7ea-478e-9e06-ebe0fc102879/scratchpad/repos/`:
`dreamerv3` (HEAD `e3f0224`), `dreamerv3-torch` (`6ef8646`), `edwhu_dreamer4-jax`,
`lucidrains_dreamer4`, `vijayabhaskar-ev_dreamer_v4`, `nicklashansen_dreamer4`, `open-dreamer`,
`mmbench2`, `IamCreateAI_Dreamerv4-MC`, `RajatDandekar_dreamer4-coinrun`,
`machines-in-motion_dreamer-v4`, `diamond` (`5bcd159`), `iris` (`24326aa`),
`diffusion-forcing` (`475e0bc`), `dfot` (`530f8bf`).

These live in a session scratchpad and will not survive. Re-clone with `--depth 200` if you want to
follow up on any file:line reference in this document.
