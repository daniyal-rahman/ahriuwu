# Dreamer 4 / Dreamer 3 implementations — survey, and what they do about action copying

**Date:** 2026-08-27. Research pass only: no code changed, no training started.

**Why this document exists.** Our Phase-2 BC movement head learned to copy its own
fed-in action history instead of reading pixels
(`docs/MOVEMENT_HEAD_BLIND_2026-08-26.md`). Swapping the fed-in action history moves the
model's command **150.9deg**; swapping the **pixels** moves it **41.0deg** — the action
channel beats vision ~3:1. Dreamer 4's architecture *also* lets the policy attend to
past actions, and the paper reports no defence against copying. So: **why does that hurt
us and apparently not them?** This survey enumerates every implementation that exists,
reads the ones that can be read, and answers that question in §6.

**Evidence labelling.** Every claim is tagged:
- **[HARD]** — the source file or paper text was read; file:line is quoted.
- **[INFER]** — reasoned from a README, paper prose, or repo metadata without reading
  the implementing code.
- **[UNVERIFIED]** — stated plainly where it could not be checked.

---

## 0. Executive summary

1. **There is no official Dreamer 4 release.** `danijar/dreamer4` and `danijar/dreamerv4`
   both 404, and `danijar.com/project/dreamer4/` links only to the paper, a Twitter
   thread and four evaluation videos. **[HARD]**
2. **Nine third-party Dreamer 4 repos have any traction** (>=8 stars). **Only three
   implement a policy at all.** Five are world-model-only (Phase 1); one is
   inference-only.
3. **Every implementation that has a policy feeds it the PREVIOUS action, never the one
   it is predicting.** Four independent codebases converged on a one-step shift.
   Our code has the same one-step separation by a different route (we drop `n=0` from
   the MTP sum). **We are not off-by-one.** **[HARD]** — this kills the most obvious
   candidate explanation, see §6-H0.
4. **Not one of them has any anti-copy machinery.** No action dropout, no null-action
   conditioning, no CFG, no stop-gradient on the action stream, no action-free actor
   branch. Nor does the paper: the word "dropout" appears exactly **once** in
   arXiv 2509.24527 and refers to the tokenizer's MAE patch dropout. **[HARD]**
5. **Nobody reports a blind / copy / no-pixel policy baseline.** Not the Dreamer 4
   paper, not the Dreamer 3 paper, not any of the nine repos. The closest artifacts are
   two *action-shuffle sensitivity* probes (§2.2, §2.4) — and one of them, run on its
   own trained policy, produces numbers consistent with heavy action-copying that the
   authors do not interpret as such. **Our `move_event_ce` vs blind-table bar is a
   stricter diagnostic than anything published in this ecosystem.** **[HARD]**
6. **The paper's own architecture contradicts our reading of Eq 9.** We assumed "Eq 9
   sums MTP from n=0, so h_t must exclude a_t". It doesn't: attention is block-causal so
   that "*all tokens within a time step can attend to each other and to the past*", and
   Figure 2 draws each block as `t d a z̃`. So **h_t sees a_t and Eq 9's n=0 term is a
   degenerate copy term for them too.** **[HARD]**
7. **DreamerV3 is immune to this failure mode for a structural reason: it has no
   behaviour-cloning loss at all.** The actor's only gradient comes from
   advantage-weighted log-probability of its *own imagined samples*. There is no dataset
   action label anywhere for a policy to copy. **[HARD]** Dreamer 4 introduces BC and
   with it this hazard.
8. **The decisive differences are the action representation and what BC is for.**
   Minecraft actions are per-frame binary keys + *camera deltas*; ours is a **held goal
   duplicated bit-for-bit across a whole run**. And Dreamer 4's BC is an
   *initialisation* whose successor phase trains under exactly the self-fed closed loop
   that kills us. Full argument in §6.

---

## 1. Dreamer 4 implementations

Metadata via the authenticated GitHub API on 2026-08-27 (`pushed_at`, not the search
API's `updatedAt`, which tracks star changes). Stars are point-in-time. **[HARD]**

| Repo | Lang | ★ | Last push | Licence | Phases implemented | Trained results |
|---|---|---|---|---|---|---|
| `danijar/dreamer4` (**official**) | — | — | — | — | **DOES NOT EXIST (404)** | — |
| [nicklashansen/dreamer4](https://github.com/nicklashansen/dreamer4) | PyTorch | 386 | 2026-07-09 | MIT | 1 (tokenizer + dynamics); agent token present but **inert** | **yes** — HF ckpts + dataset + curves |
| [next-state/open-dreamer](https://github.com/next-state/open-dreamer) | JAX/Flax | 370 | 2026-08-05 | ⚠️ **all-rights-reserved** | 1 only | yes — full recipe + FVD published |
| [lucidrains/dreamer4](https://github.com/lucidrains/dreamer4) | PyTorch | 213 | 2026-08-25 | MIT | **1+2+3** (library, not experiment) | **no** — no ckpts, no numbers |
| [edwhu/dreamer4-jax](https://github.com/edwhu/dreamer4-jax) | JAX | 102 | 2026-07-24 | **none** | **1+2+3** on a toy dataset | curves only, no ckpts |
| [IamCreateAI/Dreamerv4-MC](https://github.com/IamCreateAI/Dreamerv4-MC) | PyTorch | 61 | 2026-07-13 | **none** | 1, **inference only** | HF weights (430M + 1.7B) |
| [vijayabhaskar-ev/dreamer_v4](https://github.com/vijayabhaskar-ev/dreamer_v4) | PyTorch | 38 | 2026-08-23 | MIT | **1+2+3 + closed-loop env eval** | **yes** — HF ckpts, 30 eval artifacts |
| [RajatDandekar/dreamer4-coinrun](https://github.com/RajatDandekar/dreamer4-coinrun) | Modal harness | 10 | 2026-08-20 | MIT (harness only) | 1 only; **contains no model code** | yes — PSNR 40.41, FVD 32.19 |
| [machines-in-motion/dreamer-v4](https://github.com/machines-in-motion/dreamer-v4) | PyTorch | 8 | 2026-07-08 | GPL-3.0 | 1 only (robotics) | yes — HF ckpts |

**Honest tally: of nine repos, exactly three implement Phase 2 (BC). Three implement
Phase 3. Two of those three were written by individuals on toy or small-scale tasks.
Nobody outside DeepMind has reproduced the Minecraft result.** **[HARD]**

Everything else the GitHub search surfaces is 0–4 stars and is a personal experiment, a
fork of lucidrains, or unrelated: `HKimiwada/Dreamer4`, `skr3178/DreamerV4`,
`Jan-Egil-G/dreamer4` (self-described fork of lucidrains),
`AntitheticalElysium/Dreamer4-Mamba-JEPA`, `vFf0621/Dreamer4-torch`,
`Jerry-111/Dreamer4-with-LeWM`, `rish-av/dreamerv4`, `GonzaloFuentes1/dreamer4`,
`BURAKT33/dreamerv4`, `marcospaulo429/dreamerv4`, `anhtuandev-04/dreamer4`,
`suk063/dreamer4`, `JWK7/Dreamer4`, `4ku/dreamer4`, `Merlin-Richter/dreamerv4-tests`,
`Irayshon/Dreamer4-jax`, `klee972/dreamer4-nnx`, `Ericonaldo/dreamer4-pytorch`,
`laheau/dreamer4-experiments`. **[INFER — metadata only.]**

Two false positives worth naming so nobody re-finds them: **`HeorhiiS/dreamerv4` was last
pushed 2025-04-11, five months *before* the Dreamer 4 paper** — a name collision, not an
implementation. **`softengg-manoj/dreamer4`** appeared in web search and now 404s. **[HARD]**

Packaging / weights **[HARD]**:
- PyPI `dreamer4` v0.19.1 (MIT, Phil Wang) is lucidrains', mirrored on Codeberg.
- HuggingFace: `nicklashansen/dreamer4` (ckpts + a 7,200-trajectory / 3.6M-frame
  dataset), `vijayabhaskarev/dreamer-v4` (~814 MB), `IamCreateAI/Dreamerv4-MC`,
  `Rooholla/dreamer-v4` (GPL-3.0, machines-in-motion).

⚠️ **Licence hygiene in this ecosystem is poor.** Three of the nine ship no licence at
all, one is all-rights-reserved (open-dreamer — its own downstream user,
`RajatDandekar/dreamer4-coinrun`, gitignores the code and ships only a harness rather
than redistribute it), one is GPL-3.0. Relevant if we ever vendor any of it.

---

## 2. Dreamer 4 — the technical comparison

### 2.0 Summary matrix

| Repo | Policy exists? | Sees `a_t`? | Sees `a_{<t}`? | Anti-copy? | Action space | BC = final? | Blind baseline? |
|---|---|---|---|---|---|---|---|
| lucidrains | **yes** | **no** (shifted) | yes, fully | **none** | caller-defined; shipped envs continuous / 4-way | no — PPO/PMPO/SPO implemented | ⚠️ ships a blind BC test that **certifies** copying |
| nicklashansen | **no** (inert agent token) | n/a | n/a | n/a | continuous torques, instantaneous | n/a — stops at WM | action-*shuffle* ratio (WM, not policy) |
| edwhu | **yes** | **no** (data convention) | yes, fully | **none** | 5-way categorical, held 4–9 steps by the behaviour policy | no — PMPO implemented | none |
| vijayabhaskar | **yes** | **no** (`F.pad` front-shift) | yes, fully | **none** | 2-D continuous torques | no — PMPO implemented | action-shuffle sensitivity on the policy |
| IamCreateAI | **no** | n/a | n/a | n/a | **VPT: per-frame binary keys + mu-law camera DELTAS** | n/a — inference only | none |
| open-dreamer | **no** | n/a | n/a | n/a | VPT keys + camera deltas | n/a — Phase 1 | none |
| machines-in-motion | **no** | n/a | n/a | n/a | robot continuous | n/a — Phase 1 | none |
| **ahriuwu (us)** | **yes** | **no** (n=0 dropped) | **yes, fully** | per-frame dropout (defeated), `--movement-action-mode` | **HELD click target, 89.9% duplicated** | **YES — BC is what we deploy** | **yes — the only one** |

### 2.1 `lucidrains/dreamer4` @ [`aef0d73`](https://github.com/lucidrains/dreamer4/tree/aef0d7343f649aa86f81c856d03111bc36e93cf3)

The most complete *library*: all four stages, on PyPI, no experiments. All line numbers
in `dreamer4/dreamer4.py` unless noted. **[HARD]**

**Layout.** Per-frame pack, agent token **last** —
[`:7919`](https://github.com/lucidrains/dreamer4/blob/aef0d7343f649aa86f81c856d03111bc36e93cf3/dreamer4/dreamer4.py#L7919):
```python
tokens, packed_tokens_shape = pack([flow_token, space_tokens, proprio_token,
    state_pred_token, registers, action_tokens, reward_tokens, maybe_aug_token,
    agent_tokens], 'b t * d')
```
Position-from-the-right is load-bearing: the agent-isolation mask identifies "special"
tokens by index. Masks at
[`:1824-1845`](https://github.com/lucidrains/dreamer4/blob/aef0d7343f649aa86f81c856d03111bc36e93cf3/dreamer4/dreamer4.py#L1824-L1845);
`out = ~(~q_is_special & k_is_special)` at `:1845` is the paper's "no other modality
attends back". Space attention is full-within-frame
([`:3041`](https://github.com/lucidrains/dreamer4/blob/aef0d7343f649aa86f81c856d03111bc36e93cf3/dreamer4/dreamer4.py#L3041),
`causal=False`); time attention is plain causal per spatial stream
([`:3043`](https://github.com/lucidrains/dreamer4/blob/aef0d7343f649aa86f81c856d03111bc36e93cf3/dreamer4/dreamer4.py#L3043)).
⚠️ With `full_spatial_attn=True` the agent-isolation mask **disappears entirely**
([`:3039`](https://github.com/lucidrains/dreamer4/blob/aef0d7343f649aa86f81c856d03111bc36e93cf3/dreamer4/dreamer4.py#L3039)); default is `False`.

**Q1 — sees `a_t`? NO.** Explicit front-shift at
[`:7802-7807`](https://github.com/lucidrains/dreamer4/blob/aef0d7343f649aa86f81c856d03111bc36e93cf3/dreamer4/dreamer4.py#L7802-L7807):
```python
if action_len == time and shift_action_tokens and not is_sequential_step:
    next_action_tokens = action_tokens
    action_tokens = pad_at_dim(action_tokens[:, :-1], (1, 0), value = 0., dim = 1)
```
balanced by an opposite-direction target pad at
[`:8281`](https://github.com/lucidrains/dreamer4/blob/aef0d7343f649aa86f81c856d03111bc36e93cf3/dreamer4/dreamer4.py#L8281)
and `targets[:, 1:]` at
[`:8303`](https://github.com/lucidrains/dreamer4/blob/aef0d7343f649aa86f81c856d03111bc36e93cf3/dreamer4/dreamer4.py#L8303).
The index arithmetic was re-implemented in numpy and executed: for `a = [10..15]`, `h_3`
sees `[0, 10, 11, 12]` and its `n=0` target is `a_3 = 13`. **No off-by-one.** Inference
matches ([`:7179`, `:7184`, `:7809-7812`]).

⚠️ **Live footgun:** `shift_action_tokens=False`
([`:7470`](https://github.com/lucidrains/dreamer4/blob/aef0d7343f649aa86f81c856d03111bc36e93cf3/dreamer4/dreamer4.py#L7470),
"set to False if actions already properly paired") produces a **perfect identity leak at
every timestep** (verified by simulation). No caller passes it, nothing asserts on it,
no test covers it.

**Q2 — sees `a_{<t}`? Yes, fully. Anti-copy: none.** Grep for
`dropout|mask|detach|stop_grad|cond_drop|drop_prob|classifier-free|null action|no_action`
finds only `aug_cfg_dropout_prob=0.1` (augmentation id, `:5249`) and
`add_reward_embed_dropout=0.1` (reward embedding, `:5193`). Neither touches actions.
With the default `actor_depth=0` / `critic_depth=0` (`:5179-5180`) the actor reads the
*same* agent token as the world model (`:7941-7942`).

**⚠️ The finding that matters most.** `test_toy_action_bc.py` is a **deliberate no-pixel
BC probe — and it is a *positive* test**:
```python
latents = zeros(1, 8, 64, 16, device = device)   # test_toy_action_bc.py:71 — ZERO observations
actions_seq = [1, 2, 3, 0] * 2                   # :66
loss = model(latents=latents, discrete_actions=..., add_autoregressive_action_loss=True)
```
It asserts the policy **can** reproduce an action sequence from action history alone with
zero visual signal. Upstream treats action-history copying as a capability to verify, not
a pathology to suppress. That is our bug, sanctioned in the reference library. **[HARD]**

**Q3 — action space.** Caller-defined (`num_discrete_actions`, `num_continuous_actions`,
`:5201-5202`). No Minecraft in the repo. Shipped envs: HalfCheetah continuous torques,
CartPole, and a toy Snake whose *action* is instantaneous but whose *effect* is
persistent (`web_env/env.py:57-58`: `self.direction = action`) — structurally the same
autocorrelation trap as our held click, unaccounted for. **[HARD]**

**Q4 — BC final? No.** `DreamTrainer` (`trainers.py:1340`) does real imagination RL with
`'ppo' | 'pmpo' | 'spo'` objectives (`:6501`; PMPO at `:6729-6784`), GAE at `:6573`,
`only_learn_policy_value_heads=True` by default (`:6500`). Four-stage end-to-end script
in `train_halfcheetah_imagination_rl.py`.

**Q5 — provenance.** No checkpoints, no W&B links, no reported numbers, no reproduction
claims. And the sole test of the agent-isolation mask
(`tests/test_dreamer.py:256`) is **both CUDA-gated and broken** — it passes a stale
`num_agent_tokens=` kwarg against a signature renamed to `num_special_tokens` (`:1897`),
so it TypeErrors on GPU and skips off it. **The agent-isolation mask is effectively
untested.** **[HARD]**

### 2.2 `nicklashansen/dreamer4` @ [`b8abafb`](https://github.com/nicklashansen/dreamer4/tree/b8abafbf4da72c59b6aa09f8499ccde0d6a37fd6)

**The headline is a negative: there is no agent and no policy. This is the world model
only.** **[HARD]** Highest-starred V4 repo, and it stops at Phase 1.

The agent token exists as a placeholder *wired to be inert*. `space_mode` defaults to
`"wm_agent_isolated"`
([`model.py:593`](https://github.com/nicklashansen/dreamer4/blob/b8abafbf4da72c59b6aa09f8499ccde0d6a37fd6/dreamer4/model.py#L593)),
whose mask is
[`model.py:248-251`](https://github.com/nicklashansen/dreamer4/blob/b8abafbf4da72c59b6aa09f8499ccde0d6a37fd6/dreamer4/model.py#L248-L251):
```python
allow = torch.where(allow_non_agent_q, ~is_k_agent, allow)   # nobody sees the agent
allow = torch.where(is_q_agent, is_k_agent, allow)           # agent sees ONLY itself
```
Every call site passes `agent_tokens=None` → zeros (`model.py:677-678`), and the returned
`h_t` is discarded everywhere (`z1_hat_full, _ = dynamics(...)`, `train_dynamics.py:227`).
`TaskEmbedder` (`model.py:543`) is defined and never instantiated. Grep for
`action_head|policy_head|actor|critic|imagination|behaviour_clon`: **zero hits.**

Action–frame alignment for *conditioning* (`train_dynamics.py:732-737`,
`wm_dataset.py:441`): the action token at frame `t` is **u_{t-1}, the action that
produced frame t** — same convention as everyone else.

**Action space:** 30 DMControl/MMBench tasks, per-step continuous torques in `[-1,1]`
zero-padded to 16 dims (`wm_dataset.py:30, 69, 442-450`). **Instantaneous, not held** —
refutes the Minecraft hypothesis for this repo. **[HARD]**

**⚠️ Worth stealing.** They log an **action-shuffle control** every log interval
([`train_dynamics.py:831-857`](https://github.com/nicklashansen/dreamer4/blob/b8abafbf4da72c59b6aa09f8499ccde0d6a37fd6/train_dynamics.py#L831-L857)):
```python
perm = torch.randperm(actions.shape[0], device=actions.device)
loss_shuffled, _ = dynamics_pretrain_loss(..., actions=actions[perm], ...)
action_shuffle_loss_ratio = loss_shuffled / loss
```
Ratio > 1 means the model genuinely uses actions. It is a first-class metric with a
committed curve (`assets/dynamics/action_shuffle_ratio.png`). It is the **mirror image**
of our blind-table bar: ours catches "ignores actions' complement" (ignores pixels),
theirs catches "ignores actions". Cheap — two extra no-grad forwards per log interval.

**Provenance — the strongest of any V4 repo:** real HF checkpoints (tokenizer 90k steps,
dynamics 40k steps), a released 3.6M-frame dataset, committed training curves, ~24h +
~48h on 8× RTX 3090. Lineage: loosely based on `edwhu/dreamer4-jax`. README points to a
successor the author calls "a strictly better implementation", `nicklashansen/mmbench2`.
**[HARD]** (mmbench2 itself **[UNVERIFIED]** — not read.)

### 2.3 `edwhu/dreamer4-jax` @ [`753d650`](https://github.com/edwhu/dreamer4-jax/tree/753d650bf107f8223e5830f5b7417293cc56512d)

Most paper-faithful, and **the only author anywhere who left a written trace of noticing
the hazard.** **[HARD]**

**Layout** (`dreamer/models.py:594-603`, assembled `:680-683`): `[ACTION(1),
SHORTCUT_SIGNAL(1), SHORTCUT_STEP(1), SPATIAL(n), REGISTER(n), AGENT(n)]`, agent last.
Space mask `mode="wm_agent"` (`models.py:245-264`): agent reads everything *including the
same frame's action token*; nobody reads the agent. Time attention is per-slot causal and
`latents_only_time=False` (`models.py:618`), so the AGENT and ACTION slots both attend
causally across time.

**Q1 — sees `a_t`? NO — by data convention.** `dreamer/data.py:376` prepends a dummy so
`actions[t]` is "the action taken from `s_{t-1}` producing `s_t`". BC targets then start
at offset **1** (`train_bc_rew_heads.py:245`):
```python
# At timestep t, predicts actions[t+1], ..., actions[t+L]
# (following Dreamer convention: action a_i happens before state s_i)
offsets = jnp.arange(1, L+1)
```
**The author reasoned it out explicitly** — `train_bc_rew_heads.py:257-258`:
> *"The first offset (n=0) predicts r_t, which depends on a_t that h_t can see."*

That is why the **reward** MTP starts at 0 and the **policy** MTP starts at 1. This is
the clearest independent confirmation that Eq 9's `n=0` action term is degenerate.

**Q2 — sees `a_{<t}`? Yes. Anti-copy: none.** Grep for
`action_dropout|cond_drop|drop_prob|classifier.free|null_action|no_action|stop_gradient|detach`
returns 6 hits, all flow-matching bootstrap targets or RL detaches.

**⚠️ And the shortcut is worth a lot in their own data.** The toy behaviour policy is
*commit-then-switch* with `hold_min=4, hold_max=9` (`data.py:280-321`,
`train_bc_rew_heads.py:69-70`), so `a_{t+1} == a_t` with probability ≈ 1 − 1/6.5 ≈ **85%**
on a 4-way action. **A pure copy head scores ~85% BC accuracy without seeing a pixel —
and this repo never measures that.** This is our bug, at 85% instead of 90%, in a
reference implementation, unmeasured. **[HARD]**

**Q3 — action space:** 5-way categorical `{up,down,left,right,null}`, instantaneous —
but *held for 4–9 steps by the behaviour policy*, which is the same statistical property
as our held click.

**Q4 — BC final? No.** `scripts/train_policy.py:1437-1497` is a real PMPO update:
TD-λ returns → advantages → sign-partitioned pos/neg log-prob terms → `β·KL(π_θ‖π_BC)`
against a frozen BC head. World model frozen in Phase 3 (`h_sg = stop_gradient(h)`,
`:1341`).

**Q5 — provenance:** all four phases, **bouncing-square toy dataset only**. Committed
training figures, real cluster paths, W&B entity. No checkpoints, no blind baseline, no
results table. Not LLM-generated — the git history shows real debugging
(`b58d6a5 "fix rew pred bug, offby one"`). README honestly scopes itself to the toy task.

### 2.4 `vijayabhaskar-ev/dreamer_v4` @ [`484759e`](https://github.com/vijayabhaskar-ev/dreamer_v4/tree/484759e1829b936d7c9434dff7c2073be9a5cd6a)

Most complete end-to-end pipeline with real measurements. **[HARD]**

**Layout** (`dynamics/dynamic_model.py:46, 205-209`): `[action(1), tau_d(1),
z_latent(32), register(4), agent(1)]`. Temporal mask is sliding-window causal, `k <= q`
inclusive, window 16 (`:114`), applied per spatial slot. Spatial firewall at `:143`:
`mask[:n_base, n_base:] = float("-inf")`. **Unit-tested**:
`tests/test_gradient_isolation.py` asserts `|∂z_hat/∂agent_embedding| == 0.0` exactly and
`|∂agent_out/∂agent_embedding| > 0`. (lucidrains' equivalent test is broken; this one
works.)

**Q1 — sees `a_t`? NO — in-model front shift** (`dynamic_model.py:191-197`):
```python
# actions: (B, T-1, A) -> pad zero at t=0 -> (B, T, A)
actions_padded = F.pad(actions, (0, 0, 1, 0))
```
so `action_token[t] = actions[t-1]`, `t=0` gets `no_action_emb`. Targets use offset 0
(`trainer.py:1155-1161`), giving the same one-step separation. Consistent at deployment
(`evaluate_env.py:277-291`: **W frames of latents but only W−1 actions**) and in
imagination (`rollout.py:95-96, 114-132`).

**Q2 — anti-copy: none.** But **this repo contains the only action-shuffle probe run on a
policy**, `dynamics/evaluate_agent.py:753-790`, with results committed at
`evaluation/agent-optionA-e040/action_shuffle_agent.csv`:
```
reward_abs_delta,0.0141
policy_mu_abs_delta_dim0,0.1899
policy_mu_abs_delta_dim1,0.1543
```
Against their `action_rmse = 0.3017`, **shuffling the action history moves the policy
mean by ~50–63% of its total error scale.** That is *their* version of our
"150.9deg vs 41.0deg". They do not interpret it as a copy diagnostic. It is confounded
(shuffling actions also perturbs the world-model latent) but it is direct evidence the
hazard is general, not ours alone. **[HARD]**

Suggestive corroboration: their **deterministic readout collapses to the random floor**
(mean 0.104 / argmax 0.102 vs random 0.094, against a sampled BC of 0.356). They
attribute it to OOD drift. Given the shuffle numbers, a partially-copying policy is at
least an equally good explanation. **[INFER]**

**Q3 — action space:** DMC `ball_in_cup_catch`, 2-D continuous torques. Also a
`generate_dataset.py` path with a **sinusoidal** exploration policy (`:52-64`) — smooth,
therefore highly autocorrelated, so `a_{t-1}` strongly predicts `a_t` here too.

**Q4 — BC final? No.** `imagination/algorithms.py` implements Eq 10 (λ-returns) and
Eq 11 (PMPO with reverse KL to a frozen Phase-2 prior); `imagination/rollout.py` does
H-step latent imagination; three executable test files.

**Q5 — provenance:** HF checkpoints (~814 MB), 30 committed per-episode CSV/JSON eval
artifacts (n=500 episodes each, 6 RL seeds × 3 BC inits),
`analysis/paper_stats.py` regenerates every README number. Reported: BC 0.356 →
imagination-RL 0.415 (+5.9 pts, 95% CI [+1.5, +10.4]), random floor 0.094. Prose is
heavily LLM-assisted; the measurements are real (97 commits, documented bug-fix history,
a public README self-correction superseding an earlier null result). **Treat the numbers
as real and the prose confidence as inflated.** **[INFER on the last point.]**

### 2.5 `IamCreateAI/Dreamerv4-MC` @ [`166d7ca`](https://github.com/IamCreateAI/Dreamerv4-MC/tree/166d7ca92ac38d57c6c67daaa0518da87f2a83af)

**World model only. No agent token, no policy, no losses, no training code** — README
says "Inference Only", "Training Code — Coming Soon". Grep for
`agent|policy|actor|critic|imagin|behavior|bc_` across `src/` and `ui/`: zero relevant
hits. **[HARD]**

Its value here is **confirming the Minecraft action representation from code**.
`src/modules/actokenizer.py:8-58` carries the MineRL/VPT mapping and `NOOP_ACTION`:
per-frame binary buttons (`forward/back/left/right/jump/sneak/sprint/attack/use/drop/
inventory/hotbar.1-9`) plus `"camera": np.array([0,0])` — a **DELTA**, mu-law quantized
into 21 bins per axis (`CameraQuantizer:100-158`). The UI accumulates mouse deltas per
frame and **resets them every tick** (`ui/inference_ui.py:99-101, 107-112`). Held keys
persist as a `Set[str] keys_down` (`:72`), so holding W yields `forward=1` every frame —
but the *representation* is per-frame binary, not a carried goal. **[HARD]**

Per-frame concat at `src/modules/dynamic_model.py:553`: 256 latent + 1 time/stride + 12
action + 4 registry = 273 tokens. Spatial attention is full-bidirectional within a frame
(`:164-171`), so image tokens read the same frame's action tokens; temporal attention is
**image tokens only** (`:262-286`) — action tokens have no cross-frame path of their own
and are re-injected per frame. Clearly run on real hardware (Triton RoPE kernel, CUDA
graphs, Chinese-language debugging comments). It is an interactive Minecraft video model
with a Dreamer-shaped backbone, not a Dreamer agent.

### 2.6 `next-state/open-dreamer` @ [`797e41f`](https://github.com/next-state/open-dreamer/tree/797e41f052b5996740938fd2fe8161f1866de3a2)

**Phase 1 only.** File tree shows `scripts/train_tokenizer.py` and
`scripts/train_dynamics.py` and nothing else; `dreamer/models.py` has an `ActionEncoder`
(`:1003`) and no policy, actor, critic, agent token, BC or PMPO. **[HARD]**

Two things worth recording:
1. Token order per timestep (`dreamer/training.py:834`):
   `jnp.concatenate([action_token, shortcut_token, spatial_tokens, register_tokens])`.
2. **It shifts actions too** — `dreamer/actions.py:54`, applied unconditionally at
   `scripts/train_dynamics.py:274`:
   ```python
   def shift_actions(actions, categorical_action_dim):
       """Shift actions right by 1, preprend noop action."""
   ```
   So the action token at frame `t` is `a_{t-1}`. **A fifth independent codebase reaching
   the same convention.** **[HARD]**

Action space is VPT's (`actions.py`, "VPT action space" section, `key_to_index` mapping
keyboard + camera categorical), consistent with §2.5.

⚠️ Licence: all-rights-reserved. Do not vendor.

### 2.7 `machines-in-motion/dreamer-v4` and `RajatDandekar/dreamer4-coinrun`

Both **Phase 1 only, no policy**. **[HARD]**

`machines-in-motion` @ [`c319a28`](https://github.com/machines-in-motion/dreamer-v4/tree/c319a283f7290a2d11c9740c92b2ae63e090af26):
the file list is `train_tokenizer.py` + `train_dynamics.py`; the README states it
implements "the Dreamer-V4 world model component (causal tokenizer and interactive
dynamics)". Token order per timestep is
`[latent_tokens : register_tokens : diff_control_token : action_tokens]`
(`dreamerv4/models/dynamics.py:142`), i.e. `a_t` is inside frame `t`'s block, with no
agent token to read it. Robot datasets (SOAR, PushT), continuous actions, HF checkpoints.

`RajatDandekar/dreamer4-coinrun` @ [`507d025`](https://github.com/RajatDandekar/dreamer4-coinrun/tree/507d02572078813df22ed26cd22f9079334ff960):
**contains no model code at all** — it is a Modal training harness plus a write-up over
`next-state/open-dreamer`, which it deliberately gitignores because of that repo's
licence. Reported tokenizer PSNR 40.41, end-to-end FVD 32.19, ~$150. Phase 1 only.

---

## 3. Dreamer 3 implementations, and why the mechanism does not transfer

### 3.1 The landscape

**[HARD]** for stars/dates/licences (authenticated `gh api`, 2026-08-27);
**[INFER]** for the "reproduced results" column — a score table in a README is a claim,
not an independent reproduction. Nothing here was run.

**Core lineage**

| Repo | ★ | pushed | licence | framework | Reproduced results |
|---|---|---|---|---|---|
| **danijar/dreamerv3** (official) | 3705 | 2026-05-25 | MIT | JAX (ninjax/embodied) | **yes** — ships `scores/*.json.gz` for atari57, atari100k, dmc_proprio, dmc_vision, dmlab30, minecraft_diamond, procgen, *with baselines* (MuZero, PPO, IMPALA, DrQv2, SAC, Rainbow) + `plot.py` |
| danijar/dreamerv2 | 1056 | 2023-01-21 | MIT | TF2 | yes — curves for all 55 games + `scores/` |
| danijar/dreamer (V1) | 621 | 2021-09-10 | MIT | TF2 | no; redirects to V2 |
| google-research/dreamer | 746 | 2020-07-14 | Apache-2.0 | TF1 | no; original V1 |

Note: `danijar/dreamerv3` has **no tags and only `main`** (31 commits) and was rewritten
on 2024-04-26 (`2411f7d`). The pre-rewrite tree that NM512 ports and most papers cite is
`8fa35f83eee1ce7e10f3dee0b766587d0a713a60`. **Both generations were checked; all
conclusions in §3.2 hold in both.** **[HARD]**

**Reimplementations**

| Repo | ★ | pushed | licence | framework | Reproduced results |
|---|---|---|---|---|---|
| **NM512/dreamerv3-torch** | 887 | 2026-03-08 | MIT | PyTorch | yes (DMC, Atari100k, Crafter plots) — ⚠️ **now self-deprecates** |
| **NM512/r2dreamer** | 169 | 2026-05-31 | MIT | PyTorch | yes — official R2-Dreamer (ICLR 2026), 8-benchmark table |
| **Eclectic-Sheep/sheeprl** | 437 | 2026-08-24 | Apache-2.0 | PyTorch + Fabric | **best head-to-head** — Crafter 12.1 vs paper 11.7 @1M; MsPacman 1542 vs 1327 @100K; admits n=1 runs. V1+V2+V3 + Plan2Explore |
| **pytorch/rl** (`sota-implementations/dreamer_v3`) | 3539 (repo) | 2026-08-27 | MIT | TorchRL | **uniquely rigorous** — `benchmark.py --seeds 0 1 2` with an *automated* acceptance gate (min final median return 700 on DMC Walker-Walk) |
| ray-project/ray (`rllib/.../dreamerv3`) | 43.6k (repo) | 2026-08-27 | Apache-2.0 | PyTorch | weak — presets, no score table |
| burchim/DreamerV3-PyTorch | 15 | 2024-02-16 | Apache-2.0 | PyTorch | best results-per-star — DMC 1M vs paper, 3 seeds × 10 eps |
| symoon11/dreamerv3-flax | 19 | 2025-11-29 | **none** | JAX/Flax | yes — Crafter 17.65 ± 2.29 over 10 seeds; documents 5 deviations |
| qiwang067/LS-Imagine | 235 | 2026-05-22 | Apache-2.0 | PyTorch | ICLR 2025 Oral + checkpoints |
| AutonomousAgentsLab/cr-dv3 | 40 | 2023-07-05 | MIT | JAX fork | Curious Replay, ICML 2023; no table |
| InexperiencedMe/NaturalDreamer | 175 | 2025-03-18 | **none** | PyTorch | partial & honest — CarRacing only |
| naivoder/dreamerv3 | 19 | 2025-08-08 | **none** | PyTorch | no — "Under Construction" |

⚠️ **Two corrections worth propagating.**
- **`NM512/dreamerv3-torch` — the de facto community PyTorch baseline, cited by most
  papers — now carries a deprecation banner** ("does not reflect [major DreamerV3]
  changes, which accounts for several GitHub Issues") redirecting to `NM512/r2dreamer`.
  Any comparison citing it as current is stale.
- **`kc-ml2/SimpleDreamer` (159★) is DreamerV1 + Plan2Explore, not V3.** Its own TODO
  still lists "dreamer-v2, dreamer-v3". It is frequently miscited.

V2-era ports worth knowing: `jurgisp/pydreamer` (240★, MIT — best-documented divergence:
an explicit table of the 5 hyperparameter differences explaining its gaps; author is a
DreamerV3 co-author), `RajGhugare19/dreamerv2` (274★, public W&B + POMDP ablation),
`jsikyoon/dreamer-torch` (142★, the port NM512 builds on), `kenjyoung/dreamerv2_JAX`
(18★, whole loop JIT'd and vmapped across seeds).

Adjacent world-model agents for contrast: `eloialonso/diamond` (2097★, MIT, per-game
per-seed JSON + playable pretrained world models), `eloialonso/iris` (900★, ⚠️ GPL-3.0),
`weipu-zhang/STORM` (143★, no licence; superseded by `OC-STORM`).

### 3.2 Why DreamerV3 cannot have our bug

**[HARD]**, verified in both the current and pre-rewrite JAX trees and in the PyTorch
port.

**(a) The actor sees `a_{t-1}` only through the recurrent state, never as an input.**
The RSSM consumes it —
[`rssm.py:80`](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/dreamerv3/rssm.py#L80)
`deter = self._core(deter, stoch, action)`, and `:145` concatenates it into the GRU
input. PyTorch proves it dimensionally:
[`networks.py:50`](https://github.com/NM512/dreamerv3-torch/blob/6ef8646d807cd10ce0c88e10a7e943211e7fc44c/networks.py#L50)
`inp_dim = self._stoch * self._discrete + num_actions`. But the actor's entire input is
[`agent.py:51-54`](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/dreamerv3/agent.py#L51-L54)
— `concat[deter, stoch]` — and PyTorch again dimensionally:
[`models.py:227`](https://github.com/NM512/dreamerv3-torch/blob/6ef8646d807cd10ce0c88e10a7e943211e7fc44c/models.py#L227)
`feat_size = config.dyn_stoch + config.dyn_deter` — **no `+ num_actions`**.

*Trap worth knowing:* the pre-rewrite JAX code calls `self.actor(sg(traj))` on a dict
that **contains** an action key (`agent.py:293`). It looks like a leak. It is not — the
MLP has an input selector and `configs.yaml:78` sets `actor: {inputs: [deter, stoch]}`.
(By contrast the Plan2Explore disagreement head at `configs.yaml:100` *does* declare
`inputs: [deter, stoch, action]` — the one directly action-conditioned module, and it is
an ensemble next-state predictor, not a policy.)

**(b) The actor is trained ONLY on imagined rollouts of its own samples.**
`agent.py:192-193` builds the rollout with `policyfn = lambda feat: sample(self.pol(...))`,
and inside the imagination step the action fed forward is the actor's own draw from step
one (`rssm.py:96-98`). The replayed data contributes only the *starting state*. The actor
loss (`agent.py:411-415`) is REINFORCE on `imgact` — its own samples — with a normalised
advantage and an entropy bonus (`actent: 3e-4`). PyTorch is identical in structure
(`models.py:356-362`, loss `:391-427`).

**(c) There is NO behaviour-cloning or action-prediction loss in DreamerV3. At all.**
This is airtight because the code *enforces* its complete loss set —
[`agent.py:237`](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/dreamerv3/agent.py#L237)
asserts `set(losses.keys()) == set(self.scales.keys())` against the eight names fixed in
`configs.yaml:86`: `{rec, rew, con, dyn, rep, policy, value, repval}`. Every supervised
target in the codebase is `obs['reward']`, `con`, or an observation reconstruction. Grep
for `behavio(u)?r.?clon|bc_loss|imitat|action_pred|action_head|supervis` across both
generations: **zero hits.** In PyTorch, `data["action"]` appears in exactly three places:
`models.py:119` (RSSM observe) and `:199, :206` (video-prediction visualisation).

> **This settles it. "Emit `a_{t-1}`" lowers no DreamerV3 loss, because there is no
> dataset action label for the actor to reproduce. The copy solution has no analogue.
> Dreamer 4 introduces behaviour cloning — and with it, this hazard.**

**(d) Actions are instantaneous, and action repeat does not duplicate labels.**
Discrete → one-hot (`nets.py:493`); continuous → symlog-squished float. Action repeat is
**env-side** (`wrappers.py:57-73`: `for _ in range(self._repeat): obs = self.env.step(action)`),
so one agent step consumes N env frames — it *collapses* frames rather than inflating the
`a_t == a_{t-1}` rate in the agent's own data. Sticky actions live inside ALE
(`atari.py:61`) and don't change the recorded label. The one genuine held action in the
whole codebase is Minecraft's `sticky_attack=30` / `sticky_jump=10`
(`embodied/envs/minecraft_flat.py:177-178`) — env-side, one env, not a property of the
algorithm.

**(e) No blind / no-observation baseline is reported.** In the 40-page paper
(2301.04104): "blind" 0 occurrences, "imitat" 0, "behavior clon" 0, "previous action" 0.
The reported ablations are the five robustness variants and two learning-signal variants
(Figs. 6, 17, 18). "Open loop" appears 7 times and always means *world-model video
prediction on ground-truth actions*, not a blind actor. `config.random_agent`
(`main.py:134-135`) exists as a debugging harness and is not a reported ablation.

---

## 4. What the Dreamer 4 paper actually says

All quotes from arXiv 2509.24527v1, text-extracted locally. **[HARD]**

### 4.1 The attention is block-causal and the action is inside the block

> "To support interactive generation, the attention is masked to be causal in time, so
> that **all tokens within a time step can attend to each other** and to the past."

Figure 2(b) draws each timestep block as `t d a z̃` — signal level, step size, **action**,
noisy latent.

> "For this, we insert agent tokens as an additional modality into the world model
> transformer and **interleave it with the image representations, actions, and register
> tokens**. […] While the agent tokens attend to themselves and all other modalities,
> **no other modalities can attend back to the agent tokens**."

**Consequence [HARD, by composition of the two quotes]:** the agent token at time `t` is
in the same block as the action token for time `t`, and blocks are internally dense. So
`h_t` **does** see `a_t`. The asymmetry protects the **world model** from causal
confusion ("*its future predictions can only be directly influenced by actions, not by
the current task*"); it does nothing to protect the **policy** from reading actions.

### 4.2 Eq 9 sums from n=0 — and that term is degenerate

> L(θ) = − Σ_{n=0}^{L} ln p_θ(a_{t+n} | h_t) − Σ_{n=0}^{L} ln p_θ(r_{t+n} | h_t)   (9)

with L = 8.

Our working assumption — recorded in `docs/BC_FIX_PLAN_2026-08-26.md` option A, *"their
Eq 9 sums from n=0, which is vacuous unless h_t excludes a_t"* — **is wrong about their
architecture.** Given §4.1, `p_θ(a_t | h_t)` is a copy read-off. `edwhu` reached the same
conclusion independently and wrote it down (§2.3). This is a correction to our own notes,
not a discovery about a bug in the paper: the term simply contributes ~0 loss.

### 4.3 The action space is instantaneous, and half of it is a delta

> "We represent keyboard actions as **23 binary distributions** and mouse actions as a
> **categorical with 121 classes using foveated discretization**."

Foveated discretization is VPT's binning of the per-frame **camera delta** — confirmed
from code in §2.5 (`CameraQuantizer`, mu-law, deltas reset every tick). So one whole half
of the Dreamer 4 action is a *difference*: mean-reverting, weakly autocorrelated, and
copying `a_t` predicts `a_{t+1}` poorly for it. The 23 keyboard bits are per-frame states
— autocorrelated while a key is held, but each is a 1-bit decision whose consequence
(the entire viewport translating) is visible in the very next frame.

### 4.4 BC is an initialisation; Phase 3 trains under the self-fed loop

Algorithm 1:
> Phase 2: Agent Finetuning — finetune world model with task inputs for policy and reward
> heads using (7) and (9).
> Phase 3: Imagination Training — optimize policy head using (11) and value head using
> (10) on trajectories generated by the world model **and the policy head**.

And in prose:
> "The rollouts are generated by **unrolling the transformer with itself**, sampling
> representations z from the flow head and **actions a from the policy head**."
> "We initialize a value head and a frozen copy of the policy head that serves as a
> behavioral prior. We only update the policy and value heads and keep the transformer
> frozen."

**This is the single most important structural fact in this document.** Phase 3's
training distribution *is* the self-fed closed loop — the exact regime in which our head
collapses. A copy-policy there produces a degenerate trajectory, earns no reward, gets
negative advantage, and PMPO (Eq 11, sign-of-advantage) pushes mass off it.

They also measure it — Figure 4, four milestones, success %:

| agent | m1 | m2 | m3 | m4 |
|---|---|---|---|---|
| BC (notask) | 64 | 8.8 | 0.2 | 0 |
| BC | 93 | 54 | 4.3 | 0.62 |
| VLA (Gemma 3) | 97 | 77 | 23 | 11 |
| **WM+BC** (their Phase 2 = our stage) | **99** | **89** | **28** | **17** |
| **Dreamer 4** (+ Phase 3) | **99** | **90** | **40** | **29** |

Read this carefully: **their Phase-2 BC is already strong on its own** (99/89), and
imagination RL adds most of its value on the hard milestones. So the answer is *not*
simply "their BC is broken too and RL rescues it". It is more specific than that (§6).

Also note (paper §4.1): *"The behavioral cloning loss is applied only on the relevant
fraction, while the dynamics loss is applied only on the uniform sequences"* of a 50/50
data mixture. Their BC never trains on the boring 50%.

### 4.5 No action dropout and no blind baseline

Full-text search: "dropout" occurs **once**, in §3.1 — *"We drop out input patches to the
encoder to improve its representations using masked autoencoding."* That is the tokenizer
MAE. There is **no action dropout, no action masking, no stop-gradient on the action
channel, no separate action-free policy pass** anywhere. **[HARD]**

Nor any blind / no-pixel / copy-previous-action / chance baseline for the policy. The
strings "chance", "no-op" and "random policy" do not occur. Every reported baseline —
VPT (finetuned), BC (notask), BC, VLA (Gemma 3), WM+BC — is a sighted policy. **[HARD]**

---

## 5. How our code compares

All references verified on `main` @ `ab8eb7c`. **[HARD]** throughout.

### 5.1 Q1 — does our policy see the CURRENT action `a_t`? Yes it does — but we are NOT off-by-one

Trace, in order:

1. `src/ahriuwu/models/dynamics.py:774-779` — the action token for frame `t` is
   concatenated into **frame t's own token set**:
   ```python
   if self.use_actions:
       if actions is not None:
           action_token = self.embed_actions(actions)
       else:
           action_token = self.no_action_embed.expand(B, T, -1)
       x = torch.cat([x, action_token.unsqueeze(2)], dim=2)
   ```
   **No shift.** `actions['movement'][:, t]` is `a_t` — the command in effect at frame
   `t` — and it lands in slot `t`.
2. `src/ahriuwu/models/dynamics.py:727-747` — one agent token per frame, then
   `agent_block(agent_tokens, x)`.
3. `src/ahriuwu/models/dynamics.py:322` — `agent_tokens + self.cross_attn(...)`.
4. `src/ahriuwu/models/dynamics.py:206-212` — the query is `(B, T, H, 1, head_dim)` and
   keys are built **per frame** (`z_flat = z_tokens.view(B*T, S, D)`). The
   cross-attention is **within-frame**: agent token `t` attends to exactly frame `t`'s
   token set — **including the action token holding `a_t`**.

So `h_t` contains `a_t`, exactly as in Dreamer 4 (§4.1).

**But the loss compensates.** `scripts/train_agent_finetune.py:790`:
```python
for n in range(1, mtp_length):  # n >= 1: predict the NEXT actions only
```
Working the indices against the four implementations that shift:

| codebase | action token at frame `t` | first BC target from `h_t` | separation |
|---|---|---|---|
| lucidrains (`:7807` + `:8303`) | `a_{t-1}` | `a_t` (n=0) | **1 step** |
| edwhu (`data.py:376` + `:245`) | `a_{t-1}` | `a_t` (offset 1) | **1 step** |
| vijayabhaskar (`F.pad`, `:193`) | `a_{t-1}` | `a_t` (offset 0) | **1 step** |
| open-dreamer (`shift_actions`) | `a_{t-1}` | — (no policy) | — |
| **ahriuwu** | **`a_t`** | **`a_{t+1}` (n=1)** | **1 step** |

**Our indexing is EQUIVALENT to theirs.** Two research passes independently suggested
"your bug is that the action token at frame t is `a_t`" — that suggestion is **wrong for
our code**, because we drop `n=0`. There is no off-by-one. Anyone re-deriving this should
stop here rather than spend GPU on a shift.

**One real difference does survive, and it is at inference.** Under their convention the
newest frame's action slot holds `a_{t-1}`, a real known past action — nothing has to be
invented. Under ours, slot `t` must hold `a_t`, which at decision time **does not exist
yet**. `scripts/agent_infer.py:222` invents it:
```python
stand_in = hist[-1] if hist else {"movement": (0.5, 0.5), "abilities": {}}
```
i.e. **it repeats the standing order into the exact slot that most strongly determines
the output.** Under a held representation that stand-in equals `a_t` ~90% of the time, so
train and inference agree — and both hand the model the answer. Their layouts never
create such a slot. Compare `vijayabhaskar/evaluate_env.py:277-291`: **W frames of
latents, W−1 actions.** This is a genuine, cheap, structural cleanup available to us
(shift our action tokens and target `a_t` at `n=0`), though on its own it does **not**
remove the copy hazard — see §6-H0.

### 5.2 Q2 — previous actions, and what prevents copying

**Sees them: yes, fully.** `src/ahriuwu/models/dynamics.py:325` —
`agent_tokens + self.self_attn(self.norm2(agent_tokens))` with `mode="temporal"` and a
causal mask. Agent token `t` reads agent tokens `<= t`, each of which already absorbed its
own frame's action token. Same as every implementation surveyed.

**What we do about it — three mechanisms, all inadequate as shipped:**

| mechanism | site | verdict |
|---|---|---|
| per-frame action dropout | `scripts/train_agent_finetune.py:993-996` | **defeated by construction** |
| `--movement-action-mode {held,event_only,none}` | `scripts/train_agent_finetune.py:983-991` | correct, but `held` is the default and is what shipped |
| `n=0` dropped from the MTP sum | `scripts/train_agent_finetune.py:790` | necessary, nowhere near sufficient |

```python
p_drop = getattr(args, "action_dropout", 0.0)
if p_drop > 0 and dynamics.use_actions and "cursor_valid" in actions:
    keep = torch.rand_like(actions["cursor_valid"], dtype=torch.float32) >= p_drop
    actions["cursor_valid"] = actions["cursor_valid"] & keep
```
This is **i.i.d. per frame**. Because the held target is byte-identical across an entire
hold run and temporal attention is causal over the action slot, the answer is hidden only
if **every** frame since the last click drops: `p^(j+1)` = **1.87%** at the shipped
`p=0.15` (`ops/bc5080_gate_watchdog.sh:28`, `ops/bc5080_clicks_watchdog.sh:43`,
`ops/bc5080_parity_watchdog.sh:50`). And `no_action_embed` is a *single learned vector*
(`src/ahriuwu/models/dynamics.py:478`, applied `:575-576`), so a dropped frame is
**identifiable** — the model knows to look at a neighbour rather than being forced to
guess.

Worse for measurement: `scripts/train_agent_finetune.py:1090` sets
`args.action_dropout = 0.0` inside `evaluate()`, so **every val number ever reported was
measured with the shortcut fully open.**

**Nothing else exists** — no stop-gradient on the action channel, no second action-free
forward pass, no structural mask removing the action token from the agent cross-attention
only. That last one (option A of `docs/BC_FIX_PLAN_2026-08-26.md`) is not implemented.

**Field context:** *no Dreamer 4 implementation has any of these either* (§2.0). Even our
defeated per-frame dropout is more than any of them ship. **There is no prior art to
port. On this specific issue we are ahead of the field and have to invent the fix.**

### 5.3 Q3 — action representation: HELD. This is the crux.

`src/ahriuwu/data/replay_dataset.py:629-642`:
```python
held_x = held_y = 0.5  # pre-first-click: screen centre (no command yet)
for i in range(T):
    e = events.get(i)
    if e is not None:
        ...
        held_x = min(max(x, 0.0), 1.0)
        held_y = min(max(y, 0.0), 1.0)
        event[i] = True
    movement[i, 0], movement[i, 1] = held_x, held_y
```

The movement action is a **frozen screen coordinate carried forward every frame until the
next click**. Consequences:

- `a_{t+1} == a_t` **bit-for-bit on 89.9% of frames.**
- The BC target at `n=1` therefore sits inside the model's own input on ~90% of frames,
  and copying it is the **Bayes-optimal** predictor given that input. **The head is not
  broken. The objective as posed has a trivial exact solution and the head found it.**
- Median hold run 5 frames, p99 67, max 728 — the answer is **replicated across every
  frame of the run**, which is precisely why i.i.d. dropout cannot touch it.
- The held target is a **hidden engine state variable**, not something the screen shows.
  The champion is camera-locked to screen centre, so a standing move order has almost no
  pixel signature. Compare Minecraft, where "W is held" repaints the entire viewport
  every frame.

**How the field compares** (all **[HARD]**, §2):

| codebase | representation | P(a_{t+1} == a_t) | copy shortcut worth |
|---|---|---|---|
| Dreamer 4 / IamCreateAI / open-dreamer | 23 per-frame binary keys + **camera DELTA** (121 bins) | high for keys, **low for the delta** | partial |
| nicklashansen | continuous torques, instantaneous | low | little |
| vijayabhaskar | 2-D continuous torques (smooth / sinusoidal data) | high-ish (smooth) | moderate |
| edwhu | 5-way categorical, **commit-then-switch, hold 4–9 steps** | **≈85%** | **~85% BC accuracy blind** |
| **ahriuwu** | **HELD screen coordinate, frozen across the run** | **89.9%, bit-for-bit** | **the whole thing** |

Note edwhu at ~85% — **the hazard is not unique to us.** What is unique is that (a) our
copy is *bit-for-bit exact* rather than merely likely, (b) our action's consequence is
nearly invisible in pixels, and (c) BC is our final policy (§5.4).

### 5.4 Q4 — is BC our final policy? YES, and that is the asymmetry

`scripts/agent_infer.py` is the deployment path and runs the Phase-2 BC head directly.
**Phase 3 (imagination RL / PMPO) is not implemented and has never been run.** Of the
three V4 repos with a policy, *all three* implement Phase 3; we are the only Dreamer-4-
shaped codebase surveyed that deploys BC as the terminal policy.

`docs/PAPER_DEVIATIONS.md` §4.5 also records that `task_id` is **never passed** by any
trainer or by `agent_infer.py`, so `dynamics.task_embed` sits at random init. Dreamer 4's
agent token carries a task embedding; **ours carries nothing but a constant, a temporal
position, and whatever it cross-attends to — where the action token sits.**

And the loop closes on itself. `scripts/agent_infer.py:221-267` builds the action history
from the agent's own emissions (`self.act_buf`, appended at `:326` and `:349`). A lookup
keyed on the previous target, fed its own previous output, is pure positive feedback —
measured directional concentration R = 0.643 / 0.739, one arbitrary direction per game.

### 5.5 Q5 — blind baselines: we are the only ones who report one

`scripts/train_agent_finetune.py:353-369`:
```python
MOVE_CE_MARGINAL    = 5.2321
MOVE_CE_BLIND_TABLE = 4.1214  # <-- THE BAR. 21x21 lookup p(next bin | prev bin), NO PIXELS.
MOVE_CE_DEPLOYED    = 4.1212
```
The deployed 146M stack matches a no-pixel count table to **0.0002 nats**.

Across the whole survey — one paper, nine V4 repos, eleven V3 repos, the V3 paper — the
only comparable artifacts are:
- `nicklashansen/dreamer4` `action_shuffle_loss_ratio` (world model, not policy) — §2.2;
- `vijayabhaskar-ev` `evaluate_agent.py:753-790` action-shuffle sensitivity on a policy,
  whose own committed numbers show the policy mean moving ~50–63% of its error scale
  under an action shuffle, uninterpreted — §2.4;
- `lucidrains` `test_toy_action_bc.py`, which is our experiment run with the **opposite
  sign convention** — it *asserts* that the policy reproduces actions from history with
  zero pixels — §2.1.

**Our instrumentation is the strictest in the ecosystem, and it is what found the bug.**

### 5.6 Three of our problems have no Dreamer counterpart at all

1. **The sticky movement gate.** Our head factorises into "did a new command fire" ×
   "which cell" (`scripts/train_agent_finetune.py:750-870`). On ~90% of frames the target
   is "no new command / keep holding", which under `movement_mode='axis'` is expressible
   only by *re-emitting the held target* — i.e. by copying. No Dreamer policy has a gate;
   every frame carries a genuine action label.
2. **The pre-first-click sentinel.** `_parse_movement_clicks` defaults the target to
   `(0.5, 0.5)` for the first ~61s of **every** game (159,330 frames, 4.2% of corpus),
   which decodes as "your move order is your own feet". Those are the only fountain/base
   frames in the corpus, so **100% of what the model ever sees of the walk-out is a label
   only copying can express.**
3. **BC loss coverage.** The paper applies BC only to the "relevant fraction" of a 50/50
   mixture (§4.4). `docs/PAPER_DEVIATIONS.md` §4.6 records that we have no such mixture
   and train BC uniformly on every window — including the sentinel window and every hold
   run.

---

## 6. The direct answer

> **"Dreamer 4 lets the policy attend to previous actions. Why does that hurt US and
> (apparently) not them?"**

Six candidate explanations. Evidence for each, then the verdict.

### H0 — "We have an off-by-one they don't." **REJECTED.**

Four independent codebases shift the action token so that frame `t` carries `a_{t-1}`
(§2.1, §2.3, §2.4, §2.6). We do not shift — but we drop `n=0` from the MTP sum
(`scripts/train_agent_finetune.py:790`), which produces **the identical one-step
separation** (§5.1 table). This was the most attractive hypothesis and it is false.
**Do not spend GPU on a shift.** The one real gain from adopting their convention is
inference hygiene: it removes the invented `a_t` slot at `agent_infer.py:222` that
currently injects the standing order into the most influential position. Cheap, worth
doing, not the cure. **[HARD]**

### H1 — "They defend against copying and we didn't." **REJECTED.**

Not one of the nine V4 repos has action dropout, null-action conditioning, CFG,
stop-gradients on the action stream, or an action-free actor branch (§2.0). Neither does
the paper (§4.5). **Our defeated per-frame dropout is strictly more defence than the
entire field ships.** Their survival is not owed to a mechanism we skipped. **[HARD]**

### H2 — "Their perception is better, so vision outcompetes the shortcut." **PARTIALLY TRUE, and the weakest leg of the real answer.**

A shortcut only wins when it beats the alternative. In Minecraft, the consequence of an
action is *maximally* visible: pressing W repaints the entire viewport next frame, and
half the action is a camera delta whose effect *is* the image transform. Vision is a
strong competitor there. In our game the champion is camera-locked to screen centre, so
a standing move order has almost no pixel signature, and our probes found ≤5% of an
oracle's recoverable signal in the latents
(`docs/MOVEMENT_HEAD_BLIND_2026-08-26.md`, CORRECTION §3).

**But this cannot be the whole story, and our own SECOND CORRECTION disproves the strong
form of it.** Cutting the movement action at inference with **no retraining** recovers a
side-conditioned top-lane policy — 42.5% → 70% TOP executed, blue/red separation 76.3deg
at perm p < 2e-4. **The visual policy was there all along; it was drowned, not absent.**
So H2 explains why the shortcut is *tempting* here, not why the outcome is catastrophic.
**[HARD]**

### H3 — "Held vs instantaneous action representation." **TRUE, and mechanically decisive.**

This determines *how much* the shortcut is worth, and it is the difference between a
nuisance and a total loss.

- Ours: a frozen screen coordinate, **byte-identical on 89.9% of frames**, replicated
  across every frame of a run (median 5, max 728). Copying is not "a good heuristic" —
  it is the **exact Bayes-optimal answer** given the model's input, and it is what makes
  i.i.d. dropout mathematically hopeless (1.87% at p=0.15).
- Theirs: 23 per-frame bits plus a **camera delta**. Half the action is a difference; the
  bits flip at key-press boundaries. Copying is a decent prior, not an identity.
- **And the hazard is visible in the field.** edwhu's own toy data is commit-then-switch
  with holds of 4–9 steps → **≈85% repeat rate**, so a pure copy head scores ~85% BC
  accuracy blind (§2.3). vijayabhaskar's committed action-shuffle numbers show their
  trained policy's mean moving 0.19/0.15 against an action RMSE of 0.30 — **~50–63%
  action-driven** (§2.4). **Neither author measures or interprets this.** The field is
  not immune; it is unmeasured, and it operates at a milder autocorrelation than ours.

**[HARD]**

### H4 — "BC is only an initialisation for them, and the final policy for us." **TRUE, and it is why a weak BC is survivable for them.**

Dreamer 4 Phase 3 optimises the policy **on rollouts the policy itself generates** —
*"unrolling the transformer with itself … actions a from the policy head"* (§4.4). That
training distribution **is** our failure regime. A copy-policy in imagination emits a
constant, earns no reward, gets negative advantage, and PMPO's sign-of-advantage update
(Eq 11) removes mass from it. **Their pipeline contains a built-in corrective for
precisely our failure mode.** All three V4 repos with a policy implement Phase 3
(§2.1, §2.3, §2.4); we implement none of it and ship the BC head.

DreamerV3 is the limiting case of the same argument: it has **no BC loss at all**
(§3.2c), so "copy `a_{t-1}`" lowers no loss anywhere, and the entropy bonus actively
penalises collapsing onto a repeat. **The copy pathology is an artifact of the BC phase
that Dreamer 4 introduced.** **[HARD]**

**Important honest caveat.** Their WM+BC row already scores 99/89/28/17 (§4.4). So it is
*not* true that "their BC is also broken and RL rescues it". H4 explains why they could
afford not to look, and why a residual copy bias costs them little. It does not by itself
explain why their BC is good.

### H5 — "Our labels made copying the only expressible answer." **TRUE, and ours alone.**

Two label-side facts with no counterpart in any Dreamer implementation:
1. The **sticky gate**: on ~90% of frames the target is "keep holding", expressible under
   `movement_mode='axis'` only by re-emitting the input (§5.6.1).
2. The **pre-first-click sentinel**: for the first ~61s of every game — the entire
   walk-out, the only fountain/base frames in the corpus — the label is "move to your own
   feet" (§5.6.2). BC was *taught* to copy during exactly the segment we then demo.

Plus no 50/50 relevance mixture, so BC trains uniformly on holds the paper's mixture
would have down-weighted (§5.6.3). **[HARD]**

### Verdict

**The right answer is H3 × H4, with H5 as our own aggravating factor. H0 and H1 are
false, and H2 is real but secondary.**

Stated as one causal chain:

> Our action is a **held goal**, so the BC target is bit-for-bit present in the model's
> own input on 90% of frames and copying is the *exact optimum* of the objective as
> written — not a shortcut the model took, but the answer we asked for (**H3**).
> Dreamer 4's action is instantaneous and half-delta, so the same architecture only
> yields a partial, lossy copy, and their labels never make copying uniquely correct
> (**H3**, **H5**). Where they still do lean on the action channel — and the field's own
> committed numbers say they do, at ~85% in edwhu's data and ~50–63% action-driven
> sensitivity in vijayabhaskar's — **it never becomes visible, because BC is only an
> initialisation and Phase 3 retrains the policy in exactly the self-fed closed loop that
> exposes copying** (**H4**). We stopped at Phase 2 and deployed it.
>
> **They are not protected. They are unmeasured and downstream-corrected. We are
> measured and terminal.**

**What follows, in priority order:**
1. **Fix the labels and the representation, together.** `prefirst_mode='heading'` **and**
   `--movement-action-mode none` (or run-level block dropout). The already-recorded
   conclusion — *"BOTH; the label fix alone leaves the copy channel dominant; the channel
   fix alone leaves the walk-out unsupervised"* — is confirmed by this survey. H3 and H5
   are separate causes and each needs its own fix.
2. **Take the free win now:** `play_live.py --movement-action-mode none` (27.5% → 10% of
   games walking to the wrong lane, no retraining).
3. **Adopt the field's action-token convention** (frame `t` carries `a_{t-1}`, target
   `a_t` at `n=0`) — not to fix the copy, but to delete the invented `a_t` slot at
   `agent_infer.py:222`. Four codebases do it this way and it removes a train/inference
   asymmetry we currently paper over.
4. **Add `action_shuffle_loss_ratio`** (nicklashansen, `train_dynamics.py:831-857`) next
   to our blind-table bar. It costs two no-grad forwards per log interval and catches the
   *inverse* failure (world model ignoring actions) that our blindness test cannot see.
5. **Treat Phase 3 as load-bearing, not optional.** Every Dreamer 4 implementation with a
   policy has it, the paper's design assumes it, and it is the mechanism that would have
   caught this bug for us. Deploying a BC head as a terminal policy is a deviation from
   Dreamer 4, not an implementation of it.
6. **Write our own test for the agent-isolation mask.** lucidrains' is broken
   (stale kwarg, CUDA-gated); nicklashansen sidesteps it. Only `vijayabhaskar-ev`'s
   `tests/test_gradient_isolation.py` works, and it is worth copying the shape of.

---

## Appendix — what could not be verified

- **The "trained results" columns are README-level evidence.** A score table is a claim,
  not an independent reproduction. Nothing in this survey was run.
- `nicklashansen/mmbench2` (the self-described "strictly better implementation") was not
  read.
- HuggingFace and paperswithcode were searched only shallowly; the HF artifacts listed
  surfaced via GitHub READMEs and the HF API search.
- `next-state/open-dreamer` was read via the GitHub API (file tree + three files), not
  cloned, because of its all-rights-reserved licence. The "no policy" conclusion rests on
  the complete file tree plus `models.py` / `training.py` / `actions.py`.
- The 19 sub-5-star Dreamer 4 repos were classified from metadata alone.
- No empirical `a_t == a_{t-1}` rate was measured in any external replay buffer; the
  edwhu ≈85% figure is derived from the generator's own `hold_min`/`hold_max` config.
