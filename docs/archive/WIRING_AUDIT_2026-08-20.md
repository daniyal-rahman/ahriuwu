# WIRING_AUDIT_2026-08-20 — seven-reviewer interface audit

Seven Opus reviewers over the whole repo, split by interface concern with deliberate
overlap: label semantics, checkpoint contracts, tensor layout, train/eval/infer
divergence, process orchestration, and two end-to-end chain-reasoners
(human-click→gradient, weights→keystroke).

Findings here are **measured**, not read. Where a reviewer proved something by
running it, the number is quoted. Confidence is CONFIRMED / LIKELY / SPECULATIVE
as the reviewer graded it.

**Four reviewers independently found the same Phase-3 defect.** That convergence is
the single strongest signal in this document.

---

## 0. The three things that block a demo

### 0.0 Mouse: the gadget works, the CALIBRATION is what's missing — CORRECTED
An earlier draft of this audit said mouse mode had "no gadget". Wrong. The Pi is a
combo gadget and does mouse; `setup_hid_combo.sh` (restored to `scripts/keysender/`)
creates `/dev/hidg1`. What is actually missing is **calibration of the absolute-XY
mapping** — the live box has a `calibrate_mouse.py` (+ `cursor_tpl.png`) that never
came back into the repo. Absolute HID is 0..32767 across the pointer's target surface,
and which surface that is (primary monitor vs virtual desktop) and any offset/scale
must be measured on the real rig, not assumed. Bring that script into the repo and
have `play_live` load its result.

### 0.1 The deployed rig is a FORK of this repo — CONFIRMED
`/mnt/storage/ahriuwu-live/` carries its own `scripts/`: **162** diff lines in
`play_live.py`, **71** in `agent_infer.py`, **101** in `hid_server.py`, **174** in
`hybrid_sender.py`, plus a `calibrate_mouse.py` that exists only there. It has no
`joint_noop` branch, no `--desktop`, no `--gate-bias`, and `expand_range` defaulting
True. Its checkpoint is `axis+gate`.

Consequence: **every fix made in this repo is absent from the thing that plays**, and
the current best checkpoints cannot run on it at all. Auditing the repo does not
describe the deployed system.

### 0.2 Greedy decode is a frozen agent — CONFIRMED, measured
| ckpt | temp | clicks/s | unique cells | casts |
|---|---|---|---|---|
| joint (vast) | **0.0** | **0.00** | **1** | **0%** |
| joint (vast) | 1.0 | 2.27 | 29 | 3.3% |
| axis+gate | **0.0** | **0.07** | **1** | **0%** |

`act_from_latent` defaults to `temperature=0.0`; so do `agent_infer` and
`eval_bc_sim`'s CLIs. `play_live`/`sim_replay`/`ab_checkpoints` default to 1.0.
Any run or eval taking the default measures a dead policy and reports plumbing OK.

### 0.3 The training corpus has no HUD — CONFIRMED
The replay corpus is rendered HUD-disabled (measured: no black band; bottom-25%
brightness 0.205 vs 0.215 whole-frame). Live games have the HUD. Nothing in the live
path masks it, so 20-30% of every live frame is content the tokenizer and policy have
never seen. `sim_replay.py:96` has a `--hud add` ablation built for this; it is not
wired into the live path.

---

## 1. Corrupts training

### 1.1 Frozen `action_embed` fitted to a DIFFERENT movement target — CONFIRMED
`rollout_stage/desktop_resume_8775_stripped.pt` has **no `movement_source` in its
args** — it predates the flag, so Phase 1 trained on legacy `cursor.screen`. Frozen
Phase-2 runs feed that embedding the click target. Measured on `NA1_5549995114`:
std 0.197/0.243 (clicks) vs 0.093/0.112 (cursor); **only 40.0% of frames share a
21-bin cell**; the cursor target never leaves (0,1) while the click target is clamped
to 0.0/1.0. `action_embed` is excluded from the trainable prefixes
(`train_agent_finetune.py:286`), so it **cannot adapt**.

Only `--unfreeze-backbone` runs escape. This is the likeliest reason the click-label
fix underdelivered on the frozen lineage.

### 1.2 Every launcher disables the documented sparse-cast fix — CONFIRMED
`--ability-pos-weight` defaults to 5.0 with the rationale "unweighted BCE collapses to
never press". **All six launchers pass 1.0**, and all five checkpoints on disk carry
1.0. Measured press rates: Q 3.6e-3, AA 8.4e-3, **R 3e-5 (1 positive in 33,714
frames), Stride 0**. Optimal constant logit for R is ~-10.4. The collapse is then
re-documented downstream in `agent_infer.py:75-78` as a fact of life.

### 1.3 13 fallback matches teach the opposite of the intent — CONFIRMED
Matches without `clicks.json` silently fall back to the cursor target AND a different
event definition. `select_val_matches` prefers click-backed games for val, so **all 13
land in train** (8.1% of frames).

```
[clicks]   bin changes with NO movement_event:      0
[fallback] bin changes with NO movement_event: 26,317 = 51% of bin changes
```
The trainer's stated invariant ("bin-change subset of event") is false on those frames.
`joint_noop` overwrites the target with NO_OP where the coordinate visibly jumped —
teaching "no new order" on 26,317 frames where one occurred. Train and val are also
scored against two different label definitions.

### 1.4 The held movement value is a stale screen coordinate — CONFIRMED
Correct as a target (only event frames scored) but it is ALSO the dynamics'
action-conditioning input on every frame, and the camera keeps moving. Measured over
100,621 hold frames (84% of all frames):

```
drift p50 0.024   p90 0.439   p99 0.807
off by >=1 of 21 cells: 46.6%    >=2 cells: 22.9%
```
Schema 3 re-projected every frame — right for the input, wrong for the target. The
schema-4 fix corrected the target and silently regressed the input. **The two
consumers want different things and the code gives them one array.**

### 1.5 Gradient checkpointing is inert — CONFIRMED
`load_frozen_dynamics` calls `dyn.eval()` and the dynamics is never returned to train
mode (only the heads are). The guard is `if self.gradient_checkpointing and
self.training` -> never taken. `create_dynamics(gradient_checkpointing=...)` is a
no-op.

Consequence: the "REQUIRED, OOMs without it" claim is wrong, the "~3x per step" cost
warning is wrong, and **the batch-size ceiling that motivated the Vast rental was an
artifact of a no-op flag**. This also explains the long-unexplained ~10 GiB gap between
an isolated probe (train mode, checkpointing active, 2.79 GiB) and the real trainer
(eval mode, full activations, 12.84 GiB).

### 1.6 Reward twohot buckets sized for returns, fed per-frame rewards — CONFIRMED
Measured over all 3,554,768 targets: **99.79% land in ONE bucket**; only 34 of 254
intervals ever occupied; bucket width 0.0236 vs median nonzero reward 0.001.
`ValueHead` inherits the same +/-3 range, where it is roughly correct — the two heads
should not share it.

---

## 2. Phase 3 is structurally broken

### 2.1 BC never trains MTP offset 0; Phase 3 reads ONLY offset 0 — CONFIRMED x4
BC runs `for n in range(1, mtp_length)` (n=0 dropped for the action-conditioning leak).
Measured on the real checkpoint after 99,421 steps:

```
heads.0.weight          norm = 0.0000e+00      heads.1..8          norm ~ 5.1
movement_heads.0.weight norm = 0.0000e+00      movement_heads.1..8 norm ~ 23
reward_head.heads.0     norm = 2.86e+01        (control: reward MTP does cover n=0)
```
Every Phase-3 read is offset 0. So its "behavior-cloned" policy is **Bernoulli(0.5)
on all 9 abilities** (4.5 presses per dreamed frame) and **uniform over 442 movement
classes**. `create_behavioral_prior` copies the same zeros, so the PMPO KL regularizes
toward uniform, not toward BC. **The entire Phase-2 policy is discarded at the
Phase-2 -> Phase-3 handoff.**

### 2.2 Phase 3 cannot load any surviving Phase-2 checkpoint — CONFIRMED
`train_imagination.py` restores `movement_gate` but never `movement_mode`,
`model_size`, `num_kv_heads`, `agent_layers`, or `soft_cap`. Verified by attempting
the load:
```
RuntimeError: register_tokens  [1,1,8,768] vs [1,1,8,512]     (model_size)
RuntimeError: movement_heads.0 [442,256]   vs [42,256]        (movement_mode)
```
Both live lineages are unreachable. `factorized_policy_kl` additionally sums a
`movement_dim` axis that a joint head does not have.

**Correction to prior claims in this repo:** joint_noop does NOT unblock Phase 3.
`log_prob` works; the rest of the imagination path is axis-only.

---

## 3. Corrupts measurement

- **`ab_checkpoints.cell_acc` is invalid** — CONFIRMED. It scores the n=1 output
  against frame f's target; measured `P(ev[f+1] | ev[f])` = 0.005-0.014, so ~99% of
  scored frames have target NO_OP. And the denominator sits inside `if gate`, making it
  |agent fired AND human fired| rather than |human fired| — so checkpoints with
  different firing rates are not comparable, defeating the script's purpose.
- **`sim_replay` — the "must PASS before a live session" gate — feeds frames ~85 apart**
  (`np.linspace`), so its 16-frame context spans ~70 s instead of 0.8 s. Its PASS bands
  were calibrated under that sampling. "READY for a live session" has never meant
  anything.
- **Three probes run the action-conditioned backbone with `actions=None`** — measured
  mean ability logit -10.46 (correct) vs -8.89 (actions=None), per-ability corr as low
  as +0.03. The `--ability-thresh -4.0` guidance was derived from this broken path;
  under real conditioning -4.0 still never casts.
- **`val/loss` is not comparable across steps** — divided by the moving training RMS.
  `rms['bc']` spans 0.557-1.438 across checkpoints. Components are raw and fine.
- **Offline rate metrics hardcode FPS=20** against a measured 12-17 fps loop, so every
  clicks/s figure on a live session is 15-60% optimistic.
- **20 fps is unreachable**: 81.4 ms model-only on an idle 5080 (resize 1.9 + tokenizer
  22.3 + dyn/heads 59.0). Rig recordings ran 17.0/17.0/17.2 fps.

---

## 4. Conventions — defaults, not rules

These are **guidelines**: the way to write new code here unless there's a reason not
to. They are not a style gate and nothing needs retrofitting for its own sake. They
exist because each one has already caused a real bug more than once, so following them
by default is cheaper than rediscovering why. Deviate when it's warranted; just do it
knowingly.

1. **Coordinate spaces — one type, one clamp owner.** Six spaces are live (normalized
   policy, label px 1280x720, region-relative px, desktop fraction, absolute HID
   0..32767, 352-frame px) with five separate clamps. Carry a normalized DESKTOP
   fraction end to end; never materialize region-relative pixels. This class has
   regressed twice and is the only one that makes the agent act confidently on a
   wrong coordinate.
2. **One movement-target contract, versioned and checked at load.** A
   `MovementTargetSpec` (source, schema, bins, clamp) written into every checkpoint;
   refuse a Phase-1 `use_actions=True` checkpoint whose args lack `movement_source`.
   Finding 1.1 is entirely this.
3. **Defaults live in exactly one place.** Six launchers override
   `--ability-pos-weight` 5.0 -> 1.0 and `--action-dropout` 0.0 -> 0.15. The parser
   default is documentation nobody runs. Adopt the `v7_train_args.sh` pattern.
4. **Deployment provenance.** Make the live box a git checkout; stamp commit,
   tokenizer sha, resolved (region, desktop, stream geometry), temperature, gate_bias
   and movement_mode into `meta.json`. Refuse `--inject hid` without a resolvable
   commit.
5. **Temperature.** 0.0 is a provably dead policy and is the default in three
   entrypoints. Remove the default so every call site must declare it.
6. **One binning helper.** `heads.discretize_movement` / `joint_encode` are the single
   source of truth; nothing outside `heads.py` may write `* 20` or `441`.
   (`eval_bc_sim`, `idm_value_test`, `ab_checkpoints` each reimplement it.)
7. **One mask polarity.** `causal_mask` is True=masked-out while the same class
   documents `mask` as True=can-attend. `state_mask` 1=valid, `cursor_valid`
   True=use-real-action, `movement_event` True=new-command. Name every boolean for
   what True means, and make True always mean keep/valid/attend.
8. **One projection implementation.** `pipeline.project` (int, viewport-clamped to
   None) vs `_Projection.project_norm` (float, unclamped) are the same camera with
   different failure modes.
9. **One frame transform.** `agent_infer.encode_frame`, `sim_replay.load_replay` and
   `pretokenize_replay_v7.load_frame` each write their own resize.
10. **A test that every parser `dest` is read somewhere in its file.** Three lines;
    would have caught `--gate-bias`/`--ability-thresh` being added and left inert.

None of these is worth a refactor on its own. Apply them to code you're already
touching.

---

## 5. Verified CLEAN (do not re-audit)

Proven correct empirically, mostly by round-trip or by planting distinguishable values:

- Bin encode/decode inverses over the full domain, incl. edges and clamping.
- `joint_encode`/`joint_decode`/`joint_to_unit`: exact over all 441 cells; NO_OP never
  collides; matches axis-mode centres to 3e-8; trainer passes (x, y) in the right order.
- `movement_logits.view(B,T,L,move_dim,bins)` axis order; `log_prob` gather and
  `sample` argmax agree.
- twohot/symlog: sum-to-1, <=2 non-zeros, `E[centers] == clamp(x)` to 2.4e-7;
  `symexp(symlog(x)) == x` over +/-1e4.
- Latent fold (B,512,16) <-> (B,32,16,16): exact inverses across all six call sites.
- Dynamics spatial fold and 2D RoPE index space agree (row-major, y=s//16, x=s%16).
- MTP slicing arithmetic `[:, :T-n, n]` vs `[:, n:]` aligns for all n.
- KV-cache temporal path: max|delta| 4.17e-07.
- Camera projection/inversion: <=1.24 px over 3977 self-frames + 7810 hero points;
  `cam_y` constant (1911.8) so the scale ambiguity is genuinely inert.
- Label timebase `gt == gt0 + i/20` exact to 3.6e-12; clicks on the same clock.
- Latent `frame_indices == arange(N)` on all 125 packs; slice contract holds.
- `STATE_TARGETS` <-> `StateHead` column order; `state_targets` resume guard works.
- `state_mask` polarity consistent producer/consumer.
- HID descriptor/packer/clamp mutually consistent (0..32767, 6 bytes, Absolute).
- Pi-side `_parse` handles both wire formats and every malformed input without
  dropping the socket.
- Every bracket-trick pgrep/pkill pattern matches its target and cannot match itself.

---

## 6. Open, unfixed at time of writing

Fixed during the audit: `setup_hid_combo.sh` restored to `keysender/`; range-expand
default actually OFF; `--desktop` mapping + hard refusal on ambiguous origin;
`--gate-bias`/`--ability-thresh` wired; `cfg.get("size")` -> `size_preset`.

Still open, roughly in fix order:
1. Sync or rebuild the deployed rig (0.1) — nothing else can be tested until this.
2. Remove the `temperature=0.0` defaults (0.2).
3. Un-`eval()` the dynamics when `--unfreeze-backbone` (1.5) — then re-measure batch.
4. Phase-3 offset-0 (2.1) and arch-flag restore (2.2).
5. Fix `ab_checkpoints` offset + denominator (3), then re-run the A/B.
6. Decide the frozen+clicks question (1.1): either always unfreeze, or re-pretrain
   Phase 1 on the click target.
7. HUD handling for live (0.3).
8. `--ability-pos-weight` (1.2) and the reward bucket range (1.6).
