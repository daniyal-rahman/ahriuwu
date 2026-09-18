# INFERENCE_FAILURE_ANALYSIS — why the agent plays badly, from first principles

**Date:** 2026-08-27 · **Subject checkpoint:** `data/phase2_bc_clicks/agent_finetune_latest.pt`
(step 102,420, `movement_source=clicks`, `movement_mode=axis`, `movement_gate=True`,
frozen backbone, `action_dropout=0.15`, `seq_len=16`, `stride=8` — read off the
checkpoint's own `args`). · **Tokenizer:** v7 `step 6000`, `512x16`.
· **Deployed launcher:** `ops/stage_desktop_standalone.sh:142-149`.

**Reported symptom.** "Garen doesn't even make it into top lane." In the replay corpus
the human walks top in essentially 100% of games from 0:15–0:45; live, the agent walked
into BOT lane.

**What this document is.** Part 1 decomposes the problem end to end before naming a single
hypothesis, and says at every link what is *measured* versus what is *assumed*. Part 2 is
**100 hypotheses** (H1–H100), each with a mechanism, a distinguishing prediction, a concrete
falsifying test and a cost, and each marked CONFIRMED / REFUTED / PARTIAL / OPEN. Part 3
ranks the open ones and gives the next ten tests to run.

**Prior art this builds on and does not re-derive.**
[`MOVEMENT_HEAD_BLIND_2026-08-26.md`](MOVEMENT_HEAD_BLIND_2026-08-26.md) (**read its
SECOND CORRECTION — that is the operative one**), [`BC_FIX_PLAN_2026-08-26.md`](BC_FIX_PLAN_2026-08-26.md),
[`WIRING_AUDIT_2026-08-20.md`](WIRING_AUDIT_2026-08-20.md),
[`DESIGN_DECISIONS.md`](DESIGN_DECISIONS.md), [`PAPER_DEVIATIONS.md`](PAPER_DEVIATIONS.md),
[`DEMO_RUNBOOK.md`](DEMO_RUNBOOK.md). MTP offset 0 and the Dreamer-implementation survey
are being worked separately; they are referenced here, not re-litigated.

**New measurements made for this document** (all offline, on real recorded latents and
real labels; scripts and numbers are quoted inline so they can be re-run):

| # | measurement | result |
|---|---|---|
| M1 | Human walk-out screen direction, frames 0–900, from `movement.heading_screen` in 146 games' labels | **blue +81.4° (R=0.985, n=71), red −170.8° (R=0.993, n=75)** — separation **107.8°**. Independently replicated on a different window (frame 200 → first click, 6+6 games): blue **+87.1°**, red **+173.0°**, separation **86°** |
| M2 | Coverage of `heading_screen` on the walk-out | 57.6% (blue) / 54.3% (red) of frames 0–900; ~80% full-game |
| M3 | Distribution of the *training labels* (click → camera → normalized, `replay_dataset.py:639`) over 7,865 clicks, 4 games | unimodal, mean **(0.495, 0.492)**, **5.3%** clamped to an edge, **2.0%** in x-bin 20, **3.0%** in y-bin 0 |
| M4 | The head's own predictive categorical on **post-first-click** windows (`sentinel` history) | unimodal at bin 10, **E[x]=0.501, E[y]=0.471** — matches M3 |
| M5 | The same head on **walk-out** windows t=30..1190 (`sentinel` = exactly what BC teacher-forced there) | **mode = bin (20, 0) = the top-right screen CORNER** on both sides; p(y=bin0)=0.316, p(x=bin20)=0.194 |
| M6 | Walk-out mean commanded direction by side, `sentinel` history, 1,873 windows | blue **+64.0°**, red **+69.4°** → **side separation 5.4°** vs the human's 107.8° |
| M7 | Same, `movement_action_mode=none` | blue **+86.7°** (5.3° from human), red **+156.0°** (33.2° from human) → separation **69.3°** |
| M8 | Gate firing rate on the walk-out vs the rest of the game, same head, same history mode | walk-out **E[p(fire)] = 0.0176 → 0.35 commands/s** (mean logit −6.61); full game **0.1140 → 2.28 commands/s** (mean logit −2.79). **6.5× fewer commands in the fountain.** Human ≈ 2/s |
| M9 | Per-game direction error, 39/40 closed-loop games, split by side | `held`: blue median **15.6°** (89% <45°), red median **115.5°** (20% <45°). `none`: blue **12.4°** (100% <45°), red **64.2°** (30% <45°) |
| M10 | `agent_temporal_pos` row norms in the deployed checkpoint | rows 0–15 ≈ 1.04–1.64 (**trained**), rows 16–255 ≈ 0.026 (**at init**) |
| M11 | Fraction of *executed* closed-loop commands landing where a live HUD element would be | `held`: **43.0% inside the minimap rectangle**, 73.5% on some HUD element (self-fed rollout, 40 games) |
| M12 | Deployed launcher flags | `play.sh` passes `--temperature 1.0` and nothing else; **no `--movement-action-mode`, no `--gate-bias`, no HUD mask** |
| M13 | `labels["team"]` availability and use | present on every game, **64 red / 61 blue** in the 125-game training corpus, computed by `lane_opponent.identify_teams` — and **never read by `replay_dataset.py`** |
| M14 | Pre-first-click window, corpus-wide (112 click-backed games, 3,266,368 frames) | **137,501 sentinel frames = 4.21%**; per-game window **p50 1,223 frames = 61 s**; first click **p50 62.4 s** (min 58.9, max 71.0) |
| M15 | Action-token geometry, deployed checkpoint | 1σ movement change → ‖Δ‖ **0.639**; **single AA press → 2.048**; whole-screen target sweep → 2.894; `‖emb(0.5,0.5) − no_action_embed‖` = **1.445** |
| M16 | `prefirst_mode` | **dead code** — no CLI flag, nothing sets the attribute, `_last_pre_valid` never leaves the dataset object |

---

# PART 1 — FIRST-PRINCIPLES ANALYSIS

## 1.0 The question, stated precisely

"Walk to top lane" is not one capability. It is a conjunction. Decompose it:

> At game time *t* in the first 45 s, the agent must (a) form a representation of the
> current screen from which (b) the map region it is standing in and (c) *which side it
> is playing* are recoverable; (d) map that to a direction; (e) decide that *now* is a
> moment to issue a command; (f) express that direction as a point on the 2-D screen;
> (g) sample it without the noise destroying it; (h) turn that point into a physical
> pointer position; (i) right-click it; and (j) have the game interpret that click as a
> move order to the corresponding *world* location.

Ten conditions. Nine of them can be individually correct while the agent still walks bot.
Every one of them is a separate place to look, and — this is the whole point of Part 1 —
each fails with a *different observable signature*.

## 1.1 The chain, link by link

Numbering used throughout Part 2.

```
  TRAIN SIDE                                        DEPLOY SIDE
  L1  replay render -> 352x352 PNG (HUD OFF)        L13 game -> gdigrab -> x264 -> UDP -> RGB
  L2  clicks.json (WORLD) -> camera -> screen        L14 HUD present; live domain
      -> clamp -> hold -> movement_event
  L3  v7 tokenizer encode -> 512x16 -> (32,16,16)   L3' same tokenizer, live frame
  L4  dynamics: noise tau~U(0.9,1), 266 tokens/frame
  L5  action conditioning: embed_actions +
      cursor_valid -> no_action_embed
  L6  agent token + agent_temporal_pos -> 4 cross-
      attention blocks -> agent_out (B,T,D)
  L7  PolicyHead: MLP -> 9 MTP offsets x
      (2 x 21-bin categorical + 1 gate logit)
  L8  gated mixture NLL, n=1..8, slice [:, :T-n, n]
  L9  optimisation (frozen backbone, 1.85 epochs)
  L10 checkpoint: args + dynamics_config + 3 state dicts
                                                    L11 GarenAgent: 16-frame ring buffer,
                                                        left-pad, self-fed act_buf,
                                                        read agent_out[:, -1], MTP n=1,
                                                        sample at T=1.0, gate -> fire
                                                    L12 bin -> centre -> normalized (x,y)
                                                    L15 (x,y) -> region px -> desktop fraction
                                                    L16 HID: corner slam -> relative travel
                                                        -> right-click
                                                    L17 the game turns the click into a
                                                        world move order and paths there
                                                    L18 we measure whether it went top
```

### L1 — Replay render → 352×352 PNG

**Known.** Frames on disk are already 352×352 (`/srv/nfs/datasets/lol_replays_16_9_772/*/frames/*.png`,
verified: `(352,352,3)`), squished from `screen_resolution [1280,720]` with the aspect
destroyed, HUD disabled (WIRING_AUDIT 0.3, measured: no black band, bottom-25% brightness
0.205 vs 0.215 whole-frame; visually confirmed here — no minimap, no ability bar, no
scoreboard; unit health bars *are* present).

**Assumed, never checked.** That the *replay/spectator* camera the frames were rendered
with has the same follow offset, zoom and FOV as a live camera-locked player camera. The
label pipeline's own constants put the focus champion at **(0.500, 0.481)**, not
(0.5,0.5) — the champion sits ~14 px above centre at 720p. Nobody has confirmed the live
camera does the same.

**Failure signature.** A camera mismatch is a *constant small bias* in every commanded
direction — it cannot produce a 115° error and it cannot flip lanes. **Not load-bearing
for this symptom.**

### L2 — Labels: `clicks.json` → screen target

**Known, and this is the single most consequential fact in the document.** `clicks.json`
stores **world** coordinates plus `game_t` (verified: first click of `NA1_5549995114` is at
`game_t = 67.216`). `_parse_movement_clicks` (`replay_dataset.py:560-694`) maps each click
to a frame, projects it through *that frame's* recovered camera, clamps to
`[0,1]` (`:639`), holds it until the next click, and sets `movement_event[i]=True` only on
click frames. Before the first click it writes the sentinel **(0.5, 0.5) with
`movement_event=False`** (`:665`, `prefirst_mode` defaults to `'sentinel'`).

Clicks never begin before ~60 s in **any** game (M14: first click p50 **62.4 s**, min 58.9,
max 71.0). So **for the entire walk-out — the exact segment being demoed — the supervision is
"issue no command; your target is your own feet."** M14: **137,501 frames = 4.21%** of the
112 click-backed games (MOVEMENT_HEAD_BLIND's 159,330 / 4.2% is the same quantity over 130
games), per-game window p50 **1,223 frames = 61 s**, and they are the *only* fountain/base
frames that exist.

**Known, and separately fatal.** `labels.json` carries a top-level `"team"`, and
`src/ahriuwu/data/lane_opponent.py:27-60` already *computes* every hero's side from spawn
position. `replay_dataset.py` never reads either. So the one bit that turns "walk to top" from
an ambiguous average into a well-posed two-way conditional is available for free and thrown
away (M13). The corpus is 64 red / 61 blue, so the two modes have equal support and averaging
them is maximally destructive.

**Known.** `prefirst_mode` — the documented fix for this exact bug — is **dead code**: no CLI
flag, nothing sets the attribute, and the `'exclude'` branch's `pre_valid` never leaves the
dataset object (M16). Every plan in this repo that says "retrain with `prefirst_mode='heading'`"
is currently un-runnable.

**M3 (new).** The post-first-click label distribution is a clean unimodal bump: mean
(0.495, 0.492), only **5.3%** of clicks clamped to an edge. So the corpus does *not* teach
corner clicking.

**Assumed.** That `heading_screen` — present in `labels.json` at
`frames[i].label.movement.heading_screen`, covering **54–58%** of walk-out frames (M2) —
is unusable. It is not: M1 shows it separates blue from red by **107.8°** with R≈0.99.
It is simply not read by the default label path.

**Failure signature.** "Trained to hold still with an ill-defined target" predicts:
(a) a heavily suppressed gate on fountain frames, (b) a categorical on those frames that is
*whatever the network extrapolates* rather than anything learned, and (c) no side
conditioning, because the label carries no direction. **All three are observed** — see
M5/M6/M8. This link is load-bearing.

### L3 — Tokenizer encode

**Known.** `agent_infer.encode_frame` (`agent_infer.py:167`) does
`cv2.resize(frame_rgb01, (352,352), INTER_AREA)` then `tok.encode`; the training path
(`pretokenize_replay_v7.load_frame:56-62`) does BGR→RGB, the same INTER_AREA resize (a
no-op, already 352²), `/255`. The 512×16→(32,16,16) fold is byte-identical in both
(`agent_infer._dyn_from_tok` vs `pretokenize_replay_v7.py:96`), and WIRING_AUDIT §5 has
verified it as an exact inverse across all six call sites. Re-encoding a clean stored
frame reproduces the stored latent to **1e-11** (`scratchpad/lane/hud.log`).

**Assumed.** That the latents carry map-region and side identity at all. Partially
checked: an ablated agent token still decodes champion map position at 2.687 nats
(MOVEMENT_HEAD_BLIND CORRECTION §3), so *something* positional is present.

**Failure signature.** A broken encode gives garbage everywhere, including in-lane —
it cannot give the clean unimodal in-lane behaviour of M4. **Refuted for this symptom.**

### L4/L5 — Dynamics and action conditioning

**Known.** Per-frame token set is `[256 latent | 8 register | 1 action | 1 cond]`
(`dynamics.py:_build_tokens`). The action token is appended at `dynamics.py:776`;
`cursor_valid=False` swaps in a **single learned vector** `no_action_embed`
(`dynamics.py:478`, `:566-576`) — so a dropped frame is *identifiable*, not merely absent.
The agent token cross-attends to that whole set (`dynamics.py:192-245`), i.e. **the policy
can read the action it is being conditioned on.**

**Known, measured.** Swapping the action history moves the command **150.9°**; swapping
the pixels moves it **41.0°** (MOVEMENT_HEAD_BLIND). Independently reproduced here: forcing
the history to `const:(0.9,0.5)` drags the head's own expectation to **E[x]=0.882** with
mode exactly bin 20 — a ~71% pass-through of the injected target.

**Failure signature.** A dominant copy channel in a *self-fed* loop predicts a direction
that (i) is arbitrary per game, (ii) is a property of the sampling seed rather than the
pixels, (iii) persists once entered. Measured: **28% lane agreement across seeds vs 34.6%
chance under `held`, versus R=1.000 determinism under `none`.** Load-bearing.

### L6 — Agent token and its temporal position

**Known (M10, new).** `agent_temporal_pos` rows 0–15 carry norms 1.04–1.64; rows 16–255 sit
at initialisation (0.0264). So the embedding for the *deployed* read position (15) **is**
trained — by the reward/aux heads, which do cover n=0 — even though the **policy** head is
never asked to produce an action from it (`train_agent_finetune.py:790` iterates `n>=1`,
`:850` slices `[:, :T-n, n]`, so at T=16 the policy only ever sees positions 0..14).
`scratchpad/pos_ablation2.py` compares the two positions in a full self-fed rollout and
finds no difference (131.9° vs 130.7°, fire rate 1.02/s vs 1.63/s).

**Failure signature.** Reading an untrained position would give a *degenerate* output —
one cell, or noise — not a structured distribution that tracks the corpus marginal in-lane.
**Largely refuted**, with the caveat that pos_ablation2's absolute numbers were taken
inside the sentinel window and only the *paired comparison* is meaningful.

### L7/L8 — Head and objective

**Known.** `axis` mode = two independent 21-bin categoricals + one gate logit per MTP
offset. Bin centre = `i/(bins-1)` (`heads.py:271`, `:432`) so bin 0 = coordinate 0.0 and
bin 20 = 1.0 — the *literal screen edge*. Loss is the sticky mixture
(`heads.py:346-366`): a **hold** frame contributes `logaddexp(log(1−g), log g + log p_cat)`,
which at the trained `g≈0.11` is dominated by `log(1−g)` and passes almost no gradient to
the categorical. Algebraic cap on the hold contribution: `−log(1−g) = 0.116`.

**Consequence, and it is the mechanism that explains M5.** On the 159,330 pre-first-click
frames the gate target is "never fire", so the categorical there is *effectively
unsupervised*. Its output on those frames is free extrapolation. Measured: the mode is the
**top-right corner** (M5).

**Known.** `heads.0` and `movement_heads.0` have weight norm exactly **0.0** — they are
zero-initialised (`heads.py:260-265`) and BC never touches n=0. Confirmed again here.
Phase 3 reads only offset 0 (WIRING_AUDIT 2.1). Irrelevant to Phase-2 inference, which
reads n=1; relevant to any Phase-3 plan.

### L9/L10 — Optimisation and checkpoint plumbing

**Known.** 102,420 steps ≈ 1.85 epochs; frozen backbone (`unfreeze_backbone` is not even
present in the saved args); `action_dropout=0.15` **per frame**, which leaves the copy
shortcut available on **98.13%** of frames because the held value is replicated across the
whole run; `evaluate()` forces `action_dropout=0`, so every val number ever reported was
measured with the shortcut fully open. `ability_pos_weight=1.0` (the documented sparse-cast
collapse). The checkpoint carries no `movement_action_mode`, so `agent_infer` defaults to
`'held'` (`agent_infer.py:106-107`).

**Assumed.** That `agent_infer` rebuilds exactly the model the trainer wrote. This is
partly *enforced* — `_load_state_dict_guarded` (`agent_infer.py:33-58`) raises on any
missing/unexpected key, and `use_actions` is taken from `dynamics_config` rather than
guessed. That guard is the reason several classic "wrong model" hypotheses are already
dead.

### L11 — The inference core

**Known.** 16-frame ring buffer; on reset the window is **left-padded by repeating the
oldest frame** (`agent_infer.py:~207`), so for the first 16 acted frames the model sees a
world with no motion — and the runbook itself records that a world model shown no change
predicts no change (one distinct movement target from 120 identical latents). The action
history is **self-fed** from the agent's own emissions; the newest slot repeats the
standing order with `fire=False`. τ is re-sampled `U(0.9,1)` every call, so two calls on
identical inputs differ. Sampling is at temperature 1.0 (greedy is a measured dead policy).

**Assumed.** That a 16-frame window is enough context to identify the map region. Never
tested against a longer window.

### L12 — Decode

**Known, verified.** Bin → centre → normalized `(x,y)` with the same top-left, y-down
convention the label pipeline uses (`replay_dataset._Projection.project_norm` emits
`y = 0.5 − …`). WIRING_AUDIT §5 verified bin encode/decode inverses over the full domain
including edges; `joint_encode`/`joint_decode` exact over all 441 cells.

**Crucially:** the normalized target is a *fraction of the frame*, and a uniform per-axis
squish is the identity on fractions — so the 352² aspect destruction cancels out on the
mouse path. It does **not** cancel in `_wasd_keys` (`play_live.py:265-273`), where
`atan2(-dy, dx)` on fractions of 1280 vs 720 turns a true 45° into 60.6°. WASD is not the
default. **Not load-bearing for the mouse path.**

### L13/L14 — Live capture and the domain gap

**Known.** The live frame carries a HUD; the corpus has none. Masking the HUD region
(21.7% of the frame) shifts latents by **2.6–5.2× a normal frame-to-frame step**
(cosine 0.89–0.92) — the model is off-distribution for a whole live game — but shifts the
*commanded direction* by only **8.2°**, and the lane stays TOP in 5–6 of 6 games. **The
bot-lane walk reproduces offline on clean replay latents with no HUD involved.**

**But** — and this has not been stated anywhere before — the HUD is not only an *input*
problem. It is an **output** problem. The policy emits a screen coordinate; a live screen's
bottom-right corner is the **minimap**, and a right-click on the minimap is a move order to
the corresponding **map** location, not to the world point under the cursor. M11: under the
deployed `held` configuration **43.0% of executed commands land inside the minimap
rectangle** in a self-fed rollout. On blue side the bottom-right of the minimap is bot
lane. This is a mechanism that converts an edge-heavy click distribution directly into
"walks to bot", and it is untested.

### L15/L16 — Screen → desktop → HID

**Known.** `play_live.py:298-299` maps the normalized target onto the capture region;
`:290` divides by the desktop size and calls `move_click`. With `--source udp` and no
`--desktop`, `_dw,_dh = region` and that division is an exact identity, so the shipped
configuration is self-consistent. The click is a **right**-click end to end
(`hybrid_sender.py:204,296` → `hid_server.py:230`, `BTN["right"]=2`).

**Assumed, and the audit rates these the most likely coherent mis-aim:**
1. The corner slam is sized from `span` (`hybrid_sender.py:251`, `int(649/127)+4 = 9`
   reports ≈ 2254 px/axis). `span` is a property of the *capture region*, not of the
   pointer surface. On any pointer surface wider than ~2254 px, or a multi-monitor virtual
   desktop, the slam never reaches a clamp and **every** click carries a constant offset.
2. `--desktop`'s help text still describes the retired absolute-HID design; following it
   (e.g. `--desktop 1920x1080` with a 1280×720 stream) compresses every click into the
   top-left ⅔×⅔.
3. `mouse_calibration.json` is flagged `"provisional": true`, records `stream_size` and
   **nothing ever validates it against the running geometry**.
4. `AA` is a no-op in mouse mode by default (`--attack-key` defaults to `None`).

**Failure signature.** A constant offset or a compression is *coherent and sustained* —
exactly the class this symptom belongs to. It is distinguishable from every model-side
hypothesis by one test: the offline rollout reproduces the bot walk *without any HID at
all*, so HID cannot be the **only** cause — but it can still be an independent second one
that a model-side fix would not remove.

### L17 — The game

**Assumed, entirely.** That a right-click at the commanded pixel produces a move order to
the world point under it. Fails if the pixel is on a HUD element (swallowed, or reinterpreted
— minimap), if the click lands on a unit (becomes an attack order), if it lands on the
shop panel, or if the champion is mid-cast. None of this has ever been observed on real
hardware; `DEMO_RUNBOOK.md:375-386` says so explicitly.

### L18 — Evaluation

**Known.** The lane verdicts come from **offline closed-loop rollouts on recorded replay
latents**, not from live sessions. `sim_replay` — the documented pre-session gate — feeds
frames ~85 apart via `np.linspace`, so its 16-frame context spans ~70 s instead of 0.8 s;
its PASS bands were calibrated under that sampling (WIRING_AUDIT §3). `ab_checkpoints.cell_acc`
is invalid. The checkpoint was chosen on **liveness**, not skill (DEMO_RUNBOOK §1a).

**So one whole branch stays open:** we may be measuring the wrong thing, and the live
failure may not be the offline failure.

## 1.2 What a *coherent sustained wrong walk* rules in and out

This is the discriminative core of the analysis. Different faults leave different traces.

| observed live behaviour | what it implies | what it rules out |
|---|---|---|
| **coherent, sustained, wrong** (this case) | a *persistent* wrong input, a *persistent* wrong bias, or a *self-reinforcing* loop | per-frame noise; a decode that is wrong only occasionally |
| jitter / thrash | sampling temperature, gate mis-calibration, or a per-frame coordinate bug | a stale-input or copy-loop explanation |
| freezing / standing still | closed gate, stale frames, dead greedy decode, dropped clicks | anything that produces movement |
| random flailing, no direction | untrained head, wrong checkpoint, garbage latents | anything that reproduces the corpus marginal in-lane |
| *right* direction, wrong distance | bin quantisation, edge clamping, aspect | anything about side or region |

**A coherent wrong walk therefore has exactly four viable generators:**

- **(G1) A self-reinforcing loop.** The head reads its own last target and re-emits a
  perturbation of it. Bounded on `[0,1]` and clamped at the bin edges, this is a random
  walk with sticky boundaries — its stationary distribution piles up in the **corners**.
  Once in a corner the direction stops changing. *Signature:* direction is a property of
  the seed, not the pixels; distribution mass concentrates at bins 0 and 20 over time.
  *Measured:* both. Under `held` self-feeding, 53.8% of executed x-values and 47.1% of
  executed y-values are in bin 20.
- **(G2) A persistent constant bias.** The head has learned one direction and applies it
  regardless of side/region. *Signature:* blue and red produce the *same* screen direction.
  *Measured:* M6 — **5.4° separation** under the trained regime, against the human's 107.8°.
- **(G3) A persistent wrong input.** The live frame is systematically off (HUD, brightness,
  staleness) so the model believes it is somewhere else. *Signature:* the offline rollout on
  clean latents would be **fine**. *Measured:* it is not fine — the bot walk reproduces
  offline. **Demoted, not eliminated:** it can still be an additive second cause live.
- **(G4) A persistent wrong transform after the model.** A constant offset/scale between
  the commanded point and where the pointer lands, or a click reinterpreted by the UI.
  *Signature:* the model's commanded direction is right and the champion's is wrong.
  *Never measured — there is no instrumented live session that logs both.*

G1 and G2 are confirmed. G3 is measured small offline. **G4 is completely open**, and the
minimap mechanism (L14) makes it a live candidate that current evidence cannot exclude.

## 1.3 KNOWN vs ASSUMED, condensed

| link | measured | assumed, never checked |
|---|---|---|
| L1 render | HUD off, 352², aspect squished | spectator vs player camera offset |
| L2 labels | clamp 5.3%, first click p50 62.4 s, sentinel on 4.21% of frames = the whole walk-out; `team` present but unread; `prefirst_mode` dead code | that `heading_screen` is unusable (it is not — M1/M2); that side must be inferred from pixels (it need not be — M13) |
| L3 tokenizer | re-encode fidelity 1e-11; fold exact | that latents encode *side* |
| L4/L5 actions | copy channel beats vision ~3:1 | that `no_action_embed` for a whole window is in-distribution |
| L6 agent token | pos rows 0–15 trained (M10); read position makes no measured difference | that a 16-frame window identifies the region |
| L7/L8 head/loss | gate ≈0.11; hold-branch cap 0.116; n=0 heads are zero | that the categorical is supervised on hold frames (it is barely) |
| L9 optimisation | dropout ineffective (98.13%); val measured with dropout off | that 1.85 epochs is enough |
| L10 plumbing | strict-ish load guard; `use_actions` from config | that the deployed tree equals the repo (it drifted 10 days once) |
| L11 inference | τ, MTP n=1, sampling all match training | that left-padding 16 copies of frame 0 is harmless |
| L12 decode | bin↔coord verified over the domain | — |
| L13/L14 live | HUD shifts latents 2.6–5.2× a step, direction 8.2° | **that a commanded pixel on the HUD still produces a world move order** |
| L15/L16 HID | shipped config is self-consistent; right-click verified in code | span vs pointer surface; calibration vs geometry; **nothing tested on hardware** |
| L17 game | — | **everything** |
| L18 eval | offline rollouts, 40 games | that offline lane == live lane |

## 1.4 The leading account (what I think is actually happening)

Stated as a causal chain, each step measured:

1. Clicks start at ~60 s, so **the walk-out has no movement supervision at all**; BC saw
   `movement_event=False` and the sentinel target on every one of those 159,330 frames.
2. The gate therefore learned "barely fire in the fountain": **0.35 commands/s** on walk-out
   frames against **2.28/s** later in the same game, under the identical history mode BC
   teacher-forced there (M8) — a **6.5×** reduction, against a human's ~2/s. *For the first
   minute the agent is close to mute.*
3. Because the gate is shut, the **categorical on those frames is effectively unsupervised**
   (the hold branch of the mixture passes ~no gradient). Its extrapolated mode is the
   **top-right screen corner** (M5), on both sides.
4. With no direction in the label, the head has **no side conditioning** on the walk-out:
   **5.4°** blue/red separation vs the human's 107.8° (M6). Its single learned direction is
   up/up-right, which is approximately blue-side top and roughly **115° wrong on red side**
   (M9: red median error 115.5°, only 20% of games within 45°).
5. The few commands it does emit are fed back into its own action history, where the copy
   channel dominates vision ~3:1; on a bounded, clamped grid that is a random walk with
   sticky corners, so the executed targets concentrate at bins 0/20 and the direction locks.
6. Live, an edge/corner command is a **screen-corner pixel**, and on a HUD-bearing screen
   the bottom-right corner is the minimap — where a right-click is a move order to a
   *different part of the map entirely*. This is unverified and is the largest single
   remaining unknown.

**The one-line version:** the walk to lane is the *only* behaviour in the corpus that was
never labelled, and the deployed launcher still runs the configuration measured to be the
worst of the three.

## 1.5 The most embarrassing thing in this document

`ops/stage_desktop_standalone.sh:142-149` writes `play.sh` as:

```bash
$PY $AHRIUWU/scripts/play_live.py \
  --phase2-ckpt $BC --tokenizer-ckpt $TOK \
  --inject ${INJECT:-dry} --hid-host $PI \
  --movement-mode ${MOVE:-mouse} --target-fps ${FPS:-20} --temperature 1.0 "$@"
```

There is no `--movement-action-mode`. Commit `ab8eb7c` ("live: cut the movement action
channel — the policy was there all along") added the flag and measured it moving executed
lane from 17/12/11 to 28/8/4 over 40 games — **and the launcher that actually runs still
defaults to the losing setting.** Same for `--gate-bias`, which WIRING_AUDIT already
flagged once as "added and left inert". This is a one-line fix that has been available,
measured, and un-deployed for a day.

**A close second.** `prefirst_mode` — the label fix on which fix option F of
`BC_FIX_PLAN_2026-08-26.md` and fix 2 of `MOVEMENT_HEAD_BLIND_2026-08-26.md` both depend —
**cannot be switched on.** `replay_dataset.py:665` reads it via
`getattr(self, "prefirst_mode", "sentinel")`, and across the entire repo that string appears
only in that `getattr`, in one comment, and in two documents. There is no CLI flag and nothing
ever sets the attribute; the `'exclude'` branch writes `pre_valid` to `self._last_pre_valid`
(`:690`), which is overwritten on every match parse and never read by `__getitem__`. Anyone
who believed they had tried the label fix changed nothing.

**A third.** `play_live.py:544` prints the checkpoint's trained action mode as
`getattr(agent, '_ckpt_movement_action_mode', 'held')` — and `agent_infer` never sets that
attribute, so the line always says "held" no matter what the checkpoint says. That is the
exact shape of the bug this project keeps rediscovering: a log line that reports the default
instead of the truth.


---

# PART 2 — 100 HYPOTHESES

**How to read an entry.** *Statement* → *Mechanism* → *Prediction* (the observable that
distinguishes it from its neighbours) → *Test* (+ rough cost) → *Prior* and *Status*.

**Status key:** `CONFIRMED` (measured true) · `REFUTED` (measured false) · `PARTIAL`
(measured, real, but shown not to be sufficient) · `OPEN`.

**Distinctness rule applied:** two entries are merged if the same single experiment
settles both. Where a family shares a test, it appears once with the family named.

---

## A. Evaluation methodology — "we are measuring the wrong thing"

**H1. The live bot-lane observation is n=1 and the offline distribution is broad, so what
we saw may be a draw, not a bias.**
*Mechanism:* under `held` the executed lane is TOP 17 / MID 12 / BOT 11 over 40 games. A
single live game landing BOT has probability ≈0.275 under the model's own distribution.
*Prediction:* five live games under the identical configuration produce ~2 TOP, ~1 MID,
~1–2 BOT rather than 5 BOT.
*Test:* five live games, log the executed lane. ~30 min of rig time.
*Prior:* **HIGH** that this partly explains the report; **LOW** that it fully does — the
distribution itself is already wrong (M6/M9).
*Status:* **OPEN.**

**H2. The "closed-loop" offline rollout is not closed-loop: the agent's commands never
change the frames it sees.**
*Mechanism:* `scratchpad/lane/rollout_batch.py` feeds recorded latents of the *human's*
trajectory while self-feeding only the action history. So the observation stream is
counterfactual — the champion on screen keeps walking top while the agent commands bot.
Compounding observational drift, which is the classic BC failure, is invisible.
*Prediction:* a rollout where the observations follow the agent (dreamed, or in-engine)
degrades faster and further than the offline number suggests.
*Test:* dream the rollout through the dynamics at d=1 for 200 frames and score the same
lane metric; or a live session. ~2 h GPU (dream) / ~30 min rig.
*Prior:* **HIGH** — this is a genuine hole in every lane number quoted anywhere.
*Status:* **OPEN.** All offline lane figures in this repo inherit the caveat.

**H3. The lane classifier's angular partition is wrong or side-inverted.**
*Prediction:* applying it to the human's own labels would not give 36/3/0.
*Test:* already run — human scores TOP 36 / MID 3 / BOT 0. *Status:* **REFUTED.**

**H4. `sim_replay`, the documented pre-session gate, has never meant anything.**
*Mechanism:* it feeds frames ~85 apart via `np.linspace`, so the 16-frame context spans
~70 s instead of 0.8 s; the PASS bands were calibrated under that sampling.
*Test:* read the code. *Status:* **CONFIRMED** (WIRING_AUDIT §3).

**H5. `ab_checkpoints.cell_acc` — the metric used to compare checkpoints — is invalid.**
*Mechanism:* scores the n=1 output against frame *f*'s target (P(ev[f+1]|ev[f]) = 0.005–0.014,
so ~99% of scored frames have target NO_OP), and its denominator sits inside `if gate`.
*Status:* **CONFIRMED** (WIRING_AUDIT §3). It was correctly not used for the deploy pick.

**H6. The acceptance bar (event-frame CE < 4.357 nats) is cleared by the broken checkpoint.**
*Mechanism:* a Markov table fitted on the 24 *training* games scores 4.1214; the deployed
head scores 4.1212 on the same 6-game val split. The 4.357 figure came from a table fitted
on 5 held-out games.
*Status:* **CONFIRMED** (first CORRECTION; independently reproduced here from
`scratchpad/label_baselines.json` + `scratchpad/check_ce_held.log`).

**H7. No metric in the repo scores the behaviour that is failing.**
*Mechanism:* event-frame CE scores only post-first-click frames, where the copy crutch is
available; the walk-out has zero events by construction, so it contributes zero rows.
*Prediction:* a checkpoint could improve CE while getting worse at walking to lane.
*Test:* add the M6/M9 direction metric (per-game commanded direction vs the champion's own
heading, reported separately for blue and red, frames 30–1190) to `evaluate()`. ~2 h.
*Prior:* **HIGH.** *Status:* **CONFIRMED** (stated in MOVEMENT_HEAD_BLIND §Fixes 3).

**H8. Every validation number ever reported was measured with the action shortcut fully
open.**
*Mechanism:* `evaluate()` forces `action_dropout = 0`.
*Status:* **CONFIRMED.**

**H9. Every reported clicks/s on a live session is 15–60% optimistic.**
*Mechanism:* offline rate metrics hardcode FPS=20 against a measured 12–17 fps loop.
*Status:* **CONFIRMED** (WIRING_AUDIT §3).

**H10. No session has ever logged the commanded target and the champion's resulting world
displacement together, so model-side and post-model causes are formally unseparable.**
*Mechanism:* `Recorder` (`play_live.py:336-360`) writes the 352² model view and the action
stream but nothing about the game's response.
*Prediction:* with both logged, the correlation between commanded screen direction and
realised champion heading immediately separates G2 (model bias) from G4 (bad transform).
*Test:* add a periodic read of the champion's screen position (template match or the HP-bar
CV reader) to the recorder; one live session. ~3 h to build, 20 min to run.
*Prior:* **HIGH diagnostic value, low prior of being itself the bug.** *Status:* **OPEN** —
**the single highest-value missing instrument.**

**H11. The checkpoint on the rig was chosen on liveness, not skill, and is the frozen
lineage whose `action_embed` cannot adapt.**
*Mechanism:* DEMO_RUNBOOK §1a picks `bc_clicks` on clicks/s, unique cells, cast rate — all
of which reward a noisier policy.
*Prediction:* a checkpoint that scores worse on liveness could score better on direction.
*Test:* run the M9 direction metric on all five checkpoints on disk. ~3 h GPU.
*Prior:* **MEDIUM.** *Status:* **OPEN.**

---

## B. Data and labels

**H12. The walk-out has no movement supervision at all: `_parse_movement_clicks` writes
the sentinel (0.5,0.5) with `movement_event=False` until the first click, and clicks never
start before ~60 s.**
*Mechanism:* `replay_dataset.py:627-642` writes `movement=(0.5,0.5)`, `event=False`;
`:665` (`prefirst_mode`) defaults to `'sentinel'`. Measured corpus-wide over 112
click-backed games / 3,266,368 frames: **137,501 sentinel frames = 4.21%**, per-game window
**p50 = 1,223 frames = 61 s** (min 1,155, max 1,390), first-click game time **p50 62.4 s**
(min 58.9, max 71.0). They are the only fountain/base frames in existence. Under the gated
loss (`train_agent_finetune.py:889-891`) every one of them lands in the *hold* branch, which
actively trains the gate toward 0.
*Prediction:* the gate should be near-shut and the categorical near-unsupervised **on
exactly those frames and not elsewhere**.
*Test:* already run — M8 (0.35 cmd/s on t=30..1190 vs 2.28 cmd/s on t≥1500) and
M5 (mode = screen corner on the walk-out, mode = bin 10 in-lane).
*Prior:* **HIGH.** *Status:* **CONFIRMED — the root cause of the specific symptom.**

**H13. `heading_screen` is available on the walk-out and carries the side signal, and the
default label path does not read it.**
*Mechanism:* `frames[i].label.movement.heading_screen` exists; `prefirst_mode='heading'`
(`replay_dataset.py:665-693`) would substitute it but is not the default.
*Prediction:* it should cover a large fraction of walk-out frames and separate blue from red.
*Test:* already run — **M2: 57.6% (blue) / 54.3% (red) coverage of frames 0–900; M1: blue
+81.4° (R=0.985) vs red −170.8° (R=0.993), separation 107.8°.**
*Prior:* **HIGH.** *Status:* **CONFIRMED available and informative** (untested as a training
signal).

**H14. Off-viewport clicks are clamped to the edge, so edge bins mean two different things
("go far that way" and "go exactly to the screen edge").**
*Mechanism:* `replay_dataset.py:639`.
*Prediction:* a measurable fraction of labels at exactly 0.0/1.0.
*Test:* already run — **M3: 5.3% of 7,865 clicks clamped; 2.0% in x-bin 20; 3.0% in y-bin 0.**
*Prior:* **LOW as a cause** — the label distribution is unimodal and centred, so the model's
50–73% corner mass (M5/M11) is *not* inherited from the labels.
*Status:* **PARTIAL** — real, but far too small to explain the head's behaviour.

**H15. 18.6–19.1% of real commands land in the same 21-bin cell as the previous one, so the
categorical cannot express them; only the gate records that anything happened.**
*Test:* measured — `scratchpad/label_baselines.log`: identical bin on 19.1% of consecutive
click targets. *Status:* **CONFIRMED.** *Prior of being causal here:* **LOW** (it costs
resolution, not direction).

**H16. The held target is a *stale* screen coordinate as the dynamics' action **input**,
because the camera keeps moving between clicks.**
*Mechanism:* WIRING_AUDIT 1.4 — over 100,621 hold frames, off by ≥1 of 21 cells on 46.6%,
≥2 cells on 22.9%. The same array serves two consumers that want different things.
*Prediction:* re-projecting the held world point each frame makes a_{t+1} ≠ a_t and
degrades the copy shortcut.
*Test:* schema change + 5k-step run. ~2.2 h GPU.
*Status:* **CONFIRMED as a defect**, **OPEN** as a fix.

**H17. 13 fallback matches teach the opposite of the intent and all land in train.**
*Mechanism:* matches without `clicks.json` fall back to the cursor target *and* a different
event definition; 26,317 bin changes (51%) carry no `movement_event`.
*Status:* **CONFIRMED** (WIRING_AUDIT 1.3). 8.1% of frames.

**H18. Train and val are scored against two different label definitions.**
*Mechanism:* `select_val_matches` prefers click-backed games for val, so the 13 fallback
games are train-only.
*Status:* **CONFIRMED** (WIRING_AUDIT 1.3). Distinct from H17: this one invalidates the
*comparison*, H17 corrupts the *gradient*.

**H19. The val split is side-imbalanced (4 red / 2 blue), so a side-conditioning failure is
invisible in every val metric.**
*Mechanism:* `val_matches_resolved` in the checkpoint args resolves to
`5549995114(red), 5550417257(red), 5551063460(blue), 5551782551(red), 5552261591(blue),
5552945604(red)`.
*Prediction:* a pooled val metric would look ~fine while red-side behaviour is 115° wrong.
*Test:* already implicit in M9 — the pooled `held` median error is 45.2° while red alone is
115.5°.
*Prior:* **MEDIUM.** *Status:* **CONFIRMED as a measurement hole.** Nothing in
`select_val_matches` (`train_agent_finetune.py:574-614`) balances sides. See H95 for the
related resume hazard.

**H20. The frame record covers only part of each game, so a large share of clicks is dropped
and the label stream is truncated.**
*Mechanism:* measured — `NA1_5549995114`: 1344/2349 clicks after the frame record ends;
`NA1_5550417257`: 1256/2262. Frames cover `gt` 1–567 s of a longer game.
*Prediction:* no effect on the walk-out (the drops are at the *end*).
*Prior:* **LOW** for this symptom. *Status:* **CONFIRMED but not causal here.**

**H21. Click→frame timebase is off by one.**
*Test:* already run — `|champion_world[i] − click.hero_xz|` is minimised at offset 0
(p50 8.6 world units, vs 22.5 at −1 and 25.8 at +2). *Status:* **REFUTED.**

**H22. `movement_event` fires on engine pathing recomputation, not only on player clicks,
inflating the event rate and teaching the gate to fire on non-decisions.**
*Mechanism:* `clicks.json` records changes of the engine's "last commanded destination".
*Prediction:* the measured event rate would exceed a plausible human APM.
*Test:* already run — 0.0970 (train) / 0.1005 (val), i.e. **1.94 clicks/s**, exactly a
Masters-level move-command rate. *Status:* **REFUTED** as an inflation problem.

**H23. A horizontal-flip augmentation would destroy side conditioning.**
*Mechanism:* the map is (approximately) diagonally symmetric; a flip maps blue-top onto
red-something.
*Test:* audit every augmentation path. Done.
*Status:* **REFUTED.** `ReplayLatentSequenceDataset.__getitem__` (`replay_dataset.py:979-1019`)
applies no augmentation at all — it slices precomputed latents. The only augmentation in the
repo is `ImageSequenceDataset._apply_augment` (`data/dataset.py:146-172`): brightness,
contrast, saturation, hue, gamma, Gaussian noise — **no flip, no rotation, no crop** — and it
is tokenizer-only (`v7_train_args.sh:34`). BC never touches pixels, so a flip is not even
reachable at Phase 2. **Side conditioning was not destroyed by augmentation; it was never
provided.**

**H24. Side is genuinely under-determined from a 16-frame camera-following window once the
champion leaves the fountain, because the terrain is mirror-symmetric.**
*Mechanism:* the camera follows the champion, so most frames show generic jungle/lane; the
disambiguating structures (fountain, base, inhibitors, the blue/red crystal) appear only in
the first seconds.
*Prediction:* a supervised probe for `team` from the agent token should be near-perfect on
fountain frames and much weaker mid-walk.
*Test:* logistic probe on the agent token for `team`, split by game time. ~1 h, no training.
*Prior:* **MEDIUM-HIGH.** *Status:* **OPEN — high diagnostic value.**

**H25. The side label exists, is computed, and is thrown away — so the only two-way
discrimination the task needs is never given to the model.**
*Mechanism:* `labels.json` has a top-level `"team"` (measured **64 red / 61 blue** over the
125-game training corpus — essentially balanced), and
`src/ahriuwu/data/lane_opponent.py:27-60` (`identify_teams`) already computes every hero's
side from spawn position (`_BLUE_SPAWN_MAX = 3000.0`). Only the opponent's *name* escapes
(`replay_dataset.py:494`, for aux HP targets). No side/team/lane key is ever added to the
`actions` dict (`replay_dataset.py:998-1009`) or to `_build_tokens` (`dynamics.py:760-803`),
and `use_game_time=False`. **The HUD — and therefore the minimap, the one always-visible side
cue — is rendered off in every training frame** (`MANIFEST.json`: `"hud": "DISABLED"`).
*Prediction:* the model averages the two modes and produces one direction.
*Test:* M6 — **5.4° separation** on the walk-out, against 86–108° in the human labels.
*Prior:* **HIGH.** *Status:* **CONFIRMED.** The cheapest conceivable fix — a one-bit side
embedding — has never been tried. Distinct from H24 (can side be *recovered from pixels*)
and from H34 (is it in the *latents*): this one is that it is available for free and discarded.

**H26. Even with side available, the corpus contains no counterexample: Garen is TOP in
145/146 games, so "go top" is learnable only as a prior, never as a discrimination.**
*Mechanism:* there is no game where the correct answer is "go bot", so the model cannot
learn *why* top is right; it can only learn the marginal.
*Prediction:* the marginal is what it learned — a single direction applied on both sides.
*Test:* M6 already shows exactly that (5.4° separation).
*Prior:* **HIGH as an explanation of the mechanism**, though the fix is the label, not the data.
*Status:* **CONFIRMED consistent.**

**H27. Nothing in the objective rewards being in lane, and Phase 3 never ran, so the policy
is pure imitation with no preference for top.**
*Mechanism:* reward is `1e-3·Δgold − 0.2·death`; Phase 3 is one degenerate 11-batch smoke run.
*Prior:* **HIGH as context, LOW as a fix path** (Phase 3 is structurally blocked).
*Status:* **CONFIRMED.**

**H28. The walk-out frames are heterogeneous — fountain idle, shopping, recall, first walk —
and the sentinel label lumps them together.**
*Mechanism:* M2 shows only ~55% of walk-out frames have a heading at all; the rest are
genuinely stationary. A single label for both teaches "usually do nothing".
*Prediction:* conditioning on `speed > 0` sharpens the heading target.
*Test:* re-fit the label statistics on moving frames only. ~15 min.
*Prior:* **MEDIUM** — matters for the *fix*, not the diagnosis. *Status:* **OPEN.**

**H29. Camera recovery fails on early frames (champion off-viewport), silently corrupting
the projection exactly where it matters.**
*Mechanism:* `_recover_cameras` fills frames with no `champion_screen` from the nearest
frame that has one (~0.05%).
*Prediction:* clustered failures near the fountain.
*Test:* count missing `champion_screen` in frames 0–900 across the corpus. ~20 min.
*Prior:* **LOW** (0.05% overall, and the projection is verified to ≤1.24 px on 3,977 frames).
*Status:* **OPEN, low.**

---

## C. Tokenizer / representation

**H30. Tube masking made HP bars and other small high-frequency detail unrecoverable.**
*Status:* **CONFIRMED as a deviation** (PAPER_DEVIATIONS 1.1; TOKENIZER_REVIEW measured the
generic-mean-bar artifact). *Prior of causing the lane symptom:* **LOW** — direction does not
live in HP bars.

**H31. The mask-ratio curriculum meant v7 never trained in the paper's MAE regime, so the
"spatial consistency" benefit was not purchased.**
*Mechanism:* mean input mask ≈0.1–0.2 against the paper's `U(0,0.9)`; a historical
`RESET_SCHEDULE` bug re-ramped the curriculum on every requeue.
*Status:* **CONFIRMED** (PAPER_DEVIATIONS 1.2). *Prior for this symptom:* **MEDIUM** —
spatial consistency is exactly what a direction readout needs.

**H32. 352×352 → 484 patch tokens (vs the paper's 960) throws away the resolution a
direction readout needs.**
*Status:* **CONFIRMED as a deviation**, impact **unmeasured**. *Prior:* **MEDIUM.**

**H33. The tokenizer bottleneck is under-used: effective rank ≈31 of 8,192, and 9 days of
continuation bought +0.28 dB — it is data/objective-limited, not capacity-limited.**
*Status:* **CONFIRMED** (TOKENIZER_REVIEW:123-126). Implies tokenizer *scale* is not the
lever.

**H34. The latents do not encode which side you are on.**
*Mechanism:* if side is absent from `z`, no downstream fix can produce side-conditioned
behaviour.
*Prediction:* a linear/MLP probe for `team` from raw latents should fail.
*Test:* probe raw v7 latents (not the agent token) for `team`, per game-time bucket. ~1 h,
no training. **Distinct from H24**, which probes the agent token — the pair localises the
loss to perception vs the agent block.
*Prior:* **MEDIUM-LOW** — the ablated agent token still decodes champion *map position* at
2.687 nats, and map position determines side.
*Status:* **OPEN — cheap, and it forks the whole fix plan.**

**H35. The 16×16 "spatial" grid is a reshape fiction — 512 perceiver latents are global
readers — so the dynamics' 2D RoPE imposes a geometry the latents do not have, damaging any
*directional* readout specifically.**
*Mechanism:* PAPER_DEVIATIONS 2.9. Direction is the one quantity that needs a consistent
spatial frame.
*Prediction:* permuting the 256 latent slots before the fold should barely change the
policy's output if the grid is fictional, and should destroy it if it is real.
*Test:* slot-permutation ablation on a trained model. ~1 h, no training. **Sharp and cheap.**
*Prior:* **MEDIUM.** *Status:* **OPEN.**

**H36. The 1280×720 → 352×352 aspect squish makes the latent space anisotropic, so "45° in
the world" is 60.6° in the image and the head must undo a non-uniform warp.**
*Mechanism:* the squish is a per-axis scale; angles are not preserved. The *normalized
target* is unaffected (fractions are invariant), but the *visual evidence* the head reads is
warped.
*Prediction:* directional errors should be larger near ±45° than near the axes.
*Test:* bin M9's per-game errors by the human's true angle. ~20 min on existing data.
*Prior:* **LOW-MEDIUM.** *Status:* **OPEN, cheap.**

**H37. The tokenizer has never seen a HUD, so live latents are out of distribution for the
whole game.**
*Test:* already run — masking 21.7% of the frame moves latents **2.6–5.2× a normal
frame-to-frame step** (cosine 0.89–0.92) but the commanded direction only **8.2°**, and the
lane stays TOP in 5–6/6.
*Status:* **PARTIAL — real and large in latent space, small in behaviour.** Cheapest close:
mask the HUD to black before `encode_frame` (`sim_replay.py:96` already has the mask).

**H38. Excluding ~450 h of YouTube from dynamics training left the world model
under-trained on map geography it could have learned for free.**
*Status:* **CONFIRMED as a deviation** (PAPER_DEVIATIONS 6.2; unlabeled:labeled is 0:1 vs
the paper's ~25:1). *Prior for this symptom:* **MEDIUM**, untestable cheaply.

**H39. The `tanh` bottleneck saturates on visually extreme frames (the fountain is the
brightest, most saturated scene in the game), compressing exactly the frames that carry the
side signal.**
*Prediction:* the fraction of bottleneck units at |tanh| > 0.99 should be markedly higher on
fountain frames than mid-lane frames.
*Test:* encode 200 fountain and 200 lane frames, histogram the pre-tanh activations. ~30 min.
*Prior:* **LOW-MEDIUM**, but it is cheap and nobody has looked.
*Status:* **OPEN.**


---

## D. Dynamics and action conditioning

**H40. The movement-action input dominates vision, so the policy is a lookup on its own last
command.**
*Mechanism:* the action token is appended to frame *t*'s set (`dynamics.py:776`) and the
agent token cross-attends to that set (`dynamics.py:192-245`); BC predicts a_{t+n}, and on a
hold frame a_{t+1} is bit-for-bit a_t on 89.9% of frames.
*Prediction:* swapping the history should move the command far more than swapping the pixels.
*Test:* already run — **history swap 150.9° vs pixel swap 41.0°**; reproduced here as a
~71% pass-through (forcing history `(0.9,0.5)` gives E[x]=0.882, mode = bin 20).
*Status:* **CONFIRMED.** Note the channel is never absent in training:
`replay_dataset.py:1007-1009` sets `cursor_valid = match_id.startswith("NA1_")`, and **every
match in the corpus is `NA1_*`** — so it is True on every frame of every training sequence,
and only the trainer's dropout/mode flags ever clear it.

**H41. `--action-dropout 0.15` cannot remove the shortcut because it is per-frame while the
held value is replicated across the whole run.**
*Mechanism:* the answer is hidden only if every frame since the last click is dropped:
`P = p^(j+1)`. At p=0.15 the shortcut survives on **98.13%** of frames.
*Status:* **CONFIRMED.** Requires run-level (block) dropout, or a representation where the
answer lives in exactly one token.

**H42. `no_action_embed` is a single learned vector, so a dropped frame is *identifiable* —
the model learns "look at a neighbour" instead of being forced to guess.**
*Mechanism:* `dynamics.py:478`, `:566-576`.
*Prediction:* dropout at any per-frame rate under-performs a scheme that makes the dropped
frame indistinguishable (e.g. resampling a plausible action).
*Test:* 5k-step run with a noise-substituted rather than sentinel-substituted action. ~2.2 h.
*Prior:* **MEDIUM.** *Status:* **OPEN.** Distinct from H41: H41 is about *rate*, this is
about *detectability*.

**H43. `movement_action_mode=none` at inference is out of distribution — BC only ever saw
`cursor_valid=False` per-frame at p=0.15, never for a whole 16-frame window.**
*Mechanism:* the model has never seen `no_action_embed` in all 16 action slots.
*Prediction:* `none` should behave erratically in ways unrelated to the intended fix — e.g.
a collapsed gate.
*Test:* already visible — the closed-loop fire rate under `none` drops to **0.37 cmd/s** (from
0.94), and the open-loop walk-out rate is **0.57 cmd/s** (mean logit −4.84). It also
*improves* direction (blue 12.4°, red 64.2°).
*Quantified:* on the deployed checkpoint, `‖emb(0.5,0.5)‖ = 1.357`,
`‖no_action_embed‖ = 0.452`, `‖emb(0.5,0.5) − no_action_embed‖ = 1.445` — a **2.3×** larger
perturbation to the action token than a 1σ real movement change (0.639). Substituting it in
all 16 slots is a large, never-trained displacement.
*Prior:* **HIGH that it is OOD, MEDIUM that the OOD-ness costs anything measurable.**
*Status:* **PARTIAL — the fix works despite being OOD; the OOD tax shows up as a mute agent.**

**H44. The frozen `action_embed` was fitted to a *different* movement target (legacy
`cursor.screen`), and it cannot adapt because it is excluded from the trainable prefixes.**
*Mechanism:* `desktop_resume_8775_stripped.pt` has no `movement_source` in its args; measured
std 0.197/0.243 (clicks) vs 0.093/0.112 (cursor), and **only 40.0% of frames share a 21-bin
cell**. `train_agent_finetune.py:286` excludes it.
*Status:* **CONFIRMED** (WIRING_AUDIT 1.1). The deployed checkpoint is on that lineage
(no `unfreeze_backbone` in its args).

**H45. Ability presses are summed into the *same* action token as movement, so the agent's
own (mis-calibrated) cast sampling perturbs the movement conditioning.**
*Mechanism:* `dynamics.embed_actions` adds all nine ability embeddings to the movement
embedding. Live, abilities are sampled at T=1.0 from logits ≈ −3.5…−5, giving ~2%/frame each
— roughly 18% of frames press *something*, against training rates of Q 3.6e-3, AA 8.4e-3,
R 3e-5.
*Quantified on the deployed checkpoint.* Movement enters through a `Linear(2, 768)`
(`dynamics.py:474`) — a rank-2 subspace plus bias — while each ability contributes a free
768-dim vector:

| perturbation to the shared action token | norm |
|---|---|
| 1σ movement change (0.20, 0.24) | 0.639 |
| **single AA press** | **2.048** |
| Q / E / Recall press | 0.86–1.04 |
| sweeping the click target across the whole screen | 2.894 |

**A single AA flag moves the action token 3.2× further than a typical movement change, and
70% as far as sweeping the target across the entire screen.** AA is also the noisiest label —
it is re-derived in `replay_dataset.py:862-901` from `spell.endswith("attack")` because the
shipped `action.type` is wrong and the `.rofl` files were deleted.
*Prediction:* forcing all abilities to zero in the history should shift the commanded
direction measurably.
*Test:* rerun the walk-out probe with the agent's sampled abilities vs all-zeros. ~1 h GPU.
(The existing probes already zero them — so the *rollout* and the *probe* differ on exactly
this axis, which is one candidate explanation for their differing corner mass: rollout
p(x=bin20)=0.372 at frames 0–20 vs probe 0.194.)
*Prior:* **MEDIUM.** *Status:* **OPEN — cheap and currently a confound between two of our
own measurements.**

**H46. τ is re-sampled `U(0.9,1)` on every `act` call, so identical inputs give different
answers, and the noise is a live contributor to the seed-dependence.**
*Mechanism:* `agent_infer.act_from_latent` draws fresh τ each call.
*Prediction:* freezing τ (or averaging over several τ draws) should raise cross-seed lane
agreement above the measured 28%.
*Test:* rerun the 40-game rollout with τ fixed at 0.95 and with a 4-sample τ average.
~4 h GPU (or ~1 h on the probe windows).
*Prior:* **MEDIUM** — under `none` the *distribution* is already deterministic (R=1.000), so
τ noise is not the main term, but it is free to remove.
*Status:* **OPEN.**

**H47. The dynamics was pretrained with `use_agent_tokens=False`, so the agent blocks are
randomly initialised at Phase 2 and get only ~1.85 epochs of a 4-layer stack.**
*Mechanism:* the production dynamics config records `use_agent_tokens: False`; M10 shows
`agent_temporal_pos` rows 16–255 still at init.
*Prediction:* agent-block depth/width should show a steep return curve.
*Test:* train two 5k-step variants at `agent_layers` 4 vs 8. ~4.5 h GPU.
*Prior:* **MEDIUM.** *Status:* **OPEN** (this is the "cheap architectural fork" the earlier
correction flagged).

**H48. The model has no clock. `use_game_time` exists and is OFF, so "walk to lane" — a
purely time-indexed behaviour — has to be inferred from pixels alone.**
*Mechanism:* `use_game_time: False` in the production dynamics config; `task_id` is never
passed either (`num_tasks=1`).
*Prediction:* a policy conditioned on game time should recover the walk-out trivially, since
the behaviour is ~deterministic given t < 60 s.
*Test:* add game-time conditioning to a 5k-step BC run, or simply probe whether game time is
decodable from the agent token. Probe ~1 h; run ~2.2 h.
*Prior:* **MEDIUM-HIGH.** *Status:* **OPEN — one of the cheapest large wins available.**

**H49. The dynamics' pretraining context regime does not match inference: context length
equals batch length (the paper requires batch > context), and 30% of pretraining sequences
were treated as independent frames while inference always uses full context.**
*Mechanism:* PAPER_DEVIATIONS 2.5 and §4 "start frames". Together these mean the model was
never trained to *rely* on a full 16-frame temporal window.
*Prediction:* the policy's output should be near-invariant to the older frames of the window.
*Test:* replace window positions 0..11 with the newest frame and re-measure the commanded
direction on the walk-out probe. **~1 h GPU, no training.**
*Prior:* **LOW-MEDIUM.** *Status:* **OPEN.** (Merged from two entries during the audit — the
single ablation above settles both.)

**H50. Shortcut forcing is off, so the model is a K=64 diffusion-forcing transformer being
run at d=1 — fine for BC (one forward, no rollout), fatal only for Phase 3.**
*Status:* **CONFIRMED as a deviation, REFUTED as a cause of the BC inference symptom** —
`agent_infer` never rolls out.

---

## E. Agent token, heads, MTP

**H51. Deployment reads agent-token position 15, which the *policy* head is never trained on
(BC slices `[:, :T-n, n]` with n≥1, so at T=16 it only supervises positions 0..14).**
*Mechanism:* `train_agent_finetune.py:790`, `:850`; `agent_infer.py:282` reads `[:, -1:]`.
*Prediction:* reading position 14 instead should behave differently.
*Test:* already run — `scratchpad/pos_ablation2.py`: pos 15 median 131.9° / 19.1% <45°;
pos 14 median 130.7° / 13.8% <45°. And M10 shows `agent_temporal_pos[15]` **is** trained
(norm 1.135, vs 0.026 for untrained rows) — by the reward/aux heads, which do cover n=0.
*Status:* **REFUTED as a first-order cause.** Caveat: the absolute numbers were taken inside
the sentinel window and only the paired comparison is meaningful.

**H52. MTP head 0 has weight norm exactly 0.0 and Phase 3 reads only offset 0.**
*Status:* **CONFIRMED** (`heads.0.weight` and `movement_heads.0.weight` norm 0.0 in the
deployed checkpoint, re-verified here). **Irrelevant to Phase-2 inference** (which reads
n=1), **fatal to Phase 3**. Being investigated separately.

**H53. The axis factorisation `p(x)·p(y)` cannot represent a bimodal directional belief and
puts mass on impossible corners.**
*Mechanism:* if the head believes "left OR up", the marginals are bimodal at
x∈{0,·} and y∈{0,·}, and the product assigns mass to the top-left corner — a direction the
head never intended. This is a *structural* generator of exactly the corner mode M5 observes.
*Prediction:* a `joint_noop` head trained on the same data should put far less mass on the
corners for the same marginals; and the observed corner mass should exceed what the
*conditional* label distribution supports.
*Test:* (a) compare the corner mass of `data/phase2_from_vast/vast_step90000.pt`
(`joint_noop` lineage) on the same walk-out windows — **~1 h GPU, no training**;
(b) compute the KL between the observed factorised joint and the empirical joint of the
labels.
*Prior:* **HIGH.** *Status:* **OPEN — the single best cheap structural test in this document.**

**H54. Bin 0 and bin 20 are the *literal screen edges* (`centre = i/(bins−1)`), so the two
most-probable classes decode to a screen corner rather than a direction.**
*Mechanism:* `heads.py:271`, `:432`.
*Prediction:* the command distribution should be edge-heavy in a way the labels are not.
*Test:* already run — **M5: p(y=bin0)=0.316, p(x=bin20)=0.194 on walk-out frames; M3: labels
put 3.0% and 2.0% there.** Under self-feeding it is worse: 53.8% of executed x-values in bin 20.
*Prior:* **HIGH.** *Status:* **CONFIRMED as a mismatch.** A polar (angle × distance)
parameterisation, or dropping the edge bins, removes the failure mode by construction
(DESIGN_DECISIONS §4 already recommends polar).

**H55. The gate and the categorical share one MLP trunk, so training the gate to be shut on
fountain frames actively drags the trunk representation away from anything the categorical
could use there.**
*Mechanism:* `PolicyHead.gate_logits` recomputes `self.mlp` (`heads.py:316-323`) — same
weights as `forward`.
*Prediction:* a two-trunk head should keep a sane categorical on walk-out frames even with a
shut gate.
*Test:* 5k-step run with separate trunks. ~2.2 h GPU.
*Prior:* **MEDIUM.** *Status:* **OPEN.** Distinct from H56 (which is about gradient
magnitude, not shared parameters).

**H56. On hold frames the mixture's hold branch passes almost no gradient to the categorical,
so the categorical is effectively unsupervised on ~90% of frames and on 100% of the walk-out.**
*Mechanism:* `heads.gated_movement_log_prob` returns `logaddexp(log(1−g), log g + log p_cat)`;
at the trained g≈0.11 the first term dominates and caps the hold contribution at
`−log(1−g)=0.116`.
*Prediction:* the head's output on frames that are always holds should be free extrapolation,
not the sentinel it was nominally given.
*Test:* already run — **M5: the mode is the top-right corner, not bin 10 (=0.5).**
*Status:* **CONFIRMED — this is the mechanism that turns H12 into a corner-clicking policy.**

**H57. Movement-head weight norms decay monotonically with MTP offset (7.04 at n=1 → 5.59 at
n=8), i.e. the far offsets are the least confident, yet they carry 7/8 of the movement
gradient.**
*Mechanism:* the loss sums n=1..8 with equal weight.
*Prediction:* down-weighting far offsets should sharpen n=1.
*Test:* 5k-step run with offset weighting ∝ 1/n. ~2.2 h GPU.
*Prior:* **MEDIUM.** *Status:* **OPEN** (norms measured here; the causal claim is untested).

**H58. One agent token per frame with a single cross-attention query is too narrow a channel
for spatial reasoning.**
*Mechanism:* each `AgentTokenBlock` performs exactly one 1-query attention read over 266
tokens (`dynamics.py:192-245`); four layers = four reads total.
*Prediction:* multiple agent queries per frame (or a small set of learned queries) should
improve directional readout more than adding depth.
*Test:* architecture variant, 5k steps. ~2.2 h GPU.
*Prior:* **MEDIUM.** *Status:* **OPEN.** Distinct from H47 (depth) — this is *width of the
read*.

**H59. `agent_layers=4`, `num_register_tokens=8`, `hidden_dim=256` are unexamined defaults.**
*Status:* **CONFIRMED unexamined** (DESIGN_DECISIONS §9). *Prior:* **LOW individually.**

**H60. The aux `StateHead` (weight 0.5) competes for the same four agent layers against
targets the frozen latents may not contain (cross-game HP R² ≈ 0.16).**
*Prediction:* setting `--aux-state-weight 0` should not hurt, and may help.
*Test:* 5k-step ablation. ~2.2 h GPU. *Prior:* **LOW-MEDIUM.** *Status:* **OPEN.**

**H61. 21×21 spends bins on distance where the decision is direction; a 10° heading error
matters and a 200-unit distance error does not.**
*Status:* **CONFIRMED as an unexamined decision** (DESIGN_DECISIONS §4). *Falsifier already
stated there:* per-bin confusion concentrated among angular neighbours. **Not yet run** —
~30 min on existing rollout data. *Prior:* **MEDIUM.**

---

## F. Loss and objective

**H62. `bc_movement` was compared against a no-gate reference, so the apparent 8×
improvement was the gate discovering that 90% of frames need no prediction.**
*Mechanism:* 6.089 is the ungated uniform CE; `bc_movement` is the gated mixture. Against the
right references the trained model is **0.035 nats better than the best blind stateless
predictor (0.7716)** and **0.117 worse than an oracle gate with a uniform categorical (0.6196)**.
*Status:* **CONFIRMED.**

**H63. The gate is supervised by encoding `movement_event` as a fake +1 bin difference in
`prev_idx` rather than being passed explicitly.**
*Mechanism:* `train_agent_finetune.py:886-892` sets `p_idx = where(ev, (m_idx+1)%bins, m_idx)`.
*Prediction:* correct in expectation, but it makes the "transition" branch score
`log g + log p_cat(target)` on frames where target == prev — a legitimate configuration the
mixture was not designed for.
*Test:* pass `movement_event` explicitly to `gated_movement_log_prob` and re-derive; check the
loss is numerically identical. ~1 h, no GPU.
*Prior:* **LOW as a bug, MEDIUM as a maintenance hazard.** *Status:* **OPEN**; already listed
as fix #5 in MOVEMENT_HEAD_BLIND.

**H64. The Phase-2 objective drops the video-prediction loss the paper keeps.**
*Status:* **CONFIRMED as a deviation** (PAPER_DEVIATIONS 4.2), but it follows from the frozen
backbone and cannot affect a frozen model's features. *Prior for this symptom:* **LOW.**

**H65. Ability BCE is unweighted (`--ability-pos-weight 1.0` in every launcher), so casts
collapse to never-press.**
*Mechanism:* measured press rates Q 3.6e-3, AA 8.4e-3, **R 3e-5 (1 positive in 33,714
frames)**, Stride 0.
*Status:* **CONFIRMED.** Affects casting, not walking — but see H45 for the coupling.

**H66. Reward twohot buckets are sized for returns and fed per-frame rewards: 99.79% of
targets land in one bucket, 34 of 254 intervals ever occupied.**
*Status:* **CONFIRMED** (WIRING_AUDIT 1.6). Affects Phase 3 and the reward head, not the
movement command.

**H67. No term in the objective penalises commanding an unreachable or off-screen target.**
*Mechanism:* the categorical is free to put mass on bins that decode to the screen edge, and
nothing costs it.
*Prediction:* an explicit penalty (or simply removing bins 0 and 20) would collapse the corner
mode without touching anything else.
*Test:* re-normalise the deployed head's logits over bins 1..19 at inference and re-run the
lane metric — **zero training, ~1 h GPU.**
*Prior:* **MEDIUM-HIGH for the *symptom*, though it treats the effect not the cause.**
*Status:* **OPEN — cheapest possible behavioural intervention.**

**H68. The gate's operating point was never calibrated: it sits at ≈ −5 (0.7%/frame,
~0.2 commands/s) against the 2–5/s it was meant to learn.**
*Mechanism:* the gate is trained by maximum likelihood on an event rate of 0.097; the
*decision* rate at deploy is a separate quantity nobody set.
*Prediction:* `--gate-bias` should move the rate without moving the direction.
*Test:* already partly done offline; **never used live** — `play.sh` does not pass it.
*Prior:* **HIGH that it matters for "does anything happen at all".**
*Status:* **CONFIRMED as un-deployed.**

**H69. The loss weights the walk-out at 4.2% of the gradient, matching its share of frames
rather than its share of what we care about.**
*Prediction:* upweighting the first 1,200 frames of each game (with real heading labels)
would move the metric far more than its 4.2% suggests.
*Test:* 5k-step run with `prefirst_mode='heading'` + a 5× weight on that window. ~2.2 h.
*Prior:* **MEDIUM-HIGH.** *Status:* **OPEN.**

**H70. `val/loss` is not comparable across steps (divided by the moving training RMS, which
spans 0.557–1.438 across checkpoints).**
*Status:* **CONFIRMED** (WIRING_AUDIT §3). Components are raw and fine.

---

## G. Optimisation and training regime

**H71. The run is 1.85 epochs and most of the learning happened in the first few thousand
steps; the remaining ~50k steps moved `move` from 0.750 to 0.727.**
*Status:* **CONFIRMED** (BC_FIX_PLAN "Costs"). *Implication:* long runs are not the lever;
the objective/labels are.

**H72. The LR schedule is WSD with `decay_steps=0` — warmup then flat forever — so the final
weights are an arbitrary point on a plateau rather than an annealed optimum.**
*Mechanism:* `lr_schedule=wsd, warmup_steps=2000, decay_steps=0` in the checkpoint args.
*Prediction:* a short decay to zero from the current checkpoint should sharpen the categorical
and reduce seed-to-seed variance without changing what it knows.
*Test:* 2k-step linear decay from the deployed checkpoint, then re-run the lane metric.
~1 h GPU.
*Prior:* **MEDIUM.** *Status:* **OPEN — cheap, and it directly targets the seed-dependence.**

**H73. `seq_len=16, stride=8` means every frame appears in two overlapping windows, halving
the effective sample count and correlating the gradient.**
*Prior:* **LOW.** *Status:* **OPEN.**

**H74. Gradient checkpointing is inert (`load_frozen_dynamics` leaves the dynamics in
`eval()`), so the batch-size ceiling that motivated the Vast rental was an artifact.**
*Status:* **CONFIRMED** (WIRING_AUDIT 1.5). Not a cause of the symptom; it is a cause of
*fewer experiments having been run*.

**H75. No EMA of the weights, so the deployed checkpoint is a single noisy iterate.**
*Prediction:* averaging the last k checkpoints should reduce cross-seed variance.
*Test:* average `agent_finetune_epoch_001/002/latest`, re-run the lane metric. ~1 h GPU,
no training.
*Prior:* **LOW-MEDIUM.** *Status:* **OPEN, very cheap.**

**H76. The frozen backbone means only ~31.7 M of 146 M parameters ever adapt to the BC task.**
*Status:* **CONFIRMED as a deviation** (PAPER_DEVIATIONS 4.1). One unfrozen run exists
(`phase2_parity`) and was *worse* on liveness; it has never been scored on direction.
*Test:* run M9 on `data/phase2_parity/agent_finetune_latest.pt`. ~1 h GPU. **OPEN.**

**H77. Checkpoint selection had no skill metric at all.**
*Status:* **CONFIRMED** (DEMO_RUNBOOK §1a is explicit about it). See H11 — distinct: H11 is
about the *lineage* chosen, this is about the *selection procedure*.

**H78. There is no seed-averaging or multi-seed training, so all conclusions rest on one
training run.**
*Prior:* **LOW-MEDIUM.** *Status:* **OPEN**, expensive.

---

## H. Checkpoint and configuration plumbing

**H79. The deployed launcher does not pass `--movement-action-mode none`, so the rig still
runs the configuration measured to be the worst of the three.**
*Mechanism:* `ops/stage_desktop_standalone.sh:142-149` writes `play.sh` with only
`--inject`, `--hid-host`, `--movement-mode`, `--target-fps`, `--temperature 1.0`.
*Prediction:* the live run reproduces the `held` numbers: 42.5% TOP, 27.5% BOT, lane at
chance across seeds.
*Test:* read the file. Done.
*Prior:* **CERTAIN.** *Status:* **CONFIRMED. This is the highest-value one-line fix in the
repo and it is a day old.**

**H80. `--gate-bias` is likewise not passed, so the live rate is pinned at the checkpoint's
raw calibration (~0.2 cmd/s), which under `none` falls further to ~0.37 cmd/s.**
*Status:* **CONFIRMED** — and WIRING_AUDIT already flagged this exact class once
("added and left inert").

**H81. The checkpoint carries no `movement_action_mode` key, so `agent_infer` silently
defaults to `'held'` — the default is invisible unless you read the constructor.**
*Mechanism:* `agent_infer.py:106-107`; verified — the key is absent from the deployed
checkpoint's args.
*Prior:* **MEDIUM as a hazard.** *Status:* **CONFIRMED.**

**H82. `play_live.py` prints `trained as: {getattr(agent, '_ckpt_movement_action_mode', 'held')}`
but `agent_infer` never sets that attribute, so the log line always says "held" regardless of
what the checkpoint says.**
*Mechanism:* grep — `_ckpt_movement_action_mode` appears only at the read site.
*Prediction:* a future checkpoint trained with `none` would be mis-reported at deploy.
*Test:* read the code. Done.
*Prior:* **CERTAIN as a defect, LOW as a current cause.** *Status:* **CONFIRMED** — exactly
the class of thing that hides a real mismatch.

**H83. The deployed rig is (or was) a fork of the repo, so fixes made here are absent from
the thing that plays.**
*Mechanism:* 162 diff lines in `play_live.py`, 71 in `agent_infer.py`, 101 in `hid_server.py`,
174 in `hybrid_sender.py`, plus a `calibrate_mouse.py` that existed only there.
*Status:* **CONFIRMED historically** (WIRING_AUDIT 0.1); `stage_desktop_standalone.sh` now
stamps a commit. **OPEN whether the rig has actually been re-staged since `ab8eb7c`.**
*Test:* `ssh desktop cat /mnt/storage/ahriuwu-live/VERSION`. 1 minute. **Do this first.**

**H84. The tree that was staged was DIRTY, so the run is not reproducible and the running
code is not any commit.**
*Mechanism:* `stage_desktop_standalone.sh` writes `dirty=yes` but proceeds. The NFS repo has
uncommitted modifications right now.
*Prior:* **MEDIUM.** *Status:* **OPEN** — check the same `VERSION` file.

**H85. A wrong checkpoint is deployed.**
*Prediction:* the preflight's `checkpoint identity` line would show a step/sha other than
102420 / the expected sha.
*Test:* run the preflight. 2 minutes.
*Prior:* **LOW** — the staging script copies and shas both files, and the preflight prints
them — but it costs nothing to check and the repo has had exactly this class of failure.
*Status:* **OPEN.**

**H86. A state-dict mismatch silently ran a partly-random model.**
*Mechanism:* the historical failure — a bare `strict=False` dropped `action_embed` whenever
`use_actions` was mis-derived.
*Status:* **REFUTED for the current code** — `_load_state_dict_guarded` (`agent_infer.py:33-58`)
raises on any unexplained missing/unexpected key, and `use_actions` is read from
`dynamics_config`, not guessed.

**H87. The tokenizer used at deploy differs from the one the training latents were built
with.**
*Mechanism:* a different tokenizer sha means the dynamics sees a latent space it was never
trained on.
*Test:* the preflight prints `tok sha=35154dca2ad0c786`; compare against the latents'
`INFO.md`. 2 minutes.
*Prior:* **LOW.** *Status:* **OPEN, trivially checkable.**

**H88. `--stream-size` and the Windows `-video_size` disagree, so the frame is decoded with
the wrong geometry.**
*Mechanism:* the reader reshapes raw bytes to `(h, w, 3)` with no validation beyond the
preflight's one-shot check.
*Prediction:* a catastrophic, obvious visual scramble — not a coherent wrong walk.
*Test:* the preflight already checks geometry once.
*Prior:* **LOW.** *Status:* **OPEN, guarded.**


---

## I. Inference loop and sampling

**H89. `prefirst_mode` — the documented fix for the root-cause label bug — is DEAD CODE. It
cannot be turned on.**
*Mechanism:* `replay_dataset.py:665` reads `getattr(self, "prefirst_mode", "sentinel")`.
Across the whole repo the string `prefirst_mode` appears **only** in that `getattr`, in a
comment in `play_live.py:539`, and in two docs. **There is no CLI flag and nothing ever sets
the attribute.** Worse, the `'exclude'` branch computes `pre_valid` (`:667-689`) and stores it
as `self._last_pre_valid` (`:690`) — a *dataset*-level attribute overwritten by every match
parse, never placed in the `md` dict (`:456-465`) and never read by `__getitem__`
(`:979-1019`). So `prefirst_mode='exclude'` would silently do nothing at all.
*Prediction:* every plan that says "retrain with `prefirst_mode='heading'`" is currently
un-runnable, and anyone who "tried it" changed nothing.
*Test:* grep. Done.
*Prior:* **CERTAIN.** *Status:* **CONFIRMED. The second-most embarrassing entry in this
document, and it silently invalidates fix option F of `BC_FIX_PLAN` and fix 2 of
`MOVEMENT_HEAD_BLIND`.** Needs a CLI flag, a plumb-through of `pre_valid` into `md`, and a
consumer in `__getitem__` before any label-fix run is meaningful.

**H90. Sampling at temperature 1.0 destroys a preference the distribution already has.**
*Mechanism:* the head's categorical is broad; one multinomial draw per axis per fired command
turns a distributional preference into a coin flip, and the self-fed loop then locks in
whatever was drawn.
*Prediction:* the *distribution* should prefer TOP deterministically while the *executed*
lane is at chance across seeds.
*Test:* already run — under `none`, distribution agreement **R = 1.000 / 100% same lane**
against executed **R = 0.709 / 55%**; under `held`, executed **28% vs 34.6% chance**.
*Prior:* **HIGH.** *Status:* **CONFIRMED.** A decode that takes the *expected direction*
(`Σ p(bin)·centre`, then project to a fixed radius) or an argmax-over-direction rather than an
independent per-axis multinomial removes this entirely, with **zero retraining**. Note the
countervailing measurement: greedy (`temperature=0`) is a measured dead policy — so the fix is
"sample the gate, but decode the direction deterministically", not "set temperature 0".

**H91. The self-fed action history is a bounded random walk with sticky bin edges, whose
stationary distribution piles up in the screen corners.**
*Mechanism:* the head re-emits a ~1.5-bin perturbation of its own last target
(`mean |argmax − prev| = 1.460` bins vs `|argmax − true| = 1.548`); on a clamped 0..20 grid
that is a random walk with reflecting/absorbing boundaries.
*Prediction:* corner mass should grow with rollout length and be far higher than the
open-loop probe's.
*Test:* already partly run — closed-loop executed commands put **53.8% of x and 47.1% of y in
bin 20**, versus **19.4% / 5.7%** for the open-loop probe on the same window; and the
distribution shifts over the rollout (`p(x=bin0)` 0.108 → 0.169 from frames 0–20 to 300–600).
Clean confirmation: simulate the measured step distribution on a clamped 0..20 lattice and
compare its stationary distribution to the observed. **~20 min, no GPU.**
*Prior:* **HIGH.** *Status:* **CONFIRMED in direction, the lattice simulation is OPEN.**
Distinct from H40 (that the copy channel exists) — this is what the copy channel *does* to a
bounded output space.

**H92. On reset the window is left-padded with 16 copies of the first frame, so for the first
second the model is looking at a world with no motion.**
*Mechanism:* `agent_infer.act_from_latent` inserts `window[0]` until the buffer is full.
*Prediction:* a world model shown no change predicts no change.
*Test:* already measured in the deployment verification — feeding one FIXED latent 120 times
yields exactly **1** distinct movement target, and uniform noise yields 2, against ~60 on real
recorded latents.
*Prior:* **MEDIUM** for the symptom (it is one second), **HIGH** as a hazard because the
action history is *seeded* from that second and then copied forward (H91).
*Status:* **CONFIRMED mechanism, OPEN as a contributor.** Cheap fix: do not act until the
buffer holds `context` real frames.

**H93. The live loop's frame spacing differs from training's and is irregular.**
*Mechanism:* training assumes a fixed 20 fps `dt`; live runs at a measured 17.0 fps with
stale frames *skipped*, so the 16-frame context spans ~0.94 s of wall time with variable
spacing, and the dynamics' temporal RoPE assumes uniform steps.
*Prediction:* degradation proportional to the stale fraction; the offline rollout (exact
20 fps) would be better than live at equal everything else.
*Test:* offline rollout with frames decimated to 17/20 and with random gaps, same lane metric.
~2 h GPU.
*Prior:* **MEDIUM.** *Status:* **OPEN.** Distinct from H4 (`sim_replay`'s 85-frame stride is a
different, much larger, error in a different tool).

**H94. Commands are silently dropped after the model decides them.**
*Mechanism:* `_click_move` (`play_live.py:275-290`) returns early if `now − _last_click_t <
click_min_interval` (0.12 s) and counts it in `clicks_suppressed`; the runbook's own example
stop-summary shows `clicks sent=214 suppressed=31` — **13% of decisions discarded**.
*Prediction:* the executed command stream is a subsample of the decided one, biased against
bursts.
*Test:* log both streams in one session and compare. Free, needs a session.
*Prior:* **LOW-MEDIUM** — it cannot flip a lane, but it interacts with H80 (an already-mute
gate).
*Status:* **CONFIRMED as a mechanism, OPEN as a contributor.**

**H95. The train/val split is recomputed rather than restored on resume, so a requeue can
silently move a "held-out" game into train.**
*Mechanism:* `select_val_matches` runs at `train_agent_finetune.py:1593`; the resume block
(`:1672-1733`) never reads back `rc["args"]["val_matches_resolved"]` — that key appears only
at `:1597`. The split is stable only while `--val-games`, `--seq-len`, `--stride` and the set
of `.pt` files are unchanged. The deployed run used `resume = auto` under a requeuing sbatch.
*Prediction:* if the corpus changed mid-run, some val games were trained on and every
"held-out" number after that point is optimistic.
*Test:* compare `val_matches_resolved` across the epoch checkpoints on disk
(`agent_finetune_epoch_001.pt`, `_002.pt`, `_latest.pt`). **~5 minutes, no GPU.**
*Prior:* **MEDIUM.** *Status:* **OPEN — trivially checkable and nobody has.**

---

## J. Decode and coordinate transform

**H96. `_wasd_keys` computes the direction from fractions of two different pixel counts, so a
true 45° reads as 60.6°; and `--capture-region` has no aspect guard, so a non-16:9 capture
silently skews x against y on the mouse path too.**
*Mechanism:* `play_live.py:265-273`, `ang = atan2(-dy, dx)` on `dx = mvx − 0.5` (fraction of
1280) and `dy = mvy − 0.5` (fraction of 720). The deadzone is an ellipse, not a circle. The
mouse path is immune to the *first* half (fractions are invariant under a per-axis squish) but
not to the second: nothing checks that the capture region's aspect matches the corpus's 16:9.
*Prediction:* a rotation of commanded directions toward the vertical, largest at ±45°.
*Test:* (a) read the code — done for WASD; (b) assert `abs(w/h − 16/9) < 0.01` in `main()`.
*Prior:* **LOW** for the shipped mouse+1280×720 configuration, **HIGH** the moment anyone
changes the capture region.
*Status:* **CONFIRMED for WASD, OPEN as a live hazard for mouse.**

---

## K. HID, mouse, timing

**H97. The corner slam is sized from `span` — a property of the capture region — not of the
pointer surface, so on any surface larger than ~2254 px per axis it never reaches a clamp and
every click carries a constant offset.**
*Mechanism:* `hybrid_sender.py:251` `n = int(max(span)/SLAM_CHUNK) + 4` = `int(649/127)+4 = 9`
reports × 127 = 1143 units ≈ 2254 px.
*Prediction:* a constant, coherent offset in every commanded position — precisely this
symptom's class — on a 2560-wide monitor or a multi-monitor virtual desktop.
*Test:* on the real rig, command the four corners and the centre and photograph where the
cursor lands (`hybrid_sender.py --mouse-test` already does this). **10 minutes on hardware.**
*Prior:* **MEDIUM-HIGH conditional on the rig's geometry, which nobody has recorded.**
*Status:* **OPEN — the top HID candidate.**

**H98. `--desktop`'s semantics are stale from the retired absolute-HID design; following the
help text compresses every click into the top-left ⅔×⅔ of the screen.**
*Mechanism:* `play_live.py:206-210, 290, 409-415` describe absolute 0..32767 coords, but
`hid_server.py:243-249` refuses absolute on a relative gadget. Passing e.g.
`--desktop 1920x1080` with a 1280×720 stream makes `mx/self.dw` shrink every fraction by 2/3.
Currently latent only because `stage_desktop_standalone.sh` omits the flag and the default
`_dw,_dh = region` makes the division an identity.
*Prediction:* if anyone ever "fixes" the config by passing the true desktop size, every click
moves up and left by a factor of 1.5.
*Test:* read the code. Done. Then delete the flag or rewrite the help.
*Prior:* **CERTAIN as a defect, LOW as a current cause.**
*Status:* **CONFIRMED. A booby trap.**

**H99. Nothing validates the mouse calibration against the running geometry, the pointer
acceleration setting, or the gadget's actual report format.**
*Mechanism:* `mouse_calibration.json` is flagged `"provisional": true` and records
`stream_size`, but `load_calibration` (`hybrid_sender.py:78-100`) reads only `span`, `chunk`,
`measured_at`, and the preflight prints the span without cross-checking `--stream-size`.
Separately: a relative sender writing 4-byte reports into a 6-byte absolute gadget is a
**silent no-op** — the agent "plays" and never aims — and `nonlinearity > 8%` means "Enhance
pointer precision" is on. The end-to-end path has **never been run against real hardware**
(`DEMO_RUNBOOK.md:375-386`).
*Prediction:* a scaled or non-linear mapping — coherent, sustained, and indistinguishable from
a model bias without H10's instrumentation.
*Test:* re-run `calibrate_mouse.py` on the rig and compare against the shipped span; assert
the stored `stream_size` equals the running one. **20 minutes on hardware.**
*Prior:* **MEDIUM.** *Status:* **OPEN.**

---

## L. Live-vs-replay domain gap

*(H36–H38 and H94 also belong to this family; they are filed under their originating link.)*

**H100. The live HUD does not just corrupt the input — it *reinterprets the output click*.**
*Mechanism:* the policy emits a screen coordinate. On a HUD-bearing screen, a right-click in
the bottom-right corner lands on the **minimap**, where a right-click is a move order to the
corresponding **map** location, not to the world point under the cursor; a right-click on a
unit is an **attack** order; a right-click on the ability bar or shop panel is swallowed. The
training corpus has no HUD, so ~100% of label targets were valid world points (M3: only 6.6%
of labels fall in a region a live HUD would occupy).
*Prediction:* an edge/corner-heavy command distribution turns into long-range map orders to
apparently unrelated parts of the map — exactly a coherent, sustained walk to bot lane on blue
side, since the bottom-right of the minimap *is* bot lane.
*Test:* already partly measured — **M11: 43.0% of executed closed-loop commands under `held`
fall inside the minimap rectangle; 73.5% on some HUD element.** The decisive test is H10's:
log the commanded pixel and the champion's realised displacement in one live session, and
check whether displacement correlates with the *minimap-decoded* map point rather than the
screen point. ~3 h to instrument, 20 min to run.
*Prior:* **HIGH** — it is the only mechanism found that converts the measured corner-clicking
into the *specific* reported lane, and it is completely untested.
*Status:* **OPEN — the largest single unknown in the document.** Note `--attack-key` also
defaults to `None`, so AA never fires in mouse mode.


---

# PART 3 — WHAT IS SETTLED, WHAT IS OPEN, AND WHAT TO DO NEXT

## 3.1 Tally

100 hypotheses. Roughly: **6 refuted outright** (H3, H21, H22, H23, H51, H86),
**~55 confirmed as true statements about the system** — but note that most of those are
confirmed *defects* whose causal contribution to *this* symptom is rated separately in the
entry, **3 partial** (H14, H37, H43: real, measured, and shown insufficient), and
**~36 open**.

**The four that are both confirmed and load-bearing for the reported symptom:**

| | |
|---|---|
| **H12** | The walk-out has zero movement supervision (137,501 sentinel frames, p50 61 s/game). |
| **H26** | The side label exists, is computed, and is discarded — and the task is 86–108° different on the two sides. |
| **H56** | The gated mixture passes ~no gradient to the categorical on hold frames, so the walk-out categorical is free extrapolation — and it extrapolates to a screen **corner**. |
| **H40 + H92** | The copy channel dominates vision ~3:1 and, self-fed on a clamped grid, becomes a random walk with corner attractors. |

**And two confirmed process failures that cost more than any of the above:**

| | |
|---|---|
| **H79/H80** | The measured free fix (`--movement-action-mode none`, plus a re-tuned `--gate-bias`) is **not in the deployed launcher**. |
| **H90** | `prefirst_mode` — the label fix every plan in this repo depends on — is **dead code with no CLI flag and a `pre_valid` that is never read**. |

## 3.2 Ranking of the open hypotheses

Score = prior × diagnostic value ÷ cost. Only open/partial entries; ties broken by cost.

| rank | H | what it decides | cost | score |
|---|---|---|---|---|
| 1 | **H83/H84/H85/H87** | is the rig even running this code and this checkpoint | 5 min | ★★★★★ |
| 2 | **H79/H80** | does the measured fix survive contact with hardware | 40 min | ★★★★★ |
| 3 | **H101 + H10** | does the HUD reinterpret the *output* click (minimap move order) | 3 h + a session | ★★★★★ |
| 4 | **H54/H67** | does the corner mode alone explain the lane, with zero retraining | 1 h GPU | ★★★★☆ |
| 5 | **H91** | does deterministic direction decoding recover the preference the distribution already has | 1 h GPU | ★★★★☆ |
| 6 | **H53** | is the axis factorisation *manufacturing* the corner mode | 1 h GPU | ★★★★☆ |
| 7 | **H24/H34/H48** | is side recoverable at all, and from where (latents vs agent token) | 1–2 h, no training | ★★★★☆ |
| 8 | **H95** | did any val game leak into train across requeues | 5 min | ★★★★☆ |
| 9 | **H45** | does the ability channel dominate the movement token in practice | 1 h GPU | ★★★☆☆ |
| 10 | **H90 → H12/H13/H69** | wire the label fix, then measure it on a *direction* bar | 3 h + 2.2 h GPU | ★★★☆☆ |
| 11 | H37 | does masking the HUD before `encode_frame` change live behaviour | 30 min | ★★★☆☆ |
| 12 | H2 | how much worse is a genuinely closed observation loop | 2 h GPU | ★★★☆☆ |
| 13 | H93 | does refusing to act until the buffer is full help | 20 min | ★★★☆☆ |
| 14 | H98/H99/H100 | is the pointer mapping itself offset or scaled | 30 min on hardware | ★★★☆☆ |
| 15 | H72/H75 | does an LR decay or checkpoint averaging cut the seed variance | 1 h GPU each | ★★☆☆☆ |
| 16 | H55/H57/H58/H47 | which architectural change buys the most per GPU-hour | 2.2 h GPU each | ★★☆☆☆ |
| 17 | H35/H39/H61 | anisotropy, tanh saturation, bin geometry | 20–30 min each | ★★☆☆☆ |
| 18 | H28/H49/H73/H78/H88/H97 | low-prior hygiene | varies | ★☆☆☆☆ |

## 3.3 Recommended next 10 tests, in order

Ordered so that each one either costs minutes or is unblocked by the one before it. Tests
1, 2, 4, 5, 6, 8 involve **no training at all**.

1. **Identify the running rig.** `ssh desktop cat /mnt/storage/ahriuwu-live/VERSION` and run
   `preflight.sh --inject hid --movement-mode mouse --source udp`. Settles H83 (fork), H84
   (dirty tree), H85 (wrong checkpoint), H87 (tokenizer sha) in one shot. **5 minutes.**
   *Do this before reading anything else in this document as actionable.*

2. **Deploy the fix that already exists.** Add `--movement-action-mode none` and a re-tuned
   `--gate-bias` (aim for ~2 clicks/s; `none` drops the raw rate to ~0.37/s) to `play.sh` in
   `ops/stage_desktop_standalone.sh`, re-stage, and run **five** live games logging the
   executed lane. Settles H79, H80 and gives H1 an n>1. **40 minutes.**

3. **Instrument the live loop to log the game's response.** Add the champion's on-screen
   position (template match, or the existing CV HP-bar reader) to `Recorder` alongside the
   commanded pixel, and run one session. This is the *only* test that separates a model-side
   cause from a post-model one (H10), and it is the decisive test for the minimap mechanism
   (H101) and for the HID mapping (H98–H100). Also run one arm with the HUD masked to black
   before `encode_frame` (the mask already exists at `sim_replay.py:96`) to close H37 live.
   **~3 h to build, 20 min to run. Highest information content in the document.**

4. **Kill the edge bins at inference.** Re-normalise the movement logits over bins 1..19
   (or, better, decode to a fixed radius) and re-run the 40-game lane metric with everything
   else identical. Settles how much of the symptom is H54/H67 alone. While the GPU is busy,
   run the **20-minute, no-GPU** lattice simulation for H92: sample the measured step
   distribution on a clamped 0..20 grid and compare its stationary distribution to the
   observed 53.8%/47.1% corner mass. **1 h GPU.**

5. **Decode the direction deterministically.** Keep the gate stochastic (greedy is a measured
   dead policy) but replace the per-axis multinomial with the categorical's expected
   direction. Same 40-game metric. Settles H91, which the existing R=1.000-vs-28% measurement
   already says should be a large win. **1 h GPU.**

6. **Score the `joint_noop` lineage on the same walk-out probe.**
   `data/phase2_from_vast/vast_step90000.pt` is a joint head; compare its corner mass and its
   blue/red separation against the axis head on identical windows. Settles H53 — whether the
   `p(x)·p(y)` factorisation is *manufacturing* the corner mode or merely reporting it.
   **1 h GPU.** If it is manufacturing it, the fix is a head change, not a data change, and
   that reorders everything below.

7. **Probe for side and for game time.** Logistic/MLP probes for `team` and for game-time
   bucket, from (a) raw v7 latents and (b) the agent token, bucketed by game time. Settles
   H24, H34 and H48 together, and forks the fix plan three ways: *give it the bit* (cheapest),
   *fix the agent block* (cheap), *fix the tokenizer* (expensive). **1–2 h, no training.**

8. **Check the val split across checkpoints.** Compare `args["val_matches_resolved"]` in
   `agent_finetune_epoch_001.pt`, `_002.pt`, `_latest.pt`. Settles H95. **5 minutes.**

9. **De-confound our own measurements.** Re-run the walk-out probe with the agent's sampled
   abilities in the history versus all-zeros. The existing probe zeroes them and the existing
   rollout does not, and their corner masses differ by ~2× — so one of our two headline
   measurements is contaminated. Settles H45 and repairs the comparison. **1 h GPU.**

10. **Wire `prefirst_mode`, then use it.** Add the CLI flag, thread `pre_valid` into the `md`
    dict and consume it in `__getitem__` (H90), then run 5,000 steps with
    `prefirst_mode='heading'` **and** run-level (block) action dropout — both, since the label
    fix alone leaves the copy channel dominant and the channel fix alone leaves the walk-out
    unsupervised. Score it on the **direction** bar (per-game commanded direction vs the
    champion's own heading, reported separately for blue and red, frames 30–1190), not on
    event-frame CE, which by construction cannot see the walk-out. **~3 h to wire, 2.2 h GPU.**
    While wiring, add the one-bit side embedding (H26) — it is nearly free and it is the only
    change that addresses the 86–108° blue/red split directly.

## 3.4 What I would *not* spend GPU on yet

- **A tokenizer retrain.** The measured plateau is objective/data-limited (+0.28 dB for 9
  days; effective rank 31 of 8,192), and test 7 above decides whether perception is even the
  binding constraint. H32 says scale is not the lever.
- **Phase 3.** It is structurally blocked in three independent ways (offset-0 heads at zero,
  arch flags not restored, gated heads rejected by `log_prob`), and its behavioural prior
  would be uniform anyway.
- **Longer BC runs.** H71: most of the learning happens in the first few thousand steps and
  the last 50k moved `move` by 0.023.
- **Anything judged on event-frame CE.** H6 and H7: the bar is cleared by the broken
  checkpoint, and the metric cannot see the failing behaviour at all.

## 3.5 The honest summary

The agent does not walk to top lane because **the walk to top lane is the one behaviour in
the corpus that was never labelled**, because **the one bit that disambiguates it (which side
you are on) is computed and thrown away**, and because on those unlabelled frames the gated
objective leaves the movement categorical unsupervised, whereupon it extrapolates to a screen
corner and the self-fed action channel locks it there. Live, that corner is where the minimap
is.

Every one of those is fixable, three of them without a GPU, and one of them
(`--movement-action-mode none` in `play.sh`) is a single line that has been measured, written
up, and left undeployed.

---

*Measurement scripts for M1–M12 were run against
`/srv/nfs/datasets/lol_replays_16_9_772`, `/srv/nfs/datasets/replay_latents_v7_bc`,
`scratchpad/lane/{rollout,probe,labels_cache}` and
`data/phase2_bc_clicks/agent_finetune_latest.pt`. Everything quoted as measured here was
produced by running code, not by reading it; everything quoted from another document carries
that document's name.*
