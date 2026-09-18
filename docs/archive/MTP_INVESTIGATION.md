# MTP offset 0: what is actually wrong, and what is not

**Date:** 2026-08-27. **Trigger:** `heads.0.weight norm = 0.0000e+00` after 102,420 steps.
**Scope:** every producer and consumer of the multi-token-prediction (MTP) axis.
**Line numbers are as of `46d94c2`** — this tree moved several times during the
investigation; grep the quoted code if they have drifted again.

Everything below marked CONFIRMED was run. LIKELY was read, not executed.

**Scope of the measurements.** All five Phase-2 checkpoints on disk were trained
with the HELD movement target and `--movement-action-mode held`; the
`--movement-interp` and default-action-channel work that landed in `6057853` /
`46d94c2` while this was running post-dates every one of them. The copy-leak
magnitudes below therefore describe the `held` lineage. Nothing about the MTP
axis or the zero head depends on that choice.

---

## Verdict

**(b), and only one of the audit's two claims survives.** The zero head is
*mechanically* expected — a zero-init tensor that appears in no loss — and
"exactly 0.0" is the correct signature of never-touched, which the optimizer
state proves. What is a real defect is what reads it:

1. **Phase 3's KL regulariser reads the dead head** (`train_imagination.py:403`),
   so the "behavioural prior" is exact uniform and the term is an **entropy
   bonus with weight 0.3** pointed at the trained offset-1 policy. Measured at
   7.29 nats where a correct KL is 0. This is the off-by-one the `MTP_OFFSET`
   fix missed, and it is still in the code.
2. **Phase 3 no longer samples at offset 0** — that half was fixed on
   2026-08-20 and the audit was never updated. It is moot anyway: Phase 3
   cannot complete a single step on any checkpoint on disk (§4).
3. **n=0 is no longer necessary to skip, for movement.** With
   `--movement-action-mode none` the movement channel is out of `h_t`, so the
   n=0 movement term is honest. Abilities still leak and must stay at n>=1.

The MTP axis itself is **consistently wired** in trainer, heads and inference —
verified empirically, not just read. There is no global off-by-one. The two
genuine off-by-ones are local: the Phase-3 prior slice and
`ab_checkpoints.cell_acc`.

**One intuitive concern is REFUTED by measurement.** BC only ever applies the
policy head to token positions 0..T-2, and `agent_infer` reads T-1 — so the
deployed read position is never supervised. It costs nothing measurable: at
position 15 the gate fires at 0.1290/frame vs 0.1295 mean over positions 0..14
(ratio 0.996), gate BCE 0.2987 vs 0.2999, token norm 5.479 vs 5.475, and both
movement CE and ability BCE sit mid-range. The policy head is a per-token MLP
with no positional parameters and the *token* at position 15 is trained by the
reward and state heads, so there is nothing position-specific to learn.

**The audit's section 2.1 is stale.** It says "Every Phase-3 read is offset 0".
That was fixed 17 minutes after the audit doc was committed
(doc `f569891` at 22:20:37, fix `4c93083` at 22:37:44, both 2026-08-20) and the
doc was never updated. Phase 3 now samples and scores at offset 1. It reads the
**prior** at offset 0, which is the part that still bites.

---

## 1. Zero-init, never touched — not decayed. CONFIRMED

`src/ahriuwu/models/heads.py:261-265` zero-inits **every** PolicyHead output
head (and `heads.py:256-257` the gate heads, `heads.py:102-103` the reward
heads, `heads.py:40-41` the state head, `heads.py:553-554` the value head). So offset 0 starts at exactly 0 and
`bc_next_action_loss` (`scripts/train_agent_finetune.py:812`) never touches it.

Decay cannot be the explanation: AdamW decoupled decay at `lr=3e-4, wd=0.1`
over 102,420 steps is a factor `(1-3e-5)^102420 = 0.046` — a 22x shrink, not
zero, and never *exactly* zero in fp32.

The decisive evidence is the optimizer state, not the weight. Every parameter
in the checkpoint has an Adam slot (`p.grad` is a zeros tensor, not `None`,
because `torch.stack` over all 9 heads puts head 0 in the graph and the unused
slice back-propagates zeros). Exactly six slots have
`exp_avg_sq` identically 0 — i.e. the gradient was exactly 0 on all 102,420
steps:

```
optimizer slots=165  reconstructed names=165  MATCH
0.000000e+00  policy.gate_heads.0.bias         (1,)
0.000000e+00  policy.gate_heads.0.weight       (1, 256)
0.000000e+00  policy.heads.0.bias              (9,)
0.000000e+00  policy.heads.0.weight            (9, 256)
0.000000e+00  policy.movement_heads.0.bias     (42,)
0.000000e+00  policy.movement_heads.0.weight   (42, 256)
1.170818e-06  dyn.agent_blocks.0.self_attn.q_proj.weight    <- next lowest
```

`gate_heads.0` is a **third** dead sub-head the audit did not list.

(Reconciling the numbers: the audit's `heads.1..8 ~ 5.1`, `movement_heads.1..8 ~ 23`,
`reward_head.heads.0 = 2.86e+01` are the **`phase2_from_vast`** lineage at 99,421
steps — joint_noop, 442 movement classes. The same tensors on `phase2_bc_clicks`
at 102,420 steps read 7.87 / 7.04 / 48.9. The zeros are identical in both.)

**Would random init have been less harmful? No — worse.** Zero logits are the
exact maximum-entropy distribution, which is at least an honest "I don't know".
A random-init head would emit an arbitrary *confident* preference (a fixed
ability bias and a fixed favourite movement bin), and Phase 3 would optimise
against a prior that is confidently wrong rather than uninformative. The bug to
fix is that anything reads head 0 at all, not the value it holds.

---

## 2. Is the MTP axis consistent? Yes — verified end to end. CONFIRMED

### 2.1 The contract

`scripts/train_agent_finetune.py:816-817`

```python
a_logits = ability_logits[:, :T - n, n, :]   # token t
a_tgt    = ability_targets[:, n:, :]         # action at t+n
```

Head `n` at token `t` predicts `a_{t+n}`. `reward_mtp_loss`
(`train_agent_finetune.py:765-766`) uses the identical slicing at `n = 0..L-1`.

### 2.2 Empirical proof, synthetic (decisive)

`scratchpad/mtp_align_unit.py` builds a world where `a_t` is a known function of
`t` and the agent token identifies `t`, trains the **real** `PolicyHead` with the
**real** `bc_next_action_loss`, then decodes with the **real** `PolicyHead.sample`
— the exact call `agent_infer` makes.

```
query frames [0..8] (each appears at token position 0, so every head saw it)
  n | head emits a_(t+n) exactly | as a_(t+n-1) | as a_(t+n+1)
  0 |          1/9               |     1/9      |    0/9
  1 |          9/9               |     0/9      |    0/9
  2 |          9/9               |     0/9      |    0/9
  3 |          9/9               |     0/9      |    0/9
  4 |          9/9               |     0/9      |    0/9
  5 |          9/9               |     0/9      |    0/9
  6 |          9/9               |     0/9      |    0/9
  7 |          9/9               |     0/9      |    0/9
  8 |          9/9               |     0/9      |    0/9
```

**Every trained head emits exactly `a_(t+n)` on every query frame, and 0/9 for
either off-by-one.** Head 0 emits `(0, 0)` — the argmax of an all-zero logit
vector, i.e. bin 0 on both axes, which happens to equal the true action at the
single frame `t=0`; that is the 1/9, not a prediction.

Two design points this needed, both found by getting them wrong first. (i) The
query frames must be ones every head actually saw: head `n` reads only token
positions `p <= T-1-n`, so a late frame is never an input to the high heads and
querying one tests extrapolation, not wiring. (ii) At least one axis must have
full period 21. Here `a_t.x = 3t mod 21` has period 7 — on x alone, head `n` and
head `n+7` are indistinguishable — and it is `a_t.y = 5t mod 21` that settles
it. A first attempt that violated both looked like a partial failure and was
nothing of the kind.

### 2.3 Empirical check on the real checkpoint

`scratchpad/mtp_probe.py`, 1,200 windows over the checkpoint's own 6 held-out
val games, teacher-forced in the training regime. It scores every MTP head `n`
against every target offset `m` — a 9x9 cross-tab. Correct wiring puts the
minimum on the diagonal.

**Read the cross-tab down the COLUMNS, not across the rows.** Within a column the
scored frames are identical, so heads are directly comparable; across a row the
frame set changes with `m`.

Ability BCE (nats/frame, summed over 9 abilities), `*` marks each COLUMN's
minimum:

```
         m=0      m=1      m=2      m=3      m=4      m=5      m=6      m=7      m=8
  n=0   6.2383   6.2383   6.2383   6.2383   6.2383   6.2383   6.2383   6.2383   6.2383
  n=1   0.1101   0.0949*  0.0943   0.0980   0.1024   0.1023   0.1043   0.1046   0.1077
  n=2   0.1112   0.0968   0.0937*  0.0947   0.0979   0.0981   0.1011   0.1027   0.1058
  n=3   0.1131   0.0991   0.0953   0.0946*  0.0961   0.0959   0.0994   0.1020   0.1058
  n=4   0.1147   0.1008   0.0972   0.0957   0.0955*  0.0939*  0.0968   0.0997   0.1051
  n=5   0.1155   0.1020   0.0991   0.0979   0.0970   0.0941   0.0956   0.0980   0.1044
  n=6   0.1128   0.1019   0.0998   0.0995   0.0988   0.0951   0.0952*  0.0963*  0.1030
  n=7   0.1070   0.1012   0.0996   0.1004   0.1010   0.0974   0.0967   0.0966   0.1020*
  n=8   0.1043*  0.1012   0.0996   0.1012   0.1028   0.0997   0.0990   0.0982   0.1021
```

The column minimum is exactly on the diagonal for m = 1, 2, 3, 4, 6; at m = 5,
7, 8 it sits one head early, by 0.0002 / 0.0003 / 0.0001 nats. **Head `n` is the
best head for target offset `n`, to within a ten-thousandth of a nat.** The
`m=0` column is the worst for every trained head — no head was ever trained on
it — and row `n=0` is flat at `9*ln2 = 6.2383`, the exact Bernoulli(0.5) value,
by construction.

Movement gives the scale reference for what a genuine off-by-one would look
like. Categorical CE on event frames (nats, both axes summed):

```
        m=0       m=1       m=2       m=3    ...    m=8
  n=1  3.160*    4.168     4.170     4.252   ...   4.753
  n=8  3.944*    4.356     4.340     4.377   ...   4.594
  (uniform = 6.089)
```

Every trained head's ROW minimum is at `m=0` by **~1.0 nat** — that is the copy
leak (`MOVEMENT_HEAD_BLIND_2026-08-26.md`): at `m=0` the target is bit-for-bit
the movement action the backbone was FED at token `t`. Restricting to transition
frames (target bin != the action fed at `t`) removes it, and empties the `m=0`
column by construction:

```
  A2, transition-only     m=1       m=2       m=3       m=8
  n=1                    4.407*    4.412     4.497     4.901
  n=2                    4.406     4.405*    4.484     4.856
  n=8                    4.484     4.469     4.507     4.677*
  counts per column:      1622      1621      1626      1795
```

**A real off-by-one is a 1.0-nat effect on this data. Every diagonal deviation
measured is 0.0001-0.02 nats.** There is no off-by-one.

A second run with `--ablate-cursor` (the `--movement-action-mode none`
condition: `cursor_valid` forced all-False, so `embed_actions` substitutes
`no_action_embed` on every frame) costs the head ~0.85 nats and lands it at the
level of the corpus marginal:

```
  transition-only CE     m=1       m=2       m=3       m=8
  n=1, action present   4.407     4.412     4.497     4.901
  n=1, movement CUT     5.252     5.247     5.264     5.315
```

That is `MOVEMENT_HEAD_BLIND_2026-08-26.md`'s central claim reproduced
independently: with the movement channel gone the head is at or below the
corpus marginal (~5.2 nats there, on a slightly different frame subset, so read
this as a level check rather than a strict comparison). Note that under
ablation the heads become indistinguishable from each other (row spread 0.02
nats), so the ablated run cannot resolve the diagonal — the alignment evidence
above comes from the ability table and the un-ablated movement table.

A second-order finding worth recording: across the 8 trained heads the spread at
any fixed target offset is <= 0.08 nats (< 2% of the CE). The MTP heads have
**not** specialised by horizon — predicting `a_{t+8}` is barely different from
predicting `a_{t+1}` on a target that is piecewise constant (median hold run 5
frames). For movement the MTP structure is close to vestigial; only the ability
heads show a clean, if tiny, horizon ordering.

### 2.4 Consumer sweep

| consumer | offset read | verdict |
|---|---|---|
| `scripts/agent_infer.py:296` `n = 1 if self.mtp > 1 else 0` | 1 | correct |
| `scripts/train_imagination.py:313-314` (sampling) | `MTP_OFFSET`=1 | correct |
| `scripts/train_imagination.py:395` (`log_prob`) | `MTP_OFFSET`=1 | correct |
| `scripts/train_imagination.py:402-403` (**prior**) | policy@1 vs **prior@0** | **BUG** |
| `scripts/train_imagination.py:318` (reward) | 0 | reward MTP does train 0; see §6 |
| `scripts/overlay_e2e.py:95-97` | 1 | correct |
| `scripts/probe_casting.py:78` | 1 | correct |
| `scripts/probes/ood_test.py:92` | 1 (hardcoded) | correct |
| `scripts/eval_bc_sim.py:88-90` | via `act_from_latent`, GT = `a_{t+1}` | correct |
| `scripts/ab_checkpoints.py:87` + `:96` | via `act_from_latent` (= `a_{f+1}`), GT = `mv[f]` | **off by one** |
| `play_live.py`, `sim_replay.py`, `play_live_preflight.py` | via `act_from_latent` | correct |

Everything that plays or evaluates goes through `act_from_latent`, so the offset
choice is centralised in one line. `ab_checkpoints` is the exception: it scores
the `a_{f+1}` output against frame `f`'s own target (already noted in
WIRING_AUDIT §3, re-confirmed here by reading).

---

## 3. The Phase-3 prior reads the dead head. CONFIRMED, measured

`scripts/train_imagination.py:401-404`:

```python
kl = factorized_policy_kl(
    a_logits[:, :, MTP_OFFSET, :], a_prior[:, :, 0, :],
    m_logits[:, :, MTP_OFFSET, :, :], m_prior[:, :, 0, :, :],
)
```

The policy is read at `MTP_OFFSET`; the prior is read at literal `0`. The prior
is `copy.deepcopy(policy_head)` (`train_imagination.py:479` and `:561`), so at
Phase-3 step 0 prior *is* policy and a correct KL must be **exactly 0**.
Measured on the real `phase2_bc_clicks` policy head:

```
KL(policy@1 || prior@0)  AS CODED = 7.286874 nats
KL(policy@1 || prior@1)  CORRECT  = 0.000000e+00 nats
  decomposition: ability KL = 5.5778 nats, movement KL = 1.7091 nats
```

Because prior@0 is exact uniform, `KL(pi || uniform) = log K - H(pi)`. Minimising
it **maximises entropy**. With `--pmpo-beta 0.3` (default) the term is
`+0.3 * 7.29 = 2.19` on the loss and its gradient pushes the offset-1 policy —
the one Phase 3 also samples from — toward Bernoulli(0.5) on 9 abilities
(4.5 presses per dreamed frame) and uniform over movement. **It is not a
vacuous regulariser; it actively erases the BC policy.** The audit's conclusion
("the entire Phase-2 policy is discarded at the handoff") survives, by a
different mechanism than the one it named.

`create_behavioral_prior` (`heads.py:593`) is **dead code** — exported in
`models/__init__.py` and called nowhere. It would copy the same zeros.

---

## 4. Phase 3 cannot take a step on any checkpoint on disk. CONFIRMED

`scratchpad/mtp_phase3_e2e.py`, run on both live lineages:

```
data/phase2_bc_clicks/agent_finetune_latest.pt
  load_phase2 -> SystemExit: This Phase-2 checkpoint has a sticky movement gate...
data/phase2_from_vast/agent_finetune_latest.pt
  load_phase2 OK: model_dim=768 use_actions=True mode=joint_noop gate=False
  run_step -> IndexError: too many indices for tensor of dimension 4
```

and with the movement actions `imagine()` actually produces (continuous xy, not
class indices) the failure comes one line earlier:

```
run_step -> TypeError: joint_noop movement targets must be LONG class indices
            (got torch.float32); a continuous (x,y) cannot express NO_OP.
```

So there are **four** independent blocks, in the order they fire:

1. `gate=True` (3 of 5 checkpoints) — `SystemExit` at load, deliberate.
2. `joint_noop` — `imagine()` stores continuous xy (`train_imagination.py:312-314`),
   `log_prob` demands long class indices (`heads.py:484-486`). `TypeError`.
3. `joint_noop` — `factorized_policy_kl` indexes a `movement_dim` axis a joint
   head does not have (`returns.py:286-288` vs `train_imagination.py:403`).
   `IndexError`.
4. Then, and only then, the offset-0 prior of §3.

Also in the same function: `imagine()` calls `sample()` with no
`prev_movement_idx`, so a sampled `NO_OP` decodes to screen centre `(0.5, 0.5)`
rather than "repeat the standing order" (`heads.py:413-415`) — a fake
click-the-middle action injected into ~28% of dreamed frames.

---

## 5. Why nobody caught it. CONFIRMED

Both preflights assert gradient on the **sum over all MTP heads**, which is
positive whenever any offset trains:

- `scripts/train_agent_finetune.py:1251-1254` —
  `move_grad_norm = sum(g.norm() for h in policy_head.movement_heads)`.
- `scripts/train_imagination.py:502-505` — same shape of assertion.

A per-offset assertion fails on head 0 at step 1.

Worse, the Phase-3 smoke test **deliberately constructs the one condition that
hides the bug** (`train_imagination.py:472-478`):

```python
# Perturb the policy off zero-init so the prior KL and movement logits are
# non-degenerate (zero-init would make policy==prior, KL==0 trivially).
```

It perturbs head 0 too, so prior@0 is not uniform and the KL comes out
plausible. Run as shipped:

```
factorized KL       = 1.894739e-03  (> 0: policy != prior)
SMOKE TEST PASSED
```

Against a real checkpoint the same expression is 7.29. The smoke test noticed
the degeneracy and worked around it in the *test* instead of in the *trainer*.
It also builds an `axis` head, so failures 2 and 3 of §4 are invisible to it.

---

## 6. Other findings on the MTP axis

- **The deployed read position is unsupervised but costs nothing. REFUTED as a
  defect.** BC slices `[:, :T-n, n]` with `n >= 1`, so at `seq_len=16` the policy
  head is only ever applied to token positions 0..14; `agent_infer` reads
  position 15 (`agent_infer.py:282`, `read_pos=-1` default at
  `agent_infer.py:71`, never exposed on any CLI). Measured teacher-forced on
  1,200 val windows (`scratchpad/mtp_gate_pos.py`, 1,200 tokens per position):

  | | positions 0..14 (mean, range) | position 15 |
  |---|---|---|
  | gate P(fire)/frame | 0.12952, 0.1269-0.1323 | **0.12900** (ratio 0.996) |
  | gate BCE vs true events | 0.3002, 0.273-0.325 | **0.29869** |
  | `\|\|h_t\|\|` | 5.475, 5.459-5.481 | **5.479** |
  | movement event CE (n=1) | 4.181, 3.83-4.47 | **3.991** |
  | ability BCE (n=1) | 0.0802, 0.044-0.110 | **0.0614** |

  Every position-15 value is inside the between-position range, and the two with
  real statistical power — the gate, at 1,200 tokens per position — are within
  0.4%.

  Same picture with the movement action cut (5.145 vs a 5.06-5.47 range). The
  policy head is a per-token MLP with no positional parameters, and the *token*
  at position 15 is trained by the reward MTP's n=0 and the state head, so there
  is nothing position-specific for it to miss.

  This corrects two things. *A comment:* `agent_infer.py:280-281` says
  "`agent_temporal_pos[15]` is likewise BC-untrained" — the **parameter** is
  trained; measured row norms are 1.04-1.64 for rows 0..15 versus 0.0265 for
  rows 16..255 (never indexed at `seq_len=16`, i.e. untouched init). *A prior
  result:* `scratchpad/pos_ablation2.log` reported a 60% command-rate difference
  between the two positions (92/1800 vs 147/1800). Those were two independent
  self-fed rollouts with diverging action histories; teacher-forced on identical
  inputs the rate difference is 0.4%.
- **`agent_temporal_pos` rows 16..255 are at random init** (240 of 256 rows,
  184k params). Harmless at `seq_len=16`, but raising `GarenAgent(context=...)`
  above 16 silently feeds untrained positional embeddings.
- **Phase 3 reads reward at offset 0**, i.e. `r_t`, the gold delta that arrived
  *at* frame t (`src/ahriuwu/rewards/reward.py:155` `_dense_solo_gold` — `Δ gold_total` per frame). The action
  `a_t` sampled from the same token cannot have caused it; offset 1 is the
  reward attributable to `a_t`. LIKELY a half-step misattribution, second-order
  next to §3.
- **`--mtp-length 1` silently trains nothing.** CONFIRMED: `for n in range(1, 1)`
  is empty, so `bc_next_action_loss` returns a literal constant —
  `bc loss = 0.0, requires_grad = False` — and the policy head gets no gradient
  at all. The preflight would catch it, but the preflight only runs under
  `--smoke-test`.
- **No other dead parameters.** Comparing the Phase-2 checkpoint's
  `dynamics_state_dict` against its Phase-1 parent
  (`rollout_stage/desktop_resume_8775_stripped.pt`): 270 tensors bit-identical,
  0 changed — the backbone freeze is exact. Every trainable slot outside the six
  above has a nonzero Adam second moment. The same six (four, where there is no
  gate) are dead in **all five** Phase-2 checkpoints on disk:

  | checkpoint | steps | mode | dead slots |
  |---|---|---|---|
  | `phase2_bc_clicks` | 102,420 | axis+gate | 6 |
  | `phase2_bc_gate1060` | 97,472 | axis+gate | 6 |
  | `phase2_from_vast` | 99,421 | joint_noop | 4 |
  | `phase2_parity` | 55,216 | joint_noop | 4 |
  | `phase2_bc_garen_act8775` | 157,806 | axis | (8-bit Adam, no `exp_avg_sq`; weights are exactly 0) |

  `task_embed.weight` is present, at random init, and never read — already
  documented as intentional at `train_agent_finetune.py:324-332`.

---

## 7. Should n=0 be trained instead? Yes, with one carve-out

The n=0 term was dropped because the agent token at `t` is built from a window
whose frame-`t` action token already contains `a_t`
(`dynamics.py:776` appends the action token to frame t's set; `embed_actions`
at `dynamics.py:551-585`).

`--movement-action-mode none` (commit `8f3e219`, `train_agent_finetune.py:177`)
forces `cursor_valid` all-False, and `embed_actions` then substitutes
`no_action_embed` for the movement embedding on every frame. So under `none`
**the movement action is not in `h_t` and n=0 movement is not a leak.**
Independently corroborated in `MOVEMENT_HEAD_BLIND_2026-08-26.md` §CORRECTION:
the held movement target decodes from the normal agent token at 2.365 nats and
at 5.578 once ablated — worse than the 5.216 corpus marginal, i.e. the ablation
is complete.

A linear probe of `a_t` from `h_t`, same frames and same split in both
conditions (`scratchpad/mtp_probe.py` section D, 19,200 tokens):

| decoded from `h_t` | movement axis 0 | axis 1 |
|---|---|---|
| action channel present (`held`) | **0.418** | **0.492** |
| movement action cut (`none`) | 1.268 | 1.377 |
| corpus marginal | 2.464 | 2.649 |

The movement action input is worth 0.85 nats of the decode — that is the leak,
measured. **Caveat, stated because it matters:** this probe uses a random 70/30
split *within* the same 6 games, and the movement target is piecewise constant
over long runs, so the absolute levels are inflated by memorisation and the
residual 1.2 nats above marginal cannot be read as "state information". The
paired contrast is valid; the level is not. For a leave-one-game-out number use
`MOVEMENT_HEAD_BLIND_2026-08-26.md` §CORRECTION (5.578 ablated vs 5.216
marginal), which is what establishes the ablation is complete.

**The carve-out: abilities are NOT removed by that flag.** `embed_actions` adds
every ability embedding unconditionally (`dynamics.py:582-583`); `cursor_valid`
gates only the movement term (`dynamics.py:576`). An n=0 *ability* term would therefore be a pure
copy of the model's own input. Corroborated by the same probe: with the movement
action cut, `a_t` for AA still decodes from `h_t` at **AUC 1.0000** (Q 0.9900,
E 0.9795) — the ability channel is untouched by `--movement-action-mode`, so it
is still literally an input. (Same split caveat as above; but here the code path
alone is decisive.) The honest configuration is:

> train n=0 for **movement only**, under `--movement-action-mode none`
> (or run-level block dropout); keep abilities at n>=1.

`event_only` is **not** sufficient for this: it sets `cursor_valid = movement_event`,
so on exactly the frames the metric scores — event frames — `a_t` is back in
`h_t`. Only `none`, or a block dropout whose run includes frame `t` itself,
makes n=0 honest. Note the interaction with `46d94c2`: `--movement-interp` now
hard-errors against `held` and directs to `event_only` **or** `none`. Of those
two only `none` is compatible with training n=0 — and with interp the point is
sharper still, since the interpolated target at `t` is a function of the *next*
click, so under `event_only` an n=0 head would read a future-dependent value out
of its own input.

An alternative to the carve-out would be to drop the ability channel too, but
nothing currently can: `action_dropout` masks only `cursor_valid`
(`train_agent_finetune.py:1015-1017`), never the ability embeddings. If ability
dropout were added, per-frame would suffice — a press is a single-frame event at
a ~0.3% rate, so it is not replicated across neighbours the way a held movement
target is, and the `p^(j+1)` argument in `MOVEMENT_HEAD_BLIND_2026-08-26.md`
does not apply.

That single change buys, at no extra GPU cost:

- Phase 3's `MTP_OFFSET` can go to 0 and the prior slice becomes correct by
  construction (Dreamer 4 Eq. 9 sums from n=0 for exactly this reason).
- Inference stops emitting `a_{t+1}` at time `t`. The current 1-frame lead is
  undocumented and interacts with a 12-17 fps loop; it may even be desirable
  given actuation latency, but nobody chose it.
- The deployed read position (T-1) enters the BC loss — cosmetically nice, but
  §6 measures the cost of not having it at ~0, so do not spend the run on this
  reason alone.

It does **not** fix perception: `MOVEMENT_HEAD_BLIND_2026-08-26.md` §SECOND
CORRECTION shows the visual policy that exists only surfaces once the movement
channel is cut, and that is orthogonal to which offset is scored.

---

## 8. Recommended fixes, in order

1. `train_imagination.py:402-403` — read the prior at `MTP_OFFSET`, not 0.
   One-line, and it is the difference between "KL is 0 at step 0" and "the KL
   term deletes the BC policy". **Do this even if nothing else changes.**
2. Assert per-offset, not summed, in both preflights; and make the Phase-3
   smoke test cover `movement_mode="joint_noop"` *without* perturbing head 0.
3. Fix the two joint_noop crashes (§4.2, §4.3) or drop `joint_noop` from the
   Phase-3 path explicitly, the way `movement_gate` already is.
4. Train n=0 for **movement only** under `--movement-action-mode none`, then set
   `MTP_OFFSET = 0`. This is the structural fix: it makes the head Phase 3 reads
   the head BC trains, and it removes the undocumented 1-frame lead at
   inference. Keep abilities at n>=1 until an ability dropout exists.
5. Delete `read_pos` and the comment above it (`agent_infer.py:277-282`). It is
   unreachable from any CLI, the comment it carries is wrong about
   `agent_temporal_pos`, and §6 shows the position it exists to work around
   costs nothing.
6. Update `WIRING_AUDIT_2026-08-20.md` §2.1: the sampling/scoring half is fixed,
   the prior half is not.

---

## Reproduce

```
scratchpad/mtp_dead_sweep.py     # §1  named dead-parameter sweep
scratchpad/mtp_align_unit.py     # §2  synthetic end-to-end axis test
scratchpad/mtp_probe.py          # §2, §6, §7  cross-tab + read position + leak probe
scratchpad/mtp_probe.py --ablate-cursor   #   the --movement-action-mode none condition
scratchpad/mtp_gate_pos.py       # §6  gate behaviour by token position
scratchpad/mtp_phase3_kl.py      # §3  KL as-coded vs correct
scratchpad/mtp_phase3_e2e.py     # §4  load + one step on real checkpoints
```
