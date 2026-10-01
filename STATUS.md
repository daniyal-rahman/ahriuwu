# STATUS — 2026-10-01 09:42 UTC

Current task: diagnose and improve JAX AFK farming, using frozen 120-second
trials. No demonstrated solution yet. Dani authorizes bounded reversible
experiments through **15:37 UTC today**; preserve checkpoints and use Slurm.
Longer-term transfer gate remains >30 CS in a 10-minute C# mirror trial.
History and evidence: `docs/EXPERIMENTS.md`, `docs/JAX_FIDELITY_LEDGER.md`.

## Running

**E66_afk_click_proposals — Slurm1831**, started09:25UTC, 1-hour allocation.
Only this thread's active training job; other users/tasks' jobs are untouched.
E46 weights, fresh Adam, original v3 inputs, LR1e-4,4epochs,gamma.99,lambda.95;
XP0, HP100gold/fullbar, personal tower900gold/fullbar, death300gold.
New learned visible-entity screen-cell proposals mixed with ordinary ground
clicks, initialized10percent. No simulator, visibility, or action-protocol change.
512updates/8.389M decisions; frozen64-game evaluations at0/128/256/512.

15 standard GPU canaries and4 proposal tests passed, including real GRU
collector/learner likelihood and finite PPO update. Required180s startup watch
passed. Fullsize compilation and endpoint-before-reset canary passed.
Frozen u0: **8.890625 CS, .015625 deaths, 4.044495 personal towerHP**;
new initial distribution differs from old E46 despite retaining old weights.
Training active; first27updates median3.6155s, versus E63 median3.3476s.
At update70 all losses finite; proposal use about2.6percent when available.
These are training diagnostics, not performance results.

Run: `/mnt/nfs/checkpoints/lanerl-jax/E66_afk_click_proposals/vec-s0-20261001-093416-d26024a8/`.
Log: `/mnt/nfs/shared/E66_afk_click_proposals-1831.out`.
Bridge: **lanerl-event-1831.service**, verified active/result updating;
expiry **2026-10-01T11:34:21.708261+00:00**.
Artifacts: `/mnt/nfs/shared/slurm-events/1831/`.
Approximate completion10:10–10:20UTC, subject to measured runtime.

## Latest evidence and next decisions

- Original best E46 frozen score: **9.5625 CS,0 deaths**. Source retained:
  `/mnt/nfs/checkpoints/lanerl-jax/E46_afk_farm/vec-s0-20260930-214903-32499e76/ckpt_010027008.msgpack`.
- E61 longer original control:6.0625CS/1death/462.117towerHP. Reward rose
  while farming deteriorated: tower diving exploited absent explicit death cost.
- E62 own-action inputs plus death300:8.828125CS/0deaths. E63 original
  inputs plus death300:8.921875CS/.015625deaths/1.442893towerHP.
  Death cost suppressed dives; neither improved farming. No extensions.
  E63 Slurm1830 COMPLETED/0:0 in38m03s;512updates, finalstep8388608,
  finite checkpoint/latest identical,512metrics/nonfinite0, exact death-cost
  accounting. E62/E63 watchers stopped after active review; registry cleaned.
- E58b scripted positive control achieved13.5CS without deaths; interface can
  farm better. E59 selected caster interventions had small/mixed returns.
  LEARN-AFK-19 direct-click probability analysis motivates E66 but does not
  establish a root cause or a66x gameplay improvement.
- E66 success requires CS>=max(10.5625,its own initialCS+1), deaths<=.05,
  then an independent training/evaluation seed repeat. Do not select a winner
  from training CS or total reward alone. If promising, replicate before claiming
  a solution; inspect normal-speed gameplay for farming/tower behavior.
- E64 longer discount horizon and E65 farming-focused reward diagnostic are
  prepared and dry-run validated, **NOT submitted**. E65 is an easier diagnostic,
  not a substitute for the intended farming-plus-tower objective.

Older E40 mirror training remains interrupted at5795updates; checkpoint
`/mnt/nfs/checkpoints/lanerl-jax/E40_tower_wave_extended/vec-s0-20260930-150246-c731fd96/ckpt_189890560.msgpack`
preserved. Do not silently resume it or alter unrelated jobs.

All starts/stops/session endings update this file in the same commit.
On completion inspect Slurm State AND ExitCode, logs, checkpoints and frozen
scores; update existing ledgers and verify bridge/registry cleanup.
