# STATUS — 2026-10-01 10:10 UTC

Current task: diagnose and improve JAX AFK farming, using frozen 120-second
trials. No demonstrated solution yet. Dani authorizes bounded reversible
experiments through **15:37 UTC today**; preserve checkpoints and use Slurm.
Longer-term transfer gate remains >30 CS in a 10-minute C# mirror trial.
History and evidence: `docs/EXPERIMENTS.md`, `docs/JAX_FIDELITY_LEDGER.md`.

## Running

**E65_afk_cs_focus — Slurm1832**, started10:09UTC,1-hour allocation.
15 GPU canaries passed (367.68s), launcher180s healthy-start watch passed.
Bridge lanerl-event-1832.service verified active/result updating; expiry2026-10-01T12:15:35.617291+00:00.
Fullsize compilation/endpoint canary passed. Frozen u0 exactly9.5625CS/0deaths/0towerHP,
reward7.3125407 entirely gold/CS; component accounting passed. Training active. Frozen u128:5.359375CS/.109375deaths/0towerHP/reward2.317232;
no early farming rescue from removing HP/tower rewards.
Run `/mnt/nfs/checkpoints/lanerl-jax/E65_afk_cs_focus/vec-s0-20261001-101545-cb09e527/`.
E46 weights, fresh Adam, original v3 inputs, LR1e-4,4epochs,gamma.99,lambda.95;
XP0, HP-loss reward0, tower reward0, death300gold. Same objective asE63 except
removing the two secondary HP/tower terms together.512updates/8.389M decisions;
frozen64-game evaluations at0/128/256/512. Success>=10.5625CS/<=.05deaths.
This is an easier diagnostic, not a substitute for the intended full objective.
Approximate completion10:45–10:50UTC; log `/mnt/nfs/shared/E65_afk_cs_focus-1832.out`.
Future frozen evaluations now report reward components with an accounting check.

E66 Slurm1831 COMPLETED/0:0 in43m32s. Final512update frozen score:
**9.34375CS,0deaths,75.222872personal towerHP,reward8.102879**.
Own initial8.890625CS/.015625deaths/4.044495towerHP/reward4.98031;
originalE46baseline9.5625CS. Some safe tower damage learned, but farming gate
failed; no extension and no solution claim. Checkpointstep8388608 finite/latest
identical;512metrics/nonfinite0; medianupdate3.62076s (~8percent slower thanE63).
Run `/mnt/nfs/checkpoints/lanerl-jax/E66_afk_click_proposals/vec-s0-20261001-093416-d26024a8/`.
Watcher1831 stopped after active review; inactive and registry clean.

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
- E66 did not pass its predeclared farming threshold. Retain the opt-in code
  and checkpoints as evidence; do not describe higher total reward as a CS fix.
- E65 now tests farming learnability without secondary reward tradeoffs. If it
  improves, independently validate and restore the intended objective before
  calling the full problem solved.
- **E67_afk_staggered queued Slurm1833** behind1832; dry-run passed,
  launcher session14791 awaits startup. GPU warmup/likelihood canary pending;
  no bridge yet, arm after startup watch. Same E63 full reward/PPO settings;
  discards shortened warmup episodes before learning and preserves full120s
  training/evaluation games.512updates8.389M plus180224untrained warmup
  decisions; estimated40min after start. LEARN-AFK-20 records evidence/limits.
- E64 longer discount horizon remains prepared, dry-run passed, NOT submitted.

Older E40 mirror training remains interrupted at5795updates; checkpoint
`/mnt/nfs/checkpoints/lanerl-jax/E40_tower_wave_extended/vec-s0-20260930-150246-c731fd96/ckpt_189890560.msgpack`
preserved. Do not silently resume it or alter unrelated jobs.

All starts/stops/session endings update this file in the same commit.
On completion inspect Slurm State AND ExitCode, logs, checkpoints and frozen
scores; update existing ledgers and verify bridge/registry cleanup.
