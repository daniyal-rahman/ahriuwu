# STATUS — 2026-10-01 10:49 UTC

Current task: diagnose and improve JAX AFK farming, using frozen 120-second
trials. No demonstrated solution yet. Dani authorizes bounded reversible
experiments through **15:37 UTC today**; preserve checkpoints and use Slurm.
Longer-term transfer gate remains >30 CS in a 10-minute C# mirror trial.
History and evidence: `docs/EXPERIMENTS.md`, `docs/JAX_FIDELITY_LEDGER.md`.

## Running

**E67_afk_staggered — Slurm1833**, started10:47UTC,1-hour allocation.
15standardGPUcanaries passed364.28s; additional warmup/GRU canary passed80.81s;
launcher180s startup watch passed. Bridge lanerl-event-1833.service verified
active/result updating, expiry2026-10-01T12:54:52.537235+00:00. Fullsize warmup next.
Same E63 full objective and PPO: E46/v3/freshAdam/LR1e-4/4epochs/gamma.99,
XP0/HP100/personal tower900/death300. Only training phases differ.
Fixed-policy warmup discards initial shortened episodes; then all training games
and frozen evaluations retain120s length.512updates8.389M plus180224untrained
warmup decisions. Success>=10.5625CS/<=.05deaths, sameinitial9.5625CS required.
Expected finish11:25–11:35UTC. Log `/mnt/nfs/shared/E67_afk_staggered-1833.out`.

E65 Slurm1832 COMPLETED/0:0 in38m01s. FrozenCS9.5625→5.359375→7.109375→3.546875;
final deaths.015625/towerHP3.191875/reward1.953154. Reward terms:gold2.210945,
death−.234375,position−.023416,XP0. RemovingHP/tower rewards did not rescue
farming, even on the simplified objective; no extension. Checkpointstep8388608
finite/latest identical;512metrics/nonfinite0; medianupdate3.3350s; sourceSHA
and disabledHP/tower settings verified. Watcher stopped after active review;
inactive/registry clean. Run
`/mnt/nfs/checkpoints/lanerl-jax/E65_afk_cs_focus/vec-s0-20261001-101545-cb09e527/`.

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
- E65 failed the simpler farming objective. Secondary reward conflict alone
  is not a sufficient explanation; do not promote reward removal as a fix.
- E67 now tests training batch phase diversity; LEARN-AFK-20.
- E64 longer discount horizon remains prepared, dry-run passed, NOT submitted.

Older E40 mirror training remains interrupted at5795updates; checkpoint
`/mnt/nfs/checkpoints/lanerl-jax/E40_tower_wave_extended/vec-s0-20260930-150246-c731fd96/ckpt_189890560.msgpack`
preserved. Do not silently resume it or alter unrelated jobs.

All starts/stops/session endings update this file in the same commit.
On completion inspect Slurm State AND ExitCode, logs, checkpoints and frozen
scores; update existing ledgers and verify bridge/registry cleanup.
