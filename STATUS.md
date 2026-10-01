# STATUS — 2026-10-01 18:57 UTC

Current authorization: renewed research budget in this thread, narrow E68
caster investigation first; no broad PPO/LR sweeps. The old budget/window below
are historical. Feature experiment conditional on execution checks.

E69 Slurm1835 FAILED/1:0 at5m36s: full1201frame E68 reproduction exact on all
checked state/action fields,12CS/0deaths, correct checkpointSHA. Probe then
failed converting typed JAX PRNG key in all-state-leaf comparator. No valid
branch conclusion. Comparator repaired; E69b prepared from saved baseline
state/carry/key, same76branches and30min cap, Slurm1836 running;12canaries/180s startup watch passed.
E69 watcher stopped after review; inactive/registry clean verified.
E69b/1836 is the only project job. Unrelated1803/1736 untouched.
Restored baseline states match E68 exactly. Tick compilation/branches pending,
no valid branch findings yet. Rough ETA5–20min; hard Slurm end19:27UTC.
Bridge `lanerl-event-1836.service` active; expiry 2026-10-01 20:00 UTC.
On wake inspect E69b/result.json and unit15/unit19 event timelines; require
all-state instrumentation agreement. Compare target CS AND local total CS;
check unit19 Q reset. Verify watcher/registry cleanup after active review.

## Previous handoff (historical)

**Follow-up paused at budget boundary:** Dani approved the narrow tick-level caster investigation. Read repository instructions and existing counterfactual launcher; no simulation, code changes or jobs started before the budget-limited notice arrived. Remaining work: reproduce E68 incidents near81s/113s at60Hz, verify attack/damage/credit ordering, then matched earlier/delayed attack branches. Requires renewed research budget. No new results or solution claim.

**Latest requested feature comparison:** Completed primary-source Tencent Solo/5v5, OpenAI Five and AlphaStar comparison (LEARN-AFK-23). Explicit HP history and combat timing are supported precedents; E67 lacks these inputs. Saved-trace projectile timings support the two caster timing explanations, without proving simulator parity or neural causality. No code/training change, no submitted or queued jobs. Next targeted work: tick-level matched-action audit, then controlled visible-history feature experiment if execution is correct.

**User-requested E68 behavior review:** Saved-trace checks identify correct-target late swing near81s and nonlethal swing/cooldown miss near113s; detailed timestamps in LEARN-AFK-22. E67 retained HP100/death300/personal tower900 rewards. Current inputs lack explicit per-minion HP trends; shared entity encoder pools before globalGRU. No proof of simulator correctness or neural causal explanation. Next narrow check: tick-level execution and visible-history dependence for these incidents; no hidden target-ID feature added. No jobs launched by this review.

**E68 VIDEO COMPLETE:** User-requested latestE67 full120s replay, Slurm1834 COMPLETED/0:0 in4m40s. Canaries/180s watch passed; checkpointSHA matches E67final. Both map and combat MP4s verified120.1s/10fps. Seed7/fullHP blue versusAFK:12CS/0deaths, one diagnostic game. Combat video `/mnt/nfs/shared/E68_afk_current_video/low_team_1/combat/replay.mp4`; map sibling `replay.mp4`. Watcher1834 stopped after active review; inactive/registryclean. No training resumed; no active jobs from this request.

# STATUS — 2026-10-01 11:30 UTC

Current task: diagnose and improve JAX AFK farming, using frozen 120-second
trials. No demonstrated solution yet. Dani authorizes bounded reversible
experiments through **15:37 UTC today**; preserve checkpoints and use Slurm.
Longer-term transfer gate remains >30 CS in a 10-minute C# mirror trial.
History and evidence: `docs/EXPERIMENTS.md`, `docs/JAX_FIDELITY_LEDGER.md`.

## No active runs — token budget reached

All this thread's submitted runs finished; no new experiment authorized beyond
the exhausted1M-token goal budget. No solution demonstrated. GPU window had
remaining time, but token budget is the stopping constraint. Existing checkpoints
and opt-in implementations retained. E64 remains prepared, not submitted.

E67 Slurm1833 COMPLETED/0:0 in41m44s. Frozen CS9.5625→10.0625→6.703125→9.921875;
final deaths.140625/personal towerHP802.399197/reward24.160935. Final reward
components:gold7.710979,death−2.109375,health−4.736133,tower23.295461,position~0.
Staggering greatly improved tower play in this seed, but failed predeclared
CS>=10.5625/deaths<=.05 gate. No independent-seed confirmation or farming fix.
Checkpointstep8388608 finite/latest identical,512metrics/nonfinite0,
medianupdate3.63235s. Fullsize warmup discarded180224untrained decisions;
initial frozen gameplay exactly matchedE46. Watcher1833 stopped after active
review; inactive/registry clean. Run
`/mnt/nfs/checkpoints/lanerl-jax/E67_afk_staggered/vec-s0-20261001-105559-3d1c1096/`.

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
- E67 demonstrates much stronger tower learning in one seed, not reliable
  farming improvement. A useful next controlled test is E65 farming-focused
  rewards with E67 phase staggering, compared against both parents. NOT
  implemented/submitted. If it improves CS, independently replicate and then
  restore/validate the intended full objective before claiming a solution.
- E64 longer discount horizon remains prepared, dry-run passed, NOT submitted.

Older E40 mirror training remains interrupted at5795updates; checkpoint
`/mnt/nfs/checkpoints/lanerl-jax/E40_tower_wave_extended/vec-s0-20260930-150246-c731fd96/ckpt_189890560.msgpack`
preserved. Do not silently resume it or alter unrelated jobs.

All starts/stops/session endings update this file in the same commit.
On completion inspect Slurm State AND ExitCode, logs, checkpoints and frozen
scores; update existing ledgers and verify bridge/registry cleanup.
