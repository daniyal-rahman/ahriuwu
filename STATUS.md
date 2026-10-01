# STATUS — 2026-10-01 19:46 UTC

Dani renewed the research budget in this thread. The old thread token limit and
15:37 compute window are obsolete. Current task: diagnose E68 caster misses,
then test visible history with controlled frozen evaluations. No broad PPO/LR
sweeps; no farming solution demonstrated. Long-term C# gate remains >30CS in
10minutes across seeds. Existing ledgers hold history and run details.

## Current phase

E71b preflight COMPLETE1839/0:0,4m16s.14GPU tests passed. E68 visible-history
audit:97022claimed historical samples,zero mismatches; one-step visible-minion
coverage97.26658%.15past samples available in all four inspected incident
frames. Single-game association evidence, not CV robustness or a learned gain.

Dani redirected the next step: list deviations from known working systems,
then establish a coherent published-reference baseline before component ablations.
LEARN-AFK-27 records the code-checked inventory, confirmed differences, unknowns,
and working priorities. Feature coverage remains the leading hypothesis, not
an established cause; global GRU can learn history. No GRU incapacity demonstrated.

E72b/1840 CANCELLED during startup at19:42UTC after5m02s (job0:0,batch0:15).
No canary-passed marker or training/frozen result; this is an intentional stop,
not evidence against training. E73b is deferred and never submitted. Opt-in
history implementation and successful E71b preflight remain preserved.
No project jobs running/queued; no result ETA. Next is reference-contract work,
not a submitted experiment. Unrelated1842/1803/1736 untouched. No1840 watcher
was armed because the startup gate was not reached; launcher exited on cancellation.

## Narrow investigation completed

- E69 reproduced every checked E68 state/action field over1201frames exactly;
  expected12CS/0deaths. Its typed-key comparator failed afterward; repaired in
  E69b. E69b1836 COMPLETED/0:0; both tick/ordinary baselines maxerror0.
- Caster15 dies80.953438s to minion14 projectile, ownwindup.1961s remaining.
  Caster19 dies113.280203s to minion20 projectile, ownCD.7847s remaining after
  nonlethal normal+Q hits. No missing champion hit or kill credit in these ticks.
- E701837 COMPLETED/0:0,6m28s. Preserve preceding useful actions and delay AA/Q:
  caster15 localCS1→2; caster19 localCS0→1. Every pre-intervention tick field
  matches baseline; no waiting swings. Ground-click displacement23.586units,
  so these are physical timing/position options, not timing-only causality.
- LEARN-AFK-24/25 contain exact timestamps, hit damage, attribution and limits.
  This is local JAX execution evidence, not a C# paired-trace parity proof.

## Feature implementation and source

Opt-in v5 adds15past10Hz health/position/known samples per visible entity via
mutual position/type association, retaining existing shared attention/pooling
and globalGRU. No raw unit IDs/targets/projectiles/AA clocks enter the actor.
Separate zero-weight projection preserves initial checkpoint outputs. Observer
memory crosses rollouts, resets with episodes and is included in checkpoints.
Experimental collector currently supports10Hz AFK; C# driver rejects v5 until
its history collection is validated. LEARN-AFK-26 records research/gates.

Current E67 aggregate frozen baseline:9.921875CS/.140625deaths, NOT solved.
Checkpoint:
`/mnt/nfs/checkpoints/lanerl-jax/E67_afk_staggered/vec-s0-20261001-105559-3d1c1096/ckpt_008388608.msgpack`
SHA256 `7f12479d61db507d23806822a6c5b1d4b129bd58fd9e83b2bed1c06621d0fe55`.
E68 trace: `/mnt/nfs/shared/E68_afk_current_video/low_team_1/trace.npz`.
E69b/E70/E71b artifacts use their experiment names under `/mnt/nfs/shared/`.

E69/E69b/E70 watchers stopped, inactive/registry clean verified. E71 failed a
float64 test fixture before training; E71b corrected it. Neither E71 attempt
needed an unattended watcher. E72/E73 prepared configs superseded before
submission by E72b/E73b to reference successful E71b. E64 remains unsubmitted;
older E40 checkpoint remains preserved and must not be silently resumed.

All starts/stops/session endings update this file in the same commit. For each
unattended job, verify startup watch then active/updating event bridge. On wake
check State AND ExitCode, frozen results/checkpoints, and watcher/registry cleanup.
