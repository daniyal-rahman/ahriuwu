# Status

**2026-10-05 — repository cut to the modern world (MODERN-025).** This branch now holds only the 26.19 JAX
world (`lanerl_jax/modern/`), its tools (`ops/modern/`) and specs (`docs/modern/`). The legacy C#-server
project is at tag `pre-modern-cleanup-2026-10-05`. The package is self-contained: the trainer's policy,
PPO and learner live in `modern/rl/`; the camera/screen model in `modern/screen.py`; sight rays in
`modern/rays.py`.

**State of the world.** Full map verified bit for bit across the MODERN-024 restructure
(`ops/modern/golden.py`); the 88-unit top-lane layout is game-identical to the 216-unit one. Fidelity
checks against 26.9 replays and Riot timelines: docs/modern/REPLAY_FIDELITY.md, MECHANICS_AUDIT.md.

**Throughput (2026-10-07, MODERN-026).** Lane AI on its minion/structure rows and target columns, scatter
packet hooks, compacted fog rays and a cumsum compaction (`core.arrays.first_true`, `jnp.nonzero` is slow on
GPU): every change golden-identical. Full map 24.0k -> ~29k env-ticks/s and top lane 89k -> 105k at 1024/4096 envs
(per change, RTX 5080); combined numbers pending. Item allow-lists (`Loadout.allowed_items`,
`data/loadouts.py`: LoLalytics 16.19 Emerald+, 6-8 completed items per champion, Riot's recommended rune pages;
`/mnt/nfs/shared/build-research/`) compile out item code no champion can hold. Every capacity has a counter in
`TickEvents` and in the trainer's `sim_overflow_max`, which must stay 0.

**Kit pruning (2026-10-16, MODERN-040).** Each world compiles state and hooks only for its loadouts' kits,
plus an optional explicit roster. Before integration with Ezreal, full-map and top-lane 3600-tick CPU goldens
against `8067b443` match every retained leaf, summary and overflow counter (427 unused-kit leaves removed).
Trainer/world/interface checks passed (6084, 6086, 6087); non-world suite 6085 completed.
The corrected roster assertion passed in Slurm step 6085.0. GPU benchmark/profile jobs 6088/6092
were superseded by exclusive jobs 6722/6723; speedup is unmeasured. Final combined validation is below.

**Free wins held back.**
- `--packet-capacity`: runs peak at 3-6 packets per tick against 512/256 slots. Drop it for 1v1; measure on real
  5v5 RL runs before changing the default.
- Minion slots stay at 40 per lane (guaranteed to hold any wave); revisit with RL data.
- `--lane-structures`: only the spawning lanes' turrets and inhibitors plus the base (top lane 88 -> 72 units).
  Same game as the full set when nothing targets the other lanes' structures (golden `--lane-only` vs
  `--lane-structures`, 3600 chaos ticks: champions, minions and every shared structure identical); opt-in.
- Walkability (`terrain.row_gaps`): exact `is_walkable` from per-row nearest-blocked gaps, golden-identical;
  +54% top lane / +25% full map on the GTX 1060.

**Champions (2026-10-16, MODERN-027..039, MODERN-041).** Garen, Jax, Nasus, Renekton, Darius, Malphite,
Aatrox, Fiora, Yone, Ambessa, Camille, K'Sante and Jayce are registered (any pairing). Ezreal is implemented
and registered with full regression passing. Ornn is dropped; the accepted
cross-role roster is in docs/modern/ROSTER.md. Known differences are listed per kit in CHAMPIONS.md.
No training experiment with the new champions yet.

**Open.**
- GPU timing remains pending; desktop Linux is available, with exclusive GPU allocations waiting.
- No training experiment on the modern world yet; decisions open: reward weights, input latency
  (`--action-delay-ticks`), start state, shop handling.

**Roster expansion (2026-10-16, MODERN-041/042).** Ezreal's 16 kit tests, 3 world tests and 79 focused
regressions passed (6089/6090), plus trainer 6/6 (6103). Top-lane golden 6091 matches all 1159 pre-existing
leaves at six checkpoints through 3600 ticks (55 Ezreal leaves added).

After integrating Ezreal and kit pruning, all 69 test modules passed: 6150 (50 modules, 889 tests), 6151
(18 world modules, 103 tests), 6152 (trainer, 6 tests), total 998. Golden 6153 matches the pruned reference
at all 12 full/top checkpoints through 3600 ticks; strict leaf-name/hash, summary and peak-counter checks
also match. Immutable tested source: `/mnt/nfs/shared/champ-expansion/roster-integrated-e1b7a22c-TEMP`.

Caitlyn remains research only. Corrected extraction 6148 completed: 1560 Headshot records, 443 with one
plausible damaged champion, 71 ambiguous, 1046 unmatched. A single candidate is not proof of attribution;
the exact trap/net attack refund remains unresolved, and net dash profiles do not settle recoil distance.
Lux assessment (MODERN-044) now has eight replay games extracted (6707). R records are LuxRMis, whose
1 s child cast time fits damage candidates better than the root record's 1.375 s; W uses accelerating
missiles. No Lux kit registered yet. Actual rollout/PPO comparison against lane-rl/jax
is implemented and queued (MODERN-043); no training speedup claim yet.

**Training throughput comparison (2026-10-16, MODERN-043).** `ops/modern/training_bench.py` exports
immutable modern parent `78abf40f` and lane-rl `bb2e5b46` (4.20, explicitly a different game), plus hashed
route/vendor inputs, under `/mnt/nfs/shared/champ-expansion/modern043-pinned-TEMP/`. Matched GRU/PPO
and natural defaults measure real rollout+policy and full PPO independently at 256/1024/4096 envs,
modern top/full layouts. Reports env, sampled-agent and learner-agent decisions/s, normalized game
seconds/wall-s, full update seconds, separate init/compile/first execution, and device peak bytes.
Four metric-math tests plus the OOM continuation test pass locally. Desktop CPU validation 6725
COMPLETED/0:0 in 7m14s: real asset/bank setup and PPO shape tracing pass for both engines/presets
(envs4, rollout2, top lane); this is validation, not a compiled PPO execution or GPU timing result.
GPU training matrix 6734 (exclusive, 8h) is queued from `modern043-pinned-v3-TEMP/`.
Unstarted 6729 was superseded to correct observation metadata: dimensions now come from the pinned
modern module (self width 56), and actions are described as buttons. Original snapshots are preserved.
Earlier 6724 was superseded for the tested OOM collector fix.
No training throughput result or speedup claimed.
Saved submitted 6088/6092 scripts reveal missing `--exclusive`/strict shell mode and inaccessible cwd,
contrary to MODERN-040's earlier description. Cancelled only those IDs; replacement simulation benchmark
6722 (exclusive, 1h30) and profile 6723 (exclusive, 1h, afterany:6722) preserve their original scenarios
and source snapshots. All jobs have NFS cwd/logs, fail-on-error scripts, no reboot/admin changes or watcher.
Desktop reports Linux, `Gres=gpu:1` but no GPU in `CfgTRES`/partition TRES; CPU jobs also occupy it.
GPU queue reasons alternate resources/unavailable. Read-only NVIDIA query sees RTX5080, driver580.173.02,
18MiB used; GPU is visible. Scheduler repair is outside this task.

**Measured throughput (MODERN-043, 2026-10-16).** GPU jobs 6722/6723 completed. Kit pruning improves
simulation top1024 134.7k ->148.8k ticks/s (+10.5%), top4096 168.3k ->178.6k (+6.1%), full1024
42.7k ->46.9k (+9.8%). Training matrix6734 was cancelled after6h30 during the final cell; completed
results survive in modern043-pinned-v3-TEMP/results-6734/. Matched10Hz top-lane rollout+policy:
modern1024 36.0k vs legacy21.7k environment decisions/s (+65.7%); modern4096 35.8k (saturation).
These are different rulesets/scenarios. Full PPO is memory-blocked: matched256 OOM in both engines;
modern natural256 also OOM. Legacy natural256 PPO completes7.0k env decisions/s, but uses one learner
against AFK versus modern self-play2, so this is not a matched full-training comparison. Memory
investigation is delegated in an isolated checkout (MODERN-045), bounded representative cases only.

**PPO OOM diagnosis (2026-10-16, MODERN-045).** Reserved for learner memory work; MODERN-043 results
and champion work remain with the parent. Added compiled memory analysis before execution, allocator
settings, largest live arrays and optional live-device profile; isolated full-rollout post-KL forward,
minibatch forward and backward compile estimates. Bounded diagnostic driver: eight top-lane full-PPO
cases, rollout128, modern matched64/128, legacy matched64, modern natural128, modern matched/natural256
at explicit .75/.90 allocator limits. Three synchronized steady updates per successful case, 12-minute
cell cap, 1h45 exclusive Slurm allocation. Eight lightweight reporting/failure tests pass. Learner and
simulator unchanged; no remedy or training throughput improvement claimed. GPU is occupied by unrelated
7433 (scheduled end 2026-10-17 06:28 UTC); no cancellation or overlapping timing.

MODERN-045 submission: immutable `/mnt/nfs/shared/champ-expansion/modern045-memory-baseline-TEMP/`
contains original MODERN-043 engine pins plus diagnostics commit `576bc70d`; all 4301 manifest hashes
verified. CPU component-compiler validation **7587** passed (desktop, 1 test in8.6s);
exclusive GPU diagnosis **7588** (1h45, afterok:7587) is pending dependency/desktop availability.
Logs: `/mnt/nfs/shared/champ-expansion/modern045-components-7587.out` and
`modern045-ppo-memory-7588.out`; results will be `modern045-memory-baseline-TEMP/results-7588/`.
Cancelled only own unstarted **7584/7585** after detecting incomplete snapshot assembly; completed and
verified it before replacement submissions. No new GPU result, allocator conclusion or learner fix yet.
Next: inspect component memory and real full-PPO results, then justify/test a tight learner change and
compare against this unchanged baseline. Session ends with 7587/7588 queued; no watcher or agent.

**Modern-package review (2026-10-16, MODERN-046).** Fixed ranged on-hit target/raw/crit/cast-ID loss and
separated launch events; sequential damage-buffer drops now reach the world/trainer overflow metric.
Hoisted the full-unit shield-order max out of the per-packet loop, preserving exact grant-order values;
trimmed stale comments/docs. Desktop CPU7606 COMPLETED/0:0 in11m42s:104 focused checks and full/top3600-tick
goldens, all12 checkpoint records exactly equal (862/732 leaves, summaries, peak counters; overflow0).
Red7597 reproduced both initial bugs; corrected supplemental7611 COMPLETED/0:0 reproduces the parent's
missing Profane Cleave and passes both fixed ranged-arrival cases. Sources/logs under
`/mnt/nfs/shared/champ-expansion/modern046-review-TEMP/` and `modern046-*.out`.
Unstarted7601 superseded;7602 failed before Python (sh/pipefail);7604 hit8GiB after58 passes, resolved by
separate processes in7606;7608's missing-visibility fixture corrected in7611. All own review jobs ended;
no unrelated jobs touched, no new GPU work, no GPU speedup claim. MODERN-045 memory diagnosis remains
separate. Unresolved: simultaneous basic/ability hits still share one holder hit record; evidence and
proposed follow-up in the MODERN-046 ledger rows.

Review fixes integrated as `76a4d59f`; parent independently checked strict equality of all12 golden
records and Python syntax. A focused follow-up agent (MODERN-047) is reproducing/fixing the remaining
simultaneous-hit event loss, with targeted regressions and runtime-cost review required. Full-PPO GPU
memory diagnosis7588 remains queued; its CPU component-compilation prerequisite7587 passed.
