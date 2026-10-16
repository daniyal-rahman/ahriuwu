# Status

**Current: 2026-10-16.** This worktree contains the 26.19 modern JAX world in `lanerl_jax/modern/`,
its trainer in `modern/train.py` + `modern/rl/`, tools in `ops/modern/`, and specs in `docs/modern/`.
Start here, then read `docs/CODEMAP.md` and `docs/modern/WORLD_IMPLEMENTATION.md`.
Detailed evidence and job history live in `docs/JAX_FIDELITY_LEDGER.md` (MODERN-027..047).

## Champions and world scope

14 kits registered: Garen, Jax, Nasus, Renekton, Darius, Malphite, Aatrox, Fiora, Yone, Ambessa,
Camille, K'Sante, Jayce, Ezreal. Known limitations remain explicit in `docs/modern/CHAMPIONS.md`.
Ornn was dropped at Dani's direction. The accepted cross-role plan is `docs/modern/ROSTER.md`;
11 planned additions remain after Ezreal. Champion implementation is paused while training memory
is investigated. Caitlyn is research only: exact trap/net attack refund and recoil evidence remain
unresolved. Lux has eight replay games extracted and child-spell timing evidence; no Lux kit registered.

The world supports two champions, one per team, not 5v5. More champions per team and ally targeting
remain prerequisites for the complete jungle/support roster, especially Lulu.

## Completed and validated

- Each world compiles only the matchup's kits or an explicit roster (MODERN-040).
- Ezreal + kit-pruning integration passed 998 tests across 69 modules (6150..6152), with identical
  full/top 3600-tick goldens at all 12 checkpoints (6153).
- Review fixes (MODERN-046, `76a4d59f`): independent ranged launch/hit metadata, accurate sequential
  damage-overflow counters, shield-order reduction hoisted out of the packet loop, stale comments trimmed.
  106 focused checks passed; all 12 golden checkpoint records match the accepted reference exactly.
- Follow-up (MODERN-047, `0e54c2d3`): simultaneous basic/basic and ability/basic hits preserve separate
  on-hit records, including item/rune/kit effects and Red Buff targets; combat resolves once per tick.
  139 focused checks passed; all 12 full/top golden records match exactly with zero overflow.
  Compiler estimates: Ezreal FLOPs +0.39%; GPU runtime cost has not been measured.

These are validation gates for their respective changes, not an additive test total. The full 69-module
suite preceded the latest review fixes; those fixes have focused regression and golden coverage.
No substantive modern-roster training experiment has run yet.

## Measured performance (before the latest review fixes)

RTX5080, kit pruning: top1024 134.7k ->148.8k env-ticks/s (+10.5%); top4096
168.3k ->178.6k (+6.1%); full1024 42.7k ->46.9k (+9.8%). Jobs6722/6723 completed.
Matched10Hz rollout + policy, top1024: modern36.0k vs legacy lane-rl/jax21.7k environment decisions/s
(+65.7%). Modern4096 reaches35.8k, showing saturation. These are different rulesets/scenarios,
not equivalent-game or full-training speedups. Results: `modern043-pinned-v3-TEMP/results-6734/`
under `/mnt/nfs/shared/champ-expansion/`; 35 cells retained, final cell cancelled after6h30.

Full PPO remains memory-blocked: matched256 OOM in both engines; modern natural256 also OOM.
Legacy natural256 PPO runs7.0k env decisions/s but uses one learner/AFK versus modern self-play2,
so it does not establish a matched end-to-end comparison.

## Active work and next actions

Only this thread's GPU memory-diagnosis job7588 remains queued for resources; unrelated7433 occupies
the desktop GPU. Do not cancel unrelated work. No review/champion subagent remains active.
CPU prerequisite7587 passed. MODERN-045 adds compiled/live memory diagnostics and eight bounded
full-PPO cases (64/128/256 envs, 75%/90% allocator fractions); no learner/allocator fix established yet.
Snapshot: `/mnt/nfs/shared/champ-expansion/modern045-memory-baseline-TEMP/`.
Log: `/mnt/nfs/shared/champ-expansion/modern045-ppo-memory-7588.out`.

1. Read7588 results, identify PPO memory cause, implement a justified fix and measure full update throughput.
2. Re-measure GPU throughput after the review fixes; retain exact-golden checks for pure refactors.
3. Resume one champion at a time, finishing evidence before implementation; Caitlyn/Lux are the researched candidates.
4. Establish a practical modern training configuration, then run a real experiment. Reward weights, input latency,
   start state and shop handling remain experiment decisions.
5. Expand to multiple champions and ally targeting before claiming a complete 5v5/support implementation.

Further performance options remain unmeasured in real training: smaller packet capacity, item allow-lists,
and `--lane-structures` (72 rather than88 top-lane units). Keep overflow counters at zero.
