# Status

**Current: 2026-10-17.** This worktree contains the 26.19 modern JAX world in `lanerl_jax/modern/`,
its trainer in `modern/train.py` + `modern/rl/`, tools in `ops/modern/`, and specs in `docs/modern/`.
Start here, then read `docs/CODEMAP.md` and `docs/modern/WORLD_IMPLEMENTATION.md`.
Detailed evidence and job history live in `docs/JAX_FIDELITY_LEDGER.md` (MODERN-027..049).

## Champions and world scope

14 kits registered: Garen, Jax, Nasus, Renekton, Darius, Malphite, Aatrox, Fiora, Yone, Ambessa,
Camille, K'Sante, Jayce, Ezreal. Known limitations remain explicit in `docs/modern/CHAMPIONS.md`.
Ornn was dropped at Dani's direction. The accepted cross-role plan is `docs/modern/ROSTER.md`;
11 planned additions remain after Ezreal. Simulator profiling/throughput is the immediate priority. Caitlyn is research only: exact trap/net attack refund and recoil evidence remain
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
- Shield valuation (MODERN-049, 2026-10-17) selects the target row before decay arithmetic.
  Desktop CPU compiler7791 confirms smaller HLO work; restricted top-lane Garen/Ezreal tick
  FLOPs estimates fall 1.16%/1.29%, with unchanged compiled memory. GPU throughput remains unmeasured.
  Validation7792 completed: 211 focused checks passed; all 12 full/top 3600-tick golden
  records match the accepted reference exactly, with zero overflow. Integrated as `38d6d04e`.

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

## Ownership and active work

Dani clarified ownership on2026-10-17: this thread owns the simulator, champions, correctness and sim
throughput under training workloads. Other agents own policy/PPO/learner and the rest of the stack.
Own PPO diagnostic7588 was stopped (final elapsed21m20s); partial results are preserved for handoff at
`/mnt/nfs/shared/champ-expansion/modern045-memory-baseline-TEMP/results-7588/`.
No learner/allocator changes were made. CPU diagnostic prerequisite7587 passed previously.

MODERN-048 improves sim measurement: five consecutive synchronized samples, normalized game-time rates,
explicit timed game window, minion occupancy, memory/overflow and frozen-game checks. Optional scripted
ability attempts exercise kits; the queued comparison will include later lane combat after5400 warm ticks.
All five timing/baseline-key/order tests and compiled CLI validation7793 passed on desktop;
exclusive GPU comparison7796 waits afterok:7793, and profiler7795 waits afterany:7796. Earlier unstarted
7794 was superseded to use Jayce/Garen for the ranged case (the tool uses top-lane rune recommendations;
Ezreal only has bottom-lane research). Comparison uses identical driver over baseline78abf40f and candidate
80bd8f0c, with default/later ability workloads, top1024/4096, full1024 and ranged Jayce/Garen1024.
Immutable sources and driver hash: `/mnt/nfs/shared/champ-expansion/sim048-throughput-TEMP/`.
MODERN-049 shield optimization is integrated and validated. Exclusive GPU comparison7797 waits afterany:7795
and isolates that change: baseline80bd8f0c versus optimized38d6d04e, with identical drivers and workloads.
Pinned sources: `/mnt/nfs/shared/champ-expansion/sim049-throughput-TEMP/`. Jobs7796/7795/7797 remain queued;
no new GPU throughput result is available. Policy/PPO code is outside this thread's scope.

Next actions:
1. Validate the revised sim benchmark and remeasure pre/post review code under identical benchmark drivers.
2. Profile current later-lane combat, optimize measured sim hotspots, and retain exact fidelity for refactors.
3. Test smaller packet capacity, item allow-lists and lane-structure reduction against comparable sim workloads;
   keep every overflow counter at zero and separate opt-in scenario changes from exact optimizations.
4. Resume one champion at a time, finishing evidence before implementation (Caitlyn/Lux researched).
5. Expand multiple champions/ally targeting before claiming complete5v5/support implementation.

Existing timings above predate the latest review fixes. New speedups remain unmeasured.
