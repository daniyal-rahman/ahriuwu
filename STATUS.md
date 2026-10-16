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
The corrected roster assertion passed in Slurm step 6085.0. GPU benchmark 6088 and dependent profile 6092
remain pending; speedup is unmeasured. Desktop Linux is available again; final combined validation is below.

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
- GPU timing of the combined branch, packet capacity 64/32 (1v1) and the allow-list: queued, desktop in Windows.
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
is in progress in an isolated agent checkout (MODERN-043); no training speedup claim yet.
