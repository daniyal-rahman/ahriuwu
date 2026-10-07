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

**Free wins held back.**
- `--packet-capacity`: runs peak at 3-6 packets per tick against 512/256 slots. Drop it for 1v1; measure on real
  5v5 RL runs before changing the default.
- Minion slots stay at 40 per lane (guaranteed to hold any wave); revisit with RL data.
- `--lane-structures`: only the spawning lanes' turrets and inhibitors plus the base (top lane 88 -> 72 units).
  Same game as the full set when nothing targets the other lanes' structures (golden `--lane-only` vs
  `--lane-structures`, 3600 chaos ticks: champions, minions and every shared structure identical); opt-in.
- Walkability (`terrain.row_gaps`): exact `is_walkable` from per-row nearest-blocked gaps, golden-identical;
  +54% top lane / +25% full map on the GTX 1060.

**Open.**
- GPU timing of the combined branch, packet capacity 64/32 (1v1) and the allow-list: queued, desktop in Windows.
- No training experiment on the modern world yet; decisions open: reward weights, input latency
  (`--action-delay-ticks`), start state, shop handling.
