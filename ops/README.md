# ops/ — tools for the modern world

All in `ops/modern/`, run as `python -m ops.modern.<tool> --help`.

| tool | what it does |
|---|---|
| `bench` | Times `jit(vmap(scan(step)))` for several batch sizes and world variants (`--lanes`, `--no-jungle`, `--no-objectives`, `--fog`). |
| `profile_tick` | Attributes traced XLA op time to source lines (GPU: disables command buffers). |
| `perf_guard` | Throughput/memory regression guard against `ops/modern/perf_baseline.json`; `sbatch slurm/modern_perf_guard.sbatch [--cpu] [--update-baseline]` (exclusive node). |
| `jaxpr_fingerprint` | Name-free fingerprint of the traced tick (full map and top lane). Equal = same computation; use it for no-op refactors. |
| `golden` | Runs both worlds 3600 ticks under seeded chaos orders and hashes every state leaf (plus a layout-free game summary); `--compare` / `--diff` check refactors bit for bit. |
| `replay_fidelity` | Compares the sim with 16.9 replay observations (respawn, death timers, gold while dead, Homeguard). |
| `replay_oracle_extract` (+ `.sbatch`) | Extracts the replay observations from the 147-game corpus. |
| `riot_stats_oracle` | Compares the stat pipeline with Riot match-v5 timeline champion stats. |
| `collision_replay`, `collision_creepblock` | Creep-block evidence from replays vs the sim's collision. |
| `items_bench` | Cost of one full item tick. |
| `fetch_map` | Extracts pinned Map11 assets from a Riot manifest (no game install). |
