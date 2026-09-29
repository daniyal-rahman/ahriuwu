# One-off probes (NOT imported by anything; NOT tests)

Each file here answered one question once. Its result is in the ledger row
named below. Re-run only to reproduce that row; do not import from here.
Run from the desktop with `/mnt/nfs` paths (see `slurm/server_train.sbatch`).

| File | Question | Answer / ledger row | Date |
|---|---|---|---|
| `perf005_ray_kernel.py`, `perf005_vision_ab.py` | Does a fused ray traversal materially improve the whole update? | PERF-005; probe-only Pallas candidate, CPU interpreter check, GPU exact-ray/vision-suite gate and paired collection/full-update A/B | 2026-09-29 |
| `perf004_profile.py`, `perf004_instrument.py`, `perf004_analyze.py` | Full simulation, observation/action and PPO GPU cost attribution at fixed N128 | PERF-004; labelled original AST, numerical/graph canary, frozen random/E31 early/mid/late inputs, HLO mixed-fusion attribution and memory | 2026-09-29 |
| `wall_reward_probe.py` | Bucket recorded champion outcomes by wall distance | Pre-existing one-off preserved unchanged at the PERF-004 baseline; related to INT-001, not rerun or newly validated here | 2026-09-29 |
| `gru_throughput.py` | Actual GRU rollout versus learner wall time and memory at 16/128 envs? | PERF-003; `ops/launch.py PERF003_gru_profile`, split/fused canary, no saved model | 2026-09-29 |
| `jax_time_breakdown.py` | Where does current JAX decision time go on capped danilogin CPU? | PERF-002: populated-lane component timings and collision/help controls; no training | 2026-09-28 |
| `profile_collector.py` | Where does a server-collector decision's wall time go? | observe() was 4 ms/env of eager JAX; now ~0.9 ms/env. `THROUGHPUT` note in STATUS | 2026-09-25 |
| `probe.sh` | 8 servers at 30 Hz vs 10 Hz: is the server tick the bottleneck? | No: 1.0 s per 256 decisions either way (Python-bound) | 2026-09-25 |
| `red_route_probe.py` | Why does red stop at (12058,12979) walking to the top lane on the C# server? | Long-route pathfinder failure; legged route works (`TEAM_WAVE_START`) | 2026-09-25 |
| `jax_red_route_probe.py`, `jax_red_chain_probe.py` | Same question in the JAX sim | Stalls at (12154,12990); route A legs work (`JAX_RED_LEGS`) | 2026-09-25 |
| `smoke_mirror.sh`, `jax_smoke.sh` | Do train → resume → frozen-eval work end to end in mirror mode? | Yes, both engines | 2026-09-25 |

PERF006 reuses `perf005_vision_ab.py --bush-ab`: paired fused-ray versus static bush-ID collection/full-update timing; semantics deliberately differ (VIS-FAST).

| `replay_learning_summary.py` | Teacher/initial BC/final policy decisions on full replay histories; reward and decoded orders | LEARN-PAIR-04 / E36; shadow actions never executed | 2026-09-29 |

| `escape_counterfactual.py` | Do spells or alternate movement rescue the exact trapped-wave state? | LEARN-PAIR-05 / E37; frozen reactive opponent, restored full history | 2026-09-29 |
