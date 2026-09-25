# One-off probes (NOT imported by anything; NOT tests)

Each file here answered one question once. Its result is in the ledger row
named below. Re-run only to reproduce that row; do not import from here.
Run from the desktop with `/mnt/nfs` paths (see `slurm/server_train.sbatch`).

| File | Question | Answer / ledger row | Date |
|---|---|---|---|
| `profile_collector.py` | Where does a server-collector decision's wall time go? | observe() was 4 ms/env of eager JAX; now ~0.9 ms/env. `THROUGHPUT` note in STATUS | 2026-09-25 |
| `probe.sh` | 8 servers at 30 Hz vs 10 Hz: is the server tick the bottleneck? | No: 1.0 s per 256 decisions either way (Python-bound) | 2026-09-25 |
| `red_route_probe.py` | Why does red stop at (12058,12979) walking to the top lane on the C# server? | Long-route pathfinder failure; legged route works (`TEAM_WAVE_START`) | 2026-09-25 |
| `jax_red_route_probe.py`, `jax_red_chain_probe.py` | Same question in the JAX sim | Stalls at (12154,12990); route A legs work (`JAX_RED_LEGS`) | 2026-09-25 |
| `smoke_mirror.sh`, `jax_smoke.sh` | Do train → resume → frozen-eval work end to end in mirror mode? | Yes, both engines | 2026-09-25 |
