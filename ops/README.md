# ops/ — production infrastructure (NOT disposable)

These run unattended and other things depend on them. `scratchpad/` is for
throwaway experiments; anything a live run needs lives here.

Current JAX operations: `login_capped.sh` bounds login-node workloads;
`sim_tests_per_file.sh` runs simulator suites per file; `desktop_suite.sh`
runs capped suites on the desktop when it is available. Read each script's
usage and the [canonical gate commands](../docs/REPRODUCING_GATES.md) before
launching. Current project scope and file-placement rules are in
[PROJECT.md](../docs/PROJECT.md). New disposable work goes into named ignored
run directories; existing scratchpad callers need an audit before relocation.

The following table is the historical BC operational runbook. Its existence
does not mean these jobs are currently running or the desktop is available.

| file | what it does |
|---|---|
| `bc_night.sh supervise` | owns the nightly BC window (06:00-18:00 UTC = 11pm-11am PT). Launch detached on the desktop; does NOT survive a reboot, re-arm after one. |
| `bc5080_gate_watchdog.sh` | keeps the BC trainer alive inside the window; resumes from the last checkpoint on crash. Started by bc_night.sh. |
| `tok_eval_watcher.py` | scores tokenizer checkpoints on fixed held-out sets as they appear. |
| `stage_desktop_standalone.sh` | builds the login/NFS-independent inference bundle at /mnt/storage/ahriuwu-live. |

Status: `bash ops/bc_night.sh status`


Tools of the 26.19 modern world (`lanerl_jax/modern/`) live in `ops/modern/`.

`ops/modern/fetch_map.py` is a TOOL for the modern world port (MODERN-010). It
extracts requested Map11 WAD members from an explicitly pinned Riot manifest;
it does not install or run League. Use an isolated Python3.11 environment with
`riotmanifest==2.10.2`, `league-tools==1.2.1`, `zstd==1.5.7.2`. Run through
`login_capped.sh`. Both extraction and normalization require new output dirs.

```sh
ops/login_capped.sh 4G 2 /path/to/asset-venv/bin/python ops/modern/fetch_map.py --manifest https://lol.dyn.riotcdn.net/channels/public/releases/4D2A50D5EDAB724A.manifest --build 16.19.8230722 --out /path/to/new-extraction --asset assets/maps/navgrid/map11/aipath_srx_2.aimesh_ngrid --asset assets/maps/deprecated/map11/cfg/objectcfg_srx.cfg --asset data/maps/shipping/map11/map11.bin
ops/login_capped.sh 4G 2 /path/to/numpy-python -m lanerl_jax.modern.data.navgrid import --ngrid /path/to/new-extraction/assets/assets/maps/navgrid/map11/aipath_srx_2.aimesh_ngrid --out /path/to/new-grid --patch 26.19 --source https://lol.dyn.riotcdn.net/channels/public/releases/4D2A50D5EDAB724A.manifest --retrieved-at 2026-09-30
```

The date above reproduces the existing artifact's provenance; a new source
retrieval should use its actual date and receive a reviewed new manifest pin.
Compare the extraction hashes to `lanerl_jax/modern/data/26.19/map11.json`.
`load_patch_map(path, patch="26.19")` requires that profile's exact manifest
hash and checks its arrays; `load_artifact` without an external manifest pin
only checks self-reported identity and accidental array corruption. Neither
loader resolves latest or substitutes the legacy map. Shared validated arrays:
`/mnt/nfs/datasets/league/26.19/map11-base/`. Raw extraction receipts:
`/mnt/nfs/shared/modern-world-map-research/live-16.19.8230722/`.

`ops/modern/items_bench.py` is a TOOL for the modern item system (MODERN-013). It
times one full `items.effects.runtime.item_tick` (two champions with six
items each, 66 units) single and vmapped; run it capped:
`ops/login_capped.sh 8G 2 .venv-jax/bin/python -m ops.modern.items_bench 64`.
The item table is rebuilt from the cached 16.19 client bins with
`python -m lanerl_jax.modern.data.build_items` (sources and sha256s are
recorded in `items_client.json`).

`ops/modern/replay_oracle_extract.py` + `replay_oracle_extract.sbatch` are a TOOL
(MODERN-015) that extracts raw client-memory observations (gold, deaths and
respawns, fountain stretches, level-ups, max-HP changes) from the 147-game
16.9 replay corpus `/mnt/nfs/datasets/lol_replays_16_9_772/`, independent of
the simulator. Submit with `sbatch ops/modern/replay_oracle_extract.sbatch` (CPU
partition, job array); per-game output, the index and job logs (`logs/`) go to
`/mnt/nfs/shared/replay-oracle-16.9/`, the compact summary to
`lanerl_jax/modern/data/oracle/replay_16_9_observations.json.gz`, which
`lanerl_jax/modern/tests/test_economy_oracle.py` reads. The 16.9 champion
HP records it is paired with (`oracle/champion_hp_16_9.json`) were fetched from
CommunityDragon 16.9 into `/mnt/nfs/shared/replay-oracle-16.9/champions-16.9/`.

`riot_stats_oracle.py` is a TOOL (MODERN-016). `extract` turns Riot match-v5
match + timeline JSON (fetched to `/mnt/nfs/shared/riot-match-v5-16.9/` by
`fetch_matches.py` there; API key from `RIOT_API_KEY`, never written to disk)
into the anonymised `lanerl_jax/modern/data/oracle/riot_16_9_frames.json.gz`;
`predict` (used by `modern/tests/test_stats_riot_oracle.py`) recomputes Riot's
`championStats` through the modern stat pipeline, shards, rune and item stat
hooks with 16.9 records (`oracle/client_16_9_stats.json`).

`ops/modern/bench.py` is a TOOL (MODERN-017/018). It times `lax.scan` of the
modern world tick (`world.tick.step`) under `jit(vmap)` for several batch
sizes with scripted in-scan orders and prints one JSON line per batch size.
World-variant flags (`--fog rays|fast|off`, `--no-jungle`, `--no-objectives`,
`--lanes`) build ablated worlds, to measure what each system costs.
`ops/modern/profile_tick.py` (TOOL, MODERN-021) traces the same scan with
`jax.profiler` and attributes device op time to source files and lines (joined
through the compiled HLO's stack-frame tables); use it to find hotspots before
optimising. Run both on the desktop through Slurm (`gpup` for GPU, `cpu` for
CPU) from an NFS code snapshot, because the desktop cannot see the worktree.

`ops/modern/perf_guard.py` + `slurm/modern_perf_guard.sbatch` are a TOOL: a
throughput and memory regression guard for the modern world tick. It runs the
`ops/modern/bench.py` scan for one fixed config per backend (256 envs on GPU,
16 on CPU; 1800 warm ticks, 150 ticks per call, fastest of 3 timed calls) and
records env-ticks per second, the compiled program's temp buffer bytes and, on
GPU, `peak_bytes_in_use` from `jax.devices()[0].memory_stats()`. It compares
them to `ops/modern/perf_baseline.json` (keyed by backend, device kind and config)
and exits 1 on a throughput drop or memory growth over `--tolerance` (15%),
3 when the key has no baseline. `--update-baseline` records the run as the new
baseline for its key; commit the JSON with the change that moved it. Submit
`sbatch slurm/modern_perf_guard.sbatch [--cpu] [--update-baseline]` (gpup,
exclusive node, 45 min; log in `/mnt/nfs/shared/modern-perf-guard/`, create it
first); set `CODE=<NFS snapshot> sbatch --export=ALL ...` to measure a snapshot
instead of `/mnt/nfs/projects/ahriuwu-lanerl-jax`.

`ops/modern/golden.py` is a TOOL (MODERN-024) for behaviour-preserving refactors of the modern world: it runs the
full-map and top-lane worlds for 3600 ticks on CPU under a fixed stream of random "chaos" orders (moves, attacks,
casts, summoners, items, buys, wards, recalls) and records a sha256 per state leaf every 600 ticks plus a
layout-independent game summary. `--out FILE` saves a run, `--compare FILE` checks a run against it (by leaf name,
or by leaf value when the state was renamed/restructured; summary differences for resized worlds). Run both sides
on the same machine and JAX version; on the login node one world needs about 10 GB and 45 min to compile.
