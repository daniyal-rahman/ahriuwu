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


`fetch_modern_map.py` is a TOOL for the modern world port (MODERN-010). It
extracts requested Map11 WAD members from an explicitly pinned Riot manifest;
it does not install or run League. Use an isolated Python3.11 environment with
`riotmanifest==2.10.2`, `league-tools==1.2.1`, `zstd==1.5.7.2`. Run through
`login_capped.sh`. Both extraction and normalization require new output dirs.

```sh
ops/login_capped.sh 4G 2 /path/to/asset-venv/bin/python ops/fetch_modern_map.py --manifest https://lol.dyn.riotcdn.net/channels/public/releases/4D2A50D5EDAB724A.manifest --build 16.19.8230722 --out /path/to/new-extraction --asset assets/maps/navgrid/map11/aipath_srx_2.aimesh_ngrid --asset assets/maps/deprecated/map11/cfg/objectcfg_srx.cfg --asset data/maps/shipping/map11/map11.bin
ops/login_capped.sh 4G 2 /path/to/numpy-python -m lanerl_jax.data.modern_map import --ngrid /path/to/new-extraction/assets/assets/maps/navgrid/map11/aipath_srx_2.aimesh_ngrid --out /path/to/new-grid --patch 26.19 --source https://lol.dyn.riotcdn.net/channels/public/releases/4D2A50D5EDAB724A.manifest --retrieved-at 2026-09-30
```

The date above reproduces the existing artifact's provenance; a new source
retrieval should use its actual date and receive a reviewed new manifest pin.
Compare the extraction hashes to `lanerl_jax/data/modern/26.19/map11.json`.
`load_patch_map(path, patch="26.19")` requires that profile's exact manifest
hash and checks its arrays; `load_artifact` without an external manifest pin
only checks self-reported identity and accidental array corruption. Neither
loader resolves latest or substitutes the legacy map. Shared validated arrays:
`/mnt/nfs/datasets/league/26.19/map11-base/`. Raw extraction receipts:
`/mnt/nfs/shared/modern-world-map-research/live-16.19.8230722/`.

`modern_items_bench.py` is a TOOL for the modern item system (MODERN-013). It
times one full `modern_item_effects.runtime.item_tick` (two champions with six
items each, 66 units) single and vmapped; run it capped:
`ops/login_capped.sh 8G 2 .venv-jax/bin/python -m ops.modern_items_bench 64`.
The item table is rebuilt from the cached 16.19 client bins with
`python -m lanerl_jax.data.build_modern_items` (sources and sha256s are
recorded in `items_client.json`).

`replay_oracle_extract.py` + `replay_oracle_extract.sbatch` are a TOOL
(MODERN-015) that extracts raw client-memory observations (gold, deaths and
respawns, fountain stretches, level-ups, max-HP changes) from the 147-game
16.9 replay corpus `/mnt/nfs/datasets/lol_replays_16_9_772/`, independent of
the simulator. Submit with `sbatch ops/replay_oracle_extract.sbatch` (CPU
partition, job array); per-game output, the index and job logs (`logs/`) go to
`/mnt/nfs/shared/replay-oracle-16.9/`, the compact summary to
`lanerl_jax/data/modern/oracle/replay_16_9_observations.json.gz`, which
`lanerl_jax/sim/tests/test_modern_economy_oracle.py` reads. The 16.9 champion
HP records it is paired with (`oracle/champion_hp_16_9.json`) were fetched from
CommunityDragon 16.9 into `/mnt/nfs/shared/replay-oracle-16.9/champions-16.9/`.

`riot_stats_oracle.py` is a TOOL (MODERN-016). `extract` turns Riot match-v5
match + timeline JSON (fetched to `/mnt/nfs/shared/riot-match-v5-16.9/` by
`fetch_matches.py` there; API key from `RIOT_API_KEY`, never written to disk)
into the anonymised `lanerl_jax/data/modern/oracle/riot_16_9_frames.json.gz`;
`predict` (used by `tests/test_modern_stats_riot_oracle.py`) recomputes Riot's
`championStats` through the modern stat pipeline, shards, rune and item stat
hooks with 16.9 records (`oracle/client_16_9_stats.json`).
