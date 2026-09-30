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
