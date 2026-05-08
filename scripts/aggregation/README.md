# Replay Data Aggregation

End-to-end flow from "I want N games of champion X at rank Y" → labeled
352×352 frames + per-frame action labels + click/cast events ready for ML
training.

## Files in this folder (the only ones you should need to edit)

```
scripts/aggregation/
├── build_manifest.py    # Riot API → manifest JSON of (champion, role, patch) matches
├── pipeline.py          # 2-pass scrape + record + post-process (Windows only)
├── overlay.py           # Render labeled overlay video from a match dir (sanity-check)
├── scan_offsets.py      # Once-per-patch memory offset re-derivation (Windows only)
├── perf_monitor.py      # Optional: 1Hz CPU/RAM/GPU sampler for batch runs
├── sync_to_nfs.py       # Optional: Windows watcher → scp finished games to NFS
├── offsets_16_9_772.json  # Current patch offsets (consumed by pipeline)
├── enable_vanguard.bat  # Enable Vanguard service (gameplay)
├── enable_vg_reboot.bat # Enable + reboot
└── disable_vg_reboot.bat # Disable + reboot (needed for memory scanning)
```

Curated seed lists (Riot IDs of high-MMR mains for a given champion) live
under `data/seeds/`, e.g. `garen_mains_masters_op_gg_2026_05_06.txt`.

Everything in `scripts/_archive/` is exploration scratch from earlier
sessions — kept for memory's sake, not part of the active flow.

## End-to-end flow

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Curated seed │→ │ build_       │→ │ pipeline.py  │→ │ NFS / local  │
│ Riot IDs     │  │ manifest.py  │  │ (LCU trigger │  │ training     │
│ (op.gg/u.gg) │  │ → manifest   │  │  + 2-pass    │  │ store        │
└──────────────┘  │   JSON       │  │  + post)     │  └──────────────┘
                  └──────────────┘  └──────────────┘
                                          │
                                          ↓
                                    ┌──────────────┐
                                    │ overlay.py   │ → overlay.mp4 (debug only)
                                    └──────────────┘
```

---

## Step 1 — Build the manifest

Two sources for the player list. Use the curated seed file (preferred —
high signal, ~30 matches/player kept) or fall back to walking the whole
Masters+ leaderboard.

**Curated seed file (preferred):**

Snapshot the op.gg per-champion leaderboard for your patch (Master / GM /
Challenger only — drop Diamond), one Riot ID per line in
`data/seeds/<champion>_mains_<source>_<date>.txt`:

```
# Curated Garen mains (Masters+) from op.gg leaderboard, snapshotted 2026-05-06.
Triton#DMCIA
AddictedToBacon#TTV
hpg#o7 o7
...
```

Then:

```bash
python scripts/aggregation/build_manifest.py \
    --champion Garen --role TOP --patch 16.9 \
    --region na1 \
    --seed-riot-ids data/seeds/garen_mains_masters_op_gg_2026_05_06.txt \
    --out data/manifests/garen_top_curated_16_9_772.json \
    --max-games 300 \
    --api-key RGAPI-...
```

Each Riot ID is resolved to a puuid via `account-v1/by-riot-id`, then their
last 50 ranked-solo matches are walked and filtered to (champion, role,
patch). Default filters: `--min-game-duration 600` (drops remakes /
early-ff), `--duration-cap 1900` (per-match pipeline ceiling).

**Leaderboard fallback:**

If you don't have a curated list, omit `--seed-riot-ids` and the script
walks every Masters+ player via League-V4 (~50 calls/min on dev keys due
to rate limits — for 300 games of a niche champ this can take an hour).

**Manifest format:**

```json
{
  "name": "Garen_TOP_curated_16_9_772",
  "champion": "Garen", "role": "TOP", "patch": "16.9",
  "n_players_walked": 21,
  "matches": [
    {
      "match_id": "NA1_5554435338",
      "game_id":  "5554435338",
      "platform": "na1",
      "champion": "Garen",
      "garen_team": "blue",
      "garen_slot": 0,
      "duration": 1837,
      "version": "16.9.772.8292",
      "summoner_name": "Triton#DMCIA",
      "kda": "6/3/5",
      "garen_win": true,
      "game_duration": 1837
    }
  ]
}
```

The script writes incrementally — partial state is on disk after each
contributing player, so an interrupted run is a no-op restart.

## Step 2 — Re-derive offsets (only when patch changes)

```bash
python scripts/aggregation/scan_offsets.py --champion Garen \
    --anchors scripts/aggregation/offsets_<prev_patch>.json \
    --output  scripts/aggregation/offsets_<new_patch>.json \
    --patch   16.9.772
```

Open a working replay first (any current-patch .rofl) and let it reach
gt~100s. The scanner derives all 21+ offsets from a live replay (PE static
analysis where possible, runtime heap scans where not). Output JSON
includes `_patch_mod_size`, `_scanned_at`, and per-field `_offset_versions`
timestamps so `pipeline.py` auto-loads the newest matching JSON by mtime.

Pipeline loud-fails on any missing offset at runtime — the JSON's
`_missing` list will tell you what didn't derive. Re-scan with a different
replay/champion if a phase comes up empty.

## Step 3 — Run the pipeline

```bash
# Single game (debug)
python scripts/aggregation/pipeline.py \
    --game-id 5554195441 --match-id NA1_5554195441 \
    --team blue --slot 0 --champion Garen \
    --duration 1900 --force

# Batch (production)
python scripts/aggregation/pipeline.py \
    --manifest data/manifests/garen_top_curated_16_9_772.json \
    --batch --champion Garen \
    --prefetch-window 10 --rofl-cleanup-behind 2
```

The pipeline triggers .rofl downloads via LCU itself — no separate download
step. `--prefetch-window N` keeps N .rofls queued ahead of the cursor;
`--rofl-cleanup-behind K` deletes the K-back .rofl after each game so disk
doesn't blow up. Defaults: 10 / 2.

**Resume / pause:** Without `--force`, completed games (those with
`labels.json`) get skipped on re-run. Just kill and re-fire to resume.
Incomplete out_dirs (no labels.json) are auto-cleared and retried. The
batch loop stops after 3 consecutive failures.

**Windows-side prerequisites:**

- League client open + logged in
- Vanguard **disabled** (else ReadProcessMemory fails) — see
  `disable_vg_reboot.bat`
- `game.cfg`: `Width=1280`, `Height=720`, `WindowMode=1`,
  `EnableReplayApi=1` (4K kills perf — pipeline checks this)
- Run via `schtasks /Run /TN <task> /IT` if launching from SSH (cam-lock
  pynput keys must reach session 1)

**Output per game** (under `$REPLAY_OUTPUT`):

```
{match_id}/
├── frames/000000.png … (~36000 frames per 30-min game @ 20fps, 352×352)
├── labels.json     # per-frame labels (champion screen pos, stats, action)
├── clicks.json     # click + cast events with game_t and world coords
├── raw_mem.json    # raw memory scrape (overlay reads this)
└── raw_cam.json    # raw cam scrape (overlay reads this)
```

Plus `pipeline.jsonl` at the root for per-phase timing.

**Throughput** (16-core desktop, May 2026):

- pass1 ≈ duration/2 wall (replay 2x, mem+cam+click scrape)
- pass2 ≈ duration wall (1x record, 99% CPU pegged on PNG encoder)
- post  ≈ duration/3 wall (overlapped with next game's pass1+pass2)
- batch steady-state ≈ 1.0× real-time → ~30min per 30min game

## Step 4 — Sanity-check with overlay.py

```bash
python scripts/aggregation/overlay.py --match-dir <output>/{match_id}
```

Renders `overlay.mp4` next to the input — champion world→screen projection,
HUD, action log, click/cast markers. **Debug only** — training reads
`frames/` + `labels.json` directly.

## Step 5 — Get data off Windows (optional)

If your training rig **is** the recording machine, you don't need this step at
all — the pipeline output sits in `$REPLAY_OUTPUT` and you point your training
loader at it. Skip ahead.

If recording happens on a separate Windows box, two options:

**Option A — push during pipeline run (`sync_to_nfs.py`):**

```bash
python scripts/aggregation/sync_to_nfs.py \
    --src C:\tmp\replay_data \
    --dataset lol_replays_16_9_772 \
    --remote <your-host-alias>:/path/to/nfs/datasets \
    --poll 30 --delete-local
```

Watches the output dir, tar-streams each `<match_id>/` to your remote once
`labels.json` exists, verifies file count, optionally deletes the Windows copy.
Prereq: `ssh <your-host-alias>` works from Windows without prompts (key auth
+ host key accepted). The script is fully decoupled from `pipeline.py` — if
you don't run it, pipeline.py just keeps the output local.

**Option B — reboot to Linux and copy from NTFS partition.**

For 300 games (~300 GB): ~10-15 min over LAN, or seconds to local NVMe via
reboot. Pick by latency tolerance.

## Step 6 — Health monitoring (optional)

`monitor.py` runs anywhere with ssh access to the recording host (and
optionally the data sink) and writes a single tail-able log of pipeline
state, sync state, disk free, backlog GB, NFS done count, plus distinct
alerts for `PIPELINE STALLED` (log mtime frozen), `LEAGUE MISSING`,
`BACKLOG GROWING`, `NO PROGRESS` (NFS count not advancing), etc.

```bash
python scripts/aggregation/monitor.py \
    --recording-host <your-windows-alias> \
    --nfs-host <your-nfs-alias> \
    --nfs-dataset /path/to/nfs/datasets/lol_replays_16_9_772 \
    --out ~/replay_monitor.log
tail -f ~/replay_monitor.log
```

Drop `--nfs-host`/`--nfs-dataset` if you don't have an off-host data sink.

`perf_monitor.py` is a separate tool that dumps 1Hz per-core CPU, RAM,
League/Python RSS, GPU util to a CSV — useful for spotting memory leaks
across long batches and isolating slow-recording causes.

## Common failures

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Pipeline hangs at "Launching replay…" | LeagueClient not running | Open it; LCU POST has nothing to send to |
| `ABORT: gt … never reached >0.5s in 45s` | Wrong patch (offset shift) OR wrong game.cfg resolution | Re-run `scan_offsets.py`; check `game.cfg` is 720p |
| `ABORT: champion 'X' not found` | Manifest `champion` doesn't match the .rofl's actual game | Check match metadata; common on remakes |
| `n_unlabeled > 5%` | pass1 mem coverage gap (cam-lock failed?) | Check per-game log for `Camera locked (key=N)` |
| `[click] owner-filter: 0/N` in log | click_vtable_rva probably shifted | Re-run `scan_offsets.py` |
| `*** skip <id>: .rofl never appeared ***` | LCU stale-state (already-deleted file metadata) | Restart LeagueClient; or accept skip |
