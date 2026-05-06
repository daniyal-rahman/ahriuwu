# Replay Data Pipeline

End-to-end flow from "I want N games of champion X at rank Y" → labeled
352×352 frames + per-frame action labels + click/cast events ready for ML
training.

## Files (the only ones you should need to edit)

```
scripts/
├── core/
│   ├── pipeline.py     # 2-pass scrape + record + post-process (Windows)
│   └── overlay.py      # Render labeled overlay video from a match dir (Windows or Mac)
├── scan_offsets.py     # Once-per-patch memory offset re-derivation (Windows)
├── perf_monitor.py     # Optional: 1Hz CPU/RAM/GPU sampler for batch runs (Windows)
└── offsets_<patch>.json # Emitted by scan_offsets, consumed by pipeline
```

Everything else under `scripts/` is exploration scratch from earlier sessions
and should not be edited; it can be archived later.

## End-to-end flow

```
┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│ U.GG / Riot │ → │ manifest    │ → │ download    │ → │ pipeline    │ → labels.json + frames + clicks.json
│ leaderboard │   │ JSON        │   │ .rofl files │   │ (2-pass)    │
└─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘
                                                              │
                                                              ↓
                                                       ┌─────────────┐
                                                       │ overlay.py  │ → overlay.mp4 (debugging only)
                                                       └─────────────┘
                                                              │
                                                              ↓
                                                       ┌─────────────┐
                                                       │ NFS sink    │ → /mnt/nfs/datasets/...
                                                       │ (sync.py)   │
                                                       └─────────────┘
```

---

## Step 1 — Find players (U.GG → Masters+ filter)

U.GG curates **per-champion** leaderboards which gives a higher hit rate of
target-champion games than Riot's general Masters+ list. For Garen NA:

> `https://u.gg/lol/champions/garen/leaderboards?region=na1`

**Manual filter step**: U.GG includes Diamond 1 and Diamond 2 players in the
top-100 list. Drop those — keep only **Master / Grandmaster / Challenger**.
From the snapshot taken 2026-05-06 there were ~70 Masters+ Garen mains; the
top 5 included explicit Riot IDs:

```
1   1v9#palco                Challenger  905 LP   75W 66L
2   Triton#DMCIA             Master      420 LP   252W 242L
3   hpg#o7 o7                Master      390 LP   107W 82L
4   AddictedToBacon#TTV      Master      240 LP   236W 219L
5   Palco Granko#split       Master       89 LP   443W 373L
```

For ranks 6+ U.GG hides the tag — you have to either click each profile to
read it, or cross-reference against the Riot League-V4 API output (see below).

**Riot League-V4 API approach (recommended for 300+ games):** instead of
scraping U.GG, just pull every Masters+ summoner directly:

```
GET /lol/league/v4/masterleagues/by-queue/RANKED_SOLO_5x5
GET /lol/league/v4/grandmasterleagues/by-queue/RANKED_SOLO_5x5
GET /lol/league/v4/challengerleagues/by-queue/RANKED_SOLO_5x5
```

Each entry has a `summonerId`; convert to `puuid` via
`/lol/summoner/v4/summoners/{summonerId}`. Lower hit rate for "Garen TOP"
(maybe ~1-2% of any Masters+ player's games) but covers the whole rank
exhaustively. For Garen-mains specifically, intersect this set with the
U.GG name list.

## Step 2 — Build the manifest

For each player puuid, pull recent ranked solo games and filter:

```python
# Pseudocode
for puuid in players_puuids:
    match_ids = GET /lol/match/v5/matches/by-puuid/{puuid}/ids?queue=420&count=100
    for mid in match_ids:
        m = GET /lol/match/v5/matches/{mid}
        if not m["info"]["gameVersion"].startswith(CURRENT_PATCH):
            continue
        for p in m["info"]["participants"]:
            if p["championName"] == "Garen" and p["teamPosition"] == "TOP":
                team = "blue" if p["teamId"] == 100 else "red"
                slot = sum(1 for q in m["info"]["participants"]
                           if q["teamId"] == p["teamId"]
                           and m["info"]["participants"].index(q)
                              < m["info"]["participants"].index(p))
                manifest_entries.append({
                    "match_id": mid,
                    "game_id": mid.split("_")[-1],
                    "champion": "Garen",
                    "garen_team": team,
                    "garen_slot": slot,
                    "duration": 1900,
                    "version": m["info"]["gameVersion"],
                })
```

Reference: `scripts/find_one_garen.py` finds a single game; expand to a
batch builder that walks N seed players × M recent matches and emits a
`{"matches": [...]}` JSON in the format above.

**Manifest format consumed by `pipeline.py`:**

```json
{
  "name": "garen_top_masters_16_9_772",
  "matches": [
    {
      "match_id": "NA1_5554195441",
      "game_id":  "5554195441",
      "champion": "Garen",
      "garen_team": "blue",
      "garen_slot": 0,
      "duration": 1900,
      "version": "16.9.772.8292"
    },
    ...
  ]
}
```

`garen_team` / `garen_slot` are legacy keys retained for back-compat —
`team` / `slot` are also accepted. `champion` is per-match (a single batch
can mix champs).

## Step 3 — Download .rofl files

LCU exposes a per-game download endpoint:

```
POST /lol-replays/v1/rofls/{gameId}/download
```

This needs the LeagueClient running and authed. After the call, the file
appears in:

```
C:\Users\<user>\Documents\League of Legends\Replays\NA1-<gameId>.rofl
```

Reference impl: `scripts/download_one_replay.py`. For batch, loop over
`manifest["matches"]`, POST each download, wait a few seconds between calls
to avoid LCU rate-limit, and verify each .rofl exists before continuing.

**Patch verification**: read the first 64 bytes of each .rofl — the patch
number ("16.9.772.8292") is in plaintext after the "RIOT" magic. If the
`version` in your manifest doesn't match, the replay won't play on the
current client and pipeline will time out at game-load. Drop those entries
before kicking off pipeline.

## Step 4 — Re-derive offsets (once per patch)

```bash
python scripts/scan_offsets.py --champion Garen \
    --anchors scripts/offsets_<prev_patch>.json \
    --output  scripts/offsets_<new_patch>.json \
    --patch   16.9.772
```

Open a working replay first (any 16.9.x .rofl) and let it reach gt~100s.
The scanner does:

| Phase | Derives                                              |
| ----- | ---------------------------------------------------- |
| 1a    | `hero_array` RVA + layout (`deref` or `inline`)      |
| 1b    | `game_time` RVA (multi-time verified — rejects stale snapshots) |
| 2     | `champion_name`, `position`, `level`, `hp`, `gold`, `vision_score`, `active_spell` (best-effort) |
| 3     | `spellbook`, `slot_array`, `slot_spell_info`, `spell_name_ptr` |
| 3.5   | `click_vtable_rva`, `click_owner_offset` (heap triple-mirror scan) |
| 3.6   | `active_spell`, `spell_info`, `cast_target` (live-cast diff probe — sole source on 16.9+) |
| 4     | `attack_target_pos` (cam-lock-free probe: filter to AAs of named champion only) |

Output JSON also includes `_patch_mod_size`, `_scanned_at`, and per-field
`_offset_versions` timestamps so the pipeline auto-loads the newest
matching JSON by mtime.

Pipeline.py loud-fails (raises `KeyError`) on any missing offset at runtime.
If a phase doesn't derive a field, the JSON's `_missing` list will say so —
re-scan with a different replay or champion before running the batch.

## Step 5 — Run the pipeline

```bash
# Single game
python scripts/core/pipeline.py \
    --game-id 5554195441 --match-id NA1_5554195441 \
    --team blue --slot 0 --champion Garen \
    --duration 1900 --force

# Batch (overlapped post-process)
python scripts/core/pipeline.py --manifest <manifest.json> --batch
```

**Important Windows-side prerequisites:**

- League client open + logged in
- Vanguard disabled (else ReadProcessMemory fails)
- `game.cfg`: `Width=1280`, `Height=720`, `WindowMode=1`, `EnableReplayApi=1`
- Run the .bat via `schtasks /Run /TN <task> /IT` if you started it from
  SSH, so the cam-lock pynput keys reach session 1

**Output per game** (under `$REPLAY_OUTPUT`, defaults to `E:\replay_data\`):

```
{match_id}/
├── frames/000000.png … (~36000 frames per 30-min game @ 20fps)
├── labels.json     # per-frame label (champion screen pos, stats, action, etc)
├── clicks.json     # click + cast events with game_t and world coords
├── raw_mem.json    # raw memory scrape (overlay reads this)
└── raw_cam.json    # raw cam scrape (overlay reads this)
```

Plus `pipeline.jsonl` at the root with per-phase timing for batch analysis.

**Throughput** (single Garen game on the 16-core desktop, May 2026):
- pass1 16.2 min wall, 17% CPU avg
- pass2 9.4 min wall, **99% CPU** (League's PNG encoder pegs all cores)
- post 9.3 min wall, 24% CPU (2 workers on 2 cores by design — overlaps with
  next game's pass1+pass2 in batch mode, hidden in steady-state)
- single-game: ~35 min wall for 30:31 game = 1.14× real-time
- batch (overlap): ~1.0× real-time once steady-state hits

## Step 6 — Sanity-check with overlay.py

```bash
python scripts/core/overlay.py --match-dir <path>/{match_id}
```

Renders `overlay.mp4` next to the input. HUD top-left, action log top-right,
clicks/casts as colored markers projected from world coords using the per-
frame cam from `raw_cam.json`. Frames outside cam coverage (typically the
first ~7s) get a red "NO CAM SAMPLE" warning.

This is **debugging only** — the training pipeline reads `frames/` +
`labels.json` directly, not the overlay video.

## Step 7 — NFS sink (data flow off Windows)

The pipeline writes locally to Windows. For training the data has to land
on Linux either via:

**Option A — direct push from Windows during pipeline run:**

```
Windows pipeline → D:\replay_data\<match_id>\         (local, ephemeral)
                 → scp/rsync to danilogin:/mnt/nfs/datasets/lol_replays_<patch>/<match_id>/
                 → (delete local)
```

A separate `sync_to_nfs.py` watcher on Windows polls the output dir, ships
each `<match_id>/` to NFS once `labels.json` exists (= post-process done),
verifies the transfer, then optionally deletes the Windows copy.

**Prerequisite:** Windows needs `ssh danilogin` set up with key-based auth
and the host key accepted. Currently this is **not configured** — first
run requires interactive `ssh windows`, then `ssh danilogin` and accept
the host key. Without that the watcher errors with "Host key verification
failed."

**Option B — reboot and copy locally:**

After the pipeline finishes, reboot the desktop to Linux and copy:

```bash
# desktop on Linux
mount /dev/<windows-ntfs-partition> /mnt/win
cp -r /mnt/win/replay_data/* /mnt/storage/data/lol_replays_<patch>/
```

`/mnt/storage/data/` is local NVMe on the desktop and is the canonical
training-data location per `~/CLAUDE.md` (NFS is shared / slower; keep
hot training data on local).

For a 300-game run: ~300 GB output → ~10-15 min over LAN to NFS, or
seconds to local NVMe via reboot. Pick by latency tolerance.

## Step 8 — Resource monitoring (optional)

Run `scripts/perf_monitor.py` alongside the pipeline to dump 1Hz samples of
per-core CPU, RAM, League/Python RSS, and GPU util to a CSV:

```bash
start /B "" python C:\path\perf_monitor.py --out C:\tmp\perf.csv --interval 1.0
python pipeline.py ...
taskkill /F /IM python.exe
```

Useful for:
- Spotting memory leaks across the batch (`league_rss_mb` trending up)
- Pinpointing where pass2 fps dipped (was it League pegged or a python
  worker stealing cores?)
- Comparing per-game timing distributions over a long run

The CSV columns are: `t_wall`, `cpu_total`, `cpu_c0..cN`, `mem_used_mb`,
`league_pid`, `league_cpu`, `league_rss_mb`, `py_pids`, `py_cpu_sum`,
`py_rss_mb_sum`, `gpu_util`, `gpu_mem_used_mb`, `gpu_mem_total_mb`.

## Common failures and recovery

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Pipeline hangs at "Launching replay…" | LeagueClient not running | Open it; the LCU POST has nothing to send to |
| `ABORT: gt at 0x… never reached a sane >0.5s value in 45s` | Wrong patch (offset shift) OR League stuck loading | Re-run `scan_offsets.py` to confirm offsets, or check if game.cfg resolution is bogus (4K kills perf) |
| `ABORT: champion 'X' not found` | Manifest `champion` doesn't match the .rofl's actual game | Verify match metadata; many manifest builders mismatch on remakes |
| `n_unlabeled > 5%` | pass1 mem coverage gap (cam-lock failed?) | Look at the per-game log — should see `Camera locked (key=N)` near the top |
| `no_cam_coverage` >> 0 in overlay | rec_start was outside pass1's mem range | Already mitigated by mem-coverage clamp in pass2; if you see this, check pass1 actually ran for the recorded duration |
| Per-game labels.json has clicks but no casts | Phase A click-vtable scan didn't find candidates (rare) | `[click] owner-filter: 0/N` in log → re-scan offsets, click_vtable_rva probably shifted |
