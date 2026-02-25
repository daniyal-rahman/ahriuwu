# Replay Movement Data Extraction

## Status: Working decoder built, projection calibration + automation pipeline next

**Last updated**: 2025-02-20
**Replay format**: ROFL2 (patch 16.3)
**Test replay**: NA1-5489605032 (31 min game, cached blocks at `/tmp/rofl_blocks_cache.pkl`)

---

## What Was Done

### 1. Replay packet decryption (COMPLETE)
Replay files (.rofl) contain zstd-compressed chunks with encrypted per-field game packets.
Each packet has a `packet_id` (netid) and `param` (entity ID), with payload bytes
encrypted using per-patch lookup tables + arithmetic transforms (ror, sub, xor, add).

Built a Unicorn Engine emulator that runs the actual League binary's deserializer functions
to decrypt packets. The deserializer does: skip (bitstream setup) -> bit_reader (3-bit tags
per field: 0=read from stream, 1=keep current, 2=default) -> field_reader (per-byte
deobfuscation) -> S-box re-encryption (anti-cheat). We capture the values AFTER field_reader
but BEFORE S-box to get the actual decoded data.

### 2. Movement packet identified (COMPLETE)
**netid=762** is the movement/pathfinding packet.

- Deserializer RVA: `0x1002930`
- Sub-object initializer RVA: `0xe03db0`
- Sub-object vtable: `0x19eb668` (slot[1] = `0xffbda0` = sub-deserializer)
- Main struct vtable: `0x19eb6c0`
- Requires malloc hook (realloc func at `0xe3d430` calls malloc `0x10f98f0`)

**Decoded structure** (from allocated 296-byte buffer):

| Offset | Type | Field |
|--------|------|-------|
| +0x008 | u32  | Entity ID |
| +0x01c | f32  | Current X (0-15000 map units) |
| +0x020 | f32  | Current Z (height) |
| +0x024 | f32  | Current Y (0-15000 map units) |
| +0x028 | u32  | Sequence number |
| +0x02c | f32  | Movement speed |
| +0x054 | f32  | Destination X |
| +0x058 | f32  | Destination Z |
| +0x05c | f32  | Destination Y |
| +0x10c | f32  | Game timestamp (seconds) |
| buf2   | str  | Champion name (e.g. "MissFortune") |

**Two packet sizes:**
- Large (100-147 bytes): Full movement with destination, speed, champion name. Requires malloc.
- Small (31-35 bytes): Position sync only (X, Y, Z written to sub-object at +0x24/+0x28/+0x2c). No malloc.

### 3. Data extraction (COMPLETE)
From test replay: **3680 movement updates, 0 errors**
- 1382 large packets (with destination + speed + champion name)
- 2298 small packets (position only)
- 10 heroes tracked across full game
- Champions found: Rengar, Lucian, Nami, Kled, Sylas, MissFortune, Janna

### 4. Also found: netid=294 position sync
~2.8/s per entity, fields: +0x18=X, +0x1c=Z(?), +0x20=Y. Simpler structure,
no allocation needed. Can supplement netid=762 for higher-rate position tracking.

---

## What Still Needs To Be Done

### Step 1: World-to-screen projection calibration
We have world-space coordinates (destination of clicks) but need screen-space pixel
coordinates for action labels.

**Decision**: Assume Garen = screen center (small camera smoothing offset is negligible).
This reduces the problem to finding one 2x2 projection matrix M:
```
click_pixel = screen_center + M @ (dest_world - garen_world)
```
M is constant for a given resolution + camera settings (locked camera, fixed zoom/angle).

**Precision needed**: Must distinguish individual minions in a wave (~50-100px apart).

**Calibration approach**: Use turret positions.
- Turrets have fixed, known world coordinates (see turret_world_coords below)
- Find frames where turrets are visible on screen in the replay video
- Manually click on 4-6 turrets to get their screen pixel positions
- With world offsets (turret_world - garen_world) and screen offsets (turret_pixel - screen_center),
  solve for M via least-squares
- Script: `scripts/calibrate_projection.py` (TODO)

**Previous CV attempts** (for reference, not using these):
- OCR "Garen" text detection: 99.7% detection rate but positional accuracy unvalidated
- Color health bar detection: 31.7% detection rate
- See `scripts/compare_detection_methods.py`

### Step 2: Replay sourcing — finding high-elo Garen replays
Need 1000s of hours of high-elo Garen gameplay.

**Approach**: Use the Riot Games API.
1. Query Challenger/Master/GM ladder endpoints for players with high Garen mastery
2. Pull their recent match IDs (filter champion=Garen)
3. Download .rofl files via the League client or replay download API
4. Replays expire each patch (~2 weeks), so set up a recurring download job

**Script**: `scripts/download_replays.py` (TODO) — queries Riot API, filters Garen games,
triggers replay downloads.

**Riot API key**: Need a development key (rate limited) or production key (apply to Riot).
Dev key: https://developer.riotgames.com/

### Step 3: Automated replay processing pipeline
Cannot manually watch each replay. Full automation required.

**League Replay API**: When a replay is playing, the League client exposes a local REST API
at `https://127.0.0.1:2999/replay/` with endpoints:
- `POST /replay/playback` — control speed, pause, seek to timestamp
- `POST /replay/render` — camera settings (lock to specific player)
- `POST /replay/recording` — start/stop built-in video recording (no OBS needed)

**Automated pipeline** (runs on Windows desktop unattended):
1. For each .rofl file:
   a. Launch replay via League client CLI
   b. Wait for game to load (poll replay API until responsive)
   c. `POST /replay/render` — lock camera to Garen player
   d. `POST /replay/playback` — set speed, seek to start
   e. `POST /replay/recording` — start recording
   f. Wait for game to end (poll playback position)
   g. Stop recording → video saved to disk
2. Decode movement data from .rofl file (our existing decoder, runs independently)
3. Sync video timestamps with movement timestamps
4. Apply projection matrix M to generate screen-space action labels

**Script**: `scripts/process_replays.py` (TODO) — orchestrates the full pipeline.

**Playback speed optimization**:
- Need 20 game-fps for training data
- Recording captures at render FPS, NOT display refresh rate (vsync must be OFF)
- If game renders at ~230fps: max speed = 230/20 = **11x**, but League caps at **8x**
- At 8x with 230fps: 230/8 ≈ 29 game-fps → comfortably above 20fps
- **A 30-min game takes ~3.75 minutes to process at 8x**
- Fallback if recording is display-capped at 60fps: max 3x speed, 10 min per replay
- Verify actual recording FPS with a test run

### Step 4: Speed optimization for decoder
Current decoder runs ~10 blocks/sec (Unicorn emulation overhead). For 3680 blocks
that's ~6 minutes per replay. Acceptable for now but could be optimized by:
- Caching the emulator state between calls
- Only decoding large packets (with destinations) if that's all we need
- Batching similar-size payloads

---

## Key Files

### In the repo
- `scripts/decode_replay_movement.py` — Main decoder script. Uses Unicorn Engine
  to emulate League's netid=762 deserializer. Run with `python scripts/decode_replay_movement.py`.
  Requires PE section dumps in `/tmp/pe_dump/`.
- `scripts/compare_detection_methods.py` — OCR vs color-based Garen detection comparison
- `src/ahriuwu/data/keylog_extractor.py` — Has `GoldTextDetector` with health bar + OCR detection

### On disk (not in repo, need to regenerate)
- `/tmp/pe_dump/text.bin` (26MB), `rdata.bin` (4.3MB), `data.bin` (618KB) — PE sections
  dumped from `League of Legends.exe` patch 16.3. These are required by the decoder.
  Regenerate by dumping sections from the exe (RVAs: text=0x1000, rdata=0x18e9000, data=0x1d0c000).
- `/tmp/rofl_blocks_cache.pkl` — 1.8M parsed blocks from replay NA1-5489605032
- `/tmp/netid_deser_map.pkl` — 470 netid-to-deserializer-RVA mappings
- `/tmp/movement_data.pkl` — Decoded movement data from test replay
- `/tmp/LeagueOfLegends.exe` — League binary copied from Windows (`scp windows:...`)
- `/tmp/ghidra_project/league_analysis` — Ghidra project with analyzed binary

### Key addresses (patch 16.3 specific, will change with patches)
```
PE image base:    0x140000000
skip:             0x1186e30
bit_reader:       0xeb67f0
malloc:           0x10f98f0
free:             0x10f9920
string_realloc:   0x1118a70
S-box table:      0x19e49d0
Mutexes:          0x1858c98, 0x1858d04, 0x1858fd4
netid=762 deser:  0x1002930
sub-obj init:     0xe03db0
sub-obj vtable:   0x19eb668
main vtable:      0x19eb6c0
netid=294 deser:  0xf98010
```

### Hero entity IDs (this replay)
Player entities: 0x400000ae through 0x400000b7 (10 heroes)

---

## Netids tested and ruled out
- **netid=1138**: Mostly zeros, not movement
- **netid=284**: Timer/counter packet (+0x38 = game timestamp, +0x3c = 25000 - timestamp)
- **netid=581**: Takes early exit in deserializer, minimal output
- **netid=294**: Position sync (X, Y, Z coordinates), NOT movement commands
- **netid=332, 459, 84, 33**: Small fixed payloads, single fields
- **netid=443**: Has entity ID + small float, not movement
- **netid=762**: **THIS IS THE MOVEMENT PACKET** (confirmed)

---

## How the decoder works (for future reference)

1. Map PE sections (.text, .rdata, .data) into Unicorn at correct RVAs
2. Patch `skip` function to `mov rax, 1; ret` (skip bitstream setup, return success)
3. Patch `free` and mutex functions to `ret` (no-ops)
4. Hook `malloc` to return memory from a separate allocation region
5. Initialize the sub-object at struct+0x10 by calling `0xe03db0` (sets vtable + S-box)
6. Set main vtable at struct+0x00
7. Set up payload in scratch memory, pass pointers via RCX/RDX/R8 (Windows x64 ABI)
8. Run deserializer at `0x1002930`
9. Capture writes to the allocated buffer region
10. Extract pre-S-box values: the LAST 4-byte write at each offset before S-box 1-byte writes

For large packets: data goes into malloc'd 296-byte buffer.
For small packets: data goes into sub-object fields at struct+0x10+offset.
