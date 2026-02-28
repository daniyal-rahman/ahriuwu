# Replay Movement Data Extraction

## Status: 16.4 decoder WORKING (pid=437 confirmed as movement packet)

**Last updated**: 2026-02-25
**Replay format**: ROFL2 (patch 16.3 and 16.4)
**Test replays**:
- NA1-5489605032 (patch 16.3, 31 min game)
- NA1-5496350713 (patch 16.4, 15 min game, Garen)

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

### 5. Patch 16.4 reverse engineering (COMPLETE)

Movement netid changed from 762 to **437** between patches (not 872 — that's emote/state data).
Block format also changed to 9-byte headers.

#### ROFL2 block format (both patches)

Both 16.3 and 16.4 use 9-byte block headers:
```
MARKER(1) + CHANNEL(1) + SIZE(1) + PID(2) + PARAM(4) + PAYLOAD(SIZE)
```
- Marker bytes: `0x91`, `0xf1`, `0xb1`, `0x31`, `0x11`
- CHANNEL (byte2): varies — `0x00`, `0x21`, `0x22`, `0x43`, etc.
- SIZE is 1 byte (max 255)
- Some frames may use 15-byte headers too (needs investigation for full coverage)

#### netid=437 (0x01B5) — CONFIRMED movement packet

**Identification method**: PID analysis of all hero-entity blocks. pid=437 has:
- Size profile matching 16.3's pid=762: small=31-34B (position sync), large=127-136B (full movement)
- Blocks distributed across 7 entities (not all 10 — scanner may miss some)
- 678 raw pattern hits, 633 after validation

**Constructor** (`0xe2c826`):
```
mov word ptr [rcx+8], 0x1B5        ; netid = 437
lea rax, [rip+...] → 0x193c440    ; sub vtable (overridden)
mov [rcx], rax
lea rax, [rip+...] → 0x1a2afe8    ; final vtable
mov [rcx], rax
mov qword ptr [rcx+0x10], 0       ; zero struct+0x10
add rcx, 0x18
call 0xe2a490                      ; init sub-object at struct+0x18
```

**Sub-object init** (`0xe2a490`):
- Sets `struct+0x18 = vtable 0x1a2af90`
- Zeroes and S-box-encrypts fields at struct+0x20..0x2b (3 floats)
- Initializes more fields at struct+0x2c, 0x30, 0x34, 0x38

**Top-level deserializer** (`0x1033400`):
1. Calls SKIP (patched to return 1)
2. Read bits from payload via bit_reader (`0xee0270`)
3. First branch:
   - bit==0: Loop through sub-objects at struct+0x14, struct+0x10, calling `0x1008a90` for each
     (reads 4 bytes, applies per-byte deobfuscation: ror, xor, bit-shuffle, S-box)
   - bit==1: Calls memset (`0x18fb880`) for init, then falls through
4. Second branch:
   - bit==0: Write default float values to struct+0x20..0x34
   - bit==1: Call vtable[1] from struct+0x18 → `0x102c9e0` (sub-object deserializer)

**Sub-object deserializer** (`0x102c9e0`):
- Reads 3 bits from stream (tag per field, same pattern as 16.3)
- Writes 3 floats to sub-object at struct+0x20, +0x24, +0x28
- S-box tables at RVA `0x1a2fb40` and `0x1a2fa40`
- Calls `0x10d6900` for S-box re-encryption

**Decoded structure** (16.4):

| Location | Offset | Type | Field |
|----------|--------|------|-------|
| HEAP | +0x10 | f32 | Current X (0-15000) |
| HEAP | +0x14 | f32 | Current Y (0-15000) |
| HEAP | +0x20 | f32 | Destination X |
| HEAP | +0x24 | f32 | Destination Z (height, ~50) |
| HEAP | +0x28 | f32 | Destination Y |
| ALLOC[0] | +0x00c | f32 | Current X (copy) |
| ALLOC[0] | +0x010 | f32 | Current Z (height) |
| ALLOC[0] | +0x014 | f32 | Current Y (copy) |
| ALLOC[0] | +0x054 | f32 | Destination X (copy) |
| ALLOC[0] | +0x058 | f32 | Destination Z (copy) |
| ALLOC[0] | +0x05c | f32 | Destination Y (copy) |
| ALLOC[0] | +0x060 | f32 | Movement speed |
| ALLOC[0] | +0x094 | u32 | Entity ID |
| ALLOC[0] | +0x098 | f32 | Game timestamp (seconds) |
| ALLOC[0] | +0x0f0 | u32 | Sequence number |
| ALLOC[2] | str | str | Champion name (e.g. "MasterYi") |

**Two packet sizes** (same as 16.3):
- Large (127-136 bytes): Full movement with destination, speed, champion name. 3 mallocs.
- Small (31-34 bytes): Position sync only (X, Y in HEAP). No malloc.

**Verified output** from 16.4 test replay (NA1-5496350713, 15min Garen game):
- 633 blocks decoded, 0 errors
- Coordinates in valid map range (X,Y = 800-14000)
- Champion names extracted: Garen, MasterYi, Zoe

#### netid=872 — NOT movement (emote/state data)

pid=872 was initially suspected as movement but produces strings like "JOKE", "TAUNT".
Entity distribution is 81% Garen (2095/2582) — too skewed for movement.
Constructor at `0xe4e2d7`, deserializer at `0xfb8f40`.

#### Key addresses — patch 16.4

```
PE image base:      0x140000000
TEXT_RVA:           0x1000
RDATA_RVA:          0x1927000
DATA_RVA:           0x1d4e000

# Shared infrastructure
skip:               0x11bcec0       (patched to mov rax,1; ret)
bit_reader:         0xee0270
malloc:             0x112cb00
free:               0x112cb30

# netid=437 movement packet (CONFIRMED)
constructor:        0xe2c826
main_vtable:        0x1a2afe8
deserializer:       0x1033400       (main vtable slot[1])
sub_init:           0xe2a490        (init sub-object at struct+0x18)
sub_vtable:         0x1a2af90       (sub-object vtable, set by init)
sub_deserializer:   0x102c9e0       (sub-object vtable slot[1])
sub_field_deser:    0x1008a90       (per-field byte reader)
sbox_reencrypt:     0x10d6900       (S-box re-encryption)
sbox_table_1:       0x1a2fb40
sbox_table_2:       0x1a2fa40

# netid=872 emote packet (NOT movement)
constructor_872:    0xe4e2d7
main_vtable_872:    0x1a28918
deserializer_872:   0xfb8f40
```

#### Comparison with 16.3

| Item | 16.3 (netid=762) | 16.4 (netid=437) |
|------|-------------------|-------------------|
| Movement netid | 762 (0x02FA) | 437 (0x01B5) |
| Deserializer | 0x1002930 | 0x1033400 |
| Main vtable | 0x19eb6c0 | 0x1a2afe8 |
| Sub-obj init | 0xe03db0 (struct+0x10) | 0xe2a490 (struct+0x18) |
| Sub vtable | 0x19eb668 | 0x1a2af90 |
| Malloc | 0x10f98f0 | 0x112cb00 |
| Free | 0x10f9920 | 0x112cb30 |
| Skip | 0x1186e30 | 0x11bcec0 |
| RDATA RVA | 0x18e9000 | 0x1927000 |
| DATA RVA | 0x1d0c000 | 0x1d4e000 |
| Alloc buf size | 296 bytes | 304 bytes |
| Entity ID offset | alloc+0x008 | alloc+0x094 |
| Game time offset | alloc+0x10c | alloc+0x098 |
| Champion name | alloc[1] | alloc[2] |

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

### Key addresses (patch-specific, will change with patches)

**Patch 16.3** (netid=762):
```
PE image base:    0x140000000
TEXT_RVA:         0x1000
RDATA_RVA:        0x18e9000
DATA_RVA:         0x1d0c000
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

**Patch 16.4** (netid=872):
```
PE image base:    0x140000000
TEXT_RVA:         0x1000
RDATA_RVA:        0x1927000
DATA_RVA:         0x1d4e000
skip:             0x11bcec0
bit_reader:       0xee0270
malloc:           0x112cb00
free:             0x112cb30
constructor:      0xe4e2d7
main_vtable:      0x1a28918
deserializer:     0xfb8f40
sub_deserializer: 0xfb9070
sub_vtable:       0x193c440
sub_obj_vtable:   0x193ca50
allocator:        0xffcaa0
cleanup:          0x2042f0
intermediate_vt:  0x1a262c8
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
