# ROFL HP decoder — v1 (negative result)

**Patch:** 16.9.772
**Game tested:** `NA1_5552884026` (Garen / red top, 1825 s, 45 544 raw_mem ticks)
**Status:** No working HP decoder. Labeled-pairs is not viable at this patch with current tooling. Recommendation: stay on the live-replay-client + raw_mem snapshot pipeline.

## TL;DR

We pulled 10 paired games (rofl on Windows ↔ raw_mem on NFS) and tried to learn an HP decoder directly from labeled pairs. **It does not work.** Across every plausible (PID, entity, byte-offset, dtype) combination on 11 candidate PIDs and a final MLP-bytes-→-HP probe, the best held-out R² we could produce against any of the ten heroes' `hp_current` was **0.006**. That is indistinguishable from chance.

This is consistent with the encryption description in maknee's [League data-scraping blog post](https://maknee.github.io/blog/2025/League-Data-Scraping/): each PID applies a unique 255-byte lookup table combined with bit rotations and arithmetic (multiply by 2050 / 32 800, XOR with `0xF2`, `0xE6`, …), and the tables / constants change every patch. The blog explicitly notes HP isn't transmitted as a state snapshot — it's reconstructed from `TakeDamage` events whose damage values are encrypted. Their working approach is to hook the live game and read CPU registers; the "alternative" they cite (memory snapshots from a running replay client) is exactly what this project already does in `scripts/aggregation/pipeline.py`.

## What was actually built

All under `scripts/rofl_decode/`:

- `parse_rofl.py` — minimal block scanner. Walks zstd magic, decompresses 93 chunks, scans each chunk for blocks matching `{0x91, 0xf1, 0xb1, 0x31, 0x11}` markers. Produces `Block(frame_idx, marker, channel, pid, param, payload)` records. **227 311 blocks across 93 frames** for the reference game. Reusable.
- `find_hero_pids.py` — filters PIDs whose `param` is dominated by hero-shaped entity ids (`0x400000ae`–`0x400000b7` in this game) with small payloads. Top hero-keyed PIDs: 132, 89, 758, 104, 452, 828.
- `correlate_v2.py` / `correlate_v3.py` — Pearson r and pattern-consistency scans across (PID, entity, sub-tag, offset, dtype, hero) for every stat in raw_mem (hp, hp_max, level, gold, gold_total). Within-frame interpolation for time alignment.
- `scan_f32_in_range.py` — looks for any (PID, offset) where the f32 reading lands in `[30, 8000]` for ≥85 % of blocks (a "looks like HP" filter without ground truth).
- `mlp_probe.py` — last-ditch test. One-hidden-layer MLP (sklearn) trained per (PID, entity, hero) to predict HP from raw payload bytes. 80/20 split, ~1k labeled pairs per fit.

## What we found

1. **`param` for hero entities is contiguous: `0x400000ae`–`0x400000b7`** (10 ids → 10 heroes). Mapping to champion names is unknown — it doesn't appear in `labels.json`, isn't a slot index, and discriminator bytes inside payloads suggest it's a runtime-allocated handle.

2. **PID 132 (5 242 hero blocks, modal payload size 9)** has the cleanest hero-keyed structure:
   ```
   byte 0  — high-entropy, ~32 unique per entity → checksum / MAC
   byte 1  — ∈ {0x08, 0x09, 0x0b}                → sub-tag (~3 message subtypes)
   bytes 2-6 — constant per entity              → entity discriminator
   byte 7  — ~125 unique values                  → encrypted data
   byte 8  — ~10 unique values                   → encrypted flag / high byte
   ```
   This is the right shape for an encrypted stat update, but no Pearson r against any of the ten heroes' HP exceeded 0.4 once you strip the spurious matches against monotone counters (gold_total, hp_max).

3. **PID 985 (1 252 blocks, payload 123–135 B, 14 distinct params including all 10 hero entities)** has an f32 at offset 10 that lives in `[600, 6100]` — i.e. *plausible HP-or-damage range* across every block. This is almost certainly the **TakeDamage** packet referenced in maknee's blog. Its f32 doesn't correlate with any hero's `hp_current` though, which fits the blog's finding: the field is the encrypted *damage delta*, and the encryption is non-linear so the float you read is not the float that was sent.

4. **No PID emits plaintext level (1–18, monotone) at any byte offset.** So whatever the existing emulator's "level extraction" used to do under 16.7, it relied on running the deserializer in Unicorn and reading post-decoder struct slots — *not* on the raw payload bytes being readable.

5. **No PID emits plaintext game time** either. The promising hit (`PID 652 size-19 offset 8 f32`, range 0 → 2316) breaks down on close inspection — the values are punctuated by `1e36`/`-3e37` outliers, which means we were skimming a constant-prefix coincidence, not a timestamp.

6. **MLP-bytes-→-HP** (the strongest learned baseline that doesn't need to know the cipher) finds nothing. Across 11 PIDs × 10 entities × 10 heroes = 1100 (PID, entity, hero) triples trained, the best held-out R² is `0.006` with 1 % of test predictions within ±5 HP. If the cipher were byte-position deterministic and HP were transmitted, an MLP with ~1k labeled pairs would saturate. It does not.

   This is the negative-result money number.

## Why labeled-pairs hits a wall here

The combination of:

- **Per-PID lookup-table + arithmetic + bit-rotation cipher** (maknee blog), where the operations *mix* bytes, so a byte-position substitution table is not the right model.
- **HP not present as a transmitted primitive** — the rofl carries `TakeDamagePacket`s (and presumably heal / regen / stat events). HP is a *client-derived* quantity. The closest you can get from pure rofl bytes is a damage event stream you'd then have to integrate.
- **Per-patch table rotation**, which means even a working decoder has a maintenance cost on every patch.

means the *amount* of labeled data we have (60+ games × 15k ticks) doesn't help much, because the modelling problem isn't "lots of noisy labeled pairs of a deterministic function" — it's "the function isn't a 1:1 lookup over input bytes alone, and the quantity we have labels for isn't even directly transmitted."

## Suggested path to extend this (if someone keeps pulling on this thread)

In rough order of effort:

1. **Decode `TakeDamagePacket` (PID 985 looks like it).** The blog gives the field offsets: target id at +16 (u32), damage at +24 (f32), source id at +36 (u32), all encrypted. Per-PID-LUT inversion needs the lookup table, which lives in `League of Legends.exe` and rotates per patch. Two ways to get it: (a) extract it via static analysis for the current patch (re-do what the existing emulator did under 16.7, but for 16.9 — non-trivial; the cipher is obfuscated), or (b) run-time hook the decryption function and dump (input, output) pairs until the table is fully observed. Option (b) is what the blog author did; it's a Windows-side job and won't run on the danilogin node.

   If we can decode damage events, HP becomes:
   ```
   hp[t] = hp_max[t] − Σ damage_received(τ ≤ t) + Σ heal_received(τ ≤ t)
   ```
   Plus respawn snaps to full hp_max. This is a viable reconstruction path *given* damage decoding.

2. **`hp_max` and `gold` should be easier than `hp_current`.** Both are sparse step functions (~17–25 changes per game) and both correspond directly to packet events (`LevelUp`, `KillReward`). Same cipher problem applies, but the decoder would have ~10× fewer labeled pairs to satisfy and the output space is much smaller, so a tiny MLP per (PID, entity) with `block_index` as an additional feature might actually fit. Worth one focused try once we have a candidate level-up PID isolated.

3. **`position` doesn't share this problem.** PID 487 in 16.7 used a 14-bit packed u32 (`x = u32 & 0x3FFF, y = (u32 >> 14) & 0x3FFF`, scale `14914 / 16384`) — that *is* a publishable plaintext encoding because the rendered replay needs continuous position. PID 487 doesn't appear in 16.9, but the same 14-bit-packed encoding is likely in whatever replaced it. A short search over (PID, offset) for u32 values whose unpacked (x, y) lie in the map bounds and produce smooth trajectories should find it without any cipher work.

4. **If someone *really* wants HP from rofl alone**: port the blog author's exception-emulator approach (Rust + INT3 breakpoints) or the existing project's Unicorn emulator to 16.9.772. Order-of-weeks effort, with re-derivation needed every patch. Almost certainly not worth it — the live-replay pipeline already produces the labels we need at high quality.

## Concrete recommendation

Drop this thread. The project's existing approach — running the replay client live and dumping `raw_mem.json` from the running game — is the right one, and is exactly the "alternative method" the blog author identifies as practical. The 60+ games already on NFS came from that pipeline and they're high-fidelity. Spend the time on the agent / tokenizer / dynamics work instead.

If someone wants to pick this up later, `scripts/rofl_decode/parse_rofl.py` is a clean, dependency-light starting point — every other script in that folder is a probe, not load-bearing.

## Artifacts

- `scripts/rofl_decode/{parse_rofl,find_hero_pids,correlate_v2,correlate_v3,scan_f32_in_range,mlp_probe}.py`
- 10 paired games on NFS:
  ```
  /mnt/nfs/datasets/lol_replays_16_9_772/NA1_<id>/rofl/replay.rofl
  /mnt/nfs/datasets/lol_replays_16_9_772/NA1_<id>/raw_mem.json
  ```
  ids: 5550028932, 5550045094, 5550067582, 5550073400, 5550083511, 5550110638, 5552749278, 5552884026, 5553776931, 5553940769

## Open questions a follow-up should answer first

- Is PID 985 actually `TakeDamagePacket`? Confirm via the offset-16 / offset-36 encrypted entity-id structure once we have a decryption path.
- What's the entity↔champion mapping for `0x400000ae`–`0x400000b7`? Almost certainly recoverable from the *first* spawn packet (a one-byte champion index keyed by a known PID), but I never went looking for it because no decoder was working anyway.
- Does the cipher state really have no per-block dependency? maknee's description ("each byte goes through multiple transformations") leaves it ambiguous. A definitive test would be to find any PID with extremely repetitive plaintext (e.g. an all-zeros pad), see if its ciphertext is constant across blocks, and conclude.
