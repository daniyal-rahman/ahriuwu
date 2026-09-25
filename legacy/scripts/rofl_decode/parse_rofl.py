#!/usr/bin/env python3
"""Parse a .rofl file into a stream of network blocks.

A .rofl is a Riot replay file: header + a sequence of zstd-compressed frames,
each containing many small per-tick blocks. Each block carries a packet-id (PID)
and an entity-id (param), plus an opaque (and per-PID-encrypted) payload.

This module is intentionally minimal — it ports the scanner from
`scripts/_archive/probes/emulator_v2.py` (parse_rofl_blocks / _scan_blocks)
into a standalone, importable function with no PE / Unicorn dependency.

Block format (heuristic, from prior reverse-engineering at patch 16.7,
re-validated empirically per-rofl in the analyser):

  off  size  meaning
   0    1   marker  (one of {0x91, 0xf1, 0xb1, 0x31, 0x11})
   1    1   channel
   2    1   payload-size (u8)
   3    2   pid     (u16, little-endian)
   5    4   param   (u32, little-endian — usually a net entity id like 0x400000xx)
   9    N   payload (size bytes, opaque/encrypted)
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Iterable, Iterator

import zstandard as zstd

ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
BLOCK_MARKERS = {0x91, 0xF1, 0xB1, 0x31, 0x11}


@dataclass
class Block:
    frame_idx: int   # which decompressed frame this block came from
    marker: int
    channel: int
    pid: int
    param: int       # entity id (typically 0x400000xx for heroes)
    payload: bytes


def _decompress_frames(data: bytes) -> list[bytes]:
    """Walk the raw .rofl bytes finding zstd frames; return the decompressed payloads."""
    dctx = zstd.ZstdDecompressor()
    frames: list[bytes] = []
    pos = 0
    while pos < len(data):
        idx = data.find(ZSTD_MAGIC, pos)
        if idx < 0:
            break
        try:
            dec = dctx.decompress(
                data[idx : idx + 0x100000],
                max_output_size=0x400000,
            )
            frames.append(dec)
        except Exception:
            pass
        pos = idx + 1
    return frames


def _scan_frame(fd: bytes, frame_idx: int) -> Iterator[Block]:
    """Walk a decompressed frame yielding Block records. Tolerant of garbage:
    if the next byte after a block isn't a marker we resync forward."""
    pos = 0
    n = len(fd)
    while pos + 9 <= n:
        marker = fd[pos]
        if marker in BLOCK_MARKERS:
            size = fd[pos + 2]
            pid = struct.unpack_from("<H", fd, pos + 3)[0]
            param = struct.unpack_from("<I", fd, pos + 5)[0]
            end = pos + 9 + size
            if end <= n:
                # Validate by checking that whatever comes next is plausible —
                # either we ran off the end or the next byte is also a marker.
                if end >= n - 9 or fd[end] in BLOCK_MARKERS:
                    yield Block(
                        frame_idx=frame_idx,
                        marker=marker,
                        channel=fd[pos + 1],
                        pid=pid,
                        param=param,
                        payload=bytes(fd[pos + 9 : end]),
                    )
                    pos = end
                    continue
        pos += 1


def parse_rofl(path: str) -> list[Block]:
    """Parse a .rofl file end-to-end. Returns a flat list of Block records,
    in file order. The block's `frame_idx` is a monotonic time-correlate (zstd
    frames roughly map to network ticks)."""
    with open(path, "rb") as f:
        data = f.read()
    frames = _decompress_frames(data)
    blocks: list[Block] = []
    for i, fd in enumerate(frames):
        blocks.extend(_scan_frame(fd, i))
    return blocks


def summarize(blocks: Iterable[Block]) -> dict:
    """Build small summary stats for sanity-checking a parse."""
    from collections import Counter, defaultdict

    pid_counts: Counter = Counter()
    pid_param_counts: dict[int, Counter] = defaultdict(Counter)
    pid_sizes: dict[int, list[int]] = defaultdict(list)
    params_seen: Counter = Counter()
    n_frames = 0

    for b in blocks:
        pid_counts[b.pid] += 1
        pid_param_counts[b.pid][b.param] += 1
        pid_sizes[b.pid].append(len(b.payload))
        params_seen[b.param] += 1
        n_frames = max(n_frames, b.frame_idx + 1)

    return {
        "n_blocks": sum(pid_counts.values()),
        "n_frames": n_frames,
        "n_pids": len(pid_counts),
        "top_pids": pid_counts.most_common(20),
        "top_params": params_seen.most_common(20),
        "pid_param_counts": pid_param_counts,
        "pid_sizes": pid_sizes,
    }


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else (
        "/mnt/nfs/datasets/lol_replays_16_9_772/NA1_5552884026/rofl/replay.rofl"
    )
    blocks = parse_rofl(path)
    s = summarize(blocks)
    print(f"file:     {path}")
    print(f"frames:   {s['n_frames']}")
    print(f"blocks:   {s['n_blocks']}")
    print(f"pids:     {s['n_pids']}")
    print()
    print("top PIDs by block count:")
    for pid, cnt in s["top_pids"]:
        sizes = s["pid_sizes"][pid]
        n_params = len(s["pid_param_counts"][pid])
        print(
            f"  pid={pid:5d}  n={cnt:7d}  "
            f"sz=[{min(sizes):>3}, {max(sizes):>3}]  "
            f"distinct_params={n_params}"
        )
    print()
    print("top params (entity ids) overall:")
    for param, cnt in s["top_params"]:
        print(f"  param=0x{param:08x}  n={cnt}")
