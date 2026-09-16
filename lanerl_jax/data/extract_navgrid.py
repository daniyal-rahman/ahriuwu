#!/usr/bin/env python
"""Extract Summoner's Rift walkability from the server's own navigation grid.

Why this exists
---------------
Terrain was the risk I expected to force an approximation in the JAX sim, and it
does not. The server's grid is 293x294 cells at 50 units, i.e. 86,142 cells --
an 86 KB boolean array, a *static shared device constant*, not per-env state. So
terrain in the JAX sim can be exact rather than approximated.

It also makes pathing a table lookup instead of an A*. Inside the laning region
(middle 50% of the top lane, within LANE_HALF_WIDTH of the lane polyline) there
are ~6,344 walkable cells, so an all-pairs next-hop table is ~40 MB as uint8.
The straight-line-plus-sliding alternative was measured and rejected: a straight
line between two corridor cells is walkable 98.4% of the time at 500 units but
only 81.0% at 1800 -- and SCREEN_RADIUS is 1800, so a click can name a point that
far away. One click in five would path wrongly.

Format is read straight off the server's own loader:
  GameServerLib/Content/Navigation/NavigationGrid.cs:95-170   header
  GameServerLib/Content/Navigation/NavigationGridCell.cs:49   ReadVersion5, 56 B/cell
Flags per GameServerCore/Enums/NavigationGridCellFlags.cs.

Verified against Map1 (version 3.1): renders as recognisable Summoner's Rift,
62.3% walkable.
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np

# NavigationGridCellFlags
HAS_GRASS = 0x1
NOT_PASSABLE = 0x2
SEE_THROUGH = 0x40

# ReadVersion5 field layout, in bytes from the start of a cell:
#   0  centerHeight f32      24  locator.X i16
#   4  sessionId    i32      26  locator.Y i16
#   8  arrivalCost  f32      28  additionalCost f32
#  12  isOpen       u32      32  hintAsGoodCell f32
#  16  heuristic    f32      36  additionalCostRefCount u32
#  20  actorList    u32      40  goodCellSessionId i32
#                            44  refHintWeight f32
#                            48  arrivalDirection i16
#                            50  flags u16            <- the only field we need
#                            52  refHintNode i16[2]
CELL_BYTES = 56
FLAGS_OFFSET = 50


def load(path: Path):
    """Return (flags (cy,cx) uint16, cell_size, min_grid, max_grid)."""
    d = path.read_bytes()
    off = 0
    major = d[off]
    off += 1
    if major != 2:
        off += 2  # minor u16
    if major not in (2, 3, 5, 7):
        raise ValueError(f"unsupported navgrid version {major}")
    if major == 7:
        raise NotImplementedError("version 7 stores flags in a separate block")

    min_grid = struct.unpack_from("<3f", d, off)
    off += 12
    max_grid = struct.unpack_from("<3f", d, off)
    off += 12
    cell_size = struct.unpack_from("<f", d, off)[0]
    off += 4
    cx, cy = struct.unpack_from("<II", d, off)
    off += 8

    n = cx * cy
    raw = np.frombuffer(d, dtype=np.uint8, count=n * CELL_BYTES, offset=off)
    raw = raw.reshape(n, CELL_BYTES)
    flags = raw[:, FLAGS_OFFSET:FLAGS_OFFSET + 2].copy().view(np.uint16).reshape(cy, cx)
    return flags, cell_size, min_grid, max_grid


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ngrid",
        type=Path,
        default=None,
    )
    ap.add_argument("--out", type=Path, default=Path("sr_walk.npy"))
    ap.add_argument("--show", action="store_true", help="print an ASCII map to eyeball it")
    args = ap.parse_args()
    if args.ngrid is None:
        from .paths import ngrid_path

        args.ngrid = ngrid_path(1)

    flags, cell_size, mn, mx = load(args.ngrid)
    cy, cx = flags.shape
    # `IsWalkable` tests BOTH flags (NavigationGrid.cs:495). Using NOT_PASSABLE
    # alone over-counts by 516 cells on Map1 -- small, and exactly the size of
    # thing that produces an unexplained trajectory divergence months later.
    walk = ((flags & NOT_PASSABLE) == 0) & ((flags & SEE_THROUGH) == 0)

    print(f"grid {cx}x{cy}  cellSize {cell_size}  cells {cx * cy}")
    print(f"min {mn}  max {mx}")
    print(f"walkable {walk.sum()} ({100 * walk.mean():.1f}%)  "
          f"grass {int(((flags & HAS_GRASS) > 0).sum())}  "
          f"seethrough {int(((flags & SEE_THROUGH) > 0).sum())}")

    np.save(args.out, walk)
    print(f"wrote {args.out} ({walk.nbytes / 1024:.0f} KB as bool)")

    if args.show:
        for r in range(0, cy, 10):
            print("".join("." if walk[r, c] else "#" for c in range(0, cx, 5)))


if __name__ == "__main__":
    main()
