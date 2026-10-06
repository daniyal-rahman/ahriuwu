"""Patch-pinned Map11 navigation grid: v7.1 NGRID reader, immutable artifacts and the host collision oracle.

    python -m lanerl_jax.modern.data.navgrid import --ngrid F --out DIR --patch P --source URL --retrieved-at DATE

Format: FrankTheBoxMonster/LoL-NGRID-converter (NGridFileReader.cs, NavGridCell.cs) @ 92943ed2. Geometry
provenance is supplied by the caller, never inferred from file names or minimap images. World coordinates are
X,Z (League is Y-up); arrays are [z, x], not flipped image rows. Raw flags, region layers and heights are kept
so later ports (e.g. vision) need not re-extract them.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import PATCH, PATCH_DIR
from ..map.terrain import StaticTerrain

SCHEMA = "lanerl-map-grid-v1"
COORDINATES = "world-xz; arrays[z,x]"
BRUSH, WALL, STRUCTURE, TRANSPARENT = 1, 2, 4, 64
BLUE_ONLY, RED_ONLY = 1024, 2048


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class ModernMapGrid:
    flags: np.ndarray
    regions: np.ndarray  # four packed bytes/cell; retain all layers
    heights: np.ndarray
    height_spacing: tuple[float, float]
    cell_size: float
    min_bounds: tuple[float, float, float]
    max_bounds: tuple[float, float, float]

    def __post_init__(self):
        if self.flags.ndim != 2 or self.flags.dtype != np.dtype("uint16"):
            raise ValueError("flags must be a 2D uint16 grid")
        if self.regions.shape != (*self.flags.shape, 4) or self.regions.dtype != np.uint8:
            raise ValueError("regions must be four uint8 layers per cell")
        if self.heights.ndim != 2 or not self.heights.size or not np.isfinite(self.heights).all():
            raise ValueError("height samples must be a finite 2D grid")
        if not math.isfinite(self.cell_size) or self.cell_size <= 0:
            raise ValueError("cell_size must be positive and finite")
        if len(self.min_bounds) != 3 or len(self.max_bounds) != 3:
            raise ValueError("bounds must be XYZ triples")
        lo, hi = np.asarray(self.min_bounds), np.asarray(self.max_bounds)
        if not (np.isfinite(lo).all() and np.isfinite(hi).all() and (hi > lo).all()):
            raise ValueError("bounds must be finite and ordered")
        if len(self.height_spacing) != 2 or any(not math.isfinite(v) or v <= 0 for v in self.height_spacing):
            raise ValueError("height spacing must be positive and finite")
        # Extents may end inside the last cell but not disagree by a whole cell.
        expected = np.ceil((hi[[2, 0]] - lo[[2, 0]]) / self.cell_size).astype(int)
        if tuple(expected) != self.flags.shape:
            raise ValueError("grid dimensions disagree with world bounds")
        for a in (self.flags, self.regions, self.heights):
            a.setflags(write=False)

    @property
    def brush(self) -> np.ndarray:
        return (self.flags & BRUSH) != 0

    @property
    def main_region(self) -> np.ndarray:
        return self.regions[..., 1] >> 4

    def walkable(self, team: int | None = None) -> np.ndarray:
        """Static mask; ``team`` 0 blue / 1 red. Gates are open only to their owning team (resolved before the
        transparent-wall mask, as gates may carry both flags); structure cells stay blocked."""
        if team not in (None, 0, 1):
            raise ValueError("team must be None, 0 (blue), or 1 (red)")
        f = self.flags
        gate = f & (BLUE_ONLY | RED_ONLY)
        owned = gate == (BLUE_ONLY if team == 0 else RED_ONLY if team == 1 else -1)
        ordinary = (f & (WALL | STRUCTURE | TRANSPARENT)) == 0
        return np.where(gate != 0, owned & ((f & (WALL | STRUCTURE)) == 0), ordinary)

    def cell(self, x: float, z: float) -> tuple[int, int] | None:
        """Strict bounds; negative fractions never truncate into cell zero."""
        if not (math.isfinite(x) and math.isfinite(z)):
            return None
        if not (self.min_bounds[0] <= x < self.max_bounds[0]
                and self.min_bounds[2] <= z < self.max_bounds[2]):
            return None
        return (math.floor((x-self.min_bounds[0])/self.cell_size),
                math.floor((z-self.min_bounds[2])/self.cell_size))

    def is_walkable(self, x: float, z: float, *, radius: float = 0,
                    team: int | None = None) -> bool:
        """Host collision oracle: every cell touched by the disk is open.

        Radius zero tests the half-open cell containing the point; positive-radius boundary contact is blocked.
        Doubles here vs float32 in JAX: sub-float32 boundary offsets are not a cross-backend contract.
        """
        if not math.isfinite(radius) or radius < 0:
            raise ValueError("radius must be nonnegative and finite")
        cell = self.cell(x, z)
        walk = self.walkable(team)
        if cell is None:
            return False
        if radius == 0:
            return bool(walk[cell[1], cell[0]])
        if (x-radius <= self.min_bounds[0] or z-radius <= self.min_bounds[2]
                or x+radius >= self.max_bounds[0] or z+radius >= self.max_bounds[2]):
            return False
        nx, nz = ((x-self.min_bounds[0])/self.cell_size,
                  (z-self.min_bounds[2])/self.cell_size)
        r = radius/self.cell_size
        # nextafter: a disk exactly touching a cell's far edge must include that negative-side neighbour.
        xs = np.arange(math.floor(np.nextafter(nx-r, -np.inf)), math.floor(nx+r)+1)
        zs = np.arange(math.floor(np.nextafter(nz-r, -np.inf)), math.floor(nz+r)+1)
        dx = np.maximum(np.abs(nx-(xs[None, :]+.5))-.5, 0)
        dz = np.maximum(np.abs(nz-(zs[:, None]+.5))-.5, 0)
        touched = dx*dx + dz*dz <= r*r
        valid = ((xs[None, :] >= 0) & (xs[None, :] < walk.shape[1])
                 & (zs[:, None] >= 0) & (zs[:, None] < walk.shape[0]))
        values = walk[np.clip(zs, 0, walk.shape[0]-1)[:, None],
                      np.clip(xs, 0, walk.shape[1]-1)[None, :]]
        return bool(np.all(~touched | (valid & values)))

    def as_jax(self, team: int | None = None):
        import jax.numpy as jnp
        return StaticTerrain(jnp.asarray(self.walkable(team)), self.cell_size,
                             self.min_bounds[0], self.min_bounds[2],
                             self.max_bounds[0], self.max_bounds[2])


def read_ngrid(data: bytes) -> ModernMapGrid:
    """Read v7.1 only; other versions fail instead of guessing. Trailing hint tables are ignored."""
    if len(data) < 39:
        raise ValueError("truncated navigation header")
    major, minor = struct.unpack_from("<BH", data)
    if (major, minor) != (7, 1):
        raise ValueError(f"unsupported navigation version {major}.{minor}; expected 7.1")
    lo = struct.unpack_from("<3f", data, 3)
    hi = struct.unpack_from("<3f", data, 15)
    cell_size, width, height = struct.unpack_from("<fII", data, 27)
    count = width * height
    if not (0 < width <= 4096 and 0 < height <= 4096):
        raise ValueError("invalid navigation dimensions")
    # v7: 48-byte cell records, then uint16 flags, then 4 region bytes per cell, then 8 x 132 bytes.
    flag_offset = 39 + count*48
    region_offset = flag_offset + count*2
    height_offset = region_offset + count*4 + 8*132
    if len(data) < height_offset + 16:
        raise ValueError("truncated navigation cells or height header")
    flags = np.frombuffer(data, "<u2", count, flag_offset).reshape(height, width).copy()
    regions = np.frombuffer(data, np.uint8, count*4, region_offset).reshape(height, width, 4).copy()
    hw, hh, hx, hz = struct.unpack_from("<IIff", data, height_offset)
    if not (0 < hw <= 8192 and 0 < hh <= 8192) or len(data) < height_offset+16+hw*hh*4:
        raise ValueError("invalid or truncated height samples")
    heights = np.frombuffer(data, "<f4", hw*hh, height_offset+16).reshape(hh, hw).copy()
    return ModernMapGrid(flags, regions, heights, (hx, hz), cell_size, lo, hi)


def write_artifact(raw: bytes, out: Path, *, patch: str, source: str,
                   retrieved_at: str, variant: str = "base") -> dict:
    """Create a new immutable artifact; never overwrites a previous extraction."""
    if not all(isinstance(x, str) and x.strip() for x in (patch, source, retrieved_at, variant)):
        raise ValueError("patch, source, retrieval date and variant are required")
    grid = read_ngrid(raw)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    arrays = out / "grid.npz"
    np.savez_compressed(arrays, flags=grid.flags, regions=grid.regions, heights=grid.heights)
    manifest = dict(schema=SCHEMA, patch=patch, map_id=11, variant=variant,
                    source=source, retrieved_at=retrieved_at, source_sha256=sha256(raw),
                    arrays_sha256=sha256(arrays.read_bytes()),
                    cell_size=grid.cell_size, min_bounds=grid.min_bounds,
                    max_bounds=grid.max_bounds, height_spacing=grid.height_spacing,
                    coordinates=COORDINATES, dynamic_terrain=False,
                    navigation_format="7.1")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


def load_artifact(path: Path, *, expected_patch: str, expected_variant: str = "base",
                  expected_manifest_sha256: str | None = None):
    """``(grid, manifest)`` after identity/integrity checks. Without an external manifest pin, the sibling
    checksums catch corruption, not a simultaneous replacement of arrays and manifest."""
    path = Path(path)
    manifest_raw = (path / "manifest.json").read_bytes()
    if expected_manifest_sha256 is not None and sha256(manifest_raw) != expected_manifest_sha256:
        raise ValueError("map manifest checksum mismatch")
    m = json.loads(manifest_raw)
    if (m.get("schema"), m.get("map_id"), m.get("patch"), m.get("variant")) != (
            SCHEMA, 11, expected_patch, expected_variant):
        raise ValueError("map artifact schema/map/patch/variant mismatch")
    if m.get("coordinates") != COORDINATES or m.get("dynamic_terrain") is not False:
        raise ValueError("unsupported map coordinate or dynamic-terrain contract")
    raw = (path / "grid.npz").read_bytes()
    if sha256(raw) != m["arrays_sha256"]:
        raise ValueError("map array checksum mismatch")
    with np.load(io.BytesIO(raw), allow_pickle=False) as a:
        grid = ModernMapGrid(a["flags"], a["regions"], a["heights"],
                             tuple(m["height_spacing"]), m["cell_size"],
                             tuple(m["min_bounds"]), tuple(m["max_bounds"]))
    return grid, m


def load_patch_map(path: Path, *, patch: str = PATCH):
    """Load external (not in git) arrays against this checkout's reviewed manifest pin; no fallback map."""
    if patch != PATCH:
        raise ValueError(f"no reviewed modern map profile for {patch!r}")
    profile = json.loads((PATCH_DIR / "map11.json").read_text())
    return load_artifact(path, expected_patch=patch, expected_variant=profile["variant"],
                         expected_manifest_sha256=profile["manifest_sha256"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("import", help="import an extracted v7.1 NGRID")
    p.add_argument("--ngrid", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--patch", required=True, help="actual source patch, not desired target")
    p.add_argument("--source", required=True, help="asset URL or client build provenance")
    p.add_argument("--retrieved-at", required=True)
    p.add_argument("--variant", default="base")
    a = parser.parse_args()
    m = write_artifact(a.ngrid.read_bytes(), a.out, patch=a.patch, source=a.source,
                       retrieved_at=a.retrieved_at, variant=a.variant)
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
