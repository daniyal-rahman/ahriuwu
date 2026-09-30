"""Explicit, patch-pinned Map11 navigation assets; independent of Map1's oracle.

TOOL: ``python -m lanerl_jax.data.modern_map import --help``.
Format evidence: FrankTheBoxMonster/LoL-NGRID-converter (NGridFileReader.cs,
NavGridCell.cs), revision 92943ed2b2d5e82c86d680e69f53f247c89aefee.
This reader describes the v7.1 file layout; it does not adopt that project's
renderer or the vendored C# pathfinder. Geometry provenance is supplied by the
caller, never inferred from the file's name or from a minimap image.

World coordinates are X,Z (League is Y-up). Arrays are [z, x], NOT image rows
flipped vertically. The raw terrain/region flags and height samples survive
normalisation so a future vision port need not re-extract them.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np

SCHEMA = "lanerl-map-grid-v1"
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
        # File extents can end inside the final grid cell, but cannot disagree
        # by a whole cell. Do not hardcode historical Map11 dimensions.
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
        """Static conservative mask; team uses sim convention 0 blue / 1 red.

        Gates are blocked unless queried for their owning team. A gate may
        combine team-only and transparent-wall flags: its permission must be
        resolved before the ordinary transparent-wall mask. Structure cells
        stay blocked here; structure destruction belongs to a later overlay.
        """
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
        """Host collision oracle: all cells touched by a positive disk are open.

        Radius zero queries only the half-open cell containing the point;
        positive-radius boundary contact is blocked. Host arithmetic uses
        doubles; JAX uses float32, so sub-float32 boundary offsets are not a
        supported cross-backend distinction. This is an explicit geometric
        contract, not a claim to reproduce the modern client's route selection.
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
        # Include the negative-side neighbour when the disk just touches its
        # far edge. Without nextafter, exact integer boundaries miss that cell.
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
        from ..sim.modern_terrain import StaticTerrain
        import jax.numpy as jnp
        return StaticTerrain(jnp.asarray(self.walkable(team)), self.cell_size,
                             self.min_bounds[0], self.min_bounds[2],
                             self.max_bounds[0], self.max_bounds[2])


def read_ngrid(data: bytes) -> ModernMapGrid:
    """Read v7.1 explicitly; unsupported formats fail instead of guessing."""
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
    # v7 separates 48-byte cell records, uint16 flags and 4 region bytes.
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
    # Trailing hint tables are not used by this importer. We do not claim their
    # interpretation, nor silently feed them to the legacy C# pathfinder.
    return ModernMapGrid(flags, regions, heights, (hx, hz), cell_size, lo, hi)


def write_artifact(raw: bytes, out: Path, *, patch: str, source: str,
                   retrieved_at: str, variant: str = "base") -> dict:
    """Create a new immutable artifact. Never overwrite a previous extraction."""
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
                    coordinates="world-xz; arrays[z,x]", dynamic_terrain=False,
                    navigation_format="7.1")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


def load_artifact(path: Path, *, expected_patch: str, expected_variant: str = "base",
                  expected_manifest_sha256: str | None = None):
    """Check identity/integrity, optionally against an external manifest pin.

    Without an external pin, sibling checksums detect accidental corruption,
    not simultaneous replacement of the arrays and their manifest.
    """
    path = Path(path)
    manifest_raw = (path / "manifest.json").read_bytes()
    if expected_manifest_sha256 is not None and sha256(manifest_raw) != expected_manifest_sha256:
        raise ValueError("map manifest checksum mismatch")
    m = json.loads(manifest_raw)
    if (m.get("schema"), m.get("map_id"), m.get("patch"), m.get("variant")) != (
            SCHEMA, 11, expected_patch, expected_variant):
        raise ValueError("map artifact schema/map/patch/variant mismatch")
    if m.get("coordinates") != "world-xz; arrays[z,x]" or m.get("dynamic_terrain") is not False:
        raise ValueError("unsupported map coordinate or dynamic-terrain contract")
    raw = (path / "grid.npz").read_bytes()
    if sha256(raw) != m["arrays_sha256"]:
        raise ValueError("map array checksum mismatch")
    import io
    with np.load(io.BytesIO(raw), allow_pickle=False) as a:
        grid = ModernMapGrid(a["flags"], a["regions"], a["heights"],
                             tuple(m["height_spacing"]), m["cell_size"],
                             tuple(m["min_bounds"]), tuple(m["max_bounds"]))
    return grid, m


def load_patch_map(path: Path, *, patch: str = "26.19"):
    """Load external arrays using this checkout's reviewed patch/manifest pin.

    Assets are not bundled in Git. The caller must explicitly supply their
    location; unknown patches never fall back to the installed legacy map.
    """
    if patch != "26.19":
        raise ValueError(f"no reviewed modern map profile for {patch!r}")
    profile = json.loads((Path(__file__).parent / "modern" / patch / "map11.json").read_text())
    return load_artifact(path, expected_patch=patch, expected_variant=profile["variant"],
                         expected_manifest_sha256=profile["manifest_sha256"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
