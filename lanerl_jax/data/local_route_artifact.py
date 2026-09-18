"""Offline source-cell × local-goal route artifact.

Unlike :mod:`route_artifact`'s prototype ``K x K`` matrix, this format follows
the actual follow-camera action topology.  For every covered source cell it
stores only goals within ``offset_radius`` cells on each axis.  Repeated local
hop gathers reconstruct a route while the goal remains fixed.

The bake uses a radius-aware static cell graph and deterministic reverse BFS.
That gives robust shortest routes and makes a whole-Map1 artifact practical,
but it is not the vendored server's closed-on-enqueue heuristic A*.  This is an
intentional, named approximation (PATH-003 in the fidelity ledger), not an
exactness claim.  The manifest pins the algorithm, navgrid bytes, radius,
coverage, and table bytes so incompatible artifacts fail closed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .navgrid import DEFAULT_NGRID, NavGrid
from .route_artifact import DIRECTION_OFFSETS, NO_ROUTE, STAY, RouteArtifactError

__all__ = [
    "LOCAL_ARTIFACT_VERSION", "LocalRouteManifest", "LocalRouteArtifact",
    "build_local_route_artifact", "load_local_route_artifact",
]


LOCAL_ARTIFACT_VERSION = 3
COMPATIBLE_ARTIFACT_VERSIONS = (2, LOCAL_ARTIFACT_VERSION)
MANIFEST_NAME = "manifest.json"
SOURCE_CELLS_NAME = "source_cells.npy"
NEXT_HOP_NAME = "next_hop.npy"
RUN_LENGTH_NAME = "run_length.npy"
TABLE_SEMANTICS = "radius-aware-local-cell-graph-reverse-bfs-v1"
TABLE_ENCODING = "row-major-two-uint4-per-byte-low-nibble-first-no-route-15"
RUN_LENGTH_SEMANTICS = (
    "uint8-number-of-consecutive-raw-hops-with-this-entrys-direction; "
    "zero-for-stay-or-no-route")
ENDPOINT_POLICY = ("source-and-goal-cell-centres; runtime-preserves-float-goal; "
                   "runtime-projects-blocked-goal-before-cell-lookup")
ALGORITHM = ("symmetric CastCircle-valid 8-neighbour graph; reverse BFS per goal; "
             "DIRECTION_OFFSETS order; search confined to goal-centred offset window")


def _sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def _array_sha256(a: np.ndarray, chunk_rows: int = 1024) -> str:
    h = hashlib.sha256()
    if a.ndim == 0:
        h.update(np.ascontiguousarray(a).tobytes())
    else:
        for start in range(0, len(a), chunk_rows):
            h.update(np.ascontiguousarray(a[start:start + chunk_rows]).tobytes())
    return h.hexdigest()


def _f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", np.float32(x)))[0]


@dataclass(frozen=True)
class LocalRouteManifest:
    version: int
    map_id: int
    ngrid_sha256: str
    grid_shape: tuple[int, int]
    cell_size_f32_bits: int
    pathfinding_radius_f32_bits: int
    offset_radius: int
    source_count: int
    source_sha256: str
    logical_table_shape: tuple[int, int, int]
    table_shape: tuple[int, int]
    table_sha256: str
    table_dtype: str
    table_semantics: str
    table_encoding: str
    endpoint_policy: str
    algorithm: str
    run_length_shape: tuple[int, int, int] | None = None
    run_length_sha256: str = ""
    run_length_dtype: str = ""
    run_length_semantics: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":")) + "\n"

    @classmethod
    def from_json(cls, raw: str) -> "LocalRouteManifest":
        try:
            data = json.loads(raw)
            data["grid_shape"] = tuple(data["grid_shape"])
            data["logical_table_shape"] = tuple(data["logical_table_shape"])
            data["table_shape"] = tuple(data["table_shape"])
            if data.get("run_length_shape") is not None:
                data["run_length_shape"] = tuple(data["run_length_shape"])
            return cls(**data)
        except (KeyError, TypeError, ValueError) as exc:
            raise RouteArtifactError("invalid local-route manifest") from exc


@dataclass(frozen=True)
class LocalRouteArtifact:
    manifest: LocalRouteManifest
    source_cells: np.ndarray
    cell_to_row: np.ndarray
    next_hop: np.ndarray
    run_length: np.ndarray | None

    def as_jax(self):
        import jax.numpy as jnp
        from ..sim.local_pathing import LocalRouteTable

        return LocalRouteTable(
            jnp.asarray(self.cell_to_row), jnp.asarray(self.next_hop),
            self.manifest.offset_radius,
            None if self.run_length is None else jnp.asarray(self.run_length))


def _normalise_coverage(coverage: Sequence[int] | np.ndarray | None,
                        traversable: np.ndarray) -> np.ndarray:
    if coverage is None:
        return np.flatnonzero(traversable).astype(np.int32)
    cells = np.asarray(coverage)
    if cells.dtype == np.bool_:
        if cells.shape != traversable.shape:
            raise RouteArtifactError(
                f"coverage shape {cells.shape}, expected {traversable.shape}")
        cells = np.flatnonzero(cells)
    cells = np.unique(np.asarray(cells, np.int32).reshape(-1))
    if not len(cells):
        raise RouteArtifactError("local route coverage is empty")
    if cells[0] < 0 or cells[-1] >= traversable.size:
        raise RouteArtifactError("local route coverage contains an out-of-grid cell")
    if not np.all(traversable.reshape(-1)[cells]):
        raise RouteArtifactError("coverage contains a cell centre blocked at this radius")
    return cells


def _traversable_centres(grid: NavGrid, radius: float) -> np.ndarray:
    out = np.zeros(grid.flags.shape, bool)
    for iy in range(grid.cell_count_y):
        for ix in range(grid.cell_count_x):
            x, y = grid.cell_center_world(ix, iy)
            out[iy, ix] = grid.is_walkable_world(x, y, radius)
    return out


def _adjacent_graph(grid: NavGrid, radius: float, covered: np.ndarray) -> np.ndarray:
    """Global-cell adjacency, columns use direction codes 1..8."""
    width, height = grid.cell_count_x, grid.cell_count_y
    result = np.full((grid.flags.size, 8), -1, np.int32)
    covered_flat = covered.reshape(-1)
    for cell in np.flatnonzero(covered_flat):
        y, x = divmod(int(cell), width)
        ox, oy = x + 0.5, y + 0.5
        for code, (dx, dy) in enumerate(DIRECTION_OFFSETS[1:], start=1):
            nx, ny = x + int(dx), y + int(dy)
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            neighbour = ny * width + nx
            if not covered_flat[neighbour]:
                continue
            if not grid.cast_circle(ox, oy, nx + 0.5, ny + 0.5, radius):
                result[cell, code - 1] = neighbour
    # Reverse BFS consumes an outgoing edge as the corresponding incoming edge.
    # CastCircle is geometrically symmetric, but its grid-line iterator has
    # exact-corner tie behavior; assert the property on the actual baked graph
    # rather than trusting the geometry. Map1/Garen measured 360,472 directed
    # edges and zero asymmetric pairs.
    reverse = {tuple(map(int, d)): i for i, d in enumerate(DIRECTION_OFFSETS[1:])}
    for cell in np.flatnonzero(covered_flat):
        for edge, neighbour in enumerate(result[cell]):
            if neighbour < 0:
                continue
            dx, dy = map(int, DIRECTION_OFFSETS[edge + 1])
            reverse_edge = reverse[(-dx, -dy)]
            if result[neighbour, reverse_edge] != cell:
                raise RouteArtifactError(
                    "local route graph has an asymmetric CastCircle edge; "
                    "reverse BFS would encode an invalid forward hop")
    return result


def _bake_local_table(cell_to_row: np.ndarray, neighbours: np.ndarray,
                      source_cells: np.ndarray, width: int, height: int,
                      offset_radius: int, output: np.ndarray,
                      run_length: np.ndarray | None = None) -> None:
    """Numba-parallel bounded reverse BFS; isolated for unit testing."""
    try:
        from numba import njit, prange
    except ImportError as exc:  # offline generation needs speed, never guess
        raise RuntimeError(
            "building a production local-route artifact requires numba") from exc

    offsets = np.asarray(DIRECTION_OFFSETS[1:], np.int32)
    opposite = np.empty(8, np.uint8)
    by_offset = {tuple(map(int, d)): i for i, d in enumerate(DIRECTION_OFFSETS)}
    for k, (dx, dy) in enumerate(offsets):
        opposite[k] = by_offset[(-int(dx), -int(dy))]

    if run_length is None:
        # Retain the small unit-test-facing helper API. Production always
        # supplies the memmapped sidecar below.
        run_length = np.zeros_like(output)

    @njit(parallel=True, cache=False)
    def bake(rows, graph, cells, grid_width, grid_height, radius, table, runs,
             dirs, reverse_codes):
        side = 2 * radius + 1
        capacity = side * side
        for goal_row in prange(cells.shape[0]):
            goal = cells[goal_row]
            gy, gx = divmod(goal, grid_width)
            seen = np.zeros(capacity, np.uint8)
            queue = np.empty(capacity, np.int32)
            head, tail = 0, 1
            queue[0] = goal
            seen[radius * side + radius] = 1
            table[goal_row, radius, radius] = np.uint8(0)
            runs[goal_row, radius, radius] = np.uint8(0)
            while head < tail:
                current = queue[head]
                head += 1
                cy, cx = divmod(current, grid_width)
                for edge in range(8):
                    source = graph[current, edge]
                    if source < 0:
                        continue
                    sy, sx = divmod(source, grid_width)
                    local_x, local_y = sx - gx + radius, sy - gy + radius
                    if not (0 <= local_x < side and 0 <= local_y < side):
                        continue
                    seen_index = local_y * side + local_x
                    if seen[seen_index] != 0:
                        continue
                    source_row = rows[source]
                    if source_row < 0:
                        continue
                    seen[seen_index] = 1
                    queue[tail] = source
                    tail += 1
                    # graph[current, edge] is current -> source. The route
                    # stored at source must take the opposite edge back toward
                    # the already-reached current cell.
                    goal_y, goal_x = gy - sy + radius, gx - sx + radius
                    code = reverse_codes[edge]
                    table[source_row, goal_y, goal_x] = code
                    # `current` was already reached by this goal's BFS, so its
                    # entry is available now.  A run continues only while the
                    # *next* raw hop is the same direction; otherwise this
                    # source begins a one-hop run.  The result is exactly the
                    # number of adjacent edges runtime may jump without
                    # skipping a collinear-compression turning point.
                    current_row = rows[current]
                    current_code = table[current_row,
                                         gy - cy + radius,
                                         gx - cx + radius]
                    if current_code == code:
                        runs[source_row, goal_y, goal_x] = (
                            runs[current_row,
                                 gy - cy + radius,
                                 gx - cx + radius] + np.uint8(1))
                    else:
                        runs[source_row, goal_y, goal_x] = np.uint8(1)

    bake(cell_to_row, neighbours, source_cells, width, height,
         offset_radius, output, run_length, offsets, opposite)


def build_local_route_artifact(out_dir: Path | str, grid: NavGrid,
                               ngrid_path: Path | str, pathfinding_radius: float,
                               offset_radius: int, *,
                               coverage: Sequence[int] | np.ndarray | None = None,
                               map_id: int = 1) -> LocalRouteManifest:
    """Build a deterministic local table without loading it all into RAM."""
    if offset_radius <= 0:
        raise RouteArtifactError("offset_radius must be positive")
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty artifact directory {out}")
    out.mkdir(parents=True, exist_ok=True)
    ngrid_path = Path(ngrid_path)

    traversable = _traversable_centres(grid, pathfinding_radius)
    source_cells = _normalise_coverage(coverage, traversable)
    covered = np.zeros(grid.flags.size, bool)
    covered[source_cells] = True
    covered = covered.reshape(grid.flags.shape)
    cell_to_row = np.full(grid.flags.size, -1, np.int32)
    cell_to_row[source_cells] = np.arange(len(source_cells), dtype=np.int32)
    neighbours = _adjacent_graph(grid, pathfinding_radius, covered)

    side = 2 * offset_radius + 1
    table_path = out / NEXT_HOP_NAME
    raw_path = out / ".next_hop_unpacked.tmp.npy"
    raw = np.lib.format.open_memmap(
        raw_path, mode="w+", dtype=np.uint8,
        shape=(len(source_cells), side, side))
    raw[:] = NO_ROUTE
    raw.flush()
    run_path = out / RUN_LENGTH_NAME
    runs = np.lib.format.open_memmap(
        run_path, mode="w+", dtype=np.uint8,
        shape=(len(source_cells), side, side))
    runs[:] = 0
    runs.flush()
    started = time.monotonic()
    _bake_local_table(cell_to_row, neighbours, source_cells,
                      grid.cell_count_x, grid.cell_count_y, offset_radius, raw,
                      runs)
    raw.flush()
    runs.flush()
    elapsed = time.monotonic() - started

    logical_width = side * side
    packed_width = (logical_width + 1) // 2
    table = np.lib.format.open_memmap(
        table_path, mode="w+", dtype=np.uint8,
        shape=(len(source_cells), packed_width))
    raw_flat = raw.reshape(len(source_cells), logical_width)
    for start in range(0, len(source_cells), 1024):
        stop = min(start + 1024, len(source_cells))
        block = raw_flat[start:stop]
        nibbles = np.where(block == NO_ROUTE, 0xF, block).astype(np.uint8)
        if logical_width & 1:
            nibbles = np.pad(nibbles, ((0, 0), (0, 1)), constant_values=0xF)
        table[start:stop] = nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)
    table.flush()
    del raw, raw_flat, runs
    raw_path.unlink()

    np.save(out / SOURCE_CELLS_NAME, source_cells, allow_pickle=False)
    manifest = LocalRouteManifest(
        version=LOCAL_ARTIFACT_VERSION,
        map_id=int(map_id),
        ngrid_sha256=_sha256_file(ngrid_path),
        grid_shape=tuple(map(int, grid.flags.shape)),
        cell_size_f32_bits=_f32_bits(grid.cell_size),
        pathfinding_radius_f32_bits=_f32_bits(pathfinding_radius),
        offset_radius=int(offset_radius),
        source_count=int(len(source_cells)),
        source_sha256=_array_sha256(source_cells),
        logical_table_shape=(int(len(source_cells)), side, side),
        table_shape=tuple(map(int, table.shape)),
        table_sha256=_array_sha256(table),
        table_dtype="uint8",
        table_semantics=TABLE_SEMANTICS,
        table_encoding=TABLE_ENCODING,
        endpoint_policy=ENDPOINT_POLICY,
        algorithm=ALGORITHM,
        run_length_shape=(int(len(source_cells)), side, side),
        run_length_sha256=_array_sha256(np.load(run_path, mmap_mode="r",
                                                allow_pickle=False)),
        run_length_dtype="uint8",
        run_length_semantics=RUN_LENGTH_SEMANTICS,
    )
    (out / MANIFEST_NAME).write_text(manifest.to_json(), encoding="utf-8")
    print(f"baked {len(source_cells):,} source rows, {table.nbytes / 2**20:.1f} MiB packed + "
          f"{run_path.stat().st_size / 2**20:.1f} MiB runs "
          f"in {elapsed:.1f}s")
    return manifest


def load_local_route_artifact(path: Path | str, *,
                              ngrid_path: Path | str = DEFAULT_NGRID,
                              pathfinding_radius: float,
                              map_id: int = 1) -> LocalRouteArtifact:
    root = Path(path)
    manifest = LocalRouteManifest.from_json(
        (root / MANIFEST_NAME).read_text(encoding="utf-8"))
    if manifest.version not in COMPATIBLE_ARTIFACT_VERSIONS:
        raise RouteArtifactError("local-route artifact version mismatch")
    if manifest.map_id != map_id:
        raise RouteArtifactError("local-route map id mismatch")
    if (manifest.table_semantics != TABLE_SEMANTICS
            or manifest.table_encoding != TABLE_ENCODING
            or manifest.endpoint_policy != ENDPOINT_POLICY
            or manifest.algorithm != ALGORITHM):
        raise RouteArtifactError("local-route artifact ABI semantics mismatch")
    if (manifest.version >= 3
            and manifest.run_length_semantics != RUN_LENGTH_SEMANTICS):
        raise RouteArtifactError("local-route run-length ABI semantics mismatch")
    ngrid_path = Path(ngrid_path)
    if manifest.ngrid_sha256 != _sha256_file(ngrid_path):
        raise RouteArtifactError("local-route navgrid hash mismatch")
    if manifest.pathfinding_radius_f32_bits != _f32_bits(pathfinding_radius):
        raise RouteArtifactError("local-route pathfinding radius mismatch")
    grid = NavGrid.load(ngrid_path)
    if (manifest.grid_shape != tuple(grid.flags.shape)
            or manifest.cell_size_f32_bits != _f32_bits(grid.cell_size)):
        raise RouteArtifactError("local-route grid metadata mismatch")

    cells = np.load(root / SOURCE_CELLS_NAME, mmap_mode="r", allow_pickle=False)
    table = np.load(root / NEXT_HOP_NAME, mmap_mode="r", allow_pickle=False)
    runs = None
    if manifest.version >= 3:
        runs = np.load(root / RUN_LENGTH_NAME, mmap_mode="r", allow_pickle=False)
    side = 2 * manifest.offset_radius + 1
    if (cells.dtype != np.int32 or cells.ndim != 1
            or len(cells) != manifest.source_count
            or _array_sha256(cells) != manifest.source_sha256):
        raise RouteArtifactError("local-route source array invalid or tampered")
    packed_width = (side * side + 1) // 2
    if (manifest.logical_table_shape != (len(cells), side, side)
            or table.dtype != np.uint8
            or tuple(table.shape) != manifest.table_shape
            or table.shape != (len(cells), packed_width)
            or manifest.table_dtype != "uint8"
            or _array_sha256(table) != manifest.table_sha256):
        raise RouteArtifactError("local-route table invalid or tampered")
    if runs is not None and (
            runs.dtype != np.uint8
            or runs.shape != (len(cells), side, side)
            or manifest.run_length_shape != tuple(runs.shape)
            or manifest.run_length_dtype != "uint8"
            or _array_sha256(runs) != manifest.run_length_sha256):
        raise RouteArtifactError("local-route run-length array invalid or tampered")
    if (not len(cells) or not np.all(cells[1:] > cells[:-1])
            or cells[0] < 0 or cells[-1] >= grid.flags.size):
        raise RouteArtifactError("local-route sources are not sorted unique cells")
    # Validate in chunks: materialising low/high/valid arrays for the entire
    # packed Map1 table would briefly turn a 231 MiB asset into ~1 GiB of host
    # memory for no semantic benefit.
    for start in range(0, len(table), 1024):
        block = np.asarray(table[start:start + 1024])
        low, high = block & 0xF, block >> 4
        valid_low = (low <= len(DIRECTION_OFFSETS) - 1) | (low == 0xF)
        valid_high = (high <= len(DIRECTION_OFFSETS) - 1) | (high == 0xF)
        if not (np.all(valid_low) and np.all(valid_high)):
            raise RouteArtifactError("local-route table contains invalid direction code")
    cell_to_row = np.full(grid.flags.size, -1, np.int32)
    cell_to_row[cells] = np.arange(len(cells), dtype=np.int32)
    return LocalRouteArtifact(manifest, cells, cell_to_row, table, runs)


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--radius", type=float, required=True)
    ap.add_argument("--offset-radius", type=int, default=40)
    ap.add_argument("--coverage", type=Path,
                    help="optional .npy bool mask or sorted flat cell ids")
    ap.add_argument("--ngrid", type=Path, default=DEFAULT_NGRID)
    ap.add_argument("--map-id", type=int, default=1)
    args = ap.parse_args()
    grid = NavGrid.load(args.ngrid)
    coverage = (None if args.coverage is None
                else np.load(args.coverage, allow_pickle=False))
    build_local_route_artifact(
        args.out, grid, args.ngrid, args.radius, args.offset_radius,
        coverage=coverage, map_id=args.map_id)


if __name__ == "__main__":
    _main()
