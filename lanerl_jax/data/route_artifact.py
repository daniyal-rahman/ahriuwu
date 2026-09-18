"""Deterministic, fail-closed storage for an *offline* Map1 route bake.

The server's ``NavigationGrid.GetPath`` is A* plus ``SmoothPath`` and cannot
run inside a JAX step.  A table can remove the search only if it is tied to the
exact navgrid bytes, pathfinding radius and covered cells that produced it.
This module defines that contract; it intentionally does not ship a large
table or make ``sim.orders`` claim that a cell-centre table is exact for an
arbitrary float-valued player click.

The matrix stores an adjacent, unsmoothed A* direction for every ordered pair
of covered cells.  Runtime reconstructs a raw cell itinerary with gathers and
then must perform the server-equivalent fixed-shape smoothing pass.  See
``docs/JAX_REWRITE_PLAN.md`` for the explicit endpoint and integration gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple, Sequence

import numpy as np

from .navgrid import DEFAULT_NGRID, NavGrid

__all__ = [
    "ARTIFACT_VERSION", "NO_ROUTE", "STAY", "DIRECTION_OFFSETS",
    "RouteStatus", "RouteArtifactError", "RouteManifest", "RouteArtifact",
    "JaxRouteTable", "build_route_artifact", "load_route_artifact",
    "lookup_next_hop",
]


ARTIFACT_VERSION = 1
TABLE_SEMANTICS = "adjacent-unsmoothed-server-A-star-cell-hop-v1"
ENDPOINT_POLICY = "cell-centre-only; arbitrary-world-endpoints-unsupported"
ALGORITHM = ("NavGrid.get_cell_path; closed-on-enqueue; flat-cell-cost; "
             "server-neighbour-order; SmoothPath-not-encoded")
MANIFEST_NAME = "manifest.json"
CELLS_NAME = "cells.npy"
NEXT_HOP_NAME = "next_hop.npy"

# A byte is deliberately used even though there are only nine valid values:
# K^2 is the dominant cost, and uint8 is the plan's measured size.  255 cannot
# be a direction and therefore represents the server's ``GetPath == null``.
NO_ROUTE = np.uint8(255)
STAY = np.uint8(0)
# Codes are stable artifact ABI, in the server's neighbour iteration order
# (dirY outer, dirX inner), with (0, 0) reserved for an already-arrived pair.
DIRECTION_OFFSETS = np.asarray(
    ((0, 0), (-1, -1), (0, -1), (1, -1),
     (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)), np.int8)
_OFFSET_TO_CODE = {tuple(map(int, xy)): i for i, xy in enumerate(DIRECTION_OFFSETS)}


class RouteStatus:
    """Result of a JAX table gather; nonzero values must remain observable."""

    READY = 0
    SOURCE_UNCOVERED = 1
    GOAL_UNCOVERED = 2
    NO_SERVER_ROUTE = 3


class RouteArtifactError(ValueError):
    """Artifact/source incompatibility.  Never fall back silently on this."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", np.float32(x)))[0]


def _array_sha256(a: np.ndarray) -> str:
    return _sha256_bytes(np.ascontiguousarray(a).tobytes())


@dataclass(frozen=True)
class RouteManifest:
    """JSON manifest that makes a route matrix self-identifying."""

    version: int
    map_id: int
    ngrid_sha256: str
    grid_shape: tuple[int, int]          # (rows, cols)
    cell_size_f32_bits: int
    pathfinding_radius_f32_bits: int
    coverage_count: int
    coverage_sha256: str
    table_shape: tuple[int, int]
    table_sha256: str
    table_dtype: str
    table_semantics: str
    endpoint_policy: str
    algorithm: str

    def to_json(self) -> str:
        # Compact, sorted JSON is deterministic and reviewable in a manifest
        # diff.  Arrays are converted explicitly because JSON has no tuples.
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":")) + "\n"

    @classmethod
    def from_json(cls, raw: str) -> "RouteManifest":
        try:
            d = json.loads(raw)
            d["grid_shape"] = tuple(d["grid_shape"])
            d["table_shape"] = tuple(d["table_shape"])
            return cls(**d)
        except (KeyError, TypeError, ValueError) as exc:
            raise RouteArtifactError("invalid route-artifact manifest") from exc


@dataclass(frozen=True)
class RouteArtifact:
    """A validated host-side artifact, ready to transfer as static JAX arrays."""

    manifest: RouteManifest
    cells: np.ndarray                    # (K,) sorted global flat cell ids, int32
    cell_to_coverage: np.ndarray         # (grid cells,) int32, -1 when absent
    next_hop: np.ndarray                 # (K,K) uint8 direction ABI above

    def as_jax(self) -> "JaxRouteTable":
        """Copy immutable lookup arrays to the current JAX device.

        This intentionally exposes only adjacent-hop gathers.  A future
        ``apply_orders(..., route_table=...)`` must reconstruct at most the
        measured waypoint bound, then smooth it against the same static
        terrain before writing ``LaneState.waypoints``.
        """
        import jax.numpy as jnp

        return JaxRouteTable(
            cell_to_coverage=jnp.asarray(self.cell_to_coverage),
            next_hop=jnp.asarray(self.next_hop),
        )


class JaxRouteTable(NamedTuple):
    """Static arrays accepted by :func:`lookup_next_hop` under ``jax.jit``."""

    cell_to_coverage: object
    next_hop: object


def lookup_next_hop(source_cell, goal_cell, table: JaxRouteTable):
    """Gather one raw A* hop and a non-silent status on device.

    ``source_cell`` and ``goal_cell`` are global flattened navgrid indices,
    not world coordinates.  The caller must map coordinates under the
    artifact's endpoint policy and preserve ``status`` in diagnostics.  In
    particular, it must not call an uncovered result an exact route and then
    quietly install the old two-point segment.
    """
    import jax.numpy as jnp

    n_grid = table.cell_to_coverage.shape[0]
    src_valid = (source_cell >= 0) & (source_cell < n_grid)
    goal_valid = (goal_cell >= 0) & (goal_cell < n_grid)
    src_safe = jnp.clip(source_cell, 0, n_grid - 1)
    goal_safe = jnp.clip(goal_cell, 0, n_grid - 1)
    src = table.cell_to_coverage[src_safe]
    goal = table.cell_to_coverage[goal_safe]
    src_covered = src_valid & (src >= 0)
    goal_covered = goal_valid & (goal >= 0)
    k = table.next_hop.shape[0]
    pair = jnp.clip(src, 0, k - 1) * k + jnp.clip(goal, 0, k - 1)
    hop = table.next_hop.reshape((-1,))[pair]
    status = jnp.where(~src_covered, RouteStatus.SOURCE_UNCOVERED,
             jnp.where(~goal_covered, RouteStatus.GOAL_UNCOVERED,
             jnp.where(hop == NO_ROUTE, RouteStatus.NO_SERVER_ROUTE,
                       RouteStatus.READY))).astype(jnp.int8)
    # The code is intentionally harmless whenever status is non-ready; only
    # the paired status decides whether a caller may consume it.
    return hop, status


def _normalise_coverage(coverage_cells: Sequence[int] | np.ndarray,
                        grid: NavGrid) -> np.ndarray:
    cells = np.asarray(coverage_cells)
    if cells.dtype == np.bool_:
        if cells.shape != grid.flags.shape:
            raise RouteArtifactError(
                f"boolean coverage has shape {cells.shape}, expected {grid.flags.shape}")
        cells = np.flatnonzero(cells)
    cells = np.asarray(cells, dtype=np.int32).reshape(-1)
    cells = np.unique(cells)
    if not len(cells):
        raise RouteArtifactError("route coverage must contain at least one cell")
    if cells[0] < 0 or cells[-1] >= grid.flags.size:
        raise RouteArtifactError("route coverage contains an out-of-grid cell")
    ys, xs = np.divmod(cells, grid.cell_count_x)
    if not all(grid.is_walkable_cell(int(x), int(y)) for x, y in zip(xs, ys)):
        raise RouteArtifactError("route coverage contains a non-walkable cell")
    return cells


def _route_hop_code(route_cells: Sequence[tuple[int, int]]) -> np.uint8:
    if len(route_cells) < 2:
        return STAY
    (x0, y0), (x1, y1) = route_cells[:2]
    try:
        return np.uint8(_OFFSET_TO_CODE[(x1 - x0, y1 - y0)])
    except KeyError as exc:  # a bake bug, never an alternate encoding
        raise RouteArtifactError(
            f"server cell path has non-adjacent first hop {(x0, y0)} -> {(x1, y1)}") from exc


def _validate_functional_graph(cells: np.ndarray, table: np.ndarray,
                               grid_shape: tuple[int, int]) -> None:
    """Reject a next-hop matrix whose repeated gathers can cycle.

    Server A* is closed-on-enqueue and need not have optimal substructure: the
    route baked from A to G does not prove that the independently baked route
    from its intermediate B to G will continue toward G.  Each goal column is
    therefore treated as a functional graph.  Every component must eventually
    reach that goal or a ``NO_ROUTE`` terminal, never a directed cycle.

    This is O(K²), one linear graph pass per goal, rather than tracing every
    source route independently (which would be O(K³)).
    """
    rows, cols = grid_shape
    k = len(cells)
    if table.shape != (k, k):
        raise RouteArtifactError("route table shape does not match coverage")
    cell_to_coverage = {int(cell): i for i, cell in enumerate(cells)}
    ys, xs = np.divmod(cells, cols)
    for goal in range(k):
        next_index = np.full(k, -1, np.int32)  # -1 is a permitted NO_ROUTE terminal
        for source, code in enumerate(table[:, goal]):
            code = int(code)
            if code == int(NO_ROUTE):
                continue
            if code == int(STAY):
                if source != goal:
                    raise RouteArtifactError(
                        f"non-goal cell {source} uses STAY for goal {goal}")
                next_index[source] = source
                continue
            if not 0 < code < len(DIRECTION_OFFSETS):
                raise RouteArtifactError(f"invalid route direction {code}")
            dx, dy = map(int, DIRECTION_OFFSETS[code])
            nx, ny = int(xs[source]) + dx, int(ys[source]) + dy
            if not (0 <= nx < cols and 0 <= ny < rows):
                raise RouteArtifactError("route direction leaves navgrid bounds")
            successor = cell_to_coverage.get(ny * cols + nx)
            if successor is None:
                raise RouteArtifactError("route direction leaves artifact coverage")
            next_index[source] = successor

        # A READY entry promises that repeated gathers can construct its whole
        # route.  Thus a NO_ROUTE terminal is valid only for that entry itself;
        # a READY chain may not run into another cell's NO_ROUTE entry halfway.
        # The goal's STAY self-loop is the sole allowed loop.
        outcome = np.zeros(k, np.int8)  # 0 unknown, 1 reaches goal, 2 NO_ROUTE
        outcome[next_index < 0] = 2
        outcome[goal] = 1
        for start in range(k):
            if outcome[start] == 2:  # this source's own table entry is NO_ROUTE
                continue
            current = start
            trail: list[int] = []
            visiting: set[int] = set()
            while current >= 0 and outcome[current] == 0:
                if current in visiting:
                    raise RouteArtifactError(
                        f"next-hop cycle for goal coverage index {goal}")
                visiting.add(current)
                trail.append(current)
                current = int(next_index[current])
            terminal = 2 if current < 0 else int(outcome[current])
            if terminal != 1:
                raise RouteArtifactError(
                    f"READY route terminates at NO_ROUTE for goal coverage index {goal}")
            for node in trail:
                outcome[node] = 1


def build_route_artifact(out_dir: Path | str, grid: NavGrid,
                         ngrid_path: Path | str,
                         coverage_cells: Sequence[int] | np.ndarray,
                         pathfinding_radius: float, *, map_id: int = 1,
                         max_expansions: int = 200_000) -> RouteManifest:
    """Bake a deterministic raw-next-hop matrix from the host server port.

    The builder refuses a coverage region unless every successful A* itinerary
    stays in it.  That closure rule prevents a deceptively compact corridor
    table from working for hop one and becoming uncovered on hop two.

    It writes exactly ``manifest.json``, ``cells.npy`` and ``next_hop.npy``.
    Callers must choose coverage explicitly; there is intentionally no default
    that could accidentally generate a multi-gigabyte full-map matrix.
    """
    out = Path(out_dir)
    ngrid_path = Path(ngrid_path)
    cells = _normalise_coverage(coverage_cells, grid)
    k = len(cells)
    for name in (MANIFEST_NAME, CELLS_NAME, NEXT_HOP_NAME):
        if (out / name).exists():
            raise FileExistsError(f"refusing to overwrite route artifact file {out / name}")
    out.mkdir(parents=True, exist_ok=True)

    cell_to_coverage = np.full(grid.flags.size, -1, np.int32)
    cell_to_coverage[cells] = np.arange(k, dtype=np.int32)
    table = np.full((k, k), NO_ROUTE, np.uint8)
    for source_i, source_flat in enumerate(cells):
        sy, sx = divmod(int(source_flat), grid.cell_count_x)
        source = grid.cell_center_world(sx, sy)
        for goal_i, goal_flat in enumerate(cells):
            if source_i == goal_i:
                table[source_i, goal_i] = STAY
                continue
            gy, gx = divmod(int(goal_flat), grid.cell_count_x)
            route = grid.get_cell_path(
                source, grid.cell_center_world(gx, gy), pathfinding_radius,
                max_expansions)
            if route is None:
                continue
            route_flat = np.fromiter(
                (y * grid.cell_count_x + x for x, y in route.cells), np.int32)
            if np.any(cell_to_coverage[route_flat] < 0):
                raise RouteArtifactError(
                    "coverage is not route-closed: "
                    f"{source_flat}->{goal_flat} leaves the selected cells")
            table[source_i, goal_i] = _route_hop_code(route.cells)

    manifest = RouteManifest(
        version=ARTIFACT_VERSION,
        map_id=int(map_id),
        ngrid_sha256=_sha256_file(ngrid_path),
        grid_shape=tuple(map(int, grid.flags.shape)),
        cell_size_f32_bits=_f32_bits(grid.cell_size),
        pathfinding_radius_f32_bits=_f32_bits(pathfinding_radius),
        coverage_count=int(k),
        coverage_sha256=_array_sha256(cells),
        table_shape=(int(k), int(k)),
        table_sha256=_array_sha256(table),
        table_dtype="uint8",
        table_semantics=TABLE_SEMANTICS,
        endpoint_policy=ENDPOINT_POLICY,
        algorithm=ALGORITHM,
    )
    _validate_functional_graph(cells, table, tuple(map(int, grid.flags.shape)))
    np.save(out / CELLS_NAME, cells, allow_pickle=False)
    np.save(out / NEXT_HOP_NAME, table, allow_pickle=False)
    (out / MANIFEST_NAME).write_text(manifest.to_json(), encoding="utf-8")
    return manifest


def load_route_artifact(path: Path | str, *, ngrid_path: Path | str = DEFAULT_NGRID,
                        pathfinding_radius: float, map_id: int = 1) -> RouteArtifact:
    """Load only an artifact compatible with this exact navgrid and radius."""
    root = Path(path)
    manifest = RouteManifest.from_json((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    if manifest.version != ARTIFACT_VERSION:
        raise RouteArtifactError(
            f"route artifact version {manifest.version}, expected {ARTIFACT_VERSION}")
    if manifest.map_id != map_id:
        raise RouteArtifactError(
            f"route artifact map id {manifest.map_id}, expected {map_id}")
    if (manifest.table_semantics != TABLE_SEMANTICS
            or manifest.endpoint_policy != ENDPOINT_POLICY
            or manifest.algorithm != ALGORITHM):
        raise RouteArtifactError("route artifact ABI semantics do not match this loader")
    if manifest.ngrid_sha256 != _sha256_file(Path(ngrid_path)):
        raise RouteArtifactError("route artifact navgrid hash does not match supplied content")
    if manifest.pathfinding_radius_f32_bits != _f32_bits(pathfinding_radius):
        raise RouteArtifactError("route artifact pathfinding radius does not match request")
    grid = NavGrid.load(ngrid_path)
    if (manifest.grid_shape != tuple(grid.flags.shape)
            or manifest.cell_size_f32_bits != _f32_bits(grid.cell_size)):
        raise RouteArtifactError("route artifact grid metadata does not match supplied navgrid")

    cells = np.load(root / CELLS_NAME, allow_pickle=False)
    table = np.load(root / NEXT_HOP_NAME, allow_pickle=False)
    if (cells.dtype != np.int32 or cells.ndim != 1
            or len(cells) != manifest.coverage_count
            or _array_sha256(cells) != manifest.coverage_sha256):
        raise RouteArtifactError("route artifact coverage array is invalid or tampered")
    if (table.dtype != np.uint8 or tuple(table.shape) != manifest.table_shape
            or table.shape != (len(cells), len(cells))
            or manifest.table_dtype != "uint8"
            or _array_sha256(table) != manifest.table_sha256):
        raise RouteArtifactError("route artifact next-hop array is invalid or tampered")
    if (not len(cells) or not np.all(cells[1:] > cells[:-1])
            or cells[0] < 0 or cells[-1] >= grid.flags.size):
        raise RouteArtifactError("route artifact coverage indices are not sorted unique grid cells")
    valid_codes = (table <= len(DIRECTION_OFFSETS) - 1) | (table == NO_ROUTE)
    if not np.all(valid_codes):
        raise RouteArtifactError("route artifact contains an invalid direction code")
    _validate_functional_graph(cells, table, tuple(map(int, grid.flags.shape)))
    cell_to_coverage = np.full(grid.flags.size, -1, np.int32)
    cell_to_coverage[cells] = np.arange(len(cells), dtype=np.int32)
    return RouteArtifact(manifest, cells, cell_to_coverage, table)


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coverage", type=Path, required=True,
                    help=".npy bool grid mask or sorted flat-cell ids; never defaults to full map")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--radius", type=float, required=True)
    ap.add_argument("--ngrid", type=Path, default=DEFAULT_NGRID)
    ap.add_argument("--map-id", type=int, default=1)
    args = ap.parse_args()
    grid = NavGrid.load(args.ngrid)
    coverage = np.load(args.coverage, allow_pickle=False)
    manifest = build_route_artifact(args.out, grid, args.ngrid, coverage,
                                    args.radius, map_id=args.map_id)
    print(manifest.to_json(), end="")


if __name__ == "__main__":
    _main()
