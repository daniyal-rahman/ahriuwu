"""The route-artifact ABI is deliberately cheap to test without a real bake."""
from __future__ import annotations

import hashlib
import json

import jax
import numpy as np
import pytest

from lanerl_jax.data.navgrid import DEFAULT_NGRID, NavGrid
from lanerl_jax.data.route_artifact import (
    DIRECTION_OFFSETS,
    NO_ROUTE,
    RouteArtifactError,
    RouteStatus,
    build_route_artifact,
    load_route_artifact,
    lookup_next_hop,
)


def _open_grid() -> NavGrid:
    return NavGrid(
        flags=np.zeros((3, 3), np.uint16), cell_size=50.0,
        min_grid=(0.0, 0.0, 0.0), max_grid=(150.0, 0.0, 150.0))


def test_builder_encodes_server_raw_adjacent_hops_deterministically(tmp_path):
    """The bake consumes pre-SmoothPath neighbours, not smoothed jumps."""
    grid = _open_grid()
    ngrid = tmp_path / "synthetic.ngrid"
    ngrid.write_bytes(b"synthetic grid identity")
    coverage = np.arange(9, dtype=np.int32)

    first = build_route_artifact(tmp_path / "one", grid, ngrid, coverage, 0.0)
    second = build_route_artifact(tmp_path / "two", grid, ngrid, coverage, 0.0)
    one = np.load(tmp_path / "one" / "next_hop.npy", allow_pickle=False)
    two = np.load(tmp_path / "two" / "next_hop.npy", allow_pickle=False)

    # Cell (0,0) reaches (2,2) through raw (1,1); SmoothPath would be allowed
    # to collapse a route, which is why the artifact stores this hop instead.
    assert one[0, 8] == 8
    assert tuple(DIRECTION_OFFSETS[one[0, 8]]) == (1, 1)
    assert first.to_json() == second.to_json()
    np.testing.assert_array_equal(one, two)


@pytest.mark.skipif(not DEFAULT_NGRID.exists(), reason="vendored Map1 navgrid not available")
def test_loader_fails_closed_and_jax_gather_reports_uncovered_cells(tmp_path):
    """A nine-cell Map1 sample checks bake, closure, loader and JAX ABI."""
    grid = NavGrid.load()
    # A real Map1 3x3 open patch. It is intentionally tiny (81 host A* calls),
    # but all its radius-35 routes are coverage-closed, so every matrix entry
    # can be compared with the server-port itinerary rather than only tested
    # for deserialisation.
    coverage = np.asarray(
        [y * grid.cell_count_x + x for y in range(3, 6) for x in range(5, 8)],
        np.int32)
    build_route_artifact(tmp_path / "artifact", grid, DEFAULT_NGRID,
                         coverage, 35.0)
    artifact = load_route_artifact(tmp_path / "artifact", pathfinding_radius=35.0)
    covered = set(map(int, artifact.cells))
    for source_i, source_flat in enumerate(artifact.cells):
        sy, sx = divmod(int(source_flat), grid.cell_count_x)
        source = grid.cell_center_world(sx, sy)
        for goal_i, goal_flat in enumerate(artifact.cells):
            gy, gx = divmod(int(goal_flat), grid.cell_count_x)
            route = grid.get_cell_path(source, grid.cell_center_world(gx, gy), 35.0)
            if source_i == goal_i:
                assert int(artifact.next_hop[source_i, goal_i]) == 0
                continue
            assert route is not None
            assert {y * grid.cell_count_x + x for x, y in route.cells} <= covered
            (x0, y0), (x1, y1) = route.cells[:2]
            expected = np.flatnonzero(np.all(
                DIRECTION_OFFSETS == (x1 - x0, y1 - y0), axis=1))[0]
            assert int(artifact.next_hop[source_i, goal_i]) == int(expected)

    table = artifact.as_jax()
    hop, status = jax.jit(lambda a, b: lookup_next_hop(a, b, table))(
        artifact.cells[0], artifact.cells[-1])
    assert int(hop) == int(artifact.next_hop[0, -1])
    assert int(status) == RouteStatus.READY

    _, uncovered = jax.jit(lambda a, b: lookup_next_hop(a, b, table))(
        artifact.cells[0], -1)
    assert int(uncovered) == RouteStatus.GOAL_UNCOVERED
    with pytest.raises(RouteArtifactError, match="radius"):
        load_route_artifact(tmp_path / "artifact", pathfinding_radius=36.0)
    other = tmp_path / "different-content.ngrid"
    other.write_bytes(b"not the Map1 navgrid")
    with pytest.raises(RouteArtifactError, match="hash"):
        load_route_artifact(tmp_path / "artifact", ngrid_path=other,
                            pathfinding_radius=35.0)

    manifest_path = tmp_path / "artifact" / "manifest.json"
    original = manifest_path.read_text()
    bad_semantics = json.loads(original)
    bad_semantics["table_semantics"] = "unknown-routing-ABI"
    manifest_path.write_text(json.dumps(bad_semantics))
    with pytest.raises(RouteArtifactError, match="semantics"):
        load_route_artifact(tmp_path / "artifact", pathfinding_radius=35.0)
    manifest_path.write_text(original)
    bad_map = json.loads(original)
    bad_map["map_id"] = 2
    manifest_path.write_text(json.dumps(bad_map))
    with pytest.raises(RouteArtifactError, match="map id"):
        load_route_artifact(tmp_path / "artifact", pathfinding_radius=35.0)
    manifest_path.write_text(original)

    # A READY first hop cannot be allowed to reach another source's NO_ROUTE:
    # otherwise lookup says READY but fixed-shape reconstruction fails midway.
    broken = artifact.next_hop.copy()
    broken[0, 2] = 5          # (5,3) -> (6,3)
    broken[1, 2] = NO_ROUTE   # intermediate route is not available
    np.save(tmp_path / "artifact" / "next_hop.npy", broken, allow_pickle=False)
    broken_manifest = json.loads(original)
    broken_manifest["table_sha256"] = hashlib.sha256(
        np.ascontiguousarray(broken).tobytes()).hexdigest()
    manifest_path.write_text(json.dumps(broken_manifest))
    with pytest.raises(RouteArtifactError, match="READY route terminates"):
        load_route_artifact(tmp_path / "artifact", pathfinding_radius=35.0)

    # A valid checksum cannot legitimise a per-goal next-hop cycle: repeated
    # JAX gathers would never finish reconstructing this route.
    cyclic = artifact.next_hop.copy()
    cyclic[0, 2] = 5  # (5,3) -> (6,3)
    cyclic[1, 2] = 4  # (6,3) -> (5,3), instead of progressing to goal index 2
    np.save(tmp_path / "artifact" / "next_hop.npy", cyclic, allow_pickle=False)
    cycle_manifest = json.loads(original)
    cycle_manifest["table_sha256"] = hashlib.sha256(
        np.ascontiguousarray(cyclic).tobytes()).hexdigest()
    manifest_path.write_text(json.dumps(cycle_manifest))
    with pytest.raises(RouteArtifactError, match="cycle"):
        load_route_artifact(tmp_path / "artifact", pathfinding_radius=35.0)
