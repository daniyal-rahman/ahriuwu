"""Bake a conservative Map11 navigation graph and next-hop table (run full bakes through Slurm).

    python -m lanerl_jax.modern.data.routes --map GRID_DIR --out DIR

Simulation routes, not Riot's pathfinder. Every edge is checked against the pinned collision grid; the
100-unit spacing may reject narrow passages, and there is never an unchecked straight-line fallback.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from ..map.pathing import FlowRoutes
from .navgrid import load_patch_map

SCHEMA = "map11-flow-v1"
ARRAYS = ("points", "cells", "next")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def clear_segment(grid, start, end, radius, team=None):
    """Sampled capsule test; samples every <= 10 u, inflated by half a step to cover the gaps."""
    length = np.linalg.norm(np.asarray(end) - start)
    count = max(1, int(np.ceil(length / 10)))
    margin = length / count / 2
    return all(grid.is_walkable(float(p[0]), float(p[1]), radius=radius + margin, team=team)
               for p in np.linspace(start, end, count + 1))


def build_routes(grid, out, *, spacing=100., radius=35.):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    xs = np.arange(grid.min_bounds[0] + spacing / 2, grid.max_bounds[0], spacing)
    zs = np.arange(grid.min_bounds[2] + spacing / 2, grid.max_bounds[2], spacing)
    # One gate-closed graph for both teams: runtime direct segments may still use their own gates.
    cells = np.full((len(zs), len(xs)), -1, np.int32)
    points = []
    for zi, z in enumerate(zs):
        for xi, x in enumerate(xs):
            if grid.is_walkable(x, z, radius=radius):
                cells[zi, xi] = len(points)
                points.append((x, z))
    points = np.asarray(points, np.float32)
    if len(points) >= 32767:
        raise ValueError("graph too large for int16 routes")
    rows, cols, weights = [], [], []
    for z, x in np.argwhere(cells >= 0):
        i = cells[z, x]
        for dz, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
            zz, xx = z + dz, x + dx
            if not (0 <= zz < len(zs) and 0 <= xx < len(xs)):
                continue
            j = cells[zz, xx]
            if j >= 0 and clear_segment(grid, points[i], points[j], radius):
                dist = float(np.linalg.norm(points[i] - points[j]))
                rows += [i, j]
                cols += [j, i]
                weights += [dist, dist]
    n = len(points)
    graph = csr_matrix((weights, (rows, cols)), shape=(n, n))
    nxt = np.lib.format.open_memmap(out / "next.npy", mode="w+", dtype=np.int16, shape=(n, n))
    for start in range(0, n, 64):
        stop = min(start + 64, n)
        _, pred = dijkstra(graph, directed=False, indices=np.arange(start, stop), return_predecessors=True)
        nxt[start:stop] = np.where(pred < 0, -1, pred).astype(np.int16)
    nxt.flush()
    np.save(out / "points.npy", points)
    np.save(out / "cells.npy", cells)
    meta = {"schema": SCHEMA, "patch": "26.19", "spacing": spacing, "radius": radius,
            "min_x": grid.min_bounds[0], "min_z": grid.min_bounds[2],
            "grid_flags_sha256": _sha(grid.flags.tobytes()),
            "files": {f"{a}.npy": _sha((out / f"{a}.npy").read_bytes()) for a in ARRAYS},
            "nodes": n, "edges": len(rows),
            "limitations": ["100-unit navigation graph", "base gates closed in graph", "static terrain only"]}
    (out / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def load_routes(path, grid):
    import jax.numpy as jnp
    path = Path(path)
    m = json.loads((path / "manifest.json").read_text())
    if m["schema"] != SCHEMA or m["patch"] != "26.19":
        raise ValueError("route profile mismatch")
    if m["grid_flags_sha256"] != _sha(grid.flags.tobytes()):
        raise ValueError("route terrain mismatch")
    arrays = {}
    for name in ARRAYS:
        file = path / f"{name}.npy"
        if _sha(file.read_bytes()) != m["files"][file.name]:
            raise ValueError("route checksum mismatch")
        arrays[name] = jnp.asarray(np.load(file, allow_pickle=False))
    return FlowRoutes(arrays["points"], arrays["cells"], arrays["next"], m["spacing"], m["min_x"], m["min_z"],
                      m["radius"]), m


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--map", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    grid, _ = load_patch_map(a.map)
    print(json.dumps(build_routes(grid, a.out), indent=2))


if __name__ == "__main__":
    main()
