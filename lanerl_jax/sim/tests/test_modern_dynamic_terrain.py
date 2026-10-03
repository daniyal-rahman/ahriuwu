"""Structure footprints and dynamic walkable masks on the pinned 26.19 navgrid."""
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.sim import modern_dynamic_terrain as DT
from lanerl_jax.sim import modern_world as MW
from lanerl_jax.sim.modern_terrain import is_walkable

MAP = Path("/mnt/nfs/shared/modern-world-map-research/grid-26.19-base")
pytestmark = pytest.mark.skipif(not MAP.exists(), reason="pinned 26.19 navgrid not mounted")


@pytest.fixture(scope="module")
def setup():
    from lanerl_jax.data.modern_map import load_patch_map
    grid, _ = load_patch_map(MAP)
    geo = json.loads((MW.DATA / "geometry.json").read_text())
    kinds, xs, ys = [1, 1], [394.0, 14340.0], [461.0, 14391.0]
    for t in geo["turrets"]:
        kinds.append(3); xs.append(t["position"][0]); ys.append(t["position"][1])
    for p in MW.INHIBITORS.values():
        kinds.append(4); xs.append(p[0]); ys.append(p[1])
    kinds += [5, 5]; xs += [1715.8, 12998.0]; ys += [1790.3, 12950.0]
    fp = DT.build_footprints(grid, kinds, xs, ys)
    terrain = tuple(grid.as_jax(t) for t in (0, 1))
    return grid, np.asarray(kinds), np.asarray(xs), np.asarray(ys), fp, terrain


def test_every_structure_owns_a_pad_of_the_right_size(setup):
    grid, kinds, xs, ys, fp, _ = setup
    cells = np.asarray(fp.cells)
    assert (cells[kinds == 1] == 0).all()
    assert ((cells[kinds == 3] >= 18) & (cells[kinds == 3] <= 21)).all()        # 5x5 turret pads
    assert ((cells[kinds == 4] >= 44) & (cells[kinds == 4] <= 61)).all()        # inhibitor pads
    assert ((cells[kinds == 5] >= 150) & (cells[kinds == 5] <= 160)).all()      # Nexus pads
    owned = np.asarray(fp.owner) >= 0
    assert (((np.asarray(grid.flags) & DT.STRUCTURE_FLAG) != 0) | ~owned).all()


def test_default_policy_keeps_destroyed_structures_blocking(setup):
    grid, kinds, xs, ys, fp, terrain = setup
    alive = jnp.zeros(len(kinds), bool)                                          # everything destroyed
    masks = DT.walkable_masks(terrain, fp, alive, DT.release_mask(kinds))
    for a, b in zip(masks, terrain):
        assert bool(jnp.all(a.walkable == b.walkable))


def test_released_inhibitor_pad_opens_and_closes_on_respawn(setup):
    grid, kinds, xs, ys, fp, terrain = setup
    k = int(np.nonzero(kinds == 4)[0][0])
    x, y = float(xs[k]), float(ys[k])
    release = DT.release_mask(kinds, {4: True})
    assert not bool(is_walkable(x, y, 30.0, terrain[0]))
    dead = jnp.ones(len(kinds), bool).at[k].set(False)
    opened = DT.walkable_masks(terrain, fp, dead, release)
    assert bool(is_walkable(x, y, 30.0, opened[0])) and bool(is_walkable(x, y, 30.0, opened[1]))
    closed = DT.walkable_masks(terrain, fp, jnp.ones(len(kinds), bool), release)
    assert not bool(is_walkable(x, y, 30.0, closed[0]))
    # A unit left on the pad when it respawns is pushed to the nearest walkable point.
    ex, ey = DT.eject(jnp.asarray([x]), jnp.asarray([y]), jnp.asarray([0]), jnp.asarray([65.0]), closed)
    assert bool(is_walkable(ex[0], ey[0], 65.0, closed[0]))
    assert 0.0 < float(np.hypot(float(ex[0]) - x, float(ey[0]) - y)) <= 450.0
