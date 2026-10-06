"""Map11 region masks: lanes, jungle, river, bases, quest lane, Homeguard endpoint."""
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.modern.core.types import KIND_CHAMPION, KIND_INHIBITOR, KIND_MINION, KIND_TURRET, WorldUnits
from lanerl_jax.modern.map import regions as R

MAP = Path("/mnt/nfs/shared/modern-world-map-research/grid-26.19-base")
pytestmark = pytest.mark.skipif(not MAP.exists(), reason="pinned 26.19 navgrid not mounted")


@pytest.fixture(scope="module")
def regions():
    from lanerl_jax.modern.data.navgrid import load_patch_map
    grid, _ = load_patch_map(MAP)
    return R.build_regions(grid)


POINTS = {  # name: (x, y, main region, side)
    "blue fountain": (394, 461, R.SPAWN, 0), "red fountain": (14340, 14391, R.SPAWN, 1),
    "blue base": (2500, 2500, R.BASE, 0), "red base": (12300, 12300, R.BASE, 1),
    "top lane (blue half)": (1300, 8000, R.TOP_LANE, 0), "top lane (red half)": (8000, 13600, R.TOP_LANE, 1),
    "mid": (7400, 7400, R.MID_LANE, None), "bot lane": (10000, 1100, R.BOT_LANE, 0),
    "blue top jungle": (3800, 7900, R.TOP_JUNGLE, 0), "dragon pit": (9870, 4409, R.BOT_RIVER, None),
    "baron pit": (4980, 10461, R.TOP_RIVER, None),
}


def test_known_points_land_in_their_regions(regions):
    for name, (x, y, code, side) in POINTS.items():
        assert int(R.region_of(x, y, regions)) == code, name
        if side is not None:
            assert int(R.side_of(x, y, regions)) == side, name
    assert int(R.region_of(-500., 100., regions)) == R.OUTSIDE


def test_quest_lane_is_the_whole_lane_outside_both_bases(regions):
    xs = jnp.asarray([1300., 8000., 2000., 2500., 7400., 3800.])
    ys = jnp.asarray([8000., 13600., 12500., 2500., 7400., 7900.])
    assert np.asarray(R.in_quest_lane(xs, ys, 2, regions)).tolist() == [True, True, True, False, False, False]
    assert np.asarray(R.lane_of(xs, ys, regions)).tolist() == [2, 2, 2, -1, 1, -1]


def test_jungle_river_base_masks(regions):
    assert bool(R.in_jungle(3800., 7900., regions)) and not bool(R.in_jungle(1300., 8000., regions))
    assert bool(R.in_river(9870., 4409., regions)) and not bool(R.in_river(7400., 7400., regions))
    assert bool(R.in_base(2500., 2500., 0, regions)) and not bool(R.in_base(2500., 2500., 1, regions))
    assert bool(R.in_base(394., 461., 0, regions))                          # fountain platform


def test_lane_progress_increases_toward_the_enemy():
    blue = [float(R.lane_progress(1300., y, 0, 2)) for y in (5000., 8000., 11000.)]
    red = [float(R.lane_progress(1300., y, 1, 2)) for y in (5000., 8000., 11000.)]
    assert blue[0] < blue[1] < blue[2] and red[0] > red[1] > red[2]


def _units(rows):
    a = np.asarray(rows, np.float64)
    n = len(rows)
    z = jnp.zeros((n,), jnp.float32)
    return WorldUnits(jnp.asarray(a[:, 0], jnp.int32), jnp.asarray(a[:, 1], jnp.int32), jnp.asarray(a[:, 2], jnp.int32),
                      jnp.asarray(a[:, 5] > 0), jnp.ones((n,), bool), jnp.asarray(a[:, 3], jnp.float32),
                      jnp.asarray(a[:, 4], jnp.float32), z, z, z, z, z, z, z, z, z, jnp.arange(n, dtype=jnp.int32), z)


def test_homeguard_endpoint_before_outer_turret_then_tracks_minions(regions):
    # Blue top: outer (981, 10441), inner (1512, 6699), inhibitor turret (1169, 4287), inhibitor (1171.6, 3569.7).
    rows = [(KIND_CHAMPION, 0, 0, 1250., 10100., 1), (KIND_TURRET, 0, 0, 981., 10441., 1),
            (KIND_TURRET, 1, 0, 1512., 6699., 1), (KIND_TURRET, 2, 0, 1169., 4287., 1),
            (KIND_INHIBITOR, 0, 0, 1171.6, 3569.7, 1), (KIND_MINION, 0, 0, 3500., 13450., 1)]
    u = _units(rows)
    slane = jnp.asarray([-1, 2, 2, 2, 2, -1])
    mlane = jnp.asarray([-1, -1, -1, -1, -1, 2])
    outer_p = float(R.lane_progress(981., 10441., 0, 2))
    end = R.homeguard_endpoint(jnp.asarray([0]), jnp.asarray([2]), 300., u, slane, mlane)
    assert float(end[0]) == pytest.approx(outer_p - 500., abs=1.)
    reached, jungle = R.homeguard_flags(jnp.asarray([1250.]), jnp.asarray([10100.]), jnp.asarray([0]), 300., u,
                                        slane, mlane, regions)
    assert bool(reached[0]) and not bool(jungle[0])
    reached, _ = R.homeguard_flags(jnp.asarray([1300.]), jnp.asarray([8000.]), jnp.asarray([0]), 300., u, slane,
                                   mlane, regions)
    assert not bool(reached[0])
    # After 14:00 the endpoint moves to 2000 before the furthest allied minion.
    late = R.homeguard_endpoint(jnp.asarray([0]), jnp.asarray([2]), 900., u, slane, mlane)
    assert float(late[0]) == pytest.approx(float(R.lane_progress(3500., 13450., 0, 2)) - 2000., abs=1.)
    # Jungle mask for Homeguard.
    _, jungle = R.homeguard_flags(jnp.asarray([3800.]), jnp.asarray([7900.]), jnp.asarray([0]), 300., u, slane,
                                  mlane, regions)
    assert bool(jungle[0])
