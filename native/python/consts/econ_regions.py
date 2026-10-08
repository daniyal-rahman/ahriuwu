"""Map11 region masks (map.regions.build_regions of the 26.19 navgrid) and the lane minion paths, for the
region queries of the DEATH phase (in_quest_lane, homeguard_flags) in native/src/champ/econ/economy.cpp."""

from functools import lru_cache


@lru_cache(maxsize=1)
def navgrid():
    """The world's navgrid (world.config.build_config's default map)."""
    from pathlib import Path

    from lanerl_jax.modern.data.navgrid import load_patch_map
    from lanerl_jax.modern.world.config import DEFAULT_MAP
    return load_patch_map(Path(DEFAULT_MAP))[0]


def consts():
    import numpy as np

    from lanerl_jax.modern.map import regions as REG
    from lanerl_jax.modern.map.lanes import LANE_PATHS
    r = REG.build_regions(navgrid())
    main = np.asarray(r.main)
    paths = np.asarray(LANE_PATHS, np.float32)
    return {"econ.regions.main": main, "econ.regions.side": np.asarray(r.side),
            "econ.regions.geom": [main.shape[0], main.shape[1], r.cell_size, r.min_x, r.min_z],
            "econ.regions.paths": paths, "econ.regions.paths_shape": list(paths.shape)}
