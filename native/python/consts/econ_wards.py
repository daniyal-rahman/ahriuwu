"""Ward world data and rune values for native/src/champ/econ/wards.cpp: the WardGrid (world data, built by
wards.ward_grid from the 26.19 navgrid) and the Deep Ward / Sixth Sense effect amounts."""


def consts():
    from .econ_regions import navgrid
    import numpy as np

    from lanerl_jax.modern import vision as MV
    from lanerl_jax.modern import wards as WD
    from lanerl_jax.modern.runes.catalog import ea
    from lanerl_jax.modern.runes.effects import domination as DOM
    g = WD.ward_grid(navgrid())
    walk = np.asarray(g.walkable)
    dw, ss = DOM.DEEP_WARD, DOM.SIXTH_SENSE
    lo, hi = ea(dw, "TTTrinketDurationIncreaseMin"), ea(dw, "TTTrinketDurationIncreaseMax")
    return {
        "econ.wards.walkable": walk, "econ.wards.region": np.asarray(g.region),
        "econ.wards.geom": [walk.shape[0], walk.shape[1], g.cell_size, g.min_x, g.min_z],
        "econ.wards.deep_level": ea(dw, "LevelThreshold"), "econ.wards.deep_hp": ea(dw, "ExtraHealth"),
        "econ.wards.deep_dur_start": lo, "econ.wards.deep_dur_span": hi - lo,      # lin(): (end - start) in Python
        "econ.wards.sixth_range2": ea(ss, DOM.SIXTH_SENSE_RANGE_KEY) * ea(ss, DOM.SIXTH_SENSE_RANGE_KEY),
        "econ.wards.sixth_level": ea(ss, "LevelThreshold"), "econ.wards.sixth_cd": ea(ss, "MeleeItemCalcValue"),
        "econ.wards.sixth_reveal": ea(ss, "RevealDuration"),
        # vision.sight_radius (vision_kwargs)
        "econ.wards.sight": [MV.CHAMPION_SIGHT, MV.MINION_SIGHT, MV.SUPER_MINION_SIGHT, MV.TURRET_SIGHT,
                             MV.NEXUS_SIGHT, MV.INHIBITOR_SIGHT, MV.WARD_SIGHT, MV.FARSIGHT_SIGHT, MV.SUPER,
                             MV.FARSIGHT_SUB],
    }
