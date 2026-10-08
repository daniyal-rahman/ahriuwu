"""Constants of items.effects.mage (native/src/champ/items/mage.cpp)."""
import numpy as np

from ._items_b import module_consts

NAMES = ("EPS", "ULT_ATTRIBUTION_WINDOW", "MALIGNANCE_TICK", "STORM_BUCKET", "STORM_SLOTS", "STORM_AOE",
         "HORIZON_FOCUS_RADIUS")


def consts():
    from lanerl_jax.modern.items.effects import mage as M
    dv = M.dv
    ramps = tuple((i, k) for i in (M.GUISE, M.LIANDRY)                    # _guise_amp loop
                  for k in ("DamageIncreasePerSecond", "DamageIncreaseMax", "BuffCounterDuration"))
    out = module_consts("mage", NAMES, ramps)
    # Python-float expressions JAX folds before converting to float32.
    p = "items.mage."
    for item, key in ((M.GUISE, "GUISE_NMAX"), (M.LIANDRY, "LIANDRY_NMAX")):   # _stacks n_max (jnp.round)
        out[p + key] = np.round(np.float32(dv(item, "DamageIncreaseMax") / dv(item, "DamageIncreasePerSecond")))
    out[p + "RIFT_NMAX"] = np.round(np.float32(dv(M.RIFTMAKER, "EternityDamageIncreaseMax")
                                               / dv(M.RIFTMAKER, "EternityDamageIncreasePerSecond")))
    tf = dv(M.ASHES, "TickFrequency")
    out[p + "ASHES_PER"] = dv(M.ASHES, "BurnFlatDamagePerSecond") * tf
    out[p + "ASHES_MONSTER"] = dv(M.ASHES, "MonsterDamageBonus") * tf
    out[p + "LUDEN_N_EXTRA"] = int(dv(M.LUDEN, "MaxCharges")) - 1
    return out
