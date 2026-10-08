"""Constants of items.effects.fighter (native/src/champ/items/fighter.cpp)."""
from ._items_b import module_consts

NAMES = ("TYRANNY", "RETRIBUTION", "RETRIBUTION_FULL", "FAMINE_BASE", "FAMINE_MELEE", "FAMINE_RANGED",
         "TAKEDOWN_WINDOW", "SHAPED_BASE", "SHAPED_LETH", "SABOTAGE_BASE", "SABOTAGE_LETH", "CARVE_PER_STACK",
         "CARVE_MAX", "CARVE_DURATION", "CARVE_ICD", "DD_MELEE", "DD_RANGED", "DD_BLEED", "DD_BUCKET", "DD_SLOTS",
         "HULL_STACKS", "HULL_RANGED", "DMP_MAX", "DMP_RATE", "DMP_FLAT_FULL", "WITS_DAMAGE", "TERM_BASE", "TERM_BAD",
         "TERM_AP", "TERM_RES_L1", "TERM_RES_STEPS", "TERM_LIGHT_MAX", "TERM_DARK_PER", "TERM_DARK_MAX")


def consts():
    from lanerl_jax.modern.items.effects import fighter as F
    out = module_consts("fighter", NAMES)
    # Python-float expressions JAX folds before converting to float32.
    out["items.fighter.CARVE_ICD_EPS"] = F.CARVE_ICD - 1e-6
    out["items.fighter.HULL_STACKS_M1"] = F.HULL_STACKS - 1
    return out
