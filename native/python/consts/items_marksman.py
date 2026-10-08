"""Constants of items.effects.marksman (native/src/champ/items/marksman.cpp)."""
from ._items_b import module_consts

NAMES = ("RECURVE_DMG", "GUINSOO_DMG", "GUINSOO_AS", "GUINSOO_MAX", "GUINSOO_DUR", "GUINSOO_PHANTOM_MAX",
         "KRAKEN_COUNT", "KRAKEN_DUR", "KRAKEN_MAX_AMP", "KRAKEN_RANGED", "YT_DUR", "YT_CD", "YT_AS", "YT_CRIT_MAX",
         "YT_AA_CDR", "YT_CRIT_CDR", "YT_CRIT_PER", "YT_RANGED", "FH_HASTE", "FH_CD", "FH_DUR", "FH_AS", "FH_N",
         "FH_CRIT", "FH_TRUE", "LDR_MAX", "LDR_HP", "HEX_AMP", "HEX_RANGE", "HEX_EXTRA", "HEX_DUR", "TAKEDOWN_WINDOW",
         "RUNAAN_RATIO", "RUNAAN_BOLTS_RANGED", "RUNAAN_BOLTS_MELEE", "STATIKK_BONUS", "STATIKK_CHAMP",
         "STATIKK_OTHER", "STATIKK_RANGE", "STATIKK_MAX_BOUNCES", "RFC_DMG", "STORM_DMG", "STORM_MS", "STORM_DUR",
         "VOLT_PCT_M", "VOLT_PCT_R", "VOLT_LETH_M", "VOLT_LETH_R", "VOLT_DUR", "VOLT_CAP", "SLING_DMG", "SLING_CD",
         "SLING_ATTACK_CDR", "COLLECTOR_THRESHOLD", "COLLECTOR_GOLD", "YOUMUU_OOC_MS", "YOUMUU_TIMER",
         "YOUMUU_RANGED", "HUBRIS_BASE", "HUBRIS_PER", "HUBRIS_DUR", "AXIOM_BASE", "AXIOM_PER_LETHALITY",
         "SERPENT_DUR", "ENERGY_MAX", "ENERGY_PER_ATTACK", "ENERGY_UNITS_PER_STACK", "ICHOR_DURATION")


def consts():
    from lanerl_jax.modern.items.effects import marksman as M
    out = module_consts("marksman", NAMES)
    # Python-float expressions JAX folds before converting to float32.
    p = "items.marksman."
    out[p + "KRAKEN_COUNT_M1"] = M.KRAKEN_COUNT - 1
    out[p + "KRAKEN_AMP_M1"] = M.KRAKEN_MAX_AMP - 1.0
    out[p + "RUNAAN_REACH"] = M.RUNAAN_ATTACK_RANGE + M.RUNAAN_EXTRA
    return out
