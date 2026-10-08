"""items.effects.starters constants (Doran's Shield, plus the values of always-emitted padded packets/grants)."""


def consts():
    from lanerl_jax.modern.items.effects import starters as M
    dv = M.dv
    p = "items.starters."
    return {p + "ds_max_regen": dv(M.DORANS_SHIELD, "MaxRegenAmount"),
            p + "ds_max_range_regen": dv(M.DORANS_SHIELD, "MaxRangeRegenAmount"),
            p + "ds_regen_duration": dv(M.DORANS_SHIELD, "RegenDuration"),
            p + "ds_range_regen_mult": dv(M.DORANS_SHIELD, "RangeRegenMult"),
            p + "ds_minion_bonus": dv(M.DORANS_SHIELD, "BonusDamageToMinions"),
            p + "seraph_shield_duration": dv(M.SERAPHS, "ShieldDuration"),
            p + "fimbul_shield_duration": dv(M.FIMBULWINTER, "ShieldDuration"),
            p + "muramana_onhit": M.MURAMANA_ONHIT,
            p + "muramana_ability_melee": M.MURAMANA_ABILITY_MELEE,
            p + "muramana_ability_ranged": M.MURAMANA_ABILITY_RANGED}
