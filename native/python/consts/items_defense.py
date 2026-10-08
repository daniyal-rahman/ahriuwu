"""Constants of items.effects.defense (native/src/champ/items/defense.cpp)."""
from ._items_b import module_consts

NAMES = ("GA_HP", "GA_DELAY", "GA_MANA", "GA_COOLDOWN", "CHAMP_COMBAT_WINDOW", "KAENIC_DURATION", "KAENIC_TAG", "EPS")
# Data values of the client calculations the port evaluates (Thornmail / Bramble TotalDamage, Heartsteel
# DamageProcCalc / ProcHealthGain, Unending Despair DrainCalc).
CALC_DV = ((3076, "BaseDamage"), (3075, "BaseDamage"), (3075, "BonusArmorDamageRatio"), (3084, "BaseDamage"),
           (3084, "HPRatio"), (3084, "DamageToMaxHealthRatio"), (2502, "BonusHealthDrainPercentage"))


def consts():
    from lanerl_jax.modern.items.effects import defense as D
    dv = D.dv
    cooldowns = tuple((i, "Cooldown") for i in (*D.LIFELINE, *D.ANNUL))     # _pick tables of Lifeline / Annul
    out = module_consts("defense", NAMES, CALC_DV + cooldowns)
    # Python-float expressions JAX folds before converting to float32.
    p = "items.defense."
    out[p + "IMMO_PERIOD"] = 1.0 / dv(D.SUNFIRE, "TicksPerSecond")
    demolish = dv(D.HEARTSTEEL, "TrackerTickRate") * dv(D.HEARTSTEEL, "NumTicksToTrigger")
    out[p + "HS_DEMOLISH"] = demolish
    out[p + "HS_DEMOLISH_EPS"] = demolish - D.EPS
    out[p + "KAENIC_OOC_EPS"] = dv(D.KAENIC, "OutOfCombatDuration") - D.EPS
    out[p + "JAK_EPS"] = dv(D.JAKSHO, "MaxStacks") - D.EPS
    out[p + "HOLLOW_WINDOW_EPS"] = dv(D.HOLLOW, "TakedownWindow") + D.EPS
    out[p + "RANDUIN_CRIT_MULT"] = 1.0 - dv(D.RANDUINS, "PercentCritDamageReduction")
    out[p + "FH_SLOW"] = abs(dv(D.FROZEN_HEART, "ASPDSlow"))
    out[p + "IMMO_MULT"] = [v for iid in (D.BAMIS, D.SUNFIRE, D.HOLLOW) for v in D._IMMO[iid]]   # (minion, monster)
    return out
