"""runes.effects.domination constants: ``ea`` values as ``runes.domination.<perk>.<name>``, ``lin`` pairs
``[start, end - start]``."""
from .runes_precision import lin_pair


def ea_values(prefix, perk, names):
    from lanerl_jax.modern.runes.catalog import ea
    return {f"{prefix}.{perk}.{n}": ea(perk, n) for n in names}


def consts():
    from lanerl_jax.modern.runes.catalog import ea
    from lanerl_jax.modern.runes.effects import domination as M
    P = "runes.domination"
    out = {}
    out.update(ea_values(P, M.ELECTROCUTE, ("BonusADRatio", "APRatio", "WindowDuration", "Cooldown")))
    out.update(ea_values(P, M.DARK_HARVEST, ("HarvestThreshold", "BaseDamage", "DamagePerSoulEssence", "ADRatio",
                                             "APRatio", "Cooldown", "CooldownResetValue")))
    out.update(ea_values(P, M.HAIL_OF_BLADES, ("Duration", "NumHits", "MaxBonusHits", "Cooldown", "BonusADRatio",
                                               "APRatio", "ASBoost", "ASBoostRanged")))
    out.update(ea_values(P, M.CHEAP_SHOT, ("Cooldown",)))
    out.update(ea_values(P, M.SUDDEN_IMPACT, ("Cooldown", "ArmedDuration")))
    out.update(ea_values(P, M.TASTE_OF_BLOOD, ("ADRatio", "APRatio", "Cooldown")))
    out.update(ea_values(P, M.TREASURE_HUNTER, ("BaseGoldAmount", "GoldGrowth")))
    out.update(ea_values(P, M.GRISLY_MEMENTOS, ("MaxStacks", "TrinketAH")))
    out.update(ea_values(P, M.RELENTLESS_HUNTER, ("StartingOOCMS", "OOCMS")))
    out.update(ea_values(P, M.ULTIMATE_HUNTER, ("StartingUltAH", "AdditionalUltAH")))
    for name, perk, a, b in (("ELEC", M.ELECTROCUTE, "DamageBase", "DamageMax"),
                             ("HOB", M.HAIL_OF_BLADES, "BonusDamageMin", "BonusDamageMax"),
                             ("CS", M.CHEAP_SHOT, "DamageIncMin", "DamageIncMax"),
                             ("SI", M.SUDDEN_IMPACT, "MinDamageTooltip", "MaxDamageTooltip"),
                             ("TOB", M.TASTE_OF_BLOOD, "HealAmount", "HealAmountMax")):
        out[f"{P}.lin.{name}"] = lin_pair(ea(perk, a), ea(perk, b))
    for name in ("ELEC_STACKS", "ELEC_DELAY", "DH_SOUL_DELAY", "DH_MIN_DAMAGE", "HOB_CANCEL_LOCKOUT", "BOUNTY_MAX"):
        out[f"{P}.{name}"] = getattr(M, name)
    from lanerl_jax.modern.runes.effects.core import COMBAT_TIMEOUT
    out[f"{P}.COMBAT_TIMEOUT"] = COMBAT_TIMEOUT
    return out
