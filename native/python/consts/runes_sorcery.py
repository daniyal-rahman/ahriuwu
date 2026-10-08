"""runes.effects.sorcery constants: ``ea`` values as ``runes.sorcery.<perk>.<name>``, module constants by name,
``lin`` pairs ``[start, end - start]`` and the Python-folded sub-expressions of the JAX formulas."""
from .runes_domination import ea_values
from .runes_precision import lin_pair


def consts():
    from lanerl_jax.modern.runes.catalog import ea
    from lanerl_jax.modern.runes.effects import sorcery as M
    P = "runes.sorcery"
    out = {f"{P}.{n}": getattr(M, n) for n in (
        "SR_THRESHOLD", "SR_DURATION", "SR_HASTE", "SR_RANGED", "SR_SLOW_RESIST", "SR_BUCKET", "SR_WINDOW_BUCKETS",
        "SR_SLOTS", "AERY_TRAVEL", "AERY_LINGER", "AERY_ACCEL", "AERY_NEAR", "AERY_SUMMONER_GAP", "COMET_DELAY",
        "COMET_RADIUS", "COMET_MAX_RANGE", "COMET_MAX_AMP", "DFT_TICK", "DFT_AMP", "DFT_SPELL", "DFT_AOE", "DFT_DOT",
        "SCORCH_DELAY", "SCORCH_CD", "MF_CAP", "EPS")}
    out[f"{P}.AERY_RETURN_SPEED"] = [x for pair in M.AERY_RETURN_SPEED for x in pair]
    out[f"{P}.AERY_K2"] = M.AERY_ACCEL + M.AERY_NEAR_ACCEL
    out[f"{P}.DFT_AMP_AT"] = M.DFT_TIME_TO_AMP - M.EPS
    out[f"{P}.lin.SR_CD"] = lin_pair(*M.SR_CD)
    for name, perk, a, b in (("AERY", M.AERY, "DamageBase", "DamageMax"),
                             ("COMET", M.COMET, "DamageBase", "DamageMax"),
                             ("COMET_CD", None, None, None),
                             ("DFT", M.DEATHFIRE, "Level1DamageTOOLTIP", "{2156e250}"),
                             ("SCORCH", M.SCORCH, "Damage", "DamageMax"),
                             ("WW", M.WATERWALKING, "MinAdaptive", "MaxAdaptive"),
                             ("ABS", M.ABSOLUTE_FOCUS, "MinAdaptive", "MaxAdaptive")):
        out[f"{P}.lin.{name}"] = lin_pair(20.0, ea(M.COMET, "RechargeTimeMin")) if perk is None \
            else lin_pair(ea(perk, a), ea(perk, b))
    out.update(ea_values(P, M.AERY, ("DamageADRatio", "DamageAPRatio")))
    out.update(ea_values(P, M.COMET, ("ADRatio", "APRatio")))
    out.update(ea_values(P, M.DEATHFIRE, ("ADRatio", "APRatio")))
    out.update(ea_values(P, M.AXIOM, ("AOEAmp", "DamageAmp")))
    out.update(ea_values(P, M.MANAFLOW, ("ManaIncrease", "Cooldown", "PercentManaRestoreCooldown",
                                         "PercentManaRestore")))
    out.update(ea_values(P, M.NIMBUS, ("Duration", "{2fd68801}", "{b0d06764}", "LowCDMSBoost", "{1c32110c}",
                                       "HighCDMSBoost")))
    out.update(ea_values(P, M.CELERITY, ("PercentHasteMod",)))
    out.update(ea_values(P, M.WATERWALKING, ("MovementSpeed",)))
    out.update(ea_values(P, M.ABSOLUTE_FOCUS, ("HealthPercent",)))
    out.update(ea_values(P, M.TRANSCENDENCE, ("LevelToTurnOn", "LevelToTurnOn2", "LevelToTurnOn3", "HasteBonus1",
                                              "HasteBonus2")))
    cel_amp = ea(M.CELERITY, "PercentHasteMod")
    out[f"{P}.CEL_MS"] = ea(M.CELERITY, "PercentMS") / (1.0 + cel_amp)
    out[f"{P}.WW_DECAY"] = ea(M.WATERWALKING, "{6f2f0d30}", 1.0)
    out[f"{P}.GS_PERIOD"] = 60.0 * ea(M.GATHERING_STORM, "UpdateAfterMinutes")
    out[f"{P}.GS_HALF"] = ea(M.GATHERING_STORM, "AdaptiveAP") / 2.0
    out[f"{P}.AXIOM_BASE"] = 1.0 - ea(M.AXIOM, "UltimateRefundBase") / 100.0
    out[f"{P}.TR_BASE"] = 1.0 - ea(M.TRANSCENDENCE, "KillCooldownRefund")
    return out
