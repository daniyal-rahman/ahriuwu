"""runes.effects.precision constants. ``lin``/``lin_growth`` pairs are ``[start, end - start]`` (Python floats, as
JAX folds them); breakpoint marks are flat ``[start, per, stop - start]`` triples."""


def lin_pair(start, end):
    return [start, end - start]


def breakpoint_marks(initial_per_level, points):
    marks = [(2, initial_per_level)] + [(int(k), float(v)) for k, v in points]
    out = []
    for i, (start, per) in enumerate(marks):
        stop = marks[i + 1][0] if i + 1 < len(marks) else 10 ** 6
        out += [start, per, stop - start]
    return out


def module_consts(prefix, mod, names):
    return {f"{prefix}.{k}": getattr(mod, k) for k in names}


def consts():
    from lanerl_jax.modern.runes.effects import precision as M
    from lanerl_jax.modern.runes.catalog import ea
    P = "runes.precision"
    out = module_consts(P, M, (
        "CONQ_MAX_STACKS", "CONQ_DURATION", "CONQ_SAME_SPELL", "CONQ_HEAL", "CONQ_HEAL_RANGED",
        "PTA_HITS", "PTA_STACK_TIME", "PTA_AMP", "PTA_OOC", "PTA_COOLDOWN",
        "LT_DURATION", "LT_MAX", "LT_AS", "LT_AS_RANGED", "LT_BOLT_RANGED", "LT_DECAY",
        "FLEET_AD", "FLEET_AP", "FLEET_RANGED_HEAL", "FLEET_MINION", "FLEET_MS", "FLEET_MS_TIME",
        "FLEET_RANGED_MS", "FLEET_FULL", "FLEET_PER_HIT", "FLEET_UNITS_PER_CHARGE",
        "ABSORB_L1", "TRIUMPH_MISSING", "TRIUMPH_MAX", "TRIUMPH_GOLD", "DELAY",
        "POM_TABLE", "POM_CD", "POM_ENERGY", "POM_TAKEDOWN", "POM_RANGED",
        "LEGEND_TAKEDOWN", "LEGEND_MINION", "LEGEND_LARGE", "LEGEND_PER_STACK",
        "COUP_BELOW", "COUP_AMP", "CUT_ABOVE", "CUT_AMP", "LS_MIN", "LS_START"))
    out[f"{P}.CONQ"] = lin_pair(M.CONQ_MIN, M.CONQ_MAX)
    out[f"{P}.PTA"] = lin_pair(M.PTA_MIN, M.PTA_MAX)
    out[f"{P}.LT_BOLT"] = lin_pair(M.LT_BOLT_MIN, M.LT_BOLT_MAX)
    out[f"{P}.FLEET_HEAL"] = lin_pair(M.FLEET_HEAL_MIN, M.FLEET_HEAL_MAX)
    out[f"{P}.ABSORB_MARKS"] = breakpoint_marks(M.ABSORB_PER, M.ABSORB_POINTS)
    out[f"{P}.LS_SPAN"] = M.LS_MAX - M.LS_MIN
    out[f"{P}.LS_WIDTH"] = M.LS_START - M.LS_END
    for perk, name in ((M.ALACRITY, "ALACRITY"), (M.HASTE, "HASTE"), (M.BLOODLINE, "BLOODLINE")):
        out[f"{P}.{name}_MAX_STACKS"] = ea(perk, "MaxLegendStacks")
    out[f"{P}.ALACRITY_BASE"] = ea(M.ALACRITY, "AttackSpeedBase")
    out[f"{P}.ALACRITY_PER"] = ea(M.ALACRITY, "AttackSpeedPerStack")
    out[f"{P}.HASTE_BASE"] = ea(M.HASTE, "HasteBase")
    out[f"{P}.HASTE_PER"] = ea(M.HASTE, "HastePerStack")
    out[f"{P}.BLOODLINE_BASE"] = ea(M.BLOODLINE, "LifeStealBase")
    out[f"{P}.BLOODLINE_PER"] = ea(M.BLOODLINE, "LifeStealPerStack")
    out[f"{P}.BLOODLINE_HP"] = ea(M.BLOODLINE, "BonusHealth")
    return out
