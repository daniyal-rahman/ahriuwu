"""runes.effects.resolve constants: module constants by name and ``lin`` pairs ``[start, end - start]``."""
from .runes_precision import lin_pair


def consts():
    from lanerl_jax.modern.runes.effects import resolve as M
    P = "runes.resolve"
    out = {f"{P}.{n}": getattr(M, n) for n in (
        "GRASP_PCT_DAMAGE", "GRASP_PCT_HEAL", "GRASP_HP_MELEE", "GRASP_HP_RANGED", "GRASP_RANGED_MOD",
        "GRASP_STACKS", "GRASP_WINDOW", "GRASP_GEN_AFTER", "_EPS",
        "AS_FLAT", "AS_PCT", "AS_DELAY", "AS_HP_RATIO", "AS_RADIUS", "AS_COOLDOWN",
        "GD_RANGE", "GD_GUARD", "GD_AP", "GD_HP", "GD_SHIELD_DURATION", "GD_BUCKET", "GD_BUCKETS",
        "DEMO_BASE_MELEE", "DEMO_BASE_RANGED", "DEMO_HP_MELEE", "DEMO_HP_RANGED", "DEMO_COOLDOWN", "DEMO_LOCK",
        "DEMO_STACKS", "FONT_RANGED", "FONT_COOLDOWN", "FONT_RANGE", "SB_HP", "SB_SHIELD", "SB_LINGER",
        "SB_ASSUMED_SHIELD_LIFE", "COND_TIME", "COND_ARMOR", "COND_MR", "COND_PCT", "SW_DURATION", "SW_RATE",
        "BP_COUNT", "BP_DURATION", "BP_COOLDOWN", "OG_RANGE", "OG_PER_TIER", "OG_HP_PER_TIER", "OG_THRESHOLD",
        "OG_PCT", "REV_HSP", "REV_CUTOFF", "REV_AMP", "UNF_RESIST", "UNF_LINGER")}
    out[f"{P}.OG_RANGE_SQ"] = M.OG_RANGE ** 2
    for name, a, b in (("AS_CAP", M.AS_CAP_MIN, M.AS_CAP_MAX), ("AS_DMG", M.AS_DMG_MIN, M.AS_DMG_MAX),
                       ("GD_SHIELD", M.GD_SHIELD_MIN, M.GD_SHIELD_MAX), ("GD_CD", M.GD_CD_MIN, M.GD_CD_MAX),
                       ("GD_THR", M.GD_THR_MIN, M.GD_THR_MAX), ("FONT", M.FONT_MIN, M.FONT_MAX),
                       ("SB", M.SB_MIN, M.SB_MAX), ("BP", M.BP_MIN, M.BP_MAX)):
        out[f"{P}.lin.{name}"] = lin_pair(a, b)
    return out
