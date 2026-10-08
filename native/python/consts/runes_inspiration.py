"""runes.effects.inspiration constants: module constants by name, the static per-item-row tables (``_tables``) and
the Python-folded sub-expressions of the JAX formulas."""
import numpy as np

from .runes_precision import lin_pair


def consts():
    from lanerl_jax.modern.runes.effects import inspiration as M
    P = "runes.inspiration"
    out = {f"{P}.{n}": getattr(M, n) for n in (
        "GA_RAYS", "GA_LENGTH", "GA_DURATION", "GA_CC_CARRY", "GA_COOLDOWN", "GA_REDUCTION", "GA_INDENT",
        "GA_SLOW_BASE", "GA_SLOW_BAD", "GA_SLOW_AP", "GA_SLOW_HSP", "GA_ALLY_RANGE",
        "SB_FIRST", "SB_BASE", "SB_PER_UNIQUE", "SB_MIN", "SB_OOC",
        "FS_DURATION", "FS_AMP", "FS_GOLD_FLAT", "FS_GOLD_MELEE", "FS_GOLD_RANGED", "FS_MODE_CD", "FS_DELAY",
        "FS_SLOTS", "FS_PUSH", "HX_MS", "HX_MS_DURATION", "HX_FLASH_GATE", "HX_COOLDOWN", "HX_COMBAT_CD",
        "HX_RANGE0", "HX_RANGE_STEP", "HX_RANGE_PERIOD", "HX_RANGE_MAX",
        "MF_AT", "MF_PER_TAKEDOWN", "MF_MS", "CB_REFUND", "BISCUIT_EVERY", "BISCUIT_COUNT", "BISCUIT_HP",
        "CI_SUMMONER", "CI_ITEM", "AV_OWN", "AV_OTHER", "AV_RANGE", "JACK_AH", "JACK_AF5", "JACK_AF10",
        "GRANT_SLOTS")}
    out[f"{P}.GA_HALF_WIDTH"] = M.GA_WIDTH / 2.0
    out[f"{P}.FS_GRACE_EPS"] = M.FS_GRACE + 1e-6
    out[f"{P}.HX_CHANNEL_EPS"] = M.HX_CHANNEL - 1e-6
    out[f"{P}.HX_MIN_EPS"] = M.HX_MIN - 1e-6
    out[f"{P}.lin.FS_CD"] = lin_pair(M.FS_CD_START, M.FS_CD_END)
    out[f"{P}.TONICS"] = [x for pair in M.TONICS for x in pair]
    t = M._tables()
    out[f"{P}.TWT_HEAL"] = [M.TWT_PCT * t["heal"][M.HEALTH_POTION], M.TWT_PCT * t["heal"][M.REFILLABLE]]
    # Fan rays without an ally: rotation of ray k (1..GA_RAYS-1) as numpy float64 cos/sin.
    rot = []
    for k in range(1, M.GA_RAYS):
        ang = M.GA_FAN * (1.0 if k % 2 else -1.0) * ((k + 1) // 2)
        rot += [np.cos(ang), np.sin(ang)]
    out[f"{P}.GA_ROT"] = rot
    out[f"{P}.tab.ids"] = t["ids"]
    out[f"{P}.tab.boots"] = t["boots"].astype(np.float32)
    out[f"{P}.tab.legendary"] = t["legendary"].astype(np.float32)
    out[f"{P}.tab.total"] = t["total"]
    out[f"{P}.tab.jack"] = t["jack"]                    # (I, types) row-major
    out[f"{P}.tab.slots"] = t["slots"]
    out[f"{P}.tab.max_stack"] = t["max_stack"]
    return out
