"""items.effects.boots constants (Steelcaps, Swiftmarch, Chainlaced, Armored Advance; Slay cap)."""


def consts():
    from lanerl_jax.modern.items.effects import boots as M
    p = "items.boots."
    out = {p + "steelcaps_mult": 1.0 - M.STEELCAPS_REDUCTION, p + "armored_mult": 1.0 - M.ARMORED_REDUCTION,
           p + "slay_max": M.SLAY_MAX, p + "swiftmarch_af": M.SWIFTMARCH_AF,
           p + "adaptive_ad_per_af": M.ADAPTIVE_AD_PER_AF}
    for name, iid in (("armored", M.ARMORED), ("chainlaced", M.CHAINLACED)):
        # (level-1 value, per level, from level, bonus-HP ratio, cooldown, duration, dtype, shield kind)
        out[p + "noxian_" + name] = list(M.NOXIAN[iid])
    return out
