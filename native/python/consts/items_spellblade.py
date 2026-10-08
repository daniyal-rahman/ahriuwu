"""items.effects.spellblade constants (Sheen, Trinity Force, Dusk and Dawn, Phage)."""


def consts():
    from lanerl_jax.modern.items.effects import spellblade as M
    p = "items.spellblade."
    return {p + "sb_cooldown": M.SB_COOLDOWN, p + "sb_window": M.SB_WINDOW,
            p + "sheen_ad": M.SHEEN_AD, p + "trinity_ad": M.TRINITY_AD, p + "dd_ad": M.DD_AD, p + "dd_ap": M.DD_AP,
            p + "dd_heal_ap": M.DD_HEAL_AP, p + "dd_heal_bonus_hp": M.DD_HEAL_BONUS_HP,
            p + "dd_extra_delay": M.DD_EXTRA_DELAY,
            p + "quicken_ms": M.QUICKEN_MS, p + "quicken_duration": M.QUICKEN_DURATION,
            p + "rage_ms": M.RAGE_MS, p + "rage_duration": M.RAGE_DURATION, p + "rage_ranged": M.RAGE_RANGED}
