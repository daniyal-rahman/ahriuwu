"""items.effects.consumables constants (dv and host-side float64 expressions JAX bakes as float32)."""


def consts():
    from lanerl_jax.modern.items.effects import consumables as M
    dv = M.dv
    ticks_hp = dv(M.HEALTH_POTION, "PotionDuration") / M.HOT_PERIOD
    ticks_rf = dv(M.REFILLABLE, "PotionDuration") / M.HOT_PERIOD
    p = "items.consumables."
    return {p + "refill_max": dv(M.REFILLABLE, "MaxCharges"),
            p + "ticks_hp": ticks_hp, p + "ticks_rf": ticks_rf,
            p + "per_hp": dv(M.HEALTH_POTION, "HealAmount") / ticks_hp,
            p + "per_rf": dv(M.REFILLABLE, "HealAmount") / ticks_rf,
            p + "elixir_duration": [M.ELIXIR_DURATION[i] for i in M.ELIXIRS],     # Iron, Sorcery, Wrath
            p + "iron_hp": M.IRON_HP, p + "iron_tenacity": M.IRON_TENACITY,
            p + "sorcery_ap": M.SORCERY_AP, p + "sorcery_true": M.SORCERY_TRUE, p + "sorcery_icd": M.SORCERY_ICD,
            p + "sorcery_mana_regen": M.SORCERY_MANA_REGEN,
            p + "wrath_ad": M.WRATH_AD, p + "wrath_drain": M.WRATH_DRAIN,
            p + "force_adaptive": dv(M.FORCE, "AdaptiveAmount"), p + "force_duration": dv(M.FORCE, "Duration"),
            p + "avarice_duration": dv(M.AVARICE, "Duration"), p + "avarice_gold": dv(M.AVARICE, "GoldAmount"),
            p + "avarice_on_hit": dv(M.AVARICE, "OnHitDamage"),
            p + "biscuit_ticks": M.BISCUIT_DURATION / M.HOT_PERIOD}
