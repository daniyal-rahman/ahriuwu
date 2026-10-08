"""items.effects.actives constants (Zhonya's, Seeker's, Youmuu's, Gunblade; values other actives' state reads)."""


def consts():
    from lanerl_jax.modern.items.effects import actives as M
    dv = M.dv
    p = "items.actives."
    return {p + "cooldown": [M._cd(i) for i in M.ACTIVE_ITEMS], p + "stasis": M.STASIS_S,
            p + "shurelya_ms": dv(M.SHURELYA, "ActiveMoveSpeed"), p + "mercurial_ms": dv(M.MERCURIAL, "MoveSpeed"),
            p + "youmuu_duration": dv(M.YOUMUU, "DurationNDV"),
            p + "youmuu_duration_ranged": dv(M.YOUMUU, "DurationNDV") * 0.667,
            p + "youmuu_ms_melee": dv(M.YOUMUU, "MeleeItemCalcValueB"),
            p + "youmuu_ms_ranged": dv(M.YOUMUU, "RangedItemCalcValueB"),
            p + "gunblade_slow": dv(M.GUNBLADE, "SlowAmount"), p + "gunblade_slow_duration": dv(M.GUNBLADE, "SlowDuration"),
            p + "rocket_raw_base": dv(M.ROCKETBELT, "BaseDamage"), p + "rocket_raw_ap": dv(M.ROCKETBELT, "APRatio"),
            p + "locket_duration": dv(M.LOCKET, "ShieldDuration"),
            p + "red_aoe": dv(M.REDEMPTION, "AOESize"), p + "red_damage": dv(M.REDEMPTION, "DamageToChampions"),
            p + "red_heal_min": dv(M.REDEMPTION, "HealMin"),
            p + "actualizer_mana_cost_mult": 1.0 + dv(M.ACTUALIZER, "ManaCostIncrease"),
            p + "actualizer_cd_rate": 1.0 + dv(M.ACTUALIZER, "CooldownTick"),
            p + "red_heal_span": dv(M.REDEMPTION, "HealMax") - dv(M.REDEMPTION, "HealMin")}
