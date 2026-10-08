"""items.effects.hydra constants (Tiamat, Stridebreaker; Titanic/Ravenous/Profane values of padded packets)."""


def consts():
    from lanerl_jax.modern.items.effects import hydra as M
    dv = M.dv
    p = "items.hydra."
    acts = list(M._ACTIVES.values())          # Tiamat, Ravenous, Profane, Stridebreaker
    return {p + "max_splash": M.MAX_SPLASH, p + "cleave_radius": M.CLEAVE_RADIUS,
            p + "stride_decay": M.STRIDE_DECAY,
            p + "active_ratio": [a[0] for a in acts], p + "active_radius": [a[1] for a in acts],
            p + "active_cooldown": [a[2] for a in acts], p + "active_base_cast": [a[3] for a in acts],
            p + "titanic_ranged": dv(M.TITANIC, "RangedEffectiveness"),
            p + "titanic_primary": dv(M.TITANIC, "PrimaryTargetHPRatio"),
            p + "titanic_active_primary": dv(M.TITANIC, "ActivePrimaryTargetHPRatio"),
            p + "titanic_splash": dv(M.TITANIC, "SplashHPRatio"),
            p + "titanic_active_splash": dv(M.TITANIC, "ActiveSplashHPRatio"),
            p + "titanic_cooldown": dv(M.TITANIC, "Cooldown"),
            p + "stride_slow": -dv(M.STRIDEBREAKER, "MSSlow"), p + "stride_duration": dv(M.STRIDEBREAKER, "Duration"),
            p + "stride_active_ms": dv(M.STRIDEBREAKER, "ActiveMS")}
