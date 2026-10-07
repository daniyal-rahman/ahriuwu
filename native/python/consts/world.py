"""World-level constants the full tick bakes in (mechanics, summoners)."""


def consts():
    import jax.numpy as jnp
    import numpy as np

    from lanerl_jax.modern.champions import summoners as S
    return {"world.flash_linspace": np.asarray(jnp.linspace(1.0, 0.0, 16)), "world.flash_range": S.FLASH_RANGE,
            **_actualizer()}


def _actualizer():
    from lanerl_jax.modern.items.effects.actives import ACTUALIZER
    from lanerl_jax.modern.items.effects.core import dv
    return {"world.actualizer_mana": dv(ACTUALIZER, "ManaCostIncrease"),
            "world.actualizer_cd": dv(ACTUALIZER, "CooldownTick")}
