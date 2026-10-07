"""combat_tick: the Tear-line transforms (starters.TRANSFORMS) with their charge thresholds."""


def consts():
    from lanerl_jax.modern.items.effects.core import dv
    from lanerl_jax.modern.items.effects.starters import TRANSFORMS
    return {"combat.transforms": [v for a, b in TRANSFORMS for v in (a, b, dv(a, "MaxMana"))]}
