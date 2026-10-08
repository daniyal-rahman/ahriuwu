"""Garen kit constants (lanerl_jax.modern.champions.garen), evaluated with the same data helpers; ``kit_consts``
is shared with ``kits_jax``."""


def kit_consts(mod, ranked_keys, extra=None) -> dict:
    """``kits.<name>.<ATTR>`` for every numeric module constant, ``kits.<name>.<slot>.<key>`` per ranked JSON value
    (indexed by rank 0..6, as ``core.ranked``), ``kits.<name>.cd.<slot>`` and ``kits.<name>.mana.<slot>`` tables
    (``core.cooldown_row`` / ``core.mana_row``, empty for manaless slots), plus ``extra`` derived Python values."""
    from lanerl_jax.modern.data.champions import cooldowns, spell, values
    name = mod.NAME
    p = f"kits.{name.lower()}."
    out = {}
    for attr in dir(mod):
        v = getattr(mod, attr)
        if attr.isupper() and isinstance(v, (int, float)) and not isinstance(v, bool):
            out[p + attr] = float(v)
        elif attr == "UNIT_TARGET_RANGE":
            out[p + attr] = list(v)
    for slot, key in ranked_keys:
        out[f"{p}{slot}.{key}"] = list(values(name, slot, key))
    for slot in "QWER":
        out[f"{p}cd.{slot}"] = list(cooldowns(name, slot))
        mana = spell(name, slot).get("mana")
        out[f"{p}mana.{slot}"] = [] if mana is None else list(mana["values"])
    out.update({p + k: v for k, v in (extra or {}).items()})
    return out


def consts():
    from lanerl_jax.modern.champions import garen as G
    ranked = [("Q", "MovementSpeedDuration"), ("Q", "BaseDamage"), ("Q", "SilenceDuration"), ("W", "BaseShield"),
              ("W", "DRPercent"), ("E", "NumTicks"), ("E", "ASPerTick"), ("E", "BaseDamagePerTick"),
              ("E", "ADRatioPerTick"), ("R", "BaseDamage"), ("R", "ExecuteDamage")]
    # Python-side expressions of the JAX code (float64, then float32 in the graph).
    extra = {"E_MIN_SPIN_EPS": G.E_MIN_SPIN - 1e-6, "E_NEAREST_MULT": 1.0 + G.E_NEAREST,
             "Q_AD_EXTRA": G.Q_AD_RATIO - 1.0, "Q_LOCK_FRACTION": 1.0 - G.Q_WINDUP_FRACTION,
             "PULSE_FRACTION": G.PASSIVE_PULSE / 5.0}
    return kit_consts(G, ranked, extra)
