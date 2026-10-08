"""Jax kit constants (lanerl_jax.modern.champions.jax), evaluated with the same data helpers."""


def consts():
    from lanerl_jax.modern.champions import jax as J
    from .kits_garen import kit_consts
    ranked = [("Q", "Damage"), ("W", "Damage"), ("E", "BaseDamage"), ("R", "SwingDamageBase"), ("R", "BaseResists"),
              ("R", "ResistsPerExtraTarget"), ("R", "PassiveBaseDamage")]
    extra = {"E_MIN_EPS": J.E_MIN - 1e-6}
    return kit_consts(J, ranked, extra)
