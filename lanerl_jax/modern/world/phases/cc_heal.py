"""Phase 8, CC / HEAL: heals and shields from kits, summoners and monsters; CC with tenacity."""
from __future__ import annotations

import jax.numpy as jnp

from ... import mechanics as M
from ...core import damage as D
from ...core import types as W
from ...items.effects import actives as A
from ...items.effects.core import ShieldGrant, effects, merge_effects, shield_grants
from ...items.effects.runtime import apply_effects
from ..config import N_CHAMPIONS, WorldConfig
from ..scratch import TickScratch
from ..state import ModernOrders, ModernState


def run(s: ModernState, orders: ModernOrders, cfg: WorldConfig, sc: TickScratch) -> tuple[ModernState, TickScratch]:
    """8. CC / HEAL: kit, summoner and monster heals and shields; CC with tenacity and slow resist."""
    c, n = N_CHAMPIONS, cfg.n_units
    now, st, kit_all, k_dmg, s_eff, s_out, out = sc.now, sc.st, sc.kit_all, sc.k_dmg, sc.s_eff, sc.s_out, sc.out
    jfx, so, sm, mai, sres, kdef = sc.jfx, sc.so, sc.sm, sc.mai, sc.sres, sc.kdef
    hp, max_hp, shields, status = sc.hp, sc.max_hp, sc.shields, sc.status
    kit_shields = ShieldGrant(*(jnp.concatenate([u, v], axis=1) for u, v in zip(kit_all.shield, k_dmg.shield)))
    m_heal = jnp.zeros((c,), jnp.float32)
    m_mana = jnp.zeros((c,), jnp.float32)
    m_shield = jnp.zeros((c,), jnp.float32)
    if jfx is not None:
        m_heal, m_shield = m_heal + jfx.heal, jnp.maximum(m_shield, jfx.shield)
    if so is not None:
        m_heal, m_mana, m_shield = m_heal + so.heal, m_mana + so.mana, jnp.maximum(m_shield, so.shield)
    extra = merge_effects([s_eff._replace(packets=D.empty_packets(0)),
                           effects(c, n, heal=kit_all.heal + k_dmg.heal, shields=kit_shields),
                           effects(c, n, heal=m_heal, mana=m_mana,
                                   shields=shield_grants(m_shield, duration=jnp.inf))], c, n)
    hp, shields, status = apply_effects(extra, sc.ictx, hp, max_hp, shields, status,
                                        heal_power=st.heal_shield_power, incoming_heal=sc.summ_world.incoming_heal)
    # Champion-sourced slows without a (C, N) source row: Exhaust, item/rune effects (Rylai's, Stridebreaker,
    # Spellblade fields ...). ``CCTimers`` is the only slow state movement reads; ``status.slow`` is unused.
    slow_cc = W.no_cc(3, n)._replace(slow=jnp.stack([s_out.exhaust_slow, out.effects.slow, extra.slow]),
                                     slow_duration=jnp.stack([s_out.exhaust_slow_duration, out.effects.slow_duration,
                                                              extra.slow_duration]))
    ten = jnp.zeros((n,), jnp.float32).at[:c].set(
        1.0 - (1.0 - st.tenacity) * (1.0 - kdef.tenacity_bonus) * (1.0 - s_out.tenacity))
    if mai is not None:                                               # Scuttler: slow immune, -100% tenacity
        jsl = slice(cfg.jungle.monster0, cfg.jungle.monster0 + cfg.jungle.n_slots)
        ten = ten.at[jsl].set(1.0 - mai.cc_duration_mult)
    champ_cc = sc.cc_now
    for extra_cc in ([] if sm is None else [sm.cc]) + ([] if jfx is None else [jfx.cc]) \
            + ([] if so is None else [so.champion_cc]):
        champ_cc = W.merge_cc(champ_cc, extra_cc)
    item_cleanse = A.world(out.state.items.actives, now)
    cc = M.apply_cc(s.cc, champ_cc, ten, sres, now, source_is_champion=jnp.ones((c,), bool),
                    cleansed=jnp.zeros((n,), bool).at[:c].set(s_out.cleanse | item_cleanse.cleanse))
    for mcc in ([] if so is None else [so.cc]) + ([] if sc.obj_cc is None else [sc.obj_cc]):
        cc = M.apply_cc(cc, mcc, ten, sres, now, source_is_champion=jnp.zeros((mcc.stun.shape[0],), bool))
    cc = M.apply_cc(cc, slow_cc, ten, sres, now, source_is_champion=jnp.ones((3,), bool))
    clean_slow = jnp.zeros((n,), bool) if kit_all.cleanse_slow is None else \
        jnp.zeros((n,), bool).at[:c].set(kit_all.cleanse_slow)
    cc = cc._replace(slow_until=jnp.where(clean_slow, now, cc.slow_until))
    return s, sc._replace(extra=extra, hp=hp, shields=shields, status=status, item_cleanse=item_cleanse, cc=cc)
