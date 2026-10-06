"""Phase 2, STATS: static champion stats (items, shards, monster buffs, kit) and the STAT.70 max-HP sync.
Writes champion ``s.hp``/``s.max_hp`` and ``s.champ.static_max_hp``."""
from __future__ import annotations

import jax.numpy as jnp

from ...core.stat_pipeline import sync_max_health
from .. import views as V
from ..config import N_CHAMPIONS, WorldConfig
from ..scratch import TickScratch
from ..state import ModernOrders, ModernState


def run(s: ModernState, orders: ModernOrders, cfg: WorldConfig, sc: TickScratch) -> tuple[ModernState, TickScratch]:
    c = N_CHAMPIONS
    champ = sc.champ
    static, st_static = V.static_stats(s, cfg, sc.caps, sc.now, cfg.dt)
    # Dynamic health is synced by combat_tick.
    old_total = s.max_hp[:c]
    new_total = st_static.max_hp + s.combat.dyn_health
    hp_c, mx_c = sync_max_health(s.hp[:c], old_total, new_total)
    hp_c = jnp.where(s.alive[:c], hp_c, s.hp[:c])
    s = s._replace(hp=s.hp.at[:c].set(hp_c), max_hp=s.max_hp.at[:c].set(mx_c),
                   champ=champ._replace(static_max_hp=st_static.max_hp))
    return s, sc._replace(static=static, st_static=st_static)
