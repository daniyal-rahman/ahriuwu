"""Phase 11, FOG: attack-reveal circles, next tick's visibility, enemy-witnessed casts."""
from __future__ import annotations

import jax.numpy as jnp

from ... import vision as MV
from .. import views as V
from ..config import N_CHAMPIONS, WorldConfig
from ..scratch import TickScratch
from ..state import ModernOrders, ModernState


def run(s: ModernState, orders: ModernOrders, cfg: WorldConfig, sc: TickScratch) -> tuple[ModernState, TickScratch]:
    c = N_CHAMPIONS
    now, s0, x, y, champ = sc.now, sc.s0, sc.x, sc.y, sc.champ
    enemy_team = 1 - s.team[:c]
    hidden = ~s0.visible[enemy_team, jnp.arange(c)] & s0.alive[:c]
    struck = sc.launched[:c] | (sc.kit_all.cast_started & (sc.cast_order.target >= 0))
    reveal = MV.reveal_step(s.reveal, hidden, struck, x[:c], y[:c], now)
    vis_next, sight_next, ray_over = V.visibility(cfg, x, y, sc.kind, sc.sub, sc.team, sc.alive, reveal, now,
                                                  wards=sc.wards, level=sc.econ.level, variant=s.terrain_variant,
                                                  jungle=sc.jungle)
    witnessed = vis_next[enemy_team, jnp.arange(c)] | ~hidden
    champ = champ._replace(seen_cast=jnp.where(sc.cast_now & witnessed[:, None], now, champ.seen_cast))
    return s, sc._replace(reveal=reveal, visible=vis_next, sight=sight_next, ray_over=ray_over, champ=champ)
