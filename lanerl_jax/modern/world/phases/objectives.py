"""Phase 2b, OBJECTIVES: epic monsters (``jungle.objectives``) and their slot writes."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from ...core import types as W
from ...jungle import objectives as OBJ
from .. import units as U
from ..config import N_CHAMPIONS, WorldConfig
from ..scratch import TickScratch
from ..state import ModernOrders, ModernState


def apply_writes(s: ModernState, so, cfg: WorldConfig) -> ModernState:
    """Write the epic slots from ``objectives_step`` (spawns, despawns, relocations, heals)."""
    w = so.writes
    sl = cfg.objectives.slots

    def put(arr, v, mask):
        return arr.at[sl].set(jnp.where(mask, jnp.asarray(v, arr.dtype), arr[sl]))
    s = U.write_units(s, OBJ.unit_write(cfg.objectives, w, cfg.n_units), s.t)
    s = s._replace(kind=put(s.kind, W.KIND_NONE, w.despawn), alive=put(s.alive, False, w.despawn),
                   x=put(s.x, w.rx, w.relocate), y=put(s.y, w.ry, w.relocate))
    hp = s.hp.at[sl].set(jnp.minimum(s.hp[sl] + so.monster_heal, s.max_hp[sl]))
    return s._replace(hp=jnp.where(s.alive, hp, s.hp), terrain_variant=jnp.asarray(so.terrain_variant, jnp.int32))


def run(s: ModernState, orders: ModernOrders, cfg: WorldConfig,
                sc: TickScratch) -> tuple[ModernState, TickScratch]:
    """2b. OBJECTIVES: epic monsters (spawns, abilities, Rift transformation). Writes ``s.obj`` and the
    epic slots."""
    if cfg.objectives is None:
        return s, sc._replace(so=None, cinfo=None)
    c, dt = N_CHAMPIONS, cfg.dt
    now, key, champ, st_static = sc.now, sc.key, sc.champ, sc.st_static
    level = s.econ.level
    cinfo = OBJ.ChampInfo(level=level, bonus_ad=st_static.bonus_ad, ap=st_static.ap,
                          bonus_hp=st_static.max_hp - st_static.base_hp, max_hp=s.max_hp[:c],
                          max_mana=st_static.max_mana, adaptive_physical=cfg.adaptive_physical)
    k_obj, key = jax.random.split(key)
    obj, so = OBJ.objectives_step(s.obj, cfg.objectives, U.units_view(s), now=now, dt=jnp.float32(dt),
                                  levels=level, damage_matrix=s.prev.damage_matrix, champ=cinfo,
                                  last_damaged=champ.last_damaged,
                                  ult_cast=s.champ.last_cast[:, 3] >= s.t - 1e-6, key=k_obj)
    s = apply_writes(s._replace(obj=obj), so, cfg)
    return s, sc._replace(so=so, cinfo=cinfo, key=key)
