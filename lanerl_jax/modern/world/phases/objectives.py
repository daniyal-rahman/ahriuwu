"""Phase 2b, OBJECTIVES: epic monsters (``jungle.objectives``) and their slot writes."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from ...core import types as W
from ...jungle import objectives as OBJ
from .. import views as V
from ..config import N_CHAMPIONS, WorldConfig
from ..scratch import TickScratch
from ..state import ModernOrders, ModernState


def apply_writes(s: ModernState, so, cfg: WorldConfig) -> ModernState:
    """Write the epic slots from ``objectives_step`` (spawns, despawns, relocations, heals)."""
    w = so.writes
    e0 = cfg.objectives.slot0
    sl = slice(e0, e0 + 8)
    n = s.kind.shape[0]

    def put(arr, v, mask=w.write):
        seg = arr[sl]
        return arr.at[sl].set(jnp.where(mask, jnp.asarray(v, arr.dtype), seg))
    new = w.write & w.new_seq
    seq = s.next_seq + jnp.cumsum(new.astype(jnp.int32)) - 1
    s = s._replace(
        kind=put(s.kind, w.kind), sub=put(s.sub, w.sub), team=put(s.team, w.team), alive=put(s.alive, True),
        targetable=put(s.targetable, True), x=put(s.x, w.x), y=put(s.y, w.y), hp=put(s.hp, w.hp),
        max_hp=put(s.max_hp, w.max_hp), armor=put(s.armor, w.armor), mr=put(s.mr, w.mr), ad=put(s.ad, w.ad),
        arange=put(s.arange, w.arange), aspeed=put(s.aspeed, w.aspeed), mspeed=put(s.mspeed, w.mspeed),
        radius=put(s.radius, w.radius), windup=put(s.windup, w.windup),
        missile_speed=put(s.missile_speed, w.missile_speed),
        spawn_seq=put(s.spawn_seq, seq, new), spawn_time=put(s.spawn_time, s.t, new),
        next_seq=s.next_seq + jnp.sum(new.astype(jnp.int32)))
    s = V.reset_slots(s, jnp.zeros((n,), bool).at[sl].set(new))
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
    obj, so = OBJ.objectives_step(s.obj, cfg.objectives, V.units_view(s), now=now, dt=jnp.float32(dt),
                                  levels=level, damage_matrix=s.damage_matrix, champ=cinfo,
                                  last_damaged=champ.last_damaged,
                                  ult_cast=s.champ.last_cast[:, 3] >= s.t - 1e-6, key=k_obj)
    s = apply_writes(s._replace(obj=obj), so, cfg)
    return s, sc._replace(so=so, cinfo=cinfo, key=key)
