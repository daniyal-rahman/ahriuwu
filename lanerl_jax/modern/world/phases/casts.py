"""Phase 3, CASTS: shop, kit casts and periodic effects, summoner spells, Smite, kit attack modifiers.
Writes ``s.econ`` (shop gold) and ``s.champ`` (inventory, group cooldowns)."""
from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from ... import champions as K
from ...champions import summoners as S
from ...champions.core import merge_out
from ...core import types as W
from ...core.stat_pipeline import ChampionStats, compose
from ...items import inventory as I
from ...items.catalog import catalog, combine_stats
from ...jungle import camps as J
from .. import units as U
from .. import views as V
from ..config import N_CHAMPIONS, WorldConfig
from ..scratch import TickScratch
from ..state import ModernOrders, ModernState


def shop(s: ModernState, cfg: WorldConfig, orders: ModernOrders, st: ChampionStats, forbid) -> tuple[Any, Any, Any]:
    """Buy/sell for each champion in the shop area or dead (ITEMS §1)."""
    cat = catalog()
    ids = jnp.asarray(cat.arrays.item_id)
    inv, gold, gcd = s.champ.inventory, s.econ.gold, s.champ.group_cd
    can = I.in_shop_area(s.x[:N_CHAMPIONS], s.y[:N_CHAMPIONS], s.team[:N_CHAMPIONS], ~s.alive[:N_CHAMPIONS])
    items, stacks, golds, cds, codes, bought, sold = [], [], [], [], [], [], []
    for c in range(N_CHAMPIONS):
        inv_c = I.Inventory(inv.item[c], inv.stack[c])
        row = jnp.argmax(ids == orders.buy[c])
        want = (orders.buy[c] > 0) & jnp.any(ids == orders.buy[c]) & ~forbid[c, row]
        if cfg.item_allowed is not None:
            want = want & jnp.asarray(cfg.item_allowed[c])[row]
        r = I.buy(inv_c, gold[c], row, can_shop=can[c] & want, level=s.econ.level[c],
                  is_ranged=st.attack_range[c] > 300.0, now=s.t, group_cd_until=gcd[c])
        inv_c, g = r.inv, r.gold
        srow = jnp.argmax(ids == orders.sell[c])
        slot = jnp.argmax(inv_c.item == srow)
        has_it = (orders.sell[c] > 0) & jnp.any(inv_c.item == srow)
        r2 = I.sell(inv_c, g, slot, can_shop=can[c] & has_it)
        inv_c = r2.inv
        items.append(inv_c.item); stacks.append(inv_c.stack); golds.append(r2.gold); cds.append(r.group_cd_until)
        codes.append(jnp.where(want, r.code, jnp.where(has_it, r2.code, 0)))
        bought.append(jnp.where(want & r.ok, orders.buy[c], 0))
        sold.append(jnp.where(has_it & r2.ok, orders.sell[c], 0))
    inv = I.Inventory(jnp.stack(items), jnp.stack(stacks))
    return (inv, jnp.stack(golds), jnp.stack(cds), jnp.stack(codes), jnp.stack(bought).astype(jnp.int32),
            jnp.stack(sold).astype(jnp.int32))


def run(s: ModernState, orders: ModernOrders, cfg: WorldConfig, sc: TickScratch) -> tuple[ModernState, TickScratch]:
    c, n, dt = N_CHAMPIONS, cfg.n_units, cfg.dt
    now, caps, st_static = sc.now, sc.caps, sc.st_static
    level = s.econ.level
    inv, gold, gcd, shop_code, bought, sold = shop(s, cfg, orders, st_static, sc.champ.forbid)
    champ = s.champ._replace(inventory=inv, group_cd=gcd)
    s = s._replace(econ=s.econ._replace(gold=gold), champ=champ)
    summ_world = combine_stats(sc.static, s.champ.dyn)
    st = compose(cfg.champion_base, level, summ_world, adaptive_physical=cfg.adaptive_physical,
                 slow=caps["slow"][:c])
    units = U.units_view(s)
    kctx = V.kit_ctx(s, cfg, st, caps, now, dt)
    ictx = V.item_ctx(s, cfg, st_static, now, dt)
    locked = now < champ.cast_lock_until
    item_casting = now < champ.item_cast_until
    can_cast = caps["can_cast"][:c] & s.alive[:c] & ~locked & ~item_casting
    order = W.CastOrder(jnp.where(can_cast, orders.cast_slot, -1).astype(jnp.int32), orders.cast_target,
                        orders.cast_x, orders.cast_y)
    kits, k_cast = K.cast(s.kits, kctx, units, order)
    kits, k_per = K.periodic(kits, kctx, units)
    kit_out = merge_out([k_cast, k_per], c, n)
    summ_req = W.CastOrder(orders.summoner_slot, orders.summoner_target, orders.summoner_x, orders.summoner_y)
    summ, s_eff, s_out = S.step(s.summoners, ictx, units, request=summ_req, now=now, dt=jnp.float32(dt),
                                summoner_haste=st.summoner_haste,
                                can_cast=caps["can_summoner"][:c] & s.alive[:c],
                                channel_interrupted=caps["stunned"][:c] | ~s.alive[:c],
                                quest_complete=s.econ.quest.complete,
                                rooted=(s.cc.root_until[:c] > now))
    sm = None
    jungle = s.jungle
    if cfg.jungle is not None:
        jungle, sm = J.smite_step(jungle, cfg.jungle, units, summ_req, s.summoners.spell, now=now,
                                  summoner_haste=st.summoner_haste, alive=s.alive[:c] & caps["can_summoner"][:c])
    kmods = K.attack_mods(kits, kctx)
    reach = st.attack_range + kmods.extra_range
    return s, sc._replace(champ=champ, st=st, summ_world=summ_world, units=units, kctx=kctx, ictx=ictx,
                          locked=locked, item_casting=item_casting, cast_order=order, kits=kits, kit_out=kit_out,
                          kmods=kmods, reach=reach, summ=summ, s_eff=s_eff, s_out=s_out, sm=sm, jungle=jungle,
                          shop_code=shop_code, bought=bought, sold=sold)
