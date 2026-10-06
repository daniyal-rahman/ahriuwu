"""Phase 7, DAMAGE: every packet of the tick through ``combat_tick`` (items, runes, damage pipeline), then kit
on-damage."""
from __future__ import annotations

import jax.numpy as jnp

from ... import champions as K
from ...champions import summoners as S
from ...combat import combat_tick
from ...core import damage as D
from ...core import types as W
from ...items import inventory as I
from ...items.effects import actives as A
from ...items.effects.core import CC, Cast
from ...jungle import objectives as OBJ
from ...lane import ai as LA
from ...runes.effects.core import rune_events
from .. import units as U
from .. import views as V
from ..config import N_CHAMPIONS, WorldConfig
from ..scratch import TickScratch
from ..state import ModernOrders, ModernState


def run(s: ModernState, orders: ModernOrders, cfg: WorldConfig, sc: TickScratch) -> tuple[ModernState, TickScratch]:
    c, n = N_CHAMPIONS, cfg.n_units
    now, caps, champ, units, kits, kctx, ictx = sc.now, sc.caps, sc.champ, sc.units, sc.kits, sc.kctx, sc.ictx
    so, sm, jfx, kit_all, s_out, towers = sc.so, sc.sm, sc.jfx, sc.kit_all, sc.s_out, sc.towers
    st_static, in_stasis = sc.st_static, sc.in_stasis
    more = list(sc.extra_atk)
    if sm is not None:
        more.append(sm.packets)
    if jfx is not None:
        more.append(jfx.packets)
    if so is not None:
        more.append(so.packets)
    base = D.concat_packets(sc.direct, sc.arrived, sc.og_pk, kit_all.packets, sc.s_eff.packets, *more)
    if so is not None:
        base = OBJ.objectives_packet_mods(s.obj, cfg.objectives, base, units, now=now)
    kdef = K.defense(kits, kctx)
    kdeb = K.debuffs(kits, kctx, units)
    t_armor, t_mr = LA.turret_defense(towers, units, now=now, slots=cfg.layout.ai_slots)
    t_mult, t_invuln = LA.structure_defense_mods(towers, units, now=now)
    armor = t_armor.at[:c].set(st_static.base_armor + st_static.bonus_armor)
    mr = t_mr.at[:c].set(st_static.base_mr + st_static.bonus_mr)
    if so is not None:                                                # Baron's Void Corruption
        armor, mr = armor - so.armor_reduction, mr - so.armor_reduction
    dfn = D.default_defense(n)._replace(
        armor=armor, magic_resist=mr, unit_class=W.damage_class(s.kind),
        received_mult=jnp.ones((n,), jnp.float32).at[:c].set(kdef.received_mult),
        received_mult_all=t_mult,
        invulnerable=t_invuln | jnp.zeros((n,), bool).at[:c].set(s_out.teleport_dash | in_stasis),
        percent_armor_reduction=kdeb.percent_armor_reduction,
        dodge_basic=jnp.zeros((n,), bool).at[:c].set(kdef.dodge_basic),
        aoe_received_mult=jnp.ones((n,), jnp.float32).at[:c].set(kdef.aoe_received_mult))
    off = LA.turret_offense(units, D.default_offense(n)._replace(unit_class=W.damage_class(s.kind),
                                                                dealt_reduction=s_out.exhaust_reduction))
    cc_now = kit_all.cc
    cc_items = CC(cc_now.slow > 0, (cc_now.stun > 0) | (cc_now.root > 0) | (cc_now.knockup > 0))
    # Overgrowth (U-15): last tick's deaths, with sight latched while the victim lived.
    deaths_prev = jnp.any(s.prev.death_seen, axis=0)
    ev = rune_events(ictx, n, game_time=now, attack_started=sc.started[:c], attack_start_target=sc.att.target[:c],
                     attack_cancelled=sc.cancelled[:c], attack_reset=sc.reset[:c], cast_id=kit_all.cast_id,
                     cc_duration=jnp.maximum(cc_now.stun, cc_now.root), impaired=caps["impaired"],
                     movement_impaired=caps["movement_impaired"],
                     impaired_by_holder=(cc_now.slow > 0) | (cc_now.stun > 0) | (cc_now.root > 0),
                     holder_cc_from_champion=s.cc.champion_cc_until[:c] > now,
                     summoner_cast=s_out.cast_event, summoner_cooldown=s_out.cast_cooldown,
                     summoner_is_teleport=s_out.is_teleport,
                     blinked=s_out.blinked | sc.dstart, flash_cooldown=S.flash_cooldown(sc.summ, now),
                     deaths=deaths_prev, purchased=sc.bought, sold=sc.sold, granted=champ.granted,
                     uses_energy=cfg.uses_energy, adaptive_physical=cfg.adaptive_physical,
                     is_turret=s.kind == W.KIND_TURRET, cc_cast_id=cc_now.cast_id,
                     cc_on_hit=jnp.zeros((c, n), bool), sight=s.sight | s.prev.death_seen, visible=sc.vis_c,
                     epic_takedown=s.prev.epic, large_monster_kill=s.prev.large,
                     in_river=V.in_river(cfg, s.x[:c], s.y[:c]))
    items0 = s.combat.items
    items0 = items0._replace(actives=A.with_aim(items0.actives, orders.cast_target, orders.cast_x, orders.cast_y))
    item_req = A.request_allowed(orders.item_active, disabled=caps["stunned"][:c], in_stasis=in_stasis)
    out = combat_tick(s.combat._replace(items=items0), I.owned_counts(champ.inventory), cfg.rune_pages, ictx,
                      U.item_units(s), attack=sc.attack,
                      cast=Cast(kit_all.cast_started, kit_all.cast_slot, sc.cast_order.target),
                      request=item_req, base_packets=base, base_offense=off, base_defense=dfn,
                      hp=s.hp, max_hp=s.max_hp, shields=s.shields, status=s.status, kills=s.prev.kills,
                      holder_stats=sc.static, cc=cc_items, ev=ev,
                      main_capacity=cfg.layout.packet_capacity, follow_up_capacity=cfg.layout.follow_up_capacity)
    hp, max_hp, shields, status = out.hp, out.max_hp, out.shields, out.status
    kits, k_dmg = K.on_damage(kits, kctx, units, out.report)
    return s, sc._replace(out=out, kdef=kdef, k_dmg=k_dmg, cc_now=cc_now, cc_items=cc_items, hp=hp, max_hp=max_hp,
                          shields=shields, status=status, kits=kits)
