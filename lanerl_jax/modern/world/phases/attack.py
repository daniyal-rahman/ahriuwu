"""Phase 6, ATTACK: the basic-attack machine, crits, attack packets and missiles, ward hits, kit
on-attack/on-hit, Crystalline Overgrowth, Minion Pushing and jungle combat effects."""
from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from ... import champions as K
from ... import mechanics as M
from ...champions.core import merge_out
from ...core import damage as D
from ...core import types as W
from ...items import effects as IE
from ...items.effects.core import Attack
from ...jungle import camps as J
from ...jungle import objectives as OBJ
from ...lane import ai as LA
from ...lane import minions as MM
from .. import views as V
from ..config import N_CHAMPIONS, WorldConfig
from ..config import layout as MW_layout
from ..scratch import TickScratch
from ..state import ModernOrders, ModernState

CAST_ID_STRIDE = 256        # attack cast ids tick*256+unit+1 stay < 2^30 (K.KIT_ID_BASE) for 2^22 ticks (~38.8 h)


def minion_pushing(s: ModernState, cfg: WorldConfig, lane_ai, now) -> tuple[Any, Any]:
    """(N,) Minion Pushing (MINIONS §4.4, MECHANICS_AUDIT #6): each lane minion's team bonus damage
    and the divisor on minion damage it takes, from its team's level lead (one champion per team, so
    the champion's level) and its lane's turret lead. Recomputed every tick (the client holds it for
    1 s, so a level-up or turret kill applies up to 1 s earlier here)."""
    n = s.kind.shape[0]
    level = s.econ.level.astype(jnp.float32)
    team = jnp.clip(s.team, 0, 1)
    turret = (s.kind == W.KIND_TURRET) & s.alive
    alive_t = jnp.stack([jnp.stack([jnp.sum(turret & (s.team == t) & (cfg.unit_lane == l)) for l in range(3)])
                         for t in (0, 1)]).astype(jnp.float32)                                     # (team, lane)
    lane = jnp.clip(lane_ai.lane, 0, 2)
    lvl_adv = level[team] - level[1 - team]
    tow_adv = alive_t[team, lane] - alive_t[1 - team, lane]
    bonus, div = MM.minion_pushing_modifiers(team_level_advantage=lvl_adv, lane_turret_advantage=tow_adv,
                                             time_s=jnp.floor(now))
    minion = (s.kind == W.KIND_MINION) & (lane_ai.lane >= 0)
    return jnp.where(minion, bonus, 0.0), jnp.where(minion, div, 1.0) * jnp.ones((n,), jnp.float32)


def run(s: ModernState, orders: ModernOrders, cfg: WorldConfig, sc: TickScratch) -> tuple[ModernState, TickScratch]:
    """6. ATTACK: attack machine, crits, attack packets, missiles, ward hits, kit on-attack/on-hit,
    Overgrowth, jungle combat effects. Writes ``s.obj``."""
    c, n, dt = N_CHAMPIONS, cfg.n_units, cfg.dt
    now, caps, champ, units, st, kmods = sc.now, sc.caps, sc.champ, sc.units, sc.st, sc.kmods
    locked, item_casting, tp_lock, in_dash = sc.locked, sc.item_casting, sc.tp_lock, sc.in_dash
    kit_out, kits, kctx, ictx, mbuf, so = sc.kit_out, sc.kits, sc.kctx, sc.ictx, sc.mbuf, sc.so
    lane_ai, jungle, desired, k_crit = sc.lane_ai, sc.jungle, sc.desired, sc.k_crit
    can_attack = (caps["can_attack"] & s.alive).at[:c].set(
        caps["can_attack"][:c] & s.alive[:c] & ~kmods.cannot_attack & ~locked & ~item_casting & ~tp_lock
        & ~in_dash)
    opt = lambda v, d: d if v is None else v                          # noqa: E731
    k_wind, k_per_ = opt(kmods.windup, jnp.zeros((c,))), opt(kmods.period, jnp.zeros((c,)))
    windup = s.windup.at[:c].set(jnp.where(k_wind > 0, k_wind, st.attack_windup))
    period = jnp.zeros((n,), jnp.float32).at[:c].set(k_per_)
    uncancel = jnp.zeros((n,), bool).at[:c].set(opt(kmods.uncancellable, jnp.zeros((c,), bool)))
    reset = jnp.zeros((n,), bool).at[:c].set(kit_out.attack_reset | kmods.attack_reset | champ.reset_next)
    if mbuf is not None:                                              # Hand of Baron empowered minions
        units = units._replace(attack_range=units.attack_range + mbuf.bonus_range,
                               attack_damage=units.attack_damage + mbuf.bonus_ad,
                               attack_speed=units.attack_speed * jnp.where(mbuf.empowered, mbuf.attack_speed_mult,
                                                                           1.0))
    att_prev = s.att
    att, launched = M.attack_step(s.att, units, desired, can_attack=can_attack, windup=windup, dt=dt, reset=reset,
                                  period=period, uncancellable=uncancel)
    started = (att.windup_left > 0) & (att_prev.windup_left <= 0)
    cancelled = (att_prev.windup_left > 0) & (att.windup_left <= 0) & ~launched
    atgt = jnp.clip(att.target, 0, n - 1)
    # Champions: crit roll at launch (X-8), Garen Q's spell attack cannot crit.
    imods = IE.attack_mods(s.combat.items, V.owned_items(champ.inventory), ictx, V.item_units(s), att.target[:c])
    no_crit = jnp.zeros((c,), bool) if kmods.cannot_crit is None else kmods.cannot_crit
    roll = jax.random.uniform(k_crit, (c,)) < st.crit_chance
    crit = launched[:c] & ~no_crit & (roll | imods.force_crit)
    crit_mult = 1.0 + (st.crit_damage - 1.0) * jnp.where(imods.force_crit, imods.crit_scale, 1.0)
    vs_struct = W.is_structure(s.kind[atgt[:c]])
    struct_raw, struct_magic = LA.T.champion_structure_attack(st.base_ad, st.bonus_ad, st.ap)
    craw = jnp.where(vs_struct, struct_raw, (st.base_ad + st.bonus_ad) * jnp.where(crit, crit_mult, 1.0))
    # Melee champions deal x1.2 to turrets (towers.json melee_champion_damage_multiplier; a
    # multiplicative factor, so applying it to raw is equivalent to post-mitigation).
    vs_turret = s.kind[atgt[:c]] == W.KIND_TURRET
    craw = craw * jnp.where(vs_turret & (s.missile_speed[:c] <= 0), 1.2, 1.0)
    cdtype = jnp.where(vs_struct & struct_magic, D.MAGIC, D.PHYSICAL)
    assert n <= CAST_ID_STRIDE
    cast_ids = (s.tick * CAST_ID_STRIDE + jnp.arange(n)).astype(jnp.int32) + 1   # unique per (tick, unit)
    ranged_c = s.missile_speed[:c] > 0
    # Champion basic attacks on wards deal 1 hit each (wards), not damage packets or on-hit.
    on_ward = (s.kind[atgt] == W.KIND_WARD) & (att.target >= 0)
    hit_c = launched[:c] & ~ranged_c & ~on_ward[:c]
    attack = Attack(launched[:c], hit_c, att.target[:c], jnp.where(launched[:c], craw, 0.0), crit)
    push_bonus, push_div = minion_pushing(s, cfg, lane_ai, now)
    lane_pk = LA.attack_packets(units, W.AttackLaunch(launched & (s.kind != W.KIND_CHAMPION), att.target,
                                                      s.missile_speed > 0, jnp.zeros((n,), bool), cast_ids),
                                now=now, ai=lane_ai, pushing_bonus=push_bonus, pushing_divisor=push_div)
    lane_pk = lane_pk._replace(cast_id=cast_ids)
    msl_speed = s.missile_speed if mbuf is None else jnp.where(mbuf.missile_speed > 0, mbuf.missile_speed,
                                                               s.missile_speed)
    if mbuf is not None:                                              # empowered siege minions vs structures
        t_struct = W.is_structure(s.kind[atgt])
        lane_pk = lane_pk._replace(raw=lane_pk.raw * jnp.where(t_struct & mbuf.empowered,
                                                               mbuf.siege_structure_mult, 1.0))
    extra_atk = []
    launch_all = W.AttackLaunch(launched, att.target, s.missile_speed > 0, jnp.zeros((n,), bool), cast_ids)
    if cfg.jungle is not None:
        j_main, j_bonus = J.monster_attack_packets(jungle, cfg.jungle, units, launch_all)
        jsl = slice(cfg.jungle.monster0, cfg.jungle.monster0 + cfg.jungle.n_slots)
        lane_pk = lane_pk._replace(raw=lane_pk.raw.at[jsl].set(j_main.raw), dtype=lane_pk.dtype.at[jsl].set(j_main.dtype),
                                   flags=lane_pk.flags.at[jsl].set(j_main.flags))
        extra_atk.append(j_bonus)
    obj_cc = None
    if so is not None:
        obj, o_raw, o_dtype, o_flags, o_extra, obj_cc = OBJ.objectives_attack(s.obj, cfg.objectives, units,
                                                                            launch_all, now=now)
        s = s._replace(obj=obj)
        esl = slice(cfg.objectives.slot0, cfg.objectives.slot0 + 8)
        on = (jnp.zeros((n,), bool).at[esl].set(True)) & (o_raw > 0)
        lane_pk = lane_pk._replace(raw=jnp.where(on, o_raw, lane_pk.raw), dtype=jnp.where(on, o_dtype, lane_pk.dtype),
                                   flags=jnp.where(on, o_flags, lane_pk.flags))
        extra_atk.append(o_extra)
    ranged = launched & (msl_speed > 0)
    flags_c = jnp.full((c,), D.BASIC_ATTACK, jnp.int32) | jnp.where(crit, jnp.int32(D.PROP_CRIT), jnp.int32(0))
    raw_all = lane_pk.raw.at[:c].set(craw)
    dtype_all = lane_pk.dtype.at[:c].set(cdtype)
    flags_all = lane_pk.flags.at[:c].set(flags_c)
    lay = MW_layout()
    w0 = lay["ward0"]
    missiles, m_over = M.spawn_missiles(s.missiles, ranged & ~on_ward, units, att.target, raw_all, dtype_all,
                                        flags_all, msl_speed, cast_ids, jnp.zeros((n,), bool).at[:c].set(crit))
    missiles, arrive = M.advance_missiles(missiles, units, dt)
    direct = D.packets(launched & ~ranged & (att.target >= 0) & ~on_ward, jnp.arange(n), jnp.maximum(att.target, 0),
                       raw_all, dtype_all, flags_all, cast_id=cast_ids)
    hit_ward = launched & on_ward & (jnp.arange(n) < c)               # champions only hit wards
    w_slot = jnp.clip(att.target - w0, 0, 2 * W.MAX_WARDS_PER_TEAM - 1)
    ward_hits = jnp.zeros((2 * W.MAX_WARDS_PER_TEAM,), jnp.int32).at[w_slot].add(hit_ward.astype(jnp.int32))
    ward_hitter = jnp.full((2 * W.MAX_WARDS_PER_TEAM,), -1, jnp.int32).at[w_slot].max(
        jnp.where(hit_ward, jnp.arange(n), -1).astype(jnp.int32))
    arrived = D.packets(arrive, missiles.src, missiles.dst, missiles.raw, missiles.dtype, missiles.flags,
                        cast_id=missiles.cast_id)
    # Champion ranged hits arriving now are on-hit for items/kits too.
    arrive_c = jnp.zeros((c,), bool).at[jnp.clip(missiles.src, 0, c - 1)].max(arrive & (missiles.src < c))
    attack = attack._replace(hit=attack.hit | arrive_c)
    launch_c = W.AttackLaunch(attack.launched, attack.target, ranged_c, crit, cast_ids[:c])
    kits, k_att = K.on_attack(kits, kctx, units, launch_c)
    kits, k_hit = K.on_hit(kits, kctx, units, launch_c._replace(launched=attack.hit))
    # Crystalline Overgrowth consumed by champion basic attacks on turrets.
    champ_hit = jnp.zeros((n, n), bool).at[:c].set((attack.hit[:, None]
                                                    & (jnp.arange(n)[None, :] == attack.target[:, None])))
    towers, og_pk = LA.overgrowth_packets(s.towers, units, champ_hit, now=now,
                                          team_level=V.decimal_team_level(s))
    kit_all = merge_out([kit_out, k_att, k_hit], c, n)
    jfx = None
    if cfg.jungle is not None:
        # Scorchclaw: enemy champion damaged (last tick's damage_matrix, 1 tick lag); Gustwalker: in brush.
        dm_cc = s.damage_matrix[:c, :c] & (s.team[:c][:, None] != s.team[:c][None, :])
        dmg_champ = jnp.where(jnp.any(dm_cc, axis=1), jnp.argmax(dm_cc, axis=1), -1).astype(jnp.int32)
        jungle, jfx = J.combat_effects(jungle, cfg.jungle, units, ictx, attack_hit=attack.hit,
                                       attack_target=attack.target, damaged_champion=dmg_champ,
                                       in_brush=V.in_brush(cfg, s.x[:c], s.y[:c], s.terrain_variant))
    return s, sc._replace(att=att, launched=launched, started=started, cancelled=cancelled, reset=reset,
                          attack=attack, missiles=missiles, m_over=m_over, direct=direct, arrived=arrived,
                          og_pk=og_pk, extra_atk=extra_atk, obj_cc=obj_cc, ward_hits=ward_hits,
                          ward_hitter=ward_hitter, kits=kits, towers=towers, kit_all=kit_all, jfx=jfx,
                          jungle=jungle, units=units)
