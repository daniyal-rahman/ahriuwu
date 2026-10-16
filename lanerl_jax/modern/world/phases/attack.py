"""Phase 6, ATTACK: the basic-attack machine, crits, attack packets and missiles, ward hits, kit
on-attack/on-hit, Crystalline Overgrowth, Minion Pushing and jungle combat effects. Writes ``s.obj``."""
from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from ... import champions as K
from ... import mechanics as M
from ...champions.core import merge_out
from ...core import damage as D
from ...core import types as W
from ...core.arrays import first_true
from ...items import effects as IE
from ...items import inventory as I
from ...items.effects.core import Attack, Owned
from ...jungle import camps as J
from ...jungle import objectives as OBJ
from ...lane import ai as LA
from ...lane import minions as MM
from .. import units as U
from .. import views as V
from ..config import N_CHAMPIONS, WorldConfig
from ..scratch import TickScratch
from ..state import ModernOrders, ModernState

CAST_ID_STRIDE = 256        # attack cast ids tick*256+unit+1 stay < 2^30 (K.KIT_ID_BASE) for 2^22 ticks (~38.8 h)


def _pack_hits(hits, *metadata):
    """Keep each holder's landings in order, without a shared per-holder hit slot."""
    h, c = hits.hit.shape
    idx, count = jax.vmap(lambda m: first_true(m, h), in_axes=1, out_axes=(1, 0))(hits.hit)
    take = lambda a: a[jnp.minimum(idx, h - 1), jnp.arange(c)[None, :]]
    hits = jax.tree.map(take, hits)
    valid = jnp.arange(h)[:, None] < count[None, :]
    hits = hits._replace(hit=hits.hit & valid, launched=hits.launched & valid)
    return hits, tuple(take(a) for a in metadata)


def _hit_targets(hits, n):
    c = hits.hit.shape[1]
    return jnp.zeros((c, n), bool).at[jnp.arange(c)[None, :], jnp.clip(hits.target, 0, n - 1)].max(hits.hit)


def _kit_hits(kits, ctx, units, hits, capacity):
    c, n = ctx.unit.shape[0], units.x.shape[0]
    at = lambda i: jax.tree.map(lambda a: a[i], hits)
    kits, first = K.on_hit(kits, ctx, units, at(0))
    # Only missile users can have several basic landings in the current roster.
    if not any(hasattr(k, "attack_missile_speed") for k in K.kits_of(kits)):
        return kits, first, jnp.int32(0)
    acc = merge_out([first], c, n)
    packets, overflow = D.compact_packets(acc.packets, capacity)
    # Adaptive Defenses is the only on-hit shield; its p_on latch allows one grant per holder.
    assert acc.shield.amount.shape[1] <= 1
    acc = acc._replace(packets=packets)

    def body(i, carry):
        kits, acc, overflow = carry
        kits, eff = K.on_hit(kits, ctx, units, at(i))
        packets, dropped = D.compact_packets(D.concat_packets(acc.packets, eff.packets), capacity)
        shield = jax.tree.map(lambda a, b: jnp.where(eff.shield.amount > 0, b, a), acc.shield, eff.shield)
        empty = jax.tree.map(lambda a: a[:, :0], first.shield)
        merged = merge_out([acc._replace(packets=D.empty_packets(0), shield=empty),
                            eff._replace(packets=D.empty_packets(0), shield=empty)], c, n)
        return kits, merged._replace(packets=packets, shield=shield), overflow + dropped

    count = jnp.max(jnp.sum(hits.launched, axis=0))
    return jax.lax.fori_loop(1, count, body, (kits, acc, overflow))


def minion_pushing(s: ModernState, cfg: WorldConfig, lane_ai, now) -> tuple[Any, Any]:
    """(N,) Minion Pushing (MINIONS §4.4, MECHANICS_AUDIT #6): lane-minion bonus damage and taken-damage divisor
    from the team's level lead and the lane's turret lead. Recomputed every tick (the client holds it 1 s)."""
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
    c, n, dt = N_CHAMPIONS, cfg.n_units, cfg.dt
    now, caps, champ, units, st, kmods = sc.now, sc.caps, sc.champ, sc.units, sc.st, sc.kmods
    kit_out, kits, kctx, ictx, mbuf = sc.kit_out, sc.kits, sc.kctx, sc.ictx, sc.mbuf
    lane_ai, jungle = sc.lane_ai, sc.jungle
    can_attack = (caps["can_attack"] & s.alive).at[:c].set(
        caps["can_attack"][:c] & s.alive[:c] & ~kmods.cannot_attack & ~sc.locked & ~sc.item_casting & ~sc.tp_lock
        & ~sc.in_dash)
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
    att, launched = M.attack_step(s.att, units, sc.desired, can_attack=can_attack, windup=windup, dt=dt, reset=reset,
                                  period=period, uncancellable=uncancel)
    started = (att.windup_left > 0) & (att_prev.windup_left <= 0)
    cancelled = (att_prev.windup_left > 0) & (att.windup_left <= 0) & ~launched
    atgt = jnp.clip(att.target, 0, n - 1)
    # Champion crit roll at launch (X-8).
    imods = IE.attack_mods(s.combat.items, Owned(I.owned_counts(champ.inventory), cfg.item_allowed), ictx,
                           U.item_units(s), att.target[:c])
    no_crit = jnp.zeros((c,), bool) if kmods.cannot_crit is None else kmods.cannot_crit
    roll = jax.random.uniform(sc.k_crit, (c,)) < st.crit_chance
    k_force = opt(kmods.force_crit, jnp.zeros((c,), bool))
    crit = launched[:c] & ~no_crit & (roll | imods.force_crit | k_force)
    crit_mult = 1.0 + (st.crit_damage - 1.0) * jnp.where(imods.force_crit, imods.crit_scale, 1.0)
    # A kit's forced crit (Fiora Bladework) sets its own multiplier; an item's forced crit wins when higher (wiki).
    k_mult = opt(kmods.crit_mult, jnp.zeros((c,), jnp.float32))
    crit_mult = jnp.where(k_force & (k_mult > 0), jnp.maximum(k_mult, jnp.where(imods.force_crit, crit_mult, 0.0)),
                          crit_mult)
    vs_struct = W.is_structure(s.kind[atgt[:c]])
    struct_raw, struct_magic = LA.T.champion_structure_attack(st.base_ad, st.bonus_ad, st.ap)
    craw = jnp.where(vs_struct, struct_raw, (st.base_ad + st.bonus_ad) * jnp.where(crit, crit_mult, 1.0))
    # A kit's modified attack damage (Jayce's Hyper Charge), crits included.
    k_dmg = opt(kmods.damage_mult, jnp.zeros((c,), jnp.float32))
    craw = craw * jnp.where(k_dmg > 0, k_dmg, 1.0)
    # Champion attack missiles: the kit's (Jayce's Mercury Cannon), else the unit's own (0 = melee).
    k_msl = K.attack_missile_speed(kits, kctx)
    missile_speed = s.missile_speed.at[:c].set(jnp.where(k_msl > 0, k_msl, s.missile_speed[:c]))
    # Melee champions deal x1.2 to turrets (towers.json melee_champion_damage_multiplier).
    vs_turret = s.kind[atgt[:c]] == W.KIND_TURRET
    craw = craw * jnp.where(vs_turret & (missile_speed[:c] <= 0), 1.2, 1.0)
    cdtype = jnp.where(vs_struct & struct_magic, D.MAGIC, D.PHYSICAL)
    assert n <= CAST_ID_STRIDE
    cast_ids = (s.tick * CAST_ID_STRIDE + jnp.arange(n)).astype(jnp.int32) + 1
    ranged_c = missile_speed[:c] > 0
    # Champion attacks on wards are ward hits, not packets or on-hit.
    on_ward = (s.kind[atgt] == W.KIND_WARD) & (att.target >= 0)
    hit_c = launched[:c] & ~ranged_c & ~on_ward[:c]
    attack = Attack(launched[:c], hit_c, att.target[:c], jnp.where(launched[:c], craw, 0.0), crit,
                    cast_id=cast_ids[:c])
    attack_launch = attack
    push_bonus, push_div = minion_pushing(s, cfg, lane_ai, now)
    lane_pk = LA.attack_packets(units, W.AttackLaunch(launched & (s.kind != W.KIND_CHAMPION), att.target,
                                                      missile_speed > 0, jnp.zeros((n,), bool), cast_ids),
                                now=now, ai=lane_ai, pushing_bonus=push_bonus, pushing_divisor=push_div)
    lane_pk = lane_pk._replace(cast_id=cast_ids)
    msl_speed = missile_speed if mbuf is None else jnp.where(mbuf.missile_speed > 0, mbuf.missile_speed,
                                                             missile_speed)
    if mbuf is not None:                                              # empowered siege minions vs structures
        t_struct = W.is_structure(s.kind[atgt])
        lane_pk = lane_pk._replace(raw=lane_pk.raw * jnp.where(t_struct & mbuf.empowered,
                                                               mbuf.siege_structure_mult, 1.0))
    extra_atk = []
    launch_all = W.AttackLaunch(launched, att.target, missile_speed > 0, jnp.zeros((n,), bool), cast_ids)
    if cfg.jungle is not None:
        j_main, j_bonus = J.monster_attack_packets(jungle, cfg.jungle, units, launch_all)
        jsl = cfg.jungle.slots
        lane_pk = lane_pk._replace(raw=lane_pk.raw.at[jsl].set(j_main.raw),
                                   dtype=lane_pk.dtype.at[jsl].set(j_main.dtype),
                                   flags=lane_pk.flags.at[jsl].set(j_main.flags))
        extra_atk.append(j_bonus)
    obj_cc = None
    if sc.so is not None:
        obj, o_raw, o_dtype, o_flags, o_extra, obj_cc = OBJ.objectives_attack(s.obj, cfg.objectives, units,
                                                                            launch_all, now=now)
        s = s._replace(obj=obj)
        esl = cfg.objectives.slots
        on = (jnp.zeros((n,), bool).at[esl].set(True)) & (o_raw > 0)
        lane_pk = lane_pk._replace(raw=jnp.where(on, o_raw, lane_pk.raw), dtype=jnp.where(on, o_dtype, lane_pk.dtype),
                                   flags=jnp.where(on, o_flags, lane_pk.flags))
        extra_atk.append(o_extra)
    ranged = launched & (msl_speed > 0)
    flags_c = jnp.full((c,), D.BASIC_ATTACK, jnp.int32) | jnp.where(crit, jnp.int32(D.PROP_CRIT), jnp.int32(0))
    # Kit split attacks (Yone's Azakana sword): that share of a melee attack is a second, magic packet.
    split = jnp.where(cdtype == D.PHYSICAL, opt(kmods.magic_split, jnp.zeros((c,), jnp.float32)), 0.0)
    split_pk = D.packets(launched[:c] & ~ranged_c & ~on_ward[:c] & (att.target[:c] >= 0) & (split > 0), jnp.arange(c),
                         jnp.maximum(att.target[:c], 0), craw * split, D.MAGIC, flags_c, cast_id=cast_ids[:c])
    extra_atk.append(split_pk)
    # ... and a true share (Camille's Precision Protocol recast), structures excepted.
    tsplit = jnp.where(vs_struct, 0.0, opt(kmods.true_split, jnp.zeros((c,), jnp.float32)))
    true_pk = D.packets(launched[:c] & ~ranged_c & ~on_ward[:c] & (att.target[:c] >= 0) & (tsplit > 0), jnp.arange(c),
                        jnp.maximum(att.target[:c], 0), craw * tsplit, D.TRUE, flags_c, cast_id=cast_ids[:c])
    extra_atk.append(true_pk)
    raw_all = lane_pk.raw.at[:c].set(craw * jnp.where(ranged_c, 1.0, 1.0 - split - tsplit))
    dtype_all = lane_pk.dtype.at[:c].set(cdtype)
    flags_all = lane_pk.flags.at[:c].set(flags_c)
    lay = cfg.layout
    w0 = lay.ward0
    missiles, m_over = M.spawn_missiles(s.missiles, ranged & ~on_ward, units, att.target, raw_all, dtype_all,
                                        flags_all, msl_speed, cast_ids, jnp.zeros((n,), bool).at[:c].set(crit))
    missiles, arrive = M.advance_missiles(missiles, units, dt)
    direct = D.packets(launched & ~ranged & (att.target >= 0) & ~on_ward, jnp.arange(n), jnp.maximum(att.target, 0),
                       raw_all, dtype_all, flags_all, cast_id=cast_ids)
    hit_ward = launched & on_ward & (jnp.arange(n) < c)
    w_slot = jnp.clip(att.target - w0, 0, 2 * W.MAX_WARDS_PER_TEAM - 1)
    ward_hits = jnp.zeros((2 * W.MAX_WARDS_PER_TEAM,), jnp.int32).at[w_slot].add(hit_ward.astype(jnp.int32))
    ward_hitter = jnp.full((2 * W.MAX_WARDS_PER_TEAM,), -1, jnp.int32).at[w_slot].max(
        jnp.where(hit_ward, jnp.arange(n), -1).astype(jnp.int32))
    arrived = D.packets(arrive, missiles.src, missiles.dst, missiles.raw, missiles.dtype, missiles.flags,
                        cast_id=missiles.cast_id)
    arrivals = arrive[:, None] & (missiles.src[:, None] == jnp.arange(c)[None, :])
    mfield = lambda a: jnp.broadcast_to(a[:, None], arrivals.shape)
    fields = (jnp.concatenate([attack.hit[None, :], arrivals]),
              jnp.concatenate([attack.target[None, :], mfield(missiles.dst)]),
              jnp.concatenate([attack.raw[None, :], mfield(missiles.raw)]),
              jnp.concatenate([attack.is_crit[None, :], mfield(missiles.crit)]))
    ids = jnp.concatenate([cast_ids[None, :c], mfield(missiles.cast_id)])
    basic = Attack(jnp.zeros_like(fields[0]), *fields, cast_id=ids)
    ranged_hit = jnp.concatenate([ranged_c[None, :], jnp.ones_like(arrivals)])
    basic, (ranged_hit,) = _pack_hits(basic, ranged_hit)
    launch_c = W.AttackLaunch(attack_launch.launched, attack_launch.target, ranged_c, crit, cast_ids[:c])
    kits, k_att = K.on_attack(kits, kctx, units, launch_c)
    kits, k_hit, hit_over = _kit_hits(kits, kctx, units,
                                    W.AttackLaunch(basic.hit, basic.target, ranged_hit, basic.is_crit, basic.cast_id),
                                    cfg.layout.packet_capacity)
    champ_hit = jnp.zeros((n, n), bool).at[:c].set(_hit_targets(basic, n))
    towers, og_pk = LA.overgrowth_packets(s.towers, units, champ_hit, now=now,
                                          team_level=V.decimal_team_level(s))
    kit_all = merge_out([kit_out, k_att, k_hit], c, n)
    oh = kit_all.on_hit_target
    ability = Attack((oh >= 0) & kit_all.on_attack_proc, oh >= 0, oh,
                     kit_all.on_hit_raw, kit_all.on_hit_crit, cast_id=kit_all.on_hit_cast_id)
    hits = jax.tree.map(lambda a, b: jnp.concatenate([a, b[None, :]]), basic, ability)
    hits, _ = _pack_hits(hits)
    attack = jax.tree.map(lambda a: a[0], hits)
    jfx = None
    if cfg.jungle is not None:
        # Scorchclaw reads last tick's damage matrix.
        dm_cc = s.prev.damage_matrix[:c, :c] & (s.team[:c][:, None] != s.team[:c][None, :])
        dmg_champ = jnp.where(jnp.any(dm_cc, axis=1), jnp.argmax(dm_cc, axis=1), -1).astype(jnp.int32)
        jungle, jfx = J.combat_effects(jungle, cfg.jungle, units, ictx, attack_hit=attack.hit,
                                       attack_target=attack.target, damaged_champion=dmg_champ,
                                       landed_hits=_hit_targets(hits, n),
                                       in_brush=V.in_brush(cfg, s.x[:c], s.y[:c], s.terrain_variant))
    return s, sc._replace(att=att, launched=launched, started=started, cancelled=cancelled, reset=reset,
                          attack=attack, attack_launch=attack_launch, hits=hits, hit_over=hit_over,
                          missiles=missiles, m_over=m_over,
                          direct=direct, arrived=arrived,
                          og_pk=og_pk, extra_atk=extra_atk, obj_cc=obj_cc, ward_hits=ward_hits,
                          ward_hitter=ward_hitter, kits=kits, towers=towers, kit_all=kit_all, jfx=jfx,
                          jungle=jungle, units=units)
