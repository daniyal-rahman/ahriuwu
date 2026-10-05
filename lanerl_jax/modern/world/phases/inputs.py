"""Phase 1, INPUT: fog-filtered orders, walk-in/buffered casts, attack-move, item stasis, skill points,
minion/camp spawns and this tick's walkable terrain."""
from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from ... import champions as KC
from ... import economy as E
from ... import mechanics as M
from ...core import types as W
from ...jungle import camps as J
from ...lane import ai as LA
from ...map import dynamic_terrain as DTR
from ...map.rift import terrain_pair
from .. import units as U
from .. import views as V
from ..config import N_CHAMPIONS, WorldConfig
from ..scratch import TickScratch
from ..state import AttackMove, ModernOrders, ModernState, QueuedCast, no_queued_cast

CAST_BUFFER_S = 0.5          # casts during a lockout / just before a cooldown ends are held this long (U: guess)
CAST_RANGE_SLACK = 5.0       # walk-in casting stops this far inside the spell's range


def spawn_minions(s: ModernState, cfg: WorldConfig, now) -> ModernState:
    """MINIONS §2 for every enabled lane (``lane.ai.spawn_lane_minions``)."""
    spawn, wr, _ = LA.spawn_lane_minions(s.spawn, s.towers, s.kind, s.alive, now=now, slot0=cfg.layout.minion0,
                                         per_lane=W.MINION_SLOTS_PER_LANE, lanes=cfg.layout.lanes)
    st = wr.stats
    s = U.write_units(s._replace(spawn=spawn), W.UnitWrite(
        mask=wr.pick, new=wr.pick, kind=W.KIND_MINION, sub=wr.sub, team=wr.team, x=wr.x, y=wr.y, hp=st.max_hp,
        max_hp=st.max_hp, radius=st.radius, armor=st.armor, magic_resist=st.magic_resist,
        attack_damage=st.attack_damage, attack_range=st.attack_range, attack_speed=st.attack_speed,
        move_speed=st.move_speed, windup=st.windup, missile_speed=st.missile_speed, bounty_gold=st.gold,
        bounty_xp=st.xp, bounty_level=jnp.max(s.econ.level)), now)
    if cfg.jungle is not None:
        jst, w = J.spawn_step(s.jungle, cfg.jungle, now=now, champion_level=s.econ.level)
        s = U.write_units(s, J.unit_write(cfg.jungle, w, cfg.n_units), now)
        s = s._replace(jungle=J.latch_pets(jst, V.owned_items(s.champ.inventory)))
    return s


def skill_up(s: ModernState, cfg: WorldConfig, orders: ModernOrders) -> Any:
    """Spend available skill points: chosen slot, else the champion's default order."""
    ranks = s.champ.ranks
    level = s.econ.level
    points = E.skill_points(level) + s.champ.bonus_points
    spent = jnp.sum(ranks, axis=1)
    order = jnp.asarray([list(V.skill_order(cfg, c)) + [0] * (20 - len(V.skill_order(cfg, c)))
                         for c in range(N_CHAMPIONS)], jnp.int32)
    auto = jnp.take_along_axis(order, jnp.clip(spent, 0, 19)[:, None], axis=1)[:, 0]
    auto_on = jnp.asarray([lo.auto_skill for lo in cfg.loadouts])
    slot = jnp.where(orders.level_up >= 0, orders.level_up, auto)
    cap = jnp.where(slot == 3, E.max_rank(level, ultimate=True), E.max_rank(level))
    slot = jnp.clip(slot, 0, 3)
    cur = jnp.take_along_axis(ranks, slot[:, None], axis=1)[:, 0]
    ok = (points > spent) & (cur < cap) & ((orders.level_up >= 0) | auto_on)
    return ranks + (jnp.arange(4)[None, :] == slot[:, None]) * ok[:, None]


def queue_casts(s: ModernState, cfg: WorldConfig, orders: ModernOrders, champ, seen, new_order, now):
    """Walk-in and buffered casting (wiki Targeting / Cast time; MECHANICS_AUDIT #4/#9).

    Returns ``(orders, queued, chase, walked_in)``: ``orders`` with this tick's cast (an incoming cast
    that can go now, else a queued one that became possible), the queue to keep, champions walking
    into range of a queued target, and champions whose walk-in cast fired this tick (ends the walk). A move, attack,
    stop or attack-move order clears the queue; so does a new cast (it replaces it)."""
    c, n = N_CHAMPIONS, cfg.n_units
    ar = jnp.arange(c)
    rng = KC.unit_target_ranges(cfg.champion_ids)                                   # (C, 4)

    def needs(slot, target):
        """(far, blocked) for a cast of ``slot`` at ``target`` now."""
        sl = jnp.clip(slot, 0, 3)
        r = rng[ar, sl]
        t = jnp.clip(target, 0, n - 1)
        d = jnp.sqrt((s.x[:c] - s.x[t]) ** 2 + (s.y[:c] - s.y[t]) ** 2)
        far = (slot >= 0) & (target >= 0) & (r > 0) & (d > r + s.radius[t] - CAST_RANGE_SLACK)
        cd = champ.cooldowns[ar, sl]
        blocked = (slot >= 0) & ((now < champ.cast_lock_until) | (now < champ.item_cast_until)
                                 | ((cd > 0) & (cd <= CAST_BUFFER_S)))
        return far, blocked

    incoming = orders.cast_slot >= 0
    far_in, blocked_in = needs(orders.cast_slot, orders.cast_target)
    hold_in = incoming & (far_in | blocked_in)
    q = champ.queued_cast
    keep = (q.slot >= 0) & ~incoming & ~new_order & (now < q.until) \
        & ((q.target < 0) | (seen(q.target) & s.alive[jnp.clip(q.target, 0, n - 1)]))
    q = QueuedCast(*(jnp.where(keep, a, b) for a, b in zip(q, no_queued_cast(c))))
    q = QueuedCast(jnp.where(hold_in, orders.cast_slot, q.slot).astype(jnp.int32),
                   jnp.where(hold_in, orders.cast_target, q.target).astype(jnp.int32),
                   jnp.where(hold_in, orders.cast_x, q.x), jnp.where(hold_in, orders.cast_y, q.y),
                   jnp.where(hold_in, jnp.where(far_in, jnp.inf, now + CAST_BUFFER_S), q.until))
    far_q, blocked_q = needs(q.slot, q.target)
    fire = (q.slot >= 0) & ~hold_in & ~far_q & ~blocked_q
    go_now = incoming & ~hold_in
    orders = orders._replace(
        cast_slot=jnp.where(go_now, orders.cast_slot, jnp.where(fire, q.slot, -1)).astype(jnp.int32),
        cast_target=jnp.where(go_now, orders.cast_target, jnp.where(fire, q.target, orders.cast_target)).astype(jnp.int32),
        cast_x=jnp.where(go_now, orders.cast_x, jnp.where(fire, q.x, orders.cast_x)),
        cast_y=jnp.where(go_now, orders.cast_y, jnp.where(fire, q.y, orders.cast_y)))
    chase = (q.slot >= 0) & far_q & ~fire
    walked_in = fire & jnp.isinf(q.until)                       # a walk-in cast arrived and fired
    q = QueuedCast(*(jnp.where(fire, b, a) for a, b in zip(q, no_queued_cast(c))))
    return orders, q, chase, walked_in


def run(s: ModernState, orders: ModernOrders, cfg: WorldConfig,
           sc: TickScratch) -> tuple[ModernState, ModernOrders, TickScratch]:
    """1. INPUT: fog-filtered orders, attack-move, stasis, skill points, spawns, this tick's terrain.

    Writes ``s.champ`` (orders, ranks), ``s.amove``, spawns, champion ``targetable``; returns the
    fog-filtered orders."""
    c, n = N_CHAMPIONS, cfg.n_units
    now = sc.now
    champ = s.champ
    # Fog: a champion can only target what its team sees; a target that enters fog is dropped.
    vis_c = s.visible[s.team[:c]]                                                     # (C, N)
    seen = lambda u: (u >= 0) & vis_c[jnp.arange(c), jnp.clip(u, 0, n - 1)]          # noqa: E731
    orders = orders._replace(attack=jnp.where(seen(orders.attack), orders.attack, -1),
                             cast_target=jnp.where(seen(orders.cast_target), orders.cast_target, -1),
                             summoner_target=jnp.where(seen(orders.summoner_target), orders.summoner_target, -1))
    zb = jnp.zeros((c,), bool)
    am_req = zb if orders.attack_move is None else orders.attack_move
    # Item stasis (Zhonya's / Stopwatch): no orders take effect while in stasis (actives).
    in_stasis = now < s.combat.items.actives.stasis_until
    am_req = am_req & ~in_stasis
    kept = jnp.where(seen(champ.attack_order), champ.attack_order, -1)
    new_order = orders.stop | orders.move | (orders.attack >= 0) | am_req
    attack_order = jnp.where(orders.stop | orders.move | am_req, -1, jnp.where(orders.attack >= 0, orders.attack, kept))
    moving = jnp.where(orders.stop | (orders.attack >= 0) | am_req, False, orders.move | champ.moving)
    goal = jnp.where(orders.move[:, None], jnp.stack([orders.move_x, orders.move_y], -1), champ.move_goal)
    # A live attack target that enters fog: walk to where it was last seen (wiki Basic attack;
    # MECHANICS_AUDIT #10) instead of standing still. The order itself is dropped.
    t_old = jnp.clip(champ.attack_order, 0, n - 1)
    lost = (champ.attack_order >= 0) & ~seen(champ.attack_order) & s.alive[t_old] & ~new_order
    # Remember where this tick's target is seen (including a target just ordered), for later ticks.
    t_now = jnp.clip(attack_order, 0, n - 1)
    seen_at = jnp.where(seen(attack_order)[:, None], jnp.stack([s.x[t_now], s.y[t_now]], -1),
                        champ.target_seen_at)
    moving = moving | lost
    goal = jnp.where(lost[:, None], champ.target_seen_at, goal)
    orders, queued, cast_chase, cast_fired = queue_casts(s, cfg, orders, champ, seen, new_order, now)
    moving = jnp.where(cast_chase, True, jnp.where(cast_fired, False, moving))
    attack_order = jnp.where(cast_chase, -1, attack_order)          # "move to cast" replaces an attack order
    t_q = jnp.clip(queued.target, 0, n - 1)
    goal = jnp.where(cast_chase[:, None], jnp.stack([s.x[t_q], s.y[t_q]], -1), goal)
    amove = s.amove
    amove = AttackMove(active=jnp.where(new_order, am_req, amove.active),
                       x=jnp.where(am_req, orders.move_x, amove.x), y=jnp.where(am_req, orders.move_y, amove.y),
                       held=jnp.where(new_order, -1, amove.held), held_seq=amove.held_seq)
    ranks = skill_up(s, cfg, orders)
    champ = champ._replace(attack_order=attack_order.astype(jnp.int32), moving=moving, move_goal=goal, ranks=ranks,
                           target_seen_at=seen_at, queued_cast=queued)
    s = s._replace(champ=champ, amove=amove)
    s = spawn_minions(s, cfg, now)
    caps = M.capabilities(s.cc, now)
    stasis_n = jnp.zeros((n,), bool).at[:c].set(in_stasis)
    s = s._replace(targetable=s.targetable.at[:c].set(~in_stasis))
    caps = {k: (v & ~stasis_n if k.startswith("can_") else v) for k, v in caps.items()}
    # Terrain this tick: Elemental Rift / Baron-pit variant, then destroyed-structure pads (policy).
    terrain = cfg.terrain
    if cfg.rift is not None:
        terrain = terrain_pair(cfg.rift, s.terrain_variant)
    if cfg.footprints is not None:
        terrain = DTR.walkable_masks(terrain, cfg.footprints, s.alive, DTR.release_mask(cfg.unit_kind))
    return s, orders, sc._replace(champ=champ, vis_c=vis_c, in_stasis=in_stasis, caps=caps, terrain=terrain,
                                  attack_order=attack_order, moving=moving, goal=goal, amove=amove)
