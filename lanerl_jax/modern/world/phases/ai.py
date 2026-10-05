"""Phase 4, AI: structures, minion/turret/monster targets and goals, champion attack targets."""
from __future__ import annotations

import jax.numpy as jnp

from ...core import types as W
from ...jungle import camps as J
from ...jungle import objectives as OBJ
from ...lane import ai as LA
from .. import views as V
from ..config import N_CHAMPIONS, WorldConfig
from ..scratch import TickScratch
from ..state import ModernOrders, ModernState


def run(s: ModernState, orders: ModernOrders, cfg: WorldConfig, sc: TickScratch) -> tuple[ModernState, TickScratch]:
    """4. AI: structure tick, minion/turret/monster targets and goals, champion targets (idle acquisition,
    attack-move).

    Writes structure/monster ``s.hp``/``s.alive``/``s.targetable``/``s.kind`` and ``s.obj``; ``towers`` and
    ``lane_ai`` stay in the scratch until MOVE."""
    c, n, dt = N_CHAMPIONS, cfg.n_units, cfg.dt
    now, units, jungle, so, champ = sc.now, sc.units, sc.jungle, sc.so, sc.champ
    attack_order, moving, amove, goal = sc.attack_order, sc.moving, sc.amove, sc.goal
    reach, vis_c = sc.reach, sc.vis_c
    towers = LA.turret_tick(s.towers, units, now=now, dt=jnp.float32(dt))
    hp_t, alive_t, targ_t = LA.structure_unit_view(towers, units)
    s = s._replace(hp=hp_t, alive=alive_t, targetable=targ_t)
    units = V.units_view(s)
    champ_vs_champ = s.damage_matrix & (s.kind == W.KIND_CHAMPION)[:, None] & (s.kind == W.KIND_CHAMPION)[None, :]
    lane_ai, desired, mgoal, stop = LA.select_targets(s.lane_ai, units, s.att, now=now, dt=jnp.float32(dt),
                                                      champion_attacked_champion=champ_vs_champ,
                                                      damage_events=s.damage_matrix, visible=s.visible)
    # Monsters: jungle camps and epic objectives choose their own targets and goals.
    mai = None
    m_goal = jnp.stack([s.x, s.y], -1)
    m_speed = jnp.zeros((n,), jnp.float32)
    m_move = jnp.zeros((n,), bool)
    if cfg.jungle is not None:
        jungle, mai = J.monster_ai(jungle, cfg.jungle, units, s.att, now=now, dt=jnp.float32(dt),
                                   damage_events=s.damage_matrix)
        j0, jn = cfg.jungle.monster0, cfg.jungle.n_slots
        jsl = slice(j0, j0 + jn)
        desired = desired.at[jsl].set(mai.desired)
        m_goal = m_goal.at[jsl].set(jnp.stack([mai.goal_x, mai.goal_y], -1))
        m_speed, m_move = m_speed.at[jsl].set(mai.move_speed), m_move.at[jsl].set(mai.moving)
        live_j = s.alive[jsl] & ~mai.despawn
        s = s._replace(hp=s.hp.at[jsl].set(jnp.where(live_j, jnp.minimum(s.hp[jsl] + mai.heal, s.max_hp[jsl]),
                                                     s.hp[jsl])),
                       alive=s.alive.at[jsl].set(live_j),
                       kind=s.kind.at[jsl].set(jnp.where(mai.despawn, W.KIND_NONE, s.kind[jsl])),
                       targetable=s.targetable.at[jsl].set(mai.targetable & live_j))
    if so is not None:
        e0 = cfg.objectives.slot0
        esl = slice(e0, e0 + 8)
        desired = desired.at[esl].set(jnp.where(so.can_attack, so.desired, -1))
        m_goal = m_goal.at[esl].set(so.goal)
        m_speed, m_move = m_speed.at[esl].set(so.move_speed), m_move.at[esl].set(so.move_active)
        obj, mbuf = OBJ.baron_minion_buffs(s.obj, cfg.objectives, V.units_view(s), now=now)
        s = s._replace(obj=obj)
    else:
        mbuf = None
    units = V.units_view(s)

    # Champions: ordered target, attack-move, or idle auto-acquisition (LA.idle_acquire; chases).
    t_ok = (attack_order >= 0) & s.alive[jnp.clip(attack_order, 0, n - 1)]
    acq = LA.champion_acquisition_range(reach, cfg.champion_base.attack_range)
    unit_c = jnp.arange(c, dtype=jnp.int32)
    auto = LA.idle_acquire(units, unit_c, vis_c, acq)
    idle = ~t_ok & ~moving & ~amove.active & s.alive[:c] & (auto >= 0)
    attack_order = jnp.where(idle, auto, attack_order).astype(jnp.int32)
    t_ok = t_ok | idle
    am = LA.attack_move_step(amove.active & s.alive[:c], amove.x, amove.y, amove.held, amove.held_seq, units,
                             unit_c, vis_c, acq)
    amove = amove._replace(active=am.active, held=am.target, held_seq=am.target_seq)
    am_tgt = am.active & (am.target >= 0) & ~t_ok
    attack_order_eff = jnp.where(am_tgt, am.target, attack_order).astype(jnp.int32)
    t_ok_eff = t_ok | am_tgt
    goal = jnp.where((am.active & ~t_ok_eff)[:, None], jnp.stack([am.goal_x, am.goal_y], -1), goal)
    moving_eff = moving | (am.active & ~t_ok_eff)
    desired = desired.at[:c].set(jnp.where(t_ok_eff, attack_order_eff, -1))
    champ = champ._replace(attack_order=attack_order)
    return s, sc._replace(towers=towers, lane_ai=lane_ai, desired=desired, lane_goal=mgoal, lane_stop=stop,
                          mai=mai, mon_goal=m_goal, mon_speed=m_speed, mon_move=m_move, mbuf=mbuf, units=units,
                          attack_order=attack_order, attack_order_eff=attack_order_eff, t_ok_eff=t_ok_eff,
                          goal=goal, moving_eff=moving_eff, champ=champ, amove=amove, jungle=jungle)
