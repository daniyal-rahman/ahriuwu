"""Phase 5, MOVE: route movement, dashes, Flash and Teleport, unit collision, move-order completion.
Writes ``s.x/y``, ``s.lane_ai``, ``s.towers``, ``s.route_anchor`` and dash state and facing (``s.champ`` and the
working ``champ``)."""
from __future__ import annotations

import jax.numpy as jnp

from ... import champions as K
from ... import collision as UC
from ... import mechanics as M
from ...champions import summoners as S
from ...core import types as W
from ...core.stat_pipeline import compose
from ...jungle import camps as J
from ...lane import ai as LA
from .. import units as U
from ..config import N_CHAMPIONS, WorldConfig
from ..scratch import TickScratch
from ..state import ModernOrders, ModernState

MOVE_ARRIVE_RADIUS = 5.0     # a move order is complete within this distance of its goal


def run(s: ModernState, orders: ModernOrders, cfg: WorldConfig, sc: TickScratch) -> tuple[ModernState, TickScratch]:
    c, n, dt = N_CHAMPIONS, cfg.n_units, cfg.dt
    now, caps, champ, units, st, reach = sc.now, sc.caps, sc.champ, sc.units, sc.st, sc.reach
    s_out = sc.s_out
    attack_order_eff, t_ok_eff, goal = sc.attack_order_eff, sc.t_ok_eff, sc.goal
    mgoal, m_goal = sc.lane_goal, sc.mon_goal
    so, mai, mbuf, lane_ai = sc.so, sc.mai, sc.mbuf, sc.lane_ai
    terrain, ictx = sc.terrain, sc.ictx
    tp_lock = s_out.teleport_channel | s_out.teleport_dash
    can_move_c = caps["can_move"][:c] & s.alive[:c] & ~sc.locked & ~tp_lock & (now >= champ.dash_until)
    tgt = jnp.clip(attack_order_eff, 0, n - 1)
    in_range = M.in_attack_range(units._replace(attack_range=units.attack_range.at[:c].set(reach)),
                                 jnp.full((n,), -1, jnp.int32).at[:c].set(jnp.where(t_ok_eff, attack_order_eff,
                                                                                    -1)))[:c]
    chase = t_ok_eff & ~in_range
    cgoal = jnp.where(chase[:, None], jnp.stack([s.x[tgt], s.y[tgt]], -1), goal)
    cact = can_move_c & (chase | (sc.moving_eff & ~t_ok_eff))
    minion = (s.kind == W.KIND_MINION) & s.alive
    monster = (s.kind == W.KIND_MONSTER) & s.alive
    gx = jnp.where(minion, mgoal[:, 0], jnp.where(monster, m_goal[:, 0], s.x)).at[:c].set(cgoal[:, 0])
    gy = jnp.where(minion, mgoal[:, 1], jnp.where(monster, m_goal[:, 1], s.y)).at[:c].set(cgoal[:, 1])
    hg_bonus = 0.0 if so is None else so.homeguard_bonus * s.econ.homeguard.active
    sres = jnp.zeros((n,), jnp.float32).at[:c].set(st.slow_resist)
    if mai is not None:                                               # Scuttler is slow immune unless fleeing
        jsl = cfg.jungle.slots
        sres = sres.at[jsl].set(mai.slow_resist)
    gust = 0.0 if cfg.jungle is None else J.gust_bonus_ms(sc.jungle, now)          # brush entry from last tick
    m_ms = LA.minion_move_speed(lane_ai, units, now)
    if mbuf is not None:
        m_ms = jnp.where(minion, jnp.maximum(m_ms, mbuf.ms_floor), m_ms)
    # Champions: Ghost/Heal, Gustwalker and Homeguard are bonus % MS before the soft caps (REPLAY_FIDELITY).
    # Non-champion slows apply here with slow resist.
    summ_world = sc.summ_world
    extra_pct = s_out.bonus_ms_pct + gust + champ.homeguard_ms + hg_bonus
    champ_ms = compose(cfg.champion_base, s.econ.level,
                       summ_world._replace(percent_move_speed=summ_world.percent_move_speed + extra_pct),
                       adaptive_physical=cfg.adaptive_physical, slow=caps["slow"][:c]).move_speed
    ms = (jnp.where(monster, sc.mon_speed, m_ms) * (1.0 - caps["slow"] * (1.0 - sres))).at[:c].set(champ_ms)
    active = ((minion & ~sc.lane_stop | monster & sc.mon_move) & caps["can_move"]).at[:c].set(cact)
    team_mv = jnp.clip(s.team, 0, 1)                                  # neutral monsters walk the blue mask
    x, y, _, anchor = M.move_step(s.x, s.y, gx, gy, ms, active, team_mv, s.radius, cfg.routes, terrain, dt,
                                  s.route_anchor, movers=cfg.layout.ward0)
    # Kit dashes follow their target unit, ignoring terrain; the item dash (Rocketbelt) starts a tick late.
    dash = sc.kit_out.dash
    pd = s.prev.pending_dash
    dash = W.Dash(*(jnp.where(pd.active & ~dash.active, a, b) for a, b in zip(pd, dash)))
    dstart = dash.active & s.alive[:c]
    dt_ = jnp.clip(dash.target, 0, n - 1)
    dx = jnp.where(dash.target >= 0, s.x[dt_], dash.to_x)
    dy = jnp.where(dash.target >= 0, s.y[dt_], dash.to_y)
    dist = jnp.sqrt((dx - s.x[:c]) ** 2 + (dy - s.y[:c]) ** 2)
    dash_until = jnp.where(dstart, now + dist / jnp.maximum(dash.speed, 1.0), champ.dash_until)
    dash_target = jnp.where(dstart, dash.target, champ.dash_target)
    in_dash = now < dash_until
    ft = jnp.clip(dash_target, 0, n - 1)
    fx = jnp.where(dash_target >= 0, s.x[ft], champ.dash_to[:, 0])
    fy = jnp.where(dash_target >= 0, s.y[ft], champ.dash_to[:, 1])
    fd = jnp.sqrt((fx - x[:c]) ** 2 + (fy - y[:c]) ** 2)
    step_d = jnp.minimum(jnp.where(dash.speed > 0, dash.speed, 1400.0) * dt, fd)
    frac = jnp.where(fd > 1e-6, step_d / jnp.maximum(fd, 1e-6), 0.0)
    cx = jnp.where(in_dash, x[:c] + (fx - x[:c]) * frac, x[:c])
    cy = jnp.where(in_dash, y[:c] + (fy - y[:c]) * frac, y[:c])
    blink = s_out.dash.active & s_out.dash.blink
    bx, by = M.blink_point(cx, cy, s_out.dash.to_x, s_out.dash.to_y, jnp.full((c,), S.FLASH_RANGE), s.team[:c],
                           s.radius[:c], terrain)
    cx, cy = jnp.where(blink, bx, cx), jnp.where(blink, by, cy)
    cx = jnp.where(s_out.teleport_arrive, s_out.teleport_x, cx)
    cy = jnp.where(s_out.teleport_arrive, s_out.teleport_y, cy)
    x, y = x.at[:c].set(cx.astype(x.dtype)), y.at[:c].set(cy.astype(y.dtype))
    # Collision (COLLISION.md). Wards don't collide; structures block only through their navgrid pads (their
    # circles reach past the pads that routes are baked around and pinned units).
    ghost = (jnp.zeros((n,), bool).at[:c].set(s_out.ghosted | (now < dash_until) | K.ghosted(sc.kits, sc.kctx))
             | LA.minion_ghosted(lane_ai, units, now))
    collide = s.alive & (s.kind != W.KIND_WARD) & ~W.is_structure(s.kind)
    x, y = UC.resolve(s.x, s.y, x, y, radius=UC.pathing_radius(s.kind, s.sub, s.radius), collide=collide,
                      ghosted=ghost, moving=active, goal_x=gx, goal_y=gy, team=team_mv,
                      clearance=jnp.minimum(s.radius, cfg.routes.radius), terrain=terrain, dt=dt,
                      movers=cfg.layout.ward0)
    facing = jnp.stack([x[:c] - s.x[:c], y[:c] - s.y[:c]], -1)
    norm = jnp.linalg.norm(facing, axis=-1, keepdims=True)
    facing = jnp.where(norm > 1e-3, facing / jnp.maximum(norm, 1e-6), champ.facing)
    moved = jnp.sqrt((x[:c] - s.x[:c]) ** 2 + (y[:c] - s.y[:c]) ** 2)
    # A move order ends on arrival or when stuck (unreachable goal); idle acquisition resumes (MECHANICS_AUDIT #2).
    to_goal = jnp.sqrt((x[:c] - champ.move_goal[:, 0]) ** 2 + (y[:c] - champ.move_goal[:, 1]) ** 2)
    stuck = cact & (moved < 0.5) & ~in_dash
    champ = champ._replace(moving=champ.moving & ~(to_goal <= MOVE_ARRIVE_RADIUS) & ~(stuck & ~chase))
    champ = champ._replace(dash_until=dash_until, dash_target=dash_target.astype(jnp.int32),
                           dash_to=jnp.where(dstart[:, None], jnp.stack([dx, dy], -1), champ.dash_to),
                           dash_speed=jnp.where(dstart, dash.speed, champ.dash_speed), facing=facing)
    s = s._replace(x=x, y=y, lane_ai=lane_ai, towers=sc.towers, champ=champ, route_anchor=anchor)
    units = U.units_view(s)._replace(attack_range=s.attack_range.at[:c].set(reach),
                                   attack_speed=s.attack_speed.at[:c].set(st.attack_speed))
    ictx = ictx._replace(x=x[:c], y=y[:c], facing_x=facing[:, 0], facing_y=facing[:, 1], moved=moved)
    return s, sc._replace(champ=champ, tp_lock=tp_lock, sres=sres, ms=ms, dstart=dstart, in_dash=in_dash,
                          units=units, ictx=ictx)
