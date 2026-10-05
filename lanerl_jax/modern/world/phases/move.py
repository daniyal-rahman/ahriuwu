"""Phase 5, MOVE: route movement, dashes, Flash and Teleport, unit collision; move-order completion."""
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
from .. import views as V
from ..config import N_CHAMPIONS, WorldConfig
from ..config import layout as MW_layout
from ..scratch import TickScratch
from ..state import ModernOrders, ModernState

MOVE_ARRIVE_RADIUS = 5.0     # a move order is complete within this distance of its goal


def run(s: ModernState, orders: ModernOrders, cfg: WorldConfig, sc: TickScratch) -> tuple[ModernState, TickScratch]:
    """5. MOVE: route movement, dashes, Flash, Teleport; collision.

    Reads: ``now, caps, champ, units, st, reach, locked, s_out`` (Flash/Teleport/Ghost), ``kit_out`` (dash),
    ``kits, kctx`` (kit ghosting), the AI goals ``attack_order_eff, t_ok_eff, goal, moving_eff, lane_goal,
    lane_stop, mon_goal, mon_speed, mon_move``, ``mai, mbuf, so, jungle, lane_ai, towers, terrain, ictx`` and
    ``s.pending_dash``.
    Writes ``s.x, s.y, s.lane_ai, s.towers, s.route_anchor``, dash state and facing in both ``s.champ``
    and ``champ``, and ``tp_lock, sres, ms, dstart, in_dash, units, ictx``."""
    c, n, dt = N_CHAMPIONS, cfg.n_units, cfg.dt
    now, caps, champ, units, st, reach = sc.now, sc.caps, sc.champ, sc.units, sc.st, sc.reach
    locked, s_out, kit_out, kits, kctx = sc.locked, sc.s_out, sc.kit_out, sc.kits, sc.kctx
    attack_order_eff, t_ok_eff, goal, moving_eff = sc.attack_order_eff, sc.t_ok_eff, sc.goal, sc.moving_eff
    mgoal, stop, m_goal, m_speed, m_move = sc.lane_goal, sc.lane_stop, sc.mon_goal, sc.mon_speed, sc.mon_move
    so, mai, mbuf, jungle, lane_ai, towers = sc.so, sc.mai, sc.mbuf, sc.jungle, sc.lane_ai, sc.towers
    terrain, ictx = sc.terrain, sc.ictx
    tp_lock = s_out.teleport_channel | s_out.teleport_dash
    can_move_c = caps["can_move"][:c] & s.alive[:c] & ~locked & ~tp_lock & (now >= champ.dash_until)
    tgt = jnp.clip(attack_order_eff, 0, n - 1)
    in_range = M.in_attack_range(units._replace(attack_range=units.attack_range.at[:c].set(reach)),
                                 jnp.full((n,), -1, jnp.int32).at[:c].set(jnp.where(t_ok_eff, attack_order_eff,
                                                                                    -1)))[:c]
    chase = t_ok_eff & ~in_range
    cgoal = jnp.where(chase[:, None], jnp.stack([s.x[tgt], s.y[tgt]], -1), goal)
    cact = can_move_c & (chase | (moving_eff & ~t_ok_eff))
    minion = (s.kind == W.KIND_MINION) & s.alive
    monster = (s.kind == W.KIND_MONSTER) & s.alive
    gx = jnp.where(minion, mgoal[:, 0], jnp.where(monster, m_goal[:, 0], s.x)).at[:c].set(cgoal[:, 0])
    gy = jnp.where(minion, mgoal[:, 1], jnp.where(monster, m_goal[:, 1], s.y)).at[:c].set(cgoal[:, 1])
    hg_bonus = 0.0 if so is None else so.homeguard_bonus * s.econ.homeguard.active
    sres = jnp.zeros((n,), jnp.float32).at[:c].set(st.slow_resist)
    if mai is not None:                                               # Scuttler: slow immune while not fleeing
        jsl = slice(cfg.jungle.monster0, cfg.jungle.monster0 + cfg.jungle.n_slots)
        sres = sres.at[jsl].set(mai.slow_resist)
    # Gustwalker's Gait from jungle state (brush entry detected at last tick's combat phase: 1 tick lag).
    if cfg.jungle is None:
        gust = 0.0
    else:
        gust = J.gust_bonus_ms(jungle, now)
    m_ms = LA.minion_move_speed(lane_ai, units, now)
    if mbuf is not None:
        m_ms = jnp.where(minion, jnp.maximum(m_ms, mbuf.ms_floor), m_ms)
    # Non-champion slows apply at use with slow resist (champions: STAT pipeline ``slow=``).
    # Champions: Ghost/Heal, Gustwalker and Homeguard are bonus % MS inside the STAT pipeline, before
    # the soft caps (26.9 replays, REPLAY_FIDELITY; adding them after the caps ran ~60 u fast).
    summ_world = sc.summ_world
    extra_pct = s_out.bonus_ms_pct + gust + champ.homeguard_ms + hg_bonus
    champ_ms = compose(cfg.champion_base, s.econ.level,
                       summ_world._replace(percent_move_speed=summ_world.percent_move_speed + extra_pct),
                       adaptive_physical=cfg.adaptive_physical, slow=caps["slow"][:c]).move_speed
    ms = (jnp.where(monster, m_speed, m_ms) * (1.0 - caps["slow"] * (1.0 - sres))).at[:c].set(champ_ms)
    active = ((minion & ~stop | monster & m_move) & caps["can_move"]).at[:c].set(cact)
    team_mv = jnp.clip(s.team, 0, 1)                                  # neutral monsters walk the blue mask
    x, y, _, anchor = M.move_step(s.x, s.y, gx, gy, ms, active, team_mv, s.radius, cfg.routes, terrain, dt,
                                  s.route_anchor, movers=MW_layout()["ward0"])
    # Kit dashes (Jax Q): follow the target unit at the dash speed (no terrain, it's a leap).
    dash = kit_out.dash
    pd = s.pending_dash                                               # item-active dash (Rocketbelt)
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
    fx, fy = jnp.where(dash_target >= 0, s.x[ft], champ.dash_to[:, 0]), jnp.where(dash_target >= 0, s.y[ft], champ.dash_to[:, 1])
    fd = jnp.sqrt((fx - x[:c]) ** 2 + (fy - y[:c]) ** 2)
    step_d = jnp.minimum(jnp.where(dash.speed > 0, dash.speed, 1400.0) * dt, fd)
    frac = jnp.where(fd > 1e-6, step_d / jnp.maximum(fd, 1e-6), 0.0)
    cx = jnp.where(in_dash, x[:c] + (fx - x[:c]) * frac, x[:c])
    cy = jnp.where(in_dash, y[:c] + (fy - y[:c]) * frac, y[:c])
    # Flash (blink onto walkable terrain) and Teleport arrival.
    blink = s_out.dash.active & s_out.dash.blink
    bx, by = M.blink_point(cx, cy, s_out.dash.to_x, s_out.dash.to_y, jnp.full((c,), S.FLASH_RANGE), s.team[:c],
                           s.radius[:c], terrain)
    cx, cy = jnp.where(blink, bx, cx), jnp.where(blink, by, cy)
    cx = jnp.where(s_out.teleport_arrive, s_out.teleport_x, cx)
    cy = jnp.where(s_out.teleport_arrive, s_out.teleport_y, cy)
    x, y = x.at[:c].set(cx.astype(x.dtype)), y.at[:c].set(cy.astype(y.dtype))
    # Unit collision (collision, COLLISION.md): avoidance steering of movers, then soft separation,
    # pathing radii, never into terrain (movement clearance on the team mask, as the MOVE clamp and eject).
    ghost = (jnp.zeros((n,), bool).at[:c].set(s_out.ghosted | (now < dash_until) | K.ghosted(kits, kctx))
             | LA.minion_ghosted(lane_ai, units, now))
    # Wards have no collision; structures block through their navgrid pads (terrain), not as unit
    # obstacles: their collision circles reach past the pads that routes are baked around, which left
    # units pinned against them (red stuck at its top inhibitor).
    collide = s.alive & (s.kind != W.KIND_WARD) & ~W.is_structure(s.kind)
    x, y = UC.resolve(s.x, s.y, x, y, radius=UC.pathing_radius(s.kind, s.sub, s.radius), collide=collide,
                      ghosted=ghost, moving=active, goal_x=gx, goal_y=gy, team=team_mv,
                      clearance=jnp.minimum(s.radius, cfg.routes.radius), terrain=terrain, dt=dt,
                      movers=MW_layout()["ward0"])
    facing = jnp.stack([x[:c] - s.x[:c], y[:c] - s.y[:c]], -1)
    norm = jnp.linalg.norm(facing, axis=-1, keepdims=True)
    facing = jnp.where(norm > 1e-3, facing / jnp.maximum(norm, 1e-6), champ.facing)
    moved = jnp.sqrt((x[:c] - s.x[:c]) ** 2 + (y[:c] - s.y[:c]) ** 2)
    # A move order ends on arrival, or where the champion can make no more progress (unreachable or
    # walled-off goal: League walks as far as it can and stops). Then idle auto-acquire resumes
    # (MECHANICS_AUDIT #2; the flag used to stay set for the rest of the game).
    to_goal = jnp.sqrt((x[:c] - champ.move_goal[:, 0]) ** 2 + (y[:c] - champ.move_goal[:, 1]) ** 2)
    stuck = cact & (moved < 0.5) & ~in_dash
    champ = champ._replace(moving=champ.moving & ~(to_goal <= MOVE_ARRIVE_RADIUS) & ~(stuck & ~chase))
    # Dash state and facing go to the working ChampionLayer too: it becomes the next state's ``champ``
    # (writing only ``s.champ`` dropped them every tick, so a Jax Q leap moved only on its start tick).
    champ = champ._replace(dash_until=dash_until, dash_target=dash_target.astype(jnp.int32),
                           dash_to=jnp.where(dstart[:, None], jnp.stack([dx, dy], -1), champ.dash_to),
                           dash_speed=jnp.where(dstart, dash.speed, champ.dash_speed), facing=facing)
    s = s._replace(x=x, y=y, lane_ai=lane_ai, towers=towers, champ=champ, route_anchor=anchor)
    units = V.units_view(s)._replace(attack_range=s.arange.at[:c].set(reach),
                                   attack_speed=s.aspeed.at[:c].set(st.attack_speed))
    ictx = ictx._replace(x=x[:c], y=y[:c], facing_x=facing[:, 0], facing_y=facing[:, 1], moved=moved)
    return s, sc._replace(champ=champ, tp_lock=tp_lock, sres=sres, ms=ms, dstart=dstart, in_dash=in_dash,
                          units=units, ictx=ictx)
