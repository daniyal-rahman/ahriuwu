"""Camille through the real compiled tick against Garen: the ramped Precision Protocol recast's true damage, Tactical
Sweep's outer slow, Hookshot onto a real wall and Wall Dive onto Garen, The Hextech Ultimatum's zone holding Garen
(one top-lane world, own compile)."""

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.modern.tests import world_harness as H

if not H.artifacts_present():
    pytest.skip("modern map/route artifacts not present", allow_module_level=True)

from lanerl_jax.modern import world as MS  # noqa: E402
from lanerl_jax.modern.core import damage as D  # noqa: E402
from lanerl_jax.modern.world import config as MW  # noqa: E402

orders = H.orders
SX, SY = 5600.0, 13620.0                                              # straight top-lane stretch toward +x


@lru_cache(maxsize=1)
def world():
    cfg = MW.build_config((MW.Loadout("Camille", items=H.ITEMS), MW.Loadout("Garen", items=H.ITEMS)),   # no runes
                          fog=False, lanes=(2,), jungle=False, objectives=False)
    tick = jax.jit(lambda s, o: MS.step(s, o, cfg))

    @jax.jit
    def run(s, o, n):
        return jax.lax.fori_loop(0, n, lambda _, s: tick(s, o)[0], s)
    return cfg, tick, run


def step(s, o):
    return world()[1](s, o)


def run(s, o, n):
    return world()[2](s, o, jnp.asarray(n, jnp.int32))


def placed(cam_xy, garen_xy, ranks=(1, 1, 1, 1), level=None):
    s = MS.init_state(world()[0])
    s = s._replace(x=s.x.at[0].set(cam_xy[0]).at[1].set(garen_xy[0]),
                   y=s.y.at[0].set(cam_xy[1]).at[1].set(garen_xy[1]), t=jnp.float32(70.0),
                   champ=s.champ._replace(ranks=s.champ.ranks.at[0].set(jnp.asarray(ranks, jnp.int32))
                                          .at[1].set(jnp.asarray([1, 1, 1, 0], jnp.int32))))
    return s


def test_record_radius_pathing_and_windup():
    cfg = world()[0]
    s = MS.init_state(cfg)
    np.testing.assert_allclose(float(s.radius[0]), 65.0)
    np.testing.assert_allclose(np.asarray(cfg.champion_base.acquisition_range)[0], 600.0)
    np.testing.assert_allclose(np.asarray(cfg.champion_base.windup_percent)[0], 0.3 / 1.552, rtol=1e-5)


def test_precision_protocol_ramped_recast_deals_true_damage():
    s = placed((SX, SY), (SX + 170.0, SY))
    s, _ = step(s, orders(cast_slot=[0, -1]))
    trues, recast = [], False
    for i in range(110):
        o = orders(attack=[1, -1])
        if i == 60 and not recast:                                    # 2 s after the first attack: recast
            o = orders(attack=[1, -1], cast_slot=[0, -1])
            recast = True
        s, ev = step(s, o)
        p = ev.report.packets
        mine = np.asarray(p.valid) & (np.asarray(p.src) == 0) & (np.asarray(p.dst) == 1)
        if (mine & (np.asarray(p.dtype) == D.TRUE)).any():
            trues.append(i)
    assert trues, "no true damage from the ramped recast"


def test_tactical_sweep_outer_slow():
    s = placed((SX, SY), (SX + 450.0, SY))
    hp = float(s.hp[1])
    hold = dict(move=[True, True], move_x=[SX, SX + 450.0], move_y=[SY, SY])   # both stand still
    s, _ = step(s, orders(cast_slot=[1, -1], cast_x=[SX + 1000.0, 0.0], cast_y=[SY, 0.0], **hold))
    s = run(s, orders(**hold), 34)                                     # 1.1 s
    assert float(s.hp[1]) < hp and float(s.cc.slow[1]) > 0.5


def test_hookshot_to_a_wall_then_wall_dive_onto_garen():
    from lanerl_jax.modern import mechanics as M
    from lanerl_jax.modern.map.terrain import is_walkable
    cfg = world()[0]
    ter = M.team_terrain(cfg.terrain, 0)
    walk = lambda x, y: bool(is_walkable(x, y, 50.0, ter))                 # noqa: E731
    lane = np.asarray(cfg.lane_path)
    found = None
    for px, py in lane[4:-4]:
        for ang in np.deg2rad(np.arange(0.0, 360.0, 15.0)):
            ux, uy = np.cos(ang), np.sin(ang)
            path = [walk(px + ux * d, py + uy * d) for d in np.arange(0.0, 801.0, 25.0)]
            if all(path[:12]) and not all(path):
                found = (float(px), float(py), ux, uy, 25.0 * path.index(False))
                break
        if found:
            break
    assert found is not None
    sx, sy, ux, uy, wall = found
    gx, gy = sx - 250.0 * ux, sy - 250.0 * uy                           # Garen behind her, away from the wall
    s = placed((sx, sy), (gx, gy))
    s, _ = step(s, orders(cast_slot=[2, -1], cast_x=[sx + 1000.0 * ux, 0.0], cast_y=[sy + 1000.0 * uy, 0.0]))
    s = run(s, MS.no_orders(), 40)                                     # grapple (<= 0.57 s) + pull (<= 0.6 s)
    along = (float(s.x[0]) - sx) * ux + (float(s.y[0]) - sy) * uy
    assert along > wall - 120.0, (along, wall)                        # pulled to the wall
    assert int(s.kits.camille.e_stage[0]) == 2                         # hanging
    hp = float(s.hp[1])
    s, _ = step(s, orders(cast_slot=[2, -1], cast_x=[float(s.x[1]), 0.0], cast_y=[float(s.y[1]), 0.0]))
    s = run(s, MS.no_orders(), 25)
    assert float(s.hp[1]) < hp and float(s.cc.stun_until[1]) > float(s.t) - 0.5


def test_hextech_ultimatum_holds_garen_in_the_zone():
    s = placed((SX, SY), (SX + 400.0, SY))
    s, _ = step(s, orders(cast_slot=[3, -1], cast_target=[1, -1]))
    s = run(s, MS.no_orders(), 17)                                     # the 0.5 s leap
    assert int(s.kits.camille.r_target[0]) == 1
    cx, cy = float(s.kits.camille.r_cx[0]), float(s.kits.camille.r_cy[0])
    s = run(s, orders(move=[False, True], move_x=[0.0, SX + 3000.0], move_y=[0.0, SY]), 60)
    d = float(np.hypot(s.x[1] - cx, s.y[1] - cy))
    assert d <= 425.0 + 1.0, d
