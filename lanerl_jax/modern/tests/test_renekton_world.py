"""Renekton through the real compiled tick: Cull the Meek Fury and healing, a Slice dash that moves him and hits
on the way, the empowered W stripping Garen's shield, Fury on the observation's resource bar and Dominus health
(one top-lane world, own compile)."""

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.modern.core import damage as D
from lanerl_jax.modern.tests import world_harness as H

if not H.artifacts_present():
    pytest.skip("modern map/route artifacts not present", allow_module_level=True)

from lanerl_jax.modern import obs as OB  # noqa: E402
from lanerl_jax.modern import world as MS  # noqa: E402
from lanerl_jax.modern.data import loadouts as DL  # noqa: E402
from lanerl_jax.modern.runes import catalog as RD  # noqa: E402
from lanerl_jax.modern.world import config as MW  # noqa: E402

orders = H.orders


@lru_cache(maxsize=1)
def world():
    cfg = MW.build_config((MW.Loadout("Renekton", items=H.ITEMS, rune_page=DL.rune_pages("Renekton")[0]),
                           MW.Loadout("Garen", items=H.ITEMS, rune_page=RD.GAREN_DEFAULT_PAGE)),
                          fog=False, lanes=(2,), jungle=False, objectives=False)
    tick = jax.jit(lambda s, o: MS.step(s, o, cfg))

    @jax.jit
    def run(s, o, n):
        return jax.lax.fori_loop(0, n, lambda _, s: tick(s, o)[0], s)
    return cfg, tick, run


def step(s, o):
    return world()[1](s, o)[0]


def run(s, o, n):
    return world()[2](s, o, jnp.asarray(n, jnp.int32))


def placed(ren_xy, garen_xy, fury=0.0, ranks=(1, 1, 1, 1)):
    cfg = world()[0]
    s = MS.init_state(cfg)
    ren = s.kits.renekton                                             # in combat: no Fury decay meanwhile
    kits = s.kits._replace(renekton=ren._replace(fury=ren.fury.at[0].set(fury), combat_at=ren.combat_at.at[0].set(70.0)))
    return s._replace(x=s.x.at[0].set(ren_xy[0]).at[1].set(garen_xy[0]),
                      y=s.y.at[0].set(ren_xy[1]).at[1].set(garen_xy[1]), t=jnp.float32(70.0), kits=kits,
                      hp=s.hp.at[0].set(300.0),                      # below half health: Fury x1.5
                      champ=s.champ._replace(ranks=s.champ.ranks.at[0].set(jnp.asarray(ranks, jnp.int32))
                                             .at[1].set(jnp.asarray([1, 1, 1, 0], jnp.int32))))


def test_record_radius_and_no_mana():
    cfg = world()[0]
    s = MS.init_state(cfg)
    np.testing.assert_allclose(float(s.radius[0]), 80.0)
    np.testing.assert_allclose(np.asarray(cfg.champion_base.acquisition_range), [600.0, 400.0])
    assert float(s.champ.mana[0]) == 0.0


def test_cull_the_meek_fury_and_heal_and_resource_bar():
    cfg = world()[0]
    mx, my = H.lane_mid(cfg)
    s = placed((mx, my), (mx + 200.0, my))
    hp_g, hp_r = float(s.hp[1]), float(s.hp[0])
    s = step(s, orders(cast_slot=[0, -1]))
    assert float(s.hp[1]) < hp_g and float(s.hp[0]) > hp_r
    np.testing.assert_allclose(float(s.kits.renekton.fury[0]), 15.0)        # 10 x 1.5: below half health
    o = jax.jit(lambda s: OB.build_modern_observation(s, 0, OB.modern_frames(cfg)[0], cfg))(s)
    mana_col = 16 + 2 * len(OB.K.KITS)                                      # after the champion one-hots
    np.testing.assert_allclose(float(o.self_vec[mana_col]), 0.15, rtol=1e-5)


def test_slice_dash_moves_and_hits_on_the_way():
    cfg = world()[0]
    mx, my = H.lane_mid(cfg)
    s = placed((mx - 300.0, my), (mx, my))
    hp_g = float(s.hp[1])
    s = step(s, orders(cast_slot=[2, -1], cast_x=[mx + 2000.0, 0.0], cast_y=[my, 0.0]))
    s = run(s, MS.no_orders(), 20)
    d = float(np.hypot(s.x[0] - (mx - 300.0), s.y[0] - my))
    assert 380.0 < d <= 460.0
    assert float(s.hp[1]) < hp_g and float(s.kits.renekton.e_dice_until[0]) > float(s.t)
    assert float(s.champ.cooldowns[0, 2]) > 0.0


def test_empowered_ruthless_predator_strips_the_shield():
    cfg = world()[0]
    mx, my = H.lane_mid(cfg)
    s = placed((mx, my), (mx + 150.0, my), fury=60.0)
    s = step(s, orders(cast_slot=[-1, 1]))                                    # Garen W shield
    assert float(D.total_shield(s.shields, s.t)[1]) > 0.0
    s = step(s, orders(cast_slot=[1, -1]))
    for _ in range(30):
        s = step(s, orders(attack=[1, -1]))
        if not bool(s.kits.renekton.w_on[0]):
            break
    assert not bool(s.kits.renekton.w_on[0])
    np.testing.assert_allclose(float(D.total_shield(s.shields, s.t)[1]), 0.0)
    assert float(s.cc.stun_until[1]) > float(s.t) + 0.5                     # 1.5 s x (1 - 0.6 Garen W tenacity)
    assert float(s.champ.cast_lock_until[0]) > float(s.t)
    np.testing.assert_allclose(float(s.kits.renekton.fury[0]), 10.0)


def test_dominus_raises_max_health():
    cfg = world()[0]
    mx, my = H.lane_mid(cfg)
    s = placed((mx, my), (mx + 2000.0, my))
    base = run(s, MS.no_orders(), 11)
    s = step(s, orders(cast_slot=[3, -1]))
    s = run(s, MS.no_orders(), 10)
    np.testing.assert_allclose(float(s.max_hp[0]), float(base.max_hp[0]) + 300.0, rtol=1e-5)
    assert float(s.kits.renekton.fury[0]) >= 20.0


def test_slice_into_a_wall_stops_short():
    """Wiki Dash: a dash whose end point is not walkable stops at the last walkable point before the wall."""
    from lanerl_jax.modern import mechanics as M
    from lanerl_jax.modern.map.terrain import is_walkable
    cfg = world()[0]
    ter = M.team_terrain(cfg.terrain, 0)
    walk = lambda x, y: bool(is_walkable(x, y, 50.0, ter))                 # noqa: E731
    # A lane point and a direction whose 450-unit dash ends inside a wall, with open ground for the first 150.
    lane = np.asarray(cfg.lane_path)
    found = None
    for px, py in lane[2:-2]:
        for ang in np.deg2rad(np.arange(0.0, 360.0, 15.0)):
            ux, uy = np.cos(ang), np.sin(ang)
            path = [walk(px + ux * d, py + uy * d) for d in np.arange(0.0, 451.0, 10.0)]
            if all(path[:16]) and not path[-1]:
                found = (float(px), float(py), ux, uy, 10.0 * path.index(False))
                break
        if found:
            break
    assert found is not None
    sx, sy, ux, uy, wall = found
    s = placed((sx, sy), (sx + 3000.0, sy))
    s = step(s, orders(cast_slot=[2, -1], cast_x=[sx + 1000.0 * ux, 0.0], cast_y=[sy + 1000.0 * uy, 0.0]))
    s = run(s, MS.no_orders(), 20)
    along = (float(s.x[0]) - sx) * ux + (float(s.y[0]) - sy) * uy
    assert 100.0 < along < wall, (along, wall)                        # dashed, and stopped before the wall
    assert walk(float(s.x[0]), float(s.y[0]))
