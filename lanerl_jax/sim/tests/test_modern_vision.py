"""Fog of war in the modern world (``modern_vision`` wired through ``modern_step``).

Fixtures are real Map11 places next to the top lane: the lane midpoint, the
lane brush beside it (navgrid brush patch, 5 cells deep), and two points about
the same distance from the midpoint, one behind a wall and one in the open.
Checks are symptoms the policy or the lane would feel: an enemy in brush is
absent from the observation and cannot be clicked or attacked, walls hide what
open ground shows, structures never fog, an attack from brush reveals the
attacker, and a cast nobody saw is not remembered.
"""
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.sim import modern_rune_data as RD
from lanerl_jax.sim import modern_world as MW
from lanerl_jax.sim import modern_world_types as W

if not MW.DEFAULT_MAP.exists() or not MW.DEFAULT_ROUTES.exists():
    pytest.skip("modern map/route artifacts not present", allow_module_level=True)

from lanerl_jax.sim import modern_step as MS  # noqa: E402
from lanerl_jax.sim import modern_vision as MV  # noqa: E402

LANE_MID = (2720.0, 13100.0)          # top-lane path midpoint (open ground)
BRUSH = (2274.0, 13558.0)             # deepest walkable cell of the lane brush (639 u from LANE_MID)
BEHIND_WALL = (3062.0, 12087.0)       # walkable, 1069 u from LANE_MID, wall on the segment
OPEN = (2193.0, 12487.0)              # walkable, 808 u from LANE_MID, clear segment
BRUSH_EDGE = (2469.0, 13357.0)        # just outside the brush, 280 u from BRUSH towards LANE_MID
OUTSIDE = (2588.0, 13236.0)           # outside the brush, 450 u from BRUSH
JAX_PAGE = RD.RunePage(RD.PRECISION, 8010, (9111, 9104, 8299), RD.RESOLVE, (8444, 8242), (5005, 5008, 5001))


@lru_cache(maxsize=1)
def world():
    lo = (MW.Loadout("Garen", items=(1055, 2003), rune_page=RD.GAREN_DEFAULT_PAGE),
          MW.Loadout("Jax", items=(1055, 2003), rune_page=JAX_PAGE))
    cfg = MW.build_config(lo)
    refresh = jax.jit(lambda s: MS.refresh_visibility(s, cfg))
    step = jax.jit(lambda s, o: MS.step(s, o, cfg))
    return cfg, refresh, step


@lru_cache(maxsize=1)
def fast_world():
    """Same world with ``fog="fast"`` (brush lookup, walls ignored); visibility only, no step."""
    cfg, *_ = world()
    fast = MW.build_config(cfg.loadouts, fog="fast")
    return fast, jax.jit(lambda s: MS.refresh_visibility(s, fast))


def place(garen, jax_, t=30.0, *, fast=False):
    cfg, refresh = fast_world() if fast else world()[:2]
    s = MS.init_state(cfg)
    s = s._replace(x=s.x.at[0].set(garen[0]).at[1].set(jax_[0]), y=s.y.at[0].set(garen[1]).at[1].set(jax_[1]),
                   t=jnp.float32(t))
    return refresh(s)


def orders(**kw):
    o = MS.no_orders()._asdict()
    for k, v in kw.items():
        o[k] = jnp.asarray(v, o[k].dtype)
    return MS.ModernOrders(**o)


def test_fixtures_are_what_they_claim():
    from lanerl_jax.data.modern_map import BRUSH as BRUSH_BIT, load_patch_map
    grid, _ = load_patch_map(MW.DEFAULT_MAP)
    flags = lambda p: int(grid.flags[grid.cell(*p)[1], grid.cell(*p)[0]])
    assert flags(BRUSH) & BRUSH_BIT and not flags(LANE_MID) & BRUSH_BIT
    for p in (LANE_MID, BRUSH, BEHIND_WALL, OPEN, BRUSH_EDGE, OUTSIDE):
        assert grid.is_walkable(*p, radius=35.0, team=0)
    assert not flags(BRUSH_EDGE) & BRUSH_BIT and not flags(OUTSIDE) & BRUSH_BIT
    assert np.hypot(BEHIND_WALL[0] - LANE_MID[0], BEHIND_WALL[1] - LANE_MID[1]) < MV.CHAMPION_SIGHT


def test_sight_radius_walls_and_structures():
    cfg, *_ = world()
    s = place(LANE_MID, OPEN)
    assert bool(s.visible[0, 1]) and bool(s.visible[1, 0])          # open ground, 808 u
    s = place(LANE_MID, BEHIND_WALL)
    assert not bool(s.visible[0, 1]) and not bool(s.visible[1, 0])  # rays: same range, wall between
    s = place(LANE_MID, BEHIND_WALL, fast=True)
    assert bool(s.visible[0, 1])                                     # fast fog ignores walls (VIS-FAST-M)
    far = (LANE_MID[0] + 1400.0 * 0.6, LANE_MID[1] - 1400.0 * 0.8)
    s = place(LANE_MID, far)
    assert not bool(s.visible[0, 1])                                 # beyond 1350 (no other viewers at t=30)
    # Structures never fog, even untargetable ones deep in the enemy base; own units always visible.
    structure = np.isin(np.asarray(s.kind), [W.KIND_TURRET, W.KIND_INHIBITOR, W.KIND_NEXUS])
    assert np.asarray(s.visible)[:, structure].all()
    assert bool(s.visible[0, 0]) and bool(s.visible[1, 1])


def test_brush_hides_from_outside_but_sees_out():
    s = place(LANE_MID, BRUSH)
    assert not bool(s.visible[0, 1])                                 # Garen outside cannot see Jax in brush
    assert bool(s.visible[1, 0])                                     # Jax sees out
    from lanerl_jax.obs import modern_builder as OB
    cfg, *_ = world()
    frames = OB.modern_frames(cfg)
    o0 = OB.build_modern_observation(s, 0, frames[0], cfg)
    o1 = OB.build_modern_observation(s, 1, frames[1], cfg)
    assert int(o0.slot_unit[0]) == -1 and float(o0.global_vec[1]) == 0.0
    assert int(o1.slot_unit[0]) == 0 and float(o1.global_vec[1]) == 1.0
    # Both in the same brush: they see each other.
    s = place((BRUSH[0] + 60.0, BRUSH[1]), BRUSH)
    assert bool(s.visible[0, 1]) and bool(s.visible[1, 0])


def test_reveal_circle_expires():
    cfg, *_ = world()
    s = place(LANE_MID, BRUSH)
    rev = MV.reveal_step(s.reveal, jnp.asarray([False, True]), jnp.asarray([False, True]),
                         s.x[:2], s.y[:2], s.t)
    vis, _ = MV.visibility(s.x, s.y, s.kind, s.sub, s.team, s.alive, rev, s.t + 1.9, cfg.vision,
                           n_fogged=MW.layout()["struct0"])
    assert bool(vis[0, 1])
    vis, _ = MV.visibility(s.x, s.y, s.kind, s.sub, s.team, s.alive, rev, s.t + 2.01, cfg.vision,
                           n_fogged=MW.layout()["struct0"])
    assert not bool(vis[0, 1])


def test_fogged_target_cannot_be_attacked_and_unseen_cast_is_not_remembered():
    _, _, step = world()
    s = place(LANE_MID, BRUSH)                                         # Garen outside, beyond Jax's acquisition
    assert not bool(s.visible[0, 1])
    c = s.champ
    s = s._replace(champ=c._replace(ranks=c.ranks.at[1].set(jnp.asarray([0, 1, 1, 0], c.ranks.dtype))))
    hp0 = float(s.hp[1])
    s, _ = step(s, orders(attack=[1, -1], cast_slot=[-1, 1]))          # Garen clicks Jax; Jax casts W in brush
    assert int(s.champ.attack_order[0]) == -1
    assert float(s.champ.last_cast[1, 1]) == pytest.approx(float(s.t))
    assert float(s.champ.seen_cast[1, 1]) < -1e8                         # nobody on blue saw it
    for _ in range(30):
        s, _ = step(s, orders(attack=[1, -1]))
    assert int(s.champ.attack_order[0]) == -1 and float(s.hp[1]) >= hp0 - 1e-3


def test_attack_from_brush_reveals_the_attacker():
    _, _, step = world()
    s = place(BRUSH, BRUSH_EDGE)                                       # Garen in brush, Jax in his reach outside
    assert not bool(s.visible[1, 0]) and bool(s.visible[0, 1])
    revealed_at = None
    for k in range(30):
        s, e = step(s, orders(attack=[1, -1]))
        if bool(e.launched[0]):
            revealed_at = float(s.t)
            break
    assert revealed_at is not None
    assert bool(s.visible[1, 0])                                         # 300 u circle around Garen
    assert float(s.reveal.until[0]) == pytest.approx(revealed_at + MV.REVEAL_DURATION)
