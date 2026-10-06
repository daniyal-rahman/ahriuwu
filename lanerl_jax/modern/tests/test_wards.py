"""Wards, trinkets, stealth and true sight (``wards`` + ``vision``) at real top-lane places (the ``test_vision``
fixtures), over 2 champions + the 16 ward slots."""
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.modern.core import types as W
from lanerl_jax.modern.world import config as MW

if not MW.DEFAULT_MAP.exists():
    pytest.skip("modern map artifact not present", allow_module_level=True)

from lanerl_jax.modern import vision as MV  # noqa: E402
from lanerl_jax.modern import wards as WD  # noqa: E402
from lanerl_jax.modern.data.navgrid import load_patch_map  # noqa: E402
from lanerl_jax.modern.runes import catalog as RD  # noqa: E402

LANE_MID = (2720.0, 13100.0)
BRUSH = (2274.0, 13558.0)             # lane brush (deepest cell)
IN_BRUSH = (2334.0, 13558.0)          # same brush patch, 60 u away
BRUSH_EDGE = (2469.0, 13357.0)        # just outside the brush, 280 u from BRUSH
OUTSIDE = (2588.0, 13236.0)           # outside the brush, 450 u from BRUSH
FAR = (2720.0 + 1400.0, 13100.0 - 1400.0)
BLUE_JUNGLE = (3800.0, 6500.0)        # blue west quadrant jungle (wolves)
TOP_RIVER = (4600.0, 10000.0)
S = 2 * W.MAX_WARDS_PER_TEAM
N = 2 + S
TEAM = jnp.asarray([0, 1], jnp.int32)
DEEP_PAGE = RD.RunePage(RD.DOMINATION, 8112, (8126, 8141, 8135), RD.PRECISION, (9111, 9104), (5008, 5008, 5001))
SIXTH_PAGE = RD.RunePage(RD.DOMINATION, 8112, (8126, 8137, 8135), RD.PRECISION, (9111, 9104), (5008, 5008, 5001))


@lru_cache(maxsize=1)
def grids():
    g, _ = load_patch_map(MW.DEFAULT_MAP)
    return WD.ward_grid(g), MV.vision_grid(g)


@lru_cache(maxsize=None)
def stepper(pages=None):
    wg, _ = grids()
    rp = None if pages is None else jnp.asarray(RD.page_counts(list(pages)))

    def f(w, now, dt, kind, rx, ry, x, y, level, tid, cc, hits, hitter, haste, alive, wvis):
        return WD.ward_step(w, now=now, dt=dt, request=WD.WardRequest(kind, rx, ry), x=x, y=y, team=TEAM,
                            alive=alive, level=level, trinket_id=tid, control_count=cc, grid=wg,
                            trinket_haste=haste, hits=hits, hitter=hitter, rune_pages=rp, ward_visible=wvis)
    return jax.jit(f)


@lru_cache(maxsize=1)
def vis_fn():
    _, vg = grids()

    def f(w, now, x, y, level, alive):
        view, orc = WD.ward_view(w, now=now, x=x, y=y, team=TEAM, alive=alive, level=level)
        kind = jnp.concatenate([jnp.full((2,), W.KIND_CHAMPION, jnp.int32),
                                jnp.where(view.alive, W.KIND_WARD, W.KIND_NONE).astype(jnp.int32)])
        sub = jnp.concatenate([jnp.zeros((2,), jnp.int32), view.sub])
        team = jnp.concatenate([TEAM, view.team])
        al = jnp.concatenate([alive, view.alive])
        ux, uy = jnp.concatenate([x, view.x]), jnp.concatenate([y, view.y])
        kw = WD.vision_kwargs(view, orc, kind, sub, al, ward_start=2)
        vis, sight = MV.visibility(ux, uy, kind, sub, team, al, MV.init_reveal(2), now, vg, n_fogged=N, **kw)
        return vis, view
    return jax.jit(f)


class World:
    """Two champions (blue 0, red 1) and the ward system, stepped by hand."""

    def __init__(self, blue, red, *, trinkets=(WD.TOTEM_ITEM, WD.TOTEM_ITEM), level=(1, 1), pages=None,
                 t=30.0, control=(0, 0)):
        self.w = WD.init_wards(2, trinkets)
        self.x = jnp.asarray([blue[0], red[0]], jnp.float32)
        self.y = jnp.asarray([blue[1], red[1]], jnp.float32)
        self.level = jnp.asarray(level, jnp.int32)
        self.tid = jnp.asarray(trinkets, jnp.int32)
        self.cc = jnp.asarray(control, jnp.int32)
        self.haste = jnp.zeros((2,), jnp.float32)
        self.alive = jnp.ones((2,), bool)
        self.t = float(t)
        self.pages = pages
        self.last_vis = jnp.zeros((2, S), bool)

    def move(self, c, p):
        self.x, self.y = self.x.at[c].set(p[0]), self.y.at[c].set(p[1])

    def step(self, dt=1.0 / 30.0, *, req=None, hits=None, hitter=None):
        kind = np.full((2,), WD.REQ_NONE, np.int32)
        rx, ry = np.zeros((2,), np.float32), np.zeros((2,), np.float32)
        for c, (k, *p) in (req or {}).items():
            kind[c] = k
            if p:
                rx[c], ry[c] = p[0]
        h = np.zeros((S,), np.int32)
        hb = np.full((S,), -1, np.int32)
        for slot, (n, who) in (hits or {}).items():
            h[slot], hb[slot] = n, who
        self.t += dt
        self.w, ev = stepper(self.pages)(self.w, jnp.float32(self.t), jnp.float32(dt), jnp.asarray(kind),
                                         jnp.asarray(rx), jnp.asarray(ry), self.x, self.y, self.level, self.tid,
                                         self.cc, jnp.asarray(h), jnp.asarray(hb), self.haste, self.alive,
                                         self.last_vis)
        self.cc = self.cc - ev.consumed_control.astype(jnp.int32)
        vis, _ = self.vis()
        self.last_vis = vis[:, 2:]
        return ev

    def run(self, seconds, dt=1.0):
        for _ in range(int(round(seconds / dt))):
            self.step(dt)

    def vis(self):
        return vis_fn()(self.w, jnp.float32(self.t), self.x, self.y, self.level, self.alive)

    def place(self, c, p, kind=WD.REQ_TRINKET):
        ev = self.step(req={c: (kind, p)})
        assert int(ev.code[c]) == WD.OK, int(ev.code[c])
        return int(ev.placed_slot[c])


def unit(slot):
    return 2 + slot


def test_fixtures_regions():
    wg, _ = grids()
    reg = lambda p: int(WD._lookup(wg, jnp.float32(p[0]), jnp.float32(p[1]))[1])
    assert reg(BLUE_JUNGLE) == WD.REGION_BLUE_JUNGLE and reg(TOP_RIVER) == WD.REGION_RIVER
    assert reg(BRUSH) == WD.REGION_OTHER
    assert float(np.hypot(BRUSH[0] - OUTSIDE[0], BRUSH[1] - OUTSIDE[1])) < WD.TOTEM_RANGE


def test_stealth_ward_in_brush_shows_enemy_in_brush_and_hides_after_two_seconds():
    w = World(OUTSIDE, IN_BRUSH)
    vis, _ = w.vis()
    assert not bool(vis[0, 1])                                   # Jax in brush, Garen outside: hidden
    slot = w.place(0, BRUSH)
    assert 0 <= slot < W.MAX_WARDS_PER_TEAM                      # blue's slot range
    vis, view = w.vis()
    assert bool(vis[0, 1])                                       # ward in the brush sees the brush
    assert float(view.sight_radius[slot]) == 900.0 and float(view.hp[slot]) == 3.0
    assert bool(vis[1, unit(slot)])                              # visible while arming (< 2 s)
    w.run(2.1, dt=0.7)
    vis, view = w.vis()
    assert bool(view.stealthed[slot]) and not bool(vis[1, unit(slot)])   # Jax stands on it, cannot see it
    assert bool(vis[0, 1])


def test_out_of_range_and_wall_placements_are_rejected():
    w = World(LANE_MID, FAR)
    ev = w.step(req={0: (WD.REQ_TRINKET, (LANE_MID[0] + 700.0, LANE_MID[1]))})
    assert int(ev.code[0]) == WD.ERR_RANGE and not bool(ev.placed[0])
    ev = w.step(req={0: (WD.REQ_TRINKET, (2600.0, 13900.0))})     # wall cell beside the lane
    assert int(ev.code[0]) in (WD.ERR_TERRAIN, WD.ERR_RANGE)
    assert int(w.w.trinket.charges[0]) == 1


def test_enemy_control_ward_reveals_and_disables_and_is_exposed():
    w = World(OUTSIDE, IN_BRUSH, control=(0, 1))
    slot = w.place(0, BRUSH)
    w.run(3.0)
    assert bool(w.vis()[0][0, 1])
    ev = w.step(req={1: (WD.REQ_CONTROL, BRUSH_EDGE)})
    assert bool(ev.placed[1]) and bool(ev.consumed_control[1]) and int(w.cc[1]) == 0
    cslot = int(ev.placed_slot[1])
    assert cslot >= W.MAX_WARDS_PER_TEAM
    w.move(0, FAR)                                               # Garen walks away: only the ward watched
    vis, view = w.vis()
    assert bool(view.disabled[slot]) and float(view.sight_radius[slot]) == 0.0
    assert not bool(vis[0, 1])                                   # disabled ward: Jax hidden again
    assert bool(vis[1, unit(slot)])                              # Control Ward true sight shows the ward
    assert bool(view.exposed[cslot]) and bool(vis[0, unit(cslot)])   # exposed while revealing it
    assert int(w.step(req={1: (WD.REQ_CONTROL, BRUSH_EDGE)}).code[1]) == WD.ERR_NO_ITEM


def test_trinket_recharge_schedule_level_and_haste():
    w = World(OUTSIDE, FAR)
    assert int(w.w.trinket.charges[0]) == 1
    w.place(0, BRUSH)
    assert int(w.w.trinket.charges[0]) == 0
    t0 = w.t
    w.run(208.0)
    assert int(w.w.trinket.charges[0]) == 0
    w.run(2.0)                                                   # 210 s at average level 1
    assert int(w.w.trinket.charges[0]) == 1 and w.t - t0 >= 209.0
    w.run(211.0)
    assert int(w.w.trinket.charges[0]) == 2                      # capped at 2
    w.run(50.0)
    assert int(w.w.trinket.charges[0]) == 2
    # Average level 18 -> 90 s; 100 trinket haste (Grisly Mementos x17 ~ 102) halves it.
    w = World(OUTSIDE, FAR, level=(18, 18))
    w.place(0, BRUSH)
    w.run(89.0)
    assert int(w.w.trinket.charges[0]) == 0
    w.run(2.0)
    assert int(w.w.trinket.charges[0]) == 1
    w = World(OUTSIDE, FAR)
    w.haste = jnp.asarray([100.0, 0.0], jnp.float32)
    w.place(0, BRUSH)
    w.run(106.0)
    assert int(w.w.trinket.charges[0]) == 1


def test_totem_duration_by_average_level_and_expiry():
    w = World(OUTSIDE, FAR, level=(1, 18))                       # average 9.5
    slot = w.place(0, BRUSH)
    dur = float(w.w.slots.expires_at[slot] - w.w.slots.placed_at[slot])
    assert dur == pytest.approx(90.0 + 30.0 * 8.5 / 17.0, abs=1e-3)
    w.run(dur - 1.0)
    assert bool(w.w.slots.alive[slot])
    ev = w.step(1.5)
    assert bool(ev.expired[slot]) and not bool(w.w.slots.alive[slot])
    assert float(ev.gold.sum()) == 0.0


def test_ward_dies_after_three_hits_and_pays_the_killer():
    w = World(OUTSIDE, BRUSH_EDGE)
    slot = w.place(0, BRUSH)
    w.run(11.0)                                                  # past the 10 s early-detection window
    for k in range(2):
        ev = w.step(hits={slot: (1, 1)})
        assert not bool(ev.killed[slot])
    assert float(w.w.slots.hp[slot]) == 1.0
    ev = w.step(hits={slot: (1, 1)})
    assert bool(ev.killed[slot]) and int(ev.killer[slot]) == 1
    assert float(ev.gold[1]) == 10.0 and float(ev.gold[0]) == 0.0 and float(ev.xp[1]) == 0.0
    assert not bool(w.w.slots.alive[slot])
    # Own team cannot damage it; an early hit pays 5 g out of the 10 g bounty.
    w = World(OUTSIDE, BRUSH_EDGE)
    slot = w.place(0, BRUSH)
    assert float(w.step(hits={slot: (1, 0)}).gold.sum()) == 0.0 and float(w.w.slots.hp[slot]) == 3.0
    ev = w.step(hits={slot: (1, 1)})
    assert float(ev.gold[1]) == 5.0
    w.step(hits={slot: (1, 1)})
    ev = w.step(hits={slot: (1, 1)})
    assert bool(ev.killed[slot]) and float(ev.gold[1]) == 5.0    # 10 g total


def test_control_ward_four_hits_regen_and_one_per_player():
    w = World(BRUSH_EDGE, OUTSIDE, control=(3, 0))
    a = w.place(0, BRUSH, WD.REQ_CONTROL)
    assert float(w.w.slots.hp[a]) == 4.0 and not np.isfinite(float(w.w.slots.expires_at[a]))
    w.run(11.0)
    w.step(hits={a: (1, 1)})
    assert float(w.w.slots.hp[a]) == 3.0
    w.run(8.0)
    assert float(w.w.slots.hp[a]) == 3.0
    w.run(1.5)                                                   # 6 s quiet + 3 s period
    assert float(w.w.slots.hp[a]) == 4.0
    for _ in range(3):
        w.step(hits={a: (1, 1)})
    ev = w.step(hits={a: (1, 1)})
    assert bool(ev.killed[a]) and float(ev.gold[1]) == 30.0
    b = w.place(0, BRUSH, WD.REQ_CONTROL)
    w.step()
    ev = w.step(req={0: (WD.REQ_CONTROL, IN_BRUSH)})             # second one replaces the first
    assert bool(ev.replaced[b]) and int(ev.placed_slot[0]) == b
    assert int(jnp.sum(w.w.slots.alive & (w.w.slots.type == WD.WardType.CONTROL))) == 1


def test_three_totems_per_player_oldest_replaced():
    w = World(OUTSIDE, FAR)
    slots = []
    for k in range(4):
        w.w = w.w._replace(trinket=w.w.trinket._replace(charges=w.w.trinket.charges.at[0].set(2)))
        w.run(1.3, dt=1.3)                                       # past the 1.25 s activation lockout
        p = (BRUSH[0] + 40.0 * k, BRUSH[1])
        ev = w.step(req={0: (WD.REQ_TRINKET, p)})
        assert int(ev.code[0]) == WD.OK
        slots.append(int(ev.placed_slot[0]))
        if k == 3:
            assert bool(ev.replaced[slots[0]]) and slots[3] == slots[0]
    assert int(jnp.sum(w.w.slots.alive)) == 3
    assert int(w.step(req={0: (WD.REQ_TRINKET, BRUSH)}).code[0]) == WD.ERR_LOCKED


def test_oracle_sweep_reveals_and_disables_stealth_ward_with_linger():
    w = World(OUTSIDE, IN_BRUSH, trinkets=(WD.TOTEM_ITEM, WD.ORACLE_ITEM))
    slot = w.place(0, BRUSH)
    w.move(0, FAR)
    w.run(3.0)
    vis, view = w.vis()
    assert bool(vis[0, 1]) and not bool(vis[1, unit(slot)])
    w.move(1, BRUSH_EDGE)
    ev = w.step(req={1: (WD.REQ_TRINKET,)})
    assert bool(ev.sweep_started[1]) and int(w.w.trinket.charges[1]) == 0
    vis, view = w.vis()
    assert bool(vis[1, unit(slot)])                              # revealed (sees into brush)
    assert bool(view.disabled[slot]) and not bool(vis[0, 1])     # and blinded
    w.move(1, (BRUSH[0] + 700.0, BRUSH[1] - 700.0))              # leaves: out of the sweep radius
    w.step(1.0)
    vis, view = w.vis()
    assert not bool(vis[1, unit(slot)]) and bool(view.disabled[slot])   # 2 s linger
    w.step(1.5)
    vis, view = w.vis()
    assert not bool(view.disabled[slot])
    w.run(6.0)                                                   # sweep over
    w.move(1, IN_BRUSH)
    assert bool(w.vis()[0][0, 1])                                # the ward sees Jax again


def test_oracle_radius_breakpoints():
    lv = jnp.asarray([1, 4, 5, 7, 8, 11, 14, 16, 17, 18])
    assert np.asarray(WD.oracle_radius(lv)).tolist() == [600, 600, 630, 630, 660, 690, 720, 720, 750, 750]


def test_farsight_ward_visible_unobstructed_and_self_destructs():
    w = World(LANE_MID, FAR, trinkets=(WD.FARSIGHT_ITEM, WD.TOTEM_ITEM), level=(9, 9))
    w.w = w.w._replace(trinket=w.w.trinket._replace(charges=w.w.trinket.charges.at[0].set(1)))
    target = (LANE_MID[0] + 2400.0, LANE_MID[1] - 2400.0)
    slot = w.place(0, target)
    _, view = w.vis()
    assert float(view.sight_radius[slot]) == 800.0 and bool(view.unobstructed[slot])
    assert not bool(view.stealthed[slot]) and float(view.hp[slot]) == 1.0
    assert int(w.w.trinket.charges[0]) == 0
    w.run(2.5)
    assert float(w.vis()[1].sight_radius[slot]) == 500.0
    w.move(1, (target[0] + 400.0, target[1]))                    # enemy champion walks into it
    w.step()
    assert np.isfinite(float(w.w.slots.triggered_at[slot]))
    assert float(w.vis()[1].sight_radius[slot]) == 800.0
    w.run(2.9, dt=0.1)
    assert bool(w.w.slots.alive[slot])
    ev = w.step(0.2)
    assert bool(ev.expired[slot]) and float(ev.gold.sum()) == 0.0


def test_trinket_swap_keeps_time_equivalent():
    w = World(OUTSIDE, FAR)
    w.w = w.w._replace(trinket=w.w.trinket._replace(charges=jnp.asarray([1, 1], jnp.int32),
                                                    progress=jnp.asarray([0.5, 0.0], jnp.float32)))
    w.tid = jnp.asarray([WD.ORACLE_ITEM, WD.TOTEM_ITEM], jnp.int32)
    w.step(1e-6)
    # 1.5 * 210 s = 315 s of Totem time = 1.97 Oracle charges (160 s).
    assert int(w.w.trinket.charges[0]) == 1
    assert float(w.w.trinket.progress[0]) == pytest.approx(315.0 / 160.0 - 1.0, abs=1e-3)
    assert int(w.w.trinket.trinket[0]) == WD.ORACLE_ITEM


def test_deep_ward_in_enemy_jungle_and_river_from_level_9():
    w = World(FAR, (BLUE_JUNGLE[0], BLUE_JUNGLE[1] + 400.0), pages=(None, DEEP_PAGE))
    slot = w.place(1, BLUE_JUNGLE)
    assert bool(w.w.slots.deep[slot]) and float(w.w.slots.hp[slot]) == 4.0
    dur = float(w.w.slots.expires_at[slot] - w.w.slots.placed_at[slot])
    assert dur == pytest.approx(90.0 + 45.0, abs=1e-3)
    river = (TOP_RIVER[0], TOP_RIVER[1] + 300.0)
    w = World(FAR, river, pages=(None, DEEP_PAGE))
    slot = w.place(1, TOP_RIVER)
    assert not bool(w.w.slots.deep[slot]) and float(w.w.slots.hp[slot]) == 3.0
    w = World(FAR, river, pages=(None, DEEP_PAGE), level=(9, 9))
    slot = w.place(1, TOP_RIVER)
    assert bool(w.w.slots.deep[slot])
    # Without the rune, or in one's own jungle, nothing changes.
    w = World((BLUE_JUNGLE[0], BLUE_JUNGLE[1] + 400.0), FAR, pages=(DEEP_PAGE, None))
    slot = w.place(0, BLUE_JUNGLE)
    assert not bool(w.w.slots.deep[slot])


def test_sixth_sense_tracks_and_reveals_from_level_11():
    for level, revealed in ((10, False), (11, True)):
        w = World(FAR, IN_BRUSH, pages=(SIXTH_PAGE, None), level=(level, level))
        slot = w.place(1, BRUSH)
        w.run(3.0)
        w.move(0, OUTSIDE)                                       # 450 u from the stealthed ward
        ev = w.step()
        assert bool(ev.sensed[0]) and bool(w.w.slots.tracked[slot])
        vis, view = w.vis()
        assert bool(view.exposed[slot]) == revealed and bool(vis[0, unit(slot)]) == revealed
        assert float(w.w.trinket.sixth_cd_until[0]) == pytest.approx(w.t + 250.0, abs=1e-3)
        assert not bool(w.step().sensed[0])                      # on cooldown


def test_turret_true_sight_and_stealth_masks_are_optional():
    _, vg = grids()
    # Blue ward (stealthed) next to a red turret: the turret's 1100 true sight shows it.
    x = jnp.asarray([FAR[0], IN_BRUSH[0], BRUSH[0], OUTSIDE[0]], jnp.float32)
    y = jnp.asarray([FAR[1], IN_BRUSH[1], BRUSH[1], OUTSIDE[1]], jnp.float32)
    kind = jnp.asarray([W.KIND_CHAMPION, W.KIND_CHAMPION, W.KIND_WARD, W.KIND_TURRET], jnp.int32)
    sub = jnp.zeros((4,), jnp.int32)
    team = jnp.asarray([0, 1, 0, 1], jnp.int32)
    alive = jnp.ones((4,), bool)
    st = jnp.asarray([False, False, True, False])
    vis, _ = MV.visibility(x, y, kind, sub, team, alive, MV.init_reveal(2), 0.0, vg, n_fogged=3, stealthed=st)
    assert bool(vis[1, 2])
    far_turret = x.at[3].set(FAR[0] + 3000.0)
    vis, _ = MV.visibility(far_turret, y, kind, sub, team, alive, MV.init_reveal(2), 0.0, vg, n_fogged=3,
                           stealthed=st)
    assert not bool(vis[1, 2]) and bool(vis[0, 1])               # Jax (same brush) seen by the ward
    vis2, _ = MV.visibility(far_turret, y, kind, sub, team, alive, MV.init_reveal(2), 0.0, vg, n_fogged=3)
    assert bool(vis2[1, 2])                                      # no stealth mask: plain unit
    assert float(MV.sight_radius(jnp.int32(W.KIND_WARD), jnp.int32(2), True)) == 500.0


def test_vision_items_and_runes_are_classified_as_world():
    from lanerl_jax.modern.items import effects as IE
    from lanerl_jax.modern.runes import effects as RE
    items = IE.coverage_report()
    for iid in (2055, 3340, 3363, 3364):
        assert items[iid].startswith("WORLD wards") and iid not in IE.DEFERRED
    runes = RE.coverage_report()
    for pid in (8137, 8141):
        assert runes[pid].startswith("WORLD") and pid not in RE.DEFERRED
