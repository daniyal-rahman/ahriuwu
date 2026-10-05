"""Integration: the modern world tick (``world.tick``) end to end.

These tests check *symptoms across systems* rather than single functions:
gold/XP/levels after real waves, a champion kill paying first blood and
setting the death timer, shop purchases changing stats, Flash and Recall
moving the champion, turrets defending, and nothing overflowing. The world and
its one compiled tick program come from ``tests.world_harness`` (shared with the
other full-tick modules; compiling takes a few minutes on CPU).
"""
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.modern import economy as E
from lanerl_jax.modern.core import types as W
from lanerl_jax.modern.items.catalog import catalog
from lanerl_jax.modern.tests import world_harness as H
from lanerl_jax.modern.world import config as MW

if not H.artifacts_present():
    pytest.skip("modern map/route artifacts not present", allow_module_level=True)

from lanerl_jax.modern import world as MS  # noqa: E402

LONG_SWORD, DORAN_BLADE, HEALTH_POTION = 1036, 1055, 2003
assert H.ITEMS == (DORAN_BLADE, HEALTH_POTION)
orders = H.orders


def world():
    return H.world(), H.step, H.run


def test_layout_and_structure_vulnerability():
    cfg, _, _ = world()
    s = MS.init_state(cfg)
    kinds = np.asarray(cfg.unit_kind)
    assert cfg.n_units == cfg.layout.n_units == 2 + 3 * 40 + 40 + 8 + 2 * W.MAX_WARDS_PER_TEAM + 30 == 216
    assert (kinds == W.KIND_TURRET).sum() == 22 and (kinds == W.KIND_INHIBITOR).sum() == 6
    assert (kinds == W.KIND_NEXUS).sum() == 2
    targ = np.asarray(s.targetable)
    tier = np.asarray(cfg.unit_sub)
    turret = kinds == W.KIND_TURRET
    # Only outer turrets are vulnerable at game start (TOWERS §2).
    assert targ[turret & (tier == 0)].all() and not targ[turret & (tier > 0)].any()
    assert not targ[(kinds == W.KIND_INHIBITOR) | (kinds == W.KIND_NEXUS)].any()


def test_ambient_gold_waves_and_no_overflow():
    cfg, _, run = world()
    s = MS.init_state(cfg)
    s, (po, mo) = run(s, MS.no_orders(), 3 * 1000)       # 100 s (champions idle in the fountain)
    assert int(po.max()) == 0 and int(mo.max()) == 0
    gold = np.asarray(s.econ.gold)
    np.testing.assert_allclose(gold, 500 + float(E.ambient_payments(0.0, float(s.t))), atol=0.05)
    minions = (np.asarray(s.kind) == W.KIND_MINION) & np.asarray(s.alive)
    assert minions.sum() > 0 and (np.asarray(s.team)[minions] == 0).any() and (np.asarray(s.team)[minions] == 1).any()


def test_shop_purchase_in_fountain_changes_stats():
    cfg, step, _ = world()
    s = MS.init_state(cfg)
    s, e = step(s, orders(buy=[LONG_SWORD, 0]))
    assert int(e.shop_code[0]) == 0
    assert float(s.econ.gold[0]) == pytest.approx(500.0 - catalog()[LONG_SWORD].total, abs=0.01)
    rows = np.asarray(s.champ.inventory.item[0])
    assert catalog().row(LONG_SWORD) in rows.tolist()


def test_flash_blinks_and_goes_on_cooldown():
    cfg, step, run = world()
    s = MS.init_state(cfg)
    s, _ = run(s, MS.no_orders(), 16 * 30)                # start-of-game summoner cooldown is 15 s
    x0, y0 = float(s.x[0]), float(s.y[0])
    s, _ = step(s, orders(summoner_slot=[0, -1], summoner_x=[x0 + 1000.0, 0.0], summoner_y=[y0 + 1000.0, 0.0]))
    moved = np.hypot(float(s.x[0]) - x0, float(s.y[0]) - y0)
    assert 300.0 < moved <= 400.0 + 1.0
    assert float(s.summoners.ready_at[0, 0]) > float(s.t) + 250.0


def test_champion_kill_first_blood_and_death_timer():
    cfg, step, run = world()
    s = MS.init_state(cfg)
    # Put Jax next to Garen at the top-lane midpoint with almost no HP.
    lane = np.asarray(cfg.lane_path)
    mx, my = lane[len(lane) // 2]
    s = s._replace(x=s.x.at[0].set(mx).at[1].set(mx + 150.0), y=s.y.at[0].set(my).at[1].set(my),
                   hp=s.hp.at[1].set(5.0), t=jnp.float32(200.0))
    s = H.refresh(s)
    kill_t = None
    for k in range(60):
        s, e = step(s, orders(attack=[1, -1]))
        if not bool(s.alive[1]):
            kill_t = float(s.t)
            break
    assert kill_t is not None
    fb = float(E.kill_gold(0.0, 1, True))                  # 400 at victim level 1
    assert float(e.economy.gold_gained[0]) == pytest.approx(fb + float(E.ambient_payments(kill_t - cfg.dt, kill_t)),
                                                            abs=0.05)
    assert float(e.economy.death_duration[1]) == pytest.approx(float(E.death_time(1, kill_t)))
    assert bool(s.econ.first_blood_done)
    # Respawn at the fountain after the timer.
    s, _ = run(s, MS.no_orders(), int(10.5 * 30))
    assert bool(s.alive[1])
    assert np.hypot(float(s.x[1]) - MW.FOUNTAINS[1][0], float(s.y[1]) - MW.FOUNTAINS[1][1]) < 1.0


def test_recall_returns_to_fountain():
    cfg, step, run = world()
    s = MS.init_state(cfg)
    lane = np.asarray(cfg.lane_path)
    mx, my = lane[len(lane) // 2]
    s = s._replace(x=s.x.at[0].set(mx), y=s.y.at[0].set(my))
    s, _ = step(s, orders(recall=[True, False]))
    s, _ = run(s, MS.no_orders(), round((E.RECALL_CAST + E.RECALL_CHANNEL) * 30) + 3)   # 0.5 s cast + 8 s
    assert np.hypot(float(s.x[0]) - MW.FOUNTAINS[0][0], float(s.y[0]) - MW.FOUNTAINS[0][1]) < 1.0


def test_idle_champions_auto_attack_in_range():
    cfg, _, run = world()
    s = MS.init_state(cfg)
    lane = np.asarray(cfg.lane_path)
    mx, my = lane[len(lane) // 2]
    s = s._replace(x=s.x.at[0].set(mx).at[1].set(mx + 200.0), y=s.y.at[0].set(my).at[1].set(my),
                   t=jnp.float32(20.0))
    s = H.refresh(s)
    hp0 = np.asarray(s.hp[:2])
    s, _ = run(s, MS.no_orders(), 3 * 30)
    hp = np.asarray(s.hp[:2])
    assert (hp < hp0 - 50.0).all()                        # both traded basic attacks without orders
