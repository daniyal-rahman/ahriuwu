"""Modern-world observation and action decoder (profile ``modern-world-v1``).

The checks are cross-module round trips rather than restatements of the
builder: the entity offsets must reconstruct world positions through the lane
frame, a click aimed at the slotted enemy must decode to an attack on that
unit for either team, and the self block must follow state changes that the
tick itself would make (cooldowns, casts, death, quest).
"""
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.sim.tests import modern_world_harness as H

if not H.artifacts_present():
    pytest.skip("modern map/route artifacts not present", allow_module_level=True)

from lanerl_jax.obs import modern_builder as OB  # noqa: E402
from lanerl_jax.obs.builder import GLOBAL_DIM, N_SLOTS, NORM_DIST  # noqa: E402
from lanerl_jax.sim import modern_step as MS  # noqa: E402
from lanerl_jax.train import modern_actions as MA  # noqa: E402
from lanerl_jax.train.actions import _screen_to_centred_lane  # noqa: E402

SELF = {name: i for i, name in enumerate((
    "s", "n", "hp", "level", "gold", "cs", "cd_q", "cd_w", "cd_e", "cd_r", "ad", "ap", "armor", "mr", "dead",
    "recalling", "garen", "jax", "enemy_garen", "enemy_jax", "mana", "shield", "summ_d", "summ_f", "next_level",
    "quest", "quest_done", "in_combat", "ms", "range"))}


@lru_cache(maxsize=1)
def world():
    """Shared world and visibility refresh (``modern_world_harness``) plus one jitted builder and decoder."""
    cfg = H.world()
    frames = OB.modern_frames(cfg)
    obs = jax.jit(lambda s: tuple(OB.build_modern_observation(s, me, frames[me], cfg) for me in (0, 1)))
    decode = jax.jit(lambda a, s: MA.modern_orders_from(a, s, frames))
    return cfg, frames, obs, decode, H.refresh


def lane_state(gap=250.0):
    cfg, *_, refresh = world()
    s = MS.init_state(cfg)
    lane = np.asarray(cfg.lane_path)
    mx, my = lane[len(lane) // 2]
    return refresh(s._replace(x=s.x.at[0].set(mx).at[1].set(mx + gap), y=s.y.at[0].set(my).at[1].set(my),
                              t=jnp.float32(30.0)))


def test_shapes_identity_and_fountain_start():
    cfg, _, obs, _, _ = world()
    o0, o1 = obs(MS.init_state(cfg))
    for o in (o0, o1):
        assert o.entities.shape == (N_SLOTS, OB.MODERN_ENTITY_DIM) and o.global_vec.shape == (GLOBAL_DIM,)
        assert o.self_vec.shape == (OB.MODERN_WORLD_SELF_DIM,)
        assert np.isfinite(np.asarray(o.self_vec)).all() and np.isfinite(np.asarray(o.entities)).all()
        assert float(o.self_vec[SELF["level"]]) == pytest.approx(1 / 20)
        assert float(o.global_vec[1]) == 0.0                     # enemy is a map away
        np.testing.assert_allclose(np.asarray(o.global_vec[2:]), 1.0)  # no enemy cast seen yet
    assert float(o0.self_vec[SELF["garen"]]) == 1.0 and float(o0.self_vec[SELF["enemy_jax"]]) == 1.0
    assert float(o1.self_vec[SELF["jax"]]) == 1.0 and float(o1.self_vec[SELF["enemy_garen"]]) == 1.0
    # Both start in their own base: lane progress is behind the own outer turret (s < 0) for both.
    assert float(o0.self_vec[SELF["s"]]) < 0 and float(o1.self_vec[SELF["s"]]) < 0


def test_entity_offsets_reconstruct_world_positions():
    cfg, frames, obs, _, _ = world()
    s = lane_state()
    for me, o in enumerate(obs(s)):
        f = frames[me]
        valid = np.asarray(o.entities[:, 0]) > 0
        assert valid[0] and int(o.slot_unit[0]) == 1 - me        # enemy champion slot
        units = np.asarray(o.slot_unit)[valid]
        ds, dn = np.asarray(o.entities[valid, 1]) * NORM_DIST, np.asarray(o.entities[valid, 2]) * NORM_DIST
        wx = float(s.x[me]) + ds * float(f.axis[0]) + dn * float(f.normal[0])
        wy = float(s.y[me]) + ds * float(f.axis[1]) + dn * float(f.normal[1])
        np.testing.assert_allclose(wx, np.asarray(s.x)[units], atol=0.5)
        np.testing.assert_allclose(wy, np.asarray(s.y)[units], atol=0.5)
        types = np.asarray(o.entities[valid, 4:10])
        assert (types.sum(1) == 1).all() and types[0, 0] == 1    # champion one-hot
        assert np.asarray(o.entities[0, 11]) == 1.0                # enemy team


def _bin_for(ds, dn):
    gx, gy = np.meshgrid(np.arange(96), np.arange(54), indexing="ij")
    a, b = _screen_to_centred_lane(jnp.asarray((gx + 0.5) / 96, jnp.float32), jnp.asarray((gy + 0.5) / 54, jnp.float32))
    i = np.argmin((np.asarray(a) - ds) ** 2 + (np.asarray(b) - dn) ** 2)
    return int(gx.flat[i]), int(gy.flat[i])


def test_click_on_observed_enemy_attacks_it_for_both_teams():
    _, _, obs, decode, _ = world()
    s = lane_state(gap=300.0)
    o = obs(s)
    bins = [_bin_for(float(o[me].entities[0, 1]) * NORM_DIST, float(o[me].entities[0, 2]) * NORM_DIST)
            for me in (0, 1)]
    mv = MA.MODERN_BUTTON_INDEX["move"]
    act = (jnp.asarray([mv, mv]), jnp.asarray([b[0] for b in bins]), jnp.asarray([b[1] for b in bins]))
    orders = decode(act, s)
    assert np.asarray(orders.attack).tolist() == [1, 0]
    assert not np.asarray(orders.move).any()
    # Q on the enemy targets it; Flash takes the cursor point.
    q, d = MA.MODERN_BUTTON_INDEX["q"], MA.MODERN_BUTTON_INDEX["summoner_d"]
    orders = decode((jnp.asarray([q, d]), act[1], act[2]), s)
    assert np.asarray(orders.cast_slot).tolist() == [0, -1] and int(orders.cast_target[0]) == 1
    assert np.asarray(orders.summoner_slot).tolist() == [-1, 0]
    assert np.hypot(float(orders.summoner_x[1]) - float(s.x[0]), float(orders.summoner_y[1]) - float(s.y[0])) < 150


def test_ground_click_moves_and_minimap_is_noop():
    _, frames, _, decode, _ = world()
    s = lane_state(gap=2000.0)
    mv = MA.MODERN_BUTTON_INDEX["move"]
    bx, by = _bin_for(600.0, 0.0)                                 # 600 u down-lane, on the lane line
    orders = decode((jnp.asarray([mv, mv]), jnp.asarray([bx, bx]), jnp.asarray([by, by])), s)
    assert np.asarray(orders.move).all() and (np.asarray(orders.attack) == -1).all()
    for me in (0, 1):
        dx, dy = float(orders.move_x[me]) - float(s.x[me]), float(orders.move_y[me]) - float(s.y[me])
        ds = dx * float(frames[me].axis[0]) + dy * float(frames[me].axis[1])
        assert ds == pytest.approx(600.0, abs=40.0)              # towards each side's enemy
    orders = decode((jnp.asarray([mv, mv]), jnp.asarray([95, 95]), jnp.asarray([53, 53])), s)
    assert not np.asarray(orders.move).any()


def test_self_block_tracks_cooldowns_casts_death_and_quest():
    _, _, obs, _, refresh = world()
    s = lane_state()
    c = s.champ
    s = s._replace(champ=c._replace(ranks=c.ranks.at[0].set(jnp.asarray([1, 0, 0, 0], c.ranks.dtype)),
                                    cooldowns=c.cooldowns.at[0, 0].set(4.0), seen_cast=c.seen_cast.at[1, 2].set(29.0)),
                   alive=s.alive.at[0].set(False),
                   econ=s.econ._replace(quest=s.econ.quest._replace(points=s.econ.quest.points.at[0].set(600.0))))
    o0, o1 = obs(refresh(s))
    assert 0.0 < float(o0.self_vec[SELF["cd_q"]]) <= 1.0      # Garen Q rank 1 on cooldown
    assert float(o0.self_vec[SELF["cd_w"]]) == 1.0             # unranked slot reads as unavailable
    assert float(o0.self_vec[SELF["dead"]]) == 1.0 and float(o0.self_vec[SELF["quest"]]) == pytest.approx(0.5)
    assert float(o0.global_vec[2 + 2]) < 0.2                   # Garen saw Jax E 1 s ago
    assert float(o1.global_vec[1]) == 0.0                      # dead Garen is not slotted for Jax


def test_shop_view_and_shop_skill_ward_buttons():
    cfg, _, obs, decode, _ = world()
    from lanerl_jax.sim.modern_item_data import catalog
    s = MS.init_state(cfg)                                     # both champions in their fountain with 500 g
    o0, _ = obs(s)
    cat = catalog()
    afford = np.asarray(o0.affordable)
    assert afford[cat.row(1036)] and not afford[cat.row(3031)]  # Long Sword yes, Infinity Edge no
    assert int(o0.inventory[6]) >= 0                             # trinket slot holds the trinket
    b = MA.MODERN_BUTTON_INDEX
    act = (jnp.asarray([b["buy"], b["level_w"]]), jnp.asarray([48, 48]), jnp.asarray([27, 27]),
           jnp.asarray([cat.row(1036), 0]))
    orders = decode(act, s)
    assert np.asarray(orders.buy).tolist() == [1036, 0] and np.asarray(orders.level_up).tolist() == [-1, 1]
    act = (jnp.asarray([b["ward"], b["attack_move"]]), jnp.asarray([50, 60]), jnp.asarray([20, 20]))
    orders = decode(act, s)
    assert np.asarray(orders.ward_kind).tolist() == [0, -1] and np.asarray(orders.attack_move).tolist() == [False, True]
    assert not np.asarray(orders.move).any()
