from types import SimpleNamespace

import numpy as np

from lanerl_jax.modern import obs as OB
from lanerl_jax.modern.bc import convert as C, observation as RO
from lanerl_jax.modern.bc.replay import Unit
from lanerl_jax.modern.frame import make_lane_frame


def fixture():
    units = [Unit(1, "champion", 0, "Garen"), Unit(2, "champion", 1, "Jax"),
             Unit(3, "turret", 0, "outer")]
    g = SimpleNamespace(t=np.array([10., 11.]), units=units, me=1, opponent=2, opponent_name="Jax",
                        index=lambda uid: next(i for i, u in enumerate(units) if u.id == uid),
                        x=np.array([[100., 200., 300.], [100., 200., 300.]]),
                        y=np.zeros((2, 3)), hp=np.array([[100., 100., 50.], [100., 100., np.nan]]),
                        max_hp=np.full((2, 3), 100.), alive=np.ones((2, 3), bool),
                        spell_rank=np.ones((2, 2, 4)), level=np.full(2, 3.), gold=np.full(2, 10.),
                        cs=np.full(2, 4), champion_stats=np.full((2, 2, 11), 10.),
                        max_mana=np.array([0., np.nan]), mana=np.array([0., np.nan]), shield=np.zeros(2),
                        last_champion_combat=np.full(2, -1e9), summoner_slot={4: 1, 5: 0},
                        opponent_casts=[(10.5, 0)])
    return g


def build(g, timers=None):
    return RO.build(g, make_lane_frame((0., 0.), (1000., 0.), (0., -1000.)),
                    C._self_columns(OB.REPLAY_PROFILE), "top", 0, True,
                    np.full((2, 6), np.nan) if timers is None else timers)


def test_direct_formatter_never_consults_sim_or_kit_tables(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("sim-derived table was consulted")
    monkeypatch.setattr(RO.CT, "descriptors", forbidden)
    monkeypatch.setattr(RO.CT, "cooldowns", forbidden)
    entities, pad, sv, gv, slots, _ = build(fixture())
    assert entities.shape == (2, 32, 20) and sv.shape[1] == OB.self_dim(OB.REPLAY_PROFILE)
    assert np.isnan(entities[1, np.flatnonzero(slots[1] == 2)[0], 3])
    assert not (slots >= 3).any()  # no template Nexus, inhibitor or missing turret
    col = C._self_columns(OB.REPLAY_PROFILE)
    for name in ("recalling", "next_level", "quest", "quest_done", "skill_points", "trinket", "cd_q"):
        assert np.isnan(sv[:, col[name]]).all()
    assert sv[0, col["mana"]] == 0 and np.isnan(sv[1, col["mana"]])
    assert np.isnan(gv[0, 2]) and gv[1, 2] == np.float32(.5 / 60.)


def test_received_timer_scaling_and_canonical_summoner_keys():
    timers = np.full((2, 6), np.nan)
    timers[:, 0], timers[:, 4] = [60., 59.], [100., 99.]
    _, _, sv, _, _, _ = build(fixture(), timers)
    col = C._self_columns(OB.REPLAY_PROFILE)
    np.testing.assert_allclose(sv[:, col["cd_q"]], [1., 59. / 60.])
    np.testing.assert_allclose(sv[:, col["summ_f"]], [100. / 300., 99. / 300.])
    assert np.isnan(sv[:, col["summ_d"]]).all()


def test_leading_unknown_positions_are_not_backfilled():
    a = np.array([[np.nan, np.nan], [1., 2.], [np.nan, np.nan]])
    filled = C._fill(a)
    assert np.isnan(filled[0]).all()
    np.testing.assert_array_equal(filled[1:], [[1., 2.], [1., 2.]])


def test_slot4plus_origin_vector_is_not_a_cast_target():
    from lanerl_jax.modern.bc.replay import _cast_point
    for slot in (4, 5, 12):
        d = dict(slot=slot, wire_vectors_xyz={"0xd0": [100., 0., 200.], "0x2c": [500., 0., 600.]})
        assert _cast_point(d) is None  # neither vector is qualified as this slot's aim
        assert _cast_point(dict(d, target_point_world_xz=[700., 800.])) == (700., 800.)
    assert _cast_point(dict(slot=0, targeting_mode=None,
                           wire_vectors_xyz={"0x2c": [500., 0., 600.]})) == (500., 600.)
