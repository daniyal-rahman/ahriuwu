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


def test_actions_preserve_native_cross_field_order_and_special_motion():
    from lanerl_jax.modern.bc.replay import _actions
    class Fields:
        def __init__(self):
            self.native = dict(fields={n: dict(update_index=[order], event_index=[order + 100]) for n, order in
                [('movement.received_path', 20), ('event.basic_attack_execution_proxy', 30),
                 ('event.ability_cast_received_proxy', 10)]})
        def rows(self, name, family):
            return [0], [.2], [1]
        def value(self, name, row):
            if name == 'movement.received_path':
                return dict(waypoints_xz=[[0., 0.], [100., 0.]], flags=4, special_motion_parameters={'speed': 1000})
            if name == 'event.basic_attack_execution_proxy':
                return dict(target_id=2)
            return dict(slot=0, targeting_mode='self')
    actions, _ = _actions(Fields(), 1, 2)
    assert [a.kind for a in actions] == ['cast', 'path', 'attack']
    assert [a.source_order for a in actions] == [10, 20, 30]
    assert actions[1].special_motion and actions[1].source_event == 120


def test_position_requires_acceptance_and_current_lifetime():
    from lanerl_jax.modern.bc.replay import _read_lane_units
    class Fields:
        def rows(self, name, family=None):
            if name == 'event.unit_death_received_proxy':
                return [0], [1.], [1]
            return [], [], []
        def value(self, name, row):
            return dict(victim_id=1)
        def asof(self, name, uid, t):
            return np.ones(len(t)) * 100.
        def series(self, name, uid):
            return dict(time_seconds=np.array([0., 2.]), values=np.array([100., 100.]),
                        available=np.ones(2, bool))
    class Model:
        def __init__(self, *args, **kwargs):
            pass
        def speed(self, *args):
            pass
        def estimate(self, uid, t):
            return dict(position_available=True, position_xz=[100., 200.],
                        policy_position_accepted=t != .5, approximation_accepted_for_bc=t != .5,
                        source=dict(movement_clock=2.1 if t == 3. else 0.))
    result = _read_lane_units(Fields(), SimpleNamespace(MinionPositions=Model),
                             [Unit(1, 'champion', 0, 'Garen')], np.array([.5, 1.5, 2.5, 3.]), 2)
    assert np.isnan(result[5][:3]).all()  # unaccepted, dead, previous-life path
    assert result[5][3, 0] == 100.


def test_damage_roles_and_observation_cutoff_are_required():
    from lanerl_jax.modern.bc.replay import _damage_notices, _last_hits, _champion_combat
    class Fields:
        qualified = False
        clock = 10.02
        def rows(self, name, family=None):
            return np.array([0]), np.array([self.clock]), np.array([50])
        def value(self, name, row):
            return dict(receiver_id=50, participant_ids=[50, 1], victim_id=50, source_id=1,
                        victim_source_roles_available=self.qualified)
    fields = Fields()
    assert _damage_notices(fields) == []
    assert np.isnan(_champion_combat(fields, 1, {50}, np.array([10.]))).all()
    assert _last_hits(fields, 1, {50: [10.]}) == []
    fields.qualified = True
    assert _last_hits(fields, 1, {50: [10.]}) == []  # later receipt cannot credit earlier state
    fields.clock = 9.99
    assert _last_hits(fields, 1, {50: [10.]}) == [10.]


def test_unknown_combat_stays_unknown_in_formatter():
    g = fixture()
    g.last_champion_combat[:] = np.nan
    g.cs = np.full(2, np.nan)
    sv = build(g)[2]
    col = C._self_columns(OB.REPLAY_PROFILE)
    assert np.isnan(sv[:, [col['in_combat'], col['cs']]]).all()


def test_decoder_rejects_cached_modules_from_another_snapshot(tmp_path, monkeypatch):
    import sys
    from types import ModuleType
    from lanerl_jax.modern.bc import replay
    snapshot = tmp_path/'requested'
    snapshot.mkdir()
    (snapshot/'position_model.py').write_text('VERSION = "test"\n')
    foreign = ModuleType('position_model')
    foreign.__file__ = str(tmp_path/'other'/'position_model.py')
    monkeypatch.setitem(sys.modules, 'position_model', foreign)
    monkeypatch.setattr(replay, 'DECODER_SCRIPTS', snapshot)
    import pytest
    with pytest.raises(ValueError, match='fresh process'):
        replay._decoder()


def test_decoder_rejects_cached_transitive_dependency_before_import(tmp_path, monkeypatch):
    import sys
    from types import ModuleType
    from lanerl_jax.modern.bc import replay
    (tmp_path/'event_value_codec.py').write_text('')
    foreign = ModuleType('event_value_codec')
    foreign.__file__ = str(tmp_path/'other'/'event_value_codec.py')
    monkeypatch.setitem(sys.modules, 'event_value_codec', foreign)
    monkeypatch.setattr(replay, 'DECODER_SCRIPTS', tmp_path)
    import pytest
    with pytest.raises(ValueError, match='event_value_codec'):
        replay._decoder()
