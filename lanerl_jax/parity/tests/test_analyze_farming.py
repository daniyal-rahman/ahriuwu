from types import SimpleNamespace

from lanerl_jax.parity.analyze_farming import attack_transition


def state(attacking, finished):
    return SimpleNamespace(is_attacking=attacking, has_auto_attacked=finished)


def test_attack_cast_and_cancel_are_distinct():
    idle, windup, complete = state(False, False), state(True, False), state(True, True)
    assert attack_transition(None, windup) == []
    assert attack_transition(idle, windup) == ['attack_started']
    assert attack_transition(windup, idle) == ['attack_ended_without_observed_cast']
    assert attack_transition(windup, complete) == ['attack_cast_finished']
    assert attack_transition(complete, state(False, True)) == ['attack_ended_after_cast']
    # A completion may occur in the same recorded tick that attacking clears.
    assert attack_transition(windup, state(False, True)) == [
        'attack_cast_finished', 'attack_ended_after_cast']


def test_recall_flag_is_not_a_death_counter(tmp_path):
    import json
    from lanerl_jax.parity.analyze_farming import analyze

    (tmp_path / 'policy').mkdir()
    (tmp_path / 'evaluation.json').write_text(json.dumps({'observation_interface':'viewport-structured-v2'}))
    (tmp_path / 'policy_policy_actions.json').write_text(json.dumps(
        {'t_ms': [120000], 'blue': [{'t': 'noop'}]}))
    (tmp_path / 'policy_obs.jsonl').write_text(json.dumps({'t': 120000, 'u': [
        {'id': 1, 'k': 'Champion', 'tm': 100, 'x': 26, 'y': 280,
         'hp': 671, 'cs': 0, 'rc': 1}]}))
    (tmp_path / 'policy/instance000.log').write_text(
        'LANERL_STATEROW t=120000 Champion|100|416,4480|687973/687973|A|10|1|-|-|1||80010,37413,45220,353280,1024|1|486400|0|0|1|0:0|0:0|0:0|0:0\n')
    result = analyze(tmp_path)
    assert result['counts']['final_deaths'] == 0
    assert result['events'] == []


def test_replay_uses_actual_intervals_and_counter_edges(tmp_path):
    import json
    import numpy as np
    from lanerl_jax.parity.analyze_farming import analyze_replay

    d = dict(t_ms=np.array([120000, 120100, 120300, 120400]),
        x=np.array([[0, 2000, 100], [0, 2000, 1000], [0, 2000, 100], [0, 2000, 100]]),
        y=np.zeros((4, 3)), kind=np.tile([1, 1, 2], (4, 1)),
        team=np.tile([0, 1, 1], (4, 1)), alive=np.ones((4, 3), bool),
        cs=np.array([[0,0],[0,0],[1,0],[1,0]]),
        deaths=np.array([[0,0],[0,0],[0,0],[1,0]]),
        target=np.full((4, 3), -1), aa_target=np.full((4, 3), -1),
        spawn_seq=np.zeros((4, 3)), order_target=np.full((4, 2), -1),
        order_kind=np.zeros((4, 2)), is_attacking=np.zeros((4, 3), bool),
        has_auto_attacked=np.zeros((4, 3), bool))
    d['alive'][-1, 0] = False
    d['is_attacking'][1, 0] = True
    d['has_auto_attacked'][2:, 0] = True
    path = tmp_path/'trace.npz'
    np.savez(path, **d, metadata=json.dumps({'environment':'test'}))
    result = analyze_replay(path)
    assert result['alive_seconds'] == .4
    assert result['near_minion_seconds'] == .2
    assert result['max_gap_ms'] == 200
    assert result['counts']['attack_started'] == 1
    assert result['counts']['attack_cast_finished'] == 1
    assert result['counts']['attack_ended_after_cast'] == 1
    assert [(e['event'],e['t_ms']) for e in result['events'] if e['event'] in ('cs','deaths')] == [
        ('cs',120300),('deaths',120400)]

    clipped = analyze_replay(path, start_ms=120100)
    assert clipped['start_ms'] == 120100
    assert clipped['alive_seconds'] == .3
    assert clipped['near_minion_seconds'] == .1


def test_e_end_timing_does_not_turn_input_association_into_causality():
    import struct
    from lanerl_jax.parity.analyze_farming import ETimeline

    def phase(elapsed):
        bits = lambda x: struct.unpack('<I', struct.pack('<f', x))[0]
        return [('GarenE', bits(elapsed), bits(3.))]

    timeline = ETimeline([1100])
    for t, buff in [(0, []), (16, phase(0)), (1016, phase(1)), (1216, []),
                    (3016, phase(0)), (6000, phase(2.984)), (6016, [])]:
        timeline.add(t, buff, dead=False)
    report = timeline.report()
    assert report['counts']['observed_starts'] == 2
    assert report['counts']['observed_ends'] == 2
    assert report['counts']['premature_ends_with_nearby_e_input'] == 1
    assert report['counts']['expiry_compatible_ends'] == 1
    assert report['observed_active_seconds'] == 4.2
    assert report['confirmed_key_cancellations'] is None
    assert report['episodes'][0]['confirmed_key_cancellation'] is None

    # Missing buff instrumentation must not become a false observed E end.
    missing = ETimeline([])
    missing.add(0, phase(.5))
    missing.add(16, None)
    assert missing.report()['counts'].get('observed_ends', 0) == 0
    assert missing.report()['episodes'][0]['censor_reason'] == 'missing_phase_instrumentation'


def test_source_life_uses_authoritative_dead_and_labels_legacy_hp_proxy():
    import pytest
    from lanerl_jax.parity.analyze_farming import source_alive
    assert source_alive({'hp':35,'dead':True}, 'viewport-structured-v3') == (False,'authoritative_dead')
    assert source_alive({'hp':35,'dead':False}, 'viewport-structured-v3') == (True,'authoritative_dead')
    assert source_alive({'hp':35}, 'viewport-structured-v2') == (True,'legacy_hp_proxy')
    for interface in (None, 'viewport-structured-v3'):
        with pytest.raises(ValueError, match='missing authoritative'):
            source_alive({'hp':35}, interface)
    for invalid in (None, 0, 1, 'false'):
        with pytest.raises(ValueError, match='malformed authoritative'):
            source_alive({'hp':35,'dead':invalid}, 'viewport-structured-v2')
