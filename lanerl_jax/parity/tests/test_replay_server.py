"""Replay controls preserve recorded timing and never overwrite evidence."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lanerl_jax.parity import replay_server


@pytest.mark.parametrize('trace', [False, True])
def test_replay_prefix_timing_trace_and_output_guard(tmp_path, monkeypatch, trace):
    from lanerl_jax.train import run_manifest
    from lanerl_train import paths
    log = SimpleNamespace(meta={'step_ticks': 6})
    class Log:
        meta = log.meta
        def __len__(self): return 9
    monkeypatch.setattr(replay_server.PolicyActionLog, 'load', lambda p: Log())
    monkeypatch.setattr(replay_server, 'ReplayWireDriver', lambda log: SimpleNamespace(unresolved=0))
    monkeypatch.setattr(run_manifest, 'git_provenance', lambda: {})
    monkeypatch.setattr(run_manifest, 'file_sha256', lambda p: 'fixture')
    monkeypatch.setattr(replay_server.subprocess, 'check_output', lambda *a, **k: b'')
    monkeypatch.setattr(paths, 'server_dir', lambda: Path('/fixture/a/b/c/d'))
    monkeypatch.setattr(replay_server, 'scan_cast_freeze', lambda frames:
                        SimpleNamespace(invalid=False, report=lambda: {'invalid': False}))
    calls = []
    def record(out, **kwargs):
        calls.append(kwargs)
        (out/'replay_obs.jsonl').write_text('{}\n')
    monkeypatch.setattr(replay_server, 'record_trace', record)
    args = SimpleNamespace(actions=tmp_path/'actions', out=tmp_path/'out', decisions=3,
        decision_trace=trace, server_dir=tmp_path, config=tmp_path/'config', port_base=49900)
    replay_server.run_replay(args)
    assert calls[0]['decisions'] == 3 and calls[0]['step_ticks'] == 6
    assert calls[0]['extra_env']['LANERL_DECISION_TRACE'] == str(int(trace))
    assert json.loads((args.out/'provenance.json').read_text())['decisions'] == 3
    with pytest.raises(FileExistsError): replay_server.run_replay(args)
    assert len(calls) == 1
    args.out = tmp_path/'invalid'; args.decisions = 10
    with pytest.raises(ValueError, match='recorded stream'): replay_server.run_replay(args)
    assert not args.out.exists()


def test_prefix_comparison_detects_wire_and_tick_changes(tmp_path):
    original, off, on = [tmp_path/name for name in ('original', 'off', 'on')]
    frame = {'t': 16, 'u': [{'k': 'Champion', 'tm': 100, 'id': 7, 'cs': 0}]}
    for directory, tag in ((original, 'policy'), (off, 'replay'), (on, 'replay')):
        (directory/tag).mkdir(parents=True)
        (directory/f'{tag}_obs.jsonl').write_text(json.dumps(frame)+'\n')
        (directory/tag/'instance000.log').write_text(
            'LANERL_STATEHASH t=17 n=1 h=0123456789abcdef\n' +
            ('LANERL_DECISION t=17 k=FinishCasting id=7 spell=Attack auto=True instant=False\n'
             if directory==on else ''))
    result = replay_server.compare_replay_prefix(original, off, on, 1)
    assert result['canonical_off_on_equal'] and result['canonical_original_off_equal']
    assert result['events']['on']['blue_auto_finish_casting'] == 1
    assert result['wire_comparisons']['off_on']['equal']
    frame['u'][0]['cs'] = 1
    (on/'replay_obs.jsonl').write_text(json.dumps(frame)+'\n')
    (on/'replay/instance000.log').write_text('LANERL_STATEHASH t=17 n=1 h=1123456789abcdef\n')
    result = replay_server.compare_replay_prefix(original, off, on, 1)
    assert not result['wire_comparisons']['off_on']['equal']
    assert not result['canonical_off_on_equal']
