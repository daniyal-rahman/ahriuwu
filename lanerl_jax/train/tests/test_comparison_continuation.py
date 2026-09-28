"""The one-shot comparison cannot select checkpoints or bypass the launcher."""
import json
from types import SimpleNamespace
import pytest
from ops import continue_heuristic_comparison as runner


def test_continuation_launches_exact_six_pairs_after_fixed_endpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    training = tmp_path/'training'; training.mkdir()
    checkpoint = training/'ckpt_000409600.msgpack'; checkpoint.write_bytes(b'fixture')
    final = {'checkpoints':[{'file':checkpoint.name,'update':400}]}
    monkeypatch.setattr(runner, 'wait_complete', lambda path, deadline: final)
    monkeypatch.setattr(runner, 'unique_run', lambda path:path/'manifest.json')
    monkeypatch.setattr(runner, 'read_manifest', lambda path:{'host':{'slurm_job':'fixture'}})
    records=[]
    monkeypatch.setattr(runner, 'record', lambda detail,states:records.append(detail))
    calls=[]
    def command(args, **kw):
        calls.append(list(map(str,args)))
        return SimpleNamespace(stdout=json.dumps({'improved_beyond_teacher_and_clone':False,'comparisons':{}}))
    monkeypatch.setattr(runner, 'command', command)
    runner.run_comparison(training)
    launch_calls=[a for a in calls if any(x.endswith('ops/launch.py') for x in a)]
    assert len(launch_calls)==12
    expected=[(e,str(s)) for e in runner.EVALS for s in (2,3)]
    for (dry,live),(experiment,seed) in zip(zip(launch_calls[::2],launch_calls[1::2]),expected):
        assert dry[:-1]==live and dry[-1]=='--dry-run'
        assert experiment in live and live[live.index('--seed')+1]==seed
        assert '--no-canary' not in live and '--no-watch' not in live
        if experiment.startswith('E27'):
            assert live[live.index('--resume')+1]==str(checkpoint)
        else:
            assert '--resume' not in live
    assert 'gate: FAIL' in records[-1]


def test_continuation_refuses_missing_fixed_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(runner,'wait_complete',lambda *args:{'checkpoints':[]})
    with pytest.raises(RuntimeError,match='predeclared u400'):
        runner.run_comparison(tmp_path)


def test_existing_evaluation_is_not_repeated(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    checkpoint=tmp_path/'ckpt_000409600.msgpack'; checkpoint.write_bytes(b'fixture')
    monkeypatch.setattr(runner,'wait_complete',lambda *args:{'checkpoints':[{'file':checkpoint.name,'update':400}]})
    monkeypatch.setattr(runner,'record',lambda *args:None)
    (tmp_path/'lanerl_jax/runs'/runner.EVALS[0]/'seed2').mkdir(parents=True)
    with pytest.raises(RuntimeError,match='duplicate evaluation'):
        runner.run_comparison(tmp_path)


def test_desktop_launch_overrides_login_cpu_cap():
    from ops import launch
    from pathlib import Path
    assert 'JAX_PLATFORMS=cuda' in launch.ENV and 'env -u XLA_FLAGS' in launch.ENV
    batch = (Path(__file__).resolve().parents[3] / 'slurm/server_train.sbatch').read_text()
    assert 'export JAX_PLATFORMS=cuda' in batch and 'unset XLA_FLAGS' in batch
