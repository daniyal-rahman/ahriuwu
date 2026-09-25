"""Cohorts accept complete consistent episodes, and stop at the first failure."""
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest

from lanerl_jax.train.server_eval_batch import read_json, run_cohort, sha256


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    from lanerl_train import paths
    checkpoint = tmp_path/'checkpoint.msgpack'
    checkpoint.write_bytes(b'fixed checkpoint fixture')
    policy = dict(action_interface='screen-click-v2', observation_interface='viewport-structured-v3')
    (tmp_path/'manifest.json').write_text(json.dumps({'config': {'train': {'policy': policy}}}))
    server = tmp_path/'server'; server.mkdir()
    (server/'GameServerLib.dll').write_bytes(b'fixed server fixture')
    config = tmp_path/'game.json'; config.write_text('{}')
    monkeypatch.setattr(paths, 'default_game_config', lambda: config)
    eval_cwd = tmp_path/'frozen-checkout'; eval_cwd.mkdir()
    args = SimpleNamespace(eval_cwd=eval_cwd, checkpoint=checkpoint, server_dir=server, out=tmp_path/'cohort',
        seeds=[2, 4, 7], seconds=.2, step_ticks=2, port_base=49500, start_near_wave=False)
    return args, config, policy


def child_recording(cmd, args, config, policy, *, code=b'pass\n'):
    out = Path(cmd[cmd.index('--out')+1]); out.mkdir()
    seed = int(cmd[cmd.index('--seed')+1])
    archive = out/'source.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        entry = tarfile.TarInfo('lanerl_jax/train/server_eval.py'); entry.size = len(code)
        tar.addfile(entry, io.BytesIO(code))
    times = [16 + round(i*1000/30) for i in range(6)]
    frames = [{'t': t, 'u': [dict(k='Champion', tm=100, cs=seed+1, hp=3,
                                 dead=i >= 2), dict(k='Champion', tm=200, cs=0, hp=600, dead=False)]}
              for i, t in enumerate(times)]
    (out/'policy_obs.jsonl').write_text(''.join(json.dumps(f)+'\n' for f in frames))
    actions = dict(t_ms=times, blue=[{}]*6, red=[{}]*6, blue_sim=[None]*6, red_sim=[None]*6,
                   meta=dict(seed=seed, deterministic=False))
    (out/'policy_policy_actions.json').write_text(json.dumps(actions))
    result = dict(seed=seed, seconds=args.seconds, step_ticks=2, decision_hz=30.,
        start_near_wave=False, checkpoint=str(args.checkpoint),
        checkpoint_sha256=sha256(args.checkpoint), environment='source-server',
        opponent='idle-fountain', observation_interface=policy['observation_interface'],
        freeze=dict(invalid=False, unchecked=False, frozen=[], frames=6),
        source=dict(server_binary_sha256=sha256(args.server_dir/'GameServerLib.dll'),
                    server_config_sha256=sha256(config), snapshot_sha256=sha256(archive)),
        behavior=dict(frames=6, champions={'100': {'deaths': 1}, '200': {'deaths': 0}}),
        cs={'100': seed+1, '200': 0}, wall_s=1.)
    (out/'evaluation.json').write_text(json.dumps(result))
    return out, result


def test_sequential_complete_cohort_preserves_every_score_without_success_claim(cohort):
    args, config, policy = cohort
    calls = []

    def runner(cmd, **kwargs):
        # Prior child must have been validated and persisted before the next starts.
        report = read_json(args.out/'cohort.json')
        assert len(report['episodes']) == len(calls)
        assert cmd[cmd.index('--server-dir')+1] == str(args.server_dir)
        assert kwargs['cwd'] == args.eval_cwd
        child_recording(cmd, args, config, policy)
        calls.append(int(cmd[cmd.index('--seed')+1]))
        return SimpleNamespace(returncode=0)

    result = run_cohort(args, runner=runner)
    assert calls == [2, 4, 7]
    assert result['status'] == 'complete'
    assert result['config']['eval_cwd'] == str(args.eval_cwd)
    assert [e['cs']['100'] for e in result['episodes']] == [3, 5, 8]
    assert result['median_cs']['100'] == 5
    assert result['median_deaths']['100'] == 1
    assert 'success' not in result and 'improved' not in result
    assert len({e['directory'] for e in result['episodes']}) == 3
    with pytest.raises(FileExistsError):
        run_cohort(args, runner=runner)
    assert calls == [2, 4, 7]


@pytest.mark.parametrize('failure', ['child', 'nonfinite', 'incomplete', 'freeze',
                                    'checkpoint', 'task', 'contract', 'deaths', 'source'])
def test_rejects_bad_episode_without_launching_later_seeds(cohort, failure):
    args, config, policy = cohort
    calls = []

    def runner(cmd, **kwargs):
        calls.append(int(cmd[cmd.index('--seed')+1]))
        out, result = child_recording(cmd, args, config, policy,
            code=b'changed\n' if failure == 'source' and len(calls) == 2 else b'pass\n')
        if failure == 'child':
            return SimpleNamespace(returncode=1)
        if failure == 'nonfinite': result['wall_s'] = float('nan')
        if failure == 'freeze': result['freeze']['invalid'] = True
        if failure == 'checkpoint': args.checkpoint.write_bytes(b'changed during evaluation')
        if failure == 'task': result['seconds'] = 600.
        if failure == 'contract': result['observation_interface'] = 'old-contract'
        if failure == 'deaths': result['behavior']['champions']['100']['deaths'] = 0
        if failure == 'incomplete':
            path = out/'policy_obs.jsonl'
            path.write_text('\n'.join(path.read_text().splitlines()[:-1])+'\n')
        (out/'evaluation.json').write_text(json.dumps(result))
        return SimpleNamespace(returncode=0)

    with pytest.raises((ValueError, RuntimeError)):
        run_cohort(args, runner=runner)
    report = read_json(args.out/'cohort.json')
    assert report['status'] == 'failed'
    assert 'median_cs' not in report
    assert calls == ([2, 4] if failure == 'source' else [2])


def test_missing_evaluation_checkout_rejected_before_reserving_output(cohort):
    args, _, _ = cohort
    args.eval_cwd = args.eval_cwd/'missing'
    with pytest.raises(ValueError, match='working directory'):
        run_cohort(args)
    assert not args.out.exists()


def existing_recordings(args, config, policy):
    directories = []
    for seed in args.seeds:
        directory = args.out.parent / f'existing-{seed}'
        child_recording(['--out',str(directory),'--seed',str(seed)], args, config, policy)
        directories.append(directory)
    return directories


def test_collect_validates_existing_episodes_without_launching_or_modifying_them(cohort):
    args, config, policy = cohort
    args.collect = existing_recordings(args, config, policy)
    original = [sha256(p/'evaluation.json') for p in args.collect]

    def forbidden(*args, **kwargs):
        raise AssertionError('collect must never launch a child')

    report = run_cohort(args, runner=forbidden)
    assert report['status'] == 'complete'
    assert report['mode'] == 'collect'
    assert [e['seed'] for e in report['episodes']] == args.seeds
    assert report['median_cs']['100'] == 5
    assert report['median_deaths']['100'] == 1
    assert report['config']['eval_cwd'] == str(args.eval_cwd)
    assert original == [sha256(p/'evaluation.json') for p in args.collect]
    assert not list(args.out.glob('seed-*'))


@pytest.mark.parametrize('failure', ['missing', 'wrong_seed', 'source'])
def test_collect_requires_complete_ordered_seed_set_and_same_source(cohort, failure):
    args, config, policy = cohort
    args.collect = existing_recordings(args, config, policy)
    if failure == 'missing':
        args.collect.pop()
    elif failure == 'wrong_seed':
        args.collect[0], args.collect[1] = args.collect[1], args.collect[0]
    else:
        # Preserve valid archive/hash consistency but change runtime content.
        directory = args.collect[1]
        archive = directory/'source.tar.gz'
        with tarfile.open(archive, 'w:gz') as tar:
            code = b'changed runtime\n'
            item = tarfile.TarInfo('lanerl_jax/train/server_eval.py'); item.size = len(code)
            tar.addfile(item, io.BytesIO(code))
        result = read_json(directory/'evaluation.json')
        result['source']['snapshot_sha256'] = sha256(archive)
        (directory/'evaluation.json').write_text(json.dumps(result))

    def forbidden(*args, **kwargs):
        raise AssertionError('collect must never launch a child')

    with pytest.raises(ValueError):
        run_cohort(args, runner=forbidden)
    if args.out.exists():
        report = read_json(args.out/'cohort.json')
        assert report['status'] == 'failed'
        assert 'median_cs' not in report
