"""Sequential frozen-policy source evaluations; no training or parallel jobs.

The cohort is one checkpoint across sampling seeds, not independent training
seeds. A completed cohort makes no farming-improvement or project-success claim.
Use ops/login_capped.sh on danilogin. Existing output directories are refused.
--collect validates existing episodes in --seeds order without launching children.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import tarfile
import time


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    def reject(value):
        raise ValueError(f'nonfinite JSON value: {value}')
    return json.loads(Path(path).read_text(), parse_constant=reject)


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def python_snapshot_digest(path):
    """Compare source contents, independent of archive timestamps or docs edits."""
    h = hashlib.sha256()
    count = 0
    with tarfile.open(path, 'r:gz') as archive:
        for member in sorted(archive.getmembers(), key=lambda m: m.name):
            if member.isfile() and member.name.endswith('.py') and 'tests' not in Path(member.name).parts:
                h.update(member.name.encode() + b'\0')
                h.update(hashlib.sha256(archive.extractfile(member).read()).digest())
                count += 1
    if not count:
        raise ValueError('source snapshot contains no Python runtime source')
    return h.hexdigest()


def validate_episode(directory, expected, seed):
    """Require complete recordings and consistent inputs before accepting scores."""
    directory = Path(directory)
    result = read_json(directory / 'evaluation.json')
    if not finite(result):
        raise ValueError('nonfinite evaluation result')
    same = dict(seed=seed, seconds=expected['seconds'], step_ticks=expected['step_ticks'],
        decision_hz=60. / expected['step_ticks'], start_near_wave=expected['start_near_wave'],
        checkpoint_sha256=expected['checkpoint_sha256'], environment='source-server',
        opponent='idle-fountain', observation_interface=expected['policy']['observation_interface'])
    for key, value in same.items():
        if result.get(key) != value:
            raise ValueError(f'evaluation mismatch for {key}: {result.get(key)!r} != {value!r}')
    if Path(result['checkpoint']).resolve() != Path(expected['checkpoint']):
        raise ValueError('evaluation used a different checkpoint path')
    freeze = result.get('freeze', {})
    if freeze.get('invalid') is not False or freeze.get('unchecked') is not False or freeze.get('frozen') != []:
        raise ValueError('freeze validation failed or was unchecked')
    source = result['source']
    for key in ('server_binary_sha256', 'server_config_sha256'):
        if source.get(key) != expected[key]:
            raise ValueError(f'evaluation mismatch for {key}')
    archive = directory / 'source.tar.gz'
    if sha256(archive) != source.get('snapshot_sha256'):
        raise ValueError('source snapshot is missing or changed')
    code_digest = python_snapshot_digest(archive)
    if expected.get('python_source_sha256') not in (None, code_digest):
        raise ValueError('Python source changed between cohort episodes')
    decisions = round(expected['seconds'] * 60. / expected['step_ticks'])
    actions = read_json(directory / 'policy_policy_actions.json')
    if not finite(actions):
        raise ValueError('nonfinite recorded action')
    for key in ('t_ms', 'blue', 'red', 'blue_sim', 'red_sim'):
        if len(actions.get(key, [])) != decisions:
            raise ValueError(f'incomplete action recording: {key}')
    if actions.get('meta', {}).get('seed') != seed or actions['meta'].get('deterministic') is not False:
        raise ValueError('evaluation sampling protocol mismatch')
    times, final = [], None
    previous_dead, wire_deaths = {}, {'100': 0, '200': 0}
    # Parse one wire frame at a time, retaining only timestamps and the final
    # frame. Do not trust requested "seconds" as proof the child finished.
    with (directory / 'policy_obs.jsonl').open() as f:
        for line in f:
            frame = json.loads(line, parse_constant=lambda v: (_ for _ in ()).throw(
                ValueError(f'nonfinite wire value: {v}')))
            if not finite(frame):
                raise ValueError('nonfinite wire observation')
            champs = {str(u['tm']): u for u in frame['u'] if u.get('k') == 'Champion'}
            if set(champs) != {'100', '200'}:
                raise ValueError('wire frame does not contain both champions')
            for team, champ in champs.items():
                if type(champ.get('dead')) is not bool:
                    raise ValueError('recording lacks authoritative champion life state')
                wire_deaths[team] += previous_dead.get(team) is False and champ['dead']
                previous_dead[team] = champ['dead']
            times.append(frame['t'])
            final = frame
    if len(times) != decisions or result['behavior']['frames'] != decisions or freeze.get('frames') != decisions:
        raise ValueError('incomplete wire/behavior/freeze recording')
    if any(b <= a for a, b in zip(times, times[1:])) or times != actions['t_ms']:
        raise ValueError('wire and action times are not aligned and strictly increasing')
    dt = expected['step_ticks'] * 1000. / 60.
    # The float server clock accumulates small drift. Exact decision counts
    # above are mandatory; allow 0.2% clock drift plus two decision intervals.
    if abs(times[-1] - expected['seconds'] * 1000.) > expected['seconds'] * 2. + 2 * dt:
        raise ValueError('recording did not reach the requested game horizon')
    scores, deaths = {}, {}
    for team in ('100', '200'):
        champ = next(u for u in final['u'] if u.get('k') == 'Champion' and u['tm'] == int(team))
        if type(champ.get('dead')) is not bool:
            raise ValueError('recording lacks authoritative champion life state')
        scores[team] = result['cs'][team]
        deaths[team] = result['behavior']['champions'][team]['deaths']
        if any(type(v) is not int or v < 0 for v in (scores[team], deaths[team])):
            raise ValueError('CS/deaths must be nonnegative integer counts')
        if scores[team] != champ['cs']:
            raise ValueError('reported CS disagrees with the final wire frame')
        if deaths[team] != wire_deaths[team]:
            raise ValueError('reported deaths disagree with authoritative wire transitions')
    return dict(seed=seed, directory=str(directory), cs=scores, deaths=deaths,
        first_t_ms=times[0], last_t_ms=times[-1], decisions=decisions,
        evaluation_sha256=sha256(directory / 'evaluation.json'),
        python_source_sha256=code_digest, wall_s=result['wall_s'])


def run_cohort(args, *, runner=None):
    from lanerl_train.paths import default_game_config
    runner = subprocess.run if runner is None else runner
    checkpoint, server = args.checkpoint.resolve(), args.server_dir.resolve()
    eval_cwd = args.eval_cwd.resolve()
    if not eval_cwd.is_dir():
        raise ValueError(f'evaluation working directory does not exist: {eval_cwd}')
    if len(set(args.seeds)) != len(args.seeds) or not args.seeds:
        raise ValueError('seeds must be nonempty and unique')
    collect = getattr(args, 'collect', None)
    if collect is not None:
        collect = [Path(directory).resolve() for directory in collect]
        if len(collect) != len(args.seeds):
            raise ValueError('collect directories must match the full requested seeds in order')
        if len(set(collect)) != len(collect):
            raise ValueError('collect directories must be unique')
        if any(not directory.is_dir() for directory in collect):
            raise ValueError('collect episode directory does not exist')
    if any(s < 0 for s in args.seeds) or not math.isfinite(args.seconds) or args.seconds <= 0 or args.step_ticks < 1:
        raise ValueError('nonnegative seeds, positive finite seconds and positive step-ticks required')
    if args.start_near_wave and args.seconds <= 120:
        raise ValueError('near-wave evaluation must end after 120 seconds')
    if round(args.seconds * 60. / args.step_ticks) < 1:
        raise ValueError('cohort must contain at least one decision per episode')
    policy = read_json(checkpoint.parent / 'manifest.json')['config']['train']['policy']
    if not policy.get('action_interface') or not policy.get('observation_interface'):
        raise ValueError('checkpoint manifest must identify action and observation contracts')
    config_path = default_game_config().resolve()
    expected = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
        policy=policy, seconds=args.seconds, step_ticks=args.step_ticks,
        start_near_wave=args.start_near_wave, server_dir=str(server), eval_cwd=str(eval_cwd),
        server_binary_sha256=sha256(server / 'GameServerLib.dll'),
        server_config=str(config_path), server_config_sha256=sha256(config_path))
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    command = shlex.join([sys.executable, '-m', 'lanerl_jax.train.server_eval_batch', *sys.argv[1:]])
    source = Path(__file__).read_bytes()
    (out / 'launcher.py').write_bytes(source)
    (out / 'command.txt').write_text(command + '\n')
    (out / 'checkpoint-manifest.json').write_bytes((checkpoint.parent / 'manifest.json').read_bytes())
    report = dict(status='running', mode='collect' if collect is not None else 'launch',
        config=expected, sampling_seeds=args.seeds, episodes=[],
        collected_directories=[str(p) for p in collect] if collect is not None else None,
        launcher_sha256=hashlib.sha256(source).hexdigest(), command=command,
        interpretation='One frozen checkpoint across sampling seeds; no matched-control or training-seed improvement claim.')
    started = time.monotonic()

    def write():
        report['wall_s'] = time.monotonic()-started
        (out / 'cohort.json').write_text(json.dumps(report, indent=2, allow_nan=False))

    def check_inputs():
        if sha256(checkpoint) != expected['checkpoint_sha256']:
            raise ValueError('checkpoint changed during cohort; use a pinned file')
        if read_json(checkpoint.parent / 'manifest.json')['config']['train']['policy'] != policy:
            raise ValueError('checkpoint policy contract changed during cohort')
        if sha256(server / 'GameServerLib.dll') != expected['server_binary_sha256'] or sha256(config_path) != expected['server_config_sha256']:
            raise ValueError('server binary/config changed during cohort')

    write()
    try:
        for index, seed in enumerate(args.seeds):
            report['active_seed'] = seed
            check_inputs()
            write()
            if collect is not None:
                child_out = collect[index]
            else:
                child_out = out / f'seed-{seed:04d}'
                cmd = [sys.executable, '-m', 'lanerl_jax.train.server_eval', str(checkpoint),
                    '--out', str(child_out), '--seed', str(seed), '--seconds', str(args.seconds),
                    '--step-ticks', str(args.step_ticks), '--port-base', str(args.port_base),
                    '--server-dir', str(server)]
                if args.start_near_wave:
                    cmd.append('--start-near-wave')
                (out / f'seed-{seed:04d}.command.txt').write_text(shlex.join(cmd)+'\n')
                with (out / f'seed-{seed:04d}.log').open('w') as log:
                    completed = runner(cmd, cwd=eval_cwd, stdout=log, stderr=subprocess.STDOUT, check=False)
                if completed.returncode != 0:
                    raise RuntimeError(f'seed {seed} child exited {completed.returncode}')
            check_inputs()
            episode = validate_episode(child_out, expected, seed)
            check_inputs()
            expected['python_source_sha256'] = episode['python_source_sha256']
            report['episodes'].append(episode)
            write()
        if [episode['seed'] for episode in report['episodes']] != list(args.seeds):
            raise ValueError('validated episodes do not cover the full requested seed set')
        report['status'] = 'complete'
        report.pop('active_seed', None)
        report['median_cs'] = {team: statistics.median(e['cs'][team] for e in report['episodes'])
                               for team in ('100', '200')}
        report['median_deaths'] = {team: statistics.median(e['deaths'][team] for e in report['episodes'])
                                   for team in ('100', '200')}
    except BaseException as exc:
        report.update(status='failed', error=repr(exc))
        raise
    finally:
        write()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--server-dir', type=Path, required=True)
    parser.add_argument('--eval-cwd', type=Path, default=Path.cwd(),
                        help='Checkout in which to run the unchanged server_eval module')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(5)))
    parser.add_argument('--collect', type=Path, nargs='+',
                        help='Existing episode directories paired with --seeds order; validate only, never launch evaluations')
    parser.add_argument('--seconds', type=float, default=600.)
    parser.add_argument('--step-ticks', type=int, default=2)
    parser.add_argument('--start-near-wave', action='store_true')
    parser.add_argument('--port-base', type=int, default=49500)
    report = run_cohort(parser.parse_args())
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
