#!/usr/bin/env python3
"""TOOL: one-shot E24 -> E25/E26/E27 continuation, only through ops/launch.py.

No parameter search or checkpoint selection. Registered as the user systemd
unit e24-comparison; removes its registry entry on exit. Writes progress into
STATUS.md and EXPERIMENTS.md, committing only those files after each stage.
"""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = Path('/mnt/nfs/shared/jobs/REGISTRY.tsv')
UNIT = 'e24-comparison'
AUTHOR = 'Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>'
EVALS = ('E25_dagger_teacher_baseline', 'E26_heuristic_teacher_baseline', 'E27_reference_teacher_final')


def command(args, **kwargs):
    print('+', ' '.join(map(str, args)), flush=True)
    return subprocess.run(list(map(str, args)), cwd=ROOT, check=True, **kwargs)


def unique_run(directory):
    paths = list(directory.glob('server-farm-*/manifest.json'))
    if len(paths) != 1:
        raise RuntimeError(f'expected one run in {directory}, found {len(paths)}')
    return paths[0]


def read_manifest(path):
    # RunDir writes manifests while the job runs; tolerate a concurrent write.
    for _ in range(5):
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            time.sleep(.2)
    raise RuntimeError(f'unreadable manifest: {path}')


def wait_complete(manifest, deadline):
    job_id = None
    while time.monotonic() < deadline:
        data = read_manifest(manifest)
        state = data.get('results', {}).get('status')
        if state == 'complete':
            job_id = data.get('host', {}).get('slurm_job')
            if job_id:
                active = subprocess.run(['squeue', '-h', '-j', str(job_id)], capture_output=True, text=True)
                if active.returncode == 0 and active.stdout.strip():
                    time.sleep(15)
                    continue
            return data
        if state in ('failed', 'diverged', 'interrupted'):
            raise RuntimeError(f'{manifest}: {state}: {data.get("results")}')
        job_id = data.get('host', {}).get('slurm_job')
        if job_id:
            active = subprocess.run(['squeue', '-h', '-j', str(job_id)], capture_output=True, text=True)
            if active.returncode == 0 and not active.stdout.strip():
                # The final manifest may precede the scheduler's state update.
                time.sleep(2)
                data = read_manifest(manifest)
                if data.get('results', {}).get('status') == 'complete':
                    return data
                raise RuntimeError(f'job {job_id} exited without a complete manifest: {manifest}')
        time.sleep(15)
    raise TimeoutError(f'comparison deadline reached while waiting for {manifest}; last job {job_id}')


def record(detail, states):
    """Do not absorb another worker's uncommitted edits into our status commit."""
    paths = ['STATUS.md', 'docs/EXPERIMENTS.md']
    for _ in range(40):
        dirty = subprocess.run(['git', 'status', '--porcelain', '--', *paths],
                               cwd=ROOT, capture_output=True, text=True, check=True).stdout
        if not dirty.strip():
            break
        time.sleep(3)
    else:
        raise RuntimeError('status files have another worker\'s uncommitted edits; refusing to overwrite')
    status = ROOT / paths[0]
    body = status.read_text()
    block = '<!-- E24_PIPELINE -->\n' + detail + '\n<!-- /E24_PIPELINE -->'
    if '<!-- E24_PIPELINE -->' in body:
        body = re.sub(r'<!-- E24_PIPELINE -->.*?<!-- /E24_PIPELINE -->', lambda _: block, body, flags=re.S)
    else:
        anchor = '## BC/DAgger → reference PPO improvement test (E24–E27)\n'
        body = body.replace(anchor, anchor + '\n' + block + '\n')
    status.write_text(body)
    ledger = ROOT / paths[1]
    lines = ledger.read_text().splitlines()
    for i, line in enumerate(lines):
        for experiment, state in states.items():
            if line.startswith(f'| `{experiment}/'):
                parts = line.split('|')
                parts[7] = ' ' + state.replace('|', '/') + ' '
                lines[i] = '|'.join(parts)
    ledger.write_text('\n'.join(lines) + '\n')
    command(['git', 'diff', '--check', '--', *paths])
    command(['git', 'commit', '--only', *paths, '-m', 'Record E24 comparison progress', '-m', detail, '-m', AUTHOR])
    # A push race need not kill a healthy run; the local progress commit remains.
    pushed = subprocess.run(['git', 'push', 'origin', 'lane-rl/jax'], cwd=ROOT)
    if pushed.returncode:
        print('Progress committed locally; push needs reconciliation.', flush=True)


def run_comparison(training_run):
    deadline = time.monotonic() + 4 * 3600
    states = {}
    manifest = training_run / 'manifest.json'
    data = wait_complete(manifest, deadline)
    checkpoint = training_run / 'ckpt_000409600.msgpack'
    matches = [c for c in data['checkpoints'] if c['file'] == checkpoint.name and c['update'] == 400]
    if not checkpoint.is_file() or not matches:
        raise RuntimeError('E24 did not produce the predeclared u400 checkpoint')
    states['E24_dagger_reference_relative'] = 'COMPLETE job 1717 at u400; frozen comparisons in progress; no improvement verdict yet'
    record('E24 completed u400. One-shot comparison is starting matched frozen baselines and the final policy; no improvement verdict yet.', states)
    for experiment in EVALS:
        completed_seeds = []
        for seed in (2, 3):
            out = ROOT / 'lanerl_jax/runs' / experiment / f'seed{seed}'
            if out.exists():
                raise RuntimeError(f'refusing a duplicate evaluation directory: {out}')
            args = [sys.executable, '-u', ROOT/'ops/launch.py', experiment, '--seed', str(seed)]
            if experiment.startswith('E27'):
                args += ['--resume', checkpoint]
            command([*args, '--dry-run'])
            states[experiment] = f'LAUNCHING seed {seed}; completed seeds {completed_seeds}; canary/startup watch required'
            record(f'{experiment} seed {seed}: launching through ops/launch.py with canary and startup watch. E24 final is fixed at u400.', states)
            command(args)
            current = unique_run(out)
            current_data = read_manifest(current)
            job = current_data.get('host', {}).get('slurm_job')
            states[experiment] = f'CANARY/WATCH PASSED seed {seed}, job {job}; completed seeds {completed_seeds}'
            record(f'{experiment} seed {seed}, job {job}: canary and startup watch passed. Waiting for all frozen episodes.', states)
            wait_complete(current, deadline)
            completed_seeds.append(seed)
            states[experiment] = f'COMPLETE seeds {completed_seeds}; frozen episodes recorded; paired verdict pending'
            record(f'{experiment} seed {seed} completed; frozen results preserved for the paired comparison.', states)
    result = command([sys.executable, ROOT/'ops/heuristic_improvement.py'], capture_output=True, text=True)
    print(result.stdout, flush=True)
    score = json.loads(result.stdout)
    verdict = score['improved_beyond_teacher_and_clone']
    details = json.dumps(score['comparisons'], sort_keys=True)
    states['E27_reference_teacher_final'] = f'COMPLETE seeds 2/3; predeclared improvement gate {"PASS" if verdict else "FAIL"}; see STATUS paired intervals'
    record(f'E24 paired frozen comparison gate: {"PASS" if verdict else "FAIL"}. '
           f'One training seed; preliminary evidence only. Differences vs baselines (mean, 95% paired bootstrap CI): {details}. '
           'All six frozen evaluations completed. One-shot continuation finished; no recurring jobs remain.', states)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-run', type=Path, required=True)
    args = p.parse_args()
    os.chdir(ROOT)
    def interrupted(signum, frame):
        raise RuntimeError(f'continuation interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        run_comparison(args.training_run.resolve())
    except BaseException as exc:
        print(f'COMPARISON STOPPED: {exc}', flush=True)
        try:
            record(f'E24 continuation STOPPED: {exc}. No new improvement claim. Inspect /mnt/nfs/shared/jobs/E24-comparison.out before restarting.', {})
        except Exception as status_exc:
            print(f'Could not record stop in STATUS: {status_exc}', flush=True)
        raise
    finally:
        if REGISTRY.exists():
            lines = REGISTRY.read_text().splitlines()
            REGISTRY.write_text('\n'.join(line for line in lines if f'\t{UNIT}\t' not in line) + '\n')


if __name__ == '__main__':
    main()
