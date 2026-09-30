"""TOOL: bounded Slurm completion -> approved current T3 thread wake test.

Launched only through ops/launch.py OPS047_slurm_events. Uses the existing
Jarvis HTTP client without modifying its services, credentials or policy.
Never copies credentials; emits one idempotent message after the turn is idle.
"""
import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path('/mnt/nfs/shared/OPS047_slurm_events')
REGISTRY = Path('/mnt/nfs/shared/jobs/REGISTRY.tsv')
UNIT = 'lanerl-ops047-events'
THREAD = '6638ec1e-0a67-4ea7-bea9-5ecfd4b3d47c'


def client():
    sys.path.insert(0, '/home/dani/codex-workspace/jarvis-control-plane')
    from jarvis_broker.t3_http_client import T3LocalHttpClient
    return T3LocalHttpClient('http://127.0.0.1:3774',
                            Path('/home/dani/.local/state/jarvis/t3-danilogin/access.token'))


def run(args):
    return subprocess.check_output(args, text=True).strip()


def launch(spec, dry_run):
    assert spec['thread_id'] == THREAD
    cmd = ['sbatch', '--parsable', '-p', 'cpu', '--nodelist=danilogin',
           '--cpus-per-task=1', '--mem=128M', '--time=00:07:00', '--array=0-3',
           '--job-name='+spec['id'], '--output=/mnt/nfs/shared/OPS047-%A_%a.out',
           str(ROOT / 'slurm/event_test.sbatch')]
    print(json.dumps({'command': cmd, 'cases': spec['cases'], 'wake_thread': THREAD}), flush=True)
    if dry_run:
        return
    if OUT.joinpath('launch.json').exists():
        raise RuntimeError('New attempt requires a new experiment ID')
    # Read-only endpoint canary, exact identity, and real compiler preflight.
    c = client()
    d = c._thread(c._thread_detail(THREAD))
    assert d['title'] == spec['thread_title']
    subprocess.run(['bash', '-n', str(ROOT/'slurm/event_test.sbatch')], check=True)
    bad = subprocess.run(['gcc', '-x', 'c', '-fsyntax-only', '-'],
                         input='int main( {\n', text=True, capture_output=True)
    assert bad.returncode == 1
    print('CANARY PASSED: exact T3 thread, script syntax, compiler failure exit1', flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    job = run(cmd).split(';')[0]
    assert job.isdigit()
    info = dict(spec, job=job, launched_at=time.time())
    OUT.joinpath('launch.json').write_text(json.dumps(info, indent=2))
    # Existing launcher watch; the deliberate failures happen afterwards.
    from launch import watch_startup
    watch_startup(spec['id'], f'/mnt/nfs/shared/OPS047-{job}_*.out', seconds=180)
    for i in range(4):
        assert 'CANARY PASSED' in Path(f'/mnt/nfs/shared/OPS047-{job}_{i}.out').read_text()
    info['startup_watch'] = 'healthy180s; all four task markers verified'
    OUT.joinpath('launch.json').write_text(json.dumps(info, indent=2))
    with REGISTRY.open('a') as f:
        f.write(f'danilogin\t{UNIT}\tOpenAI via T3 (one turn)\tOPS047 bounded Slurm event test\tsystemctl --user stop {UNIT}.service\n')
    subprocess.run(['systemd-run', '--user', '--unit='+UNIT, '--collect',
                    '-p', 'RuntimeMaxSec=900', '-p', 'MemoryMax=128M',
                    '-p', 'CPUQuota=10%', '-p', 'MemorySwapMax=0',
                    '-p', 'StandardOutput=append:'+str(OUT/'watch.log'),
                    '-p', 'StandardError=append:'+str(OUT/'watch.log'),
                    '/usr/bin/python3', str(Path(__file__).resolve()), 'watch'], check=True)
    print(f'Armed bounded idle-only bridge for Slurm {job}', flush=True)


def watch():
    info = json.loads(OUT.joinpath('launch.json').read_text())
    job = info['job']
    deadline = time.time()+850
    result = {'job': job, 'delivery': 'pending'}
    try:
        while time.time() < deadline:
            rows = run(['sacct', '-n', '-P', '-j', job,
                        '--format=JobID,State,ExitCode,NodeList']).splitlines()
            tasks = {}
            for row in rows:
                cols = row.split('|')
                if len(cols) >= 4 and cols[0] in {f'{job}_{i}' for i in range(4)}:
                    tasks[cols[0]] = {'state': cols[1], 'exit': cols[2], 'node': cols[3]}
            result['tasks'] = tasks
            OUT.joinpath('result.json').write_text(json.dumps(result, indent=2))
            terminal = {'COMPLETED', 'FAILED', 'TIMEOUT', 'CANCELLED', 'NODE_FAIL', 'OUT_OF_MEMORY'}
            if len(tasks) == 4 and all(v['state'].split()[0] in terminal for v in tasks.values()):
                with sqlite3.connect('file:/home/dani/.t3/userdata/state.sqlite?mode=ro', uri=True) as db:
                    row = db.execute('select status,active_turn_id from projection_thread_sessions where thread_id=?', (THREAD,)).fetchone()
                if row and row[0] != 'running' and not row[1]:
                    result['idle_observed_at'] = time.time()
                    result['idle_session_status'] = row[0]
                    result['delivery'] = 'dispatching'
                    OUT.joinpath('result.json').write_text(json.dumps(result, indent=2))
                    message = ('[Authorized Slurm event bridge test OPS047; automated message, not Dani typing.] '
                               'All four CPU test tasks finished and the bridge observed this conversation idle. '
                               f'Results: {json.dumps(tasks)}. '
                               'Please inspect /mnt/nfs/shared/OPS047_slurm_events/result.json and job logs; '
                               'report whether this event woke a fresh turn, record results in the existing ledgers/STATUS, '
                               'and verify the one-shot watcher has stopped. Do not restart the test or modify training.')
                    receipt = client().message_thread(thread_id=THREAD, title=info['thread_title'],
                                                     text=message, idempotency_key='ops047-slurm-'+job)
                    result.update(delivery='accepted', receipt=receipt, sent_at=time.time())
                    return
            time.sleep(10)
        result['delivery'] = 'expired_without_dispatch'
    except Exception as exc:
        # Do not expose credentials, response bodies or transcript text.
        result.update(delivery='error', error_type=type(exc).__name__,
                      error_code=getattr(exc, 'code', None))
    finally:
        OUT.joinpath('result.json').write_text(json.dumps(result, indent=2))
        lines = REGISTRY.read_text().splitlines(True)
        REGISTRY.write_text(''.join(line for line in lines if '\t'+UNIT+'\t' not in line))


if __name__ == '__main__':
    assert sys.argv[1:] == ['watch']
    watch()
