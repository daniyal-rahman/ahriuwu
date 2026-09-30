"""TOOL: one bounded, idempotent completion notification for an owned Slurm job.

Arm after the launcher's health watch; no training launch or automatic restart.
Uses the existing local T3 client and discovers this conversation from Codex's
session identity. Secrets stay in the existing credential file.
"""
import argparse
import fcntl
import getpass
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/mnt/nfs/shared/slurm-events')
REGISTRY = Path('/mnt/nfs/shared/jobs/REGISTRY.tsv')
DB = 'file:/home/dani/.t3/userdata/state.sqlite?mode=ro'
TERMINAL = {'COMPLETED', 'FAILED', 'TIMEOUT', 'CANCELLED', 'NODE_FAIL',
            'OUT_OF_MEMORY', 'BOOT_FAIL', 'DEADLINE', 'PREEMPTED', 'REVOKED'}


def client():
    sys.path.insert(0, '/home/dani/codex-workspace/jarvis-control-plane')
    from jarvis_broker.t3_http_client import T3LocalHttpClient
    return T3LocalHttpClient('http://127.0.0.1:3774',
                            Path('/home/dani/.local/state/jarvis/t3-danilogin/access.token'))


def account(job):
    data = subprocess.check_output(['sacct', '-n', '-P', '-j', job,
        '--format=JobID,State,ExitCode,User,JobName%100'], text=True, timeout=20)
    for line in data.splitlines():
        cols = line.split('|')
        if cols[0] == job and len(cols) >= 5:
            return dict(zip(('job', 'state', 'exit', 'user', 'name'), cols))
    return None


def is_terminal(row):
    return bool(row and row['state'].split()[0].rstrip('+') in TERMINAL)


def idle(thread):
    with sqlite3.connect(DB, uri=True, timeout=10) as db:
        row = db.execute('select status,active_turn_id from projection_thread_sessions where thread_id=?', (thread,)).fetchone()
    return bool(row and row[0] in ('ready', 'stopped') and not row[1])


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def register(unit, add):
    with REGISTRY.open('r+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        lines = [l for l in f.readlines() if '\t'+unit+'\t' not in l]
        if add:
            lines.append(f'danilogin\t{unit}\tOpenAI via T3 (one turn)\tBounded Slurm completion bridge\tsystemctl --user stop {unit}.service\n')
        f.seek(0); f.writelines(lines); f.truncate()


def arm(a):
    row = account(a.job)
    if not row or row['user'] != getpass.getuser() or row['name'] != a.experiment:
        raise SystemExit('Refused: job ownership/name does not match')
    session = os.environ.get('CODEX_THREAD_ID')
    if not session:
        raise SystemExit('Arm from the intended Codex conversation')
    with sqlite3.connect(DB, uri=True) as db:
        rows = db.execute('select thread_id from provider_session_runtime where resume_cursor_json like ? or runtime_payload_json like ?',
                          ('%'+session+'%', '%'+session+'%')).fetchall()
    if len(rows) != 1:
        raise SystemExit('Could not uniquely identify current T3 thread')
    thread = rows[0][0]
    c = client()
    detail = c._thread(c._thread_detail(thread))
    unit = 'lanerl-event-'+a.job
    out = BASE/a.job
    out.mkdir(parents=True, exist_ok=True)
    config = out/'config.json'
    if config.exists():
        raise SystemExit(f'Already armed/attempted: inspect {out}; do not duplicate')
    info = dict(job=a.job, experiment=a.experiment, thread=thread, title=detail['title'],
                unit=unit, deadline=time.time()+a.max_hours*3600,
                command_id='lanerl-slurm-terminal-'+a.job+'-'+thread)
    save(config, info)
    register(unit, True)
    try:
        subprocess.run(['systemd-run', '--user', '--collect', '--unit='+unit,
            '-p', f'RuntimeMaxSec={int(a.max_hours*3600)+60}',
            '-p', 'MemoryMax=128M', '-p', 'MemorySwapMax=0', '-p', 'CPUQuota=10%',
            '-p', 'StandardOutput=append:'+str(out/'watch.log'),
            '-p', 'StandardError=append:'+str(out/'watch.log'),
            '/usr/bin/python3', str(Path(__file__).resolve()), 'watch', str(config)], check=True)
    except BaseException:
        register(unit, False)
        raise
    print(f'Armed {unit}: {row["state"]}; thread {thread}; expires in {a.max_hours}h')


def watch(config):
    info = json.loads(config.read_text())
    path = config.parent/'result.json'
    result = dict(job=info['job'], experiment=info['experiment'], delivery='pending')
    def stop(signum, frame):
        raise SystemExit('watcher stopped')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while time.time() < info['deadline']:
            try:
                row = account(info['job'])
                result.update(accounting=row, checked_at=time.time())
                if is_terminal(row) and idle(info['thread']):
                    # Freeze the event text across any uncertain-response retries.
                    if 'message' not in result:
                        result['message'] = (
                            '[Authorized Slurm event bridge; automated message, not Dani typing.] '
                            f'{info["experiment"]} job {info["job"]} reached {row["state"]}, exit {row["exit"]}. '
                            f'Inspect {path} and the run logs/checkpoints/frozen evaluations. '
                            'Report completion or diagnose failure, update STATUS and the existing experiment/fidelity ledgers, '
                            'and verify watcher cleanup. Continue only previously authorized work; this event does not authorize a new training experiment.')
                    result.update(delivery='dispatching', idle_observed_at=time.time())
                    save(path, result)
                    receipt = client().message_thread(thread_id=info['thread'], title=info['title'],
                        text=result['message'], idempotency_key=info['command_id'])
                    result.update(delivery='accepted', receipt=receipt, sent_at=time.time())
                    return
                save(path, result)
            except Exception as exc:
                result.update(last_error_type=type(exc).__name__, last_error_at=time.time())
                save(path, result)
            time.sleep(30)
        result['delivery'] = 'expired_without_dispatch'
    finally:
        if result['delivery'] in ('pending', 'dispatching'):
            result['delivery'] = 'stopped_unconfirmed'
        save(path, result)
        register(info['unit'], False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('arm')
    a.add_argument('--job', required=True)
    a.add_argument('--experiment', required=True)
    a.add_argument('--max-hours', type=float, required=True)
    w = sub.add_parser('watch'); w.add_argument('config', type=Path)
    args = p.parse_args()
    if args.command == 'arm':
        if not args.job.isdigit() or not 0 < args.max_hours <= 168:
            p.error('Use a numeric single-job ID and 0 < max-hours <= 168')
        arm(args)
    else:
        watch(args.config)
