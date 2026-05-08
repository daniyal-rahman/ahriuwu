#!/usr/bin/env python3
"""
Aggregate pipeline + sync + disk + NFS state into a single tail-able logfile.

Polls every --interval seconds (default 30) and appends a snapshot to --out.
Also incrementally tails the Windows-side garen221.log and surfaces any
WARN / ERROR / FAIL / skip / ALARM / WATCHDOG lines into the same file
immediately so they don't get lost.

Run from the laptop:
    python scripts/aggregation/monitor.py --out ~/garen_monitor.log
    tail -f ~/garen_monitor.log

Uses ssh aliases `windows` (the recording host) and `danilogin` (NFS).
"""
import argparse
import datetime as dt
import os
import re
import subprocess
import sys
import time

WIN_PIPELINE_LOG = r"C:\tmp\garen221.log"
WIN_SYNC_LOG = r"C:\tmp\garen221_sync.log"
WIN_REPLAY_DATA = r"C:\tmp\replay_data"
NFS_DATASET = "/mnt/nfs/datasets/lol_replays_16_9_772"

ALERT_PATTERN = re.compile(
    r"(WARN|ERROR|FAIL|FATAL|ALARM|WATCHDOG|\*\*\*|^FAIL )", re.IGNORECASE
)


def _ssh(host, cmd, timeout=15):
    """Returns (rc, stdout, stderr). Empty stdout on rc!=0."""
    try:
        p = subprocess.run(["ssh", "-o", "ConnectTimeout=10", host, cmd],
                           capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "ssh timeout"
    except Exception as e:
        return 1, "", str(e)


def _ts():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _ps_tail(path, n=1):
    """PowerShell-quoted Get-Content tail."""
    return f'powershell -Command "Get-Content {path} -Tail {n} -ErrorAction SilentlyContinue"'


def _ps_size_in_dir(path):
    """Sum sizes of all files under path on Windows; emit as MB int."""
    return (
        f'powershell -Command "(Get-ChildItem {path} -Recurse -File -ErrorAction SilentlyContinue '
        f'| Measure-Object -Property Length -Sum).Sum / 1MB -as [int]"'
    )


def collect_pipeline_state():
    """Return (alive, last_line) for the pipeline process + log."""
    rc, out, _ = _ssh("windows", 'tasklist /FI "IMAGENAME eq python.exe" /NH')
    procs = [l for l in out.splitlines() if "python.exe" in l]
    alive = len(procs) > 0

    rc, last, _ = _ssh("windows", _ps_tail(WIN_PIPELINE_LOG, 1))
    last_line = last.strip() if rc == 0 else "(log unreachable)"
    return alive, len(procs), last_line


def collect_sync_state():
    rc, last, _ = _ssh("windows", _ps_tail(WIN_SYNC_LOG, 1))
    return last.strip() if rc == 0 else "(sync log unreachable)"


def collect_disk_free_gb():
    rc, out, _ = _ssh("windows",
        'powershell -Command "[math]::Round((Get-PSDrive C).Free/1GB,1)"')
    if rc == 0 and out.strip():
        try:
            return float(out.strip())
        except ValueError:
            return None
    return None


def collect_local_backlog_mb():
    """Total MB of replay_data on Windows (= staged backlog awaiting sync)."""
    rc, out, _ = _ssh("windows", _ps_size_in_dir(WIN_REPLAY_DATA))
    if rc == 0 and out.strip():
        try:
            return int(out.strip())
        except ValueError:
            return None
    return None


def collect_nfs_count():
    rc, out, _ = _ssh("danilogin", f"ls {NFS_DATASET}/ 2>/dev/null | grep -c '^NA1_' ")
    if rc == 0 and out.strip():
        try:
            return int(out.strip())
        except ValueError:
            return None
    return None


def fetch_new_alerts(since_byte):
    """Tail garen221.log past `since_byte` and return (new_alert_lines, new_byte_count).
    Uses awk-equivalent on Windows to seek; cheaper than re-reading the whole file."""
    rc, total_out, _ = _ssh("windows",
        f'powershell -Command "(Get-Item {WIN_PIPELINE_LOG} -ErrorAction SilentlyContinue).Length"')
    if rc != 0 or not total_out.strip():
        return [], since_byte
    try:
        total = int(total_out.strip())
    except ValueError:
        return [], since_byte
    if total <= since_byte:
        return [], total

    # Pull only the tail since last poll. PowerShell -Tail counts lines; we
    # over-fetch by reading the last 200 lines and filtering. Cheap, robust.
    rc, body, _ = _ssh("windows", _ps_tail(WIN_PIPELINE_LOG, 200))
    if rc != 0:
        return [], total
    alerts = [l.strip() for l in body.splitlines() if ALERT_PATTERN.search(l)]
    return alerts, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/garen_monitor.log"),
                    help="aggregated log path (default: ~/garen_monitor.log)")
    ap.add_argument("--interval", type=int, default=30,
                    help="poll interval in seconds (default: 30)")
    args = ap.parse_args()

    print(f"[monitor] writing to {args.out} every {args.interval}s. Tail with: tail -f {args.out}",
          file=sys.stderr, flush=True)

    last_pipeline_line = None
    last_sync_line = None
    last_byte = 0
    seen_alerts = set()  # dedupe alerts across polls (we re-tail same window)

    f = open(args.out, "a", buffering=1)  # line-buffered
    f.write(f"\n{'=' * 70}\n[{_ts()}] monitor started (poll={args.interval}s)\n{'=' * 70}\n")

    while True:
        ts = _ts()
        alive, n_procs, pipeline_line = collect_pipeline_state()
        sync_line = collect_sync_state()
        disk_gb = collect_disk_free_gb()
        backlog_mb = collect_local_backlog_mb()
        nfs_n = collect_nfs_count()
        new_alerts, last_byte = fetch_new_alerts(last_byte)

        # State block — always emit so timestamps are dense for forensics.
        flag = "ALIVE" if alive else "DOWN!"
        disk_s = f"{disk_gb:.1f}GB" if disk_gb is not None else "?"
        backlog_s = f"{backlog_mb / 1024:.1f}GB" if backlog_mb else "0GB"
        nfs_s = str(nfs_n) if nfs_n is not None else "?"

        f.write(f"\n[{ts}] {flag}  py_procs={n_procs}  disk_free={disk_s}  "
                f"local_backlog={backlog_s}  nfs_done={nfs_s}\n")

        if pipeline_line != last_pipeline_line:
            f.write(f"  PIPELINE: {pipeline_line}\n")
            last_pipeline_line = pipeline_line
        if sync_line != last_sync_line:
            f.write(f"  SYNC:     {sync_line}\n")
            last_sync_line = sync_line

        # Surface only new (not-yet-seen) alerts. Hash by content to dedupe.
        for line in new_alerts:
            if line in seen_alerts:
                continue
            seen_alerts.add(line)
            f.write(f"  [!] {line}\n")
        # Bound seen set so it doesn't grow forever — keep most recent 500
        if len(seen_alerts) > 1000:
            # crude prune: keep half. For the time horizon we care about
            # (one batch run = maybe 200 alerts), this is plenty.
            seen_alerts = set(list(seen_alerts)[-500:])

        # Fast soft-fail signals worth their own line
        if not alive:
            f.write(f"  [!] PIPELINE NOT RUNNING — no python.exe on Windows\n")
        if disk_gb is not None and disk_gb < 10:
            f.write(f"  [!] DISK CRITICAL: only {disk_gb:.1f}GB free on C:\n")
        elif disk_gb is not None and disk_gb < 25:
            f.write(f"  [!] DISK WARN: {disk_gb:.1f}GB free (peak footprint ~35GB)\n")

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
