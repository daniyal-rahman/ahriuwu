#!/usr/bin/env python3
"""
Aggregate pipeline + sync + disk + NFS state into a single tail-able logfile.

Polls every --interval seconds (default 30) and appends a snapshot to --out.
Tracks state across polls so trends / stalls / drain rates surface as their
own [!] alerts, not just at-the-moment values.

Run on any machine with ssh access to the recording host (and optionally the
NFS host). Use:

    python scripts/aggregation/monitor.py \\
        --recording-host windows \\
        --output-dir 'C:\\tmp\\replay_data' \\
        --pipeline-log 'C:\\tmp\\garen221.log' \\
        --sync-log    'C:\\tmp\\garen221_sync.log' \\
        --nfs-host    danilogin \\
        --nfs-dataset /mnt/nfs/datasets/lol_replays_16_9_772 \\
        --out ~/garen_monitor.log

Then:  tail -f ~/garen_monitor.log

Drop --nfs-host / --nfs-dataset and the NFS-side metric is silently omitted
(works fine for users with no off-host data sink).

Alert taxonomy:
  [!] DISK CRITICAL/WARN     — Windows C: drive low
  [!] PIPELINE STALLED       — log mtime hasn't advanced in --stall-seconds
  [!] PIPELINE NOT RUNNING   — no python.exe on recording host
  [!] LEAGUE MISSING         — pipeline alive but no League.exe (mid-launch crash?)
  [!] BACKLOG GROWING        — sync queue grew for 3+ consecutive polls
  [!] NO PROGRESS            — NFS game count hasn't advanced in --progress-window
  [!] HOST UNREACHABLE       — ssh failure to recording host or NFS host
  [!] <log line>             — any WARN/ERROR/FAIL/skip/ALARM/WATCHDOG
                               line tailed from pipeline log
"""
import argparse
import datetime as dt
import os
import re
import subprocess
import sys
import time
from collections import deque

ALERT_PATTERN = re.compile(
    r"(WARN|ERROR|FAIL|FATAL|ALARM|WATCHDOG|\*\*\*|^FAIL )", re.IGNORECASE
)


def _ssh(host, cmd, timeout=15):
    """Returns (rc, stdout, stderr). Empty on rc!=0 or transport failure."""
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
    return f'powershell -Command "Get-Content {path} -Tail {n} -ErrorAction SilentlyContinue"'


def _ps_size_mb(path):
    return (
        f'powershell -Command "(Get-ChildItem {path} -Recurse -File -ErrorAction SilentlyContinue '
        f'| Measure-Object -Property Length -Sum).Sum / 1MB -as [int]"'
    )


def _ps_mtime_epoch(path):
    """Returns unix timestamp of file's LastWriteTime, or empty on miss."""
    return (
        f'powershell -Command "(Get-Item {path} -ErrorAction SilentlyContinue).LastWriteTimeUtc '
        f'| ForEach-Object {{ [int][double]::Parse((Get-Date $_ -UFormat %s)) }}"'
    )


class Probe:
    """Bundle of all the things we ask each poll. Errors are recorded per-key
    so we can surface 'host unreachable' as a distinct alert instead of every
    metric reading null."""

    def __init__(self, args):
        self.args = args
        self.recording_reachable = True
        self.nfs_reachable = True

    def collect(self):
        a = self.args
        out = {
            "py_procs": None,
            "league_alive": None,
            "pipeline_last": None,
            "sync_last": None,
            "pipeline_mtime": None,
            "disk_gb": None,
            "backlog_mb": None,
            "nfs_count": None,
        }

        # Single batched ssh to the recording host — cheaper than 5 round-trips.
        # PowerShell doesn't have a clean multi-cmd-with-explicit-separators
        # idiom that survives quoting, so we fire them serially. ConnectTimeout
        # protects us from the laptop blocking 5×on a dead host.
        rc, out_str, _ = _ssh(a.recording_host, 'tasklist /FI "IMAGENAME eq python.exe" /NH')
        if rc != 0:
            self.recording_reachable = False
            return out
        self.recording_reachable = True
        out["py_procs"] = sum(1 for l in out_str.splitlines() if "python.exe" in l)

        rc, league_out, _ = _ssh(a.recording_host,
                                  'tasklist /FI "IMAGENAME eq League of Legends.exe" /NH')
        out["league_alive"] = (rc == 0 and "League of Legends.exe" in league_out)

        rc, last_p, _ = _ssh(a.recording_host, _ps_tail(a.pipeline_log, 1))
        out["pipeline_last"] = last_p.strip() if rc == 0 else None

        rc, last_s, _ = _ssh(a.recording_host, _ps_tail(a.sync_log, 1))
        out["sync_last"] = last_s.strip() if rc == 0 else None

        rc, mt, _ = _ssh(a.recording_host, _ps_mtime_epoch(a.pipeline_log))
        if rc == 0 and mt.strip().isdigit():
            out["pipeline_mtime"] = int(mt.strip())

        rc, disk, _ = _ssh(a.recording_host,
            'powershell -Command "[math]::Round((Get-PSDrive C).Free/1GB,1)"')
        try: out["disk_gb"] = float(disk.strip()) if rc == 0 else None
        except ValueError: pass

        rc, mb, _ = _ssh(a.recording_host, _ps_size_mb(a.output_dir))
        try: out["backlog_mb"] = int(mb.strip()) if rc == 0 else None
        except ValueError: pass

        if a.nfs_host and a.nfs_dataset:
            rc, n, _ = _ssh(a.nfs_host, f"ls {a.nfs_dataset}/ 2>/dev/null | grep -c '^NA1_'")
            if rc != 0:
                self.nfs_reachable = False
            else:
                self.nfs_reachable = True
                try: out["nfs_count"] = int(n.strip())
                except ValueError: pass

        return out

    def fetch_alerts(self):
        a = self.args
        rc, body, _ = _ssh(a.recording_host, _ps_tail(a.pipeline_log, 200))
        if rc != 0:
            return []
        return [l.strip() for l in body.splitlines() if l.strip() and ALERT_PATTERN.search(l)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording-host", default="windows",
                    help="ssh alias of the host running pipeline.py (default: windows)")
    ap.add_argument("--output-dir", default=r"C:\tmp\replay_data",
                    help="pipeline REPLAY_OUTPUT path on the recording host")
    ap.add_argument("--pipeline-log", default=r"C:\tmp\garen221.log",
                    help="path to the main pipeline log on the recording host")
    ap.add_argument("--sync-log", default=r"C:\tmp\garen221_sync.log",
                    help="path to the sync watcher log (omit if sync isn't used)")
    ap.add_argument("--nfs-host", default=None,
                    help="ssh alias of NFS host (omit to skip NFS metrics)")
    ap.add_argument("--nfs-dataset", default=None,
                    help="absolute remote path to dataset dir, e.g. /mnt/nfs/datasets/lol_replays_16_9_772")
    ap.add_argument("--out", default=os.path.expanduser("~/garen_monitor.log"),
                    help="aggregated log path (default: ~/garen_monitor.log)")
    ap.add_argument("--interval", type=int, default=30,
                    help="poll interval in seconds (default: 30)")
    ap.add_argument("--stall-seconds", type=int, default=600,
                    help="alert if pipeline-log mtime hasn't advanced this long (default 600 = 10min)")
    ap.add_argument("--progress-window-min", type=int, default=60,
                    help="alert if NFS game count hasn't advanced in this many minutes (default 60)")
    ap.add_argument("--disk-warn-gb", type=float, default=25.0)
    ap.add_argument("--disk-crit-gb", type=float, default=10.0)
    args = ap.parse_args()

    probe = Probe(args)

    print(f"[monitor] writing to {args.out} every {args.interval}s. "
          f"Tail with: tail -f {args.out}", file=sys.stderr, flush=True)

    f = open(args.out, "a", buffering=1)
    f.write(f"\n{'=' * 70}\n[{_ts()}] monitor started — recording_host={args.recording_host} "
            f"nfs={args.nfs_host or '(none)'}\n{'=' * 70}\n")

    last_pipeline_line = None
    last_sync_line = None
    seen_alerts = set()

    # rolling history for trend detection
    backlog_hist = deque(maxlen=4)            # last 4 polls of backlog_mb
    nfs_hist = deque(maxlen=120)              # (ts, count) — long enough for 1hr at 30s
    disk_hist = deque(maxlen=4)               # (ts, gb) — for "is pass2 actually recording?"
    last_active_pipeline_mtime = None
    last_active_pipeline_seen_at = None       # wallclock when mtime last advanced
    backlog_growth_streak = 0                 # consecutive polls of growth

    while True:
        ts_str = _ts()
        now = time.time()
        s = probe.collect()

        # ── reachability alerts (loud, distinct from data gaps) ──
        if not probe.recording_reachable:
            f.write(f"\n[{ts_str}] [!] HOST UNREACHABLE: ssh {args.recording_host} failed\n")
            time.sleep(args.interval)
            continue
        if args.nfs_host and not probe.nfs_reachable:
            f.write(f"  [!] NFS UNREACHABLE: ssh {args.nfs_host} failed\n")

        # ── compute derived state ──
        py_procs = s["py_procs"] or 0
        alive = py_procs > 0
        disk = s["disk_gb"]
        backlog_mb = s["backlog_mb"] or 0
        nfs_n = s["nfs_count"]
        league_alive = s["league_alive"]
        mtime = s["pipeline_mtime"]

        backlog_hist.append(backlog_mb)
        if nfs_n is not None:
            nfs_hist.append((now, nfs_n))
        if disk is not None:
            disk_hist.append((now, disk))

        if mtime is not None and mtime != last_active_pipeline_mtime:
            last_active_pipeline_mtime = mtime
            last_active_pipeline_seen_at = now

        # ── dense state line — always emit so the log is forensic-grade ──
        flag = "ALIVE" if alive else "DOWN!"
        disk_s = f"{disk:.1f}GB" if disk is not None else "?"
        backlog_s = f"{backlog_mb / 1024:.1f}GB"
        nfs_s = str(nfs_n) if nfs_n is not None else ("?" if args.nfs_host else "n/a")
        league_s = "Y" if league_alive else "N"

        f.write(f"\n[{ts_str}] {flag} py={py_procs} league={league_s}  "
                f"disk={disk_s}  backlog={backlog_s}  nfs={nfs_s}\n")

        if s["pipeline_last"] and s["pipeline_last"] != last_pipeline_line:
            f.write(f"  PIPELINE: {s['pipeline_last']}\n")
            last_pipeline_line = s["pipeline_last"]
        if s["sync_last"] and s["sync_last"] != last_sync_line:
            f.write(f"  SYNC:     {s['sync_last']}\n")
            last_sync_line = s["sync_last"]

        # ── alerts from log tail (deduped across polls) ──
        for line in probe.fetch_alerts():
            if line not in seen_alerts:
                seen_alerts.add(line)
                f.write(f"  [!] {line}\n")
        if len(seen_alerts) > 2000:
            seen_alerts = set(list(seen_alerts)[-1000:])

        # ── derived alerts ──
        if not alive:
            f.write(f"  [!] PIPELINE NOT RUNNING — no python.exe on {args.recording_host}\n")

        if alive and league_alive is False:
            f.write(f"  [!] LEAGUE MISSING — pipeline alive but League of Legends.exe not running "
                    f"(mid-launch crash, or between-games gap)\n")

        if (alive and last_active_pipeline_seen_at and
                (now - last_active_pipeline_seen_at) > args.stall_seconds):
            # Cross-check: pass2's recording loop writes 720p PNGs at ~1-2 GB/min
            # to the staging dir (which sits on the same C: volume). If disk is
            # dropping fast, recording is healthy and main log silence is normal
            # (pass2 prints to per-game log only). Suppress the false positive.
            disk_drop_gb_per_min = None
            if len(disk_hist) >= 2:
                t_old, gb_old = disk_hist[0]
                t_new, gb_new = disk_hist[-1]
                dt_min = (t_new - t_old) / 60
                if dt_min > 0:
                    disk_drop_gb_per_min = (gb_old - gb_new) / dt_min
            mins = (now - last_active_pipeline_seen_at) / 60
            if disk_drop_gb_per_min is not None and disk_drop_gb_per_min > 0.5:
                # Active recording — don't alarm. Note it once per stall window.
                pass
            else:
                drop_s = (f"disk_drop={disk_drop_gb_per_min:.2f}GB/min"
                          if disk_drop_gb_per_min is not None else "disk_drop=?")
                f.write(f"  [!] PIPELINE STALLED — pipeline-log mtime hasn't advanced in {mins:.1f} min "
                        f"({drop_s}; if pass2 were recording, disk would drop >0.5GB/min). Likely "
                        f"socket-hang on LCU/replay or a frozen pass2 record.\n")

        # backlog trend: 3+ consecutive polls of growth
        if len(backlog_hist) >= 3:
            recent = list(backlog_hist)[-3:]
            if recent[0] < recent[1] < recent[2] and recent[2] - recent[0] > 200:  # +200MB minimum
                backlog_growth_streak += 1
            else:
                backlog_growth_streak = 0
            if backlog_growth_streak >= 1:
                f.write(f"  [!] BACKLOG GROWING — sync falling behind: "
                        f"{recent[0]/1024:.1f} → {recent[2]/1024:.1f}GB across last 3 polls\n")

        # NFS progress: alert if count hasn't advanced in --progress-window-min
        if nfs_hist and args.nfs_host:
            cutoff = now - args.progress_window_min * 60
            old = [n for t, n in nfs_hist if t <= cutoff]
            if old and old[-1] >= nfs_n:
                gain = nfs_n - old[-1]
                f.write(f"  [!] NO PROGRESS — NFS count hasn't advanced in "
                        f"{args.progress_window_min} min (Δ={gain})\n")

        # disk threshold alerts
        if disk is not None:
            if disk < args.disk_crit_gb:
                f.write(f"  [!] DISK CRITICAL — only {disk:.1f}GB free on C: "
                        f"(< {args.disk_crit_gb}GB threshold)\n")
            elif disk < args.disk_warn_gb:
                f.write(f"  [!] DISK WARN — {disk:.1f}GB free (peak footprint ~35GB; "
                        f"< {args.disk_warn_gb}GB threshold)\n")

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
