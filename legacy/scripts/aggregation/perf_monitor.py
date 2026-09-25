"""
1 Hz resource snapshot during pipeline runs.
Tracks per-core CPU, system RAM, League + Python RSS/CPU, NVIDIA GPU util/mem.
Writes CSV and stops on receiving SIGTERM (or the parent .bat ends it via taskkill).
"""
import argparse
import csv
import os
import subprocess
import sys
import time

import psutil

PID_REFRESH_S = 5.0      # rescan League/python pids this often
DRIFT_WARN_RATIO = 1.5   # iteration > interval * this → log a warning


def gpu_snapshot():
    """nvidia-smi → (util%, mem_used_mb, mem_total_mb). None on AMD/no-GPU/timeout."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=2,
        ).decode().strip().splitlines()
        u, m_used, m_tot = (x.strip() for x in out[0].split(","))
        return int(u), int(m_used), int(m_tot)
    except FileNotFoundError:
        return None  # caller distinguishes via type
    except subprocess.TimeoutExpired:
        return "timeout"
    except Exception as e:
        return f"error:{type(e).__name__}"


def find_pids():
    """Return (league_pid_or_none, [python_pids])."""
    league = None
    py = []
    for p in psutil.process_iter(["name", "pid"]):
        n = (p.info["name"] or "").lower()
        if n == "league of legends.exe":
            league = p.info["pid"]
        elif n == "python.exe":
            py.append(p.info["pid"])
    return league, py


def _register(procs, pid):
    """Add pid to procs dict, priming cpu_percent. Idempotent. No-op if pid already tracked
    or process is gone."""
    if pid is None or pid in procs:
        return
    try:
        p = psutil.Process(pid)
        p.cpu_percent()  # prime — first call always returns 0.0
        procs[pid] = p
    except psutil.NoSuchProcess:
        pass


def main():
    ap = argparse.ArgumentParser(description="1Hz CPU/RAM/GPU sampler")
    ap.add_argument("--out", required=True, help="CSV output path")
    ap.add_argument("--interval", type=float, default=1.0)
    args = ap.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.out))
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as e:
        print(f"FATAL: cannot create output dir {out_dir}: {e}", file=sys.stderr)
        sys.exit(1)

    n_cores = psutil.cpu_count(logical=True)
    vm = psutil.virtual_memory()
    mem_total_mb = round(vm.total / 1024 / 1024, 1)
    gpu = gpu_snapshot()

    if gpu is None:
        gpu_total_mb = None
        gpu_status = "missing (nvidia-smi not found)"
    elif isinstance(gpu, str):
        gpu_total_mb = None
        gpu_status = gpu
    else:
        _, _, gpu_total_mb = gpu
        gpu_status = f"present ({gpu_total_mb} MB total)"

    league_pid, py_pids = find_pids()
    print(f"[perf_monitor] out={args.out} interval={args.interval}s cores={n_cores} "
          f"mem_total={mem_total_mb}MB gpu={gpu_status}", flush=True)
    if league_pid is None:
        print(f"[perf_monitor] WARN: League not running yet — league_* columns will be blank "
              f"until it appears", flush=True)
    else:
        print(f"[perf_monitor] tracking League pid={league_pid}, python pids={py_pids}", flush=True)

    headers = ["t_wall"]
    headers += [f"cpu_c{i}" for i in range(n_cores)]
    headers += ["mem_used_mb",
                "league_pid", "league_cpu", "league_rss_mb",
                "py_pids", "py_cpu_sum", "py_rss_mb_sum",
                "gpu_util", "gpu_mem_used_mb"]

    procs = {}
    psutil.cpu_percent(percpu=True)  # prime per-core counter
    _register(procs, league_pid)
    for pid in py_pids:
        _register(procs, pid)
    time.sleep(0.5)

    drift_warns = 0  # rate-limit drift warnings
    rows_written = 0

    try:
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(headers)
            f.flush()
            t0 = time.time()
            last_pid_refresh = t0

            while True:
                tick = time.time()
                per_core = psutil.cpu_percent(interval=None, percpu=True)
                vm = psutil.virtual_memory()

                if tick - last_pid_refresh >= PID_REFRESH_S:
                    cur_league, cur_py = find_pids()
                    if cur_league != league_pid:
                        if league_pid in procs:
                            procs.pop(league_pid, None)
                        league_pid = cur_league
                        _register(procs, league_pid)
                    for pid in cur_py:
                        _register(procs, pid)
                    py_pids = cur_py
                    last_pid_refresh = tick

                lg_cpu = lg_rss = ""
                lp = procs.get(league_pid)
                if lp is not None:
                    try:
                        lg_cpu = round(lp.cpu_percent(), 1)
                        lg_rss = round(lp.memory_info().rss / 1024 / 1024, 1)
                    except psutil.NoSuchProcess:
                        procs.pop(league_pid, None)
                        league_pid = None

                py_cpu = py_rss = 0.0
                live_py = []
                for pid in py_pids:
                    p = procs.get(pid)
                    if p is None:
                        continue
                    try:
                        py_cpu += p.cpu_percent()
                        py_rss += p.memory_info().rss
                        live_py.append(pid)
                    except psutil.NoSuchProcess:
                        procs.pop(pid, None)
                py_pids = live_py

                gpu = gpu_snapshot()
                if isinstance(gpu, tuple):
                    g_util, g_mem, _ = gpu
                else:
                    g_util = g_mem = ""

                row = [round(tick - t0, 2)]
                row += [round(c, 1) for c in per_core]
                row.append(round(vm.used / 1024 / 1024, 1))
                row += [league_pid or "", lg_cpu, lg_rss]
                row += ["|".join(str(x) for x in py_pids), round(py_cpu, 1),
                        round(py_rss / 1024 / 1024, 1)]
                row += [g_util, g_mem]
                w.writerow(row)
                f.flush()
                rows_written += 1

                elapsed = time.time() - tick
                if elapsed > args.interval * DRIFT_WARN_RATIO and drift_warns < 5:
                    print(f"[perf_monitor] WARN: sample iteration took {elapsed:.2f}s "
                          f"(> {args.interval * DRIFT_WARN_RATIO:.1f}s) — sampling will drift",
                          flush=True)
                    drift_warns += 1
                if elapsed < args.interval:
                    time.sleep(args.interval - elapsed)
    except KeyboardInterrupt:
        pass
    finally:
        print(f"[perf_monitor] stopped at t={time.time() - t0:.1f}s, "
              f"rows_written={rows_written}", flush=True)


if __name__ == "__main__":
    main()
