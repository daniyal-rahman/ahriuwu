"""
1 Hz resource snapshot during pipeline runs.
Tracks per-core CPU, system RAM, League + Python RSS/CPU, NVIDIA GPU util/mem.
Writes CSV and stops on receiving SIGTERM (or the parent .bat ends it via taskkill).
"""
import argparse, csv, os, subprocess, sys, time
import psutil


def gpu_snapshot():
    """nvidia-smi → (util%, mem_used_mb, mem_total_mb). None on AMD/no-GPU."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=2,
        ).decode().strip().splitlines()
        # one line per GPU; aggregate the first
        u, m_used, m_tot = (x.strip() for x in out[0].split(","))
        return int(u), int(m_used), int(m_tot)
    except Exception:
        return None, None, None


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


def proc_safe(pid):
    try: return psutil.Process(pid)
    except psutil.NoSuchProcess: return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="CSV output path")
    ap.add_argument("--interval", type=float, default=1.0)
    args = ap.parse_args()

    n_cores = psutil.cpu_count(logical=True)
    headers = ["t_wall", "cpu_total"]
    headers += [f"cpu_c{i}" for i in range(n_cores)]
    headers += ["mem_used_mb", "mem_total_mb",
                "league_pid", "league_cpu", "league_rss_mb",
                "py_pids", "py_cpu_sum", "py_rss_mb_sum",
                "gpu_util", "gpu_mem_used_mb", "gpu_mem_total_mb"]

    # warm up cpu_percent counters
    psutil.cpu_percent(percpu=True)
    league_pid, py_pids = find_pids()
    procs = {}
    for pid in [league_pid] + py_pids:
        if pid is None: continue
        p = proc_safe(pid)
        if p is not None:
            try: p.cpu_percent(); procs[pid] = p
            except psutil.NoSuchProcess: pass
    time.sleep(0.5)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(headers); f.flush()
        t0 = time.time()
        try:
            while True:
                tick = time.time()
                # per-core
                per_core = psutil.cpu_percent(interval=None, percpu=True)
                cpu_total = sum(per_core) / max(1, len(per_core))
                vm = psutil.virtual_memory()
                # refresh league/python pids occasionally (process can exit/restart
                # — League restarts between pass1 and pass2, and again per game in
                # batch mode, so the cached Process object can become stale).
                if int(tick - t0) % 5 == 0:
                    cur_league, cur_py = find_pids()
                    if cur_league != league_pid:
                        # League restarted (or just appeared) — drop the old
                        # Process object, register a fresh one for the new pid.
                        if league_pid in procs:
                            procs.pop(league_pid, None)
                        league_pid = cur_league
                        if league_pid is not None and league_pid not in procs:
                            p = proc_safe(league_pid)
                            if p is not None:
                                try: p.cpu_percent(); procs[league_pid] = p
                                except psutil.NoSuchProcess: pass
                    for pid in cur_py:
                        if pid not in procs:
                            p = proc_safe(pid)
                            if p:
                                try: p.cpu_percent(); procs[pid] = p
                                except psutil.NoSuchProcess: pass
                    py_pids = cur_py
                # league
                lg_cpu = lg_rss = ""
                lp = procs.get(league_pid)
                if lp is not None:
                    try:
                        lg_cpu = lp.cpu_percent()
                        lg_rss = round(lp.memory_info().rss / 1024 / 1024, 1)
                    except psutil.NoSuchProcess:
                        procs.pop(league_pid, None); league_pid = None
                # python sum
                py_cpu = py_rss = 0.0
                live_py = []
                for pid in py_pids:
                    p = procs.get(pid)
                    if p is None: continue
                    try:
                        py_cpu += p.cpu_percent()
                        py_rss += p.memory_info().rss
                        live_py.append(pid)
                    except psutil.NoSuchProcess:
                        procs.pop(pid, None)
                py_pids = live_py
                # gpu
                gu, gm, gt_ = gpu_snapshot()
                row = [round(tick - t0, 2), round(cpu_total, 1)]
                row += [round(c, 1) for c in per_core]
                row += [round(vm.used / 1024 / 1024, 1), round(vm.total / 1024 / 1024, 1)]
                row += [league_pid or "", lg_cpu, lg_rss]
                row += ["|".join(str(x) for x in py_pids), round(py_cpu, 1),
                        round(py_rss / 1024 / 1024, 1)]
                row += [gu if gu is not None else "",
                        gm if gm is not None else "",
                        gt_ if gt_ is not None else ""]
                w.writerow(row); f.flush()
                # pace
                elapsed = time.time() - tick
                if elapsed < args.interval:
                    time.sleep(args.interval - elapsed)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
