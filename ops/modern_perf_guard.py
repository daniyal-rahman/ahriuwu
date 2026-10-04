"""Throughput / memory regression guard for the modern world tick.

TOOL. Runs the ``ops.modern_world_bench`` scan (same world, scripted orders)
for one fixed config per backend, measures steady env-ticks per second (best of
``--repeats`` timed calls), the compiled program's temp buffer size
(``memory_analysis``, any backend) and peak device memory
(``jax.devices()[0].memory_stats()['peak_bytes_in_use']``, GPU only), and
compares them to ``ops/modern_perf_baseline.json`` (keyed by backend, device
kind and config). Exit status: 0 ok, 1 regression (throughput more than
``--tolerance`` below baseline, or either memory figure more than
``--tolerance`` above it), 3 no baseline for this key (record one with
``--update-baseline``).

    python -m ops.modern_perf_guard [--envs N] [--ticks 150] [--warm-ticks 1800] [--update-baseline]

Defaults: 256 envs on GPU, 16 on CPU. Prints one JSON line with the
measurement (also overflow and peak valid packets per tick, informational),
the baseline entry and the verdict.
"""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

import jax

BASELINE = Path(__file__).with_name("modern_perf_baseline.json")
DEFAULT_ENVS = {"gpu": 256, "cpu": 16}


def config_key(args, device) -> str:
    from ops.modern_world_bench import world_label
    w = world_label(args)
    return (f"{jax.default_backend()}|{device.device_kind}|envs={args.envs}|ticks={args.ticks}"
            f"|warm={args.warm_ticks}|fog={w['fog']}|jungle={int(w['jungle'])}"
            f"|objectives={int(w['objectives'])}|lanes={','.join(map(str, w['lanes']))}")


def measure(args) -> dict:
    from ops.modern_world_bench import build_world, init_batch, warm_up
    cfg, run = build_world(args)
    timed = jax.jit(jax.vmap(lambda s: run(s, args.ticks)))
    batch = init_batch(cfg, args.envs)
    t0 = time.time()
    timed = timed.lower(batch).compile()               # one compile; run this executable throughout
    temp = timed.memory_analysis()
    temp_bytes = None if temp is None else int(temp.temp_size_in_bytes)
    batch, _ = warm_up(timed, batch, args.warm_ticks, args.ticks)
    warm_s = time.time() - t0
    best, over, packets = float("inf"), 0, None
    for _ in range(args.repeats):
        t0 = time.time()
        batch, (po, mo, *used) = timed(batch)
        jax.block_until_ready(batch.t)
        best = min(best, time.time() - t0)
        over = max(over, int(po.max()), int(mo.max()))
        if used:                                       # (main, follow-up) valid packets per tick
            pm, pf = (int(u.max()) for u in used[0])
            packets = (max(pm, packets[0]), max(pf, packets[1])) if packets else (pm, pf)
    stats = jax.devices()[0].memory_stats() or {}
    peak = stats.get("peak_bytes_in_use")
    return {"env_ticks_per_s": round(args.envs * args.ticks / best, 1), "steady_s": round(best, 4),
            "program_temp_bytes": temp_bytes, "peak_device_bytes": None if peak is None else int(peak),
            "warm_compile_and_run_s": round(warm_s, 1), "overflow": over,
            "packets_max": packets and packets[0], "follow_up_packets_max": packets and packets[1]}


def compare(now: dict, base: dict, tol: float) -> list[str]:
    bad = []
    if now["env_ticks_per_s"] < (1.0 - tol) * base["env_ticks_per_s"]:
        bad.append(f"throughput {now['env_ticks_per_s']:.1f} < {1 - tol:.2f} x baseline {base['env_ticks_per_s']:.1f}")
    for k in ("peak_device_bytes", "program_temp_bytes"):
        if now.get(k) is not None and base.get(k) is not None and now[k] > (1.0 + tol) * base[k]:
            bad.append(f"{k} {now[k]} > {1 + tol:.2f} x baseline {base[k]}")
    return bad


def git_rev() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).parent, check=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    from ops.modern_world_bench import add_world_args
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--envs", type=int, default=None, help="default: 256 on GPU, 16 on CPU")
    ap.add_argument("--ticks", type=int, default=150)
    ap.add_argument("--warm-ticks", type=int, default=1800)
    ap.add_argument("--repeats", type=int, default=3, help="timed calls; the fastest counts")
    ap.add_argument("--tolerance", type=float, default=0.15)
    ap.add_argument("--baseline", type=Path, default=BASELINE)
    ap.add_argument("--update-baseline", action="store_true", help="record this run as the baseline for its key")
    add_world_args(ap)
    args = ap.parse_args()
    from lanerl_jax.jax_cache import enable_compile_cache
    enable_compile_cache()
    if args.envs is None:
        args.envs = DEFAULT_ENVS.get(jax.default_backend(), 16)
    device = jax.devices()[0]
    key = config_key(args, device)
    now = measure(args)
    doc = json.loads(args.baseline.read_text()) if args.baseline.exists() else {"entries": {}}
    entries = doc.setdefault("entries", {})
    base = entries.get(key)
    if args.update_baseline:
        entries[key] = {**now, "recorded": datetime.date.today().isoformat(), "commit": git_rev()}
        doc.pop("status", None)
        args.baseline.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
        verdict, code, problems = "baseline-updated", 0, []
    elif base is None:
        verdict, code, problems = "no-baseline", 3, [f"no baseline for {key}; rerun with --update-baseline"]
    else:
        problems = compare(now, base, args.tolerance)
        verdict, code = ("regression", 1) if problems else ("ok", 0)
    print(json.dumps({"key": key, "verdict": verdict, "problems": problems, "measured": now, "baseline": base}),
          flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
