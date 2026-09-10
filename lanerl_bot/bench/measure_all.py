#!/usr/bin/env python
"""The headline measurements: CS@10 for the bot vs a do-nothing baseline, and
the cost of an in-process episode reset.

Every number printed here comes out of a real headless game on the real server.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import run_server

OUT = Path(__file__).resolve().parent / "out"

BENCH_SUMMARY_RE = re.compile(
    r"LANERL_RESET_BENCH_SUMMARY n=(\d+) median_ms=([\d.]+) mean_ms=([\d.]+) "
    r"min_ms=([\d.]+) max_ms=([\d.]+) played_ms_per_episode=(\d+)"
)


def cs_at_10(res: dict) -> dict:
    rows = res["cs_rows"]
    out = {}
    for r in rows:
        if r["t"] <= 601_000:
            out[r["name"]] = r
    return out


def do_run(name: str, env: dict, port: int, timeout: float) -> dict:
    log = OUT / f"{name}.log"
    res = run_server.run(env, log, port=port, timeout_s=timeout)
    rec = {
        "name": name,
        "env": env,
        "wall_s": round(res["wall_s"], 1),
        "timed_out": res["timed_out"],
        "tps_median": sorted(res["tps"])[len(res["tps"]) // 2] if res["tps"] else None,
        "attach": res["attach"],
        "reset_warns": res["reset_warns"][:5],
        "fatal": res["fatal"],
        "cs_at_10min": cs_at_10(res),
        "cs_rows": res["cs_rows"],
        "reset_lines": res["reset_lines"],
        "n_episodes": len(res["episode_lines"]),
    }
    print(f"--- {name}: wall={rec['wall_s']}s tps={rec['tps_median']} ---")
    for n, r in rec["cs_at_10min"].items():
        print(f"    {n}: cs={r['cs']} gold={r['gold']} lvl={r['lvl']} "
              f"deaths={r['deaths']} t={r['t']}")
    if rec["fatal"]:
        print("    FATAL:", rec["fatal"][:2])
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    port = 5410

    base = {"LANERL_TOPONLY": "1", "LANERL_EXIT_AT": "601000"}

    def want(tag: str) -> bool:
        return args.only is None or args.only == tag

    # 1. do-nothing baseline: champions spawn and are never driven
    if want("baseline"):
        results.append(do_run("baseline_donothing",
                              base | {"LANERL_BOT": "none"}, port, 900))
        port += 1

    # 2. the bot, blue side only, across seeds
    if want("bot"):
        for s in range(args.seeds):
            results.append(do_run(f"bot_blue_seed{s}",
                                  base | {"LANERL_BOT": "blue",
                                          "LANERL_BOT_SEED": str(1234 + s)},
                                  port, 900))
            port += 1

    # 3. anchor vs anchor -- both champions driven, the real self-play opponent setting
    if want("both"):
        results.append(do_run("bot_both",
                              base | {"LANERL_BOT": "both"}, port, 900))
        port += 1

    # 4. the difficulty curriculum: same policy, degraded execution
    if want("curriculum"):
        for acc, react, derr, tag in [
            (1.00, 100, 0.00, "diamond"),
            (0.85, 250, 0.08, "gold"),
            (0.65, 450, 0.18, "bronze"),
        ]:
            results.append(do_run(
                f"curriculum_{tag}",
                base | {"LANERL_BOT": "blue",
                        "LANERL_BOT_ACCURACY": str(acc),
                        "LANERL_BOT_REACTION_MS": str(react),
                        "LANERL_BOT_DMG_ERR": str(derr)},
                port, 900))
            port += 1

    # 5. reset benchmark: play 2 game-minutes, reset, repeat
    if want("reset"):
        log = OUT / "reset_bench.log"
        res = run_server.run(
            {"LANERL_TOPONLY": "1", "LANERL_BOT": "both",
             "LANERL_RESET_BENCH": "8", "LANERL_RESET_BENCH_AT": "120000"},
            log, port=port, timeout_s=1200)
        summary = None
        for line in (OUT / "reset_bench.log").read_text(errors="replace").splitlines():
            m = BENCH_SUMMARY_RE.search(line)
            if m:
                summary = {
                    "n": int(m.group(1)), "median_ms": float(m.group(2)),
                    "mean_ms": float(m.group(3)), "min_ms": float(m.group(4)),
                    "max_ms": float(m.group(5)),
                    "played_ms_per_episode": int(m.group(6)),
                }
        rec = {"name": "reset_bench", "wall_s": round(res["wall_s"], 1),
               "summary": summary, "lines": res["reset_lines"],
               "warns": res["reset_warns"][:5], "fatal": res["fatal"]}
        print("--- reset_bench ---")
        print(json.dumps(summary, indent=2))
        for l in res["reset_lines"]:
            print("   ", l)
        results.append(rec)

    (OUT / "measure_all.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT / 'measure_all.json'}")


if __name__ == "__main__":
    main()
