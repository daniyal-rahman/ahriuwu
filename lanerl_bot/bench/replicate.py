#!/usr/bin/env python
"""Run each configuration across several seeds and report mean/spread.

Run-to-run spread on this server is real (minion AI and crit rolls use their own
RNG), so a single 10-minute game is not enough to rank two configs.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import run_server

OUT = Path(__file__).resolve().parent / "out"


def cs10(res: dict, name: str) -> dict | None:
    best = None
    for r in res["cs_rows"]:
        if r["name"] == name and r["t"] <= 601_000:
            best = r
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True, help="JSON: {tag: envdict}")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--port0", type=int, default=5600)
    ap.add_argument("--out", default="replicate.json")
    ap.add_argument("--retries", type=int, default=1)
    args = ap.parse_args()

    configs = json.loads(Path(args.configs).read_text())
    # keys starting with "_" are notes (see out/cfg_tune.json), never arms
    configs = {k: v for k, v in configs.items() if not k.startswith("_")}
    if not configs:
        raise SystemExit(f"{args.configs} defines no arms")

    # Validate EVERY arm before the first launch: a sweep whose 3rd arm names a
    # config that does not exist should die in one second, not after an hour of
    # games whose numbers turn out to be the default bot measured three times.
    # "blue" here is not a new choice, it is the old server-side default made
    # visible. Every arm in qab_configs.json omits LANERL_BOT and used to get a
    # blue-driven champion from LanerlBotConfig.DriveTeams; that default is now
    # "none", so leaving it omitted would have turned each arm into a ten-minute
    # recording of a champion standing still and reported its 0 CS as the
    # result. An arm that wants something else still overrides it in `extra`.
    base_env = {"LANERL_TOPONLY": "1", "LANERL_EXIT_AT": "601000",
                "LANERL_BOT": "blue"}

    for tag, extra in configs.items():
        run_server.check_bot_config(extra, where=f"{args.configs}:{tag}")
        # Validate the MERGED env, not `extra`: the mode usually comes from
        # base_env, and checking the arm alone would reject every valid config.
        run_server.check_bot_mode(base_env | extra, where=f"{args.configs}:{tag}")

    OUT.mkdir(parents=True, exist_ok=True)
    port = args.port0
    report = {}

    for tag, extra in configs.items():
        vals, rows, crashes = [], [], 0
        for s in range(args.seeds):
            for attempt in range(args.retries + 1):
                env = base_env | {"LANERL_BOT_SEED": str(1234 + s)} | extra
                log = OUT / f"rep_{tag}_s{s}_a{attempt}.log"
                res = run_server.run(env, log, port=port, timeout_s=900)
                port += 1
                who = "redbot" if env.get("LANERL_BOT") == "purple" else "bluebot"
                r = cs10(res, who)
                if r is None:
                    crashes += 1
                    continue
                vals.append(r["cs"])
                rows.append({"seed": 1234 + s, **{k: r[k] for k in
                             ("cs", "gold", "lvl", "deaths")}})
                break
        report[tag] = {
            "env": extra,
            "n": len(vals),
            "crashes": crashes,
            "cs_mean": round(statistics.mean(vals), 1) if vals else None,
            "cs_median": statistics.median(vals) if vals else None,
            "cs_min": min(vals) if vals else None,
            "cs_max": max(vals) if vals else None,
            "cs_stdev": round(statistics.stdev(vals), 1) if len(vals) > 1 else None,
            "rows": rows,
        }
        r = report[tag]
        print(f"{tag:28s} n={r['n']} crashes={crashes} "
              f"cs mean={r['cs_mean']} median={r['cs_median']} "
              f"range=[{r['cs_min']},{r['cs_max']}] sd={r['cs_stdev']}", flush=True)

    (OUT / args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {OUT / args.out}")


if __name__ == "__main__":
    main()
