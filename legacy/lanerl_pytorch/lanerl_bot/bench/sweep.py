#!/usr/bin/env python
"""Sweep bot knobs and report CS@10 for each, so tuning is measured rather than argued."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import run_server

OUT = Path(__file__).resolve().parent / "out"


def cs10(res: dict, name: str = "bluebot") -> dict:
    best = None
    for r in res["cs_rows"]:
        if r["name"] == name and r["t"] <= 601_000:
            best = r
    return best or {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True, help="JSON list of env dicts")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--port0", type=int, default=5500)
    args = ap.parse_args()

    grid = json.loads(Path(args.grid).read_text())
    # same reason as replicate.py: validate the whole grid up front, so a bad
    # LANERL_BOT_CONFIG costs a second rather than the whole sweep's worth of
    # games silently measuring the default bot
    base_env = {"LANERL_TOPONLY": "1", "LANERL_EXIT_AT": "601000",
                "LANERL_BOT": "blue"}
    for i, extra in enumerate(grid):
        run_server.check_bot_config(extra, where=f"{args.grid}[{i}]")
        # merged, not `extra`: the mode comes from base_env unless overridden
        run_server.check_bot_mode(base_env | extra, where=f"{args.grid}[{i}]")

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, extra in enumerate(grid):
        env = base_env | extra
        log = OUT / f"{args.tag}_{i}.log"
        res = run_server.run(env, log, port=args.port0 + i, timeout_s=900)
        r = cs10(res)
        row = {"i": i, "env": extra, "cs": r.get("cs"), "gold": r.get("gold"),
               "lvl": r.get("lvl"), "deaths": r.get("deaths"),
               "wall_s": round(res["wall_s"], 1),
               "fatal": res["fatal"][:1]}
        rows.append(row)
        print(f"[{i}] {extra} -> cs={row['cs']} lvl={row['lvl']} deaths={row['deaths']}",
              flush=True)

    (OUT / f"sweep_{args.tag}.json").write_text(json.dumps(rows, indent=2))
    print("\nbest:", max((r for r in rows if r["cs"] is not None),
                         key=lambda r: r["cs"], default=None))


if __name__ == "__main__":
    main()
