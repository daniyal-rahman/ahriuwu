"""Survey the top lane of a replay archive from the replay footers alone (no decoding).

    python tools/replay_top_survey.py INDEX.json OUT_DIR [--build 16.19.823.0722]

``INDEX.json`` is a decoder cohort index (``rows`` with ``replay_path``). For every replay the footer's ten
participant rows are read (the decoder's ``batch_export_bc.footer``, read-only import) and the two
footer-assigned TOP players are kept with their end-of-game stats. Writes ``rows.json`` (one row per replay:
version, size, footer sha256, the two top laners) and ``summary.json`` (supported-build games per top-lane
champion, top-vs-top matchup counts, build histogram, failures). Footer roles are assignments, not measured
lane occupancy.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

DECODER_SCRIPTS = Path("/mnt/nfs/projects/ahriuwu-rofl-decoder-2619/scripts/rofl_decode")
STATS = ("WIN", "LEVEL", "MINIONS_KILLED", "GOLD_EARNED", "CHAMPIONS_KILLED", "NUM_DEATHS", "ASSISTS",
         "TIME_PLAYED", "WAS_AFK", "WAS_LEAVER", "GAME_ENDED_IN_EARLY_SURRENDER")


def top_laners(stats):
    out = []
    for row in stats:
        lane = str(row.get("TEAM_POSITION") or row.get("INDIVIDUAL_POSITION") or "").strip().upper()
        if lane != "TOP":
            continue
        out.append(dict(champion=row.get("SKIN"), team=str(row.get("TEAM", "")),
                        summoners=[row.get("SUMMONER_SPELL_1"), row.get("SUMMONER_SPELL_2")],
                        **{k.lower(): row.get(k) for k in STATS}))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("index", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--build", default="16.19.823.0722", help="the decoder/sim client build")
    a = p.parse_args(argv)
    sys.path.append(str(DECODER_SCRIPTS))
    from batch_export_bc import footer
    paths = [r["replay_path"] for r in json.loads(a.index.read_text())["rows"]]
    rows, t0 = [], time.perf_counter()
    for i, path in enumerate(sorted(paths)):
        row = dict(name=Path(path).stem, replay_path=path)
        try:
            version, stats, size, sha = footer(path)
            row.update(status="indexed", version=version, size_bytes=size, footer_sha256=sha, top=top_laners(stats))
        except Exception as exc:                                     # noqa: BLE001 - recorded per replay
            row.update(status="failed", failure=f"{type(exc).__name__}: {exc}")
        rows.append(row)
        if (i + 1) % 1000 == 0:
            print(f"{i + 1}/{len(paths)} footers ({time.perf_counter() - t0:.0f}s)", flush=True)
    ok = [r for r in rows if r["status"] == "indexed"]
    build = [r for r in ok if r["version"] == a.build]
    clean = [r for r in build if len(r["top"]) == 2 and {t["team"] for t in r["top"]} == {"100", "200"}]
    per_champion = Counter(t["champion"] for r in clean for t in r["top"])
    matchups = Counter(tuple(sorted(t["champion"] for t in r["top"])) for r in clean)
    summary = dict(
        files=len(rows), indexed=len(ok), footer_failures=len(rows) - len(ok),
        versions=dict(Counter(r["version"] for r in ok)), build=a.build, build_games=len(build),
        build_games_with_two_top_laners=len(clean), top_lane_champion_games=sum(per_champion.values()),
        champions=len(per_champion),
        games_per_champion=dict(per_champion.most_common()),
        coverage=[dict(min_games=m, champions=sum(1 for c in per_champion.values() if c >= m),
                       champion_games=sum(c for c in per_champion.values() if c >= m))
                  for m in (1, 20, 50, 100, 200)],
        matchups=[dict(champions=list(k), games=v) for k, v in matchups.most_common(60)],
        distinct_matchups=len(matchups),
        scope="footer-assigned TOP role (TEAM_POSITION, else INDIVIDUAL_POSITION); one per team; not lane occupancy",
        index=str(a.index), seconds=round(time.perf_counter() - t0, 1))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "rows.json").write_text(json.dumps(rows, indent=0))
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("games_per_champion", "matchups")}, indent=1))
    print("top 30:", per_champion.most_common(30))


if __name__ == "__main__":
    main()
