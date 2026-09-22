"""
Gate 3 as an OUTCOME comparison across seeds, judged against the order floor.

WHY THIS REPLACES A ROW-BY-ROW GATE
-----------------------------------
`FLOOR-001` measured what permuting the server's own update order does: by
mid-game it changes 31-45% of minion targets and shifts when units die. A
reference that unstable cannot be matched trajectory-for-trajectory, and
chasing the last fraction of a percent of one-step residual was buying
nothing -- `AACD-001` and `HADTGT-001` between them showed that most of what
gate 1 counted was the harness, not the port.

So the gate that decides whether this simulator is fit to train on is:

    is the SIM-vs-SERVER outcome gap no larger than the
    SERVER-vs-SHUFFLED-SERVER outcome gap?

If yes, the port is as close to the server as the server is to itself under
an equally arbitrary scheduling choice, and the remaining difference cannot
matter to a policy. If no, the excess is real and worth finding.

Both halves are measured here on the same seeds with the same metrics, which
is the only way the comparison means anything.

WHAT IS COMPARED
----------------
CS, deaths, the level-up decision index for each level (the XP curve, which
`GATE3-002` showed is where the gap actually lives), and the share of
decisions inside League's 1,400 u XP radius -- the factor that decomposition
identified as the mechanism.

USAGE
    python -m lanerl_jax.parity.gate3_outcomes --seeds 0 1 2 --decisions 9000
    python -m lanerl_jax.parity.gate3_outcomes --seeds ... --shuffled
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
from pathlib import Path


def _levels(rec) -> dict:
    """level -> first decision index it was seen at."""
    ups, prev = {}, None
    for i in sorted(rec.inputs):
        lvl = rec.inputs[i]["level"]
        if lvl != prev:
            ups.setdefault(lvl, i)
            prev = lvl
    return ups


def _xp_range_share(rec, radius: float = 1400.0) -> float:
    n = inr = 0
    for v in rec.inputs.values():
        if not v["minions"]:
            continue
        d = min(math.hypot(m[0] - v["cx"], m[1] - v["cy"]) for m in v["minions"])
        n += 1
        inr += d <= radius
    return inr / n if n else float("nan")


def _one(seed: int, decisions: int, port_base: int, shuffled: int | None):
    from .gate3_first_divergence import Recorder
    from .last_hit_drive import run_oracle_in_sim, run_oracle_on_server

    out = {}
    rec = Recorder()
    sim = run_oracle_in_sim(decisions=decisions, seed=seed, on_oracle=rec)
    out["sim"] = dict(cs=sim.cs, deaths=sim.deaths, attacks=sim.attacks,
                      levels=_levels(rec), xp_share=_xp_range_share(rec))

    rec = Recorder()
    env = {"LANERL_SHUFFLE_ORDER": str(shuffled)} if shuffled is not None else None
    srv = run_oracle_on_server(decisions=decisions, bot_seed=seed,
                               port_base=port_base, on_oracle=rec,
                               extra_env=env)
    key = "shuffled" if shuffled is not None else "server"
    out[key] = dict(cs=srv.cs, deaths=srv.deaths, attacks=srv.attacks,
                    levels=_levels(rec), xp_share=_xp_range_share(rec))
    return out


def _gap(a: dict, b: dict) -> dict:
    """The per-seed difference between two outcome dicts."""
    la, lb = a["levels"], b["levels"]
    shared = sorted(set(la) & set(lb))
    lvl_gap = (statistics.mean(abs(la[l] - lb[l]) for l in shared)
               if shared else float("nan"))
    return {
        "cs": abs(a["cs"] - b["cs"]),
        "deaths": abs(a["deaths"] - b["deaths"]),
        "max_level": abs(max(la, default=0) - max(lb, default=0)),
        "levelup_lag": lvl_gap,
        "xp_share": abs(a["xp_share"] - b["xp_share"]),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--decisions", type=int, default=9000,
                    help="9000 = 5 game minutes. `GATE3-002` showed the level "
                        "gap opens by decision 7,460, so this captures the "
                        "whole mechanism at half the cost of the full 18,000.")
    ap.add_argument("--port-base", type=int, default=48100)
    ap.add_argument("--shuffled", action="store_true",
                    help="ALSO run the server with its update order permuted, "
                        "on the same seeds, to get the reference gap this "
                        "gate is judged against.")
    ap.add_argument("--save", type=Path, default=None)
    a = ap.parse_args(argv)

    rows = []
    for i, s in enumerate(a.seeds):
        r = _one(s, a.decisions, a.port_base + i * 4, None)
        if a.shuffled:
            r.update(_one(s, a.decisions, a.port_base + i * 4 + 2,
                          shuffled=1000 + s))
        rows.append({"seed": s, **r})
        print(f"  seed {s}: sim cs={r['sim']['cs']} d={r['sim']['deaths']} "
              f"| server cs={r['server']['cs']} d={r['server']['deaths']}"
              + (f" | shuffled cs={r['shuffled']['cs']} "
                 f"d={r['shuffled']['deaths']}" if a.shuffled else ""),
              flush=True)

    if a.save:
        a.save.parent.mkdir(parents=True, exist_ok=True)
        a.save.write_text(json.dumps(rows, indent=1, default=str))
        print(f"\nsaved {len(rows)} seeds -> {a.save}")

    print(f"\n-- outcome gaps over {len(rows)} seeds --")
    print(f"   {'metric':<14} {'sim vs server':>22} "
          + (f"{'server vs shuffled':>22}   verdict" if a.shuffled else ""))
    ss = [_gap(r["sim"], r["server"]) for r in rows]
    sh = [_gap(r["server"], r["shuffled"]) for r in rows] if a.shuffled else None
    for m in ("cs", "deaths", "max_level", "levelup_lag", "xp_share"):
        va = [g[m] for g in ss if not (isinstance(g[m], float) and math.isnan(g[m]))]
        sa = statistics.mean(va) if va else float("nan")
        line = f"   {m:<14} {sa:>22.3f}"
        if sh:
            vb = [g[m] for g in sh
                  if not (isinstance(g[m], float) and math.isnan(g[m]))]
            sb = statistics.mean(vb) if vb else float("nan")
            ok = "PASS" if sa <= sb else "EXCESS"
            line += f" {sb:>22.3f}   {ok}"
        print(line)

    if a.shuffled:
        print("\n   PASS means the simulator differs from the server by no more\n"
              "   than the server differs from ITSELF under an equally arbitrary\n"
              "   update order. EXCESS is the only thing worth investigating.")
    else:
        print("\n   Re-run with --shuffled for the reference gap; without it\n"
              "   these numbers have no scale and cannot pass or fail anything.")


if __name__ == "__main__":
    main()
