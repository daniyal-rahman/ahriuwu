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


def _trace_server_dir() -> Path:
    """The instrumented build. See `run_oracle_on_server`'s note: the default
    `paths.server_dir()` is bin/Release, which lacks every env-gated diagnostic
    added to the vendored source, and ignores the flags silently."""
    from lanerl_train import paths
    d = paths.server_dir().parent.parent / "Trace" / "net6.0"
    if not d.exists():
        raise SystemExit(f"instrumented build missing: {d}")
    return d


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


def _one(seed: int, decisions: int, port_base: int, shuffled: int | None,
         policy_name: str = "oracle"):
    from .gate3_first_divergence import Recorder
    from .last_hit_drive import (noisy_policy, run_oracle_in_sim,
                                 run_oracle_on_server)

    # One policy object per SEED, shared by both engines, so the arm is
    # identical on each side and any outcome difference is the engines.
    pol = noisy_policy(seed) if policy_name == "noisy" else None

    out = {}
    rec = Recorder()
    sim = run_oracle_in_sim(decisions=decisions, seed=seed, on_oracle=rec,
                            policy=pol)
    out["sim"] = dict(cs=sim.cs, deaths=sim.deaths, attacks=sim.attacks,
                      levels=_levels(rec), xp_share=_xp_range_share(rec))

    rec = Recorder()
    env = {"LANERL_SHUFFLE_ORDER": str(shuffled)} if shuffled is not None else None
    # BOTH arms run the INSTRUMENTED build. The shuffle lives only there
    # (bin/Release is a separate, older build), and if only the shuffled arm
    # switched binaries the comparison would be between BUILDS rather than
    # between update orders.
    srv = run_oracle_on_server(decisions=decisions, bot_seed=seed,
                               port_base=port_base, on_oracle=rec,
                               extra_env=env, policy=pol,
                               server_dir=_trace_server_dir())
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
    ap.add_argument("--policy", choices=("oracle", "noisy"), default="oracle",
                    help="`oracle` is the scripted last-hitter: deterministic, and since this scenario runs with bot_teams=\"none\" the seed drives NOTHING through it -- a 30-seed sweep returned 30 byte-identical outcomes. `noisy` wanders with probability 0.15, seeded, which is the only way a seed enters this experiment at all, and it visits the turret-aggro, death and off-route states the oracle never reaches.")
    ap.add_argument("--save", type=Path, default=None)
    a = ap.parse_args(argv)

    rows = []
    for i, s in enumerate(a.seeds):
        r = _one(s, a.decisions, a.port_base + i * 4, None, a.policy)
        if a.shuffled:
            r.update(_one(s, a.decisions, a.port_base + i * 4 + 2,
                          shuffled=1000 + s, policy_name=a.policy))
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

    # GUARD: a seed that changes nothing is not a sample.
    # This scenario has no bots, so `bot_seed` drives nothing and the scripted
    # oracle is deterministic; a 30-seed sweep once returned 30 byte-identical
    # outcomes and was very nearly reported as a distribution. Identical rows
    # are not a pass, they are an inert knob, and the difference is invisible
    # in every summary statistic.
    if len(rows) > 1:
        sig = {json.dumps({k: v for k, v in r.items() if k != "seed"},
                          sort_keys=True, default=str) for r in rows}
        if len(sig) == 1:
            print(f"\n!! SEEDS ARE INERT: all {len(rows)} seeds produced "
                  f"byte-identical outcomes.\n"
                  f"   This is ONE run measured {len(rows)} times, not a sample, "
                  f"and no gap below\n"
                  f"   means anything. With --policy oracle that is expected: "
                  f"the scenario runs\n"
                  f"   bot_teams=\"none\" and the oracle is deterministic, so the "
                  f"seed has nothing\n"
                  f"   to act on. Re-run with --policy noisy, where the seed "
                  f"enters the policy.")
        else:
            print(f"\n   seeds produce {len(sig)} distinct outcomes of "
                  f"{len(rows)} -- the knob is live")

    print(f"\n-- outcome gaps over {len(rows)} seeds, PAIRED --")
    print("   The three arms share a seed, so the comparison is paired and the\n"
          "   test is on the per-seed DIFFERENCE. Comparing two means without a\n"
          "   test was the first version of this and it was wrong: the spreads\n"
          "   are as large as the means (cs 2.000 +/- 2.066 against 1.533 +/-\n"
          "   2.262), so mean-vs-mean called EXCESS on all five metrics when a\n"
          "   paired test separates only two. A verdict that ignores variance is\n"
          "   not a verdict.\n")
    print(f"   {'metric':<13} {'mean diff':>10} {'95% CI':>22} {'t':>7}  verdict")
    for m in ("cs", "deaths", "max_level", "levelup_lag", "xp_share"):
        if not sh:
            va = [g[m] for g in ss
                  if not (isinstance(g[m], float) and math.isnan(g[m]))]
            print(f"   {m:<13} {statistics.mean(va) if va else float('nan'):>10.3f}"
                  f"{'  (no reference -- pass --shuffled)':>40}")
            continue
        d = [a[m] - b[m] for a, b in zip(ss, sh)
             if not (isinstance(a[m], float) and math.isnan(a[m]))
             and not (isinstance(b[m], float) and math.isnan(b[m]))]
        n = len(d)
        mu = statistics.mean(d) if n else float("nan")
        sd = statistics.stdev(d) if n > 1 else 0.0
        se = sd / math.sqrt(n) if n else float("nan")
        t = mu / se if se else float("nan")
        lo, hi = mu - 1.96 * se, mu + 1.96 * se
        v = "EXCESS" if lo > 0 else ("PASS" if hi < 0 else "INDISTINGUISHABLE")
        print(f"   {m:<13} {mu:>10.3f} {lo:>10.2f}..{hi:<10.2f} {t:>7.2f}  {v}")
    print("\n   diff > 0 means the SIM deviates from the server more than the\n"
          "   server deviates from ITSELF under a permuted update order.\n"
          "   INDISTINGUISHABLE is neither pass nor fail: at this sample size the\n"
          "   experiment cannot separate them, and that is the honest report.")

    if sh and all(
            statistics.mean([g[m] for g in sh
                             if not (isinstance(g[m], float) and math.isnan(g[m]))]
                            or [0.0]) == 0.0
            for m in ("cs", "deaths", "max_level", "levelup_lag", "xp_share")):
        print("\n!! REFERENCE GAP IS EXACTLY ZERO ON EVERY METRIC.\n"
              "   The shuffled arm almost certainly ran WITHOUT the shuffle, so both\n"
              "   arms were the same server and every verdict above is a comparison\n"
              "   against zero -- which is the exact-parity trap this gate exists to\n"
              "   escape. Measured cause, once: `paths.server_dir()` is bin/Release,\n"
              "   a SEPARATE and older build from the instrumented bin/Trace one, so\n"
              "   LANERL_SHUFFLE_ORDER was set and silently ignored. Confirm the\n"
              "   server log contains `SHUFFLE_ORDER active` before believing a PASS.")

    if a.shuffled:
        print("\n   PASS means the simulator differs from the server by no more\n"
              "   than the server differs from ITSELF under an equally arbitrary\n"
              "   update order. EXCESS is the only thing worth investigating.")
    else:
        print("\n   Re-run with --shuffled for the reference gap; without it\n"
              "   these numbers have no scale and cannot pass or fail anything.")


if __name__ == "__main__":
    main()
