#!/usr/bin/env python
"""Is this run healthy RIGHT NOW, across every way it is known to fail?

`rl_health.py` grades a finished run's optimisation. This grades a LIVE one,
and it covers the failure classes that have actually cost this project time --
most of which produce no error at all:

    the server died            a crashed instance is a restart line in a log,
                               not an exception the trainer sees
    the server got slow        freerun speed collapsing looks like "training
                               is just slow"
    the agent never laned      13,475 updates were spent with the agent
                               387 units from spawn; CS stayed 0 and nothing
                               said why
    CS flat at zero            the headline metric read 0.0 for a whole run
                               because the eval bucket held one anchor game
    the rune page came back    an in-process reset used to strip it, so every
                               episode after the first was a weaker champion
    gradients blew up          or vanished, or went NaN
    the critic never learned   value_loss looks fine while explained variance
                               is negative -- it is 57% of the network
    the policy collapsed       one button takes all the mass; the BC prior is
                               gone and entropy alone will not show it
    rollouts thrown away       40% rejected on staleness is 40% of the servers'
                               work discarded

  python lanerl/run_watch.py runs/<name>            # one report
  python lanerl/run_watch.py runs/<name> --watch 60 # every 60s until stopped
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

OK, WARN, BAD = "ok", "warn", "BAD"


def _tail_json(path: Path, limit: int = 4000) -> List[dict]:
    if not path.exists():
        return []
    lines = path.read_text(errors="ignore").splitlines()[-limit:]
    out = []
    for l in lines:
        if l.strip():
            try:
                out.append(json.loads(l))
            except Exception:
                pass
    return out


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return statistics.mean(xs) if xs else None


class Report:
    def __init__(self) -> None:
        self.rows: List[tuple] = []

    def add(self, level: str, name: str, detail: str, why: str = "") -> None:
        self.rows.append((level, name, detail, why))

    def render(self) -> int:
        bad = 0
        for level, name, detail, why in self.rows:
            tag = {OK: "  ok ", WARN: " WARN", BAD: " BAD "}[level]
            print(f"[{tag}] {name:26s} {detail}")
            if level != OK and why:
                print(f"         -> {why}")
            bad += level == BAD
        return bad


def check_servers(run: Path, rep: Report) -> None:
    """Server crashes and restarts, read from the instance logs themselves."""
    logs = list(run.glob("actor*_logs/instance*.log")) + list(
        run.glob("anchor_*/instance*.log")
    )
    if not logs:
        rep.add(WARN, "server logs", "none found yet")
        return
    crashes, tracebacks = 0, []
    for f in logs:
        try:
            txt = f.read_text(errors="ignore")
        except OSError:
            continue
        for m in re.finditer(r"(\w*Exception)[^\n]*", txt):
            crashes += 1
            if len(tracebacks) < 3:
                tracebacks.append(f"{f.name}: {m.group(1)}")
    level = OK if crashes == 0 else BAD
    rep.add(level, "server exceptions", f"{crashes} across {len(logs)} instance logs",
            "a crashed server is a silently SHORT episode, not an error the "
            "trainer raises: " + "; ".join(tracebacks))


def check_stat_canary(run: Path, rows: List[dict], rep: Report) -> None:
    """The rune page must still be on after an in-process reset."""
    ads = [r.get("first_frame_ad") for r in rows if r.get("first_frame_ad")]
    if not ads:
        rep.add(WARN, "rune-page canary", "no first_frame_ad yet (needs a finished episode)")
        return
    lo, hi = min(ads), max(ads)
    # base Garen is 57.88; base + rune/mastery pages is 78.14
    level = OK if lo > 70.0 else BAD
    rep.add(level, "rune-page canary", f"first_frame_ad {lo:.2f}..{hi:.2f}",
            "an episode started at BASE attack damage (57.88): the reset is "
            "stripping the rune/mastery page again, and every episode after "
            "the first is a different, weaker champion")
    if hi - lo > 0.01 and lo > 70.0:
        rep.add(WARN, "stat homogeneity", f"spread {hi - lo:.2f} across episodes",
                "episodes are not all starting from the same champion")


def check_throughput(rows: List[dict], rep: Report) -> None:
    d = _mean([r.get("throughput/decisions_per_s") for r in rows])
    lf = _mean([r.get("throughput/learner_frac") for r in rows])
    rej = _mean([r.get("reject_rate") for r in rows])

    if d is not None:
        # Against the 2026-09-13 measured baseline: 3,409 decisions/s at
        # 4 process actors x 12 envs. 1,000 is roughly what ONE actor does, so
        # below that on a multi-actor run means something is not running.
        level = OK if d > 1000 else WARN
        # decisions_per_s counts only rollouts the learner ACCEPTED, so a high
        # reject rate makes healthy collection look like a stall. Report the
        # collection rate too rather than leaving that trap in the headline
        # number -- it is what made a flat ~900/s look like a decline from
        # 498 to 158 and sent three investigations after the wrong thing.
        extra = ""
        if rej:
            extra = f"  (collecting ~{d / max(1e-9, 1.0 - rej):,.0f}/s, {rej:.0%} binned)"
        rep.add(level, "throughput", f"{d:,.0f} decisions/s accepted{extra}",
                "compare against 3,409/s at 4 process actors x 12 envs; check "
                "--actor-mode is 'process' (threaded actors cap at ~1.8 of 16 "
                "cores because the observation build holds the GIL)")

    if lf is not None:
        # NOT a bottleneck signal. This is the share of wall clock between two
        # CONSUMED rollouts, so it reads ~1.0 whenever the learner is simply
        # the thing being waited on in the queue. It was 0.982 on a run whose
        # actual constraint was the GIL, and reading it as "learner-bound"
        # wasted a session. Reported for continuity, never graded.
        rep.add(OK, "learner share", f"{lf:.1%} of wall clock (NOT a bottleneck signal)")

    if rej is not None:
        # 0.15 was far too lenient: the default max_staleness of 1 was binning
        # 45-80% and this check would have called 14% healthy. Any sustained
        # rejection is now worth a look, because with the bound derived from
        # queue_capacity + num_actors - 1 there should be almost none.
        rep.add(OK if rej < 0.05 else BAD, "rollouts accepted", f"{rej:.1%} rejected",
                "that share of the servers' work is being thrown away on "
                "staleness. Every rejection in the 2026-09-13 probe was stale "
                "by EXACTLY 2 against a bound of 1; check max_staleness is not "
                "pinned below queue_capacity + num_actors - 1")


def check_learner(rows: List[dict], rep: Report) -> None:
    loss = [r for r in rows if "loss/loss" in r]
    if not loss:
        rep.add(WARN, "learner", "no update rows yet")
        return
    tail = loss[-max(2, len(loss) // 5):]

    bad_loss = [r["update"] for r in loss
                if not (-1e9 < float(r["loss/loss"]) < 1e9)]
    rep.add(OK if not bad_loss else BAD, "loss finite",
            "finite" if not bad_loss else f"NON-FINITE on {len(bad_loss)} updates",
            f"first at update {bad_loss[0] if bad_loss else ''}: a NaN/Inf loss "
            f"poisons every parameter it touches")

    gn = _mean([r.get("loss/grad_norm") for r in tail])
    if gn is not None:
        level = OK if 1e-4 < gn < 50 else BAD
        rep.add(level, "gradient norm", f"{gn:.3f}",
                "vanished (nothing is learning) or exploded (the step is noise)")

    ev = _mean([r.get("loss/explained_variance") for r in tail])
    if ev is not None:
        level = OK if ev > 0.1 else WARN
        rep.add(level, "critic explains returns", f"explained_variance {ev:+.3f}",
                "the value function is not tracking the returns, so advantages "
                "are mostly noise. It is 57% of the network and starts random")

    kl = _mean([r.get("loss/kl_ref") for r in tail])
    if kl is not None:
        rep.add(OK if kl < 0.25 else WARN, "near the BC prior", f"kl_ref {kl:.3f}",
                "the policy has walked away from the prior it started from")

    exc = _mean([r.get("loss/approx_kl_excess") for r in tail])
    if exc is not None:
        rep.add(OK if exc < 0.05 else WARN, "step size", f"approx_kl excess {exc:.4f}",
                "each update moves the policy too far: lower --lr")


def check_policy(rows: List[dict], rep: Report) -> None:
    """A collapsed policy puts all its mass on one button."""
    acts = [r for r in rows if any(k.startswith("actions/") for k in r)]
    if not acts:
        rep.add(WARN, "action marginals", "not logged yet")
        return
    last = acts[-1]
    marg = {k.split("/", 1)[1]: v for k, v in last.items() if k.startswith("actions/")}
    if not marg:
        return
    top, share = max(marg.items(), key=lambda kv: kv[1])
    level = OK if share < 0.9 else BAD
    rep.add(level, "policy not collapsed",
            "  ".join(f"{k}={v:.2f}" for k, v in sorted(marg.items())),
            f"'{top}' holds {share:.0%} of the mass: the prior is gone")


def check_episodes(rows: List[dict], rep: Report) -> None:
    eps = [r for r in rows if r.get("kind") == "episode"]
    if not eps:
        rep.add(WARN, "episodes", "none finished yet (an episode is 18,000 decisions)")
        return
    cs = [r["cs_at_10"] for r in eps if r.get("cs_at_10") is not None]
    if not cs:
        rep.add(BAD, "cs_at_10 reachable", f"0 of {len(eps)} episodes reported one",
                "cs_at_10 is an ABSOLUTE ten-minute metric: an episode that "
                "ends early can never produce one")
    else:
        m = _mean(cs)
        rep.add(OK if m > 1.0 else BAD, "CS being scored",
                f"mean {m:.1f} over {len(cs)} episodes (sd {statistics.pstdev(cs):.1f})",
                "the agent is farming nothing -- check that it is reaching lane "
                "at all before reading anything else")
        if len(cs) >= 8:
            h = len(cs) // 2
            rep.add(OK, "CS trend",
                    f"first half {_mean(cs[:h]):.1f} -> last half {_mean(cs[h:]):.1f}")
    lens = [r["length_steps"] for r in eps if r.get("length_steps")]
    if lens:
        rep.add(OK if _mean(lens) > 15000 else WARN, "episode length",
                f"mean {_mean(lens):,.0f} decisions ({_mean(lens)/30:.0f}s)",
                "episodes are ending early; cs_at_10 needs the full ten minutes")
    deaths = [r.get("deaths") for r in eps if r.get("deaths") is not None]
    if deaths:
        rep.add(OK, "deaths", f"mean {_mean(deaths):.2f} per episode")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--watch", type=int, default=0, help="seconds between reports")
    args = ap.parse_args()
    run = Path(args.run_dir)

    while True:
        rows = _tail_json(run / "metrics.jsonl")
        rep = Report()
        upd = max((r.get("update", 0) for r in rows), default=0)
        print(f"\n=== {run.name}  update {upd}  "
              f"{time.strftime('%H:%M:%S')} ===")
        check_servers(run, rep)
        check_throughput(rows, rep)
        check_learner(rows, rep)
        check_policy(rows, rep)
        check_episodes(rows, rep)
        check_stat_canary(run, rows, rep)
        bad = rep.render()
        print(f"\n{'HEALTHY' if not bad else f'{bad} FAILING CHECK(S)'}")
        if not args.watch:
            return 0 if not bad else 2
        time.sleep(args.watch)


if __name__ == "__main__":
    sys.exit(main())
