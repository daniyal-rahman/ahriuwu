"""Gate 3, measured at the FIRST divergent decision instead of the last one.

Gate 3 currently asserts on one integer: CS@10 in the sim against CS@10 on the
server (today 7 against 4).  Both engines are bit-reproducible -- four server
runs identical, eight sim seeds identical, no randomness consumed -- so that
integer has a zero noise floor and is a legitimate *gate*.  It is a terrible
*measurement*.  Two deterministic simulators that agree for 12,000 decisions
and then disagree once will disagree about everything afterwards, so the final
totals are one observation of a cascade, not a sample of independent errors.
"CS 7 vs 4" therefore says almost nothing about which mechanism is wrong, and
that is why the ledger's gate-3 chain has had to be rewritten repeatedly as
the *direction* of the gap flipped (the sim used to under-farm; it now
over-farms) without the underlying cause being found.

The high-power statistic for a pair of deterministic simulators is the index
of the FIRST decision at which they diverge, plus the field that diverged.
Everything before it is verified agreement; everything after it is
contaminated and should not be quoted as evidence at all.

WHY THIS COMPARISON IS EXHAUSTIVE
---------------------------------
Both drivers call the *same* ``last_hit_oracle.decide``.  The oracle is
shared code, so it cannot itself be a source of disagreement: if the two
engines feed it identical inputs it returns identical outputs, by
construction.  Therefore every behavioural difference between the two runs is
a difference in ``decide``'s inputs, and those inputs are small and fully
enumerable:

    champ:   x, y, attack_damage, attack_range
    minions: (x, y, hp, armor, collision_radius) for each VISIBLE red minion

Recording exactly those, via the drivers' own ``on_oracle`` hook, is a
complete account of the divergence -- not a proxy for it.  The hook is used
rather than a re-implemented drive loop on purpose: `PATH-006` is the record
of what re-implementing costs (``isolation.py`` drove a raw two-point path for
weeks while the gate ran routed, so its measured effect described a run that
no longer happened).

WHAT IT CANNOT SEE
------------------
Minion identity.  The sim's ``uid`` is a recycled slot index and the server's
is a NetId, so they are not comparable and are never compared here; minions
are matched by position.  At the first divergence the two populations are by
definition still nearly identical, so nearest-neighbour matching is sound
exactly where the answer is read -- and the match residual is printed so that
assumption is checkable rather than assumed.  Far past the first divergence
the matching degrades, which is fine, because those decisions are
contaminated anyway and this module does not draw conclusions from them.
"""
from __future__ import annotations

import argparse
import math
from typing import Dict, List, Optional, Tuple

#: Tolerances for "the same value".  Positions are compared at the dump's own
#: 1/16-unit Tier-1 target; HP at 1/1024, the quantisation the wire carries.
#: These are the §3 numbers, not tuned-until-green values.
POS_TOL = 1.0 / 16.0
HP_TOL = 1.0 / 1024.0
STAT_TOL = 1.0 / 1024.0


class PathRecorder:
    """Collect the ``on_decision`` stream: the champion's position at EVERY
    decision, including the ~1,200 scripted approach decisions before the
    oracle is ever called.

    This is where the first divergence of the whole episode actually lives.
    Both drivers step at a fixed 30 Hz from t=0, so decision index ``i`` is
    the same game time in both engines regardless of how far either champion
    has walked -- which makes the position at index ``i`` directly
    comparable, and makes "the sim finished the walk-in at decision 1,242
    while the server finished at 1,190" a statement about POSITION LAG, not
    about the drivers falling out of step.

    It also measures the one thing a one-step differential structurally
    cannot.  Tier 1 re-injects the server's state every tick, so it reports
    per-tick error with the accumulation removed by construction; a champion
    that is consistently 0.08 u/tick slow scores as ~100% parity there and
    still ends a 1,200-decision walk ~96 units behind.
    """

    def __init__(self) -> None:
        self.pos: Dict[int, Tuple[float, float]] = {}
        self.approaching: Dict[int, bool] = {}
        self.alive: Dict[int, bool] = {}

    def __call__(self, ev: dict) -> None:
        self.pos[ev["i"]] = (float(ev["x"]), float(ev["y"]))
        self.approaching[ev["i"]] = bool(ev["approaching"])
        self.alive[ev["i"]] = bool(ev["alive"])


def align_lag(sim: PathRecorder, srv: PathRecorder, window: int = 400) -> List[tuple]:
    """Separate "different route" from "same route, running late".

    A raw separation of ~596 u is ambiguous: two champions walking DIFFERENT
    paths 596 u apart and two champions walking the SAME path 596 u apart
    along it produce the identical number, and they have completely different
    causes -- a routing/terrain disagreement versus a speed or start-up
    disagreement.  Only one of them is a pathing bug.

    For each sim decision ``i`` this finds the server decision ``j`` whose
    champion position is closest.  If ``j - i`` is a roughly CONSTANT positive
    offset then the sim is simply running that many decisions late along the
    same route, and the residual distance at the best ``j`` is how far off
    that route it is.  A wandering ``j - i``, or a large residual, means the
    routes themselves differ.
    """
    srv_idx = sorted(srv.pos)
    out = []
    for i in sorted(sim.pos):
        ax, ay = sim.pos[i]
        lo, hi = i - window, i + window
        best_j, best_d = None, float("inf")
        for j in srv_idx:
            if j < lo or j > hi:
                continue
            bx, by = srv.pos[j]
            d = math.hypot(ax - bx, ay - by)
            if d < best_d:
                best_j, best_d = j, d
        if best_j is not None:
            out.append((i, best_j, best_j - i, best_d))
    return out


def compare_paths(sim: PathRecorder, srv: PathRecorder) -> dict:
    """Champion position divergence per decision, over the scripted walk."""
    shared = sorted(set(sim.pos) & set(srv.pos))
    first = None
    rows = []
    for i in shared:
        (ax, ay), (bx, by) = sim.pos[i], srv.pos[i]
        d = math.hypot(ax - bx, ay - by)
        rows.append((i, d, sim.approaching[i], srv.approaching[i]))
        if first is None and d > POS_TOL:
            first = (i, d, (ax, ay), (bx, by))
    return {"first": first, "rows": rows,
            "sim_walk_end": next((i for i in shared if not sim.approaching[i]), None),
            "srv_walk_end": next((i for i in shared if not srv.approaching[i]), None)}


class Recorder:
    """Collect one ``on_oracle`` stream into per-decision records."""

    def __init__(self) -> None:
        self.inputs: Dict[int, dict] = {}
        self.decisions: Dict[int, object] = {}

    def __call__(self, ev: dict) -> None:
        i = ev["i"]
        if "decision" in ev:
            d = ev["decision"]
            self.decisions[i] = (
                ("attack", None) if d.attack is not None else
                ("move", (round(d.move[0], 3), round(d.move[1], 3)))
                if d.move is not None else ("hold", None))
            return
        champ = ev["champ"]
        self.inputs[i] = {
            "level": ev["level"],
            "cx": float(champ.x), "cy": float(champ.y),
            "ad": float(champ.attack_damage),
            "rng": float(champ.attack_range),
            "minions": sorted(
                ((float(m.x), float(m.y), float(m.hp), float(m.armor),
                  float(m.collision_radius)) for m in ev["minions"])),
        }


def _match(a: List[tuple], b: List[tuple]) -> Tuple[List[Tuple[tuple, tuple]], float]:
    """Pair two minion lists by position, greedily and symmetrically.

    Returns the pairs and the worst pairing distance, so a caller can see
    whether the matching itself is trustworthy at the index it is reading.
    """
    remaining = list(b)
    pairs: List[Tuple[tuple, tuple]] = []
    worst = 0.0
    for m in a:
        if not remaining:
            break
        j = min(range(len(remaining)),
                key=lambda k: math.hypot(remaining[k][0] - m[0],
                                         remaining[k][1] - m[1]))
        d = math.hypot(remaining[j][0] - m[0], remaining[j][1] - m[1])
        worst = max(worst, d)
        pairs.append((m, remaining.pop(j)))
    return pairs, worst


def compare(sim: Recorder, srv: Recorder, examples: int = 3) -> Optional[dict]:
    """First index at which the two oracle input streams disagree, or None."""
    shared = sorted(set(sim.inputs) & set(srv.inputs))
    for i in shared:
        a, b = sim.inputs[i], srv.inputs[i]
        why: List[str] = []
        if a["level"] != b["level"]:
            why.append(f"level {a['level']} vs {b['level']}")
        for key, tol in (("cx", POS_TOL), ("cy", POS_TOL),
                         ("ad", STAT_TOL), ("rng", STAT_TOL)):
            if abs(a[key] - b[key]) > tol:
                why.append(f"champ {key} {a[key]:.6f} vs {b[key]:.6f} "
                           f"(delta {a[key] - b[key]:+.6f}, tol {tol})")
        if len(a["minions"]) != len(b["minions"]):
            why.append(f"visible minion count {len(a['minions'])} vs "
                       f"{len(b['minions'])}")
        else:
            pairs, worst = _match(a["minions"], b["minions"])
            for (ma, mb) in pairs:
                if abs(ma[2] - mb[2]) > HP_TOL:
                    why.append(f"minion hp {ma[2]:.4f} vs {mb[2]:.4f} "
                               f"(delta {ma[2] - mb[2]:+.4f}) at "
                               f"({ma[0]:.1f},{ma[1]:.1f})")
                    break
                if math.hypot(ma[0] - mb[0], ma[1] - mb[1]) > POS_TOL:
                    why.append(f"minion position ({ma[0]:.3f},{ma[1]:.3f}) vs "
                               f"({mb[0]:.3f},{mb[1]:.3f})")
                    break
                if abs(ma[3] - mb[3]) > STAT_TOL:
                    why.append(f"minion armor {ma[3]} vs {mb[3]}")
                    break
            if worst > POS_TOL:
                why.append(f"(worst position-match residual {worst:.3f} u)")
        if why:
            return {"i": i, "why": why, "sim": a, "server": b,
                    "sim_decision": sim.decisions.get(i),
                    "server_decision": srv.decisions.get(i)}
    return None


def first_decision_divergence(sim: Recorder, srv: Recorder) -> Optional[int]:
    """First index where the two runs CHOSE differently, which is the thing
    that actually changes the episode.  Reported separately from the input
    divergence: an input can differ harmlessly for a long time before it
    crosses a threshold the oracle acts on, and the distance between those
    two indices is itself the measurement of how much slack there is.
    """
    for i in sorted(set(sim.decisions) & set(srv.decisions)):
        if sim.decisions[i] != srv.decisions[i]:
            return i
    return None


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--decisions", type=int, default=18_000)
    ap.add_argument("--port-base", type=int, default=47100)
    ap.add_argument("--examples", type=int, default=3)
    a = ap.parse_args(argv)

    from .last_hit_drive import run_oracle_in_sim, run_oracle_on_server

    sim_rec, srv_rec = Recorder(), Recorder()
    sim_path, srv_path = PathRecorder(), PathRecorder()
    print(f"sim run ({a.decisions} decisions)...", flush=True)
    sim = run_oracle_in_sim(decisions=a.decisions, on_oracle=sim_rec,
                            on_decision=sim_path)
    print(f"  sim cs={sim.cs} attacks={sim.attacks} deaths={sim.deaths} "
          f"oracle calls={len(sim_rec.inputs)}", flush=True)
    print("server run...", flush=True)
    srv = run_oracle_on_server(decisions=a.decisions, port_base=a.port_base,
                               bot_seed=4242, tag="gate3_first_divergence",
                               autobuy=False, on_oracle=srv_rec,
                               on_decision=srv_path)
    print(f"  server cs={srv.cs} attacks={srv.attacks} deaths={srv.deaths} "
          f"oracle calls={len(srv_rec.inputs)}", flush=True)

    # ---- the whole-episode first divergence, before the oracle exists ----
    paths = compare_paths(sim_path, srv_path)
    print("\n== CHAMPION POSITION, EVERY DECISION (scripted walk included) ==")
    print(f"   walk ended: sim decision {paths['sim_walk_end']}, "
          f"server decision {paths['srv_walk_end']}")
    if paths["first"] is None:
        print(f"   never diverged by more than {POS_TOL} u")
    else:
        i, d, (ax, ay), (bx, by) = paths["first"]
        print(f"   FIRST divergence > {POS_TOL} u at decision {i}: {d:.4f} u")
        print(f"     sim    ({ax:.4f}, {ay:.4f})")
        print(f"     server ({bx:.4f}, {by:.4f})")
        print("   drift curve (decision: separation in units):")
        rows = paths["rows"]
        step = max(1, len(rows) // 20)
        for j in range(0, len(rows), step):
            k, dd, sa, sb = rows[j]
            print(f"     {k:6d}  {dd:10.3f}   approaching sim={sa} server={sb}")
        worst = max(rows, key=lambda r: r[1])
        print(f"   worst separation {worst[1]:.3f} u at decision {worst[0]}")

        print("\n   -- same route running late, or a different route? --")
        lag = align_lag(sim_path, srv_path)
        step2 = max(1, len(lag) // 20)
        print("     sim_i -> best server_j   lag(j-i)   residual distance u")
        for j2 in range(0, len(lag), step2):
            i2, bj, dl, dd = lag[j2]
            print(f"     {i2:6d} -> {bj:6d}   {dl:+7d}   {dd:10.3f}")
        mid = [r for r in lag if r[3] < 50.0]
        if mid:
            lags = sorted(r[2] for r in mid)
            res = max(r[3] for r in mid)
            print(f"     over the {len(mid)} decisions whose best match is "
                  f"within 50 u: lag median {lags[len(lags) // 2]:+d}, "
                  f"range {lags[0]:+d}..{lags[-1]:+d}, worst residual "
                  f"{res:.3f} u")
            print("     A tight lag range with a small residual means ONE route "
                  "walked late;")
            print("     a wandering lag or a large residual means the routes "
                  "themselves differ.")

        print("\n   -- the first 25 decisions, where the lag is acquired --")
        print("     i      sim x,y                 server x,y              sep")
        for k in range(min(25, len(rows))):
            i3 = rows[k][0]
            ax3, ay3 = sim_path.pos[i3]
            bx3, by3 = srv_path.pos[i3]
            print(f"     {i3:4d}  ({ax3:9.3f},{ay3:9.3f})  "
                  f"({bx3:9.3f},{by3:9.3f})  {rows[k][1]:9.3f}")

    shared = sorted(set(sim_rec.inputs) & set(srv_rec.inputs))
    print(f"\noracle calls: sim {len(sim_rec.inputs)}, server "
          f"{len(srv_rec.inputs)}, shared indices {len(shared)}")
    if shared:
        print(f"first shared decision index {shared[0]}, last {shared[-1]}")

    hit = compare(sim_rec, srv_rec, examples=a.examples)
    print("\n== FIRST DIVERGENT ORACLE INPUT ==")
    if hit is None:
        print("   none: the oracle saw identical inputs at every shared index.")
        print("   Any CS gap is then NOT an input difference -- it is the")
        print("   engines acting differently on identical orders, which is a")
        print("   gate-1 question, not a gate-3 one.")
    else:
        print(f"   decision index {hit['i']}")
        for w in hit["why"]:
            print(f"     - {w}")
        print(f"   sim chose    {hit['sim_decision']}")
        print(f"   server chose {hit['server_decision']}")
        print(f"   sim champ    ({hit['sim']['cx']:.3f},{hit['sim']['cy']:.3f}) "
              f"ad={hit['sim']['ad']:.4f} rng={hit['sim']['rng']:.2f} "
              f"lvl={hit['sim']['level']} nvis={len(hit['sim']['minions'])}")
        print(f"   server champ ({hit['server']['cx']:.3f},{hit['server']['cy']:.3f}) "
              f"ad={hit['server']['ad']:.4f} rng={hit['server']['rng']:.2f} "
              f"lvl={hit['server']['level']} nvis={len(hit['server']['minions'])}")

    d_i = first_decision_divergence(sim_rec, srv_rec)
    print("\n== FIRST DIVERGENT CHOICE ==")
    if d_i is None:
        print("   none: both runs chose identically at every shared index.")
    else:
        print(f"   decision index {d_i}: sim {sim_rec.decisions[d_i]} vs "
              f"server {srv_rec.decisions[d_i]}")
        if hit is not None:
            print(f"   ({d_i - hit['i']} decisions after the first input "
                  f"difference -- that gap is the slack between 'the engines "
                  f"disagree' and 'it changes the episode')")

    print("\n== TOTALS (contaminated after the first divergence; for context "
          "only) ==")
    print(f"   sim    cs={sim.cs} attacks={sim.attacks} moves={sim.moves} "
          f"holds={sim.holds} deaths={sim.deaths} walks={sim.walks}")
    print(f"   server cs={srv.cs} attacks={srv.attacks} moves={srv.moves} "
          f"holds={srv.holds} deaths={srv.deaths} walks={srv.walks}")


if __name__ == "__main__":
    main()
