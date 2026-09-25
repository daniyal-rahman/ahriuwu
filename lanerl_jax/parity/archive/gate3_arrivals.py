"""The gate-3 swarm arrival process: the sim is not burstier, it is stickier.

:mod:`lanerl_jax.parity.archive.gate3_swarm` left gate 3 with one live lead, recorded
in the ledger as *clumping*::

    engaged decisions with      sim     server
      >= 1 attacker           2,830    3,530   (0.80x)
      >= 3 attackers          1,844    1,360   (1.36x)
      >= 8 attackers             85       39   (2.2x)
      max concurrent              11        8
      mean when >= 1            3.32     2.66

with the same total engaged exposure (9,395 against 9,381 attacker-decisions).
The reading offered was "same total, fewer and denser clumps", and the named
suspects were call-for-help propagation, damage-triggered aggro and target
de-prioritisation. This module measures the arrival process those numbers were
a summary of. It is analysis-only: it reads ``runs/g3_sim_swarm.npz``,
``runs/g3_srv_swarm.npz`` and the server's ``MRT`` log and re-drives nothing.

WHAT IT FOUND (2026-09-18)
--------------------------
**1. The 0.80x is normalisation, not clumping.** The sim is engaged for 11,698
decisions and the server for 14,794, because the sim is dead or walking more.
Per engaged decision, "at least one attacker" is **24.2% sim against 23.9%
server** -- the same number. Nothing about the arrival of the first attacker
differs.

**2. Neither engine's arrivals are Poisson, and the sim is the LESS bursty of
the two.** Index of dispersion of acquisition counts over engaged windows::

    window      0.5 s   1 s    2 s    5 s   10 s
    sim          2.61   2.80   3.33   3.40   4.02
    server       2.78   3.09   3.50   4.47   4.26

Both are ~3, i.e. strongly clustered, and the server is *higher* at every
window. Interval CVs are 2.87 (sim) and 3.03 (server) against 1.00 for a
Poisson process. Largest single-decision clump is 6 on both sides. **The
"sim is bursty, server is not" hypothesis is refuted by its own measurement.**

The burst mechanism, which both engines share, is ``TargetJustDied()``:
several red minions share one blue minion as a target, it dies, and every one
of them re-evaluates on that same tick -- ``LaneMinionAI.OnUpdate``'s trigger
is ``TargetJustDied() || FoundNewTarget(true) || minionActionTimer >= 250``,
and the first term fires for all of them at once. The champion is priority 11
and loses to every live minion, so they all land on it together. In the sim
this is directly visible: all six minions that killed the champion at 394.3 s
acquired on decision 11,710 off the same slot-13 blue minion, which died two
decisions earlier.

**3. What actually differs is how long a hold lasts.** Acquisition counts and
hold-length shape are close (51 holds / mean 267 decisions sim; 45 / 211
server), and hold length splits by minion type the same way on both -- casters
hold roughly twice as long as melees on both engines, because a caster sits at
its 550 u attack range. Composition of the attackers is identical
(22 melee / 28 caster / 1 cannon against 21 / 23 / 1; mean holds 152 / 362 /
122 decisions against 136 / 287 / 21) and so is the range at
which they acquire (median 450 u against 461 u, p90 678 against 674). The sim
is simply **1.26x slower to let go**, uniformly.

**4. The sim's call-for-help release rule is FAITHFUL -- the calls never
arrive.** Reconstructing ``ObjAIBase.TakeDamage``'s broadcast from the sim's
own trace (``ObjAIBase.cs:1127-1158``: an ally, not self, inside the
**victim's** ``AcquisitionRange`` of both the victim and the attacker, plus
``IsValidTarget``'s check that the attacker is inside the **listener's** own
range): of the 33 champion-holds that ever saw one strictly valid call,
**32 released within 3 decisions of the last one**. The one exception released
23 decisions later, inside the error of reconstructing the attacker from
``u_target``. The defensive direction works too: on the 147 (listener,
decision) pairs where a blue minion was an eligible responder to the champion's
own call, 134 were already on an eligible red attacker and 5 switched onto one.

What the sim lacks is **supply**. While the champion is held::

                                                        sim      server
    decisions with a red minion within 475 u
      of the champion taking damage                     3.96%     7.28%   0.54x
    mean blue minions within 475 u                       1.13      1.81
    decisions with ZERO blue minions within 475 u       1,713     1,289
      as a fraction of held decisions                    60.5%     36.5%

and, measured on the attackers rather than the champion, the distance from
each red minion within 600 u of the champion to the nearest blue minion is a
median **589 u** in the sim against **389 u** on the server, **71.8%** beyond
the 475 u melee broadcast radius against **44.7%**. A red minion that far from
the blue wave cannot hear a call for help, so nothing recalls it.

This is not downstream of the 394.3 s death: in the first engagement
(decisions 3,631-4,400, t = 121-147 s, before either engine has lost a
champion) the same split is already there -- median 522 u / 59.4% beyond 475 u
in the sim against 344 u / 24.9% on the server -- and the two engines' first
four acquisitions are on the *same four decisions* (3,631 / 3,655 / 3,679 /
3,680) before diverging on release: sim holds 164/417/116/634/626/512 against
server 178/153/432/128/99/674.

**5. The death, reconstructed.** Sim, 394.25 s, (2380, 13152), level 6::

    dec 10,764  t=358.8  blue slot 7 dies; red slots 8, 16, 25 acquire
                         together and hold 660 decisions (22.0 s) each.
                         Zero blue minions within 475 u for all 660.
    dec 11,216  t=373.8  blue slot 24 dies; red 20, 22, 26 acquire -> 6 held
    dec 11,315-11,374    five more acquire from no target -> peak 11 attackers
    dec 11,400  t=380.0  HP 279 / 1,133
    dec 11,483  t=382.8  HP 114; all holders released; HP regenerates to 176
    dec 11,708  t=390.2  blue slot 13 dies
    dec 11,710  t=390.3  red 9, 11, 14 (melee) and 17, 19, 23 (caster) -- all
                         six had slot 13 as their target -- acquire on one
                         decision and hold all 122 decisions (4.1 s) to the end
    dec 11,830  t=394.25 champion dies at 6 attackers

The closest server analogue is its own and only death, and it is the same
event with a different starting HP: at decision 15,464 (515.4 s) six red
minions acquire within one decision of each other, the champion has **518 /
1,150 (45%)**, and it dies 10.1 s later. The sim entered its clump at **176 /
1,133 (15.5%)**. Both clumps run with zero blue minions within 475 u for their
entire length (306/306 server, 122/122 sim). **The server does not survive a
six-minion clump; the sim just arrives at one with a quarter of the health,
because the preceding 22-second unreleased triple hold took 967 HP off it.**

So the chain is: the sim's red attackers stand further from the blue wave ->
no call for help reaches them -> holds run 1.26x longer -> overlapping holds
raise concurrency (>= 3 attackers on 15.8% of engaged decisions against 9.2%)
-> the champion is at 15% HP when the next ordinary clump lands. Where the
attackers stand is the open question, and it is a lane-state question, not an
aggro-rule one.

Ruled out along the way, with the source: ``cfh_Delay`` / ``cfh_Stick`` /
``cfh_Radius`` / ``cfh_Duration`` / ``cfh_MeleeRadius`` / ``cfh_RangedRadius``
in ``GameServerLib/Content/GlobalData/CallForHelpVariables.cs`` are parsed by
``GlobalData.cs:138-144`` and **read by nothing** -- ``grep -r
CallForHelpVariables`` over the whole tree returns only the declaration and the
parser, and Map1 does not even ship a ``Constants.json`` that sets them. The
reference server therefore has **no** call-for-help rate limit, cooldown or
independent radius: the only radius is the victim's ``AcquisitionRange`` in
``ObjAIBase.TakeDamage``. The sim matching that is parity, not a shortcut.

Usage::

    python -m lanerl_jax.parity.archive.gate3_arrivals \\
        --sim runs/g3_sim_swarm.npz --server runs/g3_srv_swarm.npz \\
        --server-log runs/g3_srv_swarm_log/instance000.log
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from ...sim.init import lane_params
from ...sim.state import Kind, Team
from .gate3_swarm import (SRV_MINION, SRV_BLUE, SRV_RED, _server_first_seen,
                          _server_mrt_decision, lane_fraction, server_occupancy)
from .last_hit_drive import WIRE_MINION_TYPE
from ..targets import parse_target_traces

__all__ = [
    "sim_holds", "server_holds", "occupancy", "dispersion", "release_audit",
    "call_supply", "report",
]

#: ``CharData.AcquisitionRange`` of a melee/cannon minion; also the radius of
#: the call-for-help broadcast when the victim is one, since
#: ``ObjAIBase.TakeDamage`` uses the VICTIM's range.
MELEE_ACQ = 475.0
#: Radius used for "is the blue wave next to the champion". The melee
#: acquisition range, for the same reason.
NEAR = 475.0


# ---------------------------------------------------------------------------
# loading and masks
# ---------------------------------------------------------------------------

def _load(path) -> dict:
    z = np.load(Path(path), allow_pickle=False)
    return {k: z[k] for k in z.files}


def _masks(d: dict, server: bool):
    """``(red, blue)`` live lane-minion masks, per engine's own schema."""
    if server:
        red = (d["u_kind"] == SRV_MINION) & (d["u_team"] == SRV_RED) & (d["u_hp"] > 0)
        blue = (d["u_kind"] == SRV_MINION) & (d["u_team"] == SRV_BLUE) & (d["u_hp"] > 0)
    else:
        live = (d["u_kind"] == Kind.LANE_MINION) & d["u_alive"]
        red = live & (d["u_team"] == Team.RED)
        blue = live & (d["u_team"] == Team.BLUE)
    return red, blue


def engaged(d: dict) -> np.ndarray:
    """Standing in lane and alive -- the phase gate 3's attribution isolates.

    The walking-in phase is excluded because 100% of the exposure gap there is
    the single `COLL-003` respawn walk, which is already attributed.
    """
    return (~d["approaching"]) & d["calive"]


# ---------------------------------------------------------------------------
# holds
# ---------------------------------------------------------------------------

def sim_holds(d: dict) -> list[dict]:
    """Every maximal run of ``u_target == 0`` for a live red lane minion.

    Slot 0 is the champion (``sim/init.py`` enumerates ``(BLUE, RED)``). A
    recycled slot is dead between its two lives, so a run cannot span two
    minions.
    """
    red, _ = _masks(d, server=False)
    on = red & (d["u_target"] == 0)
    n = on.shape[0]
    out = []
    for u in range(on.shape[1]):
        col = on[:, u]
        if not col.any():
            continue
        dif = np.diff(col.astype(np.int8))
        starts = list(np.flatnonzero(dif == 1) + 1)
        ends = list(np.flatnonzero(dif == -1) + 1)
        if col[0]:
            starts = [0] + starts
        if col[-1]:
            ends = ends + [n]
        for a, b in zip(starts, ends):
            out.append({"slot": int(u), "start": int(a), "end": int(b),
                        "hold": int(b - a),
                        "type": _SIM_MODEL.get(int(d["u_model"][a, u]), "?"),
                        "prev_target": int(d["u_target"][a - 1, u]) if a else -1})
    return sorted(out, key=lambda r: r["start"])


#: ``lane_params()`` model rows, red side. Even rows are blue.
_SIM_MODEL = {3: "melee", 5: "caster", 7: "cannon", 9: "super"}


def server_holds(d: dict, log_path) -> list[dict]:
    """The same list from ``MRT`` lines closed on the wire.

    Identical construction to :func:`gate3_swarm.server_occupancy` -- an
    acquisition opens at its joined decision and closes at the acting minion's
    next ``MRT`` line, or at the first decision the wire shows it dead (the
    case a transition log cannot close), or at the end of the episode.
    """
    traces = parse_target_traces(Path(log_path).read_text(
        errors="replace").splitlines())
    first = _server_first_seen(d)
    n = d["t_ms"].size
    ids, hp = d["u_id"], d["u_hp"]
    last_alive: dict = {}
    for i in range(n):
        for j in np.flatnonzero(hp[i] > 0):
            last_alive[int(ids[i, j])] = i
    by_id: dict = {}
    for m in traces.minion:
        by_id.setdefault(m.net_id, []).append(m)
    out = []
    for nid, lines in by_id.items():
        seen = first.get(nid)
        if seen is None or seen[2] != SRV_RED:
            continue
        lines = sorted(lines, key=lambda m: m.local_time_ms)
        for k, m in enumerate(lines):
            if m.to_kind != "Champion":
                continue
            i0 = _server_mrt_decision(d, first, m)
            if i0 is None:
                continue
            ends = [n]
            nxt = None
            if k + 1 < len(lines):
                nxt = _server_mrt_decision(d, first, lines[k + 1])
                if nxt is not None:
                    ends.append(nxt)
            if nid in last_alive:
                ends.append(last_alive[nid] + 1)
            i1 = max(i0 + 1, min(ends))
            if nxt is not None and i1 == nxt:
                reason = (f"switched to {lines[k + 1].to_kind} "
                          f"cfh={int(lines[k + 1].from_call_for_help)}")
            elif nid in last_alive and i1 == last_alive[nid] + 1:
                reason = "minion died"
            else:
                reason = "episode end"
            out.append({"net_id": nid, "start": i0, "end": i1,
                        "hold": int(i1 - i0),
                        "type": WIRE_MINION_TYPE.get(seen[3], "?"),
                        "cfh": bool(m.from_call_for_help),
                        "from_kind": m.from_kind, "reason": reason})
    return sorted(out, key=lambda r: r["start"])


def occupancy(d: dict, server: bool, log_path=None) -> np.ndarray:
    if server:
        return server_occupancy(d, Path(log_path))["occ"]
    red, _ = _masks(d, server=False)
    return (red & (d["u_target"] == 0)).sum(axis=1).astype(np.int32)


# ---------------------------------------------------------------------------
# the arrival process
# ---------------------------------------------------------------------------

def dispersion(starts: np.ndarray, eng: np.ndarray,
               windows=(15, 30, 60, 150, 300)) -> dict:
    """Index of dispersion of acquisition counts on an **engaged** clock.

    Time is measured in engaged decisions, so the walking and dead phases --
    where the two engines spend very different amounts of time -- cannot
    inflate either side's gaps. ``Fano = var/mean``; 1.0 is Poisson and
    anything well above it is bursty.
    """
    clock = np.cumsum(eng)
    starts = np.sort(np.asarray(starts)[eng[np.asarray(starts)]])
    gaps = np.diff(clock[starts]).astype(float)
    total = int(clock[-1])
    out = {
        "n_acquisitions_engaged": int(starts.size),
        "interval_mean": float(gaps.mean()) if gaps.size else float("nan"),
        "interval_median": float(np.median(gaps)) if gaps.size else float("nan"),
        "interval_cv": float(gaps.std() / gaps.mean()) if gaps.size else float("nan"),
        "simultaneous_pairs": int(np.sum(gaps <= 1)),
        "fano": {},
    }
    for w in windows:
        nb = total // w
        if nb <= 1:
            continue
        idx = np.clip(clock[starts] - 1, 0, total - 1)
        cnt = np.bincount(idx // w, minlength=nb)[:nb].astype(float)
        out["fano"][w] = (float(cnt.var() / max(cnt.mean(), 1e-12)),
                          int(cnt.max()), int((cnt > 0).sum()))
    return out


# ---------------------------------------------------------------------------
# the release rule, audited against the server's own broadcast condition
# ---------------------------------------------------------------------------

def release_audit(d: dict, holds: list[dict], slack: int = 3) -> dict:
    """Does the sim drop the champion the moment a valid call for help lands?

    Reimplements ``ObjAIBase.TakeDamage``'s broadcast
    (``ObjAIBase.cs:1127-1158``) from the traced state -- a red ally is damaged
    this decision, the holder is inside the **victim's** ``AcquisitionRange``
    of both that victim and its attacker -- and then ``IsValidTarget``'s extra
    condition, that the attacker is inside the **holder's own**
    ``AcquisitionRange``. The two radii genuinely differ (melee 475, caster
    700) and dropping the second one over-counts opportunities by 3x, so it is
    applied explicitly rather than folded in.

    The attacker is recovered as a live enemy whose ``u_target`` is the victim
    on that decision; a missile in flight from a unit that has since retargeted
    is therefore missed, which is the residual error in this audit.
    """
    tgt, alive = d["u_target"], d["u_alive"]
    x, y, hp = d["u_x"], d["u_y"], d["u_hp"]
    team, n_dec = d["u_team"], tgt.shape[0]
    n_unit = tgt.shape[1]
    rng = np.asarray(lane_params()["acquisition_range"])[d["u_model"]]
    red, _ = _masks(d, server=False)
    took = (hp[1:] < hp[:-1] - 1e-3) & alive[1:] & alive[:-1]
    idx = np.arange(n_unit)

    with_opp = released = held_through = 0
    misses = []
    for h in holds:
        u, a, b = h["slot"], h["start"], h["end"]
        offsets = []
        for i in range(a, min(b, n_dec - 1)):
            ok = False
            for v in np.flatnonzero(took[i] & red[i] & (idx != u)):
                r2 = rng[i, v] ** 2
                if (x[i, u] - x[i, v]) ** 2 + (y[i, u] - y[i, v]) ** 2 > r2:
                    continue
                for at in np.flatnonzero(alive[i] & (team[i] != team[i, v])
                                         & (tgt[i] == v)):
                    d2 = (x[i, u] - x[i, at]) ** 2 + (y[i, u] - y[i, at]) ** 2
                    if d2 <= r2 and d2 <= rng[i, u] ** 2:
                        ok = True
                        break
                if ok:
                    break
            if ok:
                offsets.append(i - a)
        if not offsets:
            continue
        with_opp += 1
        if offsets[-1] >= (b - a) - slack:
            released += 1
        else:
            held_through += 1
            misses.append({**h, "offsets": offsets})
    return {"holds_with_a_valid_call": with_opp,
            "released_within_%d_decisions" % slack: released,
            "held_through_the_last_call": held_through,
            "misses": misses}


def call_supply(d: dict, server: bool, occ: np.ndarray) -> dict:
    """How much call-for-help signal exists near the champion while it is held.

    Symmetric across engines: it needs only positions, HP and team, all of
    which the wire carries, so it measures the server with the same instrument
    as the sim -- unlike the release audit, which needs per-minion targets.
    """
    red, blue = _masks(d, server)
    hp = d["u_hp"]
    took = (hp[1:] < hp[:-1] - 1e-3) & red[1:] & red[:-1]
    dc = np.hypot(d["u_x"] - d["cx"][:, None], d["u_y"] - d["cy"][:, None])
    held = engaged(d) & (occ > 0)
    h1 = held[1:]
    ev = (took & (dc[1:] <= NEAR)).sum(axis=1)
    nb = (blue & (dc <= NEAR)).sum(axis=1)
    nr = (red & (dc <= NEAR)).sum(axis=1)

    # the attackers' own insulation: for each red minion near the champion,
    # the distance to the nearest blue minion -- i.e. to the nearest source of
    # a call for help. Subsampled by 3 decisions; the series is 30 Hz.
    dists = []
    for i in np.flatnonzero(held)[::3]:
        j = np.flatnonzero(blue[i])
        if not j.size:
            continue
        for u in np.flatnonzero(red[i] & (dc[i] <= 600.0)):
            dists.append(float(np.hypot(d["u_x"][i, j] - d["u_x"][i, u],
                                        d["u_y"][i, j] - d["u_y"][i, u]).min()))
    dists = np.asarray(dists)
    return {
        "held_decisions": int(held.sum()),
        "call_supply_rate": float((ev[h1] > 0).mean()) if h1.any() else 0.0,
        "mean_blue_within_475": float(nb[held].mean()),
        "mean_red_within_475": float(nr[held].mean()),
        "zero_blue_within_475": int((nb[held] == 0).sum()),
        "zero_blue_fraction": float((nb[held] == 0).mean()),
        "attacker_to_nearest_blue_median": float(np.median(dists)) if dists.size else float("nan"),
        "attacker_beyond_475": float((dists > 475).mean()) if dists.size else float("nan"),
        "n_attacker_samples": int(dists.size),
    }


def front_offset(d: dict, server: bool, occ: np.ndarray) -> dict:
    """Where the champion stands relative to its own wave, while held.

    Reported in lane units (``lane_fraction`` x the polyline length) because
    the top lane is an L and a raw x means different things on its two arms.
    """
    _, blue = _masks(d, server)
    cf = lane_fraction(d["cx"], d["cy"])
    from .gate3_swarm import _CUM
    held = np.flatnonzero(engaged(d) & (occ > 0))
    gap = []
    for i in held:
        j = np.flatnonzero(blue[i])
        if not j.size:
            continue
        bf = lane_fraction(d["u_x"][i, j], d["u_y"][i, j]).max()
        gap.append((cf[i] - bf) * _CUM[-1])
    gap = np.asarray(gap)
    return {"champ_ahead_of_blue_front_median_u": float(np.median(gap)),
            "ahead_fraction": float((gap > 0).mean()),
            "n": int(gap.size)}


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def _row(name, a, b, fmt="{:>10}"):
    r = f"{a / b:.2f}x" if isinstance(a, (int, float)) and b else "n/a"
    print(f"  {name:46s}{fmt.format(a):>12s}{fmt.format(b):>12s}{r:>9s}")


def report(sim_path, server_path, server_log) -> None:
    s, v = _load(sim_path), _load(server_path)
    so = occupancy(s, False)
    vo = occupancy(v, True, server_log)
    es, ev = engaged(s), engaged(v)

    print("\n=== 1. exposure, normalised by engaged time ===")
    _row("engaged decisions", int(es.sum()), int(ev.sum()))
    _row("attacker-decisions (engaged)", int(so[es].sum()), int(vo[ev].sum()))
    for th in (1, 2, 3, 5, 8):
        a, b = int((so[es] >= th).sum()), int((vo[ev] >= th).sum())
        _row(f">= {th} attackers", a, b)
        print(f"  {'':46s}{a / es.sum():>11.1%}{b / ev.sum():>12.1%}"
              f"{(a / es.sum()) / max(b / ev.sum(), 1e-9):>9.2f}x   per engaged decision")
    _row("max concurrent", int(so[es].max()), int(vo[ev].max()))

    print("\n=== 2. the arrival process, on an engaged clock ===")
    sh, vh = sim_holds(s), server_holds(v, server_log)
    ds = dispersion(np.array([h["start"] for h in sh]), es)
    dv = dispersion(np.array([h["start"] for h in vh]), ev)
    _row("acquisitions starting while engaged",
         ds["n_acquisitions_engaged"], dv["n_acquisitions_engaged"])
    _row("interval median (engaged decisions)",
         round(ds["interval_median"], 1), round(dv["interval_median"], 1))
    _row("interval CV (Poisson = 1.00)",
         round(ds["interval_cv"], 2), round(dv["interval_cv"], 2))
    _row("simultaneous arrivals (gap <= 1)",
         ds["simultaneous_pairs"], dv["simultaneous_pairs"])
    print(f"  {'index of dispersion (var/mean) by window':46s}")
    for w in sorted(ds["fano"]):
        print(f"    {w / 30:>4.1f} s window{'':28s}"
              f"{ds['fano'][w][0]:>12.2f}{dv['fano'][w][0]:>12.2f}"
              f"   max/window {ds['fano'][w][1]} vs {dv['fano'][w][1]}")
    print("  -> both ~3x Poisson; the server is the burstier of the two.")

    print("\n=== 3. hold length, the thing that actually differs ===")
    _row("holds", len(sh), len(vh))
    _row("mean hold (decisions)",
         round(sum(h["hold"] for h in sh) / len(sh), 1),
         round(sum(h["hold"] for h in vh) / len(vh), 1))
    _row("median hold (decisions)",
         int(np.median([h["hold"] for h in sh])),
         int(np.median([h["hold"] for h in vh])))
    for ty in ("melee", "caster", "cannon"):
        a = [h["hold"] for h in sh if h["type"] == ty]
        b = [h["hold"] for h in vh if h["type"] == ty]
        if a and b:
            _row(f"  {ty}: n / mean hold",
                 f"{len(a)}/{int(np.mean(a))}", f"{len(b)}/{int(np.mean(b))}")
    print("  sim acquisition prev-target:",
          dict(Counter("none" if h["prev_target"] < 0 else "had one"
                       for h in sh)))
    print("  server MRT `from`:", dict(Counter(h["from_kind"] for h in vh)),
          "cfh=1:", sum(1 for h in vh if h["cfh"]))

    print("\n=== 4. the sim's release rule against the server's broadcast ===")
    ra = release_audit(s, sh)
    for k, val in ra.items():
        if k != "misses":
            print(f"  {k:46s}{val:>12}")
    for m in ra["misses"]:
        print(f"    miss: slot {m['slot']} acq {m['start']} hold {m['hold']} "
              f"offsets {m['offsets'][:6]}")

    print("\n=== 5. call-for-help SUPPLY while the champion is held ===")
    cs, cv = call_supply(s, False, so), call_supply(v, True, vo)
    for k in cs:
        a, b = cs[k], cv[k]
        if isinstance(a, float):
            print(f"  {k:46s}{a:>12.3f}{b:>12.3f}")
        else:
            _row(k, a, b)
    fs, fv = front_offset(s, False, so), front_offset(v, True, vo)
    print(f"  {'champ ahead of own blue front (u, median)':46s}"
          f"{fs['champ_ahead_of_blue_front_median_u']:>12.0f}"
          f"{fv['champ_ahead_of_blue_front_median_u']:>12.0f}")
    print(f"  {'...ahead at all, fraction of held decisions':46s}"
          f"{fs['ahead_fraction']:>12.1%}{fv['ahead_fraction']:>12.1%}")

    print("\n=== 6. the death ===")
    sd = np.flatnonzero(s["calive"][:-1] & ~s["calive"][1:])
    vd = np.flatnonzero(v["calive"][:-1] & ~v["calive"][1:])
    print(f"  sim deaths at   {[round(float(s['t_ms'][i] / 1000), 1) for i in sd]} s")
    print(f"  server death at {[round(float(v['t_ms'][i] / 1000), 1) for i in vd]} s")
    for nm, d_, dd, occ in (("sim", s, sd, so), ("server", v, vd, vo)):
        for i in dd:
            j = max(0, i - 130)
            print(f"  {nm}: death dec {i} t={d_['t_ms'][i] / 1000:.1f}s "
                  f"hp 130 decisions earlier {d_['chp'][j]:.0f}/{d_['cmhp'][j]:.0f} "
                  f"({d_['chp'][j] / d_['cmhp'][j]:.0%}), attackers then {occ[j]}, "
                  f"at death {occ[i]}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sim", default="runs/g3_sim_swarm.npz")
    p.add_argument("--server", default="runs/g3_srv_swarm.npz")
    p.add_argument("--server-log", default="runs/g3_srv_swarm_log/instance000.log")
    a = p.parse_args(argv)
    report(a.sim, a.server, a.server_log)


if __name__ == "__main__":
    main()
