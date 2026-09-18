"""Why gate 3's sim champion is swarmed by red minions and the server's is not.

:mod:`lanerl_jax.parity.gate3_attribution` reduced gate 3's whole deficit to
one unexplained fact::

                                        sim     server
    total damage taken                3,840     2,758   (+39%)
    decisions with >=3 red <200u      1,926     1,290   (+49%)

and named the measurement that would separate its two candidate causes:

    At each red minion's moment of acquiring the champion as its target,
    record the number of live **allied blue** minions inside that minion's own
    acquisition range.

That census separates *"the red minion had no valid alternative target"* --
the champion is priority 11 (``ClassifyUnit.CHAMPION``) and loses to every
live minion (7/8/9), so a red minion can only pick it when nothing better is
in range -- from *"the champion walked into them"*.

This module computes it on both engines, plus the three secondary hypotheses
gate-3 attribution left open (blue-wave attrition, where the waves clash, and
whether aggro is ever released). It **reuses the gate's own tracers** --
``gate3_attribution``'s ``SimTracer``/``ServerTracer`` npz files, produced by
``on_decision`` callbacks on the real drivers -- rather than re-driving the
loop, which is the PATH-006 failure mode.

Observability is asymmetric and that asymmetry drives the design:

* **Sim**: exact. ``LaneState.target`` is traced per decision for every unit,
  so an acquisition is a ``target`` transition onto slot 0 and the census is a
  direct distance query on the same decision's positions.
* **Server**: the wire carries no per-minion ``TargetUnit`` at all. The only
  server-side view of minion targeting is ``LANERL_AGGRO_TRACE``'s ``MRT``
  lines (``LaneMinionAI.cs:243``), which are keyed by the acting minion's
  NetId and its own ``localTime`` -- a per-minion clock that starts at that
  minion's activation, not at a shared game clock. This module reconstructs
  game time as ``first-seen-on-the-wire + lt`` and then does the census on the
  wire snapshot at that decision. That join is **validated, not assumed**:
  every reconstructed acquisition must have the champion inside the acting
  minion's own acquisition range at the joined decision -- which is exactly
  what the server just asserted by acquiring it -- and the failure rate of
  that check is reported next to every number the join produces.

Usage (after the two ``gate3_attribution`` collections, the server one with
``--aggro-trace``)::

    python -m lanerl_jax.parity.gate3_swarm \
        --sim runs/g3_sim_swarm.npz --server runs/g3_srv_swarm.npz \
        --server-log runs/g3_srv_swarm_log/instance000.log
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..sim.init import TOP_LANE_PATH, lane_params
from ..sim.state import Kind, Team
from .last_hit_drive import WIRE_MINION_TYPE
from .targets import parse_target_traces

__all__ = ["sim_acquisitions", "server_acquisitions", "report"]

#: Wire team ids.
SRV_BLUE, SRV_RED = 100, 200
#: ``KIND_CODE`` from :mod:`gate3_attribution`.
SRV_MINION = 2

#: ``CharData.AcquisitionRange`` per wire ``MinionSpawnType``. 475 is
#: ``CharData.cs:20``'s default (melee/cannon carry no override); the caster's
#: 700 is ``Red_Minion_Wizard.json``'s explicit ``"AcquisitionRange": "700"``.
#: Identical to the sim's ``lane_params()["acquisition_range"]`` rows, which is
#: why acquisition RADIUS is not a candidate for the gap.
_MT_ACQ = {"melee": 475.0, "caster": 700.0, "cannon": 475.0, "super": 600.0}

#: The swarm threshold ``gate3_attribution`` reports on.
SWARM_RADIUS = 200.0


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def _load(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def _pct(a, b) -> str:
    return f"{100.0 * a / b:.1f}%" if b else "n/a"


def _stats(v) -> str:
    v = np.asarray(v, float)
    if v.size == 0:
        return "(none)"
    return (f"n={v.size} mean={v.mean():.2f} median={np.median(v):.1f} "
            f"p10={np.percentile(v, 10):.1f} p90={np.percentile(v, 90):.1f} "
            f"min={v.min():.1f} max={v.max():.1f}")


# ---------------------------------------------------------------------------
# lane geometry, for "where do the waves clash"
# ---------------------------------------------------------------------------

_LANE = np.asarray(TOP_LANE_PATH, np.float64)
_SEG = np.diff(_LANE, axis=0)
_SEG_LEN = np.hypot(_SEG[:, 0], _SEG[:, 1])
_CUM = np.concatenate([[0.0], np.cumsum(_SEG_LEN)])


def lane_fraction(x, y) -> np.ndarray:
    """Project onto ``TOP_LANE_PATH`` and return arc length / total, in [0, 1].

    0 is blue's end of the top lane, 1 is red's. Reported instead of raw
    coordinates because the top lane is an L and a raw x or y means different
    things on its two arms.
    """
    x = np.asarray(x, np.float64).ravel()
    y = np.asarray(y, np.float64).ravel()
    px = x[:, None] - _LANE[None, :-1, 0]
    py = y[:, None] - _LANE[None, :-1, 1]
    t = np.clip((px * _SEG[None, :, 0] + py * _SEG[None, :, 1])
                / np.maximum(_SEG_LEN[None, :] ** 2, 1e-9), 0.0, 1.0)
    cx = _LANE[None, :-1, 0] + t * _SEG[None, :, 0]
    cy = _LANE[None, :-1, 1] + t * _SEG[None, :, 1]
    d2 = (x[:, None] - cx) ** 2 + (y[:, None] - cy) ** 2
    j = np.argmin(d2, axis=1)
    i = np.arange(x.size)
    return (_CUM[j] + t[i, j] * _SEG_LEN[j]) / _CUM[-1]


# ---------------------------------------------------------------------------
# sim side: exact
# ---------------------------------------------------------------------------

def _sim_masks(d: dict):
    kind = d["u_kind"]
    team = d["u_team"]
    alive = d["u_alive"]
    red = (kind == Kind.LANE_MINION) & (team == Team.RED) & alive
    blue = (kind == Kind.LANE_MINION) & (team == Team.BLUE) & alive
    return red, blue


def sim_acquisitions(d: dict) -> dict:
    """Every decision at which a live red minion's target BECAME the champion.

    The champion is slot 0 (``sim/init.py``'s ``for i, t in enumerate((BLUE,
    RED))``), so ``target == 0`` is "attacking the oracle-driven champion".

    A slot is recycled when a minion respawns, so an event requires the slot to
    be a live red lane minion on BOTH the previous and the current decision;
    a recycled slot is dead in between and cannot produce a false positive.
    """
    tgt = d["u_target"]
    red, blue = _sim_masks(d)
    acq_by_model = np.asarray(lane_params()["acquisition_range"])
    rng = acq_by_model[d["u_model"]]                     # (dec, N)

    ii, uu = np.nonzero(red[1:] & red[:-1] & (tgt[1:] == 0) & (tgt[:-1] != 0))
    ii = ii + 1
    rows = []
    for i, u in zip(ii, uu):
        mx, my = float(d["u_x"][i, u]), float(d["u_y"][i, u])
        r = float(rng[i, u])
        db = np.hypot(d["u_x"][i] - mx, d["u_y"][i] - my)
        n_blue = int(np.sum(blue[i] & (db <= r)))
        n_blue600 = int(np.sum(blue[i] & (db <= 600.0)))
        dchamp = float(np.hypot(d["cx"][i] - mx, d["cy"][i] - my))
        # how long it then held the champion, in decisions
        hold = 0
        j = i
        while j < tgt.shape[0] and red[j, u] and tgt[j, u] == 0:
            hold += 1
            j += 1
        rows.append({
            "decision": int(i), "t_s": float(d["t_ms"][i] / 1000.0),
            "slot": int(u), "acq_range": r,
            "n_blue_in_range": n_blue, "n_blue_600": n_blue600,
            "dist_champ": dchamp,
            "n_blue_alive_total": int(np.sum(blue[i])),
            "n_red_alive_total": int(np.sum(red[i])),
            "prev_target": int(tgt[i - 1, u]),
            "hold_decisions": hold,
            "released": bool(j < tgt.shape[0] and red[j, u]),
        })
    return {"events": rows, "n_joined": len(rows), "n_total": len(rows),
            "join_ok": len(rows)}


def sim_exposure(d: dict) -> dict:
    """Continuous view: how much of the time red minions sit on the champion."""
    tgt = d["u_target"]
    red, blue = _sim_masks(d)
    on_champ = red & (tgt == 0)
    per_dec = on_champ.sum(axis=1)
    acq_by_model = np.asarray(lane_params()["acquisition_range"])
    rng = acq_by_model[d["u_model"]]
    # for every live red minion with the champion inside ITS acquisition
    # range, did it have any blue minion alternative in that same range?
    dchamp = np.hypot(d["u_x"] - d["cx"][:, None], d["u_y"] - d["cy"][:, None])
    can_see_champ = red & (dchamp <= rng) & d["calive"][:, None]
    n_opp = 0
    n_opp_no_alt = 0
    for i in np.flatnonzero(can_see_champ.any(axis=1)):
        for u in np.flatnonzero(can_see_champ[i]):
            db = np.hypot(d["u_x"][i] - d["u_x"][i, u],
                          d["u_y"][i] - d["u_y"][i, u])
            n_opp += 1
            if not np.any(blue[i] & (db <= rng[i, u])):
                n_opp_no_alt += 1
    return {
        "decisions_with_any_attacker": int(np.sum(per_dec > 0)),
        "mean_attackers_when_active": float(per_dec[per_dec > 0].mean())
        if np.any(per_dec > 0) else 0.0,
        "max_attackers": int(per_dec.max()),
        "unit_decisions_on_champ": int(per_dec.sum()),
        "red_in_range_of_champ_unit_decisions": n_opp,
        "...of which no blue alternative in range": n_opp_no_alt,
    }


# ---------------------------------------------------------------------------
# server side: MRT trace joined to the wire census
# ---------------------------------------------------------------------------

def _server_first_seen(d: dict) -> dict:
    """NetId -> (first decision index it appears on the wire, its wire type)."""
    first = {}
    ids = d["u_id"]
    kinds = d["u_kind"]
    mts = d["u_mt"] if "u_mt" in d else None
    teams = d["u_team"]
    for i in range(ids.shape[0]):
        for j in np.flatnonzero(kinds[i] > 0):
            nid = int(ids[i, j])
            if nid not in first:
                first[nid] = (i, int(kinds[i, j]), int(teams[i, j]),
                              int(mts[i, j]) if mts is not None else 255)
    return first


def server_acquisitions(d: dict, log_path: Path) -> dict:
    """Reconstruct the same census from ``MRT`` lines plus the wire snapshots.

    ``MRT`` gives ``id`` (the acting minion's NetId) and ``lt`` (its own
    ``localTime``, ms since its script activated). The wire gives every unit's
    position at every decision. Game time of the switch is therefore
    ``t_ms[first decision this NetId appears] + lt``; the census is done on the
    nearest decision to that.

    The join is checked, per event, against a fact the server itself just
    asserted: it acquired the champion, so the champion must be inside the
    acting minion's acquisition range at that moment. ``join_ok`` counts the
    events that pass; the rest are reported and excluded rather than kept.
    """
    traces = parse_target_traces(Path(log_path).read_text(
        errors="replace").splitlines())
    first = _server_first_seen(d)
    t_ms = d["t_ms"]
    ids = d["u_id"]
    kinds = d["u_kind"]
    teams = d["u_team"]
    hp = d["u_hp"]
    mts = d["u_mt"]

    # id -> column index per decision, built lazily per decision we touch
    rows = []
    skipped_unknown = 0
    skipped_absent = 0
    failed_range = []
    to_champ = [m for m in traces.minion if m.to_kind == "Champion"]
    for m in to_champ:
        seen = first.get(m.net_id)
        if seen is None:
            skipped_unknown += 1
            continue
        i0, _kind, team, mt = seen
        if team != SRV_RED:
            continue
        est = t_ms[i0] + m.local_time_ms
        i = int(np.clip(np.searchsorted(t_ms, est), 0, t_ms.size - 1))
        if i > 0 and abs(t_ms[i - 1] - est) < abs(t_ms[i] - est):
            i -= 1
        col = np.flatnonzero(ids[i] == m.net_id)
        if col.size == 0 or hp[i, col[0]] <= 0:
            skipped_absent += 1
            continue
        u = int(col[0])
        key = WIRE_MINION_TYPE.get(mt, "melee")
        r = _MT_ACQ[key]
        mx, my = float(d["u_x"][i, u]), float(d["u_y"][i, u])
        dchamp = float(np.hypot(d["cx"][i] - mx, d["cy"][i] - my))
        live_blue = (kinds[i] == SRV_MINION) & (teams[i] == SRV_BLUE) & (hp[i] > 0)
        db = np.hypot(d["u_x"][i] - mx, d["u_y"][i] - my)
        live_red = (kinds[i] == SRV_MINION) & (teams[i] == SRV_RED) & (hp[i] > 0)
        rec = {
            "decision": i, "t_s": float(t_ms[i] / 1000.0),
            "net_id": m.net_id, "acq_range": r,
            "n_blue_in_range": int(np.sum(live_blue & (db <= r))),
            "n_blue_600": int(np.sum(live_blue & (db <= 600.0))),
            "dist_champ": dchamp,
            "n_blue_alive_total": int(np.sum(live_blue)),
            "n_red_alive_total": int(np.sum(live_red)),
            "from_kind": m.from_kind, "from_priority": m.from_priority,
            "to_priority": m.to_priority, "cfh": bool(m.from_call_for_help),
            "held_ms": m.held_ms,
        }
        # the server just acquired the champion, so it MUST be in range
        if dchamp > r + 1.0:
            failed_range.append(rec)
            continue
        rows.append(rec)
    return {
        "events": rows,
        "n_mrt_to_champion": len(to_champ),
        "n_joined": len(rows),
        "skipped_unknown_netid": skipped_unknown,
        "skipped_absent_on_wire": skipped_absent,
        "failed_range_check": len(failed_range),
        "failed_examples": failed_range[:5],
        "n_mrt_total": len(traces.minion),
        "n_mrt_from_champion": sum(
            1 for m in traces.minion if m.from_kind == "Champion"),
        "releases": [m for m in traces.minion
                     if m.from_kind == "Champion"],
    }


# ---------------------------------------------------------------------------
# concurrency: the number docs/TARGET_ACQUISITION_DIFF.md called unmeasurable
# ---------------------------------------------------------------------------

def _server_mrt_decision(d: dict, first: dict, m) -> int | None:
    """Map one MRT line to the decision index it happened on."""
    seen = first.get(m.net_id)
    if seen is None:
        return None
    t_ms = d["t_ms"]
    est = t_ms[seen[0]] + m.local_time_ms
    i = int(np.clip(np.searchsorted(t_ms, est), 0, t_ms.size - 1))
    if i > 0 and abs(t_ms[i - 1] - est) < abs(t_ms[i] - est):
        i -= 1
    return i


def server_occupancy(d: dict, log_path: Path) -> dict:
    """Per-decision count of red minions holding the champion, on the SERVER.

    ``docs/TARGET_ACQUISITION_DIFF.md`` recorded this as not reconstructable
    ("`LANERL_AGGRO_TRACE` logs transitions, not a per-tick target census") and
    fell back to a 174 s .. ~292 s bound on total exposure. It IS
    reconstructable, because the trace's transitions can be closed on the
    wire rather than on the trace:

    * an acquisition of the champion opens an interval at its joined decision;
    * it closes at the next ``MRT`` line for the SAME NetId (the minion picked
      something else), or -- the case the old bound could not close -- at the
      first decision the wire shows that NetId dead or gone, which is exactly
      the "the attacking minion itself died mid-hold, so it never printed a
      departure line" case, or at the end of the episode.

    ``LaneMinionAI.OnUpdate`` only runs while ``!IsDead``, so a dead minion's
    target is inert: closing on the wire's own death is the same event the
    server sees.
    """
    traces = parse_target_traces(Path(log_path).read_text(
        errors="replace").splitlines())
    first = _server_first_seen(d)
    n = d["t_ms"].size
    by_id: dict = {}
    for m in traces.minion:
        by_id.setdefault(m.net_id, []).append(m)
    # wire lifetime per NetId: last decision on which it is alive
    ids, hp = d["u_id"], d["u_hp"]
    last_alive: dict = {}
    for i in range(n):
        for j in np.flatnonzero(hp[i] > 0):
            last_alive[int(ids[i, j])] = i

    occ = np.zeros(n, np.int32)
    spans = []
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
            if k + 1 < len(lines):
                nxt = _server_mrt_decision(d, first, lines[k + 1])
                if nxt is not None:
                    ends.append(nxt)
            if nid in last_alive:
                ends.append(last_alive[nid] + 1)
            i1 = max(i0 + 1, min(ends))
            occ[i0:i1] += 1
            spans.append((i0, i1, i1 - i0))
    return {"occ": occ, "spans": spans}


def opportunity(d: dict, server: bool) -> dict:
    """Pure geometry, identical on both engines: how often is the champion the
    ONLY thing a red minion can see?

    For every live red minion on every decision, ask two questions about that
    minion's own ``AcquisitionRange`` circle: is the (live) champion inside
    it, and is any live blue minion inside it. ``ClassifyTarget`` puts a
    champion at priority 11 and every minion at 7/8/9, so the second question
    is exactly "does this minion have a better option". No target field is
    needed, so this measures the two engines with the SAME instrument -- it is
    the cause-side counterpart of the acquisition census, which can only be
    read through each engine's own (different) observability.
    """
    if server:
        red = ((d["u_kind"] == SRV_MINION) & (d["u_team"] == SRV_RED)
               & (d["u_hp"] > 0))
        blue = ((d["u_kind"] == SRV_MINION) & (d["u_team"] == SRV_BLUE)
                & (d["u_hp"] > 0))
        key = {v_: k for k, v_ in WIRE_MINION_TYPE.items()}
        rng = np.full(d["u_mt"].shape, 475.0)
        for name, code in key.items():
            rng[d["u_mt"] == code] = _MT_ACQ[name]
    else:
        red, blue = _sim_masks(d)
        rng = np.asarray(lane_params()["acquisition_range"])[d["u_model"]]
    champ_alive = d["calive"]
    dchamp = np.hypot(d["u_x"] - d["cx"][:, None], d["u_y"] - d["cy"][:, None])
    sees_champ = red & (dchamp <= rng) & champ_alive[:, None]
    n_live = int(red.sum())
    n_sees = 0
    n_alone = 0
    for i in np.flatnonzero(sees_champ.any(axis=1)):
        bj = np.flatnonzero(blue[i])
        bx, by = d["u_x"][i, bj], d["u_y"][i, bj]
        for u in np.flatnonzero(sees_champ[i]):
            n_sees += 1
            if bj.size == 0 or not np.any(
                    np.hypot(bx - d["u_x"][i, u], by - d["u_y"][i, u])
                    <= rng[i, u]):
                n_alone += 1
    return {"red_minion_decisions": n_live,
            "champion_in_range": n_sees,
            "champion_in_range_and_no_blue_alternative": n_alone}


def sim_release_causes(d: dict) -> dict:
    """What ends a sim minion's hold on the champion.

    The server's answer is unambiguous and one-sided: all 28 observed
    departures are ``cfh=1`` call-for-help switches straight back to a
    ``LaneMinion``. The sim has no per-event cause in the trace, so it is
    reconstructed from the state at the decision the hold ends.
    """
    tgt = d["u_target"]
    red, _ = _sim_masks(d)
    rng = np.asarray(lane_params()["acquisition_range"])[d["u_model"]]
    n = tgt.shape[0]
    ii, uu = np.nonzero(red[1:] & red[:-1] & (tgt[1:] == 0) & (tgt[:-1] != 0))
    causes = {"minion died": 0, "champion died": 0, "champion left range": 0,
              "switched while champion valid and in range": 0,
              "still holding at episode end": 0}
    for i, u in zip(ii + 1, uu):
        j = i
        while j < n and red[j, u] and tgt[j, u] == 0:
            j += 1
        if j >= n:
            causes["still holding at episode end"] += 1
        elif not red[j, u]:
            causes["minion died"] += 1
        elif not d["calive"][j]:
            causes["champion died"] += 1
        elif np.hypot(d["cx"][j] - d["u_x"][j, u],
                      d["cy"][j] - d["u_y"][j, u]) > rng[j, u]:
            causes["champion left range"] += 1
        else:
            causes["switched while champion valid and in range"] += 1
    return causes


def concurrency_table(s: dict, v: dict, server_log: Path) -> None:
    """Attacker concurrency and the proximity/aggro split, side by side.

    ``gate3_attribution`` reports "decisions with >=3 red minions inside 200
    units" -- a PROXIMITY count, which a wave that has pushed past the
    champion inflates without anyone aggroing it. Splitting it against the
    count that are actually targeting the champion says which of the two the
    damage gap is.
    """
    s_red, _ = _sim_masks(s)
    s_on = s_red & (s["u_target"] == 0)
    s_occ = s_on.sum(axis=1)
    v_occ = server_occupancy(v, server_log)["occ"]

    sd = np.hypot(s["u_x"] - s["cx"][:, None], s["u_y"] - s["cy"][:, None])
    s_near = (s_red & (sd <= SWARM_RADIUS)).sum(axis=1)
    v_red = ((v["u_kind"] == SRV_MINION) & (v["u_team"] == SRV_RED)
             & (v["u_hp"] > 0))
    vd = np.hypot(v["u_x"] - v["cx"][:, None], v["u_y"] - v["cy"][:, None])
    v_near = (v_red & (vd <= SWARM_RADIUS)).sum(axis=1)

    s_alive, v_alive = s["calive"], v["calive"]
    rows = [
        ("red minions within 200u: >=1", int(np.sum(s_near >= 1)),
         int(np.sum(v_near >= 1))),
        ("red minions within 200u: >=3", int(np.sum(s_near >= 3)),
         int(np.sum(v_near >= 3))),
        ("...restricted to ALIVE champion", int(np.sum((s_near >= 3) & s_alive)),
         int(np.sum((v_near >= 3) & v_alive))),
        ("unit-decisions red within 200u", int(s_near.sum()), int(v_near.sum())),
        ("", 0, 0),
        ("red minions TARGETING champ: >=1", int(np.sum(s_occ >= 1)),
         int(np.sum(v_occ >= 1))),
        ("red minions TARGETING champ: >=3", int(np.sum(s_occ >= 3)),
         int(np.sum(v_occ >= 3))),
        ("max simultaneous attackers", int(s_occ.max()), int(v_occ.max())),
        ("unit-decisions on champion", int(s_occ.sum()), int(v_occ.sum())),
        ("...as seconds of exposure", int(s_occ.sum() / 30),
         int(v_occ.sum() / 30)),
    ]
    print(f"  {'':40s}{'sim':>10s}{'server':>10s}{'ratio':>9s}")
    for name, a, b in rows:
        if not name:
            print()
            continue
        r = f"{a / b:.2f}x" if b else "n/a"
        print(f"  {name:40s}{a:>10d}{b:>10d}{r:>9s}")
    print(f"  mean attackers when >=1 targeting      "
          f"{s_occ[s_occ > 0].mean():>10.2f}{v_occ[v_occ > 0].mean():>10.2f}")
    # damage per unit-decision of exposure
    for nm, d_, occ in (("sim", s, s_occ), ("server", v, v_occ)):
        hp = d_["chp"].astype(float)
        al = d_["calive"]
        dmg = -np.diff(hp)
        dmg = dmg[(dmg > 0) & al[1:] & al[:-1]]
        print(f"  {nm:7s}damage {dmg.sum():8.0f} over {int(occ.sum()):6d} "
              f"attacker-decisions = {dmg.sum() / max(occ.sum(), 1):.3f} "
              "per attacker-decision")


# ---------------------------------------------------------------------------
# hypotheses 2 and 3: wave attrition and where the waves clash
# ---------------------------------------------------------------------------

def wave_timeline(d: dict, server: bool) -> dict:
    if server:
        k, tm, hp = d["u_kind"], d["u_team"], d["u_hp"]
        blue = (k == SRV_MINION) & (tm == SRV_BLUE) & (hp > 0)
        red = (k == SRV_MINION) & (tm == SRV_RED) & (hp > 0)
    else:
        red, blue = _sim_masks(d)
    nb = blue.sum(axis=1)
    nr = red.sum(axis=1)
    # frontmost minion of each side, as a lane fraction
    fx = np.full(nb.size, np.nan)
    rx = np.full(nb.size, np.nan)
    for i in range(0, nb.size, 30):          # 1 Hz is plenty for a push curve
        if nb[i]:
            j = np.flatnonzero(blue[i])
            fx[i] = lane_fraction(d["u_x"][i, j], d["u_y"][i, j]).max()
        if nr[i]:
            j = np.flatnonzero(red[i])
            rx[i] = lane_fraction(d["u_x"][i, j], d["u_y"][i, j]).min()
    return {"n_blue": nb, "n_red": nr, "blue_front": fx, "red_front": rx,
            "t_s": d["t_ms"] / 1000.0}


def _minion_deaths(d: dict, server: bool) -> dict:
    """Deaths per side. Sim by slot transition, server by NetId disappearance."""
    if not server:
        alive, kind, team = d["u_alive"], d["u_kind"], d["u_team"]
        died = alive[:-1] & ~alive[1:] & (kind[:-1] == Kind.LANE_MINION)
        return {"blue": int(np.sum(died & (team[:-1] == Team.BLUE))),
                "red": int(np.sum(died & (team[:-1] == Team.RED)))}
    out = {"blue": 0, "red": 0}
    prev: dict = {}
    for i in range(d["u_id"].shape[0]):
        cur = {}
        for j in np.flatnonzero(d["u_kind"][i] == SRV_MINION):
            cur[int(d["u_id"][i, j])] = (float(d["u_hp"][i, j]),
                                         int(d["u_team"][i, j]))
        for mid, (php, ptm) in prev.items():
            if php <= 0:
                continue
            nxt = cur.get(mid)
            if nxt is None or nxt[0] <= 0:
                out["blue" if ptm == SRV_BLUE else "red"] += 1
        prev = cur
    return out


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def report(sim_path: Path, server_path: Path, server_log: Path) -> None:
    s = _load(sim_path)
    v = _load(server_path)
    ssum = json.loads(str(s["summary"]))
    vsum = json.loads(str(v["summary"]))
    print("=" * 78)
    print("GATE 3: WHY THE SIM CHAMPION IS SWARMED")
    print("=" * 78)
    print(f"sim    : {ssum}")
    print(f"server : {vsum}")

    sa = sim_acquisitions(s)
    va = server_acquisitions(v, Path(server_log))

    print("\n--- JOIN QUALITY (server side only; the sim side is exact) -------")
    print(f"  MRT lines total                : {va['n_mrt_total']}")
    print(f"  MRT ... to=Champion            : {va['n_mrt_to_champion']}")
    print(f"  joined to a wire decision      : {va['n_joined']}")
    print(f"  dropped, NetId never on wire   : {va['skipped_unknown_netid']}")
    print(f"  dropped, not on wire at t      : {va['skipped_absent_on_wire']}")
    print(f"  dropped, champion out of range : {va['failed_range_check']}")
    for e in va["failed_examples"]:
        print(f"      t={e['t_s']:.1f}s dist_champ={e['dist_champ']:.0f} "
              f"acq={e['acq_range']:.0f}")

    print("\n--- THE ACQUISITION CENSUS ---------------------------------------")
    print("  live ALLIED BLUE minions inside the acquiring red minion's own")
    print("  acquisition range, at the decision it acquired the champion")
    for name, ev in (("sim", sa["events"]), ("server", va["events"])):
        n = len(ev)
        if not n:
            print(f"\n{name}: no events")
            continue
        blue_in = np.array([e["n_blue_in_range"] for e in ev])
        blue600 = np.array([e["n_blue_600"] for e in ev])
        tot = np.array([e["n_blue_alive_total"] for e in ev])
        rtot = np.array([e["n_red_alive_total"] for e in ev])
        dch = np.array([e["dist_champ"] for e in ev])
        print(f"\n{name}: {n} acquisitions of the champion by a red minion")
        print(f"  blue minions in acq range   : {_stats(blue_in)}")
        print(f"    == 0 (no alternative)     : {int(np.sum(blue_in == 0))}"
              f"/{n} ({_pct(int(np.sum(blue_in == 0)), n)})")
        print(f"    >= 1 (had an alternative) : {int(np.sum(blue_in >= 1))}"
              f"/{n} ({_pct(int(np.sum(blue_in >= 1)), n)})")
        print(f"  blue minions within 600u    : {_stats(blue600)}")
        print(f"  blue minions ALIVE anywhere : {_stats(tot)}")
        print(f"    == 0 (whole wave dead)    : {int(np.sum(tot == 0))}"
              f"/{n} ({_pct(int(np.sum(tot == 0)), n)})")
        print(f"  red  minions ALIVE anywhere : {_stats(rtot)}")
        print(f"  distance to champion        : {_stats(dch)}")
    if sa["events"] and va["events"]:
        sb = np.array([e["n_blue_in_range"] for e in sa["events"]])
        vb = np.array([e["n_blue_in_range"] for e in va["events"]])
        print(f"\n  ratio of mean blue-in-range (sim/server): "
              f"{sb.mean() / vb.mean():.2f}" if vb.mean() else "")

    print("\n--- HOW THE SWITCH WAS MADE (server MRT fields) -------------------")
    if va["events"]:
        from collections import Counter
        fp = Counter(e["from_priority"] for e in va["events"])
        tp = Counter(e["to_priority"] for e in va["events"])
        cf = Counter(e["cfh"] for e in va["events"])
        print(f"  fromprio: {dict(sorted(fp.items()))}  (14 = had no target)")
        print(f"  toprio  : {dict(sorted(tp.items()))}  (11 = plain CHAMPION, "
              "5 = call-for-help CHAMPION_ATTACKING_MINION)")
        print(f"  cfh     : {dict(cf)}")
    if sa["events"]:
        from collections import Counter
        pv = Counter(e["prev_target"] for e in sa["events"])
        n_none = sum(c for t, c in pv.items() if t < 0)
        print(f"  sim previous target: none(-1)={n_none}, "
              f"a unit={len(sa['events']) - n_none}")

    print("\n--- HYPOTHESIS 4: IS AGGRO EVER RELEASED? ------------------------")
    srel = [e for e in sa["events"] if e["released"]]
    shold = np.array([e["hold_decisions"] for e in sa["events"]], float)
    print(f"  sim   : {len(srel)}/{len(sa['events'])} acquisitions end with the "
          "minion still alive and on a different target")
    print(f"          hold length (decisions): {_stats(shold)} "
          f"-> seconds: {_stats(shold / 30.0)}")
    rel = va["releases"]
    print(f"  server: {va['n_mrt_from_champion']} MRT lines with "
          "from=Champion (a minion leaving the champion)")
    if rel:
        held = np.array([m.held_ms for m in rel], float)
        from collections import Counter
        print(f"          held before leaving (ms): {_stats(held)}")
        print(f"          to_kind: {dict(Counter(m.to_kind for m in rel))}")
        print(f"          cfh    : {dict(Counter(m.from_call_for_help for m in rel))}")

    print("\n--- CONCURRENCY AND THE PROXIMITY/AGGRO SPLIT ---------------------")
    concurrency_table(s, v, Path(server_log))

    print("\n--- EXPOSURE (sim, exact) ----------------------------------------")
    for k, val in sim_exposure(s).items():
        print(f"  {k:48s}{val}")

    print("\n--- OPPORTUNITY (same instrument on both engines) -----------------")
    so = opportunity(s, server=False)
    vo = opportunity(v, server=True)
    print(f"  {'':52s}{'sim':>10s}{'server':>10s}{'ratio':>9s}")
    for k in so:
        r = f"{so[k] / vo[k]:.2f}x" if vo[k] else "n/a"
        print(f"  {k:52s}{so[k]:>10d}{vo[k]:>10d}{r:>9s}")
    print("\n--- WHAT ENDS A SIM HOLD -----------------------------------------")
    for k, val in sim_release_causes(s).items():
        print(f"  {k:52s}{val:>10d}")

    print("\n--- HYPOTHESIS 2: BLUE WAVE ATTRITION ----------------------------")
    st = wave_timeline(s, server=False)
    vt = wave_timeline(v, server=True)
    sd = _minion_deaths(s, server=False)
    vd = _minion_deaths(v, server=True)
    print(f"  {'':28s}{'sim':>12s}{'server':>12s}")
    print(f"  {'blue minion deaths':28s}{sd['blue']:>12d}{vd['blue']:>12d}")
    print(f"  {'red  minion deaths':28s}{sd['red']:>12d}{vd['red']:>12d}")
    print(f"  {'mean live blue':28s}{st['n_blue'].mean():>12.2f}"
          f"{vt['n_blue'].mean():>12.2f}")
    print(f"  {'mean live red':28s}{st['n_red'].mean():>12.2f}"
          f"{vt['n_red'].mean():>12.2f}")
    print(f"  {'decisions with 0 live blue':28s}"
          f"{int(np.sum(st['n_blue'] == 0)):>12d}"
          f"{int(np.sum(vt['n_blue'] == 0)):>12d}")
    print(f"  {'decisions blue < red':28s}"
          f"{int(np.sum(st['n_blue'] < st['n_red'])):>12d}"
          f"{int(np.sum(vt['n_blue'] < vt['n_red'])):>12d}")
    print("\n  live minions over time (blue/red):")
    print(f"  {'t':>6s}{'sim b':>8s}{'sim r':>8s}{'srv b':>8s}{'srv r':>8s}"
          f"{'sim clash':>11s}{'srv clash':>11s}")
    for mark in range(60, 601, 30):
        si = min(int(np.searchsorted(st["t_s"], mark)), st["t_s"].size - 1)
        vi = min(int(np.searchsorted(vt["t_s"], mark)), vt["t_s"].size - 1)
        si -= si % 30
        vi -= vi % 30
        sc = np.nanmean([st["blue_front"][si], st["red_front"][si]])
        vc = np.nanmean([vt["blue_front"][vi], vt["red_front"][vi]])
        print(f"  {mark:>6d}{st['n_blue'][si]:>8d}{st['n_red'][si]:>8d}"
              f"{vt['n_blue'][vi]:>8d}{vt['n_red'][vi]:>8d}"
              f"{sc:>11.3f}{vc:>11.3f}")

    print("\n--- HYPOTHESIS 3: WHERE THE WAVES CLASH --------------------------")
    print("  lane fraction: 0 = blue's end of the top lane, 1 = red's. "
          "'clash' is the\n  midpoint of (frontmost blue minion, "
          "rearmost red minion).")
    for name, t in (("sim", st), ("server", vt)):
        mid = np.nanmean(np.stack([t["blue_front"], t["red_front"]]), axis=0)
        mid = mid[np.isfinite(mid)]
        print(f"  {name:7s}clash point: {_stats(mid)}")
        bf = t["blue_front"][np.isfinite(t["blue_front"])]
        rf = t["red_front"][np.isfinite(t["red_front"])]
        print(f"         blue front : {_stats(bf)}")
        print(f"         red front  : {_stats(rf)}")
    schamp = lane_fraction(s["cx"], s["cy"])
    vchamp = lane_fraction(v["cx"], v["cy"])
    eng_s = ~s["approaching"] & s["calive"]
    eng_v = ~v["approaching"] & v["calive"]
    print(f"  sim    champion lane frac (engaged, alive): {_stats(schamp[eng_s])}")
    print(f"  server champion lane frac (engaged, alive): {_stats(vchamp[eng_v])}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sim", type=Path, required=True)
    p.add_argument("--server", type=Path, required=True)
    p.add_argument("--server-log", type=Path, required=True)
    a = p.parse_args(argv)
    report(a.sim, a.server, a.server_log)


if __name__ == "__main__":       # pragma: no cover
    main()
