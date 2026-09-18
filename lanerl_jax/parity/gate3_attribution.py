"""Attribute gate 3's three unexplained gaps: approach cost, deaths, level.

The canonical routed gate-3 run (2026-09-18, 18,000 decisions) leaves three
aggregates with no mechanism attached to them::

                        sim     server
    CS                    3          4
    attacks              54         86
    approach decisions 6,400      3,197
    deaths                2          1
    level reached         5          8

Each of those totals is compatible with several different bugs, and the totals
themselves cannot separate them:

* **6,400 approach decisions.** "walks slower", "walks more often" and "dies
  mid-walk and restarts" all inflate the same counter and have different
  fixes. ``SimRun.walks``/``ServerRun.walks`` split the total per walk; this
  module additionally reconstructs, for each walk, whether it ENDED at the
  last waypoint or was cut short by a death.
* **2 deaths vs 1.** A turret death and a minion-aggro death are different
  bugs. This traces the champion's HP over the decisions before each death and
  attributes the damage.
* **level 5 vs 8.** Proximity XP has four candidate failure modes: less XP per
  minion death, fewer minion deaths, the champion being out of ``ExpRadius2``
  more often, or a wrong curve. Only cumulative XP over time separates them.

**It traces the gate's own loop.** Both drivers now take an ``on_decision``
callback, so the run measured here is the run the gate runs -- routed path,
canonical call-for-help, identical waypoints. Re-implementing the drive loop
in a diagnostic is what PATH-006 records going wrong (``isolation.py`` drove
the raw two-point path while the gate ran routed, so its measured tail effect
described a run that no longer happened).

WHAT IT FOUND (2026-09-18, both sides re-run the same day)
-----------------------------------------------------------
The sim side was run against a **pristine HEAD checkout**, not the working
tree, because another session's ``sim/local_pathing.py`` SmoothPath work
(since committed as ``de5cc2b``) was already present locally and would
otherwise have been measured instead of the canonical simulator. The control
is ``c914bde``, and it reproduces the canonical run exactly -- cs 3, attacks
54, approach 6,400, deaths 2 -- so the attribution below is of the gate's own
numbers. The server side was re-run the same day and also reproduces exactly
(cs 4, attacks 86, approach 3,197, deaths 1).

For the record, the same trace at ``de5cc2b`` (SmoothPath ON) gives approach
5,027 with walks (1,242 / 2,543 / 1,242) -- the opening walk drops 1,378 ->
1,242, i.e. -9.9%, which is what a 1.177 -> 1.049 polyline-length ratio buys
and no more. cs 3 / attacks 54 / deaths 2 are unchanged, so SmoothPath moves
the approach cost and not the gate.

**Task 1 -- the 6,400 is "dies mid-walk and restarts", not "walks slower".**
Per-walk, from the drivers' own ``walks`` field::

    sim     1,378   2,630   2,392      (3 walks, 2 deaths)
    server  1,190   2,007              (2 walks, 1 death)

Decomposition of the 3,203-decision excess:

* **2,630 (82.1%)** is walk index 1 in its entirety: the sim's second death
  happened DURING a walk-in, so 1,955 decisions of walking were thrown away
  and the 675-decision respawn timer was counted as approach on top (the
  waypoint cursor is still short of the end while the champion is dead, so
  ``approaching`` never clears). The server's one death happens in lane, so
  its respawn timer lands in ``holds`` instead.
* **385 (12.0%)** is the respawn walk being dearer: 2,392 against 2,007.
* **188 (5.9%)** is the opening walk: 1,378 against 1,190.

**Movement speed and order stutter are ruled out, exactly.** On the opening
walk both engines move at the same speed to three digits -- sim 11.49 units
per decision, server 11.50, i.e. Garen's 345 move speed at 30 Hz. The opening
walk's entire gap is path LENGTH: 15,829 units against 13,680, a ratio of
1.157 against a decision ratio of 1.158. That is the same ~18% the 400-route
host comparison measured for the baked artifact's polylines, and it is all it
can buy. Re-issuing the Move order every decision costs nothing measurable,
and ``MOVE-001``'s 8-transition cap never binds.

**The route fallback is not the story either.** 324 of 6,400 approach
decisions (5.1%) report ``GOAL_OUTSIDE_WINDOW`` -- the approach legs are up
to 3,721 units and the artifact window is +/-50 cells = 2,500 -- but those
decisions run at 11.50 units each with **zero** sub-1-unit decisions. The
two-point fallback on a straight lane leg costs nothing here.

**What does slow a walk is creep block, and it slows both engines.** The
respawn walks run at 7.78 (sim) and 6.82 (server) units per decision against
11.5 when clear, because the champion walks back up the lane through a live
wave. In the sim, decisions with a minion inside 120 units average 4.45 units
of progress against 11.48 when clear.

**Task 2 -- all three deaths are the same mechanism, and none involve a
turret.** Nearest enemy turret at the moment of death: 1,612 / 2,741 units
(sim) and 1,841 (server), against a 750 turret range. Every death is a pack
of red minions grinding the champion down -- 6 of them holding target for the
whole 61-decision window in both sim deaths, 4 inside 190 units in the
server's -- in ~15-23 damage chunks, and the killing blow is a red caster
minion in all three. The sim simply takes more of it: 3,840 total damage
against 2,758, 1,926 decisions with >=3 red minions inside 200 units against
1,290, and a worst sustained swarm of 710 decisions (23.7 s, full health to
zero) against 418. The sim's second death happens at t=479 s *during* walk
index 1, which is what links Task 2 back to Task 1.

**Task 3 -- the ledger's "sim reaches 5" is stale; the canonical run reaches
7.** Measured: sim 3,891 XP / level 7, server 4,276 XP / level 8. Cause (c),
and nothing else:

* NOT (a) less XP per death. Grant sizes are identical on both sides --
  77 / 51 / 94, matching ``Red_Minion_Basic`` / ``_Wizard`` / ``_MechCannon``
  Content ``ExpGivenOnDeath`` exactly. ``ExpGivenOnDeath`` has no time ramp in
  the source (``Stats.cs`` sets it from ``CharData`` and ``StatsModifier`` has
  no such field).
* NOT (b) fewer minion deaths. **82 red minion deaths on both sides**, 64
  (sim) against 68 (server) inside ``ExpRadius2``.
* NOT (d) a wrong curve. Level-up thresholds match Map1's ``ExpCurve.json``
  (280/660/1140/1720/2400/3180/4060) on both sides.
* (c) The sim captures **5 fewer grants**: 60 against 65, and the missing five
  are all MELEE (27 against 32 at 77 XP; caster 30 = 30 and cannon 3 = 3).
  5 x 77 = 385 XP, which is the entire gap. Cumulative XP is level with the
  server to 300 s (1,861 against 1,835) and falls behind only over 360-540 s
  -- exactly the window containing both sim deaths (394 s, 479 s) and both
  respawn walks. The sim ends 169 XP short of level 8's 4,060.

So the level gap IS Task 1 restating itself, and Task 1 is mostly Task 2.
One death is worth about 2,600 approach decisions and about 385 XP here.

Usage::

    python -m lanerl_jax.parity.gate3_attribution sim    --out runs/g3_sim.npz
    python -m lanerl_jax.parity.gate3_attribution server --out runs/g3_srv.npz
    python -m lanerl_jax.parity.gate3_attribution report \
        --sim runs/g3_sim.npz --server runs/g3_srv.npz

The two collection commands are independent; the server one boots a real
server and takes several minutes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

from ..sim.local_pathing import LocalRouteStatus
from ..sim.rewards import EXP_RADIUS
from ..sim.state import Kind, Team
from .last_hit_drive import (APPROACH_WAYPOINTS, DECISIONS_600S,
                             run_oracle_in_sim, run_oracle_on_server)

__all__ = ["SimTracer", "ServerTracer", "collect_sim", "collect_server",
           "report"]

#: Wire ``GetType().Name`` -> compact code, so both engines' traces share one
#: vocabulary. 0 is "something else the lane does not model".
KIND_CODE = {"Champion": 1, "LaneMinion": 2, "LaneTurret": 3}

#: Padded per-decision unit capacity for the server trace. The wire emits every
#: AttackableUnit on the map (all 24 turrets exist even under TOPONLY), plus at
#: most `N_MINIONS`-ish live minions and 2 champions. Overflow raises rather
#: than truncating: a silently truncated unit list would look exactly like a
#: minion that never existed.
SERVER_UNIT_CAP = 128


class SimTracer:
    """``on_decision`` callback for :func:`run_oracle_in_sim`.

    Pulls only what the three questions need, but pulls it EVERY decision --
    including during the approach, which is the half of the episode the
    aggregate counters say the least about.
    """

    def __init__(self) -> None:
        self.t_ms: list = []
        self.approaching: list = []
        self.walk_index: list = []
        self.respawned: list = []
        self.cx: list = []
        self.cy: list = []
        self.chp: list = []
        self.cmhp: list = []
        self.clevel: list = []
        self.cxp: list = []
        self.ccs: list = []
        self.calive: list = []
        #: index of whoever last damaged the champion (`Champion._playerHitId`
        #: equivalent, `state.hit_flag_by`) -- direct death attribution.
        self.hit_by: list = []
        self.hit_flag_ms: list = []
        #: PATH-005's per-unit route status, persisted on `LaneState`. Anything
        #: other than READY means the champion is walking a two-point fallback
        #: rather than a routed path, and that is a different bug from a route
        #: that is merely longer than the server's.
        self.route_status: list = []
        self.n_waypoints: list = []
        self.move_order: list = []
        #: per-unit snapshots, (decisions, N_UNITS)
        self.alive: list = []
        self.hp: list = []
        self.x: list = []
        self.y: list = []
        self.kind: list = []
        self.team: list = []
        self.model: list = []
        #: who each unit is currently attacking; `== 0` means "this champion".
        self.target: list = []

    def __call__(self, rec: dict) -> None:
        st = rec["state"]
        alive = np.asarray(st.alive)
        hp = np.asarray(st.hp)
        x = np.asarray(st.x)
        y = np.asarray(st.y)
        self.t_ms.append(float(np.asarray(st.t_ms).reshape(-1)[0]))
        self.approaching.append(bool(rec["approaching"]))
        self.walk_index.append(int(rec["walk_index"]))
        self.respawned.append(bool(rec["respawned"]))
        self.cx.append(float(x[0]))
        self.cy.append(float(y[0]))
        self.chp.append(float(hp[0]))
        self.cmhp.append(float(np.asarray(st.max_hp)[0]))
        self.clevel.append(int(np.asarray(st.level)[0]))
        self.cxp.append(float(np.asarray(st.xp)[0]))
        self.ccs.append(int(np.asarray(st.cs)[0]))
        self.calive.append(bool(alive[0]))
        self.hit_by.append(int(np.asarray(st.hit_flag_by)[0]))
        self.hit_flag_ms.append(float(np.asarray(st.hit_flag_ms)[0]))
        self.route_status.append(int(np.asarray(st.route_status)[0]))
        self.n_waypoints.append(int(np.asarray(st.n_waypoints)[0]))
        self.move_order.append(int(np.asarray(st.move_order)[0]))
        self.alive.append(alive)
        self.hp.append(hp.astype(np.float32))
        self.x.append(x.astype(np.float32))
        self.y.append(y.astype(np.float32))
        self.kind.append(np.asarray(st.kind))
        self.team.append(np.asarray(st.team))
        self.model.append(np.asarray(st.model))
        self.target.append(np.asarray(st.target))

    def to_npz(self, path: Path, run) -> None:
        np.savez_compressed(
            path,
            engine=np.array("sim"),
            t_ms=np.asarray(self.t_ms, np.float64),
            approaching=np.asarray(self.approaching, bool),
            walk_index=np.asarray(self.walk_index, np.int32),
            respawned=np.asarray(self.respawned, bool),
            cx=np.asarray(self.cx, np.float32), cy=np.asarray(self.cy, np.float32),
            chp=np.asarray(self.chp, np.float32),
            cmhp=np.asarray(self.cmhp, np.float32),
            clevel=np.asarray(self.clevel, np.int32),
            cxp=np.asarray(self.cxp, np.float64),
            ccs=np.asarray(self.ccs, np.int32),
            calive=np.asarray(self.calive, bool),
            hit_by=np.asarray(self.hit_by, np.int32),
            hit_flag_ms=np.asarray(self.hit_flag_ms, np.float32),
            route_status=np.asarray(self.route_status, np.int32),
            n_waypoints=np.asarray(self.n_waypoints, np.int32),
            move_order=np.asarray(self.move_order, np.int32),
            u_alive=np.asarray(self.alive, bool),
            u_hp=np.asarray(self.hp, np.float32),
            u_x=np.asarray(self.x, np.float32),
            u_y=np.asarray(self.y, np.float32),
            u_kind=np.asarray(self.kind, np.int8),
            u_team=np.asarray(self.team, np.int8),
            u_model=np.asarray(self.model, np.int8),
            u_target=np.asarray(self.target, np.int16),
            summary=np.array(json.dumps({
                "cs": run.cs, "decisions": run.decisions,
                "approach_decisions": run.approach_decisions,
                "attacks": run.attacks, "moves": run.moves, "holds": run.holds,
                "deaths": run.deaths, "vis_mean": run.vis_mean,
                "walks": list(run.walks)})))


class ServerTracer:
    """``on_decision`` callback for :func:`run_oracle_on_server`.

    The wire carries the champion's OWN ``xp``/``lvl``/``cs`` (`LanerlControl.
    BuildObservation`), so the server side of the level question needs no
    inference at all. It carries no per-minion target, so death attribution on
    this side is by damage size and proximity rather than by a target field.
    """

    def __init__(self) -> None:
        self.t: list = []
        self.approaching: list = []
        self.walk_index: list = []
        self.respawned: list = []
        self.cx: list = []
        self.cy: list = []
        self.chp: list = []
        self.cmhp: list = []
        self.clevel: list = []
        self.cxp: list = []
        self.ccs: list = []
        self.calive: list = []
        self.u_id: list = []
        self.u_kind: list = []
        self.u_team: list = []
        self.u_x: list = []
        self.u_y: list = []
        self.u_hp: list = []
        self.u_mhp: list = []
        self.u_vb: list = []
        #: wire ``MinionSpawnType`` (``WIRE_MINION_TYPE``), 255 for non-minions.
        #: Needed because a lane minion's acquisition range depends on its type
        #: (caster 700, melee/cannon 600) and the wire has no range field.
        self.u_mt: list = []
        self.unknown_kinds: set = set()

    def __call__(self, rec: dict) -> None:
        obs = rec["obs"]
        blue = rec["blue"]
        units = obs.get("u", [])
        n = len(units)
        if n > SERVER_UNIT_CAP:
            raise RuntimeError(
                f"{n} units exceeds SERVER_UNIT_CAP={SERVER_UNIT_CAP}; raise "
                "the cap rather than truncating the unit list")
        # NetIds are uint32 and really do exceed int32 (observed
        # 4,294,540,725), so this column is int64 rather than int32.
        uid = np.zeros(SERVER_UNIT_CAP, np.int64)
        ukind = np.zeros(SERVER_UNIT_CAP, np.int8)
        uteam = np.zeros(SERVER_UNIT_CAP, np.int16)
        ux = np.zeros(SERVER_UNIT_CAP, np.float32)
        uy = np.zeros(SERVER_UNIT_CAP, np.float32)
        uhp = np.zeros(SERVER_UNIT_CAP, np.float32)
        umhp = np.zeros(SERVER_UNIT_CAP, np.float32)
        uvb = np.zeros(SERVER_UNIT_CAP, np.int8)
        umt = np.full(SERVER_UNIT_CAP, 255, np.uint8)
        for j, u in enumerate(units):
            k = u.get("k", "")
            code = KIND_CODE.get(k, 0)
            if code == 0:
                self.unknown_kinds.add(k)
            uid[j] = int(u["id"])
            ukind[j] = code
            uteam[j] = int(u.get("tm", 0))
            ux[j] = float(u.get("x", 0.0))
            uy[j] = float(u.get("y", 0.0))
            uhp[j] = float(u.get("hp", 0.0))
            umhp[j] = float(u.get("mhp", 0.0))
            uvb[j] = int(u.get("vb", 0))
            if code == KIND_CODE["LaneMinion"]:
                umt[j] = int(u.get("mt", 255))
        self.t.append(float(obs.get("t", 0)))
        self.approaching.append(bool(rec["approaching"]))
        self.walk_index.append(int(rec["walk_index"]))
        self.respawned.append(bool(rec["respawned"]))
        self.cx.append(float(rec["x"]))
        self.cy.append(float(rec["y"]))
        self.chp.append(float(blue.get("hp", 0)))
        self.cmhp.append(float(blue.get("mhp", 0)))
        self.clevel.append(int(blue.get("lvl", 0)))
        self.cxp.append(float(blue.get("xp", 0)))
        self.ccs.append(int(blue.get("cs", 0)))
        self.calive.append(bool(rec["alive"]))
        self.u_id.append(uid)
        self.u_kind.append(ukind)
        self.u_team.append(uteam)
        self.u_x.append(ux)
        self.u_y.append(uy)
        self.u_hp.append(uhp)
        self.u_mhp.append(umhp)
        self.u_vb.append(uvb)
        self.u_mt.append(umt)

    def to_npz(self, path: Path, run) -> None:
        np.savez_compressed(
            path,
            engine=np.array("server"),
            # `obs["t"]` is `(int)game.GameTime`, which is already in
            # MILLISECONDS -- a 600 s episode ends at 592,837, not 592.
            t_ms=np.asarray(self.t, np.float64),
            approaching=np.asarray(self.approaching, bool),
            walk_index=np.asarray(self.walk_index, np.int32),
            respawned=np.asarray(self.respawned, bool),
            cx=np.asarray(self.cx, np.float32), cy=np.asarray(self.cy, np.float32),
            chp=np.asarray(self.chp, np.float32),
            cmhp=np.asarray(self.cmhp, np.float32),
            clevel=np.asarray(self.clevel, np.int32),
            cxp=np.asarray(self.cxp, np.float64),
            ccs=np.asarray(self.ccs, np.int32),
            calive=np.asarray(self.calive, bool),
            u_id=np.asarray(self.u_id, np.int64),
            u_kind=np.asarray(self.u_kind, np.int8),
            u_team=np.asarray(self.u_team, np.int16),
            u_x=np.asarray(self.u_x, np.float32),
            u_y=np.asarray(self.u_y, np.float32),
            u_hp=np.asarray(self.u_hp, np.float32),
            u_mhp=np.asarray(self.u_mhp, np.float32),
            u_vb=np.asarray(self.u_vb, np.int8),
            u_mt=np.asarray(self.u_mt, np.uint8),
            unknown_kinds=np.array(json.dumps(sorted(self.unknown_kinds))),
            summary=np.array(json.dumps({
                "cs": run.cs, "decisions": run.decisions,
                "approach_decisions": run.approach_decisions,
                "attacks": run.attacks, "moves": run.moves, "holds": run.holds,
                "deaths": run.deaths, "vis_mean": run.vis_mean,
                "walks": list(run.walks),
                "log_path": str(run.log_path)})))


def collect_sim(out: Path, decisions: int = DECISIONS_600S, seed: int = 0,
                table_disabled: bool = False):
    tr = SimTracer()
    run = run_oracle_in_sim(decisions=decisions, seed=seed,
                            table_disabled=table_disabled, on_decision=tr)
    out.parent.mkdir(parents=True, exist_ok=True)
    tr.to_npz(out, run)
    return run


def collect_server(out: Path, decisions: int = DECISIONS_600S,
                   port_base: int = 46200, bot_seed: int = 4242,
                   autobuy: bool = False, log_dir: Optional[Path] = None,
                   aggro_trace: bool = False):
    """Trace the gate's own server run.

    ``aggro_trace`` turns on :data:`lanerl_jax.parity.targets.TRACE_ENV`, whose
    ``MRT`` lines are the ONLY server-side view of which unit a minion is
    attacking -- the wire has no per-minion target field. The log they land in
    is recorded as ``summary["log_path"]``, so an analysis that needs both the
    switch events and the per-decision unit census can join them.
    """
    from .targets import TRACE_ENV

    tr = ServerTracer()
    run = run_oracle_on_server(decisions=decisions, port_base=port_base,
                               bot_seed=bot_seed, tag="gate3_attribution",
                               log_dir=log_dir, autobuy=autobuy,
                               on_decision=tr,
                               extra_env=dict(TRACE_ENV) if aggro_trace else None)
    out.parent.mkdir(parents=True, exist_ok=True)
    tr.to_npz(out, run)
    return run


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

#: A respawn puts the champion in its fountain in one decision. That is a
#: teleport, not travel: including it inflates a respawn walk's measured path
#: by the full length of the lane. Anything above this in one decision at 30 Hz
#: is not movement (Garen's 345 move speed is 11.5 units per decision).
TELEPORT_STEP = 200.0


def _walked(cx, cy) -> float:
    """Total distance actually WALKED, with respawn teleports removed."""
    st = np.hypot(np.diff(np.asarray(cx, np.float64)),
                  np.diff(np.asarray(cy, np.float64)))
    return float(st[st <= TELEPORT_STEP].sum())


def _walk_segments(d: dict) -> list:
    """Reconstruct each walk-to-lane as a labelled segment.

    A segment is a maximal run of ``approaching`` decisions. How it ENDED is
    the part the ``walks`` tuple cannot carry: ``arrived`` means the next
    decision was an engaged one, ``died`` means the champion was dead at the
    end of it (the walk is restarted from the fountain on respawn).
    """
    ap = d["approaching"]
    alive = d["calive"]
    respawned = d["respawned"]
    n = len(ap)
    segs = []
    i = 0
    while i < n:
        if not ap[i]:
            i += 1
            continue
        j = i + 1
        # A death DURING a walk does not clear `approaching` -- the champion is
        # dead with the waypoint cursor still short of the end, so the driver
        # keeps counting approach decisions through the whole respawn timer and
        # then starts a fresh walk. Splitting on `respawned` reproduces the
        # driver's own `walks` tuple; a maximal run of `approaching` merges the
        # abandoned walk, the respawn wait and the redo into one segment.
        while j < n and ap[j] and not respawned[j]:
            j += 1
        died = bool(np.any(~alive[i:j]))
        segs.append({
            "start": int(i), "end": int(j), "decisions": int(j - i),
            "t_start_s": float(d["t_ms"][i] / 1000.0),
            "t_end_s": float(d["t_ms"][j - 1] / 1000.0),
            "ended": "died" if died else ("arrived" if j < n else "episode_end"),
            "dead_decisions": int(np.sum(~alive[i:j])),
            "start_xy": (float(d["cx"][i]), float(d["cy"][i])),
            "end_xy": (float(d["cx"][j - 1]), float(d["cy"][j - 1])),
            "path_len": float(_walked(d["cx"][i:j], d["cy"][i:j])),
        })
        i = j
    return segs


_ROUTE_STATUS_NAME = {v: k for k, v in vars(LocalRouteStatus).items()
                      if isinstance(v, int) and not k.startswith("_")}


def _walk_kinematics(name: str, d: dict, segs: list) -> None:
    """Separate "walks a longer path" from "walks slower" on the same path.

    The distance ratio between the sim's reconstructed polylines and the
    server's `SmoothPath` output was measured on 2026-09-18 over 400 routes on
    the production artifact: mean 1.177, max 1.414. That is a bounded ~18%, so
    it cannot by itself produce a 2.0x approach cost. World units travelled per
    decision is the number that tells the two apart -- it is exactly
    "how fast does this champion actually move", independent of which line it
    walks along.

    Server wire coordinates are int-truncated (`LanerlControl.
    BuildObservation` casts to `(int)`), so a single step of ~20 units carries
    up to ~1.4 units of quantisation noise. Block displacement over 30
    decisions (1 s) is reported next to the per-step figure because that noise
    telescopes out of it.
    """
    alive = d["calive"]
    steps = []
    blocks = []
    stalled = 0
    total = 0
    for g in segs:
        lo, hi = g["start"], g["end"]
        m = alive[lo:hi]
        sx = d["cx"][lo:hi].astype(np.float64)
        sy = d["cy"][lo:hi].astype(np.float64)
        st = np.hypot(np.diff(sx), np.diff(sy))
        live_step = st[m[1:] & m[:-1] & (st <= TELEPORT_STEP)]
        steps.append(live_step)
        total += live_step.size
        stalled += int(np.sum(live_step < 1.0))
        for b in range(lo, hi - 30, 30):
            if not np.all(alive[b:b + 31]):
                continue
            disp = float(np.hypot(d["cx"][b + 30] - d["cx"][b],
                                  d["cy"][b + 30] - d["cy"][b]))
            if disp <= 30 * TELEPORT_STEP:
                blocks.append(disp)
    st = np.concatenate(steps) if steps else np.zeros(0)
    print(f"  units/decision while walking (alive): mean={st.mean():.2f} "
          f"median={np.median(st):.2f} p10={np.percentile(st, 10):.2f} "
          f"p90={np.percentile(st, 90):.2f}" if st.size else "  no live steps")
    if st.size:
        print(f"  decisions with < 1 unit of progress: {stalled}/{total} "
              f"({100.0 * stalled / total:.1f}%)")
        print(f"  implied speed: {st.mean() * 30.0:.1f} units/s")
    if blocks:
        b = np.asarray(blocks)
        print(f"  1 s NET displacement blocks (quantisation-free): "
              f"n={b.size} mean={b.mean():.1f} median={np.median(b):.1f} "
              f"p10={np.percentile(b, 10):.1f} u/s")
    for g in segs:
        straight = float(np.hypot(g["end_xy"][0] - g["start_xy"][0],
                                  g["end_xy"][1] - g["start_xy"][1]))
        print(f"    walk {segs.index(g)}: path={g['path_len']:8.0f}u "
              f"straight={straight:8.0f}u tortuosity="
              f"{(g['path_len'] / straight if straight > 1 else float('nan')):.2f} "
              f"units/decision={g['path_len'] / max(g['decisions'], 1):.2f}")
    if "route_status" in d:
        ap = d["approaching"]
        rs = d["route_status"][ap]
        uniq, cnt = np.unique(rs, return_counts=True)
        print("  route_status over approach decisions: " + ", ".join(
            f"{_ROUTE_STATUS_NAME.get(int(u), u)}={c} "
            f"({100.0 * c / rs.size:.1f}%)" for u, c in zip(uniq, cnt)))
        nw = d["n_waypoints"][ap]
        print(f"  n_waypoints during approach: mean={nw.mean():.2f} "
              f"median={np.median(nw):.0f} max={nw.max()}")


def _deaths(d: dict) -> list:
    alive = d["calive"]
    idx = np.flatnonzero(alive[:-1] & ~alive[1:]) + 1
    return [int(i) for i in idx]


def _sim_unit_label(d: dict, i: int, u: int) -> str:
    kind = int(d["u_kind"][i, u])
    team = int(d["u_team"][i, u])
    tname = {Team.BLUE: "blue", Team.RED: "red"}.get(team, str(team))
    kname = {Kind.CHAMPION: "champion", Kind.LANE_MINION: "minion",
             Kind.TURRET: "turret"}.get(kind, f"kind{kind}")
    return f"{tname} {kname}[{u}] model={int(d['u_model'][i, u])}"


def _sim_death_report(d: dict, i: int, window: int = 60) -> dict:
    lo = max(0, i - window)
    hp = d["chp"][lo:i + 1]
    dmg = -np.diff(d["chp"][lo:i + 1])
    # who was targeting the champion over the window
    tgt = d["u_target"][lo:i + 1]            # (w, N)
    al = d["u_alive"][lo:i + 1]
    on_champ = (tgt == 0) & al
    counts = on_champ.sum(axis=0)
    attackers = [
        (_sim_unit_label(d, i, int(u)), int(counts[u]))
        for u in np.argsort(-counts)[:8] if counts[u] > 0
    ]
    # nearest enemy turret over the window
    is_turret = (d["u_kind"][i] == Kind.TURRET) & (d["u_team"][i] == Team.RED) \
        & d["u_alive"][i]
    tu = np.flatnonzero(is_turret)
    if tu.size:
        dist = np.hypot(d["u_x"][i, tu] - d["cx"][i], d["u_y"][i, tu] - d["cy"][i])
        near_t = (float(dist.min()), int(tu[int(np.argmin(dist))]))
    else:
        near_t = (float("inf"), -1)
    return {
        "decision": int(i),
        "game_time_s": float(d["t_ms"][i] / 1000.0),
        "during_walk": bool(d["approaching"][i]),
        "xy": (float(d["cx"][i]), float(d["cy"][i])),
        "hp_start_of_window": float(hp[0]),
        "max_hp": float(d["cmhp"][i]),
        "level": int(d["clevel"][i]),
        "biggest_single_decision_damage": float(dmg.max()) if dmg.size else 0.0,
        "damage_over_window": float(np.sum(dmg[dmg > 0])),
        "n_damaging_decisions": int(np.sum(dmg > 0.5)),
        "hp_tail": [round(float(v), 1) for v in d["chp"][max(0, i - 30):i + 1]],
        "last_damaged_by": _sim_unit_label(d, i, int(d["hit_by"][i]))
        if 0 <= int(d["hit_by"][i]) < d["u_kind"].shape[1] else str(int(d["hit_by"][i])),
        "attackers_targeting_champ_decisions": attackers,
        "nearest_red_turret_dist": near_t[0],
    }


def _server_death_report(d: dict, i: int, window: int = 60) -> dict:
    lo = max(0, i - window)
    dmg = -np.diff(d["chp"][lo:i + 1])
    # enemy units near the champion on the death decision
    k = d["u_kind"][i]
    tm = d["u_team"][i]
    hp = d["u_hp"][i]
    live = (k > 0) & (hp > 0)
    enemy = live & (tm == 200)
    dist = np.hypot(d["u_x"][i] - d["cx"][i], d["u_y"][i] - d["cy"][i])
    near = np.flatnonzero(enemy & (dist < 1200))
    order = near[np.argsort(dist[near])][:8]
    kname = {1: "champion", 2: "minion", 3: "turret", 0: "other"}
    turret = enemy & (k == 3)
    tdist = float(dist[turret].min()) if np.any(turret) else float("inf")
    return {
        "decision": int(i),
        "game_time_s": float(d["t_ms"][i] / 1000.0),
        "during_walk": bool(d["approaching"][i]),
        "xy": (float(d["cx"][i]), float(d["cy"][i])),
        "hp_start_of_window": float(d["chp"][lo]),
        "max_hp": float(d["cmhp"][i]),
        "level": int(d["clevel"][i]),
        "biggest_single_decision_damage": float(dmg.max()) if dmg.size else 0.0,
        "damage_over_window": float(np.sum(dmg[dmg > 0])),
        "n_damaging_decisions": int(np.sum(dmg > 0.5)),
        "hp_tail": [round(float(v), 1) for v in d["chp"][max(0, i - 30):i + 1]],
        "nearest_enemies": [
            (kname.get(int(k[u]), "?"), round(float(dist[u]), 1),
             round(float(hp[u]), 1)) for u in order],
        "nearest_red_turret_dist": tdist,
    }


def _sim_minion_deaths(d: dict) -> dict:
    """Red-minion deaths in the sim, and how many were inside ``ExpRadius2``.

    A slot is reused on respawn, so identity is the alive-flag transition on a
    slot whose ``kind`` is a lane minion on BOTH decisions -- a slot that went
    dead and was refilled by a later wave cannot be mistaken for two deaths in
    one decision at 30 Hz.
    """
    alive = d["u_alive"]
    kind = d["u_kind"]
    team = d["u_team"]
    died = alive[:-1] & ~alive[1:] & (kind[:-1] == Kind.LANE_MINION)
    out = {}
    for tname, tval in (("red", Team.RED), ("blue", Team.BLUE)):
        m = died & (team[:-1] == tval)
        ii, uu = np.nonzero(m)
        dist = np.hypot(d["u_x"][ii, uu] - d["cx"][ii], d["u_y"][ii, uu] - d["cy"][ii])
        out[tname] = {"n": int(ii.size),
                      "n_in_exp_radius": int(np.sum(dist <= EXP_RADIUS)),
                      "t_s": (d["t_ms"][ii] / 1000.0),
                      "dist": dist}
    return out


def _server_minion_deaths(d: dict) -> dict:
    """Same, by wire NetId rather than slot: ids are never reused."""
    out = {}
    n = d["u_id"].shape[0]
    prev = {}
    rec = {"red": [], "blue": []}
    for i in range(n):
        ids = d["u_id"][i]
        k = d["u_kind"][i]
        hp = d["u_hp"][i]
        tm = d["u_team"][i]
        cur = {}
        for j in np.flatnonzero(k == 2):
            cur[int(ids[j])] = (float(hp[j]), int(tm[j]), float(d["u_x"][i, j]),
                                float(d["u_y"][i, j]))
        for mid, (php, ptm, px, py) in prev.items():
            if php <= 0:
                continue
            nxt = cur.get(mid)
            if nxt is None or nxt[0] <= 0:
                x, y = (nxt[2], nxt[3]) if nxt is not None else (px, py)
                dist = float(np.hypot(x - d["cx"][i], y - d["cy"][i]))
                rec["red" if ptm == 200 else "blue"].append(
                    (float(d["t_ms"][i] / 1000.0), dist))
        prev = cur
    for tname in ("red", "blue"):
        arr = np.asarray(rec[tname], np.float64).reshape(-1, 2)
        out[tname] = {"n": int(arr.shape[0]),
                      "n_in_exp_radius": int(np.sum(arr[:, 1] <= EXP_RADIUS))
                      if arr.size else 0,
                      "t_s": arr[:, 0], "dist": arr[:, 1]}
    return out


def _load(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def _fmt_hist(vals) -> str:
    v = np.asarray(vals, float)
    if v.size == 0:
        return "(none)"
    return (f"n={v.size} sum={v.sum():.0f} mean={v.mean():.1f} "
            f"median={np.median(v):.1f} min={v.min():.0f} max={v.max():.0f}")


def report(sim_path: Path, server_path: Path) -> None:
    s = _load(sim_path)
    v = _load(server_path)
    ssum = json.loads(str(s["summary"]))
    vsum = json.loads(str(v["summary"]))

    print("=" * 78)
    print("GATE 3 ATTRIBUTION")
    print("=" * 78)
    print(f"sim    : {ssum}")
    print(f"server : {vsum}")

    # ---- task 1: the approach decisions --------------------------------
    print("\n--- TASK 1: what the approach decisions are made of --------------")
    for name, d in (("sim", s), ("server", v)):
        segs = _walk_segments(d)
        print(f"\n{name}: {len(segs)} walk(s), "
              f"{int(np.sum(d['approaching']))} approach decisions")
        print("  per-walk: " + _fmt_hist([g["decisions"] for g in segs]))
        for g in segs:
            print(f"    walk {segs.index(g)}: {g['decisions']:5d} dec "
                  f"({g['decisions']/30.0:6.1f} s)  t={g['t_start_s']:6.1f}"
                  f"->{g['t_end_s']:6.1f}s  ended={g['ended']:11s} "
                  f"dead_dec={g['dead_decisions']:4d} "
                  f"path={g['path_len']:8.0f}u "
                  f"start=({g['start_xy'][0]:.0f},{g['start_xy'][1]:.0f}) "
                  f"end=({g['end_xy'][0]:.0f},{g['end_xy'][1]:.0f})")
        live = [g["decisions"] - g["dead_decisions"] for g in segs]
        print(f"  walking-while-alive only: {_fmt_hist(live)}")
        _walk_kinematics(name, d, segs)

    # ---- task 2: the deaths -------------------------------------------
    print("\n--- TASK 2: the deaths -------------------------------------------")
    for name, d, fn in (("sim", s, _sim_death_report),
                        ("server", v, _server_death_report)):
        for i in _deaths(d):
            r = fn(d, i)
            print(f"\n{name} death @ decision {r['decision']} "
                  f"(t={r['game_time_s']:.1f}s, level {r['level']}, "
                  f"during_walk={r['during_walk']})")
            for k, val in r.items():
                if k in ("decision", "game_time_s", "level", "during_walk"):
                    continue
                print(f"    {k}: {val}")

    # ---- task 2b: why the deaths happen at all -------------------------
    print("\n--- TASK 2b: damage exposure ------------------------------------")
    print(f"{'':34s}{'sim':>10s}{'server':>10s}")
    rows = []
    for name, d, mk, red in (("sim", s, Kind.LANE_MINION, Team.RED),
                             ("server", v, 2, 200)):
        hp = d["chp"].astype(float)
        al = d["calive"]
        dmg = -np.diff(hp)
        dmg = dmg[(dmg > 0) & al[1:] & al[:-1]]
        frac = hp / np.maximum(d["cmhp"].astype(float), 1.0)
        live = (d["u_kind"] == mk) & (d["u_team"] == red) & (d["u_hp"] > 0)
        if "u_alive" in d:
            live = live & d["u_alive"]
        dist = np.hypot(d["u_x"] - d["cx"][:, None], d["u_y"] - d["cy"][:, None])
        near = (live & (dist < 200.0)).sum(axis=1)
        swarm = near >= 3
        idx = np.flatnonzero(swarm)
        longest = 0
        total = 0
        if idx.size:
            br = np.flatnonzero(np.diff(idx) > 1)
            st = np.concatenate([[idx[0]], idx[br + 1]])
            en = np.concatenate([idx[br], [idx[-1]]])
            lens = (en - st + 1)
            keep = lens >= 60
            longest = int(lens.max())
            total = int(lens[keep].sum())
        rows.append({
            "total damage taken": f"{dmg.sum():.0f}",
            "damaging decisions": f"{int((dmg > 0.5).sum())}",
            "decisions below 25% HP": f"{int(np.sum(al & (frac < 0.25)))}",
            "decisions with >=1 red minion <200u": f"{int((near > 0).sum())}",
            "decisions with >=3 (swarm)": f"{int(swarm.sum())}",
            "decisions in swarms lasting >=2 s": f"{total}",
            "longest swarm (decisions)": f"{longest}",
        })
    for k in rows[0]:
        print(f"  {k:32s}{rows[0][k]:>10s}{rows[1][k]:>10s}")

    # ---- task 3: XP and level ------------------------------------------
    print("\n--- TASK 3: experience -------------------------------------------")
    for name, d, deaths_fn in (("sim", s, _sim_minion_deaths),
                               ("server", v, _server_minion_deaths)):
        t = d["t_ms"] / 1000.0
        xp = d["cxp"]
        lvl = d["clevel"]
        print(f"\n{name}: final xp={xp[-1]:.0f} level={lvl[-1]} cs={d['ccs'][-1]}")
        marks = [120, 180, 240, 300, 360, 420, 480, 540, 600]
        row = []
        for m in marks:
            j = int(np.searchsorted(t, m))
            j = min(j, len(t) - 1)
            row.append(f"{m}s:{xp[j]:.0f}/L{lvl[j]}")
        print("  cumulative xp: " + "  ".join(row))
        ups = np.flatnonzero(np.diff(lvl) > 0) + 1
        print("  level-ups at: " + ", ".join(
            f"L{lvl[j]}@{t[j]:.0f}s(xp={xp[j]:.0f})" for j in ups))
        jumps = np.diff(xp)
        pos = jumps[jumps > 0.01]
        if pos.size:
            uniq, cnt = np.unique(np.round(pos, 2), return_counts=True)
            top = sorted(zip(uniq, cnt), key=lambda p: -p[1])[:10]
            print(f"  xp grants: n={pos.size} total={pos.sum():.0f} "
                  f"distinct sizes (top): " +
                  ", ".join(f"{u:g}x{c}" for u, c in top))
        md = deaths_fn(d)
        for tname in ("red", "blue"):
            e = md[tname]
            print(f"  {tname} minion deaths: n={e['n']} "
                  f"within {EXP_RADIUS:.0f}u of champ: {e['n_in_exp_radius']}")
        red = md["red"]
        if red["n_in_exp_radius"]:
            print(f"  xp per in-radius ENEMY minion death: "
                  f"{xp[-1] / red['n_in_exp_radius']:.2f} "
                  "(blue deaths also grant nothing to blue; both sides "
                  "computed identically)")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("sim")
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--decisions", type=int, default=DECISIONS_600S)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--table-disabled", action="store_true")
    b = sub.add_parser("server")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--decisions", type=int, default=DECISIONS_600S)
    b.add_argument("--port-base", type=int, default=46200)
    b.add_argument("--bot-seed", type=int, default=4242)
    b.add_argument("--autobuy", action="store_true")
    b.add_argument("--log-dir", type=Path, default=None)
    b.add_argument("--aggro-trace", action="store_true",
                   help="also enable LANERL_AGGRO_TRACE/LANERL_TURRET_TRACE")
    c = sub.add_parser("report")
    c.add_argument("--sim", type=Path, required=True)
    c.add_argument("--server", type=Path, required=True)
    args = p.parse_args(argv)

    if args.cmd == "sim":
        run = collect_sim(args.out, decisions=args.decisions, seed=args.seed,
                          table_disabled=args.table_disabled)
        print(f"sim run: {run}")
        print(f"wrote {args.out}")
    elif args.cmd == "server":
        run = collect_server(args.out, decisions=args.decisions,
                             port_base=args.port_base, bot_seed=args.bot_seed,
                             autobuy=args.autobuy, log_dir=args.log_dir,
                             aggro_trace=args.aggro_trace)
        print(f"server run: {run}")
        print(f"wrote {args.out}")
    else:
        report(args.sim, args.server)


if __name__ == "__main__":       # pragma: no cover
    main()
