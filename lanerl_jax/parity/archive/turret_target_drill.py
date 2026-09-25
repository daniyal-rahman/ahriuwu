"""Why the server's LaneTurret refuses targets the sim happily takes.

`one_step._compare_controller` found 1,508 turret target disagreements over the
19,800-pair corpus, every one of them "the sim holds a minion, the server holds
none", plus 1,483 spurious sim auto-attack fires downstream of them.  The
standing suspect was a **one-tick blind gap** -- `TurretAI.OnUpdate` (the only
caller of `CheckForTargets`) runs before `ObjAIBase.UpdateTarget` (the only
thing that nulls a dead target), so the tick a target dies ends with no target
and no rescan.  This module measures that apart from the alternative, "the sim
acquires where the server never does at all".

The blind gap is real and it is **1.6% of the residual** (24 of 1,508).  The
other 98.4% are ticks on which the server turret held nothing at N either: it
is idle, off cooldown, not attacking, with live enemy minions 200 units away,
for runs as long as 664 consecutive ticks (11 s).

The mechanism is an **acquisition radius that is wider than the retention
radius**, which the sim does not have:

* the candidate set is `GetUnitsInRange(pos, Stats.Range.Total, true)`
  (TurretAI.cs:41) -> `CollisionHandler.GetNearestObjects(Circle(pos, 750))`
  (ApiFunctionManager.cs:601-606) -> the quadtree leaf test
  `Circle.IntersectsWith(Circle)`, `dist^2 < (Radius + node.Radius)^2`
  (QuadTree.cs:61-64), where the node's radius is the unit's **CollisionRadius**
  (CollisionHandler.cs:69-72; 40.0 for a lane minion).  Effective acquisition
  radius: **750 + target.CollisionRadius = 790**.
* the retention test is `DistanceSquared(pos, TargetUnit.Position) > Range^2`
  (TurretAI.cs:29-32) -- **750 flat, centre to centre**.
* selection is **priority only, distance never** (TurretAI.cs:43-62,
  ClassifyUnit.cs: cannon 7 < caster 8 < melee 9), ties to list order.

So a caster minion at 764 outranks a melee minion at 196.  `CheckForTargets`
picks it; four lines later, in the same `OnUpdate`, the retention test drops it
for being past 750.  Next tick, same scan, same pick, same drop.  The turret is
blind for as long as a best-priority minion sits in the 40-unit annulus
750 <= d < 790, and a wave crashing into a tower parks minions there for
seconds at a time.

`sim/targeting.py::turret_acquire` builds its candidate mask with
`d2 <= turret_range^2` and `sim/step.py`'s `left_range` drops with the same
`P("attack_range")**2`.  One radius for both, so the annulus does not exist and
the trap never fires.  That is the exact rule to fix -- not the blind gap.

Two phases:

**Phase A (server trace only, no sim).**  Every LaneTurret target transition,
plus the annulus test: on each idle turret-tick that has a legal in-range
candidate, is there a candidate of equal or better priority in [750, 750+cr)?

**Phase B (one-step differential).**  For each scored disagreement, whether the
server held anything at N at all, whether its tick-N target was alive at N+1,
whether the sim's pick is a stale hold or a fresh acquisition, and whether the
server re-acquires by N+2.  Also attributes the spurious fires by team.

    python -m lanerl_jax.parity.archive.turret_target_drill --existing-log LOG \\
        --phase both --out-json OUT.json
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

import numpy as np

TURRET_RANGE = 750.0
#: `CollisionHandler.GetBounds` -> `Math.Max(0.5f, obj.CollisionRadius)`.  The
#: trace carries it per unit as `cr=`; a lane minion is 40960/1024 = 40.0 and a
#: champion 65.0.  This is the width of the acquisition annulus.
COLLISION_RADIUS = {"LaneMinion": 40.0, "Champion": 65.0}


#: `ClassifyUnit` (GameServerCore/Enums/ClassifyUnit.cs).  Lower wins.
CANNON, CASTER, MELEE, CHAMPION = 7, 8, 9, 11


def _classify(entity) -> int:
    """`ObjAIBase.ClassifyTarget(u)` with `victium == null`.

    `TurretAI.CheckForTargets` passes no victim (TurretAI.cs:56), so the
    call-for-help block never runs and the value is the plain type switch
    (ObjAIBase.cs:420-448).  Max HP identifies the spawn type from the trace:
    cannon 700, melee 455, caster 290 on this patch.
    """
    if entity.kind != "LaneMinion":
        return CHAMPION
    mh = entity.q_max_hp / 1024.0
    if mh > 600:
        return CANNON
    if mh > 350:
        return MELEE
    return CASTER


# --------------------------------------------------------------------------
# phase A -- server trace only
# --------------------------------------------------------------------------

def phase_a(snaps, out: dict, examples: int = 8) -> None:
    from ..diagnostic_identity import net_id_to_entity
    from ..trace import PosQ

    prev = {}                       # turret net_id -> previous tick's target
    drop_reason = collections.Counter()
    gap_hist = collections.Counter()
    idle_since, idle_prev_target = {}, {}
    ex, idle_examples = [], []
    n_turret_ticks = n_held = 0
    held_by_turret = collections.Counter()
    # the cross-tab that settles it: (holds a target?) x (is there a candidate
    # of equal-or-better priority sitting in the acquisition annulus?)
    annulus = collections.Counter()
    annulus_by_turret = collections.Counter()

    for i, sn in enumerate(snaps):
        internals = {iv.net_id: iv for iv in sn.ai_internals}
        ent = net_id_to_entity(sn)
        alive_now = set(internals)
        enemies = [iv for iv in internals.values()
                   if iv.kind in ("LaneMinion", "Champion")]

        for net_id, iv in internals.items():
            if iv.kind != "LaneTurret":
                continue
            n_turret_ticks += 1
            tgt = iv.target_net_id
            held = bool(tgt)
            n_held += held
            if held:
                held_by_turret[net_id] += 1

            # ---- the annulus test ------------------------------------
            # in_range: what the retention test (TurretAI.cs:29) would keep.
            # band:     what only the quadtree's `Range + CollisionRadius`
            #           candidate test (QuadTree.cs:61) reaches, and which the
            #           retention test then throws away the same tick.
            in_range, band = [], []
            for u in enemies:
                if u.team == iv.team:
                    continue
                e = ent.get(u.net_id)
                if e is None or e.dead:
                    continue
                d = math.hypot((u.q_x - iv.q_x) / PosQ, (u.q_y - iv.q_y) / PosQ)
                if d <= TURRET_RANGE:
                    in_range.append((_classify(e), d, u.net_id))
                elif d < TURRET_RANGE + COLLISION_RADIUS.get(u.kind, 40.0):
                    band.append((_classify(e), d, u.net_id))
            if in_range:
                p_in = min(p for p, _, _ in in_range)
                p_band = min([p for p, _, _ in band], default=99)
                # strict: the band unit always wins the priority scan.
                # tie:    it wins iff it precedes the in-range ones in the
                #         quadtree's list order, which the trace cannot show;
                #         the server's own outcome is the evidence.
                cls = ("trap_strict" if p_band < p_in
                       else "trap_tie" if p_band == p_in else "no_trap")
                key = "held" if held else "idle"
                annulus[f"{key}|{cls}"] += 1
                # the unambiguous cell: one globally best-priority candidate,
                # and it is in the band -> the pick is forced and is discarded.
                best = [c for c in in_range + band if c[0] == min(p_in, p_band)]
                if len(best) == 1:
                    annulus[f"{key}|unique_best_"
                            f"{'in_band' if best[0][1] > TURRET_RANGE else 'in_range'}"] += 1
                if not held:
                    annulus_by_turret[f"{net_id}|{cls}"] += 1
                    if cls == "no_trap" and len(idle_examples) < 15:
                        idle_examples.append(dict(
                            t_ms=sn.t_ms, turret=net_id, team=iv.team,
                            in_range=[(c[0], round(c[1], 1), c[2])
                                      for c in sorted(in_range)[:3]],
                            band=[(c[0], round(c[1], 1), c[2])
                                  for c in sorted(band)[:3]],
                            attacking=bool(iv.is_attacking),
                            aacd=iv.q_aa_cooldown))

            # ---- target transitions ----------------------------------
            pr = prev.get(net_id)
            if pr is not None and pr["t_index"] == i - 1:
                was = pr["target"]
                if was and not tgt:
                    tgt_iv = internals.get(was)
                    d = (math.hypot((tgt_iv.q_x - iv.q_x) / PosQ,
                                    (tgt_iv.q_y - iv.q_y) / PosQ)
                         if tgt_iv is not None else float("nan"))
                    reason = ("target_removed" if was not in alive_now
                              else "target_left_range" if d > TURRET_RANGE
                              else "target_alive_in_range")
                    drop_reason[reason] += 1
                    idle_since[net_id] = i
                    idle_prev_target[net_id] = (was, reason)
                    if len(ex) < examples:
                        ex.append(dict(t_ms=sn.t_ms, turret=net_id,
                                       dropped=was, reason=reason,
                                       dist=None if math.isnan(d) else round(d, 1)))
                if tgt and not was and net_id in idle_since:
                    gap = i - idle_since[net_id]
                    was_id, reason = idle_prev_target[net_id]
                    gap_hist[(reason, min(gap, 10))] += 1
                    if tgt == was_id:
                        gap_hist[(reason + "|same_unit", min(gap, 10))] += 1
                    del idle_since[net_id]
            prev[net_id] = dict(t_index=i, target=tgt)

    out["phase_a"] = {
        "n_turret_ticks": n_turret_ticks,
        "n_turret_ticks_holding": n_held,
        "held_ticks_by_turret": dict(held_by_turret),
        "drop_reason": dict(drop_reason),
        "gap_after_drop": {f"{k[0]}|gap={k[1]}": v
                           for k, v in sorted(gap_hist.items())},
        "annulus": dict(annulus),
        "idle_annulus_by_turret": dict(annulus_by_turret),
        "idle_no_trap_examples": idle_examples,
        "drop_examples": ex,
    }


# --------------------------------------------------------------------------
# phase B -- one-step differential
# --------------------------------------------------------------------------

def phase_b(snaps, out: dict, from_ms: int, to_ms: int, max_pairs: int,
            examples: int = 10) -> None:
    import jax.numpy as jnp

    from ...data.patch import load_patch
    from ...sim.init import TOP_LANE_PATH, lane_params
    from ...sim.profiles import PROFILES
    from ..diagnostic_identity import net_id_to_entity, net_id_to_injected_slot
    from ..inject import inject_snapshot, replay_wave_states
    from .one_step import compare_one_tick
    from ..trace import PosQ, StatQ

    wave_states = replay_wave_states(snaps)
    patch = load_patch()
    params = lane_params(patch)
    lane_path = jnp.asarray(np.asarray(TOP_LANE_PATH, np.float32))

    split = collections.Counter()
    fire_by = collections.Counter()
    ex = []
    n_pairs = 0

    for i in range(len(snaps) - 1):
        sn, sn1 = snaps[i], snaps[i + 1]
        if sn.t_ms < from_ms:
            continue
        if sn.t_ms > to_ms or n_pairs >= max_pairs:
            break
        dt = sn1.t_ms - sn.t_ms
        if dt <= 0 or dt > 34:
            continue
        previous = snaps[i - 1] if i else None
        if previous is not None and not (0 < sn.t_ms - previous.t_ms <= 34):
            previous = None
        state_n, report = inject_snapshot(
            sn, wave_states[i], params, PROFILES, previous_snapshot=previous)
        net_id_slots = net_id_to_injected_slot(sn, report.notes)
        tr = compare_one_tick(state_n, report.notes, sn1, params, lane_path,
                              pre_net_id_to_slot=net_id_slots)
        n_pairs += 1

        pre_iv = {iv.net_id: iv for iv in sn.ai_internals}
        post_iv = {iv.net_id: iv for iv in sn1.ai_internals}
        post2_iv = ({iv.net_id: iv for iv in snaps[i + 2].ai_internals}
                    if i + 2 < len(snaps) else {})
        real_n1 = net_id_to_entity(sn1)

        for c in tr.controller:
            if c.kind != "LaneTurret":
                continue
            iv0 = pre_iv.get(c.net_id)
            if iv0 is None:
                continue
            pre_target = iv0.target_net_id

            # ---- target identity disagreements -------------------------
            if c.sim_target_net_id != c.server_target_net_id:
                shape = (
                    "sim_holds_server_none"
                    if c.sim_target_net_id and not c.server_target_net_id
                    else "server_holds_sim_none"
                    if c.server_target_net_id and not c.sim_target_net_id
                    else "different_units")
                split[f"shape|{shape}"] += 1
                if shape == "sim_holds_server_none":
                    # was the server's tick-N target alive at N+1?
                    e = real_n1.get(pre_target) if pre_target else None
                    if not pre_target:
                        alive = "server_held_nothing_at_N"
                    elif pre_target not in post_iv:
                        alive = "N_target_removed_at_N1"
                    elif e is not None and e.dead:
                        alive = "N_target_dead_at_N1"
                    else:
                        # still there: did it leave range?
                        tiv, siv = post_iv[pre_target], post_iv.get(c.net_id)
                        d = (math.hypot((tiv.q_x - siv.q_x) / PosQ,
                                        (tiv.q_y - siv.q_y) / PosQ)
                             if siv else float("nan"))
                        alive = ("N_target_alive_left_range" if d > TURRET_RANGE
                                 else "N_target_alive_in_range")
                    split[f"alive_at_N1|{alive}"] += 1
                    stale = (c.sim_target_net_id == pre_target)
                    split[f"sim_pick|{'stale_hold' if stale else 'new_unit'}"] += 1
                    split[f"joint|{alive}|{'stale' if stale else 'new'}"] += 1
                    # does the server re-acquire at N+2?
                    iv2 = post2_iv.get(c.net_id)
                    if iv2 is not None:
                        if iv2.target_net_id == c.sim_target_net_id:
                            split["N2|server_acquires_sims_unit"] += 1
                        elif iv2.target_net_id:
                            split["N2|server_acquires_other"] += 1
                        else:
                            split["N2|server_still_none"] += 1
                    split[f"pre_attacking|{int(iv0.is_attacking)}"] += 1
                    if len(ex) < examples:
                        ex.append(dict(
                            t_ms=sn.t_ms, turret=c.net_id, team=c.team,
                            server_target_at_N=pre_target,
                            sim_target_at_N1=c.sim_target_net_id,
                            server_target_at_N1=c.server_target_net_id,
                            server_target_at_N2=(iv2.target_net_id
                                                 if iv2 else None),
                            alive=alive, stale=stale,
                            pre_attacking=bool(iv0.is_attacking),
                            pre_aacd=iv0.q_aa_cooldown / StatQ))

            # ---- spurious fires ----------------------------------------
            if c.sim_fire and not c.server_fire:
                tid = c.sim_target_net_id
                tiv = pre_iv.get(tid) if tid else None
                fire_by[f"turret_team={c.team}|victim_team="
                        f"{tiv.team if tiv else '?'}"] += 1
                fire_by[f"victim_kind={tiv.kind if tiv else '?'}"] += 1
                fire_by["total"] += 1
            if c.server_fire and not c.sim_fire:
                fire_by["server_only"] += 1
            if c.server_fire and c.sim_fire:
                fire_by["both"] += 1

    out["phase_b"] = {
        "n_pairs": n_pairs,
        "split": dict(split),
        "spurious_fires": dict(fire_by),
        "examples": ex,
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--existing-log", required=True)
    ap.add_argument("--phase", choices=["a", "b", "both"], default="both")
    ap.add_argument("--from-ms", type=int, default=0)
    ap.add_argument("--to-ms", type=int, default=10 ** 9)
    ap.add_argument("--max-pairs", type=int, default=10 ** 9)
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args(argv)

    from .tier1_full import FIRST_WAVE_MS
    from ..trace import load_trace

    trace = load_trace(Path(a.existing_log))
    snaps = [s for s in trace.snapshots if s.t_ms >= FIRST_WAVE_MS]
    out = {"n_snapshots": len(snaps),
           "t_ms": [snaps[0].t_ms, snaps[-1].t_ms] if snaps else None}
    if a.phase in ("a", "both"):
        phase_a(snaps, out)
    if a.phase in ("b", "both"):
        phase_b(snaps, out, a.from_ms, a.to_ms, a.max_pairs)
    text = json.dumps(out, indent=2, sort_keys=True, default=str)
    print(text)
    if a.out_json:
        Path(a.out_json).write_text(text)


if __name__ == "__main__":
    main()
