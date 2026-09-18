"""Name the mechanism behind each one-step residual, instead of quoting its rate.

`tier1_full`'s aggregate says *which field* disagrees and how often.  It
deliberately does not say *what the disagreement looks like*, and a rate on
its own has repeatedly been enough to argue for the wrong fix here: "move
order 91%" is consistent with "the sim holds where the server chases" and
with "the sim chases where the server holds", and those have opposite causes.

This drills one tick-window and tabulates, for every scored field that is
short of 100%:

* **move_order / waypoints** -- the full (sim value, server value) confusion,
  not a scalar.  A confusion table names the direction of the error.
* **auto-attack fire tick** -- for every disagreeing unit, the distance to
  the server's OWN dumped target position against the sim's ``idealRange``
  (``Stats.Range.Total + TargetUnit.CollisionRadius``).  If the sim swings
  where the server does not, either it is closer or it thinks its reach is
  longer, and this is the number that separates those.
* **target identity** -- the same distance/range comparison for the unit the
  sim picked and the one the server picked.
* **hp** -- split by whether a missile was in flight at the injected tick,
  which is the known blind spot (RESET-004), so a missile-driven HP error is
  never reported as a damage-formula error.

    python -m lanerl_jax.parity.tier1_residual_drill --existing-log LOG \\
        --from-ms 124000 --to-ms 136000
"""
from __future__ import annotations

import argparse
import collections
import math
from pathlib import Path

import numpy as np

ORDER_NAMES = {0: "NONE", 1: "HOLD", 2: "MOVE_TO", 3: "ATTACK_TO",
               4: "ATTACK_MOVE", 5: "STOP", 6: "CAST_SPELL"}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--existing-log", required=True)
    ap.add_argument("--from-ms", type=int, default=90_000)
    ap.add_argument("--to-ms", type=int, default=10**9)
    ap.add_argument("--max-pairs", type=int, default=2000)
    ap.add_argument("--examples", type=int, default=6)
    a = ap.parse_args(argv)

    import jax.numpy as jnp

    from ..data.patch import load_patch
    from ..sim.init import TOP_LANE_PATH, lane_params
    from ..sim.profiles import PROFILES
    from .diagnostic_identity import net_id_to_injected_slot
    from .inject import inject_snapshot, replay_wave_states
    from .one_step import compare_one_tick
    from .tier1_full import FIRST_WAVE_MS
    from .trace import PosQ, StatQ, load_trace

    trace = load_trace(Path(a.existing_log))
    snaps = [s for s in trace.snapshots if s.t_ms >= FIRST_WAVE_MS]
    wave_states = replay_wave_states(snaps)
    patch = load_patch()
    params = lane_params(patch)
    lane_path = jnp.asarray(np.asarray(TOP_LANE_PATH, np.float32))
    attack_range = np.asarray(params["attack_range"])
    collision_radius = np.asarray(params["collision_radius"])

    order_confusion = collections.Counter()
    wp_confusion = collections.Counter()
    fire_rows = []
    target_rows = []
    hp_rows = []
    n_pairs = 0
    n_scored = collections.Counter()

    for i in range(len(snaps) - 1):
        sn, sn1 = snaps[i], snaps[i + 1]
        if sn.t_ms < a.from_ms:
            continue
        if sn.t_ms > a.to_ms or n_pairs >= a.max_pairs:
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

        has_missile = any(e.kind == "SpellMissile" for e in sn.entities)
        pre_internal = {iv.net_id: iv for iv in sn.ai_internals}
        model0 = np.asarray(state_n.model)
        x0 = np.asarray(state_n.x)
        y0 = np.asarray(state_n.y)
        note_by_slot = {n.slot: n for n in report.notes}

        for m in tr.matched:
            if m.pred.ai is None or m.real.ai is None:
                continue
            n_scored[f"{m.kind}.move_order"] += 1
            if m.pred.ai.move_order != m.real.ai.move_order:
                order_confusion[(m.kind,
                                 ORDER_NAMES.get(m.pred.ai.move_order,
                                                 m.pred.ai.move_order),
                                 ORDER_NAMES.get(m.real.ai.move_order,
                                                 m.real.ai.move_order))] += 1
            if m.movement_trustworthy and m.pred.ai.waypoints != m.real.ai.waypoints:
                wp_confusion[(m.kind, m.pred.ai.waypoints,
                              m.real.ai.waypoints)] += 1
            if m.pred.q_hp != m.real.q_hp:
                hp_rows.append((m.kind, has_missile,
                                (m.pred.hp or 0.0) - (m.real.hp or 0.0)))

        real_order_by_slot = {m.slot: (m.real.ai.move_order
                                       if m.real.ai is not None else None)
                              for m in tr.matched}
        slot_of = dict(net_id_slots)
        for c in tr.controller:
            slot = c.slot
            iv = pre_internal.get(c.net_id)
            if iv is None:
                continue
            rng = attack_range[model0[slot]]

            def geometry(target_slot, target_net):
                """Distance and idealRange for one candidate target, computed
                from the INJECTED pre-tick positions (identical on both sides
                by construction) so the two candidates are comparable."""
                if target_slot is None or target_slot < 0:
                    return None
                d = math.hypot(x0[target_slot] - x0[slot],
                               y0[target_slot] - y0[slot])
                ideal = rng + collision_radius[model0[target_slot]]
                return d, ideal

            if c.sim_fire != c.server_fire:
                g = geometry(slot_of.get(c.server_target_net_id),
                             c.server_target_net_id)
                fire_rows.append(dict(
                    t_ms=sn.t_ms, kind=c.kind, net=c.net_id,
                    sim_fire=c.sim_fire, server_fire=c.server_fire,
                    server_order=ORDER_NAMES.get(
                        real_order_by_slot.get(slot), "?"),
                    d=None if g is None else g[0],
                    ideal=None if g is None else g[1],
                    margin=None if g is None else g[0] - g[1],
                    target_agrees=c.sim_target_net_id == c.server_target_net_id,
                ))
            if c.sim_target_net_id != c.server_target_net_id:
                gs = geometry(slot_of.get(c.sim_target_net_id),
                              c.sim_target_net_id)
                gv = geometry(slot_of.get(c.server_target_net_id),
                              c.server_target_net_id)
                target_rows.append(dict(
                    t_ms=sn.t_ms, kind=c.kind, net=c.net_id,
                    sim_target=c.sim_target_net_id,
                    server_target=c.server_target_net_id,
                    sim_d=None if gs is None else gs[0],
                    sim_ideal=None if gs is None else gs[1],
                    server_d=None if gv is None else gv[0],
                    server_ideal=None if gv is None else gv[1],
                ))

    print(f"pairs drilled: {n_pairs} "
          f"(t={a.from_ms}..{min(a.to_ms, snaps[-1].t_ms)})")

    print("\n-- move_order confusion (sim -> server) --")
    for (kind, sim_v, srv_v), n in order_confusion.most_common(20):
        print(f"  {n:6d}  {kind}: sim={sim_v} server={srv_v}")

    print("\n-- waypoint-count confusion (sim -> server) --")
    for (kind, sim_v, srv_v), n in wp_confusion.most_common(20):
        print(f"  {n:6d}  {kind}: sim={sim_v} server={srv_v}")

    print("\n-- auto-attack fire-tick disagreements --")
    by_shape = collections.Counter()
    margins = collections.defaultdict(list)
    for r in fire_rows:
        shape = (r["kind"], r["sim_fire"], r["server_fire"],
                 r["target_agrees"])
        by_shape[shape] += 1
        if r["margin"] is not None:
            margins[shape].append(r["margin"])
    for shape, n in by_shape.most_common():
        kind, sf, vf, ta = shape
        ms = np.asarray(margins[shape]) if margins[shape] else None
        extra = ""
        if ms is not None and len(ms):
            extra = (f"; distance-to-server-target MINUS idealRange: "
                     f"median {np.median(ms):+.3f} "
                     f"p5 {np.percentile(ms, 5):+.3f} "
                     f"p95 {np.percentile(ms, 95):+.3f}; "
                     f"{100 * np.mean(ms <= 0):.1f}% already in range")
        print(f"  {n:6d}  {kind}: sim_fire={int(sf)} server_fire={int(vf)} "
              f"target_agrees={int(ta)}{extra}")
    for r in fire_rows[:a.examples]:
        print(f"    e.g. t={r['t_ms']} {r['kind']} net={r['net']} "
              f"sim_fire={int(r['sim_fire'])} server_fire={int(r['server_fire'])} "
              f"d={r['d']} ideal={r['ideal']} margin={r['margin']}")

    print("\n-- target-identity disagreements --")
    tshape = collections.Counter()
    for r in target_rows:
        tshape[(r["kind"], r["sim_target"] == 0, r["server_target"] == 0)] += 1
    for (kind, sim_none, srv_none), n in tshape.most_common():
        print(f"  {n:6d}  {kind}: sim_has_target={not sim_none} "
              f"server_has_target={not srv_none}")
    for r in target_rows[:a.examples]:
        print(f"    e.g. t={r['t_ms']} {r['kind']} net={r['net']}: "
              f"sim->{r['sim_target']} (d={r['sim_d']}, ideal={r['sim_ideal']}) "
              f"server->{r['server_target']} (d={r['server_d']}, "
              f"ideal={r['server_ideal']})")

    print("\n-- hp disagreements, split by the missile blind spot --")
    for kind in sorted({r[0] for r in hp_rows}):
        with_m = [r[2] for r in hp_rows if r[0] == kind and r[1]]
        without_m = [r[2] for r in hp_rows if r[0] == kind and not r[1]]
        print(f"  {kind}: {len(with_m)} with a missile in flight, "
              f"{len(without_m)} without")
        for label, rows in (("with missile", with_m), ("no missile", without_m)):
            if rows:
                v = np.abs(np.asarray(rows))
                print(f"    {label}: |error| median {np.median(v):.3f} "
                      f"max {v.max():.3f} mean signed "
                      f"{np.mean(np.asarray(rows)):+.3f}")


if __name__ == "__main__":
    main()
