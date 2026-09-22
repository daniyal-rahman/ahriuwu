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
import struct
from pathlib import Path

import numpy as np

ORDER_NAMES = {0: "NONE", 1: "HOLD", 2: "MOVE_TO", 3: "ATTACK_TO",
               4: "ATTACK_MOVE", 5: "STOP", 6: "CAST_SPELL"}


def _cfh_entries(cfh_pre, net_id, iv, tol_ms: int = 2):
    """The pre-clear call-for-help map for one unit at one tick, or None.

    ``iv.q_ai_local`` is the dump's `ailocal` in StatQ, and the event's key is
    the script's `localTime` in whole ms, so the join is approximate by
    construction; ``tol_ms`` is the slack. ``None`` means "no event found",
    which is NOT the same as "the map was empty" and must not be collapsed
    with it -- a minion that did not re-evaluate this tick emits no event at
    all.
    """
    if not cfh_pre or iv is None or iv.q_ai_local is None:
        return None
    lt = int(round(iv.q_ai_local / 1024.0))
    for d in range(0, tol_ms + 1):
        for cand in ((lt - d,) if d else (lt,)) + ((lt + d,) if d else ()):
            hit = cfh_pre.get((net_id, cand))
            if hit is not None:
                return hit
    return None


def _cfh_lookup(cfh_pre, net_id, iv):
    e = _cfh_entries(cfh_pre, net_id, iv)
    return None if e is None else bool(e)


def _cfh_has(cfh_pre, net_id, iv, pick):
    e = _cfh_entries(cfh_pre, net_id, iv)
    if e is None or not pick:
        return None
    return pick in e


def _drill_move_order(rows, denom) -> None:
    """Is the LaneMinion move-order residual a mechanism or a floor?

    Called from :func:`main` with one row per disagreeing minion-tick.  Every
    print here answers one of the four questions that separate the two:
    symmetry, concentration, containment, and observability.
    """
    print("\n-- move_order: the FULL LaneMinion confusion matrix --")
    labels = sorted({r["sim"] for r in rows} | {r["srv"] for r in rows})
    if not rows:
        print("   (no LaneMinion move-order disagreements in this window)")
        return
    print("   rows = sim, cols = server; the transpose pair of every cell is")
    print("   printed beside it, because a SKEW is a missing gate and a")
    print("   balanced pair is one-tick jitter")
    cell = collections.Counter((r["sim"], r["srv"]) for r in rows)
    width = max(len(s) for s in labels) + 1
    print("        " + "".join(f"{s:>{width + 6}}" for s in labels))
    for a in labels:
        print(f"   {a:>{width}} "
              + "".join(f"{cell[(a, b)]:>{width + 6}d}" for b in labels))
    print("   off-diagonal pairs (sim=A/server=B against sim=B/server=A):")
    seen = set()
    for a in labels:
        for b in labels:
            if a == b or (b, a) in seen:
                continue
            seen.add((a, b))
            n_ab, n_ba = cell[(a, b)], cell[(b, a)]
            if not (n_ab or n_ba):
                continue
            tot = n_ab + n_ba
            print(f"     {a:>12} <-> {b:<12} {n_ab:6d} vs {n_ba:6d}  "
                  f"(skew {100 * abs(n_ab - n_ba) / tot:5.1f}% of {tot})")

    print("\n-- move_order against the SHARED pre-tick order --")
    print("   The injected order is the server's own value (`inject.py` writes")
    print("   `ent.ai.move_order` every tick), so each side can be compared to")
    print("   the SAME baseline.  That is what separates 'the server wrote an")
    print("   order this tick and we did not' from the mirror -- exactly the")
    print("   one-sidedness test that made `ORDER-003` a floor (253 vs 0).")
    verdicts = collections.Counter()
    for r in rows:
        if r["sim"] == r["pre"]:
            verdicts[f"server WROTE {r['pre']}->{r['srv']}, sim held"] += 1
        elif r["srv"] == r["pre"]:
            verdicts[f"sim WROTE {r['pre']}->{r['sim']}, server held"] += 1
        else:
            verdicts[f"both wrote, differently ({r['pre']}: "
                     f"sim {r['sim']} / server {r['srv']})"] += 1
    for v, n in verdicts.most_common(12):
        print(f"     {n:6d}  {v}")
    held = sum(n for v, n in verdicts.items() if v.startswith("server WROTE"))
    wrote = sum(n for v, n in verdicts.items() if v.startswith("sim WROTE"))
    print(f"   server-first {held}  vs  sim-first {wrote}  "
          f"(a one-sided split is a phase floor, a balanced one is jitter)")
    # The decisive table. A lane minion has exactly TWO move-order writers
    # (`LaneMinionAI.cs:96` and `RefreshWaypoints`' `:604/:655`), and on a tick
    # the unit entered `IsAttacking` the second one is unreachable
    # (`ORDER-002`). So on those ticks the order is the 250 ms controller's
    # output or nothing at all, and (pre -> sim/server) x (did each side
    # sweep) says which of the two it was without any further inference.
    print("   (pre -> sim / server) x (sim's timer rule, server's own reset):")
    print("   `sweep_pred` is the 250 ms TIMER only; the sim also re-evaluates")
    print("   on `TargetJustDied()`/call-for-help, which the timer cannot see,")
    print("   so pred=False with a sim write is an EVENT-triggered sweep")
    big = collections.Counter(
        (r["pre"], r["sim"], r["srv"], r["sweep_predicted"],
         r["sweep_observed"]) for r in rows)
    for (pre, si, sv, p, o), n in big.most_common(14):
        who = ("sim wrote" if si != pre and sv == pre else
               "server wrote" if sv != pre and si == pre else
               "both wrote" if si != pre and sv != pre else
               "NEITHER wrote (impossible: injected orders are equal)")
        print(f"     {n:6d}  pre={pre:<10} sim={si:<10} server={sv:<10} "
              f"sweep_pred={str(p):<5} server_swept={str(o):<5}  {who}")

    print("\n-- move_order: is it CONCENTRATED? --")
    print("   by minion class, against that class's own scored denominator:")
    for cls in sorted({r["cls"] for r in rows}):
        sub = [r for r in rows if r["cls"] == cls]
        d = denom[cls]
        c2 = collections.Counter((r["sim"], r["srv"]) for r in sub)
        top = c2.most_common(2)
        shape = "; ".join(f"sim={a}/server={b} {n}" for (a, b), n in top)
        print(f"     {cls:>8}: {len(sub):6d} / {d:7d} scored "
              f"({100 * len(sub) / max(1, d):.3f}%)   {shape}")
    print("   by game phase (60 s buckets), against the same denominator:")
    bucket_n = collections.Counter(r["t_ms"] // 60_000 for r in rows)
    for b in sorted(bucket_n):
        sub = [r for r in rows if r["t_ms"] // 60_000 == b]
        c2 = collections.Counter((r["sim"], r["srv"]) for r in sub)
        a1 = sum(n for (x, y), n in c2.items() if x == "HOLD")
        a2 = sum(n for (x, y), n in c2.items() if y == "HOLD")
        print(f"     t={b * 60:4d}-{b * 60 + 60:4d}s: {bucket_n[b]:6d}  "
              f"sim-said-HOLD {a1}  server-said-HOLD {a2}")
    print("   by swing phase (from the server's own aastate/windup/cooldown):")
    for ph, n in collections.Counter(r["phase"] for r in rows).most_common():
        sub = [r for r in rows if r["phase"] == ph]
        c2 = collections.Counter((r["sim"], r["srv"]) for r in sub)
        top = "; ".join(f"{a}->{b} {k}" for (a, b), k in c2.most_common(2))
        print(f"     {n:6d}  {ph}: {top}")
    print("   pre-tick (is_attacking, AutoAttackSpell.State) on the disagreeing")
    print("   tick -- `State == STATE_READY` is the server's own extra gate on")
    print("   the in-range RefreshWaypoints (`ObjAIBase.cs:1233-1242`):")
    for (at, st), n in collections.Counter(
            (r["pre_attacking"], r["pre_aa_state"]) for r in rows).most_common(6):
        print(f"     {n:6d}  is_attacking={at} aastate={st}")
    print("   did the 250 ms sweep run on this tick?  (predicted from the")
    print("   dump's own aitimer, observed from its reset at N+1):")
    for (p, o), n in collections.Counter(
            (r["sweep_predicted"], r["sweep_observed"])
            for r in rows).most_common():
        print(f"     {n:6d}  sim-rule predicted={p}  server observed={o}")

    print("\n-- move_order: the in-range MARGIN, `d - idealRange` --")
    print("   `RefreshWaypoints` picks Hold over AttackTo on the sign of this")
    print("   number.  The server evaluates it MID-tick; the dump can only")
    print("   show it at N and at N+1.  A residual whose margin straddles zero")
    print("   between those two instants is the evaluation INSTANT, not the")
    print("   rule -- i.e. `ORDER-003`'s floor showing up in this field.")
    both = [r for r in rows
            if r["margin_pre"] is not None and r["margin_post"] is not None]
    print(f"   {len(both)} of {len(rows)} rows have a mappable target at both "
          f"instants")
    if both:
        pre = np.asarray([r["margin_pre"] for r in both])
        post = np.asarray([r["margin_post"] for r in both])
        straddle = (np.sign(pre) != np.sign(post))
        print(f"     |margin| at N:   median {np.median(np.abs(pre)):8.3f} u  "
              f"p25 {np.percentile(np.abs(pre), 25):8.3f}  "
              f"p75 {np.percentile(np.abs(pre), 75):8.3f}")
        print(f"     |margin| at N+1: median {np.median(np.abs(post)):8.3f} u  "
              f"p25 {np.percentile(np.abs(post), 25):8.3f}  "
              f"p75 {np.percentile(np.abs(post), 75):8.3f}")
        print(f"     the in-range verdict FLIPS between N and N+1 on "
              f"{int(straddle.sum())}/{len(both)} "
              f"({100 * straddle.mean():.1f}%)")
        # A minion's per-tick budget is 325 u/s * 1024/60 ms = 5.417 u, so a
        # margin inside one step is reachable by a one-tick phase difference
        # and one outside it is not.  That is the line between "jitter" and
        # "wrong rule", and it is a measured quantity, not a judgement.
        step = 325.0 * (1024.0 / 60.0) / 1000.0
        for lab, v in (("N", pre), ("N+1", post)):
            print(f"     at {lab}: {100 * np.mean(np.abs(v) <= step):5.1f}% "
                  f"within ONE minion movement step ({step:.3f} u) of the "
                  f"in-range boundary")

    print("\n-- move_order: is it a SHADOW of another residual? --")
    print("   containment on the SAME (net_id, tick), the join that collapsed")
    print("   `has_auto_attacked` into `AA-002`:")
    for key in ("target_x", "fire_x", "hit_x", "attacking_x", "wp_x", "pos_x"):
        n = sum(1 for r in rows if r.get(key))
        print(f"     {n:6d} / {len(rows)}  ({100 * n / len(rows):5.1f}%) also "
              f"disagreed on {key[:-2]}")
    any_other = sum(1 for r in rows
                    if any(r.get(k) for k in ("target_x", "fire_x", "hit_x",
                                              "attacking_x", "wp_x", "pos_x")))
    print(f"     {any_other} / {len(rows)} ({100 * any_other / len(rows):.1f}%) "
          f"share a tick with ANY other scored disagreement; "
          f"{len(rows) - any_other} are move-order ONLY")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--existing-log", required=True)
    ap.add_argument("--from-ms", type=int, default=90_000)
    ap.add_argument("--to-ms", type=int, default=10**9)
    ap.add_argument("--max-pairs", type=int, default=2000)
    ap.add_argument("--examples", type=int, default=6)
    ap.add_argument("--cfh-log", default=None,
                    help="a recording carrying `CallForHelpClear` events (the "
                        "pre-clear call-for-help map). Turns CFH-002 from an "
                        "inferred signature into a direct test of whether the "
                        "map actually contained the server's pick.")
    ap.add_argument("--decision-log", default=None,
                    help="a LANERL_DECISION_TRACE=1 recording of the SAME "
                        "seed/config, for a ground-truth ORDER-003 check: "
                        "does the server's own branch log show another "
                        "unit's event EARLIER in the SAME tick than this "
                        "minion's SetTargetUnit? State diff alone cannot "
                        "see intra-tick order; the branch log can.")
    a = ap.parse_args(argv)

    # ---- CFH-002's GROUND TRUTH: the pre-clear call-for-help map ---------
    # `CallForHelpClear` (added 2026-09-21) emits `unitsAttackingAllies`
    # immediately BEFORE `LaneMinionAI` wipes it, i.e. exactly the map
    # `FoundNewTarget` just read. Its `t=` is the script's own `localTime`
    # (time since THAT minion spawned), not game time, because a Content
    # script cannot reach `_game` (`GameObject._game` is protected). The dump
    # publishes each minion's `ailocal=` every tick, so (netid, localTime)
    # aligns to a game tick -- verified on a real event: localTime 30666 for
    # id 1073743694 lands on game tick 120660 where ailocal=31402226, i.e.
    # 30666*1024 to within 242 quanta (0.24 ms).
    #
    # This is what turns `CFH-002` from an inferred signature into a direct
    # test: for a disagreeing target row, was the chooser's pre-clear map
    # actually non-empty, and did it actually contain the unit the server
    # picked?
    cfh_pre = {}
    if a.cfh_log:
        import re as _re
        pat = _re.compile(
            r"LANERL_DECISION t=(\d+) k=CallForHelpClear id=(\d+) n=(\d+)(.*)")
        n_ev = n_nonempty = 0
        with open(a.cfh_log, errors="replace") as fh:
            for line in fh:
                m = pat.search(line)
                if m is None:
                    continue
                n_ev += 1
                lt, nid, cnt, rest = (int(m.group(1)), int(m.group(2)),
                                      int(m.group(3)), m.group(4).strip())
                entries = {}
                if cnt and rest:
                    for tok in rest.split(";"):
                        if ":" in tok:
                            k, v = tok.split(":", 1)
                            try:
                                entries[int(k)] = int(v)
                            except ValueError:
                                pass
                if entries:
                    n_nonempty += 1
                cfh_pre[(nid, lt)] = entries
        print(f"pre-clear CFH maps: {n_ev} events, {n_nonempty} non-empty "
              f"({100 * n_nonempty / max(1, n_ev):.2f}%)", flush=True)

    dec_by_t = None
    if a.decision_log:
        from .decision_trace import parse_decisions
        print(f"loading decision trace: {a.decision_log}", flush=True)
        decisions = parse_decisions(a.decision_log)
        dec_by_t = collections.defaultdict(list)
        for d in decisions:
            dec_by_t[d.t_ms].append(d)
        print(f"  {len(decisions)} branch events over {len(dec_by_t)} ticks",
              flush=True)

    import jax.numpy as jnp

    from ..data.patch import load_patch
    from ..sim.init import TOP_LANE_PATH, lane_params
    from ..sim.profiles import PROFILES
    from .diagnostic_identity import net_id_to_injected_slot
    from .inject import inject_snapshot, replay_wave_states
    from .one_step import compare_one_tick
    from ..sim.minion_ai import ACTION_TIMER_MS
    from ..sim.profiles import PROFILES as PROFILE_SPECS
    from ..sim.state import Kind
    from ..sim.targeting import MinionType
    from .tier1_full import FIRST_WAVE_MS
    from .trace import PosQ, StatQ, load_trace_window

    _SUBTYPE_NAME = {MinionType.MELEE: "melee", MinionType.CASTER: "caster",
                     MinionType.CANNON: "cannon", MinionType.SUPER: "super"}

    def _class_of(model_row: int) -> str:
        kind, subtype, _team = PROFILE_SPECS[int(model_row)]
        if kind != Kind.LANE_MINION:
            return "non-minion"
        return _SUBTYPE_NAME.get(subtype, f"subtype{subtype}")

    # Windowed: the full parse builds an `Entity` for every STATEROW in a
    # ~575 MB dump, which is where this drill's 26 GB went -- `--max-pairs`
    # bounded the loop and not the parse, so even a 2,000-pair slice OOMed.
    # `load_trace_window` keeps index alignment and keeps `t_ms` on every
    # tick (which is all `replay_wave_states` reads), so the wave replay is
    # unchanged; only the rows outside the window stop being materialised.
    trace = load_trace_window(Path(a.existing_log), from_ms=a.from_ms,
                              to_ms=a.to_ms, max_snapshots=a.max_pairs + 1)
    snaps = [s for s in trace.snapshots if s.t_ms >= FIRST_WAVE_MS]
    wave_states = replay_wave_states(snaps)
    patch = load_patch()
    params = lane_params(patch)
    lane_path = jnp.asarray(np.asarray(TOP_LANE_PATH, np.float32))
    attack_range = np.asarray(params["attack_range"])
    collision_radius = np.asarray(params["collision_radius"])
    missile_speed = np.asarray(params["missile_speed"])
    pathfinding_radius = np.asarray(params["pathfinding_radius"])
    acquisition_range = np.asarray(params["acquisition_range"])

    def _team_of(model_row: int):
        _kind, _subtype, team = PROFILE_SPECS[int(model_row)]
        return int(team)

    order_confusion = collections.Counter()
    order_rows = []
    order_denom = collections.Counter()
    wp_confusion = collections.Counter()
    wp_rows_keyed = []
    fire_rows = []
    target_rows = []
    hp_rows = []
    hp_detail = []
    pos_rows = []
    hit_rows = []
    missile_census = collections.Counter()
    sweep_census = collections.Counter()
    crowd_scored = collections.Counter()
    crowd_missed = collections.Counter()
    hit_denom = collections.Counter()
    hit_miss = collections.Counter()
    bool_join = collections.Counter()
    cd_gap = []
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
        order0 = np.asarray(state_n.move_order)
        nwp0 = np.asarray(state_n.n_waypoints)
        cx0 = np.asarray(state_n.collision_x)
        cy0 = np.asarray(state_n.collision_y)
        cpres0 = np.asarray(state_n.collision_present)
        wpkey0 = np.asarray(state_n.waypoint_key)
        note_by_slot = {n.slot: n for n in report.notes}
        slot_of = dict(net_id_slots)
        net_of_slot = {slot: net for net, slot in slot_of.items()}
        live_slots = [n.slot for n in report.notes
                      if n.entity is not None and not n.entity.dead]
        dump_nwp_by_slot = {n.slot: n.entity.ai.waypoints
                            for n in report.notes
                            if n.entity is not None and n.entity.ai is not None}

        # -- AI sweep phase, predicted from the DUMP ALONE ------------------
        # `LaneMinionAI.OnUpdate` runs `ReevaluateBehavior` when
        # `minionActionTimer >= 250f` after `+= delta`, and resets the timer to
        # 0 when it does. `aitimer` is in the diagnostic stream on both ticks,
        # so whether the sim's rule predicts the server's own sweeps is
        # answerable without running the sim at all -- which matters, because
        # a target disagreement caused by sweeping on a different tick is a
        # phase bug and one caused by a different scan is not.
        nxt_internal = {jv.net_id: jv for jv in sn1.ai_internals}
        for net_id, jv in nxt_internal.items():
            pv = pre_internal.get(net_id)
            if pv is None or jv.kind != "LaneMinion":
                continue
            if pv.q_ai_timer is None or jv.q_ai_timer is None:
                continue
            t0 = pv.q_ai_timer / StatQ
            predicted = (t0 + dt) >= ACTION_TIMER_MS
            observed = jv.q_ai_timer < pv.q_ai_timer
            sweep_census[(predicted, observed)] += 1

        # -- the missile population, at N and at N+1 -----------------------
        # `inject.py` restores missiles from the diagnostic stream, so a
        # missile is only "unobservable" when its owner or target does not map
        # to an injected slot and the injector therefore DROPS it. Those two
        # cases have opposite verdicts and the old `has_missile` flag merges
        # them, so they are counted separately here.
        mis_n = list(sn.missile_internals)
        mis_n1_ids = {mi.net_id for mi in sn1.missile_internals}
        missiles_by_target = collections.defaultdict(list)
        for mi in mis_n:
            missiles_by_target[mi.target_net_id].append(mi)
        for mi in mis_n:
            injectable = (slot_of.get(mi.owner_net_id) is not None
                          and slot_of.get(mi.target_net_id) is not None)
            missile_census["total"] += 1
            missile_census["injected" if injectable else "DROPPED"] += 1
            if mi.net_id not in mis_n1_ids:
                missile_census["landed_or_expired_this_tick"] += 1

        ctrl_by_slot = {c.slot: c for c in tr.controller}

        for m in tr.matched:
            if m.pred.ai is None or m.real.ai is None:
                continue
            n_scored[f"{m.kind}.move_order"] += 1
            net = net_of_slot.get(m.slot)
            iv = pre_internal.get(net) if net is not None else None
            wp_x = bool(m.movement_trustworthy
                        and m.pred.ai.waypoints != m.real.ai.waypoints)
            if m.pred.ai.move_order != m.real.ai.move_order:
                order_confusion[(m.kind,
                                 ORDER_NAMES.get(m.pred.ai.move_order,
                                                 m.pred.ai.move_order),
                                 ORDER_NAMES.get(m.real.ai.move_order,
                                                 m.real.ai.move_order))] += 1
            if wp_x:
                wp_confusion[(m.kind, m.pred.ai.waypoints,
                              m.real.ai.waypoints)] += 1
                # keyed copy, so the containment report below can ask whether
                # this row is the SAME event as a target/fire disagreement
                # rather than an independent one
                wp_rows_keyed.append(dict(t_ms=sn.t_ms,
                                          net=net_of_slot.get(m.slot),
                                          kind=m.kind))

            # ------------- move_order residual (the 4,791) ----------------
            # `ORDER-002` took this row from 34,295 to 4,807 and `HOLD-001`
            # to 4,791, and the belief carried in the ledger since is that
            # what is left is one-tick jitter at the swing boundary -- a
            # belief formed on ONE drill window (120 cases against 116 of the
            # mirror shape). This block is what tests it at corpus scale.
            # Three things separate the three candidate explanations:
            #   * the SYMMETRY of the confusion, per class and per phase (a
            #     skewed cell is a missing gate, a balanced one is jitter);
            #   * the MARGIN `d - idealRange` evaluated both at tick N and at
            #     tick N+1, because the in-range predicate is exactly what
            #     chooses Hold over AttackTo and the server evaluates it
            #     mid-tick, between those two observable instants.  A miss
            #     whose margin STRADDLES zero across the tick is not a wrong
            #     rule, it is a rule read at a different instant -- the same
            #     serial-vs-fixed-phase floor `ORDER-003` books for target;
            #   * CONTAINMENT in the other residual rows on the same
            #     unit-tick, which is what collapsed `has_auto_attacked` and
            #     `is_attacking` into `AA-002`.
            if m.kind == "LaneMinion":
                cls = _class_of(model0[m.slot])
                order_denom[cls] += 1
                order_denom["all"] += 1
            if (m.kind == "LaneMinion"
                    and m.pred.ai.move_order != m.real.ai.move_order):
                c = ctrl_by_slot.get(m.slot)
                jv = nxt_internal.get(net)
                # The in-range predicate, at the two instants the dump can
                # see. `idealRange = Stats.Range.Total + TargetUnit
                # .CollisionRadius` -- the same expression `RefreshWaypoints`
                # is handed by `UpdateTarget`.
                def _margin(sx, sy, tnet, tq):
                    tslot = slot_of.get(tnet) if tnet else None
                    if tslot is None or tq is None:
                        return None
                    ideal = float(attack_range[model0[m.slot]]
                                  + collision_radius[model0[tslot]])
                    return math.hypot(tq[0] / PosQ - sx,
                                      tq[1] / PosQ - sy) - ideal
                mar_pre = mar_post = None
                if iv is not None and iv.target_net_id:
                    mar_pre = _margin(m.pre_x, m.pre_y, iv.target_net_id,
                                      (iv.target_q_x, iv.target_q_y))
                if jv is not None and jv.target_net_id:
                    mar_post = _margin(m.real.x, m.real.y, jv.target_net_id,
                                       (jv.target_q_x, jv.target_q_y))
                # Where in the swing is this tick?  The dump carries the
                # server's own `AutoAttackSpell.State` (`aastate`), its
                # remaining windup and its cooldown, so "one tick either side
                # of the swing boundary" is a measured bucket, not a guess.
                # `aawindup`/`aacd` are dumped in SECONDS (`Q(x, StatQ)`),
                # `dt` is the tick in MILLISECONDS -- the two must be brought
                # to the same unit or every bucket collapses onto
                # `is_attacking`.
                dt_s = dt / 1000.0
                if iv is None:
                    phase = "?"
                elif iv.is_attacking and iv.q_aa_windup is not None and \
                        iv.q_aa_windup / StatQ <= dt_s:
                    phase = "mid-swing, windup COMPLETES this tick"
                elif iv.is_attacking:
                    phase = "mid-swing, windup continues"
                elif iv.q_aa_cooldown / StatQ <= dt_s:
                    phase = "idle, cooldown ready (may fire this tick)"
                else:
                    phase = "idle, in cooldown"
                order_rows.append(dict(
                    t_ms=sn.t_ms, net=net, cls=cls,
                    sim=ORDER_NAMES.get(m.pred.ai.move_order, "?"),
                    srv=ORDER_NAMES.get(m.real.ai.move_order, "?"),
                    pre=ORDER_NAMES.get(int(order0[m.slot]), "?"),
                    phase=phase,
                    pre_attacking=None if iv is None else iv.is_attacking,
                    pre_aa_state=None if iv is None else iv.aa_state,
                    sweep_predicted=(
                        None if iv is None or iv.q_ai_timer is None
                        else (iv.q_ai_timer / StatQ + dt) >= ACTION_TIMER_MS),
                    sweep_observed=(
                        None if iv is None or jv is None
                        or iv.q_ai_timer is None or jv.q_ai_timer is None
                        else jv.q_ai_timer < iv.q_ai_timer),
                    margin_pre=mar_pre, margin_post=mar_post,
                    target_x=(None if c is None
                              else c.sim_target_net_id != c.server_target_net_id),
                    fire_x=None if c is None else c.sim_fire != c.server_fire,
                    hit_x=None if c is None else c.sim_hit != c.server_hit,
                    attacking_x=(None if c is None
                                 else c.sim_attacking != c.server_attacking),
                    wp_x=wp_x,
                    pos_x=(m.movement_trustworthy
                           and max(abs(m.pred.q_x - m.real.q_x),
                                   abs(m.pred.q_y - m.real.q_y)) > 1),
                ))

            # ---------------- position residual (the 1,742) ---------------
            # Scored on the SAME gate tier1_full uses -- componentwise
            # (L-inf) on the dump's own quantised integers -- so this drill's
            # population is that corpus row's population and not a stricter
            # or looser one.
            if m.kind == "LaneMinion" and m.movement_trustworthy:
                n_scored["LaneMinion.position_linf"] += 1
                linf = max(abs(m.pred.q_x - m.real.q_x),
                           abs(m.pred.q_y - m.real.q_y))
                # The DENOMINATOR, conditioned on crowding. A residual that is
                # 0.2% of all pairs and 8% of crowded ones is a crowding
                # result; quoting only the first hides which it is, and this
                # module exists because a rate without its population has
                # argued for the wrong fix here before.
                near_all = 0
                for oslot in live_slots:
                    if oslot == m.slot:
                        continue
                    rr = float(collision_radius[model0[m.slot]]
                               + collision_radius[model0[oslot]])
                    if ((x0[oslot] - m.pre_x) ** 2
                            + (y0[oslot] - m.pre_y) ** 2) < rr * rr:
                        near_all += 1
                # `collision.resolve_collisions` takes its candidate list
                # from the FROZEN quadtree -- live query centre against each
                # candidate's last-`UpdateQuadTree` position
                # (`_frozen_candidates`). A neighbour that overlaps live but
                # is not in the frozen list can never be escaped from, and
                # that is a different bug from getting the escape wrong.
                near_frozen = 0
                for oslot in live_slots:
                    if oslot == m.slot or not bool(cpres0[oslot]):
                        continue
                    rr = float(collision_radius[model0[m.slot]]
                               + collision_radius[model0[oslot]])
                    if ((cx0[oslot] - m.pre_x) ** 2
                            + (cy0[oslot] - m.pre_y) ** 2) < rr * rr:
                        near_frozen += 1
                bucket = min(near_all, 3)
                crowd_scored[bucket] += 1
                if linf > 1:
                    crowd_missed[bucket] += 1
                    sim_step = math.hypot(m.pred.x - m.pre_x,
                                          m.pred.y - m.pre_y)
                    srv_step = math.hypot(m.real.x - m.pre_x,
                                          m.real.y - m.pre_y)
                    still = 0.05
                    if sim_step > still and srv_step <= still:
                        shape = "sim STEPPED, server STOOD"
                    elif srv_step > still and sim_step <= still:
                        shape = "sim STOOD, server STEPPED"
                    elif sim_step <= still and srv_step <= still:
                        shape = "both stood (position error predates the tick)"
                    else:
                        shape = "both stepped"
                    # The INJECTED movement state, which is the whole input to
                    # `AttackableUnit.Move` on the sim side. A one-step
                    # position error with an exact injection can only come
                    # from this triple or from the collision pass, so both are
                    # carried rather than inferred.
                    tslot = slot_of.get(
                        iv.target_net_id) if iv is not None else None
                    d_t = ideal_t = None
                    if tslot is not None:
                        d_t = math.hypot(x0[tslot] - m.pre_x,
                                         y0[tslot] - m.pre_y)
                        ideal_t = float(attack_range[model0[m.slot]]
                                        + collision_radius[model0[tslot]])
                    near = near_all
                    # Does the server's own displacement MATCH the escape
                    # `GetCircleEscapePoint` would produce against its
                    # earliest-created overlapping neighbour? For a minion
                    # pair the resolution radii sum to 35.7437+1+35.7437 =
                    # 72.49 against an 80 u trigger, so an overlapping pair is
                    # pulled TOWARDS contact, not pushed apart -- and that
                    # sign is the thing a "push-apart" intuition gets wrong.
                    best = None
                    for oslot in live_slots:
                        if oslot == m.slot:
                            continue
                        rr = float(collision_radius[model0[m.slot]]
                                   + collision_radius[model0[oslot]])
                        dd = math.hypot(x0[oslot] - m.pre_x,
                                        y0[oslot] - m.pre_y)
                        if dd < rr and (best is None or dd < best[0]):
                            best = (dd, oslot)
                    pred_escape = pred_cos = None
                    if best is not None:
                        dd, oslot = best
                        push = (dd - (pathfinding_radius[model0[m.slot]] + 1.0)
                                - pathfinding_radius[model0[oslot]])
                        pred_escape = float(push)
                        if dd > 0 and srv_step > 1e-6:
                            ux = (x0[oslot] - m.pre_x) / dd
                            uy = (y0[oslot] - m.pre_y) / dd
                            sx = (m.real.x - m.pre_x) / srv_step
                            sy = (m.real.y - m.pre_y) / srv_step
                            pred_cos = float(ux * sx + uy * sy)
                    pos_rows.append(dict(
                        t_ms=sn.t_ms, net=net, shape=shape,
                        linf=linf / PosQ, sim_step=sim_step, srv_step=srv_step,
                        sim_order=ORDER_NAMES.get(m.pred.ai.move_order, "?"),
                        srv_order=ORDER_NAMES.get(m.real.ai.move_order, "?"),
                        pre_order=ORDER_NAMES.get(
                            int(order0[m.slot]), "?"),
                        pre_attacking=None if iv is None else iv.is_attacking,
                        pre_aa_state=None if iv is None else iv.aa_state,
                        pre_target=None if iv is None else iv.target_net_id,
                        inj_n_wp=int(nwp0[m.slot]),
                        inj_key=int(wpkey0[m.slot]),
                        dump_n_wp=dump_nwp_by_slot.get(m.slot),
                        srv_post_n_wp=(m.real.ai.waypoints
                                       if m.real.ai else None),
                        colliding_neighbours=near,
                        frozen_neighbours=near_frozen,
                        pred_escape=pred_escape, pred_cos=pred_cos,
                        d_to_target=d_t, ideal_range=ideal_t,
                        in_range=(None if d_t is None else d_t <= ideal_t),
                    ))

            # ---------------- hp residual (the 1,127) ---------------------
            if m.pred.q_hp != m.real.q_hp:
                inbound = missiles_by_target.get(net, []) if net else []
                landed = [mi for mi in inbound if mi.net_id not in mis_n1_ids]
                dropped = [mi for mi in inbound
                           if slot_of.get(mi.owner_net_id) is None
                           or slot_of.get(mi.target_net_id) is None]
                hp_rows.append((m.kind, has_missile,
                                (m.pred.hp or 0.0) - (m.real.hp or 0.0)))
                hp_detail.append(dict(
                    t_ms=sn.t_ms, kind=m.kind, net=net,
                    delta=(m.pred.hp or 0.0) - (m.real.hp or 0.0),
                    any_missile=has_missile,
                    inbound=len(inbound), inbound_landed=len(landed),
                    inbound_dropped=len(dropped),
                ))

        real_order_by_slot = {m.slot: (m.real.ai.move_order
                                       if m.real.ai is not None else None)
                              for m in tr.matched}
        # Post-tick server Entity by PRE-tick slot -- the `ORDER-003`
        # containment check below needs both the pre-tick (`note_by_slot`,
        # from the injected snapshot) and post-tick server hp/dead/position
        # for any unit a target row names, not just the minion itself.
        real_by_slot = {m.slot: m.real for m in tr.matched}
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

            rng_cls = "ranged" if missile_speed[model0[slot]] > 0 else "melee"
            if c.kind == "LaneMinion":
                if c.server_hit:
                    hit_denom[rng_cls] += 1
                if c.sim_hit != c.server_hit:
                    hit_miss[rng_cls] += 1
                # -- the aa_cooldown / boolean joins -----------------------
                # Every one of these is scored on the SAME unit-tick, so
                # "is this residual already counted by another row" is a
                # containment question and not an estimate.
                fire_x = c.sim_fire != c.server_fire
                hit_x = c.sim_hit != c.server_hit
                if c.sim_aa_cooldown_q != c.server_aa_cooldown_q:
                    cd_gap.append((abs(c.sim_aa_cooldown_q
                                       - c.server_aa_cooldown_q),
                                   fire_x or hit_x))
                if c.sim_attacking != c.server_attacking:
                    bool_join["is_attacking",
                              "covered" if (fire_x or hit_x) else "UNCOVERED"] += 1
                if (c.sim_has_auto_attacked
                        != c.server_has_auto_attacked):
                    bool_join["has_auto_attacked",
                              "covered" if (fire_x or hit_x) else "UNCOVERED"] += 1
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
                    # AA-004: was the INJECTED cooldown sitting on the dump's
                    # gate clamp?  `LanerlAim.AutoAttackCooldownRemaining`
                    # publishes `Q(Math.Max(0f, remaining), StatQ)` -- the
                    # clamp runs BEFORE the quantisation, so a cooldown that is
                    # still strictly positive but within rounding of the gate
                    # publishes as a flat 0.  Injecting that 0 tells the sim
                    # the swing is READY when on the server it was not, and the
                    # sim fires a tick early.  That is the one-sided direction
                    # observed (sim_fire=1/server_fire=0 dominating ~40:1), and
                    # it is the SAME clamp that makes `snap_cooldown_to_tick_
                    # grid` refuse to touch a dumped zero.
                    injected_cd_q=(None if iv is None
                                   else iv.q_aa_cooldown),
                    # The EXACT unclamped cooldown, when the recording carries
                    # it (`aacdbits`, added for `AA-004`). This is what decides
                    # between the two readings of a dumped 0: a strictly
                    # POSITIVE exact residue means the server genuinely was not
                    # ready and the clamp hid it (an instrument blind spot),
                    # while a zero or NEGATIVE one means the server WAS ready
                    # and declined to fire for some other reason -- which would
                    # be a missing gate in the simulator, not a dump artifact.
                    # `AutoAttackSpell.State` and the attacking flag, from the
                    # dump. The server's swing needs the FULL chain
                    # (`ObjAIBase.cs:1254-1290`): target valid, inside
                    # `Stats.Range.Total + TargetUnit.CollisionRadius`,
                    # `MovementParameters == null`, **`AutoAttackSpell.State ==
                    # STATE_READY`**, `CanAttack()`, `_autoAttackCurrentCooldown
                    # <= 0`, `!_skipNextAutoAttack`. A cooldown of exactly 0 IS
                    # ready by that `<= 0`, so a row with an exact 0 or negative
                    # residue is NOT the clamp hiding anything -- some OTHER gate
                    # held the server back, and the spell state is the first
                    # place to look. SpellState: 0 READY, 1 CASTING,
                    # 2 COOLDOWN, 3 CHANNELING.
                    injected_aa_state=None if iv is None else iv.aa_state,
                    injected_attacking=None if iv is None else iv.is_attacking,
                    injected_cd_exact=(
                        None if iv is None or iv.aa_cooldown_bits is None
                        else struct.unpack("<f", struct.pack(
                            "<i", iv.aa_cooldown_bits))[0]),
                ))
            if c.sim_target_net_id != c.server_target_net_id:
                gs = geometry(slot_of.get(c.sim_target_net_id),
                              c.sim_target_net_id)
                gv = geometry(slot_of.get(c.server_target_net_id),
                              c.server_target_net_id)
                # The pre-tick target is the SAME value on both sides by
                # construction (`RESET-001` injects the server's own NetId),
                # so comparing each post-tick value against it separates a
                # *retention* failure from an *acquisition* failure. Reporting
                # only the post-tick pair cannot tell those apart, and they
                # have different branches in `minion_ai.py`.
                pre_t = 0 if iv is None else iv.target_net_id
                # Did the 250 ms timer alone predict a sweep on this tick?
                # `LaneMinionAI.OnUpdate`'s trigger is
                # `TargetJustDied() || FoundNewTarget(true) || timer >= 250`,
                # so separating "the sim did not sweep at all" from "the sim
                # swept and its scan chose differently" is the difference
                # between a trigger bug and a scan bug.
                sw_pred = (None if iv is None or iv.q_ai_timer is None
                           else (iv.q_ai_timer / StatQ + dt) >= ACTION_TIMER_MS)
                sim_t, srv_t = c.sim_target_net_id or 0, c.server_target_net_id
                if pre_t and srv_t == pre_t and not sim_t:
                    verdict = "sim DROPPED an incumbent the server kept"
                elif not pre_t and srv_t and not sim_t:
                    verdict = "server ACQUIRED from none, sim did not"
                elif pre_t and sim_t == pre_t and not srv_t:
                    verdict = "server DROPPED an incumbent the sim kept"
                elif not pre_t and sim_t and not srv_t:
                    verdict = "sim ACQUIRED from none, server did not"
                elif pre_t and srv_t and srv_t != pre_t and sim_t == pre_t:
                    verdict = "server SWITCHED off the incumbent, sim held"
                elif pre_t and sim_t and sim_t != pre_t and srv_t == pre_t:
                    verdict = "sim SWITCHED off the incumbent, server held"
                else:
                    verdict = "both moved, to different units"

                # -------------- ORDER-003: an INDEPENDENT floor size -------
                # The floor claim is about SERIAL evaluation order: a minion
                # later in `ObjectManager`'s per-tick foreach can see a kill
                # or a call-for-help broadcast that a minion earlier in that
                # same foreach cannot, because `step.py` evaluates every
                # minion controller at once on TICK-START state. That is only
                # a *possible* explanation on a tick where something the
                # scan reads actually changed between the injected snapshot
                # (`sn`, tick-start) and the next one (`sn1`, tick-end):
                #   * the unit this row is ABOUT (the server's new/dropped
                #     target) died, took damage, or crossed the acquisition
                #     boundary this same tick -- `TargetJustDied()`/vision/
                #     range are all evaluated on LIVE state mid-tick; or
                #   * ANY ally of this minion within ITS OWN acquisition
                #     range took damage this tick -- `TakeDamage` broadcasts
                #     `OnCallForHelp` to every allied `ObjAIBase` in range of
                #     BOTH victim and attacker (`ObjAIBase.cs:1127-1152`),
                #     and `FoundNewTarget(true)` re-checks that channel every
                #     tick (not just every 250 ms).
                # If NEITHER happened, nothing relevant changed within the
                # tick at all, so a fixed-phase and a serial evaluation of
                # the SAME tick-start facts should have agreed -- and a row
                # like that is not explained by this floor.
                cls = _class_of(model0[slot])
                team = _team_of(model0[slot])
                relevant_net = (
                    srv_t if verdict.startswith("server ACQUIRED")
                    or verdict.startswith("server SWITCHED")
                    or verdict == "both moved, to different units"
                    else pre_t if verdict.startswith("server DROPPED")
                    else None)
                relevant_slot = slot_of.get(relevant_net) if relevant_net else None
                pre_note = (note_by_slot.get(relevant_slot)
                            if relevant_slot is not None else None)
                pre_e = pre_note.entity if pre_note is not None else None
                post_e = (real_by_slot.get(relevant_slot)
                          if relevant_slot is not None else None)
                relevant_died = bool(
                    pre_e is not None and not pre_e.dead
                    and (post_e is None or post_e.dead))
                relevant_hp_dropped = bool(
                    pre_e is not None and post_e is not None
                    and pre_e.q_hp is not None and post_e.q_hp is not None
                    and post_e.q_hp < pre_e.q_hp)
                minion_post = real_by_slot.get(slot)
                d_pre = (math.hypot(x0[relevant_slot] - x0[slot],
                                    y0[relevant_slot] - y0[slot])
                         if relevant_slot is not None else None)
                d_post = (math.hypot(post_e.x - minion_post.x,
                                     post_e.y - minion_post.y)
                          if post_e is not None and minion_post is not None
                          else None)
                acq = float(acquisition_range[model0[slot]])
                crossed = bool(d_pre is not None and d_post is not None
                               and (d_pre > acq) != (d_post > acq))
                ally_cfh = False
                for oslot in live_slots:
                    if oslot == slot or _team_of(model0[oslot]) != team:
                        continue
                    dd = math.hypot(x0[oslot] - x0[slot], y0[oslot] - y0[slot])
                    if dd > acq:
                        continue
                    on = note_by_slot.get(oslot)
                    if on is None or on.entity is None:
                        continue
                    oe = on.entity
                    op = real_by_slot.get(oslot)
                    if op is None:
                        if not oe.dead:
                            ally_cfh = True
                            break
                        continue
                    if (oe.q_hp is not None and op.q_hp is not None
                            and op.q_hp < oe.q_hp):
                        ally_cfh = True
                        break
                    if (not oe.dead) and op.dead:
                        ally_cfh = True
                        break
                could_be_floor = (relevant_died or relevant_hp_dropped
                                   or crossed or ally_cfh)

                # -------- GROUND TRUTH, from LANERL_DECISION_TRACE=1 -------
                # State diff can only see BEFORE/AFTER a tick. The branch log
                # is written in the server's own `_objects.Values` iteration
                # order (`ObjectManager.cs:74-81`), so two events sharing the
                # same `t` appear in the log in the order the server actually
                # produced them -- the one axis a state diff cannot recover.
                # `found_settarget` sanity-checks that this row's own
                # SetTargetUnit is where expected; `earlier_finishcast` is
                # the decisive test: did the unit this minion just acquired
                # finish an attack (dealt damage) EARLIER in this SAME tick,
                # i.e. before this minion's own controller ran? That is
                # `TakeDamage`'s `OnCallForHelp` broadcast, seen in the act.
                gt_found = gt_precedent = None
                gt_detail = ""
                if dec_by_t is not None:
                    tick_events = dec_by_t.get(sn1.t_ms, [])
                    my_pos = None
                    for i2, d in enumerate(tick_events):
                        if (d.kind == "SetTargetUnit" and d.net_id == c.net_id
                                and d.fields.get("to") == str(srv_t)):
                            my_pos = i2
                            break
                    gt_found = my_pos is not None
                    if my_pos is not None:
                        watch = {srv_t} if srv_t else set()
                        for d in tick_events[:my_pos]:
                            if (d.kind == "FinishCasting"
                                    and d.fields.get("auto") == "True"
                                    and d.net_id in watch):
                                gt_precedent = True
                                gt_detail = (f"FinishCasting auto=True "
                                            f"id={d.net_id} earlier same tick")
                                break
                        if gt_precedent is None:
                            gt_precedent = False

                # ---- CFH-002: is this row the UNOBSERVABLE call-for-help map? ----
                # `LaneMinionAI` populates `unitsAttackingAllies` from
                # `OnCallForHelp` during a tick, consumes it in
                # `FoundNewTarget`, and CLEARS it at the end of that same
                # `OnUpdate` (`LaneMinionAI.cs:100-104`, armed by `:159`).
                # `LanerlStateDump` reads that live dictionary (`aihelp=`,
                # `LanerlStateDump.cs:221`) AFTER `OnUpdate` has run, so the
                # dump sees it post-clear and is empty ~99.5% of the time.
                # An injected tick therefore starts with no call-for-help
                # information and CANNOT reproduce a call-for-help re-target
                # -- so every one is scored here as a target-selection
                # failure that the simulator had no way to get right.
                #
                # The observable signature, which needs no extra dump field:
                # the unit the SERVER picked was itself attacking an ALLY of
                # the minion doing the picking, on this tick. That is exactly
                # what `OnCallForHelp` broadcasts.
                srv_iv = pre_internal.get(srv_t) if srv_t else None
                cfh_victim = (srv_iv.target_net_id
                              if srv_iv is not None else None)
                cfh_victim_slot = (slot_of.get(cfh_victim)
                                   if cfh_victim else None)
                server_pick_attacks_my_ally = bool(
                    cfh_victim_slot is not None
                    and _team_of(model0[cfh_victim_slot]) == team)
                my_help_map_was_empty = bool(iv is None or not iv.help)
                cfh_unobservable = bool(server_pick_attacks_my_ally
                                        and my_help_map_was_empty)

                target_rows.append(dict(
                    # GIVE-001: the server's 4-second failed-to-attack rule.
                    # `ReevaluateBehavior` (`LaneMinionAI.cs:323-334`): a target
                    # that is STILL VALID is nonetheless dropped and `Ignore()`d
                    # for 500 ms once `timeSinceLastAttack >= 4000f`. The
                    # counter resets to 0 whenever `LaneMinion.IsAttacking` is
                    # true or there is no target (`:66-73`), so it measures
                    # "how long have I been unable to land a swing on this
                    # target". It is dumped as `aitsa=`, so no new
                    # instrumentation is needed to test it -- which matters,
                    # because the two previous attributions here were both made
                    # from signatures and both turned out wrong.
                    # UNITS: `aitsa` is `Q(timeSinceLastAttack, StatQ)` and
                    # the field itself is in MILLISECONDS (`+= delta`, and the
                    # rule's literal is `4000f` for four seconds). So
                    # `q / StatQ` is milliseconds, NOT seconds, and the
                    # threshold to compare against is 4000. Getting this wrong
                    # once made a 1.03 s maximum read as "1033 s", which is
                    # longer than the whole recording -- the tell that caught
                    # it was the magnitude being physically impossible.
                    injected_tsa_ms=(None if iv is None
                                     or iv.q_time_since_attack is None
                                     else iv.q_time_since_attack / StatQ),
                    cfh_pre_nonempty=_cfh_lookup(cfh_pre, c.net_id, iv),
                    cfh_pre_has_pick=_cfh_has(cfh_pre, c.net_id, iv, srv_t),
                    cfh_attacks_my_ally=server_pick_attacks_my_ally,
                    help_map_empty=my_help_map_was_empty,
                    cfh_unobservable=cfh_unobservable,
                    t_ms=sn.t_ms, kind=c.kind, net=c.net_id,
                    sim_target=c.sim_target_net_id,
                    server_target=c.server_target_net_id,
                    pre_target=pre_t, verdict=verdict,
                    sweep_predicted=sw_pred,
                    server_attacking=c.server_attacking,
                    sim_attacking=c.sim_attacking,
                    sim_d=None if gs is None else gs[0],
                    sim_ideal=None if gs is None else gs[1],
                    server_d=None if gv is None else gv[0],
                    server_ideal=None if gv is None else gv[1],
                    cls=cls, team=team,
                    relevant_died=relevant_died,
                    relevant_hp_dropped=relevant_hp_dropped,
                    crossed_acq_boundary=crossed,
                    ally_took_damage_nearby=ally_cfh,
                    could_be_floor=could_be_floor,
                    gt_found=gt_found, gt_precedent=gt_precedent,
                    gt_detail=gt_detail,
                ))

            # ---------------- aa_hit residual (the 1,041) -----------------
            # `aa_hit` is `HasAutoAttacked` going false->true, and that is
            # **windup completion for melee and ranged alike**.
            # `Spell.FinishCasting` (`Spell.cs:985-1006`) sets
            # `CastInfo.Owner.HasAutoAttacked = true` unconditionally for any
            # `IsAutoAttack` cast and only THEN branches on `!IsMelee` to
            # choose missile-creation over immediate `ApplyEffects` +
            # `AutoAttackHit`; `AutoAttackHit` never touches the flag. An
            # earlier revision of this comment said a ranged minion's flag
            # flips on missile ARRIVAL. It does not, and that reading would
            # have made this residual look like a missile-flight bug. The
            # melee/ranged split below is the test: if the source reading is
            # right the residual rate must be indistinguishable between them.
            # The missile columns are kept as the control that shows the
            # missile is NOT involved.
            if c.sim_hit != c.server_hit:
                own = [mi for mi in mis_n if mi.owner_net_id == c.net_id]
                hit_rows.append(dict(
                    t_ms=sn.t_ms, kind=c.kind, net=c.net_id,
                    ranged=bool(missile_speed[model0[slot]] > 0),
                    sim_hit=c.sim_hit, server_hit=c.server_hit,
                    own_missiles=len(own),
                    own_landed=sum(1 for mi in own
                                   if mi.net_id not in mis_n1_ids),
                    target_agrees=c.sim_target_net_id == c.server_target_net_id,
                    fire_agrees=c.sim_fire == c.server_fire,
                    cd_gap_q=c.sim_aa_cooldown_q - c.server_aa_cooldown_q,
                ))

    print(f"pairs drilled: {n_pairs} "
          f"(t={a.from_ms}..{min(a.to_ms, snaps[-1].t_ms)})")

    print("\n-- move_order confusion (sim -> server) --")
    for (kind, sim_v, srv_v), n in order_confusion.most_common(20):
        print(f"  {n:6d}  {kind}: sim={sim_v} server={srv_v}")

    _drill_move_order(order_rows, order_denom)

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

    print("\n-- target disagreements against the SHARED pre-tick target --")
    print("   (the injected target is the server's own NetId, so this splits")
    print("    retention from acquisition; the post-tick pair alone cannot)")
    vshape = collections.Counter()
    for r in target_rows:
        vshape[(r["kind"], r["verdict"])] += 1
    for (kind, verdict), n in vshape.most_common():
        print(f"  {n:6d}  {kind}: {verdict}")
    print("   did the 250 ms timer alone predict a sweep on that tick?")
    for verdict, _n in collections.Counter(
            r["verdict"] for r in target_rows).most_common(4):
        sub = [r for r in target_rows if r["verdict"] == verdict]
        yes = sum(1 for r in sub if r["sweep_predicted"])
        print(f"     {verdict}: {yes}/{len(sub)} on a timer-predicted sweep "
              f"({100 * yes / max(1, len(sub)):.1f}%)")
    print("   attacking flags on the two largest verdicts:")
    for verdict, _n in collections.Counter(
            r["verdict"] for r in target_rows).most_common(2):
        sub = [r for r in target_rows if r["verdict"] == verdict]
        flags = collections.Counter(
            (r["sim_attacking"], r["server_attacking"]) for r in sub)
        for (sa, va), n in flags.most_common(4):
            print(f"     {n:6d}  {verdict}: sim_attacking={int(sa)} "
                  f"server_attacking={int(va)}")

    print("\n-- ORDER-003: target residual DECOMPOSED, LaneMinion only --")
    print("   by minion profile x team x transition (the full table, not a rate)")
    minion_rows = [r for r in target_rows if r["kind"] == "LaneMinion"]
    dec = collections.Counter(
        (r["cls"], r["team"], r["verdict"]) for r in minion_rows)
    for (cls, team, verdict), n in dec.most_common():
        print(f"     {n:6d}  cls={cls:<8} team={team}  {verdict}")
    print("   marginals -- by profile:")
    for cls, n in collections.Counter(r["cls"] for r in minion_rows).most_common():
        print(f"     {n:6d}  {cls}")
    print("   marginals -- by team:")
    for team, n in collections.Counter(r["team"] for r in minion_rows).most_common():
        print(f"     {n:6d}  team={team}")
    print("   marginals -- by transition type:")
    for verdict, n in collections.Counter(
            r["verdict"] for r in minion_rows).most_common():
        print(f"     {n:6d}  {verdict}")

    print("\n-- CFH-002: is the residual the UNOBSERVABLE call-for-help map? --")
    print("   The server populates, consumes and CLEARS `unitsAttackingAllies`")
    print("   inside one tick; the dump reads it after the clear, so an")
    print("   injected tick begins with no call-for-help information and")
    print("   cannot reproduce a call-for-help re-target at all. Signature:")
    print("   the unit the SERVER picked was itself attacking an ALLY of the")
    print("   minion doing the picking, while that minion's own dumped help")
    print("   map was empty.")
    n_cfh = sum(1 for r in minion_rows if r["cfh_unobservable"])
    n_ally = sum(1 for r in minion_rows if r["cfh_attacks_my_ally"])
    n_empty = sum(1 for r in minion_rows if r["help_map_empty"])
    print(f"   {n_ally} / {len(minion_rows)} the server's pick was attacking "
          f"one of the chooser's allies")
    print(f"   {n_empty} / {len(minion_rows)} the chooser's dumped help map "
          f"was empty (so the sim had nothing to go on)")
    print(f"   {n_cfh} / {len(minion_rows)} "
          f"({100 * n_cfh / max(1, len(minion_rows)):.1f}%) BOTH -- i.e. "
          f"unobservable, not a simulator error")
    print("   by verdict:")
    for verdict, _n in collections.Counter(
            r["verdict"] for r in minion_rows).most_common():
        sub = [r for r in minion_rows if r["verdict"] == verdict]
        yes = sum(1 for r in sub if r["cfh_unobservable"])
        print(f"     {verdict}: {yes}/{len(sub)} "
              f"({100 * yes / max(1, len(sub)):.1f}%)")
    gt = [r for r in minion_rows if r["cfh_pre_nonempty"] is not None]
    if gt:
        print("   -- GROUND TRUTH from the pre-clear map (`CallForHelpClear`) --")
        print("      This replaces the inferred signature above with the map")
        print("      the server actually read. `None` = no clear event found for")
        print("      that unit+tick, which means it did NOT re-evaluate, and is")
        print("      NOT the same as an empty map.")
        ne = sum(1 for r in gt if r["cfh_pre_nonempty"])
        hp = sum(1 for r in gt if r["cfh_pre_has_pick"])
        print(f"      {len(gt)} / {len(minion_rows)} rows located a clear event")
        print(f"      {ne} of those had a NON-EMPTY pre-clear map")
        print(f"      {hp} of those had the SERVER'S PICK actually IN the map "
              f"-- this is the only number that confirms CFH-002 caused the row")
        agree = sum(1 for r in gt
                    if bool(r["cfh_unobservable"]) == bool(r["cfh_pre_has_pick"]))
        print(f"      heuristic agreed with ground truth on {agree}/{len(gt)}")

    print("\n-- GIVE-001: is the residual the server's 4 s give-up rule? --")
    print("   `ReevaluateBehavior` drops a STILL-VALID target and ignores it")
    print("   for 500 ms once `timeSinceLastAttack >= 4000 ms`. If the sim")
    print("   does not reproduce that, the server lets go while the sim holds")
    print("   -- which is exactly the dominant verdict here.")
    have = [r for r in minion_rows if r["injected_tsa_ms"] is not None]
    if not have:
        print("   (no dumped `aitsa` on these rows)")
    else:
        over = [r for r in have if r["injected_tsa_ms"] >= 4000.0]
        print(f"   {len(have)} rows carry a dumped time-since-attack; "
              f"{len(over)} ({100 * len(over) / len(have):.1f}%) are at or "
              f"past the 4000 ms threshold")
        print("   by verdict (n at/past 4 s / n with a value, and the max seen):")
        for verdict, _n in collections.Counter(
                r["verdict"] for r in have).most_common():
            sub = [r for r in have if r["verdict"] == verdict]
            yes = sum(1 for r in sub if r["injected_tsa_ms"] >= 4000.0)
            mx = max(r["injected_tsa_ms"] for r in sub)
            print(f"     {verdict}: {yes}/{len(sub)}  max tsa {mx:.1f} ms")
        print("   distribution of time-since-attack on these rows:")
        for lo, hi in ((0, 1), (1, 100), (100, 500), (500, 1000),
                       (1000, 2000), (2000, 3900), (3900, 4000), (4000, 1e12)):
            n = sum(1 for r in have if lo <= r["injected_tsa_ms"] < hi)
            if n:
                print(f"     {n:5d}  {lo} ms <= tsa < {hi} ms")

    left = [r for r in minion_rows if not r["cfh_unobservable"]]
    print(f"   NOT explained by the unobservable map: {len(left)}, by verdict:")
    for verdict, n in collections.Counter(
            r["verdict"] for r in left).most_common():
        print(f"     {n:6d}  {verdict}")

    print("\n-- ORDER-003: is the residual CONFINED to a possible serial-order "
          "floor? --")
    print("   `could_be_floor` = the row's relevant unit (server's new/dropped")
    print("   target) died, took damage or crossed the acquisition boundary")
    print("   THIS tick, OR an ally within this minion's own acquisition range")
    print("   took damage this tick (the call-for-help broadcast channel).")
    print("   If the residual is NOT confined to these, the floor argument")
    print("   does not cover it and something else is left unexplained.")
    n_floor = sum(1 for r in minion_rows if r["could_be_floor"])
    print(f"   {n_floor} / {len(minion_rows)} "
          f"({100 * n_floor / max(1, len(minion_rows)):.1f}%) could be floor")
    print("   breakdown by verdict:")
    for verdict, _n in collections.Counter(
            r["verdict"] for r in minion_rows).most_common():
        sub = [r for r in minion_rows if r["verdict"] == verdict]
        yes = sum(1 for r in sub if r["could_be_floor"])
        print(f"     {verdict}: {yes}/{len(sub)} ({100 * yes / max(1, len(sub)):.1f}%)")
    print("   which sub-signal explains it, among the could-be-floor rows:")
    floor_rows = [r for r in minion_rows if r["could_be_floor"]]
    for key in ("relevant_died", "relevant_hp_dropped", "crossed_acq_boundary",
                "ally_took_damage_nearby"):
        n = sum(1 for r in floor_rows if r[key])
        print(f"     {n:6d} / {len(floor_rows)}  {key}")
    print("   the NOT-covered rows, by verdict (this is what still needs an")
    print("   explanation if the count is not ~0):")
    uncovered = [r for r in minion_rows if not r["could_be_floor"]]
    for verdict, n in collections.Counter(
            r["verdict"] for r in uncovered).most_common():
        print(f"     {n:6d}  {verdict}")

    if dec_by_t is not None:
        print("\n-- ORDER-003: GROUND TRUTH from the branch log --")
        print("   Does the branch log confirm this row's SetTargetUnit at the")
        print("   tick we think it is, and is there another unit's completed")
        print("   attack EARLIER in the SAME tick, in the server's own")
        print("   `_objects.Values` iteration order -- not inferred from a")
        print("   before/after state diff, but read off the log line order?")
        found = sum(1 for r in minion_rows if r["gt_found"])
        print(f"   {found} / {len(minion_rows)} rows located their own "
              f"SetTargetUnit at the expected tick")
        prec = [r for r in minion_rows if r["gt_precedent"] is True]
        checked = [r for r in minion_rows if r["gt_precedent"] is not None]
        print(f"   {len(prec)} / {len(checked)} have an EARLIER same-tick "
              f"FinishCasting(auto=True) by the unit they just acquired "
              f"(the CFH mechanism, caught in the act)")
        print("   cross-tab against the state-diff heuristic (`could_be_floor`):")
        tab = collections.Counter(
            (r["could_be_floor"], r["gt_precedent"]) for r in checked)
        for (heur, gt), n in sorted(tab.items()):
            print(f"     {n:6d}  heuristic could_be_floor={heur}  "
                  f"ground-truth precedent={gt}")
        print("   by verdict, ground-truth precedent rate:")
        for verdict, _n in collections.Counter(
                r["verdict"] for r in checked).most_common():
            sub = [r for r in checked if r["verdict"] == verdict]
            yes = sum(1 for r in sub if r["gt_precedent"])
            print(f"     {verdict}: {yes}/{len(sub)} "
                  f"({100 * yes / max(1, len(sub)):.1f}%)")
        combined = sum(1 for r in minion_rows
                       if r["could_be_floor"] or r["gt_precedent"] is True)
        print(f"   COMBINED (heuristic OR ground-truth precedent): "
              f"{combined} / {len(minion_rows)} "
              f"({100 * combined / max(1, len(minion_rows)):.1f}%)")
        still_uncovered = [r for r in minion_rows
                           if not r["could_be_floor"]
                           and r["gt_precedent"] is not True]
        print(f"   still UNCOVERED by either check: {len(still_uncovered)}, "
              f"by verdict:")
        for verdict, n in collections.Counter(
                r["verdict"] for r in still_uncovered).most_common():
            print(f"     {n:6d}  {verdict}")
        for r in still_uncovered[:a.examples]:
            print(f"    e.g. t={r['t_ms']} net={r['net']} {r['verdict']}: "
                  f"gt_found={r['gt_found']} gt_detail={r['gt_detail']!r}")

    for r in target_rows[:a.examples]:
        print(f"    e.g. t={r['t_ms']} {r['kind']} net={r['net']}: "
              f"sim->{r['sim_target']} (d={r['sim_d']}, ideal={r['sim_ideal']}) "
              f"server->{r['server_target']} (d={r['server_d']}, "
              f"ideal={r['server_ideal']})")

    print("\n-- position disagreements (L-inf > 1/16 on the dump's integers) --")
    print(f"   scored on {n_scored['LaneMinion.position_linf']} trustworthy "
          f"LaneMinion pairs; {len(pos_rows)} disagreed")
    pshape = collections.Counter(r["shape"] for r in pos_rows)
    for shape, n in pshape.most_common():
        sub = [r for r in pos_rows if r["shape"] == shape]
        e = np.asarray([r["linf"] for r in sub])
        print(f"  {n:6d}  {shape}: L-inf median {np.median(e):.3f} u "
              f"p95 {np.percentile(e, 95):.3f} max {e.max():.3f}")
        oc = collections.Counter(
            (r["pre_order"], r["sim_order"], r["srv_order"]) for r in sub)
        for (po, so, vo), k in oc.most_common(4):
            print(f"         {k:6d}  order pre={po} -> sim={so} / server={vo}")
        ac = collections.Counter(
            (r["pre_attacking"], r["pre_aa_state"]) for r in sub)
        for (at, st), k in ac.most_common(3):
            print(f"         {k:6d}  pre-tick is_attacking={at} aa_state={st}")
        wc = collections.Counter(
            (r["dump_n_wp"], r["inj_n_wp"], r["inj_key"], r["srv_post_n_wp"])
            for r in sub)
        for (dn, inj, key, post), k in wc.most_common(4):
            print(f"         {k:6d}  waypoints: dump={dn} injected={inj} "
                  f"key={key} -> server post={post}  "
                  f"(sim can move iff key < injected)")
        rc = collections.Counter(
            (r["in_range"], r["colliding_neighbours"] > 0) for r in sub)
        for (inr, coll), k in rc.most_common(4):
            print(f"         {k:6d}  target in idealRange={inr}  "
                  f"colliding neighbour present={coll}")
        fc = collections.Counter(
            (r["colliding_neighbours"], r["frozen_neighbours"]) for r in sub)
        for (lv, fz), k in fc.most_common(5):
            print(f"         {k:6d}  neighbours overlapping LIVE={lv}, "
                  f"in the FROZEN quadtree list={fz}")
        pe = [r for r in sub if r["pred_escape"] is not None]
        if pe:
            gap = np.asarray([abs(r["srv_step"]) - abs(r["pred_escape"])
                              for r in pe])
            cos = np.asarray([r["pred_cos"] for r in pe
                              if r["pred_cos"] is not None])
            sgn = np.asarray([r["pred_escape"] for r in pe])
            print(f"         GetCircleEscapePoint against the nearest "
                  f"overlapping neighbour, on {len(pe)} rows:")
            print(f"           predicted |escape| minus the server's own "
                  f"|step|: median {np.median(-gap):+.3f} u  "
                  f"p5 {np.percentile(-gap, 5):+.3f}  "
                  f"p95 {np.percentile(-gap, 95):+.3f}")
            print(f"           predicted term sign: "
                  f"{100 * np.mean(sgn > 0):.1f}% positive "
                  f"(a PULL toward the neighbour)")
            if len(cos):
                print(f"           cos(server step, direction to that "
                      f"neighbour): median {np.median(cos):+.3f}  "
                      f"{100 * np.mean(cos > 0.9):.1f}% aligned, "
                      f"{100 * np.mean(cos < -0.9):.1f}% opposed")
    for r in pos_rows[:a.examples]:
        print(f"    e.g. t={r['t_ms']} net={r['net']} {r['shape']}: "
              f"L-inf {r['linf']:.3f} u; step sim={r['sim_step']:.4f} "
              f"server={r['srv_step']:.4f}; order pre={r['pre_order']} "
              f"sim={r['sim_order']} server={r['srv_order']}; "
              f"pre is_attacking={r['pre_attacking']} "
              f"aa_state={r['pre_aa_state']}; wp dump={r['dump_n_wp']} "
              f"inj={r['inj_n_wp']}/key{r['inj_key']} "
              f"srvpost={r['srv_post_n_wp']}; "
              f"d={None if r['d_to_target'] is None else round(r['d_to_target'], 1)} "
              f"ideal={r['ideal_range']} neigh={r['colliding_neighbours']}")

    print("\n-- position parity by CROWDING bucket (the denominator) --")
    for b in sorted(crowd_scored):
        tot_b = crowd_scored[b]
        miss = crowd_missed[b]
        label = f"{b}" if b < 3 else "3+"
        print(f"   {label} colliding neighbours: {tot_b - miss}/{tot_b} exact "
              f"({100 * (tot_b - miss) / max(1, tot_b):.3f}%), {miss} missed")

    print("\n-- AI sweep phase, predicted from the dump alone (no sim) --")
    print("   rows are (the sim's rule says it sweeps, the server's own")
    print("   aitimer shows it swept); off-diagonal is a phase error")
    tot = sum(sweep_census.values())
    for (pred, obs), n in sorted(sweep_census.items()):
        print(f"   predicted={int(pred)} observed={int(obs)}: {n} "
              f"({100 * n / max(1, tot):.3f}%)")

    print("\n-- aa_hit disagreements (HasAutoAttacked false->true) --")
    hshape = collections.Counter(
        (r["kind"], r["sim_hit"], r["server_hit"], r["own_missiles"] > 0,
         r["fire_agrees"], r["target_agrees"]) for r in hit_rows)
    for (kind, sh, vh, mis, fa, ta), n in hshape.most_common(12):
        print(f"  {n:6d}  {kind}: sim_hit={int(sh)} server_hit={int(vh)} "
              f"own_missile_in_flight={int(mis)} fire_agrees={int(fa)} "
              f"target_agrees={int(ta)}")
    if hit_rows:
        cd = np.asarray([r["cd_gap_q"] for r in hit_rows])
        print(f"   sim-minus-server AA cooldown at the disagreeing tick "
              f"(StatQ units): median {np.median(cd):+.0f} "
              f"p5 {np.percentile(cd, 5):+.0f} p95 {np.percentile(cd, 95):+.0f}; "
              f"{100 * np.mean(cd == 0):.1f}% identical")

    print("\n-- aa_hit by MELEE vs RANGED (the source-reading test) --")
    print("   `HasAutoAttacked` flips at windup completion for BOTH, so these")
    print("   rates must be indistinguishable; a ranged excess refutes that")
    for cls in ("melee", "ranged"):
        d, mm = hit_denom[cls], hit_miss[cls]
        print(f"   {cls}: {mm} mismatched unit-ticks against {d} server swing "
              f"completions ({mm / max(1, d):.3f} per completion)")

    print("\n-- aa_cooldown residual: is it the RESET-002 quantum? --")
    if cd_gap:
        g = np.asarray([v for v, _ in cd_gap])
        cov = np.asarray([c for _, c in cd_gap])
        print(f"   {len(g)} disagreeing unit-ticks; |gap| in StatQ quanta: "
              f"median {np.median(g):.0f} p95 {np.percentile(g, 95):.0f} "
              f"max {g.max():.0f}")
        for lo, hi, label in ((0, 2, "<=2 quanta (sub-tick rounding)"),
                              (3, 40, "3-40"),
                              (41, 10 ** 9, ">40 (a whole swing or more)")):
            sel = (g >= lo) & (g <= hi)
            print(f"     {int(sel.sum()):6d}  {label}: "
                  f"{100 * sel.mean():.1f}%, of which "
                  f"{100 * cov[sel].mean() if sel.sum() else 0:.1f}% are on a "
                  f"tick that also disagreed on aa_fire or aa_hit")
    print("\n-- RESIDUAL CONTAINMENT: are move_order/waypoints SEPARATE defects? --")
    print("   `RefreshWaypoints` rebuilds a 2-point chase path from the unit's")
    print("   TARGET every tick it holds one, and the move order is chosen by")
    print("   the same re-evaluation that picks the target. So a target")
    print("   disagreement mechanically implies a waypoint and often a move-")
    print("   order one. Counting them as independent residuals would triple-")
    print("   count a single cause -- which is how `AA-002` once presented as")
    print("   three separate fields. Keyed on (tick, netid).")
    tgt_keys = {(r["t_ms"], r["net"]) for r in target_rows}
    fire_keys = {(r["t_ms"], r["net"]) for r in fire_rows}
    for name, rows in (("move_order", order_rows), ("waypoints", wp_rows_keyed)):
        if not rows:
            print(f"   {name}: no rows in this window")
            continue
        k = [(r["t_ms"], r["net"]) for r in rows]
        in_t = sum(1 for x in k if x in tgt_keys)
        in_f = sum(1 for x in k if x in fire_keys)
        either = sum(1 for x in k if x in tgt_keys or x in fire_keys)
        print(f"   {name}: {len(k)} rows; {in_t} share a tick+unit with a "
              f"TARGET disagreement, {in_f} with a FIRE one, {either} with "
              f"either ({100 * either / len(k):.1f}%)")

    print("\n-- AA-004: is the fire residual the dump's COOLDOWN GATE CLAMP? --")
    print("   `Q(Math.Max(0f, remaining), StatQ)` clamps BEFORE quantising, so")
    print("   a still-positive cooldown within rounding of the gate publishes")
    print("   as a flat 0. Injecting that 0 says READY when the server was")
    print("   not, and the sim swings one tick early. Prediction: the")
    print("   sim-fires-early rows should sit overwhelmingly on a dumped 0,")
    print("   and the sim-fires-LATE rows should not.")
    early = [r for r in fire_rows if r["sim_fire"] and not r["server_fire"]]
    late = [r for r in fire_rows if r["server_fire"] and not r["sim_fire"]]
    for name, rows in (("sim fires, server does NOT", early),
                       ("server fires, sim does NOT", late)):
        known = [r for r in rows if r["injected_cd_q"] is not None]
        zero = sum(1 for r in known if r["injected_cd_q"] == 0)
        print(f"   {name}: {len(rows)} rows, {len(known)} with a dumped "
              f"cooldown, {zero} of those at the gate clamp (0)"
              + (f" = {100 * zero / len(known):.1f}%" if known else ""))
        if known:
            nz = sorted({r["injected_cd_q"] for r in known
                         if r["injected_cd_q"] != 0})[:8]
            if nz:
                print(f"     non-zero dumped cooldowns present: {nz}")
            # The decisive split, only available on a recording carrying
            # `aacdbits`. Being AT the clamp is necessary for the blind-spot
            # story but not sufficient: it is only a blind spot if the exact
            # value was strictly POSITIVE.
            clamp = [r for r in known if r["injected_cd_q"] == 0]
            exact = [r for r in clamp if r["injected_cd_exact"] is not None]
            if not exact:
                print("     (this recording has no `aacdbits`, so whether the "
                      "clamp HID a positive residue is unobservable here)")
            else:
                pos = [r for r in exact if r["injected_cd_exact"] > 0.0]
                zer = [r for r in exact if r["injected_cd_exact"] == 0.0]
                neg = [r for r in exact if r["injected_cd_exact"] < 0.0]
                print(f"     of the {len(clamp)} at the clamp, {len(exact)} "
                      f"carry an exact unclamped value:")
                print(f"       strictly POSITIVE (server NOT ready; the clamp "
                      f"hid it -> instrument blind spot): {len(pos)}")
                print(f"       exactly ZERO  (ambiguous): {len(zer)}")
                print(f"       strictly NEGATIVE (server WAS ready; something "
                      f"ELSE stopped it -> candidate sim bug): {len(neg)}")
                if pos:
                    vs = sorted({round(r["injected_cd_exact"], 9) for r in pos})
                    print(f"       positive residues seen: {vs[:6]}")
                if neg:
                    vs = sorted({round(r["injected_cd_exact"], 9) for r in neg})
                    print(f"       negative residues seen: {vs[:6]}")
                ready = zer + neg
                if ready:
                    print(f"     the {len(ready)} rows where the server was "
                          f"READY by its own `<= 0` gate and still did not "
                          f"fire -- what ELSE held it? (SpellState 0=READY "
                          f"1=CASTING 2=COOLDOWN 3=CHANNELING)")
                    tab = collections.Counter(
                        (r["injected_aa_state"], r["injected_attacking"])
                        for r in ready)
                    for (st, at), n in sorted(tab.items(),
                                              key=lambda kv: -kv[1]):
                        verdict = ("spell NOT ready -> the server cannot swing; "
                                   "a gate the sim may not apply"
                                   if st not in (0, None) else
                                   "spell READY too -> neither gate explains it")
                        print(f"       {n:5d}  aastate={st} is_attacking={at}"
                              f"   {verdict}")
                    print(f"     for contrast, the {len(pos)} clamp-hidden rows "
                          f"by the same key:")
                    tab2 = collections.Counter(
                        (r["injected_aa_state"], r["injected_attacking"])
                        for r in pos)
                    for (st, at), n in sorted(tab2.items(),
                                              key=lambda kv: -kv[1]):
                        print(f"       {n:5d}  aastate={st} is_attacking={at}")

    print("\n-- is_attacking / has_auto_attacked: contained in fire U hit? --")
    for (name, tag), n in sorted(bool_join.items()):
        print(f"   {name}: {tag} {n}")

    print("\n-- missile census (RESET-004's blind spot, measured) --")
    for k in ("total", "injected", "DROPPED", "landed_or_expired_this_tick"):
        print(f"   {k}: {missile_census[k]}")

    print("\n-- hp disagreements, against the INBOUND missile, not any missile --")
    dshape = collections.Counter(
        (r["kind"], r["inbound"] > 0, r["inbound_landed"] > 0,
         r["inbound_dropped"] > 0) for r in hp_detail)
    for (kind, inb, landed, dropped), n in dshape.most_common(12):
        print(f"  {n:6d}  {kind}: a missile was aimed at THIS unit={int(inb)} "
              f"and it left the field this tick={int(landed)} "
              f"(injector dropped it={int(dropped)})")
    for kind in sorted({r["kind"] for r in hp_detail}):
        sub = [r for r in hp_detail if r["kind"] == kind]
        d = np.abs(np.asarray([r["delta"] for r in sub]))
        vals = collections.Counter(round(abs(r["delta"]), 2) for r in sub)
        print(f"   {kind}: |error| median {np.median(d):.3f} max {d.max():.3f}; "
              f"most common magnitudes "
              f"{', '.join(f'{v}x{n}' for v, n in vals.most_common(6))}")

    print("\n-- hp disagreements, split by the legacy 'any missile' flag --")
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
