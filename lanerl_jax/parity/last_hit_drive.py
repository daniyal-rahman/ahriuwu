"""Drive the J1 gate 3 oracle against the JAX sim and against the real server.

:mod:`lanerl_jax.parity.last_hit_oracle` is the policy: pure, deterministic,
JAX-free, and identical in both places by construction. This module is the two
*drivers* -- the glue that turns each implementation's own state representation
into the ``ChampView``/``MinionView`` the oracle reads, and its ``Decision``
back into that implementation's own order format. Neither driver teaches the
oracle anything about the other side; each only exposes what its own engine
already knows about itself.

Both drivers read the champion's OWN currently-in-effect combat stats rather
than any independently-derived number, on purpose: the oracle's "would this
attack kill" test has to be calibrated to what THAT engine's next tick will
actually apply, or a mismatch between the driver's guess and the engine's
truth would show up as a spurious CS difference that has nothing to do with
the mechanic under test. Concretely:

* The **sim** driver reads ``attack_damage`` through the same
  ``params[key][state.model]`` gather :func:`lanerl_jax.sim.step.tick` uses to
  compute real damage, PLUS the same level-scaling term ``tick`` now adds
  (``params["ad_per_level"][model] * growth_sum(state.level)`` -- see below).
* The **server** driver reads ``ad``/``rng`` straight off the wire's own
  ``Stats.AttackDamage.Total`` / ``Stats.Range.Total``. Re-deriving either
  side's attack damage from a formula instead of reading each engine's own
  number is exactly the mistake ``lanerl_rl/constants.py`` documented paying
  for once already (see ``sim/init.py``'s RUNE_AD_BONUS note) -- the sim side
  above is not that mistake, because it reads the SAME per-profile table and
  the SAME per-level column ``tick`` itself reads, not an independent copy.

RESOLVED: the JAX sim did not level-scale champion combat stats
------------------------------------------------------------------
Was a real gap: ``lanerl_jax.sim.combat.stat_at_level`` implemented the
server's non-linear per-level growth curve, but nothing in
:func:`lanerl_jax.sim.step.tick` called it for the champion's attack
damage -- ``P("attack_damage")`` was the flat, level-1 profile value for the
whole episode, even though ``state.level`` itself does climb as XP
accumulates. Fixed in ``sim/step.py``'s ``tick`` and ``sim/profiles.py``'s
``ad_per_level`` column (test: ``lanerl_jax/sim/tests/
test_champion_level_scaling.py``, fails before this fix and passes after).
Measured effect on THIS gate: small (see ``lanerl_jax/parity/hp_band.py``'s
before/after report, 468 -> 473 in-band samples out of 8218) -- the sim's
champion barely levels here (1..5, against the server's 1..8) because it
keeps dying and losing its proximity-XP window, a separate, NOT resolved
issue (see below and ``docs/JAX_REWRITE_PLAN.md``'s J1 status).

RESOLVED (partially): the JAX sim had no fog of war
----------------------------------------------------
:class:`~lanerl_jax.sim.state.LaneState` still carries no visibility field,
but this driver now gates its enemy-minion candidate list through
``obs.fog.visible_to`` (a radius-based approximation of the server's real,
terrain-aware ``"vb"`` flag the server driver reads directly) rather than
every live red minion on the map. This closed most, but not all, of a large
gap: before it, the sim scored CS=2/39 attacks against the server's CS=10/535
attacks over the same 600 s. See ``docs/JAX_REWRITE_PLAN.md``'s J1 status and
this module's own gate test (``lanerl_jax/parity/tests/
test_last_hit_gate.py``) for what the CURRENT gap is and why that "535" server
figure is now known to be stale (measured before an unrelated position fix,
never refreshed) rather than a live target to close.

THE FIRST VERSION OF THIS GATE MEASURED PATHING, NOT LAST-HITTING
--------------------------------------------------------------------
Worth recording in full, because it is exactly the failure mode this gate
exists to prevent: a first cut of this driver handed ``decide()`` control from
the champion's spawn point, decision one. When no minion is in reach, the
oracle's fallback is "walk to the centroid of the enemy minions"
(``last_hit_oracle.decide``) -- and from blue's fountain that centroid is a
single ~13,500-unit order, roughly the length of the whole lane. `orders.py`
says plainly why that is fatal in the sim and nowhere else: the server routes
a move order through `GetPath`, but "on device there is no A*, so a move order
becomes the two-point line directly" (measured there at 45% of sub-1800-unit
paths already straight -- a ~13,500-unit order is nowhere near that regime).
On the server the order is pathfound around the map and the champion arrives
(the server numbers below: 382 attacks, CS 16). In the sim it became one
straight segment into terrain, the champion never reached the wave, and it
never swung: two runs of that version scored sim CS 0-1 against attacks
0-11, essentially all of it spent walking, none of it near a minion. That gap
was the pathing approximation ``orders.py`` already documents and had
nothing to do with last-hitting -- the oracle was never actually exercised.
Fixing it by adding pathfinding to the sim was explicitly out of scope (a
next-hop table is its own open design item, `docs/JAX_REWRITE_PLAN.md`), so
the fix here is a driver-side one: see :data:`APPROACH_WAYPOINTS` below.

THE FIX: A SCRIPTED, TERRAIN-SAFE APPROACH BEFORE HANDOVER
--------------------------------------------------------------
Both drivers now walk the champion along ``TOP_LANE_PATH`` -- the same
polyline the minions themselves walk -- one waypoint at a time, up to and
including the vertex where the two waves' paths cross
(``lanerl_jax.parity.record.MEETING_POINT``, the same point that module's own
scripted drive walks to first, "because a champion in its fountain has
nothing to fight"). Each leg of that polyline is short enough to already be a
straight, terrain-clear line by construction -- that is what a waypoint
polyline *is* -- so re-walking it leg by leg costs nothing extra in the sim
and matches the server's own pathing over the same short hops. Only once the
champion has arrived at the last approach waypoint does either driver call
``decide()`` at all. This is option 1 from the gate's own review: it needs no
sim changes, keeps both drivers walking an identical route, and means any CS
difference measured after handover is about last-hitting, not about how either
side got to the lane. It worked as intended for the approach itself -- see
``approach_decisions`` above -- and it does move the needle on CS (sim went
from CS 0-1 / 0-11 attacks to CS 2 / 39 attacks), but a large gap against the
server's CS 10 / 535 attacks remains. See the fog-of-war note above for the
leading remaining suspect.

Why the wire's minion-type index is not the sim's
--------------------------------------------------
The wire's ``"mt"`` field is ``GameServerCore.Enums.MinionSpawnType`` as the
server numbers it: MELEE=0, SUPER=1, CANNON=2, CASTER=3 -- caster and cannon
swapped relative to the natural reading order. That is a different numbering
from :class:`lanerl_jax.sim.targeting.MinionType` (MELEE=0, CASTER=1, CANNON=2,
SUPER=3), which exists to match ``MinionWaveTypes``' spawn order instead.
``lanerl_rl.constants.MINION_TYPE_INDEX`` already had to get this right for
the same reason; :data:`WIRE_MINION_TYPE` here is this module's copy of that
mapping, kept local so this module does not have to import ``lanerl_rl`` for
one dict.
"""
from __future__ import annotations

import inspect
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from ..obs.fog import visible_to
from ..sim.combat import growth_sum
from ..sim.config import SimConfig
from ..sim.init import TOP_LANE_PATH, init_lane
from ..sim.orders import OrderKind, Orders
from ..sim.state import Kind, Team
from ..sim.step import env_step
from .lanerl_lane import LanerlLane, forward_for
from .last_hit_oracle import (ChampView, Decision, MinionView, decide,
                              post_mitigation)

__all__ = [
    "DECISIONS_600S", "WIRE_MINION_TYPE", "APPROACH_WAYPOINTS",
    "ARRIVE_RADIUS", "FINAL_ARRIVE_RADIUS", "FINAL_ARRIVE_GRACE",
    "noisy_policy", "heuristic_policy",
    "SimRun", "ServerRun",
    "DEFAULT_GATE3_ROUTE_ARTIFACT", "gate3_route_inputs",
    "run_oracle_in_sim", "run_oracle_on_server",
]

#: 600 s at the 30 Hz decision rate (``step_ticks=2`` off a 60 Hz sim).
DECISIONS_600S = 18_000

# Gate 3 issues real Move orders during its scripted walk-in.  The server
# routes every one through GetPath and production JAX training uses this
# artifact, so the routed path is the parity default.  The former raw segment
# remains available only through ``table_disabled=True`` as a PATH-006
# isolation control.
DEFAULT_GATE3_ROUTE_ARTIFACT = (Path(__file__).resolve().parents[2] / "data" /
                                "jax_routes" / "map1_garen_r35_o50_v2")

#: wire ``MinionSpawnType`` -> patch-table minion key. See the module
#: docstring for why this is not the identity map.
WIRE_MINION_TYPE = {0: "melee", 1: "super", 2: "cannon", 3: "caster"}

#: Blue's forward-order waypoints from ``TOP_LANE_PATH``, stopping BEFORE the
#: enemy outer turret.
#:
#: This was ``[:7]``, which ends at index 6 = (3907, 13243). That is lane
#: fraction **0.604** -- past the 0.500 where the two waves actually meet, and
#: **412 units** from red's outer turret, comfortably inside its 750 range. The
#: harness was parking the champion under the enemy tower and then measuring
#: how well it farms. Index 5 = (2806, 13075) is lane 0.553 and 1,248 units
#: from that turret, outside its range.
#:
#: The gate is supposed to isolate last-hitting. Standing somewhere a champion
#: would never voluntarily stand is a property of the harness, not of the
#: simulation, and it is the third distinct way this gate has measured
#: something other than what it is for -- after the missing A* on move orders
#: and the missing fog of war.
#:
#: Old note, still true of the remaining waypoints:
#: index 6 -- ``(3907.0, 13243.0)``, the vertex where the two lanes' minion
#: waves' paths cross (bit-for-bit ``lanerl_jax.parity.record.MEETING_POINT``).
#: Walked in order before either driver ever calls ``decide()``; see the
#: module docstring for why. Roughly 14,900 units from ``CHAMPION_SPAWN[BLUE]``
#: -- about 40-44 s at Garen's move speed, well inside the pre-wave window
#: (the first wave spawns at 90 s), so the champion always finishes the
#: approach before there is anything to fight.
APPROACH_WAYPOINTS = TOP_LANE_PATH[:6]

#: How close counts as "arrived" at an approach waypoint before advancing to
#: the next one. Generous relative to one decision's travel (~20-25 game units
#: at Garen's move speed and the 30 Hz decision rate), so the champion does
#: not hover just short of the target forever waiting for an exact hit.
ARRIVE_RADIUS = 100.0
_ARRIVE_RADIUS_SQ = ARRIVE_RADIUS * ARRIVE_RADIUS

#: The LAST approach waypoint is different, and the generous radius above is
#: wrong for it.
#:
#: Intermediate waypoints are waypoints: the champion keeps walking, so
#: advancing 100 u early costs nothing -- the next leg absorbs it. The final
#: one is a STOPPING POINT. The moment it is reached the driver hands over to
#: `decide()`, which on an empty lane returns `hold`, and the champion stands
#: wherever it happened to be for the next several thousand decisions.
#:
#: Measured on the canonical gate-3 run: at the handover decision (1242) the
#: SERVER's champion is at (2806.000, 13075.000) -- exactly the waypoint,
#: 0.00 u -- while the SIM's is at (2736.387, 13008.614), **96.19 u short**,
#: which the 100 u ball accepts. The server lands exactly because its movement
#: engine carries the champion to the order's destination; the sim stops
#: because the driver stopped issuing orders. That asymmetry is in the
#: HARNESS, not in either engine's pathing.
#:
#: The 96 u then persists as a 66.4 u standing offset (median == p90 == 66.4
#: over 3,200 decisions, i.e. zero variance), which is enough to put the sim
#: outside League's 1,400 u XP radius on 11.4% of decisions against the
#: server's 0.2%, cost it the level-4 race at decision 7,460, and open the CS
#: and death gaps downstream (`GATE3-002`). Gate 3's outcome gap was being
#: charged to `PATH-007`; most of it is this constant.
FINAL_ARRIVE_RADIUS = 8.0
_FINAL_ARRIVE_RADIUS_SQ = FINAL_ARRIVE_RADIUS * FINAL_ARRIVE_RADIUS

#: Bounded wait on that tight radius. An engine that cannot close the last few
#: units -- terrain, a collision push, a movement-speed difference -- must not
#: hang the experiment; it hands over anyway and the caller reports the miss,
#: because a driver that silently waits forever is worse than one that is
#: visibly approximate. 600 decisions is 20 s, far longer than the ~4 s the
#: last leg takes.
FINAL_ARRIVE_GRACE = 600


def gate3_route_inputs(*, route_table=None, terrain=None,
                       table_disabled: bool = False):
    """Return the routed Gate-3 inputs, or the explicit raw-path ablation.

    Callers may inject already-loaded inputs for a focused test.  Otherwise
    canonical Gate 3 loads the same v2 local-route artifact production uses.
    Supplying a terrain/table pair while asking for the raw ablation is an
    error: it would make the reported mode ambiguous.
    """
    if table_disabled:
        if route_table is not None or terrain is not None:
            raise ValueError("table_disabled conflicts with route_table/terrain")
        return None, None
    if route_table is None:
        from ..data.local_route_artifact import load_local_route_artifact

        artifact = load_local_route_artifact(
            DEFAULT_GATE3_ROUTE_ARTIFACT, pathfinding_radius=35.0)
        route_table = artifact.as_jax()
    if terrain is None:
        from ..sim.terrain_jax import map1_terrain

        terrain = map1_terrain()
    return route_table, terrain



def noisy_policy(seed: int, epsilon: float = 0.15, radius: float = 450.0):
    """A wandering arm for the outcome gate, seeded so it actually varies.

    WHY THIS EXISTS. The scripted oracle is deterministic and this scenario
    has no bots (`bot_teams="none"`), so `bot_seed` and the sim seed drive
    NOTHING: a 30-seed sweep ran and returned 30 byte-identical outcomes,
    which is one run measured thirty times wearing a distribution's clothing.
    The seed has to enter through the POLICY or it does not enter at all.

    It also fixes a narrowness problem the identical-outcomes result hid. The
    oracle holds position on an empty lane, so the scripted arm visits a thin
    slice of states -- it never walks into turret range, rarely dies, and
    never takes a route the approach did not already script. Those are exactly
    the states an RL policy will spend its time in.

    DETERMINISTIC IN THE INPUTS, which is what keeps the comparison about the
    ENGINES. The draw is a pure function of (seed, decision index), so both
    engines choose the same action whenever they are fed the same scene; any
    difference in what they do is a difference in what they saw, never in what
    the policy rolled.
    """
    import hashlib

    def _u01(i: int) -> float:
        h = hashlib.blake2b(f"{seed}:{i}".encode(), digest_size=8).digest()
        return int.from_bytes(h, "big") / float(1 << 64)

    def policy(champ, minions, i: int):
        if _u01(i) >= epsilon:
            return decide(champ, minions, lethal_epsilon=0.0)
        ang = _u01(i * 2 + 1) * 2.0 * math.pi
        r = radius * (0.25 + 0.75 * _u01(i * 2 + 2))
        return Decision(attack=None,
                        move=(champ.x + r * math.cos(ang),
                              champ.y + r * math.sin(ang)))

    return policy




def _call_policy(policy, champ, minions, i, allies):
    """Call a policy with allies if it wants them, without them if not.

    `noisy_policy` and the plain oracle take `(champ, minions, i)`; the
    `LanerlBot` mirror needs the ally wave as well, because `HoldPoint` holds
    behind its OWN wave front. Sniffing the signature keeps both working rather
    than forcing a flag day on every policy.
    """
    # Signature inspection, NOT try/except TypeError. A bare except would also
    # swallow a TypeError raised INSIDE the policy and silently re-call it with
    # three arguments -- a convenient fallback that changes the semantics, which
    # is the same bug as an `or` default on a filtered list.
    n = len(inspect.signature(policy).parameters)
    return (policy(champ, minions, i, allies) if n >= 4
            else policy(champ, minions, i))


def heuristic_policy(approach_factor: float = 4.0,
                     approach_hp_frac: float = 0.0,
                     retreat_hp_frac: float = 0.25,
                     resume_hp_frac: float = 0.50,
                     retreat_distance: float = 2200.0,
                     minion_avoid_range: float = 100.0,
                     lane_corridor_width: float = 1500.0,
                     holdback_distance: float = 150.0,
                     wave_follow_margin: float = 250.0,
                     waves_met_gap: float = 900.0,
                     contact_hold_radius: float = 0.0,
                     is_blue: bool = True):
    """`LanerlBot`'s lane behaviour, ported rather than approximated.

    Every default here is the C# bot's own (`LanerlConfig.cs`): ApproachFactor
    4.0, RetreatHpFrac 0.25, ResumeHpFrac 0.50, RetreatDistance 2200,
    MinionAvoidRange 100, LaneCorridorWidth 1500, HoldbackDistance 150,
    WaveFollowMargin 250, WavesMetGap 900.

    Branches ported, in the bot's order (`LanerlBot.DecideInner:449-657`):
      1  retreat, with the `_retreating` hysteresis latch
      8  farm scan: killable (`IsLastHitPure`) then approach (`IsApproachTarget`)
      9  `MoveTo(HoldPoint())` -- the real one, see below

    WHY THE EARLIER APPROXIMATIONS FAILED, measured. `HoldPoint` was replaced by
    "back off 500 u from the nearest minion" and the server arm went from 12 CS
    to 5. The real thing (`LanerlBot.cs:1154-1224`) is:
      * the wave FRONTS, not means -- averaging counts minions still walking out
        of the base and drags the hold point halfway home;
      * a lane-corridor filter, so minions in other lanes are ignored;
      * the furthest-forward position still UN-ACQUIRED, from each minion's own
        `AcquisitionRange` -- 600 melee, 700 caster, so no single standoff number
        is correct for both;
      * hold behind its OWN wave front by `WaveFollowMargin`, falling back to the
        enemy front only when it has no wave;
      * a floor at the friendly turret, or the bot turns round at 1:30 and walks
        to meet its own minions at the base.

    `ContactHoldRadius` defaults to 0 HERE, where the bot uses 250: that branch
    orbits the wave-contact point using `_orbitPhase`, seeded per episode from
    `new Random(cfg.Seed + seedOffset + 977)`. Reproducing a seeded C# RNG
    exactly is a port risk with no upside for a determinism test, so the orbit
    is switched off and said so rather than silently approximated.

    `TryRandomAbility`, armed-Q stickiness, turret avoidance, FightToDeath and
    AggroBreakOff remain unported; each needs spell/buff/aggro state in the
    shared view.
    """
    lane = LanerlLane()
    fwd = forward_for(is_blue)
    # `FriendlyFrontTurretAlong` -- blue's own outer turret, the floor that stops
    # the bot trailing its wave home (`LanerlBot.cs:1221-1222`).
    from ..sim.init import TOP_OUTER_TURRET
    # `{0: blue, 1: red}` keyed by team index, each an (x, y). Blue's own outer
    # turret is the floor `FriendlyFrontTurretAlong` supplies.
    own_xy = TOP_OUTER_TURRET[0 if is_blue else 1]
    own_turret_f = lane.along_of((float(own_xy[0]), float(own_xy[1]))) * fwd

    def lane_point(f: float) -> Tuple[float, float]:
        """`LanePoint(f, 0)` -- lateral offset unused with the orbit off."""
        return lane.point_at(f * fwd)

    def hold_point(minions, allies, extra_standoff: float = 0.0):
        have_e = have_a = False
        e_front = a_front = 0.0
        safe_f = float("inf")
        standoff = minion_avoid_range + extra_standoff
        if extra_standoff > 0.0 and standoff < extra_standoff:
            standoff = extra_standoff
        for m, mine in [(m, False) for m in minions] + [(a, True) for a in allies]:
            if lane.distance_to((m.x, m.y)) > lane_corridor_width:
                continue
            f = lane.along_of((m.x, m.y)) * fwd
            if mine:
                if not have_a or f > a_front:
                    a_front, have_a = f, True
            else:
                if not have_e or f < e_front:     # the enemy furthest toward us
                    e_front, have_e = f, True
                limit = f - (m.acquisition_range + standoff)
                if limit < safe_f:
                    safe_f = limit
        if have_a and have_e and e_front - a_front <= waves_met_gap:
            f = 0.5 * (a_front + e_front) - holdback_distance
            return lane_point(min(f, safe_f))
        if have_a:
            hold_f = a_front - wave_follow_margin
        elif have_e:
            hold_f = e_front - waves_met_gap
        else:
            return lane_point(min(own_turret_f, safe_f)) if own_turret_f else \
                lane.point_at(lane.length * 0.5)
        if hold_f < own_turret_f:
            hold_f = own_turret_f
        return lane_point(min(hold_f, safe_f))

    st = {"retreating": False}

    def policy(champ, minions, i: int, allies=()):
        # branch 1 -- retreat, with the bot's hysteresis latch
        if champ.max_hp > 0:
            frac = champ.hp / champ.max_hp
            if frac < retreat_hp_frac:
                st["retreating"] = True
            elif frac > resume_hp_frac:
                st["retreating"] = False
            if st["retreating"]:
                # `Retreat`: `PointAt(along - forward * RetreatDistance)`
                along = lane.along_of((champ.x, champ.y))
                return Decision(attack=None,
                                move=lane.point_at(along - fwd * retreat_distance))
        # branch 8a -- plain-auto last hit
        d = decide(champ, minions, lethal_epsilon=0.0)
        if d.attack is not None:
            return d
        if not minions:
            return Decision(attack=None, move=hold_point(minions, allies))
        # branch 8c -- `IsApproachTarget`: hp <= AutoAttackDamage * factor
        cx, cy = champ.x, champ.y
        cands = [m for m in minions
                 if m.hp <= post_mitigation(champ.attack_damage, m.armor) * approach_factor]
        if cands:
            m = min(cands, key=lambda u: ((u.x - cx) ** 2 + (u.y - cy) ** 2, u.uid))
            dist = math.hypot(m.x - cx, m.y - cy)
            reach = champ.attack_range + m.collision_radius
            if dist > reach:
                ux, uy = (m.x - cx) / (dist or 1.0), (m.y - cy) / (dist or 1.0)
                step = min(dist - reach * 0.9, 400.0)
                return Decision(attack=None, move=(cx + ux * step, cy + uy * step))
        # branch 9 -- the real HoldPoint
        return Decision(attack=None, move=hold_point(minions, allies))

    return policy


def _advance_approach(x: float, y: float, idx: int,
                      respawned: bool = False,
                      stalled: int = 0) -> int:
    """Bump ``idx`` into :data:`APPROACH_WAYPOINTS` once arrived at it.

    Closed-loop on each engine's OWN reported position rather than a
    fixed decision-count schedule, so the two drivers do not need identical
    movement speeds or tick timing to stay in step -- each one advances
    exactly when IT decides it has arrived.

    ``respawned`` restarts the walk, and without it one death ended the
    experiment. The index is monotonic, so a champion that died went back to
    the fountain with the approach already exhausted and simply stood there --
    measured: dead at ~120 s, then parked at (26, 280) for the remaining 440
    seconds of a 600 s episode. The sim was being scored on 120 s of farming
    against the server's 600, which is not a comparison of last-hitting.
    """
    if respawned:
        return 0
    if idx >= len(APPROACH_WAYPOINTS):
        return idx
    tx, ty = APPROACH_WAYPOINTS[idx]
    d2 = (x - tx) ** 2 + (y - ty) ** 2
    # The final waypoint is a stopping point, not a waypoint: see
    # `FINAL_ARRIVE_RADIUS`. Advancing early there leaves the champion parked
    # up to `ARRIVE_RADIUS` from where the other engine parks, and that offset
    # -- not either engine's pathing -- drove gate 3's XP gap.
    last = idx == len(APPROACH_WAYPOINTS) - 1
    if d2 <= (_FINAL_ARRIVE_RADIUS_SQ if last else _ARRIVE_RADIUS_SQ):
        return idx + 1
    if last and stalled >= FINAL_ARRIVE_GRACE:
        # Bounded wait: hand over rather than hang. See FINAL_ARRIVE_GRACE.
        # The caller reports the miss, because a driver that waits forever is
        # worse than one that is visibly approximate.
        return idx + 1
    return idx


@dataclass(slots=True, frozen=True)
class SimRun:
    """One oracle-driven episode in the JAX sim.

    ``attacks``/``moves``/``holds`` count only decisions AFTER the scripted
    approach hands control to ``decide()`` -- see the module docstring. They
    are what isolates last-hitting from the walk-in; ``approach_decisions`` is
    reported separately so the split itself is visible.
    """

    cs: int
    decisions: int
    approach_decisions: int
    attacks: int
    moves: int
    holds: int
    #: champion deaths. Each one costs a respawn plus the walk back, so a
    #: difference here is worth more CS than any per-swing effect.
    deaths: int = 0
    #: mean red minions the oracle could SEE per engaged decision. The sim
    #: filters through `obs/fog.visible_to`, an approximation; the server
    #: filters on the wire's own `vb` flag, which is the real thing. A gap
    #: here is a fog-model gap, not a last-hitting gap.
    vis_mean: float = 0.0
    #: decisions spent on each SEPARATE walk-in, in order: index 0 is the
    #: opening walk from the fountain, each later entry one respawn walk.
    #: `approach_decisions` is their sum, and the sum on its own is not
    #: attributable -- a side that dies more walks more times, so a larger
    #: total can mean "walks slower", "walks more often", or "dies during a
    #: walk and restarts it", and those have different fixes. Measured
    #: because the routed gate-3 run split 6,400 sim against 3,197 server and
    #: the total alone could not say which of the three it was.
    walks: tuple = ()


@dataclass(slots=True, frozen=True)
class ServerRun:
    """One oracle-driven episode on the real server. See :class:`SimRun`."""

    cs: int
    decisions: int
    approach_decisions: int
    attacks: int
    moves: int
    holds: int
    deaths: int = 0
    vis_mean: float = 0.0
    #: decisions spent on each SEPARATE walk-in, in order: index 0 is the
    #: opening walk from the fountain, each later entry one respawn walk.
    #: `approach_decisions` is their sum, and the sum on its own is not
    #: attributable -- a side that dies more walks more times, so a larger
    #: total can mean "walks slower", "walks more often", or "dies during a
    #: walk and restarts it", and those have different fixes. Measured
    #: because the routed gate-3 run split 6,400 sim against 3,197 server and
    #: the total alone could not say which of the three it was.
    walks: tuple = ()
    log_path: Optional[Path] = None


def run_oracle_in_sim(
    decisions: int = DECISIONS_600S,
    seed: int = 0,
    *,
    route_table=None,
    terrain=None,
    table_disabled: bool = False,
    on_decision: Optional[Callable[[dict], None]] = None,
    on_oracle: Optional[Callable[[dict], None]] = None,
    policy: Optional[Callable] = None,
) -> SimRun:
    """Run the oracle against blue in the JAX sim; red never receives an order.

    One :func:`~lanerl_jax.sim.step.step_decision` call per decision (30 Hz,
    ``step_ticks=2`` by default -- the production decision rate). The first
    ``approach_decisions`` of them walk :data:`APPROACH_WAYPOINTS` in order and
    never call ``decide()`` at all -- see the module docstring for why. From
    then on the champion order is rebuilt from the oracle's ``Decision`` fresh
    each call, exactly the way a trained policy's orders are rebuilt each
    decision in ``lanerl_jax/train/trainer.py`` -- re-issuing the same attack
    target every decision does not interrupt an in-progress swing, because the
    swing gate is the auto-attack cooldown, not "is this a new order"
    (``sim/autoattack.py``'s docstring).

    ``on_decision``, if given, is called once per decision with the PRE-step
    state and the loop's own bookkeeping (``walk_index``, ``approaching``,
    ``respawned``). It exists so a diagnostic can trace the gate's own run
    rather than re-implementing this loop and silently drifting from it (the
    PATH-006 failure mode: ``isolation.py`` drove the raw path for weeks while
    the gate ran routed). Default ``None`` costs nothing; a callback that pulls
    device arrays pays for those syncs itself.

    ``on_oracle``, if given, is called immediately before every ``decide()``
    call with that call's EXACT inputs.  Both drivers call the same shared
    ``decide()``, so the oracle itself cannot be a source of disagreement --
    any difference in what the two engines do is a difference in what they
    FEED it.  Recording the inputs is therefore the whole of the comparison,
    and it is what lets a diagnostic find the FIRST decision index at which
    the two runs diverge instead of comparing end-of-episode totals.  That
    matters because both engines are bit-reproducible: after the first
    divergence every later number is contaminated, so a final CS of 7 against
    4 is one observation of a cascade, not four independent ones.
    """
    route_table, terrain = gate3_route_inputs(
        route_table=route_table, terrain=terrain, table_disabled=table_disabled)
    # `STRUCT-003`: one step configuration -- TOP lane waves, the tick's
    # INLINE terrain repair, call for help on, routed unless table_disabled.
    sim = SimConfig.scripted(route_table=route_table, terrain=terrain)
    params_tbl = sim.params
    params_np = {k: np.asarray(v) for k, v in params_tbl.items()}
    state = init_lane(seed=seed)

    @jax.jit
    def _step(state, order_kind, order_x, order_y, order_target):
        orders = Orders(
            kind=jnp.array([order_kind, OrderKind.NOOP], dtype=jnp.int8),
            x=jnp.array([order_x, 0.0], dtype=state.x.dtype),
            y=jnp.array([order_y, 0.0], dtype=state.y.dtype),
            target=jnp.array([order_target, -1], dtype=jnp.int8),
        )
        # Canonical parity leaves call-for-help ON by taking `step_decision`'s
        # source-faithful default.  The server has no corresponding off
        # switch: `ObjAIBase.TakeDamage` broadcasts on every landed hit,
        # including a champion last-hit swing.  Earlier gate notes treated an
        # OFF ablation as a baseline after it happened to reduce deaths here;
        # that was regression-shaped tuning, not server parity.
        return env_step(state, orders, sim)

    wp_idx = 0
    _stall = 0   # consecutive decisions on the FINAL approach waypoint
    approach_decisions = attacks = moves = holds = deaths = 0
    walks: list = []
    cur_walk = 0
    vis_counts: list = []
    prev_alive = True
    for _i in range(decisions):
        x0 = float(state.x[0])
        y0 = float(state.y[0])
        champ_alive = bool(state.alive[0])
        if prev_alive and not champ_alive:
            deaths += 1
        respawned = champ_alive and not prev_alive
        prev_alive = champ_alive
        # A death DURING a walk-in restarts it; those decisions were still
        # spent walking, so the segment is closed and recorded rather than
        # folded into the next one.
        if respawned and cur_walk:
            walks.append(cur_walk)
            cur_walk = 0
        _stall = _stall + 1 if wp_idx == len(APPROACH_WAYPOINTS) - 1 else 0
        wp_idx = _advance_approach(x0, y0, wp_idx, respawned, _stall)

        if on_decision is not None:
            on_decision({
                "i": _i, "engine": "sim", "state": state,
                "walk_index": len(walks),
                "approaching": wp_idx < len(APPROACH_WAYPOINTS),
                "wp_idx": wp_idx, "respawned": respawned,
                "alive": champ_alive, "x": x0, "y": y0,
            })

        if wp_idx < len(APPROACH_WAYPOINTS):
            approach_decisions += 1
            cur_walk += 1
            tx, ty = APPROACH_WAYPOINTS[wp_idx]
            state = _step(state, OrderKind.MOVE, tx, ty, -1)
            continue
        if cur_walk:
            walks.append(cur_walk)
            cur_walk = 0

        kind = np.asarray(state.kind)
        team = np.asarray(state.team)
        alive = np.asarray(state.alive)
        x = np.asarray(state.x)
        y = np.asarray(state.y)
        hp = np.asarray(state.hp)
        model = np.asarray(state.model)

        # Champion AD is level-scaled in the sim now (`sim/step.py`'s `tick`,
        # `profiles.py`'s `ad_per_level` column) -- read it the same way here,
        # or the oracle would judge "would this kill" against a damage number
        # lower than what `tick()` actually deals, and pass up kills it could
        # really take.
        champ_ad = float(params_np["attack_damage"][model[0]]
                         + params_np["ad_per_level"][model[0]]
                         * growth_sum(int(np.asarray(state.level)[0])))
        # POSITIONS ARE INT-TRUNCATED HERE TOO. The wire emits
        # `((int)au.Position.X)` (`LanerlControl.cs:188-189`), so the server arm's
        # policy has always seen integer coordinates while the sim arm saw exact
        # floats -- visible in gate 3's own output, where the server champion
        # reads (2806.000, 13075.000) and the sim (2736.387, 13008.614). Any
        # range or distance threshold could therefore resolve differently for a
        # wire-format reason. Truncating the sim side makes the POLICY INPUT
        # symmetric; it does not touch the simulation itself.
        champ = ChampView(
            x=float(int(x0)), y=float(int(y0)),
            attack_damage=champ_ad,
            attack_range=float(params_np["attack_range"][model[0]]),
            hp=float(int(np.asarray(state.hp)[0])),
            max_hp=float(int(np.asarray(state.max_hp)[0])),
        )
        # FOG. The server hands the driver a wire observation that has already
        # had fog of war applied; `LaneState` has no fog, so reading it
        # directly gives blue a view of minions the server's champion cannot
        # see -- including fresh, full-health ones still near red's spawn.
        #
        # That is not a cosmetic difference. The oracle's fallback is "walk to
        # the centroid of the enemy minions", so invisible minions drag the
        # centroid up the lane: the champion walked to (9712, 13138), deep in
        # red's half, met full-health minions that are never one-shot killable,
        # and died 8 times in 600 s. It had a minion in reach on 10,011
        # decisions and a *killable* one on 39.
        #
        # `obs/fog.visible_to` is the same rule the observation builder uses --
        # the union of the team's sight bubbles, not just the champion's.
        vis = np.asarray(visible_to(
            Team.BLUE, state.x, state.y, state.kind, state.team, state.alive))
        # ALLY minions too. `LanerlBot.HoldPoint` needs both wave FRONTS -- it
        # holds behind its own wave (`allyFrontF - WaveFollowMargin`) and only
        # falls back to the enemy front when it has no wave of its own
        # (`LanerlBot.cs:1214-1216`). Passing only enemies made a faithful port
        # impossible, which is why the first version guessed a fixed standoff.
        ally = np.flatnonzero((kind == Kind.LANE_MINION) & (team == Team.BLUE) & alive)
        enemy = np.flatnonzero((kind == Kind.LANE_MINION) & (team == Team.RED)
                               & alive & vis)
        minions = [
            MinionView(
                # positions int-truncated to match the wire, as for the champion
                uid=int(i), x=float(int(x[i])), y=float(int(y[i])), hp=float(hp[i]),
                armor=float(params_np["armor"][model[i]]),
                collision_radius=float(params_np["collision_radius"][model[i]]),
                acquisition_range=float(params_np["acquisition_range"][model[i]]),
            )
            for i in enemy
        ]
        allies = [
            MinionView(
                uid=int(i), x=float(int(x[i])), y=float(int(y[i])), hp=float(hp[i]),
                armor=float(params_np["armor"][model[i]]),
                collision_radius=float(params_np["collision_radius"][model[i]]),
                acquisition_range=float(params_np["acquisition_range"][model[i]]),
            )
            for i in ally
        ]
        vis_counts.append(len(minions))
        if on_oracle is not None:
            on_oracle({"i": _i, "engine": "sim", "champ": champ,
                       "minions": minions, "level": int(np.asarray(state.level)[0])})
        d = (_call_policy(policy, champ, minions, _i, allies) if policy is not None
             else decide(champ, minions, lethal_epsilon=0.0))
        if on_oracle is not None:
            on_oracle({"i": _i, "engine": "sim", "decision": d})
        if d.attack is not None:
            attacks += 1
            state = _step(state, OrderKind.ATTACK, 0.0, 0.0, d.attack)
        elif d.move is not None:
            moves += 1
            mx, my = d.move
            state = _step(state, OrderKind.MOVE, mx, my, -1)
        else:
            holds += 1
            state = _step(state, OrderKind.NOOP, 0.0, 0.0, -1)

    cs = int(np.asarray(state.cs)[0])
    return SimRun(cs=cs, decisions=decisions, approach_decisions=approach_decisions,
                  attacks=attacks, moves=moves, holds=holds, deaths=deaths,
                  vis_mean=float(np.mean(vis_counts)) if vis_counts else 0.0,
                  walks=tuple(walks + ([cur_walk] if cur_walk else [])))


def run_oracle_on_server(
    decisions: int = DECISIONS_600S,
    port_base: int = 44100,
    bot_seed: int = 4242,
    tag: str = "last_hit_oracle",
    log_dir: Optional[Path] = None,
    autobuy: bool = False,
    on_decision: Optional[Callable[[dict], None]] = None,
    on_oracle: Optional[Callable[[dict], None]] = None,
    extra_env: Optional[dict] = None,
    policy: Optional[Callable] = None,
    server_dir: Optional[Path] = None,
) -> ServerRun:
    """Run the oracle against blue on a real server; red is never sent an order.

    Booted exactly like :func:`lanerl_jax.parity.record.record_trace`:
    ``toponly=True`` (no other lanes or jungle), ``bot_teams="none"`` (the
    in-server scripted bot never touches either champion, so the oracle's
    orders are the only thing moving blue), ``step_ticks=2`` (30 Hz, matching
    the sim's default). Like the sim driver, the first ``approach_decisions``
    walk :data:`APPROACH_WAYPOINTS` in order (real ``GetPath`` pathfinding on
    this side, but the same route) before ``decide()`` is ever called -- see
    the module docstring.

    Two things cost real debugging time the first time and are handled here
    so they are not rediscovered:

    * unit id ``0`` is the wire's documented "no target" value and
      ``LanerlWire`` rejects an ``attack`` order carrying it -- never a
      concern here because :func:`~lanerl_jax.parity.last_hit_oracle.decide`
      only ever returns a real minion's NetId, never 0.
    * an ``attack`` on a unit that is not currently visible is accepted by
      the wire and then silently cleared on the very next tick
      (``ObjAIBase.UpdateTarget`` drops a target that fails
      ``IsVisibleByTeam``), so the champion never swings and the order looks
      accepted. Only ``"vb"``-visible red minions are ever offered to the
      oracle as candidates.

    ``autobuy`` defaults to **False here, deliberately inverting the server's
    own default**, because the sim has no item model at all (``ITEM-001``) and
    a comparison against a server champion carrying items is not a parity
    measurement -- it is two different champions. Measured bias from leaving it
    on: **+82.4 max HP** from t=0 (``80 x 1.03``, i.e. +12.3% at level 1),
    **+1.2 HP/s regen** on a base of 1.568 (**+77%**, up to +720 HP of healing
    over a 600 s episode), and **+25 move speed** from t=81.1 s (345 -> 370,
    +7.25%). Every existing caller already passes ``False``; the old ``True``
    default was a trap waiting for the next one, and it is also how
    ``RUNE_HP_BONUS`` came to be measured off a shop-ON dump and to silently
    carry Doran's Shield's +80 for weeks (``STAT-001``).

    Note what this does *not* mean: production RL runs DO use the auto-shop, so
    gate 3 validates the sim against a server configured the way the sim can
    represent, not against the production configuration. Closing that gap needs
    an item model, not a flag.

    ``autobuy`` controls ``LANERL_AUTOBUY`` (default server behaviour is ON --
    ``LanerlHooks.cs:366``'s gate is
    ``GetEnvironmentVariable("LANERL_AUTOBUY") != "0"``, so simply not setting
    the variable, which every earlier version of this function did, leaves it
    on). ``LanerlHooks.AutoBuyUndriven`` is fountain-gated and fires for free,
    with NO action from either driver: once at boot, on the champion's
    starting gold, before the very first observation frame is even returned
    (``lanerl_rl/reward.py``'s own note: Doran's Shield, item 1054, 475g,
    bought before Python ever sees a frame), and again on every walk back from
    a death (``ShopState.BuyOutOnRespawn``, buying out everything affordable).
    Item 1054's Content data (`Items/1054/1054.json`) grants
    ``FlatHPPoolMod=80`` and its script (`ItemPassives/ItemID_1054.cs`) adds
    ``HealthRegeneration.BaseBonus += 1.2`` -- +80 max HP and +1.2 HP/s regen
    the server's champion has from the FIRST tick, that the sim has nowhere
    (there is no item system in `LaneState` at all, and building one is well
    outside J1's scope). Left ``True`` by default so this function's behaviour
    is unchanged for any other caller; the last-hit GATE passes ``False``,
    because a gate built to isolate last-hitting should not also be silently
    scoring a defensive item the sim cannot have.

    ``on_decision`` mirrors :func:`run_oracle_in_sim`'s: called once per
    decision with the raw wire observation for that decision, so a diagnostic
    traces the gate's own run instead of re-implementing this loop.

    ``extra_env`` adds environment variables to the launched server on top of
    ``LANERL_AUTOBUY``. It exists for the env-gated, behaviour-neutral traces
    in :mod:`lanerl_jax.parity.targets` (``TRACE_ENV``): the wire carries no
    per-minion ``TargetUnit``, so ``LANERL_AGGRO_TRACE=1`` is the only way to
    observe when a server minion acquires the champion. Both switches are read
    once into a ``static readonly`` and guard only a ``Console.WriteLine``, and
    the canonical gate-3 server totals (cs 4 / attacks 86) reproduce exactly
    with them on -- see ``docs/TARGET_ACQUISITION_DIFF.md``. ``LANERL_AUTOBUY``
    cannot be overridden through here: it is a behaviour switch this function
    already owns through ``autobuy``, and letting a "trace" dict silently flip
    it is how a diagnostic ends up measuring a different run.
    """
    extra_env = dict(extra_env or {})
    if "LANERL_AUTOBUY" in extra_env:
        raise ValueError("set LANERL_AUTOBUY through autobuy=, not extra_env")
    from lanerl_train.ports import PortAllocator
    from lanerl_train.vec import ServerLaunchSpec, VecLaneEnv

    from ..data.patch import load_patch

    patch = load_patch()
    log_dir = Path(log_dir) if log_dir is not None else Path(
        tempfile.mkdtemp(prefix="last_hit_oracle_"))

    # `paths.server_dir()` is bin/Release, which is a SEPARATE build from the
    # instrumented bin/Trace one. Any env-gated diagnostic added to the vendored
    # source is absent from Release until Release is rebuilt, and the failure is
    # SILENT: the variable is set, the server ignores it, and the run looks
    # normal. Measured cost: a 24-seed outcome sweep whose shuffled-server
    # reference came back 0.000 on every metric, because both arms ran the same
    # unpatched Release binary. A reference gap of exactly zero turns every
    # comparison into one against zero -- the exact-parity trap the outcome gate
    # exists to escape.
    env = VecLaneEnv(
        1,
        spec=ServerLaunchSpec(
            server_dir=server_dir,
            toponly=True, bot_teams="none", bot_seed=bot_seed, step_ticks=2,
            extra_env={**extra_env,
                       "LANERL_AUTOBUY": "1" if autobuy else "0"}),
        log_dir=log_dir,
        ports=PortAllocator(base=port_base).allocate(1),
        step_timeout_s=180.0,
        auto_restart=False,
    )
    env.start()
    wp_idx = 0
    _stall = 0   # consecutive decisions on the FINAL approach waypoint
    approach_decisions = attacks = moves = holds = deaths = 0
    walks: list = []
    cur_walk = 0
    vis_counts: list = []
    prev_alive = True
    log_path: Optional[Path] = None
    try:
        if not all(env.alive):
            raise RuntimeError(f"server failed to boot: {env.alive}")
        log_path = Path(env.handles[0].log_path)
        for i in range(decisions):
            obs = env.last_obs[0]
            if obs is None:
                raise RuntimeError(f"no observation at decision {i}")
            units = obs.get("u", [])
            blue = next(
                u for u in units if u.get("k") == "Champion" and u.get("tm") == 100)
            bx, by = float(blue["x"]), float(blue["y"])
            champ_alive = float(blue.get("hp", 1)) > 0
            if prev_alive and not champ_alive:
                deaths += 1
            respawned = champ_alive and not prev_alive
            prev_alive = champ_alive
            # A death DURING a walk-in restarts it; those decisions were still
            # spent walking, so the segment is closed and recorded rather than
            # folded into the next one.
            if respawned and cur_walk:
                walks.append(cur_walk)
                cur_walk = 0
            _stall = _stall + 1 if wp_idx == len(APPROACH_WAYPOINTS) - 1 else 0
            wp_idx = _advance_approach(bx, by, wp_idx, respawned, _stall)

            if on_decision is not None:
                on_decision({
                    "i": i, "engine": "server", "obs": obs, "blue": blue,
                    "walk_index": len(walks),
                    "approaching": wp_idx < len(APPROACH_WAYPOINTS),
                    "wp_idx": wp_idx, "respawned": respawned,
                    "alive": champ_alive, "x": bx, "y": by,
                })

            if wp_idx < len(APPROACH_WAYPOINTS):
                approach_decisions += 1
                cur_walk += 1
                tx, ty = APPROACH_WAYPOINTS[wp_idx]
                env.step([{"blue": {"t": "move", "x": tx, "y": ty}}])
                continue
            if cur_walk:
                walks.append(cur_walk)
                cur_walk = 0

            champ = ChampView(x=bx, y=by,
                              attack_damage=float(blue["ad"]),
                              attack_range=float(blue["rng"]),
                              hp=float(blue.get("hp", 0)),
                              max_hp=float(blue.get("mhp", 0)))

            minions = []
            allies = []
            for u in units:
                if u.get("k") != "LaneMinion":
                    continue
                if u.get("tm") == 100:        # own wave -- HoldPoint needs its front
                    st = patch.minions[
                        f"{WIRE_MINION_TYPE.get(int(u.get('mt', 0)), 'melee')}_blue"]
                    allies.append(MinionView(
                        uid=int(u["id"]), x=float(u["x"]), y=float(u["y"]),
                        hp=float(u["hp"]), armor=float(st.armor),
                        collision_radius=float(st.collision_radius),
                        acquisition_range=float(st.acquisition_range)))
                    continue
                if u.get("tm") != 200:
                    continue
                if not u.get("vb", 0):        # visible to blue only
                    continue
                key = WIRE_MINION_TYPE.get(int(u.get("mt", 0)), "melee")
                stat = patch.minions[f"{key}_red"]
                minions.append(MinionView(
                    uid=int(u["id"]), x=float(u["x"]), y=float(u["y"]),
                    hp=float(u["hp"]), armor=float(stat.armor),
                    collision_radius=float(stat.collision_radius),
                    acquisition_range=float(stat.acquisition_range)))

            vis_counts.append(len(minions))
            if on_oracle is not None:
                on_oracle({"i": i, "engine": "server", "champ": champ,
                           "minions": minions, "level": int(blue.get("lvl", 0))})
            d = (_call_policy(policy, champ, minions, i, allies) if policy is not None
                 else decide(champ, minions, lethal_epsilon=0.0))
            if on_oracle is not None:
                on_oracle({"i": i, "engine": "server", "decision": d})
            if d.attack is not None:
                attacks += 1
                act = {"t": "attack", "id": d.attack}
            elif d.move is not None:
                moves += 1
                mx, my = d.move
                act = {"t": "move", "x": mx, "y": my}
            else:
                holds += 1
                act = {"t": "noop"}
            env.step([{"blue": act}])

        obs = env.last_obs[0]
        blue = next(u for u in obs["u"] if u.get("k") == "Champion" and u.get("tm") == 100)
        cs = int(blue.get("cs", 0))
    finally:
        env.close()

    return ServerRun(cs=cs, decisions=decisions, approach_decisions=approach_decisions,
                     attacks=attacks, moves=moves, holds=holds, deaths=deaths,
                     vis_mean=float(np.mean(vis_counts)) if vis_counts else 0.0,
                     walks=tuple(walks + ([cur_walk] if cur_walk else [])),
                     log_path=log_path)
