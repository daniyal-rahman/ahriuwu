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

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from ..obs.fog import visible_to
from ..sim.combat import growth_sum
from ..sim.init import TOP_LANE_PATH, init_lane, lane_params
from ..sim.orders import OrderKind, Orders, apply_orders
from ..sim.state import Kind, Team
from ..sim.step import step_decision
from .last_hit_oracle import ChampView, MinionView, decide

__all__ = [
    "DECISIONS_600S", "WIRE_MINION_TYPE", "APPROACH_WAYPOINTS",
    "ARRIVE_RADIUS", "SimRun", "ServerRun",
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


def _advance_approach(x: float, y: float, idx: int,
                      respawned: bool = False) -> int:
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
    if (x - tx) ** 2 + (y - ty) ** 2 <= _ARRIVE_RADIUS_SQ:
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
    """
    route_table, terrain = gate3_route_inputs(
        route_table=route_table, terrain=terrain, table_disabled=table_disabled)
    params_tbl = lane_params()
    params_np = {k: np.asarray(v) for k, v in params_tbl.items()}
    path = jnp.asarray(np.array(TOP_LANE_PATH, np.float32))
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
        return step_decision(apply_orders(state, orders, params_tbl,
                                          route_table=route_table,
                                          terrain=terrain), params_tbl,
                             lane_path=path)

    wp_idx = 0
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
        wp_idx = _advance_approach(x0, y0, wp_idx, respawned)

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
        champ = ChampView(
            x=x0, y=y0,
            attack_damage=champ_ad,
            attack_range=float(params_np["attack_range"][model[0]]),
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
        enemy = np.flatnonzero((kind == Kind.LANE_MINION) & (team == Team.RED)
                               & alive & vis)
        minions = [
            MinionView(
                uid=int(i), x=float(x[i]), y=float(y[i]), hp=float(hp[i]),
                armor=float(params_np["armor"][model[i]]),
                collision_radius=float(params_np["collision_radius"][model[i]]),
            )
            for i in enemy
        ]
        vis_counts.append(len(minions))
        d = decide(champ, minions, lethal_epsilon=0.0)
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
    autobuy: bool = True,
    on_decision: Optional[Callable[[dict], None]] = None,
    extra_env: Optional[dict] = None,
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

    env = VecLaneEnv(
        1,
        spec=ServerLaunchSpec(
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
            wp_idx = _advance_approach(bx, by, wp_idx, respawned)

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
                              attack_damage=float(blue["ad"]), attack_range=float(blue["rng"]))

            minions = []
            for u in units:
                if u.get("k") != "LaneMinion" or u.get("tm") != 200:
                    continue
                if not u.get("vb", 0):        # visible to blue only
                    continue
                key = WIRE_MINION_TYPE.get(int(u.get("mt", 0)), "melee")
                stat = patch.minions[f"{key}_red"]
                minions.append(MinionView(
                    uid=int(u["id"]), x=float(u["x"]), y=float(u["y"]),
                    hp=float(u["hp"]), armor=float(stat.armor),
                    collision_radius=float(stat.collision_radius)))

            vis_counts.append(len(minions))
            d = decide(champ, minions, lethal_epsilon=0.0)
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
