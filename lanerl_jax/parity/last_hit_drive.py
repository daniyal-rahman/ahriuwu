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
  compute real damage -- see the caveat below, this value does NOT currently
  grow with champion level.
* The **server** driver reads ``ad``/``rng`` straight off the wire's own
  ``Stats.AttackDamage.Total`` / ``Stats.Range.Total``, which DOES grow with
  level. Re-deriving either side's attack damage from a formula instead of
  reading each engine's own number is exactly the mistake
  ``lanerl_rl/constants.py`` documented paying for once already (see
  ``sim/init.py``'s RUNE_AD_BONUS note).

KNOWN ASYMMETRY: the JAX sim does not level-scale champion combat stats
------------------------------------------------------------------------
``lanerl_jax.sim.combat.stat_at_level`` implements the server's non-linear
per-level growth curve, but nothing in :func:`lanerl_jax.sim.step.tick` calls
it for the champion's attack damage -- ``P("attack_damage")`` is the flat,
level-1 profile value for the whole episode, even though ``state.level`` itself
does climb as CS/XP accumulate (``lanerl_jax/sim/step.py:341``). The server's
wire ``"ad"`` field is the champion's REAL, currently-levelled attack damage.
So over a 600 s lane, in which Garen can reach level 6-9, the two drivers feed
the oracle two increasingly different attack-damage numbers for reasons that
have nothing to do with last-hitting: the sim's Garen swings at level-1 power
all episode. This is a pre-existing gap in the sim (not introduced here, and
not this module's to fix -- see ``docs/JAX_REWRITE_PLAN.md``'s J1 "Not built"
list), and it is exactly the kind of thing gate 3 exists to surface: watch for
it in the CS gap and the attack counts before blaming anything else.

KNOWN ASYMMETRY: the JAX sim has no fog of war
-----------------------------------------------
:class:`~lanerl_jax.sim.state.LaneState` carries no visibility field, so the
sim driver's enemy-minion list is every LIVE red minion on the map, full
information. The server driver gates on the wire's ``"vb"`` flag (visible to
blue), which is the server's real, terrain-and-range-aware fog. Concretely
this means the sim's oracle can see -- and walk its `Decision.move` centroid
toward -- red minions still marching near their own spawn, while the server's
oracle only ever reacts to whatever is locally visible near blue's own units.
Not a bug in either driver: the sim state simply has nothing to gate on, and
adding fog to `LaneState` is out of scope here.

This is not a theoretical risk -- it is the LEADING SUSPECT for the gap that
remains after the approach fix below. With the scripted approach in place
(2026-09-16 measurement, same 600 s, same seed): sim CS=2, 39 attack
decisions; server CS=10, 535 attack decisions. Both champions now finish the
walk-in in the same ~43 s (``approach_decisions`` 1285 sim / 1286 server --
strong evidence the two engines cover the SAME route at the SAME rate, so the
approach itself is no longer the story). What differs is everything after
handover: the server's champion, gated by real local fog, gets an order of
magnitude more attack opportunities than the sim's, which is exactly what
"the centroid pulls toward whatever the sim can see, unfiltered by distance"
predicts. Last-hitting itself may still be fine in both; this asymmetry could
easily be manufacturing most of the remaining CS gap by starving the sim's
oracle of chances before last-hitting is ever tested. Confirming that needs
either fog in ``LaneState`` or a driver-side substitute (e.g. clamp the sim's
candidate list to minions within some radius of the champion) -- both out of
scope for this pass; flagging it precisely so the next one does not re-derive
it from scratch.

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
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from ..obs.fog import visible_to
from ..sim.init import TOP_LANE_PATH, init_lane, lane_params
from ..sim.orders import OrderKind, Orders, apply_orders
from ..sim.state import Kind, Team
from ..sim.step import step_decision
from .last_hit_oracle import ChampView, MinionView, decide

__all__ = [
    "DECISIONS_600S", "WIRE_MINION_TYPE", "APPROACH_WAYPOINTS",
    "ARRIVE_RADIUS", "SimRun", "ServerRun",
    "run_oracle_in_sim", "run_oracle_on_server",
]

#: 600 s at the 30 Hz decision rate (``step_ticks=2`` off a 60 Hz sim).
DECISIONS_600S = 18_000

#: wire ``MinionSpawnType`` -> patch-table minion key. See the module
#: docstring for why this is not the identity map.
WIRE_MINION_TYPE = {0: "melee", 1: "super", 2: "cannon", 3: "caster"}

#: Blue's forward-order waypoints from ``TOP_LANE_PATH``, up to and including
#: index 6 -- ``(3907.0, 13243.0)``, the vertex where the two lanes' minion
#: waves' paths cross (bit-for-bit ``lanerl_jax.parity.record.MEETING_POINT``).
#: Walked in order before either driver ever calls ``decide()``; see the
#: module docstring for why. Roughly 14,900 units from ``CHAMPION_SPAWN[BLUE]``
#: -- about 40-44 s at Garen's move speed, well inside the pre-wave window
#: (the first wave spawns at 90 s), so the champion always finishes the
#: approach before there is anything to fight.
APPROACH_WAYPOINTS = TOP_LANE_PATH[:7]

#: How close counts as "arrived" at an approach waypoint before advancing to
#: the next one. Generous relative to one decision's travel (~20-25 game units
#: at Garen's move speed and the 30 Hz decision rate), so the champion does
#: not hover just short of the target forever waiting for an exact hit.
ARRIVE_RADIUS = 100.0
_ARRIVE_RADIUS_SQ = ARRIVE_RADIUS * ARRIVE_RADIUS


def _advance_approach(x: float, y: float, idx: int) -> int:
    """Bump ``idx`` into :data:`APPROACH_WAYPOINTS` once arrived at it.

    Closed-loop on each engine's OWN reported position rather than a
    fixed decision-count schedule, so the two drivers do not need identical
    movement speeds or tick timing to stay in step -- each one advances
    exactly when IT decides it has arrived.
    """
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


@dataclass(slots=True, frozen=True)
class ServerRun:
    """One oracle-driven episode on the real server. See :class:`SimRun`."""

    cs: int
    decisions: int
    approach_decisions: int
    attacks: int
    moves: int
    holds: int
    log_path: Optional[Path] = None


def run_oracle_in_sim(decisions: int = DECISIONS_600S, seed: int = 0) -> SimRun:
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
    """
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
        return step_decision(apply_orders(state, orders), params_tbl, lane_path=path)

    wp_idx = 0
    approach_decisions = attacks = moves = holds = 0
    for _ in range(decisions):
        x0 = float(state.x[0])
        y0 = float(state.y[0])
        wp_idx = _advance_approach(x0, y0, wp_idx)

        if wp_idx < len(APPROACH_WAYPOINTS):
            approach_decisions += 1
            tx, ty = APPROACH_WAYPOINTS[wp_idx]
            state = _step(state, OrderKind.MOVE, tx, ty, -1)
            continue

        kind = np.asarray(state.kind)
        team = np.asarray(state.team)
        alive = np.asarray(state.alive)
        x = np.asarray(state.x)
        y = np.asarray(state.y)
        hp = np.asarray(state.hp)
        model = np.asarray(state.model)

        champ = ChampView(
            x=x0, y=y0,
            attack_damage=float(params_np["attack_damage"][model[0]]),
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
                 attacks=attacks, moves=moves, holds=holds)


def run_oracle_on_server(
    decisions: int = DECISIONS_600S,
    port_base: int = 44100,
    bot_seed: int = 4242,
    tag: str = "last_hit_oracle",
    log_dir: Optional[Path] = None,
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
    """
    from lanerl_train.ports import PortAllocator
    from lanerl_train.vec import ServerLaunchSpec, VecLaneEnv

    from ..data.patch import load_patch

    patch = load_patch()
    log_dir = Path(log_dir) if log_dir is not None else Path(
        tempfile.mkdtemp(prefix="last_hit_oracle_"))

    env = VecLaneEnv(
        1,
        spec=ServerLaunchSpec(
            toponly=True, bot_teams="none", bot_seed=bot_seed, step_ticks=2),
        log_dir=log_dir,
        ports=PortAllocator(base=port_base).allocate(1),
        step_timeout_s=180.0,
        auto_restart=False,
    )
    env.start()
    wp_idx = 0
    approach_decisions = attacks = moves = holds = 0
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
            wp_idx = _advance_approach(bx, by, wp_idx)

            if wp_idx < len(APPROACH_WAYPOINTS):
                approach_decisions += 1
                tx, ty = APPROACH_WAYPOINTS[wp_idx]
                env.step([{"blue": {"t": "move", "x": tx, "y": ty}}])
                continue

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
                     attacks=attacks, moves=moves, holds=holds, log_path=log_path)
