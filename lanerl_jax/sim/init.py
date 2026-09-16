"""Build a starting lane state, and spawn waves into it.

Where the numbers come from, and why two sources
------------------------------------------------
**Rules** come from the C# source.  **Stats** come from the patch table
(``Content``).  **Geometry** comes from a recorded state dump, because the map
package format is a separate parser this project does not need: every turret,
nexus, inhibitor and spawn point is already in a ``LANERL_STATEROW`` snapshot at
``t=0``, at the server's own 1/16-unit resolution.  Reading the reference is
both cheaper and more trustworthy than re-deriving it.

The measured spawn deltas are the interesting part
--------------------------------------------------
Content is **not** the whole spawn state, and pretending otherwise starts every
episode with the wrong numbers:

=====================  ==========  ============  =============================
quantity               Content     observed      gap
=====================  ==========  ============  =============================
Garen max HP @ L1         616.28    754.248047    +137.968  rune/mastery page
Garen attack damage        57.88     78.134766    +20.2548  "
Garen armor                27.536     36.536133    +9.0      "
turret max HP            1300.0    1550.0        +250.0    outer-turret bonus
=====================  ==========  ============  =============================

There is also a *staging* effect worth knowing: the champion's max HP reads
**672.0** in the ``t=0`` snapshot and 754.0 once play starts, because
``LanerlEpisode`` applies the page over the first ticks rather than at
construction. So "the value at t=0" and "the value during play" are different
questions, and this module targets the second.

These deltas are recorded as named constants with their provenance rather than
folded into the base stats, so that a modern-patch swap changes the base and
leaves the delta visible and separately checkable.
"""
from __future__ import annotations

from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from ..data.patch import PatchTable, load_patch
from .state import (
    CH_SLICE,
    MAX_WAYPOINTS,
    MI_SLICE,
    N_MINIONS,
    N_UNITS,
    TU_SLICE,
    Kind,
    LaneState,
    MoveOrder,
    Team,
    empty_state,
)
from .targeting import MinionType
from .waves import FIRST_WAVE_MS

__all__ = [
    "CHAMPION_SPAWN", "TOP_OUTER_TURRET", "MINION_SPAWN", "TOP_LANE_PATH",
    "RUNE_HP_BONUS", "TURRET_HP_BONUS",
    "lane_params", "init_lane", "spawn_minion",
]

# --- measured from a LANERL_STATEROW snapshot, 2026-09-16 -------------------
#: ``__Spawn_T1`` / ``__Spawn_T2``.
CHAMPION_SPAWN: Dict[int, Tuple[float, float]] = {
    Team.BLUE: (26.0, 280.0),
    Team.RED: (13927.0, 14175.0),
}
#: The two turrets that actually act in a TOPONLY 1v1. Blue's matches
#: ``lanerl_rl.constants.TOP_OUTER_TURRET`` exactly, which is a free cross-check
#: on the extraction.
TOP_OUTER_TURRET: Dict[int, Tuple[float, float]] = {
    Team.BLUE: (574.6, 10220.5),
    Team.RED: (3911.7, 13654.8),
}
#: Lane-minion barracks (first full-health sighting of a new minion).
MINION_SPAWN: Dict[int, Tuple[float, float]] = {
    Team.BLUE: (918.0, 1720.0),
    Team.RED: (12451.0, 13218.0),
}
#: ``LanerlLane.TopLaneDefault`` -- taken verbatim from the map script's
#: ``MinionPaths`` so minions walk the same line the server walks them along.
TOP_LANE_PATH: Tuple[Tuple[float, float], ...] = (
    (917.0, 1725.0), (1170.0, 4041.0), (861.0, 6459.0), (880.0, 10180.0),
    (1268.0, 11675.0), (2806.0, 13075.0), (3907.0, 13243.0), (7550.0, 13407.0),
    (10244.0, 13238.0), (10947.0, 13135.0), (12511.0, 12776.0),
)

#: Champion max HP above the Content base curve, from the rune/mastery page.
#:
#: Taken from the dump's exact quantised value, **772348/1024 = 754.248046875**,
#: not from the 754.0 a rounded readout shows. The difference is 0.248 HP and it
#: was the only champion field still disagreeing in a 580-second side-by-side --
#: which is a good argument for reading the oracle at its own resolution rather
#: than at display precision.
RUNE_HP_BONUS = 754.248046875 - 616.28
#: Outer turret max HP above Content. 1550 observed vs 1300 in
#: ``SRUAP_Turret_Order3``.
TURRET_HP_BONUS = 250.0

#: The rest of the rune page, measured the same way -- from the dump's own
#: quantised values in a 600 s idle run, against the Content base.
#:
#: `lanerl_rl/constants.py` states the lesson these exist to avoid, having paid
#: for it: *"constants.garen_attack_damage() read 57.88 at level 1, then 73.14
#: once the rune page was modelled, against a true 78.14 -- a re-derived server
#: quantity is wrong by however much of the server you forgot."* So these are
#: **measured deltas**, not a reconstruction of which runes the page contains.
RUNE_AD_BONUS = 78.134765625 - 57.88            # +20.2548
RUNE_ARMOR_BONUS = 36.5361328125 - 27.5361328125  # +9.0


def lane_params(patch: PatchTable | None = None, dtype=jnp.float32) -> dict:
    """The stat tables the tick gathers through ``state.model``.

    Just :func:`lanerl_jax.sim.profiles.build_profile_tables`; kept as a named
    entry point because the tick's contract is "params + state", and callers
    should not have to know which module the rows come from.
    """
    from .profiles import build_profile_tables

    return build_profile_tables(patch, dtype)


def _legacy_lane_params(patch: PatchTable | None = None, dtype=jnp.float32) -> dict:
    from .combat import attack_period, attack_speed_flat, attack_windup

    patch = patch or load_patch()
    g = patch.champion
    melee = patch.minions["melee_blue"]
    turret = next(iter(patch.turrets.values()))

    n = N_UNITS
    z = lambda v: np.full(n, v, dtype=np.float32)   # noqa: E731

    def period_windup(u):
        flat = attack_speed_flat(patch.global_attack_delay,
                                 u.attack_delay_offset_percent)
        p = attack_period(flat)
        return p, attack_windup(p, patch.global_attack_delay_cast_percent,
                                u.attack_delay_cast_offset_percent)

    gp, gw = period_windup(g)
    mp, mw = period_windup(melee)
    tp, tw = period_windup(turret)

    move = z(melee.move_speed); acq = z(melee.acquisition_range or 600.0)
    rng = z(melee.attack_range); col = z(melee.collision_radius or 48.0)
    per = z(mp); win = z(mw); ad = z(melee.base_ad); ar = z(melee.armor)

    for sl, u, (p_, w_) in ((CH_SLICE, g, (gp, gw)), (TU_SLICE, turret, (tp, tw))):
        move[sl] = u.move_speed
        acq[sl] = u.acquisition_range or 600.0
        rng[sl] = u.attack_range
        col[sl] = u.collision_radius or 48.0
        per[sl] = p_
        win[sl] = w_
        ad[sl] = u.base_ad
        ar[sl] = u.armor
    move[TU_SLICE] = 0.0        # turrets never move

    mt = np.full(n, MinionType.MELEE, np.int8)
    return {
        "move_speed": jnp.asarray(move, dtype),
        "acquisition_range": jnp.asarray(acq, dtype),
        "attack_range": jnp.asarray(rng, dtype),
        "collision_radius": jnp.asarray(col, dtype),
        "attack_period": jnp.asarray(per, dtype),
        "attack_windup": jnp.asarray(win, dtype),
        "attack_damage": jnp.asarray(ad, dtype),
        "armor": jnp.asarray(ar, dtype),
        "minion_type": jnp.asarray(mt),
    }


def init_lane(patch: PatchTable | None = None, dtype=jnp.float32,
              seed: int = 0, include_all_turrets: bool = False) -> LaneState:
    """A fresh top-lane 1v1 at ``t = 0``: two champions, two turrets, no minions.

    ``include_all_turrets`` is off by default. All 24 map turrets exist on the
    server even under ``TOPONLY`` (measured), and carrying them costs nothing in
    a masked array -- but only the top outer pair can ever act in this scenario,
    so the default keeps the state legible. Turn it on when diffing against a
    dump without scoping.
    """
    patch = patch or load_patch()
    s = empty_state(dtype=dtype, seed=seed)
    n = N_UNITS

    kind = np.zeros(n, np.int8)
    team = np.full(n, Team.NEUTRAL, np.int8)
    alive = np.zeros(n, bool)
    x = np.zeros(n, np.float32)
    y = np.zeros(n, np.float32)
    hp = np.zeros(n, np.float32)

    from .profiles import profile_id
    model = np.zeros(n, np.int8)

    champ_hp = patch.champion.hp_at_level(1) + RUNE_HP_BONUS
    for i, t in enumerate((Team.BLUE, Team.RED)):
        kind[i] = Kind.CHAMPION
        team[i] = t
        model[i] = profile_id(Kind.CHAMPION, -1, t)
        x[i], y[i] = CHAMPION_SPAWN[t]
        hp[i] = champ_hp
        alive[i] = True

    turret_base = next(iter(patch.turrets.values())).base_hp + TURRET_HP_BONUS
    t0 = TU_SLICE.start
    for j, t in enumerate((Team.BLUE, Team.RED)):
        i = t0 + j
        kind[i] = Kind.TURRET
        team[i] = t
        model[i] = profile_id(Kind.TURRET, -1, t)
        x[i], y[i] = TOP_OUTER_TURRET[t]
        hp[i] = turret_base
        alive[i] = True

    return s.replace(
        model=jnp.asarray(model),
        spawn_x=jnp.asarray(x, dtype), spawn_y=jnp.asarray(y, dtype),
        kind=jnp.asarray(kind), team=jnp.asarray(team), alive=jnp.asarray(alive),
        x=jnp.asarray(x, dtype), y=jnp.asarray(y, dtype),
        hp=jnp.asarray(hp, dtype), max_hp=jnp.asarray(hp, dtype),
        next_spawn_ms=jnp.asarray(FIRST_WAVE_MS, dtype),
        move_order=jnp.full((n,), MoveOrder.NONE, jnp.int8),
    )


def spawn_minion(state: LaneState, team, profile, hp,
                 path: jax.Array, enabled=True) -> LaneState:
    """Write one minion into the lowest free minion slot.

    Lowest free slot, not a random or round-robin one: the server's target
    acquisition breaks ties by object-collection order and ``argmin`` takes the
    lowest index, so slot assignment is part of the parity surface (see
    ``LaneState``'s docstring). Deterministic assignment keeps the two orders
    comparable.

    Silently does nothing when every slot is taken. That is the one place a
    fixed-shape sim can lose a unit, so the caller must watch it -- the measured
    p99 is 27 live minions against 40 slots, but a cap is a promise about a
    distribution, not a guarantee.
    """
    free = (~state.alive) & (jnp.arange(N_UNITS) >= MI_SLICE.start) \
        & (jnp.arange(N_UNITS) < MI_SLICE.stop)
    i = jnp.argmax(free)
    # `enabled` lets the caller invoke this every tick unconditionally: under
    # `vmap` a `cond` executes both branches anyway, so a masked write is the
    # same cost and keeps the cost visible.
    ok = jnp.any(free) & jnp.asarray(enabled)

    sx, sy = path[0, 0], path[0, 1]
    wp = jnp.zeros((MAX_WAYPOINTS, 2), state.x.dtype).at[:path.shape[0]].set(path)

    def setv(arr, v):
        return jnp.where(ok, arr.at[i].set(v), arr)

    return state.replace(
        kind=setv(state.kind, jnp.int8(Kind.LANE_MINION)),
        team=setv(state.team, jnp.asarray(team, jnp.int8)),
        alive=setv(state.alive, True),
        model=setv(state.model, jnp.asarray(profile, jnp.int8)),
        x=setv(state.x, sx), y=setv(state.y, sy),
        hp=setv(state.hp, jnp.asarray(hp, state.hp.dtype)),
        max_hp=setv(state.max_hp, jnp.asarray(hp, state.hp.dtype)),
        waypoints=jnp.where(ok, state.waypoints.at[i].set(wp), state.waypoints),
        n_waypoints=setv(state.n_waypoints, jnp.int8(path.shape[0])),
        waypoint_key=setv(state.waypoint_key, jnp.int8(1)),
        move_order=setv(state.move_order, jnp.int8(MoveOrder.MOVE_TO)),
        target=setv(state.target, jnp.int8(-1)),
        ai_timer=setv(state.ai_timer, jnp.asarray(250.0, state.x.dtype)),
    )
