"""Tier 2: run both simulations from the same start and measure where they part.

Why Tier 2 and not Tier 1
-------------------------
The plan's Tier 1 is a *one-step differential*: inject the server's state into
the sim, step both once, diff.  It is the sharpest test available and it is
**not reachable with the current oracle**, because injection needs every field
the step depends on and the state dump carries none of the following
(:data:`lanerl_jax.parity.diff.UNOBSERVABLE`): waypoint positions, the
auto-attack clock, minion-target identity, the minion AI's 250 ms timer, its
local clock, or its ignore list.  Injecting the observable subset and leaving
the rest at defaults does not produce "the server's state one step on" -- it
produces a different state that happens to share some fields.

So the honest measurement is Tier 2: identical starting conditions, identical
action stream, both free-running, and **report when and how they separate**.
That is weaker but it is real, and it is exactly what the plan says Tier 2 is
for -- quantifying drift rather than pretending it away.

What "identical start" can and cannot mean
------------------------------------------
Geometry, stats and the wave schedule are shared (all three come from the server
itself). What is *not* shared is the server's hidden initialisation: rune pages
applied over the first ticks, the exact order the object manager was populated
in, and every field above. So divergence at t=0+ is expected and the question is
its *rate*, not its existence.

Reading the result
------------------
``first_divergence_s`` is when a scoped field first exceeds tolerance.
``cs_gap`` and ``population_gap`` are the behavioural summaries that matter more
than any single field: an agent trained here has to farm on the server, and CS
is the thing it is trained to produce.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .diff import LANE_KINDS, DEFAULT_TOLERANCE, Tolerance, diff_snapshots
from .trace import Entity, Snapshot, load_trace

__all__ = ["SimVsServer", "state_to_snapshot", "compare_streams"]


def state_to_snapshot(state, t_ms: float, params=None) -> Snapshot:
    """Render a :class:`~lanerl_jax.sim.state.LaneState` as a
    :class:`~lanerl_jax.parity.trace.Snapshot`.

    Rendering into the *server's* format means the existing differ works on both
    sides unchanged, rather than there being a second comparison path that can
    disagree with the first. The AI and champion blocks are emitted even where
    the sim does not model a field, because an absent block reads as a
    structural mismatch on every row; the unmodelled fields are excluded through
    ``Tolerance.ignore_fields`` instead, which *names* them in the report.
    """
    from ..sim.state import Kind, MoveOrder, Team
    from ..parity.trace import AIBlock, ChampionBlock

    kind_name = {Kind.CHAMPION: "Champion", Kind.LANE_MINION: "LaneMinion",
                 Kind.TURRET: "LaneTurret"}
    server_team = {Team.BLUE: 100, Team.RED: 200, Team.NEUTRAL: 300}
    # `LaneState.move_order` uses the same numbering as OrderType for the subset
    # the lane exercises, so it goes out unmapped.
    ents: List[Entity] = []
    k = np.asarray(state.kind)
    alive = np.asarray(state.alive)
    for i in np.flatnonzero((k != Kind.NONE) & alive):
        i = int(i)
        ai = AIBlock(
            move_order=int(state.move_order[i]),
            waypoints=int(state.n_waypoints[i]),
            cast_spell="-", channel_spell="-", can_move=True, buffs=(),
        )
        champ = None
        if int(k[i]) == Kind.CHAMPION:
            champ = ChampionBlock(
                q_ad=int(round(float(params["attack_damage"][state.model[i]]) * 1024))
                if params is not None else 0,
                q_armor=int(round(float(params["armor"][state.model[i]]) * 1024))
                if params is not None else 0,
                q_mr=0, q_move_speed=int(round(
                    float(params["move_speed"][state.model[i]]) * 1024))
                if params is not None else 0,
                q_attack_speed_mult=1024,
                level=int(state.level[i]),
                q_gold=int(round(float(state.gold[i]) * 1024)),
                minions_killed=int(state.cs[i]),
                deaths=int(state.deaths[i]),
                skill_points=0,
                spells=((0, 0), (0, 0), (0, 0), (0, 0)),
            )
        ents.append(Entity(
            kind=kind_name.get(int(k[i]), "Other"),
            q_x=int(round(float(state.x[i]) * 16)),
            q_y=int(round(float(state.y[i]) * 16)),
            team=server_team[int(state.team[i])],
            q_hp=int(round(float(state.hp[i]) * 1024)),
            q_max_hp=int(round(float(state.max_hp[i]) * 1024)),
            dead=False, ai=ai, champ=champ,
        ))
    return Snapshot(t_ms=int(t_ms), entities=ents)


#: Fields the sim does not model yet. Named so a report says "not modelled"
#: rather than silently agreeing.
NOT_MODELLED = (
    "mr", "attack_speed", "skill_points", "buffs", "can_move",
    "cast_spell", "channel_spell",
    "spell.Q.level", "spell.W.level", "spell.E.level", "spell.R.level",
    "spell.Q.cooldown", "spell.W.cooldown", "spell.E.cooldown",
    "spell.R.cooldown",
    # `Waypoints.Count` differs structurally: the sim stores a fixed-size array
    # and the server a list, and the sim's straight-line re-path produces two
    # waypoints where the server's GetPath may produce several.
    "waypoints", "move_order",
)


@dataclass(slots=True)
class SimVsServer:
    """One comparison run."""

    times_s: List[float] = field(default_factory=list)
    sim_minions: List[int] = field(default_factory=list)
    server_minions: List[int] = field(default_factory=list)
    sim_cs: List[Tuple[int, int]] = field(default_factory=list)
    server_cs: List[Tuple[int, int]] = field(default_factory=list)
    first_divergence_s: Optional[float] = None
    diffs_at: Dict[float, str] = field(default_factory=dict)

    @property
    def population_gap(self) -> float:
        """Relative difference in median live-minion count after the waves settle."""
        n = len(self.times_s)
        lo = int(n * 0.25)
        a = float(np.median(self.sim_minions[lo:])) if n else 0.0
        b = float(np.median(self.server_minions[lo:])) if n else 0.0
        return 0.0 if b == 0 else (a - b) / b

    def report(self) -> str:
        lines = [f"{len(self.times_s)} sample points over "
                 f"{self.times_s[-1] if self.times_s else 0:.0f} s"]
        if self.sim_minions:
            n = len(self.times_s)
            lo = int(n * 0.25)
            lines.append(
                f"  live minions  sim median {np.median(self.sim_minions[lo:]):.0f}"
                f"  server median {np.median(self.server_minions[lo:]):.0f}"
                f"  ({100 * self.population_gap:+.0f}%)")
        if self.sim_cs:
            lines.append(f"  CS at end     sim {self.sim_cs[-1]}  "
                         f"server {self.server_cs[-1]}")
        lines.append(f"  first scoped divergence: "
                     f"{'none' if self.first_divergence_s is None else f'{self.first_divergence_s:.1f}s'}")
        return "\n".join(lines)


def compare_streams(server_log: Path, sim_snapshots: Dict[int, Snapshot],
                    tol: Optional[Tolerance] = None) -> SimVsServer:
    """Diff a recorded server log against sim snapshots keyed by game time."""
    from dataclasses import replace

    tol = tol or replace(DEFAULT_TOLERANCE, kinds=LANE_KINDS,
                         ignore_fields=NOT_MODELLED)
    trace = load_trace(Path(server_log))
    out = SimVsServer()
    by_t = {s.t_ms: s for s in trace}
    for t in sorted(sim_snapshots):
        srv = by_t.get(t)
        if srv is None:
            continue
        sim = sim_snapshots[t]
        out.times_s.append(t / 1000.0)
        out.sim_minions.append(sum(1 for e in sim.entities if e.kind == "LaneMinion"))
        out.server_minions.append(
            sum(1 for e in srv.entities if e.kind == "LaneMinion"))
        d = diff_snapshots(sim, srv, tol)
        if not d.clean and out.first_divergence_s is None:
            out.first_divergence_s = t / 1000.0
            out.diffs_at[t / 1000.0] = d.report(limit=6)
    return out
