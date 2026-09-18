"""2D side-by-side replay: JAX sim vs the League server, from one identical
Tier-1.5 start state.

WHY THIS EXISTS, AND WHY IT IS NOT THE CLIENT
----------------------------------------------
The user wants to *watch* JAX's simulated lane and say what looks wrong. The
obvious-looking answer is "make the real League client render JAX state" --
inject positions/HP into the vendored C# server every tick and let its
existing ENet packet path carry them to a connected client. That was assessed
(see the report this module's docstring is paired with) and is NOT what this
module does, for reasons summarised here and stated in full in the report:

* The server can only be driven to a state, not loaded with one
  (`parity/tier15.py`'s module docstring: `LanerlWire.Parse` recognises
  exactly `{"cmd":"reset"}` and a per-champion action -- there is no
  set-position/set-health/spawn/load-state command). Adding one, suppressing
  the server's own simulation so it does not fight the injection, and mapping
  JAX's minion population onto server-side objects with stable identity
  across a run is real, non-trivial C# work -- and at the end of it the
  client shows an *approximation* (teleported units, no cast/attack VFX tied
  to what actually happened, minion identity reassigned under the hood).
* Meanwhile every piece this module needs already exists and has been proven
  to work together: `state_to_snapshot` (`parity/sim_vs_server.py`) already
  renders a JAX `LaneState` into the server's own `Snapshot`/`Entity` format,
  the server's state dump is already a `Snapshot` stream (`parity/trace.py`),
  and `parity/tier15.py` already knows how to injected-start the sim from a
  server tick and free-run it against the server's own recorded tail under an
  identical order stream -- that is the "one identical Tier-1.5 start state"
  the task asked for, verbatim.

So this module is almost entirely wiring: run the Tier-1.5 injection+free-run
(mirroring `parity.tier15.run_tier15`'s setup -- that function is not reused
directly because it returns divergence summaries, not full per-tick entity
lists, and there is no public hook to make it keep the raw `Snapshot`s it
already computes internally), collect the **full** entity list each tick from
*both* sides (not just the champion scalars / disagreeing minions tier15's
report keeps), and hand them to a two-panel matplotlib animation reusing
`lanerl.render_replay`'s existing single-panel renderer's conventions
(team colours, marker sizing, `FFMpegWriter`).

Nothing under `lanerl_jax/parity/` or `lanerl_jax/sim/` is edited by this
module -- it only imports their public (and, in one case, tier15's
underscore-private but already cross-imported-within-the-package) surface.

WHAT THIS DOES NOT SHOW
------------------------
This is server *state*, not the server's *rendering*: no minimap, no HUD, no
ability VFX, no camera, no animations, no fog-of-war (both sides are drawn
fully visible -- add a fog mask if that distinction ever matters for a
specific question), and no sound. It answers "where are the units and what
are their vitals", which is what the JAX/server behavioural questions this
repo has been chasing (`COLL-003`, `HOLD-001`, `AA-001`, ...) actually needed.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..parity.action_replay import (ActionReplayError, RecordedDecision,
                                    align_action_log, decision_to_orders)
from ..parity.diagnostic_identity import net_id_to_injected_slot
from ..parity.inject import inject_snapshot, replay_wave_states
from ..parity.record import ActionLog, Fixture
from ..parity.sim_vs_server import state_to_snapshot
from ..parity.tier15 import (BLUE, RED, STEP_TICKS, StartSelection,
                             load_trace_upto, select_start)
from ..parity.trace import Snapshot

__all__ = ["FrameStreams", "build_frame_streams", "render_side_by_side",
          "write_summary_plot", "main"]

#: server team ids, matching `parity.sim_vs_server.state_to_snapshot`.
TEAM_COLOR = {100: "#4488ff", 200: "#ff4444", 300: "#888888"}
_SIZE = {"Champion": 120, "LaneTurret": 90, "LaneMinion": 18}


@dataclass(slots=True)
class FrameStreams:
    """Two time-aligned `Snapshot` sequences from one shared start state."""

    sim: List[Snapshot]
    srv: List[Snapshot]
    start: StartSelection
    truncated_reason: Optional[str]


def build_frame_streams(
    fixture: Fixture, selection_kwargs: Dict[str, object], *,
    decisions: int = 600, team: int = BLUE,
    table_disabled: bool = True, route_artifact: Optional[Path] = None,
    max_t_ms: Optional[int] = None, stop_on_death: bool = True,
    patch=None,
) -> FrameStreams:
    """Inject one server tick into the sim, then free-run both sides.

    Mirrors `parity.tier15.run_tier15`'s setup (record -> select_start ->
    inject_snapshot -> replay the fixture's own recorded order stream) but
    keeps the full entity list every tick on both sides instead of reducing
    each tick to divergence scalars -- that reduction is exactly what a
    debugging *replay* must not do.

    `table_disabled=True` by default (straight-line champion moves instead of
    the routed local table): this is a visualisation tool where "does the
    champion's move look wrong" is answered by watching it, not by a route
    fidelity number, and skipping the route-artifact load keeps this cheap to
    run repeatedly. Pass `table_disabled=False` to match tier15's own default
    if a specific pathing question is under study.
    """
    import jax
    import jax.numpy as jnp

    from ..data.patch import load_patch
    from ..parity.one_step import _tick_jit
    from ..sim.init import TOP_LANE_PATH, lane_params
    from ..sim.orders import apply_orders
    from ..sim.profiles import PROFILES
    from ..sim.state import CH_SLICE

    patch = patch or load_patch()
    params = lane_params(patch)
    lane_path = jnp.asarray(np.asarray(TOP_LANE_PATH, np.float32))

    trace = load_trace_upto(fixture.log, max_t_ms)
    actions = ActionLog.load(fixture.actions)
    action_at = align_action_log(trace.snapshots, actions)

    n_ticks = decisions * STEP_TICKS
    sel = select_start(trace, params=params, team=team,
                       tail_ticks=n_ticks + 1, **selection_kwargs)

    route_table = terrain = None
    if not table_disabled:
        from ..data.local_route_artifact import load_local_route_artifact
        from ..sim.terrain_jax import map1_terrain
        from ..train.run_train import DEFAULT_ROUTE_ARTIFACT
        art = load_local_route_artifact(
            Path(route_artifact) if route_artifact else DEFAULT_ROUTE_ARTIFACT,
            pathfinding_radius=35.0)
        route_table = art.as_jax()
        terrain = map1_terrain()

    @jax.jit
    def _order_jit(st, orders):
        return apply_orders(st, orders, params, route_table=route_table,
                            terrain=terrain)

    wave_states = replay_wave_states(trace)
    i0 = sel.index
    previous = trace[i0 - 1] if i0 else None
    if previous is not None and not (0 < trace[i0].t_ms - previous.t_ms <= 34):
        previous = None
    state, rep = inject_snapshot(trace[i0], wave_states[i0], params, PROFILES,
                                previous_snapshot=previous)
    net_id_slots = net_id_to_injected_slot(trace[i0], rep.notes)
    slot_of_net = dict(net_id_slots)

    # Freeze both champions' waypoints on injection ("assumes nothing" --
    # tier15's `--champion-waypoints freeze` control). A viz tool has no
    # stake in the first tick or two of champion movement fidelity, and
    # skipping the last-move re-seed avoids depending on tier15's private
    # `_seed_champion_orders` helper.
    n_wp = np.array(state.n_waypoints, copy=True)
    for s in (CH_SLICE.start, CH_SLICE.start + 1):
        n_wp[s] = 0
    state = state.replace(n_waypoints=jnp.asarray(n_wp))

    sim_snaps: List[Snapshot] = [
        state_to_snapshot(state, t_ms=float(state.t_ms), params=params)]
    srv_snaps: List[Snapshot] = [trace[i0]]
    rows0_deaths = None
    truncated: Optional[str] = None

    for k in range(1, n_ticks + 1):
        j = i0 + k
        if j >= len(trace):
            truncated = f"trace exhausted at index {j}"
            break
        dt = trace[j].t_ms - trace[j - 1].t_ms
        if not (0 < dt <= 34):
            truncated = (f"gap in the server dump at index {j} "
                         f"({trace[j - 1].t_ms} -> {trace[j].t_ms} ms)")
            break

        state = _tick_jit(state, params, lane_path=lane_path)
        dec = action_at.get(j)
        if dec is not None:
            try:
                orders = decision_to_orders(dec, slot_of_net)
            except ActionReplayError:
                orders = decision_to_orders(
                    RecordedDecision(dec.source_t_ms, {"t": "noop"},
                                     {"t": "noop"}), slot_of_net)
            state = _order_jit(state, orders)

        snap = trace[j]
        sim_snap = state_to_snapshot(state, t_ms=float(state.t_ms), params=params)

        if stop_on_death:
            sim_ch = sim_snap.champion(team)
            srv_ch = snap.champion(team)
            sim_deaths = sim_ch.champ.deaths if sim_ch and sim_ch.champ else None
            srv_deaths = srv_ch.champ.deaths if srv_ch and srv_ch.champ else None
            deaths = (sim_deaths, srv_deaths)
            if rows0_deaths is None:
                rows0_deaths = deaths
            if (sim_ch is None or sim_ch.dead or srv_ch is None or srv_ch.dead
                    or deaths != rows0_deaths):
                truncated = (f"champion death at tick {k} (t={snap.t_ms} ms) "
                            "-- a respawn teleport makes anything after this "
                            "not a comparison of the same situation")
                sim_snaps.append(sim_snap)
                srv_snaps.append(snap)
                break

        sim_snaps.append(sim_snap)
        srv_snaps.append(snap)

    return FrameStreams(sim=sim_snaps, srv=srv_snaps, start=sel,
                        truncated_reason=truncated)


# ---------------------------------------------------------------------------
# rendering -- deliberately close to lanerl/render_replay.py's conventions
# ---------------------------------------------------------------------------

def _scatter_xyc_s(snap: Snapshot) -> Tuple[list, list, list, list]:
    X, Y, C, S = [], [], [], []
    for e in snap.entities:
        if e.dead:
            continue
        X.append(e.x); Y.append(e.y)
        C.append(TEAM_COLOR.get(e.team, "#aaaaaa"))
        S.append(_SIZE.get(e.kind, 18))
    return X, Y, C, S


def _frame_bounds(snaps: List[Snapshot], pad: float = 300.0):
    xs = [e.x for s in snaps for e in s.entities]
    ys = [e.y for s in snaps for e in s.entities]
    if not xs:
        raise ValueError("no entities to frame -- nothing to render")
    return (min(xs) - pad, max(xs) + pad), (min(ys) - pad, max(ys) + pad)


def render_side_by_side(streams: FrameStreams, out_mp4: Path, fps: int = 20,
                        frame: str = "sim", window: float = 1600.0,
                        team: int = BLUE, overlay: bool = False) -> None:
    """Animation of both engines from one identical start, on one clock.

    ``frame`` picks the axes, and the default is deliberately **not** the
    union of both engines:

    ``"sim"``
        Bound the axes to the *sim's* units only. This sim models the top
        lane and nothing else (`LANERL_TOPONLY`), while the server's dump
        carries its jungle camps and off-lane structures -- roughly 96 units
        against 38. Framing on the union lets a krug that neither engine is
        being compared on set the scale, and squeezes the lane (the only
        thing under test) into a corner. The server's extra units are not
        hidden, just off-screen; use ``"both"`` to see them.
    ``"both"``
        The union. Honest about scope, useless for looking at a fight.
    ``"follow"``
        A fixed ``window``-wide box centred on ``team``'s champion each
        frame, from the sim's side, so both panels track the same point even
        as the two champions drift apart -- which is the drift you want to
        see.

    ``overlay`` collapses the two panels into one: sim filled, server as an
    open ring. A unit the two engines agree on is a dot inside its ring; a
    positional deviation is a dot that has visibly left its ring. Two panels
    make you eyeball-subtract; one panel does not.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    from matplotlib.lines import Line2D

    if frame not in ("sim", "both", "follow"):
        raise ValueError(f"frame must be sim/both/follow, got {frame!r}")
    static_xlim, static_ylim = _frame_bounds(
        streams.sim + streams.srv if frame == "both" else streams.sim)

    if overlay:
        fig, ax0 = plt.subplots(figsize=(9, 9))
        axes = [ax0]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        for ax, title in zip(axes, ("JAX sim", "LoLServer")):
            ax.set_title(title)
    for ax in axes:
        ax.set_xlim(*static_xlim); ax.set_ylim(*static_ylim)
        ax.set_aspect("equal"); ax.set_facecolor("#12160f")

    if overlay:
        sim_scat = axes[0].scatter([], [], s=[], c=[], zorder=3)
        srv_scat = axes[0].scatter([], [], s=[], facecolors="none",
                                   edgecolors=[], linewidths=1.3, zorder=2)
        scats = [(sim_scat, "sim"), (srv_scat, "srv")]
        leg = axes[0].legend(handles=[
            Line2D([], [], marker="o", ls="", color="#cccccc", label="JAX sim"),
            Line2D([], [], marker="o", ls="", mfc="none", color="#cccccc",
                   label="LoLServer")], loc="upper right",
            facecolor="#12160f", edgecolor="#555555", labelcolor="#dddddd")
        leg.set_zorder(5)
    else:
        scats = [(axes[0].scatter([], [], s=[], c=[]), "sim"),
                 (axes[1].scatter([], [], s=[], c=[]), "srv")]
    suptitle = fig.suptitle("", color="#dddddd", fontsize=10)
    fig.patch.set_facecolor("#1b1f18")
    for ax in axes:
        ax.tick_params(colors="#999999")
        ax.title.set_color("#dddddd")
    fig.tight_layout(rect=(0, 0, 1, 0.93 if overlay else 0.96))

    n = min(len(streams.sim), len(streams.srv))
    writer = FFMpegWriter(fps=fps, bitrate=2400)
    with writer.saving(fig, str(out_mp4), dpi=100):
        for i in range(n):
            snaps = {"sim": streams.sim[i], "srv": streams.srv[i]}
            for scat, which in scats:
                X, Y, C, S = _scatter_xyc_s(snaps[which])
                scat.set_offsets(list(zip(X, Y)) if X else np.empty((0, 2)))
                if overlay and which == "srv":
                    scat.set_edgecolor(C)
                    scat.set_sizes([s * 2.6 for s in S])
                else:
                    scat.set_color(C); scat.set_sizes(S)
            if frame == "follow":
                champ = snaps["sim"].champion(team)
                if champ is not None:
                    h = window / 2.0
                    for ax in axes:
                        ax.set_xlim(champ.x - h, champ.x + h)
                        ax.set_ylim(champ.y - h, champ.y + h)
            t_sim = streams.sim[i].t_ms / 1000.0
            t_srv = streams.srv[i].t_ms / 1000.0
            head = ("JAX sim (filled) vs LoLServer (rings)\n" if overlay
                    else "")
            suptitle.set_text(
                f"{head}decision {i}   sim t={t_sim:6.2f}s   "
                f"server t={t_srv:6.2f}s   "
                f"units sim={len(streams.sim[i].entities)} "
                f"server={len(streams.srv[i].entities)}")
            writer.grab_frame()
    print(f"wrote {out_mp4}  ({n} frames)")


def write_summary_plot(streams: FrameStreams, out_png: Path, team: int = BLUE
                       ) -> None:
    """Cheap non-animated companion: population + champion-HP curves,
    sim vs server overlaid -- the first thing to look at before spending time
    on the video.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def series(snaps):
        t = [s.t_ms / 1000.0 for s in snaps]
        pop = [sum(1 for e in s.entities if e.kind == "LaneMinion" and not e.dead)
               for s in snaps]
        ch = [s.champion(team) for s in snaps]
        hp = [(c.hp if c and c.hp is not None else float("nan")) for c in ch]
        return t, pop, hp

    t_sim, pop_sim, hp_sim = series(streams.sim)
    t_srv, pop_srv, hp_srv = series(streams.srv)

    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax[0].plot(t_sim, pop_sim, label="sim", color="#4488ff")
    ax[0].plot(t_srv, pop_srv, label="server", color="#ff4444")
    ax[0].set_ylabel("live minions on map"); ax[0].legend()
    ax[0].set_title(
        f"Tier-1.5 identical start (t={streams.start.t_ms} ms, "
        f"predicate={streams.start.predicate}) -- sim vs server")
    ax[1].plot(t_sim, hp_sim, label="sim", color="#4488ff")
    ax[1].plot(t_srv, hp_srv, label="server", color="#ff4444")
    ax[1].set_ylabel(f"champion HP (team {team})")
    ax[1].set_xlabel("game time (s), each side's own clock")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    print(f"wrote {out_png}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--fixture-dir", required=True, type=Path,
                   help="directory holding a recorded tier15 fixture, e.g. "
                        "lanerl_jax/runs/tier15_noshop")
    p.add_argument("--tag", default="drive",
                   help="fixture file prefix, e.g. 'drive' for "
                        "drive_actions.json / drive_obs.jsonl / drive/instance000.log")
    p.add_argument("--predicate", default="champ_near_minion",
                   choices=["champ_near_minion", "engaged", "index", "t_ms"])
    p.add_argument("--radius", type=float, default=90.0)
    p.add_argument("--min-attackers", type=int, default=2)
    p.add_argument("--min-hp-frac", type=float, default=0.5)
    p.add_argument("--start-index", type=int, default=None)
    p.add_argument("--start-t-ms", type=int, default=None)
    p.add_argument("--decisions", type=int, default=300)
    p.add_argument("--team", choices=["blue", "red"], default="blue")
    p.add_argument("--route-table", action="store_true",
                   help="use the routed local pathing table instead of "
                        "straight-line moves (slower to set up, more faithful)")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--frame", choices=["sim", "both", "follow"], default="sim",
                   help="axes: 'sim' bounds on the sim's units (the server's "
                        "jungle/off-lane units are out of scope and would "
                        "otherwise set the scale), 'both' the union, 'follow' "
                        "a --window box tracking the champion")
    p.add_argument("--window", type=float, default=1600.0,
                   help="side length of the --frame follow box, in units")
    p.add_argument("--overlay", action="store_true",
                   help="one panel instead of two: sim filled, server as open "
                        "rings, so a deviation is a dot outside its ring")
    p.add_argument("--no-video", action="store_true",
                   help="write the summary PNG only, skip the mp4")
    args = p.parse_args(argv)

    log = args.fixture_dir / args.tag / "instance000.log"
    if not log.exists():
        # tier15's own `record` layout also writes the log path straight
        # into `<tag>_actions.json`'s sibling; fall back to a flat layout.
        alt = args.fixture_dir / f"{args.tag}.log"
        log = alt if alt.exists() else log
    fixture = Fixture(log=log,
                      actions=args.fixture_dir / f"{args.tag}_actions.json",
                      observations=args.fixture_dir / f"{args.tag}_obs.jsonl")

    team = BLUE if args.team == "blue" else RED
    sel_kwargs: Dict[str, object] = {"predicate": args.predicate}
    if args.predicate == "champ_near_minion":
        sel_kwargs["radius"] = args.radius
        sel_kwargs["min_hp_frac"] = args.min_hp_frac
    elif args.predicate == "engaged":
        sel_kwargs["min_attackers"] = args.min_attackers
        sel_kwargs["min_hp_frac"] = args.min_hp_frac
    elif args.predicate == "index":
        sel_kwargs["index"] = args.start_index
    elif args.predicate == "t_ms":
        sel_kwargs["t_ms"] = args.start_t_ms

    streams = build_frame_streams(
        fixture, sel_kwargs, decisions=args.decisions, team=team,
        table_disabled=not args.route_table)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(streams.start.describe())
    print(f"frames captured: sim={len(streams.sim)} server={len(streams.srv)}"
         f"  truncated: {streams.truncated_reason or 'no (ran the full window)'}")
    (args.out_dir / "meta.json").write_text(json.dumps({
        "start_t_ms": streams.start.t_ms,
        "predicate": streams.start.predicate,
        "detail": streams.start.detail,
        "n_frames_sim": len(streams.sim),
        "n_frames_srv": len(streams.srv),
        "truncated_reason": streams.truncated_reason,
    }, indent=2, default=str))

    write_summary_plot(streams, args.out_dir / "summary.png", team=team)
    if not args.no_video:
        name = "overlay.mp4" if args.overlay else "side_by_side.mp4"
        render_side_by_side(streams, args.out_dir / name, fps=args.fps,
                            frame=args.frame, window=args.window, team=team,
                            overlay=args.overlay)


if __name__ == "__main__":
    main()
