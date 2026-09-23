"""PARITY-001: the policy-driven divergence gate.

WHY THIS EXISTS
---------------
`SPELL-001`: a trained policy re-cast Garen E mid-spin on 80% of its decisions
and held a permanent damage aura in the sim; the same checkpoint scored 0 CS in
the C# server. None of the 47 parity tools had ever run a trained policy's
action stream -- gate 1 is scored on an idle corpus with cooldowns and buffs
excluded, tier 1 re-injects buff phase every tick (`METH-005`), tier 1.5
compares position only -- so the one stream that contained the exploit was the
one nobody replayed. This gate replays exactly that stream.

WHAT IT DOES
------------
(a) **Record.** Play the checkpoint in the server for T seconds through
    `policy_driver.PolicyPairDriver` (both champions under the policy --
    mirror self-play, the training distribution -- or red idle), via
    `record.record_trace`, with the per-tick state dump and its INTERNAL
    stream on and `LANERL_AUTOBUY=0` (`ENT-05`). The driver writes a
    `PolicyActionLog`: each decision's wire order (attack targets as NetIds),
    the sim order it came from, and the creation-rank table.
(b) **Replay.** Free-run the sim from `init_lane()` -- no injection, both
    engines start from the canonical reset -- applying each logged decision at
    the tick the server applied it (the same boundary, `action_replay`'s
    docstring), decoded by `action_replay.decision_to_orders` with a
    NetId -> slot table built by CREATION RANK (`CreationRankMap`: the k-th
    LaneMinion NetId is the minion with ``spawn_seq == first_minion_seq + k``;
    red before blue within a wave, `RESET-004`). `level` orders are NOOPs in
    the sim, which ranks spells itself.
(c) **Diff every tick** with `diff.diff_snapshots` on a rendering of the sim
    state that INCLUDES spell cooldowns and the modelled buff lanes (see
    :func:`render_sim_snapshot` -- `sim_vs_server.state_to_snapshot` renders
    neither, which is half of why `SPELL-001` was invisible).
(d) **Report** the first divergent tick and field, the first tick of every
    field that ever diverges, and event counters from BOTH engines computed by
    the same code on the same snapshot format: casts accepted, E spins started
    and ended, E cancels, the eval's own cd2-rising-edge "spin" counter, AA
    hits (`HasAutoAttacked` rising edges), CS, deaths, gold, level.
(e) **Score** against a `server_vs_server`-style shuffle floor when one is
    given (``--floor``, made by ``--make-floor``): the same wire log replayed
    open-loop into the server twice, plain and under `LANERL_SHUFFLE_ORDER`,
    scored by this same code. Without a floor the verdict is ``UNSCORED`` --
    the numbers are printed, and unscored is not passing.

THE STEP CONFIGURATION IS THE TRAINING ONE
------------------------------------------
The sim is stepped exactly as `train/trainer.py::_env_step` steps it: both
build their configuration from `sim.config.SimConfig.training` and step it
through `sim.step.env_apply` + `env_advance`, with the same Map1 local-route
artifact `run_train.py` loads by default. No other parity tool ran this
configuration, which is how the fountain turrets walked across the map in
every RL run without a gate noticing (`COLL-004`). The gate uses
`SimConfig.gate`, which `sim/tests/test_sim_config.py` asserts differs from
the training config only in an allow-list citing ledger rows (`STRUCT-003`).
The only difference is granularity: the gate steps one tick at a time
(``step_ticks=1`` twice per decision, orders applied before the first) so it
can diff every server tick; `step_decision` is a `lax.scan` of the same `tick`,
so two one-tick calls are the same computation as one two-tick call.

WHAT IS AND IS NOT SCORED
-------------------------
Scored: position, hp, max hp, dead, gold, level, CS, deaths, all four spell
cooldowns, the modelled buffs (`TRACKED_BUFFS`), and every entity present in
one engine and not the other. Reported but not scored (:data:`REPORTED_ONLY`):
spell ranks -- the sim ranks at level-up by table and the server only when
the driver's `level` order lands, so they differ at t=0 by construction.
Ignored (:data:`NOT_MODELLED`): fields the sim does not carry or renders only
from the static profile table. Every list is echoed in the report.

    python -m lanerl_jax.parity.policy_divergence \\
        --checkpoint lanerl_jax/runs/train/<run>/ckpt_latest.msgpack \\
        --seconds 300 --out runs/parity001/<run> [--floor floor.json]
    python -m lanerl_jax.parity.policy_divergence --existing runs/parity001/<run> \\
        [--floor runs/parity001/<run>/floor.json] [--out <elsewhere>]
    python -m lanerl_jax.parity.policy_divergence --make-floor runs/parity001/<run> \\
        --server-dir <build with LANERL_SHUFFLE_ORDER> --config <cfg>
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Tuple

import numpy as np

from .action_replay import RecordedDecision, decision_to_orders
from .diff import LANE_KINDS, Tolerance, diff_snapshots
from .trace import AIBlock, ChampionBlock, Entity, Snapshot, StatQ, parse_stream

__all__ = [
    "TRACKED_BUFFS", "NOT_MODELLED", "REPORTED_ONLY",
    "TrainingStepEngine", "render_sim_snapshot", "project_server_snapshot",
    "iter_server_ticks", "EventCounters", "DivergenceTracker",
    "replay_and_diff", "replay_resync", "compare_server_logs",
    "score_against_floor", "ROUTING_APPROX", "FREE_RUN_REPORTED_ONLY",
    "ReplayWireDriver", "main",
]

#: 30 Hz decisions off a 60 Hz server: `LANERL_STEP_TICKS=2`.
TICK_MS = 1000.0 / 60.0

#: the server's buff names (`AddBuff("...")` in the vendored Garen scripts)
#: the sim models, from `spells.BUFF_NAMES`. R's pending hit is a sim-only
#: mailbox, not a buff, and has no name.
def _buff_names():
    from ..sim.spells import BUFF_NAMES
    return tuple(BUFF_NAMES)


#: Buff names compared on both sides. The server also carries GarenPassive,
#: GarenPassiveHeal and GarenPassiveCooldown, which the sim does not model as
#: buffs; they are filtered from the server rows before the diff.
TRACKED_BUFFS = ("GarenE", "GarenQ", "GarenQHaste", "GarenW", "GarenWPassive")

#: Ignored by the diff (and echoed in the report).
NOT_MODELLED: Dict[str, str] = {
    "ad": "derived stat; the sim has no live champion stat block to render",
    "armor": "derived stat; see ad",
    "mr": "derived stat; see ad",
    "move_speed": "derived stat; see ad (its EFFECT is scored through position)",
    "attack_speed": "derived stat; see ad (its EFFECT is scored through AA hits)",
    "skill_points": "the sim has no skill-point pool",
    "can_move": "not rendered by the sim",
    "cast_spell": "not rendered by the sim",
    "channel_spell": "not rendered by the sim",
    "waypoints": "fixed-size array vs list: counts differ structurally "
                 "(sim_vs_server.NOT_MODELLED)",
    "move_order": "sim_vs_server.NOT_MODELLED",
}

#: Compared, first divergence reported per field, but NOT part of the scored
#: first divergence.
REPORTED_ONLY: Dict[str, str] = {
    f"spell.{s}.level": "the sim ranks spells by table at level-up; the server "
                        "only when the driver's `level` order lands (one "
                        "decision later, and never before the first decision)"
    for s in "QWER"
}

#: `PATH-001` is `APPROX`: the route table does not reproduce the server's A*
#: tie-break, so a free-running sim champion leaves the fountain on a slightly
#: different line and never re-converges -- every later tick "diverges" from
#: one early routing choice. While that holds, free-run champion position is
#: REPORTED (first tick, ticks diverged) but not scored; position is scored per
#: decision by :func:`replay_resync` instead. Never a scalar distance
#: tolerance on free-run positions: that would hide exactly the drift this
#: gate exists to see.
ROUTING_APPROX = True
FREE_RUN_REPORTED_ONLY: Dict[str, str] = {
    "Champion.pos.x": "free-run champion position while routing is APPROX "
                      "(PATH-001); scored per decision under resync instead",
    "Champion.pos.y": "see Champion.pos.x",
}

#: Per-decision position tolerance under resync: the dump's own quantum.
RESYNC_POS_TOL_U = 1.0 / 16.0

#: `GarenECancel`'s cooldown (`E.cs`): what the server's slot-2 cooldown shows
#: during a spin. A rise to at most this many seconds is a spin START.
_E_ARMED_MAX_S = 1.1


# ---------------------------------------------------------------------------
# the sim, in the training configuration
# ---------------------------------------------------------------------------

def _default_route_artifact() -> Path:
    from ..sim.config import DEFAULT_ROUTE_ARTIFACT
    return DEFAULT_ROUTE_ARTIFACT


class TrainingStepEngine:
    """`env_apply` + `env_advance` under `SimConfig.gate` -- the trainer's
    configuration, one tick per call (`STRUCT-003`).

    The route table (~240 MB) and terrain are passed to the jitted
    `apply_orders` as ARGUMENTS rather than closed over, so they are not baked
    into the compiled program as constants; their static fields (ints/floats)
    are closed over, which is what the trainer's closure does too.
    """

    def __init__(self, route_artifact: Optional[Path] = None,
                 use_route_table: bool = True, *, sim_config=None):
        import jax

        from ..sim.config import SimConfig
        from ..sim.init import init_lane
        from ..sim.step import env_advance, env_apply

        if sim_config is None:
            artifact = None
            if use_route_table:
                artifact = (Path(route_artifact) if route_artifact
                            else _default_route_artifact())
            sim_config = SimConfig.gate(route_artifact=artifact)
        cfg = self.sim_config = sim_config
        self.params = cfg.params
        self.path = cfg.lane_path
        self._init_lane = init_lane
        self.route_artifact = cfg.route_artifact
        rt, ter = cfg.route_table, cfg.terrain

        if rt is None:
            self._apply_j = jax.jit(lambda s, o: env_apply(s, o, cfg))
            self._apply = lambda s, o: self._apply_j(s, o)
        else:
            rt_arrays = (rt.cell_to_row, rt.next_hop, rt.run_length)
            ter_arrays = (ter.walkable, ter.walkable_prefix)

            def _ap(s, o, rta, tera):
                r = rt._replace(cell_to_row=rta[0], next_hop=rta[1],
                                run_length=rta[2])
                t = ter._replace(walkable=tera[0], walkable_prefix=tera[1])
                return env_apply(s, o, cfg.replace(route_table=r, terrain=t))

            self._apply_j = jax.jit(_ap)
            self._apply = lambda s, o: self._apply_j(s, o, rt_arrays, ter_arrays)

        self._tick = jax.jit(lambda s: env_advance(s, cfg))

    def init(self):
        return self._init_lane()

    def apply(self, state, orders):
        return self._apply(state, orders)

    def tick(self, state):
        return self._tick(state)

    def describe(self) -> dict:
        return {**self.sim_config.describe(),
                "fingerprint": self.sim_config.fingerprint(),
                "granularity": "1 tick per call, x2 per decision"}

    def resync(self, state, champ: Mapping[int, "ChampInternal"]):
        """Teleport each champion to the server's exact state: position,
        collision-quadtree position, and waypoint list + key (the list the
        server holds after this tick's orders; a Move applied next replaces
        it, as on the server)."""
        import jax.numpy as jnp
        x, y = state.x, state.y
        cx, cy = state.collision_x, state.collision_y
        wp, nw, key = state.waypoints, state.n_waypoints, state.waypoint_key
        cap = wp.shape[1]
        for team, c in champ.items():
            i = CHAMP_SLOT[team]
            x, y = x.at[i].set(c.x), y.at[i].set(c.y)
            cx, cy = cx.at[i].set(c.cx), cy.at[i].set(c.cy)
            if c.wps:
                w = np.zeros(wp.shape[1:], np.float32)
                pts = np.asarray(c.wps[:cap], np.float32)
                w[:len(pts)] = pts
                wp = wp.at[i].set(jnp.asarray(w, wp.dtype))
                nw = nw.at[i].set(len(pts))
                key = key.at[i].set(min(c.key, len(pts)))
        return state.replace(x=x, y=y, collision_x=cx, collision_y=cy,
                             waypoints=wp, n_waypoints=nw, waypoint_key=key)

    def champion_routes(self, state) -> Dict[int, dict]:
        """Per team: exact position, the remaining waypoints (from the key)
        and the last Move's ``route_status``."""
        import jax
        g = jax.device_get({"x": state.x, "y": state.y, "wp": state.waypoints,
                            "n": state.n_waypoints, "key": state.waypoint_key,
                            "rs": state.route_status})
        out = {}
        for team, i in CHAMP_SLOT.items():
            n, k = int(g["n"][i]), int(g["key"][i])
            out[team] = {"x": float(g["x"][i]), "y": float(g["y"][i]),
                         "remaining": [tuple(map(float, g["wp"][i, q]))
                                       for q in range(k, n)],
                         "route_status": int(g["rs"][i])}
        return out


#: Server team -> sim unit slot. Champions are unit slots 0 (blue) and 1
#: (red) in every `init_lane` state (`render_sim_snapshot` relies on it too).
CHAMP_SLOT = {100: 0, 200: 1}


_FETCH = ("kind", "alive", "team", "x", "y", "hp", "max_hp", "level", "gold",
          "cs", "deaths", "spell_level", "spell_cooldown",
          "has_auto_attacked", "spawn_seq", "t_ms")


def fetch(state) -> Dict[str, np.ndarray]:
    """The fields the renderer and the counters read, as numpy, in one copy.

    Buffs are fetched by server NAME (``buff:<name>``, ``(N,)`` bool) plus
    E's elapsed clock, which the renderer needs for the cancel projection."""
    import jax
    from ..sim.spells import active_by_name
    out = {k: getattr(state, k) for k in _FETCH}
    for name, on in active_by_name(state.buffs).items():
        out["buff:" + name] = on
    out["e_elapsed_s"] = state.buffs.e.elapsed_s
    return jax.device_get(out)


def render_sim_snapshot(f: Mapping[str, np.ndarray], t_ms: int) -> Snapshot:
    """A sim state (as :func:`fetch` returns it) in the server dump's format.

    Differs from `sim_vs_server.state_to_snapshot` in the three places that
    made `SPELL-001` invisible: spell cooldowns and ranks are rendered, the
    modelled buff lanes are rendered under the server's buff names, and a dead
    champion is rendered as a DEAD row (the server keeps the object) rather
    than dropped. It also reads numpy, not per-element jax, because it runs on
    every tick.

    One projection: during an E spin the server's slot 2 holds
    `GarenECancel`, whose cooldown is 1 s from the cast (`E.cs`), and the dump
    reports THAT; the sim keeps E's own cooldown at 0 for the spin and gates
    the cancel on the spin's elapsed clock ``>= E_CANCEL_MIN_S``. So E's
    cooldown is rendered as ``max(0, E_CANCEL_MIN_S - elapsed)`` while the
    spin is up -- the quantity the server shows -- and as the spell cooldown
    otherwise.
    """
    from ..sim.spells import E_CANCEL_MIN_S, Slot
    from ..sim.state import Kind, Team

    kind_name = {Kind.CHAMPION: "Champion", Kind.LANE_MINION: "LaneMinion",
                 Kind.TURRET: "LaneTurret"}
    server_team = {Team.BLUE: 100, Team.RED: 200, Team.NEUTRAL: 300}
    names = _buff_names()
    kind, alive = f["kind"], f["alive"]
    ents: List[Entity] = []
    for i in np.flatnonzero(kind != Kind.NONE):
        i = int(i)
        k = int(kind[i])
        is_champ = k == Kind.CHAMPION
        if not alive[i] and not is_champ:
            continue
        buffs = tuple(sorted(nm for nm in names if f["buff:" + nm][i]))
        ai = AIBlock(move_order=0, waypoints=0, cast_spell="-",
                     channel_spell="-", can_move=True, buffs=buffs)
        champ = None
        if is_champ:
            c = i                          # champions are unit slots 0/1
            spells = []
            for s in range(4):
                cd = float(f["spell_cooldown"][c, s])
                if s == Slot.E and f["buff:GarenE"][i]:
                    cd = max(0.0, E_CANCEL_MIN_S - float(f["e_elapsed_s"][i]))
                spells.append((int(f["spell_level"][c, s]),
                               int(round(max(cd, 0.0) * StatQ))))
            champ = ChampionBlock(
                q_ad=0, q_armor=0, q_mr=0, q_move_speed=0,
                q_attack_speed_mult=0, level=int(f["level"][i]),
                q_gold=int(round(float(f["gold"][i]) * StatQ)),
                minions_killed=int(f["cs"][i]), deaths=int(f["deaths"][i]),
                skill_points=0, spells=tuple(spells))
        ents.append(Entity(
            kind=kind_name.get(k, "Other"),
            q_x=int(round(float(f["x"][i]) * 16)),
            q_y=int(round(float(f["y"][i]) * 16)),
            team=server_team[int(f["team"][i])],
            q_hp=int(round(float(f["hp"][i]) * StatQ)),
            q_max_hp=int(round(float(f["max_hp"][i]) * StatQ)),
            dead=not bool(alive[i]), ai=ai, champ=champ))
    return Snapshot(t_ms=int(t_ms), entities=ents)


#: `LanerlConfig.StartingGold`. The server's dumped gold is the WALLET, which
#: starts here; the sim's is EARNINGS from 0 (`OBS-04`). With the shop off
#: (`LANERL_AUTOBUY=0`, which the gate sets -- `ENT-05`) the wallet never
#: drops, so earnings = wallet - 475 exactly, and a shop that DID run shows up
#: as a gold divergence at t=0 rather than being absorbed.
STARTING_GOLD = 475.0


def project_server_snapshot(snap: Snapshot,
                            gold_offset: float = STARTING_GOLD) -> Snapshot:
    """The server row in the sim's terms: tracked buffs only, gold as
    earnings, a ready cooldown as 0."""
    keep = set(TRACKED_BUFFS)
    q_off = int(round(gold_offset * StatQ))
    out = []
    for e in snap.entities:
        if e.ai is not None and e.ai.buffs:
            b = tuple(sorted(x for x in e.ai.buffs if x in keep))
            if b != e.ai.buffs:
                e = dataclasses.replace(e, ai=dataclasses.replace(e.ai, buffs=b))
        if e.champ is not None:
            # A ready spell's server cooldown sits one tick BELOW zero
            # (-17 quanta = -1/60 s: the countdown overshoots and nothing
            # resets it until the next cast); the sim clamps at 0. Both mean
            # "ready", so the server side is clamped too.
            spells = tuple((lv, max(cd, 0)) for lv, cd in e.champ.spells)
            e = dataclasses.replace(e, champ=dataclasses.replace(
                e.champ, q_gold=e.champ.q_gold - q_off, spells=spells))
        out.append(e)
    return dataclasses.replace(snap, entities=out)


# ---------------------------------------------------------------------------
# the server dump, streamed
# ---------------------------------------------------------------------------

_INT_CHAMP = re.compile(
    r"LANERL_INTERNAL t=-?\d+ ai id=(\d+) kind=Champion team=(\d+) .*?\bhasaa=(\d)")
_INT_MINION = re.compile(r"LANERL_INTERNAL t=-?\d+ ai id=(\d+) kind=LaneMinion ")
#: The exact float32 position (``xbits``/``ybits``), the waypoint list the
#: server holds AFTER this tick's orders (``wps``, in PosQ units) with its
#: ``CurrentWaypointKey``, and the collision quadtree position.
_INT_CHAMP_POS = re.compile(
    r"LANERL_INTERNAL t=-?\d+ ai id=\d+ kind=Champion team=(\d+) .*?"
    r"\bxbits=(-?\d+) ybits=(-?\d+) .*?\bwpkey=(\d+) wps=(\S+) .*?"
    r"\bcollbits=(-?\d+),(-?\d+)")


def _bits_f32(b: str) -> float:
    return float(np.array([int(b)], np.int64).astype(np.int32).view(np.float32)[0])


@dataclasses.dataclass(frozen=True)
class ChampInternal:
    """One champion's exact state from a `LANERL_INTERNAL` row."""
    x: float
    y: float
    cx: float
    cy: float
    key: int
    wps: Tuple[Tuple[float, float], ...]

    @property
    def remaining(self) -> Tuple[Tuple[float, float], ...]:
        return self.wps[self.key:]


def _parse_champ_internal(line: str) -> Optional[Tuple[int, ChampInternal]]:
    m = _INT_CHAMP_POS.search(line)
    if m is None:
        return None
    wps = () if m.group(5) == "-" else tuple(
        tuple(int(v) / 16.0 for v in p.split(",")) for p in m.group(5).split(";"))
    return int(m.group(1)), ChampInternal(
        _bits_f32(m.group(2)), _bits_f32(m.group(3)), _bits_f32(m.group(6)),
        _bits_f32(m.group(7)), int(m.group(4)), wps)


def iter_server_ticks(path, to_ms: Optional[int] = None
                      ) -> Iterator[Tuple[Snapshot, dict]]:
    """Yield ``(snapshot, extra)`` per server tick, one tick in memory at a time.

    ``extra`` carries what the canonical rows do not: ``hasaa`` per champion
    team (from the INTERNAL stream, None when the log has none), the
    LaneMinion NetIds present that tick, and ``champ``: a
    :class:`ChampInternal` per team when the INTERNAL rows carry exact bits
    (what :func:`replay_resync` teleports to). Each snapshot goes through
    `trace.parse_stream`, so the ``n=`` row-count check still applies.
    """
    buf: List[str] = []
    extra: dict = {"hasaa": {}, "minions": [], "champ": {}}

    def flush():
        snaps = parse_stream(buf).snapshots
        if len(snaps) != 1:
            raise ValueError(f"expected one snapshot per STATEHASH, got {len(snaps)}")
        return snaps[0]

    with Path(path).open(errors="replace") as fh:
        for line in fh:
            if "LANERL_STATEHASH" in line:
                if buf:
                    s = flush()
                    if to_ms is not None and s.t_ms > to_ms:
                        return
                    yield s, extra
                buf = [line]
                extra = {"hasaa": {}, "minions": [], "champ": {}}
            elif "LANERL_STATEROW" in line:
                if buf:
                    buf.append(line)
            elif "LANERL_INTERNAL" in line and buf:
                m = _INT_CHAMP.search(line)
                if m is not None:
                    extra["hasaa"][int(m.group(2))] = m.group(3) == "1"
                    c = _parse_champ_internal(line)
                    if c is not None:
                        extra["champ"][c[0]] = c[1]
                    continue
                m = _INT_MINION.search(line)
                if m is not None:
                    extra["minions"].append(int(m.group(1)))
    if buf:
        s = flush()
        if to_ms is None or s.t_ms <= to_ms:
            yield s, extra


# ---------------------------------------------------------------------------
# counters and divergence bookkeeping
# ---------------------------------------------------------------------------

_COUNTER_KEYS = ("casts_accepted", "e_spin_starts", "e_spin_ends", "e_cancels",
                 "e_cd_rising_edges", "q_casts", "w_casts", "r_casts",
                 "aa_hits", "cs", "deaths", "gold", "level")


class EventCounters:
    """Per-team event counts off a stream of snapshots, identical for both engines.

    * ``e_spin_starts`` -- `GarenE` buff rising edges: spins that STARTED.
    * ``e_spin_ends`` -- E cooldown rising edges ABOVE the 1 s cancel
      cooldown: spins that ended, by expiry or cancel, and started the rank
      cooldown. ``e_cancels`` -- spins whose buff lasted under 2.9 s.
    * ``e_cd_rising_edges`` -- EVERY rising edge of the E cooldown: the
      counter `rl_eval_vs_server` calls "spins". On the server that is TWO per
      spin (the 1 s `GarenECancel` cooldown at the start and the rank
      cooldown at the end), so it is reported beside the two real counts to
      make that visible, not as a spin count.
    * ``q_casts``/``w_casts`` -- `GarenQ`/`GarenW` buff rising edges;
      ``r_casts`` -- R cooldown rising edges. ``casts_accepted`` is
      ``e_spin_starts + q + w + r`` (an E cancel is a second accepted cast on
      the server and is counted separately in ``e_cancels``).
    * ``aa_hits`` -- `HasAutoAttacked` rising edges (set when the swing's
      windup completes -- for melee Garen, when the damage lands).
    * ``cs``/``deaths``/``gold``/``level`` -- last values seen.
    """

    RISE_Q = 2      # StatQ quanta (~2 ms) of slack on a "rising" cooldown

    def __init__(self):
        self.c = {t: {k: 0 for k in _COUNTER_KEYS} for t in (100, 200)}
        self._prev: Dict[int, dict] = {}
        self.ticks = 0

    def update(self, snap: Snapshot, hasaa: Optional[Mapping[int, bool]] = None):
        self.ticks += 1
        for team in (100, 200):
            e = snap.champion(team)
            if e is None:
                continue
            c = self.c[team]
            p = self._prev.setdefault(team, {"buffs": (), "cd": [0, 0, 0, 0],
                                             "aa": False, "e_on": None})
            buffs = e.ai.buffs if e.ai is not None else ()
            cds = [int(sp[1]) for sp in e.champ.spells]
            e_on = "GarenE" in buffs
            if e_on and "GarenE" not in p["buffs"]:
                c["e_spin_starts"] += 1
                p["e_on"] = snap.t_ms
            if not e_on and p["e_on"] is not None:
                if snap.t_ms - p["e_on"] < 2900:
                    c["e_cancels"] += 1
                p["e_on"] = None
            for name, key in (("GarenQ", "q_casts"), ("GarenW", "w_casts")):
                if name in buffs and name not in p["buffs"]:
                    c[key] += 1
            if cds[2] > p["cd"][2] + self.RISE_Q:
                c["e_cd_rising_edges"] += 1
                if cds[2] > _E_ARMED_MAX_S * StatQ:
                    c["e_spin_ends"] += 1
            if cds[3] > p["cd"][3] + self.RISE_Q:
                c["r_casts"] += 1
            aa = bool((hasaa or {}).get(team, False))
            if aa and not p["aa"]:
                c["aa_hits"] += 1
            c["casts_accepted"] = (c["e_spin_starts"] + c["q_casts"]
                                   + c["w_casts"] + c["r_casts"])
            c["cs"] = int(e.champ.minions_killed)
            c["deaths"] = int(e.champ.deaths)
            c["gold"] = round(e.champ.q_gold / StatQ, 1)
            c["level"] = int(e.champ.level)
            p.update(buffs=buffs, cd=cds, aa=aa)

    def as_dict(self) -> dict:
        return {("blue" if t == 100 else "red"): dict(v) for t, v in self.c.items()}


def _diff_names(d) -> List[Tuple[str, str]]:
    """``(field key, human line)`` per disagreement in one SnapshotDiff."""
    out = []
    for ed in d.entity_diffs:
        who = f"{ed.kind}(team={ed.team})"
        if ed.is_unmatched:
            side = "sim" if ed.only_in == "left" else "server"
            e = ed.entity
            where = "" if e is None else f" at ({e.x:.1f},{e.y:.1f})"
            out.append((f"{ed.kind}.only_in_{side}", f"{who}{where} only in {side}"))
        else:
            for f in ed.fields:
                out.append((f"{ed.kind}.{f.name}", f"{who} {f}"))
    return out


def _is_scored(key: str) -> bool:
    if ROUTING_APPROX and key in FREE_RUN_REPORTED_ONLY:
        return False
    return key.split(".", 1)[1] not in REPORTED_ONLY


def _champ_summary(snap: Snapshot) -> dict:
    out = {}
    for team, side in ((100, "blue"), (200, "red")):
        e = snap.champion(team)
        if e is None:
            continue
        out[side] = {"x": round(e.x, 2), "y": round(e.y, 2), "hp": e.hp,
                     "dead": e.dead, "buffs": list(e.ai.buffs if e.ai else ()),
                     "spells": [[lv, round(cd / StatQ, 3)] for lv, cd in e.champ.spells],
                     "cs": e.champ.minions_killed,
                     "gold": round(e.champ.q_gold / StatQ, 1)}
    return out


class DivergenceTracker:
    """First divergence overall and per field, and how many ticks each field diverged."""

    def __init__(self, tol: Tolerance):
        self.tol = tol
        # One champion per team: they are matched by TEAM, never by position.
        # The positional matcher's 8-unit radius is right for minions near a
        # divergence and wrong for the one entity per group whose identity is
        # never in doubt -- with it, two champions 9 units apart read as two
        # unrelated units "only in sim" / "only in server".
        self.champ_tol = dataclasses.replace(tol, match_radius_q=1 << 40)
        self.first: Optional[dict] = None
        self.first_by_field: Dict[str, dict] = {}
        self.ticks_by_field: Dict[str, int] = {}
        self.ticks = 0
        self.max_match_distance: float = 0.0

    def update(self, left: Snapshot, right: Snapshot, *, tick: int,
               decision: Optional[int], left_name: str = "sim",
               right_name: str = "server") -> None:
        self.ticks += 1

        def split(snap):
            ch = [e for e in snap.entities if e.kind == "Champion"]
            rest = [e for e in snap.entities if e.kind != "Champion"]
            return (Snapshot(t_ms=snap.t_ms, entities=ch),
                    Snapshot(t_ms=snap.t_ms, entities=rest))

        (lc, lr), (rc, rr) = split(left), split(right)
        dc = diff_snapshots(lc, rc, self.champ_tol)
        dr = diff_snapshots(lr, rr, self.tol)
        self.max_match_distance = max(self.max_match_distance,
                                      dr.max_match_distance_q / 16.0)
        if dc.clean and dr.clean:
            return
        pairs = _diff_names(dc) + _diff_names(dr)
        seen = set()
        for k, line in pairs:
            if k in seen:
                continue
            seen.add(k)
            self.first_by_field.setdefault(k, {"t_ms": left.t_ms, "example": line})
            self.ticks_by_field[k] = self.ticks_by_field.get(k, 0) + 1
        scored = [(k, s) for k, s in pairs if _is_scored(k)]
        if scored and self.first is None:
            self.first = {
                "t_ms": left.t_ms, "tick": tick, "decision": decision,
                "fields": sorted({k for k, _ in scored}),
                "diffs": [s for _, s in scored][:12],
                "n_diffs": len(scored),
                f"{left_name}_champions": _champ_summary(left),
                f"{right_name}_champions": _champ_summary(right),
            }

    def as_dict(self) -> dict:
        scored = {k: v for k, v in self.first_by_field.items() if _is_scored(k)}
        reported = {k: v for k, v in self.first_by_field.items() if not _is_scored(k)}
        by_t = lambda kv: kv[1]["t_ms"]  # noqa: E731
        return {"ticks_compared": self.ticks,
                "first_divergence": self.first,
                "first_divergence_by_field": dict(sorted(scored.items(), key=by_t)),
                "reported_only_first_by_field": dict(sorted(reported.items(),
                                                            key=by_t)),
                "max_minion_match_distance_units": round(self.max_match_distance, 3),
                "diverged_ticks_by_field": dict(sorted(self.ticks_by_field.items(),
                                                       key=lambda kv: -kv[1]))}


def gate_tolerance() -> Tolerance:
    return Tolerance(kinds=LANE_KINDS, ignore_fields=tuple(NOT_MODELLED))


# ---------------------------------------------------------------------------
# (b)+(c): replay the log through the sim and diff every tick
# ---------------------------------------------------------------------------

def _tick_of(t_ms: float) -> int:
    return int(round(float(t_ms) / TICK_MS))


def _for_sim(order: Mapping) -> Mapping:
    """A wire order the sim can decode: `level` has no sim counterpart."""
    return {"t": "noop"} if order.get("t") == "level" else order


def _decision_orders(log, k: int, ranks, f):
    """Decision ``k`` as sim `Orders`, NetIds mapped by creation rank.

    Returns ``(orders, unmapped_sides, rank_mismatches)``: an attack on a
    NetId with no live slot is sent as a noop and its side reported."""
    table = ranks.slot_table(f["spawn_seq"], f["alive"], f["kind"])
    sides, unmapped, rank_mismatch = [], [], 0
    for side, wire, rec in (("blue", log.blue[k], log.blue_sim[k]),
                            ("red", log.red[k], log.red_sim[k])):
        w = _for_sim(wire)
        if w.get("t") == "attack":
            nid = int(w.get("id", 0))
            if rec and rec.get("target_spawn_seq") is not None \
                    and ranks.spawn_seq_of(nid) != rec["target_spawn_seq"]:
                rank_mismatch += 1
            if nid not in table:
                unmapped.append(side)
                w = {"t": "noop"}
        sides.append(w)
    orders = decision_to_orders(
        RecordedDecision(log.t_ms[k], sides[0], sides[1]), table)
    return orders, unmapped, rank_mismatch


def replay_and_diff(log, server_log: Path, *, engine=None,
                    to_ms: Optional[int] = None,
                    on_tick=None) -> dict:
    """Free-run the sim on ``log`` (a `PolicyActionLog`) and diff every tick.

    ``engine`` defaults to :class:`TrainingStepEngine`. Alignment follows the
    server's own snapshot sequence: snapshot 0 is the reset (``t=0``) and is
    compared against `init_lane()` unstepped; every later snapshot is ONE
    sim tick on, and a gap in the dump (a step other than ~16.7 ms) raises.
    Never by float clocks -- the server's `GameTime` is itself a float
    accumulator and drifts off the ``j * 16.667`` grid by more than a ms over
    a minute. Decision ``k`` is applied right after the tick whose snapshot
    time is within 1 ms of its recorded time (which is truncated; the dump
    rounds), before that tick's snapshot -- where the server applies it
    (`action_replay`'s docstring).
    """
    from ..sim.state import Kind

    engine = engine if engine is not None else TrainingStepEngine()
    ranks = log.rank_map()
    tracker = DivergenceTracker(gate_tolerance())
    sim_c, srv_c = EventCounters(), EventCounters()
    unmapped = {"blue": 0, "red": 0}
    rank_mismatch = 0
    missed = 0
    internals = False
    state = None
    k, n_dec, last_dec = 0, len(log.t_ms), None
    prev_t = None
    last_sim = last_srv = None
    clock_skew = 0.0
    t0 = time.time()
    for j, (srv, extra) in enumerate(iter_server_ticks(server_log, to_ms=to_ms)):
        ranks.add_minions(extra["minions"])
        internals = internals or bool(extra["hasaa"])
        if j == 0:
            if srv.t_ms != 0:
                raise ValueError(f"the dump starts at t={srv.t_ms}, not at the "
                                 "reset; the sim cannot be aligned to it")
            state = engine.init()
        else:
            dt = srv.t_ms - prev_t
            if not (TICK_MS - 2.0 <= dt <= TICK_MS + 2.0):
                raise ValueError(f"dump steps {prev_t} -> {srv.t_ms} ms: not one "
                                 "tick, so the lockstep replay would slip")
            state = engine.tick(state)
            while k < n_dec and log.t_ms[k] < srv.t_ms - 1:
                missed += 1                # a decision between two dumps
                k += 1
            if k < n_dec and abs(log.t_ms[k] - srv.t_ms) <= 1:
                f = fetch(state)
                orders, um, rm = _decision_orders(log, k, ranks, f)
                for side in um:
                    unmapped[side] += 1
                rank_mismatch += rm
                state = engine.apply(state, orders)
                last_dec = k
                k += 1
        prev_t = srv.t_ms
        f = fetch(state)
        clock_skew = max(clock_skew, abs(float(f["t_ms"]) - srv.t_ms))
        sim_snap = render_sim_snapshot(f, srv.t_ms)
        srv_snap = project_server_snapshot(srv)
        tracker.update(sim_snap, srv_snap, tick=j, decision=last_dec)
        champ = np.flatnonzero(f["kind"] == Kind.CHAMPION)
        sim_c.update(sim_snap, {100 if int(f["team"][i]) == 0 else 200:
                                bool(f["has_auto_attacked"][i]) for i in champ})
        srv_c.update(srv_snap, extra["hasaa"] if extra["hasaa"] else None)
        last_sim, last_srv = sim_snap, srv_snap
        if on_tick is not None:
            on_tick(j, srv.t_ms)
    out = tracker.as_dict()
    out.update({
        "t_ms_last": None if last_srv is None else last_srv.t_ms,
        "decisions_applied": 0 if last_dec is None else last_dec + 1,
        "decisions_missed": missed,
        "counters": {"sim": sim_c.as_dict(), "server": srv_c.as_dict()},
        "attack_orders_unmapped_in_sim": unmapped,
        "attack_rank_disagreements": rank_mismatch,
        "sim_clock_max_skew_ms": round(clock_skew, 3),
        "server_has_internals": internals,
        "step_config": engine.describe() if hasattr(engine, "describe") else None,
        "wall_s": round(time.time() - t0, 1),
        "final": {"sim": None if last_sim is None else _champ_summary(last_sim),
                  "server": None if last_srv is None else _champ_summary(last_srv)},
    })
    return out


def _route_equal(a, b, tol: float = RESYNC_POS_TOL_U) -> bool:
    """Two waypoint lists agree point for point to the dump's quantum (the
    server's are 1/16-quantised, so an identical route differs by <= 1/32)."""
    return len(a) == len(b) and all(
        abs(p[0] - q[0]) <= tol and abs(p[1] - q[1]) <= tol for p, q in zip(a, b))


def _host_route(grid, src, order: Mapping, srv: "ChampInternal",
                radius: float):
    """What the host port of `GetPath` says the server walks from ``src``
    for this decision: ``(remaining waypoints, get_path_was_null)``, or
    ``(None, False)`` when the server did not path this tick (its key is not
    a fresh 1).

    A Move goes to the float32 click, through `LanerlControl`'s null
    fallback (``[Position, click]``). Any other order is judged against the
    server's own final waypoint (e.g. an attack chase), dequantised."""
    if srv.key != 1 or len(srv.wps) < 2:
        return None, False
    if order.get("t") == "move":
        goal = (float(np.float32(order["x"])), float(np.float32(order["y"])))
    else:
        goal = srv.wps[-1]
    path = grid.get_path(tuple(src), tuple(goal), radius)
    null = path is None or len(path) < 2
    if null:
        path = [tuple(src), goal]
    return [tuple(p) for p in path[1:]], null


def replay_resync(log, server_log: Path, *, engine=None,
                  to_ms: Optional[int] = None, grid=None,
                  radius: Optional[float] = None, max_examples: int = 25,
                  on_tick=None) -> dict:
    """Champion position scored PER DECISION under teleport + waypoint resync.

    At every decision tick, before the orders are applied, each sim champion
    is reset to the server's exact state from the INTERNAL row (float32
    position bits, collision position, waypoint list and key); the orders are
    applied and the sim runs to the next decision, where its position is
    compared with the server's exact one at ``RESYNC_POS_TOL_U`` per axis.
    So every interval starts from agreement and one routing choice cannot
    pollute every later tick, which is what free-run position does while
    `PATH-001` is `APPROX`.

    An interval outside tolerance is classified:

    * ``path001`` -- the sim's waypoints after the order differ from the
      server's, and the host port of `GetPath` (``data/navgrid.py``, which
      reproduces the server's waypoints) DOES reproduce the server's: the
      route table chose a different A* tie-break. Counted, and reported as
      the PATH-001 rate.
    * ``unexplained`` -- anything else: identical waypoints and still off; a
      route the host port does not reproduce either; or a server ``GetPath``
      null (the host port returns None and the server walks ``[Position,
      click]``) that the sim routed around -- that is `PATH-008`, which the
      sim reproduces, not a tie-break. The gate FAILS on any of these;
      ``unexplained_by_reason`` says which.
    """
    from ..data.navgrid import GAREN_PATHFINDING_RADIUS, NavGrid

    engine = engine if engine is not None else TrainingStepEngine()
    grid = grid if grid is not None else NavGrid.load()
    radius = GAREN_PATHFINDING_RADIUS if radius is None else radius
    ranks = log.rank_map()
    k, n_dec = 0, len(log.t_ms)
    state, prev_t = None, None
    pending: Dict[int, dict] = {}
    counts = {"intervals": 0, "within_tol": 0, "path001": 0, "unexplained": 0}
    by_order: Dict[str, Dict[str, int]] = {}
    status_counts: Dict[int, int] = {}
    examples: List[dict] = []
    reasons: Dict[str, int] = {}
    max_err = 0.0
    t0 = time.time()

    def close(team, srv_now: "ChampInternal", sim_now: dict):
        nonlocal max_err
        p = pending[team]
        ex, ey = sim_now["x"] - srv_now.x, sim_now["y"] - srv_now.y
        err = max(abs(ex), abs(ey))
        max_err = max(max_err, err)
        kind = p["order"].get("t", "?")
        bo = by_order.setdefault(kind, {"intervals": 0, "within_tol": 0,
                                        "path001": 0, "unexplained": 0})
        counts["intervals"] += 1
        bo["intervals"] += 1
        if err <= RESYNC_POS_TOL_U:
            counts["within_tol"] += 1
            bo["within_tol"] += 1
            return
        srv = p["srv"]
        same = _route_equal(p["sim_remaining"], srv.remaining)
        host, host_null = ((None, False) if same else
                           _host_route(grid, (srv.x, srv.y), p["order"], srv,
                                       radius))
        reproduced = host is not None and _route_equal(host, srv.remaining)
        # A server null the sim routed around is `PATH-008`, not a tie-break:
        # the host port explains it, but the sim is supposed to reproduce it.
        cls = "path001" if reproduced and not host_null else "unexplained"
        counts[cls] += 1
        bo[cls] += 1
        if cls == "unexplained":
            why = ("server GetPath null the sim did not reproduce (PATH-008)"
                   if reproduced else
                   "same waypoints, still off" if same else
                   "route the host port does not reproduce either")
            reasons[why] = reasons.get(why, 0) + 1
        if cls == "unexplained" and len(examples) < max_examples:
            examples.append({
                "decision": p["k"], "t_ms": p["t_ms"], "team": team,
                "order": dict(p["order"]), "err_u": round(err, 4),
                "route_status": p["route_status"], "why": why,
                "waypoints_agree": same,
                "sim": [[round(v, 3) for v in q] for q in p["sim_remaining"]][:8],
                "server": [[round(v, 3) for v in q] for q in srv.remaining][:8],
                "host": None if host is None
                else [[round(v, 3) for v in q] for q in host][:8]})

    for j, (srv, extra) in enumerate(iter_server_ticks(server_log, to_ms=to_ms)):
        ranks.add_minions(extra["minions"])
        if j == 0:
            if srv.t_ms != 0:
                raise ValueError(f"the dump starts at t={srv.t_ms}, not at the reset")
            state = engine.init()
        else:
            dt = srv.t_ms - prev_t
            if not (TICK_MS - 2.0 <= dt <= TICK_MS + 2.0):
                raise ValueError(f"dump steps {prev_t} -> {srv.t_ms} ms: not one tick")
            state = engine.tick(state)
            while k < n_dec and log.t_ms[k] < srv.t_ms - 1:
                k += 1
            if k < n_dec and abs(log.t_ms[k] - srv.t_ms) <= 1:
                champ = extra.get("champ") or {}
                if set(champ) != set(CHAMP_SLOT):
                    raise ValueError(
                        f"t={srv.t_ms}: resync needs both champions' INTERNAL rows "
                        "with xbits/wps/collbits; this dump has "
                        f"{sorted(champ)}")
                if pending:
                    now = engine.champion_routes(state)
                    for team in CHAMP_SLOT:
                        close(team, champ[team], now[team])
                f = fetch(state)
                orders, _, _ = _decision_orders(log, k, ranks, f)
                state = engine.apply(engine.resync(state, champ), orders)
                after = engine.champion_routes(state)
                pending = {}
                for team, side in ((100, "blue"), (200, "red")):
                    order = getattr(log, side)[k]
                    if order.get("t") == "move":
                        rs = after[team]["route_status"]
                        status_counts[rs] = status_counts.get(rs, 0) + 1
                    pending[team] = {"k": k, "t_ms": srv.t_ms, "order": order,
                                     "srv": champ[team],
                                     "sim_remaining": after[team]["remaining"],
                                     "route_status": after[team]["route_status"]}
                k += 1
        prev_t = srv.t_ms
        if on_tick is not None:
            on_tick(j, srv.t_ms)
    n = max(counts["intervals"], 1)
    from ..sim.local_pathing import LocalRouteStatus
    names = {v: k_ for k_, v in vars(LocalRouteStatus).items()
             if not k_.startswith("_") and isinstance(v, int)}
    return {
        "tolerance_u": RESYNC_POS_TOL_U,
        **counts,
        "path001_rate": round(counts["path001"] / n, 4),
        "unexplained_rate": round(counts["unexplained"] / n, 4),
        "max_err_u": round(max_err, 4),
        "by_order": by_order,
        "move_route_status": {names.get(s, str(s)): c
                              for s, c in sorted(status_counts.items())},
        "unexplained_by_reason": reasons,
        "unexplained_examples": examples,
        "wall_s": round(time.time() - t0, 1),
    }


def compare_server_logs(a: Path, b: Path, to_ms: Optional[int] = None) -> dict:
    """Two server recordings, scored by the same tracker and counters.

    The floor: ``a`` a plain run, ``b`` the same orders under
    `LANERL_SHUFFLE_ORDER`. Aligned on tick index; a tick missing from either
    side is skipped and counted.
    """
    tracker = DivergenceTracker(gate_tolerance())
    ca, cb = EventCounters(), EventCounters()
    ib = iter_server_ticks(b, to_ms=to_ms)
    pending = next(ib, None)
    missing = 0
    for sa, ea in iter_server_ticks(a, to_ms=to_ms):
        ja = _tick_of(sa.t_ms)
        while pending is not None and _tick_of(pending[0].t_ms) < ja:
            pending = next(ib, None)
            missing += 1
        if pending is None or _tick_of(pending[0].t_ms) != ja:
            missing += 1
            continue
        sb, eb = pending
        pending = next(ib, None)
        pa, pb = project_server_snapshot(sa), project_server_snapshot(sb)
        tracker.update(pa, pb, tick=ja, decision=None, left_name="a",
                       right_name="b")
        ca.update(pa, ea["hasaa"] or None)
        cb.update(pb, eb["hasaa"] or None)
    out = tracker.as_dict()
    out.update({"counters": {"a": ca.as_dict(), "b": cb.as_dict()},
                "ticks_missing": missing})
    return out


#: Counters the floor gates on, per team.
GATED_COUNTERS = ("casts_accepted", "e_spin_starts", "e_spin_ends", "aa_hits",
                  "cs", "deaths")


def score_against_floor(report: dict, floor: Optional[dict]) -> dict:
    """PASS / FAIL / UNSCORED.

    PASS needs all three: the sim agrees with the server at least as long as
    the server agrees with ITSELF under a reordered update loop (scored first
    divergence no earlier than the floor's); every gated counter's
    sim-vs-server gap is no larger than the server-vs-shuffled gap; and, while
    routing is APPROX, every per-decision champion position interval under
    resync (:func:`replay_resync`) is within tolerance or explained by the
    PATH-001 tie-break -- whose rate is returned, not gated. One floor
    run is one sample of the reordering noise; the report says which clause
    failed so a thin floor can be judged, not trusted.
    """
    if not floor:
        return {"verdict": "UNSCORED",
                "why": "no floor file: the counters and first divergence are "
                       "reported, but nothing says how much of either is the "
                       "server's own order-dependence (FLOOR-001). Unscored is "
                       "not passing."}
    fails = []
    # Champion position, per decision under resync (free-run position is
    # reported only while routing is APPROX). The floor for it is zero: the
    # champions are order-insensitive (`FLOOR-001`), so any interval the
    # PATH-001 tie-break does not explain is a sim defect.
    rs = report.get("position_resync")
    if ROUTING_APPROX and rs is None:
        fails.append("champion position not scored: no per-decision resync "
                     "pass in this report")
    elif rs is not None and rs.get("unexplained", 0) > 0:
        fails.append(f"champion position: {rs['unexplained']} of "
                     f"{rs['intervals']} decision intervals off by more than "
                     f"{RESYNC_POS_TOL_U} u that the PATH-001 tie-break does "
                     "not explain")
    fd = (report.get("first_divergence") or {}).get("t_ms")
    ffd = (floor.get("first_divergence") or {}).get("t_ms")
    if fd is not None and (ffd is None or fd < ffd):
        fails.append(f"first divergence at {fd} ms, before the floor's "
                     f"{ffd if ffd is not None else 'never'}")
    fc = floor.get("counters", {})
    rc = report.get("counters", {})
    for side in ("blue", "red"):
        for k in GATED_COUNTERS:
            try:
                gap = abs(rc["sim"][side][k] - rc["server"][side][k])
                allow = abs(fc["b"][side][k] - fc["a"][side][k])
            except KeyError:
                continue
            if gap > allow:
                fails.append(f"{side}.{k}: sim-vs-server gap {gap} > floor {allow}")
    return {"verdict": "FAIL" if fails else "PASS", "failures": fails,
            "floor_first_divergence_t_ms": ffd,
            "path001_rate": None if rs is None else rs.get("path001_rate")}


# ---------------------------------------------------------------------------
# (a) recording, and the open-loop replay the floor needs
# ---------------------------------------------------------------------------

class ReplayWireDriver:
    """``driver(obs, i)`` that re-sends a recorded wire log OPEN-LOOP.

    Attack targets are NetIds of the ORIGINAL run; a second server process
    hands out NetIds by the same counter only as long as every creation
    happened in the same order, which a shuffled update loop does not
    promise. So each target goes NetId -> `spawn_seq` (recorded ranks) ->
    NetId (this server's ranks, learned live), and one that does not resolve
    is sent as a noop and counted.
    """

    def __init__(self, log):
        from .policy_driver import CreationRankMap
        self.log = log
        self.recorded = log.rank_map()
        self.live = CreationRankMap()
        self.unresolved = 0

    def _map(self, order: Mapping) -> dict:
        if order.get("t") != "attack":
            return dict(order)
        seq = self.recorded.spawn_seq_of(int(order.get("id", 0)))
        nid = self.live.netid_of_spawn_seq(seq) if seq is not None else None
        if not nid:
            self.unresolved += 1
            return {"t": "noop"}
        return {"t": "attack", "id": int(nid)}

    def __call__(self, obs, i):
        if obs is None:
            return None
        self.live.observe(obs)
        if i >= len(self.log):
            return {"blue": {"t": "noop"}, "red": {"t": "noop"}}
        return {"blue": self._map(self.log.blue[i]), "red": self._map(self.log.red[i])}


def _record(out: Path, *, tag: str, decisions: int, driver, server_dir,
            config_path, port_base: int, extra_env: Optional[dict] = None) -> Path:
    from .record import record_trace
    env = {"LANERL_AUTOBUY": "0", **(extra_env or {})}
    return Path(record_trace(out, decisions=decisions, port_base=port_base,
                             tag=tag, extra_env=env, server_dir=server_dir,
                             config_path=config_path, driver=driver))


def record_policy_run(checkpoint: str, out: Path, *, seconds: float,
                      red: str = "policy", deterministic: bool = False,
                      seed: int = 0, server_dir=None, config_path=None,
                      port_base: int = 41000):
    """(a): one server, the checkpoint driving it, dump + `PolicyActionLog`."""
    from lanerl_rl import constants as C

    from .policy_driver import PolicyPairDriver, load_params

    policy, params, label = load_params(checkpoint)
    pair = PolicyPairDriver(policy, params, red=red, deterministic=deterministic,
                            seed=seed, meta={"checkpoint": str(checkpoint),
                                             "label": label,
                                             "server_dir": str(server_dir or ""),
                                             "config": str(config_path or "")})
    decisions = int(round(seconds * C.DECISION_HZ))
    log_path = _record(out, tag="policy", decisions=decisions, driver=pair,
                       server_dir=server_dir, config_path=config_path,
                       port_base=port_base)
    pair.log.meta["server_log"] = str(log_path)
    pair.log.meta["driver_counts"] = pair.counts
    pair.log.save(out / "policy_policy_actions.json")
    return log_path, pair.log


def _wire_counts(orders: List[dict]) -> dict:
    out: Dict[str, int] = {}
    for o in orders:
        out[o.get("t", "?")] = out.get(o.get("t", "?"), 0) + 1
    return out


def _headline(rep: dict) -> str:
    fd = rep.get("first_divergence")
    lines = [f"PARITY-001 {rep['verdict']['verdict']}: "
             f"{rep['ticks_compared']} ticks compared to t={rep['t_ms_last']} ms"]
    if fd is None:
        lines.append("  no scored divergence")
    else:
        lines.append(f"  first scored divergence: t={fd['t_ms']} ms (tick "
                     f"{fd['tick']}, decision {fd['decision']}): "
                     + ", ".join(fd["fields"]))
        for s in fd["diffs"][:6]:
            lines.append(f"    {s}")
    fb = list(rep["first_divergence_by_field"].items())[:10]
    if fb:
        lines.append("  first tick per field: "
                     + ", ".join(f"{k}@{v['t_ms']}" for k, v in fb))
    for side in ("blue", "red"):
        s, v = rep["counters"]["sim"][side], rep["counters"]["server"][side]
        lines.append(f"  {side:4s} " + "  ".join(
            f"{k}={s[k]}/{v[k]}" for k in ("casts_accepted", "e_spin_starts",
                                            "e_spin_ends", "e_cancels",
                                            "e_cd_rising_edges", "aa_hits", "cs",
                                            "deaths", "gold", "level"))
            + "   (sim/server)")
    rs = rep.get("position_resync")
    if rs:
        lines.append(
            f"  champion position per decision (resync, tol {rs['tolerance_u']} u): "
            f"{rs['within_tol']}/{rs['intervals']} within, {rs['path001']} PATH-001 "
            f"tie-break ({rs['path001_rate']:.2%}), {rs['unexplained']} unexplained; "
            f"move route status {rs['move_route_status']}")
    if rep["verdict"].get("failures"):
        lines += [f"  FAIL {f}" for f in rep["verdict"]["failures"]]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="",
                    help="a RunDir ckpt_*.msgpack, or 'random'")
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--floor", type=Path, default=None,
                    help="floor JSON from --make-floor")
    ap.add_argument("--existing", type=Path, default=None,
                    help="re-score a recording already in this directory "
                         "(no server)")
    ap.add_argument("--make-floor", type=Path, default=None,
                    help="replay this recording's wire log open-loop into the "
                         "server twice, plain and under LANERL_SHUFFLE_ORDER, "
                         "and write floor.json. Needs a build with the shuffle "
                         "patch (bin/Trace); two sequential server runs.")
    ap.add_argument("--shuffle-seed", type=int, default=1)
    ap.add_argument("--red", choices=("policy", "idle"), default="policy")
    ap.add_argument("--deterministic", action="store_true",
                    help="argmax. OFF by default: training sampled.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--server-dir", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--port-base", type=int, default=41000)
    ap.add_argument("--route-artifact", type=Path, default=None)
    ap.add_argument("--no-route-table", action="store_true",
                    help="NOT the training configuration; for debugging only")
    a = ap.parse_args(argv)

    if a.make_floor is not None:
        from .policy_driver import PolicyActionLog
        d = a.make_floor
        log = PolicyActionLog.load(d / "policy_policy_actions.json")
        runs = {}
        for tag, env in (("floor_a", {}),
                         ("floor_b", {"LANERL_SHUFFLE_ORDER": str(a.shuffle_seed)})):
            drv = ReplayWireDriver(log)
            runs[tag] = _record(d, tag=tag, decisions=len(log), driver=drv,
                                server_dir=a.server_dir, config_path=a.config,
                                port_base=a.port_base, extra_env=env)
            print(f"{tag}: {runs[tag]} (unresolved attacks {drv.unresolved})")
        floor = compare_server_logs(runs["floor_a"], runs["floor_b"])
        floor.update({"kind": "policy_divergence_floor",
                      "a": str(runs["floor_a"]), "b": str(runs["floor_b"]),
                      "shuffle_seed": a.shuffle_seed})
        (d / "floor.json").write_text(json.dumps(floor, indent=1))
        print(f"wrote {d / 'floor.json'}; floor first divergence "
              f"{(floor.get('first_divergence') or {}).get('t_ms')}")
        return 0

    from .policy_driver import PolicyActionLog
    if a.existing is not None:
        log = PolicyActionLog.load(a.existing / "policy_policy_actions.json")
        server_log = Path(log.meta.get("server_log")
                          or a.existing / "policy" / "instance000.log")
        if not server_log.exists():        # recorded under another mount
            server_log = a.existing / "policy" / "instance000.log"
        # `--out` keeps a re-score from overwriting the recording's own report.
        out = a.out or a.existing
        out.mkdir(parents=True, exist_ok=True)
    else:
        if not a.checkpoint:
            ap.error("--checkpoint is required (or --existing / --make-floor)")
        out = a.out or Path("lanerl_jax/runs/parity001") / time.strftime("%Y%m%d-%H%M%S")
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        server_log, log = record_policy_run(
            a.checkpoint, out, seconds=a.seconds, red=a.red,
            deterministic=a.deterministic, seed=a.seed, server_dir=a.server_dir,
            config_path=a.config, port_base=a.port_base)
        print(f"recorded {len(log)} decisions in {time.time() - t0:.0f} s: {server_log}")

    engine = TrainingStepEngine(a.route_artifact, use_route_table=not a.no_route_table)
    to_ms = int(a.seconds * 1000) if a.existing is None or a.seconds else None
    rep = replay_and_diff(log, server_log, engine=engine, to_ms=to_ms)
    rep["position_resync"] = replay_resync(log, server_log, engine=engine,
                                           to_ms=to_ms)
    floor = json.loads(a.floor.read_text()) if a.floor else None
    rep = {"gate": "PARITY-001", "checkpoint": log.meta.get("checkpoint"),
           "label": log.meta.get("label"), "red": log.meta.get("red"),
           "deterministic": log.meta.get("deterministic"),
           "seconds": a.seconds, "server_log": str(server_log),
           "orders": {"blue": _wire_counts(log.blue), "red": _wire_counts(log.red)},
           "not_scored": {"ignored": NOT_MODELLED, "reported_only": REPORTED_ONLY,
                          "reported_only_free_run": (FREE_RUN_REPORTED_ONLY
                                                     if ROUTING_APPROX else {}),
                          "tracked_buffs": list(TRACKED_BUFFS)},
           **rep}
    rep["verdict"] = score_against_floor(rep, floor)
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    print(_headline(rep))
    print(f"wrote {out / 'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
