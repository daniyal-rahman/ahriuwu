"""Record parity fixtures: a server episode plus the actions that drove it.

Why the actions are recorded, not just the states
-------------------------------------------------
A Tier-1 one-step differential injects the server's state into the JAX sim,
steps *both* under the same action, and diffs the result.  That requires the
action stream, and it must be the action stream the server actually executed --
not one re-derived later from a policy, which would silently differ the moment
anything about the policy or its RNG changed.  So both are written together,
keyed by game time, and a fixture with mismatched lengths is rejected at load.

Why a scripted drive rather than a policy
-----------------------------------------
The same reason ``test_state_hash_pairs`` uses one: a policy is a variable, and
a fixture whose content depends on which checkpoint happened to be lying around
is not a fixture.  :func:`scripted_action` is deterministic in the decision
index and deliberately mixes movement, attack-moves and casts -- a move-only
drive never touches the cast, channel and buff machinery, which is exactly
where two of this project's three simulation bugs lived.

The server is booted with ``LANERL_TOPONLY=1`` (jungle and the other two lanes
never spawn) and ``bot_teams="none"`` (no scripted bot drives the champions, so
the recorded actions are the *only* thing moving them).

Three streams come out of one run and all three are kept:

* the **state dump** (``LANERL_STATE_DUMP_FULL=1``), one snapshot per tick;
* the **observation stream**, one per decision -- the only place champion
  ``TargetUnit``/``IsAttacking`` appear at all (:mod:`lanerl_jax.parity.targets`);
* the **target traces** (``LANERL_AGGRO_TRACE`` / ``LANERL_TURRET_TRACE``), both
  behaviour-neutral, covering minion retargets and turret targets.

Recording all three together is not redundancy: the state dump cannot see
targeting, and the observation stream cannot see anything the fog hides or
anything below the decision rate.  A fixture missing either one cannot support
the §3 parity targets.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from lanerl_rl import constants as C
from lanerl_rl.frame import LaneFrame
from lanerl_rl.projection import screen_to_world_centred

from ..train.actions import MINIMAP_X_MIN, MINIMAP_Y_MIN
from .targets import TRACE_ENV

__all__ = ["ActionLog", "Fixture", "scripted_action", "record_trace", "record_fixture"]

#: Seed for the server-side bot RNG. Fixed so a fixture is reproducible; the
#: bots are off anyway, but the seed also strides other server-side streams.
SEED = 4242


@dataclass(slots=True)
class ActionLog:
    """The action issued to each side at each decision, keyed by game time."""

    t_ms: List[int]
    blue: List[dict]
    red: List[dict]

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(
            {"t_ms": self.t_ms, "blue": self.blue, "red": self.red}, indent=0))

    @classmethod
    def load(cls, path: Path) -> "ActionLog":
        d = json.loads(Path(path).read_text())
        out = cls(t_ms=d["t_ms"], blue=d["blue"], red=d["red"])
        if not (len(out.t_ms) == len(out.blue) == len(out.red)):
            raise ValueError(
                f"{path}: action log lengths disagree "
                f"({len(out.t_ms)}/{len(out.blue)}/{len(out.red)}). A fixture whose "
                "actions do not line up with its states cannot support a one-step "
                "differential."
            )
        return out


def _champs(obs: Mapping) -> Dict[int, dict]:
    return {u["tm"]: u for u in obs.get("u", []) if u.get("k") == "Champion"}


#: A vertex of the top-lane polyline, where the waves meet. The drive walks
#: here first, because a champion in its fountain has nothing to fight.
MEETING_POINT = (3907.0, 13243.0)
#: `Stats.Range.Total + collision radius` for Garen against a minion, with slack.
ENGAGE_RANGE = 170.0
#: Compatibility name for callers that only need a conservative distance
#: guard. The actual contract is stronger: every scripted Move below is one of
#: the deployed 96x54 projected bin centres, excluding the minimap rectangle.
#: The farthest legal bin centre is about 2,281.27 units from the champion.
LOCAL_MOVE_RADIUS = 2282.0


@lru_cache(maxsize=2)
def _legal_click_offsets(team: int) -> tuple[tuple[float, float], ...]:
    """World offsets for every policy-reachable non-minimap screen bin."""
    enemy = C.TEAM_RED if team == C.TEAM_BLUE else C.TEAM_BLUE
    frame = LaneFrame(
        C.TOP_OUTER_TURRET[team], C.TOP_OUTER_TURRET[enemy],
        C.NEXUS_POSITION[team])
    offsets: list[tuple[float, float]] = []
    for sx in C.SCREEN_X_VALUES:
        for sy in C.SCREEN_Y_VALUES:
            sx_f, sy_f = float(sx), float(sy)
            if sx_f >= MINIMAP_X_MIN and sy_f >= MINIMAP_Y_MIN:
                continue
            ds, dn = screen_to_world_centred(0.0, 0.0, sx_f, sy_f)
            offsets.append(frame.to_world_vector(ds, dn))
    return tuple(offsets)


def _local_move_toward(ch: Mapping, x: float, y: float) -> dict:
    """Choose the legal projected screen bin nearest an arbitrary target."""
    cx, cy = float(ch["x"]), float(ch["y"])
    dx, dy = float(x) - cx, float(y) - cy
    ox, oy = min(_legal_click_offsets(int(ch["tm"])),
                 key=lambda p: (p[0] - dx) ** 2 + (p[1] - dy) ** 2)
    return {"t": "move", "x": cx + ox, "y": cy + oy}


def _nearest_enemy(obs: Mapping, ch: Mapping):
    """``(NetId, distance)`` of the nearest **visible** enemy, or ``(None, inf)``.

    Visibility matters, and it is the reason the first version of this drive
    recorded zero swings. Ordering an attack on a fogged unit *succeeds* --
    ``SetTargetUnit`` takes it -- and then ``ObjAIBase.UpdateTarget`` clears it
    on the very next tick, because ``!TargetUnit.IsVisibleByTeam(Team)`` is one
    of its drop conditions. The order looks accepted and the champion never
    swings. The wire's ``vb``/``vr`` flags are the fog gate, so they are checked
    here rather than trusting distance.
    """
    best, best_d = None, float("inf")
    vis_key = "vb" if ch.get("tm") == 100 else "vr"
    for u in obs.get("u", []):
        if u.get("tm") == ch.get("tm") or "id" not in u:
            continue
        if u.get("k") not in ("LaneMinion", "Champion"):
            continue
        if not u.get(vis_key, 0):
            continue
        d = math.hypot(u["x"] - ch["x"], u["y"] - ch["y"])
        if d < best_d:
            best, best_d = int(u["id"]), d
    return best, best_d


def scripted_action(obs: Optional[Mapping], i: int) -> Optional[Dict[str, dict]]:
    """A fixed, deterministic drive that exercises move / attack / cast.

    Mirrors ``lanerl_train.tests.test_state_hash_pairs._scripted_action`` in
    spirit -- same reasons, same cadence -- with **real** attack orders added,
    because the auto-attack state machine is the mechanic last-hitting depends
    on and a fixture that never swings cannot constrain it.

    The attack target is resolved from the observation, and this is not a
    detail. ``id: 0`` is the wire's documented "no target" value and
    ``LanerlWire`` **rejects** an attack carrying it -- *"an attack with no
    target is not an order -- it used to be a silent no-op"*. The first version
    of this function sent ``{"t": "attack", "id": 0}``, every one of those
    orders was dropped by the server, and the whole recorded corpus contained
    zero champion swings while looking perfectly healthy. Caught by
    ``test_target_traces_on_a_real_server``, which is why that test asserts a
    champion acquired a target at all rather than just that the log parses.

    ``cast`` is different: there ``id`` is optional and 0 legitimately means
    "no target", so the Garen E line below is correct as written.
    """
    if obs is None:
        return None
    out: Dict[str, dict] = {}
    for team, sign in ((100, 1.0), (200, -1.0)):
        ch = _champs(obs).get(team)
        if ch is None:
            continue
        side = "blue" if team == 100 else "red"
        tid, dist = _nearest_enemy(obs, ch)

        if i % 23 == 0:
            out[side] = {"t": "cast", "slot": 2, "id": 0}      # Garen E, self-cast
        elif tid is not None and dist <= ENGAGE_RANGE:
            # in range: alternate swinging and shuffling, so the fixture covers
            # both a clean attack cadence and cancels mid-wind-up
            out[side] = ({"t": "attack", "id": tid} if i % 5 else
                         _local_move_toward(
                             ch,
                             float(ch["x"]) + sign * 60.0 * math.cos(i * 0.7),
                             float(ch["y"]) + sign * 60.0 * math.sin(i * 0.7)))
        elif tid is not None and dist < 2500.0:
            tgt = next(u for u in obs["u"] if u.get("id") == tid)
            out[side] = _local_move_toward(ch, tgt["x"], tgt["y"])
        elif i % 7 == 0:
            out[side] = {"t": "noop"}
        else:
            # nothing to fight yet: walk to where the waves meet
            out[side] = _local_move_toward(ch, *MEETING_POINT)
    return out


def record_trace(
    out_dir: Path,
    decisions: int = 600,
    port_base: int = 41000,
    step_ticks: int = 2,
    tag: str = "fixture",
    extra_env: Optional[Mapping[str, str]] = None,
    server_dir: Optional[Path] = None,
    config_path: Optional[Path] = None,
) -> Path:
    """Boot one server, drive it, and return the path to its log.

    The log carries the STATEHASH/STATEROW stream; parse it with
    :func:`lanerl_jax.parity.trace.load_trace`.

    ``server_dir`` and ``config_path`` select WHICH server binary and which
    script package to record against. They default to the stock build, i.e. the
    uninstrumented one. Every observability field -- ``aacdbits``, ``aagate``,
    ``CallForHelpClear`` -- lives only in the build under ``bin/Trace/net6.0``
    and the isolated ``Content-trace`` package, so a recording made without
    these two arguments will parse cleanly, hash correctly, and contain none of
    them. That is the failure this pair exists to prevent: it is silent, and it
    is the same shape as `METH-003`.
    """
    from lanerl_train.ports import PortAllocator
    from lanerl_train.vec import ServerLaunchSpec, VecLaneEnv

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = VecLaneEnv(
        1,
        spec=ServerLaunchSpec(
            toponly=True,
            bot_teams="none",
            bot_seed=SEED,
            step_ticks=step_ticks,
            server_dir=Path(server_dir) if server_dir else None,
            config_path=Path(config_path) if config_path else None,
            extra_env={"LANERL_STATE_DUMP": "1", "LANERL_STATE_DUMP_FULL": "1",
                       "LANERL_STATE_DUMP_INTERNALS": "1",
                       **TRACE_ENV, **(dict(extra_env) if extra_env else {})},
        ),
        log_dir=out_dir / tag,
        ports=PortAllocator(base=port_base).allocate(1),
        step_timeout_s=180.0,
        auto_restart=False,
    )
    env.start()
    actions = ActionLog(t_ms=[], blue=[], red=[])
    obs_path = out_dir / f"{tag}_obs.jsonl"
    try:
        if not all(env.alive):
            raise RuntimeError(f"server failed to boot: {env.alive}")
        with obs_path.open("w") as obs_fh:
            for i in range(decisions):
                obs = env.last_obs[0]
                act = scripted_action(obs, i)
                if obs is not None:
                    # The observation BEFORE the action -- so a fixture row reads
                    # "in this state, this action was issued", which is the tuple
                    # a one-step differential needs. Writing it after would pair
                    # each action with the state it produced instead.
                    obs_fh.write(json.dumps(obs, separators=(",", ":")) + "\n")
                    if act is not None:
                        actions.t_ms.append(int(obs.get("t", -1)))
                        actions.blue.append(act.get("blue", {"t": "noop"}))
                        actions.red.append(act.get("red", {"t": "noop"}))
                env.step([act])
        log = Path(env.handles[0].log_path)
    finally:
        env.close()

    actions.save(out_dir / f"{tag}_actions.json")
    return log


@dataclass(slots=True, frozen=True)
class Fixture:
    """Everything one recorded episode produces, and where it landed."""

    log: Path            # state dump + target traces (server stdout)
    actions: Path        # ActionLog json
    observations: Path   # one control-channel observation per decision, jsonl

    def load(self):
        """-> (Trace, TargetTraces, ActionLog, [obs dicts])."""
        import json as _json

        from .targets import parse_target_traces
        from .trace import load_trace

        text = self.log.read_text(errors="replace").splitlines()
        return (
            load_trace(self.log),
            parse_target_traces(text),
            ActionLog.load(self.actions),
            [_json.loads(ln) for ln in self.observations.read_text().splitlines() if ln],
        )


def record_fixture(out_dir: Path, decisions: int = 600, port_base: int = 41000,
                   tag: str = "fixture",
                   extra_env: Optional[Mapping[str, str]] = None,
                   server_dir: Optional[Path] = None,
                   config_path: Optional[Path] = None) -> Fixture:
    """Record one episode and return the paths to all three streams."""
    log = record_trace(out_dir, decisions=decisions, port_base=port_base, tag=tag,
                       server_dir=server_dir, config_path=config_path,
                       extra_env=extra_env)
    out_dir = Path(out_dir)
    return Fixture(log=log,
                   actions=out_dir / f"{tag}_actions.json",
                   observations=out_dir / f"{tag}_obs.jsonl")


def _main(argv: Optional[List[str]] = None) -> int:
    """Record one instrumented episode from the command line.

    This exists because the runbook's "re-record the instrumented corpus" step
    had no command behind it: the AA-004 recording was made by an ad-hoc script
    that was never committed, so the single most expensive artefact in the
    project was the one step a reader could not repeat.
    """
    import argparse

    from lanerl_train import paths

    trace_server = (paths.server_dir().parent.parent / "Trace" / "net6.0")
    trace_cfg = Path(__file__).resolve().parents[2] / "lanerl" / "cfg" / "garen1v1_trace.json"

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tag", default="rec")
    ap.add_argument(
        "--game-seconds", type=float, default=420.0,
        help="Wall time of game to record. Must exceed ~200 s: the waves do not "
             "clash until ~110 s, so a shorter run exercises none of the "
             "instrumented sites and 'passes' while proving nothing.")
    ap.add_argument("--port-base", type=int, default=41000)
    ap.add_argument(
        "--stock", action="store_true",
        help="Record against the STOCK build instead of the instrumented one. "
             "The result will carry no aacdbits/aagate/CallForHelpClear and "
             "will not say so -- only pass this deliberately.")
    ap.add_argument("--server-dir", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=None)
    args = ap.parse_args(argv)

    from lanerl_rl import constants as C

    decisions = int(round(args.game_seconds * C.DECISION_HZ))

    server_dir = args.server_dir
    config_path = args.config
    if not args.stock:
        server_dir = server_dir or trace_server
        config_path = config_path or trace_cfg
        for p in (server_dir, config_path):
            if not Path(p).exists():
                raise SystemExit(
                    f"instrumented build missing: {p}\n"
                    "Build it first -- see lanerl/patch_observability.py's docstring.")

    # The instrumented BUILD is only half of it. `CallForHelpClear` and the
    # minion branch stream (`trigger=`/`order=`/`timer=`) are both compiled in
    # but gated at runtime on `LANERL_DECISION_TRACE`, so recording against
    # `bin/Trace` without it yields a log that is instrumented in the assembly
    # and silent in the file -- which is exactly what happened on the first
    # `aa005_gate` recording, and nothing said so. Default it ON for an
    # instrumented recording, since that is the entire reason for choosing one.
    extra_env = None if args.stock else {"LANERL_DECISION_TRACE": "1"}

    fx = record_fixture(args.out, decisions=decisions, port_base=args.port_base,
                        tag=args.tag, server_dir=server_dir,
                        config_path=config_path, extra_env=extra_env)

    # A script that fails to compile does NOT stop the server; it silently stops
    # being the AI and the trace still parses (METH-003). Never hand back a log
    # without saying which it was.
    from .script_health import check_script_load
    print(f"log: {fx.log}")
    # Say which instrumented streams are actually IN the file. Every one of
    # these has now failed silently at least once: the field absent because the
    # build was stock, the emit absent because the env gate was off, the script
    # absent because it failed to compile and the server carried on without it.
    # Streamed, not `read_text`: these logs run to hundreds of MB and this
    # runs on a shared 6-core login node where a needless resident copy is
    # somebody else's problem.
    needles = ((" aacdbits=", "AA-004 unclamped cooldown"),
               (" aagate=", "AA-005 swing gate word"),
               ("CallForHelpClear", "CFH-002 pre-clear map"),
               ("trigger=", "minion branch stream"))
    counts = {n: 0 for n, _ in needles}
    with fx.log.open(errors="replace") as fh:
        for line in fh:
            for needle, _ in needles:
                if needle in line:
                    counts[needle] += 1
    for needle, what in needles:
        n = counts[needle]
        print(f"  {'OK ' if n else 'ABSENT'}  {what} ({n} lines)")
    print(f"actions: {fx.actions}")
    print(f"observations: {fx.observations}")
    print(f"script health: {check_script_load(fx.log)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
