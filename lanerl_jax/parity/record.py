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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional

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
                         {"t": "move",
                          "x": float(ch["x"]) + sign * 60.0 * math.cos(i * 0.7),
                          "y": float(ch["y"]) + sign * 60.0 * math.sin(i * 0.7)})
        elif tid is not None and dist < 2500.0:
            tgt = next(u for u in obs["u"] if u.get("id") == tid)
            out[side] = {"t": "move", "x": float(tgt["x"]), "y": float(tgt["y"])}
        elif i % 7 == 0:
            out[side] = {"t": "noop"}
        else:
            # nothing to fight yet: walk to where the waves meet
            out[side] = {"t": "move", "x": MEETING_POINT[0], "y": MEETING_POINT[1]}
    return out


def record_trace(
    out_dir: Path,
    decisions: int = 600,
    port_base: int = 41000,
    step_ticks: int = 2,
    tag: str = "fixture",
) -> Path:
    """Boot one server, drive it, and return the path to its log.

    The log carries the STATEHASH/STATEROW stream; parse it with
    :func:`lanerl_jax.parity.trace.load_trace`.
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
            extra_env={"LANERL_STATE_DUMP": "1", "LANERL_STATE_DUMP_FULL": "1",
                       **TRACE_ENV},
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
                   tag: str = "fixture") -> Fixture:
    """Record one episode and return the paths to all three streams."""
    log = record_trace(out_dir, decisions=decisions, port_base=port_base, tag=tag)
    out_dir = Path(out_dir)
    return Fixture(log=log,
                   actions=out_dir / f"{tag}_actions.json",
                   observations=out_dir / f"{tag}_obs.jsonl")
