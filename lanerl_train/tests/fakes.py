"""Fakes that let the whole orchestration be tested without a game server.

The fakes reproduce the *contract*, not the behaviour: :class:`FakeInstance`
speaks the same strict lockstep as ``LanerlControl`` (one observation, then a
blocking read of exactly one action line), because that ordering is what the vec
runner's send-all/receive-all structure depends on.  Everything else -- the
policy, the adapter, the encoder, the learner -- is deliberately identity-like
so that a misrouted action shows up as a wrong number rather than as noise.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from lanerl_train.vec import InstanceDied

BLUE_TEAM = 100
RED_TEAM = 200


def make_obs(
    t_ms: int,
    blue_hp: int = 600,
    red_hp: int = 600,
    blue_cs: int = 0,
    red_cs: int = 0,
) -> Dict[str, Any]:
    """An observation shaped like ``LanerlControl.BuildObservation``.

    ``cs`` is present because the server emits it on every champion, and the
    headline skill metric (CS@10) is read off exactly this field: a fake that
    omitted it could not tell a readout taken from the final frame of an
    episode from one taken from the post-reset frame, which is precisely the
    bug the boundary tests exist to catch.
    """
    return {
        "t": int(t_ms),
        "u": [
            {
                "id": 1, "k": "Champion", "tm": BLUE_TEAM, "x": 1000, "y": 12000,
                "hp": blue_hp, "mhp": 600, "vb": 1, "vr": 1,
                "gold": 475, "xp": 0, "lvl": 1, "cs": int(blue_cs),
                "cd0": -1, "cd1": -1, "cd2": -1, "cd3": -1,
            },
            {
                "id": 2, "k": "Champion", "tm": RED_TEAM, "x": 3000, "y": 13000,
                "hp": red_hp, "mhp": 600, "vb": 1, "vr": 1,
                "gold": 475, "xp": 0, "lvl": 1, "cs": int(red_cs),
                "cd0": -1, "cd1": -1, "cd2": -1, "cd3": -1,
            },
            {
                "id": 3, "k": "LaneMinion", "tm": RED_TEAM, "x": 2000, "y": 12500,
                "hp": 477, "mhp": 477, "vb": 1, "vr": 1,
            },
        ],
    }


class FakeInstance:
    """A scripted stand-in for one server process + control socket."""

    def __init__(
        self,
        index: int,
        step_ms: int = 66,
        die_after_sends: Optional[int] = None,
        fail_starts: int = 0,
        stall_forever: bool = False,
        start_t_ms: int = 0,
        cs_per_step: int = 0,
        champ_dies_at_ms: Optional[int] = None,
    ):
        self.index = int(index)
        self.step_ms = int(step_ms)
        self.die_after_sends = die_after_sends
        self.fail_starts = int(fail_starts)
        self.stall_forever = bool(stall_forever)
        self.start_t_ms = int(start_t_ms)
        self.t_ms = int(start_t_ms)
        #: CS accrued per decision, cleared by a reset like the real champion's
        #: ``ChampStats.MinionsKilled`` is (``LanerlEpisode.ResetChampStats``).
        self.cs_per_step = int(cs_per_step)
        #: Game time at which the blue champion's hp hits 0, so a test can end
        #: an episode on a death rather than on the clock.
        self.champ_dies_at_ms = champ_dies_at_ms
        self.cs = 0
        self.outbox: List[str] = []
        self.received: List[Dict[str, Any]] = []
        #: Lines the server would have marked Fatal and executed nothing from.
        self.rejected: List[Dict[str, Any]] = []
        self.sends = 0
        self.starts = 0
        self.closed = False
        self.dead = False
        self.resets = 0

    # -- InstanceHandle ----------------------------------------------------

    def start(self) -> None:
        self.starts += 1
        if self.starts <= self.fail_starts:
            raise InstanceDied(f"fake instance {self.index}: scripted start failure")
        self.closed = False
        self.dead = False
        self.sends = 0
        self.t_ms = self.start_t_ms
        self.cs = 0
        self.outbox = [] if self.stall_forever else [json.dumps(self._obs())]

    def _obs(self) -> Dict[str, Any]:
        dead = self.champ_dies_at_ms is not None and self.t_ms >= self.champ_dies_at_ms
        return make_obs(
            self.t_ms,
            blue_hp=0 if dead else 600,
            blue_cs=self.cs,
            red_cs=self.cs,
        )

    def send_line(self, line: str) -> None:
        if self.dead or self.closed:
            raise InstanceDied(f"fake instance {self.index}: send on a dead channel")
        action = json.loads(line)
        self.received.append(action)
        self.sends += 1
        if self.die_after_sends is not None and self.sends > self.die_after_sends:
            self.dead = True
            return
        # Faithful to LanerlWire.Parse: a reset is EXACTLY {"cmd":"reset"}, and
        # an unknown top-level key makes the whole line Fatal -- no reset and no
        # orders, with the step still advancing. A fake that accepted the old
        # {"reset":1} would have hidden that breakage from every test here.
        if set(action) - {"blue", "red"}:
            if dict(action) == {"cmd": "reset"}:
                self.resets += 1
                self.t_ms = 0
                self.cs = 0
            else:
                self.rejected.append(dict(action))
                self.t_ms += self.step_ms
                self.cs += self.cs_per_step
        else:
            self.t_ms += self.step_ms
            self.cs += self.cs_per_step
        if not self.stall_forever:
            self.outbox.append(json.dumps(self._obs()))

    def read_line(self) -> Optional[str]:
        if self.dead:
            raise InstanceDied(f"fake instance {self.index}: control channel closed")
        if self.closed:
            raise InstanceDied(f"fake instance {self.index}: read on a closed channel")
        return self.outbox.pop(0) if self.outbox else None

    def fileno(self) -> int:
        return -1  # not selectable; the vec runner falls back to polling

    def is_alive(self) -> bool:
        return not (self.dead or self.closed)

    def diagnostics(self) -> str:
        return f"[fake {self.index} sends={self.sends} starts={self.starts} dead={self.dead}]"

    def close(self) -> None:
        self.closed = True

    # -- assertions helpers ------------------------------------------------

    @property
    def last_action(self) -> Optional[Dict[str, Any]]:
        return self.received[-1] if self.received else None


class FakeAdapter:
    """Tags each observation with its own (instance, side) so misrouting is visible."""

    def __init__(self, index: int, side: str):
        self.index = int(index)
        self.side = side
        self.resets = 0
        self.builds = 0

    def reset(self) -> None:
        self.resets += 1

    def build(self, raw: Mapping[str, Any], side: str) -> Dict[str, Any]:
        assert side == self.side, f"adapter for {self.side} was handed {side}"
        self.builds += 1
        return {"instance": self.index, "side": side, "t": int(raw["t"])}


@dataclass
class FakePolicy:
    """Counts forwards and echoes its input, so the scatter can be checked."""

    name: str = "fake"
    version: int = 0
    calls: int = 0
    batch_sizes: List[int] = field(default_factory=list)
    seen_resets: List[List[bool]] = field(default_factory=list)

    def initial_state(self, batch: int) -> Dict[str, Any]:
        return {"batch": batch, "steps": 0}

    def act_batch(
        self,
        observations: Sequence[Mapping[str, Any]],
        state: Dict[str, Any],
        resets: Optional[Sequence[bool]] = None,
        deterministic: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        self.calls += 1
        self.batch_sizes.append(len(observations))
        self.seen_resets.append(list(resets) if resets is not None else [])
        if state["batch"] != len(observations):
            raise AssertionError(
                f"batch width changed under the recurrent state: state was built for "
                f"{state['batch']} slots, got {len(observations)}"
            )
        state = {"batch": state["batch"], "steps": state["steps"] + 1}
        actions = [
            {"instance": o["instance"], "side": o["side"], "policy": self.name}
            for o in observations
        ]
        return actions, state


class FakeEncoder:
    """Encodes the identity of the slot into the order, so a swap is detectable."""

    def encode(
        self, action: Mapping[str, Any], raw: Mapping[str, Any], side: str
    ) -> Dict[str, Any]:
        return {
            "t": "move",
            "x": float(action["instance"]),
            "y": 1.0 if action["side"] == "blue" else 2.0,
            "policy": action["policy"],
        }


class FakeLearner:
    """A learner whose weights are one integer, so a resume is checkable by eye."""

    def __init__(self) -> None:
        self.weights = 0
        self.updates = 0
        self.seen: List[Any] = []

    def update(self, batch: Any) -> Dict[str, float]:
        self.updates += 1
        self.weights += 1
        self.seen.append(batch)
        return {"policy": float(self.weights), "value": 0.5}

    def policy_payload(self) -> Dict[str, Any]:
        return {"weights": self.weights}

    def state_payload(self) -> Dict[str, Any]:
        return {"weights": self.weights, "updates": self.updates}

    def load_payload(self, payload: Mapping[str, Any]) -> None:
        self.weights = int(payload["weights"])
        self.updates = int(payload.get("updates", 0))


class CountingCollect:
    """A ``collect`` callable for :class:`lanerl_train.run.ActorLoop`."""

    def __init__(self, steps: int = 8, raise_after: Optional[int] = None):
        self.steps = int(steps)
        self.raise_after = raise_after
        self.calls = 0
        self.versions: List[int] = []
        self.lock = threading.Lock()

    def __call__(self, actor_id: int, payload: Mapping[str, Any], version: int):
        from lanerl_train.run import Rollout

        with self.lock:
            self.calls += 1
            n = self.calls
            self.versions.append(version)
        if self.raise_after is not None and n > self.raise_after:
            raise RuntimeError(f"scripted actor failure on call {n}")
        return Rollout(
            actor_id=actor_id,
            param_version=version,
            steps=self.steps,
            data={"payload": dict(payload), "call": n},
        )
