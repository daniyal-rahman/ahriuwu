"""Pairs of runs that must produce IDENTICAL state-hash streams.

WHY PAIRS AND NOT CURVES
------------------------
Every bug this project has shipped produced a smooth, believable learning
curve while the simulation was broken -- a champion frozen by a cast flag that
survived death, another by a channel that survived a reset, a map that lost
four turrets over a run, and before those an eval that scored an argmax policy
which never moved. A curve cannot falsify any of them. A diff against a known
answer can, and "these two runs must be bit-identical" is the strongest known
answer available without a second implementation.

``LanerlStateDump`` (LANERL_STATE_DUMP=1) emits one line per decision:

    LANERL_STATEHASH t=<gametime> n=<entities> h=<fnv1a64>

covering every entity's kind, team, position, health and death flag, and for
champions the stat block, level, gold, CS, move order, waypoints, buffs, spell
ranks/cooldowns and the cast/channel flags. Two runs that should agree must
produce the same sequence of ``h``.

WHAT EACH PAIR CATCHES
----------------------
* same seed, same actions, two processes  -> nondeterminism (unseeded RNG,
  thread interleaving, hash-ordered iteration)
* fresh process vs the same episode after N resets -> state leaking across a
  reset, which is the shape of three separate bugs here

The remaining pairs from the plan (speed invariance, process/env layout,
actor-vs-learner log-probs) live elsewhere: the first two need two differently
loaded machines or process trees rather than one pytest, and the last is a
pure-Python check that needs no server.

NOTE ON SCOPE. A failure here is real, but a PASS is not proof of determinism
in general -- it is proof for the driven action sequence and the entity set
these episodes happen to produce. Longer and more varied drives are strictly
better, which is what LANERL_PAIR_DECISIONS is for.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Mapping, Optional

import pytest

from lanerl_train import paths
from lanerl_train.ports import PortAllocator
from lanerl_train.vec import ServerLaunchSpec, VecLaneEnv

pytestmark = pytest.mark.skipif(
    not paths.server_available(),
    reason="vendored server build not available (or LANERL_SKIP_SERVER_TESTS=1)",
)

HASH_RE = re.compile(r"LANERL_STATEHASH t=(-?\d+) n=(\d+) h=([0-9a-f]{16})")

DECISIONS = int(os.environ.get("LANERL_PAIR_DECISIONS", "240"))
RESETS = int(os.environ.get("LANERL_PAIR_RESETS", "25"))
SEED = 4242


def _boot(tmp_path: Path, n: int, base_port: int, tag: str) -> VecLaneEnv:
    env = VecLaneEnv(
        n,
        # Same bot_seed for every instance: the whole point is that two
        # instances are the SAME experiment. Production deliberately strides
        # the seed per env so the bot stream differs; here that would make the
        # comparison meaningless.
        spec=ServerLaunchSpec(
            toponly=True, bot_teams="none", bot_seed=SEED,
            # FULL is passed through so that, once a hash diverges, the
            # same test can be re-run to emit the per-entity rows that say
            # WHICH field moved. That is the whole debugging workflow.
            extra_env={
                "LANERL_STATE_DUMP": "1",
                "LANERL_STATE_DUMP_FULL":
                    os.environ.get("LANERL_STATE_DUMP_FULL", "0"),
            },
        ),
        log_dir=tmp_path / tag,
        ports=PortAllocator(base=base_port).allocate(n),
        step_timeout_s=180.0,
        auto_restart=False,
    )
    env.start()
    return env


def _hashes(log: Path) -> List[str]:
    return [m.group(3) for m in HASH_RE.finditer(log.read_text(errors="replace"))]


def _champs(obs: Mapping):
    return {u["tm"]: u for u in obs["u"] if u.get("k") == "Champion"}


def _scripted_action(obs: Optional[Mapping], i: int):
    """A FIXED action sequence -- no policy, so the network is not a variable.

    Deliberately mixes movement, casts and attacks: a move-only drive leaves
    the cast and channel machinery untouched, which is exactly the machinery
    two of the three bugs lived in.
    """
    if obs is None:
        return None
    out = {}
    for team, sign in ((100, 1.0), (200, -1.0)):
        ch = _champs(obs).get(team)
        if ch is None:
            continue
        if i % 23 == 0:
            out["blue" if team == 100 else "red"] = {"t": "cast", "slot": 2, "id": 0}
        elif i % 7 == 0:
            out["blue" if team == 100 else "red"] = {"t": "noop"}
        else:
            ang = (i % 40) / 40.0 * 6.28318
            import math
            out["blue" if team == 100 else "red"] = {
                "t": "move",
                "x": float(ch["x"]) + sign * 500.0 * math.cos(ang),
                "y": float(ch["y"]) + sign * 500.0 * math.sin(ang),
            }
    return out


def _first_divergence(a: List[str], b: List[str]) -> str:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return (f"first divergence at decision {i}: {x} vs {y} "
                    f"(the {i - 1}th agreed)")
    return f"no divergence in the common prefix; lengths {len(a)} vs {len(b)}"


@pytest.mark.slow
def test_two_identical_instances_produce_identical_state_hashes(tmp_path):
    """Nondeterminism. Same seed, same actions, two servers, one machine.

    This FAILED when written, and the cause is worth keeping: Garen's
    passive was applied from the only ``Task.Run`` in the entire Content
    script tree -- a background thread polling ``Thread.Sleep(1000)`` until
    the game clock started, then calling AddBuff. Two servers diverged on
    the FIRST simulated tick, one holding GarenPassive+GarenPassiveHeal at
    t=17 and the other neither.

    It is now applied from ``ICharScript.OnUpdate``, which the game loop
    calls (``ObjAIBase.cs:1070``), so the buff lands on the first tick past
    the threshold every time. Deterministic by construction, not by luck --
    which is the point: there is no longer a race to lose.
    """
    env = _boot(tmp_path, 2, base_port=47300, tag="determinism")
    try:
        assert all(env.alive), f"an instance failed to boot: {env.alive}"
        for i in range(DECISIONS):
            obs = env.last_obs[0]
            act = _scripted_action(obs, i)
            env.step([act, act])   # THE SAME action to both
        logs = [Path(h.log_path) for h in env.handles]
    finally:
        env.close()

    a, b = _hashes(logs[0]), _hashes(logs[1])
    assert len(a) > 50, f"only {len(a)} state hashes; is LANERL_STATE_DUMP on?"
    assert a == b, (
        "two servers given the same seed and the same actions diverged. "
        "Something in the simulation is nondeterministic -- an unseeded RNG, "
        "thread interleaving, or iteration over a hash-ordered collection. "
        "Every A/B in this project is worthless until this passes. "
        + _first_divergence(a, b)
    )


@pytest.mark.slow
def test_an_episode_after_n_resets_matches_a_fresh_process(tmp_path):
    """Reset leakage -- the shape of three separate bugs here.

    Instance 0 is reset ``RESETS`` times and then driven; instance 1 is
    reset ONCE and driven identically. If a reset were perfect the two
    episodes would be the same episode, hash for hash.
    """
    env = _boot(tmp_path, 2, base_port=47400, tag="resets")
    try:
        assert all(env.alive), f"an instance failed to boot: {env.alive}"
        # wear instance 0 in: play and reset, repeatedly
        for _ in range(RESETS):
            for i in range(60):
                env.step([_scripted_action(env.last_obs[0], i), None])
            res = env.reset_episodes([0])
            assert all(res.alive), f"instance died during wear-in: {res.died}"
        # now put BOTH on a fresh episode and drive them the same way
        res = env.reset_episodes([0, 1])
        assert all(res.alive), f"instance died on the comparison reset: {res.died}"
        mark = [len(_hashes(Path(h.log_path))) for h in env.handles]
        for i in range(DECISIONS):
            act = _scripted_action(env.last_obs[0], i)
            env.step([act, act])
        logs = [Path(h.log_path) for h in env.handles]
    finally:
        env.close()

    worn = _hashes(logs[0])[mark[0]:]
    fresh = _hashes(logs[1])[mark[1]:]
    assert len(worn) > 50 and len(fresh) > 50, (
        f"too few hashes after the comparison reset: {len(worn)} / {len(fresh)}"
    )
    n = min(len(worn), len(fresh))
    assert worn[:n] == fresh[:n], (
        f"an episode on a server reset {RESETS} times is not the episode a "
        f"once-reset server plays. State is leaking across the reset -- the "
        f"shape of the mid-cast freeze, the stuck recall channel and the "
        f"vanishing turrets. Re-run with LANERL_STATE_DUMP_FULL=1 and diff "
        f"the LANERL_STATEROW lines at that decision to see which field. "
        + _first_divergence(worn[:n], fresh[:n])
    )
