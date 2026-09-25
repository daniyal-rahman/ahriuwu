"""Fog must come from the SERVER, on every frame, in production.

The server computes real per-side visibility, terrain line-of-sight and all,
and puts it on the wire as ``vb``/``vr``. ``visible_ids_for`` uses it when
present and otherwise silently substitutes ``ApproxFogModel`` -- a radius-only
model with no terrain at all.

Two things make that substitution dangerous rather than merely approximate:

1. It is ALL-OR-NOTHING per frame. One unit missing the field discards the
   server's answer for every unit in that frame.
2. It warns ONCE per process, via ``warnings.warn`` with a module-level
   ``_FOG_WARNED`` latch. In a training run that single warning is one line
   among millions, emitted from an actor subprocess, and after it the agent
   goes on seeing through walls for the rest of the run with no further
   signal.

A policy trained on radius-only fog has learned a different game: it can see
an enemy standing behind the lane wall that a real client could not. Nothing
downstream would look wrong -- CS, reward and loss curves are all perfectly
happy.

So this asserts the thing the project actually wants to be true: against a
real server, ``ObservationBuilder.fog_source`` is ``"server"`` for every
frame of a real episode, for both teams.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from lanerl_train import paths
from lanerl_train.ports import PortAllocator
from lanerl_train.vec import ServerLaunchSpec, VecLaneEnv

from lanerl_rl import constants as C
from lanerl_rl.frame import decode_frame, visible_ids_for
from lanerl_rl.obs import ObservationBuilder

pytestmark = pytest.mark.skipif(
    not paths.server_available(),
    reason="vendored server build not available (or LANERL_SKIP_SERVER_TESTS=1)",
)

DECISIONS = 150


@pytest.mark.slow
def test_every_frame_of_a_real_episode_uses_server_fog(tmp_path):
    env = VecLaneEnv(
        1,
        spec=ServerLaunchSpec(toponly=True, bot_teams="both"),
        log_dir=tmp_path / "logs",
        ports=PortAllocator(base=47600).allocate(1),
        step_timeout_s=180.0,
        auto_restart=False,
    )
    env.start()
    try:
        assert all(env.alive), f"instance failed to boot: {env.alive}"
        builders = {t: ObservationBuilder(t) for t in (C.TEAM_BLUE, C.TEAM_RED)}
        sources = {t: set() for t in builders}
        missing_example = {}
        frames = 0

        for _ in range(DECISIONS):
            raw = env.last_obs[0]
            if raw is not None:
                frames += 1
                for team, b in builders.items():
                    b.build(decode_frame(raw))
                    sources[team].add(b.fog_source)
                if frames == 1 or "approx" in sources[C.TEAM_BLUE]:
                    # capture WHICH units lack the field, so a failure names
                    # the culprit instead of just saying "some unit"
                    bad = [u.get("k") for u in raw.get("u", ())
                           if u.get("vb") is None and u.get("vr") is None]
                    if bad:
                        missing_example = {k: bad.count(k) for k in set(bad)}
            env.step([None])
    finally:
        env.close()

    assert frames > 50, f"only {frames} frames observed; the probe is vacuous"
    for team, seen in sources.items():
        assert seen == {"server"}, (
            f"team {team} fell back to ApproxFogModel on at least one frame "
            f"(sources seen: {sorted(seen)}). That model is radius-only with "
            f"NO terrain line-of-sight, so the agent can see through walls, "
            f"and the fallback warns once per process and then stays silent. "
            f"Unit kinds on the wire with neither vb nor vr: "
            f"{missing_example or 'none found -- the frame may have been empty'}"
        )


@pytest.mark.slow
def test_the_probe_would_notice_the_fallback(tmp_path):
    """A frame with the visibility field stripped must report ``approx``.

    Without this, a passing test above could mean "server fog everywhere" or
    "the check does not work".
    """
    env = VecLaneEnv(
        1,
        spec=ServerLaunchSpec(toponly=True, bot_teams="both"),
        log_dir=tmp_path / "logs2",
        ports=PortAllocator(base=47700).allocate(1),
        step_timeout_s=180.0,
        auto_restart=False,
    )
    env.start()
    try:
        for _ in range(20):
            env.step([None])
        raw = env.last_obs[0]
        assert raw is not None
    finally:
        env.close()

    intact = decode_frame(raw)
    _, src = visible_ids_for(intact, C.TEAM_BLUE)
    assert src == "server", "the live server is not emitting vb/vr at all"

    stripped = {**raw, "u": [{k: v for k, v in u.items() if k not in ("vb", "vr")}
                             for u in raw["u"]]}
    _, src2 = visible_ids_for(decode_frame(stripped), C.TEAM_BLUE)
    assert src2 == "approx", (
        "stripping vb/vr did not trigger the fallback, so fog_source cannot "
        "distinguish server fog from the approximation and the test above is "
        "vacuous"
    )
