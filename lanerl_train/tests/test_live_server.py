"""End-to-end against the real headless server.

Marked ``slow`` and skipped when the vendored build is missing, but this is the
only test that proves the thing the fakes only model: that ``VecLaneEnv`` speaks
the actual ``LanerlControl`` protocol, that two instances on distinct ports both
survive (a shared port kills all but the first), and that a batched forward over
several live instances scatters to the right ones.

Run it deliberately::

    pytest lanerl_train/tests/test_live_server.py -m slow -s
"""

from __future__ import annotations

import math

import pytest

from lanerl_train import paths
from lanerl_train.ports import PortAllocator
from lanerl_train.vec import (
    EpisodeSpec,
    ServerLaunchSpec,
    SideAssignment,
    VecDriver,
    VecLaneEnv,
)

from .fakes import FakeAdapter, FakeEncoder, FakePolicy

pytestmark = pytest.mark.skipif(
    not paths.server_available(),
    reason="vendored server build not available (or LANERL_SKIP_SERVER_TESTS=1)",
)

N = 2  # kept small: the login node has 6 cores; the 16-instance figure is desktop's


def champions(obs):
    return {u["tm"]: u for u in obs["u"] if u["k"] == "Champion"}


@pytest.fixture(scope="module")
def live_env(tmp_path_factory):
    spec = ServerLaunchSpec(step_ticks=4, toponly=True, bot_teams="none")
    env = VecLaneEnv(
        N,
        spec=spec,
        log_dir=tmp_path_factory.mktemp("vec_logs"),
        ports=PortAllocator(base=23400).allocate(N),
        step_timeout_s=120.0,
        auto_restart=False,
    )
    try:
        env.start()
        yield env
    finally:
        env.close()


@pytest.mark.slow
def test_every_instance_boots_on_its_own_port_and_answers(live_env):
    dead = [i for i, a in enumerate(live_env.alive) if not a]
    assert not dead, f"dead instances: {dead}"
    ports = {p.control for p in live_env.ports} | {p.game for p in live_env.ports}
    assert len(ports) == 2 * N, "a shared port kills every instance but the first"
    for i, obs in enumerate(live_env.last_obs):
        assert obs is not None, f"instance {i} produced no first observation"
        assert "t" in obs and "u" in obs
        assert set(champions(obs)) == {100, 200}, "both champions must be present"


@pytest.mark.slow
def test_lockstep_holds_and_game_time_advances(live_env):
    t0 = [o["t"] for o in live_env.last_obs]
    steps = 30
    for _ in range(steps):
        res = live_env.step([None] * N)
        assert all(res.alive), f"instance died mid-step: {res.died}"
    t1 = [o["t"] for o in live_env.last_obs]
    # 4 ticks per decision at 60 Hz
    expected_ms = steps * 4 * 1000 / 60
    for i, (a, b) in enumerate(zip(t0, t1)):
        assert b > a, f"instance {i}: game time did not advance"
        assert abs((b - a) - expected_ms) < expected_ms * 0.5, f"instance {i}: {b - a}ms"


@pytest.mark.slow
def test_per_team_fog_flags_are_present(live_env):
    """The actor/critic split needs them; ``obs.py`` gates on exactly these."""
    for obs in live_env.last_obs:
        assert all("vb" in u and "vr" in u for u in obs["u"])


@pytest.mark.slow
def test_a_batched_move_reaches_the_right_champion_in_the_right_instance(live_env):
    """The scatter, proved by the world: each instance's champion must move
    toward the target *that instance's* action named."""
    start = [champions(o)[100] for o in live_env.last_obs]
    targets = [(u["x"] + 1500 * (1 if i == 0 else -1), u["y"] + 1500) for i, u in enumerate(start)]
    for _ in range(150):
        actions = [
            {"blue": {"t": "move", "x": float(tx), "y": float(ty)}} for tx, ty in targets
        ]
        res = live_env.step(actions)
        assert all(res.alive), f"instance died: {res.died}"
    for i, (s, (tx, ty)) in enumerate(zip(start, targets)):
        now = champions(live_env.last_obs[i])[100]
        moved = math.dist((s["x"], s["y"]), (now["x"], now["y"]))
        before = math.dist((s["x"], s["y"]), (tx, ty))
        after = math.dist((now["x"], now["y"]), (tx, ty))
        assert moved > 200, f"instance {i}: champion did not move ({moved:.0f} units)"
        assert after < before, f"instance {i}: moved away from its own target"


@pytest.mark.slow
def test_in_process_reset_rewinds_the_clock(live_env):
    before = [o["t"] for o in live_env.last_obs]
    assert all(t > 0 for t in before)
    res = live_env.reset_episodes(range(N))
    assert all(res.alive), f"instance died during reset: {res.died}"
    for i, obs in enumerate(res.obs):
        assert obs is not None
        assert obs["t"] < before[i], f"instance {i}: reset did not rewind ({obs['t']}ms)"


@pytest.mark.slow
def test_driver_runs_two_policies_over_live_instances(live_env):
    blue, red = FakePolicy("blue"), FakePolicy("red")
    driver = VecDriver(
        live_env,
        policies={"blue": blue, "red": red},
        adapter_factory=lambda i, side: FakeAdapter(i, side),
        encoder=FakeEncoder(),
        assignments=[SideAssignment(blue="blue", red="red") for _ in range(N)],
        episode=EpisodeSpec(max_game_ms=10**9),
    )
    for _ in range(10):
        result, dones = driver.step()
        assert all(result.alive), f"instance died: {result.died}"
    assert blue.calls == 10 and red.calls == 10, "one forward per policy per step"
    assert blue.batch_sizes == [N] * 10


@pytest.mark.slow
def test_a_killed_instance_is_restarted_on_the_same_port(tmp_path, caplog):
    """The server is not perfectly stable, so this path is not hypothetical.

    It also proves the restart can rebind: the control listener is killed
    without a graceful close, and a socket stuck in TIME_WAIT would make the
    fresh process fail to bind its port -- a failure that would otherwise only
    show up hours into a run.
    """
    import logging
    import os
    import signal

    spec = ServerLaunchSpec(step_ticks=4, toponly=True, bot_teams="none")
    env = VecLaneEnv(
        2,
        spec=spec,
        log_dir=tmp_path / "logs",
        ports=PortAllocator(base=23500).allocate(2),
        step_timeout_s=120.0,
        auto_restart=True,
        max_restarts_per_instance=2,
    )
    try:
        env.start()
        for _ in range(5):
            env.step([None] * 2)
        victim = env.handles[1]
        pid = victim.proc.pid
        port_before = victim.ports.control

        with caplog.at_level(logging.ERROR, logger="lanerl_train.vec"):
            os.killpg(os.getpgid(pid), signal.SIGKILL)
            res = env.step([None] * 2)
            for _ in range(3):  # the death may land on this step or the next
                if 1 in res.died:
                    break
                res = env.step([None] * 2)

        assert 1 in res.died, "killing a server must be detected, not absorbed"
        assert "INSTANCE DEATH 1/2" in caplog.text
        assert "RESTARTING instance 1" in caplog.text
        assert env.alive[1] is True, "the instance was not brought back"
        assert env.handles[1].ports.control == port_before
        assert env.handles[1].proc.pid != pid
        assert env.restarts[1] == 1

        # the survivor kept going, and both step again afterwards
        assert env.alive[0] is True
        after = env.step([None] * 2)
        assert all(after.alive)
        assert all(o is not None for o in after.obs)
    finally:
        env.close()
