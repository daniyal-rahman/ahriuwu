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
import os
from pathlib import Path

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

def selected_server_dir():
    """One explicit binary selection for eligibility and every live launch."""
    return Path(os.environ.get("LANERL_TEST_SERVER_DIR", paths.server_dir())).resolve()


def _selected_server_available():
    selected = selected_server_dir()
    available = (selected / "GameServerConsole").is_file() or (
        (selected / "GameServerConsole.dll").is_file()
        and (paths.dotnet_root() / "dotnet").is_file())
    if "LANERL_TEST_SERVER_DIR" in os.environ:
        if os.environ.get("LANERL_SKIP_SERVER_TESTS") == "1":
            raise RuntimeError("explicit live server requested but LANERL_SKIP_SERVER_TESTS=1")
        if not available:
            raise RuntimeError(f"explicit live server is unavailable: {selected}")
    return available and os.environ.get("LANERL_SKIP_SERVER_TESTS") != "1"


pytestmark = pytest.mark.skipif(
    not _selected_server_available(), reason="default live server unavailable; select LANERL_TEST_SERVER_DIR",
)

N = 2  # kept small: the login node has 6 cores; the 16-instance figure is desktop's


def champions(obs):
    return {u["tm"]: u for u in obs["u"] if u["k"] == "Champion"}


@pytest.fixture(scope="module")
def live_env(tmp_path_factory):
    spec = ServerLaunchSpec(step_ticks=4, toponly=True, bot_teams="none",
                            server_dir=selected_server_dir())
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

    spec = ServerLaunchSpec(step_ticks=4, toponly=True, bot_teams="none",
                            server_dir=selected_server_dir())
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


@pytest.mark.slow
def test_coordinate_click_hits_visible_unit_and_empty_ground_moves(tmp_path):
    """Exercise server hit-testing, not a client-selected entity ID."""
    import os
    from pathlib import Path

    override = os.environ.get("LANERL_TEST_SERVER_DIR")
    spec = ServerLaunchSpec(step_ticks=4, toponly=True, bot_teams="none",
                            server_dir=selected_server_dir())
    env = VecLaneEnv(1, spec=spec, log_dir=tmp_path / "click_logs",
                     ports=PortAllocator(base=48700).allocate(1),
                     step_timeout_s=120.0, auto_restart=False)
    try:
        env.start()
        # Approach through ordinary navigation; no teleport/setup entity pointer.
        for _ in range(2400):
            obs = env.last_obs[0]
            blue = champions(obs)[100]
            candidates = [u for u in obs["u"] if u["k"] == "LaneMinion"
                          and u["tm"] == 200 and u["hp"] > 0 and u["vb"]
                          and math.dist((blue["x"], blue["y"]), (u["x"], u["y"])) < 700]
            if candidates:
                break
            result = env.step([{"blue": {"t": "move", "x": 3000.0, "y": 12700.0}}])
            assert all(result.alive)
        assert candidates, "setup never brought a visible enemy into the local view"
        unit = min(candidates, key=lambda u: math.dist((blue["x"], blue["y"]), (u["x"], u["y"])))
        result = env.step([{"blue": {"t": "click", "button": "attack_move",
                                     "x": unit["x"], "y": unit["y"]}}])
        assert all(result.alive)
        assert champions(env.last_obs[0])[100]["tgt"] == unit["id"]
        # Ordinary right-click on vacant ground must clear the selected target.
        blue = champions(env.last_obs[0])[100]
        point = (blue["x"] - 300, blue["y"] - 300)
        env.step([{"blue": {"t": "click", "button": "move", "x": point[0], "y": point[1]}}])
        for _ in range(5):
            assert champions(env.last_obs[0])[100]["tgt"] == 0
            env.step([None])
        # A-click on empty ground retains attack-move semantics. Unlike the
        # old decoder it can acquire a nearby minion without clicking its body.
        obs = env.last_obs[0]
        blue = champions(obs)[100]
        nearby = [u for u in obs['u'] if u['k'] == 'LaneMinion'
                  and u['tm'] == 200 and u['hp'] > 0 and u['vb']]
        assert nearby
        unit = min(nearby, key=lambda u: math.dist((blue['x'], blue['y']), (u['x'], u['y'])))
        point = (unit['x'] - 120, unit['y'] - 120)
        env.step([{'blue': {'t': 'click', 'button': 'attack_move',
                            'x': point[0], 'y': point[1]}}])
        acquired = False
        for _ in range(120):
            frame = env.last_obs[0]
            targets = {u['id'] for u in frame['u'] if u['k'] == 'LaneMinion'
                       and u['tm'] == 200 and u['vb'] and u['hp'] > 0}
            acquired |= champions(frame)[100]['tgt'] in targets
            env.step([None])
        assert acquired, 'A-click never acquired a visible minion from ground'
    finally:
        env.close()


@pytest.mark.slow
def test_disabled_q_keyboard_presses_do_not_refresh_empowerment(tmp_path):
    """HUD-disabled Q must be ignored, not repeatedly re-cast at zero CD."""
    import os
    from pathlib import Path
    override = os.environ.get('LANERL_TEST_SERVER_DIR')
    env = VecLaneEnv(1, spec=ServerLaunchSpec(step_ticks=4, toponly=True,
                     bot_teams='none', server_dir=selected_server_dir()),
                     log_dir=tmp_path/'q_hud', ports=PortAllocator(base=48800).allocate(1),
                     step_timeout_s=120., auto_restart=False)
    try:
        env.start()
        env.step([{'blue': {'t': 'level', 'slot': 0}}])
        me = champions(env.last_obs[0])[100]
        assert me['se'][0] == 1 and me['sl'][0] == 1
        click = {'blue': {'t': 'click', 'button': 'q', 'x': me['x'], 'y': me['y']}}
        env.step([click])
        me = champions(env.last_obs[0])[100]
        assert me['se'][0] == 0 and me['cd0'] == 0
        # Six seconds of repeated presses must not extend the 4.5s buff.
        for _ in range(90):
            env.step([click])
        me = champions(env.last_obs[0])[100]
        assert me['cd0'] > 1000, 'disabled Q presses kept refreshing its empowerment'
    finally:
        env.close()


@pytest.mark.slow
def test_dead_positive_hp_rejects_gameplay_until_respawn(tmp_path):
    """A real corpse may regenerate HP; keyboard eligibility must use IsDead."""
    import json
    import os
    from pathlib import Path

    override = os.environ.get('LANERL_TEST_SERVER_DIR')
    env = VecLaneEnv(1, spec=ServerLaunchSpec(step_ticks=6, toponly=True,
        bot_teams='none', server_dir=selected_server_dir(),
        extra_env={'LANERL_AUTOBUY': '0'}), log_dir=tmp_path/'dead_control',
        ports=PortAllocator(base=48900).allocate(1), step_timeout_s=120., auto_restart=False)
    evidence = []
    try:
        env.start()
        me = champions(env.last_obs[0])[100]
        assert me.get('dead') is False, 'requires authoritative-death server build'
        env.step([{'blue': {'t': 'level', 'slot': 2}}])
        # Earn proximity XP behind the wave, without casts or attacks. Level
        # four leaves enough death-screen time for the passive to heal a corpse.
        env.step([{'blue': {'t': 'move', 'x': 1500., 'y': 11800.}}])
        for _ in range(5000):
            me = champions(env.last_obs[0])[100]
            assert not me['dead'], 'setup died before reaching the required level'
            if me['lvl'] >= 4:
                break
            env.step([None])
        assert me['lvl'] >= 4, 'setup never earned proximity XP'
        assert me['cs'] == 0, 'setup unexpectedly farmed'
        evidence.append({'event': 'setup', 't': env.last_obs[0]['t'], 'champion': me})
        # Ordinary navigation into the opposing wave/turret causes a real death.
        env.step([{'blue': {'t': 'move', 'x': 3907., 'y': 13243.}}])
        for _ in range(1800):
            me = champions(env.last_obs[0])[100]
            if me['dead']:
                break
            env.step([None])
        assert me['dead'], 'setup did not die under enemy fire'
        assert me['cd2'] <= 0, 'setup must not leave an existing E running'
        dead_cs = me['cs']
        dead_at = env.last_obs[0]['t']
        evidence.append({'event': 'death', 't': dead_at, 'champion': me})
        positive_hp_samples = 0
        commands = [{'t': 'click', 'button': b, 'x': 1950., 'y': 12350.}
                    for b in ('e', 'move', 'e', 'attack_move', 'e', 'q', 'e', 'w', 'e', 'r')]
        commands += [{'t': 'recall'}, {'t': 'move', 'x': 1950., 'y': 12350.},
                     {'t': 'attack', 'id': champions(env.last_obs[0])[200]['id']}]
        for i in range(600):
            me = champions(env.last_obs[0])[100]
            if not me['dead']:
                break
            positive_hp_samples += me['hp'] > 0
            assert me['cs'] == dead_cs, 'dead keyboard inputs earned CS'
            assert me['cd2'] <= 0, 'dead keyboard E started a spin/cooldown'
            assert not me.get('rc', 0), 'dead keyboard recall started a channel'
            evidence.append({'event': 'dead_input', 't': env.last_obs[0]['t'],
                             'champion': me, 'command': commands[i % len(commands)]})
            env.step([{'blue': commands[i % len(commands)]}])
        assert not me['dead'], 'champion never respawned'
        assert positive_hp_samples > 0, 'regression did not exercise positive-HP corpse'
        assert me['cs'] == dead_cs
        evidence.append({'event': 'respawn', 't': env.last_obs[0]['t'], 'champion': me})
        env.step([{'blue': {'t': 'click', 'button': 'e', 'x': me['x'], 'y': me['y']}}])
        me = champions(env.last_obs[0])[100]
        assert not me['dead'] and me['cd2'] > 0, 'alive E did not work after respawn'
        evidence.append({'event': 'alive_e', 't': env.last_obs[0]['t'], 'champion': me})
        print(json.dumps({'dead_at_ms': dead_at, 'positive_hp_dead_samples': positive_hp_samples,
                          'cs_before_after_death': [dead_cs, me['cs']], 'post_respawn_e_cd': me['cd2']}))
    finally:
        (tmp_path/'dead_control_evidence.json').write_text(json.dumps(evidence, indent=2))
        env.close()


@pytest.mark.slow
def test_reset_skill_setup_does_not_advance_the_other_server(tmp_path):
    import os
    from pathlib import Path
    import numpy as np
    from lanerl_jax.train.server_train import ServerCollector
    override = os.environ.get('LANERL_TEST_SERVER_DIR')
    collector = ServerCollector(2, tmp_path, 48600, 600.,
                                server_dir=selected_server_dir())
    try:
        peer_before = collector.env.last_obs[1]
        collector.restart_done(np.array([True, False]))
        assert collector.env.last_obs[1] == peer_before
        assert collector.episodes == [1, 0]
    finally:
        collector.close()
