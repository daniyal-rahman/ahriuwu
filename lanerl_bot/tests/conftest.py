"""Shared fixtures: running the real headless server, and where its artefacts live.

Tests that need a server are marked `slow` and skip cleanly when the build is
missing or when LANERL_SKIP_SERVER_TESTS=1, so the pure-model tests stay runnable
anywhere.
"""
from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

# The NFS export is mounted at /srv/nfs on danilogin and /mnt/nfs on desktop.
# Never hardcode either: resolve everything from this file's own location.
_PROJECTS = Path(__file__).resolve().parents[3]
_VENDOR = str(_PROJECTS / "lanerl-vendor")
_REPO = _PROJECTS / "ahriuwu-lanerl"

REPO = _REPO
SERVER_DIR = Path(
    _VENDOR + "/LoLServer/GameServerConsole/bin/Release/net6.0"
)
DOTNET_ROOT = Path(_VENDOR + "/dotnet")
CONFIG = REPO / "lanerl_bot/configs/garen1v1_bot.json"
ARTIFACTS = REPO / "lanerl_bot/bench/out"

sys.path.insert(0, str(REPO))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: boots a real game server (tens of seconds)")


def server_available() -> bool:
    if os.environ.get("LANERL_SKIP_SERVER_TESTS") == "1":
        return False
    return (SERVER_DIR / "GameServerConsole.dll").exists() and (DOTNET_ROOT / "dotnet").exists()


requires_server = pytest.mark.skipif(
    not server_available(),
    reason="server build or dotnet SDK not available (or LANERL_SKIP_SERVER_TESTS=1)",
)


def free_port() -> int:
    """An unused loopback port.

    Fixed ports (this used to be 5901/5902) collide with any other server on the
    node -- a stray bench process, a parallel job, the smoke test. The loser of
    that race exits without playing, run_server returns a stub log, and every
    test that parses it degrades to `assert []`. That is exactly how the reset
    suite went red for a run: not a regression, a port collision.
    """
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def run_server(env_extra: dict, log: Path, port: int | None = None,
               timeout_s: float = 600.0, expect: str | None = None) -> str:
    """Run one headless game to completion and return its combined output.

    `expect` is a marker the run MUST emit. Without it a server that died on
    startup returns an empty string and the caller asserts against nothing --
    a green-looking vacuous pass. Missing marker is now a loud failure with the
    log tail attached, so the next one is diagnosable from the pytest output.
    """
    if port is None:
        port = free_port()
    env = dict(os.environ)
    env["DOTNET_ROOT"] = str(DOTNET_ROOT)
    env["LANERL_HEADLESS"] = "1"
    env["LANERL_FREERUN"] = "1"
    env.update({k: str(v) for k, v in env_extra.items()})

    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("wb") as fh:
        proc = subprocess.Popen(
            [str(DOTNET_ROOT / "dotnet"), "./GameServerConsole.dll",
             "--config", str(CONFIG), "--port", str(port)],
            cwd=SERVER_DIR, env=env, stdout=fh, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        timed_out = False
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    proc.kill()
                proc.wait(timeout=30)

    text = log.read_text(errors="replace")
    tail = "\n".join(text.splitlines()[-15:])
    if "Address already in use" in text or "AddressAlreadyInUse" in text:
        raise RuntimeError(f"port {port} was taken; server never played:\n{tail}")
    if timed_out:
        raise RuntimeError(f"server did not exit within {timeout_s}s on port {port}:\n{tail}")
    if expect is not None and expect not in text:
        raise RuntimeError(f"server never emitted {expect!r} on port {port}:\n{tail}")
    return text


@pytest.fixture(scope="session")
def artifacts() -> Path:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS


@pytest.fixture(scope="session")
def selftest_output(artifacts) -> str:
    """The C# damage model's own dump of its pure functions."""
    if not server_available():
        pytest.skip("server unavailable")
    log = artifacts / "pytest_selftest.log"
    # a missing dump used to skip(); a skip on a broken server reads as green
    text = run_server({"LANERL_SELFTEST": "1"}, log, timeout_s=300,
                      expect="LANERL_SELFTEST_END")
    return text


@pytest.fixture(scope="session")
def reset_run(artifacts) -> tuple[str, Path]:
    """A short game that resets three times, with the full state record kept."""
    if not server_available():
        pytest.skip("server unavailable")
    rec = artifacts / "pytest_reset.jsonl"
    rec.unlink(missing_ok=True)
    log = artifacts / "pytest_reset.log"
    text = run_server(
        {
            "LANERL_TOPONLY": "1",
            # No bot: the reset is asserted in isolation. With a bot attached the
            # very next tick puts it at the fountain, where auto-buy immediately
            # spends the restored starting gold -- correct behaviour, but it races
            # the 10Hz recorder and makes "gold == 475" a coin flip.
            "LANERL_BOT": "none",
            "LANERL_RECORD": str(rec),
            "LANERL_RESET_EVERY": "150000",
            "LANERL_MAX_EPISODES": "3",
        },
        log, timeout_s=600, expect="LANERL_EXIT",
    )
    return text, rec
