#!/usr/bin/env python
"""Run one headless LoLServer game with a given lanerl env and collect its output.

Everything the bot and the reset expose is opt-in through env vars, so a run is
fully described by the env dict -- which is what makes a sweep of difficulty
anchors a loop over dicts rather than a loop over config files.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path

# The NFS export is mounted at /srv/nfs on danilogin and /mnt/nfs on desktop.
# Never hardcode either: resolve everything from this file's own location.
_PROJECTS = Path(__file__).resolve().parents[3]
_VENDOR = str(_PROJECTS / "lanerl-vendor")
_REPO = _PROJECTS / "ahriuwu-lanerl"

SERVER_DIR = Path(
    _VENDOR + "/LoLServer/GameServerConsole/bin/Release/net6.0"
)
DOTNET_ROOT = Path(_VENDOR + "/dotnet")
DEFAULT_CONFIG = Path(
    str(_REPO / "lanerl_bot/configs/garen1v1_bot.json")
)

CS_RE = re.compile(
    r"LANERL_CS t=(\d+) name=(\S+) team=(\d+) cs=(\d+) gold=(\d+) lvl=(\d+) "
    r"hp=(\d+)/(\d+) deaths=(\d+)"
)


def resolve_bot_config(value: str) -> Path:
    """A bench config's LANERL_BOT_CONFIG value -> an absolute path on this node.

    Bench configs name their bot configs repo-relative ("configs/anchor_gold.json")
    so that they carry no mount point at all: the export is reachable under a
    different prefix on each Slurm node (see the note at the top of this file), so
    an absolute value silently pointed at nothing on the other one.  Absolute values
    are still accepted -- recorded runs used them -- but they have to resolve here,
    and check_bot_config() says so if they do not.
    """
    p = Path(value).expanduser()
    return p if p.is_absolute() else _REPO / p


def check_bot_config(env_extra: dict, where: str = "") -> dict:
    """Resolve and validate LANERL_BOT_CONFIG, raising if it does not exist.

    LanerlConfig.Load() used to ignore a set-but-unresolvable LANERL_BOT_CONFIG
    with no warning and run the bot's compiled-in defaults, so an A/B compared the
    default bot against itself and reported seed noise as a tuning result.  The
    server now throws too; this raises first so a sweep dies in a second instead
    of after an hour of games.
    """
    value = env_extra.get("LANERL_BOT_CONFIG")
    if value in (None, ""):
        return dict(env_extra)
    path = resolve_bot_config(str(value))
    if not path.is_file():
        raise FileNotFoundError(
            f"LANERL_BOT_CONFIG={value!r} does not resolve on "
            f"{os.uname().nodename} (tried {path})"
            + (f" [{where}]" if where else "")
            + " -- that arm would run the DEFAULT bot, not the config it names."
        )
    return dict(env_extra) | {"LANERL_BOT_CONFIG": str(path)}


def run(env_extra: dict, log: Path, config: Path = DEFAULT_CONFIG,
        port: int = 5119, timeout_s: float = 1800.0) -> dict:
    env_extra = check_bot_config(env_extra, where=str(log.name))
    config = Path(config)
    if not config.is_file():
        raise FileNotFoundError(
            f"--config {config} does not exist on {os.uname().nodename}; the "
            "server would exit before playing a tick and the empty log would "
            "parse as a clean game with zero CS."
        )
    env = dict(os.environ)
    env["DOTNET_ROOT"] = str(DOTNET_ROOT)
    env["LANERL_HEADLESS"] = "1"
    env["LANERL_FREERUN"] = "1"
    env.update({k: str(v) for k, v in env_extra.items()})

    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    with log.open("wb") as fh:
        proc = subprocess.Popen(
            [str(DOTNET_ROOT / "dotnet"), "./GameServerConsole.dll",
             "--config", str(config), "--port", str(port)],
            cwd=SERVER_DIR, env=env, stdout=fh, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            proc.wait(timeout=timeout_s)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    proc.kill()
                proc.wait(timeout=30)

    wall = time.monotonic() - t0
    return parse_log(log) | {"wall_s": wall, "timed_out": timed_out}


def parse_log(log: Path) -> dict:
    text = log.read_text(errors="replace")
    cs_rows = []
    for m in CS_RE.finditer(text):
        cs_rows.append({
            "t": int(m.group(1)), "name": m.group(2), "team": int(m.group(3)),
            "cs": int(m.group(4)), "gold": int(m.group(5)), "lvl": int(m.group(6)),
            "hp": int(m.group(7)), "mhp": int(m.group(8)), "deaths": int(m.group(9)),
        })
    resets = [l for l in text.splitlines() if l.startswith("LANERL_RESET_BENCH")]
    episodes = [l for l in text.splitlines() if l.startswith("LANERL_EPISODE")]
    tps = [float(m) for m in re.findall(r"LANERL_TPS ([\d.]+) ticks/s", text)]
    attach = [l for l in text.splitlines() if l.startswith("LANERL_BOT_ATTACH")]
    warns = [l for l in text.splitlines() if "LANERL_RESET_WARN" in l]
    fatal = [l for l in text.splitlines() if " FATAL " in l or "Unhandled exception" in l]
    return {
        "cs_rows": cs_rows,
        "reset_lines": resets,
        "episode_lines": episodes,
        "tps": tps,
        "attach": attach,
        "reset_warns": warns,
        "fatal": fatal[:10],
    }


def cs_at(rows: list, t_ms: int) -> dict:
    """CS per champion at (or just before) a game time."""
    out = {}
    for r in rows:
        if r["t"] <= t_ms + 1000:
            out[r["name"]] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE")
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--port", type=int, default=5119)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    env_extra = {}
    for kv in args.env:
        k, _, v = kv.partition("=")
        env_extra[k] = v

    res = run(env_extra, args.log, args.config, args.port, args.timeout)
    summary = {
        "env": env_extra,
        "wall_s": res["wall_s"],
        "timed_out": res["timed_out"],
        "tps_median": sorted(res["tps"])[len(res["tps"]) // 2] if res["tps"] else None,
        "attach": res["attach"],
        "reset_lines": res["reset_lines"],
        "episode_lines": res["episode_lines"][:5],
        "n_episodes": len(res["episode_lines"]),
        "reset_warns": res["reset_warns"][:10],
        "fatal": res["fatal"],
        "cs_at_600k": cs_at(res["cs_rows"], 600_000),
        "cs_rows": res["cs_rows"],
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2))
    printable = dict(summary)
    printable.pop("cs_rows", None)
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
