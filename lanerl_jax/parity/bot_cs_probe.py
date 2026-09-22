"""Measure the C# LanerlBot's own CS on the server, with NO orders sent.

The acceptance test compared a Python MIRROR of the bot's farm core against the
server and got cs=9 on the server arm. The bot itself is reported to reach 30+
per 10-minute trial, so the mirror -- not the simulator -- is what needs
validating first. This establishes the target the mirror has to hit before any
sim-vs-server comparison through it means anything.

`actions[i] = None` sends a bare `{}`; `LanerlControl.ApplyActions` finds no
side key and leaves standing orders alone, so the in-server bot drives freely.
"""
import sys
from pathlib import Path
from lanerl_train.vec import ServerLaunchSpec, VecLaneEnv

DEC = int(sys.argv[1]) if len(sys.argv) > 1 else 18000  # 18k = 10 game min at 30 Hz
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 59100
log = Path("/tmp/bot_probe_logs"); log.mkdir(parents=True, exist_ok=True)

# Mirrors `run_oracle_on_server`'s construction exactly: port_base and
# log_dir belong to VecLaneEnv, ports come from PortAllocator, and the spec
# carries only the scenario. The one difference is `bot_teams="blue"` -- the
# in-server C# bot drives the champion and this sends no orders at all.
from lanerl_train.ports import PortAllocator

env = VecLaneEnv(
    1,
    spec=ServerLaunchSpec(
        toponly=True, bot_teams="blue", bot_seed=4242, step_ticks=2,
        extra_env={"LANERL_AUTOBUY": "0"}),
    log_dir=log,
    ports=PortAllocator(base=PORT).allocate(1),
    step_timeout_s=180.0,
    auto_restart=False,
)
env.start()
try:
    if not all(env.alive):
        raise SystemExit(f"server failed to boot: {env.alive}")
    last = {}
    # The observation is `obs["u"]`, a flat unit list -- the blue champion is
    # `k == "Champion" and tm == 100` (`last_hit_drive.py:873`). There is no
    # `obs["blue"]`; looking for one silently yielded cs=None for a whole run.
    for i in range(DEC):
        r = env.step([None])
        obs = r.obs[0] if r.obs else None
        if obs and obs.get("u"):
            b = next((u for u in obs["u"]
                      if u.get("k") == "Champion" and u.get("tm") == 100), None)
            if b:
                last = b
        if i % 3000 == 0:
            print(f"PROBE t={i:>6} cs={last.get('cs')} lvl={last.get('lvl')} "
                  f"hp={last.get('hp')}", flush=True)
    print(f"PROBE FINAL decisions={DEC} cs={last.get('cs')} lvl={last.get('lvl')}")
finally:
    env.close()
