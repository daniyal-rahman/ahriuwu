"""Record the TRAINED policy's positions + actions for the map plot."""
import json, math, os, socket, subprocess, sys, time
# derive the repo from this file: the export is /srv/nfs on danilogin and
# /mnt/nfs on desktop, so either literal resolves on exactly one node
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
V=os.path.join(os.path.dirname(_REPO), "lanerl-vendor"); BIN=f"{V}/LoLServer/GameServerConsole/bin/Release/net6.0"
CK=sys.argv[1]; OUT=sys.argv[2]; HORIZON=int(sys.argv[3])
def free():
    with socket.socket() as s: s.bind(("127.0.0.1",0)); return s.getsockname()[1]
if CK=="random":
    from lanerl_train.eval_vs_bot import random_policy as pol
else:
    from lanerl_train.eval_vs_bot import trained_policy; pol=trained_policy(CK)
cp,gp=free(),free()
env=dict(os.environ,DOTNET_ROOT=f"{V}/dotnet",LANERL_HEADLESS="1",LANERL_FREERUN="1",
         LANERL_BOT="purple",LANERL_CONTROL_PORT=str(cp),LANERL_STEP_TICKS="2")
p=subprocess.Popen([f"{BIN}/GameServerConsole","--config",
  os.path.join(_REPO, "lanerl/cfg/garen1v1.json"),"--port",str(gp)],
  cwd=BIN,env=env,stdout=open(OUT+".log","w"),stderr=subprocess.STDOUT)
rows=[]
try:
    s=None
    for _ in range(90):
        try: s=socket.create_connection(("127.0.0.1",cp),timeout=2); break
        except OSError: time.sleep(1)
    f=s.makefile("rwb")
    def step(a):
        f.write((json.dumps(a,separators=(",",":"))+"\n").encode()); f.flush()
        l=f.readline(); return json.loads(l) if l else None
    raw=json.loads(f.readline())
    while raw and raw["t"] < HORIZON:
        act=pol(raw); b=act.get("blue",{})
        ch={u["tm"]:u for u in raw["u"] if u["k"]=="Champion"}
        me=ch.get(100)
        if me: rows.append({"t":raw["t"],"x":me["x"],"y":me["y"],"hp":me["hp"],
                            "cs":me.get("cs"),"a":b.get("t","noop")})
        raw=step(act)
    json.dump(rows, open(OUT,"w"))
    print(f"wrote {OUT}  n={len(rows)}")
finally:
    p.terminate()
    try: p.wait(timeout=10)
    except Exception: p.kill()
