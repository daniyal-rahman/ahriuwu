"""Dump a short bot-vs-bot game as newline-delimited raw frames, for fixtures."""
import json,os,socket,subprocess,sys,time
V="/srv/nfs/projects/lanerl-vendor"; BIN=f"{V}/LoLServer/GameServerConsole/bin/Release/net6.0"
CFG="/srv/nfs/projects/ahriuwu-lanerl/lanerl/cfg/garen1v1.json"
OUT=sys.argv[1]; HOR=int(sys.argv[2])
def free():
    with socket.socket() as s: s.bind(("127.0.0.1",0)); return s.getsockname()[1]
cp,gp=free(),free()
env=dict(os.environ,DOTNET_ROOT=f"{V}/dotnet",LANERL_HEADLESS="1",LANERL_FREERUN="1",
         LANERL_BOT="both",LANERL_TOPONLY="1",LANERL_CONTROL_PORT=str(cp),LANERL_STEP_TICKS="2")
p=subprocess.Popen([f"{BIN}/GameServerConsole","--config",CFG,"--port",str(gp)],
    cwd=BIN,env=env,stdout=open("/tmp/rec.log","w"),stderr=subprocess.STDOUT)
n=0
try:
    s=None
    for _ in range(90):
        try: s=socket.create_connection(("127.0.0.1",cp),timeout=2); break
        except OSError: time.sleep(1)
    f=s.makefile("rwb"); empty=(json.dumps({},separators=(",",":"))+"\n").encode()
    raw=json.loads(f.readline())
    with open(OUT,"w") as out:
        while raw and raw["t"]<HOR:
            out.write(json.dumps(raw,separators=(",",":"))+"\n"); n+=1
            f.write(empty); f.flush()
            line=f.readline()
            if not line: break
            raw=json.loads(line)
    print(f"wrote {OUT}  frames={n}  t={raw['t']/1000:.0f}s")
finally:
    p.terminate()
    try: p.wait(timeout=10)
    except Exception: p.kill()
