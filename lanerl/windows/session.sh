#!/usr/bin/env bash
# One command to put a checkpoint on screen in the real client.
#
#   lanerl/windows/session.sh --checkpoint ckpt_eval/foo.pt            # RL vs RL
#   lanerl/windows/session.sh --checkpoint foo.pt --mode solo          # vs nobody
#   lanerl/windows/session.sh --status                                 # what is up
#   lanerl/windows/session.sh --stop                                   # tear down
#
# Every step here is one that has already failed silently at least once. The
# whole point of the script is that each is CHECKED rather than assumed:
#
#   * the Windows server build goes stale. It is rebuilt from current source
#     and redeployed every time, because a server missing the ad/sl/mt wire
#     fields feeds the policy a different observation than it trained on and
#     nothing errors.
#   * the deploy fails while a server is running -- the DLLs are locked. So
#     everything is stopped BEFORE the copy, and the copy is verified.
#   * the SSH tunnel is the single most common failure. `ssh -L` accepts
#     LOCALLY before dialling the remote, so a connect "succeeding" proves
#     nothing. It is verified with `ss -tlnp` and the script refuses to
#     continue without it.
#   * `schtasks /run` on an already-running task is a silent no-op, so the
#     task is always ended first.
#   * LANERL_TOPONLY must match training. A top-only-trained policy on the
#     full three-lane map walks off the lane axis and collects nothing --
#     observed: level 1 and 0 CS at three minutes.
#
# See docs/archive/WINDOWS_RENDERED_INFERENCE.md for why each of those is true.
set -uo pipefail
cd "$(dirname "$0")/../.."
REPO=$(pwd)

PY=/home/dani/miniconda3/envs/ml/bin/python
VENDOR=/srv/nfs/projects/lanerl-vendor
WIN=windows
MODE=duel              # duel = RL vs RL | solo = RL vs an idle champion
CKPT=""
SKIP_BUILD=0
ACTION=start

while [ $# -gt 0 ]; do
  case "$1" in
    --checkpoint) CKPT="$2"; shift 2;;
    --mode)       MODE="$2"; shift 2;;
    --skip-build) SKIP_BUILD=1; shift;;
    --status)     ACTION=status; shift;;
    --stop)       ACTION=stop; shift;;
    -h|--help)    sed -n '2,30p' "$0"; exit 0;;
    *) echo "unknown arg: $1"; exit 2;;
  esac
done

case "$MODE" in
  duel) PORT=5200; TASK=lanerl_play_rl;   PS1NAME=play_rl.ps1;   SIDES="blue,red"; LOGTAG=rl;;
  solo) PORT=5201; TASK=lanerl_play_solo; PS1NAME=play_solo.ps1; SIDES="blue";     LOGTAG=solo;;
  *) echo "--mode must be duel or solo"; exit 2;;
esac
OUT="lanerl/logs/play_${LOGTAG}.jsonl"
DRIVERLOG="lanerl/logs/play_${LOGTAG}.out"

say() { printf '\n== %s\n' "$*"; }
die() { printf '\nFAILED: %s\n' "$*" >&2; exit 1; }

stop_everything() {
  say "stopping"
  for p in $(ps -eo pid,cmd --no-headers | grep "[p]lay_remote.py" | awk '{print $1}'); do
    kill -9 "$p" 2>/dev/null
  done
  # NEVER pkill -f here: the pattern matches this script's own command line and
  # kills the shell running it. That has happened three times.
  for p in $(ps -eo pid,cmd --no-headers | grep "[s]sh -N" | grep -E "L (5200|5201):" | awk '{print $1}'); do
    kill "$p" 2>/dev/null
  done
  ssh -o ConnectTimeout=20 "$WIN" "schtasks /end /tn lanerl_play_rl"   >/dev/null 2>&1
  ssh -o ConnectTimeout=20 "$WIN" "schtasks /end /tn lanerl_play_solo" >/dev/null 2>&1
  ssh -o ConnectTimeout=25 "$WIN" "powershell -NoProfile -Command \"foreach (\\\$n in @('League of Legends','GameServerConsole')) { Get-Process \\\$n -EA SilentlyContinue | Stop-Process -Force -EA SilentlyContinue }\"" >/dev/null 2>&1
  sleep 3
}

if [ "$ACTION" = stop ]; then stop_everything; echo "stopped."; exit 0; fi

if [ "$ACTION" = status ]; then
  echo "tunnels:";  ss -tlnp 2>/dev/null | grep -E ":(5200|5201)" || echo "  none bound"
  echo "policy:";   ps -eo pid,etime,cmd --no-headers | grep "[p]lay_remote.py" || echo "  not running"
  echo "windows:";  ssh -o ConnectTimeout=20 "$WIN" "powershell -NoProfile -Command \"'  server=' + @(Get-Process GameServerConsole -EA SilentlyContinue).Count + ' client=' + @(Get-Process 'League of Legends' -EA SilentlyContinue).Count\"" 2>/dev/null
  for t in rl solo; do
    f="lanerl/logs/play_${t}.jsonl"
    [ -f "$f" ] && echo "  $t: $(wc -l < "$f") decisions"
  done
  exit 0
fi

[ -n "$CKPT" ] || die "--checkpoint is required"
[ -f "$CKPT" ] || die "no such checkpoint: $CKPT"

stop_everything

if [ "$SKIP_BUILD" = 0 ]; then
  say "publishing the Windows server from current source"
  ( cd "$VENDOR/LoLServer" && DOTNET_CLI_TELEMETRY_OPTOUT=1 ../dotnet/dotnet publish \
      GameServerConsole/GameServerConsole.csproj -c Release -r win-x64 --self-contained \
      -p:SolutionDir="$VENDOR/LoLServer/" -o "$VENDOR/winpub_rl" ) >/tmp/lanerl_publish.log 2>&1 \
    || { tail -5 /tmp/lanerl_publish.log; die "publish failed"; }

  say "deploying to the Windows box"
  # Nothing may hold the DLLs open, which is why stop_everything ran first.
  scp -q -o ConnectTimeout=25 -r "$VENDOR"/winpub_rl/* "$WIN":'C:/lanerl/winpub_rl/' \
    || die "deploy failed -- is a server still running? (--stop, then retry)"
  ssh -o ConnectTimeout=25 "$WIN" "powershell -NoProfile -Command \"(Get-Item C:\lanerl\winpub_rl\GameServerLib.dll).LastWriteTime\"" 2>/dev/null | tail -1
fi

say "deploying the launcher scripts"
scp -q -o ConnectTimeout=25 lanerl/windows/play_rl.ps1 lanerl/windows/play_rl.bat \
    lanerl/windows/play_solo.ps1 lanerl/windows/play_solo.bat "$WIN":'C:/lanerl/' \
  || die "could not copy the launchers"
for t in lanerl_play_rl:play_rl.bat lanerl_play_solo:play_solo.bat; do
  n="${t%%:*}"; b="${t##*:}"
  ssh -o ConnectTimeout=25 "$WIN" "schtasks /create /tn $n /tr \"C:\\lanerl\\$b\" /sc once /st 00:00 /ru daniz /it /f" >/dev/null 2>&1
done

say "opening the SSH tunnel on $PORT and PROVING it is bound"
nohup ssh -N -o ConnectTimeout=25 -o ServerAliveInterval=20 -o ExitOnForwardFailure=yes \
      -L "$PORT:127.0.0.1:$PORT" "$WIN" >/tmp/lanerl_tunnel_$PORT.log 2>&1 &
for i in $(seq 1 15); do
  ss -tlnp 2>/dev/null | grep -q ":$PORT " && break
  sleep 1
done
ss -tlnp 2>/dev/null | grep -q ":$PORT " \
  || { cat /tmp/lanerl_tunnel_$PORT.log; die "tunnel on $PORT never bound"; }
echo "   tunnel $PORT bound"

say "arming the policy (it waits for a real observation, not just a connect)"
rm -f "$OUT" "$DRIVERLOG"
PYTHONPATH="$REPO" nohup "$PY" -u lanerl/play_remote.py \
  --checkpoint "$CKPT" --host 127.0.0.1 --port "$PORT" --sides "$SIDES" \
  --device cpu --max-game-ms 900000 --connect-timeout-s 900 \
  --out "$OUT" > "$DRIVERLOG" 2>&1 &
echo "   policy pid $!"

say "starting the game on Windows ($MODE)"
ssh -o ConnectTimeout=25 "$WIN" "schtasks /run /tn $TASK" 2>&1 | tail -1

say "waiting for the policy to attach"
for i in $(seq 1 90); do
  if grep -q "attached at" "$DRIVERLOG" 2>/dev/null; then
    echo "   ATTACHED -- game is live"
    break
  fi
  if grep -q "no control channel" "$DRIVERLOG" 2>/dev/null; then
    die "policy gave up connecting -- see $DRIVERLOG"
  fi
  if ! ps -eo cmd --no-headers | grep -q "[p]lay_remote.py"; then
    tail -3 "$DRIVERLOG"; die "policy died"
  fi
  sleep 10
done

grep -q "attached at" "$DRIVERLOG" 2>/dev/null || die "timed out waiting to attach"
sleep 20
echo
echo "decisions so far: $(wc -l < "$OUT" 2>/dev/null || echo 0)"
echo "capture:          $OUT   (feed to lanerl/viz_overlay.py)"
echo "driver log:       $DRIVERLOG"
echo "status/teardown:  lanerl/windows/session.sh --status | --stop"
