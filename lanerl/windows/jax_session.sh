#!/usr/bin/env bash
# Publish, deploy and run the bounded JAX -> real-client smoke test.
#
#   lanerl/windows/jax_session.sh
#   lanerl/windows/jax_session.sh --status
#   lanerl/windows/jax_session.sh --stop
#   lanerl/windows/jax_session.sh --prepare-only
set -uo pipefail
cd "$(dirname "$0")/../.."
REPO=$(pwd)
VENDOR=/mnt/nfs/projects/lanerl-vendor
SERVER="$VENDOR/LoLServer-emit"
DOTNET="$VENDOR/dotnet/dotnet"
PY="$VENDOR/../ahriuwu-lanerl-jax/.venv-jax/bin/python"
WIN=windows
PORT=5202
TASK=lanerl_play_jax
DURATION_MS=20000
SKIP_BUILD=0
ACTION=start
LOG=lanerl/logs/play_jax.out

while [ $# -gt 0 ]; do
  case "$1" in
    --skip-build) SKIP_BUILD=1; shift;;
    --duration-ms) DURATION_MS="$2"; shift 2;;
    --prepare-only) ACTION=prepare; shift;;
    --status) ACTION=status; shift;;
    --stop) ACTION=stop; shift;;
    -h|--help) sed -n '2,12p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

say() { printf '\n== %s\n' "$*"; }
die() { printf '\nFAILED: %s\n' "$*" >&2; exit 1; }

stop_local() {
  for p in $(ps -eo pid,cmd --no-headers | grep "[l]anerl_jax.emit.runtime" | awk '{print $1}'); do
    kill "$p" 2>/dev/null || true
  done
  for p in $(ps -eo pid,cmd --no-headers | grep "[s]sh -N" | grep "L $PORT:" | awk '{print $1}'); do
    kill "$p" 2>/dev/null || true
  done
}

stop_everything() {
  say "stopping JAX smoke"
  stop_local
  ssh -o ConnectTimeout=20 "$WIN" "schtasks /end /tn $TASK" >/dev/null 2>&1 || true
  ssh -o ConnectTimeout=20 "$WIN" "powershell -NoProfile -Command \"foreach (\\\$n in @('League of Legends','GameServerConsole')) { Get-Process \\\$n -EA SilentlyContinue | Stop-Process -Force -EA SilentlyContinue }\"" >/dev/null 2>&1 || true
}

if [ "$ACTION" = stop ]; then stop_everything; echo stopped.; exit 0; fi
if [ "$ACTION" = status ]; then
  echo "tunnel:"; ss -tlnp 2>/dev/null | grep ":$PORT " || echo "  not bound"
  echo "driver:"; ps -eo pid,etime,cmd --no-headers | grep "[l]anerl_jax.emit.runtime" || echo "  not running"
  [ -f "$LOG" ] && tail -8 "$LOG"
  ssh -o ConnectTimeout=20 "$WIN" "powershell -NoProfile -Command \"'server=' + @(Get-Process GameServerConsole -EA SilentlyContinue).Count + ' client=' + @(Get-Process 'League of Legends' -EA SilentlyContinue).Count; Get-Content C:\\lanerl\\play_jax.log -Tail 5 -EA SilentlyContinue\"" 2>/dev/null || true
  exit 0
fi

[ -x "$DOTNET" ] || die "missing dotnet: $DOTNET"
[ -x "$PY" ] || die "missing JAX Python: $PY"
[ -e "$SERVER/.git" ] || die "missing isolated server: $SERVER"

say "installing the recorder and JAX relay patches idempotently"
python3 lanerl/patch_packet_recording.py --server "$SERVER" || die "recorder patch failed"
python3 lanerl/patch_jax_relay.py --server "$SERVER" || die "relay patch failed"

if [ "$SKIP_BUILD" = 0 ]; then
  say "publishing patched server for Windows"
  ( cd "$SERVER" && DOTNET_CLI_TELEMETRY_OPTOUT=1 "$DOTNET" publish \
      GameServerConsole/GameServerConsole.csproj -c Release -r win-x64 --self-contained \
      -p:SolutionDir="$SERVER/" -o "$VENDOR/winpub_jax" ) \
      >/tmp/lanerl_jax_publish.log 2>&1 \
    || { tail -20 /tmp/lanerl_jax_publish.log; die "Windows publish failed"; }
fi
test -f "$VENDOR/winpub_jax/GameServerConsole.exe" || die "no published Windows executable"

if [ "$ACTION" = prepare ]; then
  echo "prepared: $VENDOR/winpub_jax/GameServerConsole.exe"
  exit 0
fi

stop_everything
say "deploying server and launchers to Windows"
scp -q -o ConnectTimeout=25 -r "$VENDOR"/winpub_jax/* "$WIN":'C:/lanerl/winpub_jax/' \
  || die "server deploy failed"
scp -q -o ConnectTimeout=25 lanerl/windows/play_jax.ps1 lanerl/windows/play_jax.bat \
  "$WIN":'C:/lanerl/' || die "launcher deploy failed"
ssh -o ConnectTimeout=25 "$WIN" "schtasks /create /tn $TASK /tr \"C:\\lanerl\\play_jax.bat\" /sc once /st 00:00 /ru daniz /it /f" >/dev/null \
  || die "could not create scheduled task"

say "opening and proving SSH relay tunnel on $PORT"
nohup ssh -N -o ConnectTimeout=25 -o ServerAliveInterval=20 -o ExitOnForwardFailure=yes \
  -L "$PORT:127.0.0.1:$PORT" "$WIN" >/tmp/lanerl_jax_tunnel.log 2>&1 &
for _i in $(seq 1 15); do
  ss -tlnp 2>/dev/null | grep -q ":$PORT " && break
  sleep 1
done
ss -tlnp 2>/dev/null | grep -q ":$PORT " \
  || { cat /tmp/lanerl_jax_tunnel.log; die "tunnel never bound"; }

say "starting real server/client on Windows"
ssh -o ConnectTimeout=25 "$WIN" "schtasks /run /tn $TASK" | tail -1 \
  || die "could not start Windows scheduled task"

say "waiting for the remote relay listener (without connecting a probe client)"
for _i in $(seq 1 60); do
  ssh -o ConnectTimeout=10 "$WIN" "powershell -NoProfile -Command \"if (Select-String C:\\lanerl\\server_jax.log -Pattern 'LANERL_JAX_RELAY listening' -Quiet -EA SilentlyContinue) { exit 0 } else { exit 1 }\"" \
    >/dev/null 2>&1 && break
  sleep 2
done
ssh -o ConnectTimeout=10 "$WIN" "powershell -NoProfile -Command \"if (Select-String C:\\lanerl\\server_jax.log -Pattern 'LANERL_JAX_RELAY listening' -Quiet -EA SilentlyContinue) { exit 0 } else { exit 1 }\"" \
  >/dev/null 2>&1 || die "Windows server never opened the JAX relay"

say "preparing the JAX sim (it connects only after every executable is compiled)"
mkdir -p lanerl/logs
rm -f "$LOG"
LANERL_VENDOR_ROOT="$VENDOR" JAX_PLATFORMS=cpu PYTHONPATH="$REPO" \
  nohup "$PY" -u -m lanerl_jax.emit.runtime --port "$PORT" \
    --takeover-ms 135000 --duration-ms "$DURATION_MS" >"$LOG" 2>&1 &
DRIVER_PID=$!
echo "driver pid $DRIVER_PID"

say "waiting for JAX ownership"
for _i in $(seq 1 90); do
  grep -q '^ACTIVE ' "$LOG" 2>/dev/null && break
  kill -0 "$DRIVER_PID" 2>/dev/null || { tail -20 "$LOG"; die "JAX driver died"; }
  sleep 5
done
grep -q '^ACTIVE ' "$LOG" 2>/dev/null || { tail -20 "$LOG"; die "takeover timed out"; }
echo "JAX OWNS THE LIVE CLIENT"
echo "driver log: $LOG"
echo "Windows log: C:\\lanerl\\play_jax.log"
echo "status/teardown: lanerl/windows/jax_session.sh --status | --stop"
