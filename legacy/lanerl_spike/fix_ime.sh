#!/usr/bin/env bash
# Surgical version. The 9296-symlink sweep made the manifest walk explode
# (the generator follows symlinks, so every case variant multiplied the tree).
# In practice only ONE path is ever refused: Data/Menu/IMEConfig.xml -- disk has
# DATA/menu. So drop the sweep, add just that one alias, and rebuild.
set -uo pipefail
ROOT=/mnt/storage/lanerl; NFS=/mnt/nfs/projects; OUT=$NFS/ahriuwu-lanerl/lanerl/logs
D=$ROOT/client/extracted/RADS/solutions/lol_game_client_sln/releases/0.0.1.68/deploy
PROJ=$ROOT/client/extracted/RADS/projects

echo "removing the case-variant sweep..."
python3 - <<'PY'
import os
D="/mnt/storage/lanerl/client/extracted/RADS/solutions/lol_game_client_sln/releases/0.0.1.68/deploy/DATA"
n=0
for root, dirs, files in os.walk(D, followlinks=False):
    for name in list(dirs):
        p=os.path.join(root,name)
        if os.path.islink(p):
            os.unlink(p); n+=1
print(f"removed {n} symlinks")
PY
ln -sfn menu "$D/DATA/Menu"
echo "aliased DATA/Menu -> menu"

python3 "$NFS/ahriuwu-lanerl/lanerl/rlsm.py" build "$D" 1 263 "$PROJ/lol_game_client/releases/0.0.1.7/releasemanifest"
for q in lol_game_client_en_gb/releases/0.0.0.235 LEVELS/releases/0.0.0.235 \
         LEVELS/releases/0.0.1.7 DATA/releases/0.0.0.235 DATA/releases/0.0.1.7; do
    cp -f "$PROJ/lol_game_client/releases/0.0.1.7/releasemanifest" "$PROJ/$q/releasemanifest"
done

export WINEPREFIX=$ROOT/prefix DISPLAY=:1 WINEDLLOVERRIDES="mscoree,mshtml=;d3d9=n"
export DXVK_STATE_CACHE_PATH=$ROOT/dxvk_cache DOTNET_ROOT=$NFS/lanerl-vendor/dotnet
export LANERL_BOT=blue   # was the server-side default until 2026-09-10;
                         # DriveTeams now defaults to "none", so stating it
                         # here keeps this script doing what it always did
W=$ROOT/wine/bin/wine
$ROOT/wine/bin/wineserver -k 2>/dev/null; sleep 2
cd $NFS/lanerl-vendor/LoLServer/GameServerConsole/bin/Release/net6.0
$DOTNET_ROOT/dotnet ./GameServerConsole.dll --config $NFS/ahriuwu-lanerl/lanerl/cfg/garen1v1.json >"$OUT/server_i.log" 2>&1 &
for i in $(seq 1 40); do grep -q "Game is ready" "$OUT/server_i.log" 2>/dev/null && break; sleep 2; done
cd "$D"
$W "League of Legends.exe" "8394" "LoLLauncher.exe" "" "127.0.0.1 5119 17BLOhi6KZsTtldTsizvHg== 1" >"$OUT/client_i.log" 2>&1 &
for i in $(seq 1 45); do grep -q "Accepted client version" "$OUT/server_i.log" 2>/dev/null && { echo "CONNECTED"; break; }; sleep 4; done
for t in 1 2 3 4 5 6; do
    sleep 60
    ffmpeg -loglevel error -y -f x11grab -video_size 2560x1440 -i :1 -frames:v 1 "$OUT/shots/i_$t.png" >/dev/null 2>&1
    echo "  ${t}m procs=$(pgrep -fc 'League of Legends' || echo 0) png=$(stat -c%s "$OUT/shots/i_$t.png" 2>/dev/null)B notinmanifest=$(grep -c 'not in the manifest' "$OUT/client_i.log")"
done
echo "--- trace:"; grep "League of Legends.exe" "$OUT/client_i.log" | tail -6
