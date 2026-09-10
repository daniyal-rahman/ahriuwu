#!/usr/bin/env bash
# Stand up the pieces needed to RENDER the 4.20 client on the desktop node, no root:
#   portable wine (wow64 -> runs the 32-bit client without i386 host libs)
#   Xvfb via conda (no X session exists on this box)
#   the 4.20 client itself (2.41 GB rar)
# DXVK/wine will render on the 5080 via nvidia_icd and present into the virtual display,
# which ffmpeg (already installed) can then grab.
set -uo pipefail

ROOT=/mnt/storage/lanerl
mkdir -p "$ROOT"/{wine,client,env}
cd "$ROOT"

echo "=== [1/4] portable wine (wow64) ==="
WINE_TAR=wine-11.16-amd64-wow64.tar.xz
if [ ! -x "$ROOT/wine/bin/wine" ]; then
    curl -sSL -o "$WINE_TAR" \
      "https://github.com/Kron4ek/Wine-Builds/releases/download/11.16/$WINE_TAR" || echo "wine dl FAILED"
    tar -xf "$WINE_TAR" -C wine --strip-components=1 && rm -f "$WINE_TAR"
fi
"$ROOT/wine/bin/wine" --version 2>&1 | head -2

echo "=== [2/4] Xvfb + extraction tools via conda ==="
CONDA=/home/dani/miniconda3/bin/conda
if [ ! -x "$ROOT/env/bin/Xvfb" ]; then
    "$CONDA" create -y -p "$ROOT/env" -c conda-forge xorg-server p7zip 2>&1 | tail -3
fi
"$ROOT/env/bin/Xvfb" -help 2>&1 | head -1
ls "$ROOT/env/bin/" | grep -iE "xvfb|7z" | head

echo "=== [3/4] 4.20 client (2.41 GB) ==="
if [ ! -f client/lol420.rar ]; then
    # the direct link is single-use/expiring, so re-scrape it at download time
    curl -sL "https://www.mediafire.com/file/rm32t1nbbca6zph/League-of-Legends-4.20.rar/file" -o mf.html
    DL=$(grep -oE 'https://download[0-9]*\.mediafire\.com/[^"'"'"']*' mf.html | head -1)
    echo "resolved: ${DL:0:80}..."
    curl -sL --retry 3 -o client/lol420.rar "$DL"
fi
ls -lh client/lol420.rar 2>/dev/null

echo "=== [4/4] extract ==="
if [ ! -d client/extracted ]; then
    mkdir -p client/extracted
    "$ROOT/env/bin/7z" x -y -o"client/extracted" client/lol420.rar 2>&1 | tail -5
fi
find client/extracted -maxdepth 2 -iname "*.exe" 2>/dev/null | head -10
du -sh client/extracted 2>/dev/null
