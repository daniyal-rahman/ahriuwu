#!/usr/bin/env bash
# Part 2: the pieces part 1 got wrong.
#   - conda-forge has NO `xorg-server`; the Xvfb package is `xorg-xvfb`
#   - conda-forge p7zip is 16.02 (RAR5 support unreliable); the client rar is RAR5,
#     so use RARLAB's official static unrar instead
set -uo pipefail
ROOT=/mnt/storage/lanerl
cd "$ROOT"

echo "=== [1/3] Xvfb (conda-forge xorg-xvfb) ==="
CONDA=/home/dani/miniconda3/bin/conda
if [ ! -x "$ROOT/env/bin/Xvfb" ]; then
    "$CONDA" create -y -p "$ROOT/env" -c conda-forge xorg-xvfb 2>&1 | tail -4
fi
ls "$ROOT/env/bin/" 2>/dev/null | grep -i xvfb
"$ROOT/env/bin/Xvfb" -help 2>&1 | head -2

echo "=== [2/3] static unrar (RAR5-capable) ==="
if [ ! -x "$ROOT/bin/unrar" ]; then
    mkdir -p "$ROOT/bin" && cd "$ROOT"
    curl -sSL -o rar.tgz "https://www.rarlab.com/rar/rarlinux-x64-712.tar.gz"
    tar -xzf rar.tgz && cp rar/unrar "$ROOT/bin/" && rm -rf rar rar.tgz
fi
"$ROOT/bin/unrar" 2>&1 | head -2

echo "=== [3/3] extract the 4.20 client ==="
cd "$ROOT"
if [ ! -d client/extracted/RADS ] && [ -z "$(ls -A client/extracted 2>/dev/null)" ]; then
    mkdir -p client/extracted
    "$ROOT/bin/unrar" x -y client/lol420.rar client/extracted/ 2>&1 | tail -6
fi
echo "--- extracted size:"; du -sh client/extracted 2>/dev/null
echo "--- the game executable:"; find client/extracted -iname "League of Legends.exe" 2>/dev/null | head -3
echo "--- top level:"; ls client/extracted 2>/dev/null | head
