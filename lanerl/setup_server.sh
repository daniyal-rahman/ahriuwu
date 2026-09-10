#!/usr/bin/env bash
# Build the LeagueSandbox GameServer (net6.0) on Linux, user-local dotnet SDK.
# No root, no mono, no wine, no game client -- server only.
# Installs into the NFS vendor dir so both Slurm nodes see the same toolchain
# (/home/dani is per-node, so it cannot live there).
set -euo pipefail

# derived, not literal: the export is /srv/nfs on danilogin and /mnt/nfs on
# desktop, so either literal resolves on exactly one node.
VENDOR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)/lanerl-vendor"
DOTNET_ROOT="$VENDOR/dotnet"
export DOTNET_ROOT
export PATH="$DOTNET_ROOT:$PATH"
export DOTNET_CLI_TELEMETRY_OPTOUT=1
export DOTNET_NOLOGO=1
# NOTE: do NOT override NUGET_PACKAGES. GameServerConsole.csproj has a post-build
# Exec that copies LENet/LeagueSandbox libs out of the DEFAULT ~/.nuget path; an
# override makes that cp fail with MSB3073 (build error, everything else compiles).

echo "=== [1/3] dotnet SDK 6.0 (user-local) ==="
if [ ! -x "$DOTNET_ROOT/dotnet" ]; then
    curl -sSL https://dot.net/v1/dotnet-install.sh -o "$VENDOR/dotnet-install.sh"
    bash "$VENDOR/dotnet-install.sh" --channel 6.0 --install-dir "$DOTNET_ROOT" --no-path
fi
"$DOTNET_ROOT/dotnet" --version

echo "=== [1b/3] submodules (LeaguePackets + Content) ==="
cd "$VENDOR/GameServer"
git submodule update --init --recursive --depth 1 2>&1 | tail -5
du -sh LeaguePackets Content/LeagueSandbox-Default 2>/dev/null

echo "=== [2/3] restore ==="
cd "$VENDOR/GameServer"
"$DOTNET_ROOT/dotnet" restore 2>&1 | tail -5

echo "=== [3/3] build ==="
"$DOTNET_ROOT/dotnet" build --no-restore -c Release 2>&1 | tail -15

echo "=== artifacts ==="
find "$VENDOR/GameServer" -name "GameServerConsole*" -path "*Release*" 2>/dev/null | head
