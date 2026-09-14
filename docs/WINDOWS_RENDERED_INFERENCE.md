# Watching the agent play: rendered inference on native Windows

How to put the trained policy on screen in the real 4.20 client, record it, and
overlay what the network was thinking.

This has now been reconstructed from scratch three times. It is written down
because every step below is something that cost hours to find and that fails
*silently* or with a misleading error if you get it wrong.

## The shape of it

```
  Windows desktop (dual boot -- the Linux side is DOWN while this runs)
  ├── GameServerConsole.exe      the vendored LoLServer, LANERL_HEADLESS=1
  │     ├── game port 5119       <- the League client connects here
  │     └── LANERL_CONTROL_PORT  <- the policy connects here
  ├── League of Legends.exe      renders, connects to 127.0.0.1
  └── screen recording
```

The policy can run either on Windows (needs torch there) or on **danilogin**
over Tailscale, talking to the control port. The client MUST talk to
`127.0.0.1`.

## Non-obvious requirements

Each of these was a multi-hour failure the first time.

**Run the server ON WINDOWS.** `danilogin` drops inbound LAN UDP, so a
Linux-hosted server gives the client a bare "Failed to Connect" (verified with
an `nc -u` listener: zero packets arrive). Loopback sidesteps it entirely.

**`LANERL_HEADLESS=1` even WITH a client attached.** Otherwise the server
counts "2/2 disconnected" and quits `forcedStart` seconds after ready, killing
the run while the client is still on its connect-retry dialog.

**Set env vars inside a `.bat`, and launch the client from a `.bat`.**
PowerShell's `Start-Process -ArgumentList` DROPS the empty-string argument,
which shifts the client's argv and it never connects. The argv is positional
and the empty string is load-bearing:

```bat
start "" "League of Legends.exe" "8394" "LoLLauncher.exe" "" "127.0.0.1 5119 17BLOhi6KZsTtldTsizvHg== 1"
```

**ssh lands in Windows session 0, which cannot render or screen-capture.**
Bridge to the interactive desktop with a scheduled task:

```bat
schtasks /create /tn lanerl_play /tr "C:\lanerl\play.bat" /sc once /st 00:00 /ru daniz /it /f
schtasks /run /tn lanerl_play
```

**`LockCamera=1` must be in the USER config**, `client\Config\Game.cfg` -- not
just `deploy\DATA\cfg\defaults\Game.cfg`. The user config overrides, and the
key may be absent rather than 0. **The overlay depends on this**:
`lanerl_rl.projection` models a champion-centred locked camera, so a free
camera offsets every annotation and it reads as bad aim rather than a bad
assumption.

**Publish the server self-contained, with `SolutionDir`.** Windows needs no
dotnet installed. `SolutionDir` is required or a post-build `cp` resolves to
`*Undefined*lib\.` and fails *after* a clean compile:

```bash
cd /srv/nfs/projects/lanerl-vendor/LoLServer
../dotnet/dotnet publish GameServerConsole/GameServerConsole.csproj \
  -c Release -r win-x64 --self-contained \
  -p:SolutionDir=/srv/nfs/projects/lanerl-vendor/LoLServer/ \
  -o /srv/nfs/projects/lanerl-vendor/winpub_rl
```

This cross-publishes fine from Linux, so it can be built on `danilogin` while
the desktop is on Windows.

**Rebuild before every session.** `C:\lanerl\winpub` is whatever was published
the last time someone did this. A stale server still *runs* -- it just omits
wire fields the policy was trained on (`ad`, `sl`, `mt` were added 2026-09-12/13),
so `_aa_damage` falls to 0, spell ranks get guessed, and minion subtypes vanish.
The agent then behaves unlike the thing you measured, and nothing errors.

## What is on the Windows box

```
C:\lanerl\
  winpub\            published server (REBUILD THIS)
  gameinfo.json      match config
  client\            the 4.20 client, RADS tree
  srv.bat            LANERL_HEADLESS=1 + GameServerConsole.exe --config gameinfo.json
  launch_client.bat  taskkill, cd to deploy\, start with the argv above
  cam.bat            camera lock
  record.bat         screen capture
```

## Recording and the overlay

Capture and render are deliberately separate -- the recording happens on
Windows and the overlay renders on Linux, so they are never in the same place
at once.

1. `lanerl/viz_capture.py --checkpoint <ckpt> --port-base <p>` alongside the
   game, writing one JSONL row per decision: champion and every actionable
   entity in **world and screen** coordinates, all four head distributions,
   the wire order, and the critic value. Run it **without `--freerun`** -- the
   sim must advance at wall-clock rate or the video and the JSONL cannot be
   aligned.
2. Record the screen. Note the in-game clock at the first video frame.
3. On Linux: `lanerl/viz_overlay.py --capture <jsonl> --t0-ms <that clock>`,
   which writes RGBA frames plus the exact ffmpeg command to composite them.

## Known dead end

The client under Linux/wine. Do not spend time there without a new client
source: the MediaFire copy never shipped the `lol_game_client` RADS project,
and renderer init stalls at `r3dRenderLayer::SetMode: Initializing` under every
renderer and display combination. That is an asset problem, not an environment
one. Everything else on that path (portable wine, `lanerl/rlsm.py` RADS
manifest round-tripping, DXVK on the 5080) does work and is reusable.
