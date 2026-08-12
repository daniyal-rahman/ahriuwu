# ahriuwu — Desktop Inference Handoff

**Everything for running the live Garen agent lives on THIS machine's local disk.**
No NFS, no login node needed once you're here. Last validated: 2026-08-09.

---

## TL;DR — run a live session (3 steps)

```bash
# 1. (on the Pi 192.168.1.144) start the keyboard server  [your existing rig]
# 2. (on Windows) start the game + run your ffmpeg stream to THIS box:
#      ...  -f mpegts udp://192.168.1.100:5000?pkt_size=1316
# 3. (here, desktop Linux — GPU must be free) :
cd /mnt/storage/ahriuwu-live
INJECT=hid ./play.sh --source udp
```
That captures the Windows stream, runs the model (~17 fps measured), decodes to
mouse/keys, sends them to the Pi, and RECORDS everything (see Recording below).

Dry test first (no keys sent, just prints + records):
```bash
INJECT=dry ./play.sh --source udp
```

---

## Architecture (2-machine)

```
Windows PC:  League + ffmpeg gdigrab -> H.264 zerolatency -> mpegts/UDP
                                   |  udp://192.168.1.100:5000
                                   v
Linux desktop (THIS box, 5080):  StreamCapture -> tokenizer -> world model
                                   -> WASD + casts  --TCP :9999-->  Pi (192.168.1.144)
                                                                     |
Pi HID keyboard  --USB-->  Windows PC  (presses keys in the game)  <-+
```
- **This box = the brain.** Model runs on the 5080. It ONLY receives the video
  stream and emits keystrokes; it does not run League.
- **Desktop LAN IP = 192.168.1.100** — this is the address your Windows ffmpeg
  sends to (`udp://192.168.1.100:5000`).
- **Pi = the hands**, at 192.168.1.144:9999, speaks `press <key>` / `release <key>`.

---

## Where everything is  (`/mnt/storage/ahriuwu-live`, local HDD)

| path | what |
|---|---|
| `env.sh` | sets `$PY $TOK $BC $AHRIUWU` — `source` it for manual commands |
| `play.sh` | the launcher (wraps play_live.py with the local ckpts) |
| `preflight.sh` | pre-game checks (capture/agent/fps/HID/API) |
| `loopback_test.sh` | self-contained stream->model test (no Windows needed) |
| `checkpoints/tokenizer_v7.pt` | frozen v7 tokenizer (512x16 -> 16x16x32) |
| `checkpoints/phase2_bc.pt` | the agent (gated action-model, gs80930) |
| `scripts/` | the code (play_live.py, agent_infer.py, keysender/, src via PYTHONPATH) |
| `recordings/session_*/` | one dir per run (see Recording) |
| `src/` (under scripts PYTHONPATH) | ahriuwu library (models, live/, vision/) |

Env: conda `ml` at `/home/dani/miniconda3/envs/ml` (this box's local home). Has
torch+cu, cv2, mss. `ffmpeg` is system `/usr/bin/ffmpeg`.

---

## Keybinds — SET THESE IN-GAME

> NOTE (2026-08-10): movement default is now **mouse right-click-to-move** (matches
> training: the policy outputs a click TARGET). The WASD table below applies only
> to `--movement wasd`. In mouse mode LoL STOCK binds are used (Q=q W=w E=e R=r).
> Also: on this client **F is bound to GHOST, not Ignite** — fix or the agent's
> 'Ignite' fires the wrong summoner.

Movement is **WASD**, so abilities were moved OFF w/a/s/d. Bind in the LoL client:

| action | key | | action | key |
|---|---|---|---|---|
| move | W A S D | | R | r |
| Q | q | | Flash | g |
| **W (spell)** | **v** | | Ignite | f |
| E | e | | Recall | b |

- Garen W is normally on `w` → moved to **v** (w is walk-up now). The startup
  guard REFUSES to run live if any ability key still collides with WASD.
- AA: keyboard-only can't right-click a target. Default relies on the mode's
  auto-attack. If your mode has an attack-move key, pass `--attack-key <key>`.
- Camera must be **locked/centered on champion** — the WASD decode assumes Garen
  is at screen-center. (Unlock breaks direction; tell the agent-manager if you
  play unlocked and it'll switch to a pixel-based champion locator.)

---

## Recording (on by default)

Every run writes `recordings/session_<timestamp>/`:
- **`model_view_352.mp4`** — the exact 352x352 frames the model saw (post-
  transform, NOT the raw 720p). This is what to watch when debugging behavior.
- **`actions.jsonl`** — one line per frame: `{i, t, movement:[x,y], wasd:[...],
  casts:[...], aa, reward_pred, ms:{cap,enc,act,inj}}`.
- **`meta.json`** — checkpoints, args, keybinds, model flags.

Disable with `--no-record`. Change location with `--record-dir <path>`.

---

## Verify without Windows (loopback)

```bash
bash /mnt/storage/ahriuwu-live/loopback_test.sh   # runs a test stream -> model
tail -f /tmp/pl.log                               # watch fps + actions
```
Expected: `~17 fps` when the GPU is free, WASD output varying per frame, a new
`recordings/session_*` dir. (Content is a test pattern, so actions are
meaningless — this only proves the plumbing.)

---

## Performance notes

- **Full loop on a FREE GPU: ~17.4 fps** (cap 2 / enc 11 / act 44 / inj 0 ms) — BELOW
  the 20 fps target. The world-model forward (`act`) alone eats 44 ms of the 50 ms
  budget. (An earlier '26 fps' figure in this doc was WRONG: that benchmark fed
  pre-made latents and skipped the tokenizer encode.) Free the GPU before playing;
  under other CUDA load it collapses further (7.8 fps measured during BC training).
- First frame is slow (~300 ms, one-time CUDA/kernel warmup); steady state ~40 ms.
- Latency budget printed each second: `[cap N enc N act N inj N ms]`. `act`
  (world-model forward) dominates. If total >50 ms, something else is on the GPU.

---

## Troubleshooting

- **"no stream frame within 30s"** → Windows ffmpeg isn't reaching here. Check the
  Windows cmd targets `udp://192.168.1.100:5000`, both boxes on 192.168.1.x, no
  firewall on UDP 5000. Test locally with `loopback_test.sh`.
- **hangs after "connected"/no keys move** → Pi server not running, or wrong
  `--hid-host`. Default is 192.168.1.144. `INJECT=dry` to isolate (model-only).
- **"ability keys collide with WASD"** → a bind is still on w/a/s/d; fix
  `DEFAULT_KEYS` in `scripts/play_live.py` and your in-game binds.
- **low fps** → something else on the GPU (`nvidia-smi`); kill it.
- ssh sessions that run play_live directly die on disconnect (SIGHUP). For long
  runs use `setsid nohup ... &` or a tmux, and tail the log.

---

## Current model state / caveats (be honest with expectations)

- The agent is the **gated action-model checkpoint, mid-training (~epoch 0.8)**.
  It is playable for a plumbing/behavior test but NOT final — it'll improve with
  more epochs. The gate had just started discriminating (fires more on real
  movement changes than on holds).
- Perception: the tokenizer does NOT reliably encode HP/gold (a separate CV
  HP-bar reader exists in `src/ahriuwu/vision/` and a live-client state API in
  `src/ahriuwu/live/` — neither is wired into the live loop yet).
- This is behavior-cloning only (imitates Masters+ replays). No RL/imagination
  in the live policy yet.
- First real-game unknowns (not yet tested): whether Vanguard tolerates the Pi
  HID keyboard, and end-to-end stream latency over your LAN.

---

## Handing off to the desktop-inference agent

A dedicated agent now manages this box's inference. It can: run preflight +
loopback checks, launch/stop sessions, watch fps + recordings, and analyze the
`actions.jsonl` / `model_view_352.mp4` after a game to debug behavior. Point it
at a session dir and ask for an analysis, or tell it to start/stop a run.
