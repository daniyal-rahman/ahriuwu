# DEMO_RUNBOOK — running the Garen agent on real hardware

Three machines. Commands are labelled with the one they run on.

```
  WINDOWS PC ──gdigrab→x264→mpegts/UDP:5000──▶ DESKTOP (RTX 5080)
      ▲                                            │
      │                                            │ play_live.py
      │                                            ▼
      └── real USB keyboard+mouse ── PI (USB-HID gadget) ◀── TCP :9999
```

The Windows PC runs the game and is the HID *host*. The desktop runs the model.
The Pi is a USB gadget plugged into the Windows PC, and is what makes the input
indistinguishable from a real keyboard and mouse — Vanguard rejects synthetic
input, so `--inject pynput` cannot play a real game and exists only for desktop
plumbing tests.

---

## 0. What you are demoing, honestly

Read this before you start so nothing is a surprise mid-game.

- **The model has never seen a HUD.** The training corpus is rendered with the
  HUD disabled (measured: no black band; bottom-25% brightness 0.205 vs 0.215
  whole-frame). A live game has one, so 20–30% of every frame you feed it is
  content the tokenizer and the policy have never encountered. Nothing in the
  live path masks it. **This is a known, unfixed risk, and it is the single
  biggest reason the agent may behave worse live than it does offline.** There
  is no fix available in the time we have; `sim_replay.py --hud add` exists to
  quantify it offline and has not been wired into the live path.
- **20 fps is unreachable.** 81.4 ms model-only on an idle 5080 (resize 1.9 +
  tokenizer 22.3 + dynamics/heads 59.0). The rig has measured 17.0 fps. The loop
  is fine down to ~10 fps. `--target-fps 20` is a ceiling, not a promise.
- **Temperature 1.0 is mandatory.** Greedy decode is a measured dead policy on
  every checkpoint on disk: 0.00 clicks/s, one movement cell, zero casts. Every
  command below passes `--temperature 1.0` and `play.sh` hardcodes it.
- **The checkpoint was picked on liveness, not skill.** See §1a. There is no
  ground-truth score here: the measurements say the policy *acts*, not that it
  acts *well*.
- **Casting is rare.** The sparse-cast fix (`--ability-pos-weight 5.0`) was
  overridden to 1.0 in every launcher that produced the checkpoints on disk.
  Measured press rates in training: Q 3.6e-3, AA 8.4e-3, R 3e-5. Expect the
  agent to move far more than it casts. That is a training artefact, not a live
  bug.

---

## 1a. Which checkpoint, and why

Measured offline on **real recorded latents at `--temperature 1.0`**: 3 replays
x 600 frames x 2 seeds = 3600 frames per candidate, identical inputs across all
three. `ab_checkpoints.py`'s `cell_acc` was **not** used — it is broken (scores
MTP offset 1 against frame *f*'s target, and its denominator sits inside the
gate).

| ckpt | clicks/s | uniq cells | top-cell frac | non-AA casts | mean target | step |
|---|---|---|---|---|---|---|
| **`data/phase2_bc_clicks/…latest.pt`** | **2.70** | **60.0** | **0.057** | **1.36%** | (0.498, 0.492) | 102420 |
| `data/phase2_from_vast/vast_step90000.pt` | 2.45 | 49.3 | 0.060 | 0.67% | (0.505, 0.479) | 100329 |
| `data/phase2_parity/…latest.pt` | 2.06 | 45.7 | 0.071 | 1.08% | (0.581, 0.397) skewed | 55216 |

All three are alive — none collapses to one cell, none is silent. `bc_clicks`
wins on every liveness axis at once, so it is the deployed default.

**Read the caveats.**
- These are **liveness** metrics. They cannot distinguish competence from noise,
  and a policy that acts more can score higher simply by being noisier.
- `bc_clicks` is the **frozen** lineage (`axis` head + gate), whose
  `action_embed` was fitted to a *different* movement target and cannot adapt
  (WIRING_AUDIT 1.1). The preflight prints `** FROZEN backbone lineage **` for
  it. That is expected, not an error.
- Sample is ~3.5 min of gameplay per checkpoint, all from **early-game** windows
  (~frame 3000). Between-replay variance is large (clicks/s ranged 1.53–3.26),
  so the 2.70-vs-2.45 gap is inside the noise on any single replay — `bc_clicks`
  earns the pick by matching or beating the others on *every* file and *both*
  seeds. R and Flash never fire in any candidate, almost certainly because they
  are not yet available that early; do not read that as "R is broken".

**Fallback if it looks erratic on the day** — the unfrozen `joint_noop` one,
also healthy:
```bash
BC_SRC=/mnt/nfs/projects/ahriuwu/data/phase2_from_vast/vast_step90000.pt \
  bash /mnt/nfs/projects/ahriuwu/ops/stage_desktop_standalone.sh
```
Do not use `phase2_parity` — fewest steps, fewest cells, off-centre bias.

Tokenizer: `rollout_stage/transformer_tokenizer_latest.pt`, sha256
`35154dca2ad0c786…`, step 6000, `latent_dim=16 x num_latents=512` folding to the
`(1, 32, 16, 16)` dynamics latent all three checkpoints expect.

---

## 1. ONE TIME — deploy to the desktop

The deployed rig at `/mnt/storage/ahriuwu-live/` was, until now, an unmanaged
fork of the repo: 162 diff lines in `play_live.py`, no `joint_noop` branch, no
`--gate-bias`, and a checkpoint the repo documents as invalid. Re-stage it so
what runs is what is in git.

**[desktop]**
```bash
ssh desktop
bash /mnt/nfs/projects/ahriuwu/ops/stage_desktop_standalone.sh
```

Healthy output ends with:
```
[stage] VERSION:
[stage]   commit=902d8ed...
[stage]   dirty=no
[stage]   phase2_sha256_16=...
VERIFY OK: use_actions=True movement_mode=axis gated=True move=(0.5, 0.48) bf16=True
[stage] DONE. Standalone tree at /mnt/storage/ahriuwu-live, stamped 902d8ed.
```

- `dirty=yes` means the NFS repo had uncommitted changes when you staged. The
  run is then not reproducible; commit first if you care.
- The script **preserves** `scripts/keysender/mouse_calibration.json` on the
  desktop across re-stages. That file is the only artefact measured against your
  actual screen.
- To deploy a different checkpoint: `BC_SRC=/path/to.pt bash .../stage_desktop_standalone.sh`

---

## 2. ONE TIME — the Pi's HID gadget

The gadget must be **relative**-mouse. The repo's setup script used to build an
absolute one; a relative sender writing 4-byte reports into a 6-byte absolute
gadget is a silent no-op — the agent "plays" and never aims.

**[pi]**
```bash
sudo MOUSE_MODE=rel /path/to/scripts/keysender/setup_hid_combo.sh
python3 /path/to/scripts/keysender/hid_server.py          # leave running
```

Healthy:
```
HID combo gadget up on UDC ...: /dev/hidg0 (keyboard), /dev/hidg1 (rel mouse, 4-byte reports)
mouse mode: rel (probed from report_desc)
mouse gadget: /dev/hidg1 (relative)
hid_server: kb=/dev/hidg0 mouse=/dev/hidg1 rel, listening :9999
```

If you see `mouse gadget: unavailable` you are about to play with no mouse at
all — every click is dropped. If you see `mouse mode: abs`, re-run the setup
script with `MOUSE_MODE=rel`.

> The Pi's copy of `scripts/keysender/` must come from this commit. An older
> `hid_server.py` has no `status` verb and the preflight will say so explicitly.

---

## 3. Windows side — game settings and the stream

### 3a. Game settings (these matter)

| Setting | Value | Why |
|---|---|---|
| Resolution | 1280x720 windowed/borderless | matches the stream geometry |
| **Camera lock** | **ON** (`Y`) | the mouse slams the top-left corner before every command; with edge-pan active the camera would lurch each time |
| Edge pan / screen scroll | OFF | same reason |
| Enhance pointer precision | **OFF** (Windows mouse settings) | travel then scales linearly with report size and the calibration is exact. If it is ON the calibration still works but is only approximate |
| Ability binds | Q=`q` W=`v` E=`e` R=`r` Flash=`g` Ignite=`f` Recall=`b` Stride=`3` | must match `DEFAULT_KEYS` in `play_live.py` |
| Practice tool / custom game | yes | do not do this in a real game with other players |

### 3b. The stream

**[windows]**
```
ffmpeg -f gdigrab -framerate 20 -offset_x 0 -offset_y 0 -video_size 1280x720 -i desktop ^
  -c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p -g 20 -b:v 6M ^
  -f mpegts udp://<DESKTOP_IP>:5000?pkt_size=1316
```

- `-framerate 20` is the one that matters. The first live session died because
  the stream delivered **2–3 new frames per second** into a 17 fps loop, so 58%
  of consecutive frames were byte-identical.
- If ffmpeg reports `frame= ... fps=3`, the encoder is CPU-starved. Drop to
  `-framerate 15`, keep `-preset ultrafast`, and close anything else heavy.
- Capture the game's region, not a scaled 4K desktop. Scaling costs more CPU
  than the encode.
- `-video_size` must equal the `--stream-size` the desktop expects (1280x720).

---

## 4. ONE TIME per rig — calibrate the mouse

Only needed if the Windows pointer speed, "Enhance pointer precision", or the
resolution has changed since the span was last measured. A provisional
calibration (649 x 367 units, chunk 30) ships in the repo, and the preflight
tells you it is provisional.

Requires the stream to be flowing (it reads the cursor back off the video) and
the game to be focused. It never clicks, so it is safe mid-game.

**[desktop]**
```bash
source /mnt/storage/ahriuwu-live/env.sh
$PY $AHRIUWU/scripts/keysender/calibrate_mouse.py --host $PI --udp-port 5000
```

Healthy tail:
```
x: units/px=0.5070  SPAN=649.0  nonlinearity=2.1%  (n=5)
y: units/px=0.5097  SPAN=367.0  nonlinearity=3.4%  (n=5)
MOUSE_SPAN = (649.0, 367.0)
saved -> .../scripts/keysender/mouse_calibration.json
```

- `nonlinearity > 8%` → "Enhance pointer precision" is on. Turn it off and
  re-run for an exact mapping; the run is still usable if you don't.
- `no travel detected (false positive?)` on most rows → the cursor template is
  not matching. Make sure the game is focused and the cursor is in the game
  world, not over the HUD or the shop panel.
- `NOT ENOUGH GOOD SAMPLES` → nothing is written; the previous calibration
  stands.

**Do not re-calibrate mid-game.** The addressing is corner-relative, so errors
do not accumulate and there is nothing to correct for.

---

## 5. Every session — preflight

This is the gate. It exits non-zero and refuses to bless a broken rig.

**[desktop]**
```bash
/mnt/storage/ahriuwu-live/preflight.sh --inject hid --movement-mode mouse --source udp
```

Healthy:
```
=== play_live preflight ===
  [OK]   provenance: commit 902d8ed... via VERSION (staged)
  [OK]   checkpoint identity: step=102420 movement_mode=axis gate=True size=medium sha=... | tok sha=35154dca2ad0c786  ** FROZEN backbone lineage **
  [OK]   mouse calibration: span=649x367 chunk=30 <- .../mouse_calibration.json (measured ...)
  [OK]   HID gadget: keyboard=yes mouse=rel at 192.168.1.144:9999
  [OK]   HID mouse reports: 10 relative reports accepted, socket alive, no click sent
  [OK]   UDP stream: 1280x720, 18.4 new frames/s (221/240 polls fresh), mean brightness 0.14
  [OK]   agent load: use_actions=True movement_mode=axis gate=True bf16=True
  [OK]   inference speed: full path (resize+tokenizer+dynamics+heads) 17.8 fps
  [OK]   policy not degenerate: 34 distinct targets, gate fired 19/120 (~2.7 clicks/s at 17 fps)
=== ALL CRITICAL CHECKS PASSED ===
```

### Failure → meaning → fix

| Message | What is wrong | Fix |
|---|---|---|
| `provenance: cannot identify the running code` | not staged and not a git checkout | re-run `stage_desktop_standalone.sh` |
| `provenance: tree is DIRTY` (warning) | uncommitted edits | fine for a demo, not reproducible |
| `checkpoint identity: FileNotFoundError` | wrong path in `env.sh` | re-stage, or fix `BC`/`TOK` |
| `** FROZEN backbone lineage **` | expected for the deployed checkpoint (§1a) — its `action_embed` was fitted to a different movement target and cannot adapt | not a failure; the fallback in §1a is the unfrozen one |
| `mouse calibration: no measured calibration` | the JSON is missing | §4, or accept the fallback |
| `** PROVISIONAL **` | calibration carried over, not re-measured | fine if Windows pointer settings are unchanged |
| `HID gadget: Connection refused` | `hid_server.py` is not running on the Pi | §2 |
| `HID gadget: hid_server has NO mouse gadget` | `/dev/hidg1` missing — **every click would be silently dropped** | `sudo MOUSE_MODE=rel setup_hid_combo.sh` then restart the server |
| `HID gadget: mouse is 'abs' but hybrid_sender speaks RELATIVE` | wrong descriptor | same as above |
| `never answered 'status'` | the Pi is running an old `hid_server.py` | copy this commit's `scripts/keysender/` to the Pi |
| `UDP stream: no stream frame within 15s` | ffmpeg not running, wrong IP, or firewall | §3b; check the desktop IP |
| `UDP stream: only 2.6 NEW frames/s` | **the failure that killed session 1** | raise `-framerate` on Windows / lower encoder load |
| `UDP stream: geometry (720,1280,3) != expected` | `-video_size` vs `--stream-size` mismatch | make them equal |
| `UDP stream: essentially black` | capturing the wrong screen/region | fix `-offset_x/-offset_y` |
| `inference speed: 6.1 fps < required 10` | something else is on the GPU | `nvidia-smi`; kill it |
| `policy is degenerate: 1 distinct target` | greedy decode leaked in | ensure `--temperature 1.0` |

---

## 6. Dry run — watch it think before it touches anything

Never go straight to `--inject hid`. This sends nothing.

**[desktop]**
```bash
INJECT=dry /mnt/storage/ahriuwu-live/play.sh
```

Healthy log line (one per ~20 acted frames):
```
[version] commit=902d8ed dirty=no <- VERSION (staged)
frame    100 | clicks=  16(2.70/s) casts=['Q'] rew=+0.03 | 16.8fps acted (model cap 20.3) [cap1 enc11 act39 inj0ms] stale=8%
```

Read it as:
- `clicks=N(R/s)` — **the number to watch.** Humans issue ~2/s. `0.00/s` means
  the policy is not committing new move commands; see `--gate-bias` below.
- `casts=[...]` — mostly `-`. Rare casts are expected (§0).
- `16.8fps acted` — how fast the agent is actually stepping. `(model cap 12.4)`
  is what the GPU alone could do; if *acted* is far below *cap*, the stream is
  the bottleneck, not the model.
- `stale=8%` — rolling fraction of loop polls that saw a repeated frame.
  **`stale > 25%` prints a `** STALE STREAM **` warning.** Stale frames are
  skipped, not fed to the model, so a high number degrades responsiveness rather
  than freezing the agent — but fix the stream anyway.
- `** NO NEW FRAME FOR 3.2s **` — the Windows ffmpeg died.

Let it run ~30 s. If clicks/s is near zero, add `--gate-bias 2.0` (raise until
clicks/s lands near 2). It shifts the firing *rate* without changing *where* the
agent clicks.

---

## 7. Live

**[desktop]**
```bash
INJECT=hid /mnt/storage/ahriuwu-live/play.sh
```

Add `--gate-bias <n>` if the dry run needed it. Everything else is already set
by `play.sh` (`--movement-mode mouse`, `--temperature 1.0`).

First 10 seconds, expect: the cursor snaps to the top-left corner and then out
to a point in the game world, roughly twice a second, right-clicking each time.
The corner visit is by design — it is how a relative mouse re-zeroes.

### Stopping safely

**`Ctrl+C` in the play_live terminal.** This is clean: it releases every held
key, clears mouse buttons, sends `reset` to the Pi and closes the socket. The Pi
also clears all inputs on disconnect, so killing the process hard is survivable.

If something is stuck anyway:
```bash
# [pi] — nuclear: drop the gadget, all inputs die with it
echo "" | sudo tee /sys/kernel/config/usb_gadget/g1/UDC
```
Or just unplug the Pi's USB cable.

Stop summary looks like:
```
stopped after 1840 acted frames in 108s (17.0 fps acted); 340/2130 loop iterations saw a repeated frame (16%).
mouse: {'moves': 214, 'slams': 214, 'last_cmd_ms': 104.3}  clicks sent=214 suppressed=31
```

Sessions record to `/mnt/storage/ahriuwu-live/recordings/session_*/` — the exact
352x352 frames the model saw, every action, and `meta.json` including the
commit and checkpoint shas.

---

## 8. What is verified and what is not

**Verified offline, by running real code on real bytes** (`tests/test_mouse_shim.py`):
- `hid_server._parse` on every wire form the sender emits and on malformed input.
- `RelMouse` report packing against the 4-byte descriptor; clamping; no stuck button.
- The corner-relative geometry, driven through the real `HybridKeyboard` into a
  simulated clamping screen with pointer acceleration: every target within 1% on
  an ideal rig; error does not grow over 200 commands; repeating one target is
  exactly repeatable; ≤0.25 error even with a 12%-wrong span; worst-case command
  106 ms (18 slam + 35 travel reports at 2 ms).
- Calibration file loading, including corrupt and implausible files.
- The `status` verb round-trips through the real server loop, and a burst of
  mouse/key/garbage lines does not drop the socket.

**NOT verified — no hardware was available:**
- **Nothing has been tested against the actual Pi or the actual Windows PC.**
  There is no Pi and no Windows box in this environment.
- That `MOUSE_SPAN = (649, 367)` is still correct for the current Windows
  pointer settings. It is carried over, flagged provisional.
- That the USB gadget sustains ~500 reports/s. `MOUSE_INTERVAL` is 2 ms; if the
  endpoint polls slower, `_send` blocks and each command takes longer than the
  106 ms measured. Watch `last_cmd_ms` in the stop summary — if it is far above
  ~110 ms, raise `MOUSE_INTERVAL` is *not* the fix; the gadget is the limit and
  the effective click rate simply caps out.
- That the corner slam does not disturb the camera. It should not with camera
  lock on and edge pan off, but this has never been seen on a real screen.
- The end-to-end latency from a game event to a click landing.
- Anything about how the policy behaves on HUD-bearing frames (§0).

---

## 9. Quick reference

```bash
# [desktop] deploy
bash /mnt/nfs/projects/ahriuwu/ops/stage_desktop_standalone.sh

# [pi] gadget + server
sudo MOUSE_MODE=rel scripts/keysender/setup_hid_combo.sh
python3 scripts/keysender/hid_server.py

# [windows] stream
ffmpeg -f gdigrab -framerate 20 -video_size 1280x720 -i desktop -c:v libx264 ^
  -preset ultrafast -tune zerolatency -pix_fmt yuv420p -g 20 -b:v 6M ^
  -f mpegts udp://<DESKTOP_IP>:5000?pkt_size=1316

# [desktop] gate, then dry, then live
/mnt/storage/ahriuwu-live/preflight.sh --inject hid --movement-mode mouse
INJECT=dry /mnt/storage/ahriuwu-live/play.sh
INJECT=hid /mnt/storage/ahriuwu-live/play.sh

# [desktop] one-off mouse aim test, no game input (moves to 4 corners + centre)
source /mnt/storage/ahriuwu-live/env.sh
$PY $AHRIUWU/scripts/keysender/hybrid_sender.py --host $PI --mouse-test
```
