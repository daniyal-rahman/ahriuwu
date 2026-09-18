# Silent-failure audit — lanerl

Scope: `/srv/nfs/projects/ahriuwu-lanerl/{lanerl,lanerl_rl,lanerl_bot}`,
`lanerl/logs/` (125 files, 430 MB), and
`/srv/nfs/projects/lanerl-vendor/LoLServer/GameServerLib/Lanerl/`.
Audited 2026-09-09 on `danilogin`. Nothing outside `lanerl_audit/` was modified.

Ranked by likelihood x damage. Everything below is a real mechanism with a
file:line, not a category of risk.

---

## STATUS as of 2026-09-12 — read this before acting on anything below

Re-checked every P0/P1 against the current tree. **A fixed finding left standing
is as misleading as a stale comment**, so the table says which are dead.

| # | title | status |
|---|---|---|
| 1 | `LANERL_BOT_CONFIG` silently ignored | **FIXED twice.** `LanerlConfig.cs:373-390` throws `FileNotFoundException`; `run_server.check_bot_config()` raises before launch. The *consequence* stands: the six `rep_tune*.json` / `rep_retreat.json` arms remain unfounded. |
| 2 | build checks printed success while failing | **HISTORICAL.** The evidence is still in `lanerl/logs/`; no current build script greps for success. |
| 3 | `run_server.run()` never checks rc | **STILL TRUE.** No `rc` / `died` in the returned dict. Cited lines drifted: `run_all.txt:19,21` -> `:23,25`. |
| 4 | broken Content script invisible | **STILL TRUE** (`Package.cs:327-332` logs at Debug and returns true; `Spell.cs:149` `?? new SpellScriptEmpty()`). But the inference that this explains `smoke2/3`'s `455 -> 455` is **NOT supported** — the bot farms 31 CS with those same WARNs. |
| 5 | `control_smoke.py` vacuous PASS | **FIXED.** |
| 6 | suites invisible to bare `pytest`; both red | **FIXED.** `testpaths` covers all four; `lanerl_train/tests` = 230 passed / 12 skipped. |
| 7 | `setup_server.sh` builds a different tree | **HALF FIXED.** The `/srv/nfs` literal is gone; it still builds `$VENDOR/GameServer`, and the trailing `find … | head` still exits 0 on nothing. |
| 8 | `LanerlBot.cs` records a Q cast that threw | **STILL TRUE** — `LanerlBot.cs:584-586`. |
| 9 | `bench_process_restart.py` has never run | **FIXED.** `bench/out/process_restart.json` holds real data. |
| 10 | 8 hardcoded ports, one overlap | **MOSTLY FIXED.** 2 `port-literal` left; the 5810/5911 overlap is gone. |
| 11 | server tests report a crash as SKIP | **MOSTLY FIXED.** 5 `skip-hides-failure` remain. |
| 12 | no `set -e` in 31 of 34 shells | **PARTLY.** The six named scripts are fixed; 28 `shell-no-errexit` + 3 `shell-no-pipefail` remain elsewhere. |
| 14 | "39 byte-identical PNGs" | **STALE.** `lanerl/logs/shots/` now has 94 PNGs, 10 distinct md5s. |
| 16 | `LanerlControl` swallows every action | **MOSTLY FIXED** — `LanerlWire` is a real JSON parser, `Complain()` counts and reports. **Still true:** `_listener.Start()` + `AcceptTcpClient()` in the constructor with no bind handling or accept timeout. |

Also drifted in the text below: `lanerl/logs` is now **1.8 GB** (not 430 MB),
`rads_D.log` is 139 MB (not 145), `/` has 100 GiB free (not 107).

Two facts that make several of these worse than they look:

* `/mnt/nfs` is a **symlink to `/srv/nfs` on `danilogin` only**. On `desktop`,
  `/mnt/nfs` is the real mount and `/srv/nfs` does not exist. Every `/srv/nfs`
  literal is a node-specific path that *silently* resolves here and dies there.
* **Nothing in `lanerl/`, `lanerl_rl/` or `lanerl_bot/` is tracked by git**
  (`git ls-files lanerl lanerl_rl lanerl_bot` -> 0). Neither is
  `lanerl-vendor/LoLServer/GameServerLib/Lanerl/` (`?? GameServerLib/Lanerl/`).
  There is no committed baseline to diff a silent regression against.

---

## P0 — will burn hours, and some already have

### 1. A missing `LANERL_BOT_CONFIG` is silently ignored; the arm runs the default bot

`lanerl-vendor/LoLServer/GameServerLib/Lanerl/LanerlConfig.cs:161`

```csharp
var path = Environment.GetEnvironmentVariable("LANERL_BOT_CONFIG");
if (!string.IsNullOrEmpty(path) && File.Exists(path))
{ ...load... }
// no else. No warning. cfg stays at its built-in defaults.
```

If the path does not resolve, the bot runs its compiled-in defaults and prints
nothing. An A/B then compares **the default bot against itself** and the
difference it reports is pure seed noise.

Six paths recorded in the bench configs do not exist right now:

| referenced from | missing config |
|---|---|
| `lanerl_bot/bench/out/cfg_tune.json:2,3` | `configs/tuned.json`, `configs/tuned_r150.json` |
| `lanerl_bot/bench/out/cfg_tune2.json:3,4,5` | `configs/tuned_look600.json`, `tuned_look800.json`, `tuned_look400_r150.json` |
| `lanerl_bot/bench/out/cfg_retreat.json:3` | `configs/hp25.json` |

So `rep_tune.json` (`tuned_r80` 36.3 vs `tuned_r150` 39.7), the three missing
arms of `rep_tune2.json`, and `rep_retreat.json`'s `retreat_hp25` (38.0) are
unfounded — those arms cannot have loaded a config.

**Worse, this is the node bug too.** `cfg_confirm.json:3-5` and
`cfg_curriculum.json:2-4` point the bronze/gold/diamond anchors at `/srv/nfs`.
Run the curriculum on `desktop` and all three anchors silently collapse into one
identical default bot — a difficulty ladder with no rungs, reported as numbers.

**Fix.** In `LanerlConfig.Load()`:

```csharp
if (!string.IsNullOrEmpty(path))
{
    if (!File.Exists(path))
        throw new FileNotFoundException("LANERL_BOT_CONFIG does not resolve on this node: " + path);
    Console.WriteLine("LANERL_BOT_CONFIG " + path);   // and echo the loaded values
    ...
}
```
Then make every config path relative to the repo, not to a mount point.
`preflight.sh` step 5 already fails on this.

---

### 2. Three build checks printed success while the build was failing — and the results were used

Exactly the incident that motivated this audit, three times in `lanerl/logs/`:

| file:line | content |
|---|---|
| `lanerl/logs/fix-604.out:3-4` | `1 Error(s)` ... then `BUILD_DONE` |
| `lanerl/logs/qfix-605.out:1-3` | two x `error NETSDK1064: Package Crc32.NET, version 1.2.0 was not found` ... then `DONE` |
| `lanerl/logs/ctlbuild-610.out:1` | the whole file is `DONE\n` (5 bytes) |
| `lanerl/logs/ctlb2-611.out:6,10,12` | the rerun 13 s later: `Build FAILED.` / `1 Error(s)` / `Time Elapsed 00:00:00.49` |
| `lanerl/logs/restore-612.out:8,10,13-15` | `error CS0103: The name 'BASIC_AA_BY_OWNER' does not exist` / `1 Error(s)` — then `=== verify the vision fix is actually IN the built dll ===` prints `1` and passes |

Downstream, `lanerl/logs/qab4-609.out:2` and `qab5-616.out:2` report
`q_ON_after_fix` with `crashes=0` and clean means. Those A/B numbers were
measured against a binary that the two failed builds could not have produced.
`qab5` is 3x *worse* than baseline (11.2 vs 31.9 CS) and that was not flagged.

`lanerl/logs/b3-613.out`, `b4-617.out`, `b5-620.out` are three byte-identical
32-byte files containing only `Build succeeded.\n    0 Error(s)\n` — no
`Time Elapsed`, no project lines. Every genuine build log here has
`Time Elapsed` (`build-472.out:32`, `ctlb2-611.out:12`, `restore-612.out:12`).
These are grep-filtered captures: a failure would have been discarded the same
way the success was.

**Fix.** Never grep a build for success. Build, check `$?`, and prove the
artifact moved:

```bash
set -euo pipefail
before=$(stat -c %Y "$BIN/GameServerLib.dll" 2>/dev/null || echo 0)
"$DOTNET" build --no-restore -c Release "$SLN" > "$LOG" 2>&1   # set -e catches it
after=$(stat -c %Y "$BIN/GameServerLib.dll")
[ "$after" -gt "$before" ] || { echo "BUILD PRODUCED NO NEW DLL"; exit 1; }
```

---

### 3. `run_server.run()` never checks the return code — a 1-second death reads as a clean 10-minute game

`lanerl_bot/bench/run_server.py:39-70`

```python
proc.wait(timeout=timeout_s)      # rc discarded
...
return parse_log(log) | {"wall_s": wall, "timed_out": timed_out}
```

A server that dies instantly (port taken, missing dll, unresolvable config)
returns `{"cs_rows": [], "tps": [], "fatal": [], "wall_s": 0.4}`. Callers:

* `measure_all.py:33-56` prints `wall=0.4s tps=None`, no CS lines, writes
  `measure_all.json`, exits 0.
* `sweep.py:36-42` records `cs=None` and takes `max()` over what remains.
* `replicate.py:51-53` is the only one that notices — as a `crashes` counter
  that nothing asserts on. **`bench/out/run_all.txt:19,21` shows `crashes=1`
  for both `bot_blue` and `noq`.** Two of 27 launches died on 2026-09-08 and
  nobody looked.

`parse_log`'s `fatal` filter (`run_server.py:87`) matches only `" FATAL "` and
`"Unhandled exception"`. It does not match the failure this server actually
has (finding 4), nor a `dotnet` startup error, nor a bind failure.

**Fix.** In `run()`: `rc = proc.wait(...)`; return `"rc": rc` and
`"died": rc not in (0, -9, -15) or wall < 30`. Make `replicate`/`sweep`/
`measure_all` raise on `died` rather than averaging over it.

---

### 4. A broken Content script is invisible to every build and turns the ability into a no-op

The whole chain, verified in this checkout:

1. `GameServerLib/Scripting/CSharp/CSharpScriptEngine.cs:107-114` — on a compile
   error it **removes the offending syntax tree and retries**, returning
   `CompilationStatus.SomeCompiled`.
2. `GameServerLib/Content/Package.cs:327-332` — `SomeCompiled` logs at
   **`_logger.Debug`** and **`return true`**. The package reports as loaded.
3. `GameServerLib/GameObjects/Spell/Spell.cs:149` —
   `Script = CreateObjectStatic<ISpellScript>(...) ?? new SpellScriptEmpty();`

Net effect: the server boots normally, plays a full game, and the ability
silently does nothing. `dotnet build` never sees these files — they are
Roslyn-compiled at runtime.

`LoLServer` currently has **uncommitted edits to exactly these files**:
`Content/LeagueSandbox-Scripts/Buffs/Garen/GarenQ.cs` (today 17:38),
`Characters/Garen/Q.cs` (today 17:22), `Maps/Map1/LevelScript.cs`.

**And it is already happening.** `preflight.sh` booted the current build and
found 132 `Could not find script` WARNs, including Garen's own:

```
Could not find script: Spells.GarenBasicAttack
Could not find script: Spells.GarenCritAttack
Could not find script: Spells.GarenPassive
Could not find script: Spells.GarenRPreCast
```

That is very likely why `lanerl/logs/smoke2-615.out:7` and `smoke3-618.out:7`
both read `[5] attack order: minion hp 455 -> 455` — the attack landed nothing.

**Fix.** Add `"Script compilation error"` and `"Could not find script"` to
`run_server.parse_log`'s fatal filter, promote `Package.cs:333` from `Debug` to
`Error`, and boot-check before every experiment (`preflight.sh` step 5 does).

---

### 5. `control_smoke.py` prints "CONTROL CHANNEL WORKS" when the thing it tests did nothing

`lanerl_rl/control_smoke.py`

* `:139-141` — it prints `minion hp {hp0} -> {hp1}` and **never asserts the
  minion took damage**. There is no assertion on the attack at all.
* `:142-146` — if no enemy minion is in the observation it prints
  `[5] attack: ... skipped` and then unconditionally prints
  `CONTROL CHANNEL WORKS — the policy can now drive the game.` and `return 0`.
* `:49` — `stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL`. The server's
  output is thrown away, including `LANERL_CONTROL error: ...`
  (`LanerlControl.cs:90`) and every Roslyn error from finding 4.
* `:54-59` — 120 x 1 s connect retries with no `proc.poll()` check. A server
  that died at t=0 costs two minutes and reports only
  `FAIL: server never opened the control port`, with the reason discarded at `:49`.

Evidence: 3 of the 4 recorded PASSes were vacuous —
`lanerl/logs/smoke-614.out:6,8` (skipped, then PASS),
`smoke2-615.out:7,9` and `smoke3-618.out:7,9` (`455 -> 455`, then PASS).
Only `smoke4-619.out:15` actually killed the minion.

**Fix.** `assert hp1 < hp0, f"attack dealt no damage: {hp0} -> {hp1}"`; make
"no minion in observation" a failure, not a skip; write the server log to a file
instead of `DEVNULL` and print its tail on any failure; poll `proc` in the
connect loop and abort with the log the moment it exits.

---

### 6. The lanerl test suites are invisible to a bare `pytest` — and both are red right now

`pyproject.toml`:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
```

A bare `pytest` at the worktree root collects **only `tests/`**. Neither
`lanerl_bot/tests/` nor `lanerl_rl/tests/` is ever collected, so their failures
never appear in any CI-style output. Measured just now:

```
pytest -q                                 ->  1 failed, 15 passed   (tests/ only)
pytest lanerl_bot/tests -q -m "not slow"  ->  1 failed, 21 passed
pytest lanerl_rl/tests  -q -m "not slow"  ->  34 failed, 33 passed, 12 errors
```

* `lanerl_bot/tests/test_last_hit.py:201` — `NameError: name '_VENDOR' is not
  defined`. The `_PROJECTS`/`_VENDOR` block was added to `content.py:16-18`,
  `bench/run_server.py:21-23` and `tests/conftest.py:20-22` but not here. The
  test that checks the item build path against real content has been erroring
  since that refactor.
* `lanerl_rl` is red from an **in-flight** change: `frame.py` (edited 19:38
  today) changed `LaneFrame.__init__`'s arity and `lanerl_rl/obs.py:301` still
  passes 5 args -> `TypeError: LaneFrame.__init__() takes from 3 to 4 positional
  arguments but 5 were given`. That also takes down `lanerl_rl/audit.py` — the
  gate that proves the actor observation contains no server-only state
  (`test_audit.py::test_audit_passes`). *Flagged, not a defect: another agent is
  mid-edit. The point is that nothing would have told you.*

**Fix.** `testpaths = ["tests", "lanerl_bot/tests", "lanerl_rl/tests"]`, and
run `lanerl_rl/audit.py` (it exits 1 correctly) as its own gate.

---

### 7. `lanerl/setup_server.sh` builds a different tree than anything runs

`lanerl/setup_server.sh:8,26,30,34`

```bash
VENDOR=/srv/nfs/projects/lanerl-vendor
cd "$VENDOR/GameServer"          # <- builds this
"$DOTNET_ROOT/dotnet" build --no-restore -c Release
```

Everything at runtime uses `$VENDOR/`**`LoLServer`**`/GameServerConsole/bin/Release/net6.0`
(`lanerl_bot/bench/run_server.py:25`, `lanerl_bot/tests/conftest.py:25`,
`lanerl_rl/env.py:273`, `lanerl_rl/control_smoke.py:28`).

`lanerl-vendor/GameServer/` has **no `GameServerLib/Lanerl/` directory at all**
and was last built 2026-09-03 08:21. `LoLServer/` was built 2026-09-09 19:24.
Running `setup_server.sh` succeeds, prints its artifact list, and rebuilds
nothing any experiment loads. This is finding 2's damage without needing a
compile error.

Also `:37` — `find ... | head` exits 0 when it finds nothing, so the
`=== artifacts ===` section can be empty and the script still exits 0.

**Fix.** Point it at `LoLServer`, delete or rename `lanerl-vendor/GameServer`,
and derive `VENDOR` from `${BASH_SOURCE[0]}` instead of `/srv/nfs`.

---

### 8. `LanerlBot.cs` records a Q cast that threw as a cast that happened

`lanerl-vendor/.../Lanerl/LanerlBot.cs:288-291`

```csharp
try { qSpell.Cast(_champ.Position, qKillable.Position, _champ); } catch { }
_lastQCastMs = _game.GameTime;
_qTargetId   = qKillable.NetId;
```

If `Cast` throws, the bot still stamps its own cooldown and target. Q never
fires, the bot's state says it did, and the only symptom is lower CS — read as a
config difference. Directly on top of today's `qab*` Q-on/Q-off A/B, and on top
of `Spells.GarenQ`-adjacent scripts being edited today.

The same pattern with a real risk of hiding an engine change:
`LanerlEpisode.cs:228,237` (spell rebind + cooldown reset both `catch { }`, then
the code proceeds as if they worked) and `LanerlEpisode.cs:304` (`TryRevive`
returns `false` on any exception; the caller counts `BuildingsUnrevivable`,
which `test_reset.py:49` does assert on — that one is fine).

**Fix.**
```csharp
bool cast = false;
try { qSpell.Cast(...); cast = true; }
catch (Exception e) { Console.WriteLine("LANERL_Q_CAST_FAIL " + e.Message); }
if (!cast) { ClearTarget(); return; }
```
and add `LANERL_Q_CAST_FAIL` to `run_server.parse_log`'s fatal list.

---

## P1 — will produce a wrong number rather than no number

### 9. `bench_process_restart.py` has never run

`lanerl_bot/bench/bench_process_restart.py:27,29` use `_VENDOR`, which is never
defined in the file. Verified: `NameError: name '_VENDOR' is not defined` at
import. The `_PROJECTS`/`_VENDOR` block was added to its three siblings and
missed here. The "process restart costs 12 s" number that `test_reset.py:150`
compares against therefore has no live producer.

**Fix.** Copy the three lines from `run_server.py:21-23`.

### 10. Ports: 8 hardcoded literals, and one pair that already overlaps

| port | where |
|---|---|
| 5119 | `run_server.py:40,113` (default), `lanerl_rl/env.py:294` (default) |
| 5410 | `measure_all.py:67` |
| 5901, 5902 | `lanerl_bot/tests/conftest.py:93,119` |
| 5911-5914 | `lanerl_bot/tests/test_cs_baseline.py:34,41,69,99` |
| 5810 (+100 -> 5910) | `lanerl_rl/control_smoke.py:37,48` |

`control_smoke.py 1811` uses game port **5911** — `test_cs_baseline.py:34`'s
do-nothing baseline. Two pytest runs at once collide on all six.
`lanerl/scaling_test.sh:1-3` documents the earlier version of this bug:
*"every instance needs its OWN PORT -- they all defaulted to 5119, so only the
first bound successfully and the rest died silently."* The lesson was applied to
that one script and nowhere else. The `scaling/*.log` files log no port at all,
so a collision there is undetectable after the fact.

**Fix.** One helper, used everywhere:
```python
def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); return s.getsockname()[1]
```
and make the failure loud: catch the bind exception in `LanerlControl`'s
constructor and print `LANERL_CONTROL_BIND_FAILED <port>` before exiting.

### 11. Server tests report a crashed server as a SKIP

`lanerl_bot/tests/conftest.py`

* `:45-48` — `requires_server` is `pytest.mark.skipif(not server_available(), ...)`,
  evaluated at **import time**. Missing build (or `LANERL_SKIP_SERVER_TESTS=1`)
  -> all 8 server tests skip and pytest **exits 0** with a green summary.
* `:94-95` — `if "LANERL_SELFTEST_END" not in text: pytest.skip(...)`. The server
  ran and *crashed*, and that is reported as a skip. The Python<->C# damage-parity
  tests then silently do not run.
* `:67-70` — `except subprocess.TimeoutExpired: pass`, and the return code is
  never read. A 600-second timeout is indistinguishable from success.

**Fix.** `pytest.fail("server produced no self-test dump — see " + log)` instead
of `skip`; keep `skip` only for `LANERL_SKIP_SERVER_TESTS=1`, and make preflight
refuse to launch an experiment when the build is absent, so "all skipped" can
never be mistaken for "all passed".

### 12. Shell scripts: no `set -e` in 31 of 34, plus 6 unconditional success banners

`lanerl_bot/bench/out/run_all.sh` is the clearest:

```bash
set -u                                            # :2  -- no -e, no pipefail
$PY measure_all.py --only reset 2>&1 | tail -20   # :6  -- exit status = tail's
$PY replicate.py ... 2>&1 | tail -8               # :8
echo "=== ALL DONE ==="                           # :9  -- unconditional
```

Same shape in `run_confirm.sh:4`, `run_curriculum.sh:4`, `lanerl/run_final.sh:61`,
`lanerl/scaling_test.sh:22`, `lanerl/stage_display.sh:50`.
`lanerl_bot/bench/out/final_run.txt` shows the run that produced one of these
being CANCELLED by Slurm mid-experiment.

Only `lanerl/setup_server.sh` has `set -euo pipefail`.

**Fix.** `set -euo pipefail` at the top of each; gate every banner on the real
status. Where a failure genuinely is acceptable, write `|| true` explicitly so
it is a decision, not an accident.

### 13. The RADS gate counted 2 errors out of 564,207

`lanerl/fix_rads*.sh` compute `not-in-manifest: $(grep -c 'not in the manifest' ...)`
and report the count as the verdict.
`lanerl/logs/radsD-516.out:13` reads `not-in-manifest: 2` while `rads_D.log` has
**564,207 lines containing `ERROR`** (280,850 `SignalSoftRepair`, 279,684
`GetFileArchiveMetadata: Missing naked file`). `radsE-517.out:13` is the same.
`radsF-519.out:13-14` passes the same gate with `loaded level? 2` against
`287800` in the D run — a five-order-of-magnitude regression the gate could not
see.

Same class: `lanerl/logs/ab-522.out:4` prints `surface-crash? 0` while
`ab_A.log:43` has `terminate called after throwing an instance of 'dxvk::DxvkError'`
and `EXIT=3`; `ab-522.out:7` has an **empty** `exit:` field.
`cd-525.out:13,26` reports `exit:` blank + `nullderef: 0` for two runs that both
exited 143 (`cd_C.log:18`, `cd_D.log:17`).

**Fix.** A verdict must assert a positive marker *and* the absence of the
negative one, and fail when it finds neither. Grep total `ERROR` lines, not one
hand-picked substring.

### 14. Recorded artefacts that are one frozen frame

`lanerl/logs/shots/` contains **39 byte-identical PNGs** (md5
`c96add5993aa9b512f2a96f89e0e719e`), including `t030.png ... t600.png` — the
entire ten-minute "gameplay" capture. `gameplay.mp4` (496 KB, `02:21:13`) is that
still image. `game-498.out:8,10,13,29` reports `CLIENT DIED (see client.log)`,
then `GAME STARTED at ~78s`, then lists the artefact, then prints its sanity
metric as `0` — and nothing acted on any of it.

`gameplay_final.mp4` was produced inside the `work-548` window where every
sample line read `procs=0` (`work-548.out:5-19`).

`lanerl/logs/ffmpeg.log` is **0 bytes**; `wineboot.log` is 0 bytes while
`disp-491.out:6` prints `prefix OK`.

**Fix.** After any capture, assert the artefact is not degenerate: distinct
frame hashes >= N, and file size / duration above a floor. A one-line
`md5sum shots/*.png | awk '{print $1}' | sort -u | wc -l` would have caught all
39 at once.

### 15. Two of eight scaling instances crashed on a real race; the runner said "done"

`lanerl/logs/scale2-595.out:4` — `N=8 done (6 instances reported)`.
`scaling/n8_i3.log:219-227` and `n8_i5.log:219-227` (both files *end* there):

```
System.InvalidOperationException: Collection was modified; enumeration operation may not execute.
   at LeagueSandbox.GameServer.ObjectManager.TeamHasVisionOn(TeamId, GameObject) in ObjectManager.cs:line 293
   at LeagueSandbox.GameServer.Game.GameLoop() in Game.cs:line 385
```

Those two logs are 33 KB / 227 lines; the other six are 760 KB / ~4,928 lines and
end on `Game is over`. And the whole earlier run was arithmetic-free:
`scale-592.out:1` `bc: command not found` x 40, with every result line printed as
`0.0 ticks/s ( 0.00x real)` and `AGGREGATE A =  x real time` — an empty number in
a formatted table, exit 0.

Note `ObjectManager.cs` is one of the files modified in the working tree.

**Fix.** `[ "$reported" -eq "$N" ] || { echo "FAIL: $reported/$N"; exit 1; }`,
and grep the per-instance logs for `Exception`. Also drop the
`pkill -f GameServerConsole` at `scaling_test.sh:15,21,23` — it kills every
server on the node, including another experiment's.

### 16. `LanerlControl` swallows every action it cannot execute

`lanerl-vendor/.../Lanerl/LanerlControl.cs:157-190`

* No `default:` on the `switch` — an unrecognised order type is a silent no-op,
  which is indistinguishable from the policy choosing to stand still.
* `case "cast"` returns silently on three separate guards (`:178,181`) and then
  `try { sp.Cast(...) } catch { }` at `:186`.
* `Num()` returns `float.NaN` for an absent key and `:181` does
  `(uint)Num(a, "id")` — an unchecked NaN->uint conversion.
* `:59` `_listener.Start()` and `:60` `AcceptTcpClient()` run in the
  **constructor**, with no bind error handling and no accept timeout. A taken
  port throws out of a constructor; a trainer that never attaches hangs forever
  holding the port.

**Fix.** Add `default: Console.WriteLine("LANERL_CONTROL unknown order: " + type); break;`,
count and periodically report rejected actions, and wrap `Start()` so a bind
failure prints `LANERL_CONTROL_BIND_FAILED <port>` and exits non-zero.

---

## P2 — real, lower blast radius

17. **`lanerl_rl/env.py:273,275`** — the only remaining hardcoded `/srv/nfs`
    defaults in Python (`DEFAULT_SERVER_DIR`, `DEFAULT_DOTNET_ROOT`). Its
    siblings all derive from `Path(__file__).resolve().parents[N]`.
    *(env.py is being edited; flagging only.)*
    Same in `lanerl/patch_server.py:13`, `patch_freerun.py:11`,
    `patch_toponly.py:9`, `record_headless.sh:7`, `serve_for_windows.sh:5`.
    The checker lists all 69.

18. **Already-burned mount mismatch**: `lanerl/logs/qab3-608.out:7,25` —
    `FileNotFoundError: '/srv/nfs/projects'` then
    `PermissionError: [Errno 13] Permission denied: '/srv/nfs'`, from a script
    living at `/mnt/nfs/...` writing to `/srv/nfs/...`.
    `qab2-607.out:13` — `FileNotFoundError: '/tmp/qab_configs.json'` (`/tmp` is
    per-node). Both died within 16 s and nothing downstream noticed.

19. **`lanerl/logs/headless-557.out:2-5,15`** — two minutes of
    `state.jsonl: No such file or directory` and `ticks=0`, then
    `=== recorded 4846 ticks, 41M ===`. `render2-560.out:1` then rendered the
    truncated file without noticing the gap. That same `state.jsonl` is what
    `lanerl_rl/tests/conftest.py:13-19` loads — and `.gitignore:40 logs/`
    excludes it from git, so `recording_path` **`pytest.skip`s silently** on any
    machine that does not happen to have the 42 MB file.

20. **log4net writes to a file with a Windows separator inside the build output.**
    `GameServerConsole/App.config` `<file value="Logs\" />` produces
    `bin/Release/net6.0/Logs\LeagueSandbox_08.09.2026.log` — 35 MB from one day,
    `appendToFile=true`, in the directory a `dotnet clean` deletes. It is the only
    archive of the Roslyn errors from finding 4, and `/` is at 94% (107 GiB free;
    `lanerl/logs/` alone is 430 MB with `rads_D.log` at 145 MB).

21. **`lanerl/logs/render-559.out:18`** —
    `inter-wave gaps: ['1s','1s','1s','1s','1s','32s','1s','1s']  (real League: 30s)`.
    The check printed the expected value next to a wrong one and wrote the mp4
    anyway.

22. **`pub-570.out:4-7`** — `cp: cannot stat '*Undefined*lib\.'` /
    `error MSB3073` / no `GameServerConsole.exe` produced. Caught only because
    someone reran it (`pub2-571.out:8` `EXE-OK`), not because 570 signalled.

23. **Vacuous-by-construction assertions.** `test_reset.py` is genuinely good —
    it asserts against the server's own 10 Hz record, not against the reset's
    return value, and `:96-97` explicitly justifies why `gold < 700` can fail.
    The one to watch is `test_cs_baseline.py:63` `assert 20 <= bot_cs <= 90`: the
    measured spread is 25-45 and `rep_final.json` has seeds landing at **4** and
    **11**, so this band is wide enough to pass a genuinely broken bot on a lucky
    seed while failing on an unlucky one.

---

## Deliverables

```
lanerl_audit/FINDINGS.md               this file
lanerl_audit/check_silent_failures.py  mechanical gate, exits non-zero on findings
lanerl_audit/preflight.sh              run before any experiment; fails fast
lanerl_audit/findings.json             machine-readable checker output
```

### `check_silent_failures.py` — current output on this tree

```
high      35  hardcoded-mount            medium    34  hardcoded-mount
high       6  config-silent-fallback     medium    31  shell-no-errexit
high       6  unconditional-done         medium     9  swallowed-exception
high       2  tests-not-collected        medium     6  config-silent-fallback
high       2  undefined-name             medium     5  skip-hides-failure
high       1  unchecked-returncode       medium     4  pipeline-masks-failure
                                         medium     3  shell-no-pipefail
                                         low       14  ambiguous-grep
                                         low        8  port-literal
TOTAL 166        exit 1
```

`--severity high` gates on the 52 that matter; `--json` for tooling.
The checker treats a crash inside itself as a HIGH finding, never as a pass.

### `preflight.sh` — verified working

Steps, fail-fast: NFS resolves on this node -> conda env imports -> server binary
exists **and no `.cs` is newer than it** -> every Content script compiles (boots
the server once) -> every `LANERL_BOT_CONFIG` resolves -> no HIGH checker finding
-> both test suites green.

On this node it correctly **failed at step 5** on the six missing bot configs.
With `--allow-missing-config` it reached the boot check and reported:

```
ok    all 810 Content scripts compiled; server reached 'Game is ready'
warn  132 'Could not find script' WARNs -- those abilities are no-ops:
        Spells.GarenBasicAttack / GarenCritAttack / GarenPassive / GarenRPreCast
```

which is finding 4 firing on the live build.

---

## If you fix five things

1. Make `LanerlConfig.Load()` **throw** on a set-but-unresolvable
   `LANERL_BOT_CONFIG`, and re-run any sweep that used `cfg_tune*.json` or
   `cfg_retreat.json` (finding 1).
2. Make `run_server.run()` return and check `proc.returncode`, and add
   `"Script compilation error"` / `"Could not find script"` to its fatal filter
   (findings 3, 4).
3. Never grep a build for success — check `$?` and check that the dll's mtime
   moved (finding 2). Re-run the `qab4`/`qab5` A/B.
4. Fix `control_smoke.py` to assert the attack landed and to keep the server log
   (finding 5). Then find out why `Spells.GarenBasicAttack` does not load.
5. `testpaths = ["tests", "lanerl_bot/tests", "lanerl_rl/tests"]`, and put
   `preflight.sh` in front of every experiment (finding 6).
