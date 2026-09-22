#!/usr/bin/env python3
"""Make two blind spots observable: `AA-004` (clamped cooldown), `CFH-002` (cleared map).

Idempotent. Refuses to touch the CANONICAL script package -- see ISOLATION,
which exists because getting it wrong is silent (`METH-003`).

WHY THESE TWO
-------------
Both are places where the dump DESTROYS the state a parity test needs, so a
residual gets scored against the simulator for something the simulator was
never given.

`AA-004`  `LanerlAim.AutoAttackCooldownRemaining` applies `Math.Max(0f, ...)`
          and `LanerlStateDump` then quantises, so a cooldown that is still
          strictly POSITIVE but within rounding of zero publishes as a flat 0,
          indistinguishable from a genuinely ready swing. Inject that 0 and the
          sim fires a swing the server had not allowed. Measured: 453/453
          sim-fires-early rows sit on a dumped 0, and once the exact value is
          visible, 74 of 124 carry a true positive residue (5.5e-07/1.3e-06 s).

`CFH-002` `LaneMinionAI` fills `unitsAttackingAllies`, reads it in
          `FoundNewTarget`, and CLEARS it at the end of the same `OnUpdate`.
          The dump reads that dictionary AFTER `OnUpdate`, so it only ever sees
          it post-clear. NOTE the outcome: emitting it pre-clear REFUTED the
          attribution built on it -- the map is non-empty on only 1.03% of
          clear events and contained the server's pick in 0 of 43 disagreeing
          rows. The instrument is kept because that refutation is exactly what
          it is for, not because it explained anything.

WHAT IT PATCHES
---------------
Assemblies (REBUILD REQUIRED, see BUILDING):
  * `Lanerl/LanerlAim.cs`        -- adds `AutoAttackCooldownRemainingRaw`,
                                    the same counter with no `Math.Max` floor
  * `Lanerl/LanerlStateDump.cs`  -- adds `aacdraw=` (quantised) and
                                    `aacdbits=` (float32 bit pattern)

Script package (NO rebuild -- Roslyn compiles it at boot), ISOLATED COPY ONLY:
  * `LaneMinionAI.OnUpdate`      -- emits `CallForHelpClear` before the wipe

`aacd=` IS NEVER MODIFIED. Every recorded corpus and every parser depends on
it; changing it would silently invalidate all historical numbers. Both new
fields are ADDITIVE.

WHY `aacdbits` AND NOT JUST `aacdraw`
-------------------------------------
`aacdraw` removes the clamp but keeps the 1/1024 quantisation, and the residue
at issue is SUB-quantum: a turret period of 1.1836 s over a 1/60 s tick is
71.016 ticks, leaving +0.00027 s = 0.28 quanta. Rounding flattens that to 0
exactly like the clamp did. Only the exact bits separate "+1.3 us, not ready"
from "0" and from "-1 tick, ready". Shipping only `aacdraw` would have looked
like a fix and measured nothing -- it is kept alongside purely because it is
cheap and human-readable.

ISOLATION, AND WHY THE FAILURE IS SILENT
----------------------------------------
`Content/` is shared by every build. A script that fails to compile does NOT
stop the server: `ContentManager` logs `Loaded some C# scripts from package`
instead of `Loaded all` and the game runs WITHOUT that script. A broken
`LaneMinionAI` yields a complete, well-formed, entirely invalid trace from a
server whose minions have no AI (`METH-003`). So the emit goes only into
`lanerl-vendor/Content-trace/`, and this script refuses to run if the canonical
copy already references it.

BUILDING
--------
    export DOTNET_ROOT=/srv/nfs/projects/lanerl-vendor/dotnet
    ops/login_capped.sh 8G 3 $DOTNET_ROOT/dotnet build \\
      /srv/nfs/projects/lanerl-vendor/LoLServer/GameServerConsole/GameServerConsole.csproj \\
      -c Release \\
      -o /srv/nfs/projects/lanerl-vendor/LoLServer/GameServerConsole/bin/Trace/net6.0 \\
      -p:SolutionDir=/srv/nfs/projects/lanerl-vendor/LoLServer/

Build to `bin/Trace/net6.0`, NEVER over `bin/Release` (every parity run uses
that binary), and at the same directory DEPTH because the server resolves
Content relative to its executable. Drive it with `lanerl/cfg/garen1v1_trace.json`,
whose `CONTENT_PATH` is ABSOLUTE -- a relative one is silently ignored
(`Config.cs` falls back to `GetContentPath()` when `Directory.Exists` is false).

VERIFYING (do not skip -- every failure mode here is quiet)
-----------------------------------------------------------
    python -m lanerl.patch_observability --verify
    # then, on any recording whose numbers will be quoted:
    python -c "from pathlib import Path; \\
      from lanerl_jax.parity.script_health import check_script_load; \\
      print(check_script_load(Path('<log>')))"      # must be Loaded all

Behaviour neutrality is expected BY CONSTRUCTION (both fields live in
`DescribeInternals`, which `Describe()` never calls) and was nonetheless
MEASURED: `LANERL_STATEROW` digest `7e24091cef70d2aeb319`, 1,219,899 rows,
identical with instrumentation ON and OFF.

Use a window longer than 200 s. Waves clash around t=110 s, so a 120 s window
exercises neither emit site and would "pass" while proving nothing -- that
near-miss already happened once (`METH-002`).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_VENDOR = _HERE.parents[1] / "lanerl-vendor"
_SRV = _VENDOR / "LoLServer"
_ISO = _VENDOR / "Content-trace"

_AIM = _SRV / "GameServerLib/Lanerl/LanerlAim.cs"
_DUMP = _SRV / "GameServerLib/Lanerl/LanerlStateDump.cs"
_ISO_AI = _ISO / "LeagueSandbox-Scripts/AIScripts/LaneMinionAI.cs"
_CANON_AI = _SRV / "Content/LeagueSandbox-Scripts/AIScripts/LaneMinionAI.cs"

_RAW_ACCESSOR = '''
        /// <summary>
        /// AA-004: the SAME private counter as <see cref="AutoAttackCooldownRemaining"/>,
        /// but WITHOUT the <c>Math.Max(0f, ...)</c> floor. That floor runs before
        /// <c>LanerlStateDump</c> quantises the value, so a cooldown that is still
        /// strictly positive but within rounding of zero is dumped as a flat 0 --
        /// indistinguishable from a genuinely-ready swing, which makes an injected
        /// one-step sim fire a swing the server had not allowed yet. Additive: it
        /// does not change <see cref="AutoAttackCooldownRemaining"/> or any caller.
        /// </summary>
        public static float AutoAttackCooldownRemainingRaw(ObjAIBase unit)
        {
            if (unit == null) return 0f;
            if (_aaCooldownField == null)
            {
                _aaCooldownField = typeof(ObjAIBase).GetField(
                    "_autoAttackCurrentCooldown",
                    BindingFlags.NonPublic | BindingFlags.Instance);
            }
            if (_aaCooldownField == null) return 0f;
            var v = _aaCooldownField.GetValue(unit);
            return v is float f ? f : 0f;
        }
'''

_DUMP_ANCHOR = '" aacd=" + Q(LanerlAim.AutoAttackCooldownRemaining(ai), StatQ) +'
_DUMP_ADD = '''
                // AA-004: `aacd=` above is clamped to >=0 BEFORE quantisation, so a
                // still-positive sub-quantum residue is flattened to a flat 0. Both
                // fields below are ADDITIVE; `aacd=` is untouched because every
                // recorded corpus and every parser depends on it. `aacdbits` is the
                // load-bearing one -- `aacdraw` is still quantised at 1/1024 and so
                // cannot resolve the sub-quantum residue that is the whole point.
                " aacdraw=" + Q(LanerlAim.AutoAttackCooldownRemainingRaw(ai), StatQ) +
                " aacdbits=" + Bits(LanerlAim.AutoAttackCooldownRemainingRaw(ai)) +'''

_CFH_ANCHOR = "                    callsForHelpMayBeCleared = false;"
_CFH_ADD = '''                    // CFH-002 (ISOLATED Content-trace copy ONLY -- the canonical
                    // package must stay clean, METH-003). `unitsAttackingAllies` is
                    // filled by OnCallForHelp, READ by FoundNewTarget, and wiped on
                    // the next line, and the state dump reads it after OnUpdate has
                    // returned -- so the dump can only ever see it empty. Emit the
                    // contents here, which is exactly what FoundNewTarget just saw.
                    // `localTime` is time-since-spawn, NOT game time (a Content
                    // script cannot reach `_game`, which is protected); align it to
                    // a game tick through the dump's own `ailocal=`.
                    if (LeagueSandbox.GameServer.Lanerl.LanerlDecisionTrace.Enabled)
                    {
                        string lanerlCfh = "n=" + unitsAttackingAllies.Count;
                        if (unitsAttackingAllies.Count > 0)
                        {
                            lanerlCfh += " " + string.Join(";", unitsAttackingAllies
                                .Select(kv => kv.Key.NetId + ":" + kv.Value));
                        }
                        LeagueSandbox.GameServer.Lanerl.LanerlDecisionTrace.Emit(
                            localTime, "CallForHelpClear", LaneMinion.NetId, lanerlCfh);
                    }
'''


def _insert_once(path: Path, anchor: str, addition: str, marker: str) -> str:
    if not path.exists():
        return f"MISSING {path}"
    text = path.read_text()
    if marker in text:
        return f"already installed: {path.name}"
    if anchor not in text:
        raise SystemExit(
            f"REFUSING TO CONTINUE: anchor not found in {path}.\n"
            f"  anchor: {anchor!r}\n"
            "The vendored tree has moved under this patch. Re-derive the "
            "anchor by hand rather than letting a partial install through -- "
            "a half-patched dump produces a well-formed, invalid trace.")
    path.write_text(text.replace(anchor, anchor + addition, 1))
    return f"patched: {path.name}"


def verify() -> int:
    """Report install state of every site. Exit non-zero if inconsistent."""
    checks = [
        ("LanerlAim.AutoAttackCooldownRemainingRaw", _AIM,
         "AutoAttackCooldownRemainingRaw"),
        ("LanerlStateDump aacdbits=", _DUMP, "aacdbits="),
        ("LanerlStateDump aacdraw=", _DUMP, "aacdraw="),
        ("Content-trace CallForHelpClear", _ISO_AI, "CallForHelpClear"),
    ]
    ok = True
    for name, path, needle in checks:
        present = path.exists() and needle in path.read_text()
        print(f"  {'OK ' if present else 'MISSING'}  {name}")
        ok &= present
    canon_dirty = _CANON_AI.exists() and "CallForHelpClear" in _CANON_AI.read_text()
    print(f"  {'FAIL' if canon_dirty else 'OK '}  canonical Content/ is clean "
          f"of the emit")
    if canon_dirty:
        print("\nCANONICAL PACKAGE IS CONTAMINATED. Every build shares it, and "
              "a compile error there silently removes minion AI instead of "
              "failing (METH-003). Revert it before running anything.")
        return 2
    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true",
                    help="report install state and exit; changes nothing")
    a = ap.parse_args()

    if a.verify:
        sys.exit(verify())

    if _CANON_AI.exists() and "CallForHelpClear" in _CANON_AI.read_text():
        raise SystemExit(
            "REFUSING TO CONTINUE: the CANONICAL LaneMinionAI.cs already "
            "references CallForHelpClear. That package is shared by every "
            "build and a compile error in it silently removes minion AI "
            "rather than failing (METH-003). Revert it first.")
    if not _ISO_AI.exists():
        raise SystemExit(
            f"REFUSING TO CONTINUE: no isolated script tree at {_ISO}. "
            "Run `python -m lanerl.patch_decision_trace` first -- it builds "
            "the isolated Content-trace tree this patch writes into.")

    print(_insert_once(_AIM, "        }", _RAW_ACCESSOR,
                       "AutoAttackCooldownRemainingRaw")
          if "AutoAttackCooldownRemaining" in _AIM.read_text() else
          "MISSING AutoAttackCooldownRemaining in LanerlAim.cs")
    print(_insert_once(_DUMP, _DUMP_ANCHOR, _DUMP_ADD, "aacdbits="))
    print(_insert_once(_ISO_AI, _CFH_ANCHOR, "\n" + _CFH_ADD, "CallForHelpClear"))
    print("\nNow REBUILD (assemblies changed) -- see this module's docstring, "
          "then `--verify`, then check `Loaded all` on the first recording.")


if __name__ == "__main__":
    main()
