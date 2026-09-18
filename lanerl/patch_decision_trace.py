#!/usr/bin/env python3
"""LANERL_DECISION_TRACE=1 -> the server emits the BRANCH it took, not just state.

Installs `METH-002`. Idempotent, and it refuses to modify the canonical script
package -- see the isolation section below, which exists because getting that
wrong is silent.

WHY
---
`LanerlStateDump` says what is true after a tick. One wrong branch propagates
into every field it touches, so a porting bug has to be inferred backwards from
consequences. Measured cost: `ORDER-005` is ONE unported line
(`Spell.FinishCasting` ends with `UpdateMoveOrder(OrderType.Hold, true)` for any
non-`InstantCast` cast, and an auto-attack is a cast) and presented as a
4,791-count move-order residual read as symmetric jitter for weeks.

WHAT IT PATCHES
---------------
Assemblies (rebuild required, see BUILDING):
  * `Lanerl/LanerlDecisionTrace.cs`   -- new, copied from `vendor_patches/`
  * `ObjAIBase.UpdateMoveOrder`       -- accepted writes, after the script veto
  * `ObjAIBase.SetTargetUnit`         -- both endpoints of a real transition
  * `Spell.FinishCasting`             -- with `auto=` / `instant=`

Script package (NO rebuild -- Roslyn compiles it at boot), ISOLATED COPY ONLY:
  * `LaneMinionAI.OnUpdate`'s trigger -- which of the three arms fired

THE SCRIPT PACKAGE MUST BE ISOLATED, AND THE FAILURE IS SILENT
---------------------------------------------------------------
`Content/` is shared by every build. A script referencing `LanerlDecisionTrace`
against an assembly that predates it does NOT crash the server: the compile
error is logged, `ContentManager` reports `Loaded some C# scripts from package`
instead of `Loaded all`, and the game runs WITHOUT that script. A broken
`LaneMinionAI` therefore yields a complete, well-formed, entirely invalid trace
from a server whose minions have no AI (`METH-003`,
`lanerl_jax/parity/script_health.py`).

So this builds a private tree at `lanerl-vendor/Content-trace/`: a RELATIVE
symlink to the 113 MB `LeagueSandbox-Default` (unchanged, and relative so it
resolves under both `/srv/nfs` and `/mnt/nfs`) beside a real copy of the 8.8 MB
`LeagueSandbox-Scripts`. Select it with `lanerl/cfg/garen1v1_trace.json`, whose
`gameInfo.CONTENT_PATH` is ABSOLUTE: `Config.cs:69-74` falls back silently to
executable-relative when `Directory.Exists` is false, and a relative value
resolves against the server's CWD -- not the config, not `server_dir`.

BUILDING (never over bin/Release -- every parity run uses that binary)
---------------------------------------------------------------------
    export DOTNET_ROOT=/mnt/nfs/projects/lanerl-vendor/dotnet
    export PATH=$DOTNET_ROOT:$PATH
    cd /mnt/nfs/projects/lanerl-vendor/LoLServer
    dotnet build GameServerConsole/GameServerConsole.csproj -c Release \
      -o GameServerConsole/bin/Trace/net6.0 \
      -p:SolutionDir=/mnt/nfs/projects/lanerl-vendor/LoLServer/

`bin/Trace/net6.0` must be at THAT depth: Content is resolved relative to the
executable when `CONTENT_PATH` misses. `-p:SolutionDir` is required because a
post-build step copies `lib/` through `$(SolutionDir)`.

Then: `ServerLaunchSpec(server_dir=<bin/Trace/net6.0>,
config_path=<lanerl/cfg/garen1v1_trace.json>)`.

VERIFYING (all three, every time a call site is added)
------------------------------------------------------
1. `script_health.assert_all_scripts_loaded(log)` -- a skipped script is silent.
2. state rows byte-identical with the trace ON. 200 s idle, seed 0:
   `ce6f118002541655e528` / 1,159,704 rows.
3. the new call site actually FIRED. A 120 s window contains no combat at all
   (minions spawn at t=90 s at opposite ends of the lane), so it reports a
   clean neutrality pass while exercising zero `FinishCasting` sites.
"""
import pathlib
import shutil

# The NFS export is mounted at /srv/nfs on danilogin and /mnt/nfs on desktop.
# Never hardcode either: resolve everything from this file's own location.
_HERE = pathlib.Path(__file__).resolve().parent
_VENDOR = _HERE.parents[1] / "lanerl-vendor"
_SRV = _VENDOR / "LoLServer"
_ISO = _VENDOR / "Content-trace"

MARKER = "LanerlDecisionTrace"
MARKER_NS = "LeagueSandbox.GameServer.Lanerl.LanerlDecisionTrace"


def _patch(path: pathlib.Path, old: str, new: str, what: str) -> bool:
    s = path.read_text()
    if MARKER in s:
        print(f"  already patched: {what}")
        return False
    assert old in s, f"anchor not found in {what} -- the vendored source moved"
    path.write_text(s.replace(old, new, 1))
    print(f"  patched: {what}")
    return True


def install_trace_class() -> None:
    dst = _SRV / "GameServerLib/Lanerl/LanerlDecisionTrace.cs"
    src = _HERE / "vendor_patches/LanerlDecisionTrace.cs"
    if dst.exists() and dst.read_text() == src.read_text():
        print("  already installed: LanerlDecisionTrace.cs")
        return
    dst.write_text(src.read_text())
    print("  installed: LanerlDecisionTrace.cs")


def setup_isolated_content() -> None:
    """Private Content tree, so a script edit cannot reach the canonical one."""
    _ISO.mkdir(exist_ok=True)
    link = _ISO / "LeagueSandbox-Default"
    if not link.is_symlink():
        if link.exists():
            raise SystemExit(f"{link} exists and is not a symlink -- refusing")
        link.symlink_to(pathlib.Path("../LoLServer/Content/LeagueSandbox-Default"))
        print("  linked: LeagueSandbox-Default (relative, mount-portable)")
    else:
        print("  already linked: LeagueSandbox-Default")
    scripts = _ISO / "LeagueSandbox-Scripts"
    if not scripts.exists():
        shutil.copytree(_SRV / "Content/LeagueSandbox-Scripts", scripts)
        print("  copied: LeagueSandbox-Scripts (8.8 MB, the editable part)")
    else:
        print("  already present: LeagueSandbox-Scripts")


EMIT = "LeagueSandbox.GameServer.Lanerl.LanerlDecisionTrace.Emit"


def patch_update_move_order() -> None:
    f = _SRV / "GameServerLib/GameObjects/AttackableUnits/AI/ObjAIBase.cs"
    _patch(f,
           "            MoveOrder = order;\n\n            if ((MoveOrder == OrderType.OrderNone",
           "            MoveOrder = order;\n\n"
           "            // LANERL: the accepted write, after the script veto, so the\n"
           "            // trace records orders that actually took effect.\n"
           f"            {EMIT}(\n"
           "                _game.GameTime, \"UpdateMoveOrder\", NetId,\n"
           "                \"order=\" + order + \" publish=\" + publish);\n\n"
           "            if ((MoveOrder == OrderType.OrderNone",
           "ObjAIBase.UpdateMoveOrder")


def patch_set_target_unit() -> None:
    f = _SRV / "GameServerLib/GameObjects/AttackableUnits/AI/ObjAIBase.cs"
    s = f.read_text()
    if s.count(MARKER) >= 2:
        print("  already patched: ObjAIBase.SetTargetUnit")
        return
    old = ("                ApiEventManager.OnTargetLost.Publish(this, TargetUnit);\n"
           "            }\n\n            TargetUnit = target;\n")
    new = ("                ApiEventManager.OnTargetLost.Publish(this, TargetUnit);\n"
           "            }\n\n"
           "            // LANERL: the TRANSITION, both endpoints, only when it is one.\n"
           "            // A sim-side transition is derivable from state already; what\n"
           "            // this adds is the exact tick and the ordering WITHIN it.\n"
           "            if (!ReferenceEquals(TargetUnit, target)\n"
           f"                && {MARKER_NS}.Enabled)\n"
           "            {\n"
           f"                {EMIT}(\n"
           "                    _game.GameTime, \"SetTargetUnit\", NetId,\n"
           "                    \"from=\" + (TargetUnit == null ? 0u : TargetUnit.NetId)\n"
           "                    + \" to=\" + (target == null ? 0u : target.NetId)\n"
           "                    + \" networked=\" + networked);\n"
           "            }\n\n            TargetUnit = target;\n")
    assert old in s, "SetTargetUnit anchor not found -- vendored source moved"
    f.write_text(s.replace(old, new, 1))
    print("  patched: ObjAIBase.SetTargetUnit")


def patch_finish_casting() -> None:
    f = _SRV / "GameServerLib/GameObjects/Spell/Spell.cs"
    _patch(f,
           "        public void FinishCasting()\n        {\n            if (CastInfo.IsAutoAttack)\n            {",
           "        public void FinishCasting()\n        {\n"
           "            // LANERL: the highest-value branch in the server for this port.\n"
           "            // Its tail is not spell-specific -- every non-InstantCast cast\n"
           "            // ends with UpdateMoveOrder(Hold, true) -- and an auto-attack IS\n"
           "            // a cast. `ORDER-005`. `_game` is Spell's own field because\n"
           "            // GameObject._game is protected; InstantCast is a SpellDataFlags\n"
           "            // bit, not a CastInfo property.\n"
           f"            {EMIT}(\n"
           "                _game.GameTime, \"FinishCasting\",\n"
           "                CastInfo.Owner != null ? CastInfo.Owner.NetId : 0u,\n"
           "                \"spell=\" + (SpellName ?? \"?\")\n"
           "                + \" auto=\" + CastInfo.IsAutoAttack\n"
           "                + \" instant=\" + SpellData.Flags.HasFlag(SpellDataFlags.InstantCast));\n\n"
           "            if (CastInfo.IsAutoAttack)\n            {",
           "Spell.FinishCasting")


def main() -> None:
    print("assemblies (rebuild required):")
    install_trace_class()
    patch_update_move_order()
    patch_set_target_unit()
    patch_finish_casting()
    print("isolated content:")
    setup_isolated_content()
    canon = _SRV / "Content/LeagueSandbox-Scripts/AIScripts/LaneMinionAI.cs"
    if MARKER in canon.read_text():
        raise SystemExit(
            "REFUSING TO CONTINUE: the CANONICAL LaneMinionAI.cs references "
            f"{MARKER}. Every build shares that tree, and a script that fails "
            "to compile is silently skipped rather than fatal (METH-003), so "
            "this would disable minion AI in every run while still producing "
            "plausible traces. Revert it: git checkout -- <that file>.")
    print("canonical script package: clean (verified)")
    print("\nNOT applied here: the `LaneMinionAI.OnUpdate` trigger split. It\n"
          "lives in the ISOLATED tree only, it must preserve the short-circuit\n"
          "`A && (B || C || D)` order exactly (`FoundNewTarget` takes a flag and\n"
          "is not known to be pure), and its correctness is established by the\n"
          "state-row digest rather than by review -- so it is applied and\n"
          "verified deliberately, not as a side effect of running this script.\n"
          "Measured trigger split, 200 s idle: Sweep250 7,671 / TargetJustDied\n"
          "75 / CallForHelp 53 -- the 250 ms sweep is 98.4% of all minion\n"
          "re-evaluations, which is what `INJ-003` was silently removing.")


if __name__ == "__main__":
    main()
