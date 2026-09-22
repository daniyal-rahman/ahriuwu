"""
Make the server say WHO moved a unit, not just that it moved.

WHY
---
Gate 1's three largest open residual families are `target` (522 misses),
`move_order` (354) and `waypoints` (317). All three are state diffs: they
say which unit disagreed and by how much, and they cannot say which branch
of the server chose the value. Every attempt to close that gap by argument
has failed -- four inferred attributions have been REFUTED in this project
(`CFH-002`, `GIVE-001`, `TGT-STALE`, `AA-005`), each by the first direct
measurement of the thing being argued about.

The structural fact that makes direct measurement cheap here: each of the
three fields has exactly ONE write site in the whole server.

    ObjAIBase.TargetUnit   <- ObjAIBase.SetTargetUnit      (ObjAIBase.cs:1001)
    ObjAIBase.MoveOrder    <- ObjAIBase.UpdateMoveOrder    (ObjAIBase.cs:1371)
    AttackableUnit.Waypoints <- AttackableUnit.SetWaypoints (AttackableUnit.cs:1017)

So a caller tag on three methods is a COMPLETE answer key for 1,193 of the
open rows: for any disagreeing unit-tick, the trace names the C# line that
set the server's value, and the port either has that line or does not.

HOW THE CALLER IS CAPTURED
--------------------------
`[CallerMemberName]` / `[CallerLineNumber]`, not a stack walk. The compiler
substitutes literals at each call site, so the cost of a tagged call is
passing two constants -- there is no reflection, no `StackTrace`, and no
behaviour change when the trace is off. Verified safe to widen these three
signatures: none is `virtual`, none is overridden, none is declared on an
interface in `GameServerCore`, and none is used as a method group. Optional
parameters bind at the call site, so every existing caller -- including the
Roslyn-compiled `Content/` scripts, which compile against the assembly --
keeps compiling untouched.

THE TWO SILENT BRANCHES
-----------------------
Two of these methods can decline to do anything and return without a trace
of having been asked, which is exactly the shape of bug a state diff cannot
see:

  * `UpdateMoveOrder` returns early when `OnUnitUpdateMoveOrder.Publish`
    vetoes the order. The observed residual is 22 rows of `sim=HOLD /
    server=MOVE_TO` against 0 the other way -- a 100% one-sided skew, which
    is what a veto the port does not implement looks like.
  * `SetWaypoints` returns false on four separate conditions (null, count,
    origin mismatch, `!CanChangeWaypoints()`). 122 of 124 waypoint rows are
    `sim=1 / server=2`: the server keeps a path where the simulator has
    stopped. Which of the four rejections the port is over-applying is not
    derivable from the count.

Both now emit, with the reason distinguished, so "the server declined" and
"the server was never asked" stop looking identical.

BEHAVIOUR NEUTRALITY IS THE CONTRACT
------------------------------------
Every emit is gated on `LanerlDecisionTrace.Enabled` (env `LANERL_DECISION_TRACE=1`)
and reads state without writing any. The canonical corpus's `LANERL_STATEROW`
stream must stay byte-identical -- sha1 cd13aad2cd09772cd5a2d9921f6b766f54a6ae7a --
and that is checked by re-recording, not assumed.

USAGE

    python -m lanerl.patch_writesites
    python -m lanerl.patch_writesites --verify

Then rebuild; the build command is in `lanerl/patch_observability.py`'s docstring.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_VENDOR = _HERE.parents[1] / "lanerl-vendor"
_SRV = _VENDOR / "LoLServer"

_OBJAI = _SRV / "GameServerLib/GameObjects/AttackableUnits/AI/ObjAIBase.cs"
_UNIT = _SRV / "GameServerLib/GameObjects/AttackableUnits/AttackableUnit.cs"
_OBJMGR = _SRV / "GameServerLib/ObjectManager.cs"

_CMN = "System.Runtime.CompilerServices.CallerMemberName"
_CLN = "System.Runtime.CompilerServices.CallerLineNumber"
_TRACE = "LeagueSandbox.GameServer.Lanerl.LanerlDecisionTrace"

# --- 1. SetTargetUnit: who set the target -----------------------------------

_TGT_SIG_A = "        public void SetTargetUnit(AttackableUnit target, bool networked = false)\n"
_TGT_SIG_B = (
    "        public void SetTargetUnit(AttackableUnit target, bool networked = false,\n"
    f"            [{_CMN}] string lanerlTgtCaller = \"\",\n"
    f"            [{_CLN}] int lanerlTgtLine = 0)\n"
)

_TGT_EMIT_A = "                    + \" networked=\" + networked);\n"
_TGT_EMIT_B = (
    "                    + \" networked=\" + networked\n"
    "                    // GATE1-003: the transition alone leaves 24 rows where the\n"
    "                    // minion's script gate never opened and the server changed\n"
    "                    // target anyway. `TGT-NULLOUT` is live and does NOT clear\n"
    "                    // them, so the mechanism is neither the script nor the\n"
    "                    // dead/invisible null-out. This names the C# line instead\n"
    "                    // of leaving it to be argued about.\n"
    "                    + \" caller=\" + lanerlTgtCaller + \"@\" + lanerlTgtLine);\n"
)

# --- 2. UpdateMoveOrder: who set it, and when it was vetoed -----------------

_ORD_SIG_A = "        public void UpdateMoveOrder(OrderType order, bool publish = true)\n"
_ORD_SIG_B = (
    "        public void UpdateMoveOrder(OrderType order, bool publish = true,\n"
    f"            [{_CMN}] string lanerlOrdCaller = \"\",\n"
    f"            [{_CLN}] int lanerlOrdLine = 0)\n"
)

_ORD_VETO_A = (
    "                if (!ApiEventManager.OnUnitUpdateMoveOrder.Publish(this, order))\n"
    "                {\n"
    "                    return;\n"
    "                }\n"
)
_ORD_VETO_B = (
    "                if (!ApiEventManager.OnUnitUpdateMoveOrder.Publish(this, order))\n"
    "                {\n"
    "                    // The order was REQUESTED and refused by a script. Without\n"
    "                    // this the refusal is indistinguishable from never asking,\n"
    "                    // and the two have opposite fixes in the port.\n"
    f"                    {_TRACE}.Emit(\n"
    "                        _game.GameTime, \"UpdateMoveOrderVetoed\", NetId,\n"
    "                        \"order=\" + order + \" was=\" + MoveOrder\n"
    "                        + \" caller=\" + lanerlOrdCaller + \"@\" + lanerlOrdLine);\n"
    "                    return;\n"
    "                }\n"
)

_ORD_EMIT_A = "                \"order=\" + order + \" publish=\" + publish);\n"
_ORD_EMIT_B = (
    "                \"order=\" + order + \" was=\" + lanerlOrdWas + \" publish=\" + publish\n"
    "                + \" caller=\" + lanerlOrdCaller + \"@\" + lanerlOrdLine);\n"
)

# The previous value has to be captured BEFORE the assignment. Emitting
# `MoveOrder` from below `MoveOrder = order;` reads back the value just
# written, so `was=` equalled `order=` on all 203,467 events in the first
# recording -- a field that looked populated and carried no information. The
# transition is the entire point: the open residual is 22 rows of
# `sim=HOLD / server=MOVE_TO` against 0 the other way.
_ORD_WAS_A = "            MoveOrder = order;\n"
_ORD_WAS_B = (
    "            var lanerlOrdWas = MoveOrder.ToString();\n"
    "            MoveOrder = order;\n"
)

# --- 3. SetWaypoints: who set the path, and why it was refused --------------

_WPT_SIG_A = "        public bool SetWaypoints(List<Vector2> newWaypoints)\n"
_WPT_SIG_B = (
    "        public bool SetWaypoints(List<Vector2> newWaypoints,\n"
    f"            [{_CMN}] string lanerlWptCaller = \"\",\n"
    f"            [{_CLN}] int lanerlWptLine = 0)\n"
)

_WPT_BODY_A = (
    "            if (newWaypoints == null || newWaypoints.Count <= 1 || newWaypoints[0] != Position || !CanChangeWaypoints())\n"
    "            {\n"
    "                return false;\n"
    "            }\n"
    "\n"
    "            _movementUpdated = true;\n"
)
_WPT_BODY_B = (
    "            if (newWaypoints == null || newWaypoints.Count <= 1 || newWaypoints[0] != Position || !CanChangeWaypoints())\n"
    "            {\n"
    "                // 122 of 124 waypoint residuals are `sim=1 / server=2` -- the\n"
    "                // server holds a path where the port has stopped. Four separate\n"
    "                // conditions produce that `false`, and the count cannot say\n"
    f"                // which one the port is over-applying. {_TRACE.rsplit('.', 1)[1]}\n"
    "                // is off by default, so this costs one branch on a cached bool.\n"
    f"                if ({_TRACE}.Enabled)\n"
    "                {\n"
    f"                    {_TRACE}.Emit(\n"
    "                        _game.GameTime, \"SetWaypointsRejected\", NetId,\n"
    "                        \"reason=\" + (newWaypoints == null ? \"null\"\n"
    "                            : newWaypoints.Count <= 1 ? \"count\"\n"
    "                            : newWaypoints[0] != Position ? \"origin\"\n"
    "                            : \"cannotchange\")\n"
    "                        + \" n=\" + (newWaypoints == null ? 0 : newWaypoints.Count)\n"
    "                        + \" caller=\" + lanerlWptCaller + \"@\" + lanerlWptLine);\n"
    "                }\n"
    "                return false;\n"
    "            }\n"
    "\n"
    f"            if ({_TRACE}.Enabled)\n"
    "            {\n"
    "                // The ACCEPTED path: count plus its far end. The count alone is\n"
    "                // what the residual already measures; the endpoint is what says\n"
    "                // whether the port disagrees about the path or about the goal.\n"
    "                var lanerlEnd = newWaypoints[newWaypoints.Count - 1];\n"
    f"                {_TRACE}.Emit(\n"
    "                    _game.GameTime, \"SetWaypoints\", NetId,\n"
    "                    \"n=\" + newWaypoints.Count\n"
    "                    + \" endx=\" + lanerlEnd.X.ToString(\"F2\", System.Globalization.CultureInfo.InvariantCulture)\n"
    "                    + \" endy=\" + lanerlEnd.Y.ToString(\"F2\", System.Globalization.CultureInfo.InvariantCulture)\n"
    "                    + \" order=\" + (this is LeagueSandbox.GameServer.GameObjects.AttackableUnits.AI.ObjAIBase lanerlAi\n"
    "                        ? lanerlAi.MoveOrder.ToString() : \"-\")\n"
    "                    + \" caller=\" + lanerlWptCaller + \"@\" + lanerlWptLine);\n"
    "            }\n"
    "\n"
    "            _movementUpdated = true;\n"
)

# --- 4. RefreshWaypoints: the chase decision --------------------------------
# NOTE: deliberately NO caller tag here. `RefreshWaypoints` IS virtual and IS
# overridden (`BaseTurret.RefreshWaypoints`), so widening its signature would
# break the override. The state at entry is what matters anyway.

_REF_A = (
    "        public virtual void RefreshWaypoints(float idealRange)\n"
    "        {\n"
    "            if (MovementParameters != null)\n"
    "            {\n"
    "                return;\n"
    "            }\n"
)
_REF_B = (
    "        public virtual void RefreshWaypoints(float idealRange)\n"
    "        {\n"
    f"            if ({_TRACE}.Enabled)\n"
    "            {\n"
    "                // Entry state, before any of the branches below run. This is the\n"
    "                // one place that turns a target into a path AND promotes the\n"
    "                // move order to AttackTo, so the `move_order` and `waypoints`\n"
    "                // residuals -- already measured to be ONE mechanism, 73%/60%\n"
    "                // co-located with target/fire -- both pass through here.\n"
    f"                {_TRACE}.Emit(\n"
    "                    _game.GameTime, \"RefreshWaypoints\", NetId,\n"
    "                    \"ideal=\" + idealRange.ToString(\"F2\", System.Globalization.CultureInfo.InvariantCulture)\n"
    "                    + \" order=\" + MoveOrder\n"
    "                    + \" tgt=\" + (TargetUnit == null ? 0u : TargetUnit.NetId)\n"
    "                    + \" mp=\" + (MovementParameters != null ? 1 : 0)\n"
    "                    + \" cast=\" + (_castingSpell != null ? 1 : 0)\n"
    "                    + \" chan=\" + (ChannelSpell != null ? 1 : 0)\n"
    "                    + \" stc=\" + (SpellToCast != null ? 1 : 0)\n"
    "                    + \" wps=\" + (Waypoints == null ? 0 : Waypoints.Count)\n"
    "                    + \" wpkey=\" + CurrentWaypointKey);\n"
    "            }\n"
    "\n"
    "            if (MovementParameters != null)\n"
    "            {\n"
    "                return;\n"
    "            }\n"
)


# --- 5. Untarget: WHICH of the two engine paths cleared the target ----------
# `caller=Untarget@973` is 204 of the corpus's 4,616 target writes -- the
# largest engine-side site, 6.5x the null-out already ported as `TGT-NULLOUT`.
# But `Untarget` has two callers with completely different semantics, and the
# tag as emitted cannot tell them apart:
#
#   ObjAIBase.LateUpdate  -- this unit drops an untargetable target, at the
#                            END of the tick, and only when `!IsUseable`
#                            (the OPPOSITE polarity to `UpdateTarget`'s test).
#   ObjectManager.StopTargeting -- a BROADCAST over every ObjAIBase in the
#                            game, whose only non-script caller is the first
#                            line of `AttackableUnit.Die`. Every unit pointing
#                            at the corpse is cleared INSIDE the death tick,
#                            during damage resolution -- not on anyone's next
#                            update.
#
# The second is a different mechanism from anything the port implements, and
# its timing is exactly the shape of the residual (`TargetJustDied` enriched
# x739). Passing the caller through turns that from a good argument into a
# count.
_UNT_SIG_A = "        public void Untarget(AttackableUnit target)\n"
_UNT_SIG_B = (
    "        public void Untarget(AttackableUnit target,\n"
    f"            [{_CMN}] string lanerlUntCaller = \"\",\n"
    f"            [{_CLN}] int lanerlUntLine = 0)\n"
)

_UNT_BODY_A = (
    "            if (TargetUnit == target)\n"
    "            {\n"
    "                SetTargetUnit(null, true);\n"
    "            }\n"
)
_UNT_BODY_B = (
    "            if (TargetUnit == target)\n"
    "            {\n"
    "                SetTargetUnit(null, true,\n"
    "                    \"Untarget<\" + lanerlUntCaller, lanerlUntLine);\n"
    "            }\n"
)

_STOP_SIG_A = "        public void StopTargeting(AttackableUnit target)\n"
_STOP_SIG_B = (
    "        public void StopTargeting(AttackableUnit target,\n"
    f"            [{_CMN}] string lanerlStopCaller = \"\",\n"
    f"            [{_CLN}] int lanerlStopLine = 0)\n"
)

_STOP_BODY_A = "                    ai.Untarget(target);\n"
_STOP_BODY_B = (
    "                    ai.Untarget(target,\n"
    "                        \"StopTargeting<\" + lanerlStopCaller, lanerlStopLine);\n"
)


# --- 6. Die: the timestamp the death-tick hypothesis needs -----------------
# `AttackableUnit.Die`'s FIRST statement is `ObjectManager.StopTargeting(this)`,
# which clears every pointer to the corpse during damage resolution -- inside
# the death tick, before any unit's next update. The port clears on the
# following tick. That is a one-tick phase difference and it is the shape of
# the residual (`TargetJustDied` enriched x739 over its base rate). Emitting
# the death itself makes the two timestamps directly comparable instead of
# leaving the argument to rest on the call graph.
_DIE_A = (
    "        public virtual void Die(DeathData data)\n"
    "        {\n"
    "            _game.ObjectManager.StopTargeting(this);\n"
)
_DIE_B = (
    "        public virtual void Die(DeathData data)\n"
    "        {\n"
    f"            {_TRACE}.Emit(\n"
    "                _game.GameTime, \"Die\", NetId,\n"
    "                \"kind=\" + GetType().Name\n"
    "                + \" killer=\" + (data == null || data.Killer == null\n"
    "                    ? 0u : data.Killer.NetId));\n"
    "            _game.ObjectManager.StopTargeting(this);\n"
)

_EDITS = [
    (_OBJAI, "SetTargetUnit signature", _TGT_SIG_A, _TGT_SIG_B, "lanerlTgtCaller"),
    (_OBJAI, "SetTargetUnit emit", _TGT_EMIT_A, _TGT_EMIT_B, "caller=\" + lanerlTgtCaller"),
    (_OBJAI, "UpdateMoveOrder signature", _ORD_SIG_A, _ORD_SIG_B, "lanerlOrdCaller"),
    (_OBJAI, "UpdateMoveOrder veto emit", _ORD_VETO_A, _ORD_VETO_B, "UpdateMoveOrderVetoed"),
    (_OBJAI, "UpdateMoveOrder was-capture", _ORD_WAS_A, _ORD_WAS_B, "lanerlOrdWas"),
    (_OBJAI, "UpdateMoveOrder accept emit", _ORD_EMIT_A, _ORD_EMIT_B, '" was=" + lanerlOrdWas'),
    (_OBJAI, "RefreshWaypoints entry emit", _REF_A, _REF_B, "\"RefreshWaypoints\""),
    (_OBJAI, "Untarget signature", _UNT_SIG_A, _UNT_SIG_B, "lanerlUntCaller"),
    (_OBJAI, "Untarget passthrough", _UNT_BODY_A, _UNT_BODY_B, '"Untarget<"'),
    (_OBJMGR, "StopTargeting signature", _STOP_SIG_A, _STOP_SIG_B, "lanerlStopCaller"),
    (_OBJMGR, "StopTargeting passthrough", _STOP_BODY_A, _STOP_BODY_B, '"StopTargeting<"'),
    (_UNIT, "Die emit", _DIE_A, _DIE_B, '"Die", NetId'),
    (_UNIT, "SetWaypoints signature", _WPT_SIG_A, _WPT_SIG_B, "lanerlWptCaller"),
    (_UNIT, "SetWaypoints reject/accept emit", _WPT_BODY_A, _WPT_BODY_B, "SetWaypointsRejected"),
]


def _apply(path: Path, label: str, old: str, new: str, marker: str) -> str:
    """
    Idempotent, exact-match replacement. Never guesses at whitespace.

    `marker` is matched against the WHOLE FILE, and `ObjAIBase.cs` carries
    three tagged methods, so the markers have to be unique per SITE and not
    merely per edit. A shared `lanerlCaller` made edits 3+ report "already
    present" after edit 1 landed -- leaving `UpdateMoveOrder` untagged while
    its emit referenced the parameter, which does not compile.
    """
    src = path.read_text()
    if marker in src:
        return f"  skip  {label}: already present"
    n = src.count(old)
    if n != 1:
        raise SystemExit(
            f"ANCHOR {label}: expected exactly 1 occurrence in {path.name}, found {n}.\n"
            "The vendored source moved. Re-derive the anchor rather than loosening it."
        )
    path.write_text(src.replace(old, new))
    return f"  ADD   {label}"


def verify() -> int:
    checks = [(p, lab, mk) for p, lab, _o, _n, mk in _EDITS]
    bad = 0
    for path, label, marker in checks:
        ok = path.exists() and marker in path.read_text()
        print(f"  {'OK  ' if ok else 'MISS'}  {label}")
        bad += 0 if ok else 1
    print(f"{len(checks) - bad}/{len(checks)} checks OK")
    return 1 if bad else 0


def uninstall() -> None:
    """Exact reverse. Needed because the vendored tree is not ours to commit,
    so `git checkout` is not available as an undo."""
    for path, label, old, new, _marker in _EDITS:
        src = path.read_text()
        if new in src:
            path.write_text(src.replace(new, old))
            print(f"  DEL   {label}")
        else:
            print(f"  skip  {label}: not present")


def main() -> None:
    if "--verify" in sys.argv:
        raise SystemExit(verify())
    if "--uninstall" in sys.argv:
        return uninstall()
    for path, label, old, new, marker in _EDITS:
        if not path.exists():
            raise SystemExit(f"missing vendored source: {path}")
        print(_apply(path, label, old, new, marker))
    print("\nNow rebuild -- see lanerl/patch_observability.py's docstring.")


if __name__ == "__main__":
    main()
