"""Read the server's BRANCH stream, not its state stream.

WHY
---
`parity/trace.py` parses `LANERL_STATEROW`: what is true after a tick. That is
the right primitive for determinism and reset-leakage checks and the wrong one
for finding a porting bug, because one wrong branch propagates into every field
it touches and you are left reasoning backwards from consequences.

What that cost, measured on this project:

* `ORDER-005` -- `Spell.FinishCasting`'s tail runs
  `UpdateMoveOrder(OrderType.Hold, true)` for any non-`InstantCast` cast, and
  an auto-attack IS a cast. One unported line. In the state diff it presented
  as a **4,791-count** `move order` residual that was read as symmetric
  boundary jitter for weeks; it was neither symmetric (4,769 against 22) nor
  jitter. It took a dedicated agent a full day to isolate.
* `AA-002` -- the dump rounds windup to 1/1024 s and a blue melee minion's
  final windup tick has 2.67e-5 s left, so it publishes a flat 0. That surfaced
  as 1,041 `aa_hit` misses PLUS 1,126 `hp` misses PLUS parts of three boolean
  fields, and cost a day to collapse back to one mechanism.

The first run of the branch trace emitted **528 `FinishCasting` events, every
one of them `auto=True instant=False`** -- i.e. it states `ORDER-005`'s premise
outright, as one grep, from a 200 s idle recording.

THE SERVER SIDE
---------------
`GameServerLib/Lanerl/LanerlDecisionTrace.cs` in the vendored tree, gated on
`LANERL_DECISION_TRACE=1`. Call sites so far:

* `ObjAIBase.UpdateMoveOrder` -- every accepted write, emitted AFTER the script
  veto, so the trace records orders that actually took effect.
* `Spell.FinishCasting` -- with `auto=` and `instant=`, the flag that decides
  whether the move-order tail above runs at all.

Deliberately NOT a poller. Its sibling `LanerlHooks.TurretTrace` samples every
250 ms and reports a turret target that differs from the last sample, so it
gives transitions but never reasons, and cannot see a transition shorter than
its own interval -- which at a 16.67 ms tick is most of them. Reasons are the
whole point, so these emit from inside the branch.

BEHAVIOUR NEUTRALITY -- MEASURED, NOT ASSUMED
----------------------------------------------
A diagnostic that perturbs what it observes would move every parity number in
this repo for a reason that is not a bug. So the contract is that the state
stream is byte-identical with the trace **switched on**, and it is checked in
that form:

* 120 s idle, instrumented binary: canonical `d39f16cba113024717e8` / 639,542
  rows, reproduced with the trace OFF **and** with it ON.
* 200 s idle (a window that actually contains combat): OFF and ON both
  `ce6f118002541655e528` / 1,159,704 rows, with 54,253 branch events emitted
  alongside -- 53,725 `UpdateMoveOrder` and 528 `FinishCasting`.

The 120 s check alone would have been misleading and nearly was: minions spawn
at t=90 s at opposite ends of the lane and cannot meet before 120 s, so that
window contains **no combat at all** and exercised zero `FinishCasting` call
sites. A "verified neutral" claim from it would have shipped an instrument
whose second call site had never executed. Any future call site needs a window
that provably exercises it.

BUILDING IT
-----------
The vendored tree carries several uncommitted local patches that are not part
of this work, so this build deliberately does NOT overwrite the canonical
binary that every parity run uses::

    export DOTNET_ROOT=/mnt/nfs/projects/lanerl-vendor/dotnet
    export PATH=$DOTNET_ROOT:$PATH
    cd /mnt/nfs/projects/lanerl-vendor/LoLServer
    dotnet build GameServerConsole/GameServerConsole.csproj -c Release \
      -o GameServerConsole/bin/Trace/net6.0 \
      -p:SolutionDir=/mnt/nfs/projects/lanerl-vendor/LoLServer/

`bin/Trace/net6.0` is at the SAME DEPTH as `bin/Release/net6.0` on purpose: the
server resolves its Content tree relative to the executable, so a build placed
anywhere else boots and then dies in `ContentManager.GetDependenciesFromPackage`.
`-p:SolutionDir` is required because a post-build step copies `lib/` through
`$(SolutionDir)`, which is undefined when building the project rather than the
solution.

Point a run at it with `ServerLaunchSpec(server_dir=...)`.
"""
from __future__ import annotations

import argparse
import collections
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List

__all__ = ["Decision", "parse_decisions", "summarise", "main"]

_LINE = re.compile(
    r"LANERL_DECISION t=(?P<t>-?\d+) k=(?P<k>\w+) id=(?P<id>\d+)(?P<rest>.*)$")


@dataclass(slots=True)
class Decision:
    """One branch the server took."""

    t_ms: int
    kind: str
    net_id: int
    fields: Dict[str, str]


def parse_decisions(path: Path | str) -> List[Decision]:
    """Parse a server log's branch stream.

    Tolerant of the state stream being interleaved (it always is) and of the
    log4net timestamp prefix, which is wall-clock and therefore never part of
    a comparison.
    """
    out: List[Decision] = []
    with Path(path).open(errors="replace") as fh:
        for line in fh:
            m = _LINE.search(line)
            if m is None:
                continue
            fields = {}
            for tok in m.group("rest").split():
                if "=" in tok:
                    k, _, v = tok.partition("=")
                    fields[k] = v
            out.append(Decision(int(m.group("t")), m.group("k"),
                                int(m.group("id")), fields))
    return out


def summarise(decisions: List[Decision]) -> str:
    by_kind = collections.Counter(d.kind for d in decisions)
    lines = [f"{len(decisions)} branch events"]
    for kind, n in by_kind.most_common():
        lines.append(f"  {kind:<20} {n}")
        shapes = collections.Counter(
            " ".join(f"{k}={v}" for k, v in sorted(d.fields.items())
                     if k in ("order", "auto", "instant", "publish"))
            for d in decisions if d.kind == kind)
        for shape, m in shapes.most_common(8):
            if shape:
                lines.append(f"      {shape:<44} {m}")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("log", type=Path)
    a = ap.parse_args(argv)
    print(summarise(parse_decisions(a.log)))


if __name__ == "__main__":
    main()
