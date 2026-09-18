"""Did the server actually load its AI scripts?

THE FAILURE THIS CATCHES
------------------------
`Content/LeagueSandbox-Scripts` is Roslyn-compiled at boot, and a script that
fails to compile does NOT stop the server. `CSharpScriptEngine` logs the
compile errors, `ContentManager` logs

    Loaded some C# scripts from package: LeagueSandbox-Scripts

instead of the usual ``Loaded all``, and the game proceeds **without that
script**. Measured directly: a deliberately broken `LaneMinionAI.cs` produced a
complete, well-formed 122-snapshot state dump from a server whose lane minions
had no AI whatsoever.

Nothing else in this harness would notice. The trace parses, the hashes are
self-consistent, the entity counts are plausible, and every parity number
computed from it is garbage. It is `INJ-003`'s shape -- a disabled controller
rather than a perturbed value -- which is the class of bug this project has now
been bitten by twice, both times invisible to every field-level agreement rate.

The tell is one word, so check for it rather than trusting that a recording
that ran is a recording that was valid.

WHY A SCRIPT WOULD BE BROKEN IN THE FIRST PLACE
-----------------------------------------------
Because instrumenting the minion AI is worth doing: `LaneMinionAI.OnUpdate`'s
trigger has three arms -- `TargetJustDied()`, `FoundNewTarget(true)` and
`minionActionTimer >= 250f` -- which all produce the same observable, so which
one fired is unrecoverable from state. Editing scripts is therefore a thing
this project will keep wanting to do, and `Content/` is shared by every build
(it is resolved through `CONTENT_PATH`, or executable-relative when that path
does not exist -- see `parity/decision_trace.py` for the isolation recipe).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

__all__ = ["ScriptLoadStatus", "check_script_load", "assert_all_scripts_loaded"]

_ALL = "Loaded all C# scripts from package"
_SOME = "Loaded some C# scripts from package"
_ERR = "Script compilation error in script"


class ScriptLoadStatus(tuple):
    """`(ok, packages_partial, compile_errors)`."""

    @property
    def ok(self) -> bool:
        return self[0]

    @property
    def partial(self) -> List[str]:
        return self[1]

    @property
    def errors(self) -> List[str]:
        return self[2]


def check_script_load(log: Path | str) -> ScriptLoadStatus:
    """Scan a server log for partial script loads.

    Returns ok=False if any package loaded partially, listing the offending
    package lines and the distinct scripts that failed to compile.
    """
    partial: List[str] = []
    errors: List[str] = []
    saw_all = False
    with Path(log).open(errors="replace") as fh:
        for line in fh:
            if _SOME in line:
                partial.append(line.strip())
            elif _ALL in line:
                saw_all = True
            elif _ERR in line:
                # "...in script <path>: CS1234" -- keep the path, drop the code,
                # so one broken file does not look like six failures.
                tail = line.split(_ERR, 1)[1].strip()
                path = tail.rsplit(":", 1)[0].strip()
                if path not in errors:
                    errors.append(path)
    return ScriptLoadStatus((not partial and saw_all, partial, errors))


def assert_all_scripts_loaded(log: Path | str) -> None:
    """Raise unless every script package loaded completely.

    Call this on any recording whose numbers will be quoted. A recording made
    against a partially-loaded script package is not a weaker measurement, it
    is a measurement of a different game.
    """
    st = check_script_load(log)
    if st.ok:
        return
    detail = ""
    if st.errors:
        detail = "\n  failed to compile:\n    " + "\n    ".join(st.errors)
    if not st.partial and not st.errors:
        detail = ("\n  no 'Loaded all/some' line at all -- this log may be "
                  "truncated, or from a server version that predates the "
                  "message, so the check could not be made.")
    raise AssertionError(
        f"server did not load all its C# scripts: {log}\n"
        f"  {' / '.join(st.partial) if st.partial else '(no partial-load line)'}"
        f"{detail}\n"
        "  A partially-loaded script package still produces a complete, "
        "well-formed, entirely invalid trace -- see this module's docstring.")


if __name__ == "__main__":
    import sys
    bad = 0
    for arg in sys.argv[1:]:
        st = check_script_load(arg)
        print(f"{'OK  ' if st.ok else 'BAD '} {arg}"
              + ("" if st.ok else f"  errors={st.errors} partial={len(st.partial)}"))
        bad += not st.ok
    sys.exit(1 if bad else 0)
