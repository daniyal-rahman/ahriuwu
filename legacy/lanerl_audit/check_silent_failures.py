#!/usr/bin/env python
"""Mechanical pre-flight gate for the silent-failure classes that have already
burned this project.

Every check here exists because the corresponding failure actually happened, not
because a linter has a rule for it.  Run it before an experiment; it exits
non-zero if anything it can decide mechanically is wrong.

    /home/dani/miniconda3/envs/ml/bin/python lanerl_audit/check_silent_failures.py
    ... --severity high      # gate on the burn-hours tier only
    ... --json report.json   # machine-readable

Exit codes:  0 clean, 1 findings at or above the requested severity, 2 the
checker itself could not run (which is itself a failure, not a pass).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Locations, resolved from this file so the checker is node-portable itself.
# /srv/nfs and /mnt/nfs are the SAME export seen from two different nodes.
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
PROJECTS = REPO.parent
VENDOR = PROJECTS / "lanerl-vendor"
SERVER = VENDOR / "LoLServer"
BIN = SERVER / "GameServerConsole/bin/Release/net6.0"
CONTENT = SERVER / "Content"
LANERL_CS = SERVER / "GameServerLib/Lanerl"

SCAN_DIRS = [REPO / "lanerl", REPO / "lanerl_rl", REPO / "lanerl_bot"]
SKIP_PARTS = {"__pycache__", ".git", "obj", "bin", "node_modules", ".pytest_cache",
              "lanerl_audit"}

SEVERITIES = {"high": 3, "medium": 2, "low": 1}


@dataclass
class Finding:
    severity: str
    check: str
    where: str
    what: str
    fix: str
    extra: dict = field(default_factory=dict)


FINDINGS: list[Finding] = []


def add(severity: str, check: str, where: str, what: str, fix: str, **extra) -> None:
    FINDINGS.append(Finding(severity, check, where, what, fix, extra))


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(PROJECTS))
    except ValueError:
        return str(p)


def walk(roots, suffixes):
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file() or p.suffix not in suffixes:
                continue
            if SKIP_PARTS & set(p.parts):
                continue
            yield p


def lines_of(p: Path) -> list[str]:
    try:
        return p.read_text(errors="replace").splitlines()
    except OSError:
        return []


# Measured fields that only run_server / replicate / sweep / measure_all write.
_MEASURED_KEYS = {"wall_s", "cs_rows", "cs_mean", "cs_at_600k", "cs_at_10min",
                  "crashes", "tps_median", "trials", "timed_out"}


def is_recorded_result(p: Path) -> bool:
    """True for a bench OUTPUT (a recorded run) rather than an input config.

    Decided by CONTENT, not by filename.  The old filename test
    (`startswith(("rep_", "measure_", "sweep_", "grid"))`) was wrong in both
    directions: `grid1.json` is an *input* to sweep.py --grid and was being
    downgraded, while `qab4.json`, `instr.json` and `reset_smoke.json` are
    results and were being treated as live inputs.

    The distinction matters because nothing in the tree ever *reads* a result
    file -- grep confirms only `--out` writes them.  A mount literal recorded in
    one is provenance (where a past run happened), not a path any job will try
    to open, so it cannot be the exit-53 failure.  It stays a finding, because
    the run it records is not reproducible from the other node; it is just not
    the burn-hours tier.
    """
    if p.suffix != ".json":
        return False
    try:
        doc = json.loads(p.read_text(errors="replace"))
    except (OSError, ValueError):
        return False

    def measured(v) -> bool:
        return isinstance(v, dict) and bool(_MEASURED_KEYS & set(v))

    if measured(doc):
        return True
    values = doc.values() if isinstance(doc, dict) else doc
    return isinstance(values, (list, type({}.values()))) and any(
        measured(v) for v in values)


# ---------------------------------------------------------------------------
# 1. Hardcoded NFS mount points.
#    /srv/nfs exists on danilogin, /mnt/nfs on desktop -- same export, and on
#    the wrong node the literal simply does not resolve.  Slurm dies with exit
#    53 and writes no output file at all when this is in --output.
# ---------------------------------------------------------------------------
MOUNT_RE = re.compile(r"(/srv/nfs|/mnt/nfs)(?![a-zA-Z0-9_])")
ALLOW_COMMENT = re.compile(r"^\s*(#|//|--)")


def check_hardcoded_mounts() -> None:
    for p in walk(SCAN_DIRS + [LANERL_CS], {".py", ".sh", ".sbatch", ".cs", ".json"}):
        for i, line in enumerate(lines_of(p), 1):
            m = MOUNT_RE.search(line)
            if not m:
                continue
            if ALLOW_COMMENT.match(line):
                continue  # the "never hardcode either" advisory comments
            if is_recorded_result(p):
                # a recorded run is inert data -- nothing reads it back, so the
                # literal cannot be the exit-53 failure.  See is_recorded_result().
                sev = "medium"
            else:
                sev = "high" if p.suffix == ".json" or "config" in line.lower() \
                    else "medium"
            add(sev, "hardcoded-mount", f"{rel(p)}:{i}",
                f"literal {m.group(1)} -- resolves on one Slurm node only",
                "derive from Path(__file__).resolve().parents[N] (py) or "
                "${BASH_SOURCE[0]} (sh); never write the mount point",
                line=line.strip()[:160])


# ---------------------------------------------------------------------------
# 2. Config paths that do not resolve.
#    Was: LanerlConfig.cs:161 `if (!IsNullOrEmpty(path) && File.Exists(path))`
#    -- a LANERL_BOT_CONFIG that did not resolve was IGNORED with no warning and
#    the bot ran built-in defaults, so an A/B compared a config to itself.
#    Load() now throws on a set-but-unresolvable path and run_server.check_bot_config()
#    raises before the first launch, so this is loud at run time.  It stays a
#    finding: a dangling path still means that arm cannot run, and every recorded
#    result carrying one was measured back when the fallback was silent.
# ---------------------------------------------------------------------------
# Match the env var itself rather than "any /srv/nfs path ending .json".  Bench
# configs now name their targets REPO-RELATIVE (a config must not carry a mount
# point), and the old absolute-only pattern would have skipped those entirely --
# turning this check, and preflight.sh step 5, silently vacuous.
CFG_RE = re.compile(
    r'"LANERL_BOT_CONFIG"\s*:\s*"([^"]+)"'          # JSON  {"LANERL_BOT_CONFIG": "..."}
    r"|LANERL_BOT_CONFIG\s*=\s*[\"']?([^\"'\s;)]+)"  # shell/py  LANERL_BOT_CONFIG=...
)


def check_bot_config_paths() -> None:
    roots = [REPO / "lanerl_bot", REPO / "lanerl", REPO / "lanerl_rl"]
    for p in walk(roots, {".json", ".sh", ".py"}):
        for i, line in enumerate(lines_of(p), 1):
            for groups in CFG_RE.findall(line):
                cand = next((g for g in groups if g), None)
                if cand is None or "$" in cand or "{" in cand:
                    continue  # a variable, not a path we can decide mechanically
                target = Path(cand)
                if not target.is_absolute():
                    target = REPO / target
                if target.exists():
                    continue
                # A RECORDED result is not an input: a dangling path there means
                # the result cannot be reproduced and, if the file never existed,
                # that arm measured the default bot.  Detected by content --
                # see is_recorded_result().
                is_result = is_recorded_result(p)
                add("medium" if is_result else "high", "config-silent-fallback",
                    f"{rel(p)}:{i}",
                    f"LANERL_BOT_CONFIG target does not resolve: {cand} "
                    f"(-> {target}). This arm cannot run the config it names."
                    + (" This is a recorded result, measured while LanerlConfig.Load() "
                       "still fell back silently: that arm ran the DEFAULT bot and the "
                       "number is not reproducible."
                       if is_result else
                       " LanerlConfig.Load() and run_server.check_bot_config() now both "
                       "raise on it, so the sweep dies at launch instead of quietly "
                       "measuring the default bot -- but the arm is still dead."),
                    "create the config, or retire the arm (see the _note in "
                    "bench/out/cfg_tune.json)",
                    path=cand)


# ---------------------------------------------------------------------------
# 3. Subprocess return codes that are never inspected.
# ---------------------------------------------------------------------------
POPEN_RE = re.compile(r"subprocess\.Popen\(")
RC_RE = re.compile(r"returncode|\.poll\(\)|check_returncode|check=True|check_call|check_output")


def check_unchecked_returncode() -> None:
    for p in walk(SCAN_DIRS, {".py"}):
        text = "\n".join(lines_of(p))
        if not POPEN_RE.search(text):
            continue
        if RC_RE.search(text):
            continue
        for i, line in enumerate(lines_of(p), 1):
            if POPEN_RE.search(line):
                add("high", "unchecked-returncode", f"{rel(p)}:{i}",
                    "subprocess.Popen(...) and nothing in this file ever reads "
                    "returncode/poll(): a server that dies in 1s is indistinguishable "
                    "from a clean 10-minute run",
                    "capture rc = proc.wait(); treat rc not in (0, -9, -15) OR an "
                    "implausibly short wall time as a hard error, not an empty result")


# ---------------------------------------------------------------------------
# 4. Failure-swallowing exception handlers.
# ---------------------------------------------------------------------------
BARE_EXC_RE = re.compile(r"^\s*except\s*(Exception\s*)?:\s*(#.*)?$")


def check_swallowed_exceptions() -> None:
    for p in walk(SCAN_DIRS, {".py"}):
        ls = lines_of(p)
        for i, line in enumerate(ls, 1):
            if not BARE_EXC_RE.match(line):
                continue
            body = [x.strip() for x in ls[i:i + 3] if x.strip()]
            if body and body[0] in {"pass", "continue"} and not body[0].startswith("#"):
                add("medium", "swallowed-exception", f"{rel(p)}:{i}",
                    "broad except with a silent pass/continue",
                    "narrow the exception type and log what was swallowed")

    # C#: `catch { }` with no body and no explanatory comment on the line
    for p in walk([LANERL_CS], {".cs"}):
        for i, line in enumerate(lines_of(p), 1):
            if re.search(r"catch\s*(\([^)]*\))?\s*\{\s*\}\s*$", line):
                add("medium", "swallowed-exception", f"{rel(p)}:{i}",
                    "C# catch {} with an empty, uncommented body -- a throwing "
                    "engine call degrades to a no-op that the caller records as done",
                    "log the exception, or set a failure flag the caller checks",
                    line=line.strip()[:160])


# ---------------------------------------------------------------------------
# 5. Shell scripts that continue past a failure, and the "prints DONE anyway"
#    pattern that shipped a stale binary for hours.
# ---------------------------------------------------------------------------
DONE_RE = re.compile(r'^\s*echo\s+.*\b(DONE|OK|PASS|SUCCESS|COMPLETE)\b', re.I)
PIPE_SWALLOW_RE = re.compile(r"\|\s*(tail|head|grep|wc)\b")


def check_shell_strictness() -> None:
    for p in walk(SCAN_DIRS, {".sh"}):
        ls = lines_of(p)
        head = "\n".join(ls[:12])
        has_e = re.search(r"set\s+-[a-z]*e", head) is not None
        has_pipefail = "pipefail" in head
        if not has_e:
            add("medium", "shell-no-errexit", f"{rel(p)}:1",
                "no `set -e`: a failing command does not stop the script, so later "
                "steps run against whatever the failed step left behind",
                "add `set -euo pipefail` (and an explicit `|| true` where a "
                "failure really is acceptable)")
        if not has_pipefail:
            add("medium", "shell-no-pipefail", f"{rel(p)}:1",
                "no `set -o pipefail`: `cmd | tail` reports tail's status, so a "
                "failing cmd is invisible",
                "add `set -o pipefail`")
        for i, line in enumerate(ls, 1):
            if DONE_RE.match(line) and not has_e:
                add("high", "unconditional-done", f"{rel(p)}:{i}",
                    "prints a success banner unconditionally in a script without "
                    "`set -e` -- this is the build-check bug verbatim",
                    "gate the banner on the real exit status, or add `set -e`",
                    line=line.strip()[:160])
            if PIPE_SWALLOW_RE.search(line) and not has_pipefail:
                add("medium", "pipeline-masks-failure", f"{rel(p)}:{i}",
                    "pipes into tail/head/grep/wc without pipefail: the producer's "
                    "exit status is discarded",
                    "set -o pipefail, or capture ${PIPESTATUS[0]}",
                    line=line.strip()[:160])


# ---------------------------------------------------------------------------
# 6. greps that can match neither success nor failure.
#    `dotnet build | grep -E "error CS|Build succeeded"` printed DONE while the
#    build was failing on a missing NuGet package.
# ---------------------------------------------------------------------------
GREP_RE = re.compile(r"grep[^|;&\n]*")


def check_ambiguous_grep() -> None:
    for p in walk(SCAN_DIRS, {".sh", ".py"}):
        for i, line in enumerate(lines_of(p), 1):
            if "grep" not in line:
                continue
            if ALLOW_COMMENT.match(line):
                continue
            g = GREP_RE.search(line)
            if not g:
                continue
            frag = g.group(0)
            # a grep used as a *test* that is not followed by an explicit
            # else-branch or `|| exit` on the same line
            if re.search(r"(-c|-l|-q)\b", frag) and not re.search(r"\|\||&&|exit|if\s", line):
                add("low", "ambiguous-grep", f"{rel(p)}:{i}",
                    "grep used as a check with no branch on its exit status: a "
                    "pattern that matches neither success nor failure reads as pass",
                    "assert on the positive marker AND on the absence of the "
                    "negative one, and fail when neither appears",
                    line=line.strip()[:160])


# ---------------------------------------------------------------------------
# 7. Port literals: parallel instances that all default to the same port.
#    Only the first binds; the rest die quietly and the run measures one
#    instance N times.
# ---------------------------------------------------------------------------
PORT_RE = re.compile(r"\b(?:port\s*[:=]\s*|--port[ =]|PORT=)\D{0,3}(\d{4,5})\b", re.I)


def check_port_collisions() -> None:
    seen: dict[int, list[str]] = {}
    for p in walk(SCAN_DIRS, {".py", ".sh"}):
        for i, line in enumerate(lines_of(p), 1):
            if ALLOW_COMMENT.match(line):
                continue
            for m in PORT_RE.finditer(line):
                port = int(m.group(1))
                if not (1024 <= port <= 65535):
                    continue
                seen.setdefault(port, []).append(f"{rel(p)}:{i}")
    for port, locs in sorted(seen.items()):
        if len(locs) > 1:
            add("high", "port-collision", locs[0],
                f"port {port} is a literal default in {len(locs)} places: "
                + ", ".join(locs)
                + " -- run two of these at once and all but the first die silently",
                "allocate a free port at run time "
                "(bind to 0 and read getsockname()[1]) or derive it from "
                "$SLURM_JOB_ID; never share a literal default")
        elif len(locs) == 1:
            add("low", "port-literal", locs[0],
                f"hardcoded port {port}: safe alone, collides under any parallel run",
                "take the port from an argument with no shared default")


# ---------------------------------------------------------------------------
# 8. Stale build artifacts.  The C# side is the one that already burned hours.
# ---------------------------------------------------------------------------
def _newest(root: Path, pattern: str, exclude=("obj", "bin")) -> tuple[float, Path | None]:
    best, bestp = 0.0, None
    if not root.exists():
        return best, bestp
    for p in root.rglob(pattern):
        if set(exclude) & set(p.parts):
            continue
        try:
            m = p.stat().st_mtime
        except OSError:
            continue
        if m > best:
            best, bestp = m, p
    return best, bestp


def check_stale_build() -> None:
    dll = BIN / "GameServerLib.dll"
    exe = BIN / "GameServerConsole"
    if not dll.exists():
        add("high", "missing-build", rel(dll),
            "the server library the benchmarks load does not exist",
            "build LoLServer (NOT lanerl-vendor/GameServer -- that is a different, "
            "Lanerl-free checkout)")
        return
    dll_m = dll.stat().st_mtime
    for sub in ("GameServerLib", "GameServerCore", "GameServerConsole"):
        newest, path = _newest(SERVER / sub, "*.cs")
        if path is not None and newest > dll_m:
            add("high", "stale-build", rel(path),
                f"source is newer than {rel(dll)} "
                f"({newest - dll_m:.0f}s) -- every run is measuring the OLD binary",
                "rebuild before the experiment; make the run refuse to start "
                "when any .cs is newer than the dll")
    if not exe.exists():
        add("medium", "missing-apphost", rel(exe),
            "the native apphost is missing but lanerl_rl/control_smoke.py invokes it "
            "directly (not via `dotnet X.dll`)",
            "rebuild, or switch control_smoke.py to `dotnet GameServerConsole.dll`")


def check_stale_content_scripts() -> None:
    """Content/*.cs is Roslyn-compiled at RUNTIME.

    A compile error there does not fail any build: CSharpScriptEngine.Load()
    drops the broken tree, Package.LoadScripts() returns *true* on SomeCompiled,
    and Spell.cs binds SpellScriptEmpty -- the ability silently does nothing.
    """
    stamp = REPO / "lanerl_audit/.content_compile_ok"
    newest, path = _newest(CONTENT, "*.cs")
    if path is None:
        add("high", "missing-content", rel(CONTENT),
            "no Content scripts found; the server will boot with no abilities",
            "check out the Content submodule")
        return
    if not stamp.exists():
        add("medium", "content-unverified", rel(path),
            "no record that these Roslyn scripts have ever been compiled on this "
            "checkout; a broken one degrades to SpellScriptEmpty with no build error",
            "run lanerl_audit/preflight.sh, which boots the server and greps for "
            "'Script compilation error'")
        return
    if newest > stamp.stat().st_mtime:
        add("high", "content-stale", rel(path),
            "a Content script changed since the last verified compile; if it does "
            "not compile the ability becomes a silent no-op (SpellScriptEmpty)",
            "re-run lanerl_audit/preflight.sh")


# ---------------------------------------------------------------------------
# 9. Tests that are invisible or that pass when the subject is absent.
# ---------------------------------------------------------------------------
def check_test_visibility() -> None:
    pyproject = REPO / "pyproject.toml"
    if pyproject.exists():
        text = pyproject.read_text(errors="replace")
        m = re.search(r"testpaths\s*=\s*(\[[^\]]*\])", text)
        if m:
            paths = re.findall(r'"([^"]+)"', m.group(1))
            for d in SCAN_DIRS:
                td = d / "tests"
                if not td.exists():
                    continue
                covered = any(str(td).startswith(str(REPO / p)) for p in paths)
                if not covered:
                    add("high", "tests-not-collected", f"{rel(pyproject)}",
                        f"testpaths={paths} does not include {rel(td)}: a bare "
                        "`pytest` at the repo root never collects these tests, so "
                        "their failures are invisible",
                        f'add "{td.relative_to(REPO)}" to testpaths')

    for p in walk(SCAN_DIRS, {".py"}):
        if "tests" not in p.parts and not p.name.startswith("test_") \
                and p.name != "conftest.py":
            continue
        for i, line in enumerate(lines_of(p), 1):
            if re.search(r"pytest\.skip\(", line) and not re.search(
                    r"allow_module_level", line):
                add("medium", "skip-hides-failure", f"{rel(p)}:{i}",
                    "pytest.skip() inside a fixture/test body: a crashed or "
                    "unresponsive server is reported as a SKIP and the suite exits 0",
                    "pytest.fail() when the subject was supposed to be available; "
                    "reserve skip for a genuinely absent prerequisite",
                    line=line.strip()[:160])
            if re.search(r"skipif\(\s*not\s+\w+\(\)", line):
                add("medium", "collection-time-skip", f"{rel(p)}:{i}",
                    "skipif evaluated at import time: if the build is missing at "
                    "collection, every server test skips and pytest exits 0 with a "
                    "green summary",
                    "make the absence of the build an explicit hard error in a "
                    "pre-flight step, so 'all skipped' can never read as 'all passed'")


# ---------------------------------------------------------------------------
# 10. Files that cannot even be imported (the NameError class).
# ---------------------------------------------------------------------------
def check_python_importable() -> None:
    import ast
    for p in walk(SCAN_DIRS, {".py"}):
        try:
            tree = ast.parse(p.read_text(errors="replace"), filename=str(p))
        except SyntaxError as e:
            add("high", "syntax-error", f"{rel(p)}:{e.lineno}",
                f"file does not parse: {e.msg}", "fix the syntax error")
            continue
        # module-level names used before any binding exists anywhere in the module
        bound: set[str] = set(dir(__builtins__)) | {
            "__file__", "__name__", "__doc__", "__package__"}
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for a in node.names:
                    bound.add((a.asname or a.name).split(".")[0])
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bound.add(node.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                bound.add(node.id)
            elif isinstance(node, ast.arg):
                bound.add(node.arg)
            elif isinstance(node, ast.alias):
                bound.add((node.asname or node.name).split(".")[0])
            elif isinstance(node, ast.ExceptHandler) and node.name:
                # `except OSError as exc` binds a bare str on the handler, NOT a
                # Name/Store node -- without this every `as` name reads as
                # undefined. This produced a false HIGH on env.py's connect
                # retry loop, which is correct code.
                bound.add(node.name)
            elif isinstance(node, ast.MatchAs) and node.name:
                bound.add(node.name)          # `case ... as y`, same blind spot
        reported: set[str] = set()
        for node in tree.body:
            at_module_level = not isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            for sub in ast.walk(node):
                if not (isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load)):
                    continue
                if sub.id in bound or sub.id in __builtins__.__dict__:
                    continue
                if sub.id in reported:
                    continue
                reported.add(sub.id)
                when = ("on import -- this module has never run"
                        if at_module_level else "the moment that function is called")
                add("high", "undefined-name", f"{rel(p)}:{sub.lineno}",
                    f"undefined name `{sub.id}`: raises NameError {when}",
                    "define it (the sibling files derive it from "
                    "Path(__file__).resolve().parents[N])")


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--severity", choices=list(SEVERITIES), default="low",
                    help="fail on findings at or above this level (default: low)")
    ap.add_argument("--json", type=Path, help="also write findings as JSON")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    checks = [
        check_python_importable,
        check_hardcoded_mounts,
        check_bot_config_paths,
        check_unchecked_returncode,
        check_swallowed_exceptions,
        check_shell_strictness,
        check_ambiguous_grep,
        check_port_collisions,
        check_stale_build,
        check_stale_content_scripts,
        check_test_visibility,
    ]
    for fn in checks:
        try:
            fn()
        except Exception as e:  # a checker that dies must not read as "clean"
            add("high", "checker-crashed", fn.__name__,
                f"{type(e).__name__}: {e}", "fix the checker; a crash is not a pass")

    order = {"high": 0, "medium": 1, "low": 2}
    FINDINGS.sort(key=lambda f: (order[f.severity], f.check, f.where))

    counts = {s: sum(1 for f in FINDINGS if f.severity == s) for s in SEVERITIES}
    if not args.quiet:
        print(f"silent-failure check   repo={rel(REPO)}   node={os.uname().nodename}")
        print("=" * 78)
        cur = None
        for f in FINDINGS:
            if f.severity != cur:
                cur = f.severity
                print(f"\n----- {cur.upper()} -----")
            print(f"[{f.check}] {f.where}")
            print(f"    {f.what}")
            print(f"    FIX: {f.fix}")
            if f.extra.get("line"):
                print(f"    >>> {f.extra['line']}")
        print("\n" + "=" * 78)
        print(f"high={counts['high']}  medium={counts['medium']}  low={counts['low']}")

    if args.json:
        args.json.write_text(json.dumps([asdict(f) for f in FINDINGS], indent=2))

    threshold = SEVERITIES[args.severity]
    failing = [f for f in FINDINGS if SEVERITIES[f.severity] >= threshold]
    if failing:
        if not args.quiet:
            print(f"FAIL: {len(failing)} finding(s) at severity >= {args.severity}")
        return 1
    if not args.quiet:
        print("PASS")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        print("checker aborted -- treat as FAIL, not as clean", file=sys.stderr)
        sys.exit(2)
