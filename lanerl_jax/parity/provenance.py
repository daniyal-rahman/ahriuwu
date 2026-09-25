"""What code did this measurement actually run?

WHY THIS EXISTS
---------------
Every fidelity claim in `docs/JAX_FIDELITY_LEDGER.md` has the same shape: run
the corpus, change one thing, run it again, attribute the delta to the thing
that changed. That argument has one silent failure mode -- the "after" run
executing the "before" code -- and **no field-level agreement rate can detect
it**, because a run that silently used the old source produces a perfectly
self-consistent set of numbers. It just produces the previous run's numbers,
and the honest conclusion "the fix did nothing" is indistinguishable from the
dishonest one "the fix was never loaded".

This happened once, observed directly: a test run reconstructed a value from
the pre-fix constant while the simulator in the same process used the post-fix
one, and an identical rerun minutes later passed. The mechanism was not
established -- NFS attribute caching was the obvious suspect and was tested
and **refuted** (a fresh process on the compute node sees new content at t+0
at every delay measured). So this module does not try to prevent the cause.
It makes the *effect* impossible to miss.

`INJ-003` is the precedent that makes this worth a module rather than a
comment: an injector bug disabled an entire controller for every Tier-1
minion number ever taken, and was invisible for exactly the same reason --
the thing that was wrong was upstream of everything being measured.

WHAT IT FINGERPRINTS
--------------------
Not the files on disk. The **bytecode Python actually loaded**, reached
through the imported module objects, plus the live values of the named
constants. A stale `__pycache__` entry, a shadowed module earlier on
`sys.path`, an editable install pointing somewhere unexpected and a compute
node reading a different tree all change this digest; re-reading the source
files would catch only the last of them.

USE
---
Print it at the top of any run whose numbers will be quoted, and put the
digest next to the number in the ledger::

    from lanerl_jax.parity.provenance import print_provenance
    print_provenance()

Two runs that are supposed to differ by one fix **must** differ here. Two
runs that are supposed to be identical must match here. If a before/after
pair shows the same digest, the comparison is void -- do not reason about
the numbers, re-run it.
"""
from __future__ import annotations

import hashlib
import importlib
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

__all__ = ["MODULES", "CONSTANTS", "module_digest", "constant_values",
          "provenance", "format_provenance", "print_provenance"]

#: The modules whose behaviour a fidelity number depends on. Deliberately
#: explicit rather than "everything imported": a digest that moves when an
#: unrelated module changes is a digest nobody trusts, and a digest nobody
#: trusts is worse than none.
MODULES: Tuple[str, ...] = (
    "lanerl_jax.sim.step",
    "lanerl_jax.sim.combat",
    "lanerl_jax.sim.orders",
    "lanerl_jax.sim.minion_ai",
    "lanerl_jax.sim.autoattack",
    "lanerl_jax.sim.missiles",
    "lanerl_jax.sim.movement",
    "lanerl_jax.sim.collision",
    "lanerl_jax.sim.targeting",
    "lanerl_jax.sim.profiles",
    "lanerl_jax.sim.init",
    "lanerl_jax.parity.inject",
    "lanerl_jax.parity.archive.one_step",
)

#: Live constants worth printing verbatim, because a digest tells you
#: *that* something moved and these tell you *what*. Each one has been wrong
#: at least once; see `STAT-001`, `STAT-002`, `INJ-003`.
CONSTANTS: Tuple[Tuple[str, str], ...] = (
    ("lanerl_jax.sim.init", "MASTERY_HP_FLAT_BONUS"),
    ("lanerl_jax.sim.init", "MASTERY_HP_PERCENT_BONUS"),
    ("lanerl_jax.sim.init", "MASTERY_AD_PER_LEVEL_BONUS"),
    ("lanerl_jax.sim.init", "RUNE_AD_BONUS"),
    ("lanerl_jax.sim.init", "RUNE_ARMOR_BONUS"),
)


def _code_objects(obj: Any, seen: set) -> Iterable[types.CodeType]:
    """Every code object reachable from a module, nested functions included."""
    if id(obj) in seen:
        return
    seen.add(id(obj))
    code = getattr(obj, "__code__", None)
    if code is not None:
        yield code
        for const in code.co_consts:
            if isinstance(const, types.CodeType):
                yield const
    if isinstance(obj, type):
        for name in sorted(vars(obj)):
            yield from _code_objects(vars(obj)[name], seen)


def module_digest(names: Iterable[str] = MODULES) -> Tuple[str, Dict[str, str]]:
    """`(combined, per_module)` over the bytecode that is loaded right now.

    Hashes `co_code` plus the repr of `co_consts`, so a changed literal (the
    usual shape of a fidelity fix -- see `STAT-002`'s 3.5 -> 4.05) moves the
    digest even though the instruction stream is byte-identical.
    """
    per: Dict[str, str] = {}
    for name in names:
        mod = importlib.import_module(name)
        h = hashlib.sha256()
        seen: set = set()
        for attr in sorted(vars(mod)):
            value = vars(mod)[attr]
            if isinstance(value, types.ModuleType):
                continue            # imports, not this module's own behaviour
            if getattr(value, "__module__", name) != name:
                continue            # re-exported from elsewhere; hashed there
            for code in _code_objects(value, seen):
                h.update(code.co_code)
                h.update(repr(code.co_consts).encode())
        # Module-level scalars are NOT inside any function's `co_consts` --
        # they are globals, looked up by name at call time. A fidelity fix is
        # very often exactly one of these (`STAT-002` is `3.5 -> 4.05`), so a
        # digest over bytecode alone would sit still through the change it
        # most needs to catch. Hash the UPPER_CASE scalars too.
        for attr in sorted(vars(mod)):
            if not attr.isupper():
                continue
            value = vars(mod)[attr]
            if isinstance(value, (int, float, bool, str, bytes)) or (
                    isinstance(value, tuple)
                    and all(isinstance(v, (int, float, bool, str, bytes))
                            for v in value)):
                h.update(f"{attr}={value!r}".encode())
        per[name] = h.hexdigest()[:12]
    combined = hashlib.sha256(
        "".join(f"{k}:{per[k]}" for k in sorted(per)).encode()).hexdigest()[:12]
    return combined, per


def constant_values(pairs: Iterable[Tuple[str, str]] = CONSTANTS
                    ) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for mod_name, attr in pairs:
        try:
            out[f"{mod_name.split('.')[-1]}.{attr}"] = getattr(
                importlib.import_module(mod_name), attr)
        except AttributeError:
            # A constant that has been renamed or removed is itself news.
            out[f"{mod_name.split('.')[-1]}.{attr}"] = "<absent>"
    return out


#: The login node exports `/srv/nfs`; the compute node mounts the same export
#: at `/mnt/nfs`. Any absolute path baked into a file therefore resolves on
#: exactly one of the two hosts.
_MOUNT_ALIASES = (("/srv/nfs/", "/mnt/nfs/"), ("/mnt/nfs/", "/srv/nfs/"))


def _resolve_git_dir(start: Path) -> Path | None:
    """Find a usable `--git-dir`, translating between the two mount paths.

    This tree is a **worktree**, so its `.git` is a file containing an
    absolute `gitdir:` pointer into the main checkout -- written on the login
    node as `/srv/nfs/...`, which does not exist on the compute node, where
    the same bytes are reachable only under `/mnt/nfs/...`. Plain `git
    rev-parse` there fails with "not a git repository" naming a path that is
    genuinely absent.
    """
    for parent in (start, *start.parents):
        dot = parent / ".git"
        if dot.is_dir():
            return dot
        if dot.is_file():
            text = dot.read_text().strip()
            if not text.startswith("gitdir:"):
                return None
            target = Path(text.split(":", 1)[1].strip())
            if target.exists():
                return target
            for src, dst in _MOUNT_ALIASES:
                if str(target).startswith(src):
                    alt = Path(str(target).replace(src, dst, 1))
                    if alt.exists():
                        return alt
            return None
    return None


def _git(*args: str) -> str:
    """`-c safe.directory=*` because the compute node reads the repo over NFS
    under a different mount path, which trips git's dubious-ownership check.
    That check exits non-zero with an *empty stdout* -- it fails silently into
    looking like "no git here", which is the one answer this module must never
    quietly give. Every failure below is returned as text, never as "".
    """
    here = Path(__file__).resolve().parent
    git_dir = _resolve_git_dir(here)
    cmd: List[str] = ["git", "-c", "safe.directory=*"]
    if git_dir is not None:
        cmd += ["--git-dir", str(git_dir), "--work-tree", str(here.parent.parent)]
    cmd += list(args)
    try:
        r = subprocess.run(cmd, cwd=here, capture_output=True, text=True,
                           timeout=10)
        if r.returncode != 0:
            first = (r.stderr.strip().splitlines() or ["(no stderr)"])[0]
            return f"<git failed: {first}>"
        return r.stdout.strip()
    except Exception as exc:
        return f"<unavailable: {type(exc).__name__}: {exc}>"


def provenance() -> Dict[str, Any]:
    combined, per = module_digest()
    return {
        "bytecode_digest": combined,
        "per_module": per,
        "constants": constant_values(),
        "git_head": _git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "package_path": str(Path(
            importlib.import_module("lanerl_jax").__file__).parent),
        "python": sys.version.split()[0],
        "host": _hostname(),
    }


def _hostname() -> str:
    import socket
    return socket.gethostname()


def format_provenance(p: Dict[str, Any] | None = None) -> str:
    p = p or provenance()
    lines = [
        "--- provenance -------------------------------------------------",
        f"  bytecode digest : {p['bytecode_digest']}"
        f"   (git {p['git_head']}{'+dirty' if p['git_dirty'] else ''})",
        f"  package         : {p['package_path']}  on {p['host']}",
    ]
    for k, v in p["constants"].items():
        lines.append(f"  {k:<44} = {v}")
    lines.append(
        "  A before/after pair MUST differ in the digest above. If it does")
    lines.append(
        "  not, the comparison is void -- re-run it, do not interpret it.")
    lines.append(
        "----------------------------------------------------------------")
    return "\n".join(lines)


def print_provenance(file=None) -> Dict[str, Any]:
    p = provenance()
    print(format_provenance(p), file=file or sys.stdout, flush=True)
    return p


if __name__ == "__main__":
    print_provenance()
