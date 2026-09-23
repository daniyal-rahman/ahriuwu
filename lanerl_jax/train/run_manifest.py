"""Checkpoints that a stranger can navigate, and a manifest that says what a run WAS.

WHY THIS EXISTS
---------------
Before this, a training run produced a console log and nothing else. No
checkpoint, no config on disk, no provenance. Three concrete costs already paid:

* a node failure killed a 45-minute run at update ~450 and there was nothing to
  resume from;
* `PPOConfig`'s docstring says its defaults come from
  `runs/rl-league-0915e/resolved_config.json` -- a run whose config file is the
  only record of what it did, and which nothing in this repo can regenerate;
* the gate-4 throughput figure came from a script that was never committed, so
  it went stale without anyone noticing (`JAX_FIDELITY_LEDGER.md` says so).

So every run writes a directory that answers, without reference to a shell
history or a chat log: what code ran, on what hardware, with what config, what
it scored, and which checkpoint is which.

LAYOUT
------
    runs/<run_id>/
        manifest.json      config + provenance + results, written at start and
                           updated at every checkpoint
        README.md          the same thing in prose, for a human with no context
        ckpt_<step>.msgpack  flax-serialised params + opt_state
        ckpt_latest.msgpack  a copy of the newest, so resume needs no globbing
        metrics.jsonl      one line per chunk: every logged scalar

`run_id` is `<tag>-<YYYYmmdd-HHMMSS>-<git sha>`, so two runs of the same tag
never collide and the sha is visible without opening anything.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

__all__ = ["RunDir", "git_provenance"]


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:
        return "?"


def _worktree_git_env() -> dict:
    """A `git` environment that works on a node which cannot see `/srv/nfs`.

    This checkout is a WORKTREE, so `.git` is a file holding
    `gitdir: /srv/nfs/projects/ahriuwu/.git/worktrees/ahriuwu-lanerl-jax`. The
    Slurm GPU node mounts the same filesystem at `/mnt/nfs` and has no `/srv`
    at all, so every git command there failed with "not a git repository" --
    and `_run` returns "?" on failure, so the run id came out as
    `gpusmoke-20260923-050059-` with an empty sha and the manifest recorded
    `dirty: false` for a tree it had not looked at. That is worse than no
    provenance, because it reads as a clean checkout.

    Returns extra environment for the git subprocess, or `{}` if the ordinary
    path already works.
    """
    if _run(["git", "rev-parse", "--git-dir"]) not in ("?", ""):
        return {}
    dotgit = Path(".git")
    if not dotgit.is_file():
        return {}
    ref = dotgit.read_text().strip()
    if not ref.startswith("gitdir:"):
        return {}
    target = ref.split(":", 1)[1].strip()
    # The ONLY translation, and it is the one the cluster actually needs
    # (`docs`/memory: `/srv/nfs` on the login node is `/mnt/nfs` on `desktop`).
    # Applied only when the recorded path is absent and the translated one is
    # present, so it cannot silently point at a different repository.
    if not Path(target).exists():
        alt = target.replace("/srv/nfs/", "/mnt/nfs/", 1)
        if Path(alt).exists():
            target = alt
        else:
            return {}
    return {"GIT_DIR": target, "GIT_WORK_TREE": str(Path.cwd())}


def source_fingerprint(root: Path | None = None) -> dict:
    """sha256 over the simulator and trainer sources that actually ran.

    Independent of git, deliberately. A sha plus a dirty flag cannot tell two
    dirty trees apart, and this project has already had a GPU node execute
    PRE-EDIT source over NFS while the sha said otherwise. Two runs whose
    `files` count and `sha256` agree ran the same Python; if they disagree, the
    manifest says so even when git is unavailable on the node.
    """
    import hashlib

    root = root or Path.cwd()
    h, n = hashlib.sha256(), 0
    for f in sorted((root / "lanerl_jax").rglob("*.py")):
        if "__pycache__" in f.parts or "/tests/" in str(f):
            continue
        h.update(str(f.relative_to(root)).encode())
        h.update(f.read_bytes())
        n += 1
    return {"sha256": h.hexdigest()[:16], "files": n,
            "root": str(root), "note": "lanerl_jax/**/*.py excluding tests"}


def git_provenance() -> dict:
    """What code actually ran.

    `dirty` matters more than `sha` here: this project has already committed a
    tree that a concurrent `git checkout` had reverted underneath it, so a run
    whose sha looks right can still have executed different source. A dirty run
    is not reproducible from the sha alone and the manifest says so out loud.
    """
    env = _worktree_git_env()
    if env:
        os.environ.update(env)
    sha = _run(["git", "rev-parse", "HEAD"])
    dirty = _run(["git", "status", "--porcelain"])
    return {
        "sha": sha,
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(dirty),
        # Split on whitespace, do NOT slice a fixed 3 characters. `_run`
        # applies `.strip()`, which eats the leading space of porcelain's first
        # line (" M path" -> "M path"), so a fixed [3:] cut one character too
        # many off the FIRST entry only -- it recorded
        # "anerl_jax/train/run_train.py". A provenance record with a subtly wrong
        # filename is worse than none.
        "dirty_files": [l.split(maxsplit=1)[-1]
                        for l in dirty.splitlines() if l.strip()][:40],
        "describe": _run(["git", "describe", "--always", "--dirty"]),
        # Empty unless the worktree gitdir had to be path-translated for this
        # node; present means git was NOT readable at the recorded path.
        "gitdir_translated": env.get("GIT_DIR", ""),
        "source": source_fingerprint(),
    }


def _jsonable(v: Any) -> Any:
    if is_dataclass(v) and not isinstance(v, type):
        return {k: _jsonable(x) for k, x in asdict(v).items()}
    if hasattr(v, "_asdict"):                      # NamedTuple
        return {k: _jsonable(x) for k, x in v._asdict().items()}
    if isinstance(v, Mapping):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return repr(v)


class RunDir:
    """One run's directory, its manifest, and its checkpoints."""

    def __init__(self, root: Path, tag: str, config: Mapping[str, Any],
                 notes: str = ""):
        prov = git_provenance()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.run_id = f"{tag}-{stamp}-{prov['sha'][:8]}"
        self.path = Path(root) / self.run_id
        self.path.mkdir(parents=True, exist_ok=True)
        self.manifest = {
            "run_id": self.run_id,
            "tag": tag,
            "notes": notes,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git": prov,
            "host": {
                "hostname": socket.gethostname(),
                "python": platform.python_version(),
                "slurm_job": os.environ.get("SLURM_JOB_ID"),
                "gpu": _run(["nvidia-smi", "--query-gpu=name",
                             "--format=csv,noheader"]),
            },
            "config": _jsonable(config),
            "checkpoints": [],
            "results": {},
        }
        self._metrics = (self.path / "metrics.jsonl").open("a")
        self.write()

    # ---- manifest ---------------------------------------------------------
    def write(self) -> None:
        (self.path / "manifest.json").write_text(
            json.dumps(self.manifest, indent=2, sort_keys=True))
        (self.path / "README.md").write_text(self._readme())

    def _readme(self) -> str:
        m = self.manifest
        g, h = m["git"], m["host"]
        cfg = json.dumps(m["config"], indent=2, sort_keys=True)
        res = json.dumps(m["results"], indent=2, sort_keys=True)
        cks = "\n".join(f"  - `{c['file']}` at step {c['step']}"
                        f" (update {c['update']})" for c in m["checkpoints"]) or "  - none yet"
        return f"""# {m['run_id']}

{m['notes'] or '_no notes given_'}

Started `{m['started_utc']}` on `{h['hostname']}`{f" (slurm job {h['slurm_job']})" if h['slurm_job'] else ''}
GPU: {h['gpu'] or 'none'} · Python {h['python']}

## What code ran

- branch `{g['branch']}` at `{g['sha']}` (`{g['describe']}`)
- working tree **{'DIRTY -- this run is NOT reproducible from the sha alone' if g['dirty'] else 'clean'}**
{chr(10).join(f'  - modified: `{f}`' for f in g['dirty_files']) if g['dirty'] else ''}

## Config

```json
{cfg}
```

## Results

```json
{res}
```

## Checkpoints

{cks}

`ckpt_latest.msgpack` is a copy of the newest one, so resuming never needs a glob.

## Reproducing

```bash
git checkout {g['sha']}
sbatch slurm/rl_train.sbatch --tag {m['tag']} \\
{chr(10).join(f"    --{k.replace('_','-')} {v}" for k, v in sorted(m['config'].get('cli', {}).items()))}
```

Metrics: one JSON object per chunk in `metrics.jsonl`.
"""

    def log(self, row: Mapping[str, Any]) -> None:
        self._metrics.write(json.dumps(_jsonable(row)) + "\n")
        self._metrics.flush()

    def set_results(self, **kv) -> None:
        self.manifest["results"].update(_jsonable(kv))
        self.write()

    # ---- checkpoints ------------------------------------------------------
    #: How many timestamped checkpoints to keep besides `ckpt_latest`. They are
    #: ~55 MiB each, so an unrotated run at `--ckpt-every 1` fills the disk: a
    #: 600-update run at chunk 20 would write 30 of them, 1.6 GiB, for no
    #: benefit over the last few.
    keep_checkpoints: int = 3

    def save(self, step: int, update: int, payload: Any) -> Path:
        from flax.serialization import to_bytes
        f = self.path / f"ckpt_{step:09d}.msgpack"
        f.write_bytes(to_bytes(payload))
        (self.path / "ckpt_latest.msgpack").write_bytes(f.read_bytes())
        self.manifest["checkpoints"].append(
            {"file": f.name, "step": int(step), "update": int(update),
             "bytes": f.stat().st_size,
             "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        # rotate: keep the newest `keep_checkpoints`, always keep ckpt_latest
        kept = self.manifest["checkpoints"]
        if self.keep_checkpoints and len(kept) > self.keep_checkpoints:
            for old in kept[:-self.keep_checkpoints]:
                victim = self.path / old["file"]
                if victim.exists():
                    victim.unlink()
                old["pruned"] = True
            self.manifest["checkpoints"] = (
                [c for c in kept if c.get("pruned")][-2:]
                + kept[-self.keep_checkpoints:])
        self.write()
        return f

    def close(self) -> None:
        self.manifest["finished_utc"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.write()
        self._metrics.close()
