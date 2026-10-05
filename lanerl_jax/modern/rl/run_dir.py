"""A training run's directory: ``manifest.json`` (config, provenance, checkpoints, results),
``metrics.jsonl`` (one row per chunk) and rotated ``ckpt_<step>.msgpack`` plus ``ckpt_latest.msgpack``."""
from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import os
import platform
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

KEEP_CHECKPOINTS = 3


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True,
                              cwd=Path(__file__).parent).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "?"


def _jsonable(v: Any) -> Any:
    if hasattr(v, "_asdict"):
        v = v._asdict()
    if isinstance(v, Mapping):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v if isinstance(v, (str, int, float, bool)) or v is None else str(v)


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class RunDir:
    """``<root>/<tag>-<stamp>-<sha>`` (a numeric suffix if taken), created atomically."""

    def __init__(self, root: Path, tag: str, config: Mapping[str, Any], notes: str = ""):
        sha, dirty = _git("rev-parse", "HEAD"), _git("status", "--porcelain")
        base = f"{tag}-{time.strftime('%Y%m%d-%H%M%S')}-{sha[:8]}"
        self.run_id, k = base, 0
        while True:
            self.path = Path(root) / self.run_id
            try:
                self.path.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                k += 1
                self.run_id = f"{base}-{k}"
        self.manifest = {
            "run_id": self.run_id, "tag": tag, "notes": notes, "started_utc": _utc(),
            "git": {"sha": sha, "branch": _git("rev-parse", "--abbrev-ref", "HEAD"), "dirty": bool(dirty),
                    "dirty_files": [ln.split(maxsplit=1)[-1] for ln in dirty.splitlines() if ln.strip()][:40]},
            "host": {"hostname": socket.gethostname(), "python": platform.python_version(),
                     "slurm_job": os.environ.get("SLURM_JOB_ID"), "xla_flags": os.environ.get("XLA_FLAGS", "")},
            "packages": {p: _version(p) for p in ("jax", "jaxlib", "flax", "optax", "numpy")},
            "config": _jsonable(config), "checkpoints": [], "results": {}}
        self._metrics = (self.path / "metrics.jsonl").open("a")
        self.write()

    def write(self) -> None:
        (self.path / "manifest.json").write_text(json.dumps(self.manifest, indent=2, sort_keys=True))

    def log(self, row: Mapping[str, Any]) -> None:
        self._metrics.write(json.dumps(_jsonable(row)) + "\n")
        self._metrics.flush()

    def set_results(self, **kv) -> None:
        self.manifest["results"].update(_jsonable(kv))
        self.write()

    def save(self, step: int, update: int, payload: Any, *, latest: bool = True) -> Path:
        """Write a checkpoint; ``latest=False`` keeps it (e.g. a divergence post-mortem) without making it
        ``ckpt_latest``. Only the newest ``KEEP_CHECKPOINTS`` are kept."""
        from flax.serialization import to_bytes
        f = self.path / (f"ckpt_{step:09d}.msgpack" if latest else f"ckpt_{step:09d}_diverged.msgpack")
        f.write_bytes(to_bytes(payload))
        if latest:
            tmp = self.path / "ckpt_latest.msgpack.tmp"               # atomic for concurrent readers
            tmp.write_bytes(f.read_bytes())
            tmp.replace(self.path / "ckpt_latest.msgpack")
        kept = self.manifest["checkpoints"] + [{"file": f.name, "step": int(step), "update": int(update),
                                                 "latest": bool(latest), "utc": _utc()}]
        for old in kept[:-KEEP_CHECKPOINTS]:
            (self.path / old["file"]).unlink(missing_ok=True)
        self.manifest["checkpoints"] = kept[-KEEP_CHECKPOINTS:]
        self.write()
        return f

    def close(self) -> None:
        self.manifest["finished_utc"] = _utc()
        self.write()
        self._metrics.close()


def _version(pkg: str) -> str:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return "absent"
