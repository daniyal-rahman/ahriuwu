"""Submitting training jobs, with log paths that are valid on the target node.

Slurm opens the ``-o``/``-e`` files *on the node that runs the job*, not on the
submitter.  Since the shared export is mounted at ``/srv/nfs`` on danilogin and
``/mnt/nfs`` on desktop, ``sbatch -w desktop -o /srv/nfs/...`` produces a job
that fails before it can print why.  Every path that crosses a node boundary
here goes through :func:`lanerl_train.paths.on_node`, which raises rather than
passing an untranslated path along.
"""

from __future__ import annotations

import logging
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from . import paths

__all__ = ["SlurmJob", "render_script", "submit"]

log = logging.getLogger("lanerl_train.slurm")


@dataclass
class SlurmJob:
    """One ``sbatch`` submission.

    ``log_dir`` is given in *local* terms and translated for ``node``; the
    command is rendered with a ``cd`` into the node's view of the repo for the
    same reason.
    """

    name: str
    node: str
    command: Sequence[str]
    log_dir: Path
    partition: str = "cpu"
    cpus_per_task: int = 16
    time_limit: Optional[str] = None
    env: Dict[str, str] = field(default_factory=dict)
    python: Optional[str] = None

    def log_path_on_node(self) -> Path:
        return paths.on_node(Path(self.log_dir) / f"{self.name}-%j.out", self.node)

    def workdir_on_node(self) -> Path:
        return paths.on_node(paths.repo_root(), self.node)

    def sbatch_args(self) -> List[str]:
        args = [
            "--job-name",
            self.name,
            "-p",
            self.partition,
            "-w",
            self.node,
            "-c",
            str(self.cpus_per_task),
            "-o",
            str(self.log_path_on_node()),
        ]
        if self.time_limit:
            args += ["-t", self.time_limit]
        return args


def render_script(job: SlurmJob) -> str:
    """The job script body.

    ``set -euo pipefail`` is not decoration: an unchecked return code inside a
    job is exactly the silent failure this project keeps paying for.
    """
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(job.workdir_on_node()))}",
        f"export PYTHONPATH={shlex.quote(str(job.workdir_on_node()))}:${{PYTHONPATH:-}}",
    ]
    for k, v in sorted(job.env.items()):
        lines.append(f"export {k}={shlex.quote(str(v))}")
    lines.append('echo "node=$(hostname) job=${SLURM_JOB_ID:-none} started=$(date -Is)"')
    cmd = list(job.command)
    if job.python:
        cmd = [job.python] + cmd
    lines.append(" ".join(shlex.quote(c) for c in cmd))
    lines.append('echo "finished=$(date -Is) rc=$?"')
    return "\n".join(lines) + "\n"


def submit(job: SlurmJob, script_dir: Optional[Path] = None, dry_run: bool = False) -> str:
    """Write the script and ``sbatch`` it.  Returns the job id, or the script path."""
    d = Path(script_dir) if script_dir else Path(job.log_dir)
    d.mkdir(parents=True, exist_ok=True)
    script = d / f"{job.name}.sbatch"
    script.write_text(render_script(job))
    script.chmod(0o755)
    argv = ["sbatch", *job.sbatch_args(), str(paths.on_node(script, job.node))]
    if dry_run:
        log.info("dry run: %s", " ".join(shlex.quote(a) for a in argv))
        return str(script)
    proc = subprocess.run(argv, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"sbatch failed (rc={proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
        )
    log.info("submitted: %s", proc.stdout.strip())
    return proc.stdout.strip()
