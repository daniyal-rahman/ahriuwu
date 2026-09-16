"""Where the vendored server lives, resolved from this file rather than typed.

The two nodes disagree about the mount point
--------------------------------------------
``danilogin`` has ``/mnt/nfs`` as a symlink to ``/srv/nfs``; the compute node
has only ``/mnt/nfs``.  So an absolute literal works on exactly one of them,
and a Slurm job that resolved its repo root from a hardcoded path is a failure
this project has already had -- ``slurm_lanerl_train.sbatch``'s own comment
records a job resolving ``REPO`` to ``/var/spool/slurmd`` because ``sbatch``
copies the batch script to the node's spool.

``lanerl_train.paths`` solves it by walking up from ``__file__``, and this
mirrors that: the vendor tree is a sibling of the repo, so resolving relative
to this module works under either mount.

``LANERL_VENDOR_ROOT`` overrides it, for a tree that is not a sibling.
"""
from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "package_root", "repo_root", "projects_root", "vendor_root",
    "content_root", "ngrid_path",
]


def package_root() -> Path:
    """``.../lanerl_jax``."""
    return Path(__file__).resolve().parent.parent


def repo_root() -> Path:
    return package_root().parent


def projects_root() -> Path:
    """The directory holding both this repo and ``lanerl-vendor``."""
    return repo_root().parent


def vendor_root() -> Path:
    env = os.environ.get("LANERL_VENDOR_ROOT")
    return Path(env) if env else projects_root() / "lanerl-vendor"


def content_root() -> Path:
    return vendor_root() / "LoLServer/Content/LeagueSandbox-Default"


def ngrid_path(map_id: int = 1) -> Path:
    return content_root() / f"AIMesh/Map{map_id}/AIPath.aimesh_ngrid"
