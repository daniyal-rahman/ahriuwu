"""Where things are, on which node.

The NFS export is mounted at a *different absolute path* on each node
(``/srv/nfs`` on danilogin, ``/mnt/nfs`` on desktop).  Two rules follow, and both
have already cost this project real time:

1. **Never hardcode the local mount.**  Everything here is resolved from
   ``__file__``, so the same source file works from either side.
2. **A path written for another node must be translated.**  Slurm's ``-o`` log
   path is opened *by the target node*, so ``sbatch -w desktop -o /srv/nfs/...``
   produces a job that dies before it prints anything -- silently, from the
   submitter's point of view.  :func:`on_node` is the only sanctioned way to
   build such a path.

The node->mount table is topology, not a path constant: it says where a *remote*
node keeps the same export.  Override it with ``LANERL_NODE_MOUNTS``
(``"desktop=/mnt/nfs,danilogin=/srv/nfs"``) rather than editing this file.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Dict, Optional

__all__ = [
    "PathResolutionError",
    "node_mounts",
    "local_node",
    "package_root",
    "repo_root",
    "projects_root",
    "vendor_root",
    "server_dir",
    "server_binary",
    "dotnet_root",
    "server_available",
    "bot_config_dir",
    "default_game_config",
    "runs_root",
    "local_mount",
    "on_node",
]


class PathResolutionError(RuntimeError):
    """Raised when a path cannot be resolved or translated.

    Deliberately fatal: a wrong path on this project shows up as a job that
    produces no output at all, which is far more expensive to debug than a
    stack trace at submit time.
    """


_DEFAULT_NODE_MOUNTS: Dict[str, str] = {
    "desktop": "/mnt/nfs",
    "danilogin": "/srv/nfs",
}


def node_mounts() -> Dict[str, str]:
    """``{node: mount point of the shared export on that node}``."""
    raw = os.environ.get("LANERL_NODE_MOUNTS")
    if not raw:
        return dict(_DEFAULT_NODE_MOUNTS)
    out: Dict[str, str] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        node, sep, mount = item.partition("=")
        if not sep or not node.strip() or not mount.strip():
            raise PathResolutionError(
                f"LANERL_NODE_MOUNTS entry {item!r} is not 'node=/mount'; refusing to guess"
            )
        out[node.strip()] = mount.strip().rstrip("/") or "/"
    if not out:
        raise PathResolutionError("LANERL_NODE_MOUNTS is set but empty")
    return out


def local_node() -> str:
    """This node's short hostname."""
    return socket.gethostname().split(".")[0]


# -- the project tree ------------------------------------------------------


def package_root() -> Path:
    """``.../ahriuwu-lanerl/lanerl_train``."""
    return Path(__file__).resolve().parent


def repo_root() -> Path:
    """``.../ahriuwu-lanerl``."""
    return package_root().parent


def projects_root() -> Path:
    """The directory holding both ``ahriuwu-lanerl`` and ``lanerl-vendor``."""
    return repo_root().parent


def vendor_root() -> Path:
    return projects_root() / "lanerl-vendor"


def server_dir() -> Path:
    return vendor_root() / "LoLServer/GameServerConsole/bin/Release/net6.0"


def server_binary() -> Path:
    """The self-contained launcher.

    ``lanerl_rl.control_smoke`` runs ``GameServerConsole`` directly rather than
    ``dotnet GameServerConsole.dll``; both exist in the publish directory.  The
    direct binary avoids one process layer per instance, which matters at N=16.
    """
    return server_dir() / "GameServerConsole"


def dotnet_root() -> Path:
    return vendor_root() / "dotnet"


def server_available() -> bool:
    """True when a real server can actually be launched from this node."""
    if os.environ.get("LANERL_SKIP_SERVER_TESTS") == "1":
        return False
    return server_binary().exists() or (
        (server_dir() / "GameServerConsole.dll").exists() and (dotnet_root() / "dotnet").exists()
    )


def bot_config_dir() -> Path:
    return repo_root() / "lanerl_bot/configs"


def default_game_config() -> Path:
    """The 1v1 Garen map/config the control smoke test uses."""
    return repo_root() / "lanerl/cfg/garen1v1.json"


def runs_root() -> Path:
    """Root for training run directories.

    Defaults inside the repo so both nodes see the same run over NFS; override
    with ``LANERL_RUNS_DIR`` to put rollouts on node-local disk.
    """
    env = os.environ.get("LANERL_RUNS_DIR")
    return Path(env).resolve() if env else repo_root() / "runs"


# -- cross-node translation ------------------------------------------------


def local_mount(mounts: Optional[Dict[str, str]] = None) -> str:
    """The mount prefix, on *this* node, of the shared export.

    Discovered from ``__file__`` -- never hardcoded -- then checked against the
    topology table so a surprise layout fails here instead of inside a job.
    """
    mounts = mounts or node_mounts()
    here = str(repo_root())
    candidates = [m for m in set(mounts.values()) if here == m or here.startswith(m + "/")]
    if not candidates:
        raise PathResolutionError(
            f"this checkout lives at {here}, which is under none of the known mount points "
            f"{sorted(set(mounts.values()))}. Set LANERL_NODE_MOUNTS to describe the real "
            f"topology; do not hardcode a path."
        )
    # Longest match wins, so a nested mount cannot be mistaken for its parent.
    return max(candidates, key=len)


def on_node(path: os.PathLike | str, node: str, mounts: Optional[Dict[str, str]] = None) -> Path:
    """Rewrite a locally-valid path so it names the same file **on** ``node``.

    Use this for every Slurm ``-o``/``-e`` path and every path passed to a job
    that will run elsewhere.  Raises rather than returning the input unchanged:
    an untranslated path is exactly the failure mode this exists to prevent.
    """
    mounts = mounts or node_mounts()
    if node not in mounts:
        raise PathResolutionError(
            f"unknown node {node!r}; known nodes are {sorted(mounts)}. "
            f"Add it to LANERL_NODE_MOUNTS."
        )
    p = Path(path).resolve()
    here = local_mount(mounts)
    s = str(p)
    if s != here and not s.startswith(here + "/"):
        raise PathResolutionError(
            f"{p} is not under this node's shared mount {here}, so it has no meaning on "
            f"{node!r}. Put it on the export, or pass a node-local path deliberately."
        )
    target = mounts[node].rstrip("/") or "/"
    rest = s[len(here) :].lstrip("/")
    return Path(target) / rest if rest else Path(target)
