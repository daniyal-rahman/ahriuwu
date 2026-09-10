"""Path resolution across nodes, and port blocks that cannot collide.

Both failure modes here are silent by nature: a Slurm log path that is invalid
on the target node produces a job with no output at all, and a shared control
port kills every server instance but the first while the trainer sits blocked on
a socket.  Neither announces itself, so both get tests.
"""

from __future__ import annotations

import socket

import pytest

from lanerl_train import paths
from lanerl_train.ports import (
    InstancePorts,
    PortAllocationError,
    PortAllocator,
    assert_unique,
    is_port_free,
)


# -- paths -----------------------------------------------------------------


def test_repo_root_is_resolved_from_file_not_hardcoded():
    root = paths.repo_root()
    assert (root / "lanerl_train" / "paths.py").exists()
    # The package must sit under one of the known mount points, whichever node
    # this is running on.
    assert str(root).startswith(paths.local_mount())


def test_local_mount_is_one_of_the_known_mounts():
    assert paths.local_mount() in set(paths.node_mounts().values())


def test_on_node_translates_between_the_two_mounts():
    mounts = paths.node_mounts()
    here = paths.local_mount()
    other = [(n, m) for n, m in mounts.items() if m != here]
    if not other:
        pytest.skip("only one mount configured")
    node, mount = other[0]
    src = paths.repo_root() / "metrics.jsonl"
    out = paths.on_node(src, node)
    assert str(out).startswith(mount)
    assert str(out).endswith("metrics.jsonl")
    assert str(out) != str(src)


def test_on_node_is_identity_for_this_node():
    mounts = paths.node_mounts()
    here = paths.local_mount()
    this = [n for n, m in mounts.items() if m == here]
    assert this, "the local mount must belong to some node in the table"
    src = paths.repo_root() / "a.log"
    assert paths.on_node(src, this[0]) == src


def test_on_node_refuses_an_unknown_node():
    with pytest.raises(paths.PathResolutionError, match="unknown node"):
        paths.on_node(paths.repo_root(), "not-a-node")


def test_on_node_refuses_a_path_outside_the_shared_mount(tmp_path, monkeypatch):
    monkeypatch.setenv("LANERL_NODE_MOUNTS", f"a={paths.local_mount()},b=/elsewhere")
    with pytest.raises(paths.PathResolutionError, match="not under this node's shared mount"):
        paths.on_node("/root/nowhere/x.log", "b")


def test_node_mounts_env_override_and_validation(monkeypatch):
    monkeypatch.setenv("LANERL_NODE_MOUNTS", "alpha=/mnt/a,beta=/mnt/b")
    assert paths.node_mounts() == {"alpha": "/mnt/a", "beta": "/mnt/b"}
    monkeypatch.setenv("LANERL_NODE_MOUNTS", "garbage")
    with pytest.raises(paths.PathResolutionError):
        paths.node_mounts()


def test_runs_root_respects_the_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("LANERL_RUNS_DIR", str(tmp_path))
    assert paths.runs_root() == tmp_path.resolve()


# -- ports -----------------------------------------------------------------


def test_allocate_gives_sixteen_disjoint_blocks():
    ports = PortAllocator(base=41000).allocate(16)
    assert len(ports) == 16
    flat = [p.control for p in ports] + [p.game for p in ports]
    assert len(set(flat)) == 32
    assert_unique(ports)  # must not raise


def test_assert_unique_catches_a_shared_port():
    shared = [InstancePorts(0, 5119, 5120), InstancePorts(1, 5119, 5121)]
    with pytest.raises(PortAllocationError, match="assigned twice"):
        assert_unique(shared)


def test_allocator_skips_a_port_that_is_actually_in_use():
    base = 41100
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("127.0.0.1", base))
    holder.listen(1)
    try:
        assert not is_port_free(base)
        ports = PortAllocator(base=base).allocate(2)
        assert all(p.control != base and p.game != base for p in ports)
    finally:
        holder.close()


def test_allocator_never_reissues_a_block():
    alloc = PortAllocator(base=41200)
    first = alloc.allocate(4)
    second = alloc.allocate(4)
    a = {p.control for p in first} | {p.game for p in first}
    b = {p.control for p in second} | {p.game for p in second}
    assert not (a & b)


def test_allocator_raises_rather_than_sharing_when_it_cannot_find_blocks():
    alloc = PortAllocator(base=41300, max_scan=0)
    with pytest.raises(PortAllocationError):
        alloc.allocate(1)


def test_control_port_lands_in_the_environment():
    p = InstancePorts(3, 41500, 41501)
    assert p.as_env() == {"LANERL_CONTROL_PORT": "41500"}
