"""`PPO-12`: the manifest says what ran, and the README's reproduce command runs.

The README's "Reproducing" block used to print every recorded CLI value as
``--flag value``, so a ``store_true`` flag came out as ``--no-route-table
False`` (argparse rejects it) and ``--notes`` with a space came out unquoted;
Path-valued options were dropped from the record entirely. These tests parse
the command back with the REAL parser rather than inspecting its text.
"""
from __future__ import annotations

import json
import re
import shlex
from types import SimpleNamespace

from lanerl_jax.train.run_manifest import RunDir, reproduce_command
from lanerl_jax.train.run_train import build_parser, cli_record, manifest_config
from lanerl_jax.train.trainer import TrainConfig

ARGV = ["--tag", "t1", "--updates", "7", "--lr", "3e-4",
        "--no-route-table", "--no-value-clip",
        "--notes", "sweep B arm a1 (--lr 1e-5); it's \"quoted\" $HOME",
        "--route-artifact", "/tmp/a dir/routes v2",
        "--resume", "/tmp/ckpt latest.msgpack"]


def _parse_back(command: str):
    tokens = shlex.split(command.replace("\\\n", " "))
    assert tokens[:2] == ["sbatch", "slurm/rl_train.sbatch"], tokens[:2]
    return build_parser().parse_args(tokens[2:])


def test_reproduce_command_parses_back_to_the_same_run():
    a = build_parser().parse_args(ARGV)
    cli = cli_record(a)
    # Paths are recorded, not dropped
    assert cli["route_artifact"] == "/tmp/a dir/routes v2"
    assert cli["resume"] == "/tmp/ckpt latest.msgpack"
    back = _parse_back(reproduce_command(a.tag, cli))
    assert back.tag == a.tag
    assert cli_record(back) == cli


def test_the_written_readme_command_parses_back(tmp_path):
    """The same check against the README a real `RunDir` writes."""
    a = build_parser().parse_args(ARGV)
    run = RunDir(tmp_path, a.tag, {"cli": cli_record(a)}, notes=a.notes)
    run.close()
    readme = (run.path / "README.md").read_text()
    block = re.search(r"## Reproducing\n\n```bash\ngit checkout \S*\n(.*?)\n```",
                      readme, re.S)
    assert block, readme
    back = _parse_back(block.group(1))
    assert cli_record(back) == cli_record(a)

    m = json.loads((run.path / "manifest.json").read_text())
    pk = m["software"]["packages"]
    for lib in ("jax", "jaxlib", "flax", "optax"):
        assert pk[lib] not in ("", "absent", None), (lib, pk)
    assert "XLA_FLAGS" in m["software"]["env"]


def test_manifest_config_records_policy_and_route_artifact_by_content(tmp_path):
    art = tmp_path / "routes"
    art.mkdir()
    (art / "manifest.json").write_text('{"table_sha256": "abc"}')
    sim = SimpleNamespace(route_artifact=str(art), route_digest="abc:def",
                          describe=lambda: {"name": "training"},
                          fingerprint=lambda: "f00")
    cfg = TrainConfig()
    a = build_parser().parse_args([])
    got = manifest_config(cfg, a, sim, 1)
    assert got["policy"] == cfg.policy
    assert got["route_artifact"]["path"] == str(art)
    assert got["route_artifact"]["content_sha256"] == "abc:def"
    import hashlib
    assert got["route_artifact"]["manifest_sha256"] == hashlib.sha256(
        b'{"table_sha256": "abc"}').hexdigest()
    assert got["sim_config"] == {"name": "training", "fingerprint": "f00"}
    unrouted = SimpleNamespace(route_artifact=None, route_digest=None,
                               describe=lambda: {"name": "x"},
                               fingerprint=lambda: "0")
    assert manifest_config(cfg, a, unrouted, 1)["route_artifact"] is None
