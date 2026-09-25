"""Log tailing (where CS@10 and fatals come from) and Slurm submission paths."""

from __future__ import annotations

import pytest

from lanerl_train import paths
from lanerl_train.serverlog import LogTail, cs_at, parse_cs_line
from lanerl_train.slurm import SlurmJob, render_script

CS_LINE = "LANERL_CS t=600000 name=Garen team=100 cs=32 gold=2100 lvl=8 hp=530/1100 deaths=0"


# -- serverlog -------------------------------------------------------------


def test_cs_line_parses_into_its_fields():
    row = parse_cs_line(CS_LINE)
    assert (row.t_ms, row.team, row.cs, row.deaths) == (600_000, 100, 32, 0)
    assert (row.hp, row.max_hp, row.level) == (530, 1100, 8)


def test_a_non_cs_line_parses_to_none():
    assert parse_cs_line("LANERL_TPS 2580.0 ticks/s") is None


def test_cs_at_takes_the_latest_row_per_team_at_or_before_the_mark():
    rows = [
        parse_cs_line("LANERL_CS t=300000 name=Garen team=100 cs=15 gold=1 lvl=1 hp=1/1 deaths=0"),
        parse_cs_line(CS_LINE),
        parse_cs_line("LANERL_CS t=900000 name=Garen team=100 cs=60 gold=1 lvl=1 hp=1/1 deaths=0"),
    ]
    at10 = cs_at(rows, 600_000)
    assert at10[100].cs == 32, "a row from 15 minutes must not contaminate CS@10"


def test_a_team_with_no_row_is_absent_not_zero():
    """Reporting 0 for a crashed episode would quietly drag the headline metric down."""
    assert cs_at([parse_cs_line(CS_LINE)], 600_000).keys() == {100}


def test_log_tail_follows_appends_and_never_splits_a_line(tmp_path):
    path = tmp_path / "instance000.log"
    tail = LogTail(path)
    assert tail.poll().lines == []  # tolerates the file not existing yet

    path.write_text("LANERL_CONTROL client attached\n")
    ev = tail.poll()
    assert ev.control_attached and tail.attached

    # a half-written line must not be parsed
    with path.open("a") as fh:
        fh.write(CS_LINE[:40])
    assert tail.poll().cs_rows == []
    with path.open("a") as fh:
        fh.write(CS_LINE[40:] + "\n")
    ev = tail.poll()
    assert len(ev.cs_rows) == 1 and ev.cs_rows[0].cs == 32
    tail.close()


def test_log_tail_flags_the_server_saying_it_is_dying(tmp_path):
    path = tmp_path / "i.log"
    path.write_text(
        "ok\nLANERL_CONTROL error: Object reference not set\nUnhandled exception. System.X\n"
    )
    ev = LogTail(path).poll()
    assert len(ev.fatal_lines) == 2


def test_rewind_starts_a_restarted_instance_from_a_clean_slate(tmp_path):
    path = tmp_path / "i.log"
    path.write_text(CS_LINE + "\n")
    tail = LogTail(path)
    assert len(tail.poll().cs_rows) == 1
    tail.rewind()
    assert tail.cs_rows == []
    path.write_text(CS_LINE + "\n")
    assert len(tail.poll().cs_rows) == 1


# -- slurm -----------------------------------------------------------------


def other_node():
    here = paths.local_mount()
    for node, mount in paths.node_mounts().items():
        if mount != here:
            return node, mount
    return None, None


def test_the_log_path_is_translated_for_the_node_that_will_open_it(tmp_path):
    node, mount = other_node()
    if node is None:
        pytest.skip("only one mount configured")
    job = SlurmJob(
        name="lanerl-selfplay",
        node=node,
        command=["-m", "lanerl_train.run"],
        log_dir=paths.repo_root() / "runs/logs",
        python="/home/dani/miniconda3/envs/ml/bin/python",
    )
    args = job.sbatch_args()
    out = args[args.index("-o") + 1]
    assert out.startswith(mount), "sbatch -o is opened on the target node, not the submitter"
    assert "%j" in out
    assert args[args.index("-w") + 1] == node


def test_a_log_dir_off_the_shared_export_is_refused():
    node, _ = other_node()
    if node is None:
        pytest.skip("only one mount configured")
    job = SlurmJob(name="x", node=node, command=["true"], log_dir="/root/nowhere")
    with pytest.raises(paths.PathResolutionError):
        job.sbatch_args()


def test_the_script_fails_the_job_on_an_unchecked_error():
    node = paths.local_node()
    if node not in paths.node_mounts():
        pytest.skip(f"hostname {node} is not in the node table")
    job = SlurmJob(
        name="x", node=node, command=["-m", "lanerl_train.run"],
        log_dir=paths.repo_root() / "runs/logs", env={"LANERL_PORT_BASE": "21000"},
    )
    script = render_script(job)
    assert "set -euo pipefail" in script
    assert "export LANERL_PORT_BASE=21000" in script
    assert str(paths.on_node(paths.repo_root(), node)) in script
