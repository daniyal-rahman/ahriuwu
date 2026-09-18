"""Every command this repo documents as canonical must actually resolve.

This is a mechanical check, not a behavioural one: it does not run a gate, it
only asserts that the things the gate documentation tells you to type exist.
That is worth a test file on its own because this repo has now paid for the
same failure three separate times, each time in a way that looked like a
result rather than like a broken command:

* `slurm/parity_g1.sbatch` and `slurm/parity_s2.sbatch` pointed `REPO` at
  agent worktrees that had been deleted.  `bd39a1b` had already removed the
  `s1` pair for exactly this, and the other two survived it.
* `slurm/parity_g1.sbatch`'s usage line said
  `python -m lanerl_jax.runs.tier1_full`.  `lanerl_jax/runs` is a gitignored
  *output* directory, not a package; the module is `lanerl_jax.parity.
  tier1_full`.
* `lanerl_jax/parity/tier2_batch.py` told the reader to hand its command to
  `sbatch slurm/parity_g2.sbatch`, which is not in the tree.

None of these fail loudly at the moment someone reads them.  They fail later,
under slurm, as an empty output file -- which is the `docs/`-recorded
silent-failure class this project keeps rediscovering.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SBATCH = sorted((REPO / "slurm").glob("*.sbatch"))

#: `--chdir`/`--output`/`REPO` must be spelled `/mnt/nfs`, because slurm opens
#: `--output` on the COMPUTE node and only `danilogin` has both mount points.
#: A `/srv/nfs` spelling produces a job that dies before writing anything.
MOUNT_PREFIX = "/mnt/nfs/"


def _resolves(p: str) -> bool:
    """Does this NFS path name a real directory, from whichever host we are on?

    `danilogin` has BOTH `/srv/nfs` and `/mnt/nfs` (same export); `desktop` has
    only `/mnt/nfs`. An earlier version of this file translated `/mnt/nfs` ->
    `/srv/nfs` unconditionally and therefore passed on the login node and
    failed every path on `desktop` -- which is the exact mount-portability bug
    this module exists to catch, committed inside the catcher. Accept the path
    if EITHER spelling of the same export resolves here.
    """
    return any(Path(c).is_dir() for c in
               (p, p.replace("/mnt/nfs/", "/srv/nfs/", 1),
                p.replace("/srv/nfs/", "/mnt/nfs/", 1)))


@pytest.mark.parametrize("script", SBATCH, ids=lambda p: p.name)
def test_sbatch_repo_still_exists(script: Path):
    """A REPO pointing at a deleted worktree is a job that cannot run."""
    m = re.search(r"^REPO=(\S+)", script.read_text(), re.M)
    if m is None:
        pytest.skip(f"{script.name} defines no REPO")
    repo = m.group(1)
    assert repo.startswith(MOUNT_PREFIX), (
        f"{script.name}: REPO={repo} must be spelled {MOUNT_PREFIX}... -- the "
        "compute node has no /srv/nfs mount")
    assert _resolves(repo), (
        f"{script.name}: REPO={repo} does not exist. If its worktree was "
        "consolidated away, delete this script rather than leaving a job that "
        "fails after the queue wait.")


@pytest.mark.parametrize("script", SBATCH, ids=lambda p: p.name)
def test_sbatch_directive_paths_still_exist(script: Path):
    """`--chdir` and `--output` are opened by slurm on the COMPUTE node.

    A stale directory there is the worst of the three failures in this file's
    docstring: the job is accepted, waits in the queue, then dies before it can
    write the output file that would have told you why.

    Only shared-NFS paths are checkable from here.  `/mnt/storage/...` and
    `/scratch/...` are deliberately node-local (tokenizer latents and scratch
    parity output live there), so their existence says nothing from the login
    node and asserting on it would just be a test that lies when `desktop` is
    the only machine that could answer.  What IS checkable, and what has
    actually broken: an NFS path spelled `/srv/nfs` (the login-node-only
    spelling), and an NFS path whose directory is gone.
    """
    text = script.read_text()
    for flag in ("chdir", "output"):
        for raw in re.findall(rf"^#SBATCH --{flag}=(\S+)", text, re.M):
            assert not raw.startswith("/srv/nfs/"), (
                f"{script.name}: --{flag}={raw} is the login-node spelling. "
                "slurm opens this on the compute node, which has only "
                f"{MOUNT_PREFIX} -- the job dies before writing anything.")
            if not raw.startswith(MOUNT_PREFIX):
                continue                   # node-local; unverifiable from here
            base = raw.split("%")[0]
            want = base if flag == "chdir" else str(Path(base).parent)
            assert _resolves(want), (
                f"{script.name}: --{flag}={raw} points at {want}, which does "
                "not exist under either mount spelling. The job will die "
                "before writing any output.")


@pytest.mark.parametrize("script", SBATCH, ids=lambda p: p.name)
def test_sbatch_documented_modules_are_importable(script: Path):
    """`python -m x.y` in a usage comment must name a real module."""
    import importlib.util

    text = script.read_text()
    for mod in set(re.findall(r"python -m ([A-Za-z_][\w.]*)", text)):
        if not mod.startswith("lanerl"):
            continue
        assert importlib.util.find_spec(mod) is not None, (
            f"{script.name} documents `python -m {mod}`, which does not "
            f"resolve. `lanerl_jax/runs` in particular is a gitignored output "
            f"directory, not a package.")


def test_referenced_sbatch_scripts_exist():
    """A module telling you to `sbatch X` must not name a script we deleted."""
    missing: list[str] = []
    for src in sorted((REPO / "lanerl_jax").rglob("*.py")):
        if src.resolve() == Path(__file__).resolve():
            continue                       # this file names them to explain them
        for ref in set(re.findall(r"sbatch\s+(slurm/[\w.-]+\.sbatch)",
                                  src.read_text())):
            if not (REPO / ref).exists():
                missing.append(f"{src.relative_to(REPO)} -> {ref}")
    assert not missing, "referenced sbatch script(s) missing: " + "; ".join(missing)
