# Working here

**Start:** `STATUS.md`, then `docs/CODEMAP.md` and `docs/modern/WORLD_IMPLEMENTATION.md`. Findings go in
`docs/JAX_FIDELITY_LEDGER.md` rows; do not write new status or report files.

**Rules**
1. Keep the code tight: short comments that say why or cite a source, no narrative history, no dead
   code. Values come from the specs in `docs/modern/` (cite the section once).
2. Refactors must preserve behaviour and show it: `python -m ops.modern.jaxpr_fingerprint --against REF`
   (identical traced tick) or `python -m ops.modern.golden --compare REF` (identical trajectories).
3. Starting or stopping a run, or ending a session, updates `STATUS.md` in the same commit.
4. Reversible actions (git mv, tags, new files) proceed; irreversible ones (deleting branches, worktrees,
   data) wait for Dani.
5. All GPU and heavy CPU work goes through Slurm on the desktop (`gpup`, `cpu`); timing runs take the
   whole node (`--exclusive`). Kill jobs by ID only.
6. Parallel agents: one git worktree each; touch only your area; commit small.
