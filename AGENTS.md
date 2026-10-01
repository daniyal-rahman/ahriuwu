# Working here (read fully; it is short)

**Start:** `STATUS.md` (what runs, what is next), then `docs/CODEMAP.md`
(what is live). Findings go in `docs/JAX_FIDELITY_LEDGER.md` rows; runs go in
`docs/EXPERIMENTS.md` rows. Do not write new status/report files.

**Goal now:** Systematically debug and improve AFK farming in the JAX simulator,
per Dani’s subsequent explicit authorization. Use frozen evaluations and controlled
experiments; STATUS records the active compute window and next steps. The longer-term
transfer goal remains >30 CS in a 10-minute C# mirror trial over several seeds.

**Layout:** C# server = `/srv/nfs/projects/lanerl-vendor/LoLServer` (patches:
`lanerl/patches/`, canonical build `bin/DeadProbe`). RL = `lanerl_jax/train/`
(`server_train.py` collector, `learner.py`, `policy.py`, `ppo.py`). JAX sim =
`lanerl_jax/sim/`. Launchers = `experiments/`. One-offs = `lanerl_jax/probes/`.
History = `legacy/`, `docs/archive/`.

**Rules**
1. A run starts only via `ops/launch.py <ID>` from `experiments/<ID>.json`;
   new config = new ID + row. Never build a launch in the shell: no `ls -d`
   path capture (zsh drops the trailing slash), no env-var plumbing, no
   `--resume` for a new experiment (use `--init-from`: fresh optimizer,
   schedule and budget). `--dry-run` first; the canary must pass, and the launcher's 3-minute
   post-start watch must report the job healthy before you move on.
2. Starting or stopping a run, or ending a session, rewrites `STATUS.md` in
   the same commit. No STATUS edit = not finished.
3. Classify what you add: live path, tool, probe (with README row), or legacy.
   Never leave scripts in `runs/` or `/tmp` that a ledger row depends on.
4. Quote only frozen-policy evaluations as results; training CS is `train`.
5. Reversible actions (git mv, tags, new files) proceed; irreversible ones
   (delete branches/worktrees/builds/data, vendor edits) wait for Dani.
6. Desktop: servers and GPU live there; keep port bases < 32768; one core
   per server; cap login-node work with `ops/login_capped.sh`. Never yield to
   `llm-serve`. Kill jobs by ID; never `pkill -f`.
7. Never commit inside the vendor tree; server changes are patches in
   `lanerl/patches/` with a README row and a rebuilt `DeadProbe`.
8. Parallel agents: one git worktree and one experiment ID each; touch only
   your ID's runs; commit small, rebase on `lane-rl/jax` before handoff.

9. Learning experiments: research relevant literature, state a concrete hypothesis,
   budget and success/stop criteria, implement, compare frozen results against
   the existing baseline, then iterate. Prefer established defaults; distinguish
   published evidence from our engineering estimates.

10. Unattended Slurm runs: after the required startup watch, arm the event bridge
    before ending the turn: `python3 ops/slurm_event_bridge.py arm --job JOB_ID
    --experiment EXACT_SLURM_JOB_NAME --max-hours HOURS` (one shell line).
    Set a bounded lifetime covering the remaining job time plus queue/delivery
    margin; record the unit and expiry in STATUS. It discovers this T3 thread,
    checks job ownership/name, registers a capped one-shot watcher, and sends
    one idempotent completion/failure message only when this conversation is idle.
    Use this for current and future runs; do not substitute an ETA or active
    model polling for a wakeup. Verify the service is active and its result JSON
    is updating before claiming it is armed. Classify Slurm State AND ExitCode
    (TIMEOUT can have exit0). On wake, inspect logs/checkpoints/frozen evaluations,
    update the existing ledgers/STATUS, and verify watcher/registry cleanup.
    Stop with `systemctl --user stop lanerl-event-JOB_ID.service`; artifacts are
    `/mnt/nfs/shared/slurm-events/JOB_ID/`. No credentials in logs or Git.
    This tool currently handles single job IDs, not arrays; OPS047 is the array
    test. If bridge delivery is unavailable, say so explicitly; do not claim a
    scheduled check. A notification does not authorize a new experiment.

11. Dani's standing research authorization: proceed with bounded, reversible follow-up experiments on available Slurm compute when evidence makes them worth testing; do not wait for another confirmation. Preserve experiment IDs, budgets, canaries, frozen comparisons and event bridges. Before ending a turn, state actual running/queued jobs (or explicitly none), the experiment and rough ETA; distinguish planned follow-ups from submitted jobs. Irreversible actions still require approval.

12. Collaboration: questions and tentative ideas are discussion, not automatic
    implementation requests. Continue agreed work; change implementation when the
    idea is concrete or Dani explicitly asks. Use the existing launcher, integrated
    smoke tests and completion bridge; do not repeatedly poll or rerun checks without
    a new failure or change. Group coherent edits into useful commits. Keep replies
    direct: give ETAs in Pacific time (PDT/PST) or relative minutes; explain an
    experiment's setup alongside its ID, distinguish observations
    from hypotheses, and never assume a plateau proves the model cannot learn.
