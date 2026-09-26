# Working here (read fully; it is short)

**Start:** `STATUS.md` (what runs, what is next), then `docs/CODEMAP.md`
(what is live). Findings go in `docs/JAX_FIDELITY_LEDGER.md` rows; runs go in
`docs/EXPERIMENTS.md` rows. Do not write new status/report files.

**Goal now:** PPO from random init reaching >30 CS in a 10-minute mirror trial
on the C# server (frozen evaluation, several seeds). JAX is paused until then.

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
   schedule and budget). `--dry-run` first; the canary must pass.
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
