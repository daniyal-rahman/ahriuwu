# STATUS (rewrite in place; last edit 2026-09-25 16:10 UTC, Claude)

**Goal now:** a randomly initialised PPO policy that scores >30 CS in a
10-minute mirror trial on the C# server, evaluated frozen over seeds. JAX
runs are paused until that gate is met.

**Running (desktop, 14 server processes + GPU learner):**
- `mirror-wave-s0`: C# mirror self-play, both champions start behind their
  first wave at 120 s, 10 envs. Train CS per episode 3.5 → ~12.5 (max 27)
  after ~200 episodes. Resumed twice after rank-loop crashes (fixed).
- `idle-wave2-s0`: C# blue vs idle red, 4 envs. Train CS 4 → ~10.
- Check: `python ops/server_train_status.py lanerl_jax/runs/server_train/<run>`.

**Done today:** Codex's uncommitted server-first work committed (`35210dc`);
collector 3-5x faster; mirror mode, resume, frozen eval; lr 1e-5 identified as
the dead learner; REW-11 (shaping paid for sitting in base) and PPO-15
(entropy favoured coordinate buttons) found and fixed; rank loop no longer
aborts; red setup route legged in both engines; legacy code moved to
`legacy/`; CODEMAP, EXPERIMENTS, patch and probe indexes written.

**Next:**
1. Screen-click verification probe on the live server (does the inverse
   projection land on the intended minion; how wide is a click cell).
2. Frozen evaluation of `mirror-wave-s0` at its next checkpoint, 5 episodes.
3. If train CS plateaus below 30: adopt the process-parallel collector
   (`lanerl_train/procactor.py`, 4,829 dec/s at 96 servers) into
   `server_train.py` and run 3 seeds.

**Open bugs / unknowns:** attack completion is not attributed (Codex saw zero
logged basic-attack completions in a 20-CS episode); `trainer.py` still has
the PPO-15 bias; ENT-02 vs AA-007 (retarget mid-windup) unmeasured.

**Needs Dani's approval (irreversible):** delete branches `bronze`,
`eval-results-jan12`, `report/2026-09-02`, `t3code/4b6bc7fc` (all tagged
`archive/*` now); remove the 8 frozen worktrees
`/srv/nfs/projects/ahriuwu-*-20260925` (1.3 GB; source is in each run's
`source.tar.gz`); delete superseded server builds `HudProbe`, `ScreenClick`,
`Release`.

Rules: whoever starts or stops a run edits this file in the same commit.
History lives in `docs/EXPERIMENTS.md` and `docs/JAX_FIDELITY_LEDGER.md`.
