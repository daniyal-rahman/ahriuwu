# STATUS (rewrite in place; last edit 2026-09-25 17:20 UTC, Claude)

**Goal now:** a randomly initialised PPO policy that scores >30 CS in a
10-minute mirror trial on the C# server, evaluated frozen over seeds. JAX
runs are paused until that gate is met.

**Runs (desktop) — BOTH DIED 16:20-16:30 UTC on OPS-003 (ephemeral-range ports at episode reset); the desktop then went "Not responding" in slurm. A watcher relaunches both on port bases 21700/21900 when the node returns, resuming from the last checkpoints:**
- `mirror-wave-s0`: C# mirror self-play, both champions start behind their
  first wave at 120 s, 10 envs. Train CS per episode 3.5 → ~12.5 (max 27)
  after ~200 episodes. Resumed three times: two rank-loop crashes (fixed)
  and one "address already in use" on an episode-reset server restart
  (fixed: ports rotate, three attempts).
- `idle-wave2-s0`: C# blue vs idle red, 4 envs. Train CS 4 → ~10.
- Check: `python ops/server_train_status.py lanerl_jax/runs/server_train/<run>`.

**Done today:** Codex's uncommitted server-first work committed (`35210dc`);
collector 3-5x faster; mirror mode, resume, frozen eval; lr 1e-5 identified as
the dead learner; REW-11 (shaping paid for sitting in base) and PPO-15
(entropy favoured coordinate buttons) found and fixed; rank loop no longer
aborts; red setup route legged in both engines; legacy code moved to
`legacy/`; CODEMAP, EXPERIMENTS, patch and probe indexes written.

**Screen-click verified on the live server (probe 1, `lanerl_jax/probes/screen_click_probe.py`):**
a click cell is 30 x 36 world units at screen centre (smaller than a minion);
attack-move clicks on a minion acquired it; ground clicks land within
quantisation (7-22 u) and the champion arrives within 1 u. Probe 2
(`screen_click_probe2.py`): right-click and attack-move on a minion set the
server target for BLUE and RED, raw and through the grid + lane frame, at
65-685 u. Probe 1's right-click misses were at 600-1000 u where cell
quantisation exceeds the minion's collision radius; attack-move auto-acquire
covers that range. Verdict: the click interface is implemented correctly.

**Next:**
1. Frozen evaluation of `mirror-wave-s0` at its next checkpoint, 5 episodes.
3. If train CS plateaus below 30: adopt the process-parallel collector
   (`lanerl_train/procactor.py`, 4,829 dec/s at 96 servers) into
   `server_train.py` and run 3 seeds.

**Open bugs / unknowns:** attack completion is not attributed (Codex saw zero
logged basic-attack completions in a 20-CS episode); `trainer.py` still has
the PPO-15 bias; ENT-02 vs AA-007 (retarget mid-windup) unmeasured.

**Approved and done 2026-09-25:** branches `bronze`, `eval-results-jan12`,
`report/2026-09-02`, `lane-rl/spike`, `t3code/4b6bc7fc` and the agent
worktree branch deleted (tags `archive/*` pushed to origin); the eight
frozen `ahriuwu-*-20260925` worktrees and `_deprecated/ahriuwu-lanerl`
removed; server builds `HudProbe`, `ScreenClick` deleted and `Release`
replaced by a symlink to `DeadProbe`. Remaining branches: `main`,
`lane-rl/jax` (active), `t3code/9283ee84` (another T3 thread's worktree).

Rules: whoever starts or stops a run edits this file in the same commit.
History lives in `docs/EXPERIMENTS.md` and `docs/JAX_FIDELITY_LEDGER.md`.
