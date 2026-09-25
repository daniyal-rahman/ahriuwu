# STATUS (rewrite in place; last edit 2026-09-25 17:45 UTC, Claude)

**Goal now:** a randomly initialised PPO policy that scores >30 CS in a
10-minute mirror trial on the C# server, evaluated frozen over seeds. JAX
runs are paused until that gate is met.

**Running (desktop, relaunched 17:40 UTC on port bases 21700/21900 after OPS-003 killed both and the node rebooted; resumed from their last checkpoints). Launchers: `experiments/E01_mirror_wave.sh`, `E02_idle_wave.sh`:**
- `mirror-wave-s0`: C# mirror self-play, both champions start behind their
  first wave at 120 s, 10 envs. Train CS per episode 3.5 → ~12.5 (max 27)
  after ~200 episodes. Resumed three times: two rank-loop crashes (fixed)
  and one "address already in use" on an episode-reset server restart
  (fixed: ports rotate, three attempts).
- `idle-wave2-s0` (E02): STOPPED 18:40 UTC to free cores for the multi-process collector throughput test; resumable from its last checkpoint with `RESUME=... experiments/E02_idle_wave.sh`. Train CS 4 → ~10, noisy.
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

**First frozen evaluation (update 760, 5 mirror episodes, 20:28 UTC):** blue
mean 19.4 CS (11-27), red mean 16.0 (11-29), 0-2 deaths each. Below the 30
gate. Update 1080: blue 14.2 (5-22), red 21.0. Train CS has been FLAT at
16-18 per episode since about update 500: the run has plateaued at half the
gate. Diagnosis: the button mix swings between strategies (E-spin 81% at u460,
attack-move 38% at u820, Q 63% at u1126) with KL ~1e-2 per update: the
policy wanders rather than converges. Dani's replay review found the real defects: 46-54% of movement clicks land on
unwalkable ground (the server then walks straight into the wall, hence the
edge-hugging), and the +0.005*xp term paid 1.11x the CS term, teaching both
champions to camp the brush beside the wave. Fixed: clicks snap to the nearest
standable cell (PATH-010 table, both engines), `--xp-weight` (E04 uses 0).
E03 (lr test) stopped; E04 branches E01's state at update 1520 with the two
fixes at the same lr; E01 keeps running as the control.

**Next:**
1. Periodic frozen evaluation every ~300 updates (`ops/periodic_eval.sh`, results in `runs/EVAL/summary.jsonl`).
3. Multi-process collector (`--workers N`, `MultiProcessCollector`) is
   implemented and smoke-tested: 12 envs mirror, workers=1 vs 3 gave the same
   ~770 decisions/s on 6 cores while E01 held 8 cores -- the desktop is
   SERVER-CPU-bound at ~14 servers, not Python-bound any more. More throughput
   needs more cores (or fewer ticks per decision), not more workers.
4. If E01 plateaus below 30: run seeds 1-2 (`SEED=1 experiments/E01_mirror_wave.sh`).

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
