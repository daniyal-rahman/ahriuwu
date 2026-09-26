# STATUS (rewrite in place; last edit 2026-09-25 17:45 UTC, Claude)

**Goal now:** a randomly initialised PPO policy that scores >30 CS in a
10-minute mirror trial on the C# server, evaluated frozen over seeds. JAX
runs are paused until that gate is met.

**INTERFACE ORACLE (22:25 UTC):** the scripted last-hitter through the policy's own
observation->click path scores 60 CS vs idle red and 45 (blue) / 28 (red) in a
scripted mirror, deterministic across servers. The gate is reachable through
this interface; the trained policies' 15-22 is a learning gap. Scripted RED vs idle blue: 74 CS, 0 deaths, so the 45/28 mirror split is wave
interaction, not a side bug.
Next diagnostics: behaviour-clone the oracle (representability), then PPO from
that init (does the learner preserve a 40-CS policy?).

**Plan (Dani, 22:40 UTC):** pause E07; run the three diagnostics (oracle done,
BC representability, PPO-from-BC learner test); then a heuristics-initialised
GRU on the JAX sim to a compare point; then cross-play E04 vs that agent on
the C# server. **BC representability (23:20 UTC):** the MLP clone matches the scripted teacher
99% (button) / 74% (click x,y) per step but scores 5.3 / 3.8 CS sampled in a
mirror (teacher 45 / 28): sampled clicks a cell off miss the minion and the
errors compound. Argmax eval running. E08 = PPO from this clone (learner test).
**Running:** E08; BC diagnostic training (mlp, gru) on 19,196 x 8 scripted mirror
steps (`runs/ORACLE/demos`); their frozen evaluations follow automatically.
JAX shared-loop GRU probe: ~430 dec/s at 8 envs (server-level), OOM at 32
envs beside llm-serve, so the JAX leg needs the Anakin port; deferred. E07 PAUSED at u178 (relaunched 21:55 UTC with `--init-from`: E06 u3140 params, fresh
optimizer/schedule, own 3000-update budget) vs a FROZEN E06 checkpoint. The first
E07 attempt inherited E06's counter and schedule (767 updates at ~0 lr, evals
failed) and is recorded as invalid.
E06 STOPPED 19:25 UTC at update 3800: plateau ~17 CS (band 13-22 over 11 frozen
evaluations since 4M decisions); it is the control. Earlier E06 notes: Frozen u320: 5.3 / 6.0 CS; u620: 8.5 / 13.5 (learning); u920: 2.5 / 1.5 --
COLLAPSE caused by OPS-004 (rank loop starving level-9+ champions of
decisions), not by the policy. Fixed 09:50 UTC; E06 resumed from its u620
checkpoint. Post-fix frozen u1260: 15.3 / 14.3; u1580: 14.3 / 24.5; u1900: 11.3 / 14.8
u2200: 20.3 / 17.5 (best); u2520: 16.8 / 13.8; u2840: 12.5 / 14.3; u3140: 15.5 / 14.3; u3460: 22.5 / 20.8 (best; both sides > 20); u3780: 18.3 / 16.0. Plateau band 14-20 frozen since
u1260 (E01's band). Leak fixed (eager scan re-traced per update): memory now
+2.4 MB/update, 7.4 s/update. E07 launched 18:30 UTC (Dani's go-ahead): E06's learner continued
against E06's u2200 checkpoint as a FROZEN opponent, learner side alternating
per server (`--opponent frozen`). Runs beside E06. Parallel-server audit of
E06: all 20 agent slots average 13.7-17.7 CS, no outlier server, zero rank/
restart/fatal/complaint events, all servers at 309 ticks/s. OOM-killed at
u1931 (slurm 10 GB limit, 12:55 UTC); resumed 14:30 UTC at 20 GB from u1920
with per-update RSS now logged (`rss_gb`). All numbers past level 9 before the fix are suspect.
Throughput 12 s/update = 213 dec/s with the node to itself; the GRU's
BPTT update (128-step scan x 16 minibatch-epochs) is ~2/3 of that and is
the next thing to optimise if E06 shows learning. The desktop was on Windows 04:11-06:37 UTC, which cancelled E04 and E06. Dani's decision (04:55 UTC): continue ONLY the GRU arm
with published defaults (E06). E04 stays stopped (frozen 18.8/24.5 at u2020,
train chunks 27-31 at u3070; its checkpoints remain). 

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
champions to camp the brush beside the wave. Fixed on the SERVER (`screen-click-v3`, build `ClickV3`, PATH-011): unwalkable
click targets resolve to the closest reachable point, as the real client does.
XP is enemy-only on the server (checked, SIDE-001) and stays, at weight 0.002
(was 0.005, which out-paid CS).
E03 (lr test) stopped; E04 branches E01's state at update 1520 on ClickV3 with
XP 0.002 at the same lr. FIRST SIGNAL (2026-09-26 00:40 UTC): E04's first 12
training episodes average 28.1 CS (30-42 in eight of them) against E01's
17-21 at the same point: the wall-click fix is the biggest single gain so far.
Frozen eval at E04 update 1820 pending. E01 (control, DeadProbe) still running.

E05 launched 01:20 UTC (see Running).

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
