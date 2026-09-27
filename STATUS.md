# STATUS (rewrite in place; last edit 2026-09-27 13:55 UTC, Claude)

**Goal now:** a randomly initialised PPO policy that scores >30 CS in a
10-minute mirror trial on the C# server, evaluated frozen over seeds.

**LEARNER TEST VERDICT (09:20 UTC): NEGATIVE.** Constrained PPO fine-tuning
from the DAgger-2 clone (45 / 50 CS frozen) destroys it in every form tried:
E10 plain (KL 0.38 in one update), E11 (lr 5e-5, KL stop 0.02): 0 CS by u140,
E12b (same, after E12a critic warm-up, value loss 21 -> 1.2): frozen u140
10.8 / 12.0, u280 2.5 / 2.3, u440 0.3 / 1.3, training CS 0 by u520.
E12a with actor lr 0 kept the clone intact (training CS peaks 30.6 at
mid-episode every 47 updates), so the collapse is caused by the actor update
itself, not by the init or the collector. A working PPO does not move a
policy from +45 return to 0 with a correlated reward. The learner, or the
reward as the learner sees it, is the open bug. Candidates, in order:
advantage normalisation on near-zero-variance batches, entropy 0.01/3 per
head pulling apart a sharp policy (entropy 0.3 -> 1.0 in E11), deaths
(-2, ~4 per update across 20 agents in mirror) dominating the per-update
signal, GAE/done handling in the [N,T] batch. E12b cancelled (its job).

**WALL HUGGING = CLICK INTERFACE (INT-001, 13:10 UTC), Dani's top priority:**
not bushes (server vision ignores grass; sim has none), not the potential
(0 inside a corridor that contains the walls), not XP. Clicks onto unwalkable
ground resolve to the closest reachable point = the wall; 43-49% of E06's
clicks were unwalkable (51-58% near walls), so a diffuse policy is pulled to
walls and held. Random policy, same seed: 35% of frames within 150 u of a
wall with resolve, 6% with the new `--unwalkable-click noop`. **E15** =
no-prior GRU (fixed core) + noop clicks, RUNNING since 11:08 UTC (job 1603,
`runs/E15_noprior_gru_noopclick`, evaluator armed every 100 updates): ~8 s per
update, 51% of movement clicks dropped at update 37, post_kl ~0.04. Masking unwalkable cells (joint 2-D click
head) is the principled follow-up.

**WHY THE PRIOR DEGRADES (PPO-16, 11:40 UTC):** E12b applied ONE minibatch
step per update: the KL stop measures pre-step KL, the first step is always
applied at KL 0, the second was over 0.02 every time (`kl_stopped` 0.94), and
the logged KL averages applied steps, so it read 0.000 while the policy moved.
One fresh-Adam step at lr 5e-5 from the clone = KL 0.33 (pg), 0.46 (entropy
term alone, grad norm 0.05: Adam is scale-invariant), 0.11 (value through the
shared trunk); at lr 1e-5 = 0.015; at 2.5e-4 = 8.7. 600 blind steps of KL
~0.3 with every corrective step withheld is a random walk off the prior.
`post_kl` metric added to `server_train`. Replays of E12b u140 and u600
rendering (`runs/EVAL/replay_E12b_*`).

**ARCH-001 (10:35 UTC): the plain GRU core is near-blind** (heads see only the GRU
output; BC cannot fit 16 sequences; MLP fits the same data to 98%). Fixed as
opt-in `--core-norm --core-residual`; E06's ~17 plateau is partly this. The
JAX-leg GRU uses the fixed wiring.

**JAX LEG (running, per Dani's 22:40 plan):** heuristics-initialised GRU on
the JAX sim to a compare point, then E04 vs that agent on the C# server.
**DAgger-1 GRU clone on JAX: 41.0 / 44.4 CS sampled mirror (26-59 / 34-58)** -- the
JAX heuristics-init GRU compare point; cross-play E04 vs it and its server
mirror transfer are running (`runs/EVAL/XPLAY_E04_vs_jaxgru`, `XFER_jaxgru_mirror`);
round 3 running. E13 (no-prior JAX GRU, snap clicks) at u~80: 16 CS vs idle.
Step 1 done: scripted last-hitter on the JAX sim vs idle red = 65 CS on all
16 envs (deterministic sim: identical trajectories), 4792 decisions each in
140 s wall (~550 dec/s at 16 envs, 5.5 GB GPU). Mirror scripted demos
recording (`runs/JAX_ORACLE/demos_mirror`). Step 2: GRU clone
(`train/bc_diag.py --core gru`, `runs/BC/bc-gru-*`), then DAgger rounds on
JAX rollouts. Also running: **E13** = no-prior fixed-wiring GRU on JAX (job 1576, `runs/E13_jax_gru_scratch`), the JAX twin of E06.
Step 3: PPO on JAX from the GRU clone (`jax_train.py
--init-from --core gru --preset standard`). Step 4: `ops/launch.py eval
--ckpt <E04> --opponent frozen --opponent-ckpt <jax gru>` on the C# server.
Porting issues: the shared-loop JAX trainer is server-speed (430-550
dec/s); the fast Anakin trainer lacks GRU/init-from/frozen opponent; parity
gaps COLL-005, SPELL-013, STAT-003 mean the JAX agent's clicks may transfer
imperfectly; an observation-contract match (viewport-structured-v3) is what
makes the cross-play possible at all.

**History of the diagnostics (09-26/27):** interface oracle 60 CS vs idle,
74 as red, 45 / 28 scripted mirror; MLP clone 94-97% per-step accuracy but
5-10 CS closed-loop (compounding error); DAgger-1 24 / 33; DAgger-2 45 / 50;
cross-play vs the DAgger-2 clone on the server: E04 27.5 (21-33), E06 17.3.

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
