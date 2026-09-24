# Baseline audit: bottlenecks and feasibility (hardening gate item 5)

2026-09-24. The contract format and principles come from `docs/EXPERIMENT_METHOD.md`.
Values marked **(est.)** are estimates. Everything else was measured, and the
source is given. Scratch scripts and raw outputs are in `/tmp/baseline_audit/`.
None of them is committed.

**Run audited.** Baseline config: actor lr 1e-5, critic lr 1e-5, KL stop off, masked
heads, feedforward core, 256 envs x 128 steps, 600 s episodes at 30 Hz, mirror
self-play, zero-sum reward. The code is `54371e9`; the only dirty entry is `.venv-gpu`.
- Leg A: `lanerl_jax/runs/train/diag1-20260923-223406-54371e99`, updates 0-440.
  The node was marked failed and the job requeued.
- Leg B: `lanerl_jax/runs/train/diag1b-20260923-231936-54371e99`, 1,600 updates
  resumed from A's update-400 checkpoint. Params and opt_state were restored; env and RNG are fresh.

**One update** is 32,768 env-decisions (65,536 champion-decisions).

## 0. Headline

| | Value | Source |
|---|---|---|
| CS@10 plateau (sim, mirror) | **29.4 +/- 0.8** (updates 1700-2000); peak window 31.2 (1400-1700); flat from ~update 800 | `metrics.jsonl`, both legs |
| Final checkpoint in the sim, 16 champion-episodes | **31.4 CS** (range 19-44), 3.1 deaths, 38% of enemy-minion deaths last-hit | section 2 |
| Final checkpoint in the server vs the scripted bot (n=1) | **1 CS vs bot 40**. Blue froze in `CastSpell/GarenQAttack` from ~180 s to the end | section 3 |
| PARITY-001 gate, 120 s | **FAIL**, 2 clauses: first divergence at 92,614 ms (`COLL-005` minion), and 60/7,198 unexplained champion-position intervals. Every counter matches | section 3 |
| Cost | 3.08 s/update, 10.6k env-decisions/s, GPU 86-100%, 12.5 of 16.3 GB | section 4 |
| **Hardened code `214bc3a`, job 1313** | **CS@10 0.04-0.77 over updates 180-520**. diag1 read 7.9-21 at the same updates. Cancelled at 01:15; bisect job 1314 is running | section 4 |

**Verdict (section 7):**
1. Two blockers come before any research experiment:
   - the `214bc3a` learning regression;
   - the server Q-cast freeze.
2. On the `54371e9` baseline, eval-time interventions show little headroom in last-hit timing:
   - the last-hit oracle gains +2 CS;
   - the windup lock gains +0.4 CS.
3. Where the evidence points instead:
   - **presence**: 20 of the 50 lost opportunities per episode happen while dead or walking back after a death;
   - **a policy that barely moves at lr 1e-5**: KL 0.0009 per update, the critic dominates the shared-trunk gradient 9-13x, and the movement heads are still 95% of maximum entropy after 2,000 updates.

## 1. Learning curves

![learning curves](/tmp/baseline_audit/curves.png)

- Figure: `/tmp/baseline_audit/curves.png`, from `plot.py`; the stitched data is in `stitched.json`.
- Leg A is plotted thin and leg B thick. The dashed line marks the resume at update 400.
- The grey band (400-560) is the fresh-env transient. The re-staggered first episodes are partial, so there is no CS readout until update 560.
- Windows below are means (min-max) of the 20-update chunk rows.

| Metric | 0-400 | 600-1000 | 1000-2000 | Reading |
|---|---|---|---|---|
| CS@10 | 6.9 (u160) -> 17.5 (u400); 21.2 at u420 | 28.2 (sd 1.4) | 29.4-31.2 | Rises until ~u800, flat after. The last 300 updates are 1.8 below the 1400-1700 window. |
| Factored entropy (ceiling 12.05) | 6.10 -> 9.88 at u140 | 9.28 (8.99-9.47) | 9.34 (9.22-9.50) | Flat at 77% of the ceiling. The early rise is a **composition effect**: button mass moves to move/attack_move, and those unlock the near-uniform screen heads. It is not diffusion (section 2, per head). |
| approx_kl per update | 0.0019 | 0.0009 | 0.0009 (0.0006-0.0018) | The policy barely moves: 20x below a typical 0.02 target. |
| clip_frac / dual_clip_frac | 1.45% / 0 | 0.62% / 0 | 0.61% / 0 | The PPO clip almost never binds. |
| value_explained_var | 0.70 (0.35-0.88) | 0.75 | 0.74 (0.67-0.81) | The critic tracks but does not improve after u600. |
| value_loss | 0.17 | 0.66 | 0.56 | Follows return scale. |
| returns_absmax / value_absmax | 11.7 / 3.8 | 30.5 / 18.0 | 25.4 / 11.6 | Value covers 45-60% of the return extremes (kills, deaths). |
| grad_norm (pre-clip, clip 1.0) | 4.2 | 24.7 | 11.7 | 100% of minibatches clipped after u200. The clip is a per-step renormaliser, not a safety valve. |
| lane_dist (u) | 3,047 | 1,246 | 1,216 | Settles by u600. |
| route_nonready | 8.1% | 7.7% | 4.4% | |
| Reward per term | `reward_{money,exp,hp_point,death}` = **0.0 in every row** | | | Zero-sum: the blue+red mean cancels by construction. Only shaping (~5e-5/decision) is visible. **The logged per-term breakdown is uninformative.** Per-champion values are in section 2. |

**The plateau (u800 on):**
- **Entropy:** flat.
- **Critic:** tracks at EV ~0.74.
- **KL:** 0.0009.
- **Grads:** clipped 100%.
- **Direction:** CS drifts inside +/-1.5.

Nothing in these metrics says the policy is still improving. At lr 1e-5 it is moving very slowly.

**Cross-run comparison at matched updates.** These are screens, not rankings: seed spread is large (`EXPERIMENT_METHOD.md` section 3).

| Run | Code | Config | CS@10 near u300 |
|---|---|---|---|
| sweepB-a0-control | `32a581c` | identical to diag1 | 23.3 |
| seedsB-b0-s1 | `0714fd3` | same, seed 1 | 8.4 (u280) |
| diag1 | `54371e9` | baseline | 13.3 (u300), 17.5 (u400) |
| base1 (job 1313) | `214bc3a` | baseline, 8,000 planned | **0.04 (u300), 0.27-0.64 (u460-520)** |

- Code between sweep B and diag1: the STRUCT-001 buff lifecycle, SPELL-006/007/010, AA-006, and eval-path OBS fixes. Commits are in `git log 32a581c..54371e9`.
- The other sweep B arms changed critic lr to 3e-4 and target_kl to 0.02 as well, so they cannot isolate actor lr.

## 2. What the policy does (failure cases)

**Setup:**
- Final `diag1b` checkpoint (update 2000), current tree `214bc3a`, training `SimConfig`, local route table.
- 8 envs x 18,000 decisions (one full 600 s episode each), sampled actions, seed 123.
- Ran on the desktop CPU under `systemd-run MemoryMax=6G CPUQuota=400%`, beside the training job.
- Scripts: `rollout.py`, `analyse.py`, `cancel_check.py`, `far.py`. Outputs: `rollout_ep8.npz`, `analysis_ep8.json`.

**Replication check:** `windup_lock.py none` re-implements `trainer._env_step.one` and reproduces the 16 final CS values exactly.

**The trained policy does not break on the newer sim.** It scores **31.4 CS** there, matching its training readout of 29-31.

| Measure | Value (per champion-episode unless stated) |
|---|---|
| Action mix, all decisions | attack_move 75.4%, move 22.1%, E 1.0%, Q 0.54%, R 0.46%, noop 0.44%, recall 0.03%, W 0.02% |
| Action mix by minute | attack_move 87% in min 0, 70-75% in min 2-9; move 12% -> 22-27% |
| Presses | E 186 (28.2 spins), Q 98, R 84 (R rank 1 only from level 6), W 3, recall 5 |
| CS cumulative by minute | 0, 0, 4.1, 7.0, 10.5, 14.6, 18.4, 22.1, 27.0, 31.4 (about 4 CS/min from minute 3) |
| Occupancy | in lane corridor 57%, alive outside corridor 33%, dead 9.8%, recall channel 0.1% |
| Deaths | 3.1 (50 in 16). Turret was targeting the victim in 3 of 50. Deaths at 157-590 s. |
| Damage taken by source **(approx. attribution: split over units targeting the champion)** | minions 3,192 HP, enemy champion 673, turret 163, unattributed 222 |
| Gold / XP / level at 10 min | gold 1,238-2,623; level 6-10. Blue minus red: gold -1,385..+125, XP -4,025..+1,395; blue trails in XP in 7/8 envs (n=8, untested for significance). |
| Per-champion reward, mean abs episode sum | money 11.9, exp 6.6, hp_point 2.5, death 1.0, shaping 0.95 |

**Per-head entropy of the final checkpoint.** Source: `headent.txt`; 4 envs x 128 steps at t=64/243/422 s.

| Head | Entropy | Max | % of max | Max prob |
|---|---|---|---|---|
| button | 0.71-0.73 | 2.08 | 35% | p(attack_move) 0.72 |
| target (pointer, ~24 valid slots) | 0.68-0.74 | 3.2 | 22% | 0.70-0.75 |
| screen_x | 4.34 | 4.564 | **95%** | 0.040 (uniform 0.010) |
| screen_y | 3.84 | 3.989 | **96%** | - |

After 2,000 updates the movement heads are essentially untrained. The policy navigates by `attack_move` on a target.

### Where CS is lost

Enemy lane-minion deaths average **81.6 per champion-episode (1,306 total)**. Each was classified by the champion's distance at death and by what the champion was doing while the minion was one-hit. "One-hit" means HP <= the champion's AA damage D(L) = 78.1 + 4.05 x growth(L); minion armour is 0.

| Cause | <=300 u | 300-800 u | >800 u | Total |
|---|---|---|---|---|
| **last-hit by the champion** | 31.1 | 0.3 | - | **31.4** |
| champion dead | 1.9 | 2.1 | 2.0 | 6.0 |
| out of reach while one-hit | 1.0 | 2.0 | 15.4 | 18.4 |
| never one-hit before dying (needed anticipation) | 0.8 | 1.0 | 4.9 | 6.7 |
| AA cooldown not ready throughout the window | 7.2 | - | - | 7.2 |
| own swing in flight on it, killed first (late/stolen) | 4.9 | - | - | 4.9 |
| ready, in reach, not attacking (buttons in the window: attack_move 67%, move 29%) | 4.2 | 0.1 | - | 4.3 |
| E spinning (cannot attack) | 2.7 | - | - | 2.7 |

- **Absence dominates the far column.** For 231 of the 252 far misses where the champion was alive and outside the corridor, it had died within the previous 60 s. The median distance was 6,026 u from the minion and 8,435 u from the enemy champion, so the champion was not being zoned.
  - Death plus walk-back costs **~20 opportunities per episode (est. 6.0 + 14.4)**. That is ~6.5 minion deaths per death, against a death reward of -1 (about 1.5 melee minions of `money`).
- **Swings.**
  - Of 3,678 swings started on enemy minions: 332 killed, 681 were non-lethal pushes, and 92 lost the target to another unit first.
  - **2,573 (70%) were cancelled by the policy's own next order**: an attack on another unit 1,277, a move 723, an attack re-issued on the same target 475, other 86.
  - A completed windup takes 9 decisions. Cancelled windups ended after a median of 3.
  - Killing swings started at minion HP/D median 0.69 (p90 1.2), so target choice at swing start is sensible.

**Eval-time headroom diagnostics** (`windup_lock.py`):
- Same checkpoint, same seed, 8 envs, the full episode, applied to both champions.
- The standard error of a 16-champion mean is **~1.7 CS (est.** from sd 6.9; envs are paired).

| Intervention | CS@10 | Deaths | What it tests |
|---|---|---|---|
| none | 31.4 | 3.1 | replication |
| windup lock (NOOP while own AA winds up) | 31.9 | 3.0 | Do self-cancelled swings cost CS? **No.** |
| last-hit oracle: attack a one-hit enemy minion in AA reach | 33.1 | 2.9 | Headroom in last-hit timing at the policy's own positions: **~+2** |
| same, reach + 100 u | 33.8 | 3.2 | same |
| screen heads at temperature 0.3 / 0.1 | 26.7 / 22.3 | 3.6 / 4.3 | Is there latent direction in the movement heads? **No.** Sharpening hurts. |
| retreat at HP < 30% / < 45% (move to own turret if targeted, else recall) | 23.4 / 25.9 | 0.6 / 1.0 | Is dying the cost? It removes 2-2.5 deaths but loses 5.5-8 CS: **a crude retreat costs more than the deaths**. |

**What this cannot distinguish.** Near-miss timing is not the bottleneck (+2 CS). Presence is where the remaining opportunities go. Whether better HP management would recover them is not settled: the only test was a scripted recall rule, and its recall trips cost as much as the deaths.

## 3. Server check

**PARITY-001 gate.** Final `diag1b` checkpoint, 120 s, mirror, sampled, `LANERL_AUTOBUY=0`.
- Build: `bin/Trace/net6.0`, `--config` given as an absolute path.
- Report: `lanerl_jax/runs/parity001/diag1b-120s/report.json`; log: `/tmp/baseline_audit/gate.out`; wall 86 s + 132 s recording.

| | Result |
|---|---|
| Verdict | **FAIL** (floor `obsharden-b0ctl-120s/floor.json`: 0 divergent ticks) |
| First scored divergence | t=92,614 ms, `LaneMinion.pos` of red wave-1 melee B, +1.125/+0.44 u. The same tick, minion and mechanism as the `COLL-005` triage (deferred terrain repair, `APPROX`, kept for training by decision `214bc3a`). |
| Champion position, per-decision resync | 6,930 / 7,198 within 1/16 u; 208 PATH-001 (2.89%); **60 unexplained**, all "same waypoints, still off", max 15.6 u. The listed examples are all blue, t=65-81 s: 22 attack orders on the red outer turret (NetId 1073742821) from ~5 km away, plus 3 moves. **Undiagnosed and unowned.** obsharden had 0 unexplained. |
| Other fields | `Champion.gold` from 93,648 ms (the REW-10 one-quantum residual); minion `only_in_*` from 92,681 |
| Counters, sim/server | blue: casts 6/6, E starts 6/6, ends 6/6, cancels 2/2, E-cd edges 12/12, AA hits 0/0, CS 0/0, deaths 0/0, gold 56.0/56.1, level 1/1. red: 4/4, 4/4, 4/4, 3/3, 8/8, 0/0, 0/0, 0/0, 56.0/56.1, 1/1. **Every gated counter gap is 0.** |
| Orders applied | 3,600 decisions per side, 0 missed. blue: attack 2,956, move 555, noop 72, cast 16. red: attack 3,026, move 500, noop 61, cast 11, recall 1. |
| Coverage limit | 120 s contains no fights, CS or deaths, so the gate still says nothing about the last-hit path. The floor is also only 120 s. |

**Harness hazard found (not a code change).** The first attempt passed `--config lanerl/cfg/garen1v1_trace.json` as a relative path.
- The server silently loaded a default config: **Shaco vs Ezreal** with different stats (hp 637/536 vs 672).
- The gate caught it only because HP differs at t=0 (`FAIL at 0 ms`).
- The counters (6/0 casts) looked like a sim bug.
- Kept at `lanerl_jax/runs/parity001/diag1b-120s-relcfg-shaco-ezreal/` and `/tmp/baseline_audit/gate_relcfg.out`.
- `policy_divergence`/`record` should resolve `--config` to an absolute path, or refuse a relative one.

**Against the scripted bot.**
- Command: `tools/rl_eval_vs_server.py --episodes 1 --max-game-ms 600000`, `LANERL_AUTOBUY=0`.
- Output: `/tmp/baseline_audit/eval_vs_bot.json`, server log `eval_vs_bot_logs/instance000.log`.

| | Agent (blue) | Bot (red) |
|---|---|---|
| CS@10 | **1** | 40 |
| Gold / level | 1,433 / 2 | 1,883 / 9 |
| Deaths | 1 | 0 |

**Cause: a server-side freeze.**
- `LANERL_MOVE` rows show blue in `order=CastSpell canmove=False casting=GarenQAttack` from t=180 s: alive, at (2862, 13402) near the red turret. It died there at ~200 s.
- After respawning it stayed frozen at the fountain (26, 264) in the same state until 600 s. Every later 60 s trace row shows `off=1713`.
- It lost 360 s of play, so the CS number measures the freeze and not the policy.
- The sim has no such state. The policy presses Q ~98 times per episode, so this matters beyond one run.
- **No ledger row covers a persistent `GarenQAttack` cast.** SPELL-013 covers only the linger of Q ends.
- The eval printed 65 `cast_unranked` orders. The eval's own docstring deviation (Q shown ready during its window, `OBS-02`) was live.

## 4. Measured cost

| Quantity | Value | Source |
|---|---|---|
| Wall per update, steady | **3.08 s** (median 20-update chunk 61.6 s leg B, 65.4 s leg A) | `metrics.jsonl` `wall_s` |
| Env-decisions/s | **10.6k** steady; manifest 10,429 incl. compile | diag1b manifest |
| Compile + first chunk | 122.6 s (diag1b), so compile ~61 s. base1 (`214bc3a`): 173 s, so compile ~107 s | chunk 0 `wall_s` |
| Stall | leg A chunk 260-280 took 1,177 s (the node incident) | leg A `wall_s` |
| 2,000 / 8,000 updates | 1.7 h / 6.8 h, plus compile | derived |
| GPU (job 1313, 5 samples, 01:00 UTC) | util 100, 100, 86, 86, 100%; **12,508 MiB of 16,303** (77%); 105-255 W | `nvidia-smi` |
| Host memory, training process | RSS 4.36-4.47 GB of 12 GB reserved | `ps` on desktop |
| Host CPU, training process | 121-132% of the 4 cores reserved | `ps`; `scontrol show job 1313`: cpu=4, mem=12G |
| Sim CPU rollout (8 envs, 4 cores) | 18,000 decisions in 330 s, so ~440 env-decisions/s; compile 19 s; RSS 3.7 GB | `rollout.py` log |

**Headroom:**

| Addition | Estimate | Basis / what must be benchmarked |
|---|---|---|
| Bigger model | VRAM headroom 3.8 GB (23%) **(est.)**. `XLA_PYTHON_CLIENT_PREALLOCATE=false`, so 12.5 GB is the allocator high-water mark, not the minimum need. | Benchmark: peak `memory_stats()` split between rollout and learner at the intended size. |
| GRU core, BPTT over 128 | Activations ~16,384 x 512 x ~6 gates x 4 B ~ 0.2 GB per minibatch **(est.)** | Benchmark fwd+bwd+opt at 128 seqs x 128 steps. Rollout then carries hidden state (small). |
| KL reference policy | One extra forward per learner sample. Less than 1/3 of learner fwd+bwd **(est.)**; its share of the 3.08 s is unknown | Benchmark the rollout-vs-learner time split first (`lanerl_jax/train/benchmark.py`). None of these runs recorded it. |
| Temporal transformer | not estimated | KV/sequence buffers (method doc section 6) |
| CPU | 2.7 of 4 reserved cores idle | GPU-bound. Extra CPU does not speed training. |

**Hardened-code run, job 1313.**
- Run dir: `lanerl_jax/runs/train/base1-20260924-004438-214bc3a4`; log: `rl_base1-1313.out`.
- It ran 520 updates at 3.3 s/update (median chunk 65.9 s).
- **CS@10: 0.19 (u180), 0.05 (u220), 0.46 (u260), 0.04 (u300), 0.77 (u340), 0.54 (u380), 0.36 (u420), 0.64 (u460), 0.27 (u500).** diag1 read 7.9-21 at the same updates.
- Lane distance was comparable (1,000-2,800), so the champions reach the lane.
- Entropy was 0.2-1.1 lower, and EV mostly higher (0.80-0.96; 0.40 at u500).
- route_nonready is ~0 by definition since PATH-008 (SERVER_NULL is now counted as exact). This is a metric change, not a behaviour change.
- Cancelled at 01:15:33. `slurm` job 1314 (`runs/bisect/bisect_cs.sh`) started at 01:15:49.
- **Evidence for the bisect:** the `diag1b` policy scores 31.4 CS on the `214bc3a` sim (section 2). So the regression is not in the last-hit mechanics that a trained policy uses. It is in learning from scratch, or in something early-game that stops the first last hits being discovered.
- Changes in `54371e9..214bc3a` under `train/` and `sim/`:
  - `reward.py`: ambient-rate constant;
  - `trainer.py`: route_nonready metric only;
  - `sim/rewards.py`: level_for_xp indexing, ambient gold timer;
  - `sim/step.py`, `sim/spells.py`: SPELL-012/013 linger rows, PATH-008, ENT-06/07/10/12.

## 5. Correctness-check inventory

Static read of `lanerl_jax/*/tests` and `lanerl_jax/tests`; not executed here.

Kinds: **N** = checked against numbers worked by hand or by an independent implementation; **S** = checked against the server or C#; **P** = property/invariant; **L** = lint/structural.

| What it guarantees | Files (tests) | Kind |
|---|---|---|
| GAE with a mid-rollout reset and a final bootstrap; masked log-prob and gradient vs numpy; clipped value loss; target_kl freezes params and opt state; a NaN KL withholds the update | `train/tests/test_ppo_hand_worked.py` (5) | N |
| GAE, surrogate, value loss and dual clip vs the torch `lanerl_rl.ppo`; entropy ceiling 12.050; log-prob counts only the used heads | `train/tests/test_ppo.py` (9); **skipped without torch** | N |
| Actor/learner log-prob agreement, through the loop for every head class; masked slots never sampled; reset is a `where`; episode 600 s vs horizon 120 s; phase stagger; cs@10 NaN until an episode ends and excluding partials; target_kl enforced; critic_lr applied | `train/tests/test_trainer.py` (14) | P |
| Policy head widths; padding has no effect; pointer target; jit/vmap; initial heads uniform | `train/tests/test_policy.py` (11) | P |
| Action decoder vs the reference projection, both sides | `train/tests/test_actions.py` (3) | N/P |
| Every config field is read (RL-004); manifest and `--resume` argparse round-trip | `test_config_fields_read.py` (3), `test_run_manifest.py` (3) | L |
| Lane potential; shaping telescopes; one-step zero-sum [0.871, -0.871] | `train/tests/test_lane_approach.py` (11) | N/P |
| Gold/XP/kill rules, ambient gold, hit-flag, turret gold split | `sim/tests/test_rewards.py` (29) | N/S |
| Obs layout, fog, quantisation, cast memory; obs availability == cast gate | `obs/tests/test_obs.py` (13), `sim/tests/test_obs_matches_cast_gate.py` (4) | P |
| Sim mechanics vs C# rules: movement, collision, terrain, minion AI, update order, missiles, level tables, orders, waves | `test_movement_jax` (8), `test_collision` (15), `test_terrain_jax` (8), `test_minion_ai` (19), `test_update_order` (23), `test_missiles` (7), `test_level_tables` (7), `test_orders` (11), `test_lane` (18, tier-3 distributional), `parity/tests/test_waves` (9, exact vs a 300 s recording) | S |
| Spells and damage formulas | `test_spells` (56), `test_combat` (20), `test_e_edge_trace_golden` (3) | N/P |
| Invariants: turrets never move; slot reuse; buff lifecycle under random streams; training `SimConfig` == trainer, gate differs only by allow-list | `test_invariants` (2), `test_buff_lifecycle_properties` (9), `test_sim_config` (6), `test_step_phase_order` (3, L) | P/L |
| Parity instruments name corrupted fields; divergence gate first-tick/field; resync scoring | `parity/tests/test_trace_and_diff` (19), `test_policy_divergence` (13), `test_tier15` (12), others | P |
| Eval driver: NetId map, action-log round trip, OBS-04/05 rebuild, W passive | `parity/tests/test_policy_driver.py` (12) | P/S (synthetic wire) |
| CS@10 oracle equal in sim and server | `parity/tests/test_last_hit_gate.py:228` | S; **`slow` and needs a live server**, so not run by default |
| Lints: buff-lane indices, canonical commands, ledger IDs append-only | `lanerl_jax/tests/*` (18) | L |

**Gaps: where a bug could hide with no test covering it.**
1. **Checkpoint round trip.** Nothing checks save -> load -> equal params, opt_state and step. `--resume` is only argparse-tested, and this audit's run was stitched through a resume.
2. **Time-limit truncation.** `gae()` has no truncation input, so the 600 s end and each env's random first-episode deadline are treated as terminal (PPO-09 `BOUNDED`). No test either way.
3. **Advantage normalisation**, the loss composition `pl + vc*vl - ec*H`, and the fact that `clip_by_global_norm` covers actor and critic together are untested. The last one matters: section 7 shows the critic owning the clipped norm.
4. **Zero-sum over a whole rollout.** Only one primed step is tested. Also, **the per-term training metrics are structurally zero** (section 1), so a broken term would not show.
5. **Nothing checks the cs@10 value** against a known last-hit count.
6. **Red-side decoding inside the rollout.** A symmetric bug would pass the mirror tests. The blue XP deficit in 7/8 envs (section 2) is unexplained.
7. **Eval/server path:**
   - no whole-observation diff of a recorded server state vs the sim obs;
   - no test that the server ran the intended champions (the relative `--config` hazard, section 3);
   - no detection of a champion frozen in a cast.
8. **Suite coverage.** `ops/sim_tests_per_file.sh` covers only `sim/tests`; `train/`, `obs/` and `parity/` tests are in no per-file gate. Default runs skip silently without torch or Content, and the `slow` server gates never run by default.
9. **Per-head learning.** Only the factored entropy is logged. The movement heads at 95% of maximum were invisible in training metrics (section 2).

## 6. Open fidelity items

Source: `docs/JAX_FIDELITY_LEDGER.md` rows with status OPEN, APPROX, SUSPECTED or BOUNDED; line numbers are given. The judgement asks one question: can this affect **CS@10 in this baseline**, or **sim-to-server transfer** of this policy?

Named in the brief but **not open**: GATE3-001 `VERIFIED`, FLOOR-001 `VERIFIED`, SPELL-013 `FIXED`, TEST-001 `FIXED`, REW-10 `FIXED` (the one-quantum residual is still visible in this gate), PATH-008 `FIXED`.

| ID (line) | Status | Judgement for this baseline |
|---|---|---|
| **COLL-005** (264) | APPROX, kept by decision | **Low for CS.** 75/7,201 ticks, minions only, max 20 u. It is this gate's first divergence and will keep failing it until minion scoring moves to per-tick injection. |
| **PATH-001** (160) | APPROX | **Med for transfer.** 2.89% of intervals here, all Moves. Moves are only 22% of this policy's decisions, and it navigates by attack targets. |
| **OBS-02** (282) | APPROX | **Med for transfer.** In the server, Q shows as ready during its own window. The policy presses Q ~98 times per episode, and the server eval froze in `GarenQAttack` (section 3). |
| **OBS-11 / STAT-003** (289/193) | OPEN / APPROX | **None for CS.** The MR gap (32.1 vs 44.16) only affects R/magic damage; minions and AAs are physical. Obs shift 0.06. |
| **ENT-13** (262) | SUSPECTED | **Low.** Only corpse positions are pushed. |
| **ENT-03** (252) | OPEN | **Low for CS, real for League.** Turret aggro rules are copied from the server, so this transfers to the server. Only 3/50 deaths were under turret fire. |
| **ENT-08** (257) | OPEN (low) | **None.** Inner turrets are out of reach in 10 min. |
| **PPO-09** (276) | BOUNDED | **Low-med.** 600 s and random first-episode deadlines are treated as terminal. t/600 is observed, but the stagger deadline is not. It adds 256 spurious terminals per run and biases values late in the episode. |
| PPO-13 (280) | OPEN (design) | Low-med. The button head is not masked by `cast_locked`, so presses are wasted. The R press rate of 0.46% with rank 0 until level 6 is one example. |
| REW-05 (292) | BOUNDED | **Med.** Death = -1, about 1.5 melee minions of reward. Measured opportunity cost here is ~6.5 minion deaths per death (section 2). |
| GATE3-002 / GATE3-003 (226/228) | OPEN | Med: a direct CS/XP measurement. Over 24 seeds the CS difference is 0.00 [-1.27, 1.27], but deaths are +0.54/episode higher in the sim. Deaths are this baseline's largest presence cost. |
| PATH-007 (309) | OPEN | Med. The fountain exit route is 1.73 s late. A candidate for the blue-side XP deficit? Untested. |
| SPELL-001 (237) | FIXED, semantics OPEN | Med-high. E cadence is 28 spins per episode. The early-cancel semantics are unsettled. |
| ITEM-001 (191) | APPROX | None with `LANERL_AUTOBUY=0`, as used here. High if autobuy is on. |
| GATE1-003, ORDER-003 (306/197) | OPEN / BOUNDED | Med, small: minion retarget residuals are inside the order floor. |
| SPELL-009 (245), PATH-002/003/005, OBS-001/08, ENT-04 | APPROX | Low-med: action/obs details. None is on the last-hit path. |
| PATH-004, COLL-002, MOVE-000, BOUND-001, RESET-*, PROG-001, METH-001, GATE1-001, ENUM-001, ENT-11, PATH-006, PERF-001, OPS-001, STRUCT-004, SPELL-000, SCOPE-001 | BOUNDED / APPROX | None or low for this baseline: bounds never bind, reset/injection tooling only, or out of slice. |

No open row lies directly on the last-hit path: minion HP, AA damage, windup, missile, gold on kill. Those rows are all VERIFIED or FIXED.

## 7. Bottleneck verdict and candidate experiments

| # | Suspect | For | Against / cannot tell |
|---|---|---|---|
| B0 | **`214bc3a` learning regression** (blocker) | CS@10 < 1 for 520 updates vs 8-21 at `54371e9`, same config | Seed not ruled out, but 0.3 is far below every earlier seed (8.4-23.3). Bisect running. |
| B1 | **Server Q-cast freeze** (blocker for any server number) | 360 s frozen in `GarenQAttack`; CS 1 vs 40 | n=1; not reproduced yet. |
| B2 | **Optimisation throughput**: lr 1e-5 plus a critic-dominated shared trunk | KL 0.0009/update and clip 0.6% (section 1). The movement heads are uniform after 2,000 updates (sharpening does not help, so there is no latent signal). The pre-clip gradient split below: the value term is 9-13x the policy term on the shared trunk, and 100% of minibatches are clipped. | Sweep B's higher-lr arms were worse at 300 updates, but critic lr 3e-4 and target_kl 0.02 changed with them. Adam largely undoes a constant clip scale on the actor heads. |
| B3 | **Presence**: deaths and walk-back | ~20 of 50 lost opportunities per episode come from dead or post-death walking (92% of far misses are within 60 s of a death). 3.1 deaths; 75-79% of damage taken is from minions (approx. attribution). | A crude retreat removed deaths but cost CS. Unknown whether a learned HP policy has headroom, and B2 may be why it has not been learned. |
| - | Last-hit timing / swing commitment | not supported | Oracle +2 CS; windup lock +0.4. |

**Gradient split** (`gradsplit.py`, `/tmp/baseline_audit/desktop/gradsplit_4.json`):
- Final checkpoint, one 128-step batch of 4 envs x 2 champions (1,024 samples) at each game time.
- Norms are before the clip; "trunk" is everything except Dense_7-10 and `value_head`.

| t (s) | policy: heads / trunk | value: head / trunk | total | Value/policy on trunk | Clip factor |
|---|---|---|---|---|---|
| 124 | 1.98 / 1.64 | 0.39 / 3.40 | 4.4 | 2.1x | 0.23 |
| 243 | 1.34 / 1.20 | 1.07 / 10.8 | 10.6 | 9.0x | 0.094 |
| 422 | 1.38 / 1.22 | 0.94 / 16.3 | 16.6 | 13.4x | 0.060 |
| 542 | 1.91 / 1.70 | 0.79 / 17.2 | 17.3 | 10.1x | 0.058 |

Returns in these batches are |R| <= 4. Training batches reach 20-30, so the imbalance in training is likely larger **(est.)**. Entropy-term gradients are ~0.01.

**Ranking: the blockers first, then experiments by information per GPU-hour.**

**E0. Bisect the `214bc3a` regression** (blocker; job 1314 already running)
- **Suspected bottleneck:** a `54371e9..214bc3a` sim/reward change stops the first last hits being discovered.
- **Existing evidence:** the base1 vs diag1 table (section 4). The trained policy is unaffected on the new sim (31.4 CS).
- **Implementation check:** the same seed and config at each bisect commit; cs@10 and cs_episodes at u300-400; provenance guard (`parity/provenance.py`, METH-001).
- **Measured cost:** 400 updates = 21 min at 3.08 s/update. About 4 bisect points (9 commits) = **1.4 GPU-h**.
- **Success criterion:** one commit moves cs@10 at u400 from >= 8 to < 2, reproduced on a second seed.
- **Stopping criterion:** if no single commit reproduces it, run 2 seeds of `54371e9` vs `214bc3a` to u600 before treating it as seed variance.

**E1. Server Q-freeze** (blocker; CPU only)
- **Suspected bottleneck:** the server leaves the champion in `CastSpell/GarenQAttack` with `canmove=False`, persisting through death and respawn.
- **Existing evidence:** `eval_vs_bot_logs/instance000.log`, from t=180 s.
- **Implementation check:** re-run with `--trace-every-s 5` and the recorded orders via `--replay` to reproduce; check whether the Q-empowered target died or went out of range mid-cast.
- **Measured cost:** ~6 min wall per 600 s episode on the login node (1.7-2.2x realtime); 0 GPU.
- **Success criterion:** a root-caused ledger row, then 4 episodes vs the bot with no frozen champion (every 5 s trace row moves).
- **Stopping criterion:** none. It is a prerequisite for any server claim.

**E2. Critic/trunk decoupling at the same lr** (tests B2)
- **Suspected bottleneck:** the critic owns ~90% of the clipped gradient on the shared trunk, so the actor learns through features shaped for value prediction.
- **Existing evidence:** the gradient-split table; EV 0.74 flat; value covers 45-60% of return extremes; screen heads untrained.
- **Implementation check:**
  - arm (a): a stop-gradient from `value_head` into the trunk, with a separate critic MLP on the same encoder output;
  - arm (b): separate global-norm clips for actor and critic.
  - Before any run: a unit test that the policy-loss gradient is unchanged by the value term, and a logged per-group gradient norm.
- **Measured cost:** arm (a) adds one MLP forward/backward. Its time is **unknown: benchmark first.** 2 arms x 2 seeds x 2,000 updates ~ **6.8 GPU-h** at 1.7 h each.
- **Success criterion:** at u2000, cs@10 above the diag1 1700-2000 window (29.4) by more than the seed spread, in both seeds; and screen-head entropy below 90% of maximum.
- **Stopping criterion:** at u1000, cs@10 not above control in either seed, and the policy share of the trunk gradient not higher. Inconclusive if the seeds disagree.

**E3. Actor step size, isolated** (tests B2's lr half)
- **Suspected bottleneck:** at lr 1e-5 the policy moves 0.0009 KL per update, so the plateau is slow optimisation, not a converged objective.
- **Existing evidence:** KL, clip_frac and entropy are flat from u800. Sweep B changed lr together with critic lr and target_kl, so it does not answer this.
- **Implementation check:** change ONLY the actor lr (3e-5 and 1e-4) against the exact diag1 config, with KL stop off, so approx_kl is the dose readout. Per-head entropy logged.
- **Measured cost:** 2 arms x 2 seeds x 1,000 updates ~ **3.4 GPU-h**.
- **Success criterion:** cs@10 at u1000 at or above diag1's u1000 value (28-29) plus the seed spread, with approx_kl rising in proportion.
- **Stopping criterion:** cs@10 at u600 below diag1's u600 (27) in both seeds, or kl_ref drift with a CS drop (the lr 3e-4 failure mode).

**E4. Presence / HP management diagnostic** (tests B3; cheap and runs first; then possibly a reward or curriculum arm)
- **Suspected bottleneck:** deaths to the minion wave cost ~6.5 opportunities each, and REW-05 prices a death at ~1.5 minions.
- **Existing evidence:** the section 2 tables; retreat diagnostic negative for a crude rule.
- **Implementation check:** more eval-time interventions on the final checkpoint, 0 GPU:
  - retreat to the turret only, without recall;
  - untarget-on-low-HP.
  - If one recovers at least 3 CS, the headroom is real.
  - Then one training arm: death weight -1 -> -3 (the gold-equivalent of ~6 minions), 2 seeds.
- **Measured cost:** diagnostics are ~6 min of CPU each. A training arm is 2 x 1.7 GPU-h.
- **Success criterion:** diagnostic +3 CS at 16 champion-episodes; training arm: deaths down and cs@10 up at u2000 in both seeds.
- **Stopping criterion:** diagnostics give < +2 CS, which means presence is not recoverable by HP play and the loss is structural (walk distance, respawn).

**What the evidence cannot distinguish.**
- B2 and B3 may be one problem: a slow optimiser would also be why HP management was never learned.
- All section 2 numbers come from one checkpoint and one seed, with 16 champion-episodes (SE ~1.7 CS). The diagnostics are screens.
- The per-term reward metrics are structurally zero. So nothing in the training record says which term moved CS. The per-champion sums in section 2 are the only term-level evidence.
