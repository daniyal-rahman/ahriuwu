# Experiments (one row per run; append, never rewrite a verdict)

Metric for the current gate: mean CS per frozen-policy 10-minute episode, C#
server, mirror self-play, over seeds. Training-time CS (changing weights) is
labelled `train`.

| Run dir (under `lanerl_jax/runs/`) | Date | Engine | Task | Learner | Budget | Result | Verdict |
|---|---|---|---|---|---|---|---|
| `server_first_20260925/dead-train/seed0` (Codex) | 09-25 | C# | idle red, near-wave, 30 Hz, 2 envs | lr 1e-5, ent 0.001, adv-norm | 175k dec | uniform buttons after 684 updates; train CS 0-19 | no learning: lr too small |
| `server_train/idle-wave-s0` | 09-25 08:42 | C# | idle, near-wave, 10 Hz, 4 envs | lr 3e-4, ent 0.001 | 2.5M dec | collapsed to `recall`, sat in fountain (train CS 3-7) | REW-11 shaping bug |
| `server_train/mirror-s0` | 09-25 08:53 | C# | mirror, from fountain, 10 Hz, 12 envs | lr 3e-4, ent 0.01 | 5.3M dec | move/attack_move/R only, entropy 9.6, CS 0.1 | PPO-15 entropy bias |
| `server_train/idle-wave2-s0` = **E02_idle_wave** seed 0 | 09-25 14:44 | C# | idle, near-wave, 10 Hz, 4 envs | lr 3e-4, ent 0.001, +xp, no adv-norm | running | train CS 4 → ~10 by 190 episodes | running |
| `server_train/mirror-wave-s0` = **E01_mirror_wave** seed 0 (5 dirs; the last is a slurm requeue from a stale checkpoint after the 23:20 reboot) | 09-25 14:45 | C# | mirror, near-wave, 10 Hz, 10 envs | same | stopped 09-26 01:20 (control, plateaued) | train CS 3.5 → ~12.5 by 190 episodes; max 27 | running |
| `E03_mirror_wave_lr1e-4/seed0` | 09-25 21:15 | C# | E01 state at update 1120, continued: mirror, near-wave, 10 Hz, 6 envs x 256 | lr 1e-4 (was 3e-4), otherwise E01 | running | — | running: is the plateau a step-size problem? |
| `E03_mirror_wave_lr1e-4/seed0` | 09-25 21:15 | C# | (stopped 22:40 after ~90 updates, superseded by E04; no eval) | | | | stopped: two larger defects found first |
| `E04_mirror_wave_snap_noxp/seed0` | 09-25 22:45 | C# | E01 state at update 1520, continued: mirror, near-wave, 10 Hz, 6 envs x 256 | lr 3e-4; ClickV3 server build (PATH-011); XP weight 0.002 | stopped 09-26 04:11 (desktop to Windows); not resumed by Dani's decision | frozen u2020: 18.8 / 24.5; train chunks 27-31 by u3070 | running: do wall-clicks (46-54% of movement clicks) and XP camping explain the plateau? |
| `E05_mirror_wave_gru_standard/seed0` | 09-26 01:20 | C# | from scratch: mirror, near-wave, 10 Hz, 10 servers x 128, ClickV3 | GRU core; PPOConfig.standard (lr 2.5e-4 annealed, lambda 0.95, entropy 0.01, clip-grad 0.5, adv-norm, no KL stop); XP 0.002 | stopped at 912 updates (2.3M dec) | frozen u860: blue 6.5 (5-8), red 10.0 (3-19); train flat 8-12; entropy pinned 9.0 | entropy 0.01 on the 3-head SUM is 3x CleanRL's per-head push; superseded by E06 |
| `E06_mirror_wave_gru_ent3/seed0` | 09-26 03:25 | C# | as E05 | as E05 with entropy 0.01/3 (now the preset default) | 6000 updates | cancelled at u280 by the Windows boot; resumes automatically | train CS 3 -> 8-9 by u280; the only arm continuing |
| `server_train/jax-mirror-wave-s0` | 09-25 15:27 | JAX | same as above | same | cancelled at 0 updates | — | JAX deferred until C# gate |

| `throughput_server_20260925/mp-w{1,3}` | 09-25 18:45 | C# | 12 envs mirror, 6 cores, E01 running alongside | probe, 6 updates | 1.0 s per 768 decisions for BOTH workers=1 and workers=3 | desktop is server-CPU-bound at ~14 servers |

Frozen evaluations (the only numbers that count for the gate):

| Eval dir (`runs/EVAL/`) | Checkpoint | Task | Episodes | Blue CS (mean / median / min-max) | Red CS | Deaths | Verdict |
|---|---|---|---|---|---|---|---|
| `server-farm-s0-20260925-202409-047737a5` | E01 seed 0, update 760 (`eval_u760.msgpack`) | mirror, near-wave, 600 s, 10 Hz, sampled | 5 envs x 1 | 19.4 / 20 / 11-27 | 16.0 / 12 / 11-29 | 0-2 per agent | below the 30-CS gate; keep training |
| `EVAL/eval_u1080.out` | E01 seed 0, update 1080 | same | 5 envs x 1 | 14.2 / 13 / 5-22 | 21.0 / - / - | - | no improvement over u760; train CS flat at 16-18 since ~u500 |
| `EVAL/eval_u2020.out` | E04 seed 0, update 2020 (500 updates after the branch) | mirror, near-wave, ClickV3 | 4 envs x 1 | 18.8 / 18.5 / 16-22 | 24.5 / 25.5 / 20-27 | - | both sides above E01's frozen 14-21; below the 30 gate; train chunks 22-29 |
| `EVAL/eval_u320.out` | E06 seed 0, update 320 (0.8M decisions) | mirror, near-wave, ClickV3 | 4 envs x 1 | 5.3 / 4.5 / 3-9 | 6.0 / 6.0 / 2-10 | - | untrained level, as expected this early (E01 was ~5 train CS at the same budget) |
| `EVAL/eval_u620.out` | E06 seed 0, update 620 (1.6M decisions) | same | 4 envs x 1 | 8.5 / 9 / 4-12 | 13.5 / 13 / 11-17 | - | learning: train chunks 10 -> 15, attack-move now 45% of actions |
| `EVAL/eval_u920.out` | E06 seed 0, update 920 | same | 4 envs x 1 | 2.5 / 2.5 / 1-4 | 1.5 / 1.5 / 1-2 | - | INVALID: OPS-004 rank loop starved level-9+ champions (one decision per 1.9 s); run resumed from u620 after the fix |
| `EVAL/eval_u1260.out` | E06 seed 0, update 1260 (3.2M decisions; resumed segment, OPS-004 fixed, 0 rank warnings) | same | 4 envs x 1 | 15.3 / 16.5 / 7-21 | 14.3 / 11 / 7-28 | - | rising again (u620: 8.5/13.5); at E01's level at the same budget |
| `EVAL/eval_u1580.out`, `eval_u1900.out` | E06 seed 0, updates 1580 / 1900 (4.0M / 4.9M decisions) | same | 4 envs x 1 each | 14.3 (8-22) / 11.3 | 24.5 (16-36) / 14.8 | - | plateau band 11-25; train chunks 13-18. Job OOM-killed at u1931 (10 GB limit); resumed at 20 GB |
| `EVAL/eval_u2200.out` | E06 seed 0, update 2200 (5.6M decisions) | same | 4 envs x 1 | 20.3 / 23 / 11-24 | 17.5 / 17.5 / 16-19 | - | best so far; train chunks 16-20, recent episodes 23-29 |
| `EVAL/eval_u2520.out` | E06 seed 0, update 2520 (6.5M decisions; leak-fixed segment) | same | 4 envs x 1 | 16.8 / 17.5 / 11-21 | 13.8 / 14.5 / 7-19 | - | plateau band 14-20 both sides since u1260; memory now +2.4 MB/update (was +10) |
| `EVAL/eval_u2840.out` | E06 seed 0, update 2840 (7.3M decisions) | same | 4 envs x 1 | 12.5 / 13 / 9-15 | 14.3 / 15 / 5-22 | - | still in the 12-20 band; train chunks 12-15 |
