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
| `server_train/mirror-wave-s0` = **E01_mirror_wave** seed 0 (4 dirs: crash, resume, resume, resume; a 5th empty dir from a bad resume path was removed) | 09-25 14:45 | C# | mirror, near-wave, 10 Hz, 10 envs | same | running | train CS 3.5 → ~12.5 by 190 episodes; max 27 | running |
| `E03_mirror_wave_lr1e-4/seed0` | 09-25 21:15 | C# | E01 state at update 1120, continued: mirror, near-wave, 10 Hz, 6 envs x 256 | lr 1e-4 (was 3e-4), otherwise E01 | running | — | running: is the plateau a step-size problem? |
| `E03_mirror_wave_lr1e-4/seed0` | 09-25 21:15 | C# | (stopped 22:40 after ~90 updates, superseded by E04; no eval) | | | | stopped: two larger defects found first |
| `E04_mirror_wave_snap_noxp/seed0` | 09-25 22:45 | C# | E01 state at update 1520, continued: mirror, near-wave, 10 Hz, 6 envs x 256 | lr 3e-4; clicks snapped to standable ground; XP weight 0 | running | — | running: do wall-clicks (46-54% of movement clicks) and XP camping explain the plateau? |
| `server_train/jax-mirror-wave-s0` | 09-25 15:27 | JAX | same as above | same | cancelled at 0 updates | — | JAX deferred until C# gate |

| `throughput_server_20260925/mp-w{1,3}` | 09-25 18:45 | C# | 12 envs mirror, 6 cores, E01 running alongside | probe, 6 updates | 1.0 s per 768 decisions for BOTH workers=1 and workers=3 | desktop is server-CPU-bound at ~14 servers |

Frozen evaluations (the only numbers that count for the gate):

| Eval dir (`runs/EVAL/`) | Checkpoint | Task | Episodes | Blue CS (mean / median / min-max) | Red CS | Deaths | Verdict |
|---|---|---|---|---|---|---|---|
| `server-farm-s0-20260925-202409-047737a5` | E01 seed 0, update 760 (`eval_u760.msgpack`) | mirror, near-wave, 600 s, 10 Hz, sampled | 5 envs x 1 | 19.4 / 20 / 11-27 | 16.0 / 12 / 11-29 | 0-2 per agent | below the 30-CS gate; keep training |
| `EVAL/eval_u1080.out` | E01 seed 0, update 1080 | same | 5 envs x 1 | 14.2 / 13 / 5-22 | 21.0 / - / - | - | no improvement over u760; train CS flat at 16-18 since ~u500 |
