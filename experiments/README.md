# Experiments: one launcher per contracted experiment

Every run that counts is started with `ops/launch.py <ID>` from a JSON spec
here (`<ID>.json`: args, slurm resources, port base, init/opponent checkpoints).
The launcher resolves and checks every path, refuses ephemeral-range ports
and duplicate job names, runs a 3-minute CANARY of the exact config, writes
`launch.json` into the run directory, then submits. Evaluations:
`ops/launch.py eval --ckpt <ckpt> [--opponent frozen --opponent-ckpt <ckpt>]`.
The `.sh` launchers for E01-E05 are the historical form and are not to be
copied. The run directory is `lanerl_jax/runs/<ID>/`.

Rules:
- One ID = one question, one config. Changing anything but seed or resume
  path is a new ID (copy the script, bump the number, add the row).
- Each ID owns a port base (table below) so parallel experiments never
  collide. Bases stay below 32768 (OPS-003).
- Seeds: `SEED=1 experiments/E01_mirror_wave.sh`. Resume: `RESUME=<ckpt>`.
- The desktop is the only node that runs servers. Two experiments may run
  at once (16 cores, ~1 core per server); the GPU is shared.
- When a run starts or stops, rewrite STATUS.md and add/settle its row.

| ID | Question | Port base | Engine |
|---|---|---|---|
| E01_mirror_wave | mirror self-play from the near-wave start: does PPO from scratch learn to farm? | 21700 | C# |
| E02_idle_wave | same learner, idle opponent: cleanest learner check | 21900 | C# |
| E03_mirror_wave_lr1e-4 | E01 continued from update 1120 at lr 1e-4: is the plateau a step-size problem? | 22100 | C# |
| EVAL_frozen | frozen-policy evaluation of any checkpoint | 22500 | C# |
| E04_mirror_wave_snap_noxp | E01 continued with click snapping + XP weight 0: do wall-clicks and XP camping explain the plateau? | 22700 | C# |
| E05_mirror_wave_gru_standard | from scratch, GRU core, published PPO defaults, ClickV3 | 22900 | C# |
| E06_mirror_wave_gru_ent3 | E05 with the per-head-scaled entropy coefficient | 23100 | C# |
| E07_frozen_opponent | E06 continued against a frozen E06 checkpoint (past-self opponent) | 23300 | C# |
| (next) | | 23500 | |
| E24_dagger_reference_relative | BC+DAgger-2 → reference PPO, relative gold/XP/lane only, fixed heuristic opponent, final u400 | 28300 | C# |
| E25_dagger_teacher_baseline | Frozen DAgger-2 vs heuristic, evaluation seeds 2/3 | 28500 | C# |
| E26_heuristic_teacher_baseline | Heuristic vs heuristic, matched seeds 2/3 | 28700 | C# |
| E27_reference_teacher_final | Frozen E24 u400 vs heuristic, matched seeds 2/3; requires --resume | 28900 | C# |

E24 protocol: reuse the completed server BC+DAgger-2 clone (the manifest
names `ORACLE/dagger2.npz`), fresh optimizer, 8 servers/8 cores, 400 updates,
relative gold/20 + .008 relative XP + 5 delta lane potential only, no prior
KL penalty, detached critic. gpuhog keeps this short fixed-budget test from
being preempted; hard limit 2 h. No BC retraining is needed for this arm.

Once desktop is Linux/Slurm-ready, use `ops/launch.py` from the repo root
(with the capped Python wrapper on the login node) for E24, E25 and E26;
E25/E26 each run `--seed 2` and `--seed 3`. Dry-run each command first;
canary and startup watch must pass. Run these sequentially to avoid competing
for the GPU. After E24 completes, use its explicit
`ckpt_000409600.msgpack` with E27 `--resume <path>`, seeds 2 and 3 (dry-run
first). Never use an intermediate checkpoint or silently rerun an evaluation.
E27 cannot run as a random-policy evaluation when the checkpoint is omitted.
Then run `ops/login_capped.sh 12G 3 .venv-jax/bin/python ops/heuristic_improvement.py`.
The scorer requires all 16 paired games for each policy, matching binary and
settings, and a verified u400 checkpoint initialized from the evaluated clone.
Success requires positive 95% paired-bootstrap lower bounds on relative return
and gold-difference gains versus BOTH baselines, with no lower mean own CS.
This is preliminary evidence from one training seed, not a multi-seed claim.
