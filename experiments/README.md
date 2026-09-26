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
