# Experiments: one launcher per contracted experiment

Every run that counts is started from a script here, never from an ad hoc
command line. The script name is the experiment ID used in
`docs/EXPERIMENTS.md`; the run directory is `lanerl_jax/runs/<ID>/`.

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
| (next) | | 22100 | |
