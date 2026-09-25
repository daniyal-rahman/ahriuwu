# ahriuwu: autonomous League of Legends

Goal: win modern League 5v5 against D2-level opponents. Current laboratory: a
ten-minute Garen mirror lane on a source-available server, trained by PPO
from random initialisation through screen clicks.

- **[STATUS.md](STATUS.md)** — what is running, what was just done, what is next. Read this first.
- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) — every run, one row, with verdict.
- [docs/CODEMAP.md](docs/CODEMAP.md) — which code is live, tooling, diagnostic or legacy.
- [docs/JAX_FIDELITY_LEDGER.md](docs/JAX_FIDELITY_LEDGER.md) — the lab notebook: every finding as a row.
- [docs/PROJECT.md](docs/PROJECT.md) — scope and roadmap; [docs/EXPERIMENT_METHOD.md](docs/EXPERIMENT_METHOD.md) — how experiments are contracted.
- [lanerl/patches/README.md](lanerl/patches/README.md) — server patches and which build is canonical.
- [AGENTS.md](AGENTS.md) — working rules for agents.

Launch a C# training run: `sbatch slurm/server_train.sbatch --envs 10 --opponent mirror --start-near-wave ...`
(see the file header). The August Dreamer pipeline is archived at
[docs/archive/README_dreamer_pipeline_2026-08.md](docs/archive/README_dreamer_pipeline_2026-08.md).
