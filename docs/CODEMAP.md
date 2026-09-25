# Code map: what is live, what is tooling, what is history

Updated 2026-09-25. LIVE = on the training or evaluation path today. TOOL =
reusable, run by hand. DIAG = one-off diagnostic, versioned for reproduction.
LEGACY = kept for history only; nothing imports it.

| Path | Class | Role |
|---|---|---|
| `lanerl_jax/train/server_train.py` | LIVE | C# server PPO collector (idle / mirror), resume, frozen eval; the entry point for all current runs |
| `lanerl_jax/train/learner.py`, `ppo.py`, `policy.py` | LIVE | shared PPO loss/optimiser, factored heads, the network |
| `lanerl_jax/obs/builder.py`, `obs/frame.py`, `obs/fog.py`, `obs/vision.py` | LIVE | observation contract `viewport-structured-v3` |
| `lanerl_jax/parity/policy_driver.py` (`StateRebuilder`, wire helpers) | LIVE | wire frame → `LaneState` for the observation builder (shared with the evaluation driver) |
| `lanerl_train/vec.py`, `ports.py`, `paths.py` | LIVE | server process launch, lockstep step, port allocation, node path translation |
| `lanerl_rl/projection.py`, `constants.py`, `frame.py` | LIVE | camera model, click grid, button list, lane frame |
| `lanerl_rl/ppo.py`, `model.py` | TOOL (reference only) | the original PyTorch PPO/GAE that `train/tests/test_ppo.py` checks the JAX port against |
| `lanerl/patches/`, `lanerl/cfg/` | LIVE | vendor server patches (see its README) and the game configs |
| `experiments/` | LIVE | one launcher per experiment ID; the only way runs start |
| `slurm/server_train.sbatch`, `ops/server_train_status.py`, `ops/desktop_suite.sh`, `ops/login_capped.sh` | TOOL | launch and monitor on the desktop; capped CPU work |
| `lanerl_jax/train/jax_train.py`, `jax_farm.py`, `jax_eval.py` | TOOL (paused) | same learner on the JAX sim; not run until the C# gate is met |
| `lanerl_jax/train/trainer.py`, `run_train.py`, `slurm/rl_train.sbatch` | TOOL (paused) | the Anakin JAX trainer (256 envs). Still has the PPO-15 entropy bias |
| `lanerl_jax/sim/` | TOOL (paused) | the JAX simulator; only used by the JAX arms |
| `lanerl_jax/parity/policy_divergence.py`, `record.py`, `replay_server.py`, `render_recording.py`, `analyze_farming.py` | TOOL | sim-vs-server gate, recordings, replay viewers |
| `lanerl_jax/parity/archive/`, `lanerl_jax/probes/` | DIAG | one-off probes with README indexes pointing at ledger rows |
| `lanerl_jax/train/entropy_audit.py`, `throughput_audit.py`, `benchmark.py` | DIAG | audits cited by ledger rows |
| `lanerl_jax/runs/` (ignored) | data | run directories; each has manifest, metrics, source tarball |
| `lanerl_train/__main__.py`, `run.py`, `procactor.py`, `lanerl_rl/env.py`, `model.py`, `lanerl_bot/` | LEGACY (candidate) | the PyTorch server RL stack. Not on the current path, but `procactor.py` holds the process-parallel collector (4,829 dec/s at 96 servers) that `server_train.py` should adopt |
| `legacy/` | LEGACY | August Dreamer pipeline (`src/`, `tests/`), `scripts/` (244 files), `scratchpad/` (367 files), `lanerl_spike/` (wine/Windows client and old server scripts), audit reports. Nothing imports them (checked 2026-09-25) |
| `docs/archive/`, `docs/PORT_AUDIT_*.md`, `docs/TICK_*`, `docs/TIER*` | LEGACY docs | frozen investigations; cite, do not update |

Every test under a package's `tests/` is a regression that runs in CI-style
suites (`ops/desktop_suite.sh`); a file that measures one checkpoint once is a
DIAG and lives in `probes/` or `parity/archive/`, never in `tests/`.
