# Code map: what is live, what is tooling, what is history

Updated 2026-09-25. LIVE = on the training or evaluation path today. TOOL =
reusable, run by hand. DIAG = one-off diagnostic, versioned for reproduction.
LEGACY = kept for history only; nothing imports it.

| Path | Class | Role |
|---|---|---|
| `slurm/wave_replay.sbatch` | TOOL | CPU-only E41 scenario recording/rendering through launcher; probe indexed in probes/README |
| `lanerl_jax/train/wave_scenario.py`, `wave_scenario_train.py`, `slurm/wave_scenario.sbatch` | LIVE / TOOL | E39 level3 tower-wave curriculum, finite horizon, HP-role paired resets and frozen held-out evaluations |
| `lanerl_jax/train/replay_audit.py` | TOOL | Optional frozen teacher/initial-policy comparison on full replay observation histories; no shadow action execution |
| `ops/replay_pair.py`, `slurm/replay_pair.sbatch` | TOOL | Matched final-checkpoint minimap videos via launcher; existing frozen evaluator and renderer |
| `lanerl_jax/train/paired_vec_train.py`, `slurm/paired_vec.sbatch` | LIVE / TOOL | Bounded paired initialization training using vec_train (BC/random or opt-in button-bias variant with frozen retention gate); checkpoint/signal handling and integrated frozen held-out evaluations |
| `lanerl_jax/obs/ray_kernel.py` | LIVE | Fused CUDA full-visibility traversal; CPU reference in vision.py; PERF005/PERF006 |
| `slurm/vision_smoke.sbatch` | TOOL | PERF007 production CUDA visibility integration gate via launcher |
| `slurm/bush_ab.sbatch` | TOOL | PERF006 fused-ray vs static bush-ID A/B via launcher; no saved model |
| `slurm/vision_ab.sbatch` | TOOL | PERF-005 bounded vision-kernel A/B via `ops/launch.py`; correctness first, no saved model or profiler |
| `slurm/full_profile.sbatch` | TOOL | PERF-004 full GPU diagnostic via `ops/launch.py` and `gru_profile_launch.py`; canary first, fixed workload, source traces and memory dumps, no saved model |
| `ops/gru_profile_launch.py`, `slurm/gru_profile.sbatch` | TOOL | Bounded PERF-003 GPU diagnostic through `ops/launch.py`; exact GRU split/fused canary, rollout/learner time and memory, no model checkpoints |
| `lanerl_jax/train/server_train.py` | LIVE | C# server PPO collector (idle / mirror), resume, frozen eval; the entry point for all current runs |
| `lanerl_jax/train/learner.py`, `ppo.py`, `policy.py` | LIVE | shared PPO loss/optimiser, factored heads, the network |
| `lanerl_jax/obs/builder.py`, `obs/frame.py`, `obs/fog.py`, `obs/vision.py` | LIVE | observation contract `viewport-structured-v3` |
| `lanerl_jax/parity/policy_driver.py` (`StateRebuilder`, wire helpers) | LIVE | wire frame → `LaneState` for the observation builder (shared with the evaluation driver) |
| `lanerl_train/vec.py`, `ports.py`, `paths.py` | LIVE | server process launch, lockstep step, port allocation, node path translation |
| `lanerl_rl/projection.py`, `constants.py`, `frame.py` | LIVE | camera model, click grid, button list, lane frame |
| `lanerl_rl/ppo.py`, `model.py` | TOOL (reference only) | the original PyTorch PPO/GAE that `train/tests/test_ppo.py` checks the JAX port against |
| `lanerl/patches/`, `lanerl/cfg/` | LIVE | vendor server patches (see its README) and the game configs |
| `experiments/` | LIVE | one launcher per experiment ID; the only way runs start |
| `ops/continue_heuristic_comparison.py` | TOOL | Registered one-shot E24 continuation: waits for training, runs six frozen comparisons via launcher, records/commits status and scores the fixed endpoint |
| `ops/heuristic_improvement.py` | TOOL | Read-only paired frozen E25/E26/E27 scorer for the predeclared E24 teacher-improvement test; refuses missing/mismatched cohorts |
| `ops/figures/readme_figures.py` | TOOL | regenerates every README figure (`docs/figures/`) from run metrics and `runs/EVAL/summary.jsonl`; read-only on runs |
| `slurm/server_train.sbatch`, `ops/server_train_status.py`, `ops/desktop_suite.sh`, `ops/login_capped.sh` | TOOL | launch and monitor on the desktop; capped CPU work |
| `lanerl_jax/train/jax_train.py`, `jax_farm.py`, `jax_eval.py` | TOOL (paused) | same learner on the JAX sim; not run until the C# gate is met |
| `lanerl_jax/train/trainer.py`, `run_train.py`, `slurm/rl_train.sbatch` | TOOL (paused) | the Anakin JAX trainer (256 envs), migrated to the shared reference PPO update (PPO-17) |
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

TOOL: `ops/slurm_event_test.py`, `slurm/event_test.sbatch`: OPS047 bounded Slurm-to-T3 idle wake test, launched through `ops/launch.py`.

TOOL: `ops/slurm_event_bridge.py`: reusable bounded single-job completion/failure wake into the current T3 conversation; arm after launcher health watch. OPS047 retains the separate array test.

TOOL: `ops/figures/afk_training_curve.py` regenerates `docs/figures/E46_training_curve.png` from frozen E46 evals and labeled train diagnostics.

TOOL: `ops/figures/afk_lr_comparison.py` regenerates the E50–E53 frozen CS/deaths/personal tower comparison in `docs/figures/E50_E51_comparison.png`.

TOOL: `slurm/credit_audit.sbatch` / `credit-audit` launcher engine executes indexed E54 AFK learning diagnostic; `Transition.cs_delta` is exact per-decision CS telemetry.

| `slurm/behavior_audit.sbatch` | TOOL | E58 bounded GPU frozen behavior audit via launcher; probe indexed in probes/README |
