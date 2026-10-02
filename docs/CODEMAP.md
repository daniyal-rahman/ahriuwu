# Code map: what is live, what is tooling, what is history

Updated 2026-10-01. LIVE = on the training or evaluation path today. TOOL =
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
| `lanerl_jax/train/server_train.py` | LIVE | C# server PPO collector (idle / mirror), resume, frozen eval; AFK JAX work uses wave_scenario_train.py / vec_train.py |
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
| `lanerl_jax/sim/` | LIVE | JAX simulator used by the current AFK training and frozen evaluation path |
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

TOOL: `ops/figures/afk_lr_comparison.py` regenerates the E50–E53 frozen CS/deaths/personal tower comparison in `docs/figures/E50_E51_comparison.png`; `--group debug` plots available E60/E61/E62/E63/E65/E66/E67 frozen results in `docs/figures/afk_debug_comparison.png` (skips runs without results).

TOOL: `slurm/credit_audit.sbatch` / `credit-audit` launcher engine executes indexed E54 AFK learning diagnostic; `Transition.cs_delta` is exact per-decision CS telemetry.

| `slurm/behavior_audit.sbatch` | TOOL | E58 bounded GPU frozen behavior audit via launcher; probe indexed in probes/README |

| `slurm/counterfactual_audit.sbatch` | TOOL | E59 bounded GPU same-state caster forks via launcher |

| `obs/builder.py`, `train/policy.py` own-action v4 option | LIVE (opt-in experiment) | E60 ownAA/E state inputs with zero-weight checkpoint migration; defaultv3 unchanged, C#wire guard |

LIVE (opt-in experimental): `lanerl_jax/train/click_proposals.py` projects visible hostile observation rows to screen cells and mixes learned proposal probabilities with factored ground clicks. PolicyConfig.click_proposals defaults False; E66/LEARN-AFK-19. No environment target IDs or hidden state. Shared sampler/learner and policy driver use exact joint screen-cell probabilities; only AFK wave-scenario launch is enabled until broader validation.

LIVE (opt-in): `wave_scenario_train.py:warmup_staggered_runner` discards initial shortened episodes without learning, then starts E67 from varied real game phases; frozen evaluation remains full-length and synchronized. Default off; LEARN-AFK-20.

TOOL: `slurm/afk_tick_audit.sbatch` / `afk-tick-audit` launcher engine runs the indexed E69 frozen tick diagnostic; no training or production simulator changes.

LIVE (opt-in experimental, LEARN-AFK-26): `obs/visible_history.py` associates existing visible entity rows by position/type and appends15past health/position samples plus known bits. `train/policy.py` v5 adds a zero-weight projection; `vec_train.py` carries/resets/checkpoints observer memory. Enabled only for10Hz AFK scenarios; C# driver refuses until wire history is validated. Actor never receives diagnostic IDs.

TOOL: `slurm/visible_history_audit.sbatch` / `visible-history-audit` launcher engine runs E71 GPU canaries and the indexed E68 visible association audit before bounded paired learning.

LIVE (opt-in experimental, LEARN-AFK-29): `obs/combat_features.py` v6 appends visible/public combat stats and own readiness/buffs/ranks to v3. `policy.py` zero-projection migration preserves original inputs; collector and integrated evaluator share augmentation. C# driver rejects until own readiness wire/video estimator is validated. No actor enemy clocks/target IDs or exact-HP upgrade.

TOOL: `slurm/combat_feature_audit.sbatch` / `combat-feature-audit` runs E74 GPU contract/regression tests and writes a passed preflight artifact; no learned result.

TOOL: `ops/score_combat_features.py` reads completed E75c/E76c frozen evaluations, checks matched specs/cohorts and E67 initial retention, and scores only update512 against LEARN-AFK-29 gates; no training/checkpoint mutation.

LIVE (opt-in): `train/action_masks.py` masks buttons using own observed availability; shared policy forward keeps actor/learner identical. `wave_scenario_train.py` exposes action_mask/click_mask experiment switches. Existing `actions.py` map mask supplies standable screen cells. LEARN-AFK-30, E77 automatic GPU canary.

Checkout organization, 2026-10-01: three active worktrees, confirmed against T3
thread metadata and Dani's explicit champion-work exception:

| Workstream | Branch | Checkout |
|---|---|---|
| Champion additions | `t3code/toplane-champion-overlap` | `/home/dani/.t3/worktrees/ahriuwu/t3code-12dab3cf` |
| Modern simulator | `t3code/modern-jax-sim-port` | `/home/dani/.t3/worktrees/ahriuwu/t3code-86665351` |
| RL learning proof | `lane-rl/jax` | `/srv/nfs/projects/ahriuwu-lanerl-jax` |

Five retired helper/client/boot worktrees and the old main working copy were
moved intact into `/mnt/nfs/projects/_archive/`, with original paths and restore
notes in `MANIFEST.tsv`. Dirty and ignored files and branch refs are preserved.
The archived main owns the common Git store; `/srv/nfs/projects/ahriuwu` remains
a compatibility symlink for T3 and existing dependencies. `git worktree repair`
updated registrations. Archived worktrees remain registered, deliberately.
`lanerl-vendor/LoLServer` and the non-Git `lanerl-modern-MOD001-v2` build remain
dependencies, not additional active feature workstreams. Other projects and
their jobs are outside this cleanup.

LIVE (opt-in, LEARN-AFK-33): `VecConfig.cs_only` and wave-scenario spec
`cs_only` select literal per-champion deltaCS reward, bypassing every shaping/gold
term. Frozen evaluation asserts episode reward=CS. E78 retains E67 inputs/actions/LR; E79 larger-LR attempt cancelled.

LIVE (opt-in study, LEARN-AFK-37): `train/afk_imitation.py` collects scripted and
learner trajectories through `vec_train` and trains the existing GRU with
observation-only script labels, full-prefix replay and DAgger aggregation. Its
`blue_actor` collector hook is AFK-only and explicitly refuses PPO learning.
`train/wave_evaluation.py` shares the existing frozen evaluation implementation
between PPO and imitation; `wave_scenario_train.py` accepts a complete immutable
imitation handoff with checkpoint SHA and initial physical-retention gate.
TOOL: existing `slurm/wave_scenario.sbatch` dispatches `afk-imitation` and runs
`train/tests/test_afk_imitation.py` within the integrated startup suite. E81/E82
configs select this path; STATUS records actual run state and completion bridges.
