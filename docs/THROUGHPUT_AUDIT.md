# Throughput audit: gate-4 acting vs end-to-end training (2026-09-24)

Written for: "gate 4 said 56k decisions/s, training does 10.6k env-steps/s --
where did 5x go, and what is recoverable?" All numbers are from the RTX 5080
(`desktop`, slurm `gpup`) at HEAD `47920ba` unless a commit is named.
`XLA_PYTHON_CLIENT_PREALLOCATE=false` as in `slurm/rl_train.sbatch`.
Compile is excluded from every steady number, and every timing is wrapped in
`block_until_ready`.

**Harness:** `lanerl_jax/train/throughput_audit.py` (modes `pool`, `split`,
`phases`, `determ`), driven by `ops/perf_audit.sh <job>`. The raw rows are in
`lanerl_jax/runs/perf/audit.jsonl` and the logs in
`lanerl_jax/runs/perf/pa-*-<jobid>.out` (jobs 1319-1330). `split` times the
REAL trainer programs `make_train(...).rollout` and `run_chunk(runner, 1)`
separately, and takes learner = update - rollout. It runs them on a
heterogeneous state pool: the trainer's own staggered rollout, 70 rollouts
(0-303 s of game, 26-53 live units) under the trained `diag1b` checkpoint's
policy. The `split` numbers are median of 5 (baseline) or 3 (knobs) steady
calls.

## Units -- the discrepancy is partly a unit mix-up

`benchmark.py` computes `decisions_per_s = n_envs / step_time`. So a gate-4
"decision" is an **env-step**: both champions act, and the sim runs 2 ticks.
It is not a champion-decision.

| Unit | Definition | Relation |
|---|---|---|
| env-step/s | one env advanced one 30 Hz decision | base unit |
| champion-decision/s | 2 per env-step | = 2 x env-step/s |
| game-min/hour | env-steps/s x (1/30 s) x 3600 / 60 | = 2 x env-step/s |

So gate 4's 56,985 env-steps/s = **114k game-min/hour**: that is where the
"~100k game-minutes/hour" came from. Training at 10.6k env-steps/s = **21.3k
game-min/hour**. Measured in the same unit, the gap is **5.36x**, not 2.6x.

## 1. The gap, decomposed (each factor measured)

| # | Step | env-steps/s | Factor | How measured |
|---|---|---|---|---|
| 0 | Gate-4 figure as quoted: acting only, **no SmoothPath**, **4096 envs**, one 150 s warm state **broadcast** to all envs, random-init policy | 56,985 (HEAD); 57,670 at `d47ab44`; 58,255 logged 09-18 | -- | `benchmark --envs 4096 --no-smooth`, job 1319 |
| 1 | + SmoothPath (training runs it: `build_local_waypoints(smooth=True)` default) | 51,640 | 0.906 | `benchmark --envs 4096` (canonical), job 1319 |
| 2 | 4096 -> **256 envs** (the training batch) | 28,155 | **0.545** | `benchmark --envs 256`, job 1319 |
| 3 | benchmark step -> the trainer's real rollout: realistic heterogeneous states + trained policy + reward, reset-where, transition writes | 23,942 (1.369 s / 128 steps = 10.69 ms/step) | 0.850 | `split` rollout, job 1320 |
| 4 | + **learner** (4 epochs x 4 minibatches x 16,384 samples, fwd+bwd+Adam) | 11,288 (update 2.903 s; learner 1.534 s = 53%) | **0.472** | `split` update, job 1320 |
| 5 | harness -> logged run (`diag1b` median chunk 61.6 s / 20) | 10,640 (3.08 s/update) | 0.943 | `metrics.jsonl` `wall_s` |

The product is 56,985 x 0.906 x 0.545 x 0.850 x 0.472 x 0.943 = 10,645,
against 10,640 logged. **The two big factors are the learner (2.1x) and the
batch size (1.8x).** Neither is a regression. Gate 4 deliberately measured
acting only, at 4096 envs.

Notes on the factors:
- **Factor 3** is partly state realism. At 4096 envs the phase-sum cost is
  75.9 ms/step on the gate's broadcast state and 90.4 ms on the heterogeneous
  pool (0.84). The broadcast state hides the deferred terrain repair almost
  entirely: 0.29 ms vs 6.22 ms (section 5). **The gate protocol flatters the
  sim by ~16%.**
- **Factor 5** is not a stall. The harness's 5-update chunk gives 2.89/2.93 s
  per update, the same as single updates. The logged per-chunk time ranges
  58.7-70.0 s and tracks game phase (section 3). The harness pool sits at
  0-303 s of game; training spans 0-600 s.
- **Game phase matters by itself.** In the pool warm-up (job 1320), the same
  256-env rollout cost 1.01 s at t~9 s (26 units, no minions), rising to
  1.39-1.44 s at t>=170 s (40-44 units): +40%. The from-scratch `determ` runs
  (first 20 updates, <=85 s of game) run at 2.43 s/update, against 2.90 s
  mid-game.

## 2. Rollout vs learner, and what the GPU is doing (XLA profile)

Profile source: `jax.profiler` traces of one rollout call and one update call,
baseline config (`lanerl_jax/runs/perf/prof_base/`). The learner was
separated by time window.

| Program | Kernels | GPU span | Kernel-busy | Composition |
|---|---|---|---|---|
| rollout (128 steps x 256 envs) | 763k (**~6,000 kernels per env-step**, mean 1.6 us) | 1.76 s | **70%** (0.55 s of inter-kernel gaps) | elementwise 49%, reductions 16%, CUB radix sorts 10%, D2D memcpy 10%, scatter 8%, matmul 4% |
| learner (16 minibatch steps) | 6.4k | 1.49 s | **100%** | matmul 51% (0.76 s), reductions 25% (LayerNorm/softmax/attention), transposes 14%, elementwise 10% |

- **The rollout is launch/latency-bound.** Its kernels are tiny, and every
  XLA `while_loop` trip (route spiral, terrain repair, line walks) adds loop
  control. Envs scale sub-linearly in cost (section 4), and this is why.
- **The learner is compute-bound but not matmul-bound.** With
  `jax_default_matmul_precision=highest` (full fp32) the learner is only 8%
  slower (1.66 vs 1.53 s). With `bfloat16` it is unchanged (1.534 s). So the
  default already runs reduced-precision GEMMs, and a precision flag buys
  nothing.
- The learner's cost is proportional to **samples**, not envs: 1.49-1.53 s
  per 65,536 samples x 4 epochs at every env count, and 3.00-3.04 s for
  131,072.

## 3. Stalls: none found

| Check | Result | Evidence |
|---|---|---|
| Per-chunk spikes | **None** except the known node incident (`diag1` chunk 13, 1,176.8 s) | Every `metrics.jsonl` under `runs/train` and `runs/bisect` (25 runs), chunk deltas vs median. No other chunk exceeds 1.1x median as a single spike. `diag1b` chunks 3-10 sit at 67-70 s (+12%): a smooth ~10-chunk hump while the staggered first episodes are mid-game, not a spike. `base1` shows the same at chunks 3-17. |
| Checkpoint writes (55 MB to NFS, every 5 chunks) | **no measurable cost** | `diag1b`: checkpoint-chunk median 61.45 s vs other chunks 61.60 s. Runs with `--ckpt-every 0` (bisect `scratch-*`): 60.5-64.3 s median, the same as runs with checkpoints. |
| Recompile per chunk | **none** | `step_fn = jax.jit(run_chunk, static_argnums=1)` gets the same `n=20` every chunk. The harness `run_chunk(r, 5)` compiled once (44 s), then ran at 2.89/2.93 s per update, equal to `n=1`. Chunk 0 of `diag1b` = 122.6 s = ~61 s compile + 61.6 s. Only a remainder chunk (`updates % chunk != 0`) would compile a second program. |
| Host round trips per chunk (metrics to host, params finiteness check, W&B `log_step`) | **<= 3.6 s per 61.6 s chunk (<= 6%) upper bound**, and not separable from the game-phase difference | logged 61.6 s vs harness 20 x 2.90 = 58.0 s. The GPU is idle only during this Python work between chunks. |
| GPU idle at 1 Hz | **never idle for a second** | `nvidia-smi dmon -s um` on `desktop`, 240 s during job 1316 (bisect training, 256 envs): SM 85-100% (mean 92.4, min 85). mem-controller alternates 0% / 80-85% with the ~3 s update period. FB 12,508 MB. **Caveat:** nvidia-smi's "SM%" is "a kernel was resident", so it reads 85-100% while the profile shows the rollout 30% idle at microsecond scale. `nvidia-smi` cannot see this bottleneck. |

## 4. Scaling knobs (steady state, VRAM = `peak_bytes_in_use`)

Harness baseline 256x128, mb 4, ep 4: **2.903 s/update = 11,288 env-steps/s,
peak 7.56 GiB**. XLA's allocator limit is 11.59 GiB (the default 0.75
fraction applies even with preallocation off).

| Config | Samples/update | Rollout s | Learner s | Update s | env-steps/s | vs base | Peak VRAM |
|---|---|---|---|---|---|---|---|
| 256 x 128, mb4, ep4 (baseline) | 65,536 | 1.369 | 1.534 | 2.903 | 11,288 | -- | 7.56 GiB |
| 512 x 64 | 65,536 | 0.927 | 1.509 | 2.437 | 13,448 | +19% | 7.61 |
| 1024 x 32 | 65,536 | 0.779 | 1.491 | 2.270 | 14,438 | +28% | 7.72 |
| 2048 x 16 | 65,536 | 0.767 | 1.534 | 2.301 | 14,238 | +26% | 8.00 |
| 256 x 128, **mb 8** | 65,536 | 1.367 | 1.533 | 2.900 | 11,300 | 0% | **4.14** |
| 1024 x 32, mb 8 | 65,536 | 0.778 | 1.485 | 2.263 | 14,479 | +28% | 4.33 |
| 256 x 128, **epochs 2** | 65,536 | 1.369 | 0.796 | 2.165 | 15,135 | +34% | 7.56 |
| **1024 x 32, epochs 2** | 65,536 | 0.779 | 0.742 | 1.521 | **21,551** | **+91%** | 7.74 |
| 512 x 128, mb4 | 131,072 | 1.849 | -- | **OOM** (10.62 GiB single alloc) | -- | -- | -- |
| 1024 x 64, mb4 | 131,072 | 1.542 | -- | **OOM** (10.62 GiB) | -- | -- | -- |
| 1024 x 128, mb4 | 262,144 | 3.084 | -- | **OOM** (25.33 GiB; matches the sbatch comment) | -- | -- | -- |
| 512 x 128, mb 8 | 131,072 | 1.850 | 3.043 | 4.893 | 13,394 | +19% | 7.89 |
| 1024 x 64, mb 8 | 131,072 | 1.544 | 3.005 | 4.549 | 14,406 | +28% | 7.99 |
| precision `bfloat16` (flag) | 65,536 | 1.367 | 1.534 | 2.901 | 11,296 | 0% | 7.56 |
| precision `highest` (fp32) | 65,536 | 1.395 | 1.660 | 3.054 | 10,729 | -5% | 7.56 |
| `XLA_PYTHON_CLIENT_PREALLOCATE=true` | 65,536 | 1.366 | 1.538 | 2.904 | 11,285 | 0% | 7.53 |

- **The acting ceiling is ~42k env-steps/s** (rollout 0.77 s per 32,768
  env-steps at 1024-2048 envs), inside the trainer on realistic states. Past
  1024 envs nothing more is gained.
- **The OOMs are the learner, not the rollout.** A 32,768-sample minibatch
  needs a 10.6 GiB buffer. `mb 8` halves the peak, so any doubled batch fits
  at the baseline minibatch size.
- **bf16 is not trivially available.** The precision flag is a no-op here,
  and `PolicyConfig` has no dtype field. Upper bound if it were plumbed
  through: matmul is 0.76 s of the 1.49 s learner, so halving it would give
  at most ~+15% at baseline. **Not measured.**

## 5. Sim step cost at HEAD vs PERF-001's 2026-09-18 profile

Each phase is its own `jit`, on fixed inputs, median of 5 x 20 calls
(`phases`, job 1321). PERF-001 used the gate state at 4096 envs with a
random-init policy. Here the policy is the trained `diag1b` one. "Parts sum"
exceeds a fused step because the phases cannot fuse across `jit` boundaries.

| ms per env-step batch | PERF-001 09-18 (4096, gate state) | HEAD 4096, gate state | HEAD 4096, heterogeneous pool | **HEAD 256, pool (training size)** |
|---|---|---|---|---|
| observation (both champions) | 10.005 | 9.953 | 9.902 | 0.927 |
| policy forward | 14.486 (+decode) | 15.701 | 15.684 | 1.406 |
| sample + decode | (incl. above) | 0.044 | 0.043 | 0.043 |
| `apply_orders` non-routing | 0.926 | 0.827 | 0.828 | 0.257 |
| routing (routed - raw control) | 4.349 | **10.914** | **18.778** | **3.337** |
| `env_advance` (training mode) | 44.737 | 38.444 | 45.172 | 5.473 |
|   of which deferred terrain repair | -- | 0.292 | **6.218** | **1.395** |
|   of which collision sweep (vs `enable_collision=False`) | -- | 3.281 | 4.143 | 1.814 |
| parts sum | 74.4 | 75.9 | 90.4 | 11.44 |

What moved:
- **Routing is 2.5-4.3x PERF-001's figure**: 10.9 ms on the gate state and
  18.8 ms on the heterogeneous pool, against 4.35 ms then. `step_decision`
  got cheaper (44.7 -> 38.4 ms), so the gate's total barely changed. The
  likely cause is the policy, not a code regression: a trained policy issues
  far Moves, and PERF-001 documents the terrain-exit spiral's p99 of 88 trips,
  where one straggler lane makes the whole `vmap` pay. HEAD and `d47ab44`
  benchmark within 1% (section 6), which rules out a code regression. The
  spiral's per-lane distribution was not re-measured under the trained
  policy.
- **The deferred terrain repair's `while_loop` is a straggler cost that the
  gate state hides**: 0.29 ms broadcast vs 6.2 ms heterogeneous at 4096.
- **At training size (256 envs), routing + repair are 4.7 of 11.4 ms (41%)**
  of the per-step sim/policy cost. Policy forward is only 1.4 ms (12%).

## 6. Regression verdict vs 2026-09-18: **no regression** (acting-only, gate workload)

Canonical `benchmark`, same GPU and protocol, jobs 1319 (HEAD vs `d47ab44` in
one job) and 1327 (intermediate commits):

| Commit | 4096 routed+smooth | 4096 `--no-smooth` | 256 routed+smooth | 256 sim-only |
|---|---|---|---|---|
| `d47ab44` (gate 4, 09-18) | 51,291 | 57,670 | 29,350 | 85,336 |
| `6bf6968` (09-22) | 52,148 | | | |
| `2813e91` (stagger) | 52,156 | | | |
| `54371e9` (baseline cleanup) | 52,169 | | | |
| `2d0fdb2` (PATH-008 routing) | 52,189 | | | |
| `a365a52` (SPELL-013) | 52,190 | | | |
| **HEAD `47920ba`** | **51,640** (+0.7%) | **56,985** (-1.2%) | 28,155 (-4.1%) | 76,502 (-10%) |

Job-to-job noise is about 1%: `a365a52` and HEAD differ only in parity tools
and measure 52,190 vs 51,640 in different jobs. So at the gate batch (4096)
the hardening (buff rewrite, SimConfig, PATH-008, slot resets, SPELL-012/013,
deferred repair) cost **nothing measurable**. At 256 envs there is a possible
small sim-only slowdown (-10% sim-only, -4% full loop, one sample each), and
at 256 envs that moves training by at most ~2%. Limitation: the benchmark
workload (a random-init policy on one broadcast state) rarely exercises the
hardened spell/buff paths. The training-state numbers in section 5 were not
taken at `d47ab44`.

## 7. Determinism (added on request): training is not reproducible across processes

Measured with `throughput_audit determ` (jobs 1324-1326, 1328, 1330): two
from-scratch 20-update runs of the baseline config, seed 0, in one process,
repeated in a second process. SHA-256 of all metrics and of the final params.

| XLA_FLAGS | Same process, 2 runs | Across 2 processes | s/update (steady, first 20 updates) | Cost |
|---|---|---|---|---|
| (default) | bit-identical | **differ** (metrics `0a9543b2...` vs `75b43c4a...`). Update 0 identical; divergence starts at **update 1**, the first update after a learner step (lane_dist 8031.061 vs 8031.018). Entropy at u20 7.450 vs 6.750. | 2.428 | -- |
| `--xla_gpu_deterministic_ops=true` | identical | **identical** (`34c81d8b...`) | 3.781 | **+56% time (-36% throughput)** |
| `--xla_gpu_exclude_nondeterministic_ops=true` | identical | identical (same bits as above) | 3.789 | +56% |
| `--xla_gpu_autotune_level=0` | identical | identical (**same bits** as above) | 3.625 | +49% |
| **autotune once and pin**: one process with `--xla_gpu_dump_autotune_results_to=F` (2.9 MB textproto), then processes with `--xla_gpu_load_autotune_results_from=F` (job 1330) | identical | **identical**: dump process and 2 load processes all `b6d72a68...` | **2.447** (dump process 2.442) | **+0.8% (noise)**. Compile also drops from ~124 s to ~25 s. |

**Cause:** GEMM autotuning picks different kernels in different processes.
Disabling autotuning alone produces exactly the deterministic-ops bits, so
there are no nondeterministic atomics in this program at these settings. The
rollout is deterministic for fixed params, and divergence enters through the
first gradient step.

**What this means for the 1.86 vs 18.69 CS rerun (13c30d4):** the two runs
were different compiled programs numerically, and chaos from update 1 onward
did the rest. **A single seed does not identify a run here.** Seed spread and
run-to-run spread are the same kind of noise, so the bisect's
one-run-per-commit CS readings are screens, not measurements.

**Recommendation:** pin the autotune results, and do not use the
deterministic flags. It is free (+0.8%, within noise), makes a run
reproducible across processes, and saves ~100 s of compile per run. The
deterministic flags cost 36-56% of throughput for the same guarantee.
Caveats:
- the pinned file is keyed to the exact HLO, so it must be regenerated
  whenever the trainer code, config shapes (envs, rollout, minibatches,
  epochs), jaxlib or GPU/driver change;
- add `--xla_gpu_require_complete_aot_autotune_results=true` so a stale file
  fails loudly instead of silently re-autotuning (not tested);
- record the file's hash in the manifest.

Pinned bits are a DIFFERENT (equally valid) trajectory from any earlier run's,
so runs from before the pin are not reproduced by it. This is a sbatch/launcher
change and was not applied here.

## 8. Recoverable speedups, ranked (measured; none applied -- training defaults unchanged)

| Rank | Change | Measured gain (env-steps/s) | VRAM | Cost / risk |
|---|---|---|---|---|
| 1 | **epochs 4 -> 2 AND 1024 envs x 32 steps** | 11.3k -> **21.6k (+91%)**, i.e. 43k game-min/h | 7.7 GiB | Two algorithmic changes at once. Needs a learning-curve check vs baseline at equal SAMPLES (not equal updates), >= 2 seeds, given the seed spread in section 7. |
| 2 | **epochs 4 -> 2** alone | +34% (15.1k) | unchanged | Half the gradient steps per sample. Sample efficiency may drop and cancel the gain. With lr 1e-5 and `grad_clipped` ~0.97, fewer steps could slow learning. Needs a check. |
| 3 | **n_envs 1024 x rollout 32** (samples/update unchanged) | +28% (14.4k). 512 x 64: +19%. >1024: nothing more | 7.7 GiB (4.3 with mb 8) | GAE window 128 -> 32 steps (1.07 s of game), so more bootstrapping on the critic. 4x more envs per batch (more decorrelated). Each env's game advances 4x slower per update, so a 600 s episode takes 562 updates per env, but CS@10 samples per update are unchanged (4x the envs). Needs a learning check. |
| 4 | **minibatches 8** | 0% speed | **7.6 -> 4.1 GiB** | Twice the Adam steps of half the size. Its value is headroom: it makes 131k-sample batches fit, which OOM at mb 4. Not a speedup by itself. |
| 5 | Sim-side: routing spiral + deferred-repair stragglers (`PERF-001`) | **upper bound +24% at baseline** (4.7 of 11.4 ms/step at 256, rollout 47% of the update). A larger share at config 1, where the rollout is 51% | -- | Needs sim changes (not in scope). First step: re-measure the spiral trip distribution under the TRAINED policy (section 5), since routing at 4096 envs is 10.9-18.8 ms now vs 4.35 ms on 09-18. |
| 6 | bf16 policy (dtype plumbed through `PolicyConfig`) | **not measured**. Upper bound ~+15% at baseline (matmul 0.76 of 1.49 s learner) | lower | The precision flag is a no-op (default GEMMs are already reduced precision). Needs code plus a numerical check of log-prob agreement between actor and learner. |
| -- | Preallocation / mem fraction, checkpoint frequency, W&B, chunk size | **0%** measured (2.904 vs 2.903 s; checkpoint chunks = other chunks) | | Nothing to recover. |
| -- | Deterministic XLA flags | **-36% (cost, not gain)** | | See section 7. Pinned autotune results give the same guarantee at +0.8% (noise) and save ~100 s of compile per run. |

Not recoverable without changing the problem: gate 4's 4096-env,
broadcast-state, no-learner number. Training at 4096 envs cannot fit a
learner batch (OOM at 131k samples with mb 4). The acting ceiling inside the
trainer is ~42k env-steps/s at 1024+ envs.

## Reproduce

```
sbatch --partition=gpup --gres=gpu:1 --cpus-per-task=4 --mem=12G --time=0:30:00 \
  --chdir=/mnt/nfs/projects/ahriuwu-lanerl-jax -J pa-<job> \
  -o /mnt/nfs/projects/ahriuwu-lanerl-jax/lanerl_jax/runs/perf/%x-%j.out \
  ops/perf_audit.sh <acting|trend|split|phases|envs|knobs|followup|determ|determ_flag FLAGS|determ_pin>
```
`split` must run first: it writes the state pool `lanerl_jax/runs/perf/pool256.npz`.
The `acting`/`trend` jobs need the snapshot worktrees
`/srv/nfs/projects/ahriuwu-perf-{d47ab44,6bf6968,2813e91}` (with `.venv-gpu`
and `data/jax_routes` symlinked to the live tree), plus the existing
`ahriuwu-bisect-*` trees.
