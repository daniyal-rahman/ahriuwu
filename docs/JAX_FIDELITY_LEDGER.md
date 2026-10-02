# STATUS (rewrite in place; last edit 2026-09-29, Codex)

**E46 AFK TRAINING — Slurm1787:** User selected single AFK arm with health/tower shaping.
8 GPU canaries,180s health watch and compiled endpoint gate passed. Baseline64-game frozen eval complete; PPO updates active/finite at~3.4s/update.120s rounds,10.027M learner decisions; source E40u5795.
Frozen u0 blue CS4.9375(lowHP)/5.96875(fullHP), deaths1.0/.875 per120s.
Run `/mnt/nfs/checkpoints/lanerl-jax/E46_afk_farm/vec-s0-20260930-214903-32499e76/`.
Health penalty100gold/fullbar; enemy top outer tower900gold/fullbar damage.

**E40 CHECKPOINTED / INTERRUPTED TO FREE GPU FOR E46:** User requested AFK now.
Job1780 exited cleanly after SIGUSR1 at5795updates (189890560 decisions).
Params/optimizer/full rollout state preserved in `/mnt/nfs/checkpoints/lanerl-jax/E40_tower_wave_extended/vec-s0-20260930-150246-c731fd96/ckpt_189890560.msgpack`.
No automatic resume/watchdog; E46 is the new active training task.

**E45b COMPLETE — CPU1786 exit0 in14m29s:**8canaries/180s watch passed.
HP-sensitive actor; critic V−.038 vs16 full-horizon returns mean+.127(SE.007).
Temperature .5/1/2/argmax did not rescue incident. Two isolated PPO updates
changed probabilities but held-out CS unchanged (near1/0/2/1, original all0).
No production weights changed. Full evidence/limits LEARN-PAIR-09.
E45 CPU1785 failed on diagnostic state-copy typo before audit; corrected retry
restored exact E44 case, branch-control error0. E40 job1780 continues unchanged.

**E44 COMPLETE — CPU1784 exit0,5m18s:** Original plus4 fresh policy paths miss caster; directed branch kills exact target at1.8s, same HP outcomes. Diagnostic initial GAE positive for directed branch, negative for natural paths; not historical training attribution. Normal-speed branch videos saved; LEARN-PAIR-08. E40 job1780 unchanged.

**E43 COMPLETE — CPU1783 exit0 in9m08s:** Both original E41 mirror traces
and restored branch controls match exactly. At30.9s fullHP caster opportunity,
normal continuation0CS vs targeted attack1CS at1.8s, same champion HP changes.
Source side-frame audit confirms canonical coordinates, no raw world XY or side bit.
Details LEARN-PAIR-08. E40 job1780 continues unchanged; no reward/physics edits.

**E42 COMPLETE — CPU1782 exit0 in4m49s:** Current E40u4400 highHP vs old E34 lowHP,75s normal-speed
video with state-driven spell/attack cues. CPU caster opportunity and restored
state intervention audit;8canaries/180s watch passed. All branch controls exact.
Normal-speed animated video in `/mnt/nfs/shared/E42_caster_trade_audit/low_team_1/combat_blue/replay.mp4`.
Trade intervention improved net HP exchange but lost CS and8s reward; missed
melee attack recoverable. No one-auto caster windows in this old-opponent game.
E40 training unchanged. No HP reward added.

**E41 VIDEOS COMPLETE — CPU1781:** Exit0 in4m17s;4 canaries and180s
startup watch passed. Frozen E40u4050,75s mirror, both HP roles. Normal-speed
75.1s MP4s and interactive HTML in `/mnt/nfs/shared/E41_wave_video/low_team_{0,1}/`.
Diagnostic CS blue/red4/2 and3/3; zero deaths. Not aggregate learning scores.
Parameter/experience/reward/credit-assignment audit recorded LEARN-PAIR-07.
No training modifications; E40 job1780 continues on desktop. No E41 watcher remains.

**E40 EXTENDED TRAINING STARTING — Slurm1780:** Dani requested immediate restart and longer
unattended training. E39b final weights, fresh optimizer and learning-rate
schedule,10000additional updates (~12h), same scenario/reward/model. Original
E34 remains fixed evaluation opponent. Checkpoint every50updates; periodic
frozen evaluations. Dry-run,5 GPU canaries and180s launcher health watch passed.
E40 has passed4000updates; periodic frozen evaluations saved. At u4000
vs fixed E34: fullHP CS2.75/deaths.0625; lowHP CS2.156/deaths.125.
Survival improved on lowHP, farming is not improving monotonically.
Capped CPU checkpoint check: E39b final step/schema/finite parameters verified;
original E34 opponent loads correctly and differs from learner weights.
Log `/mnt/nfs/shared/E40_tower_wave_extended-1780.out`.
No optimizer speedup installed.

**E39b COMPLETE — Slurm1779:** Exit0 after1h23m15s; finished2026-09-30
03:36:37UTC at1000updates/32.768M champion decisions. Study and manifest agree;
final frozen mirror and fixed-original-E34 evaluations saved. No own jobs or
watchers remain. Continuation restored u13 optimizer/schedule but restarted live
episodes because the legacy checkpoint lacked rollout state.

Frozen75s held-out scenario,64games/mode, one training seed: against unchanged
E34, BLUE fullHP CS3.56→4.47, deaths0.5625→0.0625, gold advantage+158→+272;
BLUE lowHP CS2.19→1.84, deaths0.875→0.84375, gold difference−76→−147.
Mixed result: advantaged-role improvement, disadvantaged-role farming not solved.
Mirror fullHP meanCS3.42→3.80; lowHP2.14→1.72. Not a10-minute transfer result.
Artifacts `/mnt/nfs/checkpoints/lanerl-jax/E39b_tower_wave_advantage/vec-s0-20260930-021812-01c96dd1/`;
log `/mnt/nfs/shared/E39b_tower_wave_advantage-1779.out`.

PERF-008 remains CODE REVIEW ONLY. PPO batching/collision optimization tests
were not launched; no new performance result or background optimization work.
Next: discuss asymmetric learning result and bounded PPO batching trial.

**E39b INTERRUPTED — DESKTOP BOOT TO WINDOWS:** Slurm1778 cancelled at
2026-09-29 22:57:25 UTC after13updates/425,984 champion decisions. Node is
DRAINED with reason `boot to windows`. Signal handler saved params and optimizer
atupdate13; manifest/study report interrupted. Checkpoint decodes, all arrays
finite, latest byte-identical to `ckpt_000425984.msgpack` (SHA256
4f9368cef611b8fb62942ffd738deac90135447bda356a6fe228062e05bdc353).
At that interruption no own jobs/watchers remained; continuation1779 subsequently completed.

Experiment startup validated:5 GPU tests,180s watch, actual scenario/checkpoint,
compiled endpoint gate and both initial64-game frozen evaluations passed.
Training ~4.1s/update, finite losses.75s level3 tower/wave rounds,70/100HP
roles; source trained random-start E34, fresh optimizer.1000update target
unfinished; requires desktop Linux/Slurm availability to continue.
Initial mirror role means: fullHP3.42CS vs lowHP2.14CS; deaths0.594 vs0.891.
Low side still~72%HP at first enemy sight. These are PRETRAINING scenario
baselines, not learning improvements; no post-training frozen evaluation yet.
Artifacts `/mnt/nfs/checkpoints/lanerl-jax/E39b_tower_wave_advantage/`;
log `/mnt/nfs/shared/E39b_tower_wave_advantage-1778.out`. E38b superseded.

**E39 FAILED BEFORE TRAINING (Slurm1777):** Four GPU canaries/startup,
actual scenario/checkpoint and compiled endpoint gates passed. Initial frozen
evaluation then failed on alive-spell reporting shape (time/team/feature index).
Fixed helper; both scenario regressions passed. E39b same-protocol retry
launched as Slurm1778 and passed startup (see above);
no PPO updates or scores from E39. Calibration next-wave61–63s remains valid.

**E37b COMPLETE:** Slurm1775 exit0 in9m16s; all10 branches. Restored full
state/carry/RNG exactly; recorded40s control error0, sampled reconstruction
mismatches0. Baseline dies5.70s,+0CS; one E survives40s,+11CS; E then1s
reverse survives,+12CS. W does not reduce damage: existing C# stale-local
bug deliberately mirrored in JAX, verified source read-only. No vendor edits.
Evidence: LEARN-PAIR-05 and `/mnt/nfs/shared/E37b_escape_counterfactual/`.

**E38b FAILED BEFORE TRAINING:** Slurm1776 exit1 in4m07s. Startup canary
and180s watch passed, then initialization failed: helper assumed a named
`button` layer; checkpoint uses Flax `Dense_7`. No training/evaluation result.
Corrected layer lookup from PolicyConfig, replaced synthetic-schema test with
actual GRU forward-pass regression (passed), and checked the actual E33 BC
checkpoint shift. No retry launched during the strategic discussion; no own
jobs remain. Next authorized experiment remains the bounded E exploration
comparison; broader data/architecture direction is proposed, not implemented.

**E36 LEARNING AUDIT COMPLETE:** CPU Slurm job1773 completed exit0 in30m04s;
GPU job1772 cancelled before execution because desktop was unavailable.
Fixed two replay-only defects: missing checkpoint noop-click handling and red
button/cursor placeholders. Eight tests/canary passed;180s startup gate passed;
all three checkpoints unchanged and hashes verified. No training changes.

Corrected seed7 mirror diagnostic CS: initial BC46/37, final BC46/38,
random21/32. These single games do not reproduce/prove the aggregate decline.
Discarded movement clicks while alive: initial BC2.45%, final BC7.64%, random53.77%.
BC inherits a pure last-hitter that tanks waves without a combat/escape rule.
At the blocked-wave example, W/E are ready but total spell probability is1.4e-7;
631HP lost in5s with zero reward; death reward0 and respawn penalty22.5s later.
Initial BC exhibits the same basic failure. No PPO/physics bug established.

Evidence and limits: LEARN-PAIR-04. Artifacts:
`/mnt/nfs/shared/E36b_learning_audit_cpu/`; corrected videos:
`/mnt/nfs/shared/E36_corrected_videos/{bc_initial,bc_final,random_final}/replay.mp4`.
Normal-speed12s incident: `.../blocked_wave/replay.mp4`. E35 videos are superseded
for diagnosis; overnight vectorized frozen evaluations remain unaffected.
Next discussion: targeted escape counterfactual and a bounded exploration test
that preserves farming, rather than another unchanged long training run.
No own jobs/watchers remain; no persistent services created. Clearly labeled
E36b staged inputs remain under `/mnt/nfs/shared/E36b_learning_audit_cpu-staged-1773/`.

**OVERNIGHT E33/E34 COMPLETE:** Slurm1768 COMPLETED exit0 after7h02m07s.
Both arms finished2500 updates /81.92M champion decisions each. Only initialization
differed: BC vs random; N128/T128, standard4x4 PPO, lr1e-5 annealed, entropy0,
detached critic, full visibility/map, relative reward and mirror training.
No restart, nonfinite-loss flags, or overnight hyperparameter changes.

**Frozen final, 128 games/mode from16 held-out start states, one training seed:**
- BC vs fixed heuristic: CS45.50→37.76, gold difference
  -66.41→-136.05. Final mirror36.11/34.53 CS.
  Fine-tuning did NOT improve the prior; gradual loss of farming performance remains.
- Random vs fixed heuristic: CS5.78→24.65, gold difference
  -639.79→+784.46. Final mirror23.95/24.21 CS.
  Positive learning; still below30CS in JAX mirror. No C# transfer test this run.
These are fixed final checkpoints, not selected best intermediates. The arms
were not played head-to-head. Common conservative LR means this does not test
the best possible scratch recipe.

All twelve frozen evaluations/arm (u0,500,1000,1500,2000,2500 ×mirror/heuristic)
and2500 metric rows/arm verified. Final checkpoint and optimizer arrays decode
and are finite; final step81920000, manifests finished. Median updates4.52s BC,
4.60s random. CPU/GPU canaries, startup and early monitoring passed.
Artifacts: `/mnt/nfs/checkpoints/lanerl-jax/E33_E34_overnight/` (study.json,
per-arm manifests, metrics, evaluations, initial/intermediate/final checkpoints).
No own jobs/watchers remain; no external watchdog/service was created. Scratch
staging is job-labeled under `/scratch/E33_E34_overnight-1768/`, auto-retained30days.
Next: discuss the BC fine-tuning objective/learning issue and scratch-policy
behavior before another experiment. Full frozen trajectory table: LEARN-PAIR-02.

**PERF-006/007 COMPLETE — FUSED RAYS INSTALLED:** Job 1766: full fused rays
2.028s collect / 4.656s update; bush lookup 1.978s / 4.620s. Only 0.8% extra
whole-update throughput, with differing outputs: reject simplification per
Dani's decision rule. Full visibility preserved; fused CUDA implementation
selected (PERF005 showed 1.481x whole-update improvement over original).
CPU retains reference (11 CPU vision tests passed). PERF007 job 1767
completed: exact GPU ray/nested-vmap comparison and all 11 production-dispatch
vision tests passed; launcher reported complete/canary passed. No active jobs
or watchers from this work. Changes committed/pushed; default keeps full rules.

**PERF-004 MEASUREMENTS / SCOPE UPDATE (09-29, Dani):** All six fixed-N128/T128
unprofiled collection, PPO and complete-update timing cohorts finished. Complete
updates: 6.315–6.822s (4,803–5,189 champion decisions/s); collection 3.720–4.233s,
learner 2.603–2.611s. Job 1760 FAILED at the final profiler session (SIGSEGV),
after saving all ordinary timings; no active profiling job remains. Job 1759
saved usable random early/mid collection traces; both jobs passed numerical /
lowered-graph canaries and 180s startup watches. Do not claim missing N128
learner/late GPU traces succeeded. Sources and limitations: PERF-004 ledger.

Collection ray loops across tick vision, observations, decoding and order
bookkeeping occupy ~55% of the profiled collection span; collision ~19%,
waypoint loops ~8%. These are profiler spans, not exact unprofiled wall slices.
Learner temporary memory 5.43GB, with 4.25GiB in large transformer tensors.

Dani has now authorised implementation work and prioritises structural changes
with potential multiples of end-to-end improvement, not standalone single-digit
percentage fixes. Stop expanding the broad profile. No performance edits made
yet; next work should be a bounded, parity-checked structural experiment with
collection AND full-update A/B timing. Environment-count sweeps remain deferred.
Existing wall_reward_probe was preserved unchanged at the committed baseline.

**GRU GPU PROFILE COMPLETE (09-29, Dani):** `PERF003_gru_profile` / job **1757**
finished successfully. Dry-run, split-vs-fused numerical canary and 180-s startup
watch passed. Actual GRU, T128, 4 epochs x 4 minibatches, collector-prepared
near-wave bank with 20-s jitter, same fixed inputs for 3 synchronized repeats.
16 envs: collect 2.816 s + learn 1.074 s = 3.890 s/update, 1,053 champion-dec/s.
128 envs: collect 3.966 s + learn 2.551 s = 6.517 s/update, 5,028 champion-dec/s
(4.78x throughput for 8x envs; split 61% collection / 39% learning).
These are SEPARATE compiled stages; sum excludes host I/O and is not a fused
production benchmark. Learner includes GAE, all 16 PPO gradient steps and
post-update diagnostics. Compilation alone: 199 / 333 s at 16 / 128 envs.
128-env learner temporary buffers 5.429 GB; rollout 0.073 GB; params 0.025 GB;
Adam 0.049 GB. Peak live JAX allocations 6.125 GB; retained pool 10.740 GB;
allocator limit 12.440 GB (single GPU snapshot 11,160 MiB device-used).
256-env fit/speed and larger-batch learning quality remain unmeasured.
Dani requested PROFILE ONLY, then discussion: no optimisations or training
settings changed; only timing hooks added. No benchmark checkpoint saved,
all optimizer outputs discarded, no active work left from this session.
Results/provenance: PERF-003 ledger row, `runs/PERF003_gru_profile/*/result.json`.
Existing Claude frozen check 1754 completed (64 envs, lr 0, median 3093 dec/s,
4.43 GB, 42.27 aggregate CS over 26 full champion episodes; not a per-side
or multi-seed gate). Existing 512-env MLP benchmark 1756 failed rc 1; cause
not diagnosed here. Neither job was modified.

**CPU TIMING CHECK (09-28, Dani; desktop remains unavailable):** Read today’s
Claude handoff; no training/evaluation launches or production changes. PERF-002
in `docs/JAX_FIDELITY_LEDGER.md` records the capped CPU probe: one populated
150-s lane, 44 live entities, 6 ticks/decision. Separate warmed component medians:
observe 1.05 ms, GRU+sample 3.13, decode 2.64, apply/routing 1.37, sim 10.63
(~57% of their sum). Disabling collision gives 10.34 ms; disabling call-for-help
10.12 ms. Not GPU or PPO throughput; fixed idle-champion fixture, quota noise.
Next candidate to discuss: minion waypoint N² sort/66-step cluster scan and its
conditional becoming both-branches execution under vmap. Also correct prior
throughput claims: `vec_bench.py` measured the OLD MLP trainer, not `vec_train`
GRU; the large measured gain mostly came from increasing env count. Frozen vec
validation and actual GRU GPU rollout/learner timing remain pending. Probe
`lanerl_jax/probes/jax_time_breakdown.py`; no persistent jobs created.

**DESKTOP DOWN (booted to Windows 19:53 UTC): E31/E32 CANCELLED at updates 82 / 112;
both have `ckpt_latest.msgpack` and resume with `--resume`.** Where they were: E31 (no
prior) episode CS 8.5 (updates 30-60) -> 18.0 (60-100), entropy 10.6 -> 8.6, ev 0.71 --
LEARNING from scratch at ~14 s/update; E32 (heuristic init, lr 1e-5) 43.9 -> 46.2,
entropy 0.80, ev -0.38 (critic not fitting yet) -- holding, not yet improving.

**VECTORISED JAX TRAINER BUILT (Dani, 09-28): `lanerl_jax/train/vec_train.py`.** The
whole rollout is one `jax.lax.scan` over vmapped envs (no host loop): GRU carry in the
scan (reset on done), NEAR-WAVE reset from a BANK of collector-prepared states (same
seeds/legs/jitter as `JaxFarmCollector`, `bank/setup.jsonl`), relative reward identical
to `server_train.relative_reward` (test), dropped/masked unwalkable clicks, `--init-from`,
scripted opponent, RunDir checkpoints that `jax_eval`/`ops/jax_periodic_eval.sh` load
unchanged (evaluator glob now also matches `vec-*`). CPU smoke + checkpoint reload OK.
Throughput of the scan architecture (old `trainer.make_train`, GPU shared with E31/E32):
16 envs 254 dec/s (0.94 GB), 64 envs 983 dec/s (2.35 GB) -- LINEAR in envs (the step is
latency-bound), vs ~230 dec/s for the 16-env collector path. 128+ env runs died
(shared GPU; exit 1, likely OOM beside 11 GB of E31/E32) before the desktop went down.
TESTS GREEN (`lanerl_jax/train/tests/test_vec_train.py`, 4): reward equality with the
collector path; actor/learner log-prob agreement EXACT for mlp, gru and gru+scripted red
(first version scrambled GRU sequences by folding the agent axis after a time/env swap --
the agreement test caught it); loop runs, full episodes end, metrics finite.
Sweep with the GPU free (gpup partition; the cpu partition does NOT preempt llm-serve's
11.7 GB and OOMs): 128 envs 2237 dec/s (4.26 GB), 256 envs 4360 dec/s (8.09 GB); 512 pending.
Frozen-checkpoint check pending (`runs/VECCHECK`: E32 init at lr 0, 64 envs; episode CS
must land at the collector's 44 / 45).
NEXT when desktop is back: (1) resume E31/E32 (`--resume`), or replace them with vec runs;
(2) vec sweep 64/128/256/512 envs for dec/s + peak GB with the GPU free;
(3) E31/E32-style runs on the vec path (256+ envs), then C# cross-play finals.

**JAX TRAINING (Dani, 09-28 17:40 UTC):** parity established, so training moves to JAX.
E31 = no prior (fixed GRU, noop clicks, reference PPO, relative reward), E32 = heuristic
init (DAgger-3 GRU clone) with lr 1e-5, detached critic, no KL; both 16 envs x 11000
updates (23M steps, ~1.5 days at ~230 dec/s each), evaluated every 500 updates on
JAX (`ops/jax_periodic_eval.sh`, `runs/EVAL/jax_summary.jsonl`); finals cross-played on
the C# server. Night-shift cron REMOVED at Dani's request (he will schedule separately).
**Goal now:** a randomly initialised PPO policy that scores >30 CS in a
10-minute mirror trial on the C# server, evaluated frozen over seeds.
**E19 FINAL (09-28 05:30 UTC): 34.0 / 40.4 CS over 12 episodes per side (22-41 / 31-48),
no prior, seed 1, XP 0 -- ABOVE THE GATE** (E15 seed 0: 28.5 / 31.8).

**Dani's two questions (09-28):** (1) JAX vs C# parity for the scripted bot within
1-3 CS, solo and mirror; (2) can PPO improve a prior, after replacing the custom
update with a transcription of PureJaxRL `ppo_rnn.py` (handed to a second agent;
handoff in the chat log). Parity (PARITY-002): both engines deterministic (4 server seeds identical); solo JAX 70
vs server 65, mirror 48 / 43 vs 45 / 43. Wave POPULATIONS match within 4% by type
with the same cadence and lifetimes; the residual is wave-clash TIMING (fronts
2-3k u apart at moments), and the ~5-CS solo offset flips sign between measurement
paths. Side-by-side videos in `runs/PARITY/side_by_side/` (solo and mirror). ROOT CAUSE FOUND:
the JAX red-lane path's barracks-exit vertex was 65-119 u off the server's walked
line (every JAX red minion lost ~1 s / ~100 u joining it); `TOP_LANE_PATH`'s red end
is now the server's measured line. Residual: the server's red minions crawl ~0.4 s
through the barracks turns (unmodelled). Single deterministic games are CHAOTIC in CS
under sub-second offsets, so parity is now measured as a MEAN over 16 games with
seeded start jitter (`--start-jitter-s 20`, both engines): RESULT (16 games/engine): solo JAX 66.3 vs server 65.2; mirror 48.1 / 47.0 vs
49.9 / 49.7 -- differences +1.1 / -1.8 / -2.7 with standard errors ~1.7: WITHIN NOISE
and within the 1-3 target on means. Question 1 answered (with the lane-path fix; the
server's 0.4-s barracks crawl remains unmodelled).
## BC/DAgger → reference PPO improvement test (E24–E27)

<!-- E24_PIPELINE -->
E24 paired frozen comparison gate: FAIL. One training seed; preliminary evidence only. Differences vs baselines (mean, 95% paired bootstrap CI): {"clone": {"cs": {"ci95": [-21.8125, -13.0625], "mean": -17.5}, "gold_diff": {"ci95": [-391.5625, -135.3125], "mean": -268.125}, "reward_return": {"ci95": [-25.523846676171523, -7.773956573242685], "mean": -16.8813491863948}}, "teacher": {"cs": {"ci95": [-35.5640625, -22.8125], "mean": -29.4375}, "gold_diff": {"ci95": [-484.0703125, -264.375], "mean": -377.1875}, "reward_return": {"ci95": [-32.17358934475733, -14.330526756613928], "mean": -23.229222236925125}}}. All six frozen evaluations completed. One-shot continuation finished; no recurring jobs remain.
<!-- /E24_PIPELINE -->

Dani requested a minimal demonstration of learning beyond the heuristic
using ONLY delta-relative gold, delta-relative XP and lane keep. Prepared
E24: existing server BC+DAgger-2 checkpoint (150 epochs on `dagger2.npz`),
fresh reference optimizer, fixed heuristic opponent, alternating sides,
8 servers/8 cores, 400 updates, lr 1e-5 annealed, detached critic, no KL prior.
Reward explicitly `(own Δgold - enemy Δgold)/20 + .008*(own ΔXP - enemy ΔXP)
+ 5*Δlane_potential`; no direct CS/death reward.

Prerequisite REW-13 fixed: frozen opponents had lost enemy stats (making
relative reward own-only) and red evaluation rows were labeled blue.
E25 evaluates the unchanged clone, E26 the heuristic, E27 the fixed u400
PPO checkpoint, all against the heuristic on identical held-out seeds 2/3,
16 games each, 8 per side. The paired scorer in `ops/heuristic_improvement.py`
requires reward/gold improvement over BOTH baselines and no lower mean CS.
No intermediate-checkpoint selection; one training seed is preliminary.

RUNNING: E24 job **1717** on desktop/gpuhog, 2026-09-28 07:15 UTC.
Dani confirmed Linux was back; dry-run passed, live canary passed in 28 s,
and the 400-update run has finite learner metrics. **180-second startup watch passed.**
Execution caveat (OPS-005): the capped login launcher propagated JAX_PLATFORMS=cpu,
so this fixed run learns on desktop CPU while reserving the GPU. It remains valid
and continues to u400. Subsequent canaries/evaluations explicitly select CUDA
and clear login XLA_FLAGS; no seed/checkpoint is being selected based on results.
Run: `lanerl_jax/runs/E24_dagger_reference_relative/seed0/server-farm-s0-20260928-071520-b95b2fc9`.
E25/E26/E27 comparisons remain pending; no improvement claimed. 18 distinct
capped tests passed before launch. Full protocol is in `experiments/README.md`.
One-shot continuation `e24-comparison.service` is **ACTIVE** (PID 764627), waiting for E24,
then launch E25/E26/E27 seeds 2/3 sequentially, with dry-runs, canaries and
startup watches, and score them. Registered in `/mnt/nfs/shared/jobs/REGISTRY.tsv`;
no paid services. Log `/mnt/nfs/shared/jobs/E24-comparison.out`; stop with
`systemctl --user stop e24-comparison.service` (E24 job 1717 is independent).
Continuation/backend regression checks passed (5 tests). The worker commits
start/stop progress in these existing docs and removes its
registry entry when done. Other agents' jobs remain untouched.

## Reference PPO port

Live learner ported to PureJaxRL `ppo_rnn.py` revision
`31756b197773a52db763fdbe6d635e4b46522a73` (PPO-17): reference GAE, clipped
surrogate/value loss, per-minibatch advantage normalisation, trajectory
shuffling and epoch/minibatch scans, single clipped Adam eps 1e-5 and
update-boundary linear annealing. Removed dual clip, KL early stop,
applied-only summaries, usage-weighted entropy and the target-head path.
LanePolicy, click masking, optional prior KL/detached critic, GRU batch
layout, post_kl, trunk gradient norms and explained variance are preserved.
`standard()` now uses gamma .99 and 4 x 4 epochs/minibatches; legacy CLI
`--target-kl` is accepted but ignored, unequal critic lr is rejected.
Network code changes are comments only. Sim/parity/collector behavior and
experiment JSONs are untouched. Paused trainer imports/update call migrated
to the same shared loop. Vendored oracle and license are test support.

Validation: reference loss/gradients/GAE agree within 1e-6; trajectory
integrity, optimizer steps/schedule, GRU reset/replay and masked click checks
pass. 47 distinct focused tests passed (33 + 23 with 9 repeated) under
`ops/login_capped.sh 12G 3 .venv-jax/bin/python -m pytest`; all 146 train
tests collect. Server help and the unchanged E14 dry-run command work. No training or evaluation jobs launched. Old optimizer
states are incompatible; the next arm uses E12a `--init-from`, E14 settings
plus `--detach-critic`, in a NEW experiment ID owned/launched by the other
agent, against E21 frozen 45.8 / 43.0. Explicitly set `--reward farm` to
match E14/E21: E14 JSON omits reward and today's default is relative. The
new gamma and removal of KL stopping are intentional comparison differences.

## Previous session notes

**E15 FINAL (20:50 UTC): 28.5 / 31.8 CS over 12 episodes per side (23-37 / 18-44),
combined 30.1, deaths ~0.1 -- a no-prior GRU AT the gate after ARCH-001 (GRU
wiring) + INT-001 (no wall attraction). Not yet clearly above it: late checkpoints
swing 9-31. E18 at full rate scrambled it (u260 10.3 / 5.8, post_kl 0.12); E18b continues at lr 5e-5.**

**LEARNER TEST VERDICT (09:20 UTC): NEGATIVE.** Constrained PPO fine-tuning
from the DAgger-2 clone (45 / 50 CS frozen) destroys it in every form tried:
E10 plain (KL 0.38 in one update), E11 (lr 5e-5, KL stop 0.02): 0 CS by u140,
E12b (same, after E12a critic warm-up, value loss 21 -> 1.2): frozen u140
10.8 / 12.0, u280 2.5 / 2.3, u440 0.3 / 1.3, training CS 0 by u520.
E12a with actor lr 0 kept the clone intact (training CS peaks 30.6 at
mid-episode every 47 updates), so the collapse is caused by the actor update
itself, not by the init or the collector. A working PPO does not move a
policy from +45 return to 0 with a correlated reward. The learner, or the
reward as the learner sees it, is the open bug. Candidates, in order:
advantage normalisation on near-zero-variance batches, entropy 0.01/3 per
head pulling apart a sharp policy (entropy 0.3 -> 1.0 in E11), deaths
(-2, ~4 per update across 20 agents in mirror) dominating the per-update
signal, GAE/done handling in the [N,T] batch. E12b cancelled (its job).

**LEARNER, Dani's challenge (09-28 02:30 UTC):** a near-monotone decline from the
prior is a systematic gradient, not noise. E21 = E14 with the critic DETACHED from
the trunk + per-term trunk gradient norms: value term into the trunk = 0, policy
term 10-27, entropy term 0.06, and entropy still rises 0.31 -> 0.59 in 7 updates
(E14's rate). So the critic is not the driver; the policy-gradient term is
(softmax-saturation asymmetry: reinforcing a rare action raises entropy, suppressing
one changes nothing). **E21 FINAL 45.8 / 43.0 over 8 ep/side: the prior HOLDS with the critic detached**
(E14/E16 slid to ~38). The u280 4-episode row was noise. The critic's gradient
through the shared trunk was the systematic drift; a separate critic is the fix. `probes/prior_drift_probe.py`: the collapse
rewrote the trunk (feature cosine 0.36 vs prior) while staying sharp = step size;
the slow slide keeps the representation (cosine 0.9+) and 93-96% of click mass
near the prior's click but the peak cell wanders (matches 30-48%) = precision
loss, no directed shift. E22 (KL-to-prior 2.0) FINAL 47.2 / 43.5: holds the prior like E21. Both fixes preserve a
45-CS prior over 400 updates; neither improves it (the improve-on-prior question moves
to the reference PPO port).

**REWARD (Dani, 04:40 UTC):** the server runs never used the delta-gold/delta-XP/lane-keep
reward; they used Codex's farm reward (+1 CS, -2 death, shaping, xp). `--reward relative`
now implements (own - enemy) gold/20 + 0.008 (own - enemy) xp + shaping, no death
term; E23 = E15 recipe with it, queued.

**PER-STEP RECORD (23:00 UTC):** every run now logs value, reward terms, click mass and
position per step (`<run>/diag/`, `ops/diag_report.py`). On E12b/E06/E15 it shows: the
critic never values the wall; the degenerate policy's only reward is XP, best at the
wall (2.3 vs 0.8/min); 60-65% of clicks at the wall are unwalkable for every
checkpoint. E17 (JAX learner test) 37.1 / 41.5 from a 44 / 45 clone = the server's
behaviour. E19 = E15 recipe with XP 0, seed 1, launching.

**WALL HUGGING = CLICK INTERFACE (INT-001, 13:10 UTC), Dani's top priority:**
not bushes (server vision ignores grass; sim has none), not the potential
(0 inside a corridor that contains the walls), not XP. Clicks onto unwalkable
ground resolve to the closest reachable point = the wall; 43-49% of E06's
clicks were unwalkable (51-58% near walls), so a diffuse policy is pulled to
walls and held. Random policy, same seed: 35% of frames within 150 u of a
wall with resolve, 6% with the new `--unwalkable-click noop`. **E15** =
no-prior GRU (fixed core) + noop clicks, RUNNING since 11:08 UTC (job 1603,
`runs/E15_noprior_gru_noopclick`, evaluator armed every 100 updates): ~8 s per
update, 51% of movement clicks dropped at update 37, post_kl ~0.04. Masking is now implemented (`--click-mask`: the same heads, the click
renormalised over walkable cells, PPO consistent under the mask; 3 tests); E20 = E15
recipe with the mask, launching.

**WHY THE PRIOR DEGRADES (PPO-16, 11:40 UTC):** E12b applied ONE minibatch
step per update: the KL stop measures pre-step KL, the first step is always
applied at KL 0, the second was over 0.02 every time (`kl_stopped` 0.94), and
the logged KL averages applied steps, so it read 0.000 while the policy moved.
One fresh-Adam step at lr 5e-5 from the clone = KL 0.33 (pg), 0.46 (entropy
term alone, grad norm 0.05: Adam is scale-invariant), 0.11 (value through the
shared trunk); at lr 1e-5 = 0.015; at 2.5e-4 = 8.7. 600 blind steps of KL
~0.3 with every corrective step withheld is a random walk off the prior.
`post_kl` metric added to `server_train`. Replays of E12b u140 and u600
rendering (`runs/EVAL/replay_E12b_*`).

**ARCH-001 (10:35 UTC): the plain GRU core is near-blind** (heads see only the GRU
output; BC cannot fit 16 sequences; MLP fits the same data to 98%). Fixed as
opt-in `--core-norm --core-residual`; E06's ~17 plateau is partly this. The
JAX-leg GRU uses the fixed wiring.

**JAX LEG (running, per Dani's 22:40 plan):** heuristics-initialised GRU on
the JAX sim to a compare point, then E04 vs that agent on the C# server.
**DAgger-1 GRU clone on JAX: 41.0 / 44.4 CS sampled mirror (26-59 / 34-58)** -- the
JAX heuristics-init GRU compare point; **cross-play on the C# server: E04 29.5 CS (22-33) vs the JAX GRU clone;
the JAX clone in a server mirror 31.4 / 29.2 (JAX 41 / 44: ~70% transfers)**;
DAgger-2 clone: **44.2 / 45.4** (round 3 done; clone
`runs/BC/bc-grunr-20260927-121409-bee9ebca`). Next on JAX when cores free: PPO from
that clone with the PPO-16 settings (lr 1e-5, KL stop, entropy 0) = the JAX learner test. E13 (no-prior JAX GRU, snap clicks) at u~80: 16 CS vs idle.
Step 1 done: scripted last-hitter on the JAX sim vs idle red = 65 CS on all
16 envs (deterministic sim: identical trajectories), 4792 decisions each in
140 s wall (~550 dec/s at 16 envs, 5.5 GB GPU). Mirror scripted demos
recording (`runs/JAX_ORACLE/demos_mirror`). Step 2: GRU clone
(`train/bc_diag.py --core gru`, `runs/BC/bc-gru-*`), then DAgger rounds on
JAX rollouts. **E13** (no-prior fixed-wiring GRU on JAX, `runs/E13_jax_gru_scratch`) PAUSED at
~u90 (14:20 UTC) to give E14 its cores; resume with `sbatch slurm/jax_train.sbatch ... --resume <ckpt_latest>`
(see `experiments/E13_jax_gru_scratch.json`). E14 (PPO-16 fix arm, lr 1e-5 from E12a) COMPLETE at u400 in 24 min: post_kl
0.006-0.023 per update, training CS windows 31 / 24 / 27 where E12b had 3 / 1 / 0:
the prior survives: **final frozen 37.5 / 36.4** (prior 45.5 / 49.5, E12b 0).
E16 (entropy 0) complete: final frozen 38.5 / 38.8 = E14's 37.5 / 36.4; the bonus is
not the drift. Learner status: lr 1e-5 + KL stop HOLDS a 45-CS prior at ~38 over 400
updates (E12b: 0) but does not improve it; the no-prior E15 does improve (24.5 at u1220). E15 frozen: u100 12.8 / 11.0,
u280 20.8 / 12.0, u700 14.2 / 14.2, u800 14.8 / 17.8, u940 13.3 / 17.8, u1220 24.5 / 21.8, u1740 25.3 / 24.0, **u1880 31.0 / 31.3 (28-37 / 28-34) -- ABOVE THE
30-CS GATE, no prior, frozen, both sides** (4 episodes each); u2460 over 8 episodes 28.5 / 29.6 (19-37 / 22-37): AT the gate,
noisy; u2660 over 8: 19.0 / 18.5. Late checkpoints swing 9-31: the gate is
reached but not held; the 12-episode final at u3000 is next, then E18 continues
from E15's params with a fresh anneal (`experiments/E18_e15_continue.json`). E06's best was 22.5 / 20.8. Periodic evaluators now use one tag per run+update (two shared a log).
Step 3: PPO on JAX from the GRU clone (`jax_train.py
--init-from --core gru --preset standard`). Step 4: `ops/launch.py eval
--ckpt <E04> --opponent frozen --opponent-ckpt <jax gru>` on the C# server.
Porting issues: the shared-loop JAX trainer is server-speed (430-550
dec/s); the fast Anakin trainer lacks GRU/init-from/frozen opponent; parity
gaps COLL-005, SPELL-013, STAT-003 mean the JAX agent's clicks may transfer
imperfectly; an observation-contract match (viewport-structured-v3) is what
makes the cross-play possible at all.

**History of the diagnostics (09-26/27):** interface oracle 60 CS vs idle,
74 as red, 45 / 28 scripted mirror; MLP clone 94-97% per-step accuracy but
5-10 CS closed-loop (compounding error); DAgger-1 24 / 33; DAgger-2 45 / 50;
cross-play vs the DAgger-2 clone on the server: E04 27.5 (21-33), E06 17.3.

envs beside llm-serve, so the JAX leg needs the Anakin port; deferred. E07 PAUSED at u178 (relaunched 21:55 UTC with `--init-from`: E06 u3140 params, fresh
optimizer/schedule, own 3000-update budget) vs a FROZEN E06 checkpoint. The first
E07 attempt inherited E06's counter and schedule (767 updates at ~0 lr, evals
failed) and is recorded as invalid.
E06 STOPPED 19:25 UTC at update 3800: plateau ~17 CS (band 13-22 over 11 frozen
evaluations since 4M decisions); it is the control. Earlier E06 notes: Frozen u320: 5.3 / 6.0 CS; u620: 8.5 / 13.5 (learning); u920: 2.5 / 1.5 --
COLLAPSE caused by OPS-004 (rank loop starving level-9+ champions of
decisions), not by the policy. Fixed 09:50 UTC; E06 resumed from its u620
checkpoint. Post-fix frozen u1260: 15.3 / 14.3; u1580: 14.3 / 24.5; u1900: 11.3 / 14.8
u2200: 20.3 / 17.5 (best); u2520: 16.8 / 13.8; u2840: 12.5 / 14.3; u3140: 15.5 / 14.3; u3460: 22.5 / 20.8 (best; both sides > 20); u3780: 18.3 / 16.0. Plateau band 14-20 frozen since
u1260 (E01's band). Leak fixed (eager scan re-traced per update): memory now
+2.4 MB/update, 7.4 s/update. E07 launched 18:30 UTC (Dani's go-ahead): E06's learner continued
against E06's u2200 checkpoint as a FROZEN opponent, learner side alternating
per server (`--opponent frozen`). Runs beside E06. Parallel-server audit of
E06: all 20 agent slots average 13.7-17.7 CS, no outlier server, zero rank/
restart/fatal/complaint events, all servers at 309 ticks/s. OOM-killed at
u1931 (slurm 10 GB limit, 12:55 UTC); resumed 14:30 UTC at 20 GB from u1920
with per-update RSS now logged (`rss_gb`). All numbers past level 9 before the fix are suspect.
Throughput 12 s/update = 213 dec/s with the node to itself; the GRU's
BPTT update (128-step scan x 16 minibatch-epochs) is ~2/3 of that and is
the next thing to optimise if E06 shows learning. The desktop was on Windows 04:11-06:37 UTC, which cancelled E04 and E06. Dani's decision (04:55 UTC): continue ONLY the GRU arm
with published defaults (E06). E04 stays stopped (frozen 18.8/24.5 at u2020,
train chunks 27-31 at u3070; its checkpoints remain). 

**Done today:** Codex's uncommitted server-first work committed (`35210dc`);
collector 3-5x faster; mirror mode, resume, frozen eval; lr 1e-5 identified as
the dead learner; REW-11 (shaping paid for sitting in base) and PPO-15
(entropy favoured coordinate buttons) found and fixed; rank loop no longer
aborts; red setup route legged in both engines; legacy code moved to
`legacy/`; CODEMAP, EXPERIMENTS, patch and probe indexes written.

**Screen-click verified on the live server (probe 1, `lanerl_jax/probes/screen_click_probe.py`):**
a click cell is 30 x 36 world units at screen centre (smaller than a minion);
attack-move clicks on a minion acquired it; ground clicks land within
quantisation (7-22 u) and the champion arrives within 1 u. Probe 2
(`screen_click_probe2.py`): right-click and attack-move on a minion set the
server target for BLUE and RED, raw and through the grid + lane frame, at
65-685 u. Probe 1's right-click misses were at 600-1000 u where cell
quantisation exceeds the minion's collision radius; attack-move auto-acquire
covers that range. Verdict: the click interface is implemented correctly.

**First frozen evaluation (update 760, 5 mirror episodes, 20:28 UTC):** blue
mean 19.4 CS (11-27), red mean 16.0 (11-29), 0-2 deaths each. Below the 30
gate. Update 1080: blue 14.2 (5-22), red 21.0. Train CS has been FLAT at
16-18 per episode since about update 500: the run has plateaued at half the
gate. Diagnosis: the button mix swings between strategies (E-spin 81% at u460,
attack-move 38% at u820, Q 63% at u1126) with KL ~1e-2 per update: the
policy wanders rather than converges. Dani's replay review found the real defects: 46-54% of movement clicks land on
unwalkable ground (the server then walks straight into the wall, hence the
edge-hugging), and the +0.005*xp term paid 1.11x the CS term, teaching both
champions to camp the brush beside the wave. Fixed on the SERVER (`screen-click-v3`, build `ClickV3`, PATH-011): unwalkable
click targets resolve to the closest reachable point, as the real client does.
XP is enemy-only on the server (checked, SIDE-001) and stays, at weight 0.002
(was 0.005, which out-paid CS).
E03 (lr test) stopped; E04 branches E01's state at update 1520 on ClickV3 with
XP 0.002 at the same lr. FIRST SIGNAL (2026-09-26 00:40 UTC): E04's first 12
training episodes average 28.1 CS (30-42 in eight of them) against E01's
17-21 at the same point: the wall-click fix is the biggest single gain so far.
Frozen eval at E04 update 1820 pending. E01 (control, DeadProbe) still running.

E05 launched 01:20 UTC (see Running).

**Next:**
1. Periodic frozen evaluation every ~300 updates (`ops/periodic_eval.sh`, results in `runs/EVAL/summary.jsonl`).
3. Multi-process collector (`--workers N`, `MultiProcessCollector`) is
   implemented and smoke-tested: 12 envs mirror, workers=1 vs 3 gave the same
   ~770 decisions/s on 6 cores while E01 held 8 cores -- the desktop is
   SERVER-CPU-bound at ~14 servers, not Python-bound any more. More throughput
   needs more cores (or fewer ticks per decision), not more workers.
4. If E01 plateaus below 30: run seeds 1-2 (`SEED=1 experiments/E01_mirror_wave.sh`).

**Open bugs / unknowns:** attack completion is not attributed (Codex saw zero
logged basic-attack completions in a 20-CS episode); `trainer.py` still has
the PPO-15 bias; ENT-02 vs AA-007 (retarget mid-windup) unmeasured.

**Approved and done 2026-09-25:** branches `bronze`, `eval-results-jan12`,
`report/2026-09-02`, `lane-rl/spike`, `t3code/4b6bc7fc` and the agent
worktree branch deleted (tags `archive/*` pushed to origin); the eight
frozen `ahriuwu-*-20260925` worktrees and `_deprecated/ahriuwu-lanerl`
removed; server builds `HudProbe`, `ScreenClick` deleted and `Release`
replaced by a symlink to `DeadProbe`. Remaining branches: `main`,
`lane-rl/jax` (active), `t3code/9283ee84` (another T3 thread's worktree).

Rules: whoever starts or stops a run edits this file in the same commit.
History lives in `docs/EXPERIMENTS.md` and `docs/JAX_FIDELITY_LEDGER.md`.


| ID | Finding / evidence | Status |
|---|---|---|
| OPS-047 | User-authorized idle wake test: Slurm1788 array on danilogin (success, compiler exit1, application exit7, Slurm timeout). Existing Jarvis T3 client targets only thread6638ec1e-0a67-4ea7-bea9-5ecfd4b3d47c; one deterministic command ID, preserves model/modes, no credential copies. Bounded systemd watcher waits for terminal accounting and an idle conversation. `/mnt/nfs/shared/OPS047_slurm_events/{launch,result}.json`, task logs `/mnt/nfs/shared/OPS047-1788_*.out`. | PASS: canary/180s watch; all four expected terminal states and logs verified. Bridge observed ready/no active turn at22:11:44UTC, accepted at22:11:46UTC, and automated message started a fresh turn after prior final. Watcher inactive, registry removed. Timeout parent reports0:0 despite TIMEOUT (batch0:15): classify State plus ExitCode. One combined notification tested; not four independent wakeups, cancellations/OOM/node failure, or restart recovery. |

| OPS-048 | Reusable `ops/slurm_event_bridge.py` derives current T3 thread from Codex session; validates owned single job and exact name, bounded/registered/capped watcher, State+ExitCode accounting, idle-only deterministic command ID, retry on transient errors, SIGTERM/finally cleanup. Terminal classification checks passed (TIMEOUT0, cancellation, RUNNING, absent); existing T3 read canary passed. E46 job1787 armed as `lanerl-event-1787.service`, expiry 2026-10-01 01:14 UTC, live accounting verified RUNNING. | Uses OPS047-proven delivery path; E46 terminal delivery verified22:27:17UTC after COMPLETED0:0; watcher inactive/registry removed. AGENTS rule10 applies going forward; no training changes. |

| LEARN-AFK-01 | E46 Slurm1787 finished exit0 in42m51s,612updates/10,027,008 learner decisions from E40u5795. Frozen120s AFK evals64games/checkpoint (32 per initial HP role), one training seed. Combined CS at u0/64/128/306/612:5.453125/8.53125/8.640625/8.546875/9.5625; deaths:.9375/.015625/0/0/0. LowHP final9.1875CS vs4.9375 initial; fullHP9.9375 vs5.96875. Final mean endHP99.02%/98.28%; final0deaths both roles. | Positive AFK farming/survival learning, most early gains by1.05M decisions; not active-opponent/10-minute transfer or isolated causal effect of HP versus tower shaping. Frozen summaries do not separately report tower damage; training tower reward is not frozen evidence. |
| LEARN-AFK-01 validation | `/mnt/nfs/checkpoints/lanerl-jax/E46_afk_farm/vec-s0-20260930-214903-32499e76/`: all5 frozen evals and612 metrics; loss_nonfinite0, median3.220s/update. Study/manifest complete; final `ckpt_010027008.msgpack` step10027008,201 numeric arrays finite,latest byte-identical; SHA256680889327db7db60503dd0f3b7a0926886f32f2f3ae69eb38b79d3c34265570d. | Event delivery accepted and fresh turn observed; watcher inactive/registry removed. No further experiment launched. |

| REPLAY-AFK-01 | E49 Slurm1800 exit0; final E46u612 checkpoint SHA680889327db7db60503dd0f3b7a0926886f32f2f3ae69eb38b79d3c34265570d. Full120s seed7 offset−45, blue fullHP, red parked fountain with NOOP actions. Blue7CS/0deaths, red0/0. Full-map and animated combat MP4s120.1s at1x; `/mnt/nfs/shared/E49_afk_final_video/low_team_1/`. Nine startup canaries/180s watch passed. | Single diagnostic game, not frozen64-game aggregate. Event delivered; watcher inactive and registry removed. |

| LEARN-AFK-02 | E46 curve generated by `ops/figures/afk_training_curve.py`, PNG in docs/figures. Five frozen points only; lines interpolate, one training seed. LR3e-5 annealed per update toward0. Train u1–64 mean approxKL.00759, clip.0981; u307–612 mean gold reward.00547, XP.00758, tower.00924, health−.000893 per learner decision. Source `_relative_reward` credits towerHP loss regardless of damage source and XP independently of last hits. | Quick initial gains then slower progress; not proof LR is limiting. Minion tower damage and proximity XP permit reward without personal tower hits/CS. Proposed matched3e-5 vs1e-4 constant LR from E46 final, fixed reward/budget2M decisions, frozen eval comparison; no run authorized/launched here. PPO reference discussion: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/ . |

| LEARN-AFK-03 | User-authorized E50/E51 cleanup: xp_scale0, personal Garen tower shaping, unchanged HP penalty. Telemetry accumulates champion-source effective turret HP removal in existing hit order, capped at remaining HP; minions/overkill/dead targets/other victims excluded. No damage/kill/observation dynamics changed. Old checkpoint parameters load; old full rollout-state schema not migrated. Five CPU tests passed including actual Garen tower hit and telemetry-independent HP; GPU canaries and180s watch passed on E50 job1804. Frozen eval adds personal tower damage. | ConstantLR3e-5 vs1e-4, E46final init/fresh Adam,128updates/2.097M each;64 frozen games at0/64/128. Directional criterion>=.5CS improvement over control,<=.05 extra deaths/game; bounded/nonfinite stop. No separate reward ablation per user. PPO LR rationale in LEARN-AFK-02. |

| LEARN-AFK-04 | E50 job1804 complete exit0,15m14s;128updates/2,097,152 decisions,median3.212s/update. Frozen64games at u0/64/128: CS9.5625/9.453125/8.03125; deaths0/.03125/.171875; personal towerHP0/7.4876/20.1505; shaped return6.3293/6.0193/3.0785 under SAME cleaned reward. Final lowHP8.6875CS/.03125deaths; fullHP7.375CS/.3125deaths. Checkpoint/latest identical,all numeric arrays finite,128 metric rows,zero loss_nonfinite. `/mnt/nfs/checkpoints/lanerl-jax/E50_afk_personal_lr3e5/vec-s0-20260930-230245-a12b1fbe/`. | LowerLR arm regressed; reward cleanup not automatically a learning improvement. E51 submitted1805 from SAME E46final, not E50, to finish authorized comparison. E50 bridge cleanup verified. |

| LEARN-AFK-05 | E51 job1805 complete exit0,14m21s;128updates/2.097M. Frozen64games,120s each at u0/64/128: CS9.5625/8.84375/8.859375; deaths0/0/0; personal towerHP0/0/0; new-reward return6.3293/5.0781/5.9312. E50 final8.03125CS/.171875deaths/20.1505HP. Same baseline frozen values exactly. E51 mean train approxKL.02693,clip.2584,postKL.03534 versus E50 .01092/.14083/.01618. Both checkpoints finite,final step2097152,latest identical;128 metrics each,zero loss_nonfinite. E51 `/mnt/nfs/checkpoints/lanerl-jax/E51_afk_personal_lr1e4/vec-s0-20260930-231817-279e25e5/`. | HigherLR improves over lowerLR by.828CS and avoids observed deaths, meeting directional arm criterion, but BOTH below starting9.5625CS. No successful personal tower learning in higherLR arm. One training seed; no causal diagnosis from this alone. Event cleanup verified; no new experiment. Plot `docs/figures/E50_E51_comparison.png`. |

| LEARN-AFK-06 | E51 source `_sample` directly samples logits (T1), entropy_coef.001 constant. First/last16 train rows: entropy6.1669/6.6051; explained variance−.0035/−.0624; value loss.0498/.0247; E selection.1123/.019. Thus total entropy did not collapse, but this is distribution-dependent and not state-coverage evidence. Critic is detached linear readout, cannot learn trunk features; poor bootstrapped-target explained variance motivates shared-gradient test, not proof of cause. | User authorized targeted experiments at higherLR: E52 entropy.01, E53 shared critic with entropy.001; sameE46init,128updates each,E51 control. New metrics head entropy,value bias,and GAE on positive gold (descriptive,not counterfactual). Reference: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/ (entropy regularization/shared features); .01 engineering choice. |

| LEARN-AFK-06 validation | Two CPU tests passed: exact reference loss/gradient agreement with added diagnostics; detached/shared critic initial outputs identical, gradient routing as intended. E52 Slurm1813 passed11 GPU canaries and180s startup watch. | Completion bridge armed; E53 already authorized next. Frozen outcomes pending. |

| LEARN-AFK-07 | E52 entropy.01,LR1e-4 complete1813 exit0,14m24s. Frozen64games120s u0/64/128: CS9.5625/8.796875/2.9375; deaths0/0/0; personal towerHP0/.8526/0; cleaned reward6.3293/6.1861/1.8550. First/last16 train entropy6.6727/9.1130,button.7882/1.3671,x3.2024/4.1568,y2.6820/3.5892; EV−.3256/.2260,value bias.0779/.00805. Positive raw GAE on gold-event fraction .451/.628 (window-average; bootstrapped/descriptive,not proof of correct credit). Final checkpoint step2097152 and all arrays finite,latest identical,128 rows/nonfinite flags0. `/mnt/nfs/checkpoints/lanerl-jax/E52_afk_entropy/vec-s0-20261001-000248-6ddcc544/`. | High entropy worsened farming versus E51 final8.8594; no extension. Shared-critic E53 submitted1814 from same originalE46,not E52. E52 bridge cleanup verified. |

| LEARN-AFK-08 | E53 sharedcritic,LR1e-4,entropy.001 complete1814 exit0,14m29s. Frozen64games120s u0/64/128: CS9.5625/7.9375/7.96875,deaths0 throughout,personal towerHP0; cleaned reward6.3293/5.6487/5.6079. First/last16 train EV.1818/.4925,value loss.1108/.01284,value bias.0436/.00778,entropy6.6763/7.3984,approxKL.1023/.01322. Positive raw GAE fraction on gold .506/.661. Final checkpoint step2097152/all numeric arrays finite/latest identical,128 metrics/nonfinite0. `/mnt/nfs/checkpoints/lanerl-jax/E53_afk_shared_critic/vec-s0-20261001-001821-fd41aed1/`. | Better fit to changing bootstrapped targets did not improve CS. Neither E52 nor E53 qualifies for authorized bounded extension; none launched. One seed per configuration, no root cause proven. All-arm plot in docs/figures/E50_E51_comparison.png; E53 bridge cleanup verified. Next proposed gold-event advantage/policy-update audit. |

| LEARN-AFK-09 | E54 completed32 isolated production-shaped updates:16 E46freshAdam,16 E51restored runner/Adam. Independent GAE maxerror7.75e-7; replay likelihood2.29e-5,value4.53e-6. Actual CS-event frames1696/1894; positive normalized advantage81.90/89.28%, mean advantage.255/.371, mean chosen-action delta logprob−.052/+.027, probability rises38.03/52.96%. Positive-credit hit subset rises42.19/55.00%. Preceding1s credit positive76.19/86.85%, delta logprob−.038/+.012. Early fixed hit anchor final mean logprob delta−1.406/−.512. Mean approximateKL.06766/.01547, clip.37875/.20306. NonCSpositivegold counts7050/13844, max7.63e-7, total.00535/.01053 versus actualCS gold1264.5/1423: old positivegold-count diagnostic contaminated, reward magnitude negligible. | Evidence `/mnt/nfs/shared/E54_afk_credit_audit/{result.json,from_E46/,after_E51/}`; probe `lanerl_jax/probes/afk_credit_audit.py`. Supports investigating update interference/drift, not absent reward or simple likelihood/GAE mismatch. Negative sampled logprob drift also occurs normally under policy change; fixed-history likelihood is not frozen CS, hit-frame action may not cause kill, preceding window limited1s. Does not isolate LR versus Adam reset/reward change or establish optimal update count. Next controlled same-batch headwise/update-budget audit, then frozen evaluation. No exported weights. |

| LEARN-AFK-10 | E55b1821 complete4m04s,6canaries/watch pass. E55 failed5.5e-5 parameter reproduction; corrected all branches to production learn path, standard maxparametererror0 both stages. One selected same batch perstage (E46fresh u2,220hit frames; E51restored u129,194). Button mean delta logprob standard−.04605/−.00370;LR3e-5−.000126/+.01041;1epoch+.00898/+.03231;noentropy−.06825/+.00165. Exact button KL standard.03756/.00655,lowLR.00197/.00218,1epoch.00975/.00264. E51 standard clickY delta+.0343 while button−.0037: heads move differently. | `/mnt/nfs/shared/E55b_afk_update_branches/result.json`; no checkpoint exported/frozen gameplay evaluation. Supports bounded behavioral test of fewer epochs, not proven farming gain. Early E46 batch only42.7% positive normalized CS-event credit, E51 88.7%; E46 likelihood reduction cannot be attributed to overriding positive feedback. Buttons recorded at kill may not cause kill, many Q/E; coordinates only29/103 used-screen hitframes, no equivalent-click mass measured. Mean delta-logprob is not percent arithmetic probability or CS change. One batch perstage, no replicated significance. E54 broader aggregate and this selectedbatch differ. Watcher inactive/registry clean. |

| LEARN-AFK-11 planned | E56/E57 one-vs-four epoch behavioral test after E55b. Fixed collected-decision budget; record wall-time efficiency too. PPO multiple-pass rationale: https://arxiv.org/abs/1707.06347 ; oneepoch benefit is engineering hypothesis, not published task optimum. | Same E46final, fresh optimizers, LR1e-4;64game frozen cohorts0/64/128. +.5CS versus control/<=.05 extra deaths directional gate, original9.5625 baseline separately. One seed. |

| LEARN-AFK-11 E56 interim | Oneepoch1822 complete11m57s; frozen64games CS9.5625/5.53125/3.640625 at0/64/128,0deaths/0personal towerHP. Reward6.3293/1.7242/1.9084. Median update_s1.8504 after first5,128 rows/nonfinite0; checkpoint step2097152 finite/latest identical,manifest epochs1. | E55b likelihood improvement did not predict better gameplay. Do not extend E56. E57 matched fourpass control1823 submitted under existing authorization; final comparison pending. E56 watcher cleaned. |

| LEARN-AFK-11 final | E57 control1823 COMPLETED0:0,14m22s. Frozen64game120s CS9.5625/8.84375/8.859375 at0/64/128, equals E51 aggregate endpoints; oneepoch E56 9.5625/5.53125/3.640625. Deaths/towerHP0 both. Median update afterfirst5 E56 1.850424s/E57 3.250184s (1.756x raw throughput); full Slurm11m57s/14m22s includes startup/evals. E57 paramsfinite,step2097152,latestidentical,128metrics/nonfinite0,sourceSHA matchesE56,epochs4 confirmed. | Reject oneepoch at tested LR/budget; neither improves original policy. Improved sampled-button likelihood in E55b did not predict gameplay. Both actor AND critic get fewer updates; last16 trainEV−.729/−.062 and postKL.03698/.02141 on differing trajectories are suggestive, not causal evidence. One trainingseed; control repeat is deterministic reproduction, not independent seed replication. No new job/extension. E57 bridge inactive/registry cleaned. Evidence `/mnt/nfs/checkpoints/lanerl-jax/E57_afk_four_epochs/vec-s0-20261001-061021-5056795c/` and E56 ledger path. |

| LEARN-AFK-12 planned | E58 systematic frozen behavior audit; observation-only scripted control, actual decoded orders, instantaneous onehit/inrange/readiness windows, reward decomposition, full-episode MonteCarlo value targets. Source self input16 omits ownAA timers; GRU receives observations, no sampled previousaction. | Alias not yet proof of limiting learning; diagnostic32games/policy, no optimal-oracle claim. Overnight9h authorized to15:37UTC; goal1Mtokens. |

| LEARN-AFK-12 results | E58b1825 completed5m40s;32frozen120s games perpolicy on same bank/seeds, diagnostic set differs from training frozen64. E46/E57/scripted CS9.90625/8.90625/13.5;deaths0. Gold reward7.328/6.750/10.750;HP penalty−.798/−.866/−3.060;personal tower0/.1485/0;total6.530/6.032/7.690. Discount.99 startreturn.3211/.3357/.3456. E46/E57 initialV.7721/.3976; fullMC EV−.7918/.2741. OwnAA timer alias maxobsdelta0. Ready inrange onehit frames52/30/1144; target-killable decodedattack conditional0/0/.3776. | `/mnt/nfs/shared/E58b_afk_behavior_audit/result.json`,perpolicyNPZ. Scripted achievable farming control supports interface capability; higherCS takes moredamage, and discounted advantage small. E57 critic improved despite lowerCS, so critic error alone not sufficient. Synthetic own-timer alias proves missing currentinput, not causal learning limitation. Opportunities omit approach/ongoing attack and do not guarantee time-to-hit. Scripted32 episodes deterministic duplicate bankstates, not32independent samples. E58 initial1824 failed NumPy tracerhelper; E58b corrected. Bridge stopped after livegoal review before idle delivery. |

| LEARN-AFK-13 planned | E59 matched same-state caster option intervention,2s directed then policy,<=16cases/four paired seeds. True discounted rewards2/8/fullhorizon,critic GAE,health reward rescore100/25/0. | Multi-action option, not single-action causal attribution. Only visible onehit caster candidates; some may die before arrival. No policy update. |

| LEARN-AFK-13 results | E591826 complete4m53s,24captured/16selected cases,four paired continuations. Directed up-to2s option all forced clicks resolved to target. Natural/directed CS by2s .4375/.625;by8s1.0469/1.1406;remainder5.21875/5.34375. Discounted returns .2069/.2675 by2s,.4374/.4875 by8s,.4841/.5362 remainder.7/16case return differences positive; small mixed effect, not significance claim. HP100/25/0 fullreturn differences+.0521/+.0505/+.0500. InitialVmean.6158; firstGAE natural−.0673/directed−.0263. | Evidence /mnt/nfs/shared/E59_afk_caster_forks/{result.json,cases.msgpack,forks.npz}. No major reversed-credit/HP-penalty mechanism demonstrated in selected cases. Multi-action offpolicy option, not actual training advantage. Watcher stopped afterlive review, no new training inthis diagnostic. |
| LEARN-AFK-14 planned | E60 append ownAA cooldown/2,windup,attacking,Eactive to selfvector; versionv4,20selfdims. Insert4zero context rows between16self and6global; old params untouched, initial logits/carry preserved withinfloat tolerance,new rows receivegradients.2 CPUtests pass; realcollector likelihood GPUtest inlaunchcanary. Same E46init/LR/rewards4epochs,256updates, matched E61longv3control. | Own-state observation gap inE58; recurrent-action-history rationale example https://arxiv.org/html/1611.05763v3 section2.2/3 (paper usespreviousaction/reward; not proof ofthis feature set). No enemyhiddenstate; C#driver refuses v4 becausewire lacks AA telemetry. Experimental hypothesis; frozen resultpending. |

| LEARN-AFK-14 validation | E601827 passed14GPUcanaries including own-input visibility, zero-weight migration/gradient, and real GRU collector/learner agreement;180s health watch passed. E58 coarse100ms windup-end/zeroCD-end/increase counts pergame E46 36.16/10.78/4.75,E57 33.59/12.78/25.03,script14/.5/.25. Enemyminion deaths17.59/16.84/14 versusCS9.91/8.91/13.5. | Windup indicators are proxies, not exact cancellation causality; targetdeath may legitimately cancel. Candidate still unproven pending frozen learning scores. Bridge1827 armed. |

| LEARN-AFK-14 final | E60 completed1827/0:0,256updates4.194M. Frozen64games final CS6.171875/deaths.78125/personal towerHP360.99844/reward9.85192 versusinitial9.5625/0/0/6.32926. Finalcheckpoint finite/latest identical; all256metrics finite. | Failed farming gate; learned tower damage earns higher objective despite deaths/fewerCS. At128 deathgames31/64 meanreward7.652 versus5.472 nondeathgames; observational association, not proof each death is optimal. No explicit death penalty; lastfewHP cost can be smaller than rewarded towerhit. E61 matched longer control next; reward correction candidate prepared next. Watcher stopped after active review/cleaned. |

| LEARN-AFK-15 prepared | E62 opt-in death_loss_gold300; own death counter delta charges once, not dead frames/respawn; default0 preserves controls. Retain E60 inputs/HP100/personal tower900/XP0 and E46fresh start.512updates, frozen128/256/512. | E60 reward rose56percent while CS fell35percent and deaths rose.781/game.300gold is simulator champion base gold (sim/rewards.py), engineering scale not literature optimum. Test safe farming/tower metrics; reward totals across changed objectives are not comparable. Existing PPO reference arxiv.org/abs/1707.06347 does not prescribe task rewards. Control E61 keepsdeath0. |

| LEARN-AFK-15 validation | CPU death-event regression passed; GPU launch includes it via relative_reward selector. Wave worker now accepts train_seed (default0) for independent action/reset RNG replication; scenario bank and frozen evaluation seeds stay fixed for matched comparisons. E61 unchanged defaultseed0/death0. | No new seed run submitted yet; source checkpoint parameter initialization stays fixed by design. |

| LEARN-AFK-16 literature / untested | OpenAI Five Figure18 uses attention over unit embeddings conditioned on sampled action for target selection; AlphaStar official blog describes autoregressive pointer head. Our actor instead samples independent96x54 screen coordinates. Sources https://cdn.openai.com/dota-2.pdf p47 and https://deepmind.google/blog/alphastar-mastering-the-real-time-strategy-game-starcraft-ii . | Possible action-representation burden; not evidence that adding a pointer fixes this task. Preserve screen-click/server resolution contract; any future proposal must output ordinary observed screen coordinates, never privileged target IDs. Deferred while E61/E62 investigate duration/reward. Autoattack code already preserves same-target swings but cancels changed-target windups (parity requirement); coarse cancellations alone do not prove policy error. |

| LEARN-AFK-15 rescore | E60u25614/64 death-free games mean10.1429CS/149.9116towerHP/reward7.7859;50deathgames5.06CS/420.1028towerHP/reward10.4304. Rescoring identical trajectories with300gold death cost gives wholecohort reward−1.86683 versusinitial6.32926 (no deaths). | Conditional subsets are selected outcomes, not causal evidence or a new frozen policy result. Shows corrected accounting would rank current dive-heavy behavior below original; actual retraining result pending. |

| LEARN-AFK-14 control128 | E61u128 frozen evaluation JSON exactly matches E57u128 (all episode rows/metrics),8.859375CS/0deaths/0towerHP. Initial evaluation also9.5625CS. | Default0 death option/defaultseed0 retain control behavior; longer training remains in progress. Plot tool extended with --group debug; shows completed frozen checkpoints only, no reward comparisons across different objectives. |

| LEARN-AFK-14 control256 | E61u256 frozen64games CS8.984375/deaths.0625/towerHP11.330745/reward5.462119. Matched E60u2566.171875/.78125/360.99844/9.85192. | Own-input variant learns substantially more tower damage with worse farming/survival; neither passes original farming gate. E61 continues to512; E62 death-cost candidate dry-run passed. |

| LEARN-AFK-12 metric limitation | actions.orders_from emits ATTACK when cursor hits hostile, otherwise ATTACK_MOVE for attack_move button. step.py target acquisition chooses nearest visible enemy for ground attack-move. E58 targeted_killable counts only decoded ATTACK, excluding subsequent automatic acquisition. | Zero direct-target clicks must NOT be read as zero attack attempts/successful acquisitions. Nearest-enemy fallback could favor melee minions without precise caster selection; causal frequency unmeasured. Policy button rate alone is not targeting competence. No mechanics changed. |

| LEARN-AFK-14 longcontrol final | E61 Slurm1828 COMPLETED/0:0,36m10s,512updates8.389M. Frozen64games final6.0625CS/1death/462.11734personal towerHP/reward12.75195. Rescored death300 gives−2.24805 versusinitial6.32926. Finalstep8388608/finite/latest identical;512metrics/nonfinite0; medianupdate3.26169s; checkpointSHA88d31ea00262ca6aa53064489c0bf661a7a36b3f24a0561b266a48547083a70b. | Unchanged v3 eventually reaches same risky tower outcome as E60, strengthening reward-mismatch evidence. Own inputs may accelerate this learned behavior; no general farming improvement. Original lowCS issues need not all share this cause. E62 now tests explicit death cost. Bridge stopped after active review; inactive/registry removed. |

| LEARN-AFK-15 launch validation | E62 Slurm1829 passed15GPUcanaries in365.93s, including death-event accounting and own-input GRU likelihood; mandatory180s health watch passed. Bridge active/result updating, expires10:16:21UTC. | Bounded512update test now running; no frozen improvement claim yet. |

| LEARN-AFK-15 matched control planned | E63 uses originalv3inputs with same300gold death cost asE62, E46start/freshAdam/seed0/LR1e-4/4epochs/512updates. | Isolates reward-only change againstE61 and input change againstE62; avoids crediting new inputs for a reward-only improvement. Same predeclared farming/survival gate; independentseed validation required. Prepared, not submitted. |

| LEARN-AFK-17 literature / deferred | Current PPO.standard uses gamma.99 at10Hz (10s discount horizon), lambda.95. OpenAI Five paper section4.5/figure6 finds longer horizons improve a skilled agent; AppendixC uses lambda.95 with horizons60–840s, including180s baseline. Source https://cdn.openai.com/dota-2.pdf pp13–14,30. | Domain precedent for testing a longer discount horizon if corrected rewards still fail; not evidence our120s AFK task needs full-game settings. Keep lambda.95 initially to isolate gamma rather than calling lambda.99 a published default. Not implemented/submitted; prioritize E62/E63 reward comparison first. |

| LEARN-AFK-15 E62u128 | Frozen64games E628.8125CS/.03125deaths/7.61418towerHP/reward5.94133. Matched E60u1287.640625/.484375/179.35231; original9.5625/0/0. | Death cost suppresses risky tower outcome and recovers some CS versus own-input no-cost arm; does not yet improve original farming. Continues to256/512; E63 reward-only control queued. |

| LEARN-AFK-15 E62u256 | Frozen64games9.25CS/.0625deaths/12.18325towerHP/reward5.34366. | Recovers toward original9.5625CS, not a demonstrated farming gain; complete512budget. |

| LEARN-AFK-17 candidate | E64 prepared gamma.999 (100s horizon), retainslambda.95/v3/death300 and other E63 settings. Waveworker exposes optional discount, default.99 unchanged. | Domain precedent above; gamma extends value targets while direct GAE trace remains about2s, so improved long-range credit still depends on critic learning. Conditional follow-up after reward comparisons, not submitted. |

| LEARN-AFK-15 E62 final | Slurm1829 COMPLETED/0:0,37m27s,512updates8.389M. Frozen64games final8.828125CS/0deaths/1.642749towerHP/reward6.36881; initial9.5625/0/0/6.32926. Final golddiff135.782 versus146.2508 initial. CheckpointSHA3f6c68a529f533783f3fad1def96380e0cde639347374b4714bacc0d7e5da009,step8388608 finite/latest identical;512metrics/nonfinite0; exact live deathcost accounting; medianupdate3.2861s. | Death cost removes dive outcome but does not improve farming. Tiny total reward rise with less gold does not establish meaningful learning gain; other health/position/tower terms offset CS loss. No extension. E63 v3 correctedreward control next. Bridge stopped after active review; inactive/registry clean. |

| LEARN-AFK-18 prepared | E65 removes HP-loss and tower shaping together, retains gold/CS, death300 and lane potential; same E63 v3/source/optimizer/PPO. | Positive-control learning objective: test farming learnability without secondary HP/tower tradeoffs. Does not isolate the two removed terms or solve the intended multitask objective alone. Frozen CS/survival gate and fixed512update budget; final reward must be restored and validated if this diagnostic succeeds. Prepared, not submitted. |

| LEARN-AFK-15 E63 launch validation | Slurm1830 passed15canaries in365.77s and180s health watch. Bridge active/result updating, expires10:54:14UTC. | Original-input/death300 control now running; no new improvement claim. E64/E65 remain prepared only. |

| LEARN-AFK-19 direct-click mass | Read-only cappedCPU analysis of16saved E59 states/carries with exactE57sourceSHA; value reproduction error7.152557373046875e-07. Mean4.9375 of5184gridcells hit selected caster. Cursor-button probability.67172; conditional coordinate mass.0006958; full direct-click probability.00053054. Hypothetical10percent uniform observed enemy-minion center proposals gives.0350348 (~66x). Artifact /mnt/nfs/shared/LEARN-AFK-19/click_mass.json; probe afk_click_mass.py. | Selected missed-caster cases, not global behavior. Ground attack-move can auto-acquire; this is direct-click mass only. Proposal calculation ranks noHP/privilegedtarget and is not a trained policy or demonstrated CS gain. Exact geometry includes visible blockers/minimap. Supports testing learned observation-based click proposals while retaining physical screen-click protocol. |

| LEARN-AFK-19 implementation / CPU validation | Opt-in click_proposals adds two namedheads (query/gate), initial10percent learned candidate mass; oldweights/core/value retained. Candidates are projected visible hostile centres; no HP ranking/privilegedentityID. Exact joint distribution sums duplicatecells; shared sampling/logprob/entropy/postKL/priorKL and greedy driver respect it.3newCPUtests pass (mixture/sample/entropy/mask/emptygradient/migration/newheadgradient);21existing PPO/mask/reference regressions pass. Implemented projection hits target in16/16saved cases, same.03503 hypothetical mass. | Newphysical distribution intentionally differs fromold; E66 must improve its own measuredinitial aswellas original. FullsizeGPU/realGRU update canary pending. Defaults stayoff; no running experiment altered. E64/E65 remain prepared; targeted E66 prioritized next over generic tuning based on measured click sparsity. |

| LEARN-AFK-15 E63 interim | Frozen1288.140625CS/.203125deaths/57.47196towerHP/reward.15175;2568.953125CS/0deaths/3.19188towerHP/reward5.53388. | Recovers survival but remains beloworiginal9.5625CS. Complete512budget; E66 targeting candidate queued next. |

| LEARN-AFK-19 evaluation support | Waveworker optional eval_seed defaults2007, unchanged existing comparisons; enables a fresh frozen sampling cohort in any independentseed replication. Debug figure includes E63/E66 as available and explicitly notes E66 changes initial action distribution despite preserving checkpoint weights. | No new seed run launched; fresh evaluation must compare its own initial policy rather than assuming old9.5625 baseline applies. |

| LEARN-AFK-19 scope limits | Click-mass measurement uses E57 and its saved recurrent history; E66 starts from E46 and needs its own initial frozen evaluation. E59 cases excluded spin/windup but did not require AA cooldown0; low direct-click mass is not proof every selected caster should be attacked. E59 causal forks gave mixed/small return gains. | Targeting architecture remains a testable hypothesis, not an established root cause or66x gameplay/speed improvement. No simulation behavior or visibility changed. |

| LEARN-AFK-19 readiness / causal limits |5/16saved cases AAready; their direct-click probability.0004039. Reusing E59 forks: ready cases directed2s option mean deltaCS0 at2s, +.15 overremainder, discountedreturn+.0483;11cooling cases+.2727CS at2s,+.1136remainder,return+.0538. | Small selected subsets, paired4continuations each; no strong immediate causal gain in ready subset. Low target probability is not proof a mischosen action. E66 is therefore an architectural learning test motivated by representation/literature, not a claimed66x performance fix. |

| LEARN-AFK-15 E63 final | Slurm1830 COMPLETED/0:0,38m03s,512updates8.389M. Frozen64game120s final8.921875CS/.015625deaths/1.442893personal towerHP/reward6.079413 versusinitial9.5625/0/0/6.329263. CheckpointSHA999f8606f9f1d440bf970426da0ab08489418b0f7cd8da9992eab5e1a3f87c20,step8388608 finite/latest identical;512metrics/nonfinite0; death-cost accounting maxerror0; medianupdate3.3476s. | Original-input correctedreward control also fails farming gate. Survival improved relative to no-death-cost E61, but neither input variant solves CS. No extension. Watcher stopped after active review; inactive/registry clean. E66 starts next; E64/E65 unsubmitted. |

| LEARN-AFK-19 GPU startup | E66 Slurm1831 passed15standard GPU canaries (365.94s) and4proposal tests (120.54s), including actualGRU collector/learner likelihood agreement and finite PPO update. Launcher180s healthy-start watch passed. Bridge lanerl-event-1831.service active/result updating, expiry2026-10-01T11:34:21.708261+00:00. | Fullsize training compilation/evaluation pending; tests establish implementation consistency, not learning improvement. |

| LEARN-AFK-19 own initial baseline | Fullsize E66 compilation and endpoint-before-reset canary passed. Frozen64game120s u0:8.890625CS/.015625deaths/4.044495personal towerHP/reward4.98031; low/fullHP CS8.625/9.15625. Run /mnt/nfs/checkpoints/lanerl-jax/E66_afk_click_proposals/vec-s0-20261001-093416-d26024a8. | New10percent mixture is not an immediate gain versusoriginal9.5625CS. Predeclared success threshold remains10.5625CS/<=.05deaths; learning gain and independent replication still required. |

| LEARN-AFK-19 u128 | E66 frozen64games6.203125CS/0deaths/0towerHP/reward3.93398 versusowninitial8.890625CS. | No targeting improvement at first checkpoint; finish predeclared512budget. E65 diagnostic queued next after correctedreward controls failed farming gate. |

| LEARN-AFK-18 launch | E65 Slurm1832 queued after dry-run; launcher89112 awaits canary/watch. OriginalE46/v3/PPO/death300 asE63; HP/tower rewards0. | Tests whether simplified objective supports better farming; neither isolates individual removed term nor solves intended multitask objective alone. Frozen>=10.5625CS/<=.05deaths gate;512updates8.389M,1hSlurm. |

| LEARN-AFK-19 u256 | E66 frozen64games7.8125CS/0deaths/0towerHP/reward5.3628768 versusowninitial8.890625CS/.015625deaths/4.044495towerHP/reward4.9803097. | Partial recovery fromu1286.203125CS but belowinitial. Higher total reward with lower CS again shows objective tradeoffs; no architecture win. Continue512budget, E65 queued. |

| LEARN-AFK-20 temporal batches | Waveworker explicitly sets stagger_initial=False although VecConfig defaultTrue; all128environments share game clock/reset. E54 saved from_E46/update_0001.npz:0CS/exact0reward, rawadvSD.270656, mean chosen-action deltaLogP−.245253. after_E51/update_0136.npz:0CS/meanabsreward6.2864e-10, advSD.113693, deltaLogP−.028373. Mature reset batches132/141 terminate all128envs together. | Critic-driven credit without immediate rewards can be legitimate; neither these aggregates nor negative selected-action mean prove a wrong update. Synchronized phases plausibly reduce batch diversity. RSL-RL runner supports randomized initial episode lengths: https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/runners/on_policy_runner.py . Implementation precedent, not a demonstrated remedy here. |

| LEARN-AFK-20 prepared | E67 opt-in training staggering; discard all shortened first episodes under fixed policy before optimization, retain real carries/RNG/states, require all deadlines full and learner step unchanged. Every frozen eval explicitly disables staggering. Existing objective/E46/v3/PPO asE63.512updates8.389M plus11warmup rollouts180224untrained envdecisions. Syntax/dry-run passed; dedicated realGRU warmup/likelihood GPUcanary pending. | Prepared, not submitted. Requires sameinitial9.5625CS, final>=10.5625CS/<=.05deaths and independentseed repeat. This tests temporal batch diversity without changing the learned or evaluated horizon. No current run altered; E65 queued first. |

| LEARN-AFK-18 reward reporting | Future waveworker frozen episodes/summaries now include accumulated reward_terms, bounded at each first terminal; assert component sum matches total within FP tolerance. Existing simulation and policy sampling unchanged. | E65 and later runs can distinguish gold, death, health, tower and positioning contributions directly. Does not retroactively add components to E66 or earlier evaluations. Syntax check passed; first live full-evaluation accounting gate pending. |

| LEARN-AFK-19 final | E66 Slurm1831 COMPLETED/0:0,43m32s,512updates8.389M. Frozen64game final9.34375CS/0deaths/75.222872towerHP/reward8.102879 versusowninitial8.890625/.015625/4.044495/4.980310 andoriginalE469.5625CS. Step8388608 finite/latest identical,512metrics/nonfinite0, sourceSHAverified, proposalflagTrue; finalSHA3429b9c7b569e76401975c26ddc951073cc3bd2f8bbc03caae03488092bbc4de. Medianupdate3.62076s (~8percent slowerthanE63); last32train mean proposal masswhenavailable.02677. | Some safe tower damage learned and owninitial CS partly improved, but missespredeclared farming gate anddoesnot beat original. No extension orsolutionclaim. Watcher stoppedafteractive review; inactive/registryclean. E65 starts next. |

| LEARN-AFK-20 launch | E67 Slurm1833 queued after1832 via launcher after renewed dry-run; startup session14791 active, GPUcanary/watch pending. | Standing user authorization: bounded controlled follow-up from measured batch synchronization.512updates8.389M +180224warmup decisions; full E63 objective retained. Bridge must be armed after healthy-start watch. |

| LEARN-AFK-18 startup validated | E65 Slurm1832 passed15GPUcanaries in367.68s and launcher180s startup watch. Bridge lanerl-event-1832.service verified active/result updating, expires2026-10-01T12:15:35.617291+00:00. | Fullsize worker/evaluation compilation next; no new performance result. E67 remains queued Slurm1833 with startupwatch14791 attached. |

| LEARN-AFK-18 initial | E65 fullsize compilation/endpoint canary passed; frozen64game u0 exactly9.5625CS/0deaths/0towerHP. Reward7.312540699 entirely gold/CS; approach/death/XP0, HP/tower disabled. New component accounting invariant passed. | Same initial physical behavior asE63/E46, with simpler objective. Training active, no improvement result yet. |

| LEARN-AFK-18 u128 | E65 frozen64games5.359375CS/.109375deaths/0towerHP/reward2.31723246 versusinitial9.5625CS/0deaths. | No early rescue from eliminating HP/tower rewards. Complete512budget; result does not yet prove secondary rewards irrelevant, but simple reward removal is not sufficient at128updates. E67 phase test queued next with fullobjective. |

| LEARN-AFK-18 final | E65 Slurm1832 COMPLETED/0:0,38m01s. FrozenCS9.5625→5.359375→7.109375→3.546875; finaldeaths.015625/towerHP3.191875/reward1.953154. Final reward gold2.210945/death−.234375/approach−.023416/XP0. Step8388608 finite/latest identical;512metrics/nonfinite0;medianupdate3.3350s; initSHA matchesE46,HP/tower0verified. FinalSHA9455f0e211a4faa4c821d5ded0388fa24f31f3f0acc137c45da373b3505b8611. | Simplifying positive reward to farming did not rescue learning. Does not prove all reward designs irrelevant, but secondaryHP/tower conflict alone is insufficient. No extension. Bridge stopped after active review; inactive/registryclean. E67 phase test starts next. |

| LEARN-AFK-20 startup verified | E67 Slurm1833 passed15standardGPUcanaries364.28s and1realGRU stagger-warmup test80.81s: weights/optimizer/step unchanged, deadlines full, realphase spread, subsequentdone=done_full and collector/learner likelihood agree. Launcher180s watch passed. Bridge lanerl-event-1833.service active/result updating, expires2026-10-01T12:54:52.537235+00:00. | Fullsize warmup/initial frozen evaluation pending; no learning claim. |

| LEARN-AFK-20 u128 | E67 fullsize stagger warmup passed, frozen initial exactly9.5625CS/0deaths/0towerHP/reward6.329263. Frozenu12810.0625CS/.140625deaths/207.267403towerHP/reward9.270423. | First positive CS direction in this control series but below>=10.5625CS and above<=.05death thresholds. No solution claim; finish512budget, independentseed validation required if final passes. |

| LEARN-AFK-20 final | E67 Slurm1833 COMPLETED/0:0,41m44s. FrozenCS9.5625→10.0625→6.703125→9.921875; finaldeaths.140625/towerHP802.399197/reward24.160935. Reward terms gold7.710979/death−2.109375/health−4.736133/tower23.295461/position~0. Step8388608 finite/latestidentical,512metrics/nonfinite0,medianupdate3.63235s; SHA7f12479d61db507d23806822a6c5b1d4b129bd58fd9e83b2bed1c06621d0fe55. | Strong tower improvement in one seed but fails farming/survival gate; no independentseed validation or solutionclaim. Watcher stoppedafteractive review, inactive/registryclean. Goal tokenbudget exhausted; no newruns. Proposed next factorial comparison: farming-focused E65 rewards plus E67 phase staggering; not implemented/submitted. |

| LEARN-AFK-21 current video | E68 Slurm1834 COMPLETED/0:0,4m40s. LatestE67final SHA7f12479d61db507d23806822a6c5b1d4b129bd58fd9e83b2bed1c06621d0fe55 verified in trace; seed7/offset−45/fullHP blue versusAFK. Frozen single120s game:12CS/0deaths. Map/combat videos120.1s,10fps,1x at /mnt/nfs/shared/E68_afk_current_video/low_team_1/. | User-requested behavior inspection, not aggregate performance or new training. Canaries/180s watch passed; watcher stopped after active review, inactive/registryclean. |

| LEARN-AFK-22 user video timing review | Read-only E68 trace (E67final),100ms snapshots. At80.22s caster15 HP25.7/cooldown.585s;80.62HP1.7/cooldown.185;80.82cooldown−.015;80.92samecaster aa_target15/windup.229;81.02dead with CS unchanged9. At112.63caster19 HP190.7/windup.263;112.93HP38.5/AAcooldown1.135;113.23HP15.5/cooldown.835;113.33dead,CS unchanged11. At14.8 missedmelee4 HP19 while swingtarget2;14.9melee4dead,15.2target2CS gained. | Different observed miss patterns: correcttarget/late swing, prior nonlethal swing followedby cooldown, and another target prioritized.100ms trace cannot resolve exact damage sources/cancellation ordering or establish simulator parity; requires tick-level execution audit before declaring bug or policy fault. No neural activation/causal input intervention performed. |

| LEARN-AFK-22 observation/reward review | E67manifest HP100/fullbar,death300,personal tower900/fullbar: HP penalty was retained. Builder supplies current relativeposition/HP rounded1/60/type/team/subtype at10Hz; nearest-distance ordering, no minion targetID/explicitHP slope/projectile channel. E67 entity encoder uses shared attention then maskedmax/mean pooling before globalGRU; no per-entity recurrent state or slot positional embedding. Caster1.7/290HP rounds to0 but remains valid/alive entity. | Pure row permutation should not alter this pooled architecture mathematically; do not claim slot reshuffling is a demonstrated bug. Per-minion temporal tracking still must be inferred through pooled history. CV-compatible candidate: matched visible position/health-bar history and observed own attack timing; hidden targetIDs unnecessary. No feature/model change or new training launched. |

| LEARN-AFK-23 literature feature audit | Primary sources: [Tencent Solo AAAI2020](https://cdn.aaai.org/ojs/6144/6144-13-9369-1-10-20200513.pdf), Fig2/algorithm: observable unit attributes, local obstacles, game state; maxpool before LSTM, target attention, availability masks. Detailed timing/history inventory not specified there. [Tencent 5v5 supplement](https://proceedings.neurips.cc/paper/2020/file/06d5ae105ea1bea4d800bc96491876e9-Supplemental.pdf), Table4: minion HP/speed/visibility/income/position/team/type; turret HP/locked target/attack speed; hero status/stats/skills/items; spatial skill regions/bullets/obstacles/bushes. Invisible-opponent inputs explicitly value-only. [OpenAI Five](https://cdn.openai.com/dota-2.pdf), Table4: per-unit last16 HP samples, maxHP, damage/speed, attack/animation timing, incoming creep-projectile ETA, estimated melee attackers; previous sampled action. Some timing/attacker quantities are scripted visible-observation estimates. [AlphaStar](https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf), ExtendedDataTable1: unit HP/shields/energy/position/attackCD/orders/buffs plus map/player state. | E67 has current roundedHP only, no per-unit temporal features/projectile channel, own AA phase or previous sampled action. Literature supports testing richer visible temporal inputs, not a causal claim or necessity of privileged targetIDs. Tencent/AlphaStar inventories do not explicitly establish HP-history stacks. Five and Tencent also pool before recurrent core: pooling alone is not demonstrated root cause. Existing E62 own-action features alone did not fix farming. |

| LEARN-AFK-23 incoming-projectile follow-up | Read-only capped NumPy inspection of E68 trace.npz, video-relative seconds. At80.82 caster15 HP1.7, two incoming raw23-damage projectiles have distance/speed estimates .124/.140s; at80.92 estimates .024/.040s while Garen windup still .229s; dead by81.02 without CS. At112.93 caster19 HP38.5 after nonlethal swing, incoming estimates .268/.346s; at113.23 HP15.5 and remaining nearest .046s; dead113.33 while Garen AA cooldown .735s. | Consistent with projectile competition and AA readiness explaining these misses; does not establish exact tick damage attribution, optimal counterfactual, C# parity or neural cause. ETA uses diagnostic raw projectile target IDs and current positions, not actor-visible features or an exact collision-time calculation. No policy inputs changed. Visible HP history is a low-complexity candidate; projectile/animation tracking would require a separate CV-compatible estimator. No new run submitted. |

| LEARN-AFK-24 plan | E69 exact E68 seed7 replay and two caster15/19 tick audits. Return-only diagnostic AST exposes starts/hits, damage rows, killer and CS. Matched baseline must reproduce E68 and every ordinary env_step state leaf. Earlier and delayed physical-click options begin at77s/110s with200ms timing grid;76 local branches,30min CPU cap. | No production physics/policy change. Existing published feature comparison remains LEARN-AFK-23; this experiment first tests execution and local timing opportunity. Selected cases, no aggregate farming or C# parity claim. |

| LEARN-AFK-24 exact replay gate | E69 Slurm1835: all1201 E68 frames reproduce with maxerror0 in every checked state field, actions and decoded orders, including HP/positions/AA clocks/targets/missiles/CS. Final12CS/0deaths; exact specified checkpointSHA.12canaries and180s launcher watch passed. | Tick instrumentation/branches still pending; no causal or sim-correctness conclusion yet. Completion bridge verified active/updating; E69/reproduction.json retains fieldwise evidence. |

| LEARN-AFK-24 diagnostic repair | E69 FAILED1835/1:0 at5m36s after full exact replay gate. First instrumented baseline comparison attempted np.asarray on LaneState.key (typed JAX PRNG), raising TypeError. No branch results. Comparator now checks structure/finiteness and compares key_data for typed keys. E69b reuses serialized baseline state/carry/key, checks restored fields and retains all ordinary-step/tick gates. | Probe bug, not evidence of simulator failure. E69 watcher stopped, registry cleanup checked. Unique retry ID/spec preserves failed run. |

| LEARN-AFK-24 tick findings | E69b COMPLETED1836/0:0 in8m52s; both instrumented baselines all-state-leaf maxerror0 against ordinary env_step and recorded E68 boundaries. Caster15: own87.3083 hit79.636359s leaves60.6917HP; minions reduce to1.6917HP; own next swing starts80.836734s; minion14 projectile23 kills80.953438s with ownwindup.1961s,CS9 unchanged. Caster19: own87.3083 hit112.513297s; Q reset/skip112.546641s, real swing112.579984s, Q152.2316 hit112.896750s leaves38.4602HP. Projectile22 deals23 at113.213516s, projectile20 kills113.280203s with ownCD.7847s,CS11 unchanged. | No missing champion hit or CS award in either baseline death tick; no simultaneous champion/minion lethal race, so same-tick attribution approximation does not explain these misses. This is exact JAX execution evidence, not a C# matched-trace parity proof or neural-cause attribution. Per-tick NPZ and sparse event JSON in /mnt/nfs/shared/E69b_afk_tick_audit/. |
| LEARN-AFK-24 physical options |76branches including2baselines;1622/1622 directed clicks resolve intended caster. Unit15 early6/20 and delay10/20 secure target, but local totalCS never exceeds baseline1 (previous unit7 CS traded away). Unit19 early1/17 and delay8/17 secure target and localCS1 versus baseline0. No own start/hit during any designated waiting prefix. Delay uses repeated near-self movement; some unit15 moves become unwalkable NOOPs, so timing also changes position/wave interaction. | Establishes local recoverability without simulator changes, not aggregate learned improvement. Next narrow refinement: preserve unit7 kill before withholding unit15's nonlethal swing, and preserve unit19's first normal hit before delaying Q. One ground move then NOOP avoids cumulative waiting drift; exact baseline/physical-click/zero-swing wait gates retained. No new training yet. |

| LEARN-AFK-25 plan | E70 narrows E69b physical options: preserve earlier unit7 kill, withhold unit15 next AA from79.2196s and release on200ms grid. Preserve unit19 normal hit112.5133s, delay Q from112.53s on100ms grid then physical target click. One executable ground move then NOOP, no repeated drift; zero-swing/no-target waiting and intended-target click assertions.20branches including2 exact baselines,30min CPU cap. | Hypothesis: delaying the specific nonlethal hit can recover CS while retaining preceding useful action. Still a physical multi-action option with small movement, not neural causality or aggregate evaluation. No training/physics change. |

| LEARN-AFK-25 results | E70 COMPLETED1837/0:0 in6m28s;20branches. Both instrumented/ordinary baselines maxerror0; every intervened branch prefix before withholding is array-exact across saved tick fields. Unit15: keep unit7 kill, withhold from79.2195625s; releases80.8200625/81.020125s secure caster at81.1535/81.3535625s, raising localCS1→2. Earlier7 tested releases fail. Representative hit81.1535s:43HP, sole87.30825 champion damage, killer0/CS10. Unit19: retain87.30825 hit112.513297s, withhold Q from112.529969s;7/9 delayedQ branches recoverCS0→1. Representative Q release112.930094s, then target click:113.396906s hit132.69174HP for152.23155, sole champion damage, killer0/CS12. | No start/hit during any waiting interval; all directed clicks resolve target; each uses one ground MOVE thenNOOP. Wait movement23.586units in both cases: physical timing/position option, not timing-only identification. Baseline death times/attribution reproduced; simulator can award CS correctly after delayed attacks. These are two selected local improvements, not a trained-policy/aggregate or C# parity result. Existing E69/E70 attack_seconds summary labels used nominal100ms offset; exact releases above read per-order timestamps, and future probe output uses actual timestamps. |
| LEARN-AFK-26 feature hypothesis | LEARN-AFK-24/25 now establish executable delayed-hit opportunities and correct local JAX hit/death/CS ordering. Test whether per-entity visible HP/position history improves learned timing beyond the existing pooled globalGRU. Prior literature precedent remains LEARN-AFK-23; no repeated search or claim of causal necessity. Proposed15past samples at10Hz (1.5s) plus current observation, with known masks and conservative visible position/type association; no actor target IDs/projectiles/hidden clocks. Zero-weight separate entity projection preserves v3 start. | First preflight visible association fidelity and actor/learner/reset/resume/migration gates. Then matched E67-initialized original/history PPO arms, fresh optimizers, same existing hyperparameters/rewards/staggering and512updates each (8.389Mdecisions/arm); frozen64games at0/128/256/512. Success requires finalCS>=10.921875 and >=control+1 with deaths<=.15, followed by independent confirmation if promising. Failure is no support for this bounded fine-tuning implementation, not proof history is useless. Runs not yet submitted. |

| LEARN-AFK-26 preflight fixture repair | E71 FAILED1838/1:0 at1m01s before history audit or training.6GPU tests passed; collector test inserted jax.random.normal weights without dtype while pytest enablesx64, promoting GRU output fromfloat32 tofloat64. Fixture now explicitlyfloat32 (kernel.dtype); production history code unchanged. E71b retains original gates/20min cap; prepared training configs superseded by E72b/E73b to point at E71b gate. | Test bug, not an observed production failure or a passed collector gate. GPU collector/reset/resume and E68 association tests remain required. No watcher was armed because startup canary failed. |

| LEARN-AFK-26 preflight passed | E71b COMPLETED1839/0:0 in4m16s; launcher reported complete/canary passed.14GPU tests cover history invariance/masking, float32 zero-weight migration/gradients, nonzero-history actor/learner likelihood, reset/resume/update, unchanged v3 AFK/rewards. E68 audit1201frames:97022/97022 claimed historical samples agree with offline identity+spawn labels (zero mismatches); consecutive-visible-minion one-step coverage6761/6951=97.26658%. All15past samples present at caster15 frames792/808 and caster19 frames1125/1129. | Predeclared99.9% precision/80% coverage gates passed on this one game, not general CV robustness or farming improvement. No actor IDs or raw attack timing used. Artifacts /mnt/nfs/shared/E71b_visible_history_preflight/{result.json,visible_histories.npz}. No watcher needed: completed during active startup watch/review. Proceed with matched E72b/E73b, original hyperparameters, initial frozen retention and final CS/death gate. |


### LEARN-AFK-27 — reference-deviation inventory (2026-10-01)

Dani redirected the next step from a history-only ablation to listing deviations,
then implementing a coherent published reference before ablations. E72b/1840
cancelled during startup checks; E73b never submitted. E71b established input
implementation correctness on one trace, not necessity or learned benefit.
A GRU can learn temporal information; no evidence establishes that ours cannot.
It sees pooled entity embeddings, not the raw per-unit sequence. This describes
its inference burden, not a demonstrated bottleneck. Pooling before recurrence
also exists in the references; slot permutation is not a bug.

Reference anchors: Tencent Solo AAAI2020 (LEARN-AFK-23 link, Figure2, algorithm,
implementation section and Table6) for the intended 1v1 system; OpenAI Five
AppendixE/Table4 for the more explicit feature inventory. Tencent5v5 is a
separate reference, not an undocumented specification of Tencent Solo. Do not
call a combination of these papers an exact reproduction. Current-side checks:
obs/builder.py, train/policy.py, train/ppo.py, train/vec_train.py,
train/wave_scenario_train.py and experiments/E67_afk_staggered.json.

| Component | Actual E67 path | Published comparison / evidence status | Priority / constraint |
|---|---|---|---|
| Unit health and combat stats | Rounded current HP fraction (1/60), relative position, type/team/subtype; no per-unit maxHP, damage, attack speed/range | Five Table4 supplies maxHP and richer combat stats. Tencent Solo's detailed inventory is underspecified; 5v5's is richer but separate | High feature gap; distinguish visible static/stat estimates from hidden state |
| Attack phase and incoming damage | No entity attack animation/timing or projectile inputs; no own AA cooldown/windup in E67 | Five supplies timing, incoming creep-projectile ETA and estimated melee attackers; some are scripted estimates. Tencent5v5 has spatial bullet channels | High feature gap; observable estimation required for future video. No privileged minion target IDs |
| Per-unit temporal inputs | Current frame only; global GRU learns memory | Five explicitly has last16 HP samples. Tencent Solo does not establish an explicit HP stack | High inventory item, not proof of necessity. Our15past position+HP stack is an adaptation, not a faithful full Five interface |
| Previous action and own status | No previous sampled action; own four ability cooldown/lock scalars, AD/AP/armor/MR/death/recall, but sparse buff/action state | Five includes previous action and richer status. AlphaStar includes orders/buffs/CD; not a mandate to use hidden state | Include in feature contract; earlier E62 own-action-only failure does not test complete reference inputs |
| Spatial encoder | Visible entities plus self/global vectors; no local-map CNN/channel input | Tencent Solo uses obstacle and hero-position image channels with CNN alongside unit/game features; Tencent5v5 includes additional spatial channels | Confirmed architecture/input difference; likely less specific to these in-lane misses |
| Target/action structure | Separate button,96-bin x,54-bin y heads from same core; no sampled-action-conditioned target head in E67 | Tencent Solo target attention with independent action labels/dependency masks (Eq1–3); Five sampled-action-conditioned unit selection | High secondary suspect. E66 mixture proposals were not a complete reference implementation. Preserve physical clicks by projecting a chosen observed entity to screen |
| Action availability | E67 click_mask=False; cooldowns observed but no full button availability mask. Decoder rejects/changes some invalid requests | Tencent Solo explicitly masks unavailable actions and physical restrictions | Confirmed exploration difference; mask only observable/known legality |
| Recurrent trunk | Shared128d,2-layer/4-head entity attention; max+mean pooling;4x1024 MLP;512 GRU with input norm/residual | Tencent Solo type encoders/pooling and1024 LSTM; Five4096 LSTM | Confirmed difference, not evidence GRU fails. Reproduce chosen reference before cell/size ablations; no requirement to copy Five's full scale |
| Critic gradient path | detach_critic=True: value head cannot train shared representation | Current code confirms deliberate deviation; exact equivalent gradient treatment in chosen paper not yet established | Explicit verification item, not a confirmed paper mismatch or higher-ranked cause |
| PPO and temporal credit | Standard clipped PPO,4epochs,gamma.99 at10Hz (~10s discount horizon),lambda.95;128-step recurrent sequences | Tencent dual-clip PPO and16-step LSTM sequences; Five much longer discount horizons,32-step iterations and different sample reuse | Confirmed algorithm/training differences; hyperparameters interact with collection regime. Do not restart sweeps or assume larger gamma fixes seconds-scale timing |
| Reward | Gold difference + lane approach shaping; XP0, HP-loss100/death300/personal tower900 in gold units; no extra CS-event reward | Tencent Table6 includes explicit last-hit bonus alongside gold, XP and other terms; zero-sum design | Confirmed objective difference. Compare definitions/scales, not raw coefficients across games. E65 farming-only failure is not reproduction of Tencent reward |
| Task, initialization and scale |120s AFK Garen scenario, fixed offset banks, inherited E46/E67 weights,128envs; bounded fine-tunes | Tencent full-game zero-start/self-play and much larger distributed training; Five different game/team task and scale | Intentional task/resource adaptations. Fine-tuning alone cannot establish failure of a newly matched architecture trained from scratch |

Working ranking: feature coverage first as a *hypothesis*, action/target design
and availability next; reward/credit/critic and recurrent implementation remain
real audit items. No newly established cause outranks feature coverage. The
exact two-incident tick tests lower suspicion of lost damage/CS there, but do
not certify simulator parity globally. Existing failures of individual add-ons
are not negative tests of a coherent published system.

Next deliverable is a reference contract: Tencent Solo as the intended primary
architecture/algorithm anchor, explicit unknown feature details, and a separately
labelled Five-inspired feature supplement where needed. Map every chosen input
to video/HUD/static knowledge or an observable estimator; list unavoidable game
and compute adaptations. Then train that declared baseline and compare frozen
outcomes, only afterward remove/add components. No new training submitted under
this direction yet. Do not silently resume E72b/E73b or label a hybrid exact parity.


| LEARN-AFK-27 source correction | Direct recheck of Tencent Solo Eq1–3 and Figure2: its labels are explicitly decoupled/independent; target query is FC(LSTM) against unit keys, with dependencies handled by masks. The p(t given a) notation does not by itself establish a sampled-button autoregressive network. Five Figure18 does explicitly condition target selection on sampled action. | Corrects overly broad preceding shorthand about action conditioning. Tencent's unit-attention target representation and dependency masks remain confirmed differences; independent coordinate heads alone are not a Tencent mismatch. |
| LEARN-AFK-27 objective detail | Tencent Eq3 sums per-label probability-ratio objectives. Our factored_log_prob sums used-head log probabilities and policy_loss exponentiates their difference, giving one joint-action ratio, then standard clipping. Tencent further uses dual clipping (epsilon.2,c3) for negative advantages; ours has only epsilon.2. | Distinct confirmed objective difference beyond the word PPO. Copying only dual clipping would still not reproduce Tencent's stated objective. No claim this difference caused the caster misses. |
| LEARN-AFK-27 known matches and missing settings | Tencent reports Adam initialLR1e-4 and GAE lambda.95: both match E67. It reports gamma.997 and a46s reward half-life,1024 LSTM,16steps,1600vector features and2image channels. Its evaluation cadence133ms must not be conflated with the training cadence implied by its stated half-life. | Record actual agreements, not a list where everything is called a defect. At our10Hz gamma.99 has6.90s half-life; matching46s would require approximately.998494, whereas copying.997 yields23.07s. A reference contract must state whether it matches per-decision discount or physical-time credit. Do not invent undisclosed feature definitions or optimizer coefficients. |
| LEARN-AFK-27 additional Five preprocessing gap | Five AppendixE normalizes float observations by running mean/std and clips to[-5,5]. Our builder uses fixed scaling constants and quantized HP fractions. | Confirmed preprocessing difference, not causal proof. Any copied running statistics must be checkpointed and frozen consistently during evaluation. Not implemented. |

Reference-contract draft, still documentation only:

- **Tencent components directly specified:** type-wise unit encoders and maxpool,
  separate retained unit keys for target attention, image/vector/game encoders,
  LSTM1024, independent action labels with dependency/availability masks,
  component-ratio objective plus dual clipping, published reward categories,
  Adam LR1e-4/lambda.95. Unspecified layer widths, feature encodings, exact masks,
  entropy/value weighting, sequence-state handling and reward transforms need
  explicit engineering choices or stronger primary implementation evidence.
- **Current known adaptations that cannot be concealed:** League/Garen rather
  than Honor of Kings; physical mouse/keyboard outputs rather than game-core
  unit targets; one GPU; AFK farming evaluation rather than professional 1v1
  win rate. Selecting an observed entity internally can still produce a normal
  click, but target-to-screen mapping/collisions must be specified and tested.
- **Input provenance:** visible bars/positions/types come from the observable
  scene; own stats/ability availability can come from HUD; previous action comes
  from our controller. Public static combat data is a separate declared source.
  Animation phase, facing, projectiles and per-unit history require observation
  tracking/estimation. Raw minion target IDs, hidden cooldowns and fogged current
  state must not become actor inputs. Merely naming an estimator is not evidence
  it can recover that field accurately from video.
- **Five-derived supplement is optional and explicitly separate:** richer
  combat/timing/history features and running normalization have a published
  precedent, but adding them does not fill Tencent's unpublished1600-vector
  specification exactly. E71b validated one visible-history implementation,
  not this whole supplement. Choice of primary reference was asked asynchronously;
  no answer is assumed and no implementation/training depends on it yet.
- **Learning evaluation once a baseline is declared:** keep frozen E67 as the
  existing behavior comparator; evaluate the chosen architecture trained as a
  baseline, not just a migrated short fine-tune that may preserve old habits.
  Record intentional budget/task differences. Only after a working baseline
  should component removals/additions be interpreted as ablations. No new run
  budget or success claim is being smuggled into this documentation audit.


### LEARN-AFK-28 — official Tencent implementation reference found

Read-only source audit, no new code/run. Official
[hok_env repository](https://github.com/tencent-ailab/hok_env) exposes a later
HoK1v1 baseline; inspected commit `c6e0029a5e4b6e037a049804fdbe23ac74484311`.
It is a distinct versioned implementation reference, not proof of the exact
AAAI2020 professional Tencent Solo model. This improves reproducibility options
without inventing the missing2020 details.

| Evidence | Finding | Consequence for deviation inventory |
|---|---|---|
| [Official environment observation table](https://aiarena.tencent.com/hok/doc/environments/index.html#observations) | Documented creep18-vector contains HP, HP fraction,maxHP,attack power,kill income,positions/distances,type/alive/team and buff mark. Hero features include normal-attack availability and richer combat/skill state; public features include nearest enemy bullet position/distance. | A concrete Tencent1v1 feature reference exists beyond the2020paper. E67 omits several documented fields. This table does not establish an explicit HP-history stack, creep-projectile ETA or universal AA-windup field. Video-compatible mapping still required; do not copy hidden opponent cooldowns just because an API exposes them. |
| [Pinned baseline config](https://github.com/tencent-ailab/hok_env/blob/c6e0029a5e4b6e037a049804fdbe23ac74484311/aiarena/1v1/common/config.py) |725observation scalars,512LSTM,16steps,targetembedding32,LR1e-4,gamma.995,lambda.95,entropy beta.025; six action components. | Different from2020paper's1600vectors/2image channels/1024LSTM/gamma.997. Our512memory size is not smaller than this later baseline; GRU vs LSTM and trunk still differ. Do not conflate these baselines. |
| [Pinned PyTorch network/loss](https://github.com/tencent-ailab/hok_env/blob/c6e0029a5e4b6e037a049804fdbe23ac74484311/aiarena/1v1/common/algorithm_torch.py) | Type-specific MLP processing,512 concat projection,512LSTM,32dtarget attention,valueMLP512→64→1. Per-component weighted surrogate; ratio capped3 with PPO.2. Value MSE contributes through shared LSTM (no detach on this path), unlike E67; no value clipping in this loss. | Resolves the critic-gradient question for this specific official implementation, not the unspecified2020implementation. Reveals value clipping and value-head depth differences too. Network/learner code now inspectable; no port or improvement claim. |
| Documentation/config version boundary | Official web table ends at491features (128perhero); pinned code expects725 (235perhero+14mainhero+25global+minion/turret blocks). Creep blocks are18in both, but this does not establish every field's version identity. | Must select a consistent code/schema version before claiming reproduction. A live docs table plus current config cannot silently form one interface. Primary reference selection remains open; training deferred. |

Revalidated E69 reproduction.json:1201frames, all recorded comparisons0,
checkpointSHA matches7f12479...,12CS/0deaths. Re-read E70 representative NPZs:
all saved fields before intervention array-identical over132ticks(unit15) and
150ticks(unit19); branchJSON localCS1→2 and0→1, killer0 on delayed kills.
Actual order releases80.8200625s and112.93009375s match LEARN-AFK-25.
This strengthens artifact traceability only, not neural-cause attribution.


### LEARN-AFK-29 — Tencent-documented combat feature package

Dani delegated the reference choice, balancing minimal change and plausibility
of improving farming. Select the documented HoK1v1 combat-feature semantics,
retain E67's GRU/PPO/reward/action heads initially. This supersedes the proposed
history-only study. Later official implementation and2020paper remain distinct;
we do not claim to reproduce either full network or resolve their whole schema.
The491-vs725 distinction is handled by explicitly using the documented field
semantics as reference, not importing inconsistent tensor offsets from both.

Implemented opt-in v6:27entity fields (original16+11),33self fields (16+17).
New entity fields: bar-derived HP points,maxHP,base/live-level AD,attack range,
attack speed,current movement speed,profile kill income,self distance,absolute
lane s/n,level. New self: bar-derived HP/maxHP,range/AS/movement speed,own AA
availability,Q/W/E active bits,Q/W/E/R ranks and four cast-availability bits.
No stack, projectile estimates or previous actions added. Own AD/armor/MR,
relative positions,HP fractions and other original features remain unchanged.
This is an applicable combat-stat/readiness package, not the entire documented
HoK observation: hero-specific HoK kits/items/mana don't apply to this Garen
sim; nearest enemy-hero bullet tracking and other reference differences remain
explicitly unimplemented. No claim of full feature parity.

Provenance/limits: public type/level/profile stats plus visible scene/HUD.
Current HP points use the old1/60-rounded bar times maxHP, never precise stateHP.
Only existing visible slots receive stats; padding zero. No enemy AA clocks,
raw target IDs or fogged current data are actor inputs. Own readiness reads
simulator own cooldown/status as an observation prototype, with no target/range
oracle; a video estimator and wire support must be validated later. Driver
rejectsv6 for C# until then. Current no-items scope is explicit; profile damage
is AD, not target-specific post-mitigation damage or a scripted kill forecast.

Two separate bias-free zero projections preserve the old entity/context matrix
operations, allowing identical initial policy behavior and trainable new inputs.
No physics or vendor changes. E74 GPU preflight tests provenance/quantization,
readiness/padding/permutation, migration/gradients and nonzero actor/learner
likelihood across reset plus finite update, alongside existingv3canaries.
Prepared E75control/E76feature use E67final/freshAdam, original settings and
512updates8.389M decisions each,1hSlurm/55minworker cap. Require E74pass and
initial64-game frozen retention before learning. Final gate featureCS>=10.921875
and >=control+1, deaths<=.15, then independent confirmation if promising.
Failed bounded fine-tuning does not disprove the full reference or the features.
No training submitted at this planning/implementation entry.

| LEARN-AFK-29 E74 retry | E74/1848 FAILED1:0,1m48s. Four tests passed including nonzero actor/learner/reset/update; migration error~1.97e-5 exceeded1e-6. Test used default CUDA matrix precision; worker sets highest. Retry uses production precision, same tolerance. Separately, fountain attack_period0 requires a finite observation: saturate AS at10/s and add zero-period regression. | E74b unique retry; E75b/E76b prepared with new gate. Old training IDs never launched. No training or passed preflight claim. E74 ended before canary marker, no watcher. Precision explanation remains a hypothesis pending retry. |

| LEARN-AFK-29 E74b writer failure | E74b/1850 FAILED1:0,4m22s after all13GPUtests passed and canary marker. Migration passes at production highest precision with unchanged tolerance; finite fountain regression passes. Worker result writer ran git rev-parse but .git references a login-only /srv worktree path. | Use launcher launch.json source after checking job/spec identity. E74c reruns same tests with writer-only fix; no training yet. E75c/E76c replace never-submitted b configs solely for gate path. Job failed before unattended handoff, no bridge needed. |

| LEARN-AFK-29 preflight passed | E74c/1853 COMPLETED0:0,4m21s;13GPUtests passed, result.json passed with matching launch job/spec/source7b8e573. Original launcher falsely reported failure because worker lacked PROFILE COMPLETE marker. Add marker and authoritative COMPLETED/0:0+canary fallback; invoke launcher verifier read-only on1853, which reports complete/canary passed. Negative checks reject TIMEOUT0:0,FAILED,CANCELLED and RUNNING. | Quality gate passed; no learned gain. No test rerun needed for reporting-only repair. Completed under active watch, no bridge required. E75c/E76c prepared with matched settings and same final gates. |

| LEARN-AFK-29 field mapping clarification | Tencent's documented creep block explicitly supplies HP/maxHP,AD,income and geometry; attack/movement speed and attack range appear in its hero/turret inventories. v6 uses a common public-stat schema across our unit types, extending those fields to minions using known profiles. | This shared-schema adaptation is explicit, not a claim that every new minion field appears in Tencent's creep18-vector. Baseline remains feature alignment rather than exact full-interface parity. Endpoint scorer parsed the real128team-episode E67 cohort, reproduced9.921875blueCS and rejected an altered initialCS record. |

| LEARN-AFK-29 control startup | E75c/1856 RUNNING;15base tests plus1stagger test passed, launcher reported healthy after180s. Fixed-policy warmup discards shortened episodes; initial frozen retention still pending. Event service lanerl-event-1856 active and result timestamp advances; expires2026-10-01T21:53:59UTC. | No frozen learning result yet. E76c not submitted. Handoff to bounded completion bridge; review terminal State+ExitCode and frozen endpoint before feature arm. |

| LEARN-AFK-29 initial control retention | E75c/1856 all64 frozen games/bothteams pass initial E67 reference checks. Read-only scorer independently verifies recorded update0 against E67update512:blue9.921875CS/.140625deaths. | Confirms intended unchanged starting behavior; training active, no learned result yet. |

| LEARN-AFK-29 control intermediate | E75c update128 frozen64games:8.640625CS/.328125deaths/749.9967personal towerHP, versus initial9.921875/.140625/802.3992. Training finite through132updates. | Intermediate regression, not final outcome. Continue declared512 endpoint. E76c must exceed both initial absolute CS gate and final control; no hyperparameter change or best-intermediate selection. |

| LEARN-AFK-29 control final | E75c/1856 COMPLETED0:0,41m26s,512updates/8.389M decisions;512metric rows,zero reported nonfinite losses,no traceback. Manifest matches spec and all64initial games/bothteams retain E67. Final frozen64:9.453125CS/.109375deaths/764.71753personal towerHP; initial9.921875/.140625/802.39920. LowHP32:9.25CS/.09375deaths; fullHP32:9.65625/.125. Final checkpoint SHA256 cfe534cbeca916fde8f8329c30a44ef7814ed0faf905781c5d458fa7b9de44b6. | Extra unchanged-input training yields -.46875CS and -.03125deaths; no farming improvement or feature conclusion. This is the declared final control, not a selected intermediate. E76c dry-run passed but remains unsubmitted while automatic goal continuation is paused at user stand-down request. Bridge notification accepted; service inactive and registry entry absent. No running/queued project jobs. |

| LEARN-AFK-29 user HP revision | User requests direct HP/maxHP for the reference baseline. Opt-in combat extras now read current HP for already-visible units and self, retaining maxHP and unchanged original v3 columns. Regression checks direct1.7HP even when the original bar rounds to0; hidden-unit masking remains. Syntax checked only; revised GPU canaries pending. | Supersedes bar-reconstruction semantics above for future runs; earlier E74c gate is not evidence for this revision. Exact HP is a simulator-reference input, not a demonstrated video measurement. No new experiment launched. Projectile draft is separate/uncommitted, scratch initialization pending. |

### LEARN-AFK-30 — requested legality masks, projectile idea remains separate

User authorizes button/map masks. Combat self vector grows33→35 with own
move/recall availability. Shared policy forward masks dead/locked buttons and
unavailable Q/W/E/R, preserving E cancel and queueable attack-move on AA cooldown.
NOOP always legal. No target IDs or tactical kill thresholds. R target/range
and target-conditioned click masks remain unimplemented: this is own readiness
plus map masking, not complete Tencent action parity. Existing joint screen
mask excludes unstandable map cells and minimap; all-true fallback only when no
standable cells exist. Actor and learner use identical masked distributions.
Scenario launcher exposes both switches. Existing automatic combat submission
suite now tests masked sampling/logprob/reset/finite update; E77 verifies these
and existing map-mask/v3 regressions. No trained result or new training launch.
Projectile integration draft removed from live imports; untracked draft remains
untested and outside this study. Direct HP follows preceding user revision.

Re-read official491-field environment table: fields256–259 describe nearest
enemy-hero projectile x/z/distance. Listed18-field minion block has visibility,
alive/team/type, positions/distances,HP/fraction/maxHP,AD,income,buff mark;
no explicit target, cooldown, windup or animation phase. This is specific to
that documented interface, not proof about the original2020professional model
or later725-field baseline. Raw game state availability is not actor-feature
availability. Source: https://aiarena.tencent.com/hok/doc/environments/index.html

### LEARN-AFK-31 — actual public Tencent pretrained artifacts

Official hok_env releases include3v3baseline checkpoints (v2.0.1), but its
current tree/releases do not establish original2020Tencent Solo weights.
A separate official Tencent repository, hokoff, explicitly provides1v1 and3v3
multi-level pretrained models:
https://github.com/tencent-ailab/hokoff#multi-level-models
Pinned source9b35f7e5891ad98df45a36e3a18f5192a31e72f4. Official1v1 ZIP HEAD
returned200,269962162bytes. Read only ~75KB of HTTP-range ZIP directory data:
74entries, TensorFlow level0–7 directories containing model.ckpt.index,
model.ckpt.data and model.ckpt.meta plus hero_config.json. No full download,
weight loading, conversion or transfer run. Baseline evaluator imports
TensorFlow compat.v1 and BasicLSTMCell; these are incompatible with loading
weights directly into our GRU. These later checkpoints are a concrete transfer
candidate, not evidence the professional2020system's weights are released.

Transfer work would first verify matching checkpoint/network/schema, then map
League observations/scales/missing fields, preserve or faithfully port the
pretrained recurrent network, adapt the HoK buttons/directions/unit-target
outputs to Garen's physical clicks, and evaluate fine-tuning against the same
architecture initialized randomly. Cross-game benefit is unproven. Current
mask/feature study is separate and does not perform any of this transfer.

| LEARN-AFK-30 completed gate | E77/1857 COMPLETED, ExitCode0:0, elapsed4m27s, ended2026-10-01T21:28:35UTC. Existing log has19passed tests (8+4+7), CANARY PASSED and PROFILE COMPLETE; result.json passed with sourceb63227e5 and matching experiment/job. | Reviewed after interrupted handoff; no re-run. Masks/direct HP passed implementation checks, not a farming evaluation. No1857 bridge service or registry entry; no RL jobs running or queued. |

### LEARN-AFK-31 — weight quality and transfer recommendation, continued

Primary sources newly inspected: [HoKoff paper, AppendixD/F](https://arxiv.org/html/2408.10556v2),
[actual evaluator config](https://github.com/tencent-ailab/hokoff/blob/9b35f7e5891ad98df45a36e3a18f5192a31e72f4/hok1v1/offline_eval/config/common_config.py),
[TensorFlow evaluator](https://github.com/tencent-ailab/hokoff/blob/9b35f7e5891ad98df45a36e3a18f5192a31e72f4/hok1v1/offline_eval/baselinemodel/algorithm.py).

| Question | Finding | Consequence |
|---|---|---|
| How good are released weights? | HoKoff publishes a ladder of dual-clip-PPO checkpoints. Table13 reports level7 beating level6 in70% of games, level6 beating5 in73%; evaluation fixes the hero to LuBan. The authors describe varying human-level abilities, but this is not an independent human-rank calibration. | Useful trained opponents/teachers in their own game; no published perfect-CS benchmark identified. Do not attach original Solo's professional-match or99.81% public-match result to this released checkpoint. |
| Do their inputs match ours? | Runtime config:725features =235ownhero+235enemyhero+14public+8x18creeps+4x18structures+25global. Current E67 uses our entity/self/global encodings; opt-in combat inputs only partially align semantics. | Matching names such as HP is insufficient: ordering, scaling, hero-specific fields, units, slot limits and masks differ. The old491-field documentation is not a full725-field specification. |
| Can we load into our GRU? | Evaluator uses TensorFlow BasicLSTMCell512 and six heads of sizes12/16/16/16/16/8. Our recurrent trunk and mouse/keyboard action interface differ. | Preserve/port the LSTM and its encoders first, then adapt League observations and targets to physical clicks. Alternatively distill an adapted teacher into our GRU. Neither is direct checkpoint loading. |
| Does it solve video perception? | Tencent uses game-core observations, with invisible units defaulted. Even original Solo's two spatial channels are game-core maps, not rendered video. | Video-to-feature tracking is separate. Positions/bars/HUD/static knowledge are plausible sources; exactHP, timing and inferred missing state need calibrated estimates. Train/evaluate with those errors before claiming video transfer. |

Recommendation: use the implementation as a reference, defer cross-game weights
as the primary repair. A credible port/adapter/comparison is an estimated
several days to weeks of engineering, conditional on resolving feature semantics;
this is an estimate, not a scheduled commitment. Framework conversion is a smaller
problem than transferring HoK hero mechanics/targeting to Garen. Weight benefit
must beat random initialization of the SAME adapted architecture on the same
frozen League cohort. No checkpoint loaded or new implementation/run submitted.

Rechecked the official491-field observation table directly: offsets256–258
are the nearest enemy-hero bullet position/distance, not all caster missiles.
The listed creep18-fields omit explicit missile/attack-phase inputs. This supports
the possibility of farming without explicit creep-projectile tokens; it does
not prove those tokens are useless, or establish the full725/original1600 schema.
Recurrent HP/position observations can supply indirect timing information;
different hidden attack phases can nevertheless remain ambiguous. Projectiles
remain a hypothesis, not an implemented solution.

### LEARN-AFK-32 — avoidable waste, pushing and demonstration discussion

Dani clarifies the question: why leave obtainable CS/tower damage against AFK
(roughly9CS versus a suggested12–18opportunities), not whether any learning ever
occurred. That opportunity denominator is not measured by the aggregate frozen
CS mean; do not silently promote it to a certified ceiling. LEARN-AFK-24/25
already establish two recoverable local misses, without proving aggregate maxima.

| Evidence / hypothesis | Interpretation |
|---|---|
| E46 frozen120s AFK CS5.453125→9.5625 and deaths.9375→0; E67 final9.921875 | Learning occurred, but substantial inefficiency remains. Earlier gain does not answer Dani's optimization question. |
| E75c: unchanged-input E67 continuation, fresh Adam,512updates/8.389Mdecisions, same reward and hyperparameters; frozen64 CS9.921875→9.453125 | This bounded continuation did not improve farming. It cannot establish that longer training, another seed, or scratch training will fail. |
| Leading timing hypothesis: a coarse repeated-attack routine earns enough reward to persist, while learning the better wait/target sequence requires precise exploration and credit | Compatible with recoverable caster misses; not a diagnosed neural cause. HP precision, own availability, action structure, representation, reward and PPO updates remain alternatives. Discounting alone is weak explanation for a subsecond missed last hit. |
| Tencent Solo uses target attention, legality masks, explicit last-hit reward, richer observations and vastly more experience | [Original paper](https://arxiv.org/pdf/1912.09729) reports48P40GPUs+18000CPUcores per hero and full-game self-play. These make discovery/reinforcement of successful play more plausible, but neither isolate the cause nor demonstrate perfect last-hitting. |
| Pushing/proxying: current120s level3 resets, gamma.99 at10Hz, XP0, HP-loss/death penalties and personal-only tower damage credit | Reward half-life6.90s; a benefit30s later is weighted4.90%. Proxy travel can incur immediate risk and delayed benefit; allied-minion tower damage receives no direct personal-tower reward. Gold still rewards farming. Hard push/proxy are not established optimal under this objective; CS and immediate tower damage can trade off. |

More training remains a valid hypothesis. The established
[grokking result](https://arxiv.org/abs/2201.02177) concerns delayed held-out
generalization after fitting small algorithmic training sets; our unsolved
training behavior does not demonstrate that pattern. Prefer a declared longer
budget and fixed frozen checkpoints over assuming a breakthrough will arrive.
No longer run proposed as an already-submitted experiment.

Human demonstrations are promising and have local precedent: the older
BC/DAgger ledger reports frozen JAX mirror clones around41–45CS on a different,
longer task; later PPO often degraded them. These are not120s AFK comparator
scores. Existing `train/bc_diag.py` and `train/bc_dagger.py` provide a starting
point; no new imitation implementation here.

Discussion proposal: record synchronized policy-visible observations and actual
mouse/keyboard actions while Dani plays the SAME simulator scenario, including
waiting, last hits and pushing. Plain video lacks reliable action labels. Start
with5two-minute demonstrations for a narrow fitting check, then20–50varied games
(24k–60kdecisions at10Hz) and corrections on states the learner visits. This is
an engineering data budget, not a guarantee or a literature-derived minimum.
Repeated optimization over the same game does not supply new state coverage.
[DAgger](https://proceedings.mlr.press/v15/ross11a.html) motivates gathering
corrections on the learner's own states instead of only expert trajectories.

First score demonstrated better behavior under the CURRENT reward on matched
starts. If higher CS/tower play earns less, the objective is misaligned with the
desired behavior. If it earns more but PPO does not reach it, investigate discovery,
credit and optimization. If cloning cannot fit even demonstration sequences,
inspect observation/action alignment, available information, capacity and optimizer;
do not conclude missing features from that failure alone. If it fits recordings
but fails fresh rollouts, coverage and accumulated errors matter. If cloned play
works then deteriorates under PPO, isolate the reward/update stage.

For later interpretability, demonstrations also label wait-versus-attack decisions
and expected kill windows. Decoding that distinction from hidden state is only
correlation; controlled memory/input interventions with frozen behavior are needed
to claim the policy uses it. Behavioral comparisons above are the first useful
debugging step. This entry records discussion/recommendations, not authorization
to implement every suggestion.

### LEARN-AFK-33 — literal CS-only continuation and alternatives to human play

Dani explicitly requests approximately40min of existing-policy training with CS
reward and nothing else. E78 selects512updates/8,388,608 learner decisions, based
on E75c512updates completing in41m26s.55minworker/1hSlurm ceiling; actual runtime
is an estimate. E67checkpoint/freshAdam, v3 inputs, unmasked original actions,
GRU, gamma.99/LR1e-4/4epochs, seed0 and staggered120s AFK episodes unchanged.
E77-approved features/masks are not enabled in this reward-only comparison.

Reward is exactly nextCS-currentCS for each champion, with no enemy subtraction,
gold/XP/position/health/death/tower terms. Existing entropy regularization and PPO
optimizer settings remain unchanged; they are not environment rewards. This is a
CS-only discounted objective, not a claim of undiscounted episode optimization.
E65 previously retained gold/death/position reward, used E46 initialization and
unstaggered collection; its failure does not answer this exact request. Tencent's
explicit last-hit term (LEARN-AFK-27/31) is relevant precedent for rewarding last
hits, not evidence that our pure-CS configuration will succeed.

Frozen64games at0/128/256/512. Initial cohort must reproduce E67physical metrics;
only reward comparison is disabled explicitly because that is the intervention.
Every frozen episode checks reward=CS and all non-CS terms=0. Integrated existing
submission suite adds a JIT reward-isolation canary with simultaneous unrelated
state changes and asymmetric CS counts. Primary endpoint is final meanCS>=10.921875
(E67+1); report paired episode differences and compare E75ccontrol9.453125. No
best-intermediate selection. Deaths and personal tower damage remain diagnostics,
not extra rewards or a survival success gate that would contradict the test.
Stop for nonfinite training, contract/canary failure or budget. No blind extension.

Human demonstration recommendation revised: the playable JAX interface is not
currently convenient, so building it is not the preferred prerequisite. Discussed
alternatives, not submitted follow-ups: (1) run the existing scripted last-hitter
on the SAME AFK starts to measure a feasible CS comparator and miss types;
(2) use existing BC/DAgger with scripted labels to test whether the current
observations/network can acquire and retain better timing; teacher privileged
inputs, if any, must be disclosed and need not be deployable actor inputs;
(3) short randomized last-hit situations followed by full-wave frozen evaluation
can test whether PPO learns the local skill but fails to discover/retain it in
full episodes. A scripted comparator is not a mathematical optimality bound.
Existing E69/E70 counterfactual tools already demonstrate local opportunities;
avoid repeating those checks without a new question. No manual client, teacher
experiment, observation change or Tencent port implemented in this turn.

| LEARN-AFK-33 LR revision | Before submission, Dani requests a roughly10x-or-evidence-based LR increase. E79 replaces never-submitted E78 with LR3e-4 (3x); all other E78 settings/gates retained. E75c512update metrics at1e-4: median sampledpostKL.02684,p95.03596,max.05630; approxKL median.01601; clip_frac median.19711; median gradient norm2.04032, clipping fraction1.0. E50lower3e-5 medianpostKL.01433 versus E51at1e-4 .02747; higher arm had better finalCS8.8594vs8.0313 but neither exceeded9.5625initial. | Steps are not demonstrably too small;3x is an exploratory larger-step choice, not an inferred optimum.10x is not supported by these diagnostics. Reward and LR now both differ from E75c; no isolated causal attribution. Existing sample-based postKL replays128step recurrent segments from cached initial carry, not full-episode hidden-state recomputation or exact categorical KL. PPO paper https://arxiv.org/abs/1707.06347 supports measuring policy movement but clipping is not a hard KL bound. No adaptive LR controller added. |

| LEARN-AFK-33 LR clarification | Dani clarifies the aim is a sensible effective step, not a larger numerical LR. Cancel just-submitted E79/2160; select original never-submitted E78 at1e-4. E75c stable KL/clipping do not establish excessive or insufficient movement, and the older3e-5arm performed worse. | Withdraw unsupported3x increase.1e-4 is the established working setting, not a claimed optimum; pure-CS comparison now isolates the reward intervention against E75c. No automatic LR adaptation. Terminal/pre-training status pending verification. |

| LEARN-AFK-33 E79 cancellation verified | Slurm2160 CANCELLED/0:0 at46s; log ends during initial pytest, no CANARY PASSED marker and no training directory. | Zero training; cancelled job was ours. No bridge/registry entry created. Original E78 spec remains unchanged and its dry-run already passed. |

| LEARN-AFK-33 existing automated comparator | Re-read E58b: observation-only scripted control already achieved13.5CS versus E46policy9.90625 and E57policy8.90625 on the same32diagnostic120s starts; no deaths. These are a different cohort/checkpoint from current E67frozen64, and scripted duplicate trajectories are not independent samples. | Do not repeat a broad teacher audit just to rediscover achievable improvement. Prefer reusing the existing teacher/BC path or a short randomized last-hit curriculum if E78does not help. No human-play interface required for either option. |

| LEARN-AFK-33 E78 startup | Slurm2161 RUNNING;17integrated GPU tests passed including new literal-CS isolation, original collector/actor likelihood and stagger warmup. Launcher reports healthy after180s. Sourcecf0ff0b; existing E67weights, LR1e-4,512updates. | Full worker/initial frozen retention still pending; no learned outcome. Bridge lanerl-event-2161.service armed, expiry2026-10-02T00:18:41.052256+00:00; service active, registry present and result timestamp advancement verified. |

### LEARN-AFK-34 — Tencent image channels and the future video adapter

Question-only source review; no implementation or new experiment. Re-read
[Tencent Solo Figure2/architecture/System Setup](https://arxiv.org/pdf/1912.09729):
image features use convolutions; unit vectors and game-state vectors use FCs,
including FC/ReLU unit encoders. Their encodings combine before the LSTM. The
specified image inputs are two game-core channels: obstacles and hero positions,
not rendered RGB footage. Thus that image branch does not establish a hidden
source of caster-projectile observations.

The separate [Tencent5v5 paper Table1, page15](https://arxiv.org/pdf/2011.12692)
uses6x17x17 spatial features: ally/enemy skill-damage regions, ally/enemy skill
bullets, obstacles and bushes. This is concrete projectile-map precedent but
neither raw screenshot perception nor proof of all minion auto-attack projectile
coverage. Keep these systems separate from Solo and later HoKoff checkpoints.

Recommendation under discussion: retain a compact policy interface and train
video perception separately with supervised recorded-footage labels. Define and
validate observable feature estimates early, including missing detections, health
precision and latency, rather than assuming exact simulator inputs can simply be
replaced later. The adapter can output tracked entities/projectiles and optional
small spatial channels; it need not force the policy to consume raw pixels.
A small CNN on compact semantic maps is plausible on our5080 but needs a measured
throughput comparison. For last-hit timing, continuous projectile position/motion
features may preserve precision better than a coarse occupancy grid. This is an
engineering hypothesis, not a selected implementation or proven learning benefit.

Compute distinction: end-to-end pixels add rendering, image-encoder training and
perception sample complexity. At128envs x128steps, even256x256RGB frames occupy
3GiB as uncompresseduint8 (12GiBfloat32), before activations, model/optimizer or
multiple frames; streaming, compression and cached encodings can reduce this, so
it is an illustrative storage calculation, not a measured throughput limit.
Current JAX state simulation supplies no game-faithful League RGB renderer.
Compact maps/lists can instead be generated in simulation and estimated by the
future video adapter. E78continues unchanged; no new feature branch launched.

| LEARN-AFK-33 E78 final | Slurm2161 COMPLETED0:0 in42m28s, ended2026-10-01 3:53PM Pacific(PDT);512updates/8,388,608decisions. Frozen64games120s at0/128/256/512: CS9.921875/10.765625/11.375/10.0; deaths.140625/.015625/.015625/.078125; personal towerHP802.399197/48.152639/7.814034/181.035824. Final lowHP32:10.25CS/.0625deaths; fullHP32:9.75CS/.09375deaths. Final pairedCS delta versus initial mean+.078125,median0,30better/8equal/26worse,range−7to+6; final versus E75c9.453125 is+.546875. | Fails predeclared final>=10.921875CS. Temporary improvement at256 was not retained at512; do not promote the intermediate to a successful endpoint. Reward removal alone did not produce the targeted sustained improvement at this budget/LR. One training seed, not proof longer training/features cannot help or causal evidence LR is too high. |
| LEARN-AFK-33 E78 contract/diagnostics | Manifest confirms cs_only=True, LR1e-4,gamma.99,4epochs,staggered128envs,originalv3 and exact E67initSHA. Initial physical-retention marker passed. Independently checked all4frozen cohorts/bothteams: episode reward=CS=CSreward term, every other reward term0.512metric rows/nonfinite flags0/no traceback; full completion marker. Final8388608checkpoint/latest SHA identical:cd136aa512550c22c8df6e0a7ea58d9ebea3ae520bf0478b5331ea5b7a69be91. Run /mnt/nfs/checkpoints/lanerl-jax/E78_afk_cs_only/vec-s0-20261001-221924-cf0ff0bb/. | No extra evaluation or test rerun. Median sampledpostKL.026605,max.047038; first/last32means.024414/.026747. Clip fraction first/last32mean.203589/.205034; entropy7.41933→6.58365; EV.750704→.867087; value loss.086421→.020467. These train diagnostics show no obvious runaway policy-KL spike, but do not certify recurrent behavioral stability or identify the cause of CS decline. |
| LEARN-AFK-33 E78 bridge cleanup | Result delivery accepted with authoritative COMPLETED/0:0 accounting; lanerl-event-2161.service inactive and registry entry absent. | Completed as budgeted; no RL job running/queued, no follow-up submitted. Review supports discussing retention of improved timing under continued PPO, alongside existing curriculum/teacher options; no new experiment authorized by the event. |

### LEARN-AFK-35 — E80 twelve-hour CS-only continuation

Explicit user request: queue the same setup for 12 hours. Continue from E78's
**final** update512 parameters (10.0 frozen CS), with fresh Adam as required for
a new experiment. Do not select E78's better intermediate checkpoint. Keep
constant LR1e-4, entropy.001, gamma.99, four PPO epochs, original v3/GRU/unmasked
actions, 128 environments x128 steps, seed0 and staggered 120s AFK episodes.
Environment reward remains exactly +1 per own credited CS and nothing else.

Hypothesis: longer training can acquire and retain better last-hit behavior;
E78's transient rise to11.375 then fall to10.0 does not establish a learning
ceiling. Existing PPO reference (https://arxiv.org/abs/1707.06347) and E78's
stable sampled policy-KL support retaining the established update configuration,
not predicting success. Duration estimate is engineering extrapolation from
E78's mean3.632849s/update plus startup/compilation/evaluation overhead.

Budget: 11,520 updates /188,743,680 additional learner decisions; expected about
11h50 including startup, worker cap42,300s (11h45), Slurm cap12h. Existing
integrated canaries, checkpoint cadence and signal handling are reused. Frozen
64-game suite at0,512,1024,2048,3072,4096,5120,6144,7168,8192,9216,10240,11520.
Initial physical and reward trajectories must reproduce E78final; every frozen
episode must satisfy reward=CS with all other reward terms zero.

Primary endpoint: final update11520 meanCS>=11.0 (start+1); report paired episode
deltas and all scheduled means. Secondary retention: each of the final three
scheduled means >=11.0. Deaths and tower damage are diagnostics. No selection of
the best intermediate, human demonstrations, feature changes or adaptive LR.
Stop for nonfinite/contract/canary failure, signal or cap. An early cap is
incomplete against the declared endpoint, not a completed success. No automatic
extension; one training seed cannot establish generalization or grokking.

| LEARN-AFK-35 setup | E80_afk_cs_only_12h prepared; E78 final SHA256 cd136aa512550c22c8df6e0a7ea58d9ebea3ae520bf0478b5331ea5b7a69be91. | Submission/startup/bridge pending; no new outcome. |

| LEARN-AFK-35 startup | E80 Slurm2171 RUNNING; source b7130a5; integrated canaries and launcher180s healthy signal passed. Started2026-10-02T00:28:42UTC (Oct1 5:28PM PDT); 12h hard cutoff Oct2 5:28AM PDT, expected about5:20AM. | Bridge lanerl-event-2171.service active/registered and result timestamp advances; expiry 2026-10-02T14:36:09.928633+00:00. No frozen learning outcome yet. |

### LEARN-AFK-36 — diagnostic discussion while E80 runs

Dani asks to discuss likely causes, fixes and parallels; this is not an instruction
to implement another hypothesis. E80 continues unchanged. E78's frozen rise from
9.921875 to11.375 followed by10.0 makes acquisition versus retention a concrete
question, not proof of catastrophic forgetting, a representation ceiling or an
incorrect LR. E54's earlier hit-associated probability drift motivates that
question but does not identify causal actions; E56's one-epoch failure already
shows that improving such a proxy need not improve gameplay. Avoid repeating
unfocused LR/epoch sweeps.

Existing E58b scripted comparator gets13.5CS versus E46/E57 9.90625/8.90625 on its
32-game diagnostic cohort. It uses observation tensors plus declared public
static combat knowledge, no hidden LaneState. This establishes improvement is
possible through current inputs/clicks on that cohort, not that the present GRU
can already implement it or that near-perfect CS needs no projectile information.
E69/E70 establish two locally recoverable attacks without missing damage/CS in
those traces; they do not rule out other simulator faults.

Recommended discriminating next step, discussion only: adapt the existing
scripted-teacher BC/DAgger path to this exact 120s AFK task and GRU/action contract,
evaluate actual closed-loop CS on held-out starts, then test retention under
unchanged CS-only PPO. Labels should also cover states visited by the clone;
rare attack/wait decisions matter more than aggregate imitation accuracy. A
successful clone followed by PPO regression points toward update/credit problems;
failure to clone is ambiguous until labels, optimization and a smaller randomized
last-hit task are checked. Preserve/recompute recurrent observation histories
when comparing timing decisions; identical single frames do not imply identical
information for a recurrent policy. No human-play interface is needed.

If a local task becomes necessary: randomize HP, range and attack timing in short
last-hit situations, retain original controls and history, then validate transfer
to full waves. A restricted attack/wait action diagnostic can isolate targeting
from timing, but is not a deployable policy result. Add observations only when a
matched diagnostic supports an information bottleneck; privileged teacher labels
are distinct from giving hidden state to the deployable actor.

Relevant primary literature, analogies rather than diagnoses:
- DAgger, https://arxiv.org/abs/1011.0686: a learner's own actions change which
  states it encounters; training only on expert trajectories can compound errors.
  This matches the older repository BC/DAgger history, on a different task.
- Reverse Curriculum Generation, https://arxiv.org/abs/1707.05300: learn difficult
  manipulation from starts near success, then expand the start distribution.
  Analogous proposed use is short timing tasks before whole-wave farming.
- No Representation, No Trust, https://arxiv.org/abs/2405.00662: PPO on Atari and
  MuJoCo can lose representational capacity and performance despite critic fit.
  We have not measured that mechanism here; ordinary policy-gradient interference
  is distinct from established representation collapse. No PFO/reset implemented.

Strategic distinction: at10Hz gamma.99 gives a6.90s reward half-life (a CS30s
away has weight about.049). This makes delayed pushing/proxy benefits less
valuable than immediate CS; short120s starts and pure-CS reward also change the
objective. It does not prove why seconds-scale last hits are missed, and perfect
CS does not necessarily imply maximal tower damage. E80 tests longer training
within the existing task, not longer in-game planning or discovery of every strategy.

### LEARN-AFK-37 — authorized acquisition, retention and local-skill sequence

Dani explicitly requests running/managing all proposed stages in order, then
clarifies that artificial delays must not be added to the model. E69/E70's
withheld attacks were counterfactual diagnostic interventions only. This sequence
adds no latency or mandatory wait and changes no combat physics. E80 continues.

E81 hypothesis: current v3/GRU/click interface can learn stronger AFK farming
when given explicit labels from the existing observation-only scripted player.
Teacher uses declared static combat knowledge, never LaneState or unit IDs.
Initialize from E78 FINAL, not its better intermediate. Use original architecture
and interfaces with fresh supervised Adam1e-4, gradient clip.5, no entropy loss,
used-coordinate-head cross entropy, weight4 for attack_move labels. Stage0:
32teacher episodes on16train offsets (-120..120 step16), paired HP roles. Stage1
and2: each64sampled learner episodes on the same bank, relabel and aggregate
all previous data.30/20/20epochs, batch8episodes x128steps. Approximate valid
samples depend on terminal timing;1280padded samples/episode imply7,782,400
supervised training samples across stages. Replay the full observation prefix
with CURRENT parameters before each differentiated window; never zero the GRU
at arbitrary window boundaries. Labels/physics retain native10Hz timing.

Frozen64 unchanged held-out offset/HP/seed2007 cohort at initial, BC, DAgger1,
DAgger2 plus the scripted teacher on that SAME cohort. Initial E78-final physical
and reward retention required. Primary endpoint is final DAgger2 CS>=max(11,
.85*teacherCS); no best-stage selection. Training starts are expanded from E78's
four offsets to16 to give32distinct scripted traces; this is an explicit teaching
data choice, not an isolated reward/optimizer comparison against E80. Full-game
CS is the outcome; imitation accuracy alone cannot pass the gate. Scripted
deterministic duplicates in the64eval suite are not64independent teacher trials.

Before export, collect64separate frozen-clone training episodes and fit the
DETACHED value head to discounted finite-episode CS returns for512minibatches,
using a fresh Adam state. Require exact equality of every actor parameter before
and after. This reduces the stale/untrained-critic confound in E82; it does not
guarantee critic accuracy. The final actor's frozen stage3 result is unchanged by
this value-only fit. Export final.msgpack, evaluations.jsonl and handoff.json only
after complete protocol, with SHA and competence flag.2hSlurm/105minworker cap;
45–90min is an unmeasured engineering estimate. Interruptions/nonfinite/contract
failures export no complete handoff. Existing canaries plus history/reset, used
heads, teacher-label execution/PPO refusal, and frozen-actor regression tests are
integrated into the launcher; no separate GPU smoke job or login-node training.

E82 hypothesis: CS-only PPO can preserve a demonstrated farming skill. Use E81
FINAL parameters/calibrated value head, fresh Adam and the unchanged E78 settings
(LR1e-4,entropy.001,gamma.99,4epochs,128envs x128steps,staggered120s,original four
training offsets).512updates/8.389Mdecisions,55minworker/1hSlurm; expect~40min.
Handoff SHA and frozen64initial physical/reward equality required. Fixed frozen
0/128/256/512; final drop>1CS flags retention failure, within1CS is retention at
that tolerance, gain>=1CS is improvement. Report paired episode differences and
all checkpoints without selecting the best. Preserve the original E81 artifact
as the immutable frozen control. If E81 fails competence, still execute this
authorized continuation but do not call it preservation of a demonstrated stronger
skill. Retention change alone does not distinguish critic, gradient interference,
entropy or representation mechanisms. No adaptive LR/reset/anchor loss added.

E83 is authorized/reserved after E82, with a proposed <=1hSlurm cap. Finalize
the implementation against their evidence: short randomized last-hit starts,
original GRU/observations/clicks and native physics, teaching/PPO comparison,
held-out targeting/timing outcomes and full120s transfer. Do not introduce forced
pauses, latency or privileged actor features. Full recurrent history matters;
a failed local learning run is not by itself proof of an information bottleneck.
It is not implemented/submitted yet; exact budget/seed/cohort/success criteria
must be recorded in its own spec before launch.

Primary methodology references already reviewed in LEARN-AFK-36: DAgger
https://arxiv.org/abs/1011.0686 for learner-state relabelling;
https://arxiv.org/abs/1707.05300 for near-goal starts in sparse-reward manipulation.
Neither establishes our hyperparameters or predicts a result. These are
prior-assisted diagnostics, not from-scratch PPO or C# transfer successes.

| LEARN-AFK-37 staging | E81 implementation and E81/E82 JSON specs prepared; Python syntax checks and launcher dry-runs pass. GPU integration tests will run with E81 after E80 releases the GPU. | Only E80/2171 currently submitted/running; its active bridge is the next wake. On wake, review E80 then launch E81 without asking again; repeat existing health watch/event bridge for each successor. No extra timer/service created. E83 remains an authorized subsequent implementation, not a claimed queued job. |

| LEARN-AFK-35 cancellation | Dani requests cancelling E80 and doing the authorized diagnostic sequence first. Slurm2171 CANCELLED/0:0 after58m17s at2026-10-02T01:26:59UTC (Oct1 6:26:59PM PDT). Graceful worker final save update809/13,254,656decisions; study/manifest interrupted; no nonfinite flag in final metric. | Frozen64 u0/u512:CS10.0/11.328125, deaths.078125/.03125, personal towerHP181.035824/10.180336. No u809 frozen evaluation; declared u11520 endpoint and retention gate incomplete. Generic PROFILE COMPLETE footer does not override CANCELLED accounting. Final artifact E80_afk_cs_only_12h/vec-s0-20261002-003708-b7130a5c/ckpt_013254656.msgpack. |
| LEARN-AFK-35 watcher cleanup | lanerl-event-2171.service stopped/inactive; registry entry absent. result.json says stopped_unconfirmed and retains preceding RUNNING poll. | Final state verified independently via sacct; no completion wake expected. E81/E82/E83 remain authorized and are now the immediate priority; do not automatically resume E80. |

| LEARN-AFK-37 E81 startup | After Dani cancels E80 and explicitly prioritizes the diagnostic sequence, E81/2172 launches from f4cdde7 at2026-10-02T01:28:44UTC (Oct1 6:28:44PM PDT). All21 integrated GPU tests pass:16existing in367.72s plus5imitation contracts in63.07s. Launcher reports healthy after180s. | No learning outcome yet. Expected finish7:15–8:00PM PDT,2h hard cutoff8:28:44PM. lanerl-event-2172.service active/registered, result heartbeat advances1790905005.8164654→1790905035.9146721; expiry2026-10-02T05:36:45.323792+00:00 (Oct1 10:36:45PM PDT). E82/E83 remain authorized but unsubmitted; continue on E81 wake. Initial bridge-file read raced service creation; bounded retry verified the advancing heartbeat, no watcher change/restart needed. |

| LEARN-AFK-37 E81 final | Slurm2172 COMPLETED/0:0 in33m06s, ended2026-10-02T02:01:50UTC (Oct1 7:01:50PM PDT). Full30/20/20epochs and512critic minibatches complete. Frozen64 initial/BC/DAgger1/DAgger2:CS10.0/7.28125/10.84375/11.75; deaths.078125/.421875/.15625/.0625; towerHP181.035824/.821375/0/0. Same-cohort scripted teacher13.5CS/0deaths. Final low/fullHP CS12.0625/11.4375. | Final11.75 >= declared max(11,.85*13.5)=11.475; teaching gate passed. Paired final-initial CS+1.75,median+2,40better/11equal/13worse,range−4..+9. Current observations/GRU/clicks support better farming with supervised guidance. Not perfect-CS sufficiency, an independent seed replication, or proof of a sole PPO cause. Continued epochs and learner-state relabelling both change across stages, so their individual effects are not isolated. |
| LEARN-AFK-37 E81 integrity | All4frozen cohorts/bothteams reward=CS and other terms0; initial retention marker passed.21startup tests passed, no traceback, full completion marker. Exported final.msgpack and final ckpt_008306688.msgpack SHA256 both1b8ed30cf913132b8d7045e45f40cd09d9cec205a86a5a3301e99532b08f24e0; exported evaluations byte-identical to run JSONL. Manifest/handoff complete and competent; runtime actor-invariance gate passed after critic fit. | Run /mnt/nfs/checkpoints/lanerl-jax/E81_afk_gru_dagger/gru-dagger-s0-20261002-013626-f4cdde73/.70supervised epoch metric rows finite;8critic logging rows finite, last update512. Critic logged training loss.392978@64→.320626@512 uses different sampled batches; not a held-out fit/EV gate. This reduces a confound but does not certify a good critic. Training attack_exact is a window-averaged ratio including no-attack windows, not a global conditional attack accuracy; do not quote it as one. |
| LEARN-AFK-37 E81 bridge and next stage | Completion delivery accepted; lanerl-event-2172.service inactive, registry entry absent. | E82 already authorized by Dani's sequence instruction, now launch from the verified final E81 handoff. Fixed512updates/8.389Mdecisions of original CS-only PPO; frozen64 initial must match11.75, final>=10.75 is retention at predeclared1CS tolerance, >=12.75 improvement. E83 remains authorized afterward; no E80 restart. |

| LEARN-AFK-37 E82 startup | E82/2174 RUNNING from source9c16cef; started2026-10-02T02:04:45UTC (Oct1 7:04:45PM PDT).17integrated GPU canaries pass (16base+stagger warmup); launcher reports healthy after180s. Uses verified E81final11.75CS actor/value-head handoff, unchanged planned512update CS-only PPO. | No learned result yet. Estimated finish7:45PM PDT;1h hard cutoff8:04:45PM. lanerl-event-2174.service active/registered, heartbeat1790907170.9956532→1790907201.1002336; expiry2026-10-02T05:12:49.306819+00:00 (Oct1 10:12:49PM PDT). On wake review frozen retention and proceed with authorized E83; E80 stays cancelled. |


| LEARN-AFK-37 E82 final | Slurm2174 COMPLETED/0:0 in40m24s, ended2026-10-02T02:45:09UTC (Oct1 7:45:09PM PDT). Frozen64 CS11.75/12.171875/12.953125/14.125 at0/128/256/512; deaths.0625/0/.015625/0; towerHP0 throughout. Final low/fullHP14.21875/14.03125. Paired final-initial:+2.375CS,median3,43better/12equal/9worse,range−3..7. | Passes retention>=10.75 and improvement>=12.75. Exceeds same-cohort scripted teacher13.5 by.625. The taught actor plus fitted critic can improve under this PPO setup; no inevitable-forgetting or no-learning diagnosis. Supports acquisition/distribution hypothesis, does not isolate supervised actor changes from critic fitting, establish seed robustness, perfect CS, pushing optimality or C# transfer. |
| LEARN-AFK-37 E82 integrity | All4frozen cohorts/bothteams reward=CS and other terms0; initial E81 trajectory-retention gate and endpoint canary passed.512metric rows, loss_nonfinite0 throughout;60NaN CS/gold/XP train episode summaries correspond to updates without episode completions, no optimization nonfinites. Sampled postKL median.04263349/max.14198102 (not exact whole-policy KL). Final ckpt_008388608.msgpack and ckpt_latest.msgpack SHA256 d5529d6c4d1362b27f28c2bad539a26ff42c2abe7311c414c3fcf82bca7bc357. | Run /mnt/nfs/checkpoints/lanerl-jax/E82_afk_clone_ppo_retention/vec-s0-20261002-021315-9c16cef5/. Manifest source SHA matches E81 handoff; study complete512, manifest finished512. Bridge delivery accepted; lanerl-event-2174.service inactive and registry entry absent. Next E83 follows Dani's existing sequence authorization, not authority conferred by the wake event. E80 remains cancelled. |

### LEARN-AFK-38 — short farming acquisition and transfer (E83)

E82 improved the taught policy to14.125CS; the previous policy's10CS plateau
is not evidence the network cannot learn. Hypothesis: useful local farming
behavior is hard to discover under the old full-wave distribution. Concentrated
opportunities may help PPO acquire it without labels. Contrast with supervised
teaching from the SAME E78final actor/critic; do not initialize PPO from the
successful E82 and call that independent acquisition.

E83_afk_local_skill: unmodified simulator and10Hz controls, original v3/GRU512,
all spells remain available. Fresh level3 starts on the native top-lane midpoint;
1 or3 enemy minions with matching allied competition, normal melee/caster stats,
enemyHP uniform.10–.65 of max, hero initial along-path distance100–300. Train64
seed73; unseen64 seed9073; half each enemy count; rollout evaluation seed2007.
12.8s horizon, next ordinary wave at30s, no imposed delay or cooldown. Every
recurrent history begins at the actual task start; no stale/zeroed midgame carry.
These are synthetic initial conditions, not an assertion of natural-state
frequency. All actor features still come through the original observation builder.

Independent fresh-optimizer arms from immutable E78final:
(1)64scripted episodes,40epochs;64own-policy episodes relabelled and aggregated,
20epochs,8episodes x128windows, attack-label weight4, LR1e-4/clip.5 and current
parameter full-prefix replay. (2)256standard CS-only PPO updates,128envs x128steps
=4,194,304decisions,4epochs,LR1e-4 constant,entropy.001,gamma.99. No adaptive LR,
extra feature, forced wait, or reset of production model weights. BC and PPO
budgets differ: this is an exploratory feasibility study, not equal-compute ranking.
Short reset timing and distribution both change; no isolated sparse-credit claim.

Frozen initial, scripted comparator and both fixed final short cohorts. Primary
score is CS/initial enemy count separately in1 and3 enemy strata; useful local
improvement requires+.15 absolute fraction in BOTH versus initialization.
Teaching feasibility requires>=.85 of teacher fraction in BOTH, interpretable
only if teacher>=.5 each. Original64x120s suite at initial (exact E78 retention)
and both learned endpoints: transfer improvement>=11CS versus initial10;
E82's14.125 remains the established full-task baseline. No best-checkpoint pick
and no automatic promotion for short-task scores alone. Log direct decoded
same-target/different-target/no-attack commands when script requests an attack
and native AA clocks are ready; also attacks while script positions. MOVE can
decode into ATTACK, so button labels alone are insufficient. These are script
agreement counts, NOT optimality, actual swing counts or causal lost-CS counts;
spells and existing held orders can be valid alternatives. Diagnostic state never
enters actor inputs. Diagnostic collection must reproduce frozen per-game CS.

Worker50min/Slurm1h bound; expected30–50min including startup (engineering
estimate). Stop on integration/retention/nonfinite failure, signal or time bound;
both fixed arms required for completion. Integrated tests cover fresh reset
clocks/health/count/split, recurrent imitation contracts, decoded target categories,
identical physical transitions and terminal CS for diagnostic vs normal rollout.
Reference rationale remains DAgger https://arxiv.org/abs/1011.0686 and near-goal
curricula https://arxiv.org/abs/1707.05300 already reviewed in LEARN-AFK-36/37;
neither validates this synthetic geometry, budget or a LoL performance prediction.

Interpretation guard: if a short-task stratum already starts above.85 collected
fraction, the+.15 gain gate has insufficient ceiling; report existing local
competence/headroom, not an inability to learn. Command diagnostics count decision
frames, not independent opportunities or guaranteed missed kills. If spells solve
the short task, that is legitimate farming performance but not proof of precise
AA timing; existing frozen spell-selection telemetry remains available.

| LEARN-AFK-38 startup | E83/2176 RUNNING from source f5b8677, started2026-10-02T02:58:48UTC (Oct1 7:58:48PM PDT). All24integrated GPU tests passed:16base in364.45s plus8imitation/local contracts in198.51s. Launcher reports healthy after180s. | No learned result yet. Expected finish8:30–8:50PM PDT,1h hard cutoff8:58:48PM. Bridge lanerl-event-2176.service active/registered, checked_at advances1790910523.3273835→1790910613.6237524; expiry2026-10-02T06:08:42.562201+00:00. Review both fixed arms, transfer and diagnostic limitations on completion wake; verify cleanup. E80 remains cancelled. |


| LEARN-AFK-38 final | E83/2176 COMPLETED0:0 in30m56s, ended2026-10-02T03:29:44UTC (Oct1 8:29:44PM PDT). Both40+20supervised epochs and256PPO updates completed. Frozen32games per short stratum: initial one/three enemy CS fractions.75/.583333; script0/.96875; teaching.50/.75; PPO1.0/.791667. Raw short aggregate CS1.25/1.453125/1.375/1.6875 respectively. | PPO passes declared local+.15 fraction improvement in both strata (+.25/+.208333); teaching fails. Scripted single-minion0 fails the predeclared>=.5 control prerequisite, so the global teaching-feasibility comparison is not interpretable as a neural learning-capacity test. Three-minion teaching.75 is also below.85×.96875=.8234375. No promotion from local scores. |
| LEARN-AFK-38 full-task transfer | Frozen64x120s initial/teaching/PPO CS10.0/4.875/1.03125, deaths.078125/.953125/.015625, towerHP181.035824/0/0. Teaching pairedCS−5.125,4better/0equal/60worse; PPO−8.96875,0better/0equal/64worse. | Both fail declared>=11CS transfer. Local PPO learns useful behavior on the synthetic distribution, but this curriculum severely regresses normal farming. Supports a task-distribution/retention problem in THIS short-only setup, not a diagnosis of why the earlier full-wave policy plateaued, catastrophic forgetting mechanism, insufficient information, or universal PPO failure. E82's14.125CS remains the strongest full-task endpoint in this sequence. |
| LEARN-AFK-38 script limitation | Saved TRAIN round0 first episodes: single-minion teacher0CS, all4128valid labels are MOVE.29/32single-minion starts have1350visible killable observation frames under the script's HP-upper-bound/AD calculation; minimum distance162.2955, none within its conservative155-unit gate,589within native165-unit reach. Three-minion teacher training84CS/32episodes. | The script never requests an AA in these single-minion first episodes because killable targets stay outside its conservative range gate. Training-observation evidence, not an exact replay of held-out failures or proof that collision versus click quantization caused the spacing. The synthetic setup failed to establish a competent teacher in both strata before fitting; future teaching protocols should enforce that existing prerequisite before training. Do not retrofit the teacher or relabel this completed experiment a success. |
| LEARN-AFK-38 command diagnostics | Short frozen initial/teaching/PPO ready-script decision frames109/77/59; same-target direct ATTACK0/47/0; different-target ATTACK0/2/0; no direct ATTACK109/28/59; direct ATTACK while script positions7/345/5. Runtime diagnostic collection reproduces each frozen short-game CS exactly. Frozen short spell selections44.03125/.015625/53.265625; full120s458.0625/3.71875/727.8125. | Teaching increases agreement with the scripted command choices while full-task performance worsens. PPO's short gain is not evidence of matching the script's AA timing. Counts are decision frames, not independent opportunities, executed swings, successful casts, damage attribution or causal missed-CS events; spells, held orders and attack-move fallback remain legal alternatives. |
| LEARN-AFK-38 integrity | All7frozen cohorts/bothteams satisfy reward=CS, other terms0; original E78 full initial-retention gate passed.24integrated GPU tests, no traceback, PROFILE COMPLETE, study/manifest complete with both arms.60finite teaching epoch rows,256PPO updates with loss_nonfinite0; two NaN CS/gold/XP train summaries correspond to no completed episodes. SampledpostKL median.02557453699/max.04291269183.129valid first-episode decisions padded to256; each collected dataset8256valid observations;1,310,720padded supervised samples and4,194,304PPO decisions. | Run /mnt/nfs/checkpoints/lanerl-jax/E83_afk_local_skill/local-s0-20261002-030844-f5b8677c/. Source E78final SHA cd136aa512550c22c8df6e0a7ea58d9ebea3ae520bf0478b5331ea5b7a69be91 verified. teaching_final.msgpack SHA256 0e8b521a75ce8b5211088aa6f87da4ffb0d8f64060d57eaf0d3e7d7dca0d3b93. ppo_final.msgpack, ckpt_latest and ckpt_005505024.msgpack SHA256 all811fe2a8649a6123844f6469d3ad226ea751011c679256aa0dabc4c4adb4a147. |
| LEARN-AFK-38 cleanup and disposition | Bridge2176 delivered accepted, service inactive and registry entry absent. E81–E83 authorized sequence complete; no RL job running/queued. E80 remains cancelled. | Retain E82, do not promote E83. Recommendation only: evaluate E82 on broader held-out starts/seeds, then choose bounded full-wave continuation/replication; preserve normal-wave exposure and full-task retention gates. No new experiment submitted by this completion review, no automatic12h restart and no feature port. |


| LEARN-AFK-38 interpretation discussion | Dani questions whether imperfect CS after substantial experience indicates a bug or severe sample inefficiency. E82's8,388,608decisions at10Hz equal~233simulated hours for+2.375frozenCS; many frames repeat similar situations, so this is not233hours of distinct examples or a formal efficiency comparison. Current code uses independent button/96x/54y heads, used-coordinate likelihood, on-policy PPO, gamma.99/lambda.95 and detached critic features. | E81/E82 establish some learning, not efficient acquisition or near-perfect-CS sufficiency. Plausible failures include searching an unnecessarily difficult click/action space, credit or shared-parameter update interference, and insufficient information in the available history. No new bug found by this read. E69/E70 physical alternatives and E54/E55 reward-time likelihood audits only cover parts of the chain. Discussion recommendation: a bounded actual-wave decision audit connecting known useful sequences to their sampling probability, credit and post-update probability, with true recurrent prefixes, before a long extension. Not an implemented/spec'd/submitted experiment; retain E82, no E80 restart. |


### LEARN-AFK-39 — user-requested Sol audit and cheaper action-search diagnostics

Dani requests a Sol bug-hunting subagent and proposes20circular movement choices
plus an optional visible target, then explicitly asks for cheaper tests before
a12h training run. Sol read-only audit E84_sol_bug_audit launched against isolated
c6ecdd9 snapshot /mnt/nfs/projects/_archive/ahriuwu-sol-learning-audit-20261002.
No model/action-interface change or Slurm experiment has been submitted here.
Archive manifest records retention; only the three original project workstreams
remain active. Final audit findings follow below when available.

Proposed bounded protocol, still discussion rather than a launch spec:
1. Build a shared test bank from real E82 full-wave episodes with complete GRU
   histories. Compare legal physical move/attack/hold/ability choices and short
   sequences under paired continuations. Rank by actual total CS, not target-only
   kills, and validate selected alternatives on separate continuation seeds.
   Check whether20directions+visible-target choices cover useful alternatives.
   Sum original-policy probability over equivalent useful actions; include ground
   attack-move auto-acquisition. Low centre-click mass alone is insufficient.
2. On the same cases, compare empirical action-return differences against learned
   values/credit. Separate one-step action comparisons from forced multi-action
   options. Forced forks provide diagnostic labels only; do not feed them to PPO
   as on-policy samples. Sampling uncertainty and fixed continuation matter.
3. Use ordinary on-policy data and isolated optimizer/parameter copies to measure
   the change in useful-action probability after a normal update; recompute full
   prefixes for the recurrent comparison. If needed, fit a small verified decision
   set with supervised labels to distinguish fitting from search/credit. Individual
   example regressions under a mixed batch are not by themselves a learner bug.
4. If warranted, compare held-out prediction of verified action outcomes from
   current observation histories versus histories plus diagnostic own-attack and
   projectile state. Split by episode; provide privileged fields only to the
   diagnostic model. A gap motivates feature/representation work, not a proof
   that a different recurrent encoder could not infer that information.

First bundle intended60–90min Slurm cap (engineering estimate; not launched),
with reusable canaries and completion bridge. Exact case counts, candidate set,
seeds, comparisons and gates must be fixed in a new JSON before submission.
Relevant evidence already exists: E69/E70 physical recoveries; E54/E55 numerical
credit/update contracts; LEARN-AFK-19 direct-click mass and E66proposal trial.
E66 final9.34375CS versus original9.5625 did not solve farming; its earlier reward,
initialization and mixture design differ from the proposed categorical menu.
Do not repeat low click-probability findings as proof of a root cause.


| LEARN-AFK-39 Sol source audit | User-requested GPT-6-Sol read-only review E84_sol_bug_audit completed at isolated c6ecdd9 snapshot. No new confirmed live correctness defect. Inspected CS-delta reward, post-action done/carry reset, agent-major recurrent sequence layout, GAE and matching sampled/learned likelihood paths: vec_train.py:266–368, learner.py:42–90, ppo.py:65–145. No code changes, runtime tests or Slurm jobs by the agent. | Static review narrows hypotheses; it does not certify the whole system or replace a targeted reproducer. Existing E54/E55 numerical contracts make another generic GAE/likelihood check low value. Audit snapshot retained under _archive and logged in MANIFEST.tsv; no new active project workstream. |
| LEARN-AFK-39 clipping coupling | E82 enables detach_critic (wave_scenario_train.py:136), and policy.py:323–325 stops value gradients at shared features. However learner.py:36–39 globally clips the combined actor and value-head gradient before Adam. Thus value-head gradient magnitude changes the common scaling applied to actor gradients. | This confirms gradient coupling, NOT smaller actual Adam steps or a root cause. Adam can largely cancel uniform scaling, especially with fresh moments. Proposed matched same-batch/same-optimizer diagnostic: retain versus exclude value-head gradient contribution from the clipping norm; report actor parameter delta, full-prefix policy KL and useful-action probability, not only gradient norms. Isolated diagnostic branch, not a production optimizer change. |
| LEARN-AFK-39 action/information hypotheses | Sol confirms known absence of previous sampled action and own AA phase in E82's v3 actor inputs; independent button/x/y heads and unconditional coordinate entropy are existing design choices. Prior E59/E66 sparse direct-click/proposal results remain relevant, including auto-acquisition and failed farming-gain caveats. | No claim that the20-direction menu solves the problem or that missing features impose a demonstrated perfect-CS ceiling. User's12h proposal remains deferred pending cheaper diagnostic discussion; no model/action-port implementation or training submission. First proposed bundle measures actual alternatives and original/compact-menu coverage, credit accuracy and update response; held-out information comparison is a later discriminator if necessary. |
