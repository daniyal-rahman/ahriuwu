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
| MODERN-001 | 2026-09-30 planning audit: data/patch.py provides an intended data seam, but sim/config.py constructs Map1 terrain/vision/routes; init.py includes measured legacy rune/mastery/stat deltas and old map coordinates; waves.py documents the legacy ~36.4s wave cycle; spells.py and step.py deliberately preserve ineffective W damage reduction; state.py fixes two champions/40 minions/24 turrets and profiles.py uses 19 level rows. CODEMAP's paused-JAX labels lag later STATUS entries. | RESEARCH/PLAN ONLY. Preserve legacy behavior under an explicit legacy ruleset; modern port needs behavioral, geometry, state and interface changes, not only stat replacement. Dani clarified: champion kits belong to other agents; this thread owns map, towers, minions and remaining non-champion systems. Lane-first delivery is a milestone, not a scope restriction. |
| MODERN-002 | Target proposed from [Riot patch26.19 notes](https://www.leagueoflegends.com/en-us/news/game-updates/league-of-legends-patch-26-19-notes/): normal PC Summoner's Rift. The same notes contain separate Classic/Arena changes; those must not enter this profile. Current-patch top quest free Teleport cooldown changes420→390s and upgraded Unleashed cooldown330–240→300–210s. [Riot Data Dragon documentation](https://developer.riotgames.com/docs/lol) distinguishes static data versions from regional client versions. | Phase1: immutable manifest with patch, mode, region/build, asset versions/hashes, retrieval date, map identity, supported loadouts, and per-field source/confidence. Audit all relevant intervening patches/hotfixes; reject missing required values instead of inheriting legacy defaults. Exact client/data versions not yet resolved. |
| MODERN-003 | [Riot26.1](https://www.leagueoflegends.com/en-us/news/game-updates/patch-26-1-notes/) changed lane wave speed/cadence/economy, turret plates/durability, Crystalline Overgrowth, Homeguard and role quests, including top level cap20. These are season-start evidence, not a claim all values remain unchanged in26.19. [Riot26.10](https://www.leagueoflegends.com/en-us/news/game-updates/league-of-legends-patch-26-10-notes/) removes attacking an allied minion as a trigger for adding the enemy champion to minion aggro priority. [26.11](https://www.leagueoflegends.com/en-us/news/game-updates/league-of-legends-patch-26-11-notes/) explicitly discusses the resulting ranged-minion behavior. | Phase2–3: introduce separate PatchData, Ruleset, MapSpec and scenario/loadout inputs at SimConfig; select rule implementations outside tick JIT, retain pure array state and scan/vmap. Replace modern geometry/brush/routes together, then spawn schedule, targeting/aggro, XP/gold, turrets/plates/overgrowth, recall/death/fountain/Homeguard and role quest. Test level/capacity bounds explicitly. Modern map source and measured targeting/timing remain open gates. |
| MODERN-004 | [Riot26.5](https://www.leagueoflegends.com/en-us/news/game-updates/league-of-legends-patch-26-5-notes/) changes Garen Q movement duration and E AD scaling. [Riot26.14](https://www.leagueoflegends.com/en-us/news/game-updates/league-of-legends-patch-26-14-notes/) changes R base true damage150/250/350→125/200/275, retaining25/30/35% missing-health scaling. These patches demonstrate that remembered modern values are insufficient. | CHAMPION WORK EXCLUDED per Dani; sources retained as interface context only. Champion agents own kit formulas/state machines. This thread defines shared damage/mitigation/shield/status/event, collision/pathing, visibility and economy interfaces, with explicit source/target/type/tags and ordered timing. Shared loadout/item/rune/summoner/recall systems belong in the non-champion integration plan; no concurrent champion-file rewrite. |
| MODERN-005 | Existing observations, action decoding, resets, near-wave banks, replay renderers, policy schemas and checkpoints depend on legacy state/map assumptions. Reward shaping in E46 encodes legacy tower/health valuation; copying weights or shaping would not establish a modern baseline. | Phase5: version observation/action/state schemas; expose supported modern actions and observable buffs, shields, quest, inventory and summoner state without enemy hidden-state leakage. Rebuild scenario banks and routing by profile hash; reject incompatible resume, allow explicit weight migration only. Separate game gold/XP/events from training shaping; preserve existing PPO while evaluating simulator changes. |
| MODERN-006 | C# server is a legacy regression oracle only. Riot's documented Live Client Data API exposes player/game information, not a complete server simulation trace. Modern timing, collision, geometry and event-order claims need suitable patch-tagged client observations/recordings plus documented/data-derived fixtures. No modern client captures have been acquired. | Phase6 fidelity gate: legacy regression unchanged; independent modern arithmetic/boundary fixtures; controlled generic attack/cancel and champion integration contracts, minion aggro/brush, wave collision, tower/plate, death/recall, quest and loadout scenarios on both sides. Exact checks for discrete events and counts; tolerances declared from measurement resolution for movement/time. Long frozen10min tests report CS/XP/gold/deaths/tower progress and uncertainty across starts; CS alone cannot validate mechanics. Unresolved material rules prevent a fidelity-complete claim. |
| MODERN-007 | Execution sequence: evidence/profile and champion interface contract; behavior-preserving legacy extraction; modern terrain/vision; minions/structures; economy/quests/shared systems; jungle/objectives; interfaces and fidelity/performance integration. Each is a reviewable commit series with focused tests before the next dependency. Champion additions belong to other agents. Full-map world support requires configurable static capacities, all three lane paths, neutral monsters, wards, inhibitors/Nexus, objectives and terrain variants; initial lane milestone must not be presented as a complete modern world. | Proposed compute budget after plan review: capped CPU unit checks; separately registered Slurm correctness canary≤30min; fixed-N128/T128 performance comparison≤30min, measuring compile, collect, complete PPO update and peak memory with synchronized repeats. Budgets are estimates, split/increase explicitly if needed; no runs launched/IDs allocated now. Investigate >20% full-update regression as an engineering review trigger, never remove required mechanics to meet it. Any later learning experiment needs its own hypothesis/budget/frozen baseline/success criteria. |
| OPS-047 | User-authorized idle wake test: Slurm1788 array on danilogin (success, compiler exit1, application exit7, Slurm timeout). Existing Jarvis T3 client targets only thread6638ec1e-0a67-4ea7-bea9-5ecfd4b3d47c; one deterministic command ID, preserves model/modes, no credential copies. Bounded systemd watcher waits for terminal accounting and an idle conversation. `/mnt/nfs/shared/OPS047_slurm_events/{launch,result}.json`, task logs `/mnt/nfs/shared/OPS047-1788_*.out`. | PASS: canary/180s watch; all four expected terminal states and logs verified. Bridge observed ready/no active turn at22:11:44UTC, accepted at22:11:46UTC, and automated message started a fresh turn after prior final. Watcher inactive, registry removed. Timeout parent reports0:0 despite TIMEOUT (batch0:15): classify State plus ExitCode. One combined notification tested; not four independent wakeups, cancellations/OOM/node failure, or restart recovery. |

| OPS-048 | Reusable `ops/slurm_event_bridge.py` derives current T3 thread from Codex session; validates owned single job and exact name, bounded/registered/capped watcher, State+ExitCode accounting, idle-only deterministic command ID, retry on transient errors, SIGTERM/finally cleanup. Terminal classification checks passed (TIMEOUT0, cancellation, RUNNING, absent); existing T3 read canary passed. E46 job1787 armed as `lanerl-event-1787.service`, expiry 2026-10-01 01:14 UTC, live accounting verified RUNNING. | Uses OPS047-proven delivery path; E46 terminal delivery verified22:27:17UTC after COMPLETED0:0; watcher inactive/registry removed. AGENTS rule10 applies going forward; no training changes. |

| LEARN-AFK-01 | E46 Slurm1787 finished exit0 in42m51s,612updates/10,027,008 learner decisions from E40u5795. Frozen120s AFK evals64games/checkpoint (32 per initial HP role), one training seed. Combined CS at u0/64/128/306/612:5.453125/8.53125/8.640625/8.546875/9.5625; deaths:.9375/.015625/0/0/0. LowHP final9.1875CS vs4.9375 initial; fullHP9.9375 vs5.96875. Final mean endHP99.02%/98.28%; final0deaths both roles. | Positive AFK farming/survival learning, most early gains by1.05M decisions; not active-opponent/10-minute transfer or isolated causal effect of HP versus tower shaping. Frozen summaries do not separately report tower damage; training tower reward is not frozen evidence. |
| LEARN-AFK-01 validation | `/mnt/nfs/checkpoints/lanerl-jax/E46_afk_farm/vec-s0-20260930-214903-32499e76/`: all5 frozen evals and612 metrics; loss_nonfinite0, median3.220s/update. Study/manifest complete; final `ckpt_010027008.msgpack` step10027008,201 numeric arrays finite,latest byte-identical; SHA256680889327db7db60503dd0f3b7a0926886f32f2f3ae69eb38b79d3c34265570d. | Event delivery accepted and fresh turn observed; watcher inactive/registry removed. No further experiment launched. |

| REPLAY-AFK-01 | E49 Slurm1800 exit0; final E46u612 checkpoint SHA680889327db7db60503dd0f3b7a0926886f32f2f3ae69eb38b79d3c34265570d. Full120s seed7 offset−45, blue fullHP, red parked fountain with NOOP actions. Blue7CS/0deaths, red0/0. Full-map and animated combat MP4s120.1s at1x; `/mnt/nfs/shared/E49_afk_final_video/low_team_1/`. Nine startup canaries/180s watch passed. | Single diagnostic game, not frozen64-game aggregate. Event delivered; watcher inactive and registry removed. |

| LEARN-AFK-02 | E46 curve generated by `ops/figures/afk_training_curve.py`, PNG in docs/figures. Five frozen points only; lines interpolate, one training seed. LR3e-5 annealed per update toward0. Train u1–64 mean approxKL.00759, clip.0981; u307–612 mean gold reward.00547, XP.00758, tower.00924, health−.000893 per learner decision. Source `_relative_reward` credits towerHP loss regardless of damage source and XP independently of last hits. | Quick initial gains then slower progress; not proof LR is limiting. Minion tower damage and proximity XP permit reward without personal tower hits/CS. Proposed matched3e-5 vs1e-4 constant LR from E46 final, fixed reward/budget2M decisions, frozen eval comparison; no run authorized/launched here. PPO reference discussion: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/ . |

| LEARN-AFK-03 | User-authorized E50/E51 cleanup: xp_scale0, personal Garen tower shaping, unchanged HP penalty. Telemetry accumulates champion-source effective turret HP removal in existing hit order, capped at remaining HP; minions/overkill/dead targets/other victims excluded. No damage/kill/observation dynamics changed. Old checkpoint parameters load; old full rollout-state schema not migrated. Five CPU tests passed including actual Garen tower hit and telemetry-independent HP; GPU canaries and180s watch passed on E50 job1804. Frozen eval adds personal tower damage. | ConstantLR3e-5 vs1e-4, E46final init/fresh Adam,128updates/2.097M each;64 frozen games at0/64/128. Directional criterion>=.5CS improvement over control,<=.05 extra deaths/game; bounded/nonfinite stop. No separate reward ablation per user. PPO LR rationale in LEARN-AFK-02. |

| LEARN-AFK-04 | E50 job1804 complete exit0,15m14s;128updates/2,097,152 decisions,median3.212s/update. Frozen64games at u0/64/128: CS9.5625/9.453125/8.03125; deaths0/.03125/.171875; personal towerHP0/7.4876/20.1505; shaped return6.3293/6.0193/3.0785 under SAME cleaned reward. Final lowHP8.6875CS/.03125deaths; fullHP7.375CS/.3125deaths. Checkpoint/latest identical,all numeric arrays finite,128 metric rows,zero loss_nonfinite. `/mnt/nfs/checkpoints/lanerl-jax/E50_afk_personal_lr3e5/vec-s0-20260930-230245-a12b1fbe/`. | LowerLR arm regressed; reward cleanup not automatically a learning improvement. E51 submitted1805 from SAME E46final, not E50, to finish authorized comparison. E50 bridge cleanup verified. |

| MODERN-008 | Revised world scope: [Riot26.1](https://www.leagueoflegends.com/en-us/news/game-updates/patch-26-1-notes/) documents Faelights, changed wards/Homeguard, removed Atakhan, and revised epic objectives. Existing vision grid, ray kernel and routing are Map1-derived; a visual map replacement cannot establish modern walkability or brush/vision semantics. | Plan includes provenance-checked modern terrain, collision/navigation/height/brush layers, spawn/structure/camp coordinates, all lane splines, plants, wards/trinkets/reveal/Faelights, camp spawn/leash/reset/rewards, dragons/terrain variants, Grubs/Herald/Baron/Elder and their team/minion effects, inhibitors/super waves/Nexus/end state. Audit changes through26.19 before locking values. Source availability for modern geometry and live-client measurements remains the largest fidelity dependency. |


| ID | Evidence / finding | Decision / remaining work |
|---|---|---|
| MODERN-009 | Dani authorized implementation and prioritized static top lane/alcove, wave spawning, minion aggro, turrets/plates/Crystalline Overgrowth, and shared health/resistance/item/rune modifier ordering. Champion kits are owned by other worktrees. | Execute those systems in that order, with focused tests and evidence per slice. DEFERRED TODO: elemental terrain transformations, dynamic terrain overlays (including destroyed-structure collision), full modern vision/Faelights/wards/reveal, jungle spawn/leash/reset/objective logic and jungler-specific items. Preserve raw map brush/height/region data now. Item active TODOs except Tiamat and Stridebreaker, which stay in scope. These exclusions prevent a full modern-world fidelity claim. |
| MODERN-010 | Extracted real EUW1 client build16.19.8230722 (public patch26.19) directly from [Riot release manifest](https://lol.dyn.riotcdn.net/channels/public/releases/4D2A50D5EDAB724A.manifest), discovered via [manifest index](https://github.com/Morilli/riot-manifests/blob/master/LoL/EUW1/windows/lol-game-client/16.19.8230722.txt). RMAN SHA256 f7b09f71ae917190df3cb7e7b254724e3fb485417a02f1d20b19f6866adea73c. Selective Map11 WAD extraction acquired navigation, object CFG and map11.bin without installing League or mounting Windows. Default, SR_Seasonal_Map and Worlds_SR_Seasonal MapSkin records all reference AIPath_SRX_2.aimesh_ngrid (decoded with CDTB1.3.0; map11.bin SHA45d16148616eb3da612e31f422afb46d756d5fc615f8387b7429343bda5da88a). | TOOL ops/fetch_modern_map.py; pinned profile lanerl_jax/data/modern/26.19/map11.json. Extraction receipts/raw assets retained in /mnt/nfs/shared/modern-world-map-research/live-16.19.8230722/. Stable normalized artifact /mnt/nfs/datasets/league/26.19/map11-base/. No client installation, vendor edit, training job or service created. Config references establish the base asset, not runtime overlay/route fidelity. |
| MODERN-011 | Current navigation SHA b7551b91dcdc3dec0228ff2df63483399b0b040552d2115b65d9fe1faf35bd93, v7.1, 296z×295x cells at50 units, translated X/Z origin; 53724 conservative open cells,2129 brush-flag cells. Top alcove present; main-region label11. Reader layout independently based on [NGRID converter revision92943ed](https://github.com/FrankTheBoxMonster/LoL-NGRID-converter/tree/92943ed2b2d5e82c86d680e69f53f247c89aefee). A historical2024 reference was used only for reader comparison, never as26.19 terrain. | Implemented immutable NumPy layers, strict bounds, team gates, explicit disk-contact contract, float32 JIT/vmap queries, version/patch/variant/checksum rejection and a reviewed external manifest pin. Capped CPU test_modern_map.py:13 passed in2.33s after final loader addition. Real pinned artifact also loaded successfully. Exact client pathfinding/collision-radius treatment remains unmeasured. Radius0 uses a half-open cell; positive-radius wall contact is conservatively blocked. Static-grid TOOL only, not integrated into SimConfig or production movement/vision. |
| MODERN-012 | Wiki current-history research: [Minion oldid4068797](https://wiki.leagueoflegends.com/en-us/Minion?oldid=4068797), [Turret oldid4070072](https://wiki.leagueoflegends.com/en-us/Turret?oldid=4070072), [Armor oldid4070944](https://wiki.leagueoflegends.com/en-us/Armor?oldid=4070944). Intervening patches matter: [26.3](https://www.leagueoflegends.com/en-us/news/game-updates/league-of-legends-patch-26-3-notes/) revised Bulwark20–100→30–50; [26.10](https://www.leagueoflegends.com/en-us/news/game-updates/league-of-legends-patch-26-10-notes/) removed champion-attacking-minion aggro priority. Client BarracksConfig unit spacing0.8s differs from wiki0.792s. Wiki flat armor-reduction allocation descriptions conflict. | RESEARCH, not implemented or all independently measured. Resolve mode-specific wave schedule and engine cadence before pinning fixtures; resolve resistance-pool allocation before implementing ordered shared modifiers. Do not blindly inherit either legacy C# behavior or contradictory wiki formulas. Pending map integration: authoritative spawn/structure coordinates and lane routes, collision/pathfinding interface, map-specific resets/observations. These are still in scope, not deferred TODOs. |

| CHAMP-001 scope | 2026-09-30 source inventory of `/srv/nfs/projects/lanerl-vendor/LoLServer/Content/LeagueSandbox-Scripts/{Characters,Buffs}`; active package selected by `lanerl/cfg/garen1v1.json:134`, with LeagueSandbox-Default dependency. Compared with ranks 1–20 in `docs/ROADMAP_CHAMPIONS.md`, Dani's saved 2026-09-23 all-elo list, not a refreshed popularity ranking. Vendor README targets client 4.20. | Read-only source audit; no builds, server smoke tests, ports, training or vendor edits. Spell-file coverage is NOT kit completeness/correctness or modern-patch fidelity. |
| CHAMP-001 overlap | 15/20 have champion spell source; 13/20 have Q.cs, W.cs, E.cs and R.cs: Darius, Garen, Nasus, Malphite, Mordekaiser, Aatrox, Jax, Renekton, Gangplank, Irelia, Tryndamere, Volibear, Fiora. | Includes existing Garen: 12 additional four-slot source candidates. |
| CHAMP-001 partial | Teemo (#7): Q/BlindingDart and basic attacks; no W/E/R scripts. Its CharScript adds range on level-up and resets spell slot1 on kills, not a trustworthy Teemo passive. Yasuo (#13): Q1/Q2/Q3 and E; no W/R scripts. | Incomplete source, not ready full kits. |
| CHAMP-001 absent | Yone (#1), Sett (#5), Jayce (#12), Yorick (#17), Illaoi (#19): no champion script directory or matching champion-path C# files found anywhere in vendor checkout. | Assets/stat JSON alone do not count as executable implementation. |
| CHAMP-001 old kits | Aatrox: health-cost Q/E and W healing/damage toggle; Mordekaiser: ChildrenOfTheGrave + pet scripts; Gangplank: RaiseMorale E; Irelia: HitenStyle W, EquilibriumStrike E, TranscendentBlades R; Volibear: Q flip and R chain lightning; Fiora: FioraDance/Blade Waltz R. See each champion's Characters directory. | Six four-slot candidates are substantially pre-rework implementations; reusable mechanics/reference, not modern kits. Other candidates also need a modern mechanics audit. |
| CHAMP-001 gaps | Aatrox Passive.cs requests AatroxPassive, but no matching class exists in scripts. Tryndamere/Q.cs: CurrentMana is set to zero before fury is read for healing. Darius/Q.cs applies outer-hit DariusHemo twice and contains no heal; actual bleed implementation exists in Buffs/Darius/Counter.cs (do not mistake the separate Passive.cs placeholder for absence of all passive code). No CharScriptNasus/CharScriptMalphite or named SoulEater/GraniteShield script found. | Concrete source gaps/quirks; no runtime validation. Avoid copying defects blindly. Nasus/Malphite passive search is evidence of a gap, not a complete audit of all engine/data behavior. |
| CHAMP-001 next candidates | Darius (#2), Nasus (#4), Malphite (#6), Jax (#10), Renekton (#11), Tryndamere (#16) have four-slot source and recognizable kits; Garen already ported. | Engineering shortlist for deeper audit, not approval to implement or evidence they are bug-free. Jax R is an older version as shown by Jax/R.cs and buffs. |
| CHAMP-001 full-slot inventory | Aatrox, Ahri, Akali, Alistar, Amumu, Anivia, Annie, Ashe, Blitzcrank, Brand, Caitlyn, Cassiopeia, ChoGath, Darius, DrMundo, Draven, Evelynn, Ezreal, Fiddlesticks, Fiora, Fizz, Galio, Gangplank, Garen, Gragas, Hecarim, Irelia, Jax, Jinx, Katarina, Kayle, Khazix, Kogmaw, LeBlanc, Lissandra, Malphite, MasterYi, Mordekaiser, Nasus, Pantheon, Poppy, Renekton, Riven, Ryze, Sion, Talon, Taric, Tristana, Tryndamere, Twitch, Vayne, Volibear, Warwick, XinZhao, Zed. | 55 champion directories with Q/W/E/R files. Filename inventory only; non-overlap kits not individually audited. |
| CHAMP-001 partial inventory | Corki, Graves, Karthus, Kassadin, LeeSin, Leona, Lucian, Lulu, Lux, Nidalee, Olaf, Shaco, Shen, Sivir, Teemo, Yasuo. Additional Diana/ Kalista directories contain nonstandard or passive/basic-attack scaffolding without named Q/W/E/R files. | 16 directories with some named Q/W/E/R files; Yasuo Q variants counted through its E directory presence. |


| ID | Finding / evidence | Status |
|---|---|---|
| CHAMP-002 selection | Select Jax for the next champion implementation based on near-even modern Garen matchup and modest lane gold gap, not minimal historical kit changes. Tryndamere is the least-changed kit candidate by structural assessment; Renekton also retains its core kit. Jax 13.1 changed E damage type/scaling and added R AoE hit, hit-dependent resists and every-second-hit passive while active. Riot: https://www.leagueoflegends.com/en-us/news/game-updates/patch-13-1-notes/ . Tryndamere range changed125→175 in13.17: https://www.leagueoflegends.com/en-us/news/game-updates/patch-13-17-notes/ . | Research/selection only; user requested next steps. No implementation or experiment launched. Least-changed assessment is qualitative, not a counted full patch-history diff. |
| CHAMP-002 modern differences | Current Riot champion pages confirm Nasus R reduces Q cooldown and grants resists; Renekton empowered W destroys shields; Darius passive grants AD at max stacks and deals physical bleed (vendor Counter.cs uses magic); Malphite has Thunderclap W versus vendor Obduracy. URLs `https://www.leagueoflegends.com/en-us/champions/{nasus,renekton,darius,malphite,tryndamere}/`. | Similar spell names do not imply modern mechanics. Historical code also contains implementation defects independent of patch changes. |
| CHAMP-002 matchup provenance | U.GG, top, World, Ranked Solo, Diamond2+, patch26.19 and previous26.18, observed2026-09-30 in T3 browser with visible filters verified. Current https://u.gg/lol/champions/garen/counter?rank=diamond_2_plus ; previous same URL plus `&patch=16_18`. U.GG lists OPPONENT WR and OPPONENT GD15: rows below convert both to GAREN perspective. | Human full-game WR and mean gold difference at15min; not lane win rate, isolated1v1 balance, or predicted simulator WR. Samples small enough for several percentage points of uncertainty. No causal/no-jungle claim. |
| CHAMP-002 Jax | Garen WR/GD15/n:26.19 **50.60%/+315/755**;26.18 **52.90%/+234/1276**. | Selected; smallest absolute current lane gap of shortlist, near-even full-game result. |
| CHAMP-002 Malphite | Garen WR/GD15/n:26.19 **48.91%/+444/366**;26.18 **48.53%/+368/579**. | Alternative; less even lane, missing passive source identified in CHAMP-001. |
| CHAMP-002 Renekton | Garen WR/GD15/n:26.19 **53.06%/−326/458**;26.18 **47.97%/−193/640**. | Modest lane gap but WR flips across patches; do not call stable advantage. |
| CHAMP-002 Darius | Garen WR/GD15/n:26.19 **48.20%/−668/583**;26.18 **47.40%/−639/1000**. | Larger lane disadvantage for Garen. |
| CHAMP-002 Nasus | Garen WR/GD15/n:26.19 **54.36%/+896/631**;26.18 **50.80%/+820/1378**. | Large lane advantage for Garen even when full-game WR is close. |
| CHAMP-002 Tryndamere | Garen WR/GD15/n:26.19 **46.67%/−1038/285**;26.18 **41.35%/−945/445**. | Reject for this balanced-lane selection despite stable kit identity. |
| CHAMP-002 Jax source defects | Vendor Buffs/Jax/JaxCounterStrike.cs has empty TakeDamage callback; no dodge implemented there, GameObjects/Spell/CastTarget.cs explicitly TODO Dodge/Miss. E cooldown formula reads Spells[0] (Q) rank rather than E and multiplies by1+CDR. Characters/Jax/R.cs and its buff implement older R, not13.1 active. | C# is source material, not a validated oracle for modern Jax. Need dodge/on-hit suppression, AoE mitigation, recast/expiry, correct cooldown and modern R tests. |
| CHAMP-002 Garen vendor patches | `lanerl/patches/server-q-cast-freeze.patch`: SERVER-001 fixes orphaned basic-attack cast after Q swaps attack spell; cancellation clears casting pointer/order so retarget/death/interruption cannot freeze champion. Other registered patches: screen-click-v1 (point/hit-test input), screen-click-v2 (attack-move/hostile right-click/visibility), server-hud-ability-state (enabled bits and disabled-key rejection), server-dead-control (authoritative dead flag/live-only input rejection), screen-click-v3 (unwalkable click resolution). | Six registered patches; five are shared engine/control infrastructure, not Garen balance edits. README intro says five despite six entries; build descriptions disagree with AGENTS, so runtime build not recertified here. |
| CHAMP-002 Garen sim fixes | Historical ledger at commit a365a52 and current sim/spells.py, step.py, test_spells.py: SPELL001 stop infinite E-refresh exploit and implement cancel after1s;002 cancel in-flight AA during spin;003 six ticks not seven;004 target-edge radius;005 ghosting buff lookup;006 death no longer gives Q/E free cooldown reset;007 W-adjusted resists for E/R;011 legal rank progression;012/013 exact Q/E expiry, cooldown timing and one-row listed-buff linger. STRUCT001 replaced positional buff lanes with typed lifecycle. Passive minion UnitTag collision/level11 exemption ported (48af3f2); champion stat growth/rune/mastery/regen corrections also landed. | These are historical server-parity fixes, NOT modernization. Source/history inspection only; no tests rerun for this documentation task. |
| CHAMP-002 Garen retained issues | Current spells.py: W active damage multiplier deliberately1 because server subtracts stale pre-listener damage; W passive applies0.96*base+1.2*flat instead of clean bonus, mirrored intentionally. Q/E buffs persist through death including corpse spin. Q active recast/rank0 casts are refused by sim/driver despite permissive historical server. Modern Garen requires separate treatment: current W shielding/tenacity and kill-based resist stacks, modern E rules, true-damage R. Existing docs/MODERN_PATCH_DELTA.md is research, not proof implementation landed. | Preserve old parity profile; do not silently mix modern Jax with old/bug-compatible Garen and claim modern matchup fidelity. |
| CHAMP-002 implementation sequence | Pin modern patch and explicit Garen/Jax rules; audit Jax P/Q/W/E/R source/data against it; introduce per-champion profiles/dispatch, mana, targeted jump and buff state; implement passive AS, Q leap, W reset/on-hit, E dodge/AoE reduction/recast/stun, modern R. Keep existing Garen baseline intact while adding modern profile. Test Garen-Q into Jax-E, Garen-E vs AoE reduction, Garen-R vs resists, silence/recast timing, rank/cooldown, resource and death reset boundaries. Then scripted both-side farming/trades and frozen seeded comparisons before PPO. | Next steps, not executed. Modern human WR guides roster choice, not a target to force by changing champion balance. |


| ID / finding | Evidence / implementation | Status / boundary |
|---|---|---|
| CHAMP-003 contract | User approved Garen+Jax modernization in both engines using this worktree as isolation. Pin Riot26.19 / DataDragon16.19.1; shipped BIN snapshots from `https://raw.communitydragon.org/16.19/game/data/characters/{garen,jax}/{garen,jax}.bin.json` in `lanerl_jax/data/modern_26_19/`. DDragon AD-growth0 is not used; BIN4.5 Garen/4.25 Jax. | Explicit modern lane profile; historical defaults remain unchanged. No balance multipliers fitted to human WR. Old baseline is commit17d845c on this branch. |
| CHAMP-003 Garen | Modern base/growth stats; Q reset,30/rank+1.5AD total hit,1.5s silence,35% haste,slow cleanse,range boost and cast-time CD; W65–145+.18bonusHP shield/.75s tenacity,4s25–41%DR and .2resist/kill cap30; E7+AS ticks,nearest25%,25%armor shred,modern rankCD; R true125/200/275+.25/.30/.35missingHP; passive8s delay/minion exception. BIN breakpoints apply AT7/14, reaching10.1% maxHP/5s at18. | Both engines. Stale damage-local hook fixed; historical W passive bug removed from modern rules. No corpse spin or free death cooldown reset. Attack swapping eliminated from modern Q, so old shared-AA/restore-null paths are no longer used. |
| CHAMP-003 Jax | Base/growth stats and mana; passive8 AS stacks with per-level breakpoints; targeted Q ally/enemy leap and W consumption; W attack reset and magic on-hit; E non-turret AA dodge before on-hit,25% champion AoE reduction,1s recast,magic damage scaling with dodges and maxHP,stun,E-rank CD; R AoE cast,conditional resist duration,third/second-hit magic passive. | Both engines. Old empty dodge callback/wrong cooldown rank and old R replaced. Scripted C# lane exposed a null animation lookup in original basic-attack script; replaced. |
| CHAMP-003 plumbing | `ChampionState`, modern dispatch in orders/tick/status, mana/CC/dash/shield/death timers, modern stats/availability in observations, allied Q cursor target in both decoders. `modern.init_lane(names)` + `SimConfig.modern(names)`; modern self-vector28 includes identity/resources/buffs; `PolicyConfig(self_dim=28)`. C# modern wire exports resources/buffs and per-champion skill order; reset recreates per-owner combat state. | Old16-feature policy weights/collector are NOT migrated. Legacy StateRebuilder explicitly rejects modern C# wire instead of silently treating Jax as Garen. Next integration is modern collector/training profile selection, then frozen seeded comparisons before PPO. No training launched. |
| CHAMP-003 isolation | `ops/modern_server.py` materializes a separate tree at `/mnt/nfs/projects/lanerl-modern-MOD001-v2/LoLServer`; `bin/DeadProbe` rebuilt with vendored.NET6. Sources and exported `server-modern-champions-26.19.patch` checked in; patch dry-run against vendor succeeds. All server runs use worktree snapshots, CPU Slurm and ports27100/27101. | Shared vendor and existing jobs untouched. Incomplete first copy archived with MANIFEST entry. No background service/watcher needed: validation jobs completed during their bounded startup watches. |
| CHAMP-003 validation | MOD001–MOD007 rows in EXPERIMENTS retain failures and fixes. MOD006/1811 passed45 Python checks,16 actual C# combat assertions,wire/reset and180s scripted lane/1887 frames with both champions reaching lane and no ERROR logs. MOD007 adds Garen passive level-boundary and JAX240-tick scan/dtype checks. | Final MOD007 result recorded below after accounting. Scripted lane is a mechanics smoke, not a learning score or balance estimate; scripted controller died/respawned and finished0CS. No frozen-policy performance claim. |
| CHAMP-003 remaining fidelity | Existing map/minions/turrets remain historical; no modern item/rune/ward/jungle/fishing/quest port. Jax leap uses discrete target interpolation at1400u/s; Garen Q lunge displacement and R temporary reveal are not modeled. Champion effects are validated by independent examples, not modern-client packet differential traces. | Do not advertise bit-exact modern League parity. Follow up with targeted modern-client timing/vision traces and implement the remaining spatial/vision details before using this matchup as a fidelity benchmark. |

| CHAMP-003 final validation | MOD007 / Slurm1812 COMPLETED/0:0 in40s:47 Python tests including240-tick JIT scan/vmap,21 C# combat assertions,1887 wire frames including180s lane. Both reached lane; no ERROR log lines. Separate capped CPU legacy-collector rejection test passed (48 distinct Python checks total). Server DLL SHA256 `841115bb6e3e818a62e3f2c9487115b0e149519d6cd2d9be1ed0dc8540728765`. Artifacts `/mnt/nfs/shared/MOD007_modern_champions/`; patch dry-run `MOD007-patch-dry-run.log`. | Core lane port validated; no training or frozen-policy performance evaluation. Existing collector migration and spatial/vision fidelity gaps above remain next steps. No active own jobs or watchers. |
