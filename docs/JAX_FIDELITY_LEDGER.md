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
