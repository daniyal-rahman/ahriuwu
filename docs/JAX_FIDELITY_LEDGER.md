# End-to-end code review — running plan and progress

Updated 2026-10-25. Owner: this independent review thread. This section is the
single maintained review plan/progress record; older ledger entries below are
historical evidence. STATUS.md only carries the required short session handoff.

**Scope:** decoder and reconstruction → observations/tensors/labels → BC trainer
and evaluation → simulator/JAX correctness. Diagnose symptoms, reproduce causes,
fix verified defects, then simplify where that improves auditability. Passing a
wiring check does not establish source fidelity or competent play.

**Deferred:** action-space redesign. Later compare a source-faithful Tencent-style
baseline adapted explicitly for BC against the Fable/modern conditional proposal.
Do not start this comparison or change target representations during source review.

| Step | State | Evidence required to advance |
|---|---|---|
| 1. Reconcile current code, prior fixes, active paths and evidence | Initial inventory complete | Immutable ref inventory, pipeline map, fix-integration matrix |
| 2. Independently validate decoder/reconstruction | In progress: evidence selection | Source-qualified semantics plus synchronized recording comparisons; errors/missingness by event/state |
| 3. Trace source through tensorization | Planned | Named field provenance, receipt cutoffs, identity/coordinate/mask checks and human-readable example traces |
| 4. Audit trainer/evaluator and research claims | Planned | Mask/normalization/sampling/reset/resume checks; covered tiny-set memorization and nontrivial heldout behavior |
| 5. Review simulator rules and JAX semantics | Planned | Independent mechanic evidence, phase/reset/capacity checks, eager/compiled/batched consistency |
| 6. Fix and simplify verified issues | Throughout | Reproduction before fix; regression/independent evidence after; behavior-preservation checks for refactors |
| 7. Tune and run deferred architecture comparisons | Deferred until gates pass | Fixed splits/budgets/baselines; isolated changes; per-match uncertainty; faithful baseline deviations explicit |

## Integration and workspace checkpoint (2026-10-25)

Dani authorized landing fixes and reviewing workspace/snapshot hygiene before
continuing deeper review. Current modern/bc has been merged into this review branch, retaining experimental
flags/auxiliary/next-event/calibration/transfer paths; combined CPU/source/GPU
checks passed. Canonical modern/bc now contains the preserved combined history. No main merge: fetched origin/main
is September3 c1ecec9b and modern/bc is941commits ahead, so a main PR would pull
unreviewed historical scope. Preserve user AGENTS.md/CLAUDE.md worktree edits.

Workspace census:41registered,23already archived,13dirty. Several archived
worktrees contain uncommitted changes; archival location does not prove landed
or disposable. Full status/ancestry inventory in
[worktrees-before.json](/mnt/nfs/shared/modern-bc/review-20261025/worktrees-before.json).
Do not delete or move active/app-bound worktrees blindly. Source/data snapshots
are immutable reproducibility inputs, not alternate current checkouts. Current
v3loader rejects older contracts and verifies artifact hashes; combined finalizer
also rejects shards mixing source commits/contracts. Checkpoints bind dataset
contract and burn sampling; initialization transfers weights explicitly with donor
provenance/fresh optimizer, whereas incompatible resume is rejected.

Merge resolution retains both sets of evidence/docs/tests. The decoder source
binding defect now fails closed for top-level and transitive cached local modules;
use a fresh process to switch decoder snapshots.12focused source/emitter tests
passed4.26s. Full combined tests/raw reconversion/GPU checks remain required
before landing. Initial13143passed117tests but3mocked loss fixtures lacked newly
required entity inputs; fixing fixtures without relaxing eligibility. GPU13144
never started and was intentionally forgotten/cancelled after dependency failure.
Fresh attempt inputs will preserve this failed receipt. No old job inputs or
checkpoints rewritten.

Workspace disposition:
[worktree-disposition.json](/mnt/nfs/shared/modern-bc/review-20261025/worktree-disposition.json).
Of23already archived,5are dirty.8nonarchived are dirty,6clean app-bound branches
have independent commits,2clean app-bound snapshots are fully contained in
canonical domain branches,2clean checkouts are current review/rewards branches.
The two contained snapshots are direct-handoff decoding (contained in decoder)
and old sim-port (contained in modern/bc). They are app-managed paths, so moving
them blindly would leave stale app workspace bindings. No deletion/prune or
history rewrite; preserve branches and unique changes. Cross-domain main
consolidation needs its own review, not a blanket rebase.

## Landing gate and snapshot disposition

Combined code committed as merge2e7d575f, preserving current BCparentf10e4f42.
Fixture-only correction524c508c passes11targeted tests6.24s. Registered CPU13145
runs all120tests plus24real reconversions and profile; GPU13146afterok13145 runs
32step control/resume/evaluation, exact serialization, bounded continuation,
independent random-label and history diagnostics. Immutable inputs:
`/mnt/nfs/shared/modern-bc/integration-20261025/attempt-2/` with source archive,
SHA256, source-manifest and scripts. Attempt1/failed13143remain preserved;
cancelled13144must not be restarted. Production landing is **cleared** by these
checks. CPU13145now verified COMPLETED/0:120tests passed46.43s;
24native reconversions178.26s, v3contract/source524c508c, all24artifact hash sets
and missingCS/combat reverified.152270valid frames,0postburn omissions/0duplicate
supervision. GPU13146COMPLETED/0 verified:exact parameter/optimizer serialization,
32step resume maxabs0.0001184493/relativeL2 0.0000298733 within1e-3/1e-4bounds;
independent random-label accuracy0.1114685 over12748frames versus1/9chance.
Scratch13145/13146absent. Script recovered from Slurm's initial /srv cwd warning
by explicit private-scratch cd; future submissions should set a portable --chdir.
Actual validation remains allcontinue (73.0468% majority baseline), not competence.

Targetmodern/bc advanced toec804ca4 with only STATUS/ledger changes; merged both
without changing tested code. Verified source/ops paths remain identical to
524c508c. **Landed on modern/bc at ee344016** by fast-forward from ec804ca4.
[landing-receipt.json](/mnt/nfs/shared/modern-bc/integration-20261025/landing-receipt.json)
confirms tested source-code equality and byte-identical AGENTS/CLAUDE user edits.
Published canonical branch to `origin/modern/bc` with normal push and upstream tracking.
Review and canonical branches now share both histories; no old experiment was
replaced by an older snapshot. Main consolidation remains separate.

Historical snapshots/data/checkpoints are retained and identified by contract,
source commit and content hashes. They are never copied over a mutable current
checkout or relabelled compatible. Existing v1/v2datasets must be reconstructed
under v3, not have their manifests edited to pass loaders. Explicit weight-only
initialization retains donor provenance and starts a fresh optimizer; it does not
constitute a v3resume or promote donor quality. No bulk data rebuild launched
solely for cleanup. All future validation jobs stage independent copies from the
named immutable source. No active Slurm job or another agent's worktree changed.

## Historical checkpoint before integration

- Production review baseline: `modern/bc` resolved to **f10e4f4274bbbdbe3d7a2d7057e7a7fdcb639604**.
  Independent review snapshot: **8adeecc9c37d3fcc8158ea5dcc841a2174674dac**.
  Ref names move; findings here bind these SHAs, not an unspecified latest checkout.
- Current production still declares `replay-lane-events-v1`; reviewed v2/v3 fixes
  are isolated on the review branch. Current code still has fixed-stride windows,
  padding-only pointer masks, separately interpreted event history, unqualified
  damage attribution and position-availability-only acceptance. Prior review passes
  do not clear these production paths. Reconcile integration before fresh quality claims.
- New source-provenance reproduction: `_decoder()` appends DECODER_SCRIPTS and
  imports globally named modules. Switching the requested snapshot A→B in one
  process returns the original A modules for both loader and position model.
  Exact current function reproduced using two synthetic directories. This proves
  the helper does not bind imports to the requested source; it does not prove an
  existing frozen job imported incorrect modules. Repair must account for the
  loader's transitive local imports, not merely change sys.path order.

## Fix-integration matrix at the pinned production baseline

Function AST comparisons bind unchanged logic to the initial reviewed source
5a369951; dependency/behavior claims still use the specific tests/receipts below.
Evidence: [integration-functions.json](/mnt/nfs/shared/modern-bc/review-20261025/integration-functions.json).

| Area | Current production evidence | Isolated review status |
|---|---|---|
| Native event order/motion provenance | `_actions` unchanged from initial review | Fixed; raw audit and regressions passed |
| Position qualification/lifetime | `_read_lane_units` unchanged | Conservative source-v3fix;24raw conversions/94tests passed |
| CS/damage attribution | `_damage_notices`/`_last_hits` unchanged | Source-v3missingness/cutoff fix validated on24perspectives |
| Supervision windows | `event_windows` unchanged; trainer still calls without burn-aware stride | Overlapping windows verified on828perspectives;0postburn omissions/duplicates |
| Event/history consistency | `label_events`/`previous_received_events` unchanged | Shared interpretation/ambiguity/death history fixed and tested |
| Artifact/split enforcement | `load_event_games` unchanged | Hash/identity/match-split checks implemented and tested |
| Pointer eligibility | `targets` changed for angle experiment but still masks padding only | Qualified attack/GarenR eligibility implemented and tested |
| Strict memorization acceptance | Selector/aggregate strict gates manually rechecked; no per-button target-success requirement | Still open; old tiny fixture not evidence of all-mode learning |

This is an integration gap, not evidence the other branch regressed a merged fix.
New experiment code requires reconciliation when carrying fixes forward; do not
blindly replace the newer policy/trainer with an older reviewed snapshot.

## Pipeline map being verified

| Boundary | Current implementation | Review question |
|---|---|---|
| Native export → replay fields/positions | decoder `event_training_loader`/`position_model`, BC `replay.py` | Which code actually executes, and which fields are accepted rather than merely available? |
| Replay → shared observations | `observation.py`, `fog.py`, `convert.py` | Source qualification, approved visibility approximation, identity and coordinate ownership |
| Replay/observations → event examples | `event_dataset.py`, `event_labels.py`, `event_inputs.py`, `event_convert.py` | Causal lag, ambiguity, coverage, metadata and artifact contracts |
| Examples → loss windows | `event_batches.py`, `event_train.py`, `event_loss.py` | Exact supervision coverage, effective sampling distribution and recurrent context |
| Policy → scored predictions | `event_policy.py`, `event_eval.py`, `event_metrics.py`, `event_next.py` | Actual-button behavior, target eligibility, timing/reset protocol and baseline comparison |
| Historical pipelines | `train.py`, `dataset.py`, `labels.py`, `render.py`, `gate.py` | What remains used versus historical? Do not delete based on filenames/imports alone |

## Symptom measurements

- Reproducible inventory tool: `python3 -m ops.modern.bc_review_inventory --refs modern/bc HEAD --output NEW_PATH.json`.
  Reads committed tracked Python at refs resolved once; no ML imports or compute job.
- Production baseline: **36 non-test files / 5,406 lines; 26 test files / 1,567 lines**.
  Includes multiple pipeline generations; counts are a triage indicator, not a bloat verdict.
  Largest modules, imports, function lengths and literal schema constants are in
  [inventory-initial.json](/mnt/nfs/shared/modern-bc/review-20261025/inventory-initial.json).
- New import reproduction:
  [decoder-import-repro.json](/mnt/nfs/shared/modern-bc/review-20261025/decoder-import-repro.json).
- Existing independent measurements remain BC-REVIEW/BC-SYSTEM rows below:
  omitted/duplicate supervision, target coverage, missingness/qualification,
  action frequencies, continue-only baseline, exact versus bounded resume.
  Next automate the current-code symptom checks without interpreting a heuristic as proof.

## Recording evidence and next concrete actions

1. Initial fix-integration matrix completed above. Next prepare a compatible
   integration of reviewed corrections alongside newer next-event/history/angle
   experiments; the newer experiments have not themselves been cleared.
2. Verify existing recording receipts before new XPS capture. Newer coupled clips
   under `/mnt/nfs/datasets/lol_replay_probes/20261014-xps/coverage-ten-001/`
   have per-image brackets/camera records. Jayce receipt names NA1-5650853252,
   1,870 frames, transfer_verified and hashes, but acceptance is **not_evaluated**.
   A transfer receipt alone does not validate images/timing/decoded semantics.
3. Select a short sequence with independently usable timing and movement/attack/cast
   evidence, then trace it through the current BC formatter. The older 301-frame
   Jax overlay is not the only evidence available; newer capture cadence is irregular,
   and repeated video frames must never be counted as independent observations.
4. Present the source trace, quantified errors, rejected/unknown fields and reasons.
   Do not label receipt proxies as literal clicks or use captured memory to fill training data.

No new Slurm jobs, client interactions, subagents, or action-space experiments
started for this checkpoint. No production checkout edits. Review tool and record
changes are isolated on `review/bc-quality-20261024`.

---

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
| MODERN-013 | Item system implemented from client 16.19.8230722 (CommunityDragon 16.19 = build; `items.cdtb.bin.json` sha256 6880f35d…2ab62726): `lanerl_jax/data/build_modern_items.py` -> `items_client.json` (215 items = 210 in-store + 4 transforms + quest component 3867; 157 groups with max-ownable; recipes, sell modifiers, data values, calculations, spells). `modern_inventory` (recipe consumption, groups after consumption, stacks, level/ranged/buff gates, Elixir 5 s cd, shop r1000/dead), `modern_damage` (DMG/HEAL/SHIELD pipeline: summed dealt amps, multiplied received mods, unit-class 0.55/0.60, negative resist kept, Lifeline pre-shield, typed decaying shields, Death's Dance physical/magic storage, spell shield, execute, LS/omnivamp split with 33% modified ratio, GW 40%; sequential pass only for champions/stateful units, exact parallel running sums otherwise), `modern_item_effects` (10 family modules + registry; `coverage_report()` enforces every item implemented, STATS_ONLY 39, or DEFERRED 10). Hydra line/Stridebreaker actives in scope (offset-100 r450 circles, windup-capped casts, Titanic reset). Fixtures ITEMS.md F1–F18 covered; capped CPU tests 195 passed (item framework/runtime/10 family files + modern stats/towers/minions). `ops/modern_items_bench.py` CPU 2 cores: 1.7 ms/tick single, 0.56 ms/env vmapped 64, packet overflow 0. | World tick does NOT dispatch items yet; `modern_items.item_loadout_stats` rejects behavioural loadouts. Integrator must apply transforms/consumption, mana/gold/revive/attack reset, module helpers (Navori/Axiom cooldowns, RFC/Hexoptics range, Serpent's Fang, Boarding Party) and supply Kills/CC/ability tags. Known gaps: ally-targeted support effects are helpers only (1v1), no cast ids/mana costs on packets, no epic-monster flag, vision parts and non-Hydra actives deferred (MODERN-009); live-measurement items ITEMS.md §15 use documented defaults. GPU cost not yet measured. |
| MODERN-014 | Runes, shards and modifier ordering implemented from client 16.19.8230722 (`perks.cdtb.bin.json`): `lanerl_jax/data/build_modern_runes.py` -> `runes_client.json` (62 selectable runes, 7 shards, 5 styles; CLASSIC effect amounts + calculations; hash slots resolved by perk id). `modern_rune_data` (page legality, client substitutions, `lin`/`lin_growth`/breakpoints/level tables), `modern_rune_effects` (5 tree modules; `coverage_report()`: 60 implemented, 2 vision DEFERRED, 7 shards STATIC), `modern_stat_pipeline` (STAT.00–70: growth, flat/%/multiplicative, dynamic adaptive force, AS cap/windup, MS soft caps + Celerity, haste caps, tenacity groups, max-HP sync), `modern_stats` fixes (negative resist kept D1, additive dealt amps D4, dynamic adaptive D5), packets with `cast_id` and DMG.70 `block`, `modern_combat.combat_tick` (items + runes in one tick; dynamic armor/MR and max HP folded; second-generation trigger packets carried). Rune-granted items 2010/2150–2152/2422 added to the item catalog (220). Capped CPU run `ops/login_capped.sh 12G 4 … pytest` over modern combat/rune/item/stats/towers/minions tests: **342 passed in 48 min** (2026-10-01). | World tick does NOT call `combat_tick` yet. Integrator must supply `RuneEvents` and apply `RuneOutputs` (grants, purchase blocks, skill points, cooldown refunds, Hexflash, ghosting) per RUNES_IMPLEMENTATION.md. Known gaps: ally-targeted rune parts inert in 1v1, Axiom ult heal/shield amp, Shield Bash shield duration, Hexflash non-Cosmic summoner haste, Cheap Shot same-instance CC, Spellbook beyond swap timing (U-16); RUNES §10 U-01…U-22 use documented defaults. GPU cost not measured. |
| MODERN-015 | Economy/progression and Top role quest implemented: `build_modern_economy.py` -> `economy_client.json` (CLASSIC map11 experience curve/mod data/death times/kill-gold config + classic constants), `modern_economy` (ambient, levels to 20, minion XP split + comeback, kill credit 15 s, kill XP, bounty state machine, assist gold, structure shares, death timer/respawn, recall, Homeguard, fountain, `economy_step`), `modern_role_quest`. **Replay oracle**: `ops/replay_oracle_extract.{py,sbatch}` (Slurm CPU array) extracted raw client-memory observations from 145 recorded 16.9 games (`/mnt/nfs/datasets/lol_replays_16_9_772`, sha256 per source in the summary) into `lanerl_jax/data/modern/oracle/`, plus 16.9 champion HP records for 169 champions. Oracle-driven spec corrections: ambient first payment 65.0 s (U-E-1), death-time scaling continuous from 15:00 and none before (U-E-7), L19–20 death time = L18 (U-E-6), first-blood bonus shared at 50% with assisters. Confirmed: BRW table, first blood 400, assist cap incl. shutdowns, fountain 2%/0.25 s + Homeguard 8% missing/0.5 s, level-up full HP gain, growth curve G(L) + scaling shard on 19,887 level-ups (90.8% exact). Capped CPU: economy fixture + oracle tests 20 passed (2026-10-01). | World tick does NOT call `economy_step` yet. Oracle cannot see current gold, XP, hidden bounty history or quest points; team-advantage shutdown suppression/objective bounties disabled; victim depreciation base unverified; quest lane geometry/cadence and Homeguard decay are spec defaults. Riot match-v5 timelines (needs API key) would extend the oracle to XP, gold and full champion stats vs items/runes. |
| MODERN-016 | Riot match-v5 oracle for the 147 recorded 16.9 games (dev key via env, not stored; raw JSON with player names only in `/mnt/nfs/shared/riot-match-v5-16.9/`): anonymised extract `oracle/riot_16_9_frames.json.gz` (43,470 participant-minutes, 8,162 kills) + 16.9 champion/item records `oracle/client_16_9_stats.json` (CommunityDragon 16.9). Checked: XP-to-level table 100%; base kill gold by level (+FB) 98.3% of clean first deaths; champion stats via `modern_stat_pipeline` + shards + rune `stats` + item `dynamic_stats` exact-integer match (42,000 frames): %APen .986, MPen .956, omnivamp .949, LS .917, tenacity .816, MR .751, armor .740, AP .737, AD .663, AS .632, HP .582 (.72 non-support), MS .509; rune hooks raise MS exact rate +8 pts; remaining HP gaps have the implemented sizes (Biscuits 30/biscuit, Bloodline 85, Overgrowth 3/tier, RoA 100). `tests/test_modern_stats_riot_oracle.py`. | Riot timeline `abilityHaste`/flat `armorPen` always 0 (unchecked); stats truncated; AS = 100(1+bonus). Unmatched remainder = champion passives, stacking items, support quest upgrades (no purchase event), unobservable rune stacks — not modelled by the oracle, not simulator gaps per se. |
| MODERN-017 | Modern world integrated: `modern_step.step(state, orders, cfg)` composes Map11 config (`modern_world`: pinned grid/routes, 2 champions + 40 minion slots + 22 turrets + 6 inhibitors + 2 Nexuses, structure chain, validated loadouts), `modern_mechanics` (attack machine, missiles, CC timers/tenacity, route movement, Flash landing), `modern_lane_ai` (minion/turret targeting, Call for Help post-26.10, attack packets incl. turret %HP shots, Warming Up, spawn stats, plates/Bulwark/backdoor/Overgrowth; fixes: exact Overgrowth curve, negative resists, siege gold 49+U, melee count, 0.8 s spacing, Nexus turrets locked, inhibitor/Nexus tiers), `modern_champions` (Garen/Jax kits on packets/CC/stat contract, cast ids, PROP_ULTIMATE), `modern_summoners` (Flash, TP/Unleashed incl. quest, Ignite, Exhaust, Barrier, Heal, Ghost, Cleanse; Smite deferred), `modern_combat` (items+runes), `modern_economy`/quest. New `modern_damage` fields: dodge_basic, aoe_received_mult, received_mult_all (backdoor on true damage). Rune gap fixes: Shield Bash shield duration, Hexflash total summoner haste, Cheap Shot same-instance CC via cc_cast_id/cc_on_hit, Relentless Hunter modern combat clock. Tests: world integration 6 passed (21 min capped CPU), mechanics 4, lane AI+minions+towers 43, champions 43, summoners 50, rune/combat after gap fixes 72. Scripted 4-min game: ambient from 65 s, first blood +400, respawn, kill trade, XP/levels, no overflow; ~48 ms/tick single env on 2 CPU cores. | Kit approximations kept (13; Garen Q no dash, E crit ×1.3 fixed, W stacks on any kill, Jax R resists need a champion hit …) — champion workstream (MODERN-004). Champions act only on orders (no auto-acquire); all-visible (no fog/brush); top-lane waves only; quest lane = not in fountain; Homeguard endpoint/jungle masks unwired; Nexus position INFERRED-L; ally effects inert; RL obs/action schema not implemented (MODERN-005); GPU perf unmeasured (MODERN-007). Draft `modern_world.py` archived to `/mnt/nfs/projects/_archive/ahriuwu-modern-world-draft-20261002/`. |
| MODERN-018 | Modern world follow-ups and RL interface profile `modern-world-v1`: idle champions auto-attack the nearest targetable enemy in attack range (INFERRED-M client default); `ChampionLayer.last_cast` (observed-cast memory) and `cs`; `modern_step.champion_stats` (displayed stats). `obs/modern_builder.py`: legacy `Observation` container (32×16 entities, same slot blocks, 6 global) + 32-column modern self block; per-team lane frames from the top outer turrets/Nexuses (`modern_frames`). `train/modern_actions.py`: screen-click decode to `ModernOrders` with buttons legacy 8 + summoner_d/f, same hit-testing. `ops/modern_world_bench.py` (vmapped scan benchmark). Tests: `obs/tests/test_modern_obs.py` 5 passed (frame round trip of entity offsets, click-on-observed-enemy → attack for both teams, ground click direction, minimap no-op, self block vs cooldown/cast/death/quest), world idle auto-attack test. | No fog: every enemy cast counts as witnessed and all units are visible (MODERN-009). No attack-move order (targetless attack-move = move). Shop/skill-up not in the action space. Not wired into a trainer, reward or scenario bank. Reflection frame assumes chiral symmetry; Jax Q is unit-targeted so it holds, but skillshot kits would break it. GPU benchmark queued (MODERN-007). |
| MODERN-019 | Modern fog of war (`sim/modern_vision.py`, spec `docs/modern/VISION.md` VIS.01–13): sight radius centre to centre (champion/super/turret/Nexus 1350, minions 1200 from client `perceptionBubbleRadius`/wiki, inhibitor 0 INFERRED-L), walls opaque except transparent/always-visible cells, brush opaque outside-in and to rays through it, structures never fogged, own team always visible, attack reveal (hidden champion launching an attack or a unit-targeted cast opens a fixed 300 u / 2 s circle for the enemy team). Rays reuse the legacy supercover caster (fused CUDA / CPU reference) over the 26.19 navgrid (same flag bits; 40 brush patches vs wiki 39). `ModernState.visible` (2,N), `sight` (N,N), `reveal`; computed at the end of each tick; `refresh_visibility` for hand-edited states; `build_config(fog=False)` restores all-visible. Gates: lane AI `select_targets(visible)`, champion attack/cast/summoner targets (a target entering fog is dropped), idle auto-attack, rune `sight`/`visible`, observation slots, click hit-testing, `ChampionLayer.seen_cast` (only witnessed enemy casts). Builder bug fixed: untargetable structures were hidden from the observation. Pre-existing bug fixed: the pinned 26.19 champion snapshots lacked `primaryAbilityResource`, so Jax had 0 mana and could never cast; mana now read from it (Jax 339 +52/lvl, 1.64/s +0.14/lvl; Garen 0). Tests: `sim/tests/test_modern_vision.py` 6 (fixtures, range/walls/structures, brush both ways + observation, reveal expiry, fogged attack dropped + unseen cast not remembered, attack from brush reveals), obs 5, world 7 + mechanics 4 (two capped runs). GPU before fog (job 2187, RTX 5080): 1 env 5.7 ms/tick, 64 envs 7.8k env-ticks/s, 512 envs 11.2k env-ticks/s, 1024 envs OOM (6.7 GiB alloc). | Map11 `ca_RevealAttackerRange` 400 / `TimeOut` 4.5 unexplained (U-VIS-1); reveal circle fixed, not following (U-VIS-2); brush-to-other-brush rays (U-VIS-3). Wards/trinkets, stealth, true sight, Faelights, nearsight, dynamic terrain deferred. Fog GPU cost: job 2203 RTX 5080, rays fog vs none 11.17k vs 11.21k env-ticks/s at 512 envs (no measurable cost); default fog is rays, `fog="fast"` kept as an option. |
| MODERN-020 | Game-rule completion and full-world integration (rules only). Agent-built modules, integrated by the lead into `modern_step`: wards/trinkets/stealth/true sight + vision runes (`modern_wards`, WARDS.md); jungle camps, Scuttle, monster AI, camp rewards, Smite, jungle pets (`modern_jungle`, JUNGLE.md); 26.19 epic objectives (Voidgrubs, Rift Herald + Mercenary, Drakes/Soul/Elder, Baron; Atakhan/Blood Roses/Feats removed in 26.1), team buffs, 21 Elemental Rift x Baron-pit terrain variants from client navgrid overlays (`modern_objectives`, `modern_dynamic_terrain_rift`, OBJECTIVES.md); Garen/Jax kits to 26.19 client data (CHAMPIONS.md, 7 small approximations left); all-lane waves, structure pads, map regions (quest lane, Homeguard endpoint/jungle, river), attack-move, idle acquisition, game end, remaining item actives (`modern_lane_ai`, `modern_dynamic_terrain`, `modern_map_regions`, `modern_item_actives`, LANES_TERRAIN.md). World layout 216 units (2 champions, 120 minion, 48 monster, 16 ward, 30 structure slots; structures last); Nexus at navgrid pad centres. Economy gains extra gold/XP/epic, recall-channel and pet minion-reward inputs; vision gains extra sight sources (Scuttle shrine). Kit hooks wired (attack-target kind, root, kit attack orders, windup/period overrides, uncancellable windups, kit extra range, Garen E ghosting); stale `s.kits` attack mods fixed. Pre-existing bug fixed: Jax 0 mana (MODERN-019). Actions: 19 buttons incl. level-ups, buy (catalog row), sell/use item (slot), ward, control ward, real attack-move; `Loadout.auto_skill`. Observation `ModernObservation`: 32x20 entities (monster/epic/ward/control columns, neutral team), self block with skill points and trinket charges, inventory and exact shop affordability. Tests: `test_modern_world_rules.py` (whole-world symptoms) + every subsystem suite (see report). | Remaining: only Garen and Jax, one per team (no allies: ally-targeted effects inert, no ganks); Hand of Baron siege splash not applied; Gromp magic bonus at launch; neutral monsters walk the blue mask; per-spec INFERRED values and unresolved items (VISION U-VIS-1..3, WARDS U-W-1..7, JUNGLE pathing/patience, OBJECTIVES Rift timing/Baron shred/Rodeo, LANES_TERRAIN pad release). |
| MODERN-021 | Modern world throughput, wiring and structure review (2026-10-03). Tools: `ops/modern_world_bench.py` world-variant flags (`--fog rays|fast|off`, `--no-jungle`, `--no-objectives`, `--lanes`); `ops/modern_world_profile.py` attributes traced XLA op time to source file:line via the compiled HLO stack-frame tables. CPU desktop, 8 cores, 16 envs, after 1800 warm ticks: HEAD 1e15e06 (top lane, 72 units) 41 ms/tick; MODERN-020 world (216 units) 191 ms/tick; ablations fog off 134, fog fast 136, no jungle/objectives 163, top lane only + no jungle/objectives/fog 128 (fixed 216-slot shapes dominate, not lane count). Profile before: walkability (route/segment disk tests) 31%, ray fog 21%, item `nearest_k` double argsort 13%, rune `first_instance` P×P (P=512) 9%, lane AI 5%; collision ~0. Exact fixes (randomized equivalence checks vs the old code, 0 mismatches): `nearest_k` = `lax.top_k` (stable ties); `first_instance` = lexsort groups + segment_min; `eject` tests every unit but ring-searches only the first 16 stuck units per call (rest on later ticks; equal end state); `segment_clear(max_radius=)` sizes the `is_walkable` cell window from the static route radius (35) + sampling inflation (9x9 -> 5x5/7x7 cells; 12k real-map segments identical), `route_next` radius above `routes.radius` fails closed. After: 108 ms/tick (1.77x). Wiring fixes (review, 12): Overgrowth death latch `death_seen` (start-of-tick sight); item/rune slows routed through CC (`apply_cc`, tenacity applies); item-active cast lockout (`item_cast_until` for movable casts); objective mana; structure credit mask independent of plates; wards excluded from item units/on-hit; ward click radius 65; jungle pet damaged-champion/brush inputs and Gustwalker MS; non-champion slows; epic takedowns in `killed_units`; cast-id stride 256; stale champion unit columns (`ad/armor/mr/arange/aspeed/mspeed`) written each tick; observation `champion_stats` shares the tick stat composition (monster buffs). Tests: step 7, world rules 9 (3 new), vision 6, obs 6, affected unit suites passed. | GPU RTX 5080 (jobs 2229/2233, 150 timed ticks after 1800 warm): HEAD 1 env 5.8 ms, 1024 envs 87 ms/tick = 11.7k env-ticks/s; MODERN-020 1 env 6.7 ms, 1024 envs OOM (24.9 GiB buffer); after CPU fixes 1 env 6.4 ms, 256 envs 32 ms (7.9k), 1024 OOM (10.8 GiB). Memory cause (compiled-HLO buffer ranking): `route_next` connection checks (216 units x 25 candidates x 33 samples) gathered each 5x5 window cell-by-cell (~68 MiB/env index tensor) and per-unit team-grid selects (18 MiB/env each in move_step/eject). Exact fix: `is_walkable` reads its window with one `dynamic_slice` of a blocked-padded grid; `team_view` stacks team masks and reads a per-unit `layer` (200k real-map queries identical). Final: 256 envs 31 ms (8.2k env-ticks/s), 512 envs 54 ms (9.5k), fog off at 512 51 ms (fog ~5% on GPU); 1024 still OOM (7.4 GiB) -> per-env memory caps the batch near 512; next: bound routing memory (fewer connection checks or chunked `lax.map` over units). GPU profiler run crashed (CUDA launch failure), not repeated. Remaining CPU cost: ray fog ~32% (CPU reference caster), routing ~19% (25-candidate connection checks per unit per tick), lane AI ~10%. Behaviour changes: item/rune slows now get tenacity; Hydra/Tiamat casts root for min(0.2 s, windup); >16 units stuck in closed terrain take extra ticks to eject. Structural debt listed in WORLD_IMPLEMENTATION.md (phase split of `step`, world-subsystem protocol, one unit table, champion registry, last-tick latches, naming). |
| MODERN-022 | Structure and throughput (2026-10-03/04, d2d35be). `step` split into phase functions over a documented `TickScratch` (bitwise identical to the monolith over 2000 ticks, full and minimal worlds). Movement steers by a cached per-unit route anchor (`route_follow`; at most 16 nearest-node searches per tick; only slots before the wards): 4.2x cheaper movement, 6.7x less temp memory, 400 real-map routes arrive 328 vs 324 before. Exact rewrites: `nearest_k` as k argmin rounds (top GPU kernel before), sequential damage pass stops at the real packet count (40 random packet sets identical). Fixes: structures no longer collide (pads block), eject uses the movement clearance (red pinned at its own inhibitor/turret pads), dash state and facing persist, Homeguard/Ghost/Heal/Gustwalker MS before the soft caps, Recall 0.5 s cast + grace on the whole recall (replay-fidelity findings). Compilation cache (`lanerl_jax/jax_cache.py`), shared-program test harness (28 full-tick tests 135 s cold / 92 s warm, previously OOM at 8.4 GB), `ops/modern_perf_guard.py`, `train/modern_vec_train.py` + `train/scan_ppo.py`, `ops/modern_replay_fidelity.py` (145 games). GPU RTX 5080: 256 envs 20.3k env-ticks/s, 512 23.7k, 1024 24.4k, 2048 23.0k (start of MODERN-021: 8.2k at 256, OOM at 1024; HEAD 72-unit world 11.7k at 1024). CPU 16 envs 85 ms/tick. | Collision cost (MODERN-023) not in these GPU numbers. Pending-cast buffer window and route-replan budget are unsourced defaults. |
| MODERN-023 | Mechanics audit and collision (2026-10-04, 79ffc6a). `docs/modern/MECHANICS_AUDIT.md` (29 items; replay checks confirm instant acceleration/turning and windup timing). `modern_collision.py`: steering avoidance on client pathing radii (champions 35, minions 35.7/55.7), `docs/modern/COLLISION.md`; creep-block proxy legacy p10 0.38 with multi-second pins vs new 1.00 vs replays 0.79 (2.8 short stalls/min). Fixes: move orders end on arrival/no progress (idle auto-attack was blocked after any move), walk-in and buffered casts (`QueuedCast`), one-tick windup grace, Minion Pushing, fog-lost target chased to last seen, stop button and selection-radius clicks (action profile modern-world-v2), trainer `--action-delay-ticks`. Tests: world rules 17, step 7, vision 6, obs 6, collision 12, helpers 5, mechanics 5, champions 61, modern trainer 6; CPU 16 envs 95 ms/tick (collision ~-10%). | Sim now creep-blocks less than replays (no short stalls); contact distance max vs sum and `ai_PostAvoidance*` semantics unknown. Fog chase does not resume the attack when the target reappears. Latency default 0 (user decision). |
| MODERN-024 | Package and structure cleanup, scenario-sized layout (2026-10-05, 3c5b777 + cde7541). The 26.19 world moved from flat `sim/modern_*.py` (plus `obs/modern_builder.py`, `train/modern_*.py`, `data/modern*`, `data/build_modern_*`, ops `modern_*`) into the `lanerl_jax/modern/` package, layered core -> map -> rules -> world -> interface, tools in `ops/modern/` (old -> new paths: `docs/modern/WORLD_IMPLEMENTATION.md` Layout; `git log --follow`). `world/step.py` split verbatim into `state`/`views`/`units`/`scratch`/`phases/*`/`tick` (traced jaxpr identical, full and top-lane worlds). `world.config.Layout` sizes the unit blocks by the scenario (full map 216 units, top lane without jungle/objectives 88; packet capacities follow: 512/256 vs 256/128). One spawn path (`core.types.UnitWrite` + `world.units.write_units`), state columns named as `WorldUnits`, dead `lane_wp` column removed, champion registry `champions.KITS` (id, skill order, rune traits per kit), `ModernState.prev` groups the one-tick lags, `sight` champion rows only, patch data via `modern.data.PATCH_DIR`, lane geometry in `map.lanes`, `item_tick` in `combat`. Verification: new `ops/modern/golden.py` (3600 ticks of seeded chaos orders, CPU): full map every state leaf bitwise identical to the pre-restructure code except the trimmed `sight`; top lane (88 vs 216 units) game summary identical at every checkpoint (champion x/y/hp/mana/gold/xp/cs, minions and minion HP per team, 30 structure HPs); peak packets 6 (full) / 4 (top), no overflow. Tests (login-node CPU, per file): 1131 passed; 1 pre-existing failure, `sim/tests/test_sim_config.py` cites PARITY-001 (no ledger row; fails identically at bf1bf75). | GPU timing pending (desktop in Windows): job queued to time legacy 4.20 (policy in loop and sim only), modern full map and top lane in one session, then the perf guard. Note: THROUGHPUT_AUDIT's 51.6k env-steps/s for 4.20 includes the policy forward. Not done on purpose: a jungle/objectives hook protocol (two systems with different hook shapes; explicit per-phase blocks kept). |
| MODERN-027 | Nasus (75), the third 26.19 kit (2026-10-12): `champions/nasus.py`, rules NAS.* with evidence in `docs/modern/CHAMPIONS.md` (CLIENT bins fetched 2026-10-12 with Malphite/Renekton/Darius/Tryndamere into `cdragon-16.19/`, checksums in `SHA256SUMS`; wiki raw pages in `cdragon-16.19/wiki-2026-10-12-champions/`). World pieces any new kit needs: `data/build_champions.py` regenerates the pinned records (Garen/Jax snapshots semantically unchanged, now with `slots` and record radii); `ChampionBase.gameplay_radius/pathing_radius/acquisition_range` replace the hardcoded 65/35/400; point-target walk-in (`POINT_TARGET_RANGE`); `KitCtx.tenacity`; kit `attack_speed_cripple` applied to the composed champion attack speed (previously no cripple reached attack speed; Frozen Heart's item cripple is still unwired); observation self width 28 + 2 x kits. Evidence settled: Spirit Fire DoT is per second (client tooltip and V14.18; the wiki's x10 total is wrong); the R cap of 240/s applies to all units; 26.18 Wither tenacity-scaled steps. Tests (login CPU and Slurm cpu jobs 4191/4192): test_nasus 39, test_nasus_world 5, champions 61, collision/loadouts/step_helpers/rune_framework/mechanics 100 total with champions, step 7, world_rules 17, obs 6, vision 6, train 6. Golden 3600 ticks (job 4194): every pre-existing leaf bitwise identical to HEAD by name in both worlds (the tool's positional verdict trips on the 35 new Nasus leaves). | Approximations in CHAMPIONS.md (Nasus Q on wards/turrets, Wither cleanse, one fire per Nasus, size not in hitbox). No Nasus RL run. |
| MODERN-028 | Renekton (58), the fourth 26.19 kit (2026-10-12): `champions/renekton.py`, rules REN.* in `docs/modern/CHAMPIONS.md`. Fury is kit state (0-100; +5 per landed attack, ability gains with per-cast caps, x1.5 below half health, decay after 12 s out of combat, 50 to empower Q/W/Dice) and fills the observation's resource bar (`champions.resource_fraction`). World hooks: `KitCtx.move_speed` (E dash speed 750 + MS), `KitOut.strip_shields` (empowered W destroys shields before the tick's damage), `cast_lockout` honoured from any hook (W post-strike 0.525 s). Tests: test_renekton 21, test_renekton_world 5 (dash moves 450 and hits on the way, empowered W strips Garen's W shield and stuns 1.5 s x Garen's 60% tenacity, Fury on the obs bar, Dominus +300 HP), kit suites 121, step 7, world_rules 17, vision 6, obs 6 (Slurm cpu jobs 4195-4197). Golden 3600 ticks (job 4198 rerun): every pre-existing leaf bitwise identical to MODERN-027 by name. | Approximations in CHAMPIONS.md (W on-hit effects once, no 0.2 s W cast time, dash ignores terrain, Dice window from dash end). No Renekton RL run. |
| MODERN-029 | Darius (122), the fifth 26.19 kit (2026-10-12): `champions/darius.py`, rules DAR.* in `docs/modern/CHAMPIONS.md`: Hemorrhage stacks (C, N) with 4 ticks per application, x2.5 vs monsters (26.4), Noxian Might AD by level; Decimate heft/ghost, blade/handle and missing-health heal; Crippling Strike crit bonus, slow, kill refund; Apprehend armor pen and 50-degree pull; Noxian Guillotine leap, stack-scaled true damage, recast window / rank-3 reset. World hook: kit `displacement` (forced movement, ghosted) applied in the move phase. Tests: test_darius 25, test_darius_world 4 (Apprehend moves Garen to ~150 from Darius, knocked up then slowed 40%; bleed and Decimate heal through the pipeline; rank-3 execution resets R), kit suites 164, obs 6 (Slurm cpu 4200/4201). Golden 3600 ticks: every pre-existing leaf bitwise identical to MODERN-028. | Pull speed 1500 INFERRED-L; no fear after executions; pulls and leap ignore terrain. No Darius RL run. |
| MODERN-030 | Malphite (54), the sixth 26.19 kit (2026-10-12): `champions/malphite.py`, rules MAL.* in `docs/modern/CHAMPIONS.md`: Granite Shield refilling one kit-sourced shield, Thunderclap passive armor (x3 under the shield), empowered attack and aftershock cones (x1.8 monsters), Seismic Shard slow and speed steal, Ground Slam cripple, Unstoppable Force dash and knock-up. World: `Shields.source` (a sourced grant replaces the unit's earlier one; `ShieldGrant.kind = SHIELD_* + 16 x SOURCE_*`) and `KitCtx.kit_shield`. **World bug fixed:** after a dash's first tick the move phase advanced it at the 1400 fallback instead of its own speed (Jax Q is 1400, so it never showed); Renekton E (750 + MS), Malphite R (1500 + MS) and Rocketbelt (1500) now move at their speeds. Tests: test_malphite 25, test_malphite_world 3, kit suites 189, renekton/darius world 5/4, world_rules 17, step 7, items defense/support/starters/runtime/framework 21/17/19/3/23, combat 5, runes_resolve 23, summoners 50 (Slurm cpu 4202-4208). Golden 3600 ticks: every pre-existing leaf bitwise identical to MODERN-029. | Thunderclap cone half-angle 45 deg INFERRED-L (no client or wiki value); speed steal from current MS; strongest cripple only. No Malphite RL run. |
| MODERN-031 | Nasus, Renekton, Darius and Malphite unregistered (2026-10-12) at the user's direction: no kit ships with an INFERRED value or an engineering approximation. Their code, records and tests stay in the tree (tests skip); the world pieces from MODERN-027..030 stay (exact, golden-verified). Each kit returns alone after its CHAMPIONS.md approximation list is empty; values no source gives are measured in the live client by a written protocol. | Order Nasus, Renekton, Darius, Malphite. Registered kits: Garen, Jax. |
| MODERN-032 | Nasus, Renekton, Darius and Malphite re-registered (2026-10-12) after an interaction-level audit; small differences are documented per kit in CHAMPIONS.md (limitations), not left as hidden guesses. Fixes: (1) **Darius Apprehend** measured on 14 pulls of Garen in the 16.9 replay corpus (20 fps positions): drag at 1784 u/s to ~45 from Darius, rebound over 0.5 s to ~230 (was a guessed single 1500 u/s move to 150); (2) **dashes and terrain** per wiki Dash: a point dash crosses a wall only to a walkable end point, else stops short (Rocketbelt always stops); Malphite R lands and Renekton E ends where the dash actually stops (`KitCtx.dashing`); (3) **size** scales the gameplay radius (wiki Unit size): Nasus R 80 -> 104/108/112, Renekton R +20%, Malphite 8% per 100 armor (cap 35%); Nasus storm reaches max(400, 320 + gameplay radius). Tests (Slurm cpu 4218/4260): champions 61, nasus 39, renekton 21, darius 25, malphite 25, world tests 5/6/4/3, step 7, world_rules 17, obs 6, collision 12. Golden 3600 ticks (4220): every pre-existing leaf bitwise identical by name. | Remaining differences are listed per kit and in the shared section of CHAMPIONS.md (cast-time cost timing, spell shields vs kit CC, Malphite cone angle INFERRED). |
| MODERN-033 | Aatrox (266), the seventh 26.19 kit (2026-10-13): `champions/aatrox.py`, rules AAT.* in `docs/modern/CHAMPIONS.md`: Deathbringer Stance (max-health proc, heal, level-scaled static cooldown with 2/4 s refunds), The Darkin Blade's three casts with the wiki hitbox areas and sweetspots, Infernal Chains chain, tether and pull, Umbral Dash (castable mid-cast) and its champion-damage heal, World Ender. World: `immobile` kit hook (held in place without a cast lock, so Aatrox can E during Q/W cast times); CODE_AATROX_P/W2 cast codes. **World bug fixed:** records that give the attack as `mAttackCastTime`/`mAttackTotalTime` (Aatrox, Fiora, Ambessa, Camille, Ornn) got the 0.3 default windup; now cast/total (wiki Attack speed; Aatrox 19.737%). Infernal Chains' zone (>= 573), pull time (0.25 s) and airborne time (0.5 s) measured on 19 clean pulls in the 16.9 replay corpus (`/mnt/nfs/shared/champ-expansion/aatrox-measurements/`). Tests (Slurm cpu, desktop): test_aatrox 43, test_aatrox_world 6, champions 61, nasus 39, renekton 21, darius 25, malphite 25, obs 6, collision 12, nasus/renekton/darius/malphite world 5/6/4/3, step 7, world_rules 17 (4524/4525/4527). Golden 3600 ticks, top-lane world (4528): every pre-existing leaf identical by name; the full-map world exceeds the 8 GB a Slurm job can take now and is pending. | Infernal Chains zone radius 650 between the replay lower bound 573 and the wiki escape-speed note (~612-690), circular; fear modelled as a stun; extended wall dash not modelled; remaining small differences in CHAMPIONS.md "Aatrox limitations". No Aatrox RL run. |
| MODERN-034 | Fiora (114), the eighth 26.19 kit (2026-10-13): `champions/fiora.py`, rules FIO.* in `docs/modern/CHAMPIONS.md`: Duelist's Dance vitals (direction arcs, wiki alternation with world-key randomness, any damage from the vital's side strikes it: true damage, heal, decaying speed), Lunge (speed from bonus MS, wiki stab priority, applies on-hit, half cooldown on a hit), Riposte (parry, debuff immunity, stab that stuns after a parried immobilize), Bladework (two attacks: slow, then forced crit at 160-200% + bonus crit damage), Grand Challenge (four vitals, proximity speed, Victory Zone). World: `KitCtx.rand`, `KitDefense.parry` -> `Defense.parry` (non-turret packets negated), `cc_immune` / `on_parry` hooks in the CC phase (CC, slows and newly started forced movement dropped; `ChampionLayer.disp_ignored`), `KitAttackMods.force_crit` / `crit_mult`, `KitOut.on_hit_target` (ability on-hit reaches items/runes as a landed hit). Client data overrides stale wiki text: Riposte 50% slow + 25% cripple (V14.4), vital speed decays over 2 s, Bladework window 3 s. **World bug fixed:** a point dash's first tick steered toward the previous dash's stored end point (the fountain initially), so every point dash (Renekton E, Malphite R, Aatrox E, Fiora Q) took a wrong first step and ended one step short; target dashes (Jax Q) were unaffected. Tests (Slurm cpu, desktop): test_fiora 16, test_fiora_world 5, champions 61, malphite 25 (4538); nasus 39, renekton 21, darius 25, aatrox 43, obs 6, collision 12 (4531); aatrox/renekton/malphite/darius/nasus world 6/6/3/4/5, step 7, world_rules 17 (4535). Golden 3600 ticks, top-lane world (4536): every pre-existing leaf identical by name (full-map world pending, 8 GB job cap). | Vitals ignore sight; a blocked hit strikes no vital; Lunge cooldown starts at the stab; Riposte does not consume shields or block kit cripples/Ignite; Bladework does not crit structures (world rule). No Fiora RL run. |
| MODERN-035 | Yone (777), the ninth 26.19 kit (2026-10-13): `champions/yone.py`, rules YON.* in `docs/modern/CHAMPIONS.md`: Intent (crit doubling, excess crit to AD, 95% crits), Steel and Spirit (alternating half-magic attacks), Mortal Steel (attack-speed-scaled cast time and cooldown without haste, per-target crits, basic damage + on-hit/on-attack on the first target, two-stack dash and whirlwind), Spirit Cleave (cone, %max-health mixed damage, minion floor, monster cap, champion-scaled shield), Soul Unbound (spirit form, marks of health damage, return with true-damage detonation), Fate Sealed (stun, blink 200 beyond the last champion, gust, pull, knock-up). World: the resource pool holds only mana (arType 0) and energy (1) (Yone's Flow record would have read as 500 mana); `KitCtx.rand_n` (C, N); `KitOut.on_attack_proc`; `KitAttackMods.magic_split` (second, magic packet of a melee attack); kit blinks (`Dash.blink` teleports by the Flash landing rule). Measured on the 16.9 replays (`/mnt/nfs/shared/champ-expansion/yone-measurements/`): Soul Unbound's return runs at ~2950 u/s (5 returns, 2920-2965; the wiki's "1200, pending for test" is the outbound speed) and the body is left 300 behind the cast point. Tests (Slurm cpu, desktop): test_yone 19, test_yone_world 5 (4545); yone/fiora/aatrox/champions/nasus/renekton/darius/malphite 19/16/43/61/39/21/25/25, obs 6, collision 12 (4546); fiora/aatrox/renekton/malphite/darius/nasus world 5/6/6/3/4/5, step 7, world_rules 17 (4547). Golden 3600 ticks, top-lane world (4548): every pre-existing leaf identical by name. | Spirit Cleave's client crit fields unused (wiki silent); body point not pushed out of terrain; Azakana crits double both halves; small differences in CHAMPIONS.md "Yone limitations". No Yone RL run. |
| MODERN-036 | Ambessa (799), the tenth 26.19 kit (2026-10-13): `champions/ambessa.py`, rules AMB.* in `docs/modern/CHAMPIONS.md`: energy (200, 10/s) through the world pool; Drakehound's Step (a move or attack command around each ability lockout dashes 175-350 at 770 (+60 at 6/11/16) + MS, stopping at walls); Medarda Maxim stacks and the chained empowered attack (uncancellable, 0.35/0.30/0.25 s total attack time, bonus damage with life steal, energy refund); Cunning Sweep / Sundering Slam (edge and first-target full damage, monster rules); Repudiation (shield, empowered smash after blocking champion/large monster/turret damage); Lacerate (decaying slow re-applied per tick, second spin after a dash); Public Execution (armor pen and ability healing; seize, suppression, slam). World: `KitCtx.move_goal/moving/attack_order/last_order_at/life_steal`, `ChampionLayer.last_order_at` (time of the last move/attack command), `KitOut.dash_stops_at_walls`. The wiki pages are at 26.20 (Drakehound's Step nerfed to 5-25 + 20% bonus AD there); the 26.19 client values are used. Tests (Slurm cpu, desktop): test_ambessa 10, test_ambessa_world 5 (4550); ambessa/yone/fiora/aatrox/champions/nasus/renekton/darius/malphite 10/19/16/43/61/39/21/25/25, obs 6, collision 12 (4551); yone/fiora/aatrox/renekton/malphite/darius/nasus world 5/5/6/6/3/4/5, step 7, world_rules 17 (4552). Golden 3600 ticks, top-lane world (4553): every pre-existing leaf identical by name. | Chained attack time from the client override, not the wiki's "+50% attack speed" (INFERRED-M, replay cadence agrees); suppression modelled as a stun; small differences in CHAMPIONS.md "Ambessa limitations". No Ambessa RL run. |
| MODERN-037 | Camille (164), the eleventh 26.19 kit (2026-10-13): `champions/camille.py`, rules CAM.* in `docs/modern/CHAMPIONS.md`: Adaptive Defenses (typed shield from each enemy's recorded damage, cooldown after the shield ends), Precision Protocol (empowered attack, recast with the ramped x2 bonus and 40-100% true conversion), Tactical Sweep (1.1 s wind-up, outer-cone %max-health bonus, decaying slow, healing), Hookshot / Wall Dive (grapple to the first wall within 800, pull at 1050 + MS, 1 s hang bound to the wall, dive 400/800 stopping on a champion and stunning), The Hextech Ultimatum (untargetable leap, zone leash, knock-away, current-health on-hit). World: `mechanics.first_wall` and `KitCtx.terrain`; `leash` (move phase clamps a unit inside a circle after all movement); `untargetable` (kit untargetability before orders are filtered); `KitAttackMods.true_split`. Hookshot's pull speed checked on the 16.9 replays (`/mnt/nfs/shared/champ-expansion/camille-measurements/`). Tactical Sweep's cooldown uses the record's `Cooldown` (26.16: 12-10 s); its legacy `cooldownTime` field is stale. Tests (Slurm cpu, desktop): test_camille 11, test_camille_world 5 (4557); camille/ambessa/yone/fiora/aatrox/champions/nasus/renekton/darius/malphite 11/10/19/16/43/61/39/21/25/25, obs 6, collision 12 (4558); ambessa/yone/fiora/aatrox/renekton/malphite/darius/nasus world 5/5/5/6/6/3/4/5, step 7, world_rules 17 (4559). Golden 3600 ticks, top-lane world (4560): every pre-existing leaf identical by name. | Tactical Sweep wind-up from the wiki (1.1 s) not the client ChargeDuration 0.75 (INFERRED-M); hang end without a dive INFERRED-L; zone is a circle; small differences in CHAMPIONS.md "Camille limitations". No Camille RL run. |
| MODERN-038 | K'Sante (897), the twelfth 26.19 kit (2026-10-13): `champions/ksante.py`, rules KSA.* in `docs/modern/CHAMPIONS.md`: Dauntless Instinct (marks, the mark proc with minion floor / monster cap, All Out's %max-health bonus on attacks, the mark and each cast's nearest ability target), Ntofo Strikes (bonus-resist cast time and cooldown, rectangle, stacks, the shockwave missile with its 500 u/s pull and concurrent 0.8 s stun), Path Maker (0.4-1 s charge with damage reduction and displacement immunity, recast while stunned / on attack-move, dash with carry at 1.3x and the charge-scaled stun, All Out's true damage), Footwork (location / ally dash, shield), All Out (tenacity-free root, pull 300 or over terrain to 450 past its far side at 900-1700 u/s, landing 250 beyond, vanish, slam, 15 s form: 65% health threshold, base-resist loss, attack speed, omnivamp, bonus-armor penetration, mana refill / decay / restore, free basics, cooldown after it ends). Kit displacements now ignore terrain and collision in the move phase (as the contract said; a unit left in terrain is ejected when its displacement ends), traced only in matches with a displacing kit. World: `CCOut.tenacity_free` (also Ambessa's suppression now ignores tenacity); `displacement_immune`; `hp_cap`; `mana_override`; `KitCtx.bonus_armor` / `bonus_mr` / `raw_cast_slot` / `attack_move_cmd`; `ItemStats.base_armor_loss` / `base_mr_loss` / `percent_bonus_armor_pen` with `Defense.bonus_armor` and `Offense.percent_bonus_armor_pen` in mitigation; `mechanics.wall_exit`. Client vs wiki resolved from the 26.19 client, the intact wiki patch history (rev 4078187; the live page was vandalised 2026-10-10) and 16.9 replay measurements (`/mnt/nfs/shared/champ-expansion/ksante-measurements/`: dash speeds, carry, R pull speed and landing). Tests (Slurm cpu, desktop): test_ksante 15, test_ksante_world 5 (4647/4648, after the Q3 width fix); every other module passes (4605 non-world; 4606 world modules: aatrox/ambessa/camille/darius/fiora/malphite/nasus/renekton/yone world 6/5/5/4/5/3/5/6/5, world_rules 17); test_train 6 (4656; now needs ~28 GB, was ~20). Golden 3600 ticks, top-lane world (4669): every pre-existing leaf identical by name (an earlier unconditional terrain override drifted one minion and one missile position by float rounding; bisected to `move.py`, fixed by the per-match gate). | 26.20 Path Maker changes not applied (data is 26.19); shockwave pulls at ~408 u/s in 3 of 6 replay samples unexplained; All Out's pull direction for overlapping units unknown (one replay sample 25 deg off); Q3 half-width 70 (client convention; wiki width 100); small differences in CHAMPIONS.md "K'Sante limitations". No K'Sante RL run. |
| MODERN-039 | Jayce (126), the thirteenth 26.19 kit (2026-10-13): `champions/jayce.py`, rules JAY.* in `docs/modern/CHAMPIONS.md`: Mercury Hammer (melee, bonus resists, empowered magic hit; To the Skies! leap to 225 short of the target in 0.5 s with a 300 slam and slow; Lightning Field 4 ticks and mana on hit; Thundering Blow root, %max-health hit, knockback, 0.4 s Q lock) and Mercury Cannon (ranged 500 with 2000 u/s missiles; shred on the next attack; Shock Blast missile with first-hit detonation and Acceleration Gate supercharge; Hyper Charge AS, damage multiplier and post-effect cooldown; gate haste), Transform with Hextech Capacitor. World: per-kit rank tables (`MAX_RANKS` / `INNATE_RANKS`, 6 basic ranks, innate R; BIN default 10 s cooldown), inactive-form cooldowns (`ChampionLayer.alt_cooldowns`, `KitOut.swap_cooldowns` / `alt_cooldown_start`), `KitOut.mana_gain`, `KitAttackMods.damage_mult`, `ItemStats.attack_range`, `attack_missile_speed` (champion ranged attacks; a missile's on-hit resolves on its own target), stance-dependent `unit_target_range`; Mercury Cannon spells pinned as Q2/W2/E2/R2. Client vs wiki resolved with 16.9 replay measurements (`/mnt/nfs/shared/champ-expansion/jayce-measurements/`, 6 games): Shock Blast 1050 / 1600 ranges (missile castRange 1300 / 2000 rejected), Thundering Blow knockback to max(475, d + 250) from Jayce over 0.35 s (client / wiki 600 rejected; 46 samples), leap duration and landing offset. Tests (Slurm cpu, desktop): test_jayce 14, test_jayce_world 4; full suite 51 non-world modules 871 passed (4697), 15 world modules 100 passed (4698), test_train 6 (4691, ~28 GB). Golden 3600 ticks, top-lane world (4692): every pre-existing leaf identical by name. | Knockback rule is an empirical fit (the client logic behind it unknown); Lightning Field tick phase INFERRED-L; Hyper Charge persisting through Transform and the third attack ending it at launch INFERRED-L; blast radii 170/250 from the wiki (unmeasurable); gate hastes only Jayce (no allies); inactive-stance cooldowns not observed; small differences in CHAMPIONS.md "Jayce limitations". No Jayce RL run. |

| MODERN-040 | Static kit pruning (2026-10-16): recovered `throughput/kit-roster` / `38be6c4e` without modifying its archived checkout. `build_config` compiles the loadouts' champions plus an optional pool (`roster="all"` restores every kit); cached roster-specific `ChampionState` types and hook dispatch omit absent kits, including observation cooldown/rank and cast-range tables. Benchmark/profile CLI accepts `--champions` / `--roster`. Observation identity width remains registry-wide. CPU source snapshots `throughput-head-TEMP` and `throughput-roster-TEMP` under `/mnt/nfs/shared/champ-expansion/` exactly match all tracked `lanerl_jax` / `ops/modern` files at `8067b443` and `38be6c4e`. `python3 -m ops.modern.golden --diff` on both saved full/top pairs: identical at all 6 checkpoints through tick 3600; strict name checks also confirm every retained leaf hash (862 full / 732 top), summary and peak counter identical, with exactly 427 absent-kit leaves removed and no new leaves. | Trajectory gate PASS for Garen/Jax full map and top lane; no GPU speedup claim. Slurm 6084: trainer 6 passed; 6086: world modules 42 passed; 6087: world/interface modules 58 passed. Non-world 6085 still running, 574 passed so far; its new roster test originally failed because the optional Jayce `damage_mult` was None versus zero, now normalized in the test only. Focused corrected validation PASS: 2 tests in 53.64 s, Slurm step 6085.0 (`thr-recovery-6085-step.out`); queued standalone 6145 cancelled before execution to reuse the existing allocation. GPU 6088 benchmark (`--exclusive`, 1h30) pending desktop availability; 6092 profile (`--exclusive`, 1h) waits afterany:6088. Their immutable NFS source snapshots and `/mnt/nfs/shared/champ-expansion/thr-*` logs retained; no duplicate GPU runs, no reboot/cancellation of existing jobs, no watcher. Distinct rosters require distinct pytrees/compilations; no checkpoint migration or training experiment. |
| MODERN-041 | Ezreal (81), fourteenth 26.19 kit (2026-10-16): `champions/ezreal.py`, pinned client record and rules EZR.* in CHAMPIONS.md. Mystic Shot first-hit missile with basic damage/on-hit/on-attack, spell effects and 1.5 s flat cooldown refunds; Essence Flux marks/detonation/mana return; Arcane Shift blink and homing bolt with mark priority; Trueshot Barrage piercing missile, epic/non-epic damage split; Rising Spell Force. Engine `KitOut.cooldown_refund` merges and applies after cooldown starts/rune refunds. Accepted cross-role roster in ROSTER.md; Ornn dropped at user direction. | Focused desktop Slurm tests 6089: Ezreal 16, champions 61, Jayce 14; 6090: Ezreal world 3, Jayce world 4; 6103: trainer 6. Golden 6091 top lane: 1159 pre-existing leaves identical by name at all six checkpoints through 3600 ticks vs golden-jayce-top-v2.json; 55 new leaves. Full regression incomplete: desktop booted into Windows, 6104/6105 requeued then superseded by 6146/6147 with fail-on-error runners and NFS cwd. After integration with MODERN-040, 6146/6147 superseded by combined 6150 (50 modules), 6151 (18 world modules), 6152 (trainer), 6153 (full/top golden compare); immutable source roster-integrated-e1b7a22c-TEMP and fail-on-error scripts in validation-TEMP. Completed: 6150 889 tests/50 modules; 6151 103 tests/18 world modules; 6152 6 trainer tests (998 total). 6153 full/top golden identical across all 12 checkpoints; strict leaf dictionaries, summaries and peak counters also identical to the pruned reference. Kit limitations (including inferred W passive timing/range and one-tick E targeting) are explicit in CHAMPIONS.md. No Ezreal RL experiment. |
| MODERN-042 | Caitlyn research (2026-10-16), no kit registered. Client/wiki evidence cached under cdragon-16.19; replay scripts and outputs in `/mnt/nfs/shared/champ-expansion/roster-measurements/`. Q timing supports 0.625 s + d/2200; R supports 0.375 s cast + 1 s channel + d/3200. Frame extraction 6106 contains 424 E records; simple plateau filter retains 309 with mixed ~350/~390 distances, so recoil distance is not settled. Old HS matching selected the first damaged enemy regardless of distance (including >10k away); those assignments cannot establish Headshot range/refunds. | `cait_frames_v2.py` bounds frame windows, rejects distant/impossible-timing HP losses and reports every plausible candidate, retaining ambiguity. Synthetic checks reject distant damage and preserve multiple candidates; this is filtering, not proof of attribution. Corrected 20-game extraction 6148 completed: 1560 Headshots; 443 single-candidate, 71 ambiguous, 1046 unmatched. Single-candidate status is not causal attribution; no exact refund inferred. Trap positions are not logged; idle movement cannot identify trap roots. Ezreal regression is complete; Caitlyn implementation waits on reliable Headshot timing evidence. Lux assessed next. |

| MODERN-043 | Training throughput harness (2026-10-16), `ops/modern/training_bench.py` + Slurm scripts. Modern parent **78abf40ff0142764a8a5ebcd2ed50ec126f5b4c7** vs lane-rl **bb2e5b466be278e7510868d1e081fb77ae33a200** (requested pin; later 3050eddc only E165 docs, dirty BC checkout untouched). Git archives, copied route/vendor inputs and per-file SHA256 manifest at `/mnt/nfs/shared/champ-expansion/modern043-pinned-TEMP/`; no moving HEAD imports. Fresh process per engine/preset/env/layout/phase, GPU required, persistent cache disabled, synchronized real collect vs full trainer update, 5 steady samples after separate init/bank-compile, explicit lower/compile and first execution. 256/1024/4096 envs, rollout128; top comparison plus modern full-map context; matched GRU normalization/residual, trunk widths, self-play2 learners, PPO lr2.5e-4/gamma.99/lambda.95/clip.2/value.5/entropy.01÷3/4 epochs/4 minibatches/no LR anneal at10Hz; natural defaults separately labeled. Actual policy/PPO/vec configs, parameter count, tick/decision Hz, fog, layout, loadouts, reset scenario, versions/device, peak allocator bytes, individual timings and OOM/errors retained in JSON/logs. Env decisions/s and sampled-agent decisions/s (2×) are separate from learner-agent decisions/s (natural old AFK1 vs modern2); game-seconds/wall-s uses env decisions × ticks/decision ÷ tickHz, never agent count. | Metric tests **4 PASS** plus OOM continuation **1 PASS** (caught/fixed duplicate `case` keyword on failed cells), CLI/bytecompile/shell syntax PASS. CPU desktop validation **6725 COMPLETED/0:0**,7m14s: both engines/presets passed real asset/bank setup plus eval_shape through PPO (envs4/rollout2/top; shape validation, not compiled PPO execution or CPU speed evidence). Matched policy params modern6,146,347 vs legacy6,163,295; modern natural MLP4,570,411. GPU matrix **6734** queued (exclusive8h) from modern043-pinned-v3-TEMP; unstarted 6729 cancelled to correct observation metadata (widths read from pinned module constants; buttons rather than heads). Five lightweight tests rerun PASS; syntax and shell checks PASS. CPU shape validation remains applicable (only descriptive metadata changed). GPU results not yet measured; cancelled unstarted **6724** to stage tested OOM fix in `modern043-pinned-v2-TEMP/`, SHA manifest references preserved original source snapshots. Unavoidable differences: old **4.20 Garen mirror**, historical map/runes/masteries/no items,60Hz,56x16 rows incl24 projectiles/self16/8 buttons, level3 120s asymmetric-HP tower curriculum vs modern **26.19 Garen/Jax**,30Hz,32x20/self56/19 buttons (7 masked),16 scripted60s hold starts, modern items/runes/wards/quest/fog; full adds three lanes/jungle/objectives. No pure-sim-to-training extrapolation or equivalent-game speedup claim. 6088/6092 submitted scripts saved: both lack exclusive/strict shell and use inaccessible archived cwd, correcting MODERN-040's claim. Cancelled only **6088/6092**; replacements **6722** sim (exclusive1h30), **6723** profile (exclusive1h, **afterany:6722**), same source snapshots/scenarios, NFS cwd/logs, fail-on-error, no duplicate old runs. Desktop back on Linux; Gres=gpu:1 yet CfgTRES/partition TRES omit GPU, also occupied by unrelated CPU work; read-only NVIDIA query sees RTX5080/driver580.173.02/18MiB used; GPU scheduling unresolved, no admin changes. No champion edits or agents. |
| MODERN-043 results | GPU6722/6723 completed: simulation before/pruned top1024 134655/148752 ticks/s (+10.5%), top4096 168324/178646 (+6.1%), full1024 42680/46855 (+9.8%). Training6734 cancelled after6h30 during final natural4096/full/PPO cell; completed cells retained in modern043-pinned-v3-TEMP/results-6734/results.jsonl. Matched10Hz top1024 rollout+policy: modern36021 vs legacy21738 environment decisions/s (+65.7%), respectively72041/43476 sampled-agent decisions/s; matchedtop4096 modern35760 (no aggregate scaling benefit). Natural moderntop1024 rollout36321 and4096 39349; full1024 13760. | Full PPO memory barrier: all completed matched PPO cells OOM, including256; modern natural PPO OOM at256/1024/4096. Legacy natural256 PPO7033 env decisions/s (one learner/AFK, not matched to modern2 self-play), peak6.25GB. Modern matched256 allocation request10.40GiB; natural2567.26GiB, under allocator limit~12.44GB on16GB GPU. Cannot conclude matched end-to-end training speed or safe training batch size. Source/error logs preserved; focused memory diagnosis delegated MODERN-045; no complete matrix pass claimed. |
| MODERN-044 | Lux assessment (2026-10-16), no kit registered. Client root records, child missile records and Rift tooltips inspected in cdragon-16.19/lux.bin.json and lol.stringtable.json. Root R has spellCastTime 1.375, but the replay-recorded child LuxRMis has 1.0. W missiles explicitly decelerate outbound (2200, -2300, floor 500) and accelerate on return (50, +1800, cap 2400), so a fixed-speed boomerang would lose behaviour. E has a five-second recast zone, one-second slow linger and sight; Rift passive tooltip says R consumes then refreshes Illumination. | Replay extraction 6707 completed on desktop CPU: eight games, 519 Q, 307 W, 750 E and 135 LuxRMis records. Reproducible candidate analysis: roster-measurements/lux_timing.py + lux-events-6707.out. 70/135 R casts have >=100 HP loss within 0.1 s of 1.0, versus 8/135 near 1.375; 125/519 Q casts have >=60 HP loss consistent with 0.25 s + (distance - assumed 65 radius)/1200 within 0.1 s. These are candidate associations, not attacker-attributed hits; no damage-ratio conclusion. Wiki requests were blocked; primary pinned client tooltips retained as evidence. Need settle exact Q range/cooldown and W return/shield semantics before implementation; no guessed kit shipped. |

| MODERN-045 | Full-PPO GPU OOM diagnosis (2026-10-16), reserved separately from parent's MODERN-043 results. Existing immutable modern043-pinned-v3-TEMP/results-6734 has 35/36 cells; no matrix rerun. Source inspection identifies whole-epoch shuffled observation copies, recurrent attention residuals and full-rollout post-update KL as candidates, not proven causes. Diagnostic worker now emits compiled argument/output/alias/temp bytes and estimated execution footprint before first execution, allocator settings, live-array summary/device stats; optional isolated full-batch KL-forward/minibatch-forward/minibatch-backward compilation and device memory profile. Bounded driver covers modern matched64/128, legacy matched64, modern natural128; modern matched/natural256 each at .75/.90 allocator fractions, rollout128, three real steady updates, fresh process/case, 12-minute cell timeout, 1h45 exclusive desktop GPU job. | Eight lightweight tests PASS, including timeout/OOM evidence retention and allocator isolation; shell syntax/diff checks PASS. Learner/simulator unchanged, no semantics/trajectory claim needed and no fix/speedup claimed. GPU occupied by unrelated7433 (end currently2026-10-17 06:28UTC); no cancellation, admin changes, watcher or agents. GPU job/source pin recorded below on submission. |
| MODERN-045 submission | Hash-verified immutable `/mnt/nfs/shared/champ-expansion/modern045-memory-baseline-TEMP/` (4301 files), original engine pins plus diagnostics commit **576bc70d**. CPU component compilation test covers real MLP and GRU forward/backward layouts without executing a rollout; submitted **7587**, desktop CPU2/mem2GiB/time10min. Exclusive GPU **7588**, time1h45, afterok:7587; results target `results-7588/`, logs `champ-expansion/modern045-{components-7587,ppo-memory-7588}.out`. | Both pending (CPU resources/GPU dependency and desktop occupied). Cancelled only own unstarted **7584/7585** after snapshot assembly failure; completed snapshot and verified every hash before replacement submissions. Eight lightweight tests PASS; real compilation test pending Slurm, no new measured GPU throughput/peak or code fix. Next inspect attribution/allocator cases, then test a justified learner change against unchanged source. No simulation edits or further agents. |

| MODERN-046 review | Isolated modern-package review (2026-10-16), base `af89e4e4`; memory-diagnostics files untouched. **P1 fixed:** `world/phases/attack.py:166` conflated a new launch and an older missile's hit. A simultaneous launch at another unit redirected landed hooks; ordinary arrivals supplied raw0/critfalse, suppressing Profane Hydra Cleave. Launch/hit records now stay separate through kit/item/rune hooks, with landed target/raw/crit/cast ID from the missile. **P2 fixed:** `combat.py:255` now propagates both sequential resolution overflow counts; previously65 champion-directed packets resolved64 while the aggregate counter falsely stayed0. | Desktop CPU red7597 reproduces missing W detonation and hidden overflow (2 failures,27.03s). Split-process7606 COMPLETED/0:0 in11m42s:104 focused checks (Ezreal/Jayce world, combat, item framework/runtime/defense/marksman, Domination). Supplemental7611 COMPLETED/0:0: parent ordinary arrival fails specifically at missing Profane Cleave, both fixed arrival cases pass (14.83s red /14.62s green). Immutable sources under `/mnt/nfs/shared/champ-expansion/modern046-review-TEMP/` (`red`, `green-v2`, corrected `red-v3`/`green-v4`); NFS logs `modern046-{red,green,ranged}-JOBID.out`. Earlier7602 failed before Python on sh/pipefail;7604 reached58 passes then exceeded8GiB with accumulated compiled programs;7608 exposed the supplemental fixture's missing visibility, corrected in7611. No unrelated jobs changed or new GPU runs. |
| MODERN-046 cost | **P2 targeted refactor:** profile6723 (`thr-profile-6723.out`, roster trace JSON/XPlane under `roster-profile-TEMP-6723`) attributes21.87% /1.5191ms per1024-env tick to `core/damage.py`. The global shield-order max at original line185 accounts for7.81% /0.5425ms across all callers, not exclusively the sequential loop. `resolve` now finds the next global grant order once per pass and carries it, advancing only for positive grants, removing the full-unit reduction per sequential packet. Trimmed24 net stale test-environment lines, displacement-refactor history, and the obsolete32-wide observation self-block description. | **Fidelity PASS:**7606 full/top3600-tick goldens match `validation-TEMP/golden-integrated-e1b7a22c.json` at all12 checkpoints. Strict record equality confirms leaf names/hashes (862 full /732 top), summaries and every peak counter, all overflow0. Results `modern046-review-TEMP/golden-green-{full,top}.json`; final Python executable ASTs match validated snapshots (subsequent doc/whitespace edits only). Operation attribution uses disabled command buffers; savings and GPU throughput remain unmeasured. **Unfixed structural limit:** `world/phases/attack.py:185` replaces a basic hit with an ability hit when both land together (e.g. Ezreal Q plus a basic attack at another unit); multiple basic-missile arrivals also collapse to one event. Proposed follow-up: dispatch each landed hit independently through existing hooks with per-hit metadata and separate launch bookkeeping; not implemented or regression-validated here. |
| MODERN-047 simultaneous hits | **P1 fixed**, parent `76a4d59f`: compiled-world7619 proves two basics damage their targets while only the first gets Recurve Bow. Packed holder records preserve every landed basic/ability target/raw/crit/cast ID. Hooks fold direct hits, missile slots, then the ability hit; actual launches stay independent. Extra kit/item/rune hooks run in dynamic loops only for additional landings, accumulating bounded packets with counted drops; combat resolves once. Ezreal/Fiora/Yone supply ability metadata; Red Buff consumes all landed targets via a sparse mask. | **139 unique focused checks pass** across7644/7650: Ezreal world12, combat9, jungle19, Jayce world4, Fiora16, Yone19, marksman19, Precision41. Covers same/different-target basics, Q+basic, each with/without a new launch; Recurve/Profane procs, mark detonation, crit cooldown changes, launch counts, padding, third-hit PTA and compaction overflow. Initial green7620 passes6 cases. Final7650 COMPLETED/0:0 in10m58s. Own intermediate7626/7632 and unstarted7633 superseded;7637 found the new item's zero-stack fixture,7644 a unique-target helper used as a packet counter; corrected assertions pass. Sources/logs: `/mnt/nfs/shared/champ-expansion/modern047-hits-TEMP/`, `modern047-*.out`. |
| MODERN-047 fidelity / cost | **PASS:**7650 full/top3600-tick goldens match `validation-TEMP/golden-integrated-e1b7a22c.json` at all12 checkpoints. Strict complete-record equality includes862/732 leaf names/hashes, summaries and peak counters; overflow0. `golden-v6-{full,top}.json`; all116 implementation files byte-identical across cost snapshot `green-v4`, world checks `green-v5` and final `green-v6`. Compiler comparison7638 COMPLETED/0:0 in2m32s: restricted-item top-lane CPU, fog off, no runes, vs Garen; input/output bytes unchanged. Temporary bytes Garen338168→344120 (+5.8KiB), Ezreal451960→432752 (−18.8KiB); Ezreal estimated FLOPs+0.39%, bytes accessed+0.10%. | Static compiler estimates, **not runtime/GPU throughput**;7433 occupies the desktop, so no overlapping timing. Ties follow direct→missile-slot→ability order; existing missile/packet capacities and per-holder item/rune launch reservations remain. Current on-hit hooks grant only Camille's single latched shield; assertions guard shield-fold contracts. No GPU job, memory-diagnostic edits, unrelated cancellation, or active own watcher/job. |

| MODERN-049 shield valuation | 2026-10-17, parent implementation `0e54c2d3` (checkout `05562988` differs only in docs). `core/damage.py` selects the target shield row before `shield_value` in grants and both sequential-packet valuations; arithmetic, slot choice, source replacement and order counter are unchanged. Desktop CPU compiler7791 COMPLETED/0:0: baseline optimized grant HLO retains decay division/multiply/minimum on `[216,6]`; candidate values only `[6]`. Standalone216-unit grant with supplied order: estimated FLOPs16964→194, accessed bytes114446→73042, temporary bytes5392→272; resolve: FLOPs60728→15520, bytes266822→145201, temporary18672→12656. | Restricted-item top-lane CPU tick, fog off/no runes, Garen/Ezreal vs Garen: estimated FLOPs13299535→13144847 (−1.16%) /13880361→13700637 (−1.29%); bytes635530624→635141760 /637454080→636995840; argument/output/temporary memory unchanged. **Static estimates, not runtime/GPU speedups.** Immutable base/green sources, submitted scripts and optimized HLO: `/mnt/nfs/shared/champ-expansion/modern049-shields-TEMP/`; log `modern049-cost-7791.out`. No GPU job or timing measurement; parent owns those. |

| MODERN-049 fidelity | Desktop CPU7792 COMPLETED/0:0 in17m41s: **211 checks pass** (combat9, item framework23, item runtime3, defensive items21, marksman19, Malphite25, Ezreal world12, jungle19, Jayce world4, Fiora16, Yone19, Precision41). Existing Yone float64→float32 scatter warning remains. `python -m ops.modern.golden --compare` passes full/top3600 ticks, six checkpoints each, against `validation-TEMP/golden-integrated-e1b7a22c.json`; strict complete-record equality confirms all12 checkpoints, including862/732 leaf names/hashes, summaries and peak counters; overflow0. | `modern049-shields-TEMP/golden-{full,top}.json`, log `/mnt/nfs/shared/champ-expansion/modern049-validate-7792.out`. Tested candidate damage source byte-identical to the committed file; only that implementation file differs between immutable base/green snapshots. No gameplay/parameter changes, PPO/policy/tools edits, GPU execution (timed or untimed), unrelated cancellation, service or watcher; no own active job. GPU compiler/runtime effects remain for the parent measurement. |
| MODERN-049 integration / GPU measurement | Parent integrated as `38d6d04e`; independently checked tested damage file byte equality and strict equality of all12 complete golden records. MODERN-048 CPU7793 completed0:0: all5 tests and compiled tiny ability-order CLI pass, overflow0. Exclusive GPU7797 queued afterany:7795, using the shared five-case sim-only driver over80bd8f0c and38d6d04e. | Immutable comparison and manifest: `/mnt/nfs/shared/champ-expansion/sim049-throughput-TEMP/`. Existing pre/post-review comparison7796 and profiler7795 remain queued. GPU throughput effect remains unmeasured; no PPO/policy changes. |



| MODERN-048 | Simulator ownership/measurement (2026-10-17): user assigns this thread sim/champions/correctness and sim speed in training; PPO/policy/learner belong to other agents. Stopped own running diagnostic7588 (final elapsed21m20s), preserving partial MODERN-045 results for handoff; no learner changes. `bench` now reports median of synchronized samples, tick/game-time throughput, actual timed game interval, minion occupancy, process peak memory and aggregate overflow; frozen games or overflow fail timing. Optional deterministic staggered ability attempts retain ordinary lane orders. `perf_guard` separates ability workload baselines while preserving old lane keys. New exclusive Slurm sim comparison/profile script supports later combat after5400 warm ticks. | Four lightweight timing/key tests pass locally; shell syntax and diff checks pass. Compiled order/CLI validation7793 queued; exclusive comparison7796 afterok:7793 and profile7795 afterany:7796. Superseded own unstarted7794 to use the supported top-lane Jayce/Garen ranged loadout. Immutable sim048-throughput-TEMP baseline78abf40f/candidate80bd8f0c use identical overlaid driver SHA256; cases include early lane control and later ability workloads, top1024/4096, full1024 and Jayce/Garen1024. Same driver is overlaid on old/new snapshots; no speedup claimed and no policy inference/PPO cost included. Damage-path optimization separately delegated MODERN-049. |

| MODERN-050 baseline identity | 2026-10-17: `ops/modern/perf_guard.py` previously omitted packet capacity, item allow-lists, lane-structure reduction, champions and roster from its baseline key despite accepting those world flags. Distinct scenarios could reuse a baseline and misattribute their cost difference. Keys now append each non-default setting; existing default keys remain unchanged. | Five lightweight summary/key unit checks pass on CPU, including independent keys for every omitted variant. Tool-only change; no sim refactor or runtime speed claim. Jobs7796/7795/7797 remain pending for GPU resources/dependencies; no unrelated jobs touched. |

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


| CHAMP-004 collector wiring | Explicit `--modern-champions BLUE,RED` on server_train/jax_train, Garen/Jax any pairing; modern stats/state/28-feature policy inputs, Jax rank order, generated bare-champion config, single/worker collector/reset propagation. Checkpoint reload rejects ordered-pair or width mismatches. Legacy defaults remain16-wide and reject modern wire. `LANERL_VENDOR_ROOT` now resolves the runtime/vendor from worktree snapshots. | LIVE opt-in; frozen evaluation through server_train uses the same collector. No PPO experiment launched. Older server_eval/parity-recording path remains legacy; ordinary ops/launch server/JAX modes still target canonical checkout, requiring branch integration before a production launch. |
| CHAMP-004 wire contract | Schema2 exposes own resources, buffs, Jax pending R, stun/silence/casting, accepted-cast counters. Enemy cast memory is gated by prior-frame authoritative visibility/viewport; Jax W consumption/E expiry cannot create a fake cast. CC booleans become positive observation-gate sentinels, not measured remaining duration. | Observation reconstruction only, not a resumable C# simulation snapshot. Historical world and CHAMP-003 spatial/vision omissions remain. |
| CHAMP-004 legacy checks | Capped CPU modern/collector/server_train/policy_driver subset:37 passed,6 failed in109.78s. All6 failures also reproduced with pre-change HEAD policy_driver.py loaded (6/6 in2.87s): old test-policy doubles have no cfg attribute, while existing recurrent code reads policy.cfg. Logs `/mnt/nfs/shared/MOD009-python.log`, `MOD009-baseline-tests.log`. | Existing test-fixture failures, not introduced by champion wiring; no unrelated fixture rewrite. Modern desktop gate reported separately. |
| CHAMP-004 final validation | MOD009/1817 COMPLETED/0:0 in1m48s:58 targeted Python checks (including JAX collector profile/reset),21 C# combat assertions,180s scripted lane,12 frozen4s agent episodes through the normal source collector. MLP Garen/Jax one-worker and GRU Jax/Garen two-worker checkpoint reload/reset gates pass; no collector ERROR logs. DLL SHA256 `0451c56fb0a7559de357604c734127129bdc7ba65e008bfc95f7e30831878b79`; patch dry-run passed. `/mnt/nfs/shared/MOD009_modern_collector/collector-result.json`, source snapshot and launch.json retain evidence. | Wiring validated, not a performance or balance result (random tiny policies,0CS). No PPO experiment. MOD008 failure retained in EXPERIMENTS; shared vendor unchanged. |


| BC replay-only hold | 2026-10-22, thread 200b87b6: source audit in `docs/modern/BC.md` section 10 finds sim-derived fog, structures, cooldown/descriptor normalization, summoner/recall/XP/skill mechanics and quest/trinket defaults in the BC formatter. Removed own-spell table fallbacks (including legacy Garen rebuild) and kit/catalog targeting rules; received countdown gaps stay unknown, with causal cast invalidation. | 17 focused local CPU checks pass. Conversion/relabel/finalize/load entrypoints now refuse the old mixed-source path before IO/cache/skip-existing. No data deleted, reconvert or training launched. Team fog unavailable in current decoder; a mask-aware observation contract is still required. This is a hold, not a certified replay-only dataset or a sim fidelity change. |

| BC heuristic fog | 2026-10-22, thread 200b87b6: Dani allows rough fog as an explicit exception. `bc.fog` computes circular team sight from current live replay allies, with 1350/1200/1350 champion/minion/turret radii, and records approximate provenance. Converter visibility no longer calls sim vision. | 21 focused local CPU checks pass. Walls/brush/wards/stealth/reveals remain omitted; allies outside policy slots supply sight, template-only targets do not. Other BC source holds remain. CPU probe12794 submitted with registered receipt to inspect countdown alignment/available fields; no reconvert or training started. |

| BC countdown source qualification | 2026-10-22: probe12794 COMPLETED/0 shows36/36 selected casts have separate cast-start schedules ignored by the BC reader; all sampled schedules have unqualified clock alignment, and nearby925 operands explicitly mark timer/slot semantics unvalidated. Reader now joins aliases to packet evidence by actor/source event/clock and requires timer+slot qualification; decoder ancestry is no longer sufficient. | 25 focused CPU checks pass. Cast schedules inventoried without asserting exact readiness; historical400-game percentages remain converter coverage only. Real-export reader validation12796 submitted/registered, output under `/mnt/nfs/shared/modern-bc/`. No reconversion or training. See BC.md section12. |

| BC remaining field defaults | 2026-10-22: source-check12796 COMPLETED/0 on all ten actors in one export:3399 unqualified broadcasts,2377 separate unqualified-clock cast schedules. Held formatter now takes matched turret positions/life/HP from replay without template/full-health defaults, reconstructs summoner slots only from qualified received timers, and leaves recall channel unknown instead of applying sim duration. | 27 focused local CPU checks pass, including empty template structures, unknown HP, summoner key mapping and unknown recall. No reconversion/training. Normalization/descriptors, progression/quest/trinket, reader gaps/causal positions and policy missingness remain under the source hold. |

| BC v4 replay profile | 2026-10-22: direct replay observation builder replaces world initialization/kit stats. Missing scalar/rank evidence and unsupported descriptors/progression remain NaN; policy learns explicit missingness and masks columns never observed in BC during RL. Fixed timer scaling, qualified sources, causal positions, visible-target filtering and versioned cache/manifest checks. | CPU12851 COMPLETED/0:41 BC checks;16 pilot windows across six champion/role cases yield6285 train/2330 val valid decisions with preserved base-game split and successful finalized cache loading. Additional lightweight checks30/30. GPU memorization and rollout acceptance are next; no full v6 run or sim behavior claim. BC.md section14. |

| BC v4 evaluation assertion | GPU diagnostic12858 FAILED/1 before optimizer start:42 CPU checks passed, one test expected the former evaluation bucket set. Fixture/assertion now validates champion/role groups and joint click accuracy bounds. | No training result. Re-submit after this test-only fix; diagnostic data and source implementation unchanged. |

| BC v4 memorization pass | GPU12859 COMPLETED/0:43 tests;512 steps, loss10.399→1.405, casts34/35, click distance54.85→3.03, exact clicks60.5%. | Fixed training windows only. Before rollouts, v4 sim tier default corrected to0 to match BC; old profile defaults preserved. Gate adds lagged-pose alive base fraction. Next: joint targeting/held-out metrics and Garen/Jax rollouts; economy baseline still has buying masked. |

| BC evaluation serialization | Rollout attempt12866 FAILED/1 after43 passing tests, before any sim episode: offline evaluation report contained a NumPy float32. Public evaluation now converts scalar leaves to native numbers; the existing evaluation test now asserts JSON serialization. | Diagnostic checkpoint/data unchanged. Retry offline/rollout checks; no sim performance result yet. Storage preflight found165GB free on desktop scratch and718GB on NFS; do not stage an oversized full cache. |

| BC v6 rollout retry/storage | Replacement12869 registered on gpup after JSON-scalar fix;43 prior preflight tests passed, but12866 ran no sim episodes. Desktop has165GB free NVMe. Pilot extrapolation over58470 champion-games (~473.6M decisions) estimates current cache502.6GiB; omitting always-missing self columns and losslessly packing categorical entity rows estimates105.1GiB at unchanged numeric precision. | Await actual12869 outputs. Compact cache still needs implementation/equality validation before full training. No deletion or full-corpus run. |

| BC pilot12869 and scratch lifecycle |43 checks passed; fixed targeted joint9/19, held-out button74.6% vs82.5% majority, no observed cast recall. Gate failed on nonexistent Transition.self_vec; corrected to obs_self. Obsolete474GB cache reclaimed with Dani authorization and no active jobs;638GB now free. | Tiny memorization checkpoint is not generalized. Retry episodes from isolated code/cache with exit cleanup; full training unstarted. Host scratch guidance updated; no periodic service installed. |

| BC rollout12873 | COMPLETED/0;43 checks;8 Garen+8 Jax episodes at100ms observation lag,540s against AFK. Mean CS2.25/1.25, deaths0.25/1, base0.31%/1.40%, overflow0. Private scratch removed. | Integration passes; tiny memorization model is poor and held-out casts failed. Buying remains disabled. CPU12879 measures full-length representative conversion/cache sizing before scale-up. |

| BC full-length staging12879 | Failed before conversion because cp archive-mode permission preservation is unsupported on scratch; EXIT cleanup verified. Metadata-free independent copy validated across544 decoder files by SHA256. | Replacement CPU12880 registered,40min/16GB; verifies staged hashes. Conversion/cache sizing remains pending; no BC semantic change. |

| BC archive provenance12880 | Decoder staging verified544 files; conversion then failed because provenance assumed `.git`. Added explicit archive full-SHA record support with missing/malformed record rejection;10 focused converter checks passed. | No saved full-length game/sizing result. Retry from a newly pinned archive, full BC preflight first; observation/label semantics unchanged. |

| Full-length12882 plan metadata |44 BC tests passed;8 games converted,2.9–5.7s each. Finalize failed on missing vocabulary in temporary sizing plan. Worker now copies complete source-plan metadata and validates consumed keys/vocabulary/splits before conversion. | CPU12883 registered; same converter6b20bdcf, no semantic changes. No finalized dataset/cache sizing result yet; scratch cleanup verified. |

| BC full-length12883 and conversion12886 |12883 COMPLETED/0:44 tests,8x8100 decisions,40799/14414 valid train/val, finalized/cache-loaded.2.9–5.7s/game,2.34GiB max RSS; full-cache extrapolation475.9GiB fits currently638GB free with reserve. | Full replay-only CPU array12886 registered:58470 champion-games,8 shards max2 concurrent,12h/10GB each, source6b20bdcf, original split. No broad training/PPO until full validation; economy gap separate. |

| Full conversion12886 complete / BC-only scope |8/8 shards COMPLETED/0,58470 NPZ/JSON pairs, no failures. CPU audit13005 finalizes and checks full arrays/contracts/split separation. Dani directs proceeding with heuristic fog without decoder wait; RL/PPO removed. | Opt-in within-epoch held-out monitoring and plateau/nonfinite stopping implemented; actual training smoke required before full run. Decoder v18 remains partial, reserved for later integration. |

| BC monitoring137e30b3 and queued training | Group-spread sampling regression passes. GPU13007 tests all45 checks and real normal/forced-stop training paths. Full GPU13009 depends afterok on audit13005+smoke13007 and passed receipt files. |3 epochs ceiling,500-step held-out512 monitoring, event+click plateau patience4; unknown cooldowns unsmoothed, standalone BC dropout0. No actual full-run result yet; native FoW deferred by Dani, no RL. |

| BC monitor smoke13007 passed |45 tests; real normal training64steps/four interim evaluations and saved sampled best; forced plateau stops4steps/two checks. Audit13005 metadata finalization yields52590/5880 train/val perspectives,340720945/38055900 valid labels. | Workflow pass only. Full-array scan2000/341s, audit limit extended4h; full train13009 still queued behind audit. Counts are7309 unique matches, not58470 matches. |

| BC full audit13005 passed / training13009 active |58470 perspective arrays from7309 unique26.19 matches audited,473607000 decisions,340720945/38055900 valid train/val.44.24GB NPZ,5629s audit. | Both prerequisite receipts passed;13009 running in staging/cache preparation, no optimizer output yet. Standalone heuristic-FoW BC; no RL/PPO. |

| BC13009 requeue staging failure | Reached10000 steps, sampled event+click NLL7.398/accuracy91.23%/distance12.71, cast recall weak. Slurm Restarts=1; second attempt capacity guard found prior503.8GB cache. NFS checkpoint10000 intact. | Resume must skip completed batches and retain total3-epoch ceiling. Wrapper cleanup/requeue recovery is being repaired; no reconversion or RL. |

| Resume retry13011/13012 | Source104b05b8 skips completed batches and RNG draws within original epoch budget. GPU13011 verifies old4step→64total path.13012 queued afterok, resumes checkpoint>=10000, recovers terminal13009 data via independent copies/progressive reclamation and validated cache path relocation. | Append logs, signal forwarding/wait and own-orphan cleanup repair wrapper failure. Actual resume pass pending; cast quality still weak, no RL. |

| Resume13011 pass /13012 accounting failure |45 tests and4→64total-step resume pass.13012 failed before touching data: compute-node sacct localhost6819 unreachable. Exact replacement check (login-recorded terminal receipt + live squeue absence + donor owner) passes on desktop. | Retry13013 registered; source104b05b8, checkpoint>=10000, unchanged recipe. Cache/checkpoint intact, no reconversion/RL. |

| BC13013 finished / quality insufficient | Early stop38500, best sampled36500, event+click7.1787; sampled accuracy91.32% vs82.15% majority, Q/W/E/R argmax recall7.54/1.43/4.92/0.40%, weak joint targeting. Full epoch2 accuracy91.17%, click distance12.56. Scratch cleaned. | Standalone BC goal not met. GPU13014 scores best36500 on full held-out set with action-probability and joint-target calibration diagnostics before more training. No RL. |

| Full best-checkpoint13014 |5880 held-out perspectives,33.23M scored frames; accuracy91.34%, click distance12.37. Q/W/E/R sampling rates~6–7x labels despite weak argmax recall; joint near-target probabilities<1%. | Increasing cast weights is not justified. Small frozen-policy action-conditioned aiming vs shared-head ablation plus independent calibration evaluation next; default policy output behavior preserved. No RL/native-FoW dependency. |

| Controlled readout/calibration13015 | Registered GPU20GB/4cores/1h;46 tests precede frozen36500 readout extraction.128train/64calibration/64assessment perspectives; calibration/assessment replay bases disjoint. Shared vs action-conditioned heads use same1000 balanced batches. | Oracle-button aiming diagnostic, not end-to-end policy success. Weighted-probability correction tested separately. No production action or broad retraining change yet. |

| Readout13015 mixed / target audit13017 |46 tests. Independent assessment calibration NLL0.373→0.291, Q rate4.19→0.686% vs0.696% human, but cast argmax nearzero. Conditional aiming Q/E distance improves; near-target/NLL mixed. Summoner/ward100% within2 suspicious. | Audit13017 compares raw points/vectors vs caster/label cells;13016 import failure corrected. No production promotion or new full run yet. |

| Target origin bug13017 |26 perspectives/228 casts;224 selected0xd0 points,217 near caster (<50units), center-clustered labels, no resolved points.0x2c not qualified as replacement. | Remove slot4+origin fallback; new conversion target contract, old-array loader masks summoner/trinket targets without changing observations/noops. Readout gains on those labels invalid. Corrected probe next; no full reconversion. |

| Target-guarded probe13018 | Source63c8d575 removes origin fallback and masks ambiguous historical targets at load time;48 tests precede same256-perspective controlled experiment. | Observations/cache reusable, no23h rebuild. New aiming/calibration results pending; historical summoner/ward gains are discarded. |

| Corrected13018 / next pointer diagnostic |48 checks pass; calibration improves, but conditioned frozen readout has mixed aiming metrics and is not promoted. | Preserve real target IDs (including held attacks), reconstruct lagged slot identity on small subset with observation parity checks, test entity pointer vs baseline. No guessed target supervision/full conversion/RL. |

| BC-V6-POINTER-TARGETS | 2026-10-24 | CPU13019 registered, sourcebc6b8e14: reconstruct exact received unit IDs and lagged entity-row mapping for128 train/64 assessment perspectives. Requires BC suite, decoder hashes, matching existing observation arrays and guarded action labels. Coverage report and sidecars precede learned pointer probe; no GPU/full retraining launched. |

| BC-V6-POINTER-COVERAGE | 2026-10-24 | CPU13019 COMPLETED/0:48 tests,544 frozen decoder hashes, all192 original lagged entity/padding and accepted action labels match.85,795/85,872 exact unit IDs visible (58,908 train/26,887 assessment);77 absent excluded.952s reconstruction. `pointer-targets-13019/results.json`. |
| BC-V6-POINTER-PROBE | 2026-10-24 | GPU13020 registered, sourcebc6b8e14/worker79ecc87a4643. Frozen best36500 features, entity-only/contextual unit pointers,1000 balanced-button minibatches,128train/64assessment perspectives; nearest/first/screen-snap controls. Oracle button/unit type only a targeting diagnostic. gpup20GB/4cores/1h cap, private scratch and cleanup. |

| BC-V6-POINTER-RESULT | 2026-10-24 | GPU13020 COMPLETED/0,55.2s. Contextual pointer73.67% unit accuracy vs screen-snap66.75/entity-only65.14 on23724 heldout clicks; event70.78 vs60.70 on7066. Screen within2 74.59 vs68.46%, distance3.72 vs5.08. Q/W/E improve; R regresses on31 cases. Oracle button/type, no production promotion. `pointer-probe-13020/results.json`. |
| BC-V6-HYBRID-ROUTER | 2026-10-24 | GPU13021 registered, sourcebc6b8e14/workerb47a0f18024a. Frozen base/pointer, learn unit-ground router on128 train;512 fresh assessment perspectives exclude previous calibration/test bases, cap1024 uniform clicks/game. Evaluate ground degradation, unit gains and joint actual-button/click accuracy. gpup20GB/4cores/1h; no broad retraining/RL. |

| BC-V6-HYBRID-RESULT | 2026-10-24 | GPU13021 COMPLETED/0,71.6s. Reject hard router: actual-button joint near-click23.718→23.638%, ground near21.213→21.057% on524288 clicks/512 fresh perspectives. Unit recall19.38%, false-unit ground2.71%. Conditional unit-pointer gain does not survive routing. `hybrid-pointer-13021/results.json`. |
| BC-V6-JOINT-CLICK | 2026-10-24 | GPU13022 registered, sourcebc6b8e14/workerbab0037e5cf9: paired factorized versus joint96x54 residual click heads,128train/64selection/512 fresh assessment perspectives. Same Gaussian/held weights and minibatches,2000-step cap, selection every500/patience3 including baseline. No unit-ground routing or production changes. gpup20GB/4cores/1h. |

| BC-V6-JOINT-RESULT | 2026-10-24 | GPU13022 COMPLETED/0,99.5s. Frozen factorized/joint residuals both choose step0 under prespecified soft CE; no improvement promoted. Joint exact selection NLL improves but soft CE worsens; smoothing also applied to unit clicks in this probe, unlike native unit sigma0. Assessment remains unchanged baseline. `joint-click-13022/results.json`. |
| BC-V6-ENDTOEND-CLICK | 2026-10-24 | GPU13023 registered, sourcebc6b8e14/workerd841bb67ce9e. Full-gradient continuation control versus joint-click fine-tune;1024train/64selection/240assessment (30 bases), native target losses.1000 steps/arm, batch8, eval200/patience3/baseline0; base3e-5/head1e-3. Native loss equality and actual base-gradient/update checks required. gpup24GB/4cores/1h; no full corpus/RL/production promotion. |

| BC-V6-ENDTOEND-CONTROL | 2026-10-24 | GPU13023 FAILED4m10s after completed control, before candidate. Native loss match passed, base gradient5.404/delta0.0620; control stops600, selects baseline0 (initial6.84000 vs6.85434/6.85305/6.87758). JSON float32 reporting error; saved control checkpoint and logged metrics preserved in `endtoend-control-13023-recovered.json`. |
| BC-V6-ENDTOEND-REPAIR | 2026-10-24 | GPU13024 registered, sourcebc6b8e14/worker5fb84526f151. Trains unfinished joint arm only, reuses completed control with parameter/assessment agreement checks. JSON scalar normalization plus real NumPy nested serialization check passes; validate metric tree before optimization. Same1024/64/240 perspectives and1000-update cap; gpup24GB/4cores/1h. |

| BC-V6-CONTROL-REPRO | 2026-10-24 | GPU13024 FAILED1m09s before candidate: identical saved parameters and continuous losses pass, hard within2 differs16/1127853 clicks (0.00142 percentage points), exceeding overly tight common tolerance. Exact cause unproven. GPU13025 registered, sourcebc6b8e14/worker000bd4769814: exact parameter/count checks, tight loss tolerance, hard rates abs1e-4/distance0.01; persist full deltas and check fixed-batch repeat. No repeated control training; same bounded joint arm. |

| BC-V6-ENDTOEND-RESULT | 2026-10-24 | GPU13025 COMPLETED/0,316s, native-loss/full-gradient checks pass. Step800 joint click NLL6.16265→6.07386, joint near23.636→23.916%, but Q/W/E recall and NLL regress. No promotion. Every selection checkpoint has E NLL increase>0.05; add per-cast guard to prevent aggregate masking. |
| BC-V6-HISTORY-PREP | 2026-10-24 |192 metadata audit: own QWERDF cooldown availability all zero. Preparing exact received-cast elapsed seconds sidecars with canonical slots/lag, distinct from cooldown/readiness, plus causal tests. No production width change; CPU validation/source audit pending. |

| BC-V6-CAST-GUARD-HISTORY | 2026-10-24 | CPU13026 registered, source5450052a/worker98bf4406aa4e. Full BC tests + actual13025 cast-regression rejection,544 decoder hashes and192-perspective received-own-cast age source audit. Age is exact elapsed receipt time, not cooldown/readiness; lag preserved, NaN before first receipt.2CPUs/8GB/1h; no production observation change or GPU training. |

| BC-V6-CAST-HISTORY-AUDIT | 2026-10-24 | CPU13026 COMPLETED/0:53 tests, real13025 cast guard rejection,544 decoder hashes,192 age sidecars/771s. Among known heldout histories, QWERDF age>25.6s shares25.7/44.0/40.6/74.5/93.3/90.9%. Availability evidence only; no predictive claim. |
| BC-V6-HISTORY-READOUT | 2026-10-24 | GPU13027 registered, source5450052a/worker3e43a24fe04d: paired bias/context/context+age readouts on frozen base.128 train perspectives split by replay-base hash for fit/selection,64 previously assessed perspectives. Natural batches, inverse-weight initial bias, event NLL + cast guard,2000 steps/eval250/patience4; shuffled-age falsification. gpup20GB/4cores/1h; no production input change or broad retrain. |

| BC-V6-HISTORY-READOUT-RESULT | 2026-10-24 | GPU13027 COMPLETED/0,82s. All three heads choose baseline0; trained candidates worsen event NLL and supported QWER. Assessment0.300799 merely inverse-weight baseline; shuffled history uninformative at zero offsets. All-frame training/event-only selection mismatch limits negative conclusion. No history/input promotion. |
| BC-V6-HISTORY-CURVE | 2026-10-24 | GPU13028 registered, source5450052a/workerea5ba252f569. Bias versus98-parameter received-age-bin calibrator; same99fit/29selection/64assessment, matched held0.3 training/selection CE, cast guards,2000/eval250/patience4. Empirical age-bin QWER counts and shuffled-age falsification. gpup20GB/4cores/1h; no base or production input change. |

| BC-V6-HISTORY-CURVE-RESULT | 2026-10-24 | GPU13028 COMPLETED/0,65.8s. Selected1000: weighted NLL0.403222→0.402078, event NLL worsens and heldout R4.7248→4.7759 exceeds0.05 guard. Shuffled ages retain most gain; age-specific weighted gain~0.00048. No promotion or scale-up. |
| BC-V6-REPLAY-AUDIT | 2026-10-24 | GPU13029 failed first clip on packed-loader scalar indexing; fixed slice plus actual loader check. GPU13030 registered, source5450052a/workerce8a2b20dcd3, gpup8GB/4cores/20min:8 deterministic replay-only Garen/Jax/Ezreal/Lux human/model prediction clips. No simulator, training or closed-loop claim. Visual audit follows receipt. |

| BC-V6-REPLAY-AUDIT-RESULT | 2026-10-24 | GPU13030 completed8 deterministic clips. Desktop/mobile themed previews and actual selector/scrubber checks pass. All8 center raw argmax presses choose Move;2 targeted casts near,3 aimed casts far. Illustration, no fresh-test or closed-loop claim. `replay-behavior-13030/replay-comparison.html`. |
| BC-CLEAN-RETRAIN-PREP | 2026-10-24 | Prepare fresh baseline with repaired target masks, unsupported D/F/ward outputs disabled, same v4 arrays. Add cast guard bootstrap reference (start2000 for fresh run) so untrained uniform priors do not block legitimate learning; resume default0. Tests/smoke pending before full Slurm dependency. |

| BC-V7-CLEAN-QUEUED | 2026-10-24 | Source1c5fab88. GPU13031 test/monitor smoke gates GPU13032 via afterok and durable smoke result. Fresh corrected-label baseline on58470 perspectives/7309 26.19 matches; existing v4 inputs, D/F/ward outputs off, no reconversion or old parameters.3epochs cap/eval500/sample512/patience4, cast guard reference2000/0.05/min100;24GB/4cores/16h, private staging/TERM persistence/cleanup. No simulator/RL; quality unproven. |

| BC-V7-CLEAN-SMOKE-PASS | 2026-10-24 | GPU13031 COMPLETED/0:54 tests, actual64-step/four-evaluation normal run and4-step/two-evaluation forced stop. Checkpoints, baseline retention, cast-guard activation and unsupported D/F/ward masks pass. Durable smoke result passed; GPU13032 released and RUNNING, no optimizer metrics yet. Existing replay arrays reused; no conversion/simulator/RL. |

| BC-V7-CLEAN-CANCELLED | 2026-10-24 |13032 CANCELLED by UID1000 after3h10m45s; no automatic restart. Trainer TERM saved6453 latest checkpoint on NFS, sampled best5500 score7.5057 retained. Cache build3250s/503.8GB. Weak cast recall remains; last6000 rejected Q regression. Handler status preempted is not proof of scheduler pre-emption. Scratch cleanup not yet independently verified. |

| BC-V7-RESUME-PREP | 2026-10-24 | Dani authorized resume6453. Found optimizer/cursor resume reset monitor state; explicit donor monitoring now recovers fixed sample, cast reference2000, best5500 and stale1, excluding future records. Recovery and real resume tests gate full continuation. Original3-epoch ceiling/recipe retained. |

| BC-V7-RESUME-QUEUED | 2026-10-24 | Source2568861a; smoke13033 gates resume13034. Actual smoke tests inherited best/reference/plateau and cursor. Full restores6453/best5500/reference2000/stale1, original3epoch ceiling. Desktop scratch cleanup verified684.7GB free. Measured prior staging~46min/cache54min, training~4.3ksteps/h;2–4h plateau or10–12h budget estimate excluding queue. Spell recall still weak; detailed rate/timing/aim evaluation precedes next targeted experiment/new-data scaling. |

| BC-V7-RESUME-SMOKE-PASS | 2026-10-24 |13033 COMPLETED/0,55 tests and actual2→4step resume preserved best bytes/reference/stale and inherited early stop. Durable passed result released13034, RUNNING in staging with no optimizer output yet. /srv startup cwd missing on desktop; scripts recover with explicit private-code cd. Future submission cwd must use /mnt/nfs. |

| BC-V7-FAILURE-AUDIT | 2026-10-24 | CPU13035 registered2CPU/6GB/45min on danilogin while13034 uses desktop GPU.48 replay perspectives for supervision/context coverage; clean5500 gradients on8 natural/8 spell-centered train windows compare native task contribution conflict. No optimizer/simulator. Code-level risks: own-readiness/previous-action gaps,25.6s resets, anonymous extra champions, action-independent factorized aim. Test impact before targeted loss/sampling/input ablation; no current-run changes. |

| BC-V7-FAILURE-AUDIT-RETRY | 2026-10-24 |13035 failed before metrics on packed entity tuple indexing. v2 uses supported time slice, verifies dense shape, then indexes. CPU13036 registered with unchanged48-perspective/gradient diagnostic. No findings yet; GPU13034 unaffected. |

| BC-V7-FAILURE-AUDIT-RESULT | 2026-10-24 |13036 passed109.9s:48 perspectives,97.96/97.47% click mass movement/basic attacks; no own-CD values, mana present atall4352spell events; extra enemy visible24.27/18.72%. Eight gradient batches show weak/mixed click vs spell-button cosine, not systematic conflict. CPU13037 paired native vs globally normalized5x spell aim, unchanged button loss,200updates/arm, disjoint-base selection/assessment and movement/cast gates. Diagnostic only; GPU13034 unaffected. |

| BC-V7-AIM-BALANCE-RESULT | 2026-10-24 |13037 passed36min, both arms retain baseline0 due event-button NLL guard. Weighted spell aim improves selection CE more than control atall3checks, but no accepted quality gain; small native control also deteriorates. Baseline Q expected315vs59human suggests calibration distortion. CPU13038 fixed-model64-perspective ranking/calibration audit, no optimizer or GPU contention, distinguishes cast discrimination from prior bias. |

| BC-V7-CAST-RANKING-RESULT | 2026-10-24 |13038 passed675.5s/64perspectives/358473frames. Exact QWER AP3.73/1.85/3.06/0.78%, above rare priors but weak;0.5s-near AP29.2/16.0/25.1/8.2%. Inverse class weights substantially correct expected counts but barely alter ranking and erase argmax casts. Calibration-only fix insufficient. No new broad/frozen-head run on early5500; main13034 continues, then compare same diagnostic on later clean parameters before input ablation. |

| BC-V7-EXPERT-SCOPE-CANCEL | 2026-10-24 | Exact plan scope30–840s/alive/lane-corridor,172champions,6574/735 match split; not full-match training. Target modes affect loss but not inputs/conditional heads.13034 CANCELLED byUID1000, last accepted6500 score7.46719, latest7000 payload readable; no terminal manifest or cleanup proof, no automatic restart. Expert explanation/visual checked against actual manifest and implementation. |

| BC-V2-DESIGN-PIVOT | 2026-10-24 | Dani adopted Lane BC v2 and requested a Git checkpoint. Pre-pivot tag at e62b4a72; BC.md§15 authoritative amended specification. Ahri-mid592(521/71), Garen-top236(210/26), B23879 verified raw files. Event/metadata primitives and regression tests added, not wired into trainer or Slurm-tested yet. No existing jobs active/resubmitted; v2 raw audit next. |

| BC-V2-SOURCE-AUDIT-QUEUED | 2026-10-24 | CPU13041 registered against pivot118edf40/worker e32a0efa90fc,2cores/8GB/1h desktop. Full61 expected BC tests,24 target-role raw perspectives, frozen544 decoder hashes, observation parity/target/collision/interval audit and metadata-only B match reservation inventory. No v2 training or full-audit completion claim. |

| BC-V2-SOURCE-AUDIT-PASS | 2026-10-24 |13041 passed61tests/544hashes/24observation parity,116s raw audit. Ahri Q/E/R points100%,Wself100%; Garen QWEself,R14/14unit. Cast+path co-bins26%Ahri/39%Garen; preserve raw order and targets, audit reissues before interpreting clicks. B23879metadata-matched,Ahri2637/Garen2477matches. New conditional/event implementation pending integration tests, not trained. |

| BC-V2-CONVERT-MISSING-ANCHOR | 2026-10-24 |13042 passed69tests then failed on second raw perspective: co-movement screen projection used NaN delayed own position. Preserve source/timing, mask aim for invalid scope/missing anchor; regression added. Partial dataset has no complete manifest. V2 gradient/training unrun; fresh conversion retry next. |
| BC-V2-CONVERT-RETRY | 2026-10-24 | CPU13043 registered, sourceb9dc57e7, fixed missing co-movement anchor. Full suite plus24 source-audited examples, fresh outputs; no model training yet. |
| BC-V2-GRADIENT-SMOKE-QUEUED | 2026-10-24 | GPU13044 registered afterok13043; source6d109f91, full model/two fixed64-step windows/200updates, finite gradient and30% loss reduction gate. Checkpoint/results NFS, private immutable source/data staging. No generalization claim. |

| BC-V2-EVAL-FOUNDATIONS | 2026-10-24 | Stateful shared-policy replay evaluator and event/target metrics implemented; six focused CPU tests pass4.88s. Mask gaps prevent false tolerant matches; AP ties handled; padding/burn masks verified. Evaluation retains replay-conditioned receipt history, so not closed-loop evidence. Await13043/13044 registered gates; model evaluator unrun. |
| BC-V2-CONVERT-PASS | 2026-10-24 |13043 passed70tests/24perspectives/158037valid frames,119.44s conversion; complete immutable fixture manifest. New sharded target converter and deterministic whole-match calibration reservation prepared. Await GPU13044; no v2 quality claim. |

| BC-V2-TARGET-CONVERT-QUEUED | 2026-10-24 |13045 array4/max2,source5956656a; finalizer13047 afterok.828target perspectives, Ahri470/51/71 andGaren185/25/26 train/calibration/val, frozenplan5f975d01bdcc, match-wide isolation. Full suite shard0, per-game decoder/parity/mask gates. New bounded learner/evaluator unrun; await13044 gradient result. |

| BC-V2-GRADIENT-PASS | 2026-10-24 |13044 passed finite gradients/updates and loss4.5477→1.5281 at200/177.87s,6.903Mparams; aim7.7206→2.8845. Button67.89%,NLL0.6548 is not strict memorization. Longer strict gate added.13049 registered32-step learner/control-vs-resume at16/full-replay evaluator smoke. No held-out quality claim. |
| BC-V2-STRICT-AND-JOINT-METRICS | 2026-10-24 |13050 registered1000-update strict fixed-window check, >98%button/<0.05NLL. Per-button target metrics add joint success including missed buttons; three focused metric tests pass.13045/13047/13049 pending registered results, no broad Ahri training or quality promotion. |
| BC-V2-TARGET-CONVERT-PASS | 2026-10-24 |13045 all4 and13047 completed:828sets,5420481valid frames,74tests,0.593GB NPZ,1188–1205s/shard. Ahri Q/E about15% aim clipped; raw direction retained for angle ablation. GarenR296/305 pointer coverage; stop0labels. Frozen splits intact.13049 running/13050pending at receipt check; no baseline launch yet. |
| BC-V2-RESUME-COMPARISON-FAIL | 2026-10-24 |13049 training32/control and16→32/resume plus6791-frame stateful eval completed; final maxabs parameter7.426e-5>1e-5 failed. Loss difference<=7.564e-5; numerical cause unproven. Probe adds exact checkpoint/optimizer roundtrip and repeated continuations before accepting bounded equivalence. No broad baseline. |
| BC-V2-RESUME-PROBE-QUEUED | 2026-10-24 |13055 registered source5fd69a8f, GPU20GB/20min; exact roundtrip/counter gate and repeated16→32 continuations, explicit numeric bounds. Donor13049 preserved; no blind rerun or accepted exact-resume claim. Await13055/13050; no broad baseline. |


| BC-REVIEW-001-SCOPE | 2026-10-24 | Independent review of source5a369951 on branch `review/bc-quality-20261024`, read BC thread200b87b6 positions4074–4285 including Dani's pasted Fable spec4150 and authoritative BC.md§15 amendments. Scope: replay-only Ahri-mid/Garen-top early lane, not RL/full-match/client integration. Reviewed replay/observation, current event labels/inputs/model/loss/batches/train/eval/metrics/conversion and relevant legacy contracts. No production fixes.75 BC tests pass in frozen-source CPU13052; passing tests do not cover the issues below. |
| BC-REVIEW-002-ORDER-P1 | 2026-10-24 | `replay._actions` reads path/attack/cast fields separately then stable-sorts only receipt time (replay.py540–563), discarding native update_index/event provenance. CPU13054 checked original24 perspectives,30–840s before alive/corridor filtering:356 same-clock QWER/multipoint-path pairs,356/356 actually cast-before-path in native update_index. Reader instead emits path-before-cast. Example NA1-5650989418 Garen249.9346160888672:cast order82905,path82914. `previous_received_events` consequently remembers cast instead of the later path at that clock. P1: retain native order/provenance and sort by(clock,source order); do not claim raw-order preservation from the current sidecars. Results/examples: `/mnt/nfs/shared/modern-bc/review-20261024/results-raw.json`. |
| BC-REVIEW-003-MOTION-P1 | 2026-10-24 | Same raw audit:56875 paths,429 special-motion,423 multipoint special-motion;203 special paths within200ms of own QWER. `replay._actions` discards flags/special_motion_parameters and `label_events` treats every2+point path as a move intention. Special movement can be an effect rather than an issued move; current labels cannot disambiguate it. P1: retain source motion kind and mark affected movement supervision ambiguous/excluded pending qualification; do not infer every special path is a voluntary click. This affects primary move labels, cast-adjacent co-movement and previous-event features. Ordinary path receipt still does not prove an input click. |
| BC-REVIEW-004-WINDOW-P1 | 2026-10-24 | `event_train.py66–79` constructs fixed stride256 windows once; `event_batch` zeros first32weights every time. Those frame labels never receive direct supervision at any update (they still affect recurrent context). CPU13053 all828 perspectives:686275/5420481 valid positions omitted. Actual470Ahri training perspectives:378532/2992280(12.65%), including2123/17226Q,2023/16607W,685/5413E,294/2531R. P1: provide preceding burn context with overlapping supervision windows or stateless-randomized offsets retaining exact resume; add whole-dataset coverage check. No additional training can recover labels whose loss weights are alwayszero. |
| BC-REVIEW-005-POINTER-P1 | 2026-10-24 | `EventPolicy.targets` event_policy.py129–141 masks padding only. Known allied units remain attack candidates and allied/minion/turret rows remain GarenR candidates; all-empty sets return arbitrary row0. CPU13053:103078/107637(95.76%) aim-valid attack frames have allied candidates (Ahri82213/84553,Garen20865/23084). This measures invalid candidate exposure, not observed model misclick frequency. P1: pass qualified action-specific team/type eligibility consistently through train/inference/metrics; safely suppress/skip target actions when no candidate exists. Unknown ability eligibility must remain unknown instead of guessed illegal. |
| BC-REVIEW-006-AIM-COVERAGE | 2026-10-24 | Full828/592Ahri profiling13053:AhriQ21801 labels,18573aim-valid,3219clipped(14.77%);E6899/5832/1059(15.35%);R3220/2726/469(14.57%). A few additional missing targets remain. `event_labels` correctly preserves buttons and masks clipped aim, but target metrics condition on aim_valid and cannot describe this missing tail. Movement119k/959kAhri clicks clipped too. P2: report coverage next to accuracy, audit clipped direction/range separately, qualify and test directionalQ/E labels before interpreting grid gains as skillshot fidelity. `angle` exists for point casts but ANGLE_TARGET is never assigned in current labels; switching heads alone does not recover aim_valid for clipped directions. Keep variable-distanceR semantics separate. |
| BC-REVIEW-007-MOVE-INTENT | 2026-10-24 | Full13053:1439986 in-scope valid-bin multipoint path records;509995(35.42%) within150u of previous multipoint destination within500ms;79686(5.53%) identical endpoint within500ms. Median interval201ms, endpoint shift242.7u. This is a reissue diagnostic, not proof that those commands are redundant or that collapsing them is correct. Ahri previous-received-point copy baseline on737457 eligible move aims achieves18.39% within2cells/axis, median9.43cells; still substantial nontrivial movement. Co-paths occur on12391/52967Ahri cast bins(23.39%),6900/17622Garen(39.16%); default co_move_weight0 means preserved co-movement has no training effect. Primary labels discard375/70964raw QWER via same-bin arbitration. Keep literal events vs order intent vs motion effects distinct; evaluate any dedup/order-change ablation on frozen match splits. |
| BC-REVIEW-008-ATTACK-AMBIGUITY | 2026-10-24 | `label_events` and `previous_received_events` independently implement the same attack dedup proxy. Only multipoint paths reset active_attack; onepoint paths are ignored. Synthetic attack(.1,target99),onepoint(.2),attack(.6,target99) yields continue at.6 and stale .1 attack history. Onepoint path can be arrival/attack stop, so resetting all of them would also invent inputs. P2: preserve interruption/ambiguity evidence, source-test attack starts, and centralize the shared event interpretation; do not treat all attack executions or all onepoint paths as clicks. Stop haszero positive labels in all828 perspectives. The advertised stop behavior remains unsupported, deliberately so until source-qualified. |
| BC-REVIEW-009-EVALUATION-P2 | 2026-10-24 | `sequence_metrics` matches events within0.5s but scores joint target success only on exact-frame correct buttons. A correct target emitted100mslate gets eventTP yet joint target failure; there is no target-on-matched-event/world-ID comparison. Keep existing exact diagnostic and add consistent tolerant joint metric with stable unit IDs/world target coordinates. `predict_game` consumes human received history and carries state through whole replay, while training zeros everywindow; neither scan resets on death despite BC.md§15 protocol. Label-shuffle/history-corruption controls, calibrated emitter, stateful/reset ablation and live adapter are pending, not validated by75tests or memorization. No useful closed-loop/real-client claim. |
| BC-REVIEW-010-EVIDENCE-AND-GATES | 2026-10-24 | Positives: current human target_button conditions only aim readout, not recurrent/button state (regression-tested); target/button validity separate; delayed unit identity matching; hidden-enemy cast memory filtered; unknowns retain missingness; received own ages are not asserted cooldown legality; A splits/calibration reserve match-wide; B unopened. Decoder position_model uses strictly earlier paths and speed receipts at/before query; loading the whole replay is not by itself future-path leakage. Metadata implementation supplies raw modes/rank cooldown references, not full qualified range/width/speed/recast catalog; self/global parity is not part of the current reused-observation assertion (entities/pad only). Before broader quality claims: source-order/motion fixes, window coverage, eligible pointer, trusted target coverage, shuffled-label/history controls, consistent event+target scoring. CPU13052/13053/13054 completed; results.json/results-full.json/results-raw.json, frozen scripts and logs under `/mnt/nfs/shared/modern-bc/review-20261024/`; scratch cleanup verified. Repro code `ops/modern/bc_quality_profile.py`. Findings are not production fixes or new policy-quality evidence. |


| BC-REVIEW-FIX-QUALIFICATION | 2026-10-24 | Dani authorized verifying/fixing findings and further review. Native order/window/pointer defects confirmed by raw source and regression cases. Special/flagged paths are ambiguous input evidence, not proven voluntary moves; conservatively mask that supervision/history without calling every special path a dash or synthesizing stops. Onepoint interruption cannot prove an attack restart, so mask ambiguity instead of inventing a click. Clipping is deliberate validity handling; fix coverage metrics rather than promote unverified angles. Ordinary re-clicks remain unchanged. |
| BC-REVIEW-FIX-BROADER | 2026-10-24 | Additional confirmed gaps: optional smoothing crosses invalid gaps; GRU/previous-order history survive death; grid includes canonical minimap; loader accepts changed artifact contents behind unchanged manifest and fails to enforce match splits itself. Implemented fixes plus consistent tolerant joint target scores using stable IDs/world projection, full reused-field parity and native co-order without fabricated time offsets. New event-v2/artifact/sampling contracts explicitly reject old arrays and incompatible resume.31 focused lightweight tests passed5.93s total. Full suite/raw conversion/GPU checks pending desktop recovery. |

| BC-REVIEW-FIX-VALIDATION-QUEUED | 2026-10-24 | Corrected source196698a8 includes128b9357 fixes and merged current diagnostics.31 focused tests passed locally;91 tests collect0.64s. Pending registered CPU13058(full suite/24raw conversions/reused self+global parity/content hashes/window profile),13059(828target rebuild,4shards/max2,afterok13058),13060(finalizer/full profile afterokarray),GPU13061(learner/resume/eval plus independent random-label and absent/stale-history probes,afterok13058). Superseded unstarted13056/13057 forgotten/cancelled, not retried. Desktop drained/not responding, reason boot to Windows. Verification remains incomplete; no production integration or new policy acceptance. Source/scripts/logs in `/mnt/nfs/shared/modern-bc/review-fixes-20261024/`. |
| BC-REVIEW-LARGER-LIMITS | 2026-10-24 | Event proxies cannot recover every input, failed command, forced-motion intent or readiness/recast state. Fixed-window training versus full-state replay evaluation and receipt-history versus deployed feedback still require measured controls; these are not solved by teacher-button separation. Pre-fix target data has only235GarenR train/40validation labels, so millions of correlated frames do not justify fine-grained rare-action quality claims. Additional source-contract question: frozen position_model returns available special-path estimates while policy_position_accepted/approximation_accepted_for_bc are false, yet replay reader consumes position_available alone; quantify affected observed frames and reconcile acceptance before asserting special-motion position fidelity. Do not silently invent authoritative locations or erase known cast buttons to hide that uncertainty. |


| BC-SYSTEM-001-OBSERVABILITY-P1 | 2026-10-24 | Continued large-system review after event fixes. `observation.build`/`event_inputs.SELF_FIELDS` discard own ability-haste/attack-speed data and absolute HP/resource pools even though `ReplayGame` retains them. Repro `ops/modern/bc_system_audit.py` creates identical complete formatter outputs after changing own haste/attack-speed, doubling HP/maxHP, or changing50/100mana to500/1000. Repeated same-target attack execution also leaves previous-event history identical, so latest attack evidence is lost rather than represented separately from issued-order changes. Synthetic equality proves information loss; it does not measure learned performance impact or prove that every counterfactual is common. Prioritize source-qualified own numeric pools/modifiers and independent last-execution memory before capacity sweeps; keep raw AS modifiers distinct from qualified attacks/second and confirmed cast ages distinct from true readiness. |
| BC-SYSTEM-002-DAMAGE-CS-P1 | 2026-10-24 | Frozen decoder `combat_effects.damage_notice` explicitly emits victim/source IDs None, victim_source_roles_available=False, consumer_application_executed=False; export manifest declares received_damage_notice_roles_available=False. BC `replay._damage_notices` nevertheless assigns receiver=victim/participant_ids[1]=source; `_last_hits` then presents the inferred count as CS. Synthetic unqualified notice at10.02credits a death at10.00 because tolerance admits34msfuture evidence, violating the stated observation receipt cutoff (not proof of information later than the eventual action). in_combat also derives from this participant-pair interpretation and is a proxy, not established engine combat state. Repro output `/mnt/nfs/shared/modern-bc/review-fixes-20261024/system-synthetic-expanded.json`. Recommend mask CS until attribution is qualified, or explicitly separate a causal/provenance-marked proxy; never substitute knownzero for unavailable CS. Corpus frequency of suspect credits awaits CPU13062. |
| BC-SYSTEM-003-LIFETIME-P1 | 2026-10-24 | Synthetic actual `_read_lane_units` call: ordinary path received0s, death1s, positiveHP2s, query2.5s returns x770.25 from the0s path. The integrator has no lifetime reset and reader only gates the query by alive, so a respawn can resume predeath movement evidence without a new pose. This can affect positions, team-vision circles and lane-scope labels; impact is not measured yet. Same function consumes special-path estimates when position_available=True while policy_position_accepted=False and approximation_accepted_for_bc=False. Source-specific qualification flags must not be replaced by the global manifest's broad acceptance marker. CPU13062 counts own/paired alive queries with predeath paths and available-but-unaccepted positions, before fog/corridor filtering. |
| BC-SYSTEM-004-MEMORIZATION-GATE-P1 | 2026-10-24 | Reproduced `event_smoke` selector against immutable13043 fixture used by strict13050. Ahri NA1-5654339924 start6996:54valid,32continue/19move/1attack/1W/1E, noQ/R. Garen NA1-5653093002 start6484:55valid,38continue/13move/2attack/1W/1R. Total109valid; missing both sole AhriE and GarenR still permits107/109=98.165%accuracy and meanbuttonNLL0.02953 if their true-label probabilities are0.2 and others nearly1, satisfying both strict button thresholds. No per-button target success gate exists. Thus strict memorization is not evidence that Q/R conditional skillshot branches work; Q has no positive example at all. Require explicit per-champion/button/mode support, per-class button success and pointer/point target success on that support. Existing frozen fixture/GPU jobs remain wiring/numerical probes, not clearance of this issue. |
| BC-SYSTEM-005-HYPOTHESIS-CHECKS | 2026-10-24 | Inspected all802unique export manifests referenced by828target perspectives (167402711metadata bytes):0declared decode gaps,0quarantine starts,0unusable manifests. Reader does not consume gap.scope, but this is a future robustness concern, not an observed cause in the current target corpus. Frozen cast_targets.py explicitly records owner-approved asset-plus-evidence point-family rule; do not equate every received endpoint with a literal mouse click or reject all point labels merely because full cast application remains unqualified. Offline event metrics still consume replay-provided receipt history; actual-button conditioning does not establish closed-loop behavior. Random-label/history probes are diagnostics, not a live adapter. |
| BC-SYSTEM-006-RAW-AUDIT-QUEUED | 2026-10-24 | CPU13062 registered to this review thread,2cores/4GB/15min desktop. Independent frozen worker96fd90181653 over source196698a8;24native source perspectives, damage-role/current-CS-rule/future-notice counts, own-stat availability, own/paired position acceptance and stale postdeath paths. Private per-game copies/decoder snapshot,20GiBreserve, cleanup and durable output `review-fixes-20261024/system-raw-result.json` under `/mnt/nfs/shared/modern-bc/`. Queued behind drained Windows desktop; no actual rates yet. Does not mutate source-final snapshot or validation13058..13061. |

| BC-REVIEW-FIX-CPU-PASS | 2026-10-24 | CPU13058 completed0:91tests passed43.17s;24reconversions passed133.09s with entity/padding/self/global parity and finalized hashes. Corrected profile:0post-initial-burn omissions and0duplicate supervised frames;572valid initial-burn frames deliberately context-only versus19490legacy omissions. Allied rows counted by profile remain visible entities, not model-eligible pointer candidates; eligibility is checked separately. Source196698a8 remains an earlier event-fix validation, not a source-fidelity sign-off.13059first2shards running,13060dependency/13061resources pending at inspection; desktop recovered. |
| BC-SYSTEM-007-RAW-RESULT | 2026-10-24 | CPU13062 durable output over24native perspectives:687046/687046damage notices lack qualified roles;2728current-rule last-damage CS credits,0future-notice credits observed (synthetic timing defect still reproduced). Alive own/paired position queries370405:5475available-but-unaccepted estimates(1.478%),158predeath-path reuses(0.043%). Counts precede visibility/corridor filtering, not all policy-scored frames. Own queried stats194400timestamps:haste finite194400/positive75308;attack-speed modifier finite194400/positive194400;maxHP finite/positive194400;maxresource finite194400/positive97200 (manaless Garen legitimatelyzero). No learned performance effect measured. Scratch directories13058/13062 absent after completion; desktop sacct database unavailable, durable logs/results inspected. |
| BC-SYSTEM-008-SOURCE-REPAIR | 2026-10-24 | Implemented conservative v3 repair: require per-estimate acceptance, require postdeath movement evidence, require delayed own anchor for scope; CS unavailable pending qualified kill attribution, damage roles require explicit qualification/IDs, causal last-damage diagnostic disallows later receipts, unqualified combat remainsNaN through formatter.8formatter/source regression tests passed2.34s locally. v3 contract rejects earlier v2arrays. Full tests and real-source reconversion still required; retained own-stat/execution-memory omissions and weak memorization gate remain open. |

| BC-SYSTEM-009-REPAIR-VALIDATION | 2026-10-24 | Registered CPU13068, source1ee090a0,2CPU/4GB/15min. Full94test suite plus24raw reconversions, rejection of old shared observations, rebuilt-reference parity, CS/combat missingness, finalized artifact hashes and window profile. Immutable `source-qualified.tar.gz`/SHA256 and worker/scripts under review-fixes-20261024. Await actual output; no source-v3production promotion. Earlier13059..13061 event-v2jobs remain separate frozen diagnostics. |

| BC-SYSTEM-010-RECEIPT-AND-RETRY | 2026-10-24 |13062 COMPLETED/0 receipt confirmed; durable24-perspective counts re-read.13068 FAILED/1 before tests because scheduled CPU node lacked/scratch; mktemp was sole log error. Registered replacement13071 explicitly pins desktop with same frozen1ee090a0source,2CPU/4GB/15min. No code/test failure claimed and no duplicate retry needed when13068receipt arrives. Await13071. |

| BC-REVIEW-FIX-GPU-PASS | 2026-10-24 |13061 COMPLETED/0 actual logs/results inspected, frozen196698a8event-v2.32step control/resume maxabs0.0001168475/relativeL2 0.0000150532 passes explicit1e-3/1e-4bounds, serialization params/optimizer exact. Independent per-match random-label heldout accuracy0.1127141 over12962frames vs1/9chance. Finite learner updates/evaluator/diagnostic outputs complete; scratch absent. Does not validate newer v3source repairs. Durable results `/mnt/nfs/checkpoints/lanerl-jax/modern-bc/bc-review-fixed-13061/results.json`. |
| BC-SYSTEM-011-CONTINUE-BASELINE | 2026-10-24 |13061 raw validation predictscontinue on12962/12962frames;9503continue truths yield0.733143accuracy, exactly majority baseline. Misses2979move/301attack/66Q/72W/21E/7R (plus other noncontinue labels); all actual target joint success0. Teacher pointer accuracy57–60% does not rescue missing actual buttons; point within2cells0.08–0.66%. History-off and20frame-stale probes retain same allcontinue output: NLLraw0.75901/off0.76173/stale0.75975. Thus these probes at32steps are uninformative about useful history dependence; no competence demonstrated. This is a short wiring run, not evidence that architecture cannot learn or justification for unmeasured reweighting. Future behavioral acceptance must compare majority/event-rate baselines and per-class recall/joint-target scores on frozen heldout matches, alongside source qualification and memorization coverage gates. |

| BC-SYSTEM-012-SOURCE-V3-PASS | 2026-10-24 |13071 COMPLETED/0:94tests passed40.63s,24native reconversions165.15s. Frozen1ee090a0 contractreplay-lane-events-v3. Worker rejects all24old shared observations, rebuilds shared references, verifies parity/shape/missingCS+combat and manifests; independent read verifies all24artifact hashes and allCS/combatNaN. Corrected windows retain0post-initial-burn omissions/0duplicate supervision; scratch absent. Does not constitute full828v3conversion/GPU/behavior validation. |
| BC-SYSTEM-013-DIFFERENTIAL-COVERAGE | 2026-10-24 | Same24perspectives v2→qualifiedv3:valid153965→152270(-1.10%);GarenE438→366(-16.44%),Q391→389,W116→110,R14→13;AhriQ423→417,W429→418,E138→132,R60→59. Per-result acceptance/current-lifetime/delayed-anchor filtering is necessary source qualification but removes states unevenly by action; remaining data cannot be assumed representative of missing states. This does not prove special motion alone caused every omission (filters combined), and no learned effect measured. Raw cast/path receipts remain preserved. Track action-conditioned retention and qualify missing-state reconstruction before claiming combat coverage; do not restore rejected locations or count these casts as negatives. |

| BC-REVIEW-FIX-FULL-SHARDS | 2026-10-24 |13059all4COMPLETED/0; inspected each actual log, all207/207perspectives (828total). Frozen196698a8event-v2 conversion, predates source-v3position/damage qualification. Existing dependent13060handles finalization/profile; no duplicate work launched and no final dataset quality/promotion inferred from shard success. |

| BC-REVIEW-FIX-FULL-PROFILE-PASS | 2026-10-24 |13060COMPLETED/0; actual finalizer log has no errors, finalized v2manifest passed828perspectives/828artifact hash sets. Profile5284445valid frames (Ahri3684018/Garen1600427),0post-initial-burn omissions/0duplicates.18647valid first-burn frames context-only; legacy fixed-window omissions669133 on same corrected data. Confirms window fix at full scale; retains source196698a8/v2limits, superseded source qualification requires fullv3rebuild. Original13058..13061graph settled; no additional jobs submitted. |

| BC-REVIEW-ROADMAP-20261025 | 2026-10-25 | Dani requested end-to-end review with symptom indicators: inventory current branches/integrated fixes; independently synchronized XPS overlay/source validation; tensor provenance/causality/identity/missingness/coordinate checks; controlled action-space comparison; trainer-paper/behavior gates; bounded hyperparameter studies; simulator/JAX semantics; targeted simplification. Current modern/bc8343868b differs from review branch; old receipts are snapshot-specific. Lightweight current-branch census:BC36non-testPython files/5376lines,25testfiles/1524lines; includes historical and event pipelines, not a bloat verdict. Existing decoder replay-overlay docs describe301-frame Jax clip with schedule-derived times,59camera polls,first20frames uncovered, unknown replayheight and HPbaranchor; native camera-math proof does not resolve clock/anchor uncertainty. Reuse and independently verify this infrastructure before new capture. Action proposal:retain rawworld receipts,learn semantic buttons/unitidentity/qualified direction or relativepoint distributions,compare against current96x54grid with targetcoverage and representation-oracle errors; no blanket angle conversion or inferred literal clicks. No new run/model acceptance. |

| BC-V2-STRICT-REQUEUE-FAIL | 2026-10-24 |13050 Restarts=1, empty output already exists, original log overwritten; no completed metrics or strict-model verdict.13063 registered identical source/budget with no-requeue/append logs, fresh job output. Owned13055 running first attempt, Requeue disabled. Future BC wrappers require explicit restart policy; no baseline launched. |
| BC-V2-RESUME-PROBE-PASS | 2026-10-24 |13055 exact parameter/optimizer serialization, count16x2; continuation replicas identical. Across original/probe maxabs1.1275e-4,relativeL2<=2.799e-5,final-loss spread7.153e-6. Bounded numerical resume accepted, no cross-process bitwise/kernel-cause claim. Strict13063 outstanding; baseline not launched. |
| BC-V2-BATCH-BENCHMARK-QUEUED | 2026-10-24 | Exclusive13067 afterok13063, sourcef2891e0a, batch8/32/64 separate16-step processes,window256/burn32,24GB/45min. Capacity/throughput only.13066 intentionally cancelled before execution to add node exclusivity; no automatic restart. Broad baseline still gated. |
| BC-V2-STRICT-NLL-MISS | 2026-10-24 |13063 ran1000updates/217.86s:99.08%button accuracy,NLL0.06303>0.05,loss0.81843,aimCE2.49509. Constant-LR late oscillation. Prepare same-checkpoint/moments1000→1200 at3e-5, unchanged strict criteria; no broad baseline.13067 must be rebound after failed dependency. |
| BC-V2-STRICT-POLISH-QUEUED | 2026-10-24 |13069 source19e43e03 resumes13063 exact windows/optimizer,1000→1200 total at3e-5 with unchanged98%/0.05 gate.13067 intentionally cancelled for failed dependency; exclusive benchmark13070 afterok13069 replaces it. No broad baseline; await receipts. |

| BC-V2-STRICT-MEMORIZATION-PASS | 2026-10-24 |13069 restored1000→1200 at3e-5/161.97s,100%button accuracy,NLL0.002799,loss0.74956; unchanged98%/0.05 gate passes. Aim CE2.46658 includes nonzero soft-target entropy. Two fixed windows only, no held-out/leakage claim. Await exclusive13070 throughput then broad Ahri baseline. |
| BC-V2-AHRI-BASELINE-QUEUED | 2026-10-24 |13070 passed batch8/32/64 capacity;43.80kfps at64 selected.13074 baseline1000updates/b64/window256/burn32,470train/71val,seed0;51calibration unused.13075 matched unconditional afterok13074, same sampling/budget/heads. No quality results yet; report aggregation/match bootstrap implemented. |
| BC-V2-AHRI-BASELINE-RESULT | 2026-10-25 |13074 passed1000 updates/71val/451441frames. Greedy0attack/QWER despite expected ratios0.94–1.05; Q/E oracle-button within2 only1.57/1.41%,attackpointer68.68%. No promotion.13075 pending result. Added isolated angle/cast-aim flags and sampled-target/RNG evaluator + tests, not yet validated or trained. |
| BC-V2-AIM-EMITTER-ABLATIONS-QUEUED | 2026-10-25 | CPU13076 validates source73fe070d; GPU13077 frozen-baseline greedy/sample0 assessment with angular metrics,13078 Q/E angle same label support,13079 grid+3x cast aim. Both learners1000/b64 same seed/schedule/draws as13074. No combined changes, no test/calibration fitting or promotion. Await13075 and new receipts. |
| BC-V2-CONDITIONING-CONTROL-RESULT | 2026-10-25 |13075 same71val: conditional Q/E within2 +0.305/+0.282pp, movement+0.483pp; attackpointer-0.277pp,Q/E exactAPslightly worse. Both greedyQWER0, no useful quality/promotion.13076 passes78tests/35.66s. CPU13080 registered paired500-match-bootstrap comparison;13077–13079 gates cleared. |
| BC-V2-CONDITIONING-PAIRED-INTERVALS | 2026-10-25 |13080 passed71matches/500bootstrap draws: Q/E within2 delta intervals include0; movement+0.341..0.634pp; Q/W exact-frame AP intervals negative. One training seed only. No practical conditioning-only promotion; await13077–13079, no new jobs. |
| BC-V2-SAMPLED-EMITTER-RESULT | 2026-10-25 |13077 passed71val: sampled Q/W/E/R F1 .150/.107/.053/.096 at near-human rates; movement.806,attack.365. Teacher grid angular error Q34.78/E87.49deg mean per-match median. Greedy-zero issue partly emitter choice; timing/aim still weak. Added count-preserving timing null, unrun; await13078/13079. |
| BC-V2-TIMING-NULL-QUEUED | 2026-10-25 | CPU13081 sourceedc27926,32 permutations of13077 buttons within valid segments, counts preserved exactly. Frequency-only timing comparison; no matched-target/closed-loop claim. Await13081/13078/13079, no promotion. |
| BC-V2-TIMING-NULL-RESULT | 2026-10-25 |13081 passed32 count-preserving permutations. Actual/null median F1 move.806/.746,attack.365/.243,Q.150/.067,W.107/.067,E.053/.027,R.096/.036. Timing exceeds frequency control but spells weak; dense moveF1 mostly retained when shuffled. Not generalization CI or play-quality proof. Await13078/13079; no promotion. |
| BC-V2-ANGLE-RESULT | 2026-10-25 |13078 same71val/1000updates: Q angle34.78→12.37deg,E87.49→27.36 (mean match medians),within10 42.52/23.70%; movement within2 drops0.628pp,timing little changed. Strong aim component, no policy promotion.13082 sampled angle assessment registered; own-age/smoothing controls implemented, unvalidated. |
| BC-V2-TIMING-CONTROLS-QUEUED | 2026-10-25 |13083 full-suite gate,source2d36adc8;13084 own-age-only removal,13085 smoothing±0.1s,13086±0.2s on angle baseline.1000/b64 matchingseed/draws, greedy+sampled71val. No compound weighting/history changes, no calibration/B fitting. Await registered results; no promotion. |
| BC-V2-CAST-AIM-WEIGHT-RESULT | 2026-10-25 |13079 grid+3x target loss: Qwithin2 1.57→6.06%,angular34.78→19.13deg; E1.41→1.55%,87.49→78.16deg. Movementstable,timingmixed; angle-only still better.13083 passes79tests/37.31s.13087 registered angle+3x interaction, same1000/b64/seed0; no promotion. |
| BC-V2-SAMPLED-ANGLE-RESULT | 2026-10-25 |13082 Q/E sampled F1.145/.047, little timing change. Exact-frame Qaim26/49,E0/1 too sparse for operational joint claim. Implemented tolerant matched-direction scorer with index/gap/wrap tests and observed-opponent heuristic diagnostic, unrun. Assessor fingerprint provenance added; no promotion. |
| BC-V2-MATCHED-AIM-QUEUED | 2026-10-25 | CPU13088 sourcee4288d17, cached71val grid/angle sampled predictions + original labels,2regression tests, tolerant direction agreement and visible-opponent diagnostic. Predicted/received indices distinct; no new supervision/hit-rate claim. Await13088/13084–13087; no promotion. |
| BC-V2-MATCHED-AIM-RESULT | 2026-10-25 |13088 passes2tests; angle joint-known recall Q6.59%,E0.705%. Visible-opponent heuristic medians6.41/6.04deg beat learned12.18/21.13 on same1301Q/456E frames. Added optional full-circle opponent-relative head/global missing fallback, no new labels/params; tests pending. No policy promotion; await timing/weight arms. |
| BC-V2-RELATIVE-COORDINATES-QUEUED | 2026-10-25 |13089 full-suite83tests/38.62s completed;13090 relative-angle arm registered against8c668da2, same1000/b64/seed0/no smoothing/weighting change. Full-circle support/global missing fallback, no extra parameters or supervision. No quality result yet; await13084–13087/13090. |
| BC-V2-OWN-AGE-RESULT | 2026-10-25 |13084 removal worsens Q/E exactAP6.254/2.313→5.387/2.019%,Q/R sampledF1.145/.096→.135/.069; E sample mixed,aimlittlechanged. Retain explicit ages provisionally, no engine readiness.13091 paired-report job after remaining arms,500bootstrap/71matches, same-seed only. |
| BC-V2-SMOOTH1-RESULT | 2026-10-25 |13085 ±0.1s: sampled Q/W/E/R F1.146/.112/.052/.101 versus.145/.109/.047/.096, small; movementwithin2 drops9.30→8.43%,angleQ/E11.99/25.85deg. No practical timing fix/promotion yet; await±0.2s/weight/relative and paired intervals. |
| BC-V2-SMOOTH2-RESULT | 2026-10-25 |13086 Q/W/E/R sampledF1.185/.126/.071/.092,nearAPQ/E.319/.144,aimstable; small movement/attack regressions. Timing candidate, not promotion. Separate-calibration temperature/rate module implemented with fingerprint gate/test, unrun; no validation fitting or future-input masks. |
| BC-V2-SEPARATE-CALIBRATION-QUEUED | 2026-10-25 |13092 tests→13093 smooth2 frozen51cal collect→13094 fit temperature/rate biases by3-seed calibration castF1→13095 frozen71val assessment. Sourcecefee640, fingerprints, no validation fitting/retraining/future-input/cooldown masks. Candidate only; await registered results and remaining angle/weight comparisons. |

| BC-V2-CALIBRATION-TESTS-PASS | 2026-10-25 |13092 actual log confirms84tests/39.12s,COMPLETED/0. Collection13093 dependency cleared;13094 fit and13095 frozen validation remain gated. No fitted-emitter or model-quality result yet; await registered callbacks, no duplicate work/promotion. |

| BC-V2-ANGLE-AIM3-RESULT | 2026-10-25 |13087 completed1000updates/655.46s/71val. Q/E mean per-match median aim12.37/27.36→8.79/22.37deg,within10 42.52/23.70→56.00/31.45%. Sampled Q/W/E/RF1.1546/.1055/.0466/.0840, timing mixed; movementwithin2 9.304→9.685%. Component candidate only, one training/sampling seed; paired13091 and relative13090 pending. No policy promotion/new jobs. |

| BC-V2-RELATIVE-ANGLE-RESULT | 2026-10-25 |13090 completed1000updates/662.82s/71val. Q/E teacher mean per-match median12.37/27.36→8.93/14.27deg,within10 42.52/23.70→55.30/51.76%. E better than angle+3x22.37deg; sampled Q/W/E/RF1.147/.108/.050/.095 largely unchanged. Movewithin2 9.304→8.624%; component candidate, one seed, no policy promotion. Paired13091 prerequisites cleared; calibration chain pending. |

| BC-V2-PAIRED-ABLATIONS-RESULT | 2026-10-25 |13091 passed71matches/500paired draws. Smooth2 Q/W/E sampledF1 delta CIs all positive; attack negative. Relative Q/E angle-error delta CIs negative, movementwithin2 negative. Aim3 aim improves; smooth1 timing uncertain. Conditional on one training/sampling seed, no multiplicity correction/policy promotion.13096 registered relative+smooth2 interaction,source8c668da2,same1000/b64/seed0,weight1; registered calibration chain unchanged. |

| BC-V2-CALIBRATION-FIT-RESULT | 2026-10-25 |13093/13094 completed.51unique calibration perspectives,frozen smooth2 step1000,metadata/checkpoint fingerprint and actual manifest hash consistent. Rate-fitted T1/.7/.5 calibration QWERF1 .11343/.14858/.17489 averaged3draws;T.5 selected,stop unsupported/disabled. Selection-set only,no validation quality claim.13095 frozen71val dependency cleared;13096 combined learner pending. |

| BC-V2-FROZEN-CALIBRATION-VAL-RESULT | 2026-10-25 |13095 passed71val with frozen T.5/bias,checkpoint/dataset fingerprints consistent. Same smooth2 model raw→cal Q/W/E/RF1 .185/.126/.071/.092→.228/.179/.114/.175,mean.118→.174;rates.963/.978/.931/.986. Attack.358→.393,move.801→.795;teacher aim unchanged. One training/sampling seed,no interval/closed-loop or promotion. Await13096; new checkpoint needs own calibration. |

| BC-V2-COMBINED-RESULT-AND-CALIBRATION | 2026-10-25 |13096 completed1000updates/659.07s/71val. Relative+smooth2 Q/E teacher error8.90/14.28deg,movementwithin2 10.21%,raw Q/W/E/RF1.178/.130/.065/.109. Aim gains survive,raw timing mixed versus smooth2 alone.13097 collect→13098 own frozen calibration fit→13099 val registered,sourcecefee640,donor13096; no bias reuse/B/retraining/policy promotion. |

| BC-V2-COMBINED-CALIBRATION-FIT | 2026-10-25 |13097/13098 completed51cal collection/fit,frozen relative+smooth2 step1000,checkpoint4a623acb…/dataset fingerprints consistent. T1/.7/.5 calibration castF1.11291/.14291/.17038;own T.5/bias frozen. Selection-set only;13099 held-out scoring dependency cleared, no promotion/new jobs. |

| BC-V2-CALIBRATED-COMBINED-AND-REPLICATION | 2026-10-25 |13099 passed71val:Q/W/E/RF1.230/.169/.103/.203,mean.176 versus smooth2.174. Aim8.90/14.28deg retained,movementwithin2 10.21%;timing bottleneck remains,no promotion.13100/13101 matched seed1 smooth2/relative+smooth2 replications registered,same1000/b64,seed jointly changes init/draws/eval. No validation calibration fitting;await callbacks. |

| BC-V2-SMOOTH2-SEED1-RESULT | 2026-10-25 |13100 completed1000updates/658.43s/71val. Raw Q/W/E/RF1.171/.125/.076/.071,mean.111 versus seed0.118;cast rates.821/.879/.861/.843. Teacher aim9.56/23.44deg,attackpointer72.26%,movewithin2 8.62%. Material seed sensitivity; no comparison to calibrated scores or promotion. Await matched13101;no new jobs. |

| BC-V2-SEED1-RELATIVE-AND-CONVERGENCE | 2026-10-25 |13101 passed:matched seed1 Q/E aim9.56/23.44→8.19/13.97deg,timing meanF1.111→.113 raw. Relative aim gain repeats,timing weak.13102 4000-update relative+smooth2 seed0 and13103 train71/val71 curve1000/2000/4000 registered. New4000-step schedule,within-run convergence diagnostic,no calibration/B/RL or promotion. |

| BC-V2-DURATION-TRAIN-COMPLETE | 2026-10-25 |13102 completed4000updates/1893.34s,4NFS checkpoints verified. Last100-batch averages1000→4000 buttonNLL.6567→.6108,aimNLL6.400→6.003,accuracy71.38→72.98%;training-only.13103 fixed train/val curve dependency cleared,no generalization or policy promotion claim. |

| BC-V2-DURATION-CURVE-RESULT | 2026-10-25 |13103 passed6assessments:within4k schedule val castmeanF1.1434→.1896,train4k.1994,aimQ/E6.67/10.67deg. No large observed train-val gap on fixed71train/71val;not data-ceiling proof.13104/13105/13106 own4k calibration chain and13107 bounded8k freshschedule registered. Current formulation still learns;no policy promotion/B/RL,remaining leakage/history/deploy gates unchanged. |

| BC-V2-4000-CALIBRATION-FIT | 2026-10-25 |13104/13105 passed51cal,step4000,checkpoint4b3f302f…/manifest consistent. Rate-fitted T1/.7/.5 calibration castF1.19013/.24042/.27364;T.5 frozen. Selection-set only;13106 val gate cleared,13107 bounded8k registered. No promotion/new jobs. |

| BC-V2-8000-RESULT-AND-CALIBRATION | 2026-10-25 |13107 completed8k/3539.94s,raw valcastmeanF1.190→.215,Q/W/E improve,Rflat;aim6.45/10.65deg,attackF1.472. E/RnearAP regresses,so no uniform ranking/policy claim.13108 calibration51+train71 diagnostic→13109 fit→13110 frozenval registered;wait4k/8k calibrated and train-gap evidence before further duration. |

| BC-V2-CALIBRATED-4000-VAL | 2026-10-25 |13106 passed71val/frozenfit4k:Q/W/E/RF1.346/.289/.232/.237,mean.276 versus1k.176;attackF1.481,move.835. Rates.954/.933/.955/.854,teacher aim6.67/10.67deg. Stronger offline candidate,still belowcastF1.5;exactjointR0,not hit/closed-loop proof. Await8k13108–13110,no new jobs/promotion. |

| BC-V2-8000-TRAIN-GAP | 2026-10-25 |13108 passed51cal+71train assessments,8k checkpoint e969520f…. Raw castmeanF1 train.246/val.215,gap.031 versus4k.010;Q/E/R ranking gaps grow. Train subset first71sorted,not full train/causal estimate. Held-out gains persist but emerging generalization gap argues against blind extension. Await13109/13110,no new jobs/promotion. |

| BC-V2-8000-CALIBRATION-FIT | 2026-10-25 |13109 passed51cal,fingerprintse969520f… consistent. T1/.7/.5 castF1.22149/.25642/.26778,T.5 selected. Selection score slightly below4k.27364,not held-out winner evidence.13110 frozenval pending;no new training/promotion. |

| BC-V2-8000-CALIBRATED-PLATEAU | 2026-10-25 |13110 passed71val:mean castF1.264 versus4k.276;E/attack improve,R regresses. Keep4k reference,stop duration extension. Added previous-event-only toggle preserving cast ages/default behavior/parameter shapes,regression unrun. Audit replay history dependence before next architecture change;not leakage/closed-loop proof. |

| BC-V2-PREVIOUS-EVENT-ABLATION-QUEUED | 2026-10-25 |13111 fullsuite gate→13112 matched4k previous-event-only removal,source21e2b806. Cast ages retained,seed0/schedule/draws/support same;no test or quality result yet. Inference perturbation control not implemented. Private staging/NFS/no-requeue,awaitcallbacks. |

| BC-V2-PREVIOUS-EVENT-TESTS-PASS | 2026-10-25 |13111 actual log85passed/37.71s,including previous-command mask isolation and cast-history sensitivity.13112 gate cleared;no learned quality/closed-loop result yet,no new jobs/promotion. |

| BC-V2-NO-PREVIOUS-EVENT-AND-DROPOUT | 2026-10-25 |13112 passed4k/71val:raw castmeanF1.190→.152,movewithin2 17.70→6.55%,Q/Eaim5.17/10.04deg. History useful,not deploy/leakage proof. Implemented training-only grouped previous-event dropout/default0,independent RNG preserving draws/cast timers/labels,regression unrun. Next gate and.3 trial,no promotion. |

| BC-V2-PREVIOUS-DROPOUT-QUEUED | 2026-10-25 |13113 fullsuite→13114 matched4k dropout.3 registered,source2dfd229e. Whole-window previous-event masking only,cast timers retained,normal-history finalval. No test/quality/closed-loop result yet;awaitcallbacks. |

| BC-V2-GROUPED-DROPOUT-TESTS-PASS | 2026-10-25 |13113 actual log86passed/37.78s;mask grouping/reproducibility/cast preservation/input immutability validated.13114 dependency cleared;no learned-quality/deploy evidence yet,no new jobs/promotion. |

| BC-V2-DROPOUT-RESULT-AND-SENSITIVITY | 2026-10-25 |13114 passed4k/71val:raw castmeanF1.183 versus.190 reference,Q/Eaim6.13/9.93deg,movewithin2 17.17%. Most normal-history performance retained,robustness unknown.13115 frozen previous-event mask assessment both4k checkpoints registered,source2dfd229e/worker d766e5b8de76,checkpoint/effectiveconfig provenance. No simulated errors/closed-loop or promotion. |

| BC-V2-HISTORY-SENSITIVITY-RESULT | 2026-10-25 |13115 passed71val both4k:missinghistory castmeanF1 reference.190→.076,dropout.183→.153;dropout Q/Eaimstable,moveaccuracy17.17→6.11% remains sensitive. Missing-group robustness only,not policy-error/closed-loop/seed proof.13116/13117/13118 own dropout calibration chain registered,source2dfd229e;candidate only,no promotion. |

| BC-V2-DROPOUT-CALIBRATION-FIT | 2026-10-25 |13116/13117 passed51cal,4k/dropout.3,checkpoint79971010…/dataset consistent. Rate-fitted T1/.7/.5 castF1.18850/.23364/.25621,T.5 frozen;selection-set only,lower thanreference. Await13118 to judge normal-performance/robustness tradeoff,no new jobs/promotion. |

| BC-V2-DROPOUT-VAL-AND-AUXILIARY-PREP | 2026-10-25 |13118 passed71val:calcastmean.263 versusref.276,Rregresses,Q/Eaimimproves;keep robustness candidate/reference distinct. Optional future replay-position/HP targets added,whole-horizon visibility/death/gap/jump masks,existingheads,no INPUTS changes. Tests unrun,lambda.1 isolated trial next. No sim/closed-loop/promotion. |

| BC-V2-AUXILIARY-QUEUED | 2026-10-25 |13119 fullsuite gate→13120 matched4k lambda.1 future-state trial,source7836e18a. Dropout0 isolates auxiliary terms,existingheads/inputschema. Coverage/count/lossmass logged;tests/quality pending,private staging/NFS/no-requeue,no promotion. |

| BC-V2-AUXILIARY-TESTS-PASS | 2026-10-25 |13119 actual log90passed/38.50s,including future-anchor/mask and loss-gradient regressions.13120 dependency cleared;no learned auxiliary quality result yet,no new jobs/promotion. |

| BC-V2-AUXILIARY-RESULT-AND-CALIBRATION | 2026-10-25 |13120 passed4k/71val:rawcastmean.188 versus.190ref,Q/Eaim5.79/10.23deg. Activeauxloss.2185,~10951own/3657enemy/10567HP targetrows perlatebatch. No clear timing benefit.13121/13122/13123 own frozen calibration registered;aux optional,next-event/delay experiment planned/unimplemented,no promotion. |

| BC-V2-AUXILIARY-CALIBRATION-FIT | 2026-10-25 |13121/13122 passed51cal/4k/aux.1,checkpoint033ae259…/actualmanifest consistent. Rate-fitted T1/.7/.5 castF1.18726/.22897/.25431,T.5 frozen;selection-set only. Await13123,no new jobs/promotion. |

| BC-V2-AUXILIARY-VAL-AND-NEXT-EVENT-PREP | 2026-10-25 |13123 passed71val:calcastmean.263 versus.276reference,aux remainsoff. Implemented optional nexttype/conditionaldelay head and targets,causal immediate marginal,no future inputs,currentaim support preserved. Newhead/label factorization jointly changes,tests unrun;next gate/trial,no promotion/B/RL. |

| BC-V2-NEXT-EVENT-QUEUED | 2026-10-25 |13124 fullsuite→13125 4k next-event horizon20 trial registered,source3f25ce66. Unsmooth/noaux/nodropout,causal shared evaluator,temporal labels+emission+delayhead jointly change. Data/draws/aim support fixed;tests/quality pending,no promotion. |

| BC-V2-NEXT-EVENT-TESTS-PASS | 2026-10-25 |13124 actual log95passed/40.26s;temporal target masks,causal marginal probability,teacher independence and conditional-delay gradient checks pass.13125 dependency cleared,no learned-quality claim/new jobs/promotion. |

| BC-V2-NEXT-EVENT-RESULT-AND-CALIBRATION | 2026-10-25 |13125 passed4k/71val/6.989Mparams:rawcastmean.170 versus.190ref,nearAPworse,exactAPallcastsbetter(E.027→.054). Delaylossactive1.571;labels/smoothing differ,not established winner.13127/13128/13129 immediate-probability calibration chain registered,source3f25ce66;no baseline switch/promotion. |

| BC-V2-NEXT-EVENT-CALIBRATION-FIT | 2026-10-25 |13127/13128 passed51cal/4k/horizon20,checkpoint9f53e19b…/manifest consistent. Immediate-rate-fitted T1/.7/.5 castF1.16848/.20172/.21086,T.5 frozen. Selection-only,13129 pending,no switch/new jobs/promotion. |

| BC-V2-NEXT-EVENT-VAL-AND-MEMORY-PREP | 2026-10-25 |13129 passed71val:calcastmean.202 versus.276ref,allcastF1lower;rejectthisconfiguration. Added optional GRU-only periodic evalreset with continuous emitterRNG/defaultfullstate;regression unrun,next frozen256reset diagnostic. No statecarry training/closed-loop/promotion. |

| BC-V2-MEMORY-RESET-QUEUED | 2026-10-25 |13130 fullsuite→13131 frozen4k GRUreset256 registered,sourcee0322297,same71val/T1/sample0,PRNG/historyinputs preserved. Compare savedfullstate13103,not statecarry training. Tests/quality pending,private staging/NFS/no-requeue,no promotion. |

| BC-V2-MEMORY-RESET-TESTS-PASS | 2026-10-25 |13130 actual log96passed/40.28s,resetboundary/RNGstream regressionpasses.13131 dependencycleared,no memory-quality claim/new jobs/promotion. |

| BC-V2-MEMORY-RESULT-AND-SIMPLER-KIT-CONTROL | 2026-10-25 |13131 passed71val:full/reset castmean.18966/.19102,othermetricsalmostsame.13133 early/later boundarypaired diagnostic queued.13132 Garen185/25/26 train1600→owncal→val queued,sourcee0322297;selfcast-kit timing diagnostic,not equaldata/qualitypromotion. No sim/B/RL. |

| BC-V2-MEMORY-BOUNDARY-RESULT | 2026-10-25 |13133 passed:phase32:82 full/reset castF1.19345/.19475,later.18572/.18580,APnearlysame. Burn32 excluded both,not startup test;no large supervised-boundary deficit,so statecarry lowpriority. Await13132 Garen,no new jobs/promotion. |

| BC-V2-GAREN-RESULT-AND-TRANSFER-PILOT | 2026-10-25 |13132 passed185/25/26:calQ/W/E/RF1.174/.105/.270/.075,R40events/pointer92.5%;timingweakdespitesimplekit,notequaldataAhri comparison.13134 array4max2→13137 convert/finalize512other-midtrainperspectives registered,planf2c00b915126/sourcee0322297,allAeval/cal matches excluded,Buntouched. Conversion only,mixedtraining/warmstart notimplemented,no promotion. |

| BC-V2-TRANSFER-DATA-PASS-AND-INITIALIZER | 2026-10-25 |13134all4/13137 passed512views/499matches/3.391Mvalidframes,.368GBNPZ,finalmanifestverified. Addedalltrain mode/strictweights-onlyinitializer+provenance/freshoptimizer,resumecontractpreserved;tests unrun. Planmixed1000→Ahri4000 versusAhri1000→Ahri4000 samephaseLR. Conversiononly,no transfer quality/promotion. |

| BC-V2-TRANSFER-RECIPE-QUEUED | 2026-10-25 |13139 tests→13140 mixed512pretrain1k;13141 Ahri1kdonor13096→4kcontrol;13142 mixed1k→4kafter13140. Source75ecb810,phase2lr1e-4/freshoptimizer/same470train/71val,5000totalstepsboth. No80/20 finetune/allchamp proof/calfit yet. Private snapshots/NFS/no-requeue;tests/quality pending,no promotion. |

| BC-V2-TRANSFER-INITIALIZER-TESTS-PASS | 2026-10-25 |13139 actual log98passed/40.09s,weights-only/provenance/architecture/tree/shape checks pass.13140/13141 gatescleared,13142 after13140. No transfer-quality claim/new jobs/promotion. |

| BC-V2-MIXED-PRETRAIN-COMPLETE | 2026-10-25 |13140 passed1000updates/512views/15519windows/664.62s,datasethashverified,NFSdonor c164a6da…. TrainingbuttonNLL1.040→.663,not target/transfer quality.13142 fine-tune dependencycleared,13141control pending;no promotion/new jobs. |

| BC-V2-STAGED-AHRI-CONTROL-RESULT | 2026-10-25 |13141 passedphase2 4k/1910.93s,weights-onlyAhri1k donor/lr1e-4 verified. Rawcastmean.16873,Q/Eaim7.41/11.43deg;stagingnotautomaticimprovement. Matchedcontrolfor13142,not equalbudget original4k. Awaittransfer,no new jobs/promotion. |

| BC-V2-TRANSFER-RAW-RESULT-AND-PAIRED-CALIBRATION | 2026-10-25 |13142 passedphase2:rawcastmean.16748 versusstagedcontrol.16873,EnearAPimproves/aimworsens. No pilot timinggain yet.13148 owncalfits/frozen71val/500pairedbootstrap registered,source75ecb810;13147 intentionallycancelledpending tofix localfitstaging,do not restart. One seed,no promotion/allchampverdict. |
