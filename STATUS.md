# STATUS (rewrite in place; last edit 2026-09-29 18:45 UTC, Codex)

**E37 RUNNING (job1774):** Restore the exact E36b trapped-wave state and recurrent
history; compare40s frozen branches with W/E/Q and alternate movement.
CPU Slurm preserves the reference backend; desktop is back for subsequent
learning work. No training configuration changed.

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
