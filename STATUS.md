# Status

**Current: 2026-10-17.** This worktree contains the 26.19 modern JAX world in `lanerl_jax/modern/`,
its trainer in `modern/train.py` + `modern/rl/`, tools in `ops/modern/`, and specs in `docs/modern/`.
Start here, then read `docs/CODEMAP.md` and `docs/modern/WORLD_IMPLEMENTATION.md`.
Detailed evidence and job history live in `docs/JAX_FIDELITY_LEDGER.md` (MODERN-027..050).

## Champions and world scope

14 kits registered: Garen, Jax, Nasus, Renekton, Darius, Malphite, Aatrox, Fiora, Yone, Ambessa,
Camille, K'Sante, Jayce, Ezreal. Known limitations remain explicit in `docs/modern/CHAMPIONS.md`.
Ornn was dropped at Dani's direction. The accepted cross-role plan is `docs/modern/ROSTER.md`;
11 planned additions remain after Ezreal. Simulator profiling/throughput is the immediate priority. Caitlyn is research only: exact trap/net attack refund and recoil evidence remain
unresolved. Lux has eight replay games extracted and child-spell timing evidence; no Lux kit registered.

The world supports two champions, one per team, not 5v5. More champions per team and ally targeting
remain prerequisites for the complete jungle/support roster, especially Lulu.

## Completed and validated

- Each world compiles only the matchup's kits or an explicit roster (MODERN-040).
- Ezreal + kit-pruning integration passed 998 tests across 69 modules (6150..6152), with identical
  full/top 3600-tick goldens at all 12 checkpoints (6153).
- Review fixes (MODERN-046, `76a4d59f`): independent ranged launch/hit metadata, accurate sequential
  damage-overflow counters, shield-order reduction hoisted out of the packet loop, stale comments trimmed.
  106 focused checks passed; all 12 golden checkpoint records match the accepted reference exactly.
- Follow-up (MODERN-047, `0e54c2d3`): simultaneous basic/basic and ability/basic hits preserve separate
  on-hit records, including item/rune/kit effects and Red Buff targets; combat resolves once per tick.
  139 focused checks passed; all 12 full/top golden records match exactly with zero overflow.
  Compiler estimates: Ezreal FLOPs +0.39%; GPU runtime cost has not been measured.
- Shield valuation (MODERN-049, 2026-10-17) selects the target row before decay arithmetic.
  Desktop CPU compiler7791 confirms smaller HLO work; restricted top-lane Garen/Ezreal tick
  FLOPs estimates fall 1.16%/1.29%, with unchanged compiled memory. GPU throughput remains unmeasured.
  Validation7792 completed: 211 focused checks passed; all 12 full/top 3600-tick golden
  records match the accepted reference exactly, with zero overflow. Integrated as `38d6d04e`.

These are validation gates for their respective changes, not an additive test total. The full 69-module
suite preceded the latest review fixes; those fixes have focused regression and golden coverage.
No substantive modern-roster training experiment has run yet.

## Measured performance (before the latest review fixes)

RTX5080, kit pruning: top1024 134.7k ->148.8k env-ticks/s (+10.5%); top4096
168.3k ->178.6k (+6.1%); full1024 42.7k ->46.9k (+9.8%). Jobs6722/6723 completed.
Matched10Hz rollout + policy, top1024: modern36.0k vs legacy lane-rl/jax21.7k environment decisions/s
(+65.7%). Modern4096 reaches35.8k, showing saturation. These are different rulesets/scenarios,
not equivalent-game or full-training speedups. Results: `modern043-pinned-v3-TEMP/results-6734/`
under `/mnt/nfs/shared/champ-expansion/`; 35 cells retained, final cell cancelled after6h30.

## Ownership and active work

Dani clarified ownership on2026-10-17: this thread owns the simulator, champions, correctness and sim
throughput under training workloads. Other agents own policy/PPO/learner and the rest of the stack.
Own PPO diagnostic7588 was stopped (final elapsed21m20s); partial results are preserved for handoff at
`/mnt/nfs/shared/champ-expansion/modern045-memory-baseline-TEMP/results-7588/`.
No learner/allocator changes were made. CPU diagnostic prerequisite7587 passed previously.

MODERN-048 improves sim measurement: five consecutive synchronized samples, normalized game-time rates,
explicit timed game window, minion occupancy, memory/overflow and frozen-game checks. Optional scripted
ability attempts exercise kits; the queued comparison will include later lane combat after5400 warm ticks.
All five timing/baseline-key/order tests and compiled CLI validation7793 passed on desktop;
exclusive GPU comparison7796 waits afterok:7793, and profiler7795 waits afterany:7796. Earlier unstarted
7794 was superseded to use Jayce/Garen for the ranged case (the tool uses top-lane rune recommendations;
Ezreal only has bottom-lane research). Comparison uses identical driver over baseline78abf40f and candidate
80bd8f0c, with default/later ability workloads, top1024/4096, full1024 and ranged Jayce/Garen1024.
Immutable sources and driver hash: `/mnt/nfs/shared/champ-expansion/sim048-throughput-TEMP/`.
MODERN-049 shield optimization is integrated and validated. Exclusive GPU comparison7797 waits afterany:7795
and isolates that change: baseline80bd8f0c versus optimized38d6d04e, with identical drivers and workloads.
Pinned sources: `/mnt/nfs/shared/champ-expansion/sim049-throughput-TEMP/`. Jobs7796/7795/7797 remain queued;
no new GPU throughput result is available. Policy/PPO code is outside this thread's scope.

MODERN-050: perf-guard baseline keys now distinguish packet capacity, item allow-lists, lane-structure
reduction, champions and roster. Default keys remain compatible. Five lightweight summary/key checks pass;
no simulator behaviour changes. GPU jobs7796/7795/7797 are still pending on2026-10-17.

Next actions:
1. Validate the revised sim benchmark and remeasure pre/post review code under identical benchmark drivers.
2. Profile current later-lane combat, optimize measured sim hotspots, and retain exact fidelity for refactors.
3. Test smaller packet capacity, item allow-lists and lane-structure reduction against comparable sim workloads;
   keep every overflow counter at zero and separate opt-in scenario changes from exact optimizations.
4. Resume one champion at a time, finishing evidence before implementation (Caitlyn/Lux researched).
5. Expand multiple champions/ally targeting before claiming complete5v5/support implementation.

Existing timings above predate the latest review fixes. New speedups remain unmeasured.

## Modern behaviour cloning (branch modern/bc, thread 200b87b6)

**2026-10-22 rollout12873 COMPLETED/0.** All43 BC checks passed;8 Garen
and8 Jax nine-minute episodes completed against AFK opponents with zero overflow.
Mean CS2.25/1.25, deaths0.25/1.0, kills0.875/0.375; mean base fraction0.31%/1.40%.
This confirms integration, not useful policy quality: the tiny checkpoint only
memorized fixed windows and already failed held-out cast generalization. Buying
remains disabled; no inference that the broader economy mismatch is resolved.
Private scratch directory was removed on exit; NVMe still638GB free.

**Full-length12883 COMPLETED/0.** All44 checks passed;8 full8100-decision
games converted/finalized and both splits cache-loaded (40799 valid train,
14414 valid val). Conversion+save2.9–5.7s/game; max RSS2.34GiB. Dataset6.35MB,
cache69.91MB for64800 decisions; sample extrapolation475.9GiB full cache.
This fits638GB free NVMe with reserve, subject to actual full-corpus occupancy.
Private scratch cleanup verified. Result: `v6-pilot-20261022/full-length-result.json`
under `/mnt/nfs/shared/modern-bc`.

**2026-10-23 full conversion12886 COMPLETED/0,8/8 shards.** Verified58470
NPZ+metadata pairs,7309 replay bases, no failed games in shard receipts or failure
records. Converter6b20bdcf throughout. Per-shard5.7–6.0h; full run~23h at2 workers.
Scratch cleanup verified,638GB free. This is the approved heuristic-fog dataset.
Dani explicitly says do not wait on decoder work; continue standalone BC.
RL/PPO and economy work are removed from the current plan.

**Full audit13005 COMPLETED/0.** All58470 player-perspective arrays from7309
unique26.19 matches passed source/fog/converter, split separation, shape,
finite-or-missing and action/cell checks.473607000 total decisions;
340720945 train/38055900 held-out valid labels. Compressed NPZ44.24GB;
audit5629s (~94min). Durable `audit-result.json` passed.

**BC13009 FAILED after automatic requeue, checkpoint preserved.** First
attempt built503814199620 bytes of cache and trained10000 steps; sampled held-out
event+click NLL7.398, button accuracy91.23%, click distance12.71cells, but cast
recall remains weak (Q3.45%,E2.15%,W0.57%,R0.09%). No useful-policy claim.
Slurm reports Restarts=1. The restarted wrapper allocated fresh staging while
its first attempt's orphaned private503.8GB cache remained; capacity guard stopped
it before duplication. Old private stage exists and no process uses it.
Checkpoint `ckpt_000010000.msgpack` in
`/mnt/nfs/checkpoints/lanerl-jax/modern-bc/modern-bc-v6-allroles-20261023-145300-?`.
Resume must retain the three-epoch total ceiling, not add three more epochs.
Trainer now replays shuffle/augmentation RNG draws and skips completed batches.
Resume smoke13011 COMPLETED/0:45 checks; old4step checkpoint resumed to
original64-step ceiling with four completed batches skipped. Resume budget path
verified.13012 failed before touching donor data because desktop `sacct` used
an unreachable localhost accounting service. Terminal13009 state is now recorded
from danilogin sacct; runtime verifies that receipt, donor ownership and absence
from live `squeue`, avoiding the compute-node accounting service. This exact check
passed on desktop. No trainer/source/data changes since resume smoke.

**2026-10-24 BC13013 COMPLETED/0, early-stopped at38500 total steps.**
Best sampled checkpoint36500, event+click NLL7.1787; four subsequent checks
failed minimum improvement. Best sampled button accuracy91.32% vs82.15% majority,
click exact6.00%, distance12.10cells. Q/W/E/R argmax recall7.54%/1.43%/4.92%/0.40%;
joint button+click exact Q0.17%,E0.52%,W0.02%,R0%. Not yet strong standalone BC.
Full epoch2 held-out accuracy91.17% vs82.22%, event NLL0.9158, click NLL6.3211,
click exact5.14%, distance12.56. Final best36500 still needs full held-out scoring.
Run `/mnt/nfs/checkpoints/lanerl-jax/modern-bc/modern-bc-v6-resume-13013-20261023-194821-?`;
`best_partial.msgpack` is preserved independently of rolling checkpoints.
Private scratch cleanup verified;638GB free NVMe, no orphan13013 directory.

**Best checkpoint evaluation13014 COMPLETED/0**, all5880 held-out
perspectives,33233804 scored decisions. Button accuracy91.34% vs82.22% majority;
click exact5.88%,distance12.37. Q/W/E/R argmax recall10.83%/2.05%/3.95%/0.57%.
Expected sampled Q/W/E/R rates4.19/2.04/2.20/0.355% vs human0.679/0.311/0.320/0.0565%:
~6–7x over-casting, not simply absent cast probability. Mean probability assigned
to true Q/W/E/R labels19.55/10.07/12.18/6.10%; expected joint button+click within
2cells peraxis0.87/0.33/0.67/0.19%. Cast weighting5 and held-move weighting0.3
produce uncalibrated probabilities; increasing cast weights is not the next fix.
Shared click head is dominated by movement; investigate conditioning aim on action.
Scratch cleanup verified,638GB free. Native FoW/RL remain deferred.

**Readout probe13015 COMPLETED/0,46 tests passed.** Independent64-perspective
assessment: fitted button calibration lowers NLL0.3731→0.2912, accuracy91.38→92.08%;
Q sampled share4.19%→0.686% vs0.696% human. W/E/R rates also much closer, while
argmax cast recall drops nearzero. Calibration repairs rate distortion, not
missing cast intent. Fit used64 different validation replay perspectives/bases.
Frozen action-conditioned aiming improves Q distance22.89→15.98cells and
E17.17→14.20, but Q within2cells7.43→10.59%, E19.16→17.06%; W/R NLL worsens.
Summoner/ward near-target rates reach100% on small supports65/97/130, suspiciously
suggesting caster-centered labels rather than general aiming ability.
No production conditional-head/calibration promotion or full retrain yet.

**Target audit13017 COMPLETED/0**,26 perspectives,228 raw slot4/5/12 casts.
224 used0xd0 points:217 within50units of caster, with converted summoner/ward
clicks clustered near center. No resolved target points in these slots.0x2c
usually differs substantially but is not promoted as a qualified replacement.
The apparent100% near-target readout gain is invalid as evidence of aiming.
Removed slot4+0xd0 fallback in replay reader. New conversions carry
`label_target_contract=received-targets-no-d0-v2`; qualified unit/resolved points
remain permitted, unresolved targets excluded. Existing v4 arrays lack the marker,
so loader conservatively masks summoner_d/f/ward labels rather than inventing
noops or silently training origin clicks. Observation/cache arrays are reused.
Excluded-target counts are recorded in training manifests. No23h reconversion.

**Corrected probe13018 COMPLETED/0,48 tests.** Bad legacy targets excluded.
Calibration assessment NLL0.3698→0.2859, Q sampled4.19→0.672% (human0.696%);
argmax cast recall still nearzero. Conditional readout not promoted: Q distance
22.89→16.06 but within2 only7.47→9.76%; E within2 worsens19.25→16.61%,
W/R NLL worsens; movement slightly worsens. Frozen global features plus action
conditioning alone are not an adequate aiming fix. No new full run.

**CPU13019 COMPLETED/0:**48 BC tests and544 decoder SHA256 checks passed.
All192 perspectives reproduce the original lagged entity/padding arrays and
accepted button/click labels. Exact received target IDs mapped to85,795 visible
unit clicks of85,872 eligible (58,908 train/26,887 assessment);77 targets absent
from lagged visible rows excluded.192 perspectives took952s plus tests/staging.
Evidence: `v6-full-20261022/pointer-targets-13019/results.json`.

**GPU13020 COMPLETED/0:** frozen contextual pointer improves heldout exact
unit selection66.75% (screen snapping)→73.67%,23724 scored unit clicks;
new event orders60.70→70.78% (7066). Entity-only pointer65.14%, showing benefit
from frozen recurrent context. Screen within2 cells68.46→74.59%; mean Manhattan
screen distance5.08→3.72. Q/W/E within2 rise39.5/51.9/49.9→62.1/68.3/63.5%;
R worsens67.7→38.7% on31 examples. Context pointer train76.0% vs73.7% assessment,
but spell-specific train/heldout gaps are large. Sourcebc6b8e14,55s job worker.
Evidence `v6-full-20261022/pointer-probe-13020/results.json`. This supplied oracle
button and unit-click type, not end-to-end policy evidence. No promotion.

**GPU13021 COMPLETED/0,71.6s worker:** hard unit/ground router rejected.
On524288 sampled clicks from512 fresh heldout perspectives, actual-button hybrid
within2 falls24.509→24.429%; joint button/near-click23.718→23.638%; screen
Manhattan distance12.235→12.332. Unit-only near improves61.20→61.96%, but ground
near falls21.213→21.057% (ground91.76% of clicks). Router recognizes only19.38%
of unit clicks and misroutes2.71% ground clicks. Oracle button does not rescue it.
Q/E small gains do not justify overall degradation. No pointer/router promotion.
Evidence `v6-full-20261022/hybrid-pointer-13021/results.json`.

**GPU13022 COMPLETED/0,99.5s worker:** neither frozen residual head beats
unchanged baseline under selection. Factorized soft CE6.56737→6.58335 at500;
joint6.56737→6.56862 at1000 (best trained soft CE, still worse). Both select step0
and stop at1500, so assessment remains baseline: NLL6.17073, within2 24.174%,
joint23.348% across524288 clicks. Joint exact NLL improved on selection but not
the prespecified smoothed criterion. This probe smoothed unit clicks too; native
BC uses exact unit targets, so it does not rule out joint heads under native loss.
No frozen readout/pointer/router promoted. End frozen-feature-only probes.
Evidence `v6-full-20261022/joint-click-13022/results.json`.

**GPU13023 FAILED after4m10s:** completed continuation control, then reporting
failed on NumPy float32 JSON serialization before joint arm started. Initial
native loss equivalence passed (6.255844 vs6.255849), base gradient norm5.404 and
parameter delta0.0620 prove actual full-gradient updates. Control selection6.84000
initial;200/400/600 scores6.85434/6.85305/6.87758, so baseline0 retained and stopped.
Assessment of baseline:1365775 valid frames/1127853 clicks, click NLL6.16265,
within2 24.354%, joint23.634%. Best control checkpoint persisted to NFS. Recovered
control score/gradient/assessment evidence from actual log into
`endtoend-control-13023-recovered.json`; no successful candidate result exists.

**GPU13024 FAILED after1m09s before candidate training:** saved parameter
identity and continuous loss reproduction checks passed, but hard within2-rate
assertion was too tight:274675→274691 successes of1127853 clicks, absolute
rate delta0.000014186 (0.00142 percentage points). The joint arm remains unrun.
Exact reason for the argmax difference is not established; near-tied numerical
predictions are plausible, not proven. Do not count this as candidate evidence.

**GPU13025 COMPLETED/0,316s worker:** full-gradient joint model chooses step800.
Selection6.84000→6.74844; actual-button assessment click NLL6.16265→6.07386,
joint near-click23.636→23.916%, button accuracy91.490→91.651%. But Q/W/E recall
7.568/2.624/3.657→5.692/1.858/2.026%; per-class NLL also worsens. No promotion.
Every sampled selection checkpoint has E NLL degradation>0.05 despite aggregate
improvement, so aggregate-only selection was insufficient. Evidence
`v6-full-20261022/endtoend-click-13025/results.json`; initial native losses,
nonzero full-base gradients/updates and control reproduction passed.

Implementing fixed-sample cast guard: default per-QWER NLL regression limit0.05,
minimum100 events, blocks best checkpoint replacement and consumes early-stop
patience. Retain initial baseline as best_partial even when no candidate passes;
epoch selection uses same guard. Tests and real-result regression replay pending.

Input audit: all192 pointer-subset perspectives have zero availability in QWERDF
cooldown columns. GRU training windows reset at256 decisions (25.6s). Preparing
received-own-cast age sidecars on128 train/64 assessment perspectives. Exact replay
receipt elapsed seconds, unknown before first receipt, canonical summoner mapping,
original observation lag. These are not cooldown duration/readiness and do not
change production observations. Causality/mapping tests and CPU source audit next.
**CPU13026 COMPLETED/0:**53 BC tests pass; real13025 regression replay rejects
all sampled candidate selection checkpoints and heldout Q/W/E.544 decoder hashes
verified.192 age sidecars completed in771s, preserving received timestamps,
canonical summoners, NaN before first receipt and exact stored observation lag.
Heldout410529 supervised frames: known Q/W/E/R/D/F histories374832/331986/341601/
167489/261368/306740. Of known histories,25.7/44.0/40.6/74.5/93.3/90.9% extend
beyond25.6s. This proves extra history exists, not that it improves prediction.
Evidence `v6-full-20261022/cast-history-13026/results.json`.

**GPU13027 COMPLETED/0,82s:**99 fit/29 selection/64 assessment perspectives,
replay-base disjoint fit/selection. Bias/context/context+age all select step0 and
stop1000. Selection event NLL starts0.48815, every trained checkpoint worsens
and trips all QWER guards. Assessment0.300799 NLL is unchanged inverse-class-weight
baseline, not a learned gain. Shuffled history identical because selected history
weights stayed zero; falsification is uninformative. No predictive age benefit
shown. Probe optimized natural all-frame loss but selected event-only loss; this
objective mismatch limits interpretation. No input/model promotion.

**GPU13028 COMPLETED/0,65.8s:** age-bin calibrator selects1000; bias selects0.
Assessment weighted button NLL0.403222→0.402078; unweighted0.300801→0.298085,
but event NLL0.502648→0.503030 and R NLL4.7248→4.7759 exceeds guard0.05 on
heldout data. Shuffled ages retain most gain (weighted0.402559); age-specific
benefit only~0.00048 weighted NLL. No production age input/calibrator promotion.
Current BC remains best36500; standalone-quality goal still unmet. Stop age-head
experiments. Decoder read-only recheck: unchanged043b9a83/v18; production exports
unchanged, partial timer estimates do not supply newly qualified readiness data.

Replay-only behavior audit: deterministic first qualified Q/R or Q/E events after
180s from SHA-sorted heldout Garen/Jax/Ezreal/Lux perspectives,8 clips of8.1s.
Fixed replay observations with original model predictions, click heatmaps and
optional previously fitted calibration. Predictions do not change replay states;
no simulator or training. Produces actual observation/prediction evidence before
any further architecture change, not standalone rollout success evidence.
**GPU13029 FAILED** during first clip extraction: packed entity loader accepts
slices, not scalar indexes. No training or completed artifact. Fixed to one-row
slice and added actual packed-loader shape check before inference.
**GPU13030 COMPLETED/0:**8 deterministic replay clips exported622451 bytes.
Interactive comparison `replay-behavior-13030/replay-comparison.html` checked at
728px dark/360px light; real scene selector, calibration selector and time scrubber
checks pass without browser errors. All8 center presses choose Move by raw argmax;
Garen R/Jax Q click targets are near human target, Ezreal Q/R and Lux Q are far.
Illustrations only, not a fresh generalization test or closed-loop evaluation.

Next decisive run: fresh clean BC baseline on existing58470 perspectives, rather
than more readouts of the pre-target-repair checkpoint. Conservative target masks
reuse v4 observations; disable unqualified D/F/ward action outputs. No reconversion,
new inputs, simulator state or RL. This should have preceded extended head probes.
Added explicit cast-guard bootstrap start so fresh training is not judged against
an untrained uniform class prior; resume default remains immediate. Source1c5fab88.

**2026-10-24 clean smoke13031 COMPLETED/0:**54 BC tests passed. Actual training
completed64 steps/four interim evaluations with saved checkpoints and cast-guard
activation; forced plateau stopped at4 steps/two checks and retained step0 best.
D/F/ward output-mask assertions passed. Durable `v7-clean-20261024/smoke-result.json`
is passed. Full clean13032 released and RUNNING; no optimizer metrics yet.
Existing converted replay arrays are reused; no reconversion, simulator or RL.

**GPU13031 registered:** full BC tests + real monitored-training/forced-stop smoke;
checks best_partial baseline preservation, guard activation and unsupported output
masking. gpup2CPUs/8GB/20min. Durable `v7-clean-20261024/smoke-result.json`.
**GPU13032 submitted afterok:13031:** fresh clean v7 on58470 player perspectives
from7309 patch26.19 matches (52590train/5880validation). Same185/32x20/6 inputs,
no reconversion/history/architecture additions. Unsupported D/F/ward outputs off;
legacy target masks active. Batch128/window256/burn32, maximum3 epochs, eval500
on512 fixed validation perspectives, patience4/min_delta0.001, cast guard start2000
with max0.05 NLL increase/min100 events. gpup1GPU/4CPUs/24GB/16h limit; no requeue,
TERM forwarded so NFS checkpoints persist before cleanup. Scratch684.7GB free on
inspection, enough for private44.2GB source +~503.8GB cache +20GiB reserve.
Logs/source/scripts in `/mnt/nfs/shared/modern-bc/v7-clean-20261024`; full job starts
only if smoke succeeds. Early stop rather than mandatory epochs. No simulator/RL.
Replay comparison published inline after desktop/mobile and interaction checks.

Direct unit-target selection probe: Labels now expose actual replay target IDs for explicit and held
unit clicks without changing existing numeric/button labels. Reconstruct slot
identity only on this small subset, require observation arrays match existing
cached inputs, and map actual target IDs into the lagged visible slots. No nearest-
coordinate guess supplies supervision. Learn a pointer over entity rows against
baselines before considering any production action representation change.

**GPU smoke13007 COMPLETED/0:**45 BC tests passed; actual training verified
baseline and four within-epoch evaluations/checkpoints, normal completion at64
steps, preserved best sampled checkpoint, and forced plateau stopping at4steps.
These are workflow checks on one train/one held-out perspective, not policy
quality evidence. Durable `train-smoke-result.json` passed. Source137e30b3.
Full GPU BC13009 was released by `afterok:13005:13007` and asserts both durable
passed receipts before staging.24GB/4cores,16h cap,3 epochs maximum (no minimum),
500-step checkpoint/evaluation on512 fixed champion/role-spread held-out games;
early stop after4 checks without0.001 improvement in event-button+click NLL.
Nonfinite training/gradients stop immediately. Best sampled checkpoint preserved
separately, full validation at epoch boundaries. Enemy/stat dropout0; cast button
weight5, summoners3, ward/recall/stop2. No cooldown-based timing smoothing.
Training independently copies dataset+source to private NVMe; preflights projected
cache with15% headroom plus20GiB reserve and checks reserve throughout cache build.
Durable checkpoints on NFS; private staging cleaned on exit. Native decoder v18
is partial and deferred per Dani. No RL/PPO or economy work in current scope.
Logs `modern-bc-v6-monitor-smoke-13007.out` and `modern-bc-v6-train-13009.out`
under `/mnt/nfs/shared/modern-bc/v6-full-20261022`. Await automatic receipts.

**Full-length12880 FAILED/1 after decoder staging verified544 files.**
The first conversion reached provenance recording, which assumed a Git worktree;
isolated archives have no `.git`. Converter now accepts a validated full SHA
from archive `.bc-source-commit`, retaining Git dirty tracking in checkouts.
Regression test covers valid, missing and malformed archive records.
A new pinned archive and preflight will precede retry submission. No sizing
result or finalized full-length dataset was produced. Previous attempt details:
Attempt12879 failed before conversion: `cp -a` could not preserve source
permissions on scratch. Replaced with independent metadata-free copies and
verified all544 decoder files against snapshot SHA256 hashes on desktop.
Cleanup of failed staging verified. No BC implementation/data semantics changed.
Eight selected champion-games retain the existing train/val split, now30..840s
(8100 decisions each). Measures conversion time, peak RSS, dataset/cache bytes;
finalizes and loads both splits. Source snapshot2258de68 and decoder scripts
copied/hashed before launch; data/code/cache use independent private scratch.
Outputs persist under `v6-pilot-20261022/full-length-dataset` and
`full-length-result.json`; log `modern-bc-v6-full-length-12880.out` there.
Inspect automatic receipt before scaling to full corpus. No broad training/PPO.

**2026-10-22 rollout12869 failed in base-fraction reporting.** All43 checks
passed and offline evaluation completed. Fixed-window casts34/35, but targeted
joint button/click exact9/19; held-out button accuracy74.6% versus82.5% majority,
with zero observed held-out cast recall. This tiny memorization checkpoint does
not generalize. Gate incorrectly used `Transition.self_vec`; corrected to
`obs_self`. Retry Garen/Jax episodes from an immutable source snapshot.

Scratch cleanup authorized by Dani: removed obsolete474GB
`/scratch/dani/bc-allroles-cache-TEMP` (only cache .bin/.json files; no active
Slurm jobs), plus this thread's completed pilot/eval staging. Free NVMe now638GB.
The current~502.6GiB full-cache estimate fits with20GiB reserve; actual capacity
must still be checked before staging. Compaction is optional pending measured
need. Host guidance now permits disposable scratch cleanup, requires independent
copies/private job paths, code snapshots, capacity checks and exit cleanup.
No periodic cleaner was installed; the old30-day cleanup claim was unverified.
Full conversion/training remain unstarted. Isolated rollout retry **12873** is
registered on gpup (50min), source snapshot2258de68, private scratch/cache and
EXIT cleanup. Log: `/mnt/nfs/shared/modern-bc/v6-pilot-20261022/modern-bc-v6-rollouts-12873.out`.
Inspect its automatic completion receipt next.


**2026-10-22 rollout attempt12866 stopped before episodes.** All43 checks passed,
but offline evaluation could not JSON-serialize a NumPy float32 metric. Evaluation
now returns native scalar numbers, with a serialization assertion. No checkpoint
or data change; retry the evaluation/rollout job. Capacity check: desktop NVMe
has165GB free, NFS718GB. Full cache preparation must fit that limit and preserve
existing data; full conversion/training remain unstarted.


**2026-10-22 memorization12859 passed.** All43 BC checks passed, then512
updates reduced fixed-window loss10.399→1.405, click NLL8.556→1.379 and click
distance54.85→3.03 cells. Cast accuracy34/35 (97.1%); button accuracy92.1%.
This is memorized training data, not generalization. Checkpoint:
`/mnt/nfs/checkpoints/lanerl-jax/modern-bc/modern-bc-v6-overfit-20261022-113216-2e9dd8f6/ckpt_000000512.msgpack`.
Next: targeted-cast joint accuracy, held-out pilot evaluation and8 Garen /8 Jax
full-lane episodes. Found and fixed a rollout contract mismatch before starting:
v4 sim observations now default tier0, matching unknown-tier BC (older profiles
retain tier8). Gate will report alive time within2500 units of fountain. Buying
remains masked in this baseline, so economy mismatch remains explicit. No full
v6 run is submitted.


**2026-10-22 GPU diagnostic attempt12858 stopped before training.** CPU preflight
had42 passing checks and one old assertion expecting only the prior evaluation
buckets. Updated the fixture/assertion to check champion/role groups and joint
button-click metric bounds. No optimizer step ran. Re-submit the same diagnostic
with the corrected test; source/observation implementation remains9b1a8a25.


**2026-10-22 v4 implementation and conversion pilot passed.** New direct replay
formatter replaces world-state construction. Unsupported fields stay NaN;
v4 policy consumes missing masks and records columns never observed during BC
so they also remain masked in RL. Fixed timer scaling, unknown tier, causal
positions, visible-target label filtering, versioned per-game source checks and
cache identity are implemented. Historical datasets remain rejected. Job12851
COMPLETED/0:41 BC checks and16 early/late windows across six champion/role cases,
6285 valid train /2330 val decisions, finalized and cache-loaded. Small additional
source/finalization/cache changes have30 focused checks passing; the next GPU
job will rerun the full BC suite before the memorization diagnostic. No full v6
training or rollout acceptance yet. Current implementation/evidence supersedes
the interim source holds below; see `docs/modern/BC.md` section14.


**2026-10-22 source check passed; more defaults removed.** Job12796 COMPLETED/0:
one actual export, all ten actors,3399 unqualified broadcast records and2377
separate cast schedules with unqualified clock alignment. Removed copied
structure positions/full-health/life defaults; matched turrets now use received
state and missing HP stays unknown. Summoner cooldown tables and sim recall
channel duration are removed; missing timer/channel fields stay unknown. 27
focused CPU checks pass. Remaining work: cooldown/descriptor normalization,
progression/quest/trinket fields, reader-default/causal-position handling and
policy availability masks. No active diagnostic, reconversion or v6 training.
Evidence: `docs/modern/BC.md` section13; rough fog remains approved.


**2026-10-22 countdown probe completed.** Job12794 exit0: all36 sampled casts
across the six requested champion-slots have separate cast-schedule payloads
that BC ignored; their clock-alignment qualification is false. Nearby broadcast
operands also explicitly mark timer/slot semantics unvalidated. Reader now
requires matching packet-level timer/slot evidence, no longer decoder ancestry,
and inventories cast schedules separately. 25 focused CPU checks pass. Real
export reader check **12796** submitted on desktop CPU, registered to this chat;
inspect its receipt next. Details: `docs/modern/BC.md` section12. Other source and
availability repairs remain; no reconvert/v6. Historical audit percentages are
coverage measurements, not proof of authoritative countdown semantics.


**2026-10-22 heuristic fog approved.** Dani permits approximate fog only. Added
`bc/fog.py`: current live replay allies supply circular sight (champion/turret
1350, minion 1200), including allied champions outside observation slots. No sim
fog call, wall/brush/ward/stealth/reveal mechanics or template-only sight targets;
per-game provenance identifies the approximation. 21 focused CPU checks pass.
Other source holds remain; no reconvert/v6 has started. Storage-aware cooldown
probe job **12794** submitted on desktop CPU, registered to this chat; inspect its
automatic receipt/output next, then continue non-fog source repairs. Evidence and
remaining limitations: `docs/modern/BC.md` sections 10–11. The prior unresolved
fog-choice statement below is superseded by this approval.


**2026-10-22 replay-only hold.** Dani requires all BC inputs/labels to use replay
evidence. Full source inventory: `docs/modern/BC.md` section 10. Beyond own spell
cooldowns, the existing formatter uses sim fog, structure defaults, kit cooldowns
hidden in champion-table descriptors/normalizers, summoner/recall/XP/skill rules,
and quest/trinket defaults; missing values have no policy availability masks.
Removed both own-spell kit cooldown fallback paths and kit/catalog targeting
fallbacks; unknown countdowns remain NaN and casts invalidate stale timers.
Conversion, relabeling, finalization and training data loads now fail explicitly
before IO, including skip-existing/cache paths. v1-v5 data/checkpoints are
preserved as historical mixed-source results; **no reconvert or v6 is running**.
17 focused CPU checks pass. Player-team fog is explicitly unavailable in the
current decoder sources; masked spectator pretraining versus qualified human-view
BC is the unresolved observation contract. No simulator or decoder changes.



Datasets `bc_modern_v1/v2` and runs v1-v3 are in `docs/modern/BC.md` sections 8-9 (v3 = class-weighted
casts, gate vs afk Jax: CS 37, deaths 2.5; its episode video is
`/mnt/nfs/shared/modern-bc/videos/garen-bc-v3-vs-afk-jax-seed1.mp4`). Running on 2026-10-18: the Jax-top
replay export (`/mnt/nfs/datasets/rofl_bc_2619_jax_top_native_20261018`, Slurm array 7969 + re-run 8126,
decoder release v7; pilot conversion of two games passed, job 8134) for a Jax clone and the Garen-vs-Jax video.

Multi-champion work (`docs/modern/MULTICHAMP.md`, approved 2026-10-18): the champion table
(`modern/data/26.19/champion_table.json`, 173 champions from the decoder catalogs
`/mnt/nfs/datasets/rofl_champion_catalogs_2619_20261018` + Data Dragon), observation profile
`modern-world-v2` (identity embeddings + kit descriptors; default stays v1), the converter/trainer/gate plumbing
and the footer survey of the 16.19 archive (`/mnt/nfs/datasets/rofl_top_survey_2619_20261018`: 7,658 build
games, 15,316 top-lane champion-games, 157 champions) are in (fb3ddcf7). The regression passed (two seeds: CS 46/45 vs 37/38, deaths 2.9/3.1 vs 2.5/2.3, lane
distance 1110/1160 vs 1680/1620) and `modern-world-v2` is the default observation profile; checkpoints
without a profile field are read as v1. The Jax export is complete (379 games) and its v2 dataset
(`/mnt/nfs/shared/bc_modern_jax_v2obs`, 367 games) is training the Jax clone (job 8444). The bulk top-lane
export (`rofl_bc_2619_top_all_native_20261018`, 7,455 games, all eight chunks 8299/8307/8309/9451/9452/9454/9455/9456 queued, 2,074 done at 08:15 UTC Oct 19; chunk 8309 waiting on desktop memory (other threads' jobs hold 26 of 29 GB), rerun 10666 for rows 2035-2040; rows 7000-7043 done on danilogin (array 10673, cancelled after that: ~10 min/game); rows 6000-6999 and 7044-7454 on the Mac via puller job 12008, 6 at a time (00:00-09:00 PT window, lifted by Dani until 23:24 PDT Oct 19; see batch README). From ~16:10 PDT Oct 19 new tasks on both hosts use the opt-in v9 native runtime (tag 2619-decoder-throughput-v9, per-host builds; pilots identical NPZ/index, 14-17% less CPU). Registry lost-update race from danilogin-side writes found and fixed Oct 19: 1 game re-exported, 30 games keep consistent token ids but an overwritten history entry (batch README). Decoder switched 2026-10-19 after ~2,470 games, at Dani's request, to the decoder thread's verified commit 75e3e4e (compiled event_value_codec + direct update handoff, no release tag, pinned in the batch as decoder-snapshot-direct): pilot of 8 fresh games + 2 re-exports passed (gameplay values and converter arrays identical to v7, 35% faster, 28% less RAM); remaining desktop chunks 11147/11155/11156/11157 run on it with 2.5 GB per task)
is running on the desktop.

**2026-10-19 22:10 PDT (Dani: a BC model by the morning).** Bulk export at 4,994 of 7,455 games. Desktop rows
<5000 moved to a two-rows-per-task array 12329 (Slurm gives each task a whole core; one decode uses one
thread); rows 5000-5999 went to the Mac pool (puller 12333). First all-top-lane clone, `modern-bc-v5-top13`:
dataset `/mnt/nfs/shared/bc_modern_top13_v2obs` = both top laners of every finished bulk game for the 13
top-lane sim kits (4,219 champion-games available; `dataset plan --champions`, fe5f30e4), plan 12353,
convert 12354 (96 tasks, 4 at a time at 6.4 GB each, no task starts after it could run past 02:45 PDT, so
a partial dataset is a uniform sample), finalize 12356, smoke train 12355 (3 games, the new `--cache`
memory-mapped loader), train 12357 (v3 recipe, 8 epochs, games memory-mapped from desktop /scratch; the
full set does not fit in host memory). Gate against the afk opponent to follow.

**2026-10-19 23:30 PDT: the 13-kit run is withdrawn** (jobs 12354-12357 cancelled; the convert tasks had
also died of a converter memory leak, fixed in 046f80ee/eb9decee). Dani's target: one BC model on every
champion and every lane role but jungle, trained after the bulk decode finishes; the sim keeps running 1v1
top. b1cbd900 adds it: `convert_game(role=, extra_champions=)`, observation profile `modern-world-v3` (v2 +
role one-hot; RL passes role top), the other eight champions in the ward block of the converted state
(fogged by the world's own vision), decisions near them kept. The traced 1v1 observation is identical
(job 12498) and default conversions are byte-identical (12499). No decoder change: every export already
holds all ten players' actions. Scale: ~59,600 champion-games; conversion throughput being profiled
(12513 CPU / 12514 GPU) before the dataset run.

**2026-10-19 23:55 PDT.** Dataset `/mnt/nfs/shared/bc_modern_allroles_v3obs` (plan 12546): 42,670 laner-games
from 5,334 decoded games, every role but jungle, 172 champions, v3 + extra champions. GPU conversion
12553_0 then 12555 (array 1-5 %1; 8 workers share the card; GPU-vs-CPU arrays differ by float32 rounding
only, <1e-6, fog/labels identical, job 12545). The remaining ~2,000 bulk rows (3000-5999) moved to the Mac
pool; sentinel 11739 held until the Mac finishes. Desktop /scratch/ahriuwu (old world-model latents)
deleted with Dani's OK for the training cache (638 GB free).

**2026-10-20 15:10 PDT.** Bulk decode final (7,429 ok, 7 partial, 10 failed: FiddleSticks; batch README).
All-roles dataset converted: 58,424 laner-games (52,547 train / 5,877 val), 46 FiddleSticks refused (no
ability data). The per-champion check found Q/W/E/R casts dropped as not representable: the decoder masks
the aim point of abilities whose asset cast mode it does not know (Syndra Q, Viktor E, Zeri W/E, Kayle Q,
Varus E, ...: 100% of those spells, ~5% of all ability casts; 240-game audit 12623), and self-targeted or
self-mode casts (Annie E, Karma E, Lux E recast) were dropped too. cf052d85 aims unknown-mode casts at the
received endpoint (wire 0x2c: equal to the resolved point on all 122,857 casts with both) and labels self
casts without a cursor; `dataset relabel` writes `games/<game>.labels.npz` (overrides the label arrays; with
the fixes off it reproduces the old labels exactly). Training on the old labels (12567: open-file limit,
fixed in bb7850ad by one cache file per game; resubmit 12624, loaded 52,547 games in 23 min) was cancelled
to give the relabel the desktop. Ally-targeted casts stay not representable (no ally targeting). Next:
relabel array, finalize, train `modern-bc-v5-allroles`, gate vs afk (Garen, Jax).

**2026-10-24 clean13032 CANCELLED by UID1000 after3h10m45s.** Slurm records
explicit cancellation, while trainer labels its TERM handler `preempted`; this
does not establish scheduler pre-emption. No automatic restart. Cache build
completed3250s/503.8GB. Training reached6453 steps and saved NFS latest plus
`ckpt_000006453.msgpack`; sampled best5500 preserved separately (score7.5057,
button accuracy91.01%, click distance13.22). Cast recall remains weak; no
standalone-quality claim. Last evaluation6000 rejected Q regression despite
improved aggregate7.480. Run `modern-bc-v7-clean-13032-20261024-072516-?` under
`/mnt/nfs/checkpoints/lanerl-jax/modern-bc`. Scratch cleanup not independently
verified (desktop SSH check pending); sole checkpoints are durable on NFS.

**Resume authorized by Dani.** Prepare6453-step continuation with original
3-epoch ceiling and unchanged recipe. Resume now optionally restores donor
monitoring: same validation sample, step2000 cast reference, best5500 checkpoint
and stale1 at6000, instead of resetting them. Add recovery test plus real
resume/forced-stop smoke before releasing the full job. No new conversion/RL.

**Resume13034 queued afterok:13033**, source2568861a, user-authorized. Smoke13033
runs full BC suite and actual2→4-step resume: preserve best0 bytes, reference2,
stale1 and stop at inherited patience2. Full continuation restores checkpoint6453,
best5500, reference2000 and stale1; recipe and3-epoch total ceiling unchanged.
Both registered with this thread. GPU full24GB/4CPUs/16h cap/no-requeue; immutable
source/private deep copies, NFS checkpointing and EXIT cleanup. Desktop reachable
again; no13032 scratch directory remains,684.7GB free. Cache must be rebuilt
(~54min measured) and arrays staged (~46min previously); no replay reconversion.
Rough resumed wall time2–4h if plateau soon,10–12h if reaching remaining budget,
excluding queue/pre-emption. First resumed evaluation expected~2h after launch.

Observed clean progress500→6000: event+click8.364→7.480, click distance17.96→13.31;
accepted best5500 score7.5057. At5500 Q/W/E/R exact recall1.20/0.0108/0.911/0%;
overall91.01% accuracy is not useful cast evidence. Next24h: continue while guarded
validation improves, score best checkpoint's rate/timing/target behavior by
champion/role and replay clips (~1–3h after run). Next24–48h conditional on results:
if spells remain weak, choose one targeted timing/class-imbalance or aiming
experiment with matched baseline, instead of more unfocused frozen-head probes.
Calibration may repair probability rates but is not a substitute for cast intent.
New-patch data scaling follows a defensible clean baseline; no RL or guessed CD.

**2026-10-24 resume smoke13033 COMPLETED/0:**55 tests passed. Actual2→4-step
resume preserved best0 checkpoint bytes, stale1 and cast reference2, then stopped
at inherited patience2. Durable `resume-smoke-result.json` passed. Full13034
released and RUNNING (input staging, no resumed optimizer metrics yet). Startup
Slurm cwd warning uses /tmp fallback because /srv alias is absent on desktop;
both scripts explicitly cd into private extracted code before execution, and
the actual smoke passed. Future submissions should specify --chdir=/mnt/nfs/projects/ahriuwu-modern-bc.

**Predictive failure audit13035 registered**, responding to Dani's request to
work ahead while13034 trains. CPU partition explicitly danilogin2CPUs/6GB/45min,
no GPU contention and no optimizer or simulator execution. Source2568861a plus
`failure-audit-TEMP.py` frozen at submission; private /tmp code/data/checkpoint
copies removed on EXIT, NFS results.48 deterministic champion/role-spread
perspectives (32train16val): valid/event/held and weighted button/click mass,
spell-context primary/anonymous enemy visibility, known mana/rank/CD coverage.
Accepted clean5500 checkpoint:8 natural+8 cast-centered256step training windows
measure trunk gradient norms/cosines for button/click/spell-button/spell-click
native loss contributions. Diagnostic strata only, no heldout quality claim.
Results gate task-balancing/sampling experiment versus input coverage work.

Code confirms no previous-action input, GRU reset each25.6s and missing qualified
own cooldowns; anonymous extra champions lack identity embeddings. Shared
factorized screen head is independent of chosen button. These predict repeat-cast
timing, teamfight specificity and multi-target aiming limits; their actual impact
is unmeasured. Earlier age readout failed and oracle pointer gains failed routing,
so neither is promoted or repeated blindly. Fixed-reference cast NLL guard may
reject aggregate gains; evaluate rate/precision/tolerant recall jointly before
interpreting early stop as architectural evidence. No active-training recipe change.

**Failure audit13035 FAILED before metrics:** packed entity wrapper requires a
time slice, not NumPy tuple indexing. Correct v2 materializes one game's dense
rows via `entities[:]` and asserts shape before indexing; CPU13036 registered
with the same bounded/no-update diagnostic. Original worker retained for
evidence; active13034 recipe/checkpoints unaffected. No audit findings yet.

**CPU13036 COMPLETED/0,109.9s.**48 stratified perspectives:197587 valid train/
103350 validation decisions,4352 QWER events total. Movement/basic-attack button
accounts for97.964%/97.468% of weighted click supervision; QWER only2.036%/2.532%.
Even after5x button weights, movement+noop comprise87.26%/85.35% button mass.
Own cooldowns known at0 sampled spell events; mana known atall4352. Extra anonymous
enemy champions visible at24.27%/18.72% of spells, and are the only visible
enemies at7.71%/6.45%. Paired enemy absent44–45% includes farming/self casts;
not evidence those labels are wrong. Gradient audit:8 natural/8 cast-centered
windows, click-versus-spell-button cosines approximately-0.113..0.076; no
consistent destructive alignment. Natural click norms1.51–2.73x button norms.
Small diagnostic, not causal proof or full-corpus statistics.

**CPU13037 registered:** paired native control versus spell-aim x5 weighting
(~9.4% expected click mass instead of2.0%, global normalization retains mean
click coefficient). Same clean5500 parameters, button weights, natural sampled
windows, fresh Adam3e-5/clip0.5,batch4/window256/burn32;200 updates/arm max.
Selection every50, patience3 including baseline.32train perspectives,16 previously
audited validation perspectives split by replay base into selection/assessment,
4 fixed windows/game. Select lower spell-click CE only with movement CE+0.03,
distance+0.25, event-button NLL+0.01 and supported per-QWER NLL+0.05 gates.
Original loss equivalence asserted before updates. Assessment logs expected cast
counts, exact recall and joint targeting. Existing validation reuse, small subset,
no fresh-test or promotion claim. CPUdanilogin2cores/6GB/90min cap, independent
private /tmp staging and NFS outputs; GPU13034 unchanged, still staging at check.

**CPU13037 COMPLETED/0,2175s (36min).** Both arms select baseline0; all50/100/150
updates fail prespecified event-button NLL+0.01 guard, so no promotion and saved
assessment remains baseline. Selection spell-click CE baseline7.98077: native
50/100/150=7.82395/7.89457/7.87294; weighted=7.74550/7.76365/7.74116. Treatment
consistently better aiming than matched control, but event NLL rises0.013–0.023
(vs native0.012–0.021). QWER conditional NLL improves in both, movement CE within
gates. This is not proof weighting cannot help; small-batch fresh-Adam native
control also fails. Only58 selection/101 assessment spell clicks; R assessment5.
Baseline assessment expected Q315 vs59 human, W103vs27/E110vs27/R19vs5: same
class-weight calibration concern as earlier v6, despite low argmax recall.

**CPU13038 registered, no optimizer:** fixed clean5500 checkpoint,64 existing
assessment perspectives/full valid frames. Measure per-QWER average precision
(exact and0.5s-near-press positives), expected counts and argmax recall, fixed
inverse-button-weight correction, and oracle human-count ranking diagnostic.
Separates ranking/discrimination from biased probabilities before further
training changes. Existing validation reused, no fresh-test claim; human-count
threshold not deployment policy, inverse weights cannot exactly undo held-label
weighting.2CPUs/6GB/45min danilogin, private code/data/checkpoint copies and EXIT
cleanup, NFS outputs. Main13034 unchanged, cache22000 at receipt check.

**CPU13038 COMPLETED/0,675.5s:**64 existing heldout perspectives,358473 valid
decisions at clean5500. Q/W/E/R exact AP3.732/1.845/3.062/0.780% versus class
prevalence0.696/0.312/0.372/0.0787%; signal above prior, weak exact localization.
Within0.5s-near labels AP29.23/16.03/25.09/8.20% versus near prevalence
7.12/3.23/3.80/0.803%. These are frame rankings, not event-match recall or a
deployable policy. At oracle human-count budget exact precision7.69/5.36/6.75/
0.71%. No inference that unmatched frames are necessarily invalid human choices.

Raw expected counts Q/W/E/R14454/5863/7376/1325 versus2496/1120/1334/282human.
Fixed inverse-button-weight correction gives3591/1394/1830/321, but exact AP
3.782/1.815/2.986/0.780% barely changes and argmax cast recall goeszero. Counts
correction is not timing recovery; held-label weights prevent exact analytic
calibration. Casting-context probabilities are elevated around human casts,
but calibration alone cannot recover a strong policy at this checkpoint.

Earlier history readout already had a64-wide nonlinear context+age head, so do
not claim it lacked interactions or blindly rerun the same approach. It used the
old pre-target-repair checkpoint and mismatched all-frame training/event selection;
negative result is limited, but no history improvement is established. Do not
launch another broad retrain or frozen head on this early checkpoint now. Let
main13034 produce later clean parameters; repeat the same fixed ranking/rate/
timing evaluation on the selected checkpoint before deciding whether causal
own-action history requires a full-gradient input ablation. No deployment or
production change promoted. Main13034 still active in cache construction at check.

**2026-10-24 scope correction / expert review:** actual finalized plan uses
30–840s (00:30–14:00) at10Hz, not full matches. Valid labels additionally require
alive and inside the lane corridor, even with extra_champions=True.7309 matches
provide58470 early-game player perspectives;172 champions; match-base split
6574train/735val. Earlier "full" meant full configured8100-step window and
corpus, not full-match or unrestricted-map coverage. Tier is constant0. This
limits standalone full-game claims and requires future data-scope expansion.

Current target-type supervision: received self/no-cursor modes mask aim loss;
received unit targets use one-hot grid cells, other received points Gaussian
sigma1. No explicit skillshot metadata input or targeting-mode head, no
action-conditioned aim; shared independent button/X/Y heads. Actual185-wide
self input has158 never-observed columns masked, leaving27 potentially observed
(including role/IDs/tier). No kit descriptions, own qualified cooldowns or prior
action inputs. Reviewer brief uses exact run manifest and current source, not
PPO defaults. Within-epoch monitor's event CE retains class weights; per-class
NLL diagnostic is unweighted. Burn-in loss mask does not detach BPTT through
the32-step prefix. Model6,243,435 parameters, entity2x4head128/FFN256, context256,
MLP4x1024→512, residual LayerNorm-input GRU512.

**Resume13034 CANCELLED by UID1000 after1h57m50s.** Completed staging/cache,
restored best5500/stale1/reference2000, then validated6500 score7.46719 and
accepted it. E recall1.99%,Q1.10%,W0.194%,R0% at6500. Saved step7000 checkpoint
before next evaluation; no completed7000 metrics. Slurm batch endedSIGKILL,
manifest has no terminal status; do not infer graceful completion or cleanup.
Best6500 and latest7000 checkpoint payloads readable on NFS. No auto-restart.
Scratch cleanup unverified. Target-supervision explanation passes728dark/360light
previews and all3 target-mode checks; published inline. No training changes.

**2026-10-24 DESIGN PIVOT — Lane BC v2.** Dani adopted the supplied redesign;
section15 of `docs/modern/BC.md` now supersedes v6/v7 training directions.
Tag `bc-v7-before-v2-20261024` preserves the pre-pivot commit. Current worktree and
branch retained; do not resume cancelled v7 jobs by default. Verified Ahri-mid
592 perspectives(521train/71val), Garen-top236(210/26); enough for the proposed
dev loop without assuming that more corpus size fixes representation. Corpus B
ledger23879 verified raw files; no decoded/training readiness claim. Static client
metadata is explicitly authorized in v2, with source masks/hashes, not sim defaults.
Critical amendments: match-wide heldout isolation; mass-preserving smoothing;
no assumed fixed Ahri R/no blanket refractory; issued versus confirmed history;
separate calibration; no repeated final-test tuning; real-client adapter unbuilt.
New event/metadata foundation modules and tests are committed as unvalidated
scaffolding, not a trained v2 model. Unrelated AGENTS/CLAUDE edits remain untouched.

**V2 validation13041 registered**, source118edf40, worker e32a0efa90fc. Full BC
suite plus24 deterministic Ahri-mid/Garen-top raw perspectives (6train/6val each):
544 frozen decoder hashes, original observation/lag parity, per-spell target
coverage/origin points/intervals and raw bin collisions. Consistent read-only
SQLite snapshots inventory B and propose whole-match target reservations without
reading performance labels for selection. No bulk decode or training. CPU desktop
2cores/8GB/1h cap; fresh private scratch/code/catalog/data copies,30GiB initial
reserve,20GiB per-game reserve, EXIT cleanup; results on NFS under
`/mnt/nfs/shared/modern-bc/v2-20261024`. This is a sampled source gate, not all13
complete audit requirements. SSH desktop check timed out; Slurm accepted the job,
so no claim about old13034 staging cleanup or this job's runtime progress yet.

**V2 source audit13041 COMPLETED/0:**61 tests and544 frozen decoder hashes pass;
24 role-matched perspectives reproduce original lagged entity/padding observations.
Ahri Q423/E138/R62 all have received points; W430 all self. Garen Q394/W120/E443
all self; R14/14 targets in delayed rows. Raw cast+path bins268/1050Ahri and
378/959Garen are26%/39%, so v2 must preserve sub-frame records and co-movement
rather than silently discard them; these are server paths, not proven clicks.
B inventory matches23879 verified files to metadata;2637 matches containAhri,
2477Garen. Whole-match reservation proposal stored in audit output, no outcomes
used for selection or B decoding. No claim all13audit gates are complete.

Desktop Linux and slurmd are now reachable; stale drained/Windows state cleared
without an agent scheduler change. New v2 implementation in progress: event-only
labels with independent button/aim masks, causal previous-receipt/stable-target
history, visible-caster memory and compact inputs; button-conditioned pointer/
joint-grid policy with co-movement/aux outputs, baseline unweighted losses.
None is wired into a completed v2 training run. Next gate is full tests plus
24-perspective sidecar construction and actual GPU loss/update smoke.


**V2 conversion13042 FAILED after69 tests passed.** First perspective saved;
second encountered a missing delayed player position in cast-adjacent movement
projection. Partial directory has no completed manifest and cannot be loaded.
Fix preserves raw source/order timing and masks unknown/out-of-scope movement aim.
Regression test added; recurrent fixture batch loader implemented. No v2 training
has run. Retry uses a fresh source snapshot and output directory.

**V2 retry13043 registered**, sourceb9dc57e7, CPU2cores/8GB/1h; repeats full
suite and24-perspective conversion after the missing-anchor fix, with fresh
private staging and outputs. Gradient smoke implementation being prepared;
no GPU allocation or training started yet.

**V2 GPU smoke13044 registered afterok:13043**, source6d109f91, gpup1GPU/
20GB/30min. Full-size conditional model, two fixed training windows,200 real
Adam updates; checks finite nonzero gradients, parameter movement and30% loss
reduction, saves checkpoint/results on NFS and cleans private staging. This is
only a memorization/wiring gate; held-out quality and shuffled-label control
remain separate. Event evaluation now being built while dependency runs.


Stateful replay evaluator implemented with shared policy encode/target methods:
actual-button versus teacher-button target readouts remain explicitly separate;
full-perspective carry with128-step chunks, no future labels fed into state.
Event matching is restricted to contiguous valid segments. Exact/near-frame AP
is named as a frame diagnostic, not event AP. Six focused CPU tests passed4.88s
for missing-anchor handling, file identity, padding/burn-in/incomplete manifests,
AP ties and matching across invalid gaps. Real model evaluation remains unrun.
Await registered13043 conversion and afterok13044 gradient receipts; no broader
v2 training or quality result yet. Unrelated AGENTS/CLAUDE edits untouched.

**V2 conversion13043 COMPLETED/0:**70 tests passed33.61s; all24 source-audited
perspectives converted in119.44s,158037 valid frames, completed manifest on NFS
at `v2-20261024/dataset-smoke-13043`. Missing-anchor regression passes. Dependent
GPU13044 is registered; await its actual result. Target-only sharded converter
implemented, reserving match-wide A-training calibration before full828-example
Ahri-mid/Garen-top conversion. B remains untouched.


**Target conversion13045 registered**, source5956656a,4shards/max2 concurrent,
CPU2cores/8GB/90min each; finalizer13047 afterok entire array.828 perspectives:
Ahri470train/51calibration/71val; Garen185/25/26. Frozen plan5f975d01bdcc,
match-wide deterministic calibration assignment from A train only. Full latest
BC suite in shard0; raw decoder hashes/parity/shape/target masks per perspective;
source/code/catalog/reference deep copies into private staging and20GiB reserve.
Complete manifest exposed only after every shard passes. ETA35–45min conversion
from119s/24 sample, excluding contention/queue. No B decoding, no new supervision.
Bounded v2 learner implemented with exact global-step sampling/resume budget,
checkpoint-on-signal and raw stateful validation; still unrun, no quality claim.


**V2 gradient13044 COMPLETED/0:**6,903,134 parameters,200 actual GPU updates,
177.87s. Loss4.5477→1.5281, aim CE7.7206→2.8845, finite nonzero gradients,
parameter delta5.1869. Button accuracy67.89%, NLL0.6548: passes basic wiring
but does not yet establish strict memorization. Added strict98%/0.05 button gate
for longer bounded fixed-window check. No held-out policy quality claim.
**Learner/resume/eval smoke13049 registered**, source6f114254,32 total steps,
resume at16 and compare final weights to uninterrupted control, then one full
held-out fixture replay through stateful evaluator. GPU20GB/30min/private staging.
Target conversion13045 and finalizer13047 remain registered; await callbacks.

**Strict fixed-window13050 registered**, sourceb8b8f957,1000 updates on the
same two fixture windows, button accuracy>98% and NLL<0.05 required in addition
to the prior finite-gradient/loss-drop gate. This tests the remaining strict
memorization requirement rather than reinterpreting200-step wiring as success.
Per-button target metrics now include joint success over all eligible targets,
so missed buttons count as failures; correct-button conditional accuracy remains
separate. Three focused metric tests passed, including this denominator case.
Active registered work: target conversion13045/finalizer13047, learner/resume/
eval smoke13049 and strict memorization13050. Await receipts; no broad Ahri
training or accepted quality result yet. Existing unrelated edits untouched.

**Target conversion13045 COMPLETED/0 all4shards; finalizer13047 COMPLETED/0.**
828 NPZ/metadata/raw-event sets,5420481 valid frames; shard durations1188–1205s.
Completed versioned manifest at `v2-20261024/target-events-5956656a`; compressed
NPZ total0.593GB. Full74-test suite passed in shard0. No source/lag parity or
shape/mask failures. Ahri3778050 valid frames, Garen1642431; all prior split
counts preserved. Derived summary `v2-20261024/target-summary-13045.json`.
Ahri Q21801 events/18573 valid aim/3219 clipped; E6899/5832/1059;
R3220/2726/469. Raw directions retained for later angle ablation, no reconversion
needed; off-screen point labels currently mask aim only. Attack pointers
Ahri84553/85455, Garen23084/23767; Garen R296/305. Stop has zero received labels,
so no learned-stop claim. Not all13source-audit diagnostics are complete.
At receipt handling,13049 RUNNING52s and13050 PENDING; await registered callbacks.
No broad Ahri baseline launched before learner/strict gates. B untouched.

**Learner smoke13049 FAILED at final comparison**, not during training/eval.
Uninterrupted32 steps and resumed16→32 both completed; one full held-out replay
scored6791 valid frames. Max parameter difference7.426e-5 exceeded1e-5 assertion;
max logged loss difference7.564e-5, first resumed loss differs9.54e-7. Cause not
established; do not call exact resume validated or relax solely to pass. Added
GPU probe: byte-exact checkpoint roundtrip, optimizer count16, two continuations
from the same checkpoint, relative/global and end-loss comparisons with explicit
numerical bounds. No broad baseline started. Strict13050 remains registered.

**Resume numerical probe13055 registered**, source5fd69a8f, GPU20GB/20min.
Stages immutable13049 donor checkpoint/results and24-perspective fixture into
private scratch. Checks exact parameter/optimizer serialization roundtrip,
optimizer step16, two separate16→32 continuations, and original control/resume
comparisons. Declared bounds maxabs1e-3/relativeL2 1e-4/end-loss spread1e-3;
results remain diagnostic, not bitwise-determinism proof. Await13055 and strict
memorization13050 callbacks. Dataset828 remains finalized; broader training gated.

**Strict13050 FAILED before a completed model result:**Slurm Restarts=1;
restarted script encountered existing empty NFS output directory and refused it.
Original attempt's stdout was overwritten, so no inference about its loss curve
or the original restart cause. No strict-memorization verdict exists.
**Strict retry13063 registered** with identical b8b8f957 code/1000-step gate,
fresh job-private staging/output, explicit no-requeue and append logs. No model
changes or larger budget.13055 was RUNNING first attempt(Restarts=0); disabled
requeue for this owned diagnostic as well. No polling loop; await both receipts.
BC.md records explicit short-job restart policy; broader training remains gated.

**Resume probe13055 COMPLETED/0:**checkpoint parameter/optimizer roundtrip
exact, both optimizer counters16. Two same-process continuation replicas agree
bitwise. Original control/resume versus probe: maxabs<=1.1275e-4, relativeL2
<=2.799e-5; final-loss spread7.153e-6. Accept bounded numerical resume fidelity
for this fixture, not cross-process bitwise determinism or a proved kernel cause.
No learner logic/loss/tolerance changed to hide a checkpoint mismatch. Original
13049 final strict assertion remains failed evidence;13055 supplies the separate
numerical comparison. Strict13063 remains the gate before broad Ahri learning.

**Batch benchmark13067 registered afterok:13063**, sourcef2891e0a, exclusive
GPU node/24GB/45min, no-requeue/append logs. Separate16-update processes for
batch8/32/64,window256,burn32 on470 train Ahri perspectives; no validation or
quality-selection claim. Fresh deep-copied target dataset and code; durable
checkpoints/throughput on NFS.13066 was intentionally cancelled before execution
and replaced to include required node exclusivity; do not auto-resubmit13066.
Await strict13063 and benchmark13067 receipts before selecting baseline budget.

**Strict13063 FAILED its declared button-NLL gate after1000 completed updates.**
Final accuracy99.08% exceeds98%, NLL0.06303 misses0.05; total loss4.5477→0.8184,
aim CE2.4951 (soft-target entropy is nonzero). Late constant-LR losses oscillate;
no NaN, checkpoint or runtime failure. Add bounded1000→1200 continuation of the
same saved params/moments/windows at LR3e-5 rather than3e-4; strict accuracy/NLL
gates unchanged. This tests optimizer settling, not a larger model or generalization.
Benchmark13067 dependency cannot pass after13063 failure; cancel/rebind to this
new gate. No broad baseline started, no strict-success claim.

**Strict polish13069 registered**, source19e43e03, GPU20GB/20min, saved13063
params and optimizer state, same fixed windows, total ceiling1200 (200 extra),
LR3e-5. Source/data/donor copied privately with combined-size+20GiB reserve;
no-requeue/append logs. Benchmark13067 intentionally cancelled because its
13063 afterok dependency failed; replacement registered afterok13069. Await
terminal receipts rather than polling; no broad Ahri training yet.
Replacement exclusive batch benchmark is13070, same f2891e0a source and16-update
batch8/32/64 recipe; no quality comparison or extra dev-training budget implied.


**Strict polish13069 COMPLETED/0:**restored1000→1200,LR3e-5,161.97s including
compile. Fixed109 supervised frames reach100% button accuracy,NLL0.002799;
total loss0.81843→0.74956,unit loss contribution5.36e-8. Strict98%/<0.05 gate
passes without changed thresholds. Aim CE2.46658 includes Gaussian-target entropy,
so nonzero total loss does not imply failed memorization. Same two windows only;
no held-out quality/leakage-control claim. Benchmark13070 now eligible afterok;
await measured batch8/32/64 throughput before selecting first broad Ahri budget.
No additional jobs submitted on this receipt; no broad Ahri training launched.

**Exclusive batch benchmark13070 COMPLETED/0:**batch8/32/64 all fit and train
16 updates on470 Ahri train perspectives. Median update0.1822/0.2728/0.3740s,
11.24k/30.03k/43.80k frames/s; batch64 selected. Each process39–57s including
load/compile; throughput only, no quality comparison. Eligible13992 train windows.
**Ahri baseline13074 registered**, sourcef2891e0a,1000 total updates,batch64,
window256/burn32,seed0,Adam3e-4 cosine→3e-5,100-step warmup,clip0.5. Natural
unweighted event+aim losses, history on, conditional targets, no co-move/auxloss.
470train perspectives sampled with replacement;64000 window draws (~4.57 times
eligible-window count), not four exact epochs. Save250/500/750/1000; full71val
stateful raw evaluation after training.51calibration perspectives untouched.
GPU24GB/1h/no-requeue/append/private immutable data+code copies. Expected6–8min
training from benchmark, evaluation additional. No model-quality result yet.
Matched unconditional control registered afterok13074: same source/seed/batches/
budget/target parameterization, only conditional_aim=False. No automatic promotion.
Matched unconditional control job ID13075, afterok13074. The first comparison is
identical conditional-vs-unconditional target parameterization; no simultaneous
angle/head/history changes. Report aggregator implemented for summed event counts,
per-button joint/conditional target success and match-bootstrap paired intervals;
macro frame AP and mean per-match median point errors explicitly named, not pooled
AP or pooled medians. Aggregator syntax checked; real broad validation unrun.

**Ahri baseline13074 COMPLETED/0**,1000 updates/661.46s training,6.903Mparams,
b64 median0.3818s/update. Full71val/451441valid frames: greedy emits0attacks and
0QWER/recalls; expected QWER counts0.942/1.047/0.955/0.926x human. Exact-frame
macro AP Q/W/E/R0.06355/0.03521/0.02073/0.02579. Teacher Q/E within2cells only
1.57/1.41%,mean per-match median25.43/24.69cells; attack pointer68.68%.
Movement teacher within2 9.93%,mean per-match median10.89cells. Not useful policy
quality and no promotion. Raw greedy diagnostic is not intended sampled-emitter
quality.13075 matched unconditional remains registered; await its result.
Implemented isolated Q/E angle-label and3x cast-aim ablation flags, preserving
original aim-valid support for parameterization comparison (no clipped-label
recovery yet). Added sampled emitter with RNG carried across chunks and targets
conditioned on sampled buttons, plus checkpoint assessor/prediction export.
Tests added for weighting isolation, label/input preservation and sampled-target/
chunk invariance. Not Slurm validated yet; no new ablation training launched.
Grid-target evaluation now also converts predicted cells through the shared static
camera projection to direction error against saved replay angles. This supplies
angular metrics for a fair grid-versus-angle comparison; no simulator rollout or
new state source. Direction support is reported explicitly; inputs unchanged.

**Ablation/emitter validation13076 registered**, source73fe070d, CPU2/8GB/15min,
full suite including new label/loss isolation and sampled-target chunk/RNG tests.
Three GPU jobs gated afterok13076, same immutable source/private staging:
**13077** re-assesses frozen baseline greedy plus categorical T1/seed0 on71val,
exports prediction summaries and direction metrics, no fitting or retraining.
**13078** Ahri Q/E angle-head ablation, original valid-label support unchanged.
**13079** original grid with3x cast-aim loss only, button loss unchanged.
Both learning arms match baseline1000updates/b64/window256/burn32/seed0/schedule;
no angle+weight combination yet.51calibration and B untouched. No promotion.
All no-requeue/append, logs/checkpoints NFS; runtime training measured11min for
first baseline including startup, plus evaluation/queue. Paired13075 remains
registered, no result inspected yet. Baseline summary saved to
`v2-20261024/baseline-summary-13074.json`; await registered callbacks.

**Matched unconditional13075 COMPLETED/0**,same71val match IDs as13074.
Conditional versus unconditional teacher Q within2 1.570%vs1.265%, E1.410%vs1.128%,
movement9.932%vs9.449%; attack pointer68.678%vs68.956%. Macro exact-frame Q AP
6.355%vs6.583%, E2.073%vs2.189%; greedy QWER zero in both. These are small,
mixed changes and neither policy is useful; no conditional-head quality promotion.
Summary `v2-20261024/conditioning-summary-13075.json`.
**Ablation/emitter tests13076 COMPLETED/0:**78 passed35.66s, including isolated
cast-aim weighting, angle-label/input support, sampled target follows emitted
button, and PRNG continuity across chunks.13077/13078/13079 dependencies cleared.
**CPU paired report13080 registered**, source73fe070d,500 match-bootstrap draws,
conditional-minus-unconditional intervals on same71matches, private snapshots/
NFS output. This assesses uncertainty, not a new training run. Await registered
13077 emitter,13078 angle,13079 cast-weight and13080 reports; calibration/B unused.
