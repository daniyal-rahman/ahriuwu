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

**Active: finalization/full array audit13005**, desktop CPU6GB/2cores,2h limit.
Checks source/fog/converter contract, train/val base separation, all expected
array shapes, finite-or-missing numerics and action/cell bounds. Manifest plus
`audit-result.json` persist under `/mnt/nfs/shared/modern-bc/v6-full-20261022`.
Log `modern-bc-v6-audit-13005.out` there. Broad training waits for this audit.

Implementing opt-in within-epoch BC checkpoints and fixed champion/role-spread
held-out evaluations, plateau stopping on event-button+click NLL, and immediate
nonfinite training/gradient stopping. No three-epoch minimum. A short isolated
training smoke will verify the actual checkpoint/evaluation path before full BC.
Native decoder v18 is released as partial player view; complete rendered/terrain
FoW remains unqualified and existing production exports are unchanged. Per Dani,
this no longer blocks the approved heuristic baseline. No training/PPO active yet.

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
