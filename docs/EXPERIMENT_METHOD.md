# How experiments are chosen and run (2026-09-23)

Written after the architecture review turned into a list of candidate changes.
The scarce resource is **trustworthy experiments**, not GPU-hours: an idea is
worth nothing if we cannot implement it correctly, afford a meaningful
comparison, or interpret the result. This project has already paid for all
three failure modes: a Dreamer-4 replication whose compute estimate was off
by months; a 46-CS headline that was a simulator exploit (`SPELL-001`); a
sweep whose arms ran different code because the tree changed under them.

**Order of work: harden the baseline first** (below, "Hardening gate"). No
research experiment starts until the baseline passes it.

## 1. Every experiment gets a contract, written before any code

> suspected bottleneck -> existing evidence -> implementation check ->
> measured cost -> success criterion -> stopping criterion

A **stop** means "not worth pursuing under this budget", not "the idea is
false". **Inconclusive** is a legitimate result; one disappointing run is
screening evidence, not a verdict.

## 2. Three questions before committing to anything

| Question | Evidence required |
|---|---|
| Does it address OUR problem? | a diagnosed failure, prior work, and a mechanism for why it would help |
| Can we implement a VALID test? | reference code, independently checkable numbers, a small test of the mechanism itself |
| Can we AFFORD to answer it? | measured memory and runtime at the intended size, INCLUDING the number of comparison runs, not "one run fits" |

## 3. Guarding against AI-written experiments being wrong

`Implementation Matters` (Engstrom et al. 2020) found implementation details
explained most of PPO's advantage over TRPO. "A lost to B" often means "our A
was worse".

- **Start from a reference implementation**, pin its version, change one thing.
  Never rewrite the learner, the architecture and the action space at once.
- **Numerical checks, not a second model's approval.** For this PPO stack:
  rollout-time log-probs equal recomputed ones for unchanged weights and
  identical history/masks; GAE matches hand-worked trajectories including
  resets and truncations; the masked conditional log-prob and its gradient
  match an independent small implementation.
- **Validate the mechanism before the League number.** A memory core must
  pass a controlled history-dependent task first; a new loss must match its
  reference computation.
- **Independent training runs, not repeated evaluation of one checkpoint.**
  Seed spread here is as large as the arm gaps at 300 updates (sweep B a0:
  23.3 CS seed 0, 8.4 seed 1), so short-run arm tables are screens, not
  rankings.

## 4. Avoid reinventing the wheel without trusting papers blindly

| Evidence status | Action |
|---|---|
| Strong, closely matching setting | adopt the implementation, verify the integration, do not re-run the field's debate |
| Evidence elsewhere, important mismatch here | test THAT mismatch specifically (action structure, recurrence, compute scale, demonstrations) |
| Plausible, weakly supported | one bounded exploratory project with a feasibility test and a stopping rule |

A review must report isolated vs bundled gains, dependencies (did the
transformer also need a new loss, longer sequences, pretraining, more
tuning?), real training cost, and negative results. See
`docs/ARCH_LIT_REVIEW.md`.

## 5. Where the interesting research should come from

Find where the current model is **inefficient**, using diagnostic
interventions that make one thing easier and look for headroom:

| Suspected inefficiency | Cheap diagnostic | Research a positive result justifies |
|---|---|---|
| Useful situations are rare | train on a mix emphasising failed situations; evaluate held-out and on normal lanes | curricula, scenario selection, Go-Explore-style restarts |
| History is hard to use | add an explicit summary of legitimately observed events; matched retrain | event memory, GRU, temporal transformer, hybrids |
| Learning targets are weak/noisy | on a small state set, estimate consequences by repeated simulator continuations; can the network learn them? | better critics, counterfactual supervision, planning teachers |

Candidate tracks from the review (not yet contracted): the learning engine
(PPO vs PQN-style parallel Q-learning, critic loss as classification per
"Stop Regressing", SimBa-style normalisation/residuals); a fair temporal
transformer trial including its real rollout-memory cost (GTrXL, AMAGO,
Actor-Learner Distillation for a transformer learner with an LSTM actor);
and ONE ambitious track -- either reusable skill/mechanics learning (masked
entity prediction, ACT-style action chunks) or simulator-generated teacher
supervision -- not both.

## 6. Measured cost, this machine (RTX 5080, 16 GB), 2026-09-23

From sweep B's manifests (256 envs x 128 steps, 4 minibatches, 4 epochs,
feedforward policy, route table on):

| Unit | Wall | Notes |
|---|---|---|
| 1 PPO update | ~3.3 s | ~9.5-10.2k env-decisions/s incl. compile |
| 300-update screen arm | ~17 min | ~2 full episodes per env; a screen, not a ranking |
| 1 full 10-min episode per env | ~7.8 min | 141 updates |
| 2,000 updates | ~1.8 h | first point where CS is readable |
| 8,000 updates | ~7.3 h | fits the 8 h slurm window |

**Not yet measured, must be before committing:** a GRU core with BPTT over
the 128-step rollout (activation memory scales with T x batch); a temporal
transformer (KV/sequence buffers); a KL reference policy (a second forward
pass); an autoregressive decoder. Benchmark each at the intended batch,
sequence length and precision, forward+backward+optimizer, compile timed
separately, `block_until_ready()` around every timing.

## Hardening gate (before any research experiment)

1. Sim suite green per file (`ops/sim_tests_per_file.sh`), with `TEST-001`
   resolved, not waived.
2. Every config field that affects a run is READ by the code and RECORDED in
   the manifest (the `RL-004` class); optional machinery not needed for the
   baseline (league/PFSP, unported reward terms, frame stack, alpha anneal) is
   removed, not left declared.
3. The eval path reconstructs what the sim trains on (`OBS-04/05/06`).
4. The policy-divergence gate has a measured shuffle floor, so its verdict is
   scored (`PARITY-001`).
5. A bottleneck-and-feasibility audit of a real baseline run: learning curves,
   failure cases, measured memory/runtime, the list of correctness checks and
   what each one covers.
6. Only then: experiments as git snapshots (worktree at a commit, runs written
   to the live tree's `runs/`), one contract each.

## Recording an experiment without creating document sprawl

The 2026-09-24 conversation reaffirmed the method above. Its architecture
ideas are candidates, not accepted implementations. Current scope and
hardening priorities are in [PROJECT.md](PROJECT.md); gate verdicts remain
in the fidelity ledger. In particular, simulator/server agreement is evidence
about practical control and transfer, not a demand to perfect every tick
before investigating a demonstrated farming failure.

Before starting research, add one compact contract under this section. Link
existing evidence and manifests rather than repeating their measurements:

```text
ID / question / owner / date / status:
Suspected bottleneck and existing evidence:
Intervention, reference implementation/version, fixed baseline:
Independent implementation check:
Measured feasibility and total comparison budget:
Evaluation: opponents, side swaps, seeds, holdouts, metrics, selection rule:
Success / stopping criteria:
Run IDs and evidence links:
Decision, reason, limitations, next check:
```

No new research contract is approved by the presence of this template. A
correctness repair can proceed as a diagnosis plus regression; do not label
it an architecture experiment or compare its scores as if mechanics were fixed.

Each run lives under a named ignored `lanerl_jax/runs/` directory. Use the
existing `lanerl_jax.train.run_manifest` machinery for training. For diagnostics,
record equivalent provenance in the sidecar/run README: exact command and cwd,
resolved settings, seeds, source commit, actual dirty patch and relevant
untracked source if present, input/checkpoint/route/vendor versions or hashes,
software and hardware, elapsed time, exit status, and artifact paths. A dirty
flag or hash alone does not preserve the source. Evidence cited by a decision
must have versioned reproduction code or a retained source snapshot; a script
left only in `/tmp` is insufficient.

Distinguish a new run, crash recovery, and an exact continuation. Identify
whether evaluation uses historical code or current code with old weights.
Record both sides' farming/combat outcomes and failure cases so a combined
score cannot hide one side stalling. Keep unsuccessful and inconclusive runs
linked alongside successful ones. At completion, update the contract's decision
and the appropriate ledger row; preserve detailed run output as evidence.


## SERVER-FIRST-01 — uncontested farming, 2026-09-25

Status: ongoing until consistent farming improvement. Dani selected this
stopping point on 2026-09-25, rather than stopping after diagnosis or after
the JAX comparison. Working operational target proposed in the conversation:
three independently trained seeds, five frozen evaluation episodes per seed,
each seed's trained median at least 30 CS by game time 600 s and above its
paired untrained median. Record all episodes, deaths and attack engagement;
replays must show repeated minion attacks/last-hits. No hidden-input/control
violations or cast-freeze failures may count as successes. The near-wave task
is an intermediate test; report its 120-second setup explicitly and confirm
the final result from normal fountain spawn. This threshold is a first
functional farming milestone, not lane strength or D2-level performance.

Dani explicitly requested this
baseline to separate learner failures from JAX dynamics. This supersedes the
requirement to finish simulator parity before this server-only investigation.

- Question: can randomly initialized PPO learn useful farming on the source
  server, then under the same learner and task on JAX?
- No pretrained checkpoint, demonstrations, imitation loss or reference-policy
  prior. Structured observations are intentional: pixel interpretation is
  deferred. Actor entities must be alive, visible by the environment's own fog
  rules (including brush), and inside the local viewport. Actor outputs remain
  button plus screen coordinates. Diagnostic traces may contain hidden state.
- First task: ten-minute top lane with the enemy champion idle in its fountain;
  both teams' minions and turrets remain active. This is uncontested farming,
  not a matchup result. Start with blue, then evaluate both sides. Keep spawn,
  skill progression, rewards, decision rate and episode boundaries explicit.
- Reuse the same policy, PPO loss/optimizer, observation layout and reward code
  across the source-server and JAX collectors. JAX as the learner library is
  allowed; source-server rollouts must never call the JAX dynamics step.
- Verify screen/fog exclusion, sampled/recomputed likelihood agreement, finite
  updates, real parameter changes, episode reset homogeneity and Q retarget
  recovery before spending a training budget. Measure collector throughput and
  memory before choosing the budget; do not substitute game counts for equal
  decision budgets.
- Evaluate frozen initial and final policies with the same sampling protocol;
  include an lr=0 control and independent seeds before claiming learning.
  Report CS, deaths, wave proximity, attack completion and invalid/frozen runs
  per side. Save minimap and combat replays from the actual evaluated episodes.
- Stop on nonfinite optimization, frozen champions, hidden-information leaks,
  dropped entities or heterogeneous resets. Flat CS alone is an inconclusive
  learning result requiring trajectory inspection, not permission to tune many
  things at once.
- JAX comparison and cross-play follow a credible server baseline and matching
  visibility rules. Modern-client fidelity remains a separate later question.

Initial feasibility: two servers on desktop CPU, 128-step rollouts, four PPO
updates: 1,024 decisions in 15.33 s including compilation; later updates took
1.07–1.41 s per 256 decisions. Reset smoke: two finite updates and fresh-process
resets with identical initial HUD stats. The first screen budget is 512 updates
x 2 servers x 128 steps = 131,072 decisions, seed 0, lr=critic_lr=1e-5.
Reward is explicitly +1/CS, -2/death, plus 5 times discounted lane-approach
potential (zero terminal potential); no damage, XP or ambient-gold reward.
This replaces the old mirror reward for BOTH planned comparison collectors.
Skill ranking uses a fixed environment progression and costs extra server
ticks; those ticks must be accounted for in the subsequent matched JAX task.
The first screen and its initial/final frozen evaluations completed with 0 CS.
The final policy reached lane but never came within 585.67 units of an enemy
minion. See the fidelity ledger for evidence and limitations; SERVER-FIRST-02
tests whether removing the trip from fountain changes minion engagement.


### SERVER-FIRST-02 — fixed near-wave start

The first source-only screen completed 131,072 decisions with finite updates
and changed parameters, but no CS. Its frozen policy reached top lane and
level 6, while never approaching an enemy minion closer than 585.67 units.
The untrained control stayed near the fountain. This motivates separating
navigation/exploration from last-hitting rather than changing the learner.

Only the reset setup changes: a non-learning controller walks to (1950,12350)
and hands control over at 120 game seconds, just behind the first blue wave.
This point is based on the untouched source control's first minion damage at
124.909 s, blue (2157,12474), red (2259,12573). Setup actions are excluded from
PPO batches; weights are freshly random, with no demonstrations or prior loss.
The first attempted setup point (3600,13100) was inside enemy turret range;
its guard rejected the run before learning. It is not an RL result.

Episodes still end at game time 600 s; the policy controls approximately the
last 480 seconds. Report this as a near-wave scenario, not full-match farming.
Deaths still respawn normally; only episode boundaries repeat the setup.
Require reached position, alive champion and zero setup CS. Keep the same
policy/PPO/reward settings as SERVER-FIRST-01. Initial budget: 512 updates x
2 envs x 128 steps per seed, fresh seeds 0 and 1. Score frozen initial/final
policies on the same setup. A positive result requires improved CS and actual
minion attack engagement; finite optimization alone does not count. Preserve
zero/failing results and inspect trajectories before changing another factor.

### SERVER-FIRST-03 — actual attack-move controls

The v1 replay diagnosis found no observed autoattacks, with targets acquired
only briefly. Inspecting the source handler showed that the exposed A-click
was converted to plain movement on empty ground, and right-click ignored
hostile cursor hits. Correct the control interface, retaining buttons and
screen coordinates only. Fresh seed 0/1 runs keep SERVER-FIRST-02's learner,
reward, near-wave setup, 30 Hz and 131,072-decision budget. This is a control
repair comparison, not proof of improved optimization. Desktop disconnection
interrupted observation of these runs; do not treat partial checkpoints as
completed budget results.

The independent audit found no PPO algebra defect, but highlighted absent own
attack-animation/history inputs and sparse reward exposure. Before choosing
another training intervention, compare frozen v2 policies at 30 versus 10 Hz
for fixed 600-second episodes and paired sampling seeds. This diagnoses
execution sensitivity; changing frequency at a fixed decision budget also
changes wave exposure. Any subsequent training arm must report both decisions
and simulated game time and retain a matched-game-time checkpoint. The source
collector now accepts `--step-ticks`; gamma and lambda adapt to preserve their
physical time horizons. A 12-tick fresh-process reset smoke passed two finite
updates, but is not a learning result.

### SERVER-FIRST-04 — corrected HUD baseline and longer exposure

The source HUD repair prevents repeated disabled Q presses from refreshing
empowerment. Frozen random seed 0 now scores 14 CS at 30 Hz and 11 at 10 Hz
in one near-wave episode each; this is control evidence, not learning. Keep
30 Hz, the existing reward and architecture, and train fresh seed 0 for
2,048 updates (2 environments x 128 decisions = 524,288 decisions). Preserve
the 131,072-decision checkpoint for the earlier budget comparison. This is
a baseline feasibility run; if frozen evaluation improves, repeat independent
training seeds and the five-evaluation protocol above before claiming success.
Do not infer a frequency winner from one paired episode. The separate
HudProbe binary and viewport-structured-v2 are required. The frozen checkout
is `ahriuwu-server-hudtrain-20260925`; capped login execution is used while
desktop SSH remains unavailable. Stop this run on invalid controls, nonfinite
updates, or its stated budget; the overall farming objective remains active.

Stopped at update 447: the dead-input investigation confirmed E can start
while dead in this server interface, and HP regeneration makes the adapter's
HP-based alive inference incorrect. The 10 Hz control includes four CS gained
from a dead-started spin. Preserve these trajectories as defect evidence;
they cannot establish legitimate farming. See the fidelity ledger.

### SERVER-FIRST-05 — authoritative dead-state baseline

Repair the concrete invalid-control mechanism before restarting: expose the
server's authoritative champion dead flag, reject live-only keyboard/mouse
commands while dead, and use that flag for actor state and death reward.
Require a live positive-HP corpse regression, no new E cast from dead input,
normal control after respawn, and one death penalty per death. Version the
semantic observation contract so historical v2 checkpoints cannot silently
load as corrected policies. Do not bundle new E-active features or reward
changes into this repair.

After those checks and a finite update/reset smoke pass, restart fresh random
seed 0 with SERVER-FIRST-04's settings and budgets: 2 envs, 128-step rollouts,
30 Hz, 2,048 updates with a preserved 512-update checkpoint, same near-wave
task and PPO/reward. Preserve initial weights and compare frozen evaluations
using the same corrected interface. Measured previous login cost was roughly
12–14 s/update at two CPUs; the full budget is therefore several hours, not
the old desktop timing. Stop on invalid controls, nonfinite updates or budget.
This remains an intermediate gate; the three-seed/five-evaluation criterion
and normal-fountain confirmation remain required for the overall goal.

Use the otherwise free evaluation CPU for one frozen update-120 diagnostic
(30,720 decisions, approximately the first completed training episodes).
Preserve that ordinary checkpoint before rotation. Its seed-0 episode checks
learned control execution early; it does not replace the update-512/final
comparisons or justify a hyperparameter change from one score.

Run a bounded live zero-learning-rate control through the same frozen v3
collector and shared learner: seed 0, two environments, 128-step rollouts,
two updates, near-wave setup, 600-second episode horizon and 30 Hz. Require
finite updates and byte-identical initial/final parameters with both actor
and critic learning rates zero. This checks the disabled optimizer path;
512 decisions do not constitute a ten-minute farming evaluation or an
equal-budget training ablation. The five frozen initial-policy evaluations
remain the behavioral no-learning baseline.

### JAX-FARM-01 — shared learner, independent dynamics

After production-routed setup and reset pass, validate the shared learner
with one environment, four decisions and one update. Require finite losses,
rollout/recomputed likelihood agreement, changed parameters and preserved
source/configuration; this smoke does not measure farming. Then a fresh
seed-0 screen uses the same 2 environments, 128-decision rollout, 30 Hz,
near-wave setup, reward and PPO configuration as SERVER-FIRST-04, initially
512 updates (131,072 decisions). Compare frozen initial/final policies on
the source server as well as JAX. Report disagreement; do not select a JAX
checkpoint solely on its simulator score.

Use `train/jax_train.py`, whose learner loop is shared with source training.
Record the route artifact and simulator fingerprint. Known remaining
differences include landmark routing, rank-event ordering and STAT-003's
missing MR rune bonus (including the actor's own-MR feature). This is a
task-matched diagnostic, not a claim of exact mechanical equivalence. Stop
on setup failures, nonfinite updates or the stated budget; expand seeds
only after evaluating the screen. The source baseline continues independently.

The CPU-map arm stopped at update 88 after finding extra EDT ground-click
normalization in its adapter. The corrected raw-screen adapter passed its
focused and production setup checks. A future arm must also use the corrected
dead-state observation contract from SERVER-FIRST-05. Keep source farming as
the primary gate; do not spend a fresh full JAX training budget while the
source interface is failing a known control invariant.
