# STATUS — 2026-10-02 overnight AFK diagnostics and continuation

E81→E82→E83 is complete. The best completed full-task endpoint in this sequence is **E82:
14.125 CS, zero deaths, zero personal tower damage**, over64 frozen120s AFK games.
Teaching on full waves improved the earlier10.0CS policy to11.75; subsequent
CS-only PPO improved it to14.125. This is evidence of learning with the existing
observations/GRU/clicks, not perfect CS or an isolated explanation of the plateau.

**Oct2 9:24AM PDT terminal review:** E87/2186 CANCELLED by UID1000, ExitCode0:0,
after7h29m22s. Worker handled the interruption and saved update6982;
114,393,088additional decisions, versus planned8192/134,217,728.
Cancellation reason beyond the recorded UID is not established. No learner
exception/nonfinite failure observed; this was not the scheduled time limit.
Frozen64 CS at updates0/128/512/1024/2048/3072/4096/5120/6144 is
14.125/11.296875/13.28125/14.328125/13.4375/10.640625/12.34375/13.6875/13.4375.
Latest evaluated weights have100.663M additional decisions; saved update6982
has no frozen evaluation. Latest deaths.46875/game, personal towerHP2.55924;
paired CS change−.6875 (27better/8equal/29worse) versus initialization.
No sustained gain observed; best intermediate+0.203125 is not endpoint success.
Planned8192primary/retention and final-three secondary gates were not completed.
Initial exact E82 equality passed; all6982updates have finite core learner
metrics and literal reward=CS, every other logged reward component zero.
Curves: `docs/figures/E87_training_curve.png` and `afk_checkpoint_history.png`;
CSV retains82frozen evaluations across22training arms, including repeated starts.
History separates different setups; saved but unevaluated weights have no score.

**Recommendation revised after Dani's duplication objection:** the proposed
imitation→small-task PPO→full-wave progression substantially repeats E81/E83/
E85/E86. Their limitations alone do not justify another broad sequence. First
separate remaining last-hit misses in the zero-death E82 policy from the loss
of survival during E87 continuation. Existing frozen episode data now locate
the latter: at6144,34surviving games improve paired CS14.05882→15.47059(+1.41176);
30games with death regress14.2→11.13333(−3.06667). All initial E82 games survived.
At3072,23surviving games add6CS total;41games with death lose229CS total.
These are post-outcome groups and associations, not a causal effect of death,
proof of improved last-hit timing, or independent confirmation of a hypothesis.
Next proposed diagnostic: compare good/bad checkpoints on matched full-wave
trajectories, identify death source and preceding decisions, and separate CS
lost before/after danger/death. This directly targets continuation regression.
If risky pursuit of near-term CS is supported, longer discount horizon under
the same CS-only reward is an untested comparison; gamma.99 discounts a30s-later
reward to.049 versus.741at.999. No such run is submitted. This would address
survival/retention, not explain all minions missed by the original E82 policy.
Rising sampled KL and smaller-step comparisons remain hypotheses, not LR proof.
No new training submitted. Do not automatically restart cancelled E87.

**Independent-baseline design discussion:** Dani asks which implementation to
try. Recommend pinned SB3-Contrib RecurrentPPO v2.9.0/PyTorch,
MultiInputLstmPolicy with one256-unit LSTM each for actor/critic and small64x64
heads. Existing v3 numeric observations/masks, original8/96/54 physical actions,
CS-only120s AFK at10Hz; flatten numeric inputs with invalid entity rows zeroed.
Keep SB3's collection/buffer/GAE/recurrent minibatching/loss/optimizer code.
A batched VecEnv adapter handles JAX simulation and exact episode-end semantics.
Our current PPO is already adapted from PureJaxRL31756b, so another port of that
reference would share ancestry. SB3's LSTM and unconditional MultiDiscrete
likelihood differ from our GRU and conditional screen likelihood: this is a
whole-training-setup comparison, not an isolated learner-bug test.
Proposed initial comparison: fresh initialization for both SB3 and existing
JAX recipe,1024updates x128envs x128steps =16,777,216decisions/arm, same frozen
suite, match LR1e-4/4epochs/4minibatches/gamma.99/lambda.95/entropy.001. Replicate
a positive difference on a second training seed before calling it reliable.
JAX→NumPy→Torch collection overhead needs measurement before a wall-time budget;
no implementation, experiment ID, job or weight conversion authorized by this
design question alone. Existing E88 census work is separate and untouched.
Reference: https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html

E83's short-task curriculum improved local PPO farming but badly regressed
normal farming. Do not promote either E83 checkpoint or call it a timing fix.

## Jobs and completion bridges

**Overnight authority:** Dani explicitly authorizes9–10hours from approximately
2026-10-02T07:57UTC (12:57AM PDT), aiming to finish by9:57AM PDT and no later
than **10:57AM PDT /2026-10-02T17:57UTC**. Run and interpret the bounded diagnostics,
then submit ONE long checkpointed run using the remaining window. Choose from the
results; if undecided, use the requested20circular movement choices plus visible
target interface. This authorizes successive experiments after completion wakes;
the event alone need not supply new authorization. Do not revive cancelled E80.

**No new training from this design discussion.** Existing AFK census diagnostics
E88d/2193 and E88e/2194 were observed RUNNING,~34–35min remaining to their Slurm
limits at the read-only check. They belong to already ongoing census work;
their code/launchers/jobs were left untouched. E87_afk_long_continuation / Slurm2186 ended
2026-10-02T16:24:16UTC (**9:24:16AM PDT**), CANCELLED by1000/0:0; batch finished
cleanly at16:24:20UTC. Sourcef53965c, started08:54:54UTC. All17startup tests and
required180s watch passed. Log records cancellation then PROFILE COMPLETE;
study and manifest correctly say interrupted. No restart or replacement queued.
The queued modern-world benchmark belongs to the separate modern-simulator
workstream; leave it and unrelated jobs untouched.

**Completion bridge cleaned:** delivery accepted2026-10-02T16:24:19.735UTC;
lanerl-event-2186.service inactive/dead, Resultsuccess, REGISTRY entry absent.
`/mnt/nfs/shared/slurm-events/2186/result.json` records terminal accounting.
No further watcher needed. All149manifest checkpoint records point to existing
files. Final `ckpt_114393088.msgpack` and `ckpt_latest.msgpack` both83,083,080bytes,
SHA256 d76ff8fdd995719cec58b9b8965253d040aa84d6a031f657c74f7bb0e55f0e5e.
Run: `/mnt/nfs/checkpoints/lanerl-jax/E87_afk_long_continuation/vec-s0-20261002-090325-f53965ce/`.
Preserve final/intermediate checkpoints. Final6982performance is unknown;
6144is the last frozen score. No8192endpoint or final-three gate claim.

E86b/2185
COMPLETED0:0 in5m05s, ended2026-10-02T08:48:59UTC (**1:48:59AM PDT**).
Both batches and fresh/restored Adam comparisons complete. Production params,
optimizer and RNG reproduce exactly; old likelihood maxerror1.431e-5, E85
full-prefix value error8.345e-7. Bridge delivered accepted, service inactive,
registry entry absent. No model exported.

E86b verdict: no evidence that detached value-head clipping materially suppresses
actor learning here. Its first-minibatch squared-gradient share is.868%/.265%;
excluding it changes actor clip scale by+.437%/+.133%. Actor total step norm
standard→variant: fresh.613420→.613152/.523994→.524567; restored.488447→.475237/
.364972→.363474. Exact training-sequence KL standard .05356/.06610 fresh,
.04850/.03477 restored: policy steps are not vanishing. Positive-advantage and
CS-event mean log probabilities increase in all standard comparisons. Both E85
recovery-case representative clicks increase under all standard updates with
complete GRU prefixes replayed at new weights. Two batches/two such cases are
limited evidence, not a guarantee of correct credit or efficient optimal farming.

**Interrupted long-run protocol: E87_afk_long_continuation**, was2186 above.
E82final/freshAdam, unchanged CS-only full-wave PPO and LR1e-4;8192updates/
134,217,728additional decisions. Checkpoint every50updates (~3min); frozen64 at
0/128/512, then every1024through8192. Initial trajectories must equal E82final.
Final improvement gate>=15.125CS; retention>=13.125; final3scheduled means>=15.125
is a secondary sustained-improvement check. No best-checkpoint substitution.
Worker8h30/Slurm8h45, plus absolute worker cutoff **10:50AM PDT/17:50UTC** to
leave cleanup before10:57AM. Expected~8htraining plus startup/evaluation.

Choice rationale: E82's frozen curve was still increasing; E85 did not show broad
single-click recovery and E86b did not validate the proposed clipping fix. Keep
learning settings fixed and resolve whether longer exposure improves this stronger
endpoint. The20direction/visible-target proposal remains a possible later study,
not a demonstrated fix. Missing information/longer action sequences remain open;
only2E85recovery cases cannot support a credible held-out feature comparison.
The diagnostic bundle and interrupted long-run evidence are recorded. Further
diagnostic recommendations remain discussion; this cancellation wake starts no
new experiment or automatic extension of the overnight window.

E86/2184 FAILED5:0 in2m14s,
ended2026-10-02T08:41:08UTC (1:41:08AM PDT), before the worker ran.
Eight JAX canaries passed in130.97s; separate test_ppo.py skipped wholesale
because Torch is unavailable, producing pytest exit5. No optimizer result or
checkpoint. Bridge was never armed; service inactive and registry absent.
E86b repeated the same diagnostic with that optional
reference invocation removed;8JAX tests and all runtime equivalence/history gates
remain.30minSlurm/20minworker cap. This is a wrapper retry, not a changed hypothesis.

E85/2183
COMPLETED0:0 in10m23s, ended2026-10-02T08:22:45UTC (**1:22:45AM PDT**).
All32real-wave cases completed; eight integrated tests, full-prefix replay
(maxerror2.332e-6), literal CS reward, duplicate-control and finite gates passed.
Bridge delivery accepted, service inactive, registry entry absent.

E85 result: the chosen menu single click adds+.046875CS over8s and+.0078125CS
through episode end versus paired natural continuation; best sampled original
click adds+.0625/+.0703125. Repeating the chosen menu cell for1s loses−.2734375/
−.3828125CS. Only2/32cases (6,22) pass the prespecified +.5CS8s/nonnegative-tail
criterion, below the>=8gate. Discovery menu advantage+.21875CS shrinks on new
seeds. No broad isolated-click recovery demonstrated, no weights exported.
These are selected-state counterfactual deltas, NOT full-game checkpoint gains.
The32original diagnostic games average14.875CS under different RNG than the
standard64suite; retain14.125 as the established E82 comparison.

Credit diagnostic: for mean absolute discounted-return differences>=.05,
GAE difference has the same sign in16/20menu and15/17sampled-action cases;
separate repeated-action arm19/23. This exploratory thresholded description
suggests credit is not universally reversed, not proof of an unbiased or adequate
critic. Only4validation continuations/case and opportunity selection, not a census
of recoverable misses; single-click intervention cannot rule out positioning or
multi-action search. No confirmed learning bug or information ceiling.
Artifacts `/mnt/nfs/shared/E85_afk_action_search/`: complete result.json,
histories.npz, cases.msgpack, candidate_actions.npz, discovery.npz, validation.npz.

E86b implementation: Sol6090710 integrated as8d42460; root added full-prefix
readout and parameter norms. Archived agent worktree remains registered with
manual deletion approval, branch lane-rl/e86-optimizer-audit@6090710; manifest
updated after handoff. Only the three original active project workstreams remain.

E83 / Slurm2176 COMPLETED0:0 in30m56s, ended Oct1 **8:29:44PM PDT**.
Both fixed arms completed:40+20supervised epochs and256PPO updates.
All24integrated GPU tests and required launcher health watch passed.
Bridge2176 delivery accepted; lanerl-event-2176.service inactive and registry
entry absent. E81/2172 and E82/2174 bridges also delivered and cleaned up.
Other projects' jobs are outside this workstream.

E80/2171 remains CANCELLED by Dani; do not restart the12h run automatically.
It stopped after58m17s, saved update809/13,254,656decisions, status interrupted.
Last frozen u512 was11.328125CS/.03125deaths, versus initial10/.078125;
no frozen u809 evaluation and no planned11520endpoint. Bridge2171 stopped,
service/registry cleaned; sacct CANCELLED0:0 overrides its stale RUNNING poll.

## E83 — short farming acquisition and full-task transfer

Both arms independently started from E78final10.0CS parameters with fresh Adam.
64training/64held-out starts,12.8s horizon,1 or3 enemy minions plus allied
competition; varied HP/range, native10Hz controls/physics, all spells available.
No artificial action delay, new actor inputs or stale midgame recurrent carry.
Teaching:64scripted episodes/40epochs plus64learner episodes relabelled/20epochs.
PPO:256updates x128envs x128steps =4,194,304decisions, LR1e-4, CS-only.

| Frozen endpoint | One-minion CS fraction | Three-minion CS fraction | Normal120s CS | Normal deaths |
|---|---:|---:|---:|---:|
| Initial E78 |.75|.583333|10.0|.078125|
| Scripted short-task control |0|.96875|not evaluated here|—|
| Short-task teaching |.50|.75|4.875|.953125|
| Short-task PPO |1.0|.791667|1.03125|.015625|

32held-out games per short stratum;64games for each normal-task evaluation.
PPO passes the declared local+.15 fraction gate in BOTH strata (+.25/+.208333),
but both arms fail normal-task>=11CS transfer. PPO worsened all64normal games
versus initialization; teaching worsened60/64. Both final tower-damage means0.
The teaching feasibility precondition fails because the scripted single-minion
control is0, not>=.5; do not interpret this as neural imitation incapacity.

Saved round0 teaching observations show no ATTACK labels in single-minion
first episodes.29/32training cases contain1350visible killable decision frames,
but minimum distance162.2955 exceeds the script's155-unit conservative gate;
589frames lie within native165-unit reach. This establishes a script range-gate
problem on these training starts; collision/click quantization is a hypothesis,
not isolated causality or a reproduced held-out trace. The short-task design
failed to establish a competent control in both strata before supervised fitting.
Future teaching studies should make that prerequisite an actual pre-training gate.

Decoded-command diagnostics reproduce per-game frozen short CS. Ready-script
frames / same-target direct attacks: initial109/0, teaching77/47, PPO59/0.
These are decision-frame agreement counts, not independent opportunities,
actual swing counts, optimal actions or causal missed-CS counts. Spells and
held/attack-move orders remain valid alternatives. Local CS improvement alone
does not prove AA timing improved. Frozen spell selections rise with local PPO;
selections are not confirmed casts or damage attribution.

E83 source f5b8677; run:
/mnt/nfs/checkpoints/lanerl-jax/E83_afk_local_skill/local-s0-20261002-030844-f5b8677c/.
Study/manifest complete, both arms listed.60finite teaching metric rows;
256PPO rows, loss_nonfinite0, sampledpostKL median.0255745/max.0429127.
Two train episode-summary NaNs correspond to no completed episodes.
All7frozen cohorts/bothteams reward=CS with other reward terms0; initial full
trajectory retention passed.129valid decisions per first short episode,
padded to256 for recurrent training;1,310,720padded supervised samples total.
PPO final, ckpt_latest and ckpt_005505024.msgpack SHA256:
811fe2a8649a6123844f6469d3ad226ea751011c679256aa0dabc4c4adb4a147.
Teaching final SHA256:
0e8b521a75ce8b5211088aa6f87da4ffb0d8f64060d57eaf0d3e7d7dca0d3b93.
Detailed protocol, gates and limitations: LEARN-AFK-38; run row in EXPERIMENTS.

## Retained full-task results and next direction

E81/2172 completed33m06s. Frozen64 CS10.0 initial→7.28125BC→10.84375DAgger1
→11.75DAgger2; teacher13.5, declared11.475 competence gate passed.
Final deaths.0625/towerHP0. Full30/20/20epochs and512critic-only minibatches
completed; exact actor preservation during critic fitting and handoff verified.
Relabelling versus extra epochs and critic quality were not separately isolated.
E81 export: /mnt/nfs/checkpoints/lanerl-jax/E81_afk_gru_dagger/final.msgpack
SHA256 1b8ed30cf913132b8d7045e45f40cd09d9cec205a86a5a3301e99532b08f24e0.

E82/2174 completed40m24s,512updates/8.389Mdecisions, E81final/freshAdam,
CS-only PPO/LR1e-4. Frozen64 CS11.75/12.171875/12.953125/14.125 at0/128/256/512;
final deaths0/towerHP0, retention>=10.75 and improvement>=12.75 both passed.
Paired gain+2.375CS;43better/12equal/9worse. Same-suite teacher13.5CS.
Checkpoint: /mnt/nfs/checkpoints/lanerl-jax/E82_afk_clone_ppo_retention/vec-s0-20261002-021315-9c16cef5/ckpt_008388608.msgpack
SHA256 d5529d6c4d1362b27f28c2bad539a26ff42c2abe7311c414c3fcf82bca7bc357.
Original E78 final10.0CS and E75c9.453125CS are weaker continuation endpoints,
not evidence that longer training cannot help. E78 reward accounting was literal
CS-only; final ckpt008388608 SHA cd136aa512550c22c8df6e0a7ea58d9ebea3ae520bf0478b5331ea5b7a69be91.

Current discussion: Dani challenges the sample efficiency of basic CS learning;
partial improvement does not resolve that concern. E82 alone used8,388,608decisions
(~233simulated hours at10Hz) for+2.375CS. This is not a sample-efficiency benchmark
against another learner, and repeated decision frames are not independent cases.
Revise the next-step recommendation: before a long continuation, diagnose actual
recoverable misses from normal waves end to end—physical alternatives, current
probability of useful action sequences, their assigned credit, and whether a
learner update increases useful-action probability with correct GRU history.
E69/E70 validated two physical alternatives; E54/E55 audited credit/likelihood
but actions at reward time need not have caused the CS. Those are partial checks,
not a completed causal learning audit. The subsequent overnight authorization makes the concrete E85/E86 protocol active;
see jobs above for submission status.
Keep E82 as the reference. Full-wave generalization/continuation remain subsequent
options, not a substitute for resolving the learning-efficiency question.
Action search, credit/update interference and missing observable information are
hypotheses, not established causes. Frame-level ambiguity alone does not prove a
feature ceiling because the GRU also receives history. No projectile/LR changes.
E80 remains cancelled. The subsequent overnight instruction authorizes a NEW
long experiment chosen after the diagnostics, within the window above.
Any next run needs its own concrete hypothesis/spec/ID/gates and launcher/bridge.


Diagnostic rationale (E85/E86 implement bounded portions of this chain): use one
bank of actual E82 full-wave situations, retain complete observation prefixes,
and replay native physical alternatives with paired continuation seeds. First
validate which alternatives recover CS and whether the compact20-direction plus
visible-target menu covers them. Score total CS as well as the selected minion;
validate chosen alternatives on separate continuation seeds. Measure aggregate
probability over equivalent useful clicks, including ground attack-move effects,
not only one target-centre pixel. This is a local comparison, not a perfect-CS oracle.
Then compare actual outcome differences with critic/GAE credit and audit one
normal on-policy PPO update on isolated weights. Forced branches are diagnostic
labels, never fed into the ordinary learner as though they were on-policy data.
Track useful-action probability using full prefixes recomputed at new parameters;
a single positively credited example need not improve under a mixed batch.
A small supervised fit can distinguish failure to fit known decisions from
failure to discover them. If needed, compare current-history inputs with diagnostic
own-attack/projectile inputs on held-out episodes; a gain shows useful predictive
information/representation, not proof of an information-theoretic ceiling.
Sol identified code-confirmed gradient coupling, not a measured training defect:
actor and value-head gradients are globally clipped together despite detach_critic.
Measure actual actor parameter/policy movement with and without value-head
contribution on the same batch/optimizer state; Adam may cancel uniform gradient
scaling, so gradient norms alone cannot establish actor-step suppression.
Intended first bundled diagnostic budget:60–90min Slurm maximum, implementation
and exact acceptance gates still to be finalized; no12h run until evidence review.

Prior related action experiment must not be forgotten: LEARN-AFK-19/E66 found
low direct-target click mass on16selected older cases, but its learned10%proposal
mixture did not beat the original farming baseline. Different reward/init from
E82 and not equivalent to a compact categorical action menu; neither a proof
that the20-direction idea works nor a decisive refutation. Do not repeat that
experiment and describe the action-search suspicion as a new discovery.

## Earlier diagnosis and preserved drafts

E69b/E70 exactly reproduce selected E68 caster misses and establish recoverable
physical attack-timing options; no lost champion hit/CS in those cases.
LEARN-AFK-24/25 record limits; this is not global simulator/C# parity.
E71b passed visible-history preflight only; E72b cancelled, E73b unsubmitted.
E77 passed revised combat-feature/mask preflight19tests, no learned result.
Features/masks stay opt-in; E76c remains unsubmitted/superseded.
Untracked lanerl_jax/obs/projectiles.py is an untested disconnected draft,
preserved untouched. No projectile, vision, manual-play client or Tencent-weight
port was included in E81–E83. LEARN-AFK-34 clarifies Tencent Solo's structured
spatial maps are not RGB screenshots. Those options remain discussion topics.
Longer-term transfer gate: >30CS in10minute C# mirror trials across seeds.

## Three active worktrees

- Champions: t3code/toplane-champion-overlap at
  /home/dani/.t3/worktrees/ahriuwu/t3code-12dab3cf (explicit user keep).
- Modern sim: t3code/modern-jax-sim-port at
  /home/dani/.t3/worktrees/ahriuwu/t3code-86665351.
- RL learning proof: lane-rl/jax at /srv/nfs/projects/ahriuwu-lanerl-jax.

Five retired worktrees plus old main moved intact to /mnt/nfs/projects/_archive/,
logged in MANIFEST.tsv, no automatic deletion. Branches and dirty/ignored files
preserved. Old /srv/nfs/projects/ahriuwu is a compatibility symlink to archived
main, which owns the shared Git store. Worktree registrations repaired; archived
worktrees deliberately remain registered. Vendor/build dependencies retained.
T3 metadata read only; no messages sent or thread records changed.

Sol audit worktree: /mnt/nfs/projects/_archive/ahriuwu-sol-learning-audit-20261002,
branch lane-rl/e86-optimizer-audit@6090710, agent finished and integrated. Retained
registered under _archive, logged in MANIFEST.tsv, manual deletion approval required.

Reference throughput: E82updates6–512 mean3.53118s, median3.53682s,90th3.61401s.
Last64 sampledpostKL mean.03956, explainedvariance.92120, rawgradnorm7.22556,
clippedfraction1.0. These alone do not identify an LR error. E87 uses the same
LR to isolate a larger training budget from the stronger E82 initialization.
A plateau still would not prove an information ceiling or impossibility of learning.

Open distinction for interpretation: gamma.99 at10Hz weights10s-later CS by.366,
30s-later by.049, but.3s-later by.970. This may matter for pushing/proxy planning;
it does not itself explain a one-swing last-hit error. E87 holds discount fixed.
