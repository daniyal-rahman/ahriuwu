# STATUS — 2026-10-02 completed AFK diagnosis sequence

E81→E82→E83 is complete. The best full-task endpoint in this sequence is **E82:
14.125 CS, zero deaths, zero personal tower damage**, over64 frozen120s AFK games.
Teaching on full waves improved the earlier10.0CS policy to11.75; subsequent
CS-only PPO improved it to14.125. This is evidence of learning with the existing
observations/GRU/clicks, not perfect CS or an isolated explanation of the plateau.

E83's short-task curriculum improved local PPO farming but badly regressed
normal farming. Do not promote either E83 checkpoint or call it a timing fix.

## Jobs and completion bridges

**No RL training/evaluation jobs running or queued. No next run submitted.**
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
not a completed causal learning audit. Specify a bounded experiment only after
this protocol is concrete; none submitted or implemented in this discussion.
Keep E82 as the reference. Full-wave generalization/continuation remain subsequent
options, not a substitute for resolving the learning-efficiency question.
Action search, credit/update interference and missing observable information are
hypotheses, not established causes. Frame-level ambiguity alone does not prove a
feature ceiling because the GRU also receives history. No projectile/LR changes.
No automatic12h restart or new feature port. Dani's standing bounded-research
authorization remains; the explicitly requested E81–E83 sequence is complete.
Any next run needs its own concrete hypothesis/spec/ID/gates and launcher/bridge.

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
