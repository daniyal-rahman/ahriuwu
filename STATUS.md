# STATUS — 2026-10-02 authorized AFK diagnosis sequence

Dani cancelled E80 and explicitly prioritized E81→E82→E83 first.
E80/2171 CANCELLED0:0 after58m17s at Oct1 6:26:59PM PDT. The worker handled
termination and saved update809 /13,254,656 decisions, status interrupted.
Last completed frozen64 evaluation was update512:11.328125CS/.03125deaths,
versus initial10.0CS/.078125deaths; towerHP10.180336 versus181.035824.
No frozen evaluation at the stopped809checkpoint and no11520endpoint result.
Do not label the cancelled duration experiment successful or restart it.
E81 / Slurm2172 COMPLETED0:0 in33m06s, ended Oct1 7:01:50PM PDT.
Frozen64 CS:10.0 initial →7.28125 BC →10.84375 DAgger1 →11.75 DAgger2.
Same-cohort scripted teacher13.5; final passes declared11.475 threshold.
Final deaths.0625, personal towerHP0. Paired gain+1.75CS;40better/11equal/13worse.
Current inputs/GRU can acquire better farming with teaching; this is not perfect
CS, proof of a sole PPO mechanism, or a matched isolation of relabelling versus
more supervised training. E82/2174 now COMPLETED0:0 in40m24s, ending Oct1
7:45:09PM PDT. Frozen64 CS11.75→12.171875→12.953125→14.125 at0/128/256/512;
final deaths0, towerHP0. Retention>=10.75 and improvement>=12.75 both passed.
PPO improved the taught policy by2.375CS and exceeded the scripted teacher13.5.
This supports investigating acquisition/exploration and training distribution;
it does not isolate imitation from critic fitting or prove perfect-CS sufficiency.

E78 CS-only continuation completed. Literal+1/CS, no other environment reward;
E67 parameters/freshAdam, original v3/GRU/actions, LR1e-4,512updates/8.389Mdecisions.
Frozen64games per checkpoint,120s AFK: CS9.921875→10.765625→11.375→10.0 at
updates0/128/256/512. Final gain+.078125 versus initial; +.546875 versus E75c
unchanged-reward continuation9.453125. Fails declared finalCS>=10.921875 gate.
Intermediate improvement was not retained; no best-intermediate success claim.
Final deaths.078125 versus initial.140625; personal towerHP181.035824 versus802.399197.
All256frozen games (both teams) satisfy reward=CS and other terms0. Initial physical
retention passed. MedianpostKL.026605, maximum.047038; no obvious KL explosion.
These are one training seed's results, not a diagnosis of LR/representation or
proof that more training cannot help. E80 was subsequently explicitly authorized,
then cancelled as recorded above.

## Jobs

**No RL job currently running or queued. E83 is prepared for launch next.**
E82/2174 COMPLETED0:0;512updates/8.389Mdecisions. All4 frozen cohorts/bothteams
reward=CS, other terms0; initial E81 retention and endpoint checks passed.
Paired final-minus-initial:+2.375CS, median3;43better/12equal/9worse, range−3..7.
Final checkpoint and ckpt_latest SHA256:
d5529d6c4d1362b27f28c2bad539a26ff42c2abe7311c414c3fcf82bca7bc357.
Run: /mnt/nfs/checkpoints/lanerl-jax/E82_afk_clone_ppo_retention/vec-s0-20261002-021315-9c16cef5/.
All512 loss_nonfinite flags0;60rows have NaN episode summaries because no episode
ended in that update, not nonfinite optimization. postKL median.0426335/max.141981;
these are sampled-action rollout diagnostics, not exact whole-policy KL.
Bridge2174 delivery accepted, service inactive and registry entry absent.
E83 short-task teaching/PPO comparison implemented; integrated canaries and
launch still pending. E80 remains cancelled; no automatic12h restart.
E81 source f4cdde7;21integrated canaries passed; full70supervised epochs and
512value-head minibatches completed. Final checkpoint and exported copy SHA256:
1b8ed30cf913132b8d7045e45f40cd09d9cec205a86a5a3301e99532b08f24e0.
Run: /mnt/nfs/checkpoints/lanerl-jax/E81_afk_gru_dagger/gru-dagger-s0-20261002-013626-f4cdde73/.
Handoff complete/competent; export matches final checkpoint and frozen JSONL.
Initial retention and exact actor preservation during value fitting passed.
Critic fit completed, but no held-out critic-quality gate was imposed; do not
assume critic error is ruled out if subsequent PPO regresses.
Bridge2172 delivered accepted, service inactive and registry entry absent.
E82 started from final11.75CS and passed both retention/improvement gates at14.125.
E83 remains authorized for the subsequent short timing diagnosis.
E80 saved /mnt/nfs/checkpoints/lanerl-jax/E80_afk_cs_only_12h/vec-s0-20261002-003708-b7130a5c/ckpt_013254656.msgpack.
Slurm accounting CANCELLED/0:0 is authoritative despite generic PROFILE COMPLETE
log footer. Worker study/manifest correctly say interrupted809. Bridge2171 stopped;
service inactive and registry entry absent. Its stopped_unconfirmed result retains
stale RUNNING accounting from the preceding poll, not the final Slurm state.
Continue E81→E82→E83 now, without another user prompt.
E78 / Slurm2161 COMPLETED0:0 in42m28s, ended3:53PM Pacific (PDT).
17integrated canaries and180s launcher watch passed.512metric rows, no reported
nonfinite loss flags or traceback. Final checkpoint/latest SHA256:
cd136aa512550c22c8df6e0a7ea58d9ebea3ae520bf0478b5331ea5b7a69be91.
Run: /mnt/nfs/checkpoints/lanerl-jax/E78_afk_cs_only/vec-s0-20261001-221924-cf0ff0bb/.
Bridge2161 delivery accepted; lanerl-event-2161.service inactive, registry entry
absent. E80 subsequently cancelled by Dani; E81/E82 completed; next job is E83.
E79/2160 was cancelled after46s during canaries, before any training.
E77mask/directHP preflight1857 completed19tests; no learned feature result.
Other projects and their jobs remain outside this workstream.

## Three active worktrees

- Champions: t3code/toplane-champion-overlap at
  /home/dani/.t3/worktrees/ahriuwu/t3code-12dab3cf (kept by explicit user request).
- Modern sim: t3code/modern-jax-sim-port at
  /home/dani/.t3/worktrees/ahriuwu/t3code-86665351.
- RL learning proof: lane-rl/jax at /srv/nfs/projects/ahriuwu-lanerl-jax.

Five retired worktrees plus the old main working copy moved intact into
/mnt/nfs/projects/_archive/, logged in MANIFEST.tsv with no automatic deletion.
All branch refs and dirty/ignored files preserved. Old /srv/nfs/projects/ahriuwu
is a compatibility symlink to archived main, which owns the shared Git store.
Git worktree registrations repaired; archived worktrees intentionally registered.
CODEMAP lists active paths and retained vendor/build dependencies. T3 metadata was
read only; no messages sent or thread records changed.

## Current learning state

E46 frozen120s AFK CS5.453125→9.5625 with deaths.9375→0 demonstrated improvement.
E67 aggregate frozen baseline9.921875CS/.140625deaths remains unsolved:
/mnt/nfs/checkpoints/lanerl-jax/E67_afk_staggered/vec-s0-20261001-105559-3d1c1096/ckpt_008388608.msgpack
SHA256 7f12479d61db507d23806822a6c5b1d4b129bd58fd9e83b2bed1c06621d0fe55.
E75c unchanged-input continuation1856 completed512updates/8.389Mdecisions;
final frozen64:9.453125CS/.109375deaths. No gain; not proof more training cannot help.
Its event bridge delivered and cleaned up. E76c remains unsubmitted/superseded;
scratch initialization remains pending, not a running experiment.

E69b/E70 reproduce E68 exactly and establish two locally recoverable caster misses
with physical wait/attack options; no lost champion damage/CS in those cases.
LEARN-AFK-24/25 hold exact timing and limits. This is not global sim/C# parity.
E71b passed visible-history preflight only; E72b cancelled, E73b never submitted.
E74c passed its earlier feature revision; E77 now covers revised directHP/masks.

Combat features/masks are opt-in, self35fields, with queued AA and E cancellation
preserved. R target-conditioned legality remains absent. v5/v6 C# drivers still
reject unsupported observation extensions. Untracked obs/projectiles.py is an
untested, disconnected draft preserved untouched; not part of E77 or any result.

## Authorized next sequence — execute on completion wakes

Dani: "Ok j run ur set of expirements in order all of em/ u manage".
Subsequent constraint: no artificial delay added to the model. All stages keep
native10Hz actions and simulator timing; holding off an attack is a learned or
scripted action choice, never an injected latency or forced pause.

1. E80/2171 cancelled on Dani's request; terminal review and watcher cleanup done.
   Dani explicitly says to do E81/E82/E83 first. Do not resume E80 automatically.
2. E81_afk_gru_dagger COMPLETED2172; final frozen11.75CS versus teacher13.5,
   competence threshold11.475 passed. All planned stages complete; final actor
   unchanged by value-head fit. Export/handoff verified; bridge cleanup complete.
   No automatic extension or selection of an intermediate checkpoint.
3. E82_afk_clone_ppo_retention COMPLETED2174: fixed512updates, final14.125CS,
   deaths0; initial equality and both declared gates passed. Source/checkpoint,
   pureCS accounting and completion bridge cleanup verified.
4. E83_afk_local_skill implemented/prepared, not submitted yet. Matched E78 final
   starts: scripted64 short episodes/40epochs then learner64 relabelled/20epochs
   versus256 CS-only PPO updates (4.194M decisions), fresh optimizers/LR1e-4.
  64 train and64 unseen test starts:12.8s,1 or3 enemy minions plus allied
   competition, random HP/range, full original observations/GRU/clicks/spells,
   native physics; no imposed cooldown/delay. True history from fresh starts.
   Frozen short initial/teacher/final endpoints plus original64x120s transfer;
   decoded command agreement separates targeting from missed ready-script
   opportunities descriptively, not causal lost-CS attribution. Fixed endpoints.
   Local gate:+.15 collected fraction in BOTH1/3 strata; teacher feasibility
   >=.85teacher fraction if teacher>=.5 each; transfer gate>=11CS vs initial10.
   E82 14.125 remains the established full-task result.1hSlurm/50minworker cap,
   expected30–50min. Unequal training budgets; exploratory, not equal-compute
   algorithm ranking. Protocol/spec LEARN-AFK-38; run integrated canaries and
   existing health watch, arm bridge before handoff.

Arm/verify a bounded event bridge after each launch's required health watch,
record unit/expiry, then let it run without manual training polling. If a stage
fails technically, diagnose/repair with a unique retry ID and redirect unsubmitted
successors to that completed handoff; never consume a partial/best checkpoint.
All stages are authorized; findings do not authorize unrelated feature ports.
E81's integrated21GPU tests and full learning/handoff passed; syntax, shell
parsing and both launcher dry-runs also passed. E82 passed; E83 outcome remains pending.
LEARN-AFK-37 contains the protocol, evidence limits and source references.

## Other recommendations under discussion

Tencent implementation transfer, video adapter, richer observations and action
representation remain discussion options. LEARN-AFK-34 clarifies that Solo's
image channels are structured game-core maps, not RGB screenshots. No projectile,
vision, manual-play client or Tencent-weight port is included in this sequence.
The existing untracked projectile draft remains untouched. Long-term transfer
gate remains >30CS in10minute C# mirror trials across seeds.
