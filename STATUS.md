# STATUS — 2026-10-02 authorized AFK diagnosis sequence

Dani cancelled E80 and explicitly prioritized E81→E82→E83 first.
E80/2171 CANCELLED0:0 after58m17s at Oct1 6:26:59PM PDT. The worker handled
termination and saved update809 /13,254,656 decisions, status interrupted.
Last completed frozen64 evaluation was update512:11.328125CS/.03125deaths,
versus initial10.0CS/.078125deaths; towerHP10.180336 versus181.035824.
No frozen evaluation at the stopped809checkpoint and no11520endpoint result.
Do not label the cancelled duration experiment successful or restart it.
E81 is next for immediate submission under the existing sequence authorization.

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
proof that more training cannot help. The subsequent explicit user request
authorizes E80; the earlier completion event alone did not.

## Jobs

**E80 cancelled; E81 immediate launch pending. No RL job currently running/queued.**
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
absent. E80 subsequently cancelled by Dani; next job is E81.
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
2. E81_afk_gru_dagger is implemented and dry-run validated, NOT submitted.
   Launch now using `python3 ops/launch.py E81_afk_gru_dagger`.
   Existing integrated canaries plus new imitation contracts run automatically.
   Teach the existing v3/GRU/physical-click policy from E78 FINAL weights using
   32scripted episodes, then two rounds of64learner episodes relabelled by the
   script. Aggregate data;30/20/20epochs, LR1e-4,128step windows with full-prefix
   recurrent replay, attack-label weight4. Frozen64 held-out games after each
   stage and a same-cohort teacher comparator. Final competence gate is
   CS>=max(11,.85*teacherCS); no best-stage selection. Fit detached value head
   on frozen-clone returns, asserting exact actor preservation, before export.
   Estimated45–90min, first-use estimate; hard2hSlurm/105minworker cap.
3. E82_afk_clone_ppo_retention is configured and dry-run validated, NOT submitted.
   After E81 terminal review and complete handoff, launch via the same launcher.
   Uses E81 FINAL parameters and calibrated value head, verified handoff SHA,
   fresh Adam and unchanged E78 CS-only PPO;512updates/~40min,1hSlurm cap.
   Frozen64 at0/128/256/512; initial trajectory equality with E81final required.
   Compare against the immutable frozen clone; flag >1CS final regression.
   If E81 is below competence gate, still run this authorized stage, but label
   it unsuccessful-clone continuation rather than preservation of proven skill.
4. E83 reserved for the short timing diagnostic, not implemented/submitted yet.
   After E82, implement/run a bounded <=1h comparison of scripted teaching and
   CS-only PPO on short last-hit starts, with original inputs/clicks and ordinary
   physics. Randomize observable HP/range and normal wave configurations;
   separate target-choice and attack-timing outcomes. No injected action delays.
   Preserve correct recurrent history; use held-out starts and full120s transfer
   evaluation. Finalize the concrete scenario/spec and integrated canaries from
   E81/E82 evidence before launch; no need to ask Dani for another approval.

Arm/verify a bounded event bridge after each launch's required health watch,
record unit/expiry, then let it run without manual training polling. If a stage
fails technically, diagnose/repair with a unique retry ID and redirect unsubmitted
successors to that completed handoff; never consume a partial/best checkpoint.
All stages are authorized; findings do not authorize unrelated feature ports.
GPU execution tests for new code are pending E81's integrated startup suite;
syntax, shell parsing and both launcher dry-runs are the completed local checks.
LEARN-AFK-37 contains the protocol, evidence limits and source references.

## Other recommendations under discussion

Tencent implementation transfer, video adapter, richer observations and action
representation remain discussion options. LEARN-AFK-34 clarifies that Solo's
image channels are structured game-core maps, not RGB screenshots. No projectile,
vision, manual-play client or Tencent-weight port is included in this sequence.
The existing untracked projectile draft remains untouched. Long-term transfer
gate remains >30CS in10minute C# mirror trials across seeds.
