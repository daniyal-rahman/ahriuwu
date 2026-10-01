# STATUS — 2026-10-01 CS-only continuation

Current user task: run the existing policy with CS reward only for roughly40min,
while discussing better diagnostic options than requiring human play. E79 prepared (supersedes unsubmitted E78):
E67 parameters, fresh optimizer, unchanged v3/GRU/actions, LR3e-4 (3x),512updates/8.389M
learner decisions; literal+1/CS, no other environment reward. Frozen64games at
0/128/256/512; initial physical behavior retained, reward intentionally changed.
Final primary gate CS>=10.921875; deaths/tower damage reported as outcomes.
Code adds integrated CS-isolation smoke test and frozen reward=CS assertions.
No manual-play client or new teacher/curriculum experiment authorized by this
ideation alone. Recommendations recorded in LEARN-AFK-33.

## Jobs

**E79_afk_cs_only_lr3e4 PREPARED, not submitted yet.** Expected runtime~40min;
55minworker/1hSlurm cap. Normal dry-run, canary/startup watch and event bridge
required. No RL job currently running or queued.
E77_combat_mask_preflight / Slurm1857 COMPLETED0:0 in4m27s, ended21:28:35UTC.
Existing log:19GPU tests passed, CANARY PASSED and PROFILE COMPLETE; result.json
passed for matching job/spec/sourceb63227e5. This resolves the previous interrupted
handoff's pending gate. No1857 event service or registry entry. No tests rerun.
E77 checks direct HP, own action availability, standable-map masks and shared
actor/learner/reset/update behavior; it is not a learned farming result.

Other workstreams have their own jobs: modern-sim replay extraction2010/2011 and
analysis2022 were active/queued during review; touchline1803 also runs. Those jobs,
their compute and their handoffs are outside this RL session; none were stopped.

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

## Recommendations under discussion

Tencent HoKoff publishes later1v1 TensorFlow512LSTM checkpoints, not established
original2020Solo weights. Published strongest-level comparison:70% wins vs next
level with LuBan; no perfect-CS benchmark. Inputs725 and six action heads differ
from ours. Full input semantics and cross-game transfer benefit remain unverified.
Use it as an implementation reference; direct weight transfer is not a quick fix.

Human play is currently inconvenient; defer client work. Prefer evaluating the
existing scripted teacher on matched AFK starts, then a teacher/short last-hit
curriculum diagnostic if needed after the CS-only result. Existing older BC/DAgger
success makes this plausible; those longer mirror-task scores are not comparable
to current120s AFK. LEARN-AFK-32 records sample-budget estimates and hypotheses,
including short reward horizon/personal tower credit for pushing. No experiment
created from these discussion ideas. Long-term gate remains >30CS in10minute C#
mirror trials across seeds. Future submitted runs use normal launcher, integrated
canaries/startup watch and bounded completion bridge; no repeated model polling.
