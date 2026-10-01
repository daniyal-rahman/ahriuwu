# STATUS — 2026-10-01 CS-only continuation

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
proof that more training cannot help. No new experiment authorized by this event.

## Jobs

**No RL jobs running or queued; no pending ETA or watcher.**
E78 / Slurm2161 COMPLETED0:0 in42m28s, ended3:53PM Pacific (PDT).
17integrated canaries and180s launcher watch passed.512metric rows, no reported
nonfinite loss flags or traceback. Final checkpoint/latest SHA256:
cd136aa512550c22c8df6e0a7ea58d9ebea3ae520bf0478b5331ea5b7a69be91.
Run: /mnt/nfs/checkpoints/lanerl-jax/E78_afk_cs_only/vec-s0-20261001-221924-cf0ff0bb/.
Bridge2161 delivery accepted; lanerl-event-2161.service inactive, registry entry
absent. No restart, extension, LR sweep, curriculum or feature run submitted.
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

## Recommendations under discussion

Tencent HoKoff publishes later1v1 TensorFlow512LSTM checkpoints, not established
original2020Solo weights. Published strongest-level comparison:70% wins vs next
level with LuBan; no perfect-CS benchmark. Inputs725 and six action heads differ
from ours. Full input semantics and cross-game transfer benefit remain unverified.
Use it as an implementation reference; direct weight transfer is not a quick fix.

Human play is currently inconvenient; defer client work. E58b already showed
scripted13.5CS versus policies~9–10 on its diagnostic cohort. Prefer a short
randomized last-hit curriculum or reuse existing scripted BC/DAgger if needed
after E78; avoid repeating a broad teacher audit. The temporary CS gain now also
makes retention across PPO updates a concrete question. These remain discussion options. Existing older BC/DAgger
success makes this plausible; those longer mirror-task scores are not comparable
to current120s AFK. LEARN-AFK-32 records sample-budget estimates and hypotheses,
including short reward horizon/personal tower credit for pushing. No experiment
created from these discussion ideas. Long-term gate remains >30CS in10minute C#
mirror trials across seeds. Future submitted runs use normal launcher, integrated
canaries/startup watch and bounded completion bridge; no repeated model polling.

Latest discussion (LEARN-AFK-34): Solo uses obstacle/hero-position game-core
channels through convolutions, not RGB screenshots. Tencent5v5 separately has
6x17x17 maps including skill bullets; caster-AA coverage remains unestablished.
Recommend retaining compact structured inputs with a separately trained video
adapter; small spatial CNN remains an optional hypothesis. No code/run change.
E78 completed; no additional RL jobs submitted.
