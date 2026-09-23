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
