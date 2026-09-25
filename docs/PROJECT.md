# Project direction and repository map

Updated 2026-09-24 from Dani's instructions. This file owns project scope,
navigation, and organizational decisions. The [fidelity ledger](JAX_FIDELITY_LEDGER.md)
owns current technical findings; [experiment method](EXPERIMENT_METHOD.md)
owns research procedure. Do not copy their result tables here.

## Destination and current scope

**Build an agent that can play a real game of modern League of Legends,
in 5v5 against D2-level opponents, and win.** Evaluation must eventually
measure real match outcomes; simulator reward, mirror-lane CS, and a single
win are intermediate evidence. The final opponent protocol, lineup, patch,
number of games, and success threshold remain to be defined.

Today’s laboratory is a ten-minute, top-lane Garen mirror 1v1 on a legacy
server whose source we can inspect. The JAX simulator accelerates training;
the source-available server provides an independent mechanics reference and
execution target. Neither is the final modern game.

Current priority (2026-09-25): establish a randomly initialized, source-server-only
PPO farming baseline, then compare the same learner/task on JAX (SERVER-FIRST-01
in EXPERIMENT_METHOD). Structured screen-limited observations are intended; pixel
recognition is deferred. Dani confirmed on 2026-09-25 that the desktop was
switched to Windows and instructed us to use danilogin for now. Keep substantial
work capped there; do not attempt desktop launches until Dani changes that
availability. Harden
navigation, visibility, the action interface, and the RL/server loop as needed. The replay investigation shows poor farming
and asymmetric behavior; attack-chase routing and screen-click semantics
are under repair. Older pointer-policy replay controls are diagnostic
evidence, not validation of the replacement action interface. Read the
ledger for measured results and remaining checks; baseline readiness is
not established by this organizational update.

**Action contract:** the actor emits buttons and screen coordinates. The
environment/server interface projects and resolves the click against the
world to determine movement, attack, or spell behavior. Selecting an entity
by ID/name directly from the policy is not allowed. An omniscient replay
viewer is a diagnostic tool, not the actor's observation.

## Expansion order

| Stage | What it must establish |
|---|---|
| Now: ten-minute Garen mirror | Reliable control, reachable wave interactions, useful farming, interpretable learning, server execution |
| Longer horizons / full games | Objectives and consequences beyond lane; explicit terminal vs training-cutoff semantics |
| More champions and matchups | Shared mechanics, held-out matchup strength, measured adaptation to new champions |
| Modern League mechanics | A pinned modern patch and modern-game validation; precedes claims about modern League strength |
| 5v5 | Cooperation and team win outcomes, then the D2-level real-game evaluation |

This is the rough order requested on 2026-09-24, not a locked dependency
schedule. Modernization may move earlier, including before 5v5. The
[2026-09-23 champion notes](ROADMAP_CHAMPIONS.md) proposed modernization
immediately after the initial loop; they remain historical planning context.
Choose the next stage from evidence and measured cost, and record the reason
here when that decision is made.

Replay imitation initialization with a diminishing KL prior remains the
intended research direction, not a claim that the present baseline implements
it. PPO alternatives, recurrence vs temporal attention, critic changes,
mechanics transfer, and selective simulator supervision are candidate
experiments. The attached architecture conversation is substantially captured
in [EXPERIMENT_METHOD](EXPERIMENT_METHOD.md) and
[ARCH_LIT_REVIEW](ARCH_LIT_REVIEW.md); its suggestions are not an instruction
to implement every idea or treat an architecture as the predetermined winner.

## Where things belong

| Material | Location and rule |
|---|---|
| Agent entry instructions | Root `AGENTS.md`; short links and invariants, not a second project history |
| Scope / roadmap / organization | This file |
| Current correctness and measurement verdicts | `docs/JAX_FIDELITY_LEDGER.md`; update the relevant row and reproduction command |
| Experiment selection and contracts | `docs/EXPERIMENT_METHOD.md`; literature evidence in `docs/ARCH_LIT_REVIEW.md` |
| Server mechanics reference | Existing `docs/PORT_AUDIT_*.md` and content/mechanics notes |
| JAX runtime and reusable tools | `lanerl_jax/{sim,obs,train,data,parity}/`; shared replay tools remain `lanerl_jax/replay.py`, `replay_render.py`, `replay_player.html` |
| Real-server bridge / evaluation | `lanerl/`, `lanerl_rl/`, `lanerl_bot/`, `lanerl_train/`; inspect imports before treating any as obsolete |
| Durable regressions | Owning package's `tests/`; cross-cutting JAX infrastructure in `lanerl_jax/tests/` |
| Reusable launches / resource controls | `ops/`; scheduler wrappers in `slurm/` |
| Disposable probes and generated artifacts | Named `lanerl_jax/runs/<investigation>/` directories (ignored); no new root-level scripts or loose outputs |
| Finished, versioned one-off parity probes | Existing `lanerl_jax/parity/archive/`, with ledger linkage; preserve canonical commands |
| Historical research | `docs/archive/`, existing archived scripts, and legacy `src/ahriuwu/`; do not cite as current |

A one-off **test** that catches a durable bug belongs in the normal suite.
A script that measures one checkpoint once is a **diagnostic**, even if its
filename starts with `test_`. Save it with its investigation; promote the
smallest generally useful tool if others must reproduce the finding. Keep
large traces/video/cache files outside versioned source. An ignored run is
not a backup: retain valuable artifacts on persistent storage and record
their location and hashes before removing local copies.

## Cleanup proposal and audited candidates

Inventory on 2026-09-24 used `git ls-files` (tracked files only): 367 entries
under `scratchpad/`, 277 under `scripts/`, 120 under `docs/`. Counts identify
review areas; they do not establish that files are redundant. No bulk files
were moved or deleted in this pass.

| Candidate | Finding | Proposed disposition / check before changing |
|---|---|---|
| Root `README.md` | August Dreamer pipeline still presented as current | Add current entry links and explicitly mark the preserved body historical; later archive the body after checking inbound links |
| `scratchpad/audit_cache/*.npz`, `scratchpad/dreamq_stills/`, `scratchpad/hp_recon_stills/`, `scratchpad/lh_*.npz` | Generated data/figures are tracked despite artifact ignores | Inventory hashes, referenced evidence and regeneration commands; retain a durable artifact copy, then untrack in a separate migration |
| `scratchpad/bc_night.sh`, `bc5080_gate_watchdog.sh`, `tok_eval_watcher.py` | Names overlap with `ops/` and old launch references remain | Compare contents and active launchers; migrate callers to the canonical operational copy before retiring duplicates |
| `scratchpad/audit_align*.py`, `audit_projection*.py`, `cs_readout_probe*.py`, `lh_*.py` | Numbered/variant investigation families mix source and outputs | Group by the question answered; promote a reusable tool or freeze evidence, then retire only proven-unused variants |
| `scripts/_archive/`, `scripts/_deprecated/`, legacy `scripts/` | Multiple historical groupings coexist | Keep current paths until import/CLI/reference checks establish a safe migration; consolidate archive navigation first |
| `lanerl_jax/parity/archive/` | Already has an evidence-to-script index, but reports stale old paths | Reuse this pattern; fix references when each probe is next touched, do not create another archive hierarchy |
| `lanerl_jax/runs/nofarm/`, `hardening_20260924/`, `minimap_20260924/` | Useful ignored evidence includes scripts and derived outputs | Keep artifacts; promote any unique reproduction code required by a ledger claim; distinguish historical weights from current-code evaluation |
| New replay, combat, chase, telemetry and ops tests | Different behaviors, not automatically duplicate because written during one session | Keep behavioral regressions; review common fixtures and overlapping assertions within each subsystem |
| `.gitignore` | Unanchored `data/` conflicts with anchored source exceptions | Audit ignored source with `git check-ignore -v` before a targeted rule change; do not broadly expose large data directories |
| Root `audit_agent_finetune.py`, Windows-named batch file, old metadata | Historical root-level clutter | Check callers and provenance; relocate in the legacy pass rather than alongside active runtime fixes |
| `ops/README.md` | Only documents old BC operations | Add current JAX launch links; preserve old runbook until operational dependencies are checked |

Migration order:

1. **Now:** establish navigation, file-placement rules, source-of-truth ownership,
   and experiment record requirements. Preserve runtime paths during hardening.
2. **Next bounded cleanup:** inventory the named scratchpad families and their
   inbound references; separate artifacts from scripts; promote required
   reproduction tools. Record old path → new path and affected callers in the
   migration change. Verify only affected CLIs/imports and retained evidence.
3. **After active fixes stabilize:** consolidate redundant tests/fixtures and
   legacy script entry points; archive stale root documentation. Preserve
   unique bug coverage and canonical reproduction commands.

Prefer small migrations with reviewable diffs. File count reduction is not the
success metric: a new agent should find the current goal, code, evidence,
reproduction command, and reason for the next action without reading chat.

## Keeping the record current

The agent making a change owns its corresponding record. Update it in the
same change or when the investigation finishes, rather than creating a new
handoff document. At a scope change, update this page; at a technical result,
update the ledger; at an experiment decision, update its contract. The next
agent checks these records against the tree and runs before relying on them.

Keep active entries brief: question, status, evidence/run link, decision and
reason, next unresolved check. Distinguish planned, running, complete,
inconclusive, and stopped. Supersede stale current conclusions with links to
the frozen evidence; do not silently erase the reason an earlier choice was
made. Review organization when an investigation closes or a reusable tool
emerges; no separate daily reporting ceremony is needed.
