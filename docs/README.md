# Docs

**Start here: [`PROJECT.md`](PROJECT.md)** for the goal, current scope,
repository map, and cleanup proposal. Research procedure and experiment
contracts live in [`EXPERIMENT_METHOD.md`](EXPERIMENT_METHOD.md); candidate
research evidence lives in [`ARCH_LIT_REVIEW.md`](ARCH_LIT_REVIEW.md).

**Technical findings: [`JAX_FIDELITY_LEDGER.md`](JAX_FIDELITY_LEDGER.md).** It is
the running technical record. Every behaviour of the JAX simulator that is not proven
identical to the vendored C# server has a row in it, every Phase 1 gate has a
verdict in it, and every number quoted anywhere else in this repo should be
traceable to it. Read the gate dashboard before interpreting an RL failure as a
learning failure.

The reference and historical documents below do not own current technical
findings. Keep results in the ledger rather than duplicating its tables.

### Source-server PPO and replays

`lanerl_jax.train.server_train` uses the shared Flax/PPO learner with real
source-server processes. It does not step JAX dynamics. The initial task has
an idle red champion and fresh random weights; its contract and measured
budget are SERVER-FIRST-05 in EXPERIMENT_METHOD. Structured observations use
the server's fog flags plus the viewport, and own HUD stats/ability enablement and authoritative death state.
JAX brush/wall vision is implemented with conservative corner handling;
direct server comparison and production throughput validation remain open.

Example commands (cap substantial work on danilogin; use a capped scope and
an immutable checkout for desktop training):

```sh
ops/login_capped.sh 6G 2 .venv-jax/bin/python -m lanerl_jax.train.server_train --envs 2 --rollout 128 --updates 2 --server-dir ../lanerl-vendor/LoLServer/GameServerConsole/bin/DeadProbe/net6.0
ops/login_capped.sh 6G 2 .venv-jax/bin/python -m lanerl_jax.train.server_eval path/to/ckpt_latest.msgpack --out lanerl_jax/runs/server-eval --server-dir ../lanerl-vendor/LoLServer/GameServerConsole/bin/DeadProbe/net6.0
ops/login_capped.sh 5G 2 .venv-jax/bin/python -m lanerl_jax.parity.render_recording lanerl_jax/runs/server-eval --out lanerl_jax/runs/server-eval/trace.npz
ops/login_capped.sh 2G 1 .venv-jax/bin/python -m lanerl_jax.replay_render lanerl_jax/runs/server-eval/trace.npz --out-dir lanerl_jax/runs/server-eval/map --video
ops/login_capped.sh 2G 1 .venv-jax/bin/python -m lanerl_jax.replay_render lanerl_jax/runs/server-eval/trace.npz --out-dir lanerl_jax/runs/server-eval/combat --view combat --video
```

For a frozen checkpoint across five sampling seeds, use the sequential cohort
launcher. Pin the checkpoint first; do not point it at a rotating latest file.
`--eval-cwd` selects the immutable evaluator checkout (default: current cwd).

```sh
ops/login_capped.sh 6G 1 .venv-jax/bin/python -m lanerl_jax.train.server_eval_batch path/to/pinned.msgpack --out lanerl_jax/runs/server-cohort --server-dir ../lanerl-vendor/LoLServer/GameServerConsole/bin/DeadProbe/net6.0 --start-near-wave --seeds 0 1 2 3 4
```

The launcher stops on an invalid episode or changed inputs and writes individual
scores and medians after all requested seeds complete. This is one training
seed's evaluation cohort, not five independently trained agents.
To combine already completed evaluations, add `--collect dir0 dir1 ...` in
the same order as `--seeds`. This validates the existing recordings and source
provenance without launching games or modifying episode directories; use a
new `--out` directory for the combined result.

`initial.msgpack` preserves the exact random starting policy for evaluation.
Each training run records its command, source archive, vendor patch and binary
hash, config, metrics, and checkpoints. A finite smoke run is not evidence of
learned farming. Episode resets launch fresh processes to preserve runes.
Server replay health/positions are dump-quantized. Separate melee swing victims
and R cast timers are unavailable; held targets and actual missiles are shown.

### Client-free policy replay

The current contracts are `screen-click-v2` and `viewport-structured-v3`.
Earlier checkpoints require their frozen code/server; recorded traces remain
viewable. Source training currently requires the HUD-capable build selected
above; it fails closed when required ability-enable or Boolean death fields are absent.

Record a checkpoint in the current training simulator, without training,
using the trainer's own sampling and decoding path:

```sh
ops/login_capped.sh 6G 2 .venv-jax/bin/python -m lanerl_jax.replay \
  --checkpoint path/to/ckpt_latest.msgpack \
  --out lanerl_jax/runs/replay/trace.npz --label "Checkpoint name"
ops/login_capped.sh 2G 1 .venv-jax/bin/python -m lanerl_jax.replay_render \
  lanerl_jax/runs/replay/trace.npz --out-dir lanerl_jax/runs/replay/view --video
```

Open `view/replay.html` for pause, seek, speed, and blue/red close-up controls,
or watch `view/replay.mp4` (ten minutes at 10x speed by default). `preview.png`
is the three-minute snapshot. The map is the actual navigation grid, with
champions, minions, turrets, health bars, current commands, held targets, and
CS. The viewer is omniscient; the policy still uses its own fog-limited
observation. Command arrows apply to the displayed **pre-action** state.

For a combat microscope without a trained checkpoint:

```sh
ops/login_capped.sh 5G 2 .venv-jax/bin/python -m lanerl_jax.replay \
  --scripted brawler --seconds 180 --out lanerl_jax/runs/combat/trace.npz
ops/login_capped.sh 2G 1 .venv-jax/bin/python -m lanerl_jax.replay_render \
  lanerl_jax/runs/combat/trace.npz --out-dir lanerl_jax/runs/combat/view \
  --view combat --video --speed 1 --start-seconds 118 --end-seconds 140
```

Combat view shows recorded missiles, attack/victim links, remaining routes,
raw clicks, HP/CS, and ability activity/cooldowns. The player adds zoom,
Blue/Red/wave following, hover inspection and a clickable-screen footprint.
HP changes are **net HP**, not source-attributed damage. Old traces lacking
missiles say unavailable. Scripted brawler is an AA/movement diagnostic
controller, not a learned policy; its inactive abilities are expected.

The NPZ keeps every decision (30 Hz); the player samples ten times per game second.
A source archive is saved before capture. The JSON sidecar records its hash,
checkpoint hash, source provenance, seed, simulator
configuration, and per-side behavior counts. This is **current-code evaluation
of saved weights**, not a reconstruction of an old training run. There are no
episode resets inside the capture. Checkpoints predating a mechanics fix may
behave differently now. Findings belong in the fidelity ledger.

### The plan
- [`PROJECT.md`](PROJECT.md) — current project scope, expansion order, and
  organizational ownership; takes precedence over historical roadmap ordering.
- [`EXPERIMENT_METHOD.md`](EXPERIMENT_METHOD.md) — trustworthy comparisons,
  feasibility, contracts, and reproducibility requirements.
- [`ROADMAP_CHAMPIONS.md`](ROADMAP_CHAMPIONS.md) — saved 2026-09-23 champion
  selection and earlier roadmap context; not current pick-rate data.
- [`JAX_REWRITE_PLAN.md`](JAX_REWRITE_PLAN.md) — scope, phases and the gate
  definitions themselves. Gate *results* live in the ledger; this file defers
  to it.

### Server reference — what the C# actually does
Durable port notes, written while reading `lanerl-vendor`. These describe the
**server**, so they age with the vendor tree, not with our simulator.

`PORT_AUDIT_AI` · `PORT_AUDIT_COMBAT` · `PORT_AUDIT_CONSTANTS` ·
`PORT_AUDIT_ITEMS_API` · `PORT_AUDIT_LIFECYCLE` · `PORT_AUDIT_MOVEMENT` ·
`PORT_AUDIT_NAVGRID` · `PORT_AUDIT_WAVES` · `CONTENT_SCRIPT_MECHANICS` ·
`LEAGUE_MECHANICS_CONCEPTS` · `MODERN_PATCH_DELTA`

### Frozen gate evidence
Long-form measurements the ledger cites. Each is a snapshot of one
investigation; when its conclusion changes, the **ledger** changes and these
keep their original numbers with a date.

`ONE_STEP_DIFFERENTIAL` · `TIER1_POST_REORDER` · `TIER2_DIVERGENCE` ·
`TICK_DIVERGENCE_TRACE` · `TICK_PARITY_AUDIT` · `PERTURBATION_RESPONSE` ·
`TARGET_ACQUISITION_DIFF` · `CALL_FOR_HELP_SWITCH_RATE`

### `archive/`
The Dreamer / world-model / BC era, plus its figures. Superseded by the JAX
rewrite and kept only so old code comments resolve. **Do not update these and
do not cite them as current.** Anything still true about the server belongs in
a port audit; anything still true about the simulator belongs in the ledger.

---

## Adding to the docs

Default to **not** adding a file. The failure mode this layout exists to fix is
real and recent: gate results lived in three documents at once, two of them
went stale without anyone noticing, and a throughput figure was quoted for
weeks from a script that was never committed.

- A new deviation, approximation or bound → **a ledger row**, in the same
  commit that introduces it.
- A new measurement of an existing deviation → **update that ledger row.**
  Strike what it replaces rather than appending; a row with two numbers in it
  is a row nobody trusts.
- A long investigation worth preserving in full → a new frozen-evidence file,
  **and** a ledger row that points at it. Never the file alone.
- Something about the C# server → the relevant port audit.

Every number that is gate evidence must name the command that produced it. See
the ledger's "Canonical commands" section; `lanerl_jax/tests/test_canonical_commands.py`
is the mechanical guard that those commands still resolve.
