# Docs

**Start here: [`JAX_FIDELITY_LEDGER.md`](JAX_FIDELITY_LEDGER.md).** It is the one
running document. Every behaviour of the JAX simulator that is not proven
identical to the vendored C# server has a row in it, every Phase 1 gate has a
verdict in it, and every number quoted anywhere else in this repo should be
traceable to it. Read the gate dashboard before interpreting an RL failure as a
learning failure.

Everything else here is one of three things, and none of them is a place to
record new findings:

### The plan
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
