# Working on ahriuwu

Read [docs/PROJECT.md](docs/PROJECT.md) first: goal, current priority, file map,
and cleanup plan. Read the relevant rows in
[docs/JAX_FIDELITY_LEDGER.md](docs/JAX_FIDELITY_LEDGER.md) before interpreting
simulator or RL behavior. Research work follows
[docs/EXPERIMENT_METHOD.md](docs/EXPERIMENT_METHOD.md).

- The destination is winning modern League 5v5 against D2-level opponents.
  The current laboratory is ten-minute Garen mirror lanes; passing it does
  not establish full-game strength or champion generalization.
- Actor actions are buttons plus screen coordinates. Resolve the click to
  terrain/entities in the environment/server interface; never give the actor
  a direct entity-ID target shortcut. Keep privileged diagnostic state out
  of actor observations.
- Inspect existing code/docs and the working-tree diff before editing.
  Preserve other agents' work. Check active work ownership before parallel edits.
- Put reusable code in its owning package, operational launchers in `ops/`,
  and durable regressions beside the subsystem they protect. Put disposable
  investigation code and outputs under a named ignored run directory. Promote
  any script needed to reproduce cited evidence into versioned source.
- Add findings to the fidelity ledger, not a new status/report file. Keep
  project scope in PROJECT, experiment contracts in EXPERIMENT_METHOD, and
  historical evidence frozen. Update the relevant record with the code change
  or completed investigation; record why, evidence, and the next unresolved check.
- Do not prune tests by age or file count. Keep independent behavioral
  regressions; consolidate redundant fixtures/assertions only after checking
  their distinct coverage. Do not add tests for documentation-only changes.
- Record the command, resolved configuration, code/input provenance, seeds,
  runtime, and outcome for evidence-producing runs. A dirty-tree hash without
  the actual source changes is not enough to reconstruct a run.
- Respect the user's current machine availability. On danilogin, cap JAX and
  other substantial workloads with `ops/login_capped.sh <memory> <cpus> ...`.
  Do not infer desktop availability from old runbooks.
