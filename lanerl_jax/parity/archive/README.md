# parity/archive -- one-off probes whose answer is already a ledger row

Moved here by `PARITY-001` (2026-09-23) so the live parity core is legible. Each
file answered one question once; the answer lives in `docs/JAX_FIDELITY_LEDGER.md`,
not in re-running the file. Nothing in `sim/`, `train/`, `obs/` or the live
`parity/` modules imports any of them. Run as
`python -m lanerl_jax.parity.archive.<name>` (relative imports were re-pointed
one level up). Rows marked "nearest" are not named by the ledger; the file's
docstring holds the result and the row given is the one it fed.

**Two are scripts, not modules: `bot_cs_probe.py` launches a C# server and
`leg_probe.py` runs a sim at IMPORT time** -- do not import them to check them.

| file | what it measured | ledger row(s) |
|---|---|---|
| `bot_cs_probe.py` | the C# `LanerlBot`'s own CS on the server with no orders sent (33 CS reference) | `ACCEPT-001` (also quoted by `RL-001`/`RL-002`, superseded by `SPELL-001`) |
| `leg_probe.py` | whether the 545 u detour on lane leg wp0->wp1 is the route table or the waypoint follower | `PATH-007` |
| `truncation_bound_probe.py` | the true per-tick maxima behind `MAX_STEPS_PER_TICK` / `MAX_ESCAPES_PER_UNIT` over a combat corpus | `BOUND-001` (its copy of the ghost-flag line is `SPELL-005`'s second site) |
| `tier1_same_wave_collision.py` | the ten same-wave collision residuals: a rounded `MINION_SPAWN` constant, not collision; `--rounded-spawn` is the regression control | `RESET-004` |
| `tier1_wave_spawn_detail.py` | every wave-spawn population mismatch on a fresh trace after the spawn-timing (`t_now`) fix | nearest: gate-1 dashboard row ("Wave spawning: 6 of 120 spawn ticks") / `RESET-004` |
| `tier1_bias_crowding.py` | whether the residual along-heading position bias concentrates where a unit has >=2 overlapping neighbours | nearest: `INJ-002` (the crowding gradient that flattened once the injector bug was fixed), `COLL-002` |
| `tier1_collision_sequential.py` | a sequential (Gauss-Seidel, multi-push, creation-order) collision reference against the sim's Jacobi single-push pass | nearest: `COLL-002`, `INJ-002` |
| `turret_target_drill.py` | why the server's LaneTurret refuses targets the sim takes (1,508 turret target disagreements) | `TURRET-001` |
| `writesite_join.py` | tier-1 residual rows joined against the server's own write-site branch stream | `WRITE-001` |
| `reach_hp_mortality.py` | gate 3's in-reach HP asymmetry and minion mortality census | nearest: `GATE3-002` (hp_band lead), `STAT-002` |
| `gate3_attribution.py` | gate 3's approach-cost / deaths / level gaps attributed to one respawn walk | `GATE3-001`, `COLL-003` (restored with `LEDGER-001`) |
| `gate3_decompose.py` | gate 3's outcome gap broken into factors countable on both sides | `GATE3-002` (mechanism: `PATH-007`) |
| `gate3_swarm.py` | why the sim champion is swarmed by red minions (+39% damage taken) | nearest: `CFH-001` ("clumping" lead), `GATE3-002` |
| `gate3_arrivals.py` | the swarm arrival process: the sim is stickier, not burstier; call-for-help release audit | `CFH-001` |
| `isolation.py` | whether the champion stands inside its own wave; drove a raw two-point path while the gate ran routed | `PATH-006`, `METH-004` |

Kept LIVE, not archived, although named `gate3_*`: `gate3_first_divergence.py`
(the first-divergence method `PARITY-001` reuses) and `gate3_outcomes.py` (the
gate-3 outcome driver scored against the shuffle floor, `GATE3-003`), which
also imports `gate3_first_divergence`.

Known stale pointers that could not be edited from here (outside this change's
scope): `sim/init.py` and `sim/collision.py` docstrings still cite
`parity.tier1_collision_sequential` at its old path; the ledger cites every
file above at its old `parity/<name>.py` path.
