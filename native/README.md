# Native lane slice (exploration)

A C++ port of the top-lane slice of the 26.19 world tick (`lanerl_jax/modern/world/tick.py`), to measure what a
CPU simulator would cost against the JAX one. Champions are not simulated: they stay idle in the fountain.

Ported: wave spawning, structures (regen, vulnerability, backdoor, Overgrowth clocks, plates, kills), minion and
turret targeting (`lane.ai.select_targets`), route movement (`mechanics.move_step`), unit collision
(`collision.resolve`), the attack machine, minion/turret attack packets, missiles, damage mitigation and
resolution, deaths, terrain eject, fog (`vision.visibility`, `rays.clear_ray_reference`).

## Layout

- `src/geom.hpp`: walkability, segment checks, route steering, sight rays.
- `src/rules.hpp`: minion and turret formulas.
- `src/world.hpp`: the static `World` and the `Env` field list (`LANESIM_ENV_FIELDS`, mirroring `ModernState`).
- `src/tick.cpp`: the tick, one function per JAX phase or function it ports.
- `src/api.cpp`: C API (world from named values, one env in caller memory, OpenMP batches, profile).
- `python/lanesim.py`: ctypes binding; `NativeWorld(cfg)` from a JAX `WorldConfig`, `env_from_state`.
- `../ops/native/diff_tick.py`: per-tick differential test; `../ops/native/bench.py`: throughput.

Build: `make` (writes `build/liblanesim.so`; `-march=native`, so build on the machine that runs it).

## Verification

`diff_tick` imports every JAX state, steps it natively and compares every field with the next JAX state, so
errors do not compound. Every discrete field (kinds, targets, spawns, deaths, structures, missiles, visibility)
must match exactly; floats match bit for bit except at rounding level, where XLA contracts some multiply-adds
differently (`std::fma` is used where that was found: move steps, missiles).

Speedups are exact (same answers as the JAX computation): idle route slots are memoized, walkability has
clearance fast paths (`Terrain::cheb`, `Terrain::clear`, segment samples skipped by a Lipschitz bound),
pair scans use squared-distance pre-checks and per-team foe lists, damage events and recent attacks are kept
as lists (`ev_*`, `rec*`, native-only caches), fog stops at the first clear ray (nearest viewer first).

## Known JAX bug carried over

The Minion Pushing bonus (`amp` from `minion_pushing`) is computed in `lane.ai.attack_packets` but the
direct and missile packets are built without it, so it is never applied; only the divisor is.
