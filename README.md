# lanerl-jax modern: a 26.19 Summoner's Rift simulator in JAX

A fixed-shape, `jit`/`vmap`-able simulation of a patch-26.19 League of Legends game (Garen vs Jax) with
every modern system: minion waves in all three lanes, turrets and structures, jungle camps and epic
objectives, fog of war and wards, both champion kits, summoner spells, items and runes through one damage
pipeline, economy and the top role quest. A PPO trainer acts through screen clicks, as a player does.

- [docs/modern/WORLD_IMPLEMENTATION.md](docs/modern/WORLD_IMPLEMENTATION.md): start here. Package layers,
  unit layout, the tick's phase order, verification and known gaps.
- [docs/modern/README.md](docs/modern/README.md): the per-system specs (values and evidence).
- [docs/CODEMAP.md](docs/CODEMAP.md): where everything lives. [STATUS.md](STATUS.md): current state.
- [docs/JAX_FIDELITY_LEDGER.md](docs/JAX_FIDELITY_LEDGER.md): every finding as a row.

```python
from lanerl_jax.modern import world
cfg = world.build_config(loadouts)            # host-side: map, routes, structures, loadouts
s = world.init_state(cfg)
s, events = world.step(s, world.no_orders(), cfg)   # one 30 Hz tick; jit/vmap/scan it
```

Train: `python -m lanerl_jax.modern.train --help`. Tools (benchmark, profiler, perf guard, fidelity
checks): [ops/README.md](ops/README.md). Tests: `python -m pytest lanerl_jax/modern/tests` (full-tick
tests compile for minutes; run them through Slurm).

The legacy C#-server project (LeagueSandbox parity sim, server-side PPO, probes, experiments) was removed
from this branch on 2026-10-05; it is at tag `pre-modern-cleanup-2026-10-05`.
