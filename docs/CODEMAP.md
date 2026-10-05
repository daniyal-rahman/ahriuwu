# Code map

Everything is the `lanerl_jax/modern/` package, layered so each layer imports only the ones above it:

| layer | modules |
|---|---|
| contract and rule math | `core/` — `types` (WorldUnits, UnitWrite, attack/cast/CC/dash records), `stats`, `stat_pipeline` (STAT.*), `damage` (DMG.* packets, shields, heals) |
| map | `map/` — `terrain`, `pathing` (route graph), `lanes`, `regions`, `dynamic_terrain` (structure pads), `rift` (Elemental Rift / Baron pit) |
| rules | `mechanics` (attack machine, missiles, CC, movement), `collision`, `vision` + `rays`/`ray_kernel`, `lane/` (minions, towers, lane AI), `jungle/` (camps, objectives), `wards`, `economy`, `role_quest`, `champions/` (kit registry, Garen, Jax, summoners), `items/` (catalog, inventory, loadout, effects), `runes/` (catalog, effects), `combat` (items + runes around the damage pipeline) |
| world | `world/` — `config` (build_config, Layout), `state`, `units`, `views`, `scratch`, `phases/*` (one module per tick phase), `tick` (`step`) |
| interface | `obs`, `actions`, `screen`, `frame`, `train` + `rl/` (policy, PPO, learner, run directory) |
| data | `data/` — pinned 26.19 tables (`26.19/`), replay/Riot oracles (`oracle/`), loaders and `build_*` regenerators |

Tests: `lanerl_jax/modern/tests/`. Tools: `ops/modern/` ([ops/README.md](../ops/README.md)). Specs:
`docs/modern/`. Pinned map arrays and routes live on NFS (`/mnt/nfs/shared/modern-world-map-research/`,
`/mnt/nfs/shared/WORLD001_map_routes/`).
