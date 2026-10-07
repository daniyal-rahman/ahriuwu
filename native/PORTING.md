# Porting a JAX module to the native champion layer

Goal: a faithful C++ port of the Garen-vs-Jax top-lane world (`ops/modern/bench.py --allowlist` world:
`--lanes 2 --no-jungle --no-objectives`, Garen and Jax restricted to their allow-lists). The JAX code is the
specification. Port literally: same arrays, same order of operations, same packet emission order.

## Verification standard

Every ported function is registered and replayed against real JAX calls captured from a running world:

    native/build.sh                                         # clang build -> native/build-clang/liblanesim.so
    PYTHONPATH=$PWD JAX_PLATFORMS=cpu /mnt/nfs/projects/ahriuwu-lanerl-jax/.venv-gpu/bin/python \
        -m ops.native.test_hooks captured items.fighter      # replays /mnt/nfs/shared/THROWAWAY-native001/captures/items.fighter.*.pkl

A function passes when every output leaf matches: discrete leaves exactly, float leaves exactly or within 1e-5
relative (XLA fuses multiply-adds; clang with `-ffp-contract=fast` matches most of them). Captures come from
`ops/native/capture.py` (chaos orders, rich champions buying and activating random allow-listed items).
The test prints "not ported yet: ..." for captured functions without a native registration.

## Conventions

- Types: `native/src/gen/types.hpp` is generated from the JAX NamedTuples (`ops/native/gen_types.py`); every
  struct has the JAX fields in JAX order. Never edit it; regenerate if a type is missing (add an instance to
  `examples()`).
- Arrays are `Arr<T>` (flattened row-major): `(C,)` index `c`, `(N,)` index `j`, `(C, N)` index `c * n + j`,
  `(C, S)` shield grants `c * S + k`. `C = 2` champions (holder c is world unit `ctx.unit[c]`, = c). Bools are
  `uint8_t`. Scalars (`()` leaves) are plain fields.
- Signatures: the native function takes the JAX function's arguments in order (positional arguments, then
  keyword arguments in the order the JAX caller passes them — see the captured `kw` keys), by value or const
  reference, and returns the JAX result (a struct, or `std::tuple<...>` for tuple results). Register it:

      LANESIM_TEST(items_fighter_on_hit, "items.fighter.on_hit", fighter::on_hit);   // name = capture name

- Shared helpers: `native/src/champ/core.hpp` (packets, `Effects` merge, `holds`/`can_hold`, defense/debuff
  combination, kit `KitOut`/`CCOut` merge, cast ids and timers), `stats.hpp` (stat pipeline), `rng.hpp`
  (bit-exact `jax.random` split/fold_in/uniform). Add area-local helpers in your area's header, not in shared
  files; tell the integrator if a shared helper is missing or wrong.
- Constants: anything the JAX code computes on the host (catalog `dv`, rune `ea`, champion `values`/`cooldowns`,
  economy tables, python float literals derived from data) goes in `native/python/consts/<area>_<module>.py` as
  `consts() -> {"<area>.<module>.<name>": value}`, computed with the same Python helper calls, and is read in
  C++ with `data::f("...")` / `data::table("...")` into function-local statics. Python float literals written
  inline in JAX expressions become `float` literals (`0.35f`) — JAX turns them into float32.
- Float semantics: float32 everywhere; keep JAX's expression structure (`a * b + c` stays that, so the compiler
  can contract it as XLA does); `jnp.where(c, a, b)` -> ternary; `jnp.clip` -> min/max; `argmin`/`argmax` take the
  first index on ties; `jnp.round` is round-half-to-even (`std::nearbyint`); integer `//` and `%` are floor
  division and Python modulo.
- Packets: produce the same padded arrays JAX does (same length, same `valid` flags, same order): downstream
  compaction keeps valid packets in emission order, which decides sequential resolution on champions.
- Item ownership: `holds(own, id, c)` is false for an item no holder can hold (`Owned.allowed`), exactly like the
  JAX compile-time gate. Code paths only reachable through items outside the Garen/Jax allow-lists can be
  omitted, but the module's hooks must still run for the reachable items and leave other state untouched.
- Keep it readable: one C++ function per JAX function, named the same, short comment naming the JAX source.

## Layout

    native/src/champ/items/<module>.cpp     items.effects.<module> hooks
    native/src/champ/runes/<module>.cpp     runes.effects.<module> hooks
    native/src/champ/kits/<kit>.cpp         champions.<kit> hooks, champions.summoners
    native/src/champ/econ/*.cpp             economy, role_quest, wards, inventory/shop
    native/python/consts/<area>_<module>.py constants
