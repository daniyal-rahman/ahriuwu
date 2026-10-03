"""Jungle pet items 1101 Scorchclaw Pup, 1102 Gustwalker Hatchling, 1103 Mosstomper Seedling.

Spec: docs/modern/JUNGLE.md §6. The item-side part is the *damage modifier*: holders deal +10%
damage to non-epic monsters (PATCH 26.1 "Monster damage amp 25% => 10%"; not true damage, so
not Smite or the pet's own hits, WIKI V25.S1.3 hotfix). The pet type is latched so the effect
survives the egg's consumption at the final evolution ("The item's effects ... persist", WIKI).

Everything that needs the jungle (pet attacks, treats, evolutions, Smite upgrades, kill
heal/mana, bonus XP, the 50% damage taken from monsters, the Monster Hunter / minion-XP rules
and the three evolution buffs) lives in ``modern_jungle``, which reads the same latched type
(``modern_jungle.latch_pets``).

Epic monsters are excluded through ``State.epic`` (N,) bool, which the world sets once (the
epic slots); by default every monster counts as non-epic.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from ..modern_damage import CLASS_MONSTER, TRUE
from .core import effects, row

SCORCHCLAW, GUSTWALKER, MOSSTOMPER = 1101, 1102, 1103
PETS = (SCORCHCLAW, GUSTWALKER, MOSSTOMPER)
MONSTER_AMP = 0.10

COVERAGE = {
    SCORCHCLAW: "jungle pet: +10% non-true damage to non-epic monsters (latched); pet attacks, treats, "
                "Smite upgrades, kill rewards and Scorchclaw's Slash in modern_jungle",
    GUSTWALKER: "jungle pet: +10% non-true damage to non-epic monsters (latched); pet attacks, treats, "
                "Smite upgrades, kill rewards and Gustwalker's Gait in modern_jungle",
    MOSSTOMPER: "jungle pet: +10% non-true damage to non-epic monsters (latched); pet attacks, treats, "
                "Smite upgrades, kill rewards and Mosstomper's Courage in modern_jungle",
}


class State(NamedTuple):
    pet: Any                # (C,) bool: holder owns or owned (consumed at evolution) a jungle pet
    epic: Any               # (N,) bool: epic monster slots (set by the world)


def init(n_champions: int, n_units: int) -> State:
    return State(jnp.zeros((n_champions,), bool), jnp.zeros((n_units,), bool))


def _owned(own) -> Any:
    out = jnp.zeros(own.shape[:1], bool)
    for iid in PETS:
        out = out | (own[:, row(iid)] > 0)
    return out


def periodic(state: State, own, ctx, units):
    return state._replace(pet=state.pet | _owned(own)), effects(ctx.level.shape[0], units.x.shape[0])


def packet_amp(state: State, own, ctx, units, p) -> Any:
    """(P,) +10% on non-true packets from a pet holder to a non-epic monster."""
    holder = state.pet | _owned(own)
    src_is = p.src[:, None] == ctx.unit[None, :]
    from_holder = jnp.any(src_is & holder[None, :], axis=1)
    n = units.x.shape[0]
    dst = jnp.clip(p.dst, 0, n - 1)
    monster = (units.cls[dst] == CLASS_MONSTER) & ~state.epic[dst]
    return jnp.where(p.valid & from_holder & monster & (p.dtype != TRUE), MONSTER_AMP, 0.0).astype(jnp.float32)
