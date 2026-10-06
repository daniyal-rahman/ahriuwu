"""Jungle pet items' damage modifier (JUNGLE.md §6): +10% non-true damage to non-epic monsters.

Patch 26.1 cut the amp from 25% to 10%; true damage (Smite, pet hits) is excluded (wiki V25.S1.3 hotfix). The pet is
latched so the effect survives the egg's consumption at evolution; everything else about pets is in
``jungle.camps``. ``State.epic`` (N,) marks epic slots, set by the world.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from ...core.damage import CLASS_MONSTER, TRUE
from .core import effects, holds_any

SCORCHCLAW, GUSTWALKER, MOSSTOMPER = 1101, 1102, 1103
PETS = (SCORCHCLAW, GUSTWALKER, MOSSTOMPER)
MONSTER_AMP = 0.10

_PET_NOTE = "jungle pet: damage amp vs non-epic monsters (latched); pet rules in jungle.camps"
COVERAGE = {SCORCHCLAW: _PET_NOTE, GUSTWALKER: _PET_NOTE, MOSSTOMPER: _PET_NOTE}


class State(NamedTuple):
    pet: Any                # (C,) bool: owns or owned a pet (latched)
    epic: Any               # (N,) bool


def init(n_champions: int, n_units: int) -> State:
    return State(jnp.zeros((n_champions,), bool), jnp.zeros((n_units,), bool))


def periodic(state: State, own, ctx, units):
    return state._replace(pet=state.pet | holds_any(own, PETS)), effects(ctx.level.shape[0], units.x.shape[0])


def packet_amp(state: State, own, ctx, units, p) -> Any:
    holder = state.pet | holds_any(own, PETS)
    src_is = p.src[:, None] == ctx.unit[None, :]
    from_holder = jnp.any(src_is & holder[None, :], axis=1)
    n = units.x.shape[0]
    dst = jnp.clip(p.dst, 0, n - 1)
    monster = (units.cls[dst] == CLASS_MONSTER) & ~state.epic[dst]
    return jnp.where(p.valid & from_holder & monster & (p.dtype != TRUE), MONSTER_AMP, 0.0).astype(jnp.float32)
