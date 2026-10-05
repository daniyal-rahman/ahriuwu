"""The world unit table: the (N,) unit columns of ``ModernState``, the views subsystems read
(``WorldUnits``, item ``Units``) and the one write path for spawned rows (``write_units``)."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

from .. import mechanics as M
from ..core import types as W
from ..items.effects.core import Units

if TYPE_CHECKING:
    from .state import ModernState


def units_view(s: ModernState) -> W.WorldUnits:
    """``WorldUnits`` of the state (targetable implies alive)."""
    return W.WorldUnits(**{f: getattr(s, f) for f in W.WorldUnits._fields if f != "targetable"},
                        targetable=s.targetable & s.alive)


def item_units(s: ModernState) -> Any:
    """``items.effects.core.Units`` view of the world (wards are not item/on-hit targets)."""
    return Units(x=s.x, y=s.y, team=s.team, cls=W.damage_class(s.kind), alive=s.alive, hp=s.hp, max_hp=s.max_hp,
                 radius=s.radius, targetable=s.targetable & s.alive & (s.kind != W.KIND_WARD),
                 is_siege_or_super=(s.kind == W.KIND_MINION) & (s.sub >= 2),
                 bonus_hp=jnp.zeros_like(s.hp), armor=s.armor, magic_resist=s.magic_resist)


def reset_slots(s: ModernState, mask) -> ModernState:
    """Fresh attack state and CC timers for (re)spawned slots."""
    put = lambda arr, v: jnp.where(mask, jnp.asarray(v, arr.dtype), arr)   # noqa: E731
    return s._replace(att=s.att._replace(target=put(s.att.target, -1), windup_left=put(s.att.windup_left, 0.0),
                                         cooldown_left=put(s.att.cooldown_left, 0.0)),
                      cc=M.CCTimers(*(put(v, 0.0) for v in s.cc)))


def write_units(s: ModernState, w: W.UnitWrite, now) -> ModernState:
    """Apply a subsystem's ``UnitWrite`` (minion waves, camp spawns, epic monsters): the written rows
    become alive and targetable; fresh spawns get the next ``spawn_seq`` values, ``spawn_time = now``
    and a reset slot."""
    put = lambda arr, v, m=w.mask: jnp.where(m, jnp.asarray(v, arr.dtype), arr)   # noqa: E731
    cols = {f: put(getattr(s, f), getattr(w, f)) for f in W.UNIT_COLUMNS}
    for f in ("bounty_gold", "bounty_xp", "bounty_level"):
        if getattr(w, f) is not None:
            cols[f] = put(getattr(s, f), getattr(w, f))
    new = w.new.astype(jnp.int32)
    s = s._replace(**cols, alive=put(s.alive, True), targetable=put(s.targetable, True),
                   spawn_seq=put(s.spawn_seq, s.next_seq + jnp.cumsum(new) - 1, w.new),
                   spawn_time=put(s.spawn_time, now, w.new), next_seq=s.next_seq + jnp.sum(new))
    return reset_slots(s, w.new)
