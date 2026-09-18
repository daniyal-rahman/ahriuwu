"""Recover canonical entity identity from the optional diagnostic NetIds."""
from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

from .inject import UnitInjectionNote
from .trace import Entity, Snapshot

__all__ = ["net_id_to_entity", "net_id_to_injected_slot"]


def _pair_internals(snapshot: Snapshot, entities: Iterable[Entity]):
    """Yield ``(internal, entity)`` using the injector's greedy correspondence."""
    unmatched = list(snapshot.ai_internals)
    for entity in entities:
        candidates = [v for v in unmatched
                      if v.kind == entity.kind and v.team == entity.team]
        if not candidates:
            continue
        internal = min(
            candidates,
            key=lambda v: ((v.q_x - entity.q_x) ** 2
                           + (v.q_y - entity.q_y) ** 2))
        unmatched.remove(internal)
        yield internal, entity


def net_id_to_entity(snapshot: Snapshot) -> Dict[int, Entity]:
    return {internal.net_id: entity
            for internal, entity in _pair_internals(snapshot, snapshot.entities)}


def net_id_to_injected_slot(snapshot: Snapshot,
                            notes: Sequence[UnitInjectionNote]) -> Dict[int, int]:
    entities = [note.entity for note in notes if note.entity is not None]
    slot_by_entity = {id(note.entity): note.slot for note in notes
                      if note.entity is not None}
    return {internal.net_id: slot_by_entity[id(entity)]
            for internal, entity in _pair_internals(snapshot, entities)}
