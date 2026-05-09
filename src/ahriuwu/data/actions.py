"""Action space for LoL Garen gameplay.

Movement: continuous (x, y) in [0, 1] normalized screen coordinates.
    (0.5, 0.5) = center / no signal.
Abilities: Q W E R D F item B — each binary, 0 or 1 per frame.
"""

from dataclasses import dataclass
from typing import TypedDict

import torch

from ..constants import MOVEMENT_DIM, ABILITY_KEYS


class ActionDict(TypedDict, total=False):
    """Type hint for raw action dicts as they appear in labels/clicks."""
    movement: list[float]  # [x, y] in [0, 1]
    Q: int
    W: int
    E: int
    R: int
    D: int
    F: int
    item: int
    B: int


@dataclass
class ActionSpace:
    """LoL action space for Garen gameplay.

    Movement: continuous (x, y) coordinates in [0, 1], normalized screen space.
        (0.5, 0.5) = center / no signal.

    Ability keys (binary each):
        Q W E R   - Champion abilities
        D F       - Summoner spells
        item      - Any item active used
        B         - Recall
    """

    MOVEMENT_DIM = MOVEMENT_DIM
    ABILITY_KEYS = ABILITY_KEYS

    @staticmethod
    def to_tensor_dict(action_dict: dict) -> dict[str, torch.Tensor]:
        """Convert a raw action dict to tensors.

        Returns ``{movement: float (2,), <ability>: long ()}``.
        """
        mov = action_dict.get("movement", [0.5, 0.5])
        if isinstance(mov, (int, float)):
            mov = [0.5, 0.5]
        return {
            "movement": torch.tensor(mov, dtype=torch.float32),
            **{k: torch.tensor(action_dict.get(k, 0), dtype=torch.long) for k in ABILITY_KEYS},
        }

    @staticmethod
    def stack_tensor_dicts(tensor_dicts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Stack a list of per-frame tensor dicts into ``(T, 2)`` movement and ``(T,)`` abilities."""
        keys = ["movement"] + ABILITY_KEYS
        return {k: torch.stack([d[k] for d in tensor_dicts]) for k in keys}

    @staticmethod
    def empty_action() -> dict:
        return {"movement": [0.5, 0.5], **{k: 0 for k in ABILITY_KEYS}}


def collate_actions(
    batch_actions: list[dict[str, torch.Tensor] | None],
) -> dict[str, torch.Tensor] | None:
    """Stack action dicts across the batch dimension.

    Returns ``{movement: (B, T, 2), <ability>: (B, T)}`` or None if all entries are None.
    """
    valid = [a for a in batch_actions if a is not None]
    if not valid:
        return None
    keys = ["movement"] + ABILITY_KEYS
    return {k: torch.stack([a[k] for a in valid]) for k in keys}
