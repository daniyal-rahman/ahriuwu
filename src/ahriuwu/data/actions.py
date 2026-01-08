"""Action space definition for LoL Garen gameplay.

All actions are discrete/categorical with factorized embeddings.

Movement: 18 classes (0-17 = evenly spaced 20° directions)
Abilities: Q, W, E, R, D, F - each binary
Item: single binary (any item active used)
Recall: B - binary
"""

from dataclasses import dataclass
from typing import TypedDict

import torch


class ActionDict(TypedDict, total=False):
    """Type hint for action dictionaries."""
    movement: int  # 0-17
    Q: int         # 0-1
    W: int         # 0-1
    E: int         # 0-1
    R: int         # 0-1
    D: int         # 0-1
    F: int         # 0-1
    item: int      # 0-1
    B: int         # 0-1


# Constants for action space sizes
MOVEMENT_CLASSES = 18  # 0-17 = directions (20° apart)
ABILITY_KEYS = ['Q', 'W', 'E', 'R', 'D', 'F', 'item', 'B']


@dataclass
class ActionSpace:
    """LoL action space for Garen gameplay.

    Movement encoding (18 classes):
        0-17: evenly spaced directions (20° apart, starting from 0°=East)
            0: 0° (East/Right)
            1: 20°
            2: 40°
            ...
            9: 180° (West/Left)
            ...
            17: 340°

    Ability keys (binary each):
        Q, W, E, R: Champion abilities
        D, F: Summoner spells
        item: Any item active used
        B: Recall
    """

    MOVEMENT_CLASSES = MOVEMENT_CLASSES
    ABILITY_KEYS = ABILITY_KEYS

    @staticmethod
    def angle_to_direction(angle_degrees: float) -> int:
        """Convert angle in degrees to direction class (0-17).

        Args:
            angle_degrees: Angle in degrees (0=East, 90=North counter-clockwise)

        Returns:
            Direction class 0-17 (20° buckets)
        """
        # Normalize to [0, 360)
        angle = angle_degrees % 360
        # Each bucket is 20°, with bucket 0 centered at 0°
        # So bucket boundaries are at -10°, 10°, 30°, 50°, etc.
        bucket = int((angle + 10) / 20) % 18
        return bucket

    @staticmethod
    def direction_to_angle(direction: int) -> float:
        """Convert direction class (0-17) to angle in degrees.

        Args:
            direction: Direction class 0-17

        Returns:
            Angle in degrees (center of bucket)
        """
        return direction * 20.0

    @staticmethod
    def to_tensor_dict(action_dict: dict) -> dict[str, torch.Tensor]:
        """Convert raw action dict to tensor dict.

        Args:
            action_dict: Single action with keys 'movement', 'Q', etc.

        Returns:
            Dict with same keys but torch.long tensors
        """
        return {
            'movement': torch.tensor(action_dict.get('movement', 0), dtype=torch.long),
            **{k: torch.tensor(action_dict.get(k, 0), dtype=torch.long) for k in ABILITY_KEYS}
        }

    @staticmethod
    def stack_tensor_dicts(tensor_dicts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Stack a list of tensor dicts into batched tensors.

        Args:
            tensor_dicts: List of dicts from to_tensor_dict()

        Returns:
            Dict with stacked tensors of shape (T,) for each key
        """
        keys = ['movement'] + ABILITY_KEYS
        return {k: torch.stack([d[k] for d in tensor_dicts]) for k in keys}

    @staticmethod
    def empty_action() -> dict[str, int]:
        """Return an empty action dict (all zeros)."""
        return {
            'movement': 0,
            **{k: 0 for k in ABILITY_KEYS}
        }


def collate_actions(batch_actions: list[dict[str, torch.Tensor] | None]) -> dict[str, torch.Tensor] | None:
    """Collate action dicts from a batch.

    Args:
        batch_actions: List of action dicts (each with shape (T,) tensors) or None

    Returns:
        Batched action dict with shape (B, T) tensors, or None if all None
    """
    # Filter out None entries
    valid_actions = [a for a in batch_actions if a is not None]

    if not valid_actions:
        return None

    # Stack along batch dimension
    keys = ['movement'] + ABILITY_KEYS
    return {k: torch.stack([a[k] for a in valid_actions]) for k in keys}
