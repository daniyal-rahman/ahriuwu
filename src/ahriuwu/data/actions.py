"""Action space definition for LoL Garen gameplay.

Movement: continuous (x, y) in [0, 1] normalized screen coordinates.
    (0.5, 0.5) = center of screen (default when no data available).
    Previously used 18 directional buckets (deprecated, kept for backward compat).
Abilities: Q, W, E, R, D, F - each binary
Item: single binary (any item active used)
Recall: B - binary
"""

from dataclasses import dataclass
from typing import TypedDict

import torch


class ActionDict(TypedDict, total=False):
    """Type hint for action dictionaries."""
    movement: list[float]  # [x, y] in [0, 1], or legacy int 0-17
    Q: int         # 0-1
    W: int         # 0-1
    E: int         # 0-1
    R: int         # 0-1
    D: int         # 0-1
    F: int         # 0-1
    item: int      # 0-1
    B: int         # 0-1


# Continuous movement dimension (x, y)
MOVEMENT_DIM = 2

ABILITY_KEYS = ['Q', 'W', 'E', 'R', 'D', 'F', 'item', 'B']


@dataclass
class ActionSpace:
    """LoL action space for Garen gameplay.

    Movement: continuous (x, y) coordinates in [0, 1], normalized screen space.
        (0.5, 0.5) = center of screen (default when no data).
        Legacy: 18 discrete directions (20 degrees apart) — deprecated.

    Ability keys (binary each):
        Q, W, E, R: Champion abilities
        D, F: Summoner spells
        item: Any item active used
        B: Recall
    """

    MOVEMENT_DIM = MOVEMENT_DIM
    ABILITY_KEYS = ABILITY_KEYS

    @staticmethod
    def direction_to_angle(direction: int) -> float:
        """Convert direction class (0-17) to angle in degrees.

        DEPRECATED: Use continuous (x, y) coordinates instead.

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
                movement is [x, y] floats in [0, 1].

        Returns:
            Dict with 'movement' as float tensor (2,) and abilities as long tensors
        """
        mov = action_dict.get('movement', [0.5, 0.5])
        if isinstance(mov, (int, float)):
            # Legacy: single int direction class -> default center
            mov = [0.5, 0.5]
        return {
            'movement': torch.tensor(mov, dtype=torch.float32),
            **{k: torch.tensor(action_dict.get(k, 0), dtype=torch.long) for k in ABILITY_KEYS}
        }

    @staticmethod
    def stack_tensor_dicts(tensor_dicts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Stack a list of tensor dicts into batched tensors.

        Args:
            tensor_dicts: List of dicts from to_tensor_dict()

        Returns:
            Dict with movement as (T, 2) float tensor and abilities as (T,) long tensors
        """
        keys = ['movement'] + ABILITY_KEYS
        return {k: torch.stack([d[k] for d in tensor_dicts]) for k in keys}

    @staticmethod
    def empty_action() -> dict[str, float | int | list]:
        """Return an empty action dict (center movement, no abilities)."""
        return {
            'movement': [0.5, 0.5],
            **{k: 0 for k in ABILITY_KEYS}
        }


def encode_action(movement: int, abilities: dict[str, bool] | None = None) -> int:
    """Encode action components into a single discrete action index.

    Encoding scheme (fits in 128 actions):
    - Movement (0-17): 18 directions
    - Ability flags: encoded as bits, combined with movement

    Simple encoding: movement + 18 * (ability_bit)
    where ability_bit is 0-6 representing the most significant ability pressed.
    This gives 18 * 7 = 126 actions, plus 2 reserved.

    For simplicity, if no abilities, just use movement directly.
    If any ability, use movement + 18 * (1 + ability_priority).

    Args:
        movement: Movement direction (0-17)
        abilities: Dict of ability keys to bool values

    Returns:
        Single action index in [0, 127]
    """
    movement = movement % 18  # Clamp to valid range

    if abilities is None:
        return movement

    # Priority order for abilities (most important first)
    priority = ['R', 'Q', 'E', 'W', 'D', 'F', 'item', 'B']

    for i, key in enumerate(priority):
        if abilities.get(key, False):
            # Encode as movement + 18 * (1 + priority_index)
            # This gives 18 + 18*8 = 162 max, but we cap at 127
            action = movement + 18 * (1 + i)
            return min(action, 127)

    # No ability pressed, just movement
    return movement


def decode_action(action: int) -> tuple[int, str | None]:
    """Decode action index back to movement and ability.

    Args:
        action: Single action index in [0, 127]

    Returns:
        (movement, ability_key) where ability_key is None if no ability
    """
    if action < 18:
        return action, None

    priority = ['R', 'Q', 'E', 'W', 'D', 'F', 'item', 'B']
    ability_idx = (action // 18) - 1
    movement = action % 18

    if ability_idx < len(priority):
        return movement, priority[ability_idx]

    return movement, None


def collate_actions(batch_actions: list[dict[str, torch.Tensor] | None]) -> dict[str, torch.Tensor] | None:
    """Collate action dicts from a batch.

    Args:
        batch_actions: List of action dicts or None.
            movement: (T, 2) float tensor
            abilities: (T,) long tensors each

    Returns:
        Batched action dict with movement (B, T, 2) float and abilities (B, T) long,
        or None if all None
    """
    # Filter out None entries
    valid_actions = [a for a in batch_actions if a is not None]

    if not valid_actions:
        return None

    # Stack along batch dimension
    keys = ['movement'] + ABILITY_KEYS
    return {k: torch.stack([a[k] for a in valid_actions]) for k in keys}
