"""Compute reward signals from game state."""

from dataclasses import dataclass


@dataclass
class RewardConfig:
    """Reward function weights from project spec."""

    gold_scale: float = 0.01  # ~0.2 per CS, ~3.0 per kill
    health_advantage_scale: float = 5.0  # ±0.5 per trade
    death_penalty: float = -5.0  # ~500 gold equivalent


@dataclass
class RewardInfo:
    """Reward breakdown for a single frame."""

    total: float
    gold_reward: float
    health_reward: float
    death_reward: float

    # Raw values used
    gold_delta: int
    health_advantage_delta: float
    died: bool


class RewardExtractor:
    """Compute rewards from sequential game states."""

    def __init__(self, config: RewardConfig | None = None):
        self.config = config or RewardConfig()
        self.prev_state: dict | None = None

    def reset(self):
        """Reset state for new video/episode."""
        self.prev_state = None

    def compute_reward(self, state: dict) -> RewardInfo:
        """Compute reward for current state given previous state.

        Args:
            state: Dict with keys: gold, cs, player_health, enemy_health, game_time_seconds

        Returns:
            RewardInfo with reward breakdown
        """
        gold_reward = 0.0
        health_reward = 0.0
        death_reward = 0.0
        gold_delta = 0
        health_advantage_delta = 0.0
        died = False

        if self.prev_state is not None:
            # Gold reward
            prev_gold = self.prev_state.get("gold") or 0
            curr_gold = state.get("gold") or 0
            gold_delta = curr_gold - prev_gold

            # Only reward positive gold gains (ignore spending)
            if gold_delta > 0:
                gold_reward = gold_delta * self.config.gold_scale

            # Health advantage reward
            prev_player_hp = self.prev_state.get("player_health") or 0.5
            prev_enemy_hp = self.prev_state.get("enemy_health") or 0.5
            curr_player_hp = state.get("player_health") or 0.5
            curr_enemy_hp = state.get("enemy_health") or 0.5

            prev_advantage = prev_player_hp - prev_enemy_hp
            curr_advantage = curr_player_hp - curr_enemy_hp
            health_advantage_delta = curr_advantage - prev_advantage

            health_reward = health_advantage_delta * self.config.health_advantage_scale

            # Death detection (health drops to 0 or very low suddenly)
            if prev_player_hp > 0.1 and curr_player_hp < 0.05:
                died = True
                death_reward = self.config.death_penalty

        # Update previous state
        self.prev_state = state.copy()

        total = gold_reward + health_reward + death_reward

        return RewardInfo(
            total=total,
            gold_reward=gold_reward,
            health_reward=health_reward,
            death_reward=death_reward,
            gold_delta=gold_delta,
            health_advantage_delta=health_advantage_delta,
            died=died,
        )


def compute_rewards_for_sequence(
    states: list[dict],
    config: RewardConfig | None = None,
) -> list[RewardInfo]:
    """Compute rewards for a sequence of game states.

    Args:
        states: List of game state dicts from OCR
        config: Optional reward config

    Returns:
        List of RewardInfo, same length as states
    """
    extractor = RewardExtractor(config)
    return [extractor.compute_reward(s) for s in states]
