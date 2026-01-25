"""Prediction heads for agent training.

Phase 2 (Agent Finetuning): RewardHead, PolicyHead
Phase 3 (Imagination Training): ValueHead (initialized later)

Reference: DreamerV4 Section 3.3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RewardHead(nn.Module):
    """Reward prediction head with symexp twohot output.

    Predicts rewards for current and future timesteps using MTP (Multi-Token Prediction).
    Uses symexp twohot representation for robust learning across reward scales.

    Reference: DreamerV4 Section 3.3 "Behavior cloning and reward model"
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_buckets: int = 255,
        mtp_length: int = 8,
        bucket_low: float = -20.0,
        bucket_high: float = 20.0,
    ):
        """Initialize reward head.

        Args:
            input_dim: Dimension of agent token features
            hidden_dim: Hidden layer dimension
            num_buckets: Number of twohot buckets (paper uses 255)
            mtp_length: Multi-token prediction length (paper uses 8)
            bucket_low: Lower bound for symlog buckets
            bucket_high: Upper bound for symlog buckets
        """
        super().__init__()
        self.num_buckets = num_buckets
        self.mtp_length = mtp_length
        self.bucket_low = bucket_low
        self.bucket_high = bucket_high

        # Shared MLP backbone
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # MTP heads: predict reward for t, t+1, ..., t+L-1
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, num_buckets) for _ in range(mtp_length)
        ])

        # Initialize output weights to small values for stable start
        # Note: True zero init breaks gradient flow! Use small values instead.
        for head in self.heads:
            nn.init.normal_(head.weight, std=0.01)
            nn.init.zeros_(head.bias)

        # Register bucket centers as buffer
        self.register_buffer(
            "bucket_centers",
            torch.linspace(bucket_low, bucket_high, num_buckets)
        )

    def forward(self, agent_tokens: torch.Tensor) -> torch.Tensor:
        """Predict reward distributions.

        Args:
            agent_tokens: (B, T, D) agent token features

        Returns:
            (B, T, L, num_buckets) logits for each timestep and MTP offset
        """
        x = self.mlp(agent_tokens)  # (B, T, hidden_dim)

        # Predict for each MTP offset
        logits = torch.stack([head(x) for head in self.heads], dim=2)
        return logits  # (B, T, L, num_buckets)

    def predict(self, agent_tokens: torch.Tensor) -> torch.Tensor:
        """Get expected reward values.

        Args:
            agent_tokens: (B, T, D) agent token features

        Returns:
            (B, T, L) predicted reward values (in original scale via symexp)
        """
        from .returns import twohot_decode, symexp

        logits = self.forward(agent_tokens)
        symlog_values = twohot_decode(logits, self.bucket_centers)
        return symexp(symlog_values)


class PolicyHead(nn.Module):
    """Policy head for action prediction.

    Phase 2: Trained with behavioral cloning (cross-entropy with human actions)
    Phase 3: Trained with PMPO on imagined trajectories

    Uses MTP to predict actions for multiple future timesteps.

    Reference: DreamerV4 Section 3.3
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        mtp_length: int = 8,
    ):
        """Initialize policy head.

        Args:
            input_dim: Dimension of agent token features
            action_dim: Number of discrete actions
            hidden_dim: Hidden layer dimension
            mtp_length: Multi-token prediction length (paper uses 8)
        """
        super().__init__()
        self.action_dim = action_dim
        self.mtp_length = mtp_length

        # Shared MLP backbone
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # MTP heads: predict action for t, t+1, ..., t+L-1
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, action_dim) for _ in range(mtp_length)
        ])

    def forward(self, agent_tokens: torch.Tensor) -> torch.Tensor:
        """Predict action logits.

        Args:
            agent_tokens: (B, T, D) agent token features

        Returns:
            (B, T, L, action_dim) logits for each timestep and MTP offset
        """
        x = self.mlp(agent_tokens)  # (B, T, hidden_dim)

        # Predict for each MTP offset
        logits = torch.stack([head(x) for head in self.heads], dim=2)
        return logits  # (B, T, L, action_dim)

    def sample(self, agent_tokens: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Sample actions from policy.

        Args:
            agent_tokens: (B, T, D) agent token features
            temperature: Sampling temperature (1.0 = standard, <1 = greedy)

        Returns:
            (B, T, L) sampled action indices
        """
        logits = self.forward(agent_tokens)  # (B, T, L, action_dim)

        if temperature == 0:
            return logits.argmax(dim=-1)

        probs = F.softmax(logits / temperature, dim=-1)
        # Sample from categorical
        B, T, L, A = probs.shape
        probs_flat = probs.view(-1, A)
        actions_flat = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)
        return actions_flat.view(B, T, L)

    def log_prob(self, agent_tokens: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute log probability of actions.

        Args:
            agent_tokens: (B, T, D) agent token features
            actions: (B, T, L) action indices

        Returns:
            (B, T, L) log probabilities
        """
        logits = self.forward(agent_tokens)  # (B, T, L, action_dim)
        log_probs = F.log_softmax(logits, dim=-1)

        # Gather log probs for taken actions
        actions_expanded = actions.unsqueeze(-1)  # (B, T, L, 1)
        return log_probs.gather(-1, actions_expanded).squeeze(-1)


class ValueHead(nn.Module):
    """Value head for imagination training (Phase 3 only).

    Initialized at start of Phase 3, not during Phase 2.
    Predicts discounted sum of future rewards using symexp twohot.

    Reference: DreamerV4 Section 3.3 "Reinforcement learning"
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_buckets: int = 255,
        bucket_low: float = -20.0,
        bucket_high: float = 20.0,
    ):
        """Initialize value head.

        Args:
            input_dim: Dimension of agent token features
            hidden_dim: Hidden layer dimension
            num_buckets: Number of twohot buckets
            bucket_low: Lower bound for symlog buckets
            bucket_high: Upper bound for symlog buckets
        """
        super().__init__()
        self.num_buckets = num_buckets
        self.bucket_low = bucket_low
        self.bucket_high = bucket_high

        # MLP for value prediction
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_buckets),
        )

        # Initialize output weights to zero (paper recommendation)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        # Register bucket centers as buffer
        self.register_buffer(
            "bucket_centers",
            torch.linspace(bucket_low, bucket_high, num_buckets)
        )

    def forward(self, agent_tokens: torch.Tensor) -> torch.Tensor:
        """Predict value distribution logits.

        Args:
            agent_tokens: (B, T, D) agent token features

        Returns:
            (B, T, num_buckets) logits
        """
        return self.mlp(agent_tokens)

    def predict(self, agent_tokens: torch.Tensor) -> torch.Tensor:
        """Get expected value estimates.

        Args:
            agent_tokens: (B, T, D) agent token features

        Returns:
            (B, T) predicted values (in original scale via symexp)
        """
        from .returns import twohot_decode, symexp

        logits = self.forward(agent_tokens)
        symlog_values = twohot_decode(logits, self.bucket_centers)
        return symexp(symlog_values)
