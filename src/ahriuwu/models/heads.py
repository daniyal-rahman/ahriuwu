"""Prediction heads for agent training.

Phase 2 (Agent Finetuning): RewardHead, PolicyHead
Phase 3 (Imagination Training): ValueHead (initialized later)

Reference: DreamerV4 Section 3.3
"""

import copy

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

        # Zero-init output heads: initial predictions are zero/uniform,
        # which is a good starting point. Hidden layers keep default init.
        for head in self.heads:
            nn.init.zeros_(head.weight)
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

    Phase 2: Trained with behavioral cloning
    Phase 3: Trained with PMPO on imagined trajectories

    Predicts:
    - Discrete ability actions (cross-entropy with human actions)
    - Continuous movement (x, y) in [0, 1] via sigmoid + MSE

    Uses MTP to predict actions for multiple future timesteps.

    Reference: DreamerV4 Section 3.3
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        mtp_length: int = 8,
        movement_dim: int = 2,
    ):
        """Initialize policy head.

        Args:
            input_dim: Dimension of agent token features
            action_dim: Number of discrete actions (for ability predictions)
            hidden_dim: Hidden layer dimension
            mtp_length: Multi-token prediction length (paper uses 8)
            movement_dim: Continuous movement dimensions (default 2 for x, y)
        """
        super().__init__()
        self.action_dim = action_dim
        self.mtp_length = mtp_length
        self.movement_dim = movement_dim

        # Shared MLP backbone
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # MTP heads for discrete ability actions: predict action for t, t+1, ..., t+L-1
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, action_dim) for _ in range(mtp_length)
        ])

        # MTP heads for continuous movement (x, y) prediction
        self.movement_heads = nn.ModuleList([
            nn.Linear(hidden_dim, movement_dim) for _ in range(mtp_length)
        ])

        # Zero-init output heads: initial predictions are zero/uniform,
        # which is a good starting point. Hidden layers keep default init.
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        for head in self.movement_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, agent_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict action logits and movement coordinates.

        Args:
            agent_tokens: (B, T, D) agent token features

        Returns:
            tuple of:
                ability_logits: (B, T, L, action_dim) logits for discrete actions
                movement_pred: (B, T, L, 2) predicted (x, y) in [0, 1]
        """
        x = self.mlp(agent_tokens)  # (B, T, hidden_dim)

        # Discrete ability predictions
        ability_logits = torch.stack([head(x) for head in self.heads], dim=2)

        # Continuous movement predictions with sigmoid to bound in [0, 1]
        movement_pred = torch.stack(
            [torch.sigmoid(mhead(x)) for mhead in self.movement_heads], dim=2
        )

        return ability_logits, movement_pred

    def sample(self, agent_tokens: torch.Tensor, temperature: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample actions from policy.

        Args:
            agent_tokens: (B, T, D) agent token features
            temperature: Sampling temperature (1.0 = standard, <1 = greedy)

        Returns:
            tuple of:
                (B, T, L) sampled discrete action indices
                (B, T, L, 2) predicted movement coordinates
        """
        ability_logits, movement_pred = self.forward(agent_tokens)

        if temperature == 0:
            return ability_logits.argmax(dim=-1), movement_pred

        probs = F.softmax(ability_logits / temperature, dim=-1)
        B, T, L, A = probs.shape
        probs_flat = probs.view(-1, A)
        actions_flat = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)
        return actions_flat.view(B, T, L), movement_pred

    def log_prob(self, agent_tokens: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute log probability of discrete actions.

        Args:
            agent_tokens: (B, T, D) agent token features
            actions: (B, T, L) action indices

        Returns:
            (B, T, L) log probabilities
        """
        ability_logits, _ = self.forward(agent_tokens)
        log_probs = F.log_softmax(ability_logits, dim=-1)

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

        # Zero-init output layer: initial predictions are zero/uniform,
        # which is a good starting point. Hidden layers keep default init.
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


# ---------------------------------------------------------------------------
# Frozen behavioral prior (Fix 3)
# ---------------------------------------------------------------------------

def create_behavioral_prior(policy_head: PolicyHead) -> PolicyHead:
    """Create a frozen deep copy of a policy head to serve as behavioral prior.

    The prior is used for KL regularization during imagination training (Phase 3)
    to prevent the policy from diverging too far from the behavior-cloned policy.

    Args:
        policy_head: Trained policy head (typically after Phase 2)

    Returns:
        Frozen deep copy of the policy head (all requires_grad=False)
    """
    prior = copy.deepcopy(policy_head)
    prior.requires_grad_(False)
    return prior


def kl_to_prior(
    policy_logits: torch.Tensor,
    prior_logits: torch.Tensor,
) -> torch.Tensor:
    """Compute KL divergence from current policy to frozen prior (categorical).

    KL(policy || prior) = sum(policy * (log policy - log prior))

    Args:
        policy_logits: (..., num_actions) logits from current policy
        prior_logits: (..., num_actions) logits from frozen behavioral prior

    Returns:
        (...,) KL divergence per state (non-negative)
    """
    policy_log_probs = F.log_softmax(policy_logits, dim=-1)
    prior_log_probs = F.log_softmax(prior_logits, dim=-1)
    policy_probs = policy_log_probs.exp()

    kl = (policy_probs * (policy_log_probs - prior_log_probs)).sum(dim=-1)
    return kl


def kl_to_prior_continuous(
    policy_mean: torch.Tensor,
    policy_std: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_std: torch.Tensor,
) -> torch.Tensor:
    """Compute KL divergence for continuous (Gaussian) distributions.

    KL(N(mu1,s1) || N(mu2,s2)) = log(s2/s1) + (s1^2 + (mu1-mu2)^2)/(2*s2^2) - 0.5

    Summed over the last dimension (e.g., 2 for x,y movement).

    Args:
        policy_mean: (..., D) mean of current policy
        policy_std: (..., D) std of current policy
        prior_mean: (..., D) mean of frozen prior
        prior_std: (..., D) std of frozen prior

    Returns:
        (...,) KL divergence per state
    """
    var_ratio = (policy_std / prior_std).pow(2)
    diff_sq = ((policy_mean - prior_mean) / prior_std).pow(2)
    kl = 0.5 * (var_ratio + diff_sq - 1 - var_ratio.log())
    return kl.sum(dim=-1)


# ---------------------------------------------------------------------------
# Freeze/unfreeze utilities for imagination training (Fix 4)
# ---------------------------------------------------------------------------

def freeze_for_imagination(
    model: nn.Module,
    dynamics_attr: str = "dynamics",
    freeze_reward: bool = False,
) -> nn.Module:
    """Freeze the dynamics transformer for imagination training.

    During imagination (Phase 3), gradients flow through the frozen dynamics
    model to train the policy and value heads. The dynamics weights themselves
    are not updated.

    Args:
        model: Agent model containing dynamics, policy, value, reward submodules
        dynamics_attr: Attribute name for the dynamics transformer
        freeze_reward: If False (default), reward head stays unfrozen
            (paper trains reward head during imagination too)

    Returns:
        The model (modified in-place)
    """
    # Freeze dynamics transformer
    dynamics = getattr(model, dynamics_attr, None)
    if dynamics is not None:
        dynamics.requires_grad_(False)

    # Optionally freeze reward head
    if freeze_reward:
        reward_head = getattr(model, "reward_head", None)
        if reward_head is not None:
            reward_head.requires_grad_(False)

    return model


def unfreeze_all(model: nn.Module) -> nn.Module:
    """Unfreeze all parameters in the model.

    Call this to restore full training after imagination phase, or when
    switching between training phases.

    Args:
        model: Agent model

    Returns:
        The model (modified in-place)
    """
    model.requires_grad_(True)
    return model


# ---------------------------------------------------------------------------
# Agent token interleaving for dynamics sequence (Fix 5)
# ---------------------------------------------------------------------------

class AgentTokenProvider(nn.Module):
    """Produces agent tokens for insertion into the dynamics transformer sequence.

    During finetuning (Phase 3), agent tokens are concatenated into the dynamics
    spatial sequence: [latent_tokens, register_tokens, action_token, condition_token, agent_tokens].

    Agent tokens attend to everything in the sequence, but other tokens do NOT
    attend to agent tokens (to prevent causal confusion -- future predictions
    should not be influenced by agent-specific tokens).

    After the transformer processes the full sequence, agent token embeddings are
    extracted and passed through the policy/reward/value MLP heads.

    Args:
        embed_dim: Dimension matching the dynamics transformer
        num_agent_tokens: Number of agent tokens to insert (default 1)
        task_embed_dim: Dimension of task embedding input (0 = no task input)
    """

    def __init__(
        self,
        embed_dim: int,
        num_agent_tokens: int = 1,
        task_embed_dim: int = 0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_agent_tokens = num_agent_tokens

        # Learned agent token embeddings
        self.agent_tokens = nn.Parameter(
            torch.randn(1, num_agent_tokens, embed_dim) * 0.02
        )

        # Optional task embedding projection
        if task_embed_dim > 0:
            self.task_proj = nn.Linear(task_embed_dim, embed_dim)
        else:
            self.task_proj = None

    def get_agent_tokens(
        self,
        batch_size: int,
        task_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate agent tokens for concatenation into dynamics sequence.

        Args:
            batch_size: Batch size B
            task_embed: (B, task_embed_dim) optional task embedding

        Returns:
            (B, num_agent_tokens, embed_dim) agent tokens
        """
        tokens = self.agent_tokens.expand(batch_size, -1, -1).clone()

        if task_embed is not None and self.task_proj is not None:
            # Project task embedding and add to agent tokens
            task_feat = self.task_proj(task_embed)  # (B, embed_dim)
            tokens = tokens + task_feat.unsqueeze(1)

        return tokens

    @staticmethod
    def build_agent_attention_mask(
        num_non_agent: int,
        num_agent: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build attention mask for agent token interleaving.

        Agent tokens can attend to everything (non-agent + agent).
        Non-agent tokens can attend to non-agent tokens only.

        The full sequence is: [non_agent_tokens..., agent_tokens...]

        Args:
            num_non_agent: Number of non-agent tokens (latents + registers + action + condition)
            num_agent: Number of agent tokens
            device: Device for the mask tensor

        Returns:
            (N_total, N_total) boolean mask where True = can attend
        """
        N = num_non_agent + num_agent

        mask = torch.zeros(N, N, dtype=torch.bool, device=device)

        # Non-agent tokens see only non-agent tokens
        mask[:num_non_agent, :num_non_agent] = True

        # Agent tokens see everything (non-agent + agent self-attention)
        mask[num_non_agent:, :] = True

        return mask

    @staticmethod
    def extract_agent_embeddings(
        sequence: torch.Tensor,
        num_agent_tokens: int,
    ) -> torch.Tensor:
        """Extract agent token embeddings from the end of the processed sequence.

        Args:
            sequence: (B, N_total, D) full sequence after transformer processing
            num_agent_tokens: Number of agent tokens at the end

        Returns:
            (B, num_agent_tokens, D) agent token embeddings
        """
        return sequence[:, -num_agent_tokens:, :]
