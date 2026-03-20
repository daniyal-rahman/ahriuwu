"""Return computation and reward representation utilities.

Implements:
- symlog/symexp: Scale-invariant reward transformation
- twohot: Distributional representation for robust learning
- lambda returns: TD(λ) return computation for imagination training

Reference: DreamerV4 Section 3.3, DreamerV3 Appendix B
"""

import torch
import torch.nn.functional as F


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric logarithmic transformation.

    Compresses large positive and negative values while preserving small values.
    symlog(x) = sign(x) * log(1 + |x|)

    Args:
        x: Input tensor

    Returns:
        Transformed tensor with compressed scale
    """
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of symlog.

    symexp(x) = sign(x) * (exp(|x|) - 1)

    Args:
        x: Symlog-transformed tensor

    Returns:
        Original scale tensor
    """
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def twohot_encode(
    x: torch.Tensor,
    bucket_centers: torch.Tensor,
) -> torch.Tensor:
    """Encode scalar values as soft two-hot distributions.

    Represents a scalar as a weighted combination of two adjacent buckets.
    This provides a differentiable distributional representation.

    Args:
        x: (...,) scalar values (should be symlog-transformed)
        bucket_centers: (num_buckets,) center values for each bucket

    Returns:
        (..., num_buckets) soft two-hot encoding
    """
    num_buckets = bucket_centers.shape[0]
    low = bucket_centers[0]
    high = bucket_centers[-1]

    # Clamp to bucket range
    x_clamped = x.clamp(low, high)

    # Find bucket width
    bucket_width = (high - low) / (num_buckets - 1)

    # Find lower bucket index
    lower_idx = ((x_clamped - low) / bucket_width).floor().long()
    lower_idx = lower_idx.clamp(0, num_buckets - 2)
    upper_idx = lower_idx + 1

    # Compute interpolation weight
    lower_val = bucket_centers[lower_idx]
    upper_weight = (x_clamped - lower_val) / bucket_width
    upper_weight = upper_weight.clamp(0, 1)
    lower_weight = 1 - upper_weight

    # Create two-hot encoding
    shape = x.shape
    twohot = torch.zeros(*shape, num_buckets, device=x.device, dtype=x.dtype)
    twohot.scatter_(-1, lower_idx.unsqueeze(-1), lower_weight.unsqueeze(-1))
    twohot.scatter_(-1, upper_idx.unsqueeze(-1), upper_weight.unsqueeze(-1))

    return twohot


def twohot_decode(
    logits: torch.Tensor,
    bucket_centers: torch.Tensor,
) -> torch.Tensor:
    """Decode twohot logits to expected values.

    Args:
        logits: (..., num_buckets) logits
        bucket_centers: (num_buckets,) center values for each bucket

    Returns:
        (...,) expected values (in symlog scale)
    """
    probs = F.softmax(logits, dim=-1)
    return (probs * bucket_centers).sum(dim=-1)


def twohot_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    bucket_centers: torch.Tensor,
) -> torch.Tensor:
    """Compute cross-entropy loss for twohot targets.

    Args:
        logits: (..., num_buckets) predicted logits
        targets: (...,) target values (should be symlog-transformed)
        bucket_centers: (num_buckets,) center values for each bucket

    Returns:
        Scalar loss
    """
    target_twohot = twohot_encode(targets, bucket_centers)
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(target_twohot * log_probs).sum(dim=-1)
    return loss.mean()


def compute_lambda_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    continues: torch.Tensor,
    gamma: float = 0.997,
    lambda_: float = 0.95,
) -> torch.Tensor:
    """Compute λ-returns for value learning.

    R^λ_t = r_t + γ * c_t * [(1 - λ) * v_{t+1} + λ * R^λ_{t+1}]

    where c_t indicates non-terminal states (from data, not learned).

    Reference: DreamerV4 Equation 10

    Args:
        rewards: (B, T) reward values (original scale, not symlog)
        values: (B, T) value predictions (original scale)
        continues: (B, T) continuation flags (1.0 = continues, 0.0 = terminal)
        gamma: Discount factor (paper uses 0.997)
        lambda_: TD(λ) parameter (paper uses 0.95)

    Returns:
        (B, T) λ-returns (original scale)
    """
    B, T = rewards.shape

    # Initialize returns tensor
    returns = torch.zeros_like(rewards)

    # Bootstrap: the return at the horizon is the value estimate at the last step.
    # The loop iterates from T-2 down to 0 because the return at T-1 IS the
    # bootstrap value (there is no reward beyond the horizon to incorporate).
    next_return = values[:, -1]
    returns[:, -1] = next_return

    # Compute returns backwards from T-2 to 0
    for t in range(T - 2, -1, -1):
        next_value = values[:, t + 1]
        next_return = rewards[:, t] + gamma * continues[:, t] * (
            (1 - lambda_) * next_value + lambda_ * next_return
        )
        returns[:, t] = next_return

    return returns


def compute_advantages(
    returns: torch.Tensor,
    values: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """Compute advantages for policy gradient.

    A_t = R^λ_t - V(s_t)

    Args:
        returns: (B, T) λ-returns
        values: (B, T) value predictions
        normalize: Whether to normalize advantages

    Returns:
        (B, T) advantages
    """
    advantages = returns - values

    if normalize and advantages.numel() > 1:
        mean = advantages.mean()
        std = advantages.std() + 1e-8
        advantages = (advantages - mean) / std

    return advantages


def compute_mtp_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mtp_length: int,
    criterion,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute MTP (Multi-Token Prediction) loss matching DreamerV4 Eq 9.

    For each MTP offset n=0..mtp_length-1, aligns prediction at position t
    with target at position t+n. n=0 predicts the current timestep.
    Paper uses L=8 with sum from n=0 to L inclusive, so mtp_length=9.

    Args:
        logits: (B, T, L, ...) predicted logits from head
        targets: (B, T, ...) target values (actions or symlog rewards)
        mtp_length: L, number of MTP offsets
        criterion: Loss function(pred, target) -> scalar loss per element
        mask: Optional (B,) bool mask. If provided, loss only computed
              where mask is True.

    Returns:
        Scalar loss averaged over all valid positions and offsets
    """
    B, T = targets.shape[:2]
    total_loss = torch.tensor(0.0, device=logits.device)
    num_terms = 0

    for n in range(mtp_length):
        valid_T = T - n  # positions 0..T-n-1 can predict t+n
        if valid_T <= 0:
            continue

        pred = logits[:, :valid_T, n]     # (B, valid_T, ...)
        target = targets[:, n:n + valid_T]  # (B, valid_T, ...)

        if mask is not None:
            # Expand mask to match: (B,) -> (B, 1, ...) broadcast
            pred = pred[mask]
            target = target[mask]

        if pred.numel() > 0:
            total_loss = total_loss + criterion(pred, target)
            num_terms += 1

    return total_loss / max(num_terms, 1)


def compute_pmpo_loss(
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    policy_logits: torch.Tensor,
    prior_logits: torch.Tensor,
    alpha: float = 0.5,
    beta: float = 0.3,
) -> torch.Tensor:
    """Compute PMPO policy loss (DreamerV4 Eq 11).

    PMPO uses only the SIGN of advantages, splitting states into positive
    (A >= 0) and negative (A < 0) sets. This balances positive and negative
    feedback without needing advantage normalization.

    L = (1-a)/|D-| sum_{D-} ln pi - a/|D+| sum_{D+} ln pi + b * KL[pi || pi_prior]

    The KL term is computed as the full categorical KL divergence over the
    action distribution, not a single-sample log-ratio approximation.

    Args:
        log_probs: (N,) log pi_theta(a|s) for sampled actions
        advantages: (N,) raw advantages A_t = R^lambda_t - v_t (NOT normalized)
        policy_logits: (N, A) full logits from current policy (for KL computation)
        prior_logits: (N, A) full logits from frozen behavioral prior
        alpha: Weight for positive advantages (0.5 = equal balance)
        beta: KL regularization weight (0.3 = weak prior)

    Returns:
        Scalar PMPO loss
    """
    pos_mask = advantages >= 0
    neg_mask = ~pos_mask

    # Maximize log-prob for states with positive advantage
    loss_pos = -log_probs[pos_mask].mean() if pos_mask.any() else torch.tensor(0.0, device=log_probs.device)

    # Minimize log-prob for states with negative advantage
    loss_neg = log_probs[neg_mask].mean() if neg_mask.any() else torch.tensor(0.0, device=log_probs.device)

    # Full categorical KL divergence: KL[pi_theta || pi_prior]
    # KL(p||q) = sum_a p(a) * (log p(a) - log q(a))
    policy_log_probs = F.log_softmax(policy_logits, dim=-1)
    prior_log_probs = F.log_softmax(prior_logits, dim=-1)
    policy_probs = policy_log_probs.exp()
    kl = (policy_probs * (policy_log_probs - prior_log_probs)).sum(dim=-1).mean()

    return (1 - alpha) * loss_neg + alpha * loss_pos + beta * kl


class RunningRMS:
    """Running root mean square for loss normalization.

    DreamerV4 normalizes losses by their running RMS to balance them.
    """

    # Minimum RMS value to prevent division-by-near-zero on early batches
    # when the EMA has not yet accumulated meaningful statistics.
    MIN_RMS = 1e-4

    def __init__(self, decay: float = 0.99):
        """Initialize running RMS.

        Args:
            decay: EMA decay factor
        """
        self.decay = decay
        self.rms = None

    def update(self, value: torch.Tensor) -> torch.Tensor:
        """Update RMS and return normalized value.

        Args:
            value: Scalar loss value

        Returns:
            Normalized loss (value / rms)
        """
        value_sq = value.detach() ** 2

        if self.rms is None:
            self.rms = value_sq
        else:
            # Fix device mismatch after checkpoint resume (rms loaded as CPU tensor)
            if self.rms.device != value_sq.device:
                self.rms = self.rms.to(value_sq.device)
            self.rms = self.decay * self.rms + (1 - self.decay) * value_sq

        # Clamp RMS to avoid instability when the first few loss values are
        # very small (e.g. zero-init heads producing near-zero losses).
        self.rms = torch.clamp(self.rms, min=self.MIN_RMS ** 2)

        return value / (torch.sqrt(self.rms) + 1e-8)

    def state_dict(self) -> dict:
        """Return serializable state."""
        return {"rms": self.rms.item() if self.rms is not None else None, "decay": self.decay}

    def load_state_dict(self, state: dict):
        """Restore from serialized state."""
        self.decay = state.get("decay", self.decay)
        rms_val = state.get("rms")
        self.rms = torch.tensor(rms_val) if rms_val is not None else None


def normalize_losses(losses: dict[str, torch.Tensor], rms_dict: dict[str, RunningRMS]) -> torch.Tensor:
    """Normalize and sum multiple losses.

    Args:
        losses: Dict of loss name -> loss value
        rms_dict: Dict of loss name -> RunningRMS tracker

    Returns:
        Normalized total loss
    """
    total = 0.0
    for name, loss in losses.items():
        if name not in rms_dict:
            rms_dict[name] = RunningRMS()
        total = total + rms_dict[name].update(loss)
    return total
