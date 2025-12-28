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
    device = rewards.device
    dtype = rewards.dtype

    # Initialize returns tensor
    returns = torch.zeros_like(rewards)

    # Bootstrap from final value
    next_return = values[:, -1]

    # Compute returns backwards in time
    for t in range(T - 1, -1, -1):
        # TD target: r_t + γ * c_t * v_{t+1}
        if t < T - 1:
            next_value = values[:, t + 1]
        else:
            next_value = values[:, t]  # Bootstrap

        td_target = rewards[:, t] + gamma * continues[:, t] * next_value

        # λ-return: blend TD target with full return
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


class RunningRMS:
    """Running root mean square for loss normalization.

    DreamerV4 normalizes losses by their running RMS to balance them.
    """

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
            self.rms = self.decay * self.rms + (1 - self.decay) * value_sq

        return value / (torch.sqrt(self.rms) + 1e-8)


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
