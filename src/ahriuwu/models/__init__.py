"""Model implementations for world model training."""

from .transformer_tokenizer import TransformerTokenizer, create_transformer_tokenizer
from .losses import TokenizerLoss, VGGPerceptualLoss, LPIPSLoss, MAELoss, psnr, LPIPS_AVAILABLE
from .diffusion import (
    DiffusionSchedule,
    TimestepEmbedding,
    x_prediction_loss,
    ramp_weight,
    ShortcutForcing,
)
from .dynamics import DynamicsTransformer, create_dynamics
from .heads import (
    RewardHead,
    PolicyHead,
    ValueHead,
    AgentTokenProvider,
    create_behavioral_prior,
    kl_to_prior,
    kl_to_prior_continuous,
    freeze_for_imagination,
    unfreeze_all,
)
from .returns import (
    symlog,
    symexp,
    twohot_encode,
    twohot_decode,
    twohot_loss,
    compute_lambda_returns,
    compute_advantages,
    pmpo_loss,
    RunningRMS,
    normalize_losses,
)

__all__ = [
    # Tokenizer (Transformer - DreamerV4)
    "TransformerTokenizer",
    "create_transformer_tokenizer",
    # Losses
    "TokenizerLoss",
    "VGGPerceptualLoss",
    "LPIPSLoss",
    "MAELoss",
    "psnr",
    "LPIPS_AVAILABLE",
    # Diffusion
    "DiffusionSchedule",
    "TimestepEmbedding",
    "x_prediction_loss",
    "ramp_weight",
    "ShortcutForcing",
    # Dynamics
    "DynamicsTransformer",
    "create_dynamics",
    # Heads (Phase 2+)
    "RewardHead",
    "PolicyHead",
    "ValueHead",
    "AgentTokenProvider",
    "create_behavioral_prior",
    "kl_to_prior",
    "kl_to_prior_continuous",
    "freeze_for_imagination",
    "unfreeze_all",
    # Returns and utilities
    "symlog",
    "symexp",
    "twohot_encode",
    "twohot_decode",
    "twohot_loss",
    "compute_lambda_returns",
    "compute_advantages",
    "pmpo_loss",
    "RunningRMS",
    "normalize_losses",
]
